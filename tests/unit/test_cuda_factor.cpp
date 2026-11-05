// tests/unit/test_cuda_factor.cpp
// CudaSolverBackend::factor(): cuDSS analysis reuse, multiple right-hand
// sides, require_spd, recovery after a failed factorization, the dense_block
// ordering, the LU fallback, fp32 refinement with its switch to fp64, MAGSTAT on cuDSS,
// and hybrid mode's device limit.

#include <gtest/gtest.h>
#include "solver/solver_backend.hpp"
#include "hilbert_system.hpp"
#include "dense_block_system.hpp"
#ifdef HAVE_CUDA
#include "core/exceptions.hpp"
#include "solver/cuda_solver_backend.hpp"
#include "magnetic/solver.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <string>
#endif

using namespace vibestran;

#ifndef HAVE_CUDA
TEST(CudaFactor, CudaNotCompiled) {
    GTEST_SKIP() << "CUDA backend not compiled — skipping CUDA factor tests";
}
#else

namespace {

using CsrData = SparseMatrixBuilder::CsrData;

CsrData to_csr(const Eigen::MatrixXd &A, bool full = false) {
    CsrData csr;
    csr.n = static_cast<int>(A.rows());
    csr.symmetry = full ? SymmetryStorage::Full : SymmetryStorage::Lower;
    csr.row_ptr.push_back(0);
    for (int i = 0; i < csr.n; ++i) {
        for (int j = 0; j <= (full ? csr.n - 1 : i); ++j) {
            if (A(i, j) != 0.0) {
                csr.col_ind.push_back(j);
                csr.values.push_back(A(i, j));
            }
        }
        csr.row_ptr.push_back(static_cast<int>(csr.col_ind.size()));
    }
    csr.nnz = static_cast<int>(csr.values.size());
    return csr;
}

/// Tridiagonal diag 2 + shift, off-diagonals -1, plus a coupling to i-3 so
/// cuDSS reorders a non-trivial pattern. SPD for shift > 0.
Eigen::MatrixXd banded(int n, double shift, double coupling = 0.0) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = 2.0 + shift + 2 * std::abs(coupling);
        if (i > 0)
            A(i, i - 1) = A(i - 1, i) = -1.0;
        if (i > 2 && coupling != 0.0)
            A(i, i - 3) = A(i - 3, i) = coupling;
    }
    return A;
}

Eigen::MatrixXd solution(int n, int nrhs) {
    Eigen::MatrixXd X(n, nrhs);
    for (int k = 0; k < nrhs; ++k)
        for (int i = 0; i < n; ++i)
            X(i, k) = std::sin(0.37 * (i + 1) * (k + 1)) + 0.2 * k;
    return X;
}

double relative_error(const std::vector<double> &x, const Eigen::MatrixXd &expected) {
    if (x.size() != static_cast<std::size_t>(expected.size()))
        return INFINITY;
    const Eigen::Map<const Eigen::MatrixXd> X(x.data(), expected.rows(), expected.cols());
    return (X - expected).norm() / expected.norm();
}

std::span<const double> columns(const Eigen::MatrixXd &B) {
    return {B.data(), static_cast<std::size_t>(B.size())};
}

std::unique_ptr<LinearFactor> take(Result<std::unique_ptr<LinearFactor>> made) {
    if (auto *e = std::get_if<Error>(&made)) {
        ADD_FAILURE() << e->message;
        return nullptr;
    }
    return std::get<std::unique_ptr<LinearFactor>>(std::move(made));
}

std::vector<double> solved(Result<std::vector<double>> x) {
    if (auto *e = std::get_if<Error>(&x)) {
        ADD_FAILURE() << e->message;
        return {};
    }
    return std::get<std::vector<double>>(std::move(x));
}

class CudaFactorTest : public ::testing::Test {
protected:
    // cppcheck-suppress unusedFunction -- called by GTest framework
    void SetUp() override {
        backend_ = CudaSolverBackend::try_create();
        if (!backend_.has_value())
            GTEST_SKIP() << "CUDA not available on this system";
    }
    std::optional<CudaSolverBackend> backend_;
};

} // namespace

TEST_F(CudaFactorTest, SolvesManyRightHandSidesWithOneAnalysis) {
    const Eigen::MatrixXd A = banded(300, 0.01, 0.3);
    const Eigen::MatrixXd X = solution(300, 4);
    const Eigen::MatrixXd B = A * X;
    auto factor = take(backend_->factor(to_csr(A), {.require_spd = true}));
    ASSERT_TRUE(factor);
    EXPECT_EQ(factor->method(), "cudss-cholesky");
    EXPECT_LT(relative_error(solved(factor->solve(columns(B), 4)), X), 1e-11);
    // A narrower block reuses the same factor.
    EXPECT_LT(relative_error(solved(factor->solve(columns(B.col(2)), 1)), X.col(2)), 1e-11);
    EXPECT_EQ(factor->stats().symbolic_analyses, 1);
    EXPECT_EQ(factor->stats().numeric_factorizations, 1);
    EXPECT_EQ(factor->stats().solves, 5);
}

TEST_F(CudaFactorTest, RefactorReusesTheAnalysisAndUsesTheNewValues) {
    auto factor = take(backend_->factor(to_csr(banded(120, 0.0, 0.2)), {.require_spd = true}));
    ASSERT_TRUE(factor);
    // Same pattern, very different values: a stale factorization misses by O(1).
    const Eigen::MatrixXd A2 = 5.0 * banded(120, 3.0, 0.2);
    const Eigen::MatrixXd X = solution(120, 2);
    ASSERT_FALSE(factor->refactor(to_csr(A2)));
    EXPECT_LT(relative_error(solved(factor->solve(columns(A2 * X), 2)), X), 1e-12);
    EXPECT_EQ(factor->stats().symbolic_analyses, 1);
    EXPECT_EQ(factor->stats().numeric_factorizations, 2);

    // Full storage of the same matrix is a different input pattern.
    ASSERT_FALSE(factor->refactor(to_csr(A2, true)));
    EXPECT_LT(relative_error(solved(factor->solve(columns(A2 * X), 2)), X), 1e-12);
    EXPECT_EQ(factor->stats().symbolic_analyses, 2);

    const Eigen::MatrixXd A3 = banded(120, 1.0, 0.0); // couplings removed
    ASSERT_FALSE(factor->refactor(to_csr(A3)));
    EXPECT_LT(relative_error(solved(factor->solve(columns(A3 * X), 2)), X), 1e-12);
    EXPECT_EQ(factor->stats().symbolic_analyses, 3);
}

TEST_F(CudaFactorTest, DenseBlockOrdersWithMinimumDegree) {
    const Eigen::MatrixXd A = test::gap_like(50, 600);
    const Eigen::MatrixXd X = solution(A.rows(), 2);
    const Eigen::MatrixXd B = A * X;
    auto plain = take(backend_->factor(to_csr(A), {.require_spd = true}));
    auto dense = take(backend_->factor(to_csr(A), {.require_spd = true, .dense_block = true}));
    ASSERT_TRUE(plain && dense);
    EXPECT_EQ(plain->ordering(), "nested dissection");
    EXPECT_EQ(dense->ordering(), "amd");
    // cond(A) is about 87.
    EXPECT_LT(relative_error(solved(plain->solve(columns(B), 2)), X), 1e-11);
    EXPECT_LT(relative_error(solved(dense->solve(columns(B), 2)), X), 1e-11);
    // New values reuse the AMD analysis.
    const Eigen::MatrixXd A2 = 2.0 * A + Eigen::MatrixXd::Identity(A.rows(), A.cols());
    const Eigen::MatrixXd B2 = A2 * X;
    ASSERT_FALSE(dense->refactor(to_csr(A2)));
    EXPECT_LT(relative_error(solved(dense->solve(columns(B2), 2)), X), 1e-11);
    EXPECT_EQ(dense->stats().symbolic_analyses, 1);
}

TEST_F(CudaFactorTest, MinimumDegreeKeepsNestedDissectionOnTheGpu) {
    // minimum_degree is for the CPU backends; cuDSS's GPU factorization
    // needs nested dissection's wide elimination tree.
    const Eigen::MatrixXd A = test::gap_like(50, 0);
    const Eigen::MatrixXd X = solution(A.rows(), 2);
    auto amd = take(backend_->factor(to_csr(A), {.require_spd = true, .minimum_degree = true}));
    ASSERT_TRUE(amd);
    EXPECT_EQ(amd->ordering(), "nested dissection");
    EXPECT_LT(relative_error(solved(amd->solve(columns(A * X), 2)), X), 1e-11);
    // A new pattern is analyzed again.
    Eigen::MatrixXd A2 = A;
    A2(0, A.rows() - 1) = A2(A.rows() - 1, 0) = -0.5;
    ASSERT_FALSE(amd->refactor(to_csr(A2)));
    EXPECT_EQ(amd->stats().symbolic_analyses, 2);
    EXPECT_LT(relative_error(solved(amd->solve(columns(A2 * X), 2)), X), 1e-11);
}

TEST_F(CudaFactorTest, RequireSpdFailureIsReportedAndRecoverable) {
    // cuDSS reports a failed Cholesky only through CUDSS_DATA_INFO, which then
    // stays set: without a reset, the next factorization of SPD values does
    // nothing and its solves are wrong.
    const Eigen::MatrixXd spd = banded(80, 0.0, 0.25);
    const Eigen::MatrixXd indefinite = banded(80, -0.9, 0.25);
    ASSERT_LT(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(indefinite).eigenvalues()(0), 0.0);
    const Eigen::MatrixXd X = solution(80, 1);

    auto refused = backend_->factor(to_csr(indefinite), {.require_spd = true});
    ASSERT_TRUE(std::holds_alternative<Error>(refused));
    EXPECT_NE(std::get<Error>(refused).message.find("positive definite"), std::string::npos)
        << std::get<Error>(refused).message;

    auto factor = take(backend_->factor(to_csr(spd), {.require_spd = true}));
    ASSERT_TRUE(factor);
    EXPECT_TRUE(factor->refactor(to_csr(indefinite)));
    EXPECT_TRUE(std::holds_alternative<Error>(factor->solve(columns(spd * X), 1)));
    for (const double shift : {0.0, 2.0}) {
        const Eigen::MatrixXd A = banded(80, shift, 0.25);
        ASSERT_FALSE(factor->refactor(to_csr(A)));
        EXPECT_LT(relative_error(solved(factor->solve(columns(A * X), 1)), X), 1e-11)
            << "shift " << shift;
    }
    EXPECT_EQ(factor->stats().symbolic_analyses, 1);
}

TEST_F(CudaFactorTest, WithoutRequireSpdIndefiniteFallsBackToLu) {
    const Eigen::MatrixXd indefinite = banded(60, -0.9, 0.25);
    const Eigen::MatrixXd X = solution(60, 2);
    auto factor = take(backend_->factor(to_csr(indefinite)));
    ASSERT_TRUE(factor);
    EXPECT_EQ(factor->method(), "cudss-lu");
    EXPECT_LT(relative_error(solved(factor->solve(columns(indefinite * X), 2)), X), 1e-10);
    // Each fallback is reported for the F06.
    const auto first = backend_->notices().take();
    ASSERT_EQ(first.size(), 1u);
    EXPECT_NE(first[0].find("not numerically positive definite"), std::string::npos) << first[0];
    EXPECT_NE(first[0].find("solved with LU"), std::string::npos) << first[0];
    // SPD values go back to Cholesky; the LU analysis stays for later.
    const Eigen::MatrixXd spd = banded(60, 0.5, 0.25);
    ASSERT_FALSE(factor->refactor(to_csr(spd)));
    EXPECT_EQ(factor->method(), "cudss-cholesky");
    EXPECT_LT(relative_error(solved(factor->solve(columns(spd * X), 2)), X), 1e-12);
    EXPECT_TRUE(backend_->notices().take().empty());
    ASSERT_FALSE(factor->refactor(to_csr(indefinite)));
    EXPECT_EQ(factor->method(), "cudss-lu");
    EXPECT_EQ(factor->stats().symbolic_analyses, 2); // one Cholesky, one LU
    EXPECT_EQ(backend_->notices().take().size(), 1u);
}

TEST(CudaFactorFp32, SolvesAreRefinedToDoubleAccuracy) {
    auto backend = CudaSolverBackend::try_create(CudaSolverPrecision::Float32);
    if (!backend.has_value())
        GTEST_SKIP() << "CUDA not available on this system";
    // cond(H_5) ~ 4.8e5: an fp32 factor alone is accurate to ~1e-2.
    const auto [K, F] = test::scaled_hilbert(5);
    // Two columns: F (solution 1) and -2F (solution -2).
    std::vector<double> B(F);
    std::transform(F.begin(), F.end(), std::back_inserter(B),
                   [](double f) { return -2.0 * f; });
    auto factor = take(backend->factor(K, {.require_spd = true}));
    ASSERT_TRUE(factor);
    const auto X = solved(factor->solve(B, 2));
    ASSERT_EQ(X.size(), 10u);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(X[i], 1.0, 1e-11);
        EXPECT_NEAR(X[5 + i], -2.0, 2e-11);
    }
    EXPECT_EQ(factor->method(), "cudss-cholesky/fp32");
    EXPECT_GT(factor->stats().solves, 0);
    EXPECT_TRUE(backend->notices().take().empty()) << "fp32 sufficed: nothing to report";
}

TEST(CudaFactorFp32, UnrefinableFp32SolveSwitchesToFp64) {
    auto backend = CudaSolverBackend::try_create(CudaSolverPrecision::Float32);
    if (!backend.has_value())
        GTEST_SKIP() << "CUDA not available on this system";
    // Free-free tridiagonal plus delta I: eigenvalues from delta to ~4, so
    // cond ~ 4e7. Its fp32 Cholesky succeeds, but the fp32 factor cannot
    // refine the solution, so the solve refactors in fp64.
    const int n = 50;
    const double delta = 1e-7;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = (i == 0 || i == n - 1 ? 1.0 : 2.0) + delta;
        if (i > 0)
            A(i, i - 1) = A(i - 1, i) = -1.0;
    }
    const Eigen::MatrixXd X = solution(n, 2);
    auto factor = take(backend->factor(to_csr(A), {.require_spd = true}));
    ASSERT_TRUE(factor);
    EXPECT_EQ(factor->method(), "cudss-cholesky/fp32");
    // cond * eps ~ 1e-8 for an unrefined fp64 Cholesky.
    EXPECT_LT(relative_error(solved(factor->solve(columns(A * X), 2)), X), 1e-7);
    EXPECT_EQ(factor->method(), "cudss-cholesky");
    EXPECT_EQ(factor->stats().symbolic_analyses, 2); // fp32, then fp64
    const auto switched = backend->notices().take();
    ASSERT_EQ(switched.size(), 1u);
    EXPECT_NE(switched[0].find("could not be refined"), std::string::npos) << switched[0];
    EXPECT_NE(switched[0].find("factored again in fp64"), std::string::npos) << switched[0];
    ASSERT_FALSE(factor->refactor(to_csr(A)));       // stays in fp64
    EXPECT_EQ(factor->method(), "cudss-cholesky");
    EXPECT_EQ(factor->stats().symbolic_analyses, 2);
    EXPECT_TRUE(backend->notices().take().empty()) << "the switch is reported once";
}

TEST(CudaFactorFp32, LegacySolveRetriedInFp64ReportsTheFp64Solve) {
    // solve() (SOL 101) with the matrix of UnrefinableFp32SolveSwitchesToFp64:
    // the fp32 solve cannot be refined and is repeated in fp64. The accuracy
    // report must describe the fp64 solve -- none when it is not refined,
    // a converged one when it is -- never the failed fp32 attempt.
    const int n = 50;
    const double delta = 1e-7;
    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i) {
        builder.add(i, i, (i == 0 || i == n - 1 ? 1.0 : 2.0) + delta);
        if (i > 0) {
            builder.add(i, i - 1, -1.0);
            builder.add(i - 1, i, -1.0);
        }
    }
    const auto K = builder.build_csr();
    const std::vector<double> F(n, 1e-7);
    for (const bool refine : {false, true}) {
        SCOPED_TRACE(refine);
        auto backend = CudaSolverBackend::try_create(CudaSolverPrecision::Float32);
        if (!backend.has_value())
            GTEST_SKIP() << "CUDA not available on this system";
        backend->set_refinement(refine);
        const auto u = backend->solve(K, F);
        // K 1 = delta 1, so u = 1 everywhere; cond ~ 4e7 bounds the error.
        for (const double ui : u)
            EXPECT_NEAR(ui, 1.0, 1e-6);
        const auto notices = backend->notices().take();
        ASSERT_FALSE(notices.empty());
        EXPECT_NE(notices[0].find("solved again in fp64"), std::string::npos) << notices[0];
        const auto report = backend->last_refinement();
        if (refine) {
            ASSERT_TRUE(report.has_value());
            EXPECT_TRUE(report->converged);
        } else {
            EXPECT_FALSE(report.has_value())
                << "stale fp32 report: estimated error " << report->estimated_error;
        }
    }
}

TEST(CudaFactorFp32, FailedFp32CholeskyIsRetriedInFp64) {
    auto backend = CudaSolverBackend::try_create(CudaSolverPrecision::Float32);
    if (!backend.has_value())
        GTEST_SKIP() << "CUDA not available on this system";
    // cond(H_8) ~ 1.5e10: SPD in double but not numerically SPD in float, so
    // the fp32 Cholesky fails. Under require_spd that is not a verdict on the
    // matrix: the factor retries in fp64, accurate to about cond * eps.
    const auto [K, F] = test::scaled_hilbert(8);
    auto factor = take(backend->factor(K, {.require_spd = true}));
    ASSERT_TRUE(factor);
    EXPECT_EQ(factor->method(), "cudss-cholesky");
    EXPECT_EQ(factor->stats().symbolic_analyses, 2);
    const auto retried = backend->notices().take();
    ASSERT_EQ(retried.size(), 1u);
    EXPECT_NE(retried[0].find("factored again in fp64"), std::string::npos) << retried[0];
    const auto x = solved(factor->solve(F, 1));
    ASSERT_EQ(x.size(), 8u);
    EXPECT_LT(test::max_error_from_one(x), 1e-4);
    // A genuinely indefinite matrix is still refused after the fp64 retry.
    const Eigen::MatrixXd indefinite = banded(30, -0.9, 0.25);
    auto refused = backend->factor(to_csr(indefinite), {.require_spd = true});
    ASSERT_TRUE(std::holds_alternative<Error>(refused));
    EXPECT_NE(std::get<Error>(refused).message.find("positive definite"), std::string::npos);
}

namespace {
// Planar MAGSTAT deck on [-1,1]^2 with n x n quads: a saturating steel core
// in the middle and two opposite winding slots in series.
std::string planar_winding_deck(int n, double amps) {
    std::ostringstream d;
    d << "SOL MAGSTAT\nCEND\nLOAD=1\nBEGIN BULK\nMAG2D,0,0.02\n"
      << "MATMAG,1,1\nPMAG2D,1,1\nMATMAG,2,,1\nPMAG2D,2,2\n"
      << "TABLEBH,1\n,0,0,.1,100,.3,2000,1,100000,ENDT\n";
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i)
            d << "GRID," << j * (n + 1) + i + 1 << ",," << -1.0 + 2.0 * i / n << ","
              << -1.0 + 2.0 * j / n << ",0\n";
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const int a = j * (n + 1) + i + 1;
            const bool core = std::abs(2 * i + 1 - n) <= 2 && std::abs(2 * j + 1 - n) <= 2;
            d << "CQUAD4," << j * n + i + 1 << "," << (core ? 2 : 1) << "," << a << ","
              << a + 1 << "," << a + n + 2 << "," << a + n + 1 << "\n";
        }
    // Slots: the column left and right of the core, over the core's rows.
    for (int j = n / 2 - 1; j <= n / 2; ++j) {
        d << "ESET,1," << j * n + (n / 2 - 2) + 1 << "\n";
        d << "ESET,2," << j * n + (n / 2 + 1) + 1 << "\n";
    }
    d << "COIL2D,11,1,100\nCOIL2D,12,2,-100\nCIRCUIT,1,,11,1,12,1\n"
      << "ICIRC,1,1," << amps << "\nENDDATA\n";
    return d.str();
}
} // namespace

TEST_F(CudaFactorTest, MagstatOnCudssMatchesTheCpuSolution) {
    // Saturating steel: several Newton steps, each a cuDSS refactor of the
    // same pattern, then the inductance block solve.
    auto parsed = magnetic::parse_string(planar_winding_deck(8, 1000.0));
    ASSERT_TRUE(std::holds_alternative<magnetic::Model>(parsed))
        << std::get<Error>(parsed).message;
    const auto &model = std::get<magnetic::Model>(parsed);
    auto cpu = magnetic::solve(model);
    auto gpu = magnetic::solve(model, *backend_);
    ASSERT_TRUE(std::holds_alternative<magnetic::Results>(cpu)) << std::get<Error>(cpu).message;
    ASSERT_TRUE(std::holds_alternative<magnetic::Results>(gpu)) << std::get<Error>(gpu).message;
    const auto &c = std::get<magnetic::Results>(cpu).cases.at(0);
    const auto &g = std::get<magnetic::Results>(gpu).cases.at(0);
    EXPECT_GT(c.iterations, 2) << "the deck should be nonlinear";
    EXPECT_NEAR(g.energy, c.energy, 1e-10 * std::abs(c.energy));
    EXPECT_LT((g.potential - c.potential).norm(), 1e-9 * c.potential.norm());
    ASSERT_EQ(g.circuit_inductance.rows(), 1);
    EXPECT_NEAR(g.circuit_inductance(0, 0), c.circuit_inductance(0, 0),
                1e-9 * std::abs(c.circuit_inductance(0, 0)));
    EXPECT_GT(c.circuit_inductance(0, 0), 0.0);
}

// ── Hybrid mode ───────────────────────────────────────────────────────────────

void* cuda_test_hold_all_but(std::size_t leave);
void cuda_test_release(void* held);

namespace {

/// VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB for one scope.
struct DeviceBudget {
    explicit DeviceBudget(long mib) { setenv("VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB", std::to_string(mib).c_str(), 1); }
    ~DeviceBudget() { unsetenv("VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB"); }
    DeviceBudget(const DeviceBudget&) = delete;
    DeviceBudget& operator=(const DeviceBudget&) = delete;
};

/// 7-point Laplacian on a cube with a load from a known solution: enough fill
/// that the factor's device memory is a real amount.
struct CubeSystem {
    CsrData K;
    std::vector<double> exact, F;

    explicit CubeSystem(int m) {
        const auto node = [m](int i, int j, int k) { return i + m * (j + m * k); };
        const int n = m * m * m;
        K.n = n;
        K.symmetry = SymmetryStorage::Lower;
        K.row_ptr.reserve(static_cast<std::size_t>(n) + 1);
        K.row_ptr.push_back(0);
        for (int k = 0; k < m; ++k)
            for (int j = 0; j < m; ++j)
                for (int i = 0; i < m; ++i) {
                    if (k > 0) push(node(i, j, k - 1), -1.0);
                    if (j > 0) push(node(i, j - 1, k), -1.0);
                    if (i > 0) push(node(i - 1, j, k), -1.0);
                    push(node(i, j, k), 6.1);
                    K.row_ptr.push_back(static_cast<int>(K.col_ind.size()));
                }
        K.nnz = static_cast<int>(K.values.size());
        exact.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) exact[static_cast<std::size_t>(i)] = std::sin(0.01 * i) + 2.0;
        F = product(exact);
    }

    [[nodiscard]] std::vector<double> product(const std::vector<double>& x) const {
        std::vector<double> y(x.size(), 0.0);
        for (int i = 0; i < K.n; ++i)
            for (int p = K.row_ptr[static_cast<std::size_t>(i)]; p < K.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
                const auto a = static_cast<std::size_t>(i);
                const auto b = static_cast<std::size_t>(K.col_ind[static_cast<std::size_t>(p)]);
                y[a] += K.values[static_cast<std::size_t>(p)] * x[b];
                if (a != b) y[b] += K.values[static_cast<std::size_t>(p)] * x[a];
            }
        return y;
    }

    [[nodiscard]] double error(const std::vector<double>& u) const {
        double worst = 0.0;
        for (std::size_t i = 0; i < u.size(); ++i) worst = std::max(worst, std::abs(u[i] - exact[i]));
        return worst;
    }

private:
    void push(int column, double value) {
        K.col_ind.push_back(column);
        K.values.push_back(value);
    }
};

/// The lowest budget that cuDSS accepts, and what it reports there and at the
/// next budget refused.
struct HybridFloor {
    long lowest_mib = 0;
    CudssHybridMemory lowest{};
    bool saw_registration_off = false;
    std::string refusal;
};

HybridFloor find_floor(CudaSolverBackend& backend, const CubeSystem& cube, long above_mib) {
    HybridFloor floor;
    long high = above_mib, low = 0;
    while (high - low > 1) {
        const long middle = (high + low) / 2;
        const DeviceBudget budget(middle);
        try {
            (void)backend.solve(cube.K, cube.F);
            // Above the factor's own estimate the budget does not call for hybrid mode.
            if (const auto hybrid = backend.last_hybrid_memory()) {
                floor.lowest_mib = middle;
                floor.lowest = *hybrid;
                floor.saw_registration_off |= !hybrid->host_registration;
            }
            high = middle;
        } catch (const SolverError& e) {
            floor.refusal = e.what();
            low = middle;
        }
    }
    return floor;
}

} // namespace

class CudaHybridTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto created = CudaSolverBackend::try_create();
        if (!created) GTEST_SKIP() << "No CUDA device";
        backend_.emplace(std::move(*created));
        cube_ = &cube();
        floor_ = &shared_floor(*backend_);
        ASSERT_GT(floor_->lowest_mib, 0) << "no budget was accepted";
    }

    static const CubeSystem& cube() {
        static const CubeSystem system(32);
        return system;
    }
    // Finding the minimum takes one analysis per step, so the tests share it.
    static const HybridFloor& shared_floor(CudaSolverBackend& backend) {
        static const HybridFloor floor = find_floor(backend, cube(), 512);
        return floor;
    }

    std::optional<CudaSolverBackend> backend_;
    const CubeSystem* cube_ = nullptr;
    const HybridFloor* floor_ = nullptr;
};

// The defect this mode had: left to itself, cuDSS sets its limit from the whole card
// and fails at once when other allocations hold part of it. This runs with no budget,
// so the limit has to come from what is really free.
TEST_F(CudaHybridTest, AFactorTooLargeForFreeMemorySolvesWithoutABudget) {
    // cuDSS 0.7.1 estimates 1317 MiB on the device for this cube, with a minimum of
    // 643 MiB. A smaller cube leaves no room between the estimate and what the
    // analysis needs for itself: running that out of memory poisons the CUDA context.
    const CubeSystem larger(64);
    const std::size_t leave = std::size_t(1200) * 1024 * 1024;
    void* held = cuda_test_hold_all_but(leave);
    std::vector<double> u;
    std::optional<CudssHybridMemory> hybrid;
    std::string failure;
    try {
        u = backend_->solve(larger.K, larger.F);
        hybrid = backend_->last_hybrid_memory();
    } catch (const SolverError& e) {
        failure = e.what();
    }
    cuda_test_release(held);
    ASSERT_TRUE(failure.empty()) << failure;
    ASSERT_TRUE(hybrid.has_value()) << "the factor fit in the " << leave / (1024 * 1024) << " MiB left free";
    EXPECT_LE(hybrid->device_limit_bytes, static_cast<std::int64_t>(leave));
    EXPECT_GE(hybrid->device_limit_bytes, hybrid->device_minimum_bytes);
    EXPECT_LT(larger.error(u), 1e-10);
}

// A budget is the limit cuDSS is given, down to the minimum it reports, and below that
// the error says so instead of blaming the matrix or trying LU.
TEST_F(CudaHybridTest, ABudgetSetsTheLimitAndABudgetBelowTheMinimumIsRefused) {
    EXPECT_EQ(floor_->lowest.device_limit_bytes, floor_->lowest_mib * 1024 * 1024);
    EXPECT_LE(floor_->lowest.device_minimum_bytes, floor_->lowest.device_limit_bytes);
    EXPECT_TRUE(floor_->saw_registration_off);
    EXPECT_NE(floor_->refusal.find("needs at least"), std::string::npos) << floor_->refusal;
    EXPECT_NE(floor_->refusal.find("VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB"), std::string::npos) << floor_->refusal;
    EXPECT_NE(floor_->refusal.find("fp32"), std::string::npos) << floor_->refusal;

    const DeviceBudget at_the_floor(floor_->lowest_mib);
    const auto u = backend_->solve(cube_->K, cube_->F);
    EXPECT_TRUE(backend_->last_hybrid_memory().has_value());
    EXPECT_TRUE(backend_->last_solve_used_cholesky());
    EXPECT_LT(cube_->error(u), 1e-10);
}

// An fp32 factorization in hybrid mode still refines to double accuracy, and its
// refusal does not suggest what the run is already doing.
TEST_F(CudaHybridTest, Fp32InHybridModeRefinesAndItsRefusalDoesNotSuggestFp32) {
    auto created = CudaSolverBackend::try_create(CudaSolverPrecision::Float32);
    ASSERT_TRUE(created.has_value());
    CudaSolverBackend fp32(std::move(*created));
    {
        const DeviceBudget budget(floor_->lowest_mib);
        const auto u = fp32.solve(cube_->K, cube_->F);
        EXPECT_TRUE(fp32.last_hybrid_memory().has_value());
        EXPECT_LT(cube_->error(u), 1e-10);
    }
    const DeviceBudget too_small(1);
    try {
        (void)fp32.solve(cube_->K, cube_->F);
        ADD_FAILURE() << "a budget of 1 MiB was accepted";
    } catch (const SolverError& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("needs at least"), std::string::npos) << message;
        EXPECT_EQ(message.find("fp32"), std::string::npos) << message;
    }
}

// The retained factor takes the same path: several right-hand sides after an analysis
// with one, a refactorization with the same limit, and a refusal as an error value.
TEST_F(CudaHybridTest, ARetainedFactorInHybridModeSolvesManyRightHandSidesAndRefactors) {
    {
        const DeviceBudget budget(floor_->lowest_mib);
        auto created = backend_->factor(cube_->K);
        ASSERT_TRUE(std::holds_alternative<std::unique_ptr<LinearFactor>>(created))
            << std::get<Error>(created).message;
        auto& factor = *std::get<std::unique_ptr<LinearFactor>>(created);
        const auto n = static_cast<std::size_t>(cube_->K.n);
        std::vector<double> B(n * 3);
        for (std::size_t k = 0; k < 3; ++k)
            for (std::size_t i = 0; i < n; ++i) B[n * k + i] = cube_->F[i] * static_cast<double>(k + 1);
        auto solved = factor.solve(B, 3);
        ASSERT_TRUE(std::holds_alternative<std::vector<double>>(solved)) << std::get<Error>(solved).message;
        const auto& X = std::get<std::vector<double>>(solved);
        for (std::size_t k = 0; k < 3; ++k)
            for (std::size_t i = 0; i < n; i += 97)
                ASSERT_NEAR(X[n * k + i], cube_->exact[i] * static_cast<double>(k + 1), 1e-9) << "column " << k;

        CsrData doubled = cube_->K;
        std::transform(doubled.values.begin(), doubled.values.end(), doubled.values.begin(),
                       [](double v) { return 2.0 * v; });
        const auto status = factor.refactor(doubled);
        ASSERT_FALSE(status.has_value()) << status->message;
        auto again = factor.solve(cube_->F, 1);
        ASSERT_TRUE(std::holds_alternative<std::vector<double>>(again)) << std::get<Error>(again).message;
        for (std::size_t i = 0; i < n; i += 97)
            ASSERT_NEAR(std::get<std::vector<double>>(again)[i], cube_->exact[i] / 2.0, 1e-9);
    }
    const DeviceBudget too_small(1);
    const auto refused = backend_->factor(cube_->K);
    ASSERT_TRUE(std::holds_alternative<Error>(refused));
    EXPECT_NE(std::get<Error>(refused).message.find("needs at least"), std::string::npos)
        << std::get<Error>(refused).message;
}

#endif // HAVE_CUDA
