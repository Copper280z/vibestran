// tests/unit/test_cuda_factor.cpp
// CudaSolverBackend::factor(): cuDSS analysis reuse, multiple right-hand
// sides, require_spd, recovery after a failed factorization, the dense_block
// ordering, the LU fallback, fp32 refinement with its switch to fp64, and MAGSTAT on cuDSS.

#include <gtest/gtest.h>
#include "solver/solver_backend.hpp"
#include "hilbert_system.hpp"
#include "dense_block_system.hpp"
#ifdef HAVE_CUDA
#include "solver/cuda_solver_backend.hpp"
#include "magnetic/solver.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <sstream>
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

#endif // HAVE_CUDA
