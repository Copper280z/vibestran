// tests/unit/test_linear_factor.cpp
// SolverBackend::factor(): factor/solve separation, multiple right-hand
// sides, symbolic reuse across refactor(), the require_spd guarantee, the
// dense_block ordering, and the default adapter for backends without a
// reusable factorization.

#include <gtest/gtest.h>
#include "core/exceptions.hpp"
#include "solver/solver_backend.hpp"
#include "dense_block_system.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <string>
#if defined(EIGEN_CHOLMOD_SUPPORT) && !defined(HAVE_ACCELERATE)
#include <SuiteSparse_config.h>
#endif

using namespace vibestran;

namespace {

using CsrData = SparseMatrixBuilder::CsrData;

/// CSR of the nonzeros of a dense symmetric A: its lower triangle, or every
/// entry when full.
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

/// SPD tridiagonal matrix diag 2 + shift, off-diagonals -1. Its condition
/// number grows like n^2 for shift 0, so the solve is not trivially exact.
Eigen::MatrixXd tridiagonal(int n, double shift) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = 2.0 + shift;
        if (i > 0)
            A(i, i - 1) = A(i - 1, i) = -1.0;
    }
    return A;
}

/// Right-hand sides with a known solution: B = A X.
struct System {
    Eigen::MatrixXd X, B;
};
System system_for(const Eigen::MatrixXd &A, int nrhs) {
    System s;
    s.X = Eigen::MatrixXd(A.rows(), nrhs);
    for (int k = 0; k < nrhs; ++k)
        for (int i = 0; i < A.rows(); ++i)
            s.X(i, k) = std::sin(0.3 * (i + 1) * (k + 1)) + 0.1 * k;
    s.B = A * s.X;
    return s;
}

double relative_error(const std::vector<double> &x, const Eigen::MatrixXd &expected) {
    const Eigen::Map<const Eigen::MatrixXd> X(x.data(), expected.rows(), expected.cols());
    return (X - expected).norm() / expected.norm();
}

std::unique_ptr<LinearFactor> take(Result<std::unique_ptr<LinearFactor>> made) {
    if (auto *e = std::get_if<Error>(&made))
        ADD_FAILURE() << e->message;
    return std::holds_alternative<Error>(made)
               ? nullptr
               : std::get<std::unique_ptr<LinearFactor>>(std::move(made));
}

std::vector<double> solved(Result<std::vector<double>> x) {
    if (auto *e = std::get_if<Error>(&x)) {
        ADD_FAILURE() << e->message;
        return {};
    }
    return std::get<std::vector<double>>(std::move(x));
}

std::span<const double> columns(const Eigen::MatrixXd &B) {
    return {B.data(), static_cast<std::size_t>(B.size())};
}

/// Forwards solve() to the CPU direct backend but, like cuDSS, declares an
/// LU fallback, so it only has the default factor() adapter.
class IndefiniteFallbackBackend final : public SolverBackend {
public:
    std::vector<double> solve(const CsrData &K, const std::vector<double> &F,
                              const FactorRatioCheckPolicy *p = nullptr) override {
        ++solves;
        return inner_.solve(K, F, p);
    }
    [[nodiscard]] std::string_view name() const noexcept override { return "fallback"; }
    [[nodiscard]] bool may_factor_indefinite() const noexcept override { return true; }
    int solves{0};

private:
    EigenSolverBackend inner_;
};

/// A backend whose solve() fails the way the GPU backends do: by throwing,
/// or by returning a nonfinite result.
class BrokenBackend final : public SolverBackend {
public:
    explicit BrokenBackend(bool throws) : throws_(throws) {}
    std::vector<double> solve(const CsrData &K, const std::vector<double> &,
                              const FactorRatioCheckPolicy * = nullptr) override {
        if (throws_)
            throw SolverError("device ran out of memory");
        std::vector<double> x(static_cast<std::size_t>(K.n), 1.0);
        x.back() = std::numeric_limits<double>::quiet_NaN();
        return x;
    }
    [[nodiscard]] std::string_view name() const noexcept override { return "broken"; }

private:
    bool throws_;
};

} // namespace

TEST(LinearFactor, SolvesManyRightHandSidesWithOneFactorization) {
    const Eigen::MatrixXd A = tridiagonal(40, 0.0);
    const System s = system_for(A, 3);
    EigenSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    EXPECT_EQ(factor->size(), 40);

    const auto X = solved(factor->solve(columns(s.B), 3));
    ASSERT_EQ(X.size(), 120u);
    // cond(A) ~ 700, so a backward-stable solve is accurate to ~1e-13.
    EXPECT_LT(relative_error(X, s.X), 1e-12);

    // A single column gives the same answer as its column of the block.
    const auto x1 = solved(factor->solve(columns(s.B.col(1)), 1));
    EXPECT_LT(relative_error(x1, s.X.col(1)), 1e-12);

    const FactorStats stats = factor->stats();
    EXPECT_EQ(stats.symbolic_analyses, 1);
    EXPECT_EQ(stats.numeric_factorizations, 1);
    EXPECT_EQ(stats.solves, 4);
}

TEST(LinearFactor, RefactorReusesTheAnalysisAndUsesTheNewValues) {
    const Eigen::MatrixXd A = tridiagonal(30, 0.0);
    // Same pattern, different values: a stale numeric factor would solve A,
    // not A2, and miss the expected solution by O(1).
    Eigen::MatrixXd A2 = 3.0 * tridiagonal(30, 0.5);
    A2(0, 0) += 4.0;
    const System s2 = system_for(A2, 2);

    EigenSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A), {.require_spd = true}));
    ASSERT_TRUE(factor);
    ASSERT_FALSE(factor->refactor(to_csr(A2)));
    EXPECT_LT(relative_error(solved(factor->solve(columns(s2.B), 2)), s2.X), 1e-12);
    EXPECT_EQ(factor->stats().symbolic_analyses, 1);
    EXPECT_EQ(factor->stats().numeric_factorizations, 2);

    // A changed pattern is analyzed again, never misread with the old one.
    Eigen::MatrixXd A3 = tridiagonal(30, 1.0);
    for (int i = 2; i < 30; ++i)
        A3(i, i - 2) = A3(i - 2, i) = 0.25;
    const System s3 = system_for(A3, 1);
    ASSERT_FALSE(factor->refactor(to_csr(A3)));
    EXPECT_LT(relative_error(solved(factor->solve(columns(s3.B), 1)), s3.X), 1e-12);
    EXPECT_EQ(factor->stats().symbolic_analyses, 2);
    EXPECT_EQ(factor->stats().numeric_factorizations, 3);
}

TEST(LinearFactor, FullAndLowerStorageGiveTheSameSolution) {
    const Eigen::MatrixXd A = tridiagonal(25, 0.1);
    const System s = system_for(A, 2);
    EigenSolverBackend backend;
    auto lower = take(backend.factor(to_csr(A)));
    auto full = take(backend.factor(to_csr(A, true)));
    ASSERT_TRUE(lower && full);
    const auto xl = solved(lower->solve(columns(s.B), 2));
    const auto xf = solved(full->solve(columns(s.B), 2));
    EXPECT_LT(relative_error(xl, s.X), 1e-12);
    EXPECT_LT(relative_error(xf, s.X), 1e-12);
}

TEST(LinearFactor, RequireSpdRejectsIndefiniteMatricesWithoutFallback) {
    // Nonsingular but indefinite: eigenvalues of the tridiagonal shifted
    // below zero for the lowest modes.
    const Eigen::MatrixXd A = tridiagonal(20, -0.5);
    ASSERT_LT(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(A).eigenvalues()(0), 0.0);
    EigenSolverBackend backend;

    auto refused = backend.factor(to_csr(A), {.require_spd = true});
    ASSERT_TRUE(std::holds_alternative<Error>(refused));
    EXPECT_NE(std::get<Error>(refused).message.find("positive definite"),
              std::string::npos)
        << std::get<Error>(refused).message;

    // Without the requirement it is solved: by the LDLT fallback after a
    // failed Cholesky (Accelerate, Simplicial), or by CHOLMOD itself, whose
    // CholmodDecomposition may choose a simplicial LDL^T that accepts it.
    const System s = system_for(A, 1);
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    const std::string_view method = factor->method();
    EXPECT_TRUE(method.find("ldlt") != std::string_view::npos || method == "cholmod")
        << method;
    EXPECT_LT(relative_error(solved(factor->solve(columns(s.B), 1)), s.X), 1e-10);
}

TEST(LinearFactor, DenseBlockOrdersWithMinimumDegree) {
    // 2500 unknowns, 600 of them coupled densely: above CHOLMOD's dense-row
    // threshold 10 sqrt(n) = 500, where its AMD fill estimate (600 rows x
    // 2500 columns) sends the default analysis to METIS.
    const Eigen::MatrixXd A = test::gap_like(50, 600);
    const System s = system_for(A, 2);
    EigenSolverBackend backend;

    auto plain = take(backend.factor(to_csr(A), {.require_spd = true}));
    auto dense = take(backend.factor(to_csr(A), {.require_spd = true, .dense_block = true}));
    ASSERT_TRUE(plain && dense);
    // Accelerate defaults to METIS, and CHOLMOD picks it here on the
    // misjudged estimate; with dense_block both order with AMD. A set
    // VIBESTRAN_ACCELERATE_ORDER overrides either choice.
    if (std::getenv("VIBESTRAN_ACCELERATE_ORDER") == nullptr) {
#if defined(HAVE_ACCELERATE) || defined(EIGEN_CHOLMOD_SUPPORT)
        EXPECT_EQ(plain->ordering(), "metis");
#endif
        EXPECT_EQ(dense->ordering(), "amd");
    }
    // cond(A) is about 87, so either ordering solves to ~1e-14.
    EXPECT_LT(relative_error(solved(plain->solve(columns(s.B), 2)), s.X), 1e-11);
    EXPECT_LT(relative_error(solved(dense->solve(columns(s.B), 2)), s.X), 1e-11);

    // The ordering is part of the analysis that new values reuse.
    const Eigen::MatrixXd A2 = 2.0 * A + Eigen::MatrixXd::Identity(A.rows(), A.cols());
    const System s2 = system_for(A2, 1);
    ASSERT_FALSE(dense->refactor(to_csr(A2)));
    EXPECT_LT(relative_error(solved(dense->solve(columns(s2.B), 1)), s2.X), 1e-11);
    EXPECT_EQ(dense->stats().symbolic_analyses, 1);
    if (std::getenv("VIBESTRAN_ACCELERATE_ORDER") == nullptr)
        EXPECT_EQ(dense->ordering(), "amd");
}

TEST(LinearFactor, MinimumDegreeOrdersAPlanarMeshWithAmd) {
    // A planar 5-point mesh (no dense block): minimum_degree orders it with
    // AMD where Accelerate defaults to METIS, and keeps AMD when a new
    // pattern needs a new analysis (a sliding band's next position).
    const Eigen::MatrixXd A = test::gap_like(50, 0);
    const System s = system_for(A, 2);
    EigenSolverBackend backend;
    auto plain = take(backend.factor(to_csr(A), {.require_spd = true}));
    auto amd = take(backend.factor(to_csr(A), {.require_spd = true, .minimum_degree = true}));
    ASSERT_TRUE(plain && amd);
    if (std::getenv("VIBESTRAN_ACCELERATE_ORDER") == nullptr) {
#if defined(HAVE_ACCELERATE)
        EXPECT_EQ(plain->ordering(), "metis");
#endif
#if defined(HAVE_ACCELERATE) || defined(EIGEN_CHOLMOD_SUPPORT)
        EXPECT_EQ(amd->ordering(), "amd");
#endif
    }
    EXPECT_LT(relative_error(solved(plain->solve(columns(s.B), 2)), s.X), 1e-11);
    EXPECT_LT(relative_error(solved(amd->solve(columns(s.B), 2)), s.X), 1e-11);

    Eigen::MatrixXd A2 = A;
    A2(0, A.rows() - 1) = A2(A.rows() - 1, 0) = -0.5;
    const System s2 = system_for(A2, 1);
    ASSERT_FALSE(amd->refactor(to_csr(A2)));
    EXPECT_EQ(amd->stats().symbolic_analyses, 2);
    EXPECT_LT(relative_error(solved(amd->solve(columns(s2.B), 1)), s2.X), 1e-11);
#if defined(HAVE_ACCELERATE) || defined(EIGEN_CHOLMOD_SUPPORT)
    if (std::getenv("VIBESTRAN_ACCELERATE_ORDER") == nullptr)
        EXPECT_EQ(amd->ordering(), "amd");
#endif
}

TEST(LinearFactor, FailedRefactorCannotReturnTheOldSolution) {
    const Eigen::MatrixXd A = tridiagonal(20, 0.0);
    const System s = system_for(A, 1);
    EigenSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A), {.require_spd = true}));
    ASSERT_TRUE(factor);
    // Same pattern, indefinite values.
    EXPECT_TRUE(factor->refactor(to_csr(tridiagonal(20, -0.5))));
    EXPECT_TRUE(std::holds_alternative<Error>(factor->solve(columns(s.B), 1)));
    // A successful refactor makes it usable again.
    ASSERT_FALSE(factor->refactor(to_csr(A)));
    EXPECT_LT(relative_error(solved(factor->solve(columns(s.B), 1)), s.X), 1e-12);
}

#if defined(EIGEN_CHOLMOD_SUPPORT) && !defined(HAVE_ACCELERATE)
namespace {
// SuiteSparse's allocator, failing while armed: cholmod_analyze() then returns
// no factor, as it does for a problem too large for memory or for its index
// range.
bool g_fail_allocations = false;
void *(*g_malloc)(size_t) = nullptr;
void *(*g_calloc)(size_t, size_t) = nullptr;
void *(*g_realloc)(void *, size_t) = nullptr;
void *failing_malloc(size_t size) { return g_fail_allocations ? nullptr : g_malloc(size); }
void *failing_calloc(size_t count, size_t size) {
    return g_fail_allocations ? nullptr : g_calloc(count, size);
}
void *failing_realloc(void *p, size_t size) {
    return g_fail_allocations ? nullptr : g_realloc(p, size);
}

class FailingCholmodAllocator {
public:
    FailingCholmodAllocator() {
        g_malloc = SuiteSparse_config_malloc_func_get();
        g_calloc = SuiteSparse_config_calloc_func_get();
        g_realloc = SuiteSparse_config_realloc_func_get();
        SuiteSparse_config_malloc_func_set(failing_malloc);
        SuiteSparse_config_calloc_func_set(failing_calloc);
        SuiteSparse_config_realloc_func_set(failing_realloc);
    }
    ~FailingCholmodAllocator() {
        g_fail_allocations = false;
        SuiteSparse_config_malloc_func_set(g_malloc);
        SuiteSparse_config_calloc_func_set(g_calloc);
        SuiteSparse_config_realloc_func_set(g_realloc);
    }
    FailingCholmodAllocator(const FailingCholmodAllocator &) = delete;
    FailingCholmodAllocator &operator=(const FailingCholmodAllocator &) = delete;
    void arm(bool fail) { g_fail_allocations = fail; }
};
} // namespace

TEST(LinearFactor, CholmodAnalysisFailureIsAnErrorAndRecoverable) {
    // Eigen's wrapper reports success for an analysis that produced no
    // factor and then dereferences it in factorize(): the 16.9M-equation
    // plate segfaulted this way. The failure must be an error naming the
    // step and cause, with no LDL^T fallback, and the factor must work
    // again once memory is available.
    const Eigen::MatrixXd A = test::gap_like(50, 0);
    Eigen::MatrixXd A2 = A;
    A2(0, A.rows() - 1) = A2(A.rows() - 1, 0) = -0.5; // a new pattern
    const System s = system_for(A, 1), s2 = system_for(A2, 1);
    EigenSolverBackend backend;
    FailingCholmodAllocator allocator;

    allocator.arm(true);
    auto made = backend.factor(to_csr(A), {.minimum_degree = true});
    allocator.arm(false);
    ASSERT_TRUE(std::holds_alternative<Error>(made));
    const auto &message = std::get<Error>(made).message;
    EXPECT_NE(message.find("cholmod symbolic analysis failed: out of memory"), std::string::npos)
        << message;
    EXPECT_NE(message.find(std::format("n={}", A.rows())), std::string::npos) << message;

    auto factor = take(backend.factor(to_csr(A), {.minimum_degree = true}));
    ASSERT_TRUE(factor);
    allocator.arm(true);
    const auto failed = factor->refactor(to_csr(A2));
    allocator.arm(false);
    ASSERT_TRUE(failed.has_value());
    EXPECT_NE(failed->message.find("symbolic analysis failed"), std::string::npos) << failed->message;
    EXPECT_EQ(factor->method(), "cholmod");
    EXPECT_TRUE(std::holds_alternative<Error>(factor->solve(columns(s2.B), 1)));
    ASSERT_FALSE(factor->refactor(to_csr(A2)));
    EXPECT_LT(relative_error(solved(factor->solve(columns(s2.B), 1)), s2.X), 1e-11);
    EXPECT_LT(relative_error(solved(take(backend.factor(to_csr(A), {}))->solve(columns(s.B), 1)), s.X), 1e-11);
}
#endif

TEST(LinearFactor, ShapeErrorsAreReported) {
    const Eigen::MatrixXd A = tridiagonal(10, 0.0);
    EigenSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    const std::vector<double> short_rhs(9, 1.0);
    EXPECT_TRUE(std::holds_alternative<Error>(factor->solve(short_rhs, 1)));
    const std::vector<double> two_columns(20, 1.0);
    EXPECT_TRUE(std::holds_alternative<Error>(factor->solve(two_columns, 3)));
    EXPECT_TRUE(factor->refactor(to_csr(tridiagonal(11, 0.0))));
    EXPECT_TRUE(std::holds_alternative<Error>(backend.factor(CsrData{{0}, {}, {}, 0, 0})));
}

TEST(LinearFactor, AdapterSolvesThroughAnyBackend) {
    // EigenPCGSolverBackend has no factor() of its own: each column is a
    // full PCG solve, refined to its 1e-8 relative-error tolerance.
    const Eigen::MatrixXd A = tridiagonal(40, 0.05);
    const System s = system_for(A, 3);
    EigenPCGSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A), {.require_spd = true}));
    ASSERT_TRUE(factor);
    EXPECT_EQ(factor->method(), backend.name());
    EXPECT_LT(relative_error(solved(factor->solve(columns(s.B), 3)), s.X), 1e-7);
    EXPECT_EQ(factor->stats().solves, 3);
    EXPECT_EQ(factor->stats().numeric_factorizations, 3);

    // refactor() takes effect on the next solve.
    const Eigen::MatrixXd A2 = tridiagonal(40, 2.0);
    const System s2 = system_for(A2, 1);
    ASSERT_FALSE(factor->refactor(to_csr(A2)));
    EXPECT_LT(relative_error(solved(factor->solve(columns(s2.B), 1)), s2.X), 1e-7);
}

TEST(LinearFactor, AdapterRefusesSpdWhenTheBackendMayFallBack) {
    const Eigen::MatrixXd A = tridiagonal(10, 0.0);
    IndefiniteFallbackBackend backend;
    auto refused = backend.factor(to_csr(A), {.require_spd = true});
    ASSERT_TRUE(std::holds_alternative<Error>(refused));
    EXPECT_NE(std::get<Error>(refused).message.find("fallback"), std::string::npos);
    EXPECT_EQ(backend.solves, 0);

    const System s = system_for(A, 2);
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    EXPECT_LT(relative_error(solved(factor->solve(columns(s.B), 2)), s.X), 1e-12);
    EXPECT_EQ(backend.solves, 2);
}

TEST(LinearFactor, AdapterTurnsBackendFailuresIntoErrors) {
    const Eigen::MatrixXd A = tridiagonal(6, 0.0);
    const std::vector<double> b(6, 1.0);

    BrokenBackend throwing(true);
    auto factor = take(throwing.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    auto thrown = factor->solve(b, 1);
    ASSERT_TRUE(std::holds_alternative<Error>(thrown));
    EXPECT_NE(std::get<Error>(thrown).message.find("device ran out of memory"),
              std::string::npos);

    BrokenBackend nonfinite(false);
    factor = take(nonfinite.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    auto nan = factor->solve(b, 1);
    ASSERT_TRUE(std::holds_alternative<Error>(nan));
    EXPECT_NE(std::get<Error>(nan).message.find("nonfinite"), std::string::npos);
}
