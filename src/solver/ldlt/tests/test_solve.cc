#include <gtest/gtest.h>
#include "test_helpers.h"
#include <random>
#include <initializer_list>

static double residual(const CSCLower &A, const std::vector<double> &x,
                       const std::vector<double> &b) {
    auto Ax = spmv_sym(A, x);
    std::vector<double> r = b;
    for (size_t i = 0; i < r.size(); ++i) r[i] -= Ax[i];
    double nb = norm2(b);
    return norm2(r) / (nb > 0 ? nb : 1.0);
}

static CSCLower make_exchange2() {
    CSCLower A; A.n = 2;
    A.Ap = {0, 2, 3};
    A.Ai = {0, 1, 1};
    A.Ax = {0.0, 1.0, 0.0};
    return A;
}

static CSCLower make_delayed_indefinite3() {
    CSCLower A; A.n = 3;
    A.Ap = {0, 2, 4, 5};
    A.Ai = {0, 1, 1, 2, 2};
    A.Ax = {0.0, 1.0, 2.0, 1.0, 3.0};
    return A;
}

static CSCLower make_diag(std::initializer_list<double> diag) {
    CSCLower A; A.n = static_cast<int32_t>(diag.size());
    A.Ap.assign(A.n + 1, 0);
    A.Ai.resize(A.n);
    A.Ax.resize(A.n);
    int32_t j = 0;
    for (double d : diag) {
        A.Ap[j + 1] = A.Ap[j] + 1;
        A.Ai[j] = j;
        A.Ax[j] = d;
        ++j;
    }
    return A;
}

static CSCLower make_dense_indefinite_from_ldl(int32_t n) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> U(-0.25, 0.25);
    std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
    std::vector<double> D(n);
    for (int32_t j = 0; j < n; ++j) {
        L[j + static_cast<size_t>(j) * n] = 1.0;
        D[j] = (j % 2 == 0 ? 1.0 : -1.0) * (1.0 + 0.2 * j);
        for (int32_t i = j + 1; i < n; ++i)
            L[i + static_cast<size_t>(j) * n] = U(rng);
    }

    CSCLower A; A.n = n;
    A.Ap.assign(n + 1, 0);
    for (int32_t j = 0; j < n; ++j) A.Ap[j + 1] = A.Ap[j] + (n - j);
    A.Ai.resize(A.Ap[n]);
    A.Ax.resize(A.Ap[n]);
    for (int32_t j = 0; j < n; ++j) {
        int32_t p = A.Ap[j];
        for (int32_t i = j; i < n; ++i) {
            double a = 0.0;
            for (int32_t kk = 0; kk <= j; ++kk)
                a += L[i + static_cast<size_t>(kk) * n] * D[kk] *
                     L[j + static_cast<size_t>(kk) * n];
            A.Ai[p] = i;
            A.Ax[p] = a;
            ++p;
        }
    }
    return A;
}

TEST(Solve, Tridiag20_AMD) {
    auto A = make_tridiag_spd(20);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_AMD;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    ASSERT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> b(A.n); for (auto &v : b) v = U(rng);
    std::vector<double> x = b;
    ASSERT_EQ(ldlt_solve(N, 1, x.data(), A.n), LDLT_OK);
    EXPECT_LT(residual(A, x, b), 1e-10);
    ldlt_free_numeric(N); ldlt_free_symbolic(S);
}

TEST(Solve, Indefinite2x2Pivot) {
    auto A = make_exchange2();
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_NATURAL;
    opt.pivot = LDLT_PIVOT_BUNCH_KAUFMAN;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    ASSERT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);

    std::vector<double> b = {2.0, 3.0};
    std::vector<double> x = b;
    ASSERT_EQ(ldlt_solve(N, 1, x.data(), A.n), LDLT_OK);
    EXPECT_LT(residual(A, x, b), 1e-12);

    ldlt_free_numeric(N); ldlt_free_symbolic(S);
}

TEST(Solve, DelayedPivotNaturalOrdering) {
    auto A = make_delayed_indefinite3();
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_NATURAL;
    opt.relax_supernodes = 0;
    opt.pivot = LDLT_PIVOT_BUNCH_KAUFMAN;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ASSERT_GT(ldlt_nsuper(S), 1);
    ldlt_numeric *N = nullptr;
    ASSERT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);

    std::vector<double> b = {1.0, -2.0, 4.0};
    std::vector<double> x = b;
    ASSERT_EQ(ldlt_solve(N, 1, x.data(), A.n), LDLT_OK);
    EXPECT_LT(residual(A, x, b), 1e-11);

    ldlt_free_numeric(N); ldlt_free_symbolic(S);
}

TEST(Solve, ConstructedDenseIndefiniteMultiRHS) {
    auto A = make_dense_indefinite_from_ldl(8);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_NATURAL;
    opt.pivot = LDLT_PIVOT_BUNCH_KAUFMAN;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    ASSERT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);

    int32_t nrhs = 3;
    std::vector<double> B(static_cast<size_t>(A.n) * nrhs);
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = 0.1 + 0.03 * static_cast<double>(i);
    auto X = B;
    ASSERT_EQ(ldlt_solve(N, nrhs, X.data(), A.n), LDLT_OK);
    for (int32_t r = 0; r < nrhs; ++r) {
        std::vector<double> x(X.begin()+r*A.n, X.begin()+(r+1)*A.n);
        std::vector<double> b(B.begin()+r*A.n, B.begin()+(r+1)*A.n);
        EXPECT_LT(residual(A, x, b), 1e-10);
    }

    ldlt_free_numeric(N); ldlt_free_symbolic(S);
}

/* Larger dense indefinite matrix: exercises supernodes with multiple 2x2 pivot blocks. */
static void run_indef_solve_check(int32_t n, int32_t seed, int32_t nrhs,
                                  ldlt_ordering order, double tol) {
    auto A = make_dense_indefinite_from_ldl(n);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = order;
    opt.pivot = LDLT_PIVOT_BUNCH_KAUFMAN;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    ASSERT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> B(static_cast<size_t>(A.n) * nrhs);
    for (auto &v : B) v = U(rng);
    auto X = B;
    ASSERT_EQ(ldlt_solve(N, nrhs, X.data(), A.n), LDLT_OK);
    for (int32_t r = 0; r < nrhs; ++r) {
        std::vector<double> x(X.begin()+r*A.n, X.begin()+(r+1)*A.n);
        std::vector<double> b(B.begin()+r*A.n, B.begin()+(r+1)*A.n);
        EXPECT_LT(residual(A, x, b), tol) << "rhs=" << r << " n=" << n;
    }
    ldlt_free_numeric(N); ldlt_free_symbolic(S);
}

TEST(Solve, DenseIndef_n16_Natural)  { run_indef_solve_check(16,  42, 3, LDLT_ORDER_NATURAL, 1e-9); }
TEST(Solve, DenseIndef_n32_Natural)  { run_indef_solve_check(32,  43, 3, LDLT_ORDER_NATURAL, 1e-9); }
TEST(Solve, DenseIndef_n64_Natural)  { run_indef_solve_check(64,  44, 4, LDLT_ORDER_NATURAL, 1e-9); }
TEST(Solve, DenseIndef_n16_AMD)      { run_indef_solve_check(16,  45, 3, LDLT_ORDER_AMD,     1e-9); }
TEST(Solve, DenseIndef_n64_AMD)      { run_indef_solve_check(64,  46, 4, LDLT_ORDER_AMD,     1e-9); }

/* Sparse indefinite tridiagonal-like: alternating positive/negative diagonal. */
static CSCLower make_indef_tridiag(int32_t n) {
    CSCLower A; A.n = n;
    A.Ap.assign(n + 1, 0);
    for (int32_t j = 0; j < n; ++j) A.Ap[j+1] = A.Ap[j] + (j < n-1 ? 2 : 1);
    A.Ai.resize(A.Ap[n]); A.Ax.resize(A.Ap[n]);
    for (int32_t j = 0; j < n; ++j) {
        int32_t p = A.Ap[j];
        A.Ai[p] = j; A.Ax[p] = (j % 2 == 0 ? 3.0 : -3.0);
        if (j < n-1) { A.Ai[p+1] = j+1; A.Ax[p+1] = 1.0; }
    }
    return A;
}

static void run_sparse_indef_check(int32_t n, int32_t nrhs, ldlt_ordering order, double tol) {
    auto A = make_indef_tridiag(n);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = order;
    opt.pivot = LDLT_PIVOT_BUNCH_KAUFMAN;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    ASSERT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);
    std::mt19937 rng(77);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> B(static_cast<size_t>(A.n) * nrhs);
    for (auto &v : B) v = U(rng);
    auto X = B;
    ASSERT_EQ(ldlt_solve(N, nrhs, X.data(), A.n), LDLT_OK);
    for (int32_t r = 0; r < nrhs; ++r) {
        std::vector<double> x(X.begin()+r*A.n, X.begin()+(r+1)*A.n);
        std::vector<double> b(B.begin()+r*A.n, B.begin()+(r+1)*A.n);
        EXPECT_LT(residual(A, x, b), tol) << "rhs=" << r << " n=" << n;
    }
    ldlt_free_numeric(N); ldlt_free_symbolic(S);
}

TEST(Solve, SparseIndef_Tridiag50_Natural)  { run_sparse_indef_check(50,  2, LDLT_ORDER_NATURAL, 1e-9); }
TEST(Solve, SparseIndef_Tridiag100_Natural) { run_sparse_indef_check(100, 3, LDLT_ORDER_NATURAL, 1e-9); }
TEST(Solve, SparseIndef_Tridiag50_AMD)      { run_sparse_indef_check(50,  2, LDLT_ORDER_AMD,     1e-9); }
TEST(Solve, SparseIndef_Tridiag200_AMD)     { run_sparse_indef_check(200, 4, LDLT_ORDER_AMD,     1e-9); }

TEST(Factor, PivotingHonorsRequirePd) {
    auto A = make_diag({2.0, -3.0, 4.0});
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_NATURAL;
    opt.pivot = LDLT_PIVOT_BUNCH_KAUFMAN;
    opt.require_pd = 1;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    EXPECT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N),
              LDLT_ERR_INDEFINITE);
    ldlt_free_symbolic(S);
}

TEST(Solve, Tridiag20_Natural) {
    auto A = make_tridiag_spd(20);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_NATURAL;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    ASSERT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> b(A.n); for (auto &v : b) v = U(rng);
    std::vector<double> x = b;
    ASSERT_EQ(ldlt_solve(N, 1, x.data(), A.n), LDLT_OK);
    EXPECT_LT(residual(A, x, b), 1e-10);

    ldlt_free_numeric(N); ldlt_free_symbolic(S);
}

TEST(Solve, Tridiag50_MultiRHS) {
    auto A = make_tridiag_spd(50);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_NATURAL;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    ASSERT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);

    int32_t nrhs = 4;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> B(A.n * nrhs);
    for (auto &v : B) v = U(rng);
    auto X = B;
    ASSERT_EQ(ldlt_solve(N, nrhs, X.data(), A.n), LDLT_OK);
    for (int32_t r = 0; r < nrhs; ++r) {
        std::vector<double> x(X.begin()+r*A.n, X.begin()+(r+1)*A.n);
        std::vector<double> b(B.begin()+r*A.n, B.begin()+(r+1)*A.n);
        EXPECT_LT(residual(A, x, b), 1e-10);
    }
    ldlt_free_numeric(N); ldlt_free_symbolic(S);
}
