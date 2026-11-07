#include <gtest/gtest.h>
#include "test_helpers.h"
extern "C" {
#include "../src/internal.h"
}
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>
#include <string>
#include <utility>

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

struct Fac {
    ldlt_symbolic *S = nullptr;
    ldlt_numeric *N = nullptr;
    ldlt_status analyze = LDLT_OK, factor = LDLT_OK;
    ~Fac() { ldlt_free_numeric(N); ldlt_free_symbolic(S); }
};

void run(Fac &R, const CSCLower &A, const ldlt_options &opt) {
    R.analyze = ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &R.S);
    if (R.analyze == LDLT_OK)
        R.factor = ldlt_factorize(R.S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &R.N);
}

ldlt_options options(ldlt_ordering order, ldlt_pivot_strategy pivot = LDLT_PIVOT_NONE) {
    ldlt_options opt;
    ldlt_options_default(&opt);
    opt.ordering = order;
    opt.pivot = pivot;
    return opt;
}

CSCLower from_triplets(int32_t n, std::vector<std::tuple<int32_t, int32_t, double>> t) {
    for (auto &[i, j, v] : t) if (i < j) std::swap(i, j);
    std::sort(t.begin(), t.end(), [](const auto &a, const auto &b) {
        return std::pair(std::get<1>(a), std::get<0>(a)) < std::pair(std::get<1>(b), std::get<0>(b));
    });
    CSCLower A;
    A.n = n;
    A.Ap.assign(n + 1, 0);
    for (const auto &[i, j, v] : t) {
        if (!A.Ai.empty() && A.Ap[j + 1] > 0 && A.Ai.back() == i) { A.Ax.back() += v; continue; }
        A.Ai.push_back(i);
        A.Ax.push_back(v);
        A.Ap[j + 1]++;
    }
    for (int32_t j = 0; j < n; ++j) A.Ap[j + 1] += A.Ap[j];
    return A;
}

/* The 7-point Laplacian on a grid, shifted. Its eigenvalues are known in closed form, so the
   number of negative pivots can be checked exactly at any size. */
CSCLower grid_laplacian(int32_t nx, int32_t ny, int32_t nz, double shift) {
    std::vector<std::tuple<int32_t, int32_t, double>> t;
    auto id = [&](int32_t x, int32_t y, int32_t z) { return x + nx * (y + ny * z); };
    for (int32_t z = 0; z < nz; ++z)
        for (int32_t y = 0; y < ny; ++y)
            for (int32_t x = 0; x < nx; ++x) {
                int32_t a = id(x, y, z);
                t.emplace_back(a, a, 6.0 - shift);
                if (x + 1 < nx) t.emplace_back(id(x + 1, y, z), a, -1.0);
                if (y + 1 < ny) t.emplace_back(id(x, y + 1, z), a, -1.0);
                if (z + 1 < nz) t.emplace_back(id(x, y, z + 1), a, -1.0);
            }
    return from_triplets(nx * ny * nz, std::move(t));
}

int32_t grid_negative_eigenvalues(int32_t nx, int32_t ny, int32_t nz, double shift) {
    const double pi = std::acos(-1.0);
    int32_t count = 0;
    for (int32_t i = 1; i <= nx; ++i)
        for (int32_t j = 1; j <= ny; ++j)
            for (int32_t k = 1; k <= nz; ++k) {
                double l = 6.0 - 2.0 * (std::cos(i * pi / (nx + 1)) + std::cos(j * pi / (ny + 1)) +
                                        std::cos(k * pi / (nz + 1)));
                if (l < shift) count++;
            }
    return count;
}

/* Random symmetric pattern; off-diagonals in [-1, 1], diagonal from the callback. */
template <class Diag>
CSCLower random_sparse(int32_t n, int32_t per_row, unsigned seed, Diag diag) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> pick(0, n - 1);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    std::set<std::pair<int32_t, int32_t>> seen;
    std::vector<std::tuple<int32_t, int32_t, double>> t;
    for (int32_t i = 0; i < n; ++i) {
        t.emplace_back(i, i, diag(i));
        for (int32_t c = 0; c < per_row; ++c) {
            int32_t j = pick(rng);
            if (j == i || !seen.insert(std::minmax(i, j)).second) continue;
            t.emplace_back(i, j, val(rng));
        }
    }
    return from_triplets(n, std::move(t));
}

double backward_error(const CSCLower &A, const std::vector<double> &x, const std::vector<double> &b) {
    auto y = spmv_sym(A, x);
    double rmax = 0, xmax = 0, bmax = 0;
    std::vector<double> rowsum(A.n, 0.0);
    for (int32_t j = 0; j < A.n; ++j)
        for (int32_t p = A.Ap[j]; p < A.Ap[j + 1]; ++p) {
            rowsum[A.Ai[p]] += std::abs(A.Ax[p]);
            if (A.Ai[p] != j) rowsum[j] += std::abs(A.Ax[p]);
        }
    double anorm = *std::max_element(rowsum.begin(), rowsum.end());
    for (int32_t i = 0; i < A.n; ++i) {
        rmax = std::max(rmax, std::abs(y[i] - b[i]));
        xmax = std::max(xmax, std::abs(x[i]));
        bmax = std::max(bmax, std::abs(b[i]));
    }
    return rmax / (anorm * xmax + bmax);
}

std::vector<double> random_vector(int32_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    std::vector<double> b(n);
    for (auto &v : b) v = val(rng);
    return b;
}

/* Cyclic Jacobi: the eigenvalues of a small dense symmetric matrix, independently of the library. */
std::vector<double> dense_eigenvalues(const CSCLower &A) {
    const int32_t n = A.n;
    std::vector<double> M(static_cast<size_t>(n) * n, 0.0);
    auto at = [&](int32_t i, int32_t j) -> double & { return M[i + static_cast<size_t>(j) * n]; };
    for (int32_t j = 0; j < n; ++j)
        for (int32_t p = A.Ap[j]; p < A.Ap[j + 1]; ++p) at(A.Ai[p], j) = at(j, A.Ai[p]) = A.Ax[p];
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0;
        for (int32_t j = 0; j < n; ++j)
            for (int32_t i = 0; i < j; ++i) off += at(i, j) * at(i, j);
        if (off < 1e-26) break;
        for (int32_t p = 0; p < n; ++p)
            for (int32_t q = p + 1; q < n; ++q) {
                if (at(p, q) == 0.0) continue;
                double tau = (at(q, q) - at(p, p)) / (2.0 * at(p, q));
                double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                for (int32_t k = 0; k < n; ++k) {
                    double a = at(k, p), b = at(k, q);
                    at(k, p) = c * a - s * b;
                    at(k, q) = s * a + c * b;
                }
                for (int32_t k = 0; k < n; ++k) {
                    double a = at(p, k), b = at(q, k);
                    at(p, k) = c * a - s * b;
                    at(q, k) = s * a + c * b;
                }
            }
    }
    std::vector<double> e(n);
    for (int32_t i = 0; i < n; ++i) e[i] = at(i, i);
    return e;
}

ldlt_factor_info info_of(const ldlt_numeric *N) {
    ldlt_factor_info I;
    ldlt_get_info(N, &I);
    return I;
}

/* What elimination really gives, one column at a time, on the matrix in its final order. */
std::vector<std::vector<int32_t>> brute_force_structure(const CSCLower &A, const int32_t *iperm) {
    std::vector<std::set<int32_t>> col(A.n);
    for (int32_t j = 0; j < A.n; ++j)
        for (int32_t p = A.Ap[j]; p < A.Ap[j + 1]; ++p) {
            auto [lo, hi] = std::minmax(iperm[A.Ai[p]], iperm[j]);
            if (lo != hi) col[lo].insert(hi);
        }
    std::vector<std::vector<int32_t>> out(A.n);
    for (int32_t j = 0; j < A.n; ++j) {
        out[j].assign(col[j].begin(), col[j].end());
        if (col[j].empty()) continue;
        int32_t parent = *col[j].begin();
        for (int32_t r : col[j]) if (r != parent) col[parent].insert(r);
    }
    return out;
}

}  // namespace

/* The analysis never forms L, so check what it derives against L's real structure. */
TEST(Analyze, SupernodesMatchEliminationStructure) {
    for (ldlt_ordering order : {LDLT_ORDER_NATURAL, LDLT_ORDER_AUTO}) {
        for (int relax : {0, 16}) {
            for (unsigned seed : {1u, 2u, 3u}) {
                auto A = random_sparse(400, 3, seed, [](int32_t) { return 8.0; });
                auto opt = options(order);
                opt.relax_supernodes = relax;
                ldlt_symbolic *S = nullptr;
                ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
                auto structure = brute_force_structure(A, S->iperm);
                int64_t exact_lnz = 0;
                for (const auto &c : structure) exact_lnz += static_cast<int64_t>(c.size()) + 1;
                int64_t panel_lnz = 0;
                for (int32_t s = 0; s < S->nsuper; ++s) {
                    const ldlt_super &Ss = S->super[s];
                    int32_t last = Ss.first_col + Ss.width - 1;
                    /* A supernode has the rows below of its last column, whatever was merged into it. */
                    std::vector<int32_t> rows(S->rows_pool + Ss.row_off,
                                              S->rows_pool + Ss.row_off + Ss.nrows_below);
                    ASSERT_EQ(rows, structure[last]) << "supernode " << s << " seed " << seed;
                    for (int32_t c = Ss.first_col; c < last; ++c) {
                        /* Each column's own entries must fit inside the supernode's front. */
                        for (int32_t r : structure[c]) {
                            bool inside = r <= last || std::binary_search(rows.begin(), rows.end(), r);
                            ASSERT_TRUE(inside) << "column " << c << " row " << r;
                        }
                    }
                    int64_t k = Ss.width, m = Ss.nrows_below;
                    panel_lnz += k * (k + 1) / 2 + m * k;
                }
                EXPECT_EQ(S->lnz, panel_lnz);
                if (relax == 0) EXPECT_EQ(S->lnz, exact_lnz);
                else EXPECT_GE(S->lnz, exact_lnz);
                ldlt_free_symbolic(S);
            }
        }
    }
}

TEST(Analyze, AssemblyMapsPlaceEveryChildRowInItsParentFront) {
    auto A = random_sparse(600, 3, 7, [](int32_t) { return 8.0; });
    auto opt = options(LDLT_ORDER_AUTO);
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    for (int32_t c = 0; c < S->nsuper; ++c) {
        int32_t p = S->super_parent[c];
        if (p < 0) { EXPECT_EQ(S->super[c].nrows_below, 0); continue; }
        const ldlt_super &Sc = S->super[c], &Sp = S->super[p];
        const int32_t *rows = S->rows_pool + Sc.row_off;
        const int32_t *prows = S->rows_pool + Sp.row_off;
        const int32_t *emap = S->edge_pos + Sc.row_off;
        for (int32_t t = 0; t < Sc.nrows_below; ++t) {
            int32_t e = emap[t];
            int32_t row = e < Sp.width ? Sp.first_col + e : prows[e - Sp.width];
            ASSERT_EQ(row, rows[t]);
            if (t > 0) {
                ASSERT_GT(e, emap[t - 1]);
            }
        }
    }
    ldlt_free_symbolic(S);
}

TEST(Analyze, RejectsMalformedInput) {
    auto A = make_tridiag_spd(5);
    auto opt = options(LDLT_ORDER_NATURAL);
    ldlt_symbolic *S = nullptr;

    auto bad = A; bad.Ap[0] = 1;
    EXPECT_EQ(ldlt_analyze(bad.n, bad.Ap.data(), bad.Ai.data(), &opt, &S), LDLT_ERR_INPUT);
    bad = A; std::swap(bad.Ap[2], bad.Ap[3]);
    EXPECT_EQ(ldlt_analyze(bad.n, bad.Ap.data(), bad.Ai.data(), &opt, &S), LDLT_ERR_INPUT);
    bad = A; bad.Ai[1] = 99;
    EXPECT_EQ(ldlt_analyze(bad.n, bad.Ap.data(), bad.Ai.data(), &opt, &S), LDLT_ERR_INPUT);
    /* An entry above the diagonal. */
    bad = A; bad.Ai[2] = 0;
    EXPECT_EQ(ldlt_analyze(bad.n, bad.Ap.data(), bad.Ai.data(), &opt, &S), LDLT_ERR_INPUT);
    /* Rows out of order, and a repeated row. */
    bad = A; std::swap(bad.Ai[0], bad.Ai[1]);
    EXPECT_EQ(ldlt_analyze(bad.n, bad.Ap.data(), bad.Ai.data(), &opt, &S), LDLT_ERR_INPUT);
    bad = A; bad.Ai[1] = bad.Ai[0];
    EXPECT_EQ(ldlt_analyze(bad.n, bad.Ap.data(), bad.Ai.data(), &opt, &S), LDLT_ERR_INPUT);

    /* A user ordering that is not a permutation. */
    auto user = options(LDLT_ORDER_USER);
    const std::vector<int32_t> repeated = {0, 1, 2, 3, 3};
    user.user_perm = repeated.data();
    EXPECT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &user, &S), LDLT_ERR_INPUT);
    const std::vector<int32_t> out_of_range = {0, 1, 2, 3, 5};
    user.user_perm = out_of_range.data();
    EXPECT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &user, &S), LDLT_ERR_INPUT);
    user.user_perm = nullptr;
    EXPECT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &user, &S), LDLT_ERR_INPUT);
    EXPECT_EQ(S, nullptr);
}

/* A graph with no edges: METIS's arithmetic on it has to survive x86, where dividing by zero traps. */
TEST(Analyze, DiagonalMatrixUnderEveryOrdering) {
    auto A = from_triplets(6, {{0, 0, 2.0}, {1, 1, -4.0}, {2, 2, 8.0}, {3, 3, -1.0}, {4, 4, 5.0}, {5, 5, 10.0}});
    const std::vector<double> b = {2.0, 4.0, 8.0, 1.0, 5.0, 10.0};
    for (auto pivot : {LDLT_PIVOT_NONE, LDLT_PIVOT_THRESHOLD}) {
        for (auto order : {LDLT_ORDER_AUTO, LDLT_ORDER_AMD, LDLT_ORDER_METIS, LDLT_ORDER_NATURAL}) {
            Fac R; run(R, A, options(order, pivot));
            if (R.analyze == LDLT_ERR_UNSUPPORTED) continue;
            ASSERT_EQ(R.factor, LDLT_OK);
            auto x = b;
            ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), A.n), LDLT_OK);
            const std::vector<double> expected = {1.0, -1.0, 1.0, -1.0, 1.0, 1.0};
            for (int i = 0; i < A.n; ++i) EXPECT_DOUBLE_EQ(x[i], expected[i]);
            EXPECT_EQ(info_of(R.N).n_negative, 2);
            EXPECT_EQ(info_of(R.N).n_positive, 4);
        }
    }
}

TEST(Analyze, EmptyAndSingleUnknown) {
    for (auto pivot : {LDLT_PIVOT_NONE, LDLT_PIVOT_THRESHOLD}) {
        CSCLower E;
        E.Ap = {0};
        Fac R; run(R, E, options(LDLT_ORDER_AUTO, pivot));
        ASSERT_EQ(R.factor, LDLT_OK);
        EXPECT_EQ(ldlt_solve(R.N, 1, nullptr, 0), LDLT_ERR_INPUT);
        EXPECT_EQ(ldlt_solve(R.N, 0, nullptr, 0), LDLT_OK);
        EXPECT_EQ(info_of(R.N).n_positive, 0);

        auto One = from_triplets(1, {{0, 0, -4.0}});
        Fac Q; run(Q, One, options(LDLT_ORDER_AUTO, pivot));
        ASSERT_EQ(Q.factor, LDLT_OK);
        double x = 2.0;
        ASSERT_EQ(ldlt_solve(Q.N, 1, &x, 1), LDLT_OK);
        EXPECT_DOUBLE_EQ(x, -0.5);
        EXPECT_EQ(info_of(Q.N).n_negative, 1);
    }
}

/* A missing ordering must not quietly become no ordering: fill would explode without a word. */
TEST(Analyze, OrderingsAreHonouredOrRefused) {
    auto A = grid_laplacian(12, 12, 12, 0.0);
    int64_t natural_lnz = 0;
    {
        Fac R; run(R, A, options(LDLT_ORDER_NATURAL));
        ASSERT_EQ(R.analyze, LDLT_OK);
        natural_lnz = ldlt_lnz(R.S);
    }
    const struct { ldlt_ordering order; bool built; } cases[] = {
        {LDLT_ORDER_AMD, LDLT_HAVE_AMD != 0}, {LDLT_ORDER_METIS, LDLT_HAVE_METIS != 0}};
    for (const auto &c : cases) {
        Fac R; run(R, A, options(c.order));
        if (!c.built) { EXPECT_EQ(R.analyze, LDLT_ERR_UNSUPPORTED); continue; }
        ASSERT_EQ(R.analyze, LDLT_OK);
        EXPECT_EQ(ldlt_ordering_used(R.S), c.order);
        EXPECT_LT(ldlt_lnz(R.S), natural_lnz * 7 / 10);
    }
    Fac R; run(R, A, options(LDLT_ORDER_AUTO));
    ASSERT_EQ(R.analyze, LDLT_OK);
    ldlt_ordering expected = LDLT_HAVE_METIS ? LDLT_ORDER_METIS
                           : LDLT_HAVE_AMD ? LDLT_ORDER_AMD : LDLT_ORDER_NATURAL;
    EXPECT_EQ(ldlt_ordering_used(R.S), expected);
}

/* A factorization given another matrix's pattern would otherwise solve the wrong system. */
TEST(Factor, RefusesAPatternOtherThanTheAnalyzedOne) {
    auto A = grid_laplacian(3, 3, 3, 0.0);
    auto opt = options(LDLT_ORDER_AUTO);
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    auto moved = A;
    moved.Ai[1] = moved.Ai[1] + 1;
    EXPECT_EQ(ldlt_factorize(S, moved.Ap.data(), moved.Ai.data(), moved.Ax.data(), &opt, &N), LDLT_ERR_INPUT);
    auto shorter = make_tridiag_spd(A.n);
    EXPECT_EQ(ldlt_factorize(S, shorter.Ap.data(), shorter.Ai.data(), shorter.Ax.data(), &opt, &N), LDLT_ERR_INPUT);
    EXPECT_EQ(N, nullptr);
    ldlt_free_symbolic(S);
}

/* The new values reach the factors, through the same analysis. */
TEST(Factor, RefactorizationUsesTheNewValues) {
    auto A = grid_laplacian(8, 8, 8, 0.0);
    auto opt = options(LDLT_ORDER_AUTO);
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    auto b = random_vector(A.n, 5);
    for (double scale : {1.0, 3.0}) {
        auto B = A;
        for (auto &v : B.Ax) v *= scale;
        ldlt_numeric *N = nullptr;
        ASSERT_EQ(ldlt_factorize(S, B.Ap.data(), B.Ai.data(), B.Ax.data(), &opt, &N), LDLT_OK);
        auto x = b;
        ASSERT_EQ(ldlt_solve(N, 1, x.data(), A.n), LDLT_OK);
        EXPECT_LT(backward_error(B, x, b), 1e-14);
        ldlt_free_numeric(N);
    }
    ldlt_free_symbolic(S);
}

/* Fronts of every size, in both paths, with the subtree phase running: the shifted grid's
   inertia is known exactly. */
TEST(Factor, GridInertiaIsExactForBothPaths) {
    const int32_t nx = 22, ny = 20, nz = 18;
    for (double shift : {0.0, 0.3, 1.0, 3.0}) {
        auto A = grid_laplacian(nx, ny, nz, shift);
        int32_t expected = grid_negative_eigenvalues(nx, ny, nz, shift);
        auto b = random_vector(A.n, 11);
        for (auto pivot : {LDLT_PIVOT_NONE, LDLT_PIVOT_THRESHOLD}) {
            /* Far from positive definite, no pivoting is simply unstable: a tiny pivot can
               wreck the factors, and their inertia along with them. */
            if (pivot == LDLT_PIVOT_NONE && shift > 0.5) continue;
            auto opt = options(LDLT_ORDER_AUTO, pivot);
            opt.nthreads = 4;
            Fac R; run(R, A, opt);
            ASSERT_EQ(R.factor, LDLT_OK);
            auto I = info_of(R.N);
            EXPECT_EQ(I.n_negative, expected) << "shift " << shift << " pivot " << pivot;
            EXPECT_EQ(I.n_positive, A.n - expected);
            EXPECT_EQ(I.n_zero, 0);
            auto x = b;
            ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), A.n), LDLT_OK);
            /* Without pivoting a shifted grid can lose accuracy. With it, entries of L up to
               1/u = 100 can still cost a couple of digits against machine precision. */
            EXPECT_LT(backward_error(A, x, b), pivot == LDLT_PIVOT_NONE ? 1e-8 : 1e-12);
        }
    }
}

/* Saddle points have zero diagonals, so nothing can be done without 2 x 2 or delayed pivots.
   By Sylvester's law with H positive definite and B of full rank, the inertia is (n_H, n_B). */
TEST(Factor, SaddlePointInertiaAndSolve) {
    const int32_t nh = 500, nb = 120;
    std::mt19937 rng(3);
    std::uniform_int_distribution<int32_t> pick(0, nh - 1);
    std::vector<std::tuple<int32_t, int32_t, double>> t;
    for (int32_t i = 0; i < nh; ++i) t.emplace_back(i, i, 1.0 + (i % 5));
    for (int32_t i = 0; i < nb; ++i) {
        /* Each row of B owns a column, which gives it full rank. */
        t.emplace_back(nh + i, i * 4, 1.0);
        for (int c = 0; c < 3; ++c) {
            int32_t j = pick(rng);
            if (j == i * 4) continue;
            t.emplace_back(nh + i, j, (j % 2 ? 0.5 : -0.5));
        }
    }
    auto A = from_triplets(nh + nb, t);
    auto b = random_vector(A.n, 8);
    for (ldlt_ordering order : {LDLT_ORDER_NATURAL, LDLT_ORDER_AUTO}) {
        for (int32_t threads : {1, 4}) {
            auto opt = options(order, LDLT_PIVOT_THRESHOLD);
            opt.nthreads = threads;
            Fac R; run(R, A, opt);
            ASSERT_EQ(R.factor, LDLT_OK);
            auto I = info_of(R.N);
            EXPECT_EQ(I.n_positive, nh);
            EXPECT_EQ(I.n_negative, nb);
            EXPECT_EQ(I.n_zero, 0);
            auto x = b;
            ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), A.n), LDLT_OK);
            EXPECT_LT(backward_error(A, x, b), 1e-13);
        }
    }
}

/* Zero diagonal first: this column has nothing acceptable in its own front, and has to wait
   for its parent. */
TEST(Factor, ColumnsWithNoAcceptablePivotAreDelayed) {
    /* A long chain of zero-diagonal columns, each coupled to the next only. */
    const int32_t n = 64;
    std::vector<std::tuple<int32_t, int32_t, double>> t;
    for (int32_t i = 0; i < n; ++i) {
        t.emplace_back(i, i, i % 2 ? 2.0 : 0.0);
        if (i + 1 < n) t.emplace_back(i + 1, i, 1.0);
    }
    auto A = from_triplets(n, t);
    auto opt = options(LDLT_ORDER_NATURAL, LDLT_PIVOT_THRESHOLD);
    opt.relax_supernodes = 0;
    Fac R; run(R, A, opt);
    ASSERT_EQ(R.factor, LDLT_OK);
    auto I = info_of(R.N);
    EXPECT_GT(I.n_delayed + I.n_2x2, 0);
    auto e = dense_eigenvalues(A);
    int32_t neg = 0;
    for (double v : e) if (v < 0) neg++;
    EXPECT_EQ(I.n_negative, neg);
    EXPECT_EQ(I.n_positive, n - neg);
    auto b = random_vector(n, 2);
    auto x = b;
    ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), n), LDLT_OK);
    EXPECT_LT(backward_error(A, x, b), 1e-14);
}

/* Small random indefinite matrices against eigenvalues computed without the library. */
TEST(Factor, RandomIndefiniteInertiaMatchesEigenvalues) {
    for (unsigned seed = 0; seed < 40; ++seed) {
        int32_t n = 30 + static_cast<int32_t>(seed) * 4;
        double sgn = seed % 3 == 0 ? 0.0 : 1.0;
        auto A = random_sparse(n, 2 + seed % 5, seed, [&](int32_t i) {
            /* Mixed signs and, a third of the time, empty diagonals. */
            return sgn * ((i * 2654435761u + seed) % 7 < 3 ? -1.5 : 1.5);
        });
        auto e = dense_eigenvalues(A);
        int32_t neg = 0, pos = 0;
        for (double v : e) { if (v < -1e-9) neg++; else if (v > 1e-9) pos++; }
        if (neg + pos != n) continue;  /* numerically singular draws say nothing about inertia */
        for (auto order : {LDLT_ORDER_NATURAL, LDLT_ORDER_AUTO}) {
            Fac R; run(R, A, options(order, LDLT_PIVOT_THRESHOLD));
            ASSERT_EQ(R.factor, LDLT_OK) << "seed " << seed;
            auto I = info_of(R.N);
            EXPECT_EQ(I.n_negative, neg) << "seed " << seed;
            EXPECT_EQ(I.n_positive, pos) << "seed " << seed;
            auto b = random_vector(n, seed + 100);
            auto x = b;
            ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), n), LDLT_OK);
            EXPECT_LT(backward_error(A, x, b), 1e-11) << "seed " << seed;

            /* What the API reports must give the same inertia. */
            std::vector<double> d(n), off(n);
            std::vector<int32_t> partner(n);
            ldlt_get_d(R.N, d.data(), off.data(), partner.data());
            int32_t rneg = 0;
            for (int32_t i = 0; i < n; ++i) {
                if (partner[i] < 0) { if (d[i] < 0) rneg++; continue; }
                ASSERT_EQ(partner[partner[i]], i);
                ASSERT_EQ(off[i], off[partner[i]]);
                if (partner[i] < i) continue;
                double det = d[i] * d[partner[i]] - off[i] * off[i];
                if (det < 0) rneg += 1;
                else if (d[i] < 0) rneg += 2;
            }
            EXPECT_EQ(rneg, neg);
        }
    }
}

/* Both a front that Cholesky can do and one it can't, past the block size, so the BLAS paths
   run: a mixed-sign dense matrix goes through the blocked kernel and signed update. */
TEST(Factor, DenseFrontsWithoutPivoting) {
    for (int32_t n : {5, 20, 49, 130, 300}) {
        for (bool mixed : {false, true}) {
            auto A = random_sparse(n, n, 21 + n, [&](int32_t i) {
                return (mixed && i % 3 == 0 ? -1.0 : 1.0) * (2.0 * n);
            });
            auto opt = options(LDLT_ORDER_NATURAL);
            Fac R; run(R, A, opt);
            ASSERT_EQ(R.factor, LDLT_OK);
            /* Strict diagonal dominance fixes the inertia by the signs of the diagonal. */
            int32_t neg = 0;
            if (mixed) for (int32_t i = 0; i < n; ++i) if (i % 3 == 0) neg++;
            auto I = info_of(R.N);
            EXPECT_EQ(I.n_negative, neg);
            EXPECT_EQ(I.n_positive, n - neg);
            std::vector<double> d(n);
            ldlt_get_d(R.N, d.data(), nullptr, nullptr);
            /* The product of D is det(A). The first pivot in natural order is A_00 itself. */
            EXPECT_DOUBLE_EQ(d[0], A.Ax[A.Ap[0]]);
            auto b = random_vector(n, 3);
            auto x = b;
            ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), n), LDLT_OK);
            EXPECT_LT(backward_error(A, x, b), 1e-14);
        }
    }
}

/* A free-free spring chain: the classic singular stiffness matrix. */
TEST(Held, ZeroPivotActsAsAConstraint) {
    auto A = from_triplets(3, {{0, 0, 1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {2, 1, -1.0}, {2, 2, 1.0}});
    auto opt = options(LDLT_ORDER_NATURAL);
    Fac R; run(R, A, opt);
    ASSERT_EQ(R.factor, LDLT_OK);
    auto I = info_of(R.N);
    EXPECT_EQ(I.n_zero, 1);
    EXPECT_EQ(I.n_positive, 2);
    int32_t rows[3] = {-1, -1, -1};
    ASSERT_EQ(ldlt_get_held(R.N, nullptr), 1);
    ASSERT_EQ(ldlt_get_held(R.N, rows), 1);
    EXPECT_EQ(rows[0], 2);
    double d[3];
    ldlt_get_d(R.N, d, nullptr, nullptr);
    EXPECT_EQ(d[2], 0.0);

    /* A self-equilibrating load has an exact solution with the held unknown at zero. */
    std::vector<double> x = {1.0, 0.0, -1.0};
    auto b = x;
    ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), 3), LDLT_OK);
    EXPECT_EQ(x[2], 0.0);
    auto y = spmv_sym(A, x);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(y[i], b[i], 1e-14);

    /* A load the mechanism can't carry shows up in the residual, which is the caller's check. */
    std::vector<double> z = {1.0, 0.0, 0.0};
    ASSERT_EQ(ldlt_solve(R.N, 1, z.data(), 3), LDLT_OK);
    EXPECT_EQ(z[2], 0.0);
    y = spmv_sym(A, z);
    EXPECT_NEAR(y[2], -1.0, 1e-14);

    auto pd = options(LDLT_ORDER_NATURAL);
    pd.require_pd = 1;
    Fac P; run(P, A, pd);
    EXPECT_EQ(P.factor, LDLT_ERR_INDEFINITE);
}

/* Holding must equal deleting the row and column, including when the zero pivot is inside a
   large front that Cholesky can't take. Checked against the factorization of the reduced matrix. */
TEST(Held, MatchesDeletingTheRowAndColumn) {
    const int32_t n = 200, zero_at = 73;
    auto full = random_sparse(n, n, 5, [&](int32_t) { return 2.0 * n; });
    /* Disconnect one unknown entirely. */
    std::vector<std::tuple<int32_t, int32_t, double>> tf, tr;
    for (int32_t j = 0; j < n; ++j)
        for (int32_t p = full.Ap[j]; p < full.Ap[j + 1]; ++p) {
            int32_t i = full.Ai[p];
            double v = full.Ax[p];
            bool removed = i == zero_at || j == zero_at;
            tf.emplace_back(i, j, removed ? 0.0 : v);
            if (!removed) tr.emplace_back(i > zero_at ? i - 1 : i, j > zero_at ? j - 1 : j, v);
        }
    auto A = from_triplets(n, tf);
    auto Red = from_triplets(n - 1, tr);
    auto bred = random_vector(n - 1, 6);
    std::vector<double> b(n);
    for (int32_t i = 0; i < n; ++i) b[i] = i == zero_at ? 0.0 : bred[i > zero_at ? i - 1 : i];

    for (auto pivot : {LDLT_PIVOT_NONE, LDLT_PIVOT_THRESHOLD}) {
        for (auto order : {LDLT_ORDER_NATURAL, LDLT_ORDER_AUTO}) {
            Fac R; run(R, A, options(order, pivot));
            ASSERT_EQ(R.factor, LDLT_OK);
            int32_t held = -1;
            ASSERT_EQ(ldlt_get_held(R.N, &held), 1);
            EXPECT_EQ(held, zero_at);
            auto x = b;
            ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), n), LDLT_OK);
            EXPECT_EQ(x[zero_at], 0.0);

            Fac Q; run(Q, Red, options(order));
            ASSERT_EQ(Q.factor, LDLT_OK);
            auto xr = bred;
            ASSERT_EQ(ldlt_solve(Q.N, 1, xr.data(), n - 1), LDLT_OK);
            for (int32_t i = 0; i < n; ++i) {
                if (i == zero_at) continue;
                EXPECT_NEAR(x[i], xr[i > zero_at ? i - 1 : i], 1e-13);
            }
        }
    }
}

TEST(Held, AllZeroMatrixWithPivoting) {
    auto A = from_triplets(4, {{0, 0, 0.0}, {1, 0, 0.0}, {1, 1, 0.0}, {2, 2, 3.0}, {3, 3, 0.0}});
    Fac R; run(R, A, options(LDLT_ORDER_NATURAL, LDLT_PIVOT_THRESHOLD));
    ASSERT_EQ(R.factor, LDLT_OK);
    auto I = info_of(R.N);
    EXPECT_EQ(I.n_zero, 3);
    EXPECT_EQ(I.n_positive, 1);
    std::vector<double> x = {1.0, 1.0, 6.0, 1.0};
    ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), 4), LDLT_OK);
    EXPECT_EQ(x[0], 0.0);
    EXPECT_EQ(x[1], 0.0);
    EXPECT_NEAR(x[2], 2.0, 1e-15);
    EXPECT_EQ(x[3], 0.0);
}

TEST(Factor, NonFiniteValuesAreReported) {
    for (auto pivot : {LDLT_PIVOT_NONE, LDLT_PIVOT_THRESHOLD}) {
        auto A = make_tridiag_spd(40);
        A.Ax[A.Ap[20] + 1] = std::nan("");
        Fac R; run(R, A, options(LDLT_ORDER_AUTO, pivot));
        EXPECT_EQ(R.factor, LDLT_ERR_NUMERIC);
        auto B = make_tridiag_spd(40);
        B.Ax[B.Ap[5]] = INFINITY;
        Fac Q; run(Q, B, options(LDLT_ORDER_AUTO, pivot));
        EXPECT_EQ(Q.factor, LDLT_ERR_NUMERIC);
    }
}

/* D and the order, in the caller's numbering, with scaling undone. */
TEST(Api, DAndPermAreInTheCallersNumbering) {
    /* Widely scaled, so a missed unscaling would show. */
    auto A = from_triplets(5, {{0, 0, 3e6}, {1, 1, 0.0}, {2, 1, 2e-3}, {2, 2, 0.0}, {3, 3, -5e-4}, {4, 4, 7.0}});
    for (auto order : {LDLT_ORDER_NATURAL, LDLT_ORDER_AUTO}) {
        Fac R; run(R, A, options(order, LDLT_PIVOT_THRESHOLD));
        ASSERT_EQ(R.factor, LDLT_OK);
        double d[5], e[5];
        int32_t partner[5];
        ldlt_get_d(R.N, d, e, partner);
        EXPECT_DOUBLE_EQ(d[0], 3e6);
        EXPECT_DOUBLE_EQ(d[3], -5e-4);
        EXPECT_DOUBLE_EQ(d[4], 7.0);
        EXPECT_EQ(partner[0], -1);
        EXPECT_EQ(partner[1], 2);
        EXPECT_EQ(partner[2], 1);
        EXPECT_DOUBLE_EQ(e[1], 2e-3);
        EXPECT_DOUBLE_EQ(e[2], 2e-3);
        auto I = info_of(R.N);
        EXPECT_EQ(I.n_2x2, 1);
        EXPECT_EQ(I.n_positive, 3);
        EXPECT_EQ(I.n_negative, 2);
    }
}

/* Reassemble A from what the API gives, on a matrix with no 2 x 2 pivots: D in elimination
   order against a direct dense factorization in that same order. */
TEST(Api, PermAndDDescribeTheFactorization) {
    for (auto pivot : {LDLT_PIVOT_NONE, LDLT_PIVOT_THRESHOLD}) {
        auto A = grid_laplacian(7, 6, 5, 0.0);
        const int32_t n = A.n;
        Fac R; run(R, A, options(LDLT_ORDER_AUTO, pivot));
        ASSERT_EQ(R.factor, LDLT_OK);
        std::vector<int32_t> perm(n);
        ldlt_get_perm(R.N, perm.data());
        auto sorted = perm;
        std::sort(sorted.begin(), sorted.end());
        for (int32_t i = 0; i < n; ++i) ASSERT_EQ(sorted[i], i);
        std::vector<double> d(n);
        ldlt_get_d(R.N, d.data(), nullptr, nullptr);

        std::vector<double> M(static_cast<size_t>(n) * n, 0.0);
        std::vector<int32_t> ip(n);
        for (int32_t k = 0; k < n; ++k) ip[perm[k]] = k;
        for (int32_t j = 0; j < n; ++j)
            for (int32_t p = A.Ap[j]; p < A.Ap[j + 1]; ++p) {
                int32_t a = ip[A.Ai[p]], b = ip[j];
                M[a + static_cast<size_t>(b) * n] = M[b + static_cast<size_t>(a) * n] = A.Ax[p];
            }
        for (int32_t k = 0; k < n; ++k) {
            double pivotv = M[k + static_cast<size_t>(k) * n];
            EXPECT_NEAR(d[perm[k]], pivotv, 1e-12 * std::abs(pivotv));
            for (int32_t j = k + 1; j < n; ++j) {
                double l = M[j + static_cast<size_t>(k) * n] / pivotv;
                if (l == 0.0) continue;
                for (int32_t i = k + 1; i < n; ++i)
                    M[i + static_cast<size_t>(j) * n] -= l * M[i + static_cast<size_t>(k) * n];
            }
        }
    }
}

/* The largest |A_ii / D_i| is what flags a nearly singular stiffness matrix. */
TEST(Api, MaxRatioFindsTheBadRow) {
    const double eps = 1e-10;
    /* Row 1's pivot is 1 + 1e-10 - 1 once row 0 is eliminated. */
    auto A = from_triplets(3, {{0, 0, 1.0}, {1, 0, 1.0}, {1, 1, 1.0 + eps}, {2, 2, 5.0}});
    Fac R; run(R, A, options(LDLT_ORDER_NATURAL));
    ASSERT_EQ(R.factor, LDLT_OK);
    auto I = info_of(R.N);
    EXPECT_EQ(I.max_ratio_row, 1);
    EXPECT_NEAR(I.max_ratio, (1.0 + eps) / eps, 1e6);
}

/* The schedule must not change the answer. */
TEST(Threads, ResultDoesNotDependOnThreadCount) {
    auto A = grid_laplacian(24, 22, 20, 0.0);
    auto Indef = grid_laplacian(24, 22, 20, 1.0);
    for (auto pivot : {LDLT_PIVOT_NONE, LDLT_PIVOT_THRESHOLD}) {
        const CSCLower &M = pivot == LDLT_PIVOT_NONE ? A : Indef;
        auto b = random_vector(M.n, 17);
        std::vector<double> ref;
        for (int32_t threads : {1, 2, 3, 8}) {
            auto opt = options(LDLT_ORDER_AUTO, pivot);
            opt.nthreads = threads;
            Fac R; run(R, M, opt);
            ASSERT_EQ(R.factor, LDLT_OK);
            /* Far above where a factorization stays on one thread, or this tests nothing. */
            ASSERT_GT(ldlt_flops(R.S), 1e8);
            auto x = b;
            ASSERT_EQ(ldlt_solve(R.N, 1, x.data(), M.n), LDLT_OK);
            if (ref.empty()) { ref = x; continue; }
            double diff = 0, size = 0;
            for (int32_t i = 0; i < M.n; ++i) {
                diff = std::max(diff, std::abs(x[i] - ref[i]));
                size = std::max(size, std::abs(ref[i]));
            }
            EXPECT_LT(diff, 1e-10 * size) << "threads " << threads;
        }
    }
}

TEST(Solve, ManyRightHandSidesMatchOneAtATime) {
    for (auto pivot : {LDLT_PIVOT_NONE, LDLT_PIVOT_THRESHOLD}) {
        auto A = grid_laplacian(12, 11, 10, pivot == LDLT_PIVOT_NONE ? 0.0 : 0.8);
        const int32_t n = A.n, nrhs = 7, ldb = n + 3;
        Fac R; run(R, A, options(LDLT_ORDER_AUTO, pivot));
        ASSERT_EQ(R.factor, LDLT_OK);
        std::vector<double> B(static_cast<size_t>(ldb) * nrhs, -123.0);
        std::vector<std::vector<double>> one(nrhs);
        for (int32_t r = 0; r < nrhs; ++r) {
            one[r] = random_vector(n, 40 + r);
            std::copy(one[r].begin(), one[r].end(), B.begin() + static_cast<size_t>(r) * ldb);
            ASSERT_EQ(ldlt_solve(R.N, 1, one[r].data(), n), LDLT_OK);
        }
        ASSERT_EQ(ldlt_solve(R.N, nrhs, B.data(), ldb), LDLT_OK);
        for (int32_t r = 0; r < nrhs; ++r) {
            for (int32_t i = 0; i < n; ++i)
                EXPECT_NEAR(B[i + static_cast<size_t>(r) * ldb], one[r][i], 1e-10 * (1.0 + std::abs(one[r][i])));
            /* The padding between columns is not touched. */
            for (int32_t i = n; i < ldb; ++i) EXPECT_EQ(B[i + static_cast<size_t>(r) * ldb], -123.0);
        }
    }
}

/* Only with the reference BLAS: OpenBLAS spins forever if it can't map a work buffer, so with
   it the library never gets the chance to report anything. */
#if defined(__linux__) && !LDLT_USE_SYSTEM_BLAS && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
/* Walks an address-space limit through the range where the library starts to run out, for
   one path. 0 means every run came back with a status, and some of each kind. */
static int walk_memory_limit(ldlt_pivot_strategy pivot) {
    auto A = grid_laplacian(20, 18, 16, pivot == LDLT_PIVOT_NONE ? 0.0 : 1.0);
    auto opt = options(LDLT_ORDER_AUTO, pivot);
    opt.nthreads = 4;
    /* First a full run, which starts the thread pool. */
    Fac Full; run(Full, A, opt);
    if (Full.factor != LDLT_OK) return 1;
    auto b = random_vector(A.n, 4);

    rlimit saved{};
    if (getrlimit(RLIMIT_AS, &saved) != 0) return 2;
    std::ifstream statm("/proc/self/statm");
    size_t pages = 0;
    statm >> pages;
    const size_t in_use = pages * static_cast<size_t>(sysconf(_SC_PAGESIZE));
    int failures = 0, successes = 0;
    for (size_t extra = 0; extra <= (size_t)1 << 30 && successes < 3; extra += (size_t)2 << 20) {
        rlimit lim = saved;
        lim.rlim_cur = in_use + extra;
        if (setrlimit(RLIMIT_AS, &lim) != 0) return 3;
        ldlt_symbolic *S = nullptr;
        ldlt_numeric *N = nullptr;
        ldlt_status sf = ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S);
        if (sf == LDLT_OK) sf = ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N);
        if (setrlimit(RLIMIT_AS, &saved) != 0) return 4;
        if (sf == LDLT_OK) {
            successes++;
            /* What survives a brush with the limit must still be right. */
            auto x = b;
            if (ldlt_solve(N, 1, x.data(), A.n) != LDLT_OK) return 5;
            if (!(backward_error(A, x, b) < 1e-11)) return 6;
        } else if (sf == LDLT_ERR_NOMEM) {
            failures++;
        } else {
            return 7;
        }
        ldlt_free_numeric(N);
        ldlt_free_symbolic(S);
    }
    if (failures == 0) return 8;
    if (successes == 0) return 9;
    return 0;
}

/* Running out of memory anywhere, including inside a worker thread, must come back as a status.
   In a process of its own, since memory an earlier test freed would otherwise still be mapped. */
TEST(Factor, OutOfMemoryComesBackAsAStatus) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(std::exit(walk_memory_limit(LDLT_PIVOT_NONE)), ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(std::exit(walk_memory_limit(LDLT_PIVOT_THRESHOLD)), ::testing::ExitedWithCode(0), "");
}
#endif

static std::string write_mm(const char *name, const std::string &text) {
    auto path = std::filesystem::temp_directory_path() / name;
    FILE *f = std::fopen(path.string().c_str(), "w");
    std::fputs(text.c_str(), f);
    std::fclose(f);
    return path.string();
}

TEST(MatrixMarket, GeneralFileKeepsOnlyItsLowerTriangle) {
    auto p = write_mm("ldlt_general.mtx",
        "%%MatrixMarket matrix coordinate real general\n3 3 5\n1 1 4\n1 2 7\n2 1 7\n2 2 5\n3 3 6\n");
    int32_t n, *Ap, *Ai; double *Ax;
    ASSERT_EQ(ldlt_read_mm_symmetric(p.c_str(), &n, &Ap, &Ai, &Ax), LDLT_OK);
    EXPECT_EQ(Ap[3], 4);
    EXPECT_EQ(Ai[1], 1);
    EXPECT_EQ(Ax[1], 7.0);  /* not 14, the two triangles' entries added */
    free(Ap); free(Ai); free(Ax);
}

TEST(MatrixMarket, SymmetricFileMayStoreEitherTriangle) {
    auto p = write_mm("ldlt_upper.mtx",
        "%%MatrixMarket matrix coordinate real symmetric\n% comment\n3 3 4\n3 3 6\n1 2 7\n2 2 5\n1 1 4\n");
    int32_t n, *Ap, *Ai; double *Ax;
    ASSERT_EQ(ldlt_read_mm_symmetric(p.c_str(), &n, &Ap, &Ai, &Ax), LDLT_OK);
    ASSERT_EQ(n, 3);
    std::vector<int32_t> ap(Ap, Ap + 4), ai(Ai, Ai + 4);
    std::vector<double> ax(Ax, Ax + 4);
    EXPECT_EQ(ap, (std::vector<int32_t>{0, 2, 3, 4}));
    EXPECT_EQ(ai, (std::vector<int32_t>{0, 1, 1, 2}));
    EXPECT_EQ(ax, (std::vector<double>{4, 7, 5, 6}));
    free(Ap); free(Ai); free(Ax);
}

TEST(MatrixMarket, RejectsAnOutOfRangeIndex) {
    auto p = write_mm("ldlt_bad.mtx", "%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n3 1 1.0\n");
    int32_t n, *Ap, *Ai; double *Ax;
    EXPECT_EQ(ldlt_read_mm_symmetric(p.c_str(), &n, &Ap, &Ai, &Ax), LDLT_ERR_INPUT);
}
