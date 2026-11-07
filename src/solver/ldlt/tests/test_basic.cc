#include <gtest/gtest.h>
#include "test_helpers.h"
extern "C" {
#include "../src/internal.h"
}

TEST(Etree, LowerCscUsesRowOrder) {
    /* Lower triangle for edges (2,0), (2,1), (3,1).  Davis's symmetric etree
       algorithm must see row 2's columns together before row 3, yielding 0->2,
       1->2, 2->3. */
    const int32_t Ap[] = {0, 2, 5, 6, 7};
    const int32_t Ai[] = {
        0, 2,
        1, 2, 3,
        2,
        3,
    };
    int32_t parent[4] = {-2, -2, -2, -2};

    ASSERT_EQ(ldlt_etree(4, Ap, Ai, parent), LDLT_OK);
    EXPECT_EQ(parent[0], 2);
    EXPECT_EQ(parent[1], 2);
    EXPECT_EQ(parent[2], 3);
    EXPECT_EQ(parent[3], -1);
}

TEST(Analyze, Tridiag5) {
    auto A = make_tridiag_spd(5);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_NATURAL;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    EXPECT_EQ(ldlt_n(S), 5);
    EXPECT_GT(ldlt_nsuper(S), 0);
    ldlt_free_symbolic(S);
}

TEST(Analyze, AssemblyTreeEdgeMaps) {
    auto A = make_tridiag_spd(8);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_NATURAL;
    opt.relax_supernodes = 0;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ASSERT_GT(S->nsuper, 1);

    std::vector<int32_t> rowmap(A.n, -1);
    for (int32_t p = 0; p < S->nsuper; ++p) {
        const ldlt_super &Sp = S->super[p];
        const int32_t *prb = S->rows_pool + Sp.row_off;
        for (int32_t i = 0; i < Sp.width; ++i) rowmap[Sp.first_col + i] = i;
        for (int32_t i = 0; i < Sp.nrows_below; ++i) rowmap[prb[i]] = Sp.width + i;

        for (int32_t cp = S->child_ptr[p]; cp < S->child_ptr[p+1]; ++cp) {
            int32_t c = S->child_idx[cp];
            const ldlt_super &Sc = S->super[c];
            const int32_t *crb = S->rows_pool + Sc.row_off;
            const int32_t *emap = S->edge_pos + Sc.row_off;
            EXPECT_EQ(S->super_parent[c], p);
            for (int32_t i = 0; i < Sc.nrows_below; ++i) {
                ASSERT_GE(emap[i], 0);
                ASSERT_LT(emap[i], Sp.width + Sp.nrows_below);
                EXPECT_EQ(rowmap[crb[i]], emap[i]);
            }
        }

        for (int32_t i = 0; i < Sp.width; ++i) rowmap[Sp.first_col + i] = -1;
        for (int32_t i = 0; i < Sp.nrows_below; ++i) rowmap[prb[i]] = -1;
    }
    ldlt_free_symbolic(S);
}

TEST(Factor, Tridiag10) {
    auto A = make_tridiag_spd(10);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.ordering = LDLT_ORDER_NATURAL;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    ASSERT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);
    ldlt_free_numeric(N);
    ldlt_free_symbolic(S);
}

TEST(Factor, AcceptsPivotRequest) {
    auto A = make_tridiag_spd(4);
    ldlt_options opt; ldlt_options_default(&opt);
    opt.pivot = LDLT_PIVOT_BUNCH_KAUFMAN;
    ldlt_symbolic *S = nullptr;
    ASSERT_EQ(ldlt_analyze(A.n, A.Ap.data(), A.Ai.data(), &opt, &S), LDLT_OK);
    ldlt_numeric *N = nullptr;
    EXPECT_EQ(ldlt_factorize(S, A.Ap.data(), A.Ai.data(), A.Ax.data(), &opt, &N), LDLT_OK);
    ldlt_free_numeric(N);
    ldlt_free_symbolic(S);
}
