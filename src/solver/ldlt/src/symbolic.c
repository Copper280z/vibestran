/* Symbolic phase: ordering -> elimination tree -> column counts -> postorder ->
   supernodes -> per-supernode row lists and assembly-tree maps. The pattern of L is never
   formed: after the ordering, the work is close to linear in nnz(A) plus the total size
   of the fronts' row lists, which is far smaller than nnz(L). */

#include "internal.h"
#include <limits.h>

/* Allocations that live only for this phase, released together however it ends. */
typedef struct {
    void *p[24];
    int count;
} scratch;

static void *scratch_alloc(scratch *T, size_t count, size_t elem)
{
    if (T->count >= (int)(sizeof(T->p) / sizeof(T->p[0]))) return NULL;
    void *q = ldlt_alloc_array(count, elem);
    if (q) T->p[T->count++] = q;
    return q;
}

static void scratch_free(scratch *T)
{
    for (int i = 0; i < T->count; ++i) free(T->p[i]);
    T->count = 0;
}

/* Sort row indices. Most lists are short and arrive as a few sorted runs. */
static void sort_rows(int32_t *a, int32_t n, int32_t *tmp)
{
    if (n <= 256) {
        for (int32_t i = 1; i < n; ++i) {
            int32_t v = a[i], j = i - 1;
            while (j >= 0 && a[j] > v) { a[j+1] = a[j]; --j; }
            a[j+1] = v;
        }
        return;
    }
    enum { BITS = 11, BUCKETS = 1 << BITS };
    int32_t *from = a, *to = tmp;
    for (int shift = 0; shift < 33; shift += BITS) {
        int32_t pos[BUCKETS + 1];
        memset(pos, 0, sizeof(pos));
        for (int32_t i = 0; i < n; ++i)
            pos[(((uint32_t)from[i] >> shift) & (BUCKETS - 1)) + 1]++;
        for (int b = 0; b < BUCKETS; ++b) pos[b+1] += pos[b];
        for (int32_t i = 0; i < n; ++i)
            to[pos[((uint32_t)from[i] >> shift) & (BUCKETS - 1)]++] = from[i];
        int32_t *t = from; from = to; to = t;
    }
    /* Three passes: the sorted result is in tmp. */
    if (from != a) memcpy(a, from, (size_t)n * sizeof(int32_t));
}

/* Whether to merge a supernode into the one after it, which holds its parent. Sizes are
   full rectangles (rows x columns). */
static int relax_accept(int relax, int64_t width, int64_t merged_rows, int path_like,
                        int64_t cur_size, int64_t next_size, int64_t merged_size,
                        int64_t fund_size)
{
    int64_t added = merged_size - cur_size - next_size;
    int64_t fill_budget = (int64_t)relax * width;
    int64_t growth_num = (int64_t)(relax + 1);

    if (path_like) {
        fill_budget = width / 2;
        growth_num = 2;
    } else if (merged_rows > 4096) {
        fill_budget /= 2;
        growth_num = (growth_num + 1) / 2;
    }
    if (added > fill_budget) return 0;
    return merged_size <= growth_num * fund_size;
}

/* Fills sn_first (length nsuper + 1) and returns nsuper. Everything comes from the column
   counts: because the next supernode holds the parent of this one's last column, this
   one's rows are contained in the next one's columns and rows, so a merged supernode has
   exactly the next one's rows below it. */
static int32_t find_supernodes(int32_t n, const int32_t *parent, const int32_t *count,
                               const int32_t *nchild, int relax, int32_t *fund_first,
                               int32_t *sn_first)
{
    if (n == 0) { sn_first[0] = 0; return 0; }
    int32_t nfund = 0;
    fund_first[0] = 0;
    for (int32_t j = 1; j < n; ++j) {
        int same = parent[j-1] == j && count[j-1] == count[j] + 1;
        if (!same) fund_first[++nfund] = j;
    }
    fund_first[++nfund] = n;

    int32_t nsuper = 0;
    sn_first[0] = 0;
    int32_t cur_start = 0, cur_end = fund_first[1];
    const int64_t cur_rows = (int64_t)count[cur_end - 1] - 1;
    int64_t cur_size = (int64_t)(cur_end - cur_start) * (cur_rows + cur_end - cur_start);
    int64_t cur_fund_size = cur_size;
    for (int32_t f = 1; f < nfund; ++f) {
        int32_t next_start = fund_first[f], next_end = fund_first[f+1];
        int64_t next_rows = (int64_t)count[next_end - 1] - 1;
        int64_t next_size = (int64_t)(next_end - next_start) * (next_rows + next_end - next_start);
        int merged = 0;
        int32_t up = parent[cur_end - 1];
        if (relax > 0 && up >= next_start && up < next_end) {
            int64_t width = (int64_t)(next_end - cur_start);
            int64_t merged_size = width * (width + next_rows);
            int64_t merged_total = merged_size / (width > 0 ? width : 1);
            int path_like = width <= 16 && merged_total <= width + 4 &&
                            up == next_start && nchild[next_start] <= 1;
            int64_t fund_size = cur_fund_size + next_size;
            if (relax_accept(relax, width, merged_total, path_like, cur_size, next_size,
                             merged_size, fund_size)) {
                merged = 1;
                cur_end = next_end;
                cur_size = merged_size;
                cur_fund_size = fund_size;
            }
        }
        if (!merged) {
            sn_first[++nsuper] = next_start;
            cur_start = next_start;
            cur_end = next_end;
            cur_size = next_size;
            cur_fund_size = next_size;
        }
    }
    sn_first[++nsuper] = n;
    return nsuper;
}

ldlt_status ldlt_build_symbolic(int32_t n, const int32_t *Ap, const int32_t *Ai,
                                const ldlt_options *opt, ldlt_symbolic *S)
{
    ldlt_status st = LDLT_ERR_NOMEM;
    scratch T = {{0}, 0};
    int32_t *B0p = NULL, *B0i = NULL;
    S->n = n;

    size_t un = (size_t)n;
    int32_t *perm0 = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    int32_t *iperm0 = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    int32_t *parent0 = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    int32_t *post = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    int32_t *count0 = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    int32_t *count = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    int32_t *nchild = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    int32_t *inv_post = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    int32_t *fund_first = (int32_t*)scratch_alloc(&T, un + 1, sizeof(int32_t));
    int32_t *sn_first = (int32_t*)scratch_alloc(&T, un + 1, sizeof(int32_t));
    int32_t *mark = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    int32_t *sort_tmp = (int32_t*)scratch_alloc(&T, un, sizeof(int32_t));
    S->perm = (int32_t*)ldlt_alloc_array(un, sizeof(int32_t));
    S->iperm = (int32_t*)ldlt_alloc_array(un, sizeof(int32_t));
    S->parent = (int32_t*)ldlt_alloc_array(un, sizeof(int32_t));
    S->col_to_super = (int32_t*)ldlt_alloc_array(un, sizeof(int32_t));
    if (!perm0 || !iperm0 || !parent0 || !post || !count0 || !count || !nchild || !inv_post ||
        !fund_first || !sn_first || !mark || !sort_tmp ||
        !S->perm || !S->iperm || !S->parent || !S->col_to_super) goto done;

    st = ldlt_compute_ordering(n, Ap, Ai, opt, perm0, &S->ordering);
    if (st != LDLT_OK) goto done;
    for (int32_t i = 0; i < n; ++i) iperm0[perm0[i]] = i;

    /* The tree and the counts, on the ordered matrix. */
    st = ldlt_permute_lower_pattern(n, Ap, Ai, iperm0, &B0p, &B0i, NULL);
    if (st != LDLT_OK) goto done;
    st = ldlt_etree(n, B0p, B0i, parent0);
    if (st != LDLT_OK) goto done;
    st = ldlt_postorder(n, parent0, NULL, post);
    if (st != LDLT_OK) goto done;
    st = ldlt_col_counts(n, B0p, B0i, parent0, post, count0);
    if (st != LDLT_OK) goto done;
    free(B0p); free(B0i); B0p = B0i = NULL;

    /* The final numbering: a postorder of the same tree, with each node's heaviest child
       last. A postorder does not change the fill or the counts. */
    st = ldlt_postorder(n, parent0, count0, post);
    if (st != LDLT_OK) goto done;
    for (int32_t k = 0; k < n; ++k) inv_post[post[k]] = k;
    for (int32_t k = 0; k < n; ++k) {
        int32_t j = post[k];
        S->perm[k] = perm0[j];
        S->iperm[perm0[j]] = k;
        S->parent[k] = parent0[j] < 0 ? -1 : inv_post[parent0[j]];
        count[k] = count0[j];
    }
    const int32_t *parent = S->parent;
    S->perm_identity = 1;
    for (int32_t i = 0; i < n; ++i)
        if (S->perm[i] != i) { S->perm_identity = 0; break; }

    st = ldlt_permute_lower_pattern(n, Ap, Ai, S->iperm, &S->Bp, &S->Bi, &S->Bmap);
    if (st != LDLT_OK) goto done;
    st = LDLT_ERR_NOMEM;
    S->nnz = Ap[n];
    S->Bdiag = (int32_t*)ldlt_alloc_array(un, sizeof(int32_t));
    if (!S->Bdiag) goto done;
    for (int32_t j = 0; j < n; ++j) {
        S->Bdiag[j] = -1;
        for (int32_t p = S->Bp[j]; p < S->Bp[j+1]; ++p)
            if (S->Bi[p] == j) S->Bdiag[j] = p;
    }

    for (int32_t i = 0; i < n; ++i) nchild[i] = 0;
    for (int32_t i = 0; i < n; ++i) if (parent[i] != -1) nchild[parent[i]]++;
    int relax = (opt && opt->relax_supernodes > 0) ? opt->relax_supernodes : 0;
    int32_t nsuper = find_supernodes(n, parent, count, nchild, relax, fund_first, sn_first);
    S->nsuper = nsuper;

    size_t uns = (size_t)nsuper;
    S->super = (ldlt_super*)ldlt_calloc_array(uns, sizeof(ldlt_super));
    S->panel_off = (int64_t*)ldlt_alloc_array(uns + 1, sizeof(int64_t));
    S->super_parent = (int32_t*)ldlt_alloc_array(uns, sizeof(int32_t));
    S->child_ptr = (int32_t*)ldlt_calloc_array(uns + 1, sizeof(int32_t));
    S->child_idx = (int32_t*)ldlt_alloc_array(uns, sizeof(int32_t));
    int32_t *lvl = (int32_t*)scratch_alloc(&T, uns, sizeof(int32_t));
    int32_t *width_at = (int32_t*)scratch_alloc(&T, uns + 1, sizeof(int32_t));
    if (!S->super || !S->panel_off || !S->super_parent || !S->child_ptr ||
        !S->child_idx || !lvl || !width_at) goto done;

    /* Sizes. The last column's count gives the rows below the merged supernode. */
    int64_t pool = 0, lnz = 0;
    double flops = 0.0;
    S->panel_off[0] = 0;
    for (int32_t s = 0; s < nsuper; ++s) {
        int32_t fc = sn_first[s], end = sn_first[s+1];
        int64_t k = end - fc;
        int64_t m = (int64_t)count[end - 1] - 1;
        for (int32_t j = fc; j < end; ++j) S->col_to_super[j] = s;
        S->super[s].first_col = fc;
        S->super[s].width = (int32_t)k;
        S->super[s].nrows_below = (int32_t)m;
        S->super[s].row_off = pool;
        pool += m;
        S->panel_off[s+1] = S->panel_off[s] + (k + m) * k;
        lnz += k * (k + 1) / 2 + m * k;
        double R = (double)(k + m), K = (double)k;
        flops += K*R*R - K*K*R + K*K*K/3.0;
    }
    S->rows_pool_len = pool;
    S->lnz = lnz;
    S->flops = flops;

    /* The assembly tree. */
    for (int32_t s = 0; s < nsuper; ++s) {
        int32_t p = parent[sn_first[s+1] - 1];
        int32_t sp = p == -1 ? -1 : S->col_to_super[p];
        S->super_parent[s] = sp;
        if (sp >= 0) S->child_ptr[sp+1]++;
    }
    for (int32_t s = 0; s < nsuper; ++s) S->child_ptr[s+1] += S->child_ptr[s];
    {
        int32_t *cur = (int32_t*)scratch_alloc(&T, uns, sizeof(int32_t));
        if (!cur) goto done;
        if (nsuper > 0) memcpy(cur, S->child_ptr, uns * sizeof(int32_t));
        for (int32_t s = 0; s < nsuper; ++s) {
            int32_t p = S->super_parent[s];
            if (p >= 0) S->child_idx[cur[p]++] = s;
        }
    }

    int32_t maxlvl = 0;
    for (int32_t s = 0; s < nsuper; ++s) lvl[s] = 0;
    for (int32_t s = 0; s < nsuper; ++s) {
        int32_t p = S->super_parent[s];
        if (p >= 0 && lvl[p] < lvl[s] + 1) {
            lvl[p] = lvl[s] + 1;
            if (lvl[p] > maxlvl) maxlvl = lvl[p];
        }
    }
    S->nlevels = nsuper > 0 ? maxlvl + 1 : 0;
    for (int32_t i = 0; i <= nsuper; ++i) width_at[i] = 0;
    S->max_level_width = 0;
    for (int32_t s = 0; s < nsuper; ++s)
        if (++width_at[lvl[s]] > S->max_level_width) S->max_level_width = width_at[lvl[s]];

    /* Row lists: each supernode's rows are those of its columns in A and those of its
       children, less its own columns. Children come first in this numbering. */
    S->rows_pool = (int32_t*)ldlt_alloc_array((size_t)pool, sizeof(int32_t));
    S->edge_pos = (int32_t*)ldlt_alloc_array((size_t)pool, sizeof(int32_t));
    if (!S->rows_pool || !S->edge_pos) goto done;
    for (int32_t i = 0; i < n; ++i) mark[i] = -1;
    for (int32_t s = 0; s < nsuper; ++s) {
        const ldlt_super *Ss = &S->super[s];
        int32_t fc = Ss->first_col, end = fc + Ss->width;
        int32_t *rows = S->rows_pool + Ss->row_off;
        int64_t m = 0;
        int64_t cap = Ss->nrows_below;
        for (int32_t c = fc; c < end; ++c) {
            for (int32_t p = S->Bp[c]; p < S->Bp[c+1]; ++p) {
                int32_t r = S->Bi[p];
                if (r >= end && mark[r] != s) {
                    if (m == cap) { st = LDLT_ERR_INPUT; goto done; }
                    mark[r] = s;
                    rows[m++] = r;
                }
            }
        }
        for (int32_t cp = S->child_ptr[s]; cp < S->child_ptr[s+1]; ++cp) {
            const ldlt_super *Sc = &S->super[S->child_idx[cp]];
            const int32_t *crows = S->rows_pool + Sc->row_off;
            for (int32_t t = 0; t < Sc->nrows_below; ++t) {
                int32_t r = crows[t];
                if (r >= end && mark[r] != s) {
                    if (m == cap) { st = LDLT_ERR_INPUT; goto done; }
                    mark[r] = s;
                    rows[m++] = r;
                }
            }
        }
        if (m != cap) { st = LDLT_ERR_INPUT; goto done; }
        sort_rows(rows, (int32_t)m, sort_tmp);
    }

    /* Each child row's position in its parent's front. Both lists are sorted. */
    for (int32_t c = 0; c < nsuper; ++c) {
        int32_t p = S->super_parent[c];
        const ldlt_super *Sc = &S->super[c];
        if (p < 0) continue;
        const ldlt_super *Sp = &S->super[p];
        const int32_t *crows = S->rows_pool + Sc->row_off;
        const int32_t *prows = S->rows_pool + Sp->row_off;
        int32_t *emap = S->edge_pos + Sc->row_off;
        int32_t pfc = Sp->first_col, pk = Sp->width, pm = Sp->nrows_below;
        int32_t q = 0;
        for (int32_t t = 0; t < Sc->nrows_below; ++t) {
            int32_t r = crows[t];
            if (r < pfc + pk) {
                emap[t] = r - pfc;
            } else {
                while (q < pm && prows[q] < r) ++q;
                if (q == pm || prows[q] != r) { st = LDLT_ERR_INPUT; goto done; }
                emap[t] = pk + q;
            }
        }
    }
    st = LDLT_OK;

done:
    free(B0p); free(B0i);
    scratch_free(&T);
    return st;
}

/* The input contract: lower triangle in CSC, strictly increasing rows in each column. */
static int valid_lower_csc(int32_t n, const int32_t *Ap, const int32_t *Ai)
{
    if (Ap[0] != 0) return 0;
    for (int32_t j = 0; j < n; ++j) {
        if (Ap[j+1] < Ap[j]) return 0;
        int32_t prev = j - 1;
        for (int32_t p = Ap[j]; p < Ap[j+1]; ++p) {
            int32_t i = Ai[p];
            if (i <= prev || i >= n) return 0;
            prev = i;
        }
    }
    return 1;
}

ldlt_status ldlt_analyze(int32_t n, const int32_t *Ap, const int32_t *Ai,
                         const ldlt_options *opt, ldlt_symbolic **out)
{
    if (!out) return LDLT_ERR_INPUT;
    *out = NULL;
    if (n < 0 || !Ap || (n > 0 && !Ai)) return LDLT_ERR_INPUT;
    if (!valid_lower_csc(n, Ap, Ai)) return LDLT_ERR_INPUT;
    ldlt_options local; ldlt_options_default(&local);
    if (opt) local = *opt;
    if (local.pivot != LDLT_PIVOT_NONE &&
        local.pivot != LDLT_PIVOT_BUNCH_KAUFMAN)
        return LDLT_ERR_UNSUPPORTED;

    ldlt_symbolic *S = (ldlt_symbolic*)ldlt_calloc_array(1, sizeof(*S));
    if (!S) return LDLT_ERR_NOMEM;
    ldlt_status st = ldlt_build_symbolic(n, Ap, Ai, &local, S);
    if (st != LDLT_OK) {
        ldlt_free_symbolic(S);
        return st;
    }
    *out = S;
    return LDLT_OK;
}
