/* Dense kernels for one front: factor an R x k panel into unit lower L and D, and subtract
   L21 D L21^T from the m x m trailing block C (m = R - k). Only lower triangles are read
   or written. A pivot that is exactly zero is held: its D is +inf and its column of L is
   zero, so it updates nothing and the solve returns exactly zero for that unknown. That is
   the same as deleting the row and column, or an SPC on that unknown. */

#include "internal.h"
#include "blas_shim.h"
#include <math.h>

#define PIVOT_ZERO 1e-300
#define SMALL_K 8     /* below this, a plain loop beats the BLAS call overhead */
#define SMALL_M 48
#define BLOCK 48
#define SIGNED_BATCH 256
#define PARALLEL_ENTRIES 2000000  /* entries touched before a plain loop is worth threading */

#if LDLT_HAVE_LAPACK
extern void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
#endif

/* Grows by half again at least: every new block means page faults on all of it. */
double *ldlt_ws_doubles(double **p, size_t *cap, size_t need)
{
    if (*cap < need) {
        size_t want = *cap + *cap / 2;
        if (want < need) want = need;
        free(*p);
        *p = (double*)ldlt_alloc_array(want, sizeof(double));
        *cap = *p ? want : 0;
    }
    return *p;
}

void ldlt_ws_free(ldlt_thread_ws *ws)
{
    free(ws->rowmap); free(ws->C); free(ws->save); free(ws->batch); free(ws->runs);
    free(ws->pos); free(ws->stack);
    memset(ws, 0, sizeof(*ws));
}

/* Unblocked, right-looking. */
static ldlt_status ldl_unblocked(double *P, int32_t ld, int32_t R, int32_t k, double *D,
                                 int require_pd)
{
    for (int32_t j = 0; j < k; ++j) {
        double *cj = P + (size_t)j*ld;
        double dj = cj[j];
        if (!isfinite(dj)) return LDLT_ERR_NUMERIC;
        if (fabs(dj) < PIVOT_ZERO) {
            if (require_pd) return LDLT_ERR_INDEFINITE;
            D[j] = INFINITY;
            memset(cj + j, 0, (size_t)(R - j) * sizeof(double));
            continue;
        }
        if (dj < 0.0 && require_pd) return LDLT_ERR_INDEFINITE;
        D[j] = dj;
        double inv = 1.0 / dj;
        for (int32_t i = j + 1; i < R; ++i) cj[i] *= inv;
        for (int32_t c = j + 1; c < k; ++c) {
            double *cc = P + (size_t)c*ld;
            double scale = dj * cj[c];
            if (scale == 0.0) continue;
            for (int32_t i = c; i < R; ++i) cc[i] -= cj[i] * scale;
        }
        cj[j] = 0.0;
    }
    return LDLT_OK;
}

/* Blocked: scalar on the diagonal block, BLAS on the rows below and the rest of the panel. */
static ldlt_status ldl_blocked(double *P, int32_t ld, int32_t R, int32_t k, double *D,
                               int require_pd, ldlt_thread_ws *ws)
{
    double *W = NULL;
    if (k > BLOCK) {
        W = ldlt_ws_doubles(&ws->batch, &ws->batch_cap, (size_t)k * BLOCK);
        if (!W) return LDLT_ERR_NOMEM;
    }
    for (int32_t b = 0; b < k; b += BLOCK) {
        int32_t jb = k - b < BLOCK ? k - b : BLOCK;
        double *Pb = P + b + (size_t)b*ld;
        ldlt_status st = ldl_unblocked(Pb, ld, jb, jb, D + b, require_pd);
        if (st != LDLT_OK) return st;
        int32_t below = R - b - jb, tc = k - b - jb;
        if (below <= 0) continue;
        double *A21 = Pb + jb;
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                    below, jb, 1.0, Pb, ld, A21, ld);
        /* A21 now holds L21 D. Keep the part the rest of the panel needs, then divide. */
        for (int32_t c = 0; c < jb; ++c) {
            double *col = A21 + (size_t)c*ld;
            double d = D[b + c];
            if (isinf(d)) {
                memset(col, 0, (size_t)below * sizeof(double));
                if (tc > 0) memset(W + (size_t)c*tc, 0, (size_t)tc * sizeof(double));
                continue;
            }
            if (tc > 0) memcpy(W + (size_t)c*tc, col, (size_t)tc * sizeof(double));
            double inv = 1.0 / d;
            for (int32_t i = 0; i < below; ++i) col[i] *= inv;
        }
        if (tc > 0) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, below, tc, jb,
                        -1.0, A21, ld, W, tc, 1.0, A21 + (size_t)jb*ld, ld);
        }
    }
    return LDLT_OK;
}

/* C -= L21 D L21^T for small fronts. Four columns of L at a time, so each column of C is
   loaded and stored once per group. */
static void update_small(double *restrict C, int32_t ldC, int32_t m, int32_t k,
                         const double *restrict L21, int32_t ld, const double *D)
{
    for (int32_t j = 0; j < m; ++j) {
        double *restrict cj = C + (size_t)j*ldC;
        int32_t c = 0;
        for (; c + 4 <= k; c += 4) {
            const double *l0 = L21 + (size_t)c*ld, *l1 = l0 + ld, *l2 = l1 + ld, *l3 = l2 + ld;
            double a0 = isinf(D[c])   ? 0.0 : l0[j] * D[c];
            double a1 = isinf(D[c+1]) ? 0.0 : l1[j] * D[c+1];
            double a2 = isinf(D[c+2]) ? 0.0 : l2[j] * D[c+2];
            double a3 = isinf(D[c+3]) ? 0.0 : l3[j] * D[c+3];
            for (int32_t i = j; i < m; ++i)
                cj[i] -= l0[i]*a0 + l1[i]*a1 + l2[i]*a2 + l3[i]*a3;
        }
        for (; c < k; ++c) {
            const double *l0 = L21 + (size_t)c*ld;
            double a0 = isinf(D[c]) ? 0.0 : l0[j] * D[c];
            for (int32_t i = j; i < m; ++i) cj[i] -= l0[i]*a0;
        }
    }
}

/* C -= sum over c of d_c l_c l_c^T through dsyrk, positive and negative columns in separate
   batches. dsyrk does half the flops of a dgemm for the same update. */
ldlt_status ldlt_update_signed(double *C, int32_t ldC, int32_t m, int32_t k,
                               const double *L21, int32_t ld, const double *D,
                               ldlt_thread_ws *ws)
{
    if (m <= 0 || k <= 0) return LDLT_OK;
    if (k <= SMALL_K && m <= SMALL_M) {
        update_small(C, ldC, m, k, L21, ld, D);
        return LDLT_OK;
    }
    int32_t cap = k < SIGNED_BATCH ? k : SIGNED_BATCH;
    double *W = ldlt_ws_doubles(&ws->batch, &ws->batch_cap, (size_t)m * cap);
    if (!W) return LDLT_ERR_NOMEM;
    for (int sign = 1; sign >= -1; sign -= 2) {
        int32_t used = 0;
        for (int32_t c = 0; c < k; ++c) {
            double d = D[c];
            if (isinf(d) || (sign > 0 ? d <= 0.0 : d >= 0.0)) continue;
            double s = sqrt(fabs(d));
            const double *lc = L21 + (size_t)c*ld;
            double *wc = W + (size_t)used*m;
            for (int32_t i = 0; i < m; ++i) wc[i] = lc[i] * s;
            if (++used == cap) {
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, m, used,
                            -(double)sign, W, m, 1.0, C, ldC);
                used = 0;
            }
        }
        if (used > 0) {
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, m, used,
                        -(double)sign, W, m, 1.0, C, ldC);
        }
    }
    return LDLT_OK;
}

/* From a Cholesky factor in the k x k block and L21 scaled by it, unit lower L and D. */
void ldlt_cholesky_to_ldl(double *P, int32_t ld, int32_t R, int32_t k, double *D, int nth)
{
    int big = nth > 1 && (int64_t)R * k >= PARALLEL_ENTRIES;
    (void)big; (void)nth;
#if LDLT_HAVE_OPENMP
    #pragma omp parallel for if(big) num_threads(nth) schedule(static)
#endif
    for (int32_t c = 0; c < k; ++c) {
        double *col = P + (size_t)c*ld;
        double l = col[c];
        double inv = 1.0 / l;
        D[c] = l * l;
        for (int32_t i = c + 1; i < R; ++i) col[i] *= inv;
        col[c] = 0.0;
    }
}

#if LDLT_HAVE_LAPACK
/* Runs LAPACK's Cholesky on the k x k block. On failure the block is restored. */
int ldlt_try_potrf(double *P, int32_t ld, int32_t k, double *save)
{
    for (int32_t j = 0; j < k; ++j)
        memcpy(save + (size_t)j*k + j, P + (size_t)j*ld + j, (size_t)(k - j) * sizeof(double));
    char uplo = 'L';
    int nk = k, lda = ld, info = 0;
    dpotrf_(&uplo, &nk, P, &lda, &info);
    if (info == 0) return 1;
    for (int32_t j = 0; j < k; ++j)
        memcpy(P + (size_t)j*ld + j, save + (size_t)j*k + j, (size_t)(k - j) * sizeof(double));
    return 0;
}
#endif

ldlt_status ldlt_front_nopiv(double *P, int32_t R, int32_t k, double *D,
                             double *C, int32_t ldC, int require_pd, ldlt_thread_ws *ws)
{
    int32_t m = R - k;
    if (k <= SMALL_K) {
        ldlt_status st = ldl_unblocked(P, R, R, k, D, require_pd);
        if (st != LDLT_OK) return st;
        return ldlt_update_signed(C, ldC, m, k, P + k, R, D, ws);
    }
#if LDLT_HAVE_LAPACK
    double *save = ldlt_ws_doubles(&ws->save, &ws->save_cap, (size_t)k * k);
    if (!save) return LDLT_ERR_NOMEM;
    if (ldlt_try_potrf(P, R, k, save)) {
        if (m > 0) {
            cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                        m, k, 1.0, P, R, P + k, R);
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                        m, k, -1.0, P + k, R, 1.0, C, ldC);
        }
        ldlt_cholesky_to_ldl(P, R, R, k, D, ws->nth);
        return LDLT_OK;
    }
    if (require_pd) return LDLT_ERR_INDEFINITE;
#endif
    ldlt_status st = ldl_blocked(P, R, R, k, D, require_pd, ws);
    if (st != LDLT_OK) return st;
    return ldlt_update_signed(C, ldC, m, k, P + k, R, D, ws);
}
