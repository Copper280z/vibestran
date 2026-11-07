/* Multifrontal LDL^T with threshold pivoting, for indefinite matrices.

   The matrix is first scaled symmetrically so its entries are about 1. In a front, a pivot
   must then keep every entry of L below 1/u, with u far smaller than Bunch-Kaufman's 0.64,
   since a sparse factorization has somewhere better to send a bad pivot than a dense one:
   a column that has no acceptable pivot in this front is delayed, which makes it a fully
   summed column of the parent's front, where more of the matrix has been added to it.

   Most columns of most matrices need no pivoting at all, so the fully summed columns go
   through in blocks, each tried first without pivoting and checked afterwards. A block that
   fails that takes its pivots one at a time from among a few columns, kept up to date, and
   the columns with no acceptable pivot move behind the rest to be tried again later. The
   rest of the fully summed columns are updated when each block ends, through the BLAS, and
   the contribution block once, when the front is finished. */

#include "internal.h"
#include "blas_shim.h"
#include <math.h>

#define PIVOT_SMALL 1e-20       /* of a scaled matrix, whose entries are about 1 */
#define PIVOT_BLOCK 32
#define TRY_BLOCK_MAX 128
#define TRY_BLOCK_MIN 16
#define SIGNED_BATCH 256
#define RUIZ_PASSES 3

#if LDLT_HAVE_LAPACK
extern void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
#endif

typedef struct {
    int32_t ndelay;
    int32_t nc;        /* ndelay + rows below */
    int32_t *delayed;  /* ndelay permuted-space ids */
    double *A;         /* packed lower: delayed first, then the rows below */
} pivot_contrib;

/* Symmetric equilibration (Ruiz): repeatedly divide each row and column by the square root
   of its largest entry. Scales B in place. A row of zeros keeps scale 1. */
static ldlt_status ruiz_scale(const ldlt_symbolic *S, double *Bx, double *scale)
{
    int32_t n = S->n;
    double *rmax = (double*)ldlt_alloc_array((size_t)n, sizeof(double));
    if (!rmax) return LDLT_ERR_NOMEM;
    for (int32_t i = 0; i < n; ++i) scale[i] = 1.0;
    for (int32_t pass = 0; pass < RUIZ_PASSES; ++pass) {
        for (int32_t i = 0; i < n; ++i) rmax[i] = 0.0;
        for (int32_t j = 0; j < n; ++j) {
            for (int32_t p = S->Bp[j]; p < S->Bp[j+1]; ++p) {
                double a = fabs(Bx[p]);
                if (!isfinite(a)) { free(rmax); return LDLT_ERR_NUMERIC; }
                int32_t i = S->Bi[p];
                if (a > rmax[i]) rmax[i] = a;
                if (a > rmax[j]) rmax[j] = a;
            }
        }
        for (int32_t i = 0; i < n; ++i)
            rmax[i] = rmax[i] > 0.0 ? 1.0 / sqrt(rmax[i]) : 1.0;
        for (int32_t j = 0; j < n; ++j) {
            for (int32_t p = S->Bp[j]; p < S->Bp[j+1]; ++p)
                Bx[p] *= rmax[S->Bi[p]] * rmax[j];
            scale[j] *= rmax[j];
        }
    }
    free(rmax);
    return LDLT_OK;
}

static void swap_d(double *a, double *b) { double t = *a; *a = *b; *b = t; }

/* Exchanges front positions a < b in a lower-stored symmetric front, including their rows
   in the columns already eliminated. */
static void sym_swap(double *F, int32_t R, int32_t *vars, int32_t a, int32_t b)
{
    if (a == b) return;
    int32_t tv = vars[a]; vars[a] = vars[b]; vars[b] = tv;
    for (int32_t c = 0; c < a; ++c) swap_d(F + a + (size_t)c*R, F + b + (size_t)c*R);
    swap_d(F + a + (size_t)a*R, F + b + (size_t)b*R);
    for (int32_t i = a + 1; i < b; ++i) swap_d(F + i + (size_t)a*R, F + b + (size_t)i*R);
    for (int32_t i = b + 1; i < R; ++i) swap_d(F + i + (size_t)a*R, F + i + (size_t)b*R);
}

/* Largest |F(i, j)| over rows i >= e, leaving out rows j and skip. */
static double col_max(const double *F, int32_t R, int32_t e, int32_t j, int32_t skip)
{
    double mx = 0.0;
    for (int32_t i = e; i < j; ++i) {
        if (i == skip) continue;
        double a = fabs(F[j + (size_t)i*R]);
        if (a > mx) mx = a;
    }
    const double *cj = F + (size_t)j*R;
    int32_t lo = j + 1;
    if (skip > j) {
        for (int32_t i = lo; i < skip; ++i) { double a = fabs(cj[i]); if (a > mx) mx = a; }
        lo = skip + 1;
    }
    for (int32_t i = lo; i < R; ++i) { double a = fabs(cj[i]); if (a > mx) mx = a; }
    return mx;
}

/* Position in [e, be) of the largest off-diagonal entry of column j among the candidates. */
static int32_t best_partner(const double *F, int32_t R, int32_t e, int32_t be, int32_t j)
{
    double mx = 0.0;
    int32_t q = -1;
    for (int32_t i = e; i < j; ++i) {
        double a = fabs(F[j + (size_t)i*R]);
        if (a > mx) { mx = a; q = i; }
    }
    for (int32_t i = j + 1; i < be; ++i) {
        double a = fabs(F[i + (size_t)j*R]);
        if (a > mx) { mx = a; q = i; }
    }
    return q;
}

/* A pivot from the candidates [e, be), or *size = -1. Size 0 means a negligible column. */
static ldlt_status choose_pivot(const double *F, int32_t R, int32_t e, int32_t be, double u,
                                int32_t *p, int32_t *q, int32_t *size)
{
    *size = -1;
    for (int32_t j = e; j < be; ++j) {
        double ajj = F[j + (size_t)j*R];
        if (!isfinite(ajj)) return LDLT_ERR_NUMERIC;
        double cmax = col_max(F, R, e, j, -1);
        if (fabs(ajj) < PIVOT_SMALL && cmax < PIVOT_SMALL) {
            *p = j; *size = 0;
            return LDLT_OK;
        }
        if (fabs(ajj) >= PIVOT_SMALL && fabs(ajj) >= u * cmax) {
            *p = j; *size = 1;
            return LDLT_OK;
        }
        int32_t k = best_partner(F, R, e, be, j);
        if (k < 0) continue;
        double akk = F[k + (size_t)k*R];
        double ajk = j < k ? F[k + (size_t)j*R] : F[j + (size_t)k*R];
        double det = ajj * akk - ajk * ajk;
        if (fabs(det) < PIVOT_SMALL * ajk * ajk || fabs(det) == 0.0) continue;
        double mj = col_max(F, R, e, j, k);
        double mk = col_max(F, R, e, k, j);
        double limit = fabs(det) / u;
        if (fabs(akk)*mj + fabs(ajk)*mk <= limit && fabs(ajk)*mj + fabs(ajj)*mk <= limit) {
            *p = j < k ? j : k;
            *q = j < k ? k : j;
            *size = 2;
            return LDLT_OK;
        }
    }
    return LDLT_OK;
}

static void hold_pivot(double *F, int32_t R, int32_t e, double *d, double *off)
{
    memset(F + e + (size_t)e*R, 0, (size_t)(R - e) * sizeof(double));
    d[e] = INFINITY;
    off[e] = 0.0;
}

/* Pivot at e: turns column e into L and updates columns up to be. */
static void apply_1x1(double *F, int32_t R, int32_t e, int32_t be, double *d, double *off)
{
    double *ce = F + (size_t)e*R;
    double dj = ce[e];
    double inv = 1.0 / dj;
    d[e] = dj;
    off[e] = 0.0;
    for (int32_t i = e + 1; i < R; ++i) ce[i] *= inv;
    for (int32_t c = e + 1; c < be; ++c) {
        double *cc = F + (size_t)c*R;
        double w = ce[c] * dj;
        if (w == 0.0) continue;
        for (int32_t i = c; i < R; ++i) cc[i] -= ce[i] * w;
    }
    ce[e] = 0.0;
}

static void apply_2x2(double *F, int32_t R, int32_t e, int32_t be, double *d, double *off)
{
    double *c0 = F + (size_t)e*R, *c1 = c0 + R;
    double a = c0[e], b = c0[e+1], c = c1[e+1];
    double det = a * c - b * b;
    d[e] = a; d[e+1] = c;
    off[e] = b; off[e+1] = 0.0;
    double i00 = c / det, i01 = -b / det, i11 = a / det;
    for (int32_t i = e + 2; i < R; ++i) {
        double x = c0[i], y = c1[i];
        c0[i] = i00 * x + i01 * y;
        c1[i] = i01 * x + i11 * y;
    }
    for (int32_t k = e + 2; k < be; ++k) {
        double *ck = F + (size_t)k*R;
        double w0 = a * c0[k] + b * c1[k];
        double w1 = b * c0[k] + c * c1[k];
        for (int32_t i = k; i < R; ++i) ck[i] -= c0[i] * w0 + c1[i] * w1;
    }
    c0[e] = 0.0; c0[e+1] = 0.0; c1[e+1] = 0.0;
}

/* Applies the pivots in [bs, el) to the fully summed columns [be, nfs). */
static ldlt_status update_trailing(double *F, int32_t R, int32_t bs, int32_t el, int32_t be,
                                   int32_t nfs, const double *d, const double *off,
                                   ldlt_thread_ws *ws)
{
    int32_t np = el - bs, nt = nfs - be;
    if (np <= 0 || nt <= 0) return LDLT_OK;
    double *W = ldlt_ws_doubles(&ws->batch, &ws->batch_cap, (size_t)np * nt);
    if (!W) return LDLT_ERR_NOMEM;
    for (int32_t c = bs; c < el; ++c) {
        const double *lc = F + be + (size_t)c*R;
        double *wc = W + (size_t)(c - bs)*nt;
        if (off[c] != 0.0) {
            const double *ln = lc + R;
            double *wn = wc + nt;
            double a = d[c], b = off[c], g = d[c+1];
            for (int32_t t = 0; t < nt; ++t) {
                wc[t] = a * lc[t] + b * ln[t];
                wn[t] = b * lc[t] + g * ln[t];
            }
            ++c;
        } else if (isinf(d[c])) {
            memset(wc, 0, (size_t)nt * sizeof(double));
        } else {
            for (int32_t t = 0; t < nt; ++t) wc[t] = d[c] * lc[t];
        }
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, R - be, nt, np,
                -1.0, F + be + (size_t)bs*R, R, W, nt,
                1.0, F + be + (size_t)be*R, R);
    return LDLT_OK;
}

/* Subtracts L21 D L21^T from the contribution block at the end of the front. A 2 x 2 pivot
   is split by its eigendecomposition into two signed rank-one terms. */
static ldlt_status update_contribution(double *F, int32_t R, int32_t nfs, int32_t nelim,
                                       const double *d, const double *off, ldlt_thread_ws *ws)
{
    int32_t m = R - nfs;
    if (m <= 0 || nelim <= 0) return LDLT_OK;
    int32_t cap = nelim + 1 < SIGNED_BATCH ? nelim + 1 : SIGNED_BATCH;
    double *W = ldlt_ws_doubles(&ws->batch, &ws->batch_cap, (size_t)m * cap);
    if (!W) return LDLT_ERR_NOMEM;
    double *C = F + nfs + (size_t)nfs*R;
    for (int sign = 1; sign >= -1; sign -= 2) {
        int32_t used = 0;
        for (int32_t c = 0; c < nelim; ++c) {
            const double *lc = F + nfs + (size_t)c*R;
            if (off[c] != 0.0) {
                const double *ln = lc + R;
                double a = d[c], b = off[c], g = d[c+1];
                double tau = (g - a) / (2.0 * b);
                double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
                double cs = 1.0 / sqrt(1.0 + t * t), sn = t * cs;
                double l1 = a - t * b, l2 = g + t * b;
                if (sign > 0 ? l1 > 0.0 : l1 < 0.0) {
                    double s = sqrt(fabs(l1));
                    double *w = W + (size_t)used*m;
                    for (int32_t i = 0; i < m; ++i) w[i] = (cs * lc[i] - sn * ln[i]) * s;
                    ++used;
                }
                if (used == cap) {
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, m, used,
                                -(double)sign, W, m, 1.0, C, R);
                    used = 0;
                }
                if (sign > 0 ? l2 > 0.0 : l2 < 0.0) {
                    double s = sqrt(fabs(l2));
                    double *w = W + (size_t)used*m;
                    for (int32_t i = 0; i < m; ++i) w[i] = (sn * lc[i] + cs * ln[i]) * s;
                    ++used;
                }
                ++c;
            } else {
                double dc = d[c];
                if (isinf(dc) || (sign > 0 ? dc <= 0.0 : dc >= 0.0)) continue;
                double s = sqrt(fabs(dc));
                double *w = W + (size_t)used*m;
                for (int32_t i = 0; i < m; ++i) w[i] = lc[i] * s;
                ++used;
            }
            if (used == cap) {
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, m, used,
                            -(double)sign, W, m, 1.0, C, R);
                used = 0;
            }
        }
        if (used > 0) {
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, m, used,
                        -(double)sign, W, m, 1.0, C, R);
        }
    }
    return LDLT_OK;
}

/* Tries to eliminate [bs, be) as they stand, in place, and checks the result afterwards:
   Cholesky if the block is positive definite, otherwise an unpivoted LDL^T of the block.
   The rows below go through the BLAS, so a block that passes costs little more than no
   pivoting at all. Returns 1 if it passed, with the fully summed columns after it updated.
   Otherwise the front is as it was. */
static int try_block(double *F, int32_t R, int32_t bs, int32_t be, int32_t nfs, double u,
                     double *d, double *off, ldlt_thread_ws *ws, ldlt_status *st)
{
    int32_t np = be - bs, below = R - be, nt = nfs - be;
    double *A11 = F + bs + (size_t)bs*R;
    double *A21 = A11 + np;
    double *save = ldlt_ws_doubles(&ws->save, &ws->save_cap, (size_t)(R - bs) * np);
    if (!save) { *st = LDLT_ERR_NOMEM; return 0; }
    size_t lds = (size_t)(R - bs);
    for (int32_t j = 0; j < np; ++j)
        memcpy(save + (size_t)j*lds + j, A11 + (size_t)j*R + j,
               (size_t)(R - bs - j) * sizeof(double));

    int ok = 1, chol = 0;
#if LDLT_HAVE_LAPACK
    {
        char uplo = 'L';
        int nk = np, lda = R, info = 0;
        dpotrf_(&uplo, &nk, A11, &lda, &info);
        chol = info == 0;
        if (!chol) {
            for (int32_t j = 0; j < np; ++j)
                memcpy(A11 + (size_t)j*R + j, save + (size_t)j*lds + j,
                       (size_t)(np - j) * sizeof(double));
        }
    }
#endif
    double bound = 1.0 / u;
    if (!chol) {
        for (int32_t j = 0; j < np && ok; ++j) {
            double *cj = A11 + (size_t)j*R;
            double dj = cj[j];
            if (!(fabs(dj) >= PIVOT_SMALL) || !isfinite(dj)) { ok = 0; break; }
            d[bs + j] = dj;
            double inv = 1.0 / dj;
            for (int32_t i = j + 1; i < np; ++i) {
                cj[i] *= inv;
                if (!(fabs(cj[i]) <= bound)) ok = 0;
            }
            for (int32_t c = j + 1; c < np; ++c) {
                double *cc = A11 + (size_t)c*R;
                double w = dj * cj[c];
                for (int32_t i = c; i < np; ++i) cc[i] -= cj[i] * w;
            }
        }
    }
    if (ok && below > 0) {
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                    chol ? CblasNonUnit : CblasUnit, below, np, 1.0, A11, R, A21, R);
    }
    if (ok) {
        for (int32_t c = 0; c < np && ok; ++c) {
            const double *col = A11 + (size_t)c*R;
            double limit = bound * (chol ? col[c] : fabs(d[bs + c]));
            double mx = 0.0;
            /* Without Cholesky, the block's own L was checked above and these still carry D. */
            for (int32_t i = chol ? c + 1 : np; i < R - bs; ++i) {
                double a = fabs(col[i]);
                if (a > mx) mx = a;
            }
            if (!(mx <= limit)) ok = 0;
        }
    }
    if (!ok) {
        for (int32_t j = 0; j < np; ++j)
            memcpy(A11 + (size_t)j*R + j, save + (size_t)j*lds + j,
                   (size_t)(R - bs - j) * sizeof(double));
        return 0;
    }

    double *Tr = F + be + (size_t)be*R;
    if (chol) {
        if (nt > 0) {
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, nt, np, -1.0, A21, R, 1.0, Tr, R);
            if (R > nfs) {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, R - nfs, nt, np,
                            -1.0, A21 + nt, R, A21, R, 1.0, Tr + nt, R);
            }
        }
        ldlt_cholesky_to_ldl(A11, R, R - bs, np, d + bs, ws->nth);
    } else {
        double *W = NULL;
        if (nt > 0) {
            W = ldlt_ws_doubles(&ws->batch, &ws->batch_cap, (size_t)nt * np);
            if (!W) { *st = LDLT_ERR_NOMEM; return 0; }
        }
        for (int32_t c = 0; c < np; ++c) {
            double *col = A21 + (size_t)c*R;
            if (nt > 0) memcpy(W + (size_t)c*nt, col, (size_t)nt * sizeof(double));
            double inv = 1.0 / d[bs + c];
            for (int32_t i = 0; i < below; ++i) col[i] *= inv;
            A11[(size_t)c*R + c] = 0.0;
        }
        if (nt > 0) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, below, nt, np,
                        -1.0, A21, R, W, nt, 1.0, Tr, R);
        }
    }
    for (int32_t c = bs; c < be; ++c) off[c] = 0.0;
    return 1;
}

/* Nothing is left to delay to at a root, so its last pivots come from the largest entry of
   what remains, which always passes the test (Bunch-Parlett). Returns the columns done. */
static int32_t root_pivot(double *F, int32_t R, int32_t el, double u, int32_t *vars,
                          double *d, double *off)
{
    double amax = 0.0, dmax = 0.0;
    int32_t p = -1, q = -1, dp = -1;
    for (int32_t j = el; j < R; ++j) {
        const double *cj = F + (size_t)j*R;
        double aj = fabs(cj[j]);
        if (aj > dmax) { dmax = aj; dp = j; }
        for (int32_t i = j + 1; i < R; ++i) {
            double a = fabs(cj[i]);
            if (a > amax) { amax = a; p = j; q = i; }
        }
    }
    if (dmax < PIVOT_SMALL && amax < PIVOT_SMALL) {
        for (int32_t j = el; j < R; ++j) hold_pivot(F, R, j, d, off);
        return R - el;
    }
    if (dp >= 0 && dmax >= u * amax && dmax >= PIVOT_SMALL) {
        sym_swap(F, R, vars, el, dp);
        apply_1x1(F, R, el, R, d, off);
        return 1;
    }
    sym_swap(F, R, vars, el, p);
    if (q == el) q = p;
    sym_swap(F, R, vars, el + 1, q);
    apply_2x2(F, R, el, R, d, off);
    return 2;
}

static ldlt_status eliminate(double *F, int32_t R, int32_t nfs, int is_root, double u,
                             int32_t *vars, double *d, double *off, int32_t *nelim_out,
                             ldlt_thread_ws *ws)
{
    *nelim_out = 0;
    if (nfs <= 0) return LDLT_OK;
    ldlt_status st = LDLT_OK;
    int32_t el = 0, re = nfs;
    int32_t try_size = TRY_BLOCK_MAX;
    int progress = 0;
    while (el < nfs) {
        if (el == re) {
            if (progress) {
                re = nfs;
                progress = 0;
                continue;
            }
            if (!is_root) break;
            el += root_pivot(F, R, el, u, vars, d, off);
            re = nfs;
            continue;
        }
        int32_t bs = el;
        if (re - bs >= TRY_BLOCK_MIN) {
            int32_t be = bs + try_size < re ? bs + try_size : re;
            if (try_block(F, R, bs, be, nfs, u, d, off, ws, &st)) {
                el = be;
                progress = 1;
                if (try_size < TRY_BLOCK_MAX) try_size *= 2;
                continue;
            }
            if (st != LDLT_OK) return st;
            /* Trying costs more than the work it saves when most blocks fail. */
            if (try_size > TRY_BLOCK_MIN) try_size /= 2;
        }
        int32_t be = bs + PIVOT_BLOCK < re ? bs + PIVOT_BLOCK : re;
        while (el < be) {
            int32_t p = -1, q = -1, size = -1;
            st = choose_pivot(F, R, el, be, u, &p, &q, &size);
            if (st != LDLT_OK) return st;
            if (size < 0) break;
            sym_swap(F, R, vars, el, p);
            if (size == 0) {
                hold_pivot(F, R, el, d, off);
                el += 1;
            } else if (size == 1) {
                apply_1x1(F, R, el, be, d, off);
                el += 1;
            } else {
                if (q == el) q = p;
                sym_swap(F, R, vars, el + 1, q);
                apply_2x2(F, R, el, be, d, off);
                el += 2;
            }
            progress = 1;
        }
        st = update_trailing(F, R, bs, el, be, nfs, d, off, ws);
        if (st != LDLT_OK) return st;
        for (int32_t x = be - 1; x >= el; --x) {
            int32_t y = --re;
            sym_swap(F, R, vars, x, y);
        }
    }
    *nelim_out = el;
    return update_contribution(F, R, nfs, el, d, off, ws);
}

static size_t packed_col(int32_t n, int32_t j)
{
    return (size_t)j * (size_t)n - ((size_t)j * (size_t)(j - 1)) / 2;
}

/* A contribution with delayed columns does not map to increasing positions. */
static void extend_add_general(double *F, int32_t R, const double *src, int32_t nc,
                               const int32_t *pos)
{
    for (int32_t j = 0; j < nc; ++j) {
        size_t off = packed_col(nc, j) - (size_t)j;
        int32_t pj = pos[j];
        for (int32_t i = j; i < nc; ++i) {
            int32_t pi = pos[i];
            double v = src[off + i];
            if (pi >= pj) F[pi + (size_t)pj*R] += v;
            else          F[pj + (size_t)pi*R] += v;
        }
    }
}

static ldlt_status factor_node(const ldlt_symbolic *S, ldlt_numeric *N, int32_t s,
                               pivot_contrib *contrib, const double *Bx, int require_pd,
                               double u, ldlt_thread_ws *ws)
{
    const ldlt_super *Ss = &S->super[s];
    int32_t fc = Ss->first_col, k = Ss->width, m = Ss->nrows_below;
    const int32_t *rb = S->rows_pool + Ss->row_off;
    ldlt_pivot_node *node = &N->nodes[s];

    int32_t nd = 0;
    for (int32_t cp = S->child_ptr[s]; cp < S->child_ptr[s+1]; ++cp)
        nd += contrib[S->child_idx[cp]].ndelay;
    int32_t nfs = k + nd, R = nfs + m;

    double *F = ldlt_ws_doubles(&ws->C, &ws->C_cap, (size_t)R * R);
    int32_t *vars = (int32_t*)ldlt_alloc_array((size_t)R, sizeof(int32_t));
    double *d = (double*)ldlt_alloc_array((size_t)nfs, sizeof(double));
    double *off = (double*)ldlt_alloc_array((size_t)nfs, sizeof(double));
    if (ws->pos_cap < (size_t)R) {
        free(ws->pos);
        ws->pos = (int32_t*)ldlt_alloc_array((size_t)R, sizeof(int32_t));
        ws->pos_cap = ws->pos ? (size_t)R : 0;
    }
    node->vars = vars;
    node->d = d;
    node->e = off;
    if (!F || !vars || !d || !off || !ws->pos) return LDLT_ERR_NOMEM;
    ldlt_zero_lower(F, R, R, ws->nth);

    int32_t *rowmap = ws->rowmap;
    for (int32_t i = 0; i < k; ++i) { vars[i] = fc + i; rowmap[fc + i] = i; }
    for (int32_t i = 0; i < m; ++i) { vars[nfs + i] = rb[i]; rowmap[rb[i]] = nfs + i; }
    for (int32_t lc = 0; lc < k; ++lc) {
        double *col = F + (size_t)lc*R;
        for (int32_t p = S->Bp[fc + lc]; p < S->Bp[fc + lc + 1]; ++p)
            col[rowmap[S->Bi[p]]] += Bx[p];
    }

    int32_t next_delayed = k;
    size_t low = ws->stack_top;
    for (int32_t cp = S->child_ptr[s]; cp < S->child_ptr[s+1]; ++cp) {
        int32_t c = S->child_idx[cp];
        pivot_contrib *pc = &contrib[c];
        if (pc->nc == 0) continue;
        const int32_t *emap = S->edge_pos + S->super[c].row_off;
        int32_t *pos = ws->pos;
        for (int32_t t = 0; t < pc->ndelay; ++t) {
            pos[t] = next_delayed;
            vars[next_delayed++] = pc->delayed[t];
        }
        for (int32_t t = pc->ndelay; t < pc->nc; ++t) {
            int32_t e = emap[t - pc->ndelay];
            pos[t] = e < k ? e : e + nd;
        }
        if (pc->ndelay == 0) {
            ldlt_status st = ldlt_extend_add(F, R, R, NULL, 0, pc->A, pc->nc, pos, ws);
            if (st != LDLT_OK) return st;
        } else {
            extend_add_general(F, R, pc->A, pc->nc, pos);
        }
        ldlt_contrib_release(ws, pc->A, &low);
        free(pc->delayed);
        memset(pc, 0, sizeof(*pc));
    }
    ws->stack_top = low;

    int32_t nelim = 0;
    int is_root = S->super_parent[s] < 0;
    ldlt_status st = eliminate(F, R, nfs, is_root, u, vars, d, off, &nelim, ws);
    if (st != LDLT_OK) return st;
    if (require_pd) {
        for (int32_t c = 0; c < nelim; ++c)
            if (off[c] != 0.0 || !(d[c] > 0.0) || isinf(d[c])) return LDLT_ERR_INDEFINITE;
    }

    node->nelim = nelim;
    node->nrows = R;
    node->ndelay = nfs - nelim;
    if (nelim > 0) {
        node->L = (double*)ldlt_alloc_array((size_t)R * nelim, sizeof(double));
        if (!node->L) return LDLT_ERR_NOMEM;
        memcpy(node->L, F, (size_t)R * nelim * sizeof(double));
    }
    pivot_contrib *out = &contrib[s];
    out->ndelay = nfs - nelim;
    out->nc = R - nelim;
    if (out->nc > 0) {
        st = ldlt_pack_lower(F + nelim + (size_t)nelim*R, R, out->nc, &out->A, ws);
        if (st != LDLT_OK) return st;
    }
    if (out->ndelay > 0) {
        out->delayed = (int32_t*)ldlt_alloc_array((size_t)out->ndelay, sizeof(int32_t));
        if (!out->delayed) return LDLT_ERR_NOMEM;
        memcpy(out->delayed, vars + nelim, (size_t)out->ndelay * sizeof(int32_t));
    }
    return LDLT_OK;
}

static void finish_info(const ldlt_symbolic *S, ldlt_numeric *N, const double *Bx)
{
    ldlt_factor_info *fi = &N->info;
    fi->lnz = 0;
    for (int32_t s = 0; s < S->nsuper; ++s) {
        const ldlt_pivot_node *node = &N->nodes[s];
        fi->n_delayed += node->ndelay;
        for (int32_t c = 0; c < node->nelim; ++c) {
            fi->lnz += node->nrows - c;
            double dc = node->d[c];
            if (node->e[c] != 0.0) {
                double det = dc * node->d[c+1] - node->e[c] * node->e[c];
                fi->n_2x2++;
                if (det < 0.0) { fi->n_positive++; fi->n_negative++; }
                else if (dc > 0.0) fi->n_positive += 2;
                else fi->n_negative += 2;
                fi->lnz += node->nrows - c - 1;
                ++c;
                continue;
            }
            if (isinf(dc)) { fi->n_zero++; continue; }
            if (dc > 0.0) fi->n_positive++; else fi->n_negative++;
            int32_t v = node->vars[c];
            /* Both sides of the ratio scale alike. */
            double aii = S->Bdiag[v] >= 0 ? fabs(Bx[S->Bdiag[v]]) : 0.0;
            double r = aii / fabs(dc);
            if (r > fi->max_ratio) { fi->max_ratio = r; fi->max_ratio_row = S->perm[v]; }
        }
    }
}

ldlt_status ldlt_factorize_threshold(const ldlt_symbolic *S, ldlt_numeric *N, double *Bx,
                                     const ldlt_options *opt, int nth,
                                     const ldlt_schedule *P, int top_blas_threads)
{
    double u = opt->pivot_threshold > 0.0 ? opt->pivot_threshold : 0.01;
    if (u > 0.5) u = 0.5;
    int require_pd = opt->require_pd;

    N->pivoted = 1;
    N->scale = (double*)ldlt_alloc_array((size_t)S->n, sizeof(double));
    N->nodes = (ldlt_pivot_node*)ldlt_calloc_array((size_t)S->nsuper, sizeof(ldlt_pivot_node));
    pivot_contrib *contrib =
        (pivot_contrib*)ldlt_calloc_array((size_t)S->nsuper, sizeof(pivot_contrib));
    if (!N->scale || !N->nodes || !contrib) { free(contrib); return LDLT_ERR_NOMEM; }
    ldlt_status status = ruiz_scale(S, Bx, N->scale);

    if (status == LDLT_OK && P->nsub > 0) {
#if LDLT_HAVE_OPENMP
        #pragma omp parallel num_threads(nth)
#endif
        {
            ldlt_thread_ws ws;
            memset(&ws, 0, sizeof(ws));
            ws.nth = 1;
            ws.rowmap = (int32_t*)ldlt_alloc_array((size_t)S->n, sizeof(int32_t));
            ldlt_blas_this_thread_single();
#if LDLT_HAVE_OPENMP
            #pragma omp for schedule(dynamic, 1)
#endif
            for (int32_t t = 0; t < P->nsub; ++t) {
                ldlt_status seen;
#if LDLT_HAVE_OPENMP
                #pragma omp atomic read
#endif
                seen = status;
                if (seen != LDLT_OK) continue;
                ldlt_status st = ws.rowmap ? ldlt_ws_reserve_stack(&ws, P->sub_stack[t])
                                           : LDLT_ERR_NOMEM;
                for (int32_t s = P->first[t]; s <= P->root[t] && st == LDLT_OK; ++s) {
                    ws.blocks_to_heap = s == P->root[t];
                    st = factor_node(S, N, s, contrib, Bx, require_pd, u, &ws);
                }
                if (st != LDLT_OK) {
                    /* What's in the stack goes when the thread's buffers do. */
                    size_t unused = 0;
                    for (int32_t s = P->first[t]; s <= P->root[t]; ++s) {
                        if (contrib[s].A) ldlt_contrib_release(&ws, contrib[s].A, &unused);
                        contrib[s].A = NULL;
                    }
#if LDLT_HAVE_OPENMP
                    #pragma omp atomic write
#endif
                    status = st;
                }
            }
            ldlt_blas_this_thread_multi();
            ldlt_ws_free(&ws);
        }
    }
    if (status == LDLT_OK) {
        ldlt_blas_global_threads(top_blas_threads);
        ldlt_thread_ws ws;
        size_t mf = (size_t)P->top_max_front;
        status = ldlt_ws_prepare_top(&ws, S->n, mf * mf, P, nth);
        for (int32_t t = 0; t < P->ntop && status == LDLT_OK; ++t)
            status = factor_node(S, N, P->top[t], contrib, Bx, require_pd, u, &ws);
        /* A block left in the stack is gone with it, so nothing may free it afterwards. */
        for (int32_t s = 0; s < S->nsuper; ++s) {
            size_t unused = 0;
            if (contrib[s].A) ldlt_contrib_release(&ws, contrib[s].A, &unused);
            contrib[s].A = NULL;
        }
        ldlt_ws_free(&ws);
    }
    /* After a failed subtree phase, every block left is on the heap. */
    for (int32_t s = 0; s < S->nsuper; ++s) { free(contrib[s].A); free(contrib[s].delayed); }
    free(contrib);
    if (status == LDLT_OK) finish_info(S, N, Bx);
    return status;
}
