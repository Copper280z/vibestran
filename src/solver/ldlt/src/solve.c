/* Triangular solves with the factors: L y = S P b, D z = y, L^T x = z, b = P^T S x, where S
   is the pivoting path's scaling. Up the tree and back down, one supernode at a time. */

#include "internal.h"
#include "blas_shim.h"
#include <math.h>

/* Below this width, plain loops beat the BLAS call overhead. */
#define SMALL_SOLVE 4

static void forward_node(int32_t k, int32_t m, int32_t nrhs, const double *L, int32_t ld,
                         double *top, int32_t ldt, double *below, int32_t ldb_)
{
    if (k <= SMALL_SOLVE) {
        for (int32_t r = 0; r < nrhs; ++r) {
            double *x = top + (size_t)r*ldt;
            double *y = below + (size_t)r*ldb_;
            for (int32_t j = 0; j < k; ++j) {
                const double *col = L + (size_t)j*ld;
                double xj = x[j];
                for (int32_t i = j + 1; i < k; ++i) x[i] -= col[i] * xj;
                for (int32_t t = 0; t < m; ++t) y[t] -= col[k + t] * xj;
            }
        }
        return;
    }
    if (nrhs == 1) {
        cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, k, L, ld, top, 1);
        if (m > 0)
            cblas_dgemv(CblasColMajor, CblasNoTrans, m, k, -1.0, L + k, ld, top, 1, 1.0, below, 1);
        return;
    }
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                k, nrhs, 1.0, L, ld, top, ldt);
    if (m > 0)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, nrhs, k,
                    -1.0, L + k, ld, top, ldt, 1.0, below, ldb_);
}

static void backward_node(int32_t k, int32_t m, int32_t nrhs, const double *L, int32_t ld,
                          double *top, int32_t ldt, const double *below, int32_t ldb_)
{
    if (k <= SMALL_SOLVE) {
        for (int32_t r = 0; r < nrhs; ++r) {
            double *x = top + (size_t)r*ldt;
            const double *y = below + (size_t)r*ldb_;
            for (int32_t j = k - 1; j >= 0; --j) {
                const double *col = L + (size_t)j*ld;
                double acc = x[j];
                for (int32_t i = j + 1; i < k; ++i) acc -= col[i] * x[i];
                for (int32_t t = 0; t < m; ++t) acc -= col[k + t] * y[t];
                x[j] = acc;
            }
        }
        return;
    }
    if (nrhs == 1) {
        if (m > 0)
            cblas_dgemv(CblasColMajor, CblasTrans, m, k, -1.0, L + k, ld, below, 1, 1.0, top, 1);
        cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasUnit, k, L, ld, top, 1);
        return;
    }
    if (m > 0)
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, nrhs, m,
                    -1.0, L + k, ld, below, ldb_, 1.0, top, ldt);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                k, nrhs, 1.0, L, ld, top, ldt);
}

static void solve_panels(const ldlt_numeric *N, int32_t nrhs, double *X, int32_t ldX, double *W)
{
    const ldlt_symbolic *S = N->S;
    for (int32_t s = 0; s < S->nsuper; ++s) {
        const ldlt_super *Ss = &S->super[s];
        int32_t fc = Ss->first_col, k = Ss->width, m = Ss->nrows_below;
        const int32_t *rb = S->rows_pool + Ss->row_off;
        const double *L = N->L + S->panel_off[s];
        if (m > 0) memset(W, 0, (size_t)m * nrhs * sizeof(double));
        forward_node(k, m, nrhs, L, k + m, X + fc, ldX, W, m);
        for (int32_t r = 0; r < nrhs; ++r) {
            double *x = X + (size_t)r*ldX;
            const double *w = W + (size_t)r*m;
            for (int32_t t = 0; t < m; ++t) x[rb[t]] += w[t];
        }
    }
    for (int32_t r = 0; r < nrhs; ++r) {
        double *x = X + (size_t)r*ldX;
        for (int32_t j = 0; j < S->n; ++j) x[j] /= N->D[j];
    }
    for (int32_t s = S->nsuper - 1; s >= 0; --s) {
        const ldlt_super *Ss = &S->super[s];
        int32_t fc = Ss->first_col, k = Ss->width, m = Ss->nrows_below;
        const int32_t *rb = S->rows_pool + Ss->row_off;
        const double *L = N->L + S->panel_off[s];
        for (int32_t r = 0; r < nrhs; ++r) {
            const double *x = X + (size_t)r*ldX;
            double *w = W + (size_t)r*m;
            for (int32_t t = 0; t < m; ++t) w[t] = x[rb[t]];
        }
        backward_node(k, m, nrhs, L, k + m, X + fc, ldX, W, m);
    }
}

static void solve_pivoted(const ldlt_numeric *N, int32_t nrhs, double *X, int32_t ldX, double *W,
                          int32_t ldW)
{
    const ldlt_symbolic *S = N->S;
    for (int32_t s = 0; s < S->nsuper; ++s) {
        const ldlt_pivot_node *node = &N->nodes[s];
        int32_t k = node->nelim, R = node->nrows;
        if (k == 0) continue;
        for (int32_t r = 0; r < nrhs; ++r) {
            const double *x = X + (size_t)r*ldX;
            double *w = W + (size_t)r*ldW;
            for (int32_t i = 0; i < k; ++i) w[i] = x[node->vars[i]];
            for (int32_t i = k; i < R; ++i) w[i] = 0.0;
        }
        forward_node(k, R - k, nrhs, node->L, R, W, ldW, W + k, ldW);
        for (int32_t r = 0; r < nrhs; ++r) {
            double *x = X + (size_t)r*ldX;
            const double *w = W + (size_t)r*ldW;
            for (int32_t i = 0; i < k; ++i) x[node->vars[i]] = w[i];
            for (int32_t i = k; i < R; ++i) x[node->vars[i]] += w[i];
        }
    }
    for (int32_t s = 0; s < S->nsuper; ++s) {
        const ldlt_pivot_node *node = &N->nodes[s];
        for (int32_t c = 0; c < node->nelim; ++c) {
            int32_t v = node->vars[c];
            if (node->e[c] != 0.0) {
                int32_t v1 = node->vars[c+1];
                double a = node->d[c], b = node->e[c], g = node->d[c+1];
                double det = a * g - b * b;
                for (int32_t r = 0; r < nrhs; ++r) {
                    double *x = X + (size_t)r*ldX;
                    double p = x[v], q = x[v1];
                    x[v]  = (g * p - b * q) / det;
                    x[v1] = (a * q - b * p) / det;
                }
                ++c;
            } else {
                for (int32_t r = 0; r < nrhs; ++r) X[v + (size_t)r*ldX] /= node->d[c];
            }
        }
    }
    for (int32_t s = S->nsuper - 1; s >= 0; --s) {
        const ldlt_pivot_node *node = &N->nodes[s];
        int32_t k = node->nelim, R = node->nrows;
        if (k == 0) continue;
        for (int32_t r = 0; r < nrhs; ++r) {
            const double *x = X + (size_t)r*ldX;
            double *w = W + (size_t)r*ldW;
            for (int32_t i = 0; i < R; ++i) w[i] = x[node->vars[i]];
        }
        backward_node(k, R - k, nrhs, node->L, R, W, ldW, W + k, ldW);
        for (int32_t r = 0; r < nrhs; ++r) {
            double *x = X + (size_t)r*ldX;
            const double *w = W + (size_t)r*ldW;
            for (int32_t i = 0; i < k; ++i) x[node->vars[i]] = w[i];
        }
    }
}

ldlt_status ldlt_solve(const ldlt_numeric *N, int32_t nrhs, double *B, int32_t ldb)
{
    if (!N || nrhs < 0 || (!B && nrhs > 0)) return LDLT_ERR_INPUT;
    const ldlt_symbolic *S = N->S;
    int32_t n = S->n;
    if (ldb < n) return LDLT_ERR_INPUT;
    if (nrhs == 0 || n == 0) return LDLT_OK;

    int32_t maxR = 1;
    if (N->pivoted) {
        for (int32_t s = 0; s < S->nsuper; ++s)
            if (N->nodes[s].nrows > maxR) maxR = N->nodes[s].nrows;
    } else {
        for (int32_t s = 0; s < S->nsuper; ++s)
            if (S->super[s].nrows_below > maxR) maxR = S->super[s].nrows_below;
    }
    double *W = (double*)ldlt_alloc_array((size_t)maxR * nrhs, sizeof(double));
    if (!W) return LDLT_ERR_NOMEM;

    int in_place = S->perm_identity && !N->scale;
    double *X = B;
    int32_t ldX = ldb;
    if (!in_place) {
        X = (double*)ldlt_alloc_array((size_t)n * nrhs, sizeof(double));
        if (!X) { free(W); return LDLT_ERR_NOMEM; }
        ldX = n;
        for (int32_t r = 0; r < nrhs; ++r) {
            const double *b = B + (size_t)r*ldb;
            double *x = X + (size_t)r*n;
            if (N->scale) for (int32_t i = 0; i < n; ++i) x[i] = b[S->perm[i]] * N->scale[i];
            else          for (int32_t i = 0; i < n; ++i) x[i] = b[S->perm[i]];
        }
    }
    if (N->pivoted) solve_pivoted(N, nrhs, X, ldX, W, maxR);
    else solve_panels(N, nrhs, X, ldX, W);
    if (!in_place) {
        for (int32_t r = 0; r < nrhs; ++r) {
            double *b = B + (size_t)r*ldb;
            const double *x = X + (size_t)r*n;
            if (N->scale) for (int32_t i = 0; i < n; ++i) b[S->perm[i]] = x[i] * N->scale[i];
            else          for (int32_t i = 0; i < n; ++i) b[S->perm[i]] = x[i];
        }
        free(X);
    }
    free(W);
    return LDLT_OK;
}
