/* Minimal Matrix Market reader for coordinate real or pattern matrices.
   Returns the lower triangle in CSC (rows >= cols, increasing in each column), 0-indexed.
   A symmetric file may store either triangle; a general file keeps only its lower one.
   Repeated entries are summed. Pattern matrices are loaded with value 1.0 for each
   structural entry; callers should not treat them as validated numeric matrices. */

#include "internal.h"
#include <stdio.h>
#include <string.h>

static int starts_with(const char *s, const char *pre) {
    return strncmp(s, pre, strlen(pre)) == 0;
}

ldlt_status ldlt_read_mm_symmetric(const char *path, int32_t *n_out,
                                   int32_t **Ap_out, int32_t **Ai_out, double **Ax_out)
{
    FILE *f = fopen(path, "r");
    if (!f) return LDLT_ERR_INPUT;
    char line[1024];
    if (!fgets(line, sizeof(line), f) || !starts_with(line, "%%MatrixMarket")) {
        fclose(f);
        return LDLT_ERR_INPUT;
    }
    int symmetric = strstr(line, "symmetric") != NULL;
    int pattern = strstr(line, "pattern") != NULL;
    if (symmetric && pattern) {
        fprintf(stderr,
                "ldlt warning: loading symmetric pattern Matrix Market file '%s' "
                "with implicit value 1.0 entries; this is a structural pattern, "
                "not a validated numeric matrix.\n",
                path);
    }
    do {
        if (!fgets(line, sizeof(line), f)) { fclose(f); return LDLT_ERR_INPUT; }
    } while (line[0] == '%');
    long M, N, nnz;
    if (sscanf(line, "%ld %ld %ld", &M, &N, &nnz) != 3 || M != N || N < 0 || nnz < 0 ||
        N > INT32_MAX || nnz > INT32_MAX) {
        fclose(f);
        return LDLT_ERR_INPUT;
    }

    size_t cap = (size_t)nnz;
    int32_t *ri = (int32_t*)ldlt_alloc_array(cap, sizeof(int32_t));
    int32_t *ci = (int32_t*)ldlt_alloc_array(cap, sizeof(int32_t));
    double *vx = (double*)ldlt_alloc_array(cap, sizeof(double));
    int32_t *order = (int32_t*)ldlt_alloc_array(cap, sizeof(int32_t));
    int32_t *order2 = (int32_t*)ldlt_alloc_array(cap, sizeof(int32_t));
    int32_t *count = (int32_t*)ldlt_calloc_array((size_t)N + 2, sizeof(int32_t));
    int32_t *Ap = (int32_t*)ldlt_calloc_array((size_t)N + 1, sizeof(int32_t));
    int32_t *Ai = NULL;
    double *Ax = NULL;
    ldlt_status st = LDLT_ERR_NOMEM;
    if (!ri || !ci || !vx || !order || !order2 || !count || !Ap) goto fail;

    st = LDLT_ERR_INPUT;
    int32_t actual = 0;
    for (long t = 0; t < nnz; ++t) {
        long i, j;
        double v = 1.0;
        int nread = pattern ? fscanf(f, "%ld %ld", &i, &j)
                            : fscanf(f, "%ld %ld %lf", &i, &j, &v);
        if (nread < (pattern ? 2 : 3) || i < 1 || i > N || j < 1 || j > N) goto fail;
        i--; j--;
        if (i < j) {
            if (!symmetric) continue;
            long tmp = i; i = j; j = tmp;
        }
        ri[actual] = (int32_t)i;
        ci[actual] = (int32_t)j;
        vx[actual] = v;
        actual++;
    }

    /* Stable counting sorts: by row, then by column. */
    for (int32_t t = 0; t < actual; ++t) count[ri[t] + 1]++;
    for (long i = 0; i < N; ++i) count[i+1] += count[i];
    for (int32_t t = 0; t < actual; ++t) order[count[ri[t]]++] = t;
    memset(count, 0, ((size_t)N + 2) * sizeof(int32_t));
    for (int32_t t = 0; t < actual; ++t) count[ci[t] + 1]++;
    for (long i = 0; i < N; ++i) count[i+1] += count[i];
    for (int32_t t = 0; t < actual; ++t) order2[count[ci[order[t]]]++] = order[t];

    st = LDLT_ERR_NOMEM;
    Ai = (int32_t*)ldlt_alloc_array((size_t)actual, sizeof(int32_t));
    Ax = (double*)ldlt_alloc_array((size_t)actual, sizeof(double));
    if (!Ai || !Ax) goto fail;
    int32_t out = 0;
    int32_t col = 0;
    for (int32_t t = 0; t < actual; ++t) {
        int32_t e = order2[t];
        while (col < ci[e]) Ap[++col] = out;
        if (out > Ap[col] && Ai[out-1] == ri[e]) {
            Ax[out-1] += vx[e];
        } else {
            Ai[out] = ri[e];
            Ax[out] = vx[e];
            out++;
        }
    }
    while (col < N) Ap[++col] = out;

    fclose(f);
    free(ri); free(ci); free(vx); free(order); free(order2); free(count);
    *n_out = (int32_t)N; *Ap_out = Ap; *Ai_out = Ai; *Ax_out = Ax;
    return LDLT_OK;

fail:
    fclose(f);
    free(ri); free(ci); free(vx); free(order); free(order2); free(count);
    free(Ap); free(Ai); free(Ax);
    return st;
}
