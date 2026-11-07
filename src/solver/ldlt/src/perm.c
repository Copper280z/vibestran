#include "internal.h"

/* Pattern of B = P A P^T (lower triangle). Row indices within a column are not sorted:
   nothing downstream needs them to be, and leaving them out keeps this linear. With map
   non-NULL, map[p] is where entry p of A lands in B, so a later factorization can place
   the values without redoing any of this. */
ldlt_status ldlt_permute_lower_pattern(int32_t n,
    const int32_t *Ap, const int32_t *Ai, const int32_t *iperm,
    int32_t **Bp_out, int32_t **Bi_out, int32_t **map_out)
{
    int32_t nnz = Ap[n];
    int32_t *Bp = (int32_t*)ldlt_calloc_array((size_t)n + 1, sizeof(int32_t));
    int32_t *Bi = (int32_t*)ldlt_alloc_array((size_t)nnz, sizeof(int32_t));
    int32_t *cur = (int32_t*)ldlt_alloc_array((size_t)n, sizeof(int32_t));
    int32_t *map = map_out ? (int32_t*)ldlt_alloc_array((size_t)nnz, sizeof(int32_t)) : NULL;
    if (!Bp || !Bi || !cur || (map_out && !map)) {
        free(Bp); free(Bi); free(cur); free(map);
        return LDLT_ERR_NOMEM;
    }
    for (int32_t j = 0; j < n; ++j) {
        int32_t nj = iperm[j];
        for (int32_t p = Ap[j]; p < Ap[j+1]; ++p) {
            int32_t ni = iperm[Ai[p]];
            Bp[(ni < nj ? ni : nj) + 1]++;
        }
    }
    for (int32_t i = 0; i < n; ++i) Bp[i+1] += Bp[i];
    if (n > 0) memcpy(cur, Bp, (size_t)n * sizeof(int32_t));
    for (int32_t j = 0; j < n; ++j) {
        int32_t nj = iperm[j];
        for (int32_t p = Ap[j]; p < Ap[j+1]; ++p) {
            int32_t ni = iperm[Ai[p]];
            int32_t col = ni < nj ? ni : nj;
            int32_t row = ni < nj ? nj : ni;
            int32_t q = cur[col]++;
            Bi[q] = row;
            if (map) map[p] = q;
        }
    }
    free(cur);
    *Bp_out = Bp; *Bi_out = Bi;
    if (map_out) *map_out = map;
    return LDLT_OK;
}
