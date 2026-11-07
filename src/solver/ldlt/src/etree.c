/* Elimination tree and postorder.
   Lower-triangle CSC input; etree built via Liu's path-compression algorithm. */

#include "internal.h"

/* Liu/Davis etree with path compression. Iterates rows in increasing order
   (outer = "k"), walking upward from each upper-triangle row index of column k.
   Input is lower-CSC, so we first build per-row lists of cols j<i (Up/Ui), then
   process each row's upper entries against the in-progress ancestor[] chain. */
ldlt_status ldlt_etree(int32_t n, const int32_t *Ap, const int32_t *Ai, int32_t *parent) {
    int32_t *Up = (int32_t*)ldlt_calloc_array((size_t)n+1, sizeof(int32_t));
    if (!Up) return LDLT_ERR_NOMEM;
    for (int32_t k = 0; k < n; ++k)
        for (int32_t p = Ap[k]; p < Ap[k+1]; ++p) {
            int32_t i = Ai[p];
            if (i > k) Up[i+1]++;
        }
    for (int32_t i = 0; i < n; ++i) Up[i+1] += Up[i];
    int32_t *Ui = (int32_t*)ldlt_alloc_array((size_t)Up[n], sizeof(int32_t));
    int32_t *cur = (int32_t*)ldlt_alloc_array((size_t)n, sizeof(int32_t));
    int32_t *anc = (int32_t*)ldlt_alloc_array((size_t)n, sizeof(int32_t));
    if (!Ui || !cur || !anc) { free(Up); free(Ui); free(cur); free(anc); return LDLT_ERR_NOMEM; }
    if (n > 0) memcpy(cur, Up, (size_t)n * sizeof(int32_t));
    for (int32_t k = 0; k < n; ++k)
        for (int32_t p = Ap[k]; p < Ap[k+1]; ++p) {
            int32_t i = Ai[p];
            if (i > k) Ui[cur[i]++] = k;
        }
    free(cur);

    for (int32_t k = 0; k < n; ++k) { parent[k] = -1; anc[k] = -1; }
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t p = Up[i]; p < Up[i+1]; ++p) {
            int32_t s = Ui[p];                  /* col, s < i */
            while (s != -1 && s < i) {
                int32_t nxt = anc[s];
                anc[s] = i;
                if (nxt == -1) { parent[s] = i; break; }
                s = nxt;
            }
        }
    }
    free(anc); free(Up); free(Ui);
    return LDLT_OK;
}

static int32_t tdfs(int32_t root, int32_t k, int32_t *head, const int32_t *next,
                    int32_t *post, int32_t *stack) {
    int32_t top = 0;
    stack[0] = root;
    while (top >= 0) {
        int32_t p = stack[top];
        int32_t i = head[p];
        if (i == -1) { post[k++] = p; --top; }
        else { head[p] = next[i]; stack[++top] = i; }
    }
    return k;
}

/* Postorder of the forest. With a weight (each 0 <= weight[j] <= n) the children of a
   node are visited in increasing weight, so the heaviest child comes right before its
   parent. Giving it the column count puts the child most likely to join the parent's
   supernode next to it. */
ldlt_status ldlt_postorder(int32_t n, const int32_t *parent, const int32_t *weight,
                           int32_t *post) {
    int32_t *head = (int32_t*)ldlt_alloc_array((size_t)n + 1, sizeof(int32_t));
    int32_t *next = (int32_t*)ldlt_alloc_array((size_t)n + 1, sizeof(int32_t));
    int32_t *stk  = (int32_t*)ldlt_alloc_array((size_t)n + 1, sizeof(int32_t));
    int32_t *wh = weight ? (int32_t*)ldlt_alloc_array((size_t)n + 1, sizeof(int32_t)) : NULL;
    int32_t *wn = weight ? (int32_t*)ldlt_alloc_array((size_t)n + 1, sizeof(int32_t)) : NULL;
    if (!head || !next || !stk || (weight && (!wh || !wn))) {
        free(head); free(next); free(stk); free(wh); free(wn);
        return LDLT_ERR_NOMEM;
    }
    for (int32_t i = 0; i < n; ++i) head[i] = -1;
    if (weight) {
        for (int32_t w = 0; w <= n; ++w) wh[w] = -1;
        for (int32_t j = n-1; j >= 0; --j) { wn[j] = wh[weight[j]]; wh[weight[j]] = j; }
        for (int32_t w = n; w >= 0; --w) {
            for (int32_t j = wh[w]; j != -1; j = wn[j]) {
                if (parent[j] == -1) continue;
                next[j] = head[parent[j]];
                head[parent[j]] = j;
            }
        }
    } else {
        for (int32_t j = n-1; j >= 0; --j) {
            if (parent[j] == -1) continue;
            next[j] = head[parent[j]];
            head[parent[j]] = j;
        }
    }
    int32_t k = 0;
    for (int32_t j = 0; j < n; ++j) {
        if (parent[j] != -1) continue;
        k = tdfs(j, k, head, next, post, stk);
    }
    free(head); free(next); free(stk); free(wh); free(wn);
    return LDLT_OK;
}

/* Number of entries in each column of L, diagonal included, without forming L: the
   skeleton-graph method of Gilbert, Ng and Peyton (1994). Column j's count is the number
   of row subtrees it belongs to, and each row subtree is the union of the paths from its
   leaves to the row. Walking the columns in postorder finds those leaves in order, and
   the least common ancestor of consecutive leaves says where two paths join and must not
   be counted twice. Close to linear in nnz(A). */
ldlt_status ldlt_col_counts(int32_t n, const int32_t *Ap, const int32_t *Ai,
                            const int32_t *parent, const int32_t *post,
                            int32_t *count) {
    int32_t *w = (int32_t*)ldlt_alloc_array((size_t)n * 4 + 1, sizeof(int32_t));
    if (!w) return LDLT_ERR_NOMEM;
    int32_t *first = w, *maxfirst = w + n, *prevleaf = w + 2*(size_t)n, *anc = w + 3*(size_t)n;
    for (int32_t i = 0; i < n; ++i) {
        first[i] = -1; maxfirst[i] = -1; prevleaf[i] = -1; anc[i] = i;
    }
    for (int32_t k = 0; k < n; ++k) {
        int32_t j = post[k];
        count[j] = (first[j] == -1) ? 1 : 0;
        for (; j != -1 && first[j] == -1; j = parent[j]) first[j] = k;
    }
    for (int32_t k = 0; k < n; ++k) {
        int32_t j = post[k];
        if (parent[j] != -1) count[parent[j]]--;
        for (int32_t p = Ap[j]; p < Ap[j+1]; ++p) {
            int32_t i = Ai[p];
            if (i <= j || first[j] <= maxfirst[i]) continue;
            maxfirst[i] = first[j];
            int32_t leaf = prevleaf[i];
            prevleaf[i] = j;
            count[j]++;
            if (leaf != -1) {
                int32_t q = leaf;
                while (q != anc[q]) q = anc[q];
                for (int32_t t = leaf; t != q; ) {
                    int32_t up = anc[t];
                    anc[t] = q;
                    t = up;
                }
                count[q]--;
            }
        }
        if (parent[j] != -1) anc[j] = parent[j];
    }
    for (int32_t j = 0; j < n; ++j)
        if (parent[j] != -1) count[parent[j]] += count[j];
    free(w);
    return LDLT_OK;
}
