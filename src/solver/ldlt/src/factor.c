/* Multifrontal LDL^T: each supernode assembles its front from its entries of A and its
   children's contribution blocks, eliminates its columns, and hands what is left of the
   front to its parent.

   Parallelism has two phases. The lower part of the tree splits into independent subtrees,
   run by OpenMP threads with a single-threaded BLAS, so no thread waits on another and
   each one works down a subtree in postorder. What is left is the top of the tree, where
   there are few but large fronts: those run one at a time with a threaded BLAS. */

#include "internal.h"
#include "blas_shim.h"
#include <math.h>
#include <stdio.h>

#if LDLT_HAVE_OPENMP
#include <omp.h>
#endif

#if LDLT_HAVE_OPENBLAS
extern void openblas_set_num_threads(int);
extern int openblas_get_num_threads(void);
#endif

#if defined(LDLT_BLAS_ACCELERATE) && defined(__clang__)
#if __has_include(<vecLib/thread_api.h>)
#include <vecLib/thread_api.h>
#define LDLT_ACCELERATE_THREAD_API 1
#endif
#endif

/* A subtree is split off when it holds at most this share of one thread's work. More
   subtrees than threads is what lets dynamic scheduling even out their sizes. */
#define SUBTREES_PER_THREAD 4
#define PARALLEL_ENTRIES 2000000  /* entries touched before a plain loop is worth threading */
/* A factorization below this stays on one thread. On 27-point grids on a Ryzen 7 5700, one thread
   is faster than eight up to about 5e6 flops and within a quarter of them at 2e7, a couple of
   milliseconds' work. With three such programs sharing the cores, a team is 30 to 100 times slower. */
#define SERIAL_FLOPS 2e7
/* The start of the top fronts' buffers, which small fronts all share, gets its faults over with
   up front. Beyond this, the large fronts that reach it fill them in parallel anyway, and
   touching it all would keep memory resident that the upper triangles never use. */
#define TOUCH_AHEAD ((size_t)128 << 20)

void ldlt_blas_this_thread_single(void)
{
#if LDLT_ACCELERATE_THREAD_API
    if (__builtin_available(macOS 15.0, *)) BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
#endif
}

void ldlt_blas_this_thread_multi(void)
{
#if LDLT_ACCELERATE_THREAD_API
    if (__builtin_available(macOS 15.0, *)) BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
#endif
}

/* OpenBLAS's thread count is global; Accelerate's is per thread (see above). Any other
   threaded BLAS must be limited by the caller during the subtree phase. */
int ldlt_blas_global_threads(int n)
{
#if LDLT_HAVE_OPENBLAS
    int old = openblas_get_num_threads();
    if (n > 0) openblas_set_num_threads(n);
    return old;
#else
    (void)n;
    return 0;
#endif
}

void ldlt_schedule_free(ldlt_schedule *P)
{
    free(P->first); free(P->root); free(P->sub_stack); free(P->top);
    memset(P, 0, sizeof(*P));
}

typedef struct { double work; int32_t root; } subtree_key;

static int cmp_subtree(const void *a, const void *b)
{
    double x = ((const subtree_key*)a)->work, y = ((const subtree_key*)b)->work;
    return x < y ? 1 : (x > y ? -1 : 0);
}

ldlt_status ldlt_schedule_build(const ldlt_symbolic *S, int nth, ldlt_schedule *P)
{
    int32_t ns = S->nsuper;
    size_t uns = (size_t)ns;
    memset(P, 0, sizeof(*P));
    double *total = (double*)ldlt_alloc_array(uns, sizeof(double));
    int32_t *first = (int32_t*)ldlt_alloc_array(uns, sizeof(int32_t));
    subtree_key *keys = (subtree_key*)ldlt_alloc_array(uns, sizeof(subtree_key));
    P->first = (int32_t*)ldlt_alloc_array(uns, sizeof(int32_t));
    P->root = (int32_t*)ldlt_alloc_array(uns, sizeof(int32_t));
    P->sub_stack = (size_t*)ldlt_alloc_array(uns, sizeof(size_t));
    P->top = (int32_t*)ldlt_alloc_array(uns, sizeof(int32_t));
    if (!total || !first || !keys || !P->first || !P->root || !P->sub_stack || !P->top) {
        free(total); free(first); free(keys);
        ldlt_schedule_free(P);
        return LDLT_ERR_NOMEM;
    }
    double all = 0.0;
    for (int32_t s = 0; s < ns; ++s) { total[s] = 0.0; first[s] = s; }
    for (int32_t s = 0; s < ns; ++s) {
        double k = S->super[s].width, m = S->super[s].nrows_below, R = k + m;
        /* Flops, plus moving the contribution block, which runs at memory speed. */
        double w = k*R*R - k*k*R + k*k*k/3.0 + 4.0*m*m + 200.0;
        total[s] += w;
        all += w;
        int32_t p = S->super_parent[s];
        if (p >= 0) {
            total[p] += total[s];
            if (first[s] < first[p]) first[p] = first[s];
        }
    }
    double limit = all / (double)((nth > 1 ? nth : 1) * SUBTREES_PER_THREAD);
    int32_t nkeys = 0;
    if (nth > 1) {
        for (int32_t s = 0; s < ns; ++s) {
            int32_t p = S->super_parent[s];
            if (total[s] <= limit && (p < 0 || total[p] > limit)) {
                keys[nkeys].work = total[s];
                keys[nkeys].root = s;
                nkeys++;
            }
        }
        qsort(keys, (size_t)nkeys, sizeof(subtree_key), cmp_subtree);
    }
    for (int32_t i = 0; i < nkeys; ++i) {
        int32_t root = keys[i].root;
        P->root[i] = root;
        P->first[i] = first[root];
        /* A subtree's own blocks form a stack too. Its last goes on the heap. */
        size_t in_use = 0, most = 0;
        for (int32_t s = first[root]; s < root; ++s) {
            for (int32_t cp = S->child_ptr[s]; cp < S->child_ptr[s+1]; ++cp) {
                size_t cm = (size_t)S->super[S->child_idx[cp]].nrows_below;
                in_use -= cm * (cm + 1) / 2;
            }
            size_t m = (size_t)S->super[s].nrows_below;
            in_use += m * (m + 1) / 2;
            if (in_use > most) most = in_use;
        }
        P->sub_stack[i] = most;
    }
    P->nsub = nkeys;
    for (int32_t s = 0; s < ns; ++s)
        if (nth <= 1 || total[s] > limit) P->top[P->ntop++] = s;

    /* The top fronts' buffers are set up once, so each page faults only once however many
       fronts use it. A block is released before its parent's is written, so contribution
       blocks form a stack in postorder. */
    size_t held = 0;
    for (int32_t t = 0; t < P->ntop; ++t) {
        int32_t s = P->top[t];
        int32_t m = S->super[s].nrows_below, R = S->super[s].width + m;
        if (m > P->top_max_rows) P->top_max_rows = m;
        if (R > P->top_max_front) P->top_max_front = R;
        for (int32_t cp = S->child_ptr[s]; cp < S->child_ptr[s+1]; ++cp) {
            int32_t c = S->child_idx[cp];
            if (nth <= 1 || total[c] > limit) {
                size_t cm = (size_t)S->super[c].nrows_below;
                held -= cm * (cm + 1) / 2;
            }
        }
        held += (size_t)m * ((size_t)m + 1) / 2;
        if (held > P->top_stack) P->top_stack = held;
    }
    free(total); free(first); free(keys);
    return LDLT_OK;
}

/* Page faults on fresh memory are by far the slowest part of assembling a large front if
   they all land on one thread, so take them in parallel first. */
static void ldlt_touch(void *ptr, size_t bytes, int nth)
{
    const size_t chunk = (size_t)1 << 21;
    int64_t nchunks = (int64_t)((bytes + chunk - 1) / chunk);
    char *base = (char*)ptr;
    (void)nth;
#if LDLT_HAVE_OPENMP
    #pragma omp parallel for if(nth > 1 && nchunks > 1) num_threads(nth) schedule(dynamic, 1)
#endif
    for (int64_t i = 0; i < nchunks; ++i) {
        size_t off = (size_t)i * chunk;
        memset(base + off, 0, bytes - off < chunk ? bytes - off : chunk);
    }
}

/* Between subtrees the stack is empty, so nothing in it needs to survive this. */
ldlt_status ldlt_ws_reserve_stack(ldlt_thread_ws *ws, size_t doubles)
{
    ws->stack_top = 0;
    return ldlt_ws_doubles(&ws->stack, &ws->stack_cap, doubles) || doubles == 0
               ? LDLT_OK : LDLT_ERR_NOMEM;
}

ldlt_status ldlt_ws_prepare_top(ldlt_thread_ws *ws, int32_t n, size_t C_doubles,
                                const ldlt_schedule *P, int nth)
{
    memset(ws, 0, sizeof(*ws));
    ws->nth = nth;
    ws->rowmap = (int32_t*)ldlt_alloc_array((size_t)n, sizeof(int32_t));
    if (!ws->rowmap) return LDLT_ERR_NOMEM;
    if (C_doubles > 0) {
        ws->C = (double*)ldlt_alloc_array(C_doubles, sizeof(double));
        if (!ws->C) return LDLT_ERR_NOMEM;
        ws->C_cap = C_doubles;
        ldlt_touch(ws->C, C_doubles * sizeof(double) < TOUCH_AHEAD ? C_doubles * sizeof(double)
                                                                   : TOUCH_AHEAD, nth);
    }
    if (P->top_stack > 0) {
        ws->stack = (double*)ldlt_alloc_array(P->top_stack, sizeof(double));
        if (!ws->stack) return LDLT_ERR_NOMEM;
        ws->stack_cap = P->top_stack;
        ldlt_touch(ws->stack, P->top_stack * sizeof(double) < TOUCH_AHEAD
                                  ? P->top_stack * sizeof(double) : TOUCH_AHEAD, nth);
    }
    return LDLT_OK;
}

/* A contribution block that lived in the stack only lowers *low. */
void ldlt_contrib_release(const ldlt_thread_ws *ws, double *A, size_t *low)
{
    uintptr_t a = (uintptr_t)A, b = (uintptr_t)ws->stack;
    if (ws->stack && a >= b && a < b + ws->stack_cap * sizeof(double)) {
        size_t off = (a - b) / sizeof(double);
        if (off < *low) *low = off;
        return;
    }
    free(A);
}

static size_t packed_lower_col(int32_t n, int32_t j)
{
    return (size_t)j * (size_t)n - ((size_t)j * (size_t)(j - 1)) / 2;
}

/* Adds a packed lower contribution block into a front whose first k columns are the panel
   P (leading dimension R) and whose rest is C (first row of C is front row k). emap gives
   each block row's front position and must be increasing. It is mostly long contiguous
   runs, so the additions are run by run. */
ldlt_status ldlt_extend_add(double *P, int32_t R, int32_t k, double *C, int32_t ldC,
                            const double *src, int32_t nc, const int32_t *emap,
                            ldlt_thread_ws *ws)
{
    if (nc <= 0) return LDLT_OK;
    size_t need = 2 * (size_t)nc + 2;
    if (ws->runs_cap < need) {
        free(ws->runs);
        ws->runs = (int32_t*)ldlt_alloc_array(need, sizeof(int32_t));
        ws->runs_cap = ws->runs ? need : 0;
        if (!ws->runs) return LDLT_ERR_NOMEM;
    }
    int32_t *rs = ws->runs, *run_of = ws->runs + (size_t)nc + 1;
    int32_t nr = 0;
    rs[0] = 0;
    run_of[0] = 0;
    for (int32_t i = 1; i < nc; ++i) {
        if (emap[i] != emap[i-1] + 1) rs[++nr] = i;
        run_of[i] = nr;
    }
    rs[++nr] = nc;

    int big = ws->nth > 1 && (int64_t)nc * nc / 2 >= PARALLEL_ENTRIES;
    (void)big;
#if LDLT_HAVE_OPENMP
    #pragma omp parallel for if(big) num_threads(ws->nth) schedule(dynamic, 16)
#endif
    for (int32_t j = 0; j < nc; ++j) {
        size_t off = packed_lower_col(nc, j) - (size_t)j;
        int32_t fj = emap[j];
        double *dst;
        int32_t shift;
        if (fj < k) { dst = P + (size_t)fj*R; shift = 0; }
        else        { dst = C + (size_t)(fj - k)*ldC; shift = k; }
        int32_t i = j;
        for (int32_t t = run_of[j]; t < nr; ++t) {
            int32_t end = rs[t+1];
            int64_t d = (int64_t)emap[i] - shift - i;
            for (; i < end; ++i) dst[i + d] += src[off + i];
        }
    }
    return LDLT_OK;
}

ldlt_status ldlt_pack_lower(const double *C, int32_t ldC, int32_t m, double **out,
                            ldlt_thread_ws *ws)
{
    size_t count = (size_t)m * ((size_t)m + 1) / 2;
    double *A;
    if (ws->stack && !ws->blocks_to_heap && count <= ws->stack_cap - ws->stack_top) {
        A = ws->stack + ws->stack_top;
        ws->stack_top += count;
    } else {
        A = (double*)ldlt_alloc_array(count, sizeof(double));
        if (!A) return LDLT_ERR_NOMEM;
    }
    int nth = ws->nth;
    int big = nth > 1 && (int64_t)m * m / 2 >= PARALLEL_ENTRIES;
    (void)big; (void)nth;
#if LDLT_HAVE_OPENMP
    #pragma omp parallel for if(big) num_threads(nth) schedule(dynamic, 64)
#endif
    for (int32_t j = 0; j < m; ++j)
        memcpy(A + packed_lower_col(m, j), C + j + (size_t)j*ldC,
               (size_t)(m - j) * sizeof(double));
    *out = A;
    return LDLT_OK;
}

void ldlt_zero_lower(double *C, int32_t ldC, int32_t m, int nth)
{
    int big = nth > 1 && (int64_t)m * m / 2 >= PARALLEL_ENTRIES;
    (void)big; (void)nth;
#if LDLT_HAVE_OPENMP
    #pragma omp parallel for if(big) num_threads(nth) schedule(dynamic, 64)
#endif
    for (int32_t j = 0; j < m; ++j)
        memset(C + j + (size_t)j*ldC, 0, (size_t)(m - j) * sizeof(double));
}

static ldlt_status factor_node(const ldlt_symbolic *S, ldlt_numeric *N, int32_t s,
                               double **contrib, const double *Bx, int require_pd,
                               ldlt_thread_ws *ws)
{
    const ldlt_super *Ss = &S->super[s];
    int32_t fc = Ss->first_col, k = Ss->width, m = Ss->nrows_below, R = k + m;
    const int32_t *rb = S->rows_pool + Ss->row_off;
    double *P = N->L + S->panel_off[s];
    double *C = NULL;
    int32_t *rowmap = ws->rowmap;

    if (m > 0) {
        C = ldlt_ws_doubles(&ws->C, &ws->C_cap, (size_t)m * m);
        if (!C) return LDLT_ERR_NOMEM;
        ldlt_zero_lower(C, m, m, ws->nth);
    }
    if (ws->nth > 1 && (int64_t)R * k >= PARALLEL_ENTRIES / 8)
        ldlt_touch(P, (size_t)R * k * sizeof(double), ws->nth);
    for (int32_t i = 0; i < k; ++i) rowmap[fc + i] = i;
    for (int32_t i = 0; i < m; ++i) rowmap[rb[i]] = k + i;

    /* The panel started out zero. */
    for (int32_t lc = 0; lc < k; ++lc) {
        double *col = P + (size_t)lc*R;
        for (int32_t p = S->Bp[fc + lc]; p < S->Bp[fc + lc + 1]; ++p)
            col[rowmap[S->Bi[p]]] = Bx[p];
    }
    size_t low = ws->stack_top;
    for (int32_t cp = S->child_ptr[s]; cp < S->child_ptr[s+1]; ++cp) {
        int32_t c = S->child_idx[cp];
        const ldlt_super *Sc = &S->super[c];
        ldlt_status st = ldlt_extend_add(P, R, k, C, m, contrib[c], Sc->nrows_below,
                                         S->edge_pos + Sc->row_off, ws);
        if (st != LDLT_OK) return st;
        ldlt_contrib_release(ws, contrib[c], &low);
        contrib[c] = NULL;
    }
    ws->stack_top = low;

    ldlt_status st = ldlt_front_nopiv(P, R, k, N->D + fc, C, m, require_pd, ws);
    if (st != LDLT_OK) return st;
    if (m > 0) return ldlt_pack_lower(C, m, m, &contrib[s], ws);
    return LDLT_OK;
}

static ldlt_status factorize_nopiv_phases(const ldlt_symbolic *S, ldlt_numeric *N,
                                          const double *Bx, const ldlt_options *opt, int nth,
                                          const ldlt_schedule *P, double **contrib,
                                          int top_blas_threads)
{
    ldlt_status status = LDLT_OK;
    int require_pd = opt->require_pd;
    if (P->nsub > 0) {
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
                    st = factor_node(S, N, s, contrib, Bx, require_pd, &ws);
                }
                if (st != LDLT_OK) {
                    /* What's in the stack goes when the thread's buffers do. */
                    size_t unused = 0;
                    for (int32_t s = P->first[t]; s <= P->root[t]; ++s) {
                        if (contrib[s]) ldlt_contrib_release(&ws, contrib[s], &unused);
                        contrib[s] = NULL;
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
        if (status != LDLT_OK) return status;
    }
    ldlt_blas_global_threads(top_blas_threads);
    ldlt_thread_ws ws;
    size_t mm = (size_t)P->top_max_rows;
    status = ldlt_ws_prepare_top(&ws, S->n, mm * mm, P, nth);
    for (int32_t t = 0; t < P->ntop && status == LDLT_OK; ++t)
        status = factor_node(S, N, P->top[t], contrib, Bx, require_pd, &ws);
    /* A block left in the stack is gone with it, so nothing may free it afterwards. */
    for (int32_t s = 0; s < S->nsuper; ++s) {
        size_t unused = 0;
        if (contrib[s]) ldlt_contrib_release(&ws, contrib[s], &unused);
        contrib[s] = NULL;
    }
    ldlt_ws_free(&ws);
    return status;
}

/* Inertia and the largest |A_ii / D_i| from D. */
static void finish_info_nopiv(const ldlt_symbolic *S, ldlt_numeric *N, const double *Bx)
{
    ldlt_factor_info *fi = &N->info;
    for (int32_t j = 0; j < S->n; ++j) {
        double d = N->D[j];
        if (isinf(d)) { fi->n_zero++; continue; }
        if (d > 0.0) fi->n_positive++; else fi->n_negative++;
        double aii = S->Bdiag[j] >= 0 ? fabs(Bx[S->Bdiag[j]]) : 0.0;
        double r = aii / fabs(d);
        if (r > fi->max_ratio) { fi->max_ratio = r; fi->max_ratio_row = S->perm[j]; }
    }
}

static ldlt_status remap_values(const ldlt_symbolic *S, const int32_t *Ap, const int32_t *Ai,
                                const double *Ax, int nth, double **Bx_out)
{
    int32_t n = S->n;
    if (Ap[n] != S->nnz) return LDLT_ERR_INPUT;
    double *Bx = (double*)ldlt_alloc_array((size_t)S->nnz, sizeof(double));
    if (!Bx) return LDLT_ERR_NOMEM;
    int bad = 0;
    (void)nth;
#if LDLT_HAVE_OPENMP
    #pragma omp parallel for schedule(static, 4096) reduction(|:bad) num_threads(nth) \
        if(nth > 1 && S->nnz >= PARALLEL_ENTRIES)
#endif
    for (int32_t j = 0; j < n; ++j) {
        int32_t nj = S->iperm[j];
        for (int32_t p = Ap[j]; p < Ap[j+1]; ++p) {
            int32_t ni = S->iperm[Ai[p]];
            int32_t q = S->Bmap[p];
            /* The pattern must be the one that was analyzed. */
            if (S->Bi[q] != (ni > nj ? ni : nj)) { bad = 1; continue; }
            Bx[q] = Ax[p];
        }
    }
    if (bad) { free(Bx); return LDLT_ERR_INPUT; }
    *Bx_out = Bx;
    return LDLT_OK;
}

ldlt_status ldlt_factorize(const ldlt_symbolic *S,
                           const int32_t *Ap, const int32_t *Ai, const double *Ax,
                           const ldlt_options *opt, ldlt_numeric **out)
{
    if (!out) return LDLT_ERR_INPUT;
    *out = NULL;
    if (!S || !Ap || (S->n > 0 && (!Ai || !Ax))) return LDLT_ERR_INPUT;
    ldlt_options local;
    ldlt_options_default(&local);
    if (opt) local = *opt;
    if (local.pivot != LDLT_PIVOT_NONE && local.pivot != LDLT_PIVOT_THRESHOLD)
        return LDLT_ERR_UNSUPPORTED;

#if LDLT_HAVE_OPENMP
    int nth = local.nthreads > 0 ? local.nthreads : omp_get_max_threads();
#else
    int nth = 1;
#endif
    if (nth < 1 || S->flops < SERIAL_FLOPS) nth = 1;

    double *Bx = NULL;
    ldlt_status st = remap_values(S, Ap, Ai, Ax, nth, &Bx);
    if (st != LDLT_OK) return st;

    ldlt_numeric *N = (ldlt_numeric*)ldlt_calloc_array(1, sizeof(*N));
    double **contrib = NULL;
    ldlt_schedule P;
    memset(&P, 0, sizeof(P));
    st = LDLT_ERR_NOMEM;
    if (!N) goto done;
    N->S = S;
    N->info.lnz = S->lnz;
    N->info.max_ratio_row = -1;
    st = ldlt_schedule_build(S, nth, &P);
    if (st != LDLT_OK) goto done;

    /* The top of the tree gets the requested thread count, or OpenBLAS's own setting. */
    int saved_blas = ldlt_blas_global_threads(0);
    int top_blas = nth == 1 ? 1 : local.nthreads > 0 ? local.nthreads : saved_blas;
    if (local.pivot == LDLT_PIVOT_THRESHOLD) {
        ldlt_blas_global_threads(1);
        st = ldlt_factorize_threshold(S, N, Bx, &local, nth, &P, top_blas);
    } else {
        int64_t Lsz = S->panel_off[S->nsuper];
        uint64_t avail = ldlt_available_memory_bytes();
        uint64_t lbytes = (uint64_t)Lsz * sizeof(double);
        if (avail > 0 && (long double)lbytes > (long double)avail * 0.90L) {
            fprintf(stderr,
                    "ldlt: warning: numeric factor storage requires %.2f GB, available %.2f GB"
                    " -- performance may suffer due to swapping.\n",
                    (double)lbytes / 1073741824.0, (double)avail / 1073741824.0);
        }
        N->L = (double*)ldlt_calloc_array((size_t)Lsz, sizeof(double));
        N->D = (double*)ldlt_alloc_array((size_t)S->n, sizeof(double));
        contrib = (double**)ldlt_calloc_array((size_t)S->nsuper, sizeof(double*));
        st = LDLT_ERR_NOMEM;
        if (N->L && N->D && contrib) {
            if (P.nsub > 0) ldlt_blas_global_threads(1);
            st = factorize_nopiv_phases(S, N, Bx, &local, nth, &P, contrib, top_blas);
        }
        if (st == LDLT_OK) finish_info_nopiv(S, N, Bx);
    }
    ldlt_blas_global_threads(saved_blas);

done:
    if (contrib) {
        for (int32_t s = 0; s < S->nsuper; ++s) free(contrib[s]);
        free(contrib);
    }
    ldlt_schedule_free(&P);
    free(Bx);
    if (st != LDLT_OK) {
        ldlt_free_numeric(N);
        return st;
    }
    *out = N;
    return LDLT_OK;
}
