#ifndef LDLT_INTERNAL_H
#define LDLT_INTERNAL_H

#include "ldlt/ldlt.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef LDLT_USE_SYSTEM_BLAS
#define LDLT_USE_SYSTEM_BLAS 0
#endif
#ifndef LDLT_HAVE_AMD
#define LDLT_HAVE_AMD 0
#endif
#ifndef LDLT_HAVE_METIS
#define LDLT_HAVE_METIS 0
#endif
#ifndef LDLT_HAVE_OPENMP
#define LDLT_HAVE_OPENMP 0
#endif
#ifndef LDLT_USE_POSIX_SIGNALS
#define LDLT_USE_POSIX_SIGNALS 0
#endif
#ifndef LDLT_HAVE_LAPACK
#define LDLT_HAVE_LAPACK 0
#endif
#ifndef LDLT_HAVE_OPENBLAS
#define LDLT_HAVE_OPENBLAS 0
#endif

/* ---- ldlt_symbolic / ldlt_numeric internals ---- */

typedef struct {
    int32_t first_col;     /* global col index of first col in supernode */
    int32_t width;         /* k: number of cols */
    int32_t nrows_below;   /* m: rows strictly below the diagonal block */
    int64_t row_off;       /* offset into rows_pool and edge_pos, length nrows_below */
} ldlt_super;

struct ldlt_symbolic {
    int32_t n;
    int32_t nsuper;
    int32_t perm_identity;
    int32_t nlevels;
    int32_t max_level_width;
    ldlt_ordering ordering;/* the ordering actually used */
    int32_t *perm;         /* length n: perm[k] = original col placed at position k */
    int32_t *iperm;        /* inverse */
    int32_t *parent;       /* etree of permuted matrix, length n */
    int32_t *col_to_super; /* length n */
    ldlt_super *super;     /* length nsuper; supernodes are in postorder */
    int32_t *rows_pool;    /* concatenated sorted rows_below across supernodes */
    int64_t  rows_pool_len;
    int32_t *edge_pos;     /* for each rows_pool entry, its position in the parent's front */
    int32_t *super_parent; /* supernodal etree, length nsuper; -1 for roots */
    int32_t *child_ptr;    /* children of each supernode, length nsuper+1 */
    int32_t *child_idx;    /* child_ptr[nsuper] entries */
    /* Sizes: per-supernode L panel size = (width+nrows_below)*width doubles. */
    int64_t *panel_off;    /* length nsuper+1; offsets into a packed L array (in doubles) */
    int32_t  nnz;          /* entries of A */
    int32_t *Bp, *Bi;      /* pattern of P A P^T, lower triangle, rows unsorted */
    int32_t *Bmap;         /* entry p of A is entry Bmap[p] of B */
    int32_t *Bdiag;        /* position in B of each diagonal entry, -1 if A has none */
    int64_t  lnz;
    double   flops;        /* multiply-add pairs for an unpivoted factorization */
};

/* With pivoting, what each supernode eliminated. */
typedef struct {
    int32_t nelim;
    int32_t nrows;
    int32_t ndelay; /* columns passed to the parent */
    int32_t *vars;  /* length nrows: permuted-space id of each row */
    double *L;      /* nrows x nelim, unit lower, zero under a 2 x 2 block's first column */
    double *d;      /* length nelim */
    double *e;      /* length nelim: e[c] != 0 starts a 2 x 2 block at (c, c+1) */
} ldlt_pivot_node;

struct ldlt_numeric {
    const ldlt_symbolic *S; /* not owned */
    int32_t pivoted;
    double *L;              /* no pivoting: panels at panel_off */
    double *D;              /* no pivoting: by permuted column; +inf for a held pivot */
    ldlt_pivot_node *nodes; /* pivoting: one per supernode */
    double *scale;          /* pivoting: the symmetric scaling, by permuted column */
    ldlt_factor_info info;
};

/* ---- helpers ---- */
/* calloc and malloc with overflow checks; they never return NULL for a zero size. */
void *ldlt_alloc_array(size_t count, size_t elem);
void *ldlt_calloc_array(size_t count, size_t elem);
uint64_t ldlt_available_memory_bytes(void);
double ldlt_wall_time_seconds(void);

/* etree.c */
ldlt_status ldlt_etree(int32_t n, const int32_t *Ap, const int32_t *Ai, int32_t *parent);
ldlt_status ldlt_postorder(int32_t n, const int32_t *parent, const int32_t *weight,
                           int32_t *post);
ldlt_status ldlt_col_counts(int32_t n, const int32_t *Ap, const int32_t *Ai,
                            const int32_t *parent, const int32_t *post,
                            int32_t *colcount);

/* perm.c */
ldlt_status ldlt_permute_lower_pattern(int32_t n,
    const int32_t *Ap, const int32_t *Ai, const int32_t *iperm,
    int32_t **Bp_out, int32_t **Bi_out, int32_t **map_out);

/* ordering.c */
ldlt_status ldlt_compute_ordering(int32_t n, const int32_t *Ap, const int32_t *Ai,
                                  const ldlt_options *opt, int32_t *perm,
                                  ldlt_ordering *used);

/* symbolic.c */
ldlt_status ldlt_build_symbolic(int32_t n, const int32_t *Ap, const int32_t *Ai,
                                const ldlt_options *opt, ldlt_symbolic *S);

/* factor.c / factor_kernels.c / factor_pivot.c */

/* Per-thread buffers; they grow to the largest front that thread sees. */
typedef struct {
    int nth;            /* threads a kernel may start itself: above 1 only at the top of the tree */
    int32_t *rowmap;    /* length n */
    double *C; size_t C_cap;
    double *save; size_t save_cap;
    double *batch; size_t batch_cap;
    int32_t *runs; size_t runs_cap;
    int32_t *pos; size_t pos_cap;
    double *stack; size_t stack_cap, stack_top; /* contribution blocks, last in first out */
    int blocks_to_heap;  /* a subtree's last block outlives the thread's stack */
} ldlt_thread_ws;

/* Subtrees are the supernode ranges [first[t], root[t]], largest first. */
typedef struct {
    int32_t nsub;
    int32_t *first, *root;
    size_t *sub_stack;      /* doubles of contribution blocks held at once in each, unpivoted */
    int32_t ntop;
    int32_t *top;
    int32_t top_max_rows;   /* largest rows below, and largest front, among the top fronts */
    int32_t top_max_front;
    size_t top_stack;       /* doubles of contribution blocks held at once at the top, unpivoted */
} ldlt_schedule;

double *ldlt_ws_doubles(double **p, size_t *cap, size_t need);
void ldlt_ws_free(ldlt_thread_ws *ws);
ldlt_status ldlt_schedule_build(const ldlt_symbolic *S, int nth, ldlt_schedule *P);
void ldlt_schedule_free(ldlt_schedule *P);
void ldlt_blas_this_thread_single(void);
void ldlt_blas_this_thread_multi(void);
int ldlt_blas_global_threads(int n);
ldlt_status ldlt_extend_add(double *P, int32_t R, int32_t k, double *C, int32_t ldC,
                            const double *src, int32_t nc, const int32_t *emap,
                            ldlt_thread_ws *ws);
ldlt_status ldlt_pack_lower(const double *C, int32_t ldC, int32_t m, double **out,
                            ldlt_thread_ws *ws);
ldlt_status ldlt_ws_prepare_top(ldlt_thread_ws *ws, int32_t n, size_t C_doubles,
                                const ldlt_schedule *P, int nth);
ldlt_status ldlt_ws_reserve_stack(ldlt_thread_ws *ws, size_t doubles);
void ldlt_contrib_release(const ldlt_thread_ws *ws, double *A, size_t *low);
void ldlt_zero_lower(double *C, int32_t ldC, int32_t m, int nth);
ldlt_status ldlt_front_nopiv(double *P, int32_t R, int32_t k, double *D,
                             double *C, int32_t ldC, int require_pd, ldlt_thread_ws *ws);
ldlt_status ldlt_update_signed(double *C, int32_t ldC, int32_t m, int32_t k,
                               const double *L21, int32_t ld, const double *D,
                               ldlt_thread_ws *ws);
void ldlt_cholesky_to_ldl(double *P, int32_t ld, int32_t R, int32_t k, double *D, int nth);
int ldlt_try_potrf(double *P, int32_t ld, int32_t k, double *save);
ldlt_status ldlt_factorize_threshold(const ldlt_symbolic *S, ldlt_numeric *N, double *Bx,
                                     const ldlt_options *opt, int nth,
                                     const ldlt_schedule *P, int top_blas_threads);

/* io_mm.c (used by tests) */
ldlt_status ldlt_read_mm_symmetric(const char *path, int32_t *n_out,
                                   int32_t **Ap_out, int32_t **Ai_out,
                                   double **Ax_out);

#endif
