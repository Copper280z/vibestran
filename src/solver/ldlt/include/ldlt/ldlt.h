#ifndef LDLT_H
#define LDLT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ldlt_symbolic ldlt_symbolic;
typedef struct ldlt_numeric  ldlt_numeric;

typedef enum {
    LDLT_OK = 0,
    LDLT_ERR_INPUT,
    LDLT_ERR_NOMEM,
    LDLT_ERR_INDEFINITE,
    LDLT_ERR_SINGULAR,
    LDLT_ERR_UNSUPPORTED,
    LDLT_ERR_NUMERIC,   /* a NaN or infinity turned up during factorization */
} ldlt_status;

typedef enum {
    LDLT_ORDER_NATURAL = 0,
    LDLT_ORDER_AMD,
    LDLT_ORDER_METIS,
    LDLT_ORDER_USER,
    LDLT_ORDER_AUTO,    /* METIS if compiled in, else AMD, else natural */
} ldlt_ordering;

typedef enum {
    LDLT_PIVOT_NONE = 0,
    LDLT_PIVOT_THRESHOLD,     /* 1x1 and 2x2 pivots with a threshold test, on a scaled matrix */
    LDLT_PIVOT_BUNCH_KAUFMAN = LDLT_PIVOT_THRESHOLD, /* the former name */
} ldlt_pivot_strategy;

typedef struct {
    ldlt_ordering ordering;
    const int32_t *user_perm;       /* used when ordering == LDLT_ORDER_USER */
    int32_t  relax_supernodes;      /* amalgamation slack; 0 disables relaxation */
    int32_t  nthreads;              /* 0 => OpenMP default */
    ldlt_pivot_strategy pivot;
    double   pivot_threshold;       /* u: entries of L stay under 1/u. <= 0 => 0.01, capped at 0.5 */
    int      require_pd;            /* if non-zero, fail on non-positive D */
} ldlt_options;

void ldlt_options_default(ldlt_options *opt);

/* Input: lower triangle of symmetric matrix in CSC, strictly increasing row indices per
   column, Ap[0] == 0. Anything else returns LDLT_ERR_INPUT. An ordering that is not
   compiled in returns LDLT_ERR_UNSUPPORTED. */
ldlt_status ldlt_analyze(int32_t n,
                         const int32_t *Ap, const int32_t *Ai,
                         const ldlt_options *opt,
                         ldlt_symbolic **out);

ldlt_status ldlt_factorize(const ldlt_symbolic *S,
                           const int32_t *Ap, const int32_t *Ai,
                           const double  *Ax,
                           const ldlt_options *opt,
                           ldlt_numeric **out);

/* Without pivoting, a pivot that comes out exactly zero is held rather than an error: that
   unknown is fixed at zero, as if constrained, and is reported by ldlt_get_held. Set
   require_pd to get LDLT_ERR_INDEFINITE instead. Check the residual of a solve if the
   matrix may be singular: holding a pivot says nothing about whether the load is carried.
   With pivoting, a pivot is held when its whole column is negligible. */

/* B is column-major, n rows, nrhs cols, leading dim ldb. Solved in place. */
ldlt_status ldlt_solve(const ldlt_numeric *N,
                       int32_t nrhs, double *B, int32_t ldb);

/* Diagnostics. */
int32_t ldlt_n(const ldlt_symbolic *S);
int32_t ldlt_nsuper(const ldlt_symbolic *S);
int64_t ldlt_lnz(const ldlt_symbolic *S);
double  ldlt_flops(const ldlt_symbolic *S);
int32_t ldlt_nlevels(const ldlt_symbolic *S);
int32_t ldlt_max_level_width(const ldlt_symbolic *S);
ldlt_ordering ldlt_ordering_used(const ldlt_symbolic *S);

/* What a factorization found. */
typedef struct {
    int32_t n_positive, n_negative; /* signs of D; by Sylvester's law, the matrix's inertia */
    int32_t n_zero;                 /* held pivots */
    int32_t n_2x2;                  /* 2 x 2 pivots, each counted once */
    int32_t n_delayed;              /* times an unknown was passed up to a parent front */
    int64_t lnz;                    /* stored entries of L, diagonal included */
    double  max_ratio;              /* largest |A_ii / D_i| over 1 x 1 pivots */
    int32_t max_ratio_row;          /* where, in the caller's numbering; -1 if none */
} ldlt_factor_info;

void ldlt_get_info(const ldlt_numeric *N, ldlt_factor_info *info);

/* A = P^T L D L^T P with P the elimination order: perm[k] is the row of A eliminated k-th.
   Pivoting changes the order, so it is read from the factors. */
void ldlt_get_perm(const ldlt_numeric *N, int32_t *perm);

/* D by row of A, length n each. d[i] is the pivot for row i, 0 if it was held. For a 2 x 2
   pivot on rows i and j: partner[i] = j, partner[j] = i and e[i] = e[j] is the off-diagonal;
   otherwise partner[i] = -1 and e[i] = 0. e and partner may be NULL. */
void ldlt_get_d(const ldlt_numeric *N, double *d, double *e, int32_t *partner);

/* Writes the rows of A with held pivots, ascending, and returns how many. rows may be NULL,
   to get the count. */
int32_t ldlt_get_held(const ldlt_numeric *N, int32_t *rows);

void ldlt_free_symbolic(ldlt_symbolic *S);
void ldlt_free_numeric(ldlt_numeric *N);

const char *ldlt_status_str(ldlt_status s);

#ifdef __cplusplus
}
#endif
#endif /* LDLT_H */
