#include "internal.h"
#include <limits.h>
#include <stdio.h>

#if LDLT_HAVE_AMD
#include <amd.h>
#endif
#if LDLT_HAVE_METIS
#include <metis.h>
#endif

#if LDLT_HAVE_METIS && LDLT_USE_POSIX_SIGNALS
#include <setjmp.h>
#include <signal.h>
#if defined(SIGBUS) && defined(SIGSEGV)
#define LDLT_METIS_SIGNAL_TRAP 1
#else
#define LDLT_METIS_SIGNAL_TRAP 0
#endif
#else
#define LDLT_METIS_SIGNAL_TRAP 0
#endif

#if LDLT_HAVE_METIS && LDLT_METIS_SIGNAL_TRAP
static sigjmp_buf g_metis_jmp;
static volatile sig_atomic_t g_metis_signal = 0;
static struct sigaction g_old_sigbus;
static struct sigaction g_old_sigsegv;

static void metis_fault_handler(int sig) {
    g_metis_signal = sig;
    siglongjmp(g_metis_jmp, 1);
}

static void install_metis_fault_handlers(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = metis_fault_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGBUS, &sa, &g_old_sigbus);
    sigaction(SIGSEGV, &sa, &g_old_sigsegv);
}

static void restore_metis_fault_handlers(void) {
    sigaction(SIGBUS, &g_old_sigbus, NULL);
    sigaction(SIGSEGV, &g_old_sigsegv, NULL);
}
#endif

#if LDLT_HAVE_METIS
static double metis_memory_factor(void) {
    const char *env = getenv("LDLT_METIS_MEMORY_FACTOR");
    if (!env || !*env) return 16.0;
    char *end = NULL;
    double v = strtod(env, &end);
    if (end == env)
        return 16.0;
    return v;
}

static int metis_memory_preflight(int32_t n, int32_t graph_nnz) {
    uint64_t avail = ldlt_available_memory_bytes();
    double factor = metis_memory_factor();
    if (avail == 0 || factor <= 0.0)
        return 1;

    long double graph_bytes =
        ((long double)n + 1.0L + (long double)graph_nnz + 2.0L*(long double)n)
        * (long double)sizeof(idx_t);
    long double need = graph_bytes * (long double)factor;
    long double limit = (long double)avail * 0.90L;
    if (need <= limit)
        return 1;

    fprintf(stderr,
            "ldlt: warning: METIS ordering memory guard: estimated %.2Lf GB exceeds "
            "available %.2f GB (n=%d, graph_nnz=%d, factor=%.1f) -- "
            "performance may suffer due to swapping. "
            "Set LDLT_METIS_MEMORY_FACTOR=0 to disable this guard.\n",
            need / 1073741824.0L,
            (double)avail / 1073741824.0,
            (int)n, (int)graph_nnz, factor);
    return 1;
}
#endif

static ldlt_status order_amd(int32_t n, const int32_t *Ap, const int32_t *Ai, int32_t *perm) {
#if LDLT_HAVE_AMD
    /* AMD takes the full pattern with sorted columns. Walking the lower triangle by column
       appends earlier columns first and then this column's own rows, so they stay sorted. */
    int32_t *Fp = (int32_t*)ldlt_calloc_array((size_t)n+1, sizeof(int32_t));
    if (!Fp) return LDLT_ERR_NOMEM;
    int64_t total = 0;
    for (int32_t j = 0; j < n; ++j)
        for (int32_t p = Ap[j]; p < Ap[j+1]; ++p) {
            int32_t i = Ai[p];
            Fp[j+1]++; total++;
            if (i != j) { Fp[i+1]++; total++; }
        }
    if (total > INT32_MAX) { free(Fp); return LDLT_ERR_NOMEM; }
    for (int32_t i = 0; i < n; ++i) Fp[i+1] += Fp[i];
    int32_t *Fi = (int32_t*)ldlt_alloc_array((size_t)Fp[n], sizeof(int32_t));
    int32_t *cur = (int32_t*)ldlt_alloc_array((size_t)n, sizeof(int32_t));
    if (!Fi || !cur) { free(Fp); free(Fi); free(cur); return LDLT_ERR_NOMEM; }
    if (n > 0) memcpy(cur, Fp, (size_t)n * sizeof(int32_t));
    for (int32_t j = 0; j < n; ++j)
        for (int32_t p = Ap[j]; p < Ap[j+1]; ++p) {
            int32_t i = Ai[p];
            Fi[cur[j]++] = i;
            if (i != j) Fi[cur[i]++] = j;
        }
    free(cur);
    int result = amd_order(n, Fp, Fi, perm, NULL, NULL);
    free(Fp); free(Fi);
    if (result == AMD_OUT_OF_MEMORY) return LDLT_ERR_NOMEM;
    return (result == AMD_OK || result == AMD_OK_BUT_JUMBLED) ? LDLT_OK : LDLT_ERR_INPUT;
#else
    (void)n; (void)Ap; (void)Ai; (void)perm;
    return LDLT_ERR_UNSUPPORTED;
#endif
}

static ldlt_status order_metis(int32_t n, const int32_t *Ap, const int32_t *Ai, int32_t *perm) {
#if LDLT_HAVE_METIS
    /* expand to full pattern (no diagonal) */
    int32_t *Fp = (int32_t*)calloc((size_t)n+1, sizeof(int32_t));
    if (!Fp) return LDLT_ERR_NOMEM;
    for (int32_t j = 0; j < n; ++j)
        for (int32_t p = Ap[j]; p < Ap[j+1]; ++p) {
            int32_t i = Ai[p];
            if (i == j) continue;
            if (Fp[j+1] == INT32_MAX || Fp[i+1] == INT32_MAX) {
                free(Fp);
                return LDLT_ERR_NOMEM;
            }
            Fp[j+1]++; Fp[i+1]++;
        }
    int64_t prefix = 0;
    for (int32_t i = 0; i < n; ++i) {
        prefix += Fp[i+1];
        if (prefix > INT32_MAX) {
            free(Fp);
            return LDLT_ERR_NOMEM;
        }
        Fp[i+1] = (int32_t)prefix;
    }
    metis_memory_preflight(n, Fp[n]);

    int32_t *Fi = (int32_t*)malloc((size_t)(Fp[n] > 0 ? Fp[n] : 1) * sizeof(int32_t));
    int32_t *cur = (int32_t*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    if (!Fi || !cur) {
        free(Fp);
        free(Fi);
        free(cur);
        return LDLT_ERR_NOMEM;
    }
    memcpy(cur, Fp, (size_t)n * sizeof(int32_t));
    for (int32_t j = 0; j < n; ++j)
        for (int32_t p = Ap[j]; p < Ap[j+1]; ++p) {
            int32_t i = Ai[p];
            if (i == j) continue;
            Fi[cur[j]++] = i; Fi[cur[i]++] = j;
        }
    free(cur);
    idx_t opts[METIS_NOPTIONS]; METIS_SetDefaultOptions(opts);
    idx_t *iperm = (idx_t*)malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    idx_t *p     = (idx_t*)malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    /* METIS may be built with 64-bit indices. */
    idx_t *xadj = (idx_t*)malloc(((size_t)n + 1) * sizeof(idx_t));
    idx_t *adjncy = (idx_t*)malloc((size_t)(Fp[n] > 0 ? Fp[n] : 1) * sizeof(idx_t));
    if (!iperm || !p || !xadj || !adjncy) {
        free(p); free(iperm); free(xadj); free(adjncy); free(Fp); free(Fi);
        return LDLT_ERR_NOMEM;
    }
    for (int32_t i = 0; i <= n; ++i) xadj[i] = Fp[i];
    for (int32_t i = 0; i < Fp[n]; ++i) adjncy[i] = Fi[i];
    free(Fi); Fi = NULL;
    idx_t nn = n;
    int rc;
#if LDLT_METIS_SIGNAL_TRAP
    g_metis_signal = 0;
    if (sigsetjmp(g_metis_jmp, 1) == 0) {
        install_metis_fault_handlers();
        rc = METIS_NodeND(&nn, xadj, adjncy, NULL, opts, p, iperm);
        restore_metis_fault_handlers();
    } else {
        restore_metis_fault_handlers();
        fprintf(stderr,
                "ldlt: METIS_NodeND trapped signal %d; treating ordering as out of memory.\n",
                (int)g_metis_signal);
        rc = METIS_ERROR_MEMORY;
    }
#else
    rc = METIS_NodeND(&nn, xadj, adjncy, NULL, opts, p, iperm);
#endif
    if (rc == METIS_OK) for (int32_t i = 0; i < n; ++i) perm[i] = (int32_t)p[i];
    free(p); free(iperm); free(xadj); free(adjncy); free(Fp);
    if (rc == METIS_OK) return LDLT_OK;
    return rc == METIS_ERROR_MEMORY ? LDLT_ERR_NOMEM : LDLT_ERR_INPUT;
#else
    (void)n; (void)Ap; (void)Ai; (void)perm;
    return LDLT_ERR_UNSUPPORTED;
#endif
}

/* A requested ordering that is not compiled in or fails is an error rather than a silent
   switch to no ordering at all, whose fill can be orders of magnitude worse. */
ldlt_status ldlt_compute_ordering(int32_t n, const int32_t *Ap, const int32_t *Ai,
                                  const ldlt_options *opt, int32_t *perm,
                                  ldlt_ordering *used) {
    ldlt_ordering which = opt->ordering;
    if (which == LDLT_ORDER_AUTO) {
#if LDLT_HAVE_METIS
        which = LDLT_ORDER_METIS;
#elif LDLT_HAVE_AMD
        which = LDLT_ORDER_AMD;
#else
        which = LDLT_ORDER_NATURAL;
#endif
    }
    *used = which;
    /* Nothing to order. METIS divides by the vertex count and dies with SIGFPE on x86 at n = 0. */
    if (n <= 1 && which != LDLT_ORDER_USER) {
        if (n == 1) perm[0] = 0;
        return LDLT_OK;
    }
    switch (which) {
    case LDLT_ORDER_NATURAL:
        for (int32_t i = 0; i < n; ++i) perm[i] = i;
        return LDLT_OK;
    case LDLT_ORDER_USER: {
        if (!opt->user_perm) return LDLT_ERR_INPUT;
        char *seen = (char*)ldlt_calloc_array((size_t)n, 1);
        if (!seen) return LDLT_ERR_NOMEM;
        for (int32_t i = 0; i < n; ++i) {
            int32_t v = opt->user_perm[i];
            if (v < 0 || v >= n || seen[v]) { free(seen); return LDLT_ERR_INPUT; }
            seen[v] = 1;
            perm[i] = v;
        }
        free(seen);
        return LDLT_OK;
    }
    case LDLT_ORDER_AMD:
        return order_amd(n, Ap, Ai, perm);
    case LDLT_ORDER_METIS:
        return order_metis(n, Ap, Ai, perm);
    case LDLT_ORDER_AUTO:
        break;
    }
    return LDLT_ERR_INPUT;
}
