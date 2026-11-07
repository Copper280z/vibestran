#include "internal.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

void *ldlt_alloc_array(size_t count, size_t elem) {
    if (count == 0 || elem == 0) { count = 1; elem = 1; }
    if (count > SIZE_MAX / elem) return NULL;
    return malloc(count * elem);
}

void *ldlt_calloc_array(size_t count, size_t elem) {
    if (count == 0 || elem == 0) { count = 1; elem = 1; }
    return calloc(count, elem);
}

uint64_t ldlt_available_memory_bytes(void) {
#if defined(_WIN32)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (!GlobalMemoryStatusEx(&statex))
        return 0;
    return (uint64_t)statex.ullAvailPhys;
#elif defined(__APPLE__)
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    vm_statistics64_data_t vmstat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

    if (host_page_size(host, &page_size) != KERN_SUCCESS) {
        mach_port_deallocate(mach_task_self(), host);
        return 0;
    }
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vmstat,
                          &count) != KERN_SUCCESS) {
        mach_port_deallocate(mach_task_self(), host);
        return 0;
    }

    uint64_t pages = (uint64_t)vmstat.free_count
                   + (uint64_t)vmstat.inactive_count
                   + (uint64_t)vmstat.speculative_count;
    mach_port_deallocate(mach_task_self(), host);
    return pages * (uint64_t)page_size;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) != 0)
        return 0;
    return (uint64_t)si.freeram * (uint64_t)si.mem_unit;
#else
    return 0;
#endif
}

double ldlt_wall_time_seconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int initialized = 0;
    LARGE_INTEGER now;
    if (!initialized) {
        if (!QueryPerformanceFrequency(&freq))
            return (double)clock() / (double)CLOCKS_PER_SEC;
        initialized = 1;
    }
    if (!QueryPerformanceCounter(&now))
        return (double)clock() / (double)CLOCKS_PER_SEC;
    return (double)now.QuadPart / (double)freq.QuadPart;
#elif defined(TIME_UTC)
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) == TIME_UTC)
        return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
    return (double)clock() / (double)CLOCKS_PER_SEC;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

void ldlt_options_default(ldlt_options *opt) {
    opt->ordering = LDLT_ORDER_AUTO;
    opt->user_perm = NULL;
    opt->relax_supernodes = 16;
    opt->nthreads = 0;
    opt->pivot = LDLT_PIVOT_NONE;
    opt->pivot_threshold = 0.0;
    opt->require_pd = 0;
}

const char *ldlt_status_str(ldlt_status s) {
    switch (s) {
    case LDLT_OK: return "ok";
    case LDLT_ERR_INPUT: return "invalid input";
    case LDLT_ERR_NOMEM: return "out of memory";
    case LDLT_ERR_INDEFINITE: return "matrix indefinite";
    case LDLT_ERR_SINGULAR: return "matrix singular";
    case LDLT_ERR_NUMERIC: return "non-finite value during factorization";
    case LDLT_ERR_UNSUPPORTED: return "feature not supported";
    }
    return "unknown";
}

int32_t ldlt_n(const ldlt_symbolic *S)      { return S ? S->n : 0; }
int32_t ldlt_nsuper(const ldlt_symbolic *S) { return S ? S->nsuper : 0; }
int64_t ldlt_lnz(const ldlt_symbolic *S)    { return S ? S->lnz : 0; }
double  ldlt_flops(const ldlt_symbolic *S)  { return S ? S->flops : 0.0; }
int32_t ldlt_nlevels(const ldlt_symbolic *S) { return S ? S->nlevels : 0; }
int32_t ldlt_max_level_width(const ldlt_symbolic *S) { return S ? S->max_level_width : 0; }
ldlt_ordering ldlt_ordering_used(const ldlt_symbolic *S) { return S ? S->ordering : LDLT_ORDER_NATURAL; }

void ldlt_get_info(const ldlt_numeric *N, ldlt_factor_info *info) {
    if (N && info) *info = N->info;
}

void ldlt_get_perm(const ldlt_numeric *N, int32_t *perm) {
    if (!N || !perm) return;
    const ldlt_symbolic *S = N->S;
    if (!N->pivoted) {
        memcpy(perm, S->perm, (size_t)S->n * sizeof(int32_t));
        return;
    }
    int32_t k = 0;
    for (int32_t s = 0; s < S->nsuper; ++s)
        for (int32_t c = 0; c < N->nodes[s].nelim; ++c)
            perm[k++] = S->perm[N->nodes[s].vars[c]];
}

void ldlt_get_d(const ldlt_numeric *N, double *d, double *e, int32_t *partner) {
    if (!N || !d) return;
    const ldlt_symbolic *S = N->S;
    if (e) memset(e, 0, (size_t)S->n * sizeof(double));
    if (partner) for (int32_t i = 0; i < S->n; ++i) partner[i] = -1;
    if (!N->pivoted) {
        for (int32_t j = 0; j < S->n; ++j)
            d[S->perm[j]] = isinf(N->D[j]) ? 0.0 : N->D[j];
        return;
    }
    for (int32_t s = 0; s < S->nsuper; ++s) {
        const ldlt_pivot_node *node = &N->nodes[s];
        for (int32_t c = 0; c < node->nelim; ++c) {
            int32_t v = node->vars[c], row = S->perm[v];
            double sv = N->scale[v];
            d[row] = isinf(node->d[c]) ? 0.0 : node->d[c] / (sv * sv);
            if (node->e[c] != 0.0) {
                int32_t v1 = node->vars[c+1], row1 = S->perm[v1];
                double s1 = N->scale[v1];
                d[row1] = node->d[c+1] / (s1 * s1);
                if (e) e[row] = e[row1] = node->e[c] / (sv * s1);
                if (partner) { partner[row] = row1; partner[row1] = row; }
                ++c;
            }
        }
    }
}

static int cmp_i32(const void *a, const void *b) {
    int32_t x = *(const int32_t*)a, y = *(const int32_t*)b;
    return (x > y) - (x < y);
}

int32_t ldlt_get_held(const ldlt_numeric *N, int32_t *rows) {
    if (!N) return 0;
    const ldlt_symbolic *S = N->S;
    int32_t count = 0;
    if (!N->pivoted) {
        for (int32_t j = 0; j < S->n; ++j) {
            if (!isinf(N->D[j])) continue;
            if (rows) rows[count] = S->perm[j];
            count++;
        }
    } else {
        for (int32_t s = 0; s < S->nsuper; ++s) {
            const ldlt_pivot_node *node = &N->nodes[s];
            for (int32_t c = 0; c < node->nelim; ++c) {
                if (!isinf(node->d[c])) continue;
                if (rows) rows[count] = S->perm[node->vars[c]];
                count++;
            }
        }
    }
    if (rows) qsort(rows, (size_t)count, sizeof(int32_t), cmp_i32);
    return count;
}

void ldlt_free_symbolic(ldlt_symbolic *S) {
    if (!S) return;
    free(S->perm); free(S->iperm);
    free(S->parent);
    free(S->col_to_super);
    free(S->super);
    free(S->rows_pool);
    free(S->edge_pos);
    free(S->super_parent);
    free(S->child_ptr); free(S->child_idx);
    free(S->panel_off);
    free(S->Bp); free(S->Bi); free(S->Bmap); free(S->Bdiag);
    free(S);
}

void ldlt_free_numeric(ldlt_numeric *N) {
    if (!N) return;
    free(N->L); free(N->D); free(N->scale);
    if (N->nodes) {
        for (int32_t s = 0; s < N->S->nsuper; ++s) {
            free(N->nodes[s].vars); free(N->nodes[s].L);
            free(N->nodes[s].d); free(N->nodes[s].e);
        }
        free(N->nodes);
    }
    free(N);
}

