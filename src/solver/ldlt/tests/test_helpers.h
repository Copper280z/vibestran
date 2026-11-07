#pragma once
#include "ldlt/ldlt.h"
#include <vector>
#include <random>
#include <cmath>

struct CSCLower {
    int32_t n = 0;
    std::vector<int32_t> Ap, Ai;
    std::vector<double> Ax;
};

/* Build SPD tridiagonal with diag = 4, off-diag = -1. Lower triangle CSC. */
inline CSCLower make_tridiag_spd(int32_t n) {
    CSCLower A; A.n = n;
    A.Ap.assign(n+1, 0);
    /* col j has entries (j,j) and (j+1,j) if j<n-1 -> 2 entries, last col has 1 */
    for (int32_t j = 0; j < n; ++j) A.Ap[j+1] = A.Ap[j] + (j < n-1 ? 2 : 1);
    A.Ai.resize(A.Ap[n]); A.Ax.resize(A.Ap[n]);
    for (int32_t j = 0; j < n; ++j) {
        int32_t p = A.Ap[j];
        A.Ai[p] = j; A.Ax[p] = 4.0;
        if (j < n-1) { A.Ai[p+1] = j+1; A.Ax[p+1] = -1.0; }
    }
    return A;
}

/* Compute y = A x where A is symmetric, lower-triangle CSC. */
inline std::vector<double> spmv_sym(const CSCLower &A, const std::vector<double> &x) {
    std::vector<double> y(A.n, 0.0);
    for (int32_t j = 0; j < A.n; ++j) {
        for (int32_t p = A.Ap[j]; p < A.Ap[j+1]; ++p) {
            int32_t i = A.Ai[p];
            double v = A.Ax[p];
            y[i] += v * x[j];
            if (i != j) y[j] += v * x[i];
        }
    }
    return y;
}

inline double norm2(const std::vector<double> &v) {
    double s = 0.0; for (double x : v) s += x*x; return std::sqrt(s);
}
