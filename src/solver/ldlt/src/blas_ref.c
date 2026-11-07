/* Reference implementations of the small CBLAS subset we use.
   Compiled and linked only when LDLT_USE_SYSTEM_BLAS == 0.
   Triple-loop, single-threaded; correctness over speed. */

#include "blas_shim.h"

#if !LDLT_USE_SYSTEM_BLAS

static inline double *ELT(double *A, int i, int j, int ld) { return &A[i + (size_t)j*ld]; }
static inline const double *CELT(const double *A, int i, int j, int ld) { return &A[i + (size_t)j*ld]; }

void cblas_dscal(int N, double alpha, double *X, int incX) {
    for (int i = 0; i < N; ++i) X[i*incX] *= alpha;
}

void cblas_daxpy(int N, double alpha, const double *X, int incX, double *Y, int incY) {
    for (int i = 0; i < N; ++i) Y[i*incY] += alpha * X[i*incX];
}

void cblas_dgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                 int M, int N, int K, double alpha,
                 const double *A, int lda, const double *B, int ldb,
                 double beta, double *C, int ldc) {
    (void)order; /* assume ColMajor */
    /* Scale C by beta */
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i)
            *ELT(C,i,j,ldc) = beta * (*ELT(C,i,j,ldc));
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            double bkj;
            if (transB == CblasNoTrans) bkj = *CELT(B,k,j,ldb);
            else                         bkj = *CELT(B,j,k,ldb);
            double s = alpha * bkj;
            for (int i = 0; i < M; ++i) {
                double aik;
                if (transA == CblasNoTrans) aik = *CELT(A,i,k,lda);
                else                         aik = *CELT(A,k,i,lda);
                *ELT(C,i,j,ldc) += s * aik;
            }
        }
    }
}

void cblas_dsyrk(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                 int N, int K, double alpha,
                 const double *A, int lda, double beta, double *C, int ldc) {
    (void)order;
    if (uplo == CblasLower) {
        for (int j = 0; j < N; ++j) {
            for (int i = j; i < N; ++i) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) {
                    double aik = (trans == CblasNoTrans) ? *CELT(A,i,k,lda) : *CELT(A,k,i,lda);
                    double ajk = (trans == CblasNoTrans) ? *CELT(A,j,k,lda) : *CELT(A,k,j,lda);
                    s += aik * ajk;
                }
                *ELT(C,i,j,ldc) = beta * (*ELT(C,i,j,ldc)) + alpha * s;
            }
        }
    } else {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i <= j; ++i) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) {
                    double aik = (trans == CblasNoTrans) ? *CELT(A,i,k,lda) : *CELT(A,k,i,lda);
                    double ajk = (trans == CblasNoTrans) ? *CELT(A,j,k,lda) : *CELT(A,k,j,lda);
                    s += aik * ajk;
                }
                *ELT(C,i,j,ldc) = beta * (*ELT(C,i,j,ldc)) + alpha * s;
            }
        }
    }
}

/* Solves op(A) * X = alpha B (Left) or X * op(A) = alpha B (Right). In-place in B. */
void cblas_dtrsm(CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                 CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
                 int M, int N, double alpha,
                 const double *A, int lda, double *B, int ldb) {
    (void)order;
    if (alpha != 1.0) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i)
                *ELT(B,i,j,ldb) *= alpha;
    }
    int unit = (diag == CblasUnit);
    if (side == CblasLeft) {
        /* op(A) is M x M, B is M x N */
        if (uplo == CblasLower && transA == CblasNoTrans) {
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < M; ++i) {
                    double s = *ELT(B,i,j,ldb);
                    for (int k = 0; k < i; ++k)
                        s -= *CELT(A,i,k,lda) * (*ELT(B,k,j,ldb));
                    if (!unit) s /= *CELT(A,i,i,lda);
                    *ELT(B,i,j,ldb) = s;
                }
        } else if (uplo == CblasLower && transA == CblasTrans) {
            for (int j = 0; j < N; ++j)
                for (int i = M-1; i >= 0; --i) {
                    double s = *ELT(B,i,j,ldb);
                    for (int k = i+1; k < M; ++k)
                        s -= *CELT(A,k,i,lda) * (*ELT(B,k,j,ldb));
                    if (!unit) s /= *CELT(A,i,i,lda);
                    *ELT(B,i,j,ldb) = s;
                }
        } else if (uplo == CblasUpper && transA == CblasNoTrans) {
            for (int j = 0; j < N; ++j)
                for (int i = M-1; i >= 0; --i) {
                    double s = *ELT(B,i,j,ldb);
                    for (int k = i+1; k < M; ++k)
                        s -= *CELT(A,i,k,lda) * (*ELT(B,k,j,ldb));
                    if (!unit) s /= *CELT(A,i,i,lda);
                    *ELT(B,i,j,ldb) = s;
                }
        } else /* Upper, Trans */ {
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < M; ++i) {
                    double s = *ELT(B,i,j,ldb);
                    for (int k = 0; k < i; ++k)
                        s -= *CELT(A,k,i,lda) * (*ELT(B,k,j,ldb));
                    if (!unit) s /= *CELT(A,i,i,lda);
                    *ELT(B,i,j,ldb) = s;
                }
        }
    } else {
        /* Right: B is M x N, op(A) is N x N */
        if (uplo == CblasLower && transA == CblasTrans) {
            /* X * A^T = B, A lower => same as solving X * U = B with U = A^T upper */
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j) {
                    double s = *ELT(B,i,j,ldb);
                    for (int k = 0; k < j; ++k)
                        s -= *CELT(A,j,k,lda) * (*ELT(B,i,k,ldb));
                    if (!unit) s /= *CELT(A,j,j,lda);
                    *ELT(B,i,j,ldb) = s;
                }
        } else if (uplo == CblasLower && transA == CblasNoTrans) {
            for (int i = 0; i < M; ++i)
                for (int j = N-1; j >= 0; --j) {
                    double s = *ELT(B,i,j,ldb);
                    for (int k = j+1; k < N; ++k)
                        s -= *CELT(A,k,j,lda) * (*ELT(B,i,k,ldb));
                    if (!unit) s /= *CELT(A,j,j,lda);
                    *ELT(B,i,j,ldb) = s;
                }
        } else if (uplo == CblasUpper && transA == CblasNoTrans) {
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j) {
                    double s = *ELT(B,i,j,ldb);
                    for (int k = 0; k < j; ++k)
                        s -= *CELT(A,k,j,lda) * (*ELT(B,i,k,ldb));
                    if (!unit) s /= *CELT(A,j,j,lda);
                    *ELT(B,i,j,ldb) = s;
                }
        } else /* Upper, Trans */ {
            for (int i = 0; i < M; ++i)
                for (int j = N-1; j >= 0; --j) {
                    double s = *ELT(B,i,j,ldb);
                    for (int k = j+1; k < N; ++k)
                        s -= *CELT(A,j,k,lda) * (*ELT(B,i,k,ldb));
                    if (!unit) s /= *CELT(A,j,j,lda);
                    *ELT(B,i,j,ldb) = s;
                }
        }
    }
}

#endif /* !LDLT_USE_SYSTEM_BLAS */

/* Column-major only, unit stride or not. */
void cblas_dgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans, int M, int N, double alpha,
                 const double *A, int lda, const double *X, int incX,
                 double beta, double *Y, int incY) {
    (void)order;
    if (trans == CblasNoTrans) {
        for (int i = 0; i < M; ++i) Y[(size_t)i*incY] *= beta;
        for (int j = 0; j < N; ++j) {
            double a = alpha * X[(size_t)j*incX];
            const double *col = A + (size_t)j*lda;
            for (int i = 0; i < M; ++i) Y[(size_t)i*incY] += col[i] * a;
        }
    } else {
        for (int j = 0; j < N; ++j) {
            const double *col = A + (size_t)j*lda;
            double acc = 0.0;
            for (int i = 0; i < M; ++i) acc += col[i] * X[(size_t)i*incX];
            Y[(size_t)j*incY] = beta * Y[(size_t)j*incY] + alpha * acc;
        }
    }
}

void cblas_dtrsv(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                 int N, const double *A, int lda, double *X, int incX) {
    (void)order; (void)uplo; /* lower only */
    if (trans == CblasNoTrans) {
        for (int j = 0; j < N; ++j) {
            const double *col = A + (size_t)j*lda;
            if (diag != CblasUnit) X[(size_t)j*incX] /= col[j];
            double xj = X[(size_t)j*incX];
            for (int i = j + 1; i < N; ++i) X[(size_t)i*incX] -= col[i] * xj;
        }
    } else {
        for (int j = N - 1; j >= 0; --j) {
            const double *col = A + (size_t)j*lda;
            double acc = X[(size_t)j*incX];
            for (int i = j + 1; i < N; ++i) acc -= col[i] * X[(size_t)i*incX];
            X[(size_t)j*incX] = diag != CblasUnit ? acc / col[j] : acc;
        }
    }
}
