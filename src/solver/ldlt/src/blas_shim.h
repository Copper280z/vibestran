#ifndef LDLT_BLAS_SHIM_H
#define LDLT_BLAS_SHIM_H

#include "internal.h"

#if LDLT_USE_SYSTEM_BLAS
#  if defined(LDLT_BLAS_ACCELERATE)
#    include <vecLib/cblas.h>
#  else
#    include <cblas.h>
#  endif
#else

typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_ORDER;
typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
typedef enum { CblasUpper=121, CblasLower=122 } CBLAS_UPLO;
typedef enum { CblasNonUnit=131, CblasUnit=132 } CBLAS_DIAG;
typedef enum { CblasLeft=141, CblasRight=142 } CBLAS_SIDE;

void cblas_dgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
                 int M, int N, int K, double alpha,
                 const double *A, int lda, const double *B, int ldb,
                 double beta, double *C, int ldc);
void cblas_dsyrk(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE,
                 int N, int K, double alpha,
                 const double *A, int lda, double beta, double *C, int ldc);
void cblas_dtrsm(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG,
                 int M, int N, double alpha,
                 const double *A, int lda, double *B, int ldb);
void cblas_dgemv(CBLAS_ORDER, CBLAS_TRANSPOSE, int M, int N, double alpha,
                 const double *A, int lda, const double *X, int incX,
                 double beta, double *Y, int incY);
void cblas_dtrsv(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG,
                 int N, const double *A, int lda, double *X, int incX);
void cblas_dscal(int N, double alpha, double *X, int incX);
void cblas_daxpy(int N, double alpha, const double *X, int incX, double *Y, int incY);

#endif /* LDLT_USE_SYSTEM_BLAS */

#endif
