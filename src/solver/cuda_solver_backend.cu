// src/solver/cuda_solver_backend.cu
// CUDA sparse solver backend: cuSOLVER sparse Cholesky + QR fallback.
//
// Solver selection logic:
//   1. Try cusolverSpDcsrlsvchol (device-side sparse Cholesky, AMD reordering).
//      Optimal for SPD FEM stiffness matrices; fills in far less than QR.
//   2. If Cholesky reports singularity (singularity != -1) or the residual
//      is too large, retry with cusolverSpDcsrlsvqr (device-side sparse QR,
//      AMD reordering) which handles non-SPD and mildly ill-conditioned
//      matrices.
//
// Both paths allocate GPU memory, copy data host→device, run the
// factorisation and solve on-device, then copy the solution device→host.
// cuSOLVER manages its own internal scratch allocations.
//
// Note: this file is only compiled when the CUDA backend is enabled in
// meson (have_cuda_backend=true), so HAVE_CUDA is guaranteed to be defined.
// The #ifdef guard is in the header only, to prevent the class from being
// declared in non-CUDA builds.
//
// <format> / std::format is intentionally avoided here: nvcc uses its
// bundled g++-12 as the host compiler, and g++-12 does not provide <format>.
// Use std::string concatenation and std::to_string() for error messages.
#define HAVE_CUDA 1

#include "solver/cuda_solver_backend.hpp"
// Include exceptions.hpp directly (not types.hpp) to avoid pulling in <format>.
#include "core/exceptions.hpp"

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace nastran {

// ── RAII helpers
// ──────────────────────────────────────────────────────────────

struct CudaContext {
  cusolverSpHandle_t cusolver = nullptr;
  cusparseHandle_t cusparse = nullptr;
  std::string device_name;
  bool use_single_precision = false;
};

// RAII wrapper for device memory allocated with cudaMalloc.
template <typename T> struct DeviceBuffer {
  T *ptr = nullptr;

  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) {
    if (count == 0)
      return;
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T));
    if (err != cudaSuccess)
      throw SolverError(
          "CUDA solver: cudaMalloc failed for " + std::to_string(count) +
          " elements: " + cudaGetErrorString(err));
  }
  ~DeviceBuffer() {
    if (ptr)
      cudaFree(ptr);
  }
  // Non-copyable, movable.
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
    if (this != &o) {
      if (ptr)
        cudaFree(ptr);
      ptr = o.ptr;
      o.ptr = nullptr;
    }
    return *this;
  }

  // Upload count elements from a host pointer.
  void upload(const T *host, std::size_t count) {
    cudaError_t err =
        cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw SolverError(
          std::string("CUDA solver: cudaMemcpy H→D failed: ") +
          cudaGetErrorString(err));
  }

  // Download count elements to a host pointer.
  void download(T *host, std::size_t count) const {
    cudaError_t err =
        cudaMemcpy(host, ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw SolverError(
          std::string("CUDA solver: cudaMemcpy D→H failed: ") +
          cudaGetErrorString(err));
  }
};

// ── Constructor / destructor
// ──────────────────────────────────────────────────

CudaSolverBackend::CudaSolverBackend(std::unique_ptr<CudaContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaSolverBackend::~CudaSolverBackend() {
  if (!ctx_)
    return;
  if (ctx_->cusparse)
    cusparseDestroy(ctx_->cusparse);
  if (ctx_->cusolver)
    cusolverSpDestroy(ctx_->cusolver);
}

CudaSolverBackend::CudaSolverBackend(CudaSolverBackend &&) noexcept = default;
CudaSolverBackend &
CudaSolverBackend::operator=(CudaSolverBackend &&) noexcept = default;

// ── Factory
// ───────────────────────────────────────────────────────────────────

std::optional<CudaSolverBackend>
CudaSolverBackend::try_create(bool use_single_precision) noexcept {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    return std::nullopt;

  // Select device 0; a future extension could let the user pick.
  if (cudaSetDevice(0) != cudaSuccess)
    return std::nullopt;

  auto ctx = std::make_unique<CudaContext>();

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
    ctx->device_name = props.name;
  ctx->use_single_precision = use_single_precision;

  if (cusolverSpCreate(&ctx->cusolver) != CUSOLVER_STATUS_SUCCESS)
    return std::nullopt;

  if (cusparseCreate(&ctx->cusparse) != CUSPARSE_STATUS_SUCCESS) {
    cusolverSpDestroy(ctx->cusolver);
    return std::nullopt;
  }

  return CudaSolverBackend(std::move(ctx));
}

// ── Accessors
// ─────────────────────────────────────────────────────────────────

std::string_view CudaSolverBackend::name() const noexcept {
  return "CUDA cuSOLVER sparse Cholesky";
}

bool CudaSolverBackend::last_solve_used_cholesky() const noexcept {
  return last_cholesky_;
}

std::string_view CudaSolverBackend::device_name() const noexcept {
  return ctx_->device_name;
}

bool CudaSolverBackend::uses_single_precision() const noexcept {
  return ctx_->use_single_precision;
}

// ── Residual validation
// ───────────────────────────────────────────────────────

// Compute relative residual ||K*u - F||_2 / ||F||_2 on the CPU.
// Used to detect garbage solutions from near-singular matrices that cuSOLVER's
// Cholesky factorises without reporting singularity (e.g. SPSD matrices where
// the near-zero pivot exceeds the factorisation tolerance).
static double relative_residual(const SparseMatrixBuilder::CsrData &K,
                                const std::vector<double> &u,
                                const std::vector<double> &F) {
  const int n = K.n;
  double res_sq = 0.0;
  double rhs_sq = 0.0;
  for (int row = 0; row < n; ++row) {
    double row_val = 0.0;
    for (int idx = K.row_ptr[row]; idx < K.row_ptr[row + 1]; ++idx)
      row_val += K.values[idx] * u[K.col_ind[idx]];
    double diff = row_val - F[row];
    res_sq += diff * diff;
    rhs_sq += F[row] * F[row];
  }
  if (rhs_sq == 0.0)
    return (res_sq == 0.0) ? 0.0 : 1.0;
  return std::sqrt(res_sq / rhs_sq);
}

// ── solve_single_precision
// ────────────────────────────────────────────────────
// Float32 Cholesky path: downcasts inputs to float, runs cusolverSpScsrlsvchol
// on-device, upcasts result back to double.  Halves device memory usage versus
// the double path at the cost of ~7 significant digits.
static std::vector<double>
solve_single_precision(cusolverSpHandle_t cusolver, const std::string &device_name,
                       const SparseMatrixBuilder::CsrData &K,
                       const std::vector<double> &F) {
  const int n = K.n;
  const int nnz = K.nnz;

  // Downcast CSR values and RHS to float.
  std::vector<float> values_f(nnz);
  std::vector<float> F_f(n);
  for (int i = 0; i < nnz; ++i)
    values_f[i] = static_cast<float>(K.values[i]);
  for (int i = 0; i < n; ++i)
    F_f[i] = static_cast<float>(F[i]);

  std::size_t alloc_bytes =
      (std::size_t)nnz * sizeof(float)    // d_values
      + (std::size_t)(n + 1) * sizeof(int) // d_row_ptr
      + (std::size_t)nnz * sizeof(int)     // d_col_ind
      + (std::size_t)n * sizeof(float)     // d_F
      + (std::size_t)n * sizeof(float);    // d_u
  std::size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::clog << "[cuda] single-precision device memory: allocating "
            << alloc_bytes / (1024.0 * 1024.0) << " MiB"
            << " (free=" << free_bytes / (1024.0 * 1024.0) << " MiB"
            << ", total=" << total_bytes / (1024.0 * 1024.0) << " MiB)\n";

  DeviceBuffer<float> d_values(nnz);
  DeviceBuffer<int>   d_row_ptr(n + 1);
  DeviceBuffer<int>   d_col_ind(nnz);
  DeviceBuffer<float> d_F(n);
  DeviceBuffer<float> d_u(n);

  d_values.upload(values_f.data(), nnz);
  d_row_ptr.upload(K.row_ptr.data(), n + 1);
  d_col_ind.upload(K.col_ind.data(), nnz);
  d_F.upload(F_f.data(), n);

  cusparseMatDescr_t descr = nullptr;
  if (cusparseCreateMatDescr(&descr) != CUSPARSE_STATUS_SUCCESS)
    throw SolverError("CUDA solver: cusparseCreateMatDescr failed");
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

  int singularity = -1;
  cusolverStatus_t status =
      cusolverSpScsrlsvchol(cusolver, n, nnz, descr,
                            d_values.ptr, d_row_ptr.ptr, d_col_ind.ptr,
                            d_F.ptr,
                            /*tol=*/1e-6f,
                            /*reorder=*/1,
                            d_u.ptr, &singularity);

  cusparseDestroyMatDescr(descr);

  if (status == CUSOLVER_STATUS_ALLOC_FAILED)
    throw SolverError(
        "CUDA solver: single-precision Cholesky internal alloc failed "
        "(matrix too large for device workspace, n=" + std::to_string(n) +
        ", nnz=" + std::to_string(nnz) + ")");
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw SolverError(
        "CUDA solver: cusolverSpScsrlsvchol failed with status " +
        std::to_string(static_cast<int>(status)));
  if (singularity != -1)
    throw SolverError(
        "CUDA solver: single-precision Cholesky reported singularity at row " +
        std::to_string(singularity) +
        " -- check boundary conditions (SPCs)");

  std::vector<float> u_f(n);
  d_u.download(u_f.data(), n);

  // Upcast result to double.
  std::vector<double> u(n);
  for (int i = 0; i < n; ++i)
    u[i] = static_cast<double>(u_f[i]);

  std::clog << "[cuda] single-precision Cholesky solve: n=" << n
            << ", nnz=" << nnz << ", device='" << device_name << "'\n";
  return u;
}

// ── solve
// ─────────────────────────────────────────────────────────────────────

std::vector<double>
CudaSolverBackend::solve(const SparseMatrixBuilder::CsrData &K,
                         const std::vector<double> &F) {
  const int n = K.n;
  const int nnz = K.nnz;

  if (n == 0)
    throw SolverError("CUDA solver: stiffness matrix is empty -- no free DOFs");
  if (static_cast<int>(F.size()) != n)
    throw SolverError("CUDA solver: force vector size " +
                      std::to_string(F.size()) + " != matrix size " +
                      std::to_string(n));

  if (ctx_->use_single_precision) {
    last_cholesky_ = true;
    return solve_single_precision(ctx_->cusolver, ctx_->device_name, K, F);
  }

  // ── Upload CSR matrix and RHS to device ──────────────────────────────────
  {
    double alloc_mib =
        ((std::size_t)nnz * sizeof(double)    // d_values
         + (std::size_t)(n + 1) * sizeof(int) // d_row_ptr
         + (std::size_t)nnz * sizeof(int)     // d_col_ind
         + (std::size_t)n * sizeof(double)    // d_F
         + (std::size_t)n * sizeof(double))   // d_u
        / (1024.0 * 1024.0);
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::clog << "[cuda] device memory: allocating " << alloc_mib << " MiB"
              << " (free=" << free_bytes / (1024.0 * 1024.0) << " MiB"
              << ", total=" << total_bytes / (1024.0 * 1024.0) << " MiB)\n";
  }

  DeviceBuffer<double> d_values(nnz);
  DeviceBuffer<int>    d_row_ptr(n + 1);
  DeviceBuffer<int>    d_col_ind(nnz);
  DeviceBuffer<double> d_F(n);
  DeviceBuffer<double> d_u(n);

  d_values.upload(K.values.data(), nnz);
  d_row_ptr.upload(K.row_ptr.data(), n + 1);
  d_col_ind.upload(K.col_ind.data(), nnz);
  d_F.upload(F.data(), n);

  // Build cusparse matrix descriptor (general, zero-based indexing).
  cusparseMatDescr_t descr = nullptr;
  if (cusparseCreateMatDescr(&descr) != CUSPARSE_STATUS_SUCCESS)
    throw SolverError("CUDA solver: cusparseCreateMatDescr failed");

  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

  std::vector<double> u(n, 0.0);
  int singularity = -1;

  // ── Path 1: device-side sparse Cholesky ──────────────────────────────────
  // reorder=1: AMD (Approximate Minimum Degree) reordering minimises fill-in.
  // tol=1e-10: singularity threshold; singularity is set to first near-zero
  //            pivot row if the matrix is (nearly) singular.
  // All matrix/vector pointers are device pointers.
  cusolverStatus_t status =
      cusolverSpDcsrlsvchol(ctx_->cusolver, n, nnz, descr,
                            d_values.ptr, d_row_ptr.ptr, d_col_ind.ptr,
                            d_F.ptr,
                            /*tol=*/1e-10,
                            /*reorder=*/1,
                            d_u.ptr, &singularity);

  // CUSOLVER_STATUS_ALLOC_FAILED (2): cuSOLVER's internal workspace for the
  // sparse Cholesky factorisation can be many times larger than the input data,
  // and may exceed available device memory even when our explicit allocations
  // succeed.  QR requires even more workspace, so falling back to QR would
  // also fail.  Direct the user to --cuda-single-precision instead.
  if (status == CUSOLVER_STATUS_ALLOC_FAILED) {
    cusparseDestroyMatDescr(descr);
    throw SolverError(
        "CUDA solver: Cholesky internal alloc failed -- matrix too large for "
        "device workspace (n=" + std::to_string(n) +
        ", nnz=" + std::to_string(nnz) +
        "). Retry with --cuda-single-precision to halve GPU memory usage.");
  }
  if (status != CUSOLVER_STATUS_SUCCESS) {
    cusparseDestroyMatDescr(descr);
    throw SolverError(
        "CUDA solver: cusolverSpDcsrlsvchol failed with status " +
        std::to_string(static_cast<int>(status)));
  }

  bool chol_ok = true;

  if (chol_ok) {
    // Download solution to validate residual.
    d_u.download(u.data(), n);

    // cuSOLVER's Cholesky can silently "succeed" for near-singular SPSD
    // matrices if the near-zero pivot exceeds the factorization tolerance.
    // Validate the residual to catch these cases before returning a garbage
    // solution.
    chol_ok = (singularity == -1);
    if (chol_ok) {
      double rel_res = relative_residual(K, u, F);
      if (rel_res > 1e-2) {
        std::clog << "[cuda] Cholesky residual " << rel_res
                  << " > 1e-2 -- retrying with sparse QR\n";
        chol_ok = false;
      }
    }
  }

  if (chol_ok) {
    last_cholesky_ = true;
    cusparseDestroyMatDescr(descr);
    std::clog << "[cuda] Cholesky solve: n=" << n << ", nnz=" << nnz
              << ", device='" << ctx_->device_name << "'\n";
    return u;
  }

  // ── Path 2: device-side sparse QR fallback ───────────────────────────────
  // cusolverSpDcsrlsvqr handles non-SPD and mildly ill-conditioned matrices.
  // reorder=1: AMD reordering.
  if (singularity != -1)
    std::clog << "[cuda] Cholesky reported singularity at row " << singularity
              << " -- retrying with sparse QR\n";

  // Reset device solution buffer to zero before QR solve.
  cudaMemset(d_u.ptr, 0, n * sizeof(double));
  singularity = -1;

  status =
      cusolverSpDcsrlsvqr(ctx_->cusolver, n, nnz, descr,
                          d_values.ptr, d_row_ptr.ptr, d_col_ind.ptr,
                          d_F.ptr,
                          /*tol=*/1e-10,
                          /*reorder=*/1,
                          d_u.ptr, &singularity);

  cusparseDestroyMatDescr(descr);

  if (status == CUSOLVER_STATUS_ALLOC_FAILED)
    throw SolverError(
        "CUDA solver: QR internal alloc failed -- matrix too large for "
        "cuSOLVER device workspace (n=" + std::to_string(n) +
        ", nnz=" + std::to_string(nnz) + ")");
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw SolverError(
        "CUDA solver: cusolverSpDcsrlsvqr failed with status " +
        std::to_string(static_cast<int>(status)));

  if (singularity != -1)
    throw SolverError("CUDA solver: stiffness matrix is singular at row " +
                      std::to_string(singularity) +
                      " -- check boundary conditions (SPCs)");

  d_u.download(u.data(), n);

  // QR succeeded; validate residual as a final sanity check.
  double rel_res_qr = relative_residual(K, u, F);
  if (rel_res_qr > 1e-6)
    throw SolverError(
        "CUDA solver: QR produced large residual " +
        std::to_string(rel_res_qr) +
        " -- stiffness matrix is singular or very ill-conditioned. "
        "Check boundary conditions (SPCs)");

  last_cholesky_ = false;
  std::clog << "[cuda] QR solve: n=" << n << ", nnz=" << nnz << ", device='"
            << ctx_->device_name << "'\n";
  return u;
}

} // namespace nastran
