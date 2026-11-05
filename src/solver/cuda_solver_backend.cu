// src/solver/cuda_solver_backend.cu
// CUDA sparse solver backend using NVIDIA cuDSS (direct sparse solver library).
//
// Solver selection logic:
//   1. Try CUDSS_MTYPE_SPD (Cholesky, AMD reordering).
//      Optimal for SPD FEM stiffness matrices; less fill-in and workspace than
//      LU.
//   2. If Cholesky factorisation fails or the residual is too large, retry with
//      CUDSS_MTYPE_GENERAL (LU with partial pivoting, AMD reordering) which
//      handles non-SPD and mildly ill-conditioned matrices.
//
// Both paths allocate GPU memory, execute all cuDSS phases on-device
// (ANALYSIS → FACTORIZATION → SOLVE), then copy the solution host←device.
//
// For large problems where cuDSS's internal workspace exceeds device memory,
// fp32 mode reduces the device memory footprint by downcasting inputs to float
// before the solve and upcasting the result back to double. When the factor
// still does not fit, cuDSS hybrid mode keeps it in host memory
// (analyze_within_device_memory).
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
#include "core/logger.hpp"
#include "solver/iterative_refinement.hpp"

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace vibestran {

// ── RAII helpers
// ──────────────────────────────────────────────────────────────

struct CudaContext {
  cudssHandle_t cudss = nullptr;
  std::string device_name;
  CudaSolverPrecision precision = CudaSolverPrecision::Float64;
};

// RAII wrapper for device memory allocated with cudaMalloc.
template <typename T> struct DeviceBuffer {
  T *ptr = nullptr;

  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) {
    if (count == 0)
      return;
    cudaError_t err =
        cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T));
    if (err != cudaSuccess)
      throw SolverError("CUDA solver: cudaMalloc failed for " +
                        std::to_string(count) +
                        " elements: " + cudaGetErrorString(err));
  }
  ~DeviceBuffer() {
    if (ptr)
      cudaFree(ptr);
  }
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

  void upload(const T *host, std::size_t count) {
    cudaError_t err =
        cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw SolverError(std::string("CUDA solver: cudaMemcpy H→D failed: ") +
                        cudaGetErrorString(err));
  }

  void download(T *host, std::size_t count) const {
    cudaError_t err =
        cudaMemcpy(host, ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw SolverError(std::string("CUDA solver: cudaMemcpy D→H failed: ") +
                        cudaGetErrorString(err));
  }
};

// RAII wrappers for cuDSS objects.
struct CuDSSConfig {
  cudssConfig_t cfg = nullptr;
  CuDSSConfig() {
    if (cudssConfigCreate(&cfg) != CUDSS_STATUS_SUCCESS)
      throw SolverError("CUDA solver: cudssConfigCreate failed");
  }
  ~CuDSSConfig() {
    if (cfg)
      cudssConfigDestroy(cfg);
  }
  CuDSSConfig(const CuDSSConfig &) = delete;
  CuDSSConfig &operator=(const CuDSSConfig &) = delete;
};

struct CuDSSData {
  cudssHandle_t handle;
  cudssData_t data = nullptr;
  explicit CuDSSData(cudssHandle_t h) : handle(h) {
    if (cudssDataCreate(handle, &data) != CUDSS_STATUS_SUCCESS)
      throw SolverError("CUDA solver: cudssDataCreate failed");
  }
  ~CuDSSData() {
    if (data)
      cudssDataDestroy(handle, data);
  }
  CuDSSData(const CuDSSData &) = delete;
  CuDSSData &operator=(const CuDSSData &) = delete;

  /// Releases the analysis and its device memory for a new first analysis.
  void reset() {
    cudssDataDestroy(handle, data);
    data = nullptr;
    if (cudssDataCreate(handle, &data) != CUDSS_STATUS_SUCCESS)
      throw SolverError("CUDA solver: cudssDataCreate failed");
  }
};

struct CuDSSMatrix {
  cudssMatrix_t mat = nullptr;
  CuDSSMatrix() = default;
  ~CuDSSMatrix() {
    if (mat)
      cudssMatrixDestroy(mat);
  }
  CuDSSMatrix(const CuDSSMatrix &) = delete;
  CuDSSMatrix &operator=(const CuDSSMatrix &) = delete;
};

// ── Constructor / destructor
// ──────────────────────────────────────────────────

CudaSolverBackend::CudaSolverBackend(std::unique_ptr<CudaContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaSolverBackend::~CudaSolverBackend() {
  if (!ctx_)
    return;
  if (ctx_->cudss)
    cudssDestroy(ctx_->cudss);
}

CudaSolverBackend::CudaSolverBackend(CudaSolverBackend &&) noexcept = default;
CudaSolverBackend &
CudaSolverBackend::operator=(CudaSolverBackend &&) noexcept = default;

// ── Factory
// ───────────────────────────────────────────────────────────────────

std::optional<CudaSolverBackend>
CudaSolverBackend::try_create(CudaSolverPrecision precision) noexcept {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    return std::nullopt;

  if (cudaSetDevice(0) != cudaSuccess)
    return std::nullopt;

  auto ctx = std::make_unique<CudaContext>();

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
    ctx->device_name = props.name;
  ctx->precision = precision;

  if (cudssCreate(&ctx->cudss) != CUDSS_STATUS_SUCCESS)
    return std::nullopt;

  return CudaSolverBackend(std::move(ctx));
}

// ── Accessors
// ─────────────────────────────────────────────────────────────────

std::string_view CudaSolverBackend::name() const noexcept {
  return "CUDA cuDSS sparse direct solver";
}

bool CudaSolverBackend::last_solve_used_cholesky() const noexcept {
  return last_cholesky_;
}

std::string_view CudaSolverBackend::device_name() const noexcept {
  return ctx_->device_name;
}

CudaSolverPrecision CudaSolverBackend::precision() const noexcept {
  return ctx_->precision;
}

std::vector<std::pair<std::string, std::string>> CudaSolverBackend::details() const {
  const bool fp32 = ctx_->precision == CudaSolverPrecision::Float32;
  return {{"Device", ctx_->device_name},
          {"Method", "cuDSS sparse Cholesky"},
          {"Indefinite fallback", "LU"},
          {"cuDSS version", std::to_string(CUDSS_VERSION_MAJOR) + "." +
                                std::to_string(CUDSS_VERSION_MINOR) + "." +
                                std::to_string(CUDSS_VERSION_PATCH)},
          {"Precision", fp32 ? "fp32 factorization, refined in fp64 (fp64 when "
                               "fp32 cannot converge)"
                             : "fp64"},
          {"Iterative refinement", fp32 ? "always" : refine_ ? "on" : "off"}};
}

// ── cudss_execute_phase: thin error-checking wrapper
// ─────────────────────────────────────────────────────

static cudssStatus_t cudss_execute(cudssHandle_t handle, cudssPhase_t phase,
                                   cudssConfig_t cfg, cudssData_t data,
                                   cudssMatrix_t A, cudssMatrix_t x,
                                   cudssMatrix_t b) {
  return cudssExecute(handle, phase, cfg, data, A, x, b);
}

/// Solves K x = b with a retained factorization: b and x are in double, the
/// factorization in Scalar.
using FactorSolve = std::function<std::vector<double>(const std::vector<double> &)>;

// ── Device memory and cuDSS hybrid mode
// ───────────────────────────────────────────────────────────
//
// cuDSS behaviour this relies on (measured with cuDSS 0.7.1 on a 12 GiB card):
//   - Hybrid mode sets its device limit from the card's total memory unless told
//     otherwise, so with other programs on the GPU the factorization fails with
//     CUDSS_STATUS_ALLOC_FAILED at once. The limit must be set.
//   - The limit counts everything cuDSS holds on the device, including what its
//     analysis keeps, so it is sized from the free memory before the analysis.
//     It does not bound the transient peak of the analysis itself.
//   - A limit below CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN makes the factorization
//     fail with CUDSS_STATUS_INVALID_VALUE.
//   - Without host memory registration the factorization took half the time, but
//     the minimum rose by 20% to 100%.

static constexpr std::int64_t kMiB = 1024 * 1024;

/// VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB in bytes, 0 when unset. Invalid values leave
/// automatic sizing, as VIBESTRAN_PCG_DEVICE_MEMORY_MIB does.
static std::int64_t cudss_device_budget() noexcept {
  const char *env = std::getenv("VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB");
  if (!env)
    return 0;
  const std::string_view text(env);
  std::int64_t mib = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), mib);
  if (error != std::errc{} || end != text.data() + text.size() || mib <= 0 ||
      mib > std::numeric_limits<std::int64_t>::max() / kMiB)
    return 0;
  return mib * kMiB;
}

static std::int64_t device_free_bytes() {
  std::size_t free_bytes = 0, total_bytes = 0;
  const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err != cudaSuccess)
    throw SolverError(std::string("CUDA solver: cudaMemGetInfo failed: ") +
                      cudaGetErrorString(err));
  return static_cast<std::int64_t>(free_bytes);
}

/// What can still be done when cuDSS is out of device memory.
static std::string memory_advice(bool fp32) {
  std::string advice = cudss_device_budget() > 0
                           ? "Raise VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB or free device memory"
                           : "Free device memory";
  return advice + (fp32 ? "." : ", or retry with --cuda-precision=fp32 to reduce it.");
}

/// Ordering and symbolic factorization. When the factor does not fit on the
/// device (or under VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB) it continues in hybrid mode,
/// which keeps the factor in host memory, and returns its settings.
/// Throws SolverError when the device cannot hold even hybrid mode's minimum.
static std::optional<CudssHybridMemory>
analyze_within_device_memory(cudssHandle_t handle, cudssConfig_t cfg, CuDSSData &data,
                             cudssMatrix_t A, cudssMatrix_t x, cudssMatrix_t b,
                             const std::string &label, bool fp32) {
  const auto analyze = [&] {
    return cudssExecute(handle, CUDSS_PHASE_ANALYSIS, cfg, data.data, A, x, b);
  };
  const std::int64_t budget = cudss_device_budget();
  cudssStatus_t st = analyze();
  if (st == CUDSS_STATUS_SUCCESS) {
    // Layout: [0]=device stable, [1]=device peak, [2]=host stable, [3]=host peak
    std::int64_t estimates[16] = {0};
    cudssDataGet(handle, data.data, CUDSS_DATA_MEMORY_ESTIMATES, &estimates,
                 sizeof(estimates), nullptr);
    vibestran::log_debug(
        "[cuda] cuDSS memory estimates (" + label + "):"
        " device stable=" + std::to_string(estimates[0] / kMiB) + " MiB"
        " peak=" + std::to_string(estimates[1] / kMiB) + " MiB"
        ", host stable=" + std::to_string(estimates[2] / kMiB) + " MiB"
        " peak=" + std::to_string(estimates[3] / kMiB) + " MiB");
    const std::int64_t allowed = budget > 0 ? budget : static_cast<std::int64_t>(device_free_bytes() * 0.85);
    if (estimates[1] <= allowed)
      return std::nullopt;
    vibestran::log_debug("[cuda] cuDSS (" + label + "): device peak estimate " +
                         std::to_string(estimates[1] / kMiB) + " MiB exceeds " +
                         std::to_string(allowed / kMiB) + " MiB -- enabling hybrid mode");
  } else if (st == CUDSS_STATUS_ALLOC_FAILED) {
    vibestran::log_debug("[cuda] cuDSS (" + label +
                         "): analysis alloc failed -- enabling hybrid mode");
  } else {
    throw SolverError("CUDA solver: cuDSS analysis failed (" + label +
                      "), status=" + std::to_string(static_cast<int>(st)));
  }

  // Hybrid mode has to be set before a data object's first analysis.
  data.reset();
  const std::int64_t free_bytes = device_free_bytes();
  const std::int64_t limit = budget > 0 ? std::min(budget, free_bytes)
                                        : static_cast<std::int64_t>(free_bytes * 0.85);
  int hybrid_mode = 1;
  cudssConfigSet(cfg, CUDSS_CONFIG_HYBRID_MODE, &hybrid_mode, sizeof(hybrid_mode));
  std::int64_t minimum = 0;
  for (int registration : {0, 1}) {
    cudssConfigSet(cfg, CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY, &registration,
                   sizeof(registration));
    st = analyze();
    if (st == CUDSS_STATUS_SUCCESS) {
      cudssDataGet(handle, data.data, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, &minimum,
                   sizeof(minimum), nullptr);
      if (minimum <= limit) {
        std::int64_t estimates[16] = {0};
        cudssDataGet(handle, data.data, CUDSS_DATA_MEMORY_ESTIMATES, &estimates,
                     sizeof(estimates), nullptr);
        std::int64_t device_limit = limit;
        st = cudssConfigSet(cfg, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, &device_limit,
                            sizeof(device_limit));
        if (st != CUDSS_STATUS_SUCCESS)
          throw SolverError("CUDA solver: setting the cuDSS hybrid device limit failed (" + label +
                            "), status=" + std::to_string(static_cast<int>(st)));
        vibestran::log_info(
            "[cuda] cuDSS hybrid memory (" + label + "): device limit " +
            std::to_string(limit / kMiB) + " MiB, minimum " +
            std::to_string(minimum / kMiB) + " MiB, factor in host memory (estimate " +
            std::to_string(estimates[3] / kMiB) + " MiB), host registration " +
            (registration ? "on" : "off"));
        return CudssHybridMemory{limit, minimum, registration != 0};
      }
      vibestran::log_debug("[cuda] cuDSS (" + label + "): hybrid minimum " +
                           std::to_string(minimum / kMiB) + " MiB exceeds the " +
                           std::to_string(limit / kMiB) + " MiB limit with host registration " +
                           (registration ? "on" : "off"));
    } else if (st != CUDSS_STATUS_ALLOC_FAILED) {
      throw SolverError("CUDA solver: cuDSS hybrid analysis failed (" + label +
                        "), status=" + std::to_string(static_cast<int>(st)));
    }
    data.reset();
  }
  throw SolverError("CUDA solver: cuDSS hybrid mode (" + label + ") needs " +
                    (minimum > 0 ? "at least " + std::to_string(minimum / kMiB) + " MiB of"
                                 : std::string("more")) +
                    " device memory, but its limit is " + std::to_string(limit / kMiB) +
                    " MiB (" + std::to_string(free_bytes / kMiB) + " MiB free). " +
                    memory_advice(fp32));
}

// ── factor_cudss: run analysis + factorization for a given matrix type, then
// hand use_factor a solve with that factorization (valid only during the
// call; it reuses d_F and d_u as the right-hand side and solution buffers).
// Returns false (without throwing) for recoverable factorisation failures so
// the caller can retry with a different matrix type.
// Throws SolverError for unrecoverable errors (bad status, alloc failure, etc.)
template <typename Scalar>
static bool factor_cudss(cudssHandle_t handle, int n, int nnz,
                         DeviceBuffer<Scalar> &d_values,
                         DeviceBuffer<int> &d_row_ptr,
                         DeviceBuffer<int> &d_col_ind, DeviceBuffer<Scalar> &d_F,
                         DeviceBuffer<Scalar> &d_u, cudssMatrixType_t mtype,
                         cudssMatrixViewType_t mview,
                         cudaDataType_t scalar_type, const std::string &label,
                         std::optional<CudssHybridMemory> &hybrid,
                         const std::function<void(const FactorSolve &)> &use_factor) {
  // Reset solution buffer.
  cudaMemset(d_u.ptr, 0, (std::size_t)n * sizeof(Scalar));

  CuDSSConfig cfg;
  CuDSSData solver_data(handle);

  // Configure reordering: CUDSS_ALG_DEFAULT uses AMD-like nested dissection.
  cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
  cudssConfigSet(cfg.cfg, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg,
                 sizeof(reorder_alg));

  // Build the sparse matrix descriptor.
  // Pass rowEnd=NULL: cuDSS treats rowStart as a standard n+1 CSR offsets
  // array. The caller selects the matrix view so SPD solves can consume the
  // compact lower-triangular symmetric storage directly.
  CuDSSMatrix A_mat;
  cudssStatus_t st = cudssMatrixCreateCsr(&A_mat.mat,
                                          /*nrows=*/static_cast<int64_t>(n),
                                          /*ncols=*/static_cast<int64_t>(n),
                                          /*nnz=*/static_cast<int64_t>(nnz),
                                          /*rowStart=*/d_row_ptr.ptr,
                                          /*rowEnd=*/nullptr,
                                          /*colIndices=*/d_col_ind.ptr,
                                          /*values=*/d_values.ptr,
                                          /*indexType=*/CUDA_R_32I,
                                          /*valueType=*/scalar_type, mtype,
                                          mview, CUDSS_BASE_ZERO);
  if (st != CUDSS_STATUS_SUCCESS)
    throw SolverError("CUDA solver: cudssMatrixCreateCsr failed, status=" +
                      std::to_string(static_cast<int>(st)));

  // RHS dense vector.
  CuDSSMatrix b_mat;
  st = cudssMatrixCreateDn(&b_mat.mat,
                           /*nrows=*/static_cast<int64_t>(n),
                           /*ncols=*/1,
                           /*ld=*/static_cast<int64_t>(n), d_F.ptr, scalar_type,
                           CUDSS_LAYOUT_COL_MAJOR);
  if (st != CUDSS_STATUS_SUCCESS)
    throw SolverError("CUDA solver: cudssMatrixCreateDn (rhs) failed, status=" +
                      std::to_string(static_cast<int>(st)));

  // Solution dense vector.
  CuDSSMatrix x_mat;
  st = cudssMatrixCreateDn(&x_mat.mat, static_cast<int64_t>(n), 1,
                           static_cast<int64_t>(n), d_u.ptr, scalar_type,
                           CUDSS_LAYOUT_COL_MAJOR);
  if (st != CUDSS_STATUS_SUCCESS)
    throw SolverError("CUDA solver: cudssMatrixCreateDn (sol) failed, status=" +
                      std::to_string(static_cast<int>(st)));

  constexpr bool fp32 = std::is_same_v<Scalar, float>;
  hybrid = analyze_within_device_memory(handle, cfg.cfg, solver_data, A_mat.mat, x_mat.mat,
                                        b_mat.mat, label, fp32);

  // ── Factorisation ─────────────────────────────────────────────────────────
  st = cudss_execute(handle, CUDSS_PHASE_FACTORIZATION, cfg.cfg,
                     solver_data.data, A_mat.mat, x_mat.mat, b_mat.mat);
  if (st == CUDSS_STATUS_ALLOC_FAILED)
    throw SolverError(
        "CUDA solver: cuDSS factorisation ran out of memory (" + label + ", n=" +
        std::to_string(n) + ", nnz=" + std::to_string(nnz) +
        (hybrid ? ", hybrid device limit " +
                      std::to_string(hybrid->device_limit_bytes / kMiB) + " MiB). "
                : std::string("). ")) +
        memory_advice(fp32));
  // A rejected configuration says nothing about the matrix, and LU would not help.
  if (st == CUDSS_STATUS_INVALID_VALUE)
    throw SolverError("CUDA solver: cuDSS rejected the factorisation settings (" + label +
                      ", n=" + std::to_string(n) + ")" +
                      (hybrid ? ", hybrid device limit " +
                                    std::to_string(hybrid->device_limit_bytes / kMiB) +
                                    " MiB, minimum " +
                                    std::to_string(hybrid->device_minimum_bytes / kMiB) + " MiB"
                              : std::string()));
  // Non-success for SPD factorisation likely means matrix is not positive
  // definite; return false so the caller can retry with GENERAL type.
  if (st != CUDSS_STATUS_SUCCESS) {
    vibestran::log_debug("[cuda] cuDSS " + label +
                        " factorisation failed, status=" +
                        std::to_string(static_cast<int>(st)) +
                        " -- will retry with LU");
    return false;
  }

  // ── Solve with the factorization ──────────────────────────────────────────
  // In fp32 the right-hand side is scaled to unit max-norm before the
  // downcast: refinement residuals are tiny and must not lose range.
  std::vector<Scalar> b(static_cast<std::size_t>(n)), x(static_cast<std::size_t>(n));
  const FactorSolve solve = [&](const std::vector<double> &rhs) {
    double scale = 1.0;
    if constexpr (!std::is_same_v<Scalar, double>) {
      double m = 0.0;
      for (double v : rhs)
        m = std::max(m, std::abs(v));
      if (m > 0.0 && std::isfinite(m))
        scale = m;
    }
    for (int i = 0; i < n; ++i)
      b[static_cast<std::size_t>(i)] =
          static_cast<Scalar>(rhs[static_cast<std::size_t>(i)] / scale);
    d_F.upload(b.data(), n);
    const cudssStatus_t s = cudss_execute(handle, CUDSS_PHASE_SOLVE, cfg.cfg,
                                          solver_data.data, A_mat.mat,
                                          x_mat.mat, b_mat.mat);
    if (s != CUDSS_STATUS_SUCCESS)
      throw SolverError("CUDA solver: cuDSS solve failed (" + label + ", n=" +
                        std::to_string(n) + "), status=" +
                        std::to_string(static_cast<int>(s)));
    d_u.download(x.data(), n);
    std::vector<double> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
      out[static_cast<std::size_t>(i)] =
          static_cast<double>(x[static_cast<std::size_t>(i)]) * scale;
    return out;
  };
  use_factor(solve);
  return true;
}

// ── solve_and_refine
// ─────────────────────────────────────────────────────────

/// u = solve(F), refined against the double-precision K when `refine`
/// (otherwise only assessed: residual, backward error, epsilon).  An fp32
/// factorization contracts the error by about cond(K) * 6e-8 per correction,
/// so it is allowed more corrections.
static std::vector<double> solve_and_refine(const SparseMatrixBuilder::CsrData &K,
                                            const std::vector<double> &F,
                                            const FactorSolve &solve, bool fp32, bool refine,
                                            const std::string &label,
                                            RefinementReport &report) {
  std::vector<double> u = solve(F);
  RefinementOptions opts;
  opts.max_corrections = !refine ? 0 : fp32 ? 30 : 10;
  report = refine_solution(K, F, u, refine ? solve : FactorSolve{}, opts);
  if (!refine) {
    vibestran::log_debug("[cuda] " + label +
                         std::format(" solve: epsilon {:.2e}, backward error {:.2e}",
                                     report.epsilon, report.backward_error));
    return u;
  }
  vibestran::log_debug(
      "[cuda] " + label + " iterative refinement: " +
      std::to_string(report.corrections) + " correction(s), estimated error " +
      std::format("{:.2e} -> {:.2e}, backward error {:.2e}", report.initial_error,
                  report.estimated_error, report.backward_error) +
      (report.converged ? "" : " (not converged)"));
  return u;
}


// ── solve
// ─────────────────────────────────────────────────────────────────────

std::vector<double>
CudaSolverBackend::solve(const SparseMatrixBuilder::CsrData &K,
                         const std::vector<double> &F,
                         const FactorRatioCheckPolicy *) {
  const int n = K.n;

  if (n == 0)
    throw SolverError("CUDA solver: stiffness matrix is empty -- no free DOFs");
  if (static_cast<int>(F.size()) != n)
    throw SolverError("CUDA solver: force vector size " +
                      std::to_string(F.size()) + " != matrix size " +
                      std::to_string(n));
  last_refinement_.reset();
  last_hybrid_.reset();
  if (ctx_->precision == CudaSolverPrecision::Float32) {
    std::vector<double> u = solve_typed<float>(K, F);
    if (last_refinement_->converged)
      return u;
    // An fp32 factorization contracts the error by cond(K) * 6e-8 per
    // correction; beyond cond(K) ~ 1e7 (thin shells and solids) it cannot
    // be refined, and its solution is unusable.
    notices().add(
        "cuDSS fp32 factorization could not be refined to double accuracy "
        "(estimated relative error " +
        std::format("{:.2e}", last_refinement_->estimated_error) +
        ": the matrix is too ill-conditioned for fp32); solved again in fp64");
    // An unrefined fp64 solve records no report; the fp32 one must not stand
    // in for it.
    last_refinement_.reset();
  }
  return solve_typed<double>(K, F);
}

template <typename Scalar>
std::vector<double>
CudaSolverBackend::solve_typed(const SparseMatrixBuilder::CsrData &K,
                               const std::vector<double> &F) {
  constexpr bool fp32 = std::is_same_v<Scalar, float>;
  // An fp32 factorization is only usable refined.
  const bool refine = fp32 || refine_;
  const cudaDataType_t scalar_type = fp32 ? CUDA_R_32F : CUDA_R_64F;
  const std::string suffix = fp32 ? "/float" : "";
  const int n = K.n;

  const SparseMatrixBuilder::CsrData *K_spd = &K;
  SparseMatrixBuilder::CsrData lower_csr;
  if (!K.stores_lower_triangle_only()) {
    lower_csr = K.lower_triangle();
    K_spd = &lower_csr;
  }
  const int nnz = K_spd->nnz;

  const auto to_scalar = [](const std::vector<double> &v) {
    std::vector<Scalar> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
      out[i] = static_cast<Scalar>(v[i]);
    return out;
  };

  // ── Allocate and upload ───────────────────────────────────────────────────
  {
    const double alloc_mib = ((std::size_t)nnz * sizeof(Scalar)    // d_values
                              + (std::size_t)(n + 1) * sizeof(int) // d_row_ptr
                              + (std::size_t)nnz * sizeof(int)     // d_col_ind
                              + (std::size_t)n * sizeof(Scalar)    // d_F
                              + (std::size_t)n * sizeof(Scalar))   // d_u
                             / (1024.0 * 1024.0);
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    vibestran::log_debug(
        "[cuda] device memory (" + std::string(fp32 ? "fp32" : "fp64") +
        "): allocating " + std::to_string(alloc_mib) + " MiB"
        " (free=" + std::to_string(free_bytes / (1024UL * 1024UL)) + " MiB"
        ", total=" + std::to_string(total_bytes / (1024UL * 1024UL)) + " MiB)");
  }

  DeviceBuffer<int> d_row_ptr(n + 1);
  DeviceBuffer<int> d_col_ind(nnz);
  DeviceBuffer<Scalar> d_F(n);
  DeviceBuffer<Scalar> d_u(n);
  d_row_ptr.upload(K_spd->row_ptr.data(), n + 1);
  d_col_ind.upload(K_spd->col_ind.data(), nnz);

  std::vector<double> u;
  RefinementReport report;

  // ── Path 1: Cholesky (SPD) ────────────────────────────────────────────────
  bool chol_ok = false;
  {
    DeviceBuffer<Scalar> d_values(nnz);
    if constexpr (fp32)
      d_values.upload(to_scalar(K_spd->values).data(), nnz);
    else
      d_values.upload(K_spd->values.data(), nnz);
    chol_ok = factor_cudss<Scalar>(
        ctx_->cudss, n, nnz, d_values, d_row_ptr, d_col_ind, d_F, d_u,
        CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, scalar_type, "SPD" + suffix, last_hybrid_,
        [&](const FactorSolve &solve) {
          u = solve_and_refine(K, F, solve, fp32, refine, "Cholesky" + suffix, report);
        });
  }
  // A Cholesky solution that indicates a singular K (the factorization
  // "succeeded" on an indefinite or singular matrix) is retried with LU.  In
  // fp32 a failed refinement means cond(K) is too large for fp32, which LU
  // does not help; solve() retries in fp64.
  if (chol_ok && !fp32 && indicates_mechanism(report, refine)) {
    vibestran::log_debug(
        "[cuda] Cholesky" + suffix + " solution indicates a singular matrix (" +
        std::format("estimated error {:.2e}, epsilon {:.2e}", report.estimated_error,
                    report.epsilon) + ") -- retrying with LU");
    chol_ok = false;
  }
  if (chol_ok) {
    last_cholesky_ = true;
  } else {
    // ── Path 2: LU (GENERAL) fallback ──────────────────────────────────────
    const SparseMatrixBuilder::CsrData *K_lu = &K;
    SparseMatrixBuilder::CsrData expanded_csr;
    if (K.stores_lower_triangle_only()) {
      expanded_csr = K.expanded_symmetric();
      K_lu = &expanded_csr;
    }
    DeviceBuffer<Scalar> d_values_lu(K_lu->nnz);
    DeviceBuffer<int> d_row_ptr_lu(K_lu->n + 1);
    DeviceBuffer<int> d_col_ind_lu(K_lu->nnz);
    if constexpr (fp32)
      d_values_lu.upload(to_scalar(K_lu->values).data(), K_lu->nnz);
    else
      d_values_lu.upload(K_lu->values.data(), K_lu->nnz);
    d_row_ptr_lu.upload(K_lu->row_ptr.data(), K_lu->n + 1);
    d_col_ind_lu.upload(K_lu->col_ind.data(), K_lu->nnz);

    const bool lu_ok = factor_cudss<Scalar>(
        ctx_->cudss, K_lu->n, K_lu->nnz, d_values_lu, d_row_ptr_lu,
        d_col_ind_lu, d_F, d_u, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
        scalar_type, "LU" + suffix, last_hybrid_, [&](const FactorSolve &solve) {
          u = solve_and_refine(K, F, solve, fp32, refine, "LU" + suffix, report);
        });
    if (!lu_ok)
      throw SolverError("CUDA solver: LU" + suffix + " factorisation failed (n=" +
                        std::to_string(n) +
                        ") -- stiffness matrix may be singular. Check boundary "
                        "conditions (SPCs)");
    // cuDSS perturbs tiny pivots instead of failing, so a singular K
    // factorizes; its solution then does not equilibrate the load.
    if (!fp32 && indicates_mechanism(report, refine))
      throw SolverError(
          "CUDA solver: the LU" + suffix + " solution does not equilibrate the load (" +
          std::format("epsilon {:.2e}, relative residual {:.2e}", report.epsilon,
                      report.relative_residual) +
          (refine ? std::format(", estimated relative error {:.2e} after {} correction(s)",
                                report.estimated_error, report.corrections)
                  : std::string()) +
          ", n=" + std::to_string(n) +
          ") -- stiffness matrix is singular. Check boundary conditions (SPCs)");
    last_cholesky_ = false;
    notices().add("The matrix is not numerically positive definite: cuDSS Cholesky" +
                  suffix + " failed and it was solved with LU");
  }

  // The accuracy achieved is reported by the caller from last_refinement()
  // (an unrefined solve is assessed by the caller).
  if (refine)
    last_refinement_ = report;
  vibestran::log_info("[cuda] " + std::string(last_cholesky_ ? "Cholesky" : "LU") +
                      suffix + " solve: n=" + std::to_string(n) +
                      ", nnz=" + std::to_string(nnz) + ", device='" +
                      ctx_->device_name + "'");
  return u;
}

// ── Phased factorization: factor()
// ─────────────────────────────────────────────────────────────────────
//
// cuDSS behaviour this relies on (measured with cuDSS 0.7.1):
//   - A Cholesky factorization of a matrix that is not positive definite
//     returns CUDSS_STATUS_SUCCESS; only CUDSS_DATA_INFO (> 0) reports it,
//     and solves with that factor are garbage.
//   - CUDSS_DATA_INFO is sticky: after a failure, later factorizations of
//     valid values silently do nothing until INFO is reset to 0. Resetting
//     it keeps the analysis; re-running the analysis alone does not help.
// So every factorization resets INFO first and checks it afterwards.

namespace {

using CsrData = SparseMatrixBuilder::CsrData;

void check_cudss(cudssStatus_t st, const std::string &what) {
  if (st != CUDSS_STATUS_SUCCESS)
    throw SolverError("CUDA solver: " + what + " failed, status=" +
                      std::to_string(static_cast<int>(st)));
}

/// One cuDSS problem: a matrix type, its CSR on the device, and the solver
/// data that holds its analysis and factorization. CUDA failures throw
/// SolverError; CudaFactor turns them into errors at its public methods.
template <typename Scalar> class CudssProblem {
public:
  static constexpr cudaDataType_t kType =
      std::is_same_v<Scalar, float> ? CUDA_R_32F : CUDA_R_64F;

  CudssProblem(cudssHandle_t handle, const CsrData &storage,
               cudssMatrixType_t mtype, cudssMatrixViewType_t mview,
               cudssAlgType_t reorder_alg, std::string label)
      : handle_(handle), data_(handle), n_(storage.n), nnz_(storage.nnz),
        label_(std::move(label)), row_ptr_(static_cast<std::size_t>(n_) + 1),
        col_ind_(static_cast<std::size_t>(nnz_)),
        values_(static_cast<std::size_t>(nnz_)) {
    row_ptr_.upload(storage.row_ptr.data(), static_cast<std::size_t>(n_) + 1);
    col_ind_.upload(storage.col_ind.data(), static_cast<std::size_t>(nnz_));
    cudssConfigSet(cfg_.cfg, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg,
                   sizeof(reorder_alg));
    check_cudss(cudssMatrixCreateCsr(&A_.mat, n_, n_, nnz_, row_ptr_.ptr,
                                     nullptr, col_ind_.ptr, values_.ptr,
                                     CUDA_R_32I, kType, mtype, mview,
                                     CUDSS_BASE_ZERO),
                "cudssMatrixCreateCsr (" + label_ + ")");
    dense(1);
  }

  void load(const std::vector<double> &values) {
    if constexpr (std::is_same_v<Scalar, double>) {
      values_.upload(values.data(), values.size());
    } else {
      std::vector<Scalar> v(values.begin(), values.end());
      values_.upload(v.data(), v.size());
    }
  }

  /// Ordering and symbolic factorization, in cuDSS hybrid mode when the device
  /// cannot hold the factor.
  void analyze() {
    hybrid_ = analyze_within_device_memory(handle_, cfg_.cfg, data_, A_.mat, X_.mat, B_.mat,
                                           label_, std::is_same_v<Scalar, float>);
  }

  /// Numeric factorization of the loaded values. Returns CUDSS_DATA_INFO:
  /// 0 on success, otherwise the failing pivot.
  [[nodiscard]] int factorize() {
    int info = 0;
    check_cudss(cudssDataSet(handle_, data_.data, CUDSS_DATA_INFO, &info,
                             sizeof(info)),
                "cudssDataSet(INFO)");
    const cudssStatus_t st = execute(CUDSS_PHASE_FACTORIZATION);
    if (st == CUDSS_STATUS_ALLOC_FAILED)
      throw SolverError(
          "CUDA solver: cuDSS factorization ran out of memory (" + label_ + ", n=" +
          std::to_string(n_) + ", nnz=" + std::to_string(nnz_) +
          (hybrid_ ? ", hybrid device limit " +
                         std::to_string(hybrid_->device_limit_bytes / kMiB) + " MiB). "
                   : std::string("). ")) +
          memory_advice(std::is_same_v<Scalar, float>));
    check_cudss(st, "cuDSS factorization (" + label_ + ")");
    check_cudss(cudssDataGet(handle_, data_.data, CUDSS_DATA_INFO, &info,
                             sizeof(info), nullptr),
                "cudssDataGet(INFO)");
    return info;
  }

  /// X = K^-1 B for nrhs columns of n values. In fp32 each column is scaled
  /// to unit max-norm before the downcast so small refinement residuals keep
  /// their range.
  void solve(const double *B, double *X, int nrhs) {
    dense(nrhs);
    const auto n = static_cast<std::size_t>(n_);
    std::vector<double> scale(static_cast<std::size_t>(nrhs), 1.0);
    std::vector<Scalar> b(n * static_cast<std::size_t>(nrhs));
    for (int k = 0; k < nrhs; ++k) {
      const double *column = B + n * static_cast<std::size_t>(k);
      if constexpr (!std::is_same_v<Scalar, double>) {
        double m = 0.0;
        for (std::size_t i = 0; i < n; ++i)
          m = std::max(m, std::abs(column[i]));
        if (m > 0.0 && std::isfinite(m))
          scale[static_cast<std::size_t>(k)] = m;
      }
      for (std::size_t i = 0; i < n; ++i)
        b[n * static_cast<std::size_t>(k) + i] =
            static_cast<Scalar>(column[i] / scale[static_cast<std::size_t>(k)]);
    }
    rhs_.upload(b.data(), b.size());
    check_cudss(execute(CUDSS_PHASE_SOLVE), "cuDSS solve (" + label_ + ")");
    sol_.download(b.data(), b.size());
    for (int k = 0; k < nrhs; ++k)
      for (std::size_t i = 0; i < n; ++i)
        X[n * static_cast<std::size_t>(k) + i] =
            static_cast<double>(b[n * static_cast<std::size_t>(k) + i]) *
            scale[static_cast<std::size_t>(k)];
  }

private:
  cudssStatus_t execute(cudssPhase_t phase) {
    return cudssExecute(handle_, phase, cfg_.cfg, data_.data, A_.mat, X_.mat,
                        B_.mat);
  }

  /// Right-hand side and solution descriptors with exactly cols columns.
  void dense(int cols) {
    if (cols == cols_)
      return;
    const auto size = static_cast<std::size_t>(n_) * static_cast<std::size_t>(cols);
    if (cols > capacity_) {
      rhs_ = DeviceBuffer<Scalar>(size);
      sol_ = DeviceBuffer<Scalar>(size);
      capacity_ = cols;
    }
    for (CuDSSMatrix *m : {&B_, &X_}) {
      if (m->mat)
        cudssMatrixDestroy(m->mat);
      m->mat = nullptr;
    }
    check_cudss(cudssMatrixCreateDn(&B_.mat, n_, cols, n_, rhs_.ptr, kType,
                                    CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateDn (rhs)");
    check_cudss(cudssMatrixCreateDn(&X_.mat, n_, cols, n_, sol_.ptr, kType,
                                    CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateDn (solution)");
    cols_ = cols;
  }

  cudssHandle_t handle_;
  CuDSSConfig cfg_;
  CuDSSData data_;
  std::optional<CudssHybridMemory> hybrid_;
  int n_, nnz_;
  std::string label_;
  DeviceBuffer<int> row_ptr_, col_ind_;
  DeviceBuffer<Scalar> values_, rhs_, sol_;
  int capacity_{0}, cols_{0};
  CuDSSMatrix A_, B_, X_;
};

/// Cholesky with an LU fallback in one precision. The analyses are kept
/// while the sparsity pattern is unchanged.
template <typename Scalar> class CudssSystem {
public:
  CudssSystem(cudssHandle_t handle, const FactorOptions &options, Notices &notices)
      : handle_(handle), require_spd_(options.require_spd),
        // ALG_3 is approximate minimum degree; the default is nested
        // dissection, kept for FactorOptions::minimum_degree: its wide
        // elimination tree is what the GPU factorization parallelizes.
        reorder_alg_(options.dense_block ? CUDSS_ALG_3 : CUDSS_ALG_DEFAULT),
        notices_(notices) {}

  [[nodiscard]] Status factor(const CsrData &K, FactorStats &stats) {
    if (!same_pattern(K)) {
      n_ = K.n;
      lower_only_ = K.stores_lower_triangle_only();
      row_ptr_ = K.row_ptr;
      col_ind_ = K.col_ind;
      spd_.reset();
      lu_.reset();
    }
    const CsrData *lower = &K;
    CsrData lower_copy;
    if (!K.stores_lower_triangle_only()) {
      lower_copy = K.lower_triangle();
      lower = &lower_copy;
    }
    const bool analyze_spd = !spd_;
    if (analyze_spd)
      spd_ = std::make_unique<CudssProblem<Scalar>>(
          handle_, *lower, CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, reorder_alg_,
          "Cholesky" + suffix());
    spd_->load(lower->values);
    if (analyze_spd) {
      spd_->analyze();
      ++stats.symbolic_analyses;
    }
    const int info = spd_->factorize();
    ++stats.numeric_factorizations;
    if (info == 0) {
      use_lu_ = false;
      return std::nullopt;
    }
    if (require_spd_)
      return Error{"cuDSS Cholesky" + suffix() + " factorization failed at pivot " +
                   std::to_string(info) +
                   ": the matrix is not numerically positive definite"};

    const CsrData *full = &K;
    CsrData full_copy;
    if (K.stores_lower_triangle_only()) {
      full_copy = K.expanded_symmetric();
      full = &full_copy;
    }
    const bool analyze_lu = !lu_;
    if (analyze_lu)
      lu_ = std::make_unique<CudssProblem<Scalar>>(
          handle_, *full, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, reorder_alg_,
          "LU" + suffix());
    lu_->load(full->values);
    if (analyze_lu) {
      lu_->analyze();
      ++stats.symbolic_analyses;
    }
    const int lu_info = lu_->factorize();
    ++stats.numeric_factorizations;
    if (lu_info != 0)
      return Error{"cuDSS LU" + suffix() + " factorization failed at pivot " +
                   std::to_string(lu_info) +
                   " -- the matrix may be singular. Check boundary conditions (SPCs)"};
    use_lu_ = true;
    notices_.add("The matrix is not numerically positive definite: cuDSS Cholesky" +
                 suffix() + " failed at pivot " + std::to_string(info) +
                 " and it was solved with LU");
    return std::nullopt;
  }

  void solve(const double *B, double *X, int nrhs) {
    (use_lu_ ? lu_ : spd_)->solve(B, X, nrhs);
  }

  [[nodiscard]] std::string_view method() const noexcept {
    if constexpr (std::is_same_v<Scalar, float>)
      return use_lu_ ? "cudss-lu/fp32" : "cudss-cholesky/fp32";
    else
      return use_lu_ ? "cudss-lu" : "cudss-cholesky";
  }

private:
  [[nodiscard]] static std::string suffix() {
    return std::is_same_v<Scalar, float> ? "/float" : "";
  }

  [[nodiscard]] bool same_pattern(const CsrData &K) const {
    return n_ == K.n && lower_only_ == K.stores_lower_triangle_only() &&
           row_ptr_ == K.row_ptr && col_ind_ == K.col_ind;
  }

  cudssHandle_t handle_;
  bool require_spd_;
  cudssAlgType_t reorder_alg_;
  Notices &notices_;
  int n_{0};
  bool lower_only_{false};
  std::vector<int> row_ptr_, col_ind_;
  std::unique_ptr<CudssProblem<Scalar>> spd_, lu_;
  bool use_lu_{false};
};

class CudaFactor final : public LinearFactor {
public:
  CudaFactor(cudssHandle_t handle, CudaSolverPrecision precision,
             const FactorOptions &options, Notices &notices)
      : handle_(handle), require_spd_(options.require_spd), notices_(notices) {
    system_options_.require_spd = options.require_spd;
    system_options_.dense_block = options.dense_block;
    if (precision == CudaSolverPrecision::Float32)
      fp32_ = std::make_unique<CudssSystem<float>>(handle, system_options_, notices);
    else
      fp64_ = std::make_unique<CudssSystem<double>>(handle, system_options_, notices);
  }

  [[nodiscard]] int size() const noexcept override { return n_; }
  [[nodiscard]] FactorStats stats() const noexcept override { return stats_; }
  [[nodiscard]] std::string_view method() const noexcept override {
    return fp32_ ? fp32_->method() : fp64_->method();
  }
  [[nodiscard]] std::string_view ordering() const noexcept override {
    return system_options_.dense_block ? "amd" : "nested dissection";
  }

  [[nodiscard]] Status refactor(const CsrData &K) override {
    ready_ = false;
    if (K.n == 0)
      return Error{"Matrix is empty -- no free DOFs"};
    if (K.row_ptr.size() != static_cast<std::size_t>(K.n) + 1 ||
        K.col_ind.size() != static_cast<std::size_t>(K.nnz) ||
        K.values.size() != static_cast<std::size_t>(K.nnz))
      return Error{"Malformed CSR matrix: n=" + std::to_string(K.n) +
                   ", nnz=" + std::to_string(K.nnz)};
    if (n_ != 0 && K.n != n_)
      return Error{"refactor() of a " + std::to_string(n_) +
                   "-equation factor with a " + std::to_string(K.n) +
                   "-equation matrix"};
    // Transitional boundary: the cuDSS helpers throw SolverError, and this
    // is where those failures become error values (MAGNETICS_ROADMAP.md, S11).
    try {
      n_ = K.n;
      Status status;
      if (fp32_) {
        K_ = K; // fp32 solves are refined against K in double
        status = fp32_->factor(K, stats_);
        if (status) {
          // A matrix that is SPD in double can fail Cholesky in float once
          // cond(K) approaches 1/eps_float, so only fp64 decides.
          notices_.add(status->message + "; factored again in fp64");
          switch_to_fp64();
          status = fp64_->factor(K, stats_);
        }
      } else {
        status = fp64_->factor(K, stats_);
      }
      if (!status) {
        ready_ = true;
        vibestran::log_debug("[cuda] factor: " + std::string(method()) + ", order=" +
                             std::string(ordering()) +
                             ", n=" + std::to_string(K.n) + ", nnz=" +
                             std::to_string(K.nnz) + ", symbolic analyses " +
                             std::to_string(stats_.symbolic_analyses));
      }
      return status;
    } catch (const std::exception &e) {
      return Error{e.what()};
    }
  }

  [[nodiscard]] Result<std::vector<double>> solve(std::span<const double> B,
                                                  int nrhs) override {
    if (!ready_)
      return Error{"solve() with a factor whose last factorization failed"};
    const auto n = static_cast<std::size_t>(n_);
    if (nrhs < 1 || B.size() != n * static_cast<std::size_t>(nrhs))
      return Error{"Right-hand side has " + std::to_string(B.size()) +
                   " values; expected " + std::to_string(n_) + " x " +
                   std::to_string(nrhs)};
    try {
      std::vector<double> X(B.size());
      if (fp32_ && !solve_refined(B, X, nrhs)) {
        // cond(K) is too large for fp32: refactor in fp64 and solve again.
        notices_.add("cuDSS fp32 factorization could not be refined to double "
                     "accuracy (the matrix is too ill-conditioned for fp32); "
                     "factored again in fp64");
        const CsrData K = std::move(K_);
        switch_to_fp64();
        if (auto e = fp64_->factor(K, stats_)) {
          ready_ = false;
          return *e;
        }
      }
      if (fp64_)
        fp64_->solve(B.data(), X.data(), nrhs);
      if (!std::all_of(X.begin(), X.end(), [](double x) { return std::isfinite(x); }))
        return Error{std::string(method()) + " solve produced a nonfinite solution"};
      stats_.solves += nrhs;
      return X;
    } catch (const std::exception &e) {
      return Error{e.what()};
    }
  }

private:
  void switch_to_fp64() {
    fp64_ = std::make_unique<CudssSystem<double>>(handle_, system_options_, notices_);
    fp32_.reset();
    K_ = CsrData{};
  }

  /// fp32 solve of every column, refined against K in double. False when a
  /// column does not converge.
  bool solve_refined(std::span<const double> B, std::vector<double> &X, int nrhs) {
    const auto n = static_cast<std::size_t>(n_);
    fp32_->solve(B.data(), X.data(), nrhs);
    const FactorSolve correction = [&](const std::vector<double> &r) {
      std::vector<double> d(n);
      fp32_->solve(r.data(), d.data(), 1);
      return d;
    };
    RefinementOptions opts;
    opts.max_corrections = 30;
    for (int k = 0; k < nrhs; ++k) {
      const auto offset = n * static_cast<std::size_t>(k);
      std::vector<double> u(X.begin() + offset, X.begin() + offset + n);
      const RefinementReport report =
          refine_solution(K_, B.subspan(offset, n), u, correction, opts);
      if (!report.converged)
        return false;
      std::copy(u.begin(), u.end(), X.begin() + offset);
    }
    return true;
  }

  cudssHandle_t handle_;
  bool require_spd_;
  FactorOptions system_options_; // without the MAXRATIO policy, which cuDSS skips
  Notices &notices_;
  int n_{0};
  bool ready_{false};
  CsrData K_;
  std::unique_ptr<CudssSystem<float>> fp32_;
  std::unique_ptr<CudssSystem<double>> fp64_;
  FactorStats stats_;
};

} // namespace

Result<std::unique_ptr<LinearFactor>>
CudaSolverBackend::factor(const SparseMatrixBuilder::CsrData &K,
                          const FactorOptions &options) {
  if (options.factor_ratio_policy != nullptr)
    vibestran::log_debug("[cuda] factor(): MAXRATIO is not checked by the cuDSS factor");
  auto factor = std::make_unique<CudaFactor>(ctx_->cudss, ctx_->precision, options,
                                             notices());
  if (auto e = factor->refactor(K))
    return *e;
  return std::unique_ptr<LinearFactor>(std::move(factor));
}

} // namespace vibestran
