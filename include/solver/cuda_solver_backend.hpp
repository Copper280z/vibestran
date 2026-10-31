#pragma once
// include/solver/cuda_solver_backend.hpp
// CUDA sparse solver backend using NVIDIA cuDSS (direct sparse solver library).
//
// Algorithm: Sparse Cholesky (CUDSS_MTYPE_SPD) for SPD FEM stiffness matrices,
// with automatic fallback to sparse LU (CUDSS_MTYPE_GENERAL) if the Cholesky
// factorization fails or its solution cannot be refined.  Factorization and
// triangular solves run on-device via cudssExecute; the solution is then
// refined with the retained factorization and compensated residuals computed
// on the host when refinement is enabled (iterative_refinement.hpp).
//
// For large problems where cuDSS's internal workspace exceeds device memory,
// fp32 mode halves the device memory footprint by factorizing in float;
// refinement with double-precision residuals then recovers double accuracy
// when cond(K) is below about 1e7.  When it cannot, the solve is repeated in
// fp64 (with a warning).
//
// Use try_create() to construct — returns nullopt when no CUDA device is
// present so the caller can fall back without exception handling.

#ifdef HAVE_CUDA

#include "solver/solver_backend.hpp"
#include <memory>
#include <optional>

namespace vibestran {

enum class CudaSolverPrecision {
    Float64,
    Float32,
};

// Opaque RAII wrapper around cuSOLVER/cuSPARSE handles.
// Defined in cuda_solver_backend.cu to keep CUDA headers out of this file.
struct CudaContext;

class CudaSolverBackend final : public SolverBackend {
public:
    ~CudaSolverBackend() override;
    CudaSolverBackend(CudaSolverBackend&&) noexcept;
    CudaSolverBackend& operator=(CudaSolverBackend&&) noexcept;
    CudaSolverBackend(const CudaSolverBackend&) = delete;
    CudaSolverBackend& operator=(const CudaSolverBackend&) = delete;

    /// Factory — returns nullopt when no CUDA device is available.
    /// precision: downcast inputs to float before the GPU solve and upcast the
    /// result back to double when Float32 is selected.
    [[nodiscard]] static std::optional<CudaSolverBackend>
    try_create(CudaSolverPrecision precision = CudaSolverPrecision::Float64) noexcept;

    /// Solve K*u = F using cuDSS sparse Cholesky (with LU fallback).
    /// Throws SolverError on failure.
    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::string_view name() const noexcept override;

    /// solve() falls back to LU when Cholesky fails.
    [[nodiscard]] bool may_factor_indefinite() const noexcept override { return true; }

    // ── Diagnostics (valid after each solve()) ─────────────────────────────

    /// Returns true if the most recent solve used sparse Cholesky (false = LU fallback).
    [[nodiscard]] bool last_solve_used_cholesky() const noexcept;

    /// Precision mode selected for this backend instance.
    [[nodiscard]] CudaSolverPrecision precision() const noexcept;

    /// GPU device name reported by CUDA runtime.
    [[nodiscard]] std::string_view device_name() const noexcept;

    /// Refinement of the last solve with its factorization and compensated
    /// residuals (iterative_refinement.hpp).
    [[nodiscard]] std::optional<RefinementReport> last_refinement() const noexcept override {
        return last_refinement_;
    }

    /// Refine fp64 solves (fp32 factorizations are always refined).
    void set_refinement(bool enabled) noexcept override { refine_ = enabled; }

private:
    explicit CudaSolverBackend(std::unique_ptr<CudaContext> ctx) noexcept;

    /// solve() with the factorization in Scalar (double or float).
    template <typename Scalar>
    [[nodiscard]] std::vector<double> solve_typed(const SparseMatrixBuilder::CsrData& K,
                                                  const std::vector<double>& F);

    std::unique_ptr<CudaContext> ctx_;
    bool last_cholesky_{true};
    bool refine_{false};
    std::optional<RefinementReport> last_refinement_;
};

} // namespace vibestran

#endif // HAVE_CUDA
