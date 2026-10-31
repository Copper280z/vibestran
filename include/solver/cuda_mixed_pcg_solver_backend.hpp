#pragma once
// include/solver/cuda_mixed_pcg_solver_backend.hpp
// Experimental CUDA mixed-precision PCG backend.
//
// Algorithm: float32 inner PCG solves with IC0 → ILU0 → Jacobi fallback,
// wrapped by iterative refinement with compensated double-precision residuals
// on the original system (iterative_refinement.hpp).  Converges when cond(K)
// is below about 1e7; beyond that the fp32 corrections do not contract and
// solve() throws.
//
// This backend is intentionally separate from the stable CUDA PCG backend so
// mixed-precision experimentation does not complicate the maintained float64
// solver path.

#ifdef HAVE_CUDA

#include "solver/solver_backend.hpp"
#include <memory>
#include <optional>

namespace vibestran {

struct CudaPCGContext;

class CudaMixedPCGSolverBackend final : public SolverBackend {
public:
    ~CudaMixedPCGSolverBackend() override;
    CudaMixedPCGSolverBackend(CudaMixedPCGSolverBackend&&) noexcept;
    CudaMixedPCGSolverBackend& operator=(CudaMixedPCGSolverBackend&&) noexcept;
    CudaMixedPCGSolverBackend(const CudaMixedPCGSolverBackend&) = delete;
    CudaMixedPCGSolverBackend& operator=(const CudaMixedPCGSolverBackend&) = delete;

    /// Factory — returns nullopt when no CUDA device is available.
    /// @param tolerance  Target relative error of the solution (<=0 selects
    ///                   default 1e-6).
    /// @param max_iters  Maximum inner PCG iterations per correction solve.
    [[nodiscard]] static std::optional<CudaMixedPCGSolverBackend>
    try_create(double tolerance = 0.0, int max_iters = 0) noexcept;

    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::vector<std::pair<std::string, std::string>> details() const override;
    [[nodiscard]] int last_iteration_count() const noexcept override;
    [[nodiscard]] double last_estimated_error() const noexcept override;
    [[nodiscard]] std::optional<RefinementReport> last_refinement() const noexcept override;
    [[nodiscard]] std::string_view device_name() const noexcept;

private:
    explicit CudaMixedPCGSolverBackend(std::unique_ptr<CudaPCGContext> ctx) noexcept;
    std::unique_ptr<CudaPCGContext> ctx_;
};

} // namespace vibestran

#endif // HAVE_CUDA
