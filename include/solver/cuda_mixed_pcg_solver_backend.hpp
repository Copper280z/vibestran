#pragma once
// include/solver/cuda_mixed_pcg_solver_backend.hpp
// Experimental CUDA mixed-precision PCG backend.
//
// Large resident systems use fp32 AMG preconditioning with an fp64 Krylov
// operator and reductions. Small resident systems use fp32 IC0/ILU0/Jacobi.
// Large budgeted systems use the common fp64 device-owned AMG spill path,
// falling back to a host hierarchy when the GPU working state cannot fit. All paths refine
// against the original fp64 matrix using compensated residuals and reject
// solutions that fail the requested forward-error estimate.

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
    /// @param device_memory_budget  Bytes; 0 uses VIBESTRAN_PCG_DEVICE_MEMORY_MIB
    ///                               or free VRAM. Oversized matrices use partitioned PCG (AMG for large systems).
    [[nodiscard]] static std::optional<CudaMixedPCGSolverBackend>
    try_create(double tolerance = 0.0, int max_iters = 0,
               std::size_t device_memory_budget = 0) noexcept;

    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    /// Large systems (n >= 50000) keep the AMG hierarchy and device workspace
    /// for every right-hand side and same-pattern refactor(); smaller ones
    /// use the default adapter of solve().
    [[nodiscard]] Result<std::unique_ptr<LinearFactor>> factor(
        const SparseMatrixBuilder::CsrData& K, const FactorOptions& options = {}) override;

    [[nodiscard]] bool uses_near_nullspace(int n) const noexcept override { return n >= 50000; }

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
