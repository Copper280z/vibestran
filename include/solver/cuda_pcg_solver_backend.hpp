#pragma once
// include/solver/cuda_pcg_solver_backend.hpp
// CUDA Preconditioned Conjugate Gradient solver backend.
//
// Large systems use diagonally equilibrated smoothed-aggregation AMG with
// fp64 Krylov iterations; small resident systems use colored IC0/ILU0/Jacobi.
// AMG factors only its small coarse problem on the CPU (at most 256 equations).
//
// When estimated resident storage exceeds the device budget, a matrix row
// prefix stays on the GPU and the CPU computes the remaining rows. Only
// compact inputs/outputs for CPU rows cross PCIe per fine product. Large solves
// keep Krylov vectors and the AMG hierarchy on the GPU when that working state
// fits; smaller budgets retain the host hierarchy. Matrix construction and
// refinement still require host RAM; this does not implement disk-backed solves.
//
// Convergence: the PCG solution is refined with compensated double-precision
// residuals (iterative_refinement.hpp) until the estimated relative error
// ||d||_inf / ||u||_inf is below tolerance (default 1e-8).  A relative-residual
// test is unattainable for ill-conditioned K (cond(K) eps ||K|| ||u|| / ||F||
// can exceed any small threshold even for the exact solution in double).
//
// Use try_create() to construct — returns nullopt when no CUDA device is
// present so the caller can fall back without exception handling.

#ifdef HAVE_CUDA

#include "solver/solver_backend.hpp"
#include <memory>
#include <optional>

namespace vibestran {

// Opaque RAII context (defined in cuda_pcg_solver_backend.cu).
struct CudaPCGContext;

class CudaPCGSolverBackend final : public SolverBackend {
public:
    ~CudaPCGSolverBackend() override;
    CudaPCGSolverBackend(CudaPCGSolverBackend&&) noexcept;
    CudaPCGSolverBackend& operator=(CudaPCGSolverBackend&&) noexcept;
    CudaPCGSolverBackend(const CudaPCGSolverBackend&) = delete;
    CudaPCGSolverBackend& operator=(const CudaPCGSolverBackend&) = delete;

    /// Factory — returns nullopt when no CUDA device is available.
    /// @param tolerance             Target relative error of the solution
    ///                               (<=0 selects the default 1e-8).
    /// @param max_iters             Maximum PCG iterations (0 = default: 10000).
    /// @param device_memory_budget  Bytes for resident storage estimation/partitioning;
    ///                               0 uses VIBESTRAN_PCG_DEVICE_MEMORY_MIB or free VRAM.
    [[nodiscard]] static std::optional<CudaPCGSolverBackend>
    try_create(double tolerance = 0.0,
               int max_iters = 0,
               std::size_t device_memory_budget = 0) noexcept;

    /// Solve K*u = F using AMG PCG for large systems, IC0/ILU0/Jacobi for small ones.
    /// Throws SolverError if the system does not converge or on GPU errors.
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

    // ── Diagnostics (valid after each solve()) ─────────────────────────────

    /// Number of PCG iterations used in the most recent solve().
    [[nodiscard]] int last_iteration_count() const noexcept override;

    /// Relative residual ||F - K u||_2 / ||F||_2 of the most recent solve().
    [[nodiscard]] double last_relative_residual() const noexcept;

    /// Estimated relative error of the refined solution (iterative_refinement.hpp).
    [[nodiscard]] double last_estimated_error() const noexcept override;

    [[nodiscard]] std::optional<RefinementReport> last_refinement() const noexcept override;

    /// GPU device name reported by the CUDA runtime.
    [[nodiscard]] std::string_view device_name() const noexcept;

private:
    explicit CudaPCGSolverBackend(std::unique_ptr<CudaPCGContext> ctx) noexcept;
    std::unique_ptr<CudaPCGContext> ctx_;
};

} // namespace vibestran

#endif // HAVE_CUDA
