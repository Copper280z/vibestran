#pragma once
// include/solver/vulkan_solver_backend.hpp
// Vulkan PCG with smoothed-aggregation AMG on large fp64 systems and Jacobi
// otherwise. Both paths use compensated refinement.
//
// Two execution paths are selected automatically at solve() time:
//
// Each PCG solve stops at a loose relative residual (inner_tolerance) and is
// refined with compensated double-precision residuals on the host until the
// estimated relative error is below `tolerance`; fp32 failing that retries in
// fp64 when the device supports it.
//
//  Full-GPU path  — entire K matrix + working vectors fit in VRAM.
//                   SpMV, dot products, vector ops all run as GPU dispatches.
//                   AMG uses a small CPU coarse solve (at most 256 equations).
//
//  Partitioned   — K is too large for the device budget.
//                   A fixed row prefix stays on the GPU; CPU computes the rest.
//                   Large systems keep vectors and the AMG hierarchy on GPU;
//                   only compact CPU-row inputs/outputs cross PCIe.
//                   A host hierarchy is used when the working state cannot fit.
//                   Small partitioned Jacobi solves use CPU vectors. K must fit in RAM.
//
// Use try_create() to construct — returns nullopt if Vulkan is unavailable
// so the caller can fall back to EigenSolverBackend without exception handling.

#ifdef HAVE_VULKAN

#include "solver/solver_backend.hpp"
#include "solver/vulkan_context.hpp"
#include <memory>
#include <optional>

namespace vibestran {

struct VulkanSolverConfig {
  int max_iterations = 10000;
  /// Target relative error of the solution.  The GPU PCG solves are refined
  /// with double-precision compensated residuals on the host
  /// (iterative_refinement.hpp), so an fp32 inner solve still reaches it;
  /// the refinement's correction size is the error estimate.
  double tolerance = 1e-8;
  /// Relative residual ||r|| / ||b|| at which each inner PCG solve stops
  /// (the first solve and each correction).
  double inner_tolerance = 5e-5;
  /// false: an inner PCG solve that stagnates or reaches max_iterations
  /// returns its iterate (refinement then judges it) instead of throwing.
  bool throw_on_no_convergence = true;
  double vram_headroom = 0.15; // fraction of VRAM reserved for driver
  std::size_t device_memory_budget = 0; // bytes; 0 uses VIBESTRAN_PCG_DEVICE_MEMORY_MIB or automatic budget
  bool force_tiled =
      false; // force the small-system partitioned path; large AMG uses the device budget
  int stagnation_window =
      0; // optional diagnostic cutoff; disabled for production convergence
  double stagnation_threshold =
      0.01; // minimum fractional improvement required per window (1%)
  int min_dofs_for_gpu =
      50000; // fall back to Eigen CPU for problems smaller than this
  bool use_double = false; // request float64 for the Jacobi path; large AMG systems use fp64
                           // (requires shaderFloat64)
};

class VulkanSolverBackend final : public SolverBackend {
public:
  ~VulkanSolverBackend() override;
  VulkanSolverBackend(VulkanSolverBackend &&) noexcept;
  VulkanSolverBackend &operator=(VulkanSolverBackend &&) noexcept;
  VulkanSolverBackend(const VulkanSolverBackend &) = delete;
  VulkanSolverBackend &operator=(const VulkanSolverBackend &) = delete;

  /// Factory — returns nullopt when Vulkan is unavailable.
  /// The caller should then instantiate EigenSolverBackend instead.
  [[nodiscard]] static std::optional<VulkanSolverBackend>
  try_create(const VulkanSolverConfig &cfg = {});

  /// Solve K*u = F using PCG.
  /// Throws SolverError on failure (consistent with EigenSolverBackend).
  [[nodiscard]] std::vector<double>
  solve(const SparseMatrixBuilder::CsrData &K,
        const std::vector<double> &F,
        const FactorRatioCheckPolicy *factor_ratio_policy = nullptr) override;

  [[nodiscard]] std::string_view name() const noexcept override;
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> details() const override;
  // cppcheck-suppress unusedFunction -- queried via SolverBackend in linear_static.cpp
  [[nodiscard]] bool requires_full_symmetric_csr() const noexcept override {
    return true;
  }

  // ── Diagnostics (valid after each solve()) ─────────────────────────────
  /// True if the most recent solve used the full-GPU path.
  [[nodiscard]] bool last_solve_was_full_gpu() const noexcept;
  /// Number of PCG iterations in the most recent solve.
  [[nodiscard]] int last_iteration_count() const noexcept override;
  /// Relative residual ||F - K u|| / ||F|| of the returned solution.
  [[nodiscard]] double last_residual_norm() const noexcept;
  /// Estimated relative error of the returned solution (refinement estimate).
  [[nodiscard]] double last_estimated_error() const noexcept override;
  [[nodiscard]] std::optional<RefinementReport> last_refinement() const noexcept override {
    return last_refinement_;
  }
  /// Applies to the CPU direct fallback below min_dofs_for_gpu; the GPU PCG
  /// is always refined.
  void set_refinement(bool enabled) noexcept override { refine_direct_ = enabled; }

  // Opaque pipeline state — forward-declared here, defined in
  // vulkan_pipelines.hpp. Declared before private: so build_pipelines() (a free
  // function) can name this type.
  struct Pipelines;

private:
  explicit VulkanSolverBackend(VulkanContext ctx,
                               const VulkanSolverConfig &cfg);

  /// Returns true if K + working vectors fit in available VRAM.
  [[nodiscard]] bool
  fits_in_vram(const SparseMatrixBuilder::CsrData &K, std::size_t available) const noexcept;

  VulkanContext ctx_;
  VulkanSolverConfig cfg_;
  bool last_full_gpu_{false};
  int last_iters_{0};
  double last_residual_{0.0};
  std::optional<RefinementReport> last_refinement_;
  bool refine_direct_{false};

  std::unique_ptr<Pipelines> pipelines_;
};

} // namespace vibestran

#endif // HAVE_VULKAN
