// src/solver/vulkan_pcg_internal.hpp  (internal, not installed)
// Forward declarations for the two PCG paths.
// Both functions throw SolverError on unrecoverable failure.
#pragma once
#ifdef HAVE_VULKAN

#include "solver/vulkan_solver_backend.hpp"
#include "vulkan_pipelines.hpp"
#include <vector>

namespace vibestran {

std::vector<double> solve_vulkan_amg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& k, const std::vector<double>& rhs,
    double tolerance, int max_iterations, int& iterations, RefinementReport& report,
    std::size_t budget, bool& fully_resident);

std::vector<double> solve_vulkan_hybrid_amg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& k, const std::vector<double>& rhs,
    std::size_t budget, double tolerance, int max_iterations, int& iterations, RefinementReport& report);

// Matrix storage and descriptors survive all refinement correction solves.
class VulkanPcgWorkspace {
public:
    virtual ~VulkanPcgWorkspace() = default;
    virtual std::vector<double> solve(const std::vector<double>& rhs,
        const VulkanSolverConfig& cfg, int& iterations, double& residual) = 0;
};

std::unique_ptr<VulkanPcgWorkspace> make_full_pcg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& K, bool use_double);

// Defined in vulkan_pcg_tiled.cpp
// Returns solution vector u.  Throws SolverError on failure.
std::unique_ptr<VulkanPcgWorkspace> make_partitioned_pcg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& K, bool use_double, std::size_t budget);

} // namespace vibestran

#endif // HAVE_VULKAN
