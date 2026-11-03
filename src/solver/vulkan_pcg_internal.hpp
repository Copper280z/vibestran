// src/solver/vulkan_pcg_internal.hpp  (internal, not installed)
// Forward declarations for the two PCG paths.
// Both functions throw SolverError on unrecoverable failure.
#pragma once
#ifdef HAVE_VULKAN

#include "solver/vulkan_solver_backend.hpp"
#include "vulkan_pipelines.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace vibestran {

namespace detail {
class AmgSpmv;
struct AmgPcgResult;
}

// AMG PCG for n >= 50000 on fp64 devices (vulkan_amg.cpp): device hierarchy
// and CG with a fine-row prefix under the budget, or the host hierarchy when
// the fixed device state does not fit. fully_resident is set whenever the
// hierarchy is (re)built. accept() records each solve's diagnostics and
// returns an error for one that missed the accuracy target.
using VulkanAmgAccept = std::function<Status(const detail::AmgPcgResult&)>;
std::vector<double> solve_vulkan_amg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& k, const std::vector<double>& rhs, double tolerance,
    int max_iterations, std::size_t budget, bool& fully_resident, const VulkanAmgAccept& accept);
// The hierarchy and workspace persist across right-hand sides and
// same-pattern refactor(); budget() is queried whenever it is built.
std::unique_ptr<LinearFactor> make_vulkan_amg_factor(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& k, double tolerance, int max_iterations,
    std::function<std::size_t()> budget, bool& fully_resident, VulkanAmgAccept accept);

// Fine products for the host hierarchy: a GPU row prefix within the budget
// (vulkan_pcg_tiled.cpp).
std::function<std::unique_ptr<detail::AmgSpmv>(const SparseMatrixBuilder::CsrData&)>
vulkan_partitioned_spmv(VulkanContext& ctx, Pipelines& pl, std::size_t budget);

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
