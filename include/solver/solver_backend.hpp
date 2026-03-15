#pragma once
// include/solver/solver_backend.hpp
// Abstract solver backend: separates the linear algebra solve from FE assembly.
//
// GPU readiness:
//   This interface allows swapping the Eigen CPU backend for:
//   - CudaSolverBackend  (cuSOLVER / cuSPARSE)
//   - VulkanSolverBackend (compute shader Cholesky)
// without touching the FE assembly code.

#include "core/sparse_matrix.hpp"
#include <vector>
#include <string_view>

namespace nastran {

/// Abstract interface for linear system solvers: K * u = F
class SolverBackend {
public:
    virtual ~SolverBackend() = default;

    /// Solve K*u = F.
    /// K is provided as CSR data (from SparseMatrixBuilder::build_csr()).
    /// Returns displacement vector u.
    [[nodiscard]] virtual std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F) = 0;

    /// Human-readable backend name (for logging)
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

/// CPU backend using Eigen's sparse Cholesky (SimplicialLLT)
class EigenSolverBackend final : public SolverBackend {
public:
    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F) override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "Eigen SimplicialLLT (CPU)";
    }
};

} // namespace nastran
