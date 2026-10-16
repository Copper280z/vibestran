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
#include "solver/iterative_refinement.hpp"
#include <optional>
#include <vector>
#include <string_view>

namespace vibestran {

struct FactorRatioCheckPolicy;

/// Abstract interface for linear system solvers: K * u = F
class SolverBackend {
public:
    virtual ~SolverBackend() = default;

    /// Solve K*u = F.
    /// K is provided as CSR data (from SparseMatrixBuilder::build_csr()).
    /// factor_ratio_policy applies PARAM,MAXRATIO to backends that can inspect
    /// the diagonal of the actual factorization; unsupported backends ignore it.
    /// Returns displacement vector u.
    [[nodiscard]] virtual std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) = 0;

    /// Human-readable backend name (for logging)
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// Number of solver iterations from the most recent solve(), or -1 if the
    /// backend is a direct solver (no iterations).
    [[nodiscard]] virtual int last_iteration_count() const noexcept { return -1; }

    /// Estimated relative residual from the most recent solve(), or -1.0 if
    /// the backend is a direct solver or the estimate is unavailable.
    [[nodiscard]] virtual double last_estimated_error() const noexcept { return -1.0; }

    /// Iterative-refinement report of the most recent solve(), or nullopt
    /// when it was not refined (a direct solve without set_refinement(true);
    /// see iterative_refinement.hpp).
    [[nodiscard]] virtual std::optional<RefinementReport> last_refinement() const noexcept {
        return std::nullopt;
    }

    /// Opt-in iterative refinement of a direct solve with its factorization
    /// (--refine or PARAM,REFINE,YES; off by default, as in Nastran).
    /// Backends whose accuracy depends on refinement (PCG, fp32
    /// factorizations) always refine and ignore this.
    virtual void set_refinement(bool enabled) noexcept { (void)enabled; }

    /// Returns true when this backend still requires a full symmetric CSR
    /// matrix rather than lower-triangular symmetric storage.
    [[nodiscard]] virtual bool requires_full_symmetric_csr() const noexcept {
        return false;
    }

};

/// CPU backend using Eigen's sparse Cholesky.
/// Solver priority: Apple Accelerate > SuiteSparse CHOLMOD > SimplicialLLT.
/// A single backward-stable solve is accurate to about cond(K) eps.  With
/// set_refinement(true) the solution is refined with the factorization and
/// compensated residuals until the corrections fall to round-off
/// (iterative_refinement.hpp), so it is accurate to working precision unless
/// cond(K) eps approaches 1.
class EigenSolverBackend final : public SolverBackend {
public:
    void set_refinement(bool enabled) noexcept override { refine_ = enabled; }

    [[nodiscard]] std::optional<RefinementReport> last_refinement() const noexcept override {
        return last_refinement_;
    }

    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::string_view name() const noexcept override {
#if defined(HAVE_ACCELERATE)
        return "Apple Accelerate (CPU)";
#elif defined(EIGEN_CHOLMOD_SUPPORT)
        return "SuiteSparse CHOLMOD (CPU)";
#else
        return "Eigen SimplicialLLT (CPU)";
#endif
    }

private:
    bool refine_{false};
    std::optional<RefinementReport> last_refinement_;
};

/// CPU backend using Eigen's Preconditioned Conjugate Gradient with Incomplete
/// Cholesky preconditioning. Memory footprint is O(nnz) — no fill-in from
/// factorization — making it suitable for very large sparse systems that would
/// exhaust memory with a direct solver.
///
/// The solve is restarted PCG with iterative refinement (compensated
/// residuals, iterative_refinement.hpp): it converges when the estimated
/// relative error of the solution falls below the tolerance.  A matrix too
/// ill-conditioned for the preconditioner, or singular, is reported as an
/// error rather than returned inaccurate.
class EigenPCGSolverBackend final : public SolverBackend {
public:
    /// @param tolerance   Target relative error of the solution (corrections
    ///                    ||d||_inf / ||u||_inf); also the relative-residual
    ///                    tolerance of the first CG solve.
    /// @param max_iters   Maximum CG iterations per solve (0 = Eigen default: 2*n).
    explicit EigenPCGSolverBackend(double tolerance = 1e-8, int max_iters = 0)
        : tolerance_(tolerance), max_iters_(max_iters) {}

    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "Eigen PCG + IncompleteCholesky (CPU)";
    }

    /// Number of CG iterations (all solves) used in the most recent solve().
    [[nodiscard]] int last_iteration_count() const noexcept override { return last_iters_; }

    /// Estimated relative error of the solution after the most recent solve().
    [[nodiscard]] double last_estimated_error() const noexcept override { return last_error_; }

    [[nodiscard]] std::optional<RefinementReport> last_refinement() const noexcept override {
        return last_refinement_;
    }

private:
    double tolerance_;
    int    max_iters_;
    int    last_iters_{0};
    double last_error_{0.0};
    std::optional<RefinementReport> last_refinement_;
};

} // namespace vibestran
