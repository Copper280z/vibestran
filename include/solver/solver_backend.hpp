#pragma once
// include/solver/solver_backend.hpp
// Abstract solver backend: separates the linear algebra solve from FE assembly.
//
// GPU readiness:
//   This interface allows swapping the Eigen CPU backend for:
//   - CudaSolverBackend  (cuSOLVER / cuSPARSE)
//   - VulkanSolverBackend (compute shader Cholesky)
// without touching the FE assembly code.

#include "core/notices.hpp"
#include "core/result.hpp"
#include "core/sparse_matrix.hpp"
#include "solver/iterative_refinement.hpp"
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include <string>
#include <string_view>
#include <utility>

namespace vibestran {

struct FactorRatioCheckPolicy;

/// Options for SolverBackend::factor().
struct FactorOptions {
    /// Report a matrix that is not numerically positive definite as an error
    /// instead of falling back to an indefinite (LDLT or LU) factorization.
    bool require_spd{false};
    /// PARAM,MAXRATIO check of the factor diagonal, applied where the backend
    /// can inspect it. Copied by factor(); it need not outlive the call.
    const FactorRatioCheckPolicy* factor_ratio_policy{nullptr};
    /// The matrix holds a dense block, rows coupled to thousands of others
    /// (an air gap's harmonic coupling), so direct backends order with
    /// minimum degree, which eliminates the block last. Nested dissection
    /// (METIS) cuts through the sparse rest and roughly triples the factor's
    /// work. CHOLMOD also stops setting dense rows aside in AMD: it then
    /// estimates AMD's fill as if every such row filled across all columns
    /// and falls back to METIS.
    bool dense_block{false};
    /// CPU direct backends order with minimum degree (AMD) instead of METIS.
    /// For planar meshes of up to about 10^5 unknowns AMD's fill is within
    /// about 15% of nested dissection's and it orders several times faster,
    /// which matters when the pattern changes often (a sliding band at every
    /// MOTION position). cuDSS keeps nested dissection: AMD's deep, narrow
    /// elimination tree leaves its GPU factorization little to do in
    /// parallel (1.5x slower on a planar motor sweep).
    bool minimum_degree{false};
};

/// Work done by a LinearFactor since it was created.
struct FactorStats {
    int symbolic_analyses{0};
    int numeric_factorizations{0};
    int solves{0}; ///< right-hand sides solved
};

/// A factorization of a symmetric matrix K, from SolverBackend::factor().
/// It solves any number of right-hand sides and can be refactored with new
/// values. The symbolic analysis (ordering and symbolic factor) is reused
/// while the sparsity pattern is unchanged.
class LinearFactor {
public:
    virtual ~LinearFactor() = default;

    /// Number of equations.
    [[nodiscard]] virtual int size() const noexcept = 0;

    /// Replace the factorization with that of K, which must have the same
    /// size. The symbolic analysis is kept when K has the pattern of the
    /// matrix last factored and is redone otherwise. On error the factor
    /// cannot solve until a refactor() succeeds.
    [[nodiscard]] virtual Status refactor(const SparseMatrixBuilder::CsrData& K) = 0;

    /// Solve K X = B for nrhs right-hand sides stored column by column in B
    /// (size() * nrhs values). Returns X in the same layout. A nonfinite
    /// solution is an error.
    [[nodiscard]] virtual Result<std::vector<double>> solve(std::span<const double> B,
                                                            int nrhs = 1) = 0;

    /// The factorization in use, e.g. "accelerate-llt" or "simplicial-ldlt".
    [[nodiscard]] virtual std::string_view method() const noexcept = 0;

    /// The fill-reducing ordering of the last symbolic analysis, e.g. "amd"
    /// or "metis"; empty where the factor does not order (an adapter).
    [[nodiscard]] virtual std::string_view ordering() const noexcept { return {}; }

    [[nodiscard]] virtual FactorStats stats() const noexcept = 0;
};

/// Abstract interface for linear system solvers: K * u = F
class SolverBackend {
public:
    virtual ~SolverBackend() = default;

    /// Factor K for repeated solves. The default adapts solve(): it keeps a
    /// copy of K and calls solve(K, b) for each right-hand side, converting
    /// exceptions to errors, so it works with every backend but reuses no
    /// work. Backends with a reusable factorization override it. The factor
    /// may refer to this backend, which must outlive it.
    [[nodiscard]] virtual Result<std::unique_ptr<LinearFactor>> factor(
        const SparseMatrixBuilder::CsrData& K, const FactorOptions& options = {});

    /// True when solve() may fall back to an indefinite factorization (such
    /// as LU) instead of failing. The default factor() then refuses
    /// FactorOptions::require_spd, because it cannot prevent the fallback.
    [[nodiscard]] virtual bool may_factor_indefinite() const noexcept { return false; }

    /// Solve K*u = F. Throws SolverError on failure; factor() reports the
    /// same failures as error values.
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

    /// F06 SOLVER rows beyond the name: method, ordering, precision,
    /// refinement, device and so on.
    [[nodiscard]] virtual std::vector<std::pair<std::string, std::string>> details() const {
        return {};
    }

    /// Warnings about how solves were done (precision or method fallbacks,
    /// MAXRATIO breaches) that the analysis writes to the F06. Factors from
    /// factor() record theirs here too.
    [[nodiscard]] Notices& notices() noexcept { return notices_; }

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

private:
    Notices notices_;
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
    /// Analyzes the pattern once and reuses it in refactor(). Without
    /// require_spd, a matrix that is not positive definite falls back to
    /// LDLT, as solve() does.
    [[nodiscard]] Result<std::unique_ptr<LinearFactor>> factor(
        const SparseMatrixBuilder::CsrData& K, const FactorOptions& options = {}) override;

    void set_refinement(bool enabled) noexcept override { refine_ = enabled; }

    [[nodiscard]] std::optional<RefinementReport> last_refinement() const noexcept override {
        return last_refinement_;
    }

    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> details() const override;

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

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> details() const override;

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
