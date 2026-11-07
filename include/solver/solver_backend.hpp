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
#include "solver/near_nullspace.hpp"
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include <string>
#include <string_view>
#include <utility>

namespace vibestran {

struct FactorRatioCheckPolicy;

namespace detail {
struct AmgPcgResult;
}

/// The execution chosen for an AMG hierarchy, planned from its exact device
/// storage once it is built and against the memory free at that moment.
struct AmgDeviceReport {
    enum class Execution {
        Resident,     ///< The whole hierarchy and CG on the device.
        FineRowSpill, ///< A suffix of the fine operator's rows on the CPU.
        Host,         ///< The hierarchy on the host; the fixed state did not fit.
    };
    Execution execution{Execution::Host};
    int gpu_rows{0};             ///< Fine rows on the device (not reported for Host).
    int cpu_rows{0};             ///< Fine rows on the CPU.
    std::size_t planned_bytes{0}; ///< Device bytes of the plan (0 for Host).
    std::size_t budget_bytes{0};  ///< Free memory less headroom, within any explicit budget.
    /// Device memory the engine took, measured after its first solve; 0 until then.
    std::size_t measured_bytes{0};
    /// The device applies the finest P and R from A, the smoother and T instead
    /// of storing them: the same operator in less memory, for more products.
    bool computed_transfers{false};
};

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

/// What a direct factorization found about K's pivots, by row of K.
struct PivotReport {
    double max_ratio{0.0};          ///< Largest |K_ii / D_i|.
    int max_ratio_row{-1};          ///< Where; -1 when there is none.
    double max_ratio_diagonal{0.0}; ///< K_ii there.
    double max_ratio_pivot{0.0};    ///< D_i there.
    int negative_pivots{0};
    std::vector<int> held_rows;     ///< Exactly zero pivots, whose unknowns were held at zero.
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

    /// The pivots of the last factorization, where the factor can see them.
    [[nodiscard]] virtual std::optional<PivotReport> pivot_report() const { return std::nullopt; }
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

    /// The pivots of the most recent solve(), where the backend can see them.
    [[nodiscard]] virtual std::optional<PivotReport> last_pivot_report() const { return std::nullopt; }

    /// Where the last AMG hierarchy ran; empty if the last solve built none.
    [[nodiscard]] virtual std::optional<AmgDeviceReport> last_amg_device_report() const noexcept {
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

    /// True when solve() of an n-equation system would use set_near_nullspace().
    [[nodiscard]] virtual bool uses_near_nullspace(int n) const noexcept {
        (void)n;
        return false;
    }

    /// The near-nullspace of the matrices of later solve() calls, for their
    /// preconditioner; null for none. factor() does not use it.
    // cppcheck-suppress unusedFunction -- called from linear_static.cpp
    void set_near_nullspace(std::shared_ptr<const NearNullspace> modes) noexcept {
        near_nullspace_ = std::move(modes);
    }

protected:
    [[nodiscard]] const std::shared_ptr<const NearNullspace>& near_nullspace() const noexcept {
        return near_nullspace_;
    }

private:
    Notices notices_;
    std::shared_ptr<const NearNullspace> near_nullspace_;
};

/// The default CPU direct backend, on every platform: the vendored LDL^T library
/// (src/solver/ldlt), supernodal and threaded. It factors without pivoting, tries Cholesky on
/// each front and holds an exactly zero pivot's unknown at zero. A matrix with negative pivots
/// is refactored with threshold pivoting, unless FactorOptions::require_spd makes that an
/// error. last_pivot_report() gives K_ii/D_i and the held rows, which the analyses turn into
/// F06 messages.
class LdltSolverBackend final : public SolverBackend {
public:
    [[nodiscard]] Result<std::unique_ptr<LinearFactor>> factor(
        const SparseMatrixBuilder::CsrData& K, const FactorOptions& options = {}) override;

    void set_refinement(bool enabled) noexcept override { refine_ = enabled; }

    [[nodiscard]] std::optional<RefinementReport> last_refinement() const noexcept override {
        return last_refinement_;
    }

    [[nodiscard]] std::optional<PivotReport> last_pivot_report() const override {
        return last_pivots_;
    }

    /// The analyses apply PARAM,MAXRATIO from last_pivot_report(), so the policy is unused.
    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> details() const override;

    [[nodiscard]] std::string_view name() const noexcept override { return "LDL^T library (CPU)"; }

private:
    bool refine_{false};
    std::optional<RefinementReport> last_refinement_;
    std::optional<PivotReport> last_pivots_;
};

/// CPU backend using Eigen's sparse Cholesky (--backend=cpu-eigen), to check the default one.
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

/// CPU backend: conjugate gradients in fp64, preconditioned by smoothed-aggregation
/// AMG, on OpenMP threads. Memory is O(nnz), with no fill-in from factorization, which
/// suits large systems that would exhaust memory with a direct solver.
///
/// Each solve is refined with compensated residuals (iterative_refinement.hpp) and
/// converges when the estimated relative error of the solution falls below the
/// tolerance. A matrix too ill-conditioned for the preconditioner, or singular, is
/// reported as an error rather than returned inaccurate. The result does not depend
/// on the number of threads, bit for bit.
class CpuPCGSolverBackend final : public SolverBackend {
public:
    /// @param tolerance   Target relative error of the solution (corrections
    ///                    ||d||_inf / ||u||_inf); also the relative-residual
    ///                    tolerance of the first CG solve.
    /// @param max_iters   Maximum CG iterations per solve (0 = default: 10000).
    explicit CpuPCGSolverBackend(double tolerance = 1e-8, int max_iters = 0)
        : tolerance_(tolerance), max_iters_(max_iters) {}

    /// The hierarchy is built once for any number of right-hand sides, and refactor()
    /// with an unchanged pattern keeps its aggregates and transfers.
    [[nodiscard]] Result<std::unique_ptr<LinearFactor>> factor(
        const SparseMatrixBuilder::CsrData& K, const FactorOptions& options = {}) override;

    [[nodiscard]] bool uses_near_nullspace(int) const noexcept override { return true; }

    [[nodiscard]] std::vector<double> solve(
        const SparseMatrixBuilder::CsrData& K,
        const std::vector<double>& F,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "CPU PCG + AMG";
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> details() const override;

    /// Number of CG iterations (all solves) used in the most recent solve.
    [[nodiscard]] int last_iteration_count() const noexcept override { return last_iters_; }

    /// Estimated relative error of the solution after the most recent solve.
    [[nodiscard]] double last_estimated_error() const noexcept override { return last_error_; }

    [[nodiscard]] std::optional<RefinementReport> last_refinement() const noexcept override {
        return last_refinement_;
    }

private:
    [[nodiscard]] Status accept_amg(const detail::AmgPcgResult& result);

    double tolerance_;
    int    max_iters_;
    int    last_iters_{0};
    double last_error_{0.0};
    std::optional<RefinementReport> last_refinement_;
};

} // namespace vibestran
