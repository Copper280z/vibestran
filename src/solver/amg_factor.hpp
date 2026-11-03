#pragma once
// A prepared AMG system for SolverBackend::factor(): the equilibrated matrix,
// hierarchy and device workspace are built once and reused for every
// right-hand side. refactor() with an unchanged sparsity pattern keeps the
// aggregates and transfer operators and recomputes only the Galerkin
// operators, smoothers and coarse inverse (AMGCL's rebuild); a changed
// pattern, or a rebuilt hierarchy that needs far more iterations than a
// fresh one did, builds the hierarchy again.
#include "amg_pcg.hpp"
#include "solver/solver_backend.hpp"
#include <algorithm>
#include <functional>

namespace vibestran::detail {
// A backend's cycle and Krylov iteration on a hierarchy it does not own.
class AmgEngine {
public:
    virtual ~AmgEngine() = default;
    virtual std::pair<int, double> solve(const std::vector<double>& b, double tolerance, std::vector<double>& x) = 0;
    // The hierarchy's values changed; its patterns did not.
    virtual void values_changed() = 0;
};

class AmgSystem {
public:
    using EngineFactory = std::function<std::unique_ptr<AmgEngine>(const ScaledAmgMatrix&, const AmgHierarchy&)>;

    AmgSystem(const SparseMatrixBuilder::CsrData& k, double tolerance, EngineFactory make)
        : tolerance_(tolerance), make_(std::move(make)) { build(k); }

    AmgPcgResult solve(const SparseMatrixBuilder::CsrData& k, const std::vector<double>& rhs) {
        const auto start = std::chrono::steady_clock::now();
        auto result = refine_amg(k, rhs, scaled_->scale, tolerance_,
            [&](const std::vector<double>& b, double tol, std::vector<double>& x) { return engine_->solve(b, tol, x); });
        log_amg_timing(k.n, scaled_->matrix->nnz, std::exchange(setup_ms_, 0.0),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), result);
        // Compared only with a solve on a freshly built hierarchy.
        if (!rebuilt_) { if (fresh_iterations_ < 0) fresh_iterations_ = result.iterations; }
        else if (fresh_iterations_ >= 0 && result.iterations > 2 * fresh_iterations_) stale_ = true;
        rebuilt_ = false;
        return result;
    }

    // New values of the matrix last prepared. Returns true when only the
    // values were rebuilt; false when the hierarchy was built again.
    bool refactor(const SparseMatrixBuilder::CsrData& k) {
        const auto start = std::chrono::steady_clock::now();
        if (!stale_) {
            auto scaled = std::make_unique<ScaledAmgMatrix>(k);
            if (hierarchy_->rebuild(scaled->matrix)) {
                // In place: engines refer to this object.
                *scaled_ = std::move(*scaled);
                engine_->values_changed();
                rebuilt_ = true;
                setup_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                log_debug(std::format("[amg-pcg] rebuilt values with the existing transfer operators in {:.3f} ms", setup_ms_));
                return true;
            }
        }
        build(k);
        return false;
    }

    const AmgHierarchy& hierarchy() const { return *hierarchy_; }

private:
    void build(const SparseMatrixBuilder::CsrData& k) {
        const auto start = std::chrono::steady_clock::now();
        engine_.reset();
        scaled_ = std::make_unique<ScaledAmgMatrix>(k);
        hierarchy_ = std::make_unique<AmgHierarchy>(scaled_->matrix);
        log_debug("[amg-pcg] " + hierarchy_->describe());
        engine_ = make_(*scaled_, *hierarchy_);
        fresh_iterations_ = -1;
        stale_ = rebuilt_ = false;
        setup_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    double tolerance_;
    EngineFactory make_;
    std::unique_ptr<ScaledAmgMatrix> scaled_;
    std::unique_ptr<AmgHierarchy> hierarchy_;
    std::unique_ptr<AmgEngine> engine_;
    double setup_ms_{0};
    int fresh_iterations_{-1};
    bool rebuilt_{false}, stale_{false};
};

// LinearFactor over an AmgSystem. accept() records the backend diagnostics
// and turns a solve that missed the accuracy target into an error.
class AmgFactor final : public LinearFactor {
public:
    using Accept = std::function<Status(const AmgPcgResult&)>;
    using Make = std::function<std::unique_ptr<AmgSystem>(const SparseMatrixBuilder::CsrData&)>;

    AmgFactor(SparseMatrixBuilder::CsrData k, Make make, Accept accept)
        : k_(std::move(k)), make_(std::move(make)), accept_(std::move(accept)), system_(make_(k_)) {
        ++stats_.symbolic_analyses;
        ++stats_.numeric_factorizations;
    }

    [[nodiscard]] int size() const noexcept override { return k_.n; }
    [[nodiscard]] std::string_view method() const noexcept override { return "amg-pcg"; }
    [[nodiscard]] FactorStats stats() const noexcept override { return stats_; }

    [[nodiscard]] Status refactor(const SparseMatrixBuilder::CsrData& k) override {
        if (k.n != k_.n)
            return Error{std::format("refactor() of a {}-equation AMG factor with a {}-equation matrix", k_.n, k.n)};
        if (k.row_ptr.size() != std::size_t(k.n) + 1 || k.col_ind.size() != std::size_t(k.nnz) ||
            k.values.size() != std::size_t(k.nnz))
            return Error{std::format("Malformed CSR matrix: n={}, nnz={}", k.n, k.nnz)};
        try {
            k_ = k;
            ++stats_.numeric_factorizations;
            if (!system_) system_ = make_(k_);
            else if (!system_->refactor(k_)) ++stats_.symbolic_analyses;
        } catch (const std::exception& e) {
            system_.reset();
            return Error{std::format("AMG PCG refactor: {}", e.what())};
        }
        return std::nullopt;
    }

    [[nodiscard]] Result<std::vector<double>> solve(std::span<const double> B, int nrhs) override {
        if (!system_) return Error{"AMG PCG factor is unusable after a failed refactor()"};
        const auto n = std::size_t(k_.n);
        if (nrhs < 1 || B.size() != n * std::size_t(nrhs))
            return Error{std::format("Right-hand side has {} values; expected {} x {}", B.size(), n, nrhs)};
        std::vector<double> X;
        X.reserve(B.size());
        for (int c = 0; c < nrhs; ++c) {
            const auto column = B.subspan(std::size_t(c) * n, n);
            AmgPcgResult result;
            try {
                result = system_->solve(k_, std::vector<double>(column.begin(), column.end()));
            } catch (const std::exception& e) {
                return Error{std::format("AMG PCG: {}", e.what())};
            }
            ++stats_.solves;
            if (auto e = accept_(result)) return *e;
            if (!std::ranges::all_of(result.solution, [](double v) { return std::isfinite(v); }))
                return Error{"AMG PCG produced a nonfinite solution"};
            X.insert(X.end(), result.solution.begin(), result.solution.end());
        }
        return X;
    }

private:
    SparseMatrixBuilder::CsrData k_;
    Make make_;
    Accept accept_;
    std::unique_ptr<AmgSystem> system_;
    FactorStats stats_;
};

// The hybrid engine: host hierarchy with an optional device fine prefix.
class HostAmgEngine final : public AmgEngine {
public:
    HostAmgEngine(const ScaledAmgMatrix& scaled, const AmgHierarchy& h, AmgSpmvFactory factory, int max_iterations)
        : scaled_(scaled), h_(h), factory_(std::move(factory)), max_iterations_(max_iterations) { values_changed(); }
    std::pair<int, double> solve(const std::vector<double>& b, double tolerance, std::vector<double>& x) override {
        return amg_->solve(b, tolerance, max_iterations_, x);
    }
    void values_changed() override {
        amg_.reset();
        if (factory_) {
            const auto& a = *scaled_.matrix;
            fine_.n = int(a.nrows); fine_.nnz = int(a.nnz); fine_.symmetry = SymmetryStorage::Full;
            fine_.row_ptr.assign(a.ptr, a.ptr + a.nrows + 1);
            fine_.col_ind.assign(a.col, a.col + a.nnz);
            fine_.values.assign(a.val, a.val + a.nnz);
        }
        amg_ = std::make_unique<HostAmg>(h_, factory_ ? factory_(fine_) : nullptr);
    }
private:
    const ScaledAmgMatrix& scaled_;
    const AmgHierarchy& h_;
    AmgSpmvFactory factory_;
    int max_iterations_;
    SparseMatrixBuilder::CsrData fine_;
    std::unique_ptr<HostAmg> amg_;
};

// Host-hierarchy AMG PCG. The optional factory supplies a partly device-
// resident fine product; every other operator and vector stays on the host.
inline AmgPcgResult solve_hybrid_amg(const SparseMatrixBuilder::CsrData& k, const std::vector<double>& f,
    double tolerance, int max_iterations, AmgSpmvFactory factory) {
    const int limit = max_iterations > 0 ? max_iterations : 10000;
    AmgSystem system(k, tolerance, [&](const ScaledAmgMatrix& scaled, const AmgHierarchy& h) {
        return std::make_unique<HostAmgEngine>(scaled, h, factory, limit);
    });
    return system.solve(k, f);
}
} // namespace vibestran::detail
