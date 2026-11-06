#pragma once
// Smoothed-aggregation preconditioning for the iterative backends, with
// diagonal equilibration and the same compensated refinement/forward-error
// acceptance used by the other iterative paths.
//
// AMGCL builds the aggregates, smoothed transfer operators and Galerkin
// products on the host. The cycle and the Krylov iteration are the backends'
// own, so they can keep scalars on the device, fuse vector updates into the
// sparse products, and skip the pre-smoothing product with a zero initial
// guess (r = f - A*0), which a generic V-cycle cannot know to omit.
#define AMGCL_NO_BOOST
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include "core/sparse_matrix.hpp"
#include "core/exceptions.hpp"
#include "core/logger.hpp"
#include "solver/iterative_refinement.hpp"
#include "amg_mode_aggregation.hpp"
#include "amg_result.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <format>
#include <string>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace vibestran::detail {
using AmgHostCsr = amgcl::backend::builtin<double, int, int>::matrix;

// D^-1/2 K D^-1/2 as full CSR with sorted rows and exact zeros removed.
// Lower-triangle storage is expanded here, in parallel, without an
// intermediate full copy. A row with no stiffness at all (a DOF nothing connects to)
// gets a unit diagonal here, which K does not have: the solve then leaves it at zero
// unless it is loaded, and a load on it fails refinement against K.
struct ScaledAmgMatrix {
    std::shared_ptr<AmgHostCsr> matrix;
    std::vector<double> scale;

    explicit ScaledAmgMatrix(const SparseMatrixBuilder::CsrData& k)
        : matrix(std::make_shared<AmgHostCsr>()), scale(k.n, 0.0) {
        const int n = k.n;
        const bool lower = k.stores_lower_triangle_only();
        std::vector<int> count(std::size_t(n) + 1, 0), own(n, 0);
        int bad_row = -1;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = k.row_ptr[i]; j < k.row_ptr[i + 1]; ++j) {
                const int c = k.col_ind[j];
                if (c == i) {
                    if (k.values[j] > 0 && std::isfinite(k.values[j])) scale[i] = 1 / std::sqrt(k.values[j]);
                    else std::atomic_ref<int>(bad_row).store(i);
                }
                if (k.values[j] == 0) continue;
                ++own[i];
                if (lower && c != i) std::atomic_ref<int>(count[c + 1]).fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (bad_row >= 0) {
            const auto diag = [&] {
                for (int j = k.row_ptr[bad_row]; j < k.row_ptr[bad_row + 1]; ++j)
                    if (k.col_ind[j] == bad_row) return k.values[j];
                return 0.0;
            }();
            throw SolverError(std::format("AMG PCG: invalid diagonal {} at row {} (n={})", diag, bad_row, n));
        }
        std::vector<int> empty_rows;
        for (int i = 0; i < n; ++i) {
            if (scale[i] == 0) {
                if (own[i] != 0 || count[i + 1] != 0)
                    throw SolverError(std::format("AMG PCG: row {} has off-diagonal entries but no diagonal, so the "
                        "matrix is not positive semidefinite (n={})", i, n));
                scale[i] = 1;
                empty_rows.push_back(i);
                ++own[i];
            }
            count[i + 1] += own[i];
        }
        std::partial_sum(count.begin(), count.end(), count.begin());
        auto& a = *matrix;
        a.set_size(n, n, false);
        a.set_nonzeros(count[n]);
        std::copy(count.begin(), count.end(), a.ptr);
        for (const int i : empty_rows) {
            a.col[a.ptr[i]] = i;
            a.val[a.ptr[i]] = 1;
        }
        std::vector<int> fill(n, 0);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            int pos = a.ptr[i];
            for (int j = k.row_ptr[i]; j < k.row_ptr[i + 1]; ++j) {
                if (k.values[j] == 0) continue;
                const int c = k.col_ind[j];
                const double v = k.values[j] * scale[i] * scale[c];
                a.col[pos] = c; a.val[pos++] = v;
                if (lower && c != i) {
                    const int t = a.ptr[c] + own[c] + std::atomic_ref<int>(fill[c]).fetch_add(1, std::memory_order_relaxed);
                    a.col[t] = i; a.val[t] = v;
                }
            }
        }
        // Transposed entries arrive in thread order; sorting makes the rows
        // (and so every later sum) independent of scheduling.
        amgcl::backend::sort_rows(a);
    }
};

// Without modes, the hierarchy AMGCL's amg<smoothed_aggregation, spai0> builds
// with coarse_enough = 256 (scalar aggregation, eps_strong 0.08, relaxation 1, no
// spectral radius estimate), with the weights bounded for stability where SPAI-0
// passes kAmgSmootherBound. With modes, ModeAggregation.
inline constexpr unsigned kAmgCoarseEnough = 256;

struct AmgLevel {
    std::shared_ptr<AmgHostCsr> A, P, R;
    // Diagonal smoother weights, within kAmgSmootherBound (empty on a directly
    // solved level).
    std::vector<double> smoother;
    // With modes, on the finest level: T of P = (I - M A) T, from which a device
    // can apply P and R without storing them.
    std::shared_ptr<const AmgTentative> tentative{};
};

// Levels from fine to coarse. The coarsest level is solved with a dense
// inverse when it has at most coarse_enough rows; otherwise (aggregation
// found no strong couplings) it is smoothed only, as AMGCL does.
class AmgHierarchy {
public:
    // With modes (of K; scale is the equilibration of fine), aggregates are
    // whole grid points and carry the modes down the levels.
    explicit AmgHierarchy(std::shared_ptr<AmgHostCsr> fine, const NearNullspace* modes = nullptr,
        std::span<const double> scale = {}, unsigned coarse_enough = kAmgCoarseEnough)
        : coarse_enough_(coarse_enough), system_(modes != nullptr) {
        std::optional<ModeAggregation> by_mode;
        if (modes) by_mode.emplace(*modes, scale);
        std::shared_ptr<AmgHostCsr> a = std::move(fine);
        while (a->nrows > coarse_enough_) {
            AmgLevel level{a, nullptr, nullptr, smoother(*a)};
            try {
                std::tie(level.P, level.R) = by_mode ? by_mode->transfer_operators(*a, level.smoother)
                                                     : coarsening_.transfer_operators(*a);
            } catch (const amgcl::error::empty_level&) {
                levels.push_back(std::move(level));
                return;
            }
            if (by_mode && levels.empty()) level.tentative = by_mode->patches();
            amgcl::backend::sort_rows(*level.P);
            amgcl::backend::sort_rows(*level.R);
            a = coarsening_.coarse_operator(*a, *level.P, *level.R);
            amgcl::backend::sort_rows(*a);
            levels.push_back(std::move(level));
        }
        levels.push_back({a, nullptr, nullptr, {}});
        factor_coarse();
    }

    // Same aggregates, new values: AMGCL's rebuild, with the finest transfer of a
    // hierarchy with modes smoothed again. Returns false when the new fine matrix
    // does not share the original sparsity pattern.
    bool rebuild(std::shared_ptr<AmgHostCsr> fine) {
        const auto& old = *levels.front().A;
        if (fine->nrows != old.nrows || fine->nnz != old.nnz ||
            !std::equal(fine->ptr, fine->ptr + fine->nrows + 1, old.ptr) ||
            !std::equal(fine->col, fine->col + fine->nnz, old.col)) return false;
        std::shared_ptr<AmgHostCsr> a = std::move(fine);
        for (auto& level : levels) {
            level.A = a;
            if (!level.smoother.empty()) level.smoother = smoother(*a);
            // A device applying P from A and M uses the new values, so stored P
            // must too: which one a plan chose must not change the operator.
            if (level.tentative) {
                std::tie(level.P, level.R) = smoothed_transfer(*a, level.smoother, *level.tentative->csr());
                amgcl::backend::sort_rows(*level.P);
                amgcl::backend::sort_rows(*level.R);
            }
            if (level.P) {
                a = coarsening_.coarse_operator(*a, *level.P, *level.R);
                amgcl::backend::sort_rows(*a);
            }
        }
        if (!coarse_inverse.empty()) factor_coarse();
        return true;
    }

    std::vector<AmgLevel> levels;
    // Dense inverse of the coarsest operator, row-major (it is symmetric).
    std::vector<double> coarse_inverse;

    bool coarse_direct() const { return !coarse_inverse.empty(); }
    std::string describe() const {
        std::string out;
        std::size_t operators = 0, transfers = 0, rows = 0;
        for (const auto& l : levels) {
            out += std::format("{}{}x{} ({} nnz", out.empty() ? "" : ", ", l.A->nrows, l.A->ncols, l.A->nnz);
            if (l.P) out += std::format(", P {} nnz, R {} nnz", l.P->nnz, l.R->nnz);
            out += ")";
            operators += l.A->nnz;
            transfers += l.P ? l.P->nnz + l.R->nnz : 0;
            rows += l.A->nrows;
        }
        const auto fine_nnz = double(levels.front().A->nnz);
        return std::format("{} levels: {}; operator+transfer nnz/fine nnz={:.3f}; operator complexity={:.3f}, "
            "transfer complexity={:.3f}, grid complexity={:.3f}; coarse {}",
            levels.size(), out, double(operators + transfers) / fine_nnz, double(operators) / fine_nnz,
            double(transfers) / fine_nnz, double(rows) / double(levels.front().A->nrows),
            coarse_direct() ? "dense inverse" : "smoothed");
    }

private:
    std::vector<double> smoother(const AmgHostCsr& a) const { return system_ ? l1_jacobi_weights(a) : bounded_spai0_weights(a); }

    void factor_coarse() {
        const auto& a = *levels.back().A;
        const auto n = Eigen::Index(a.nrows);
        if (a.nrows > coarse_enough_) {
            levels.back().smoother = smoother(a);
            coarse_inverse.clear();
            return;
        }
        Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) dense(i, a.col[j]) = a.val[j];
        const Eigen::LLT<Eigen::MatrixXd> llt(dense);
        Eigen::MatrixXd inverse;
        if (llt.info() == Eigen::Success) {
            inverse = llt.solve(Eigen::MatrixXd::Identity(n, n));
        } else {
            // Singular to working precision, as a free-floating part makes it: the
            // pseudo-inverse, which stays positive semidefinite. With a load the matrix
            // can balance, conjugate gradients converge in its range; with one it can't,
            // refinement against K rejects the result.
            const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(dense);
            const Eigen::VectorXd& values = eigen.eigenvalues();
            const double cutoff = double(n) * std::numeric_limits<double>::epsilon() * values.cwiseAbs().maxCoeff();
            const Eigen::VectorXd inverted = values.unaryExpr([cutoff](double v) { return v > cutoff ? 1 / v : 0.0; });
            inverse = eigen.eigenvectors() * inverted.asDiagonal() * eigen.eigenvectors().transpose();
            log_debug(std::format("[amg-pcg] coarsest level of {} rows is singular: using its pseudo-inverse", n));
        }
        coarse_inverse.assign(inverse.data(), inverse.data() + inverse.size());
    }

    amgcl::coarsening::smoothed_aggregation<amgcl::backend::builtin<double, int, int>> coarsening_;
    unsigned coarse_enough_;
    bool system_;
};

// One V-cycle with one pre- and one post-smoothing step, starting from a zero
// guess. Ops owns three vectors per level, f (input), x and y (output):
//   smooth_zero(l):  x = M f                 (M (f - A*0); no product)
//   residual(l):     y = f - A x             (y as scratch)
//   restriction(l):  f[l+1] = R y
//   coarse(L):       y = C^-1 f, or two smoothing steps
//   prolongation(l): x += P y[l+1]
//   smooth_post(l):  y = x + M (f - A x)
template<class Ops> void amg_vcycle(Ops& ops, std::size_t levels) {
    for (std::size_t l = 0; l + 1 < levels; ++l) {
        ops.smooth_zero(l);
        ops.residual(l);
        ops.restriction(l);
    }
    ops.coarse(levels - 1);
    for (std::size_t l = levels - 1; l-- > 0;) {
        ops.prolongation(l);
        ops.smooth_post(l);
    }
}

// Relative residual targets of the inner CG solves on the equilibrated system.
// The initial solve, then each refinement correction. Acceptance is decided by
// refinement against the original fp64 matrix.
inline constexpr double kAmgCorrectionTolerance = 1e-3;

// Refinement driver. inner(b, tolerance, x) solves the equilibrated system for
// a right-hand side scaled to unit infinity norm and returns its iterations.
template<class Inner>
AmgPcgResult refine_amg(const SparseMatrixBuilder::CsrData& original, const std::vector<double>& rhs,
    const std::vector<double>& scale, double tolerance, Inner&& inner) {
    AmgPcgResult result;
    bool first = true;
    const int n = original.n;
    const auto correction = [&](const std::vector<double>& f) {
        std::vector<double> host(n);
        double norm = 0;
        for (int i = 0; i < n; ++i) {
            host[i] = f[i] * scale[i];
            norm = std::max(norm, std::abs(host[i]));
        }
        if (norm == 0) return std::vector<double>(n, 0);
        std::ranges::transform(host, host.begin(), [norm](double v) { return v / norm; });
        const double target = first ? std::min(tolerance, 1e-8) : kAmgCorrectionTolerance;
        first = false;
        std::vector<double> x(n);
        const auto [iterations, residual] = inner(host, target, x);
        result.iterations += iterations;
        for (int i = 0; i < n; ++i) x[i] *= norm * scale[i];
        log_debug(std::format("[amg-pcg] inner iterations={}, scaled residual={:.3e}", iterations, residual));
        return x;
    };
    result.solution = correction(rhs);
    RefinementOptions options;
    options.tolerance = tolerance;
    options.max_corrections = 20;
    result.refinement = refine_solution(original, rhs, result.solution, correction, options);
    return result;
}

inline void log_amg_timing(int n, std::size_t nnz, double setup_ms, double solve_ms, const AmgPcgResult& r) {
    log_debug(std::format("[amg-pcg] n={}, nnz={}, setup={:.3f} ms, iteration/refinement={:.3f} ms, iterations={}, estimated error={:.3e}",
        n, nnz, setup_ms, solve_ms, r.iterations, r.refinement.estimated_error));
}

// Host V-cycle and CG for the hybrid path, whose fine product may use a
// device prefix (AmgSpmv); every coarse level and vector stays in host RAM.
class AmgSpmv {
public:
    virtual ~AmgSpmv() = default;
    virtual void multiply(std::span<const double> x, std::span<double> y) = 0;
};

class HostAmg {
public:
    HostAmg(const AmgHierarchy& h, std::unique_ptr<AmgSpmv> fine) : h_(h), fine_(std::move(fine)) {
        for (const auto& l : h.levels) {
            const auto n = l.A->nrows;
            work_.push_back({std::vector<double>(n), std::vector<double>(n), std::vector<double>(n), std::vector<double>(n)});
        }
    }

    // y = A_l x; the fine level may run partly on the device.
    void multiply(std::size_t l, const std::vector<double>& x, std::vector<double>& y) {
        if (l == 0 && fine_) fine_->multiply(x, y);
        else product(*h_.levels[l].A, x, y);
    }
    void smooth_zero(std::size_t l) {
        auto& w = work_[l];
        const auto& m = h_.levels[l].smoother;
        each(w.x.size(), [&](std::ptrdiff_t i) { w.x[i] = m[i] * w.f[i]; });
    }
    void residual(std::size_t l) {
        auto& w = work_[l];
        multiply(l, w.x, w.t);
        each(w.y.size(), [&](std::ptrdiff_t i) { w.y[i] = w.f[i] - w.t[i]; });
    }
    void restriction(std::size_t l) { product(*h_.levels[l].R, work_[l].y, work_[l + 1].f); }
    void coarse(std::size_t l) {
        auto& w = work_[l];
        if (!h_.coarse_direct()) { smooth_zero(l); smooth_post(l); return; }
        const auto n = std::ptrdiff_t(w.f.size());
        each(n, [&](std::ptrdiff_t i) {
            double sum = 0;
            for (std::ptrdiff_t j = 0; j < n; ++j) sum += h_.coarse_inverse[i * n + j] * w.f[j];
            w.y[i] = sum;
        });
    }
    void prolongation(std::size_t l) {
        auto& w = work_[l];
        product(*h_.levels[l].P, work_[l + 1].y, w.t);
        each(w.x.size(), [&](std::ptrdiff_t i) { w.x[i] += w.t[i]; });
    }
    void smooth_post(std::size_t l) {
        auto& w = work_[l];
        multiply(l, w.x, w.t);
        const auto& m = h_.levels[l].smoother;
        each(w.y.size(), [&](std::ptrdiff_t i) { w.y[i] = w.x[i] + m[i] * (w.f[i] - w.t[i]); });
    }

    // z = B f: one application of the preconditioner. z is valid until the next.
    const std::vector<double>& cycle(const std::vector<double>& f) {
        work_[0].f = f;
        amg_vcycle(*this, h_.levels.size());
        return work_[0].y;
    }

    // PCG on the equilibrated fine operator from x = 0, as AMGCL's CG.
    std::pair<int, double> solve(const std::vector<double>& b, double tolerance, int max_iterations,
                                 std::vector<double>& x) {
        const auto n = b.size();
        std::vector<double> r = b, p(n), q(n);
        std::fill(x.begin(), x.end(), 0.0);
        const double norm_b = std::sqrt(dot(b, b));
        double res = norm_b, rho = 0;
        int it = 0;
        for (; it < max_iterations && res > tolerance * norm_b; ++it) {
            const auto& z = cycle(r);
            const double rho_new = dot(r, z);
            const double beta = it ? rho_new / rho : 0.0;
            rho = rho_new;
            each(n, [&](std::ptrdiff_t i) { p[i] = z[i] + beta * p[i]; });
            multiply(0, p, q);
            const double pq = dot(p, q);
            if (!(pq > 0)) break;
            const double alpha = rho / pq;
            each(n, [&](std::ptrdiff_t i) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; });
            res = std::sqrt(dot(r, r));
        }
        return {it, norm_b > 0 ? res / norm_b : 0.0};
    }

private:
    struct Work { std::vector<double> f, x, y, t; };
    template<class F> static void each(std::size_t n, F&& f) {
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(n); ++i) f(i);
    }
    static void product(const AmgHostCsr& a, const std::vector<double>& x, std::vector<double>& y) {
        each(a.nrows, [&](std::ptrdiff_t i) {
            double sum = 0;
            for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) sum += a.val[j] * x[a.col[j]];
            y[i] = sum;
        });
    }
    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
        for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(a.size()); ++i) sum += a[i] * b[i];
        return sum;
    }

    const AmgHierarchy& h_;
    std::unique_ptr<AmgSpmv> fine_;
    std::vector<Work> work_;
};

// Bilinear-form checks of the applied cycle B on a system too large to form it.
// With rho(M A) < 2 on every level and an SPD coarse solve, B is SPD and the
// eigenvalues of B A lie in (0, 1], so E = I - B A has A-norm below 1. Each number
// is the worst seen, on fixed pseudo-random vectors.
struct AmgCycleProbe {
    double asymmetry{0};     // |x'B y - y'B x| / sqrt(x'B x y'B y)
    double min_energy{1e300}; // x'B x / x'x; positive
    double max_energy{0};    // z'A z / f'z with z = B f; at most 1
    double contraction{0};   // ||E x||_A / ||x||_A over repeated cycles; at most 1
    double last_contraction{0}; // the same on the last cycle: the multigrid rate

    [[nodiscard]] bool valid(double tolerance = 1e-8) const {
        return asymmetry <= tolerance && min_energy > 0 && max_energy <= 1 + tolerance && contraction <= 1 + tolerance;
    }
};

inline AmgCycleProbe probe_amg_cycle(const AmgHierarchy& h, int vectors = 4, int cycles = 12) {
    HostAmg amg(h, nullptr);
    const auto n = h.levels.front().A->nrows;
    const auto dot = [](const std::vector<double>& a, const std::vector<double>& b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    };
    std::mt19937_64 random(20261103);
    std::normal_distribution<double> normal;
    const auto vector = [&] {
        std::vector<double> v(n);
        std::ranges::generate(v, [&] { return normal(random); });
        return v;
    };
    AmgCycleProbe out;
    std::vector<double> az(n);
    for (int v = 0; v < vectors; ++v) {
        const auto x = vector(), y = vector();
        const auto bx = amg.cycle(x);
        const auto by = amg.cycle(y);
        const double xbx = dot(x, bx), yby = dot(y, by);
        out.asymmetry = std::max(out.asymmetry, std::abs(dot(x, by) - dot(y, bx)) / std::sqrt(std::abs(xbx * yby)));
        out.min_energy = std::min(out.min_energy, xbx / dot(x, x));
        amg.multiply(0, bx, az);
        out.max_energy = std::max(out.max_energy, dot(bx, az) / xbx);
    }
    // Multigrid iteration on A e = 0 from a random error.
    auto e = vector();
    amg.multiply(0, e, az);
    double energy = dot(e, az);
    for (int c = 0; c < cycles; ++c) {
        const auto& correction = amg.cycle(az);
        std::transform(e.begin(), e.end(), correction.begin(), e.begin(), std::minus<>{});
        amg.multiply(0, e, az);
        const double next = dot(e, az);
        out.last_contraction = std::sqrt(next / energy);
        out.contraction = std::max(out.contraction, out.last_contraction);
        energy = next;
    }
    return out;
}

// The factory's matrix outlives the returned product.
using AmgSpmvFactory = std::function<std::unique_ptr<AmgSpmv>(const SparseMatrixBuilder::CsrData&)>;
} // namespace vibestran::detail
