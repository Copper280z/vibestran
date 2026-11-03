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
#include "amg_result.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <format>
#include <string>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

namespace vibestran::detail {
using AmgHostCsr = amgcl::backend::builtin<double, int, int>::matrix;

// D^-1/2 K D^-1/2 as full CSR with sorted rows and exact zeros removed.
// Lower-triangle storage is expanded here, in parallel, without an
// intermediate full copy.
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
        for (int i = 0; i < n; ++i) {
            if (scale[i] == 0) throw SolverError(std::format("AMG PCG: missing diagonal at row {} (n={})", i, n));
            count[i + 1] += own[i];
        }
        std::partial_sum(count.begin(), count.end(), count.begin());
        auto& a = *matrix;
        a.set_size(n, n, false);
        a.set_nonzeros(count[n]);
        std::copy(count.begin(), count.end(), a.ptr);
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

// The hierarchy AMGCL's amg<smoothed_aggregation, spai0> builds with
// coarse_enough = 256 (scalar aggregation, eps_strong 0.08, relaxation 1, no
// spectral radius estimate). Block aggregation by node, a rigid-body near-
// nullspace, a looser or stronger threshold and the spectral estimate were
// all slower on the captured plate, motor and thermal matrices.
inline constexpr unsigned kAmgCoarseEnough = 256;

struct AmgLevel {
    std::shared_ptr<AmgHostCsr> A, P, R;
    // SPAI-0 diagonal: a_ii / sum_j a_ij^2 (empty on a directly solved level).
    std::vector<double> smoother;
};

// Levels from fine to coarse. The coarsest level is solved with a dense
// inverse when it has at most coarse_enough rows; otherwise (aggregation
// found no strong couplings) it is smoothed only, as AMGCL does.
class AmgHierarchy {
public:
    explicit AmgHierarchy(std::shared_ptr<AmgHostCsr> fine) {
        std::shared_ptr<AmgHostCsr> a = std::move(fine);
        while (a->nrows > kAmgCoarseEnough) {
            AmgLevel level{a, nullptr, nullptr, spai0(*a)};
            try {
                std::tie(level.P, level.R) = coarsening_.transfer_operators(*a);
            } catch (const amgcl::error::empty_level&) {
                levels.push_back(std::move(level));
                return;
            }
            amgcl::backend::sort_rows(*level.P);
            amgcl::backend::sort_rows(*level.R);
            a = coarsening_.coarse_operator(*a, *level.P, *level.R);
            amgcl::backend::sort_rows(*a);
            levels.push_back(std::move(level));
        }
        levels.push_back({a, nullptr, nullptr, {}});
        factor_coarse();
    }

    // Same transfer operators, new values: AMGCL's rebuild. Returns false when
    // the new fine matrix does not share the original sparsity pattern.
    bool rebuild(std::shared_ptr<AmgHostCsr> fine) {
        const auto& old = *levels.front().A;
        if (fine->nrows != old.nrows || fine->nnz != old.nnz ||
            !std::equal(fine->ptr, fine->ptr + fine->nrows + 1, old.ptr) ||
            !std::equal(fine->col, fine->col + fine->nnz, old.col)) return false;
        std::shared_ptr<AmgHostCsr> a = std::move(fine);
        for (auto& level : levels) {
            level.A = a;
            if (!level.smoother.empty()) level.smoother = spai0(*a);
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
        std::size_t nnz = 0;
        for (const auto& l : levels) {
            out += std::format("{}{}x{} ({} nnz)", out.empty() ? "" : ", ", l.A->nrows, l.A->ncols, l.A->nnz);
            nnz += l.A->nnz + (l.P ? l.P->nnz + l.R->nnz : 0);
        }
        return std::format("{} levels: {}; operator+transfer nnz/fine nnz={:.3f}; coarse {}",
            levels.size(), out, double(nnz) / double(levels.front().A->nnz),
            coarse_direct() ? "dense inverse" : "smoothed");
    }

private:
    static std::vector<double> spai0(const AmgHostCsr& a) {
        std::vector<double> m(a.nrows);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(a.nrows); ++i) {
            double num = 0, den = 0;
            for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
                den += a.val[j] * a.val[j];
                if (a.col[j] == i) num += a.val[j];
            }
            m[i] = num / den;
        }
        return m;
    }

    void factor_coarse() {
        const auto& a = *levels.back().A;
        const auto n = Eigen::Index(a.nrows);
        if (a.nrows > kAmgCoarseEnough) {
            levels.back().smoother = spai0(a);
            coarse_inverse.clear();
            return;
        }
        Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) dense(i, a.col[j]) = a.val[j];
        const Eigen::LLT<Eigen::MatrixXd> llt(dense);
        if (llt.info() != Eigen::Success)
            throw SolverError(std::format("AMG coarse Cholesky failed: n={}, nnz={}, Eigen status={}",
                n, a.nnz, int(llt.info())));
        const Eigen::MatrixXd inverse = llt.solve(Eigen::MatrixXd::Identity(n, n));
        coarse_inverse.assign(inverse.data(), inverse.data() + inverse.size());
    }

    amgcl::coarsening::smoothed_aggregation<amgcl::backend::builtin<double, int, int>> coarsening_;
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
            work_[0].f = r;
            amg_vcycle(*this, h_.levels.size());
            const auto& z = work_[0].y;
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

// The factory's matrix outlives the returned product.
using AmgSpmvFactory = std::function<std::unique_ptr<AmgSpmv>(const SparseMatrixBuilder::CsrData&)>;
} // namespace vibestran::detail
