#pragma once
// Smoothed aggregation for systems: it carries a near-nullspace, with aggregates
// of whole grid points, and a smoother that is stable on any matrix.
//
// Three things differ from AMGCL's scalar defaults, each needed on structures:
//  - AMGCL's block aggregation needs every point to own the same number of
//    equations, which a single partly constrained grid breaks, and real decks
//    mix 6-DOF shells, 3-DOF solids, RBEs and partial SPCs. Grouping by the
//    model's own grid points has no such limit.
//  - AMGCL smooths P with a filtered matrix that lumps weak couplings into the
//    diagonal. That preserves the constant vector, whose row sums stay zero, but
//    not a rigid rotation: the lumped row gains sum a_ij (B_j - B_i), which grows
//    with the lever arm. Here P is smoothed with the whole matrix.
//  - SPAI-0 weights a_ii / sum a_ij^2 exceed 2 / lambda_max on a system, whose
//    many small couplings keep the denominator small while lambda_max grows, so
//    the smoother amplifies high frequencies and the V-cycle is not positive
//    definite, which conjugate gradients need. See l1_jacobi_weights().
//
// The strength graph, the greedy aggregates and the QR of the modes on each
// aggregate are AMGCL's.
#define AMGCL_NO_BOOST
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/tentative_prolongation.hpp>
#include "core/exceptions.hpp"
#include "solver/near_nullspace.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

namespace vibestran::detail {

// Per-row Jacobi weights m_i = (4/3) / (rho g_i), g_i = sum_j |a_ij|. The l1
// norm keeps a dense row (an RBE's independent grid) from weakening every other
// row, as a global spectral radius would. By Gershgorin rho(G^-1 A) <= 1, but it
// is often far less on shells, so it is measured: rho comes from a few power
// iterations, and the margin of 4/3 against the limit of 2 covers their
// underestimate. The start vector and the serial sums make the weights
// independent of the thread count.
inline std::vector<double> l1_jacobi_weights(const amgcl::backend::builtin<double, int, int>::matrix& a) {
    constexpr int kPowerIterations = 15;
    const auto n = std::ptrdiff_t(a.nrows);
    std::vector<double> g(a.nrows, 0.0), x(a.nrows), y(a.nrows), m(a.nrows);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) g[std::size_t(i)] += std::abs(a.val[j]);
        // Quasi-random, so no mode of a structured mesh is missed by symmetry.
        const double golden = 0.6180339887498949 * double(i + 1);
        x[std::size_t(i)] = golden - std::floor(golden) - 0.5;
    }
    const auto norm = [](const std::vector<double>& v) {
        return std::sqrt(std::accumulate(v.begin(), v.end(), 0.0, [](double sum, double e) { return sum + e * e; }));
    };
    double rho = 1;
    for (int it = 0; it < kPowerIterations; ++it) {
        const double scale = 1 / norm(x);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            double sum = 0;
            for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) sum += a.val[j] * x[std::size_t(a.col[j])];
            y[std::size_t(i)] = scale * sum / g[std::size_t(i)];
        }
        rho = norm(y);
        x.swap(y);
    }
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) m[std::size_t(i)] = (4.0 / 3.0) / (rho * g[std::size_t(i)]);
    return m;
}

class ModeAggregation {
public:
    using Csr = amgcl::backend::builtin<double, int, int>::matrix;

    // The modes of the matrix equilibrated with scale: D^-1/2 K D^-1/2 has the
    // nullspace D^1/2 B, and scale holds D^-1/2.
    ModeAggregation(const NearNullspace& modes, std::span<const double> scale)
        : point_(modes.point), points_(modes.points) {
        const auto n = point_.size();
        if (modes.cols <= 0 || modes.modes.size() != n * std::size_t(modes.cols) || scale.size() != n)
            throw SolverError(std::format("ModeAggregation: {} modes of {} values for {} equations and {} scales",
                modes.cols, modes.modes.size(), n, scale.size()));
        nullspace_.cols = modes.cols;
        nullspace_.B = modes.modes;
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(n); ++i)
            for (int c = 0; c < modes.cols; ++c)
                nullspace_.B[std::size_t(i) * std::size_t(modes.cols) + std::size_t(c)] /= scale[std::size_t(i)];
    }

    struct Aggregates {
        std::size_t count{0};
        std::vector<std::ptrdiff_t> id;  // per equation; negative for none
    };

    // Grid points connect when their blocks' largest entries do, as in AMGCL's
    // pointwise aggregation; all of a point's equations share its aggregate.
    // An aggregate with fewer equations than modes cannot hold them and is
    // dropped, leaving its equations to the smoother.
    [[nodiscard]] Aggregates aggregates(const Csr& a) const {
        const auto n = std::ptrdiff_t(a.nrows);
        if (point_.size() != std::size_t(n))
            throw SolverError(std::format("ModeAggregation: {} point numbers for a {}-row matrix", point_.size(), n));
        // Equations by point.
        std::vector<int> start(std::size_t(points_) + 1, 0), equation(point_.size());
        for (const int p : point_) ++start[std::size_t(p) + 1];
        std::partial_sum(start.begin(), start.end(), start.begin());
        {
            auto next = start;
            for (std::ptrdiff_t i = 0; i < n; ++i) equation[std::size_t(next[std::size_t(point_[std::size_t(i)])]++)] = int(i);
        }
        // The point graph: the largest absolute entry of each block.
        auto graph = std::make_shared<Csr>();
        graph->set_size(std::size_t(points_), std::size_t(points_), true);
#pragma omp parallel
        {
            std::vector<int> seen(std::size_t(points_), -1);
#pragma omp for schedule(static)
            for (int p = 0; p < points_; ++p) {
                int width = 0;
                for (int e = start[std::size_t(p)]; e < start[std::size_t(p) + 1]; ++e)
                    for (int j = a.ptr[equation[std::size_t(e)]]; j < a.ptr[equation[std::size_t(e)] + 1]; ++j) {
                        const auto q = std::size_t(point_[std::size_t(a.col[j])]);
                        if (seen[q] != p) { seen[q] = p; ++width; }
                    }
                graph->ptr[std::size_t(p) + 1] = width;
            }
        }
        graph->scan_row_sizes();
        graph->set_nonzeros();
#pragma omp parallel
        {
            std::vector<int> position(std::size_t(points_), -1);
#pragma omp for schedule(static)
            for (int p = 0; p < points_; ++p) {
                const int begin = graph->ptr[std::size_t(p)];
                int end = begin;
                for (int e = start[std::size_t(p)]; e < start[std::size_t(p) + 1]; ++e)
                    for (int j = a.ptr[equation[std::size_t(e)]]; j < a.ptr[equation[std::size_t(e)] + 1]; ++j) {
                        const int q = point_[std::size_t(a.col[j])];
                        const double v = std::abs(a.val[j]);
                        if (position[std::size_t(q)] < begin) {
                            position[std::size_t(q)] = end;
                            graph->col[end] = q;
                            graph->val[end++] = v;
                        } else {
                            auto& m = graph->val[position[std::size_t(q)]];
                            m = std::max(m, v);
                        }
                    }
            }
        }
        amgcl::backend::sort_rows(*graph);

        amgcl::coarsening::plain_aggregates::params prm;
        prm.eps_strong = eps_strong_;
        amgcl::coarsening::plain_aggregates groups(*graph, prm);

        // Keep the aggregates that can hold every mode.
        std::vector<std::ptrdiff_t> size(groups.count, 0), renumber(groups.count, -1);
        for (int p = 0; p < points_; ++p)
            if (groups.id[std::size_t(p)] >= 0)
                size[std::size_t(groups.id[std::size_t(p)])] += start[std::size_t(p) + 1] - start[std::size_t(p)];
        Aggregates out;
        for (std::size_t g = 0; g < groups.count; ++g)
            if (size[g] >= nullspace_.cols) renumber[g] = std::ptrdiff_t(out.count++);
        out.id.resize(std::size_t(n));
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            const auto g = groups.id[std::size_t(point_[std::size_t(i)])];
            out.id[std::size_t(i)] = g < 0 ? std::ptrdiff_t(-1) : renumber[std::size_t(g)];
        }
        return out;
    }

    // The tentative prolongation: on each aggregate, the Q of the modes' QR,
    // whose R become the modes of the next level.
    [[nodiscard]] std::shared_ptr<Csr> tentative(const Aggregates& groups) {
        return amgcl::coarsening::tentative_prolongation<Csr>(point_.size(), groups.count, groups.id, nullspace_, 1);
    }

    // P = (I - M A) P_tent, with M the level's smoother weights.
    [[nodiscard]] std::tuple<std::shared_ptr<Csr>, std::shared_ptr<Csr>>
    transfer_operators(const Csr& a, const std::vector<double>& weights) {
        const auto n = std::ptrdiff_t(a.nrows);
        const Aggregates groups = aggregates(a);
        // Coarsening that would not reduce the system leaves nothing to gain.
        if (groups.count == 0 || groups.count * std::size_t(nullspace_.cols) >= std::size_t(n))
            throw amgcl::error::empty_level();
        const auto initial = tentative(groups);

        Csr smoother;
        smoother.set_size(a.nrows, a.ncols, false);
        smoother.set_nonzeros(a.nnz);
        std::copy(a.ptr, a.ptr + a.nrows + 1, smoother.ptr);
        std::copy(a.col, a.col + a.nnz, smoother.col);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < n; ++i)
            for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j)
                smoother.val[j] = (a.col[j] == i ? 1.0 : 0.0) - weights[std::size_t(i)] * a.val[j];
        auto p = amgcl::backend::product(smoother, *initial);
        auto r = amgcl::backend::transpose(*p);

        // The next level: each aggregate owns one equation per mode.
        const auto cols = nullspace_.cols;
        point_.resize(groups.count * std::size_t(cols));
        for (std::size_t i = 0; i < point_.size(); ++i) point_[i] = int(i / std::size_t(cols));
        points_ = int(groups.count);
        eps_strong_ *= 0.5F;
        return {std::move(p), std::move(r)};
    }

private:
    std::vector<int> point_;
    int points_;
    amgcl::coarsening::nullspace_params nullspace_;
    float eps_strong_{0.08F};
};

} // namespace vibestran::detail
