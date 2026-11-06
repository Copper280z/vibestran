#pragma once
// Smoothed aggregation for systems: it carries a near-nullspace, with aggregates
// of whole grid points, and the smoother weights with a bound that holds for any
// SPD matrix.
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
//  - SPAI-0 weights exceed 2 / lambda_max on a system, so the smoother amplifies
//    high frequencies and the V-cycle is not positive definite. See
//    kAmgSmootherBound and l1_jacobi_weights().
//
// The strength graph, the greedy aggregates and the QR of the modes on each
// aggregate are AMGCL's. The tentative prolongation is AMGCL's too, except that
// an aggregate leaves out a NearNullspace optional column it cannot use, so
// aggregates can differ in their number of coarse equations.
//
// P = (I - M A) T and R = P^T need not be stored: a device that holds A and M can
// apply them from T alone, which is the same operator for one product with A per
// transfer. T then takes far less memory than P and R. See AmgTentative.
#define AMGCL_NO_BOOST
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/tentative_prolongation.hpp>
#include <amgcl/detail/qr.hpp>
#include "core/exceptions.hpp"
#include "solver/near_nullspace.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

namespace vibestran::detail {

// A V-cycle with one pre- and one post-smoothing step and an SPD coarse solve is
// SPD, as conjugate gradients needs, when rho(M A) < 2 on every level. For SPD A
// and g_i = sum_j |a_ij|, Gershgorin gives rho(G^-1 A) <= 1, and for diagonal M
//   rho(M A) = max y'A y / y'M^-1 y <= w rho(G^-1 A) <= w   whenever m_i <= w / g_i.
// So this bound certifies any weights below it, whatever the matrix.
inline constexpr double kAmgSmootherBound = 1.9;

inline std::vector<double> amg_row_l1_norms(const amgcl::backend::builtin<double, int, int>::matrix& a) {
    std::vector<double> g(a.nrows, 0.0);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(a.nrows); ++i)
        for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) g[std::size_t(i)] += std::abs(a.val[j]);
    return g;
}

// Per-row Jacobi weights m_i = w / g_i. The l1 norm keeps a dense row (an RBE's
// independent grid) from weakening every other row, as a global spectral radius
// would. rho(G^-1 A) is often far below 1 on structures, and w = (4/3) / rho
// smooths best, so rho comes from a few power iterations. They only tune w: an
// underestimate cannot pass kAmgSmootherBound, and rho <= 1 is a floor of 4/3.
// The start vector and the serial sums make the weights independent of the
// thread count.
inline std::vector<double> l1_jacobi_weights(const amgcl::backend::builtin<double, int, int>::matrix& a) {
    constexpr int kPowerIterations = 15;
    const auto n = std::ptrdiff_t(a.nrows);
    const std::vector<double> g = amg_row_l1_norms(a);
    std::vector<double> x(a.nrows), y(a.nrows), m(a.nrows);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
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
    // In this order a rho of zero or NaN also lands on the bound.
    const double w = std::min(kAmgSmootherBound, (4.0 / 3.0) / std::min(rho, 1.0));
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) m[std::size_t(i)] = w / g[std::size_t(i)];
    return m;
}

// SPAI-0 weights a_ii / sum_j a_ij^2. These suit a scalar Laplacian, but on a
// system many small couplings keep the denominator small while lambda_max grows:
// m lambda_max reaches 2.9 on a hexahedral bar.
inline std::vector<double> spai0_weights(const amgcl::backend::builtin<double, int, int>::matrix& a) {
    std::vector<double> m(a.nrows);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(a.nrows); ++i) {
        double num = 0, den = 0;
        for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
            den += a.val[j] * a.val[j];
            if (a.col[j] == i) num += a.val[j];
        }
        m[std::size_t(i)] = num / den;
    }
    return m;
}

// SPAI-0 where it is within kAmgSmootherBound, which on a diagonally dominant
// row it always is: a_ii m_i <= 1 while g_i <= 2 a_ii.
inline std::vector<double> bounded_spai0_weights(const amgcl::backend::builtin<double, int, int>::matrix& a) {
    std::vector<double> m = spai0_weights(a);
    const std::vector<double> g = amg_row_l1_norms(a);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(a.nrows); ++i)
        m[std::size_t(i)] = std::min(m[std::size_t(i)], kAmgSmootherBound / g[std::size_t(i)]);
    return m;
}

// The tentative prolongation T of one transfer, stored once for T and T^T as one
// dense block of values per aggregate, a row per equation, with no column indices.
// Rows of equations in no aggregate follow in a last patch of no columns.
struct AmgTentative {
    // Per patch, and once more to end the last: first row, first value, first
    // column. Four ints each, so that devices can index it as a flat array.
    std::vector<std::array<int, 4>> patches;
    std::vector<int> equation;  // the fine equation of each row
    std::vector<int> row_patch; // the patch of each row
    std::vector<int> col_patch; // the patch of each coarse equation
    std::vector<double> values;
    std::size_t fine_rows{0};

    // Device bytes with values of value_bytes each.
    [[nodiscard]] std::size_t bytes(std::size_t value_bytes) const {
        return values.size() * value_bytes + (equation.size() + row_patch.size() + col_patch.size()) * sizeof(int) +
            patches.size() * sizeof(patches[0]);
    }

    // T in CSR.
    [[nodiscard]] std::shared_ptr<amgcl::backend::builtin<double, int, int>::matrix> csr() const {
        auto t = std::make_shared<amgcl::backend::builtin<double, int, int>::matrix>();
        t->set_size(fine_rows, col_patch.size());
        t->ptr[0] = 0;
        std::fill(t->ptr + 1, t->ptr + fine_rows + 1, 0);
        for (std::size_t g = 0; g + 1 < patches.size(); ++g)
            for (int row = patches[g][0]; row < patches[g + 1][0]; ++row)
                t->ptr[equation[std::size_t(row)] + 1] = patches[g + 1][2] - patches[g][2];
        t->scan_row_sizes();
        t->set_nonzeros();
        for (std::size_t g = 0; g + 1 < patches.size(); ++g) {
            const int width = patches[g + 1][2] - patches[g][2];
            for (int row = patches[g][0]; row < patches[g + 1][0]; ++row) {
                const auto at = t->ptr[equation[std::size_t(row)]];
                for (int k = 0; k < width; ++k) {
                    t->col[at + k] = patches[g][2] + k;
                    t->val[at + k] = values[std::size_t(patches[g][1] + (row - patches[g][0]) * width + k)];
                }
            }
        }
        return t;
    }
};

// P = (I - M A) T and R = P^T, with M the level's smoother weights.
inline std::tuple<std::shared_ptr<amgcl::backend::builtin<double, int, int>::matrix>,
                  std::shared_ptr<amgcl::backend::builtin<double, int, int>::matrix>>
smoothed_transfer(const amgcl::backend::builtin<double, int, int>::matrix& a, const std::vector<double>& weights,
                  const amgcl::backend::builtin<double, int, int>::matrix& t) {
    amgcl::backend::builtin<double, int, int>::matrix smoother;
    smoother.set_size(a.nrows, a.ncols, false);
    smoother.set_nonzeros(a.nnz);
    std::copy(a.ptr, a.ptr + a.nrows + 1, smoother.ptr);
    std::copy(a.col, a.col + a.nnz, smoother.col);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(a.nrows); ++i)
        for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j)
            smoother.val[j] = (a.col[j] == i ? 1.0 : 0.0) - weights[std::size_t(i)] * a.val[j];
    auto p = amgcl::backend::product(smoother, t);
    auto r = amgcl::backend::transpose(*p);
    return {std::move(p), std::move(r)};
}

class ModeAggregation {
public:
    using Csr = amgcl::backend::builtin<double, int, int>::matrix;

    // The modes of the matrix equilibrated with scale: D^-1/2 K D^-1/2 has the
    // nullspace D^1/2 B, and scale holds D^-1/2.
    ModeAggregation(const NearNullspace& modes, std::span<const double> scale)
        : point_(modes.point), points_(modes.points), optional_(modes.optional_cols) {
        const auto n = point_.size();
        if (modes.cols <= 0 || modes.optional_cols < 0 || modes.optional_cols > modes.cols ||
            modes.modes.size() != n * std::size_t(modes.cols) || scale.size() != n)
            throw SolverError(std::format("ModeAggregation: {} modes ({} optional) of {} values for {} equations and {} scales",
                modes.cols, modes.optional_cols, modes.modes.size(), n, scale.size()));
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
    // An aggregate with fewer equations than the modes it has cannot hold them
    // and is dropped, leaving its equations to the smoother.
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
        // How many optional columns are zero on the whole of each aggregate.
        std::vector<std::ptrdiff_t> absent(groups.count, optional_);
        if (optional_ > 0) {
            const auto cols = std::size_t(nullspace_.cols);
            std::vector<char> seen(groups.count * std::size_t(optional_), 0);
            for (std::size_t i = 0; i < std::size_t(n); ++i) {
                const auto g = groups.id[std::size_t(point_[i])];
                if (g < 0) continue;
                for (int c = 0; c < optional_; ++c) {
                    if (nullspace_.B[i * cols + cols - std::size_t(optional_) + std::size_t(c)] == 0.0) continue;
                    char& flag = seen[std::size_t(g) * std::size_t(optional_) + std::size_t(c)];
                    if (!flag) { flag = 1; --absent[std::size_t(g)]; }
                }
            }
        }
        Aggregates out;
        for (std::size_t g = 0; g < groups.count; ++g)
            if (size[g] >= nullspace_.cols - absent[g]) renumber[g] = std::ptrdiff_t(out.count++);
        out.id.resize(std::size_t(n));
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            const auto g = groups.id[std::size_t(point_[std::size_t(i)])];
            out.id[std::size_t(i)] = g < 0 ? std::ptrdiff_t(-1) : renumber[std::size_t(g)];
        }
        return out;
    }

    // The tentative prolongation: on each aggregate, the Q of the modes' QR,
    // whose R become the modes of the next level. Every column of the modes stays
    // on the next level, so an optional column an aggregate leaves out for being
    // in the span of the others is still carried there, as its coordinates.
    [[nodiscard]] std::shared_ptr<Csr> tentative(const Aggregates& groups) {
        const auto n = point_.size();
        const auto cols = std::size_t(nullspace_.cols);
        // Equations by aggregate, keeping their order within it.
        std::vector<std::size_t> start(groups.count + 1, 0);
        for (const auto g : groups.id)
            if (g >= 0) ++start[std::size_t(g) + 1];
        std::partial_sum(start.begin(), start.end(), start.begin());
        std::vector<int> equation(start.back());
        {
            auto next = start;
            for (std::size_t i = 0; i < n; ++i)
                if (groups.id[i] >= 0) equation[next[std::size_t(groups.id[i])]++] = int(i);
        }
        // Each aggregate's columns, then where its coarse equations start.
        std::vector<std::vector<int>> kept(groups.count);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t g = 0; g < std::ptrdiff_t(groups.count); ++g)
            kept[std::size_t(g)] = kept_columns(std::size_t(g), start, equation);
        coarse_start_.assign(groups.count + 1, 0);
        for (std::size_t g = 0; g < groups.count; ++g) coarse_start_[g + 1] = coarse_start_[g] + kept[g].size();
        const auto coarse = coarse_start_.back();

        auto p = std::make_shared<Csr>();
        p->set_size(n, coarse);
        p->ptr[0] = 0;
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(n); ++i) {
            const auto g = groups.id[std::size_t(i)];
            p->ptr[i + 1] = g < 0 ? 0 : int(kept[std::size_t(g)].size());
        }
        p->scan_row_sizes();
        p->set_nonzeros();

        std::vector<double> next(coarse * cols, 0.0);
#pragma omp parallel
        {
            amgcl::detail::QR<double> qr;
            std::vector<double> part;
#pragma omp for schedule(static)
            for (std::ptrdiff_t ig = 0; ig < std::ptrdiff_t(groups.count); ++ig) {
                const auto g = std::size_t(ig);
                const auto& use = kept[g];
                const auto first = start[g];
                const auto d = int(start[g + 1] - first), m = int(use.size());
                const auto value = [&](int r, std::size_t c) {
                    return nullspace_.B[std::size_t(equation[first + std::size_t(r)]) * cols + c];
                };
                part.resize(std::size_t(d) * use.size());
                for (int k = 0; k < m; ++k)
                    for (int r = 0; r < d; ++r)
                        part[std::size_t(r + d * k)] = value(r, std::size_t(use[std::size_t(k)]));
                qr.factorize(d, m, part.data(), amgcl::detail::col_major);
                const auto base = coarse_start_[g];
                for (int k = 0; k < m; ++k)
                    for (int c = 0; c < m; ++c)
                        next[(base + std::size_t(k)) * cols + std::size_t(use[std::size_t(c)])] = qr.R(k, c);
                // The coordinates of the columns the aggregate left out.
                for (std::size_t c = cols - std::size_t(optional_); c < cols; ++c) {
                    if (std::ranges::find(use, int(c)) != use.end()) continue;
                    for (int k = 0; k < m; ++k) {
                        double sum = 0;
                        for (int r = 0; r < d; ++r) sum += qr.Q(r, k) * value(r, c);
                        next[(base + std::size_t(k)) * cols + c] = sum;
                    }
                }
                for (int r = 0; r < d; ++r) {
                    const auto row = p->ptr[equation[first + std::size_t(r)]];
                    for (int k = 0; k < m; ++k) {
                        p->col[row + k] = int(base) + k;
                        p->val[row + k] = qr.Q(r, k);
                    }
                }
            }
        }
        nullspace_.B.swap(next);
        auto patches = std::make_shared<AmgTentative>();
        patches->fine_rows = n;
        patches->values.resize(p->nnz);
        patches->equation = equation;
        patches->row_patch.resize(equation.size());
        patches->col_patch.resize(coarse);
        patches->patches.reserve(groups.count + 2);
        std::size_t value = 0;
        for (std::size_t g = 0; g < groups.count; ++g) {
            patches->patches.push_back({int(start[g]), int(value), int(coarse_start_[g]), 0});
            std::fill(patches->row_patch.begin() + std::ptrdiff_t(start[g]), patches->row_patch.begin() + std::ptrdiff_t(start[g + 1]), int(g));
            std::fill(patches->col_patch.begin() + std::ptrdiff_t(coarse_start_[g]), patches->col_patch.begin() + std::ptrdiff_t(coarse_start_[g + 1]), int(g));
            for (std::size_t r = start[g]; r < start[g + 1]; ++r) {
                const auto row = p->ptr[equation[r]];
                for (std::size_t k = 0; k < kept[g].size(); ++k) patches->values[value++] = p->val[std::size_t(row) + k];
            }
        }
        // Equations in no aggregate, whose rows of T are zero.
        patches->patches.push_back({int(equation.size()), int(value), int(coarse), 0});
        for (std::size_t i = 0; i < n; ++i)
            if (groups.id[i] < 0) {
                patches->equation.push_back(int(i));
                patches->row_patch.push_back(int(groups.count));
            }
        patches->patches.push_back({int(patches->equation.size()), int(value), int(coarse), 0});
        patches_ = std::move(patches);
        return p;
    }

    // P = (I - M A) P_tent, with M the level's smoother weights.
    [[nodiscard]] std::tuple<std::shared_ptr<Csr>, std::shared_ptr<Csr>>
    transfer_operators(const Csr& a, const std::vector<double>& weights) {
        const auto n = std::ptrdiff_t(a.nrows);
        const Aggregates groups = aggregates(a);
        // Coarsening that would not reduce the system leaves nothing to gain.
        if (groups.count == 0) throw amgcl::error::empty_level();
        const auto initial = tentative(groups);
        if (initial->ncols >= std::size_t(n)) throw amgcl::error::empty_level();

        auto [p, r] = smoothed_transfer(a, weights, *initial);

        // The next level: each aggregate owns one equation per column it kept.
        point_.resize(coarse_start_.back());
        for (std::size_t g = 0; g < groups.count; ++g)
            for (std::size_t i = coarse_start_[g]; i < coarse_start_[g + 1]; ++i) point_[i] = int(g);
        points_ = int(groups.count);
        eps_strong_ *= 0.5F;
        return {std::move(p), std::move(r)};
    }

    // T of the last tentative() in patches.
    [[nodiscard]] const std::shared_ptr<const AmgTentative>& patches() const { return patches_; }

private:
    // The columns aggregate g carries: all of the required ones, and each
    // optional one with a part outside the span of the columns before it. The
    // limit is far above round-off and far below anything a mode can still add.
    [[nodiscard]] std::vector<int> kept_columns(std::size_t g, const std::vector<std::size_t>& start,
                                                const std::vector<int>& equation) const {
        constexpr double kDependent = 1e-8;
        const auto cols = std::size_t(nullspace_.cols);
        const auto required = cols - std::size_t(optional_);
        std::vector<int> kept(cols);
        std::iota(kept.begin(), kept.end(), 0);
        if (optional_ == 0) return kept;
        kept.resize(required);
        const auto first = start[g], d = start[g + 1] - first;
        const auto value = [&](std::size_t r, std::size_t c) {
            return nullspace_.B[std::size_t(equation[first + r]) * cols + c];
        };
        // Modified Gram-Schmidt in column order. Nearly zero parts of the required
        // columns are left out of the basis used for the test.
        std::vector<std::vector<double>> basis;
        std::vector<double> part(d);
        for (std::size_t c = 0; c < cols; ++c) {
            double norm = 0;
            for (std::size_t r = 0; r < d; ++r) {
                part[r] = value(r, c);
                norm += part[r] * part[r];
            }
            for (const auto& q : basis) {
                double along = 0;
                for (std::size_t r = 0; r < d; ++r) along += q[r] * part[r];
                for (std::size_t r = 0; r < d; ++r) part[r] -= along * q[r];
            }
            double rest = 0;
            for (std::size_t r = 0; r < d; ++r) rest += part[r] * part[r];
            const bool independent = norm > 0 && rest > kDependent * kDependent * norm;
            if (independent && kept.size() < d) {
                const double scale = 1 / std::sqrt(rest);
                for (std::size_t r = 0; r < d; ++r) part[r] *= scale;
                basis.push_back(part);
                if (c >= required) kept.push_back(int(c));
            }
        }
        return kept;
    }

    std::vector<int> point_;
    int points_;
    int optional_;
    amgcl::coarsening::nullspace_params nullspace_;
    std::vector<std::size_t> coarse_start_;
    std::shared_ptr<const AmgTentative> patches_;
    float eps_strong_{0.08F};
};

} // namespace vibestran::detail
