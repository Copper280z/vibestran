#pragma once
// AMG PCG for the CPU. It shares its hierarchy with the device backends but not its
// cycle, for two reasons.
//
// A desktop's memory bandwidth is used up by a few threads, so what limits a cycle is
// the bytes it reads, not the arithmetic. Consecutive rows with the same columns share
// one list, and the cycle works in fp32. Conjugate gradients stay in fp64 on the fp64
// matrix, so the accuracy of the solution does not depend on the cycle's.
//
// A CPU can run Gauss-Seidel, which takes fewer iterations than the Jacobi smoothing a
// device runs, for the same bytes: a forward sweep, half a product for the residual
// and a backward sweep are two products' worth, as two Jacobi steps are.
#include "amg_factor.hpp"
#include <array>
#include <atomic>
#include <numeric>
#include <optional>

namespace vibestran::detail {

// Row storage of a sparse matrix's pattern: start[i] is the row's first value and
// list[i] its first column, and the rows of a grid point or an aggregate, having
// the same columns, have the same list. A matrix's values stay in row order, so
// this pattern serves them in any precision.
struct AmgRowLists {
    std::vector<int> start, list, columns;

    explicit AmgRowLists(const AmgHostCsr& a) : AmgRowLists(std::vector<int>(a.ptr, a.ptr + a.nrows + 1), a.col) {}
    AmgRowLists(std::vector<int> ptr, const int* col) : start(std::move(ptr)), list(start.size() - 1) {
        const auto n = std::ptrdiff_t(list.size());
        std::vector<char> repeats(list.size(), 0);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 1; i < n; ++i)
            repeats[i] = std::equal(col + start[i - 1], col + start[i], col + start[i], col + start[i + 1]);
        std::size_t total = 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            if (repeats[i]) { list[i] = list[i - 1]; continue; }
            list[i] = int(total);
            total += std::size_t(start[i + 1] - start[i]);
        }
        columns.resize(total);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < n; ++i)
            if (!repeats[i]) std::copy(col + start[i], col + start[i + 1], columns.begin() + list[i]);
    }

    [[nodiscard]] int length(std::ptrdiff_t i) const { return start[i + 1] - start[i]; }
    [[nodiscard]] const int* row(std::ptrdiff_t i) const { return columns.data() + list[i]; }
};

// A level with fewer nonzeros than this is one block: it gains nothing from threads.
inline constexpr std::size_t kAmgNonzerosPerBlock = 32768;

// Rows split into ranges of about equal nonzeros, fixed by the matrix and not the
// thread count, so that a dense row does not leave one thread with most of the work.
inline std::vector<int> amg_row_blocks(const int* ptr, std::size_t rows, std::size_t limit = 64) {
    const auto total = std::size_t(ptr[rows]);
    const auto count = std::clamp(total / kAmgNonzerosPerBlock, std::size_t(1), limit);
    std::vector<int> blocks(count + 1, 0);
    for (std::size_t k = 1; k < count; ++k)
        blocks[k] = int(std::lower_bound(ptr, ptr + rows + 1, int(total * k / count)) - ptr);
    blocks[count] = int(rows);
    return blocks;
}

template<class F> void amg_each_block(const std::vector<int>& blocks, F&& f) {
    const auto count = std::ptrdiff_t(blocks.size()) - 1;
#pragma omp parallel for schedule(static) if (count > 1)
    for (std::ptrdiff_t k = 0; k < count; ++k) f(blocks[k], blocks[k + 1]);
}

// Summed block by block in a fixed order, so independent of the thread count.
inline double amg_fixed_dot(const std::vector<double>& a, const std::vector<double>& b) {
    constexpr std::ptrdiff_t kBlock = std::ptrdiff_t(1) << 13;
    const auto n = std::ptrdiff_t(a.size());
    std::vector<double> sums(std::size_t((n + kBlock - 1) / kBlock), 0.0);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t k = 0; k < std::ptrdiff_t(sums.size()); ++k) {
        double sum = 0;
        const std::ptrdiff_t last = std::min(n, (k + 1) * kBlock);
        for (std::ptrdiff_t i = k * kBlock; i < last; ++i) sum += a[i] * b[i];
        sums[k] = sum;
    }
    return std::accumulate(sums.begin(), sums.end(), 0.0);
}

template<class Real> inline Real amg_row_sum(const Real* a, const int* c, int count, const Real* x) {
    Real sum = 0;
    for (int j = 0; j < count; ++j) sum += a[j] * x[c[j]];
    return sum;
}

// The order of a Gauss-Seidel sweep that threads can share and that is still exact.
// Rows split into blocks. A block's rows that couple to no remaining row of another
// block are swept at the same time as the other blocks' such rows. If a few colors of
// blocks then separate what remains, each color is a stage; otherwise what remains is
// split again into half as many blocks, which could at best run a quarter of these
// at once. A sweep is a few stages run in turn, and it is one exact Gauss-Seidel sweep
// in the resulting order. Fixed by the matrix, so independent of the thread count.
struct AmgSweepOrder {
    std::vector<int> order, position;          // new to old, old to new
    std::vector<std::vector<int>> stages;      // block boundaries, new numbering

    explicit AmgSweepOrder(const AmgHostCsr& a) {
        constexpr std::size_t kBlocks = 16;
        order.reserve(a.nrows);
        position.resize(a.nrows);
        std::vector<int> remaining(a.nrows), tag(a.nrows, -1);
        std::iota(remaining.begin(), remaining.end(), 0);
        for (std::size_t count = a.nnz < kAmgNonzerosPerBlock ? 1 : kBlocks; !remaining.empty(); count = std::max<std::size_t>(count / 2, 1)) {
            const std::size_t total = std::accumulate(remaining.begin(), remaining.end(), std::size_t(0), [&](std::size_t sum, int i) {
                return sum + std::size_t(a.ptr[i + 1] - a.ptr[i]);
            });
            std::size_t seen = 0;
            for (const int i : remaining) {
                tag[i] = int(std::min(count - 1, seen * count / std::max<std::size_t>(total, 1)));
                seen += std::size_t(a.ptr[i + 1] - a.ptr[i]);
            }
            std::vector<char> inside(remaining.size(), 1);
            if (count > 1) {
#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t r = 0; r < std::ptrdiff_t(remaining.size()); ++r) {
                    const int i = remaining[r];
                    for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j)
                        if (tag[a.col[j]] >= 0 && tag[a.col[j]] != tag[i]) { inside[r] = 0; break; }
                }
            }
            std::vector<std::vector<int>> taken(count);
            std::vector<int> rest;
            for (std::size_t r = 0; r < remaining.size(); ++r) {
                if (inside[r]) { taken[std::size_t(tag[remaining[r]])].push_back(remaining[r]); tag[remaining[r]] = -1; }
                else rest.push_back(remaining[r]);
            }
            add_stage(taken);
            if (const auto colors = few_colors(a, rest, tag, count)) {
                for (int color = 0; color <= *std::ranges::max_element(*colors); ++color) {
                    for (auto& block : taken) block.clear();
                    for (const int i : rest)
                        if ((*colors)[std::size_t(tag[i])] == color) taken[std::size_t(tag[i])].push_back(i);
                    add_stage(taken);
                }
                rest.clear();
            }
            for (const int i : rest) tag[i] = -1;
            remaining = std::move(rest);
        }
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t k = 0; k < std::ptrdiff_t(a.nrows); ++k) position[order[k]] = int(k);
    }

private:
    void add_stage(const std::vector<std::vector<int>>& blocks) {
        std::vector<int> boundaries{int(order.size())};
        for (const auto& block : blocks) {
            if (block.empty()) continue;
            order.insert(order.end(), block.begin(), block.end());
            boundaries.push_back(int(order.size()));
        }
        if (boundaries.size() > 1) stages.push_back(std::move(boundaries));
    }

    // The color of each block, when at most four keep blocks that couple through the
    // given rows apart.
    static std::optional<std::vector<int>> few_colors(const AmgHostCsr& a, const std::vector<int>& rows, const std::vector<int>& tag, std::size_t count) {
        constexpr int kMostColors = 4;
        if (rows.empty()) return std::nullopt;
        std::vector<char> coupled(count * count, 0);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t r = 0; r < std::ptrdiff_t(rows.size()); ++r) {
            const int i = rows[r];
            for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
                const int other = tag[a.col[j]];
                if (other >= 0 && other != tag[i]) std::atomic_ref<char>(coupled[std::size_t(tag[i]) * count + std::size_t(other)]).store(1, std::memory_order_relaxed);
            }
        }
        std::vector<int> colors(count, 0);
        for (std::size_t b = 0; b < count; ++b) {
            std::array<bool, kMostColors> used{};
            for (std::size_t o = 0; o < b; ++o)
                if (coupled[b * count + o]) used[std::size_t(colors[o])] = true;
            const auto free = std::ranges::find(used, false);
            if (free == used.end()) return std::nullopt;
            colors[b] = int(free - used.begin());
        }
        return colors;
    }
};

// z = B r: one V-cycle.
class AmgCpuCycle {
public:
    virtual ~AmgCpuCycle() = default;
    virtual void apply(const std::vector<double>& r, std::vector<double>& z) = 0;
};

// The cycle of amg_vcycle in Real, with a forward sweep before the coarse correction
// and a backward one after, in the order of AmgSweepOrder. That smoother is M = D + L
// in that order, so M + M' - A = D is positive definite and on any SPD matrix the
// cycle is SPD with the eigenvalues of B A in (0, 1], with no weight to choose.
template<class Real> class AmgCpuCycleOf final : public AmgCpuCycle {
public:
    explicit AmgCpuCycleOf(const AmgHierarchy& h) : h_(h), levels_(h.levels.size()) {
        std::vector<std::optional<AmgSweepOrder>> orders(h.levels.size());
        for (std::size_t l = 0; l < h.levels.size(); ++l)
            if (!h.levels[l].smoother.empty()) orders[l].emplace(*h.levels[l].A);
        for (std::size_t l = 0; l < h.levels.size(); ++l) {
            const auto& from = h.levels[l];
            auto& to = levels_[l];
            const AmgSweepOrder* order = orders[l] ? &*orders[l] : nullptr;
            const AmgSweepOrder* next = l + 1 < orders.size() && orders[l + 1] ? &*orders[l + 1] : nullptr;
            to.f.resize(from.A->nrows);
            to.y.resize(from.A->nrows);
            if (from.P) {
                to.p.emplace(*from.P, order, next);
                to.r.emplace(*from.R, next, order);
            }
            if (order) to.sweep.emplace(*from.A, *order);
            to.x.resize(order ? from.A->nrows : 0);
        }
        if (orders.front()) fine_order_ = std::move(orders.front()->order);
    }

    void apply(const std::vector<double>& r, std::vector<double>& z) override {
        auto& fine = levels_.front();
        const auto n = std::ptrdiff_t(r.size());
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < n; ++i) fine.f[i] = Real(r[fine_order_.empty() ? i : fine_order_[i]]);
        amg_vcycle(*this, levels_.size());
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < n; ++i) z[fine_order_.empty() ? i : fine_order_[i]] = double(fine.y[i]);
    }

    void smooth_zero(std::size_t l) {
        auto& v = levels_[l];
        const Sweep& s = *v.sweep;
        for (const auto& stage : s.stages)
            amg_each_block(stage, [&](int from, int to) {
                for (int i = from; i < to; ++i)
                    v.x[i] = s.m[i] * (v.f[i] - amg_row_sum(s.lower.data() + s.lower_start[i], s.lists.row(i), s.lower_length(i), v.x.data()));
            });
    }
    // The forward sweep's equations hold, which leaves the entries after the diagonal.
    void residual(std::size_t l) {
        auto& v = levels_[l];
        const Sweep& s = *v.sweep;
        amg_each_block(s.blocks, [&](int from, int to) {
            for (int i = from; i < to; ++i)
                v.y[i] = -amg_row_sum(s.upper.data() + s.upper_start[i], s.lists.row(i) + s.lower_length(i) + 1, s.upper_length(i), v.x.data());
        });
    }
    void restriction(std::size_t l) {
        auto& v = levels_[l];
        auto& next = levels_[l + 1];
        const Transfer& r = *v.r;
        amg_each_block(r.blocks, [&](int from, int to) {
            for (int i = from; i < to; ++i)
                next.f[i] = amg_row_sum(r.values.data() + r.lists.start[i], r.lists.row(i), r.lists.length(i), v.y.data());
        });
    }
    void coarse(std::size_t l) {
        auto& v = levels_[l];
        if (!h_.coarse_direct()) { smooth_zero(l); smooth_post(l); return; }
        // In fp64: when the coarsest level is the whole hierarchy, this is the cycle.
        const auto n = std::ptrdiff_t(v.f.size());
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            double sum = 0;
            for (std::ptrdiff_t j = 0; j < n; ++j) sum += h_.coarse_inverse[i * n + j] * double(v.f[j]);
            v.y[i] = Real(sum);
        }
    }
    void prolongation(std::size_t l) {
        auto& v = levels_[l];
        const auto& next = levels_[l + 1];
        const Transfer& p = *v.p;
        amg_each_block(p.blocks, [&](int from, int to) {
            for (int i = from; i < to; ++i)
                v.x[i] += amg_row_sum(p.values.data() + p.lists.start[i], p.lists.row(i), p.lists.length(i), next.y.data());
        });
    }
    // y = x + M'^-1 (f - A x) is one backward sweep, whose diagonal term cancels x.
    void smooth_post(std::size_t l) {
        auto& v = levels_[l];
        const Sweep& s = *v.sweep;
        for (auto stage = s.stages.rbegin(); stage != s.stages.rend(); ++stage)
            amg_each_block(*stage, [&](int from, int to) {
                for (int i = to - 1; i >= from; --i) {
                    const int* c = s.lists.row(i);
                    const int d = s.lower_length(i);
                    v.y[i] = s.m[i] * (v.f[i] - amg_row_sum(s.lower.data() + s.lower_start[i], c, d, v.x.data()) -
                                       amg_row_sum(s.upper.data() + s.upper_start[i], c + d + 1, s.upper_length(i), v.y.data()));
                }
            });
    }

private:
    // The rows of a matrix in the order `rows`, its columns renumbered by `columns`.
    struct Renumbered {
        std::vector<int> ptr, col;
        std::vector<Real> values;

        Renumbered(const AmgHostCsr& a, const AmgSweepOrder* rows, const AmgSweepOrder* columns, bool sorted)
            : ptr(a.nrows + 1, 0), col(a.nnz), values(a.nnz) {
            const auto n = std::ptrdiff_t(a.nrows);
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                const int from = rows ? rows->order[i] : int(i);
                ptr[i + 1] = ptr[i] + a.ptr[from + 1] - a.ptr[from];
            }
#pragma omp parallel
            {
                std::vector<std::pair<int, double>> row;
#pragma omp for schedule(dynamic, 1024)
                for (std::ptrdiff_t i = 0; i < n; ++i) {
                    const int from = rows ? rows->order[i] : int(i);
                    row.clear();
                    for (int j = a.ptr[from]; j < a.ptr[from + 1]; ++j)
                        row.emplace_back(columns ? columns->position[a.col[j]] : a.col[j], a.val[j]);
                    // Nearly sorted already: an order moves few rows far.
                    if (sorted) {
                        if (row.size() > 256) std::ranges::sort(row, {}, &std::pair<int, double>::first);
                        else for (std::size_t k = 1; k < row.size(); ++k)
                            for (std::size_t m = k; m > 0 && row[m - 1].first > row[m].first; --m) std::swap(row[m - 1], row[m]);
                    }
                    for (std::size_t k = 0; k < row.size(); ++k) {
                        col[std::size_t(ptr[i]) + k] = row[k].first;
                        values[std::size_t(ptr[i]) + k] = Real(row[k].second);
                    }
                }
            }
        }
    };

    struct Transfer {
        std::vector<Real> values;
        AmgRowLists lists;
        std::vector<int> blocks;

        Transfer(const AmgHostCsr& a, const AmgSweepOrder* rows, const AmgSweepOrder* columns) : Transfer(Renumbered(a, rows, columns, false)) {}
        explicit Transfer(Renumbered m)
            : values(std::move(m.values)), lists(std::move(m.ptr), m.col.data()), blocks(amg_row_blocks(lists.start.data(), lists.list.size())) {}
    };

    // A level's operator apart from its diagonal, with the entries before and after
    // the diagonal in separate arrays: a sweep or a residual then reads one of them
    // in sequence, where in one array it would skip over the other's.
    struct Sweep {
        AmgRowLists lists;
        std::vector<int> blocks;
        std::vector<std::vector<int>> stages;
        std::vector<int> diagonal, lower_start, upper_start;
        std::vector<Real> lower, upper, m;

        [[nodiscard]] int lower_length(int i) const { return diagonal[i]; }
        [[nodiscard]] int upper_length(int i) const { return lists.length(i) - diagonal[i] - 1; }

        Sweep(const AmgHostCsr& a, const AmgSweepOrder& order) : Sweep(Renumbered(a, &order, &order, true), order) {}
        Sweep(Renumbered s, const AmgSweepOrder& order)
            : lists(std::move(s.ptr), s.col.data()), blocks(amg_row_blocks(lists.start.data(), lists.list.size())), stages(order.stages) {
            const auto n = std::ptrdiff_t(lists.list.size());
            diagonal.resize(lists.list.size());
            std::atomic<int> missing{-1};
            amg_each_block(blocks, [&](int from, int to) {
                for (int i = from; i < to; ++i) {
                    const int* c = lists.row(i);
                    diagonal[i] = int(std::lower_bound(c, c + lists.length(i), i) - c);
                    if (diagonal[i] == lists.length(i) || c[diagonal[i]] != i) missing.store(i, std::memory_order_relaxed);
                }
            });
            if (const int row = missing.load(); row >= 0)
                throw SolverError(std::format("AMG CPU cycle: row {} of a level of {} rows has no diagonal", row, n));
            lower_start.assign(lists.list.size() + 1, 0);
            upper_start.assign(lists.list.size() + 1, 0);
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                lower_start[i + 1] = lower_start[i] + lower_length(int(i));
                upper_start[i + 1] = upper_start[i] + upper_length(int(i));
            }
            lower.resize(std::size_t(lower_start[n]));
            upper.resize(std::size_t(upper_start[n]));
            m.resize(lists.list.size());
            amg_each_block(blocks, [&](int from, int to) {
                for (int i = from; i < to; ++i) {
                    const Real* row = s.values.data() + lists.start[i];
                    std::copy(row, row + lower_length(i), lower.begin() + lower_start[i]);
                    std::copy(row + lower_length(i) + 1, row + lists.length(i), upper.begin() + upper_start[i]);
                    // A floating part's modes reach a coarse level with no stiffness.
                    const Real inverse = Real(1) / row[lower_length(i)];
                    m[i] = row[lower_length(i)] > 0 && std::isfinite(inverse) ? inverse : Real(0);
                }
            });
        }
    };

    struct Level {
        std::optional<Sweep> sweep;
        std::optional<Transfer> p, r;
        std::vector<Real> f, x, y;
    };

    const AmgHierarchy& h_;
    std::vector<int> fine_order_;
    std::vector<Level> levels_;
};

// Conjugate gradients in fp64 on the equilibrated fine matrix. Its product reads the
// matrix through row lists too, without the column indices of its CSR.
class AmgCpuEngine final : public AmgEngine {
public:
    AmgCpuEngine(const ScaledAmgMatrix& scaled, const AmgHierarchy& h, int max_iterations)
        : scaled_(scaled), h_(h), max_iterations_(max_iterations) { values_changed(); }

    std::pair<int, double> solve(const std::vector<double>& b, double tolerance, std::vector<double>& x) override {
        const auto n = std::ptrdiff_t(b.size());
        std::vector<double> r = b, p(b.size()), q(b.size()), z(b.size());
        std::fill(x.begin(), x.end(), 0.0);
        const double norm_b = std::sqrt(amg_fixed_dot(b, b));
        double res = norm_b, rho = 0;
        int it = 0;
        for (; it < max_iterations_ && res > tolerance * norm_b; ++it) {
            cycle_->apply(r, z);
            const double rho_new = amg_fixed_dot(r, z);
            const double beta = it ? rho_new / rho : 0.0;
            rho = rho_new;
#pragma omp parallel for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
            product(p, q);
            const double pq = amg_fixed_dot(p, q);
            if (!(pq > 0)) break;
            const double alpha = rho / pq;
#pragma omp parallel for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
            res = std::sqrt(amg_fixed_dot(r, r));
        }
        return {it, norm_b > 0 ? res / norm_b : 0.0};
    }

    void values_changed() override {
        const auto start = std::chrono::steady_clock::now();
        const auto& a = *scaled_.matrix;
        lists_ = std::make_unique<const AmgRowLists>(a);
        blocks_ = amg_row_blocks(a.ptr, a.nrows);
        // Alone, the coarsest inverse is the whole cycle and has to be fp64.
        if (h_.levels.size() > 1) cycle_ = std::make_unique<AmgCpuCycleOf<float>>(h_);
        else cycle_ = std::make_unique<AmgCpuCycleOf<double>>(h_);
        log_debug(std::format("[amg-cpu] cycle prepared in {:.3f} ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()));
    }

private:
    void product(const std::vector<double>& x, std::vector<double>& y) const {
        const auto& lists = *lists_;
        const double* values = scaled_.matrix->val;
        amg_each_block(blocks_, [&](int from, int to) {
            for (int i = from; i < to; ++i)
                y[i] = amg_row_sum(values + lists.start[i], lists.row(i), lists.length(i), x.data());
        });
    }

    const ScaledAmgMatrix& scaled_;
    const AmgHierarchy& h_;
    int max_iterations_;
    std::unique_ptr<const AmgRowLists> lists_;
    std::vector<int> blocks_;
    std::unique_ptr<AmgCpuCycle> cycle_;
};

inline constexpr int kAmgCpuIterationLimit = 10000;

inline AmgSystem::EngineFactory amg_cpu_engine(int max_iterations) {
    const int limit = max_iterations > 0 ? max_iterations : kAmgCpuIterationLimit;
    return [limit](const ScaledAmgMatrix& scaled, const AmgHierarchy& h) { return std::make_unique<AmgCpuEngine>(scaled, h, limit); };
}
} // namespace vibestran::detail
