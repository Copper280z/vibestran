// tests/unit/test_amg_cpu.cpp
// The CPU's AMG cycle: Gauss-Seidel in an order threads can share, in fp32, on
// compact storage. In fp64 the cycle has to be a valid preconditioner with the same
// spectrum bound as the other backends' (test_amg_vcycle.cpp holds the model-by-frame
// check); in fp32 it differs by rounding, and conjugate gradients in fp64 on the fp64
// matrix still reach the accuracy of a direct solve.

#include <gtest/gtest.h>
#include "amg_cpu.hpp"
#include "structural_system.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace vibestran;
using vibestran::test::assemble;

namespace {

std::vector<std::string> decks() {
    return {test::hex_bar(4), test::shell_sheet(16), test::mixed_assembly(6)};
}

std::vector<double> random_vector(std::size_t n, unsigned seed) {
    std::mt19937_64 random(seed);
    std::normal_distribution<double> normal;
    std::vector<double> v(n);
    std::ranges::generate(v, [&] { return normal(random); });
    return v;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double relative_difference(const std::vector<double>& a, const std::vector<double>& b) {
    double top = 0, size = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        top = std::max(top, std::abs(a[i] - b[i]));
        size = std::max(size, std::abs(b[i]));
    }
    return top / size;
}

template<class Real>
std::vector<double> cycle_of(const detail::AmgHierarchy& h, const std::vector<double>& r) {
    detail::AmgCpuCycleOf<Real> cycle(h);
    std::vector<double> z(r.size());
    cycle.apply(r, z);
    return z;
}

} // namespace

TEST(AmgRowLists, ReadsTheMatrixItWasBuiltFromAndSharesColumnsWithinGridPoints) {
    const auto r = assemble(test::shell_sheet(16), false);
    const detail::ScaledAmgMatrix scaled(r.k);
    const auto& a = *scaled.matrix;
    const detail::AmgRowLists lists(a);
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(a.nrows); ++i) {
        ASSERT_EQ(lists.length(i), a.ptr[i + 1] - a.ptr[i]);
        ASSERT_EQ(lists.start[i], a.ptr[i]);
        ASSERT_TRUE(std::equal(a.col + a.ptr[i], a.col + a.ptr[i + 1], lists.row(i))) << "row " << i;
    }
    EXPECT_LT(lists.columns.size(), a.nnz) << "no rows shared a list";
}

TEST(AmgRowLists, RowsOfDifferentColumnsNeverShareAList) {
    // Rows of equal length but different columns are not repeats.
    detail::AmgHostCsr a;
    a.set_size(3, 3, false);
    a.set_nonzeros(6);
    const int ptr[] = {0, 2, 4, 6};
    const int col[] = {0, 1, 0, 1, 1, 2};
    std::copy(std::begin(ptr), std::end(ptr), a.ptr);
    std::copy(std::begin(col), std::end(col), a.col);
    const detail::AmgRowLists lists(a);
    EXPECT_EQ(lists.list[0], lists.list[1]);
    EXPECT_NE(lists.list[1], lists.list[2]);
    EXPECT_EQ(lists.columns.size(), 4u);
    EXPECT_EQ(lists.row(2)[0], 1);
    EXPECT_EQ(lists.row(2)[1], 2);
}

TEST(AmgRowBlocks, CoverEveryRowInOrderAndBalanceADenseRow) {
    // One row holds half the nonzeros, so a block for it has little else.
    constexpr int kOthers = 99;
    constexpr int kNonzerosEach = 8 * int(detail::kAmgNonzerosPerBlock) / kOthers;
    std::vector<int> ptr{0};
    for (int i = 0; i < kOthers + 1; ++i)
        ptr.push_back(ptr.back() + (i == 50 ? kOthers * kNonzerosEach : kNonzerosEach));
    const auto blocks = detail::amg_row_blocks(ptr.data(), std::size_t(kOthers + 1), 8);
    ASSERT_EQ(blocks.size(), 9u);
    EXPECT_EQ(blocks.front(), 0);
    EXPECT_EQ(blocks.back(), kOthers + 1);
    EXPECT_TRUE(std::is_sorted(blocks.begin(), blocks.end()));
    for (std::size_t k = 0; k + 1 < blocks.size(); ++k)
        EXPECT_LE(ptr[std::size_t(blocks[k + 1])] - ptr[std::size_t(blocks[k])], kOthers * kNonzerosEach + 2 * ptr.back() / 8)
            << "block " << k;
    // Too few nonzeros to gain from threads.
    const std::vector<int> small{0, 1, 2, 3};
    EXPECT_EQ(detail::amg_row_blocks(small.data(), 3).size(), 2u);
}

// What makes a sweep exact on threads: no two rows being updated at the same time
// are coupled, in every stage of every order, whatever the matrix.
TEST(AmgSweepOrder, IsAPermutationWithNoCouplingBetweenTheBlocksOfAStage) {
    const auto check = [](const detail::AmgHostCsr& a) -> void {
        const detail::AmgSweepOrder order(a);
        ASSERT_EQ(order.order.size(), a.nrows);
        std::vector<int> seen(a.nrows, 0);
        for (std::size_t k = 0; k < a.nrows; ++k) {
            ++seen[std::size_t(order.order[k])];
            ASSERT_EQ(order.position[std::size_t(order.order[k])], int(k));
        }
        ASSERT_TRUE(std::ranges::all_of(seen, [](int count) { return count == 1; }));
        std::vector<int> stage_of(a.nrows), block_of(a.nrows);
        int next = 0;
        for (std::size_t s = 0; s < order.stages.size(); ++s) {
            const auto& stage = order.stages[s];
            ASSERT_EQ(stage.front(), next);
            next = stage.back();
            for (std::size_t b = 0; b + 1 < stage.size(); ++b)
                for (int k = stage[b]; k < stage[b + 1]; ++k) {
                    stage_of[std::size_t(k)] = int(s);
                    block_of[std::size_t(k)] = int(b);
                }
        }
        ASSERT_EQ(next, int(a.nrows));
        for (std::size_t i = 0; i < a.nrows; ++i) {
            const auto new_i = std::size_t(order.position[i]);
            for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
                const auto new_j = std::size_t(order.position[std::size_t(a.col[j])]);
                if (stage_of[new_i] == stage_of[new_j])
                    ASSERT_EQ(block_of[new_i], block_of[new_j]) << "rows " << i << " and " << a.col[j];
            }
        }
    };
    for (const auto& bulk : decks()) {
        const auto r = assemble(bulk, false);
        const detail::ScaledAmgMatrix scaled(r.k);
        const detail::AmgHierarchy h(scaled.matrix, r.modes.get(), scaled.scale);
        for (const auto& level : h.levels) check(*level.A);
    }
    // Large enough to split, on a regular mesh: most of a sweep runs on many threads.
    const auto r = assemble(test::hex_bar(10), false);
    const detail::ScaledAmgMatrix scaled(r.k);
    const auto stages = [&] { return detail::AmgSweepOrder(*scaled.matrix).stages; }();
    EXPECT_GE(stages.front().size() - 1, 4u) << "first stage is not parallel";
    EXPECT_LE(stages.size(), 6u);
}

TEST(AmgFixedDot, IsTheInnerProductToRounding) {
    for (const std::size_t n : {std::size_t(1), std::size_t(100), std::size_t(8192), std::size_t(8193), std::size_t(50000)}) {
        const auto a = random_vector(n, 1), b = random_vector(n, 2);
        EXPECT_NEAR(detail::amg_fixed_dot(a, b), dot(a, b), 1e-10 * double(n)) << "n=" << n;
    }
    EXPECT_EQ(detail::amg_fixed_dot({}, {}), 0.0);
}

// Rounding the cycle's values leaves a nearly symmetric, positive definite operator
// that differs from the exact one by much less than it would take to slow conjugate
// gradients, including on a smoothed coarsest level.
TEST(AmgCpuCycle, InFp32DiffersFromFp64ByRoundingOnly) {
    for (const auto& bulk : decks()) {
        const auto r = assemble(bulk, false);
        const detail::ScaledAmgMatrix scaled(r.k);
        for (const unsigned coarse_enough : {40u, 1u}) {
            SCOPED_TRACE(std::format("n={} coarse_enough={}", r.k.n, coarse_enough));
            const detail::AmgHierarchy h(scaled.matrix, r.modes.get(), scaled.scale, coarse_enough);
            ASSERT_GE(h.levels.size(), 2u);
            const auto x = random_vector(r.f.size(), 3), y = random_vector(r.f.size(), 4);
            const auto bx = cycle_of<float>(h, x), by = cycle_of<float>(h, y);
            EXPECT_LE(std::abs(dot(y, bx) - dot(x, by)) / std::sqrt(dot(x, bx) * dot(y, by)), 1e-3);
            EXPECT_GT(dot(x, bx), 0.0);
            EXPECT_GT(dot(y, by), 0.0);
            const auto exact = cycle_of<double>(h, x);
            std::vector<double> error(x.size());
            std::ranges::transform(bx, exact, error.begin(), std::minus<>{});
            EXPECT_LE(std::sqrt(dot(error, error) / dot(exact, exact)), 1e-2);
        }
    }
}

// fp64 refinement against the fp64 matrix makes the accuracy independent of the
// cycle's precision, in either storage of K.
TEST(AmgCpuSolve, ReachesTheDirectSolutionFromFp32CyclesInEitherStorage) {
    for (const auto& bulk : decks()) {
        const auto r = assemble(bulk);
        for (const auto& k : {r.k, r.k.expanded_symmetric()}) {
            SCOPED_TRACE(std::format("n={} lower={}", k.n, k.stores_lower_triangle_only()));
            detail::AmgSystem system(k, 1e-8, detail::amg_cpu_engine(5000), r.modes);
            const auto result = system.solve(k, r.f);
            EXPECT_TRUE(result.refinement.converged);
            EXPECT_GT(result.iterations, 0);
            EXPECT_LE(relative_difference(result.solution, r.u), 1e-7);
        }
    }
}

// Gauss-Seidel is what pays for the sweeps. Rounding in the cycle must not give that
// back: fp32 stays close to the same cycle in fp64, and both take fewer iterations
// than the other backends' Jacobi smoothing on structures.
TEST(AmgCpuSolve, TakesFewerIterationsThanTheOtherBackendsCycle) {
    for (const auto& bulk : decks()) {
        const auto r = assemble(bulk, false);
        detail::AmgSystem system(r.k, 1e-8, detail::amg_cpu_engine(5000), r.modes);
        const int fp32 = system.solve(r.k, r.f).iterations;
        const int jacobi = detail::solve_hybrid_amg(r.k, r.f, 1e-8, 5000, {}, r.modes).iterations;
        EXPECT_LT(fp32, jacobi) << "n=" << r.k.n;
    }
}

TEST(AmgCpuSolve, RefactorSolvesTheNewValuesOnTheOldPattern) {
    const auto r = assemble(test::hex_bar(4));
    const auto changed = test::with_stiff_bands(r.k, r.u);
    detail::AmgSystem system(r.k, 1e-8, detail::amg_cpu_engine(5000), r.modes);
    ASSERT_TRUE(system.solve(r.k, r.f).refinement.converged);
    ASSERT_TRUE(system.refactor(changed.k));
    const auto result = system.solve(changed.k, changed.f);
    EXPECT_TRUE(result.refinement.converged);
    EXPECT_LE(relative_difference(result.solution, r.u), 1e-7);
}

// Blocks fixed by the matrix and sums in a fixed order: thread count is not an input.
TEST(AmgCpuSolve, IsBitwiseIndependentOfTheThreadCount) {
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
    const auto r = assemble(test::mixed_assembly(6), false);
    const auto solve_with = [&](int count) {
        omp_set_num_threads(count);
        detail::AmgSystem system(r.k, 1e-8, detail::amg_cpu_engine(5000), r.modes);
        return system.solve(r.k, r.f);
    };
    const auto one = solve_with(1);
    const auto many = solve_with(4);
    omp_set_num_threads(threads);
    EXPECT_EQ(one.iterations, many.iterations);
    EXPECT_TRUE(one.solution == many.solution);
#else
    GTEST_SKIP() << "built without OpenMP";
#endif
}
