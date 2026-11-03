// tests/unit/test_cuda_pcg_amg.cpp
// The CUDA PCG backends' AMG path (n >= 50000): the device hierarchy fully
// resident, with a CPU suffix of fine rows, and the host-hierarchy fallback,
// through solve() and through factor() with several right-hand sides and a
// same-pattern refactor().

#include <gtest/gtest.h>
#include "solver/solver_backend.hpp"
#include "pcg_partition_problem.hpp"
#ifdef HAVE_CUDA
#include "solver/cuda_mixed_pcg_solver_backend.hpp"
#include "solver/cuda_pcg_solver_backend.hpp"
#include <memory>
#include <optional>
#include <variant>
#endif

using namespace vibestran;

#ifndef HAVE_CUDA
TEST(CudaPcgAmg, CudaNotCompiled) {
    GTEST_SKIP() << "CUDA backend not compiled — skipping CUDA PCG AMG tests";
}
#else

namespace {

// 0: resident; 13 MiB: part of the fine rows on the CPU; 1 MiB: below the
// fixed device state, so the hierarchy stays on the host.
constexpr std::size_t kBudgets[] = {0, 13U * 1024U * 1024U, 1024U * 1024U};

template <class Backend> std::optional<Backend> create(std::size_t budget) {
    return Backend::try_create(0.0, 0, budget);
}

// solve() and factor() of one backend on the material-interface network and
// its stiffened variant. The fine operator is shared by CG and the cycle,
// so a CPU-row product that lost a coupling into device rows, or a refactor
// that left the old values (or stale pointers) in a captured graph, misses
// the exact solution by O(1) at the interface.
template <class Backend> void check_amg(const char* label) {
    const auto problem = test::pcg_partition_problem(50003);
    const auto stiffer = test::stiffened(problem, 30.0);
    const auto n = std::size_t(problem.matrix.n);
    for (const std::size_t budget : kBudgets) {
        auto backend = create<Backend>(budget);
        if (!backend) GTEST_SKIP() << "CUDA not available on this system";
        SCOPED_TRACE(std::string(label) + " budget=" + std::to_string(budget));

        const auto x = backend->solve(problem.matrix, problem.rhs);
        ASSERT_TRUE(backend->last_refinement() && backend->last_refinement()->converged);
        for (std::size_t i = 0; i < n; ++i) ASSERT_NEAR(x[i], problem.exact[i], 2e-8) << i;
        const auto zero = backend->solve(problem.matrix, std::vector<double>(n, 0.0));
        for (double v : zero) ASSERT_EQ(v, 0.0);

        auto made = backend->factor(problem.matrix);
        ASSERT_TRUE(std::holds_alternative<std::unique_ptr<LinearFactor>>(made)) << std::get<Error>(made).message;
        auto factor = std::get<std::unique_ptr<LinearFactor>>(std::move(made));
        EXPECT_EQ(factor->method(), "amg-pcg");
        std::vector<double> B(2 * n);
        for (std::size_t i = 0; i < n; ++i) { B[i] = problem.rhs[i]; B[n + i] = -3 * problem.rhs[i]; }
        auto X = factor->solve(B, 2);
        ASSERT_TRUE(std::holds_alternative<std::vector<double>>(X)) << std::get<Error>(X).message;
        const auto& columns = std::get<std::vector<double>>(X);
        for (std::size_t i = 0; i < n; ++i) {
            ASSERT_NEAR(columns[i], problem.exact[i], 2e-8) << i;
            ASSERT_NEAR(columns[n + i], -3 * problem.exact[i], 6e-8) << i;
        }
        ASSERT_FALSE(factor->refactor(stiffer.matrix).has_value());
        X = factor->solve(stiffer.rhs, 1);
        ASSERT_TRUE(std::holds_alternative<std::vector<double>>(X)) << std::get<Error>(X).message;
        const auto& y = std::get<std::vector<double>>(X);
        for (std::size_t i = 0; i < n; ++i) ASSERT_NEAR(y[i], stiffer.exact[i], 2e-8) << i;
        EXPECT_EQ(factor->stats().symbolic_analyses, 1);
        EXPECT_EQ(factor->stats().numeric_factorizations, 2);
        EXPECT_EQ(factor->stats().solves, 3);
    }
}

} // namespace

TEST(CudaPcgAmg, Fp64HierarchyResidentSpilledAndOnTheHost) {
    check_amg<CudaPCGSolverBackend>("cuda-pcg");
}

TEST(CudaPcgAmg, MixedHierarchyResidentSpilledAndOnTheHost) {
    check_amg<CudaMixedPCGSolverBackend>("cuda-pcg-mixed");
}

#endif
