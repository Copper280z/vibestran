// tests/unit/test_cuda_pcg_amg.cpp
// The CUDA PCG backends' AMG path (n >= 50000): the device hierarchy fully
// resident, with a CPU suffix of fine rows, and the host-hierarchy fallback,
// through solve() and through factor() with several right-hand sides and a
// same-pattern refactor().

#include <gtest/gtest.h>
#include "solver/solver_backend.hpp"
#include "pcg_partition_problem.hpp"
#include "structural_system.hpp"
#ifdef HAVE_CUDA
#include "solver/cuda_mixed_pcg_solver_backend.hpp"
#include "solver/cuda_pcg_solver_backend.hpp"
#include <algorithm>
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

void* cuda_test_hold_all_but(std::size_t leave);
void cuda_test_release(void* held);

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

// A structure with rigid-body modes on every engine: resident, with fine rows
// on the CPU, and with the host hierarchy. The modes must reach the hierarchy
// (without them this bar takes tens of thousands of iterations), every engine
// must converge to the direct solution, and the budget changes where the
// products run but not the method.
template <class Backend> void check_structure(const char* label) {
    const auto r = test::assemble(test::hex_bar(17));
    ASSERT_GE(r.k.n, 50000);
    ASSERT_TRUE(r.modes);
    const double top = test::largest_magnitude(r.u);
    // The host cycle is the one proved SPD with its spectrum in (0, 1] by
    // test_amg_vcycle.cpp. The device cycles apply the same operator, so CG sees
    // the same iterations, up to rounding (and fp32 rounding in the mixed hierarchy).
    const int host_iterations = test::host_amg_iterations(r);
    for (const std::size_t budget : {std::size_t(0), std::size_t(120) * 1024 * 1024, std::size_t(1024) * 1024}) {
        auto backend = create<Backend>(budget);
        if (!backend) GTEST_SKIP() << "CUDA not available on this system";
        SCOPED_TRACE(std::string(label) + " budget=" + std::to_string(budget));
        ASSERT_TRUE(backend->uses_near_nullspace(r.k.n));
        backend->set_near_nullspace(r.modes);
        const auto x = backend->solve(r.k, r.f);
        ASSERT_TRUE(backend->last_refinement() && backend->last_refinement()->converged);
        EXPECT_LE(backend->last_iteration_count(), 120);
        EXPECT_NEAR(double(backend->last_iteration_count()), double(host_iterations), 0.15 * host_iterations + 2);
        for (std::size_t i = 0; i < x.size(); ++i) ASSERT_NEAR(x[i], r.u[i], 1e-6 * top) << i;
        // A factor keeps the same hierarchy across solves and a refactor; 2K solves to half.
        auto made = backend->factor(r.k);
        ASSERT_TRUE(std::holds_alternative<std::unique_ptr<LinearFactor>>(made));
        auto factor = std::get<std::unique_ptr<LinearFactor>>(std::move(made));
        for (const double scale : {1.0, 2.0}) {
            SCOPED_TRACE(scale);
            if (scale != 1.0) {
                auto doubled = r.k;
                std::transform(doubled.values.begin(), doubled.values.end(), doubled.values.begin(),
                               [scale](double value) { return value * scale; });
                ASSERT_FALSE(factor->refactor(doubled));
            }
            auto solved = factor->solve(r.f);
            ASSERT_TRUE(std::holds_alternative<std::vector<double>>(solved));
            const auto& u = std::get<std::vector<double>>(solved);
            EXPECT_NEAR(double(backend->last_iteration_count()), double(host_iterations), 0.15 * host_iterations + 2);
            for (std::size_t i = 0; i < u.size(); ++i) ASSERT_NEAR(u[i], r.u[i] / scale, 1e-6 * top) << i;
        }
    }
}

// Every hierarchy gets a plan from its exact storage, whatever the size of the
// deck and the memory free: this is what keeps a hierarchy larger than any
// estimate from the matrix alone off the device, and off an allocation failure.
template <class Backend> void check_plan(const char* label) {
    using Execution = AmgDeviceReport::Execution;
    const auto r = test::assemble(test::hex_bar(17));
    const double top = test::largest_magnitude(r.u);
    // With hold, this much memory is left to the solve: starting a device context
    // takes more than a full device has, so that happens first.
    const auto solve = [&](std::size_t budget, std::optional<std::size_t> leave = std::nullopt) {
        auto backend = create<Backend>(budget);
        EXPECT_TRUE(backend);
        backend->set_near_nullspace(r.modes);
        const std::unique_ptr<void, void (*)(void*)> held(leave ? cuda_test_hold_all_but(*leave) : nullptr, cuda_test_release);
        const auto x = backend->solve(r.k, r.f);
        EXPECT_TRUE(backend->last_refinement() && backend->last_refinement()->converged);
        for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], r.u[i], 1e-6 * top) << i;
        const auto report = backend->last_amg_device_report();
        EXPECT_TRUE(report);
        return report.value_or(AmgDeviceReport{});
    };
    if (!create<Backend>(0)) GTEST_SKIP() << "CUDA not available on this system";
    SCOPED_TRACE(label);
    constexpr std::size_t kMiB = 1024 * 1024;

    // No budget given, plenty of memory: planned anyway, and the plan is right.
    const auto resident = solve(0);
    EXPECT_EQ(resident.execution, Execution::Resident);
    EXPECT_EQ(resident.cpu_rows, 0);
    EXPECT_EQ(resident.gpu_rows, r.k.n);
    EXPECT_GT(resident.planned_bytes, 0U);
    EXPECT_LE(resident.planned_bytes, resident.budget_bytes);
    EXPECT_GE(double(resident.measured_bytes), 0.9 * double(resident.planned_bytes));
    EXPECT_LE(double(resident.measured_bytes), 1.1 * double(resident.planned_bytes) + double(64 * kMiB));

    // A budget below the resident plan moves fine rows to the CPU, within it.
    const auto spill = solve(resident.planned_bytes * 9 / 10);
    EXPECT_EQ(spill.execution, Execution::FineRowSpill);
    EXPECT_GT(spill.cpu_rows, 0);
    EXPECT_EQ(spill.gpu_rows + spill.cpu_rows, r.k.n);
    EXPECT_LE(spill.planned_bytes, spill.budget_bytes);
    EXPECT_LE(double(spill.measured_bytes), double(spill.budget_bytes) + double(64 * kMiB));

    // Below the fixed state, the host hierarchy.
    const auto host = solve(kMiB);
    EXPECT_EQ(host.execution, Execution::Host);
    EXPECT_EQ(host.gpu_rows, 0);

    // The same, from memory really in use with no budget given. The plan must see
    // what is free once the hierarchy is built, not what a budget says. What is
    // left: the headroom plus 75% of the resident plan, then under the headroom.
    constexpr std::size_t kHeadroom = 256 * kMiB;
    const auto tight = solve(0, kHeadroom + resident.planned_bytes * 3 / 4);
    EXPECT_EQ(tight.execution, Execution::FineRowSpill);
    EXPECT_LE(tight.planned_bytes, tight.budget_bytes);
    EXPECT_LE(tight.budget_bytes, resident.planned_bytes);
    const auto full = solve(0, 64 * kMiB);
    EXPECT_EQ(full.execution, Execution::Host);
    EXPECT_EQ(full.budget_bytes, 0U);
}

} // namespace

TEST(CudaPcgAmg, EveryHierarchyIsPlannedFp64) {
    check_plan<CudaPCGSolverBackend>("fp64");
}

TEST(CudaPcgAmg, EveryHierarchyIsPlannedMixed) {
    check_plan<CudaMixedPCGSolverBackend>("mixed");
}

TEST(CudaPcgAmg, Fp64HierarchyResidentSpilledAndOnTheHost) {
    check_amg<CudaPCGSolverBackend>("cuda-pcg");
}

TEST(CudaPcgAmg, MixedHierarchyResidentSpilledAndOnTheHost) {
    check_amg<CudaMixedPCGSolverBackend>("cuda-pcg-mixed");
}

TEST(CudaPcgAmg, StableBackendCarriesRigidBodyModes) {
    check_structure<CudaPCGSolverBackend>("stable");
}

TEST(CudaPcgAmg, MixedBackendCarriesRigidBodyModes) {
    check_structure<CudaMixedPCGSolverBackend>("mixed");
}

#endif
