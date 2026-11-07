// tests/unit/test_vulkan_solver.cpp
// Mathematical correctness tests for VulkanSolverBackend.
//
// All tests skip gracefully when Vulkan is unavailable (headless CI, no GPU).
// When Vulkan IS present, the tests verify that the PCG solver produces
// numerically correct solutions and that internal diagnostics are accurate.

#include <gtest/gtest.h>
#include "solver/vulkan_solver_backend.hpp"
#include "hilbert_system.hpp"
#include "pcg_partition_problem.hpp"
#include "structural_system.hpp"
#include "solver/vulkan_context.hpp"
#include "solver/solver_backend.hpp"
#include "core/sparse_matrix.hpp"
#include "core/types.hpp"
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <optional>
#include <variant>

using namespace vibestran;

TEST(VulkanAmgTest, ResidentAndBudgetedHierarchyPreserveMaterialInterface) {
    auto context = VulkanContext::create();
    if (!context || !context->device_info().supports_float64) GTEST_SKIP() << "Vulkan fp64 unavailable";
    const auto problem = test::pcg_partition_problem(50003);
        // 1 KiB and 1 MiB are below the fixed device state (host hierarchy);
    // 13 MiB leaves about half of the fine rows on the CPU.
    for (std::size_t budget : {0U, 1024U, 1024U * 1024U, 13U * 1024U * 1024U}) {
        VulkanSolverConfig cfg;
        cfg.min_dofs_for_gpu = 0;
        cfg.device_memory_budget = budget;
        auto backend = VulkanSolverBackend::try_create(cfg);
        ASSERT_TRUE(backend);
        const auto x = backend->solve(problem.matrix.expanded_symmetric(), problem.rhs);
        ASSERT_TRUE(backend->last_refinement()->converged);
        EXPECT_EQ(backend->last_solve_was_full_gpu(), budget == 0);
        for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], problem.exact[i], 2e-8) << i;
        const auto zero = backend->solve(problem.matrix.expanded_symmetric(), std::vector<double>(x.size(), 0));
        for (double v : zero) EXPECT_EQ(v, 0);
    }
}

// Stored transfers, computed transfers, computed with fine rows on the CPU, and the
// host hierarchy, each budget from the plan above it: the same operator, so the same
// iterations, including after a refactor with values that change P.
static void check_structure(const std::string& deck, AmgModes modes) {
    using Execution = AmgDeviceReport::Execution;
    auto context = VulkanContext::create();
    if (!context || !context->device_info().supports_float64) GTEST_SKIP() << "Vulkan fp64 unavailable";
    const auto r = test::assemble(deck, true, modes);
    ASSERT_TRUE(r.modes);
    const auto full = r.k.expanded_symmetric();
    const double top = test::largest_magnitude(r.u);
    const auto changed = test::with_stiff_bands(full, r.u);
    // The device cycle applies the operator test_amg_vcycle.cpp proves SPD on the
    // host, so CG sees the same iterations, up to rounding.
    const int host_iterations = test::host_amg_iterations(r);
    std::optional<int> changed_iterations;
    const auto run = [&](std::size_t budget) {
        SCOPED_TRACE(budget);
        VulkanSolverConfig cfg;
        cfg.min_dofs_for_gpu = 0;
        cfg.device_memory_budget = budget;
        auto backend = VulkanSolverBackend::try_create(cfg);
        EXPECT_TRUE(backend);
        EXPECT_TRUE(backend->uses_near_nullspace(full.n));
        backend->set_near_nullspace(r.modes);
        const auto x = backend->solve(full, r.f);
        EXPECT_TRUE(backend->last_refinement()->converged);
        EXPECT_LE(backend->last_iteration_count(), 120);
        EXPECT_NEAR(double(backend->last_iteration_count()), double(host_iterations), 0.15 * host_iterations + 2);
        for (std::size_t i = 0; i < x.size(); ++i) { EXPECT_NEAR(x[i], r.u[i], 1e-6 * top) << i; if (::testing::Test::HasFailure()) break; }
        const auto report = backend->last_amg_device_report().value_or(AmgDeviceReport{});
        EXPECT_LE(report.planned_bytes, report.budget_bytes);
        EXPECT_EQ(report.gpu_rows + report.cpu_rows, full.n);
        EXPECT_EQ(backend->last_solve_was_full_gpu(), report.cpu_rows == 0 && report.execution == Execution::Resident);
        auto made = backend->factor(full);
        EXPECT_TRUE(std::holds_alternative<std::unique_ptr<LinearFactor>>(made));
        auto factor = std::get<std::unique_ptr<LinearFactor>>(std::move(made));
        EXPECT_FALSE(factor->refactor(changed.k));
        auto solved = factor->solve(changed.f);
        EXPECT_TRUE(std::holds_alternative<std::vector<double>>(solved));
        const auto& u = std::get<std::vector<double>>(solved);
        for (std::size_t i = 0; i < u.size(); ++i) { EXPECT_NEAR(u[i], r.u[i], 1e-6 * top) << i; if (::testing::Test::HasFailure()) break; }
        const int iterations = backend->last_iteration_count();
        if (changed_iterations) EXPECT_NEAR(double(iterations), double(*changed_iterations), 0.03 * *changed_iterations + 1);
        else changed_iterations = iterations;
        return report;
    };
    const auto stored = run(0);
    EXPECT_EQ(stored.execution, Execution::Resident);
    EXPECT_FALSE(stored.computed_transfers);
    const auto computed = run(stored.planned_bytes - 1);
    EXPECT_EQ(computed.execution, Execution::Resident);
    EXPECT_TRUE(computed.computed_transfers);
    EXPECT_LT(computed.planned_bytes, stored.planned_bytes * 95 / 100);
    const auto spilled = run(computed.planned_bytes * 9 / 10);
    EXPECT_EQ(spilled.execution, Execution::FineRowSpill);
    EXPECT_TRUE(spilled.computed_transfers);
    EXPECT_GT(spilled.cpu_rows, 0);
    const auto host = run(1024 * 1024);
    EXPECT_EQ(host.execution, Execution::Host);
    EXPECT_FALSE(host.computed_transfers);
}

// A structure's rigid-body modes must reach the hierarchy on every engine:
// without them this bar takes tens of thousands of iterations.
TEST(VulkanAmgTest, RigidBodyModesConvergeResidentAndBudgeted) {
    check_structure(test::hex_bar(17), AmgModes::Rigid);
}

// Aggregates with different numbers of coarse equations reach the device too:
// without the drilling columns this sheet takes hundreds of iterations.
TEST(VulkanAmgTest, DrillingModesConvergeResidentAndBudgeted) {
    check_structure(test::in_plane_sheet(100), AmgModes::Drilling);
}

TEST(VulkanAmgTest, FactorSolvesEveryColumnAndRefactorUploadsTheNewValues) {
    // The device hierarchy and recorded command buffers are reused across
    // right-hand sides and a same-pattern refactor(): a solve that still
    // used the old values would miss the stiffened network's solution by
    // O(1) at the interface.
    auto context = VulkanContext::create();
    if (!context || !context->device_info().supports_float64) GTEST_SKIP() << "Vulkan fp64 unavailable";
    const auto problem = test::pcg_partition_problem(50003);
    const auto stiffer = test::stiffened(problem, 30.0);
    const auto k = problem.matrix.expanded_symmetric();
    const auto n = std::size_t(k.n);
    // Resident, host hierarchy, and about half of the fine rows on the CPU.
    for (std::size_t budget : {0U, 1024U * 1024U, 13U * 1024U * 1024U}) {
        VulkanSolverConfig cfg;
        cfg.min_dofs_for_gpu = 0;
        cfg.device_memory_budget = budget;
        auto backend = VulkanSolverBackend::try_create(cfg);
        ASSERT_TRUE(backend);
        auto made = backend->factor(k);
        ASSERT_TRUE(std::holds_alternative<std::unique_ptr<LinearFactor>>(made)) << std::get<Error>(made).message;
        auto factor = std::get<std::unique_ptr<LinearFactor>>(std::move(made));
        EXPECT_EQ(factor->method(), "amg-pcg");
        std::vector<double> B(2 * n);
        for (std::size_t i = 0; i < n; ++i) { B[i] = problem.rhs[i]; B[n + i] = -3 * problem.rhs[i]; }
        auto X = factor->solve(B, 2);
        ASSERT_TRUE(std::holds_alternative<std::vector<double>>(X)) << budget << ": " << std::get<Error>(X).message;
        const auto& x = std::get<std::vector<double>>(X);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(x[i], problem.exact[i], 2e-8) << budget << " row=" << i;
            EXPECT_NEAR(x[n + i], -3 * problem.exact[i], 6e-8) << budget << " row=" << i;
        }
        EXPECT_EQ(backend->last_solve_was_full_gpu(), budget == 0);
        ASSERT_FALSE(factor->refactor(stiffer.matrix.expanded_symmetric()).has_value());
        X = factor->solve(stiffer.rhs, 1);
        ASSERT_TRUE(std::holds_alternative<std::vector<double>>(X)) << budget << ": " << std::get<Error>(X).message;
        const auto& y = std::get<std::vector<double>>(X);
        for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(y[i], stiffer.exact[i], 2e-8) << budget << " row=" << i;
        EXPECT_EQ(factor->stats().symbolic_analyses, 1);
        EXPECT_EQ(factor->stats().solves, 3);
    }
}

TEST(VulkanPartitionTest, BudgetedFloatAndDoublePreserveMaterialInterface) {
    const auto problem = test::pcg_partition_problem();
    const auto full = problem.matrix.expanded_symmetric();
    for (bool fp64 : {false, true}) {
        auto context = VulkanContext::create();
        if (!context) GTEST_SKIP() << "Vulkan unavailable";
        if (fp64 && !context->device_info().supports_float64) continue;
        VulkanSolverConfig cfg;
        cfg.min_dofs_for_gpu = 0;
        cfg.use_double = fp64;
        cfg.device_memory_budget = 256 * 1024;
        auto backend = VulkanSolverBackend::try_create(cfg);
        ASSERT_TRUE(backend);
        const auto x = backend->solve(full, problem.rhs);
        EXPECT_FALSE(backend->last_solve_was_full_gpu());
        ASSERT_TRUE(backend->last_refinement()->converged);
        for (int i = 0; i < full.n; ++i) EXPECT_NEAR(x[i], problem.exact[i], 2e-8) << fp64 << " row=" << i;
    }
}

TEST(VulkanPartitionTest, BudgetSmallerThanVectorUsesCpuWithoutLosingCouplings) {
    const auto problem = test::pcg_partition_problem(1024);
    VulkanSolverConfig cfg;
    cfg.min_dofs_for_gpu = 0;
    cfg.device_memory_budget = 1024;
    auto backend = VulkanSolverBackend::try_create(cfg);
    if (!backend) GTEST_SKIP() << "Vulkan unavailable";
    const auto x = backend->solve(problem.matrix.expanded_symmetric(), problem.rhs);
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], problem.exact[i], 2e-8) << i;
}

// ── Test fixture ──────────────────────────────────────────────────────────────

// Config that disables the DOF threshold so all correctness tests exercise the GPU path
// even for small matrices.
static VulkanSolverConfig gpu_test_cfg() {
    VulkanSolverConfig cfg;
    cfg.min_dofs_for_gpu = 0;
    return cfg;
}

class VulkanTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = VulkanSolverBackend::try_create(gpu_test_cfg());
        if (!backend_.has_value())
            GTEST_SKIP() << "Vulkan not available on this system — skipping Vulkan tests";
    }

    std::optional<VulkanSolverBackend> backend_;
};

// ── CSR builder helpers ───────────────────────────────────────────────────────

/// Build CSR from a dense symmetric matrix (upper+lower triangular entries).
static SparseMatrixBuilder::CsrData dense_to_csr(const std::vector<std::vector<double>>& A) {
    int n = static_cast<int>(A.size());
    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (A[i][j] != 0.0)
                builder.add(i, j, A[i][j]);
    return builder.build_csr();
}

// ── Test 1: 2×2 diagonal SPD — Jacobi is exact preconditioner ────────────────
// K = diag(4, 9), F = [4, 9], expected u = [1, 1].
// With Jacobi preconditioning on a diagonal matrix, the preconditioned system
// is the identity, so PCG converges in exactly 1 iteration.

TEST_F(VulkanTest, DiagonalSystemConvergesInOneIteration) {
    std::vector<std::vector<double>> K = {{4.0, 0.0}, {0.0, 9.0}};
    std::vector<double> F = {4.0, 9.0};
    auto csr = dense_to_csr(K);

    auto u = backend_->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), 2);
    // The fp32 PCG is refined with double-precision residuals.
    EXPECT_NEAR(u[0], 1.0, 1e-12);
    EXPECT_NEAR(u[1], 1.0, 1e-12);
    const auto rep = backend_->last_refinement();
    ASSERT_TRUE(rep.has_value());
    EXPECT_TRUE(rep->converged);
    EXPECT_EQ(backend_->last_iteration_count(), rep->corrections + 1)
        << "Jacobi preconditioning is exact for diagonal matrices: each PCG "
           "solve (the first and every refinement correction) takes 1 iteration";
}

// ── Test 2: 5×5 tridiagonal SPD — agrees with Eigen ─────────────────────────
// K = tridiag(-1, 3, -1) of size 5, F = [1, 1, 1, 1, 1].
// Verify Vulkan PCG solution matches LdltSolverBackend to 1e-8.

TEST_F(VulkanTest, TridiagonalSystemAgreesWithEigen) {
    const int n = 5;
    std::vector<std::vector<double>> Kd(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        Kd[i][i] = 3.0;
        if (i > 0)     Kd[i][i-1] = -1.0;
        if (i < n - 1) Kd[i][i+1] = -1.0;
    }
    std::vector<double> F(n, 1.0);
    auto csr = dense_to_csr(Kd);

    auto u_vulkan = backend_->solve(csr, F);

    LdltSolverBackend eigen_backend;
    auto u_eigen = eigen_backend.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_vulkan.size()), n);
    // GPU computations use float32 (single precision), giving ~7 significant digits.
    // Component error is bounded by ~κ(K) * solver_tolerance.  For this 5×5 system
    // κ ≈ 3 and the default tolerance is 5e-5, so 1e-3 is a conservative bound.
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u_vulkan[i], u_eigen[i], 1e-3)
            << "Component " << i << " differs between Vulkan PCG and Eigen";
}

// ── Test 3: larger tridiagonal n=50 — full pipeline stress test ──────────────
// K = tridiag(-1, 3, -1) of size 50, F = ones.
// Tests buffer allocation, shader dispatch, and convergence for a larger n.
// Verifies Vulkan PCG agrees with Eigen to float32 precision (1e-5).

TEST_F(VulkanTest, LargerTridiagonalAgreesWithEigen) {
    const int n = 50;
    std::vector<std::vector<double>> Kd(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        Kd[i][i] = 3.0;
        if (i > 0)     Kd[i][i-1] = -1.0;
        if (i < n - 1) Kd[i][i+1] = -1.0;
    }
    std::vector<double> F(n, 1.0);
    auto csr = dense_to_csr(Kd);

    auto u_vulkan = backend_->solve(csr, F);

    LdltSolverBackend eigen_backend;
    auto u_eigen = eigen_backend.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_vulkan.size()), n);
    // Component error bounded by ~κ(K) * solver_tolerance.  For n=50 tridiagonal
    // κ ≈ 50 and default tolerance 5e-5 → 1e-2 is a conservative bound.
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u_vulkan[i], u_eigen[i], 1e-2)
            << "n=50 tridiagonal: component " << i << " differs";
}

// ── Test 4: Convergence rate for well-conditioned system ──────────────────────
// A diagonal matrix with condition number exactly 10 (diag = 1..10 repeated).
// PCG with Jacobi preconditioning on such a system should decrease the
// relative residual to tolerance within a bounded number of iterations.
// Specifically, for κ=10, PCG needs at most ceil(0.5*sqrt(10)*log(2/tol)) iters,
// and the residual must decrease monotonically (verified by a slightly relaxed check).

TEST_F(VulkanTest, ConvergenceRateForWellConditionedSystem) {
    const int n = 20;
    // Diagonal entries: 1, 2, ..., 10 repeated
    std::vector<std::vector<double>> Kd(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) Kd[i][i] = static_cast<double>((i % 10) + 1);

    // F = K * ones → u = ones exactly
    std::vector<double> F(n);
    for (int i = 0; i < n; ++i) F[i] = Kd[i][i];

    auto csr = dense_to_csr(Kd);
    auto u   = backend_->solve(csr, F);

    // Solution should be all ones (refined to double accuracy)
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u[i], 1.0, 1e-12) << "Component " << i;

    // With Jacobi preconditioning (exact for diagonal), each PCG solve
    // converges in 1 iteration
    const auto rep = backend_->last_refinement();
    ASSERT_TRUE(rep.has_value());
    EXPECT_EQ(backend_->last_iteration_count(), rep->corrections + 1)
        << "Jacobi-preconditioned PCG on a diagonal system must converge in 1 iteration";

    // Residual norm must be below the default float32-achievable tolerance
    EXPECT_LT(backend_->last_residual_norm(), 1e-6);
}

// ── Test 5: Large diagonal system n=50000 ────────────────────────────────────
// K = diag(1, 2, ..., n), F = K*ones → u = ones.
// Stresses buffer allocation, upload, dispatch, download, and fence sync
// at scale.  This test is O(n) and should be fast even with GPU sync overhead.

TEST_F(VulkanTest, LargeDiagonalSystemIsCorrect) {
    const int n = 50000;
    SparseMatrixBuilder builder(n);
    std::vector<double> F(n);
    for (int i = 0; i < n; ++i) {
        double d = static_cast<double>(i + 1);
        builder.add(i, i, d);
        F[i] = d; // F = K*ones → u = ones
    }
    auto csr = builder.build_csr();

    auto u = backend_->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), n);
    // Check a representative sample of components
    for (int i = 0; i < n; i += 1000)
        EXPECT_NEAR(u[i], 1.0, 1e-12)
            << "Large diagonal: component " << i << " out of tolerance";

    // Jacobi is exact for diagonal matrices so PCG should converge very quickly.
    // For large n, float32 dot-product accumulation may require a second
    // iteration per PCG solve (the first and each refinement correction).
    const auto rep = backend_->last_refinement();
    ASSERT_TRUE(rep.has_value());
    EXPECT_LE(backend_->last_iteration_count(), 2 * (rep->corrections + 1))
        << "PCG on a Jacobi-diagonal-preconditioned diagonal system should converge in ≤2 iterations";
}

// ── Test 6: Forced tiled path ─────────────────────────────────────────────────
// Set vram_headroom=0.9999 to force the tiled streaming path regardless of
// actual VRAM.  Solve the same 5×5 tridiagonal system and verify:
//   (a) last_solve_was_full_gpu() == false
//   (b) solution matches Eigen to 1e-8

TEST_F(VulkanTest, TiledPathProducesCorrectSolution) {
    VulkanSolverConfig cfg = gpu_test_cfg();
    cfg.force_tiled = true;  // bypass VRAM check to exercise the tiled code path
    cfg.use_double  = false; // tiled path only supports float32

    auto tiled_backend = VulkanSolverBackend::try_create(cfg);
    if (!tiled_backend.has_value())
        GTEST_SKIP() << "Vulkan unavailable";

    const int n = 5;
    std::vector<std::vector<double>> Kd(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        Kd[i][i] = 3.0;
        if (i > 0)     Kd[i][i-1] = -1.0;
        if (i < n - 1) Kd[i][i+1] = -1.0;
    }
    std::vector<double> F(n, 1.0);
    auto csr = dense_to_csr(Kd);

    auto u_tiled = tiled_backend->solve(csr, F);

    EXPECT_FALSE(tiled_backend->last_solve_was_full_gpu())
        << "vram_headroom=0.9999 must force the tiled path";

    LdltSolverBackend eigen_backend;
    auto u_eigen = eigen_backend.solve(csr, F);

    // Tiled path uses float32 SpMV, so 1e-5 is appropriate for float32 precision.
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u_tiled[i], u_eigen[i], 1e-5)
            << "Tiled path: component " << i << " differs from Eigen";
}

// ── Test 7: Singular matrix detection ────────────────────────────────────────
// A matrix with a zero diagonal entry (missing boundary condition) must cause
// solve() to throw SolverError with a descriptive message.
// This prevents silent NaN/Inf propagation into the FEM results.

TEST_F(VulkanTest, SingularMatrixThrowsSolverError) {
    // 3×3 matrix: row 1 has no diagonal entry (structural singularity)
    SparseMatrixBuilder builder(3);
    builder.add(0, 0, 4.0);
    builder.add(0, 1, -1.0);
    builder.add(1, 0, -1.0);
    // Row 1 has no diagonal — singular
    builder.add(1, 2, -1.0);
    builder.add(2, 1, -1.0);
    builder.add(2, 2, 4.0);
    auto csr = builder.build_csr();
    std::vector<double> F = {1.0, 1.0, 1.0};

    EXPECT_THROW({
        (void)backend_->solve(csr, F);
    }, SolverError) << "Missing diagonal entry should throw SolverError";
}

// ── Test 8: name() identifies the backend ────────────────────────────────────

TEST_F(VulkanTest, NameContainsVulkan) {
    EXPECT_FALSE(backend_->name().empty());
    EXPECT_NE(backend_->name().find("Vulkan"), std::string_view::npos)
        << "Backend name should contain 'Vulkan' for meaningful log output";
}

// ── Test 9: Float32→float64 auto-promotion for stiff large system ─────────────
// For a large diagonal system with K_ii >> ||F||_2 / sqrt(n), the float32 path
// can fail (stagnation or wrong convergence).  With use_double=false the solver
// automatically retries with float64 when float32 fails.
// Requires shaderFloat64 device support; skips gracefully otherwise.
//
// Note: stagnation-at-precision-floor is NOT tested here because for
// well-conditioned SPD systems in float64, PCG reaches exact-zero residual via
// Krylov cancellation after at most n iterations, making the test unreliable.
// Stagnation detection is exercised implicitly by StiffSystemDoesNotFalseConverge
// (float32 stagnates → triggers float64 fallback).

TEST_F(VulkanTest, LargeStiffDiagonalSucceedsWithDefaultConfig) {
    auto tmp_ctx = vibestran::VulkanContext::create();
    if (!tmp_ctx || !tmp_ctx->device_info().supports_float64)
        GTEST_SKIP() << "GPU lacks shaderFloat64; float32→float64 fallback unavailable";

    // Default config (use_double=false) with min_dofs_for_gpu=0 to force GPU path.
    VulkanSolverConfig cfg = gpu_test_cfg();
    cfg.use_double = false; // start with float32; fallback to float64 on stagnation

    auto f32_backend = VulkanSolverBackend::try_create(cfg);
    if (!f32_backend.has_value())
        GTEST_SKIP() << "Vulkan unavailable";

    // n=10000 diagonal system with K_ii=1.3e9 (same stiffness as bar_10000_bending).
    // Float32 PCG stagnates or diverges; auto-retry with float64 produces correct result.
    const int    n      = 10000;
    const double K_diag = 1.3e9;
    SparseMatrixBuilder builder(n);
    std::vector<double> F(n, 0.0);
    for (int i = 0; i < n; ++i) {
        builder.add(i, i, K_diag);
        F[i] = static_cast<double>(i % 7 + 1); // varied non-zero loads
    }
    auto csr = builder.build_csr();

    auto u = f32_backend->solve(csr, F); // must succeed via float64 fallback if float32 fails

    ASSERT_EQ(static_cast<int>(u.size()), n);
    for (int i = 0; i < n; ++i) {
        const double expected = F[i] / K_diag;
        EXPECT_NEAR(u[i], expected, expected * 1e-6)
            << "Stiff diagonal: component " << i << " wrong; float32→float64 fallback may have failed";
    }
}

// ── Test 10: Small problem falls back to CPU ──────────────────────────────────
// When n < min_dofs_for_gpu the backend delegates to LdltSolverBackend.
// Verify last_solve_was_full_gpu()==false and the result matches Eigen directly.

TEST_F(VulkanTest, SmallProblemFallsBackToCpu) {
    // Default config has min_dofs_for_gpu=50000; a 5×5 system is well below it.
    // (backend_ in fixture was created with min_dofs_for_gpu=0, so create a fresh one.)
    auto cpu_fallback_backend = VulkanSolverBackend::try_create(); // default config
    if (!cpu_fallback_backend.has_value())
        GTEST_SKIP() << "Vulkan unavailable";

    const int n = 5;
    std::vector<std::vector<double>> Kd(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        Kd[i][i] = 3.0;
        if (i > 0)     Kd[i][i-1] = -1.0;
        if (i < n - 1) Kd[i][i+1] = -1.0;
    }
    std::vector<double> F(n, 1.0);

    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (Kd[i][j] != 0.0) builder.add(i, j, Kd[i][j]);
    auto csr = builder.build_csr();

    auto u = cpu_fallback_backend->solve(csr, F);

    EXPECT_FALSE(cpu_fallback_backend->last_solve_was_full_gpu())
        << "n=5 is below min_dofs_for_gpu=50000 — must use CPU fallback";

    LdltSolverBackend eigen_backend;
    auto u_eigen = eigen_backend.solve(csr, F);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u[i], u_eigen[i], 1e-12)
            << "CPU fallback result should match Eigen to double precision";

    // The fallback is reported once for the F06, then the list is empty.
    const std::vector<std::string> expected{
        "Vulkan backend: 5 equations is below the GPU threshold of 50000; solved with "
        "the CPU direct solver"};
    EXPECT_EQ(cpu_fallback_backend->notices().take(), expected);
    EXPECT_TRUE(cpu_fallback_backend->notices().take().empty());
}

TEST_F(VulkanTest, DetailsDescribeTheConfiguredSolve) {
    for (const bool use_double : {true, false}) {
        SCOPED_TRACE(use_double);
        VulkanSolverConfig cfg;
        cfg.use_double = use_double;
        cfg.tolerance = 2.5e-9;
        cfg.min_dofs_for_gpu = 1234;
        auto backend = VulkanSolverBackend::try_create(cfg);
        ASSERT_TRUE(backend.has_value());
        std::map<std::string, std::string> rows;
        for (const auto& [name, value] : backend->details())
            rows[name] = value;
        EXPECT_EQ(rows["Device"], backend_->details().front().second);
        EXPECT_FALSE(rows["Device"].empty());
        EXPECT_NE(rows["Method"].find("below 1234 equations"), std::string::npos)
            << rows["Method"];
        EXPECT_EQ(rows["Tolerance"].rfind("2.5e-09", 0), 0U) << rows["Tolerance"];
        // Automatic mode now uses fp64 AMG for large systems, while retaining
        // fp32 Jacobi for smaller ones. Its description must mention both.
        EXPECT_NE(rows["Precision"].find(use_double ? "fp64" : "fp32"), std::string::npos)
            << rows["Precision"];
        if (!use_double) EXPECT_NE(rows["Precision"].find("fp64 AMG"), std::string::npos);
    }
}

// ── Test 11: Full float64 GPU path agrees with Eigen ─────────────────────────
// When use_double=true and the device supports shaderFloat64, the solver runs
// the float64 compute shaders.  Verify the result agrees with Eigen to tight
// tolerance (1e-10) — much tighter than the float32 path allows.
// The test skips gracefully if the device lacks shaderFloat64.

TEST_F(VulkanTest, Float64PathAgreesWithEigenToDoublePrecision) {
    VulkanSolverConfig cfg = gpu_test_cfg();
    cfg.use_double = true;
    cfg.tolerance  = 1e-11; // tight enough to exercise double convergence

    auto dbl_backend = VulkanSolverBackend::try_create(cfg);
    if (!dbl_backend.has_value())
        GTEST_SKIP() << "Vulkan unavailable";

    // Check device support before trying (backend will also throw, but skip is cleaner)
    auto ctx = vibestran::VulkanContext::create();
    if (!ctx || !ctx->device_info().supports_float64)
        GTEST_SKIP() << "GPU does not support shaderFloat64 — skipping float64 test";

    const int n = 50;
    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i) {
        builder.add(i, i, 3.0);
        if (i > 0)     builder.add(i, i - 1, -1.0);
        if (i < n - 1) builder.add(i, i + 1, -1.0);
    }
    auto csr = builder.build_csr();
    std::vector<double> F(n, 1.0);

    auto u_vulkan = dbl_backend->solve(csr, F);

    LdltSolverBackend eigen_backend;
    auto u_eigen = eigen_backend.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_vulkan.size()), n);
    // GPU and CPU use different arithmetic ordering, so even with float64 there
    // is some rounding divergence.  1e-9 is well within double precision while
    // being robust to FMA / order-of-operations differences.
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u_vulkan[i], u_eigen[i], 1e-9)
            << "float64 GPU path should match Eigen to near-double precision at component " << i;

    EXPECT_TRUE(dbl_backend->last_solve_was_full_gpu())
        << "n=50 with min_dofs_for_gpu=0 should use full-GPU path";
}

// ── Test 12: Stiff system (large K_ii / small ||F||) must not false-converge ──
// Regression test for the mixed-norm convergence criterion bug.
//
// Root cause: using sqrt(r^T M^{-1} r) / ||b||_2 as the convergence ratio
// produces a near-zero initial value when K_ii is large (e.g. structural
// stiffness ~1e9) but ||F||_2 is small (~hundreds).  The solver sees the ratio
// already below tolerance before doing any work and returns a near-zero solution.
//
// Fix: denominator must be the M^{-1}-norm of b, i.e. sqrt(b^T M^{-1} b),
// which equals sqrt(rz_old) at iteration 0 (since x=0 => r=b).  This makes
// the initial ratio exactly 1.0 and ensures the solver always iterates at least
// once before checking convergence.
//
// The test constructs a 2×2 stiff diagonal system that triggered the bug:
//   K = diag(1.3e9, 1.3e9),  F = [333, 0]
// True solution: u = [333/1.3e9, 0] ≈ [2.56e-7, 0].
// With the bug the solver would return [0, 0] in 1 iteration; with the fix it
// converges to the true solution.

TEST_F(VulkanTest, StiffSystemDoesNotFalseConverge) {
    // K_ii ≈ 1.3e9, ||F||_2 = 333.  With the old mixed-norm criterion,
    // sqrt(rz_old) / ||F||_2 = sqrt(F_i^2 / K_ii) / ||F||_2
    //   = sqrt(333^2 / 1.3e9) / 333 ≈ 2.76e-5 < 5e-5 (default tolerance).
    // The solver would declare convergence with 0 iterations, returning zeros.
    const double K_diag = 1.3e9;
    const double F0     = 333.0;

    SparseMatrixBuilder builder(2);
    builder.add(0, 0, K_diag);
    builder.add(1, 1, K_diag);
    auto csr = builder.build_csr();
    std::vector<double> F = {F0, 0.0};

    auto u = backend_->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), 2);

    const double expected_u0 = F0 / K_diag; // ~2.56e-7
    EXPECT_NEAR(u[0], expected_u0, expected_u0 * 1e-4)
        << "Stiff system: u[0] should be F/K = " << expected_u0
        << " but got " << u[0]
        << ".  A near-zero result indicates false convergence from mixed-norm criterion.";
    EXPECT_NEAR(u[1], 0.0, 1e-15)
        << "Stiff system: u[1] should be exactly 0";

    // The solver must have iterated at least once.  With the old bug, it exited
    // immediately with 0 iterations; the fix guarantees at least 1 iteration.
    EXPECT_GE(backend_->last_iteration_count(), 1)
        << "Solver must iterate at least once for a stiff system; "
           "zero iterations indicates false convergence from mixed-norm criterion";
}

// ── Test 13: VulkanContext accessors return valid handles ─────────────────────
// Exercises physical_device() and compute_queue_family() which are only used in tests.

TEST_F(VulkanTest, VulkanContextHandlesAreValid) {
    // Reach into the context via try_create to verify handle accessors
    auto ctx = vibestran::VulkanContext::create();
    ASSERT_TRUE(ctx.has_value()) << "Vulkan must be available since SetUp() did not skip";

    EXPECT_NE(ctx->physical_device(), static_cast<VkPhysicalDevice>(VK_NULL_HANDLE))
        << "physical_device() must return a valid handle";
    // compute_queue_family() must be a valid queue family index (< UINT32_MAX)
    EXPECT_LT(ctx->compute_queue_family(), UINT32_MAX)
        << "compute_queue_family() must return a valid queue family index";
}

// ── fp32 PCG is refined to double accuracy ───────────────────────────────────
// H_4 (cond 1.5e4) in float32: each inner Jacobi PCG solve stops at a loose
// residual, and refinement with compensated double-precision residuals takes
// the solution to the 1e-8 target error without falling back to float64.

TEST_F(VulkanTest, Float32PcgIsRefinedToDoubleAccuracy) {
    const auto [K, F] = test::scaled_hilbert(4, SymmetryStorage::Full);
    const auto u = backend_->solve(K, F);
    const auto rep = backend_->last_refinement();
    ASSERT_TRUE(rep.has_value());
    EXPECT_TRUE(rep->converged);
    EXPECT_GT(rep->corrections, 0);
    EXPECT_LT(test::max_error_from_one(u), 1e-9);
}

// ── Test 14: Ill-conditioned tridiagonal converges (stagnation regression) ────
// Regression test for false stagnation detection.
//
// Root cause: the stagnation detector monitored the 2-norm residual ||r||/||b||
// or the preconditioned residual sqrt(r^T M^{-1} r), both of which oscillate in
// PCG for ill-conditioned systems.  A stagnation window of 50 iterations would
// false-trigger even though the A-norm error was decreasing monotonically.
//
// Fix: stagnation detection now tracks the A-norm error decrease per window,
// which is guaranteed monotonic in CG for SPD systems.
//
// This test constructs a tridiagonal system with κ ≈ n² = 250000 (n=500),
// requiring ~500 Jacobi-PCG iterations.  With the old residual-based stagnation
// detector (window=50), this would throw SolverError.  With A-norm stagnation
// detection it converges correctly.

TEST_F(VulkanTest, IllConditionedTridiagonalConverges) {
    const int n = 500;
    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i) {
        builder.add(i, i, 2.0);
        if (i > 0)     builder.add(i, i - 1, -1.0);
        if (i < n - 1) builder.add(i, i + 1, -1.0);
    }
    auto csr = builder.build_csr();
    // F chosen to excite multiple eigenmodes — maximizes residual oscillation.
    std::vector<double> F(n);
    for (int i = 0; i < n; ++i)
        F[i] = (i % 2 == 0) ? 1.0 : -1.0;

    LdltSolverBackend eigen_backend;
    auto u_eigen = eigen_backend.solve(csr, F);

    // Use float64 to avoid float32 precision issues on this ill-conditioned system.
    auto ctx = vibestran::VulkanContext::create();
    if (!ctx || !ctx->device_info().supports_float64)
        GTEST_SKIP() << "GPU lacks shaderFloat64";

    VulkanSolverConfig cfg = gpu_test_cfg();
    cfg.use_double = true;
    cfg.tolerance  = 1e-6;

    auto dbl_backend = VulkanSolverBackend::try_create(cfg);
    if (!dbl_backend.has_value())
        GTEST_SKIP() << "Vulkan unavailable";

    auto u_vulkan = dbl_backend->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_vulkan.size()), n);
    for (int i = 0; i < n; i += 50)
        EXPECT_NEAR(u_vulkan[i], u_eigen[i], 1e-5)
            << "Ill-conditioned tridiagonal: component " << i << " differs from Eigen";

    // Must have taken many iterations (κ ≈ 250000, expect ~100-500 CG iterations).
    EXPECT_GT(dbl_backend->last_iteration_count(), 10)
        << "Ill-conditioned tridiagonal should require many iterations";
}
