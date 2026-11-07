// tests/unit/test_pcg_solver.cpp
// Mathematical correctness tests for PCG solver backends.
//
// CpuPCGSolverBackend tests run unconditionally (CPU only, no GPU required).
// CudaPCGSolverBackend tests are skipped gracefully when CUDA is unavailable.
//
// Tests verify:
//  - Small exact-solution systems (diagonal, tridiagonal)
//  - Agreement with the direct Eigen backend to near-solver tolerance
//  - FEM-scale stiffness values don't cause false convergence
//  - Well-conditioned and mildly ill-conditioned systems converge
//  - Zero RHS returns zero displacement without iterating
//  - Singular matrix detection (non-positive pAp)
//  - Diagnostics (iteration count, residual) are populated correctly

#include <gtest/gtest.h>
#include "structural_system.hpp"
#include "solver/solver_backend.hpp"
#include "core/sparse_matrix.hpp"
#include "core/exceptions.hpp"
#include "hilbert_system.hpp"
#include "pcg_partition_problem.hpp"
#include <cmath>
#include <vector>

#ifdef HAVE_CUDA
#include "solver/cuda_mixed_pcg_solver_backend.hpp"
#include "solver/cuda_pcg_solver_backend.hpp"

TEST(CudaAmgTest, ResidentAndBudgetedHierarchyPreserveMaterialInterface) {
    const auto problem = vibestran::test::pcg_partition_problem(50003);
    for (std::size_t budget : {0U, 1024U, 1024U * 1024U, 8U * 1024U * 1024U}) {
        auto stable = vibestran::CudaPCGSolverBackend::try_create(1e-8, 10000, budget);
        auto mixed = vibestran::CudaMixedPCGSolverBackend::try_create(1e-8, 10000, budget);
        if (!stable || !mixed) GTEST_SKIP() << "CUDA unavailable";
        for (vibestran::SolverBackend* backend : {static_cast<vibestran::SolverBackend*>(&*stable),
                                               static_cast<vibestran::SolverBackend*>(&*mixed)}) {
            const auto x = backend->solve(problem.matrix, problem.rhs);
            ASSERT_TRUE(backend->last_refinement()->converged);
            for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], problem.exact[i], 2e-8) << i;
            const auto zero = backend->solve(problem.matrix, std::vector<double>(x.size(), 0));
            for (double v : zero) EXPECT_EQ(v, 0);
        }
    }
}

TEST(CudaPartitionTest, DoubleAndMixedPreserveMaterialInterface) {
    const auto problem = vibestran::test::pcg_partition_problem();
    // Exercise both a nonempty GPU prefix and a budget too small for even
    // the input vector; both must preserve the material-interface couplings.
    for (std::size_t budget : {1024U, 256U * 1024U}) {
        auto stable = vibestran::CudaPCGSolverBackend::try_create(1e-8, 10000, budget);
        auto mixed = vibestran::CudaMixedPCGSolverBackend::try_create(1e-8, 10000, budget);
        if (!stable || !mixed) GTEST_SKIP() << "CUDA unavailable";
        for (vibestran::SolverBackend* backend : {static_cast<vibestran::SolverBackend*>(&*stable),
                                               static_cast<vibestran::SolverBackend*>(&*mixed)}) {
            const auto x = backend->solve(problem.matrix, problem.rhs);
            ASSERT_TRUE(backend->last_refinement()->converged);
            for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], problem.exact[i], 2e-8) << i;
            const auto zero = backend->solve(problem.matrix, std::vector<double>(x.size(), 0.0));
            for (double v : zero) EXPECT_EQ(v, 0.0);
        }
    }
}

TEST(CudaPartitionTest, ColoredResidentSolveRecoversOriginalEquationOrdering) {
    const auto problem = vibestran::test::pcg_partition_problem(4096);
    auto stable = vibestran::CudaPCGSolverBackend::try_create(1e-8);
    auto mixed = vibestran::CudaMixedPCGSolverBackend::try_create(1e-8);
    if (!stable || !mixed) GTEST_SKIP() << "CUDA unavailable";
    for (vibestran::SolverBackend* backend : {static_cast<vibestran::SolverBackend*>(&*stable),
                                           static_cast<vibestran::SolverBackend*>(&*mixed)}) {
        const auto x = backend->solve(problem.matrix, problem.rhs);
        ASSERT_TRUE(backend->last_refinement()->converged);
        for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], problem.exact[i], 2e-8) << i;
    }
}
#endif

using namespace vibestran;

// ── Shared helper ─────────────────────────────────────────────────────────────

static SparseMatrixBuilder::CsrData make_tridiagonal(int n, double diag,
                                                     double off,
                                                     bool lower_only = false) {
    SparseMatrixBuilder b(n);
    for (int i = 0; i < n; ++i) {
        b.add(i, i, diag);
        if (i > 0) {
            b.add(i, i - 1, off);
            if (!lower_only)
                b.add(i - 1, i, off);
        }
    }
    return b.build_csr();
}

static SparseMatrixBuilder::CsrData make_diagonal(int n, double val) {
    SparseMatrixBuilder b(n);
    for (int i = 0; i < n; ++i) b.add(i, i, val);
    return b.build_csr();
}

// ══════════════════════════════════════════════════════════════════════════════
// CpuPCGSolverBackend tests
// ══════════════════════════════════════════════════════════════════════════════

class CpuPCGTest : public ::testing::Test {
protected:
    std::vector<double> solve(const SparseMatrixBuilder::CsrData& csr,
                              const std::vector<double>& force) {
        return pcg_.solve(csr, force);
    }

    int last_iteration_count() const {
        return pcg_.last_iteration_count();
    }

    double last_estimated_error() const {
        return pcg_.last_estimated_error();
    }

private:
    CpuPCGSolverBackend pcg_;
};

// ── Test 1: 2×2 diagonal system — exact solution ──────────────────────────────
// K = diag(4, 9), F = [8, 27], u = [2, 3].
// PCG converges in one iteration: on so few equations the hierarchy is the
// coarsest level alone, whose inverse is exact.

TEST_F(CpuPCGTest, DiagonalSystemExactSolution) {
    SparseMatrixBuilder b(2);
    b.add(0, 0, 4.0);
    b.add(1, 1, 9.0);
    auto csr = b.build_csr();
    std::vector<double> F = {8.0, 27.0};

    auto u = solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), 2);
    EXPECT_NEAR(u[0], 2.0, 1e-9);
    EXPECT_NEAR(u[1], 3.0, 1e-9);
    // IC0 is exact for diagonal K, so CG converges in 0 or 1 iterations.
    EXPECT_LE(last_iteration_count(), 1)
        << "PCG with exact IC0 preconditioner should converge in at most 1 step";
}

// ── Test 2: 5×5 tridiagonal — agrees with direct Eigen ───────────────────────
// K = tridiag(-1, 3, -1), F = ones.  Condition number moderate (~8).

TEST_F(CpuPCGTest, TridiagonalAgreesWithDirectSolver) {
    const int n = 5;
    auto csr = make_tridiagonal(n, 3.0, -1.0);
    std::vector<double> F(n, 1.0);

    auto u_pcg = solve(csr, F);

    LdltSolverBackend direct;
    auto u_direct = direct.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_pcg.size()), n);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u_pcg[i], u_direct[i], 1e-7)
            << "PCG component " << i << " disagrees with direct solver";
}

// ── Test 3: n=500 FEM-like banded SPD matrix ─────────────────────────────────
// K is a banded diagonally dominant SPD matrix with pattern mimicking 1-D FEM.
// κ(K) is moderate; PCG should converge quickly.

TEST_F(CpuPCGTest, LargeBandedSpdAgreesWithDirect) {
    const int n = 500;
    SparseMatrixBuilder b(n);
    for (int i = 0; i < n; ++i) {
        b.add(i, i, static_cast<double>(n + 1));
        if (i > 0)     { b.add(i, i - 1, -1.0); b.add(i - 1, i, -1.0); }
        if (i > 1)     { b.add(i, i - 2, -0.5); b.add(i - 2, i, -0.5); }
    }
    auto csr = b.build_csr();
    std::vector<double> F(n);
    for (int i = 0; i < n; ++i) F[i] = static_cast<double>((i % 7) + 1);

    auto u_pcg = solve(csr, F);

    LdltSolverBackend direct;
    auto u_direct = direct.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_pcg.size()), n);
    for (int i = 0; i < n; i += 50)
        EXPECT_NEAR(u_pcg[i], u_direct[i], 1e-6)
            << "n=500 banded PCG: component " << i << " disagrees with direct solver";

    EXPECT_LT(last_estimated_error(), 1e-7)
        << "PCG estimated error should be below tolerance";
}

// ── Test 4: FEM-scale stiffness values — no false convergence ────────────────
// K_ii = 1.3e9 (large structural stiffness), F_i = 333.
// Verifies PCG does not converge falsely on a stiff diagonal system.
// True solution: u_i = 333 / 1.3e9 ≈ 2.56e-7.

TEST_F(CpuPCGTest, StiffDiagonalDoesNotConvergeFalsely) {
    const double K_diag = 1.3e9;
    const double F_val  = 333.0;
    const int    n      = 100;
    auto csr = make_diagonal(n, K_diag);
    std::vector<double> F(n, F_val);

    auto u = solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), n);
    const double expected = F_val / K_diag;
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u[i], expected, expected * 1e-6)
            << "Stiff diagonal: component " << i << " wrong";
}

// ── Test 5: zero RHS — returns zero without iterating ────────────────────────

TEST_F(CpuPCGTest, ZeroRhsReturnZeroDisplacement) {
    auto csr = make_tridiagonal(10, 3.0, -1.0);
    std::vector<double> F(10, 0.0);

    auto u = solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), 10);
    for (int i = 0; i < 10; ++i)
        EXPECT_NEAR(u[i], 0.0, 1e-15) << "Zero RHS: component " << i << " non-zero";
}

// ── Test 6: iteration count and error diagnostics are populated ───────────────

TEST_F(CpuPCGTest, DiagnosticsPopulatedAfterSolve) {
    auto csr = make_tridiagonal(50, 4.0, -1.0);
    std::vector<double> F(50, 1.0);

    (void)solve(csr, F);

    EXPECT_GE(last_iteration_count(), 0)
        << "Iteration count should be non-negative after solve";
    EXPECT_LT(last_estimated_error(), 1e-7)
        << "Estimated error should be below tolerance after convergence";
}

// ── Test 7: well-conditioned n=1000 tridiagonal ───────────────────────────────
// κ ≈ (2n/π)² ≈ 4e5 for tridiag(−1,2,−1) n=1000.
// AMG removes the low frequencies that dominate the condition number.
// Test that PCG converges to tolerance 1e-8 within a reasonable iteration count.

TEST_F(CpuPCGTest, LargeTridiagonalConvergesWithinBound) {
    const int n = 1000;
    auto csr = make_tridiagonal(n, 2.0, -1.0);
    std::vector<double> F(n, 1.0);

    // Use default tolerance (1e-8); the test will throw if PCG doesn't converge.
    auto u = solve(csr, F);  // NOLINT — return value intentionally checked below

    ASSERT_EQ(static_cast<int>(u.size()), n);

    // Verify by comparing with direct solver on a subset of components.
    LdltSolverBackend direct;
    auto u_direct = direct.solve(csr, F);
    for (int i = 0; i < n; i += 100) {
        const double tol = std::max(1e-6, std::abs(u_direct[i]) * 1e-10);
        EXPECT_NEAR(u[i], u_direct[i], tol)
            << "n=1000 tridiagonal: component " << i << " disagrees";
    }

    // Multigrid on a 1D Laplacian: nearly independent of n, where a preconditioner
    // that only removes high frequencies would need on the order of sqrt(n).
    EXPECT_LT(last_iteration_count(), 20)
        << "PCG iteration count unexpectedly high for a 1D Laplacian with AMG";
}

// ── Test 8: a singular matrix is reported, not returned ──────────────────────
// An unconstrained 1D bar, tridiag(-1, 2, -1) with free ends, has the rigid body
// mode in its nullspace: no solution exists for a load with net force, and PCG must
// not hand back whatever its last iterate was.

TEST(CpuPCGSolver, SingularSystemWithNetLoadWritesItsIterateUnderAFatalMessage) {
    const int n = 2000;
    SparseMatrixBuilder b(n);
    for (int i = 0; i < n; ++i) {
        b.add(i, i, i == 0 || i == n - 1 ? 1.0 : 2.0);
        if (i > 0) { b.add(i, i - 1, -1.0); b.add(i - 1, i, -1.0); }
    }
    CpuPCGSolverBackend pcg(1e-8, 200);
    const auto csr = b.build_csr();
    const std::vector<double> F(n, 1.0);
    const auto u = pcg.solve(csr, F);
    ASSERT_TRUE(pcg.last_refinement().has_value());
    EXPECT_FALSE(pcg.last_refinement()->converged);
    const auto fatals = pcg.notices().take_fatals();
    ASSERT_EQ(fatals.size(), 1u);
    EXPECT_NE(fatals[0].find("did not converge"), std::string::npos) << fatals[0];
    // A net load on a free system has no solution, so the iterate cannot carry it.
    EXPECT_TRUE(indicates_mechanism(*pcg.last_refinement(), /*refined=*/true));
    EXPECT_EQ(u.size(), F.size());
}

// ── Test 9: a real structure, with the modes the analysis supplies ───────────
// The modes are what make a structure converge in a few dozen iterations, and the
// analysis supplies them for every system this backend solves.

TEST(CpuPCGSolver, UsesTheNearNullspaceTheAnalysisSupplies) {
    const auto record = test::assemble(test::shell_sheet(24));
    ASSERT_TRUE(record.modes);
    CpuPCGSolverBackend pcg;
    EXPECT_TRUE(pcg.uses_near_nullspace(record.k.n));
    EXPECT_TRUE(pcg.uses_near_nullspace(1)) << "no size threshold: a small system gets its modes too";

    const auto without = pcg.solve(record.k, record.f);
    const int constant = pcg.last_iteration_count();
    pcg.set_near_nullspace(record.modes);
    const auto with = pcg.solve(record.k, record.f);
    const int modes = pcg.last_iteration_count();
    EXPECT_LT(modes, constant);
    for (std::size_t i = 0; i < with.size(); ++i) {
        const double tolerance = 1e-6 * std::max(1e-3, std::abs(record.u[i]));
        EXPECT_NEAR(with[i], record.u[i], tolerance) << i;
        EXPECT_NEAR(without[i], record.u[i], tolerance) << i;
    }
}

// ── Test 10: name() returns a non-empty string with 'PCG' ─────────────────────

TEST(CpuPCGTest_Static, NameContainsPCG) {
    CpuPCGSolverBackend pcg;
    EXPECT_FALSE(pcg.name().empty());
    EXPECT_NE(pcg.name().find("PCG"), std::string_view::npos)
        << "Backend name should contain 'PCG'";
}

// ══════════════════════════════════════════════════════════════════════════════
// CudaPCGSolverBackend tests
// ══════════════════════════════════════════════════════════════════════════════

#ifndef HAVE_CUDA
TEST(CudaPCGTest, CudaNotCompiled) {
    GTEST_SKIP() << "CUDA backend not compiled — skipping CUDA PCG tests";
}
#else

class CudaPCGTest : public ::testing::Test {
protected:
    // cppcheck-suppress unusedFunction -- called by GTest framework
    void SetUp() override {
        backend_ = CudaPCGSolverBackend::try_create();
        if (!backend_.has_value())
            GTEST_SKIP() << "CUDA not available on this system — skipping CUDA PCG tests";
    }

    std::optional<CudaPCGSolverBackend> backend_;
};

class CudaMixedPCGTest : public ::testing::Test {
protected:
    // cppcheck-suppress unusedFunction -- called by GTest framework
    void SetUp() override {
        backend_ = CudaMixedPCGSolverBackend::try_create();
        if (!backend_.has_value())
            GTEST_SKIP() << "CUDA not available on this system — skipping CUDA mixed PCG tests";
    }

    std::optional<CudaMixedPCGSolverBackend> backend_;
};

// ── Test 9: 2×2 diagonal system — exact solution ─────────────────────────────

TEST_F(CudaPCGTest, DiagonalSystemExactSolution) {
    SparseMatrixBuilder b(2);
    b.add(0, 0, 4.0);
    b.add(1, 1, 9.0);
    auto csr = b.build_csr();
    std::vector<double> F = {8.0, 27.0};

    auto u = backend_->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), 2);
    EXPECT_NEAR(u[0], 2.0, 1e-7);
    EXPECT_NEAR(u[1], 3.0, 1e-7);
    EXPECT_LE(backend_->last_iteration_count(), 1)
        << "CUDA PCG should converge in at most 1 step on a diagonal system with IC0";
}

// ── Test 10: tridiagonal — agrees with Eigen direct solver ───────────────────

TEST_F(CudaPCGTest, TridiagonalAgreesWithEigen) {
    const int n = 5;
    auto csr = make_tridiagonal(n, 3.0, -1.0);
    std::vector<double> F(n, 1.0);

    auto u_cuda = backend_->solve(csr, F);

    LdltSolverBackend eigen;
    auto u_eigen = eigen.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_cuda.size()), n);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u_cuda[i], u_eigen[i], 1e-6)
            << "CUDA PCG component " << i << " disagrees with Eigen";
}

TEST_F(CudaPCGTest, LowerTriangularTridiagonalAgreesWithEigen) {
    const int n = 8;
    auto csr = make_tridiagonal(n, 3.0, -1.0, true);
    ASSERT_TRUE(csr.stores_lower_triangle_only());
    std::vector<double> F(n, 1.0);

    auto u_cuda = backend_->solve(csr, F);

    LdltSolverBackend eigen;
    auto u_eigen = eigen.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_cuda.size()), n);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u_cuda[i], u_eigen[i], 1e-6)
            << "CUDA PCG lower-triangular component " << i
            << " disagrees with Eigen";
}

// ── Test 11: larger n=200 tridiagonal — residual below tolerance ──────────────

TEST_F(CudaPCGTest, LargerTridiagonalConverges) {
    const int n = 200;
    auto csr = make_tridiagonal(n, 2.0, -1.0);
    std::vector<double> F(n, 1.0);

    auto u_cuda = backend_->solve(csr, F);

    LdltSolverBackend eigen;
    auto u_eigen = eigen.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_cuda.size()), n);
    for (int i = 0; i < n; i += 20)
        EXPECT_NEAR(u_cuda[i], u_eigen[i], 1e-5)
            << "n=200 tridiagonal: CUDA PCG component " << i << " disagrees";

    EXPECT_LT(backend_->last_relative_residual(), 1e-7);
}

// ── Test 12: FEM-scale stiffness values ───────────────────────────────────────

TEST_F(CudaPCGTest, StiffDiagonalDoesNotConvergeFalsely) {
    const double K_diag = 1.3e9;
    const double F_val  = 333.0;
    const int    n      = 100;
    auto csr = make_diagonal(n, K_diag);
    std::vector<double> F(n, F_val);

    auto u = backend_->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), n);
    const double expected = F_val / K_diag;
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(u[i], expected, expected * 1e-5)
            << "Stiff diagonal: CUDA PCG component " << i << " wrong";
}

// ── Test 13: zero RHS ─────────────────────────────────────────────────────────

TEST_F(CudaPCGTest, ZeroRhsReturnZeroDisplacement) {
    auto csr = make_tridiagonal(10, 3.0, -1.0);
    std::vector<double> F(10, 0.0);

    auto u = backend_->solve(csr, F);

    ASSERT_EQ(static_cast<int>(u.size()), 10);
    for (int i = 0; i < 10; ++i)
        EXPECT_NEAR(u[i], 0.0, 1e-15) << "Zero RHS: component " << i << " non-zero";
    EXPECT_EQ(backend_->last_iteration_count(), 0);
}

// ── Test 14: n=500 banded SPD agrees with Eigen ───────────────────────────────

TEST_F(CudaPCGTest, LargeBandedSpdAgreesWithEigen) {
    const int n = 500;
    SparseMatrixBuilder b(n);
    for (int i = 0; i < n; ++i) {
        b.add(i, i, static_cast<double>(n + 1));
        if (i > 0)     { b.add(i, i - 1, -1.0); b.add(i - 1, i, -1.0); }
        if (i > 1)     { b.add(i, i - 2, -0.5); b.add(i - 2, i, -0.5); }
    }
    auto csr = b.build_csr();
    std::vector<double> F(n);
    for (int i = 0; i < n; ++i) F[i] = static_cast<double>((i % 7) + 1);

    auto u_cuda = backend_->solve(csr, F);

    LdltSolverBackend eigen;
    auto u_eigen = eigen.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_cuda.size()), n);
    for (int i = 0; i < n; i += 50)
        EXPECT_NEAR(u_cuda[i], u_eigen[i], 1e-5)
            << "n=500 banded: CUDA PCG component " << i << " disagrees with Eigen";
}

// ── Test 15: name() and device_name() are non-empty ──────────────────────────

TEST_F(CudaPCGTest, NameAndDeviceNameAreNonEmpty) {
    EXPECT_FALSE(backend_->name().empty());
    EXPECT_NE(backend_->name().find("CUDA"), std::string_view::npos)
        << "Backend name should contain 'CUDA'";
    EXPECT_FALSE(backend_->device_name().empty())
        << "device_name() should return the GPU name";
}

TEST_F(CudaMixedPCGTest, TridiagonalAgreesWithEigenAfterRefinement) {
    const int n = 200;
    auto csr = make_tridiagonal(n, 2.0, -1.0);
    std::vector<double> F(n, 1.0);

    auto u_cuda = backend_->solve(csr, F);

    LdltSolverBackend eigen;
    auto u_eigen = eigen.solve(csr, F);

    ASSERT_EQ(static_cast<int>(u_cuda.size()), n);
    for (int i = 0; i < n; i += 20)
        EXPECT_NEAR(u_cuda[i], u_eigen[i], 5e-3)
            << "n=200 tridiagonal mixed PCG component " << i
            << " disagrees with Eigen";

    EXPECT_LT(backend_->last_estimated_error(), 1e-4);
}


// ── Ill-conditioned systems: the error, not the residual, is controlled ──────
// cond(H_8) ~ 1.5e10: a relative-residual stopping test is not a statement
// about accuracy here.  The refined CUDA PCG solution is accurate to its
// target relative error (1e-8) and reports the refinement estimate.

TEST_F(CudaPCGTest, IllConditionedSystemIsAccurateToTheTargetError) {
    const auto [K, F] = test::scaled_hilbert(8);
    const auto u = backend_->solve(K, F);
    const auto rep = backend_->last_refinement();
    ASSERT_TRUE(rep.has_value());
    EXPECT_TRUE(rep->converged);
    EXPECT_LE(backend_->last_estimated_error(), 1e-8);
    EXPECT_LT(test::max_error_from_one(u), 1e-7);
}

// Kershaw's matrix (J. Comput. Phys. 26, 1978) is SPD (eigenvalues 0.17 and
// 5.83, each twice) but IC(0) meets a negative pivot (-5) in its last row.
// The preconditioner is retried on K + shift diag(K) (here shift 0.256
// succeeds) instead of falling back to the nonsymmetric ILU0, as shell and
// thin-solid stiffness matrices need.
TEST_F(CudaPCGTest, IncompleteCholeskyBreakdownIsHandledWithADiagonalShift) {
    SparseMatrixBuilder b(4);
    const double A[4][4] = {{3, -2, 0, 2}, {-2, 3, -2, 0}, {0, -2, 3, -2}, {2, 0, -2, 3}};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (A[i][j] != 0.0) b.add(i, j, A[i][j]);
    const auto K = b.build_csr();
    // F = K (1, 2, 3, 4)
    const std::vector<double> F{7.0, -2.0, -3.0, 8.0};
    const auto u = backend_->solve(K, F);
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(u[static_cast<std::size_t>(i)], i + 1.0, 1e-10);
    EXPECT_TRUE(backend_->last_refinement()->converged);
}

// fp32 inner PCG on H_5 (cond 4.8e5): each correction contracts the error by
// about cond * 6e-8, so refinement still reaches the 1e-6 default target.
TEST_F(CudaMixedPCGTest, Fp32InnerSolvesAreRefinedToTheTargetError) {
    const auto [K, F] = test::scaled_hilbert(5);
    const auto u = backend_->solve(K, F);
    const auto rep = backend_->last_refinement();
    ASSERT_TRUE(rep.has_value());
    EXPECT_TRUE(rep->converged);
    EXPECT_GT(rep->corrections, 0);
    EXPECT_LT(test::max_error_from_one(u), 1e-5);
}

// A matrix too ill-conditioned for fp32 corrections (H_9, cond 4.9e11) must
// be reported as a failure, not returned as a converged solution.
TEST_F(CudaMixedPCGTest, HopelesslyIllConditionedSystemIsReportedFatal) {
    const auto [K, F] = test::scaled_hilbert(9);
    const auto u = backend_->solve(K, F);
    EXPECT_EQ(u.size(), F.size());
    ASSERT_TRUE(backend_->last_refinement().has_value());
    EXPECT_FALSE(backend_->last_refinement()->converged);
    const auto fatals = backend_->notices().take_fatals();
    ASSERT_EQ(fatals.size(), 1u);
    EXPECT_NE(fatals[0].find("did not converge"), std::string::npos) << fatals[0];
}

#endif // HAVE_CUDA
