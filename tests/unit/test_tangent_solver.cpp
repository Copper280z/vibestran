// tests/unit/test_tangent_solver.cpp
// TangentSolver: conjugate gradients on a new tangent preconditioned by the
// factor of an earlier one. With K2 = K1 + a rank-k change, the preconditioned
// operator K1^-1 K2 is the identity plus rank k, so CG converges in at most
// k + 1 iterations; that is the property that makes a reused factor cheap
// when a nonlinear material changes in a few cells. A tangent too far from
// the factor, or indefinite, must still be solved exactly, by refactoring.

#include <gtest/gtest.h>
#include "solver/tangent_solver.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace vibestran;

namespace {

using CsrData = SparseMatrixBuilder::CsrData;

CsrData lower_csr(const Eigen::MatrixXd &A) {
    CsrData csr;
    csr.n = static_cast<int>(A.rows());
    csr.symmetry = SymmetryStorage::Lower;
    csr.row_ptr.push_back(0);
    for (int i = 0; i < csr.n; ++i) {
        for (int j = 0; j <= i; ++j)
            if (A(i, j) != 0.0) {
                csr.col_ind.push_back(j);
                csr.values.push_back(A(i, j));
            }
        csr.row_ptr.push_back(static_cast<int>(csr.col_ind.size()));
    }
    csr.nnz = static_cast<int>(csr.values.size());
    return csr;
}

/// A 5-point Laplacian on an nx-by-nx grid with a coefficient per cell
/// (row-major cells, (nx-1)^2 of them) and a small definite shift: the
/// tangent of a scalar potential problem whose permeability varies by cell.
/// Changing one cell's coefficient is a rank-one change of the matrix.
Eigen::MatrixXd laplacian(int nx, const Eigen::VectorXd &coefficient) {
    const int n = nx * nx;
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n) * 1e-3;
    for (int cy = 0; cy + 1 < nx; ++cy)
        for (int cx = 0; cx + 1 < nx; ++cx) {
            const double c = coefficient[cy * (nx - 1) + cx];
            // c (e_i - e_j)(e_i - e_j)^T for each of the cell's 4 edges: the
            // Laplacian of a 4-cycle, rank 3.
            const int corner[4] = {cy * nx + cx, cy * nx + cx + 1,
                                   (cy + 1) * nx + cx + 1, (cy + 1) * nx + cx};
            for (int k = 0; k < 4; ++k) {
                const int i = corner[k], j = corner[(k + 1) % 4];
                A(i, i) += c;
                A(j, j) += c;
                A(i, j) -= c;
                A(j, i) -= c;
            }
        }
    return A;
}

std::vector<double> solved(Result<std::vector<double>> x) {
    if (auto *e = std::get_if<Error>(&x)) {
        ADD_FAILURE() << e->message;
        return {};
    }
    return std::get<std::vector<double>>(std::move(x));
}

double relative_residual(const Eigen::MatrixXd &A, const std::vector<double> &x,
                         const Eigen::VectorXd &b) {
    const Eigen::Map<const Eigen::VectorXd> X(x.data(), A.rows());
    return (b - A * X).norm() / b.norm();
}

Eigen::VectorXd load(int n) {
    Eigen::VectorXd b(n);
    for (int i = 0; i < n; ++i)
        b[i] = std::sin(0.37 * (i + 1)) + 0.2;
    return b;
}

constexpr int kGrid = 16;
constexpr int kCells = (kGrid - 1) * (kGrid - 1);

} // namespace

TEST(TangentSolver, CellChangesConvergeWithinTheirRankPlusOne) {
    // A cell's term spans the 3 nonconstant modes of its 4 corners, so
    // changing k cells changes the matrix by rank at most 3k: CG with the old
    // factor converges in at most 3k + 1 iterations in exact arithmetic,
    // whatever the size of the change.
    const Eigen::VectorXd mu = Eigen::VectorXd::Constant(kCells, 1.0);
    Eigen::VectorXd saturated = mu;
    saturated[20] = 1e-3;  // a cell driven into saturation
    saturated[100] = 50.0; // and one the other way
    const Eigen::MatrixXd K1 = laplacian(kGrid, mu), K2 = laplacian(kGrid, saturated);
    const Eigen::VectorXd b = load(K1.rows());

    EigenSolverBackend backend;
    TangentSolverOptions options;
    options.factor.require_spd = true;
    options.reuse_factor = true;
    TangentSolver solver(backend, options);
    ASSERT_FALSE(solver.update(lower_csr(K1)));
    ASSERT_FALSE(solver.update(lower_csr(K2)));
    EXPECT_FALSE(solver.factor_is_current());

    const auto x = solved(solver.solve({b.data(), static_cast<size_t>(b.size())}, 1, 1e-12));
    ASSERT_EQ(x.size(), static_cast<size_t>(b.size()));
    EXPECT_LE(relative_residual(K2, x, b), 1e-12);
    const auto &stats = solver.stats();
    EXPECT_EQ(stats.numeric_factorizations, 1);
    EXPECT_EQ(stats.iterative_solves, 1);
    EXPECT_EQ(stats.failed_iterative, 0);
    EXPECT_LE(stats.iterations, 3 * 2 + 1);
    EXPECT_GE(stats.iterations, 2); // the change is not trivial
    // The same answer as a direct solve of K2.
    const Eigen::VectorXd exact = K2.llt().solve(b);
    EXPECT_LE((Eigen::Map<const Eigen::VectorXd>(x.data(), b.size()) - exact).norm(),
              1e-9 * exact.norm());
}

TEST(TangentSolver, DistantTangentRefactorsAndSolvesExactly) {
    // Every cell's coefficient changes by up to two decades: K1^-1 K2 has
    // a spread spectrum and CG cannot converge in 3 iterations, so the
    // solve refactors with K2 and is direct from then on.
    const Eigen::VectorXd mu = Eigen::VectorXd::Constant(kCells, 1.0);
    Eigen::VectorXd varied(kCells);
    for (int c = 0; c < kCells; ++c)
        varied[c] = std::pow(10.0, 2.0 * std::sin(1.3 * c));
    const Eigen::MatrixXd K1 = laplacian(kGrid, mu), K2 = laplacian(kGrid, varied);
    const Eigen::VectorXd b = load(K1.rows());

    EigenSolverBackend backend;
    TangentSolverOptions options;
    options.factor.require_spd = true;
    options.reuse_factor = true;
    options.max_iterations = 3;
    TangentSolver solver(backend, options);
    ASSERT_FALSE(solver.update(lower_csr(K1)));
    ASSERT_FALSE(solver.update(lower_csr(K2)));
    const std::span<const double> rhs{b.data(), static_cast<size_t>(b.size())};
    const auto x = solved(solver.solve(rhs, 1, 1e-10));
    EXPECT_LE(relative_residual(K2, x, b), 1e-10); // direct: round-off
    EXPECT_TRUE(solver.factor_is_current());
    auto stats = solver.stats();
    EXPECT_EQ(stats.failed_iterative, 1);
    EXPECT_EQ(stats.iterations, 3);
    EXPECT_EQ(stats.numeric_factorizations, 2);
    EXPECT_EQ(stats.symbolic_analyses, 1); // the pattern is K1's
    EXPECT_EQ(stats.iterative_solves, 0);

    // The next solve uses K2's own factor: no more iterations.
    (void)solved(solver.solve(rhs, 1, 1e-10));
    EXPECT_EQ(solver.stats().iterations, 3);
    EXPECT_EQ(solver.stats().numeric_factorizations, 2);
}

TEST(TangentSolver, IndefiniteTangentFallsBackToItsOwnFactorization) {
    // Past a limit point (buckling) a structural tangent is indefinite. CG
    // does not apply; the solver must notice and factor the tangent itself,
    // here with the backend's LDLT fallback (require_spd off).
    const Eigen::VectorXd mu = Eigen::VectorXd::Constant(kCells, 1.0);
    const Eigen::MatrixXd K1 = laplacian(kGrid, mu);
    const Eigen::MatrixXd K2 =
        K1 - 0.5 * Eigen::MatrixXd::Identity(K1.rows(), K1.cols());
    ASSERT_LT(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(K2).eigenvalues()[0], 0);
    const Eigen::VectorXd b = load(K1.rows());

    EigenSolverBackend backend;
    TangentSolverOptions options;
    options.reuse_factor = true;
    TangentSolver solver(backend, options);
    ASSERT_FALSE(solver.update(lower_csr(K1)));
    ASSERT_FALSE(solver.update(lower_csr(K2)));
    const auto x = solved(solver.solve({b.data(), static_cast<size_t>(b.size())}, 1, 1e-10));
    EXPECT_LE(relative_residual(K2, x, b), 1e-10);
    EXPECT_EQ(solver.stats().failed_iterative, 1);
    EXPECT_EQ(solver.stats().numeric_factorizations, 2);
}

TEST(TangentSolver, ExactSolvesAndNewPatternsFactorTheCurrentTangent) {
    const Eigen::VectorXd mu = Eigen::VectorXd::Constant(kCells, 1.0);
    Eigen::VectorXd changed = mu;
    changed[7] = 3.0;
    const Eigen::MatrixXd K1 = laplacian(kGrid, mu), K2 = laplacian(kGrid, changed);
    // Two right-hand sides, the second zero.
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(K1.rows(), 2);
    B.col(0) = load(K1.rows());
    const std::span<const double> rhs{B.data(), static_cast<size_t>(B.size())};

    EigenSolverBackend backend;
    TangentSolverOptions options;
    options.factor.require_spd = true;
    options.reuse_factor = true;
    TangentSolver solver(backend, options);
    ASSERT_FALSE(solver.update(lower_csr(K1)));
    ASSERT_FALSE(solver.update(lower_csr(K2)));
    // Tolerance 0 asks for a direct solve: K2 is factored first.
    auto X = solved(solver.solve(rhs, 2, 0));
    ASSERT_EQ(X.size(), static_cast<size_t>(B.size()));
    EXPECT_LE(relative_residual(K2, X, B.col(0)), 1e-10); // direct: round-off
    EXPECT_EQ(solver.stats().numeric_factorizations, 2);
    EXPECT_EQ(solver.stats().iterations, 0);

    // A third tangent by CG; the zero column needs no iterations and is zero.
    ASSERT_FALSE(solver.update(lower_csr(K1)));
    X = solved(solver.solve(rhs, 2, 1e-12));
    EXPECT_LE(relative_residual(K1, X, B.col(0)), 1e-12);
    for (Eigen::Index i = K1.rows(); i < B.size(); ++i)
        EXPECT_EQ(X[static_cast<size_t>(i)], 0.0);
    EXPECT_EQ(solver.stats().iterative_solves, 2);
    EXPECT_EQ(solver.stats().numeric_factorizations, 2);

    // A tangent with one more coupling has a new pattern: factored now, with
    // a new analysis, since the old factor's pattern cannot be refactored.
    Eigen::MatrixXd K3 = K1;
    K3(0, K1.rows() - 1) = K3(K1.rows() - 1, 0) = -1e-4;
    ASSERT_FALSE(solver.update(lower_csr(K3)));
    EXPECT_TRUE(solver.factor_is_current());
    EXPECT_EQ(solver.stats().symbolic_analyses, 2);
    EXPECT_EQ(solver.stats().numeric_factorizations, 3);
}

TEST(TangentSolver, WithoutReuseEveryTangentIsFactored) {
    const Eigen::VectorXd mu = Eigen::VectorXd::Constant(kCells, 1.0);
    Eigen::VectorXd changed = mu;
    changed[7] = 3.0;
    const Eigen::MatrixXd K2 = laplacian(kGrid, changed);
    const Eigen::VectorXd b = load(K2.rows());

    EigenSolverBackend backend;
    TangentSolver solver(backend, {});
    ASSERT_FALSE(solver.update(lower_csr(laplacian(kGrid, mu))));
    ASSERT_FALSE(solver.update(lower_csr(K2)));
    EXPECT_TRUE(solver.factor_is_current());
    // A tolerance does not turn the direct solve into CG.
    const auto x = solved(solver.solve({b.data(), static_cast<size_t>(b.size())}, 1, 1e-2));
    EXPECT_LE(relative_residual(K2, x, b), 1e-10); // direct: round-off
    EXPECT_EQ(solver.stats().numeric_factorizations, 2);
    EXPECT_EQ(solver.stats().symbolic_analyses, 1);
    EXPECT_EQ(solver.stats().iterations, 0);
}
