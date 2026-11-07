// tests/unit/test_ldlt_backend.cpp
// LdltSolverBackend, the default CPU direct backend: its factorization against
// solutions, pivots and inertia known in closed form. The library's own tests are in
// src/solver/ldlt/tests.

#include "solver/solver_backend.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <cmath>
#include <string>

using namespace vibestran;

namespace {

using CsrData = SparseMatrixBuilder::CsrData;

CsrData to_csr(const Eigen::MatrixXd &A, bool full = false) {
    CsrData csr;
    csr.n = static_cast<int>(A.rows());
    csr.symmetry = full ? SymmetryStorage::Full : SymmetryStorage::Lower;
    csr.row_ptr.push_back(0);
    for (int i = 0; i < csr.n; ++i) {
        for (int j = 0; j <= (full ? csr.n - 1 : i); ++j)
            if (A(i, j) != 0.0) {
                csr.col_ind.push_back(j);
                csr.values.push_back(A(i, j));
            }
        csr.row_ptr.push_back(static_cast<int>(csr.col_ind.size()));
    }
    csr.nnz = static_cast<int>(csr.values.size());
    return csr;
}

/// A 2D Laplacian with a small shift: fill, supernodes and a tree to order.
Eigen::MatrixXd grid_laplacian(int side, double shift) {
    const int n = side * side;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
            const int i = x + side * y;
            A(i, i) = 4.0 + shift;
            if (x > 0)
                A(i, i - 1) = A(i - 1, i) = -1.0;
            if (y > 0)
                A(i, i - side) = A(i - side, i) = -1.0;
        }
    return A;
}

Eigen::VectorXd known_solution(int n) {
    Eigen::VectorXd x(n);
    for (int i = 0; i < n; ++i)
        x(i) = std::sin(0.37 * (i + 1));
    return x;
}

std::unique_ptr<LinearFactor> take(Result<std::unique_ptr<LinearFactor>> made) {
    if (auto *e = std::get_if<Error>(&made)) {
        ADD_FAILURE() << e->message;
        return nullptr;
    }
    return std::get<std::unique_ptr<LinearFactor>>(std::move(made));
}

Eigen::VectorXd solved(LinearFactor &factor, const Eigen::VectorXd &b) {
    auto x = factor.solve(std::span<const double>(b.data(), static_cast<std::size_t>(b.size())));
    if (auto *e = std::get_if<Error>(&x)) {
        ADD_FAILURE() << e->message;
        return Eigen::VectorXd::Zero(b.size());
    }
    const auto &v = std::get<std::vector<double>>(x);
    return Eigen::Map<const Eigen::VectorXd>(v.data(), b.size());
}

int negative_eigenvalues(const Eigen::MatrixXd &A) {
    const Eigen::VectorXd values = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(A).eigenvalues();
    return static_cast<int>((values.array() < 0.0).count());
}

} // namespace

TEST(LdltBackend, LowerAndFullStorageGiveTheSameSolution) {
    const Eigen::MatrixXd A = grid_laplacian(30, 0.01);
    const Eigen::VectorXd x = known_solution(900);
    const Eigen::VectorXd b = A * x;
    LdltSolverBackend backend;
    auto lower = take(backend.factor(to_csr(A)));
    auto full = take(backend.factor(to_csr(A, true)));
    ASSERT_TRUE(lower && full);
    EXPECT_EQ(lower->method(), "ldlt");
    EXPECT_LT((solved(*lower, b) - x).norm() / x.norm(), 1e-11);
    EXPECT_LT((solved(*full, b) - x).norm() / x.norm(), 1e-11);
    ASSERT_TRUE(lower->pivot_report().has_value());
    EXPECT_EQ(lower->pivot_report()->negative_pivots, 0);
    EXPECT_TRUE(lower->pivot_report()->held_rows.empty());
}

TEST(LdltBackend, RefactorKeepsTheAnalysisUntilThePatternChanges) {
    Eigen::MatrixXd A = grid_laplacian(12, 0.5);
    const Eigen::VectorXd x = known_solution(144);
    LdltSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    EXPECT_EQ(factor->stats().symbolic_analyses, 1);
    EXPECT_EQ(factor->stats().numeric_factorizations, 1);

    const Eigen::MatrixXd doubled = 2.0 * A;
    ASSERT_FALSE(factor->refactor(to_csr(doubled)).has_value());
    EXPECT_EQ(factor->stats().symbolic_analyses, 1);
    EXPECT_EQ(factor->stats().numeric_factorizations, 2);
    EXPECT_LT((solved(*factor, doubled * x) - x).norm() / x.norm(), 1e-12);

    A(100, 3) = A(3, 100) = -0.5;
    ASSERT_FALSE(factor->refactor(to_csr(A)).has_value());
    EXPECT_EQ(factor->stats().symbolic_analyses, 2);
    EXPECT_LT((solved(*factor, A * x) - x).norm() / x.norm(), 1e-12);

    const auto wrong_size = factor->refactor(to_csr(grid_laplacian(3, 0.5)));
    ASSERT_TRUE(wrong_size.has_value());
    EXPECT_NE(wrong_size->message.find("144-equation"), std::string::npos) << wrong_size->message;
    EXPECT_TRUE(std::holds_alternative<Error>(factor->solve(std::vector<double>(144, 1.0))));
}

TEST(LdltBackend, SeveralRightHandSidesMatchSolvingThemOneAtATime) {
    const Eigen::MatrixXd A = grid_laplacian(20, 0.1);
    Eigen::MatrixXd X(400, 3);
    for (int k = 0; k < 3; ++k)
        X.col(k) = known_solution(400).array() * (k + 1.0) + k;
    const Eigen::MatrixXd B = A * X;
    LdltSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    auto result = factor->solve(std::span<const double>(B.data(), 1200), 3);
    ASSERT_TRUE(std::holds_alternative<std::vector<double>>(result));
    const auto &values = std::get<std::vector<double>>(result);
    EXPECT_LT((Eigen::Map<const Eigen::MatrixXd>(values.data(), 400, 3) - X).norm() / X.norm(), 1e-12);
    EXPECT_EQ(factor->stats().solves, 3);
}

TEST(LdltBackend, AFreeSpringIsHeldAndASelfEquilibratedLoadIsStillCarried) {
    // Rows 1 and 3 are a spring with nothing holding it, so K is singular with one null
    // mode. Between them, a load summing to zero has solutions; holding one row at zero
    // picks the one with the other at f / k.
    const double k = 250.0, f = 10.0;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4, 4);
    A(0, 0) = 3.0;
    A(2, 2) = 5.0;
    A(1, 1) = A(3, 3) = k;
    A(1, 3) = A(3, 1) = -k;
    Eigen::VectorXd b(4);
    b << 6.0, f, 10.0, -f;
    LdltSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    const auto report = factor->pivot_report();
    ASSERT_TRUE(report.has_value());
    ASSERT_EQ(report->held_rows.size(), 1u);
    const int held = report->held_rows[0];
    ASSERT_TRUE(held == 1 || held == 3) << held;
    const Eigen::VectorXd x = solved(*factor, b);
    EXPECT_DOUBLE_EQ(x(held), 0.0);
    EXPECT_DOUBLE_EQ(x(0), 2.0);
    EXPECT_DOUBLE_EQ(x(2), 2.0);
    EXPECT_NEAR(x(1) - x(3), f / k, 1e-14);
    EXPECT_LT((A * x - b).norm(), 1e-12);
}

TEST(LdltBackend, AnEmptyRowIsHeldAndALoadOnItIsLeftInTheResidual) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
    A(0, 0) = A(2, 2) = 2.0;
    const Eigen::VectorXd b = Eigen::Vector3d(2.0, 7.0, 4.0);
    LdltSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    ASSERT_EQ(factor->pivot_report()->held_rows, std::vector<int>{1});
    const Eigen::VectorXd x = solved(*factor, b);
    EXPECT_DOUBLE_EQ(x(0), 1.0);
    EXPECT_DOUBLE_EQ(x(1), 0.0);
    EXPECT_DOUBLE_EQ(x(2), 2.0);
}

TEST(LdltBackend, RatioOfTwoSpringsInSeriesIsTheirStiffnessRatio) {
    // Grounded soft spring 1 in series with stiff spring s. In either elimination order the
    // later pivot's K_ii / D_i is exactly s + 1.
    const double s = 1e9;
    Eigen::MatrixXd A(2, 2);
    A << s, -s, -s, s + 1.0;
    LdltSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    const auto report = factor->pivot_report();
    ASSERT_TRUE(report.has_value());
    EXPECT_NEAR(report->max_ratio / (s + 1.0), 1.0, 1e-6);
    ASSERT_TRUE(report->max_ratio_row == 0 || report->max_ratio_row == 1);
    EXPECT_DOUBLE_EQ(report->max_ratio_diagonal, A(report->max_ratio_row, report->max_ratio_row));
    EXPECT_NEAR(std::abs(report->max_ratio_diagonal / report->max_ratio_pivot) / report->max_ratio, 1.0, 1e-12);
}

TEST(LdltBackend, AnIndefiniteMatrixIsRefactoredWithPivotingAndItsInertiaIsExact) {
    const Eigen::MatrixXd A = grid_laplacian(15, -1.5);
    const int negative = negative_eigenvalues(A);
    ASSERT_GT(negative, 0);
    const Eigen::VectorXd x = known_solution(225);
    LdltSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    EXPECT_EQ(factor->method(), "ldlt-threshold");
    EXPECT_EQ(factor->stats().numeric_factorizations, 2);
    EXPECT_EQ(factor->stats().symbolic_analyses, 1);
    EXPECT_EQ(factor->pivot_report()->negative_pivots, negative);
    EXPECT_LT((solved(*factor, A * x) - x).norm() / x.norm(), 1e-10);
    const auto notices = backend.notices().take();
    ASSERT_EQ(notices.size(), 1u);
    EXPECT_NE(notices[0].find("threshold pivoting"), std::string::npos) << notices[0];
}

TEST(LdltBackend, AZeroDiagonalWithCouplingIsIndefiniteEvenThoughNoPivotComesOutNegative) {
    // [0 B; B^T 0] has as many negative eigenvalues as positive. Without pivoting every first
    // pivot is exactly zero, so only the zero diagonal can show that holding them is wrong.
    const int m = 8;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * m, 2 * m);
    for (int i = 0; i < m; ++i) {
        A(i, m + i) = A(m + i, i) = 3.0 + i;
        if (i > 0)
            A(i, m + i - 1) = A(m + i - 1, i) = 1.0;
    }
    const Eigen::VectorXd x = known_solution(2 * m);
    LdltSolverBackend backend;
    auto factor = take(backend.factor(to_csr(A)));
    ASSERT_TRUE(factor);
    EXPECT_EQ(factor->method(), "ldlt-threshold");
    EXPECT_EQ(factor->pivot_report()->negative_pivots, m);
    EXPECT_TRUE(factor->pivot_report()->held_rows.empty());
    EXPECT_LT((solved(*factor, A * x) - x).norm() / x.norm(), 1e-11);
}

TEST(LdltBackend, RequireSpdRefusesIndefiniteAndSingularMatrices) {
    LdltSolverBackend backend;
    Eigen::MatrixXd singular = Eigen::MatrixXd::Zero(3, 3);
    singular(0, 0) = 1.0;
    singular(1, 1) = singular(2, 2) = 2.0;
    singular(1, 2) = singular(2, 1) = -2.0;
    for (const Eigen::MatrixXd &A : {Eigen::MatrixXd(grid_laplacian(10, -1.5)), singular}) {
        auto refused = backend.factor(to_csr(A), {.require_spd = true});
        ASSERT_TRUE(std::holds_alternative<Error>(refused));
        EXPECT_NE(std::get<Error>(refused).message.find("positive definite"), std::string::npos)
            << std::get<Error>(refused).message;
    }
    EXPECT_TRUE(backend.notices().take().empty());
}

TEST(LdltBackend, OrderingFollowsTheFactorOptionsAndDoesNotChangeTheAnswer) {
    const Eigen::MatrixXd A = grid_laplacian(25, 0.2);
    const Eigen::VectorXd x = known_solution(625);
    const Eigen::VectorXd b = A * x;
    LdltSolverBackend backend;
    auto metis = take(backend.factor(to_csr(A)));
    auto amd = take(backend.factor(to_csr(A), {.minimum_degree = true}));
    auto dense = take(backend.factor(to_csr(A), {.dense_block = true}));
    ASSERT_TRUE(metis && amd && dense);
    EXPECT_EQ(metis->ordering(), "metis");
    EXPECT_EQ(amd->ordering(), "amd");
    EXPECT_EQ(dense->ordering(), "amd");
    for (LinearFactor *factor : {metis.get(), amd.get(), dense.get()})
        EXPECT_LT((solved(*factor, b) - x).norm() / x.norm(), 1e-12) << factor->ordering();
}
