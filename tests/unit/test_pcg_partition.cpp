#include <gtest/gtest.h>
#include "pcg_partition_problem.hpp"
#include "partitioned_pcg.hpp"
#include "pcg_coloring.hpp"
#include "solver/iterative_refinement.hpp"
#include "amg_factor.hpp"
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include "amg_device_spill.hpp"

using namespace vibestran;

TEST(PcgPartitionTest, DeviceOwnedVectorsGatherEveryCrossPartitionCouplingWithinBudget) {
    const auto problem = test::pcg_partition_problem(2049);
    const auto k = problem.matrix.expanded_symmetric();
    const detail::AmgHostCsr a(std::tie(k.n, k.row_ptr, k.col_ind, k.values));
    constexpr std::size_t fixed = 123456;
    EXPECT_FALSE(detail::AmgDevicePartition::plan(a, fixed - 1, fixed));
    for (const int cut : {0, k.n / 2, k.n * 3 / 4, k.n * 9 / 10, k.n}) {
        const auto budget = detail::AmgDevicePartition::budget_for_rows(a, cut, fixed);
        const auto plan = detail::AmgDevicePartition::plan(a, budget, fixed);
        ASSERT_TRUE(plan);
        ASSERT_EQ(plan->gpu_rows, cut);
        EXPECT_LE(plan->device_bytes, budget);
        std::vector<double> input(plan->input_columns.size()), output(k.n - cut), product(k.n);
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = problem.exact[plan->input_columns[i]];
        plan->multiply_cpu(a, input, output);
        for (int row = 0; row < cut; ++row)
            for (int j = k.row_ptr[row]; j < k.row_ptr[row + 1]; ++j)
                product[row] += k.values[j] * problem.exact[k.col_ind[j]];
        std::copy(output.begin(), output.end(), product.begin() + cut);
        for (int row = 0; row < k.n; ++row) EXPECT_NEAR(product[row], problem.rhs[row], 1e-11) << cut << ':' << row;
        if (cut && cut < k.n) { EXPECT_LT(plan->input_columns.front(), cut); } // coupling into GPU-owned rows
        if (cut == k.n) { EXPECT_TRUE(plan->input_columns.empty()); }
    }
    const auto limited = detail::AmgDevicePartition::plan(a, fixed + detail::amg_csr_bytes(a, k.n), fixed, std::size_t(k.n) * 8);
    ASSERT_TRUE(limited);
    EXPECT_LT(limited->gpu_rows, k.n);
    EXPECT_LE(std::size_t(a.ptr[limited->gpu_rows]) * 8, std::size_t(k.n) * 8);
}

TEST(PcgAmgTest, EquilibrationAndRefinementPreserveInterfaceFluxAcrossDifferentUnits) {
    const auto problem = test::pcg_partition_problem(2049);
    auto matrix = problem.matrix;
    std::vector<double> units(matrix.n), rhs(matrix.n);
    for (int i = 0; i < matrix.n; ++i) {
        units[i] = i % 3 == 0 ? 1e-6 : (i % 3 == 1 ? 1e6 : 1);
        rhs[i] = problem.rhs[i] * units[i];
    }
    for (int i = 0; i < matrix.n; ++i)
        for (int j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; ++j)
            matrix.values[j] *= units[i] * units[matrix.col_ind[j]];
    for (const auto& k : {matrix, matrix.expanded_symmetric()}) {
        const auto result = detail::solve_hybrid_amg(k, rhs, 1e-10, 1000, {});
        ASSERT_TRUE(result.refinement.converged);
        EXPECT_LT(result.refinement.relative_residual, 1e-9);
        for (int i = 0; i < k.n; ++i)
            EXPECT_NEAR(result.solution[i] * units[i], problem.exact[i], 2e-8) << i;
    }
}

TEST(PcgPartitionTest, CpuSuffixAndResidentPrefixPreserveInterfaceFlux) {
    const auto problem = test::pcg_partition_problem();
    const auto k = problem.matrix.expanded_symmetric();
    constexpr std::size_t budget = 256 * 1024;
    const int cut = detail::pcg_prefix_rows(k, budget, sizeof(double));
    ASSERT_GT(cut, 0);
    ASSERT_LT(cut, k.n);
    const auto bytes = [&](int rows) {
        return std::size_t(k.n + rows) * sizeof(double) + std::size_t(rows + 1) * sizeof(int) +
            std::size_t(k.row_ptr[rows]) * (sizeof(double) + sizeof(int)) + 65536;
    };
    EXPECT_LE(bytes(cut), budget);
    EXPECT_GT(bytes(cut + 1), budget);
    // A descriptor's range is a separate constraint from total VRAM. If the
    // full input vector cannot be bound, every row must remain on the CPU.
    EXPECT_EQ(detail::pcg_prefix_rows(k, budget, sizeof(double),
                                     std::size_t(k.n) * sizeof(double) - 1), 0);
    const std::size_t range = std::size_t(k.n) * sizeof(double);
    const int limited = detail::pcg_prefix_rows(k, bytes(k.n), sizeof(double), range);
    ASSERT_GT(limited, 0);
    ASSERT_LT(limited, k.n);
    EXPECT_LE(std::size_t(k.row_ptr[limited]) * sizeof(double), range);
    EXPECT_GT(std::size_t(k.row_ptr[limited + 1]) * sizeof(double), range);
    std::vector<double> result(k.n, 0);
    // Compute prefix separately; the suffix must include couplings back into it.
    for (int row = 0; row < cut; ++row)
        for (int j = k.row_ptr[row]; j < k.row_ptr[row + 1]; ++j)
            result[row] += k.values[j] * problem.exact[k.col_ind[j]];
    detail::pcg_cpu_rows<double>(k, k.values, problem.exact, result, cut);
    for (int i = 0; i < k.n; ++i) EXPECT_NEAR(result[i], problem.rhs[i], 1e-11) << i;
}

TEST(PcgPartitionTest, SymmetricColorPermutationPreservesLowerAndFullOperators) {
    const auto problem = test::pcg_partition_problem(1536);
    for (const auto& k : {problem.matrix, problem.matrix.expanded_symmetric()}) {
        const auto colored = detail::color_pcg_matrix(k);
        ASSERT_TRUE(colored.has_value());
        std::vector<double> x(k.n);
        for (int i = 0; i < k.n; ++i) x[colored->new_index[i]] = problem.exact[i];
        const auto flux = colored->matrix.multiply(x);
        for (int i = 0; i < k.n; ++i) EXPECT_NEAR(flux[colored->new_index[i]], problem.rhs[i], 1e-11) << i;
        EXPECT_EQ(colored->matrix.nnz, k.nnz);
    }
}

namespace {
std::unique_ptr<detail::AmgSystem> host_amg_system(const SparseMatrixBuilder::CsrData& k) {
    return std::make_unique<detail::AmgSystem>(k, 1e-10, [](const detail::ScaledAmgMatrix& scaled, const detail::AmgHierarchy& h) {
        return std::make_unique<detail::HostAmgEngine>(scaled, h, detail::AmgSpmvFactory{}, 1000);
    });
}
} // namespace

TEST(PcgAmgTest, ZeroGuessVcycleIsAmgclsPreconditioner) {
    // The cycle skips the product of the first pre-smoothing step, whose
    // input is zero. Five CG iterations with it and with AMGCL's own
    // amg<smoothed_aggregation, spai0> + cg must give the same iterate up to
    // rounding: a wrong term in any level's smoothing, transfer or coarse
    // solve changes the preconditioner, and with it the iterate, at O(1).
    const auto problem = test::pcg_partition_problem(4099);
    const detail::ScaledAmgMatrix scaled(problem.matrix);
    std::vector<double> b(problem.rhs.size());
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = problem.rhs[i] * scaled.scale[i];

    const detail::AmgHierarchy h(scaled.matrix);
    ASSERT_GE(h.levels.size(), 3U) << h.describe();
    ASSERT_TRUE(h.coarse_direct());
    detail::HostAmg ours(h, nullptr);
    std::vector<double> x(b.size());
    EXPECT_EQ(ours.solve(b, 0.0, 5, x).first, 5);

    using Backend = amgcl::backend::builtin<double>;
    using Reference = amgcl::make_solver<
        amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
        amgcl::solver::cg<Backend>>;
    Reference::params prm;
    prm.precond.coarse_enough = detail::kAmgCoarseEnough;
    prm.solver.tol = 0;
    prm.solver.maxiter = 5;
    const Reference reference(*scaled.matrix, prm);
    std::vector<double> expected(b.size(), 0.0);
    const auto [iterations, residual] = reference(b, expected);
    EXPECT_EQ(iterations, 5U);

    double scale = 0, difference = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        scale = std::max(scale, std::abs(expected[i]));
        difference = std::max(difference, std::abs(x[i] - expected[i]));
    }
    EXPECT_LT(difference, 1e-10 * scale);
}

TEST(PcgAmgTest, RefactorWithTheSamePatternKeepsTheTransferOperatorsAndSolvesTheNewMatrix) {
    const auto problem = test::pcg_partition_problem(4099);
    const auto stiffer = test::stiffened(problem, 30.0);
    auto system = host_amg_system(problem.matrix);
    const auto* transfer = system->hierarchy().levels.front().P.get();
    const auto fine_nnz = system->hierarchy().levels.front().A->nnz;
    ASSERT_TRUE(system->solve(problem.matrix, problem.rhs).refinement.converged);

    ASSERT_TRUE(system->refactor(stiffer.matrix));
    EXPECT_EQ(system->hierarchy().levels.front().P.get(), transfer);
    const auto result = system->solve(stiffer.matrix, stiffer.rhs);
    ASSERT_TRUE(result.refinement.converged);
    for (int i = 0; i < stiffer.matrix.n; ++i) EXPECT_NEAR(result.solution[i], stiffer.exact[i], 2e-8) << i;
    // The rebuilt Galerkin operators are those of the new matrix: the same
    // number of iterations as a hierarchy built for it from scratch, give or
    // take a few for the aggregates chosen on the old values.
    const auto fresh = host_amg_system(stiffer.matrix)->solve(stiffer.matrix, stiffer.rhs);
    EXPECT_LE(std::abs(result.iterations - fresh.iterations), fresh.iterations / 5 + 2)
        << result.iterations << " vs " << fresh.iterations;

    // A dropped entry changes the pattern: the hierarchy is built again.
    auto pruned = stiffer;
    auto& k = pruned.matrix;
    const int row = k.n - 1, j = k.row_ptr[row];
    ASSERT_NE(k.col_ind[j], row);
    const double dropped = k.values[j];
    const int c = k.col_ind[j];
    pruned.rhs[row] -= dropped * pruned.exact[c];
    pruned.rhs[c] -= dropped * pruned.exact[row];
    k.col_ind.erase(k.col_ind.begin() + j);
    k.values.erase(k.values.begin() + j);
    for (int i = row + 1; i <= k.n; ++i) --k.row_ptr[i];
    --k.nnz;
    EXPECT_FALSE(system->refactor(k));
    EXPECT_EQ(system->hierarchy().levels.front().A->nnz, fine_nnz - 2);
    const auto rebuilt = system->solve(k, pruned.rhs);
    ASSERT_TRUE(rebuilt.refinement.converged);
    for (int i = 0; i < k.n; ++i) EXPECT_NEAR(rebuilt.solution[i], pruned.exact[i], 2e-8) << i;
}

TEST(PcgAmgTest, FactorSolvesEveryColumnAndCountsOneAnalysisPerPattern) {
    const auto problem = test::pcg_partition_problem(4099);
    const auto stiffer = test::stiffened(problem, 0.1);
    const auto n = std::size_t(problem.matrix.n);
    detail::AmgFactor factor(problem.matrix, [](const SparseMatrixBuilder::CsrData& k) { return host_amg_system(k); },
        [](const detail::AmgPcgResult& r) -> Status {
            if (r.refinement.converged) return std::nullopt;
            return Error{"not converged"};
        });
    // Columns: F, -2 F and a zero load.
    std::vector<double> B(3 * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) { B[i] = problem.rhs[i]; B[n + i] = -2 * problem.rhs[i]; }
    auto X = factor.solve(B, 3);
    ASSERT_TRUE(std::holds_alternative<std::vector<double>>(X));
    auto x = std::get<std::vector<double>>(X);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], problem.exact[i], 2e-8) << i;
        EXPECT_NEAR(x[n + i], -2 * problem.exact[i], 4e-8) << i;
        EXPECT_EQ(x[2 * n + i], 0.0) << i;
    }
    EXPECT_FALSE(factor.refactor(stiffer.matrix).has_value());
    X = factor.solve(stiffer.rhs, 1);
    ASSERT_TRUE(std::holds_alternative<std::vector<double>>(X));
    x = std::get<std::vector<double>>(X);
    for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(x[i], stiffer.exact[i], 2e-8) << i;
    const auto stats = factor.stats();
    EXPECT_EQ(stats.symbolic_analyses, 1);
    EXPECT_EQ(stats.numeric_factorizations, 2);
    EXPECT_EQ(stats.solves, 4);

    auto smaller = test::pcg_partition_problem(4098);
    EXPECT_TRUE(factor.refactor(smaller.matrix).has_value());
    EXPECT_TRUE(std::holds_alternative<Error>(factor.solve(std::vector<double>(n - 1), 1)));
}
