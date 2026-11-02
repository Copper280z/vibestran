#include <gtest/gtest.h>
#include "pcg_partition_problem.hpp"
#include "partitioned_pcg.hpp"
#include "pcg_coloring.hpp"
#include "solver/iterative_refinement.hpp"
#include "amg_hybrid.hpp"
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
