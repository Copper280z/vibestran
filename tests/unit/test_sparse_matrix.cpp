// tests/unit/test_sparse_matrix.cpp
// Tests for SparseMatrixBuilder: verifies that add() and triplets() accumulate
// entries correctly, and that build_csr() produces the expected structure.
// Also verifies the LdltSolverBackend::name() identification string.

#include <gtest/gtest.h>
#include <map>
#ifdef HAVE_TBB
#include <tbb/task_arena.h>
#endif
#include "core/sparse_matrix.hpp"
#include "solver/solver_backend.hpp"
#include <array>
#include <utility>

using namespace vibestran;

// ── SparseMatrixBuilder::add and triplets (only used in tests) ────────────────

TEST(SparseMatrixBuilder, AddSingleTriplet) {
    SparseMatrixBuilder builder(3);
    builder.add(0, 0, 5.0);
    const auto& t = builder.triplets();
    ASSERT_EQ(t.size(), 1u);
    EXPECT_EQ(t[0].row, 0);
    EXPECT_EQ(t[0].col, 0);
    EXPECT_DOUBLE_EQ(t[0].value, 5.0);
}

TEST(SparseMatrixBuilder, AddMultipleTriplets) {
    SparseMatrixBuilder builder(4);
    builder.add(0, 0, 1.0);
    builder.add(1, 1, 2.0);
    builder.add(2, 2, 3.0);
    builder.add(0, 2, -0.5);
    const auto& t = builder.triplets();
    EXPECT_EQ(t.size(), 4u);
}

TEST(SparseMatrixBuilder, TripletsAccumulateDuplicates) {
    // Duplicate entries in triplets are summed on build_csr(),
    // but triplets() should return them as-is (pre-finalization).
    SparseMatrixBuilder builder(2);
    builder.add(0, 0, 3.0);
    builder.add(0, 0, 4.0);
    const auto& t = builder.triplets();
    ASSERT_EQ(t.size(), 2u);
    // Both entries present; their sum (7.0) will appear in the assembled matrix
    double sum = t[0].value + t[1].value;
    EXPECT_DOUBLE_EQ(sum, 7.0);
}

TEST(SparseMatrixBuilder, BuildCsrFromAddEntriesCorrect) {
    // 2×2 diagonal matrix: K = diag(3, 5)
    SparseMatrixBuilder builder(2);
    builder.add(0, 0, 3.0);
    builder.add(1, 1, 5.0);
    auto csr = builder.build_csr();
    EXPECT_EQ(csr.n, 2);
    EXPECT_EQ(csr.nnz, 2);
    ASSERT_EQ(csr.row_ptr.size(), 3u);
    EXPECT_EQ(csr.row_ptr[0], 0);
    EXPECT_EQ(csr.row_ptr[1], 1);
    EXPECT_EQ(csr.row_ptr[2], 2);
}

TEST(SparseMatrixBuilder, MergeFromCombinesThreadLocalTriplets) {
    SparseMatrixBuilder builder(3);
    SparseMatrixBuilder local_a(3, 0);
    SparseMatrixBuilder local_b(3, 0);

    local_a.add(0, 0, 1.25);
    local_a.add(1, 2, -2.0);
    local_b.add(0, 0, 0.75);
    local_b.add(2, 1, 3.5);

    builder.merge_from(std::move(local_a));
    builder.merge_from(std::move(local_b));

    auto csr = builder.build_csr();
    ASSERT_EQ(csr.row_ptr.size(), 4u);
    EXPECT_EQ(csr.nnz, 3);

    EXPECT_EQ(csr.row_ptr[0], 0);
    EXPECT_EQ(csr.row_ptr[1], 1);
    EXPECT_EQ(csr.row_ptr[2], 2);
    EXPECT_EQ(csr.row_ptr[3], 3);

    EXPECT_EQ(csr.col_ind[0], 0);
    EXPECT_DOUBLE_EQ(csr.values[0], 2.0);
    EXPECT_EQ(csr.col_ind[1], 2);
    EXPECT_DOUBLE_EQ(csr.values[1], -2.0);
    EXPECT_EQ(csr.col_ind[2], 1);
    EXPECT_DOUBLE_EQ(csr.values[2], 3.5);
}

// Duplicates are summed in insertion order, and floating-point addition does not
// associate: 1e16 + 1 - 1e16 is 0 one way and 1 another. So this sees any change of
// order, whichever part or group the terms arrived in.
TEST(SparseMatrixBuilder, BuildCsrSumsInInsertionOrderAcrossPartsAndThreads) {
    constexpr int kRows = 40000;
    const auto build = [] {
        SparseMatrixBuilder builder(kRows);
        for (int part = 0; part < 50; ++part) {
            SparseMatrixBuilder local(kRows);
            for (int row = 0; row < kRows; ++row) {
                const double sign = part % 2 == 0 ? 1.0 : -1.0;
                local.add(row, row, sign * 1e16);
                local.add(row, row, part + 1.0);
                if (row > 0) local.add(row, (row * 7 + part) % row, 1.0 / (part + row));
            }
            builder.merge_from(std::move(local));
            builder.add(part, part, 0.25);
        }
        return builder.build_csr();
    };
    const auto parallel = build();

    // The serial sum of the same triplets in the same order.
    std::vector<std::map<int, double>> expected(kRows);
    for (int part = 0; part < 50; ++part) {
        for (int row = 0; row < kRows; ++row) {
            const double sign = part % 2 == 0 ? 1.0 : -1.0;
            expected[row][row] += sign * 1e16;
            expected[row][row] += part + 1.0;
            if (row > 0) expected[row][(row * 7 + part) % row] += 1.0 / (part + row);
        }
        expected[part][part] += 0.25;
    }
    EXPECT_TRUE(parallel.stores_lower_triangle_only());
    for (int row = 0; row < kRows; ++row) {
        const int first = parallel.row_ptr[row];
        ASSERT_EQ(parallel.row_ptr[row + 1] - first, int(expected[row].size())) << row;
        int at = first;
        for (const auto& [column, value] : expected[row]) {
            ASSERT_EQ(parallel.col_ind[at], column) << row;
            ASSERT_EQ(parallel.values[at], value) << row;
            ++at;
        }
    }
#ifdef HAVE_TBB
    SparseMatrixBuilder::CsrData serial;
    tbb::task_arena(1).execute([&] { serial = build(); });
    EXPECT_TRUE(serial.values == parallel.values);
    EXPECT_TRUE(serial.col_ind == parallel.col_ind);
    EXPECT_TRUE(serial.row_ptr == parallel.row_ptr);
#endif
}

TEST(SparseMatrixBuilder, BuildCsrOfNothingHasEmptyRows) {
    SparseMatrixBuilder builder(3);
    const auto csr = builder.build_csr();
    EXPECT_EQ(csr.nnz, 0);
    EXPECT_EQ(csr.row_ptr, (std::vector<int>{0, 0, 0, 0}));
}

TEST(SparseMatrixBuilder, BuildCsrNoticesEntriesAboveTheDiagonal) {
    SparseMatrixBuilder lower(2), full(2);
    lower.add(1, 0, 1.0);
    full.add(0, 1, 1.0);
    EXPECT_TRUE(lower.build_csr().stores_lower_triangle_only());
    EXPECT_FALSE(full.build_csr().stores_lower_triangle_only());
}

TEST(SparseMatrixBuilder, ElementAssemblyBuildsLowerTriangleCsr) {
    SparseMatrixBuilder builder(2);
    const std::array<int32_t, 2> dofs = {0, 1};
    const std::array<double, 4> ke = {
        4.0, -1.5,
       -1.5,  3.0,
    };

    builder.add_element_stiffness(dofs, ke);
    auto csr = builder.build_csr();

    EXPECT_TRUE(csr.stores_lower_triangle_only());
    EXPECT_EQ(csr.nnz, 3);
    ASSERT_EQ(csr.row_ptr.size(), 3u);
    EXPECT_EQ(csr.row_ptr[0], 0);
    EXPECT_EQ(csr.row_ptr[1], 1);
    EXPECT_EQ(csr.row_ptr[2], 3);

    EXPECT_EQ(csr.col_ind[0], 0);
    EXPECT_DOUBLE_EQ(csr.values[0], 4.0);
    EXPECT_EQ(csr.col_ind[1], 0);
    EXPECT_DOUBLE_EQ(csr.values[1], -1.5);
    EXPECT_EQ(csr.col_ind[2], 1);
    EXPECT_DOUBLE_EQ(csr.values[2], 3.0);
}

TEST(SparseMatrixBuilder, LowerTriangleMatvecMatchesExpandedSymmetric) {
    SparseMatrixBuilder builder(2);
    const std::array<int32_t, 2> dofs = {0, 1};
    const std::array<double, 4> ke = {
        4.0, -1.5,
       -1.5,  3.0,
    };

    builder.add_element_stiffness(dofs, ke);
    auto lower = builder.build_csr();
    auto full = lower.expanded_symmetric();

    const std::array<double, 2> x = {2.0, -1.0};
    const auto y_lower = lower.multiply(x);
    const auto y_full = full.multiply(x);

    EXPECT_TRUE(lower.stores_lower_triangle_only());
    EXPECT_FALSE(full.stores_lower_triangle_only());
    EXPECT_EQ(full.nnz, 4);
    ASSERT_EQ(y_lower.size(), 2u);
    ASSERT_EQ(y_full.size(), 2u);
    EXPECT_DOUBLE_EQ(y_lower[0], 9.5);
    EXPECT_DOUBLE_EQ(y_lower[1], -6.0);
    EXPECT_EQ(y_lower, y_full);
}

TEST(SparseMatrixBuilder, LowerTriangleExtractsSymmetricStorageFromFullCsr) {
    SparseMatrixBuilder builder(3);
    builder.add(0, 0, 4.0);
    builder.add(0, 1, -1.0);
    builder.add(1, 0, -1.0);
    builder.add(1, 1, 3.0);
    builder.add(1, 2, -2.0);
    builder.add(2, 1, -2.0);
    builder.add(2, 2, 5.0);

    auto full = builder.build_csr();
    auto lower = full.lower_triangle();

    EXPECT_FALSE(full.stores_lower_triangle_only());
    EXPECT_TRUE(lower.stores_lower_triangle_only());
    EXPECT_EQ(lower.n, 3);
    EXPECT_EQ(lower.nnz, 5);
    ASSERT_EQ(lower.row_ptr.size(), 4u);
    EXPECT_EQ(lower.row_ptr[0], 0);
    EXPECT_EQ(lower.row_ptr[1], 1);
    EXPECT_EQ(lower.row_ptr[2], 3);
    EXPECT_EQ(lower.row_ptr[3], 5);

    EXPECT_EQ(lower.col_ind[0], 0);
    EXPECT_DOUBLE_EQ(lower.values[0], 4.0);
    EXPECT_EQ(lower.col_ind[1], 0);
    EXPECT_DOUBLE_EQ(lower.values[1], -1.0);
    EXPECT_EQ(lower.col_ind[2], 1);
    EXPECT_DOUBLE_EQ(lower.values[2], 3.0);
    EXPECT_EQ(lower.col_ind[3], 1);
    EXPECT_DOUBLE_EQ(lower.values[3], -2.0);
    EXPECT_EQ(lower.col_ind[4], 2);
    EXPECT_DOUBLE_EQ(lower.values[4], 5.0);
}

TEST(SparseMatrixBuilder, SizeReturnsConstructorArg) {
    SparseMatrixBuilder builder(42);
    EXPECT_EQ(builder.size(), 42);
}

// ── LdltSolverBackend::name (only used in tests) ─────────────────────────────

TEST(LdltSolverBackend, NameIsNonEmpty) {
    LdltSolverBackend backend;
    EXPECT_FALSE(backend.name().empty());
}

TEST(EigenSolverBackend, NameIdentifiesEigenBackend) {
    EigenSolverBackend backend;
    // The name must identify the underlying factorization so log output is
    // meaningful.  When SuiteSparse is available the name is "SuiteSparse
    // CHOLMOD (CPU)"; otherwise "Eigen SimplicialLLT (CPU)".
    const auto n = backend.name();
    const bool is_accel   = n.find("Accelerate") != std::string_view::npos;
    const bool is_cholmod = n.find("CHOLMOD") != std::string_view::npos;
    const bool is_eigen   = n.find("Eigen")   != std::string_view::npos;
    EXPECT_TRUE(is_accel || is_cholmod || is_eigen)
        << "Unexpected backend name: " << n;
}
