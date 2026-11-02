#pragma once
#include "core/sparse_matrix.hpp"
#include <algorithm>
#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace vibestran::detail {
struct ColoredPcgMatrix {
    SparseMatrixBuilder::CsrData matrix;
    std::vector<int> new_index; // original equation -> permuted equation
    int colors{0};
};

// Symmetric graph coloring groups independent equations, shortening the
// triangular dependency chains of IC0. Limit colors to avoid replacing a
// useful geometric ordering with a deep, ineffective multicolor ordering.
inline std::optional<ColoredPcgMatrix> color_pcg_matrix(const SparseMatrixBuilder::CsrData& k) {
    constexpr int max_colors = 32;
    std::vector<int> colors(k.n);
    std::array<int, max_colors> used, counts{};
    used.fill(-1);
    int nc = 0;
    for (int row = 0; row < k.n; ++row) {
        for (int j = k.row_ptr[row]; j < k.row_ptr[row + 1]; ++j) {
            const int col = k.col_ind[j];
            if (col < row && k.values[j] != 0) used[colors[col]] = row;
        }
        int color = 0;
        while (color < max_colors && used[color] == row) ++color;
        if (color == max_colors) return {};
        colors[row] = color;
        ++counts[color];
        nc = std::max(nc, color + 1);
    }
    ColoredPcgMatrix out;
    out.colors = nc;
    out.new_index.resize(k.n);
    std::array<int, max_colors> offsets{};
    for (int color = 1; color < nc; ++color) offsets[color] = offsets[color - 1] + counts[color - 1];
    for (int row = 0; row < k.n; ++row) out.new_index[row] = offsets[colors[row]]++;
    auto& a = out.matrix;
    a.n = k.n; a.nnz = k.nnz; a.symmetry = k.symmetry;
    a.row_ptr.assign(std::size_t(k.n) + 1, 0);
    a.col_ind.resize(k.nnz); a.values.resize(k.nnz);
    for (int row = 0; row < k.n; ++row)
        for (int j = k.row_ptr[row]; j < k.row_ptr[row + 1]; ++j) {
            int r = out.new_index[row], c = out.new_index[k.col_ind[j]];
            if (k.stores_lower_triangle_only() && r < c) std::swap(r, c);
            ++a.row_ptr[r + 1];
        }
    for (int row = 0; row < k.n; ++row) a.row_ptr[row + 1] += a.row_ptr[row];
    auto cursor = a.row_ptr;
    for (int row = 0; row < k.n; ++row)
        for (int j = k.row_ptr[row]; j < k.row_ptr[row + 1]; ++j) {
            int r = out.new_index[row], c = out.new_index[k.col_ind[j]];
            if (k.stores_lower_triangle_only() && r < c) std::swap(r, c);
            const int at = cursor[r]++;
            a.col_ind[at] = c; a.values[at] = k.values[j];
        }
    std::vector<std::pair<int, double>> sorted;
    for (int row = 0; row < k.n; ++row) {
        sorted.clear();
        for (int j = a.row_ptr[row]; j < a.row_ptr[row + 1]; ++j) sorted.emplace_back(a.col_ind[j], a.values[j]);
        std::sort(sorted.begin(), sorted.end());
        int j = a.row_ptr[row];
        for (const auto& [col, value] : sorted) { a.col_ind[j] = col; a.values[j++] = value; }
    }
    return out;
}
}
