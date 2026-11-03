#pragma once
#include "core/sparse_matrix.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace vibestran::test {
struct PcgPartitionProblem {
    SparseMatrixBuilder::CsrData matrix;
    std::vector<double> exact, rhs;
};

// A grounded conducting/spring network with a material interface and long
// cross-partition edges. Assemble loads independently by edge flux, so dropping
// either side's coupling or permuting loads incorrectly changes the solution.
inline PcgPartitionProblem pcg_partition_problem(int n = 8192) {
    PcgPartitionProblem out;
    out.exact.resize(n); out.rhs.resize(n);
    for (int i = 0; i < n; ++i) out.exact[i] = std::sin(0.071 * i) + 0.2 * std::cos(0.019 * i);
    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i) {
        const double ground = 1.0 + (i % 7);
        builder.add(i, i, ground);
        out.rhs[i] += ground * out.exact[i];
        for (int stride : {1, 17, 133}) {
            const int j = i + stride;
            if (j >= n) continue;
            const double conductance = (i < n / 2 ? 1.0 : 100.0) / std::sqrt(double(stride));
            builder.add(i, i, conductance); builder.add(j, j, conductance);
            builder.add(j, i, -conductance);
            const double flux = conductance * (out.exact[i] - out.exact[j]);
            out.rhs[i] += flux; out.rhs[j] -= flux;
        }
    }
    out.matrix = builder.build_csr();
    return out;
}

// The problem with every coupling inside the stiffer half scaled by factor:
// the same sparsity pattern, a different operator, and loads for the same
// exact solution.
inline PcgPartitionProblem stiffened(const PcgPartitionProblem& problem, double factor) {
    auto out = problem;
    std::fill(out.rhs.begin(), out.rhs.end(), 0.0);
    auto& k = out.matrix;
    const bool lower = k.stores_lower_triangle_only();
    for (int i = 0; i < k.n; ++i)
        for (int j = k.row_ptr[i]; j < k.row_ptr[i + 1]; ++j) {
            const int c = k.col_ind[j];
            if (i >= k.n / 2 && c >= k.n / 2) k.values[j] *= factor;
            out.rhs[i] += k.values[j] * out.exact[c];
            if (lower && c != i) out.rhs[c] += k.values[j] * out.exact[i];
        }
    return out;
}
}
