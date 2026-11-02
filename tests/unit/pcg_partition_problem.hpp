#pragma once
#include "core/sparse_matrix.hpp"
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
}
