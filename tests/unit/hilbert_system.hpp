#pragma once
// tests/unit/hilbert_system.hpp
// An ill-conditioned SPD test system with an exactly known solution, for the
// iterative-refinement tests of the CPU and GPU backends.

#include "core/sparse_matrix.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

namespace vibestran::test {

/// The Hilbert matrix of order n scaled by lcm(1..2n-1) to integers (exact in
/// double for n <= 10), in lower or full storage, and F = K * (1, ..., 1)
/// computed exactly, so the exact solution is u = 1.  cond(H_n) is about
/// 1.5e4 (n = 4), 4.8e5 (5), 1.5e7 (6), 1.5e10 (8), 1.6e13 (10).
inline std::pair<SparseMatrixBuilder::CsrData, std::vector<double>>
scaled_hilbert(int n, SymmetryStorage storage = SymmetryStorage::Lower) {
  long long l = 1;
  for (long long k = 2; k <= 2 * n - 1; ++k) {
    long long a = l, b = k;
    while (b != 0) {
      const long long t = a % b;
      a = b;
      b = t;
    }
    l = l / a * k;
  }
  SparseMatrixBuilder::CsrData K;
  K.n = n;
  K.symmetry = storage;
  K.row_ptr.push_back(0);
  std::vector<double> F(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const long long q = l / (i + j + 1); // exact: i + j + 1 divides l
      const auto v = static_cast<double>(q);
      if (j <= i || storage == SymmetryStorage::Full) {
        K.col_ind.push_back(j);
        K.values.push_back(v);
      }
      F[static_cast<std::size_t>(i)] += v; // exact: < 2^53
    }
    K.row_ptr.push_back(static_cast<int>(K.col_ind.size()));
  }
  K.nnz = static_cast<int>(K.values.size());
  return {K, F};
}

/// max_i |u_i - 1|: the error against the exact solution of scaled_hilbert().
inline double max_error_from_one(const std::vector<double> &u) {
  return std::accumulate(u.begin(), u.end(), 0.0,
                         [](double e, double v) { return std::max(e, std::abs(v - 1.0)); });
}

} // namespace vibestran::test
