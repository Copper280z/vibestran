#pragma once
// tests/unit/dense_block_system.hpp
// An SPD matrix shaped like a magnetostatic tangent with an air gap, for the
// FactorOptions::dense_block tests of the CPU and GPU backends.

#include <Eigen/Dense>
#include <cmath>

namespace vibestran::test {

/// A 5-point Laplacian on an nx-by-nx grid, shifted to be definite, whose
/// last `ring` unknowns also carry a dense positive semidefinite block
/// V V^T (V with 8 columns), the way the harmonic air gap couples every gap
/// node to every other. With ring > 10 sqrt(nx^2), CHOLMOD's AMD sets the
/// ring aside as dense rows, and its fill estimate (each set-aside row
/// filling every column) passes the ratio at which CHOLMOD switches to METIS.
/// With ring = 0 it is a plain planar mesh.
inline Eigen::MatrixXd gap_like(int nx, int ring) {
  const int n = nx * nx;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
  for (int y = 0; y < nx; ++y)
    for (int x = 0; x < nx; ++x) {
      const int i = y * nx + x;
      A(i, i) = 4.1;
      if (x > 0)
        A(i, i - 1) = A(i - 1, i) = -1.0;
      if (y > 0)
        A(i, i - nx) = A(i - nx, i) = -1.0;
    }
  Eigen::MatrixXd v(ring, 8);
  for (int i = 0; i < ring; ++i)
    for (int k = 0; k < 8; ++k)
      v(i, k) = (1.0 + std::sin(0.7 * (i + 1) * (k + 1))) / std::sqrt(ring);
  A.bottomRightCorner(ring, ring) += v * v.transpose();
  return A;
}

} // namespace vibestran::test
