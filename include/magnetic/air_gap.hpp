#pragma once
// Air-gap element (AGE) for planar MAGSTAT: an unmeshed air annulus between
// two parts of the mesh, coupled through the exact Laplace solution in the
// annulus that matches the piecewise-linear potential on both circles.
#include "magnetic/mesh.hpp"

namespace vibestran::magnetic {
// Relative tolerance on circularity, concentricity and the annulus being
// empty, as a fraction of the radius.
inline constexpr double kAirGapTolerance = 1e-6;

// Finds the air gap of a planar mesh whose parts are not joined by cells.
// A connected or 3D mesh has none. A disconnected planar mesh must have
// exactly one gap, and it must join every part; otherwise the error names
// the grids and radii that failed.
[[nodiscard]] Result<std::optional<AirGap>> find_air_gap(const Mesh &mesh);

// Fourier coefficients of the piecewise-linear (in angle) function through
// the nodal values on circle c: rows are the mean, then cos and sin of each
// harmonic 1..harmonics, so that f(t) = c0 + sum c_n cos(nt) + s_n sin(nt).
[[nodiscard]] Eigen::MatrixXd gap_transform(const GapCircle &c, int harmonics);

// Stiffness per unit depth over [inner nodes, outer nodes]: (1/2) a^T K a is
// the energy per unit depth of the annulus field for those nodal potentials.
[[nodiscard]] Eigen::MatrixXd air_gap_stiffness(const AirGap &gap);

// Whether x (in the plane) lies in the annulus.
[[nodiscard]] bool in_air_gap(const AirGap &gap, const Vector &x);

// B at x in the annulus from the nodal potential a of the planar mesh.
[[nodiscard]] Vector air_gap_field(const AirGap &gap, const Eigen::VectorXd &a,
                                   const Vector &x);
} // namespace vibestran::magnetic
