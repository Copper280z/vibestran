#pragma once
// Air-gap element (AGE) for planar MAGSTAT: the elements of a PAIRGAP
// property are replaced by the exact Laplace solution in the air annulus
// that matches the piecewise-linear potential on both of its circles.
#include "magnetic/mesh.hpp"

namespace vibestran::magnetic {
// Relative tolerance on circularity, concentricity and the annulus being
// empty, as a fraction of the radius.
inline constexpr double kAirGapTolerance = 1e-6;

// The air gap of PAIRGAP property pid, from its elements (the field mesh
// excludes them). They must form one annulus between two meshed parts: two
// closed boundaries, each on a circle and shared with field cells, about a
// common center. The error names the grids, elements and radii that fail.
[[nodiscard]] Result<AirGap> make_air_gap(const Mesh &mesh, const Model &model,
                                          int pid);

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
