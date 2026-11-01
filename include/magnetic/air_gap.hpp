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

// Harmonics a circle's trace resolves: half its grid count (unfolded in a
// sector model).
[[nodiscard]] int own_harmonics(const GapCircle &c);
// Harmonics the two circles couple through, and the AGE's torque sums: the
// coarser circle's own_harmonics.
[[nodiscard]] int coupled_harmonics(const AirGap &gap);

// Stiffness per unit depth over [inner nodes, outer nodes]: (1/2) a^T K a is
// the energy per unit depth of the annulus field for those nodal potentials,
// each the circle's sign times its grid's potential: each circle's self
// terms to its own_harmonics and the coupling to coupled_harmonics, whatever
// NHARM. In a sector model that is the whole machine's gap.
[[nodiscard]] Eigen::MatrixXd air_gap_stiffness(const AirGap &gap);

// Whether x (in the plane) lies in the annulus.
[[nodiscard]] bool in_air_gap(const AirGap &gap, const Vector &x);

// The annulus field of one state, the nodal potential a of the planar mesh:
// the two circles' harmonics are computed once, so evaluating many points
// (probes, the drawn gap cells) costs a sum over the harmonics each. gap
// must outlive the series. With a sliding band it is the same: the annulus
// field of the solved traces, which converges at second order where the
// band's own bilinear field is first order.
class AirGapSeries {
public:
  AirGapSeries(const AirGap &gap, const Eigen::VectorXd &a);
  // B at x in the annulus.
  [[nodiscard]] Vector field(const Vector &x) const;
  // The potential A at x in the annulus, the series the element solves: on
  // the circles it is the harmonics of the nodal trace, not the trace
  // itself.
  [[nodiscard]] double potential(const Vector &x) const;

private:
  struct Point {
    double a{0}, da_dr{0}, da_dtheta{0}, r{0}, theta{0};
  };
  [[nodiscard]] Point evaluate(const Vector &x) const;
  const AirGap *gap_;
  Eigen::VectorXd inner_, outer_; // harmonics, as gap_transform's rows
};

// Torque per unit depth on the part inside the gap (in a sector model, on
// the whole machine's), about the plane normal
// u x v: -dW/d(delta) for a rigid rotation delta of the inner part with its
// nodal potentials held, i.e. of the inner circle's harmonics. The part
// outside the gap takes the opposite torque. It equals the Maxwell stress
// torque on any circle in the annulus.
[[nodiscard]] double air_gap_torque(const AirGap &gap, const Eigen::VectorXd &a);
// The gradient of air_gap_torque with respect to the nodal potentials a
// (zero off the circles). The torque is bilinear, so gradient . a is twice
// the torque. With the response da/di of a solve to a circuit's current it
// gives dT/di, which by reciprocity is the circuit's dlambda/d(delta).
[[nodiscard]] Eigen::VectorXd air_gap_torque_gradient(const AirGap &gap,
                                                      const Eigen::VectorXd &a);

// The sliding band (PAIRGAP METHOD=BAND): gap.band_layers rings of
// gap.band_points elements on a uniform angular grid, psi_k = 2 pi k / band_points from the
// gap's u axis, fixed in the plane. Each grid point's potential on a circle
// is that circle's piecewise-linear trace averaged over the point's hat
// function, so turning a part only moves the traces under the band. In log-polar coordinates
// xi = ln(r / r1) / ln(r2 / r1) the annulus is a rectangle with constant
// coefficients, and each element is bilinear in (xi, theta): exact for the
// mean, ln r term. For harmonic n one layer is too stiff by about
// (n ln(r2/r1))^4 / 45 in self energy, and M layers divide the error by
// about M^2. Fields in the gap are AirGapSeries's, from the traces.

// What a gap stiffness acts on: the unknown and its sign for each inner
// circle entry, each outer one, and for a band its interior point rings 1
// to M-1, N points each (unfolded around a sector model like the circles).
[[nodiscard]] std::vector<std::pair<int, double>> gap_entries(const AirGap &gap);
// Stiffness per unit depth over gap_entries, like air_gap_stiffness (which
// has no interior rings), but sparse: an entry couples only to entries
// whose band points share an element.
[[nodiscard]] Eigen::SparseMatrix<double> band_stiffness(const AirGap &gap);
// -dW/d(delta) of the band's energy for a rigid rotation delta of the inner
// part, its traces sampled at the turned angles, per unit depth (whole
// machine in a sector model).
[[nodiscard]] double band_torque(const AirGap &gap, const Eigen::VectorXd &a);
// The gradient of band_torque with respect to a; band_torque is bilinear,
// so gradient . a is twice the torque.
[[nodiscard]] Eigen::VectorXd band_torque_gradient(const AirGap &gap,
                                                   const Eigen::VectorXd &a);
} // namespace vibestran::magnetic
