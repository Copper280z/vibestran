#pragma once
// DRIVE: phase currents set from the electrical angle at each MOTION
// position, the amplitude-invariant Park transform of the phase quantities,
// and THETA0 AUTO's fit of the no-load linkage. A DRIVE follows a rotation
// (positions and THETA0 in degrees, POLES the pole pairs) or a translation
// (in m, POLES the pole pitch tau).

#include "magnetic/model.hpp"
#include <span>

namespace vibestran::magnetic {

// One electrical period of the position: 360 / poles degrees of a
// rotation, two pole pitches (m) of a translation.
[[nodiscard]] double electrical_period(const Drive &drive, const Motion &motion);

// The electrical angle in radians at a position of motion, for theta0 in
// the same unit: poles (phi - theta0) for a rotation in degrees,
// pi (x - x0) / tau for a translation.
[[nodiscard]] double electrical_angle(const Drive &drive, const Motion &motion,
                                      double theta0, double position);

// The terminal current of each phase at electrical angle theta_e (radians).
// SINE: i_d cos(theta_k) - i_q sin(theta_k). SIX: +peak while the phase's
// sinusoidal reference is within 60 electrical degrees of its positive peak,
// -peak within 60 degrees of its negative peak, and 0 between, so two phases
// conduct at a time and the fundamental is in phase with SINE's.
[[nodiscard]] std::vector<double> phase_currents(const Drive &drive,
                                                 double theta_e);

// The m x 2 matrix C with rows (cos theta_k, -sin theta_k): the phase
// quantities of d and q components are C (d, q)^T, and the amplitude-invariant
// Park transform is (2/m) C^T.
[[nodiscard]] Eigen::MatrixXd park_columns(int phases, double theta_e);

// The first harmonic of samples over one period, at angles 2 pi j / n:
// value ~ mean + amplitude cos(angle - phase).
struct Fundamental {
  double mean{0}, amplitude{0}, phase{0}; // phase in radians, in (-pi, pi]
};
[[nodiscard]] Fundamental fundamental(std::span<const double> samples);

// THETA0 AUTO from the no-load linkage of each phase at the positions
// j / 8 of an electrical period, j = 0 .. 7 (alignment_positions):
// linkages[j][k] is phase k at position j. Phase 1's fundamental gives
// theta0 (its peak is the d axis lined up with phase 1), in [0, period),
// in the motion's unit. Each later phase must
// lag phase 1 by 2 pi (k - 1) / m electrical radians, within half the phase
// spacing; otherwise the error says how the sequence differs.
struct Alignment {
  double theta0{0};                  // degrees (a translation's: m)
  double linkage{0};                 // phase 1's fundamental amplitude, Wb
  std::vector<double> lag;           // electrical degrees behind phase 1
};
[[nodiscard]] Result<Alignment>
align_drive(const Drive &drive, const Motion &motion,
            const std::vector<std::vector<double>> &linkages);

// The positions (degrees, or m for a translation) of the THETA0 AUTO
// pre-sweep: 8 over one electrical period from 0.
[[nodiscard]] std::vector<double> alignment_positions(const Drive &drive,
                                                      const Motion &motion);

} // namespace vibestran::magnetic
