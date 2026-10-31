#include "magnetic/drive.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>

namespace vibestran::magnetic {
namespace {
constexpr double pi = std::numbers::pi, degree = pi / 180;
// x in (-180, 180].
double wrapped(double degrees) {
  const double out = std::remainder(degrees, 360.);
  return out == -180 ? 180 : out;
}
} // namespace

double electrical_angle(const Drive &drive, double theta0, double degrees) {
  return drive.poles * (degrees - theta0) * degree;
}

std::vector<double> phase_currents(const Drive &drive, double theta_e) {
  const auto m = static_cast<int>(drive.phases.size());
  const double gamma = drive.gamma * degree;
  const double id = -drive.peak * std::sin(gamma), iq = drive.peak * std::cos(gamma);
  std::vector<double> out;
  for (int k = 0; k < m; ++k) {
    const double theta = theta_e - 2 * pi * k / m;
    if (!drive.six) {
      out.push_back(id * std::cos(theta) - iq * std::sin(theta));
      continue;
    }
    // The sinusoidal current is peak cos(theta + gamma + 90 degrees); the
    // block conducts within 60 degrees of either peak.
    const double reference = wrapped((theta + gamma) / degree + 90);
    if (reference >= -60 && reference < 60)
      out.push_back(drive.peak);
    else if (reference >= 120 || reference < -120)
      out.push_back(-drive.peak);
    else
      out.push_back(0);
  }
  return out;
}

Eigen::MatrixXd park_columns(int phases, double theta_e) {
  Eigen::MatrixXd c(phases, 2);
  for (int k = 0; k < phases; ++k) {
    const double theta = theta_e - 2 * pi * k / phases;
    c(k, 0) = std::cos(theta);
    c(k, 1) = -std::sin(theta);
  }
  return c;
}

Fundamental fundamental(std::span<const double> samples) {
  const auto n = static_cast<double>(samples.size());
  double mean = 0, a = 0, b = 0;
  for (size_t j = 0; j < samples.size(); ++j) {
    const double angle = 2 * pi * static_cast<double>(j) / n;
    mean += samples[j] / n;
    a += 2 / n * samples[j] * std::cos(angle);
    b += 2 / n * samples[j] * std::sin(angle);
  }
  return {mean, std::hypot(a, b), std::atan2(b, a)};
}

std::vector<double> alignment_positions(const Drive &drive) {
  std::vector<double> out;
  for (int j = 0; j < 8; ++j)
    out.push_back(360. * j / (8 * drive.poles));
  return out;
}

Result<Alignment> align_drive(const Drive &drive,
                              const std::vector<std::vector<double>> &linkages) {
  const auto m = drive.phases.size();
  std::vector<Fundamental> phases;
  double scale = 0;
  for (size_t k = 0; k < m; ++k) {
    std::vector<double> samples;
    for (const auto &row : linkages) {
      samples.push_back(row[k]);
      scale = std::max(scale, std::abs(row[k]));
    }
    phases.push_back(fundamental(samples));
  }
  if (!(phases[0].amplitude > 1e-9 * scale) || scale == 0)
    return Error{std::format(
        "THETA0 AUTO: phase 1 (circuit {}) links no flux that varies with the "
        "rotor position at no load (fundamental {:.3e} Wb, largest linkage "
        "{:.3e} Wb), so it has no d axis to find; give THETA0",
        drive.phases[0], phases[0].amplitude, scale)};
  Alignment out;
  out.linkage = phases[0].amplitude;
  const double period = 360. / drive.poles;
  out.theta0 = std::fmod(phases[0].phase / degree / drive.poles + period, period);
  if (out.theta0 >= period)
    out.theta0 -= period;
  bool forward = true, reverse = true;
  std::string measured;
  for (size_t k = 0; k < m; ++k) {
    const double lag =
        std::fmod((phases[k].phase - phases[0].phase) / degree + 720, 360.);
    out.lag.push_back(lag);
    if (k == 0)
      continue;
    if (!(phases[k].amplitude > 1e-6 * phases[0].amplitude))
      return Error{std::format(
          "THETA0 AUTO: phase {} (circuit {}) links almost no rotor flux "
          "({:.3e} Wb against phase 1's {:.3e} Wb), so its place in the phase "
          "sequence cannot be checked",
          k + 1, drive.phases[k], phases[k].amplitude, phases[0].amplitude)};
    const double expected = 360. * static_cast<double>(k) / static_cast<double>(m),
                 half = 180. / static_cast<double>(m);
    forward &= std::abs(wrapped(lag - expected)) < half;
    reverse &= std::abs(wrapped(lag + expected)) < half;
    measured += std::format("{}phase {} (circuit {}) lags phase 1 by {:.1f}, "
                            "expected {:.1f}",
                            measured.empty() ? "" : "; ", k + 1, drive.phases[k],
                            lag, expected);
  }
  if (forward)
    return out;
  if (reverse)
    return Error{std::format(
        "THETA0 AUTO: the phase sequence opposes the direction of increasing "
        "position: {} electrical degrees. List the phases after the first in "
        "the opposite order{}",
        measured, m == 3 ? " (swap C2 and C3)" : "")};
  return Error{std::format(
      "THETA0 AUTO: the phases are not in sequence: {} electrical degrees, "
      "each within {:.1f}. Check the phase circuits and their polarities",
      measured, 180. / static_cast<double>(m))};
}
} // namespace vibestran::magnetic
