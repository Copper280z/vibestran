#include "magnetic/model.hpp"
#include <cmath>

namespace vibestran::magnetic {
Response material_response(const Material &material,
                           const std::vector<std::pair<double, double>> &curve,
                           const Vector &b, const Vector &remanence) {
  Response r;
  if (curve.empty()) {
    r.tangent = Matrix::Identity() / (mu0 * material.mur);
    const Vector shifted = b - remanence;
    r.h = r.tangent * shifted;
    r.energy = 0.5 * shifted.dot(r.h);
    return r;
  }
  const double magnitude = b.norm();
  double h = 0, slope = curve[1].second / curve[1].first;
  for (size_t i = 1; i < curve.size(); ++i) {
    const double start = curve[i - 1].first, end = curve[i].first;
    slope = (curve[i].second - curve[i - 1].second) / (end - start);
    const double dx = std::min(magnitude, end) - start;
    r.energy += curve[i - 1].second * dx + 0.5 * slope * dx * dx;
    h = curve[i - 1].second + slope * dx;
    if (magnitude <= end)
      break;
    if (i + 1 == curve.size()) {
      const double extra = magnitude - end;
      slope = 1 / mu0;
      r.energy += h * extra + 0.5 * slope * extra * extra;
      h += slope * extra;
      r.extrapolated = true;
    }
  }
  if (magnitude == 0)
    r.tangent = slope * Matrix::Identity();
  else {
    const Vector direction = b / magnitude;
    const double secant = h / magnitude;
    r.h = h * direction;
    r.tangent = secant * Matrix::Identity() +
                (slope - secant) * direction * direction.transpose();
  }
  return r;
}
} // namespace vibestran::magnetic
