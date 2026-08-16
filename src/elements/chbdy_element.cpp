// src/elements/chbdy_element.cpp
// CHBDY boundary element implementation.

#include "elements/chbdy_element.hpp"
#include "elements/thermal_elements.hpp"   // resolve_thermal_material

#include <array>
#include <cmath>
#include <format>
#include <numeric>

namespace vibestran {

namespace {

[[nodiscard]] double triangle_area(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return 0.5 * (b - a).cross(c - a).norm();
}

} // namespace

ChbdyElementImpl::ChbdyElementImpl(const ChbdyElement &data, const Model &model)
    : data_(data), model_(model) {
  // PHBDY owns both the film material and radiation/vector-flux properties.
  if (auto it = model_.phbdy_properties.find(data_.pid);
      it != model_.phbdy_properties.end()) {
    const PHBDY &property = it->second;
    absorptivity_ = property.absorptivity;
    if (property.mid.value != 0) {
      auto material = model_.mat4_materials.find(property.mid);
      if (material == model_.mat4_materials.end()) {
        throw SolverError(std::format(
            "CHBDY {} PHBDY {} references MAT4 ID {} which is not defined",
            data_.eid.value, data_.pid.value, property.mid.value));
      }
      film_h_ = material->second.k;
    }
  }
  compute_geometry();
}

void ChbdyElementImpl::compute_geometry() {
  const auto &n = data_.nodes;
  switch (data_.geom) {
  case ChbdyType::POINT: {
    if (n.size() != 1)
      throw SolverError(std::format("CHBDY POINT {} needs 1 node", data_.eid.value));
    double af = 1.0;
    if (auto it = model_.phbdy_properties.find(data_.pid);
        it != model_.phbdy_properties.end())
      af = it->second.af;
    area_ = af;
    area_frac_ = {area_};
    N_int_ = Eigen::VectorXd::Constant(1, area_);
    conv_ = Eigen::MatrixXd::Constant(1, 1, film_h_ * area_);
    break;
  }
  case ChbdyType::AREA3: {
    if (n.size() != 3)
      throw SolverError(std::format("CHBDY AREA3 {} needs 3 nodes", data_.eid.value));
    const Vec3 r1 = model_.node(n[0]).position;
    const Vec3 r2 = model_.node(n[1]).position;
    const Vec3 r3 = model_.node(n[2]).position;
    area_ = triangle_area(r1, r2, r3);
    if (area_ < 1e-20)
      throw SolverError(std::format("CHBDY AREA3 {} degenerate", data_.eid.value));
    normal_ = (r2 - r1).cross(r3 - r1).normalized();
    area_frac_ = {area_ / 3.0, area_ / 3.0, area_ / 3.0};
    N_int_ = Eigen::VectorXd::Constant(3, area_ / 3.0);
    // Consistent (Galerkin) convection: K = H·A/12 · [[2,1,1],[1,2,1],[1,1,2]]
    conv_.resize(3, 3);
    const double c = film_h_ * area_ / 12.0;
    conv_ << 2 * c, c, c,
             c, 2 * c, c,
             c, c, 2 * c;
    break;
  }
  case ChbdyType::AREA4: {
    if (n.size() != 4)
      throw SolverError(std::format("CHBDY AREA4 {} needs 4 nodes", data_.eid.value));
    const Vec3 r1 = model_.node(n[0]).position;
    const Vec3 r2 = model_.node(n[1]).position;
    const Vec3 r3 = model_.node(n[2]).position;
    const Vec3 r4 = model_.node(n[3]).position;
    // Diagonal cross-product area (NASTRAN-95 hbdys formula)
    const Vec3 d13 = r3 - r1, d24 = r4 - r2;
    area_ = 0.5 * d13.cross(d24).norm();
    if (area_ < 1e-20)
      throw SolverError(std::format("CHBDY AREA4 {} degenerate", data_.eid.value));
    normal_ = d13.cross(d24).normalized();
    // NASTRAN-95 HBDYS formulation. Ai is twice the area of the triangle
    // opposite node i. This preserves the correct nonuniform tributary areas
    // for tapered and mildly warped quadrilaterals.
    const std::array<Vec3, 4> edge{
        r3 - r2, r4 - r3, r1 - r4, r2 - r1};
    const std::array<Vec3, 4> cross{
        edge[0].cross(edge[1]), edge[1].cross(edge[2]),
        edge[2].cross(edge[3]), edge[3].cross(edge[0])};
    // Convexity guard (NASTRAN-95 HBDYS fatal 3090): the H/48 conductance
    // and (S − Ai)/12 tributary areas are only valid for convex quads; a
    // reentrant corner would produce negative areas.
    const double convex = cross[0].dot(cross[1]) * cross[0].dot(cross[2]) *
                          cross[0].dot(cross[3]);
    if (convex <= 0.0)
      throw SolverError(std::format(
          "CHBDY {} AREA4 is non-convex or degenerate", data_.eid.value));
    const std::array<double, 4> opposite_twice_area{
        cross[0].norm(), cross[1].norm(), cross[2].norm(), cross[3].norm()};
    const double area_sum = std::accumulate(opposite_twice_area.begin(),
                                            opposite_twice_area.end(), 0.0);
    area_frac_.resize(4);
    N_int_.resize(4);
    conv_.resize(4, 4);
    for (int i = 0; i < 4; ++i) {
      area_frac_[static_cast<size_t>(i)] =
          (area_sum - opposite_twice_area[static_cast<size_t>(i)]) / 12.0;
      N_int_(i) = area_frac_[static_cast<size_t>(i)];
      for (int j = 0; j < 4; ++j) {
        if (i == j) {
          conv_(i, j) = film_h_ *
              2.0 * (area_sum - opposite_twice_area[static_cast<size_t>(i)]) /
              48.0;
        } else {
          conv_(i, j) = film_h_ *
              (area_sum - opposite_twice_area[static_cast<size_t>(i)] -
               opposite_twice_area[static_cast<size_t>(j)]) / 48.0;
        }
      }
    }
    break;
  }
  default:
    throw SolverError(std::format(
        "CHBDY {} geometry type not yet supported", data_.eid.value));
  }
}

Eigen::MatrixXd ChbdyElementImpl::convection_conductance() const {
  return conv_;
}

Eigen::VectorXd ChbdyElementImpl::applied_flux_load(double q) const {
  return N_int_ * q;
}

Eigen::VectorXd ChbdyElementImpl::applied_flux_load_per_node(
    std::span<const double> q_per_node) const {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(static_cast<int>(area_frac_.size()));
  for (size_t i = 0; i < area_frac_.size(); ++i)
    out(static_cast<int>(i)) = q_per_node[i] * area_frac_[i];
  return out;
}

Eigen::VectorXd ChbdyElementImpl::directional_flux_load(
    double q0, const Vec3 &direction) const {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(static_cast<int>(area_frac_.size()));
  if (normal_.norm() < 0.5)
    return out;  // POINT or undefined normal — QVECT not meaningful
  // Inward-facing component: q_eff = q0 · max(0, −E·n̂).  NASTRAN-95 ignores
  // faces oriented away from the source (DOT ≥ 0 → no load).
  const double dot = direction.x * normal_.x + direction.y * normal_.y
                   + direction.z * normal_.z;
  if (dot >= 0.0)
    return out;
  const double q_eff = absorptivity_ * q0 * (-dot);
  for (size_t i = 0; i < area_frac_.size(); ++i)
    out(static_cast<int>(i)) = q_eff * area_frac_[i];
  return out;
}

} // namespace vibestran
