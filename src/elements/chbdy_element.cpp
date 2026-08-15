// src/elements/chbdy_element.cpp
// CHBDY boundary element implementation.

#include "elements/chbdy_element.hpp"
#include "elements/thermal_elements.hpp"   // resolve_thermal_material

#include <cmath>
#include <format>

namespace vibestran {

namespace {

[[nodiscard]] double triangle_area(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return 0.5 * (b - a).cross(c - a).norm();
}

} // namespace

ChbdyElementImpl::ChbdyElementImpl(const ChbdyElement &data, const Model &model)
    : data_(data), model_(model) {
  // Material: MAT4.k holds the film coefficient H for CHBDY.
  if (auto it = model_.mat4_materials.find(data_.mid);
      it != model_.mat4_materials.end()) {
    film_h_ = it->second.k;
  } else if (data_.mid.value != 0) {
    throw SolverError(std::format(
        "CHBDY {} references MAT4 ID {} which is not defined",
        data_.eid.value, data_.mid.value));
  }
  // T_amb: from PHBDY if present.
  if (auto it = model_.phbdy_properties.find(data_.pid);
      it != model_.phbdy_properties.end()) {
    t_amb_ = it->second.t_amb;
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
    // Uniform lumping for applied flux: ∫N_i dA = A_face / 4 for a flat quad
    // under bilinear shape functions (row-summing the 2×2-Gauss consistent
    // mass matrix).  Keeps total applied load = q·A_face for uniform flux.
    area_frac_ = {area_ / 4.0, area_ / 4.0, area_ / 4.0, area_ / 4.0};
    N_int_ = Eigen::VectorXd::Constant(4, area_ / 4.0);
    // Lumped (row-summed) convection matrix: K_ii = H · A_face / 4.
    // Equivalent to the consistent bilinear matrix after row-sum lumping;
    // sufficient for steady-state where T is single-valued at each node.
    conv_ = Eigen::MatrixXd::Zero(4, 4);
    for (int i = 0; i < 4; ++i)
      conv_(i, i) = film_h_ * area_ / 4.0;
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

Eigen::VectorXd ChbdyElementImpl::ambient_rhs() const {
  // P_i = H · T_amb · ∫ N_i dA  (consistent integral; matches conv_ rowsum)
  // Use conv_.rowwise().sum() · T_amb so the resulting steady-state with
  // uniform T = T_amb yields zero net heat — i.e. the rowsum of conv_ equals
  // H · ∫ N_i dA exactly.
  return conv_.rowwise().sum() * t_amb_;
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
  const double q_eff = q0 * (-dot);
  for (size_t i = 0; i < area_frac_.size(); ++i)
    out(static_cast<int>(i)) = q_eff * area_frac_[i];
  return out;
}

} // namespace vibestran
