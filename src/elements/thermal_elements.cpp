// src/elements/thermal_elements.cpp
// Implementations of the volume/line thermal elements declared in
// thermal_elements.hpp.  CHBDY (surface boundary) elements live in
// chbdy_element.cpp.

#include "elements/thermal_elements.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <functional>
#include <variant>

namespace vibestran {

namespace {

[[nodiscard]] std::array<Vec3, 2>
node_pair(const Model &model, NodeId a, NodeId b) {
  return {model.node(a).position, model.node(b).position};
}

template <int N>
[[nodiscard]] std::array<Vec3, N>
gather_nodes(const Model &model, std::span<const NodeId> ids) {
  std::array<Vec3, N> out;
  for (int i = 0; i < N; ++i) {
    // NodeId{0} is the "absent midnode" sentinel for variable-noded elements
    // — its slot is unused after midnode-shape redistribution, so position
    // doesn't matter; use the origin as a placeholder.
    out[static_cast<size_t>(i)] = (ids[static_cast<size_t>(i)].value == 0)
        ? Vec3{0, 0, 0}
        : model.node(ids[static_cast<size_t>(i)]).position;
  }
  return out;
}

[[nodiscard]] double bar_area(const Property &prop) {
  if (const auto *pbar = std::get_if<PBar>(&prop))
    return pbar->A;
  if (const auto *pbarl = std::get_if<PBarL>(&prop))
    return pbarl->A;
  if (const auto *pbeam = std::get_if<PBeam>(&prop))
    return pbeam->A;
  return 0.0;
}

[[nodiscard]] double shell_thickness(const Property &prop) {
  if (const auto *ps = std::get_if<PShell>(&prop))
    return ps->t;
  return 0.0;
}

// Element local in-plane frame for a flat 3-D shell facet.  e1 along first
// edge, e3 = facet normal, e2 = e3 × e1.  Returns the projected 2-D node
// coords plus the (e1, e2) basis (for transforming material conductivity
// tensors / mapping out flux components into the local frame).
struct ShellFrame {
  Vec3 e1{0,0,0}, e2{0,0,0}, e3{0,0,0};
  std::vector<std::array<double, 2>> xy;  // (x, y) per node, local frame
};

[[nodiscard]] ShellFrame build_shell_frame(std::span<const Vec3> nodes) {
  ShellFrame f;
  const Vec3 r0 = nodes[0];
  const Vec3 r1 = nodes[1];
  const Vec3 r2 = nodes[nodes.size() > 3 ? 3 : 2];
  Vec3 v01 = r1 - r0;
  Vec3 v02 = r2 - r0;
  f.e3 = v01.cross(v02).normalized();
  f.e1 = v01.normalized();
  f.e2 = f.e3.cross(f.e1);
  f.xy.resize(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    const Vec3 d = nodes[i] - r0;
    f.xy[i] = {d.dot(f.e1), d.dot(f.e2)};
  }
  return f;
}

// Project a 3-D conductivity tensor onto the 2-D local frame:
//   k2D_ij = e_i^T · k · e_j  (i,j ∈ {e1, e2})
[[nodiscard]] Eigen::Matrix2d
project_k_to_plane(const Eigen::Matrix3d &k, const Vec3 &e1, const Vec3 &e2) {
  Eigen::Matrix2d k2;
  Eigen::Vector3d v1(e1.x, e1.y, e1.z);
  Eigen::Vector3d v2(e2.x, e2.y, e2.z);
  k2(0, 0) = v1.transpose() * k * v1;
  k2(0, 1) = v1.transpose() * k * v2;
  k2(1, 0) = k2(0, 1);
  k2(1, 1) = v2.transpose() * k * v2;
  return k2;
}

} // namespace

// ── Material helpers ─────────────────────────────────────────────────────────

ThermalMaterial resolve_thermal_material(const Model &model, MaterialId mid) {
  ThermalMaterial tm;
  if (auto it = model.mat4_materials.find(mid); it != model.mat4_materials.end()) {
    tm.isotropic = true;
    tm.k = it->second.k;
    tm.cp = it->second.cp;
    tm.k_tensor = Eigen::Matrix3d::Identity() * tm.k;
    return tm;
  }
  if (auto it = model.mat5_materials.find(mid); it != model.mat5_materials.end()) {
    tm.isotropic = false;
    const auto &m = it->second;
    tm.k_tensor << m.kxx, m.kxy, m.kxz,
                   m.kxy, m.kyy, m.kyz,
                   m.kxz, m.kyz, m.kzz;
    tm.k = (m.kxx + m.kyy + m.kzz) / 3.0; // representative value
    tm.cp = m.cp;
    return tm;
  }
  throw SolverError(std::format(
      "Material {} has no MAT4 or MAT5 thermal definition", mid.value));
}

MaterialId property_thermal_mid(const Model &model, PropertyId pid) {
  if (!model.properties.count(pid))
    throw SolverError(std::format(
        "Property {} undefined (referenced by thermal element)", pid.value));
  const Property &prop = model.property(pid);
  if (const auto *p = std::get_if<PSolid>(&prop))
    return p->mid;
  if (const auto *p = std::get_if<PShell>(&prop))
    return p->mid1;
  if (const auto *p = std::get_if<PBar>(&prop))
    return p->mid;
  if (const auto *p = std::get_if<PBarL>(&prop))
    return p->mid;
  if (const auto *p = std::get_if<PBeam>(&prop))
    return p->mid;
  throw SolverError(std::format(
      "Property {} has no material reference suitable for thermal analysis",
      pid.value));
}

// ── ThermalLine ──────────────────────────────────────────────────────────────
// 1-D axial conduction on a 2-node element.  Ke = (kA/L) · [[1,-1],[-1,1]]

class ThermalLine final : public ThermalElement {
public:
  ThermalLine(ElementId eid, ElementType type, PropertyId pid,
              std::array<NodeId, 2> nodes, const Model &model)
      : eid_(eid), type_(type), pid_(pid), nodes_(nodes), model_(model) {
    const auto pts = node_pair(model_, nodes_[0], nodes_[1]);
    length_ = (pts[1] - pts[0]).norm();
    if (length_ < 1e-20)
      throw SolverError(std::format("Thermal line element {} has zero length",
                                    eid_.value));
    area_ = bar_area(model_.property(pid_));
    const MaterialId mid = property_thermal_mid(model_, pid_);
    mat_ = resolve_thermal_material(model_, mid);
  }

  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] ElementType type() const noexcept override { return type_; }
  [[nodiscard]] int num_nodes() const noexcept override { return 2; }
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), 2};
  }

  [[nodiscard]] Eigen::MatrixXd conductance_matrix() const override {
    const double c = mat_.k * area_ / length_;
    Eigen::MatrixXd Ke(2, 2);
    Ke << c, -c, -c, c;
    return Ke;
  }

  [[nodiscard]] Eigen::VectorXd
  volumetric_heat_load(double q_vol) const override {
    const double half = 0.5 * q_vol * area_ * length_;
    Eigen::VectorXd p(2);
    p << half, half;
    return p;
  }

  [[nodiscard]] Eigen::VectorXd
  heat_flux(std::span<const double> t) const override {
    Eigen::VectorXd q(1);
    q(0) = -mat_.k * (t[1] - t[0]) / length_;
    return q;
  }

private:
  ElementId eid_;
  ElementType type_;
  PropertyId pid_;
  std::array<NodeId, 2> nodes_;
  const Model &model_;
  double length_{0.0};
  double area_{0.0};
  ThermalMaterial mat_;
};

// ── ThermalTetra4 ────────────────────────────────────────────────────────────
// Linear tetrahedron: constant ∇N → closed-form Ke = V · Bᵀ k B.

class ThermalTetra4 final : public ThermalElement {
public:
  ThermalTetra4(ElementId eid, PropertyId pid,
                std::array<NodeId, 4> nodes, const Model &model)
      : eid_(eid), pid_(pid), nodes_(nodes), model_(model) {
    const MaterialId mid = property_thermal_mid(model_, pid_);
    mat_ = resolve_thermal_material(model_, mid);
    compute_geometry();
  }

  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] ElementType type() const noexcept override {
    return ElementType::CTETRA4;
  }
  [[nodiscard]] int num_nodes() const noexcept override { return 4; }
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), 4};
  }

  [[nodiscard]] Eigen::MatrixXd conductance_matrix() const override {
    return volume_ * B_.transpose() * mat_.k_tensor * B_;
  }

  [[nodiscard]] Eigen::VectorXd
  volumetric_heat_load(double q_vol) const override {
    return Eigen::VectorXd::Constant(4, 0.25 * q_vol * volume_);
  }

  [[nodiscard]] Eigen::VectorXd
  heat_flux(std::span<const double> t) const override {
    Eigen::Vector4d T;
    T << t[0], t[1], t[2], t[3];
    return -(mat_.k_tensor * (B_ * T));
  }

private:
  void compute_geometry() {
    const auto c = gather_nodes<4>(model_, node_ids());
    Eigen::Matrix4d A;
    for (int i = 0; i < 4; ++i)
      A.row(i) << 1.0, c[i].x, c[i].y, c[i].z;
    const double det = A.determinant();
    volume_ = std::abs(det) / 6.0;
    if (volume_ < 1e-20)
      throw SolverError(std::format(
          "Thermal CTETRA4 {} has degenerate volume", eid_.value));
    // ∇N_i = cof(A) / det along x,y,z rows.  Sign of det ⇒ sign of B; the
    // outer Bᵀ k B kills any global sign, but the flux recovery uses raw B,
    // so divide by det (with sign) not |det|.
    Eigen::Matrix4d cofA = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        Eigen::Matrix3d m3;
        int ri = 0;
        for (int r = 0; r < 4; ++r) {
          if (r == j) continue;
          int ci = 0;
          for (int c2 = 0; c2 < 4; ++c2) {
            if (c2 == i) continue;
            m3(ri, ci++) = A(r, c2);
          }
          ++ri;
        }
        cofA(i, j) = std::pow(-1.0, i + j) * m3.determinant();
      }
    }
    B_.resize(3, 4);
    for (int n = 0; n < 4; ++n) {
      B_(0, n) = cofA(1, n) / det;
      B_(1, n) = cofA(2, n) / det;
      B_(2, n) = cofA(3, n) / det;
    }
  }

  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, 4> nodes_;
  const Model &model_;
  ThermalMaterial mat_;
  double volume_{0.0};
  Eigen::MatrixXd B_;  // 3 × 4
};

// ── Generic isoparametric helper for 3D Gauss elements ──────────────────────
//
// Templated over: number of nodes N, and shape-function functor that produces
// (N, dN/dξ, dN/dη, dN/dζ) at a given (ξ, η, ζ).

namespace {

template <int N, typename Shape>
[[nodiscard]] Eigen::MatrixXd
solid_conductance(const std::array<Vec3, N> &coords, const Eigen::Matrix3d &kt,
                  const std::vector<std::array<double, 4>> &gauss /*xi,eta,zeta,w*/,
                  const Shape &shape) {
  Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero(N, N);
  for (const auto &gp : gauss) {
    const auto sd = shape(gp[0], gp[1], gp[2]);
    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    for (int n = 0; n < N; ++n) {
      J(0, 0) += sd.dN0[n] * coords[n].x;
      J(0, 1) += sd.dN0[n] * coords[n].y;
      J(0, 2) += sd.dN0[n] * coords[n].z;
      J(1, 0) += sd.dN1[n] * coords[n].x;
      J(1, 1) += sd.dN1[n] * coords[n].y;
      J(1, 2) += sd.dN1[n] * coords[n].z;
      J(2, 0) += sd.dN2[n] * coords[n].x;
      J(2, 1) += sd.dN2[n] * coords[n].y;
      J(2, 2) += sd.dN2[n] * coords[n].z;
    }
    const double det = J.determinant();
    const Eigen::Matrix3d Jinv = J.inverse();
    // J as built is (parameter × physical): J(r, c) = ∂x_c/∂ξ_r, i.e. the
    // transpose of the usual Jacobian.  Physical gradients are therefore
    // ∇N_n = J⁻¹·(∂N_n/∂ξ, ∂N_n/∂η, ∂N_n/∂ζ).
    Eigen::MatrixXd B(3, N);
    for (int n = 0; n < N; ++n) {
      const double a = sd.dN0[n], b = sd.dN1[n], cc = sd.dN2[n];
      B(0, n) = Jinv(0, 0) * a + Jinv(0, 1) * b + Jinv(0, 2) * cc;
      B(1, n) = Jinv(1, 0) * a + Jinv(1, 1) * b + Jinv(1, 2) * cc;
      B(2, n) = Jinv(2, 0) * a + Jinv(2, 1) * b + Jinv(2, 2) * cc;
    }
    Ke += (B.transpose() * kt * B) * (std::abs(det) * gp[3]);
  }
  return Ke;
}

template <int N, typename Shape>
[[nodiscard]] Eigen::VectorXd
solid_vol_load(const std::array<Vec3, N> &coords, double q_vol,
               const std::vector<std::array<double, 4>> &gauss,
               const Shape &shape) {
  Eigen::VectorXd p = Eigen::VectorXd::Zero(N);
  for (const auto &gp : gauss) {
    const auto sd = shape(gp[0], gp[1], gp[2]);
    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    for (int n = 0; n < N; ++n) {
      J(0, 0) += sd.dN0[n] * coords[n].x;
      J(0, 1) += sd.dN0[n] * coords[n].y;
      J(0, 2) += sd.dN0[n] * coords[n].z;
      J(1, 0) += sd.dN1[n] * coords[n].x;
      J(1, 1) += sd.dN1[n] * coords[n].y;
      J(1, 2) += sd.dN1[n] * coords[n].z;
      J(2, 0) += sd.dN2[n] * coords[n].x;
      J(2, 1) += sd.dN2[n] * coords[n].y;
      J(2, 2) += sd.dN2[n] * coords[n].z;
    }
    const double det = std::abs(J.determinant());
    for (int n = 0; n < N; ++n)
      p(n) += sd.N[n] * q_vol * det * gp[3];
  }
  return p;
}

template <int N, typename Shape>
[[nodiscard]] Eigen::VectorXd
solid_centroid_flux(const std::array<Vec3, N> &coords,
                    const Eigen::Matrix3d &kt,
                    std::span<const double> t,
                    double xi, double eta, double zeta,
                    const Shape &shape) {
  const auto sd = shape(xi, eta, zeta);
  Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
  for (int n = 0; n < N; ++n) {
    J(0, 0) += sd.dN0[n] * coords[n].x;
    J(0, 1) += sd.dN0[n] * coords[n].y;
    J(0, 2) += sd.dN0[n] * coords[n].z;
    J(1, 0) += sd.dN1[n] * coords[n].x;
    J(1, 1) += sd.dN1[n] * coords[n].y;
    J(1, 2) += sd.dN1[n] * coords[n].z;
    J(2, 0) += sd.dN2[n] * coords[n].x;
    J(2, 1) += sd.dN2[n] * coords[n].y;
    J(2, 2) += sd.dN2[n] * coords[n].z;
  }
  const Eigen::Matrix3d Jinv = J.inverse();
  // ∇T = Σ_n T_n·J⁻¹·∂N_n/∂ξ — same transposed-Jacobian convention as
  // solid_conductance above.
  Eigen::Vector3d gradT = Eigen::Vector3d::Zero();
  for (int n = 0; n < N; ++n) {
    const double a = sd.dN0[n], b = sd.dN1[n], cc = sd.dN2[n];
    const double dnx = Jinv(0, 0) * a + Jinv(0, 1) * b + Jinv(0, 2) * cc;
    const double dny = Jinv(1, 0) * a + Jinv(1, 1) * b + Jinv(1, 2) * cc;
    const double dnz = Jinv(2, 0) * a + Jinv(2, 1) * b + Jinv(2, 2) * cc;
    gradT(0) += dnx * t[n];
    gradT(1) += dny * t[n];
    gradT(2) += dnz * t[n];
  }
  return -(kt * gradT);
}

// ── Hex8 shape functions ───────────────────────────────────────────────────
struct Hex8Shape {
  std::array<double, 8> N, dN0, dN1, dN2;
};
[[nodiscard]] Hex8Shape hex8_shape(double xi, double eta, double zeta) {
  const std::array<double, 8> sx{-1, 1, 1, -1, -1, 1, 1, -1};
  const std::array<double, 8> sy{-1, -1, 1, 1, -1, -1, 1, 1};
  const std::array<double, 8> sz{-1, -1, -1, -1, 1, 1, 1, 1};
  Hex8Shape s;
  for (int i = 0; i < 8; ++i) {
    s.N[i] = 0.125 * (1 + sx[i] * xi) * (1 + sy[i] * eta) * (1 + sz[i] * zeta);
    s.dN0[i] = 0.125 * sx[i] * (1 + sy[i] * eta) * (1 + sz[i] * zeta);
    s.dN1[i] = 0.125 * sy[i] * (1 + sx[i] * xi)  * (1 + sz[i] * zeta);
    s.dN2[i] = 0.125 * sz[i] * (1 + sx[i] * xi)  * (1 + sy[i] * eta);
  }
  return s;
}
[[nodiscard]] std::vector<std::array<double, 4>> hex8_gauss() {
  const double g = 1.0 / std::sqrt(3.0);
  std::vector<std::array<double, 4>> v;
  v.reserve(8);
  for (double xi : {-g, g})
    for (double eta : {-g, g})
      for (double zeta : {-g, g})
        v.push_back({xi, eta, zeta, 1.0});
  return v;
}

// ── Tetra10 shape functions ────────────────────────────────────────────────
struct Tet10Shape {
  std::array<double, 10> N, dN0, dN1, dN2;
};
[[nodiscard]] Tet10Shape tet10_shape(double L1, double L2, double L3) {
  const double L4 = 1.0 - L1 - L2 - L3;
  Tet10Shape s;
  s.N = {L1 * (2 * L1 - 1), L2 * (2 * L2 - 1), L3 * (2 * L3 - 1),
         L4 * (2 * L4 - 1),
         4 * L1 * L2, 4 * L2 * L3, 4 * L1 * L3,
         4 * L1 * L4, 4 * L2 * L4, 4 * L3 * L4};
  s.dN0 = {4 * L1 - 1, 0, 0, -(4 * L4 - 1),
           4 * L2, 0, 4 * L3,
           4 * (L4 - L1), -4 * L2, -4 * L3};
  s.dN1 = {0, 4 * L2 - 1, 0, -(4 * L4 - 1),
           4 * L1, 4 * L3, 0,
           -4 * L1, 4 * (L4 - L2), -4 * L3};
  s.dN2 = {0, 0, 4 * L3 - 1, -(4 * L4 - 1),
           0, 4 * L2, 4 * L1,
           -4 * L1, -4 * L2, 4 * (L4 - L3)};
  return s;
}
[[nodiscard]] std::vector<std::array<double, 4>> tet10_gauss() {
  // 4-point Gauss for tetrahedra (degree-2 exact).  Reference volume = 1/6,
  // weights sum to 1/6.
  const double a = (5.0 - std::sqrt(5.0)) / 20.0;
  const double b = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
  const double w = 1.0 / 24.0;
  return {{a, a, a, w}, {b, a, a, w}, {a, b, a, w}, {a, a, b, w}};
}

// ── Hex20 shape functions (serendipity 20-node brick) ─────────────────────
// Node ordering (MSC NASTRAN CHEXA):
//   1..8   = corners at (±1,±1,±1) — see hex8_shape for the sign table.
//   9..12  = midnodes on the four bottom (zeta=-1) edges:
//              9 on (1,2)  [along xi]   10 on (2,3) [along eta]
//             11 on (3,4)  [along xi]   12 on (4,1) [along eta]
//   13..16 = midnodes on the four top (zeta=+1) edges, same xi/eta pattern.
//   17..20 = midnodes on the four vertical edges (1,5)(2,6)(3,7)(4,8) [along zeta].
struct Hex20Shape {
  std::array<double, 20> N{}, dN0{}, dN1{}, dN2{};
};

// edge-direction tag for midnode m (8..19): 0 = along xi, 1 = along eta,
// 2 = along zeta.  Used to pick the right (1 - α²) factor in the serendipity
// midnode formula.
static constexpr std::array<int, 12> kHex20EdgeAxis{
    0, 1, 0, 1,    // 9..12 (bottom)
    0, 1, 0, 1,    // 13..16 (top)
    2, 2, 2, 2,    // 17..20 (vertical)
};

[[nodiscard]] Hex20Shape hex20_shape(double xi, double eta, double zeta) {
  const std::array<double, 8> sx{-1, 1, 1, -1, -1, 1, 1, -1};
  const std::array<double, 8> sy{-1, -1, 1, 1, -1, -1, 1, 1};
  const std::array<double, 8> sz{-1, -1, -1, -1, 1, 1, 1, 1};
  // Midnode reference coordinates: signs for the two non-edge axes.
  // (the third coordinate, along the edge, is zero)
  // axis: 0=along xi → sy,sz fixed, xi free; 1=along eta → sx,sz fixed; 2=along zeta
  // For axis 0 (along xi), we use (sy, sz). For axis 1, (sx, sz). For axis 2, (sx, sy).
  // Below tables encode the two non-edge ±1 signs for each midnode 9..20.
  static const std::array<std::array<double, 2>, 12> mid_signs{{
      {-1, -1}, { 1, -1}, { 1, -1}, {-1, -1},   // 9..12 (bottom, axis 0/1/0/1)
      {-1,  1}, { 1,  1}, { 1,  1}, {-1,  1},   // 13..16 (top)
      {-1, -1}, { 1, -1}, { 1,  1}, {-1,  1},   // 17..20 (vertical, axis 2: sx,sy)
  }};

  Hex20Shape s;
  // First pass: midnodes (need them for the corner correction)
  for (int m = 0; m < 12; ++m) {
    const int axis = kHex20EdgeAxis[m];
    const double a = mid_signs[m][0];  // first non-edge sign
    const double b = mid_signs[m][1];  // second non-edge sign
    if (axis == 0) {
      // along xi: signs are (sy=a, sz=b); shape = ¼(1-ξ²)(1+a·η)(1+b·ζ)
      const double f = 0.25 * (1 - xi * xi) * (1 + a * eta) * (1 + b * zeta);
      s.N[8 + m] = f;
      s.dN0[8 + m] = 0.25 * (-2 * xi) * (1 + a * eta) * (1 + b * zeta);
      s.dN1[8 + m] = 0.25 * (1 - xi * xi) * a * (1 + b * zeta);
      s.dN2[8 + m] = 0.25 * (1 - xi * xi) * (1 + a * eta) * b;
    } else if (axis == 1) {
      // along eta: signs (sx=a, sz=b); shape = ¼(1+a·ξ)(1-η²)(1+b·ζ)
      const double f = 0.25 * (1 + a * xi) * (1 - eta * eta) * (1 + b * zeta);
      s.N[8 + m] = f;
      s.dN0[8 + m] = 0.25 * a * (1 - eta * eta) * (1 + b * zeta);
      s.dN1[8 + m] = 0.25 * (1 + a * xi) * (-2 * eta) * (1 + b * zeta);
      s.dN2[8 + m] = 0.25 * (1 + a * xi) * (1 - eta * eta) * b;
    } else {
      // along zeta: signs (sx=a, sy=b); shape = ¼(1+a·ξ)(1+b·η)(1-ζ²)
      const double f = 0.25 * (1 + a * xi) * (1 + b * eta) * (1 - zeta * zeta);
      s.N[8 + m] = f;
      s.dN0[8 + m] = 0.25 * a * (1 + b * eta) * (1 - zeta * zeta);
      s.dN1[8 + m] = 0.25 * (1 + a * xi) * b * (1 - zeta * zeta);
      s.dN2[8 + m] = 0.25 * (1 + a * xi) * (1 + b * eta) * (-2 * zeta);
    }
  }
  // Corners: N_i = ⅛(1+sx·ξ)(1+sy·η)(1+sz·ζ)(sx·ξ + sy·η + sz·ζ - 2)
  for (int i = 0; i < 8; ++i) {
    const double a = (1 + sx[i] * xi);
    const double b = (1 + sy[i] * eta);
    const double c = (1 + sz[i] * zeta);
    const double t = sx[i] * xi + sy[i] * eta + sz[i] * zeta - 2.0;
    s.N[i] = 0.125 * a * b * c * t;
    s.dN0[i] = 0.125 * (sx[i] * b * c * t + a * b * c * sx[i]);
    s.dN1[i] = 0.125 * (a * sy[i] * c * t + a * b * c * sy[i]);
    s.dN2[i] = 0.125 * (a * b * sz[i] * t + a * b * c * sz[i]);
  }
  return s;
}

[[nodiscard]] std::vector<std::array<double, 4>> hex20_gauss() {
  // 3×3×3 Gauss rule (27 points, exact for polynomials up to degree 5 in each
  // direction — covers the (2,2,2) integrand BᵀkB for hex20).
  const std::array<double, 3> g{-std::sqrt(0.6), 0.0, std::sqrt(0.6)};
  const std::array<double, 3> w{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
  std::vector<std::array<double, 4>> v;
  v.reserve(27);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        v.push_back({g[i], g[j], g[k], w[i] * w[j] * w[k]});
  return v;
}

// ── Transition (variable-noded) midnode redistribution ────────────────────
// When a midnode m on edge (a, b) is absent, fold its shape function into the
// two adjacent corners:  N_a += ½·N_m,  N_b += ½·N_m,  N_m := 0.  Same for
// every derivative.  This preserves partition-of-unity and degrades the edge
// interpolation from quadratic to linear, exactly the standard serendipity
// "transition element" formulation.
template <int N, typename Shape>
void redistribute_absent_midnodes(
    Shape &s, std::span<const bool> present,
    std::span<const std::array<int, 2>> edges /* size N - n_corners */) {
  for (size_t i = 0; i < edges.size(); ++i) {
    const int m = static_cast<int>(i) + (N - static_cast<int>(edges.size()));
    if (present[static_cast<size_t>(m)]) continue;
    const int a = edges[i][0];
    const int b = edges[i][1];
    auto fold = [&](std::array<double, N> &arr) {
      const double half = 0.5 * arr[static_cast<size_t>(m)];
      arr[static_cast<size_t>(a)] += half;
      arr[static_cast<size_t>(b)] += half;
      arr[static_cast<size_t>(m)] = 0.0;
    };
    fold(s.N);
    fold(s.dN0);
    fold(s.dN1);
    fold(s.dN2);
  }
}

// Tet10 edge → adjacent-corner table (indices into nodes_[10]).
static constexpr std::array<std::array<int, 2>, 6> kTet10Edges{{
    {0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3},
}};

// Hex20 edge → adjacent-corner table.
static constexpr std::array<std::array<int, 2>, 12> kHex20Edges{{
    {0, 1}, {1, 2}, {2, 3}, {3, 0},   // 9..12 (bottom)
    {4, 5}, {5, 6}, {6, 7}, {7, 4},   // 13..16 (top)
    {0, 4}, {1, 5}, {2, 6}, {3, 7},   // 17..20 (vertical)
}};

// ── Penta6 shape functions ────────────────────────────────────────────────
struct Penta6Shape {
  std::array<double, 6> N, dN0, dN1, dN2;
};
[[nodiscard]] Penta6Shape penta6_shape(double L1, double L2, double zeta) {
  const double L3 = 1.0 - L1 - L2;
  const double zm = 0.5 * (1 - zeta);
  const double zp = 0.5 * (1 + zeta);
  Penta6Shape s;
  s.N = {L1 * zm, L2 * zm, L3 * zm, L1 * zp, L2 * zp, L3 * zp};
  s.dN0 = {zm, 0, -zm, zp, 0, -zp};
  s.dN1 = {0, zm, -zm, 0, zp, -zp};
  s.dN2 = {-0.5 * L1, -0.5 * L2, -0.5 * L3,
           0.5 * L1,  0.5 * L2,  0.5 * L3};
  return s;
}
[[nodiscard]] std::vector<std::array<double, 4>> penta6_gauss() {
  // 3 triangle pts × 2 axial.  Triangle: centroidal 3-point at midsides,
  // weights 1/6.  Axial: 2-pt Gauss at ±1/√3, weights 1.
  const double g = 1.0 / std::sqrt(3.0);
  const double tw = 1.0 / 6.0;
  std::vector<std::array<double, 4>> v;
  v.reserve(6);
  const std::array<std::array<double, 2>, 3> tri{{
      {0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}}};
  for (double zeta : {-g, g})
    std::transform(tri.begin(), tri.end(), std::back_inserter(v),
                   [zeta, tw](const std::array<double, 2> &t) {
                     return std::array<double, 4>{t[0], t[1], zeta, tw};
                   });
  return v;
}

} // namespace

// ── ThermalHexa8 ─────────────────────────────────────────────────────────────

class ThermalHexa8 final : public ThermalElement {
public:
  ThermalHexa8(ElementId eid, PropertyId pid,
               std::array<NodeId, 8> nodes, const Model &model)
      : eid_(eid), pid_(pid), nodes_(nodes), model_(model) {
    const MaterialId mid = property_thermal_mid(model_, pid_);
    mat_ = resolve_thermal_material(model_, mid);
  }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] ElementType type() const noexcept override { return ElementType::CHEXA8; }
  [[nodiscard]] int num_nodes() const noexcept override { return 8; }
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), 8};
  }
  [[nodiscard]] Eigen::MatrixXd conductance_matrix() const override {
    return solid_conductance<8>(coords(), mat_.k_tensor, hex8_gauss(), hex8_shape);
  }
  [[nodiscard]] Eigen::VectorXd
  volumetric_heat_load(double q_vol) const override {
    return solid_vol_load<8>(coords(), q_vol, hex8_gauss(), hex8_shape);
  }
  [[nodiscard]] Eigen::VectorXd
  heat_flux(std::span<const double> t) const override {
    return solid_centroid_flux<8>(coords(), mat_.k_tensor, t, 0.0, 0.0, 0.0,
                                  hex8_shape);
  }

private:
  std::array<Vec3, 8> coords() const { return gather_nodes<8>(model_, node_ids()); }
  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, 8> nodes_;
  const Model &model_;
  ThermalMaterial mat_;
};

// ── ThermalTetra10 ───────────────────────────────────────────────────────────

class ThermalTetra10 final : public ThermalElement {
public:
  ThermalTetra10(ElementId eid, PropertyId pid,
                 std::array<NodeId, 10> nodes, const Model &model)
      : eid_(eid), pid_(pid), nodes_(nodes), model_(model) {
    const MaterialId mid = property_thermal_mid(model_, pid_);
    mat_ = resolve_thermal_material(model_, mid);
    for (int i = 0; i < 10; ++i)
      present_[static_cast<size_t>(i)] = (nodes_[static_cast<size_t>(i)].value != 0);
  }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] ElementType type() const noexcept override { return ElementType::CTETRA10; }
  [[nodiscard]] int num_nodes() const noexcept override { return 10; }
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), 10};
  }
  [[nodiscard]] Eigen::MatrixXd conductance_matrix() const override {
    return solid_conductance<10>(coords(), mat_.k_tensor, tet10_gauss(),
                                 make_shape());
  }
  [[nodiscard]] Eigen::VectorXd volumetric_heat_load(double q_vol) const override {
    return solid_vol_load<10>(coords(), q_vol, tet10_gauss(), make_shape());
  }
  [[nodiscard]] Eigen::VectorXd
  heat_flux(std::span<const double> t) const override {
    return solid_centroid_flux<10>(coords(), mat_.k_tensor, t,
                                   0.25, 0.25, 0.25, make_shape());
  }

private:
  std::array<Vec3, 10> coords() const { return gather_nodes<10>(model_, node_ids()); }
  // Returns a shape callable that applies midnode redistribution for any
  // absent midnodes (variable-noded CTETRA support).
  std::function<Tet10Shape(double, double, double)> make_shape() const {
    return [present = present_](double L1, double L2, double L3) {
      Tet10Shape s = tet10_shape(L1, L2, L3);
      redistribute_absent_midnodes<10>(s, std::span<const bool>(present),
                                       std::span<const std::array<int, 2>>(kTet10Edges));
      return s;
    };
  }

  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, 10> nodes_;
  const Model &model_;
  ThermalMaterial mat_;
  std::array<bool, 10> present_{};
};

// ── ThermalHexa20 ────────────────────────────────────────────────────────────
// Serendipity 20-node hex.  Supports the full 20-node element AND any subset
// (8-20 nodes) where one or more midside nodes are omitted: absent midnodes
// have their shape contribution folded into the two adjacent corners, so the
// edge interpolation along an omitted edge degrades from quadratic to linear.

class ThermalHexa20 final : public ThermalElement {
public:
  ThermalHexa20(ElementId eid, PropertyId pid,
                std::array<NodeId, 20> nodes, const Model &model)
      : eid_(eid), pid_(pid), nodes_(nodes), model_(model) {
    const MaterialId mid = property_thermal_mid(model_, pid_);
    mat_ = resolve_thermal_material(model_, mid);
    for (int i = 0; i < 20; ++i)
      present_[static_cast<size_t>(i)] = (nodes_[static_cast<size_t>(i)].value != 0);
  }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] ElementType type() const noexcept override { return ElementType::CHEXA20; }
  [[nodiscard]] int num_nodes() const noexcept override { return 20; }
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), 20};
  }
  [[nodiscard]] Eigen::MatrixXd conductance_matrix() const override {
    return solid_conductance<20>(coords(), mat_.k_tensor, hex20_gauss(),
                                 make_shape());
  }
  [[nodiscard]] Eigen::VectorXd volumetric_heat_load(double q_vol) const override {
    return solid_vol_load<20>(coords(), q_vol, hex20_gauss(), make_shape());
  }
  [[nodiscard]] Eigen::VectorXd
  heat_flux(std::span<const double> t) const override {
    return solid_centroid_flux<20>(coords(), mat_.k_tensor, t,
                                   0.0, 0.0, 0.0, make_shape());
  }

private:
  std::array<Vec3, 20> coords() const { return gather_nodes<20>(model_, node_ids()); }
  std::function<Hex20Shape(double, double, double)> make_shape() const {
    return [present = present_](double xi, double eta, double zeta) {
      Hex20Shape s = hex20_shape(xi, eta, zeta);
      redistribute_absent_midnodes<20>(s, std::span<const bool>(present),
                                       std::span<const std::array<int, 2>>(kHex20Edges));
      return s;
    };
  }

  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, 20> nodes_;
  const Model &model_;
  ThermalMaterial mat_;
  std::array<bool, 20> present_{};
};

// ── ThermalPenta6 ────────────────────────────────────────────────────────────

class ThermalPenta6 final : public ThermalElement {
public:
  ThermalPenta6(ElementId eid, PropertyId pid,
                std::array<NodeId, 6> nodes, const Model &model)
      : eid_(eid), pid_(pid), nodes_(nodes), model_(model) {
    const MaterialId mid = property_thermal_mid(model_, pid_);
    mat_ = resolve_thermal_material(model_, mid);
  }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] ElementType type() const noexcept override { return ElementType::CPENTA6; }
  [[nodiscard]] int num_nodes() const noexcept override { return 6; }
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), 6};
  }
  [[nodiscard]] Eigen::MatrixXd conductance_matrix() const override {
    return solid_conductance<6>(coords(), mat_.k_tensor, penta6_gauss(), penta6_shape);
  }
  [[nodiscard]] Eigen::VectorXd volumetric_heat_load(double q_vol) const override {
    return solid_vol_load<6>(coords(), q_vol, penta6_gauss(), penta6_shape);
  }
  [[nodiscard]] Eigen::VectorXd
  heat_flux(std::span<const double> t) const override {
    return solid_centroid_flux<6>(coords(), mat_.k_tensor, t,
                                  1.0 / 3.0, 1.0 / 3.0, 0.0, penta6_shape);
  }

private:
  std::array<Vec3, 6> coords() const { return gather_nodes<6>(model_, node_ids()); }
  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, 6> nodes_;
  const Model &model_;
  ThermalMaterial mat_;
};

// ── ThermalTria3 ────────────────────────────────────────────────────────────
// Constant-gradient triangular shell, in-plane conduction integrated through
// PSHELL thickness.

class ThermalTria3 final : public ThermalElement {
public:
  ThermalTria3(ElementId eid, PropertyId pid,
               std::array<NodeId, 3> nodes, const Model &model)
      : eid_(eid), pid_(pid), nodes_(nodes), model_(model) {
    mat_ = resolve_thermal_material(model_, property_thermal_mid(model_, pid_));
    thickness_ = shell_thickness(model_.property(pid_));
    if (thickness_ <= 0.0)
      throw SolverError(std::format(
          "Thermal CTRIA3 {}: PSHELL thickness must be > 0", eid_.value));
    compute_geometry();
  }

  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] ElementType type() const noexcept override { return ElementType::CTRIA3; }
  [[nodiscard]] int num_nodes() const noexcept override { return 3; }
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), 3};
  }

  [[nodiscard]] Eigen::MatrixXd conductance_matrix() const override {
    return area_ * thickness_ * B_.transpose() * k2_ * B_;
  }

  [[nodiscard]] Eigen::VectorXd
  volumetric_heat_load(double q_vol) const override {
    // Treat q_vol as volumetric (W/m³); multiply by element volume = A·t.
    return Eigen::VectorXd::Constant(3, q_vol * area_ * thickness_ / 3.0);
  }

  [[nodiscard]] Eigen::VectorXd
  heat_flux(std::span<const double> t) const override {
    Eigen::Vector3d T;
    T << t[0], t[1], t[2];
    return -(k2_ * (B_ * T));  // 2-D flux in element-local (e1, e2) frame
  }

private:
  void compute_geometry() {
    const auto pts = gather_nodes<3>(model_, node_ids());
    const ShellFrame f = build_shell_frame(std::span<const Vec3>(pts.data(), 3));
    e1_in_basic_ = f.e1;
    e2_in_basic_ = f.e2;
    const double x1 = f.xy[0][0], y1 = f.xy[0][1];
    const double x2 = f.xy[1][0], y2 = f.xy[1][1];
    const double x3 = f.xy[2][0], y3 = f.xy[2][1];
    const double two_A = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    area_ = 0.5 * std::abs(two_A);
    if (area_ < 1e-20)
      throw SolverError(std::format(
          "Thermal CTRIA3 {} degenerate area", eid_.value));
    B_.resize(2, 3);
    const double inv = 1.0 / two_A;
    B_(0, 0) = (y2 - y3) * inv;
    B_(0, 1) = (y3 - y1) * inv;
    B_(0, 2) = (y1 - y2) * inv;
    B_(1, 0) = (x3 - x2) * inv;
    B_(1, 1) = (x1 - x3) * inv;
    B_(1, 2) = (x2 - x1) * inv;
    k2_ = project_k_to_plane(mat_.k_tensor, e1_in_basic_, e2_in_basic_);
  }

  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, 3> nodes_;
  const Model &model_;
  ThermalMaterial mat_;
  double thickness_{0.0};
  double area_{0.0};
  Vec3 e1_in_basic_{0,0,0};
  Vec3 e2_in_basic_{0,0,0};
  Eigen::MatrixXd B_;     // 2 × 3
  Eigen::Matrix2d k2_;
};

// ── ThermalQuad4 ────────────────────────────────────────────────────────────
// 4-node bilinear isoparametric shell, in-plane conduction with 2×2 Gauss.
// Nodes are projected into the mean plane built from corners 1, 2, 4.

class ThermalQuad4 final : public ThermalElement {
public:
  ThermalQuad4(ElementId eid, PropertyId pid,
               std::array<NodeId, 4> nodes, const Model &model)
      : eid_(eid), pid_(pid), nodes_(nodes), model_(model) {
    mat_ = resolve_thermal_material(model_, property_thermal_mid(model_, pid_));
    thickness_ = shell_thickness(model_.property(pid_));
    if (thickness_ <= 0.0)
      throw SolverError(std::format(
          "Thermal CQUAD4 {}: PSHELL thickness must be > 0", eid_.value));
    cache_frame();
  }

  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] ElementType type() const noexcept override { return ElementType::CQUAD4; }
  [[nodiscard]] int num_nodes() const noexcept override { return 4; }
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), 4};
  }

  [[nodiscard]] Eigen::MatrixXd conductance_matrix() const override {
    Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero(4, 4);
    const double g = 1.0 / std::sqrt(3.0);
    for (double xi : {-g, g}) for (double eta : {-g, g}) {
      auto [B, det] = jacobian_B(xi, eta);
      Ke += thickness_ * (B.transpose() * k2_ * B) * std::abs(det);
    }
    return Ke;
  }

  [[nodiscard]] Eigen::VectorXd
  volumetric_heat_load(double q_vol) const override {
    Eigen::VectorXd p = Eigen::VectorXd::Zero(4);
    const double g = 1.0 / std::sqrt(3.0);
    for (double xi : {-g, g}) for (double eta : {-g, g}) {
      auto [B, det] = jacobian_B(xi, eta);
      (void)B;
      const std::array<double, 4> N{
          0.25 * (1 - xi) * (1 - eta), 0.25 * (1 + xi) * (1 - eta),
          0.25 * (1 + xi) * (1 + eta), 0.25 * (1 - xi) * (1 + eta)};
      for (int n = 0; n < 4; ++n)
        p(n) += N[n] * q_vol * thickness_ * std::abs(det);
    }
    return p;
  }

  [[nodiscard]] Eigen::VectorXd
  heat_flux(std::span<const double> t) const override {
    auto [B, det] = jacobian_B(0.0, 0.0);
    (void)det;
    Eigen::Vector4d T;
    T << t[0], t[1], t[2], t[3];
    return -(k2_ * (B * T));
  }

private:
  void cache_frame() {
    const auto pts = gather_nodes<4>(model_, node_ids());
    const ShellFrame f = build_shell_frame(std::span<const Vec3>(pts.data(), 4));
    for (int i = 0; i < 4; ++i) {
      xy_[i][0] = f.xy[i][0];
      xy_[i][1] = f.xy[i][1];
    }
    k2_ = project_k_to_plane(mat_.k_tensor, f.e1, f.e2);
  }

  std::pair<Eigen::MatrixXd, double>
  jacobian_B(double xi, double eta) const {
    const std::array<double, 4> dNdxi{
        -0.25 * (1 - eta),  0.25 * (1 - eta),
         0.25 * (1 + eta), -0.25 * (1 + eta)};
    const std::array<double, 4> dNdeta{
        -0.25 * (1 - xi), -0.25 * (1 + xi),
         0.25 * (1 + xi),  0.25 * (1 - xi)};
    Eigen::Matrix2d J = Eigen::Matrix2d::Zero();
    for (int n = 0; n < 4; ++n) {
      J(0, 0) += dNdxi[n]  * xy_[n][0];
      J(0, 1) += dNdxi[n]  * xy_[n][1];
      J(1, 0) += dNdeta[n] * xy_[n][0];
      J(1, 1) += dNdeta[n] * xy_[n][1];
    }
    const double det = J.determinant();
    const Eigen::Matrix2d Jinv = J.inverse();
    Eigen::MatrixXd B(2, 4);
    for (int n = 0; n < 4; ++n) {
      B(0, n) = Jinv(0, 0) * dNdxi[n] + Jinv(0, 1) * dNdeta[n];
      B(1, n) = Jinv(1, 0) * dNdxi[n] + Jinv(1, 1) * dNdeta[n];
    }
    return {B, det};
  }

  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, 4> nodes_;
  const Model &model_;
  ThermalMaterial mat_;
  double thickness_{0.0};
  std::array<std::array<double, 2>, 4> xy_{};
  Eigen::Matrix2d k2_;
};

// ── Factory ──────────────────────────────────────────────────────────────────

std::unique_ptr<ThermalElement>
make_thermal_element(const ElementData &data, const Model &model) {
  switch (data.type) {
  case ElementType::CTETRA4: {
    std::array<NodeId, 4> n{data.nodes[0], data.nodes[1], data.nodes[2],
                            data.nodes[3]};
    return std::make_unique<ThermalTetra4>(data.id, data.pid, n, model);
  }
  case ElementType::CHEXA8: {
    std::array<NodeId, 8> n{data.nodes[0], data.nodes[1], data.nodes[2],
                            data.nodes[3], data.nodes[4], data.nodes[5],
                            data.nodes[6], data.nodes[7]};
    return std::make_unique<ThermalHexa8>(data.id, data.pid, n, model);
  }
  case ElementType::CTETRA10: {
    // Variable-noded: data.nodes is always size 10 from the parser, with
    // NodeId{0} placeholders for any omitted midnodes.
    std::array<NodeId, 10> n{NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0},
                             NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0},
                             NodeId{0}, NodeId{0}};
    for (size_t i = 0; i < std::min<size_t>(10, data.nodes.size()); ++i)
      n[i] = data.nodes[i];
    return std::make_unique<ThermalTetra10>(data.id, data.pid, n, model);
  }
  case ElementType::CHEXA20: {
    std::array<NodeId, 20> n{
        NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0},
        NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0},
        NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0},
        NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0}, NodeId{0}};
    for (size_t i = 0; i < std::min<size_t>(20, data.nodes.size()); ++i)
      n[i] = data.nodes[i];
    return std::make_unique<ThermalHexa20>(data.id, data.pid, n, model);
  }
  case ElementType::CPENTA6: {
    std::array<NodeId, 6> n{data.nodes[0], data.nodes[1], data.nodes[2],
                            data.nodes[3], data.nodes[4], data.nodes[5]};
    return std::make_unique<ThermalPenta6>(data.id, data.pid, n, model);
  }
  case ElementType::CBAR:
  case ElementType::CBEAM: {
    std::array<NodeId, 2> n{data.nodes[0], data.nodes[1]};
    return std::make_unique<ThermalLine>(data.id, data.type, data.pid, n, model);
  }
  case ElementType::CTRIA3: {
    std::array<NodeId, 3> n{data.nodes[0], data.nodes[1], data.nodes[2]};
    return std::make_unique<ThermalTria3>(data.id, data.pid, n, model);
  }
  case ElementType::CQUAD4: {
    std::array<NodeId, 4> n{data.nodes[0], data.nodes[1], data.nodes[2],
                            data.nodes[3]};
    return std::make_unique<ThermalQuad4>(data.id, data.pid, n, model);
  }
  default:
    return nullptr;  // unsupported (CQUAD4/CTRIA3 deferred, CHBDY handled elsewhere)
  }
}

} // namespace vibestran
