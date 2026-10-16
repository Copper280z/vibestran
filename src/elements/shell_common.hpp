#pragma once
// src/elements/shell_common.hpp
// Internal helpers shared by the continuum-mechanics-based shell elements
// (shell_quad4.cpp, shell_tria3.cpp): vector utilities, the mapping of
// covariant strain rows to a local Cartesian frame, the section at a point,
// and the material direction from THETA/MCID.

#include "elements/shell_section.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <format>
#include <numbers>
#include <utility>

namespace vibestran::shell_detail {

using V3 = Eigen::Vector3d;

inline V3 to_eigen(const Vec3 &v) { return {v.x, v.y, v.z}; }

inline V3 unit(const V3 &v, const char *element, ElementId eid, const char *what) {
  const double n = v.norm();
  if (!(n > 0.0) || !std::isfinite(n))
    throw SolverError(std::format("{} {}: degenerate geometry ({} has zero length)",
                                  element, eid.value, what));
  return v / n;
}

/// skew(a) b = a x b.
inline Eigen::Matrix3d skew(const V3 &a) {
  Eigen::Matrix3d S;
  S << 0.0, -a.z(), a.y(), a.z(), 0.0, -a.x(), -a.y(), a.x(), 0.0;
  return S;
}

/// Contravariant in-plane base vectors of {g_r, g_s, n}.
inline std::pair<V3, V3> dual_basis(const V3 &gr, const V3 &gs, const V3 &n) {
  Eigen::Matrix3d G;
  G.col(0) = gr;
  G.col(1) = gs;
  G.col(2) = n;
  const Eigen::Matrix3d Gi = G.inverse();
  return {Gi.row(0).transpose(), Gi.row(1).transpose()};
}

/// Local Cartesian frame at a point and the map from covariant strain
/// components to it: C(i, a) = g^i . e_a with {g^i} the contravariant basis
/// of {g_r, g_s, g_t}.  e3 is the surface normal, e1 the element output x
/// axis projected onto the tangent plane.
struct LocalFrame {
  V3 e1, e2, e3;
  Eigen::Matrix3d C;
  double dA{0.0};
};

inline LocalFrame local_frame(const V3 &gr, const V3 &gs, const V3 &gt,
                              const V3 &element_x) {
  LocalFrame f;
  const V3 nrm = gr.cross(gs);
  f.dA = nrm.norm();
  f.e3 = nrm / f.dA;
  f.e1 = (element_x - element_x.dot(f.e3) * f.e3).normalized();
  f.e2 = f.e3.cross(f.e1);
  Eigen::Matrix3d G;
  G.col(0) = gr;
  G.col(1) = gs;
  G.col(2) = gt;
  const Eigen::Matrix3d Gi = G.inverse(); // rows are g^r, g^s, g^t
  for (int i = 0; i < 3; ++i) {
    f.C(i, 0) = Gi.row(i).dot(f.e1);
    f.C(i, 1) = Gi.row(i).dot(f.e2);
    f.C(i, 2) = Gi.row(i).dot(f.e3);
  }
  return f;
}

/// sum_ij P(i, a) Q(j, b) E_ij for the symmetric covariant tensor E with
/// rows (rr, ss, rs, rt, st) and E_tt = 0.
template <class RowT>
RowT cartesian_pair(const Eigen::Matrix3d &P, const Eigen::Matrix3d &Q, int a, int b,
                    const RowT &rr, const RowT &ss, const RowT &rs, const RowT &rt,
                    const RowT &st) {
  return P(0, a) * Q(0, b) * rr + P(1, a) * Q(1, b) * ss +
         (P(0, a) * Q(1, b) + P(1, a) * Q(0, b)) * rs +
         (P(0, a) * Q(2, b) + P(2, a) * Q(0, b)) * rt +
         (P(1, a) * Q(2, b) + P(2, a) * Q(1, b)) * st;
}

/// Cartesian component eps_ab from the covariant rows (rr, ss, rs, rt, st)
/// (tensor components; multiply by 2 for engineering shear).
template <class RowT>
RowT cartesian(const LocalFrame &f, int a, int b, const RowT &rr, const RowT &ss,
               const RowT &rs, const RowT &rt, const RowT &st) {
  return cartesian_pair(f.C, f.C, a, b, rr, ss, rs, rt, st);
}

/// First-order change of the metric through the thickness.  The base
/// vectors at thickness coordinate t are g_r + t x_b,r, g_s + t x_b,s and
/// x_b, so dC/dt = (-G^-1 dG/dt G^-1) projected on the frame, and
/// det G(t) = det G(0) (1 + t tr(G^-1 dG/dt)).
struct MetricRate {
  Eigen::Matrix3d C1;  ///< dC/dt
  double trace{0.0};   ///< tr(G^-1 dG/dt)
};

inline MetricRate metric_rate(const LocalFrame &f, const V3 &gr, const V3 &gs,
                              const V3 &gt, const V3 &gt_r, const V3 &gt_s) {
  Eigen::Matrix3d G, dG = Eigen::Matrix3d::Zero();
  G.col(0) = gr;
  G.col(1) = gs;
  G.col(2) = gt;
  dG.col(0) = gt_r;
  dG.col(1) = gt_s;
  const Eigen::Matrix3d Gi = G.inverse();
  const Eigen::Matrix3d dGi = -Gi * dG * Gi;
  MetricRate m;
  for (int i = 0; i < 3; ++i) {
    m.C1(i, 0) = dGi.row(i).dot(f.e1);
    m.C1(i, 1) = dGi.row(i).dot(f.e2);
    m.C1(i, 2) = dGi.row(i).dot(f.e3);
  }
  m.trace = (Gi * dG).trace();
  return m;
}

/// The section at a point: scaled for the local thickness (PSHELL grid
/// thicknesses) and rotated into the point's frame.
inline ShellSection point_section(const ShellSection &nominal, bool scale,
                                  double thickness, const V3 &material_dir,
                                  const LocalFrame &f) {
  const ShellSection base =
      scale ? nominal.scaled_thickness(thickness / nominal.thickness) : nominal;
  const V3 m = material_dir - material_dir.dot(f.e3) * f.e3;
  return base.rotated(std::atan2(m.dot(f.e2), m.dot(f.e1)));
}

/// Material x direction of a shell element: THETA (degrees) from the G1-G2
/// side, or the MCID x axis, projected onto the plane normal to `n`.
inline V3 material_direction(const Model &model, const ElementData &data,
                             const char *element, const V3 &n, const V3 &g1,
                             const V3 &g2, const V3 &centre) {
  if (data.mcid.has_value()) {
    const auto it = model.coord_systems.find(*data.mcid);
    if (it == model.coord_systems.end())
      throw SolverError(std::format("{} {}: MCID {} is not defined", element,
                                    data.id.value, data.mcid->value));
    const Mat3 R =
        rotation_matrix(it->second, Vec3{centre.x(), centre.y(), centre.z()});
    V3 ax(R(0, 0), R(1, 0), R(2, 0));
    ax -= ax.dot(n) * n;
    return unit(ax, element, data.id, "MCID x axis projected on the element");
  }
  V3 side = g2 - g1;
  side -= side.dot(n) * n;
  side = unit(side, element, data.id, "G1-G2 side");
  const double th = data.theta.value_or(0.0) * std::numbers::pi / 180.0;
  return std::cos(th) * side + std::sin(th) * n.cross(side);
}

/// Grid thickness of corner `i`: PSHELL grid thickness (TFLAG = 1 relative)
/// or the nominal section thickness.
inline double corner_thickness(const ElementData &data, int i, double nominal,
                               bool is_pshell, bool &scaled) {
  double t = nominal;
  const auto k = static_cast<std::size_t>(i);
  if (is_pshell && k < data.corner_thickness.size() &&
      data.corner_thickness[k].has_value()) {
    t = (data.tflag == 1) ? *data.corner_thickness[k] * nominal
                          : *data.corner_thickness[k];
    scaled = true;
  }
  return t;
}

} // namespace vibestran::shell_detail
