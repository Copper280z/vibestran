// src/elements/shell_tria3.cpp
// MITC3+ CTRIA3 with stabilised drill rotations; see shell_tria3.hpp.

#include "elements/shell_tria3.hpp"
#include "elements/shell_directors.hpp"
#include "shell_common.hpp"
#include <cmath>
#include <format>

namespace vibestran {

namespace {

using namespace shell_detail;
constexpr int NE = 20; // 18 grid DOFs + 2 bubble rotations
using Row = Eigen::Matrix<double, 1, NE>;
using B3 = Eigen::Matrix<double, 3, NE>;
using B2 = Eigen::Matrix<double, 2, NE>;
using KE = Eigen::Matrix<double, NE, NE>;

constexpr const char *kName = "CTRIA3";

// h1 = 1 - r - s, h2 = r, h3 = s.
constexpr std::array<double, 3> kHr{-1.0, 1.0, 0.0};
constexpr std::array<double, 3> kHs{-1.0, 0.0, 1.0};
inline std::array<double, 3> hfun(double r, double s) { return {1.0 - r - s, r, s}; }

// Dunavant 7-point rule (degree 5) on the unit triangle; weights sum to 1/2.
struct QuadPt {
  double r, s, w;
};
const std::array<QuadPt, 7> &quadrature() {
  static const std::array<QuadPt, 7> pts = [] {
    const double a1 = 0.059715871789770, b1 = 0.470142064105115,
                 w1 = 0.132394152788506;
    const double a2 = 0.797426985353087, b2 = 0.101286507323456,
                 w2 = 0.125939180544827;
    return std::array<QuadPt, 7>{{{1.0 / 3.0, 1.0 / 3.0, 0.5 * 0.225},
                                  {b1, b1, 0.5 * w1},
                                  {a1, b1, 0.5 * w1},
                                  {b1, a1, 0.5 * w1},
                                  {b2, b2, 0.5 * w2},
                                  {a2, b2, 0.5 * w2},
                                  {b2, a2, 0.5 * w2}}};
  }();
  return pts;
}

struct Geometry {
  std::array<V3, 3> x;
  std::array<V3, 3> V;
  std::array<double, 3> a{};  // director lengths (t, or t / (V . n) at a SNORM)
  std::array<double, 3> t{};  // shell thicknesses (normal to the element)
  V3 W4;          // a4 V4 = (a1 V1 + a2 V2 + a3 V3) / 3
  V3 b1, b2;      // bubble rotation axes, orthogonal to W4
  V3 n, e1;       // element normal and output x axis
  V3 gr, gs;      // constant covariant base vectors
  V3 offset;
  V3 material_dir;
  ShellSection section;
  bool scale_thickness{false};
};

Geometry build_geometry(const ElementData &data, const Model &model) {
  Geometry g;
  std::array<V3, 3> X;
  for (int i = 0; i < 3; ++i)
    X[static_cast<size_t>(i)] =
        to_eigen(model.node(data.nodes[static_cast<size_t>(i)]).position);
  g.n = unit((X[1] - X[0]).cross(X[2] - X[0]), kName, data.id, "element normal");
  g.e1 = unit((X[1] - X[0]) - (X[1] - X[0]).dot(g.n) * g.n, kName, data.id, "G1-G2 side");
  g.offset = data.zoffs * g.n;
  for (int i = 0; i < 3; ++i)
    g.x[static_cast<size_t>(i)] = X[static_cast<size_t>(i)] + g.offset;
  g.gr = g.x[1] - g.x[0];
  g.gs = g.x[2] - g.x[0];

  g.section = build_shell_section(model, data.pid, data.id, kName);
  const bool is_pshell = std::holds_alternative<PShell>(model.property(data.pid));
  g.W4 = V3::Zero();
  for (int i = 0; i < 3; ++i) {
    g.V[static_cast<size_t>(i)] = shell_grid_director(model, data, i, g.n);
    const double t =
        corner_thickness(data, i, g.section.thickness, is_pshell, g.scale_thickness);
    if (!(t > 0.0))
      throw SolverError(std::format("{} {}: non-positive thickness at grid {}", kName,
                                    data.id.value, data.nodes[static_cast<size_t>(i)].value));
    // At a SNORM the thickness stays normal to the element (see
    // shell_quad4.cpp); averaged directors keep length t.
    g.t[static_cast<size_t>(i)] = t;
    g.a[static_cast<size_t>(i)] =
        model.snorms.contains(data.nodes[static_cast<size_t>(i)])
            ? t / g.V[static_cast<size_t>(i)].dot(g.n)
            : t;
    g.W4 += g.a[static_cast<size_t>(i)] * g.V[static_cast<size_t>(i)] / 3.0;
  }
  const V3 w4 = g.W4.normalized();
  g.b1 = (g.e1 - g.e1.dot(w4) * w4).normalized();
  g.b2 = w4.cross(g.b1);

  g.material_dir = material_direction(model, data, kName, g.n, X[0], X[1],
                                      (X[0] + X[1] + X[2]) / 3.0);
  return g;
}

/// f_i = h_i - f4/3 (i = 1..3) and f4 = 27 r s (1 - r - s), with
/// derivatives [LLB Eq. 7].
struct BubbleFns {
  std::array<double, 4> f, fr, fs;
};
BubbleFns bubble(double r, double s) {
  const auto h = hfun(r, s);
  const double f4 = 27.0 * r * s * (1.0 - r - s);
  const double f4r = 27.0 * s * (1.0 - 2.0 * r - s);
  const double f4s = 27.0 * r * (1.0 - r - 2.0 * s);
  BubbleFns b;
  for (int i = 0; i < 3; ++i) {
    b.f[static_cast<size_t>(i)] = h[static_cast<size_t>(i)] - f4 / 3.0;
    b.fr[static_cast<size_t>(i)] = kHr[static_cast<size_t>(i)] - f4r / 3.0;
    b.fs[static_cast<size_t>(i)] = kHs[static_cast<size_t>(i)] - f4s / 3.0;
  }
  b.f[3] = f4;
  b.fr[3] = f4r;
  b.fs[3] = f4s;
  return b;
}

/// Row of vec . u_b with u_b = 1/2 [sum a_i w_i (theta_i x V_i) + w4 (theta_4 x W4)].
Row rotation_row(const Geometry &g, const std::array<double, 4> &w, const V3 &vec) {
  Row row = Row::Zero();
  for (int k = 0; k < 3; ++k)
    row.segment<3>(6 * k + 3) = (0.5 * g.a[static_cast<size_t>(k)] * w[static_cast<size_t>(k)] *
                                 g.V[static_cast<size_t>(k)].cross(vec))
                                    .transpose();
  const V3 c4 = 0.5 * w[3] * g.W4.cross(vec);
  row(18) = c4.dot(g.b1);
  row(19) = c4.dot(g.b2);
  return row;
}

Row translation_row(const std::array<double, 3> &w, const V3 &vec) {
  Row row = Row::Zero();
  for (int k = 0; k < 3; ++k)
    row.segment<3>(6 * k) = w[static_cast<size_t>(k)] * vec.transpose();
  return row;
}

/// x_b = 1/2 sum a_i h_i V_i (the bubble terms cancel) and its derivatives.
V3 xb_at(const Geometry &g, const std::array<double, 3> &h) {
  V3 v = V3::Zero();
  for (int k = 0; k < 3; ++k)
    v += 0.5 * g.a[static_cast<size_t>(k)] * h[static_cast<size_t>(k)] * g.V[static_cast<size_t>(k)];
  return v;
}

/// Displacement-based covariant transverse shear strains at (r, s).
std::pair<Row, Row> shear_rows(const Geometry &g, double r, double s) {
  const BubbleFns b = bubble(r, s);
  const V3 xb = xb_at(g, hfun(r, s));
  const Row ert = 0.5 * (rotation_row(g, b.f, g.gr) + translation_row(kHr, xb));
  const Row est = 0.5 * (rotation_row(g, b.f, g.gs) + translation_row(kHs, xb));
  return {ert, est};
}

struct PointB {
  B3 Be{B3::Zero()};
  B3 Bk{B3::Zero()};
  B2 Bg{B2::Zero()};
  LocalFrame frame;
  double thickness{0.0};
  double m{0.0}; // volume factor dV = (1 + m z) dA dz (m = 2H)
  Eigen::Matrix2d curv{Eigen::Matrix2d::Zero()}; // director curvature b_ab
};

PointB point_b(const Geometry &g, double r, double s) {
  PointB p;
  const auto h = hfun(r, s);
  const BubbleFns b = bubble(r, s);
  double thick = 0.0;
  for (int k = 0; k < 3; ++k)
    thick += h[static_cast<size_t>(k)] * g.t[static_cast<size_t>(k)];
  p.thickness = thick;
  const V3 xb = xb_at(g, h), xb_r = xb_at(g, kHr), xb_s = xb_at(g, kHs);

  // Membrane: constant-strain triangle (MITC3+ uses the displacement-based
  // membrane of the flat 3-node element).
  const Row mrr = translation_row(kHr, g.gr), mss = translation_row(kHs, g.gs);
  const Row mrs = 0.5 * (translation_row(kHs, g.gr) + translation_row(kHr, g.gs));

  // Bending, first order in t, with the bubble-enriched rotations.
  const Row brr = rotation_row(g, b.fr, g.gr) + translation_row(kHr, xb_r);
  const Row bss = rotation_row(g, b.fs, g.gs) + translation_row(kHs, xb_s);
  const Row brs = 0.5 * (rotation_row(g, b.fs, g.gr) + rotation_row(g, b.fr, g.gs) +
                         translation_row(kHs, xb_r) + translation_row(kHr, xb_s));

  // MITC3+ assumed transverse shear [LLB Eqs. 16-17, Table 1].
  constexpr double d = 1.0e-4;
  const auto A = shear_rows(g, 1.0 / 6.0, 2.0 / 3.0);
  const auto B = shear_rows(g, 2.0 / 3.0, 1.0 / 6.0);
  const auto C = shear_rows(g, 1.0 / 6.0, 1.0 / 6.0);
  const auto D = shear_rows(g, 1.0 / 3.0 + d, 1.0 / 3.0 - 2.0 * d);
  const auto E = shear_rows(g, 1.0 / 3.0 - 2.0 * d, 1.0 / 3.0 + d);
  const auto F = shear_rows(g, 1.0 / 3.0 + d, 1.0 / 3.0 + d);
  const Row chat = (F.first - D.first) - (F.second - E.second);
  const Row ert = 2.0 / 3.0 * (B.first - 0.5 * B.second) +
                  1.0 / 3.0 * (C.first + C.second) + 1.0 / 3.0 * chat * (3.0 * s - 1.0);
  const Row est = 2.0 / 3.0 * (A.second - 0.5 * A.first) +
                  1.0 / 3.0 * (C.first + C.second) + 1.0 / 3.0 * chat * (1.0 - 3.0 * r);

  p.frame = local_frame(g.gr, g.gs, xb, g.e1);
  const Row Z = Row::Zero();
  p.Be.row(0) = cartesian(p.frame, 0, 0, mrr, mss, mrs, ert, est);
  p.Be.row(1) = cartesian(p.frame, 1, 1, mrr, mss, mrs, ert, est);
  p.Be.row(2) = 2.0 * cartesian(p.frame, 0, 1, mrr, mss, mrs, ert, est);
  // d(eps)/dz = (2 / t) d(eps)/dt (see shell_quad4.cpp); Nastran kappa is
  // its negative.
  const double k = -2.0 / thick;
  // d(eps_ab)/dt: the first-order covariant strain plus the mid-surface
  // strain carried through the metric's change across the thickness (the
  // z / R terms of a curved shell; zero on a flat element with normal
  // directors), as in shell_quad4.cpp.
  const MetricRate rate = metric_rate(p.frame, g.gr, g.gs, xb, xb_r, xb_s);
  p.m = 2.0 / thick * rate.trace;
  p.curv = director_curvature(p.frame, xb_r, xb_s, 0.5 * thick);
  const auto cart_t = [&](int a, int b) -> Row {
    const Eigen::Matrix3d &C = p.frame.C;
    return cartesian_pair(C, C, a, b, brr, bss, brs, Z, Z) +
           cartesian_pair(rate.C1, C, a, b, mrr, mss, mrs, ert, est) +
           cartesian_pair(C, rate.C1, a, b, mrr, mss, mrs, ert, est);
  };
  p.Bk.row(0) = k * cart_t(0, 0);
  p.Bk.row(1) = k * cart_t(1, 1);
  p.Bk.row(2) = 2.0 * k * cart_t(0, 1);
  p.Bg.row(0) = 2.0 * cartesian(p.frame, 0, 2, mrr, mss, mrs, ert, est);
  p.Bg.row(1) = 2.0 * cartesian(p.frame, 1, 2, mrr, mss, mrs, ert, est);
  return p;
}

ShellSection section_at(const Geometry &g, const PointB &p) {
  return point_section(g.section, g.scale_thickness, p.thickness, g.material_dir,
                       p.frame)
      .with_volume_factor(p.m);
}

/// Thermal resultants per degree, with the curved-geometry correction of the
/// CQUAD4 (see thermal_resultants in shell_quad4.cpp).
struct ThermalResultants {
  V3 N, M;
  Eigen::Vector2d Q;
};
ThermalResultants thermal_resultants(const Geometry &g, const ShellSection &sec,
                                     const PointB &p) {
  ThermalResultants t{sec.thermal_N, sec.thermal_M, Eigen::Vector2d::Zero()};
  if (sec.A.determinant() == 0.0)
    return t;
  const V3 free_strain = sec.A.fullPivLu().solve(sec.thermal_N);
  if (free_strain.isZero(0.0))
    return t;
  // The free expansion u = E (x - x0), anisotropic for MAT2/MAT8/PCOMP.
  const Eigen::Matrix3d E =
      free_expansion_gradient(free_strain, p.curv, p.frame.e1, p.frame.e2, p.frame.e3);
  Eigen::Matrix<double, NE, 1> uT = Eigen::Matrix<double, NE, 1>::Zero();
  for (int k = 0; k < 3; ++k)
    uT.segment<3>(6 * k) = E * (g.x[static_cast<size_t>(k)] - g.x[0]);
  const V3 re = p.Be * uT - free_strain;
  const V3 rk = p.Bk * uT;
  t.N += sec.A * re + sec.B * rk;
  t.M += sec.B * re + sec.D * rk;
  t.Q = sec.T * (p.Bg * uT);
  return t;
}

/// Full 20 x 20 stiffness (grid DOFs and bubble rotations), in reference-
/// surface DOFs.
KE full_stiffness(const Geometry &g) {
  KE K = KE::Zero();
  double shear = 0.0;
  for (const QuadPt &q : quadrature()) {
    const PointB p = point_b(g, q.r, q.s);
    const ShellSection sec = section_at(g, p);
    Eigen::Matrix<double, 6, 6> C;
    C << sec.A, sec.B, sec.B, sec.D;
    Eigen::Matrix<double, 6, NE> Bmk;
    Bmk << p.Be, p.Bk;
    const double w = q.w * p.frame.dA;
    K += w * (Bmk.transpose() * C * Bmk + p.Bg.transpose() * sec.T * p.Bg);
    shear += w * sec.A(2, 2);
  }
  // Drill rotation constraint (Hughes & Brezzi 1989).  MITC3+ has no
  // stiffness for rotations about the director, so each corner's drill
  // rotation theta . n is tied to the in-plane rotation omega of the
  // translation field (constant on the CST).  It vanishes for rigid
  // motions and constant-strain states.
  const auto [gr_up, gs_up] = dual_basis(g.gr, g.gs, g.n);
  Row omega = Row::Zero();
  for (int k = 0; k < 3; ++k) {
    const V3 grad = kHr[static_cast<size_t>(k)] * gr_up + kHs[static_cast<size_t>(k)] * gs_up;
    omega.segment<3>(6 * k) = 0.5 * g.n.cross(grad).transpose();
  }
  for (int k = 0; k < 3; ++k) {
    Row diff = -omega;
    diff.segment<3>(6 * k + 3) += g.n.transpose();
    K += ShellTria3::kDrillStabilisation * shear / 3.0 * diff.transpose() * diff;
  }
  return K;
}

Eigen::Matrix<double, 18, 18> offset_transform(const Geometry &g) {
  Eigen::Matrix<double, 18, 18> W = Eigen::Matrix<double, 18, 18>::Identity();
  if (g.offset.squaredNorm() != 0.0)
    for (int k = 0; k < 3; ++k)
      W.block<3, 3>(6 * k, 6 * k + 3) = -skew(g.offset);
  return W;
}

std::array<NodeId, 3> tria_nodes(const ElementData &data) {
  if (data.nodes.size() != 3)
    throw SolverError(std::format("CTRIA3 {} needs 3 grids, got {}", data.id.value,
                                  data.nodes.size()));
  return {data.nodes[0], data.nodes[1], data.nodes[2]};
}

/// Thermal load in the full 20 DOFs (reference-surface).
Eigen::Matrix<double, NE, 1> full_thermal(const Geometry &g,
                                          std::span<const double> temps, double t_ref) {
  Eigen::Matrix<double, NE, 1> f = Eigen::Matrix<double, NE, 1>::Zero();
  for (const QuadPt &q : quadrature()) {
    const auto h = hfun(q.r, q.s);
    double T = 0.0;
    for (int k = 0; k < 3; ++k)
      T += h[static_cast<size_t>(k)] *
           (temps.empty() ? t_ref : temps[static_cast<size_t>(k)]);
    const double dT = T - t_ref;
    if (dT == 0.0)
      continue;
    const PointB p = point_b(g, q.r, q.s);
    const ShellSection sec = section_at(g, p);
    const ThermalResultants tr = thermal_resultants(g, sec, p);
    f += q.w * p.frame.dA * dT *
         (p.Be.transpose() * tr.N + p.Bk.transpose() * tr.M + p.Bg.transpose() * tr.Q);
  }
  return f;
}

} // namespace

ShellTria3::ShellTria3(const ElementData &data, const Model &model)
    : eid_(data.id), nodes_(tria_nodes(data)), model_(model), data_(data) {}

LocalKe ShellTria3::stiffness_matrix() const {
  const Geometry g = build_geometry(data_, model_);
  const KE K = full_stiffness(g);
  // Condense the bubble rotations.
  const Eigen::Matrix<double, 18, 18> Kcc = K.topLeftCorner<18, 18>();
  const Eigen::Matrix<double, 18, 2> Kcb = K.topRightCorner<18, 2>();
  const Eigen::Matrix2d Kbb = K.bottomRightCorner<2, 2>();
  const Eigen::Matrix<double, 18, 18> Kc = Kcc - Kcb * Kbb.inverse() * Kcb.transpose();
  const Eigen::Matrix<double, 18, 18> W = offset_transform(g);
  return W.transpose() * Kc * W;
}

LocalKe ShellTria3::mass_matrix() const {
  const Geometry g = build_geometry(data_, model_);
  Eigen::Matrix<double, 18, 18> M = Eigen::Matrix<double, 18, 18>::Zero();
  for (const QuadPt &q : quadrature()) {
    const auto h = hfun(q.r, q.s);
    const PointB p = point_b(g, q.r, q.s);
    const ShellSection sec =
        g.scale_thickness ? g.section.scaled_thickness(p.thickness / g.section.thickness)
                          : g.section;
    Eigen::Matrix<double, 3, 18> Nm = Eigen::Matrix<double, 3, 18>::Zero();
    Eigen::Matrix<double, 3, 18> Nb = Eigen::Matrix<double, 3, 18>::Zero();
    for (int k = 0; k < 3; ++k) {
      Nm.block<3, 3>(0, 6 * k) = h[static_cast<size_t>(k)] * Eigen::Matrix3d::Identity();
      // Per unit normal distance z: a point at height z lies z a/t along V.
      Nb.block<3, 3>(0, 6 * k + 3) = -h[static_cast<size_t>(k)] *
                                     g.a[static_cast<size_t>(k)] /
                                     g.t[static_cast<size_t>(k)] *
                                     skew(g.V[static_cast<size_t>(k)]);
    }
    M += q.w * p.frame.dA *
         (sec.mass_per_area * Nm.transpose() * Nm +
          sec.mass_first_moment * (Nm.transpose() * Nb + Nb.transpose() * Nm) +
          sec.mass_second_moment * Nb.transpose() * Nb);
  }
  const Eigen::Matrix<double, 18, 18> W = offset_transform(g);
  return W.transpose() * M * W;
}

LocalFe ShellTria3::thermal_load(std::span<const double> temperatures,
                                 double t_ref) const {
  const Geometry g = build_geometry(data_, model_);
  const KE K = full_stiffness(g);
  const Eigen::Matrix<double, NE, 1> f = full_thermal(g, temperatures, t_ref);
  const Eigen::Matrix<double, 18, 2> Kcb = K.topRightCorner<18, 2>();
  const Eigen::Matrix2d Kbb = K.bottomRightCorner<2, 2>();
  const Eigen::Matrix<double, 18, 1> fc =
      f.head<18>() - Kcb * Kbb.inverse() * f.tail<2>();
  return offset_transform(g).transpose() * fc;
}

std::vector<EqIndex> ShellTria3::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> result;
  result.reserve(NUM_DOFS);
  for (NodeId nid : nodes_) {
    const auto &blk = dof_map.block(nid);
    for (int d = 0; d < 6; ++d)
      result.push_back(blk.eq[d]);
  }
  return result;
}

std::array<ShellPointResponse, 4>
ShellTria3::recover(std::span<const double> global_displacements,
                    std::span<const double> temperatures, double t_ref) const {
  if (global_displacements.size() != NUM_DOFS)
    throw SolverError(std::format("CTRIA3 {}: expected 18 displacements, got {}",
                                  eid_.value, global_displacements.size()));
  const Geometry g = build_geometry(data_, model_);
  Eigen::Matrix<double, 18, 1> ug;
  for (int i = 0; i < NUM_DOFS; ++i)
    ug(i) = global_displacements[static_cast<size_t>(i)];
  Eigen::Matrix<double, NE, 1> u = Eigen::Matrix<double, NE, 1>::Zero();
  u.head<18>() = offset_transform(g) * ug;
  // Bubble rotations from the condensed equations.
  const KE K = full_stiffness(g);
  const Eigen::Matrix<double, NE, 1> f = full_thermal(g, temperatures, t_ref);
  const Eigen::Matrix2d Kbb = K.bottomRightCorner<2, 2>();
  u.tail<2>() = Kbb.inverse() * (f.tail<2>() - K.bottomLeftCorner<2, 18>() * u.head<18>());

  auto at = [&](double r, double s) {
    const PointB p = point_b(g, r, s);
    const ShellSection sec = section_at(g, p);
    const auto h = hfun(r, s);
    double T = 0.0;
    for (int k = 0; k < 3; ++k)
      T += h[static_cast<size_t>(k)] *
           (temperatures.empty() ? t_ref : temperatures[static_cast<size_t>(k)]);
    const double dT = T - t_ref;
    const ThermalResultants tr = thermal_resultants(g, sec, p);
    ShellPointResponse out;
    out.membrane_strain = p.Be * u;
    out.curvature = p.Bk * u;
    out.transverse_shear_strain = p.Bg * u;
    out.membrane_resultant =
        sec.A * out.membrane_strain + sec.B * out.curvature - tr.N * dT;
    out.bending_moment = sec.B * out.membrane_strain + sec.D * out.curvature - tr.M * dT;
    out.transverse_shear = sec.T * out.transverse_shear_strain - tr.Q * dT;
    out.thickness = p.thickness;
    return out;
  };
  return {at(1.0 / 3.0, 1.0 / 3.0), at(0.0, 0.0), at(1.0, 0.0), at(0.0, 1.0)};
}

} // namespace vibestran
