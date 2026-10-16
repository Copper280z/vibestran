// src/elements/shell_quad4.cpp
// MITC4+/D CQUAD4; see include/elements/shell_quad4.hpp for the formulation
// and references ([KBZ] = Ko, Bathe & Zhang 2025).

#include "elements/shell_quad4.hpp"
#include "elements/shell_directors.hpp"
#include <cmath>
#include <format>
#include <numbers>

namespace vibestran {

namespace {

using V3 = Eigen::Vector3d;
using Row = Eigen::Matrix<double, 1, 24>;
using B3 = Eigen::Matrix<double, 3, 24>;
using B2 = Eigen::Matrix<double, 2, 24>;

constexpr double kSqrt3 = std::numbers::sqrt3;
constexpr double kGauss = 1.0 / std::numbers::sqrt3;

// Nastran grid order in natural coordinates: G1(-1,-1) G2(1,-1) G3(1,1)
// G4(-1,1).
constexpr std::array<double, 4> kXi{-1.0, 1.0, 1.0, -1.0};
constexpr std::array<double, 4> kEta{-1.0, -1.0, 1.0, 1.0};

double hfun(int i, double r, double s) {
  return 0.25 * (1.0 + kXi[i] * r) * (1.0 + kEta[i] * s);
}
double hfun_r(int i, double s) { return 0.25 * kXi[i] * (1.0 + kEta[i] * s); }
double hfun_s(int i, double r) { return 0.25 * kEta[i] * (1.0 + kXi[i] * r); }

V3 to_eigen(const Vec3 &v) { return {v.x, v.y, v.z}; }

V3 unit(const V3 &v, ElementId eid, const char *what) {
  const double n = v.norm();
  if (!(n > 0.0) || !std::isfinite(n))
    throw SolverError(std::format("CQUAD4 {}: degenerate geometry ({} has zero length)",
                                  eid.value, what));
  return v / n;
}

// Skew matrix: skew(a) b = a x b.
Eigen::Matrix3d skew(const V3 &a) {
  Eigen::Matrix3d S;
  S << 0.0, -a.z(), a.y(), a.z(), 0.0, -a.x(), -a.y(), a.x(), 0.0;
  return S;
}

/// Contravariant in-plane base vectors of {g_r, g_s, n}.
std::pair<V3, V3> dual_basis(const V3 &gr, const V3 &gs, const V3 &n) {
  Eigen::Matrix3d G;
  G.col(0) = gr;
  G.col(1) = gs;
  G.col(2) = n;
  const Eigen::Matrix3d Gi = G.inverse();
  return {Gi.row(0).transpose(), Gi.row(1).transpose()};
}

/// Edge of the element for the drill-membrane strain, traversed G_a -> G_b
/// counter-clockwise about V^D.
struct DrillEdge {
  int a{0}, b{0};
  bool r_edge{false}; // edge at s = +-1 (only the r derivative is kept)
  double side{0.0};   // s (r_edge) or r (otherwise) of the edge
  V3 w;               // ((x_b - x_a) x V^D) / 8
  V3 gr, gs;          // covariant base vectors at the edge midpoint
};

struct Geometry {
  std::array<V3, 4> x;   // reference-surface grid positions (offset applied)
  std::array<V3, 4> V;   // directors
  std::array<double, 4> a{};
  V3 xr, xs, xd, n;      // characteristic vectors, n = V^D
  V3 mr, ms;             // dual basis of plane P
  V3 e1, e2;             // Nastran element frame (e3 = n)
  V3 offset;             // ZOFFS n
  double j0{0.0};
  // MITC4+/D membrane constants (KBZ Eq. 25).
  double cr{0.0}, cs{0.0}, aA{0.0}, aB{0.0}, aC{0.0}, aD{0.0}, aE{0.0};
  double n1{0.0}, n2{0.0}, n3{0.0}, n4{0.0}, n5{0.0};
  double m1{0.0}, m2{0.0}, m3{0.0}, m4{0.0}, m5{0.0};
  std::array<Row, 5> tied{}; // e_rr(A), e_rr(B), e_ss(C), e_ss(D), e_rs(E)
  std::array<DrillEdge, 4> edges{};
  V3 material_dir;
  ShellSection section;  // nominal, material frame
  double t_nominal{0.0};
  bool scale_thickness{false};

  [[nodiscard]] V3 gr_at(double s) const { return xr + s * xd; }
  [[nodiscard]] V3 gs_at(double r) const { return xs + r * xd; }
  [[nodiscard]] double jac(double r, double s) const {
    return gr_at(s).cross(gs_at(r)).dot(n);
  }
};

/// Displacement-based mid-surface covariant membrane strain rows at (r, s).
std::array<Row, 3> membrane_rows(const Geometry &g, double r, double s) {
  const V3 gr = g.gr_at(s), gs = g.gs_at(r);
  std::array<Row, 3> e{Row::Zero(), Row::Zero(), Row::Zero()};
  for (int k = 0; k < 4; ++k) {
    const double hr = hfun_r(k, s), hs = hfun_s(k, r);
    e[0].segment<3>(6 * k) = hr * gr.transpose();
    e[1].segment<3>(6 * k) = hs * gs.transpose();
    e[2].segment<3>(6 * k) = 0.5 * (hs * gr + hr * gs).transpose();
  }
  return e;
}

Geometry build_geometry(const ElementData &data, const Model &model) {
  Geometry g;
  const ElementId eid = data.id;
  std::array<V3, 4> X;
  for (int i = 0; i < 4; ++i)
    X[static_cast<size_t>(i)] =
        to_eigen(model.node(data.nodes[static_cast<size_t>(i)]).position);

  // Nastran element frame from the grid positions.
  const V3 d13 = unit(X[2] - X[0], eid, "diagonal G1-G3");
  const V3 d24 = unit(X[3] - X[1], eid, "diagonal G2-G4");
  g.n = unit(d13.cross(d24), eid, "element normal");
  g.e1 = unit(d13 - d24, eid, "element x axis");
  g.e2 = g.n.cross(g.e1);

  g.offset = data.zoffs * g.n;
  for (int i = 0; i < 4; ++i)
    g.x[static_cast<size_t>(i)] = X[static_cast<size_t>(i)] + g.offset;

  g.xr = g.xs = g.xd = V3::Zero();
  for (int i = 0; i < 4; ++i) {
    const V3 &xi = g.x[static_cast<size_t>(i)];
    g.xr += 0.25 * kXi[i] * xi;
    g.xs += 0.25 * kEta[i] * xi;
    g.xd += 0.25 * kXi[i] * kEta[i] * xi;
  }
  std::tie(g.mr, g.ms) = dual_basis(g.xr, g.xs, g.n);
  g.j0 = g.jac(0.0, 0.0);
  for (int i = 0; i < 4; ++i)
    if (g.jac(kXi[i], kEta[i]) <= 0.0)
      throw SolverError(std::format(
          "CQUAD4 {}: non-positive Jacobian at grid {} (concave or inverted "
          "element)",
          eid.value, data.nodes[static_cast<size_t>(i)].value));

  // Directors: shared grid directors (shell_directors.hpp), seeded with the
  // unit normal of this element's surface at each corner.
  for (int i = 0; i < 4; ++i) {
    const V3 own = unit(g.gr_at(kEta[i]).cross(g.gs_at(kXi[i])), eid, "corner normal");
    g.V[static_cast<size_t>(i)] = shell_grid_director(model, data, i, own);
  }
  // Section and thickness.
  g.section = build_shell_section(model, data.pid, eid, "CQUAD4");
  g.t_nominal = g.section.thickness;
  const bool is_pshell = std::holds_alternative<PShell>(model.property(data.pid));
  for (int i = 0; i < 4; ++i) {
    double t = g.t_nominal;
    if (is_pshell && static_cast<size_t>(i) < data.corner_thickness.size() &&
        data.corner_thickness[static_cast<size_t>(i)].has_value()) {
      const double ti = *data.corner_thickness[static_cast<size_t>(i)];
      t = (data.tflag == 1) ? ti * g.t_nominal : ti;
      g.scale_thickness = true;
    }
    if (!(t > 0.0))
      throw SolverError(std::format("CQUAD4 {}: non-positive thickness at grid {}",
                                    eid.value, data.nodes[static_cast<size_t>(i)].value));
    g.a[static_cast<size_t>(i)] = t;
  }

  // Material x direction: THETA from the G1-G2 side, or the MCID x axis,
  // projected onto the element plane.
  if (data.mcid.has_value()) {
    const auto it = model.coord_systems.find(*data.mcid);
    if (it == model.coord_systems.end())
      throw SolverError(std::format("CQUAD4 {}: MCID {} is not defined",
                                    eid.value, data.mcid->value));
    const V3 centre = 0.25 * (X[0] + X[1] + X[2] + X[3]);
    const Mat3 R = rotation_matrix(it->second, Vec3{centre.x(), centre.y(), centre.z()});
    V3 ax(R(0, 0), R(1, 0), R(2, 0));
    ax -= ax.dot(g.n) * g.n;
    g.material_dir = unit(ax, eid, "MCID x axis projected on the element");
  } else {
    V3 side = X[1] - X[0];
    side -= side.dot(g.n) * g.n;
    side = unit(side, eid, "G1-G2 side");
    const double th = data.theta.value_or(0.0) * std::numbers::pi / 180.0;
    g.material_dir = std::cos(th) * side + std::sin(th) * g.n.cross(side);
  }

  // MITC4+/D membrane constants [KBZ Eq. 25, Appendix B].
  g.cr = g.xd.dot(g.mr);
  g.cs = g.xd.dot(g.ms);
  const double d = g.cr * g.cr + g.cs * g.cs - 1.0;
  if (std::abs(d) < 1e-8)
    throw SolverError(std::format(
        "CQUAD4 {}: element is too distorted for the MITC4+ membrane field "
        "(d = {:.3g})",
        eid.value, d));
  g.aA = g.cr * (g.cr - 1.0) / (2.0 * d);
  g.aB = g.cr * (g.cr + 1.0) / (2.0 * d);
  g.aC = g.cs * (g.cs - 1.0) / (2.0 * d);
  g.aD = g.cs * (g.cs + 1.0) / (2.0 * d);
  g.aE = 2.0 * g.cr * g.cs / d;
  // g_i^j = (centre covariant base i) . (contravariant base j at the point).
  auto coeff = [&](double r, double s) {
    const auto [gr_up, gs_up] = dual_basis(g.gr_at(s), g.gs_at(r), g.n);
    struct C { double rr, rs, sr, ss; };
    return C{g.xr.dot(gr_up), g.xr.dot(gs_up), g.xs.dot(gr_up), g.xs.dot(gs_up)};
  };
  const auto Ab = coeff(0.0, kGauss), Bb = coeff(0.0, -kGauss);
  const auto Cb = coeff(kGauss, 0.0), Db = coeff(-kGauss, 0.0);
  g.n1 = 0.5 * (Ab.rr * Ab.rr - Bb.rr * Bb.rr);
  g.n2 = 0.5 * (Ab.rs * Ab.rs - Bb.rs * Bb.rs);
  g.n3 = 0.5 * (Ab.rr * Ab.rr + Bb.rr * Bb.rr);
  g.n4 = 0.5 * (Ab.rr * Ab.rs + Bb.rr * Bb.rs);
  g.n5 = 0.5 * (Ab.rr * Ab.rs - Bb.rr * Bb.rs);
  g.m1 = 0.5 * (Cb.ss * Cb.ss - Db.ss * Db.ss);
  g.m2 = 0.5 * (Cb.sr * Cb.sr - Db.sr * Db.sr);
  g.m3 = 0.5 * (Cb.ss * Cb.ss + Db.ss * Db.ss);
  g.m4 = 0.5 * (Cb.sr * Cb.ss + Db.sr * Db.ss);
  g.m5 = 0.5 * (Cb.sr * Cb.ss - Db.sr * Db.ss);
  // Tied displacement-based strains at A(0,1), B(0,-1), C(1,0), D(-1,0),
  // E(0,0).
  g.tied[0] = membrane_rows(g, 0.0, 1.0)[0];
  g.tied[1] = membrane_rows(g, 0.0, -1.0)[0];
  g.tied[2] = membrane_rows(g, 1.0, 0.0)[1];
  g.tied[3] = membrane_rows(g, -1.0, 0.0)[1];
  g.tied[4] = membrane_rows(g, 0.0, 0.0)[2];

  // Drill edges, counter-clockwise G1->G2->G3->G4->G1.
  const std::array<std::array<int, 2>, 4> pairs{{{0, 1}, {1, 2}, {2, 3}, {3, 0}}};
  const std::array<bool, 4> r_edge{true, false, true, false};
  const std::array<double, 4> side{-1.0, 1.0, 1.0, -1.0};
  for (int e = 0; e < 4; ++e) {
    DrillEdge &de = g.edges[static_cast<size_t>(e)];
    de.a = pairs[static_cast<size_t>(e)][0];
    de.b = pairs[static_cast<size_t>(e)][1];
    de.r_edge = r_edge[static_cast<size_t>(e)];
    de.side = side[static_cast<size_t>(e)];
    de.w = (g.x[static_cast<size_t>(de.b)] - g.x[static_cast<size_t>(de.a)]).cross(g.n) /
           8.0;
    const double rm = de.r_edge ? 0.0 : de.side, sm = de.r_edge ? de.side : 0.0;
    de.gr = g.gr_at(sm);
    de.gs = g.gs_at(rm);
  }
  return g;
}

/// Strain-displacement matrices and frame at one point.
struct PointB {
  B3 Be{B3::Zero()}; // reference-surface strain [exx, eyy, gxy]
  B3 Bk{B3::Zero()}; // curvature, Nastran sign
  B2 Bg{B2::Zero()}; // transverse shear strain [gxz, gyz]
  double dA{0.0};
  double thickness{0.0};
  V3 e1, e2, e3;
};

/// MITC4+/D assumed membrane strain (centre-basis covariant, KBZ Eq. 25)
/// plus the drill-membrane strain (KBZ Eqs. 16-21, see header for the sign),
/// transformed to the local covariant basis at (r, s).
std::array<Row, 3> assumed_membrane(const Geometry &g, double r, double s) {
  const double lam = g.j0 / g.jac(r, s);
  // S (6x5) maps the tied strains to the strain coefficients.
  Eigen::Matrix<double, 6, 5> S;
  S << 0.5 - g.aA, 0.5 - g.aB, -g.aC, -g.aD, -g.aE, //
      -g.aA, -g.aB, 0.5 - g.aC, 0.5 - g.aD, -g.aE,   //
      0.0, 0.0, 0.0, 0.0, 1.0,                       //
      0.5, -0.5, 0.0, 0.0, 0.0,                      //
      0.0, 0.0, 0.5, -0.5, 0.0,                      //
      g.aA, g.aB, g.aC, g.aD, g.aE;
  Eigen::Matrix<double, 3, 6> R;
  R << 1.0 / lam + kSqrt3 * g.n1 * s, kSqrt3 * g.n2 * s, 2.0 * kSqrt3 * g.n5 * s,
      g.n3 * s, g.n4 * s, g.n1 * s / kSqrt3, //
      kSqrt3 * g.m2 * r, 1.0 / lam + kSqrt3 * g.m1 * r, 2.0 * kSqrt3 * g.m5 * r,
      g.m4 * r, g.m3 * r, g.m1 * r / kSqrt3, //
      0.0, 0.0, 1.0 / lam, 0.0, 0.0, 0.0;
  R *= lam;
  const Eigen::Matrix<double, 3, 5> RS = R * S;
  std::array<Row, 3> bar{Row::Zero(), Row::Zero(), Row::Zero()};
  for (int c = 0; c < 3; ++c)
    for (int k = 0; k < 5; ++k)
      bar[static_cast<size_t>(c)] += RS(c, k) * g.tied[static_cast<size_t>(k)];

  // Drill-membrane strain in the centre basis.  theta^D_k = theta_k . V^D;
  // each edge contributes (theta_b - theta_a) times a geometric factor.
  for (const DrillEdge &de : g.edges) {
    double crr = 0.0, css = 0.0, crs = 0.0;
    if (de.r_edge) {
      const double hr = -r * (1.0 + de.side * s); // d/dr of (1-r^2)(1+side s)/2
      crr = hr * de.gr.dot(de.w);
      crs = 0.5 * hr * de.gs.dot(de.w);
    } else {
      const double hs = -s * (1.0 + de.side * r);
      css = hs * de.gs.dot(de.w);
      crs = 0.5 * hs * de.gr.dot(de.w);
    }
    const std::array<double, 3> cc{crr, css, crs};
    for (int c = 0; c < 3; ++c) {
      const double v = lam * cc[static_cast<size_t>(c)];
      bar[static_cast<size_t>(c)].segment<3>(6 * de.b + 3) += v * g.n.transpose();
      bar[static_cast<size_t>(c)].segment<3>(6 * de.a + 3) -= v * g.n.transpose();
    }
  }

  // Centre basis -> local basis (KBZ Eq. 25 Q matrix / Eq. 21).
  const double grr = 1.0 + g.cr * s, grs = g.cs * s, gsr = g.cr * r,
               gss = 1.0 + g.cs * r;
  Eigen::Matrix3d Q;
  Q << grr * grr, grs * grs, 2.0 * grr * grs, //
      gsr * gsr, gss * gss, 2.0 * gsr * gss,   //
      grr * gsr, grs * gss, grr * gss + grs * gsr;
  std::array<Row, 3> out{Row::Zero(), Row::Zero(), Row::Zero()};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k)
      out[static_cast<size_t>(i)] += Q(i, k) * bar[static_cast<size_t>(k)];
  return out;
}

/// x_b = 1/2 sum a_k h_k V_k and its derivatives at (r, s).
struct DirectorField {
  V3 xb, xb_r, xb_s;
};
DirectorField director_field(const Geometry &g, double r, double s) {
  DirectorField f{V3::Zero(), V3::Zero(), V3::Zero()};
  for (int k = 0; k < 4; ++k) {
    const V3 v = 0.5 * g.a[static_cast<size_t>(k)] * g.V[static_cast<size_t>(k)];
    f.xb += hfun(k, r, s) * v;
    f.xb_r += hfun_r(k, s) * v;
    f.xb_s += hfun_s(k, r) * v;
  }
  return f;
}

/// Row of g . u_b,(.) with u_b = 1/2 sum a_k w_k (theta_k x V_k), for the
/// weights w_k: g . (theta x V) = theta . (V x g).
Row rotation_row(const Geometry &g, const std::array<double, 4> &w, const V3 &vec) {
  Row row = Row::Zero();
  for (int k = 0; k < 4; ++k) {
    const V3 c = 0.5 * g.a[static_cast<size_t>(k)] * w[static_cast<size_t>(k)] *
                 g.V[static_cast<size_t>(k)].cross(vec);
    row.segment<3>(6 * k + 3) = c.transpose();
  }
  return row;
}

/// Row of vec . u_m,(.) for the shape-function weights w_k.
Row translation_row(const std::array<double, 4> &w, const V3 &vec) {
  Row row = Row::Zero();
  for (int k = 0; k < 4; ++k)
    row.segment<3>(6 * k) = w[static_cast<size_t>(k)] * vec.transpose();
  return row;
}

/// Covariant transverse shear strain rows e_rt and e_st (displacement
/// based, t = 0) at (r, s).
std::pair<Row, Row> shear_rows(const Geometry &g, double r, double s) {
  const DirectorField f = director_field(g, r, s);
  std::array<double, 4> h{}, hr{}, hs{};
  for (int k = 0; k < 4; ++k) {
    h[static_cast<size_t>(k)] = hfun(k, r, s);
    hr[static_cast<size_t>(k)] = hfun_r(k, s);
    hs[static_cast<size_t>(k)] = hfun_s(k, r);
  }
  const Row ert = 0.5 * (rotation_row(g, h, g.gr_at(s)) + translation_row(hr, f.xb));
  const Row est = 0.5 * (rotation_row(g, h, g.gs_at(r)) + translation_row(hs, f.xb));
  return {ert, est};
}

PointB point_b(const Geometry &g, double r, double s) {
  PointB p;
  const V3 gr = g.gr_at(s), gs = g.gs_at(r);
  const DirectorField f = director_field(g, r, s);
  std::array<double, 4> hr{}, hs{};
  double thick = 0.0;
  for (int k = 0; k < 4; ++k) {
    hr[static_cast<size_t>(k)] = hfun_r(k, s);
    hs[static_cast<size_t>(k)] = hfun_s(k, r);
    thick += hfun(k, r, s) * g.a[static_cast<size_t>(k)];
  }
  p.thickness = thick;

  // Covariant strain rows (local basis): membrane (assumed), bending first
  // order term, transverse shear (MITC4 assumed).
  const std::array<Row, 3> em = assumed_membrane(g, r, s);
  const Row b_rr = rotation_row(g, hr, gr) + translation_row(hr, f.xb_r);
  const Row b_ss = rotation_row(g, hs, gs) + translation_row(hs, f.xb_s);
  const Row b_rs = 0.5 * (rotation_row(g, hs, gr) + rotation_row(g, hr, gs) +
                          translation_row(hs, f.xb_r) + translation_row(hr, f.xb_s));
  const Row ertA = shear_rows(g, 0.0, 1.0).first;
  const Row ertB = shear_rows(g, 0.0, -1.0).first;
  const Row estC = shear_rows(g, 1.0, 0.0).second;
  const Row estD = shear_rows(g, -1.0, 0.0).second;
  const Row ert = 0.5 * (1.0 + s) * ertA + 0.5 * (1.0 - s) * ertB;
  const Row est = 0.5 * (1.0 + r) * estC + 0.5 * (1.0 - r) * estD;

  // Local Cartesian frame: surface normal and the element x axis projected.
  p.e3 = gr.cross(gs).normalized();
  p.e1 = (g.e1 - g.e1.dot(p.e3) * p.e3).normalized();
  p.e2 = p.e3.cross(p.e1);
  p.dA = gr.cross(gs).norm();

  // Contravariant basis of {g_r, g_s, g_t = x_b}; C(i, a) = g^i . e_a.
  Eigen::Matrix3d G;
  G.col(0) = gr;
  G.col(1) = gs;
  G.col(2) = f.xb;
  const Eigen::Matrix3d Gi = G.inverse(); // rows are g^r, g^s, g^t
  Eigen::Matrix3d C;
  for (int i = 0; i < 3; ++i) {
    C(i, 0) = Gi.row(i).dot(p.e1);
    C(i, 1) = Gi.row(i).dot(p.e2);
    C(i, 2) = Gi.row(i).dot(p.e3);
  }
  // Cartesian component eps_ab from covariant rows (rr, ss, rs, rt, st).
  auto cart = [&](int a, int b, const Row &rr, const Row &ss, const Row &rs,
                  const Row &rt, const Row &st) -> Row {
    return C(0, a) * C(0, b) * rr + C(1, a) * C(1, b) * ss +
           (C(0, a) * C(1, b) + C(1, a) * C(0, b)) * rs +
           (C(0, a) * C(2, b) + C(2, a) * C(0, b)) * rt +
           (C(1, a) * C(2, b) + C(2, a) * C(1, b)) * st;
  };
  const Row Z = Row::Zero();
  p.Be.row(0) = cart(0, 0, em[0], em[1], em[2], ert, est);
  p.Be.row(1) = cart(1, 1, em[0], em[1], em[2], ert, est);
  p.Be.row(2) = 2.0 * cart(0, 1, em[0], em[1], em[2], ert, est);
  // Physical strain gradient d(eps)/dz = (2 / a) d(eps)/dt; Nastran kappa
  // is its negative.
  const double k = -2.0 / thick;
  p.Bk.row(0) = k * cart(0, 0, b_rr, b_ss, b_rs, Z, Z);
  p.Bk.row(1) = k * cart(1, 1, b_rr, b_ss, b_rs, Z, Z);
  p.Bk.row(2) = 2.0 * k * cart(0, 1, b_rr, b_ss, b_rs, Z, Z);
  p.Bg.row(0) = 2.0 * cart(0, 2, em[0], em[1], em[2], ert, est);
  p.Bg.row(1) = 2.0 * cart(1, 2, em[0], em[1], em[2], ert, est);

  return p;
}

/// Thermal resultants per unit temperature rise at a point.
///
/// The section gives [N_T; M_T] for a flat shell.  On a curved element the
/// directors are inextensible, so the free isotropic expansion u = alpha dT x
/// is not in the element's displacement space: its best approximation,
/// u_m = alpha dT x_m with zero rotations, also carries the element's
/// geometric curvature and director-tilt strains.  Those are added as
/// thermal generalised strains (the element's own strain operator applied
/// to that field, minus the flat part), so free expansion is stress free.
/// alpha is the mean in-plane free thermal strain per degree; the
/// correction is exactly zero on flat elements with normal directors.
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
  const double alpha = 0.5 * (free_strain(0) + free_strain(1));
  if (alpha == 0.0)
    return t;
  Eigen::Matrix<double, 24, 1> uT = Eigen::Matrix<double, 24, 1>::Zero();
  for (int k = 0; k < 4; ++k)
    uT.segment<3>(6 * k) = alpha * (g.x[static_cast<size_t>(k)] - g.x[0]);
  const V3 re = p.Be * uT - V3(alpha, alpha, 0.0);
  const V3 rk = p.Bk * uT;
  const Eigen::Vector2d rg = p.Bg * uT;
  t.N += sec.A * re + sec.B * rk;
  t.M += sec.B * re + sec.D * rk;
  t.Q = sec.T * rg;
  return t;
}

/// Section at a point: thickness-scaled (PSHELL grid thicknesses) and
/// rotated into the point's local frame.
ShellSection point_section(const Geometry &g, const PointB &p) {
  const ShellSection base =
      g.scale_thickness ? g.section.scaled_thickness(p.thickness / g.t_nominal)
                        : g.section;
  V3 m = g.material_dir - g.material_dir.dot(p.e3) * p.e3;
  const double phi = std::atan2(m.dot(p.e2), m.dot(p.e1));
  return base.rotated(phi);
}

/// Rigid offset from the grids to the reference surface: u_ref = W u_grid.
Eigen::Matrix<double, 24, 24> offset_transform(const Geometry &g) {
  Eigen::Matrix<double, 24, 24> W = Eigen::Matrix<double, 24, 24>::Identity();
  if (g.offset.squaredNorm() == 0.0)
    return W;
  // u + theta x d = u - skew(d) theta
  for (int k = 0; k < 4; ++k)
    W.block<3, 3>(6 * k, 6 * k + 3) = -skew(g.offset);
  return W;
}

constexpr std::array<double, 2> kGaussPts{-kGauss, kGauss};

} // namespace

namespace {
std::array<NodeId, 4> quad_nodes(const ElementData &data) {
  if (data.nodes.size() != 4)
    throw SolverError(std::format("CQUAD4 {} needs 4 grids, got {}", data.id.value,
                                  data.nodes.size()));
  return {data.nodes[0], data.nodes[1], data.nodes[2], data.nodes[3]};
}
} // namespace

ShellQuad4::ShellQuad4(const ElementData &data, const Model &model)
    : eid_(data.id), pid_(data.pid), nodes_(quad_nodes(data)), model_(model),
      data_(data) {}

std::array<Vec3, 3> ShellQuad4::element_frame() const {
  const Geometry g = build_geometry(data_, model_);
  return {Vec3{g.e1.x(), g.e1.y(), g.e1.z()}, Vec3{g.e2.x(), g.e2.y(), g.e2.z()},
          Vec3{g.n.x(), g.n.y(), g.n.z()}};
}

LocalKe ShellQuad4::stiffness_matrix() const {
  const Geometry g = build_geometry(data_, model_);
  Eigen::Matrix<double, 24, 24> K = Eigen::Matrix<double, 24, 24>::Zero();
  double area = 0.0, shear_stiffness = 0.0;
  for (const double r : kGaussPts)
    for (const double s : kGaussPts) {
      const PointB p = point_b(g, r, s);
      const ShellSection sec = point_section(g, p);
      Eigen::Matrix<double, 6, 6> C;
      C << sec.A, sec.B, sec.B, sec.D;
      Eigen::Matrix<double, 6, 24> Bmk;
      Bmk << p.Be, p.Bk;
      K += p.dA * (Bmk.transpose() * C * Bmk + p.Bg.transpose() * sec.T * p.Bg);
      area += p.dA;
      shear_stiffness += p.dA * sec.A(2, 2);
    }

  // Drill-rotation stabilisation (Hughes & Brezzi 1989).  The drill
  // membrane strain depends only on differences of nodal drill rotations,
  // and on a rectangle the edge contributions of the alternating pattern
  // (-, +, -, +) cancel, so a uniform drill rotation and that hourglass
  // pattern carry no strain energy.  A small penalty on theta - omega,
  // with omega = 1/2 n . curl(u_m) the in-plane rotation of the
  // translation field, removes both; it is integrated at the 2x2 Gauss
  // points (one point would miss the hourglass) and vanishes for rigid
  // motions and constant-strain states, so the patch tests are unaffected.
  {
    const double gamma = kDrillStabilisation * shear_stiffness / area;
    for (const double r : kGaussPts)
      for (const double s : kGaussPts) {
        // Rotations about the local surface normal, so that rigid rotations
        // satisfy theta . n = omega exactly on warped elements too.
        const V3 gr = g.gr_at(s), gs = g.gs_at(r);
        const V3 nl = gr.cross(gs).normalized();
        const auto [gr_up, gs_up] = dual_basis(gr, gs, nl);
        Row diff = Row::Zero();
        for (int k = 0; k < 4; ++k) {
          // curl(h_k u_k) = grad h_k x u_k, and n . (a x u) = u . (n x a).
          const V3 grad = hfun_r(k, s) * gr_up + hfun_s(k, r) * gs_up;
          diff.segment<3>(6 * k) = -0.5 * nl.cross(grad).transpose();
          diff.segment<3>(6 * k + 3) = hfun(k, r, s) * nl.transpose();
        }
        K += gamma * gr.cross(gs).norm() * diff.transpose() * diff;
      }
  }

  const Eigen::Matrix<double, 24, 24> W = offset_transform(g);
  return W.transpose() * K * W;
}

LocalKe ShellQuad4::mass_matrix() const {
  const Geometry g = build_geometry(data_, model_);
  Eigen::Matrix<double, 24, 24> M = Eigen::Matrix<double, 24, 24>::Zero();
  for (const double r : kGaussPts)
    for (const double s : kGaussPts) {
      const PointB p = point_b(g, r, s);
      const ShellSection sec =
          g.scale_thickness ? g.section.scaled_thickness(p.thickness / g.t_nominal)
                            : g.section;
      Eigen::Matrix<double, 3, 24> Nm = Eigen::Matrix<double, 3, 24>::Zero();
      Eigen::Matrix<double, 3, 24> Nb = Eigen::Matrix<double, 3, 24>::Zero();
      for (int k = 0; k < 4; ++k) {
        const double h = hfun(k, r, s);
        Nm.block<3, 3>(0, 6 * k) = h * Eigen::Matrix3d::Identity();
        // theta x V = -skew(V) theta
        Nb.block<3, 3>(0, 6 * k + 3) = -h * skew(g.V[static_cast<size_t>(k)]);
      }
      M += p.dA * (sec.mass_per_area * Nm.transpose() * Nm +
                   sec.mass_first_moment * (Nm.transpose() * Nb + Nb.transpose() * Nm) +
                   sec.mass_second_moment * Nb.transpose() * Nb);
    }
  const Eigen::Matrix<double, 24, 24> W = offset_transform(g);
  return W.transpose() * M * W;
}

LocalFe ShellQuad4::thermal_load(std::span<const double> temperatures,
                                 double t_ref) const {
  const Geometry g = build_geometry(data_, model_);
  Eigen::Matrix<double, 24, 1> f = Eigen::Matrix<double, 24, 1>::Zero();
  for (const double r : kGaussPts)
    for (const double s : kGaussPts) {
      double T = 0.0;
      for (int k = 0; k < 4; ++k)
        T += hfun(k, r, s) * temperatures[static_cast<size_t>(k)];
      const double dT = T - t_ref;
      if (dT == 0.0)
        continue;
      const PointB p = point_b(g, r, s);
      const ShellSection sec = point_section(g, p);
      const ThermalResultants tr = thermal_resultants(g, sec, p);
      f += p.dA * dT *
           (p.Be.transpose() * tr.N + p.Bk.transpose() * tr.M + p.Bg.transpose() * tr.Q);
    }
  const Eigen::Matrix<double, 24, 24> W = offset_transform(g);
  return W.transpose() * f;
}

std::vector<EqIndex> ShellQuad4::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> result;
  result.reserve(NUM_DOFS);
  for (NodeId nid : nodes_) {
    const auto &blk = dof_map.block(nid);
    for (int d = 0; d < 6; ++d)
      result.push_back(blk.eq[d]);
  }
  return result;
}

std::array<ShellPointResponse, 5>
ShellQuad4::recover(std::span<const double> global_displacements,
                    std::span<const double> temperatures, double t_ref) const {
  if (global_displacements.size() != NUM_DOFS)
    throw SolverError(std::format("CQUAD4 {}: expected 24 displacements, got {}",
                                  eid_.value, global_displacements.size()));
  const Geometry g = build_geometry(data_, model_);
  Eigen::Matrix<double, 24, 1> ug;
  for (int i = 0; i < NUM_DOFS; ++i)
    ug(i) = global_displacements[static_cast<size_t>(i)];
  const Eigen::Matrix<double, 24, 1> u = offset_transform(g) * ug;

  auto at = [&](double r, double s) {
    const PointB p = point_b(g, r, s);
    const ShellSection sec = point_section(g, p);
    double T = 0.0;
    for (int k = 0; k < 4; ++k)
      T += hfun(k, r, s) *
           (temperatures.empty() ? t_ref : temperatures[static_cast<size_t>(k)]);
    const double dT = T - t_ref;
    ShellPointResponse out;
    out.membrane_strain = p.Be * u;
    out.curvature = p.Bk * u;
    out.transverse_shear_strain = p.Bg * u;
    const ThermalResultants tr = thermal_resultants(g, sec, p);
    out.membrane_resultant = sec.A * out.membrane_strain + sec.B * out.curvature -
                             tr.N * dT;
    out.bending_moment = sec.B * out.membrane_strain + sec.D * out.curvature -
                         tr.M * dT;
    out.transverse_shear = sec.T * out.transverse_shear_strain - tr.Q * dT;
    out.thickness = p.thickness;
    return out;
  };

  std::array<ShellPointResponse, 5> res;
  res[0] = at(0.0, 0.0);
  // Gauss-point values, then bilinear extrapolation to the corners.
  std::array<ShellPointResponse, 4> gp;
  for (int k = 0; k < 4; ++k)
    gp[static_cast<size_t>(k)] = at(kXi[k] * kGauss, kEta[k] * kGauss);
  for (int c = 0; c < 4; ++c) {
    ShellPointResponse &o = res[static_cast<size_t>(c + 1)];
    o = ShellPointResponse{};
    for (int k = 0; k < 4; ++k) {
      const double w = hfun(k, kXi[c] * kSqrt3, kEta[c] * kSqrt3);
      const ShellPointResponse &v = gp[static_cast<size_t>(k)];
      o.membrane_strain += w * v.membrane_strain;
      o.curvature += w * v.curvature;
      o.transverse_shear_strain += w * v.transverse_shear_strain;
      o.membrane_resultant += w * v.membrane_resultant;
      o.bending_moment += w * v.bending_moment;
      o.transverse_shear += w * v.transverse_shear;
      o.thickness += w * v.thickness;
    }
  }
  return res;
}

} // namespace vibestran
