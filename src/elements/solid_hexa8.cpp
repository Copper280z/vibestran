// src/elements/solid_hexa8.cpp
// Q1/E21 enhanced assumed strain CHEXA8; see include/elements/solid_hexa8.hpp.

#include "elements/solid_hexa8.hpp"
#include "elements/solid_elements.hpp" // CHexa8::shape_functions
#include <Eigen/Dense>
#include <cmath>
#include <format>
#include <vector>

namespace vibestran {

namespace {

using Mat6 = Eigen::Matrix<double, 6, 6>;
using B6 = Eigen::Matrix<double, 6, 24>;
using M6 = Eigen::Matrix<double, 6, SolidHexa8::NUM_MODES>;
using GMat = Eigen::Matrix<double, SolidHexa8::NUM_MODES, 24>;
using HMat = Eigen::Matrix<double, SolidHexa8::NUM_MODES, SolidHexa8::NUM_MODES>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

/// Smallest scaled Jacobian (det J over the product of the row lengths of
/// J) at which a corner is evaluated directly rather than extrapolated.
constexpr double kMinScaledJacobian = 1.0e-12;

Mat6 isotropic_D(double E, double nu) {
  const double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const double mu = E / (2.0 * (1.0 + nu));
  Mat6 D = Mat6::Zero();
  D.topLeftCorner<3, 3>().setConstant(lam);
  for (int i = 0; i < 3; ++i) {
    D(i, i) = lam + 2.0 * mu;
    D(i + 3, i + 3) = mu;
  }
  return D;
}

/// Gauss-Legendre points and weights of order 2 or 3 on [-1, 1].
std::pair<std::vector<double>, std::vector<double>> gauss(int order) {
  if (order == 3) {
    const double a = std::sqrt(0.6);
    return {{-a, 0.0, a}, {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0}};
  }
  const double a = 1.0 / std::sqrt(3.0);
  return {{-a, a}, {1.0, 1.0}};
}

/// Jacobian J(i, j) = dx_j / dxi_i at a point, and the shape data.
struct PointGeometry {
  CHexa8::ShapeData sd;
  Eigen::Matrix3d J;
  Eigen::Matrix3d Jinv; // Jinv(k, p) = dxi_p / dx_k
  double detJ{0.0};
};

PointGeometry point_geometry(const std::array<Eigen::Vector3d, 8> &X, double xi, double eta,
                             double zeta) {
  PointGeometry p;
  p.sd = CHexa8::shape_functions(xi, eta, zeta);
  p.J.setZero();
  for (size_t n = 0; n < 8; ++n) {
    p.J.row(0) += p.sd.dNdxi[n] * X[n].transpose();
    p.J.row(1) += p.sd.dNdeta[n] * X[n].transpose();
    p.J.row(2) += p.sd.dNdzeta[n] * X[n].transpose();
  }
  p.detJ = p.J.determinant();
  p.Jinv = p.J.inverse();
  return p;
}

double scaled_jacobian(const PointGeometry &p) {
  const double rows = p.J.row(0).norm() * p.J.row(1).norm() * p.J.row(2).norm();
  return rows > 0.0 ? p.detJ / rows : 0.0;
}

/// Compatible strain-displacement matrix (xx yy zz xy yz zx, engineering).
B6 compatible_B(const PointGeometry &p) {
  B6 B = B6::Zero();
  for (int n = 0; n < 8; ++n) {
    const auto k = static_cast<size_t>(n);
    const Eigen::Vector3d dN =
        p.Jinv * Eigen::Vector3d(p.sd.dNdxi[k], p.sd.dNdeta[k], p.sd.dNdzeta[k]);
    const int c = 3 * n;
    B(0, c) = dN(0);
    B(1, c + 1) = dN(1);
    B(2, c + 2) = dN(2);
    B(3, c) = dN(1);
    B(3, c + 1) = dN(0);
    B(4, c + 1) = dN(2);
    B(4, c + 2) = dN(1);
    B(5, c) = dN(2);
    B(5, c + 2) = dN(0);
  }
  return B;
}

/// Natural-to-Cartesian transformation of engineering strains:
/// eps_kl = sum_pq (dxi_p/dx_k) (dxi_q/dx_l) eps_pq.
Mat6 strain_transform(const Eigen::Matrix3d &Jinv) {
  static constexpr std::array<int, 6> I{0, 1, 2, 0, 1, 2};
  static constexpr std::array<int, 6> J{0, 1, 2, 1, 2, 0};
  Mat6 T;
  for (size_t r = 0; r < 6; ++r) {
    const int k = I[r], l = J[r];
    for (size_t c = 0; c < 6; ++c) {
      const int p = I[c], q = J[c];
      if (r < 3)
        T(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
            c < 3 ? Jinv(k, p) * Jinv(k, p) : Jinv(k, p) * Jinv(k, q);
      else
        T(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
            c < 3 ? 2.0 * Jinv(k, p) * Jinv(l, p)
                  : Jinv(k, p) * Jinv(l, q) + Jinv(k, q) * Jinv(l, p);
    }
  }
  return T;
}

/// Natural-frame Q1/E21 modes at (xi, eta, zeta).
M6 natural_modes(double xi, double eta, double zeta) {
  M6 m = M6::Zero();
  m(0, 0) = xi;
  m(1, 1) = eta;
  m(2, 2) = zeta;
  m(3, 3) = xi;
  m(3, 4) = eta;
  m(4, 5) = eta;
  m(4, 6) = zeta;
  m(5, 7) = zeta;
  m(5, 8) = xi;
  m(0, 9) = xi * eta;
  m(0, 10) = xi * zeta;
  m(1, 11) = eta * xi;
  m(1, 12) = eta * zeta;
  m(2, 13) = zeta * xi;
  m(2, 14) = zeta * eta;
  m(3, 15) = xi * zeta;
  m(3, 16) = eta * zeta;
  m(4, 17) = eta * xi;
  m(4, 18) = zeta * xi;
  m(5, 19) = zeta * eta;
  m(5, 20) = xi * eta;
  return m;
}

/// Thermal strain alpha (T - T_ref) at a point with shape functions N.
Vec6 thermal_strain(const CHexa8::ShapeData &sd, std::span<const double> temperatures,
                    double t_ref, double alpha) {
  Vec6 e = Vec6::Zero();
  if (temperatures.empty() || alpha == 0.0)
    return e;
  double T = 0.0;
  for (size_t n = 0; n < 8; ++n)
    T += sd.N[n] * temperatures[n];
  e.head<3>().setConstant(alpha * (T - t_ref));
  return e;
}

/// Everything the element operators need, built once per call.
struct Setup {
  std::array<Eigen::Vector3d, 8> X;
  Mat6 D;
  double alpha{0.0};
  double rho{0.0};
  bool enhanced{false};
  std::vector<double> pts, wts;
  struct Gp {
    double xi, eta, zeta, w; // w includes det J
    PointGeometry geo;
    B6 B;
    M6 M;
  };
  std::vector<Gp> gps;
  // Enhanced strain data.
  Mat6 T0{Mat6::Zero()};
  double detJ0{0.0};
  GMat G{GMat::Zero()};
  Eigen::Matrix<double, SolidHexa8::NUM_MODES, 1> scale; // S = diag(H)^-1/2
  Eigen::LLT<HMat> llt;
  GMat HinvG{GMat::Zero()};

  [[nodiscard]] M6 modes(double xi, double eta, double zeta, double detJ) const {
    return (detJ0 / detJ) * T0 * natural_modes(xi, eta, zeta);
  }

  /// H^-1 R through the equilibrated Cholesky factor.
  template <class Rhs> [[nodiscard]] Rhs solve_H(const Rhs &R) const {
    Rhs Y = scale.asDiagonal() * R;
    Y = llt.solve(Y);
    return scale.asDiagonal() * Y;
  }
};

Setup build(ElementId eid, const std::array<NodeId, 8> &nodes, const Model &model,
            const PSolid &ps, SolidHexa8::Formulation form, int order) {
  Setup s;
  for (size_t n = 0; n < 8; ++n) {
    const Vec3 &p = model.node(nodes[n]).position;
    s.X[n] = Eigen::Vector3d(p.x, p.y, p.z);
  }
  const Mat1 &mat = model.material(ps.mid);
  s.D = isotropic_D(mat.E, mat.nu);
  s.alpha = mat.A;
  s.rho = mat.rho;
  s.enhanced = form == SolidHexa8::Formulation::EnhancedStrain;
  std::tie(s.pts, s.wts) = gauss(order);

  double volume = 0.0;
  for (size_t k = 0; k < s.pts.size(); ++k)
    for (size_t j = 0; j < s.pts.size(); ++j)
      for (size_t i = 0; i < s.pts.size(); ++i) {
        Setup::Gp g{s.pts[i], s.pts[j], s.pts[k], 0.0, {}, B6::Zero(), M6::Zero()};
        g.geo = point_geometry(s.X, g.xi, g.eta, g.zeta);
        if (!(g.geo.detJ > 0.0))
          throw SolverError(std::format(
              "CHEXA8 {}: non-positive Jacobian {:.6g} at Gauss point ({:.4f}, {:.4f}, "
              "{:.4f}); the element is inverted, collapsed or too distorted",
              eid.value, g.geo.detJ, g.xi, g.eta, g.zeta));
        g.w = s.wts[i] * s.wts[j] * s.wts[k] * g.geo.detJ;
        volume += g.w;
        g.B = compatible_B(g.geo);
        s.gps.push_back(g);
      }
  if (!(volume > 0.0))
    throw SolverError(std::format("CHEXA8 {}: non-positive volume {:.6g}", eid.value, volume));

  if (s.enhanced) {
    const PointGeometry c = point_geometry(s.X, 0.0, 0.0, 0.0);
    if (!(c.detJ > 0.0))
      throw SolverError(std::format("CHEXA8 {}: non-positive centroid Jacobian {:.6g}",
                                    eid.value, c.detJ));
    s.detJ0 = c.detJ;
    s.T0 = strain_transform(c.Jinv);
    HMat H = HMat::Zero();
    for (auto &g : s.gps) {
      g.M = s.modes(g.xi, g.eta, g.zeta, g.geo.detJ);
      const M6 DM = s.D * g.M;
      s.G += g.w * DM.transpose() * g.B;
      H += g.w * g.M.transpose() * DM;
    }
    for (int i = 0; i < SolidHexa8::NUM_MODES; ++i) {
      if (!(H(i, i) > 0.0))
        throw SolverError(std::format(
            "CHEXA8 {}: enhanced-strain stiffness H is not positive definite (mode {}, "
            "H = {:.6g}); the element is degenerate or its material is not positive "
            "definite",
            eid.value, i + 1, H(i, i)));
      s.scale(i) = 1.0 / std::sqrt(H(i, i));
    }
    s.llt.compute(s.scale.asDiagonal() * H * s.scale.asDiagonal());
    if (s.llt.info() != Eigen::Success)
      throw SolverError(std::format(
          "CHEXA8 {}: enhanced-strain stiffness H is not positive definite; the "
          "element is degenerate or its material is not positive definite",
          eid.value));
    s.HinvG = s.solve_H(s.G);
  }
  return s;
}

/// Internal-mode thermal load q = int M' D eps_th dV.
Eigen::Matrix<double, SolidHexa8::NUM_MODES, 1>
internal_thermal_load(const Setup &s, std::span<const double> temperatures, double t_ref) {
  Eigen::Matrix<double, SolidHexa8::NUM_MODES, 1> q =
      Eigen::Matrix<double, SolidHexa8::NUM_MODES, 1>::Zero();
  for (const auto &g : s.gps)
    q += g.w * g.M.transpose() * (s.D * thermal_strain(g.geo.sd, temperatures, t_ref, s.alpha));
  return q;
}

} // namespace

SolidHexa8::SolidHexa8(ElementId eid, PropertyId pid, std::array<NodeId, NUM_NODES> node_ids,
                       const Model &model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {
  const auto &prop = model_.property(pid_);
  if (!std::holds_alternative<PSolid>(prop))
    throw SolverError(
        std::format("CHEXA8 {}: property {} is not a PSOLID", eid_.value, pid_.value));
  const PSolid &ps = std::get<PSolid>(prop);
  formulation_ = ps.isop == SolidFormulation::Full ? Formulation::Full
                                                   : Formulation::EnhancedStrain;
  order_ = ps.integration_order == 3 ? 3 : 2;
}

LocalKe SolidHexa8::stiffness_matrix() const {
  const Setup s = build(eid_, nodes_, model_, std::get<PSolid>(model_.property(pid_)),
                        formulation_, order_);
  Eigen::Matrix<double, 24, 24> K = Eigen::Matrix<double, 24, 24>::Zero();
  for (const auto &g : s.gps)
    K += g.w * g.B.transpose() * s.D * g.B;
  if (s.enhanced)
    K -= s.G.transpose() * s.HinvG;
  return 0.5 * (K + K.transpose());
}

LocalKe SolidHexa8::mass_matrix() const {
  const Setup s = build(eid_, nodes_, model_, std::get<PSolid>(model_.property(pid_)),
                        Formulation::Full, order_);
  LocalKe M = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
  if (s.rho == 0.0)
    return M;
  for (const auto &g : s.gps)
    for (int a = 0; a < 8; ++a)
      for (int b = 0; b < 8; ++b) {
        const double v = s.rho * g.geo.sd.N[static_cast<size_t>(a)] *
                         g.geo.sd.N[static_cast<size_t>(b)] * g.w;
        for (int d = 0; d < 3; ++d)
          M(3 * a + d, 3 * b + d) += v;
      }
  return M;
}

LocalFe SolidHexa8::thermal_load(std::span<const double> temperatures, double t_ref) const {
  const Setup s = build(eid_, nodes_, model_, std::get<PSolid>(model_.property(pid_)),
                        formulation_, order_);
  LocalFe f = LocalFe::Zero(NUM_DOFS);
  if (s.alpha == 0.0 || temperatures.empty())
    return f;
  for (const auto &g : s.gps)
    f += g.w * g.B.transpose() * (s.D * thermal_strain(g.geo.sd, temperatures, t_ref, s.alpha));
  if (s.enhanced)
    f -= s.G.transpose() * s.solve_H(internal_thermal_load(s, temperatures, t_ref));
  return f;
}

std::array<SolidHexa8::PointResponse, 9>
SolidHexa8::recover(std::span<const double> displacements, std::span<const double> temperatures,
                    double t_ref) const {
  if (displacements.size() != NUM_DOFS)
    throw SolverError(std::format("CHEXA8 {}: recovery needs {} displacements, got {}",
                                  eid_.value, NUM_DOFS, displacements.size()));
  const Setup s = build(eid_, nodes_, model_, std::get<PSolid>(model_.property(pid_)),
                        formulation_, order_);
  const Eigen::Map<const Eigen::Matrix<double, 24, 1>> u(displacements.data());
  Eigen::Matrix<double, NUM_MODES, 1> Hq = Eigen::Matrix<double, NUM_MODES, 1>::Zero();
  if (s.enhanced && s.alpha != 0.0 && !temperatures.empty())
    Hq = s.solve_H(internal_thermal_load(s, temperatures, t_ref));

  // Total strain at a point with a positive Jacobian.
  auto direct = [&](double xi, double eta, double zeta) -> Vec6 {
    const PointGeometry p = point_geometry(s.X, xi, eta, zeta);
    Vec6 e = compatible_B(p) * u;
    if (s.enhanced) {
      const M6 M = s.modes(xi, eta, zeta, p.detJ);
      e += M * (Hq - s.HinvG * u);
    }
    return e;
  };
  static constexpr std::array<std::array<double, 3>, 9> kPoints{{{0.0, 0.0, 0.0},
                                                                  {-1.0, -1.0, -1.0},
                                                                  {1.0, -1.0, -1.0},
                                                                  {1.0, 1.0, -1.0},
                                                                  {-1.0, 1.0, -1.0},
                                                                  {-1.0, -1.0, 1.0},
                                                                  {1.0, -1.0, 1.0},
                                                                  {1.0, 1.0, 1.0},
                                                                  {-1.0, 1.0, 1.0}}};
  std::array<PointResponse, 9> out;
  for (size_t k = 0; k < kPoints.size(); ++k) {
    const auto [xi, eta, zeta] = kPoints[k];
    const PointGeometry p = point_geometry(s.X, xi, eta, zeta);
    Vec6 e;
    if (scaled_jacobian(p) > kMinScaledJacobian) {
      e = direct(xi, eta, zeta);
    } else {
      // Trilinear extrapolation from the 2x2x2 Gauss points.
      const double g = 1.0 / std::sqrt(3.0);
      e = Vec6::Zero();
      for (const auto &c : kPoints) {
        if (c[0] == 0.0)
          continue; // the centre is not a sampling point
        const double w = (1.0 + c[0] * xi / g) * (1.0 + c[1] * eta / g) *
                         (1.0 + c[2] * zeta / g) / 8.0;
        e += w * direct(c[0] * g, c[1] * g, c[2] * g);
      }
    }
    out[k].strain = e;
    out[k].stress = s.D * (e - thermal_strain(p.sd, temperatures, t_ref, s.alpha));
  }
  return out;
}

std::vector<EqIndex> SolidHexa8::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> idx;
  idx.reserve(NUM_DOFS);
  for (NodeId n : nodes_)
    for (int d = 0; d < 3; ++d)
      idx.push_back(dof_map.eq_index(n, d));
  return idx;
}

} // namespace vibestran
