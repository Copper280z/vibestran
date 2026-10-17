// src/elements/solid_hexa20.cpp
// Projected assumed-strain CHEXA20; see include/elements/solid_hexa20.hpp.

#include "elements/solid_hexa20.hpp"
#include "elements/hexa20_shape.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <format>
#include <vector>

namespace vibestran {

namespace {

constexpr int kDofs = SolidHexa20::NUM_DOFS;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using BMat = Eigen::Matrix<double, 6, kDofs>;
using Row = Eigen::Matrix<double, 1, kDofs>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

/// Smallest scaled Jacobian at which a corner is evaluated directly rather
/// than extrapolated (as for the CHEXA8).
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

/// J(i, j) = dx_j / dxi_i at a point, and the shape data.
struct PointGeometry {
  std::array<double, 3> xi{};
  Hexa20ShapeData sd;
  Eigen::Matrix3d J;
  Eigen::Matrix3d Jinv; // Jinv(k, p) = dxi_p / dx_k
  double detJ{0.0};
};

PointGeometry point_geometry(const std::array<Eigen::Vector3d, 20> &X,
                             const std::array<double, 3> &xi) {
  PointGeometry p;
  p.xi = xi;
  p.sd = hexa20_shape_functions(xi[0], xi[1], xi[2]);
  p.J.setZero();
  for (std::size_t a = 0; a < 20; ++a)
    for (int i = 0; i < 3; ++i)
      p.J.row(i) += p.sd.dN[a][static_cast<std::size_t>(i)] * X[a].transpose();
  p.detJ = p.J.determinant();
  p.Jinv = p.J.inverse();
  return p;
}

double scaled_jacobian(const PointGeometry &p) {
  const double rows = p.J.row(0).norm() * p.J.row(1).norm() * p.J.row(2).norm();
  return rows > 0.0 ? p.detJ / rows : 0.0;
}

/// Compatible strain-displacement matrix (xx yy zz xy yz zx, engineering).
BMat compatible_B(const PointGeometry &p) {
  BMat B = BMat::Zero();
  for (int a = 0; a < 20; ++a) {
    const auto &d = p.sd.dN[static_cast<std::size_t>(a)];
    const Eigen::Vector3d g = p.Jinv * Eigen::Vector3d(d[0], d[1], d[2]);
    const int c = 3 * a;
    B(0, c) = g(0);
    B(1, c + 1) = g(1);
    B(2, c + 2) = g(2);
    B(3, c) = g(1);
    B(3, c + 1) = g(0);
    B(4, c + 1) = g(2);
    B(4, c + 2) = g(1);
    B(5, c) = g(2);
    B(5, c + 2) = g(0);
  }
  return B;
}

/// Cartesian engineering strains -> natural components (e11, e22, e33, g12,
/// g23, g31) in the unit frame t_i = J.row(i) / |J.row(i)|.
Mat6 natural_transform(const Eigen::Matrix3d &J) {
  std::array<Eigen::Vector3d, 3> t;
  for (int i = 0; i < 3; ++i)
    t[static_cast<std::size_t>(i)] = J.row(i).transpose().normalized();
  static constexpr std::array<std::array<int, 2>, 6> kPairs{
      {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {2, 0}}};
  Mat6 T;
  for (std::size_t r = 0; r < 6; ++r) {
    const Eigen::Vector3d &a = t[static_cast<std::size_t>(kPairs[r][0])];
    const Eigen::Vector3d &b = t[static_cast<std::size_t>(kPairs[r][1])];
    const double f = r < 3 ? 1.0 : 2.0;
    for (std::size_t c = 0; c < 6; ++c) {
      const int k = kPairs[c][0], l = kPairs[c][1];
      // Strain tensor of a unit engineering component c.
      const double v = c < 3 ? a(k) * b(k) : 0.5 * (a(k) * b(l) + a(l) * b(k));
      T(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = f * v;
    }
  }
  return T;
}

/// A polynomial space as a list of exponent triples.
using Space = std::vector<std::array<int, 3>>;

/// S_i: degree <= 2 in each coordinate and <= 1 in xi_i (18 monomials).
Space normal_strain_space(int i) {
  Space s;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c) {
        const std::array<int, 3> e{a, b, c};
        if (e[static_cast<std::size_t>(i)] <= 1)
          s.push_back(e);
      }
  return s;
}

/// P8: the trilinear monomials.
Space trilinear_space() {
  Space s;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 2; ++c)
        s.push_back({a, b, c});
  return s;
}

Eigen::VectorXd evaluate(const Space &s, const std::array<double, 3> &xi) {
  Eigen::VectorXd v(static_cast<Eigen::Index>(s.size()));
  for (std::size_t m = 0; m < s.size(); ++m)
    v(static_cast<Eigen::Index>(m)) = std::pow(xi[0], s[m][0]) * std::pow(xi[1], s[m][1]) *
                                      std::pow(xi[2], s[m][2]);
  return v;
}

/// det(J)-weighted L2 projection onto a space: coefficients (m x 60) of the
/// projection of a row field given at the Gauss points.
class Projection {
public:
  Projection(Space space, const std::vector<PointGeometry> &gps, const std::vector<double> &w,
             ElementId eid)
      : space_(std::move(space)) {
    const auto m = static_cast<Eigen::Index>(space_.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(m, m);
    S_.reserve(gps.size());
    for (std::size_t g = 0; g < gps.size(); ++g) {
      S_.push_back(evaluate(space_, gps[g].xi));
      W += w[g] * S_.back() * S_.back().transpose();
    }
    llt_.compute(W);
    if (llt_.info() != Eigen::Success)
      throw SolverError(std::format(
          "CHEXA20 {}: assumed-strain projection matrix is not positive definite; the "
          "element is degenerate",
          eid.value));
    w_ = w;
  }

  /// Coefficients of the projection of field rows f_g (one row per point).
  [[nodiscard]] Eigen::MatrixXd coefficients(const std::vector<Row> &f) const {
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(space_.size()), kDofs);
    for (std::size_t g = 0; g < f.size(); ++g)
      R += w_[g] * S_[g] * f[g];
    return llt_.solve(R);
  }

  [[nodiscard]] Row at(const Eigen::MatrixXd &coef, const std::array<double, 3> &xi) const {
    return evaluate(space_, xi).transpose() * coef;
  }

private:
  Space space_;
  std::vector<Eigen::VectorXd> S_;
  std::vector<double> w_;
  Eigen::LLT<Eigen::MatrixXd> llt_;
};

Vec6 thermal_strain(const Hexa20ShapeData &sd, std::span<const double> temperatures,
                    double t_ref, double alpha) {
  Vec6 e = Vec6::Zero();
  if (temperatures.empty() || alpha == 0.0)
    return e;
  double T = 0.0;
  for (std::size_t a = 0; a < 20; ++a)
    T += sd.N[a] * temperatures[a];
  e.head<3>().setConstant(alpha * (T - t_ref));
  return e;
}

/// The element's strain operator, built once per call.
class StrainOperator {
public:
  StrainOperator(ElementId eid, const std::array<NodeId, 20> &nodes, const Model &model,
                 const PSolid &ps, bool assumed)
      : eid_(eid), assumed_(assumed) {
    for (std::size_t a = 0; a < 20; ++a) {
      const Vec3 &p = model.node(nodes[a]).position;
      X_[a] = Eigen::Vector3d(p.x, p.y, p.z);
    }
    const Mat1 &mat = model.material(ps.mid);
    D_ = isotropic_D(mat.E, mat.nu);
    alpha_ = mat.A;
    rho_ = mat.rho;

    const double g = std::sqrt(0.6);
    const std::array<double, 3> pts{-g, 0.0, g};
    const std::array<double, 3> wts{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    for (std::size_t k = 0; k < 3; ++k)
      for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = 0; i < 3; ++i) {
          PointGeometry p = point_geometry(X_, {pts[i], pts[j], pts[k]});
          if (!(p.detJ > 0.0))
            throw SolverError(std::format(
                "CHEXA20 {}: non-positive Jacobian {:.6g} at Gauss point ({:.4f}, {:.4f}, "
                "{:.4f}); the element is inverted, collapsed or too distorted",
                eid.value, p.detJ, p.xi[0], p.xi[1], p.xi[2]));
          w_.push_back(wts[i] * wts[j] * wts[k] * p.detJ);
          B_.push_back(compatible_B(p));
          gps_.push_back(std::move(p));
        }
    double volume = 0.0;
    for (double w : w_)
      volume += w;
    if (!(volume > 0.0))
      throw SolverError(std::format("CHEXA20 {}: non-positive volume {:.6g}", eid.value, volume));
    volume_ = volume;
    if (assumed_)
      build_assumed();
    else
      bbar_ = B_;
  }

  [[nodiscard]] const std::vector<BMat> &gauss_operators() const { return bbar_; }
  [[nodiscard]] const std::vector<PointGeometry> &gauss_points() const { return gps_; }
  [[nodiscard]] const std::vector<double> &weights() const { return w_; }
  [[nodiscard]] const Mat6 &D() const { return D_; }
  [[nodiscard]] double alpha() const { return alpha_; }
  [[nodiscard]] double rho() const { return rho_; }
  [[nodiscard]] const std::array<Eigen::Vector3d, 20> &X() const { return X_; }

  /// Strain operator at an arbitrary natural point (positive Jacobian).
  [[nodiscard]] BMat at(const PointGeometry &p) const {
    if (!assumed_)
      return compatible_B(p);
    const BMat Bs = Bm_ + deviation(p) - Dmean_;
    return dilatation_projected(Bs, p.xi);
  }

private:
  /// A(B - Bm) at a point from the stored projection coefficients.
  [[nodiscard]] BMat deviation(const PointGeometry &p) const {
    BMat nat;
    for (int r = 0; r < 6; ++r)
      nat.row(r) = proj_[static_cast<std::size_t>(r < 3 ? r : 3)].at(
          coef_[static_cast<std::size_t>(r)], p.xi);
    const Mat6 T = natural_transform(p.J);
    return T.partialPivLu().solve(nat);
  }

  [[nodiscard]] BMat dilatation_projected(const BMat &Bs, const std::array<double, 3> &xi) const {
    const Row d = Bs.row(0) + Bs.row(1) + Bs.row(2);
    const Row dp = proj_[3].at(vol_coef_, xi);
    BMat out = Bs;
    for (int r = 0; r < 3; ++r)
      out.row(r) += (dp - d) / 3.0;
    return out;
  }

  void build_assumed() {
    for (int i = 0; i < 3; ++i)
      proj_.emplace_back(normal_strain_space(i), gps_, w_, eid_);
    proj_.emplace_back(trilinear_space(), gps_, w_, eid_);

    Bm_.setZero();
    for (std::size_t g = 0; g < gps_.size(); ++g)
      Bm_ += w_[g] * B_[g];
    Bm_ /= volume_;

    // Natural components of B - Bm at each point, projected per component.
    std::array<std::vector<Row>, 6> rows;
    for (std::size_t g = 0; g < gps_.size(); ++g) {
      const BMat nat = natural_transform(gps_[g].J) * (B_[g] - Bm_);
      for (std::size_t r = 0; r < 6; ++r)
        rows[r].push_back(nat.row(static_cast<Eigen::Index>(r)));
    }
    for (std::size_t r = 0; r < 6; ++r)
      coef_[r] = proj_[r < 3 ? r : 3].coefficients(rows[r]);

    // Zero-mean correction.
    std::vector<BMat> dev;
    dev.reserve(gps_.size());
    Dmean_.setZero();
    for (std::size_t g = 0; g < gps_.size(); ++g) {
      dev.push_back(deviation(gps_[g]));
      Dmean_ += w_[g] * dev.back();
    }
    Dmean_ /= volume_;

    std::vector<BMat> Bs;
    std::vector<Row> vol;
    for (std::size_t g = 0; g < gps_.size(); ++g) {
      Bs.push_back(Bm_ + dev[g] - Dmean_);
      vol.push_back(Bs.back().row(0) + Bs.back().row(1) + Bs.back().row(2));
    }
    vol_coef_ = proj_[3].coefficients(vol);
    bbar_.clear();
    for (std::size_t g = 0; g < gps_.size(); ++g)
      bbar_.push_back(dilatation_projected(Bs[g], gps_[g].xi));
  }

  ElementId eid_;
  bool assumed_;
  std::array<Eigen::Vector3d, 20> X_;
  Mat6 D_;
  double alpha_{0.0};
  double rho_{0.0};
  double volume_{0.0};
  std::vector<PointGeometry> gps_;
  std::vector<double> w_;
  std::vector<BMat> B_;
  std::vector<BMat> bbar_;
  // Assumed-strain data: projections onto S_1, S_2, S_3 and P8.
  std::vector<Projection> proj_;
  BMat Bm_{BMat::Zero()};
  BMat Dmean_{BMat::Zero()};
  std::array<Eigen::MatrixXd, 6> coef_;
  Eigen::MatrixXd vol_coef_;
};

} // namespace

SolidHexa20::SolidHexa20(ElementId eid, PropertyId pid, std::array<NodeId, NUM_NODES> node_ids,
                         const Model &model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {
  for (std::size_t a = 8; a < 20; ++a)
    if (nodes_[a].value == 0)
      throw SolverError(std::format(
          "CHEXA {} omits midside grid G{}: CHEXA elements with some but not all midside "
          "grids are not supported for structural analysis; give all 12 midside grids or "
          "none",
          eid_.value, a + 1));
  const auto &prop = model_.property(pid_);
  if (!std::holds_alternative<PSolid>(prop))
    throw SolverError(
        std::format("CHEXA20 {}: property {} is not a PSOLID", eid_.value, pid_.value));
  const PSolid &ps = std::get<PSolid>(prop);
  formulation_ =
      ps.isop == SolidFormulation::Full ? Formulation::Full : Formulation::AssumedStrain;
}

LocalKe SolidHexa20::stiffness_matrix() const {
  const StrainOperator op(eid_, nodes_, model_, std::get<PSolid>(model_.property(pid_)),
                          formulation_ == Formulation::AssumedStrain);
  Eigen::Matrix<double, kDofs, kDofs> K = Eigen::Matrix<double, kDofs, kDofs>::Zero();
  const auto &Bb = op.gauss_operators();
  for (std::size_t g = 0; g < Bb.size(); ++g)
    K.noalias() += op.weights()[g] * Bb[g].transpose() * (op.D() * Bb[g]);
  return 0.5 * (K + K.transpose());
}

LocalKe SolidHexa20::mass_matrix() const {
  const StrainOperator op(eid_, nodes_, model_, std::get<PSolid>(model_.property(pid_)),
                          false);
  LocalKe M = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
  if (op.rho() == 0.0)
    return M;
  const auto &gps = op.gauss_points();
  for (std::size_t g = 0; g < gps.size(); ++g)
    for (std::size_t a = 0; a < 20; ++a)
      for (std::size_t b = 0; b < 20; ++b) {
        const double v = op.rho() * gps[g].sd.N[a] * gps[g].sd.N[b] * op.weights()[g];
        for (int d = 0; d < 3; ++d)
          M(3 * static_cast<Eigen::Index>(a) + d, 3 * static_cast<Eigen::Index>(b) + d) += v;
      }
  return M;
}

LocalFe SolidHexa20::thermal_load(std::span<const double> temperatures, double t_ref) const {
  const StrainOperator op(eid_, nodes_, model_, std::get<PSolid>(model_.property(pid_)),
                          formulation_ == Formulation::AssumedStrain);
  LocalFe f = LocalFe::Zero(NUM_DOFS);
  if (op.alpha() == 0.0 || temperatures.empty())
    return f;
  if (temperatures.size() != NUM_NODES)
    throw SolverError(std::format("CHEXA20 {}: thermal load needs {} temperatures, got {}",
                                  eid_.value, NUM_NODES, temperatures.size()));
  const auto &Bb = op.gauss_operators();
  for (std::size_t g = 0; g < Bb.size(); ++g)
    f += op.weights()[g] * Bb[g].transpose() *
         (op.D() * thermal_strain(op.gauss_points()[g].sd, temperatures, t_ref, op.alpha()));
  return f;
}

std::array<SolidHexa20::PointResponse, 9>
SolidHexa20::recover(std::span<const double> displacements, std::span<const double> temperatures,
                     double t_ref) const {
  if (displacements.size() != NUM_DOFS)
    throw SolverError(std::format("CHEXA20 {}: recovery needs {} displacements, got {}",
                                  eid_.value, NUM_DOFS, displacements.size()));
  if (!temperatures.empty() && temperatures.size() != NUM_NODES)
    throw SolverError(std::format("CHEXA20 {}: recovery needs {} temperatures, got {}",
                                  eid_.value, NUM_NODES, temperatures.size()));
  const StrainOperator op(eid_, nodes_, model_, std::get<PSolid>(model_.property(pid_)),
                          formulation_ == Formulation::AssumedStrain);
  const Eigen::Map<const Eigen::Matrix<double, kDofs, 1>> u(displacements.data());

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
  for (std::size_t k = 0; k < kPoints.size(); ++k) {
    const PointGeometry p = point_geometry(op.X(), kPoints[k]);
    Vec6 e;
    if (scaled_jacobian(p) > kMinScaledJacobian) {
      e = op.at(p) * u;
    } else {
      // Trilinear extrapolation from the 2x2x2 Gauss points.
      const double g = 1.0 / std::sqrt(3.0);
      e = Vec6::Zero();
      for (std::size_t c = 1; c < kPoints.size(); ++c) {
        const auto &s = kPoints[c];
        const double w = (1.0 + s[0] * p.xi[0] / g) * (1.0 + s[1] * p.xi[1] / g) *
                         (1.0 + s[2] * p.xi[2] / g) / 8.0;
        e += w * (op.at(point_geometry(op.X(), {s[0] * g, s[1] * g, s[2] * g})) * u);
      }
    }
    out[k].strain = e;
    out[k].stress = op.D() * (e - thermal_strain(p.sd, temperatures, t_ref, op.alpha()));
  }
  return out;
}

std::vector<Eigen::Matrix<double, 6, SolidHexa20::NUM_DOFS>>
SolidHexa20::gauss_point_strain_operators() const {
  const StrainOperator op(eid_, nodes_, model_, std::get<PSolid>(model_.property(pid_)),
                          formulation_ == Formulation::AssumedStrain);
  return op.gauss_operators();
}

std::vector<EqIndex> SolidHexa20::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> idx;
  idx.reserve(NUM_DOFS);
  for (NodeId n : nodes_)
    for (int d = 0; d < 3; ++d)
      idx.push_back(dof_map.eq_index(n, d));
  return idx;
}

} // namespace vibestran
