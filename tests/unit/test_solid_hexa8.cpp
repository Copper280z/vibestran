// tests/unit/test_solid_hexa8.cpp
// Element-level tests of the Q1/E21 CHEXA8 (solid_hexa8.hpp): zero-energy
// modes, covariance under rotation and node relabelling, consistency of the
// condensed thermal load and recovery, and the consistent mass.  The
// solver-level comparison with MYSTRAN and the exact patch tests are in
// tests/integration/test_hexa8_reference.cpp.

#include <gtest/gtest.h>
#include "elements/solid_hexa8.hpp"
#include <Eigen/Eigenvalues>
#include <cmath>

using namespace vibestran;

namespace {

using Shape = std::array<std::array<double, 3>, 8>;

const Shape kCube{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}};
// A randomly distorted element (all Jacobians positive).
const Shape kDistorted{{{0.0, 0.0, 0.0},
                        {1.2, 0.1, -0.1},
                        {1.1, 0.9, 0.2},
                        {-0.1, 1.3, 0.1},
                        {0.1, -0.2, 0.9},
                        {1.3, 0.2, 1.1},
                        {0.9, 1.1, 1.3},
                        {0.2, 0.8, 1.0}}};
const Shape kNeedle{{{0, 0, 0}, {20, 0, 0}, {20, 1, 0}, {0, 1, 0}, {0, 0, 1}, {20, 0, 1}, {20, 1, 1}, {0, 1, 1}}};
const Shape kSlab{{{0, 0, 0}, {10, 0, 0}, {10, 10, 0}, {0, 10, 0}, {0, 0, 1}, {10, 0, 1}, {10, 10, 1}, {0, 10, 1}}};
const Shape kTrapezoid{{{0, 0, 0}, {2, 0, 0}, {1.5, 1, 0}, {0.5, 1, 0}, {0, 0, 1}, {2, 0, 1}, {1.5, 1, 1}, {0.5, 1, 1}}};

Model hexa_model(const Shape &xyz, double nu = 0.3, SolidFormulation isop = SolidFormulation::EAS,
                 int order = 2, double alpha = 0.0) {
  Model m;
  for (int i = 0; i < 8; ++i) {
    GridPoint g;
    g.id = NodeId{i + 1};
    const auto &p = xyz[static_cast<size_t>(i)];
    g.position = Vec3{p[0], p[1], p[2]};
    m.nodes[g.id] = g;
  }
  Mat1 mat;
  mat.id = MaterialId{1};
  mat.E = 1.0e6;
  mat.nu = nu;
  mat.G = mat.E / (2.0 * (1.0 + nu));
  mat.rho = 3.0;
  mat.A = alpha;
  m.materials[mat.id] = mat;
  PSolid ps;
  ps.pid = PropertyId{1};
  ps.mid = MaterialId{1};
  ps.isop = isop;
  ps.integration_order = order;
  m.properties[ps.pid] = ps;
  return m;
}

SolidHexa8 element(const Model &m, std::array<int, 8> order = {1, 2, 3, 4, 5, 6, 7, 8}) {
  const auto id = [&](size_t i) { return NodeId{order[i]}; };
  return SolidHexa8(ElementId{1}, PropertyId{1},
                    {id(0), id(1), id(2), id(3), id(4), id(5), id(6), id(7)}, m);
}

int zero_modes(const Eigen::MatrixXd &K) {
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
  const Eigen::VectorXd ev = es.eigenvalues();
  const double tol = 1e-9 * ev.cwiseAbs().maxCoeff();
  int zeros = 0;
  for (int i = 0; i < ev.size(); ++i) {
    EXPECT_GT(ev(i), -tol) << "negative eigenvalue " << ev(i);
    if (std::abs(ev(i)) < tol)
      ++zeros;
  }
  return zeros;
}

} // namespace

TEST(SolidHexa8, HasExactlySixZeroEnergyModes) {
  // Rigid-body modes only, for distorted, 20:1 needle, 10:10:1 slab and
  // trapezoidal shapes, compressible and nearly incompressible, 2x2x2 and
  // 3x3x3, enhanced and full.  An enhanced mode set with a compatible strain
  // in its span (e.g. adding e_xixi = eta zeta, "E24") would show 3 more.
  for (const Shape *s : {&kCube, &kDistorted, &kNeedle, &kSlab, &kTrapezoid})
    for (double nu : {0.3, 0.4999})
      for (int order : {2, 3})
        for (SolidFormulation f : {SolidFormulation::EAS, SolidFormulation::Full}) {
          const Model m = hexa_model(*s, nu, f, order);
          EXPECT_EQ(zero_modes(element(m).stiffness_matrix()), 6)
              << "nu " << nu << " order " << order;
        }
}

TEST(SolidHexa8, StiffnessRotatesWithTheElement) {
  // K(R X) = Q K(X) Q^T, Q = blockdiag(R): no preferred direction.
  const Eigen::Matrix3d R =
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.3, -0.5, 0.8).normalized()).toRotationMatrix();
  Shape rotated;
  for (size_t i = 0; i < 8; ++i) {
    const Eigen::Vector3d x(kDistorted[i][0], kDistorted[i][1], kDistorted[i][2]);
    const Eigen::Vector3d y = R * x;
    rotated[i] = {y(0), y(1), y(2)};
  }
  const Eigen::MatrixXd K0 = element(hexa_model(kDistorted)).stiffness_matrix();
  const Eigen::MatrixXd K1 = element(hexa_model(rotated)).stiffness_matrix();
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(24, 24);
  for (int i = 0; i < 8; ++i)
    Q.block<3, 3>(3 * i, 3 * i) = R;
  EXPECT_LT((K1 - Q * K0 * Q.transpose()).cwiseAbs().maxCoeff(), 1e-9 * K0.cwiseAbs().maxCoeff());
}

TEST(SolidHexa8, StiffnessDoesNotDependOnTheNodeNumbering) {
  // Orientation-preserving relabellings: a cyclic permutation of the local
  // axes (xi -> eta -> zeta) and a 90-degree turn about zeta.  The element
  // stiffness, mapped back to the original grid order, is unchanged.
  const Model m = hexa_model(kDistorted);
  const Eigen::MatrixXd K0 = element(m).stiffness_matrix();
  for (const std::array<int, 8> &order :
       {std::array<int, 8>{1, 4, 8, 5, 2, 3, 7, 6}, std::array<int, 8>{2, 3, 4, 1, 6, 7, 8, 5}}) {
    const Eigen::MatrixXd K = element(m, order).stiffness_matrix();
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(24, 24); // local dof -> original dof
    for (int i = 0; i < 8; ++i)
      P.block<3, 3>(3 * (order[static_cast<size_t>(i)] - 1), 3 * i).setIdentity();
    EXPECT_LT((P * K * P.transpose() - K0).cwiseAbs().maxCoeff(), 1e-9 * K0.cwiseAbs().maxCoeff());
  }
}

TEST(SolidHexa8, FreeThermalExpansionIsStressFreeOnADistortedElement) {
  // The condensed thermal load must be K times the free expansion
  // u = alpha dT x (the internal-mode part G' H^-1 q vanishes only for
  // uniform temperature), and recovery of that displacement must give zero
  // stress at the centre and the corners.  Linear temperature: the exact
  // strain alpha T(x) I is not compatible in a single trilinear element, so
  // only the uniform case is exact here; the solver-level gradient test is
  // in test_hexa8_reference.cpp.
  const double alpha = 1e-5, dT = 100.0;
  const Model m = hexa_model(kDistorted, 0.3, SolidFormulation::EAS, 2, alpha);
  const SolidHexa8 e = element(m);
  const std::vector<double> T(8, dT);
  const Eigen::VectorXd f = e.thermal_load(T, 0.0);
  Eigen::VectorXd u(24);
  for (int i = 0; i < 8; ++i)
    for (int d = 0; d < 3; ++d)
      u(3 * i + d) = alpha * dT * kDistorted[static_cast<size_t>(i)][static_cast<size_t>(d)];
  const Eigen::VectorXd Ku = e.stiffness_matrix() * u;
  EXPECT_LT((Ku - f).cwiseAbs().maxCoeff(), 1e-10 * f.cwiseAbs().maxCoeff());
  const auto pts = e.recover(std::span<const double>(u.data(), 24), T, 0.0);
  for (const auto &p : pts) {
    EXPECT_LT(p.stress.cwiseAbs().maxCoeff(), 1e-9 * 1e6 * alpha * dT);
    EXPECT_NEAR(p.strain(0), alpha * dT, 1e-12);
    EXPECT_NEAR(p.strain(3), 0.0, 1e-12);
  }
}

TEST(SolidHexa8, RecoveredConstantStrainIsExactAtEveryPoint) {
  // A homogeneous strain field u = A x on a distorted element is recovered
  // exactly at the centre and all corners: the condensed internal modes are
  // zero for it.  (Extrapolated corner recovery is checked by the
  // MacNeal-Harder patch test, whose concave element has one.)
  const Model m = hexa_model(kDistorted);
  const SolidHexa8 e = element(m);
  Eigen::Matrix3d A;
  A << 1.0, 0.4, -0.2, 0.1, -0.5, 0.3, 0.2, 0.6, 0.8;
  A *= 1e-3;
  Eigen::VectorXd u(24);
  for (int i = 0; i < 8; ++i) {
    const auto &x = kDistorted[static_cast<size_t>(i)];
    u.segment<3>(3 * i) = A * Eigen::Vector3d(x[0], x[1], x[2]);
  }
  const Eigen::Matrix3d eps = 0.5 * (A + A.transpose());
  for (const auto &p : e.recover(std::span<const double>(u.data(), 24), {}, 0.0)) {
    EXPECT_NEAR(p.strain(0), eps(0, 0), 1e-15);
    EXPECT_NEAR(p.strain(1), eps(1, 1), 1e-15);
    EXPECT_NEAR(p.strain(2), eps(2, 2), 1e-15);
    EXPECT_NEAR(p.strain(3), 2 * eps(0, 1), 1e-15);
    EXPECT_NEAR(p.strain(4), 2 * eps(1, 2), 1e-15);
    EXPECT_NEAR(p.strain(5), 2 * eps(2, 0), 1e-15);
  }
}

TEST(SolidHexa8, ConsistentMassCarriesTheElementMass) {
  // Sum of one direction's mass terms = rho V; the element volume of the
  // trapezoid is (2 + 1) / 2 * 1 * 1.
  const Model m = hexa_model(kTrapezoid);
  const Eigen::MatrixXd M = element(m).mass_matrix();
  double total = 0.0;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
      total += M(3 * i, 3 * j);
  EXPECT_NEAR(total, 3.0 * 1.5, 1e-12);
  EXPECT_LT((M - M.transpose()).cwiseAbs().maxCoeff(), 1e-15);
}

TEST(SolidHexa8, InvertedElementsAreRejected) {
  Shape inverted = kCube;
  std::swap(inverted[1], inverted[3]); // mirrors the bottom face
  std::swap(inverted[5], inverted[7]);
  EXPECT_THROW((void)element(hexa_model(inverted)).stiffness_matrix(), SolverError);
}
