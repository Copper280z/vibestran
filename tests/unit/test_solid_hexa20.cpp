// tests/unit/test_solid_hexa20.cpp
// Element-level tests of the projected assumed-strain CHEXA20
// (solid_hexa20.hpp): rank, exact constant strain and constant-stress
// consistency on curved distorted shapes, invariance under rotation and node
// relabelling, free thermal expansion, mass, and input errors.  The
// benchmark-level checks (thin curved shells, patch, Lame, aspect ratio) are
// in tests/integration/test_hexa20_reference.cpp.

#include <gtest/gtest.h>
#include "elements/hexa20_shape.hpp"
#include "elements/solid_hexa20.hpp"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <random>

using namespace vibestran;

namespace {

using Shape = std::array<Eigen::Vector3d, 20>;
using Mat60 = Eigen::Matrix<double, 60, 60>;

Shape parent() {
  Shape X;
  for (std::size_t a = 0; a < 20; ++a)
    X[a] = Eigen::Vector3d(kHexa20Nodes[a][0], kHexa20Nodes[a][1], kHexa20Nodes[a][2]);
  return X;
}

Shape scaled(const Eigen::Vector3d &s) {
  Shape X = parent();
  for (auto &x : X)
    x = x.cwiseProduct(s);
  return X;
}

/// Randomly distorted, with curved, off-centre midside nodes.
Shape curved_distorted() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Shape X = scaled({2.0, 1.5, 1.0});
  for (std::size_t a = 0; a < 20; ++a)
    for (int k = 0; k < 3; ++k)
      X[a](k) += (a < 8 ? 0.12 : 0.3) * u(rng);
  return X;
}

/// A doubly curved shell segment like the pinched hemisphere's elements:
/// R = 10, t = 0.04, spanning 0.2 rad each way (aspect ratio about 50).
Shape shell_segment() {
  Shape X;
  for (std::size_t a = 0; a < 20; ++a) {
    const auto &c = kHexa20Nodes[a];
    const double th = 0.1 * c[0], ph = 0.1 * c[1], r = 10.0 + 0.02 * c[2];
    X[a] = Eigen::Vector3d(r * std::cos(ph) * std::cos(th), r * std::cos(ph) * std::sin(th),
                           r * std::sin(ph));
  }
  return X;
}

Model hexa_model(const Shape &X, double nu = 0.3, SolidFormulation isop = SolidFormulation::EAS,
                 double alpha = 0.0) {
  Model m;
  for (std::size_t a = 0; a < 20; ++a) {
    GridPoint g;
    g.id = NodeId{static_cast<int>(a) + 1};
    g.position = Vec3{X[a](0), X[a](1), X[a](2)};
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
  ps.integration_order = 3;
  m.properties[ps.pid] = ps;
  return m;
}

SolidHexa20 element(const Model &m, const std::array<int, 20> &order) {
  return SolidHexa20(ElementId{1}, PropertyId{1},
                     hexa20_node_array(std::array<NodeId, 20>{
                         NodeId{order[0]},  NodeId{order[1]},  NodeId{order[2]},
                         NodeId{order[3]},  NodeId{order[4]},  NodeId{order[5]},
                         NodeId{order[6]},  NodeId{order[7]},  NodeId{order[8]},
                         NodeId{order[9]},  NodeId{order[10]}, NodeId{order[11]},
                         NodeId{order[12]}, NodeId{order[13]}, NodeId{order[14]},
                         NodeId{order[15]}, NodeId{order[16]}, NodeId{order[17]},
                         NodeId{order[18]}, NodeId{order[19]}}),
                     m);
}

std::array<int, 20> identity_order() {
  std::array<int, 20> o{};
  for (int i = 0; i < 20; ++i)
    o[static_cast<std::size_t>(i)] = i + 1;
  return o;
}

SolidHexa20 element(const Model &m) { return element(m, identity_order()); }

/// Orthonormal basis of the six rigid-body motions of the shape.
Eigen::Matrix<double, 60, 6> rigid_modes(const Shape &X) {
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  for (const auto &x : X)
    c += x / 20.0;
  Eigen::Matrix<double, 60, 6> R = Eigen::Matrix<double, 60, 6>::Zero();
  for (std::size_t a = 0; a < 20; ++a) {
    const auto r = static_cast<Eigen::Index>(3 * a);
    for (int k = 0; k < 3; ++k) {
      R(r + k, k) = 1.0;
      R.block<3, 1>(r, 3 + k) = Eigen::Vector3d::Unit(k).cross(X[a] - c);
    }
  }
  return Eigen::HouseholderQR<Eigen::Matrix<double, 60, 6>>(R).householderQ() *
         Eigen::Matrix<double, 60, 6>::Identity();
}

/// The 54 non-rigid eigenvalues of K, in ascending order.
Eigen::VectorXd deformation_spectrum(const Mat60 &K, const Shape &X) {
  const auto Q = rigid_modes(X);
  const Mat60 P = Mat60::Identity() - Q * Q.transpose();
  const Eigen::SelfAdjointEigenSolver<Mat60> es(P * K * P);
  return es.eigenvalues().tail(54);
}

Eigen::Matrix<double, 60, 1> linear_field(const Shape &X, const Eigen::Matrix3d &A) {
  Eigen::Matrix<double, 60, 1> u;
  for (std::size_t a = 0; a < 20; ++a)
    u.segment<3>(static_cast<Eigen::Index>(3 * a)) = A * X[a];
  return u;
}

/// Engineering Voigt strain (xx yy zz xy yz zx) of the symmetric part of A.
Eigen::Matrix<double, 6, 1> voigt(const Eigen::Matrix3d &A) {
  const Eigen::Matrix3d e = 0.5 * (A + A.transpose());
  Eigen::Matrix<double, 6, 1> v;
  v << e(0, 0), e(1, 1), e(2, 2), 2 * e(0, 1), 2 * e(1, 2), 2 * e(2, 0);
  return v;
}

std::vector<Eigen::Matrix3d> constant_strain_fields() {
  std::vector<Eigen::Matrix3d> out;
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j) {
      Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
      A(i, j) = 1e-3;
      A(j, i) = 1e-3;
      out.push_back(A);
    }
  return out;
}

} // namespace

TEST(SolidHexa20, HasOnlyTheSixRigidBodyZeroModes) {
  // Rigid motions give zero force to round-off, and no deformation mode is
  // soft compared with the standard element (a spurious mode would show as
  // an eigenvalue ratio near 0; 2x2x2 integration has six).  At nu = 0.4999
  // the B-bar relieves the volumetric stiffness, so the ratio is about
  // (1 - 2 nu) there.  Shapes: cube, curved-distorted, 20:1:1 needle,
  // 10:10:1 slab, trapezoid, and a thin doubly curved shell segment.
  Shape trap = parent();
  for (auto &x : trap)
    x(0) *= 1.0 + 0.4 * x(1);
  const std::vector<std::pair<const char *, Shape>> shapes{
      {"cube", parent()},
      {"curved-distorted", curved_distorted()},
      {"needle", scaled({10.0, 0.5, 0.5})},
      {"slab", scaled({5.0, 5.0, 0.5})},
      {"trapezoid", trap},
      {"shell segment", shell_segment()}};
  for (const auto &[name, X] : shapes)
    for (double nu : {0.3, 0.4999}) {
      const Model m = hexa_model(X, nu);
      const Mat60 K = element(m).stiffness_matrix();
      const Mat60 Kfull = element(hexa_model(X, nu, SolidFormulation::Full)).stiffness_matrix();
      EXPECT_LT((K * rigid_modes(X)).cwiseAbs().maxCoeff(), 1e-13 * K.cwiseAbs().maxCoeff())
          << name << " nu " << nu;
      const Eigen::VectorXd ev = deformation_spectrum(K, X);
      const Eigen::VectorXd evf = deformation_spectrum(Kfull, X);
      const double floor = nu > 0.49 ? 1e-3 : 1e-2;
      EXPECT_GT((ev.array() / evf.array()).minCoeff(), floor) << name << " nu " << nu;
    }
}

TEST(SolidHexa20, ConstantStrainIsExactEverywhereOnCurvedDistortedShapes) {
  // The patch test at element level: every linear field gives its constant
  // strain at all 27 Gauss points and at the centre and corners, with
  // curved, off-centre midside nodes (MYSTRAN's interpolated variant missed
  // this by 1e-3).
  for (const Shape &X : {curved_distorted(), shell_segment()}) {
    const Model m = hexa_model(X);
    const SolidHexa20 e = element(m);
    const auto Bg = e.gauss_point_strain_operators();
    for (const auto &A : constant_strain_fields()) {
      const auto u = linear_field(X, A);
      const auto eps = voigt(A);
      for (const auto &B : Bg)
        EXPECT_LT((B * u - eps).cwiseAbs().maxCoeff(), 1e-9 * eps.cwiseAbs().maxCoeff());
      for (const auto &p : e.recover(std::span<const double>(u.data(), 60), {}, 0.0))
        EXPECT_LT((p.strain - eps).cwiseAbs().maxCoeff(), 1e-9 * eps.cwiseAbs().maxCoeff());
    }
  }
}

TEST(SolidHexa20, DoesNoExtraWorkAgainstConstantStress) {
  // The other half of the mesh patch test: K u for a linear u (the nodal
  // forces of its constant stress) equals that of the compatible element.
  // A local-frame projection without the zero-mean correction reproduces
  // constant strain but fails this by 2.6% and the mesh patch test by 5.8%.
  for (const Shape &X : {curved_distorted(), shell_segment()}) {
    const Mat60 K = element(hexa_model(X)).stiffness_matrix();
    const Mat60 Kf = element(hexa_model(X, 0.3, SolidFormulation::Full)).stiffness_matrix();
    for (const auto &A : constant_strain_fields()) {
      const auto u = linear_field(X, A);
      const Eigen::Matrix<double, 60, 1> fK = K * u, fF = Kf * u;
      EXPECT_LT((fK - fF).cwiseAbs().maxCoeff(), 1e-11 * fF.cwiseAbs().maxCoeff());
    }
  }
}

TEST(SolidHexa20, StiffnessIsInvariantToRotationAndNodeRelabelling) {
  // K(R X) = Q K(X) Q^T; and the 24 orientation-preserving relabellings of
  // the local axes (permutations with sign changes of xi, eta, zeta) give the
  // same stiffness once mapped back: no direction is preferred.
  const Shape X = curved_distorted();
  const Mat60 K0 = element(hexa_model(X)).stiffness_matrix();
  const Eigen::Matrix3d R =
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.3, -0.5, 0.8).normalized()).toRotationMatrix();
  Shape Xr = X;
  for (auto &x : Xr)
    x = R * x;
  const Mat60 Kr = element(hexa_model(Xr)).stiffness_matrix();
  Mat60 Q = Mat60::Zero();
  for (int a = 0; a < 20; ++a)
    Q.block<3, 3>(3 * a, 3 * a) = R;
  EXPECT_LT((Kr - Q * K0 * Q.transpose()).cwiseAbs().maxCoeff(), 1e-12 * K0.cwiseAbs().maxCoeff());

  const Model m = hexa_model(X);
  int relabellings = 0;
  std::array<int, 3> perm{0, 1, 2};
  do {
    for (int signs = 0; signs < 8; ++signs) {
      Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
      for (int i = 0; i < 3; ++i)
        T(i, perm[static_cast<std::size_t>(i)]) = (signs >> i) & 1 ? -1.0 : 1.0;
      if (T.determinant() < 0.0)
        continue;
      // New local node i sits at natural point T * xi_i of the old element.
      std::array<int, 20> order{};
      for (std::size_t i = 0; i < 20; ++i) {
        const Eigen::Vector3d p =
            T * Eigen::Vector3d(kHexa20Nodes[i][0], kHexa20Nodes[i][1], kHexa20Nodes[i][2]);
        for (std::size_t j = 0; j < 20; ++j)
          if ((Eigen::Vector3d(kHexa20Nodes[j][0], kHexa20Nodes[j][1], kHexa20Nodes[j][2]) - p)
                  .norm() < 1e-12)
            order[i] = static_cast<int>(j) + 1;
      }
      const Mat60 K = element(m, order).stiffness_matrix();
      Mat60 P = Mat60::Zero();
      for (int i = 0; i < 20; ++i)
        P.block<3, 3>(3 * (order[static_cast<std::size_t>(i)] - 1), 3 * i).setIdentity();
      EXPECT_LT((P * K * P.transpose() - K0).cwiseAbs().maxCoeff(),
                1e-12 * K0.cwiseAbs().maxCoeff());
      ++relabellings;
    }
  } while (std::next_permutation(perm.begin(), perm.end()));
  EXPECT_EQ(relabellings, 24);
}

TEST(SolidHexa20, FreeThermalExpansionIsStressFree) {
  // Uniform temperature on the curved distorted element: the thermal load is
  // K times the free expansion u = alpha dT x, and recovery of that
  // displacement gives zero stress at the centre and corners.
  const double alpha = 1e-5, dT = 100.0;
  const Shape X = curved_distorted();
  const Model m = hexa_model(X, 0.3, SolidFormulation::EAS, alpha);
  const SolidHexa20 e = element(m);
  const std::vector<double> T(20, dT);
  const Eigen::VectorXd f = e.thermal_load(T, 0.0);
  const auto u = linear_field(X, alpha * dT * Eigen::Matrix3d::Identity());
  const Eigen::VectorXd Ku = e.stiffness_matrix() * u;
  EXPECT_LT((Ku - f).cwiseAbs().maxCoeff(), 1e-10 * f.cwiseAbs().maxCoeff());
  for (const auto &p : e.recover(std::span<const double>(u.data(), 60), T, 0.0)) {
    EXPECT_LT(p.stress.cwiseAbs().maxCoeff(), 1e-9 * 1e6 * alpha * dT);
    EXPECT_NEAR(p.strain(0), alpha * dT, 1e-12);
  }
}

TEST(SolidHexa20, ConsistentMassCarriesTheElementMassAndLumpsPositive) {
  // One direction's mass terms sum to rho V; V = 8, the parent cube with x
  // scaled by 1 + 0.4 y (mean 1).  The HRZ lumped mass is positive at every
  // grid, unlike row-sum lumping, which gives negative corner masses for the
  // 20-node element.
  Shape X = parent();
  for (auto &x : X)
    x(0) *= 1.0 + 0.4 * x(1);
  const Model m = hexa_model(X);
  const SolidHexa20 e = element(m);
  const auto M = e.mass_matrix();
  double total = 0.0;
  for (int a = 0; a < 20; ++a)
    for (int b = 0; b < 20; ++b)
      total += M(3 * a, 3 * b);
  EXPECT_NEAR(total, 3.0 * 8.0, 1e-10);
  EXPECT_LT((M - M.transpose()).cwiseAbs().maxCoeff(), 1e-14);
  const auto L = e.lumped_mass_matrix();
  EXPECT_GT(L.diagonal().minCoeff(), 0.0);
  EXPECT_NEAR(L.diagonal().sum(), 3.0 * 3.0 * 8.0, 1e-10);
}

TEST(SolidHexa20, OmittedMidsideGridsAndInvertedElementsAreRejected) {
  const Model m = hexa_model(parent());
  auto order = identity_order();
  std::array<NodeId, 20> ids = hexa20_node_array(std::array<NodeId, 20>{
      NodeId{1},  NodeId{2},  NodeId{3},  NodeId{4},  NodeId{5},  NodeId{6},  NodeId{7},
      NodeId{8},  NodeId{9},  NodeId{10}, NodeId{11}, NodeId{0},  NodeId{13}, NodeId{14},
      NodeId{15}, NodeId{16}, NodeId{17}, NodeId{18}, NodeId{19}, NodeId{20}});
  EXPECT_THROW(SolidHexa20(ElementId{1}, PropertyId{1}, ids, m), SolverError);

  // Mirror the element through x = 0: a negative Jacobian everywhere.
  Shape Xm = parent();
  for (auto &x : Xm)
    x(0) = -x(0);
  EXPECT_THROW((void)element(hexa_model(Xm), order).stiffness_matrix(), SolverError);
}
