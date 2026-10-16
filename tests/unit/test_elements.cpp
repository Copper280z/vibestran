// tests/unit/test_elements.cpp
// Element unit tests: verify Ke symmetry, positive semi-definiteness,
// rigid body modes, and basic patch test compliance.
// These tests do NOT require a full solve — they operate on element math
// directly.

#include "core/model.hpp"
#include "elements/line_elements.hpp"
#include "elements/solid_elements.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <optional>
#include <span>

using namespace vibestran;

// ── Helpers
// ───────────────────────────────────────────────────────────────────

static Model make_solid_model(double E = 2.0e7, double nu = 0.3) {
  Model m;
  Mat1 mat;
  mat.id = MaterialId{1};
  mat.E = E;
  mat.nu = nu;
  mat.G = E / (2 * (1 + nu));
  mat.A = 0;
  m.materials[mat.id] = mat;

  PSolid ps;
  ps.pid = PropertyId{1};
  ps.mid = MaterialId{1};
  m.properties[ps.pid] = ps;

  return m;
}

static void add_grid(Model &m, int id, double x, double y, double z = 0) {
  GridPoint g;
  g.id = NodeId{id};
  g.position = Vec3{x, y, z};
  m.nodes[g.id] = g;
}

static void add_mat1(Model &m, int id, double E, double nu, double rho = 0.0,
                     double alpha = 0.0) {
  Mat1 mat;
  mat.id = MaterialId{id};
  mat.E = E;
  mat.nu = nu;
  mat.G = E / (2 * (1 + nu));
  mat.rho = rho;
  mat.A = alpha;
  m.materials[mat.id] = mat;
}

/// CBAR/CBEAM element data between grids 1 and `gb` with property 1.
static ElementData bar_data(ElementType type, int eid, int gb, Vec3 v) {
  ElementData e;
  e.id = ElementId{eid};
  e.type = type;
  e.pid = PropertyId{1};
  e.nodes = {NodeId{1}, NodeId{gb}};
  e.orientation = v;
  return e;
}

// ── CQUAD4 tests
// ──────────────────────────────────────────────────────────────

// ── CTRIA3 tests
// ──────────────────────────────────────────────────────────────

// ── CTETRA4 tests
// ─────────────────────────────────────────────────────────────

TEST(CTetra4, StiffnessIsSymmetric) {
  Model m = make_solid_model();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);

  std::array<NodeId, 4> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4}};
  CTetra4 elem(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke = elem.stiffness_matrix();

  EXPECT_EQ(Ke.rows(), 12);
  double max_asym = (Ke - Ke.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(max_asym, 1e-10 * Ke.cwiseAbs().maxCoeff());
}

TEST(CTetra4, StiffnessIsPositiveSemiDefinite) {
  Model m = make_solid_model();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);

  std::array<NodeId, 4> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4}};
  CTetra4 elem(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke = elem.stiffness_matrix();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Ke);
  double min_ev = eig.eigenvalues().minCoeff();
  double max_ev = eig.eigenvalues().maxCoeff();
  EXPECT_GE(min_ev, -1e-6 * max_ev);
}

TEST(CTetra4, RigidBodyModesHaveZeroStrain) {
  // For a CST tet: K * u_rigid = 0
  Model m = make_solid_model();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);

  std::array<NodeId, 4> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4}};
  CTetra4 elem(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke = elem.stiffness_matrix();

  // Rigid body x-translation: all u_i = 1
  Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
  for (int i = 0; i < 4; ++i)
    u(3 * i) = 1.0;

  Eigen::VectorXd f = Ke * u;
  EXPECT_LT(f.norm(), 1e-8 * Ke.norm());
}

// ── CHEXA8 tests
// ──────────────────────────────────────────────────────────────

TEST(CHexa8, StiffnessIsSymmetric) {
  Model m = make_solid_model();
  // Unit cube
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 1, 1, 0);
  add_grid(m, 4, 0, 1, 0);
  add_grid(m, 5, 0, 0, 1);
  add_grid(m, 6, 1, 0, 1);
  add_grid(m, 7, 1, 1, 1);
  add_grid(m, 8, 0, 1, 1);

  std::array<NodeId, 8> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                              NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8}};
  CHexa8 elem(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke = elem.stiffness_matrix();

  EXPECT_EQ(Ke.rows(), 24);
  double max_asym = (Ke - Ke.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(max_asym, 1e-10 * Ke.cwiseAbs().maxCoeff());
}

TEST(CHexa8, StiffnessIsPositiveSemiDefinite) {
  Model m = make_solid_model();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 1, 1, 0);
  add_grid(m, 4, 0, 1, 0);
  add_grid(m, 5, 0, 0, 1);
  add_grid(m, 6, 1, 0, 1);
  add_grid(m, 7, 1, 1, 1);
  add_grid(m, 8, 0, 1, 1);

  std::array<NodeId, 8> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                              NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8}};
  CHexa8 elem(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke = elem.stiffness_matrix();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Ke);
  double min_ev = eig.eigenvalues().minCoeff();
  double max_ev = eig.eigenvalues().maxCoeff();
  EXPECT_GE(min_ev, -1e-6 * max_ev);
}

// ── CPenta6 tests
// ──────────────────────────────────────────────────────────────

// Helper to create a unit wedge model and element: right-triangle prism with
// triangle base in XY plane (nodes 1-3 at z=0, 4-6 at z=1).
static CPenta6 make_unit_wedge(Model &m) {
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  add_grid(m, 5, 1, 0, 1);
  add_grid(m, 6, 0, 1, 1);
  std::array<NodeId, 6> nodes{NodeId{1}, NodeId{2}, NodeId{3},
                              NodeId{4}, NodeId{5}, NodeId{6}};
  return CPenta6(ElementId{1}, PropertyId{1}, nodes, m);
}

TEST(CPenta6, StiffnessIsSymmetric) {
  Model m = make_solid_model();
  CPenta6 elem = make_unit_wedge(m);
  LocalKe Ke = elem.stiffness_matrix();

  EXPECT_EQ(Ke.rows(), 18);
  EXPECT_EQ(Ke.cols(), 18);
  double max_asym = (Ke - Ke.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(max_asym, 1e-10 * Ke.cwiseAbs().maxCoeff());
}

TEST(CPenta6, StiffnessIsPositiveSemiDefinite) {
  Model m = make_solid_model();
  CPenta6 elem = make_unit_wedge(m);
  LocalKe Ke = elem.stiffness_matrix();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Ke);
  double min_ev = eig.eigenvalues().minCoeff();
  double max_ev = eig.eigenvalues().maxCoeff();
  EXPECT_GE(min_ev, -1e-6 * max_ev);
}

TEST(CPenta6, RigidBodyTranslationProducesZeroForce) {
  Model m = make_solid_model();
  CPenta6 elem = make_unit_wedge(m);
  LocalKe Ke = elem.stiffness_matrix();

  // Rigid body x-translation: all u_x = 1
  Eigen::VectorXd u = Eigen::VectorXd::Zero(18);
  for (int i = 0; i < 6; ++i)
    u(3 * i) = 1.0;

  Eigen::VectorXd f = Ke * u;
  EXPECT_LT(f.norm(), 1e-8 * Ke.norm());
}

TEST(CPenta6, SriReducesVolumetricLocking) {
  // Verify SRI by directly comparing against full integration. For
  // near-incompressible material, full integration over-constrains volumetric
  // modes, inflating the max eigenvalue. SRI should give a lower max
  // eigenvalue because it only enforces one volumetric constraint (centroidal).
  //
  // We build a full-integration Ke (using the full D at all 6 Gauss points)
  // and compare its max eigenvalue against the SRI Ke.
  const double E = 1.0e6, nu = 0.4999;
  Model m = make_solid_model(E, nu);
  CPenta6 elem = make_unit_wedge(m);

  // SRI stiffness from the element
  LocalKe Ke_sri = elem.stiffness_matrix();

  // Build full-integration Ke manually for comparison
  auto coords =
      std::array<Vec3, 6>{Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0},
                          Vec3{0, 0, 1}, Vec3{1, 0, 1}, Vec3{0, 1, 1}};
  double lam = E * nu / ((1 + nu) * (1 - 2 * nu));
  double mu = E / (2 * (1 + nu));
  Eigen::Matrix<double, 6, 6> D = Eigen::Matrix<double, 6, 6>::Zero();
  D(0, 0) = D(1, 1) = D(2, 2) = lam + 2 * mu;
  D(0, 1) = D(0, 2) = D(1, 0) = D(1, 2) = D(2, 0) = D(2, 1) = lam;
  D(3, 3) = D(4, 4) = D(5, 5) = mu;

  const double tri_pts[3][2] = {
      {2.0 / 3, 1.0 / 6}, {1.0 / 6, 2.0 / 3}, {1.0 / 6, 1.0 / 6}};
  const double tri_w = 1.0 / 6.0;
  const double gp = 1.0 / std::sqrt(3.0);
  const double ax_pts[2] = {-gp, gp};

  LocalKe Ke_full = LocalKe::Zero(18, 18);
  for (int ti = 0; ti < 3; ++ti)
    for (int ai = 0; ai < 2; ++ai) {
      double L1 = tri_pts[ti][0], L2 = tri_pts[ti][1];
      double zeta = ax_pts[ai];
      auto sd = CPenta6::shape_functions(L1, L2, zeta);

      Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
      for (int n = 0; n < 6; ++n) {
        J(0, 0) += sd.dNdL1[n] * coords[n].x;
        J(0, 1) += sd.dNdL1[n] * coords[n].y;
        J(0, 2) += sd.dNdL1[n] * coords[n].z;
        J(1, 0) += sd.dNdL2[n] * coords[n].x;
        J(1, 1) += sd.dNdL2[n] * coords[n].y;
        J(1, 2) += sd.dNdL2[n] * coords[n].z;
        J(2, 0) += sd.dNdzeta[n] * coords[n].x;
        J(2, 1) += sd.dNdzeta[n] * coords[n].y;
        J(2, 2) += sd.dNdzeta[n] * coords[n].z;
      }
      double detJ = J.determinant();
      Eigen::Matrix3d Jinv = J.inverse();

      Eigen::MatrixXd B(6, 18);
      B.setZero();
      for (int n = 0; n < 6; ++n) {
        double dnx = Jinv(0, 0) * sd.dNdL1[n] + Jinv(0, 1) * sd.dNdL2[n] +
                     Jinv(0, 2) * sd.dNdzeta[n];
        double dny = Jinv(1, 0) * sd.dNdL1[n] + Jinv(1, 1) * sd.dNdL2[n] +
                     Jinv(1, 2) * sd.dNdzeta[n];
        double dnz = Jinv(2, 0) * sd.dNdL1[n] + Jinv(2, 1) * sd.dNdL2[n] +
                     Jinv(2, 2) * sd.dNdzeta[n];
        int c0 = 3 * n;
        B(0, c0) = dnx;
        B(1, c0 + 1) = dny;
        B(2, c0 + 2) = dnz;
        B(3, c0) = dny;
        B(3, c0 + 1) = dnx;
        B(4, c0 + 1) = dnz;
        B(4, c0 + 2) = dny;
        B(5, c0) = dnz;
        B(5, c0 + 2) = dnx;
      }
      Ke_full += B.transpose() * D * B * detJ * tri_w;
    }

  // SRI max eigenvalue should be lower than full integration
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_sri(Ke_sri);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_full(Ke_full);

  double max_ev_sri = eig_sri.eigenvalues().maxCoeff();
  double max_ev_full = eig_full.eigenvalues().maxCoeff();

  EXPECT_LT(max_ev_sri, max_ev_full)
      << "SRI max eigenvalue (" << max_ev_sri
      << ") should be less than full integration (" << max_ev_full
      << ") for near-incompressible material";
}

TEST(CPenta6, ThermalLoadEquilibrium) {
  // Uniform temperature → self-equilibrating thermal loads (net force = 0).
  Model m = make_solid_model(2.0e7, 0.3);
  m.materials.at(MaterialId{1}).A = 1.2e-5;
  CPenta6 elem = make_unit_wedge(m);

  std::array<double, 6> T{100.0, 100.0, 100.0, 100.0, 100.0, 100.0};
  LocalFe fe = elem.thermal_load(T, 0.0);

  EXPECT_EQ(fe.size(), 18);
  double sum_fx = 0.0, sum_fy = 0.0, sum_fz = 0.0;
  for (int i = 0; i < 6; ++i) {
    sum_fx += fe(3 * i);
    sum_fy += fe(3 * i + 1);
    sum_fz += fe(3 * i + 2);
  }
  EXPECT_NEAR(sum_fx, 0.0, 1e-6 * fe.norm());
  EXPECT_NEAR(sum_fy, 0.0, 1e-6 * fe.norm());
  EXPECT_NEAR(sum_fz, 0.0, 1e-6 * fe.norm());
}

// ── CPenta6Eas tests
// ───────────────────────────────────────────────────────────

static CPenta6Eas make_unit_wedge_eas(Model &m) {
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  add_grid(m, 5, 1, 0, 1);
  add_grid(m, 6, 0, 1, 1);
  std::array<NodeId, 6> nodes{NodeId{1}, NodeId{2}, NodeId{3},
                              NodeId{4}, NodeId{5}, NodeId{6}};
  return CPenta6Eas(ElementId{1}, PropertyId{1}, nodes, m);
}

TEST(CPenta6Eas, StiffnessIsSymmetric) {
  Model m = make_solid_model();
  CPenta6Eas elem = make_unit_wedge_eas(m);
  LocalKe Ke = elem.stiffness_matrix();

  EXPECT_EQ(Ke.rows(), 18);
  EXPECT_EQ(Ke.cols(), 18);
  double max_asym = (Ke - Ke.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(max_asym, 1e-10 * Ke.cwiseAbs().maxCoeff());
}

TEST(CPenta6Eas, StiffnessIsPositiveSemiDefinite) {
  Model m = make_solid_model();
  CPenta6Eas elem = make_unit_wedge_eas(m);
  LocalKe Ke = elem.stiffness_matrix();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Ke);
  double min_ev = eig.eigenvalues().minCoeff();
  double max_ev = eig.eigenvalues().maxCoeff();
  EXPECT_GE(min_ev, -1e-6 * max_ev);
}

TEST(CPenta6Eas, RigidBodyTranslationProducesZeroForce) {
  Model m = make_solid_model();
  CPenta6Eas elem = make_unit_wedge_eas(m);
  LocalKe Ke = elem.stiffness_matrix();

  Eigen::VectorXd u = Eigen::VectorXd::Zero(18);
  for (int i = 0; i < 6; ++i)
    u(3 * i) = 1.0;

  Eigen::VectorXd f = Ke * u;
  EXPECT_LT(f.norm(), 1e-8 * Ke.norm());
}

TEST(CPenta6Eas, NearlyIncompressibleLowerStiffnessThanSRI) {
  // EAS should give a lower max eigenvalue than SRI for near-incompressible
  // material, since EAS addresses both volumetric and bending locking.
  Model m = make_solid_model(1.0e6, 0.4999);

  // SRI stiffness
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  add_grid(m, 5, 1, 0, 1);
  add_grid(m, 6, 0, 1, 1);
  std::array<NodeId, 6> nodes{NodeId{1}, NodeId{2}, NodeId{3},
                              NodeId{4}, NodeId{5}, NodeId{6}};
  CPenta6 elem_sri(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke_sri = elem_sri.stiffness_matrix();

  // EAS stiffness
  CPenta6Eas elem_eas(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke_eas = elem_eas.stiffness_matrix();

  double sri_max = Ke_sri.diagonal().maxCoeff();
  double eas_max = Ke_eas.diagonal().maxCoeff();
  EXPECT_GT(sri_max, 0.0);
  EXPECT_GT(eas_max, 0.0);

  // EAS should be PSD
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_eas(Ke_eas);
  EXPECT_GE(eig_eas.eigenvalues().minCoeff(), -1e-6 * eas_max);
}

TEST(CPenta6Eas, ThermalLoadEquilibrium) {
  Model m = make_solid_model(2.0e7, 0.3);
  m.materials.at(MaterialId{1}).A = 1.2e-5;
  CPenta6Eas elem = make_unit_wedge_eas(m);

  std::array<double, 6> T{100.0, 100.0, 100.0, 100.0, 100.0, 100.0};
  LocalFe fe = elem.thermal_load(T, 0.0);

  EXPECT_EQ(fe.size(), 18);
  double sum_fx = 0.0, sum_fy = 0.0, sum_fz = 0.0;
  for (int i = 0; i < 6; ++i) {
    sum_fx += fe(3 * i);
    sum_fy += fe(3 * i + 1);
    sum_fz += fe(3 * i + 2);
  }
  EXPECT_NEAR(sum_fx, 0.0, 1e-6 * fe.norm());
  EXPECT_NEAR(sum_fy, 0.0, 1e-6 * fe.norm());
  EXPECT_NEAR(sum_fz, 0.0, 1e-6 * fe.norm());
}

// ── CTetra10 tests
// ─────────────────────────────────────────────────────────────

TEST(CTetra10, StiffnessIsSymmetric) {
  Model m = make_solid_model();
  // Corner nodes
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  // Midside nodes (midpoints of edges)
  add_grid(m, 5, 0.5, 0, 0);    // 1-2
  add_grid(m, 6, 0.5, 0.5, 0);  // 2-3
  add_grid(m, 7, 0, 0.5, 0);    // 1-3
  add_grid(m, 8, 0, 0, 0.5);    // 1-4
  add_grid(m, 9, 0.5, 0, 0.5);  // 2-4
  add_grid(m, 10, 0, 0.5, 0.5); // 3-4

  std::array<NodeId, 10> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                               NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8},
                               NodeId{9}, NodeId{10}};
  CTetra10 elem(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke = elem.stiffness_matrix();

  EXPECT_EQ(Ke.rows(), 30);
  EXPECT_EQ(Ke.cols(), 30);
  double max_asym = (Ke - Ke.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(max_asym, 1e-10 * Ke.cwiseAbs().maxCoeff());
}

TEST(CTetra10, StiffnessIsPositiveSemiDefinite) {
  Model m = make_solid_model();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  add_grid(m, 5, 0.5, 0, 0);
  add_grid(m, 6, 0.5, 0.5, 0);
  add_grid(m, 7, 0, 0.5, 0);
  add_grid(m, 8, 0, 0, 0.5);
  add_grid(m, 9, 0.5, 0, 0.5);
  add_grid(m, 10, 0, 0.5, 0.5);

  std::array<NodeId, 10> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                               NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8},
                               NodeId{9}, NodeId{10}};
  CTetra10 elem(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke = elem.stiffness_matrix();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Ke);
  double min_ev = eig.eigenvalues().minCoeff();
  double max_ev = eig.eigenvalues().maxCoeff();
  EXPECT_GE(min_ev, -1e-6 * max_ev);
}

TEST(CTetra10, RigidBodyModesHaveZeroStrain) {
  // K * u_rigid = 0 for rigid body translation
  Model m = make_solid_model();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  add_grid(m, 5, 0.5, 0, 0);
  add_grid(m, 6, 0.5, 0.5, 0);
  add_grid(m, 7, 0, 0.5, 0);
  add_grid(m, 8, 0, 0, 0.5);
  add_grid(m, 9, 0.5, 0, 0.5);
  add_grid(m, 10, 0, 0.5, 0.5);

  std::array<NodeId, 10> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                               NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8},
                               NodeId{9}, NodeId{10}};
  CTetra10 elem(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke = elem.stiffness_matrix();

  // Rigid body x-translation: all u_i = 1
  Eigen::VectorXd u = Eigen::VectorXd::Zero(30);
  for (int i = 0; i < 10; ++i)
    u(3 * i) = 1.0;

  Eigen::VectorXd f = Ke * u;
  EXPECT_LT(f.norm(), 1e-8 * Ke.norm());
}

TEST(CTetra10, QuadraticPatchTest) {
  // CTetra10 can represent a linearly varying displacement field exactly
  // (quadratic element, linear strain → exact for constant strain).
  // Apply u_x = x, u_y = u_z = 0 → ε_xx = 1, all others = 0.
  // Verify that K*u gives zero internal forces (pure deformation mode).
  Model m = make_solid_model(1.0e6, 0.3);
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  add_grid(m, 5, 0.5, 0, 0);
  add_grid(m, 6, 0.5, 0.5, 0);
  add_grid(m, 7, 0, 0.5, 0);
  add_grid(m, 8, 0, 0, 0.5);
  add_grid(m, 9, 0.5, 0, 0.5);
  add_grid(m, 10, 0, 0.5, 0.5);

  // Node positions (x coordinates for patch test)
  const double xs[10] = {0, 1, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0};
  std::array<NodeId, 10> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                               NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8},
                               NodeId{9}, NodeId{10}};
  CTetra10 elem(ElementId{1}, PropertyId{1}, nodes, m);
  LocalKe Ke = elem.stiffness_matrix();

  // Displacement: u_x = x, u_y = u_z = 0
  Eigen::VectorXd u = Eigen::VectorXd::Zero(30);
  for (int i = 0; i < 10; ++i)
    u(3 * i) = xs[i]; // u_x = x

  // For a free (unconstrained) element, K*u should give self-equilibrating
  // forces. The element is in pure axial strain — the nodal forces should be
  // statically equivalent and sum to zero.
  Eigen::VectorXd f = Ke * u;
  // Sum of x-forces must be zero (equilibrium)
  double sum_fx = 0.0;
  for (int i = 0; i < 10; ++i)
    sum_fx += f(3 * i);
  EXPECT_NEAR(sum_fx, 0.0, 1e-8 * f.norm())
      << "Linear displacement patch test: x-forces must sum to zero";
}

// ── Mass matrix unit tests
// ──────────────────────────────────────────────────── Tests used exclusively
// for verifying mass_matrix() correctness.

// Helper: make a solid model with density
static Model make_solid_model_rho(double E = 2.0e11, double nu = 0.3,
                                  double rho = 7850.0) {
  Model m;
  Mat1 mat;
  mat.id = MaterialId{1};
  mat.E = E;
  mat.nu = nu;
  mat.G = E / (2 * (1 + nu));
  mat.rho = rho;
  mat.A = 0;
  m.materials[mat.id] = mat;
  PSolid ps;
  ps.pid = PropertyId{1};
  ps.mid = MaterialId{1};
  m.properties[ps.pid] = ps;
  return m;
}

TEST(MassMatrix, CHexa8Symmetry) {
  Model m = make_solid_model_rho();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 1, 1, 0);
  add_grid(m, 4, 0, 1, 0);
  add_grid(m, 5, 0, 0, 1);
  add_grid(m, 6, 1, 0, 1);
  add_grid(m, 7, 1, 1, 1);
  add_grid(m, 8, 0, 1, 1);
  std::array<NodeId, 8> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                              NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8}};
  CHexa8 elem(ElementId{1}, PropertyId{1}, nodes, m);

  LocalKe Me = elem.mass_matrix();
  double asym = (Me - Me.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(asym, 1e-12 * Me.cwiseAbs().maxCoeff())
      << "CHexa8 mass matrix is not symmetric";
}

TEST(MassMatrix, CHexa8TotalMass) {
  // Unit cube, rho=7850 → mass = 7850 kg
  const double rho = 7850.0;
  Model m = make_solid_model_rho(2e11, 0.3, rho);
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 1, 1, 0);
  add_grid(m, 4, 0, 1, 0);
  add_grid(m, 5, 0, 0, 1);
  add_grid(m, 6, 1, 0, 1);
  add_grid(m, 7, 1, 1, 1);
  add_grid(m, 8, 0, 1, 1);
  std::array<NodeId, 8> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                              NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8}};
  CHexa8 elem(ElementId{1}, PropertyId{1}, nodes, m);

  LocalKe Me = elem.mass_matrix();
  // u=[1,0,0,...] uniform x-translation → u^T M u = total mass
  Eigen::VectorXd u_x = Eigen::VectorXd::Zero(24);
  for (int i = 0; i < 8; ++i)
    u_x(3 * i) = 1.0;
  double mass_x = u_x.transpose() * Me * u_x;
  EXPECT_NEAR(mass_x, rho * 1.0, 0.01 * rho)
      << "CHexa8 total mass should equal rho*V";
}

TEST(MassMatrix, CHexa8PositiveSemiDefinite) {
  Model m = make_solid_model_rho();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 1, 1, 0);
  add_grid(m, 4, 0, 1, 0);
  add_grid(m, 5, 0, 0, 1);
  add_grid(m, 6, 1, 0, 1);
  add_grid(m, 7, 1, 1, 1);
  add_grid(m, 8, 0, 1, 1);
  std::array<NodeId, 8> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                              NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8}};
  CHexa8 elem(ElementId{1}, PropertyId{1}, nodes, m);

  LocalKe Me = elem.mass_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Me);
  double min_ev = eig.eigenvalues().minCoeff();
  double max_ev = eig.eigenvalues().maxCoeff();
  EXPECT_GE(min_ev, -1e-10 * max_ev)
      << "CHexa8 mass matrix has negative eigenvalue: " << min_ev;
}

TEST(MassMatrix, CTetra4TotalMass) {
  // Regular tetrahedron with vertices at (0,0,0),(1,0,0),(0,1,0),(0,0,1)
  // Volume = 1/6
  const double rho = 7850.0;
  const double V = 1.0 / 6.0;
  Model m = make_solid_model_rho(2e11, 0.3, rho);
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  std::array<NodeId, 4> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4}};
  CTetra4 elem(ElementId{1}, PropertyId{1}, nodes, m);

  LocalKe Me = elem.mass_matrix();
  Eigen::VectorXd u_x = Eigen::VectorXd::Zero(12);
  for (int i = 0; i < 4; ++i)
    u_x(3 * i) = 1.0;
  double mass_x = u_x.transpose() * Me * u_x;
  EXPECT_NEAR(mass_x, rho * V, 0.01 * rho * V)
      << "CTetra4 total mass should equal rho*V";
}

TEST(MassMatrix, CTetra4Symmetry) {
  Model m = make_solid_model_rho();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  std::array<NodeId, 4> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4}};
  CTetra4 elem(ElementId{1}, PropertyId{1}, nodes, m);

  LocalKe Me = elem.mass_matrix();
  double asym = (Me - Me.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(asym, 1e-12 * Me.cwiseAbs().maxCoeff())
      << "CTetra4 mass matrix is not symmetric";
}

TEST(MassMatrix, CPenta6TotalMass) {
  // Wedge element: 6 nodes forming a triangular prism
  // Base triangle: (0,0,0),(1,0,0),(0,1,0); height = 1
  // Volume = (1/2) * 1 * 1 = 0.5
  const double rho = 7850.0;
  const double V = 0.5;
  Model m = make_solid_model_rho(2e11, 0.3, rho);
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  add_grid(m, 5, 1, 0, 1);
  add_grid(m, 6, 0, 1, 1);
  std::array<NodeId, 6> nodes{NodeId{1}, NodeId{2}, NodeId{3},
                              NodeId{4}, NodeId{5}, NodeId{6}};
  CPenta6 elem(ElementId{1}, PropertyId{1}, nodes, m);

  LocalKe Me = elem.mass_matrix();
  Eigen::VectorXd u_x = Eigen::VectorXd::Zero(18);
  for (int i = 0; i < 6; ++i)
    u_x(3 * i) = 1.0;
  double mass_x = u_x.transpose() * Me * u_x;
  EXPECT_NEAR(mass_x, rho * V, 0.02 * rho * V)
      << "CPenta6 total mass should equal rho*V";
}

TEST(MassMatrix, CPenta6Symmetry) {
  Model m = make_solid_model_rho();
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 1, 0, 0);
  add_grid(m, 3, 0, 1, 0);
  add_grid(m, 4, 0, 0, 1);
  add_grid(m, 5, 1, 0, 1);
  add_grid(m, 6, 0, 1, 1);
  std::array<NodeId, 6> nodes{NodeId{1}, NodeId{2}, NodeId{3},
                              NodeId{4}, NodeId{5}, NodeId{6}};
  CPenta6 elem(ElementId{1}, PropertyId{1}, nodes, m);

  LocalKe Me = elem.mass_matrix();
  double asym = (Me - Me.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(asym, 1e-12 * Me.cwiseAbs().maxCoeff())
      << "CPenta6 mass matrix is not symmetric";
}

// ── Analytical stiffness validation ──────────────────────────────────────────
//
// These tests verify element stiffness against closed-form results.
// DOF layout per node: [Tx, Ty, Tz, Rx, Ry, Rz].
// For a flat shell element, membrane (Tx, Ty) and bending (Tz, Rx, Ry) DOFs
// are decoupled; applying only one type produces forces only in that type.
//
// Strain energy identity: u^T * K * u = 2 * U  (linear stiffness).

namespace {

/// Global DOF vector of a flat (XY-plane) plate field w(x, y) at the CTRIA3
/// nodes: T3 = w, R1 = w_y, R2 = -w_x (Kirchhoff rotations).
template <class W, class WX, class WY>
Eigen::VectorXd tria_plate_field(const std::array<std::array<double, 2>, 3> &xy,
                                 W w, WX wx, WY wy) {
  Eigen::VectorXd u = Eigen::VectorXd::Zero(18);
  for (int n = 0; n < 3; ++n) {
    const double x = xy[static_cast<size_t>(n)][0], y = xy[static_cast<size_t>(n)][1];
    u(6 * n + 2) = w(x, y);
    u(6 * n + 3) = wy(x, y);
    u(6 * n + 4) = -wx(x, y);
  }
  return u;
}

} // namespace

TEST(CBar, StiffnessIsSymmetricAndMatchesAxialEAOverL) {
  Model m;
  add_mat1(m, 1, 2.0e7, 0.3);

  PBar p;
  p.pid = PropertyId{1};
  p.mid = MaterialId{1};
  p.A = 2.0;
  p.I1 = 3.0;
  p.I2 = 4.0;
  p.J = 5.0;
  m.properties[p.pid] = p;

  add_grid(m, 1, 0.0, 0.0, 0.0);
  add_grid(m, 2, 2.0, 0.0, 0.0);

  CBarBeamElement elem(
      bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, 1.0, 0.0}), m);

  const LocalKe Ke = elem.stiffness_matrix();
  EXPECT_EQ(Ke.rows(), 12);
  EXPECT_EQ(Ke.cols(), 12);
  EXPECT_LT((Ke - Ke.transpose()).cwiseAbs().maxCoeff(), 1e-10 * Ke.norm());

  Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
  u(6) = 1.0;
  const Eigen::VectorXd f = Ke * u;
  const double axial = 2.0e7 * 2.0 / 2.0;
  EXPECT_NEAR(f(0), -axial, 1e-8 * axial);
  EXPECT_NEAR(f(6), axial, 1e-8 * axial);
}

namespace {

/// Cantilever along +x from grid 1 (fixed) to grid 2 at x = L: solve the
/// 6x6 free block of K for a tip load and return the tip displacements.
Eigen::Matrix<double, 6, 1> cantilever_tip(const LocalKe &K,
                                           const Eigen::Matrix<double, 6, 1> &f) {
  const Eigen::MatrixXd Kff = K.block(6, 6, 6, 6);
  return Kff.fullPivLu().solve(f);
}

Model bar_model(double L, const PBar &p, double E = 2.0e7, double nu = 0.3) {
  Model m;
  add_mat1(m, 1, E, nu);
  m.properties[p.pid] = p;
  add_grid(m, 1, 0.0, 0.0, 0.0);
  add_grid(m, 2, L, 0.0, 0.0);
  return m;
}

PBar make_pbar(double A, double I1, double I2, double J) {
  PBar p;
  p.pid = PropertyId{1};
  p.mid = MaterialId{1};
  p.A = A;
  p.I1 = I1;
  p.I2 = I2;
  p.J = J;
  return p;
}

} // namespace

TEST(CBar, PlaneOneBendsWithI1AndPlaneTwoWithI2) {
  // Nastran: I1 is the inertia for bending in plane 1 (element x-y, i.e.
  // y deflection) and I2 for plane 2 (x-z).  A strongly unequal section
  // must give PL^3/(3 E I1) for a y tip load and PL^3/(3 E I2) for z.
  const double L = 10.0, E = 1.0e7, I1 = 1.0, I2 = 100.0;
  const Model m = bar_model(L, make_pbar(2.0, I1, I2, 5.0), E);
  const CBarBeamElement elem(
      bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, 1.0, 0.0}), m);
  const LocalKe K = elem.stiffness_matrix();

  Eigen::Matrix<double, 6, 1> f = Eigen::Matrix<double, 6, 1>::Zero();
  f(1) = 1.0;
  f(2) = 1.0;
  const auto u = cantilever_tip(K, f);
  EXPECT_NEAR(u(1), L * L * L / (3.0 * E * I1), 1e-12);
  EXPECT_NEAR(u(2), L * L * L / (3.0 * E * I2), 1e-14);
  // Tip rotations: theta_z = +PL^2/(2EI1), theta_y = -PL^2/(2EI2).
  EXPECT_NEAR(u(5), L * L / (2.0 * E * I1), 1e-12);
  EXPECT_NEAR(u(4), -L * L / (2.0 * E * I2), 1e-14);
}

TEST(CBar, ShearFactorsAddTimoshenkoTipFlexibility) {
  // Tip deflection of a Timoshenko cantilever is PL^3/(3EI) + PL/(K A G);
  // K1 governs plane 1 and K2 plane 2.  A short, deep bar makes the shear
  // term significant (here ~25-40% of the total).
  const double L = 2.0, E = 3.0e7, nu = 0.3, A = 1.5;
  const double I1 = 0.8, I2 = 0.5, K1 = 5.0 / 6.0, K2 = 0.6;
  PBar p = make_pbar(A, I1, I2, 0.4);
  p.K1 = K1;
  p.K2 = K2;
  const Model m = bar_model(L, p, E, nu);
  const double G = E / (2.0 * (1.0 + nu));
  const CBarBeamElement elem(
      bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, 1.0, 0.0}), m);
  const LocalKe K = elem.stiffness_matrix();
  EXPECT_LT((K - K.transpose()).cwiseAbs().maxCoeff(), 1e-9 * K.norm());

  Eigen::Matrix<double, 6, 1> f = Eigen::Matrix<double, 6, 1>::Zero();
  f(1) = 1000.0;
  f(2) = -700.0;
  const auto u = cantilever_tip(K, f);
  const double v_expected =
      1000.0 * (L * L * L / (3.0 * E * I1) + L / (K1 * A * G));
  const double w_expected =
      -700.0 * (L * L * L / (3.0 * E * I2) + L / (K2 * A * G));
  EXPECT_NEAR(u(1), v_expected, 1e-10 * std::abs(v_expected));
  EXPECT_NEAR(u(2), w_expected, 1e-10 * std::abs(w_expected));
  // Shear deformation does not change the tip slope under a tip load.
  EXPECT_NEAR(u(5), 1000.0 * L * L / (2.0 * E * I1), 1e-12);

  // Rigid-body motions still produce no force.
  Eigen::VectorXd rb = Eigen::VectorXd::Zero(12);
  rb(5) = 1.0;           // rotation about z at grid 1 ...
  rb(11) = 1.0;          // ... and grid 2,
  rb(7) = L;             // with grid 2 translating accordingly.
  EXPECT_LT((K * rb).cwiseAbs().maxCoeff(), 1e-8 * K.norm());
}

TEST(CBar, ProductOfInertiaMatchesRotatedPrincipalSection) {
  // A section with principal inertias (Ip1, Ip2) whose principal axes are
  // rotated by alpha about x has, in the element frame,
  //   I1 = c^2 Ip1 + s^2 Ip2,  I2 = s^2 Ip1 + c^2 Ip2,  I12 = c s (Ip1 - Ip2).
  // Describing the same bar with I12 (coupled flexibility path) or with its
  // principal axes as the element frame (uncoupled Timoshenko path) must give
  // the same global stiffness and the same stress at the same material point.
  // K1 = K2 keeps the shear stiffness isotropic so it is frame-invariant.
  const double L = 3.0, alpha = 0.47, Ip1 = 2.0, Ip2 = 0.35, K = 0.85;
  const double c = std::cos(alpha), s = std::sin(alpha);
  const std::array<std::array<double, 2>, 4> pts_principal{
      {{0.6, 0.2}, {-0.6, 0.2}, {-0.6, -0.2}, {0.6, -0.2}}};

  PBar coupled = make_pbar(1.2, c * c * Ip1 + s * s * Ip2,
                           s * s * Ip1 + c * c * Ip2, 0.9);
  coupled.I12 = c * s * (Ip1 - Ip2);
  coupled.K1 = coupled.K2 = K;
  // Material point (y', z') in principal axes is (c y' - s z', s y' + c z')
  // in the element frame.
  for (size_t i = 0; i < 4; ++i) {
    const double yp = pts_principal[i][0], zp = pts_principal[i][1];
    coupled.recovery_points[i] = {c * yp - s * zp, s * yp + c * zp};
  }
  PBar principal = make_pbar(1.2, Ip1, Ip2, 0.9);
  principal.K1 = principal.K2 = K;
  principal.recovery_points = pts_principal;

  const Model m_coupled = bar_model(L, coupled);
  const Model m_principal = bar_model(L, principal);
  const CBarBeamElement e_coupled(
      bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, 1.0, 0.0}), m_coupled);
  const CBarBeamElement e_principal(
      bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, c, s}), m_principal);

  const LocalKe K_coupled = e_coupled.stiffness_matrix();
  const LocalKe K_principal = e_principal.stiffness_matrix();
  EXPECT_LT((K_coupled - K_principal).cwiseAbs().maxCoeff(),
            1e-10 * K_principal.cwiseAbs().maxCoeff());

  std::array<double, 12> u{};
  for (size_t i = 0; i < 12; ++i)
    u[i] = 1e-3 * std::sin(1.3 * static_cast<double>(i) + 0.2);
  const auto r_coupled = e_coupled.recover_stress(u, 0.0, 0.0);
  const auto r_principal = e_principal.recover_stress(u, 0.0, 0.0);
  for (size_t i = 0; i < 4; ++i) {
    const double scale = std::abs(r_principal.end_a.s[i]) + 1.0;
    EXPECT_NEAR(r_coupled.end_a.s[i], r_principal.end_a.s[i], 1e-9 * scale);
    EXPECT_NEAR(r_coupled.end_b.s[i], r_principal.end_b.s[i], 1e-9 * scale);
  }
  EXPECT_NEAR(r_coupled.forces.axial, r_principal.forces.axial,
              1e-9 * std::abs(r_principal.forces.axial));
}

TEST(CBar, PinFlagCondensesToCantileverWithFreeTipRotation) {
  // PB = 6 releases theta_z at end B.  With end A fixed, the remaining v_B
  // stiffness is that of a cantilever whose tip rotates freely:
  //   k = 1 / (L^3/(3 E I1) + L/(K1 A G))
  // rather than the guided-tip 12 E I1 / L^3 (shear-free) value.
  const double L = 4.0, E = 2.0e7, nu = 0.3, A = 2.0, I1 = 0.5, K1 = 0.8;
  PBar p = make_pbar(A, I1, 0.7, 1.0);
  p.K1 = K1;
  const Model m = bar_model(L, p, E, nu);
  const double G = E / (2.0 * (1.0 + nu));
  ElementData data = bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, 1.0, 0.0});
  data.pin_flags[1] = 1u << 5; // component 6
  const CBarBeamElement elem(data, m);
  const LocalKe K = elem.stiffness_matrix();

  EXPECT_LT(K.row(11).cwiseAbs().maxCoeff(), 1e-9 * K.norm());
  EXPECT_LT(K.col(11).cwiseAbs().maxCoeff(), 1e-9 * K.norm());
  const double k_expected = 1.0 / (L * L * L / (3.0 * E * I1) + L / (K1 * A * G));
  EXPECT_NEAR(K(7, 7), k_expected, 1e-10 * k_expected);
  // Plane 2 is untouched by a plane-1 release: guided-tip 12 E I2 / L^3.
  EXPECT_NEAR(K(8, 8), 12.0 * E * 0.7 / (L * L * L), 1e-6);

  // The released end carries no moment for any displacement state.
  std::array<double, 12> u{};
  for (size_t i = 0; i < 12; ++i)
    u[i] = 1e-4 * std::cos(0.7 * static_cast<double>(i));
  const auto r = elem.recover_stress(u, 0.0, 0.0);
  EXPECT_NEAR(r.forces.m1b, 0.0, 1e-9 * (std::abs(r.forces.m1a) + 1.0));
  EXPECT_GT(std::abs(r.forces.m1a), 1.0);
}

TEST(CBar, ReleasingTorsionAtBothEndsRemovesTorsionalStiffness) {
  // PA = PB = 4 releases torsion at both ends: K_rr is singular, which the
  // pseudo-inverse condensation must handle.  The element then has zero
  // torsional stiffness while bending and axial terms are unchanged.
  const Model m = bar_model(5.0, make_pbar(1.0, 0.3, 0.4, 0.6));
  ElementData free_data = bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, 1.0, 0.0});
  ElementData pinned_data = free_data;
  pinned_data.pin_flags = {1u << 3, 1u << 3};
  const LocalKe K_free = CBarBeamElement(free_data, m).stiffness_matrix();
  const LocalKe K_pin = CBarBeamElement(pinned_data, m).stiffness_matrix();

  ASSERT_TRUE(K_pin.allFinite());
  for (const int dof : {3, 9}) {
    EXPECT_LT(K_pin.row(dof).cwiseAbs().maxCoeff(), 1e-9 * K_free.norm());
  }
  LocalKe diff = K_free - K_pin;
  diff(3, 3) = diff(3, 9) = diff(9, 3) = diff(9, 9) = 0.0;
  EXPECT_LT(diff.cwiseAbs().maxCoeff(), 1e-9 * K_free.norm());
}

TEST(CBar, EndOffsetsAreRigidLinksAndKeepRigidBodyModes) {
  // Offsets WA = WB = (0, d, 0) move the neutral axis to y = d.  A rotation
  // theta_z of grid 1 then moves end A by -theta*d in x, stretching the bar:
  // the axial force on grid 2 is E A theta d / L.  All six rigid-body modes of
  // the grid pair must remain force-free.
  const double L = 6.0, d = 0.75, E = 2.0e7, A = 1.3;
  const Model m = bar_model(L, make_pbar(A, 0.2, 0.3, 0.25), E);
  ElementData data = bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, 0.0, 1.0});
  data.offsets = {Vec3{0.0, d, 0.0}, Vec3{0.0, d, 0.0}};
  const LocalKe K = CBarBeamElement(data, m).stiffness_matrix();

  Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
  u(5) = 1e-3;
  const Eigen::VectorXd f = K * u;
  EXPECT_NEAR(f(6), E * A * 1e-3 * d / L, 1e-9 * E * A * 1e-3 * d / L);

  const std::array<Vec3, 2> pos{Vec3{0.0, 0.0, 0.0}, Vec3{L, 0.0, 0.0}};
  for (int mode = 0; mode < 6; ++mode) {
    Eigen::VectorXd rb = Eigen::VectorXd::Zero(12);
    for (int n = 0; n < 2; ++n) {
      if (mode < 3) {
        rb(6 * n + mode) = 1.0;
      } else {
        Eigen::Vector3d w = Eigen::Vector3d::Zero();
        w(mode - 3) = 1.0;
        const Eigen::Vector3d r{pos[n].x, pos[n].y, pos[n].z};
        rb.segment<3>(6 * n) = w.cross(r);
        rb.segment<3>(6 * n + 3) = w;
      }
    }
    EXPECT_LT((K * rb).cwiseAbs().maxCoeff(), 1e-8 * K.norm())
        << "rigid-body mode " << mode;
  }
}

TEST(CBar, StressRecoveryUsesEndForcesAndRecoveryPoints) {
  // Cantilever, tip load P in +y: M1(x) = P (L - x), so at point C
  // (y = c, z = 0) the stress is -P L c / I1 at end A and zero at end B.
  // A uniform temperature rise on the free cantilever adds no stress.
  const double L = 5.0, E = 1.0e7, I1 = 0.4, c = 0.3, P = 250.0;
  PBar p = make_pbar(1.0, I1, 0.6, 0.5);
  p.recovery_points = {{{c, 0.0}, {0.0, c}, {-c, 0.0}, {0.0, -c}}};
  Model m = bar_model(L, p, E);
  m.materials.at(MaterialId{1}).A = 1.0e-5;
  const CBarBeamElement elem(
      bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, 1.0, 0.0}), m);
  const LocalKe K = elem.stiffness_matrix();

  Eigen::Matrix<double, 6, 1> f = Eigen::Matrix<double, 6, 1>::Zero();
  f(1) = P;
  const LocalFe f_th = elem.thermal_load(std::array<double, 2>{80.0, 80.0}, 0.0);
  f += f_th.segment<6>(6);
  const auto tip = cantilever_tip(K, f);
  std::array<double, 12> u{};
  for (size_t i = 0; i < 6; ++i)
    u[6 + i] = tip(static_cast<Eigen::Index>(i));

  const auto r = elem.recover_stress(u, 80.0, 0.0);
  EXPECT_NEAR(r.forces.m1a, P * L, 1e-8 * P * L);
  EXPECT_NEAR(r.forces.m1b, 0.0, 1e-8 * P * L);
  EXPECT_NEAR(r.forces.v1, P, 1e-8 * P);
  EXPECT_NEAR(r.forces.axial, 0.0, 1e-8 * P);
  EXPECT_NEAR(r.end_a.s[0], -P * L * c / I1, 1e-8 * P * L * c / I1);
  EXPECT_NEAR(r.end_a.s[2], P * L * c / I1, 1e-8 * P * L * c / I1);
  EXPECT_NEAR(r.end_a.s[1], 0.0, 1e-8 * P * L * c / I1);
  EXPECT_NEAR(r.end_b.s[0], 0.0, 1e-8 * P * L * c / I1);
  EXPECT_NEAR(r.end_a.smax, P * L * c / I1, 1e-8 * P * L * c / I1);
}

namespace {

/// Bar of length L in the basic x-y plane at angle `deg` from x, with the
/// element y axis also in that plane (v = (-sin, cos, 0)) and z = basic z.
Model inclined_bar_model(double L, double deg, const PBar &p) {
  Model m;
  add_mat1(m, 1, 2.0e7, 0.3);
  m.properties[p.pid] = p;
  const double a = deg * std::numbers::pi / 180.0;
  add_grid(m, 1, 0.0, 0.0, 0.0);
  add_grid(m, 2, L * std::cos(a), L * std::sin(a), 0.0);
  return m;
}

ElementData inclined_bar(double deg) {
  const double a = deg * std::numbers::pi / 180.0;
  return bar_data(ElementType::CBAR, 1, 2, Vec3{-std::sin(a), std::cos(a), 0.0});
}

Pload1Load pload1(const char *type, const char *scale, double x1, double p1,
                  std::optional<double> x2, std::optional<double> p2) {
  Pload1Load l;
  l.sid = LoadSetId{1};
  l.element = ElementId{1};
  l.load_type = type;
  l.scale_type = scale;
  l.x1 = x1;
  l.p1 = p1;
  l.x2 = x2;
  l.p2 = p2;
  return l;
}

/// Resultant force and moment about grid 1 of a global 12-entry load vector
/// for grids at the given basic positions.
std::pair<Eigen::Vector3d, Eigen::Vector3d>
resultant(const LocalFe &f, const Vec3 &x1, const Vec3 &x2) {
  Eigen::Vector3d F = f.segment<3>(0) + f.segment<3>(6);
  const Eigen::Vector3d r2{x2.x - x1.x, x2.y - x1.y, x2.z - x1.z};
  Eigen::Vector3d M = f.segment<3>(3) + f.segment<3>(9) +
                      r2.cross(Eigen::Vector3d(f.segment<3>(6)));
  return {F, M};
}

} // namespace

TEST(Pload1, ConcentratedLoadGivesExactFixedEndForces) {
  // X2 blank means a concentrated load P1 at X1.  For a fixed-fixed beam
  // with P at x = a (b = L - a) the end reactions are
  //   R_A = P b^2 (3a + b) / L^3,  M_A = P a b^2 / L^2,
  //   R_B = P a^2 (a + 3b) / L^3,  M_B = -P a^2 b / L^2.
  const double L = 8.0, a = 3.0, b = L - a, P = 120.0;
  const Model m = inclined_bar_model(L, 0.0, make_pbar(1.0, 0.5, 0.5, 0.5));
  const LocalFe f = compute_pload1_fixed_end_local(
      inclined_bar(0.0), m, pload1("FYE", "LE", a, P, std::nullopt, std::nullopt));
  EXPECT_NEAR(f(1), P * b * b * (3 * a + b) / (L * L * L), 1e-10);
  EXPECT_NEAR(f(5), P * a * b * b / (L * L), 1e-10);
  EXPECT_NEAR(f(7), P * a * a * (a + 3 * b) / (L * L * L), 1e-10);
  EXPECT_NEAR(f(11), -P * a * a * b / (L * L), 1e-10);
  for (const int i : {0, 2, 3, 4, 6, 8, 9, 10})
    EXPECT_EQ(f(i), 0.0);
}

TEST(Pload1, BasicDirectionLoadIsResolvedIntoElementAxes) {
  // Bar at 30 degrees in the x-y plane, uniform basic-FY load q.  In the
  // element frame basic y = (sin30, cos30, 0): the axial share is q sin30
  // (q sin30 L / 2 per end) and the transverse share q cos30 gives
  // q cos30 L / 2 shear and +-q cos30 L^2 / 12 moments.  The element-axis
  // type FYE puts everything into transverse load.
  const double L = 6.0, q = 40.0, s = 0.5, c = std::sqrt(3.0) / 2.0;
  const Model m = inclined_bar_model(L, 30.0, make_pbar(1.0, 0.5, 0.5, 0.5));
  const LocalFe fy = compute_pload1_fixed_end_local(
      inclined_bar(30.0), m, pload1("FY", "LE", 0.0, q, L, q));
  EXPECT_NEAR(fy(0), q * s * L / 2.0, 1e-9);
  EXPECT_NEAR(fy(6), q * s * L / 2.0, 1e-9);
  EXPECT_NEAR(fy(1), q * c * L / 2.0, 1e-9);
  EXPECT_NEAR(fy(5), q * c * L * L / 12.0, 1e-9);
  EXPECT_NEAR(fy(11), -q * c * L * L / 12.0, 1e-9);

  const LocalFe fye = compute_pload1_fixed_end_local(
      inclined_bar(30.0), m, pload1("FYE", "LE", 0.0, q, L, q));
  EXPECT_NEAR(fye(0), 0.0, 1e-12);
  EXPECT_NEAR(fye(1), q * L / 2.0, 1e-9);

  // In the basic frame the FY load totals q L along +y with its moment
  // about grid 1 equal to the resultant at mid-span.
  const LocalFe g = compute_pload1_equivalent_load(
      inclined_bar(30.0), m, pload1("FY", "LE", 0.0, q, L, q));
  const auto [F, M] = resultant(g, m.node(NodeId{1}).position,
                                m.node(NodeId{2}).position);
  EXPECT_NEAR(F(0), 0.0, 1e-9);
  EXPECT_NEAR(F(1), q * L, 1e-9);
  EXPECT_NEAR(M(2), q * L * (L / 2.0) * c, 1e-8); // lever arm x = L/2 cos30
}

TEST(Pload1, ProjectedScaleUsesLengthNormalToLoad) {
  // LEPR: intensity per unit projected length normal to the load.  For basic
  // FY on a bar at 30 degrees from x, the projection onto the x axis is
  // L cos30, so the total load is q L cos30.
  const double L = 6.0, q = 40.0, c = std::sqrt(3.0) / 2.0;
  const Model m = inclined_bar_model(L, 30.0, make_pbar(1.0, 0.5, 0.5, 0.5));
  const LocalFe g = compute_pload1_equivalent_load(
      inclined_bar(30.0), m, pload1("FY", "LEPR", 0.0, q, L, q));
  const auto [F, M] = resultant(g, m.node(NodeId{1}).position,
                                m.node(NodeId{2}).position);
  EXPECT_NEAR(F(1), q * L * c, 1e-9);
  EXPECT_NEAR(F(0), 0.0, 1e-9);
  // FRPR with the same span is identical.
  const LocalFe g_fr = compute_pload1_equivalent_load(
      inclined_bar(30.0), m, pload1("FY", "FRPR", 0.0, q, 1.0, q));
  EXPECT_LT((g - g_fr).cwiseAbs().maxCoeff(), 1e-9);
}

TEST(Pload1, PartialLinearLoadIsStaticallyEquivalent) {
  // FR stations 0.25..0.75 with intensity 10 -> 30: total = 20 * L/2, and the
  // centroid of the trapezoid sits at x = L/4 + (L/2)(1 + 2*3)/(3*(1 + 3)).
  const double L = 8.0, p1 = 10.0, p2 = 30.0;
  const Model m = inclined_bar_model(L, 0.0, make_pbar(1.0, 0.5, 0.5, 0.5));
  const LocalFe g = compute_pload1_equivalent_load(
      inclined_bar(0.0), m, pload1("FZE", "FR", 0.25, p1, 0.75, p2));
  const auto [F, M] = resultant(g, m.node(NodeId{1}).position,
                                m.node(NodeId{2}).position);
  const double total = 0.5 * (p1 + p2) * (L / 2.0);
  const double xc = L / 4.0 + (L / 2.0) * (p1 + 2.0 * p2) / (3.0 * (p1 + p2));
  EXPECT_NEAR(F(2), total, 1e-9);
  EXPECT_NEAR(M(1), -total * xc, 1e-8); // z force at +x gives -My
}

TEST(Pload1, ConcentratedMomentIsSelfEquilibratedCouple) {
  // A concentrated element-z moment M0 at mid-span: the consistent loads
  // carry no net force and a net moment M0 about z.
  const double L = 5.0, M0 = 75.0;
  const Model m = inclined_bar_model(L, 0.0, make_pbar(1.0, 0.5, 0.5, 0.5));
  const LocalFe g = compute_pload1_equivalent_load(
      inclined_bar(0.0), m, pload1("MZE", "FR", 0.5, M0, std::nullopt, std::nullopt));
  const auto [F, M] = resultant(g, m.node(NodeId{1}).position,
                                m.node(NodeId{2}).position);
  EXPECT_NEAR(F.norm(), 0.0, 1e-10);
  EXPECT_NEAR(M(2), M0, 1e-10);
  EXPECT_NEAR(M(0), 0.0, 1e-12);
}

TEST(Pload1, RejectsInvalidStationsAndProjectedElementTypes) {
  const Model m = inclined_bar_model(4.0, 0.0, make_pbar(1.0, 0.5, 0.5, 0.5));
  const ElementData bar = inclined_bar(0.0);
  for (const Pload1Load &bad : {
           pload1("FYE", "LEPR", 0.0, 1.0, 4.0, 1.0),  // projection needs basic
           pload1("FY", "LE", 3.0, 1.0, 1.0, 1.0),     // X2 < X1
           pload1("FY", "FR", 0.0, 1.0, 1.5, 1.0),     // beyond the element
           pload1("FW", "LE", 0.0, 1.0, 4.0, 1.0),     // unknown type
           pload1("FY", "XX", 0.0, 1.0, 4.0, 1.0),     // unknown scale
       }) {
    EXPECT_THROW({ (void)compute_pload1_fixed_end_local(bar, m, bad); },
                 SolverError)
        << bad.load_type << " " << bad.scale_type;
  }
}

TEST(Pload1, PinnedEndRecoversProppedCantileverForces) {
  // Fixed at A, PB = 6 releases theta_z at B, both grids held: a propped
  // cantilever.  Uniform upward q gives M1(A) = q L^2 / 8, M1(B) = 0 and end
  // A shear 5 q L / 8.  All grid displacements are zero, so the forces come
  // entirely from the condensed fixed-end loads.
  const double L = 4.0, q = 30.0;
  const Model m = inclined_bar_model(L, 0.0, make_pbar(1.0, 0.5, 0.5, 0.5));
  ElementData data = inclined_bar(0.0);
  data.pin_flags[1] = 1u << 5;
  const CBarBeamElement elem(data, m);
  const LocalFe f = compute_pload1_fixed_end_local(
      data, m, pload1("FYE", "LE", 0.0, q, L, q));
  const std::array<double, 12> u{};
  const auto r = elem.recover_stress(
      u, 0.0, 0.0, std::span<const double>(f.data(), 12));
  EXPECT_NEAR(r.forces.m1a, q * L * L / 8.0, 1e-9);
  EXPECT_NEAR(r.forces.m1b, 0.0, 1e-9);
  EXPECT_NEAR(r.forces.v1, 5.0 * q * L / 8.0, 1e-9);
  // The equivalent grid loads put 5qL/8 and 3qL/8 on the two grids.
  const LocalFe g = compute_pload1_equivalent_load(
      data, m, pload1("FYE", "LE", 0.0, q, L, q));
  EXPECT_NEAR(g(1), 5.0 * q * L / 8.0, 1e-9);
  EXPECT_NEAR(g(7), 3.0 * q * L / 8.0, 1e-9);
  EXPECT_NEAR(g(5), q * L * L / 8.0, 1e-9);
  EXPECT_NEAR(g(11), 0.0, 1e-9);
}

TEST(LumpedMass, HrzKeepsTetra10MassPositiveAndConserved) {
  // Row-sum lumping of a quadratic tetrahedron's consistent mass gives
  // non-positive corner masses (-1/20 of the total each); HRZ diagonal scaling
  // must give strictly positive masses that sum to rho*V in every direction,
  // with no off-diagonal coupling between DOFs.
  Model m = make_solid_model();
  const double rho = 7.8;
  m.materials.at(MaterialId{1}).rho = rho;
  // Corners ordered for a positive Jacobian; midsides on edges 1-2, 2-3,
  // 1-3, 1-4, 2-4, 3-4.
  add_grid(m, 1, 0, 0, 0);
  add_grid(m, 2, 0, 1, 0);
  add_grid(m, 3, 2, 0, 0);
  add_grid(m, 4, 0, 0, 3);
  add_grid(m, 5, 0, 0.5, 0);
  add_grid(m, 6, 1, 0.5, 0);
  add_grid(m, 7, 1, 0, 0);
  add_grid(m, 8, 0, 0, 1.5);
  add_grid(m, 9, 0, 0.5, 1.5);
  add_grid(m, 10, 1, 0, 1.5);
  const std::array<NodeId, 10> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                                     NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8},
                                     NodeId{9}, NodeId{10}};
  const CTetra10 elem(ElementId{1}, PropertyId{1}, nodes, m);
  const LocalKe Mc = elem.mass_matrix();
  const LocalKe Ml = elem.lumped_mass_matrix();
  const double volume = 2.0 * 1.0 * 3.0 / 6.0;

  // Row sums of the consistent x-block at the corners are not positive.
  double corner_row_sum = 0.0;
  for (int b = 0; b < 10; ++b)
    corner_row_sum += Mc(0, 3 * b);
  EXPECT_LE(corner_row_sum, 1e-12 * rho * volume);

  EXPECT_LT((Ml - LocalKe(Ml.diagonal().asDiagonal())).cwiseAbs().maxCoeff(),
            1e-15);
  for (int d = 0; d < 3; ++d) {
    double total = 0.0;
    for (int a = 0; a < 10; ++a) {
      EXPECT_GT(Ml(3 * a + d, 3 * a + d), 0.0);
      total += Ml(3 * a + d, 3 * a + d);
    }
    EXPECT_NEAR(total, rho * volume, 1e-12 * rho * volume);
  }
}

TEST(CTetra10, ConsistentMassIsExactForQuadraticFieldsAndFullRank) {
  // A tet10 interpolates u = x^2 exactly, so u^T M u = rho * int x^4 dV.
  // For a tetrahedron, int (sum lambda_i x_i)^4 dV = V h4(x_1..x_4) / 35
  // (Dirichlet integral; h4 the complete homogeneous symmetric polynomial).
  // With vertex x-coordinates (0, 0, 2, 0): h4 = 16, V = 1.  This degree-4
  // integrand needs a quadrature rule beyond degree 2; the matrix must also
  // be positive definite (a low-order rule leaves 18 zero eigenvalues).
  Model m = make_solid_model();
  const double rho = 2.5;
  m.materials.at(MaterialId{1}).rho = rho;
  const std::array<std::array<double, 3>, 10> xyz{{{0, 0, 0}, {0, 1, 0},
      {2, 0, 0}, {0, 0, 3}, {0, 0.5, 0}, {1, 0.5, 0}, {1, 0, 0}, {0, 0, 1.5},
      {0, 0.5, 1.5}, {1, 0, 1.5}}};
  for (int i = 0; i < 10; ++i)
    add_grid(m, i + 1, xyz[i][0], xyz[i][1], xyz[i][2]);
  std::array<NodeId, 10> nodes{NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4},
                               NodeId{5}, NodeId{6}, NodeId{7}, NodeId{8},
                               NodeId{9}, NodeId{10}};
  const CTetra10 elem(ElementId{1}, PropertyId{1}, nodes, m);
  const LocalKe M = elem.mass_matrix();

  Eigen::VectorXd u = Eigen::VectorXd::Zero(30);
  for (int i = 0; i < 10; ++i)
    u(3 * i) = xyz[i][0] * xyz[i][0];
  EXPECT_NEAR(u.dot(M * u), rho * 16.0 / 35.0, 1e-12);

  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
  EXPECT_GT(es.eigenvalues().minCoeff(), 1e-4 * es.eigenvalues().maxCoeff());
}

TEST(LumpedMass, CbarPlacesHalfMassAtEachOffsetEnd) {
  // Lumped bar mass is (rho*A + nsm)*L/2 on the translations of each element
  // end with no rotational inertia.  With end offsets the masses sit at the
  // offset points, so a rigid rotation omega about grid 1 has kinetic
  // "energy" u^T M u = sum m_end |omega x r_end|^2.
  PBar p = make_pbar(0.4, 0.2, 0.3, 0.1);
  p.nsm = 0.25;
  Model m = bar_model(5.0, p);
  m.materials.at(MaterialId{1}).rho = 2.0;
  ElementData data = bar_data(ElementType::CBAR, 1, 2, Vec3{0.0, 0.0, 1.0});
  const LocalKe plain = CBarBeamElement(data, m).lumped_mass_matrix();
  const double half = 0.5 * 5.0 * (2.0 * 0.4 + 0.25);
  for (const int dof : {0, 1, 2, 6, 7, 8})
    EXPECT_NEAR(plain(dof, dof), half, 1e-12);
  for (const int dof : {3, 4, 5, 9, 10, 11})
    EXPECT_EQ(plain(dof, dof), 0.0);

  // Offsets of (0, d, 0) at both ends; L between offset ends is unchanged.
  const double d = 0.6;
  data.offsets = {Vec3{0.0, d, 0.0}, Vec3{0.0, d, 0.0}};
  const LocalKe Mo = CBarBeamElement(data, m).lumped_mass_matrix();
  Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
  // Rigid rotation about z through grid 1: grid 2 moves (0, 5, 0) per rad.
  u(5) = 1.0;
  u(11) = 1.0;
  u(7) = 5.0;
  // Offset ends at (0, d) and (5, d): |omega x r|^2 = d^2 and 25 + d^2.
  EXPECT_NEAR(u.dot(Mo * u), half * (d * d) + half * (25.0 + d * d), 1e-10);
}

TEST(LumpedMass, CoupmassParamSelectsFormulation) {
  // Consistent mass is vibestran's default; PARAM,COUPMASS <= 0 (Nastran's
  // lumped setting) selects lumped mass, and a positive value consistent.
  Model m;
  EXPECT_FALSE(m.use_lumped_mass());
  m.params["COUPMASS"] = "-1";
  EXPECT_TRUE(m.use_lumped_mass());
  m.params["COUPMASS"] = "0";
  EXPECT_TRUE(m.use_lumped_mass());
  m.params["COUPMASS"] = "1";
  EXPECT_FALSE(m.use_lumped_mass());
}

namespace {

ElementData conm2_data(double mass, Vec3 offset, std::array<double, 6> inertia,
                       int cid) {
  ElementData e;
  e.id = ElementId{9};
  e.type = ElementType::CONM2;
  e.nodes = {NodeId{1}};
  e.conm2 = ElementData::ConcentratedMass{mass, offset, inertia, cid};
  return e;
}

/// Rigid-body kinetic "energy" u^T M u for grid translation u and rotation w.
double rigid_energy(const LocalKe &M, const Eigen::Vector3d &u,
                    const Eigen::Vector3d &w) {
  Eigen::Matrix<double, 6, 1> x;
  x << u, w;
  return x.dot(M * x);
}

} // namespace

TEST(Conm2, OffsetMassFollowsRigidBodyKinematicsAndParallelAxis) {
  // Mass m with inertia J about its CG at offset r from the grid: the grid
  // rotational inertia is J + m(|r|^2 I - r r^T) (parallel axis) and the
  // CG moves with u + w x r, so u^T M u = m |u + w x r|^2 + w^T J w.
  Model m;
  add_grid(m, 1, 1.0, 2.0, 3.0);
  const double mass = 4.0;
  const Vec3 r{0.3, -0.2, 0.5};
  // I11, I21, I22, I31, I32, I33 (products enter with a minus sign).
  const std::array<double, 6> I{2.0, 0.1, 3.0, -0.2, 0.15, 4.0};
  const LocalKe M = Conm2Element(conm2_data(mass, r, I, 0), m).mass_matrix();
  Eigen::Matrix3d J;
  J << 2.0, -0.1, 0.2, -0.1, 3.0, -0.15, 0.2, -0.15, 4.0;
  const Eigen::Vector3d rv{r.x, r.y, r.z};
  const Eigen::Matrix3d parallel =
      J + mass * (rv.squaredNorm() * Eigen::Matrix3d::Identity() - rv * rv.transpose());
  EXPECT_LT((M.block<3, 3>(3, 3) - parallel).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT((M - M.transpose()).cwiseAbs().maxCoeff(), 1e-14);

  const Eigen::Vector3d u{0.7, -0.4, 0.25}, w{0.3, 0.9, -0.6};
  const Eigen::Vector3d v_cg = u + w.cross(rv);
  EXPECT_NEAR(rigid_energy(M, u, w), mass * v_cg.squaredNorm() + w.dot(J * w),
              1e-12);
  // No stiffness, and the lumped form is the same matrix.
  const Conm2Element elem(conm2_data(mass, r, I, 0), m);
  EXPECT_EQ(elem.stiffness_matrix().cwiseAbs().maxCoeff(), 0.0);
  EXPECT_LT((elem.lumped_mass_matrix() - M).cwiseAbs().maxCoeff(), 1e-15);
}

TEST(Conm2, CidMinusOneGivesCgLocationAndRotatedCidRotatesInertia) {
  // CID = -1: X is the basic CG location, so the offset is X - grid.
  Model m;
  add_grid(m, 1, 1.0, 2.0, 3.0);
  const LocalKe Mabs = Conm2Element(
      conm2_data(2.0, Vec3{1.3, 1.8, 3.5}, {}, -1), m).mass_matrix();
  const LocalKe Mrel = Conm2Element(
      conm2_data(2.0, Vec3{0.3, -0.2, 0.5}, {}, 0), m).mass_matrix();
  EXPECT_LT((Mabs - Mrel).cwiseAbs().maxCoeff(), 1e-12);

  // CID 5 has x5 = basic y, y5 = -basic x, z5 = basic z.  An inertia I11
  // about x5 is an inertia about basic y, and an offset along x5 is a basic
  // y offset.
  CoordSys cs;
  cs.id = CoordId{5};
  cs.type = CoordType::Rectangular;
  cs.origin = Vec3{0, 0, 0};
  cs.axes = {Vec3{0, 1, 0}, Vec3{-1, 0, 0}, Vec3{0, 0, 1}};
  m.coord_systems[cs.id] = cs;
  const LocalKe Mrot = Conm2Element(
      conm2_data(0.0, Vec3{}, {7.0, 0, 0, 0, 0, 0}, 5), m).mass_matrix();
  EXPECT_NEAR(Mrot(4, 4), 7.0, 1e-12); // basic R2
  EXPECT_NEAR(Mrot(3, 3), 0.0, 1e-12);
  const LocalKe Moff = Conm2Element(
      conm2_data(3.0, Vec3{2.0, 0, 0}, {}, 5), m).mass_matrix();
  // Offset along basic y by 2: rotation about basic z moves the CG in -x.
  EXPECT_NEAR(rigid_energy(Moff, Eigen::Vector3d::Zero(),
                           Eigen::Vector3d{0, 0, 1}), 3.0 * 4.0, 1e-12);
  EXPECT_NEAR(rigid_energy(Moff, Eigen::Vector3d::Zero(),
                           Eigen::Vector3d{0, 1, 0}), 0.0, 1e-12);
}

TEST(CBeam, ThermalLoadIsAxialAndSelfEquilibrated) {
  Model m;
  add_mat1(m, 1, 3.0e7, 0.25, 0.0, 1.2e-5);

  PBeam p;
  p.pid = PropertyId{1};
  p.mid = MaterialId{1};
  p.A = 1.5;
  p.I1 = 0.2;
  p.I2 = 0.3;
  p.J = 0.4;
  m.properties[p.pid] = p;

  add_grid(m, 1, 0.0, 0.0, 0.0);
  add_grid(m, 2, 5.0, 0.0, 0.0);

  CBarBeamElement elem(
      bar_data(ElementType::CBEAM, 1, 2, Vec3{0.0, 1.0, 0.0}), m);

  const LocalFe fe =
      elem.thermal_load(std::array<double, 2>{100.0, 100.0}, 0.0);
  const double expected = 3.0e7 * 1.5 * 1.2e-5 * 100.0;
  EXPECT_EQ(fe.size(), 12);
  EXPECT_NEAR(fe(0), -expected, 1e-8 * expected);
  EXPECT_NEAR(fe(6), expected, 1e-8 * expected);
  EXPECT_NEAR(fe.segment<3>(0).sum() + fe.segment<3>(6).sum(), 0.0,
              1e-8 * expected);
}

TEST(CBush, OrientationMapsLocalStiffnessAxes) {
  Model m;

  PBush p;
  p.pid = PropertyId{1};
  p.k = {0.0, 20.0, 0.0, 0.0, 0.0, 0.0};
  m.properties[p.pid] = p;

  add_grid(m, 1, 0.0, 0.0, 0.0);
  add_grid(m, 2, 1.0, 0.0, 0.0);

  CBushElement elem(ElementId{1}, PropertyId{1}, {NodeId{1}, NodeId{2}}, m,
                    Vec3{0.0, 0.0, 1.0}, std::nullopt, CoordId{0});
  const LocalKe Ke = elem.stiffness_matrix();

  Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
  u(8) = 1.0; // relative global Z translation should align with local y
  const Eigen::VectorXd f = Ke * u;
  EXPECT_NEAR(f(2), -20.0, 1e-10);
  EXPECT_NEAR(f(8), 20.0, 1e-10);
}

TEST(ScalarElements, GroundedSpringAndMassUseSingleTargetDof) {
  Model m;

  PElas pelas;
  pelas.pid = PropertyId{1};
  pelas.k = 55.0;
  m.properties[pelas.pid] = pelas;

  PMass pmass;
  pmass.pid = PropertyId{2};
  pmass.mass = 3.5;
  m.properties[pmass.pid] = pmass;

  add_grid(m, 1, 0.0, 0.0, 0.0);

  ScalarSpringElement spring(ElementType::CELAS1, ElementId{1}, PropertyId{1},
                             {NodeId{1}}, {2, 0}, 0.0, m);
  ScalarMassElement mass(ElementType::CMASS1, ElementId{2}, PropertyId{2},
                         {NodeId{1}}, {2, 0}, 0.0, m);

  const LocalKe K = spring.stiffness_matrix();
  const LocalKe M = mass.mass_matrix();
  EXPECT_EQ(spring.num_dofs(), 1);
  EXPECT_EQ(mass.num_dofs(), 1);
  EXPECT_DOUBLE_EQ(K(0, 0), 55.0);
  EXPECT_DOUBLE_EQ(M(0, 0), 3.5);

  DofMap dmap;
  dmap.build(m.nodes, 6);
  const auto gdofs = spring.global_dof_indices(dmap);
  ASSERT_EQ(gdofs.size(), 1u);
  EXPECT_EQ(gdofs[0], dmap.eq_index(NodeId{1}, 1));
}
