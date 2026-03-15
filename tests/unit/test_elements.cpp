// tests/unit/test_elements.cpp
// Element unit tests: verify Ke symmetry, positive semi-definiteness,
// rigid body modes, and basic patch test compliance.
// These tests do NOT require a full solve — they operate on element math directly.

#include <gtest/gtest.h>
#include "elements/cquad4.hpp"
#include "elements/ctria3.hpp"
#include "elements/solid_elements.hpp"
#include "core/model.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>

using namespace nastran;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Model make_shell_model(double E=2.0e7, double nu=0.3, double t=0.1) {
    Model m;
    // Steel-like material
    Mat1 mat;
    mat.id = MaterialId{1}; mat.E=E; mat.nu=nu;
    mat.G = E/(2*(1+nu)); mat.A=0;
    m.materials[mat.id] = mat;

    PShell ps;
    ps.pid=PropertyId{1}; ps.mid1=MaterialId{1}; ps.t=t; ps.tst=0.833333;
    m.properties[ps.pid] = ps;

    return m;
}

static Model make_solid_model(double E=2.0e7, double nu=0.3) {
    Model m;
    Mat1 mat;
    mat.id=MaterialId{1}; mat.E=E; mat.nu=nu;
    mat.G=E/(2*(1+nu)); mat.A=0;
    m.materials[mat.id] = mat;

    PSolid ps;
    ps.pid=PropertyId{1}; ps.mid=MaterialId{1};
    m.properties[ps.pid] = ps;

    return m;
}

static void add_grid(Model& m, int id, double x, double y, double z=0) {
    GridPoint g;
    g.id = NodeId{id}; g.position = Vec3{x,y,z};
    m.nodes[g.id] = g;
}

// ── CQUAD4 tests ──────────────────────────────────────────────────────────────

TEST(CQuad4, StiffnessIsSymmetric) {
    Model m = make_shell_model();
    add_grid(m,1,0,0); add_grid(m,2,1,0); add_grid(m,3,1,1); add_grid(m,4,0,1);

    std::array<NodeId,4> nodes{NodeId{1},NodeId{2},NodeId{3},NodeId{4}};
    CQuad4 elem(ElementId{1}, PropertyId{1}, nodes, m);

    LocalKe Ke = elem.stiffness_matrix();
    EXPECT_EQ(Ke.rows(), 24);
    EXPECT_EQ(Ke.cols(), 24);

    double max_asymmetry = (Ke - Ke.transpose()).cwiseAbs().maxCoeff();
    EXPECT_LT(max_asymmetry, 1e-10 * Ke.cwiseAbs().maxCoeff())
        << "Stiffness matrix is not symmetric, max asymmetry = " << max_asymmetry;
}

TEST(CQuad4, StiffnessIsPositiveSemiDefinite) {
    // A correctly formulated element Ke must be PSD.
    // It has 6 rigid body modes (3 translations + 3 rotations) → 6 zero eigenvalues
    // For a plate element: 3 in-plane RBMs + 3 out-of-plane (1 translation + 2 rotations)
    Model m = make_shell_model();
    add_grid(m,1,0,0); add_grid(m,2,1,0); add_grid(m,3,1,1); add_grid(m,4,0,1);

    std::array<NodeId,4> nodes{NodeId{1},NodeId{2},NodeId{3},NodeId{4}};
    CQuad4 elem(ElementId{1}, PropertyId{1}, nodes, m);
    LocalKe Ke = elem.stiffness_matrix();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Ke);
    Eigen::VectorXd eigenvalues = eig.eigenvalues();

    // All eigenvalues >= 0 (allowing small numerical noise)
    double min_ev = eigenvalues.minCoeff();
    double max_ev = eigenvalues.maxCoeff();
    EXPECT_GE(min_ev, -1e-6 * max_ev)
        << "Negative eigenvalue = " << min_ev << " (indicates non-PSD stiffness)";
}

TEST(CQuad4, RigidBodyTranslationProducesZeroForce) {
    // If all nodes displace by the same amount, internal forces must be zero.
    Model m = make_shell_model();
    add_grid(m,1,0,0); add_grid(m,2,2,0); add_grid(m,3,2,2); add_grid(m,4,0,2);

    std::array<NodeId,4> nodes{NodeId{1},NodeId{2},NodeId{3},NodeId{4}};
    CQuad4 elem(ElementId{1}, PropertyId{1}, nodes, m);
    LocalKe Ke = elem.stiffness_matrix();

    // Rigid body translation in x: all u_i = 1, all others = 0
    Eigen::VectorXd u_rbm = Eigen::VectorXd::Zero(24);
    for (int i = 0; i < 4; ++i) u_rbm(6*i) = 1.0;

    Eigen::VectorXd f = Ke * u_rbm;
    EXPECT_LT(f.norm(), 1e-8 * Ke.norm()) << "Rigid body translation produces non-zero forces";
}

TEST(CQuad4, ThermalLoadSymmetricHeating) {
    // If all nodes have the same temperature, thermal load should produce
    // uniform membrane stress. With no BCs, load should be non-zero but symmetric.
    Model m = make_shell_model(2.0e7, 0.3, 0.1);
    // Add thermal expansion
    m.materials.at(MaterialId{1}).A = 1.2e-5;

    add_grid(m,1,0,0); add_grid(m,2,1,0); add_grid(m,3,1,1); add_grid(m,4,0,1);
    std::array<NodeId,4> nodes{NodeId{1},NodeId{2},NodeId{3},NodeId{4}};
    CQuad4 elem(ElementId{1}, PropertyId{1}, nodes, m);

    std::array<double,4> T{100.0, 100.0, 100.0, 100.0};
    LocalFe fe = elem.thermal_load(T, 0.0);

    EXPECT_EQ(fe.size(), 24);
    // For uniform heating of a square element, by symmetry the
    // x-forces on nodes {1,4} should equal those on {2,3} but opposite sign
    // (net zero for the element in equilibrium)
    double sum_fx = 0.0;
    for (int i = 0; i < 4; ++i) sum_fx += fe(6*i); // sum of x-forces
    EXPECT_NEAR(sum_fx, 0.0, 1e-6 * fe.norm())
        << "Sum of x-forces from uniform thermal should be zero";
}

// ── CTRIA3 tests ──────────────────────────────────────────────────────────────

TEST(CTria3, StiffnessIsSymmetric) {
    Model m = make_shell_model();
    add_grid(m,1,0,0); add_grid(m,2,1,0); add_grid(m,3,0,1);

    std::array<NodeId,3> nodes{NodeId{1},NodeId{2},NodeId{3}};
    CTria3 elem(ElementId{1}, PropertyId{1}, nodes, m);
    LocalKe Ke = elem.stiffness_matrix();

    EXPECT_EQ(Ke.rows(), 18);
    double max_asym = (Ke - Ke.transpose()).cwiseAbs().maxCoeff();
    EXPECT_LT(max_asym, 1e-10 * Ke.cwiseAbs().maxCoeff());
}

TEST(CTria3, StiffnessIsPositiveSemiDefinite) {
    Model m = make_shell_model();
    add_grid(m,1,0,0); add_grid(m,2,1,0); add_grid(m,3,0,1);

    std::array<NodeId,3> nodes{NodeId{1},NodeId{2},NodeId{3}};
    CTria3 elem(ElementId{1}, PropertyId{1}, nodes, m);
    LocalKe Ke = elem.stiffness_matrix();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Ke);
    double min_ev = eig.eigenvalues().minCoeff();
    double max_ev = eig.eigenvalues().maxCoeff();
    EXPECT_GE(min_ev, -1e-6 * max_ev);
}

TEST(CTria3, ZeroAreaThrows) {
    Model m = make_shell_model();
    // Collinear nodes → zero area
    add_grid(m,1,0,0); add_grid(m,2,1,0); add_grid(m,3,2,0);

    std::array<NodeId,3> nodes{NodeId{1},NodeId{2},NodeId{3}};
    CTria3 elem(ElementId{1}, PropertyId{1}, nodes, m);
    EXPECT_THROW(elem.stiffness_matrix(), SolverError);
}

// ── CTETRA4 tests ─────────────────────────────────────────────────────────────

TEST(CTetra4, StiffnessIsSymmetric) {
    Model m = make_solid_model();
    add_grid(m,1,0,0,0); add_grid(m,2,1,0,0);
    add_grid(m,3,0,1,0); add_grid(m,4,0,0,1);

    std::array<NodeId,4> nodes{NodeId{1},NodeId{2},NodeId{3},NodeId{4}};
    CTetra4 elem(ElementId{1}, PropertyId{1}, nodes, m);
    LocalKe Ke = elem.stiffness_matrix();

    EXPECT_EQ(Ke.rows(), 12);
    double max_asym = (Ke - Ke.transpose()).cwiseAbs().maxCoeff();
    EXPECT_LT(max_asym, 1e-10 * Ke.cwiseAbs().maxCoeff());
}

TEST(CTetra4, StiffnessIsPositiveSemiDefinite) {
    Model m = make_solid_model();
    add_grid(m,1,0,0,0); add_grid(m,2,1,0,0);
    add_grid(m,3,0,1,0); add_grid(m,4,0,0,1);

    std::array<NodeId,4> nodes{NodeId{1},NodeId{2},NodeId{3},NodeId{4}};
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
    add_grid(m,1,0,0,0); add_grid(m,2,1,0,0);
    add_grid(m,3,0,1,0); add_grid(m,4,0,0,1);

    std::array<NodeId,4> nodes{NodeId{1},NodeId{2},NodeId{3},NodeId{4}};
    CTetra4 elem(ElementId{1}, PropertyId{1}, nodes, m);
    LocalKe Ke = elem.stiffness_matrix();

    // Rigid body x-translation: all u_i = 1
    Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
    for (int i = 0; i < 4; ++i) u(3*i) = 1.0;

    Eigen::VectorXd f = Ke * u;
    EXPECT_LT(f.norm(), 1e-8 * Ke.norm());
}

// ── CHEXA8 tests ──────────────────────────────────────────────────────────────

TEST(CHexa8, StiffnessIsSymmetric) {
    Model m = make_solid_model();
    // Unit cube
    add_grid(m,1,0,0,0); add_grid(m,2,1,0,0); add_grid(m,3,1,1,0); add_grid(m,4,0,1,0);
    add_grid(m,5,0,0,1); add_grid(m,6,1,0,1); add_grid(m,7,1,1,1); add_grid(m,8,0,1,1);

    std::array<NodeId,8> nodes{NodeId{1},NodeId{2},NodeId{3},NodeId{4},
                                NodeId{5},NodeId{6},NodeId{7},NodeId{8}};
    CHexa8 elem(ElementId{1}, PropertyId{1}, nodes, m);
    LocalKe Ke = elem.stiffness_matrix();

    EXPECT_EQ(Ke.rows(), 24);
    double max_asym = (Ke - Ke.transpose()).cwiseAbs().maxCoeff();
    EXPECT_LT(max_asym, 1e-10 * Ke.cwiseAbs().maxCoeff());
}

TEST(CHexa8, StiffnessIsPositiveSemiDefinite) {
    Model m = make_solid_model();
    add_grid(m,1,0,0,0); add_grid(m,2,1,0,0); add_grid(m,3,1,1,0); add_grid(m,4,0,1,0);
    add_grid(m,5,0,0,1); add_grid(m,6,1,0,1); add_grid(m,7,1,1,1); add_grid(m,8,0,1,1);

    std::array<NodeId,8> nodes{NodeId{1},NodeId{2},NodeId{3},NodeId{4},
                                NodeId{5},NodeId{6},NodeId{7},NodeId{8}};
    CHexa8 elem(ElementId{1}, PropertyId{1}, nodes, m);
    LocalKe Ke = elem.stiffness_matrix();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Ke);
    double min_ev = eig.eigenvalues().minCoeff();
    double max_ev = eig.eigenvalues().maxCoeff();
    EXPECT_GE(min_ev, -1e-6 * max_ev);
}
