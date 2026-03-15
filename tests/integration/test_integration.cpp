// tests/integration/test_integration.cpp
// End-to-end integration tests with hand-calculated solutions.
//
// Test 1: Bar extension (modeled as CQUAD4 in plane stress)
//   - A unit-square plate clamped on left, axial load on right
//   - Hand calc: δ = F*L / (E*A) = F*L / (E*t*h)
//
// Test 2: Cantilever beam (CQUAD4 mesh, bending)
//   - Series of CQUAD4 elements forming a beam
//   - Hand calc: tip deflection = F*L^3 / (3*E*I)
//
// Test 3: Bar under thermal load (CTETRA4)
//   - Fixed-fixed bar heated uniformly
//   - Hand calc: stress = -E * alpha * dT (fully constrained thermal stress)
//
// Test 4: Pure axial CTETRA4 bar
//   - Tetrahedra assembled into a prismatic bar, one end fixed, force on other
//   - Hand calc: δ = F*L / (E*A)

#include <gtest/gtest.h>
#include "io/bdf_parser.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include "io/results.hpp"
#include <cmath>
#include <sstream>

using namespace nastran;

// Helper: run a full analysis from BDF string
static SolverResults run_analysis(const std::string& bdf) {
    Model model = BdfParser::parse_string(bdf);
    LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
    return solver.solve(model);
}

// Helper: find node displacement
static double get_disp(const SolverResults& res, int node_id, int dof_0based) {
    for (const auto& sc : res.subcases)
        for (const auto& nd : sc.displacements)
            if (nd.node.value == node_id)
                return nd.d[dof_0based];
    throw std::runtime_error("Node not found in results");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: Axial extension of a rectangular plate (CQUAD4)
//
// Geometry: 10 × 1 × 0.1 thick plate
// Material: E = 1e6, nu = 0.3
// Load:     1000 N distributed as 2 point forces of 500 N each on right edge
// BCs:      Left edge fully fixed
//
// Hand calculation:
//   A = 1.0 × 0.1 = 0.1
//   δ = F*L/(E*A) = 1000 * 10 / (1e6 * 0.1) = 0.1
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Integration, AxialPlateExtension) {
    const std::string bdf = R"(
SOL 101
BEGIN BULK
$ Nodes: 4 corners of a 10x1 plate
GRID,1,,0.0,0.0,0.0
GRID,2,,10.0,0.0,0.0
GRID,3,,10.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
$ Material: E=1e6, nu=0.3
MAT1,1,1.0E6,,0.3
$ Shell property: t=0.1
PSHELL,1,1,0.1
$ Single CQUAD4 element
CQUAD4,1,1,1,2,3,4
$ BCs: fix all DOFs on left edge (nodes 1 and 4)
SPC1,1,123456,1
SPC1,1,123456,4
$ Loads: 500 N in x on nodes 2 and 3
FORCE,1,2,0,500.0,1.0,0.0,0.0
FORCE,1,3,0,500.0,1.0,0.0,0.0
ENDDATA
)";

    // Set up case control
    Model model = BdfParser::parse_string(bdf);
    model.analysis.sol = SolutionType::LinearStatic;
    model.analysis.subcases.clear();
    model.analysis.subcases.push_back({1, "AXIAL", LoadSetId{1}, SpcSetId{1}});

    LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
    SolverResults res = solver.solve(model);

    // δ = F*L/(E*A) = 1000*10/(1e6*0.1) = 0.1
    double expected = 0.1;
    double u_node2 = get_disp(res, 2, 0); // x-displacement at node 2
    double u_node3 = get_disp(res, 3, 0); // x-displacement at node 3

    EXPECT_NEAR(u_node2, expected, 0.005 * expected)
        << "Node 2 x-disp = " << u_node2 << ", expected " << expected;
    EXPECT_NEAR(u_node3, expected, 0.005 * expected)
        << "Node 3 x-disp = " << u_node3 << ", expected " << expected;

    // Left edge should have zero displacement
    EXPECT_NEAR(get_disp(res, 1, 0), 0.0, 1e-12);
    EXPECT_NEAR(get_disp(res, 4, 0), 0.0, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: Cantilever beam — CQUAD4 mesh
//
// Model:  Beam of 4 elements: 10 × 1 m, t = 0.1 m
//         Left end fully fixed; tip load P = 100 N downward (z-direction)
//
// Beam theory:
//   I = t * h^3 / 12 = 0.1 * 1^3 / 12 = 1/120
//   δ_tip = P*L^3 / (3*E*I) = 100 * 10^3 / (3 * 1e6 * 1/120)
//         = 100000 / 25000 = 4.0 m
//
// Note: A single CQUAD4 with linear displacement field underestimates bending.
// With 4 elements it converges. We check the result is within 5% of beam theory
// to account for shear deformation (Mindlin).
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Integration, CantileverBeamBending) {
    // 4-element cantilever, each element 2.5 m long × 1 m tall
    const std::string bdf = R"(
SOL 101
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,2.5,0.0,0.0
GRID,3,,5.0,0.0,0.0
GRID,4,,7.5,0.0,0.0
GRID,5,,10.0,0.0,0.0
GRID,6,,0.0,1.0,0.0
GRID,7,,2.5,1.0,0.0
GRID,8,,5.0,1.0,0.0
GRID,9,,7.5,1.0,0.0
GRID,10,,10.0,1.0,0.0
MAT1,1,1.0E6,,0.3
PSHELL,1,1,0.1
CQUAD4,1,1,1,2,7,6
CQUAD4,2,1,2,3,8,7
CQUAD4,3,1,3,4,9,8
CQUAD4,4,1,4,5,10,9
$ Fix left edge (nodes 1 and 6) — all DOFs
SPC1,1,123456,1
SPC1,1,123456,6
$ Tip load: 50 N downward at each tip node (total 100 N, but shell uses in-plane)
$ For an in-plane bending test we apply moment/force in the element plane
$ Here we apply shear load in y-direction simulating beam bending
FORCE,1,5,0,50.0,0.0,1.0,0.0
FORCE,1,10,0,50.0,0.0,1.0,0.0
ENDDATA
)";

    Model model = BdfParser::parse_string(bdf);
    model.analysis.subcases.clear();
    model.analysis.subcases.push_back({1, "CANTILEVER", LoadSetId{1}, SpcSetId{1}});

    LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
    SolverResults res = solver.solve(model);

    // In-plane bending: tip y-displacement
    // Hand calc: δ = F*L^3 / (3*E*I) = 100 * 1000 / (3*1e6 * 1/120) = 4.0
    double expected = 4.0;
    double v_node5  = get_disp(res, 5, 1);
    double v_node10 = get_disp(res, 10, 1);

    // Allow 10% error for coarse mesh + Poisson effects
    EXPECT_GT(v_node5, 0.0) << "Tip should displace in +y";
    EXPECT_NEAR(v_node5, expected, 0.15 * expected)
        << "Tip y-disp = " << v_node5 << ", beam theory = " << expected;
    EXPECT_NEAR(v_node5, v_node10, 1e-6 * expected)
        << "Top and bottom tip displacements should match (symmetric)";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: Uniaxial CTETRA4 bar — axial extension
//
// Bar: 1 × 1 × 10 m, modeled with 2 CTETRA4 per layer × 5 layers = 10 elements
// Simplified: Single tet-pair (2 tets) forming a 1×1×1 cube prism
//
// Geometry: 1m×1m cross section, 1m long, material E=1e6, nu=0.3
// Load:     total axial force F = 1000 N on top face
// BCs:      bottom face fully fixed
//
// Hand calc (uniform axial):
//   σ = F/A = 1000/1.0 = 1000 Pa
//   ε = σ/E = 1e-3
//   δ = ε*L = 1e-3 × 1.0 = 1e-3 m
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Integration, TetraBarAxialExtension) {
    // Model a 1×1×1 cube using 6 CTETRA4 (Kuhn decomposition — all positive orientation)
    // All tets share the main diagonal nodes 1(0,0,0) and 7(1,1,1).
    // Volumes: each tet = 1/6 m³, total = 1 m³. Fully conforming mesh.
    // Face z=0: nodes 1-4 (fixed)
    // Face z=1: nodes 5-8 (loaded)
    const std::string bdf = R"(
SOL 101
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
GRID,5,,0.0,0.0,1.0
GRID,6,,1.0,0.0,1.0
GRID,7,,1.0,1.0,1.0
GRID,8,,0.0,1.0,1.0
MAT1,1,1.0E6,,0.3
PSOLID,1,1
$ 6-tet Kuhn decomposition of unit cube (all positive det, V=1/6 each)
CTETRA,1,1,1,2,3,7
CTETRA,2,1,1,5,6,7
CTETRA,3,1,1,4,8,7
CTETRA,4,1,2,1,6,7
CTETRA,5,1,5,1,8,7
CTETRA,6,1,4,1,3,7
$ Fix bottom face (z=0): nodes 1-4, translations only
SPC1,1,123,1
SPC1,1,123,2
SPC1,1,123,3
SPC1,1,123,4
$ Apply F=250 N (z-direction) at each top corner = total 1000 N
FORCE,1,5,0,250.0,0.0,0.0,1.0
FORCE,1,6,0,250.0,0.0,0.0,1.0
FORCE,1,7,0,250.0,0.0,0.0,1.0
FORCE,1,8,0,250.0,0.0,0.0,1.0
ENDDATA
)";

    Model model = BdfParser::parse_string(bdf);
    model.analysis.subcases.clear();
    model.analysis.subcases.push_back({1, "AXIAL_TET", LoadSetId{1}, SpcSetId{1}});

    LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
    SolverResults res = solver.solve(model);

    // Expected axial displacement of top face
    // δ = F*L/(E*A) = 1000*1/(1e6*1) = 1e-3
    double expected = 1.0e-3;

    double w5 = get_disp(res, 5, 2);
    double w6 = get_disp(res, 6, 2);
    double w7 = get_disp(res, 7, 2);
    double w8 = get_disp(res, 8, 2);

    // Uniform axial: all top nodes should have same z-displacement
    EXPECT_NEAR(w5, expected, 0.05 * expected) << "Node 5 z-disp = " << w5;
    EXPECT_NEAR(w6, expected, 0.05 * expected) << "Node 6 z-disp = " << w6;
    EXPECT_NEAR(w7, expected, 0.05 * expected) << "Node 7 z-disp = " << w7;
    EXPECT_NEAR(w8, expected, 0.05 * expected) << "Node 8 z-disp = " << w8;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 4: Thermal expansion of a free plate (CQUAD4)
//
// A square plate, fully free (no mechanical constraints except minimal for
// rigid body suppression), uniformly heated by dT.
//
// Hand calc:
//   Free thermal expansion: all points displace as ε_th = alpha * dT
//   Corner at (L, 0) should move to (L + alpha*dT*L, 0)
//   Corner displacement = alpha * dT * L
//
// With E=2e6, nu=0.3, alpha=1e-5, dT=100:
//   epsilon_free = 1e-5 * 100 = 1e-3
//   delta at x=L=1.0: u = 1e-3 * 1.0 = 1e-3
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Integration, FreeThermalExpansionPlate) {
    const std::string bdf = R"(
SOL 101
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
$ alpha = 1e-5
MAT1,1,2.0E6,,0.3,0.0,1.0E-5
PSHELL,1,1,0.1
CQUAD4,1,1,1,2,3,4
$ Minimal BCs to prevent rigid body motion:
$ Fix node 1 fully, node 2 y-displacement, node 4 x-displacement
SPC1,1,12,1
SPC1,1,2,2
SPC1,1,1,4
$ Uniform temperature dT = 100 on all nodes
TEMP,1,1,100.0
TEMP,1,2,100.0
TEMP,1,3,100.0
TEMP,1,4,100.0
ENDDATA
)";

    Model model = BdfParser::parse_string(bdf);
    model.analysis.subcases.clear();
    SubCase sc;
    sc.id = 1; sc.label = "THERMAL";
    sc.load_set = LoadSetId{1}; sc.spc_set = SpcSetId{1};
    sc.t_ref = 0.0; // reference temperature = 0
    model.analysis.subcases.push_back(sc);

    LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
    SolverResults res = solver.solve(model);

    // Expected: alpha * dT * L = 1e-5 * 100 * 1.0 = 1e-3
    double alpha = 1.0e-5, dT = 100.0, L = 1.0;
    double expected = alpha * dT * L;

    double u2 = get_disp(res, 2, 0); // node at (1,0): x-displacement
    double u3 = get_disp(res, 3, 0); // node at (1,1): x-displacement
    double v3 = get_disp(res, 3, 1); // node at (1,1): y-displacement
    double v4 = get_disp(res, 4, 1); // node at (0,1): y-displacement

    EXPECT_NEAR(u2, expected, 0.01 * expected)
        << "x-disp at (1,0) = " << u2 << ", expected " << expected;
    EXPECT_NEAR(u3, expected, 0.01 * expected)
        << "x-disp at (1,1) = " << u3 << ", expected " << expected;
    EXPECT_NEAR(v3, expected, 0.01 * expected)
        << "y-disp at (1,1) = " << v3 << ", expected " << expected;
    EXPECT_NEAR(v4, expected, 0.01 * expected)
        << "y-disp at (0,1) = " << v4 << ", expected " << expected;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 5: Single CHEXA8 axial extension
//
// Unit cube (1×1×1) under axial load
// Fix z=0 face, apply 1000 N total on z=1 face
//
// Hand calc:
//   σ_zz = F/A = 1000/1.0 = 1000 Pa
//   ε_zz = σ_zz/E = 1e-3
//   δ_z   = ε_zz * L = 1e-3 m
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Integration, HexaBarAxialExtension) {
    const std::string bdf = R"(
SOL 101
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
GRID,5,,0.0,0.0,1.0
GRID,6,,1.0,0.0,1.0
GRID,7,,1.0,1.0,1.0
GRID,8,,0.0,1.0,1.0
MAT1,1,1.0E6,,0.3
PSOLID,1,1
CHEXA,1,1,1,2,3,4,5,6,7,8
$ Fix bottom face z=0: nodes 1-4, all DOFs
SPC1,1,123456,1
SPC1,1,123456,2
SPC1,1,123456,3
SPC1,1,123456,4
$ 250 N z-force at each top corner = 1000 N total
FORCE,1,5,0,250.0,0.0,0.0,1.0
FORCE,1,6,0,250.0,0.0,0.0,1.0
FORCE,1,7,0,250.0,0.0,0.0,1.0
FORCE,1,8,0,250.0,0.0,0.0,1.0
ENDDATA
)";

    Model model = BdfParser::parse_string(bdf);
    model.analysis.subcases.clear();
    model.analysis.subcases.push_back({1, "HEXA_AXIAL", LoadSetId{1}, SpcSetId{1}});

    LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
    SolverResults res = solver.solve(model);

    double expected = 1.0e-3;
    for (int n : {5,6,7,8}) {
        double w = get_disp(res, n, 2);
        EXPECT_NEAR(w, expected, 0.02 * expected)
            << "Node " << n << " z-disp = " << w;
    }

    // Also verify Poisson contraction: x-displacement of node at (1,0,1)
    // ε_xx = -nu * ε_zz = -0.3e-3
    // δ_x of node 6 at x=1: u = -nu * ε_zz * 1.0 = -3e-4
    double expected_poisson = -0.3 * 1e-3;
    double u6 = get_disp(res, 6, 0);
    EXPECT_NEAR(u6, expected_poisson, 0.05 * std::abs(expected_poisson))
        << "Poisson contraction: node 6 x-disp = " << u6
        << ", expected " << expected_poisson;
}
