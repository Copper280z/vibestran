// tests/integration/test_mpc_frames.cpp
// Explicit MPC sets and the components that MPC, CELAS and CMASS cards name on a
// grid with a displacement frame, checked against closed-form solutions.

#include "elements/element_factory.hpp"
#include "elements/line_elements.hpp"
#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/eigensolver_backend.hpp"
#include "solver/linear_static.hpp"
#include "solver/modal.hpp"
#include "solver/solver_backend.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>

using namespace vibestran;

namespace {

SubCaseResults solve_static(const std::string& deck) {
    Model model = BdfParser::parse_string(deck);
    LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
    return solver.solve(model).subcases.at(0);
}

ModalSubCaseResults solve_modal(const std::string& deck) {
    Model model = BdfParser::parse_string(deck);
    ModalSolver solver(std::make_unique<SpectraEigensolverBackend>());
    return solver.solve(model).subcases.at(0);
}

const NodeDisplacement& disp(const SubCaseResults& r, int node) {
    const auto it = std::ranges::find(r.displacements, NodeId{node}, &NodeDisplacement::node);
    if (it == r.displacements.end())
        throw std::runtime_error(std::format("no displacement for point {}", node));
    return *it;
}

const SpcForce& force_at(const std::vector<SpcForce>& forces, int node) {
    const auto it = std::ranges::find(forces, NodeId{node}, &SpcForce::node);
    if (it == forces.end())
        throw std::runtime_error(std::format("no constraint force at point {}", node));
    return *it;
}

const RodResult& rod(const SubCaseResults& r, int eid) {
    const auto it = std::ranges::find(r.rod_results, ElementId{eid}, &RodResult::eid);
    if (it == r.rod_results.end())
        throw std::runtime_error(std::format("no result for rod {}", eid));
    return *it;
}

void expect_grid_balance(const SubCaseResults& r, const double tolerance) {
    for (const auto& grid : r.gp_forces)
        for (const double total : grid.totals)
            EXPECT_NEAR(total, 0.0, tolerance) << "grid " << grid.node.value;
}

// E = 1e7, nu = 0.25 so G = 4e6. PROD 1 has A = 2 only, PROD 2 also J = 1.5.
const std::string kRodProps =
    "MAT1,1,1.0E7,,0.25\n"
    "PROD,1,1,2.0\n"
    "PROD,2,1,2.0,1.5\n";

/// Axial stiffness of a PROD 1 or 2 rod of length L, and torsional of a PROD 2.
constexpr double axial_k(const double L) { return 1.0e7 * 2.0 / L; }
constexpr double torsional_k(const double L) { return 4.0e6 * 1.5 / L; }

std::string static_deck(const std::string& bulk, const std::string& cases = "") {
    return "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nDISP = ALL\nSPCF = ALL\nMPCF = ALL\n"
           "GPFORCE = ALL\nFORCE = ALL\n" + cases +
           "BEGIN BULK\n" + kRodProps + bulk + "ENDDATA\n";
}

/// A rectangular system at the origin, turned about basic Z so that its X axis
/// is `degrees` above basic X.
std::string turned_frame(const int cid, const double degrees) {
    const double angle = degrees * std::numbers::pi / 180.0;
    std::ostringstream card;
    card << std::setprecision(17) << "CORD2R," << cid << ",,0.,0.,0.,0.,0.,1.,\n,"
         << std::cos(angle) << "," << std::sin(angle) << ",0.\n";
    return card.str();
}

std::string real_field(const double value) { return std::format("{:.17e}", value); }

// Five rods in a line; the rods' common grids follow one another through the
// sets of the MPCADD. A closed ring: grid 6 is tied back to the fixed grid 1, and
// grids 2 to 5 move together, so the load divides equally between the end rods.
const std::string kChainGrids =
    "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,20.,0.,0.\nGRID,4,,30.,0.,0.\n"
    "GRID,5,,40.,0.,0.\nGRID,6,,50.,0.,0.\n"
    "CROD,12,1,1,2\nCROD,23,1,2,3\nCROD,34,1,3,4\nCROD,45,1,4,5\nCROD,56,1,5,6\n"
    "SPC1,1,1,1\nRBE2,32,3,1,2\n"
    "FORCE,1,2,,200.,1.,0.,0.\nFORCE,1,3,,300.,1.,0.,0.\n"
    "FORCE,1,4,,400.,1.,0.,0.\nFORCE,1,5,,500.,1.,0.,0.\n";
const std::string kChainMpcs =
    "MPC,34,3,1,1.,4,1,-1.\nMPC,45,4,1,1.,5,1,-1.\nMPC,61,6,1,1.,1,1,-1.\n";

void expect_chain_solution(const SubCaseResults& r) {
    const double shared = 1400.0 / (2.0 * axial_k(10.0));
    for (const int grid : {2, 3, 4, 5})
        EXPECT_NEAR(disp(r, grid).d[0], shared, 1e-12) << "grid " << grid;
    EXPECT_NEAR(disp(r, 6).d[0], 0.0, 1e-12);
    EXPECT_NEAR(rod(r, 12).axial_force, 700.0, 1e-7);
    EXPECT_NEAR(rod(r, 34).axial_force, 0.0, 1e-7);
    EXPECT_NEAR(rod(r, 56).axial_force, -700.0, 1e-7);
    // The end rod reaches the support through the tie, so grid 1 reacts it all.
    EXPECT_NEAR(force_at(r.spc_forces, 1).f[0], -1400.0, 1e-7);
    EXPECT_NEAR(force_at(r.mpc_forces, 6).f[0], -700.0, 1e-7);
    EXPECT_NEAR(force_at(r.mpc_forces, 1).f[0], 700.0, 1e-7);
    expect_grid_balance(r, 1e-7);
}

} // namespace

// ── MPCADD ────────────────────────────────────────────────────────────────────

TEST(MpcAdd, SolutionDoesNotDependOnWhereTheCardsStand) {
    expect_chain_solution(solve_static(static_deck(
        "MPCADD,2,34,45,61\n" + kChainMpcs + kChainGrids, "MPC = 2\n")));
    expect_chain_solution(solve_static(static_deck(
        kChainGrids + kChainMpcs + "MPCADD,2,34,45,61\n", "MPC = 2\n")));
    expect_chain_solution(solve_static(static_deck(
        kChainGrids + "MPC,34,3,1,1.,4,1,-1.\nMPCADD,2,34,45,61\n"
        "MPC,45,4,1,1.,5,1,-1.\nMPC,61,6,1,1.,1,1,-1.\n", "MPC = 2\n")));
}

TEST(MpcAdd, OnlyTheSelectedSetApplies) {
    // Selecting one source leaves the others out: with only 3 = 4 tied, grid 6 is
    // free, grids 5 and 6 carry no load past the 500 at grid 5, and so on.
    const auto r = solve_static(static_deck(
        "MPCADD,2,34,45,61\n" + kChainMpcs + kChainGrids, "MPC = 34\n"));
    const double k = axial_k(10.0);
    // Grid 2 = 3 = 4 by the RBE2 and the tie, so rod 12 carries all 1400.
    EXPECT_NEAR(rod(r, 12).axial_force, 1400.0, 1e-7);
    EXPECT_NEAR(disp(r, 4).d[0], 1400.0 / k, 1e-12);
    EXPECT_NEAR(disp(r, 5).d[0], 1400.0 / k + 500.0 / k, 1e-12);
    EXPECT_NEAR(disp(r, 6).d[0], disp(r, 5).d[0], 1e-12);
}

TEST(MpcAdd, TakesPrecedenceOverMpcCardsOfItsOwnSid) {
    const Model model = BdfParser::parse_string(
        "BEGIN BULK\nMPCADD,5,6\nMPC,5,1,1,1.,2,1,-1.\nMPC,6,3,1,1.,4,1,-1.\nENDDATA\n");
    const auto set = model.mpcs_for_set(MpcSetId{5});
    ASSERT_EQ(set.size(), 1U);
    EXPECT_EQ(set[0]->terms[0].node, NodeId{3});
    EXPECT_EQ(model.mpcs_for_set(MpcSetId{6}).size(), 1U);
}

TEST(MpcAdd, MayNameItsOwnSidOrAnotherMpcadd) {
    const Model model = BdfParser::parse_string(
        "BEGIN BULK\nMPCADD,9,7,8\nMPCADD,7,7,6\nMPCADD,7,5\n"
        "MPC,5,1,1,1.,2,1,-1.\nMPC,6,3,1,1.,4,1,-1.\nMPC,7,5,1,1.,6,1,-1.\n"
        "MPC,8,7,1,1.,8,1,-1.\nENDDATA\n");
    EXPECT_EQ(model.mpcs_for_set(MpcSetId{7}).size(), 3U);
    const auto all = model.mpcs_for_set(MpcSetId{9});
    ASSERT_EQ(all.size(), 4U);
    // In the order the cards name them, which sets the preferred dependents.
    EXPECT_EQ(all[0]->terms[0].node, NodeId{5});
    EXPECT_EQ(all[1]->terms[0].node, NodeId{3});
    EXPECT_EQ(all[2]->terms[0].node, NodeId{1});
    EXPECT_EQ(all[3]->terms[0].node, NodeId{7});
}

TEST(MpcAdd, RejectsWhatWouldSilentlyLoseConstraints) {
    try {
        (void)BdfParser::parse_string("BEGIN BULK\nMPC,5,1,1,1.,2,1,-1.\nMPCADD,9,5,6\nENDDATA\n");
        FAIL() << "an undefined set was accepted";
    } catch (const ParseError& e) {
        EXPECT_NE(std::string(e.what()).find("MPCADD 9 names MPC set 6"), std::string::npos)
            << e.what();
    }
    try {
        (void)BdfParser::parse_string("BEGIN BULK\nMPCADD,1,2\nMPCADD,2,1\nENDDATA\n");
        FAIL() << "a cycle was accepted";
    } catch (const ParseError& e) {
        EXPECT_NE(std::string(e.what()).find("includes it"), std::string::npos) << e.what();
    }
}

// ── MPC components on a CD grid ───────────────────────────────────────────────

TEST(CdMpc, ComponentsAreReadInTheDisplacementFrame) {
    // MYSTRAN's SB-RODLOAD-MPC-FOR-SING with a transverse load added. The MPC
    // normal is basic Y, written in a frame 22.5 degrees above basic X, so the
    // grid slides along basic X: the rod takes 100 and the MPC takes 30.
    const double angle = 22.5 * std::numbers::pi / 180.0;
    const double c = std::cos(angle), s = std::sin(angle), L = 10.0;
    const auto r = solve_static(static_deck(
        turned_frame(12, 22.5) +
        "GRID,101,,0.,0.,0.\nGRID,102,,10.,0.,0.,12,3456\nCROD,1,1,101,102\nSPC1,1,123456,101\n"
        "MPC,4,102,2," + real_field(c) + ",102,1," + real_field(s) + "\n"
        "FORCE,1,102,0,1.,100.,30.,0.\n", "MPC = 4\n"));
    const double u = 100.0 / axial_k(L);
    EXPECT_NEAR(disp(r, 102).d[0], u * c, 1e-12);
    EXPECT_NEAR(disp(r, 102).d[1], -u * s, 1e-12);
    EXPECT_NEAR(rod(r, 1).axial_force, 100.0, 1e-7);
    // K u - F at grid 102 is -30 along basic Y, reported in the frame.
    const SpcForce& mpc = force_at(r.mpc_forces, 102);
    EXPECT_EQ(mpc.coord, CoordId{12});
    EXPECT_NEAR(mpc.f[0], -30.0 * s, 1e-7);
    EXPECT_NEAR(mpc.f[1], -30.0 * c, 1e-7);
    expect_grid_balance(r, 1e-7);
}

TEST(CdMpc, TiesComponentsOfDifferentFrames) {
    // Grid 2's T1 is basic Y, along its rod; grid 4's T1 is basic X, along its own.
    // The tie puts the two rods in parallel. Read in basic it would tie grid 2's
    // transverse motion instead and leave the rods uncoupled.
    const double P = 300.0, k1 = axial_k(10.0), k2 = axial_k(20.0);
    const auto r = solve_static(static_deck(
        turned_frame(12, 90.0) +
        "GRID,1,,0.,0.,0.\nGRID,2,,0.,10.,0.,12\nGRID,3,,20.,0.,0.\nGRID,4,,40.,0.,0.\n"
        "CROD,1,1,1,2\nCROD,2,1,3,4\nSPC1,1,123456,1,3\nMPC,7,2,1,1.,4,1,-1.\n"
        "FORCE,1,2,0,1.,0.,300.,0.\n", "MPC = 7\n"));
    const double u = P / (k1 + k2);
    EXPECT_NEAR(disp(r, 2).d[0], u, 1e-12);
    EXPECT_NEAR(disp(r, 4).d[0], u, 1e-12);
    EXPECT_NEAR(disp(r, 2).d[1], 0.0, 1e-12);
    EXPECT_NEAR(rod(r, 1).axial_force, k1 * u, 1e-7);
    EXPECT_NEAR(rod(r, 2).axial_force, k2 * u, 1e-7);
    // Equal and opposite multiplier terms on the two T1 components.
    EXPECT_NEAR(force_at(r.mpc_forces, 2).f[0], -k2 * u, 1e-7);
    EXPECT_NEAR(force_at(r.mpc_forces, 4).f[0], k2 * u, 1e-7);
    expect_grid_balance(r, 1e-7);
}

TEST(CdMpc, CylindricalFrameFollowsTheGrid) {
    // Radial T1 is held at a grid at 45 degrees between equal rods along X and Y,
    // so the grid can only slide tangentially, with stiffness k in that direction.
    const double P = 100.0, k = axial_k(10.0), root2 = std::numbers::sqrt2;
    const auto r = solve_static(static_deck(
        "CORD2C,3,,0.,0.,0.,0.,0.,1.,\n,1.,0.,0.\n"
        "GRID,1,,20.,10.,0.\nGRID,2,,10.,10.,0.,3\nGRID,3,,10.,20.,0.\n"
        "CROD,1,1,1,2\nCROD,2,1,3,2\nSPC1,1,123456,1,3\nMPC,4,2,1,1.\n"
        "FORCE,1,2,0,1.,100.,0.,0.\n", "MPC = 4\n"));
    EXPECT_NEAR(disp(r, 2).d[0], 0.0, 1e-12);
    EXPECT_NEAR(disp(r, 2).d[1], -P / (root2 * k), 1e-12);
    EXPECT_NEAR(rod(r, 1).axial_force, -P / 2.0, 1e-7);
    EXPECT_NEAR(rod(r, 2).axial_force, P / 2.0, 1e-7);
    EXPECT_NEAR(force_at(r.mpc_forces, 2).f[0], -P / root2, 1e-7);
    EXPECT_NEAR(force_at(r.mpc_forces, 2).f[1], 0.0, 1e-7);
    expect_grid_balance(r, 1e-7);
}

TEST(CdMpc, RotationsTurnWithTheSameAxes) {
    // Grid 4's R1 is the twist of its rod along basic Y; tied to the twist of a
    // rod along basic X, the two share the moment.
    const double M = 90.0, kt = torsional_k(10.0);
    const auto r = solve_static(static_deck(
        turned_frame(12, 90.0) +
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,20.,0.,0.\nGRID,4,,20.,10.,0.,12\n"
        "CROD,1,2,1,2\nCROD,2,2,3,4\nSPC1,1,123456,1,3\nMPC,4,2,4,1.,4,4,-1.\n"
        "MOMENT,1,2,0,1.,90.,0.,0.\n", "MPC = 4\n"));
    const double theta = M / (2.0 * kt);
    EXPECT_NEAR(disp(r, 2).d[3], theta, 1e-12);
    EXPECT_NEAR(disp(r, 4).d[3], theta, 1e-12);
    EXPECT_NEAR(rod(r, 1).torque, M / 2.0, 1e-7);
    EXPECT_NEAR(rod(r, 2).torque, M / 2.0, 1e-7);
    expect_grid_balance(r, 1e-7);
}

TEST(CdMpc, AutoSpcTakesOnlyTheComponentsTheMpcLeaves) {
    // T1 of a frame at 45 degrees is held by a one-term MPC. T2 keeps the rod's
    // stiffness k sin^2(45) = k / 2, so it must stay free, and T3 has none, so
    // AUTOSPC takes it along with a load on it.
    const double P = 50.0, k = axial_k(10.0), root2 = std::numbers::sqrt2;
    const auto r = solve_static(static_deck(
        turned_frame(12, 45.0) +
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.,12\nCROD,1,1,1,2\nSPC1,1,123456,1\nMPC,4,2,1,1.\n"
        "FORCE,1,2,12,50.,0.,1.,0.\nFORCE,1,2,0,75.,0.,0.,1.\n", "MPC = 4\n"));
    EXPECT_NEAR(disp(r, 2).d[0], 0.0, 1e-12);
    EXPECT_NEAR(disp(r, 2).d[1], 2.0 * P / k, 1e-12);
    EXPECT_EQ(disp(r, 2).d[2], 0.0);
    EXPECT_NEAR(rod(r, 1).axial_force, -root2 * P, 1e-7);
    EXPECT_NEAR(force_at(r.spc_forces, 2).f[2], -75.0, 1e-7);
    expect_grid_balance(r, 1e-7);
}

TEST(CdMpc, ModalUsesTheSameEquations) {
    // A mass on grid 2 between the two rods of TiesComponentsOfDifferentFrames.
    const double k1 = axial_k(10.0), k2 = axial_k(20.0), m = 2.5;
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMPC = 7\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,1\n" + kRodProps +
        turned_frame(12, 90.0) +
        "GRID,1,,0.,0.,0.\nGRID,2,,0.,10.,0.,12\nGRID,3,,20.,0.,0.\nGRID,4,,40.,0.,0.\n"
        "CROD,1,1,1,2\nCROD,2,1,3,4\nSPC1,1,123456,1,3\nMPC,7,2,1,1.,4,1,-1.\n"
        "CONM2,5,2,,2.5\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 1U);
    EXPECT_NEAR(r.modes[0].eigenvalue / ((k1 + k2) / m), 1.0, 1e-10);
}

// ── CELAS and CMASS components on a CD grid ───────────────────────────────────

TEST(CdScalar, ElementMatrixActsAlongTheDisplacementAxis) {
    const std::string deck =
        "SOL 101\nCEND\nBEGIN BULK\n" + turned_frame(12, 30.0) +
        "GRID,1,,0.,0.,0.,12\nGRID,2,,1.,0.,0.\nCELAS2,7,40.,1,2,2,1\nENDDATA\n";
    const Model model = BdfParser::parse_string(deck);
    const auto element = make_element(model.elements.at(0), model);
    // e_y of the frame at grid 1, then basic X at grid 2 with the opposite sign.
    const double c = std::cos(std::numbers::pi / 6.0), s = 0.5;
    Eigen::Vector3d d(-s, c, -1.0);
    const LocalKe K = element->stiffness_matrix();
    ASSERT_EQ(K.rows(), 3);
    EXPECT_LT((K - 40.0 * d * d.transpose()).norm(), 1e-12);
    DofMap dofs;
    dofs.build(model.nodes, 6);
    EXPECT_EQ(element->global_dof_indices(dofs),
              (std::vector<EqIndex>{dofs.eq_index(NodeId{1}, 0), dofs.eq_index(NodeId{1}, 1),
                                    dofs.eq_index(NodeId{2}, 0)}));
}

TEST(CdScalar, GroundedSpringResistsOnlyItsAxis) {
    // T1 of a frame 30 degrees above basic X takes the load's projection and AUTOSPC
    // reacts the rest on T2: read in basic, the spring would take all of it.
    const double P = 200.0, k = 500.0, c = std::cos(std::numbers::pi / 6.0), s = 0.5;
    const auto r = solve_static(static_deck(
        turned_frame(12, 30.0) +
        "GRID,1,,0.,0.,0.,12\nCELAS2,7,500.,1,1\nFORCE,1,1,0,200.,1.,0.,0.\n"));
    EXPECT_NEAR(disp(r, 1).d[0], P * c / k, 1e-12);
    EXPECT_NEAR(disp(r, 1).d[1], 0.0, 1e-12);
    EXPECT_NEAR(disp(r, 1).d[2], 0.0, 1e-12);
    // K u - F on T2 is -P (e2 . x) = P s.
    EXPECT_NEAR(force_at(r.spc_forces, 1).f[1], P * s, 1e-9);
    EXPECT_NEAR(force_at(r.spc_forces, 1).f[0], 0.0, 1e-9);
    expect_grid_balance(r, 1e-9);
}

TEST(CdScalar, SpringsInSeriesAcrossFramesAndAScalarPoint) {
    // A load along T2 of grid 1 runs through the spring to scalar point 9, then
    // the spring from there to ground.
    const double P = 60.0, k1 = 300.0, k2 = 200.0;
    const auto r = solve_static(static_deck(
        turned_frame(12, 40.0) +
        "GRID,1,,0.,0.,0.,12\nSPOINT,9\nCELAS2,7,300.,1,2,9\nCELAS2,8,200.,9\n"
        "FORCE,1,1,12,60.,0.,1.,0.\n"));
    EXPECT_NEAR(disp(r, 9).d[0], P / k2, 1e-12);
    EXPECT_NEAR(disp(r, 1).d[1], P / k1 + P / k2, 1e-12);
    EXPECT_NEAR(disp(r, 1).d[0], 0.0, 1e-12);
    expect_grid_balance(r, 1e-9);
}

TEST(CdScalar, EnforcedDisplacementDrivesTheSpring) {
    // An SPC value on T1 of the frame drives grid 2 through a divider of 3:1, and
    // the reaction appears on T1, through the element's basic components.
    const auto r = solve_static(static_deck(
        turned_frame(12, 30.0) +
        "GRID,1,,0.,0.,0.,12\nGRID,2,,1.,0.,0.\nCELAS2,7,30.,1,1,2,1\nCELAS2,8,10.,2,1\n"
        "SPC,1,1,1,0.4\n"));
    EXPECT_NEAR(disp(r, 1).d[0], 0.4, 1e-12);
    EXPECT_NEAR(disp(r, 2).d[0], 0.3, 1e-12);
    EXPECT_NEAR(force_at(r.spc_forces, 1).f[0], 3.0, 1e-9);
    EXPECT_NEAR(force_at(r.spc_forces, 1).f[1], 0.0, 1e-9);
    expect_grid_balance(r, 1e-9);
}

TEST(CdScalar, RodAndSpringShareAGridInItsFrame) {
    // A spring on T2 of a 90-degree frame is basic -X, along the rod, so the two
    // act in parallel. Read in basic it would stand across the rod.
    const double P = 1000.0, kr = axial_k(10.0), ks = 1.0e6;
    const auto r = solve_static(static_deck(
        turned_frame(12, 90.0) +
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.,12\nCROD,1,1,1,2\nSPC1,1,123456,1\n"
        "CELAS2,7,1.0E6,2,2\nFORCE,1,2,0,1000.,1.,0.,0.\n"));
    const double u = P / (kr + ks);
    EXPECT_NEAR(disp(r, 2).d[1], -u, 1e-12);
    EXPECT_NEAR(rod(r, 1).axial_force, kr * u, 1e-7);
    expect_grid_balance(r, 1e-7);
}

TEST(CdScalar, GravityActsOnTheMassAlongItsAxis) {
    // M = m e1 e1^T, so 3 g along basic X loads T1 by m g cos(30) only.
    const double m = 1.5, g = 3.0, k = 500.0, c = std::cos(std::numbers::pi / 6.0);
    const auto r = solve_static(static_deck(
        turned_frame(12, 30.0) +
        "GRID,1,,0.,0.,0.,12\nCELAS2,7,500.,1,1\nCMASS2,8,1.5,1,1\nGRAV,1,0,3.,1.,0.,0.\n"));
    EXPECT_NEAR(disp(r, 1).d[0], m * g * c / k, 1e-12);
    EXPECT_NEAR(force_at(r.spc_forces, 1).f[1], 0.0, 1e-9);
    expect_grid_balance(r, 1e-9);
}

TEST(CdScalar, MassOnASkewAxisSeesTheSeriesCompliance) {
    // Unequal rods along X and Y hold the grid; the mass acts along an axis at
    // angle t only, so the massless cross direction relaxes and the flexibility
    // along the axis is cos^2(t) / kx + sin^2(t) / ky. Read in basic, the mass
    // would act along X with lambda = kx / m.
    const double t = std::numbers::pi / 6.0, m = 4.0;
    const double kx = axial_k(10.0), ky = axial_k(25.0);
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,1\n" + kRodProps +
        turned_frame(12, 30.0) +
        "GRID,1,,10.,0.,0.\nGRID,2,,0.,25.,0.\nGRID,3,,0.,0.,0.,12\n"
        "CROD,1,1,1,3\nCROD,2,1,2,3\nSPC1,1,123456,1,2\nCMASS2,8,4.,3,1\nENDDATA\n");
    const double flexibility = std::pow(std::cos(t), 2) / kx + std::pow(std::sin(t), 2) / ky;
    ASSERT_EQ(r.modes.size(), 1U);
    EXPECT_NEAR(r.modes[0].eigenvalue * m * flexibility, 1.0, 1e-9);
    // The shape is the static deflection under a force along the axis, K^-1 e1,
    // scaled so that m T1^2 = 1. In the frame its T2 is (1 / ky - 1 / kx) sin cos
    // for a unit force, and T1 is the flexibility.
    const auto shape = std::ranges::find(r.modes[0].shape, NodeId{3}, &NodeDisplacement::node);
    ASSERT_NE(shape, r.modes[0].shape.end());
    const double t2_over_t1 = (1.0 / ky - 1.0 / kx) * std::sin(t) * std::cos(t) / flexibility;
    EXPECT_EQ(shape->coord, CoordId{12});
    EXPECT_NEAR(std::abs(shape->d[0]), 1.0 / std::sqrt(m), 1e-9);
    EXPECT_NEAR(shape->d[1] / shape->d[0], t2_over_t1, 1e-9);
}

TEST(CdScalar, MassAlongAnObliqueAxisCountsOneFiniteMode) {
    // Asking for more modes than the mass has finite ones returns only those: the
    // two translations both carry mass on the diagonal, but its rank is one.
    const double kx = axial_k(10.0), ky = axial_k(25.0), m = 4.0, t = std::numbers::pi / 6.0;
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,2\n" + kRodProps +
        turned_frame(12, 30.0) +
        "GRID,1,,10.,0.,0.\nGRID,2,,0.,25.,0.\nGRID,3,,0.,0.,0.,12\n"
        "CROD,1,1,1,3\nCROD,2,1,2,3\nSPC1,1,123456,1,2\nCMASS2,8,4.,3,1\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 1U);
    const double flexibility = std::pow(std::cos(t), 2) / kx + std::pow(std::sin(t), 2) / ky;
    EXPECT_NEAR(r.modes[0].eigenvalue * m * flexibility, 1.0, 1e-9);
}

TEST(CdScalar, SemiDefiniteMassAmongOtherDofsGivesTheExactModes) {
    // Two copies of the skew oscillator at different masses and a rod with a mass
    // on a basic grid: three finite modes among 6 massed DOFs, with two requested
    // so the iterative path runs. The rod's kx / 16 is the lowest.
    const double kx = axial_k(10.0), ky = axial_k(25.0), t = std::numbers::pi / 6.0;
    const double flexibility = std::pow(std::cos(t), 2) / kx + std::pow(std::sin(t), 2) / ky;
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,2\n" + kRodProps +
        turned_frame(12, 30.0) +
        "GRID,1,,10.,0.,0.\nGRID,2,,0.,25.,0.\nGRID,3,,0.,0.,0.,12\n"
        "GRID,4,,110.,0.,0.\nGRID,5,,100.,25.,0.\nGRID,6,,100.,0.,0.,12\n"
        "GRID,7,,210.,0.,0.\nGRID,8,,200.,0.,0.\n"
        "CROD,1,1,1,3\nCROD,2,1,2,3\nCROD,3,1,4,6\nCROD,4,1,5,6\nCROD,5,1,7,8\n"
        "SPC1,1,123456,1,2,4,5,7\nCMASS2,8,4.,3,1\nCMASS2,9,1.,6,1\nCONM2,10,8,,16.\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 2U);
    EXPECT_NEAR(r.modes[0].eigenvalue / (kx / 16.0), 1.0, 1e-8);
    EXPECT_NEAR(r.modes[1].eigenvalue / (1.0 / (4.0 * flexibility)), 1.0, 1e-8);
}

TEST(CdScalar, TwoAxisOscillatorKeepsItsFrequencies) {
    // Springs and masses on T1 and T2 of the same frame decouple exactly.
    const double angle = 25.0;
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,2\n" +
        turned_frame(12, angle) +
        "GRID,1,,0.,0.,0.,12\nCELAS2,1,300.,1,1\nCELAS2,2,800.,1,2\n"
        "CMASS2,3,2.,1,1\nCMASS2,4,1.,1,2\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 2U);
    EXPECT_NEAR(r.modes[0].eigenvalue, 150.0, 1e-8);
    EXPECT_NEAR(r.modes[1].eigenvalue, 800.0, 1e-8);
}

// ── Mode shapes on a CD grid ──────────────────────────────────────────────────

TEST(CdModeShape, CylindricalFrameReportsARadialModeOnT1) {
    // A radial rod at 45 degrees with a mass on its free end: the motion is all T1
    // of the cylindrical frame, whereas basic would split it between X and Y.
    const double m = 2.5;
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,1\n" + kRodProps +
        "CORD2C,3,,0.,0.,0.,0.,0.,1.,\n,1.,0.,0.\n"
        "GRID,1,,20.,20.,0.\nGRID,2,,10.,10.,0.,3\nCROD,1,1,1,2\nSPC1,1,123456,1\n"
        "CONM2,5,2,,2.5\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 1U);
    const double L = 10.0 * std::numbers::sqrt2;
    EXPECT_NEAR(r.modes[0].eigenvalue / (axial_k(L) / m), 1.0, 1e-10);
    const auto shape = std::ranges::find(r.modes[0].shape, NodeId{2}, &NodeDisplacement::node);
    ASSERT_NE(shape, r.modes[0].shape.end());
    EXPECT_EQ(shape->coord, CoordId{3});
    EXPECT_NEAR(std::abs(shape->d[0]), 1.0 / std::sqrt(m), 1e-10);
    EXPECT_NEAR(shape->d[1], 0.0, 1e-10);
    EXPECT_NEAR(shape->d[2], 0.0, 1e-10);
}

TEST(CdModeShape, RotationsAreReportedInTheFrameToo) {
    // A rod along basic Y twists about R1 of a 90-degree frame, against a scalar
    // inertia on that component.
    const double inertia = 3.0, kt = torsional_k(10.0);
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,1\n" + kRodProps +
        turned_frame(12, 90.0) +
        "GRID,1,,0.,0.,0.\nGRID,2,,0.,10.,0.,12\nCROD,1,2,1,2\nSPC1,1,123456,1\n"
        "CMASS2,5,3.,2,4\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 1U);
    EXPECT_NEAR(r.modes[0].eigenvalue / (kt / inertia), 1.0, 1e-10);
    const auto shape = std::ranges::find(r.modes[0].shape, NodeId{2}, &NodeDisplacement::node);
    ASSERT_NE(shape, r.modes[0].shape.end());
    EXPECT_NEAR(std::abs(shape->d[3]), 1.0 / std::sqrt(inertia), 1e-10);
    EXPECT_NEAR(shape->d[4], 0.0, 1e-10);
}

TEST(CdModeShape, GridsWithoutAFrameStayInBasic) {
    // Grid 4 has no CD: its T1 is the common motion of the tie, as is grid 2's.
    const double m = 2.5;
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMPC = 7\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,1\n" + kRodProps +
        turned_frame(12, 90.0) +
        "GRID,1,,0.,0.,0.\nGRID,2,,0.,10.,0.,12\nGRID,3,,20.,0.,0.\nGRID,4,,40.,0.,0.\n"
        "CROD,1,1,1,2\nCROD,2,1,3,4\nSPC1,1,123456,1,3\nMPC,7,2,1,1.,4,1,-1.\n"
        "CONM2,5,2,,2.5\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 1U);
    const auto grid = [&](const int id) {
        return *std::ranges::find(r.modes[0].shape, NodeId{id}, &NodeDisplacement::node);
    };
    EXPECT_EQ(grid(4).coord, CoordId{0});
    EXPECT_NEAR(std::abs(grid(2).d[0]), 1.0 / std::sqrt(m), 1e-10);
    EXPECT_NEAR(grid(2).d[0], grid(4).d[0], 1e-10);
    EXPECT_NEAR(grid(2).d[1], 0.0, 1e-10);
}

TEST(CdModeShape, NormMaxIsUnitInTheReportedFrame) {
    // The skew oscillator of MassOnASkewAxisSeesTheSeriesCompliance. Its largest
    // basic component is Y, but its largest reported component must be one: T1.
    const double t = std::numbers::pi / 6.0;
    const double kx = axial_k(10.0), ky = axial_k(25.0);
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,1,,,,MAX\n" + kRodProps +
        turned_frame(12, 30.0) +
        "GRID,1,,10.,0.,0.\nGRID,2,,0.,25.,0.\nGRID,3,,0.,0.,0.,12\n"
        "CROD,1,1,1,3\nCROD,2,1,2,3\nSPC1,1,123456,1,2\nCMASS2,8,4.,3,1\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 1U);
    const double flexibility = std::pow(std::cos(t), 2) / kx + std::pow(std::sin(t), 2) / ky;
    const auto shape = std::ranges::find(r.modes[0].shape, NodeId{3}, &NodeDisplacement::node);
    ASSERT_NE(shape, r.modes[0].shape.end());
    EXPECT_NEAR(std::abs(shape->d[0]), 1.0, 1e-12);
    EXPECT_NEAR(shape->d[1] / shape->d[0],
                (1.0 / ky - 1.0 / kx) * std::sin(t) * std::cos(t) / flexibility, 1e-9);
    // The mass acts on T1 alone, and T1 is one.
    EXPECT_NEAR(r.modes[0].gen_mass, 4.0, 1e-9);
}

// ── Generalized mass ──────────────────────────────────────────────────────────

TEST(GeneralizedMass, NormMaxIsThatOfTheReportedShape) {
    // Two equal masses and springs from ground. With g the golden ratio the shapes
    // are (1 / g, 1) at lambda = (k / m) (3 - sqrt 5) / 2 and (1, -1 / g) at
    // (k / m) (3 + sqrt 5) / 2, each with generalized mass m (1 + 1 / g^2). The
    // generalized stiffness phi^T K phi, worked from the springs, must equal
    // lambda times it.
    const double k = 4.0, m = 2.0, g = (1.0 + std::sqrt(5.0)) / 2.0;
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,2,,,,MAX\nSPOINT,1,2\n"
        "CELAS2,1,4.,1\nCELAS4,2,4.,1,2\nCMASS2,3,2.,1\nCMASS2,4,2.,2\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 2U);
    const double expected = m * (1.0 + 1.0 / (g * g));
    EXPECT_NEAR(r.modes[0].gen_mass, expected, 1e-10);
    EXPECT_NEAR(r.modes[1].gen_mass, expected, 1e-10);
    const auto stiffness = [&](const double a, const double b) {
        return k * a * a + k * (b - a) * (b - a);
    };
    EXPECT_NEAR(r.modes[0].eigenvalue * r.modes[0].gen_mass, stiffness(1.0 / g, 1.0), 1e-9);
    EXPECT_NEAR(r.modes[1].eigenvalue * r.modes[1].gen_mass, stiffness(1.0, -1.0 / g), 1e-9);
}

TEST(GeneralizedMass, NormMaxCountsTheMassOfDependentDofs) {
    // Two masses tied by an MPC move as one, so both count at a shape of one.
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMPC = 7\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,1,,,,MAX\nSPOINT,1,2\n"
        "CELAS2,1,60.,1\nCMASS2,3,2.,1\nCMASS2,4,3.,2\nMPC,7,2,,1.,1,,-1.\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 1U);
    EXPECT_NEAR(r.modes[0].eigenvalue, 12.0, 1e-9);
    EXPECT_NEAR(r.modes[0].gen_mass, 5.0, 1e-10);
}

TEST(GeneralizedMass, NormMassIsOne) {
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 9\nBEGIN BULK\nEIGRL,9,,,1\nSPOINT,1\n"
        "CELAS2,1,60.,1\nCMASS2,3,5.,1\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 1U);
    EXPECT_DOUBLE_EQ(r.modes[0].gen_mass, 1.0);
    EXPECT_NEAR(std::abs(r.modes[0].shape.at(0).d[0]), 1.0 / std::sqrt(5.0), 1e-12);
}
