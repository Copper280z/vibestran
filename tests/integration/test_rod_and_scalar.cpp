// tests/integration/test_rod_and_scalar.cpp
// CROD and CONROD elements and scalar points, checked against closed-form solutions: truss
// statics, thermal stress, torsion, axial and torsional vibration, spring-mass
// chains, MPC equations through a scalar point and heat flow to a scalar ambient.

#include "core/quality_checks.hpp"
#include "elements/element_factory.hpp"
#include "elements/line_elements.hpp"
#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/analysis_support.hpp"
#include "solver/eigensolver_backend.hpp"
#include "solver/heat_transfer_steady.hpp"
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

SubCaseResults solve_heat(const std::string& deck) {
    Model model = BdfParser::parse_string(deck);
    HeatTransferSteadySolver solver(std::make_unique<EigenSolverBackend>());
    return solver.solve(model).subcases.at(0);
}

const NodeDisplacement& disp(const SubCaseResults& r, int node) {
    const auto it = std::ranges::find(r.displacements, NodeId{node}, &NodeDisplacement::node);
    if (it == r.displacements.end())
        throw std::runtime_error(std::format("no displacement for point {}", node));
    return *it;
}

const SpcForce* spc_at(const SubCaseResults& r, int node) {
    const auto it = std::ranges::find(r.spc_forces, NodeId{node}, &SpcForce::node);
    return it == r.spc_forces.end() ? nullptr : &*it;
}

const RodResult& rod(const SubCaseResults& r, int eid) {
    const auto it = std::ranges::find(r.rod_results, ElementId{eid}, &RodResult::eid);
    if (it == r.rod_results.end())
        throw std::runtime_error(std::format("no result for rod {}", eid));
    return *it;
}

double temperature(const SubCaseResults& r, int node) {
    const auto it = std::ranges::find(r.temperatures, NodeId{node}, &NodeTemperature::node);
    if (it == r.temperatures.end())
        throw std::runtime_error(std::format("no temperature for point {}", node));
    return it->temperature;
}

// E = 1e7, nu = 0.25 so G = 4e6; rho = 0.1, alpha = 1e-5.
// PROD 1: A = 2, J = 1.5, C = 0.5, NSM = 0.3.
const std::string kRodProps =
    "MAT1,1,1.0E7,4.0E6,0.25,0.1,1.0E-5\n"
    "PROD,1,1,2.0,1.5,0.5,0.3\n";

constexpr double kE = 1.0e7, kG = 4.0e6, kRho = 0.1, kAlpha = 1.0e-5;
constexpr double kA = 2.0, kJ = 1.5, kC = 0.5, kNsm = 0.3;

std::string static_deck(const std::string& bulk, const std::string& cases = "") {
    return "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nDISP = ALL\nSPCF = ALL\nMPCF = ALL\n"
           "GPFORCE = ALL\nFORCE = ALL\nSTRESS = ALL\nSTRAIN = ALL\n" + cases +
           "BEGIN BULK\n" + kRodProps + bulk + "ENDDATA\n";
}

} // namespace

// ── The element ───────────────────────────────────────────────────────────────

TEST(CRodElement, SkewRodStiffensOnlyItsAxisAndTwist) {
    // Any motion's energy must be EA/L (axial stretch)^2 + GJ/L (twist)^2, so
    // transverse motion and rigid rotation of the rod cost nothing.
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\n" + kRodProps +
        "GRID,1,,1.,2.,3.\nGRID,2,,4.,6.,15.\nCROD,7,1,1,2\nENDDATA\n");
    const CRodElement element(model.elements.at(0), model);
    const LocalKe k = element.stiffness_matrix();
    ASSERT_EQ(k.rows(), 12);
    EXPECT_LT((k - k.transpose()).norm(), 1e-9 * k.norm());

    const Eigen::Vector3d d(3, 4, 12); // length 13
    const double L = 13.0;
    const Eigen::Vector3d n = d / L;
    Eigen::Vector3d t = n.cross(Eigen::Vector3d::UnitX()).normalized();

    Eigen::VectorXd stretch = Eigen::VectorXd::Zero(12);
    stretch.segment<3>(6) = n;
    const Eigen::VectorXd f = k * stretch;
    EXPECT_NEAR(n.dot(f.segment<3>(6)), kE * kA / L, 1e-6);
    EXPECT_NEAR(n.dot(f.segment<3>(0)), -kE * kA / L, 1e-6);

    Eigen::VectorXd transverse = Eigen::VectorXd::Zero(12);
    transverse.segment<3>(6) = t;
    EXPECT_LT((k * transverse).norm(), 1e-6);

    Eigen::VectorXd twist = Eigen::VectorXd::Zero(12);
    twist.segment<3>(9) = n;
    EXPECT_NEAR(twist.dot(k * twist), kG * kJ / L, 1e-6);

    Eigen::VectorXd bend = Eigen::VectorXd::Zero(12);
    bend.segment<3>(9) = t;
    EXPECT_LT((k * bend).norm(), 1e-6);

    // Small rigid rotation about basic Z through grid 1.
    Eigen::VectorXd rigid = Eigen::VectorXd::Zero(12);
    rigid.segment<3>(6) = Eigen::Vector3d::UnitZ().cross(d);
    rigid(5) = rigid(11) = 1.0;
    EXPECT_LT((k * rigid).norm(), 1e-6 * k.norm());
}

TEST(CRodElement, MassMatricesCarryTheWholeMassInEveryTranslation) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\n" + kRodProps +
        "GRID,1,,0.,0.,0.\nGRID,2,,3.,4.,0.\nCROD,7,1,1,2\nENDDATA\n");
    const CRodElement element(model.elements.at(0), model);
    const double total = (kRho * kA + kNsm) * 5.0;
    for (const LocalKe& m : {LocalKe(element.mass_matrix()), LocalKe(element.lumped_mass_matrix())}) {
        for (int c = 0; c < 3; ++c) {
            Eigen::VectorXd rigid = Eigen::VectorXd::Zero(12);
            rigid(c) = rigid(c + 6) = 1.0;
            EXPECT_NEAR(rigid.dot(m * rigid), total, 1e-12) << "direction " << c;
        }
    }
    const LocalKe consistent = element.mass_matrix();
    EXPECT_NEAR(consistent(0, 0), total / 3.0, 1e-12);
    EXPECT_NEAR(consistent(0, 6), total / 6.0, 1e-12);
    // Rotary inertia about the axis only, and none when lumped.
    const Eigen::Vector3d n(0.6, 0.8, 0.0);
    Eigen::VectorXd spin = Eigen::VectorXd::Zero(12);
    spin.segment<3>(3) = n;
    spin.segment<3>(9) = n;
    EXPECT_NEAR(spin.dot(consistent * spin), kRho * kJ * 5.0, 1e-12);
    const LocalKe lumped = element.lumped_mass_matrix();
    EXPECT_EQ(spin.dot(lumped * spin), 0.0);
}

TEST(CRodElement, RejectsZeroLengthAndWrongProperty) {
    Model same_point = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\n" + kRodProps +
        "GRID,1,,1.,1.,1.\nGRID,2,,1.,1.,1.\nCROD,7,1,1,2\nENDDATA\n");
    EXPECT_THROW(CRodElement(same_point.elements.at(0), same_point), SolverError);
    Model wrong_card = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nMAT1,1,1.0E7,,0.3\nPBAR,1,1,1.,1.,1.,1.\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nCROD,7,1,1,2\nENDDATA\n");
    EXPECT_THROW(CRodElement(wrong_card.elements.at(0), wrong_card), SolverError);
}

// ── Statics ───────────────────────────────────────────────────────────────────

TEST(CRodStatics, DeterminateTrussMatchesTheMethodOfJoints) {
    // Right triangle: 1 pinned, 2 on a roller above 1's level, load +X at 3.
    // Joints give rod 1 no force, rod 2 -P and diagonal rod 3 +P*sqrt(2); virtual
    // work gives the displacements of grid 3.
    const double P = 1000.0, b = 100.0, EA = kE * kA, diagonal = b * std::numbers::sqrt2;
    const auto r = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,100.,0.,0.\nGRID,3,,100.,100.,0.\n"
        "CROD,1,1,1,2\nCROD,2,1,2,3\nCROD,3,1,1,3\n"
        "SPC1,1,123,1\nSPC1,1,23,2\nSPC1,1,3,3\nFORCE,1,3,0,1000.,1.,0.,0.\n"));
    EXPECT_NEAR(rod(r, 1).axial_force, 0.0, 1e-7);
    EXPECT_NEAR(rod(r, 2).axial_force, -P, 1e-7);
    EXPECT_NEAR(rod(r, 3).axial_force, P * std::numbers::sqrt2, 1e-7);
    EXPECT_NEAR(rod(r, 3).axial_stress, P * std::numbers::sqrt2 / kA, 1e-7);
    EXPECT_NEAR(rod(r, 2).axial_strain, -P / EA, 1e-14);

    const double v = -P * b / EA;
    const double u = -v + 2.0 * P * diagonal / EA;
    EXPECT_NEAR(disp(r, 3).d[0], u, 1e-12);
    EXPECT_NEAR(disp(r, 3).d[1], v, 1e-12);

    ASSERT_NE(spc_at(r, 1), nullptr);
    ASSERT_NE(spc_at(r, 2), nullptr);
    EXPECT_NEAR(spc_at(r, 1)->f[0], -P, 1e-7);
    EXPECT_NEAR(spc_at(r, 1)->f[1], -P, 1e-7);
    EXPECT_NEAR(spc_at(r, 2)->f[1], P, 1e-7);
    for (const auto& grid : r.gp_forces)
        for (const double total : grid.totals)
            EXPECT_NEAR(total, 0.0, 1e-7) << "grid " << grid.node.value;
}

TEST(CRodStatics, RestrainedRodCarriesThermalStressAndFreeRodDoesNot) {
    const double dT = 50.0;
    const auto clamped = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC1,1,123,1,2\n"
        "TEMP,1,1,50.\nTEMP,1,2,50.\n", "TEMP(LOAD) = 1\n"));
    EXPECT_NEAR(rod(clamped, 1).axial_stress, -kE * kAlpha * dT, 1e-9);
    EXPECT_NEAR(rod(clamped, 1).axial_force, -kE * kA * kAlpha * dT, 1e-9);
    EXPECT_NEAR(rod(clamped, 1).axial_strain, -kAlpha * dT, 1e-15);
    // The rod pushes the support at grid 2 outwards, so the support pushes back.
    ASSERT_NE(spc_at(clamped, 2), nullptr);
    EXPECT_NEAR(spc_at(clamped, 2)->f[0], -kE * kA * kAlpha * dT, 1e-8);

    // A linear temperature acts through its average.
    const auto free_end = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC1,1,123,1\n"
        "TEMP,1,1,0.\nTEMP,1,2,100.\n", "TEMP(LOAD) = 1\n"));
    EXPECT_NEAR(disp(free_end, 2).d[0], kAlpha * dT * 10.0, 1e-14);
    EXPECT_NEAR(rod(free_end, 1).axial_stress, 0.0, 1e-8);
}

TEST(CRodStatics, TorsionFollowsTheSkewAxisWithStressStrainAndSign) {
    // Moment about the axis of a 3-4-5 rod. The rotation is T L / (G J) along
    // the axis, whatever components AUTOSPC removes for the bending directions.
    const double T = 600.0, L = 5.0;
    const auto r = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,3.,4.,0.\nCROD,1,1,1,2\nSPC1,1,123456,1\n"
        "SPC1,1,123,2\nMOMENT,1,2,0,600.,0.6,0.8,0.\n"));
    const auto& d = disp(r, 2).d;
    EXPECT_NEAR(0.6 * d[3] + 0.8 * d[4], T * L / (kG * kJ), 1e-12);
    EXPECT_NEAR(rod(r, 1).torque, T, 1e-8);
    EXPECT_NEAR(rod(r, 1).axial_force, 0.0, 1e-8);
    EXPECT_NEAR(rod(r, 1).torsional_stress, kC * T / kJ, 1e-8);
    EXPECT_NEAR(rod(r, 1).torsional_strain, kC * T / kJ / kG, 1e-14);
}

TEST(CRodStatics, MarginsOfSafetyUseTheAllowableOfTheStressSign) {
    const std::string grids =
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,20.,0.,0.\nCROD,1,2,1,2\nCROD,2,2,2,3\n"
        "MAT1,2,1.0E7,4.0E6,,,,,,+\n+,1000.,800.,300.\nPROD,2,2,1.0,1.0,1.0\n"
        "SPC1,1,123,1,3\nSPC1,1,23,2\nFORCE,1,2,0,100.,1.,0.,0.\n";
    // Load P at the midpoint of a clamped span splits equally: P/2 each.
    const auto r = solve_static(static_deck(grids));
    EXPECT_NEAR(rod(r, 1).axial_stress, 50.0, 1e-9);
    EXPECT_NEAR(rod(r, 2).axial_stress, -50.0, 1e-9);
    EXPECT_NEAR(rod(r, 1).axial_margin.value(), 1000.0 / 50.0 - 1.0, 1e-9);
    EXPECT_NEAR(rod(r, 2).axial_margin.value(), 800.0 / 50.0 - 1.0, 1e-9);
    EXPECT_FALSE(rod(r, 1).torsional_margin.has_value());

    const auto no_allowable = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC1,1,123,1\nSPC1,1,23,2\n"
        "FORCE,1,2,0,100.,1.,0.,0.\n"));
    EXPECT_FALSE(rod(no_allowable, 1).axial_margin.has_value());
}

TEST(CRodStatics, RodSpringsInParallelWithACantileverBar) {
    // A rod from the tip of a CBAR to ground adds EA/L in parallel with 3EI/L^3,
    // and AUTOSPC must leave the bar's grids alone: the tip rotation is that of
    // the bar's own share of the load.
    const double L = 100.0, EI = 1.0e7 * 8.0, rod_k = kE * kA / 50.0, P = 1000.0;
    const auto r = solve_static(static_deck(
        "PBAR,3,1,4.,8.,8.,8.\nGRID,1,,0.,0.,0.\nGRID,2,,100.,0.,0.\nGRID,3,,100.,50.,0.\n"
        "CBAR,1,3,1,2,0.,0.,1.\nCROD,2,1,2,3\nSPC1,1,123456,1\nSPC1,1,123,3\n"
        "FORCE,1,2,0,1000.,0.,1.,0.\n"));
    const double bar_k = 3.0 * EI / (L * L * L);
    const double tip = P / (bar_k + rod_k);
    EXPECT_NEAR(disp(r, 2).d[1], tip, 1e-9);
    EXPECT_NEAR(disp(r, 2).d[5], bar_k * tip * L * L / (2.0 * EI), 1e-9);
    EXPECT_NEAR(rod(r, 2).axial_force, -rod_k * tip, 1e-6);
}

TEST(CRodStatics, GravityReactsTheWholeRodAndNonStructuralMass) {
    const double g = 386.0, total = (kRho * kA + kNsm) * 30.0;
    for (const std::string param : {std::string(), std::string("PARAM,COUPMASS,-1\n")}) {
        const auto r = solve_static(static_deck(
            param + "GRID,1,,0.,0.,0.\nGRID,2,,0.,0.,10.\nGRID,3,,0.,0.,30.\nCROD,1,1,1,2\nCROD,2,1,2,3\n"
                    "SPC1,1,123,1\nSPC1,1,12,2,3\nGRAV,1,0,386.,0.,0.,-1.\n"));
        ASSERT_NE(spc_at(r, 1), nullptr);
        EXPECT_NEAR(spc_at(r, 1)->f[2], total * g, 1e-7);
    }
}

TEST(CRodStatics, CdFramesTakeTheSameLoad) {
    // Grid 2's CD system has X along the rod, 30 degrees above basic X. The
    // singular directions are then exactly its T2 and T3, so the displacement
    // along the rod is F L / EA with nothing else.
    const double F = 500.0, L = 20.0, c = std::cos(std::numbers::pi / 6.0), s = 0.5;
    std::ostringstream deck;
    deck << std::setprecision(17);
    deck << "CORD2R,5,,0.,0.,0.,0.,0.,1.,\n," << c << "," << s << ",0.\n"
         << "GRID,1,,0.,0.,0.\nGRID,2,," << L * c << "," << L * s << ",0.,5\n"
         << "CROD,1,1,1,2\nSPC1,1,123,1\nFORCE,1,2,5,500.,1.,0.,0.\n";
    const auto r = solve_static(static_deck(deck.str()));
    EXPECT_NEAR(disp(r, 2).d[0], F * L / (kE * kA), 1e-11);
    EXPECT_NEAR(disp(r, 2).d[1], 0.0, 1e-11);
    EXPECT_NEAR(rod(r, 1).axial_force, F, 1e-7);
}

// ── AUTOSPC ───────────────────────────────────────────────────────────────────

TEST(RodAutoSpc, ConstrainsExactlyTheDirectionsARodLeavesFree) {
    // X-aligned chain with J = 0 and no constraints beyond the anchor: T2, T3 and
    // all rotations are singular at grids 2 and 3, and T1 stays free.
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nSPC = 1\nBEGIN BULK\nMAT1,1,1.0E7,,0.3\nPROD,1,1,1.0\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,20.,0.,0.\nCROD,1,1,1,2\nCROD,2,1,2,3\n"
        "SPC1,1,123,1\nENDDATA\n");
    const SubCase sc = model.analysis.subcases.at(0);
    const DofMap dofs = build_analysis_dof_map(model, sc);
    EXPECT_EQ(dofs.num_free_dofs(), 2);
    EXPECT_TRUE(dofs.is_free(NodeId{2}, 0));
    EXPECT_TRUE(dofs.is_free(NodeId{3}, 0));
    for (const int c : {1, 2, 3, 4, 5})
        EXPECT_FALSE(dofs.is_free(NodeId{2}, c)) << "component " << c + 1;
}

TEST(RodAutoSpc, TorsionKeepsTheAxisRotationFree) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nSPC = 1\nBEGIN BULK\n" + kRodProps +
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC1,1,123456,1\nENDDATA\n");
    const DofMap dofs = build_analysis_dof_map(model, model.analysis.subcases.at(0));
    EXPECT_TRUE(dofs.is_free(NodeId{2}, 0));
    EXPECT_TRUE(dofs.is_free(NodeId{2}, 3));
    EXPECT_EQ(dofs.num_free_dofs(), 2);
}

TEST(RodAutoSpc, GridsSpanningSpaceKeepEveryTranslation) {
    // A statically determinate space truss apex: three rods in general
    // directions make the 3x3 translation block full rank, though no two are
    // aligned with a component.
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nSPC = 1\nBEGIN BULK\nMAT1,1,1.0E7,,0.3\nPROD,1,1,1.0\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,0.,10.,0.\nGRID,4,,3.,3.,8.\n"
        "CROD,1,1,1,4\nCROD,2,1,2,4\nCROD,3,1,3,4\nSPC1,1,123,1,2,3\nENDDATA\n");
    const DofMap dofs = build_analysis_dof_map(model, model.analysis.subcases.at(0));
    EXPECT_EQ(dofs.num_free_dofs(), 3);
    for (const int c : {0, 1, 2})
        EXPECT_TRUE(dofs.is_free(NodeId{4}, c));
}

TEST(RodAutoSpc, SkewChainsSolveWithTheAxialResultExact) {
    // Rods along (1,1,1): rank one at each free grid, in no component's
    // direction. The axial force is determinate whichever components go.
    const auto r = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,10.,10.\nGRID,3,,20.,20.,20.\nCROD,1,1,1,2\nCROD,2,1,2,3\n"
        "SPC1,1,123,1\nFORCE,1,3,0,300.,1.,1.,1.\n"));
    const double force = 300.0 * std::sqrt(3.0);
    EXPECT_NEAR(rod(r, 1).axial_force, force, 1e-6);
    EXPECT_NEAR(rod(r, 2).axial_force, force, 1e-6);
}

TEST(RodAutoSpc, LoadOnASingularDirectionAppearsAsASpcForce) {
    // Nothing resists +Y at grid 2, so AUTOSPC takes the load. It must be visible
    // in the reactions and the grid balance, not disappear.
    const auto r = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC1,1,123,1\n"
        "FORCE,1,2,0,75.,0.,1.,0.\n"));
    ASSERT_NE(spc_at(r, 2), nullptr);
    EXPECT_NEAR(spc_at(r, 2)->f[1], -75.0, 1e-9);
    EXPECT_EQ(disp(r, 2).d[1], 0.0);
    for (const auto& grid : r.gp_forces)
        for (const double total : grid.totals)
            EXPECT_NEAR(total, 0.0, 1e-9);
}

TEST(RodAutoSpc, UnreactedGridsDoNotFillTheSpcForceTable) {
    const auto r = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC1,1,123,1\n"
        "FORCE,1,2,0,75.,1.,0.,0.\n"));
    EXPECT_NE(spc_at(r, 1), nullptr);
    EXPECT_EQ(spc_at(r, 2), nullptr);
}

TEST(RodAutoSpc, ParamNoLeavesTheSingularityToTheSolver) {
    // Nothing resists the load at grid 2 once AUTOSPC is off, so it is unbalanced.
    const std::string bulk =
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC1,1,123456,1\nFORCE,1,2,0,75.,0.,1.,0.\n";
    Model on = BdfParser::parse_string(static_deck(bulk));
    Model off = BdfParser::parse_string(static_deck("PARAM,AUTOSPC,NO\n" + bulk));
    EXPECT_EQ(build_analysis_dof_map(on, on.analysis.subcases.at(0)).num_free_dofs(), 2);
    EXPECT_EQ(build_analysis_dof_map(off, off.analysis.subcases.at(0)).num_free_dofs(), 6);
    LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
    const SolverResults results = solver.solve(off);
    EXPECT_FALSE(results.summary.notices.empty());
}

TEST(RodAutoSpc, MpcComponentsAreLeftToTheMpc) {
    // Grid 2's T2 is tied to grid 3's T2 by an MPC and a spring holds grid 3, so
    // the load at grid 2 reaches the spring: AUTOSPC on that T2 would lose it.
    const auto r = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,20.,5.,0.\nCROD,1,1,1,2\n"
        "CELAS2,5,1000.,3,2\nSPC1,1,123,1\nSPC1,1,13,3\nMPC,4,2,2,1.,3,2,-1.\n"
        "FORCE,1,2,0,50.,0.,1.,0.\n", "MPC = 4\n"));
    EXPECT_NEAR(disp(r, 2).d[1], 0.05, 1e-10);
    EXPECT_NEAR(disp(r, 3).d[1], 0.05, 1e-10);
}

// ── Vibration ─────────────────────────────────────────────────────────────────

TEST(CRodModal, SingleElementFrequenciesAreExact) {
    // Fixed-free, one element: K = EA/L, M = mL/3 consistent or mL/2 lumped.
    const std::string bulk =
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 1\nBEGIN BULK\nEIGRL,1,,,1\n"
        "MAT1,1,1.0E7,,0.3,0.1\nPROD,1,1,2.0\nGRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\n"
        "CROD,1,1,1,2\nSPC1,1,123,1\n";
    const double k = 1.0e7 * 2.0 / 10.0, m = 0.1 * 2.0 * 10.0;
    const auto consistent = solve_modal(bulk + "ENDDATA\n");
    EXPECT_NEAR(consistent.modes.at(0).eigenvalue, k / (m / 3.0), 1e-5 * k / m);
    const auto lumped = solve_modal(bulk + "PARAM,COUPMASS,-1\nENDDATA\n");
    EXPECT_NEAR(lumped.modes.at(0).eigenvalue, k / (m / 2.0), 1e-5 * k / m);
}

TEST(CRodModal, AxialAndTorsionalModesConvergeToTheContinuum) {
    // Fixed-free bar: omega_n = (2n - 1) pi / (2 L) * sqrt(E / rho), and with G in
    // place of E for torsion of a circular section, where J is the polar inertia.
    const int n = 40;
    const double L = 100.0;
    std::ostringstream deck;
    deck << "SOL 103\nCEND\nSPC = 1\nMETHOD = 1\nBEGIN BULK\nEIGRL,1,,,4\n"
         << "MAT1,1,1.0E7,4.0E6,0.25,0.1\nPROD,1,1,2.0,1.5\n";
    for (int i = 0; i <= n; ++i)
        deck << "GRID," << i + 1 << ",," << L * i / n << ",0.,0.\n";
    for (int i = 0; i < n; ++i)
        deck << "CROD," << i + 1 << ",1," << i + 1 << "," << i + 2 << "\n";
    deck << "SPC1,1,123456,1\nENDDATA\n";
    const auto r = solve_modal(deck.str());
    const double base = std::numbers::pi / (2.0 * L);
    std::vector<double> expected;
    for (const int k : {1, 3})
        for (const double modulus : {1.0e7, 4.0e6})
            expected.push_back(std::pow(k * base, 2) * modulus / 0.1);
    std::ranges::sort(expected);
    ASSERT_GE(r.modes.size(), 3U);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(r.modes[i].eigenvalue / expected[i], 1.0, 2e-3) << "mode " << i + 1;
}

// ── Heat transfer ─────────────────────────────────────────────────────────────

TEST(CRodHeat, ConductionAndSloadFollowFouriersLaw) {
    // Rod with k = 50, A = 2, L = 10: a power q into the far end drops q L / (k A)
    // across it from the fixed grid, and the flux is k dT / L = q / A.
    const auto r = solve_heat(
        "SOL 153\nCEND\nSPC = 1\nLOAD = 1\nBEGIN BULK\nMAT4,1,50.\nPROD,1,1,2.0\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC,1,1,0,20.\nSLOAD,1,2,100.\nENDDATA\n");
    EXPECT_NEAR(temperature(r, 2), 20.0 + 100.0 * 10.0 / (50.0 * 2.0), 1e-9);
    ASSERT_EQ(r.heat_fluxes.size(), 1U);
    EXPECT_EQ(r.heat_fluxes[0].etype, ElementType::CROD);
    EXPECT_NEAR(std::abs(r.heat_fluxes[0].q[0]), 100.0 / 2.0, 1e-9);
}

TEST(ScalarPointHeat, AmbientScalarPointSetsTheConvectionSink) {
    // Rod from 100 degrees to grid 2, which convects to scalar point 50 at 20
    // degrees through h A = 4 * 0.5: series resistance gives grid 2.
    const auto r = solve_heat(
        "SOL 153\nCEND\nSPC = 1\nBEGIN BULK\nMAT4,1,50.\nMAT4,2,4.\nPROD,1,1,2.0\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nSPOINT,50\nCROD,1,1,1,2\n"
        "PHBDY,5,2,0.5\nCHBDY,9,5,POINT,2,,,,,+\n+,50\nSPC,1,1,0,100.\nSPC,1,50,,20.\nENDDATA\n");
    const double rod = 50.0 * 2.0 / 10.0, film = 4.0 * 0.5;
    EXPECT_NEAR(temperature(r, 2), (rod * 100.0 + film * 20.0) / (rod + film), 1e-9);
    EXPECT_NEAR(temperature(r, 50), 20.0, 1e-12);
}

// ── Scalar point cards ────────────────────────────────────────────────────────

TEST(ScalarPointParsing, DefinesPointsByListThruRepeatAndConnection) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nSPOINT,1,2,2\nSPOINT,10,THRU,12\n"
        "GRID,5,,1.,2.,3.\nCELAS2,1,10.,20,0,21\nCELAS4,2,10.,30,31\nCELAS2,3,10.,5,1,40\nENDDATA\n");
    for (const int id : {1, 2, 10, 11, 12, 20, 21, 30, 31, 40})
        EXPECT_TRUE(model.node(NodeId{id}).scalar) << id;
    EXPECT_FALSE(model.node(NodeId{5}).scalar);
    EXPECT_EQ(model.scalar_point_count(), 10U);
    const auto& mixed = model.elements.at(2);
    EXPECT_EQ(mixed.components[0], 1);
    EXPECT_EQ(mixed.components[1], 1);
}

TEST(ScalarPointParsing, GroundedScalarElementsAcceptEitherBlankEnd) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nCELAS2,1,10.,,,7\nCELAS4,2,10.,,8\nCMASS4,3,2.,9\nENDDATA\n");
    for (const auto& element : model.elements) {
        EXPECT_EQ(element.nodes.size(), 1U);
        EXPECT_EQ(element.components[0], 1);
    }
}

TEST(ScalarPointParsing, CardOrderDoesNotMatter) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nSPC,1,4,,0.5\nMPC,2,4,,1.,5,,-1.\nSPOINT,4,5\nENDDATA\n");
    EXPECT_EQ(model.spcs.at(0).dofs.mask, 1);
    EXPECT_EQ(model.mpcs.at(0).terms.at(0).dof, 1);
    EXPECT_EQ(model.mpcs.at(0).terms.at(1).dof, 1);
    EXPECT_EQ(model.mpcs.at(0).terms.at(1).coeff, -1.0);
}

TEST(ScalarPointParsing, MpcContinuationsKeepBlankComponents) {
    // Scalar terms have a blank component, and fixed-field continuations have a
    // blank alignment field before the third term.
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nSPOINT,1,2,3\n"
        "MPC     1       1               1.0     2               2.0             +\n"
        "+               3               3.0\nENDDATA\n");
    const auto& terms = model.mpcs.at(0).terms;
    ASSERT_EQ(terms.size(), 3U);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(terms[size_t(i)].node, NodeId{i + 1});
        EXPECT_EQ(terms[size_t(i)].dof, 1);
        EXPECT_EQ(terms[size_t(i)].coeff, double(i + 1));
    }
}

TEST(ScalarPointParsing, RejectsMisuse) {
    const auto parse = [](const std::string& bulk) {
        return BdfParser::parse_string("SOL 101\nCEND\nBEGIN BULK\n" + bulk + "ENDDATA\n");
    };
    // An ID can name a grid or a scalar point, not both.
    EXPECT_THROW(parse("GRID,1,,0.,0.,0.\nSPOINT,1\n"), ParseError);
    // A blank component is valid only for a scalar point.
    EXPECT_THROW(parse("GRID,1,,0.,0.,0.\nCELAS2,1,10.,1\n"), ParseError);
    EXPECT_THROW(parse("GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nCELAS4,1,10.,1,2\n"), ParseError);
    EXPECT_THROW(parse("CELAS2,1,10.\n"), ParseError);
    EXPECT_THROW(parse("PROD,1,1,-2.\n"), ParseError);

    const auto validate = [&](const std::string& bulk) { parse(bulk).validate(); };
    EXPECT_THROW(validate("SPOINT,1\nSPC,1,1,2\n"), SolverError);
    EXPECT_THROW(validate("SPOINT,1\nMPC,1,1,2,1.\n"), SolverError);
    EXPECT_THROW(validate("SPOINT,1\nCELAS2,1,10.,1,3\n"), SolverError);
    EXPECT_THROW(validate("SPOINT,1\nFORCE,1,1,0,1.,1.,0.,0.\n"), SolverError);
    EXPECT_THROW(validate("SPOINT,1\nMOMENT,1,1,0,1.,1.,0.,0.\n"), SolverError);
    EXPECT_THROW(validate("SPOINT,1\nGRID,2,,1.,0.,0.\nMAT1,1,1.,,0.3\nPROD,1,1,1.\nCROD,1,1,1,2\n"), SolverError);
    EXPECT_THROW(validate("SPOINT,1\nGRID,2,,1.,0.,0.\nRBE2,1,2,123,1\n"), SolverError);
    EXPECT_THROW(validate("SPOINT,1\nACCEL1,1,0,1.,1.,0.,0.\n,1\n"), SolverError);
    // A grid with no such component or an unknown point in an MPC.
    EXPECT_THROW(validate("GRID,1,,0.,0.,0.\nMPC,1,1,,1.\n"), SolverError);
    EXPECT_THROW(validate("MPC,1,7,1,1.\n"), SolverError);
}

TEST(CRodParsing, FollowsTheCardDefinition) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,2.,0.,0.\n"
        "CROD,7,,1,2,8,4,2,3\nPROD,7,1,1.5,0.5,0.25,0.75\nENDDATA\n");
    ASSERT_EQ(model.elements.size(), 2U);
    // A blank PID is the EID.
    EXPECT_EQ(model.elements[0].pid, PropertyId{7});
    EXPECT_EQ(model.elements[1].id, ElementId{8});
    EXPECT_EQ(model.elements[1].pid, PropertyId{4});
    const auto& prod = std::get<PRod>(model.properties.at(PropertyId{7}));
    EXPECT_EQ(prod.A, 1.5);
    EXPECT_EQ(prod.J, 0.5);
    EXPECT_EQ(prod.C, 0.25);
    EXPECT_EQ(prod.nsm, 0.75);
}

// ── Scalar point solutions ────────────────────────────────────────────────────

TEST(ScalarPointStatics, SpringChainWithSloadAndSpc) {
    // Springs in series from a fixed scalar point: each extension is F / k, and
    // the only reaction is at point 1, on its one component.
    const auto r = solve_static(static_deck(
        "SPOINT,1,THRU,3\nCELAS4,1,100.,1,2\nCELAS4,2,50.,2,3\nSPC,1,1,0\nSLOAD,1,3,10.\n"));
    EXPECT_NEAR(disp(r, 2).d[0], 10.0 / 100.0, 1e-12);
    EXPECT_NEAR(disp(r, 3).d[0], 10.0 / 100.0 + 10.0 / 50.0, 1e-12);
    EXPECT_TRUE(disp(r, 3).scalar);
    for (int c = 1; c < 6; ++c)
        EXPECT_EQ(disp(r, 3).d[size_t(c)], 0.0);
    ASSERT_NE(spc_at(r, 1), nullptr);
    EXPECT_NEAR(spc_at(r, 1)->f[0], -10.0, 1e-11);
    for (const auto& grid : r.gp_forces)
        for (const double total : grid.totals)
            EXPECT_NEAR(total, 0.0, 1e-11);
}

TEST(ScalarPointStatics, LoadCombinationScalesSload) {
    const auto r = solve_static(static_deck(
        "SPOINT,2\nCELAS2,1,10.,2\nSLOAD,7,2,3.\nLOAD,1,2.,1.5,7\n"));
    EXPECT_NEAR(disp(r, 2).d[0], 2.0 * 1.5 * 3.0 / 10.0, 1e-12);
}

TEST(ScalarPointStatics, SloadOnAGridActsOnT1OfItsDisplacementFrame) {
    // Grid 2's CD X axis is basic Y, along the rod to grid 3, so the load
    // shortens the rod by F L / (E A) and T1 reports it.
    const auto r = solve_static(static_deck(
        "CORD2R,5,,0.,0.,0.,0.,0.,1.,\n,0.,1.,0.\nGRID,2,,0.,0.,0.,5\nGRID,3,,0.,10.,0.\n"
        "CROD,1,1,2,3\nSPC1,1,123,3\nSLOAD,1,2,3.\n"));
    EXPECT_NEAR(disp(r, 2).d[0], 3.0 * 10.0 / (kE * kA), 1e-12);
    EXPECT_NEAR(rod(r, 1).axial_force, -3.0, 1e-8);
}

TEST(ScalarPointStatics, SpringToGridAndEnforcedScalarValue) {
    // A nonzero SPC on a scalar point drives a grid through a spring against a
    // second spring: a divider of 3:1 on the enforced value.
    const auto r = solve_static(static_deck(
        "SPOINT,10\nGRID,2,,1.,0.,0.\nCELAS2,1,30.,2,1,10\nCELAS2,2,10.,2,1\n"
        "SPC,1,10,0,0.4\nSPC1,1,23456,2\n"));
    EXPECT_NEAR(disp(r, 2).d[0], 0.4 * 30.0 / 40.0, 1e-12);
    EXPECT_NEAR(disp(r, 10).d[0], 0.4, 1e-12);
    ASSERT_NE(spc_at(r, 10), nullptr);
    EXPECT_NEAR(spc_at(r, 10)->f[0], 30.0 * (0.4 - 0.3), 1e-10);
}

TEST(ScalarPointStatics, ScalarPointEnforcesAnMpcValueOnARod) {
    // The Nastran idiom for a nonhomogeneous MPC: u(2) - s = 0 with s enforced.
    const double d = 0.01;
    const auto r = solve_static(static_deck(
        "SPOINT,50\nGRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC1,1,123,1\n"
        "SPC,1,50,0,0.01\nMPC,4,2,1,1.,50,,-1.\n", "MPC = 4\n"));
    EXPECT_NEAR(disp(r, 2).d[0], d, 1e-12);
    EXPECT_NEAR(rod(r, 1).axial_force, kE * kA * d / 10.0, 1e-7);
    ASSERT_NE(spc_at(r, 50), nullptr);
    EXPECT_NEAR(spc_at(r, 50)->f[0], kE * kA * d / 10.0, 1e-7);
}

TEST(ScalarPointStatics, GravityActsOnGridsNotScalarPoints) {
    // Scalar mass has no rigid-body motion, so GRAV loads only the grid's mass,
    // even through a CMASS between the two.
    const auto r = solve_static(static_deck(
        "SPOINT,9\nGRID,2,,0.,0.,0.\nCMASS2,1,5.,9\nCELAS2,2,10.,9\nCMASS2,3,2.,2,1\n"
        "CELAS2,4,10.,2,1\nSPC1,1,23456,2\nGRAV,1,0,3.,1.,0.,0.\n"));
    EXPECT_EQ(disp(r, 9).d[0], 0.0);
    EXPECT_NEAR(disp(r, 2).d[0], 2.0 * 3.0 / 10.0, 1e-12);
}

TEST(ScalarPointModal, SpringMassChainMatchesItsExactEigenvalues) {
    // n equal masses and springs, grounded at one end, free at the other:
    // lambda_j = (k / m) (2 - 2 cos((2j - 1) pi / (2n + 1))).
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 7\nBEGIN BULK\nEIGRL,7,,,3\nSPOINT,1,THRU,3\n"
        "CELAS2,1,4.,1\nCELAS4,2,4.,1,2\nCELAS4,3,4.,2,3\nPMASS,8,1.\nCMASS3,4,8,1\nCMASS3,5,8,2\nCMASS4,6,1.,3\nENDDATA\n");
    ASSERT_EQ(r.modes.size(), 3U);
    for (int j = 1; j <= 3; ++j) {
        const double expected = 4.0 * (2.0 - 2.0 * std::cos((2 * j - 1) * std::numbers::pi / 7.0));
        EXPECT_NEAR(r.modes[size_t(j - 1)].eigenvalue, expected, 1e-9) << "mode " << j;
    }
    // The fundamental mode rises monotonically towards the free end.
    const auto& shape = r.modes[0].shape;
    ASSERT_EQ(shape.size(), 3U);
    EXPECT_TRUE(shape[2].scalar);
    EXPECT_LT(std::abs(shape[0].d[0]), std::abs(shape[1].d[0]));
    EXPECT_LT(std::abs(shape[1].d[0]), std::abs(shape[2].d[0]));
}

TEST(ScalarPointModal, ScalarOscillatorOnARod) {
    // A scalar oscillator coupled to a rod tip: two-DOF eigenproblem.
    const auto r = solve_modal(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 7\nBEGIN BULK\nEIGRL,7,,,2\n"
        "MAT1,1,1000.,,0.3\nPROD,1,1,1.\nGRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nSPOINT,9\n"
        "CROD,1,1,1,2\nCELAS2,2,50.,2,1,9\nCMASS2,3,2.,2,1\nCMASS2,4,1.,9\nSPC1,1,123,1\nENDDATA\n");
    const double kr = 100.0, ks = 50.0, m1 = 2.0, m2 = 1.0;
    const double trace = (kr + ks) / m1 + ks / m2, det = kr * ks / (m1 * m2);
    const double disc = std::sqrt(trace * trace - 4.0 * det);
    ASSERT_EQ(r.modes.size(), 2U);
    EXPECT_NEAR(r.modes[0].eigenvalue, 0.5 * (trace - disc), 1e-8);
    EXPECT_NEAR(r.modes[1].eigenvalue, 0.5 * (trace + disc), 1e-8);
}

TEST(ScalarPointQuality, ScalarPointsAreNeverCoincidentGrids) {
    // They all sit at the origin in storage, like grid 1; merging them would
    // collapse the model.
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nGRID,1,,0.,0.,0.\nSPOINT,2,3\nCELAS4,1,1.,2,3\nCELAS2,2,1.,1,1,2\nENDDATA\n");
    const auto topology = check_topology(model, 1e-8);
    EXPECT_TRUE(topology.duplicate_node_pairs.empty());
}

// ── CONROD ────────────────────────────────────────────────────────────────────

TEST(ConrodParsing, FollowsTheCardDefinition) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,2.,0.,0.\n"
        "CONROD,7,2,3,4,1.5,0.5,0.25,0.75\nCONROD,8,1,2,6,3.5\nENDDATA\n");
    ASSERT_EQ(model.elements.size(), 2U);
    const ElementData& full = model.elements[0];
    EXPECT_EQ(full.type, ElementType::CONROD);
    EXPECT_EQ(full.id, ElementId{7});
    EXPECT_EQ(full.nodes, (std::vector<NodeId>{NodeId{2}, NodeId{3}}));
    const auto& section = std::get<PRod>(model.property(full.pid));
    EXPECT_EQ(section.mid, MaterialId{4});
    EXPECT_EQ(section.A, 1.5);
    EXPECT_EQ(section.J, 0.5);
    EXPECT_EQ(section.C, 0.25);
    EXPECT_EQ(section.nsm, 0.75);
    // J, C and NSM default to zero.
    const auto& area_only = std::get<PRod>(model.property(model.elements[1].pid));
    EXPECT_EQ(area_only.mid, MaterialId{6});
    EXPECT_EQ(area_only.A, 3.5);
    EXPECT_EQ(area_only.J, 0.0);
    EXPECT_EQ(area_only.C, 0.0);
    EXPECT_EQ(area_only.nsm, 0.0);
}

TEST(ConrodParsing, RejectsMisuse) {
    const auto parse = [](const std::string& bulk) {
        return BdfParser::parse_string("SOL 101\nCEND\nBEGIN BULK\n" + bulk + "ENDDATA\n");
    };
    const std::string grids = "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\n";
    EXPECT_THROW(parse(grids + "CONROD,1,1,2,1,-1.\n"), ParseError);
    EXPECT_THROW(parse(grids + "CONROD,1,1,2,1,1.,-1.\n"), ParseError);
    EXPECT_THROW(parse(grids + "CONROD,-1,1,2,1,1.\n"), ParseError);
    EXPECT_THROW(parse(grids + "CONROD,1,1,2,1,1.\nCONROD,1,2,1,1,1.\n"), ParseError);
    EXPECT_THROW(parse(grids + "CONROD,1,1,2,1,1.,,,,+\n+,5\n"), ParseError);

    const auto validate = [&](const std::string& bulk) { parse(bulk).validate(); };
    EXPECT_THROW(validate(grids + "CONROD,1,1,2,9,1.\n"), SolverError);
    EXPECT_THROW(validate(grids + "MAT1,1,1.,,0.3\nCONROD,1,1,3,1,1.\n"), SolverError);
    EXPECT_THROW(validate("SPOINT,1\nGRID,2,,1.,0.,0.\nMAT1,1,1.,,0.3\nCONROD,1,1,2,1,1.\n"),
                 SolverError);
}

TEST(ConrodElement, ZeroLengthNamesTheCard) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nMAT1,1,1.0E7,,0.3\nGRID,1,,1.,1.,1.\nGRID,2,,1.,1.,1.\n"
        "CONROD,7,1,2,1,1.\nENDDATA\n");
    try {
        const CRodElement rod_element(model.elements.at(0), model);
        FAIL() << "a zero-length CONROD must throw";
    } catch (const SolverError& error) {
        EXPECT_NE(std::string(error.what()).find("CONROD 7"), std::string::npos) << error.what();
    }
}

TEST(ConrodQuality, ItsSectionIsNotOrphanedAndUsesItsMaterial) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nMAT1,1,1.0E7,,0.3\nMAT1,2,1.0E7,,0.3\nGRID,1,,0.,0.,0.\n"
        "GRID,2,,1.,0.,0.\nCONROD,7,1,2,1,1.\nENDDATA\n");
    const auto topology = check_topology(model, 1e-8);
    EXPECT_TRUE(topology.orphaned_properties.empty());
    EXPECT_EQ(topology.orphaned_materials, std::vector<MaterialId>{MaterialId{2}});
}

TEST(ConrodStatics, EveryFieldActsAsItsClosedFormRequires) {
    // E = 1e7, G = 4e6, A = 2, J = 1.5, C = 0.5 and L = 10, under P = 1000 and T = 300.
    const double P = 1000.0, T = 300.0, L = 10.0;
    const auto r = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCONROD,5,1,2,1,2.0,1.5,0.5,0.3\n"
        "SPC1,1,123456,1\nFORCE,1,2,0,1000.,1.,0.,0.\nMOMENT,1,2,0,300.,1.,0.,0.\n"));
    EXPECT_NEAR(disp(r, 2).d[0], P * L / (kE * kA), 1e-14);
    EXPECT_NEAR(disp(r, 2).d[3], T * L / (kG * kJ), 1e-14);
    const RodResult& result = rod(r, 5);
    EXPECT_EQ(result.type, ElementType::CONROD);
    EXPECT_NEAR(result.axial_force, P, 1e-8);
    EXPECT_NEAR(result.torque, T, 1e-8);
    EXPECT_NEAR(result.axial_stress, P / kA, 1e-8);
    EXPECT_NEAR(result.torsional_stress, kC * T / kJ, 1e-8);
    EXPECT_NEAR(result.axial_strain, P / (kA * kE), 1e-14);
    EXPECT_NEAR(result.torsional_strain, kC * T / (kJ * kG), 1e-14);
    ASSERT_NE(spc_at(r, 1), nullptr);
    EXPECT_NEAR(spc_at(r, 1)->f[0], -P, 1e-8);
    EXPECT_NEAR(spc_at(r, 1)->f[3], -T, 1e-8);

    // NSM and rho A both act through gravity: (0.2 + 0.3) * 10 * 386.
    const auto gravity = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,0.,0.,10.\nCONROD,5,1,2,1,2.0,1.5,0.5,0.3\n"
        "SPC1,1,123,1\nSPC1,1,12,2\nGRAV,1,0,386.,0.,0.,-1.\n"));
    ASSERT_NE(spc_at(gravity, 1), nullptr);
    EXPECT_NEAR(spc_at(gravity, 1)->f[2], (kRho * kA + kNsm) * L * 386.0, 1e-8);
}

TEST(ConrodStatics, SectionStaysApartFromAProdOfTheSameNumber) {
    // PROD 5 gives CROD 6 an area of 8 while CONROD 5 keeps its own area of 2, so
    // the same load stretches them by a ratio of 4.
    const double P = 1000.0, L = 10.0;
    const auto r = solve_static(static_deck(
        "PROD,5,1,8.0\nGRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,0.,5.,0.\nGRID,4,,10.,5.,0.\n"
        "CONROD,5,1,2,1,2.0\nCROD,6,5,3,4\nSPC1,1,123,1,3\n"
        "FORCE,1,2,0,1000.,1.,0.,0.\nFORCE,1,4,0,1000.,1.,0.,0.\n"));
    EXPECT_NEAR(disp(r, 2).d[0], P * L / (kE * 2.0), 1e-14);
    EXPECT_NEAR(disp(r, 4).d[0], P * L / (kE * 8.0), 1e-14);
    EXPECT_NEAR(rod(r, 5).axial_stress, P / 2.0, 1e-8);
    EXPECT_NEAR(rod(r, 6).axial_stress, P / 8.0, 1e-8);
}

TEST(ConrodStatics, MixesWithCrodInADeterminateTruss) {
    // The truss of CRodStatics.DeterminateTrussMatchesTheMethodOfJoints with its
    // loaded member and diagonal as CONRODs.
    const double P = 1000.0, b = 100.0, EA = kE * kA, diagonal = b * std::numbers::sqrt2;
    const auto r = solve_static(static_deck(
        "GRID,1,,0.,0.,0.\nGRID,2,,100.,0.,0.\nGRID,3,,100.,100.,0.\n"
        "CROD,1,1,1,2\nCONROD,2,2,3,1,2.0\nCONROD,3,1,3,1,2.0\n"
        "SPC1,1,123,1\nSPC1,1,23,2\nSPC1,1,3,3\nFORCE,1,3,0,1000.,1.,0.,0.\n"));
    EXPECT_EQ(rod(r, 1).type, ElementType::CROD);
    EXPECT_EQ(rod(r, 2).type, ElementType::CONROD);
    EXPECT_NEAR(rod(r, 1).axial_force, 0.0, 1e-7);
    EXPECT_NEAR(rod(r, 2).axial_force, -P, 1e-7);
    EXPECT_NEAR(rod(r, 3).axial_force, P * std::numbers::sqrt2, 1e-7);
    const double v = -P * b / EA;
    EXPECT_NEAR(disp(r, 3).d[0], -v + 2.0 * P * diagonal / EA, 1e-12);
    EXPECT_NEAR(disp(r, 3).d[1], v, 1e-12);
    ASSERT_FALSE(r.gp_forces.empty());
    for (const auto& grid : r.gp_forces)
        for (const double total : grid.totals)
            EXPECT_NEAR(total, 0.0, 1e-7) << "grid " << grid.node.value;
}

TEST(ConrodStatics, IsEquivalentToCrodAndProdUnderEveryLoad) {
    // Three skew rods meeting at grid 4, loaded by force, moment, gravity and
    // temperature. Each CONROD carries the section of PROD 1, so nothing may differ.
    const auto solve = [](const std::string& rods) {
        return solve_static(static_deck(
            "GRID,1,,0.,0.,0.\nGRID,2,,30.,0.,0.\nGRID,3,,0.,40.,0.\nGRID,4,,10.,10.,20.\n" + rods +
                "SPC1,1,123456,1,2,3\nFORCE,1,4,0,900.,1.,-2.,0.5\nMOMENT,1,4,0,70.,0.3,1.,-1.\n"
                "GRAV,1,0,386.,0.,0.,-1.\nTEMP,2,1,10.\nTEMP,2,2,20.\nTEMP,2,3,30.\nTEMP,2,4,40.\n",
            "TEMP(LOAD) = 2\n"));
    };
    const auto with_crod = solve("CROD,1,1,1,4\nCROD,2,1,2,4\nCROD,3,1,3,4\n");
    const auto with_conrod = solve(
        "CONROD,1,1,4,1,2.0,1.5,0.5,0.3\nCONROD,2,2,4,1,2.0,1.5,0.5,0.3\n"
        "CONROD,3,3,4,1,2.0,1.5,0.5,0.3\n");

    EXPECT_GT(std::abs(disp(with_crod, 4).d[0]), 0.0);
    EXPECT_EQ(disp(with_conrod, 4).d, disp(with_crod, 4).d);
    ASSERT_EQ(with_conrod.rod_results.size(), 3U);
    for (const int eid : {1, 2, 3}) {
        const RodResult& a = rod(with_conrod, eid);
        const RodResult& b = rod(with_crod, eid);
        EXPECT_EQ(a.axial_force, b.axial_force);
        EXPECT_EQ(a.torque, b.torque);
        EXPECT_EQ(a.axial_stress, b.axial_stress);
        EXPECT_EQ(a.torsional_stress, b.torsional_stress);
        EXPECT_EQ(a.axial_strain, b.axial_strain);
        EXPECT_EQ(a.torsional_strain, b.torsional_strain);
        ASSERT_NE(spc_at(with_conrod, eid), nullptr);
        EXPECT_EQ(spc_at(with_conrod, eid)->f, spc_at(with_crod, eid)->f);
    }
}

TEST(ConrodAutoSpc, ConstrainsExactlyTheDirectionsARodLeavesFree) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nSPC = 1\nBEGIN BULK\nMAT1,1,1.0E7,,0.3\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,20.,0.,0.\nCONROD,1,1,2,1,1.\nCONROD,2,2,3,1,1.\n"
        "SPC1,1,123,1\nENDDATA\n");
    const SubCase sc = model.analysis.subcases.at(0);
    const DofMap dofs = build_analysis_dof_map(model, sc);
    EXPECT_EQ(dofs.num_free_dofs(), 2);
    EXPECT_TRUE(dofs.is_free(NodeId{2}, 0));
    EXPECT_TRUE(dofs.is_free(NodeId{3}, 0));
}

TEST(ConrodModal, SingleElementFrequenciesAreExact) {
    // Fixed-free with E = 1e7, G = 4e6, rho = 0.1, A = 2, J = 0.5, NSM = 0.75 and L = 10.
    // Consistent mass gives 3 E A / ((rho A + NSM) L^2) axially, where NSM counts,
    // and 3 G / (rho L^2) in torsion, where it does not. Lumped mass has 2 in place
    // of 3 axially and no rotary mass.
    const std::string bulk =
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 1\nBEGIN BULK\nMAT1,1,1.0E7,4.0E6,0.25,0.1\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCONROD,1,1,2,1,2.0,0.5,,0.75\nSPC1,1,123456,1\n";
    const double axial = kE * 2.0 / ((kRho * 2.0 + 0.75) * 100.0);
    const auto consistent = solve_modal(bulk + "EIGRL,1,,,2\nENDDATA\n");
    ASSERT_EQ(consistent.modes.size(), 2U);
    EXPECT_NEAR(consistent.modes[0].eigenvalue, 3.0 * axial, 1e-5 * axial);
    EXPECT_NEAR(consistent.modes[1].eigenvalue, 3.0 * kG / (kRho * 100.0), 1e-5 * axial);
    const auto lumped = solve_modal(bulk + "EIGRL,1,,,1\nPARAM,COUPMASS,-1\nENDDATA\n");
    ASSERT_EQ(lumped.modes.size(), 1U);
    EXPECT_NEAR(lumped.modes[0].eigenvalue, 2.0 * axial, 1e-5 * axial);
}

TEST(ConrodHeat, ConductionFollowsFouriersLawAndKeepsItsName) {
    // k = 50, A = 2, L = 10: a power q into the far end drops q L / (k A).
    Model model = BdfParser::parse_string(
        "SOL 153\nCEND\nSPC = 1\nLOAD = 1\nFLUX = ALL\nBEGIN BULK\nMAT4,1,50.\nGRID,1,,0.,0.,0.\n"
        "GRID,2,,10.,0.,0.\nCONROD,1,1,2,1,2.0\nSPC,1,1,0,20.\nSLOAD,1,2,100.\nENDDATA\n");
    HeatTransferSteadySolver solver(std::make_unique<EigenSolverBackend>());
    const SolverResults results = solver.solve(model);
    const SubCaseResults& r = results.subcases.at(0);
    EXPECT_NEAR(temperature(r, 2), 20.0 + 100.0 * 10.0 / (50.0 * 2.0), 1e-9);
    ASSERT_EQ(r.heat_fluxes.size(), 1U);
    EXPECT_EQ(r.heat_fluxes[0].etype, ElementType::CONROD);
    EXPECT_NEAR(std::abs(r.heat_fluxes[0].q[0]), 100.0 / 2.0, 1e-9);
    std::ostringstream f06;
    F06Writer::write_thermal(results, model, f06);
    EXPECT_NE(f06.str().find("CONROD"), std::string::npos);
}
