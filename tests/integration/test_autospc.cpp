// tests/integration/test_autospc.cpp
// AUTOSPC of grid point singularities at the grids of any element, checked against
// closed-form solutions on the direct and iterative CPU backends, since a singular
// row in K is not something either could solve correctly without it.

#include "core/exceptions.hpp"
#include "elements/element_factory.hpp"
#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/analysis_support.hpp"
#include "solver/eigensolver_backend.hpp"
#include "solver/linear_static.hpp"
#include "solver/modal.hpp"
#include "solver/solver_backend.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <format>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace vibestran;

namespace {

enum class Backend { Direct, CpuPcg };

SubCaseResults solve_with(const Backend backend, const std::string& deck) {
    Model model = BdfParser::parse_string(deck);
    std::unique_ptr<SolverBackend> solver;
    if (backend == Backend::Direct)
        solver = std::make_unique<LdltSolverBackend>();
    else
        solver = std::make_unique<CpuPCGSolverBackend>();
    return LinearStaticSolver(std::move(solver)).solve(model).subcases.at(0);
}

const NodeDisplacement& disp(const SubCaseResults& r, const int node) {
    const auto it = std::ranges::find(r.displacements, NodeId{node}, &NodeDisplacement::node);
    if (it == r.displacements.end())
        throw std::runtime_error(std::format("no displacement for grid {}", node));
    return *it;
}

const SpcForce* spc_at(const SubCaseResults& r, const int node) {
    const auto it = std::ranges::find(r.spc_forces, NodeId{node}, &SpcForce::node);
    return it == r.spc_forces.end() ? nullptr : &*it;
}

std::string static_deck(const std::string& bulk) {
    return "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nDISP = ALL\nSPCF = ALL\nGPFORCE = ALL\n"
           "BEGIN BULK\nMAT1,1,1.0E7,,0.3\n" + bulk + "ENDDATA\n";
}

void expect_every_grid_balances(const SubCaseResults& r, const double tolerance = 1e-7) {
    ASSERT_FALSE(r.gp_forces.empty());
    for (const auto& grid : r.gp_forces)
        for (const double total : grid.totals)
            EXPECT_NEAR(total, 0.0, tolerance) << "grid " << grid.node.value;
}

// E = 1e7, I = 3, A = 2 and no J, so nothing resists torsion.
const std::string kBarNoTorsion = "PBAR,1,1,2.0,3.0,3.0\n";
constexpr double kE = 1.0e7, kI = 3.0;

class AutoSpcBothBackends : public ::testing::TestWithParam<Backend> {};

INSTANTIATE_TEST_SUITE_P(Cpu, AutoSpcBothBackends,
                         ::testing::Values(Backend::Direct, Backend::CpuPcg));

} // namespace

TEST_P(AutoSpcBothBackends, BarWithoutTorsionPassesTheMomentOnItToGround) {
    // A moment on the tip's R1 has no path into the bar: it stays at the tip as a
    // reaction, and the tip force bends the bar as if the moment were not there.
    const auto r = solve_with(GetParam(), static_deck(
        kBarNoTorsion +
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,20.,0.,0.\n"
        "CBAR,1,1,1,2,0.,1.,0.\nCBAR,2,1,2,3,0.,1.,0.\nSPC1,1,123456,1\n"
        "FORCE,1,3,0,100.,0.,1.,0.\nMOMENT,1,3,0,7.5,1.,0.,0.\n"));
    const double L = 20.0, P = 100.0;
    const double tip = P * L * L * L / (3.0 * kE * kI);
    EXPECT_NEAR(disp(r, 3).d[1], tip, 1e-9 * tip);
    EXPECT_EQ(disp(r, 3).d[3], 0.0);
    EXPECT_EQ(disp(r, 2).d[3], 0.0);

    ASSERT_NE(spc_at(r, 3), nullptr);
    EXPECT_NEAR(spc_at(r, 3)->f[3], -7.5, 1e-9);
    ASSERT_NE(spc_at(r, 1), nullptr);
    EXPECT_NEAR(spc_at(r, 1)->f[1], -P, 1e-7);
    EXPECT_NEAR(spc_at(r, 1)->f[5], -P * L, 1e-6);
    EXPECT_NEAR(spc_at(r, 1)->f[3], 0.0, 1e-9);
    // Held with nothing to react, so it stays out of the table.
    EXPECT_EQ(spc_at(r, 2), nullptr);
    expect_every_grid_balances(r);
}

TEST_P(AutoSpcBothBackends, ObliqueAxialReleaseHoldsTheNearestComponent) {
    // A bar along (3,0,4)/5, length 5, axially released at its tip: the tip slides
    // freely along the bar, whose direction is nearest T3, so T3 is the one held.
    // Along the bar the tip force and the T3 reaction must cancel, so with F along
    // X the reaction is -3F/4, leaving 5F/4 across the bar at k = 3EI/L^3 since the
    // tip rotation is free. The tip moves a = 5F/(4k) across the bar and slides by
    // whatever keeps T3 at zero, which puts T1 at 5a/4.
    const auto r = solve_with(GetParam(), static_deck(
        "PBAR,1,1,2.0,3.0,3.0,1.5\nGRID,1,,0.,0.,0.\nGRID,2,,3.,0.,4.\n"
        "CBAR           1       1       1       2      0.      1.      0.        +\n"
        "+                      1\n"
        "SPC1,1,123456,1\nFORCE,1,2,0,100.,1.,0.,0.\n"));
    const double F = 100.0, L = 5.0;
    const double k = 3.0 * kE * kI / (L * L * L);
    const double expected = 25.0 / 16.0 * F / k;
    EXPECT_NEAR(disp(r, 2).d[0], expected, 1e-9 * expected);
    EXPECT_NEAR(disp(r, 2).d[1], 0.0, 1e-12);
    EXPECT_EQ(disp(r, 2).d[2], 0.0);
    ASSERT_NE(spc_at(r, 2), nullptr);
    EXPECT_NEAR(spc_at(r, 2)->f[2], -0.75 * F, 1e-7);
    ASSERT_NE(spc_at(r, 1), nullptr);
    EXPECT_NEAR(spc_at(r, 1)->f[0], -F, 1e-7);
    EXPECT_NEAR(spc_at(r, 1)->f[2], 0.75 * F, 1e-7);
    EXPECT_NEAR(spc_at(r, 1)->f[4], -6.25 * F, 1e-6);
    expect_every_grid_balances(r);
}

TEST_P(AutoSpcBothBackends, OnACdGridTheHeldComponentFollowsTheSingularDirection) {
    // The tip's CD system has x along basic Y and y along basic -X, so the torsion a
    // bar along basic X does not resist is that grid's R2. A moment about basic +X
    // is -7.5 about its y axis, and comes back as a reaction of +7.5.
    const auto r = solve_with(GetParam(), static_deck(
        kBarNoTorsion +
        "CORD2R,5,,0.,0.,0.,0.,0.,1.\n,0.,1.,0.\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.,5\nCBAR,1,1,1,2,0.,1.,0.\nSPC1,1,123456,1\n"
        "FORCE,1,2,0,100.,0.,1.,0.\nMOMENT,1,2,0,7.5,1.,0.,0.\n"));
    const double L = 10.0, P = 100.0;
    const double tip = P * L * L * L / (3.0 * kE * kI);
    EXPECT_NEAR(disp(r, 2).d[0], tip, 1e-9 * tip); // the load is along the grid's x
    EXPECT_NEAR(disp(r, 2).d[4], 0.0, 1e-14);
    ASSERT_NE(spc_at(r, 2), nullptr);
    EXPECT_NEAR(spc_at(r, 2)->f[4], 7.5, 1e-9);
    expect_every_grid_balances(r);
}

// A 10 x 10 plate of one quad or two triangles, on which the section and the loads vary.
std::string membrane_plate(const bool tri, const std::string& section, const std::string& rest) {
    return "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nTEMP(LOAD) = 2\nDISP = ALL\nSPCF = ALL\nSTRESS = ALL\n"
           "BEGIN BULK\nMAT1,1,1.0E7,,0.,,1.0E-5,0.\n" + section + "\n"
           "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,10.,10.,0.\nGRID,4,,0.,10.,0.\n" +
           (tri ? "CTRIA3,1,1,1,2,3\nCTRIA3,2,1,1,3,4\n" : "CQUAD4,1,1,1,2,3,4\n") + rest +
           "ENDDATA\n";
}

void expect_out_of_plane_is_held(const SubCaseResults& r) {
    for (const int grid : {1, 2, 3, 4})
        for (const size_t c : {size_t{2}, size_t{3}, size_t{4}})
            EXPECT_EQ(disp(r, grid).d[c], 0.0) << "grid " << grid << " component " << c + 1;
}

TEST_P(AutoSpcBothBackends, MembraneOnlyShellHoldsEverythingOutOfPlane) {
    // With nothing but membrane stiffness a flat plate has nothing out of plane, so a
    // load out of plane is a reaction, and tension stretches it by P L / (E A) and
    // stresses it by P / A. A blank MID2 means that whatever MID3 says, as do zero
    // bending and shear factors. R3 stays free: each element's drill stiffness is
    // tied to the membrane.
    const std::string loads = "SPC1,1,1,1,4\nSPC1,1,2,1\n"
                              "FORCE,1,2,0,50.,1.,0.,0.\nFORCE,1,3,0,50.,1.,0.,0.\n"
                              "FORCE,1,3,0,7.,0.,0.,1.\n";
    const double stretch = 100.0 * 10.0 / (1.0e7 * 10.0 * 0.1);
    for (const bool tri : {false, true})
        for (const char* section : {"PSHELL,1,1,0.1,1,0.,1,0.", "PSHELL,1,1,0.1", "PSHELL,1,1,0.1,,,1"}) {
            SCOPED_TRACE(std::format("{} {}", tri ? "CTRIA3" : "CQUAD4", section));
            const auto r = solve_with(GetParam(), membrane_plate(tri, section, loads));
            EXPECT_NEAR(disp(r, 2).d[0], stretch, 1e-9 * stretch);
            EXPECT_NEAR(disp(r, 3).d[0], stretch, 1e-9 * stretch);
            EXPECT_NEAR(disp(r, 3).d[1], 0.0, 1e-9 * stretch);
            expect_out_of_plane_is_held(r);
            ASSERT_NE(spc_at(r, 3), nullptr);
            EXPECT_NEAR(spc_at(r, 3)->f[2], -7.0, 1e-9);
            ASSERT_FALSE(r.plate_stresses.empty());
            for (const auto& stress : r.plate_stresses) {
                // Each element reports in its own axes, and the second triangle's are at 45 degrees.
                EXPECT_NEAR(stress.von_mises, 100.0 / (10.0 * 0.1), 1e-6);
                EXPECT_NEAR(stress.mx, 0.0, 1e-12);
                EXPECT_NEAR(stress.qx, 0.0, 1e-12);
            }
        }
}

TEST_P(AutoSpcBothBackends, MembraneOnlyShellExpandsAndStressesUnderTemperature) {
    // A free plate grows by alpha dT L without stress. Held at both ends it can't, and
    // has -E alpha dT in that direction.
    const double growth = 1.0e-5 * 100.0 * 10.0;
    const std::string temperature = "TEMP,2,1,100.\nTEMP,2,2,100.\nTEMP,2,3,100.\nTEMP,2,4,100.\n";
    for (const bool tri : {false, true}) {
        SCOPED_TRACE(tri ? "CTRIA3" : "CQUAD4");
        const auto free_plate = solve_with(GetParam(), membrane_plate(tri, "PSHELL,1,1,0.1",
            "SPC1,1,1,1,4\nSPC1,1,2,1,2\n" + temperature));
        EXPECT_NEAR(disp(free_plate, 2).d[0], growth, 1e-9 * growth);
        EXPECT_NEAR(disp(free_plate, 3).d[1], growth, 1e-9 * growth);
        expect_out_of_plane_is_held(free_plate);
        for (const auto& stress : free_plate.plate_stresses)
            EXPECT_NEAR(stress.sx, 0.0, 1e-6);

        const auto held = solve_with(GetParam(), membrane_plate(tri, "PSHELL,1,1,0.1",
            "SPC1,1,12,1,2,3,4\n" + temperature));
        ASSERT_FALSE(held.plate_stresses.empty());
        for (const auto& stress : held.plate_stresses)
            EXPECT_NEAR(stress.sx, -1.0e7 * 1.0e-5 * 100.0, 1e-6);
    }
}

TEST_P(AutoSpcBothBackends, MembraneSkinOnASolidAddsItsAreaToTheSection) {
    // A 10 x 2 x 2 hex with a membrane on its top and bottom faces, pulled along x. The
    // strain is uniform, so it stretches by F L / (E (w h + 2 t w)) and thins by Poisson's
    // ratio. The solid covers the translations and the skins leave R1 and R2 uncovered.
    const auto faces = [](const bool tri, const int eid, const int a, const int b, const int c, const int d) {
        return tri ? std::format("CTRIA3,{},2,{},{},{}\nCTRIA3,{},2,{},{},{}\n", eid, a, b, c, eid + 1, a, c, d)
                   : std::format("CQUAD4,{},2,{},{},{},{}\n", eid, a, b, c, d);
    };
    for (const bool tri : {false, true}) {
        SCOPED_TRACE(tri ? "CTRIA3" : "CQUAD4");
        const auto r = solve_with(GetParam(), static_deck(
            "PSOLID,1,1\nPSHELL,2,1,0.1\n"
            "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,10.,2.,0.\nGRID,4,,0.,2.,0.\n"
            "GRID,5,,0.,0.,2.\nGRID,6,,10.,0.,2.\nGRID,7,,10.,2.,2.\nGRID,8,,0.,2.,2.\n"
            "CHEXA,1,1,1,2,3,4,5,6,\n,7,8\n" +
            faces(tri, 2, 1, 2, 3, 4) + faces(tri, 4, 5, 6, 7, 8) +
            "SPC1,1,1,1,4,5,8\nSPC1,1,2,1,2,5,6\nSPC1,1,3,1,2,3,4\n"
            "FORCE,1,2,0,100.,1.,0.,0.\nFORCE,1,3,0,100.,1.,0.,0.\n"
            "FORCE,1,6,0,100.,1.,0.,0.\nFORCE,1,7,0,100.,1.,0.,0.\n"));
        const double stretch = 400.0 * 10.0 / (1.0e7 * (2.0 * 2.0 + 2.0 * 0.1 * 2.0));
        const double thinning = -0.3 * stretch / 10.0 * 2.0;
        for (const int grid : {2, 3, 6, 7})
            EXPECT_NEAR(disp(r, grid).d[0], stretch, 1e-9 * stretch) << "grid " << grid;
        for (const int grid : {3, 4, 7, 8})
            EXPECT_NEAR(disp(r, grid).d[1], thinning, 1e-9 * stretch) << "grid " << grid;
        for (const int grid : {5, 6, 7, 8})
            EXPECT_NEAR(disp(r, grid).d[2], thinning, 1e-9 * stretch) << "grid " << grid;
        for (int grid = 1; grid <= 8; ++grid)
            for (const size_t c : {size_t{3}, size_t{4}})
                EXPECT_EQ(disp(r, grid).d[c], 0.0) << "grid " << grid << " component " << c + 1;
    }
}

TEST_P(AutoSpcBothBackends, MixedDeckMatchesTheSameDeckWithExplicitSpcs) {
    // A solid, complete shells, membranes of both element types, a rod and a bar, joined
    // at shared grids. What AUTOSPC holds at each grid follows from what connects there:
    //   1-4        solid only: no rotation stiffness
    //   5-9        a complete shell: nothing, whatever else connects
    //   10, 11, 12 membranes only: T3, R1, R2
    //   13         rod end with no J: all but T1
    //   14         bar end with no J: R2, about the bar's axis
    // The deck with those written out and AUTOSPC off is the same problem.
    const std::string model =
        "PSOLID,8,1\nPSHELL,2,1,0.1,1,,1\nPSHELL,3,1,0.1\nPROD,4,1,0.5\n" + kBarNoTorsion +
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nGRID,4,,0.,1.,0.\n"
        "GRID,5,,0.,0.,1.\nGRID,6,,1.,0.,1.\nGRID,7,,1.,1.,1.\nGRID,8,,0.,1.,1.\n"
        "GRID,9,,2.,0.5,1.\nGRID,10,,2.,1.5,1.\nGRID,11,,3.,0.5,1.\nGRID,12,,3.,1.5,1.\n"
        "GRID,13,,4.,0.5,1.\nGRID,14,,0.,2.,1.\n"
        "CHEXA,1,8,1,2,3,4,5,6,\n,7,8\nCQUAD4,2,2,5,6,7,8\nCTRIA3,3,2,6,9,7\n"
        "CTRIA3,4,3,7,9,10\nCQUAD4,5,3,9,11,12,10\nCROD,6,4,11,13\nCBAR,7,1,8,14,1.,0.,0.\n"
        "SPC1,1,123,1,2,3,4\n"
        "FORCE,1,13,0,100.,1.,0.,0.\nFORCE,1,14,0,5.,0.,0.,1.\nFORCE,1,14,0,3.,1.,0.,0.\n"
        "FORCE,1,12,0,10.,0.,1.,0.\nFORCE,1,9,0,2.,0.,0.,1.\nMOMENT,1,7,0,4.,1.,0.,0.\n"
        "FORCE,1,10,0,7.,0.,0.,1.\n";
    const auto automatic = solve_with(GetParam(), static_deck(model));
    const auto explicit_spcs = solve_with(GetParam(), static_deck(
        model + "PARAM,AUTOSPC,NO\nSPC1,1,456,1,2,3,4\nSPC1,1,345,10,11,12\nSPC1,1,23456,13\nSPC1,1,5,14\n"));

    double largest = 0.0;
    for (const auto& grid : explicit_spcs.displacements)
        largest = std::max(largest, Eigen::Map<const Eigen::Matrix<double, 6, 1>>(grid.d.data()).cwiseAbs().maxCoeff());
    ASSERT_GT(largest, 0.0);
    ASSERT_EQ(automatic.displacements.size(), explicit_spcs.displacements.size());
    for (const auto& grid : explicit_spcs.displacements)
        for (size_t c = 0; c < 6; ++c)
            EXPECT_NEAR(disp(automatic, grid.node.value).d[c], grid.d[c], 1e-10 * largest)
                << "grid " << grid.node.value << " component " << c + 1;
    ASSERT_NE(spc_at(automatic, 10), nullptr);
    EXPECT_NEAR(spc_at(automatic, 10)->f[2], -7.0, 1e-9);
    // A component AUTOSPC held with no load on it reports no row at all.
    for (const auto& reaction : explicit_spcs.spc_forces) {
        const SpcForce* held = spc_at(automatic, reaction.node.value);
        for (size_t c = 0; c < 6; ++c)
            EXPECT_NEAR(held != nullptr ? held->f[c] : 0.0, reaction.f[c], 1e-6)
                << "grid " << reaction.node.value << " component " << c + 1;
    }
    // This deck's residual is about 2e-7 with either set of constraints, all of it roundoff.
    expect_every_grid_balances(automatic, 1e-5);
}

TEST_P(AutoSpcBothBackends, LoadOnAGridNoElementConnectsIsAReaction) {
    const auto r = solve_with(GetParam(), static_deck(
        kBarNoTorsion +
        "CORD2R,5,,0.,0.,0.,0.,0.,1.\n,0.,1.,0.\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,8,,0.,5.,0.,5\nGRID,9,,0.,9.,0.\n"
        "CBAR,1,1,1,2,0.,1.,0.\nSPC1,1,123456,1\nFORCE,1,2,0,100.,0.,1.,0.\n"
        "FORCE,1,9,0,1111.,1.,0.,0.\nMOMENT,1,9,0,2222.,0.,0.,1.\n"
        "FORCE,1,8,0,10.,1.,0.,0.\n"));
    ASSERT_NE(spc_at(r, 9), nullptr);
    EXPECT_NEAR(spc_at(r, 9)->f[0], -1111.0, 1e-9);
    EXPECT_NEAR(spc_at(r, 9)->f[5], -2222.0, 1e-9);
    // On a CD grid, in its frame: basic +X is that system's -y.
    ASSERT_NE(spc_at(r, 8), nullptr);
    EXPECT_NEAR(spc_at(r, 8)->f[1], 10.0, 1e-9);
    EXPECT_NEAR(spc_at(r, 8)->f[0], 0.0, 1e-9);
    // The bar is unaffected.
    const double tip = 100.0 * 1000.0 / (3.0 * kE * kI);
    EXPECT_NEAR(disp(r, 2).d[1], tip, 1e-9 * tip);
    expect_every_grid_balances(r);
}

TEST_P(AutoSpcBothBackends, MomentAtAGridOnlySolidsConnectIsAReaction) {
    const auto r = solve_with(GetParam(), static_deck(
        "PSOLID,1,1\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nGRID,4,,0.,1.,0.\n"
        "GRID,5,,0.,0.,1.\nGRID,6,,1.,0.,1.\nGRID,7,,1.,1.,1.\nGRID,8,,0.,1.,1.\n"
        "CHEXA,1,1,1,2,3,4,5,6,+\n+,7,8\nSPC1,1,123456,1,2,3,4\n"
        "MOMENT,1,7,0,5.,0.,0.,1.\n"));
    ASSERT_NE(spc_at(r, 7), nullptr);
    EXPECT_NEAR(spc_at(r, 7)->f[5], -5.0, 1e-9);
    for (const size_t c : {size_t{0}, size_t{1}, size_t{2}})
        EXPECT_NEAR(disp(r, 7).d[c], 0.0, 1e-15);
}

TEST_P(AutoSpcBothBackends, BushHoldsWhatItsStiffnessLeavesFree) {
    const auto r = solve_with(GetParam(), static_deck(
        "PBUSH,1,K,1000.,0.,0.,0.,0.,0.\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\n"
        "CBUSH,1,1,1,2,0.,1.,0.\nSPC1,1,123456,1\n"
        "FORCE,1,2,0,50.,1.,0.,0.\nFORCE,1,2,0,30.,0.,1.,0.\n"));
    EXPECT_NEAR(disp(r, 2).d[0], 50.0 / 1000.0, 1e-12);
    EXPECT_EQ(disp(r, 2).d[1], 0.0);
    ASSERT_NE(spc_at(r, 2), nullptr);
    EXPECT_NEAR(spc_at(r, 2)->f[1], -30.0, 1e-9);
    EXPECT_NEAR(spc_at(r, 2)->f[0], 0.0, 1e-9);
}

TEST(AutoSpc, ParamNoLeavesEveryComponentFree) {
    const std::string bulk =
        kBarNoTorsion +
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,9,,0.,9.,0.\nCBAR,1,1,1,2,0.,1.,0.\n"
        "SPC1,1,123456,1\nFORCE,1,9,0,1111.,1.,0.,0.\nFORCE,1,2,0,100.,0.,1.,0.\n";
    const Model on = BdfParser::parse_string(static_deck(bulk));
    const Model off = BdfParser::parse_string(static_deck("PARAM,AUTOSPC,NO\n" + bulk));
    const DofMap held = build_analysis_dof_map(on, on.analysis.subcases.at(0));
    const DofMap kept = build_analysis_dof_map(off, off.analysis.subcases.at(0));
    EXPECT_FALSE(held.is_free(NodeId{2}, 3));
    EXPECT_TRUE(kept.is_free(NodeId{2}, 3));
    EXPECT_EQ(held.num_free_dofs(), 5);
    EXPECT_EQ(kept.num_free_dofs(), 6);
    EXPECT_FALSE(autospc_enabled(off));
    EXPECT_TRUE(autospc_enabled(on));
}

TEST(AutoSpc, WeakDrillOfALowShearSectionIsNotTakenForASingularity) {
    // Drill stiffness scales with the in-plane shear modulus, so this section's R3 is
    // far below 1e-8 of its bending. A shell's formulation gives it that stiffness by
    // design, so unlike a plain ratio test AUTOSPC keeps R3 free.
    const Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nSPC = 1\nBEGIN BULK\nMAT8,1,1.0E11,5.0E10,0.,1.,1.,1.\n"
        "PSHELL,1,1,3.0E-4,1,,1\nGRID,1,,0.,0.,0.\nGRID,2,,0.002,0.,0.\n"
        "GRID,3,,0.002,0.002,0.\nGRID,4,,0.,0.002,0.\nCQUAD4,1,1,1,2,3,4\n"
        "SPC1,1,123456,1,2,4\nENDDATA\n");
    const LocalKe ke = make_element(model.elements.at(0), model)->stiffness_matrix();
    const Eigen::Matrix3d rotation = ke.block<3, 3>(15, 15); // grid 3, the third of four
    ASSERT_GT(rotation(2, 2), 0.0);
    ASSERT_LT(rotation(2, 2), 1e-8 * rotation(0, 0));
    const DofMap dofs = build_analysis_dof_map(model, model.analysis.subcases.at(0));
    EXPECT_TRUE(dofs.is_free(NodeId{3}, 5));
    EXPECT_EQ(dofs.num_free_dofs(), 6);
}

TEST(AutoSpc, AtAGridABarSharesWithAShellOnlyTheBarsFreeEndIsHeld) {
    // A bar along the shell's normal has no J. The shell gives R3 stiffness at the
    // grid they share, and nothing does at the bar's other end.
    const Model model = BdfParser::parse_string(static_deck(
        "PSHELL,1,1,0.1,1,,1\nPBAR,2,1,100.,1000.,1000.\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nGRID,3,,10.,10.,0.\nGRID,4,,0.,10.,0.\n"
        "GRID,5,,10.,10.,10.\nCQUAD4,1,1,1,2,3,4\nCBAR,2,2,3,5,1.,0.,0.\n"
        "SPC1,1,123456,1,2,4\n"));
    const DofMap dofs = build_analysis_dof_map(model, model.analysis.subcases.at(0));
    EXPECT_TRUE(dofs.is_free(NodeId{3}, 5));
    EXPECT_FALSE(dofs.is_free(NodeId{5}, 5));
    EXPECT_EQ(dofs.num_free_dofs(), 11);
}

TEST(AutoSpc, SolidsAndCompleteShellsLeaveNoBlockSingular) {
    // analysis_spcs skips the blocks these elements cover, on the grounds that each
    // element's stiffness at its own grid is positive definite there. Check it on
    // skewed elements and on a thin, warped shell.
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nMAT1,1,1.0E7,,0.3\nPSOLID,1,1\n"
        "PSHELL,2,1,0.001,1,,1\nGRID,1000,,0.,0.,0.\nENDDATA\n");
    int next_grid = 1;
    const auto grid = [&](const Vec3 position) {
        const NodeId id{next_grid++};
        model.nodes[id] = GridPoint{id, CoordId{0}, position, CoordId{0}};
        return id;
    };
    const auto midside = [&](const NodeId a, const NodeId b) {
        const Vec3& p = model.nodes.at(a).position;
        const Vec3& q = model.nodes.at(b).position;
        return grid(Vec3{(p.x + q.x) / 2, (p.y + q.y) / 2, (p.z + q.z) / 2});
    };
    const auto corners = [&](const std::vector<Vec3>& positions) {
        std::vector<NodeId> ids;
        for (const Vec3& p : positions)
            ids.push_back(grid(p));
        return ids;
    };
    const auto add = [&](const ElementType type, const int pid, std::vector<NodeId> nodes) {
        ElementData e;
        e.id = ElementId{static_cast<int>(model.elements.size()) + 1};
        e.type = type;
        e.pid = PropertyId{pid};
        e.nodes = std::move(nodes);
        model.elements.push_back(std::move(e));
    };
    const std::vector<Vec3> skewed_box{{0., 0., 0.},   {2., 0.2, 0.},  {2.3, 1.1, 0.1}, {0.1, 1., 0.},
                                       {0.2, 0., 1.5}, {2., 0.3, 1.4}, {2.2, 1.3, 1.6}, {0.1, 1.1, 1.3}};
    add(ElementType::CHEXA8, 1, corners(skewed_box));
    {
        std::vector<NodeId> nodes = corners(skewed_box);
        const std::array<std::array<int, 2>, 12> edges{{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 5},
                                                        {2, 6}, {3, 7}, {4, 5}, {5, 6}, {6, 7}, {7, 4}}};
        const std::vector<NodeId> ends = nodes;
        for (const auto& edge : edges)
            nodes.push_back(midside(ends[static_cast<size_t>(edge[0])], ends[static_cast<size_t>(edge[1])]));
        add(ElementType::CHEXA20, 1, std::move(nodes));
    }
    const std::vector<Vec3> skewed_tet{{0., 0., 0.}, {2., 0.1, 0.}, {0.3, 1.5, 0.}, {0.2, 0.3, 1.2}};
    add(ElementType::CTETRA4, 1, corners(skewed_tet));
    {
        std::vector<NodeId> nodes = corners(skewed_tet);
        const std::array<std::array<int, 2>, 6> edges{{{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}}};
        const std::vector<NodeId> ends = nodes;
        for (const auto& edge : edges)
            nodes.push_back(midside(ends[static_cast<size_t>(edge[0])], ends[static_cast<size_t>(edge[1])]));
        add(ElementType::CTETRA10, 1, std::move(nodes));
    }
    add(ElementType::CPENTA6, 1, corners({{0., 0., 0.}, {2., 0.1, 0.}, {0.2, 1.4, 0.},
                                          {0.1, 0., 1.5}, {2.1, 0.2, 1.4}, {0.3, 1.5, 1.6}}));
    add(ElementType::CQUAD4, 2, corners({{0., 0., 0.}, {1., 0., 0.02}, {1.1, 1., 0.}, {0., 0.9, 0.03}}));
    add(ElementType::CTRIA3, 2, corners({{0., 0., 0.}, {1., 0.1, 0.}, {0.2, 1., 0.1}}));

    for (const ElementData& elem : model.elements) {
        const LocalKe ke = make_element(elem, model)->stiffness_matrix();
        const auto identity = local_dof_identity(elem, model);
        std::map<int, Eigen::Matrix<double, 6, 6>> blocks;
        for (Eigen::Index i = 0; i < ke.rows(); ++i)
            for (Eigen::Index j = 0; j < ke.cols(); ++j) {
                const auto [row_grid, row] = identity[static_cast<size_t>(i)];
                const auto [col_grid, col] = identity[static_cast<size_t>(j)];
                if (row_grid != col_grid)
                    continue;
                if (!blocks.contains(row_grid.value))
                    blocks[row_grid.value] = Eigen::Matrix<double, 6, 6>::Zero();
                blocks[row_grid.value](row, col) += ke(i, j);
            }
        const bool shell = elem.type == ElementType::CQUAD4 || elem.type == ElementType::CTRIA3;
        for (const auto& [id, block] : blocks)
            for (const int base : shell ? std::vector<int>{0, 3} : std::vector<int>{0}) {
                const Eigen::Matrix3d part = block.block<3, 3>(base, base);
                const Eigen::Vector3d values =
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(part, Eigen::EigenvaluesOnly).eigenvalues();
                EXPECT_GT(values(0), 1e-12 * values(2))
                    << "element " << elem.id.value << " grid " << id << " block " << base;
            }
    }
}

TEST(AutoSpc, NonFiniteStiffnessStopsTheRunRatherThanBeingHeld) {
    // Compared with NaN every direction would look singular, every component would be
    // held, and the run would report a solution of zeros. Poisson's ratio of 1 divides
    // by zero in the plane stress matrix.
    const Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nSPC = 1\nBEGIN BULK\nMAT1,1,1.0E7,,1.0\nPSHELL,1,1,0.1\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nCTRIA3,17,1,1,2,3\n"
        "SPC1,1,123456,1\nENDDATA\n");
    try {
        (void)analysis_spcs(model, model.analysis.subcases.at(0));
        FAIL() << "a non-finite stiffness matrix was accepted";
    } catch (const SolverError& error) {
        EXPECT_NE(std::string(error.what()).find("element 17"), std::string::npos) << error.what();
    }
}

TEST(AutoSpc, ModalBarWithoutTorsionHasNoSpuriousRigidMode) {
    // A massless bar with a mass and torsional inertia at its tip. With no J that
    // inertia would be a free rotation at zero frequency; held, the lowest modes are
    // the two equal bending modes at 3EI/(m L^3), and then the axial one at EA/(m L).
    Model model = BdfParser::parse_string(
        "SOL 103\nCEND\nSPC = 1\nMETHOD = 1\nBEGIN BULK\nEIGRL,1,,,3\nMAT1,1,1.0E7,,0.3\n" +
        kBarNoTorsion +
        "GRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCBAR,1,1,1,2,0.,1.,0.\nSPC1,1,123456,1\n"
        "CONM2,2,2,,2.\n,0.5\nENDDATA\n");
    ModalSolver solver(std::make_unique<SpectraEigensolverBackend>());
    const auto r = solver.solve(model).subcases.at(0);
    ASSERT_EQ(r.modes.size(), 3u);
    const double bending = 3.0 * kE * kI / (2.0 * 1000.0);
    const double axial = kE * 2.0 / (2.0 * 10.0);
    EXPECT_NEAR(r.modes[0].eigenvalue, bending, 1e-6 * bending);
    EXPECT_NEAR(r.modes[1].eigenvalue, bending, 1e-6 * bending);
    EXPECT_NEAR(r.modes[2].eigenvalue, axial, 1e-6 * axial);
}
