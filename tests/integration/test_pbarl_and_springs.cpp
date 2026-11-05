// tests/integration/test_pbarl_and_springs.cpp
// PBARL section constants, checked by integrating their rectangles and against the
// published MSC Nastran solution of the same bars, and CELAS force and stress
// recovery against spring networks solved by hand.

#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using namespace vibestran;

namespace {

SolverResults solve_all(const Model& model) {
    LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
    return solver.solve(model);
}

SubCaseResults solve_static(const std::string& deck) {
    return solve_all(BdfParser::parse_string(deck)).subcases.at(0);
}

const NodeDisplacement& disp(const SubCaseResults& r, int node) {
    const auto it = std::ranges::find(r.displacements, NodeId{node}, &NodeDisplacement::node);
    if (it == r.displacements.end())
        throw std::runtime_error(std::format("no displacement for point {}", node));
    return *it;
}

PBarL section(const std::string& type, const std::string& dims, const std::string& tail = "") {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nBEGIN BULK\nPBARL,1,1,," + type + ",,,,,+\n+," + dims + tail + "\nENDDATA\n");
    return std::get<PBarL>(model.properties.at(PropertyId{1}));
}

/// An axis-aligned rectangle of a section, with y and z in any origin.
struct Rect {
    double y0, y1, z0, z1;
};

/// Checks A, I1, I2 and I12 = 0 by the parallel axis theorem over rectangles, and
/// each recovery point, given about that origin, about the resulting centroid.
void expect_matches_rectangles(const PBarL& p, const std::vector<Rect>& rects,
                               const std::array<std::array<double, 2>, 4>& points) {
    double area = 0.0, first_y = 0.0, first_z = 0.0;
    for (const Rect& r : rects) {
        const double a = (r.y1 - r.y0) * (r.z1 - r.z0);
        area += a;
        first_y += a * 0.5 * (r.y0 + r.y1);
        first_z += a * 0.5 * (r.z0 + r.z1);
    }
    const double cy = first_y / area, cz = first_z / area;
    double i1 = 0.0, i2 = 0.0, i12 = 0.0;
    for (const Rect& r : rects) {
        const double by = r.y1 - r.y0, bz = r.z1 - r.z0, a = by * bz;
        const double dy = 0.5 * (r.y0 + r.y1) - cy, dz = 0.5 * (r.z0 + r.z1) - cz;
        i1 += bz * by * by * by / 12.0 + a * dy * dy;
        i2 += by * bz * bz * bz / 12.0 + a * dz * dz;
        i12 += a * dy * dz;
    }
    EXPECT_NEAR(p.A, area, 1e-12 * area);
    EXPECT_NEAR(p.I1, i1, 1e-12 * i1);
    EXPECT_NEAR(p.I2, i2, 1e-12 * i2);
    EXPECT_NEAR(i12, 0.0, 1e-12 * i1) << "the formulas assume a symmetric section";
    for (size_t k = 0; k < 4; ++k) {
        EXPECT_NEAR(p.recovery_points[k][0], points[k][0] - cy, 1e-12) << "point " << k << " y";
        EXPECT_NEAR(p.recovery_points[k][1], points[k][1] - cz, 1e-12) << "point " << k << " z";
    }
}

} // namespace

// ── Section constants ─────────────────────────────────────────────────────────

TEST(PbarlSection, ChannelMatchesItsRectanglesWithTheWebOnTheMinusZSide) {
    const double b = 0.13, h = 0.21, tw = 0.017, tf = 0.011;
    const PBarL p = section("CHAN", "0.13,0.21,0.017,0.011");
    expect_matches_rectangles(
        p,
        {{-h / 2, -h / 2 + tf, 0, b}, {h / 2 - tf, h / 2, 0, b}, {-h / 2 + tf, h / 2 - tf, 0, tw}},
        {{{h / 2, b}, {-h / 2, b}, {-h / 2, 0.0}, {h / 2, 0.0}}});
    const double web = h - 2 * tf;
    EXPECT_NEAR(p.J, (2 * b * tf * tf * tf + web * tw * tw * tw) / 3.0, 1e-18);
}

TEST(PbarlSection, TeeMatchesItsRectanglesWithTheFlangeAtPlusY) {
    const double b = 0.14, h = 0.19, tf = 0.013, tw = 0.009;
    const PBarL p = section("T", "0.14,0.19,0.013,0.009");
    expect_matches_rectangles(
        p, {{h - tf, h, -b / 2, b / 2}, {0, h - tf, -tw / 2, tw / 2}},
        {{{h, 0.0}, {h, b / 2}, {0.0, 0.0}, {h, -b / 2}}});
    EXPECT_NEAR(p.J, (b * tf * tf * tf + (h - tf) * tw * tw * tw) / 3.0, 1e-18);
}

TEST(PbarlSection, I1MatchesItsRectanglesWithOutstandsAndClearWebDepth) {
    const double outstands = 0.09, tw = 0.012, web = 0.18, h = 0.2;
    const double b = outstands + tw, tf = 0.5 * (h - web);
    const PBarL p = section("I1", "0.09,0.012,0.18,0.2");
    expect_matches_rectangles(
        p,
        {{-h / 2, -web / 2, -b / 2, b / 2}, {web / 2, h / 2, -b / 2, b / 2}, {-web / 2, web / 2, -tw / 2, tw / 2}},
        {{{h / 2, b / 2}, {-h / 2, b / 2}, {-h / 2, -b / 2}, {h / 2, -b / 2}}});
    EXPECT_NEAR(p.J, (2 * b * tf * tf * tf + web * tw * tw * tw) / 3.0, 1e-18);
}

TEST(PbarlSection, BoxMatchesItsWallsAndBredtsFormula) {
    const double b = 0.2, h = 0.3, tf = 0.01, tw = 0.02;
    const PBarL p = section("BOX", "0.2,0.3,0.01,0.02");
    expect_matches_rectangles(
        p,
        {{-h / 2, -h / 2 + tf, -b / 2, b / 2},
         {h / 2 - tf, h / 2, -b / 2, b / 2},
         {-h / 2 + tf, h / 2 - tf, -b / 2, -b / 2 + tw},
         {-h / 2 + tf, h / 2 - tf, b / 2 - tw, b / 2}},
        {{{h / 2, b / 2}, {-h / 2, b / 2}, {-h / 2, -b / 2}, {h / 2, -b / 2}}});
    const double enclosed = (b - tw) * (h - tf);
    EXPECT_NEAR(p.J, 4 * enclosed * enclosed / (2 * (b - tw) / tf + 2 * (h - tf) / tw), 1e-14);
    // Walls of one thickness give the familiar 2 t b^2 h^2 / (b + h) on the centreline.
    const PBarL even = section("BOX", "0.2,0.3,0.01,0.01");
    const double bm = 0.19, hm = 0.29;
    EXPECT_NEAR(even.J, 2 * 0.01 * bm * bm * hm * hm / (bm + hm), 1e-14);
}

TEST(PbarlSection, RodAndTubeDimensionsAreRadii) {
    const PBarL rod = section("ROD", "0.1");
    EXPECT_NEAR(rod.A, std::numbers::pi * 0.01, 1e-14);
    EXPECT_NEAR(rod.I1, std::numbers::pi * 1e-4 / 4.0, 1e-16);
    EXPECT_NEAR(rod.J, std::numbers::pi * 1e-4 / 2.0, 1e-16);
    EXPECT_EQ(rod.recovery_points[0][0], 0.1);

    const PBarL tube = section("TUBE", "0.04,0.01");
    EXPECT_NEAR(tube.A, std::numbers::pi * (0.0016 - 0.0001), 1e-14);
    EXPECT_NEAR(tube.I2, std::numbers::pi * (2.56e-6 - 1e-8) / 4.0, 1e-18);
    EXPECT_NEAR(tube.J, std::numbers::pi * (2.56e-6 - 1e-8) / 2.0, 1e-18);
    EXPECT_EQ(tube.recovery_points[1][1], 0.04);
}

TEST(PbarlSection, ValueAfterTheDimensionsIsNsmAndZeroPaddingIsAccepted) {
    EXPECT_EQ(section("BAR", "0.2,0.15", ",0.7").nsm, 0.7);
    const PBarL padded = section("BAR", "0.2,0.15,0.0,0.0");
    EXPECT_EQ(padded.nsm, 0.0);
    EXPECT_EQ(padded.dimensions, (std::vector<double>{0.2, 0.15}));
    EXPECT_EQ(section("ROD", "0.1,0.7,0.0,0.0").nsm, 0.7);
    EXPECT_EQ(section("CHAN", "0.1,0.1,0.01,0.01", ",0.7").nsm, 0.7);
    EXPECT_THROW(section("BAR", "0.2,0.15,0.0,0.5"), ParseError);
}

TEST(PbarlSection, RejectsImpossibleDimensions) {
    EXPECT_THROW(section("CHAN", "0.1,0.1,0.01"), ParseError);
    EXPECT_THROW(section("CHAN", "0.1,0.1,0.1,0.01"), ParseError);
    EXPECT_THROW(section("CHAN", "0.1,0.1,0.01,0.05"), ParseError);
    EXPECT_THROW(section("T", "0.1,0.1,0.1,0.01"), ParseError);
    EXPECT_THROW(section("T", "0.1,0.1,0.01,0.1"), ParseError);
    EXPECT_THROW(section("I1", "0.1,0.01,0.2,0.2"), ParseError);
    EXPECT_THROW(section("BOX", "0.1,0.1,0.05,0.01"), ParseError);
    EXPECT_THROW(section("BOX", "0.1,0.1,0.01,0.05"), ParseError);
    EXPECT_THROW(section("TUBE", "0.04,0.04"), ParseError);
    EXPECT_THROW(section("BAR", "0.2,-0.15"), ParseError);
    EXPECT_THROW(section("CROSS", "0.1,0.1,0.01,0.01"), ParseError);
}

// ── The bars ──────────────────────────────────────────────────────────────────

namespace {

/// One bar of MSC Nastran's solution of nas_s30_beam_section_shapes_no_shear_2
/// (the suite's reference_msc), a cantilever of L = 2.5 along -Z under an axial
/// force, M1 = 2, M2 = 3 and a torque of 1.
struct MscBar {
    int eid;
    std::string card;
    double t1, t2, r1, r2;
    std::array<double, 4> stress;
    double axial;
};

const std::vector<MscBar> kMscBars = {
    {3, "CHAN,,,,,+\n+,0.1,0.1,0.01,0.01", -1.656082e-05, 6.954748e-06, 5.563798e-06, 1.324865e-05,
     {-8.887700e+04, -4.436661e+04, 6.162262e+04, 1.711223e+04}, -2.222222e+03},
    {6, "T,,,,,+\n+,0.1,0.11,0.01,0.01", -5.569308e-05, 1.327434e-05, 1.061947e-05, 4.455446e-05,
     {-2.761062e+04, -2.058285e+05, 6.584071e+04, 1.506072e+05}, -3.111111e+03},
    {5, "I1,,,,,+\n+,0.09,0.01,0.18,0.2", -2.787412e-05, 1.363041e-06, 1.090433e-06, 2.229930e-05,
     {-9.792066e+04, -8.047373e+04, 9.792066e+04, 8.047373e+04}, -1.637427e+03},
    {8, "BAR,,,,,+\n+,0.2,0.15,0.0,0.0", -4.687499e-07, 5.555555e-07, 4.444444e-07, 3.750000e-07,
     {-5.666666e+03, -3.333332e+02, 5.666666e+03, 3.333334e+02}, -2.074074e+02},
    {4, "BOX,,,,,+\n+,0.2,0.3,0.01,0.01", -7.333385e-07, 2.588633e-07, 2.070907e-07, 5.866708e-07,
     {-7.178453e+03, -2.208277e+03, 7.178455e+03, 2.208279e+03}, -6.481481e+02},
    {7, "ROD,,,,,+\n+,0.1,0.0,0.0,0.0", -5.968310e-07, 3.978873e-07, 3.183099e-07, 4.774648e-07,
     {-2.546479e+03, -3.819718e+03, 2.546479e+03, 3.819718e+03}, -1.980595e+02},
    {1, "TUBE,,,,,+\n+,0.04,0.01,0.0,0.0", -2.340514e-05, 1.560343e-05, 1.248274e-05, 1.872411e-05,
     {-3.994477e+04, -5.991716e+04, 3.994477e+04, 5.991716e+04}, -1.320397e+03},
};

} // namespace

TEST(PbarlBar, MatchesTheMscNastranSolutionOfEverySection) {
    for (const MscBar& bar : kMscBars) {
        SCOPED_TRACE(bar.card);
        const auto r = solve_static(
            "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nDISP = ALL\nSTRESS = ALL\nBEGIN BULK\n"
            "MAT1,1,2.0E11,1.0E11,0.0\nGRID,1,,0.,0.,2.5\nGRID,2,,0.,0.,0.\n"
            "PBARL,1,1,," + bar.card + "\n" +
            std::format("CBAR,{},1,2,1,0.,1.,0.\n", bar.eid) +
            "SPC1,1,123456,1\nFORCE,1,2,0,6.22222222222,0.,0.,1.\nMOMENT,1,2,0,1.,2.,3.,1.\nENDDATA\n");
        const auto& d = disp(r, 2).d;
        EXPECT_NEAR(d[0], bar.t1, 2e-6 * std::abs(bar.t1));
        EXPECT_NEAR(d[1], bar.t2, 2e-6 * std::abs(bar.t2));
        EXPECT_NEAR(d[3], bar.r1, 2e-6 * std::abs(bar.r1));
        EXPECT_NEAR(d[4], bar.r2, 2e-6 * std::abs(bar.r2));
        ASSERT_EQ(r.line_stresses.size(), 1U);
        const auto& stress = r.line_stresses[0];
        for (size_t k = 0; k < 4; ++k) {
            EXPECT_NEAR(stress.end_a.s[k], bar.stress[k], 2e-6 * std::abs(bar.stress[k])) << "point " << k;
            EXPECT_NEAR(stress.end_b.s[k], bar.stress[k], 2e-6 * std::abs(bar.stress[k])) << "point " << k;
        }
        EXPECT_NEAR(stress.end_a.axial, bar.axial, 2e-6 * std::abs(bar.axial));
    }
}

TEST(PbarlBar, TorsionFollowsTheThinWallConstants) {
    // T L / (G J), with L = 2.5, G = 1e11 and T = 1. The constants of MSC Nastran's
    // numerical conversion differ by up to 2 percent.
    const auto twist = [](const std::string& card) {
        const auto r = solve_static(
            "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nBEGIN BULK\nMAT1,1,2.0E11,1.0E11,0.0\n"
            "GRID,1,,0.,0.,2.5\nGRID,2,,0.,0.,0.\nPBARL,1,1,," + card + "\nCBAR,1,1,2,1,0.,1.,0.\n"
            "SPC1,1,123456,1\nMOMENT,1,2,0,1.,0.,0.,1.\nENDDATA\n");
        return disp(r, 2).d[5];
    };
    const auto expected = [](const double j) { return 2.5 / (1.0e11 * j); };
    EXPECT_NEAR(twist("CHAN,,,,,+\n+,0.1,0.1,0.01,0.01"), expected(0.28e-6 / 3.0), 1e-9);
    EXPECT_NEAR(twist("T,,,,,+\n+,0.1,0.11,0.01,0.01"), expected(0.2e-6 / 3.0), 1e-9);
    EXPECT_NEAR(twist("I1,,,,,+\n+,0.09,0.01,0.18,0.2"), expected(0.38e-6 / 3.0), 1e-9);
    EXPECT_NEAR(twist("BOX,,,,,+\n+,0.2,0.3,0.01,0.01"), expected(2 * 0.01 * 0.19 * 0.19 * 0.29 * 0.29 / 0.48), 1e-12);
}

TEST(PbarlSection, ShearFactorsGiveTheAreaOfTheWallsAlongTheLoad) {
    EXPECT_DOUBLE_EQ(section("BAR", "0.2,0.15").K1, 5.0 / 6.0);
    EXPECT_DOUBLE_EQ(section("BAR", "0.2,0.15").K2, 5.0 / 6.0);
    EXPECT_DOUBLE_EQ(section("ROD", "0.1").K1, 6.0 / 7.0);
    // A tube tends to the thick rod as the bore closes and to 1/2 as the wall thins.
    EXPECT_NEAR(section("TUBE", "0.1,1e-6").K1, 6.0 / 7.0, 1e-6);
    EXPECT_NEAR(section("TUBE", "0.1,0.09999").K2, 0.5, 1e-3);

    const PBarL chan = section("CHAN", "0.13,0.21,0.017,0.011");
    EXPECT_NEAR(chan.K1 * chan.A, (0.21 - 2 * 0.011) * 0.017, 1e-14);
    EXPECT_NEAR(chan.K2 * chan.A, (5.0 / 6.0) * 2 * 0.13 * 0.011, 1e-14);
    const PBarL tee = section("T", "0.14,0.19,0.013,0.009");
    EXPECT_NEAR(tee.K1 * tee.A, (0.19 - 0.013) * 0.009, 1e-14);
    EXPECT_NEAR(tee.K2 * tee.A, (5.0 / 6.0) * 0.14 * 0.013, 1e-14);
    const PBarL i1 = section("I1", "0.09,0.012,0.18,0.2");
    EXPECT_NEAR(i1.K1 * i1.A, 0.18 * 0.012, 1e-14);
    EXPECT_NEAR(i1.K2 * i1.A, (5.0 / 6.0) * 2 * 0.102 * 0.01, 1e-14);
    const PBarL box = section("BOX", "0.2,0.3,0.01,0.02");
    EXPECT_NEAR(box.K1 * box.A, 2 * 0.02 * 0.28, 1e-14);
    EXPECT_NEAR(box.K2 * box.A, 2 * 0.01 * 0.16, 1e-14);
}

TEST(PbarlBar, TipShearAddsTheShearFlexibilityToBending) {
    // A BAR of b = 0.2, h = 0.15 and L = 2.5 under tip forces along element y and z.
    // Both planes follow F L^3 / (3 E I) + F L / (K A G) with K = 5/6.
    const auto r = solve_static(
        "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nDISP = ALL\nBEGIN BULK\nMAT1,1,2.0E11,1.0E11,0.0\n"
        "GRID,1,,0.,0.,2.5\nGRID,2,,0.,0.,0.\nPBARL,1,1,,BAR,,,,,+\n+,0.2,0.15\n"
        "CBAR,1,1,2,1,0.,1.,0.\nSPC1,1,123456,1\nFORCE,1,2,0,1000.,1.,1.,0.\nENDDATA\n");
    const double L = 2.5, E = 2.0e11, G = 1.0e11, A = 0.03, F = 1000.0;
    const double i_about_z = 0.2 * std::pow(0.15, 3) / 12.0, i_about_y = 0.15 * std::pow(0.2, 3) / 12.0;
    const double shear = F * L / ((5.0 / 6.0) * A * G);
    EXPECT_NEAR(disp(r, 2).d[1], F * L * L * L / (3 * E * i_about_z) + shear, 1e-12);
    EXPECT_NEAR(disp(r, 2).d[0], F * L * L * L / (3 * E * i_about_y) + shear, 1e-12);
}

TEST(PbarlBar, GravityCountsTheSectionAreaAndNsm) {
    // CHAN of area 0.0028, rho = 1000 and NSM = 0.7 on L = 2.5 under g = 10.
    const auto r = solve_static(
        "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nSPCF = ALL\nBEGIN BULK\nMAT1,1,2.0E11,1.0E11,0.0,1000.\n"
        "GRID,1,,0.,0.,2.5\nGRID,2,,0.,0.,0.\nPBARL,1,1,,CHAN,,,,,+\n+,0.1,0.1,0.01,0.01,0.7\n"
        "CBAR,1,1,2,1,0.,1.,0.\nSPC1,1,123456,1\nGRAV,1,0,10.,0.,0.,-1.\nENDDATA\n");
    ASSERT_EQ(r.spc_forces.size(), 1U);
    EXPECT_NEAR(r.spc_forces[0].f[2], (1000.0 * 0.0028 + 0.7) * 2.5 * 10.0, 1e-9);
}

// ── Spring recovery ───────────────────────────────────────────────────────────

namespace {

const SpringResult& spring(const SubCaseResults& r, int eid) {
    const auto it = std::ranges::find(r.spring_results, ElementId{eid}, &SpringResult::eid);
    if (it == r.spring_results.end())
        throw std::runtime_error(std::format("no result for spring {}", eid));
    return *it;
}

std::string spring_deck(const std::string& bulk) {
    return "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nDISP = ALL\nFORCE = ALL\nSTRESS = ALL\nBEGIN BULK\n" +
           bulk + "ENDDATA\n";
}

} // namespace

TEST(CelasRecovery, Celas2TakesForceAndStressFromItsCard) {
    // Tension load on grid 2 of a spring from a fixed grid 1: F = k (u1 - u2) = -P.
    const auto r = solve_static(spring_deck(
        "GRID,1,,0.,0.,0.,,123456\nGRID,2,,1.,0.,0.,,12456\nCELAS2,1,123.4,1,3,2,3,,5.0\n"
        "FORCE,1,2,,765.4,0.,0.,1.\n"));
    EXPECT_EQ(spring(r, 1).type, ElementType::CELAS2);
    EXPECT_NEAR(spring(r, 1).force, -765.4, 1e-9);
    EXPECT_NEAR(spring(r, 1).stress, 5.0 * -765.4, 1e-8);
}

TEST(CelasRecovery, GeDoesNotChangeAStaticSolution) {
    const auto with_ge = [](const std::string& ge) {
        return solve_static(spring_deck(
            "GRID,1,,0.,0.,0.,,123456\nGRID,2,,1.,0.,0.,,12456\nCELAS2,1,123.4,1,3,2,3," + ge +
            ",5.0\nFORCE,1,2,,765.4,0.,0.,1.\n"));
    };
    const auto damped = with_ge("0.04");
    const auto undamped = with_ge("");
    EXPECT_EQ(disp(damped, 2).d, disp(undamped, 2).d);
    EXPECT_EQ(spring(damped, 1).force, spring(undamped, 1).force);
    EXPECT_EQ(spring(damped, 1).stress, spring(undamped, 1).stress);
    EXPECT_THROW(with_ge("abc"), ParseError);
}

TEST(CelasRecovery, Celas1TakesItsStressCoefficientFromPelas) {
    const auto r = solve_static(spring_deck(
        "PELAS,7,200.,0.02,-3.0\nGRID,1,,0.,0.,0.,,123456\nGRID,2,,1.,0.,0.,,23456\n"
        "CELAS1,4,7,1,1,2,1\nFORCE,1,2,,50.,1.,0.,0.\n"));
    EXPECT_EQ(spring(r, 4).type, ElementType::CELAS1);
    EXPECT_NEAR(spring(r, 4).force, -50.0, 1e-10);
    EXPECT_NEAR(spring(r, 4).stress, 150.0, 1e-10);
}

TEST(CelasRecovery, SeriesAndParallelSpringsShareTheLoadByStiffness) {
    // Springs 1 (k = 100) and 2 (k = 300) in parallel from ground to grid 1, then
    // spring 3 (k = 50, grounded at its second end) in series to the loaded grid 2.
    const auto r = solve_static(spring_deck(
        "GRID,1,,0.,0.,0.,,23456\nGRID,2,,1.,0.,0.,,23456\n"
        "CELAS2,1,100.,1,1\nCELAS2,2,300.,,,1,1,,2.0\nCELAS2,3,50.,1,1,2,1\n"
        "FORCE,1,2,,40.,1.,0.,0.\n"));
    const double u1 = 40.0 / 400.0, u2 = u1 + 40.0 / 50.0;
    EXPECT_NEAR(disp(r, 1).d[0], u1, 1e-12);
    EXPECT_NEAR(disp(r, 2).d[0], u2, 1e-12);
    EXPECT_NEAR(spring(r, 1).force, 100.0 * u1, 1e-10);
    // Ground is the first end here, so the sign flips with the end the grid is on.
    EXPECT_NEAR(spring(r, 2).force, -300.0 * u1, 1e-10);
    EXPECT_NEAR(spring(r, 2).stress, 2.0 * -300.0 * u1, 1e-10);
    EXPECT_NEAR(spring(r, 3).force, 50.0 * (u1 - u2), 1e-10);
    EXPECT_EQ(spring(r, 1).stress, 0.0);
}

TEST(CelasRecovery, ScalarPointSpringsReportAsCelas1And2) {
    // Chain ground - s1 - s2 under SLOAD: CELAS3 (PELAS with S) and CELAS4 (no S).
    const auto r = solve_static(spring_deck(
        "SPOINT,10,20\nPELAS,3,10.,,4.0\nCELAS3,1,3,10\nCELAS4,2,20.,10,20\nSLOAD,1,20,6.\n"));
    EXPECT_EQ(spring(r, 1).type, ElementType::CELAS1);
    EXPECT_NEAR(spring(r, 1).force, 10.0 * 0.6, 1e-10);
    EXPECT_NEAR(spring(r, 1).stress, 4.0 * 6.0, 1e-10);
    EXPECT_EQ(spring(r, 2).type, ElementType::CELAS2);
    EXPECT_NEAR(spring(r, 2).force, -6.0, 1e-10);
    EXPECT_EQ(spring(r, 2).stress, 0.0);
}

TEST(CelasRecovery, EnforcedDisplacementStretchesTheSpring) {
    // Springs of 100 and 300 from an enforced grid through grid 2 to ground.
    const auto r = solve_static(spring_deck(
        "GRID,1,,0.,0.,0.,,23456\nGRID,2,,1.,0.,0.,,23456\nCELAS2,1,100.,1,1,2,1\n"
        "CELAS2,2,300.,2,1\nSPC,1,1,1,0.4\n"));
    const double u2 = 0.4 * 100.0 / 400.0;
    EXPECT_NEAR(disp(r, 2).d[0], u2, 1e-12);
    EXPECT_NEAR(spring(r, 1).force, 100.0 * (0.4 - u2), 1e-10);
    EXPECT_NEAR(spring(r, 2).force, 300.0 * u2, 1e-10);
}

TEST(CelasRecovery, FullyPrescribedModelHasNothingToSolve) {
    // Both ends of the spring are enforced, so there is no free DOF at all.
    const auto r = solve_static(spring_deck(
        "GRID,1,,0.,0.,0.,,13456\nGRID,2,,0.,0.,0.,,13456\nPELAS,1,3000.,,1.5\nCELAS1,1,1,1,2,2,2\n"
        "SPC,1,1,2,-0.021\nSPC,1,2,2,0.035\n"));
    EXPECT_NEAR(disp(r, 1).d[1], -0.021, 1e-15);
    EXPECT_NEAR(disp(r, 2).d[1], 0.035, 1e-15);
    EXPECT_NEAR(spring(r, 1).force, 3000.0 * (-0.021 - 0.035), 1e-10);
    EXPECT_NEAR(spring(r, 1).stress, 1.5 * 3000.0 * (-0.021 - 0.035), 1e-10);
}

TEST(CelasRecovery, AModelWithoutElementsIsStillAnError) {
    // Only a spring's DOFs may be fully prescribed: this deck has lost its elements,
    // so answering with zeros would hide that.
    EXPECT_THROW(solve_static(spring_deck("GRID,1,,0.,0.,0.\nSPC,1,1,1,0.\nCQUAD8,1,1,1,1,1,1\n")),
                 SolverError);
}

TEST(CelasRecovery, ComponentOnACdGridStretchesAlongItsFrame) {
    // T1 of frame 5 points along basic (1, 1, 0) / sqrt(2). A load F along it moves
    // the grid by F / k along that axis, and the force is -F.
    const auto r = solve_static(spring_deck(
        "CORD2R,5,0,0.,0.,0.,0.,0.,1.\n,1.,1.,0.\nGRID,2,,1.,0.,0.,5,23456\nCELAS2,1,10.,2,1\n"
        "FORCE,1,2,0,20.,1.,1.,0.\n"));
    const double F = 20.0 * std::numbers::sqrt2;
    EXPECT_NEAR(disp(r, 2).d[0], F / 10.0, 1e-10);
    EXPECT_NEAR(spring(r, 1).force, F, 1e-10);
}

TEST(CelasRecovery, F06ListsFiveToARowInTheParserColumns) {
    // Seven grounded springs of k = i, total 28, with unit stretch give force i and
    // stress 10 i.
    std::string bulk = "GRID,1,,0.,0.,0.,,23456\nFORCE,1,1,,28.,1.,0.,0.\n";
    for (int i = 1; i <= 7; ++i)
        bulk += std::format("CELAS2,{},{}.,1,1,,,,10.\n", i, i);
    Model model = BdfParser::parse_string(spring_deck(bulk));
    std::ostringstream out;
    F06Writer::write(solve_all(model), model, out);
    const std::string f06 = out.str();

    const auto number = [](const std::string& line, const int col, const int width) {
        return std::stod(line.substr(static_cast<size_t>(col - 1), static_cast<size_t>(width)));
    };
    std::istringstream lines(f06);
    std::string line;
    std::vector<std::string> force_rows, stress_rows;
    std::vector<std::string>* target = nullptr;
    int skip = 0;
    while (std::getline(lines, line)) {
        if (line.find("E N G I N E E R I N G   F O R C E S") != std::string::npos) {
            target = &force_rows;
            skip = 3;
        } else if (line.find("E L E M E N T   S T R E S S E S") != std::string::npos) {
            target = &stress_rows;
            skip = 3;
        } else if (skip > 0) {
            --skip;
        } else if (target != nullptr) {
            if (line.empty()) target = nullptr;
            else target->push_back(line);
        }
    }
    ASSERT_EQ(force_rows.size(), 2U);
    ASSERT_EQ(stress_rows.size(), 2U);
    EXPECT_NE(f06.find("F O R   E L E M E N T   T Y P E   E L A S 2"), std::string::npos);
    for (int k = 0; k < 5; ++k) {
        EXPECT_EQ(number(force_rows[0], 17 + 22 * k, 8), k + 1);
        EXPECT_NEAR(number(force_rows[0], 26 + 22 * k, 13), k + 1, 1e-6);
        EXPECT_EQ(number(stress_rows[0], 2 + 23 * k, 8), k + 1);
        EXPECT_NEAR(number(stress_rows[0], 11 + 23 * k, 13), 10.0 * (k + 1), 1e-5);
    }
    EXPECT_EQ(number(force_rows[1], 17 + 22, 8), 7);
    EXPECT_NEAR(number(stress_rows[1], 11 + 23, 13), 70.0, 1e-5);
}
