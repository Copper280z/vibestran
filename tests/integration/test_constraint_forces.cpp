// Constraint forces at supported RBE/MPC attachments, with analytical loads.
#include "core/coord_sys.hpp"
#include "io/bdf_parser.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <sstream>
#include <string>
#include <tuple>

using namespace vibestran;

namespace {

SubCaseResults solve_forces(const Model& model) {
    LinearStaticSolver solver(std::make_unique<LdltSolverBackend>());
    return solver.solve(model).subcases.at(0);
}

const SpcForce* force_at(const std::vector<SpcForce>& forces, int node) {
    const auto it = std::find_if(forces.begin(), forces.end(),
        [node](const SpcForce& f) { return f.node == NodeId{node}; });
    return it == forces.end() ? nullptr : &*it;
}

void expect_grid_balance(const SubCaseResults& result, double tolerance = 1e-9) {
    ASSERT_FALSE(result.gp_forces.empty());
    for (const auto& grid : result.gp_forces)
        for (int d = 0; d < 6; ++d)
            EXPECT_NEAR(grid.totals[static_cast<size_t>(d)], 0.0, tolerance)
                << "grid " << grid.node.value << " component " << d + 1;
}

std::array<double, 6> resultant(const Model& model, const std::vector<SpcForce>& forces) {
    std::array<double, 6> sum{};
    for (const auto& force : forces) {
        Vec3 f{force.f[0], force.f[1], force.f[2]};
        Vec3 m{force.f[3], force.f[4], force.f[5]};
        const auto& grid = model.node(force.node);
        if (force.coord != CoordId{0}) {
            const auto axes = rotation_matrix(model.coord_systems.at(force.coord), grid.position);
            f = apply_rotation(axes, f);
            m = apply_rotation(axes, m);
        }
        m = m + grid.position.cross(f);
        sum[0] += f.x; sum[1] += f.y; sum[2] += f.z;
        sum[3] += m.x; sum[4] += m.y; sum[5] += m.z;
    }
    return sum;
}

constexpr const char* equal_patch = R"(
SOL 101
CEND
SPC=1
LOAD=2
SPCFORCES=ALL
MPCFORCES=ALL
GPFORCE=ALL
BEGIN BULK
GRID,1,,0.,0.,0.
GRID,2,,2.,0.,0.
GRID,3,,2.,2.,0.
GRID,4,,0.,2.,0.
GRID,5,,1.,1.,2.
GRID,12,,2.,0.,0.
GRID,13,,2.,2.,0.
GRID,14,,0.,2.,0.
CELAS2,3,1000.,2,3,12,3
CELAS2,6,1000.,3,3,13,3
CELAS2,9,1000.,4,3,14,3
RBE3,20,,5,123456,1.,123,1,2
+,3,4
SPC1,1,123,1,12,13,14
SPC1,1,12,2,3,4
)";

} // namespace

TEST(ConstraintForces, Rbe3RecoversDirectSupportShareForForceAndGravity) {
    for (const bool gravity : {false, true}) {
        SCOPED_TRACE(gravity);
        const Model model = BdfParser::parse_string(std::string(equal_patch) +
            (gravity ? "CONM2,10,5,0,1.\nGRAV,2,0,10.,0.,0.,-1.\n"
                     : "FORCE,2,5,0,10.,0.,0.,-1.\n") + "ENDDATA\n");
        const auto result = solve_forces(model);
        for (int nid : {1, 12, 13, 14}) {
            const auto* reaction = force_at(result.spc_forces, nid);
            ASSERT_NE(reaction, nullptr);
            EXPECT_NEAR(reaction->f[2], 2.5, 1e-12) << nid;
        }
        const auto* attachment = force_at(result.mpc_forces, 1);
        ASSERT_NE(attachment, nullptr);
        EXPECT_NEAR(attachment->f[2], -2.5, 1e-12);
        const auto* reference = force_at(result.mpc_forces, 5);
        ASSERT_NE(reference, nullptr);
        EXPECT_NEAR(reference->f[2], 10.0, 1e-12);
        const auto total = resultant(model, result.spc_forces);
        const std::array<double, 6> expected{0., 0., 10., 10., -10., 0.};
        for (size_t d = 0; d < 6; ++d)
            EXPECT_NEAR(total[d], expected[d], 1e-11);
        expect_grid_balance(result);
        for (const auto& displacement : result.displacements)
            if (displacement.node == NodeId{2} || displacement.node == NodeId{3} ||
                displacement.node == NodeId{4})
                EXPECT_NEAR(displacement.d[2], -0.0025, 1e-14);
    }
}

TEST(ConstraintForces, OutputRequestsDoNotChangeRecoveredForces) {
    const Model original = BdfParser::parse_string(std::string(equal_patch) +
        "FORCE,2,5,0,10.,0.,0.,-1.\nENDDATA\n");
    const auto baseline = solve_forces(original);
    // All seven nonempty combinations exercise both internal dependencies
    // and the public contract that unrequested outputs remain empty.
    for (int mask = 1; mask < 8; ++mask) {
        SCOPED_TRACE(mask);
        Model model = original;
        auto& sc = model.analysis.subcases[0];
        sc.spc_force_print = (mask & 1) != 0;
        sc.mpc_force_print = (mask & 2) != 0;
        sc.gpforce_print = (mask & 4) != 0;
        const auto result = solve_forces(model);
        for (const auto& [requested, actual, expected] : {
                 std::tuple{sc.spc_force_print, &result.spc_forces, &baseline.spc_forces},
                 std::tuple{sc.mpc_force_print, &result.mpc_forces, &baseline.mpc_forces}}) {
            if (!requested) {
                EXPECT_TRUE(actual->empty());
                continue;
            }
            ASSERT_EQ(actual->size(), expected->size());
            for (size_t i = 0; i < actual->size(); ++i)
                for (size_t d = 0; d < 6; ++d)
                    EXPECT_NEAR((*actual)[i].f[d], (*expected)[i].f[d], 1e-12);
        }
        if (sc.gpforce_print)
            expect_grid_balance(result);
        else
            EXPECT_TRUE(result.gp_forces.empty());
    }
}

namespace {

std::string weighted_patch() {
    // 3D springs attach three corners to supports; the fourth corner is
    // supported directly. The reference is offset from the weighted center.
    std::string deck = R"(
SOL 101
CEND
SPC=1
LOAD=2
SPCFORCES=ALL
MPCFORCES=ALL
GPFORCE=ALL
BEGIN BULK
GRID,1,,0.,0.,0.
GRID,2,,2.,0.,0.
GRID,3,,2.,2.,0.
GRID,4,,0.,2.,0.
GRID,5,,0.75,1.25,3.
GRID,12,,2.,0.,0.
GRID,13,,2.,2.,0.
GRID,14,,0.,2.,0.
RBE3,20,,5,123456,1.,123,1,2.
+,123,2,3.,123,3,4.,123,4
SPC1,1,123,1,12,13,14
)";
    for (int n = 2; n <= 4; ++n)
        for (int d = 1; d <= 3; ++d)
            deck += std::format("CELAS2,{},{},{},{},{},{}\n", 3*n+d, 1000., n, d, n+10, d);
    return deck;
}

} // namespace

TEST(ConstraintForces, OffsetWeightedRbe3PreservesNodalLoadsAndSixResultants) {
    // Choose an exactly representable weighted nodal force field
    // f_i = w_i * (a + omega x x_i), a=(1,-2,3), omega=(2,1,-1).
    // It satisfies weighted least-squares stationarity. Its known corner
    // forces are below, with total (24,-30,48) and moment (86,-34,-82)
    // about the origin. The reference moment accounts for its offset.
    const std::array<std::array<double, 3>, 4> corner_forces{{
        {1., -2., 3.}, {2., -8., 2.}, {9., -12., 15.}, {12., -8., 28.}}};
    const Model model = BdfParser::parse_string(weighted_patch() + R"(
FORCE,2,5,0,1.,24.,-30.,48.
MOMENT,2,5,0,1.,-64.,-70.,-29.5
ENDDATA
)");
    const auto result = solve_forces(model);
    for (int n = 1; n <= 4; ++n) {
        const auto* mpc = force_at(result.mpc_forces, n);
        const auto* spc = force_at(result.spc_forces, n == 1 ? 1 : n+10);
        ASSERT_NE(mpc, nullptr);
        ASSERT_NE(spc, nullptr);
        for (size_t d = 0; d < 3; ++d) {
            EXPECT_NEAR(mpc->f[d], corner_forces[static_cast<size_t>(n-1)][d], 1e-10);
            EXPECT_NEAR(spc->f[d], -corner_forces[static_cast<size_t>(n-1)][d], 1e-10);
        }
    }
    for (const auto& u : result.displacements)
        if (u.node.value >= 2 && u.node.value <= 4)
            for (size_t d = 0; d < 3; ++d)
                EXPECT_NEAR(u.d[d], corner_forces[static_cast<size_t>(u.node.value-1)][d]/1000., 1e-12);
    const std::array<double, 6> expected{-24., 30., -48., -86., 34., 82.};
    const auto spc_total = resultant(model, result.spc_forces);
    const auto mpc_total = resultant(model, result.mpc_forces);
    for (size_t d = 0; d < 6; ++d) {
        EXPECT_NEAR(spc_total[d], expected[d], 1e-9);
        EXPECT_NEAR(mpc_total[d], 0.0, 1e-9);
    }
    expect_grid_balance(result);
}

TEST(ConstraintForces, OffsetConm2GravityMatchesEquivalentForceAndMoment) {
    const Model gravity = BdfParser::parse_string(weighted_patch() + R"(
CONM2,30,5,0,1.25,0.2,-0.4,0.7
GRAV,2,0,8.,0.,1.,0.
ENDDATA
)");
    const Model equivalent = BdfParser::parse_string(weighted_patch() + R"(
FORCE,2,5,0,10.,0.,1.,0.
MOMENT,2,5,0,1.,-7.,0.,2.
ENDDATA
)");
    const auto inertial = solve_forces(gravity);
    const auto point = solve_forces(equivalent);
    ASSERT_EQ(inertial.displacements.size(), point.displacements.size());
    for (size_t i = 0; i < inertial.displacements.size(); ++i)
        for (size_t d = 0; d < 6; ++d)
            EXPECT_NEAR(inertial.displacements[i].d[d], point.displacements[i].d[d], 1e-12);
    const auto total = resultant(gravity, inertial.spc_forces);
    // The center of mass is (0.95,0.85,3.7), with force (0,10,0).
    const std::array<double, 6> expected{0., -10., 0., 37., 0., -9.5};
    for (size_t d = 0; d < 6; ++d)
        EXPECT_NEAR(total[d], expected[d], 1e-9);
    for (const auto& force : inertial.spc_forces) {
        const auto* other = force_at(point.spc_forces, force.node.value);
        ASSERT_NE(other, nullptr);
        for (size_t d = 0; d < 6; ++d)
            EXPECT_NEAR(force.f[d], other->f[d], 1e-10);
    }
    expect_grid_balance(inertial);
    expect_grid_balance(point);
}

TEST(ConstraintForces, Rbe2AndExplicitMpcChainTransferForceAndLeverArmMoment) {
    // Translations at the RBE2 master are fixed; its rotations are attached
    // to a rotational spring support. The slave's y load arrives through
    // an explicit MPC. A 10 N force at x=2 produces 20 N*length in torsion.
    const Model model = BdfParser::parse_string(R"(
SOL 101
CEND
SPC=1
MPC=7
LOAD=2
SPCFORCES=ALL
MPCFORCES=ALL
GPFORCE=ALL
BEGIN BULK
GRID,1,,0.,0.,0.
GRID,2,,2.,0.,0.
GRID,3,,2.,0.,0.
GRID,11,,0.,0.,0.
CELAS2,1,1000.,1,4,11,4
CELAS2,2,1000.,1,5,11,5
CELAS2,3,1000.,1,6,11,6
SPC1,1,123,1
SPC1,1,123456,11
RBE2,20,1,123456,2
MPC,7,3,2,1.,2,2,-1.
FORCE,2,3,0,10.,0.,1.,0.
ENDDATA
)");
    const auto result = solve_forces(model);
    const auto* root = force_at(result.spc_forces, 1);
    const auto* spring = force_at(result.spc_forces, 11);
    ASSERT_NE(root, nullptr);
    ASSERT_NE(spring, nullptr);
    EXPECT_NEAR(root->f[1], -10., 1e-12);
    EXPECT_NEAR(spring->f[5], -20., 1e-12);
    const auto total = resultant(model, result.mpc_forces);
    for (double value : total)
        EXPECT_NEAR(value, 0.0, 1e-12);
    for (const auto& u : result.displacements) {
        if (u.node == NodeId{1})
            EXPECT_NEAR(u.d[5], 0.02, 1e-14);
        if (u.node == NodeId{3})
            EXPECT_NEAR(u.d[1], 0.04, 1e-14);
    }
    expect_grid_balance(result);
}

TEST(ConstraintForces, RotatedAndPrescribedSupportsKeepSpcAndMpcForcesSeparate) {
    for (const int cd : {0, 5}) {
        for (const double prescribed : {0., 0.2}) {
            SCOPED_TRACE(::testing::Message() << "CD=" << cd << " displacement=" << prescribed);
            // The spring runs along T2 of both ends, so they lie on that axis and
            // it carries no moment.
            const double spring_end_x = cd == 0 ? 0. : 1./std::sqrt(2.);
            const double spring_end_y = cd == 0 ? -1. : -1./std::sqrt(2.);
            const Model model = BdfParser::parse_string(std::format(R"(
SOL 101
CEND
SPC=1
LOAD=2
SPCFORCES=ALL
MPCFORCES=ALL
GPFORCE=ALL
BEGIN BULK
CORD2R,5,0,0.,0.,0.,0.,0.,1.,1.,1.,0.
GRID,1,,0.,0.,0.,{}
GRID,2,,0.,0.,0.
GRID,3,,{:.17g},{:.17g},0.,{}
CELAS2,1,100.,1,2,3,2
SPC,1,1,1,{}
SPC1,1,123456,3
RBE2,20,1,12,2
FORCE,2,2,0,1.,10.,-5.,0.
ENDDATA
)", cd, spring_end_x, spring_end_y, cd, prescribed));
            const auto result = solve_forces(model);
            const auto* support = force_at(result.spc_forces, 1);
            ASSERT_NE(support, nullptr);
            // The spring acts along T2 of grid 1's frame, so the support reacts
            // the load's part along T1.
            EXPECT_NEAR(support->f[0], cd == 0 ? -10. : -5./std::sqrt(2.), 1e-11);
            EXPECT_EQ(support->f[1], 0.);
            const auto* mpc = force_at(result.mpc_forces, 1);
            ASSERT_NE(mpc, nullptr);
            EXPECT_NEAR(mpc->f[0], cd == 0 ? 10. : 5./std::sqrt(2.), 1e-11);
            EXPECT_NEAR(mpc->f[1], cd == 0 ? -5. : -15./std::sqrt(2.), 1e-11);
            for (const auto& u : result.displacements) {
                if (u.node != NodeId{1})
                    continue;
                EXPECT_NEAR(u.d[0], prescribed, 1e-13);
                EXPECT_NEAR(u.d[1], cd == 0 ? -0.05 : -0.15/std::sqrt(2.), 1e-13);
            }
            const auto total = resultant(model, result.spc_forces);
            const std::array<double, 6> expected{-10., 5., 0., 0., 0., 0.};
            for (size_t d = 0; d < 6; ++d)
                EXPECT_NEAR(total[d], expected[d], 1e-11);
            expect_grid_balance(result);
        }
    }
}

TEST(ConstraintForces, FullyPrescribedRotatedSupportTransmitsRigidMotionAndReactions) {
    // Six springs resist an offset RBE2 slave. A seventh spring connects a
    // loaded free grid to that slave, exercising the affine load correction.
    for (const int cd : {0, 5}) {
        for (int motion = 0; motion < 4; ++motion) {
            SCOPED_TRACE(::testing::Message() << "CD=" << cd << " motion=" << motion);
            const Vec3 translation = (motion & 1) ? Vec3{.2, -.1, .3} : Vec3{};
            const Vec3 rotation = (motion & 2) ? Vec3{.01, -.02, .03} : Vec3{};
            const Model model = BdfParser::parse_string(std::format(R"(
SOL 101
CEND
SPC=1
LOAD=2
SPCFORCES=ALL
MPCFORCES=ALL
GPFORCE=ALL
BEGIN BULK
CORD2R,5,0,0.,0.,0.,1.,2.,3.,2.,-1.,0.
GRID,1,,0.,0.,0.,{}
GRID,2,,2.,-1.,3.
GRID,3,,2.,-1.,3.
GRID,4,,2.,-1.,3.
CELAS2,1,100.,2,1,3,1
CELAS2,2,200.,2,2,3,2
CELAS2,3,300.,2,3,3,3
CELAS2,4,400.,2,4,3,4
CELAS2,5,500.,2,5,3,5
CELAS2,6,600.,2,6,3,6
CELAS2,7,100.,4,1,2,1
SPC,1,1,1,{}
SPC,1,1,2,{}
SPC,1,1,3,{}
SPC,1,1,4,{}
SPC,1,1,5,{}
SPC,1,1,6,{}
SPC1,1,123456,3
RBE2,20,1,123456,2
FORCE,2,4,0,1.,1.,0.,0.
ENDDATA
)", cd, translation.x, translation.y, translation.z,
                rotation.x, rotation.y, rotation.z));
            const auto result = solve_forces(model);
            const Mat3 axes = cd == 0 ? Mat3::identity() :
                rotation_matrix(model.coord_systems.at(CoordId{cd}), Vec3{});
            const Vec3 basic_translation = apply_rotation(axes, translation);
            const Vec3 basic_rotation = apply_rotation(axes, rotation);
            const Vec3 arm{2., -1., 3.};
            const Vec3 slave_translation = basic_translation + basic_rotation.cross(arm);
            const std::array<double, 6> prescribed{translation.x, translation.y,
                translation.z, rotation.x, rotation.y, rotation.z};
            const std::array<double, 6> slave{slave_translation.x, slave_translation.y,
                slave_translation.z, basic_rotation.x, basic_rotation.y, basic_rotation.z};
            for (const auto& u : result.displacements) {
                if (u.node == NodeId{1} || u.node == NodeId{2}) {
                    const auto& expected = u.node == NodeId{1} ? prescribed : slave;
                    for (size_t d = 0; d < 6; ++d)
                        EXPECT_NEAR(u.d[d], expected[d], 1e-12);
                } else if (u.node == NodeId{4}) {
                    EXPECT_NEAR(u.d[0], slave[0] + .01, 1e-12);
                }
            }
            const Vec3 slave_force{100.*slave[0]-1., 200.*slave[1], 300.*slave[2]};
            const Vec3 slave_moment{400.*slave[3], 500.*slave[4], 600.*slave[5]};
            const Vec3 root_moment = slave_moment + arm.cross(slave_force);
            const std::array<double, 6> root_basic{slave_force.x, slave_force.y,
                slave_force.z, root_moment.x, root_moment.y, root_moment.z};
            const auto* root = force_at(result.spc_forces, 1);
            const auto* ground = force_at(result.spc_forces, 3);
            ASSERT_NE(root, nullptr);
            ASSERT_NE(ground, nullptr);
            for (int d = 0; d < 6; ++d) {
                double expected_root = 0.;
                for (int j = 0; j < 3; ++j)
                    expected_root += axes(j, d%3) * root_basic[3*(d/3)+j];
                EXPECT_NEAR(root->f[d], expected_root, 1e-10);
                EXPECT_NEAR(ground->f[d], -100.*(d+1)*slave[d], 1e-10);
            }
            const auto spc_total = resultant(model, result.spc_forces);
            const auto mpc_total = resultant(model, result.mpc_forces);
            const std::array<double, 6> expected_total{-1., 0., 0., 0., -3., -1.};
            for (size_t d = 0; d < 6; ++d) {
                EXPECT_NEAR(spc_total[d], expected_total[d], 1e-10);
                EXPECT_NEAR(mpc_total[d], 0., 1e-10);
            }
            expect_grid_balance(result, 1e-10);
        }
    }
}

TEST(ConstraintForces, FailedRecoveryOmitsForcesButKeepsTheDisplacementSolution) {
    // Two MPCs tie GRIDs 2 and 3 to the fixed GRID 1. Their finite 1e308
    // loads sum to an overflowing MPC force at GRID 1, so constraint forces
    // cannot be recovered. The free spring at GRID 4 is unaffected and its
    // solution, u = F/k, must still be reported.
    const Model model = BdfParser::parse_string(R"(
SOL 101
CEND
SPC=1
MPC=3
LOAD=2
DISPLACEMENT=ALL
SPCFORCES=ALL
MPCFORCES=ALL
GPFORCE=ALL
BEGIN BULK
GRID,1,,0.,0.,0.
GRID,2,,1.,0.,0.
GRID,3,,0.,1.,0.
GRID,4,,0.,0.,1.
CELAS2,1,1000.,4,1
SPC1,1,123456,1
SPC1,1,23456,2,3,4
MPC,3,2,1,1.,1,1,-1.
MPC,3,3,1,1.,1,1,-1.
FORCE,2,2,0,1.e308,1.,0.,0.
FORCE,2,3,0,1.e308,1.,0.,0.
FORCE,2,4,0,10.,1.,0.,0.
ENDDATA
)");
    LinearStaticSolver solver(std::make_unique<LdltSolverBackend>());
    const SolverResults results = solver.solve(model);
    const auto& result = results.subcases.at(0);
    EXPECT_NE(result.constraint_force_error.find("force accumulation overflow at GRID 1"),
              std::string::npos)
        << result.constraint_force_error;
    EXPECT_TRUE(result.spc_forces.empty());
    EXPECT_TRUE(result.mpc_forces.empty());
    EXPECT_TRUE(result.gp_forces.empty());
    bool found = false;
    for (const auto& u : result.displacements) {
        if (u.node == NodeId{4}) {
            found = true;
            EXPECT_NEAR(u.d[0], 0.01, 1e-15);
        }
        if (u.node == NodeId{2} || u.node == NodeId{3})
            EXPECT_EQ(u.d[0], 0.);
    }
    EXPECT_TRUE(found);

    // The F06 states why the requested force tables are absent.
    std::ostringstream f06;
    F06Writer::write(results, model, f06);
    const std::string text = f06.str();
    EXPECT_NE(text.find("WARNING: SPC, MPC and grid point forces are not output"),
              std::string::npos);
    EXPECT_NE(text.find(result.constraint_force_error), std::string::npos);
    EXPECT_EQ(text.find("S P C   F O R C E S"), std::string::npos);
    EXPECT_NE(text.find("D I S P L A C E M E N T S"), std::string::npos);
}
