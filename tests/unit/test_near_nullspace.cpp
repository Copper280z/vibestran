// tests/unit/test_near_nullspace.cpp
// Rigid-body near-nullspace of the reduced system and the multigrid built on it.
//
// These prove what the preconditioner relies on: the assembled operator of each
// element and constraint really annihilates the modes, aggregates of whole grid
// points hold them exactly, and carrying them takes the growth of the iterations
// with refinement out of structural models without changing the solution.

#include <gtest/gtest.h>
#include "amg_factor.hpp"
#include "structural_system.hpp"
#include "core/exceptions.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <map>
#include <numbers>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace vibestran;
using vibestran::test::assemble;
using vibestran::test::hex_bar;
using vibestran::test::kSteel;
using vibestran::test::mixed_assembly;
using vibestran::test::shell_sheet;
using vibestran::test::RecordingBackend;

namespace {

// y = K x for lower-triangular or full storage.
std::vector<double> product(const SparseMatrixBuilder::CsrData& k, const std::vector<double>& x) {
    std::vector<double> y(std::size_t(k.n), 0.0);
    const bool lower = k.stores_lower_triangle_only();
    for (int i = 0; i < k.n; ++i)
        for (int p = k.row_ptr[std::size_t(i)]; p < k.row_ptr[std::size_t(i) + 1]; ++p) {
            const int c = k.col_ind[std::size_t(p)];
            const double v = k.values[std::size_t(p)];
            y[std::size_t(i)] += v * x[std::size_t(c)];
            if (lower && c != i) y[std::size_t(c)] += v * x[std::size_t(i)];
        }
    return y;
}

// An unconnected clamped grid satisfies the check that a model has constraints
// and leaves K free-free, so K * mode must vanish on every row.
const std::string kFreeFree = "GRID,9999,,50.,50.,50.\nSPC1,1,123456,9999\nFORCE,2,1,0,1.,1.,0.,0.\n";

void expect_annihilated(const std::string& bulk) {
    const auto r = assemble(kSteel + bulk + kFreeFree, false);
    ASSERT_TRUE(r.modes);
    ASSERT_EQ(r.modes->rows(), r.k.n);
    ASSERT_EQ(r.modes->cols, 6);
    const double k_max = test::largest_magnitude(r.k.values);
    for (int c = 0; c < 6; ++c) {
        std::vector<double> mode(std::size_t(r.k.n));
        double mode_max = 0;
        for (int i = 0; i < r.k.n; ++i) {
            mode[std::size_t(i)] = r.modes->modes[std::size_t(i) * 6 + std::size_t(c)];
            mode_max = std::max(mode_max, std::abs(mode[std::size_t(i)]));
        }
        ASSERT_GT(mode_max, 0.5) << "mode " << c;
        const auto residual = product(r.k, mode);
        for (int i = 0; i < r.k.n; ++i)
            EXPECT_LE(std::abs(residual[std::size_t(i)]), 1e-9 * k_max * mode_max)
                << "mode " << c << " equation " << i;
    }
}

} // namespace

TEST(RigidBodyModes, Hexahedra) {
    expect_annihilated(
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.1,0.\nGRID,3,,1.2,1.,0.\nGRID,4,,0.,1.1,0.2\n"
        "GRID,5,,0.1,0.,1.\nGRID,6,,1.,0.,1.1\nGRID,7,,1.,1.3,1.\nGRID,8,,0.,1.,1.2\n"
        "CHEXA,1,1,1,2,3,4,5,6,+\n+,7,8\n");
}

TEST(RigidBodyModes, QuadraticTetrahedron) {
    expect_annihilated(
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,0.,1.,0.\nGRID,4,,0.,0.,1.\n"
        "GRID,5,,.5,0.,0.\nGRID,6,,.5,.5,0.\nGRID,7,,0.,.5,0.\nGRID,8,,0.,0.,.5\nGRID,9,,.5,0.,.5\nGRID,10,,0.,.5,.5\n"
        "CTETRA,1,1,1,2,3,4,5,6,+\n+,7,8,9,10\n");
}

TEST(RigidBodyModes, CurvedAndWarpedShells) {
    // Grids on a cylinder, the middle quadrilateral warped out of plane, and a
    // triangle sharing an edge: drilling stiffness must follow the rotation.
    expect_annihilated(
        "GRID,1,,1.,0.,0.\nGRID,2,,.92,.38,0.\nGRID,3,,.71,.71,0.\n"
        "GRID,4,,1.,0.,1.\nGRID,5,,.92,.38,1.05\nGRID,6,,.71,.71,1.\nGRID,7,,.9,.4,1.6\n"
        "CQUAD4,1,2,1,2,5,4\nCQUAD4,2,2,2,3,6,5\nCTRIA3,3,2,4,5,7\n");
}

TEST(RigidBodyModes, OffsetBarsOnShells) {
    expect_annihilated(
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nGRID,4,,0.,1.,0.\n"
        "CQUAD4,1,2,1,2,3,4\n"
        "CBAR,2,3,1,2,0.,0.,1.,,+\n+,,,0.,0.,-.02,0.,0.,-.02\n");
}

TEST(RigidBodyModes, RigidElementsMixThreeAndSixDofGrids) {
    // A hexahedron whose base is an RBE2 spider to a shell grid and whose top
    // carries an RBE3: eliminated DOFs must leave the modes consistent, with
    // grids of 3 and 6 independent equations in one system.
    const auto spider =
        "GRID,1,,0.,0.,0.\nGRID,2,,2.,0.,0.\nGRID,3,,2.,2.,0.\nGRID,4,,0.,2.,0.\nCQUAD4,1,2,1,2,3,4\n"
        "GRID,11,,.5,.5,.2\nGRID,12,,1.5,.5,.2\nGRID,13,,1.5,1.5,.2\nGRID,14,,.5,1.5,.2\n"
        "GRID,15,,.5,.5,1.2\nGRID,16,,1.5,.5,1.2\nGRID,17,,1.5,1.5,1.2\nGRID,18,,.5,1.5,1.2\n"
        "CHEXA,2,1,11,12,13,14,15,16,+\n+,17,18\n"
        "RBE2,100,1,123,11,12,13,14\n"
        "GRID,20,,1.,1.,1.5\nRBE3,101,,20,123456,1.,123,15,16,+\n+,17,18\n";
    expect_annihilated(spider);
    const auto r = assemble(kSteel + std::string(spider) + kFreeFree, false);
    std::map<int, int> per_point;
    for (const int p : r.modes->point) ++per_point[p];
    std::set<int> sizes;
    for (const auto& [p, count] : per_point) sizes.insert(count);
    EXPECT_TRUE(sizes.contains(6)) << "shell grids keep six equations";
    EXPECT_TRUE(sizes.contains(3)) << "solid grids keep three";
}

TEST(RigidBodyModes, ConstrainedEquationsHaveNoRow) {
    // Grid 1 is fully clamped, grid 2 keeps T1 and R3 only: grid numbers skip
    // the clamped grid and each equation carries its own row of the motion.
    const auto r = assemble(kSteel +
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nGRID,4,,0.,1.,0.\nCQUAD4,1,2,1,2,3,4\n"
        "SPC1,1,123456,1\nSPC1,1,2345,2\nFORCE,2,3,0,1.,0.,0.,1.\n", false);
    ASSERT_TRUE(r.modes);
    EXPECT_EQ(r.k.n, 2 + 6 + 6);
    EXPECT_EQ(r.modes->points, 3);
    std::vector<int> counts(3, 0);
    for (const int p : r.modes->point) ++counts[std::size_t(p)];
    EXPECT_EQ(counts, (std::vector<int>{2, 6, 6}));
    // The first point is grid 2 at (1, 0, 0), centroid (.5, .5, 0): T1 of a
    // rotation about z through the centroid is -(y - .5) = .5, and R3 is one.
    EXPECT_DOUBLE_EQ(r.modes->modes[0 * 6 + 0], 1.0);
    EXPECT_DOUBLE_EQ(r.modes->modes[0 * 6 + 5], 0.5);
    EXPECT_DOUBLE_EQ(r.modes->modes[1 * 6 + 5], 1.0);
    EXPECT_DOUBLE_EQ(r.modes->modes[1 * 6 + 0], 0.0);
}

TEST(RigidBodyModes, ScalarPointsHaveZeroRowsAndTheirOwnPointNumber) {
    // Scalar points sit at the origin in storage but must not move the centroid,
    // or the rigid rotations of the grids stop being exact.
    const std::string sheet = shell_sheet(3);
    const std::string scalar =
        "SPOINT,90001,90002\nCELAS4,90001,1.E3,90001,90002\nSPC,1,90002,0\n";
    const auto with = assemble(kSteel + sheet + scalar + kFreeFree, false);
    const auto without = assemble(kSteel + sheet + kFreeFree, false);
    ASSERT_TRUE(with.modes && without.modes);
    ASSERT_EQ(with.k.n, without.k.n + 1);
    ASSERT_EQ(with.modes->points, without.modes->points + 1);
    const auto cols = std::size_t(with.modes->cols);
    // Point numbering is by ascending ID, so the scalar's equation comes last.
    const auto scalar_row = std::size_t(with.k.n - 1);
    EXPECT_EQ(with.modes->point[scalar_row], with.modes->points - 1);
    for (std::size_t c = 0; c < cols; ++c)
        EXPECT_EQ(with.modes->modes[scalar_row * cols + c], 0.0);
    for (std::size_t i = 0; i < std::size_t(without.k.n) * cols; ++i)
        EXPECT_EQ(with.modes->modes[i], without.modes->modes[i]);
}

TEST(RigidBodyAmg, ScalarPointsOnAStructureSolveToTheDirectAnswer) {
    // A sheet with scalar oscillators attached at many grids: the zero mode rows
    // leave scalar equations to the smoother, and the solution must not change.
    std::ostringstream extra;
    for (int i = 1; i <= 20; ++i) {
        extra << "SPOINT," << 90000 + i << "\nCELAS2," << 90000 + i << ",1.E5," << i * 5 << ",3,"
              << 90000 + i << "\nCELAS2," << 91000 + i << ",1.E4," << 90000 + i << "\n";
    }
    extra << "SLOAD,2,90007,50.\n";
    const auto r = assemble(shell_sheet(16) + extra.str());
    ASSERT_TRUE(r.modes);
    const auto solved = detail::solve_hybrid_amg(r.k, r.f, 1e-10, 5000, {}, r.modes);
    ASSERT_LT(solved.iterations, 5000);
    const double scale = test::largest_magnitude(r.u);
    for (std::size_t i = 0; i < r.u.size(); ++i)
        EXPECT_NEAR(solved.solution[i], r.u[i], 1e-6 * scale) << "equation " << i;
}

namespace {

// A 3-D pin-jointed lattice of CRODs: grid IDs from 1, clamped at x = 0, with the
// rotations left to AUTOSPC.
std::string rod_lattice(int n) {
    std::ostringstream out;
    out << kSteel << "PROD,70,1,0.01\n";
    const auto grid = [n](int i, int j, int k) { return 1 + i + (n + 1) * (j + (n + 1) * k); };
    int eid = 1;
    for (int k = 0; k <= n; ++k)
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i <= n; ++i) {
                out << std::format("GRID,{},,{},{},{}\n", grid(i, j, k), double(i), double(j), double(k));
                if (i == 0)
                    out << "SPC1,1,123," << grid(i, j, k) << "\n";
                for (const auto& [a, b, c] : {std::tuple(1, 0, 0), std::tuple(0, 1, 0), std::tuple(0, 0, 1),
                                              std::tuple(1, 1, 0), std::tuple(1, 0, 1), std::tuple(0, 1, 1),
                                              std::tuple(1, 1, 1)})
                    if (i + a <= n && j + b <= n && k + c <= n)
                        out << std::format("CROD,{},70,{},{}\n", eid++, grid(i, j, k), grid(i + a, j + b, k + c));
            }
    out << std::format("FORCE,2,{},0,1.,0.,0.,-1.\n", grid(n, n, n));
    return out.str();
}

} // namespace

TEST(RigidBodyModes, RodLatticeAnnihilatesTheModes) {
    // AUTOSPC removes only rotations here, which rods do not couple to the
    // translations, so the free-free translations still carry every rigid motion.
    const std::string lattice = rod_lattice(3);
    const std::string unclamped = std::regex_replace(lattice, std::regex("SPC1,1,123,[0-9]+\n"), "");
    const auto r = assemble(unclamped + kFreeFree, false);
    ASSERT_TRUE(r.modes);
    const double k_max = test::largest_magnitude(r.k.values);
    for (int c = 0; c < 6; ++c) {
        std::vector<double> mode(std::size_t(r.k.n));
        for (std::size_t i = 0; i < mode.size(); ++i)
            mode[i] = r.modes->modes[i * 6 + std::size_t(c)];
        const auto y = product(r.k, mode);
        EXPECT_LT(test::largest_magnitude(y), 1e-9 * k_max) << "mode " << c;
    }
}

TEST(RigidBodyAmg, RodLatticeSolvesToTheDirectAnswer) {
    const auto r = assemble(rod_lattice(10));
    ASSERT_TRUE(r.modes);
    const auto solved = detail::solve_hybrid_amg(r.k, r.f, 1e-10, 5000, {}, r.modes);
    ASSERT_LT(solved.iterations, 5000);
    const double scale = test::largest_magnitude(r.u);
    for (std::size_t i = 0; i < r.u.size(); ++i)
        EXPECT_NEAR(solved.solution[i], r.u[i], 1e-6 * scale) << "equation " << i;
}

TEST(AmgModes, ChoicePrecedence) {
    const std::string bulk = kSteel +
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nGRID,4,,0.,1.,0.\nCQUAD4,1,2,1,2,3,4\n"
        "SPC1,1,123456,1,2\nFORCE,2,3,0,1.,0.,0.,1.\n";
    EXPECT_TRUE(assemble(bulk, true, std::nullopt).modes) << "structures have modes by default";
    EXPECT_EQ(assemble(bulk, true, std::nullopt).modes->cols, 8) << "with the drilling columns";
    EXPECT_FALSE(assemble(bulk, true, std::nullopt, "PARAM,AMGMODES,CONSTANT\n").modes);
    EXPECT_TRUE(assemble(bulk, true, std::nullopt, "PARAM,AMGMODES,RIGID\n").modes);
    EXPECT_TRUE(assemble(bulk, true, AmgModes::Rigid, "PARAM,AMGMODES,CONSTANT\n").modes)
        << "the command line overrides the deck";
    EXPECT_FALSE(assemble(bulk, true, AmgModes::Constant, "PARAM,AMGMODES,RIGID\n").modes)
        << "in both directions";
    EXPECT_THROW((void)assemble(bulk, true, std::nullopt, "PARAM,AMGMODES,BOGUS\n"), SolverError);
    EXPECT_EQ(parse_amg_modes("rigid"), AmgModes::Rigid);
    EXPECT_EQ(parse_amg_modes("Constant"), AmgModes::Constant);
    EXPECT_EQ(parse_amg_modes("drilling"), AmgModes::Drilling);
    EXPECT_EQ(assemble(bulk, true, std::nullopt, "PARAM,AMGMODES,DRILLING\n").modes->cols, 8);
    EXPECT_EQ(assemble(bulk, true, AmgModes::Drilling, "PARAM,AMGMODES,RIGID\n").modes->cols, 8)
        << "the command line overrides the deck";
    EXPECT_FALSE(parse_amg_modes("block"));
}

namespace {

detail::AmgPcgResult host_solve(const RecordingBackend::Record& r, bool modes) {
    return detail::solve_hybrid_amg(r.k, r.f, 1e-9, 5000, {}, modes ? r.modes : nullptr);
}

} // namespace

TEST(ModeAggregation, WholeGridPointsHoldEveryModeExactly) {
    const auto r = assemble(mixed_assembly(8));
    const detail::ScaledAmgMatrix scaled(r.k);
    detail::ModeAggregation aggregation(*r.modes, scaled.scale);
    const auto groups = aggregation.aggregates(*scaled.matrix);
    ASSERT_GT(groups.count, 1u);
    const auto n = std::size_t(r.k.n);

    std::map<int, std::ptrdiff_t> of_point;
    std::vector<std::size_t> size(groups.count, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto [it, fresh] = of_point.emplace(r.modes->point[i], groups.id[i]);
        EXPECT_EQ(it->second, groups.id[i]) << "a grid point split at equation " << i;
        if (groups.id[i] >= 0) ++size[std::size_t(groups.id[i])];
    }
    for (std::size_t g = 0; g < size.size(); ++g) EXPECT_GE(size[g], 6u) << "aggregate " << g;

    // P_tent has orthonormal columns, and each mode restricted to the
    // aggregated equations is in their span: P P^T B = B.
    const auto p = aggregation.tentative(groups);
    std::vector<double> coarse(p->ncols);
    for (int c = 0; c < 6; ++c) {
        std::vector<double> mode(n);
        double top = 0;
        for (std::size_t i = 0; i < n; ++i) {
            mode[i] = r.modes->modes[i * 6 + std::size_t(c)] / scaled.scale[i];
            if (groups.id[i] >= 0) top = std::max(top, std::abs(mode[i]));
        }
        std::ranges::fill(coarse, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (auto j = p->ptr[i]; j < p->ptr[i + 1]; ++j) coarse[std::size_t(p->col[j])] += p->val[j] * mode[i];
        for (std::size_t i = 0; i < n; ++i) {
            if (groups.id[i] < 0) continue;
            double back = 0;
            for (auto j = p->ptr[i]; j < p->ptr[i + 1]; ++j) back += p->val[j] * coarse[std::size_t(p->col[j])];
            EXPECT_NEAR(back, mode[i], 1e-11 * top) << "mode " << c << " equation " << i;
        }
    }
}

// A V-cycle is positive definite, as conjugate gradients needs, only if its
// smoother converges: the largest eigenvalue of M A stays below 2 on every level.
TEST(AmgSmoother, ConvergesOnEveryLevelOfAStructureWhereSpai0DoesNot) {
    const auto r = assemble(hex_bar(4));
    const detail::ScaledAmgMatrix scaled(r.k);
    const auto largest = [](const detail::AmgHostCsr& a, const std::vector<double>& m) {
        Eigen::MatrixXd scaled_a = Eigen::MatrixXd::Zero(Eigen::Index(a.nrows), Eigen::Index(a.nrows));
        for (std::size_t i = 0; i < a.nrows; ++i)
            for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j)
                scaled_a(Eigen::Index(i), a.col[j]) = std::sqrt(m[i]) * a.val[j] * std::sqrt(m[std::size_t(a.col[j])]);
        return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(scaled_a, Eigen::EigenvaluesOnly).eigenvalues().maxCoeff();
    };
    const detail::AmgHierarchy system(scaled.matrix, r.modes.get(), scaled.scale);
    ASSERT_GE(system.levels.size(), 2u);
    for (std::size_t l = 0; l + 1 < system.levels.size(); ++l) {
        const double top = largest(*system.levels[l].A, system.levels[l].smoother);
        EXPECT_LT(top, 1.9) << "level " << l;
        // Not so timid that it stops smoothing: the weights use the spectrum.
        EXPECT_GT(top, 1.0) << "level " << l;
    }
    EXPECT_GT(largest(*scaled.matrix, detail::spai0_weights(*scaled.matrix)), 2.0)
        << "SPAI-0 no longer diverges on elasticity: the reason for the bound has gone";
    const detail::AmgHierarchy scalar(scaled.matrix);
    for (std::size_t l = 0; l + 1 < scalar.levels.size(); ++l)
        EXPECT_LE(largest(*scalar.levels[l].A, scalar.levels[l].smoother), detail::kAmgSmootherBound) << "level " << l;
}

// The estimate of rho(G^-1 A) only tunes the weight. This SPD matrix has a true
// rho of 1, but the power iterations start nearly orthogonal to its top
// eigenvector and report about 0.5, which unbounded would give rho(M A) = 2.67.
TEST(AmgSmoother, AnUnderestimatedSpectrumCannotBreakTheBound) {
    const auto a = std::make_shared<detail::AmgHostCsr>();
    a->set_size(2, 2, false);
    a->set_nonzeros(4);
    a->ptr[0] = 0, a->ptr[1] = 2, a->ptr[2] = 4;
    a->col[0] = 0, a->col[1] = 1, a->col[2] = 0, a->col[3] = 1;
    a->val[0] = 1.8905764746872613, a->val[1] = a->val[2] = 0.34549150281252616, a->val[3] = 0.6545084971874738;
    const auto m = detail::l1_jacobi_weights(*a);
    Eigen::Matrix2d dense = Eigen::Matrix2d::Zero();
    for (int i = 0; i < 2; ++i)
        for (int j = a->ptr[i]; j < a->ptr[i + 1]; ++j)
            dense(i, a->col[j]) = std::sqrt(m[std::size_t(i)]) * a->val[j] * std::sqrt(m[std::size_t(a->col[j])]);
    const double top = Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(dense).eigenvalues().maxCoeff();
    EXPECT_LE(top, detail::kAmgSmootherBound * (1 + 1e-14));
}

TEST(RigidBodyAmg, DenseRowsDoNotWeakenOtherRows) {
    // One equation coupled to all others takes the global spectral radius with
    // it; only its own weight may suffer.
    constexpr int n = 2000;
    SparseMatrixBuilder builder(n);
    for (int i = 0; i < n; ++i) builder.add(i, i, 1.0);
    for (int i = 1; i < n - 1; ++i) builder.add(i + 1, i, -0.25);
    for (int i = 1; i < n; ++i) builder.add(i, 0, 0.02);
    const auto k = builder.build_csr();
    const detail::ScaledAmgMatrix scaled(k);
    const auto m = detail::l1_jacobi_weights(*scaled.matrix);
    double far = 0;
    for (int i = 100; i < n; ++i) far = std::max(far, m[std::size_t(i)]);
    EXPECT_GT(far, 10 * m[0]);
    EXPECT_GT(far, 0.3);
}

// The purpose of the near-nullspace. The solution must not change, and with the
// modes the iterations must stop growing with refinement; a hierarchy without
// them grows at least at the sqrt(cond) rate.
TEST(RigidBodyAmg, IterationsStopGrowingWithRefinementAndTheSolutionIsUnchanged) {
    struct Class {
        std::string name;
        std::vector<std::string> decks;
        double max_growth;
        int max_iterations;
    };
    const std::vector<Class> classes{
        {"solid bar", {hex_bar(4), hex_bar(8), hex_bar(12)}, 1.3, 60},
        {"thin sheet", {shell_sheet(24), shell_sheet(48), shell_sheet(72)}, 1.3, 400},
        {"mixed assembly", {mixed_assembly(8), mixed_assembly(16), mixed_assembly(24)}, 1.3, 700},
    };
    for (const auto& c : classes) {
        std::vector<int> iterations;
        for (const auto& bulk : c.decks) {
            const auto r = assemble(bulk);
            const auto rigid = host_solve(r, true);
            iterations.push_back(rigid.iterations);
            double top = 0, gap = 0;
            for (std::size_t i = 0; i < r.u.size(); ++i) {
                top = std::max(top, std::abs(r.u[i]));
                gap = std::max(gap, std::abs(rigid.solution[i] - r.u[i]));
            }
            EXPECT_LE(gap, 1e-6 * top) << c.name;
            EXPECT_LE(rigid.iterations, c.max_iterations) << c.name;
        }
        EXPECT_LE(double(iterations.back()), c.max_growth * iterations.front())
            << c.name << ": " << iterations[0] << ", " << iterations[1] << ", " << iterations[2];
        std::cout << std::format("[{}] iterations with modes: {} {} {}\n", c.name, iterations[0], iterations[1], iterations[2]);
    }
}

TEST(RigidBodyAmg, ModesImproveOnConstantsOnASolid) {
    const auto r = assemble(hex_bar(8));
    const auto rigid = host_solve(r, true);
    const auto constant = host_solve(r, false);
    EXPECT_LT(5 * rigid.iterations, constant.iterations);
    double top = 0, gap = 0;
    for (std::size_t i = 0; i < r.u.size(); ++i) {
        top = std::max(top, std::abs(constant.solution[i]));
        gap = std::max(gap, std::abs(rigid.solution[i] - constant.solution[i]));
    }
    EXPECT_LE(gap, 1e-6 * top);
}

TEST(RigidBodyAmg, RefactorKeepsTheStableSmoother) {
    const auto r = assemble(hex_bar(4));
    detail::AmgSystem system(r.k, 1e-9, [](const detail::ScaledAmgMatrix& scaled, const detail::AmgHierarchy& h) {
        return std::make_unique<detail::HostAmgEngine>(scaled, h, detail::AmgSpmvFactory{}, 2000);
    }, r.modes);
    const auto first = system.solve(r.k, r.f);
    auto doubled = r.k;
    std::ranges::transform(doubled.values, doubled.values.begin(), [](double v) { return 2 * v; });
    ASSERT_TRUE(system.refactor(doubled));
    const auto second = system.solve(doubled, r.f);
    // The same operator up to a factor: the same hierarchy, half the solution.
    EXPECT_NEAR(double(second.iterations), double(first.iterations), 2.0);
    for (std::size_t i = 0; i < first.solution.size(); i += 97)
        EXPECT_NEAR(second.solution[i], first.solution[i] / 2, 1e-6 * std::abs(first.solution[i]) + 1e-15);
}

namespace {

// A flat m x m CQUAD4 sheet, 1 by 1, without constraints or loads. reversed(i, j)
// lists an element's grids the other way round, which flips its normal.
template <class Reversed>
std::string rectangle_sheet(int m, Reversed reversed) {
    std::ostringstream out;
    const auto grid = [m](int i, int j) { return 1 + i + (m + 1) * j; };
    for (int j = 0; j <= m; ++j)
        for (int i = 0; i <= m; ++i)
            out << std::format("GRID,{},,{},{},0.\n", grid(i, j), double(i) / m, double(j) / m);
    int eid = 1;
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i)
            out << (reversed(i, j)
                ? std::format("CQUAD4,{},2,{},{},{},{}\n", eid++, grid(i, j), grid(i, j + 1), grid(i + 1, j + 1), grid(i + 1, j))
                : std::format("CQUAD4,{},2,{},{},{},{}\n", eid++, grid(i, j), grid(i + 1, j), grid(i + 1, j + 1), grid(i, j + 1)));
    return out.str();
}

std::string rectangle_sheet(int m) {
    return rectangle_sheet(m, [](int, int) { return false; });
}

std::vector<double> column(const NearNullspace& modes, int c) {
    std::vector<double> out(std::size_t(modes.rows()));
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = modes.modes[i * std::size_t(modes.cols) + std::size_t(c)];
    return out;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

RecordingBackend::Record drilling_system(const std::string& bulk) {
    return assemble(kSteel + bulk + kFreeFree, false, AmgModes::Drilling);
}

// P_tent has orthonormal columns and P P^T B = B on every aggregated equation.
void expect_holds_every_column(const NearNullspace& modes, const detail::ScaledAmgMatrix& scaled,
                               const detail::ModeAggregation::Aggregates& groups, const detail::AmgHostCsr& p) {
    const auto n = std::size_t(modes.rows());
    std::vector<double> coarse(p.ncols);
    for (int c = 0; c < modes.cols; ++c) {
        std::vector<double> mode = column(modes, c);
        double top = 0;
        for (std::size_t i = 0; i < n; ++i) {
            mode[i] /= scaled.scale[i];
            if (groups.id[i] >= 0) top = std::max(top, std::abs(mode[i]));
        }
        std::ranges::fill(coarse, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (auto j = p.ptr[i]; j < p.ptr[i + 1]; ++j) coarse[std::size_t(p.col[j])] += p.val[j] * mode[i];
        for (std::size_t i = 0; i < n; ++i) {
            if (groups.id[i] < 0) continue;
            double back = 0;
            for (auto j = p.ptr[i]; j < p.ptr[i + 1]; ++j) back += p.val[j] * coarse[std::size_t(p.col[j])];
            ASSERT_NEAR(back, mode[i], 1e-9 * top) << "mode " << c << " equation " << i;
        }
    }
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(Eigen::Index(n), Eigen::Index(p.ncols));
    for (std::size_t i = 0; i < n; ++i)
        for (auto j = p.ptr[i]; j < p.ptr[i + 1]; ++j) dense(Eigen::Index(i), p.col[j]) = p.val[j];
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(Eigen::Index(p.ncols), Eigen::Index(p.ncols));
    EXPECT_LT((dense.transpose() * dense - identity).cwiseAbs().maxCoeff(), 1e-10);
}

} // namespace

// Why these columns exist: per unit of rotation at the grids, both patterns take
// a tiny fraction of the energy of a rotation about the normal that varies
// smoothly across the sheet, and the rigid-body modes cannot represent them.
TEST(DrillingModes, BothPatternsAreSoftOnARectangleAndNotRigidMotions) {
    const auto r = drilling_system(rectangle_sheet(12));
    ASSERT_EQ(r.modes->cols, 8);
    ASSERT_EQ(r.modes->optional_cols, 2);
    const auto energy = [&](const std::vector<double>& x) { return dot(x, product(r.k, x)) / dot(x, x); };
    const auto uniform = column(*r.modes, 6), alternating = column(*r.modes, 7);

    // The smooth comparison: the same rows with a sine across the sheet.
    std::vector<double> smooth(uniform.size(), 0.0);
    int rotations = 0;
    for (std::size_t i = 0; i < smooth.size(); ++i)
        if (uniform[i] != 0) smooth[i] = std::sin(0.5 * std::numbers::pi * double(rotations++) / 13.0);
    ASSERT_EQ(rotations, 13 * 13);
    const double stiff = energy(smooth);
    EXPECT_LT(energy(uniform), 1e-3 * stiff);
    EXPECT_LT(energy(alternating), 1e-3 * stiff);

    // Least-squares residual against the rigid-body modes.
    const auto rows = Eigen::Index(uniform.size());
    Eigen::MatrixXd rigid(rows, 6);
    for (Eigen::Index i = 0; i < rows; ++i)
        for (Eigen::Index c = 0; c < 6; ++c) rigid(i, c) = r.modes->modes[std::size_t(i) * 8 + std::size_t(c)];
    const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(rigid);
    for (const auto* mode : {&uniform, &alternating}) {
        const Eigen::Map<const Eigen::VectorXd> b(mode->data(), rows);
        EXPECT_GT((b - rigid * qr.solve(b)).norm(), 0.3 * b.norm());
    }
    // The rigid columns themselves are the ones of rigid_body_modes().
    const auto plain = assemble(kSteel + rectangle_sheet(12) + kFreeFree, false, AmgModes::Rigid);
    for (std::size_t i = 0; i < std::size_t(r.k.n); ++i)
        for (std::size_t c = 0; c < 6; ++c)
            EXPECT_EQ(r.modes->modes[i * 8 + c], plain.modes->modes[i * 6 + c]);
}

TEST(DrillingModes, ColoursAlternateAcrossEveryElementEdge) {
    constexpr int m = 7;
    const auto r = drilling_system(rectangle_sheet(m));
    const auto uniform = column(*r.modes, 6), alternating = column(*r.modes, 7);
    // Only the rotation about z is present on a flat sheet, one equation per grid,
    // with the grids in ID order.
    std::vector<double> sign;
    for (std::size_t i = 0; i < uniform.size(); ++i)
        if (uniform[i] != 0) {
            EXPECT_NEAR(std::abs(uniform[i]), 1.0, 1e-14);
            sign.push_back(alternating[i] / uniform[i]);
        }
    ASSERT_EQ(sign.size(), std::size_t((m + 1) * (m + 1)));
    for (int j = 0; j <= m; ++j)
        for (int i = 0; i <= m; ++i) {
            const auto here = std::size_t(i + (m + 1) * j);
            if (i < m) EXPECT_EQ(sign[here + 1], -sign[here]);
            if (j < m) EXPECT_EQ(sign[here + std::size_t(m + 1)], -sign[here]);
        }
}

TEST(DrillingModes, ElementOrientationDoesNotMatter) {
    // Real meshes mix orientations. A checkerboard of reversed elements must give
    // one uniform rotation, not a pattern that follows each grid's first element.
    const auto consistent = drilling_system(rectangle_sheet(8));
    const auto mixed = drilling_system(rectangle_sheet(8, [](int i, int j) { return (i * 3 + j * 5) % 7 < 3; }));
    ASSERT_EQ(consistent.k.n, mixed.k.n);
    for (const int c : {6, 7}) {
        const auto a = column(*consistent.modes, c), b = column(*mixed.modes, c);
        const double sign = dot(a, b) > 0 ? 1 : -1;
        for (std::size_t i = 0; i < a.size(); ++i)
            EXPECT_NEAR(a[i], sign * b[i], 1e-12) << "column " << c << " equation " << i;
    }
}

TEST(DrillingModes, TheAxisFollowsACurvedShell) {
    // A quarter cylinder about z, with rows at theta = 0, 11.25, .. 90 degrees.
    std::ostringstream deck;
    constexpr int around = 8, along = 4;
    const auto grid = [](int i, int j) { return 1 + i + (around + 1) * j; };
    std::vector<double> theta;
    for (int j = 0; j <= along; ++j)
        for (int i = 0; i <= around; ++i) {
            theta.push_back(0.5 * std::numbers::pi * i / around);
            deck << std::format("GRID,{},,{:.12f},{:.12f},{}\n", grid(i, j), std::cos(theta.back()), std::sin(theta.back()), 0.2 * j);
        }
    int eid = 1;
    for (int j = 0; j < along; ++j)
        for (int i = 0; i < around; ++i)
            deck << std::format("CQUAD4,{},2,{},{},{},{}\n", eid++, grid(i, j), grid(i + 1, j), grid(i + 1, j + 1), grid(i, j + 1));
    const auto r = drilling_system(deck.str());
    const auto uniform = column(*r.modes, 6);
    ASSERT_EQ(uniform.size(), theta.size() * 6);
    double agreement = 0;
    for (std::size_t g = 0; g < theta.size(); ++g) {
        const Eigen::Vector3d axis(uniform[g * 6 + 3], uniform[g * 6 + 4], uniform[g * 6 + 5]);
        EXPECT_NEAR(axis.norm(), 1.0, 1e-12) << "grid " << g;
        for (int c = 0; c < 3; ++c) EXPECT_EQ(uniform[g * 6 + std::size_t(c)], 0.0);
        // Radial up to sign, with one sign over the whole shell.
        const double radial = axis.dot(Eigen::Vector3d(std::cos(theta[g]), std::sin(theta[g]), 0));
        EXPECT_GT(std::abs(radial), 0.99) << "grid " << g;
        if (g == 0) agreement = radial;
        EXPECT_GT(radial * agreement, 0) << "grid " << g;
    }
}

TEST(DrillingModes, ColumnsAreZeroAtFoldsAndAwayFromQuadrilaterals) {
    // Two sheets meeting at a right angle along x = 1, a triangle beyond the far
    // edge, a bar and a solid: only smooth CQUAD4 grids take part.
    const std::string bulk =
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nGRID,4,,0.,1.,0.\n"
        "GRID,5,,1.,0.,1.\nGRID,6,,1.,1.,1.\n"
        "CQUAD4,1,2,1,2,3,4\nCQUAD4,2,2,2,5,6,3\n"
        "GRID,7,,-1.,0.,0.\nCTRIA3,3,2,1,7,4\n"
        "GRID,8,,0.,-1.,0.\nCBAR,4,3,1,8,0.,0.,1.\n";
    const auto r = drilling_system(bulk);
    ASSERT_EQ(r.modes->cols, 8);
    std::map<int, std::vector<std::size_t>> rows;
    for (std::size_t i = 0; i < r.modes->point.size(); ++i) rows[r.modes->point[i]].push_back(i);
    ASSERT_EQ(rows.size(), 8u) << "grids 1 to 8, in order; the fixture's grid is clamped";
    const auto magnitude = [&](int point) {
        double sum = 0;
        for (const auto i : rows[point])
            for (std::size_t c = 6; c < 8; ++c) sum += std::abs(r.modes->modes[i * 8 + c]);
        return sum;
    };
    for (const int smooth : {0, 3, 4, 5}) EXPECT_GT(magnitude(smooth), 0) << "point " << smooth;
    for (const int fold : {1, 2}) EXPECT_EQ(magnitude(fold), 0) << "point " << fold;
    for (const int other : {6, 7}) EXPECT_EQ(magnitude(other), 0) << "point " << other;
}

TEST(DrillingModes, AGridWithASnormUsesItAsTheAxis) {
    // A flat sheet with one grid's normal tilted by SNORM: that grid's rotation
    // follows the normal the elements use there, and the others stay on z.
    const auto r = drilling_system(rectangle_sheet(2) + "SNORM,5,0,.6,0.,.8\n");
    const auto uniform = column(*r.modes, 6);
    ASSERT_EQ(uniform.size(), 9u * 6u);
    const auto axis = [&](std::size_t grid, std::size_t c) { return uniform[grid * 6 + 3 + c]; };
    const double sign = axis(0, 2) > 0 ? 1 : -1;
    EXPECT_NEAR(sign * axis(4, 0), .6, 1e-12);
    EXPECT_NEAR(sign * axis(4, 2), .8, 1e-12);
    for (const std::size_t grid : {std::size_t(0), std::size_t(3), std::size_t(8)})
        EXPECT_NEAR(sign * axis(grid, 2), 1.0, 1e-12) << "grid " << grid;
}

TEST(DrillingModes, ARingThatCannotBeTwoColouredStillGivesAUniformRotation) {
    // Five elements around a closed ring: the colouring conflicts on one seam, but
    // the normals agree all the way round.
    std::ostringstream deck;
    constexpr int around = 5;
    for (int i = 0; i < around; ++i) {
        const double angle = 2 * std::numbers::pi * i / around;
        deck << std::format("GRID,{},,{:.12f},{:.12f},0.\nGRID,{},,{:.12f},{:.12f},1.\n", i + 1, std::cos(angle), std::sin(angle),
                            i + 11, std::cos(angle), std::sin(angle));
    }
    for (int i = 0; i < around; ++i) {
        const int next = (i + 1) % around;
        deck << std::format("CQUAD4,{},2,{},{},{},{}\n", i + 1, i + 1, next + 1, next + 11, i + 11);
    }
    deck << "PARAM,SNORM,80.\n";
    const auto r = drilling_system(deck.str());
    const auto uniform = column(*r.modes, 6);
    for (int i = 0; i < around; ++i) {
        const double angle = 2 * std::numbers::pi * i / around;
        const Eigen::Vector3d radial(std::cos(angle), std::sin(angle), 0);
        const auto g = std::size_t(i) * 6;
        const Eigen::Vector3d axis(uniform[g + 3], uniform[g + 4], uniform[g + 5]);
        EXPECT_GT(axis.dot(radial) * Eigen::Vector3d(uniform[3], uniform[4], uniform[5]).dot(Eigen::Vector3d::UnitX()), 0.9) << "grid " << i;
    }
}

TEST(DrillingModes, WithoutCQuad4TheyAreTheRigidBodyModes) {
    const std::string triangles =
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nGRID,4,,0.,1.,0.\nCTRIA3,1,2,1,2,3\nCTRIA3,2,2,1,3,4\n";
    for (const std::string& bulk : {triangles, hex_bar(2)}) {
        const auto rigid = assemble(kSteel + bulk + kFreeFree, false, AmgModes::Rigid);
        const auto drilling = assemble(kSteel + bulk + kFreeFree, false, AmgModes::Drilling);
        EXPECT_EQ(drilling.modes->cols, 6);
        EXPECT_EQ(drilling.modes->optional_cols, 0);
        EXPECT_EQ(drilling.modes->modes, rigid.modes->modes);
    }
}

// With no optional columns the prolongation must be AMGCL's. Fast-math may round
// the two call sites differently, so they agree to round-off, not bit for bit.
TEST(ModeAggregation, RigidProlongationIsAmgcls) {
    const auto r = assemble(mixed_assembly(8));
    const detail::ScaledAmgMatrix scaled(r.k);
    detail::ModeAggregation aggregation(*r.modes, scaled.scale);
    const auto groups = aggregation.aggregates(*scaled.matrix);
    const auto ours = aggregation.tentative(groups);

    amgcl::coarsening::nullspace_params nullspace;
    nullspace.cols = r.modes->cols;
    nullspace.B = r.modes->modes;
    for (std::size_t i = 0; i < scaled.scale.size(); ++i)
        for (std::size_t c = 0; c < 6; ++c) nullspace.B[i * 6 + c] /= scaled.scale[i];
    const auto theirs = amgcl::coarsening::tentative_prolongation<detail::AmgHostCsr>(
        scaled.scale.size(), groups.count, groups.id, nullspace, 1);
    ASSERT_EQ(ours->ncols, theirs->ncols);
    ASSERT_EQ(ours->nnz, theirs->nnz);
    EXPECT_TRUE(std::equal(ours->ptr, ours->ptr + ours->nrows + 1, theirs->ptr));
    EXPECT_TRUE(std::equal(ours->col, ours->col + ours->nnz, theirs->col));
    for (std::size_t i = 0; i < ours->nnz; ++i) EXPECT_NEAR(ours->val[i], theirs->val[i], 1e-13) << "entry " << i;
}

TEST(ModeAggregation, SolidAggregatesCarrySixColumnsAndShellAggregatesEight) {
    const auto r = assemble(mixed_assembly(8), true, AmgModes::Drilling);
    ASSERT_EQ(r.modes->cols, 8);
    const detail::ScaledAmgMatrix scaled(r.k);
    detail::ModeAggregation aggregation(*r.modes, scaled.scale);
    const auto groups = aggregation.aggregates(*scaled.matrix);
    const auto p = aggregation.tentative(groups);

    // Sort the aggregates by whether any row carries a drilling column.
    std::vector<char> shell(groups.count, 0);
    std::vector<std::size_t> width(groups.count, 0);
    for (std::size_t i = 0; i < scaled.scale.size(); ++i) {
        if (groups.id[i] < 0) continue;
        const auto g = std::size_t(groups.id[i]);
        width[g] = std::size_t(p->ptr[i + 1] - p->ptr[i]);
        for (std::size_t c = 6; c < 8; ++c)
            if (r.modes->modes[i * 8 + c] != 0) shell[g] = 1;
    }
    int solids = 0, shells = 0;
    for (std::size_t g = 0; g < groups.count; ++g) {
        if (shell[g] != 0) {
            EXPECT_GE(width[g], 7u) << "aggregate " << g;
            ++shells;
        } else {
            EXPECT_EQ(width[g], 6u) << "aggregate " << g;
            ++solids;
        }
    }
    EXPECT_GT(solids, 0);
    EXPECT_GT(shells, 0);
    expect_holds_every_column(*r.modes, scaled, groups, *p);
}

TEST(ModeAggregation, AColumnInTheSpanOfTheOthersIsLeftOutButStaysExact) {
    // With no translation in the plane, a rigid rotation about the normal is the
    // uniform drilling pattern, so no aggregate can use both.
    constexpr int m = 16;
    std::ostringstream spc;
    for (int g = 1; g <= (m + 1) * (m + 1); ++g) spc << "SPC1,1,12," << g << "\n";
    const auto r = assemble(kSteel + rectangle_sheet(m) + spc.str() + "SPC1,1,3,1,2,18\nFORCE,2,289,0,1.,0.,0.,1.\n", true, AmgModes::Drilling);
    const detail::ScaledAmgMatrix scaled(r.k);
    detail::ModeAggregation aggregation(*r.modes, scaled.scale);
    const auto groups = aggregation.aggregates(*scaled.matrix);
    const auto p = aggregation.tentative(groups);
    ASSERT_GT(groups.count, 1u);
    for (std::size_t i = 0; i < scaled.scale.size(); ++i)
        if (groups.id[i] >= 0) EXPECT_LE(p->ptr[i + 1] - p->ptr[i], 7) << "equation " << i;
    expect_holds_every_column(*r.modes, scaled, groups, *p);
}

// The purpose of the columns: an in-plane load on a CQUAD4 sheet, where the
// iterations of the rigid-body hierarchy grow with refinement and these do not.
namespace {

std::vector<double> multiply(const detail::AmgHostCsr& a, const std::vector<double>& x) {
    std::vector<double> y(a.nrows, 0.0);
    for (std::size_t i = 0; i < a.nrows; ++i)
        for (auto j = a.ptr[i]; j < a.ptr[i + 1]; ++j) y[i] += a.val[j] * x[std::size_t(a.col[j])];
    return y;
}

std::vector<double> wave(std::size_t n, double phase) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = std::sin(phase * double(i)) + 0.3 * std::cos(double(i * i % 97));
    return x;
}

// What a device computes for P e and R y, from A, M and T only, against the stored P and R.
void expect_transfers_computed_from_tentative(const detail::AmgLevel& level) {
    ASSERT_TRUE(level.tentative);
    const auto t = level.tentative->csr();
    const auto tt = amgcl::backend::transpose(*t);
    const auto e = wave(t->ncols, 0.37);
    auto te = multiply(*t, e);
    const auto ate = multiply(*level.A, te);
    for (std::size_t i = 0; i < te.size(); ++i) te[i] -= level.smoother[i] * ate[i];
    const auto stored_p = multiply(*level.P, e);
    const auto y = wave(t->nrows, 0.71);
    auto my = y;
    for (std::size_t i = 0; i < my.size(); ++i) my[i] *= level.smoother[i];
    const auto amy = multiply(*level.A, my);
    auto residual = y;
    for (std::size_t i = 0; i < y.size(); ++i) residual[i] -= amy[i];
    const auto computed_r = multiply(*tt, residual);
    const auto stored_r = multiply(*level.R, y);
    const auto top = [](const std::vector<double>& v) { return test::largest_magnitude(v); };
    ASSERT_GT(top(stored_p), 0.0);
    for (std::size_t i = 0; i < te.size(); ++i) ASSERT_NEAR(te[i], stored_p[i], 1e-11 * top(stored_p)) << "P row " << i;
    for (std::size_t i = 0; i < computed_r.size(); ++i) ASSERT_NEAR(computed_r[i], stored_r[i], 1e-11 * top(stored_r)) << "R row " << i;
}

} // namespace

// T in patches is T, on grids with and without their optional columns, and rows
// of equations in no aggregate stay empty.
TEST(ComputedTransfers, PatchesAreTheTentativeProlongation) {
    for (const auto& deck : {mixed_assembly(8), test::in_plane_sheet(8), hex_bar(4)}) {
        const auto r = assemble(deck, true, AmgModes::Drilling);
        const detail::ScaledAmgMatrix scaled(r.k);
        detail::ModeAggregation aggregation(*r.modes, scaled.scale);
        const auto groups = aggregation.aggregates(*scaled.matrix);
        const auto t = aggregation.tentative(groups);
        const auto& patches = *aggregation.patches();
        const auto rebuilt = patches.csr();
        ASSERT_EQ(rebuilt->nrows, t->nrows);
        ASSERT_EQ(rebuilt->ncols, t->ncols);
        ASSERT_EQ(rebuilt->nnz, t->nnz);
        EXPECT_TRUE(std::equal(t->ptr, t->ptr + t->nrows + 1, rebuilt->ptr));
        EXPECT_TRUE(std::equal(t->col, t->col + t->nnz, rebuilt->col));
        EXPECT_TRUE(std::equal(t->val, t->val + t->nnz, rebuilt->val));
        ASSERT_EQ(patches.equation.size(), t->nrows) << "every equation has a row, so T e overwrites all of a work vector";
        auto seen = patches.equation;
        std::ranges::sort(seen);
        for (std::size_t i = 0; i < seen.size(); ++i) ASSERT_EQ(seen[i], int(i));
        ASSERT_EQ(patches.row_patch.size(), patches.equation.size());
        EXPECT_EQ(patches.patches.back()[0], int(patches.equation.size()));
        EXPECT_EQ(patches.patches.back()[1], int(patches.values.size()));
        EXPECT_EQ(patches.patches.back()[2], int(patches.col_patch.size()));
        for (std::size_t i = 0; i < t->nrows; ++i)
            if (groups.id[i] < 0) EXPECT_EQ(t->ptr[i + 1], t->ptr[i]) << i;
        // Each value once for T and T^T, with no column indices, against two CSRs.
        EXPECT_LT(patches.bytes(8), 2 * (t->nnz * 12 + (t->nrows + 1) * 4));
    }
}

// The identity the device relies on, for each kind of hierarchy with modes: its
// finest P and R are the stored ones to rounding, for any input.
TEST(ComputedTransfers, ApplyTheStoredOperators) {
    for (const auto modes : {AmgModes::Rigid, AmgModes::Drilling})
        for (const auto& deck : {mixed_assembly(8), test::in_plane_sheet(16), hex_bar(4)}) {
            const auto r = assemble(deck, true, modes);
            const detail::ScaledAmgMatrix scaled(r.k);
            const detail::AmgHierarchy h(scaled.matrix, r.modes.get(), scaled.scale, 40);
            ASSERT_GE(h.levels.size(), 2U);
            expect_transfers_computed_from_tentative(h.levels.front());
            for (std::size_t l = 1; l < h.levels.size(); ++l) EXPECT_FALSE(h.levels[l].tentative) << "finest level only";
        }
}

// AMGCL's smoothed aggregation filters A in P, so it is not (I - M A) T.
TEST(ComputedTransfers, TheScalarHierarchyKeepsItsStoredTransfers) {
    const auto r = assemble(hex_bar(4));
    const detail::ScaledAmgMatrix scaled(r.k);
    const detail::AmgHierarchy h(scaled.matrix, nullptr, {}, 40);
    ASSERT_GE(h.levels.size(), 2U);
    for (const auto& level : h.levels) EXPECT_FALSE(level.tentative);
}

// Whichever way a plan applies the transfers, a refactor must change them the
// same way: stored P is smoothed again, and the coarse operators follow from it.
TEST(ComputedTransfers, RefactorSmoothsTheFinestTransferAgain) {
    const auto r = assemble(test::in_plane_sheet(16), true, AmgModes::Drilling);
    const detail::ScaledAmgMatrix before(r.k);
    const detail::ScaledAmgMatrix after(test::with_stiff_bands(r.k, r.u).k);
    detail::AmgHierarchy h(before.matrix, r.modes.get(), before.scale, 40);
    const std::vector<double> old_values(h.levels[0].P->val, h.levels[0].P->val + h.levels[0].P->nnz);
    ASSERT_TRUE(h.rebuild(after.matrix));
    expect_transfers_computed_from_tentative(h.levels.front());
    double moved = 0;
    for (std::size_t i = 0; i < old_values.size(); ++i) moved = std::max(moved, std::abs(old_values[i] - h.levels[0].P->val[i]));
    EXPECT_GT(moved, 1e-6) << "this congruence must change P, or the test proves nothing";
    // The coarse operator is formed from the new transfer.
    const auto p = h.levels[0].P;
    const auto galerkin = amgcl::backend::product(*h.levels[0].R, *amgcl::backend::product(*h.levels[0].A, *p));
    const auto& coarse = *h.levels[1].A;
    const auto e = wave(coarse.nrows, 0.13);
    const auto expected = multiply(*galerkin, e);
    const auto actual = multiply(coarse, e);
    for (std::size_t i = 0; i < actual.size(); ++i) ASSERT_NEAR(actual[i], expected[i], 1e-10 * (1 + std::abs(expected[i]))) << i;
}

TEST(DrillingAmg, InPlaneLoadsStopGrowingWithRefinementAndTheSolutionIsUnchanged) {
    std::vector<int> rigid, drilling;
    for (const int m : {24, 48, 96}) {
        const auto plain = assemble(test::in_plane_sheet(m));
        const auto extended = assemble(test::in_plane_sheet(m), true, AmgModes::Drilling);
        const auto with_rigid = detail::solve_hybrid_amg(plain.k, plain.f, 1e-9, 5000, {}, plain.modes);
        const auto with_drilling = detail::solve_hybrid_amg(extended.k, extended.f, 1e-9, 5000, {}, extended.modes);
        rigid.push_back(with_rigid.iterations);
        drilling.push_back(with_drilling.iterations);
        const double top = test::largest_magnitude(extended.u);
        for (std::size_t i = 0; i < extended.u.size(); ++i)
            ASSERT_NEAR(with_drilling.solution[i], extended.u[i], 1e-6 * top) << "m " << m << " equation " << i;
    }
    std::cout << std::format("[in-plane sheet] rigid: {} {} {}; drilling: {} {} {}\n", rigid[0], rigid[1], rigid[2],
                             drilling[0], drilling[1], drilling[2]);
    EXPECT_LE(double(drilling[2]), 1.3 * drilling[0]);
    EXPECT_GE(double(rigid[2]), 1.5 * rigid[0]) << "the rigid hierarchy no longer struggles: the reason for the columns has gone";
    EXPECT_LT(3 * drilling[2], rigid[2]);
}
