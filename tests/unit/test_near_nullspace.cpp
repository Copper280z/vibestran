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
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace vibestran;
using vibestran::test::assemble;
using vibestran::test::hex_bar;
using vibestran::test::kSteel;
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

TEST(AmgModes, ChoicePrecedence) {
    const std::string bulk = kSteel +
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nGRID,4,,0.,1.,0.\nCQUAD4,1,2,1,2,3,4\n"
        "SPC1,1,123456,1,2\nFORCE,2,3,0,1.,0.,0.,1.\n";
    EXPECT_FALSE(assemble(bulk, true, std::nullopt).modes) << "structures default to the scalar hierarchy";
    EXPECT_FALSE(assemble(bulk, true, std::nullopt, "PARAM,AMGMODES,CONSTANT\n").modes);
    EXPECT_TRUE(assemble(bulk, true, std::nullopt, "PARAM,AMGMODES,RIGID\n").modes);
    EXPECT_TRUE(assemble(bulk, true, AmgModes::Rigid, "PARAM,AMGMODES,CONSTANT\n").modes)
        << "the command line overrides the deck";
    EXPECT_FALSE(assemble(bulk, true, AmgModes::Constant).modes);
    EXPECT_THROW((void)assemble(bulk, true, std::nullopt, "PARAM,AMGMODES,BOGUS\n"), SolverError);
    EXPECT_EQ(parse_amg_modes("rigid"), AmgModes::Rigid);
    EXPECT_EQ(parse_amg_modes("Constant"), AmgModes::Constant);
    EXPECT_FALSE(parse_amg_modes("block"));
}

namespace {

// A thin CQUAD4 cantilever sheet bending under an edge load.
std::string shell_sheet(int n) {
    std::ostringstream out;
    out << kSteel;
    const auto node = [&](int i, int j) { return 1 + i + (n + 1) * j; };
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) out << std::format("GRID,{},,{},{},0.\n", node(i, j), double(i) / n, double(j) / n);
    int eid = 1;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            out << std::format("CQUAD4,{},2,{},{},{},{}\n", eid++, node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1));
    for (int j = 0; j <= n; ++j) {
        out << "SPC1,1,123456," << node(0, j) << "\n";
        out << std::format("FORCE,2,{},0,1.,0.,0.,-1.\n", node(n, j));
    }
    return out.str();
}

// CHEXA8 blocks tied to a clamped CQUAD4 plate with RBE2 spiders: mixed grids,
// partial constraints and dense rows from one system.
std::string mixed_assembly(int n) {
    std::ostringstream out;
    out << kSteel;
    const auto plate = [&](int i, int j) { return 1 + i + (n + 1) * j; };
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) out << std::format("GRID,{},,{},{},0.\n", plate(i, j), 2.0 * i / n, 2.0 * j / n);
    int eid = 1;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            out << std::format("CQUAD4,{},2,{},{},{},{}\n", eid++, plate(i, j), plate(i + 1, j), plate(i + 1, j + 1), plate(i, j + 1));
    for (int i = 0; i <= n; ++i) out << "SPC1,1,123456," << plate(i, 0) << "," << plate(i, n) << "\n";
    for (int j = 1; j < n; ++j) out << "SPC1,1,3," << plate(0, j) << "\n";
    const int m = n / 2;
    int first = plate(n, n) + 1;
    int rbe = 1000000;
    for (int b = 0; b < 2; ++b) {
        const auto node = [&](int i, int j, int k) { return first + i + (m + 1) * (j + (m + 1) * k); };
        for (int k = 0; k <= m; ++k)
            for (int j = 0; j <= m; ++j)
                for (int i = 0; i <= m; ++i)
                    out << std::format("GRID,{},,{},{},{}\n", node(i, j, k), .25 + b + .5 * i / m, .75 + .5 * j / m, .1 + .5 * k / m);
        for (int k = 0; k < m; ++k)
            for (int j = 0; j < m; ++j)
                for (int i = 0; i < m; ++i)
                    out << std::format("CHEXA,{},1,{},{},{},{},{},{},+\n+,{},{}\n", eid++, node(i, j, k), node(i + 1, j, k),
                                       node(i + 1, j + 1, k), node(i, j + 1, k), node(i, j, k + 1), node(i + 1, j, k + 1),
                                       node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1));
        const int anchor = plate((1 + 2 * b) * n / 4, n / 2);
        for (int j = 0; j <= m; ++j)
            for (int i = 0; i <= m; ++i)
                out << std::format("RBE2,{},{},123,{}\n", rbe++, anchor, node(i, j, 0));
        out << std::format("FORCE,2,{},0,1000.,0.,0.,-1.\n", node(m / 2, m / 2, m));
        first = node(m, m, m) + 1;
    }
    return out.str();
}

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
TEST(RigidBodyAmg, SmootherConvergesOnEveryLevelWhereSpai0DoesNot) {
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
    const detail::AmgHierarchy scalar(scaled.matrix);
    EXPECT_GT(largest(*scalar.levels[0].A, scalar.levels[0].smoother), 2.0)
        << "SPAI-0 no longer diverges on elasticity: the reason for l1_jacobi_weights() has gone";
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
