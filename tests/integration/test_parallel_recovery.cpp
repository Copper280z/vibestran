// Stresses and constraint forces are recovered in chunks of elements on as many
// threads as there are. The results must not know: same order, same bits.
#include "../unit/structural_system.hpp"
#include "core/coord_sys.hpp"
#include "io/bdf_parser.hpp"
#include "solver/analysis_support.hpp"
#include "solver/heat_transfer_steady.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <format>
#include <sstream>
#include <string>
#include <vector>
#ifdef HAVE_TBB
#include <tbb/task_arena.h>
#endif

using namespace vibestran;

namespace {

// More than a few chunks of every family, with element IDs out of order, a line and
// a scalar element at grids of the plate, an MPC-coupled part, and a thermal load.
Model recovery_model() {
    std::ostringstream deck;
    deck << "SOL 101\nCEND\nSPC = 1\nLOAD = 2\nTEMPERATURE(LOAD) = 7\nDISPLACEMENT = ALL\nSTRESS = ALL\nSTRAIN = ALL\n"
            "SPCFORCES = ALL\nMPCFORCES = ALL\nGPFORCE = ALL\nBEGIN BULK\nTEMPD,7,20.\n"
         << test::mixed_assembly(32);
    constexpr int kFirst = 3000000;
    for (int i = 0; i < 16; ++i) {
        const int plate_grid = 1 + 16 + 33 * i;
        deck << std::format("CBAR,{},3,{},{},0.,0.,1.\n", kFirst + 200 - i, plate_grid, plate_grid + 33);
        deck << std::format("CELAS2,{},1000.,{},3\n", kFirst + 300 + i, plate_grid);
        deck << std::format("CROD,{},9,{},{}\n", kFirst + 400 + i, plate_grid, plate_grid + 1);
    }
    deck << "PROD,9,1,1.-4\nENDDATA\n";
    return BdfParser::parse_string(deck.str());
}

// Every number of the results in order, with the IDs that say which element or grid
// each belongs to.
std::vector<double> flattened(const SubCaseResults& r) {
    std::vector<double> out;
    const auto put = [&out](auto... values) { (out.push_back(double(values)), ...); };
    const auto array = [&out](const auto& values) { out.insert(out.end(), values.begin(), values.end()); };
    const auto point = [&](const auto& p) {
        put(p.sx, p.sy, p.sxy, p.mx, p.my, p.mxy, p.von_mises, p.nx, p.ny, p.nxy, p.qx, p.qy);
        put(p.strain.ex, p.strain.ey, p.strain.exy, p.strain.kx, p.strain.ky, p.strain.kxy, p.strain.gxz, p.strain.gyz);
    };
    for (const auto& d : r.displacements) { put(d.node.value); array(d.d); }
    for (const auto& f : r.spc_forces) { put(f.node.value); array(f.f); }
    for (const auto& f : r.mpc_forces) { put(f.node.value); array(f.f); }
    for (const auto& grid : r.gp_forces) {
        put(grid.node.value);
        array(grid.totals);
        for (const auto& row : grid.rows) { put(int(row.kind), row.eid.value); array(row.f); }
    }
    for (const auto& s : r.plate_stresses) {
        put(s.eid.value);
        point(s);
        for (const auto& n : s.nodal) { put(n.node.value); point(n); }
    }
    for (const auto& s : r.solid_stresses) { put(s.eid.value, s.sx, s.sy, s.sz, s.sxy, s.syz, s.szx, s.von_mises); }
    for (const auto& s : r.solid_strains) { put(s.eid.value, s.ex, s.ey, s.ez, s.exy, s.eyz, s.ezx, s.von_mises); }
    for (const auto& s : r.line_stresses) {
        put(s.eid.value, s.end_a.axial, s.end_a.smax, s.end_a.smin, s.end_b.axial, s.end_b.smax, s.end_b.smin);
        array(s.end_a.s);
        array(s.end_b.s);
    }
    for (const auto& s : r.rod_results) put(s.eid.value, s.axial_force, s.torque, s.axial_stress, s.torsional_stress);
    for (const auto& s : r.spring_results) put(s.eid.value, s.force, s.stress);
    return out;
}

template<class Results>
std::vector<int> element_ids(const std::vector<Results>& results) {
    std::vector<int> out(results.size());
    std::ranges::transform(results, out.begin(), [](const auto& r) { return r.eid.value; });
    return out;
}

// The IDs of a family's elements in the order the model holds them.
std::vector<int> ids_in_model(const Model& model, std::initializer_list<ElementType> types) {
    std::vector<int> out;
    for (const auto& e : model.elements)
        if (std::ranges::find(types, e.type) != types.end()) out.push_back(e.id.value);
    return out;
}

} // namespace

TEST(ParallelRecovery, ResultsAreIndependentOfTheThreadCountAndInElementOrder) {
#ifdef HAVE_TBB
    const Model model = recovery_model();
    const auto solve = [&] {
        // Not the direct backend: Accelerate is not bitwise reproducible.
        LinearStaticSolver solver(std::make_unique<CpuPCGSolverBackend>(1e-10));
        return solver.solve(model).subcases.at(0);
    };
    SubCaseResults serial;
    tbb::task_arena(1).execute([&] { serial = solve(); });
    const SubCaseResults parallel = solve();

    EXPECT_GT(model.elements.size(), std::size_t(10) * 512) << "a test of chunks needs several of them";
    ASSERT_FALSE(serial.gp_forces.empty());
    ASSERT_FALSE(serial.line_stresses.empty());
    ASSERT_FALSE(serial.rod_results.empty());
    ASSERT_FALSE(serial.spring_results.empty());
    ASSERT_FALSE(serial.solid_strains.empty());
    const auto a = flattened(serial), b = flattened(parallel);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_TRUE(a == b);

    EXPECT_EQ(element_ids(parallel.plate_stresses), ids_in_model(model, {ElementType::CQUAD4}));
    EXPECT_EQ(element_ids(parallel.solid_stresses), ids_in_model(model, {ElementType::CHEXA8}));
    EXPECT_EQ(element_ids(parallel.solid_strains), ids_in_model(model, {ElementType::CHEXA8}));
    EXPECT_EQ(element_ids(parallel.line_stresses), ids_in_model(model, {ElementType::CBAR}));
    EXPECT_EQ(element_ids(parallel.rod_results), ids_in_model(model, {ElementType::CROD}));
    EXPECT_EQ(element_ids(parallel.spring_results), ids_in_model(model, {ElementType::CELAS2}));

    // The element forces at a grid balance what it carries, whichever chunk they came
    // from, with the rows in element ID order.
    for (const auto& grid : parallel.gp_forces) {
        for (const double total : grid.totals) EXPECT_NEAR(total, 0.0, 1e-4) << "grid " << grid.node.value;
        int last = 0;
        for (const auto& row : grid.rows) {
            if (row.kind != GridForceContribution::Kind::Element) continue;
            EXPECT_GT(row.eid.value, last) << "grid " << grid.node.value;
            last = row.eid.value;
        }
    }
#else
    GTEST_SKIP() << "built without TBB: recovery is serial";
#endif
}

// Thermal K and Q go to the backend the same, to the bit, whatever the thread count.
TEST(ParallelAssembly, ThermalSystemIsIndependentOfTheThreadCount) {
#ifdef HAVE_TBB
    const int n = 14;
    const auto node = [&](int i, int j, int k) { return 1 + i + (n + 1) * (j + (n + 1) * k); };
    std::ostringstream deck;
    deck << "SOL 153\nCEND\nSUBCASE 1\nSPC=1\nLOAD=2\nTHERMAL=ALL\nBEGIN BULK\nMAT4,1,37.\nPSOLID,1,1\n";
    for (int k = 0; k <= n; ++k)
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i <= n; ++i)
                deck << std::format("GRID,{},,{},{},{}\n", node(i, j, k), (i + .3 * j) / 7., (j + .1 * k) / 3., k / 11.);
    int eid = 1;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                deck << std::format("CHEXA,{},1,{},{},{},{},{},{},+\n+,{},{}\n", eid, node(i, j, k), node(i + 1, j, k),
                                    node(i + 1, j + 1, k), node(i, j + 1, k), node(i, j, k + 1), node(i + 1, j, k + 1),
                                    node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1));
                deck << std::format("QVOL,2,{},,{}\n", 11.0 + eid % 7, eid);
                ++eid;
            }
    for (int k = 0; k <= n; ++k)
        for (int j = 0; j <= n; ++j) deck << "SPC,1," << node(0, j, k) << ",0,0.\n";
    deck << "ENDDATA\n";
    const Model model = BdfParser::parse_string(deck.str());
    EXPECT_GT(model.elements.size(), std::size_t(5) * 512) << "a test of chunks needs several of them";

    const auto assembled = [&] {
        test::RecordingBackend::Record record;
        HeatTransferSteadySolver solver(std::make_unique<test::RecordingBackend>(record, false));
        (void)solver.solve(model);
        return record;
    };
    test::RecordingBackend::Record serial;
    tbb::task_arena(1).execute([&] { serial = assembled(); });
    const auto parallel = assembled();

    ASSERT_GT(serial.k.nnz, 0);
    EXPECT_EQ(serial.k.row_ptr, parallel.k.row_ptr);
    EXPECT_EQ(serial.k.col_ind, parallel.k.col_ind);
    EXPECT_EQ(serial.k.values, parallel.k.values);
    EXPECT_EQ(serial.f, parallel.f);
#else
    GTEST_SKIP() << "built without TBB";
#endif
}

TEST(ParallelAssembly, AutoSpcIsIndependentOfTheThreadCount) {
#ifdef HAVE_TBB
    // A chain of bars with no J along a skewed line, every bar on a CD grid every
    // third grid, so the held components differ from grid to grid.
    const int bars = 3 * 512;
    std::ostringstream deck;
    deck << "SOL 101\nCEND\nSPC = 1\nBEGIN BULK\nMAT1,1,1.0E7,,0.3\nPBAR,1,1,2.0,3.0,3.0\n"
         << "CORD2R,5,,0.,0.,0.,0.3,0.4,0.8\n,0.9,0.1,0.2\n";
    for (int g = 1; g <= bars + 1; ++g)
        deck << std::format("GRID,{},,{},{},{},{}\n", g, 0.6 * g, 0.8 * g, 0.,
                            g % 3 == 0 ? "5" : "");
    for (int e = 1; e <= bars; ++e)
        deck << std::format("CBAR,{},1,{},{},0.,0.,1.\n", e, e, e + 1);
    deck << "SPC1,1,123456,1\nENDDATA\n";
    const Model model = BdfParser::parse_string(deck.str());
    const SubCase sc = model.analysis.subcases.at(0);

    const auto masks = [&] {
        std::vector<std::pair<int, unsigned>> out;
        for (const Spc& spc : analysis_spcs(model, sc))
            out.emplace_back(spc.node.value, spc.dofs.mask);
        return out;
    };
    std::vector<std::pair<int, unsigned>> serial;
    tbb::task_arena(1).execute([&] { serial = masks(); });
    std::vector<std::pair<int, unsigned>> threaded;
    tbb::task_arena(8).execute([&] { threaded = masks(); });
    EXPECT_EQ(serial.size(), std::size_t(bars) + 1) << "the anchor and each free grid";
    EXPECT_EQ(serial, threaded);
    EXPECT_TRUE(std::ranges::is_sorted(serial.begin() + 1, serial.end()));
#else
    GTEST_SKIP() << "built without TBB";
#endif
}
