// tests/integration/test_omitted_midside.cpp
// Omitted midside grids of CHEXA20 and CTETRA10 (elements/midside_nodes.hpp)
// through the whole solver.  Omitting a midside grid makes its edge linear,
// so a model with omitted grids must give the same answers as the model with
// those grids present at their edge midpoints and tied to the corners by
// MPCs, u_m = (u_a + u_b) / 2.  Checked for statics under PLOAD4 on a face
// containing an omitted grid, a point force and GRAV (consistent body load),
// for the recovered stresses and grid point force balance, and for the
// frequencies in SOL 103.

#include <gtest/gtest.h>
#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/eigensolver_backend.hpp"
#include "solver/linear_static.hpp"
#include "solver/modal.hpp"
#include "solver/solver_backend.hpp"
#include <array>
#include <cmath>
#include <format>
#include <map>
#include <string>
#include <vector>

using namespace vibestran;

namespace {

struct Grid {
  int id;
  double x, y, z;
};

struct Mesh {
  std::vector<Grid> grids;
  std::string elements;                       // element cards (with 0 slots written blank)
  std::vector<std::array<int, 3>> omitted;    // {midside grid, corner a, corner b}
  std::string loads;                          // load cards, SID 1
  std::string spcs;                           // SPC cards, SID 1
};

/// Writes the deck.  `tied`: the omitted grids are present at their edge
/// midpoints and constrained by MPCs; otherwise their slots are blank.
std::string deck(const Mesh &mesh, bool tied, bool modal) {
  std::string b = modal ? "SOL 103\nCEND\nMETHOD = 1\nSPC = 1\n"
                        : "SOL 101\nCEND\nLOAD = 1\nSPC = 1\nDISPLACEMENT = ALL\n"
                          "STRESS = ALL\nGPFORCE = ALL\n";
  if (tied)
    b += "MPC = 10\n";
  b += "BEGIN BULK\nMAT1,1,1.0E6,,0.3,2.0\nPSOLID,1,1\n";
  if (modal)
    b += "EIGRL,1,,,6\n";
  std::map<int, Grid> pos;
  for (const Grid &g : mesh.grids)
    pos[g.id] = g;
  for (const Grid &g : mesh.grids)
    b += std::format("GRID,{},,{:.17g},{:.17g},{:.17g}\n", g.id, g.x, g.y, g.z);
  std::string elems = mesh.elements;
  for (const auto &[m, a, c] : mesh.omitted) {
    const std::string tag = std::format("<{}>", m);
    const auto at = elems.find(tag);
    elems.replace(at, tag.size(), tied ? std::to_string(m) : "");
    if (tied) {
      const Grid &ga = pos.at(a), &gc = pos.at(c);
      b += std::format("GRID,{},,{:.17g},{:.17g},{:.17g}\n", m, 0.5 * (ga.x + gc.x),
                       0.5 * (ga.y + gc.y), 0.5 * (ga.z + gc.z));
      for (int d = 1; d <= 3; ++d)
        b += std::format("MPC,10,{},{},1.0,{},{},-0.5\n,,{},{},-0.5\n", m, d, a, d, c, d);
    }
  }
  b += elems;
  if (!modal)
    b += mesh.loads;
  b += mesh.spcs + "ENDDATA\n";
  return b;
}

/// A CHEXA8 (z 0..1) under a CHEXA20 (z 1..2) on a 2 x 1 footprint.  The
/// CHEXA20's bottom-face midside grids (shared with the CHEXA8, which has
/// none) and the top-edge grid G19 are omitted; its vertical-edge midside
/// grids bulge outward, so the element is curved.  Clamped at z = 0.
Mesh hexa_mesh() {
  Mesh m;
  const std::array<std::array<double, 2>, 4> xy{{{0, 0}, {2, 0}, {2, 1}, {0, 1}}};
  for (int level = 0; level < 3; ++level)
    for (int k = 0; k < 4; ++k)
      m.grids.push_back({1 + 4 * level + k, xy[static_cast<size_t>(k)][0],
                         xy[static_cast<size_t>(k)][1], static_cast<double>(level)});
  // CHEXA20 grids: G1-G4 = 5-8 (z = 1), G5-G8 = 9-12 (z = 2).
  // G9-G12 (bottom face) omitted: 101-104; G13-G16 vertical: 105-108
  // (bulging); G17, G18, G20 top present: 109, 110, 112; G19 omitted: 111.
  const std::array<std::array<double, 2>, 4> bulge{{{-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1}}};
  for (int k = 0; k < 4; ++k)
    m.grids.push_back({105 + k, xy[static_cast<size_t>(k)][0] + bulge[static_cast<size_t>(k)][0],
                       xy[static_cast<size_t>(k)][1] + bulge[static_cast<size_t>(k)][1], 1.5});
  m.grids.push_back({109, 1.0, 0.0, 2.0});
  m.grids.push_back({110, 2.0, 0.5, 2.0});
  m.grids.push_back({112, 0.0, 0.5, 2.0});
  m.omitted = {{101, 5, 6}, {102, 6, 7}, {103, 7, 8}, {104, 8, 5}, {111, 11, 12}};
  m.elements = "CHEXA,1,1,1,2,3,4,5,6\n,7,8\n"
               "CHEXA,2,1,5,6,7,8,9,10\n,11,12,<101>,<102>,<103>,<104>,105,106\n"
               ",107,108,109,110,<111>,112\n";
  // Pressure on the top face (it contains the omitted G19), a point force
  // and gravity.
  m.loads = "PLOAD4,1,2,100.,,,,9,11\nFORCE,1,10,,1.0,30.,-20.,10.\nGRAV,1,,9.81,0.3,0.,-1.\n";
  m.spcs = "SPC1,1,123,1,2,3,4\n";
  return m;
}

/// A single CTETRA10 with G6 (edge G2-G3) and G10 (edge G3-G4) omitted and
/// curved present edges, clamped on its G1-G2-G3 face.
Mesh tetra_mesh() {
  Mesh m;
  m.grids = {{1, 0, 0, 0},          {2, 1.2, 0.1, 0},       {3, 0.2, 1.1, 0.0},
             {4, 0.1, 0.2, 0.9},    {5, 0.6, 0.0, 0.0},     {7, 0.1, 0.55, 0.0},
             {8, 0.05, 0.1, 0.45},  {9, 0.7, 0.2, 0.45}};
  m.omitted = {{6, 2, 3}, {10, 3, 4}};
  m.elements = "CTETRA,1,1,1,2,3,4,5,<6>\n,7,8,9,<10>\n";
  // Pressure on face G2-G3-G4, which holds both omitted grids (G1 is the
  // corner off the face), a point force at the apex and gravity.
  m.loads = "PLOAD4,1,1,100.,,,,2,1\nFORCE,1,4,,1.0,20.,10.,-30.\nGRAV,1,,9.81,0.,1.,0.\n";
  m.spcs = "SPC1,1,123,1,2,3,5,7\n";
  return m;
}

SolverResults run_static(const std::string &bdf) {
  Model model = BdfParser::parse_string(bdf);
  LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
  return solver.solve(model);
}

std::vector<double> frequencies(const std::string &bdf) {
  Model model = BdfParser::parse_string(bdf);
  ModalSolver solver(std::make_unique<SpectraEigensolverBackend>());
  const ModalSolverResults res = solver.solve(model);
  std::vector<double> f;
  for (const auto &mode : res.subcases.at(0).modes)
    f.push_back(mode.cycles_per_sec);
  return f;
}

void expect_equivalent(const Mesh &mesh) {
  const SolverResults omitted = run_static(deck(mesh, false, false));
  const SolverResults tied = run_static(deck(mesh, true, false));
  const auto &so = omitted.subcases.at(0), &st = tied.subcases.at(0);

  // Displacements at every grid of the omitted model.
  std::map<int, std::array<double, 6>> du;
  for (const auto &nd : st.displacements)
    du[nd.node.value] = nd.d;
  double scale = 0.0;
  for (const auto &nd : so.displacements)
    for (int k = 0; k < 3; ++k)
      scale = std::max(scale, std::abs(nd.d[static_cast<size_t>(k)]));
  ASSERT_GT(scale, 0.0);
  for (const auto &nd : so.displacements) {
    ASSERT_NE(nd.node.value, 0);
    for (std::size_t k = 0; k < 3; ++k)
      EXPECT_NEAR(nd.d[k], du.at(nd.node.value)[k], 1e-10 * scale) << "grid " << nd.node.value;
  }

  // Stresses at the centre and every reported grid of every element.
  ASSERT_EQ(so.solid_stresses.size(), st.solid_stresses.size());
  for (std::size_t e = 0; e < so.solid_stresses.size(); ++e) {
    const auto &a = so.solid_stresses[e], &b = st.solid_stresses[e];
    const double s = std::max(1.0, std::abs(b.von_mises));
    EXPECT_NEAR(a.von_mises, b.von_mises, 1e-9 * s) << "element " << a.eid.value;
    EXPECT_NEAR(a.sz, b.sz, 1e-9 * s) << "element " << a.eid.value;
    std::map<int, double> vm;
    for (const auto &p : b.nodal)
      vm[p.node.value] = p.von_mises;
    for (const auto &p : a.nodal) {
      ASSERT_NE(p.node.value, 0);
      EXPECT_NEAR(p.von_mises, vm.at(p.node.value), 1e-9 * s) << "grid " << p.node.value;
    }
  }

  // Grid point force balance: no row for an omitted grid, and every grid in
  // equilibrium.
  ASSERT_FALSE(so.gp_forces.empty());
  for (const auto &gp : so.gp_forces) {
    EXPECT_NE(gp.node.value, 0);
    for (double t : gp.totals)
      EXPECT_LT(std::abs(t), 1e-8 * 100.0) << "grid " << gp.node.value;
  }

  const auto fo = frequencies(deck(mesh, false, true));
  const auto ft = frequencies(deck(mesh, true, true));
  ASSERT_GE(fo.size(), 6u);
  ASSERT_GE(ft.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_NEAR(fo[i], ft[i], 1e-8 * ft[i]) << "mode " << i + 1;
}

} // namespace

TEST(OmittedMidside, Chexa20EqualsTheModelWithMidpointGridsTiedByMpcs) {
  expect_equivalent(hexa_mesh());
}

TEST(OmittedMidside, Ctetra10EqualsTheModelWithMidpointGridsTiedByMpcs) {
  expect_equivalent(tetra_mesh());
}
