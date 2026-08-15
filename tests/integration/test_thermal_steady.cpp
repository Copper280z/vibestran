// tests/integration/test_thermal_steady.cpp
// Integration tests for linear steady-state heat transfer (SOL 153).
//
// Test 1: 1-D heat conduction through a single CHEXA8
//   - Unit cube, k = 50 W/(m·K).
//   - Two opposite faces held at 100 °C and 0 °C respectively.
//   - Analytic: linear T profile; centroidal flux qx = -k·ΔT/L = +5000 W/m².
//
// Test 2: Volumetric heat source in a fixed-fixed bar (CHEXA8)
//   - 1-D conduction with q_vol uniform; both ends held at 0 °C.
//   - Analytic centerline T = q_vol·L²/(8·k).
//
// Test 3: 1-D conduction through a single CTETRA4 prism (sanity)
//   - Two corners at 100 °C, two at 0 °C: linear field reproduces exactly.

#include "elements/chbdy_element.hpp"
#include "core/quality_checks.hpp"
#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/eigensolver_backend.hpp"
#include "solver/heat_transfer_steady.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string_view>

using namespace vibestran;

namespace {

SolverResults run_thermal(const std::string &bdf) {
  Model m = BdfParser::parse_string(bdf);
  HeatTransferSteadySolver solver(std::make_unique<EigenSolverBackend>());
  return solver.solve(m);
}

double temp_at(const SolverResults &r, int node_id) {
  for (const auto &sc : r.subcases) {
    auto it = std::find_if(sc.temperatures.begin(), sc.temperatures.end(),
                           [node_id](const NodeTemperature &nt) {
                             return nt.node.value == node_id;
                           });
    if (it != sc.temperatures.end())
      return it->temperature;
  }
  ADD_FAILURE() << "node " << node_id << " not found";
  return 0.0;
}

double flux_x_of(const SolverResults &r, int eid) {
  for (const auto &sc : r.subcases) {
    auto it = std::find_if(sc.heat_fluxes.begin(), sc.heat_fluxes.end(),
                           [eid](const ElementHeatFlux &ef) {
                             return ef.eid.value == eid;
                           });
    if (it != sc.heat_fluxes.end())
      return it->q[0];
  }
  ADD_FAILURE() << "element " << eid << " heat flux not found";
  return 0.0;
}

} // namespace

TEST(ThermalSteady, CHexa8_1D_Conduction_LinearProfile) {
  // Unit cube; faces x=0 at T=100, x=1 at T=0.  k=50.
  // Analytic: T(x) = 100·(1−x); flux qx = -k·dT/dx = -k·(-100) = 5000.
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << "  THERMAL = ALL\n"
      << "  FLUX    = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,50.0\n"
      << "PSOLID,1,1\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,1.0,1.0,0.0\n"
      << "GRID,4,,0.0,1.0,0.0\n"
      << "GRID,5,,0.0,0.0,1.0\n"
      << "GRID,6,,1.0,0.0,1.0\n"
      << "GRID,7,,1.0,1.0,1.0\n"
      << "GRID,8,,0.0,1.0,1.0\n"
      << "CHEXA,1,1,1,2,3,4,5,6,+\n"
      << "+,7,8\n"
      // Face x=0: nodes 1,4,5,8 → T = 100
      << "SPC,1,1,0,100.0,4,0,100.0\n"
      << "SPC,1,5,0,100.0,8,0,100.0\n"
      // Face x=1: nodes 2,3,6,7 → T = 0
      << "SPC,1,2,0,0.0,3,0,0.0\n"
      << "SPC,1,6,0,0.0,7,0,0.0\n"
      << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  ASSERT_EQ(r.subcases.size(), 1U);
  // Check boundary temperatures echo correctly
  EXPECT_NEAR(temp_at(r, 1), 100.0, 1e-9);
  EXPECT_NEAR(temp_at(r, 2),   0.0, 1e-9);
  // The two opposite faces fully define the field; CHEXA8 is exact for linear T.
  // (Interior nodes don't exist here — all 8 are prescribed.)
  // Heat flux: qx = +5000 (positive x because dT/dx<0 and q=-k∇T)
  EXPECT_NEAR(flux_x_of(r, 1), 5000.0, 1e-6);
}

TEST(ThermalSteady, CHexa8_VolumetricSource_FixedEndTemps) {
  // 10-element bar along x ∈ [0, L], L = 1; cross-section 1×1; k = 10.
  // q_vol = 100; ends T=0.
  // 1-D analytic: T(x) = q_vol·x·(L−x)/(2k).  Centerline (x=L/2):
  //   T_c = q_vol·L²/(8k) = 100·1/(80) = 1.25.
  const int NEL = 10;
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  LOAD = 1\n"
      << "  SPC = 2\n"
      << "  THERMAL = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,10.0\n"
      << "PSOLID,1,1\n";
  auto grid = [&](int id, double x, double y, double z) {
    bdf << "GRID," << id << ",," << x << "," << y << "," << z << "\n";
  };
  for (int i = 0; i <= NEL; ++i) {
    const double x = static_cast<double>(i) / NEL;
    grid(1 + 4 * i + 0, x, 0.0, 0.0);
    grid(1 + 4 * i + 1, x, 1.0, 0.0);
    grid(1 + 4 * i + 2, x, 1.0, 1.0);
    grid(1 + 4 * i + 3, x, 0.0, 1.0);
  }
  // CHEXA elements: node order (x,0,0)(x',0,0)(x',1,0)(x,1,0)(x,0,1)(x',0,1)(x',1,1)(x,1,1)
  for (int e = 0; e < NEL; ++e) {
    const int a = 1 + 4 * e, b = 1 + 4 * (e + 1);
    bdf << "CHEXA," << (e + 1) << ",1,"
        << a << "," << b << "," << (b + 1) << "," << (a + 1) << ","
        << (a + 3) << "," << (b + 3) << ",+\n"
        << "+," << (b + 2) << "," << (a + 2) << "\n";
  }
  // Dirichlet: hold x=0 face (nodes 1..4) and x=L face (last 4) at 0.
  bdf << "SPC,2,1,0,0.0,2,0,0.0\n"
      << "SPC,2,3,0,0.0,4,0,0.0\n";
  for (int n = 0; n < 4; ++n)
    bdf << "SPC,2," << (1 + 4 * NEL + n) << ",0,0.0\n";
  // Volume source applied to all elements
  bdf << "QVOL,1,100.0";
  for (int e = 0; e < NEL; ++e) bdf << "," << (e + 1);
  bdf << "\n";
  bdf << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  // Centerline node (x = 0.5).  Choose node at element boundary 5: id 1+4*5 = 21
  const double Tc = temp_at(r, 21);
  EXPECT_NEAR(Tc, 1.25, 0.03);  // 10-element 3-D coarsening tolerance
}

TEST(ThermalSteady, CQuad4_PlateInPlaneConduction) {
  // 1×1 plate, 10 elements along x, thickness 0.01, k = 20.
  // x=0 edge T=200, x=1 edge T=0.  Analytic: T(x) = 200(1−x).
  // Element-local flux: qx = -k·dT/dx = 4000 W/m².
  const int NEL = 10;
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << "  THERMAL = ALL\n"
      << "  FLUX = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,20.0\n"
      << "PSHELL,1,1,0.01\n";
  for (int i = 0; i <= NEL; ++i) {
    const double x = static_cast<double>(i) / NEL;
    bdf << "GRID," << (1 + 2 * i) << ",," << x << ",0.0,0.0\n";
    bdf << "GRID," << (2 + 2 * i) << ",," << x << ",1.0,0.0\n";
  }
  for (int e = 0; e < NEL; ++e) {
    const int a = 1 + 2 * e, b = 1 + 2 * (e + 1);
    bdf << "CQUAD4," << (e + 1) << ",1,"
        << a << "," << b << "," << (b + 1) << "," << (a + 1) << "\n";
  }
  bdf << "SPC,1,1,0,200.0,2,0,200.0\n";
  bdf << "SPC,1," << (1 + 2 * NEL) << ",0,0.0\n";
  bdf << "SPC,1," << (2 + 2 * NEL) << ",0,0.0\n";
  bdf << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  // Mid-plate node (x = 0.5): id 1 + 2*5 = 11 (or 12)
  EXPECT_NEAR(temp_at(r, 11), 100.0, 1e-6);
  EXPECT_NEAR(temp_at(r, 12), 100.0, 1e-6);
  // Centroidal flux: each element local-x aligns with global +x (edge 12 is
  // along +x for each strip), so qx = +4000 (positive: flow in +x direction).
  EXPECT_NEAR(flux_x_of(r, 1), 4000.0, 1e-3);
  EXPECT_NEAR(flux_x_of(r, NEL), 4000.0, 1e-3);
}

TEST(ThermalSteady, CTria3_LinearFieldReproduction) {
  // Single CTRIA3 in XY-plane, k = 2, thickness = 0.5.
  // Prescribed T = x at every node → linear field, flux = -k·(1, 0) in local x.
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << "  FLUX = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,2.0\n"
      << "PSHELL,1,1,0.5\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,0.0,1.0,0.0\n"
      << "CTRIA3,1,1,1,2,3\n"
      << "SPC,1,1,0,0.0,2,0,1.0\n"
      << "SPC,1,3,0,0.0\n"
      << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  // Element's local e1 aligns with edge 12 = global +x, so local qx = -k·1 = -2.
  EXPECT_NEAR(flux_x_of(r, 1), -2.0, 1e-9);
}

TEST(ThermalSteady, Qvect_DirectionalFluxOnCHBDY) {
  // 1×1×1 cube, +z face is a CHBDY AREA4.  QVECT applies flux of
  // magnitude q0=1000 in direction (0,0,-1) — i.e. pointing into +z face.
  // Opposite face (z=0) held at 0; sides insulated by default; QVECT supplies
  // all heat in.  1-D analytic:  qx = -k·dT/dz = q0  ⇒  T(z) = q0·z/k.
  // PHBDY absorptivity is 0.25, so the effective flux is 250 W/m² and with
  // k=10, L=1 the top temperature should be 25.
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  LOAD = 5\n"
      << "  SPC = 1\n"
      << "  THERMAL = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,10.0\n"
      << "PSOLID,1,1\n"
      << "PHBDY,2,,,0.0,0.25\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,1.0,1.0,0.0\n"
      << "GRID,4,,0.0,1.0,0.0\n"
      << "GRID,5,,0.0,0.0,1.0\n"
      << "GRID,6,,1.0,0.0,1.0\n"
      << "GRID,7,,1.0,1.0,1.0\n"
      << "GRID,8,,0.0,1.0,1.0\n"
      << "CHEXA,1,1,1,2,3,4,5,6,+\n"
      << "+,7,8\n"
      // +z face (nodes 5,6,7,8): outward normal = (0,0,+1)
      << "CHBDY,101,2,AREA4,5,6,7,8\n"
      // Hold z=0 face at 0
      << "SPC,1,1,0,0.0,2,0,0.0\n"
      << "SPC,1,3,0,0.0,4,0,0.0\n"
      // QVECT: 1000 W/m² traveling in -z direction (hits +z face inwardly)
      << "QVECT,5,1000.0,0.0,0.0,-1.0,101\n"
      << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  // Top face nodes (5..8): expected T ≈ 25
  EXPECT_NEAR(temp_at(r, 5), 25.0, 1e-6);
  EXPECT_NEAR(temp_at(r, 7), 25.0, 1e-6);
}

TEST(ThermalSteady, CoupledHeatToStaticsBar) {
  // Coupled deck: SOL 153 with two subcases.  Subcase 1 (HEAT) heats a
  // fully-constrained CHEXA8 bar.  Subcase 2 (STATICS) consumes those
  // temperatures via TEMP(LOAD)=THERMAL and computes thermal stress.
  // For a uniformly heated, fully-constrained bar: σ = -E·α·ΔT (compression).
  // Use ΔT = 100 K, α = 1e-5, E = 200e9 → σ = -2e8 Pa.
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  ANALYSIS = HEAT\n"
      << "  SPC = 1\n"
      << "  THERMAL = ALL\n"
      << "SUBCASE 2\n"
      << "  ANALYSIS = STATICS\n"
      << "  SPC = 100\n"
      << "  TEMP(LOAD) = THERMAL\n"
      << "  STRESS = ALL\n"
      << "BEGIN BULK\n"
      // Same MID used by both — MAT1 for statics, MAT4 for thermal
      << "MAT1,1,200.0E9,,0.3,,1.0E-5,0.0\n"
      << "MAT4,1,50.0\n"
      << "PSOLID,1,1\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,1.0,1.0,0.0\n"
      << "GRID,4,,0.0,1.0,0.0\n"
      << "GRID,5,,0.0,0.0,1.0\n"
      << "GRID,6,,1.0,0.0,1.0\n"
      << "GRID,7,,1.0,1.0,1.0\n"
      << "GRID,8,,0.0,1.0,1.0\n"
      << "CHEXA,1,1,1,2,3,4,5,6,+\n"
      << "+,7,8\n"
      // Heat BC: hold ALL nodes at 100 K (uniform heating → constant T)
      << "SPC,1,1,0,100.0,2,0,100.0\n"
      << "SPC,1,3,0,100.0,4,0,100.0\n"
      << "SPC,1,5,0,100.0,6,0,100.0\n"
      << "SPC,1,7,0,100.0,8,0,100.0\n"
      // Structural BC: clamp all 6 DOFs at all nodes (fully constrained cube)
      // Block axial expansion: Tx fixed at x=0 face (nodes 1,4,5,8) and x=1
      // face (nodes 2,3,6,7).  Block rigid-body lateral motion: fix Ty and Tz
      // at one corner (node 1).  Leaves 14 free DOFs; σx = -E·α·ΔT and
      // σy = σz = 0 because the lateral faces are free to expand.
      << "SPC1,100,1,1,4,5,8\n"
      << "SPC1,100,1,2,3,6,7\n"
      << "SPC1,100,23,1\n"
      << "ENDDATA\n";

  // Direct invocation of the coupled flow inside this test
  Model m = BdfParser::parse_string(bdf.str());

  // Partition + run heat
  std::vector<SubCase> heat, stat;
  for (const auto &sc : m.analysis.subcases)
    (sc.analysis_type == SubCaseAnalysis::Statics ? stat : heat).push_back(sc);
  ASSERT_EQ(heat.size(), 1U);
  ASSERT_EQ(stat.size(), 1U);

  Model heat_m = m;
  heat_m.analysis.subcases = heat;
  SolverResults th =
      HeatTransferSteadySolver(std::make_unique<EigenSolverBackend>()).solve(heat_m);

  // Materialize TEMP loads from thermal result, then run statics
  int sid = 9999;
  Model stat_m = m;
  stat_m.analysis.sol = SolutionType::LinearStatic;
  for (const auto &nt : th.subcases.back().temperatures) {
    TempLoad t;
    t.sid = LoadSetId(sid);
    t.node = nt.node;
    t.temperature = nt.temperature;
    stat_m.loads.emplace_back(t);
  }
  stat[0].temp_load_set = sid;
  stat_m.analysis.subcases = {stat[0]};
  // Need LinearStaticSolver included
  SolverResults sr;
  {
    LinearStaticSolver lss(std::make_unique<EigenSolverBackend>());
    sr = lss.solve(stat_m);
  }
  ASSERT_FALSE(sr.subcases.empty());
  ASSERT_FALSE(sr.subcases[0].solid_stresses.empty());
  const auto &ss = sr.subcases[0].solid_stresses[0];
  // x is axially constrained (Tx fixed at every node, both end faces); y, z
  // are free (only node 1 has Ty, Tz fixed for rigid-body suppression).
  //   σx = -E·α·ΔT = -2e8 Pa
  //   σy ≈ 0, σz ≈ 0  (lateral faces traction-free)
  const double expected_sx = -200.0e9 * 1.0e-5 * 100.0;  // = -2e8
  EXPECT_NEAR(ss.sx, expected_sx, 1.0e4);
  EXPECT_NEAR(ss.sy, 0.0,        1.0e4);
  EXPECT_NEAR(ss.sz, 0.0,        1.0e4);
}

TEST(ThermalSteady, CHexa20_LinearFieldReproduction_Full) {
  // Full 20-node CHEXA, k=1, linear field T = x prescribed at every node.
  // Centroidal flux should be -k·(1,0,0) = (-1, 0, 0).
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << "  FLUX = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,1.0\n"
      << "PSOLID,1,1\n";
  // 8 corners of unit cube at integer ID 1..8, 12 edge midnodes at 11..22
  struct N { int id; double x, y, z; };
  const std::array<N, 8> corners{{
      {1, 0, 0, 0}, {2, 1, 0, 0}, {3, 1, 1, 0}, {4, 0, 1, 0},
      {5, 0, 0, 1}, {6, 1, 0, 1}, {7, 1, 1, 1}, {8, 0, 1, 1}}};
  const std::array<N, 12> mids{{
      {11, 0.5, 0, 0},   {12, 1, 0.5, 0},  {13, 0.5, 1, 0},  {14, 0, 0.5, 0},
      {15, 0.5, 0, 1},   {16, 1, 0.5, 1},  {17, 0.5, 1, 1},  {18, 0, 0.5, 1},
      {19, 0, 0, 0.5},   {20, 1, 0, 0.5},  {21, 1, 1, 0.5},  {22, 0, 1, 0.5}}};
  for (const auto &c : corners)
    bdf << "GRID," << c.id << ",," << c.x << "," << c.y << "," << c.z << "\n";
  for (const auto &m : mids)
    bdf << "GRID," << m.id << ",," << m.x << "," << m.y << "," << m.z << "\n";
  // CHEXA with 20 grids: G1..G8 then G9..G20 (NASTRAN ordering)
  bdf << "CHEXA,1,1,1,2,3,4,5,6,+1\n"
      << "+1,7,8,11,12,13,14,15,16,+2\n"
      << "+2,17,18,19,20,21,22\n";
  // Prescribe T = x at every node.
  auto t = [&](int id, double x) {
    bdf << "SPC,1," << id << ",0," << x << "\n";
  };
  for (const auto &c : corners) t(c.id, c.x);
  for (const auto &m : mids) t(m.id, m.x);
  bdf << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  EXPECT_NEAR(flux_x_of(r, 1), -1.0, 1e-9);
}

namespace {

// Build a single-CHEXA20 unit-cube deck with two opposite faces held at T=0
// and uniform volumetric heat generation `q_vol`.  `axis` selects the
// conduction direction (0=x, 1=y, 2=z): the two faces normal to that axis
// are pinned at 0, every other node is free.
//
// Analytic 1-D solution with both ends at zero and uniform source:
//   T(s) = q_vol · s · (L - s) / (2·k),  centerline (s=L/2): q_vol·L²/(8k).
// A quadratic element reproduces this exactly (the solution is a polynomial
// of degree 2).
std::string build_cube_chexa20_quadratic_test(int axis, double q_vol, double k) {
  // Corners at integer ID 1..8; edge midnodes at 11..22 (matching the layout
  // used by CHexa20_LinearFieldReproduction_Full).
  struct N { int id; double x, y, z; };
  const std::array<N, 8> corners{{
      {1, 0, 0, 0}, {2, 1, 0, 0}, {3, 1, 1, 0}, {4, 0, 1, 0},
      {5, 0, 0, 1}, {6, 1, 0, 1}, {7, 1, 1, 1}, {8, 0, 1, 1}}};
  // Midnodes 9..20 (stored with IDs 11..22) — same NASTRAN ordering as
  // CHEXA20: bottom-face edges first, then top-face edges, then vertical.
  const std::array<N, 12> mids{{
      {11, 0.5, 0, 0},   {12, 1, 0.5, 0},  {13, 0.5, 1, 0},  {14, 0, 0.5, 0},
      {15, 0.5, 0, 1},   {16, 1, 0.5, 1},  {17, 0.5, 1, 1},  {18, 0, 0.5, 1},
      {19, 0, 0, 0.5},   {20, 1, 0, 0.5},  {21, 1, 1, 0.5},  {22, 0, 1, 0.5}}};

  // Compute the coordinate of each node along the conduction axis.
  auto axis_coord = [&](double x, double y, double z) {
    return (axis == 0) ? x : (axis == 1) ? y : z;
  };
  auto on_axis_face = [&](double s) {  // 0.0 or 1.0 → boundary face
    return std::abs(s - 0.0) < 1e-12 || std::abs(s - 1.0) < 1e-12;
  };

  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  LOAD = 2\n"
      << "  SPC = 1\n"
      << "  THERMAL = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1," << k << "\n"
      << "PSOLID,1,1\n";
  for (const auto &c : corners)
    bdf << "GRID," << c.id << ",," << c.x << "," << c.y << "," << c.z << "\n";
  for (const auto &m : mids)
    bdf << "GRID," << m.id << ",," << m.x << "," << m.y << "," << m.z << "\n";
  bdf << "CHEXA,1,1,1,2,3,4,5,6,+1\n"
      << "+1,7,8,11,12,13,14,15,16,+2\n"
      << "+2,17,18,19,20,21,22\n";
  // Constrain every node lying on either boundary face (corner OR midnode).
  auto add_temp = [&](int id, double T) {
    bdf << "SPC,1," << id << ",0," << T << "\n";
  };
  for (const auto &c : corners)
    if (on_axis_face(axis_coord(c.x, c.y, c.z))) add_temp(c.id, 0.0);
  for (const auto &m : mids)
    if (on_axis_face(axis_coord(m.x, m.y, m.z))) add_temp(m.id, 0.0);
  // Volumetric heat source
  bdf << "QVOL,2," << q_vol << ",1\n"
      << "ENDDATA\n";
  return bdf.str();
}

}  // namespace

TEST(ThermalSteady, CHexa20_QuadraticShapeRecovered_X) {
  // Bar conducting along +x: hold x=0 and x=1 faces at T=0; uniform q_vol = 8;
  // k = 1.  Centerline (x = 0.5) T = q_vol·L²/(8k) = 1.0 — exact for a
  // quadratic element since the analytic solution is degree-2 in x.
  // Midnodes lying on the x=0.5 plane (IDs 11, 13, 15, 17 in our layout)
  // should all read 1.0 to within tight tolerance.
  const auto r = run_thermal(build_cube_chexa20_quadratic_test(0, 8.0, 1.0));
  for (int id : {11, 13, 15, 17})
    EXPECT_NEAR(temp_at(r, id), 1.0, 1e-10) << "midnode " << id;
}

TEST(ThermalSteady, CHexa20_QuadraticShapeRecovered_Y) {
  // Same as the X test, but conducting along +y.  Midnodes at y=0.5 are
  // IDs 12, 14, 16, 18.
  const auto r = run_thermal(build_cube_chexa20_quadratic_test(1, 8.0, 1.0));
  for (int id : {12, 14, 16, 18})
    EXPECT_NEAR(temp_at(r, id), 1.0, 1e-10) << "midnode " << id;
}

TEST(ThermalSteady, CHexa20_QuadraticShapeRecovered_Z) {
  // Same as above, conducting along +z.  Midnodes at z=0.5 are the four
  // vertical-edge midnodes 19, 20, 21, 22.
  const auto r = run_thermal(build_cube_chexa20_quadratic_test(2, 8.0, 1.0));
  for (int id : {19, 20, 21, 22})
    EXPECT_NEAR(temp_at(r, id), 1.0, 1e-10) << "midnode " << id;
}

TEST(ThermalSteady, CHexa20_TransitionWithMissingMidnodes) {
  // CHEXA with 8 corners + 4 midnodes (only the 4 bottom-face edges).
  // Prescribe T = x — for the omitted-midnode edges this linearly interpolates
  // between corners, which still EXACTLY reproduces T = x.  Centroidal flux
  // should still be -k·(1,0,0) = (-1, 0, 0).
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << "  FLUX = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,1.0\n"
      << "PSOLID,1,1\n"
      << "GRID,1,,0,0,0\n"
      << "GRID,2,,1,0,0\n"
      << "GRID,3,,1,1,0\n"
      << "GRID,4,,0,1,0\n"
      << "GRID,5,,0,0,1\n"
      << "GRID,6,,1,0,1\n"
      << "GRID,7,,1,1,1\n"
      << "GRID,8,,0,1,1\n"
      // Only the 4 bottom-face edge midnodes (9,10,11,12) present; all 8
      // others omitted (literal blanks below).
      << "GRID,11,,0.5,0,0\n"
      << "GRID,12,,1,0.5,0\n"
      << "GRID,13,,0.5,1,0\n"
      << "GRID,14,,0,0.5,0\n"
      // CHEXA20 with positional blanks for the absent midnodes 13..20.
      // The third continuation line provides slots G13..G20, all left blank
      // for this transition element.
      << "CHEXA,1,1,1,2,3,4,5,6,+1\n"
      << "+1,7,8,11,12,13,14,,,+2\n"
      << "+2,,,,,,,,\n"
      << "SPC,1,1,0,0.0\n"
      << "SPC,1,2,0,1.0\n"
      << "SPC,1,3,0,1.0\n"
      << "SPC,1,4,0,0.0\n"
      << "SPC,1,5,0,0.0\n"
      << "SPC,1,6,0,1.0\n"
      << "SPC,1,7,0,1.0\n"
      << "SPC,1,8,0,0.0\n"
      << "SPC,1,11,0,0.5\n"
      << "SPC,1,12,0,1.0\n"
      << "SPC,1,13,0,0.5\n"
      << "SPC,1,14,0,0.0\n"
      << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  EXPECT_NEAR(flux_x_of(r, 1), -1.0, 1e-9);
}

TEST(ThermalSteady, CTetra10_TransitionWithMissingMidnodes) {
  // CTETRA with 4 corners + only the 3 midnodes on edges (1,2), (1,3), (1,4)
  // present (midnodes 5, 7, 8).  Linear field T = x reproduces exactly.
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << "  FLUX = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,1.0\n"
      << "PSOLID,1,1\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,0.0,1.0,0.0\n"
      << "GRID,4,,0.0,0.0,1.0\n"
      << "GRID,5,,0.5,0.0,0.0\n"  // midnode edge (1,2)
      << "GRID,7,,0.0,0.5,0.0\n"  // midnode edge (1,3)
      << "GRID,8,,0.0,0.0,0.5\n"  // midnode edge (1,4)
      // Positional CTETRA: slot 5 = midnode(1,2), 6 = (2,3) absent,
      // 7 = (1,3), 8 = (1,4), 9 = (2,4) absent, 10 = (3,4) absent.
      << "CTETRA,1,1,1,2,3,4,5,,+\n"
      << "+,7,8,,\n"
      << "SPC,1,1,0,0.0\n"
      << "SPC,1,2,0,1.0\n"
      << "SPC,1,3,0,0.0\n"
      << "SPC,1,4,0,0.0\n"
      << "SPC,1,5,0,0.5\n"
      << "SPC,1,7,0,0.0\n"
      << "SPC,1,8,0,0.0\n"
      << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  EXPECT_NEAR(flux_x_of(r, 1), -1.0, 1e-9);
}

TEST(ThermalSteady, CTetra4_LinearFieldReproduction) {
  // Unit tetrahedron with T_i = x_i prescribed (linear field along x).
  // Solver should reproduce that linear field — and centroidal flux = -k·(1,0,0).
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << "  THERMAL = ALL\n"
      << "  FLUX    = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,1.0\n"            // k = 1
      << "PSOLID,1,1\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,0.0,1.0,0.0\n"
      << "GRID,4,,0.0,0.0,1.0\n"
      << "CTETRA,1,1,1,2,3,4\n"
      << "SPC,1,1,0,0.0,2,0,1.0\n"
      << "SPC,1,3,0,0.0,4,0,0.0\n"
      << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  EXPECT_NEAR(temp_at(r, 1), 0.0, 1e-9);
  EXPECT_NEAR(temp_at(r, 2), 1.0, 1e-9);
  // q = -k * grad T = -1·(1,0,0) = (-1, 0, 0)
  EXPECT_NEAR(flux_x_of(r, 1), -1.0, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────
// Shared preamble: unit-cube CHEXA8 deck.  Node layout:
//   1(0,0,0) 2(1,0,0) 3(1,1,0) 4(0,1,0) — bottom
//   5(0,0,1) 6(1,0,1) 7(1,1,1) 8(0,1,1) — top
// Face x=0 = nodes 1,4,5,8; face x=1 = nodes 2,3,6,7.
namespace {

std::string cube_deck(double k, std::string_view extra_bulk,
                      std::string_view extra_case = "") {
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << extra_case
      << "  THERMAL = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1," << k << "\n"
      << "PSOLID,1,1\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,1.0,1.0,0.0\n"
      << "GRID,4,,0.0,1.0,0.0\n"
      << "GRID,5,,0.0,0.0,1.0\n"
      << "GRID,6,,1.0,0.0,1.0\n"
      << "GRID,7,,1.0,1.0,1.0\n"
      << "GRID,8,,0.0,1.0,1.0\n"
      << "CHEXA,1,1,1,2,3,4,5,6,+\n"
      << "+,7,8\n"
      << extra_bulk
      << "ENDDATA\n";
  return bdf.str();
}

} // namespace

// ── Convection ────────────────────────────────────────────────────────────

TEST(ThermalSteady, Convection_FixedAmbientTemp) {
  // 1-D bar with a convective end: x=0 face Dirichlet T1=100, x=1 face
  // (nodes 2,3,7,6) convecting to a FIXED ambient T∞=0 with film coefficient
  // h=10 (MAT4.K is selected by PHBDY MID), k=50, L=1.
  // Steady state: T_end = (k·T1 + h·L·T∞)/(k + h·L) = 5000/60 = 83.3̅
  // (exact: the solution T = T1 + βx is in the trilinear FE space).
  // Flux: qx = k·(T1 − T_end)/L = 50000/60 = 833.3̅ in +x.
  const auto bulk = std::string("MAT4,2,10.0\n")
                  + "PHBDY,2,2\n"
                  + "SPC,1,1,0,100.0,4,0,100.0\n"
                  + "SPC,1,5,0,100.0,8,0,100.0\n"
                  + "CHBDY,101,2,AREA4,2,3,7,6\n";
  const auto r = run_thermal(cube_deck(50.0, bulk));
  const double t_end = 5000.0 / 60.0;
  for (int nid : {2, 3, 6, 7})
    EXPECT_NEAR(temp_at(r, nid), t_end, 1e-6) << "end node " << nid;
  EXPECT_NEAR(flux_x_of(r, 1), 50000.0 / 60.0, 1e-3);
}

TEST(ThermalSteady, Convection_AmbientNodePrescribed) {
  // Same 1-D setup, but convection couples to an ambient node in all four
  // corresponding GA fields whose temperature is prescribed at 50 via SPC.
  // This exercises the surface↔fluid coupling and Dirichlet condensation.
  // T_end = (k·T1 + h·L·T_f)/(k + h·L) = (50·100 + 10·50)/60 = 91.6̅
  const auto bulk = std::string("MAT4,2,10.0\n")
                  + "PHBDY,2,2\n"
                  + "GRID,9,,5.0,5.0,5.0\n"  // ambient fluid node
                  + "SPC,1,1,0,100.0,4,0,100.0\n"
                  + "SPC,1,5,0,100.0,8,0,100.0\n"
                  + "SPC,1,9,0,50.0\n"
                  + "CHBDY,101,2,AREA4,2,3,7,6,,+A\n"
                  + "+A,9,9,9,9\n";
  const auto r = run_thermal(cube_deck(50.0, bulk));
  const double t_end = 5500.0 / 60.0;
  for (int nid : {2, 3, 6, 7})
    EXPECT_NEAR(temp_at(r, nid), t_end, 1e-6) << "end node " << nid;
  EXPECT_NEAR(temp_at(r, 9), 50.0, 1e-9);
}

TEST(ThermalSteady, Convection_AmbientNodeNoSinkIsUniform) {
  // Ambient node free (no sink anywhere except the Dirichlet face).  Energy
  // conservation: with no heat leaving the system, steady state is a uniform
  // field: every node — surface AND fluid — must settle at T1 = 100.
  const auto bulk = std::string("MAT4,2,10.0\n")
                  + "PHBDY,2,2\n"
                  + "GRID,9,,5.0,5.0,5.0\n"
                  + "SPC,1,1,0,100.0,4,0,100.0\n"
                  + "SPC,1,5,0,100.0,8,0,100.0\n"
                  + "CHBDY,101,2,AREA4,2,3,7,6,,+A\n"
                  + "+A,9,9,9,9\n";
  const auto r = run_thermal(cube_deck(50.0, bulk));
  for (int nid : {2, 3, 6, 7, 9})
    EXPECT_NEAR(temp_at(r, nid), 100.0, 1e-6) << "node " << nid;
}

// ── Applied-flux loads ─────────────────────────────────────────────────────

TEST(ThermalSteady, Qbdy1_AppliedFlux) {
  // x=0 face Dirichlet at 0; QBDY1 applies q0=500 W/m² INTO the x=1 face
  // (referenced by CHBDY EID).  1-D steady: T(x) = q·x/k → T_end = 10,
  // flux qx = −q (heat travels in −x).
  const auto bulk = std::string("SPC,1,1,0,0.0,4,0,0.0\n")
                  + "SPC,1,5,0,0.0,8,0,0.0\n"
                  + "CHBDY,101,,AREA4,2,3,7,6\n"
                  + "QBDY1,7,500.0,101\n";
  const auto r = run_thermal(cube_deck(50.0, bulk, "  LOAD = 7\n"));
  for (int nid : {2, 3, 6, 7})
    EXPECT_NEAR(temp_at(r, nid), 10.0, 1e-6) << "end node " << nid;
  EXPECT_NEAR(flux_x_of(r, 1), -500.0, 1e-3);
}

TEST(ThermalSteady, Qbdy2_UniformPerNodeFlux) {
  // QBDY2 supplies per-corner fluxes Q1..Q4.  Uniform values equal the QBDY1
  // result: each node receives q·A/4 → T_end = q·L/k = 10.
  const auto bulk = std::string("SPC,1,1,0,0.0,4,0,0.0\n")
                  + "SPC,1,5,0,0.0,8,0,0.0\n"
                  + "CHBDY,101,,AREA4,2,3,7,6\n"
                  + "QBDY2,7,101,500.0,500.0,500.0,500.0\n";
  const auto r = run_thermal(cube_deck(50.0, bulk, "  LOAD = 7\n"));
  for (int nid : {2, 3, 6, 7})
    EXPECT_NEAR(temp_at(r, nid), 10.0, 1e-6) << "end node " << nid;
}

TEST(ThermalSteady, Qhbdy_Area4UsesGeometricArea) {
  // AREA4 defines its area geometrically, so its supplied AF=7 must be ignored.
  // q0=500 on the x=1 face gives T_end = q0·L/k = 500/50 = 10.
  const auto bulk = std::string("SPC,1,1,0,0.0,4,0,0.0\n")
                  + "SPC,1,5,0,0.0,8,0,0.0\n"
                  + "QHBDY,7,AREA4,500.0,7.0,2,3,7,6\n";
  const auto r = run_thermal(cube_deck(50.0, bulk, "  LOAD = 7\n"));
  for (int nid : {2, 3, 6, 7})
    EXPECT_NEAR(temp_at(r, nid), 10.0, 1e-6) << "end node " << nid;
}

TEST(ThermalSteady, QhbdyRejectsUnsupportedGeometry) {
  const std::string bdf =
      "BEGIN BULK\n"
      "QHBDY,7,ELCYL,500.0,2.0,2,3\n"
      "ENDDATA\n";
  EXPECT_THROW((void)BdfParser::parse_string(bdf), ParseError);
}

// ── CPENTA6 ────────────────────────────────────────────────────────────────

TEST(ThermalSteady, CPenta6_LinearFieldReproduction) {
  // Unit right-triangle prism (wedge): bottom (0,0,0),(1,0,0),(0,1,0);
  // top (0,0,1),(1,0,1),(0,1,1).  Prescribe T = x at all six nodes (a linear
  // field is exactly representable by P1 shape functions) and check the
  // centroidal flux = −k·(1,0,0) with k = 1.
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << "  FLUX = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,1.0\n"
      << "PSOLID,1,1\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,0.0,1.0,0.0\n"
      << "GRID,4,,0.0,0.0,1.0\n"
      << "GRID,5,,1.0,0.0,1.0\n"
      << "GRID,6,,0.0,1.0,1.0\n"
      << "CPENTA,1,1,1,2,3,4,5,6\n"
      << "SPC,1,1,0,0.0,2,0,1.0\n"
      << "SPC,1,3,0,0.0,4,0,0.0\n"
      << "SPC,1,5,0,1.0,6,0,0.0\n"
      << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  EXPECT_NEAR(flux_x_of(r, 1), -1.0, 1e-9);
}

TEST(ThermalSteady, CPenta6_StackedBar1DConduction) {
  // Two wedges stacked in z, k = 1.  Bottom triangle held at 100, top at 0;
  // all side faces are insulated (natural BC).  T(z) = 100·(1 − z) satisfies
  // Laplace's equation, the Dirichlet ends, AND the zero-flux side condition,
  // so it is the exact solution of the mixed problem — Galerkin reproduces it
  // exactly and the three mid-plane nodes solve to 50.  Recovered flux:
  // q = −k·∇T = (0, 0, +100).
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "  SPC = 1\n"
      << "  FLUX = ALL\n"
      << "BEGIN BULK\n"
      << "MAT4,1,1.0\n"
      << "PSOLID,1,1\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,0.0,1.0,0.0\n"
      << "GRID,4,,0.0,0.0,0.5\n"
      << "GRID,5,,1.0,0.0,0.5\n"
      << "GRID,6,,0.0,1.0,0.5\n"
      << "GRID,7,,0.0,0.0,1.0\n"
      << "GRID,8,,1.0,0.0,1.0\n"
      << "GRID,9,,0.0,1.0,1.0\n"
      << "CPENTA,1,1,1,2,3,4,5,6\n"
      << "CPENTA,2,1,4,5,6,7,8,9\n"
      << "SPC,1,1,0,100.0,2,0,100.0\n"
      << "SPC,1,3,0,100.0\n"
      << "SPC,1,7,0,0.0,8,0,0.0\n"
      << "SPC,1,9,0,0.0\n"
      << "ENDDATA\n";
  const auto r = run_thermal(bdf.str());
  for (int nid : {4, 5, 6})
    EXPECT_NEAR(temp_at(r, nid), 50.0, 1e-9) << "mid node " << nid;
  for (int eid : {1, 2}) {
    const auto &ef = std::find_if(
                         r.subcases[0].heat_fluxes.begin(),
                         r.subcases[0].heat_fluxes.end(),
                         [eid](const ElementHeatFlux &h) {
                           return h.eid.value == eid;
                         });
    ASSERT_NE(ef, r.subcases[0].heat_fluxes.end());
    EXPECT_NEAR(ef->q[0], 0.0, 1e-9);
    EXPECT_NEAR(ef->q[1], 0.0, 1e-9);
    EXPECT_NEAR(ef->q[2], 100.0, 1e-9) << "element " << eid;
  }
}

// ── Semantics / edge cases ─────────────────────────────────────────────────

TEST(ThermalSteady, Tempd_IsNotADirichletDefault) {
  // TEMPD provides a *default* temperature, never a boundary condition.  Deck
  // pins only the x=0 face at 0 with SPC and declares both TEMPD,1,50 and an
  // unrelated TEMP value of 75 at node 2. With no heat sources every free node
  // must solve to 0; neither temperature-field card is a Dirichlet condition.
  const auto bulk = std::string("TEMPD,1,50.0\n")
                  + "TEMP,1,2,75.0\n"
                  + "SPC1,1,0,1,4,5,8\n";
  const auto r = run_thermal(cube_deck(50.0, bulk));
  for (int nid : {2, 3, 6, 7})
    EXPECT_NEAR(temp_at(r, nid), 0.0, 1e-9) << "free node " << nid;
}

TEST(ThermalSteady, OrphanGrid_IsConstrainedAndDoesNotPerturbSolve) {
  // GRID 99 belongs to no thermal element.  It must be constrained out of the
  // system (reported T = 0) without making K singular or disturbing the solve:
  // with only the x=0 face pinned at 100 and no sink, the steady state is a
  // uniform 100 field.
  const auto bulk = std::string("GRID,99,,5.0,5.0,5.0\n")
                  + "SPC,1,1,0,100.0,4,0,100.0\n"
                  + "SPC,1,5,0,100.0,8,0,100.0\n";
  const auto r = run_thermal(cube_deck(50.0, bulk));
  EXPECT_NEAR(temp_at(r, 99), 0.0, 1e-12);
  for (int nid : {2, 3, 6, 7})
    EXPECT_NEAR(temp_at(r, nid), 100.0, 1e-6) << "free node " << nid;
}

// ── CHBDY geometry (exercises ChbdyElementImpl::area / outward_normal,
//    which are otherwise unused in the main codebase) ────────────────────────

TEST(ThermalSteady, ChbdyGeometry_Area4AreaNormalAndPointAreaFactor) {
  std::ostringstream bdf;
  bdf << "SOL 153\n"
      << "CEND\n"
      << "SUBCASE 1\n"
      << "BEGIN BULK\n"
      << "MAT4,2,10.0\n"
      << "PHBDY,2,2,1.0,0.25,0.75\n"
      << "PHBDY,3,,2.0\n"
      << "GRID,1,,0.0,0.0,0.0\n"
      << "GRID,2,,1.0,0.0,0.0\n"
      << "GRID,3,,1.0,1.0,0.0\n"
      << "GRID,4,,0.0,1.0,0.0\n"
      << "GRID,5,,0.0,0.0,1.0\n"
      << "GRID,6,,1.0,0.0,1.0\n"
      << "GRID,7,,1.0,1.0,1.0\n"
      << "GRID,8,,0.0,1.0,1.0\n"
      << "CHBDY,101,2,AREA4,1,2,3,4,,+A\n"
      << "+A,5,6,7,8,0.0,0.0,1.0\n"
      << "CHBDY,102,3,POINT,1\n"
      << "ENDDATA\n";
  const Model m = BdfParser::parse_string(bdf.str());
  ASSERT_EQ(m.chbdy_elements.size(), 2U);

  const ChbdyElementImpl quad(m.chbdy_elements[0], m);
  EXPECT_NEAR(quad.area(), 1.0, 1e-12);
  const Vec3 n = quad.outward_normal();
  EXPECT_NEAR(n.x, 0.0, 1e-12);
  EXPECT_NEAR(n.y, 0.0, 1e-12);
  EXPECT_NEAR(n.z, 1.0, 1e-12);
  EXPECT_EQ(m.phbdy_properties.at(PropertyId{2}).mid, MaterialId{2});
  EXPECT_NEAR(m.phbdy_properties.at(PropertyId{2}).absorptivity, 0.75, 1e-12);
  ASSERT_EQ(m.chbdy_elements[0].ambient_nodes.size(), 4U);
  EXPECT_EQ(m.chbdy_elements[0].ambient_nodes[0], NodeId{5});
  EXPECT_EQ(m.chbdy_elements[0].ambient_nodes[3], NodeId{8});
  EXPECT_NEAR(m.chbdy_elements[0].orientation.z, 1.0, 1e-12);
  const Eigen::MatrixXd convection = quad.convection_conductance();
  EXPECT_NEAR(convection(0, 0), 10.0 / 8.0, 1e-12);
  EXPECT_NEAR(convection(0, 1), 10.0 / 24.0, 1e-12);

  // POINT geometry: area comes from the PHBDY area factor AF = 2.
  const ChbdyElementImpl point(m.chbdy_elements[1], m);
  EXPECT_NEAR(point.area(), 2.0, 1e-12);
}

TEST(ThermalSteady, Mat5LineConductivityProjectsOntoElementAxis) {
  const std::string bdf =
      "SOL 153\n"
      "CEND\n"
      "SUBCASE 1\n"
      "  SPC = 1\n"
      "  FLUX = ALL\n"
      "BEGIN BULK\n"
      "MAT5,1,10.0,0.0,0.0,20.0,0.0,30.0\n"
      "PBAR,1,1,1.0\n"
      "GRID,1,,0.0,0.0,0.0\n"
      "GRID,2,,0.0,1.0,0.0\n"
      "CBAR,1,1,1,2,0.0,0.0,1.0\n"
      "SPC,1,1,0,0.0,2,0,1.0\n"
      "ENDDATA\n";
  const auto result = run_thermal(bdf);
  EXPECT_NEAR(flux_x_of(result, 1), -20.0, 1e-12);
}

TEST(ThermalSteady, MissingThermalLoadTargetIsRejected) {
  const std::string bdf =
      "SOL 153\n"
      "CEND\n"
      "SUBCASE 1\n"
      "  LOAD = 7\n"
      "BEGIN BULK\n"
      "QVOL,7,100.0,999\n"
      "ENDDATA\n";
  EXPECT_THROW((void)run_thermal(bdf), SolverError);

  const std::string boundary_bdf =
      "SOL 153\n"
      "CEND\n"
      "SUBCASE 1\n"
      "  LOAD = 8\n"
      "BEGIN BULK\n"
      "QBDY1,8,100.0,998\n"
      "ENDDATA\n";
  EXPECT_THROW((void)run_thermal(boundary_bdf), SolverError);

  const std::string unsupported_volume_bdf =
      "SOL 153\n"
      "CEND\n"
      "SUBCASE 1\n"
      "  LOAD = 9\n"
      "BEGIN BULK\n"
      "GRID,1,,0.0,0.0,0.0\n"
      "CELAS2,1,100.0,1,1\n"
      "QVOL,9,100.0,1\n"
      "ENDDATA\n";
  EXPECT_THROW((void)run_thermal(unsupported_volume_bdf), SolverError);
}

TEST(ThermalSteady, CoupledSubcasesUseLatestPrecedingHeatResult) {
  std::vector<SubCase> subcases(5);
  subcases[0].analysis_type = SubCaseAnalysis::Heat;
  subcases[1].analysis_type = SubCaseAnalysis::Statics;
  subcases[2].analysis_type = SubCaseAnalysis::Heat;
  subcases[3].analysis_type = SubCaseAnalysis::Statics;
  subcases[4].analysis_type = SubCaseAnalysis::Statics;

  ASSERT_EQ(preceding_heat_result_index(subcases, 1), 0U);
  ASSERT_EQ(preceding_heat_result_index(subcases, 3), 1U);
  ASSERT_EQ(preceding_heat_result_index(subcases, 4), 1U);

  std::vector<SubCase> statics_first(1);
  statics_first[0].analysis_type = SubCaseAnalysis::Statics;
  EXPECT_FALSE(preceding_heat_result_index(statics_first, 0).has_value());
}

TEST(ThermalSteady, StandardSpcAndChbdyDeckPassesStrictQualityChecks) {
  const auto bulk = std::string("MAT4,2,10.0\n")
                  + "PHBDY,2,2\n"
                  + "GRID,9,,5.0,5.0,5.0\n"
                  + "SPC,1,1,0,100.0,4,0,100.0\n"
                  + "SPC,1,5,0,100.0,8,0,100.0\n"
                  + "SPC,1,9,0,50.0\n"
                  + "CHBDY,101,2,AREA4,2,3,7,6,,+A\n"
                  + "+A,9,9,9,9\n";
  Model model = BdfParser::parse_string(cube_deck(50.0, bulk));
  const QualityThresholds thresholds = build_thresholds(model);
  EXPECT_NO_THROW(run_quality_checks(model, thresholds));
}
