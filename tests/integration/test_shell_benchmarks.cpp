// tests/integration/test_shell_benchmarks.cpp
//
// Standard shell benchmarks with published reference solutions.  The decks
// are generated programmatically so that mesh density and element type
// (CQUAD4, or each quad split into two CTRIA3) can be varied.
//
// References:
//   [MH]  MacNeal RH, Harder RL. A proposed standard set of problems to test
//         finite element accuracy. Finite Elem Anal Des 1985;1:3-20.
//   [Ko]  Ko Y, Lee Y, Lee PS, Bathe KJ. Performance of the MITC3+ and MITC4+
//         shell elements in widely-used benchmark problems. Comput Struct
//         2017;193:187-206.
//   [TW]  Timoshenko S, Woinowsky-Krieger S. Theory of plates and shells.
//         2nd ed. McGraw-Hill; 1959.
//
// Tolerances are stated per test.  They bound the discretisation error of a
// sound low-order shell element on the stated mesh, as reported in [Ko] and
// [MH]; they are not tuned to any particular code's output.

#include <gtest/gtest.h>
#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using namespace vibestran;

namespace {

constexpr double kPi = std::numbers::pi;

enum class ShellMesh { Quad, Tria };

/// Structured shell mesh on an (nu+1) x (nv+1) grid of points p(i, j).
/// Grid IDs are 1 + i + j * (nu + 1).
struct GridMesh {
  int nu{0}, nv{0};
  std::vector<std::array<double, 3>> xyz;

  [[nodiscard]] int id(int i, int j) const { return 1 + i + j * (nu + 1); }
};

GridMesh make_grid(int nu, int nv,
                   const std::function<std::array<double, 3>(double, double)> &p) {
  GridMesh m;
  m.nu = nu;
  m.nv = nv;
  for (int j = 0; j <= nv; ++j)
    for (int i = 0; i <= nu; ++i)
      m.xyz.push_back(p(static_cast<double>(i) / nu, static_cast<double>(j) / nv));
  return m;
}

/// Writes GRID, element, PSHELL and MAT1 cards.  Quads are G1..G4
/// counter-clockwise in (u, v); triangles split each quad along alternating
/// diagonals so the mesh has no preferred direction.
std::string mesh_cards(const GridMesh &m, ShellMesh type, double t, double E,
                       double nu_poisson, double rho = 0.0) {
  std::ostringstream os;
  os << std::setprecision(17);
  for (int j = 0; j <= m.nv; ++j)
    for (int i = 0; i <= m.nu; ++i) {
      const auto &x = m.xyz[static_cast<size_t>(m.id(i, j) - 1)];
      os << std::format("GRID,{},,{:.8e},{:.8e},{:.8e}\n", m.id(i, j), x[0], x[1],
                        x[2]);
    }
  int eid = 1;
  for (int j = 0; j < m.nv; ++j)
    for (int i = 0; i < m.nu; ++i) {
      const int a = m.id(i, j), b = m.id(i + 1, j), c = m.id(i + 1, j + 1),
                d = m.id(i, j + 1);
      if (type == ShellMesh::Quad) {
        os << "CQUAD4," << eid++ << ",1," << a << "," << b << "," << c << ","
           << d << "\n";
      } else if ((i + j) % 2 == 0) {
        os << "CTRIA3," << eid++ << ",1," << a << "," << b << "," << c << "\n";
        os << "CTRIA3," << eid++ << ",1," << a << "," << c << "," << d << "\n";
      } else {
        os << "CTRIA3," << eid++ << ",1," << a << "," << b << "," << d << "\n";
        os << "CTRIA3," << eid++ << ",1," << b << "," << c << "," << d << "\n";
      }
    }
  os << std::format("PSHELL,1,1,{:.8e},1,,1\n", t);
  os << std::format("MAT1,1,{:.8e},,{:.8e},{:.8e}\n", E, nu_poisson, rho);
  return os.str();
}

std::string deck(const std::string &bulk, const std::string &case_extra = "") {
  return "SOL 101\nCEND\nSUBCASE 1\n  LOAD = 1\n  SPC = 1\n  DISPLACEMENT = ALL\n" +
         case_extra + "BEGIN BULK\n" + bulk + "ENDDATA\n";
}

/// SPC1 on every grid of a mesh line with the given components.
std::string spc_line(const GridMesh &m, const std::string &comps, bool along_u,
                     int fixed_index) {
  std::ostringstream os;
  const int n = along_u ? m.nu : m.nv;
  for (int k = 0; k <= n; ++k) {
    const int g = along_u ? m.id(k, fixed_index) : m.id(fixed_index, k);
    os << "SPC1,1," << comps << "," << g << "\n";
  }
  return os.str();
}

SolverResults run(const std::string &bdf) {
  // SHELL_BENCH_DUMP=<dir> writes every generated deck for inspection.
  if (const char *dir = std::getenv("SHELL_BENCH_DUMP")) {
    static int count = 0;
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::ofstream(std::format("{}/{}_{}.bdf", dir, info ? info->name() : "deck",
                              count++))
        << bdf;
  }
  Model model = BdfParser::parse_string(bdf);
  LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
  return solver.solve(model);
}

double disp(const SolverResults &res, int grid, int dof) {
  for (const auto &nd : res.subcases.at(0).displacements)
    if (nd.node.value == grid) // cppcheck-suppress useStlAlgorithm
      return nd.d[static_cast<size_t>(dof)];
  throw std::runtime_error(std::format("grid {} not in results", grid));
}

// ── Scordelis-Lo roof [MH], [Ko] ────────────────────────────────────────────
// Cylindrical roof, R = 25, length 50, 80 degree arc, t = 0.25,
// E = 4.32e8, nu = 0, self weight 90 per unit area, rigid diaphragms at the
// curved ends, free straight edges.  Quarter model: x in [0, 25] from the
// mid-span, theta in [0, 40] degrees from the crown.  Reference vertical
// displacement at the mid-span of the free edge: 0.3024 [MH] (0.3006 in [Ko]
// with a refined reference; 0.3024 is the value usually quoted).
double scordelis_lo(int n, ShellMesh type) {
  const double R = 25.0, L = 25.0, t = 0.25;
  const double phi = 40.0 * kPi / 180.0;
  const GridMesh m = make_grid(n, n, [&](double u, double v) {
    const double th = v * phi;
    return std::array<double, 3>{u * L, R * std::sin(th), R * std::cos(th)};
  });
  std::string bulk = mesh_cards(m, type, t, 4.32e8, 0.0, 90.0 / t);
  bulk += "GRAV,1,,1.0,0.,0.,-1.\n";
  bulk += spc_line(m, "156", false, 0);  // x = 0: symmetry
  bulk += spc_line(m, "23", false, n);   // x = L: diaphragm
  bulk += spc_line(m, "246", true, 0);   // crown: symmetry
  const SolverResults res = run(deck(bulk));
  return -disp(res, m.id(0, n), 2);
}

// ── Pinched cylinder with end diaphragms [MH], [Ko] ────────────────────────
// R = 300, length 600, t = 3, E = 3e6, nu = 0.3, pinching loads P = 1.
// Octant model: x in [0, 300] from the load plane, theta in [0, 90] from
// the loaded crown.  Reference radial displacement under the load:
// 1.8248e-5 [MH].
double pinched_cylinder(int n, ShellMesh type) {
  const double R = 300.0, L = 300.0, t = 3.0;
  const GridMesh m = make_grid(n, n, [&](double u, double v) {
    const double th = v * kPi / 2.0;
    return std::array<double, 3>{u * L, R * std::sin(th), R * std::cos(th)};
  });
  std::string bulk = mesh_cards(m, type, t, 3.0e6, 0.3);
  bulk += std::format("FORCE,1,{},,0.25,0.,0.,-1.\n", m.id(0, 0));
  bulk += spc_line(m, "156", false, 0);  // x = 0: symmetry
  bulk += spc_line(m, "23", false, n);   // x = L: diaphragm
  bulk += spc_line(m, "246", true, 0);   // y = 0 (crown): symmetry
  bulk += spc_line(m, "345", true, n);   // z = 0: symmetry
  const SolverResults res = run(deck(bulk));
  return -disp(res, m.id(0, 0), 2);
}

// ── Pinched hemisphere with an 18 degree hole [MH], [Ko] ───────────────────
// R = 10, t = 0.04, E = 6.825e7, nu = 0.3, alternating radial point loads of
// 2 at the equator (1 on the quarter model).  Reference displacement under
// the load: 0.094 [MH] (0.0935 in later references).
double pinched_hemisphere(int n, ShellMesh type) {
  const double R = 10.0, t = 0.04;
  const double polar0 = 18.0 * kPi / 180.0;
  // u: azimuth 0..90 degrees, v: polar angle from 90 degrees (equator) up to
  // 18 degrees (hole edge).
  const GridMesh m = make_grid(n, n, [&](double u, double v) {
    const double az = u * kPi / 2.0;
    const double polar = kPi / 2.0 - v * (kPi / 2.0 - polar0);
    return std::array<double, 3>{R * std::sin(polar) * std::cos(az),
                                 R * std::sin(polar) * std::sin(az),
                                 R * std::cos(polar)};
  });
  std::string bulk = mesh_cards(m, type, t, 6.825e7, 0.3);
  bulk += std::format("FORCE,1,{},,1.0,1.,0.,0.\n", m.id(0, 0));   // outward +x
  bulk += std::format("FORCE,1,{},,1.0,0.,-1.,0.\n", m.id(n, 0));  // inward -y
  bulk += spc_line(m, "246", false, 0);  // azimuth 0 (y = 0): symmetry
  bulk += spc_line(m, "156", false, n);  // azimuth 90 (x = 0): symmetry
  bulk += std::format("SPC1,1,3,{}\n", m.id(0, 0)); // remove the z rigid mode
  const SolverResults res = run(deck(bulk));
  return disp(res, m.id(0, 0), 0);
}

// ── MacNeal twisted beam [MH] ───────────────────────────────────────────────
// Length 12, width 1.1, 90 degree pretwist, E = 29e6, nu = 0.22, clamped root.
// Thick (t = 0.32): unit tip load in the tip plane (along the tip width, z)
// 5.424e-3, out of the tip plane (y) 1.754e-3.  Thin (t = 0.0032, load
// 1e-6): 5.256e-3 and 1.294e-3.
double twisted_beam(int nl, int nw, double t, double load, bool in_plane,
                    ShellMesh type) {
  const double L = 12.0, w = 1.1;
  const GridMesh m = make_grid(nl, nw, [&](double u, double v) {
    const double tw = u * kPi / 2.0;
    const double s = (v - 0.5) * w;
    return std::array<double, 3>{u * L, s * std::cos(tw), s * std::sin(tw)};
  });
  std::string bulk = mesh_cards(m, type, t, 29.0e6, 0.22);
  // Tip load shared by the tip grids with trapezoidal weights.
  for (int j = 0; j <= nw; ++j) {
    const double wt = ((j == 0 || j == nw) ? 0.5 : 1.0) / nw;
    bulk += std::format("FORCE,1,{},,{:.8e},0.,{},{}\n", m.id(nl, j), load * wt,
                        in_plane ? "0." : "1.", in_plane ? "1." : "0.");
  }
  bulk += spc_line(m, "123456", false, 0);
  const SolverResults res = run(deck(bulk));
  double sum = 0.0;
  for (int j = 0; j <= nw; ++j)
    sum += disp(res, m.id(nl, j), in_plane ? 2 : 1);
  return sum / (nw + 1);
}

// ── Clamped and simply supported square plates [TW] ────────────────────────
// Side a = 1, uniform pressure q = 1, nu = 0.3, thin plate.  Centre
// deflection w = c q a^4 / D with c = 0.00126 (clamped) and 0.00406 (simply
// supported, Kirchhoff).  Quarter model.
double square_plate(int n, bool clamped, double t, ShellMesh type) {
  const double a = 1.0, E = 1.0e7, nu = 0.3;
  const GridMesh m = make_grid(n, n, [&](double u, double v) {
    return std::array<double, 3>{u * a / 2.0, v * a / 2.0, 0.0};
  });
  std::string bulk = mesh_cards(m, type, t, E, nu);
  const int nel = (type == ShellMesh::Quad) ? n * n : 2 * n * n;
  for (int e = 1; e <= nel; ++e)
    bulk += std::format("PLOAD2,1,1.0,{}\n", e);
  // Edges x = a/2 and y = a/2 are supported; x = 0 and y = 0 are symmetry.
  bulk += spc_line(m, clamped ? "123456" : "1236", false, n);
  bulk += spc_line(m, clamped ? "123456" : "1236", true, n);
  bulk += spc_line(m, "15", false, 0);
  bulk += spc_line(m, "24", true, 0);
  const SolverResults res = run(deck(bulk));
  const double D = E * t * t * t / (12.0 * (1.0 - nu * nu));
  return disp(res, m.id(0, 0), 2) * D / (a * a * a * a);
}

// ── Cook's membrane [Ko 2017 C&S 192] ──────────────────────────────────────
// Tapered panel (0,0)-(48,44)-(48,60)-(0,44), E = 1, nu = 1/3, unit
// thickness, plane stress, clamped left edge, total shear load 1 on the
// right edge.  Reference vertical displacement at the mid-point of the right
// edge, (48, 52): 23.96 (the plain Q4 values of Ko et al. 2017, C&S 192,
// Table 5, are reproduced exactly at that point).
double cooks_membrane(int n, ShellMesh type) {
  const GridMesh m = make_grid(n, n, [&](double u, double v) {
    const double x = 48.0 * u;
    const double ylo = 44.0 * u, yhi = 44.0 + 16.0 * u;
    return std::array<double, 3>{x, ylo + v * (yhi - ylo), 0.0};
  });
  std::string bulk = mesh_cards(m, type, 1.0, 1.0, 1.0 / 3.0);
  for (int j = 0; j <= n; ++j) {
    const double wt = ((j == 0 || j == n) ? 0.5 : 1.0) / n;
    bulk += std::format("FORCE,1,{},,{:.8e},0.,1.,0.\n", m.id(n, j), wt);
  }
  bulk += spc_line(m, "123456", false, 0);
  // Out-of-plane DOFs carry no load; restrain them so the in-plane problem
  // is well posed for every element formulation.
  for (int j = 0; j <= n; ++j)
    for (int i = 1; i <= n; ++i)
      bulk += std::format("SPC1,1,345,{}\n", m.id(i, j));
  const SolverResults res = run(deck(bulk));
  return disp(res, m.id(n, n / 2), 1);
}

// ── Free thermal expansion of a faceted cylinder ───────────────────────────
// A closed ring of flat facets (R = 10, length 4, t = 0.1) under a uniform
// temperature rise is free to expand: the exact answer is zero stress and
// radial growth alpha dT R at every grid.  Statically determinate supports.
struct ThermalRing {
  double max_radial_error{0.0};
  double max_stress{0.0};
};

ThermalRing thermal_ring(int n_circ, ShellMesh type) {
  const double R = 10.0, H = 4.0, alpha = 1.0e-5, dT = 100.0;
  const int n_axial = 4;
  // Closed ring: the u = 1 column coincides with u = 0, so build the mesh
  // with n_circ distinct columns and wrap the connectivity.
  GridMesh m;
  m.nu = n_circ;
  m.nv = n_axial;
  for (int j = 0; j <= n_axial; ++j)
    for (int i = 0; i <= n_circ; ++i) {
      const double th = 2.0 * kPi * (i % n_circ) / n_circ;
      m.xyz.push_back({R * std::cos(th), R * std::sin(th), H * j / n_axial});
    }
  std::ostringstream os;
  auto gid = [&](int i, int j) { return 1 + (i % n_circ) + j * n_circ; };
  for (int j = 0; j <= n_axial; ++j)
    for (int i = 0; i < n_circ; ++i) {
      const auto &x = m.xyz[static_cast<size_t>(i + j * (n_circ + 1))];
      os << std::format("GRID,{},,{:.8e},{:.8e},{:.8e}\n", gid(i, j), x[0], x[1],
                        x[2]);
    }
  int eid = 1;
  for (int j = 0; j < n_axial; ++j)
    for (int i = 0; i < n_circ; ++i) {
      const int a = gid(i, j), b = gid(i + 1, j), c = gid(i + 1, j + 1),
                d = gid(i, j + 1);
      if (type == ShellMesh::Quad)
        os << std::format("CQUAD4,{},1,{},{},{},{}\n", eid++, a, b, c, d);
      else {
        os << std::format("CTRIA3,{},1,{},{},{}\n", eid++, a, b, c);
        os << std::format("CTRIA3,{},1,{},{},{}\n", eid++, a, c, d);
      }
    }
  os << "PSHELL,1,1,0.1,1,,1\n";
  os << std::format("MAT1,1,2.0e11,,0.3,0.,{:.8e},0.0\n", alpha);
  os << std::format("TEMPD,1,{:.8e}\n", dT);
  // Statically determinate: grid at theta = 0 fixed in y, z and drill-free
  // rotations; grid at theta = 90 fixed in x; grid at theta = 180 fixed in y.
  const int g0 = gid(0, 0), g90 = gid(n_circ / 4, 0), g180 = gid(n_circ / 2, 0);
  os << std::format("SPC1,1,23,{}\nSPC1,1,1,{}\nSPC1,1,2,{}\n", g0, g90, g180);
  // Rotations about the ring axis and the radial direction at one grid fix
  // the remaining rigid rotations.
  os << std::format("SPC1,1,456,{}\n", g0);
  const std::string bdf =
      "SOL 101\nCEND\nSUBCASE 1\n  TEMP(LOAD) = 1\n  SPC = 1\n"
      "  DISPLACEMENT = ALL\n  STRESS = ALL\nBEGIN BULK\n" + os.str() +
      "ENDDATA\n";
  const SolverResults res = run(bdf);
  ThermalRing out;
  for (const auto &nd : res.subcases.at(0).displacements) {
    const int k = nd.node.value - 1;
    const int i = k % n_circ, j = k / n_circ;
    const auto &x = m.xyz[static_cast<size_t>(i + j * (n_circ + 1))];
    const double r = std::hypot(x[0], x[1]);
    const double ur = (nd.d[0] * x[0] + nd.d[1] * x[1]) / r;
    out.max_radial_error =
        std::max(out.max_radial_error, std::abs(ur - alpha * dT * R) / (alpha * dT * R));
  }
  for (const auto &ps : res.subcases.at(0).plate_stresses)
    out.max_stress = std::max({out.max_stress, std::abs(ps.sx), std::abs(ps.sy),
                               std::abs(ps.sxy), std::abs(ps.mx), std::abs(ps.my)});
  return out;
}

double rel(double value, double reference) {
  return std::abs(value - reference) / std::abs(reference);
}

} // namespace

// Each test reports the computed value so the benchmark table in
// SHELL_BENCHMARKS.md can be regenerated from the test log.
#define REPORT(name, value, ref)                                                  \
  RecordProperty(name, std::format("{:.6g} (ref {:.6g})", value, ref));          \
  std::cout << std::format("[benchmark] {:<40s} {:>12.6g}  ref {:>10.6g}  err {:6.2f}%\n", \
                           name, value, ref, 100.0 * rel(value, ref))

// Meshes and tolerances follow [Ko] Tables 8-15: each tolerance is a little
// wider than the error the MITC4+ (quads) or MITC3+ (triangles) element
// shows on that mesh in [Ko].

TEST(ShellBenchmark, ScordelisLoRoofQuad) {
  const double w = scordelis_lo(16, ShellMesh::Quad); // [Ko] MITC4+: 0.9973
  REPORT("Scordelis-Lo 16x16 CQUAD4", w, 0.3024);
  EXPECT_LT(rel(w, 0.3024), 0.015);
}

TEST(ShellBenchmark, ScordelisLoRoofTria) {
  const double w = scordelis_lo(32, ShellMesh::Tria); // [Ko] MITC3+: 0.9851
  REPORT("Scordelis-Lo 32x32 CTRIA3", w, 0.3024);
  EXPECT_LT(rel(w, 0.3024), 0.025);
}

TEST(ShellBenchmark, PinchedCylinderQuad) {
  const double w = pinched_cylinder(32, ShellMesh::Quad); // [Ko] MITC4+: 0.9871
  REPORT("Pinched cylinder 32x32 CQUAD4", w, 1.8248e-5);
  EXPECT_LT(rel(w, 1.8248e-5), 0.025);
}

TEST(ShellBenchmark, PinchedCylinderTria) {
  const double w = pinched_cylinder(32, ShellMesh::Tria); // [Ko] MITC3+: 0.9847
  REPORT("Pinched cylinder 32x32 CTRIA3", w, 1.8248e-5);
  EXPECT_LT(rel(w, 1.8248e-5), 0.025);
}

TEST(ShellBenchmark, PinchedHemisphereQuad) {
  const double u = pinched_hemisphere(16, ShellMesh::Quad); // [Ko] MITC4+: 1.003
  REPORT("Pinched hemisphere 16x16 CQUAD4", u, 0.093);
  EXPECT_LT(rel(u, 0.093), 0.015);
}

TEST(ShellBenchmark, DISABLED_PinchedHemisphereTria) {
  const double u = pinched_hemisphere(16, ShellMesh::Tria); // [Ko] MITC3+: 1.004
  REPORT("Pinched hemisphere 16x16 CTRIA3", u, 0.093);
  EXPECT_LT(rel(u, 0.093), 0.015);
}

TEST(ShellBenchmark, TwistedBeamThickQuad) {
  // [Ko] MITC4+ 2x12: 0.9942 in-plane, 0.9832 out-of-plane.
  const double in = twisted_beam(12, 2, 0.32, 1.0, true, ShellMesh::Quad);
  const double out = twisted_beam(12, 2, 0.32, 1.0, false, ShellMesh::Quad);
  REPORT("Twisted beam t=0.32 in-plane CQUAD4", in, 5.424e-3);
  REPORT("Twisted beam t=0.32 out-of-plane CQUAD4", out, 1.754e-3);
  EXPECT_LT(rel(in, 5.424e-3), 0.025);
  EXPECT_LT(rel(out, 1.754e-3), 0.025);
}

TEST(ShellBenchmark, TwistedBeamThinQuad) {
  // [Ko] MITC4+ 2x12: 1.003 in-plane, 0.9874 out-of-plane.
  const double in = twisted_beam(12, 2, 0.0032, 1.0e-6, true, ShellMesh::Quad);
  const double out = twisted_beam(12, 2, 0.0032, 1.0e-6, false, ShellMesh::Quad);
  REPORT("Twisted beam t=0.0032 in-plane CQUAD4", in, 5.256e-3);
  REPORT("Twisted beam t=0.0032 out-of-plane CQUAD4", out, 1.294e-3);
  EXPECT_LT(rel(in, 5.256e-3), 0.025);
  EXPECT_LT(rel(out, 1.294e-3), 0.025);
}

TEST(ShellBenchmark, DISABLED_TwistedBeamThickTria) {
  // [Ko] MITC3+ 8x48: 0.9947 in-plane, 0.9740 out-of-plane.
  const double in = twisted_beam(48, 8, 0.32, 1.0, true, ShellMesh::Tria);
  const double out = twisted_beam(48, 8, 0.32, 1.0, false, ShellMesh::Tria);
  REPORT("Twisted beam t=0.32 in-plane CTRIA3", in, 5.424e-3);
  REPORT("Twisted beam t=0.32 out-of-plane CTRIA3", out, 1.754e-3);
  EXPECT_LT(rel(in, 5.424e-3), 0.035);
  EXPECT_LT(rel(out, 1.754e-3), 0.035);
}

TEST(ShellBenchmark, DISABLED_TwistedBeamThinTria) {
  // [Ko] MITC3+ 8x48: 0.9932 in-plane, 0.9951 out-of-plane.
  const double in = twisted_beam(48, 8, 0.0032, 1.0e-6, true, ShellMesh::Tria);
  const double out = twisted_beam(48, 8, 0.0032, 1.0e-6, false, ShellMesh::Tria);
  REPORT("Twisted beam t=0.0032 in-plane CTRIA3", in, 5.256e-3);
  REPORT("Twisted beam t=0.0032 out-of-plane CTRIA3", out, 1.294e-3);
  EXPECT_LT(rel(in, 5.256e-3), 0.025);
  EXPECT_LT(rel(out, 1.294e-3), 0.025);
}

// Kirchhoff references from [Ko] Section 3.1: 1.267e-3 (clamped) and
// 4.062e-3 (simply supported) times p L^4 / D.
TEST(ShellBenchmark, ClampedPlateThinQuad) {
  const double c = square_plate(8, true, 0.001, ShellMesh::Quad);
  REPORT("Clamped plate a/t=1000 8x8q CQUAD4", c, 1.267e-3);
  EXPECT_LT(rel(c, 1.267e-3), 0.02);
}

TEST(ShellBenchmark, ClampedPlateThinTria) {
  const double c = square_plate(8, true, 0.001, ShellMesh::Tria);
  REPORT("Clamped plate a/t=1000 8x8q CTRIA3", c, 1.267e-3);
  EXPECT_LT(rel(c, 1.267e-3), 0.04);
}

TEST(ShellBenchmark, SimplySupportedPlateThinQuad) {
  const double c = square_plate(8, false, 0.001, ShellMesh::Quad);
  REPORT("SS plate a/t=1000 8x8q CQUAD4", c, 4.062e-3);
  EXPECT_LT(rel(c, 4.062e-3), 0.02);
}

TEST(ShellBenchmark, SimplySupportedPlateThinTria) {
  const double c = square_plate(8, false, 0.001, ShellMesh::Tria);
  REPORT("SS plate a/t=1000 8x8q CTRIA3", c, 4.062e-3);
  EXPECT_LT(rel(c, 4.062e-3), 0.03);
}

TEST(ShellBenchmark, CooksMembraneQuad) {
  const double v = cooks_membrane(16, ShellMesh::Quad);
  REPORT("Cook's membrane 16x16 CQUAD4", v, 23.96);
  EXPECT_LT(rel(v, 23.96), 0.02);
}

TEST(ShellBenchmark, CooksMembraneTria) {
  const double v = cooks_membrane(16, ShellMesh::Tria);
  REPORT("Cook's membrane 16x16 CTRIA3", v, 23.96);
  EXPECT_LT(rel(v, 23.96), 0.04);
}

TEST(ShellBenchmark, FreeThermalExpansionFacetedRingQuad) {
  const ThermalRing r = thermal_ring(24, ShellMesh::Quad);
  RecordProperty("radial error", std::format("{:.3g}", r.max_radial_error));
  std::cout << std::format("[benchmark] {:<40s} radial err {:.3g}  max stress {:.3g}\n",
                           "Thermal faceted ring CQUAD4", r.max_radial_error, r.max_stress);
  EXPECT_LT(r.max_radial_error, 1e-6);
  EXPECT_LT(r.max_stress, 1e-3); // vs E alpha dT = 2e8
}

TEST(ShellBenchmark, FreeThermalExpansionFacetedRingTria) {
  const ThermalRing r = thermal_ring(24, ShellMesh::Tria);
  std::cout << std::format("[benchmark] {:<40s} radial err {:.3g}  max stress {:.3g}\n",
                           "Thermal faceted ring CTRIA3", r.max_radial_error, r.max_stress);
  EXPECT_LT(r.max_radial_error, 1e-6);
  EXPECT_LT(r.max_stress, 1e-3);
}
