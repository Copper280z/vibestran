// tests/integration/test_hexa20_reference.cpp
// The projected assumed-strain CHEXA20 (solid_hexa20.hpp, CHEXA20_PLAN.md)
// on decks of MYSTRAN's hexa_msc_comparison set
// (tests/integration/data/hexa20_reference, MSC Nastran input).
//
// * Thin-shell benchmarks, one element through the thickness, against the
//   independent numpy prototype of the formulation (tools/hex20_prototype)
//   and against the shell-theory references.  The comments give MYSTRAN's
//   current CHEXA20, Simcenter Nastran 2412 and ComLab.
// * Exact solutions: the curved, off-centre-midside patch test, the Lame
//   cylinder near incompressibility, a rotated 1000:1 element, free thermal
//   expansion and linear temperature fields, and rotation and relabelling
//   invariance of a distorted cantilever.
// * Frequencies: on the flat (affine) plate the element equals MYSTRAN's.

#include <gtest/gtest.h>
#include "elements/hexa20_shape.hpp"
#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/eigensolver_backend.hpp"
#include "solver/linear_static.hpp"
#include "solver/modal.hpp"
#include "solver/solver_backend.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace vibestran;

namespace {

std::filesystem::path deck_path(const std::string &deck) {
  const std::filesystem::path p = std::filesystem::path(__FILE__).parent_path() / "data" /
                                  "hexa20_reference" / (deck + ".bdf");
  if (!std::filesystem::exists(p))
    throw std::runtime_error("hexa20 reference deck not found: " + p.string());
  return p;
}

struct StaticRun {
  Model model;
  SolverResults results;
  std::map<int, std::array<double, 6>> disp;
};

const StaticRun &static_run(const std::string &deck) {
  static std::map<std::string, StaticRun> cache;
  if (const auto it = cache.find(deck); it != cache.end())
    return it->second;
  StaticRun run{BdfParser::parse_file(deck_path(deck)), {}, {}};
  LinearStaticSolver solver(std::make_unique<LdltSolverBackend>());
  run.results = solver.solve(run.model);
  for (const auto &nd : run.results.subcases.at(0).displacements)
    run.disp[nd.node.value] = nd.d;
  return cache.emplace(deck, std::move(run)).first->second;
}

double mean_or_sum(const StaticRun &run, int comp, const std::vector<int> &grids, bool mean) {
  double s = 0.0;
  for (int g : grids)
    s += run.disp.at(g)[static_cast<size_t>(comp)];
  return mean ? s / static_cast<double>(grids.size()) : s;
}

/// Every stress output point (centres and corners) of the solid elements.
std::vector<std::array<double, 6>> stress_points(const SolverResults &res) {
  std::vector<std::array<double, 6>> pts;
  for (const auto &ss : res.subcases.at(0).solid_stresses) {
    pts.push_back({ss.sx, ss.sy, ss.sz, ss.sxy, ss.syz, ss.szx});
    for (const auto &p : ss.nodal)
      pts.push_back({p.sx, p.sy, p.sz, p.sxy, p.syz, p.szx});
  }
  return pts;
}

} // namespace

TEST(Hexa20Reference, ThinShellBenchmarksMatchThePrototypeAndAreFreeOfMembraneLocking) {
  // One element through the thickness.  `proto` is the numpy prototype
  // (asnl2 in CHEXA20_PLAN.md); the shell-theory reference follows.
  // MYSTRAN's current CHEXA20 / Simcenter / ComLab in % of the reference:
  //   hemisphere 8x8 55.1 / 96.6 / 98.7, 12x12 85.2 / 99.0 / 99.5;
  //   cylinder 8x8 88.0 / 93.0 / 96.9, 12x12 96.6 / 98.3 / 99.1;
  //   Scordelis-Lo 99.0 / 99.6 / 99.7; twisted beam 99.3, 99.9 (all);
  //   simply supported plate 100.8 (all).
  const std::vector<int> tw{25, 38, 63, 76, 101, 114, 127, 140, 165, 178, 203, 216, 241};
  struct Case {
    const char *deck;
    int comp;
    std::vector<int> grids;
    bool mean;
    double proto, reference, min_percent;
  };
  const std::vector<Case> cases{
      {"CHEXA20_pinched_hemisphere_n08", 0, {1, 226, 307}, true, 9.202571246521e-02, 0.094, 97.5},
      {"CHEXA20_pinched_hemisphere_n12", 0, {1, 482, 651}, true, 9.335367277389e-02, 0.094, 99.0},
      {"CHEXA20_pinched_cylinder_n08", 2, {1, 226, 307}, true, -1.759511740345e-05, -1.8248e-5, 96.0},
      {"CHEXA20_pinched_cylinder_n12", 2, {1, 482, 651}, true, -1.807862342160e-05, -1.8248e-5, 98.8},
      {"CHEXA20_scordelis_lo", 2, {209, 298, 515}, false, -9.041304000025e-01, -0.9072, 99.5},
      {"CHEXA20_twisted_beam_inplane", 2, tw, false, 7.004407083517e-02, 0.070512, 99.2},
      {"CHEXA20_twisted_beam_outofplane", 1, tw, false, 2.277634810525e-02, 0.022802, 99.8},
      {"CHEXA20_plate_ss", 2, {1, 226, 307}, false, -1.341665835734e-01, -0.133083, 100.7},
  };
  for (const Case &c : cases) {
    const double v = mean_or_sum(static_run(c.deck), c.comp, c.grids, c.mean);
    // The hemisphere has cond(K) eps ~ 1e-7.
    EXPECT_NEAR(v, c.proto, 2e-6 * std::abs(c.proto)) << c.deck;
    EXPECT_GT(100.0 * v / c.reference, c.min_percent) << c.deck;
  }
}

TEST(Hexa20Reference, CurvedPatchTestIsExact) {
  // MacNeal-Harder 7-element patch with curved, off-centre interior midside
  // nodes; u = 1e-3 (2x + y + z, x + 2y + z, x + y + 2z) / 2 imposed on the
  // boundary.  Exact at every grid, and stress 2000 normal, 400 shear at the
  // centre and corners of every element (ComLab misses this by 0.9% and
  // 3.8%, plain 2x2x2 integration likewise).
  const StaticRun &run = static_run("patch_20_default");
  double du = 0.0;
  for (const auto &[g, d] : run.disp) {
    const Vec3 &x = run.model.node(NodeId{g}).position;
    const Eigen::Vector3d exact =
        0.5e-3 * Eigen::Vector3d(2 * x.x + x.y + x.z, x.x + 2 * x.y + x.z, x.x + x.y + 2 * x.z);
    du = std::max(du, (Eigen::Vector3d(d[0], d[1], d[2]) - exact).cwiseAbs().maxCoeff());
  }
  EXPECT_LT(du / 1e-3, 1e-12);
  const auto pts = stress_points(run.results);
  ASSERT_EQ(pts.size(), 7u * 9u);
  double ds = 0.0;
  for (const auto &s : pts)
    for (std::size_t i = 0; i < 6; ++i)
      ds = std::max(ds, std::abs(s[i] - (i < 3 ? 2000.0 : 400.0)));
  EXPECT_LT(ds / 2000.0, 1e-9);
}

TEST(Hexa20Reference, LameCylinderAndExtremeAspectRatio) {
  // Thick-walled cylinder a = 1, b = 2, internal pressure 1, plane strain,
  // distorted 4x6 mesh: exact radial displacement at r = a.  The B-bar keeps
  // nu = 0.4999 free of volumetric locking (Simcenter 93.6%, ComLab 99.8%).
  for (const auto &[deck, exact, tol] :
       {std::tuple{"cyl_20_0.3_default", 1.906667e-6, 2e-3},
        std::tuple{"cyl_20_0.4999_default", 1.999967e-6, 3e-3}}) {
    const StaticRun &run = static_run(deck);
    double s = 0.0;
    int n = 0;
    for (const auto &[g, d] : run.disp) {
      const Vec3 &x = run.model.node(NodeId{g}).position;
      const double r = std::hypot(x.x, x.y);
      if (std::abs(r - 1.0) < 1e-6) {
        s += (d[0] * x.x + d[1] * x.y) / r;
        ++n;
      }
    }
    ASSERT_GT(n, 0) << deck;
    EXPECT_NEAR(s / n, exact, tol * exact) << deck;
  }
  // One 1000 x 1 x 1000 element in an arbitrary orientation, in tension
  // through the thin direction, nu = 0: exact displacement 1e-12 (ComLab is
  // about 1500 times too flexible).
  const StaticRun &run = static_run("aspect_20_default_1e+03");
  Eigen::Vector3d total = Eigen::Vector3d::Zero();
  std::vector<int> loaded;
  for (const auto &load : run.model.loads)
    if (const auto *f = std::get_if<ForceLoad>(&load)) {
      total += Eigen::Vector3d(f->direction.x, f->direction.y, f->direction.z) * f->scale;
      loaded.push_back(f->node.value);
    }
  ASSERT_FALSE(loaded.empty());
  const Eigen::Vector3d dir = total.normalized();
  double s = 0.0;
  for (int g : loaded) {
    const auto &d = run.disp.at(g);
    s += Eigen::Vector3d(d[0], d[1], d[2]).dot(dir);
  }
  EXPECT_NEAR(s / static_cast<double>(loaded.size()), 1e-12, 1e-9 * 1e-12);
}

TEST(Hexa20Reference, FreeThermalExpansionAndLinearGradientsAreStressFree) {
  // Free expansion of the curved patch (u = alpha dT x = 1e-3 x, zero
  // stress), and T = 20 + (30, -50, 80) . x on rotated and sheared affine
  // 2x2x2 meshes, statically determinate: the exact thermal strain is
  // compatible, so the stress is zero at every output point.
  const StaticRun &th = static_run("thermal_20");
  double du = 0.0;
  for (const auto &[g, d] : th.disp) {
    const Vec3 &p = th.model.node(NodeId{g}).position;
    du = std::max(du, (Eigen::Vector3d(d[0], d[1], d[2]) - 1e-3 * Eigen::Vector3d(p.x, p.y, p.z))
                          .cwiseAbs()
                          .maxCoeff());
  }
  EXPECT_LT(du / 1e-3, 1e-12);
  for (const char *deck : {"thermal_20", "gradient_20_rotated", "gradient_20_sheared"}) {
    const auto pts = stress_points(static_run(deck).results);
    ASSERT_FALSE(pts.empty()) << deck;
    double smax = 0.0;
    for (const auto &s : pts)
      for (double v : s)
        smax = std::max(smax, std::abs(v));
    // E alpha dT is 1000 (thermal_20) to about 2000 (gradients).
    EXPECT_LT(smax, 1e-7) << deck;
  }
}

TEST(Hexa20Reference, DistortedCantileverIsInvariantToRotationAndRelabelling) {
  // The distorted 4x2x2 cantilever rotated by R, and with the connectivity
  // relabelled by a cyclic and a 90-degree permutation of the local axes:
  // R^T u_rotated and u_relabelled equal the reference displacements.  The
  // tip deflection 1.29642e-3 lies between Simcenter's 1.28841e-3 and
  // ComLab's 1.3026e-3 (MYSTRAN's current element: 1.29920e-3).
  const Eigen::Matrix3d R = (Eigen::Matrix3d() << -0.602050004254, 0.147298624708, 0.784754042702,
                             -0.457131463371, -0.869407139094, -0.187515470530, 0.654649996238,
                             -0.471629453757, 0.590761407656)
                                .finished();
  const StaticRun &ref = static_run("inv_20_reference");
  double scale = 0.0;
  for (const auto &[g, d] : ref.disp)
    scale = std::max(scale, Eigen::Vector3d(d[0], d[1], d[2]).cwiseAbs().maxCoeff());
  for (const auto &[deck, rot, tol] : {std::tuple{"inv_20_rotated", true, 1e-6},
                                       std::tuple{"inv_20_relabeled1", false, 1e-11},
                                       std::tuple{"inv_20_relabeled2", false, 1e-11}}) {
    const StaticRun &run = static_run(deck);
    double err = 0.0;
    for (const auto &[g, d0] : ref.disp) {
      const auto &d = run.disp.at(g);
      Eigen::Vector3d u(d[0], d[1], d[2]);
      if (rot)
        u = R.transpose() * u;
      err = std::max(err, (u - Eigen::Vector3d(d0[0], d0[1], d0[2])).cwiseAbs().maxCoeff());
    }
    // The rotated deck's geometry is printed to 12 digits.
    EXPECT_LT(err / scale, tol) << deck;
  }
}

TEST(Hexa20Reference, PlateAndBeamFrequencies) {
  // Simply supported square plate (flat, affine elements: the element equals
  // MYSTRAN's CHEXA20, whose frequencies are 47.6603, 119.931, 119.938,
  // 194.733, 242.47 Hz; Kirchhoff 47.99, 120, 120, 191.9, 239.9), and the
  // MacNeal trapezoidal cantilever (non-affine; Euler-Bernoulli 1.41898 Hz,
  // MYSTRAN 1.4541 Hz).  Consistent mass (PARAM,COUPMASS,2).
  Model plate = BdfParser::parse_file(deck_path("CHEXA20_plate_ss_modes"));
  ModalSolver solver(std::make_unique<SpectraEigensolverBackend>());
  const auto res = solver.solve(plate);
  const std::array<double, 5> mystran{47.6603, 119.931, 119.938, 194.733, 242.47};
  const auto &modes = res.subcases.at(0).modes;
  ASSERT_GE(modes.size(), mystran.size());
  for (std::size_t i = 0; i < mystran.size(); ++i)
    EXPECT_NEAR(modes[i].cycles_per_sec, mystran[i], 2e-5 * mystran[i]) << "mode " << i + 1;

  Model beam = BdfParser::parse_file(deck_path("CHEXA20_beam_trapezoidal_modes"));
  ModalSolver beam_solver(std::make_unique<SpectraEigensolverBackend>());
  const auto beam_res = beam_solver.solve(beam);
  const double f1 = beam_res.subcases.at(0).modes.at(0).cycles_per_sec;
  EXPECT_GT(f1, 1.41898);
  EXPECT_LT(f1, 1.4541 * 1.001);
}

TEST(Hexa20Reference, Pload4OnAQuadraticFaceGivesUniformStress) {
  // One 2 x 1 x 3 CHEXA20 block, bottom on rollers (z fixed, x fixed on
  // x = 0, y on y = 0), PLOAD4 p = 100 on the top face selected by its
  // corners G5 and G7.  The consistent load of the 8-node face (corners
  // -pA/12, midsides +pA/3) gives the exact uniform state sigma_zz = -p,
  // u = p/E (nu x, nu y, -z).
  const double E = 1e6, nu = 0.3, p = 100.0;
  std::string bdf = "SOL 101\nCEND\nDISPLACEMENT = ALL\nSTRESS = ALL\nLOAD = 1\nSPC = 1\n"
                    "BEGIN BULK\n";
  for (std::size_t a = 0; a < 20; ++a) {
    const auto &c = kHexa20Nodes[a];
    bdf += std::format("GRID,{},,{},{},{}\n", a + 1, (c[0] + 1.0), 0.5 * (c[1] + 1.0),
                       1.5 * (c[2] + 1.0));
  }
  bdf += "CHEXA,1,1,1,2,3,4,5,6\n,7,8,9,10,11,12,13,14\n,15,16,17,18,19,20\n";
  bdf += std::format("PSOLID,1,1\nMAT1,1,{},,{}\n", E, nu);
  bdf += "PLOAD4,1,1,100.,,,,5,7\n";
  bdf += "SPC1,1,3,1,2,3,4,9,10,11,12\nSPC1,1,1,1,4,12\nSPC1,1,2,1,2,9\nENDDATA\n";
  Model model = BdfParser::parse_string(bdf);
  LinearStaticSolver solver(std::make_unique<LdltSolverBackend>());
  const SolverResults res = solver.solve(model);
  for (const auto &nd : res.subcases.at(0).displacements) {
    const Vec3 &x = model.node(nd.node).position;
    EXPECT_NEAR(nd.d[0], nu * p / E * x.x, 1e-12) << nd.node.value;
    EXPECT_NEAR(nd.d[1], nu * p / E * x.y, 1e-12) << nd.node.value;
    EXPECT_NEAR(nd.d[2], -p / E * x.z, 1e-12) << nd.node.value;
  }
  const auto pts = stress_points(res);
  ASSERT_EQ(pts.size(), 9u);
  for (const auto &st : pts)
    for (std::size_t i = 0; i < 6; ++i)
      EXPECT_NEAR(st[i], i == 2 ? -p : 0.0, 1e-9 * p);
}
