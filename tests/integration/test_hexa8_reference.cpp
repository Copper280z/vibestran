// tests/integration/test_hexa8_reference.cpp
// The Q1/E21 CHEXA8 against the CHEXA comparison decks of MYSTRAN's
// hexa_msc_comparison set (tests/integration/data/hexa8_reference, MSC
// Nastran input).  The element is the formulation of MYSTRAN's CHEXA8, which
// ComLab reproduces to every printed digit, so the table checks vibestran
// against MYSTRAN 19's results to print precision; the comments carry the
// published references and Simcenter Nastran's values.  The analytic tests
// below check the exact solutions: the MacNeal-Harder patch tests, free
// thermal expansion, a linear temperature field (zero stress), and rotation
// and node-relabelling invariance.

#include <gtest/gtest.h>
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
#include <map>
#include <string>
#include <vector>

using namespace vibestran;

namespace {

std::filesystem::path deck_path(const std::string &deck) {
  const std::filesystem::path p = std::filesystem::path(__FILE__).parent_path() / "data" /
                                  "hexa8_reference" / (deck + ".bdf");
  if (!std::filesystem::exists(p))
    throw std::runtime_error("hexa8 reference deck not found: " + p.string());
  return p;
}

struct Check {
  const char *deck;
  const char *kind; // sum, diff, mean, radial, along, freq, eig, zero_modes
  int comp;
  std::vector<int> grids;
  std::vector<double> extra; // mode number, or the load direction for 'along'
  double expected;
  double rel_tol;
  double abs_tol;
};

const std::vector<Check> &checks() {
  static const std::vector<Check> table{
#include "hexa8_reference_checks.inc"
  };
  return table;
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

const ModalSolverResults &modal_run(const std::string &deck) {
  static std::map<std::string, ModalSolverResults> cache;
  if (const auto it = cache.find(deck); it != cache.end())
    return it->second;
  Model model = BdfParser::parse_file(deck_path(deck));
  ModalSolver solver(std::make_unique<SpectraEigensolverBackend>());
  return cache.emplace(deck, solver.solve(model)).first->second;
}

std::vector<double> eigenvalues(const ModalSolverResults &res) {
  std::vector<double> ev;
  for (const auto &m : res.subcases.at(0).modes)
    ev.push_back(m.eigenvalue);
  std::sort(ev.begin(), ev.end());
  return ev;
}

double evaluate(const Check &c) {
  const std::string kind = c.kind;
  if (kind == "freq")
    return modal_run(c.deck).subcases.at(0).modes.at(static_cast<size_t>(c.extra[0]) - 1).cycles_per_sec;
  if (kind == "eig")
    return eigenvalues(modal_run(c.deck)).at(static_cast<size_t>(c.extra[0]) - 1);
  if (kind == "zero_modes") {
    const auto ev = eigenvalues(modal_run(c.deck));
    const double big = *std::max_element(ev.begin(), ev.end());
    return static_cast<double>(
        std::count_if(ev.begin(), ev.end(), [big](double v) { return std::abs(v) < 1e-8 * big; }));
  }
  const StaticRun &run = static_run(c.deck);
  const auto u = [&](int g) { return run.disp.at(g); };
  const auto comp = static_cast<size_t>(c.comp);
  if (kind == "diff")
    return u(c.grids[0])[comp] - u(c.grids[1])[comp];
  double s = 0.0;
  for (int g : c.grids) {
    if (kind == "radial") {
      const Vec3 &x = run.model.node(NodeId{g}).position;
      s += u(g)[0] * x.x + u(g)[1] * x.y; // r = 1
    } else if (kind == "along") {
      s += u(g)[0] * c.extra[0] + u(g)[1] * c.extra[1] + u(g)[2] * c.extra[2];
    } else {
      s += u(g)[comp];
    }
  }
  return kind == "sum" ? s : s / static_cast<double>(c.grids.size());
}

/// Every stress output point (centres and corners) of the CHEXA8 elements.
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

TEST(Hexa8Reference, MatchesMystranQ1E21OnTheComparisonDecks) {
  // MacNeal-Harder beams (regular, trapezoidal, parallelogram, curved),
  // shell-like benchmarks (Scordelis-Lo, pinched cylinder and hemisphere,
  // twisted beam, plates), plate and beam frequencies, free-free spectra
  // (exactly six rigid modes, first deformation eigenvalue), the Lame
  // cylinder at nu = 0.3 and 0.4999 (default and ISOP = FULL), 1e3:1 and
  // 1e20:1 elements, and the IN options.
  for (const Check &c : checks()) {
    const double v = evaluate(c);
    const double tol = std::max(c.abs_tol, c.rel_tol * std::abs(c.expected));
    EXPECT_NEAR(v, c.expected, tol) << c.deck << " " << c.kind;
  }
}

namespace {

/// Displacement u = 1e-3 (2x + y + z, x + 2y + z, x + y + 2z) / 2 of the
/// MacNeal-Harder patch; its strains are 1e-3 everywhere (shear too).
Eigen::Vector3d patch_field(const Vec3 &x) {
  return 1e-3 * 0.5 * Eigen::Vector3d(2 * x.x + x.y + x.z, x.x + 2 * x.y + x.z, x.x + x.y + 2 * x.z);
}

void expect_patch_exact(const std::string &deck, double scale, double nu) {
  const StaticRun &run = static_run(deck);
  double du = 0.0;
  for (const auto &[g, d] : run.disp) {
    const Vec3 &p = run.model.node(NodeId{g}).position;
    const Eigen::Vector3d exact = scale * patch_field(Vec3{p.x / scale, p.y / scale, p.z / scale});
    du = std::max(du, (Eigen::Vector3d(d[0], d[1], d[2]) - exact).cwiseAbs().maxCoeff());
  }
  EXPECT_LT(du / (1e-3 * scale), 1e-12) << deck;
  // sigma = lambda tr(eps) + 2 mu eps, eps = 1e-3: 2000 normal and 400 shear
  // for E = 1e6, nu = 0.25.
  const double E = 1e6, lam = E * nu / ((1 + nu) * (1 - 2 * nu)), mu = E / (2 * (1 + nu));
  const double sn = (3 * lam + 2 * mu) * 1e-3, st = mu * 1e-3;
  const auto pts = stress_points(run.results);
  ASSERT_EQ(pts.size(), 7u * 9u) << deck;
  double ds = 0.0;
  for (const auto &s : pts)
    for (int i = 0; i < 6; ++i)
      ds = std::max(ds, std::abs(s[static_cast<size_t>(i)] - (i < 3 ? sn : st)));
  EXPECT_LT(ds / sn, 1e-8) << deck;
}

} // namespace

TEST(Hexa8Reference, MacNealHarderPatchTestsAreExact) {
  // Seven distorted elements, one of them with a concave corner (non-positive
  // corner Jacobian: its stresses come from the extrapolated recovery).
  // Linear displacement and constant stress at all 63 output points, for
  // nu = 0.25 and 0.495, with ISOP = FULL, and in micrometre units.
  expect_patch_exact("patch/CHEXA", 1.0, 0.25);
  expect_patch_exact("patch/CHEXA_inc", 1.0, 0.495);
  expect_patch_exact("robustness/patch_8_FULL", 1.0, 0.25);
  const StaticRun &micro = static_run("robustness/scale_8");
  double du = 0.0;
  for (const auto &[g, d] : micro.disp) {
    const Vec3 &p = micro.model.node(NodeId{g}).position;
    const Eigen::Vector3d exact = 1e-6 * patch_field(Vec3{p.x / 1e-6, p.y / 1e-6, p.z / 1e-6});
    du = std::max(du, (Eigen::Vector3d(d[0], d[1], d[2]) - exact).cwiseAbs().maxCoeff());
  }
  EXPECT_LT(du / 1e-9, 1e-12);
}

TEST(Hexa8Reference, FreeThermalExpansionAndLinearGradientsAreStressFree) {
  // Free expansion of the distorted patch: u = alpha dT x = 1e-3 x, zero
  // stress.  T = 20 + (30, -50, 80) . x on rotated and sheared 2x2x2
  // meshes, statically determinate: the exact thermal strain is a
  // compatible field, so the stress is zero everywhere, which needs the
  // internal-mode thermal load and its recovery (M H^-1 q).
  const StaticRun &th = static_run("robustness/thermal_8");
  double du = 0.0;
  for (const auto &[g, d] : th.disp) {
    const Vec3 &p = th.model.node(NodeId{g}).position;
    du = std::max(du, (Eigen::Vector3d(d[0], d[1], d[2]) - 1e-3 * Eigen::Vector3d(p.x, p.y, p.z))
                          .cwiseAbs()
                          .maxCoeff());
  }
  EXPECT_LT(du / 1e-3, 1e-12);
  for (const char *deck :
       {"robustness/thermal_8", "robustness/gradient_8_rotated", "robustness/gradient_8_sheared"}) {
    const auto pts = stress_points(static_run(deck).results);
    ASSERT_FALSE(pts.empty()) << deck;
    double smax = 0.0;
    for (const auto &s : pts)
      for (double v : s)
        smax = std::max(smax, std::abs(v));
    // E alpha dT is 1000 (thermal_8) to about 2000 (gradients).
    EXPECT_LT(smax, 1e-7) << deck;
  }
}

TEST(Hexa8Reference, DistortedCantileverIsInvariantToRotationAndRelabelling) {
  // The same distorted 4x2x2 cantilever with geometry and loads rotated by R,
  // and with the element connectivity relabelled (a cyclic and a 90-degree
  // permutation of the local axes): R^T u_rotated and u_relabelled equal the
  // reference displacements.
  const Eigen::Matrix3d R = (Eigen::Matrix3d() << -0.602050004254, 0.147298624708, 0.784754042702,
                             -0.457131463371, -0.869407139094, -0.187515470530, 0.654649996238,
                             -0.471629453757, 0.590761407656)
                                .finished();
  const StaticRun &ref = static_run("robustness/inv_8_reference");
  double scale = 0.0;
  for (const auto &[g, d] : ref.disp)
    scale = std::max(scale, Eigen::Vector3d(d[0], d[1], d[2]).cwiseAbs().maxCoeff());
  for (const auto &[deck, rot, tol] :
       {std::tuple{"robustness/inv_8_rotated", true, 1e-6},
        std::tuple{"robustness/inv_8_relabeled1", false, 1e-12},
        std::tuple{"robustness/inv_8_relabeled2", false, 1e-12}}) {
    const StaticRun &run = static_run(deck);
    double err = 0.0;
    for (const auto &[g, d0] : ref.disp) {
      const auto &d = run.disp.at(g);
      Eigen::Vector3d u(d[0], d[1], d[2]);
      if (rot)
        u = R.transpose() * u;
      err = std::max(err, (u - Eigen::Vector3d(d0[0], d0[1], d0[2])).cwiseAbs().maxCoeff());
    }
    // The rotated deck's geometry is printed to 12 digits (MYSTRAN: 1.2e-7).
    EXPECT_LT(err / scale, tol) << deck;
  }
}
