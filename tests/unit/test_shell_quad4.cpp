// tests/unit/test_shell_quad4.cpp
// Basic tests of the MITC4+/D CQUAD4 (Ko, Bathe & Zhang 2025): zero-energy
// modes, rigid-body motion and isotropy on flat, distorted and warped
// geometry, patch tests with free drill rotations, and the drill rotation's
// physical meaning.

#include <gtest/gtest.h>
#include "elements/shell_quad4.hpp"
#include "elements/shell_directors.hpp"
#include "io/bdf_parser.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <sstream>
#include <numbers>

using namespace vibestran;

namespace {

Model quad_model(const std::array<std::array<double, 3>, 4> &xyz, double t = 0.1,
                 double E = 1.0e7, double nu = 0.3) {
  Model m;
  for (int i = 0; i < 4; ++i) {
    GridPoint gp;
    gp.id = NodeId{i + 1};
    gp.position = Vec3{xyz[static_cast<size_t>(i)][0], xyz[static_cast<size_t>(i)][1],
                       xyz[static_cast<size_t>(i)][2]};
    m.nodes[gp.id] = gp;
  }
  Mat1 mat;
  mat.id = MaterialId{1};
  mat.E = E;
  mat.nu = nu;
  mat.G = E / (2.0 * (1.0 + nu));
  mat.rho = 2.0;
  m.materials[mat.id] = mat;
  PShell ps;
  ps.pid = PropertyId{1};
  ps.mid1 = ps.mid2 = ps.mid3 = MaterialId{1};
  ps.t = t;
  m.properties[ps.pid] = ps;
  return m;
}

ElementData quad_data() {
  ElementData d;
  d.id = ElementId{1};
  d.pid = PropertyId{1};
  d.type = ElementType::CQUAD4;
  d.nodes = {NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4}};
  return d;
}

/// Rigid-body displacement vector: translation c plus rotation w about the
/// origin, for the element's grids.
Eigen::VectorXd rigid(const Model &m, const Eigen::Vector3d &c,
                      const Eigen::Vector3d &w) {
  Eigen::VectorXd u(24);
  for (int i = 0; i < 4; ++i) {
    const Vec3 &p = m.node(NodeId{i + 1}).position;
    const Eigen::Vector3d x(p.x, p.y, p.z);
    u.segment<3>(6 * i) = c + w.cross(x);
    u.segment<3>(6 * i + 3) = w;
  }
  return u;
}

int zero_modes(const Eigen::MatrixXd &K) {
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (K + K.transpose()));
  const Eigen::VectorXd ev = es.eigenvalues();
  const double tol = 1e-9 * ev.cwiseAbs().maxCoeff();
  int zeros = 0;
  for (int i = 0; i < ev.size(); ++i) {
    EXPECT_GT(ev(i), -tol) << "negative eigenvalue " << ev(i);
    if (std::abs(ev(i)) < tol)
      ++zeros;
  }
  return zeros;
}

const std::array<std::array<std::array<double, 3>, 4>, 3> kShapes{{
    // Unit square.
    {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}},
    // Distorted flat quad (MacNeal patch element), tilted out of the xy plane.
    {{{0.0, 0.0, 0.0}, {0.04, 0.02, 0.01}, {0.08, 0.08, 0.03}, {0.0, 0.12, 0.02}}},
    // Warped quad: one corner lifted by 20% of the side.
    {{{0, 0, 0}, {1, 0, 0.0}, {1.1, 0.9, 0.2}, {0, 1, 0}}},
}};

} // namespace

TEST(ShellQuad4, HasExactlySixZeroEnergyModes) {
  for (const auto &xyz : kShapes) {
    const Model m = quad_model(xyz);
    const ShellQuad4 e(quad_data(), m);
    EXPECT_EQ(zero_modes(e.stiffness_matrix()), 6);
  }
}

TEST(ShellQuad4, RigidBodyMotionsProduceNoForce) {
  for (const auto &xyz : kShapes) {
    const Model m = quad_model(xyz);
    const ShellQuad4 e(quad_data(), m);
    const Eigen::MatrixXd K = e.stiffness_matrix();
    const double scale = K.cwiseAbs().maxCoeff();
    for (int k = 0; k < 6; ++k) {
      Eigen::Vector3d c = Eigen::Vector3d::Zero(), w = Eigen::Vector3d::Zero();
      if (k < 3)
        c(k) = 1.0;
      else
        w(k - 3) = 1.0;
      const Eigen::VectorXd f = K * rigid(m, c, w);
      EXPECT_LT(f.cwiseAbs().maxCoeff(), 1e-9 * scale) << "mode " << k;
    }
  }
}

TEST(ShellQuad4, StiffnessIsInvariantUnderRotationOfTheElement) {
  // Spatial isotropy: rotating the element in space rotates its stiffness.
  const auto &xyz = kShapes[2];
  const Model m = quad_model(xyz);
  const Eigen::MatrixXd K = ShellQuad4(quad_data(), m).stiffness_matrix();
  const Eigen::Matrix3d R =
      (Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized())).toRotationMatrix();
  std::array<std::array<double, 3>, 4> rot{};
  for (int i = 0; i < 4; ++i) {
    const Eigen::Vector3d p =
        R * Eigen::Vector3d(xyz[static_cast<size_t>(i)][0], xyz[static_cast<size_t>(i)][1],
                            xyz[static_cast<size_t>(i)][2]);
    rot[static_cast<size_t>(i)] = {p.x(), p.y(), p.z()};
  }
  const Model mr = quad_model(rot);
  const Eigen::MatrixXd Kr = ShellQuad4(quad_data(), mr).stiffness_matrix();
  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(24, 24);
  for (int b = 0; b < 8; ++b)
    T.block<3, 3>(3 * b, 3 * b) = R;
  EXPECT_LT((T * K * T.transpose() - Kr).cwiseAbs().maxCoeff(),
            1e-9 * K.cwiseAbs().maxCoeff());
}

namespace {

// MacNeal / Ko et al. distorted patch: square 0.24 x 0.12 ... here the
// 10 x 10 square of [KBZ] Fig. 7 with interior grids (2,2), (8,3), (8,7),
// (4,7).  Boundary grids 1-4 at the corners, interior grids 5-8.
const std::array<std::array<double, 2>, 8> kPatchXY{{
    {0, 0}, {10, 0}, {10, 10}, {0, 10}, {2, 2}, {8, 3}, {8, 7}, {4, 7}}};
const std::array<std::array<int, 4>, 5> kPatchQuads{{
    {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 4, 8, 7}, {4, 1, 5, 8}, {5, 6, 7, 8}}};

/// Runs the patch with the exact field prescribed on the four boundary
/// grids (the components listed in `comps`, values from `exact`) and
/// returns the solution.
SolverResults run_patch(const std::function<std::array<double, 6>(double, double)> &exact,
                        const std::string &comps, double t) {
  std::ostringstream os;
  os << "SOL 101\nCEND\nSUBCASE 1\n  SPC = 1\n  LOAD = 1\n  DISPLACEMENT = ALL\n"
        "  STRESS = ALL\n  FORCE = ALL\nBEGIN BULK\n";
  for (int i = 0; i < 8; ++i)
    os << std::format("GRID,{},,{},{},0.\n", i + 1, kPatchXY[static_cast<size_t>(i)][0],
                      kPatchXY[static_cast<size_t>(i)][1]);
  for (int e = 0; e < 5; ++e) {
    const auto &q = kPatchQuads[static_cast<size_t>(e)];
    os << std::format("CQUAD4,{},1,{},{},{},{}\n", e + 1, q[0], q[1], q[2], q[3]);
  }
  os << std::format("PSHELL,1,1,{},1,,1\nMAT1,1,1.0E6,,0.25\n", t);
  for (int i = 0; i < 4; ++i) {
    const auto v = exact(kPatchXY[static_cast<size_t>(i)][0], kPatchXY[static_cast<size_t>(i)][1]);
    for (char c : comps) {
      const int k = c - '1';
      os << std::format("SPC,1,{},{},{:.10e}\n", i + 1, c, v[static_cast<size_t>(k)]);
    }
  }
  os << "FORCE,1,1,,0.,1.,0.,0.\nENDDATA\n";
  Model model = BdfParser::parse_string(os.str());
  LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
  return solver.solve(model);
}

} // namespace

TEST(ShellQuad4, MembranePatchTestWithFreeDrillRotations) {
  // u = 1e-3 (x + y/2), v = 1e-3 (y + x/2): constant strain 1e-3, 1e-3,
  // 1e-3 and zero in-plane rotation.  Out-of-plane DOFs and the drill
  // rotations are free at the interior grids; boundary drills are free too
  // except at grid 1 (which removes nothing physical: the exact drill is 0).
  auto exact = [](double x, double y) {
    return std::array<double, 6>{1e-3 * (x + y / 2), 1e-3 * (y + x / 2), 0, 0, 0, 0};
  };
  const SolverResults res = run_patch(exact, "12345", 0.001);
  for (const auto &nd : res.subcases.at(0).displacements) {
    const auto &p = kPatchXY[static_cast<size_t>(nd.node.value - 1)];
    const auto v = exact(p[0], p[1]);
    EXPECT_NEAR(nd.d[0], v[0], 1e-12) << "grid " << nd.node.value;
    EXPECT_NEAR(nd.d[1], v[1], 1e-12) << "grid " << nd.node.value;
    EXPECT_NEAR(nd.d[5], 0.0, 1e-12) << "drill at grid " << nd.node.value;
  }
  // Strains are reported in each element's own frame, so compare the
  // in-plane invariants: ex + ey = 2e-3 and (ex - ey)^2 + gxy^2 = 1e-6.
  auto check = [](const ShellStrain &e) {
    EXPECT_NEAR(e.ex + e.ey, 2e-3, 1e-12);
    EXPECT_NEAR((e.ex - e.ey) * (e.ex - e.ey) + e.exy * e.exy, 1e-6, 1e-15);
    EXPECT_NEAR(e.kx, 0.0, 1e-12);
  };
  for (const auto &ps : res.subcases.at(0).plate_stresses) {
    check(ps.strain);
    for (const auto &pt : ps.nodal)
      check(pt.strain);
  }
}

TEST(ShellQuad4, BendingPatchTestGivesConstantCurvature) {
  // w = 1e-3 (x^2 + x y + y^2) / 2 with slopes theta_x = dw/dy,
  // theta_y = -dw/dx prescribed at the patch corners: the curvature
  // (Nastran sign, kappa = [w_xx, w_yy, 2 w_xy]) is 1e-3 everywhere.
  auto exact = [](double x, double y) {
    const double w = 0.5e-3 * (x * x + x * y + y * y);
    return std::array<double, 6>{0, 0, w, 0.5e-3 * (x + 2 * y), -0.5e-3 * (2 * x + y), 0};
  };
  const SolverResults res = run_patch(exact, "12345", 0.001);
  for (const auto &nd : res.subcases.at(0).displacements) {
    const auto &p = kPatchXY[static_cast<size_t>(nd.node.value - 1)];
    const auto v = exact(p[0], p[1]);
    EXPECT_NEAR(nd.d[2], v[2], 1e-9) << "grid " << nd.node.value;
    EXPECT_NEAR(nd.d[3], v[3], 1e-9) << "grid " << nd.node.value;
    EXPECT_NEAR(nd.d[4], v[4], 1e-9) << "grid " << nd.node.value;
  }
  auto check = [](const ShellStrain &e) {
    EXPECT_NEAR(e.kx + e.ky, 2e-3, 1e-9);
    EXPECT_NEAR((e.kx - e.ky) * (e.kx - e.ky) + e.kxy * e.kxy, 1e-6, 1e-12);
  };
  for (const auto &ps : res.subcases.at(0).plate_stresses) {
    check(ps.strain);
    for (const auto &pt : ps.nodal)
      check(pt.strain);
  }
}

namespace {

SolverResults run_strip(const std::string &extra, double zoffs = 0.0) {
  // Cantilever strip, length 10, width 1, t = 0.1, 20 x 2 CQUAD4, clamped
  // at x = 0; grids 1..21 at y = 0, 22..42 at y = 0.5, 43..63 at y = 1.
  std::ostringstream os;
  os << "SOL 101\nCEND\nSUBCASE 1\n  SPC = 1\n  LOAD = 1\n  DISPLACEMENT = ALL\n"
        "BEGIN BULK\n";
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i <= 20; ++i)
      os << std::format("GRID,{},,{},{},0.\n", 1 + i + 21 * j, 0.5 * i, 0.5 * j);
  int e = 1;
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 20; ++i) {
      const int a = 1 + i + 21 * j;
      os << std::format("CQUAD4,{},1,{},{},{},{},,{}\n", e++, a, a + 1, a + 22, a + 21,
                        zoffs);
    }
  os << "PSHELL,1,1,0.1,1,,1\nMAT1,1,1.0E7,,0.0\n";
  for (int j = 0; j < 3; ++j)
    os << std::format("SPC1,1,123456,{}\n", 1 + 21 * j);
  os << extra << "ENDDATA\n";
  Model model = BdfParser::parse_string(os.str());
  LinearStaticSolver solver(std::make_unique<EigenSolverBackend>());
  return solver.solve(model);
}

double strip_disp(const SolverResults &res, int grid, int dof) {
  for (const auto &nd : res.subcases.at(0).displacements)
    if (nd.node.value == grid) // cppcheck-suppress useStlAlgorithm
      return nd.d[static_cast<size_t>(dof)];
  return std::nan("");
}

} // namespace

TEST(ShellQuad4, InPlaneBendingIsUnaffectedByTheDrillRotations) {
  // Slender strip under a tip shear force: v(L) = F L^3 / (3 E I) plus a
  // small shear deflection.  The drill DOFs are free; neither the drill
  // membrane strain nor the stabilisation may stiffen the response (a
  // stabilisation of 1e-3 of the shear stiffness did, by 3-11%).
  const double F = 1.0, L = 10.0, E = 1.0e7, I = 0.1 / 12.0;
  const SolverResults res =
      run_strip("FORCE,1,21,,0.25,0.,1.,0.\nFORCE,1,42,,0.5,0.,1.,0.\n"
                "FORCE,1,63,,0.25,0.,1.,0.\n");
  const double v = F * L * L * L / (3.0 * E * I);
  // Timoshenko shear term F L / (k G A), k = 5/6, G = E/2, A = 0.1.
  const double vs = F * L / (5.0 / 6.0 * 0.5 * E * 0.1);
  EXPECT_NEAR(strip_disp(res, 42, 1), v + vs, 0.01 * v);
}

TEST(ShellQuad4, DrillRotationsAreStabilisedButCarryLittleMoment) {
  // Nastran semantics: CQUAD4 has no physical drilling stiffness (Nastran
  // adds an artificial K6ROT spring).  MITC4+/D's drill-membrane strain is
  // energetically decoupled from the translations on rectangular elements
  // (its in-plane shear part integrates to zero against the constant
  // assumed shear), so a drill moment is resisted mainly by the small
  // stabilisation: the strip rotates freely about z instead of bending.
  // This test pins that behaviour so a change is a deliberate decision.
  const double M = 1.0, L = 10.0, E = 1.0e7, I = 0.1 / 12.0;
  const SolverResults res =
      run_strip("MOMENT,1,21,,0.25,0.,0.,1.\nMOMENT,1,42,,0.5,0.,0.,1.\n"
                "MOMENT,1,63,,0.25,0.,0.,1.\n");
  const double beam_v = M * L * L / (2 * E * I);
  EXPECT_TRUE(std::isfinite(strip_disp(res, 42, 5)));
  EXPECT_GT(strip_disp(res, 42, 5), 0.0); // rotates with the moment
  EXPECT_LT(std::abs(strip_disp(res, 42, 1)), 0.01 * beam_v);
}

TEST(ShellQuad4, OffsetReferenceSurfaceCouplesAxialLoadAndBending) {
  // ZOFFS moves the shell reference surface a distance zo above the grids.
  // An axial tip load F applied at the grids then acts zo below the
  // mid-surface: the bottom fibres are pulled, so the strip curls upwards.
  // About +y the eccentric load is a moment -F zo, giving
  // theta_y(L) = -F zo L / (E I), w(L) = F zo L^2 / (2 E I), and the grid
  // (a rigid offset below the section) moves axially by the mid-surface
  // stretch F L / (E A) minus zo theta_y.
  const double F = 1.0, L = 10.0, E = 1.0e7, t = 0.1, zo = 0.5;
  const double A = 1.0 * t, Ib = 1.0 * t * t * t / 12.0;
  const std::string load = "FORCE,1,21,,0.25,1.,0.,0.\nFORCE,1,42,,0.5,1.,0.,0.\n"
                           "FORCE,1,63,,0.25,1.,0.,0.\n";
  const SolverResults flat = run_strip(load, 0.0);
  const SolverResults off = run_strip(load, zo);
  EXPECT_NEAR(strip_disp(flat, 42, 0), F * L / (E * A), 1e-6 * F * L / (E * A));
  EXPECT_NEAR(strip_disp(flat, 42, 2), 0.0, 1e-12);
  const double w = F * zo * L * L / (2.0 * E * Ib);
  EXPECT_NEAR(strip_disp(off, 42, 2), w, 0.02 * w);
  const double theta = -F * zo * L / (E * Ib);
  EXPECT_NEAR(strip_disp(off, 42, 4), theta, 0.02 * std::abs(theta));
  const double u = F * L / (E * A) - zo * theta;
  EXPECT_NEAR(strip_disp(off, 42, 0), u, 0.02 * u);
}

namespace {

Model mat8_quad(const Property &prop) {
  Model m = quad_model({{{0, 0, 0}, {2, 0, 0}, {2.3, 1.5, 0}, {0.2, 1.2, 0}}});
  Mat8 mat;
  mat.id = MaterialId{8};
  mat.e1 = 140e3;
  mat.e2 = 10e3;
  mat.nu12 = 0.3;
  mat.g12 = 5e3;
  mat.g1z = 4e3;
  mat.g2z = 3e3;
  mat.rho = 1.6e-3;
  mat.a1 = -1e-6;
  mat.a2 = 3e-5;
  m.mat8_materials[mat.id] = mat;
  m.properties[PropertyId{1}] = prop;
  return m;
}

} // namespace

TEST(ShellQuad4, SinglePlyPcompMatchesTheEquivalentPshell) {
  PShell ps;
  ps.pid = PropertyId{1};
  ps.mid1 = ps.mid2 = ps.mid3 = MaterialId{8};
  ps.t = 0.2;
  ps.tst = 5.0 / 6.0;
  PComp pc;
  pc.pid = PropertyId{1};
  pc.plies = {{MaterialId{8}, 0.2, 0.0, false}};
  const Eigen::MatrixXd Ks = ShellQuad4(quad_data(), mat8_quad(ps)).stiffness_matrix();
  const Eigen::MatrixXd Kc = ShellQuad4(quad_data(), mat8_quad(pc)).stiffness_matrix();
  EXPECT_LT((Ks - Kc).cwiseAbs().maxCoeff(), 1e-9 * Ks.cwiseAbs().maxCoeff());
  const Eigen::MatrixXd Ms = ShellQuad4(quad_data(), mat8_quad(ps)).mass_matrix();
  const Eigen::MatrixXd Mc = ShellQuad4(quad_data(), mat8_quad(pc)).mass_matrix();
  EXPECT_LT((Ms - Mc).cwiseAbs().maxCoeff(), 1e-12 * Ms.cwiseAbs().maxCoeff());
}

TEST(ShellQuad4, MaterialAngleRotatesOrthotropicAxes) {
  // A ply at 90 degrees with THETA = 0 is the same laminate as a ply at 0
  // degrees with the element material axis at THETA = 90.
  PComp p90;
  p90.pid = PropertyId{1};
  p90.plies = {{MaterialId{8}, 0.2, 90.0, false}};
  PComp p0 = p90;
  p0.plies[0].theta = 0.0;
  ElementData d90 = quad_data(), d0 = quad_data();
  d0.theta = 90.0;
  const Eigen::MatrixXd Ka = ShellQuad4(d90, mat8_quad(p90)).stiffness_matrix();
  const Eigen::MatrixXd Kb = ShellQuad4(d0, mat8_quad(p0)).stiffness_matrix();
  EXPECT_LT((Ka - Kb).cwiseAbs().maxCoeff(), 1e-9 * Ka.cwiseAbs().maxCoeff());
  // ...and differs from the unrotated laminate.
  const Eigen::MatrixXd Kc = ShellQuad4(quad_data(), mat8_quad(p0)).stiffness_matrix();
  EXPECT_GT((Ka - Kc).cwiseAbs().maxCoeff(), 1e-3 * Ka.cwiseAbs().maxCoeff());
}

TEST(ShellQuad4, MassMatrixCarriesTheTranslationalMassAndNsm) {
  const Model m0 = quad_model({{{0, 0, 0}, {2, 0, 0}, {2.3, 1.5, 0}, {0.2, 1.2, 0}}});
  Model m = m0;
  std::get<PShell>(m.properties.at(PropertyId{1})).nsm = 0.05;
  const Eigen::MatrixXd M = ShellQuad4(quad_data(), m).mass_matrix();
  // Area of the quadrilateral by the shoelace formula.
  const double area = 0.5 * std::abs((2 * 1.5 - 2.3 * 0) + (2.3 * 1.2 - 0.2 * 1.5) +
                                     (0.2 * 0 - 0 * 1.2) + (0 * 0 - 2 * 0));
  for (int d = 0; d < 3; ++d) {
    double total = 0.0;
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b)
        total += M(6 * a + d, 6 * b + d);
    EXPECT_NEAR(total, (2.0 * 0.1 + 0.05) * area, 1e-12) << "direction " << d;
  }
  EXPECT_LT((M - M.transpose()).cwiseAbs().maxCoeff(), 1e-15);
}

TEST(ShellQuad4, UnsymmetricLaminateFreeThermalCurvatureMatchesLamination) {
  // A free [0/90] plate under a uniform temperature rise: the generalised
  // strains are [eps0; kappa] = C^-1 [N_T; M_T] dT with C = [A B; B D]
  // (classical lamination theory, Nastran curvature sign).  One element,
  // statically determinate supports.
  PComp pc;
  pc.pid = PropertyId{1};
  pc.plies = {{MaterialId{8}, 0.1, 0.0, false}, {MaterialId{8}, 0.1, 90.0, false}};
  Model m = mat8_quad(pc);
  for (auto &[id, gp] : m.nodes)
    gp.position = Vec3{(id.value == 2 || id.value == 3) ? 1.0 : 0.0,
                       (id.value == 3 || id.value == 4) ? 1.0 : 0.0, 0.0};
  const ShellQuad4 e(quad_data(), m);
  const Eigen::MatrixXd K = e.stiffness_matrix();
  const double dT = 50.0;
  const std::array<double, 4> temps{dT, dT, dT, dT};
  const Eigen::VectorXd f = e.thermal_load(temps, 0.0);
  // Remove rigid motion only (3-2-1): grid 1 translations, grid 2 y and z,
  // grid 4 z.
  std::vector<int> free;
  for (int i = 0; i < 24; ++i) {
    const int node = i / 6, dof = i % 6;
    const bool fixed = (node == 0 && dof < 3) || (node == 1 && (dof == 1 || dof == 2)) ||
                       (node == 3 && dof == 2);
    if (!fixed)
      free.push_back(i);
  }
  Eigen::MatrixXd Kf(free.size(), free.size());
  Eigen::VectorXd ff(free.size());
  for (size_t i = 0; i < free.size(); ++i) {
    ff(static_cast<Eigen::Index>(i)) = f(free[i]);
    for (size_t j = 0; j < free.size(); ++j)
      Kf(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = K(free[i], free[j]);
  }
  const Eigen::VectorXd uf = Kf.ldlt().solve(ff);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(24);
  for (size_t i = 0; i < free.size(); ++i)
    u(free[i]) = uf(static_cast<Eigen::Index>(i));
  const auto resp = e.recover(std::span<const double>(u.data(), 24), temps, 0.0);

  const ShellSection s = build_shell_section(m, PropertyId{1}, ElementId{1}, "CQUAD4");
  Eigen::Matrix<double, 6, 6> C;
  C << s.A, s.B, s.B, s.D;
  Eigen::Matrix<double, 6, 1> rhs;
  rhs << s.thermal_N, s.thermal_M;
  const Eigen::Matrix<double, 6, 1> exact = C.ldlt().solve(rhs * dT);
  // The element frame of a unit square is the basic frame.
  for (int k = 0; k < 3; ++k) {
    EXPECT_NEAR(resp[0].membrane_strain(k), exact(k), 1e-6 * exact.head<3>().norm());
    EXPECT_NEAR(resp[0].curvature(k), exact(3 + k), 1e-6 * exact.tail<3>().norm());
    EXPECT_NEAR(resp[0].bending_moment(k), 0.0, 1e-6 * (s.thermal_M * dT).norm());
  }
  EXPECT_GT(exact.tail<3>().norm(), 1e-6); // the laminate does curl
}

TEST(ShellDirectors, SmoothJunctionsShareTheAverageNormalAndFoldsDoNot) {
  // Three quads around grid 2: two meet at 10 degrees (smooth), the third
  // at 90 degrees (a fold).  The shared director at grid 2 for the first
  // two is the bisector of their normals; the folded element keeps its
  // own.  PARAM,SNORM sets the fold angle.
  Model m;
  auto grid = [&m](int id, double x, double y, double z) {
    GridPoint g;
    g.id = NodeId{id};
    g.position = Vec3{x, y, z};
    m.nodes[g.id] = g;
  };
  const double a = 10.0 * std::numbers::pi / 180.0;
  grid(1, -1, 0, 0);
  grid(2, 0, 0, 0);
  grid(3, 0, 1, 0);
  grid(4, -1, 1, 0);
  grid(5, std::cos(a), 0, std::sin(a));  // continues at 10 degrees
  grid(6, std::cos(a), 1, std::sin(a));
  grid(7, 0, 0, 1);                       // folded up at 90 degrees
  grid(8, 0, 1, 1);
  auto quad = [&m](int id, std::array<int, 4> n) {
    ElementData e;
    e.id = ElementId{id};
    e.type = ElementType::CQUAD4;
    e.pid = PropertyId{1};
    for (int k : n)
      e.nodes.push_back(NodeId{k});
    m.elements.push_back(e);
  };
  quad(1, {1, 2, 3, 4});
  quad(2, {2, 5, 6, 3});
  quad(3, {2, 7, 8, 3});
  const Eigen::Vector3d n1 = shell_corner_normal(m, m.elements[0], 1);
  const Eigen::Vector3d n2 = shell_corner_normal(m, m.elements[1], 0);
  const Eigen::Vector3d n3 = shell_corner_normal(m, m.elements[2], 0);
  EXPECT_NEAR(std::acos(n1.dot(n2)), a, 1e-12);
  const Eigen::Vector3d d1 = shell_grid_director(m, m.elements[0], 1, n1);
  const Eigen::Vector3d d2 = shell_grid_director(m, m.elements[1], 0, n2);
  const Eigen::Vector3d d3 = shell_grid_director(m, m.elements[2], 0, n3);
  EXPECT_NEAR((d1 - (n1 + n2).normalized()).norm(), 0.0, 1e-12);
  EXPECT_NEAR((d1 - d2).norm(), 0.0, 1e-12);
  EXPECT_NEAR((d3 - n3).norm(), 0.0, 1e-12); // the fold keeps its own normal
  // A fold angle below 10 degrees separates elements 1 and 2 as well.
  m.params["SNORM"] = "5.0";
  EXPECT_NEAR((shell_grid_director(m, m.elements[0], 1, n1) - n1).norm(), 0.0, 1e-12);
}
