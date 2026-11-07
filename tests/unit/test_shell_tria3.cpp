// tests/unit/test_shell_tria3.cpp
// Basic tests of the MITC3+ CTRIA3 (Lee, Lee & Bathe 2014): zero-energy
// modes, rigid-body motion, spatial isotropy and invariance to the grid
// numbering, the membrane and bending patch tests, and thermal laminate
// behaviour shared with the CQUAD4.

#include <gtest/gtest.h>
#include "elements/shell_tria3.hpp"
#include "io/bdf_parser.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <format>
#include <functional>
#include <numbers>
#include <sstream>

using namespace vibestran;

namespace {

Model tria_model(const std::array<std::array<double, 3>, 3> &xyz, double t = 0.1) {
  Model m;
  for (int i = 0; i < 3; ++i) {
    GridPoint gp;
    gp.id = NodeId{i + 1};
    const auto &p = xyz[static_cast<size_t>(i)];
    gp.position = Vec3{p[0], p[1], p[2]};
    m.nodes[gp.id] = gp;
  }
  Mat1 mat;
  mat.id = MaterialId{1};
  mat.E = 1.0e7;
  mat.nu = 0.3;
  mat.G = mat.E / 2.6;
  mat.rho = 2.0;
  m.materials[mat.id] = mat;
  PShell ps;
  ps.pid = PropertyId{1};
  ps.mid1 = ps.mid2 = ps.mid3 = MaterialId{1};
  ps.t = t;
  m.properties[ps.pid] = ps;
  return m;
}

ElementData tria_data(std::array<int, 3> n = {1, 2, 3}) {
  ElementData d;
  d.id = ElementId{1};
  d.pid = PropertyId{1};
  d.type = ElementType::CTRIA3;
  for (int k : n)
    d.nodes.push_back(NodeId{k});
  return d;
}

const std::array<std::array<std::array<double, 3>, 3>, 2> kShapes{{
    {{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}},
    {{{0.1, 0.2, 0.3}, {1.3, 0.1, -0.2}, {0.4, 0.9, 0.5}}},
}};

} // namespace

TEST(ShellTria3, HasExactlySixZeroEnergyModesAndRigidBodyMotion) {
  for (const auto &xyz : kShapes) {
    const Model m = tria_model(xyz);
    const Eigen::MatrixXd K = ShellTria3(tria_data(), m).stiffness_matrix();
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (K + K.transpose()));
    const double tol = 1e-9 * es.eigenvalues().cwiseAbs().maxCoeff();
    int zeros = 0;
    for (int i = 0; i < 18; ++i) {
      EXPECT_GT(es.eigenvalues()(i), -tol);
      zeros += std::abs(es.eigenvalues()(i)) < tol ? 1 : 0;
    }
    EXPECT_EQ(zeros, 6);
    for (int k = 0; k < 6; ++k) {
      Eigen::Vector3d c = Eigen::Vector3d::Zero(), w = Eigen::Vector3d::Zero();
      (k < 3 ? c(k) : w(k - 3)) = 1.0;
      Eigen::VectorXd u(18);
      for (int i = 0; i < 3; ++i) {
        const auto &p = xyz[static_cast<size_t>(i)];
        u.segment<3>(6 * i) = c + w.cross(Eigen::Vector3d(p[0], p[1], p[2]));
        u.segment<3>(6 * i + 3) = w;
      }
      EXPECT_LT((K * u).cwiseAbs().maxCoeff(), 1e-9 * K.cwiseAbs().maxCoeff()) << k;
    }
  }
}

TEST(ShellTria3, StiffnessDoesNotDependOnTheGridNumbering) {
  // MITC3+ is spatially isotropic: renumbering the grids cyclically permutes
  // the stiffness blocks and changes nothing else.
  const Model m = tria_model(kShapes[1]);
  const Eigen::MatrixXd K = ShellTria3(tria_data({1, 2, 3}), m).stiffness_matrix();
  const Eigen::MatrixXd Kp = ShellTria3(tria_data({2, 3, 1}), m).stiffness_matrix();
  // Local block (i, j) of Kp corresponds to grids (i+1, j+1) of K.
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_LT((Kp.block<6, 6>(6 * i, 6 * j) -
                 K.block<6, 6>(6 * ((i + 1) % 3), 6 * ((j + 1) % 3)))
                    .cwiseAbs()
                    .maxCoeff(),
                1e-9 * K.cwiseAbs().maxCoeff());
}

namespace {

// The 10 x 10 distorted patch of the quad tests, each quad split into two
// triangles.
const std::array<std::array<double, 2>, 8> kPatchXY{{
    {0, 0}, {10, 0}, {10, 10}, {0, 10}, {2, 2}, {8, 3}, {8, 7}, {4, 7}}};
const std::array<std::array<int, 3>, 10> kPatchTrias{{
    {1, 2, 6}, {1, 6, 5}, {2, 3, 7}, {2, 7, 6}, {3, 4, 8},
    {3, 8, 7}, {4, 1, 5}, {4, 5, 8}, {5, 6, 7}, {5, 7, 8}}};

SolverResults run_patch(const std::function<std::array<double, 6>(double, double)> &exact,
                        double t) {
  std::ostringstream os;
  os << "SOL 101\nCEND\nSUBCASE 1\n  SPC = 1\n  LOAD = 1\n  DISPLACEMENT = ALL\n"
        "  STRESS = ALL\nBEGIN BULK\n";
  for (int i = 0; i < 8; ++i)
    os << std::format("GRID,{},,{},{},0.\n", i + 1, kPatchXY[static_cast<size_t>(i)][0],
                      kPatchXY[static_cast<size_t>(i)][1]);
  for (int e = 0; e < 10; ++e) {
    const auto &q = kPatchTrias[static_cast<size_t>(e)];
    os << std::format("CTRIA3,{},1,{},{},{}\n", e + 1, q[0], q[1], q[2]);
  }
  os << std::format("PSHELL,1,1,{},1,,1\nMAT1,1,1.0E6,,0.25\n", t);
  for (int i = 0; i < 4; ++i) {
    const auto v = exact(kPatchXY[static_cast<size_t>(i)][0], kPatchXY[static_cast<size_t>(i)][1]);
    for (int k = 0; k < 5; ++k)
      os << std::format("SPC,1,{},{},{:.10e}\n", i + 1, k + 1, v[static_cast<size_t>(k)]);
  }
  os << "FORCE,1,1,,0.,1.,0.,0.\nENDDATA\n";
  Model model = BdfParser::parse_string(os.str());
  LinearStaticSolver solver(std::make_unique<LdltSolverBackend>());
  return solver.solve(model);
}

} // namespace

TEST(ShellTria3, MembraneAndBendingPatchTests) {
  auto membrane = [](double x, double y) {
    return std::array<double, 6>{1e-3 * (x + y / 2), 1e-3 * (y + x / 2), 0, 0, 0, 0};
  };
  const SolverResults rm = run_patch(membrane, 0.001);
  for (const auto &nd : rm.subcases.at(0).displacements) {
    const auto &p = kPatchXY[static_cast<size_t>(nd.node.value - 1)];
    EXPECT_NEAR(nd.d[0], membrane(p[0], p[1])[0], 1e-12);
    EXPECT_NEAR(nd.d[1], membrane(p[0], p[1])[1], 1e-12);
  }
  for (const auto &ps : rm.subcases.at(0).plate_stresses) {
    EXPECT_NEAR(ps.strain.ex + ps.strain.ey, 2e-3, 1e-12);
    EXPECT_NEAR(std::pow(ps.strain.ex - ps.strain.ey, 2) + std::pow(ps.strain.exy, 2),
                1e-6, 1e-15);
  }
  auto bending = [](double x, double y) {
    const double w = 0.5e-3 * (x * x + x * y + y * y);
    return std::array<double, 6>{0, 0, w, 0.5e-3 * (x + 2 * y), -0.5e-3 * (2 * x + y), 0};
  };
  const SolverResults rb = run_patch(bending, 0.001);
  for (const auto &nd : rb.subcases.at(0).displacements) {
    const auto &p = kPatchXY[static_cast<size_t>(nd.node.value - 1)];
    EXPECT_NEAR(nd.d[2], bending(p[0], p[1])[2], 1e-9) << nd.node.value;
  }
  for (const auto &ps : rb.subcases.at(0).plate_stresses) {
    EXPECT_NEAR(ps.strain.kx + ps.strain.ky, 2e-3, 1e-9);
    EXPECT_NEAR(std::pow(ps.strain.kx - ps.strain.ky, 2) + std::pow(ps.strain.kxy, 2),
                1e-6, 1e-12);
  }
}

TEST(ShellTria3, TransverseShearIsRecovered) {
  // Unlike the old Kirchhoff DKT element, MITC3+ carries transverse shear:
  // a cantilever strip under a tip shear force reports Qx = F / b.
  std::ostringstream os;
  os << "SOL 101\nCEND\nSUBCASE 1\n  SPC = 1\n  LOAD = 1\n  FORCE = ALL\nBEGIN BULK\n";
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i <= 20; ++i)
      os << std::format("GRID,{},,{},{},0.\n", 1 + i + 21 * j, 0.5 * i, 1.0 * j);
  int e = 1;
  for (int i = 0; i < 20; ++i) {
    const int a = 1 + i;
    os << std::format("CTRIA3,{},1,{},{},{}\nCTRIA3,{},1,{},{},{}\n", e, a, a + 1, a + 22,
                      e + 1, a, a + 22, a + 21);
    e += 2;
  }
  os << "PSHELL,1,1,0.1,1,,1\nMAT1,1,1.0E7,,0.3\nSPC1,1,123456,1,22\n"
        "FORCE,1,21,,0.5,0.,0.,1.\nFORCE,1,42,,0.5,0.,0.,1.\nENDDATA\n";
  Model model = BdfParser::parse_string(os.str());
  LinearStaticSolver solver(std::make_unique<LdltSolverBackend>());
  const SolverResults res = solver.solve(model);
  int checked = 0;
  for (const auto &ps : res.subcases.at(0).plate_stresses) {
    if (ps.eid.value < 11 || ps.eid.value > 30)
      continue; // away from the clamp and the loaded end
    // Element frames: x along G1-G2, which is +x for the first triangle of
    // each pair and along the diagonal for the second; |Q| = F / b either
    // way.
    EXPECT_NEAR(std::hypot(ps.qx, ps.qy), 1.0, 0.05) << ps.eid.value;
    ++checked;
  }
  EXPECT_EQ(checked, 20);
}

TEST(ShellTria3, PureTwistEnergyAndMassAreExact) {
  // Constant twist w = x y is a constant-curvature state, reproduced
  // exactly; the translational mass is rho t A per direction.
  const double E = 1.0e7, nu = 0.3, t = 0.1, rho = 2.0;
  const Model m = tria_model({{{0, 0, 0}, {2, 0, 0}, {0.5, 1.5, 0}}}, t);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(18);
  for (int i = 0; i < 3; ++i) {
    const Vec3 &p = m.node(NodeId{i + 1}).position;
    u(6 * i + 2) = p.x * p.y;
    u(6 * i + 3) = p.x;
    u(6 * i + 4) = -p.y;
  }
  const ShellTria3 e(tria_data(), m);
  const double D = E * t * t * t / (12.0 * (1.0 - nu * nu));
  const double area = 0.5 * 2.0 * 1.5;
  const double exact = 0.5 * D * (1.0 - nu) / 2.0 * 4.0 * area;
  EXPECT_NEAR(0.5 * u.dot(e.stiffness_matrix() * u), exact, 1e-9 * exact);
  const Eigen::MatrixXd M = e.mass_matrix();
  for (int dir = 0; dir < 3; ++dir) {
    double total = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        total += M(6 * a + dir, 6 * b + dir);
    EXPECT_NEAR(total, rho * t * area, 1e-12);
  }
}

TEST(ShellTria3, CollinearGridsAreRejected) {
  const Model m = tria_model({{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}});
  EXPECT_THROW((void)ShellTria3(tria_data(), m).stiffness_matrix(), SolverError);
}

TEST(ShellTria3, TiltedSnormDirectorsKeepThinPlateBending) {
  // As for the CQUAD4: 45-degree SNORM directors carry the thickness
  // t / (V . n), so a thin cantilever (L/t = 100, 20 x 4 cells split into
  // triangles) deflects as the untilted one.
  const auto tip_w = [](const char *snorm) {
    std::ostringstream os;
    os << "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nDISPLACEMENT = ALL\nBEGIN BULK\n"
          "MAT1,1,1.0E7,,0.3\nPSHELL,1,1,0.01,1,,1\n";
    const auto id = [](int i, int j) { return 1 + i + 21 * j; };
    for (int j = 0; j <= 4; ++j)
      for (int i = 0; i <= 20; ++i) {
        os << std::format("GRID,{},,{},{},0.\n", id(i, j), 0.05 * i, 0.05 * j);
        if (snorm != nullptr)
          os << std::format("SNORM,{},0,{}\n", id(i, j), snorm);
      }
    int e = 1;
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 20; ++i) {
        os << std::format("CTRIA3,{},1,{},{},{}\n", e++, id(i, j), id(i + 1, j),
                          id(i + 1, j + 1));
        os << std::format("CTRIA3,{},1,{},{},{}\n", e++, id(i, j), id(i + 1, j + 1),
                          id(i, j + 1));
      }
    for (int j = 0; j <= 4; ++j) {
      os << std::format("SPC1,1,123456,{}\n", id(0, j));
      os << std::format("FORCE,1,{},,2.0E-4,0.,0.,1.\n", id(20, j));
    }
    os << "ENDDATA\n";
    Model model = BdfParser::parse_string(os.str());
    LinearStaticSolver solver(std::make_unique<LdltSolverBackend>());
    const SolverResults res = solver.solve(model);
    for (const auto &nd : res.subcases.at(0).displacements)
      if (nd.node.value == id(20, 2)) // cppcheck-suppress useStlAlgorithm
        return nd.d[2];
    return std::nan("");
  };
  const double w0 = tip_w(nullptr);
  for (const char *n : {"0.,1.,1.", "1.,0.,1."})
    EXPECT_NEAR(tip_w(n) / w0, 1.0, 5e-3) << n;
}

TEST(ShellTria3, RadialExpansionOfACylinderBendsTheWallWithoutMoment) {
  // As for the CQUAD4 (test_shell_quad4.cpp): a ring of 72 facets, each
  // split into two triangles, expanded by w.  kappa_hoop = +w / R^2 and the
  // hoop moment vanishes once the through-thickness metric and volume
  // factor are included.
  constexpr int kN = 72;
  constexpr double R = 10.0, L = 1.0, t = 1.0, nu = 0.3, w = 1.0e-3;
  Model m = tria_model(kShapes[0], t);
  m.nodes.clear();
  const auto id = [](int i, int j) { return 1 + (i % kN) + kN * j; };
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < kN; ++i) {
      const double a = 2.0 * std::numbers::pi * i / kN;
      GridPoint gp;
      gp.id = NodeId{id(i, j)};
      gp.position = Vec3{R * std::cos(a), R * std::sin(a), L * j};
      m.nodes[gp.id] = gp;
    }
  int eid = 1;
  for (int i = 0; i < kN; ++i) {
    for (const std::array<int, 3> n : {std::array<int, 3>{id(i, 0), id(i + 1, 0), id(i + 1, 1)},
                                       std::array<int, 3>{id(i, 0), id(i + 1, 1), id(i, 1)}}) {
      ElementData e = tria_data(n);
      e.id = ElementId{eid++};
      m.elements.push_back(e);
    }
  }
  const ShellTria3 el(m.elements[0], m);
  std::vector<double> u(18, 0.0);
  for (int k = 0; k < 3; ++k) {
    const Vec3 &p = m.node(m.elements[0].nodes[static_cast<size_t>(k)]).position;
    u[static_cast<size_t>(6 * k)] = w * p.x / R;
    u[static_cast<size_t>(6 * k + 1)] = w * p.y / R;
  }
  const ShellPointResponse c = el.recover(u, {}, 0.0)[0];
  const Mat1 &mat = m.material(MaterialId{1});
  const double D = mat.E * t * t * t / (12.0 * (1.0 - nu * nu));
  EXPECT_NEAR(c.membrane_strain(0), w / R, 1e-3 * w / R);
  EXPECT_NEAR(c.curvature(0), w / (R * R), 1e-2 * w / (R * R));
  EXPECT_NEAR(c.bending_moment(0), 0.0, 1e-2 * D * w / (R * R));
}
