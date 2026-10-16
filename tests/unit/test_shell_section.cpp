// tests/unit/test_shell_section.cpp
// Shell section resultants (PSHELL, PCOMP) against hand calculations from
// classical lamination theory, and the PCOMP / corner-thickness parser.

#include <gtest/gtest.h>
#include "elements/shell_section.hpp"
#include "io/bdf_parser.hpp"
#include <cmath>

using namespace vibestran;

namespace {

Model with_mat8(int mid, double e1, double e2, double nu12, double g12,
                double g1z = 0.0, double g2z = 0.0, double a1 = 0.0,
                double a2 = 0.0, double rho = 0.0) {
  Model m;
  Mat8 mat;
  mat.id = MaterialId{mid};
  mat.e1 = e1;
  mat.e2 = e2;
  mat.nu12 = nu12;
  mat.g12 = g12;
  mat.g1z = g1z;
  mat.g2z = g2z;
  mat.a1 = a1;
  mat.a2 = a2;
  mat.rho = rho;
  m.mat8_materials[mat.id] = mat;
  return m;
}

void add_mat1(Model &m, int mid, double E, double nu, double rho = 0.0,
              double alpha = 0.0) {
  Mat1 mat;
  mat.id = MaterialId{mid};
  mat.E = E;
  mat.nu = nu;
  mat.G = E / (2.0 * (1.0 + nu));
  mat.rho = rho;
  mat.A = alpha;
  m.materials[mat.id] = mat;
}

Eigen::Matrix3d mat8_Q(double e1, double e2, double nu12, double g12) {
  const double nu21 = nu12 * e2 / e1, den = 1.0 - nu12 * nu21;
  Eigen::Matrix3d Q;
  Q << e1 / den, nu12 * e2 / den, 0, nu12 * e2 / den, e2 / den, 0, 0, 0, g12;
  return Q;
}

} // namespace

TEST(ShellSection, PshellUsesEachMaterialForItsRole) {
  // MID1 membrane x T, MID2 bending x 12I/T^3 T^3/12, MID3 shear x TS/T T,
  // MID4 coupling x T^2, thermal N = A alpha (MID1).
  Model m;
  add_mat1(m, 1, 100.0, 0.25, 2.0, 1e-3);
  add_mat1(m, 2, 300.0, 0.25);
  add_mat1(m, 3, 50.0, 0.0);
  add_mat1(m, 4, 10.0, 0.0);
  PShell ps;
  ps.pid = PropertyId{7};
  ps.mid1 = MaterialId{1};
  ps.mid2 = MaterialId{2};
  ps.mid3 = MaterialId{3};
  ps.mid4 = MaterialId{4};
  ps.t = 0.2;
  ps.twelveI_t3 = 1.5;
  ps.tst = 0.8;
  ps.nsm = 0.1;
  m.properties[ps.pid] = ps;
  const ShellSection s = build_shell_section(m, ps.pid, ElementId{1}, "CQUAD4");

  const double c1 = 100.0 / (1.0 - 0.0625);
  EXPECT_NEAR(s.A(0, 0), 0.2 * c1, 1e-12);
  EXPECT_NEAR(s.A(0, 1), 0.2 * 0.25 * c1, 1e-12);
  EXPECT_NEAR(s.A(2, 2), 0.2 * 100.0 / 2.5, 1e-12);
  const double c2 = 300.0 / (1.0 - 0.0625);
  EXPECT_NEAR(s.D(0, 0), 1.5 * 0.008 / 12.0 * c2, 1e-12);
  EXPECT_NEAR(s.T(0, 0), 0.8 * 0.2 * 25.0, 1e-12);
  EXPECT_NEAR(s.T(0, 1), 0.0, 1e-15);
  EXPECT_NEAR(s.B(0, 0), 0.04 * 10.0, 1e-12);
  EXPECT_NEAR(s.thermal_N(0), 0.2 * c1 * (1.0 + 0.25) * 1e-3, 1e-12);
  EXPECT_NEAR(s.thermal_N(2), 0.0, 1e-15);
  EXPECT_NEAR(s.mass_per_area, 2.0 * 0.2 + 0.1, 1e-12);
  EXPECT_NEAR(s.mass_second_moment, 2.0 * 0.008 / 12.0, 1e-15);
  EXPECT_NEAR(s.z1, -0.1, 1e-15);
  EXPECT_NEAR(s.z2, 0.1, 1e-15);
}

TEST(ShellSection, RotationPreservesStrainEnergyAndResultantWork) {
  // For any strain in the target frame, the energy computed in the target
  // frame must equal the energy of the rotated strain in the material frame.
  Model m = with_mat8(8, 140e3, 10e3, 0.3, 5e3, 4e3, 3e3, -1e-6, 3e-5);
  PComp pc;
  pc.pid = PropertyId{1};
  pc.plies = {{MaterialId{8}, 0.1, 0.0, false}, {MaterialId{8}, 0.1, 35.0, false}};
  m.properties[pc.pid] = pc;
  const ShellSection mat = build_shell_section(m, pc.pid, ElementId{1}, "CQUAD4");
  const double phi = 0.37;
  const ShellSection loc = mat.rotated(phi);
  const Eigen::Matrix3d Te = strain_rotation(phi);
  const Eigen::Vector3d e(1.0e-3, -4.0e-4, 7.0e-4), k(0.2, -0.1, 0.05);
  const Eigen::Vector3d em = Te * e, km = Te * k;
  const double w_loc = e.dot(loc.A * e) + 2.0 * e.dot(loc.B * k) + k.dot(loc.D * k);
  const double w_mat = em.dot(mat.A * em) + 2.0 * em.dot(mat.B * km) + km.dot(mat.D * km);
  EXPECT_NEAR(w_loc, w_mat, 1e-12 * std::abs(w_mat));
  EXPECT_NEAR(e.dot(loc.thermal_N) + k.dot(loc.thermal_M),
              em.dot(mat.thermal_N) + km.dot(mat.thermal_M), 1e-15);
  const Eigen::Vector2d g(2e-3, -1e-3);
  const double c = std::cos(phi), s = std::sin(phi);
  const Eigen::Vector2d gm(c * g(0) + s * g(1), -s * g(0) + c * g(1));
  EXPECT_NEAR(g.dot(loc.T * g), gm.dot(mat.T * gm), 1e-12 * gm.dot(mat.T * gm));
}

TEST(ShellSection, PcompSingleNinetyDegreePlySwapsTheMaterialAxes) {
  Model m = with_mat8(8, 140e3, 10e3, 0.3, 5e3, 4e3, 3e3);
  PComp pc;
  pc.pid = PropertyId{1};
  pc.plies = {{MaterialId{8}, 0.5, 90.0, false}};
  m.properties[pc.pid] = pc;
  const ShellSection s = build_shell_section(m, pc.pid, ElementId{1}, "CQUAD4");
  const Eigen::Matrix3d Q = mat8_Q(140e3, 10e3, 0.3, 5e3);
  EXPECT_NEAR(s.A(0, 0), 0.5 * Q(1, 1), 1e-9);
  EXPECT_NEAR(s.A(1, 1), 0.5 * Q(0, 0), 1e-9);
  EXPECT_NEAR(s.A(0, 1), 0.5 * Q(0, 1), 1e-9);
  EXPECT_NEAR(s.A(0, 2), 0.0, 1e-9);
  EXPECT_NEAR(s.T(0, 0), 5.0 / 6.0 * 0.5 * 3e3, 1e-9);
  EXPECT_NEAR(s.T(1, 1), 5.0 / 6.0 * 0.5 * 4e3, 1e-9);
  EXPECT_NEAR(s.B.norm(), 0.0, 1e-9); // single ply centred on the reference
  EXPECT_NEAR(s.D(0, 0), 0.125 / 12.0 * Q(1, 1), 1e-9);
}

TEST(ShellSection, UnsymmetricCrossPlyCouplingMatchesHandLamination) {
  // [0/90] with plies of thickness h: ply 0 in [-h, 0], ply 90 in [0, h].
  // Physical B_p = sum Qbar (z1^2 - z0^2)/2 = h^2/2 (Q90 - Q0); Nastran's
  // curvature sign gives B = -B_p.  D = h^3/3 (Q0 + Q90).  A uniform
  // temperature rise bends it: M_T = -sum Qbar alpha (z1^2 - z0^2)/2.
  const double h = 0.25, a1 = -1e-6, a2 = 3e-5;
  Model m = with_mat8(8, 140e3, 10e3, 0.3, 5e3, 4e3, 3e3, a1, a2);
  PComp pc;
  pc.pid = PropertyId{1};
  pc.plies = {{MaterialId{8}, h, 0.0, false}, {MaterialId{8}, h, 90.0, false}};
  m.properties[pc.pid] = pc;
  const ShellSection s = build_shell_section(m, pc.pid, ElementId{1}, "CQUAD4");
  const Eigen::Matrix3d Q0 = mat8_Q(140e3, 10e3, 0.3, 5e3);
  Eigen::Matrix3d Q90 = Q0;
  std::swap(Q90(0, 0), Q90(1, 1));
  EXPECT_NEAR((s.A - h * (Q0 + Q90)).norm(), 0.0, 1e-8);
  EXPECT_NEAR((s.B + 0.5 * h * h * (Q90 - Q0)).norm(), 0.0, 1e-8);
  EXPECT_NEAR((s.D - h * h * h / 3.0 * (Q0 + Q90)).norm(), 0.0, 1e-8);
  const Eigen::Vector3d al0(a1, a2, 0.0), al90(a2, a1, 0.0);
  const Eigen::Vector3d MTp = 0.5 * h * h * (Q90 * al90 - Q0 * al0);
  EXPECT_NEAR((s.thermal_M + MTp).norm(), 0.0, 1e-12);
  EXPECT_NEAR((s.thermal_N - h * (Q0 * al0 + Q90 * al90)).norm(), 0.0, 1e-12);
}

TEST(ShellSection, ThicknessScalingFollowsTheThicknessPowers) {
  Model m;
  add_mat1(m, 1, 100.0, 0.3, 1.0, 1e-5);
  PShell ps;
  ps.pid = PropertyId{1};
  ps.mid1 = ps.mid2 = ps.mid3 = MaterialId{1};
  ps.t = 0.1;
  m.properties[ps.pid] = ps;
  const ShellSection s = build_shell_section(m, ps.pid, ElementId{1}, "CQUAD4");
  ps.t = 0.15;
  m.properties[ps.pid] = ps;
  const ShellSection thick = build_shell_section(m, ps.pid, ElementId{1}, "CQUAD4");
  const ShellSection scaled = s.scaled_thickness(1.5);
  EXPECT_NEAR((scaled.A - thick.A).norm(), 0.0, 1e-12);
  EXPECT_NEAR((scaled.D - thick.D).norm(), 0.0, 1e-12);
  EXPECT_NEAR((scaled.T - thick.T).norm(), 0.0, 1e-12);
  EXPECT_NEAR((scaled.thermal_N - thick.thermal_N).norm(), 0.0, 1e-15);
  EXPECT_NEAR(scaled.mass_per_area, thick.mass_per_area, 1e-15);
  EXPECT_NEAR(scaled.mass_second_moment, thick.mass_second_moment, 1e-15);
}

TEST(BdfParser, PcompPliesRepeatBlanksAndMirrorSymmetricLaminates) {
  const Model m = BdfParser::parse_string(R"(
SOL 101
CEND
BEGIN BULK
MAT8,8,140.e3,10.e3,0.3,5.e3,4.e3,3.e3,1.5
PCOMP,5,-0.4,0.02,,,20.,,SYM
,8,0.1,0.,YES,,,45.,
,,0.2,-45.
ENDDATA
)");
  const auto &pc = std::get<PComp>(m.property(PropertyId{5}));
  EXPECT_NEAR(pc.z0, -0.4, 1e-15);
  EXPECT_NEAR(pc.nsm, 0.02, 1e-15);
  EXPECT_NEAR(pc.tref, 20.0, 1e-15);
  EXPECT_EQ(pc.lam, LaminateOption::Sym);
  ASSERT_EQ(pc.plies.size(), 6u); // three listed plies, mirrored
  EXPECT_EQ(pc.plies[1].mid.value, 8);  // blank MID repeats the previous
  EXPECT_NEAR(pc.plies[1].t, 0.1, 1e-15); // blank T repeats the previous
  EXPECT_NEAR(pc.plies[1].theta, 45.0, 1e-15);
  EXPECT_NEAR(pc.plies[2].t, 0.2, 1e-15);
  EXPECT_NEAR(pc.plies[3].theta, -45.0, 1e-15);
  EXPECT_NEAR(pc.plies[5].theta, 0.0, 1e-15);
  EXPECT_TRUE(pc.plies[0].sout);
  EXPECT_FALSE(pc.plies[1].sout);
  EXPECT_NEAR(pc.total_thickness(), 0.8, 1e-15);
}

TEST(BdfParser, ShellCornerThicknessesAndTflag) {
  const Model m = BdfParser::parse_string(R"(
SOL 101
CEND
BEGIN BULK
CQUAD4,1,1,1,2,3,4,30.,0.05
,,,1,1.1,,0.9,1.2
CTRIA3,2,1,1,2,3
CQUAD4,3,1,1,2,3,4
ENDDATA
)");
  ASSERT_EQ(m.elements.size(), 3u);
  const auto &q = m.elements[0];
  EXPECT_EQ(q.tflag, 1);
  ASSERT_EQ(q.corner_thickness.size(), 4u);
  EXPECT_NEAR(q.corner_thickness[0].value(), 1.1, 1e-15);
  EXPECT_FALSE(q.corner_thickness[1].has_value());
  EXPECT_NEAR(q.corner_thickness[3].value(), 1.2, 1e-15);
  EXPECT_NEAR(q.zoffs, 0.05, 1e-15);
  EXPECT_NEAR(q.theta.value(), 30.0, 1e-15);
  EXPECT_TRUE(m.elements[1].corner_thickness.empty());
  EXPECT_TRUE(m.elements[2].corner_thickness.empty());
}
