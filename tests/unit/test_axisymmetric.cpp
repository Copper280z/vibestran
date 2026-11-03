// tests/unit/test_axisymmetric.cpp
// MAGAXI: axisymmetric MAGSTAT, A = A_phi e_phi on a mesh in the x-z
// half-plane (x = r). The tests are against exact solutions: a uniform axial
// field, which first-order elements represent exactly (A = B0 r / 2), and two
// infinitely long cases (flux-normal ends) whose field is piecewise
// polynomial in r but whose potential is not, so the discrete energy must
// converge at second order: an iron tube in an axial field and a thick
// solenoid. Forces are checked against the derivative of the coenergy.

#include "magnetic/model.hpp"
#include "magnetic/solver.hpp"
#include "tube_deck.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <functional>
#include <map>
#include <set>
#include <numbers>
#include <string>
#include <vector>

using namespace vibestran::magnetic;
using namespace tube_deck;

namespace {
// Simpson's rule on [a, b] with n (even) panels; exact for cubics.
double simpson(const std::function<double(double)> &f, double a, double b, int n = 64) {
  const double h = (b - a) / n;
  double sum = f(a) + f(b);
  for (int k = 1; k < n; ++k)
    sum += (k % 2 ? 4 : 2) * f(a + k * h);
  return sum * h / 3;
}

} // namespace

TEST(Axisymmetric, UniformAxialFieldIsExact) {
  // A = B0 r / 2 is linear, so quads and triangles on a graded mesh carry
  // it exactly: B is B0 e_z in every cell and at probes (one on the axis,
  // where A / r becomes dA/dr), and the energy is B0^2 / (2 mu0) times the
  // volume of the cylinder.
  constexpr double b0 = .7, radius = .05, height = .06;
  for (const bool triangles : {false, true}) {
    SCOPED_TRACE(triangles ? "triangles" : "quads");
    Axi g;
    g.r = lines(0, {{.02, 4}, {radius, 5}});
    g.z = lines(-height / 2, {{.01, 3}, {height / 2, 4}});
    g.triangles = triangles;
    g.control = "MAGBC = 1\n";
    g.cards = std::format("PMAG2D,1,1\nMATMAG,1,1.\nMBC,1,0,UNIFORM,0.,0.,{}\n"
                          "MPROBE,1,0,.013,0.,.004\nMPROBE,2,0,0.,0.,-.007\n",
                          b0);
    const auto r = solved(parsed(g));
    ASSERT_EQ(r.cases.size(), 1U);
    const auto &s = r.cases[0];
    for (const auto &b : s.b)
      EXPECT_LT((b - Eigen::Vector3d(0, 0, b0)).norm(), 1e-12 * b0);
    ASSERT_EQ(s.probe_b.size(), 2U);
    for (const auto &b : s.probe_b)
      EXPECT_LT((b - Eigen::Vector3d(0, 0, b0)).norm(), 1e-12 * b0) << b.transpose();
    const double exact = b0 * b0 / (2 * mu0) * pi * radius * radius * height;
    EXPECT_LT(relative(s.energy, exact), 1e-12);
  }
}

TEST(Axisymmetric, IronTubeInAnAxialFieldConvergesAtSecondOrder) {
  // An infinitely long iron tube (a < r < b, mu_r = 100) in air, flux-normal
  // ends (MBC NORMAL) and A = B0 R / 2 at r = R, which fixes the flux to
  // pi R^2 B0. With curl H = 0 and no radial field, H_z = H0 is the same at
  // every r: B is mu0 H0 in air and mu_r mu0 H0 in iron, with H0 from the
  // flux. The potential (1/r) int B r dr is c1 r + c2 / r in and beyond the
  // tube, not linear, so the energy error falls by about 4 per halving.
  constexpr double a = .01, b = .02, radius = .05, height = .01, b0 = .3, mur = 100;
  const double h0 = b0 * radius * radius / (mu0 * (radius * radius + (mur - 1) * (b * b - a * a)));
  const double exact =
      height * pi * mu0 * h0 * h0 * ((radius * radius - (b * b - a * a)) + mur * (b * b - a * a)) /
      2;
  for (const bool triangles : {false, true}) {
    SCOPED_TRACE(triangles ? "triangles" : "quads");
    std::vector<double> errors;
    double inner_b = 0;
    for (int k : {4, 8, 16}) {
      Axi g;
      g.r = lines(0, {{a, k}, {b, k}, {radius, 3 * k}});
      g.z = lines(0, {{height, 2}});
      g.triangles = triangles;
      g.property = [&](double rc, double) { return rc > a && rc < b ? 2 : 1; };
      g.control = "MAGBC = 1\n";
      g.cards = std::format("PMAG2D,1,1\nMATMAG,1,1.\nPMAG2D,2,2\nMATMAG,2,{}\n"
                            "MBC,1,0,UNIFORM,0.,0.,{}\nMBC,1,1,NORMAL\nMBC,1,2,NORMAL\n",
                            mur, b0);
      const auto r = solved(parsed(g));
      ASSERT_EQ(r.cases.size(), 1U);
      errors.push_back(relative(r.cases[0].energy, exact));
      inner_b = r.cases[0].b.front().z(); // the cell at the axis, bottom
    }
    EXPECT_GT(errors[0] / errors[1], 3.0) << errors[0] << " " << errors[1];
    EXPECT_GT(errors[1] / errors[2], 3.0) << errors[1] << " " << errors[2];
    EXPECT_LT(errors[2], 3e-3);
    // B itself is first order: a few parts in a thousand at 16 cells.
    EXPECT_LT(relative(inner_b, mu0 * h0), 3e-3);
  }
}

TEST(Axisymmetric, ThickSolenoidInductanceConvergesToTheExactValue) {
  // An infinitely long thick solenoid, N turns in a < r < b over the height
  // L (flux-normal ends), inside A = 0 at r = R, so no net flux crosses the
  // midplane: H = H_in inside, falls linearly by J (b - a) across the
  // winding, and is H_in - J (b - a) beyond it, with H_in from zero flux.
  // One COIL2D region is a closed winding (each turn a loop around the
  // axis): its linkage and inductance are defined. L = 2 W / I^2, with W
  // integrated exactly (Simpson's rule on the piecewise-linear H).
  constexpr double a = .01, b = .015, radius = .03, height = .02, turns = 100, amps = 2;
  const double j = turns * amps / ((b - a) * height);
  const double h_in = 2 * j *
                      ((b * b * b - a * a * a) / 3 - a * (b * b - a * a) / 2 +
                       (b - a) * (radius * radius - b * b) / 2) /
                      (radius * radius);
  const auto field = [&](double r) {
    return r < a ? h_in : r < b ? h_in - j * (r - a) : h_in - j * (b - a);
  };
  const auto density = [&](double r) { return mu0 * field(r) * field(r) / 2 * 2 * pi * r; };
  const double energy = height * (simpson(density, 0, a) + simpson(density, a, b) +
                                  simpson(density, b, radius));
  const double inductance = 2 * energy / (amps * amps);
  std::vector<double> errors;
  for (int k : {4, 8, 16}) {
    Axi g;
    g.r = lines(0, {{a, 2 * k}, {b, k}, {radius, 3 * k}});
    g.z = lines(0, {{height, 2}});
    g.property = [&](double rc, double) { return rc > a && rc < b ? 2 : 1; };
    g.control = "LOAD = 5\nMAGBC = 1\n";
    g.cards = std::format("PMAG2D,1,1\nPMAG2D,2,1\nMATMAG,1,1.\nCOIL2D,7,102,{}\n"
                          "ICIRC,5,7,{}\nMBC,1,0,ZERO\nMBC,1,1,NORMAL\nMBC,1,2,NORMAL\n",
                          turns, amps);
    const auto r = solved(parsed(g));
    ASSERT_EQ(r.cases.size(), 1U);
    const auto &s = r.cases[0];
    ASSERT_EQ(s.circuits.size(), 1U);
    // The linkage is L I, and the coenergy of a linear model its energy.
    EXPECT_LT(relative(s.circuits[0].linkage, s.circuit_inductance(0, 0) * amps), 1e-9);
    EXPECT_LT(relative(s.coenergy, s.energy), 1e-9);
    errors.push_back(relative(s.circuit_inductance(0, 0), inductance));
  }
  EXPECT_GT(errors[0] / errors[1], 3.0) << errors[0] << " " << errors[1];
  EXPECT_GT(errors[1] / errors[2], 3.0) << errors[1] << " " << errors[2];
  EXPECT_LT(errors[2], 5e-4);
}

TEST(Axisymmetric, AxialForceIsTheDerivativeOfTheCoenergy) {
  // An iron ring above a coil, both around the axis: the axial MFORCE on the
  // ring (BODY, its surface of revolution closing on the axis, and an
  // explicit contour ending on the axis) must equal dW'/dz at fixed current,
  // here by a central difference of the coenergy over decks with the ring
  // shifted by +-dz. The radial force cancels around the axis, and there is
  // no torque. Both discretizations converge to the same force (0.1% apart
  // at twice this resolution); dz is small enough that the difference's own
  // error, O(dz^2), is 3e-4 of the force.
  constexpr double dz = .0001;
  std::vector<double> coenergy;
  Vector body_force = Vector::Zero(), contour_force = Vector::Zero();
  for (int shift : {-1, 0, 1}) {
    // Rows are 1 mm; the ring (r .002..010, 2 mm tall) starts at z0, which
    // the mesh follows by moving the rows between the coil and the ring.
    // Two air layers around the ring reach the axis.
    const double z0 = .012 + shift * dz;
    Axi g;
    g.r = lines(0, {{.002, 4}, {.010, 16}, {.016, 12}, {.04, 16}});
    g.z = lines(-.03, {{0, 20}, {.004, 8}, {.008, 8}, {z0, 8}, {z0 + .002, 4},
                       {z0 + .006, 8}, {.05, 20}});
    g.property = [&](double rc, double zc) {
      if (rc > .002 && rc < .010 && zc > z0 && zc < z0 + .002)
        return 2; // iron ring
      if (rc > .002 && rc < .010 && zc > 0 && zc < .004)
        return 3; // coil
      return 1;
    };
    g.control = "LOAD = 5\nMAGBC = 1\n";
    g.cards = "PMAG2D,1,1\nPMAG2D,3,1\nMATMAG,1,1.\nPMAG2D,2,2\nMATMAG,2,800.\n"
              "COIL2D,7,103,200\nICIRC,5,7,3.\nMBC,1,0,ZERO\nMFORCE,1,BODY,102,0,2\n";
    auto m = parsed(g);
    if (shift == 0) {
      // A contour in the air around the ring, rows z0 - .002 and z0 + .004
      // and radius .013, ending on the axis. Each face runs so that its
      // tangent x e_phi (= y) points outward: out along the top, down the
      // side and back in along the bottom.
      const auto r_index = [&](double r) {
        return static_cast<int>(std::ranges::find_if(g.r, [r](double x) {
                                  return std::abs(x - r) < 1e-12;
                                }) - g.r.begin());
      };
      const auto z_index = [&](double z) {
        return static_cast<int>(std::ranges::find_if(g.z, [z](double x) {
                                  return std::abs(x - z) < 1e-12;
                                }) - g.z.begin());
      };
      const int nr = static_cast<int>(g.r.size());
      const auto grid = [nr](int i, int jj) { return 1 + i + nr * jj; };
      const int i_out = r_index(.013), j_low = z_index(z0 - .002), j_high = z_index(z0 + .004);
      ASSERT_LT(i_out, nr);
      ASSERT_LT(j_high, static_cast<int>(g.z.size()));
      std::string contour;
      auto cell = [&](int i, int jj) { return 1 + i + (nr - 1) * jj; };
      for (int i = 0; i < i_out; ++i) // top, normal +z
        contour += std::format("FSET,9,{},{},{}\n", cell(i, j_high), grid(i, j_high),
                               grid(i + 1, j_high));
      for (int jj = j_high; jj > j_low; --jj) // side, normal +r
        contour += std::format("FSET,9,{},{},{}\n", cell(i_out, jj - 1), grid(i_out, jj),
                               grid(i_out, jj - 1));
      for (int i = i_out; i > 0; --i) // bottom, normal -z
        contour += std::format("FSET,9,{},{},{}\n", cell(i - 1, j_low), grid(i, j_low),
                               grid(i - 1, j_low));
      g.cards += "MFORCE,2,FACES,9\n" + contour;
      m = parsed(g);
    }
    const auto r = solved(m);
    ASSERT_EQ(r.cases.size(), 1U);
    coenergy.push_back(r.cases[0].coenergy);
    if (shift == 0) {
      ASSERT_EQ(r.cases[0].forces.size(), 2U);
      body_force = r.cases[0].forces[0].force;
      contour_force = r.cases[0].forces[1].force;
      EXPECT_EQ(r.cases[0].forces[0].torque, Vector::Zero());
    }
  }
  const double derivative = (coenergy[2] - coenergy[0]) / (2 * dz);
  // The coil attracts the ring, toward -z.
  EXPECT_LT(derivative, 0);
  EXPECT_EQ(body_force.x(), 0);
  EXPECT_EQ(body_force.y(), 0);
  EXPECT_NEAR(body_force.z(), derivative, .01 * std::abs(derivative));
  EXPECT_NEAR(contour_force.z(), derivative, .01 * std::abs(derivative));
  EXPECT_NEAR(contour_force.z(), body_force.z(), .005 * std::abs(derivative));
}

TEST(Axisymmetric, CardsAndChecks) {
  Axi g;
  g.r = lines(0, {{.02, 4}});
  g.z = lines(0, {{.02, 4}});
  g.cards = "PMAG2D,1,1\nMATMAG,1,1.\n";
  const auto error = [&](const std::string &cards, const std::string &control = "") {
    Axi with = g;
    with.cards += cards;
    with.control = control;
    auto m = parse_string(deck(with));
    if (const auto *e = std::get_if<vibestran::Error>(&m))
      return e->message;
    auto r = solve(std::get<Model>(m));
    return error_of(r);
  };
  EXPECT_EQ(error(""), "");
  EXPECT_NE(error("MAG2D,0,.1\n").find("one MAG2D or MAGAXI"), std::string::npos);
  // Remanence must lie in the meridian plane: tangential (y, or AXIS 2 of a
  // cylindrical system about the axis) is not axisymmetric.
  EXPECT_NE(error("MATMAG,3,1.05,,1.\nPMAG2D,3,3,0,2\n").find("meridian"), std::string::npos);
  EXPECT_EQ(error("MATMAG,3,1.05,,1.\nPMAG2D,3,3,0,3\n"), "");
  EXPECT_EQ(error("CORD2C,4,,0.,0.,0.,0.,0.,1.\n,1.,0.,0.\nMATMAG,3,1.05,,1.\n"
                  "PMAG2D,3,3,4,1\n"),
            "");
  EXPECT_NE(error("CORD2C,4,,0.,0.,0.,0.,0.,1.\n,1.,0.,0.\nMATMAG,3,1.05,,1.\n"
                  "PMAG2D,3,3,4,2\n")
                .find("meridian"),
            std::string::npos);
  // A uniform field must be axial.
  EXPECT_NE(error("MBC,1,0,UNIFORM,.1,0.,0.\n", "MAGBC = 1\n").find("along the symmetry axis"),
            std::string::npos);
  // The axis is not an MBC face: neither prescribed by an FSET nor NORMAL.
  const std::string axis_faces = "FSET,8,1,1,6\n";
  EXPECT_NE(error("MBC,1,8,NORMAL\nMBC,1,0,ZERO\n" + axis_faces, "MAGBC = 1\n")
                .find("symmetry axis"),
            std::string::npos);
  // Grids at negative radius.
  Axi left = g;
  left.r = lines(-.01, {{.02, 6}});
  auto m = parse_string(deck(left));
  ASSERT_TRUE(std::holds_alternative<Model>(m)) << error_of(m);
  EXPECT_NE(error_of(solve(std::get<Model>(m))).find("x >= 0"), std::string::npos);
  // Not yet: the tubular air gap, motion, axial periodicity and DRIVE.
  EXPECT_NE(error("MOTION,1,101,0,ROTATE,0.,1.,2\n").find("not available for MAGAXI"),
            std::string::npos);
}


TEST(AxialPeriod, OneAntiPeriodicPitchIsTwoPeriodicPitches) {
  // One pitch, ANTI, standing for four, and two pitches, PERIODIC, standing
  // for the same four, meshed alike: the same field to round-off, and the
  // same whole-machine energy and phase linkages. MFORCE is the modeled
  // part's, so the two-pitch mover's force is twice the one-pitch mover's;
  // each mover spans its model, and its force surface closes across the
  // periodic sides.
  Tube one{.offset = 5, .current = 3};
  Tube two = one;
  two.pitches = 2;
  const auto a = solved(parsed(tube(one))), b = solved(parsed(tube(two)));
  ASSERT_EQ(a.cases.size(), 1U);
  ASSERT_EQ(b.cases.size(), 1U);
  const auto &x = a.cases[0], &y = b.cases[0];
  const double scale = y.potential.cwiseAbs().maxCoeff();
  EXPECT_LT(potential_difference(a, b), 1e-10 * scale);
  // The second pitch is the first with A changed in sign.
  const int nr = 22, rows = 48;
  double second = 0;
  for (int j = 0; j <= rows; ++j)
    for (int i = 0; i < nr; ++i)
      second = std::max(second, std::abs(y.potential[b.mesh.node_index.at(1 + i + nr * j)] +
                                         y.potential[b.mesh.node_index.at(1 + i + nr * (j + rows))]));
  EXPECT_LT(second, 1e-10 * scale);
  EXPECT_LT(relative(x.energy, y.energy), 1e-10);
  EXPECT_LT(relative(x.coenergy, y.coenergy), 1e-10);
  ASSERT_EQ(x.circuits.size(), 3U);
  for (size_t k = 0; k < 3; ++k)
    EXPECT_NEAR(x.circuits[k].linkage, y.circuits[k].linkage,
                1e-10 * std::abs(y.circuits[0].linkage));
  ASSERT_EQ(x.forces.size(), 1U);
  ASSERT_EQ(y.forces.size(), 1U);
  EXPECT_GT(std::abs(x.forces[0].force.z()), 1.);
  EXPECT_NEAR(2 * x.forces[0].force.z(), y.forces[0].force.z(),
              1e-9 * std::abs(y.forces[0].force.z()));
}

TEST(AxialPeriod, AntiPeriodicPitchIsTheMirroredHalfPitch) {
  // With a magnet junction at z = 0 and no current, the stack is odd about
  // every junction (A = 0 there) and even about every magnet's middle (the
  // flux crosses it normally). The plain deck of the first half pitch, A =
  // 0 at the bottom and NORMAL at the top, is then the periodic pitch's
  // first half: the same potential at its grids, and half the pitch's
  // energy, which the periodic deck reports times the four pitches it
  // stands for. The sleeve saturates, so the two agree to MAGTOL rather
  // than round-off.
  Tube periodic{.saturating = true};
  Tube half = periodic;
  half.half = true;
  const auto p = solved(parsed(tube(periodic))), h = solved(parsed(tube(half)));
  ASSERT_EQ(p.cases.size(), 1U);
  ASSERT_EQ(h.cases.size(), 1U);
  const double scale = p.cases[0].potential.cwiseAbs().maxCoeff();
  EXPECT_LT(potential_difference(h, p), 1e-8 * scale);
  // The periodic pitch finds A = 0 on its sides, which pair with sign -1.
  const int nr = 22;
  double side = 0;
  for (int i = 0; i < nr; ++i)
    side = std::max(side, std::abs(p.cases[0].potential[p.mesh.node_index.at(1 + i)]));
  EXPECT_LT(side, 1e-8 * scale);
  EXPECT_LT(relative(p.cases[0].energy, 2 * periodic.machine * h.cases[0].energy), 1e-8);
  // The sleeve saturates: its largest |B| is past the curve's knee.
  double sleeve = 0;
  for (size_t c = 0; c < p.mesh.cells.size(); ++c)
    if (p.mesh.cells[c].material == 2)
      sleeve = std::max(sleeve, p.cases[0].b[c].norm());
  EXPECT_GT(sleeve, 1.5);
}

TEST(AxialPeriod, ForceAcrossTheSidesIsTheDerivativeOfTheCoenergy) {
  // The mover spans the pitch, so its MFORCE surface (two gap layers out)
  // ends on both periodic sides and closes through the next pitch. Its
  // axial force must be the derivative of the pitch's coenergy (a quarter
  // of the whole machine's) as the mover moves at fixed current, here by a
  // central difference over decks with the mover moved by +-dz, the gap
  // sheared. The two agree to the discretization: 3e-4 at this mesh.
  constexpr double dz = 1e-5;
  Tube t{.offset = 5, .current = 3};
  std::vector<double> coenergy;
  double force = 0;
  for (int shift : {-1, 0, 1}) {
    t.move = shift * dz;
    const auto r = solved(parsed(tube(t)));
    ASSERT_EQ(r.cases.size(), 1U);
    coenergy.push_back(r.cases[0].coenergy / t.machine);
    if (shift == 0)
      force = r.cases[0].forces[0].force.z();
  }
  const double derivative = (coenergy[2] - coenergy[0]) / (2 * dz);
  EXPECT_GT(std::abs(derivative), 1.);
  EXPECT_NEAR(force, derivative, .002 * std::abs(derivative));
}

TEST(AxialPeriod, ForceSurfaceOnASideTakesTheFieldAcrossIt) {
  // Magnets 44 rows long with 4 rows of air between them, MFORCE on the
  // magnets through one layer: the surface ends 1 row beyond each magnet.
  // The same stack is cut into a pitch two ways, one coil (16 rows) apart,
  // so no coil is cut. In the first window the magnet's surface lies on the
  // bottom side (offset -1) or the top (offset 1), and its stress there
  // takes B across the side from the image cell on the other side, mapped
  // over; in the second window those faces are inside the pitch. Both
  // windows unfold to the same discrete problem, so the force agrees to
  // round-off.
  for (int offset : {-1, 1}) {
    Tube first{.offset = offset, .spacer = 4, .tube_in_body = false, .layers = 1, .current = 3};
    Tube second = first;
    second.window = 16;
    const auto a = solved(parsed(tube(first))), b = solved(parsed(tube(second)));
    ASSERT_EQ(a.cases.size(), 1U);
    ASSERT_EQ(b.cases.size(), 1U);
    EXPECT_LT(relative(a.cases[0].energy, b.cases[0].energy), 1e-10) << offset;
    ASSERT_EQ(a.cases[0].forces.size(), 1U);
    const double force = b.cases[0].forces[0].force.z();
    EXPECT_GT(std::abs(force), .1) << offset;
    EXPECT_NEAR(a.cases[0].forces[0].force.z(), force, 1e-9 * std::abs(force)) << offset;
  }
}

TEST(AxialPeriod, CardsAndChecks) {
  const auto error = [](Axi g, const std::string &from, const std::string &to) {
    const auto at = g.cards.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    g.cards.replace(at, from.size(), to);
    auto m = parse_string(deck(g));
    if (const auto *e = std::get_if<vibestran::Error>(&m))
      return e->message;
    return error_of(solve(std::get<Model>(m)));
  };
  const auto g = tube({});
  EXPECT_NE(error(g, "MPERIOD,1,ANTI,1,2,0,4,0.02", "MPERIOD,1,ANTI,1,2,0,4")
                .find("MPERIOD 1 in a MAGAXI model needs SHIFT"),
            std::string::npos);
  EXPECT_NE(error(g, "MPERIOD,1,ANTI,1,2,0,4,0.02", "MPERIOD,1,ANTI,1,2,0,0,0.02")
                .find("requires SECTORS >= 1"),
            std::string::npos);
  EXPECT_NE(error(g, "MPERIOD,1,ANTI,1,2,0,4,0.02", "MPERIOD,1,ANTI,1,2,0,4,0.01")
                .find("moved by 0.01 lands at"),
            std::string::npos);
  EXPECT_NE(error(g, "MPERIOD,1,ANTI,1,2,0,4,0.02",
                  "MPERIOD,1,ANTI,1,2,7,4,0.02\nCORD2R,7,,0.,0.,0.,1.,0.,0.\n,0.,0.,1.")
                .find("must be parallel to the MAGAXI symmetry axis"),
            std::string::npos);
  // A uniform axial field, A = B0 r / 2, is the same at every z: periodic,
  // but not anti-periodic.
  EXPECT_NE(error(g, "MBC,1,0,ZERO", "MBC,1,0,UNIFORM,0.,0.,.1")
                .find("which MPERIOD 1 requires to be minus it"),
            std::string::npos);
  Tube two{.pitches = 2};
  EXPECT_EQ(error(tube(two), "MBC,1,0,ZERO", "MBC,1,0,UNIFORM,0.,0.,.1"), "");
  // An odd pitch count is fine for ANTI: a translation has no ring to close.
  Tube three{.machine = 3};
  const auto r = solved(parsed(tube(three)));
  std::string rows;
  for (const auto &[label, text] : r.summary.solution.rows)
    if (label == "Periodic")
      rows += text;
  EXPECT_NE(rows.find("MPERIOD 1: anti-periodic pitch of 3 along the z axis of CID 0, 0.02 "
                      "long; FSET 1 moves onto FSET 2 (22 grid pairs)"),
            std::string::npos)
      << rows;
  EXPECT_NE(rows.find("the whole machine's: 3 times the pitch's"), std::string::npos) << rows;
}
