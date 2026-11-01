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
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <map>
#include <numbers>
#include <string>
#include <vector>

using namespace vibestran::magnetic;

namespace {

constexpr double pi = std::numbers::pi;


template <class T> std::string error_of(const T &result) {
  if (const auto *e = std::get_if<vibestran::Error>(&result))
    return e->message;
  return {};
}

// Grid lines from start through each (end, divisions) segment, uniform in
// each, so that material boundaries fall on lines.
std::vector<double> lines(double start, std::vector<std::pair<double, int>> segments) {
  std::vector<double> out{start};
  for (auto [end, n] : segments) {
    const double from = out.back();
    for (int k = 1; k <= n; ++k)
      out.push_back(from + (end - from) * k / n);
  }
  return out;
}

// A structured mesh of the r-z rectangle spanned by r and z, in the basic x-z
// plane (x = r), each cell's property from its center. Grid (i, j) is
// 1 + i + nr j; triangles split each quad in two. Exterior faces are FSETs:
// 1 at the bottom (first z), 2 at the top, 3 at the outer radius.
struct Axi {
  std::vector<double> r, z;
  std::function<int(double, double)> property = [](double, double) { return 1; };
  bool triangles{false};
  std::string control; // case control after CEND
  std::string cards;   // bulk data
};
std::string deck(const Axi &g) {
  const int nr = static_cast<int>(g.r.size()), nz = static_cast<int>(g.z.size());
  const auto grid = [nr](int i, int j) { return 1 + i + nr * j; };
  std::string out = "SOL MAGSTAT\nCEND\n" + g.control + "BEGIN BULK\nMAGAXI,0\n" + g.cards;
  for (int j = 0; j < nz; ++j)
    for (int i = 0; i < nr; ++i)
      out += std::format("GRID,{},,{:.17g},0.,{:.17g}\n", grid(i, j), g.r[i], g.z[j]);
  std::map<int, std::vector<int>> by_property;
  int id = 1;
  for (int j = 0; j + 1 < nz; ++j)
    for (int i = 0; i + 1 < nr; ++i) {
      const int pid = g.property((g.r[i] + g.r[i + 1]) / 2, (g.z[j] + g.z[j + 1]) / 2);
      const int a = grid(i, j), b = grid(i + 1, j), c = grid(i + 1, j + 1), d = grid(i, j + 1);
      const int first = id;
      if (g.triangles) {
        out += std::format("CTRIA3,{},{},{},{},{}\nCTRIA3,{},{},{},{},{}\n", id, pid, a, b,
                           c, id + 1, pid, a, c, d);
        id += 2;
      } else {
        out += std::format("CQUAD4,{},{},{},{},{},{}\n", id, pid, a, b, c, d);
        ++id;
      }
      for (int e = first; e < id; ++e)
        by_property[pid].push_back(e);
      // The element owning each exterior face (for triangles, the one with
      // that side).
      if (j == 0)
        out += std::format("FSET,1,{},{},{}\n", first, a, b);
      if (j + 2 == nz)
        out += std::format("FSET,2,{},{},{}\n", g.triangles ? first + 1 : first, d, c);
      if (i + 2 == nr)
        out += std::format("FSET,3,{},{},{}\n", first, b, c);
    }
  for (const auto &[pid, ids] : by_property) {
    out += std::format("ESET,{}", 100 + pid);
    for (int e : ids)
      out += std::format(",{}", e);
    out += "\n";
  }
  return out + "ENDDATA\n";
}
Model parsed(const Axi &g) {
  auto m = parse_string(deck(g));
  EXPECT_TRUE(std::holds_alternative<Model>(m)) << error_of(m);
  return std::holds_alternative<Model>(m) ? std::get<Model>(std::move(m)) : Model{};
}
Results solved(const Model &m) {
  auto r = solve(m);
  EXPECT_TRUE(std::holds_alternative<Results>(r)) << error_of(r);
  return std::holds_alternative<Results>(r) ? std::get<Results>(std::move(r)) : Results{};
}
double relative(double a, double b) { return std::abs(a - b) / std::abs(b); }

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
