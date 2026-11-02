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
  // Grid (r, z) is written at z + dz(r, z); properties follow the unmoved
  // cell centers.
  std::function<double(double, double)> dz = [](double, double) { return 0.; };
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
      out += std::format("GRID,{},,{:.17g},0.,{:.17g}\n", grid(i, j), g.r[i],
                         g.z[j] + g.dz(g.r[i], g.z[j]));
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

namespace {
// A tubular stack that repeats along the axis with a 20 mm pole pitch:
// axial magnets (r < 6 mm) alternating +z and -z, a +z magnet starting at z
// = offset (each magnet `spacer` rows shorter than the pitch, centred in
// it, with air between); a nonmagnetic tube to 6.75 mm; a 1 mm gap; coils
// 6.67 mm long (7.75 to 12.5 mm) in the sequence A, -C, B, -A, C, -B; a
// steel sleeve to 14 mm (13 mm, saturating, with a B-H curve); and air to
// 20 mm, held at A = 0. The deck is `pitches` pitches of the stack from z =
// window (in rows), meshed from z = 0, with an MPERIOD that moves the
// bottom (FSET 1) onto the top (FSET 2) and stands for `machine` pitches,
// ANTI for an odd count; or, with half, the plain deck of the first half
// pitch, the top NORMAL. MFORCE 1 is on the magnets (and the tube, with
// tube_in_body) through `layers` layers. Grid IDs depend only on the
// grid's ring and row, so the decks' grids correspond.
constexpr double pitch = .02;
struct Tube {
  int pitches{1}, machine{4}, rows{48}; // rows: per pitch
  int offset{0}, window{0}, spacer{0};  // in rows
  bool saturating{false}, half{false}, tube_in_body{true};
  int layers{2};
  double current{0}; // phase A; B and C carry -current / 2
  double move{0};    // the mover (r < 6.75 mm) moved along z, the gap sheared
};
Axi tube(const Tube &t) {
  Axi g;
  const double sleeve = t.saturating ? .013 : .014, row = pitch / t.rows;
  g.r = lines(0, {{.006, 6}, {.00675, 1}, {.00775, 4}, {.0125, 5}, {sleeve, 2}, {.02, 3}});
  const int rows = t.half ? t.rows / 2 : t.rows * t.pitches;
  g.z = lines(0, {{rows * row, rows}});
  const double start = t.offset * row, shift = t.window * row, gap = t.spacer * row / 2;
  g.property = [=](double r, double z) {
    const double at = z + shift; // in the stack
    if (r < .006) {
      const double within = std::fmod(at - start + 4 * pitch, pitch);
      if (within < gap || within > pitch - gap)
        return 7;
      return std::fmod(at - start + 4 * pitch, 2 * pitch) < pitch ? 4 : 6;
    }
    if (r < .00775)
      return r < .00675 ? 5 : 1;
    if (r < .0125)
      return 10 + static_cast<int>(std::floor(at / (pitch / 3))); // coil k
    return r < sleeve ? 2 : 1;
  };
  g.dz = [move = t.move](double r, double) {
    return move * std::clamp((.00775 - r) / .001, 0., 1.);
  };
  std::set<int> present;
  for (size_t j = 0; j + 1 < g.z.size(); ++j)
    for (size_t i = 0; i + 1 < g.r.size(); ++i)
      present.insert(g.property((g.r[i] + g.r[i + 1]) / 2, (g.z[j] + g.z[j + 1]) / 2));
  g.cards = "PARAM,MAGTOL,1e-11\nMATMAG,1,1.\nPMAG2D,1,1\nPMAG2D,5,1\nPMAG2D,7,1\n"
            "MATMAG,3,1.05,,1.2\nPMAG2D,4,3,0,3\nPMAG2D,6,3,0,-3\nMBC,1,0,ZERO\n";
  g.cards += t.saturating ? "MATMAG,2,,1\nPMAG2D,2,2\nTABLEBH,1\n,0.,0.,.5,60.,1.,120.,1.3,250.\n"
                            ",1.5,800.,1.6,1800.,1.7,4500.,1.8,10000.\n,1.9,20000.,2.,40000.,"
                            "2.1,80000.,ENDT\n"
                          : "MATMAG,2,1000.\nPMAG2D,2,2\n";
  // Coil k is phase A, C, B (circuits 1, 3, 2) for k mod 3 = 0, 1, 2, wound
  // +, -, + in even pitches and the other way in odd ones.
  std::array<std::string, 3> circuits;
  for (int k = 0; k < 6 * (t.pitches + 1); ++k) {
    if (!present.contains(10 + k))
      continue;
    const int sign = (k % 3 == 1 ? -1 : 1) * (k / 3 % 2 ? -1 : 1);
    g.cards += std::format("PMAG2D,{},1\nCOIL2D,{},{},{}\n", 10 + k, 200 + k, 110 + k, 88 * sign);
    circuits[static_cast<size_t>(std::array{0, 2, 1}[k % 3])] += std::format(",{},1", 200 + k);
  }
  for (size_t k = 0; k < 3; ++k)
    if (!circuits[k].empty())
      g.cards += std::format("CIRCUIT,{},1{}\n", k + 1, circuits[k]);
  g.control = "MAGBC = 1\n";
  if (t.current != 0) {
    g.control += "LOAD = 5\n";
    g.cards += std::format("ICIRC,5,1,{0:.17g}\nICIRC,5,2,{1:.17g}\nICIRC,5,3,{1:.17g}\n",
                           t.current, -t.current / 2);
  }
  if (t.half) {
    g.cards += "MBC,1,2,NORMAL\n";
    return g;
  }
  std::string body;
  for (int pid : {4, 5, 6})
    if (present.contains(pid) && (pid != 5 || t.tube_in_body))
      body += std::format(",{}", pid);
  g.cards += std::format("MPERIOD,1,{},1,2,0,{},{:.17g}\nESET,30,PROP{}\nMFORCE,1,BODY,30,0,{}\n",
                         t.pitches % 2 ? "ANTI" : "PERIODIC", t.machine / t.pitches,
                         t.pitches * pitch, body, t.layers);
  return g;
}
// The largest difference of the two results' potentials at the grids of a
// (whose IDs b shares), each of b's taken times sign.
double potential_difference(const Results &a, const Results &b, double sign = 1) {
  double out = 0;
  for (size_t i = 0; i < a.mesh.node_ids.size(); ++i)
    out = std::max(out, std::abs(a.cases[0].potential[static_cast<Eigen::Index>(i)] -
                                 sign * b.cases[0].potential[b.mesh.node_index.at(
                                            a.mesh.node_ids[i])]));
  return out;
}
} // namespace

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
