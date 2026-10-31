// tests/unit/test_air_gap.cpp
// Air-gap element (AGE) for planar MAGSTAT: gap detection, the harmonic
// transforms and stiffness, and solutions against analytic fields and a
// meshed gap.

#include "magnetic/air_gap.hpp"
#include "magnetic/solver.hpp"
#ifdef VIBESTRAN_HAS_HDF5
#include "../common/hdf5_results.hpp"
#endif
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <map>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <format>
#include <numbers>
#include <string>
#include <vector>

using namespace vibestran::magnetic;
using vibestran::Error;

namespace {

constexpr double pi = std::numbers::pi;

std::string error_of(const auto &result) {
  if (const auto *e = std::get_if<Error>(&result))
    return e->message;
  return {};
}

// A rotor disk and a stator ring in the z = 0 plane, as rings of grids.
// gap_rings = 0 leaves the annulus between rotor and gap_outer unmeshed;
// otherwise it is meshed with that many layers of rotor_div quads (and the
// stator then also has rotor_div divisions).
struct Machine {
  double rotor{.02}, gap_outer{.021}, stator{.04};
  int rotor_div{48}, stator_div{48}, rotor_rings{6}, stator_rings{8}, gap_rings{0};
  double stator_phase{0}; // fraction of a stator division
  double jitter{0};       // uneven angular spacing, fraction of a division
  double rotation{0};     // rigid rotation of everything about the center
  Vector center{Vector::Zero()};
  int rotor_property{1}, stator_property{1};
};

Model machine(const Machine &g) {
  Model m;
  m.options.planar = true;
  m.options.depth = .1;
  m.materials[1] = {};
  m.properties[1] = {1};
  m.subcases.push_back({});
  int next = 1;
  auto ring = [&](double r, int n, double phase) {
    std::vector<int> ids;
    for (int k = 0; k < n; ++k) {
      const double t =
          2 * pi * (k + phase + g.jitter * std::sin(1.7 * k + 3 * r / g.stator)) / n +
          g.rotation;
      m.nodes[next] = {g.center + Vector(r * std::cos(t), r * std::sin(t), 0), 0};
      ids.push_back(next++);
    }
    return ids;
  };
  auto quads = [&](const std::vector<int> &a, const std::vector<int> &b, int property) {
    const int n = static_cast<int>(a.size());
    for (int k = 0; k < n; ++k)
      m.elements.push_back({static_cast<int>(m.elements.size()) + 1, property,
                            {a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]}});
  };
  const int center = next;
  m.nodes[next++] = {g.center, 0};
  std::vector<int> previous = ring(g.rotor / g.rotor_rings, g.rotor_div, 0);
  for (int k = 0; k < g.rotor_div; ++k)
    m.elements.push_back({static_cast<int>(m.elements.size()) + 1, g.rotor_property,
                          {center, previous[k], previous[(k + 1) % g.rotor_div]}});
  for (int j = 2; j <= g.rotor_rings; ++j) {
    auto current = ring(g.rotor * j / g.rotor_rings, g.rotor_div, 0);
    quads(previous, current, g.rotor_property);
    previous = std::move(current);
  }
  const int stator_div = g.gap_rings ? g.rotor_div : g.stator_div;
  for (int j = 1; j <= g.gap_rings; ++j) {
    auto current = ring(g.rotor + (g.gap_outer - g.rotor) * j / g.gap_rings,
                        g.rotor_div, 0);
    quads(previous, current, 1);
    previous = std::move(current);
  }
  if (!g.gap_rings)
    previous = ring(g.gap_outer, stator_div, g.stator_phase);
  for (int j = 1; j <= g.stator_rings; ++j) {
    auto current = ring(g.gap_outer + (g.stator - g.gap_outer) * j / g.stator_rings,
                        stator_div, g.gap_rings ? 0 : g.stator_phase);
    quads(previous, current, g.stator_property);
    previous = std::move(current);
  }
  return m;
}

// Uniform remanence along x in the rotor (mu_r = 1), and the analytic field
// with A = 0 at the stator's outer radius R: B = (Br/2)(1 - a^2/R^2) x
// inside, and A = D (r - R^2/r) sin(theta), D = -Br a^2 / (2 R^2), outside.
void magnetize_rotor(Model &m, double br, double angle = 0) {
  m.coordinates[1] = {0, 'R', Vector::Zero(), Vector::UnitZ(),
                      Vector(std::cos(angle), std::sin(angle), 0)};
  m.materials[2] = {1, 0, br};
  m.properties[2] = {2, 1, 1, true};
}
Vector magnet_field_outside(const Machine &g, double br, const Vector &x) {
  const double a = g.rotor, big = g.stator, d = -br * a * a / (2 * big * big);
  const Vector p = x - g.center;
  const double r = p.norm(), t = std::atan2(p.y(), p.x());
  const double br_r = d * (1 - big * big / (r * r)) * std::cos(t),
               bt = -d * (1 + big * big / (r * r)) * std::sin(t);
  return {br_r * std::cos(t) - bt * std::sin(t), br_r * std::sin(t) + bt * std::cos(t), 0};
}

// A gap built directly: circles of n_inner and n_outer grids (indices
// 0.. and n_inner..) about the origin, smoothly graded so that the spacing
// varies by 30% around each circle.
AirGap direct_gap(double r1, double r2, int n_inner, int n_outer) {
  AirGap g;
  auto circle = [](double r, int n, int first, double phase) {
    GapCircle c;
    c.radius = r;
    for (int k = 0; k < n; ++k) {
      c.nodes.push_back(first + k);
      const double s = 2 * pi * (k + phase) / n;
      c.angles.push_back(-pi + s + .3 * std::sin(s));
    }
    return c;
  };
  g.inner = circle(r1, n_inner, 0, .1);
  g.outer = circle(r2, n_outer, n_inner, .45);
  g.harmonics = std::min(n_inner, n_outer) / 2;
  return g;
}

// Nodal values of A at the gap grids for a potential function of x.
Eigen::VectorXd sample(const AirGap &g, auto potential) {
  Eigen::VectorXd a(static_cast<Eigen::Index>(g.inner.nodes.size() + g.outer.nodes.size()));
  for (const GapCircle *c : {&g.inner, &g.outer})
    for (size_t k = 0; k < c->nodes.size(); ++k)
      a[c->nodes[k]] = potential(
          Vector(c->radius * std::cos(c->angles[k]), c->radius * std::sin(c->angles[k]), 0));
  return a;
}

Mesh mesh_of(const Model &m) {
  auto built = make_mesh(m);
  EXPECT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
  return std::holds_alternative<Mesh>(built) ? std::get<Mesh>(std::move(built)) : Mesh{};
}

} // namespace

TEST(AirGapStiffness, TransformIsTheFourierSeriesOfTheInterpolant) {
  const AirGap g = direct_gap(1, 1.1, 13, 17);
  const auto &c = g.inner;
  const int harmonics = 6;
  const Eigen::MatrixXd t = gap_transform(c, harmonics);
  Eigen::VectorXd values(13);
  for (int k = 0; k < 13; ++k)
    values[k] = std::cos(2.1 * k) + .3 * k;
  const Eigen::VectorXd coefficients = t * values;
  // The piecewise-linear interpolant in angle, integrated by the midpoint
  // rule on a fine grid over one period from the first grid.
  auto f = [&](double angle) {
    double x = angle;
    while (x >= c.angles[0] + 2 * pi)
      x -= 2 * pi;
    size_t k = 0;
    while (k + 1 < c.angles.size() && c.angles[k + 1] <= x)
      ++k;
    const size_t next = (k + 1) % c.angles.size();
    const double w = next ? c.angles[next] : c.angles[0] + 2 * pi;
    const double s = (x - c.angles[k]) / (w - c.angles[k]);
    return (1 - s) * values[static_cast<Eigen::Index>(k)] +
           s * values[static_cast<Eigen::Index>(next)];
  };
  const int steps = 400000;
  Eigen::VectorXd expected = Eigen::VectorXd::Zero(1 + 2 * harmonics);
  for (int i = 0; i < steps; ++i) {
    const double x = c.angles[0] + 2 * pi * (i + .5) / steps, w = 2 * pi / steps;
    expected[0] += f(x) * w / (2 * pi);
    for (int m = 1; m <= harmonics; ++m) {
      expected[2 * m - 1] += f(x) * std::cos(m * x) * w / pi;
      expected[2 * m] += f(x) * std::sin(m * x) * w / pi;
    }
  }
  EXPECT_LT((coefficients - expected).norm(), 1e-8 * expected.norm())
      << coefficients.transpose() << "\n" << expected.transpose();
}

TEST(AirGapStiffness, SymmetricWithConstantNullSpaceAndExactMeanTerm) {
  const double r1 = .02, r2 = .021;
  const AirGap g = direct_gap(r1, r2, 40, 57);
  const Eigen::MatrixXd k = air_gap_stiffness(g);
  ASSERT_EQ(k.rows(), 97);
  EXPECT_LT((k - k.transpose()).norm(), 1e-12 * k.norm());
  // A constant potential carries no field.
  EXPECT_LT((k * Eigen::VectorXd::Ones(97)).norm(), 1e-10 * k.norm());
  const Eigen::VectorXd eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(k).eigenvalues();
  EXPECT_GT(eigenvalues.minCoeff(), -1e-10 * eigenvalues.maxCoeff());
  // A different constant on each circle is a logarithmic field, represented
  // exactly: W = pi (alpha - beta)^2 / (mu0 ln(r2/r1)).
  Eigen::VectorXd a(97);
  a.head(40).setConstant(.7);
  a.tail(57).setConstant(-.2);
  const double expected = pi * .81 / (mu0 * std::log(r2 / r1));
  EXPECT_NEAR(.5 * a.dot(k * a), expected, 1e-12 * expected);
}

TEST(AirGapStiffness, UniformFieldEnergyAndGapFieldConvergeAtSecondOrder) {
  // A = B0 . (y, -x) gives B = (B0, 0.4 B0); its energy in the annulus is
  // |B|^2 / (2 mu0) * pi (r2^2 - r1^2). The traces are interpolated
  // piecewise-linearly, so the AGE converges at O(h^2). The constant is
  // large: with different divisions on the two circles their interpolation
  // errors differ, and a thin gap's energy depends on the difference of the
  // traces, amplifying it by about r / gap = 20. (In a solve the meshed
  // regions adjust the traces instead; see the magnetized rotor test.)
  const double r1 = .02, r2 = .021, b0 = .8;
  const Vector field(b0, .4 * b0, 0);
  const double exact = field.squaredNorm() / (2 * mu0) * pi * (r2 * r2 - r1 * r1);
  double previous_energy = 1, previous_field = 1;
  for (int n : {32, 64, 128}) {
    const AirGap g = direct_gap(r1, r2, n, n * 3 / 2);
    const Eigen::VectorXd a =
        sample(g, [&](const Vector &x) { return field.x() * x.y() - field.y() * x.x(); });
    const double energy = .5 * a.dot(air_gap_stiffness(g) * a);
    double field_error = 0;
    // Inside the gap: on the circles the tangential derivative of a
    // piecewise-linear trace is only first order.
    for (double t : {0., 1., 2.5, 4.})
      for (double r : {r1 + .25 * (r2 - r1), (r1 + r2) / 2, r1 + .75 * (r2 - r1)})
        field_error = std::max(
            field_error,
            (air_gap_field(g, a, Vector(r * std::cos(t), r * std::sin(t), 0)) - field).norm() /
                field.norm());
    const double energy_error = std::abs(energy - exact) / exact;
    EXPECT_LT(energy_error, previous_energy / 3.5) << n;
    EXPECT_LT(field_error, previous_field / 3.5) << n;
    previous_energy = energy_error;
    previous_field = field_error;
  }
  EXPECT_LT(previous_energy, 3e-3);
  EXPECT_LT(previous_field, 4e-3);
}

TEST(AirGapDetection, FindsTheAnnulusOnUnevenNonconformingCircles) {
  Machine g;
  g.center = Vector(.3, -.2, 0);
  g.rotor_div = 48;
  g.stator_div = 72;
  g.stator_phase = .37;
  g.jitter = .2;
  const Mesh mesh = mesh_of(machine(g));
  ASSERT_TRUE(mesh.air_gap.has_value());
  const auto &gap = *mesh.air_gap;
  EXPECT_LT((gap.center - g.center).norm(), 1e-12);
  EXPECT_NEAR(gap.inner.radius, g.rotor, 1e-14);
  EXPECT_NEAR(gap.outer.radius, g.gap_outer, 1e-14);
  EXPECT_EQ(gap.inner.nodes.size(), 48U);
  EXPECT_EQ(gap.outer.nodes.size(), 72U);
  EXPECT_EQ(gap.faces.size(), 120U);
  EXPECT_EQ(gap.harmonics, 24);
  EXPECT_LT(gap.deviation, 1e-12);
  for (const GapCircle *c : {&gap.inner, &gap.outer}) {
    EXPECT_TRUE(std::is_sorted(c->angles.begin(), c->angles.end()));
    EXPECT_LT(c->angles.back(), c->angles.front() + 2 * pi);
    for (size_t k = 0; k < c->nodes.size(); ++k) {
      const Vector d = mesh.nodes[c->nodes[k]] - gap.center;
      EXPECT_NEAR(std::atan2(d.dot(gap.v), d.dot(gap.u)), c->angles[k], 1e-12);
    }
  }
  // Gap grids are unknowns; the stator's outer boundary stays prescribed.
  auto gauged = make_gauge(mesh);
  ASSERT_TRUE(std::holds_alternative<Gauge>(gauged)) << error_of(gauged);
  const auto &gauge = std::get<Gauge>(gauged);
  for (const GapCircle *c : {&gap.inner, &gap.outer})
    for (int v : c->nodes)
      EXPECT_GE(gauge.equations[v], 0);
  int prescribed = 0;
  for (size_t v = 0; v < mesh.nodes.size(); ++v)
    if (gauge.equations[v] < 0) {
      ++prescribed;
      EXPECT_NEAR((mesh.nodes[v] - g.center).norm(), g.stator, 1e-12);
    }
  EXPECT_EQ(prescribed, 72);

  // Meshing the gap joins the parts: there is nothing to detect.
  g.gap_rings = 2;
  EXPECT_FALSE(mesh_of(machine(g)).air_gap.has_value());
}

TEST(AirGapDetection, ErrorsNameTheGridsAndRadiiThatFailed) {
  Machine g;
  auto message = [](const Model &m) { return error_of(make_mesh(m)); };

  // One rotor grid off the circle by 1e-4 of the radius.
  auto off = machine(g);
  const int moved = static_cast<int>(off.nodes.size()) - 8 * g.stator_div - 3;
  off.nodes.at(moved).position *= 1 + 1e-4;
  auto text = message(off);
  EXPECT_NE(text.find("no air gap joins them"), std::string::npos) << text;
  // The least-squares circle absorbs a little of the offset.
  EXPECT_NE(text.find("deviates from a circle by 9."), std::string::npos) << text;
  EXPECT_NE(text.find(std::format("e-05 of its radius at grid {}", moved)),
            std::string::npos)
      << text;

  // The stator shifted off the rotor's center.
  auto shifted = machine(g);
  for (auto &[id, grid] : shifted.nodes)
    if (grid.position.norm() > g.rotor * 1.01)
      grid.position += Vector(1e-5, 0, 0);
  text = message(shifted);
  EXPECT_NE(text.find("are not concentric: their centers are 1.000e-05 apart"),
            std::string::npos)
      << text;

  // An island inside the annulus.
  auto island = machine(g);
  const int base = 100000;
  const double r = (g.rotor + g.gap_outer) / 2;
  island.nodes[base] = {Vector(r, 0, 0), 0};
  island.nodes[base + 1] = {Vector(r, 1e-4, 0), 0};
  island.nodes[base + 2] = {Vector(r - 1e-4, 0, 0), 0};
  island.elements.push_back({90000, 1, {base, base + 2, base + 1}});
  text = message(island);
  EXPECT_NE(text.find(std::format("grid {} lies inside the annulus between the circles "
                                  "of radius 0.02 and 0.021",
                                  base)),
            std::string::npos)
      << text;

  // Two gaps: a ring between rotor and stator.
  g.stator = .03;
  auto ringed = machine(g);
  {
    // The outer ring, 0.032 to 0.04, cut from a second machine.
    Machine big;
    big.gap_outer = .032;
    big.stator = .04;
    big.stator_div = 60;
    auto shell = machine(big);
    std::map<int, int> renumber;
    for (const auto &[id, grid] : shell.nodes)
      if (grid.position.norm() > .0315)
        ringed.nodes[renumber[id] = 200000 + id] = grid;
    for (const auto &e : shell.elements) {
      if (!std::all_of(e.nodes.begin(), e.nodes.end(),
                       [&](int v) { return renumber.contains(v); }))
        continue;
      Element copy = e;
      copy.id += 200000;
      std::transform(copy.nodes.begin(), copy.nodes.end(), copy.nodes.begin(),
                     [&](int v) { return renumber.at(v); });
      ringed.elements.push_back(copy);
    }
  }
  text = message(ringed);
  EXPECT_NE(text.find("has 2 candidate air gaps (radii 0.02 to 0.021; 0.03 to 0.032)"),
            std::string::npos)
      << text;

  // A third part that no gap reaches.
  auto stray = machine(Machine{});
  for (int k = 0; k < 3; ++k)
    stray.nodes[base + k] = {Vector(.1 + .01 * (k == 1), .1 + .01 * (k == 2), 0), 0};
  stray.elements.push_back({90000, 1, {base, base + 1, base + 2}});
  text = message(stray);
  EXPECT_NE(text.find("Element 90000 is not joined to the rest of the magnetic mesh"),
            std::string::npos)
      << text;
}

TEST(AirGapSolve, MagnetizedRotorMatchesTheAnalyticFieldAtSecondOrder) {
  // Nonconforming circles (the stator has 1.5 times the rotor's divisions,
  // offset by a third of a division): the AGE couples them without error of
  // its own, so the polygonal magnet boundary sets the O(h^2) error.
  const double br = 1.0;
  double previous_gap = 1, previous_inside = 1;
  for (int div : {24, 48, 96}) {
    SCOPED_TRACE(div);
    Machine g;
    g.rotor_div = div;
    g.stator_div = div * 3 / 2;
    g.stator_phase = 1. / 3;
    g.rotor_rings = div / 8;
    g.stator_rings = div / 6;
    g.rotor_property = 2;
    auto m = machine(g);
    magnetize_rotor(m, br);
    const double gap = g.gap_outer - g.rotor;
    std::vector<Vector> in_gap;
    for (double t : {.3, 2.2, 4.1})
      for (double depth : {.1, .5, .9}) {
        const double r = g.rotor + depth * gap;
        in_gap.emplace_back(r * std::cos(t), r * std::sin(t), 0);
      }
    m.probes.clear();
    for (size_t i = 0; i < in_gap.size(); ++i)
      m.probes.push_back({static_cast<int>(i) + 1, in_gap[i]});
    m.probes.push_back({100, Vector(.005, .003, 0)});
    auto solved = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    const auto &c = std::get<Results>(solved).cases[0];
    double gap_error = 0;
    for (size_t i = 0; i < in_gap.size(); ++i) {
      const Vector expected = magnet_field_outside(g, br, in_gap[i]);
      gap_error = std::max(gap_error, (c.probe_b[i] - expected).norm() / expected.norm());
      EXPECT_LT((c.probe_h[i] - c.probe_b[i] / mu0).norm(), 1e-9 * c.probe_h[i].norm());
    }
    const Vector inside(br / 2 * (1 - g.rotor * g.rotor / (g.stator * g.stator)), 0, 0);
    const double inside_error = (c.probe_b.back() - inside).norm() / inside.norm();
    EXPECT_LT(gap_error, previous_gap / 3.5);
    EXPECT_LT(inside_error, previous_inside / 3.5);
    previous_gap = gap_error;
    previous_inside = inside_error;
    const auto &rows = std::get<Results>(solved).summary.solution.rows;
    EXPECT_TRUE(std::any_of(rows.begin(), rows.end(), [&](const auto &row) {
      return row.first == "Air gap" &&
             row.second.find(std::format("{} + {} grids, {} harmonics", div, div * 3 / 2,
                                         div / 2)) != std::string::npos;
    }));
  }
  EXPECT_LT(previous_gap, 1e-3);
  EXPECT_LT(previous_inside, 5e-4);
}

TEST(AirGapSolve, AgreesWithAMeshedGapAroundAnIronStator) {
  // The same machine with the gap meshed by four layers of air: both
  // discretizations approach one solution, so the energy difference
  // shrinks at O(h^2) and the gap field agrees.
  double previous = 1, previous_field = 1;
  for (int div : {48, 96, 192}) {
    SCOPED_TRACE(div);
    Machine g;
    g.rotor_div = g.stator_div = div;
    g.rotor_rings = div / 8;
    g.stator_rings = div / 6;
    g.rotor_property = 2;
    g.stator_property = 3;
    const double r = (g.rotor + g.gap_outer) / 2;
    const Vector probe(r * std::cos(.7), r * std::sin(.7), 0);
    double energy[2];
    Vector field[2];
    for (int meshed : {0, 1}) {
      g.gap_rings = meshed ? 4 : 0;
      auto m = machine(g);
      magnetize_rotor(m, 1.0);
      m.materials[3] = {1000};
      m.properties[3] = {3};
      m.probes = {{1, probe}};
      auto solved = solve(m);
      ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
      EXPECT_EQ(std::get<Results>(solved).mesh.air_gap.has_value(), !meshed);
      energy[meshed] = std::get<Results>(solved).cases[0].energy;
      field[meshed] = std::get<Results>(solved).cases[0].probe_b[0];
    }
    const double difference = std::abs(energy[0] - energy[1]) / energy[1];
    EXPECT_LT(difference, previous / 3.4);
    previous = difference;
    // The meshed gap's B is the gradient of a piecewise-linear potential,
    // first-order accurate, so the fields agree at first order.
    const double field_difference = (field[0] - field[1]).norm() / field[1].norm();
    EXPECT_LT(field_difference, previous_field / 1.4);
    previous_field = field_difference;
  }
  EXPECT_LT(previous, 2e-4);
  EXPECT_LT(previous_field, 1e-2);
}

TEST(AirGapSolve, RigidRotationLeavesEnergyAndRotatedFieldsUnchanged) {
  // Rotating the whole model, magnetization included, rotates every
  // harmonic of the traces exactly, so the solution is the same to round-off.
  Machine g;
  g.rotor_div = 40;
  g.stator_div = 54;
  g.stator_phase = .2;
  g.jitter = .15;
  g.rotor_property = 2;
  g.stator_property = 3;
  const double angle = .4;
  const double r = (g.rotor + g.gap_outer) / 2;
  const Vector probe(r * std::cos(1.1), r * std::sin(1.1), 0);
  const Eigen::Matrix3d rotate =
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Results results[2];
  for (int turned : {0, 1}) {
    Machine h = g;
    h.rotation = turned ? angle : 0;
    auto m = machine(h);
    magnetize_rotor(m, 1.0, h.rotation);
    m.materials[3] = {500};
    m.properties[3] = {3};
    m.probes = {{1, turned ? Vector(rotate * probe) : probe}};
    auto solved = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    results[turned] = std::get<Results>(std::move(solved));
  }
  const auto &a = results[0].cases[0], &b = results[1].cases[0];
  EXPECT_NEAR(a.energy, b.energy, 1e-10 * a.energy);
  EXPECT_LT((rotate * a.probe_b[0] - b.probe_b[0]).norm(), 1e-9 * a.probe_b[0].norm());
  for (size_t i = 0; i < a.b.size(); ++i)
    EXPECT_LT((rotate * a.b[i] - b.b[i]).norm(), 1e-9 * (a.b[i].norm() + 1e-3));
}

TEST(AirGapSolve, ExteriorBoundaryConditionSkipsTheGap) {
  // All air in a uniform applied field: the field inside the rotor is the
  // applied field, up to the O(h^2) interpolation of the gap traces.
  Machine g;
  g.rotor_div = 64;
  g.stator_div = 80;
  auto m = machine(g);
  const Vector applied(.3, -.1, 0);
  m.boundaries[1] = {0, applied};
  m.subcases[0].boundary = 1;
  m.probes = {{1, Vector(.004, -.006, 0)},
              {2, Vector((g.rotor + g.gap_outer) / 2, 0, 0)}};
  auto whole = solve(m);
  ASSERT_TRUE(std::holds_alternative<Results>(whole)) << error_of(whole);
  const auto &c = std::get<Results>(whole).cases[0];
  EXPECT_LT((c.probe_b[0] - applied).norm(), 2e-3 * applied.norm());
  EXPECT_LT((c.probe_b[1] - applied).norm(), 2e-3 * applied.norm());

  // An explicit FSET of the outer boundary is the same condition.
  const Mesh mesh = mesh_of(m);
  ASSERT_TRUE(mesh.air_gap.has_value());
  std::vector<bool> on_gap(mesh.faces.size(), false);
  for (int f : mesh.air_gap->faces)
    on_gap[f] = true;
  std::vector<FaceSelection> outer, with_gap_face;
  for (size_t f = 0; f < mesh.faces.size(); ++f) {
    const auto &face = mesh.faces[f];
    if (face.cells[1] >= 0)
      continue;
    const FaceSelection s{mesh.cells[face.cells[0]].id,
                          {mesh.node_ids[face.nodes[0]], mesh.node_ids[face.nodes[1]]}};
    if (!on_gap[f])
      outer.push_back(s);
    else if (with_gap_face.empty())
      with_gap_face.push_back(s);
  }
  ASSERT_EQ(outer.size(), 80U);
  m.face_sets[7] = outer;
  m.boundaries[1].faces = 7;
  auto explicit_set = solve(m);
  ASSERT_TRUE(std::holds_alternative<Results>(explicit_set)) << error_of(explicit_set);
  EXPECT_LT((std::get<Results>(explicit_set).cases[0].potential - c.potential).norm(),
            1e-12 * c.potential.norm());

  // The gap takes no boundary condition.
  m.face_sets[7].push_back(with_gap_face.front());
  const auto text = error_of(solve(m));
  EXPECT_NE(text.find("on the air gap; the air gap is coupled by the air-gap element"),
            std::string::npos)
      << text;
}

TEST(AirGapSolve, SaturatedIronAndWindingMatchAMeshedGap) {
  // Iron rotor and stator (saturating B-H curve) with a two-slot winding:
  // Newton runs through the AGE's constant block, and the circuit's linkage
  // and differential inductance agree with the meshed-gap model.
  const int div = 96;
  Results results[2];
  for (int meshed : {0, 1}) {
    Machine g;
    g.rotor_div = g.stator_div = div;
    g.rotor_rings = 12;
    g.stator_rings = 16;
    g.gap_rings = meshed ? 4 : 0;
    g.rotor_property = g.stator_property = 2;
    auto m = machine(g);
    m.materials[2] = {1, 1};
    m.properties[2] = {2};
    m.bh_tables[1] = {{0, 0}, {.1, 100}, {.3, 2000}, {1, 100000}};
    m.options.nonlinear_tolerance = 1e-11;
    for (auto &e : m.elements) {
      Vector center = Vector::Zero();
      for (int v : e.nodes)
        center += m.nodes.at(v).position / static_cast<double>(e.nodes.size());
      const double r = center.norm(), t = std::atan2(center.y(), center.x());
      if (r < .026 || r > .034)
        continue;
      if (std::abs(t) < .35 || std::abs(std::abs(t) - pi) < .35) {
        e.property = 1;
        m.element_sets[std::abs(t) < .35 ? 1 : 2].push_back(e.id);
      }
    }
    m.planar_coils = {{11, 1, 100}, {12, 2, -100}};
    m.circuits[1] = {1, 1, {{11, 1}, {12, 1}}};
    m.currents = {{1, 1, 30}};
    m.subcases = {{.id = 1, .load = 1}};
    auto solved = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    results[meshed] = std::get<Results>(std::move(solved));
  }
  const auto &age = results[0].cases[0], &mesh = results[1].cases[0];
  // Past the end of the B-H table: well into saturation.
  EXPECT_TRUE(age.extrapolated);
  EXPECT_GT(age.iterations, 3);
  EXPECT_LE(age.residual, 1e-11);
  EXPECT_LE(age.iterations, mesh.iterations + 2);
  EXPECT_NEAR(age.energy, mesh.energy, 2e-4 * mesh.energy);
  ASSERT_EQ(age.circuits.size(), 1U);
  EXPECT_NEAR(age.circuits[0].linkage, mesh.circuits[0].linkage,
              2e-4 * std::abs(mesh.circuits[0].linkage));
  EXPECT_NEAR(age.circuit_inductance(0, 0), mesh.circuit_inductance(0, 0),
              5e-4 * mesh.circuit_inductance(0, 0));
}

TEST(AirGapSolve, ForceSurfacesMayNotTouchTheGap) {
  Machine g;
  auto m = machine(g);
  const Mesh mesh = mesh_of(m);
  ASSERT_TRUE(mesh.air_gap.has_value());
  // The body around the rotor's elements reaches the gap at once.
  for (const auto &e : m.elements) {
    Vector center = Vector::Zero();
    for (int v : e.nodes)
      center += m.nodes.at(v).position / static_cast<double>(e.nodes.size());
    if (center.norm() < g.rotor)
      m.element_sets[5].push_back(e.id);
  }
  m.forces = {{1, 5, true}};
  auto text = error_of(solve(m));
  EXPECT_NE(text.find("MFORCE 1: the surface around body ESET 5 reaches the air gap"),
            std::string::npos)
      << text;
  // An explicit surface on the gap's faces.
  std::vector<FaceSelection> faces;
  for (int f : mesh.air_gap->faces) {
    const auto &face = mesh.faces[f];
    faces.push_back({mesh.cells[face.cells[0]].id,
                     {mesh.node_ids[face.nodes[0]], mesh.node_ids[face.nodes[1]]}});
  }
  m.face_sets[6] = faces;
  m.forces = {{2, 6, false}};
  text = error_of(solve(m));
  EXPECT_NE(text.find("lies on the air gap; force surfaces may not touch the air gap"),
            std::string::npos)
      << text;
}

TEST(AirGapSolve, ResultsRecordTheGap) {
  Machine g;
  g.rotor_div = 40;
  g.stator_div = 60;
  g.rotor_property = 2;
  auto m = machine(g);
  magnetize_rotor(m, 1.0);
  auto solved = solve(m);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &results = std::get<Results>(solved);
  const auto &gap = *results.mesh.air_gap;
#ifdef VIBESTRAN_HAS_HDF5
  const auto directory = std::filesystem::temp_directory_path() / "vibestran_air_gap";
  std::filesystem::create_directories(directory);
  ASSERT_FALSE(write_results(m, results, directory / "gap.F06").has_value());
  const auto file = directory / "gap.mag.vtkhdf";
  EXPECT_NEAR(vibestran::test::h5_attribute(file, "/MAGSTAT/air_gap", "inner_radius_m"),
              g.rotor, 1e-15);
  EXPECT_NEAR(vibestran::test::h5_attribute(file, "/MAGSTAT/air_gap", "outer_radius_m"),
              g.gap_outer, 1e-15);
  EXPECT_EQ(vibestran::test::h5_attribute(file, "/MAGSTAT/air_gap", "harmonics"), 20);
  // Grid IDs in angle order: the deck's own IDs, so a user can find them.
  const auto inner = vibestran::test::h5_table(file, "/MAGSTAT/air_gap/inner_grids");
  const auto outer = vibestran::test::h5_table(file, "/MAGSTAT/air_gap/outer_grids");
  ASSERT_EQ(inner.values.size(), 40U);
  ASSERT_EQ(outer.values.size(), 60U);
  for (size_t k = 0; k < inner.values.size(); ++k)
    EXPECT_EQ(inner.values[k], results.mesh.node_ids[gap.inner.nodes[k]]);
  for (double id : outer.values)
    EXPECT_NEAR(m.nodes.at(static_cast<int>(id)).position.norm(), g.gap_outer, 1e-15);
  std::ifstream in(directory / "gap.F06");
  std::stringstream f06;
  f06 << in.rdbuf();
  EXPECT_NE(f06.str().find("   Air gap .................. radius 0.02 to 0.021 m about "
                           "(0, 0, 0), 40 + 60 grids, 20 harmonics"),
            std::string::npos)
      << f06.str();
  std::filesystem::remove_all(directory);
#else
  GTEST_SKIP() << "MAGSTAT results need HDF5";
#endif
}
