// tests/unit/test_motion.cpp
// MOTION for planar MAGSTAT: rigid rotation of a rotor across the air-gap
// element. A position must be the static deck rotated to it; the air-gap
// torque must be the derivative of the coenergy and match a magnet in a
// uniform field; the cards, checks and results are those documented.

#include "magnetic/air_gap.hpp"
#include "magnetic/motion.hpp"
#include "magnetic/solver.hpp"
#include "machine_deck.hpp"
#ifdef VIBESTRAN_HAS_HDF5
#include "../common/hdf5_results.hpp"
#endif
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

using namespace vibestran::magnetic;
using namespace machine_deck;

namespace {

constexpr double degree = pi / 180;

// The rotor (PID 2) as ESET 20, turned by MOTION 1 about basic z.
std::string rotor_motion(double start, double step, int count) {
  return std::format("ESET,20,PROP,2\nMOTION,1,20,0,ROTATE,{:.17g},{:.17g},{}\n",
                     start, step, count);
}
// Iron stator (PID 3), linear or on the iron curve.
std::string stator_iron(bool saturating) {
  return saturating ? iron_table + "MATMAG,3,,1\nPMAG2D,3,3\n"
                    : "MATMAG,3,500.\nPMAG2D,3,3\n";
}
// Probes in the stator iron, in the gap and in the rotor: fixed points.
std::string probes() {
  return "MPROBE,1,0,.03,.004,0.\nMPROBE,2,0,-.0063,.0195,0.\n"
         "MPROBE,3,0,.008,-.011,0.\n";
}
double relative(double a, double b, double scale) { return std::abs(a - b) / scale; }
Subcase subcase(int id, int load, int motion) {
  Subcase out;
  out.id = id;
  out.load = load;
  out.motion = motion;
  return out;
}

} // namespace

TEST(Motion, EachPositionIsTheRotatedStaticDeck) {
  for (const bool saturating : {false, true}) {
    SCOPED_TRACE(saturating ? "saturating stator" : "linear stator");
    Machine g;
    g.stator_div = 72;
    g.rotor_property = 2;
    g.stator_property = 3;
    g.cards = magnetized_rotor(1.1) + stator_iron(saturating) + probes();
    Machine moving = g;
    moving.cards += rotor_motion(-7, 12.5, 4);
    auto m = model(moving);
    wind(m, 40, .3);
    m.subcases[0].motion = 1;
    const auto swept = solved(m);
    ASSERT_EQ(swept.cases.size(), 1U);
    ASSERT_EQ(swept.cases[0].steps.size(), 4U);
    for (int k = 0; k < 4; ++k) {
      const double phi = -7 + 12.5 * k;
      SCOPED_TRACE(std::format("{} degrees", phi));
      // What an analyst writes for this position: the rotor grids and the
      // magnet frame turned by phi, and no MOTION.
      Machine turned = g;
      turned.rotor_turn = phi * degree;
      turned.cards = magnetized_rotor(1.1, phi * degree) + stator_iron(saturating) + probes();
      auto s = model(turned);
      wind(s, 40, .3);
      const auto fixed = solved(s);
      ASSERT_EQ(fixed.cases.size(), 1U);
      const auto &a = swept.cases[0].steps[static_cast<size_t>(k)];
      const auto &b = fixed.cases[0];
      // Round-off for linear iron; Newton's tolerance (1e-8 relative
      // residual) once the curve is used, as the sweep starts each position
      // from the last.
      const double tolerance = saturating ? 1e-6 : 1e-10;
      EXPECT_EQ(a.position, phi);
      EXPECT_LT(relative(a.energy, b.energy, std::abs(b.energy)), tolerance);
      EXPECT_LT(relative(a.coenergy, b.coenergy, std::abs(b.coenergy)), tolerance);
      EXPECT_LT(relative(a.circuits[0].linkage, b.circuits[0].linkage,
                         std::abs(b.circuits[0].linkage)),
                tolerance);
      EXPECT_LT(relative(a.circuit_inductance(0, 0), b.circuit_inductance(0, 0),
                         b.circuit_inductance(0, 0)),
                saturating ? 1e-5 : 1e-10);
      // The rotor is inside the gap and the axis is the plane normal.
      EXPECT_EQ(a.torque, a.gap_torque);
      EXPECT_LT(relative(a.torque, b.gap_torque, std::abs(b.gap_torque) + 1e-3), tolerance)
          << a.torque << " " << b.gap_torque;
      for (size_t p = 0; p < 3; ++p)
        EXPECT_LT((a.probe_b[p] - b.probe_b[p]).norm(), tolerance * b.probe_b[p].norm() + 1e-14)
            << "probe " << p + 1 << ": " << a.probe_b[p].transpose() << " vs "
            << b.probe_b[p].transpose();
    }
  }
}

TEST(Motion, AirGapTorqueIsTheDerivativeOfTheCoenergy) {
  // At fixed current the torque on the rotor is dW'/dphi. A central
  // difference of the coenergy over +-h must agree with the AGE torque to
  // O(h^2), linear or saturated: the rigid parts' energy does not change
  // with phi, so the derivative is the gap's, held at the solved potential.
  for (const bool saturating : {false, true}) {
    SCOPED_TRACE(saturating ? "saturating stator" : "linear stator");
    Machine g;
    g.stator_div = 72;
    g.rotor_property = 2;
    g.stator_property = 3;
    g.cards = magnetized_rotor(1.1) + stator_iron(saturating) + "ESET,20,PROP,2\n";
    auto m = model(g);
    wind(m, 40, .3);
    const double h = .05;
    const std::vector<double> centers{3, 21, 52};
    m.subcases.clear();
    for (size_t k = 0; k < centers.size(); ++k) {
      const int id = static_cast<int>(k) + 1;
      m.motions[id] = {.id = id, .set = 20, .coordinates = 0, .count = 3,
                       .start = centers[k] - h, .step = h};
      m.subcases.push_back(subcase(id, 1, id));
    }
    const auto r = solved(m);
    ASSERT_EQ(r.cases.size(), centers.size());
    for (size_t k = 0; k < centers.size(); ++k) {
      const auto &s = r.cases[k].steps;
      const double derivative = (s[2].coenergy - s[0].coenergy) / (2 * h * degree);
      // Truncation is about 1e-7; with the curve, the coenergy carries
      // Newton's tolerance into a difference over 0.1 degree.
      EXPECT_NEAR(s[1].torque, derivative,
                  (saturating ? 2e-5 : 1e-6) * std::abs(s[1].torque));
    }
  }
}

TEST(Motion, MagnetInAUniformFieldFollowsTheAnalyticTorque) {
  // A rotor magnetized uniformly (Br along x, mu_r = 1) in air, in the
  // uniform field B0 along x that MBC holds on the outer boundary. The
  // magnet's own field adds no torque (its image in the boundary is along
  // M), so the torque is m x B0: T = -(Br / mu0) B0 S d sin(phi) for rotor
  // area S and depth d, at every position of a full turn.
  const double br = 1.0, b0 = .3, depth = .1;
  std::vector<double> errors;
  for (const int divisions : {48, 96}) {
    Machine g;
    g.rotor_div = divisions;
    g.stator_div = divisions * 3 / 2;
    g.rotor_rings = divisions / 8;
    g.stator_rings = divisions / 6;
    g.rotor_property = 2;
    g.cards = magnetized_rotor(br) + rotor_motion(0, 30, 12) +
              std::format("MBC,1,0,UNIFORM,{},0.,0.\n", b0);
    auto m = model(g);
    m.subcases = {subcase(1, 0, 1)};
    m.subcases[0].boundary = 1;
    const auto r = solved(m);
    ASSERT_EQ(r.cases.size(), 1U);
    // The rotor as meshed: a regular polygon on the gap circle.
    const double area = divisions / 2.0 * g.rotor * g.rotor * std::sin(2 * pi / divisions),
                 amplitude = br / mu0 * b0 * area * depth;
    double worst = 0;
    for (const auto &step : r.cases[0].steps) {
      const double expected = -amplitude * std::sin(step.position * degree);
      worst = std::max(worst, std::abs(step.torque - expected) / amplitude);
    }
    errors.push_back(worst);
  }
  // Measured 7.9e-4 and 2.0e-4 of the amplitude: second order.
  EXPECT_LT(errors[0], 2e-3);
  EXPECT_LT(errors[1], errors[0] / 3);
}

TEST(Motion, BandTorqueIsTheDerivativeOfTheCoenergy) {
  // The coenergy test above with METHOD=BAND: the band's energy is the
  // gap's, and the band torque its derivative for a turn of the rotor's
  // averaged traces under the band.
  for (const bool saturating : {false, true}) {
    SCOPED_TRACE(saturating ? "saturating stator" : "linear stator");
    Machine g;
    g.stator_div = 72;
    g.rotor_property = 2;
    g.stator_property = 3;
    g.gap = "PAIRGAP,9,BAND\n";
    g.cards = magnetized_rotor(1.1) + stator_iron(saturating) + "ESET,20,PROP,2\n";
    auto m = model(g);
    wind(m, 40, .3);
    const double h = .05;
    const std::vector<double> centers{3, 21, 52};
    m.subcases.clear();
    for (size_t k = 0; k < centers.size(); ++k) {
      const int id = static_cast<int>(k) + 1;
      m.motions[id] = {.id = id, .set = 20, .coordinates = 0, .count = 3,
                       .start = centers[k] - h, .step = h};
      m.subcases.push_back(subcase(id, 1, id));
    }
    const auto r = solved(m);
    ASSERT_EQ(r.cases.size(), centers.size());
    for (size_t k = 0; k < centers.size(); ++k) {
      const auto &s = r.cases[k].steps;
      const double derivative = (s[2].coenergy - s[0].coenergy) / (2 * h * degree);
      EXPECT_NEAR(s[1].torque, derivative,
                  (saturating ? 2e-5 : 1e-6) * std::abs(s[1].torque));
    }
  }
}

TEST(Motion, BandErrorCancelsOverThePolesWithTheBlankBandCount) {
  // A four-pole rotor of radial magnets on an iron hub, turning in a
  // slotless iron stator: no torque at any position, exactly. What a
  // sweep shows is discretization error, mostly the band's, which repeats
  // with the band's point spacing. With NBAND a multiple of the poles,
  // every pole sits at the same phase of it and their errors add; the
  // blank NBAND (the smallest prime count of at least the finer circle's
  // 72 grids, 73) spreads the poles over its phases and they cancel.
  std::vector<double> ripple;
  for (const char *gap : {"PAIRGAP,9,BAND,,72\n", "PAIRGAP,9,BAND\n"}) {
    SCOPED_TRACE(gap);
    Machine g;
    g.stator_div = 72;
    g.rotor_property = 2;
    g.stator_property = 3;
    g.gap = gap;
    // Positions every sixth of a band spacing, over two spacings.
    g.cards = "CORD2C,100,,0.,0.,0.,0.,0.,1.\n,1.,0.,0.\nMATMAG,2,1.,,1.1\n"
              "PMAG2D,2,2,100,1\nPMAG2D,4,2,100,-1\nMATMAG,5,800.\nPMAG2D,5,5\n" +
              stator_iron(false) + rotor_motion(0, 5. / 6, 13);
    auto m = model(g);
    auto &moving = m.element_sets[20];
    moving.clear();
    for (auto &e : m.elements) {
      if (e.property != 2)
        continue;
      const Vector c = centroid(m, e);
      e.property = c.norm() < .013 ? 5 : std::cos(2 * std::atan2(c.y(), c.x())) > 0 ? 2 : 4;
      moving.push_back(e.id);
    }
    m.subcases = {subcase(1, 0, 1)};
    const auto r = solved(m);
    ASSERT_EQ(r.mesh.air_gap->band_points, std::string(gap).find("72") != std::string::npos ? 72 : 73);
    const auto &s = r.cases[0].steps;
    ASSERT_EQ(s.size(), 13U);
    auto [low, high] = std::ranges::minmax(s | std::views::transform(&CaseResults::torque));
    ripple.push_back(high - low);
  }
  // Measured 0.44 and 0.0036 N m.
  EXPECT_LT(ripple[1], ripple[0] / 20) << ripple[0] << " N m against " << ripple[1];
}

TEST(Motion, ReusedFactorSweepMatchesFactoringEveryTangent) {
  // PARAM,MAGPCG solves each Newton step by CG preconditioned with an
  // earlier tangent's factor, to an inexact (forcing-term) tolerance; the
  // inductances and linkage rates stay direct. A sweep of a saturating
  // machine must give the results of factoring every tangent to Newton's
  // tolerance, with fewer factorizations.
  Machine g;
  g.stator_div = 72;
  g.rotor_property = 2;
  g.stator_property = 3;
  g.gap = "PAIRGAP,9,BAND\n";
  g.cards = magnetized_rotor(1.1) + stator_iron(true) + probes() +
            rotor_motion(0, 1.5, 6);
  auto direct_model = model(g);
  wind(direct_model, 40, .3);
  direct_model.subcases[0].motion = 1;
  Machine with_param = g;
  with_param.cards += "PARAM,MAGPCG,20\n";
  auto reused_model = model(with_param);
  wind(reused_model, 40, .3);
  reused_model.subcases[0].motion = 1;
  EXPECT_EQ(direct_model.options.tangent_reuse, 0);
  EXPECT_EQ(reused_model.options.tangent_reuse, 20);
  for (const char *bad : {"PARAM,MAGPCG,-1\n", "PARAM,MAGPCG,1001\n"}) {
    Machine wrong = g;
    wrong.cards += bad;
    EXPECT_NE(error_of(parse(wrong, geometry(wrong))).find("PARAM,MAGPCG"),
              std::string::npos)
        << bad;
  }
  const auto direct = solved(direct_model);
  const auto reused = solved(reused_model);
  ASSERT_EQ(direct.cases.size(), 1U);
  ASSERT_EQ(reused.cases.size(), 1U);
  ASSERT_EQ(reused.cases[0].steps.size(), 6U);
  for (size_t k = 0; k < 6; ++k) {
    SCOPED_TRACE(std::format("position {}", k));
    const auto &a = reused.cases[0].steps[k];
    const auto &b = direct.cases[0].steps[k];
    EXPECT_LE(a.residual, 1e-8);
    EXPECT_LT(relative(a.energy, b.energy, std::abs(b.energy)), 1e-7);
    EXPECT_LT(relative(a.circuits[0].linkage, b.circuits[0].linkage,
                       std::abs(b.circuits[0].linkage)),
              1e-7);
    EXPECT_LT(relative(a.torque, b.torque, std::abs(b.torque)), 1e-6);
    // The response solves see the same converged state to Newton's
    // tolerance.
    EXPECT_LT(relative(a.circuit_inductance(0, 0), b.circuit_inductance(0, 0),
                       b.circuit_inductance(0, 0)),
              1e-6);
    EXPECT_LT(relative(a.linkage_rate[0], b.linkage_rate[0],
                       std::abs(b.linkage_rate[0])),
              1e-5);
    for (size_t p = 0; p < 3; ++p)
      EXPECT_LT((a.probe_b[p] - b.probe_b[p]).norm(), 1e-6 * b.probe_b[p].norm());
  }
  // "N symbolic analyses, M numeric factorizations, R right-hand sides[, C
  // by CG ...]" in the F06 SOLVER block.
  auto usage = [](const Results &r) {
    for (const auto &[label, text] : r.summary.solver.rows)
      if (label == "Tangent reuse")
        return text;
    ADD_FAILURE() << "no Tangent reuse row";
    return std::string();
  };
  auto factorizations = [](const std::string &text) {
    const auto at = text.find(" numeric");
    const auto start = text.rfind(' ', at - 1) + 1;
    return std::stoi(text.substr(start, at - start));
  };
  const std::string direct_usage = usage(direct), reused_usage = usage(reused);
  EXPECT_EQ(direct_usage.find("by CG"), std::string::npos) << direct_usage;
  EXPECT_NE(reused_usage.find("by CG"), std::string::npos) << reused_usage;
  EXPECT_LT(2 * factorizations(reused_usage), factorizations(direct_usage))
      << reused_usage << " | " << direct_usage;
}

TEST(Motion, BandMagnetInAUniformFieldFollowsTheAnalyticTorque) {
  // The magnet in a uniform field of the test above, across METHOD=BAND:
  // T = -(Br / mu0) B0 S d sin(phi) at every position of a full turn, to
  // second order in the divisions.
  const double br = 1.0, b0 = .3, depth = .1;
  std::vector<double> errors;
  for (const int divisions : {48, 96}) {
    Machine g;
    g.rotor_div = divisions;
    g.stator_div = divisions * 3 / 2;
    g.rotor_rings = divisions / 8;
    g.stator_rings = divisions / 6;
    g.rotor_property = 2;
    g.gap = "PAIRGAP,9,BAND\n";
    g.cards = magnetized_rotor(br) + rotor_motion(0, 30, 12) +
              std::format("MBC,1,0,UNIFORM,{},0.,0.\n", b0);
    auto m = model(g);
    m.subcases = {subcase(1, 0, 1)};
    m.subcases[0].boundary = 1;
    const auto r = solved(m);
    ASSERT_EQ(r.cases.size(), 1U);
    const double area = divisions / 2.0 * g.rotor * g.rotor * std::sin(2 * pi / divisions),
                 amplitude = br / mu0 * b0 * area * depth;
    double worst = 0;
    for (const auto &step : r.cases[0].steps) {
      const double expected = -amplitude * std::sin(step.position * degree);
      worst = std::max(worst, std::abs(step.torque - expected) / amplitude);
    }
    errors.push_back(worst);
  }
  EXPECT_LT(errors[0], 3e-3);
  EXPECT_LT(errors[1], errors[0] / 3);
}

TEST(Motion, ARoundRotorHasNoCoggingAtAnyPosition) {
  // A round, unmagnetized iron rotor in a wound stator: turning it changes
  // nothing physical. Its mesh is a 48-gon, but that shape is harmonic 48
  // and above, beyond the gap's 24 harmonics, so the stator sees a round
  // rotor and every position, on the mesh pitch or between, is the same to
  // round-off, with no torque. A probe in the rotor stays in space while the
  // rotor's cells pass under it, so it repeats only at the mesh pitch.
  Machine g;
  g.stator_div = 72;
  g.rotor_property = 2;
  g.stator_property = 3;
  g.cards = "MATMAG,2,800.\nPMAG2D,2,2\n" + stator_iron(false) +
            rotor_motion(-15, 3.75, 9) + probes();
  auto m = model(g);
  wind(m, 40, .3);
  m.subcases[0].motion = 1;
  const auto r = solved(m);
  const auto &s = r.cases[0].steps;
  ASSERT_EQ(s.size(), 9U);
  const auto &reference = s.front();
  for (size_t k = 0; k < s.size(); ++k) {
    const auto &step = s[k];
    SCOPED_TRACE(std::format("{} degrees", step.position));
    EXPECT_NEAR(step.energy, reference.energy, 1e-12 * reference.energy);
    EXPECT_NEAR(step.circuits[0].linkage, reference.circuits[0].linkage,
                1e-12 * std::abs(reference.circuits[0].linkage));
    // Torque in N m against energy in J per radian.
    EXPECT_LT(std::abs(step.torque), 1e-12 * reference.energy);
    // Probes 1 and 2 are in the stator and the gap, 3 in the rotor; every
    // other position is a multiple of the 7.5 degree pitch.
    for (size_t p = 0; p < (k % 2 ? 2 : 3); ++p)
      EXPECT_LT((step.probe_b[p] - reference.probe_b[p]).norm(),
                1e-10 * reference.probe_b[p].norm())
          << "probe " << p + 1;
  }
}

TEST(Motion, CoggingRepeatsWithTheStatorSymmetry) {
  // A two-pole magnet rotor in a stator with two opposite slots and no
  // current. The stator is the same after a half turn and the magnet is
  // reversed, so the torque and energy repeat every 180 degrees and the
  // winding's linkage changes sign: the magnet frame turns with the rotor.
  Machine g;
  g.stator_div = 72;
  g.rotor_property = 2;
  g.stator_property = 3;
  g.cards = magnetized_rotor(1.1) + stator_iron(false) + rotor_motion(-40, 20, 13);
  auto m = model(g);
  wind(m, 40, .3);
  m.currents = {{1, 1, 0.}};
  m.subcases[0].motion = 1;
  const auto r = solved(m);
  const auto &s = r.cases[0].steps;
  ASSERT_EQ(s.size(), 13U); // -40 to 200 degrees
  double low = s[0].torque, high = low;
  const double linkage =
      std::accumulate(s.begin(), s.end(), 0., [](double most, const CaseResults &step) {
        return std::max(most, std::abs(step.circuits[0].linkage));
      });
  for (size_t k = 0; k + 9 < s.size(); ++k) {
    const auto &a = s[k], &b = s[k + 9];
    SCOPED_TRACE(std::format("{} and {} degrees", a.position, b.position));
    EXPECT_NEAR(b.energy, a.energy, 1e-11 * a.energy);
    EXPECT_NEAR(b.torque, a.torque, 1e-11 * a.energy);
    EXPECT_NEAR(b.circuits[0].linkage, -a.circuits[0].linkage, 1e-11 * linkage);
    low = std::min(low, a.torque);
    high = std::max(high, a.torque);
  }
  // The slots make a real cogging torque, which the comparison relies on.
  EXPECT_GT(high - low, 1e-3);
}

TEST(Motion, MovedMeshIsTheMeshOfTheMovedModel) {
  // Each MOTION position's mesh is the base mesh with its moving cells and
  // faces placed again, not a rebuilt topology. It must be exactly the mesh
  // make_mesh builds from the moved model, with the AGE and with a band.
  for (const char *gap : {"PAIRGAP,9\n", "PAIRGAP,9,BAND\n"}) {
    SCOPED_TRACE(gap);
    Machine g;
    g.rotor_property = 2;
    g.stator_property = 3;
    g.gap = gap;
    g.cards = magnetized_rotor(1.0, .2) + stator_iron(false) + rotor_motion(0, 30, 2);
    const auto m = model(g);
    const auto base = mesh_of(m);
    auto found = moving_part(m, base, m.motions.at(1));
    ASSERT_TRUE(std::holds_alternative<MovingPart>(found)) << error_of(found);
    const auto moved = moved_model(m, m.motions.at(1), std::get<MovingPart>(found), 37.5);
    const auto fresh = mesh_of(moved);
    auto built = make_mesh(moved, base);
    ASSERT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
    const auto &reused = std::get<Mesh>(built);
    EXPECT_EQ(reused.nodes, fresh.nodes);
    EXPECT_EQ(reused.edges, fresh.edges);
    ASSERT_EQ(reused.cells.size(), fresh.cells.size());
    for (size_t i = 0; i < fresh.cells.size(); ++i) {
      const auto &a = reused.cells[i], &b = fresh.cells[i];
      EXPECT_EQ(a.id, b.id);
      EXPECT_EQ(a.property, b.property) << "element " << b.id;
      EXPECT_EQ(a.material, b.material);
      EXPECT_EQ(a.points, b.points) << "element " << b.id;
      EXPECT_EQ(a.jacobian, b.jacobian);
      EXPECT_EQ(a.inverse, b.inverse);
      EXPECT_EQ(a.map_origin, b.map_origin);
      EXPECT_EQ(a.minimum, b.minimum);
      EXPECT_EQ(a.maximum, b.maximum);
      EXPECT_EQ(a.volume, b.volume);
      EXPECT_EQ(a.affine, b.affine);
      EXPECT_EQ(a.polyhedral, b.polyhedral);
      EXPECT_EQ(a.faces, b.faces);
      EXPECT_EQ(a.face_sign, b.face_sign);
    }
    ASSERT_EQ(reused.faces.size(), fresh.faces.size());
    for (size_t f = 0; f < fresh.faces.size(); ++f) {
      const auto &a = reused.faces[f], &b = fresh.faces[f];
      EXPECT_EQ(a.nodes, b.nodes);
      EXPECT_EQ(a.cells, b.cells);
      EXPECT_EQ(a.center, b.center);
      EXPECT_EQ(a.normal, b.normal);
      EXPECT_EQ(a.area, b.area);
    }
    ASSERT_TRUE(reused.air_gap && fresh.air_gap);
    for (auto circle : {&AirGap::inner, &AirGap::outer}) {
      EXPECT_EQ(((*reused.air_gap).*circle).nodes, ((*fresh.air_gap).*circle).nodes);
      EXPECT_EQ(((*reused.air_gap).*circle).angles, ((*fresh.air_gap).*circle).angles);
    }
    EXPECT_EQ(reused.air_gap->band_points, fresh.air_gap->band_points);
    EXPECT_EQ(reused.band_unknowns, fresh.band_unknowns);
    EXPECT_EQ(reused.dofs(), fresh.dofs());
  }
  // A model with other elements is not a moved copy.
  Machine g;
  g.rotor_property = 2;
  g.cards = magnetized_rotor(1.0);
  const auto m = model(g);
  const auto base = mesh_of(m);
  auto fewer = m;
  fewer.elements.pop_back();
  const auto refused = make_mesh(fewer, base);
  EXPECT_NE(error_of(refused).find("field cells"), std::string::npos) << error_of(refused);
}

TEST(Motion, MovedModelTurnsTheMovingMagnetsFramesAndNothingFixed) {
  // The rotor magnet (PID 2, CORD2R 1) shares its property with a fixed
  // magnet patch in the stator. At 30 degrees the rotor grids and the
  // rotor's remanence turn; the stator's grids, property and frame do not.
  Machine g;
  g.rotor_property = 2;
  g.cards = magnetized_rotor(1.0, .2) + rotor_motion(0, 30, 2);
  auto m = model(g);
  const auto patch = elements_within(m, .03, .032);
  ASSERT_FALSE(patch.empty());
  m.elements[static_cast<size_t>(patch.front() - 1)].property = 2;
  m.element_sets[20].erase(std::remove(m.element_sets[20].begin(), m.element_sets[20].end(),
                                       patch.front()),
                           m.element_sets[20].end());
  const auto mesh = mesh_of(m);
  const auto &motion = m.motions.at(1);
  auto found = moving_part(m, mesh, motion);
  ASSERT_TRUE(std::holds_alternative<MovingPart>(found)) << error_of(found);
  const auto &part = std::get<MovingPart>(found);
  EXPECT_TRUE(part.inner);
  EXPECT_EQ(part.sign, 1);
  EXPECT_EQ(part.elements.size(), m.element_sets.at(20).size());

  const auto moved = moved_model(m, motion, part, 30);
  const Matrix turn = Eigen::AngleAxisd(30 * degree, Vector::UnitZ()).toRotationMatrix();
  for (const auto &[id, grid] : m.nodes) {
    const Vector expected = part.grids.contains(id) ? Vector(turn * grid.position)
                                                    : grid.position;
    EXPECT_LT((moved.nodes.at(id).position - expected).norm(), 1e-17) << "grid " << id;
  }
  // The rotor's cells take a new property, the original stays with the
  // stator patch, and its frame is a turned copy of CORD2R 1.
  const auto &rotor_cell = moved.elements.front();
  ASSERT_TRUE(part.elements.contains(rotor_cell.id));
  const int pid = rotor_cell.property;
  EXPECT_NE(pid, 2);
  EXPECT_EQ(moved.elements[static_cast<size_t>(patch.front() - 1)].property, 2);
  EXPECT_EQ(moved.properties.at(2).coordinates, 1);
  EXPECT_EQ(moved.coordinates.at(1).a, m.coordinates.at(1).a);
  const auto &property = moved.properties.at(pid);
  EXPECT_EQ(property.material, 2);
  EXPECT_EQ(property.axis, 1);
  const Vector original = coordinate_axes(m.coordinates.at(1)).col(0),
               rotated = coordinate_axes(moved.coordinates.at(property.coordinates)).col(0);
  EXPECT_LT((rotated - turn * original).norm(), 1e-15);
  for (const auto &e : moved.elements) {
    if (part.elements.contains(e.id)) {
      EXPECT_EQ(e.property, pid);
    }
  }

  // A property only the rotor uses is turned in place, and zero degrees is
  // the model as given.
  m.elements[static_cast<size_t>(patch.front() - 1)].property = 1;
  const auto alone = moved_model(m, motion, part, 30);
  EXPECT_EQ(alone.elements.front().property, 2);
  EXPECT_NE(alone.properties.at(2).coordinates, 1);
  const auto still = moved_model(m, motion, part, 0);
  for (const auto &[id, grid] : m.nodes)
    EXPECT_EQ(still.nodes.at(id).position, grid.position);
  EXPECT_EQ(still.properties.at(2).coordinates, 1);
}

TEST(Motion, CardsAndCaseControl) {
  Machine g;
  g.rotor_property = 2;
  g.cards = "MATMAG,2,1.\nPMAG2D,2,2\n" + rotor_motion(-2.5, .5, 11);
  auto text = deck(g, geometry(g));
  text.replace(text.find("CEND\n"), 5,
               "CEND\nMSTEP = ALL\nSUBCASE 1\n  MOTION = 1\n  BFIELD(EVERY=4) = ALL\n"
               "SUBCASE 2\n  MSTEP = NONE\n");
  auto parsed_text = parse_string(text);
  ASSERT_TRUE(std::holds_alternative<Model>(parsed_text)) << error_of(parsed_text);
  const auto &m = std::get<Model>(parsed_text);
  const auto &motion = m.motions.at(1);
  EXPECT_EQ(motion.set, 20);
  EXPECT_EQ(motion.coordinates, 0);
  EXPECT_EQ(motion.count, 11);
  EXPECT_EQ(motion.position(0), -2.5);
  EXPECT_EQ(motion.position(10), 2.5);
  ASSERT_EQ(m.subcases.size(), 2U);
  EXPECT_EQ(m.subcases[0].motion, 1);
  EXPECT_TRUE(m.subcases[0].steps);
  EXPECT_EQ(m.subcases[0].field.every, 4);
  EXPECT_TRUE(m.subcases[0].field.enabled);
  EXPECT_EQ(m.subcases[1].motion, 0);
  EXPECT_FALSE(m.subcases[1].steps);

  auto error = [&](const std::string &cards, const std::string &control = "") {
    Machine bad = g;
    bad.cards = "MATMAG,2,1.\nPMAG2D,2,2\nESET,20,PROP,2\n" + cards;
    auto t = deck(bad, geometry(bad));
    t.replace(t.find("CEND\n"), 5, "CEND\n" + control);
    return error_of(parse_string(t));
  };
  // A rotating machine's gap is an annulus, which a part cannot slide along.
  EXPECT_NE(error("MOTION,1,20,0,TRANSLATE,0.,1.,2\n").find(
                "MOTION 1 TRANSLATE slides the part along a gap that is a line, which a "
                "model has when it repeats along it: it needs MPERIOD with SHIFT"),
            std::string::npos);
  EXPECT_NE(error("MOTION,1,20,0,SPIN,0.,1.,2\n").find(
                "MOTION TYPE must be ROTATE or TRANSLATE"),
            std::string::npos);
  EXPECT_NE(error("MOTION,1,20,0,ROTATE,0.,0.,2\n").find(
                "MOTION 1 requires COUNT >= 1, and a nonzero STEP when COUNT > 1"),
            std::string::npos);
  EXPECT_NE(error("MOTION,1,20,0,ROTATE,0.,1.,0\n").find("requires COUNT >= 1"),
            std::string::npos);
  EXPECT_NE(error("MOTION,1,21,0,ROTATE,0.,1.,2\n").find("MOTION 1 references undefined ESET 21"),
            std::string::npos);
  EXPECT_NE(error("MOTION,1,20,7,ROTATE,0.,1.,2\n").find(
                "MOTION 1 references undefined coordinates 7"),
            std::string::npos);
  EXPECT_NE(error("CORD2R,7,,0.,0.,0.,1.,0.,0.\n,0.,1.,0.\nMOTION,1,20,7,ROTATE,0.,1.,2\n")
                .find("MOTION 1: the z axis of CID 7 must be parallel to the MAG2D plane normal"),
            std::string::npos);
  EXPECT_NE(error("ESET,21,99999\nMOTION,1,21,0,ROTATE,0.,1.,2\n")
                .find("MOTION 1 ESET 21 references undefined element 99999"),
            std::string::npos);
  // ESET 22 holds an air-gap element.
  const auto geo = geometry(g);
  EXPECT_NE(error(std::format("ESET,22,{}\nMOTION,1,22,0,ROTATE,0.,1.,2\n",
                              geo.gap_elements.front()))
                .find(std::format("MOTION 1 ESET 22 contains air-gap element {} (PAIRGAP 9)",
                                  geo.gap_elements.front())),
            std::string::npos);
  EXPECT_NE(error("PARAM,AIRGAP,MESH\nMOTION,1,20,0,ROTATE,0.,1.,2\n")
                .find("MOTION 1 needs the air-gap element between the moving and fixed "
                      "parts, but PAIRGAP 9 is solved as meshed air by PARAM,AIRGAP,MESH; "
                      "meshed air cannot slide"),
            std::string::npos);
  EXPECT_NE(error("", "SUBCASE 1\n  MOTION = 3\n").find("Subcase 1 references undefined MOTION 3"),
            std::string::npos);
  EXPECT_NE(error("", "BFIELD(EVERY=0) = ALL\n").find("BFIELD describer must be (EVERY=n) with n >= 1"),
            std::string::npos);
  EXPECT_NE(error("", "BFIELD(STRIDE=2) = ALL\n").find("BFIELD describer must be (EVERY=n)"),
            std::string::npos);
  EXPECT_NE(error("", "LOAD(EVERY=2) = 1\n").find("LOAD takes no describer"), std::string::npos);
  EXPECT_NE(error("", "MSTEP = SOME\n").find("MSTEP must be ALL or NONE"), std::string::npos);

  // Without a PAIRGAP there is nothing to slide across.
  Machine meshed = g;
  meshed.gap = "PMAG2D,9,1\n";
  meshed.cards = "MATMAG,2,1.\nPMAG2D,2,2\n" + rotor_motion(0, 1, 2);
  EXPECT_NE(error_of(parse(meshed, geometry(meshed)))
                .find("MOTION 1 needs a PAIRGAP air gap between the moving and fixed parts"),
            std::string::npos);
}

TEST(Motion, TheMovingPartIsCheckedAgainstTheAirGap) {
  auto failure = [](const std::string &cards, auto edit) {
    Machine g;
    g.rotor_property = 2;
    g.stator_property = 3;
    g.cards = "MATMAG,2,1.\nPMAG2D,2,2\nMATMAG,3,1.\nPMAG2D,3,3\n" + cards;
    auto m = model(g);
    m.subcases[0].motion = 1;
    edit(m);
    return error_of(solve(m));
  };
  const auto none = [](Model &) {};
  // Part of the rotor: its outer ring stays fixed and shares grids.
  const auto partial = failure(rotor_motion(0, 5, 2), [](Model &m) {
    auto &set = m.element_sets.at(20);
    set.erase(std::remove_if(set.begin(), set.end(),
                             [&](int id) {
                               return centroid(m, m.elements[static_cast<size_t>(id - 1)]).norm() >
                                      .018;
                             }),
              set.end());
  });
  EXPECT_NE(partial.find("MOTION 1: grid "), std::string::npos) << partial;
  EXPECT_NE(partial.find("belongs to moving element"), std::string::npos) << partial;
  EXPECT_NE(partial.find("the moving and fixed parts may meet only across the air gap"),
            std::string::npos);
  // Rotor and stator together border both circles.
  const auto both = failure("ESET,20,PROP,2,3\nMOTION,1,20,0,ROTATE,0.,5.,2\n", none);
  EXPECT_NE(both.find("MOTION 1: ESET 20 borders both the inner and the outer circle of "
                      "PAIRGAP 9; the moving part must border exactly one of them"),
            std::string::npos)
      << both;
  // The stator alone may move: the outer circle then turns.
  Machine outer;
  outer.rotor_property = 2;
  outer.stator_property = 3;
  outer.cards = "MATMAG,2,1.\nPMAG2D,2,2\nMATMAG,3,1.\nPMAG2D,3,3\nESET,20,PROP,3\n"
                "MOTION,1,20,0,ROTATE,0.,5.,2\n";
  const auto stator = model(outer);
  auto found = moving_part(stator, mesh_of(stator), stator.motions.at(1));
  ASSERT_TRUE(std::holds_alternative<MovingPart>(found)) << error_of(found);
  EXPECT_FALSE(std::get<MovingPart>(found).inner);
  // An axis off the gap's center, and an axis along -z.
  const auto off = failure("ESET,20,PROP,2\nCORD2R,7,,.001,0.,0.,.001,0.,1.\n,1.,0.,0.\n"
                           "MOTION,1,20,7,ROTATE,0.,5.,2\n",
                           none);
  EXPECT_NE(off.find("MOTION 1: the rotation axis, the z axis of CID 7 through (0.001, 0, 0), "
                     "passes 1.000e-03 from the center of PAIRGAP 9"),
            std::string::npos)
      << off;
  Machine flipped;
  flipped.rotor_property = 2;
  flipped.cards = "MATMAG,2,1.\nPMAG2D,2,2\nESET,20,PROP,2\nCORD2R,7,,0.,0.,0.,0.,0.,-1.\n"
                  ",1.,0.,0.\nMOTION,1,20,7,ROTATE,0.,5.,2\n";
  const auto down = model(flipped);
  auto reversed = moving_part(down, mesh_of(down), down.motions.at(1));
  ASSERT_TRUE(std::holds_alternative<MovingPart>(reversed)) << error_of(reversed);
  EXPECT_EQ(std::get<MovingPart>(reversed).sign, -1);
}

TEST(Motion, ResultsHoldEveryPositionAndTheFieldsAreATimeSeries) {
  // Subcase 1 sweeps 0..30 degrees writing every second position's fields
  // and every position's tables; subcase 2 is static; subcase 3 sweeps
  // 5 and 25 degrees with summaries only; subcase 4 writes no fields.
  Machine g;
  g.stator_div = 60;
  g.rotor_div = 40;
  g.rotor_property = 2;
  g.stator_property = 3;
  g.cards = magnetized_rotor(1.1) + stator_iron(false) + probes() + rotor_motion(0, 10, 4) +
            "MOTION,2,20,0,ROTATE,5.,20.,2\n";
  auto m = model(g);
  wind(m, 40, .3);
  m.subcases = {subcase(1, 1, 1), subcase(2, 1, 0), subcase(3, 1, 2), subcase(4, 1, 1)};
  m.subcases[0].field.every = 2;
  m.subcases[0].steps = true;
  m.subcases[3].field.enabled = false;
  const auto r = solved(m);
  ASSERT_EQ(r.cases.size(), 4U);
  const auto &sweep = r.cases[0].steps;
  ASSERT_EQ(sweep.size(), 4U);
  // Only the written positions keep their fields.
  for (size_t k = 0; k < 4; ++k)
    EXPECT_EQ(sweep[k].b.empty(), k % 2 == 1) << k;
  EXPECT_TRUE(std::isfinite(r.cases[1].gap_torque));
  EXPECT_TRUE(r.cases[1].steps.empty());
  EXPECT_EQ(r.cases[0].iterations, 4);
#ifdef VIBESTRAN_HAS_HDF5
  using namespace vibestran::test;
  const auto directory = std::filesystem::temp_directory_path() / "vibestran_motion";
  std::filesystem::create_directories(directory);
  ASSERT_FALSE(write_results(m, r, directory / "motion.F06").has_value());
  const auto file = directory / "motion.mag.vtkhdf";

  // Per-position tables with a leading position axis, and the summary.
  const std::string one = "/MAGSTAT/subcases/1";
  EXPECT_EQ(h5_attribute(file, one, "motion"), 1);
  EXPECT_EQ(h5_attribute(file, one, "positions"), 4);
  double mean = 0, low = sweep[0].torque, high = low;
  for (const auto &step : sweep) {
    mean += step.torque / 4;
    low = std::min(low, step.torque);
    high = std::max(high, step.torque);
  }
  EXPECT_DOUBLE_EQ(h5_attribute(file, one, "torque_mean_Nm"), mean);
  EXPECT_DOUBLE_EQ(h5_attribute(file, one, "torque_peak_to_peak_Nm"), high - low);
  EXPECT_DOUBLE_EQ(h5_attribute(file, one, "torque_ripple_percent"),
                   100 * (high - low) / std::abs(mean));
  EXPECT_EQ(h5_attribute(file, one, "current_free"), 0);
  const auto positions = h5_table(file, one + "/positions_deg");
  const auto torque = h5_table(file, one + "/motion/torque_Nm");
  const auto energy = h5_table(file, one + "/energy_J");
  ASSERT_EQ(positions.values, (std::vector<double>{0, 10, 20, 30}));
  ASSERT_EQ(torque.rows(), 4U);
  for (size_t k = 0; k < 4; ++k) {
    EXPECT_EQ(torque.values[k], sweep[k].torque);
    EXPECT_EQ(energy.values[k], sweep[k].energy);
  }
  EXPECT_EQ(h5_table(file, one + "/sources/current_A").dims, (std::vector<hsize_t>{4, 2}));
  EXPECT_EQ(h5_table(file, one + "/circuits/inductance_H").dims,
            (std::vector<hsize_t>{4, 1, 1}));
  const auto probe_b = h5_table(file, one + "/probes/B_T");
  ASSERT_EQ(probe_b.dims, (std::vector<hsize_t>{4, 3, 3}));
  EXPECT_EQ(probe_b.values[(3 * 3 + 2) * 3 + 1], sweep[3].probe_b[2].y());
  // Summaries only under MSTEP = NONE; a static subcase keeps its scalars
  // and gains the air-gap torque.
  EXPECT_EQ(h5_attribute(file, "/MAGSTAT/subcases/3", "positions"), 2);
  EXPECT_FALSE(h5_exists(file, "/MAGSTAT/subcases/3/positions_deg"));
  EXPECT_EQ(h5_attribute(file, "/MAGSTAT/subcases/2", "air_gap_torque_Nm"),
            r.cases[1].gap_torque);
  EXPECT_EQ(h5_attribute(file, "/MAGSTAT/subcases/2", "energy_J"), r.cases[1].energy);

  // The time axis is every written position, in degrees. Each block shows
  // its state at the largest of its own positions not above the time.
  const std::vector<double> times{0, 5, 20, 25};
  EXPECT_FALSE(h5_exists(file, "/VTKHDF/SUBCASE_4"));
  const std::vector<std::pair<std::string, std::vector<double>>> held{
      {"SUBCASE_1", {0, 0, 1, 1}}, {"SUBCASE_2", {0, 0, 0, 0}},
      {"SUBCASE_3", {0, 0, 0, 1}}, {"COIL_11", {0, 0, 0, 0}}};
  for (const auto &[block, states] : held) {
    SCOPED_TRACE(block);
    const std::string path = "/VTKHDF/" + block;
    EXPECT_EQ(h5_attribute(file, path + "/Steps", "NSteps"), 4);
    EXPECT_EQ(h5_table(file, path + "/Steps/Values").values, times);
    const double points = h5_table(file, path + "/NumberOfPoints").values[0],
                 cells = h5_table(file, path + "/NumberOfCells").values[0];
    std::vector<double> point_offsets, cell_offsets;
    for (double k : states) {
      point_offsets.push_back(k * points);
      cell_offsets.push_back(k * cells);
    }
    EXPECT_EQ(h5_table(file, path + "/Steps/PointOffsets").values, point_offsets);
    const std::string field = block.starts_with("COIL") ? "J_per_ampere_per_m2" : "B_T";
    EXPECT_EQ(h5_table(file, path + "/Steps/CellDataOffsets/" + field).values, cell_offsets);
    EXPECT_EQ(h5_table(file, path + "/Steps/CellDataOffsets/element_id").values,
              std::vector<double>(4, 0));
    EXPECT_EQ(h5_table(file, path + "/Points").rows(),
              static_cast<size_t>((states.back() + 1) * points));
  }
  // Every cell is drawn, the air gap's last: their own field at each state,
  // and the rotor's points turned with it.
  const auto &mesh = r.mesh;
  const auto &gap = *mesh.air_gap;
  const auto cells = static_cast<size_t>(h5_table(file, "/VTKHDF/SUBCASE_1/NumberOfCells").values[0]);
  ASSERT_EQ(cells, mesh.cells.size() + gap.elements.size());
  const auto element_ids = h5_table(file, "/VTKHDF/SUBCASE_1/CellData/element_id");
  EXPECT_EQ(element_ids.values.back(), gap.elements.back());
  const auto b = h5_table(file, "/VTKHDF/SUBCASE_1/CellData/B_T");
  const auto &at20 = sweep[2];
  const auto &last = m.elements[static_cast<size_t>(gap.elements.back() - 1)];
  const Vector expected =
      AirGapSeries(*at20.air_gap, at20.potential).field(centroid(m, last));
  for (int c = 0; c < 3; ++c)
    EXPECT_EQ(b.values[(cells + cells - 1) * 3 + static_cast<size_t>(c)], expected[c]);
  const auto points = h5_table(file, "/VTKHDF/SUBCASE_1/Points");
  const auto count = static_cast<size_t>(h5_table(file, "/VTKHDF/SUBCASE_1/NumberOfPoints").values[0]);
  // Point 1 is a rotor grid (the first cell is a rotor triangle).
  const Vector p0(points.at(1, 0), points.at(1, 1), points.at(1, 2)),
      p1(points.at(count + 1, 0), points.at(count + 1, 1), points.at(count + 1, 2));
  EXPECT_LT((p1 - Eigen::AngleAxisd(20 * degree, Vector::UnitZ()) * p0).norm(), 1e-17);

  std::ifstream in(directory / "motion.F06");
  std::stringstream text;
  text << in.rdbuf();
  const auto f06 = text.str();
  for (const std::string expected_text :
       {"   Motion ................... MOTION 1 turns ESET 20 (",
        "MOTION 1: 4 positions from 0 to 30 degrees about the z axis of CID 0\n",
        "Torque on ESET 20 about the MOTION axis (Nm): mean ",
        "  STEP   POSITION(deg)          TORQUE(Nm)",
        "MOTION 2: 2 positions from 5 to 25 degrees",
        "Air-gap torque on the part inside the gap, about the plane normal (Nm): "})
    EXPECT_NE(f06.find(expected_text), std::string::npos) << expected_text << "\n" << f06;
  // One position table, and one of the back-EMF per unit speed (the AGE
  // gives dlambda/dphi of the closed circuit), both for subcase 1 alone.
  const std::string table = "  STEP   POSITION(deg)          TORQUE(Nm)",
                    emf = "  DLAMBDA/DPHI      1(V s/rad)";
  EXPECT_EQ(f06.find(table), f06.rfind(table));
  EXPECT_NE(f06.find(emf), std::string::npos);
  EXPECT_EQ(f06.find(emf), f06.rfind(emf));
  std::filesystem::remove_all(directory);
#else
  GTEST_SKIP() << "MAGSTAT results need HDF5";
#endif
}

TEST(Motion, AirGapTorqueMatchesMaxwellStressOnTheMeshedGap) {
  // The cross-check the documentation gives: the same deck with
  // PARAM,AIRGAP,MESH and an MFORCE BODY on the rotor, its surface through
  // the middle of a gap meshed in four layers (LAYER = 2, between two rings
  // of quads). The torques are two discretizations of one quantity; they
  // agree to 7e-4 at 48 divisions and 1.4e-4 at 96. A surface against a
  // band of distorted triangles converges only at first order (52% and 29%
  // off with two gap layers), which the documentation warns about.
  std::vector<double> differences;
  for (const int divisions : {48, 96}) {
    Machine g;
    g.rotor_div = divisions;
    g.stator_div = divisions * 3 / 2;
    g.rotor_rings = divisions / 8;
    g.stator_rings = divisions / 6;
    g.gap_layers = 4;
    g.rotor_property = 2;
    g.stator_property = 3;
    g.cards = magnetized_rotor(1.1, .4) + stator_iron(false) +
              "ESET,20,PROP,2\nMFORCE,1,BODY,20,0,2\n";
    auto age = model(g);
    wind(age, 40, .3);
    age.forces.clear(); // force surfaces may not touch the air-gap element
    const auto a = solved(age);
    Machine meshed = g;
    meshed.cards += "PARAM,AIRGAP,MESH\n";
    auto mesh = model(meshed);
    wind(mesh, 40, .3);
    const auto b = solved(mesh);
    ASSERT_EQ(b.cases[0].forces.size(), 1U);
    const double element = a.cases[0].gap_torque;
    differences.push_back(std::abs(element - b.cases[0].forces[0].torque.z()) /
                          std::abs(element));
  }
  EXPECT_LT(differences[0], 2e-3);
  EXPECT_LT(differences[1], differences[0] / 3);
}
