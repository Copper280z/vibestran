// tests/unit/test_drive.cpp
// DRIVE: phase currents set from the electrical angle at each MOTION
// position, THETA0 AUTO, the dq quantities, the frozen-permeability
// inductance and the back-EMF. Each guarantee is checked against what an
// analyst would run without the card, or against the analytic machine.

#include "magnetic/drive.hpp"
#include "magnetic/solver.hpp"
#include "machine_deck.hpp"
#ifdef VIBESTRAN_HAS_HDF5
#include "../common/hdf5_results.hpp"
#endif
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace vibestran::magnetic;
using namespace machine_deck;

namespace {

constexpr double degree = pi / 180;

// A three-phase machine: the rotor magnetized uniformly along magnet_angle
// (PID 2, mu_r 1) and turned by MOTION 1, a stator of air or of saturating
// iron (PID 3), and three phases in six slots of air between 26 and 34 mm.
// Phase k's axis is at beta1 + 120 (k - 1) degrees; its go slot (current
// along the plane normal) is 90 degrees ahead of the axis and its return 90
// degrees behind, so positive current makes a field along the axis. The
// phases are circuits 1, 2 and 3 of COIL2D 11 + 2k (go) and 12 + 2k
// (return).
struct ThreePhase {
  Machine g;
  // beta1 = 33.75 puts every slot's center on the center of a 7.5-degree
  // stator division, so the meshed phase axes are the nominal ones.
  double magnet_angle{25}, beta1{33.75}, turns{30}, half_width{.2};
  bool iron{false};
  std::string cards;   // MOTION, DRIVE and further bulk data
  std::string control; // case control
};
std::string three_phase_deck(ThreePhase t) {
  t.g.rotor_property = 2;
  t.g.stator_property = t.iron ? 3 : 1;
  t.g.cards = magnetized_rotor(1.0, t.magnet_angle * degree) +
              (t.iron ? iron_table + "MATMAG,3,,1\nPMAG2D,3,3\n" : "") +
              "ESET,20,PROP,2\n" + t.cards;
  auto geo = geometry(t.g);
  std::map<int, std::vector<int>> slots;
  for (auto &e : geo.elements) {
    const Vector c = std::accumulate(e.nodes.begin(), e.nodes.end(), Vector(Vector::Zero()),
                                     [&](const Vector &sum, int n) {
                                       return Vector(sum + geo.grids.at(n));
                                     }) /
                     static_cast<double>(e.nodes.size());
    if (c.norm() < .026 || c.norm() > .034)
      continue;
    const double angle = std::atan2(c.y(), c.x());
    for (int k = 0; k < 3; ++k)
      for (int side : {1, -1}) {
        const double slot = (t.beta1 + 120 * k) * degree + side * pi / 2;
        if (std::abs(std::remainder(angle - slot, 2 * pi)) < t.half_width) {
          e.property = 1;
          slots[1 + 2 * k + (side < 0)].push_back(e.id);
        }
      }
  }
  std::string winding;
  for (const auto &[set, ids] : slots) {
    winding += std::format("ESET,{}", set);
    for (int id : ids)
      winding += std::format(",{}", id);
    winding += '\n';
  }
  for (int k = 0; k < 3; ++k)
    winding += std::format("COIL2D,{},{},{:.17g}\nCOIL2D,{},{},{:.17g}\n"
                           "CIRCUIT,{},1,{},1,{},1\n",
                           11 + 2 * k, 1 + 2 * k, t.turns, 12 + 2 * k, 2 + 2 * k,
                           -t.turns, k + 1, 11 + 2 * k, 12 + 2 * k);
  auto text = deck(t.g, geo);
  text.replace(text.find("CEND\n"), 5, "CEND\n" + t.control);
  text.replace(text.find("ENDDATA\n"), 8, winding + "ENDDATA\n");
  return text;
}
Model three_phase(const ThreePhase &t) {
  auto out = parse_string(three_phase_deck(t));
  EXPECT_TRUE(std::holds_alternative<Model>(out)) << error_of(out);
  return std::holds_alternative<Model>(out) ? std::get<Model>(std::move(out)) : Model{};
}
std::string motion_card(double start, double step, int count, double speed = 0) {
  return std::format("MOTION,1,20,0,ROTATE,{:.17g},{:.17g},{}{}\n", start, step, count,
                     speed != 0 ? std::format("\n,{:.17g}", speed) : "");
}
std::string drive_card(double peak, double gamma, const std::string &theta0 = "",
                       const std::string &phases = "1,2,3", const char *wave = "SINE") {
  return std::format("DRIVE,10,{},1,{:.17g},{:.17g},{}\n,{}\n", wave, peak, gamma, theta0,
                     phases);
}
const char *driven = "SUBCASE 1\n  MOTION = 1\n  DRIVE = 10\n";
double relative(double a, double b, double scale) { return std::abs(a - b) / scale; }
// The amplitude-invariant dq transform of three phase values.
Eigen::Vector2d dq(double theta_e, const Eigen::Vector3d &abc) {
  return 2. / 3 * park_columns(3, theta_e).transpose() * abc;
}

} // namespace

TEST(Drive, PhaseCurrentsParkTransformAndFit) {
  // SINE: the Park transform of the phase currents is (-I sin gamma,
  // I cos gamma) at every electrical angle, and the currents sum to zero.
  for (int m : {3, 5}) {
    Drive d;
    d.peak = 7;
    d.gamma = 35;
    d.phases.resize(static_cast<size_t>(m));
    for (double theta : {0., .4, 2.9, -1.7}) {
      const auto i = phase_currents(d, theta);
      const Eigen::Map<const Eigen::VectorXd> abc(i.data(), m);
      const Eigen::Vector2d back = 2. / m * park_columns(m, theta).transpose() * abc;
      EXPECT_NEAR(back[0], -7 * std::sin(35 * degree), 1e-13);
      EXPECT_NEAR(back[1], 7 * std::cos(35 * degree), 1e-13);
      EXPECT_NEAR(abc.sum(), 0, 1e-13);
    }
  }
  // SIX: two phases conduct at +-I at every angle, and phase 1's
  // fundamental is (2 sqrt 3 / pi) I, in phase with SINE's.
  Drive six;
  six.six = true;
  six.peak = 2;
  six.gamma = 20;
  six.phases = {1, 2, 3};
  Drive sine = six;
  sine.six = false;
  std::vector<double> block, wave;
  const int n = 3600;
  for (int j = 0; j < n; ++j) {
    const double theta = 2 * pi * (j + .5) / n;
    const auto i = phase_currents(six, theta);
    EXPECT_EQ(std::count(i.begin(), i.end(), 0.), 1) << theta;
    EXPECT_EQ(i[0] + i[1] + i[2], 0) << theta;
    block.push_back(i[0]);
    wave.push_back(phase_currents(sine, theta)[0]);
  }
  const auto b = fundamental(block), s = fundamental(wave);
  EXPECT_NEAR(b.amplitude, 2 * std::sqrt(3.) / pi * 2, 2e-4);
  EXPECT_NEAR(std::remainder(b.phase - s.phase, 2 * pi), 0, 1e-12);
  EXPECT_NEAR(b.mean, 0, 1e-12);

  // The THETA0 fit: synthetic linkages lambda_k = L cos(p (phi - theta0) -
  // 120 (k - 1)) with a third harmonic, which 8 samples per period resolve.
  Drive d;
  d.poles = 3;
  d.phases = {4, 5, 6};
  const auto positions = alignment_positions(d);
  ASSERT_EQ(positions.size(), 8U);
  EXPECT_DOUBLE_EQ(positions[1], 15);
  auto linkages = [&](double theta0, std::array<int, 3> order) {
    std::vector<std::vector<double>> out;
    for (double phi : positions) {
      std::vector<double> row;
      for (int k : order) {
        const double theta = 3 * (phi - theta0) * degree - 2 * pi * k / 3;
        row.push_back(.02 * std::cos(theta) + .003 * std::cos(3 * theta) + .001);
      }
      out.push_back(row);
    }
    return out;
  };
  auto found = align_drive(d, linkages(97, {0, 1, 2}));
  ASSERT_TRUE(std::holds_alternative<Alignment>(found)) << error_of(found);
  const auto &a = std::get<Alignment>(found);
  EXPECT_NEAR(a.theta0, 97, 1e-12); // in [0, 120), one electrical period
  EXPECT_NEAR(a.linkage, .02, 1e-15);
  EXPECT_NEAR(a.lag[1], 120, 1e-10);
  EXPECT_NEAR(a.lag[2], 240, 1e-10);
  EXPECT_NE(error_of(align_drive(d, linkages(10, {0, 2, 1})))
                .find("the phase sequence opposes the direction of increasing position: "
                      "phase 2 (circuit 5) lags phase 1 by 240.0, expected 120.0"),
            std::string::npos);
  EXPECT_NE(error_of(align_drive(d, linkages(10, {0, 2, 1}))).find("(swap C2 and C3)"),
            std::string::npos);
  EXPECT_NE(error_of(align_drive(d, linkages(10, {0, 0, 2}))).find("the phases are not in sequence"),
            std::string::npos);
  EXPECT_NE(error_of(align_drive(d, std::vector<std::vector<double>>(8, {0., 0., 0.})))
                .find("THETA0 AUTO: phase 1 (circuit 4) links no flux that varies"),
            std::string::npos);
}

TEST(Drive, AutoFindsTheMagnetAxisAndChecksThePhaseSequence) {
  // The magnet's axis is at 25 degrees and phase 1's at 33.75, so the d
  // axis lines up with phase 1 at a rotor position of 8.75 degrees. In air
  // each phase links an exact first harmonic of the rotor position (see the
  // torque test), so the 8 samples recover it to round-off.
  ThreePhase t;
  t.cards = motion_card(0, 10, 1) + drive_card(0, 0);
  t.control = driven;
  const auto r = solved(three_phase(t));
  ASSERT_EQ(r.cases.size(), 1U);
  const auto &found = *r.cases[0].drive_summary;
  EXPECT_TRUE(found.automatic);
  EXPECT_NEAR(found.theta0, 8.75, 1e-9);
  EXPECT_NEAR(found.lag[1], 120, 1e-9);
  EXPECT_NEAR(found.lag[2], 240, 1e-9);
  EXPECT_GT(found.linkage, 0);
  // The no-load sweep runs once, reported in the header.
  const auto &rows = r.summary.solution.rows;
  EXPECT_EQ(std::count_if(rows.begin(), rows.end(),
                          [](const auto &row) {
                            return row.second.find("THETA0 AUTO with MOTION 1") !=
                                   std::string::npos;
                          }),
            1);
  // Phases 2 and 3 swapped, the sequence opposes the motion.
  t.cards = motion_card(0, 10, 1) + drive_card(0, 0, "", "1,3,2");
  EXPECT_NE(error_of(solve(three_phase(t)))
                .find("Subcase 1, DRIVE 10 THETA0 AUTO with MOTION 1: THETA0 AUTO: the "
                      "phase sequence opposes the direction of increasing position"),
            std::string::npos);
  // Without a magnet there is no d axis to find.
  t.cards = motion_card(0, 10, 1) + drive_card(0, 0) + "MATMAG,2,1.,,0.\n";
  ThreePhase plain = t;
  plain.cards = motion_card(0, 10, 1) + drive_card(0, 0);
  auto text = three_phase_deck(plain);
  text.replace(text.find("MATMAG,2,1.,,1"), 14, "MATMAG,2,1.,,0");
  EXPECT_NE(error_of(solve(std::get<Model>(parse_string(text))))
                .find("so it has no d axis to find; give THETA0"),
            std::string::npos);
}

TEST(Drive, EachPositionIsTheStaticDeckWithItsPhaseCurrents) {
  // Take DRIVE out: each position, solved as a single-position MOTION with
  // ICIRC currents equal to the DRIVE's, is the same model. THETA0 AUTO is
  // reproduced by giving the THETA0 it found.
  for (const bool saturating : {false, true}) {
    SCOPED_TRACE(saturating ? "saturating stator" : "air stator");
    ThreePhase t;
    t.iron = saturating;
    t.cards = motion_card(-10, 7.5, 4) + drive_card(20, 30);
    t.control = driven;
    const auto drive = solved(three_phase(t));
    ASSERT_EQ(drive.cases.size(), 1U);
    const auto &steps = drive.cases[0].steps;
    ASSERT_EQ(steps.size(), 4U);
    const double theta0 = drive.cases[0].drive_summary->theta0;
    const double tolerance = saturating ? 1e-6 : 1e-10;

    // The same sweep with the THETA0 found.
    t.cards = motion_card(-10, 7.5, 4) + drive_card(20, 30, std::format("{:.17g}", theta0));
    const auto given = solved(three_phase(t));
    EXPECT_FALSE(given.cases[0].drive_summary->automatic);
    for (size_t k = 0; k < 4; ++k)
      EXPECT_LT(relative(given.cases[0].steps[k].energy, steps[k].energy, steps[k].energy),
                tolerance);

    // Each position as a static deck: MOTION k + 2 holds its one position,
    // and LOAD 100 + k gives the phases the DRIVE's currents.
    std::string cards = motion_card(-10, 7.5, 4), control;
    for (size_t k = 0; k < 4; ++k) {
      cards += std::format("MOTION,{},20,0,ROTATE,{:.17g}\n", k + 2, steps[k].position);
      for (int c = 0; c < 3; ++c)
        cards += std::format("ICIRC,{},{},{:.17g}\n", 100 + k, c + 1,
                             steps[k].circuits[c].current);
      control += std::format("SUBCASE {}\n  MOTION = {}\n  LOAD = {}\n", k + 1, k + 2, 100 + k);
    }
    t.cards = cards;
    t.control = control;
    const auto plain = solved(three_phase(t));
    ASSERT_EQ(plain.cases.size(), 4U);
    const double torque_scale =
        std::accumulate(steps.begin(), steps.end(), 0., [](double most, const CaseResults &s) {
          return std::max(most, std::abs(s.torque));
        });
    for (size_t k = 0; k < 4; ++k) {
      SCOPED_TRACE(std::format("{} degrees", steps[k].position));
      const auto &a = steps[k];
      const auto &b = plain.cases[k].steps[0];
      EXPECT_LT(relative(a.energy, b.energy, a.energy), tolerance);
      EXPECT_LT(relative(a.coenergy, b.coenergy, a.coenergy), tolerance);
      EXPECT_LT(relative(a.torque, b.torque, torque_scale), tolerance);
      for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(a.circuits[c].current, b.circuits[c].current);
        EXPECT_LT(relative(a.circuits[c].linkage, b.circuits[c].linkage, 1e-3), tolerance);
        EXPECT_LT(relative(a.linkage_rate[c], b.linkage_rate[c],
                           a.linkage_rate.cwiseAbs().maxCoeff()),
                  tolerance);
        for (int d = 0; d < 3; ++d)
          EXPECT_LT(relative(a.circuit_inductance(c, d), b.circuit_inductance(c, d),
                             a.circuit_inductance(c, c)),
                    tolerance * 10);
      }
      // The dq values are the Park transform of the phase values.
      const double theta = a.drive->electrical * degree;
      EXPECT_NEAR(theta, (a.position - theta0) * degree, 1e-12);
      const Eigen::Vector3d linkage(a.circuits[0].linkage, a.circuits[1].linkage,
                                    a.circuits[2].linkage);
      EXPECT_LT((dq(theta, linkage) - a.drive->linkage).norm(), 1e-15);
    }
  }
}

TEST(Drive, TorqueIsTheDqTorqueOfASinusoidalMachine) {
  // In air the discrete machine is exactly sinusoidal. The rotor is a
  // uniformly magnetized 48-gon of unit permeability: its trace on the gap
  // circle has no harmonics between 2 and NHARM = 24, so through the AGE
  // each phase links an exact first harmonic of the position, and the
  // inductances do not depend on it. So, to round-off: the AGE torque is
  // (3/2) p (lambda_d i_q - lambda_q i_d) at every position and constant,
  // (3/2) p lambda_pm I cos(gamma); L_d = L_q; the magnet links lambda_pm
  // on d; and the back-EMF per unit speed is -p lambda_pm sin(theta_e).
  // (Measured about 1e-13.)
  for (const double gamma : {0., 30., 90.}) {
    SCOPED_TRACE(std::format("gamma {}", gamma));
    ThreePhase t;
    t.cards = motion_card(0, 30, 12) + drive_card(20, gamma);
    t.control = driven;
    const auto r = solved(three_phase(t));
    const auto &c = r.cases[0];
    const double magnet = c.drive_summary->linkage,
                 expected = 1.5 * magnet * 20 * std::cos(gamma * degree),
                 scale = 1.5 * magnet * 20;
    for (const auto &step : c.steps) {
      const auto &d = *step.drive;
      SCOPED_TRACE(std::format("{} degrees", step.position));
      EXPECT_LT(relative(step.torque,
                         1.5 * (d.linkage[0] * d.current[1] - d.linkage[1] * d.current[0]),
                         scale),
                1e-10);
      EXPECT_LT(relative(step.torque, expected, scale), 1e-10);
      EXPECT_NEAR(d.current[0], -20 * std::sin(gamma * degree), 1e-12);
      EXPECT_NEAR(d.current[1], 20 * std::cos(gamma * degree), 1e-12);
      // Non-salient, so L_d = L_q, and linear, so the frozen permeability is
      // the incremental one and the magnet linkage is lambda_pm on d.
      EXPECT_LT(relative(d.incremental(0, 0), d.incremental(1, 1), d.incremental(0, 0)), 1e-10);
      EXPECT_LT((d.apparent - d.incremental).norm(), 1e-10 * d.incremental.norm());
      EXPECT_LT(relative(d.magnet[0], magnet, magnet), 1e-10);
      EXPECT_LT(std::abs(d.magnet[1]), 1e-10 * magnet);
      // The back-EMF per unit speed: d(lambda_pm cos theta_e)/dphi.
      const double theta = d.electrical * degree;
      EXPECT_LT(relative(step.linkage_rate[0], -magnet * std::sin(theta), magnet), 1e-10);
    }
  }
}

TEST(Drive, SixStepCurrentsKeepTheDqTorque) {
  // Six-step currents in the same exactly sinusoidal machine: two phases
  // carry +-I at each position, and the AGE torque is still
  // (3/2) p (lambda_d i_q - lambda_q i_d). With L_d = L_q that is
  // (3/2) p lambda_pm i_q, with i_q the Park transform of the six-step
  // currents, which ripples over each 60-degree sector. No sample falls on
  // a switching angle.
  ThreePhase t;
  t.cards = motion_card(0, 5, 72) + drive_card(20, 0, "", "1,2,3", "SIX");
  t.control = driven;
  const auto r = solved(three_phase(t));
  const auto &c = r.cases[0];
  const auto &drive = r.cases[0].drive_summary;
  const double scale = 1.5 * drive->linkage * 20;
  Drive six;
  six.six = true;
  six.peak = 20;
  six.phases = {1, 2, 3};
  for (const auto &step : c.steps) {
    const auto &d = *step.drive;
    SCOPED_TRACE(std::format("{} degrees", step.position));
    int conducting = 0;
    for (int k = 0; k < 3; ++k)
      if (step.circuits[k].current != 0) {
        EXPECT_EQ(std::abs(step.circuits[k].current), 20);
        ++conducting;
      }
    EXPECT_EQ(conducting, 2);
    EXPECT_LT(relative(step.torque,
                       1.5 * (d.linkage[0] * d.current[1] - d.linkage[1] * d.current[0]),
                       scale),
              1e-10);
    const double theta = d.electrical * degree;
    const auto currents = phase_currents(six, theta);
    const double iq = dq(theta, Eigen::Vector3d(currents[0], currents[1], currents[2]))[1];
    EXPECT_LT(relative(step.torque, 1.5 * drive->linkage * iq, scale), 1e-10);
  }
}

TEST(Drive, BackEmfIsTheDerivativeOfTheLinkage) {
  // dlambda/dphi at fixed current comes from dT/di by reciprocity; check it
  // against a central difference of the linkage, and the static subcase's
  // value, about the plane normal, against the step's. The B-H table is
  // piecewise linear, so the linkage's derivative kinks wherever an element
  // crosses a breakpoint and a central difference is only first order
  // there: the saturated check uses a small step and a tight MAGTOL.
  for (const bool saturating : {false, true}) {
    SCOPED_TRACE(saturating ? "saturating stator" : "air stator");
    const double h = saturating ? .01 : .05;
    ThreePhase t;
    t.iron = saturating;
    t.cards = motion_card(17 - h, h, 3) + "ICIRC,5,1,25.\nICIRC,5,2,-8.\nICIRC,5,3,-17.\n" +
              (saturating ? "PARAM,MAGTOL,1e-12\n" : "");
    t.control = "LOAD = 5\nSUBCASE 1\n  MOTION = 1\n";
    const auto r = solved(three_phase(t));
    const auto &s = r.cases[0].steps;
    ASSERT_EQ(s.size(), 3U);
    const double scale = s[1].linkage_rate.cwiseAbs().maxCoeff();
    for (int c = 0; c < 3; ++c) {
      const double difference =
          (s[2].circuits[c].linkage - s[0].circuits[c].linkage) / (2 * h * degree);
      EXPECT_LT(relative(s[1].linkage_rate[c], difference, scale), saturating ? 1e-7 : 1e-6)
          << c;
    }
    // With the rotor inside the gap and the axis along the plane normal, a
    // static subcase's rate is the step's at 0 degrees.
    t.cards = motion_card(0, 1, 1) + "ICIRC,5,1,25.\nICIRC,5,2,-8.\nICIRC,5,3,-17.\n";
    t.control = "LOAD = 5\nSUBCASE 1\n  MOTION = 1\nSUBCASE 2\n";
    const auto both = solved(three_phase(t));
    for (int c = 0; c < 3; ++c)
      EXPECT_LT(relative(both.cases[1].linkage_rate[c], both.cases[0].steps[0].linkage_rate[c],
                         scale),
                1e-12);
  }
}

TEST(Drive, ApparentInductanceIsTheFrozenPermeabilityDeck) {
  // LDQ = APPARENT is the inductance of a linear deck: every iron element
  // given its own MATMAG at the loaded secant permeability |B| / |H|, the
  // magnets off. Triangles keep B uniform in each element, so the deck
  // reproduces the frozen permeability exactly. With the magnets on and the
  // DRIVE currents, that deck is the loaded state itself.
  ThreePhase t;
  t.iron = true;
  t.g.triangles = true;
  t.cards = motion_card(0, 1, 1) + drive_card(40, 45);
  t.control = driven;
  const auto m = three_phase(t);
  const auto r = solved(m);
  const auto &step = r.cases[0].steps[0];
  const auto &d = *step.drive;
  const double theta = d.electrical * degree;
  ASSERT_TRUE(step.extrapolated || d.incremental(0, 0) < .99 * d.apparent(0, 0))
      << "the iron must saturate for the comparison to mean anything";

  Model frozen = m;
  frozen.drives.clear();
  const Material &iron = m.materials.at(3);
  const auto &curve = m.bh_tables.at(iron.bh_table);
  for (auto &e : frozen.elements) {
    if (e.property != 3)
      continue;
    const Vector b = step.b[static_cast<size_t>(r.mesh.element_index.at(e.id))];
    const double h = material_response(iron, curve, b, Vector::Zero()).h.norm();
    frozen.materials[1000 + e.id] = Material{b.norm() / (mu0 * h), 0, 0};
    frozen.properties[1000 + e.id] = Property{1000 + e.id, 0, 1, false};
    e.property = 1000 + e.id;
  }
  auto with_currents = [&](bool magnets, std::array<double, 3> currents) {
    Model out = frozen;
    out.materials.at(2).br = magnets ? 1.0 : 0;
    out.currents.clear();
    for (int c = 0; c < 3; ++c)
      out.currents.push_back({7, c + 1, currents[c]});
    Subcase s;
    s.load = 7;
    s.motion = 1;
    out.subcases = {s};
    const auto solved_state = solved(out);
    Eigen::Vector3d linkage;
    for (int c = 0; c < 3; ++c)
      linkage[c] = solved_state.cases[0].steps[0].circuits[c].linkage;
    return linkage;
  };
  Eigen::Matrix3d inductance;
  for (int j = 0; j < 3; ++j) {
    std::array<double, 3> unit{0, 0, 0};
    unit[j] = 1;
    inductance.col(j) = with_currents(false, unit);
  }
  const Eigen::MatrixXd c = park_columns(3, theta);
  const Eigen::Matrix2d apparent = 2. / 3 * c.transpose() * inductance * c;
  EXPECT_LT((apparent - d.apparent).norm(), 1e-9 * d.apparent.norm());
  const std::array<double, 3> currents{step.circuits[0].current, step.circuits[1].current,
                                       step.circuits[2].current};
  const Eigen::Vector3d loaded(step.circuits[0].linkage, step.circuits[1].linkage,
                               step.circuits[2].linkage);
  // Measured 1e-10 and 7e-11: the frozen deck's own Newton tolerance.
  EXPECT_LT((with_currents(true, currents) - loaded).norm(), 1e-8 * loaded.norm());
  EXPECT_LT((dq(theta, with_currents(true, {0, 0, 0})) - d.magnet).norm(),
            1e-8 * d.magnet.norm());
}

TEST(Drive, CardsAndCaseControl) {
  ThreePhase t;
  t.cards = motion_card(0, 15, 24, 3000) + drive_card(20, 30, "12.5", "1,2,3", "SIX");
  t.control = "LDQ = INCREMENTAL\nSUBCASE 1\n  MOTION = 1\n  DRIVE = 10\nSUBCASE 2\n"
              "  MOTION = 1\n  DRIVE = 10\n  LDQ = APPARENT\n";
  const auto m = three_phase(t);
  const auto &drive = m.drives.at(10);
  EXPECT_TRUE(drive.six);
  EXPECT_EQ(drive.poles, 1);
  EXPECT_EQ(drive.peak, 20);
  EXPECT_EQ(drive.gamma, 30);
  ASSERT_TRUE(drive.theta0.has_value());
  EXPECT_EQ(*drive.theta0, 12.5);
  EXPECT_EQ(drive.phases, (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(m.motions.at(1).speed, 3000);
  ASSERT_EQ(m.subcases.size(), 2U);
  EXPECT_EQ(m.subcases[0].drive, 10);
  EXPECT_FALSE(m.subcases[0].apparent);
  EXPECT_TRUE(m.subcases[1].apparent);
  t.cards = motion_card(0, 15, 24) + "DRIVE,10,SINE,1,20.,30.,AUTO\n,1,2,3\n";
  EXPECT_FALSE(three_phase(t).drives.at(10).theta0.has_value());

  auto error = [&](const std::string &cards, const std::string &control = driven) {
    ThreePhase bad;
    bad.cards = motion_card(0, 15, 2) + "ICIRC,5,1,3.\n" + cards;
    bad.control = control;
    return error_of(parse_string(three_phase_deck(bad)));
  };
  EXPECT_NE(error("DRIVE,10,SQUARE,1,1.\n,1,2,3\n").find("DRIVE WAVE must be SINE or SIX"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SINE,0,1.\n,1,2,3\n").find("DRIVE 10 requires POLES >= 1"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SINE,1,-1.\n,1,2,3\n").find("DRIVE 10 requires IPEAK >= 0"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SINE,1,1.\n,1,2\n").find("DRIVE 10 lists 2 phase circuits; it "
                                                     "needs at least 3"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SIX,1,1.\n,1,2,3,11\n")
                .find("DRIVE 10 SIX is the three-phase six-step waveform and lists 4"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SINE,1,1.\n,1,2,2\n").find("DRIVE 10 lists circuit 2 twice"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SINE,1,1.\n,1,2,11\n")
                .find("DRIVE 10 phase 11 is a source that belongs to a CIRCUIT"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SINE,1,1.\n,1,2,4\n")
                .find("DRIVE 10 references undefined circuit or source 4"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SINE,1,1.,,,7\n,1,2,3\n")
                .find("DRIVE fields 7 and 8 must be blank"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SINE,1,1.\n,1,2,3\n", "SUBCASE 1\n  DRIVE = 10\n")
                .find("Subcase 1 DRIVE 10 needs MOTION in the same subcase"),
            std::string::npos);
  EXPECT_NE(error("", "SUBCASE 1\n  MOTION = 1\n  DRIVE = 11\n")
                .find("Subcase 1 references undefined DRIVE 11"),
            std::string::npos);
  EXPECT_NE(error("DRIVE,10,SINE,1,1.\n,1,2,3\n",
                  std::string(driven) + "  LOAD = 5\n")
                .find("Subcase 1: circuit 1 is a phase of DRIVE 10 and also has an ICIRC "
                      "current in LOAD 5"),
            std::string::npos);
  EXPECT_NE(error("", "LDQ = SECANT\n").find("LDQ must be APPARENT or INCREMENTAL"),
            std::string::npos);
}

TEST(Drive, ResultsReportTheDqQuantitiesAndTheBackEmf) {
  // No load over one electrical period at 3000 rpm: the back-EMF is
  // -omega lambda_pm sin(theta_e) to round-off (see the torque test), so its
  // fundamental constant is lambda_pm (one pole pair) with no distortion.
  ThreePhase t;
  t.cards = motion_card(0, 15, 24, 3000) + drive_card(0, 0);
  t.control = "MSTEP = ALL\nSUBCASE 1\n  MOTION = 1\n  DRIVE = 10\n  LDQ = INCREMENTAL\n";
  const auto m = three_phase(t);
  const auto r = solved(m);
  const auto &c = r.cases[0];
  ASSERT_EQ(c.steps.size(), 24U);
  for (const auto &step : c.steps)
    EXPECT_FALSE(std::isfinite(step.drive->apparent(0, 0)));
#ifdef VIBESTRAN_HAS_HDF5
  using namespace vibestran::test;
  const auto directory = std::filesystem::temp_directory_path() / "vibestran_drive";
  std::filesystem::create_directories(directory);
  ASSERT_FALSE(write_results(m, r, directory / "drive.F06").has_value());
  const auto file = directory / "drive.mag.vtkhdf";
  const std::string group = "/MAGSTAT/subcases/1/drive";
  EXPECT_EQ(h5_attribute(file, group, "theta0_auto"), 1);
  EXPECT_EQ(h5_attribute(file, group, "theta0_deg"), c.drive_summary->theta0);
  EXPECT_EQ(h5_attribute(file, group, "electrical_periods"), 1);
  EXPECT_EQ(h5_attribute(file, "/MAGSTAT/subcases/1", "speed_rpm"), 3000);
  // No current anywhere: cogging, so no ripple percentage of a zero mean.
  EXPECT_EQ(h5_attribute(file, "/MAGSTAT/subcases/1", "current_free"), 1);
  EXPECT_TRUE(std::isnan(h5_attribute(file, "/MAGSTAT/subcases/1", "torque_ripple_percent")));
  const auto constant = h5_table(file, group + "/emf_constant_fundamental_V_s_per_rad");
  const auto distortion = h5_table(file, group + "/emf_thd_percent");
  ASSERT_EQ(constant.values.size(), 3U);
  for (int k = 0; k < 3; ++k) {
    EXPECT_LT(relative(constant.values[k], c.drive_summary->linkage,
                       c.drive_summary->linkage),
              1e-10);
    EXPECT_LT(distortion.values[k], 1e-8); // percent
  }
  EXPECT_FALSE(h5_exists(file, group + "/L_dq_apparent_H"));
  EXPECT_EQ(h5_table(file, group + "/L_dq_incremental_H").dims,
            (std::vector<hsize_t>{24, 2, 2}));
  const auto angle = h5_table(file, group + "/theta_e_deg");
  const auto emf = h5_table(file, "/MAGSTAT/subcases/1/motion/emf_V");
  ASSERT_EQ(emf.dims, (std::vector<hsize_t>{24, 3}));
  const double omega = 3000 * 2 * pi / 60;
  for (size_t k = 0; k < 24; ++k) {
    EXPECT_EQ(angle.values[k], c.steps[k].drive->electrical);
    EXPECT_EQ(emf.values[3 * k], c.steps[k].linkage_rate[0] * omega);
  }
  std::ifstream in(directory / "drive.F06");
  std::stringstream f06;
  f06 << in.rdbuf();
  const auto text = f06.str();
  EXPECT_NE(text.find("DRIVE 10: sinusoidal currents, 0 A peak"), std::string::npos);
  EXPECT_NE(text.find("Phase 1 (circuit 1) back-EMF over 1 electrical period: fundamental "
                      "constant (V s/rad, peak)"),
            std::string::npos);
  EXPECT_NE(text.find("STEP   THETA_E(deg)"), std::string::npos);
  EXPECT_NE(text.find("ripple undefined (no current: cogging)"), std::string::npos);
  EXPECT_NE(text.find("EMF      1(V)"), std::string::npos);
#endif
}
