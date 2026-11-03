// tests/unit/test_line_drive.cpp
// DRIVE along a translation: the phase currents of a linear machine set from
// the electrical angle pi (x - x0) / tau at each MOTION TRANSLATE position,
// tau the pole pitch in DRIVE field 4. On the tubular and flat stacks each
// position is checked against the static deck with the mover moved by hand
// and its phase currents given by ICIRC, THETA0 AUTO against the magnets
// moved by an alignment step and the travel reversed, the thrust against
// the slope of the coenergy less the linkages' work along the drive, a pole
// pitch against reversed magnets and currents, and the results' units.

#include "magnetic/drive.hpp"
#include "tube_deck.hpp"
#ifdef VIBESTRAN_HAS_HDF5
#include "../common/hdf5_results.hpp"
#endif
#include <gtest/gtest.h>
#include <complex>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace tube_deck;

namespace {
// travelling's deck with DRIVE 10 on the phase circuits at the stack's pole
// pitch: theta0 in m, blank for AUTO.
struct Driven {
  double peak{0}, gamma{0};
  std::string theta0, phases{"1,2,3"};
  const char *wave{"SINE"};
};
Axi driven(const Tube &t, const Travel &v, const Driven &d) {
  Axi g = travelling(t, v);
  g.cards += std::format("DRIVE,10,{},{:.17g},{:.17g},{:.17g},{}\n,{}\n", d.wave, pitch,
                         d.peak, d.gamma, d.theta0, d.phases);
  g.control += "DRIVE = 10\n";
  return g;
}
// The sinusoidal phase currents at electrical angle theta (radians).
std::array<double, 3> sine_currents(double peak, double gamma_deg, double theta) {
  const double gamma = gamma_deg * pi / 180, id = -peak * std::sin(gamma),
               iq = peak * std::cos(gamma);
  std::array<double, 3> out{};
  for (int k = 0; k < 3; ++k) {
    const double t = theta - 2 * pi * k / 3;
    out[static_cast<size_t>(k)] = id * std::cos(t) - iq * std::sin(t);
  }
  return out;
}
// The amplitude-invariant dq transform of three phase values.
Eigen::Vector2d dq(double theta, const std::array<double, 3> &abc) {
  return 2. / 3 * park_columns(3, theta).transpose() * Eigen::Vector3d(abc[0], abc[1], abc[2]);
}
} // namespace

TEST(LineDrive, EachPositionIsTheStaticDeckWithItsPhaseCurrents) {
  // Take DRIVE out: each position is the static deck with the mover moved
  // there by hand and the phase currents of theta_e = pi (x - x0) / tau set
  // by ICIRC. The same model, so energy, coenergy, thrust and linkages agree
  // to round-off, and the dq currents and linkages are the Park transform
  // of the static deck's. Positions off the rows, the first before x0.
  constexpr double x0 = .0123, peak = 4, gamma = 20;
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    const Tube base{.offset = 5, .layers = 0, .gap = true, .planar = planar};
    const Travel v{.start = .37 * pitch / 48, .step = 13.3 * pitch / 48, .count = 4};
    const auto c = only(driven(base, v, {.peak = peak, .gamma = gamma, .theta0 = ".0123"}));
    ASSERT_EQ(c.steps.size(), 4U);
    ASSERT_TRUE(c.drive_summary.has_value());
    EXPECT_FALSE(c.drive_summary->automatic);
    EXPECT_EQ(c.drive_summary->theta0, x0);
    for (size_t k = 0; k < 4; ++k) {
      const double x = v.start + static_cast<double>(k) * v.step, theta = pi * (x - x0) / pitch;
      SCOPED_TRACE(x);
      const auto currents = sine_currents(peak, gamma, theta);
      Tube by_hand = base;
      by_hand.move = x;
      Axi g = tube(by_hand);
      g.control += "LOAD = 5\n";
      for (int i = 0; i < 3; ++i)
        g.cards += std::format("ICIRC,5,{},{:.17g}\n", i + 1, currents[static_cast<size_t>(i)]);
      const auto s = only(g);
      const auto &m = c.steps[k];
      ASSERT_TRUE(m.drive.has_value());
      EXPECT_NEAR(m.drive->electrical, theta * 180 / pi, 1e-12);
      EXPECT_NEAR(m.coenergy, s.coenergy, 1e-11 * std::abs(s.coenergy));
      EXPECT_NEAR(m.energy, s.energy, 1e-11 * std::abs(s.energy));
      EXPECT_GT(std::abs(s.gap_force), 1.);
      EXPECT_NEAR(m.torque, s.gap_force, 1e-9 * std::abs(s.gap_force));
      std::array<double, 3> linkages{};
      for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(m.circuits[i].current, currents[i], 1e-12);
        EXPECT_NEAR(m.circuits[i].linkage, s.circuits[i].linkage,
                    1e-10 * std::abs(s.circuits[0].linkage));
        linkages[i] = s.circuits[i].linkage;
      }
      EXPECT_LT((m.drive->current - dq(theta, currents)).norm(), 1e-12 * peak);
      EXPECT_LT((m.drive->linkage - dq(theta, linkages)).norm(),
                1e-10 * std::abs(s.circuits[0].linkage));
      // i_d = -I sin(gamma), i_q = I cos(gamma).
      EXPECT_NEAR(m.drive->current[0], -peak * std::sin(gamma * pi / 180), 1e-12);
      EXPECT_NEAR(m.drive->current[1], peak * std::cos(gamma * pi / 180), 1e-12);
    }
  }
}

TEST(LineDrive, AutoFollowsTheMagnetsAndChecksThePhaseSequence) {
  // THETA0 AUTO fits phase 1's no-load linkage at 8 positions over two pole
  // pitches. Starting the magnets one alignment step (12 rows, 5 mm)
  // further along +z shifts each phase's samples by one, cyclically over the
  // period, so the mover reaches the alignment exactly 5 mm sooner: x0 falls
  // by 5 mm (modulo the period). The coils are 120 electrical degrees apart
  // (measured 120.02 and 240.02 from the 8 samples), and
  // travelling the other way (the CID's x axis along basic -z) reverses the
  // phase sequence, which AUTO reports; so does listing B and C swapped.
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    Tube t{.offset = 5, .layers = 0, .gap = true, .planar = planar};
    const auto first = only(driven(t, {.count = 1}, {}));
    t.offset = 17;
    const auto later = only(driven(t, {.count = 1}, {}));
    ASSERT_TRUE(first.drive_summary.has_value());
    ASSERT_TRUE(later.drive_summary.has_value());
    const auto &a = *first.drive_summary, &b = *later.drive_summary;
    EXPECT_TRUE(a.automatic);
    EXPECT_GE(a.theta0, 0);
    EXPECT_LT(a.theta0, 2 * pitch);
    EXPECT_NEAR(std::remainder(b.theta0 - a.theta0 + .005, 2 * pitch), 0, 1e-12);
    EXPECT_NEAR(b.linkage, a.linkage, 1e-12 * a.linkage);
    EXPECT_NEAR(a.lag[1], 120, .5);
    EXPECT_NEAR(a.lag[2], 240, .5);
    t.offset = 5;
    for (const auto &[deck, expected] :
         {std::pair{driven(t, {.count = 1, .reversed = true}, {}),
                    "THETA0 AUTO: the phase sequence opposes the direction of increasing "
                    "position"},
          std::pair{driven(t, {.count = 1}, {.phases = "1,3,2"}),
                    "THETA0 AUTO: the phase sequence opposes the direction of increasing "
                    "position"}}) {
      const auto e = error_of(solve(parsed(deck)));
      EXPECT_NE(e.find("Subcase 1, DRIVE 10 THETA0 AUTO with MOTION 1: "), std::string::npos) << e;
      EXPECT_NE(e.find(expected), std::string::npos) << e;
    }
  }
}

TEST(LineDrive, ThrustIsTheCoenergysSlopeLessTheLinkagesWorkAlongTheDrive) {
  // Along a DRIVE sweep the currents change with the position, so the
  // coenergy's slope is the thrust plus the linkages' work on the changing
  // currents: dW'/dx = F + sum lambda_k di_k/dx, di_k/dx carrying the
  // (pi / tau) of theta_e. Central differences over the sweep's own
  // positions -dx, 0, +dx, SINE and SIX (which changes no current within
  // the sweep), for the mover moving along +z and along -z (THETA0 given:
  // AUTO would find the sequence reversed).
  constexpr double dx = 1e-6;
  for (bool planar : {false, true})
    for (const char *wave : {"SINE", "SIX"})
      for (bool reversed : {false, true}) {
        SCOPED_TRACE(std::format("{}, {}, along {}", planar ? "strip" : "tube", wave,
                                 reversed ? "-z" : "+z"));
        const Tube t{.offset = 5, .layers = 0, .gap = true, .planar = planar};
        const double at = (reversed ? -1 : 1) * 7.3 * pitch / 48;
        const auto c = only(driven(t, {.start = at - dx, .step = dx, .count = 3, .reversed = reversed},
                                   {.peak = 4, .gamma = 30, .theta0 = ".0031", .wave = wave}));
        ASSERT_EQ(c.steps.size(), 3U);
        const auto &m = c.steps[1];
        double work = 0;
        for (size_t k = 0; k < 3; ++k)
          work += m.circuits[k].linkage *
                  (c.steps[2].circuits[k].current - c.steps[0].circuits[k].current) / (2 * dx);
        const double slope = (c.steps[2].coenergy - c.steps[0].coenergy) / (2 * dx);
        EXPECT_GT(std::abs(m.torque), 1.);
        if (std::string(wave) == "SINE")
          EXPECT_GT(std::abs(work), .1 * std::abs(m.torque));
        EXPECT_NEAR(m.torque, slope - work, 1e-6 * std::abs(m.torque));
      }
}

TEST(LineDrive, APolePitchReversesTheCurrentsAndTheMagnets) {
  // Moving the mover one pole pitch of an anti-periodic stack reverses its
  // magnets under the coils, and advances theta_e by pi, which reverses
  // every phase current: the field changes sign, so thrust and coenergy
  // repeat, each linkage reverses, and the dq currents and linkages (each
  // the Park transform at an angle advanced by pi of reversed phase
  // quantities) repeat. A slotted stator, so that the thrust ripples.
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    const Tube t{.offset = 5, .layers = 0, .gap = true, .planar = planar, .teeth = 1};
    const auto c = only(driven(t, {.start = 3.1 * pitch / 48, .step = pitch, .count = 2},
                               {.peak = 4, .gamma = 15, .theta0 = ".0062"}));
    ASSERT_EQ(c.steps.size(), 2U);
    const auto &a = c.steps[0], &b = c.steps[1];
    EXPECT_GT(std::abs(a.torque), 1.);
    EXPECT_NEAR(b.torque, a.torque, 1e-10 * std::abs(a.torque));
    EXPECT_NEAR(b.coenergy, a.coenergy, 1e-11 * std::abs(a.coenergy));
    EXPECT_NEAR(b.drive->electrical - a.drive->electrical, 180, 1e-10);
    for (size_t k = 0; k < 3; ++k) {
      EXPECT_NEAR(b.circuits[k].current, -a.circuits[k].current, 1e-12);
      EXPECT_NEAR(b.circuits[k].linkage, -a.circuits[k].linkage,
                  1e-10 * std::abs(a.circuits[0].linkage));
    }
    EXPECT_LT((b.drive->current - a.drive->current).norm(), 1e-12);
    EXPECT_LT((b.drive->linkage - a.drive->linkage).norm(), 1e-10 * a.drive->linkage.norm());
  }
}

TEST(LineDrive, ResultsAreInMetresAndVoltsPerMetrePerSecond) {
  // One electrical period, two pole pitches, in 16 steps at 1.5 m/s: the
  // back-EMF's fundamental is the first harmonic of dlambda/dx over the
  // sweep, in V s/m, and in volts RMS at the speed.
  const Tube t{.offset = 5, .layers = 0, .gap = true};
  const double step = 2 * pitch / 16;
  const auto m = parsed(driven(t, {.step = step, .count = 16, .speed = 1.5}, {.peak = 2}));
  const auto results = solved(m);
  ASSERT_EQ(results.cases.size(), 1U);
  const auto &c = results.cases[0];
  ASSERT_EQ(c.steps.size(), 16U);
  std::complex<double> sum = 0;
  for (size_t k = 0; k < 16; ++k)
    sum += c.steps[k].linkage_rate[0] * std::polar(1., -2 * pi * static_cast<double>(k) / 16);
  const double fundamental = 2 * std::abs(sum) / 16;
  EXPECT_GT(fundamental, 0);
  const auto &rows = results.summary.solution.rows;
  const auto row = [&](const std::string &text) {
    return std::any_of(rows.begin(), rows.end(), [&](const auto &r) {
      return r.second.find(text) != std::string::npos;
    });
  };
  EXPECT_TRUE(row("DRIVE 10: sinusoidal currents, pole pitch 0.02 m, 2 A peak"));
  EXPECT_TRUE(row("THETA0 AUTO"));
  EXPECT_TRUE(row("DRIVE 10 THETA0 AUTO with MOTION 1 and MAGBC 1: no-load linkage at 8 "
                  "positions from 0 to 0.035 m"));
#ifdef VIBESTRAN_HAS_HDF5
  const auto directory = std::filesystem::temp_directory_path() / "vibestran_line_drive";
  std::filesystem::create_directories(directory);
  ASSERT_FALSE(write_results(m, results, directory / "tube.F06").has_value());
  const auto file = directory / "tube.mag.vtkhdf";
  using vibestran::test::h5_attribute, vibestran::test::h5_exists, vibestran::test::h5_table;
  const std::string drive = "/MAGSTAT/subcases/1/drive";
  EXPECT_NEAR(h5_attribute(file, drive, "pole_pitch_m"), pitch, 0);
  EXPECT_NEAR(h5_attribute(file, drive, "theta0_m"), c.drive_summary->theta0, 0);
  EXPECT_FALSE(h5_exists(file, drive + "/pole_pairs"));
  EXPECT_NEAR(h5_attribute(file, drive, "electrical_periods"), 1, 0);
  const auto constant = h5_table(file, drive + "/emf_constant_fundamental_V_s_per_m");
  ASSERT_EQ(constant.values.size(), 3U);
  EXPECT_NEAR(constant.at(0), fundamental, 1e-12 * fundamental);
  std::ifstream in(directory / "tube.F06");
  std::stringstream f06;
  f06 << in.rdbuf();
  for (const std::string text :
       {std::string("DRIVE 10: sinusoidal currents, 2 A peak, gamma 0 electrical degrees, "
                    "pole pitch 0.02 m, THETA0 "),
        std::string("back-EMF over 1 electrical period: fundamental constant (V s/m, peak) "),
        std::string("fundamental RMS at 1.5 m/s (V) ")})
    EXPECT_NE(f06.str().find(text), std::string::npos) << text << "\n" << f06.str();
  std::filesystem::remove_all(directory);
#else
  GTEST_SKIP() << "MAGSTAT results need HDF5";
#endif
}
