// tests/unit/test_line_motion.cpp
// MOTION TRANSLATE: a part sliding along the strip or tube air gap of a
// model that repeats along a line. Each position is checked against the
// static deck with the part moved by hand, the thrust and back-EMF against
// the slopes of the coenergy and linkage over the positions, moving the
// stator against moving the mover back, a pole pitch of an anti-periodic
// stack against reversed magnets, a sliding band along the line (fixed in
// space, converging to the element), and the results' units and the errors.

#include "tube_deck.hpp"
#ifdef VIBESTRAN_HAS_HDF5
#include "../common/hdf5_results.hpp"
#endif
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

using namespace tube_deck;

TEST(LineMotion, TransientAntiPeriodicPitchMatchesTheFullPeriod) {
  // The sleeve crosses the pitch boundary, while the magnet edges are inside
  // and across it. Compare current return and history with two explicit
  // pitches, which require only ordinary periodicity at their ends.
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "flat" : "tubular");
    const double speed = planar ? -2. : 2.;
    auto run = [&](int pitches) {
      const Tube tube{.pitches = pitches, .rows = 12, .offset = 2, .layers = 0,
                      .gap = true, .planar = planar, .teeth = 1};
      auto m = parsed(travelling(tube, {.start = .001}));
      m.options.transient = true;
      m.motions[1].speed = speed;
      m.subcases[0].time_step = 1;
      m.subcases[0].field.enabled = false;
      m.time_steps[1] = {.maximum_dt = .00061, .max_cycles = 6};
      m.conductivity = {{2, 5e4}, {3, 6e5}};
      return solved(m);
    };
    const auto half = run(1), whole = run(2);
    ASSERT_EQ(half.cases.size(), 1U);
    ASSERT_EQ(whole.cases.size(), 1U);
    const auto &a = half.cases[0], &b = whole.cases[0];
    ASSERT_TRUE(a.transient->converged) << a.failure;
    ASSERT_TRUE(b.transient->converged) << b.failure;
    const auto &t = *a.transient;
    ASSERT_EQ(t.steps_per_cycle, b.transient->steps_per_cycle);
    EXPECT_NEAR(t.dt * t.steps_per_cycle, 2 * pitch / std::abs(speed), 1e-15);
    EXPECT_GT(t.mean_loss, 0.);
    EXPECT_LT(t.mean_load * speed, 0.);
    EXPECT_NEAR(t.mean_loss, b.transient->mean_loss, 1e-7 * t.mean_loss);
    EXPECT_NEAR(t.mean_load, b.transient->mean_load, 1e-7 * std::abs(t.mean_load));
    const auto count = static_cast<size_t>(t.steps_per_cycle);
    for (size_t k = 1; k <= count; ++k)
      EXPECT_NEAR(a.steps[a.steps.size() - k].torque, b.steps[b.steps.size() - k].torque,
                  1e-7 * t.peak_to_peak + 1e-10);
  }
}

TEST(LineMotion, TranslatingTubularSleeveApproachesTheWeakReactionLoss) {
  // A uniform tubular sleeve moves through the fixed coils' DC field.
  // In the weak reaction limit its hoop electric field is v B_r, so
  // power follows from the magnetostatic radial field in the sleeve.
  const Tube t{.pitches = 2, .machine = 2, .rows = 24, .layers = 0,
               .current = 3., .gap = true};
  auto m = parsed(travelling(t, {}));
  m.materials[3].br = 0.;
  m.materials[5] = {};
  m.properties[5] = {5};
  auto stationary = m;
  stationary.subcases[0].motion = 0;
  const auto reference = solved(stationary);
  ASSERT_EQ(reference.cases.size(), 1U);
  constexpr double speed = .5, sigma = 1e4;
  double radial_energy = 0;
  for (const auto &c : reference.mesh.cells)
    if (c.property == 5)
      for (const auto &q : quadrature(reference.mesh, c)) {
        const auto curl = edge_curls(c, q.position);
        double radial = 0;
        for (size_t i = 0; i < curl.size(); ++i)
          radial += c.radial.dot(curl[i]) * reference.cases[0].potential[c.edges[i]];
        radial_energy += q.weight * out_of_plane(reference.mesh, q.position) * radial * radial;
      }
  const double expected = sigma * speed * speed * radial_energy;
  ASSERT_GT(expected, 1e-8);
  m.options.transient = true;
  m.conductivity[5] = sigma;
  m.motions[1].speed = speed;
  m.subcases[0].time_step = 1;
  m.subcases[0].field.enabled = false;
  m.time_steps[1] = {.maximum_dt = .00061, .max_cycles = 5};
  const auto run = solved(m);
  ASSERT_EQ(run.cases.size(), 1U);
  const auto &c = run.cases.front();
  ASSERT_TRUE(c.transient->converged) << c.failure;
  EXPECT_NEAR(c.transient->mean_loss, expected, .03 * expected);
}

TEST(LineMotion, EachPositionIsTheStaticDeckMovedByHand) {
  // A position is the static deck with the mover's grids moved by hand
  // (the gap's elements, which the element does not use, sheared): the same
  // model, solved by the same code, so with linear materials energy,
  // coenergy, linkage, thrust and back-EMF agree to round-off. Positions
  // off the mesh rows, and beyond a pitch, where no grid of the mover lines
  // up with the stator's and its sides lie past the stator's.
  const std::vector<double> positions{.37 * pitch / 48, 13.6 * pitch / 48, 1.3 * pitch};
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    const Tube base{.offset = 5, .layers = 0, .current = 3, .gap = true, .planar = planar};
    for (bool reversed : {false, true}) {
      SCOPED_TRACE(reversed ? "along -z" : "along +z");
      const double sign = reversed ? -1 : 1;
      Travel v{.count = 3, .reversed = reversed};
      v.start = sign * positions[0];
      v.step = sign * (positions[1] - positions[0]);
      // The third position is not on the sweep's line; solve it alone.
      auto steps = only(travelling(base, v)).steps;
      ASSERT_EQ(steps.size(), 3U);
      steps.pop_back();
      Travel far = v;
      far.start = sign * positions[2];
      far.count = 1;
      const auto alone = only(travelling(base, far)).steps;
      ASSERT_EQ(alone.size(), 1U);
      steps.push_back(alone[0]);
      for (size_t k = 0; k < positions.size(); ++k) {
        SCOPED_TRACE(positions[k]);
        Tube by_hand = base;
        by_hand.move = positions[k];
        const auto s = only(tube(by_hand));
        const auto &m = steps[k];
        EXPECT_NEAR(m.position, sign * positions[k], 1e-18);
        EXPECT_NEAR(m.energy, s.energy, 1e-11 * std::abs(s.energy));
        EXPECT_NEAR(m.coenergy, s.coenergy, 1e-11 * std::abs(s.coenergy));
        // The thrust is along the travel: the static force along +z, signed.
        EXPECT_GT(std::abs(s.gap_force), 1.);
        EXPECT_NEAR(m.torque, sign * s.gap_force, 1e-9 * std::abs(s.gap_force));
        for (size_t c = 0; c < 3; ++c) {
          const auto i = static_cast<Eigen::Index>(c);
          EXPECT_NEAR(m.circuits[c].linkage, s.circuits[c].linkage,
                      1e-10 * std::abs(s.circuits[0].linkage));
          EXPECT_NEAR(m.linkage_rate[i], sign * s.linkage_rate[i],
                      1e-9 * std::abs(s.linkage_rate[0]));
        }
      }
    }
  }
}

TEST(LineMotion, SaturatedPositionsAgreeWithTheDeckMovedByHandToNewtonsTolerance) {
  // With a B-H curve each position starts Newton from the previous one's
  // potential, the static deck from zero: both stop within MAGTOL (1e-11)
  // of the solution, so they agree to that tolerance.
  const Tube base{.offset = 5, .saturating = true, .layers = 0, .current = 6, .gap = true};
  const double step = 7.3 * pitch / 48;
  const auto sweep = only(travelling(base, {.step = step, .count = 3}));
  ASSERT_EQ(sweep.steps.size(), 3U);
  for (int k = 1; k < 3; ++k) {
    Tube by_hand = base;
    by_hand.move = k * step;
    const auto s = only(tube(by_hand));
    const auto &m = sweep.steps[static_cast<size_t>(k)];
    EXPECT_GT(s.iterations, 2);
    EXPECT_NEAR(m.coenergy, s.coenergy, 1e-9 * std::abs(s.coenergy));
    EXPECT_NEAR(m.torque, s.gap_force, 1e-7 * std::abs(s.gap_force));
    for (size_t c = 0; c < 3; ++c)
      EXPECT_NEAR(m.circuits[c].linkage, s.circuits[c].linkage,
                  1e-8 * std::abs(s.circuits[0].linkage));
  }
}

TEST(LineMotion, ThrustAndBackEmfAreTheSlopesOfCoenergyAndLinkageAlongTheTravel) {
  // At fixed currents the thrust on the moving part toward increasing
  // positions is dW'/dx, and each circuit's back-EMF per unit speed is
  // dlambda/dx: central differences over the sweep's own positions -dx, 0,
  // +dx. Moving the mover along +z, along -z (a CID whose x axis is basic
  // -z), and moving the stator (the gap's outer side) instead: each sign of
  // the side and the direction.
  constexpr double dx = 1e-6;
  for (bool planar : {false, true})
    for (bool stator : {false, true})
      for (bool reversed : {false, true}) {
        SCOPED_TRACE(std::format("{}, {} along {}", planar ? "strip" : "tube",
                                 stator ? "stator" : "mover", reversed ? "-z" : "+z"));
        const Tube t{.offset = 5, .layers = 0, .current = 3, .gap = true, .planar = planar};
        const Travel v{.start = -dx, .step = dx, .count = 3, .stator = stator,
                       .reversed = reversed};
        const auto c = only(travelling(t, v));
        ASSERT_EQ(c.steps.size(), 3U);
        const auto &mid = c.steps[1];
        const double thrust = (c.steps[2].coenergy - c.steps[0].coenergy) / (2 * dx);
        EXPECT_GT(std::abs(thrust), 1.);
        EXPECT_NEAR(mid.torque, thrust, 1e-6 * std::abs(thrust));
        for (size_t k = 0; k < 3; ++k) {
          const double rate =
              (c.steps[2].circuits[k].linkage - c.steps[0].circuits[k].linkage) / (2 * dx);
          EXPECT_NEAR(mid.linkage_rate[static_cast<Eigen::Index>(k)], rate,
                      1e-6 * std::abs(mid.linkage_rate[0]))
              << k;
        }
      }
}

TEST(LineMotion, MovingTheStatorIsMovingTheMoverBack) {
  // Only the relative position matters: the stator moved by x is the mover
  // moved by -x, the stack as a whole shifted. The element sees it as the
  // same phase difference between its sides, the same discrete problem, so
  // energy and linkage agree to round-off and the force on the stator is
  // the mover's reaction.
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    const Tube t{.offset = 5, .layers = 0, .current = 3, .gap = true, .planar = planar};
    const double x = 9.4 * pitch / 48;
    const auto a = only(travelling(t, {.start = x, .stator = true}));
    const auto b = only(travelling(t, {.start = -x}));
    ASSERT_EQ(a.steps.size(), 1U);
    ASSERT_EQ(b.steps.size(), 1U);
    const auto &s = a.steps[0], &m = b.steps[0];
    EXPECT_NEAR(s.coenergy, m.coenergy, 1e-11 * std::abs(m.coenergy));
    EXPECT_GT(std::abs(m.torque), 1.);
    // Both toward +z: the stator's force is minus the mover's.
    EXPECT_NEAR(s.torque, -m.torque, 1e-9 * std::abs(m.torque));
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_NEAR(s.circuits[c].linkage, m.circuits[c].linkage,
                  1e-10 * std::abs(m.circuits[0].linkage));
      EXPECT_NEAR(s.linkage_rate[static_cast<Eigen::Index>(c)],
                  -m.linkage_rate[static_cast<Eigen::Index>(c)],
                  1e-9 * std::abs(m.linkage_rate[0]));
    }
  }
}

TEST(LineMotion, APolePitchReversesTheMagnetsOfAnAntiPeriodicStack) {
  // In one anti-periodic pitch the mover a pitch on sits where its
  // magnets' images are reversed: without current the field is the
  // magnets', so the stator's potential, and every coil's linkage and
  // back-EMF, change sign while the coenergy and the cogging force of the
  // slotted stator, which are quadratic in it, repeat. Two pitches are the
  // gap's whole period: everything repeats. Each to round-off.
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    const Tube t{.offset = 5, .layers = 0, .gap = true, .planar = planar, .teeth = 1};
    const double x = 3.7 * pitch / 48;
    const auto c = only(travelling(t, {.start = x, .step = pitch, .count = 3}));
    ASSERT_EQ(c.steps.size(), 3U);
    const auto &first = c.steps[0];
    EXPECT_GT(std::abs(first.torque), 1e-3);
    EXPECT_GT(std::abs(first.circuits[0].linkage), 1e-4);
    for (size_t k = 1; k < 3; ++k) {
      SCOPED_TRACE(k);
      const auto &s = c.steps[k];
      const double sign = k == 1 ? -1 : 1;
      EXPECT_NEAR(s.coenergy, first.coenergy, 1e-11 * std::abs(first.coenergy));
      EXPECT_NEAR(s.torque, first.torque, 1e-9 * std::abs(first.torque));
      for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(s.circuits[i].linkage, sign * first.circuits[i].linkage,
                    1e-10 * std::abs(first.circuits[0].linkage));
        EXPECT_NEAR(s.linkage_rate[static_cast<Eigen::Index>(i)],
                    sign * first.linkage_rate[static_cast<Eigen::Index>(i)],
                    1e-9 * std::abs(first.linkage_rate[0]));
      }
    }
  }
}

TEST(LineMotion, ResultsAreInMetresNewtonsAndVoltsPerMetrePerSecond) {
  const Tube t{.offset = 5, .layers = 0, .current = 3, .gap = true};
  const double step = pitch / 12;
  const auto m = parsed(travelling(t, {.step = step, .count = 4, .speed = 1.5}));
  const auto results = solved(m);
  ASSERT_EQ(results.cases.size(), 1U);
  const auto &c = results.cases[0];
  ASSERT_EQ(c.steps.size(), 4U);
#ifdef VIBESTRAN_HAS_HDF5
  const auto directory = std::filesystem::temp_directory_path() / "vibestran_line_motion";
  std::filesystem::create_directories(directory);
  ASSERT_FALSE(write_results(m, results, directory / "tube.F06").has_value());
  const auto file = directory / "tube.mag.vtkhdf";
  using vibestran::test::h5_attribute, vibestran::test::h5_exists, vibestran::test::h5_table;
  const std::string group = "/MAGSTAT/subcases/1";
  EXPECT_NEAR(h5_attribute(file, group, "step_m"), step, 1e-18);
  EXPECT_NEAR(h5_attribute(file, group, "speed_m_per_s"), 1.5, 0);
  EXPECT_FALSE(h5_exists(file, group + "/positions_deg"));
  const auto positions = h5_table(file, group + "/positions_m");
  const auto force = h5_table(file, group + "/motion/force_N");
  const auto emf = h5_table(file, group + "/motion/emf_V");
  ASSERT_EQ(positions.rows(), 4U);
  ASSERT_EQ(force.rows(), 4U);
  ASSERT_EQ(emf.values.size(), 12U);
  double mean = 0;
  for (size_t k = 0; k < 4; ++k) {
    EXPECT_NEAR(positions.at(k), static_cast<double>(k) * step, 1e-18);
    EXPECT_EQ(force.at(k), c.steps[k].torque);
    mean += force.at(k) / 4;
    // e = v dlambda/dx: volts at 1.5 m/s.
    for (size_t i = 0; i < 3; ++i)
      EXPECT_NEAR(emf.at(k, i), 1.5 * c.steps[k].linkage_rate[static_cast<Eigen::Index>(i)],
                  1e-15);
  }
  EXPECT_NEAR(h5_attribute(file, group, "force_mean_N"), mean, 1e-12 * std::abs(mean));
  EXPECT_TRUE(h5_exists(file, group + "/circuits/linkage_rate_Wb_per_m"));
  std::ifstream in(directory / "tube.F06");
  std::stringstream f06;
  f06 << in.rdbuf();
  for (const std::string text :
       {"MOTION 1 moves ESET 40 (336 elements, 392 grids) along (0, 0, 1), the z axis of "
        "CID 0, sliding the",
        "inner side of PAIRGAP 9: 4 positions from 0 to 0.005 m",
        "MOTION 1: 4 positions from 0 to 0.005 m along the z axis of CID 0",
        "Force on ESET 40 along the direction of travel (N): mean ",
        "  STEP     POSITION(m)            FORCE(N)           ENERGY(J)",
        "         EMF      1(V)"})
    EXPECT_NE(f06.str().find(text), std::string::npos) << text << "\n" << f06.str();
  std::filesystem::remove_all(directory);
#else
  GTEST_SKIP() << "MAGSTAT results need HDF5";
#endif
}

TEST(LineMotion, ABandStaysInPlaceAndItsThrustIsTheSlopeOfTheCoenergy) {
  // A sliding band's points are fixed in space; the mover's trace slides
  // under them. A position is then the static deck moved by hand (the same
  // band, the same model) to round-off, and the band's thrust and linkage
  // rates are the slopes of the coenergy and linkages over the sweep's own
  // positions -dx, 0, +dx, exactly as the discrete model's: the averaged
  // trace moves continuously. A band that moved with the mover would add its
  // own motion to the slope.
  constexpr double dx = 1e-6;
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    const Tube base{.offset = 5, .layers = 0, .current = 3, .gap = true, .planar = planar,
                    .band = true};
    const double at = 7.3 * pitch / 48;
    const auto sweep = only(travelling(base, {.start = at - dx, .step = dx, .count = 3}));
    ASSERT_EQ(sweep.steps.size(), 3U);
    Tube by_hand = base;
    by_hand.move = at;
    const auto results = solved(parsed(tube(by_hand)));
    ASSERT_EQ(results.cases.size(), 1U);
    const auto &s = results.cases[0];
    // NBAND blank: twice (two pitches) the smallest prime of at least the
    // pitch's 48 rows; NLAYER blank: about square elements across the 1 mm
    // gap, 2 pi 1 mm / 40 mm of the period in phase, 0.157 * 106 / (2 pi).
    ASSERT_TRUE(results.mesh.air_gap.has_value());
    EXPECT_EQ(results.mesh.air_gap->band_points, 106);
    EXPECT_EQ(results.mesh.air_gap->band_layers, 3);
    const auto &m = sweep.steps[1];
    EXPECT_NEAR(m.coenergy, s.coenergy, 1e-11 * std::abs(s.coenergy));
    EXPECT_NEAR(m.torque, s.gap_force, 1e-9 * std::abs(s.gap_force));
    const double slope = (sweep.steps[2].coenergy - sweep.steps[0].coenergy) / (2 * dx);
    EXPECT_GT(std::abs(m.torque), 1.);
    EXPECT_NEAR(m.torque, slope, 1e-6 * std::abs(m.torque));
    for (size_t c = 0; c < 3; ++c) {
      const auto i = static_cast<Eigen::Index>(c);
      const double rate = (sweep.steps[2].circuits[c].linkage -
                           sweep.steps[0].circuits[c].linkage) /
                          (2 * dx);
      EXPECT_NEAR(m.linkage_rate[i], rate, 1e-6 * std::abs(m.linkage_rate[0])) << c;
      EXPECT_NEAR(m.linkage_rate[i], s.linkage_rate[i], 1e-9 * std::abs(s.linkage_rate[0]));
    }
  }
}

TEST(LineMotion, ABandConvergesToTheAgeAtSecondOrder) {
  // The loaded stack over a sweep of positions off the rows, its gap a
  // sliding band (NBAND and NLAYER blank: about square elements on the
  // finer side's spacing) and the air-gap element, refined with the rest of
  // the mesh. The band's coenergy and linkage approach the element's at
  // second order (both about 3.3 times closer per halving). Its thrust
  // carries a ripple as the mover's trace slides under the band's points,
  // about 1e-4 of the thrust here (the band's cogging), which does not fall
  // smoothly with the mesh; the thrust is the slope of the band's coenergy
  // (ABandStaysInPlaceAndItsThrustIsTheSlopeOfTheCoenergy).
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    double previous[2] = {1, 1};
    for (int refine : {1, 2}) {
      SCOPED_TRACE(refine);
      const Tube age{.rows = 48 * refine, .offset = 5 * refine, .layers = 0, .current = 3,
                     .gap = true, .planar = planar, .refine = refine};
      Tube band = age;
      band.band = true;
      const Travel v{.start = .37 * pitch / 48, .step = 1.1 * pitch / 48, .count = 9};
      const auto a = only(travelling(age, v)).steps, b = only(travelling(band, v)).steps;
      ASSERT_EQ(a.size(), 9U);
      ASSERT_EQ(b.size(), 9U);
      double coenergy = 0, linkage = 0, thrust = 0, peak = 0, flux = 0;
      for (const auto &step : a) {
        peak = std::max(peak, std::abs(step.torque));
        flux = std::max(flux, std::abs(step.circuits[0].linkage));
      }
      for (size_t k = 0; k < 9; ++k) {
        coenergy = std::max(coenergy, relative(b[k].coenergy, a[k].coenergy));
        linkage = std::max(linkage, std::abs(b[k].circuits[0].linkage - a[k].circuits[0].linkage) / flux);
        thrust = std::max(thrust, std::abs(b[k].torque - a[k].torque) / peak);
      }
      EXPECT_LT(coenergy, (refine == 1 ? 1 : .35) * previous[0]);
      EXPECT_LT(linkage, (refine == 1 ? 1 : .35) * previous[1]);
      EXPECT_LT(thrust, 2e-4);
      previous[0] = coenergy;
      previous[1] = linkage;
    }
    EXPECT_LT(previous[0], 1e-4);
    EXPECT_LT(previous[1], 2e-4);
  }
}

TEST(LineMotion, CardsNameWhatFails) {
  const auto message = [](const Axi &g, const std::string &from = "",
                          const std::string &to = "") {
    std::string text = deck(g);
    if (!from.empty())
      text = std::regex_replace(text, std::regex(from), to);
    auto m = parse_string(text);
    if (const auto *e = std::get_if<vibestran::Error>(&m))
      return e->message;
    return error_of(solve(std::get<Model>(m)));
  };
  const auto contains = [](const std::string &text, const std::string &part) {
    return text.find(part) != std::string::npos;
  };
  const Tube tube_t{.offset = 5, .layers = 0, .gap = true};
  const Tube strip_t{.offset = 5, .layers = 0, .gap = true, .planar = true};
  const Axi tube_g = travelling(tube_t, {.step = .001, .count = 2});
  const Axi strip_g = travelling(strip_t, {.step = .001, .count = 2});
  // ROTATE in a model that repeats along a line.
  std::string e = message(strip_g, "TRANSLATE", "ROTATE");
  EXPECT_TRUE(contains(e, "MOTION 1 ROTATE turns the part about an axis, but MPERIOD 1 "
                          "repeats the model along a line, where the gap is a strip"))
      << e;
  e = message(tube_g, "TRANSLATE", "ROTATE");
  EXPECT_TRUE(contains(e, "MOTION 1 ROTATE in a MAGAXI model")) << e;
  // The direction of travel: MAGAXI's along the axis, MAG2D's in the plane
  // and along the gap. CID 6's x axis is basic y, the plane normal; CID 9's
  // is basic x, across the strip.
  e = message(tube_g, "MOTION,1,40,0,", "CORD2R,6,0,0.,0.,0.,1.,0.,0.\n,0.,0.,1.\nMOTION,1,40,6,");
  EXPECT_TRUE(contains(e, "MOTION 1: the z axis of CID 6, the direction of travel, must be "
                          "parallel to the MAGAXI symmetry axis"))
      << e;
  e = message(strip_g, "MOTION,1,40,8,", "CORD2R,6,0,0.,0.,0.,1.,0.,0.\n,0.,1.,0.\nMOTION,1,40,6,");
  EXPECT_TRUE(contains(e, "MOTION 1: the x axis of CID 6, the direction of travel, must lie "
                          "in the MAG2D plane"))
      << e;
  e = message(strip_g, "MOTION,1,40,8,", "MOTION,1,40,9,");
  EXPECT_TRUE(contains(e, "MOTION 1: the direction of travel, the x axis of CID 9 (1, 0, 0), "
                          "is not along PAIRGAP 9's lines (0, 0, 1), the MPERIOD shift"))
      << e;
  // The moving part on both sides of the gap.
  e = message(tube_g, "ESET,40,PROP,4,5,6", "ESET,40,PROP,4,5,6,2");
  EXPECT_TRUE(contains(e, "MOTION 1: grid ")) << e;
}

