// tests/unit/test_periodic.cpp
// MPERIOD sector models and MBC NORMAL symmetry faces. Each guarantee is
// against the deck an analyst would write without them: the whole machine,
// meshed as the sector's turned copies with each source's images in series,
// and the whole mirrored domain.

#include "machine_deck.hpp"
#ifdef VIBESTRAN_HAS_HDF5
#include "../common/hdf5_results.hpp"
#endif
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace {
using namespace machine_deck;
constexpr double degree = pi / 180;

// A surface-magnet machine with tooth-wound coils: `slots` slots, `poles`
// pole pairs, built from `sectors` identical sectors of 360 / sectors
// degrees. With whole set the deck is the whole machine; otherwise it is
// sector 0 with an MPERIOD (ANTI when anti) pairing its sides. Grid and
// element IDs depend only on ring and division, so a sector's grids and
// cells have the IDs of the same grids and cells of the whole machine.
struct Design {
  int slots{12}, poles{7}, sectors{2};
  bool anti{true}, whole{false}, saturating{false};
  int per_pole{4}, per_slot{6}; // angular divisions of a pole and a slot pitch
  double turns{30};
  std::string cards;   // MOTION, DRIVE, ICIRC and further bulk data
  std::string control; // case control
};
// Rotor rings (iron to .016, magnets to .019), the gap to .020, and the
// stator rings (slots to .028, yoke to the A = 0 exterior at .036).
const std::vector<double> rotor_rings{.004, .008, .012, .016, .0175, .019};
const std::vector<double> stator_rings{.020, .022, .025, .028, .032, .036};
constexpr int slot_bands = 3;

// Each tooth coil's phase and polarity from the star of slots, the phase
// axes at 15 + 120 k electrical degrees (as tools/magnetic_motor_example.py).
std::vector<std::pair<int, int>> layout(const Design &d) {
  std::vector<std::pair<int, int>> out;
  for (int tooth = 0; tooth < d.slots; ++tooth) {
    const double angle = d.poles * 360.0 * tooth / d.slots;
    std::pair<int, int> best{0, 1};
    double nearest = 1e9;
    for (int phase = 0; phase < 3; ++phase)
      for (int sign : {1, -1}) {
        const double axis = 15 + 120 * phase + (sign > 0 ? 0 : 180);
        const double distance = std::abs(std::remainder(angle - axis, 360.));
        if (distance < nearest) {
          nearest = distance;
          best = {phase, sign};
        }
      }
    out.push_back(best);
  }
  return out;
}

struct Built {
  std::string deck;
  std::vector<int> grids; // every grid ID
};
Built build(const Design &d) {
  const int rotor_div = d.per_pole * 2 * d.poles / d.sectors,
            stator_div = d.per_slot * d.slots / d.sectors;
  const int copies = d.whole ? d.sectors : 1;
  const int total_rotor = rotor_div * d.sectors, total_stator = stator_div * d.sectors;
  std::string grids, elements;
  Built out;
  auto grid = [&](int ring, int j, double r, int total) {
    const int id = 1000 * ring + j + 1;
    const double t = 2 * pi * j / total;
    grids += std::format("GRID,{},,{:.17g},{:.17g},0.\n", id, r * std::cos(t), r * std::sin(t));
    out.grids.push_back(id);
    return id;
  };
  // Division j of ring `ring` (wrapping round the whole machine).
  auto id_at = [&](int ring, int j, int total) {
    return 1000 * ring + (d.whole ? j % total : j) + 1;
  };
  const int center = 999;
  grids += "GRID,999,,0.,0.,0.\n";
  out.grids.push_back(center);
  const int span_r = d.whole ? total_rotor : rotor_div + 1,
            span_s = d.whole ? total_stator : stator_div + 1;
  for (size_t k = 0; k < rotor_rings.size(); ++k)
    for (int j = 0; j < span_r; ++j)
      grid(1 + static_cast<int>(k), j, rotor_rings[k], total_rotor);
  for (size_t k = 0; k < stator_rings.size(); ++k)
    for (int j = 0; j < span_s; ++j)
      grid(11 + static_cast<int>(k), j, stator_rings[k], total_stator);
  std::map<std::pair<int, int>, std::vector<int>> coils; // (slot, half) -> elements
  std::vector<std::array<int, 3>> first_side, second_side; // element, edge grids
  auto element = [&](int id, int pid, const std::vector<int> &nodes) {
    elements += std::format("{},{},{}", nodes.size() == 3 ? "CTRIA3" : "CQUAD4", id, pid);
    for (int n : nodes)
      elements += std::format(",{}", n);
    elements += '\n';
  };
  const int rotor_end = rotor_div * copies, stator_end = stator_div * copies;
  // Rotor: PID 2 iron, 4 and 6 magnets out and in, 5 the air between them.
  for (int j = 0; j < rotor_end; ++j) {
    const int pole = j / d.per_pole;
    const bool magnet = j % d.per_pole != d.per_pole - 1;
    for (size_t band = 0; band < rotor_rings.size(); ++band) {
      const int a = static_cast<int>(band), id = 10 * (1000 * a + j) + 1;
      int pid = 2;
      if (rotor_rings[band] > .016)
        pid = magnet ? (pole % 2 == 0 ? 4 : 6) : 5;
      if (band == 0) {
        element(id, pid, {center, id_at(1, j, total_rotor), id_at(1, j + 1, total_rotor)});
        if (!d.whole && j == 0)
          first_side.push_back({id, center, id_at(1, 0, total_rotor)});
        if (!d.whole && j == rotor_div - 1)
          second_side.push_back({id, center, id_at(1, rotor_div, total_rotor)});
        continue;
      }
      const int lo = a, hi = a + 1;
      element(id, pid,
              {id_at(lo, j, total_rotor), id_at(lo, j + 1, total_rotor),
               id_at(hi, j + 1, total_rotor), id_at(hi, j, total_rotor)});
      if (!d.whole && j == 0)
        first_side.push_back({id, id_at(lo, 0, total_rotor), id_at(hi, 0, total_rotor)});
      if (!d.whole && j == rotor_div - 1)
        second_side.push_back(
            {id, id_at(lo, rotor_div, total_rotor), id_at(hi, rotor_div, total_rotor)});
    }
  }
  // Stator: PID 3 iron; slot s spans divisions 1 .. per_slot - 2 of its
  // pitch, its clockwise half the counterclockwise side of tooth s and its
  // other half the clockwise side of tooth s + 1.
  for (int j = 0; j < stator_end; ++j) {
    const int slot = j / d.per_slot, within = j % d.per_slot;
    const bool open = within >= 1 && within <= d.per_slot - 2;
    const int half = within < d.per_slot / 2 ? 0 : 1;
    for (int band = 0; band + 1 < static_cast<int>(stator_rings.size()); ++band) {
      const int lo = 11 + band, hi = 12 + band, id = 10 * (1000 * lo + j) + 1;
      const bool in_slot = open && band < slot_bands;
      element(id, in_slot ? 1 : 3,
              {id_at(lo, j, total_stator), id_at(lo, j + 1, total_stator),
               id_at(hi, j + 1, total_stator), id_at(hi, j, total_stator)});
      if (in_slot)
        coils[{slot, half}].push_back(id);
      if (!d.whole && j == 0)
        first_side.push_back({id, id_at(lo, 0, total_stator), id_at(hi, 0, total_stator)});
      if (!d.whole && j == stator_div - 1)
        second_side.push_back(
            {id, id_at(lo, stator_div, total_stator), id_at(hi, stator_div, total_stator)});
    }
  }
  // The gap: triangles zipping the rotor's outer ring to the stator's inner
  // ring within each sector, whose ends line up.
  int gap_id = 9000000;
  for (int c = 0; c < copies; ++c) {
    const int r0 = c * rotor_div, s0 = c * stator_div;
    for (int i = 0, k = 0; i < rotor_div || k < stator_div;) {
      const double next_r = (i + 1.) / rotor_div, next_s = (k + 1.) / stator_div;
      const int ri = static_cast<int>(rotor_rings.size());
      if (k == stator_div || (i < rotor_div && next_r < next_s)) {
        element(++gap_id, 9,
                {id_at(ri, r0 + i, total_rotor), id_at(ri, r0 + i + 1, total_rotor),
                 id_at(11, s0 + k, total_stator)});
        ++i;
      } else {
        element(++gap_id, 9,
                {id_at(ri, r0 + i, total_rotor), id_at(11, s0 + k + 1, total_stator),
                 id_at(11, s0 + k, total_stator)});
        ++k;
      }
    }
  }
  std::string text = "SOL MAGSTAT\nCEND\n" + d.control +
                     "BEGIN BULK\nMAG2D,0,.05\nCORD2C,10,,0.,0.,0.,0.,0.,1.\n,1.,0.,0.\n"
                     "MATMAG,1,1.\nPMAG2D,1,1\nPMAG2D,5,1\n"
                     "MATMAG,3,1.05,,1.2\nPMAG2D,4,3,10,1\nPMAG2D,6,3,10,-1\nPAIRGAP,9\n"
                     "ESET,20,PROP,2,4,5,6\n";
  text += d.saturating ? "MATMAG,2,,1\nPMAG2D,2,2\nPMAG2D,3,2\nTABLEBH,1\n"
                         ",0.,0.,.5,60.,1.,120.,1.3,250.\n,1.5,800.,1.6,1800.,1.7,4500.,"
                         "1.8,10000.\n,1.9,20000.,2.,40000.,2.1,80000.,ENDT\n"
                       : "MATMAG,2,1000.\nPMAG2D,2,2\nPMAG2D,3,2\n";
  if (!d.whole) {
    text += std::format("MPERIOD,1,{},31,32,0,{}\n", d.anti ? "ANTI" : "PERIODIC", d.sectors);
    for (const auto &[id, a, b] : first_side)
      text += std::format("FSET,31,{},{},{}\n", id, a, b);
    for (const auto &[id, a, b] : second_side)
      text += std::format("FSET,32,{},{},{}\n", id, a, b);
  }
  const auto teeth = layout(d);
  std::array<std::vector<int>, 3> phases;
  for (const auto &[key, ids] : coils) {
    const auto [slot, half] = key;
    const int tooth = (slot + half) % d.slots, set = 100 + 2 * slot + half;
    const auto [phase, sign] = teeth[static_cast<size_t>(tooth)];
    text += std::format("ESET,{}", set);
    for (int id : ids)
      text += std::format(",{}", id);
    text += std::format("\nCOIL2D,{},{},{:.17g}\n", set, set, (half ? -1 : 1) * sign * d.turns);
    phases[static_cast<size_t>(phase)].push_back(set);
  }
  for (int k = 0; k < 3; ++k) {
    text += std::format("CIRCUIT,{},1", k + 1);
    for (int set : phases[static_cast<size_t>(k)])
      text += std::format(",{},1", set);
    text += '\n';
  }
  out.deck = text + d.cards + grids + elements + "ENDDATA\n";
  return out;
}
Model model_of(const Design &d) {
  auto out = parse_string(build(d).deck);
  EXPECT_TRUE(std::holds_alternative<Model>(out)) << error_of(out);
  return std::holds_alternative<Model>(out) ? std::get<Model>(std::move(out)) : Model{};
}
Design whole(Design d) {
  d.whole = true;
  return d;
}
// NaN (an unavailable quantity on either side) counts as a failure.
double relative(double a, double b, double scale) {
  const double out = std::abs(a - b) / scale;
  return std::isnan(out) ? std::numeric_limits<double>::infinity() : out;
}
double largest(const Eigen::MatrixXd &m) {
  return m.hasNaN() ? std::numeric_limits<double>::infinity() : m.cwiseAbs().maxCoeff();
}
double largest_norm(const std::vector<Vector> &v) {
  return std::accumulate(v.begin(), v.end(), 0.0, [](double most, const Vector &x) {
    return std::max(most, x.norm());
  });
}
// Mid-gap probes at angles in every sector of the machine.
std::string gap_probes() {
  std::string out;
  for (int k = 0; k < 6; ++k) {
    const double t = (7 + 61 * k) * degree, r = .0195;
    out += std::format("MPROBE,{},0,{:.17g},{:.17g},0.\n", k + 1, r * std::cos(t), r * std::sin(t));
  }
  return out;
}

// The largest difference between two states, relative to each quantity's
// scale: the potential at the sector's grids, B in the sector's cells, the
// whole machine's energy, coenergy, gap torque, circuit linkages,
// inductances and linkage rates, and the probes.
struct Difference {
  double potential{0}, b{0}, energy{0}, torque{0}, linkage{0}, inductance{0}, rate{0},
      probe{0};
};
Difference compare(const Mesh &sector_mesh, const CaseResults &sector, const Mesh &whole_mesh,
                   const CaseResults &full) {
  Difference out;
  const double a_scale = full.potential.cwiseAbs().maxCoeff();
  for (size_t i = 0; i < sector_mesh.node_ids.size(); ++i) {
    const int w = whole_mesh.node_index.at(sector_mesh.node_ids[i]);
    out.potential = std::max(out.potential, std::abs(sector.potential[static_cast<Eigen::Index>(i)] -
                                                     full.potential[w]) /
                                                a_scale);
  }
  const double b_scale = largest_norm(full.b);
  for (size_t c = 0; c < sector_mesh.cells.size(); ++c) {
    const int w = whole_mesh.element_index.at(sector_mesh.cells[c].id);
    out.b = std::max(out.b, (sector.b[c] - full.b[static_cast<size_t>(w)]).norm() / b_scale);
  }
  out.energy = std::max(relative(sector.energy, full.energy, std::abs(full.energy)),
                        relative(sector.coenergy, full.coenergy, std::abs(full.coenergy)));
  out.torque = relative(sector.gap_torque, full.gap_torque, 1);
  const double l_scale = std::accumulate(full.circuits.begin(), full.circuits.end(), 0.0,
                                         [](double most, const CircuitResults &c) {
                                           return std::max(most, std::abs(c.linkage));
                                         }),
               i_scale = full.circuit_inductance.cwiseAbs().maxCoeff(),
               r_scale = full.linkage_rate.cwiseAbs().maxCoeff();
  for (size_t c = 0; c < full.circuits.size(); ++c) {
    out.linkage = std::max(out.linkage, relative(sector.circuits[c].linkage,
                                                 full.circuits[c].linkage, l_scale));
    out.rate = std::max(out.rate, relative(sector.linkage_rate[static_cast<Eigen::Index>(c)],
                                           full.linkage_rate[static_cast<Eigen::Index>(c)],
                                           r_scale));
  }
  out.inductance = largest(sector.circuit_inductance - full.circuit_inductance) / i_scale;
  const double p_scale = largest_norm(full.probe_b);
  for (size_t k = 0; k < full.probe_b.size(); ++k)
    out.probe = std::max(out.probe, (sector.probe_b[k] - full.probe_b[k]).norm() / p_scale);
  return out;
}

} // namespace

TEST(Periodic, AntiPeriodicHalfIsTheWholeMachine) {
  // 12 slots, 14 poles: turning by 180 degrees takes each pole to one of
  // opposite polarity and each tooth coil to one of the same phase wound
  // the other way, so the field is anti-periodic over half the machine.
  // The half model is the whole machine: its potential at the half's grids,
  // its B, and the machine's energy, torque, linkages, inductances and
  // back-EMF, to round-off with linear iron and to the Newton tolerance
  // with saturating steel. Probes in the other half of the gap read the
  // unfolded gap's field.
  for (const bool saturating : {false, true}) {
    SCOPED_TRACE(saturating ? "saturating steel" : "linear iron");
    Design d;
    d.saturating = saturating;
    d.cards = (saturating ? "ICIRC,5,1,60.\nICIRC,5,2,-25.\nICIRC,5,3,-35.\nPARAM,MAGTOL,1e-12\n"
                          : "ICIRC,5,1,8.\nICIRC,5,2,-3.\nICIRC,5,3,-5.\n") +
              gap_probes();
    d.control = "LOAD = 5\n";
    const auto half = solved(model_of(d)), full = solved(model_of(whole(d)));
    ASSERT_EQ(half.cases.size(), 1U);
    ASSERT_EQ(full.cases.size(), 1U);
    const auto diff = compare(half.mesh, half.cases[0], full.mesh, full.cases[0]);
    const double tolerance = saturating ? 1e-9 : 1e-11;
    EXPECT_LT(diff.potential, tolerance);
    EXPECT_LT(diff.b, tolerance);
    EXPECT_LT(diff.energy, tolerance);
    EXPECT_LT(diff.torque, tolerance * std::abs(full.cases[0].gap_torque) + 1e-12);
    EXPECT_LT(diff.linkage, tolerance);
    EXPECT_LT(diff.inductance, tolerance);
    EXPECT_LT(diff.rate, tolerance);
    EXPECT_LT(diff.probe, tolerance);
    EXPECT_GT(std::abs(full.cases[0].gap_torque), 1e-3);
    double steel = 0;
    for (size_t c = 0; c < full.mesh.cells.size(); ++c)
      if (full.mesh.cells[c].property == 2 || full.mesh.cells[c].property == 3)
        steel = std::max(steel, full.cases[0].b[c].norm());
    // The steel is driven well past the knee of its curve (1.3 T).
    if (saturating) {
      EXPECT_GT(steel, 1.9);
    }
    // The half has half the grids, and its gap the whole machine's
    // harmonics (of which only the odd ones are nonzero).
    EXPECT_LT(half.mesh.nodes.size(), full.mesh.nodes.size() / 2 + 40);
    ASSERT_TRUE(half.mesh.air_gap && full.mesh.air_gap);
    EXPECT_EQ(half.mesh.air_gap->harmonics, full.mesh.air_gap->harmonics);
    EXPECT_EQ(half.mesh.air_gap->sectors, 2);
  }
}

TEST(Periodic, PeriodicQuarterIsTheWholeMachine) {
  // 12 slots, 8 poles repeat every 90 degrees: a quarter (3 slots, 2 poles)
  // with PERIODIC sides is the whole machine, and so is a half.
  for (const int sectors : {4, 2}) {
    SCOPED_TRACE(std::format("{} sectors", sectors));
    Design d;
    d.poles = 4;
    d.sectors = sectors;
    d.anti = false;
    d.cards = "ICIRC,5,1,8.\nICIRC,5,2,-3.\nICIRC,5,3,-5.\n" + gap_probes();
    d.control = "LOAD = 5\n";
    const auto part = solved(model_of(d)), full = solved(model_of(whole(d)));
    const auto diff = compare(part.mesh, part.cases[0], full.mesh, full.cases[0]);
    for (double x : {diff.potential, diff.b, diff.energy, diff.linkage, diff.inductance,
                     diff.rate, diff.probe})
      EXPECT_LT(x, 1e-11);
    EXPECT_LT(diff.torque, 1e-11 * std::abs(full.cases[0].gap_torque));
  }
}

TEST(Periodic, DriveSweepOfTheHalfIsTheWholeMachines) {
  // MOTION turns the half's rotor with its periodic sides, and DRIVE with
  // THETA0 AUTO finds the same d axis: the torque, dq quantities and
  // back-EMF at each position are the whole machine's.
  Design d;
  d.cards = "MOTION,1,20,0,ROTATE,0.,1.5,6\n,1000.\nDRIVE,10,SINE,7,10.,20.\n,1,2,3\n";
  d.control = "SUBCASE 1\n  MOTION = 1\n  DRIVE = 10\n";
  const auto half = solved(model_of(d)), full = solved(model_of(whole(d)));
  ASSERT_EQ(half.cases.size(), 1U);
  const auto &hs = half.cases[0], &fs = full.cases[0];
  ASSERT_TRUE(hs.drive_summary && fs.drive_summary);
  EXPECT_NEAR(hs.drive_summary->theta0, fs.drive_summary->theta0, 1e-9);
  EXPECT_NEAR(hs.drive_summary->linkage, fs.drive_summary->linkage,
              1e-11 * fs.drive_summary->linkage);
  ASSERT_EQ(hs.steps.size(), fs.steps.size());
  double torque = 0, scale = 0, dq = 0, rate = 0;
  for (size_t k = 0; k < fs.steps.size(); ++k) {
    const auto &h = hs.steps[k], &f = fs.steps[k];
    scale = std::max(scale, std::abs(f.torque));
    torque = std::max(torque, relative(h.torque, f.torque, 1));
    dq = std::max({dq, largest(h.drive->linkage - f.drive->linkage) / f.drive->linkage.norm(),
                   largest(h.drive->incremental - f.drive->incremental) /
                       f.drive->incremental.norm(),
                   largest(h.drive->apparent - f.drive->apparent) / f.drive->apparent.norm()});
    rate = std::max(rate, largest(h.linkage_rate - f.linkage_rate) / largest(f.linkage_rate));
  }
  EXPECT_LT(torque, 1e-11 * scale);
  EXPECT_LT(dq, 1e-11);
  EXPECT_LT(rate, 1e-11);
  EXPECT_GT(scale, 1e-2);
}

namespace {
// A rectangle of square cells, mirror-symmetric about y = 0: two coil
// pairs (go and return), linear iron blocks and a magnet magnetized along
// +y across the mirror line, all the same above and below, so A is even in
// y and the flux crosses y = 0 normally. With half set it is the upper half
// with an MBC NORMAL row on y = 0. Grid and element IDs depend only on the
// cell's row and column.
std::string mirror_deck(bool half) {
  constexpr int nx = 20, ny = 8; // columns, rows per half
  constexpr double h = .005, x0 = -.05, y0 = -.04;
  const int j0 = half ? ny : 0;
  std::string out = "SOL MAGSTAT\nCEND\nLOAD = 5\n";
  if (half)
    out += "MAGBC = 1\n";
  out += "BEGIN BULK\nMAG2D,0,.1\nMATMAG,1,1.\nPMAG2D,1,1\nMATMAG,2,500.\nPMAG2D,2,2\n"
         "MATMAG,3,1.05,,1.\nPMAG2D,3,3,0,2\n";
  const auto grid = [&](int i, int j) { return 1 + i + (nx + 1) * j; };
  for (int j = j0; j <= 2 * ny; ++j)
    for (int i = 0; i <= nx; ++i)
      out += std::format("GRID,{},,{:.17g},{:.17g},0.\n", grid(i, j), x0 + h * i, y0 + h * j);
  std::map<int, std::vector<int>> coils;
  std::string normal;
  for (int j = j0; j < 2 * ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const double x = x0 + h * (i + .5), y = std::abs(y0 + h * (j + .5));
      const int id = 1 + i + nx * j;
      int pid = 1;
      if (std::abs(x) < .01 && y > .025 && y < .035)
        pid = 2;
      if (std::abs(x) < .005 && y < .01)
        pid = 3;
      // Each coil and its mirror image are regions of their own (COIL2D
      // spreads its turns over its region).
      if (y > .01 && y < .02 && std::abs(x) > .015 && std::abs(x) < .025)
        coils[(x < 0 ? 11 : 12) + (y0 + h * (j + .5) < 0 ? 2 : 0)].push_back(id);
      out += std::format("CQUAD4,{},{},{},{},{},{}\n", id, pid, grid(i, j), grid(i + 1, j),
                         grid(i + 1, j + 1), grid(i, j + 1));
      if (half && j == ny)
        normal += std::format("FSET,7,{},{},{}\n", id, grid(i, j), grid(i + 1, j));
    }
  for (const auto &[set, ids] : coils) {
    out += std::format("ESET,{}", set);
    for (int id : ids)
      out += std::format(",{}", id);
    out += std::format("\nCOIL2D,{},{},{}\n", set, set, set % 2 ? 40 : -40);
  }
  out += half ? "CIRCUIT,1,1,11,1,12,1\n" : "CIRCUIT,1,1,11,1,12,1,13,1\n,14,1\n";
  out += "ICIRC,5,1,12.\n";
  if (half)
    out += "MBC,1,0,ZERO\nMBC,1,7,NORMAL\n" + normal;
  return out + "ENDDATA\n";
}
Results mirror(bool half) {
  auto m = parse_string(mirror_deck(half));
  EXPECT_TRUE(std::holds_alternative<Model>(m)) << error_of(m);
  return solved(std::get<Model>(m));
}
} // namespace

TEST(Periodic, NormalFacesAreTheMirroredDomain) {
  // MBC NORMAL leaves the mirror line free (n x H = 0), so the half model
  // is the upper half of the whole mirrored domain: the same potential at
  // its grids and B in its cells, and half the energy and circuit linkage
  // (a boundary condition, unlike MPERIOD, does not stand for the rest).
  const auto half = mirror(true), full = mirror(false);
  const auto &h = half.cases[0], &f = full.cases[0];
  double potential = 0, b = 0;
  const double b_scale = largest_norm(f.b);
  for (size_t i = 0; i < half.mesh.node_ids.size(); ++i)
    potential = std::max(potential,
                         std::abs(h.potential[static_cast<Eigen::Index>(i)] -
                                  f.potential[full.mesh.node_index.at(half.mesh.node_ids[i])]));
  for (size_t c = 0; c < half.mesh.cells.size(); ++c)
    b = std::max(b, (h.b[c] - f.b[static_cast<size_t>(
                                  full.mesh.element_index.at(half.mesh.cells[c].id))])
                        .norm());
  EXPECT_LT(potential, 1e-12 * f.potential.cwiseAbs().maxCoeff());
  EXPECT_LT(b, 1e-12 * b_scale);
  EXPECT_LT(relative(2 * h.energy, f.energy, f.energy), 1e-12);
  EXPECT_LT(relative(2 * h.circuits[0].linkage, f.circuits[0].linkage,
                     std::abs(f.circuits[0].linkage)),
            1e-12);
  // The flux crosses the mirror line: B along y is the largest component
  // in the magnet, and A is not constant on the line.
  const auto line = half.mesh.node_index.at(1 + 10 + 21 * 8);
  const auto edge = half.mesh.node_index.at(1 + 4 + 21 * 8);
  EXPECT_GT(std::abs(h.potential[line] - h.potential[edge]), 1e-6);
  // Without the NORMAL row the line is flux-parallel, A = 0: a different field.
  auto zero = parse_string(mirror_deck(true));
  auto &m = std::get<Model>(zero);
  m.boundaries.at(1).normal.clear();
  const auto parallel = solved(m);
  EXPECT_GT(relative(parallel.cases[0].energy, h.energy, h.energy), .01);
}

TEST(Periodic, ResultsReportTheWholeMachine) {
  Design d;
  d.cards = "ICIRC,5,1,8.\n";
  d.control = "LOAD = 5\n";
  const auto m = model_of(d);
  const auto r = solved(m);
  std::string rows;
  for (const auto &[label, text] : r.summary.solution.rows)
    if (label == "Periodic" || label == "Air gap")
      rows += text + "\n";
  EXPECT_NE(rows.find("MPERIOD 1: anti-periodic sector of 2 about the z axis of CID 0"),
            std::string::npos)
      << rows;
  EXPECT_NE(rows.find("the whole machine's: 2 times the sector's"), std::string::npos) << rows;
  EXPECT_NE(rows.find("unfolded around 2 sectors, changing sign in alternate sectors: "
                      "harmonics that are not odd multiples of 1 vanish"),
            std::string::npos)
      << rows;
  EXPECT_NE(rows.find("1 on the axis held at A = 0"), std::string::npos) << rows;
#ifdef VIBESTRAN_HAS_HDF5
  using namespace vibestran::test;
  const auto directory = std::filesystem::temp_directory_path() / "vibestran_periodic";
  std::filesystem::create_directories(directory);
  ASSERT_FALSE(write_results(m, r, directory / "half.F06").has_value());
  const auto file = directory / "half.mag.vtkhdf";
  EXPECT_EQ(h5_attribute(file, "/MAGSTAT/periodic", "sectors"), 2);
  EXPECT_EQ(h5_attribute(file, "/MAGSTAT/periodic", "sector_angle_deg"), 180);
  EXPECT_EQ(h5_attribute(file, "/MAGSTAT/air_gap", "sectors"), 2);
  const auto first = h5_table(file, "/MAGSTAT/periodic/first_grids"),
             second = h5_table(file, "/MAGSTAT/periodic/second_grids");
  ASSERT_EQ(first.values.size(), second.values.size());
  // Rotor rings 1-6 and stator rings 11-16 at division 0 pair with the
  // same rings at the last division; the center is on the axis.
  EXPECT_EQ(first.values.size(), 12U);
  EXPECT_EQ(h5_table(file, "/MAGSTAT/periodic/axial_grids").values,
            std::vector<double>{999});
  // The gap's arcs, each grid once: 28 rotor divisions of the half, 36 stator.
  EXPECT_EQ(h5_table(file, "/MAGSTAT/air_gap/inner_grids").values.size(), 28U);
  EXPECT_EQ(h5_table(file, "/MAGSTAT/air_gap/outer_grids").values.size(), 36U);
#endif
}

TEST(Periodic, SidesMustMatchAndBoundariesAgree) {
  Design d;
  d.cards = "ICIRC,5,1,8.\n";
  d.control = "LOAD = 5\n";
  const auto base = build(d).deck;
  const auto fails = [](const std::string &deck) {
    auto m = parse_string(deck);
    if (const auto *e = std::get_if<Error>(&m))
      return e->message;
    return error_of(solve(std::get<Model>(m)));
  };
  const auto with = [&](const std::string &from, const std::string &to) {
    auto out = base;
    const auto at = out.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    return out.replace(at, from.size(), to);
  };
  // The sector count must turn the first side onto the second.
  EXPECT_NE(fails(with("MPERIOD,1,ANTI,31,32,0,2", "MPERIOD,1,ANTI,31,32,0,4"))
                .find("turned by 90 degrees about the axis lands at"),
            std::string::npos);
  EXPECT_NE(fails(with("MPERIOD,1,ANTI,31,32,0,2", "MPERIOD,1,ANTI,31,32,0,3"))
                .find("an odd number of times around the machine"),
            std::string::npos);
  EXPECT_NE(fails(with("MPERIOD,1,ANTI,31,32,0,2", "MPERIOD,1,ANTI,31,32,0,1"))
                .find("requires SECTORS >= 2"),
            std::string::npos);
  EXPECT_NE(fails(with("MPERIOD,1,ANTI", "MPERIOD,1,ODD")).find("PERIODIC or ANTI"),
            std::string::npos);
  EXPECT_NE(fails(with("MPERIOD,1,ANTI,31,32,0,2\n",
                       "MPERIOD,1,ANTI,31,32,0,2\nMPERIOD,2,ANTI,31,32,0,2\n"))
                .find("a model is one sector"),
            std::string::npos);
  // Every grid of the second side is an image: drop the first side's
  // outermost stator edge.
  const auto outer = std::format("FSET,31,{},{},{}\n", 10 * (1000 * 15) + 1, 15001, 16001);
  EXPECT_NE(fails(with(outer, "")).find("is not the image of any grid of the first side"),
            std::string::npos);
  // The sectors also need the gap's arcs to span the sector: MPERIOD
  // without a gap that is a sector of the annulus is caught there.
  // A uniform field is not periodic over a quarter turn.
  Design quarter = d;
  quarter.poles = 4;
  quarter.sectors = 4;
  quarter.anti = false;
  quarter.control = "LOAD = 5\nMAGBC = 1\n";
  quarter.cards += "MBC,1,0,UNIFORM,.1,0.,0.\n";
  EXPECT_NE(fails(build(quarter).deck).find("which MPERIOD 1 requires to be equal to it"),
            std::string::npos);
  // A NORMAL face on a periodic side, and grid pairs whose boundary
  // conditions differ: the outer arc's last edge NORMAL frees the corner
  // grid on the second side but not its partner on the first.
  const int last = d.per_slot * d.slots / d.sectors - 1;
  // The deck with MAGBC = 1 and these bulk cards.
  const auto with_mbc = [&](const std::string &cards) {
    auto out = with("LOAD = 5\n", "LOAD = 5\nMAGBC = 1\n");
    return out.replace(out.find("ENDDATA"), 0, cards);
  };
  const auto with_normal = [&](const std::string &fset) {
    return with_mbc("MBC,1,0,ZERO\nMBC,1,40,NORMAL\n" + fset);
  };
  EXPECT_NE(fails(with_normal(outer.substr(0, 5) + "40" + outer.substr(7)))
                .find("on a periodic side of MPERIOD 1"),
            std::string::npos);
  const int arc = 10 * (1000 * 15 + last) + 1;
  EXPECT_NE(fails(with_normal(std::format("FSET,40,{},{},{}\n", arc, 16000 + last + 1,
                                          16000 + last + 2)))
                .find("must both be on the prescribed (ZERO or UNIFORM) exterior"),
            std::string::npos);
  // MBC rows: one ZERO or UNIFORM row per SID, and NORMAL names its faces.
  EXPECT_NE(fails(with_mbc("MBC,1,0,ZERO\nMBC,1,0,ZERO\n"))
                .find("has a second ZERO or UNIFORM row"),
            std::string::npos);
  EXPECT_NE(fails(with_mbc("MBC,1,0,NORMAL\n")).find("MBC NORMAL needs an FSET"),
            std::string::npos);
  // A force surface must close within the sector: the rotor's meets both
  // sides (and the gap).
  {
    auto out = base;
    out.replace(out.find("ENDDATA"), 0, "MFORCE,1,BODY,20\n");
    const auto text = fails(out);
    EXPECT_NE(text.find("reaches a periodic side of MPERIOD 1"), std::string::npos) << text;
  }
  // Periodic sectors are planar.
  auto planar = parse_string(base);
  auto &m = std::get<Model>(planar);
  m.options.planar = false;
  EXPECT_NE(error_of(solve(m)).find("MPERIOD 1 is for MAG2D models"), std::string::npos);
}
