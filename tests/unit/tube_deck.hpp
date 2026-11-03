#pragma once
// tests/unit/tube_deck.hpp
// Structured MAGAXI (or MAG2D) decks of r-z rectangles, and the tubular
// stack that repeats along the axis, for the axisymmetric and air-gap tests:
// written as deck text and parsed as a user would run them.

#include "magnetic/model.hpp"
#include "magnetic/solver.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <functional>
#include <map>
#include <numbers>
#include <set>
#include <string>
#include <vector>

namespace tube_deck {
using namespace vibestran::magnetic;

inline constexpr double pi = std::numbers::pi;


template <class T> std::string error_of(const T &result) {
  if (const auto *e = std::get_if<vibestran::Error>(&result))
    return e->message;
  return {};
}

// Grid lines from start through each (end, divisions) segment, uniform in
// each, so that material boundaries fall on lines.
inline std::vector<double> lines(double start, const std::vector<std::pair<double, int>> &segments) {
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
  // MAG2D instead of MAGAXI: the same mesh in the basic x-z plane, depth 1
  // (CID 9: z along basic y, x along basic x). CID 8 has its x axis along
  // basic z, for an MPERIOD that shifts along it.
  bool planar{false};
  int gap{0};          // elements of this PID stay off the side FSETs 1 and 2
  std::string control; // case control after CEND
  std::string cards;   // bulk data
};
inline std::string deck(const Axi &g) {
  const int nr = static_cast<int>(g.r.size()), nz = static_cast<int>(g.z.size());
  const auto grid = [nr](int i, int j) { return 1 + i + nr * j; };
  std::string out = "SOL MAGSTAT\nCEND\n" + g.control + "BEGIN BULK\n" +
                    (g.planar ? "CORD2R,9,0,0.,0.,0.,0.,1.,0.\n,1.,0.,0.\n"
                                "CORD2R,8,0,0.,0.,0.,0.,1.,0.\n,0.,0.,1.\nMAG2D,9,1.\n"
                              : "MAGAXI,0\n") +
                    g.cards;
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
      if (j == 0 && pid != g.gap)
        out += std::format("FSET,1,{},{},{}\n", first, a, b);
      if (j + 2 == nz && pid != g.gap)
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
inline Model parsed(const Axi &g) {
  auto m = parse_string(deck(g));
  EXPECT_TRUE(std::holds_alternative<Model>(m)) << error_of(m);
  return std::holds_alternative<Model>(m) ? std::get<Model>(std::move(m)) : Model{};
}
inline Results solved(const Model &m) {
  auto r = solve(m);
  EXPECT_TRUE(std::holds_alternative<Results>(r)) << error_of(r);
  return std::holds_alternative<Results>(r) ? std::get<Results>(std::move(r)) : Results{};
}
inline double relative(double a, double b) { return std::abs(a - b) / std::abs(b); }

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
// tube_in_body) through `layers` layers (none for 0). Grid IDs depend only
// on the grid's ring and row, so the decks' grids correspond. With gap the
// gap's elements (6.75 to 7.75 mm) are PAIRGAP 9, solved by the air-gap
// element, or as air with mesh; refine divides every radial interval that
// many times; planar makes it the flat stack of the same section (MAG2D,
// r along basic x). With teeth, the cells within that many rows of a coil's
// ends are the sleeve's steel: a slotted stator, which cogs.
inline constexpr double pitch = .02;
struct Tube {
  int pitches{1}, machine{4}, rows{48}; // rows: per pitch
  int offset{0}, window{0}, spacer{0};  // in rows
  bool saturating{false}, half{false}, tube_in_body{true};
  int layers{2};
  double current{0}; // phase A; B and C carry -current / 2
  double move{0};    // the mover (r < 6.75 mm) moved along z, the gap sheared
  bool gap{false}, mesh{false}, planar{false};
  int refine{1}, teeth{0};
  // MFORCE 1 on the stator (coils and sleeve) instead: a flat stack's
  // magnets reach the edge at x = 0, where an axisymmetric stack has its axis.
  bool stator_body{false};
};
inline Axi tube(const Tube &t) {
  Axi g;
  const double sleeve = t.saturating ? .013 : .014, row = pitch / t.rows;
  const int n = t.refine;
  g.r = lines(0, {{.006, 6 * n}, {.00675, n}, {.00775, 4 * n}, {.0125, 5 * n},
                  {sleeve, 2 * n}, {.02, 3 * n}});
  g.planar = t.planar;
  g.gap = t.gap && !t.mesh ? 9 : 0; // meshed air, the gap is on the sides
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
      return r < .00675 ? 5 : t.gap ? 9 : 1;
    if (r < .0125) {
      const double end = std::round(at / (pitch / 3)) * pitch / 3;
      if (std::abs(at - end) < t.teeth * row)
        return 2;
      return 10 + static_cast<int>(std::floor(at / (pitch / 3))); // coil k
    }
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
  if (t.gap)
    g.cards += t.mesh ? "PAIRGAP,9\nPARAM,AIRGAP,MESH\n" : "PAIRGAP,9\n";
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
  for (int pid : present)
    if (t.stator_body ? pid == 2 || pid >= 10
                      : (pid == 4 || pid == 6 || (pid == 5 && t.tube_in_body)))
      body += std::format(",{}", pid);
  g.cards += std::format("MPERIOD,1,{},1,2,{},{},{:.17g}\n", t.pitches % 2 ? "ANTI" : "PERIODIC",
                         t.planar ? 8 : 0, t.machine / t.pitches, t.pitches * pitch);
  if (t.layers > 0)
    g.cards += std::format("ESET,30,PROP{}\nMFORCE,1,BODY,30,0,{}\n", body, t.layers);
  return g;
}
// The largest difference of the two results' potentials at the grids of a
// (whose IDs b shares), each of b's taken times sign.
inline double potential_difference(const Results &a, const Results &b, double sign = 1) {
  double out = 0;
  for (size_t i = 0; i < a.mesh.node_ids.size(); ++i)
    out = std::max(out, std::abs(a.cases[0].potential[static_cast<Eigen::Index>(i)] -
                                 sign * b.cases[0].potential[b.mesh.node_index.at(
                                            a.mesh.node_ids[i])]));
  return out;
}
} // namespace tube_deck
