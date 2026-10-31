#pragma once
// tests/unit/machine_deck.hpp
// Planar MAGSTAT machine models for the air-gap and motion tests: a rotor
// disk and a stator ring, with the air gap between them meshed as PAIRGAP
// elements, written as deck text and parsed as a user would run them.

#include "magnetic/solver.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <numbers>
#include <numeric>
#include <string>
#include <vector>

namespace machine_deck {
using namespace vibestran::magnetic;
using vibestran::Error;

inline constexpr double pi = std::numbers::pi;

inline std::string error_of(const auto &result) {
  if (const auto *e = std::get_if<Error>(&result))
    return e->message;
  return {};
}

// A rotor disk and a stator ring in a MAG2D deck, as rings of grids, with
// the air gap between them meshed in gap_layers layers of property 9 and
// defined by the gap cards (by default PAIRGAP,9). Intermediate gap rings
// have the rotor's divisions; the band against the stator is quads when the
// divisions conform and a strip of triangles when they do not.
struct Machine {
  double rotor{.02}, gap_outer{.021}, stator{.04};
  int rotor_div{48}, stator_div{48}, rotor_rings{6}, stator_rings{8}, gap_layers{1};
  double stator_phase{0}; // fraction of a stator division
  double jitter{0};       // uneven angular spacing, fraction of a division
  double rotation{0};     // rigid rotation of everything about the center
  double rotor_turn{0};   // further rotation of the rotor grids alone
  Vector center{Vector::Zero()};
  int rotor_property{1}, stator_property{1};
  std::string gap{"PAIRGAP,9\n"}; // what PID 9 is
  std::string cards;              // further bulk data
};
struct Geometry {
  std::map<int, Vector> grids;
  std::vector<Element> elements; // element ID = position + 1
  std::vector<int> rotor_edge, stator_edge; // grids on the gap's circles
  std::vector<int> gap_elements;
};

inline Geometry geometry(const Machine &g) {
  Geometry out;
  int next = 1;
  std::map<int, double> angle;
  auto ring = [&](double r, int n, double phase, double turn = 0) {
    std::vector<int> ids;
    for (int k = 0; k < n; ++k) {
      const double t =
          2 * pi * (k + phase + g.jitter * std::sin(1.7 * k + 3 * r / g.stator)) / n +
          g.rotation + turn;
      out.grids[next] = g.center + Vector(r * std::cos(t), r * std::sin(t), 0);
      angle[next] = t;
      ids.push_back(next++);
    }
    return ids;
  };
  auto add = [&](int property, NodeList nodes) {
    const int id = static_cast<int>(out.elements.size()) + 1;
    out.elements.push_back({id, property, std::move(nodes)});
    if (property == 9)
      out.gap_elements.push_back(id);
  };
  auto quads = [&](const std::vector<int> &a, const std::vector<int> &b, int property) {
    const int n = static_cast<int>(a.size());
    for (int k = 0; k < n; ++k)
      add(property, {a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]});
  };
  // Triangles between rings of different divisions, advancing along
  // whichever ring's next grid comes first in angle (angles unwrapped from
  // a's first grid and the b grid nearest it).
  auto zip = [&](const std::vector<int> &a, const std::vector<int> &b, int property) {
    const int na = static_cast<int>(a.size()), nb = static_cast<int>(b.size());
    auto wrapped = [](double x) { return std::remainder(x, 2 * pi); };
    int j0 = 0;
    for (int j = 1; j < nb; ++j)
      if (std::abs(wrapped(angle[b[j]] - angle[a[0]])) <
          std::abs(wrapped(angle[b[j0]] - angle[a[0]])))
        j0 = j;
    auto unwrap = [&](const std::vector<int> &ring, int first, double start) {
      const int n = static_cast<int>(ring.size());
      std::vector<double> out{start};
      for (int k = 1; k < n; ++k) {
        double d = wrapped(angle[ring[(first + k) % n]] - angle[ring[(first + k - 1) % n]]);
        out.push_back(out.back() + (d < 0 ? d + 2 * pi : d));
      }
      out.push_back(start + 2 * pi);
      return out;
    };
    const auto alpha = unwrap(a, 0, angle[a[0]]);
    const auto beta = unwrap(b, j0, angle[a[0]] + wrapped(angle[b[j0]] - angle[a[0]]));
    auto bj = [&](int j) { return b[(j0 + j) % nb]; };
    for (int i = 0, j = 0; i < na || j < nb;) {
      if (j == nb || (i < na && alpha[i + 1] < beta[j + 1])) {
        add(property, {a[i], a[(i + 1) % na], bj(j)});
        ++i;
      } else {
        add(property, {a[i % na], bj(j + 1), bj(j)});
        ++j;
      }
    }
  };
  const int center = next;
  out.grids[next++] = g.center;
  std::vector<int> previous = ring(g.rotor / g.rotor_rings, g.rotor_div, 0, g.rotor_turn);
  for (int k = 0; k < g.rotor_div; ++k)
    add(g.rotor_property, {center, previous[k], previous[(k + 1) % g.rotor_div]});
  for (int j = 2; j <= g.rotor_rings; ++j) {
    auto current = ring(g.rotor * j / g.rotor_rings, g.rotor_div, 0, g.rotor_turn);
    quads(previous, current, g.rotor_property);
    previous = std::move(current);
  }
  out.rotor_edge = previous;
  for (int j = 1; j < g.gap_layers; ++j) {
    auto current = ring(g.rotor + (g.gap_outer - g.rotor) * j / g.gap_layers,
                        g.rotor_div, 0);
    quads(previous, current, 9);
    previous = std::move(current);
  }
  auto stator = ring(g.gap_outer, g.stator_div, g.stator_phase);
  out.stator_edge = stator;
  if (g.stator_div == g.rotor_div && g.stator_phase == 0)
    quads(previous, stator, 9);
  else
    zip(previous, stator, 9);
  previous = std::move(stator);
  for (int j = 1; j <= g.stator_rings; ++j) {
    auto current = ring(g.gap_outer + (g.stator - g.gap_outer) * j / g.stator_rings,
                        g.stator_div, g.stator_phase);
    quads(previous, current, g.stator_property);
    previous = std::move(current);
  }
  return out;
}

inline std::string deck(const Machine &g, const Geometry &geo) {
  std::string out = "SOL MAGSTAT\nCEND\nBEGIN BULK\nMAG2D,0,.1\nMATMAG,1,1.\n"
                    "PMAG2D,1,1\n" +
                    g.gap + g.cards;
  for (const auto &[id, x] : geo.grids)
    out += std::format("GRID,{},,{:.17g},{:.17g},{:.17g}\n", id, x.x(), x.y(), x.z());
  for (const auto &e : geo.elements) {
    out += std::format("{},{},{}", e.nodes.size() == 3 ? "CTRIA3" : "CQUAD4", e.id,
                       e.property);
    for (int n : e.nodes)
      out += std::format(",{}", n);
    out += '\n';
  }
  return out + "ENDDATA\n";
}

inline Result<Model> parse(const Machine &g, const Geometry &geo) {
  return parse_string(deck(g, geo));
}
inline Model parsed(const Machine &g, const Geometry &geo) {
  auto out = parse(g, geo);
  EXPECT_TRUE(std::holds_alternative<Model>(out)) << error_of(out);
  return std::holds_alternative<Model>(out) ? std::get<Model>(std::move(out)) : Model{};
}
inline Model model(const Machine &g) { return parsed(g, geometry(g)); }
inline Mesh mesh_of(const Model &m) {
  auto built = make_mesh(m);
  EXPECT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
  return std::holds_alternative<Mesh>(built) ? std::get<Mesh>(std::move(built)) : Mesh{};
}
inline Results solved(const Model &m) {
  auto out = solve(m);
  EXPECT_TRUE(std::holds_alternative<Results>(out)) << error_of(out);
  return std::holds_alternative<Results>(out) ? std::get<Results>(std::move(out))
                                              : Results{};
}
inline std::vector<std::string> gap_rows(const Results &r) {
  std::vector<std::string> out;
  for (const auto &[label, text] : r.summary.solution.rows)
    if (label == "Air gap")
      out.push_back(text);
  return out;
}

// Uniform remanence br along angle in the rotor (property 2, mu_r = 1),
// and the analytic field with A = 0 at the stator's outer radius R:
// B = (Br/2)(1 - a^2/R^2) inside, and A = D (r - R^2/r) sin(theta),
// D = -Br a^2 / (2 R^2), outside.
inline std::string magnetized_rotor(double br, double angle = 0) {
  return std::format("CORD2R,1,,0.,0.,0.,0.,0.,1.\n,{:.17g},{:.17g},0.\n"
                     "MATMAG,2,1.,,{:.17g}\nPMAG2D,2,2,1,1\n",
                     std::cos(angle), std::sin(angle), br);
}
inline Vector magnet_field_outside(const Machine &g, double br, const Vector &x) {
  const double a = g.rotor, big = g.stator, d = -br * a * a / (2 * big * big);
  const Vector p = x - g.center;
  const double r = p.norm(), t = std::atan2(p.y(), p.x());
  const double br_r = d * (1 - big * big / (r * r)) * std::cos(t),
               bt = -d * (1 + big * big / (r * r)) * std::sin(t);
  return {br_r * std::cos(t) - bt * std::sin(t), br_r * std::sin(t) + bt * std::cos(t), 0};
}
inline Vector centroid(const Model &m, const Element &e) {
  return std::accumulate(e.nodes.begin(), e.nodes.end(), Vector(Vector::Zero()),
                         [&](const Vector &sum, int v) {
                           return Vector(sum + m.nodes.at(v).position);
                         }) /
         static_cast<double>(e.nodes.size());
}
inline std::vector<int> elements_within(const Model &m, double r0, double r1) {
  std::vector<int> out;
  for (const auto &e : m.elements)
    if (const double r = centroid(m, e).norm(); r > r0 && r < r1)
      out.push_back(e.id);
  return out;
}
// Two opposite slots of the stator as a winding of ordinary air (PID 1),
// driven by a 30 A circuit.
inline void wind(Model &m, double turns, double half_width) {
  for (int id : elements_within(m, .026, .034)) {
    auto &e = m.elements[static_cast<size_t>(id - 1)];
    const Vector c = centroid(m, e);
    const double t = std::atan2(c.y(), c.x());
    if (std::abs(t) < half_width || std::abs(std::abs(t) - pi) < half_width) {
      e.property = 1;
      m.element_sets[std::abs(t) < half_width ? 1 : 2].push_back(id);
    }
  }
  m.planar_coils = {{11, 1, turns}, {12, 2, -turns}};
  m.circuits[1] = {1, 1, {{11, 1}, {12, 1}}};
  m.currents = {{1, 1, 30}};
  m.subcases = {Subcase{}};
  m.subcases[0].load = 1;
}
inline const std::string iron_table = "TABLEBH,1\n,0.,0.,.1,100.,.3,2000.,1.,100000.\n,ENDT\n";

} // namespace machine_deck
