#include "magnetic/solver.hpp"
#include "solver/solver_backend.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <numbers>
#include <sstream>
#ifdef VIBESTRAN_HAS_HDF5
#include "../common/hdf5_results.hpp"
using namespace vibestran::test;
#endif

using namespace vibestran::magnetic;
namespace {
// These helpers only generate independent analytical verification fixtures.
Model box(int n, double extent = 1) {
  Model m;
  m.materials[1] = {};
  m.properties[1] = {1};
  m.subcases.push_back({});
  auto node = [n](int i, int j, int k) {
    return 1 + i + (n + 1) * (j + (n + 1) * k);
  };
  for (int k = 0; k <= n; ++k)
    for (int j = 0; j <= n; ++j)
      for (int i = 0; i <= n; ++i)
        m.nodes[node(i, j, k)] = {Vector(extent * (2.0 * i / n - 1),
                                         extent * (2.0 * j / n - 1),
                                         extent * (2.0 * k / n - 1)),
                                  0};
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        std::array<int, 3> permutation{0, 1, 2};
        do {
          std::array<int, 3> position{i, j, k};
          std::array<int, 4> vertices{};
          vertices[0] = node(i, j, k);
          for (int p = 0; p < 3; ++p) {
            ++position[permutation[p]];
            vertices[p + 1] = node(position[0], position[1], position[2]);
          }
          m.elements.push_back(
              {static_cast<int>(m.elements.size()) + 1, 1, vertices});
        } while (std::next_permutation(permutation.begin(), permutation.end()));
      }
  return m;
}
FaceSelection face_selection(const Mesh &mesh, int index, int sign) {
  const auto &f = mesh.faces[index];
  FaceSelection selection;
  selection.element = mesh.cells[f.cells[0]].id;
  selection.nodes.resize(f.nodes.size());
  for (size_t i = 0; i < f.nodes.size(); ++i)
    selection.nodes[i] = mesh.node_ids[f.nodes[i]];
  if (sign < 0)
    std::reverse(selection.nodes.begin(), selection.nodes.end());
  return selection;
}
void open_coil(Model &m, double turns = 1) {
  const auto mesh = std::get<Mesh>(make_mesh(m));
  double xmin = mesh.nodes[0].x(), xmax = xmin;
  for (const auto &x : mesh.nodes) {
    xmin = std::min(xmin, x.x());
    xmax = std::max(xmax, x.x());
  }
  for (const auto &c : mesh.cells)
    m.element_sets[1].push_back(c.id);
  for (size_t i = 0; i < mesh.faces.size(); ++i) {
    const auto &f = mesh.faces[i];
    if (f.cells[1] < 0 && (std::abs(f.center.x() - xmin) < 1e-10 ||
                           std::abs(f.center.x() - xmax) < 1e-10))
      m.face_sets[std::abs(f.center.x() - xmin) < 1e-10 ? 1 : 2].push_back(
          face_selection(mesh, static_cast<int>(i), f.normal.x() > 0 ? 1 : -1));
  }
  m.coils.push_back({1, 1, turns, 1, 2});
  m.currents.push_back({1, 1, 1});
  m.subcases[0].load = 1;
}
void loop(Model &m, double radius = 0.6, bool rectangular = false) {
  const int start = 100000;
  m.nodes[start + 0] = {Vector(-radius, -radius, 0), 0};
  m.nodes[start + 1] = {Vector(radius, -radius, 0), 0};
  m.nodes[start + 2] = {Vector(radius, radius, 0), 0};
  m.nodes[start + 3] = {Vector(-radius, radius, 0), 0};
  m.conductivity[1] = 5.8e7;
  m.wire_properties[1] = {1, rectangular, 0.08, rectangular ? 0.06 : 0};
  m.options.wire_sides = 8;
  m.options.wire_layers = 1;
  m.options.wire_step = 0.4;
  for (int i = 0; i < 4; ++i) {
    Wire w;
    w.id = start + i;
    w.property = 1;
    w.first = start + i;
    w.second = start + (i + 1) % 4;
    if (rectangular)
      w.orientation = Vector::UnitZ();
    m.wires.push_back(w);
  }
  m.wire_paths.push_back({start, start});
  m.currents.push_back({1, start, 1});
  m.subcases[0].load = 1;
  m.probes.push_back({1, Vector::Zero()});
}
std::string error_of(const auto &result) {
  const auto *e = std::get_if<Error>(&result);
  return e ? e->message : "";
}
// GRID and element cards for a model's mesh, so parser-only features (RID
// chains, card layouts) can be exercised on the analytical fixtures.
std::string mesh_cards(const Model &m) {
  std::string out;
  for (const auto &[id, g] : m.nodes)
    out += std::format("GRID,{},,{:.17g},{:.17g},{:.17g}\n", id,
                       g.position.x(), g.position.y(), g.position.z());
  for (const auto &c : m.elements) {
    const char *name = m.options.planar
                           ? (c.nodes.size() == 3 ? "CTRIA3" : "CQUAD4")
                           : (c.nodes.size() == 4   ? "CTETRA"
                              : c.nodes.size() == 6 ? "CPENTA"
                                                    : "CHEXA");
    out += std::format("{},{},{}", name, c.id, c.property);
    for (int n : c.nodes)
      out += std::format(",{}", n);
    out += '\n';
  }
  return out;
}
Vector source_biot(const Source &source, const Vector &x) {
  Vector b = Vector::Zero();
  for (const auto &c : source.mesh.cells)
    for (const auto &s : quadrature(source.mesh, c)) {
      const auto basis = flux_basis(source.mesh, c, s.position);
      Vector j = Vector::Zero();
      for (size_t i = 0; i < c.faces.size(); ++i)
        j += basis[i] * source.flux[c.faces[i]];
      Vector r = x - s.position;
      b += mu0 / (4 * std::numbers::pi) * s.weight * j.cross(r) /
           std::pow(r.norm(), 3);
    }
  return b;
}
struct FieldRegimes {
  std::vector<int> points;
  double maximum{0};
};
// Inspect the actual constitutive sampling points, not cell-average B: a
// nominally nonlinear material can otherwise stay entirely on its first slope.
FieldRegimes field_regimes(const Model &model, const Results &result,
                           const CaseResults &state) {
  const auto &curve = model.bh_tables.at(1);
  FieldRegimes out{std::vector<int>(curve.size(), 0), 0};
  for (const auto &cell : result.mesh.cells) {
    if (model.materials.at(cell.material).bh_table != 1)
      continue;
    for (const auto &q : quadrature(result.mesh, cell)) {
      Vector b = Vector::Zero();
      const auto curls = edge_curls(cell, q.position);
      for (size_t i = 0; i < curls.size(); ++i)
        b += state.potential[cell.edges[i]] * curls[i];
      const double magnitude = b.norm();
      size_t segment = 0;
      while (segment + 1 < curve.size() && magnitude > curve[segment + 1].first)
        ++segment;
      ++out.points[segment];
      out.maximum = std::max(out.maximum, magnitude);
    }
  }
  return out;
}
} // namespace

TEST(MagneticBasis,
     EdgeIntegralsCurlAndNodalGradientsOnSkewReversedTetrahedron) {
  auto built = make_mesh({Vector(0.2, 0.1, -0.1), Vector(1.2, 0.3, 0),
                          Vector(0, 1.1, 0.2), Vector(0.1, 0.2, 1.5)},
                         {{2, 0, 1, 3}});
  ASSERT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
  const auto &mesh = std::get<Mesh>(built);
  const auto &c = mesh.cells[0];
  for (int e = 0; e < 6; ++e) {
    auto edge = mesh.edges[c.edges[e]];
    Vector dx = mesh.nodes[edge[1]] - mesh.nodes[edge[0]],
           mid = (mesh.nodes[edge[1]] + mesh.nodes[edge[0]]) / 2;
    auto basis = edge_basis(c, mid);
    for (int i = 0; i < 6; ++i)
      EXPECT_NEAR(basis[i].dot(dx), i == e ? 1 : 0, 2e-14);
  }
  const Vector gradient(0.3, -0.6, 1.1);
  Vector reconstructed = Vector::Zero(), curl = Vector::Zero();
  const Vector x =
      (mesh.nodes[0] + mesh.nodes[1] + mesh.nodes[2] + mesh.nodes[3]) / 4;
  const auto basis = edge_basis(c, x);
  for (int i = 0; i < 6; ++i) {
    auto edge = mesh.edges[c.edges[i]];
    double a = gradient.dot(mesh.nodes[edge[1]] - mesh.nodes[edge[0]]);
    reconstructed += a * basis[i];
    curl += a * c.curl[i];
  }
  EXPECT_LT((reconstructed - gradient).norm(), 1e-14);
  EXPECT_LT(curl.norm(), 1e-14);
  for (int i = 0; i < 4; ++i) {
    const auto &face = mesh.faces[c.faces[i]];
    auto rt = flux_basis(mesh, c, face.center);
    for (int j = 0; j < 4; ++j)
      EXPECT_NEAR(face.area * face.normal.dot(rt[j]), i == j ? 1 : 0, 2e-14);
  }
}

TEST(MagneticGeometry, IntersectionIntegratesQuadraticsAndConservesPartition) {
  auto m = box(2);
  const auto mesh = std::get<Mesh>(make_mesh(m));
  auto source = std::get<Mesh>(
      make_mesh({Vector(-0.8, -0.7, -0.6), Vector(0.7, -0.6, -0.5),
                 Vector(-0.4, 0.8, -0.3), Vector(-0.2, -0.1, 0.9)},
                {{0, 1, 2, 3}}));
  double expected = 0, actual = 0, volume = 0;
  for (const auto &q : quadrature(source, source.cells[0]))
    expected += q.weight * (1 + q.position.squaredNorm());
  for (const auto &c : mesh.cells)
    for (const auto &q :
         intersection_quadrature(source, source.cells[0], mesh, c)) {
      volume += q.weight;
      actual += q.weight * (1 + q.position.squaredNorm());
    }
  EXPECT_NEAR(volume, source.cells[0].volume, 1e-13);
  EXPECT_NEAR(actual, expected, 1e-13);
  double identical = 0;
  for (const auto &q : intersection_quadrature(source, source.cells[0], source,
                                               source.cells[0]))
    identical += q.weight;
  EXPECT_NEAR(identical, source.cells[0].volume, 1e-14);
}

TEST(MagneticGeometry,
     CubicIntersectionRuleIntegratesEveryMonomialAfterPartition) {
  for (bool transformed : {false, true}) {
    auto transform = [transformed](Vector x) -> Vector {
      if (!transformed)
        return x;
      Matrix a;
      a << -1.2, .3, .1, .2, .9, -.1, 0, .4, 1.1;
      return Vector(8, -4, 3) + a * x;
    };
    auto background = box(2);
    for (auto &[id, g] : background.nodes) {
      (void)id;
      g.position = transform(g.position);
    }
    const auto mesh = std::get<Mesh>(make_mesh(background));
    const auto source = std::get<Mesh>(make_mesh(
        {transform(Vector(-.8, -.7, -.6)), transform(Vector(.7, -.6, -.5)),
         transform(Vector(-.4, .8, -.3)), transform(Vector(-.2, -.1, .9))},
        {{2, 0, 1, 3}}));
    const auto &cell = source.cells[0];
    std::vector<Sample> samples;
    for (const auto &target : mesh.cells) {
      auto part = intersection_quadrature(source, cell, mesh, target, 3);
      samples.insert(samples.end(), part.begin(), part.end());
    }
    auto factorial = [](int n) {
      double f = 1;
      for (int i = 2; i <= n; ++i)
        f *= i;
      return f;
    };
    for (int a = 0; a <= 3; ++a)
      for (int b = 0; b <= 3 - a; ++b)
        for (int c = 0; c <= 3 - a - b; ++c) {
          double integral = 0;
          for (const auto &q : samples) {
            const Vector r = reference_coordinates(cell, q.position);
            integral += q.weight * std::pow(r.x(), a) * std::pow(r.y(), b) *
                        std::pow(r.z(), c);
          }
          const double expected = 6 * cell.volume * factorial(a) *
                                  factorial(b) * factorial(c) /
                                  factorial(a + b + c + 3);
          EXPECT_NEAR(integral, expected, 2e-12 * cell.volume)
              << a << b << c << transformed;
        }
  }
}

TEST(MagneticMaterial, NonlinearEnergyGradientTangentAndVacuumExtrapolation) {
  Material material;
  material.bh_table = 1;
  const std::vector<std::pair<double, double>> curve{
      {0, 0}, {1, 200}, {1.5, 1000}, {2, 10000}};
  for (Vector b : {Vector(0, 0, 0), Vector(0.4, 0.3, 0.1),
                   Vector(1.2, 0.3, 0.5), Vector(2.3, 0.4, 0.1)}) {
    const auto response = material_response(material, curve, b);
    for (int j = 0; j < 3; ++j) {
      Vector step = Vector::Zero();
      step[j] = 1e-6;
      const auto hi = material_response(material, curve, b + step),
                 lo = material_response(material, curve, b - step);
      // Energy subtraction loses absolute digits above the high-field knee;
      // compare its gradient relatively and use a larger differencing step.
      EXPECT_NEAR((hi.energy - lo.energy) / 2e-6, response.h[j],
                  1e-7 + 2e-8 * std::abs(response.h[j]));
      EXPECT_LT(((hi.h - lo.h) / 2e-6 - response.tangent.col(j)).norm(), 0.003);
    }
    EXPECT_EQ(response.extrapolated, b.norm() > 2);
  }
  Material pm;
  pm.mur = 1.05;
  pm.br = 1.2;
  auto response =
      material_response(pm, {}, Vector::Zero(), Vector(0, 0, pm.br));
  EXPECT_NEAR(response.h.z(), -1.2 / (mu0 * 1.05), 1e-9);
}

TEST(MagneticParser, UnitsCoordinatesContinuationAndLoadCombinations) {
  const std::string deck = R"(SOL MAGSTAT
CEND
LOAD=20
BEGIN BULK
PARAM,MAGLEN,1.-3
CORD2R,10,,100,0,0,100,0,1
+,101,0,0
GRID,1,10,0,0,0
GRID,2,10,100,0,0
GRID,3,10,0,100,0
GRID,4,10,0,0,100
CTETRA,1,1,1,2,3,4
PSOLID,1,1
MATMAG,1,1
MATCOND,2,5.8D7
PWIRE,2,2,RECT,2,1
CWIRE,2,2,1,2,0.,0.,1.
WIREPATH,7,2
ICIRC,10,7,-2
LOAD,20,3.,-4.,10
ENDDATA
)";
  auto parsed = parse_string(deck);
  ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
  const auto &m = std::get<Model>(parsed);
  EXPECT_NEAR(m.nodes.at(2).position.x(), 0.2, 1e-15);
  EXPECT_NEAR(m.wire_properties.at(2).width, 0.002, 1e-15);
  auto factors = load_factors(m, 20);
  ASSERT_TRUE((std::holds_alternative<std::map<int, double>>(factors)));
  EXPECT_DOUBLE_EQ((std::get<std::map<int, double>>(factors).at(10)), -12);
  auto cycle = m;
  cycle.combinations[10] = {1, {{1, 20}}};
  EXPECT_TRUE(std::holds_alternative<Error>(load_factors(cycle, 20)));
  auto invalid = deck;
  invalid.replace(invalid.find("5.8D7"), 5, "-1.0 ");
  EXPECT_TRUE(std::holds_alternative<Error>(parse_string(invalid)));
}

TEST(MagneticCurrent, StraightCoilIsUniformConservativeAndHasCorrectTurns) {
  auto model = box(2);
  open_coil(model, 7);
  const auto mesh = std::get<Mesh>(make_mesh(model));
  auto prepared = prepare_sources(model, mesh);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
      << error_of(prepared);
  const auto &source = std::get<std::vector<Source>>(prepared)[0];
  EXPECT_LT(source.conservation_error, 1e-10);
  EXPECT_LT(source.transfer_error, 1e-10);
  for (const auto &c : source.mesh.cells)
    for (const auto &q : quadrature(source.mesh, c)) {
      auto basis = flux_basis(source.mesh, c, q.position);
      Vector j = Vector::Zero();
      for (size_t i = 0; i < c.faces.size(); ++i)
        j += basis[i] * source.flux[c.faces[i]];
      EXPECT_LT((j - Vector(7.0 / 4, 0, 0)).norm(), 1e-10);
    }
  auto reversed = model;
  for (auto &f : reversed.face_sets[1])
    std::swap(f.nodes[1], f.nodes[2]);
  EXPECT_TRUE(std::holds_alternative<Error>(prepare_sources(reversed, mesh)));
}

TEST(MagneticWire, FiniteLoopCurrentDirectionAndBiotSavartField) {
  auto model = box(2, 2);
  loop(model);
  auto prepared = prepare_wires(model);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
      << error_of(prepared);
  const auto &sources = std::get<std::vector<Source>>(prepared);
  ASSERT_EQ(sources.size(), 1u);
  const Vector b = source_biot(sources[0], Vector::Zero());
  const double exact = mu0 * std::sqrt(2.0) / (std::numbers::pi * 0.6);
  EXPECT_NEAR(b.z() / exact, 1, 0.01);
  EXPECT_LT(b.head<2>().norm() / exact, 1e-9);
  auto current = source_currents(model, sources, 1);
  ASSERT_TRUE(std::holds_alternative<std::vector<double>>(current));
  EXPECT_DOUBLE_EQ(std::get<std::vector<double>>(current)[0], 1);
  EXPECT_EQ(sources[0].id, 100000);
  // Reversing a segment's grid order changes nothing: the WIREPATH element
  // fixes the direction and the traversal sign records the reversal.
  std::swap(model.wires[1].first, model.wires[1].second);
  auto reversed = prepare_wires(model);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(reversed))
      << error_of(reversed);
  const auto &same = std::get<std::vector<Source>>(reversed)[0];
  EXPECT_LT((same.flux - sources[0].flux).norm(), 1e-12 * sources[0].flux.norm());
  EXPECT_EQ(same.wire_elements[1], std::make_pair(100001, -1));
  // Naming the reversed segment instead makes its G1->G2 positive, which
  // reverses the circulation.
  model.wire_paths = {{100000, 100001}};
  auto renamed = prepare_wires(model);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(renamed))
      << error_of(renamed);
  const auto &opposite = std::get<std::vector<Source>>(renamed)[0];
  EXPECT_LT((opposite.flux + sources[0].flux).norm(),
            1e-12 * sources[0].flux.norm());
  EXPECT_EQ(opposite.wire_elements[1], std::make_pair(100001, 1));
  EXPECT_LT(source_biot(opposite, Vector::Zero()).z(), 0);
  model.wire_paths = {{100000, 100001}, {100001, 100003}};
  EXPECT_NE(error_of(prepare_wires(model)).find("named by WIREPATH"),
            std::string::npos);
  model.wire_paths.clear();
  EXPECT_NE(error_of(prepare_wires(model)).find("not named by a WIREPATH"),
            std::string::npos);
}

TEST(MagneticWire, RectangularSweepAndInvalidGeometry) {
  auto model = box(2, 2);
  loop(model, 0.6, true);
  auto prepared = prepare_wires(model);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
      << error_of(prepared);
  double volume = 0;
  for (const auto &c : std::get<std::vector<Source>>(prepared)[0].mesh.cells)
    volume += c.volume;
  EXPECT_NEAR(volume, 4 * 1.2 * 0.08 * 0.06, 1e-12);
  auto bad = model;
  bad.wires[1].orientation = Vector::UnitX();
  EXPECT_TRUE(std::holds_alternative<Error>(prepare_wires(bad)));
  bad = model;
  bad.wires.push_back({200000, 1, 100000, 100002, 0, Vector::UnitZ()});
  EXPECT_TRUE(std::holds_alternative<Error>(prepare_wires(bad)));
  bad = model;
  bad.wire_properties[1].width = 2;
  bad.wire_properties[1].height = 2;
  EXPECT_TRUE(std::holds_alternative<Error>(prepare_wires(bad)));
}

TEST(MagneticWire, NonmatchingTransferAndInteriorEndpointsRejected) {
  auto model = box(3);
  model.conductivity[1] = 5.8e7;
  model.wire_properties[1] = {1, false, 0.13, 0};
  model.nodes[100000] = {Vector(-1, 0.07, 0.13), 0};
  model.nodes[100001] = {Vector(1, 0.07, 0.13), 0};
  model.wires.push_back({100000, 1, 100000, 100001, 0, {}});
  model.wire_paths.push_back({100000, 100000});
  model.options.wire_sides = 8;
  model.options.wire_layers = 1;
  model.options.wire_step = 0.5;
  const auto mesh = std::get<Mesh>(make_mesh(model));
  auto prepared = prepare_sources(model, mesh);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
      << error_of(prepared);
  const auto &source = std::get<std::vector<Source>>(prepared)[0];
  EXPECT_LT(source.transfer_error, 1e-10);
  model.nodes[100000].position.x() = -0.8;
  EXPECT_TRUE(std::holds_alternative<Error>(prepare_sources(model, mesh)));
}

TEST(MagneticSolve, UniformFieldEnergyAndZeroNetMaxwellForce) {
  auto model = box(3);
  const Vector field(0.2, -0.3, 0.4);
  model.boundaries[1] = {0, field};
  model.subcases[0].boundary = 1;
  const auto mesh = std::get<Mesh>(make_mesh(model));
  // Closed surface of the interior voxel, with independent outward orientation.
  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    const auto &f = mesh.faces[fi];
    bool on = false;
    for (int k = 0; k < 3; ++k)
      if (std::abs(std::abs(f.center[k]) - 1.0 / 3) < 1e-12)
        on = true;
    if (on && f.center.cwiseAbs().maxCoeff() < 1.0 / 3 + 1e-12)
      model.face_sets[1].push_back(face_selection(
          mesh, static_cast<int>(fi), f.normal.dot(f.center) > 0 ? 1 : -1));
  }
  model.forces.push_back({1, 1});
  model.probes.push_back({1, Vector(0.11, 0.13, 0.17)});
  auto solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &c = std::get<Results>(solved).cases[0];
  for (const auto &b : c.b)
    EXPECT_LT((b - field).norm(), 1e-12);
  EXPECT_NEAR(c.energy, 8 * field.squaredNorm() / (2 * mu0), 1e-8);
  EXPECT_LT(c.forces[0].force.norm(), 1e-8);
  EXPECT_LT(c.forces[0].torque.norm(), 1e-8);
  EXPECT_LT((c.probe_b[0] - field).norm(), 1e-12);
}

TEST(MagneticSolve, LoopEnergyReciprocityReversalAndNonlinearInductance) {
  auto model = box(6, 1.5);
  loop(model);
  model.currents.push_back({2, 100000, -1});
  model.subcases.push_back({.id = 2, .load = 2, .label = "reverse"});
  auto solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &result = std::get<Results>(solved);
  const auto &c = result.cases[0];
  EXPECT_GT(c.probe_b[0].z(), 0);
  EXPECT_NEAR(c.energy, 0.5 * c.sources[0].linkage, 1e-16);
  EXPECT_NEAR(c.differential_inductance(0, 0), c.sources[0].linkage, 1e-16);
  for (size_t i = 0; i < c.b.size(); ++i)
    EXPECT_LT((c.b[i] + result.cases[1].b[i]).norm(), 1e-15);
  model.bh_tables[1] = {
      {0, 0}, {1e-7, 0.001}, {1e-6, 0.02}, {1e-5, 2}, {1e-4, 20}};
  model.materials[2] = {1, 1};
  model.properties[2] = {2};
  for (auto &e : model.elements) {
    Vector center = Vector::Zero();
    for (int n : e.nodes)
      center += model.nodes.at(n).position / 4;
    if (center.cwiseAbs().maxCoeff() < 0.51)
      e.property = 2;
  }
  model.subcases = {{.id = 1, .load = 1},
                    {.id = 2, .load = 3},
                    {.id = 3, .load = 4}};
  model.combinations[3] = {1, {{1.001, 1}}};
  model.combinations[4] = {1, {{0.999, 1}}};
  solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &nonlinear = std::get<Results>(solved).cases;
  const auto regimes =
      field_regimes(model, std::get<Results>(solved), nonlinear[0]);
  EXPECT_GT(regimes.maximum, model.bh_tables.at(1)[2].first);
  EXPECT_GT(regimes.points[1] + regimes.points[2] + regimes.points[3], 0);
  EXPECT_GT(nonlinear[0].iterations, 1);
  EXPECT_LT(nonlinear[0].residual, 1e-8);
  const double finite_difference =
      (nonlinear[1].sources[0].linkage - nonlinear[2].sources[0].linkage) /
      0.002;
  EXPECT_NEAR(nonlinear[0].differential_inductance(0, 0) / finite_difference, 1,
              1e-4);
}

TEST(MagneticSolve, PermanentMagnetAndNonconvergenceErrors) {
  auto model = box(3);
  model.materials[2] = {1.05, 0, 1.2};
  model.properties[2] = {2, 0, 3, true};
  for (auto &e : model.elements) {
    Vector center = Vector::Zero();
    for (int n : e.nodes)
      center += model.nodes.at(n).position / 4;
    if (center.cwiseAbs().maxCoeff() < 0.34)
      e.property = 2;
  }
  model.probes = {{1, Vector(0.01, 0.02, 0.03)}};
  auto solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  EXPECT_GT(std::get<Results>(solved).cases[0].probe_b[0].z(), 0.2);
  model.properties[2].axis = -3;
  auto reverse = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(reverse)) << error_of(reverse);
  EXPECT_LT((std::get<Results>(solved).cases[0].probe_b[0] +
             std::get<Results>(reverse).cases[0].probe_b[0])
                .norm(),
            1e-12);
  model = box(6, 1.5);
  loop(model);
  model.materials[2] = {1, 1};
  model.properties[2] = {2};
  model.bh_tables[1] = {{0, 0}, {1e-7, 0.001}, {1e-6, 0.02}, {1e-5, 2}};
  for (auto &e : model.elements) {
    Vector center = Vector::Zero();
    for (int n : e.nodes)
      center += model.nodes.at(n).position / 4;
    if (center.cwiseAbs().maxCoeff() < 0.51)
      e.property = 2;
  }
  model.options.max_iterations = 1;
  EXPECT_TRUE(std::holds_alternative<Error>(solve(model)));
}

TEST(MagneticCurrent, ClosedVolumetricCoilCutAndTurnScaling) {
  auto model = box(6, 1.5);
  const auto mesh = std::get<Mesh>(make_mesh(model));
  for (const auto &c : mesh.cells) {
    Vector center = Vector::Zero();
    for (int n : c.nodes)
      center += mesh.nodes[n] / 4;
    const double radius = center.head<2>().cwiseAbs().maxCoeff();
    if (radius > 0.5 && radius < 1 && std::abs(center.z()) < 0.5)
      model.element_sets[1].push_back(c.id);
  }
  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    const auto &f = mesh.faces[fi];
    if (std::abs(f.center.y()) < 1e-12 && f.center.x() > 0.5 &&
        f.center.x() < 1 && std::abs(f.center.z()) < 0.5)
      model.face_sets[1].push_back(face_selection(mesh, static_cast<int>(fi),
                                                  f.normal.y() > 0 ? 1 : -1));
  }
  model.coils.push_back({1, 1, 20, 1, 0});
  auto prepared = prepare_sources(model, mesh);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
      << error_of(prepared);
  const auto &source = std::get<std::vector<Source>>(prepared)[0];
  EXPECT_LT(source.conservation_error, 1e-10);
  EXPECT_LT(source.transfer_error, 1e-10);
  auto section = selected_faces(model, source.mesh, 1);
  ASSERT_TRUE(
      (std::holds_alternative<std::vector<std::pair<int, int>>>(section)));
  double current = 0;
  for (auto [f, sign] : std::get<std::vector<std::pair<int, int>>>(section))
    current += sign * source.flux[f];
  EXPECT_NEAR(current, 20, 1e-10);
  EXPECT_GT(source_biot(source, Vector::Zero()).z(), 0);
  for (auto &f : model.face_sets[1])
    std::swap(f.nodes[1], f.nodes[2]);
  auto reversed = prepare_sources(model, mesh);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(reversed))
      << error_of(reversed);
  EXPECT_LT(
      (std::get<std::vector<Source>>(reversed)[0].coupling + source.coupling)
          .norm(),
      1e-10);
}

TEST(MagneticSolve, MutualInductanceReciprocityAndSuperposition) {
  auto model = box(4, 1.5);
  loop(model);
  for (int i = 0; i < 4; ++i) {
    model.nodes[100000 + i].position.z() = -0.3;
    model.nodes[200000 + i] = {
        model.nodes[100000 + i].position + Vector(0, 0, 0.6), 0};
    model.wires.push_back(
        {200000 + i, 1, 200000 + i, 200000 + (i + 1) % 4, 0, {}});
  }
  model.wire_paths.push_back({200000, 200000});
  model.currents.push_back({2, 200000, 2});
  model.combinations[3] = {1, {{1, 1}, {1, 2}}};
  model.subcases = {{.id = 1, .load = 1},
                    {.id = 2, .load = 2},
                    {.id = 3, .load = 3}};
  auto solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &cases = std::get<Results>(solved).cases;
  const auto &inductance = cases[0].differential_inductance;
  EXPECT_GT(inductance(0, 1), 0);
  EXPECT_NEAR(inductance(0, 1), inductance(1, 0), 1e-18);
  EXPECT_LT(std::abs(inductance(0, 1)),
            std::sqrt(inductance(0, 0) * inductance(1, 1)));
  for (size_t i = 0; i < cases[0].b.size(); ++i)
    EXPECT_LT((cases[0].b[i] + cases[1].b[i] - cases[2].b[i]).norm(), 1e-14);
}

// Two loops in series with opposite polarity, representing two parallel
// branches. Circuit quantities must follow from the independently computed
// source matrix: lambda_c = sum P_i lambda_i / a and
// L_c = sum P_i P_j L_ij / a^2.
TEST(MagneticCircuit, SeriesParallelCircuitsCombineSourceLinkageAndInductance) {
  auto model = box(4, 1.5);
  loop(model);
  for (int i = 0; i < 4; ++i) {
    model.nodes[100000 + i].position.z() = -0.3;
    model.nodes[200000 + i] = {
        model.nodes[100000 + i].position + Vector(0, 0, 0.6), 0};
    model.wires.push_back(
        {200000 + i, 1, 200000 + i, 200000 + (i + 1) % 4, 0, {}});
  }
  model.wire_paths.push_back({200000, 200000});
  model.circuits[7] = {7, 2, {{100000, 1}, {200000, -1}}};
  const double current = 3;
  model.currents = {{1, 7, current}};
  auto solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &r = std::get<Results>(solved);
  ASSERT_EQ(r.circuits.size(), 1U);
  EXPECT_EQ(r.circuits[0].id, 7);
  EXPECT_TRUE(r.circuits[0].closed);
  const auto &c = r.cases[0];
  EXPECT_DOUBLE_EQ(c.sources[0].current, current / 2);
  EXPECT_DOUBLE_EQ(c.sources[1].current, -current / 2);
  EXPECT_DOUBLE_EQ(c.circuits[0].current, current);
  const auto &l = c.differential_inductance;
  ASSERT_GT(std::abs(l(0, 1)), .01 * l(0, 0)); // the coupling matters
  const double combined = (l(0, 0) - l(0, 1) - l(1, 0) + l(1, 1)) / 4;
  const double inductance = c.circuit_inductance(0, 0);
  EXPECT_NEAR(inductance / combined, 1, 1e-12);
  EXPECT_NEAR(c.circuits[0].linkage /
                  ((c.sources[0].linkage - c.sources[1].linkage) / 2),
              1, 1e-12);
  // Linear material: lambda = L I and W = W' = L I^2 / 2.
  EXPECT_NEAR(c.circuits[0].linkage / (inductance * current), 1, 1e-10);
  EXPECT_NEAR(c.energy / (.5 * inductance * current * current), 1, 1e-10);
  // The same members in series carry the full current and have four times
  // the inductance.
  model.circuits[7].parallel = 1;
  auto series = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(series)) << error_of(series);
  const auto &s = std::get<Results>(series).cases[0];
  EXPECT_DOUBLE_EQ(s.sources[0].current, current);
  EXPECT_NEAR(s.circuit_inductance(0, 0) / (4 * inductance), 1, 1e-12);
  EXPECT_NEAR(s.energy / (4 * c.energy), 1, 1e-10);
}

TEST(MagneticParser, PropertySetsCircuitsAndSourceNamespaceAreValidated) {
  auto mesh = box(1);
  for (auto &c : mesh.elements)
    c.property = c.id % 2 ? 1 : 2;
  const std::string head = "SOL MAGSTAT\nCEND\nBEGIN BULK\nPSOLID,1,1\n"
                           "PSOLID,2,1\nPSOLID,3,1\nMATMAG,1,1\n" +
                           mesh_cards(mesh);
  auto parse = [&](const std::string &cards) {
    return parse_string(head + cards + "ENDDATA\n");
  };
  // A property row selects exactly the elements an explicit row lists, and
  // rows of one SID accumulate.
  auto rows = parse("ESET,5,PROP,2\nESET,6,2,4,6\nESET,7,PROP,1\nESET,7,PROP,"
                    "2\n");
  ASSERT_TRUE(std::holds_alternative<Model>(rows)) << error_of(rows);
  const auto &sets = std::get<Model>(rows).element_sets;
  EXPECT_EQ(sets.at(5), sets.at(6));
  EXPECT_EQ(sets.at(7).size(), 6U);
  EXPECT_NE(error_of(parse("ESET,5,PROP,3\n")).find("PROP 3 selects no"),
            std::string::npos);
  EXPECT_NE(error_of(parse("ESET,5,PROP,1,THRU,2\nESET,5,4\n"))
                .find("element 4 more than once"),
            std::string::npos);
  // Sources and circuits share one namespace, and current goes on the
  // circuit rather than on its members.
  const std::string wires = "MATCOND,2,5.8e7\nPWIRE,2,2,ROUND,.01\n"
                            "GRID,90,,-.5,-.5,0\nGRID,91,,.5,-.5,0\n"
                            "GRID,92,,0,.5,0\nCWIRE,90,2,90,91\n"
                            "CWIRE,91,2,92,91\nCWIRE,92,2,92,90\n"
                            "WIREPATH,10,90\n";
  EXPECT_TRUE(std::holds_alternative<Model>(
      parse(wires + "CIRCUIT,20,,10,-1\nICIRC,1,20,2.\n")));
  EXPECT_TRUE(std::holds_alternative<Model>(parse(wires + "ICIRC,1,10,2.\n")));
  for (const auto &[cards, message] :
       std::vector<std::pair<std::string, std::string>>{
           {"CIRCUIT,10,,10,1\n", "unique among"},
           {"CIRCUIT,20,,11,1\n", "member 11 is not"},
           {"CIRCUIT,20,,10,2\n", "+1 or -1"},
           {"CIRCUIT,20,0,10,1\n", "PARALLEL >= 1"},
           {"CIRCUIT,20,,10,1\nCIRCUIT,21,,10,1\n", "more than one"},
           {"CIRCUIT,20,,10,1\nICIRC,1,10,2.\n", "give the current on the"},
           {"ICIRC,1,30,2.\n", "undefined circuit or source 30"},
           {"WIREPATH,11,93\n", "undefined CWIRE 93"}}) {
    SCOPED_TRACE(cards);
    EXPECT_NE(error_of(parse(wires + cards)).find(message), std::string::npos)
        << error_of(parse(wires + cards));
  }
}

TEST(MagneticIO, IncludesFixedFieldsAndResultFieldValues) {
  const auto directory =
      std::filesystem::temp_directory_path() / "vibestran_magnetic_io";
  std::filesystem::create_directories(directory);
  const auto path = directory / "model.bdf";
  {
    std::ofstream out(path);
    out << "SOL MAGSTAT\nCEND\nMAGBC=1\nBEGIN BULK\nINCLUDE "
           "'volume.inc'\nMBC,1,0,UNIFORM,0,0,0.25\nMPROBE,1,0,0.1,0.1,"
           "0.1\nENDDATA\n";
    std::ofstream volume(directory / "volume.inc");
    volume << "GRID           1              0.      0.      0.\n"
              "GRID,2,,1,0,0\nGRID,3,,0,1,0\nGRID,4,,0,0,+1\n"
              "CTETRA,1,1,1,2,3,4\nPSOLID,1,1\nMATMAG,1,1\n";
  }
  auto detected = is_magnetic_deck(path);
  ASSERT_TRUE(std::holds_alternative<bool>(detected));
  EXPECT_TRUE(std::get<bool>(detected));
  const auto wrapper = directory / "wrapper.bdf";
  {
    std::ofstream out(wrapper);
    out << "INCLUDE 'model.bdf'\n";
  }
  auto included = is_magnetic_deck(wrapper);
  ASSERT_TRUE(std::holds_alternative<bool>(included));
  EXPECT_TRUE(std::get<bool>(included));
  auto parsed = parse_file(path);
  ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
  auto solved = solve(std::get<Model>(parsed));
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
#ifdef VIBESTRAN_HAS_HDF5
  EXPECT_FALSE(write_results(std::get<Model>(parsed), std::get<Results>(solved),
                             directory / "result.F06")
                   .has_value());
  // The included fixed-format grids carry the uniform field to the probe.
  const auto probe = h5_table(directory / "result.mag.vtkhdf",
                              "/MAGSTAT/subcases/1/probes/B_T");
  ASSERT_EQ(probe.dims, (std::vector<hsize_t>{1, 3}));
  EXPECT_NEAR(probe.at(0, 0), 0, 1e-14);
  EXPECT_NEAR(probe.at(0, 1), 0, 1e-14);
  EXPECT_NEAR(probe.at(0, 2), .25, 1e-14);
#else
  // Without HDF5 the results cannot be written, and the error says why.
  EXPECT_FALSE(output_available());
  const auto unwritten = write_results(
      std::get<Model>(parsed), std::get<Results>(solved), directory / "result.F06");
  ASSERT_TRUE(unwritten.has_value());
  EXPECT_NE(unwritten->message.find("no HDF5"), std::string::npos);
#endif
  std::ofstream cycle(directory / "cycle.inc");
  cycle << "INCLUDE 'model.bdf'\n";
  cycle.close();
  std::ofstream volume(directory / "volume.inc");
  volume << "INCLUDE 'cycle.inc'\n";
  volume.close();
  EXPECT_TRUE(std::holds_alternative<Error>(parse_file(path)));
  std::filesystem::remove_all(directory);
}

TEST(MagneticIO, IndentedCaseControlLabelsKeepTheirTextAndCase) {
  // Case control is normally indented; the label must be cut after the '='
  // of the statement itself, whatever the leading whitespace.
  const std::string bulk = "BEGIN BULK\nGRID,1,,0,0,0\nGRID,2,,1,0,0\n"
                           "GRID,3,,0,1,0\nGRID,4,,0,0,1\nCTETRA,1,1,1,2,3,4\n"
                           "PSOLID,1,1\nMATMAG,1,1\nMBC,1,0,UNIFORM,0,0,0.25\n"
                           "ENDDATA\n";
  for (const std::string indent : {"", "  ", "\t", "        "}) {
    SCOPED_TRACE(indent.size());
    const std::string deck = "SOL MAGSTAT\nCEND\nMAGBC=1\n" + indent +
                             "LABEL = Global Field\nSUBCASE 1\n" + indent +
                             "LABEL=Coil drive A\nSUBCASE 2\n" + bulk;
    auto parsed = parse_string(deck);
    ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
    const auto &subcases = std::get<Model>(parsed).subcases;
    ASSERT_EQ(subcases.size(), 2u);
    EXPECT_EQ(subcases[0].label, "Coil drive A");
    EXPECT_EQ(subcases[1].label, "Global Field");
  }
}

TEST(MagneticIO, DeckDetectionLeavesUnreadableIncludesToTheOwningParser) {
  const auto directory =
      std::filesystem::temp_directory_path() / "vibestran_magnetic_detect";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto write = [&](const std::string &name, const std::string &text) {
    std::ofstream(directory / name) << text;
    return directory / name;
  };
  const auto detect = [](const std::filesystem::path &path) {
    auto detected = is_magnetic_deck(path);
    EXPECT_TRUE(std::holds_alternative<bool>(detected)) << error_of(detected);
    return std::holds_alternative<bool>(detected) && std::get<bool>(detected);
  };
  // A structural deck is identified without opening its broken or cyclic
  // includes; those belong to the structural reader.
  write("loop.inc", "INCLUDE 'loop.inc'\n");
  EXPECT_FALSE(detect(write("structural.bdf",
                            "INCLUDE 'missing.inc'\nINCLUDE 'loop.inc'\n"
                            "SOL 101\nCEND\nBEGIN BULK\nINCLUDE "
                            "'missing_bulk.inc'\nENDDATA\n")));
  // SOL after an unreadable include is still found. The magnetic reader then
  // reports the include rather than silently dropping it.
  const auto magnetic = write("magnetic.bdf", "INCLUDE 'missing.inc'\n"
                                              "SOL MAGSTAT\nCEND\nBEGIN "
                                              "BULK\nENDDATA\n");
  EXPECT_TRUE(detect(magnetic));
  auto parsed = parse_file(magnetic);
  ASSERT_TRUE(std::holds_alternative<Error>(parsed));
  EXPECT_NE(std::get<Error>(parsed).message.find("cannot open INCLUDE"),
            std::string::npos)
      << std::get<Error>(parsed).message;
  // Detection ends at BEGIN, also when BEGIN BULK comes from an include, so a
  // later SOL-like bulk line cannot reclassify the deck.
  write("executive.inc", "SOL 101\n");
  write("front.inc", "CEND\nBEGIN BULK\n");
  EXPECT_FALSE(detect(write("executive.bdf", "INCLUDE 'executive.inc'\n")));
  EXPECT_FALSE(detect(write("bulk_first.bdf", "INCLUDE 'front.inc'\n"
                                              "SOL MAGSTAT\nENDDATA\n")));
  std::filesystem::remove_all(directory);
}

TEST(MagneticSolve, MeshRenumberingAndWireRigidRotationPreservePhysics) {
  auto model = box(4, 1.5);
  loop(model);
  auto first = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(first)) << error_of(first);
  std::map<int, Grid> nodes;
  for (const auto &[id, g] : model.nodes)
    nodes[300000 - id] = g;
  model.nodes = std::move(nodes);
  for (auto &e : model.elements) {
    for (int &n : e.nodes)
      n = 300000 - n;
    std::swap(e.nodes[0], e.nodes[1]);
  }
  for (auto &w : model.wires) {
    w.first = 300000 - w.first;
    w.second = 300000 - w.second;
  }
  auto second = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(second)) << error_of(second);
  const auto &a = std::get<Results>(first).cases[0];
  const auto &b = std::get<Results>(second).cases[0];
  EXPECT_NEAR(a.energy, b.energy, 1e-17);
  for (size_t i = 0; i < a.b.size(); ++i)
    EXPECT_LT((a.b[i] - b.b[i]).norm(), 1e-15);
  const Matrix rotation =
      Eigen::AngleAxisd(0.7, Vector(1, 2, 3).normalized()).toRotationMatrix();
  for (auto &[id, g] : model.nodes) {
    (void)id;
    g.position = rotation * g.position;
  }
  auto third = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(third)) << error_of(third);
  const auto &c = std::get<Results>(third).cases[0];
  // The polygonal circular wire frame may rotate within the section; its
  // finite-section geometric error is bounded separately from gauge invariance.
  EXPECT_NEAR(c.energy / b.energy, 1, 0.002);
  EXPECT_LT((c.probe_b[0] - rotation * b.probe_b[0]).norm() /
                b.probe_b[0].norm(),
            0.002);
}

TEST(MagneticForce, LoopTorqueApproachesMagneticMomentCrossUniformField) {
  double previous = 1;
  for (int n : {8, 16}) {
    auto model = box(n, 1.2);
    loop(model);
    model.boundaries[1] = {0, Vector(0, 0.001, 0)};
    model.subcases[0].boundary = 1;
    const auto mesh = std::get<Mesh>(make_mesh(model));
    for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
      const auto &f = mesh.faces[fi];
      if (std::abs(f.center.cwiseAbs().maxCoeff() - .9) < 1e-12)
        model.face_sets[1].push_back(face_selection(
            mesh, static_cast<int>(fi), f.normal.dot(f.center) > 0 ? 1 : -1));
    }
    model.forces.push_back({1, 1});
    auto solved = solve(model);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    const auto &force = std::get<Results>(solved).cases[0].forces[0];
    const Vector expected(-4 * .6 * .6 * .001, 0, 0);
    const double error = (force.torque - expected).norm() / expected.norm();
    EXPECT_LT(error, previous);
    previous = error;
    EXPECT_LT(force.force.norm(), 1e-5);
    if (n == 16)
      EXPECT_LT(error, 0.03) << force.torque.transpose();
  }
}

TEST(MagneticParser, LargeFieldsAndExecutiveWhitespacePreserveGeometry) {
  std::string deck = "SOL\tMAGSTAT\nCEND\nBEGIN   BULK\nGRID,1,,0,0,0\n";
  deck += std::format("{:<8}{:>16}{:>16}{:>16}{:>16}\n", "GRID*", "2", "",
                      "1.25-1", "0.");
  deck += std::format("{:<8}{:>16}\n", "*", "0.");
  deck += "GRID,3,,0,1,0\nGRID,4,,0,0,1\nCTETRA,1,1,1,2,3,4\nPSOLID,1,"
          "1\nMATMAG,1,1\nENDDATA\n";
  auto parsed = parse_string(deck);
  ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
  auto mesh = make_mesh(std::get<Model>(parsed));
  ASSERT_TRUE(std::holds_alternative<Mesh>(mesh)) << error_of(mesh);
  EXPECT_NEAR(std::get<Mesh>(mesh).cells[0].volume, 0.125 / 6, 1e-16);
}

TEST(MagneticMaterial, PermanentMagnetCoordinateFrameMatchesBasicRemanence) {
  auto model = box(3);
  model.materials[2].br = 1.2;
  model.properties[2] = {2, 0, 1, true};
  for (auto &cell : model.elements) {
    Vector center = Vector::Zero();
    for (int node : cell.nodes)
      center += model.nodes.at(node).position / 4;
    if (center.cwiseAbs().maxCoeff() < 0.34)
      cell.property = 2;
  }
  model.probes.push_back({1, Vector::Zero()});
  auto basic = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(basic)) << error_of(basic);
  model.coordinates[1] = {0, 'R', Vector::Zero(), Vector::UnitX(),
                          Vector::UnitY()};
  // Local z of this frame is basic x.
  model.properties[2] = {2, 1, 3, true};
  auto local = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(local)) << error_of(local);
  const auto &a = std::get<Results>(basic).cases[0];
  const auto &b = std::get<Results>(local).cases[0];
  EXPECT_GT(a.probe_b[0].x(), 0.2);
  EXPECT_NEAR(a.energy, b.energy, 1e-8);
  for (size_t i = 0; i < a.b.size(); ++i)
    EXPECT_LT((a.b[i] - b.b[i]).norm(), 1e-12);
}

// Two poles oriented by child systems of one rotor system must match the
// same magnets given directly in basic frames. Rotating the rotor system is
// then the only change needed to move every magnet.
std::string two_pole_deck(bool planar, const std::string &frames,
                          const std::string &properties) {
  auto model = planar ? Model{} : box(4);
  if (planar) {
    model.options.planar = true;
    auto node = [](int i, int j) { return 1 + i + 9 * j; };
    for (int j = 0; j <= 8; ++j)
      for (int i = 0; i <= 8; ++i)
        model.nodes[node(i, j)] = {Vector(-1 + i / 4., -1 + j / 4., 0), 0};
    for (int j = 0; j < 8; ++j)
      for (int i = 0; i < 8; ++i)
        model.elements.push_back({static_cast<int>(model.elements.size()) + 1,
                                  1,
                                  {node(i, j), node(i + 1, j),
                                   node(i + 1, j + 1), node(i, j + 1)}});
  }
  for (auto &cell : model.elements) {
    Vector center = Vector::Zero();
    for (int n : cell.nodes)
      center += model.nodes.at(n).position /
                static_cast<double>(cell.nodes.size());
    if (std::abs(center.y()) < .5 && std::abs(center.z()) < .5 &&
        std::abs(center.x()) < .5)
      cell.property = center.x() > 0 ? 2 : 3;
  }
  const std::string property = planar ? "PMAG2D" : "PMAG";
  std::string deck = "SOL MAGSTAT\nCEND\nBEGIN BULK\n";
  if (planar)
    deck += "MAG2D,0,0.1\nPMAG2D,1,1\n";
  else
    deck += "PSOLID,1,1\n";
  deck += "MATMAG,1,1\nMATMAG,2,1.05,,1.2\n" + frames +
          std::vformat(properties, std::make_format_args(property)) +
          mesh_cards(model) + "ENDDATA\n";
  return deck;
}

TEST(MagneticMaterial, RotorChildFramesMatchEquivalentBasicFramesIn2DAnd3D) {
  const double alpha = 35, pole = 20, degree = std::numbers::pi / 180;
  auto cosine = [&](double a) { return std::cos(a * degree); };
  auto sine = [&](double a) { return std::sin(a * degree); };
  // Rotor frame 10 rotated by alpha about z; pole frames 11 and 12 are given
  // in rotor coordinates. Pole 2 uses the rotor's negative y axis.
  const std::string rotor = std::format(
      "CORD2R,10,,0,0,0,0,0,1\n,{:.17g},{:.17g},0\n"
      "CORD2R,11,10,0,0,0,0,0,1\n,{:.17g},{:.17g},0\n"
      "CORD2R,12,10,0,0,0,0,0,1\n,1,0,0\n",
      cosine(alpha), sine(alpha), cosine(pole), sine(pole));
  const std::string chained = "{0},2,2,11,1\n{0},3,2,12,-2\n";
  // The same directions written once in basic: x axes along R(alpha+pole)x
  // and -R(alpha)y respectively.
  const std::string basic = std::format(
      "CORD2R,21,,0,0,0,0,0,1\n,{:.17g},{:.17g},0\n"
      "CORD2R,22,,0,0,0,0,0,1\n,{:.17g},{:.17g},0\n",
      cosine(alpha + pole), sine(alpha + pole), sine(alpha), -cosine(alpha));
  const std::string direct = "{0},2,2,21,1\n{0},3,2,22,1\n";
  // Reversing pole 2's frame by 180 degrees replaces the negative axis.
  const std::string reversed = rotor + "CORD2R,13,10,0,0,0,0,0,1\n,-1,0,0\n";
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar);
    std::vector<Results> results;
    for (const auto &deck :
         {two_pole_deck(planar, rotor, chained),
          two_pole_deck(planar, basic, direct),
          two_pole_deck(planar, reversed, "{0},2,2,11,1\n{0},3,2,13,2\n")}) {
      auto parsed = parse_string(deck);
      ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
      auto solved = solve(std::get<Model>(parsed));
      ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
      results.push_back(std::get<Results>(std::move(solved)));
    }
    const auto &reference = results[0].cases[0];
    EXPECT_GT(reference.energy, 0);
    for (size_t k = 1; k < results.size(); ++k) {
      const auto &other = results[k].cases[0];
      EXPECT_NEAR(other.energy / reference.energy, 1, 1e-12);
      for (size_t i = 0; i < reference.b.size(); ++i)
        EXPECT_LT((other.b[i] - reference.b[i]).norm(), 1e-12);
    }
    // The pole fields really differ, so a mixed-up axis cannot pass.
    const auto &cells = results[0].mesh.cells;
    Vector first = Vector::Zero(), second = Vector::Zero();
    for (size_t i = 0; i < cells.size(); ++i) {
      if (cells[i].property == 2)
        first += reference.b[i];
      if (cells[i].property == 3)
        second += reference.b[i];
    }
    EXPECT_GT(first.normalized().dot(Vector(cosine(alpha + pole),
                                            sine(alpha + pole), 0)),
              .9);
    EXPECT_GT(second.normalized().dot(Vector(sine(alpha), -cosine(alpha), 0)),
              .9);
  }
}

namespace {
Model brick_box(int n, bool wedges = false, bool mixed = false) {
  auto m = box(n);
  m.elements.clear();
  auto node = [n](int i, int j, int k) {
    return 1 + i + (n + 1) * (j + (n + 1) * k);
  };
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        NodeList q{node(i, j, k),
                   node(i + 1, j, k),
                   node(i + 1, j + 1, k),
                   node(i, j + 1, k),
                   node(i, j, k + 1),
                   node(i + 1, j, k + 1),
                   node(i + 1, j + 1, k + 1),
                   node(i, j + 1, k + 1)};
        if (wedges || (mixed && i == 0)) {
          m.elements.push_back({static_cast<int>(m.elements.size()) + 1,
                                1,
                                {q[0], q[1], q[2], q[4], q[5], q[6]}});
          m.elements.push_back({static_cast<int>(m.elements.size()) + 1,
                                1,
                                {q[0], q[2], q[3], q[4], q[6], q[7]}});
        } else
          m.elements.push_back({static_cast<int>(m.elements.size()) + 1, 1, q});
      }
  return m;
}
Model planar_box(int n, bool triangles = false, bool mixed = false) {
  Model m;
  m.options.planar = true;
  m.options.depth = .3;
  m.materials[1] = {};
  m.properties[1] = {1};
  m.subcases.push_back({});
  auto node = [n](int i, int j) { return 1 + i + (n + 1) * j; };
  for (int j = 0; j <= n; ++j)
    for (int i = 0; i <= n; ++i)
      m.nodes[node(i, j)] = {Vector(-1 + 2. * i / n, -1 + 2. * j / n, 0), 0};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      NodeList q{node(i, j), node(i + 1, j), node(i + 1, j + 1),
                 node(i, j + 1)};
      if (triangles || (mixed && i < n / 2)) {
        m.elements.push_back(
            {static_cast<int>(m.elements.size()) + 1, 1, {q[0], q[1], q[2]}});
        m.elements.push_back(
            {static_cast<int>(m.elements.size()) + 1, 1, {q[0], q[2], q[3]}});
      } else
        m.elements.push_back({static_cast<int>(m.elements.size()) + 1, 1, q});
    }
  return m;
}
} // namespace

TEST(MagneticMappedBasis, HexAndWedgeMomentsCurlGradientAndFluxOnTaperedCells) {
  for (bool wedge : {false, true})
    for (bool reflected : {false, true}) {
      auto model = brick_box(1, wedge);
      if (wedge)
        model.elements.resize(1);
      for (auto &[id, g] : model.nodes) {
        (void)id;
        g.position.x() *= (1 - .15 * (g.position.z() + 1));
        g.position.y() *= (1 - .15 * (g.position.z() + 1));
        if (reflected)
          g.position.x() *= -1;
      }
      auto built = make_mesh(model);
      ASSERT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
      const auto &mesh = std::get<Mesh>(built);
      const auto &c = mesh.cells[0];
      for (size_t e = 0; e < c.edges.size(); ++e) {
        auto edge = mesh.edges[c.edges[e]];
        Vector dx = mesh.nodes[edge[1]] - mesh.nodes[edge[0]];
        auto basis =
            edge_basis(c, (mesh.nodes[edge[0]] + mesh.nodes[edge[1]]) / 2);
        for (size_t i = 0; i < basis.size(); ++i)
          EXPECT_NEAR(basis[i].dot(dx), i == e ? 1 : 0, 1e-12);
      }
      for (size_t face = 0; face < c.faces.size(); ++face) {
        const auto &f = mesh.faces[c.faces[face]];
        std::vector<double> integrals(c.faces.size(), 0);
        for (const auto &q : face_quadrature(mesh, f)) {
          auto basis = flux_basis(mesh, c, q.position);
          for (size_t i = 0; i < basis.size(); ++i)
            integrals[i] += q.weight * basis[i].dot(f.normal);
        }
        for (size_t i = 0; i < integrals.size(); ++i)
          EXPECT_NEAR(integrals[i], i == face ? 1 : 0, 1e-10);
      }
      const Vector r(.2, .3, .1);
      const auto [x, j] = mapping(c, r);
      const auto shape = reference_shape(c, r);
      const auto basis = edge_basis(c, x), curls = edge_curls(c, x);
      Vector exact = Vector::Zero(), interpolated = Vector::Zero(),
             curl = Vector::Zero();
      std::map<int, double> phi;
      for (size_t i = 0; i < c.nodes.size(); ++i) {
        phi[c.nodes[i]] = std::sin(1.3 * i + .2);
        exact += phi[c.nodes[i]] * j.inverse().transpose() * shape.gradients[i];
      }
      for (size_t e = 0; e < c.edges.size(); ++e) {
        auto edge = mesh.edges[c.edges[e]];
        const double value = phi[edge[1]] - phi[edge[0]];
        interpolated += value * basis[e];
        curl += value * curls[e];
        Matrix derivative;
        for (int axis = 0; axis < 3; ++axis) {
          const Vector delta = 1e-5 * Vector::Unit(axis);
          derivative.col(axis) =
              (edge_basis(c, x + delta)[e] - edge_basis(c, x - delta)[e]) /
              2e-5;
        }
        Vector numerical(derivative(2, 1) - derivative(1, 2),
                         derivative(0, 2) - derivative(2, 0),
                         derivative(1, 0) - derivative(0, 1));
        EXPECT_LT((numerical - curls[e]).norm(), 1e-9);
      }
      EXPECT_LT((exact - interpolated).norm(), 1e-12);
      EXPECT_LT(curl.norm(), 1e-12);
    }
}

TEST(MagneticSolids,
     UniformFieldEnergyAndConservedCoilsOnHexWedgeAndMixedMeshes) {
  const Vector field(.2, -.3, .1);
  for (int kind = 0; kind < 3; ++kind) {
    auto model = brick_box(3, kind == 1, kind == 2);
    model.boundaries[1] = {0, field};
    model.subcases[0].boundary = 1;
    model.probes = {{1, Vector(.12, .31, -.17)}};
    auto solved = solve(model);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    const auto &result = std::get<Results>(solved);
    EXPECT_NEAR(result.cases[0].energy, 8 * field.squaredNorm() / (2 * mu0),
                1e-8);
    EXPECT_LT((result.cases[0].probe_b[0] - field).norm(), 1e-12);
    open_coil(model, 7);
    auto mesh = std::get<Mesh>(make_mesh(model));
    auto prepared = prepare_sources(model, mesh);
    ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
        << error_of(prepared);
    const auto &source = std::get<std::vector<Source>>(prepared)[0];
    for (const auto &c : source.mesh.cells)
      for (const auto &q : quadrature(source.mesh, c)) {
        auto basis = flux_basis(source.mesh, c, q.position);
        Vector j = Vector::Zero();
        for (size_t i = 0; i < c.faces.size(); ++i)
          j += basis[i] * source.flux[c.faces[i]];
        EXPECT_LT((j - Vector(7. / 4, 0, 0)).norm(), 1e-10);
      }
    EXPECT_LT(source.transfer_error, 1e-11);
  }
}

TEST(MagneticPlanar,
     UniformFieldDistortedMixedMeshRotationDepthAndMaxwellContour) {
  for (int kind = 0; kind < 3; ++kind) {
    auto model = planar_box(6, kind == 1, kind == 2);
    const Vector field(.3, -.2, 0);
    // Distort interior vertices while keeping the square boundary exact.
    for (auto &[id, g] : model.nodes) {
      (void)id;
      if (g.position.cwiseAbs().maxCoeff() < .99)
        g.position += Vector(.03 * std::sin(g.position.y() * 4),
                             .02 * std::cos(g.position.x() * 3), 0);
    }
    auto mesh = std::get<Mesh>(make_mesh(model));
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
      const auto &f = mesh.faces[i];
      if (f.cells[1] < 0)
        continue;
      bool inside[2]{};
      for (int k = 0; k < 2; ++k) {
        Vector center = Vector::Zero();
        for (int n : mesh.cells[f.cells[k]].nodes)
          center += mesh.nodes[n] /
                    static_cast<double>(mesh.cells[f.cells[k]].nodes.size());
        inside[k] = center.cwiseAbs().maxCoeff() < .55;
      }
      if (inside[0] != inside[1]) {
        const auto &c = mesh.cells[f.cells[inside[0] ? 0 : 1]];
        int local = static_cast<int>(
            std::find(c.faces.begin(), c.faces.end(), static_cast<int>(i)) -
            c.faces.begin());
        model.face_sets[1].push_back(
            face_selection(mesh, static_cast<int>(i), c.face_sign[local]));
      }
    }
    model.forces.push_back({1, 1});
    model.boundaries[1] = {0, field};
    model.subcases[0].boundary = 1;
    model.probes = {{1, Vector(.13, .24, 0)}};
    auto solved = solve(model);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    const auto &result = std::get<Results>(solved).cases[0];
    EXPECT_NEAR(result.energy, 4 * .3 * field.squaredNorm() / (2 * mu0), 1e-8);
    EXPECT_LT((result.probe_b[0] - field).norm(), 1e-12);
    EXPECT_LT(result.forces[0].force.norm(), 1e-8);
    EXPECT_LT(result.forces[0].torque.norm(), 1e-8);
    Matrix rotation =
        Eigen::AngleAxisd(.7, Vector(1, 2, 3).normalized()).toRotationMatrix();
    model.coordinates[5] = {0, 'R', Vector::Zero(), rotation * Vector::UnitZ(),
                            rotation * Vector::UnitX()};
    model.options.plane_coordinates = 5;
    for (auto &[id, g] : model.nodes) {
      (void)id;
      g.position = rotation * g.position;
    }
    model.boundaries[1].field = rotation * field;
    model.probes[0].position = rotation * model.probes[0].position;
    model.options.depth *= 2;
    auto rotated = solve(model);
    ASSERT_TRUE(std::holds_alternative<Results>(rotated)) << error_of(rotated);
    EXPECT_NEAR(std::get<Results>(rotated).cases[0].energy, 2 * result.energy,
                1e-8);
    EXPECT_LT(
        (std::get<Results>(rotated).cases[0].probe_b[0] - rotation * field)
            .norm(),
        1e-12);
  }
}

TEST(MagneticPlanar,
     ManufacturedAxialWindingConvergesAndObeysEnergyInductanceIdentities) {
  const double pi = std::numbers::pi, k2 = 1.25 * pi * pi;
  for (bool triangles : {false, true}) {
    double previous = 1;
    for (int n : {16, 32, 64}) {
      auto m = planar_box(n, triangles);
      for (const auto &c : m.elements) {
        Vector center = Vector::Zero();
        for (int v : c.nodes)
          center +=
              m.nodes.at(v).position / static_cast<double>(c.nodes.size());
        const double area = (triangles ? 2. : 4.) / (n * n);
        const double turns = k2 / mu0 * std::sin(pi * center.x()) *
                             std::cos(pi * center.y() / 2) * area;
        m.element_sets[c.id] = {c.id};
        m.planar_coils.push_back({c.id, c.id, turns});
        m.circuits[1000000].members.emplace_back(c.id, 1);
      }
      m.circuits[1000000].id = 1000000;
      m.currents = {{1, 1000000, 1}, {2, 1000000, -2}};
      m.subcases = {{.id = 1, .load = 1}, {.id = 2, .load = 2}};
      auto solved = solve(m);
      ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
      const auto &r = std::get<Results>(solved);
      const auto &c = r.cases[0];
      ASSERT_EQ(r.circuits.size(), 1U);
      ASSERT_TRUE(r.circuits[0].closed);
      EXPECT_NEAR(c.circuits[0].linkage, 2 * c.energy, 1e-7);
      EXPECT_NEAR(c.circuit_inductance(0, 0), c.circuits[0].linkage, 1e-7);
      EXPECT_NEAR(r.cases[1].energy, 4 * c.energy, 1e-7);
      double error = 0;
      for (const auto &cell : r.mesh.cells)
        for (const auto &q : quadrature(r.mesh, cell)) {
          Vector field = Vector::Zero();
          const auto curl = edge_curls(cell, q.position);
          for (size_t i = 0; i < cell.nodes.size(); ++i)
            field += c.potential[cell.nodes[i]] * curl[i];
          const Vector exact(-pi / 2 * std::sin(pi * q.position.x()) *
                                 std::sin(pi * q.position.y() / 2),
                             -pi * std::cos(pi * q.position.x()) *
                                 std::cos(pi * q.position.y() / 2),
                             0);
          error += q.weight * (field - exact).squaredNorm();
        }
      error = std::sqrt(error / k2);
      EXPECT_LT(error, previous * .65);
      previous = error;
      if (n == 64) {
        EXPECT_LT(error, .04);
        EXPECT_NEAR(c.energy, .3 * k2 / (2 * mu0), .008 * .3 * k2 / (2 * mu0));
      }
    }
  }
}

TEST(MagneticPlanar, PermanentMagnetReversal) {
  auto m = planar_box(12);
  // Axis 1 of a frame rotated about the plane normal: B_r = (.2, .1, 0).
  m.coordinates[1] = {0, 'R', Vector::Zero(), Vector::UnitZ(), Vector(2, 1, 0)};
  m.properties[2] = {2, 1, 1, true};
  for (auto &cell : m.elements) {
    Vector center = Vector::Zero();
    for (int n : cell.nodes)
      center += m.nodes.at(n).position / static_cast<double>(cell.nodes.size());
    if (std::abs(center.x()) < .32 && std::abs(center.y()) < .65)
      cell.property = 2;
  }
  m.materials[2] = {1.05, 0, std::hypot(.2, .1)};
  auto positive = solve(m);
  ASSERT_TRUE(std::holds_alternative<Results>(positive)) << error_of(positive);
  m.properties[2].axis = -1;
  auto negative = solve(m);
  ASSERT_TRUE(std::holds_alternative<Results>(negative)) << error_of(negative);
  const auto &a = std::get<Results>(positive).cases[0],
             &b = std::get<Results>(negative).cases[0];
  EXPECT_NEAR(a.energy, b.energy, 1e-10);
  for (size_t i = 0; i < a.b.size(); ++i)
    EXPECT_LT((a.b[i] + b.b[i]).norm(), 1e-12);
  m.properties[2].axis = 3; // the plane normal
  EXPECT_NE(error_of(solve(m)).find("remanence must lie"), std::string::npos);
}

// A ring magnetized along theta has H = 0 everywhere by Ampere's law on
// concentric circles, so B = B_r theta inside it and zero elsewhere for any
// recoil permeability. Radial magnetization of the same ring has no
// equivalent current, so B = 0 and H = -B_r r / (mu0 mu_r) inside. Both
// evaluate the cylindrical frame at every quadrature point.
TEST(MagneticPlanar, CylindricalRingMagnetizationMatchesAmpereSolutions) {
  const double br = 1.2, mur = 1.05, inner = .3, outer = .6;
  for (int axis : {2, 1}) {
    SCOPED_TRACE(axis);
    double previous = 1;
    for (int n : {32, 64}) {
      auto m = planar_box(n, true);
      m.coordinates[1] = {0, 'C', Vector::Zero(), Vector::UnitZ(),
                          Vector::UnitX()};
      m.materials[2] = {mur, 0, br};
      m.properties[2] = {2, 1, axis, true};
      for (auto &cell : m.elements) {
        Vector center = Vector::Zero();
        for (int v : cell.nodes)
          center += m.nodes.at(v).position / 3;
        if (center.norm() > inner && center.norm() < outer)
          cell.property = 2;
      }
      const double angle = .4;
      const Vector radial(std::cos(angle), std::sin(angle), 0),
          tangential(-std::sin(angle), std::cos(angle), 0);
      m.probes = {{1, .45 * radial}, {2, .1 * radial}, {3, .8 * radial}};
      auto solved = solve(m);
      ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
      const auto &c = std::get<Results>(solved).cases[0];
      const Vector expected_b =
          axis == 2 ? Vector(br * tangential) : Vector(Vector::Zero());
      const Vector expected_h =
          axis == 2 ? Vector(Vector::Zero()) : Vector(-br / (mu0 * mur) * radial);
      // The polygonal ring boundary is the only error source; it decays
      // under refinement.
      const double error =
          std::max({(c.probe_b[0] - expected_b).norm(), c.probe_b[1].norm(),
                    c.probe_b[2].norm(),
                    mu0 * (c.probe_h[0] - expected_h).norm()}) /
          br;
      EXPECT_LT(error, previous);
      previous = error;
      if (n == 64)
        EXPECT_LT(error, .03);
    }
  }
}

TEST(MagneticParser, NativeSolidsPlanarWindingCardsAndUnsupportedGeometry) {
  const std::string grids =
      "GRID,1,,0,0,0\nGRID,2,,1,0,0\nGRID,3,,1,1,0\nGRID,4,,0,1,0\n"
      "GRID,5,,0,0,1\nGRID,6,,1,0,1\nGRID,7,,1,1,1\nGRID,8,,0,1,1\n";
  for (const std::string &cell :
       {"CHEXA,1,1,1,2,3,4,5,6,7,8\n", "CPENTA,1,1,1,2,3,5,6,7\n"}) {
    auto parsed = parse_string("SOL MAGSTAT\nCEND\nBEGIN BULK\n" + grids +
                               cell + "PSOLID,1,1\nMATMAG,1,1\n");
    ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
    auto solved = solve(std::get<Model>(parsed));
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  }
  std::string deck =
      "SOL MAGSTAT\nCEND\nLOAD=1\nBEGIN BULK\nMAG2D,0,20\nPARAM,MAGLEN,.001\n" +
      grids +
      "CTRIA3,1,1,1,2,3\nCTRIA3,2,1,1,3,4\nPMAG2D,1,1\nMATMAG,1,1\n"
      "ESET,1,1\nESET,2,2\nCOIL2D,10,1,20\nCOIL2D,11,2,-20\n"
      "CIRCUIT,12,,10,1,11,1\nICIRC,1,12,3\nFSET,1,1,1,2\n";
  auto parsed = parse_string(deck);
  ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
  const auto &m = std::get<Model>(parsed);
  EXPECT_DOUBLE_EQ(m.options.depth, .02);
  auto solved = solve(m);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &r = std::get<Results>(solved);
  ASSERT_EQ(r.sources.size(), 2U);
  ASSERT_EQ(r.sources[0].planar_density.size(), 1U);
  EXPECT_NEAR(r.sources[0].planar_density[0].z(), 4e7, 1e-7);
  EXPECT_NEAR(r.sources[1].planar_density[0].z(), -4e7, 1e-7);
  // Each leg alone is unbalanced; the circuit is balanced.
  EXPECT_FALSE(r.sources[0].closed);
  ASSERT_EQ(r.circuits.size(), 1U);
  EXPECT_TRUE(r.circuits[0].closed);
  EXPECT_NEAR((r.sources[0].coupling + r.sources[1].coupling).sum(), 0, 1e-12);
  EXPECT_DOUBLE_EQ(r.cases[0].sources[1].current, 3);
  EXPECT_TRUE(std::isfinite(r.cases[0].circuits[0].linkage));
  EXPECT_FALSE(std::isfinite(r.cases[0].sources[0].linkage));
  auto repeated = deck;
  repeated.replace(repeated.find("COIL2D,11"), 9, "COIL2D,10");
  EXPECT_NE(error_of(parse_string(repeated)).find("group legs with CIRCUIT"),
            std::string::npos);
  auto bad = m;
  bad.options.planar = false;
  EXPECT_NE(error_of(solve(bad)).find("COIL2D requires"), std::string::npos);
  bad = m;
  bad.nodes[3].position.z() = .001;
  EXPECT_NE(error_of(solve(bad)).find("common plane"), std::string::npos);
  auto solid = brick_box(1);
  solid.nodes[1].position.z() += .1;
  EXPECT_TRUE(std::holds_alternative<Mesh>(make_mesh(solid)));
  EXPECT_NE(error_of(parse_string(deck + "CWIRE,99,1,1,2\n")).find("CWIRE"),
            std::string::npos);
}

TEST(MagneticSolids, DrivenWireFieldConvergesOnNativeHexAndWedgeMeshes) {
  const double expected = mu0 * std::sqrt(2.) / (std::numbers::pi * .6);
  for (bool wedge : {false, true}) {
    double previous = 1;
    for (int n : {8, 16}) {
      SCOPED_TRACE(std::format("wedge={}, n={}", wedge, n));
      auto m = brick_box(n, wedge);
      for (auto &[id, g] : m.nodes) {
        (void)id;
        g.position *= 1.5;
      }
      loop(m, .6, true);
      auto solved = solve(m);
      ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
      const auto &r = std::get<Results>(solved);
      const auto &c = r.cases[0];
      const double error = std::abs(c.probe_b[0].z() / expected - 1);
      EXPECT_LT(error, previous);
      previous = error;
      EXPECT_NEAR(c.sources[0].linkage, 2 * c.energy, 1e-15);
      EXPECT_LT(r.sources[0].transfer_error, 1e-9);
      for (const auto &cell : r.sources[0].mesh.cells)
        EXPECT_EQ(cell.nodes.size(), 8U);
      if (n == 16)
        EXPECT_LT(error, .12);
    }
  }
}

TEST(MagneticPlanar, WindingTorqueApproachesAxialCurrentMomentInUniformField) {
  double previous = 1;
  for (int n : {16, 32}) {
    auto m = planar_box(n);
    for (const auto &c : m.elements) {
      Vector center = Vector::Zero();
      for (int v : c.nodes)
        center += m.nodes.at(v).position / 4;
      if (std::abs(center.y()) < .125 && std::abs(center.x()) > .375 &&
          std::abs(center.x()) < .625)
        m.element_sets[center.x() < 0 ? 1 : 2].push_back(c.id);
    }
    m.planar_coils = {{1, 1, 20}, {2, 2, -20}};
    m.circuits[3] = {3, 1, {{1, 1}, {2, 1}}};
    m.currents = {{1, 3, 1}};
    m.boundaries[1] = {0, Vector(.01, 0, 0)};
    m.subcases = {{.id = 1, .load = 1, .boundary = 1}};
    const auto mesh = std::get<Mesh>(make_mesh(m));
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
      const auto &f = mesh.faces[i];
      if (f.cells[1] < 0)
        continue;
      bool inside[2]{};
      for (int k = 0; k < 2; ++k) {
        Vector center = Vector::Zero();
        for (int v : mesh.cells[f.cells[k]].nodes)
          center += mesh.nodes[v] / 4;
        inside[k] = center.cwiseAbs().maxCoeff() < .75;
      }
      if (inside[0] != inside[1]) {
        const auto &c = mesh.cells[f.cells[inside[0] ? 0 : 1]];
        int local = static_cast<int>(
            std::find(c.faces.begin(), c.faces.end(), static_cast<int>(i)) -
            c.faces.begin());
        m.face_sets[1].push_back(
            face_selection(mesh, static_cast<int>(i), c.face_sign[local]));
      }
    }
    m.forces = {{1, 1}};
    auto solved = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    const auto &f = std::get<Results>(solved).cases[0].forces[0];
    const double expected = -20 * .01 * .3; // 2*a*N*I*B*depth with a=0.5 m
    const double error = std::abs(f.torque.z() / expected - 1);
    EXPECT_LT(error, previous);
    previous = error;
    EXPECT_LT(f.force.norm(), 1e-9);
    if (n == 32)
      EXPECT_LT(error, .03);
  }
}

// The face set of the axis-aligned cube or square of half-width w, oriented
// outward. Faces of the cube surface have centroids with max |x_i| = w.
std::vector<FaceSelection> centered_surface(const Mesh &mesh, double w) {
  std::vector<FaceSelection> out;
  for (size_t i = 0; i < mesh.faces.size(); ++i) {
    const auto &f = mesh.faces[i];
    const Vector c = mesh.planar ? Vector(f.center.x(), f.center.y(), 0)
                                 : f.center;
    if (f.cells[1] >= 0 && std::abs(c.cwiseAbs().maxCoeff() - w) < 1e-12) {
      // The outward side is the cell whose centroid lies outside the cube.
      Vector center = Vector::Zero();
      for (int v : mesh.cells[f.cells[0]].nodes)
        center += mesh.nodes[v] /
                  static_cast<double>(mesh.cells[f.cells[0]].nodes.size());
      const int local = static_cast<int>(
          std::find(mesh.cells[f.cells[0]].faces.begin(),
                    mesh.cells[f.cells[0]].faces.end(), static_cast<int>(i)) -
          mesh.cells[f.cells[0]].faces.begin());
      const int outward =
          mesh.cells[f.cells[0]].face_sign[local] *
          (center.head<3>().cwiseAbs().maxCoeff() < w ? 1 : -1);
      out.push_back(face_selection(mesh, static_cast<int>(i), outward));
    }
  }
  return out;
}

TEST(MagneticForce, SurfacesThroughCurrentCarryingVolumeAreRejected) {
  // CWIRE: the conductor of radius 0.6 lies on the cube plane x = 0.6, so that
  // surface cuts through it. The cube at 0.9 encloses the loop entirely; cells
  // inside it still contain the conductor, which is allowed.
  auto wire = box(8, 1.2);
  loop(wire);
  wire.boundaries[1] = {0, Vector(0, 0.001, 0)};
  wire.subcases[0].boundary = 1;
  const auto wire_mesh = std::get<Mesh>(make_mesh(wire));
  wire.forces = {{1, 1}};
  wire.face_sets[1] = centered_surface(wire_mesh, .6);
  EXPECT_NE(error_of(solve(wire)).find("crosses CWIRE path 100000"),
            std::string::npos)
      << error_of(solve(wire));
  wire.face_sets[1] = centered_surface(wire_mesh, .9);
  auto enclosed = solve(wire);
  ASSERT_TRUE(std::holds_alternative<Results>(enclosed)) << error_of(enclosed);
  const Vector moment(-4 * .6 * .6 * .001, 0, 0);
  EXPECT_LT((std::get<Results>(enclosed).cases[0].forces[0].torque - moment)
                    .norm() /
                moment.norm(),
            .2);

  // Volumetric COIL filling the domain: an interior surface has winding on
  // both sides of every face.
  auto coil = box(3);
  open_coil(coil);
  const auto coil_mesh = std::get<Mesh>(make_mesh(coil));
  coil.face_sets[3] = centered_surface(coil_mesh, 1. / 3);
  coil.forces = {{1, 3}};
  EXPECT_NE(error_of(solve(coil)).find("lies inside current source 1"),
            std::string::npos)
      << error_of(solve(coil));

  // COIL2D: winding slots span 0.375 < |x| < 0.625, so the square contour at
  // 0.5 runs through them, while the one at 0.75 encloses both.
  auto planar = planar_box(16);
  for (const auto &c : planar.elements) {
    Vector center = Vector::Zero();
    for (int v : c.nodes)
      center += planar.nodes.at(v).position / 4;
    if (std::abs(center.y()) < .125 && std::abs(center.x()) > .375 &&
        std::abs(center.x()) < .625)
      planar.element_sets[center.x() < 0 ? 1 : 2].push_back(c.id);
  }
  planar.planar_coils = {{1, 1, 20}, {2, 2, -20}};
  planar.circuits[3] = {3, 1, {{1, 1}, {2, 1}}};
  planar.currents = {{1, 3, 1}};
  planar.subcases[0].load = 1;
  const auto planar_mesh = std::get<Mesh>(make_mesh(planar));
  planar.forces = {{1, 1}};
  planar.face_sets[1] = centered_surface(planar_mesh, .5);
  EXPECT_NE(error_of(solve(planar)).find("lies inside current source 1"),
            std::string::npos)
      << error_of(solve(planar));
  planar.face_sets[1] = centered_surface(planar_mesh, .75);
  EXPECT_TRUE(std::holds_alternative<Results>(solve(planar)));
}

TEST(MagneticWire, SpatialLoopFallbackPreservesTheCurrentMoment) {
  auto m = box(1);
  loop(m);
  m.nodes[100002].position.z() = .3;
  m.nodes[100002].position.x() = .4;
  auto prepared = prepare_wires(m);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
      << error_of(prepared);
  const auto &s = std::get<std::vector<Source>>(prepared)[0];
  for (const auto &c : s.mesh.cells)
    EXPECT_EQ(c.nodes.size(), 4U);
  Vector expected = Vector::Zero(), moment = Vector::Zero();
  for (int i = 0; i < 4; ++i)
    expected +=
        .5 * m.nodes.at(100000 + i)
                 .position.cross(m.nodes.at(100000 + (i + 1) % 4).position);
  for (const auto &c : s.mesh.cells)
    for (const auto &q : quadrature(s.mesh, c)) {
      const auto basis = flux_basis(s.mesh, c, q.position);
      Vector j = Vector::Zero();
      for (size_t i = 0; i < c.faces.size(); ++i)
        j += basis[i] * s.flux[c.faces[i]];
      moment += .5 * q.weight * q.position.cross(j);
    }
  EXPECT_LT((moment - expected).norm() / expected.norm(), .005);
  m.options.wire_mesh = WireMesh::Hex;
  const auto rejected = prepare_wires(m);
  ASSERT_TRUE(std::holds_alternative<Error>(rejected));
  EXPECT_NE(error_of(rejected).find("WIREMESH=HEX"), std::string::npos);
  m.options.wire_mesh = WireMesh::Tet;
  const auto selected = prepare_wires(m);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(selected))
      << error_of(selected);
  const auto &t = std::get<std::vector<Source>>(selected)[0];
  EXPECT_EQ(t.mesh.cells.size(), s.mesh.cells.size());
  EXPECT_LT((t.flux - s.flux).norm(), 1e-12);
}

TEST(MagneticMappedBasis,
     MixedInterfacesPreserveTangentialPotentialAndNormalFlux) {
  std::vector<Mesh> meshes;
  auto mixed = brick_box(2, false, true);
  for (auto &[id, g] : mixed.nodes) {
    (void)id;
    g.position.x() += .3 * g.position.y() * g.position.z();
  }
  meshes.push_back(std::get<Mesh>(make_mesh(mixed)));
  auto built = make_mesh({Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0),
                          Vector(0, 0, 1), Vector(1, 0, 1), Vector(0, 1, 1),
                          Vector(.2, .2, 2)},
                         {{0, 1, 2, 3, 4, 5}, {3, 4, 5, 6}});
  ASSERT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
  meshes.push_back(std::get<Mesh>(std::move(built)));
  for (const auto &mesh : meshes) {
    Eigen::VectorXd flux(mesh.faces.size());
    for (Eigen::Index i = 0; i < flux.size(); ++i)
      flux[i] = std::cos(i * .3);
    for (const auto &face : mesh.faces) {
      if (face.cells[1] < 0)
        continue;
      const auto &a = mesh.cells[face.cells[0]], &b = mesh.cells[face.cells[1]];
      if (a.nodes.size() == b.nodes.size())
        continue;
      for (const auto &q : face_quadrature(mesh, face)) {
        Vector potential[2] = {Vector::Zero(), Vector::Zero()};
        Vector field[2] = {Vector::Zero(), Vector::Zero()};
        Vector current[2] = {Vector::Zero(), Vector::Zero()};
        for (int side = 0; side < 2; ++side) {
          const auto &c = mesh.cells[face.cells[side]];
          const auto basis = edge_basis(c, q.position),
                     curls = edge_curls(c, q.position);
          for (size_t e = 0; e < c.edges.size(); ++e) {
            const double value = std::sin(.7 * c.edges[e]);
            potential[side] += value * basis[e];
            field[side] += value * curls[e];
          }
          const auto fb = flux_basis(mesh, c, q.position);
          for (size_t i = 0; i < c.faces.size(); ++i)
            current[side] += flux[c.faces[i]] * fb[i];
          EXPECT_LT((flux_value(c, flux, q.position) - current[side]).norm(),
                    1e-12);
        }
        EXPECT_LT((potential[0] - potential[1]).cross(q.normal).norm(), 1e-12);
        EXPECT_NEAR((field[0] - field[1]).dot(q.normal), 0, 1e-12);
        EXPECT_NEAR((current[0] - current[1]).dot(q.normal), 0, 1e-12);
      }
    }
  }
}

TEST(MagneticPlanar, LengthScalingPreservesDistortedQuadField) {
  for (double scale : {1., 1e-6, 1e-12}) {
    auto model = planar_box(4);
    for (auto &[id, g] : model.nodes) {
      (void)id;
      if (g.position.cwiseAbs().maxCoeff() < .99)
        g.position.x() += .1 * g.position.y();
      g.position *= scale;
    }
    const Vector field(.2, .3, 0);
    model.boundaries[1] = {0, field};
    model.subcases[0].boundary = 1;
    model.probes = {{1, scale * Vector(.1, .2, 0)}};
    auto solved = solve(model);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    const auto &c = std::get<Results>(solved).cases[0];
    const double energy =
        4 * .3 * field.squaredNorm() * scale * scale / (2 * mu0);
    EXPECT_NEAR(c.energy, energy, 1e-11 * energy);
    EXPECT_LT((c.probe_b[0] - field).norm(), 1e-12);
  }
}

TEST(MagneticWarped, BilinearFacesNormalsFluxesAndTwoLineIntersections) {
  auto m = brick_box(1);
  for (auto &[id, g] : m.nodes) {
    (void)id;
    g.position.z() += g.position.x() * g.position.y();
  }
  auto built = make_mesh(m);
  ASSERT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
  const auto &mesh = std::get<Mesh>(built);
  const auto &cell = mesh.cells[0];
  EXPECT_FALSE(cell.polyhedral);
  double volume = 0;
  Vector area_vector = Vector::Zero();
  for (size_t i = 0; i < cell.faces.size(); ++i) {
    const auto &face = mesh.faces[cell.faces[i]];
    std::vector<double> flux(cell.faces.size(), 0);
    for (const auto &q : face_quadrature(mesh, face)) {
      EXPECT_TRUE(point_on_face(mesh, face, q.position));
      const auto basis = flux_basis(mesh, cell, q.position);
      for (size_t j = 0; j < flux.size(); ++j)
        flux[j] += q.weight * q.normal.dot(basis[j]);
      volume += cell.face_sign[i] * q.weight * q.normal.dot(q.position) / 3;
      area_vector += cell.face_sign[i] * q.weight * q.normal;
      if (!face.flat) {
        const Vector expected =
            Vector(-q.position.y(), -q.position.x(), 1).normalized();
        EXPECT_LT((q.normal - expected).norm(), 1e-12);
      }
    }
    for (size_t j = 0; j < flux.size(); ++j)
      EXPECT_NEAR(flux[j], i == j ? 1 : 0, 1e-12);
    if (face.center.z() > .9) {
      auto roots = line_face_intersections(mesh, face, Vector(-.8, -.8, 1.16),
                                           Vector(.8, .8, 1.16));
      std::sort(roots.begin(), roots.end());
      ASSERT_EQ(roots.size(), 2U);
      EXPECT_NEAR(roots[0], .25, 1e-12);
      EXPECT_NEAR(roots[1], .75, 1e-12);
      auto long_ray = line_face_intersections(
          mesh, face, Vector(-1e8, -1e8, 1.16), Vector(1e8, 1e8, 1.16));
      std::sort(long_ray.begin(), long_ray.end());
      ASSERT_EQ(long_ray.size(), 2U);
      EXPECT_NEAR(long_ray[0], .5 - 2e-9, 1e-15);
      EXPECT_NEAR(long_ray[1], .5 + 2e-9, 1e-15);
      auto rule = line_face_intersections(mesh, face, Vector(0, -2, 1),
                                          Vector(0, 2, 1));
      std::sort(rule.begin(), rule.end());
      ASSERT_EQ(rule.size(), 2U);
      EXPECT_NEAR(rule[0], .25, 1e-12);
      EXPECT_NEAR(rule[1], .75, 1e-12);
    }
  }
  EXPECT_NEAR(volume, 8, 1e-12);
  EXPECT_LT(area_vector.norm(), 1e-12);
}

TEST(MagneticWarped, HexAndWedgeFieldsEnergyAndCurvedForceSurface) {
  for (bool wedge : {false, true}) {
    auto m = brick_box(3, wedge);
    const auto initial = std::get<Mesh>(make_mesh(m));
    for (size_t i = 0; i < initial.faces.size(); ++i) {
      const auto &f = initial.faces[i];
      bool on = false;
      for (int k = 0; k < 3; ++k)
        on |= std::abs(std::abs(f.center[k]) - 1. / 3) < 1e-12;
      if (on && f.center.cwiseAbs().maxCoeff() < 1. / 3 + 1e-12)
        m.face_sets[1].push_back(face_selection(
            initial, static_cast<int>(i), f.normal.dot(f.center) > 0 ? 1 : -1));
    }
    for (auto &[id, g] : m.nodes) {
      (void)id;
      if (wedge)
        g.position.x() += .4 * g.position.y() * g.position.z();
      else
        g.position.z() += .4 * g.position.x() * g.position.y();
    }
    const Vector field = .2 * (wedge ? Vector::UnitX() : Vector::UnitZ());
    m.boundaries[1] = {0, field};
    m.subcases[0].boundary = 1;
    m.forces = {{1, 1}};
    m.probes = {{1, Vector(.12, .11, .17)}};
    auto solved = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    const auto &r = std::get<Results>(solved);
    EXPECT_TRUE(std::any_of(r.mesh.faces.begin(), r.mesh.faces.end(),
                            [](const Face &f) { return !f.flat; }));
    EXPECT_NEAR(r.cases[0].energy, 8 * field.squaredNorm() / (2 * mu0), 1e-8);
    EXPECT_LT((r.cases[0].probe_b[0] - field).norm(), 1e-12);
    EXPECT_LT(r.cases[0].forces[0].force.norm(), 1e-8);
    EXPECT_LT(r.cases[0].forces[0].torque.norm(), 1e-8);
  }
}

TEST(MagneticWarped, MatchedClosedCoilAndWarpedCutConserveAmpereTurns) {
  auto m = brick_box(6);
  const auto initial = std::get<Mesh>(make_mesh(m));
  for (const auto &c : initial.cells) {
    const Vector center = (c.minimum + c.maximum) / 2;
    const double radius = center.head<2>().cwiseAbs().maxCoeff();
    if (radius > .3 && radius < .7 && std::abs(center.z()) < .4)
      m.element_sets[1].push_back(c.id);
  }
  for (size_t i = 0; i < initial.faces.size(); ++i) {
    const auto &f = initial.faces[i];
    if (std::abs(f.center.y()) < 1e-12 && f.center.x() > .3 &&
        f.center.x() < .7 && std::abs(f.center.z()) < .4)
      m.face_sets[1].push_back(face_selection(initial, static_cast<int>(i),
                                              f.normal.y() > 0 ? 1 : -1));
  }
  m.coils = {{1, 1, 20, 1, 0}};
  for (auto &[id, g] : m.nodes) {
    (void)id;
    g.position.y() += .4 * g.position.x() * g.position.z();
  }
  auto built = make_mesh(m);
  ASSERT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
  const auto &mesh = std::get<Mesh>(built);
  auto prepared = prepare_sources(m, mesh);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
      << error_of(prepared);
  const auto &s = std::get<std::vector<Source>>(prepared)[0];
  EXPECT_LT(s.conservation_error, 1e-10);
  EXPECT_LT(s.transfer_error, 1e-10);
  auto selection = selected_faces(m, s.mesh, 1);
  ASSERT_TRUE(
      (std::holds_alternative<std::vector<std::pair<int, int>>>(selection)))
      << error_of(selection);
  double current = 0;
  for (auto [f, sign] : std::get<std::vector<std::pair<int, int>>>(selection)) {
    EXPECT_FALSE(s.mesh.faces[f].flat);
    current += sign * s.flux[f];
  }
  EXPECT_NEAR(current, 20, 1e-10);
  for (auto &f : m.face_sets[1])
    std::reverse(f.nodes.begin(), f.nodes.end());
  auto reversed = prepare_sources(m, mesh);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(reversed))
      << error_of(reversed);
  EXPECT_LT(
      (std::get<std::vector<Source>>(reversed)[0].coupling + s.coupling).norm(),
      1e-10);
}

TEST(MagneticWarped, IndependentWireTransferConservesCurrentAndMagneticMoment) {
  for (auto kind : {WireMesh::Hex, WireMesh::Tet}) {
    SCOPED_TRACE(static_cast<int>(kind));
    for (bool wedge : {false, true}) {
      SCOPED_TRACE(wedge);
      auto m = brick_box(2, wedge);
      m.options.wire_mesh = kind;
      for (auto &[id, g] : m.nodes) {
        (void)id;
        if (wedge)
          g.position.x() += .35 * g.position.y() * g.position.z();
        else
          g.position.z() += .35 * g.position.x() * g.position.y();
      }
      loop(m, .6, true);
      if (wedge) {
        for (int i = 0; i < 4; ++i) {
          auto &x = m.nodes[100000 + i].position;
          x = Vector(x.z(), x.x(), x.y());
          m.wires[i].orientation = Vector::UnitX();
        }
      }
      auto solved = solve(m);
      ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
      const auto &result = std::get<Results>(solved);
      const auto &mesh = result.mesh;
      const auto &s = result.sources[0];
      const auto &c = result.cases[0];
      ASSERT_GT(c.energy, 0);
      EXPECT_NEAR(c.sources[0].linkage, 2 * c.energy, 1e-10 * c.energy);
      EXPECT_NEAR(c.differential_inductance(0, 0), 2 * c.energy,
                  1e-10 * c.energy);
      EXPECT_LT(s.transfer_error, 1e-9);
      for (const auto &c : s.mesh.cells)
        EXPECT_TRUE(c.polyhedral);
      Vector moment = Vector::Zero();
      for (const auto &c : s.mesh.cells)
        for (const auto &q : quadrature(s.mesh, c))
          moment += .5 * q.weight *
                    q.position.cross(flux_value(c, s.flux, q.position));
      const Vector direction = wedge ? Vector::UnitX() : Vector::UnitZ();
      Eigen::VectorXd a(mesh.edges.size());
      for (size_t e = 0; e < mesh.edges.size(); ++e) {
        const Vector x = mesh.nodes[mesh.edges[e][0]],
                     y = mesh.nodes[mesh.edges[e][1]];
        a[e] = .5 * direction.cross((x + y) / 2).dot(y - x);
      }
      EXPECT_NEAR(s.coupling.dot(a), moment.dot(direction),
                  1e-8 * moment.norm());
    }
  }
}

TEST(MagneticWarped, RejectsFoldedHexesAndWedgesMissedByVolumeQuadrature) {
  for (int count : {6, 8})
    for (bool interior : {false, true}) {
      SCOPED_TRACE(std::format("{} nodes, interior fold {}", count, interior));
      Cell c;
      c.nodes.resize(count);
      for (int i = 0; i < count; ++i)
        c.nodes[i] = i;
      const auto reference = reference_vertices(c);
      for (const auto &r : reference) {
        if (interior)
          c.points.emplace_back((r.z() - .4) * r.x() + .05 * r.y(),
                                .05 * r.x() + (r.z() - .4) * r.y(), r.z());
        else
          c.points.emplace_back(r.x() + 1.1 * r.y() * r.z(),
                                r.y() + 1.1 * r.x() * r.z(), r.z());
      }
      c.origin = c.points[0];
      initialize_mapping(c);
      // det(J) is (t-.4)^2-.05^2 or 1-1.21*t^2. Both are
      // positive at every existing volume Gauss point, yet folded elsewhere.
      for (const auto &q : reference_quadrature(c))
        EXPECT_GT(mapping(c, q.position).second.determinant(), 0);
      if (interior) {
        for (const auto &r : reference)
          EXPECT_GT(mapping(c, r).second.determinant(), 0);
        EXPECT_LT(mapping(c, Vector(0, 0, .4)).second.determinant(), 0);
      }
      auto rejected = make_mesh(c.points, {c.nodes});
      ASSERT_TRUE(std::holds_alternative<Error>(rejected));
      EXPECT_NE(error_of(rejected).find("Jacobian"), std::string::npos);
    }
}

TEST(MagneticWarped, TinyCurvedPenetrationIsDetectedAndTouchingIsAllowed) {
  const auto source = std::get<Mesh>(make_mesh(brick_box(1)));
  for (double penetration : {1e-3, 0., -1e-3}) {
    auto target = brick_box(1);
    for (auto &[id, g] : target.nodes) {
      (void)id;
      g.position.x() +=
          2.4 - penetration + .4 * g.position.y() * g.position.z();
    }
    auto built = make_mesh(target);
    ASSERT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
    const auto &mesh = std::get<Mesh>(built);
    const auto &cell = mesh.cells[0];
    // Penetration occurs only near two warped-face corners, outside all
    // ordinary volume and surface quadrature samples.
    for (const auto &q : quadrature(mesh, cell))
      EXPECT_FALSE(contains(source.cells[0], q.position));
    auto overlap = cell_intersects_interior(source, source.cells[0], cell);
    ASSERT_TRUE(std::holds_alternative<bool>(overlap)) << error_of(overlap);
    EXPECT_EQ(std::get<bool>(overlap), penetration > 0);
    bool crossing = false;
    for (int f : cell.faces) {
      auto intersects = face_intersects_interior(source, source.cells[0], mesh,
                                                 mesh.faces[f]);
      ASSERT_TRUE(std::holds_alternative<bool>(intersects))
          << error_of(intersects);
      crossing |= std::get<bool>(intersects);
    }
    EXPECT_EQ(crossing, penetration > 0);
  }
}

TEST(MagneticWarped, RayTransferMatchesPolyhedralIntersectionIntegration) {
  for (bool round : {false, true})
    for (auto kind : {WireMesh::Hex, WireMesh::Tet}) {
      auto model = round ? box(6, 1.5) : brick_box(2, false, true);
      loop(model, .6, !round);
      model.options.wire_mesh = kind;
      const auto mesh = std::get<Mesh>(make_mesh(model));
      const auto source =
          std::get<std::vector<Source>>(prepare_wires(model))[0];
      const SpatialIndex index(mesh);
      const auto clipped = couple_polyhedral_source(source, mesh, index),
                 rays = couple_curved_source(source, mesh, index);
      ASSERT_TRUE(std::holds_alternative<Eigen::VectorXd>(clipped))
          << error_of(clipped);
      ASSERT_TRUE(std::holds_alternative<Eigen::VectorXd>(rays))
          << error_of(rays);
      const auto &expected = std::get<Eigen::VectorXd>(clipped);
      EXPECT_LT((std::get<Eigen::VectorXd>(rays) - expected).norm() /
                    expected.norm(),
                1e-9);
    }
}

TEST(MagneticWarped, OpenWireTerminalsAndMaterialExclusion) {
  for (auto kind : {WireMesh::Hex, WireMesh::Tet}) {
    SCOPED_TRACE(static_cast<int>(kind));
    auto model = brick_box(2);
    model.options.wire_mesh = kind;
    for (auto &[id, g] : model.nodes) {
      (void)id;
      g.position.z() += .3 * g.position.x() * g.position.y();
    }
    model.conductivity[1] = 5.8e7;
    model.wire_properties[1] = {1, true, .1, .1};
    model.nodes[100000] = {Vector(-1, .07, .13), 0};
    model.nodes[100001] = {Vector(1, .07, .13), 0};
    model.wires.push_back({100000, 1, 100000, 100001, 0, Vector::UnitZ()});
    model.wire_paths.push_back({100000, 100000});
    model.options.wire_layers = 1;
    model.options.wire_step = .5;
    const auto mesh = std::get<Mesh>(make_mesh(model));
    auto prepared = prepare_sources(model, mesh);
    ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
        << error_of(prepared);
    EXPECT_LT(std::get<std::vector<Source>>(prepared)[0].transfer_error, 1e-9);
    auto invalid = model;
    invalid.nodes[100000].position.x() = -.8;
    auto rejected = prepare_sources(invalid, mesh);
    ASSERT_TRUE(std::holds_alternative<Error>(rejected));
    EXPECT_NE(error_of(rejected).find("terminate"), std::string::npos);
    invalid = model;
    invalid.nodes[100000].position.x() = -1.01;
    rejected = prepare_sources(invalid, mesh);
    ASSERT_TRUE(std::holds_alternative<Error>(rejected));
    EXPECT_NE(error_of(rejected).find("outside"), std::string::npos);
    invalid = model;
    invalid.materials[1].mur = 2;
    rejected = prepare_sources(invalid, mesh);
    ASSERT_TRUE(std::holds_alternative<Error>(rejected));
    EXPECT_NE(error_of(rejected).find("overlaps"), std::string::npos)
        << error_of(rejected);
  }
}

TEST(MagneticParser,
     WireMeshSelectsActualSourceTopologyAndRejectsInvalidValue) {
  const std::string prefix = "SOL MAGSTAT\nCEND\nBEGIN BULK\n";
  const std::string body = R"(GRID,1,,0,0,0
GRID,2,,1,0,0
GRID,3,,0,1,0
GRID,4,,0,0,1
MATMAG,1,1
PSOLID,1,1
CTETRA,1,1,1,2,3,4
MATCOND,2,5.8e7
PWIRE,2,2,RECT,.08,.06
CWIRE,2,2,1,2,0.,0.,1.
WIREPATH,2,2
PARAM,WIRELAY,1
PARAM,WIRESTEP,.4
ENDDATA
)";
  for (const std::string name : {"", "AUTO", "hex", "TeT"}) {
    const auto parsed = parse_string(
        prefix + (name.empty() ? "" : "PARAM,WIREMESH," + name + "\n") + body);
    ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
    const auto &model = std::get<Model>(parsed);
    EXPECT_EQ(model.options.wire_mesh, name == "TeT"   ? WireMesh::Tet
                                       : name == "hex" ? WireMesh::Hex
                                                       : WireMesh::Auto);
    const auto prepared = prepare_wires(model);
    ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
        << error_of(prepared);
    const auto &source = std::get<std::vector<Source>>(prepared)[0];
    double volume = 0;
    for (const auto &cell : source.mesh.cells) {
      EXPECT_EQ(cell.nodes.size(), name == "TeT" ? 4U : 8U);
      volume += cell.volume;
      const CurrentField current(cell, source.flux);
      for (const auto &q : quadrature(source.mesh, cell))
        EXPECT_LT(
            (current.value(q.position) - Vector(1 / (.08 * .06), 0, 0)).norm(),
            1e-9);
    }
    EXPECT_NEAR(volume, .08 * .06, 1e-14);
  }
  const auto rejected = parse_string(prefix + "PARAM,WIREMESH,WEDGE\n" + body);
  ASSERT_TRUE(std::holds_alternative<Error>(rejected));
  EXPECT_NE(error_of(rejected).find("WIREMESH"), std::string::npos);
}

TEST(MagneticMappedBasis,
     CachedCurrentMatchesFaceBasisIncludingNonzeroDivergence) {
  for (int kind = 0; kind < 3; ++kind)
    for (bool warped : {false, true})
      for (bool reflected : {false, true}) {
        SCOPED_TRACE(std::format("kind {}, warped {}, reflected {}", kind,
                                 warped, reflected));
        auto model = kind == 0 ? box(1) : brick_box(1, kind == 2);
        for (auto &[id, g] : model.nodes) {
          (void)id;
          if (warped)
            g.position.x() += .2 * g.position.y() * g.position.z();
          if (reflected)
            g.position.x() *= -1;
          g.position += Vector(11, -7, 5);
        }
        const auto built = make_mesh(model);
        ASSERT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
        const auto &mesh = std::get<Mesh>(built);
        Eigen::VectorXd flux(mesh.faces.size());
        for (Eigen::Index i = 0; i < flux.size(); ++i)
          flux[i] = std::sin(.7 * i + .3);
        for (const auto &cell : mesh.cells) {
          const CurrentField current(cell, flux);
          for (const auto &q : reference_quadrature(cell)) {
            const auto [x, j] = mapping(cell, q.position);
            const auto basis = flux_basis(mesh, cell, x);
            Vector expected = Vector::Zero();
            for (size_t i = 0; i < basis.size(); ++i)
              expected += flux[cell.faces[i]] * basis[i];
            EXPECT_LT((current.value(x) - expected).norm(), 2e-12);
            EXPECT_LT((current.weighted_reference(q.position, j) -
                       std::abs(j.determinant()) * expected)
                          .norm(),
                      2e-12);
          }
        }
      }
}

TEST(MagneticWire, HexAndTetSectionsPreserveVolumeCurrentAndFarField) {
  for (bool rectangular : {false, true}) {
    auto model = box(1, 2);
    loop(model, .6, rectangular);
    for (auto kind : {WireMesh::Hex, WireMesh::Tet}) {
      SCOPED_TRACE(std::format("rectangular {}, tet {}", rectangular,
                               kind == WireMesh::Tet));
      model.options.wire_mesh = kind;
      const auto prepared = prepare_wires(model);
      ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
          << error_of(prepared);
      const auto &source = std::get<std::vector<Source>>(prepared)[0];
      double volume = 0;
      Vector moment = Vector::Zero();
      for (const auto &cell : source.mesh.cells) {
        EXPECT_EQ(cell.nodes.size(), kind == WireMesh::Tet ? 4U : 8U);
        volume += cell.volume;
        double balance = 0;
        for (size_t i = 0; i < cell.faces.size(); ++i)
          balance += cell.face_sign[i] * source.flux[cell.faces[i]];
        EXPECT_NEAR(balance, 0, 1e-11);
        for (const auto &q : quadrature(source.mesh, cell))
          moment += .5 * q.weight *
                    q.position.cross(flux_value(cell, source.flux, q.position));
      }
      const double area = rectangular
                              ? .08 * .06
                              : 4 * .04 * .04 * std::sin(std::numbers::pi / 4);
      EXPECT_NEAR(volume, 4 * 1.2 * area, 1e-12);
      EXPECT_LT((moment - Vector(0, 0, 1.44)).norm() / 1.44, .002);
      const Vector b = source_biot(source, Vector::Zero());
      const double exact = mu0 * std::sqrt(2.) / (std::numbers::pi * .6);
      EXPECT_LT((b - Vector(0, 0, exact)).norm() / exact, .01);
    }
  }
}

TEST(MagneticWarped, TetTransferIsInvariantToLocalVertexOrderAndRigidMotion) {
  auto model = brick_box(2);
  for (auto &[id, g] : model.nodes) {
    (void)id;
    g.position.z() += .35 * g.position.x() * g.position.y();
  }
  loop(model, .6, true);
  model.options.wire_mesh = WireMesh::Tet;
  const auto mesh = std::get<Mesh>(make_mesh(model));
  const auto prepared = prepare_sources(model, mesh);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(prepared))
      << error_of(prepared);
  const auto &original = std::get<std::vector<Source>>(prepared)[0];
  auto reordered = original;
  std::vector<NodeList> cells;
  for (const auto &c : original.mesh.cells)
    cells.push_back({c.nodes[2], c.nodes[0], c.nodes[3], c.nodes[1]});
  const auto built = make_mesh(original.mesh.nodes, cells);
  ASSERT_TRUE(std::holds_alternative<Mesh>(built)) << error_of(built);
  reordered.mesh = std::get<Mesh>(built);
  reordered.flux.resize(reordered.mesh.faces.size());
  for (size_t f = 0; f < reordered.mesh.faces.size(); ++f) {
    const auto &face = reordered.mesh.faces[f];
    auto key = face.nodes;
    std::sort(key.begin(), key.end());
    const int index = original.mesh.face_index.at(key);
    reordered.flux[f] = original.flux[index] *
                        face.normal.dot(original.mesh.faces[index].normal);
  }
  const auto coupled = couple_source(reordered, mesh, SpatialIndex(mesh));
  ASSERT_TRUE(std::holds_alternative<Eigen::VectorXd>(coupled))
      << error_of(coupled);
  EXPECT_LT((std::get<Eigen::VectorXd>(coupled) - original.coupling).norm() /
                original.coupling.norm(),
            1e-8);
  const Matrix rotation =
      Eigen::AngleAxisd(.37, Vector(.3, -.4, .8).normalized())
          .toRotationMatrix();
  for (auto &[id, g] : model.nodes) {
    (void)id;
    g.position = rotation * g.position + Vector(3, -2, 4);
  }
  for (auto &wire : model.wires)
    wire.orientation = rotation * *wire.orientation;
  const auto transformed = std::get<Mesh>(make_mesh(model));
  const auto moved = prepare_sources(model, transformed);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(moved))
      << error_of(moved);
  ASSERT_EQ(std::get<std::vector<Source>>(moved)[0].mesh.cells.size(),
            original.mesh.cells.size());
  const auto &load = std::get<std::vector<Source>>(moved)[0].coupling;
  EXPECT_LT((load - original.coupling).norm() / original.coupling.norm(), 1e-8);
}

namespace {
// Three slopes and a vacuum extrapolation branch. Excitations below are chosen
// to visit each regime; field_regimes asserts that at quadrature points.
void nonlinear_steel(Model &model) {
  model.materials[2] = {1, 1};
  model.properties[2] = {2};
  model.bh_tables[1] = {{0, 0}, {.1, 100}, {.3, 2000}, {1, 100000}};
  model.options.nonlinear_tolerance = 1e-12;
}
Model nonlinear_planar(int kind) {
  auto model = planar_box(12, kind == 1, kind == 2);
  nonlinear_steel(model);
  for (auto &cell : model.elements) {
    Vector center = Vector::Zero();
    for (int n : cell.nodes)
      center +=
          model.nodes.at(n).position / static_cast<double>(cell.nodes.size());
    if (center.head<2>().cwiseAbs().maxCoeff() < .32)
      cell.property = 2;
    if (std::abs(center.x()) > .4 && std::abs(center.x()) < .8 &&
        std::abs(center.y()) < .32)
      model.element_sets[center.x() < 0 ? 1 : 2].push_back(cell.id);
    if (std::abs(center.y()) > .4 && std::abs(center.y()) < .8 &&
        std::abs(center.x()) < .32)
      model.element_sets[center.y() < 0 ? 3 : 4].push_back(cell.id);
  }
  model.planar_coils = {
      {11, 1, 100}, {12, 2, -100}, {13, 3, 100}, {14, 4, -100}};
  model.circuits[1] = {1, 1, {{11, 1}, {12, 1}}};
  model.circuits[2] = {2, 1, {{13, 1}, {14, 1}}};
  // Planar distorted elements exercise varying gradients as well as P1 cells.
  for (auto &[id, g] : model.nodes) {
    (void)id;
    g.position.x() += .12 * g.position.x() * g.position.y();
  }
  return model;
}
Model nonlinear_solid(int kind) {
  auto model = kind == 0 ? box(4) : brick_box(4, kind == 2 || kind == 4);
  nonlinear_steel(model);
  for (auto &cell : model.elements) {
    Vector center = Vector::Zero();
    for (int n : cell.nodes)
      center +=
          model.nodes.at(n).position / static_cast<double>(cell.nodes.size());
    if (center.cwiseAbs().maxCoeff() < .5)
      cell.property = 2;
  }
  if (kind >= 3)
    for (auto &[id, g] : model.nodes) {
      (void)id;
      g.position.z() += .18 * g.position.x() * g.position.y();
    }
  loop(model, .6, true);
  return model;
}
// Cases: operating point, low-field linear limit, vacuum extrapolation,
// reversal, then independent +/- perturbations of every circuit current.
void nonlinear_loads(Model &model, const std::vector<double> &amplitudes) {
  model.currents.clear();
  model.subcases.clear();
  auto append = [&](const std::vector<double> &amps) {
    const int id = static_cast<int>(model.subcases.size()) + 1;
    for (size_t i = 0; i < amps.size(); ++i)
      model.currents.push_back(
          {id, model.options.planar ? static_cast<int>(i) + 1 : 100000,
           amps[i]});
    model.subcases.push_back({.id = id, .load = id});
  };
  for (double factor : {1., .01, 20., -1.}) {
    auto amps = amplitudes;
    for (double &a : amps)
      a *= factor;
    append(amps);
  }
  for (size_t i = 0; i < amplitudes.size(); ++i)
    for (double factor : {1.0001, .9999}) {
      auto amps = amplitudes;
      amps[i] *= factor;
      append(amps);
    }
}
void check_nonlinear_response(const Model &model, const Results &result) {
  const auto &base = result.cases[0], &low = result.cases[1],
             &high = result.cases[2], &reverse = result.cases[3];
  const auto regimes = field_regimes(model, result, base);
  ASSERT_GT(regimes.points[1], 0);
  ASSERT_GT(regimes.points[2], 0);
  EXPECT_EQ(regimes.points[3], 0);
  EXPECT_GT(base.iterations, 1);
  EXPECT_GT(high.iterations, 1);
  EXPECT_EQ(low.iterations, 1);
  EXPECT_FALSE(base.extrapolated);
  EXPECT_FALSE(low.extrapolated);
  EXPECT_TRUE(high.extrapolated);
  EXPECT_LT(field_regimes(model, result, low).maximum, .1);
  EXPECT_GT(field_regimes(model, result, high).maximum, 1.);
  for (const auto &state : result.cases) {
    EXPECT_LE(state.residual, model.options.nonlinear_tolerance);
    EXPECT_TRUE(state.potential.allFinite());
    EXPECT_GT(state.energy, 0);
  }
  EXPECT_LT((base.potential + reverse.potential).norm() / base.potential.norm(),
            1e-9);
  EXPECT_NEAR(reverse.energy, base.energy, 1e-9 * base.energy);
  EXPECT_NEAR(reverse.coenergy, base.coenergy, 1e-9 * base.coenergy);
  const auto &tangent = base.circuit_inductance;
  EXPECT_LT((tangent - tangent.transpose()).norm() / tangent.norm(), 1e-12);
  EXPECT_LT((reverse.circuit_inductance - tangent).norm() / tangent.norm(),
            1e-9);
  Eigen::VectorXd amps(base.circuits.size()), linkage(base.circuits.size());
  for (size_t i = 0; i < base.circuits.size(); ++i) {
    amps[i] = base.circuits[i].current;
    linkage[i] = base.circuits[i].linkage;
    EXPECT_GT(tangent(i, i), 0);
    EXPECT_NEAR(reverse.circuits[i].linkage, -linkage[i],
                1e-9 * std::abs(linkage[i]));
    const auto &plus = result.cases[4 + 2 * i],
               &minus = result.cases[5 + 2 * i];
    const double delta = plus.circuits[i].current - minus.circuits[i].current;
    Eigen::VectorXd derivative(linkage.size());
    for (size_t j = 0; j < base.circuits.size(); ++j)
      derivative[j] =
          (plus.circuits[j].linkage - minus.circuits[j].linkage) / delta;
    EXPECT_LT((derivative - tangent.col(i)).norm() / tangent.col(i).norm(),
              1e-6);
    for (size_t j = 0; j < base.circuits.size(); ++j)
      EXPECT_NEAR(derivative[j], tangent(j, i), 1e-6 * std::abs(tangent(j, i)));
    // At fixed geometry with zero applied exterior field and no PM:
    // dW'/dI_i=lambda_i and dW/dI_i=sum_j I_j*d(lambda_j)/dI_i.
    EXPECT_NEAR((plus.coenergy - minus.coenergy) / delta, linkage[i],
                1e-6 * std::abs(linkage[i]));
    double energy_derivative = 0;
    for (size_t j = 0; j < base.circuits.size(); ++j)
      energy_derivative += base.circuits[j].current * tangent(j, i);
    EXPECT_NEAR((plus.energy - minus.energy) / delta, energy_derivative,
                1e-6 * std::abs(energy_derivative));
  }
  Eigen::VectorXd low_linkage(linkage.size()), high_linkage(linkage.size());
  for (size_t i = 0; i < base.circuits.size(); ++i) {
    low_linkage[i] = low.circuits[i].linkage;
    high_linkage[i] = high.circuits[i].linkage;
  }
  EXPECT_LT((low.circuit_inductance * (.01 * amps) - low_linkage).norm() /
                low_linkage.norm(),
            1e-10);
  EXPECT_NEAR(low.energy, .5 * .01 * amps.dot(low_linkage), 1e-10 * low.energy);
  EXPECT_LT(amps.dot(high_linkage) / 20, .99 * amps.dot(low_linkage) / .01);
  const double work = amps.dot(linkage);
  EXPECT_GT(work - amps.dot(tangent * amps), .001 * work);
  EXPECT_GT(.5 * work - base.energy, .001 * base.energy);
  if (amps.size() == 2) {
    EXPECT_GT(tangent.determinant(), 0);
    EXPECT_GT(std::abs(tangent(0, 1)),
              .001 * std::sqrt(tangent(0, 0) * tangent(1, 1)));
  }
}
class MagneticNonlinearPlanar : public ::testing::TestWithParam<int> {};
class MagneticNonlinearSolids : public ::testing::TestWithParam<int> {};
} // namespace

TEST_P(MagneticNonlinearPlanar, SaturationMutualTangentEnergyAndReversal) {
  auto model = nonlinear_planar(GetParam());
  nonlinear_loads(model, {4000, 2500});
  const auto solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &result = std::get<Results>(solved);
  const auto regimes = field_regimes(model, result, result.cases[0]);
  RecordProperty("field_cells", static_cast<int>(result.mesh.cells.size()));
  RecordProperty("free_dofs", std::get<Gauge>(make_gauge(result.mesh)).size);
  RecordProperty("base_iterations", result.cases[0].iterations);
  RecordProperty("base_Bmax_T", std::to_string(regimes.maximum));
  RecordProperty(
      "high_Bmax_T",
      std::to_string(field_regimes(model, result, result.cases[2]).maximum));
  check_nonlinear_response(model, result);
}
TEST_P(MagneticNonlinearSolids, SaturationTangentEnergyAndReversal) {
  auto model = nonlinear_solid(GetParam());
  nonlinear_loads(model, {100000});
  const auto solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &result = std::get<Results>(solved);
  const auto regimes = field_regimes(model, result, result.cases[0]);
  RecordProperty("field_cells", static_cast<int>(result.mesh.cells.size()));
  RecordProperty("free_dofs", std::get<Gauge>(make_gauge(result.mesh)).size);
  RecordProperty("base_iterations", result.cases[0].iterations);
  RecordProperty("base_Bmax_T", std::to_string(regimes.maximum));
  RecordProperty(
      "high_Bmax_T",
      std::to_string(field_regimes(model, result, result.cases[2]).maximum));
  check_nonlinear_response(model, result);
}
INSTANTIATE_TEST_SUITE_P(MagneticElements, MagneticNonlinearPlanar,
                         ::testing::Values(0, 1, 2), ([](const auto &info) {
                           return std::array{"Quad", "Tri",
                                             "Mixed"}[info.param];
                         }));
INSTANTIATE_TEST_SUITE_P(MagneticElements, MagneticNonlinearSolids,
                         ::testing::Values(0, 1, 2, 3, 4),
                         ([](const auto &info) {
                           return std::array{"Tet", "Hex", "Wedge", "WarpedHex",
                                             "WarpedWedge"}[info.param];
                         }));

TEST(MagneticMaterial, KnotContinuityRadialAndTransverseTangents) {
  Material material;
  material.bh_table = 1;
  const std::vector<std::pair<double, double>> curve{
      {0, 0}, {1, 200}, {1.5, 1000}, {2, 10000}};
  struct Knot {
    double b, h, energy, left, right;
  };
  const std::array<Knot, 3> knots{{{1, 200, 100, 200, 1600},
                                   {1.5, 1000, 400, 1600, 18000},
                                   {2, 10000, 3150, 18000, 1 / mu0}}};
  const Vector radial(.6, .8, 0), transverse(-.8, .6, 0);
  for (const auto &k : knots) {
    const auto at = material_response(material, curve, Vector(k.b, 0, 0));
    EXPECT_NEAR(at.h.x(), k.h, 1e-12 * k.h);
    EXPECT_NEAR(at.energy, k.energy, 1e-12 * k.energy);
    EXPECT_FALSE(at.extrapolated);
    for (double delta : {-1e-8, 1e-8}) {
      const double slope = delta < 0 ? k.left : k.right;
      const double magnitude = k.b + delta, h = k.h + slope * delta;
      const double energy = k.energy + k.h * delta + .5 * slope * delta * delta;
      const auto value = material_response(material, curve, magnitude * radial);
      EXPECT_LT((value.h - h * radial).norm(), 1e-10 * h);
      EXPECT_NEAR(value.energy, energy, 1e-11 * energy);
      EXPECT_LT((value.tangent * radial - slope * radial).norm(),
                1e-10 * slope);
      EXPECT_LT(
          (value.tangent * transverse - h / magnitude * transverse).norm(),
          1e-10 * h / magnitude);
      EXPECT_LT(
          (value.tangent * Vector::UnitZ() - h / magnitude * Vector::UnitZ())
              .norm(),
          1e-10 * h / magnitude);
      EXPECT_EQ(value.extrapolated, magnitude > 2);
      const auto negative =
          material_response(material, curve, -magnitude * radial);
      EXPECT_LT((negative.h + value.h).norm(), 1e-10 * h);
      EXPECT_NEAR(negative.energy, value.energy, 1e-11 * energy);
      EXPECT_LT((negative.tangent - value.tangent).norm(), 1e-10 * slope);
    }
  }
  const auto origin = material_response(material, curve, Vector::Zero());
  EXPECT_EQ(origin.energy, 0);
  EXPECT_EQ(origin.h.norm(), 0);
  EXPECT_LT((origin.tangent - 200 * Matrix::Identity()).norm(), 1e-12);
}

TEST(MagneticNonlinear,
     ContinuationReachesFullCurrentAndRejectsIncompleteStages) {
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar);
    auto model = planar ? nonlinear_planar(0) : nonlinear_solid(1);
    const std::vector<double> amplitudes =
        planar ? std::vector<double>{16000, 10000}
               : std::vector<double>{150000};
    nonlinear_loads(model, amplitudes);
    model.subcases.resize(1);
    const auto direct = solve(model);
    ASSERT_TRUE(std::holds_alternative<Results>(direct)) << error_of(direct);
    const auto &reference = std::get<Results>(direct).cases[0];
    model.options.max_iterations = planar ? 4 : 3;
    ASSERT_GT(reference.iterations, model.options.max_iterations);
    const auto retried = solve(model);
    ASSERT_TRUE(std::holds_alternative<Results>(retried)) << error_of(retried);
    const auto &continued = std::get<Results>(retried).cases[0];
    // More total iterations than the per-stage budget proves the retry ran.
    EXPECT_GT(continued.iterations, model.options.max_iterations);
    for (size_t i = 0; i < amplitudes.size(); ++i) {
      EXPECT_DOUBLE_EQ(continued.circuits[i].current, amplitudes[i]);
      EXPECT_NEAR(continued.circuits[i].linkage, reference.circuits[i].linkage,
                  1e-10 * std::abs(reference.circuits[i].linkage));
    }
    EXPECT_LE(continued.residual, model.options.nonlinear_tolerance);
    EXPECT_LT((continued.potential - reference.potential).norm() /
                  reference.potential.norm(),
              1e-9);
    EXPECT_NEAR(continued.energy, reference.energy, 1e-10 * reference.energy);
    EXPECT_LT(
        (continued.circuit_inductance - reference.circuit_inductance)
                .norm() /
            reference.circuit_inductance.norm(),
        1e-8);
    RecordProperty(planar ? "planar_direct_iterations"
                          : "solid_direct_iterations",
                   reference.iterations);
    RecordProperty(planar ? "planar_continuation_iterations"
                          : "solid_continuation_iterations",
                   continued.iterations);
    model.options.max_iterations = 1;
    const auto failed = solve(model);
    ASSERT_TRUE(std::holds_alternative<Error>(failed));
    EXPECT_NE(error_of(failed).find("continuation stage"), std::string::npos);
    EXPECT_NE(error_of(failed).find("Newton did not converge"),
              std::string::npos);
  }
}

TEST(MagneticNonlinear, PermanentMagnetDrivesSteelAcrossTheKneeAndReverses) {
  for (int kind : {0, 1}) {
    SCOPED_TRACE(kind);
    auto model = nonlinear_planar(kind);
    model.planar_coils.clear();
    model.circuits.clear();
    model.currents.clear();
    model.subcases = {{}};
    model.materials[3] = {1.05, 0, 1.2};
    model.properties[3] = {3, 0, 1, true};
    for (int set : {1, 2})
      for (int id : model.element_sets.at(set))
        for (auto &cell : model.elements)
          if (cell.id == id)
            cell.property = 3;
    const auto positive = solve(model);
    ASSERT_TRUE(std::holds_alternative<Results>(positive))
        << error_of(positive);
    const auto &result = std::get<Results>(positive);
    const auto &state = result.cases[0];
    const auto regimes = field_regimes(model, result, state);
    EXPECT_GT(state.iterations, 1);
    EXPECT_GT(regimes.points[1], 0);
    EXPECT_GT(regimes.points[2], 0);
    EXPECT_FALSE(state.extrapolated);
    EXPECT_LE(state.residual, model.options.nonlinear_tolerance);
    model.properties[3].axis = -1;
    const auto negative = solve(model);
    ASSERT_TRUE(std::holds_alternative<Results>(negative))
        << error_of(negative);
    const auto &reversed = std::get<Results>(negative).cases[0];
    EXPECT_LT((state.potential + reversed.potential).norm() /
                  state.potential.norm(),
              1e-9);
    EXPECT_NEAR(state.energy, reversed.energy, 1e-10 * state.energy);
  }
}

TEST(MagneticParser, BHCurvesRejectNonpositiveTangentsAndMissingOrigin) {
  const std::string deck = R"(SOL MAGSTAT
CEND
BEGIN BULK
GRID,1,,0,0,0
GRID,2,,1,0,0
GRID,3,,0,1,0
GRID,4,,0,0,1
CTETRA,1,1,1,2,3,4
PSOLID,1,1
MATMAG,1,,1
)";
  for (const std::string points :
       {"TABLEBH,1\n,0,0,ENDT\n",              // insufficient to define a slope
        "TABLEBH,1\n,.1,0,1,100,ENDT\n",       // missing (0,0)
        "TABLEBH,1\n,0,0,1,100,1,200,ENDT\n",  // duplicate B
        "TABLEBH,1\n,0,0,1,100,.5,200,ENDT\n", // unsorted B
        "TABLEBH,1\n,0,0,1,100,2,100,ENDT\n",  // zero tangent
        "TABLEBH,1\n,0,0,1,100,2,50,ENDT\n"}) { // negative tangent
    SCOPED_TRACE(points);
    const auto parsed = parse_string(deck + points);
    ASSERT_TRUE(std::holds_alternative<Error>(parsed));
    EXPECT_NE(error_of(parsed).find("TABLEBH 1"), std::string::npos);
  }
  // Positive differential permeability is required; the slopes themselves
  // need not increase monotonically for the constitutive tangent to be SPD.
  const auto valid = parse_string(deck + "TABLEBH,1,LINEAR\n,0,0,1,100\n,2,"
                                         "150,ENDT\n");
  ASSERT_TRUE(std::holds_alternative<Model>(valid)) << error_of(valid);
  const auto &model = std::get<Model>(valid);
  const auto response = material_response(
      model.materials.at(1), model.bh_tables.at(1), Vector(1.5, 0, 0));
  EXPECT_DOUBLE_EQ(response.h.x(), 125);
  EXPECT_DOUBLE_EQ(response.tangent(0, 0), 50);
  EXPECT_NEAR(response.tangent(1, 1), 125 / 1.5, 1e-12);
}

TEST(MagneticParser, MagnetOrientationAndTablesAreExplicit) {
  const std::string head = R"(SOL MAGSTAT
CEND
BEGIN BULK
GRID,1,,0,0,0
GRID,2,,1,0,0
GRID,3,,0,1,0
GRID,4,,0,0,1
CTETRA,1,1,1,2,3,4
CORD2C,5,,0,0,0,0,0,1
,1,0,0
)";
  auto parse = [&](const std::string &cards) {
    return parse_string(head + cards + "ENDDATA\n");
  };
  // A forgotten frame must not silently magnetize along basic x, while an
  // explicit 0 does mean basic.
  const std::string magnet = "MATMAG,1,1.05,,1.2\n";
  EXPECT_NE(error_of(parse(magnet + "PSOLID,1,1\n")).find("without an orient"),
            std::string::npos);
  EXPECT_NE(error_of(parse(magnet + "PMAG,1,1,,2\n")).find("without an orient"),
            std::string::npos);
  for (const std::string property : {"PSOLID,1,1,0\n", "PMAG,1,1,0\n"}) {
    auto parsed = parse(magnet + property);
    ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
    const auto &p = std::get<Model>(parsed).properties.at(1);
    EXPECT_TRUE(p.oriented);
    EXPECT_EQ(p.coordinates, 0);
    EXPECT_EQ(p.axis, 1);
  }
  // PSOLID CORDM is PMAG with AXIS +1; PMAG carries a signed axis.
  auto cordm = parse(magnet + "PSOLID,1,1,5\n");
  ASSERT_TRUE(std::holds_alternative<Model>(cordm)) << error_of(cordm);
  EXPECT_EQ(std::get<Model>(cordm).properties.at(1).coordinates, 5);
  EXPECT_EQ(std::get<Model>(cordm).properties.at(1).axis, 1);
  auto signed_axis = parse(magnet + "PMAG,1,1,5,-2\n");
  ASSERT_TRUE(std::holds_alternative<Model>(signed_axis))
      << error_of(signed_axis);
  EXPECT_EQ(std::get<Model>(signed_axis).properties.at(1).axis, -2);
  EXPECT_NE(error_of(parse(magnet + "PMAG,1,1,5,4\n")).find("AXIS"),
            std::string::npos);
  // An unmagnetized material ignores a missing frame.
  EXPECT_TRUE(std::holds_alternative<Model>(parse("MATMAG,1,2\nPSOLID,1,1\n")));
  EXPECT_NE(error_of(parse("MATMAG,1,1,1\nPSOLID,1,1\nTABLEBH,1\n,0,0,1,100,"
                           "ENDT\n"))
                .find("mutually exclusive"),
            std::string::npos);

  // TABLEBH reads B,H pairs across fixed-format continuations, ignoring the
  // padding blanks, and must end with ENDT.
  const std::string steel = "MATMAG,1,,1\nPSOLID,1,1\n";
  std::string fixed = std::format("{:<8}{:<8}{:<8}\n", "TABLEBH", "1", "");
  fixed += std::format("{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}\n", "", "0.", "0.",
                       ".5", "250.", "1.", "1000.");
  fixed += std::format("{:<8}{:<8}{:<8}{:<8}\n", "", "1.5", "8.+3", "ENDT");
  auto table = parse(steel + fixed);
  ASSERT_TRUE(std::holds_alternative<Model>(table)) << error_of(table);
  const std::vector<std::pair<double, double>> expected{
      {0, 0}, {.5, 250}, {1, 1000}, {1.5, 8000}};
  EXPECT_EQ(std::get<Model>(table).bh_tables.at(1), expected);
  for (const auto &[cards, message] :
       std::vector<std::pair<std::string, std::string>>{
           {"TABLEBH,1\n,0,0,1,100\n", "ending in ENDT"},
           {"TABLEBH,1\n,0,0,1,ENDT\n", "ending in ENDT"},
           {"TABLEBH,1\n,0,0,1,100,ENDT,2,200\n", "follows ENDT"},
           {"TABLEBH,1,SPLINE\n,0,0,1,100,ENDT\n", "LINEAR"},
           {"TABLEBH,1\n,0,0,1,100,ENDT\nTABLEBH,1\n,0,0,1,50,ENDT\n",
            "duplicate id 1"}}) {
    SCOPED_TRACE(cards);
    EXPECT_NE(error_of(parse(steel + cards)).find(message), std::string::npos)
        << error_of(parse(steel + cards));
  }
}

TEST(MagneticCurrent, HalfPlaneCoilCutMatchesTheExplicitSection) {
  // The square ring of ClosedVolumetricCoilCutAndTurnScaling, cut on y = 0,
  // x > 0 with normal +y. A CORD2R with local z = basic y and local x = basic
  // x states the same cut.
  auto model = box(6, 1.5);
  const auto mesh = std::get<Mesh>(make_mesh(model));
  for (const auto &c : mesh.cells) {
    Vector center = Vector::Zero();
    for (int n : c.nodes)
      center += mesh.nodes[n] / 4;
    const double radius = center.head<2>().cwiseAbs().maxCoeff();
    if (radius > 0.5 && radius < 1 && std::abs(center.z()) < 0.5)
      model.element_sets[1].push_back(c.id);
  }
  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    const auto &f = mesh.faces[fi];
    if (std::abs(f.center.y()) < 1e-12 && f.center.x() > 0.5 &&
        f.center.x() < 1 && std::abs(f.center.z()) < 0.5)
      model.face_sets[1].push_back(face_selection(mesh, static_cast<int>(fi),
                                                  f.normal.y() > 0 ? 1 : -1));
  }
  model.coils = {{1, 1, 20, 1, 0}};
  const auto explicit_cut = prepare_sources(model, mesh);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(explicit_cut))
      << error_of(explicit_cut);
  const auto &reference = std::get<std::vector<Source>>(explicit_cut)[0];
  auto cut_at = [&](const Vector &origin, const Vector &z, const Vector &x) {
    auto m = model;
    m.coordinates[9] = {0, 'R', origin, origin + z, origin + x};
    m.coils = {{1, 1, 20, 0, 0, true, 9}};
    return prepare_sources(m, mesh);
  };
  // On the mesh plane the derived faces are the explicit ones. Shifted into
  // a layer of tetrahedra, the cut zig-zags, but a conserved current has the
  // same flux through any separating surface, so the source is unchanged.
  for (double offset : {0., .1, -.3}) {
    SCOPED_TRACE(offset);
    const auto derived =
        cut_at(Vector(0, offset, 0), Vector::UnitY(), Vector::UnitX());
    ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(derived))
        << error_of(derived);
    const auto &source = std::get<std::vector<Source>>(derived)[0];
    EXPECT_LT((source.coupling - reference.coupling).norm(),
              1e-9 * reference.coupling.norm());
    EXPECT_LT((source.flux - reference.flux).norm(),
              1e-9 * reference.flux.norm());
  }
  // A plane beside the coil, a half-plane edge through the section, and a
  // half-plane through both sides of the ring are each rejected.
  EXPECT_NE(error_of(cut_at(Vector(0, 5, 0), Vector::UnitY(), Vector::UnitX()))
                .find("does not cross the winding"),
            std::string::npos);
  EXPECT_NE(
      error_of(cut_at(Vector(.75, 0, 0), Vector::UnitY(), Vector::UnitX()))
          .find("complete cross section"),
      std::string::npos);
  EXPECT_NE(error_of(cut_at(Vector(0, 0, -1), Vector::UnitY(), Vector::UnitZ()))
                .find("more than once"),
            std::string::npos);
  // The same complete-section check applies to an explicit FSET cut.
  auto partial = model;
  partial.face_sets[1].resize(partial.face_sets[1].size() / 2);
  EXPECT_NE(error_of(prepare_sources(partial, mesh)).find("complete cross"),
            std::string::npos);
}

namespace {
// Hexahedral voxels of brick_box(n) indexed by integer position.
std::array<int, 3> voxel(const Model &m, const Element &c, int n) {
  Vector center = Vector::Zero();
  for (int v : c.nodes)
    center += m.nodes.at(v).position / 8;
  std::array<int, 3> index{};
  for (int k = 0; k < 3; ++k)
    index[k] = static_cast<int>(std::floor((center[k] + 1) * n / 2));
  return index;
}
// Outward boundary of a cell set, built independently of the solver's body
// surface construction.
std::vector<FaceSelection>
region_boundary(const Mesh &mesh, const std::function<bool(const Cell &)> &in) {
  std::vector<FaceSelection> out;
  for (size_t i = 0; i < mesh.faces.size(); ++i) {
    const auto &f = mesh.faces[i];
    if (f.cells[1] < 0 || in(mesh.cells[f.cells[0]]) == in(mesh.cells[f.cells[1]]))
      continue;
    const auto &c = mesh.cells[f.cells[in(mesh.cells[f.cells[0]]) ? 0 : 1]];
    const auto k = std::find(c.faces.begin(), c.faces.end(),
                             static_cast<int>(i)) -
                   c.faces.begin();
    out.push_back(face_selection(mesh, static_cast<int>(i), c.face_sign[k]));
  }
  return out;
}
// Two x-magnetized blocks in air, 3 voxels apart: A at i in [4, 5], B at
// i in [9, 10], both with j, k in [6, 7] on a 14^3 hexahedral grid.
Model magnet_pair() {
  auto m = brick_box(14);
  m.materials[2] = {1.05, 0, 1};
  m.properties[2] = {2, 0, 1, true};
  for (auto &c : m.elements) {
    const auto [i, j, k] = voxel(m, c, 14);
    if (j >= 6 && j <= 7 && k >= 6 && k <= 7 &&
        ((i >= 4 && i <= 5) || (i >= 9 && i <= 10))) {
      c.property = 2;
      m.element_sets[i <= 5 ? 1 : 2].push_back(c.id);
    }
  }
  return m;
}
} // namespace

TEST(MagneticForce, BodySurfacesMatchExplicitSurfacesAndDiagnoseContact) {
  auto model = magnet_pair();
  const auto mesh = std::get<Mesh>(make_mesh(model));
  // Body A grown by LAYER voxels in every direction, through shared grids.
  auto grown = [&](int layers) {
    return region_boundary(mesh, [&, layers](const Cell &c) {
      const auto [i, j, k] =
          voxel(model, model.elements[mesh.element_index.at(c.id)], 14);
      return i >= 4 - layers && i <= 5 + layers && j >= 6 - layers &&
             j <= 7 + layers && k >= 6 - layers && k <= 7 + layers;
    });
  };
  for (int layers : {1, 2}) {
    SCOPED_TRACE(layers);
    auto m = model;
    m.face_sets[7] = grown(layers);
    m.forces = {{1, 1, true, 0, layers}, {2, 7, false, 0, 1}};
    auto solved = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    const auto &f = std::get<Results>(solved).cases[0].forces;
    EXPECT_GT(f[0].force.x(), 0); // A is attracted towards B
    EXPECT_LT((f[0].force - f[1].force).norm(), 1e-12 * f[0].force.norm());
    EXPECT_LT((f[0].torque - f[1].torque).norm(),
              1e-12 * (f[0].torque.norm() + f[0].force.norm()));
    ASSERT_EQ(f[0].components.size(), 1U);
    EXPECT_TRUE(f[1].components.empty());
  }
  auto with = [&](std::vector<ForceSurface> requests) {
    auto m = model;
    m.forces = std::move(requests);
    return solve(m);
  };
  // LAYER 3 reaches the voxel next to B; a half magnet touches its other
  // half; a body at the domain edge has no room for its layers.
  EXPECT_NE(error_of(with({{1, 1, true, 0, 3}})).find("within 4 layers"),
            std::string::npos);
  auto half = model;
  std::erase_if(half.element_sets[1], [&](int id) {
    return voxel(half, half.elements[id - 1], 14)[0] == 5;
  });
  half.forces = {{1, 1, true, 0, 1}};
  EXPECT_NE(error_of(solve(half)).find("touches body ESET 1"),
            std::string::npos);
  auto edge = model;
  edge.element_sets[3] = {1};
  edge.forces = {{1, 3, true, 0, 1}};
  EXPECT_NE(error_of(solve(edge)).find("domain boundary"), std::string::npos);

  // Both magnets in one request: two separate pieces whose forces are the
  // single-body results, and whose sum is the total.
  auto pair = model;
  pair.element_sets[3] = pair.element_sets[1];
  pair.element_sets[3].insert(pair.element_sets[3].end(),
                              pair.element_sets[2].begin(),
                              pair.element_sets[2].end());
  // A plastic sleeve (mu_r = 1, another material) around A changes nothing.
  pair.materials[3] = {1, 0, 0};
  pair.properties[3] = {3};
  for (auto &c : pair.elements) {
    const auto [i, j, k] = voxel(pair, c, 14);
    if (c.property == 1 && i >= 3 && i <= 6 && j >= 5 && j <= 8 && k >= 5 &&
        k <= 8)
      c.property = 3;
  }
  pair.forces = {{1, 3, true, 0, 1}, {2, 1, true, 0, 1}, {3, 2, true, 0, 1}};
  auto solved = solve(pair);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &f = std::get<Results>(solved).cases[0].forces;
  ASSERT_EQ(f[0].components.size(), 2U);
  EXPECT_LT((f[0].components[0].force - f[1].force).norm(),
            1e-12 * f[1].force.norm());
  EXPECT_LT((f[0].components[1].force - f[2].force).norm(),
            1e-12 * f[1].force.norm());
  EXPECT_LT((f[0].force - f[1].force - f[2].force).norm(),
            1e-12 * f[1].force.norm());
  // The magnets attract each other.
  EXPECT_GT(f[1].force.x(), 0);
  EXPECT_LT(f[2].force.x(), 0);
  auto unsleeved = solve([&] {
    auto m = model;
    m.forces = {{1, 1, true, 0, 1}};
    return m;
  }());
  ASSERT_TRUE(std::holds_alternative<Results>(unsleeved));
  EXPECT_LT((std::get<Results>(unsleeved).cases[0].forces[0].force - f[1].force)
                .norm(),
            1e-12 * f[1].force.norm());
  // Torque about a CID origin o is the basic torque less o x F.
  auto shifted = model;
  const Vector origin(.3, -.2, .5);
  shifted.coordinates[4] = {0, 'R', origin, origin + Vector::UnitZ(),
                            origin + Vector::UnitX()};
  shifted.forces = {{1, 1, true, 0, 1}, {2, 1, true, 4, 1}};
  auto about = solve(shifted);
  ASSERT_TRUE(std::holds_alternative<Results>(about)) << error_of(about);
  const auto &g = std::get<Results>(about).cases[0].forces;
  EXPECT_LT((g[1].torque - (g[0].torque - origin.cross(g[0].force))).norm(),
            1e-12 * g[0].force.norm());
}

TEST(MagneticForce, BodyLayersCannotAbsorbPlanarWindings) {
  for (bool triangles : {false, true}) {
    SCOPED_TRACE(triangles);
    auto m = planar_box(32, triangles);
    for (const auto &c : m.elements) {
      Vector center = Vector::Zero();
      for (int v : c.nodes)
        center += m.nodes.at(v).position / static_cast<double>(c.nodes.size());
      if (center.cwiseAbs().maxCoeff() < .125)
        m.element_sets[30].push_back(c.id);
      if (std::abs(center.y()) < .25 && std::abs(center.x()) > .375 &&
          std::abs(center.x()) < .625)
        m.element_sets[center.x() < 0 ? 10 : 20].push_back(c.id);
      if (center.cwiseAbs().maxCoeff() < .625)
        m.element_sets[40].push_back(c.id);
    }
    m.planar_coils = {{11, 10, 1}, {12, 20, -1}};
    m.circuits[1] = {1, 1, {{11, 1}, {12, 1}}};
    m.currents = {{1, 1, 100}};
    m.boundaries[1] = {0, Vector(.3, 0, 0)};
    m.subcases = {{.id = 1, .load = 1, .boundary = 1}};
    // A nonmagnetic body at the center feels no force. Its small contour
    // excludes the windings; growing it to +/-0.75 would add their -9 Nm
    // torque even though the final contour does not cut either winding.
    m.forces = {{7, 30, true, 0, 2}};
    auto small = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(small)) << error_of(small);
    const auto &f = std::get<Results>(small).cases[0].forces[0];
    EXPECT_LT(f.force.norm(), 1e-8);
    EXPECT_LT(f.torque.norm(), .02);
    m.forces[0].layers = 10;
    for (double current : {100., 0.}) {
      m.currents[0].amperes = current;
      const auto rejected = solve(m);
      ASSERT_TRUE(std::holds_alternative<Error>(rejected));
      const auto message = error_of(rejected);
      EXPECT_NE(message.find("MFORCE 7"), std::string::npos);
      EXPECT_NE(message.find("body ESET 30"), std::string::npos);
      EXPECT_NE(message.find("current source"), std::string::npos);
    }
    // Explicitly selecting both windings gives the expected Lorentz torque.
    // For quads the derived contour is the square at +/-0.75. Grid-adjacent
    // triangle layers have different corner facets, so both contours must
    // approximate the analytical torque but need not give identical values.
    m.currents[0].amperes = 100;
    m.forces = {{7, 40, true, 0, 2}, {8, 50}};
    m.face_sets[50] = centered_surface(std::get<Mesh>(make_mesh(m)), .75);
    auto selected = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(selected)) << error_of(selected);
    const auto &forces = std::get<Results>(selected).cases[0].forces;
    const double expected = -100 * .3 * .3; // 2*a*NI*Bx*depth, a=0.5
    EXPECT_NEAR(forces[0].torque.z(), expected, .03 * std::abs(expected));
    EXPECT_NEAR(forces[1].torque.z(), expected, .03 * std::abs(expected));
    if (!triangles) {
      EXPECT_LT((forces[0].force - forces[1].force).norm(), 1e-8);
      EXPECT_LT((forces[0].torque - forces[1].torque).norm(), 1e-8);
    }
  }
}

TEST(MagneticForce, BodyLayersCannotAbsorbSolidCoilsOrSweptWires) {
  // Native COIL, hex/tet CWIRE sources, and a CWIRE on a warped field mesh.
  for (int kind : {0, 1, 2, 3}) {
    SCOPED_TRACE(kind);
    auto m = brick_box(10);
    for (const auto &c : m.elements) {
      Vector center = Vector::Zero();
      for (int v : c.nodes)
        center += m.nodes.at(v).position / 8;
      if (center.cwiseAbs().maxCoeff() < .2)
        m.element_sets[30].push_back(c.id);
      if (center.cwiseAbs().maxCoeff() < .6)
        m.element_sets[40].push_back(c.id);
      const double radius = center.head<2>().cwiseAbs().maxCoeff();
      if (radius > .4 && radius < .6 && std::abs(center.z()) < .2)
        m.element_sets[10].push_back(c.id);
    }
    if (kind == 0) {
      m.coordinates[9] = {0, 'R', Vector::Zero(), Vector::UnitY(),
                          Vector::UnitX()};
      m.coils = {{11, 10, 1, 0, 0, true, 9}};
      m.currents = {{1, 11, 1}};
      m.subcases[0].load = 1;
    } else {
      if (kind == 3)
        for (auto &[id, grid] : m.nodes) {
          (void)id;
          grid.position.z() += .1 * grid.position.x() * grid.position.y();
        }
      loop(m, .5);
      m.options.wire_mesh = kind == 2 ? WireMesh::Tet : WireMesh::Hex;
    }
    m.boundaries[1] = {0, Vector(0, .001, 0)};
    m.subcases[0].boundary = 1;
    m.forces = {{7, 30, true, 0, 1}};
    auto small = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(small)) << error_of(small);
    // Layer 2 encloses the entire source while its final surface is clear.
    m.forces[0].layers = 2;
    auto rejected = solve(m);
    ASSERT_TRUE(std::holds_alternative<Error>(rejected));
    const auto message = error_of(rejected);
    EXPECT_NE(message.find("MFORCE 7: layer"), std::string::npos) << message;
    EXPECT_NE(message.find("body ESET 30"), std::string::npos) << message;
    EXPECT_NE(message.find(kind == 0 ? "current source 11"
                                     : "CWIRE path 100000"),
              std::string::npos) << message;
    // Including all field cells containing the source in the body is valid.
    // The growth now adds only air, including for the warped field mesh.
    m.forces = {{7, 40, true, 0, 1}, {8, 50}};
    const auto mesh = std::get<Mesh>(make_mesh(m));
    m.face_sets[50] = region_boundary(mesh, [](const Cell &c) {
      Vector center = Vector::Zero();
      for (const auto &p : c.points)
        center += p / static_cast<double>(c.points.size());
      return center.cwiseAbs().maxCoeff() < .8;
    });
    auto selected = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(selected)) << error_of(selected);
    const auto &forces = std::get<Results>(selected).cases[0].forces;
    EXPECT_LT((forces[0].force - forces[1].force).norm(), 1e-10);
    EXPECT_LT((forces[0].torque - forces[1].torque).norm(), 1e-10);
    EXPECT_LT(forces[0].torque.x(), 0);
  }
}

TEST(MagneticForce, PinchedBodySurfaceSumsItsTwoPieces) {
  // Single-voxel magnets C and D whose one-layer regions meet only along an
  // edge, so the derived surface has four faces on that edge. It cannot be
  // given as a closed FSET, but the boundary of the union is still exact.
  auto m = brick_box(12);
  m.materials[2] = {1.05, 0, 1};
  m.properties[2] = {2, 0, 2, true};
  for (auto &c : m.elements) {
    const auto [i, j, k] = voxel(m, c, 12);
    if (k == 5 && ((i == 3 && j == 5) || (i == 6 && j == 8))) {
      c.property = 2;
      m.element_sets[i == 3 ? 1 : 2].push_back(c.id);
      m.element_sets[3].push_back(c.id);
    }
  }
  m.forces = {{1, 3, true, 0, 1}, {2, 1, true, 0, 1}, {3, 2, true, 0, 1}};
  auto solved = solve(m);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &f = std::get<Results>(solved).cases[0].forces;
  EXPECT_GT(f[1].force.norm(), 0);
  EXPECT_LT((f[0].force - f[1].force - f[2].force).norm(),
            1e-12 * f[1].force.norm());
  EXPECT_LT((f[0].torque - f[1].torque - f[2].torque).norm(),
            1e-12 * (f[1].torque.norm() + f[1].force.norm()));
  // Sharing grids along the edge makes it one connected piece.
  EXPECT_EQ(f[0].components.size(), 1U);
}

TEST(MagneticForce, StatorBodyIncludesItsCavityWallSoTorquesBalance) {
  // A magnet rotor inside a two-tooth steel stator. The stator surface has
  // an outer wall and a cavity wall in the air gap; without the cavity wall
  // its result would be the torque on stator and rotor together, about zero.
  // The reaction torque balances the rotor torque as the mesh is refined
  // (about 21% then 1.6% imbalance on these meshes).
  double previous = 1;
  for (int scale : {1, 2}) {
    SCOPED_TRACE(scale);
    const int n = 16 * scale;
    auto m = brick_box(n);
    m.materials[2] = {1.05, 0, 1};
    m.coordinates[1] = {0, 'R', Vector::Zero(), Vector::UnitZ(),
                        Vector(std::cos(.5), std::sin(.5), 0)};
    m.properties[2] = {2, 1, 1, true};
    m.materials[3] = {1000};
    m.properties[3] = {3};
    for (auto &c : m.elements) {
      auto [i, j, k] = voxel(m, c, n);
      i /= scale;
      j /= scale;
      k /= scale;
      if (k < 6 || k > 9)
        continue;
      // Doubled coarse-index distance from the axis: rotor 1, teeth 7,
      // ring 9 to 11.
      const int radius = std::max(std::abs(2 * i - 15), std::abs(2 * j - 15));
      const bool tooth = (i == 4 || i == 11) && j >= 7 && j <= 8;
      if (radius == 1) {
        c.property = 2;
        m.element_sets[1].push_back(c.id);
      } else if ((radius >= 9 && radius <= 11) || tooth) {
        c.property = 3;
        m.element_sets[2].push_back(c.id);
      }
    }
    m.forces = {{1, 1, true, 0, 1}, {2, 2, true, 0, 1}};
    auto solved = solve(m);
    ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
    const auto &f = std::get<Results>(solved).cases[0].forces;
    const double rotor = f[0].torque.z(), stator = f[1].torque.z();
    // The magnet at 0.5 rad is pulled towards the x-axis teeth.
    EXPECT_LT(rotor, 0);
    const double imbalance = std::abs(stator / rotor + 1);
    EXPECT_LT(imbalance, previous);
    previous = imbalance;
    if (scale == 2)
      EXPECT_LT(imbalance, .03);
  }
}

TEST(MagneticIO, ProbeAndForceFramesInTheReports) {
  const auto directory =
      std::filesystem::temp_directory_path() / "vibestran_magnetic_frames";
  std::filesystem::create_directories(directory);
  const auto path = directory / "frames.bdf";
  // A uniform field; probe 1 is given and reported in a cylindrical system,
  // probe 2 is the same point in basic.
  const double r = 500, theta = 30, degree = std::numbers::pi / 180;
  {
    std::ofstream out(path);
    out << "SOL MAGSTAT\nCEND\nMAGBC=1\nBEGIN BULK\nPARAM,MAGLEN,.001\n"
        << "CORD2C,5,,0,0,0,0,0,1\n,1,0,0\n"
        << mesh_cards(box(2, 1000)) << "PSOLID,1,1\nMATMAG,1,1\n"
        << "MBC,1,0,UNIFORM,.1,.2,.3\n"
        << std::format("MPROBE,1,5,{},{},100.,5\n", r, theta)
        << std::format("MPROBE,2,0,{},{},100.\n", r * std::cos(theta * degree),
                       r * std::sin(theta * degree))
        << "ENDDATA\n";
  }
  auto parsed = parse_file(path);
  ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
  const auto &model = std::get<Model>(parsed);
  EXPECT_LT((model.probes[0].position - model.probes[1].position).norm(),
            1e-15);
  EXPECT_NEAR(model.probes[0].position.z(), .1, 1e-15);
  auto solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
#ifdef VIBESTRAN_HAS_HDF5
  ASSERT_FALSE(
      write_results(model, std::get<Results>(solved), directory / "frames.F06")
          .has_value());
  const auto file = directory / "frames.mag.vtkhdf";
  const auto cd = h5_table(file, "/MAGSTAT/probes/cd");
  const auto basic = h5_table(file, "/MAGSTAT/subcases/1/probes/B_T");
  const auto local = h5_table(file, "/MAGSTAT/subcases/1/probes/B_local_T");
  ASSERT_EQ(basic.dims, (std::vector<hsize_t>{2, 3}));
  ASSERT_EQ(local.dims, basic.dims);
  const double c = std::cos(theta * degree), s = std::sin(theta * degree);
  // Probe 1 reports radial, tangential and axial components at theta.
  EXPECT_EQ(cd.values, (std::vector<double>{5, 0}));
  EXPECT_NEAR(local.at(0, 0), .1 * c + .2 * s, 1e-12);
  EXPECT_NEAR(local.at(0, 1), -.1 * s + .2 * c, 1e-12);
  EXPECT_NEAR(local.at(0, 2), .3, 1e-12);
  for (size_t k = 0; k < 3; ++k) {
    EXPECT_NEAR(basic.at(0, k), basic.at(1, k), 1e-12);
    EXPECT_NEAR(local.at(1, k), basic.at(1, k), 1e-15);
  }
#endif
  std::filesystem::remove_all(directory);
}

TEST(MagneticParser, CutForceAndProbeCardsParseAndValidate) {
  const std::string head = "SOL MAGSTAT\nCEND\nBEGIN BULK\nPSOLID,1,1\n"
                           "MATMAG,1,1\nCORD2R,9,,0,0,0,0,1,0\n,1,0,0\n"
                           "ESET,1,PROP,1\nFSET,2,1,1,2,3\n" +
                           mesh_cards(box(1));
  auto parse = [&](const std::string &cards) {
    return parse_string(head + cards + "ENDDATA\n");
  };
  auto parsed = parse("COIL,5,1,12.,CUT,9\nMFORCE,1,BODY,1,9,2\n"
                      "MFORCE,2,FACES,2\nMPROBE,3,9,.1,.2,.3,9\n");
  ASSERT_TRUE(std::holds_alternative<Model>(parsed)) << error_of(parsed);
  const auto &m = std::get<Model>(parsed);
  ASSERT_EQ(m.coils.size(), 1U);
  EXPECT_TRUE(m.coils[0].plane_cut);
  EXPECT_EQ(m.coils[0].cut_coordinates, 9);
  EXPECT_EQ(m.coils[0].start, 0);
  ASSERT_EQ(m.forces.size(), 2U);
  EXPECT_TRUE(m.forces[0].body);
  EXPECT_EQ(m.forces[0].set, 1);
  EXPECT_EQ(m.forces[0].coordinates, 9);
  EXPECT_EQ(m.forces[0].layers, 2);
  EXPECT_FALSE(m.forces[1].body);
  EXPECT_EQ(m.forces[1].layers, 1);
  // CP 9 has local z = basic y and x = basic x, so local y = basic -z and
  // (.1, .2, .3) is basic (.1, .3, -.2).
  EXPECT_LT((m.probes[0].position - Vector(.1, .3, -.2)).norm(), 1e-15);
  EXPECT_EQ(m.probes[0].output_coordinates, 9);
  for (const auto &[cards, message] :
       std::vector<std::pair<std::string, std::string>>{
           {"MFORCE,1,SURFACE,2\n", "ID,FACES,FSET,CID"},
           {"MFORCE,1,BODY,7\n", "undefined ESET 7"},
           {"MFORCE,1,FACES,7\n", "undefined FSET 7"},
           {"MFORCE,1,BODY,1,0,0\n", "LAYER must be"},
           {"COIL,5,1,12.,CUT,4\n", "CUT requires"},
           {"COIL,5,1,12.,CUT,9\nCORD2C,4,,0,0,0,0,0,1\n,1,0,0\n"
            "COIL,6,1,1.,CUT,4\n",
            "CUT requires"}}) {
    SCOPED_TRACE(cards);
    EXPECT_NE(error_of(parse(cards)).find(message), std::string::npos)
        << error_of(parse(cards));
  }
}

TEST(MagneticParser, FieldOutputRejectsUndefinedElementsBeforeSolving) {
  const auto m = planar_box(4);
  const std::string bulk =
      "BEGIN BULK\nMAG2D,0,.1\nPMAG2D,1,1\nMATMAG,1,1\n" + mesh_cards(m) +
      "ESET,99,1,9999\nENDDATA\n";
  // Selecting a set with a bad element must fail during parsing, whether
  // inherited globally or selected by a subcase. An unused set is harmless.
  for (const std::string control : {"BFIELD=99\n", "BFIELD=99\nSUBCASE 2\n",
                                    "BFIELD=NONE\nSUBCASE 2\nBFIELD=99\n"}) {
    const auto parsed = parse_string("SOL MAGSTAT\nCEND\n" + control + bulk);
    ASSERT_TRUE(std::holds_alternative<Error>(parsed));
    EXPECT_NE(error_of(parsed).find("BFIELD ESET 99"), std::string::npos);
    EXPECT_NE(error_of(parsed).find("undefined field element 9999"),
              std::string::npos);
  }
  const auto unused = parse_string("SOL MAGSTAT\nCEND\nBFIELD=NONE\n" + bulk);
  EXPECT_TRUE(std::holds_alternative<Model>(unused)) << error_of(unused);
  // Programmatic models use the same validation; CWIRE IDs are not field
  // cell IDs and cannot appear in a field-output selection either.
  auto solid = brick_box(4);
  loop(solid);
  solid.element_sets[99] = {1, 100000};
  solid.subcases[0].field = {true, 99};
  const auto rejected = solve(solid);
  ASSERT_TRUE(std::holds_alternative<Error>(rejected));
  EXPECT_NE(error_of(rejected).find("undefined field element 100000"),
            std::string::npos);
}

TEST(MagneticIO, OutputRequestsSelectFieldSourceAndProbeFiles) {
  const auto directory =
      std::filesystem::temp_directory_path() / "vibestran_magnetic_requests";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  auto mesh = planar_box(4);
  mesh.options.planar = true;
  const auto path = directory / "requests.bdf";
  auto write = [&](const std::string &case_control) {
    std::ofstream out(path);
    out << "SOL MAGSTAT\nCEND\n"
        << case_control << "BEGIN BULK\nMAG2D,0,.1\nPMAG2D,1,1\nMATMAG,1,1\n"
        << mesh_cards(mesh)
        << "ESET,1,6\nESET,2,7\nESET,3,1,THRU,4\nCOIL2D,11,1,10\n"
           "COIL2D,12,2,-10\nCIRCUIT,1,,11,1,12,1\nICIRC,1,1,2.\n"
           "MPROBE,1,0,.1,.1,0\nENDDATA\n";
  };
  auto run = [&]() -> Status {
    auto parsed = parse_file(path);
    if (auto *e = std::get_if<Error>(&parsed))
      return *e;
    auto solved = solve(std::get<Model>(parsed));
    if (auto *e = std::get_if<Error>(&solved))
      return *e;
    return write_results(std::get<Model>(parsed), std::get<Results>(solved),
                         directory / "requests.F06");
  };
#ifdef VIBESTRAN_HAS_HDF5
  const auto file = directory / "requests.mag.vtkhdf";
  // Element ID -> the coordinates of its cell's points, read through the
  // block's own connectivity.
  auto geometry = [&](const std::string &block) {
    const auto root = "/VTKHDF/" + block;
    const auto ids = h5_table(file, root + "/CellData/element_id");
    const auto points = h5_table(file, root + "/Points");
    const auto connectivity = h5_table(file, root + "/Connectivity");
    const auto offsets = h5_table(file, root + "/Offsets");
    std::map<int, std::vector<double>> cells;
    for (size_t c = 0; c + 1 < offsets.values.size(); ++c)
      for (auto k = static_cast<size_t>(offsets.values[c]);
           k < static_cast<size_t>(offsets.values[c + 1]); ++k)
        for (size_t j = 0; j < 3; ++j)
          cells[static_cast<int>(ids.values[c])].push_back(
              points.at(static_cast<size_t>(connectivity.values[k]), j));
    return cells;
  };
  // Requests are inherited by subcases unless overridden.
  write("LOAD=1\nMSOURCE=NONE\nBFIELD=NONE\nSUBCASE 1\nSUBCASE 2\n"
        "  BFIELD=3\n  MPROBE=NONE\nSUBCASE 3\n  BFIELD=ALL\n");
  ASSERT_FALSE(run().has_value()) << run()->message;
  EXPECT_FALSE(h5_exists(file, "/VTKHDF/SUBCASE_1"));
  EXPECT_FALSE(h5_exists(file, "/VTKHDF/COIL_11"));
  EXPECT_FALSE(h5_exists(file, "/VTKHDF/Assembly/sources"));
  const auto subset = geometry("SUBCASE_2"), whole = geometry("SUBCASE_3");
  EXPECT_EQ(whole.size(), 16U);
  ASSERT_EQ(subset.size(), 4U);
  // The ESET block keeps its own compacted points, yet every cell has the
  // same corner coordinates as in the full-mesh block.
  for (const auto &[id, corners] : subset) {
    EXPECT_TRUE(id >= 1 && id <= 4) << id;
    EXPECT_EQ(corners, whole.at(id)) << id;
  }
  const auto subset_points = h5_table(file, "/VTKHDF/SUBCASE_2/Points");
  EXPECT_EQ(subset_points.dims[0],
            h5_table(file, "/VTKHDF/SUBCASE_2/NumberOfPoints").at(0));
  EXPECT_LT(subset_points.dims[0],
            h5_table(file, "/VTKHDF/SUBCASE_3/Points").dims[0]);
  EXPECT_EQ(h5_attribute(file, "/VTKHDF/SUBCASE_2", "Index"), 0);
  EXPECT_EQ(h5_attribute(file, "/VTKHDF/SUBCASE_3", "Index"), 1);
  EXPECT_TRUE(h5_exists(file, "/MAGSTAT/subcases/1/probes/B_T"));
  EXPECT_FALSE(h5_exists(file, "/MAGSTAT/subcases/2/probes"));
  EXPECT_TRUE(h5_exists(file, "/MAGSTAT/subcases/3/probes/B_T"));
  // Scalar results are written whatever the field requests.
  EXPECT_TRUE(std::isfinite(h5_attribute(file, "/MAGSTAT/subcases/2",
                                         "energy_J")));
  // The defaults write every field and source block, indexed in file order
  // and reachable through the assembly.
  write("LOAD=1\n");
  ASSERT_FALSE(run().has_value()) << run()->message;
  EXPECT_EQ(geometry("SUBCASE_1").size(), 16U);
  EXPECT_EQ(h5_attribute(file, "/VTKHDF/COIL_11", "Index"), 1);
  EXPECT_EQ(h5_attribute(file, "/VTKHDF/COIL_12", "Index"), 2);
  EXPECT_TRUE(h5_exists(file, "/VTKHDF/Assembly/fields/SUBCASE_1"));
  EXPECT_TRUE(h5_exists(file, "/VTKHDF/Assembly/sources/COIL_12"));
  // An unwritable result path is an error value naming the file and step.
  std::filesystem::remove(file);
  std::filesystem::create_directory(file);
  const auto blocked = run();
  ASSERT_TRUE(blocked.has_value());
  EXPECT_NE(blocked->message.find("HDF5 output " + file.string() +
                                  ": cannot create the file"),
            std::string::npos)
      << blocked->message;
#endif
  write("LOAD=1\nSUBCASE 1\n  MSOURCE=NONE\n");
  EXPECT_NE(run()->message.find("MSOURCE is a global request"),
            std::string::npos);
  write("LOAD=1\nBFIELD=9\n");
  EXPECT_NE(run()->message.find("undefined ESET 9"), std::string::npos);
  write("LOAD=1\nMPROBE=SOME\n");
  EXPECT_NE(run()->message.find("ALL or NONE"), std::string::npos);
  std::filesystem::remove_all(directory);
}

// ── Solver backend use ───────────────────────────────────────────────────────

namespace {
using vibestran::FactorOptions;
using vibestran::FactorStats;
using vibestran::LinearFactor;
using vibestran::SolverBackend;
using CsrData = vibestran::SparseMatrixBuilder::CsrData;

// What MAGSTAT asked of its backend.
struct BackendLog {
  int factors{0};
  bool require_spd{false};
  std::vector<int> nrhs; // per solve() call
  FactorStats stats;     // of the last factor, when it was destroyed
};

class RecordingFactor final : public LinearFactor {
public:
  RecordingFactor(std::unique_ptr<LinearFactor> inner, BackendLog &log)
      : inner_(std::move(inner)), log_(log) {}
  ~RecordingFactor() override { log_.stats = inner_->stats(); }
  int size() const noexcept override { return inner_->size(); }
  vibestran::Status refactor(const CsrData &K) override {
    return inner_->refactor(K);
  }
  vibestran::Result<std::vector<double>> solve(std::span<const double> B,
                                               int nrhs) override {
    log_.nrhs.push_back(nrhs);
    return inner_->solve(B, nrhs);
  }
  std::string_view method() const noexcept override { return inner_->method(); }
  FactorStats stats() const noexcept override { return inner_->stats(); }

private:
  std::unique_ptr<LinearFactor> inner_;
  BackendLog &log_;
};

// The CPU direct backend, recording each factor() and solve().
class RecordingBackend final : public SolverBackend {
public:
  std::vector<double> solve(const CsrData &K, const std::vector<double> &F,
                            const vibestran::FactorRatioCheckPolicy *p) override {
    return inner_.solve(K, F, p);
  }
  std::string_view name() const noexcept override { return "recording"; }
  std::vector<std::pair<std::string, std::string>> details() const override {
    return {{"Method", "recorded Cholesky"}, {"Indefinite fallback", "LDL^T"}};
  }
  vibestran::Result<std::unique_ptr<LinearFactor>>
  factor(const CsrData &K, const FactorOptions &options) override {
    ++log.factors;
    log.require_spd = options.require_spd;
    if (!notice.empty())
      notices().add(notice);
    auto made = inner_.factor(K, options);
    if (auto *e = std::get_if<Error>(&made))
      return *e;
    return std::unique_ptr<LinearFactor>(std::make_unique<RecordingFactor>(
        std::get<std::unique_ptr<LinearFactor>>(std::move(made)), log));
  }
  BackendLog log;
  std::string notice; // added to notices() by each factor() when set

private:
  vibestran::EigenSolverBackend inner_;
};

// Declares an LU fallback, as cuDSS does, so it cannot promise SPD.
class FallbackBackend final : public SolverBackend {
public:
  std::vector<double> solve(const CsrData &K, const std::vector<double> &F,
                            const vibestran::FactorRatioCheckPolicy *p) override {
    return inner_.solve(K, F, p);
  }
  std::string_view name() const noexcept override { return "lu-fallback"; }
  bool may_factor_indefinite() const noexcept override { return true; }

private:
  vibestran::EigenSolverBackend inner_;
};
} // namespace

TEST(MagneticBackend, OneAnalysisServesNewtonSubcasesAndInductances) {
  // Saturating steel: several Newton iterations in each of four subcases,
  // and four closed planar coils in two closed circuits.
  auto model = nonlinear_planar(0);
  nonlinear_loads(model, {4000, 2500});
  RecordingBackend backend;
  const auto solved = solve(model, backend);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &result = std::get<Results>(solved);
  const auto &log = backend.log;

  // One factor for the whole run, always positive definite, one symbolic
  // analysis: every tangent has the gauge's pattern.
  EXPECT_EQ(log.factors, 1);
  EXPECT_TRUE(log.require_spd);
  EXPECT_EQ(log.stats.symbolic_analyses, 1);

  // A numeric factorization per Newton step plus one of the converged
  // tangent per subcase for the inductances: nothing is factored twice.
  int newton = 0;
  for (const auto &c : result.cases)
    newton += c.iterations;
  const int subcases = static_cast<int>(result.cases.size());
  ASSERT_GE(subcases, 4);
  EXPECT_GT(newton, 2 * subcases);
  EXPECT_EQ(log.stats.numeric_factorizations, newton + subcases);

  // Newton solves one right-hand side; each subcase's inductances are one
  // block solve over the closed sources and circuits (here the two circuits;
  // the planar coils are open on their own).
  int block = 0;
  for (const auto &source : result.sources)
    block += source.closed ? 1 : 0;
  for (const auto &circuit : result.circuits)
    block += circuit.closed ? 1 : 0;
  ASSERT_EQ(block, 2);
  EXPECT_EQ(std::count(log.nrhs.begin(), log.nrhs.end(), block), subcases);
  EXPECT_EQ(std::count(log.nrhs.begin(), log.nrhs.end(), 1), newton);
  EXPECT_EQ(log.nrhs.size(), static_cast<size_t>(newton + subcases));
  EXPECT_EQ(log.stats.solves, newton + block * subcases);

  // Each column of the block lands in its own entry: the circuit
  // inductance is reciprocal and positive definite.
  for (const auto &c : result.cases) {
    const auto &L = c.circuit_inductance;
    EXPECT_GT(L(0, 0), 0);
    EXPECT_GT(L(1, 1), 0);
    const double self = std::sqrt(L(0, 0) * L(1, 1));
    EXPECT_NEAR(L(0, 1), L(1, 0), 1e-12 * self);
    EXPECT_LT(std::abs(L(0, 1)), self);
  }
}

TEST(MagneticBackend, SummaryReportsTangentWorkNewtonStepsAndBackendNotices) {
  auto model = nonlinear_planar(0);
  nonlinear_loads(model, {4000, 2500});
  RecordingBackend backend;
  backend.notice = "recording backend notice";
  const auto solved = solve(model, backend);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &result = std::get<Results>(solved);
  const auto &stats = backend.log.stats;
  auto row = [](const vibestran::F06Section &section, const std::string &name) {
    std::vector<std::string> values;
    for (const auto &[row_name, value] : section.rows)
      if (row_name == name)
        values.push_back(value);
    return values;
  };
  using Values = std::vector<std::string>;

  // The work reported is the work the backend's factor recorded.
  const auto &solver = result.summary.solver;
  EXPECT_EQ(row(solver, "Backend"), Values{"recording"});
  EXPECT_EQ(row(solver, "Method"), Values{"recorded Cholesky"});
  EXPECT_EQ(row(solver, "Tangent reuse"),
            Values{std::format("{} symbolic analysis, {} numeric factorizations, {} "
                               "right-hand sides",
                               stats.symbolic_analyses, stats.numeric_factorizations,
                               stats.solves)});
  EXPECT_EQ(stats.symbolic_analyses, 1);
  // MAGSTAT never takes the backend's indefinite fallback; the row says so
  // in place of the backend's own description.
  EXPECT_EQ(row(solver, "Indefinite fallback"),
            Values{"none, positive definite required"});

  // One row per subcase with its Newton iterations and residual.
  const auto &solution = result.summary.solution;
  EXPECT_EQ(row(solution, "Subcases"), Values{std::to_string(result.cases.size())});
  EXPECT_EQ(row(solution, "Formulation").size(), 1U);
  EXPECT_EQ(row(solution, "Formulation")[0].rfind("planar", 0), 0U);
  for (const auto &c : result.cases)
    EXPECT_EQ(row(solution, std::format("Subcase {}", c.id)),
              Values{std::format("{} Newton iteration{}, relative residual {:.3e}",
                                 c.iterations, c.iterations == 1 ? "" : "s", c.residual)});

  // The factor is made once, during the first subcase, so its notice is
  // reported once, under that subcase. B-H extrapolation is reported for
  // exactly the subcases that used it; the heaviest load here does.
  Values expected{std::format("Subcase {}: recording backend notice", result.cases[0].id)};
  bool extrapolated = false;
  for (const auto &c : result.cases)
    if (c.extrapolated) {
      extrapolated = true;
      expected.push_back(std::format(
          "Subcase {}: The B-H curve was exceeded; its vacuum-slope extrapolation was used",
          c.id));
    }
  EXPECT_TRUE(extrapolated);
  EXPECT_EQ(result.summary.notices, expected);
}

TEST(MagneticBackend, IterativeBackendThroughTheAdapterMatchesDirect) {
  // Linear model, three subcases: the PCG backend has no reusable
  // factorization, so every tangent solve is a full PCG solve.
  auto model = box(4, 1.5);
  loop(model);
  for (int i = 0; i < 4; ++i) {
    model.nodes[200000 + i] = {
        model.nodes[100000 + i].position + Vector(0, 0, 0.5), 0};
    model.wires.push_back(
        {200000 + i, 1, 200000 + i, 200000 + (i + 1) % 4, 0, {}});
  }
  model.wire_paths.push_back({200000, 200000});
  model.currents.push_back({2, 200000, 2});
  model.combinations[3] = {1, {{1, 1}, {1, 2}}};
  model.subcases = {{.id = 1, .load = 1}, {.id = 2, .load = 2}, {.id = 3, .load = 3}};

  const auto direct = solve(model);
  vibestran::EigenPCGSolverBackend pcg(1e-10);
  const auto iterative = solve(model, pcg);
  ASSERT_TRUE(std::holds_alternative<Results>(direct)) << error_of(direct);
  ASSERT_TRUE(std::holds_alternative<Results>(iterative)) << error_of(iterative);
  const auto &d = std::get<Results>(direct).cases;
  const auto &p = std::get<Results>(iterative).cases;
  ASSERT_EQ(d.size(), p.size());
  for (size_t k = 0; k < d.size(); ++k) {
    EXPECT_NEAR(p[k].energy, d[k].energy, 1e-8 * std::abs(d[k].energy));
    EXPECT_LT((p[k].differential_inductance - d[k].differential_inductance).norm(),
              1e-8 * d[k].differential_inductance.norm());
    EXPECT_LT((p[k].potential - d[k].potential).norm(), 1e-8 * d[k].potential.norm());
  }
}

TEST(MagneticBackend, BackendThatCannotPromiseSpdIsRefused) {
  auto model = box(3);
  loop(model);
  FallbackBackend backend;
  const auto solved = solve(model, backend);
  ASSERT_TRUE(std::holds_alternative<Error>(solved));
  const std::string message = error_of(solved);
  EXPECT_NE(message.find("lu-fallback"), std::string::npos) << message;
  EXPECT_NE(message.find("positive definite"), std::string::npos) << message;
}
