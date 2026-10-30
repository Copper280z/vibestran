#include "magnetic/solver.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <numbers>
#include <sstream>

using namespace vibestran::magnetic;
namespace {
// These helpers only generate independent analytical verification fixtures.
Model box(int n, double extent = 1) {
  Model m;
  m.materials[1] = {};
  m.properties[1] = 1;
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
  for (int i = 0; i < 3; ++i)
    selection.nodes[i] = mesh.node_ids[f.nodes[i]];
  if (sign < 0)
    std::swap(selection.nodes[1], selection.nodes[2]);
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
  m.currents.push_back({1, false, 1, {1}});
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
  Current current{1, true, 1, {}};
  for (int i = 0; i < 4; ++i) {
    Wire w;
    w.id = start + i;
    w.property = 1;
    w.first = start + i;
    w.second = start + (i + 1) % 4;
    if (rectangular)
      w.orientation = Vector::UnitZ();
    m.wires.push_back(w);
    current.targets.push_back(w.id);
  }
  m.currents.push_back(current);
  m.subcases[0].load = 1;
  m.probes.push_back({1, Vector::Zero()});
}
std::string error_of(const auto &result) {
  const auto *e = std::get_if<Error>(&result);
  return e ? e->message : "";
}
Vector source_biot(const Source &source, const Vector &x) {
  Vector b = Vector::Zero();
  for (const auto &c : source.mesh.cells)
    for (const auto &s : quadrature(source.mesh, c)) {
      const auto basis = flux_basis(source.mesh, c, s.position);
      Vector j = Vector::Zero();
      for (int i = 0; i < 4; ++i)
        j += basis[i] * source.flux[c.faces[i]];
      Vector r = x - s.position;
      b += mu0 / (4 * std::numbers::pi) * s.weight * j.cross(r) /
           std::pow(r.norm(), 3);
    }
  return b;
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
  pm.remanence = Vector(0, 0, 1.2);
  auto response = material_response(pm, {}, Vector::Zero());
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
IWIRE,10,-2,2
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
      for (int i = 0; i < 4; ++i)
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
  model.currents.push_back({1, true, 0.2, {100001}});
  EXPECT_TRUE(
      std::holds_alternative<Error>(source_currents(model, sources, 1)));
  model.currents.pop_back();
  std::swap(model.wires[1].first, model.wires[1].second);
  model.currents.push_back({1, true, -2, {100001}});
  auto reversed = prepare_wires(model);
  ASSERT_TRUE(std::holds_alternative<std::vector<Source>>(reversed))
      << error_of(reversed);
  current = source_currents(model, std::get<std::vector<Source>>(reversed), 1);
  ASSERT_TRUE(std::holds_alternative<std::vector<double>>(current))
      << error_of(current);
  EXPECT_DOUBLE_EQ(std::get<std::vector<double>>(current)[0], 1);
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
  model.forces.push_back({1, 1, Vector::Zero()});
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
  model.currents.push_back({2, true, -1, {100000, 100001, 100002, 100003}});
  model.subcases.push_back({2, 2, 0, "reverse"});
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
  model.materials[2] = {1, 1, Vector::Zero(), 0};
  model.properties[2] = 2;
  for (auto &e : model.elements) {
    Vector center = Vector::Zero();
    for (int n : e.nodes)
      center += model.nodes.at(n).position / 4;
    if (center.cwiseAbs().maxCoeff() < 0.51)
      e.property = 2;
  }
  model.subcases = {{1, 1, 0, ""}, {2, 3, 0, ""}, {3, 4, 0, ""}};
  model.combinations[3] = {1, {{1.001, 1}}};
  model.combinations[4] = {1, {{0.999, 1}}};
  solved = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(solved)) << error_of(solved);
  const auto &nonlinear = std::get<Results>(solved).cases;
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
  model.materials[2] = {1.05, 0, Vector(0, 0, 1.2), 0};
  model.properties[2] = 2;
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
  model.materials[2].remanence = -model.materials[2].remanence;
  auto reverse = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(reverse)) << error_of(reverse);
  EXPECT_LT((std::get<Results>(solved).cases[0].probe_b[0] +
             std::get<Results>(reverse).cases[0].probe_b[0])
                .norm(),
            1e-12);
  model = box(6, 1.5);
  loop(model);
  model.materials[2] = {1, 1, Vector::Zero(), 0};
  model.properties[2] = 2;
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
  model.currents.push_back({2, true, 2, {200000, 200001, 200002, 200003}});
  model.combinations[3] = {1, {{1, 1}, {1, 2}}};
  model.subcases = {{1, 1, 0, ""}, {2, 2, 0, ""}, {3, 3, 0, ""}};
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

TEST(MagneticIO, IncludesFixedFieldsAndResultFieldValues) {
  const auto directory =
      std::filesystem::temp_directory_path() / "vibestran_magnetic_io";
  std::filesystem::create_directories(directory);
  const auto path = directory / "model.bdf";
  {
    std::ofstream out(path);
    out << "SOL MAGSTAT\nCEND\nMAGBC=1\nBEGIN BULK\nINCLUDE "
           "'volume.inc'\nMBC,1,0,UNIFORM,0,0,0.25\nMPROBE,1,0.1,0.1,0."
           "1\nENDDATA\n";
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
  EXPECT_FALSE(write_results(std::get<Model>(parsed), std::get<Results>(solved),
                             directory / "result.F06")
                   .has_value());
  std::ifstream probe(directory / "result.probes.csv");
  std::string line;
  std::getline(probe, line);
  std::getline(probe, line);
  EXPECT_NE(line.find(",0,0,0.25,"), std::string::npos) << line;
  std::ofstream cycle(directory / "cycle.inc");
  cycle << "INCLUDE 'model.bdf'\n";
  cycle.close();
  std::ofstream volume(directory / "volume.inc");
  volume << "INCLUDE 'cycle.inc'\n";
  volume.close();
  EXPECT_TRUE(std::holds_alternative<Error>(parse_file(path)));
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
    model.forces.push_back({1, 1, Vector::Zero()});
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
  model.materials[2].remanence = Vector(1.2, 0, 0);
  model.properties[2] = 2;
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
  model.materials[2].coordinates = 1;
  model.materials[2].remanence = Vector(0, 0, 1.2);
  auto local = solve(model);
  ASSERT_TRUE(std::holds_alternative<Results>(local)) << error_of(local);
  const auto &a = std::get<Results>(basic).cases[0];
  const auto &b = std::get<Results>(local).cases[0];
  EXPECT_GT(a.probe_b[0].x(), 0.2);
  EXPECT_NEAR(a.energy, b.energy, 1e-8);
  for (size_t i = 0; i < a.b.size(); ++i)
    EXPECT_LT((a.b[i] - b.b[i]).norm(), 1e-12);
}
