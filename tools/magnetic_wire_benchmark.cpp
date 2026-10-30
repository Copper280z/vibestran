// Optional CWIRE stage benchmark; no field solve or external reference tool.
#include "magnetic/solver.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <string_view>
using namespace vibestran::magnetic;
namespace {
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
} // namespace
using Clock = std::chrono::steady_clock;
struct Measurement {
  size_t cells{0}, nq{0}, system{0};
  double preparation{0}, transfer{0}, volume{0}, continuity{0};
  Vector moment{Vector::Zero()}, biot{Vector::Zero()};
  Eigen::VectorXd load;
};
Result<Measurement> measure(const Model &model, const Mesh &mesh, bool tet) {
  Measurement result;
  auto start = Clock::now();
  auto selected = model;
  selected.options.wire_mesh = tet ? WireMesh::Tet : WireMesh::Hex;
  auto built = prepare_wires(selected);
  result.preparation =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  if (auto *e = std::get_if<Error>(&built))
    return *e;
  const auto &source = std::get<std::vector<Source>>(built)[0];
  result.cells = source.mesh.cells.size();
  for (const auto &f : source.mesh.faces)
    if (f.cells[1] >= 0)
      ++result.nq;
  result.system = result.nq + result.cells;
  start = Clock::now();
  auto coupled = couple_source(source, mesh);
  result.transfer =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  if (auto *e = std::get_if<Error>(&coupled)) {
    std::cerr << "failure preparation_ms=" << result.preparation
              << " transfer_ms=" << result.transfer << " cells=" << result.cells
              << '\n';
    return *e;
  }
  result.load = std::get<Eigen::VectorXd>(std::move(coupled));
  for (const auto &c : source.mesh.cells) {
    result.volume += c.volume;
    for (const auto &q : quadrature(source.mesh, c))
      result.moment += .5 * q.weight *
                       q.position.cross(flux_value(c, source.flux, q.position));
  }
  result.biot = source_biot(source, Vector::Zero());
  Eigen::VectorXd balance = Eigen::VectorXd::Zero(mesh.nodes.size());
  for (size_t i = 0; i < mesh.edges.size(); ++i) {
    balance[mesh.edges[i][0]] -= result.load[i];
    balance[mesh.edges[i][1]] += result.load[i];
  }
  for (const auto &f : mesh.faces)
    if (f.cells[1] < 0)
      for (int n : f.nodes)
        balance[n] = 0;
  result.continuity = balance.norm() / result.load.norm();
  if (result.continuity > 1e-9 || !result.load.allFinite())
    return Error{"Benchmark current continuity failed"};
  return result;
}
int main(int argc, char **argv) {
  int repetitions = 3;
  if (argc > 1) {
    const std::string_view argument(argv[1]);
    const auto [end, error] = std::from_chars(
        argument.data(), argument.data() + argument.size(), repetitions);
    if (error != std::errc{} || end != argument.data() + argument.size() ||
        repetitions < 1) {
      std::cerr << "Usage: magnetic_wire_benchmark [positive repetitions] "
                   "[all|flat|case_name]\n";
      return 1;
    }
  }
  if (argc > 3)
    return 1;
  bool matched = false, failed = false;
  const std::string selected = argc > 2 ? argv[2] : "all";
  std::cout << "case,scheme,run,field_cells,field_dofs,source_cells,flux_"
               "unknowns,current_system,preparation_ms,transfer_ms,total_ms,"
               "volume,moment,biot,continuity,relative_coupling_difference,"
               "relative_biot_difference\n"
            << std::setprecision(12);
  for (const std::string name : {"round_tet", "rect_tet", "rect_hex",
                                 "rect_warped_hex", "rect_warped_wedge"}) {
    if (selected == "flat" ? name.find("warped") != std::string::npos
                           : selected != "all" && selected != name)
      continue;
    matched = true;
    const bool warped = name.find("warped") != std::string::npos;
    const bool wedge = name.find("wedge") != std::string::npos;
    auto model = name.find("_tet") != std::string::npos
                     ? box(6, 1.5)
                     : brick_box(warped ? 2 : 8, wedge);
    if (warped)
      for (auto &[id, g] : model.nodes) {
        (void)id;
        if (wedge)
          g.position.x() += .35 * g.position.y() * g.position.z();
        else
          g.position.z() += .35 * g.position.x() * g.position.y();
      }
    else if (name == "rect_hex")
      for (auto &[id, g] : model.nodes) {
        (void)id;
        g.position *= 1.5;
      }
    loop(model, .6, name != "round_tet");
    if (wedge)
      for (int i = 0; i < 4; ++i) {
        auto &x = model.nodes[100000 + i].position;
        x = Vector(x.z(), x.x(), x.y());
        model.wires[i].orientation = Vector::UnitX();
      }
    auto made = make_mesh(model);
    if (auto *e = std::get_if<Error>(&made)) {
      std::cerr << e->message << '\n';
      return 1;
    }
    const auto &mesh = std::get<Mesh>(made);
    auto gauged = make_gauge(mesh);
    if (auto *e = std::get_if<Error>(&gauged)) {
      std::cerr << e->message << '\n';
      return 1;
    }
    const auto &gauge = std::get<Gauge>(gauged);
    for (int run = 0; run < repetitions; ++run) {
      Measurement pair[2];
      bool passed[2]{true, true};
      for (int order = 0; order < 2; ++order) {
        const int kind = (order + run) % 2;
        auto measurement = measure(model, mesh, kind == 1);
        if (auto *e = std::get_if<Error>(&measurement)) {
          std::cerr << name << ' ' << (kind ? "tet" : "hex") << ": "
                    << e->message << '\n';
          passed[kind] = false;
          failed = true;
          continue;
        }
        pair[kind] = std::get<Measurement>(std::move(measurement));
      }
      const double coupling =
          passed[0] && passed[1]
              ? (pair[0].load - pair[1].load).norm() / pair[0].load.norm()
              : std::numeric_limits<double>::quiet_NaN();
      const double biot =
          passed[0] && passed[1]
              ? (pair[0].biot - pair[1].biot).norm() / pair[0].biot.norm()
              : std::numeric_limits<double>::quiet_NaN();
      for (int kind = 0; kind < 2; ++kind) {
        if (!passed[kind])
          continue;
        const auto &m = pair[kind];
        std::cout << name << ',' << (kind ? "tet" : "hex") << ',' << run << ','
                  << mesh.cells.size() << ',' << gauge.size << ',' << m.cells
                  << ',' << m.nq << ',' << m.system << ',' << m.preparation
                  << ',' << m.transfer << ',' << m.preparation + m.transfer
                  << ',' << m.volume << ',' << m.moment.norm() << ','
                  << m.biot.norm() << ',' << m.continuity << ',' << coupling
                  << ',' << biot << '\n';
      }
      std::cout.flush();
    }
  }
  if (!matched)
    std::cerr << "Unknown case: " << selected << '\n';
  return failed || !matched ? 1 : 0;
}
