#include "magnetic/solver.hpp"
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <queue>
#include <set>

namespace vibestran::magnetic {
namespace {
using Triplet = Eigen::Triplet<double>;
using Sparse = Eigen::SparseMatrix<double>;
bool air(const Material &m) {
  return m.mur == 1 && m.bh_table == 0 && m.remanence.isZero(0);
}
Vector current_density(const Mesh &mesh, const Cell &c,
                       const Eigen::VectorXd &flux, const Vector &x) {
  Vector j = Vector::Zero();
  const auto basis = flux_basis(mesh, c, x);
  for (int i = 0; i < 4; ++i)
    j += flux[c.faces[i]] * basis[i];
  return j;
}
double flux_error(const Mesh &mesh, const Eigen::VectorXd &flux) {
  double error = 0;
  for (const auto &c : mesh.cells) {
    double sum = 0;
    for (int i = 0; i < 4; ++i)
      sum += c.face_sign[i] * flux[c.faces[i]];
    error = std::max(error, std::abs(sum));
  }
  return error / std::max(flux.cwiseAbs().maxCoeff(), 1e-300);
}
Result<Source> coil_source(const Model &model, const Mesh &mesh,
                           const Coil &coil) {
  Source source;
  source.id = coil.id;
  source.closed = coil.end == 0;
  source.turns = coil.turns;
  std::map<int, int> index;
  std::vector<Vector> nodes;
  std::vector<std::array<int, 4>> cells;
  const auto &ids = model.element_sets.at(coil.elements);
  for (int id : ids) {
    const auto &cell = mesh.cells[mesh.element_index.at(id)];
    if (!air(model.materials.at(cell.material)))
      return Error{"COIL volume must have nonmagnetic MATMAG (mu_r=1)"};
    std::array<int, 4> n{};
    for (int i = 0; i < 4; ++i) {
      auto [it, added] = index.emplace(mesh.node_ids[cell.nodes[i]],
                                       static_cast<int>(nodes.size()));
      if (added)
        nodes.push_back(mesh.nodes[cell.nodes[i]]);
      n[i] = it->second;
    }
    cells.push_back(n);
  }
  auto built = make_mesh(std::move(nodes), cells);
  if (auto *e = std::get_if<Error>(&built))
    return *e;
  source.mesh = std::get<Mesh>(std::move(built));
  const auto characteristic = static_cast<long>(source.mesh.nodes.size()) -
                              static_cast<long>(source.mesh.edges.size()) +
                              static_cast<long>(source.mesh.faces.size()) -
                              static_cast<long>(source.mesh.cells.size());
  if (characteristic != (source.closed ? 0 : 1))
    return Error{"COIL must be a single solid loop or an open conductor "
                 "without handles"};
  source.mesh.node_index = index;
  for (size_t i = 0; i < ids.size(); ++i) {
    source.mesh.cells[i].id = ids[i];
    source.mesh.element_index[ids[i]] = static_cast<int>(i);
  }
  auto section = selected_faces(model, source.mesh, coil.start);
  if (auto *e = std::get_if<Error>(&section))
    return *e;
  const auto &start = std::get<std::vector<std::pair<int, int>>>(section);
  std::vector<int> terminals;
  double area = 0;
  for (auto [id, sign] : start) {
    const auto &f = source.mesh.faces[id];
    area += f.area;
    if (source.closed && f.cells[1] < 0)
      return Error{"Closed COIL cut must consist of internal faces"};
    if (!source.closed) {
      if (f.cells[1] >= 0)
        return Error{"Open COIL terminal must be on its volume boundary"};
      const auto &c = source.mesh.cells[f.cells[0]];
      const int local = static_cast<int>(
          std::find(c.faces.begin(), c.faces.end(), id) - c.faces.begin());
      if (sign == c.face_sign[local])
        return Error{"COIL start FSET normals must point into the conductor"};
      terminals.push_back(id);
    }
  }
  if (!source.closed) {
    auto ending = selected_faces(model, source.mesh, coil.end);
    if (auto *e = std::get_if<Error>(&ending))
      return *e;
    for (auto [id, sign] : std::get<std::vector<std::pair<int, int>>>(ending)) {
      const auto &f = source.mesh.faces[id];
      if (f.cells[1] >= 0)
        return Error{"Open COIL end must be on its volume boundary"};
      const auto &c = source.mesh.cells[f.cells[0]];
      const int local = static_cast<int>(
          std::find(c.faces.begin(), c.faces.end(), id) - c.faces.begin());
      if (sign != c.face_sign[local])
        return Error{"COIL end FSET normals must point out of the conductor"};
      if (std::find(terminals.begin(), terminals.end(), id) != terminals.end())
        return Error{"COIL start and end overlap"};
      terminals.push_back(id);
    }
    for (int selection : {coil.start, coil.end}) {
      auto global = selected_faces(model, mesh, selection);
      if (auto *e = std::get_if<Error>(&global))
        return *e;
      for (auto [id, sign] :
           std::get<std::vector<std::pair<int, int>>>(global)) {
        (void)sign;
        if (mesh.faces[id].cells[1] >= 0)
          return Error{
              "Open COIL terminals must lie on the magnetic domain exterior"};
      }
    }
  }
  std::vector<double> rho(source.mesh.cells.size(), 1);
  auto flux = conserved_flux(source.mesh, start, terminals, true, area, rho);
  if (auto *e = std::get_if<Error>(&flux))
    return Error{std::format("COIL {}: {}", coil.id, e->message)};
  source.flux = std::get<Eigen::VectorXd>(std::move(flux)) * coil.turns;
  source.conservation_error = flux_error(source.mesh, source.flux);
  return source;
}
} // namespace

Result<Eigen::VectorXd>
conserved_flux(const Mesh &mesh, std::span<const std::pair<int, int>> section,
               std::span<const int> terminal_faces, bool stranded,
               double section_area, std::span<const double> resistivity) {
  const int nc = static_cast<int>(mesh.cells.size()),
            nf = static_cast<int>(mesh.faces.size());
  if (nc == 0 || section.empty() || !(section_area > 0) ||
      resistivity.size() != mesh.cells.size())
    return Error{"Invalid current preprocessing domain or section"};
  std::vector<bool> terminal(nf, false), seen(nc, false);
  std::queue<int> queue;
  queue.push(0);
  seen[0] = true;
  for (int i : terminal_faces) {
    if (i < 0 || i >= nf || mesh.faces[i].cells[1] >= 0)
      return Error{"Invalid current terminal face"};
    terminal[i] = true;
  }
  while (!queue.empty()) {
    int c = queue.front();
    queue.pop();
    for (int f : mesh.cells[c].faces)
      for (int neighbor : mesh.faces[f].cells)
        if (neighbor >= 0 && !seen[neighbor]) {
          seen[neighbor] = true;
          queue.push(neighbor);
        }
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    return Error{"A current source must be connected through faces"};
  std::vector<int> equation(nf, -1);
  int nq = 0;
  for (int f = 0; f < nf; ++f)
    if (mesh.faces[f].cells[1] >= 0 || terminal[f])
      equation[f] = nq++;
  if (nq == 0)
    return Error{"Current domain has no conducting faces"};
  // For a sealed conductor the sum of all cell balances is redundant.
  const int balances = terminal_faces.empty() ? nc - 1 : nc,
            total = nq + balances + 1;
  std::vector<Triplet> mass, entries;
  double trace = 0;
  for (int ci = 0; ci < nc; ++ci) {
    const auto &c = mesh.cells[ci];
    if (!(resistivity[ci] > 0))
      return Error{"Current resistivity must be positive"};
    Eigen::Matrix4d local = Eigen::Matrix4d::Zero();
    for (const auto &s : quadrature(mesh, c)) {
      auto basis = flux_basis(mesh, c, s.position);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          local(i, j) += s.weight * resistivity[ci] * basis[i].dot(basis[j]);
    }
    for (int i = 0; i < 4; ++i) {
      int row = equation[c.faces[i]];
      if (row < 0)
        continue;
      trace += local(i, i);
      for (int j = 0; j < 4; ++j)
        if (equation[c.faces[j]] >= 0)
          mass.emplace_back(row, equation[c.faces[j]], local(i, j));
      if (ci < balances) {
        entries.emplace_back(row, nq + ci, c.face_sign[i]);
        entries.emplace_back(nq + ci, row, c.face_sign[i]);
      }
    }
  }
  const double scale = nq / trace;
  std::transform(mass.begin(), mass.end(), std::back_inserter(entries),
                 [scale](const Triplet &t) {
                   return Triplet(t.row(), t.col(), t.value() * scale);
                 });
  std::set<int> cut;
  for (auto [f, sign] : section) {
    if (f < 0 || f >= nf || equation[f] < 0 || std::abs(sign) != 1 ||
        !cut.insert(f).second)
      return Error{"Invalid or duplicate cut face"};
    entries.emplace_back(equation[f], total - 1, sign);
    entries.emplace_back(total - 1, equation[f], sign);
  }
  Sparse matrix(total, total);
  matrix.setFromTriplets(entries.begin(), entries.end());
  Eigen::SparseLU<Sparse> solver;
  solver.compute(matrix);
  if (solver.info() != Eigen::Success)
    return Error{
        "Current constraints are singular; check cut topology and terminals"};
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(total);
  rhs[total - 1] = 1;
  Eigen::VectorXd solution = solver.solve(rhs);
  if (solver.info() != Eigen::Success || !solution.allFinite())
    return Error{"Current preprocessing factorization failed"};
  Eigen::VectorXd flux = Eigen::VectorXd::Zero(nf);
  for (int f = 0; f < nf; ++f)
    if (equation[f] >= 0)
      flux[f] = solution[equation[f]];
  if (stranded) {
    // Project the desired constant-magnitude winding field into the
    // conservative RT0 space. Renormalizing element vectors directly would
    // break continuity.
    rhs.head(nq).setZero();
    for (int ci = 0; ci < nc; ++ci) {
      const auto &c = mesh.cells[ci];
      for (const auto &s : quadrature(mesh, c)) {
        Vector j = current_density(mesh, c, flux, s.position);
        if (j.norm() < 1e-12 / section_area)
          return Error{"Stranded coil has a stagnant region; split branched or "
                       "unsuitable geometry"};
        j.normalize();
        j /= section_area;
        auto basis = flux_basis(mesh, c, s.position);
        for (int i = 0; i < 4; ++i)
          if (equation[c.faces[i]] >= 0)
            rhs[equation[c.faces[i]]] +=
                scale * s.weight * resistivity[ci] * basis[i].dot(j);
      }
    }
    solution = solver.solve(rhs);
    if (solver.info() != Eigen::Success || !solution.allFinite())
      return Error{"Stranded coil projection failed"};
    for (int f = 0; f < nf; ++f)
      if (equation[f] >= 0)
        flux[f] = solution[equation[f]];
  }
  double total_current = 0;
  for (auto [f, sign] : section)
    total_current += sign * flux[f];
  if (flux_error(mesh, flux) > 1e-10 || std::abs(total_current - 1) > 1e-10 ||
      (matrix * solution - rhs).norm() > 1e-9 * std::max(rhs.norm(), 1.0))
    return Error{
        "Current preprocessing failed its conservation residual check"};
  return flux;
}

Result<Eigen::VectorXd> couple_source(const Source &source, const Mesh &mesh) {
  Eigen::VectorXd coupling = Eigen::VectorXd::Zero(mesh.edges.size());
  SpatialIndex index(mesh);
  if (!source.closed) {
    for (size_t fi = 0; fi < source.mesh.faces.size(); ++fi) {
      const auto &face = source.mesh.faces[fi];
      if (face.cells[1] >= 0 || std::abs(source.flux[fi]) < 1e-12)
        continue;
      std::vector<Vector> samples{face.center};
      std::transform(face.nodes.begin(), face.nodes.end(),
                     std::back_inserter(samples),
                     [&](int n) { return source.mesh.nodes[n]; });
      for (const auto &x : samples) {
        bool exterior = false;
        const double tolerance = 1e-9 * std::sqrt(face.area);
        for (int ci : index.query(x - Vector::Constant(tolerance),
                                  x + Vector::Constant(tolerance))) {
          const auto &cell = mesh.cells[ci];
          const auto l = barycentric(cell, x);
          if (*std::min_element(l.begin(), l.end()) < -1e-8)
            continue;
          for (int i = 0; i < 4; ++i)
            if (mesh.faces[cell.faces[i]].cells[1] < 0 && std::abs(l[i]) < 1e-8)
              exterior = true;
        }
        if (!exterior)
          return Error{"Open current path must terminate on the magnetic "
                       "domain exterior"};
      }
    }
  }
  for (const auto &cell : source.mesh.cells) {
    if (!source.wire && mesh.element_index.contains(cell.id)) {
      const auto &target = mesh.cells[mesh.element_index.at(cell.id)];
      for (const auto &sample : quadrature(source.mesh, cell)) {
        const Vector j =
            current_density(source.mesh, cell, source.flux, sample.position);
        const auto basis = edge_basis(target, sample.position);
        for (int e = 0; e < 6; ++e)
          coupling[target.edges[e]] += sample.weight * basis[e].dot(j);
      }
      continue;
    }
    double volume = 0;
    for (int ci : index.query(cell.minimum, cell.maximum)) {
      const auto &target = mesh.cells[ci];
      for (const auto &sample :
           intersection_quadrature(source.mesh, cell, mesh, target)) {
        volume += sample.weight;
        const Vector j =
            current_density(source.mesh, cell, source.flux, sample.position);
        const auto basis = edge_basis(target, sample.position);
        for (int e = 0; e < 6; ++e)
          coupling[target.edges[e]] += sample.weight * basis[e].dot(j);
      }
    }
    if (std::abs(volume / cell.volume - 1) > 1e-8)
      return Error{std::format("Source {} cell {} is outside or overlaps the "
                               "magnetic domain (covered fraction {})",
                               source.id, cell.id, volume / cell.volume)};
  }
  std::vector<bool> boundary(mesh.nodes.size(), false);
  for (const auto &f : mesh.faces)
    if (f.cells[1] < 0)
      for (int n : f.nodes)
        boundary[n] = true;
  Eigen::VectorXd balance = Eigen::VectorXd::Zero(mesh.nodes.size());
  for (size_t i = 0; i < mesh.edges.size(); ++i) {
    balance[mesh.edges[i][0]] -= coupling[i];
    balance[mesh.edges[i][1]] += coupling[i];
  }
  for (size_t i = 0; i < boundary.size(); ++i)
    if (boundary[i])
      balance[i] = 0;
  if (balance.norm() > 1e-9 * std::max(coupling.norm(), 1e-300))
    return Error{"Source transfer violates discrete current continuity; check "
                 "open endpoints and source intersections"};
  return coupling;
}

Result<std::vector<Source>> prepare_sources(const Model &model,
                                            const Mesh &mesh) {
  std::vector<Source> sources;
  std::set<int> occupied;
  for (const auto &coil : model.coils) {
    for (int id : model.element_sets.at(coil.elements))
      if (!occupied.insert(id).second)
        return Error{"COIL regions overlap"};
    auto result = coil_source(model, mesh, coil);
    if (auto *e = std::get_if<Error>(&result))
      return *e;
    sources.push_back(std::get<Source>(std::move(result)));
  }
  auto wires = prepare_wires(model);
  if (auto *e = std::get_if<Error>(&wires))
    return *e;
  SpatialIndex index(mesh);
  for (auto &wire : std::get<std::vector<Source>>(wires)) {
    for (const auto &cell : wire.mesh.cells)
      for (int ci : index.query(cell.minimum, cell.maximum)) {
        const auto &target = mesh.cells[ci];
        if (air(model.materials.at(target.material)) &&
            !occupied.contains(target.id))
          continue;
        const auto samples =
            intersection_quadrature(wire.mesh, cell, mesh, target);
        const double volume = std::accumulate(
            samples.begin(), samples.end(), 0.0,
            [](double sum, const Sample &s) { return sum + s.weight; });
        if (volume > 1e-10 * cell.volume)
          return Error{"CWIRE overlaps a magnetic material or volumetric COIL"};
      }
    sources.push_back(std::move(wire));
  }
  for (auto &source : sources) {
    auto coupled = couple_source(source, mesh);
    if (auto *e = std::get_if<Error>(&coupled))
      return *e;
    source.coupling = std::get<Eigen::VectorXd>(std::move(coupled));
    source.conservation_error = flux_error(source.mesh, source.flux);
    Eigen::VectorXd divergence = Eigen::VectorXd::Zero(mesh.nodes.size());
    for (size_t e = 0; e < mesh.edges.size(); ++e) {
      divergence[mesh.edges[e][0]] -= source.coupling[e];
      divergence[mesh.edges[e][1]] += source.coupling[e];
    }
    for (const auto &f : mesh.faces)
      if (f.cells[1] < 0)
        for (int n : f.nodes)
          divergence[n] = 0;
    source.transfer_error =
        divergence.norm() / std::max(source.coupling.norm(), 1e-300);
  }
  return sources;
}

Result<std::vector<double>>
source_currents(const Model &model, std::span<const Source> sources, int set) {
  auto factored = load_factors(model, set);
  if (auto *e = std::get_if<Error>(&factored))
    return *e;
  const auto &factors = std::get<std::map<int, double>>(factored);
  std::map<int, double> coils, wires;
  for (const auto &c : model.currents)
    if (auto it = factors.find(c.set); it != factors.end())
      for (int target : c.targets)
        (c.wire ? wires : coils)[target] += it->second * c.amperes;
  std::vector<double> out;
  for (const auto &s : sources) {
    double current = coils[s.id];
    if (s.wire) {
      current =
          wires[s.wire_elements.front().first] * s.wire_elements.front().second;
      for (auto [id, sign] : s.wire_elements)
        if (std::abs(sign * wires[id] - current) >
            1e-10 * std::max({std::abs(current), std::abs(wires[id]), 1e-30}))
          return Error{std::format(
              "CWIRE path {} has discontinuous signed current at element {}",
              s.id, id)};
    }
    if (!std::isfinite(current))
      return Error{"Current LOAD combination overflow"};
    out.push_back(current);
  }
  return out;
}
} // namespace vibestran::magnetic
