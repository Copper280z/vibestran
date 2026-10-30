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
  return m.mur == 1 && m.bh_table == 0 && m.br == 0;
}
double flux_error(const Mesh &mesh, const Eigen::VectorXd &flux) {
  double error = 0;
  for (const auto &c : mesh.cells) {
    double sum = 0;
    for (size_t i = 0; i < c.faces.size(); ++i)
      sum += c.face_sign[i] * flux[c.faces[i]];
    error = std::max(error, std::abs(sum));
  }
  return error / std::max(flux.cwiseAbs().maxCoeff(), 1e-300);
}
// The mesh of a winding region, built from its field cells. Element and GRID
// IDs are kept so FSET selections resolve on it.
Result<Mesh> region_mesh(const Mesh &mesh, std::span<const int> ids) {
  std::map<int, int> index;
  std::vector<Vector> nodes;
  std::vector<int> grid_ids;
  std::vector<NodeList> cells;
  for (int id : ids) {
    const auto &cell = mesh.cells[mesh.element_index.at(id)];
    NodeList n(cell.nodes.size());
    for (size_t i = 0; i < cell.nodes.size(); ++i) {
      auto [it, added] = index.emplace(mesh.node_ids[cell.nodes[i]],
                                       static_cast<int>(nodes.size()));
      if (added) {
        nodes.push_back(mesh.nodes[cell.nodes[i]]);
        grid_ids.push_back(it->first);
      }
      n[i] = it->second;
    }
    cells.push_back(n);
  }
  auto built = make_mesh(std::move(nodes), cells, mesh.planar, mesh.normal);
  if (auto *e = std::get_if<Error>(&built))
    return *e;
  auto out = std::get<Mesh>(std::move(built));
  out.depth = mesh.depth;
  out.node_index = std::move(index);
  out.node_ids = std::move(grid_ids);
  for (size_t i = 0; i < ids.size(); ++i) {
    out.cells[i].id = ids[i];
    out.element_index[ids[i]] = static_cast<int>(i);
  }
  return out;
}
Result<Source> coil_source(const Model &model, const Mesh &mesh,
                           const Coil &coil) {
  Source source;
  source.id = coil.id;
  source.closed = coil.end == 0;
  source.turns = coil.turns;
  const auto &ids = model.element_sets.at(coil.elements);
  for (int id : ids)
    if (!air(model.materials.at(mesh.cells[mesh.element_index.at(id)].material)))
      return Error{"COIL volume must have nonmagnetic MATMAG (mu_r=1)"};
  auto built = region_mesh(mesh, ids);
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
    Eigen::MatrixXd local =
        Eigen::MatrixXd::Zero(c.faces.size(), c.faces.size());
    for (const auto &s : quadrature(mesh, c)) {
      auto basis = flux_basis(mesh, c, s.position);
      for (size_t i = 0; i < c.faces.size(); ++i)
        for (size_t j = 0; j < c.faces.size(); ++j)
          local(i, j) += s.weight * resistivity[ci] * basis[i].dot(basis[j]);
    }
    for (size_t i = 0; i < c.faces.size(); ++i) {
      int row = equation[c.faces[i]];
      if (row < 0)
        continue;
      trace += local(i, i);
      for (size_t j = 0; j < c.faces.size(); ++j)
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
        Vector j = flux_value(c, flux, s.position);
        if (j.norm() < 1e-12 / section_area)
          return Error{"Stranded coil has a stagnant region; split branched or "
                       "unsuitable geometry"};
        j.normalize();
        j /= section_area;
        auto basis = flux_basis(mesh, c, s.position);
        for (size_t i = 0; i < c.faces.size(); ++i)
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

namespace {
struct Transfer {
  Eigen::VectorXd coupling;
  double error;
};
Result<Transfer> transfer_source(const Source &source, const Mesh &mesh,
                                 const SpatialIndex &index, int order) {
  Eigen::VectorXd coupling = Eigen::VectorXd::Zero(mesh.edges.size());
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
          if (!contains(cell, x, 1e-8))
            continue;
          for (int f : cell.faces)
            if (mesh.faces[f].cells[1] < 0 &&
                point_on_face(mesh, mesh.faces[f], x, 1e-8))
              exterior = true;
        }
        if (!exterior)
          return Error{"Open current path must terminate on the magnetic "
                       "domain exterior"};
      }
    }
  }
  for (const auto &cell : source.mesh.cells) {
    const CurrentField current(cell, source.flux);
    if (!source.wire && mesh.element_index.contains(cell.id)) {
      const auto &target = mesh.cells[mesh.element_index.at(cell.id)];
      for (const auto &sample : quadrature(source.mesh, cell)) {
        const Vector j = current.value(sample.position);
        const auto basis = edge_basis(target, sample.position);
        for (size_t e = 0; e < target.edges.size(); ++e)
          coupling[target.edges[e]] += sample.weight * basis[e].dot(j);
      }
      continue;
    }
    double volume = 0;
    for (int ci : index.query(cell.minimum, cell.maximum)) {
      const auto &target = mesh.cells[ci];
      Vector integral = Vector::Zero();
      Matrix moment = Matrix::Zero();
      for (const auto &sample : intersection_quadrature(
               source.mesh, cell, mesh, target,
               cell.affine && target.affine ? (target.nodes.size() == 4 ? 2 : 3)
                                            : order)) {
        volume += sample.weight;
        const Vector j = current.value(sample.position);
        if (target.nodes.size() == 4) {
          integral += sample.weight * j;
          moment +=
              sample.weight * (sample.position - target.origin) * j.transpose();
        } else {
          const auto basis = edge_basis(target, sample.position);
          for (size_t e = 0; e < target.edges.size(); ++e)
            coupling[target.edges[e]] += sample.weight * basis[e].dot(j);
        }
      }
      if (target.nodes.size() == 4) {
        // Tetrahedral test functions are affine. Integrate the zeroth and
        // first current moments once, then contract with all six test
        // functions.
        std::array<Vector, 4> weighted;
        for (int i = 0; i < 4; ++i)
          weighted[i] = moment.transpose() * target.gradient[i];
        weighted[0] += integral;
        for (int e = 0; e < 6; ++e) {
          const auto ij = local_edges[e];
          coupling[target.edges[e]] +=
              target.edge_sign[e] *
              (target.gradient[ij[1]].dot(weighted[ij[0]]) -
               target.gradient[ij[0]].dot(weighted[ij[1]]));
        }
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
  const double error = balance.norm() / std::max(coupling.norm(), 1e-300);
  return Transfer{std::move(coupling), error};
}
} // namespace
Result<Eigen::VectorXd> couple_source(const Source &source, const Mesh &mesh,
                                      const SpatialIndex &index) {
  if (source.wire &&
      (std::any_of(mesh.cells.begin(), mesh.cells.end(),
                   [](const Cell &c) { return !c.polyhedral; }) ||
       std::any_of(source.mesh.cells.begin(), source.mesh.cells.end(),
                   [](const Cell &c) { return !c.affine; })))
    return couple_curved_source(source, mesh, index);
  return couple_polyhedral_source(source, mesh, index);
}
Result<Eigen::VectorXd> couple_polyhedral_source(const Source &source,
                                                 const Mesh &mesh,
                                                 const SpatialIndex &index) {
  double error = 0, change = 0;
  const bool adaptive =
      source.wire &&
      (std::any_of(source.mesh.cells.begin(), source.mesh.cells.end(),
                   [](const Cell &c) { return !c.affine; }) ||
       std::any_of(mesh.cells.begin(), mesh.cells.end(),
                   [](const Cell &c) { return !c.affine; }));
  std::optional<Eigen::VectorXd> previous;
  for (int order : {6, 8, 10, 12, 14}) {
    auto result = transfer_source(source, mesh, index, order);
    if (auto *e = std::get_if<Error>(&result))
      return *e;
    auto &transfer = std::get<Transfer>(result);
    error = transfer.error;
    change = previous ? (transfer.coupling - *previous).norm() /
                            std::max(transfer.coupling.norm(), 1e-300)
                      : 1;
    if (error <= 1e-9 && (!adaptive || (previous && change <= 1e-8)))
      return std::move(transfer.coupling);
    previous = std::move(transfer.coupling);
  }
  return Error{std::format(
      "Source transfer did not converge (current continuity {}, quadrature "
      "change {}); refine distorted source/field elements",
      error, change)};
}

Result<std::vector<Source>> prepare_sources(const Model &model,
                                            const Mesh &mesh) {
  std::vector<Source> sources;
  std::set<int> occupied;
  if (mesh.planar) {
    for (const auto &region : model.planar_coils) {
      Source s;
      s.id = region.id;
      s.turns = region.turns;
      // A single signed region is unbalanced; its CIRCUIT decides closure.
      s.closed = false;
      s.coupling = Eigen::VectorXd::Zero(mesh.dofs());
      const auto &ids = model.element_sets.at(region.elements);
      double area = 0;
      for (int id : ids) {
        if (!occupied.insert(id).second)
          return Error{"COIL2D winding regions overlap"};
        const auto &c = mesh.cells[mesh.element_index.at(id)];
        if (!air(model.materials.at(c.material)))
          return Error{"COIL2D region requires nonmagnetic MATMAG"};
        area += c.volume;
      }
      if (!(area > 0))
        return Error{"Empty COIL2D region"};
      for (int id : ids) {
        const auto &c = mesh.cells[mesh.element_index.at(id)];
        for (const auto &q : quadrature(mesh, c)) {
          const auto basis = edge_basis(c, q.position);
          for (size_t i = 0; i < c.nodes.size(); ++i)
            s.coupling[c.nodes[i]] += mesh.depth * q.weight * region.turns /
                                      area * basis[i].dot(mesh.normal);
        }
      }
      auto built = region_mesh(mesh, ids);
      if (auto *e = std::get_if<Error>(&built))
        return *e;
      s.mesh = std::get<Mesh>(std::move(built));
      s.planar_density.assign(s.mesh.cells.size(),
                              region.turns / area * mesh.normal);
      sources.push_back(std::move(s));
    }
    return sources;
  }
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
  const bool curved = std::any_of(mesh.cells.begin(), mesh.cells.end(),
                                  [](const Cell &c) { return !c.polyhedral; });
  std::vector<int> forbidden;
  for (size_t i = 0; i < mesh.cells.size(); ++i)
    if (!air(model.materials.at(mesh.cells[i].material)) ||
        occupied.contains(mesh.cells[i].id))
      forbidden.push_back(static_cast<int>(i));
  for (auto &wire : std::get<std::vector<Source>>(wires)) {
    if (curved || std::any_of(wire.mesh.cells.begin(), wire.mesh.cells.end(),
                              [](const Cell &c) { return !c.affine; })) {
      auto coupling = couple_curved_source(wire, mesh, index, forbidden);
      if (auto *e = std::get_if<Error>(&coupling))
        return *e;
      wire.coupling = std::get<Eigen::VectorXd>(std::move(coupling));
      sources.push_back(std::move(wire));
      continue;
    }
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
    if (source.coupling.size() == 0) {
      auto coupled = couple_source(source, mesh, index);
      if (auto *e = std::get_if<Error>(&coupled))
        return *e;
      source.coupling = std::get<Eigen::VectorXd>(std::move(coupled));
    }
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

Result<std::vector<CircuitDefinition>>
resolve_circuits(const Model &model, std::span<const Source> sources) {
  std::map<int, size_t> index;
  for (size_t i = 0; i < sources.size(); ++i)
    index.emplace(sources[i].id, i);
  std::map<int, CircuitDefinition> circuits;
  std::set<size_t> listed;
  for (const auto &[id, c] : model.circuits) {
    CircuitDefinition d;
    d.id = id;
    d.parallel = c.parallel;
    for (auto [source, polarity] : c.members) {
      auto it = index.find(source);
      if (it == index.end())
        return Error{std::format("CIRCUIT {} member {} is not a prepared "
                                 "source",
                                 id, source)};
      d.members.emplace_back(it->second, polarity);
      listed.insert(it->second);
    }
    circuits.emplace(id, std::move(d));
  }
  for (size_t i = 0; i < sources.size(); ++i)
    if (!listed.contains(i))
      circuits.emplace(sources[i].id,
                       CircuitDefinition{sources[i].id, 1, {{i, 1}}, true});
  std::vector<CircuitDefinition> out;
  for (auto &[id, d] : circuits) {
    (void)id;
    // A planar circuit is defined only when its signed turns balance; a
    // solid one only when every member is closed.
    double turns = 0, absolute = 0;
    for (auto [i, polarity] : d.members) {
      d.closed &= sources[i].closed || sources[i].mesh.planar;
      turns += polarity * sources[i].turns;
      absolute += std::abs(sources[i].turns);
    }
    if (!d.members.empty() && sources[d.members.front().first].mesh.planar)
      d.closed = std::abs(turns) <= 1e-12 * absolute;
    out.push_back(std::move(d));
  }
  return out;
}

Result<std::vector<double>>
circuit_currents(const Model &model,
                 std::span<const CircuitDefinition> circuits, int set) {
  auto factored = load_factors(model, set);
  if (auto *e = std::get_if<Error>(&factored))
    return *e;
  const auto &factors = std::get<std::map<int, double>>(factored);
  std::map<int, double> currents;
  for (const auto &c : model.currents)
    if (auto it = factors.find(c.set); it != factors.end())
      currents[c.target] += it->second * c.amperes;
  std::vector<double> out;
  for (const auto &c : circuits) {
    const double current = currents[c.id];
    if (!std::isfinite(current))
      return Error{"Current LOAD combination overflow"};
    out.push_back(current);
  }
  return out;
}

Result<std::vector<double>>
source_currents(const Model &model, std::span<const Source> sources, int set) {
  auto resolved = resolve_circuits(model, sources);
  if (auto *e = std::get_if<Error>(&resolved))
    return *e;
  const auto &circuits = std::get<std::vector<CircuitDefinition>>(resolved);
  auto currents = circuit_currents(model, circuits, set);
  if (auto *e = std::get_if<Error>(&currents))
    return *e;
  std::vector<double> out(sources.size(), 0);
  for (size_t c = 0; c < circuits.size(); ++c)
    for (auto [i, polarity] : circuits[c].members)
      out[i] = polarity * std::get<std::vector<double>>(currents)[c] /
               circuits[c].parallel;
  return out;
}
} // namespace vibestran::magnetic
