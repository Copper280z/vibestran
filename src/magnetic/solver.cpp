#include "magnetic/solver.hpp"
#include "solver/solver_backend.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <set>

namespace vibestran::magnetic {
namespace {
using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
// The gauged tangent factored by the solver backend. Newton iterations,
// continuation stages, subcases and inductances all assemble the same
// pattern, so one factor serves the whole solve and its symbolic analysis is
// done once. Every factorization must be positive definite: a failure means
// a gauge, mesh or material error, never a reason to fall back to LDLT.
class Tangent {
public:
  explicit Tangent(SolverBackend &backend) : backend_(backend) {}
  Status factor(const Sparse &stiffness) {
    const Eigen::SparseMatrix<double, Eigen::RowMajor> lower =
        stiffness.triangularView<Eigen::Lower>();
    SparseMatrixBuilder::CsrData csr;
    csr.n = static_cast<int>(lower.rows());
    csr.nnz = static_cast<int>(lower.nonZeros());
    csr.row_ptr.assign(lower.outerIndexPtr(), lower.outerIndexPtr() + csr.n + 1);
    csr.col_ind.assign(lower.innerIndexPtr(), lower.innerIndexPtr() + csr.nnz);
    csr.values.assign(lower.valuePtr(), lower.valuePtr() + csr.nnz);
    csr.symmetry = SymmetryStorage::Lower;
    if (factor_)
      return factor_->refactor(csr);
    FactorOptions options;
    options.require_spd = true;
    auto made = backend_.factor(csr, options);
    if (auto *e = std::get_if<Error>(&made))
      return *e;
    factor_ = std::get<std::unique_ptr<LinearFactor>>(std::move(made));
    return {};
  }
  // Solves for every column of rhs with the last factorization.
  Result<Eigen::MatrixXd> solve(const Eigen::MatrixXd &rhs) {
    auto x = factor_->solve({rhs.data(), static_cast<size_t>(rhs.size())},
                            static_cast<int>(rhs.cols()));
    if (auto *e = std::get_if<Error>(&x))
      return *e;
    const auto &values = std::get<std::vector<double>>(x);
    Eigen::MatrixXd out =
        Eigen::Map<const Eigen::MatrixXd>(values.data(), rhs.rows(), rhs.cols());
    // Checked here too: this file keeps IEEE semantics, the backend may not.
    if (!out.allFinite())
      return Error{"Magnetic linear solve produced a nonfinite solution"};
    return out;
  }

private:
  SolverBackend &backend_;
  std::unique_ptr<LinearFactor> factor_;
};
struct Assembly {
  Sparse stiffness;
  Eigen::VectorXd residual;
  std::vector<Vector> b, h;
  double energy{0}, force_scale{0};
  bool extrapolated{false};
};
// Property data resolved once per solve. Remanence is constant for a basic or
// rectangular frame; curvilinear frames are evaluated at each point.
struct MaterialData {
  Material material;
  const std::vector<std::pair<double, double>> *curve{nullptr};
  const CoordinateSystem *coordinates{nullptr};
  int axis{1};
  Vector remanence{Vector::Zero()};
};
using Materials = std::map<int, MaterialData>; // by property ID
Materials resolve_materials(const Model &model) {
  static const std::vector<std::pair<double, double>> empty;
  Materials out;
  for (const auto &[id, p] : model.properties) {
    MaterialData d;
    d.material = model.materials.at(p.material);
    d.curve = d.material.bh_table ? &model.bh_tables.at(d.material.bh_table)
                                  : &empty;
    d.axis = p.axis;
    if (d.material.br != 0) {
      const auto *cs =
          p.coordinates ? &model.coordinates.at(p.coordinates) : nullptr;
      if (cs && cs->type != 'R')
        d.coordinates = cs;
      else
        d.remanence = (p.axis > 0 ? 1 : -1) * d.material.br *
                      (cs ? coordinate_axes(*cs) : Matrix::Identity())
                          .col(std::abs(p.axis) - 1);
    }
    out.emplace(id, d);
  }
  return out;
}
// Remanence in basic at x.
Vector remanence_at(const MaterialData &d, const Vector &x) {
  if (!d.coordinates)
    return d.remanence;
  return (d.axis > 0 ? 1 : -1) * d.material.br *
         local_axes(*d.coordinates, x).col(std::abs(d.axis) - 1);
}
Response response_at(const MaterialData &d, const Vector &x, const Vector &b) {
  return material_response(d.material, *d.curve, b, remanence_at(d, x));
}
Assembly assemble(const Materials &materials, const Mesh &mesh,
                  const Gauge &gauge, const Eigen::VectorXd &a,
                  const Eigen::VectorXd &load) {
  Assembly out;
  out.residual = Eigen::VectorXd::Zero(gauge.size);
  std::vector<Triplet> entries;
  for (const auto &cell : mesh.cells) {
    const int ne = static_cast<int>(cell.edges.size());
    const MaterialData &material = materials.at(cell.property);
    Eigen::MatrixXd local = Eigen::MatrixXd::Zero(ne, ne);
    Eigen::VectorXd residual = Eigen::VectorXd::Zero(ne);
    Vector average_b = Vector::Zero(), average_h = Vector::Zero();
    for (const auto &s : quadrature(mesh, cell)) {
      const auto curl = edge_curls(cell, s.position);
      Vector b = Vector::Zero();
      for (int e = 0; e < ne; ++e)
        b += a[cell.edges[e]] * curl[e];
      const auto response = response_at(material, s.position, b);
      average_b += s.weight * b;
      average_h += s.weight * response.h;
      const double weight = s.weight * mesh.depth;
      out.energy += weight * response.energy;
      out.extrapolated |= response.extrapolated;
      for (int i = 0; i < ne; ++i) {
        residual[i] += weight * curl[i].dot(response.h);
        for (int j = 0; j < ne; ++j)
          local(i, j) += weight * curl[i].dot(response.tangent * curl[j]);
      }
    }
    out.b.push_back(average_b / cell.volume);
    out.h.push_back(average_h / cell.volume);
    for (int i = 0; i < ne; ++i) {
      const int row = gauge.equations[cell.edges[i]];
      if (row < 0)
        continue;
      out.residual[row] += residual[i];
      out.force_scale += residual[i] * residual[i];
      for (int j = 0; j < ne; ++j) {
        const int col = gauge.equations[cell.edges[j]];
        if (col >= 0)
          entries.emplace_back(row, col, local(i, j));
      }
    }
  }
  for (size_t e = 0; e < mesh.dofs(); ++e)
    if (gauge.equations[e] >= 0)
      out.residual[gauge.equations[e]] -= load[e];
  out.stiffness.resize(gauge.size, gauge.size);
  out.stiffness.setFromTriplets(entries.begin(), entries.end());
  return out;
}
Result<Eigen::VectorXd> boundary_potential(const Model &model, const Mesh &mesh,
                                           const Subcase &subcase) {
  Eigen::VectorXd a = Eigen::VectorXd::Zero(mesh.dofs());
  if (!subcase.boundary)
    return a;
  const auto &bc = model.boundaries.at(subcase.boundary);
  if (bc.faces) {
    auto faces = selected_faces(model, mesh, bc.faces);
    if (auto *e = std::get_if<Error>(&faces))
      return *e;
    const auto count = static_cast<size_t>(
        std::count_if(mesh.faces.begin(), mesh.faces.end(),
                      [](const Face &f) { return f.cells[1] < 0; }));
    const auto &selected = std::get<std::vector<std::pair<int, int>>>(faces);
    if (selected.size() != count)
      return Error{
          "MBC FSET must cover the entire magnetic exterior (or use FSET=0)"};
    for (auto [f, sign] : selected) {
      (void)sign;
      if (mesh.faces[f].cells[1] >= 0)
        return Error{"MBC must select exterior faces"};
    }
  }
  if (mesh.planar) {
    if (std::abs(bc.field.dot(mesh.normal)) > 1e-12 * bc.field.norm())
      return Error{"MAG2D MBC field must lie in the simulation plane"};
    for (size_t i = 0; i < mesh.nodes.size(); ++i)
      a[i] =
          bc.field.cross(mesh.nodes[i] - mesh.nodes.front()).dot(mesh.normal);
    return a;
  }
  for (size_t e = 0; e < mesh.dofs(); ++e) {
    const Vector &x = mesh.nodes[mesh.edges[e][0]];
    const Vector &y = mesh.nodes[mesh.edges[e][1]];
    a[e] = 0.5 * bc.field.cross((x + y) / 2).dot(y - x);
  }
  return a;
}
struct Equilibrium {
  Eigen::VectorXd a;
  Assembly assembly;
  int iterations{0};
  double residual{0};
};
Result<Equilibrium> equilibrium(const Model &model, const Materials &materials,
                                const Mesh &mesh, const Gauge &gauge,
                                Tangent &tangent, Eigen::VectorXd a,
                                const Eigen::VectorXd &load) {
  auto state = assemble(materials, mesh, gauge, a, load);
  double load_norm = 0;
  for (size_t e = 0; e < mesh.dofs(); ++e)
    if (gauge.equations[e] >= 0)
      load_norm += load[e] * load[e];
  const double reference =
      std::max({std::sqrt(load_norm), std::sqrt(state.force_scale),
                state.residual.norm(), 1e-30});
  int iteration = 0;
  for (; iteration <= model.options.max_iterations; ++iteration) {
    const double norm = state.residual.norm();
    if (norm / reference <= model.options.nonlinear_tolerance ||
        gauge.size == 0)
      return Equilibrium{std::move(a), std::move(state), iteration,
                         norm / reference};
    if (iteration == model.options.max_iterations)
      break;
    if (auto e = tangent.factor(state.stiffness))
      return Error{std::format("Magnetic tangent factorization failed ({}); "
                               "check mesh, gauge and material data",
                               e->message)};
    auto solved = tangent.solve(-state.residual);
    if (auto *e = std::get_if<Error>(&solved))
      return Error{std::format("Magnetic linear solve failed ({})", e->message)};
    const Eigen::VectorXd step = std::get<Eigen::MatrixXd>(std::move(solved));
    bool accepted = false;
    double alpha = 1;
    for (int trial = 0; trial < 30; ++trial) {
      Eigen::VectorXd candidate = a;
      for (size_t e = 0; e < mesh.dofs(); ++e)
        if (gauge.equations[e] >= 0)
          candidate[e] += alpha * step[gauge.equations[e]];
      auto next = assemble(materials, mesh, gauge, candidate, load);
      if (next.residual.allFinite() &&
          next.residual.norm() <= (1 - 1e-4 * alpha) * norm) {
        a = std::move(candidate);
        state = std::move(next);
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted)
      return Error{std::format("Magnetic Newton line search failed at "
                               "iteration {} (relative residual {})",
                               iteration, norm / reference)};
  }
  return Error{std::format("Magnetic Newton did not converge in {} iterations "
                           "(relative residual {})",
                           iteration, state.residual.norm() / reference)};
}
Vector field_at(const Cell &c, const Eigen::VectorXd &a, const Vector &x) {
  const auto curls = edge_curls(c, x);
  Vector b = Vector::Zero();
  for (size_t i = 0; i < curls.size(); ++i)
    b += a[c.edges[i]] * curls[i];
  return b;
}
// Field cells inside a COIL or COIL2D winding, mapped to their source ID.
std::map<int, int> winding_cells(const Model &model, const Mesh &mesh) {
  std::map<int, int> out;
  for (const auto &c : model.coils)
    for (int id : model.element_sets.at(c.elements))
      out[mesh.element_index.at(id)] = c.id;
  for (const auto &c : model.planar_coils)
    for (int id : model.element_sets.at(c.elements))
      out[mesh.element_index.at(id)] = c.id;
  return out;
}
std::string face_grids(const Mesh &mesh, const Face &f) {
  std::string out;
  for (int n : f.nodes)
    out += (out.empty() ? "" : ",") + std::to_string(mesh.node_ids[n]);
  return out;
}
// Positive-area overlap of a flat face of b with the flat face fa, when they
// are coplanar. Both faces are convex polygons.
bool coplanar_overlap(const Mesh &a, const Face &fa, const Mesh &b,
                      const Face &fb) {
  const double length = std::sqrt(fa.area);
  if (!fa.flat || !fb.flat || std::abs(fa.normal.dot(fb.normal)) < 1 - 1e-12)
    return false;
  std::vector<Vector> polygon;
  for (int n : fb.nodes) {
    if (std::abs(fa.normal.dot(b.nodes[n] - fa.center)) > 1e-10 * length)
      return false;
    polygon.push_back(b.nodes[n]);
  }
  for (size_t i = 0; i < fa.nodes.size() && !polygon.empty(); ++i) {
    const Vector x0 = a.nodes[fa.nodes[i]],
                 x1 = a.nodes[fa.nodes[(i + 1) % fa.nodes.size()]];
    Vector inward = fa.normal.cross(x1 - x0);
    if (inward.dot(fa.center - x0) < 0)
      inward = -inward;
    std::vector<Vector> clipped;
    for (size_t k = 0; k < polygon.size(); ++k) {
      const Vector &p = polygon[k], &q = polygon[(k + 1) % polygon.size()];
      const double dp = inward.dot(p - x0), dq = inward.dot(q - x0);
      if (dp >= 0)
        clipped.push_back(p);
      if ((dp < 0) != (dq < 0))
        clipped.push_back(p + dp / (dp - dq) * (q - p));
    }
    polygon = std::move(clipped);
  }
  Vector area = Vector::Zero();
  for (size_t k = 2; k < polygon.size(); ++k)
    area += (polygon[k - 1] - polygon[0]).cross(polygon[k] - polygon[0]) / 2;
  return area.norm() > 1e-10 * std::min(fa.area, fb.area);
}
// Maxwell stress gives the force on everything a surface encloses, so a face
// may bound a winding but must not cut through current-carrying volume.
Status current_free(const Mesh &mesh, const Face &f, int request,
                    const std::map<int, int> &windings,
                    std::span<const Source> sources,
                    std::span<const SpatialIndex> wire_indexes) {
  const auto first = windings.find(f.cells[0]),
             second = windings.find(f.cells[1]);
  if (first != windings.end() && second != windings.end())
    return Error{std::format(
        "MFORCE {}: surface face ({}) of element {} lies inside current "
        "source {}; move the surface clear of the winding or enclose it",
        request, face_grids(mesh, f), mesh.cells[f.cells[0]].id,
        first->second)};
  if (mesh.planar)
    return {};
  Vector lo = mesh.nodes[f.nodes[0]], hi = lo;
  for (int n : f.nodes) {
    lo = lo.cwiseMin(mesh.nodes[n]);
    hi = hi.cwiseMax(mesh.nodes[n]);
  }
  size_t wire = 0;
  for (const auto &source : sources) {
    if (!source.wire)
      continue;
    for (int ci : wire_indexes[wire].query(lo, hi)) {
      auto crossing = face_intersects_interior(
          source.mesh, source.mesh.cells[ci], mesh, f);
      if (auto *e = std::get_if<Error>(&crossing))
        return *e;
      // The conductor partition places internal faces on its centerline, so
      // a surface through the conductor can lie on them without entering any
      // single cell.
      const auto &cell = source.mesh.cells[ci];
      const bool on_internal_face = std::any_of(
          cell.faces.begin(), cell.faces.end(), [&](int g) {
            const auto &face = source.mesh.faces[g];
            return face.cells[1] >= 0 &&
                   coplanar_overlap(mesh, f, source.mesh, face);
          });
      if (std::get<bool>(crossing) || on_internal_face)
        return Error{std::format(
            "MFORCE {}: surface face ({}) of element {} crosses CWIRE path "
            "{}; move the surface clear of the conductor",
            request, face_grids(mesh, f), mesh.cells[f.cells[0]].id,
            source.id)};
    }
    ++wire;
  }
  return {};
}
bool magnetic(const Material &m) {
  return m.mur != 1 || m.bh_table != 0 || m.br != 0;
}
// A MFORCE BODY surface: every face with exactly one side in R, the body
// plus `layers` grid-adjacent layers of nonmagnetic, current-free cells,
// oriented out of R. Only the selected body may contain current sources.
// It includes cavity walls, so an annular body excludes what it surrounds.
// component[k] is the connected piece of R that face k bounds.
struct BodySurface {
  std::vector<std::pair<int, int>> faces;
  std::vector<int> component;
  int components{0};
};
Result<BodySurface> body_surface(const Model &model, const Mesh &mesh,
                                 const std::vector<std::vector<int>> &incident,
                                 const ForceSurface &request,
                                 const std::map<int, int> &windings,
                                 std::span<const Source> sources,
                                 std::span<const SpatialIndex> wire_indexes) {
  const int nc = static_cast<int>(mesh.cells.size());
  std::vector<int> layer(nc, -1); // 0 for the body, k for layer k
  std::vector<int> frontier;
  for (int id : model.element_sets.at(request.set)) {
    const int ci = mesh.element_index.at(id);
    if (layer[ci] < 0)
      frontier.push_back(ci);
    layer[ci] = 0;
  }
  auto is_magnetic = [&](int ci) {
    return magnetic(model.materials.at(mesh.cells[ci].material));
  };
  for (int k = 1; k <= request.layers; ++k) {
    std::vector<int> next;
    for (int ci : frontier)
      for (int n : mesh.cells[ci].nodes)
        for (int neighbor : incident[n])
          if (layer[neighbor] < 0) {
            if (is_magnetic(neighbor))
              return Error{
                  k == 1 ? std::format(
                               "MFORCE {}: magnetic element {} touches body "
                               "ESET {}; add it to the body set or mesh a "
                               "nonmagnetic gap between them",
                               request.id, mesh.cells[neighbor].id, request.set)
                         : std::format(
                               "MFORCE {}: magnetic element {} lies within {} "
                               "layers of body ESET {}; use a smaller LAYER or "
                               "more air layers",
                               request.id, mesh.cells[neighbor].id, k,
                               request.set)};
            const auto &cell = mesh.cells[neighbor];
            if (auto it = windings.find(neighbor); it != windings.end())
              return Error{std::format(
                  "MFORCE {}: layer {} of body ESET {} includes element {} "
                  "of current source {}; add the winding to the body set "
                  "or use a smaller LAYER or more air layers",
                  request.id, k, request.set, cell.id, it->second)};
            size_t wire = 0;
            for (const auto &source : sources) {
              if (!source.wire)
                continue;
              for (int ci : wire_indexes[wire].query(cell.minimum, cell.maximum)) {
                const auto &conductor = source.mesh.cells[ci];
                Result<bool> overlap = false;
                if (cell.polyhedral) {
                  const auto samples = intersection_quadrature(
                      source.mesh, conductor, mesh, cell);
                  const double volume = std::accumulate(
                      samples.begin(), samples.end(), 0.0,
                      [](double sum, const Sample &q) { return sum + q.weight; });
                  overlap = volume > 1e-10 * conductor.volume;
                } else
                  overlap = cell_intersects_interior(source.mesh, conductor,
                                                     cell);
                if (auto *e = std::get_if<Error>(&overlap))
                  return Error{std::format(
                      "MFORCE {}: checking layer {} of body ESET {}, element "
                      "{}, against CWIRE path {}: {}",
                      request.id, k, request.set, cell.id, source.id,
                      e->message)};
                if (std::get<bool>(overlap))
                  return Error{std::format(
                      "MFORCE {}: layer {} of body ESET {} includes element "
                      "{} containing CWIRE path {}; add all field cells "
                      "containing the conductor to the body set or use a "
                      "smaller LAYER or more air layers",
                      request.id, k, request.set, cell.id, source.id)};
              }
              ++wire;
            }
            layer[neighbor] = k;
            next.push_back(neighbor);
          }
    frontier = std::move(next);
  }
  // Connected pieces of R through shared grids.
  std::vector<int> piece(nc, -1);
  BodySurface out;
  for (int seed = 0; seed < nc; ++seed) {
    if (layer[seed] < 0 || piece[seed] >= 0)
      continue;
    std::vector<int> stack{seed};
    piece[seed] = out.components;
    while (!stack.empty()) {
      const int ci = stack.back();
      stack.pop_back();
      for (int n : mesh.cells[ci].nodes)
        for (int neighbor : incident[n])
          if (layer[neighbor] >= 0 && piece[neighbor] < 0) {
            piece[neighbor] = out.components;
            stack.push_back(neighbor);
          }
    }
    ++out.components;
  }
  for (int ci = 0; ci < nc; ++ci) {
    if (layer[ci] < 0)
      continue;
    const auto &cell = mesh.cells[ci];
    for (size_t k = 0; k < cell.faces.size(); ++k) {
      const auto &f = mesh.faces[cell.faces[k]];
      const int other = f.cells[0] == ci ? f.cells[1] : f.cells[0];
      if (other < 0)
        return Error{std::format(
            "MFORCE {}: body ESET {} is within {} cells of the domain "
            "boundary at element {}; request the force on the opposite body",
            request.id, request.set, request.layers, cell.id)};
      if (layer[other] >= 0)
        continue;
      if (is_magnetic(other))
        return Error{std::format(
            "MFORCE {}: magnetic element {} lies within {} layers of body "
            "ESET {}; use a smaller LAYER or more air layers",
            request.id, mesh.cells[other].id, request.layers + 1,
            request.set)};
      out.faces.emplace_back(cell.faces[k], cell.face_sign[k]);
      out.component.push_back(piece[ci]);
    }
  }
  return out;
}
// Validates an explicit FACES surface: closed, consistently and outwardly
// oriented, strictly inside the domain.
Status explicit_surface(const Mesh &mesh,
                        std::span<const std::pair<int, int>> selected) {
  std::map<std::array<int, 2>, std::pair<int, int>> boundary;
  for (auto [fi, sign] : selected) {
    const auto &f = mesh.faces[fi];
    if (f.cells[1] < 0)
      return Error{
          "MFORCE surface must lie strictly inside the magnetic domain"};
    // A planar contour closes at its end grids; a 3D surface at its edges.
    if (mesh.planar) {
      auto &first = boundary[{f.nodes[0], f.nodes[0]}];
      auto &second = boundary[{f.nodes[1], f.nodes[1]}];
      ++first.first;
      ++second.first;
      first.second -= sign;
      second.second += sign;
      continue;
    }
    for (size_t i = 0; i < f.nodes.size(); ++i) {
      int a = f.nodes[i], b = f.nodes[(i + 1) % f.nodes.size()],
          direction = sign;
      if (a > b) {
        std::swap(a, b);
        direction = -direction;
      }
      auto &count = boundary[{a, b}];
      ++count.first;
      count.second += direction;
    }
  }
  for (const auto &[edge, count] : boundary) {
    (void)edge;
    if (count.first != 2 || count.second != 0)
      return Error{mesh.planar ? "MFORCE planar FSET must be a closed, "
                                 "consistently oriented contour"
                               : "MFORCE FSET must be a closed, consistently "
                                 "oriented surface"};
  }
  // Positive signed enclosed volume distinguishes outward from inward
  // normals.
  double volume = 0;
  for (auto [fi, sign] : selected)
    for (const auto &q : face_quadrature(mesh, mesh.faces[fi]))
      volume += sign * q.weight * q.normal.dot(q.position) /
                (mesh.planar ? 2 : 3);
  if (!(volume > 0))
    return Error{"MFORCE normals must point outward"};
  return {};
}
Result<std::vector<ForceResult>> forces(const Model &model, const Mesh &mesh,
                                        std::span<const Source> sources,
                                        const Eigen::VectorXd &potential) {
  std::vector<ForceResult> results;
  if (model.forces.empty())
    return results;
  const auto windings = winding_cells(model, mesh);
  std::vector<SpatialIndex> wire_indexes;
  for (const auto &source : sources)
    if (source.wire)
      wire_indexes.emplace_back(source.mesh);
  std::vector<std::vector<int>> incident(mesh.nodes.size());
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci)
    for (int n : mesh.cells[ci].nodes)
      incident[n].push_back(static_cast<int>(ci));
  for (const auto &request : model.forces) {
    std::vector<std::pair<int, int>> faces;
    std::vector<int> component;
    ForceResult result;
    result.id = request.id;
    if (request.body) {
      auto derived = body_surface(model, mesh, incident, request, windings,
                                  sources, wire_indexes);
      if (auto *e = std::get_if<Error>(&derived))
        return *e;
      auto &surface = std::get<BodySurface>(derived);
      faces = std::move(surface.faces);
      component = std::move(surface.component);
      result.components.resize(surface.components);
    } else {
      auto selected = selected_faces(model, mesh, request.set);
      if (auto *e = std::get_if<Error>(&selected))
        return *e;
      faces = std::get<std::vector<std::pair<int, int>>>(std::move(selected));
      if (auto e = explicit_surface(mesh, faces))
        return *e;
    }
    const Vector origin = request.coordinates
                              ? model.coordinates.at(request.coordinates).a
                              : Vector::Zero();
    for (size_t k = 0; k < faces.size(); ++k) {
      const auto [fi, sign] = faces[k];
      const auto &f = mesh.faces[fi];
      for (int ci : f.cells)
        if (magnetic(model.materials.at(mesh.cells[ci].material)))
          return Error{std::format(
              "MFORCE {}: surface face ({}) touches magnetic element {}; "
              "both sides of every face must be mu_r=1 without remanence",
              request.id, face_grids(mesh, f), mesh.cells[ci].id)};
      if (auto e = current_free(mesh, f, request.id, windings, sources,
                                wire_indexes))
        return *e;
      for (const auto &q : face_quadrature(mesh, f)) {
        const Vector normal = sign * q.normal;
        const Vector b =
            .5 * (field_at(mesh.cells[f.cells[0]], potential, q.position) +
                  field_at(mesh.cells[f.cells[1]], potential, q.position));
        const Vector force =
            q.weight * mesh.depth / mu0 *
            (b * b.dot(normal) - .5 * b.squaredNorm() * normal);
        const Vector torque = (q.position - origin).cross(force);
        result.force += force;
        result.torque += torque;
        if (!component.empty()) {
          result.components[component[k]].force += force;
          result.components[component[k]].torque += torque;
        }
      }
    }
    results.push_back(std::move(result));
  }
  return results;
}
} // namespace

Result<Results> solve(const Model &model) {
  EigenSolverBackend backend;
  return solve(model, backend);
}

Result<Results> solve(const Model &model, SolverBackend &backend) {
  if (auto e = validate(model))
    return *e;
  auto built = make_mesh(model);
  if (auto *e = std::get_if<Error>(&built))
    return *e;
  Results result;
  result.mesh = std::get<Mesh>(std::move(built));
  const auto &mesh = result.mesh;
  const Materials materials = resolve_materials(model);
  auto gauged = make_gauge(mesh);
  if (auto *e = std::get_if<Error>(&gauged))
    return *e;
  const auto &gauge = std::get<Gauge>(gauged);
  auto prepared = prepare_sources(model, mesh);
  if (auto *e = std::get_if<Error>(&prepared))
    return *e;
  result.sources = std::get<std::vector<Source>>(std::move(prepared));
  auto resolved = resolve_circuits(model, result.sources);
  if (auto *e = std::get_if<Error>(&resolved))
    return *e;
  result.circuits = std::get<std::vector<CircuitDefinition>>(std::move(resolved));
  const auto &circuits = result.circuits;
  const int ns = static_cast<int>(result.sources.size()),
            nc = static_cast<int>(circuits.size());
  // Circuit coupling g_c = sum P_i c_i / a_c, so circuit linkage is g_c . A
  // and circuit inductance needs one tangent solve per closed circuit.
  std::vector<Eigen::VectorXd> circuit_coupling;
  for (const auto &c : circuits) {
    Eigen::VectorXd g = Eigen::VectorXd::Zero(mesh.dofs());
    for (auto [i, polarity] : c.members)
      g += polarity * result.sources[i].coupling;
    circuit_coupling.push_back(g / c.parallel);
  }
  SpatialIndex index(mesh);
  Tangent tangent(backend);
  for (const auto &subcase : model.subcases) {
    auto terminal = circuit_currents(model, circuits, subcase.load);
    if (auto *e = std::get_if<Error>(&terminal))
      return *e;
    const auto &circuit_amps = std::get<std::vector<double>>(terminal);
    std::vector<double> amps(ns, 0);
    for (int c = 0; c < nc; ++c)
      for (auto [i, polarity] : circuits[c].members)
        amps[i] = polarity * circuit_amps[c] / circuits[c].parallel;
    Eigen::VectorXd load = Eigen::VectorXd::Zero(mesh.dofs());
    for (size_t i = 0; i < amps.size(); ++i)
      load += amps[i] * result.sources[i].coupling;
    auto imposed = boundary_potential(model, mesh, subcase);
    if (auto *e = std::get_if<Error>(&imposed))
      return *e;
    const auto &initial = std::get<Eigen::VectorXd>(imposed);
    auto solved =
        equilibrium(model, materials, mesh, gauge, tangent, initial, load);
    if (std::holds_alternative<Error>(solved) && !model.bh_tables.empty()) {
      // Retry with fixed current continuation. Each stage must satisfy the same
      // convergence criterion; a failed stage is never returned as a solution.
      Eigen::VectorXd a = initial;
      int iterations = 0;
      for (int stage = 0; stage <= 8; ++stage) {
        solved = equilibrium(model, materials, mesh, gauge, tangent, a,
                             load * (stage / 8.0));
        if (auto *e = std::get_if<Error>(&solved))
          return Error{std::format("Subcase {}, continuation stage {}: {}",
                                   subcase.id, stage, e->message)};
        const auto &eq = std::get<Equilibrium>(solved);
        iterations += eq.iterations;
        a = eq.a;
      }
      std::get<Equilibrium>(solved).iterations = iterations;
    }
    if (auto *e = std::get_if<Error>(&solved))
      return Error{std::format("Subcase {}: {}", subcase.id, e->message)};
    auto eq = std::get<Equilibrium>(std::move(solved));
    CaseResults output;
    output.id = subcase.id;
    output.iterations = eq.iterations;
    output.residual = eq.residual;
    output.energy = eq.assembly.energy;
    output.extrapolated = eq.assembly.extrapolated;
    output.potential = eq.a;
    output.b = eq.assembly.b;
    output.h = eq.assembly.h;
    output.coenergy = -output.energy;
    constexpr double unavailable = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < ns; ++i) {
      const auto &source = result.sources[i];
      const double linkage = source.coupling.dot(eq.a);
      output.coenergy += amps[i] * linkage;
      output.sources.push_back({source.id, source.wire, source.closed, amps[i],
                                source.closed ? linkage : unavailable});
    }
    for (int c = 0; c < nc; ++c)
      output.circuits.push_back({circuits[c].id, circuit_amps[c],
                                 circuits[c].closed
                                     ? circuit_coupling[c].dot(eq.a)
                                     : unavailable});
    // Differential inductances g_i^T K^-1 g_j, from the converged tangent,
    // among closed sources and among closed circuits; others are unavailable.
    output.differential_inductance =
        Eigen::MatrixXd::Constant(ns, ns, unavailable);
    output.circuit_inductance = Eigen::MatrixXd::Constant(nc, nc, unavailable);
    if (gauge.size > 0) {
      // Reduced couplings of the closed sources, then of the closed circuits,
      // solved together with one factorization of the converged tangent.
      std::vector<int> closed_sources, closed_circuits;
      std::vector<const Eigen::VectorXd *> couplings;
      for (int i = 0; i < ns; ++i)
        if (result.sources[i].closed) {
          closed_sources.push_back(i);
          couplings.push_back(&result.sources[i].coupling);
        }
      for (int c = 0; c < nc; ++c)
        if (circuits[c].closed) {
          closed_circuits.push_back(c);
          couplings.push_back(&circuit_coupling[c]);
        }
      if (!couplings.empty()) {
        Eigen::MatrixXd reduced = Eigen::MatrixXd::Zero(
            gauge.size, static_cast<Eigen::Index>(couplings.size()));
        for (size_t k = 0; k < couplings.size(); ++k)
          for (size_t e = 0; e < mesh.dofs(); ++e)
            if (gauge.equations[e] >= 0)
              reduced(gauge.equations[e], static_cast<Eigen::Index>(k)) =
                  (*couplings[k])[e];
        if (auto e = tangent.factor(eq.assembly.stiffness))
          return Error{std::format(
              "Differential inductance factorization failed ({})", e->message)};
        auto responses = tangent.solve(reduced);
        if (auto *e = std::get_if<Error>(&responses))
          return Error{std::format("Differential inductance solve failed ({})",
                                   e->message)};
        const Eigen::MatrixXd linkage =
            reduced.transpose() * std::get<Eigen::MatrixXd>(responses);
        const auto first_circuit =
            static_cast<Eigen::Index>(closed_sources.size());
        for (size_t i = 0; i < closed_sources.size(); ++i)
          for (size_t j = 0; j < closed_sources.size(); ++j)
            output.differential_inductance(closed_sources[i],
                                           closed_sources[j]) =
                linkage(static_cast<Eigen::Index>(i),
                        static_cast<Eigen::Index>(j));
        for (size_t c = 0; c < closed_circuits.size(); ++c)
          for (size_t d = 0; d < closed_circuits.size(); ++d)
            output.circuit_inductance(closed_circuits[c], closed_circuits[d]) =
                linkage(first_circuit + static_cast<Eigen::Index>(c),
                        first_circuit + static_cast<Eigen::Index>(d));
      }
    } else {
      for (int i = 0; i < ns; ++i)
        for (int j = 0; j < ns; ++j)
          if (result.sources[i].closed && result.sources[j].closed)
            output.differential_inductance(i, j) = 0;
      for (int c = 0; c < nc; ++c)
        for (int d = 0; d < nc; ++d)
          if (circuits[c].closed && circuits[d].closed)
            output.circuit_inductance(c, d) = 0;
    }
    for (const auto &probe : model.probes) {
      Vector b = Vector::Zero(), h = Vector::Zero();
      int count = 0;
      for (int ci : index.query(probe.position, probe.position)) {
        const auto &cell = mesh.cells[ci];
        if (contains(cell, probe.position)) {
          const Vector field = field_at(cell, eq.a, probe.position);
          b += field;
          h += response_at(materials.at(cell.property), probe.position, field)
                   .h;
          ++count;
        }
      }
      if (count == 0)
        return Error{"MPROBE lies outside the magnetic domain: " +
                     std::to_string(probe.id)};
      output.probe_b.push_back(b / count);
      output.probe_h.push_back(h / count);
    }
    auto force = forces(model, mesh, result.sources, eq.a);
    if (auto *e = std::get_if<Error>(&force))
      return *e;
    output.forces = std::get<std::vector<ForceResult>>(std::move(force));
    result.cases.push_back(std::move(output));
  }
  return result;
}
} // namespace vibestran::magnetic
