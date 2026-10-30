#include "magnetic/solver.hpp"
#include <Eigen/SparseCholesky>
#if defined(HAVE_ACCELERATE)
#include <Eigen/AccelerateSupport>
#elif defined(EIGEN_CHOLMOD_SUPPORT)
#include <Eigen/CholmodSupport>
#endif
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <set>

namespace vibestran::magnetic {
namespace {
using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
#if defined(HAVE_ACCELERATE)
using Factor = Eigen::AccelerateLLT<Sparse>;
#elif defined(EIGEN_CHOLMOD_SUPPORT)
using Factor = Eigen::CholmodSupernodalLLT<Sparse>;
#else
using Factor = Eigen::SimplicialLLT<Sparse>;
#endif
struct Assembly {
  Sparse stiffness;
  Eigen::VectorXd residual;
  std::vector<Vector> b, h;
  double energy{0}, force_scale{0};
  bool extrapolated{false};
};
Material at_point(const Model &model, int id, const Vector &x) {
  Material m = model.materials.at(id);
  if (m.coordinates) {
    const auto &cs = model.coordinates.at(m.coordinates);
    const Vector z = (cs.b - cs.a).normalized(),
                 y = z.cross(cs.c - cs.a).normalized();
    Matrix frame;
    frame.col(0) = y.cross(z);
    frame.col(1) = y;
    frame.col(2) = z;
    const Vector p = frame.transpose() * (x - cs.a);
    if (cs.type == 'C') {
      const double theta = std::atan2(p.y(), p.x());
      m.remanence = Eigen::AngleAxisd(theta, Vector::UnitZ()) * m.remanence;
    } else if (cs.type == 'S') {
      const double theta = std::atan2(std::hypot(p.x(), p.y()), p.z()),
                   phi = std::atan2(p.y(), p.x());
      Matrix spherical;
      spherical.col(0) =
          Vector(std::sin(theta) * std::cos(phi),
                 std::sin(theta) * std::sin(phi), std::cos(theta));
      spherical.col(1) =
          Vector(std::cos(theta) * std::cos(phi),
                 std::cos(theta) * std::sin(phi), -std::sin(theta));
      spherical.col(2) = Vector(-std::sin(phi), std::cos(phi), 0);
      m.remanence = spherical * m.remanence;
    }
    m.remanence = frame * m.remanence;
  }
  return m;
}
Assembly assemble(const Model &model, const Mesh &mesh, const Gauge &gauge,
                  const Eigen::VectorXd &a, const Eigen::VectorXd &load) {
  Assembly out;
  out.residual = Eigen::VectorXd::Zero(gauge.size);
  std::vector<Triplet> entries;
  const std::vector<std::pair<double, double>> empty;
  for (const auto &cell : mesh.cells) {
    const int ne = static_cast<int>(cell.edges.size());
    Eigen::MatrixXd local = Eigen::MatrixXd::Zero(ne, ne);
    Eigen::VectorXd residual = Eigen::VectorXd::Zero(ne);
    Vector average_b = Vector::Zero(), average_h = Vector::Zero();
    for (const auto &s : quadrature(mesh, cell)) {
      const auto curl = edge_curls(cell, s.position);
      Vector b = Vector::Zero();
      for (int e = 0; e < ne; ++e)
        b += a[cell.edges[e]] * curl[e];
      const Material material = at_point(model, cell.material, s.position);
      const auto &curve =
          material.bh_table ? model.bh_tables.at(material.bh_table) : empty;
      const auto response = material_response(material, curve, b);
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
Result<Equilibrium> equilibrium(const Model &model, const Mesh &mesh,
                                const Gauge &gauge, Eigen::VectorXd a,
                                const Eigen::VectorXd &load) {
  auto state = assemble(model, mesh, gauge, a, load);
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
    Factor factor;
#if defined(HAVE_ACCELERATE)
    factor.setOrder(SparseOrderMetis);
#endif
    factor.compute(state.stiffness);
    if (factor.info() != Eigen::Success)
      return Error{"Magnetic tangent is not positive definite; check mesh, "
                   "gauge and material data"};
    const Eigen::VectorXd rhs = -state.residual;
    const Eigen::VectorXd step = factor.solve(rhs);
    if (factor.info() != Eigen::Success || !step.allFinite())
      return Error{"Magnetic linear solve failed"};
    bool accepted = false;
    double alpha = 1;
    for (int trial = 0; trial < 30; ++trial) {
      Eigen::VectorXd candidate = a;
      for (size_t e = 0; e < mesh.dofs(); ++e)
        if (gauge.equations[e] >= 0)
          candidate[e] += alpha * step[gauge.equations[e]];
      auto next = assemble(model, mesh, gauge, candidate, load);
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
Result<std::vector<ForceResult>> forces(const Model &model, const Mesh &mesh,
                                        const Eigen::VectorXd &potential) {
  std::vector<ForceResult> results;
  for (const auto &request : model.forces) {
    auto selected = selected_faces(model, mesh, request.faces);
    if (auto *e = std::get_if<Error>(&selected))
      return *e;
    std::map<std::array<int, 2>, std::pair<int, int>> boundary;
    std::map<int, std::pair<int, int>> line_boundary;
    ForceResult result;
    result.id = request.id;
    for (auto [fi, sign] :
         std::get<std::vector<std::pair<int, int>>>(selected)) {
      const auto &f = mesh.faces[fi];
      if (f.cells[1] < 0)
        return Error{
            "MFORCE surface must lie strictly inside the magnetic domain"};
      for (int ci : f.cells) {
        const auto &m = model.materials.at(mesh.cells[ci].material);
        if (m.mur != 1 || m.bh_table || !m.remanence.isZero(0))
          return Error{
              "MFORCE requires air on both sides of every surface face"};
      }
      const Vector normal = sign * f.normal;
      for (const auto &q : face_quadrature(mesh, f)) {
        const Vector b =
            .5 * (field_at(mesh.cells[f.cells[0]], potential, q.position) +
                  field_at(mesh.cells[f.cells[1]], potential, q.position));
        const Vector force =
            q.weight * mesh.depth / mu0 *
            (b * b.dot(normal) - .5 * b.squaredNorm() * normal);
        result.force += force;
        result.torque += (q.position - request.origin).cross(force);
      }
      if (mesh.planar) {
        ++line_boundary[f.nodes[0]].first;
        ++line_boundary[f.nodes[1]].first;
        line_boundary[f.nodes[0]].second -= sign;
        line_boundary[f.nodes[1]].second += sign;
        continue;
      }
      for (size_t i = 0; i < f.nodes.size(); ++i) {
        int a = f.nodes[i], bnode = f.nodes[(i + 1) % f.nodes.size()],
            direction = sign;
        if (a > bnode) {
          std::swap(a, bnode);
          direction = -direction;
        }
        auto &count = boundary[{a, bnode}];
        ++count.first;
        count.second += direction;
      }
    }
    for (const auto &[edge, count] : boundary) {
      (void)edge;
      if (count.first != 2 || count.second != 0)
        return Error{
            "MFORCE FSET must be a closed, consistently oriented surface"};
    }
    for (const auto &[node, count] : line_boundary) {
      (void)node;
      if (count.first != 2 || count.second != 0)
        return Error{"MFORCE planar FSET must be a closed, consistently "
                     "oriented contour"};
    }
    // Positive signed enclosed volume distinguishes outward from inward
    // normals.
    double volume = 0;
    for (auto [fi, sign] :
         std::get<std::vector<std::pair<int, int>>>(selected)) {
      const auto &f = mesh.faces[fi];
      volume += sign * f.area * f.normal.dot(f.center - request.origin) /
                (mesh.planar ? 2 : 3);
    }
    if (!(volume > 0))
      return Error{"MFORCE normals must point outward"};
    results.push_back(result);
  }
  return results;
}
} // namespace

Result<Results> solve(const Model &model) {
  if (auto e = validate(model))
    return *e;
  auto built = make_mesh(model);
  if (auto *e = std::get_if<Error>(&built))
    return *e;
  Results result;
  result.mesh = std::get<Mesh>(std::move(built));
  const auto &mesh = result.mesh;
  if (mesh.planar)
    for (const auto &c : mesh.cells)
      for (const auto &q : quadrature(mesh, c)) {
        const auto material = at_point(model, c.material, q.position);
        if (std::abs(material.remanence.dot(mesh.normal)) >
            1e-12 * material.remanence.norm())
          return Error{"MAG2D remanence must lie in the simulation plane"};
      }
  auto gauged = make_gauge(mesh);
  if (auto *e = std::get_if<Error>(&gauged))
    return *e;
  const auto &gauge = std::get<Gauge>(gauged);
  auto prepared = prepare_sources(model, mesh);
  if (auto *e = std::get_if<Error>(&prepared))
    return *e;
  result.sources = std::get<std::vector<Source>>(std::move(prepared));
  SpatialIndex index(mesh);
  for (const auto &subcase : model.subcases) {
    auto currents = source_currents(model, result.sources, subcase.load);
    if (auto *e = std::get_if<Error>(&currents))
      return *e;
    const auto &amps = std::get<std::vector<double>>(currents);
    Eigen::VectorXd load = Eigen::VectorXd::Zero(mesh.dofs());
    for (size_t i = 0; i < amps.size(); ++i)
      load += amps[i] * result.sources[i].coupling;
    auto imposed = boundary_potential(model, mesh, subcase);
    if (auto *e = std::get_if<Error>(&imposed))
      return *e;
    const auto &initial = std::get<Eigen::VectorXd>(imposed);
    auto solved = equilibrium(model, mesh, gauge, initial, load);
    if (std::holds_alternative<Error>(solved) && !model.bh_tables.empty()) {
      // Retry with fixed current continuation. Each stage must satisfy the same
      // convergence criterion; a failed stage is never returned as a solution.
      Eigen::VectorXd a = initial;
      int iterations = 0;
      for (int stage = 0; stage <= 8; ++stage) {
        solved = equilibrium(model, mesh, gauge, a, load * (stage / 8.0));
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
    for (size_t i = 0; i < amps.size(); ++i) {
      const auto &source = result.sources[i];
      double linkage = source.coupling.dot(eq.a);
      output.coenergy += amps[i] * linkage;
      output.sources.push_back(
          {source.id, source.wire, source.closed, amps[i],
           source.closed ? linkage : std::numeric_limits<double>::quiet_NaN()});
    }
    const int ns = static_cast<int>(result.sources.size());
    output.differential_inductance = Eigen::MatrixXd::Constant(
        ns, ns, std::numeric_limits<double>::quiet_NaN());
    for (int i = 0; i < ns; ++i)
      for (int j = 0; j < ns; ++j)
        if (result.sources[i].closed && result.sources[j].closed)
          output.differential_inductance(i, j) = 0;
    if (gauge.size > 0 && ns > 0) {
      Factor factor;
#if defined(HAVE_ACCELERATE)
      factor.setOrder(SparseOrderMetis);
#endif
      factor.compute(eq.assembly.stiffness);
      if (factor.info() != Eigen::Success)
        return Error{"Differential inductance factorization failed"};
      for (int j = 0; j < ns; ++j)
        if (result.sources[j].closed) {
          Eigen::VectorXd rhs = Eigen::VectorXd::Zero(gauge.size);
          for (size_t e = 0; e < mesh.dofs(); ++e)
            if (gauge.equations[e] >= 0)
              rhs[gauge.equations[e]] = result.sources[j].coupling[e];
          const Eigen::VectorXd response = factor.solve(rhs);
          if (factor.info() != Eigen::Success || !response.allFinite())
            return Error{"Differential inductance solve failed"};
          for (int i = 0; i < ns; ++i)
            if (result.sources[i].closed) {
              double inductance = 0;
              for (size_t e = 0; e < mesh.dofs(); ++e)
                if (gauge.equations[e] >= 0)
                  inductance += result.sources[i].coupling[e] *
                                response[gauge.equations[e]];
              output.differential_inductance(i, j) = inductance;
            }
        }
    }
    for (const auto &probe : model.probes) {
      Vector b = Vector::Zero(), h = Vector::Zero();
      int count = 0;
      for (int ci : index.query(probe.position, probe.position)) {
        const auto &cell = mesh.cells[ci];
        if (contains(cell, probe.position)) {
          const Vector field = field_at(cell, eq.a, probe.position);
          const auto material = at_point(model, cell.material, probe.position);
          const std::vector<std::pair<double, double>> empty;
          const auto &curve =
              material.bh_table ? model.bh_tables.at(material.bh_table) : empty;
          b += field;
          h += material_response(material, curve, field).h;
          ++count;
        }
      }
      if (count == 0)
        return Error{"MPROBE lies outside the magnetic domain: " +
                     std::to_string(probe.id)};
      output.probe_b.push_back(b / count);
      output.probe_h.push_back(h / count);
    }
    auto force = forces(model, mesh, eq.a);
    if (auto *e = std::get_if<Error>(&force))
      return *e;
    output.forces = std::get<std::vector<ForceResult>>(std::move(force));
    result.cases.push_back(std::move(output));
  }
  return result;
}
} // namespace vibestran::magnetic
