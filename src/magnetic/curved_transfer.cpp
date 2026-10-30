#include "magnetic/solver.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <queue>
#include <set>

namespace vibestran::magnetic {
namespace {
using Rule = std::vector<std::pair<double, double>>;
struct Target {
  const Cell *cell;
  NodeList equations;
};

// Integrate in the source reference domain. Reference-coordinate lines map to
// straight physical lines for all supported first-order source cells. Splitting
// those lines at exact bilinear-face intersections removes the field-element
// discontinuities before numerical integration; no flattened field mesh is
// used.
class RayIntegral {
public:
  RayIntegral(const Source &s, const Cell &c, const Mesh &m,
              const std::vector<int> &candidates)
      : source(s), cell(c), mesh(m), current(c, s.flux), small(gauss_rule(4)),
        large(gauss_rule(8)) {
    if (cell.nodes.size() == 4) {
      // A tetrahedral subdivision has several long diagonals. Integrating
      // along the edge closest to J avoids nearly tangential crossings caused
      // solely by the arbitrary local vertex ordering of the subdivision.
      Vector center = Vector::Zero();
      for (const auto &p : cell.points)
        center += p / 4;
      const Vector direction = current.value(center);
      double best = -1, length = 0;
      std::array<int, 2> chosen{0, 3};
      for (auto edge : local_edges) {
        const Vector d = cell.points[edge[1]] - cell.points[edge[0]];
        const double score = std::pow(d.dot(direction), 2) / d.squaredNorm();
        if (score > best + 1e-12 * direction.squaredNorm() ||
            (std::abs(score - best) <= 1e-12 * direction.squaredNorm() &&
             d.squaredNorm() > length)) {
          best = score;
          length = d.squaredNorm();
          chosen = edge;
        }
      }
      const auto vertices = reference_vertices(cell);
      reference_origin = vertices[chosen[0]];
      int column = 0;
      for (int i = 0; i < 4; ++i)
        if (i != chosen[0] && i != chosen[1])
          reference_frame.col(column++) = vertices[i] - reference_origin;
      reference_frame.col(2) = vertices[chosen[1]] - reference_origin;
      reference_inverse = reference_frame.inverse();
    }
    std::set<int> edge_set, face_set;
    double minimum_edge = std::numeric_limits<double>::infinity();
    for (int i : candidates) {
      const auto &target = mesh.cells[i];
      for (int e : target.edges) {
        edge_set.insert(e);
        const auto edge = mesh.edges[e];
        minimum_edge = std::min(
            minimum_edge, (mesh.nodes[edge[1]] - mesh.nodes[edge[0]]).norm());
      }
      face_set.insert(target.faces.begin(), target.faces.end());
    }
    edges.assign(edge_set.begin(), edge_set.end());
    faces.assign(face_set.begin(), face_set.end());
    std::map<int, int> equation;
    for (size_t i = 0; i < edges.size(); ++i)
      equation[edges[i]] = static_cast<int>(i);
    for (int i : candidates) {
      Target target{&mesh.cells[i], {}};
      for (int e : target.cell->edges)
        target.equations.push_back(equation.at(e));
      targets.push_back(std::move(target));
    }
    double current = 0;
    for (int f : cell.faces)
      current = std::max(current, std::abs(source.flux[f]));
    scale = std::max(
        current * (cell.maximum - cell.minimum).norm() / minimum_edge, 1e-30);
  }

  Result<Eigen::VectorXd> integrate(double tolerance) {
    auto section = [&](double u) {
      return adaptive([&](double v) { return ray(u, v); }, section_cuts(u),
                      tolerance / 16);
    };
    const Eigen::VectorXd value = adaptive(section, outer_cuts(), tolerance);
    if (error)
      return *error;
    if (std::abs(value[edges.size()] - 1) > 1e-8)
      return Error{std::format("Source {} cell {} is outside the magnetic "
                               "domain (covered fraction {})",
                               source.id, cell.id, value[edges.size()])};
    Eigen::VectorXd out = value.head(edges.size()) * scale;
    return out;
  }
  NodeList edges;

private:
  const Source &source;
  const Cell &cell;
  const Mesh &mesh;
  CurrentField current;
  Rule small, large;
  std::vector<Target> targets;
  NodeList faces;
  double scale{1};
  Status error;
  int rays{0};
  Vector reference_origin{Vector::Zero()};
  Matrix reference_frame{Matrix::Identity()},
      reference_inverse{Matrix::Identity()};

  Vector integration_coordinates(const Vector &x) const {
    const Vector r = reference_coordinates(cell, x);
    return cell.nodes.size() == 4
               ? Vector(reference_inverse * (r - reference_origin))
               : r;
  }

  Vector reference(double u, double v, double t) const {
    if (cell.nodes.size() == 8)
      return Vector(2 * u - 1, 2 * v - 1, 2 * t - 1);
    if (cell.nodes.size() == 6)
      return Vector(u, (1 - u) * v, 2 * t - 1);
    return reference_origin +
           reference_frame * Vector(u, (1 - u) * v, (1 - u) * (1 - v) * t);
  }

  static void finish_cuts(std::vector<double> &cuts) {
    cuts.erase(std::remove_if(cuts.begin(), cuts.end(),
                              [](double t) {
                                return !std::isfinite(t) || t < 0 || t > 1;
                              }),
               cuts.end());
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(
        std::unique(cuts.begin(), cuts.end(),
                    [](double a, double b) { return std::abs(a - b) < 1e-12; }),
        cuts.end());
  }

  // Changes in ray ownership occur at source/field boundary crossings and
  // projected field edges. Split there before adaptive integration instead of
  // repeatedly resolving these kinks with tiny rectangular panels.
  std::vector<double> outer_cuts() const {
    std::vector<double> cuts{0, 1};
    auto append = [&](const Vector &x) {
      const Vector r = integration_coordinates(x);
      cuts.push_back(cell.nodes.size() == 8 ? (r.x() + 1) / 2 : r.x());
    };
    for (int e : edges) {
      const auto edge = mesh.edges[e];
      const Vector a = mesh.nodes[edge[0]], b = mesh.nodes[edge[1]];
      for (const auto &x : {a, b})
        if (contains(cell, x))
          append(x);
      for (int f : cell.faces)
        for (double t :
             line_face_intersections(source.mesh, source.mesh.faces[f], a, b))
          append(a + t * (b - a));
    }
    for (int e : cell.edges) {
      const auto edge = source.mesh.edges[e];
      const Vector a = source.mesh.nodes[edge[0]],
                   b = source.mesh.nodes[edge[1]];
      for (int f : faces)
        for (double t : line_face_intersections(mesh, mesh.faces[f], a, b))
          append(a + t * (b - a));
    }
    finish_cuts(cuts);
    return cuts;
  }

  std::vector<double> section_cuts(double u) const {
    std::vector<double> cuts{0, 1};
    Mesh slice;
    slice.nodes = {mapping(cell, reference(u, 0, 0)).first,
                   mapping(cell, reference(u, 1, 0)).first,
                   mapping(cell, reference(u, 1, 1)).first,
                   mapping(cell, reference(u, 0, 1)).first};
    for (int f : faces)
      for (auto side : {std::array<int, 2>{0, 1}, std::array<int, 2>{3, 2}}) {
        auto values = line_face_intersections(
            mesh, mesh.faces[f], slice.nodes[side[0]], slice.nodes[side[1]]);
        cuts.insert(cuts.end(), values.begin(), values.end());
      }
    Face face;
    face.nodes =
        cell.nodes.size() == 4 ? NodeList{0, 1, 3} : NodeList{0, 1, 2, 3};
    face.normal = (slice.nodes[1] - slice.nodes[0])
                      .cross(slice.nodes[3] - slice.nodes[0])
                      .normalized();
    for (int e : edges) {
      const auto edge = mesh.edges[e];
      const Vector a = mesh.nodes[edge[0]], b = mesh.nodes[edge[1]];
      for (double t : line_face_intersections(slice, face, a, b)) {
        const Vector r = integration_coordinates(a + t * (b - a));
        cuts.push_back(cell.nodes.size() == 8 ? (r.y() + 1) / 2
                                              : r.y() / (1 - u));
      }
    }
    finish_cuts(cuts);
    return cuts;
  }

  template <class Function>
  Eigen::VectorXd adaptive(const Function &function,
                           const std::vector<double> &cuts, double tolerance) {
    struct Panel {
      double a, b, estimate;
      int depth;
      Eigen::VectorXd value;
    };
    auto evaluate = [&](double a, double b, int depth) -> Panel {
      auto accumulate = [&](double t, double weight, Eigen::VectorXd &out) {
        out += weight * function(t);
      };
      const Eigen::VectorXd coarse = line_rule(accumulate, a, b, small);
      Eigen::VectorXd fine = line_rule(accumulate, a, b, large);
      const double estimate = (fine - coarse).norm();
      return {a, b, estimate, depth, std::move(fine)};
    };
    auto compare = [](const Panel &a, const Panel &b) {
      return a.estimate < b.estimate;
    };
    std::priority_queue<Panel, std::vector<Panel>, decltype(compare)> panels(
        compare);
    Eigen::VectorXd value = Eigen::VectorXd::Zero(edges.size() + 1);
    double remaining = 0;
    for (size_t i = 1; i < cuts.size() && !error; ++i) {
      auto p = evaluate(cuts[i - 1], cuts[i], 0);
      value += p.value;
      remaining += p.estimate;
      panels.push(std::move(p));
    }
    while (!error && remaining > tolerance * std::max(1., value.norm())) {
      const auto p = panels.top();
      panels.pop();
      if (p.depth >= 24) {
        error = Error{std::format("Curved current integration did not converge "
                                  "in source cell {}; refine WIRESTEP/WIRELAY",
                                  cell.id)};
        break;
      }
      const double mid = (p.a + p.b) / 2;
      value -= p.value;
      remaining = std::max(0., remaining - p.estimate);
      for (auto interval : {std::pair{p.a, mid}, std::pair{mid, p.b}}) {
        auto child = evaluate(interval.first, interval.second, p.depth + 1);
        value += child.value;
        remaining += child.estimate;
        panels.push(std::move(child));
        if (error)
          break;
      }
    }
    return value;
  }

  template <class Function>
  Eigen::VectorXd line_rule(const Function &function, double a, double b,
                            const Rule &rule) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(edges.size() + 1);
    for (auto [t, w] : rule)
      function(a + (b - a) * t, w * (b - a), result);
    return result;
  }
  template <class Function>
  Eigen::VectorXd line_integral(const Function &function, double a, double b,
                                int depth) {
    const Eigen::VectorXd coarse = line_rule(function, a, b, small);
    Eigen::VectorXd fine = line_rule(function, a, b, large);
    if ((fine - coarse).norm() <= 2e-13 * std::max(b - a, fine.norm()) || error)
      return fine;
    if (depth >= 12) {
      error = Error{std::format(
          "Curved current line integration did not converge in source cell {}",
          cell.id)};
      return fine;
    }
    return line_integral(function, a, (a + b) / 2, depth + 1) +
           line_integral(function, (a + b) / 2, b, depth + 1);
  }

  Eigen::VectorXd ray(double u, double v) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(edges.size() + 1);
    if (error)
      return result;
    if (++rays > 200000) {
      error =
          Error{std::format("Curved source integration exceeded its refinement "
                            "budget in cell {}; refine WIRESTEP/WIRELAY",
                            cell.id)};
      return result;
    }
    const Vector first = mapping(cell, reference(u, v, 0)).first;
    const Vector last = mapping(cell, reference(u, v, 1)).first;
    std::vector<double> cuts{0, 1};
    for (int f : faces) {
      auto values = line_face_intersections(mesh, mesh.faces[f], first, last);
      cuts.insert(cuts.end(), values.begin(), values.end());
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(
        std::unique(cuts.begin(), cuts.end(),
                    [](double a, double b) { return std::abs(a - b) < 1e-12; }),
        cuts.end());
    const double reference_weight = cell.nodes.size() == 8 ? 8
                                    : cell.nodes.size() == 6
                                        ? 2 * (1 - u)
                                        : (1 - u) * (1 - u) * (1 - v);
    for (size_t i = 1; i < cuts.size(); ++i) {
      const double a = cuts[i - 1], b = cuts[i];
      if (b - a <= 1e-13)
        continue;
      const Vector mid = first + (a + b) / 2 * (last - first);
      std::vector<const Target *> owners, interior;
      const double margin = 1e-10 * (last - first).norm();
      for (const auto &target : targets) {
        const auto &c = *target.cell;
        if ((mid.array() < c.minimum.array() - margin).any() ||
            (mid.array() > c.maximum.array() + margin).any())
          continue;
        if (!contains(c, mid, 1e-9))
          continue;
        owners.push_back(&target);
        const auto shape = reference_shape(c, reference_coordinates(c, mid));
        if (*std::min_element(shape.values.begin(), shape.values.end()) > 1e-9)
          interior.push_back(&target);
      }
      if (interior.size() > 1) {
        error = Error{"CWIRE intersects overlapping magnetic cells"};
        return result;
      }
      if (!interior.empty())
        owners = std::move(interior);
      if (owners.empty())
        continue;
      auto function = [&](double t, double weight, Eigen::VectorXd &value) {
        const Vector r = reference(u, v, t);
        const auto [position, jacobian] = mapping(cell, r);
        const double measure =
            reference_weight * std::abs(jacobian.determinant());
        const Vector weighted_current =
            weight * reference_weight * current.weighted_reference(r, jacobian);
        for (const auto *target : owners) {
          const auto basis = edge_basis(*target->cell, position);
          for (size_t e = 0; e < basis.size(); ++e)
            value[target->equations[e]] +=
                basis[e].dot(weighted_current) / (scale * owners.size());
        }
        value[edges.size()] += weight * measure / cell.volume;
        if (!value.allFinite())
          error = Error{"Nonfinite mapped current integral"};
      };
      result += line_integral(function, a, b, 0);
      if (error)
        return result;
    }
    return result;
  }
};
} // namespace

Result<Eigen::VectorXd> couple_curved_source(const Source &source,
                                             const Mesh &mesh,
                                             const SpatialIndex &index,
                                             std::span<const int> forbidden) {
  const std::set<int> excluded(forbidden.begin(), forbidden.end());
  std::vector<std::vector<int>> candidates;
  for (const auto &cell : source.mesh.cells) {
    std::vector<int> cells;
    std::set<int> boundary;
    for (int ci : index.query(cell.minimum, cell.maximum)) {
      const auto &target = mesh.cells[ci];
      if ((cell.maximum.array() <= target.minimum.array()).any() ||
          (target.maximum.array() <= cell.minimum.array()).any())
        continue;
      cells.push_back(ci);
      if (excluded.contains(ci)) {
        auto overlap = cell_intersects_interior(source.mesh, cell, target);
        if (auto *e = std::get_if<Error>(&overlap))
          return *e;
        if (std::get<bool>(overlap))
          return Error{"CWIRE overlaps a magnetic material or volumetric COIL"};
      }
      for (int f : target.faces)
        if (mesh.faces[f].cells[1] < 0)
          boundary.insert(f);
    }
    if (cells.empty())
      return Error{"CWIRE lies outside the magnetic domain"};
    for (int f : boundary) {
      auto crossing =
          face_intersects_interior(source.mesh, cell, mesh, mesh.faces[f]);
      if (auto *e = std::get_if<Error>(&crossing))
        return *e;
      if (std::get<bool>(crossing))
        return Error{"CWIRE crosses outside the magnetic domain"};
    }
    candidates.push_back(std::move(cells));
  }
  if (!source.closed) {
    for (size_t i = 0; i < source.mesh.faces.size(); ++i) {
      const auto &face = source.mesh.faces[i];
      if (face.cells[1] >= 0 || std::abs(source.flux[i]) < 1e-12)
        continue;
      for (const auto &q : face_quadrature(source.mesh, face)) {
        bool exterior = false;
        const double margin = 1e-9 * std::sqrt(face.area);
        for (int ci : index.query(q.position - Vector::Constant(margin),
                                  q.position + Vector::Constant(margin)))
          for (int f : mesh.cells[ci].faces)
            if (mesh.faces[f].cells[1] < 0 &&
                point_on_face(mesh, mesh.faces[f], q.position))
              exterior = true;
        if (!exterior)
          return Error{"Open current path must terminate on the magnetic "
                       "domain exterior"};
      }
    }
  }
  for (double tolerance : {2e-10, 2e-12}) {
    Eigen::VectorXd coupling = Eigen::VectorXd::Zero(mesh.dofs());
    for (size_t i = 0; i < source.mesh.cells.size(); ++i) {
      RayIntegral integral(source, source.mesh.cells[i], mesh, candidates[i]);
      auto integrated = integral.integrate(tolerance);
      if (auto *e = std::get_if<Error>(&integrated))
        return *e;
      const auto &local = std::get<Eigen::VectorXd>(integrated);
      for (size_t e = 0; e < integral.edges.size(); ++e)
        coupling[integral.edges[e]] += local[e];
    }
    Eigen::VectorXd balance = Eigen::VectorXd::Zero(mesh.nodes.size());
    for (size_t e = 0; e < mesh.edges.size(); ++e) {
      balance[mesh.edges[e][0]] -= coupling[e];
      balance[mesh.edges[e][1]] += coupling[e];
    }
    for (const auto &f : mesh.faces)
      if (f.cells[1] < 0)
        for (int n : f.nodes)
          balance[n] = 0;
    if (balance.norm() <= 1e-9 * std::max(coupling.norm(), 1e-300))
      return coupling;
  }
  return Error{"Curved source transfer failed current continuity; refine "
               "WIRESTEP/WIRELAY"};
}
} // namespace vibestran::magnetic
