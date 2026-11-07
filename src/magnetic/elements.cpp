#include "magnetic/mesh.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <map>

namespace vibestran::magnetic {
namespace {
constexpr int hex[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                           {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
const std::array<Vector, 3> tri_grad{Vector(-1, -1, 0), Vector(1, 0, 0),
                                     Vector(0, 1, 0)};
} // namespace
Vector reference_coordinates(const Cell &c, const Vector &x) {
  Vector r = c.inverse * (x - c.map_origin);
  if (c.affine)
    return r;
  for (int i = 0; i < 24; ++i) {
    const auto [p, j] = mapping(c, r);
    const Vector step = j.inverse() * (p - x);
    r -= step;
    if (!r.allFinite() || step.norm() < 2e-13)
      break;
  }
  return r;
}

std::vector<std::array<int, 2>> cell_edges(const Cell &c) {
  if (c.planar) {
    std::vector<std::array<int, 2>> e;
    for (int i = 0; i < static_cast<int>(c.nodes.size()); ++i)
      e.push_back({i, (i + 1) % static_cast<int>(c.nodes.size())});
    return e;
  }
  if (c.nodes.size() == 4)
    return {local_edges.begin(), local_edges.end()};
  if (c.nodes.size() == 6)
    return {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5},
            {5, 3}, {0, 3}, {1, 4}, {2, 5}};
  return {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
          {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
}
std::vector<NodeList> cell_faces(const Cell &c) {
  if (c.planar) {
    std::vector<NodeList> f;
    for (auto e : cell_edges(c))
      f.push_back({e[0], e[1]});
    return f;
  }
  if (c.nodes.size() == 4)
    return {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
  if (c.nodes.size() == 6)
    return {{0, 1, 2}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
  return {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
          {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
}
Shape reference_shape(const Cell &c, const Vector &r) {
  Shape s;
  const size_t n = c.nodes.size();
  s.values.resize(n);
  s.gradients.assign(n, Vector::Zero());
  if (!c.planar && n == 4) {
    s.values = {1 - r.sum(), r.x(), r.y(), r.z()};
    s.gradients = {Vector(-1, -1, -1), Vector::UnitX(), Vector::UnitY(),
                   Vector::UnitZ()};
  } else if ((c.planar && n == 3) || n == 6) {
    const double l[3] = {1 - r.x() - r.y(), r.x(), r.y()};
    for (size_t i = 0; i < n; ++i) {
      const double side = i < 3 ? -1 : 1;
      const double z = c.planar ? 1 : (1 + side * r.z()) / 2;
      s.values[i] = l[i % 3] * z;
      s.gradients[i] = tri_grad[i % 3] * z;
      if (!c.planar)
        s.gradients[i].z() = side * l[i % 3] / 2;
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      const double z = c.planar ? 1 : (1 + hex[i][2] * r.z()) / 2;
      s.values[i] = (1 + hex[i][0] * r.x()) * (1 + hex[i][1] * r.y()) * z / 4;
      s.gradients[i].x() = hex[i][0] * (1 + hex[i][1] * r.y()) * z / 4;
      s.gradients[i].y() = hex[i][1] * (1 + hex[i][0] * r.x()) * z / 4;
      if (!c.planar)
        s.gradients[i].z() =
            hex[i][2] * (1 + hex[i][0] * r.x()) * (1 + hex[i][1] * r.y()) / 8;
    }
  }
  return s;
}
void initialize_mapping(Cell &c) {
  auto &v = c.map_coefficients;
  for (auto &x : v)
    x.setZero();
  const size_t n = c.nodes.size();
  if ((!c.planar && n == 4) || n == 3) {
    v[1] = c.points[1] - c.origin;
    v[2] = c.points[2] - c.origin;
    if (!c.planar)
      v[4] = c.points[3] - c.origin;
  } else if (n == 6) {
    v[0] = (c.points[3] - c.origin) / 2;
    v[1] = ((c.points[1] - c.points[0]) + (c.points[4] - c.points[3])) / 2;
    v[2] = ((c.points[2] - c.points[0]) + (c.points[5] - c.points[3])) / 2;
    v[4] = (c.points[3] - c.points[0]) / 2;
    v[5] = ((c.points[4] - c.points[3]) - (c.points[1] - c.points[0])) / 2;
    v[6] = ((c.points[5] - c.points[3]) - (c.points[2] - c.points[0])) / 2;
  } else
    for (size_t i = 0; i < n; ++i)
      for (int mask = 0; mask < (c.planar ? 4 : 8); ++mask) {
        double coefficient = 1. / static_cast<double>(n);
        for (int axis = 0; axis < 3; ++axis)
          if (mask & (1 << axis))
            coefficient *= hex[i][axis];
        v[mask] += coefficient * (c.points[i] - c.origin);
      }
}
std::pair<Vector, Matrix> mapping(const Cell &c, const Vector &r) {
  // Cached tensor-polynomial geometry avoids shape-vector allocation and a
  // full nodal Jacobian assembly at every inverse-map Newton iteration.
  const auto &v = c.map_coefficients;
  const double x = r.x(), y = r.y(), z = r.z();
  const Vector p = c.origin + v[0] + x * v[1] + y * v[2] + z * v[4] +
                   x * y * v[3] + x * z * v[5] + y * z * v[6] +
                   x * y * z * v[7];
  Matrix j;
  j.col(0) = v[1] + y * v[3] + z * v[5] + y * z * v[7];
  j.col(1) = v[2] + x * v[3] + z * v[6] + x * z * v[7];
  j.col(2) =
      c.planar ? c.normal : Vector(v[4] + x * v[5] + y * v[6] + x * y * v[7]);
  return {p, j};
}
std::array<double, 4> barycentric(const Cell &c, const Vector &x) {
  const Vector r = reference_coordinates(c, x);
  return {1 - r.sum(), r.x(), r.y(), r.z()};
}
bool contains(const Cell &c, const Vector &x, double tol) {
  const Vector r = reference_coordinates(c, x);
  if (!r.allFinite())
    return false;
  if ((mapping(c, r).first - x).norm() > tol * (c.maximum - c.minimum).norm())
    return false;
  if (c.planar && std::abs(r.z()) > tol)
    return false;
  if (!c.planar && c.nodes.size() == 4)
    return r.minCoeff() >= -tol && r.sum() <= 1 + tol;
  if (c.nodes.size() == 3 || c.nodes.size() == 6)
    return r.x() >= -tol && r.y() >= -tol && r.x() + r.y() <= 1 + tol &&
           std::abs(r.z()) <= 1 + tol;
  return r.cwiseAbs().maxCoeff() <= 1 + tol;
}
std::vector<Vector> edge_basis(const Cell &c, const Vector &x) {
  if (!c.planar && c.nodes.size() == 4) {
    const auto l = barycentric(c, x);
    std::vector<Vector> n(6);
    for (int e = 0; e < 6; ++e) {
      const auto ij = local_edges[e];
      n[e] = c.edge_sign[e] *
             (l[ij[0]] * c.gradient[ij[1]] - l[ij[1]] * c.gradient[ij[0]]);
    }
    return n;
  }
  const Vector r = reference_coordinates(c, x);
  if (c.planar) {
    auto s = reference_shape(c, r);
    std::vector<Vector> n;
    for (double v : s.values)
      n.push_back(v * c.normal);
    return n;
  }
  const Matrix transform =
      c.affine ? c.inverse.transpose().eval()
               : mapping(c, r).second.inverse().transpose().eval();
  const auto edges = cell_edges(c);
  std::vector<Vector> n(edges.size(), Vector::Zero());
  for (size_t e = 0; e < edges.size(); ++e) {
    const auto ij = edges[e];
    if (c.nodes.size() == 8) {
      int axis = 0;
      while (hex[ij[0]][axis] == hex[ij[1]][axis])
        ++axis;
      const int b = (axis + 1) % 3, d = (axis + 2) % 3;
      n[e][axis] = hex[ij[1]][axis] * (1 + hex[ij[0]][b] * r[b]) *
                   (1 + hex[ij[0]][d] * r[d]) / 8;
    } else {
      const double l[3] = {1 - r.x() - r.y(), r.x(), r.y()};
      if (e < 6) {
        int i = ij[0] % 3, j = ij[1] % 3;
        n[e] = (l[i] * tri_grad[j] - l[j] * tri_grad[i]) *
               (1 + (e < 3 ? -1 : 1) * r.z()) / 2;
      } else
        n[e].z() = l[ij[0]] / 2;
    }
    n[e] = c.edge_sign[e] * transform * n[e];
  }
  return n;
}
std::vector<Vector> reference_curls(const Cell &c, const Vector &r) {
  if (c.planar)
    return reference_shape(c, r).gradients;
  const auto edges = cell_edges(c);
  std::vector<Vector> n(edges.size(), Vector::Zero());
  if (c.nodes.size() == 4) {
    // Whitney: curl(l_i grad l_j - l_j grad l_i) = 2 grad l_i x grad l_j.
    const auto g = reference_shape(c, r).gradients;
    for (size_t e = 0; e < edges.size(); ++e)
      n[e] = 2.0 * g[edges[e][0]].cross(g[edges[e][1]]);
    return n;
  }
  for (size_t e = 0; e < edges.size(); ++e) {
    const auto ij = edges[e];
    if (c.nodes.size() == 8) {
      int a = 0;
      while (hex[ij[0]][a] == hex[ij[1]][a])
        ++a;
      const int b = (a + 1) % 3, d = (a + 2) % 3;
      n[e][b] = hex[ij[1]][a] * hex[ij[0]][d] * (1 + hex[ij[0]][b] * r[b]) / 8;
      n[e][d] = -hex[ij[1]][a] * hex[ij[0]][b] * (1 + hex[ij[0]][d] * r[d]) / 8;
    } else if (e < 6) {
      const int i = ij[0] % 3, k = ij[1] % 3, side = e < 3 ? -1 : 1;
      const double l[3] = {1 - r.x() - r.y(), r.x(), r.y()};
      const Vector w = l[i] * tri_grad[k] - l[k] * tri_grad[i];
      n[e] = Vector(-side * w.y() / 2, side * w.x() / 2,
                    tri_grad[i].cross(tri_grad[k]).z() * (1 + side * r.z()));
    } else
      n[e] = Vector(tri_grad[ij[0]].y() / 2, -tri_grad[ij[0]].x() / 2, 0);
  }
  return n;
}
Matrix curl_map(const Cell &c, const Matrix &j) {
  if (!c.planar)
    return j / j.determinant(); // Piola: curl -> J curl / det J
  // grad N = J^-T g, and the curl of N n is grad N x n, linear in g.
  const Matrix inverse = j.inverse().transpose();
  Matrix out;
  for (int k = 0; k < 3; ++k)
    out.col(k) = inverse.col(k).cross(c.normal);
  return out;
}
std::vector<Vector> edge_curls(const Cell &c, const Vector &x) {
  if (!c.planar && c.nodes.size() == 4)
    return c.curl;
  const Vector r = reference_coordinates(c, x);
  const Matrix map = curl_map(c, c.affine ? c.jacobian : mapping(c, r).second);
  auto n = reference_curls(c, r);
  for (size_t e = 0; e < n.size(); ++e)
    n[e] = (c.planar ? 1.0 : c.edge_sign[e]) * (map * n[e]);
  if (c.axisymmetric) {
    // curl(N e_phi) = grad N x e_phi + (N / r) e_z. On the axis A_phi = 0,
    // so A / r is dA/dr there, which the planar term carries along e_z.
    const double radius_x = radius(c, x);
    if (radius_x > 1e-12 * (c.maximum - c.minimum).norm()) {
      const auto values = reference_shape(c, r).values;
      for (size_t e = 0; e < n.size(); ++e)
        n[e] += values[e] / radius_x * c.axis;
    } else {
      for (size_t e = 0; e < n.size(); ++e)
        n[e] += n[e].dot(c.axis) * c.axis;
    }
  }
  return n;
}
std::vector<Vector> flux_basis(const Mesh &, const Cell &c, const Vector &x) {
  std::vector<Vector> n(c.faces.size(), Vector::Zero());
  if (c.nodes.size() == 4) {
    for (int i = 0; i < 4; ++i)
      n[i] = c.face_sign[i] * (x - c.points[i]) / (3 * c.volume);
    return n;
  }
  const Vector r = reference_coordinates(c, x);
  const Matrix j = c.affine ? c.jacobian : mapping(c, r).second;
  if (c.nodes.size() == 8) {
    constexpr int axes[6] = {2, 2, 1, 0, 1, 0},
                  sides[6] = {-1, 1, -1, 1, 1, -1};
    for (int i = 0; i < 6; ++i)
      n[i][axes[i]] = (r[axes[i]] + sides[i]) / 8;
  } else {
    n[0].z() = r.z() - 1;
    n[1].z() = r.z() + 1;
    n[2] = Vector(r.x(), r.y() - 1, 0) / 2;
    n[3] = Vector(r.x(), r.y(), 0) / 2;
    n[4] = Vector(r.x() - 1, r.y(), 0) / 2;
  }
  for (size_t i = 0; i < n.size(); ++i)
    n[i] = c.face_sign[i] * j * n[i] / std::abs(j.determinant());
  return n;
}
Vector flux_value(const Cell &c, const Eigen::VectorXd &flux, const Vector &x) {
  Vector value = Vector::Zero();
  if (c.nodes.size() == 4) {
    for (int i = 0; i < 4; ++i)
      value += c.face_sign[i] * flux[c.faces[i]] * (x - c.points[i]) /
               (3 * c.volume);
    return value;
  }
  const Vector r = reference_coordinates(c, x);
  const Matrix j = c.affine ? c.jacobian : mapping(c, r).second;
  if (c.nodes.size() == 8) {
    constexpr int axes[6] = {2, 2, 1, 0, 1, 0},
                  sides[6] = {-1, 1, -1, 1, 1, -1};
    for (int i = 0; i < 6; ++i)
      value[axes[i]] +=
          c.face_sign[i] * flux[c.faces[i]] * (r[axes[i]] + sides[i]) / 8;
  } else {
    const double q0 = c.face_sign[0] * flux[c.faces[0]],
                 q1 = c.face_sign[1] * flux[c.faces[1]],
                 q2 = c.face_sign[2] * flux[c.faces[2]],
                 q3 = c.face_sign[3] * flux[c.faces[3]],
                 q4 = c.face_sign[4] * flux[c.faces[4]];
    value =
        Vector(((q2 + q3 + q4) * r.x() - q4) / 2,
               ((q2 + q3 + q4) * r.y() - q2) / 2, (q0 + q1) * r.z() + q1 - q0);
  }
  return j * value / std::abs(j.determinant());
}
CurrentField::CurrentField(const Cell &cell, const Eigen::VectorXd &flux)
    : cell_(cell) {
  std::array<double, 6> q{};
  for (size_t i = 0; i < cell.faces.size(); ++i)
    q[i] = cell.face_sign[i] * flux[cell.faces[i]];
  if (cell.nodes.size() == 4) {
    offset_ = -2 * Vector(q[1], q[2], q[3]);
    slope_ = Vector::Constant(2 * (q[0] + q[1] + q[2] + q[3]));
  } else if (cell.nodes.size() == 8) {
    offset_ = Vector(q[3] - q[5], q[4] - q[2], q[1] - q[0]) / 8;
    slope_ = Vector(q[3] + q[5], q[4] + q[2], q[1] + q[0]) / 8;
  } else {
    offset_ = Vector(-q[4] / 2, -q[2] / 2, q[1] - q[0]);
    slope_ =
        Vector((q[2] + q[3] + q[4]) / 2, (q[2] + q[3] + q[4]) / 2, q[0] + q[1]);
  }
  if (cell.affine) {
    const Matrix transform =
        cell.jacobian / std::abs(cell.jacobian.determinant());
    affine_offset_ = transform * offset_;
    affine_slope_ = transform * slope_.asDiagonal() * cell.inverse;
  }
}
Vector CurrentField::weighted_reference(const Vector &r,
                                        const Matrix &jacobian) const {
  return jacobian * (offset_ + slope_.cwiseProduct(r));
}
Vector CurrentField::value(const Vector &x) const {
  if (cell_.affine)
    return affine_offset_ + affine_slope_ * (x - cell_.map_origin);
  const Vector r = reference_coordinates(cell_, x);
  const Matrix j = mapping(cell_, r).second;
  return weighted_reference(r, j) / std::abs(j.determinant());
}

std::vector<Sample> reference_quadrature(const Cell &c) {
  if (!c.planar && c.nodes.size() == 4) {
    constexpr double a = 0.58541019662496845446, b = 0.13819660112501051518;
    return {{Vector(b, b, b), 1. / 24},
            {Vector(a, b, b), 1. / 24},
            {Vector(b, a, b), 1. / 24},
            {Vector(b, b, a), 1. / 24}};
  }
  const double p[3] = {-std::sqrt(0.6), 0, std::sqrt(0.6)},
               w[3] = {5. / 9, 8. / 9, 5. / 9};
  std::vector<Sample> out;
  if (c.nodes.size() == 3 || c.nodes.size() == 6) {
    // Degree-four triangle rule (positive weights), tensor Gauss in thickness.
    for (int z = 0; z < (c.planar ? 1 : 3); ++z)
      for (int family = 0; family < 2; ++family) {
        const double a = family ? .091576213509770743 : .445948490915964886;
        const double weight =
            family ? .054975871827660934 : .111690794839005733;
        for (Vector r : {Vector(a, a, 0), Vector(1 - 2 * a, a, 0),
                         Vector(a, 1 - 2 * a, 0)}) {
          r.z() = c.planar ? 0 : p[z];
          out.push_back({r, weight * (c.planar ? 1 : w[z])});
        }
      }
  } else
    for (int k = 0; k < (c.planar ? 1 : 3); ++k)
      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
          out.push_back({Vector(p[i], p[j], c.planar ? 0 : p[k]),
                         w[i] * w[j] * (c.planar ? 1 : w[k])});
  return out;
}
std::vector<Sample> quadrature(const Mesh &, const Cell &c) {
  auto q = reference_quadrature(c);
  for (auto &s : q) {
    auto [p, j] = mapping(c, s.position);
    s.position = p;
    s.weight *= std::abs(j.determinant());
  }
  return q;
}

CellGeometry::CellGeometry(const Mesh &mesh) {
  std::map<std::pair<bool, size_t>, int> kinds; // (planar, nodes) -> rule
  cells_.reserve(mesh.cells.size());
  for (const auto &c : mesh.cells) {
    const auto [kind, added] = kinds.emplace(std::pair{c.planar, c.nodes.size()},
                                             static_cast<int>(rules_.size()));
    if (added) {
      Rule rule;
      for (const auto &s : reference_quadrature(c)) {
        rule.points.push_back(s.position);
        rule.weights.push_back(s.weight);
        const auto reference = reference_curls(c, s.position);
        rule.functions = static_cast<int>(reference.size());
        rule.curls.insert(rule.curls.end(), reference.begin(), reference.end());
        if (c.planar) {
          const auto values = reference_shape(c, s.position).values;
          rule.values.insert(rule.values.end(), values.begin(), values.end());
        }
      }
      rules_.push_back(std::move(rule));
    }
    const Rule &rule = rules_[static_cast<size_t>(kind->second)];
    cells_.push_back({kind->second, c.affine, maps_.size(), weights_.size()});
    // An affine cell's map is the one edge_curls uses, from its Jacobian.
    if (c.affine)
      maps_.push_back(curl_map(c, c.jacobian));
    for (size_t q = 0; q < rule.points.size(); ++q) {
      const auto [x, j] = mapping(c, rule.points[q]);
      weights_.push_back(rule.weights[q] * std::abs(j.determinant()));
      measures_.push_back(out_of_plane(mesh, x));
      if (c.axisymmetric)
        inverse_radii_.push_back(1 / radius(c, x));
      if (!c.affine)
        maps_.push_back(curl_map(c, j));
    }
  }
}
int CellGeometry::points(int cell) const {
  return static_cast<int>(rules_[static_cast<size_t>(cells_[static_cast<size_t>(cell)].rule)]
                              .points.size());
}
double CellGeometry::weight(int cell, int point) const {
  return weights_[cells_[static_cast<size_t>(cell)].weight + static_cast<size_t>(point)];
}
double CellGeometry::measure(int cell, int point) const {
  return measures_[cells_[static_cast<size_t>(cell)].weight + static_cast<size_t>(point)];
}
std::span<const double> CellGeometry::nodal_values(int cell, int point) const {
  const Rule &rule = rules_[cells_[static_cast<size_t>(cell)].rule];
  const auto count = static_cast<size_t>(rule.functions);
  return std::span<const double>(rule.values).subspan(static_cast<size_t>(point) * count, count);
}
void CellGeometry::curls(const Mesh &mesh, int cell, int point,
                         std::span<Vector> out) const {
  const Entry &entry = cells_[static_cast<size_t>(cell)];
  const Rule &rule = rules_[static_cast<size_t>(entry.rule)];
  const Cell &c = mesh.cells[static_cast<size_t>(cell)];
  const Matrix &map =
      maps_[entry.map + (entry.affine ? 0 : static_cast<size_t>(point))];
  const auto first = static_cast<size_t>(point) * static_cast<size_t>(rule.functions);
  for (size_t e = 0; e < static_cast<size_t>(rule.functions); ++e)
    out[e] = (c.planar ? 1.0 : c.edge_sign[e]) * (map * rule.curls[first + e]);
  if (c.axisymmetric) {
    // Every cell of an axisymmetric mesh is, so the radii index as weights.
    const double inverse =
        inverse_radii_[entry.weight + static_cast<size_t>(point)];
    for (size_t e = 0; e < static_cast<size_t>(rule.functions); ++e)
      out[e] += rule.values[first + e] * inverse * c.axis;
  }
}
Vector CellGeometry::position(const Mesh &mesh, int cell, int point) const {
  const Entry &entry = cells_[static_cast<size_t>(cell)];
  return mapping(mesh.cells[static_cast<size_t>(cell)],
                 rules_[static_cast<size_t>(entry.rule)].points[static_cast<size_t>(point)])
      .first;
}
} // namespace vibestran::magnetic
