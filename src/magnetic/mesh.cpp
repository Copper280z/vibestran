#include "magnetic/mesh.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <set>

namespace vibestran::magnetic {
namespace {
struct Disjoint {
  std::vector<int> p;
  explicit Disjoint(int n) : p(n) { std::iota(p.begin(), p.end(), 0); }
  int root(int a) {
    while (p[a] != a) {
      p[a] = p[p[a]];
      a = p[a];
    }
    return a;
  }
  bool join(int a, int b) {
    a = root(a);
    b = root(b);
    if (a == b)
      return false;
    p[a] = b;
    return true;
  }
};
std::array<Sample, 4> tetra_quadrature(const std::array<Vector, 4> &p,
                                       double volume) {
  constexpr double a = 0.58541019662496845446, b = 0.13819660112501051518;
  std::array<Sample, 4> out;
  for (int i = 0; i < 4; ++i) {
    out[i].position = Vector::Zero();
    out[i].weight = volume / 4;
    for (int j = 0; j < 4; ++j)
      out[i].position += (i == j ? a : b) * p[j];
  }
  return out;
}
} // namespace

Result<Mesh> make_mesh(std::vector<Vector> nodes,
                       const std::vector<std::array<int, 4>> &tetra) {
  Mesh mesh;
  mesh.nodes = std::move(nodes);
  std::map<std::array<int, 2>, int> edges;
  std::set<std::array<int, 4>> unique;
  for (const auto &indices : tetra) {
    Cell c;
    c.nodes = indices;
    c.id = static_cast<int>(mesh.cells.size()) + 1;
    auto key = indices;
    std::sort(key.begin(), key.end());
    if (key[0] < 0 || key[3] >= static_cast<int>(mesh.nodes.size()) ||
        std::adjacent_find(key.begin(), key.end()) != key.end() ||
        !unique.insert(key).second)
      return Error{std::format("Invalid or duplicate tetrahedron {}", c.id)};
    c.origin = mesh.nodes[indices[0]];
    Matrix j;
    for (int i = 0; i < 3; ++i)
      j.col(i) = mesh.nodes[indices[i + 1]] - c.origin;
    const double det = j.determinant();
    if (!j.allFinite() ||
        std::abs(det) <= 1e-14 * std::pow(j.colwise().norm().maxCoeff(), 3))
      return Error{std::format("Degenerate tetrahedron {}", c.id)};
    c.volume = std::abs(det) / 6;
    c.inverse = j.inverse();
    c.gradient[0] = Vector::Zero();
    for (int i = 1; i < 4; ++i) {
      c.gradient[i] = c.inverse.row(i - 1).transpose();
      c.gradient[0] -= c.gradient[i];
    }
    c.minimum = c.maximum = c.origin;
    for (int n : indices) {
      c.minimum = c.minimum.cwiseMin(mesh.nodes[n]);
      c.maximum = c.maximum.cwiseMax(mesh.nodes[n]);
    }
    for (int e = 0; e < 6; ++e) {
      const auto ij = local_edges[e];
      std::array<int, 2> edge{indices[ij[0]], indices[ij[1]]};
      c.edge_sign[e] = edge[0] < edge[1] ? 1 : -1;
      std::sort(edge.begin(), edge.end());
      const auto [it, inserted] =
          edges.emplace(edge, static_cast<int>(edges.size()));
      if (inserted)
        mesh.edges.push_back(edge);
      c.edges[e] = it->second;
      c.curl[e] =
          2.0 * c.edge_sign[e] * c.gradient[ij[0]].cross(c.gradient[ij[1]]);
    }
    for (int opposite = 0; opposite < 4; ++opposite) {
      std::array<int, 3> face{};
      int k = 0;
      for (int i = 0; i < 4; ++i)
        if (i != opposite)
          face[k++] = indices[i];
      std::sort(face.begin(), face.end());
      auto [it, inserted] =
          mesh.face_index.emplace(face, static_cast<int>(mesh.faces.size()));
      if (inserted) {
        Face f;
        f.nodes = face;
        f.normal = (mesh.nodes[face[1]] - mesh.nodes[face[0]])
                       .cross(mesh.nodes[face[2]] - mesh.nodes[face[0]]);
        f.area = f.normal.norm() / 2;
        f.normal.normalize();
        f.center =
            (mesh.nodes[face[0]] + mesh.nodes[face[1]] + mesh.nodes[face[2]]) /
            3;
        mesh.faces.push_back(f);
      }
      auto &f = mesh.faces[it->second];
      const int ci = static_cast<int>(mesh.cells.size());
      const int sign =
          f.normal.dot(f.center - mesh.nodes[indices[opposite]]) > 0 ? 1 : -1;
      if (f.cells[1] >= 0)
        return Error{"Non-manifold face shared by more than two tetrahedra"};
      if (f.cells[0] < 0)
        f.cells[0] = ci;
      else {
        const auto &previous = mesh.cells[f.cells[0]];
        auto prior =
            std::find(previous.faces.begin(), previous.faces.end(), it->second);
        if (previous.face_sign[prior - previous.faces.begin()] == sign)
          return Error{
              "Overlapping tetrahedra on the same side of a shared face"};
        f.cells[1] = ci;
      }
      c.faces[opposite] = it->second;
      c.face_sign[opposite] = sign;
    }
    mesh.cells.push_back(std::move(c));
  }
  if (mesh.cells.empty())
    return Error{"Magnetic mesh contains no tetrahedra"};
  return mesh;
}

Result<Mesh> make_mesh(const Model &model) {
  std::vector<Vector> nodes;
  std::map<int, int> index;
  std::set<int> active;
  for (const auto &c : model.elements)
    for (int n : c.nodes)
      active.insert(n);
  for (int id : active) {
    auto it = model.nodes.find(id);
    if (it == model.nodes.end())
      return Error{std::format("Undefined GRID {}", id)};
    index[id] = static_cast<int>(nodes.size());
    nodes.push_back(it->second.position);
  }
  std::vector<std::array<int, 4>> cells;
  for (const auto &c : model.elements) {
    std::array<int, 4> ids{};
    for (int i = 0; i < 4; ++i)
      ids[i] = index.at(c.nodes[i]);
    cells.push_back(ids);
  }
  auto result = make_mesh(std::move(nodes), cells);
  if (auto *error = std::get_if<Error>(&result))
    return *error;
  auto mesh = std::get<Mesh>(std::move(result));
  mesh.node_index = index;
  mesh.node_ids.assign(active.begin(), active.end());
  for (size_t i = 0; i < mesh.cells.size(); ++i) {
    mesh.cells[i].id = model.elements[i].id;
    auto prop = model.properties.find(model.elements[i].property);
    if (prop == model.properties.end())
      return Error{std::format("CTETRA {} references undefined PSOLID {}",
                               model.elements[i].id,
                               model.elements[i].property)};
    mesh.cells[i].material = prop->second;
    if (!mesh.element_index.emplace(mesh.cells[i].id, static_cast<int>(i))
             .second)
      return Error{"Duplicate CTETRA id"};
  }
  return mesh;
}

std::array<double, 4> barycentric(const Cell &c, const Vector &x) {
  Vector b = c.inverse * (x - c.origin);
  return {1 - b.sum(), b.x(), b.y(), b.z()};
}
std::array<Vector, 6> edge_basis(const Cell &c, const Vector &x) {
  const auto l = barycentric(c, x);
  std::array<Vector, 6> n;
  for (int e = 0; e < 6; ++e) {
    const auto ij = local_edges[e];
    n[e] = c.edge_sign[e] *
           (l[ij[0]] * c.gradient[ij[1]] - l[ij[1]] * c.gradient[ij[0]]);
  }
  return n;
}
std::array<Vector, 4> flux_basis(const Mesh &m, const Cell &c,
                                 const Vector &x) {
  std::array<Vector, 4> n;
  for (int i = 0; i < 4; ++i)
    n[i] = c.face_sign[i] * (x - m.nodes[c.nodes[i]]) / (3 * c.volume);
  return n;
}
std::array<Sample, 4> quadrature(const Mesh &m, const Cell &c) {
  std::array<Vector, 4> p;
  for (int i = 0; i < 4; ++i)
    p[i] = m.nodes[c.nodes[i]];
  return tetra_quadrature(p, c.volume);
}
Result<std::vector<std::pair<int, int>>>
selected_faces(const Model &model, const Mesh &mesh, int selection) {
  auto it = model.face_sets.find(selection);
  if (it == model.face_sets.end() || it->second.empty())
    return Error{std::format("Undefined or empty FSET {}", selection)};
  std::vector<std::pair<int, int>> out;
  std::set<int> seen;
  for (const auto &s : it->second) {
    auto ci = mesh.element_index.find(s.element);
    if (ci == mesh.element_index.end())
      return Error{std::format("FSET {} references undefined CTETRA {}",
                               selection, s.element)};
    std::array<int, 3> ids{};
    for (int i = 0; i < 3; ++i) {
      auto n = mesh.node_index.find(s.nodes[i]);
      if (n == mesh.node_index.end())
        return Error{std::format("FSET {} references undefined volume grid {}",
                                 selection, s.nodes[i])};
      ids[i] = n->second;
    }
    auto sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    auto f = mesh.face_index.find(sorted);
    if (f == mesh.face_index.end())
      return Error{std::format("FSET {} is not a tetrahedral face", selection)};
    const auto &face = mesh.faces[f->second];
    if (face.cells[0] != ci->second && face.cells[1] != ci->second)
      return Error{"FSET face does not belong to its specified element"};
    if (!seen.insert(f->second).second)
      return Error{"Duplicate geometric face in FSET"};
    const Vector normal = (mesh.nodes[ids[1]] - mesh.nodes[ids[0]])
                              .cross(mesh.nodes[ids[2]] - mesh.nodes[ids[0]]);
    out.emplace_back(f->second, normal.dot(face.normal) > 0 ? 1 : -1);
  }
  return out;
}

Result<Gauge> make_gauge(const Mesh &mesh) {
  const int n = static_cast<int>(mesh.nodes.size());
  Disjoint connected(n), boundary(n), tree(n);
  Gauge gauge;
  gauge.equations.assign(mesh.edges.size(), -1);
  gauge.boundary.assign(mesh.edges.size(), false);
  std::set<std::array<int, 2>> bedges;
  std::set<int> bnodes;
  int bf = 0;
  for (const auto &f : mesh.faces)
    if (f.cells[1] < 0) {
      ++bf;
      for (int i = 0; i < 3; ++i) {
        int a = f.nodes[i], b = f.nodes[(i + 1) % 3];
        boundary.join(a, b);
        bnodes.insert(a);
        if (a > b)
          std::swap(a, b);
        bedges.insert({a, b});
      }
    }
  for (const auto &e : mesh.edges)
    connected.join(e[0], e[1]);
  for (int i = 1; i < n; ++i)
    if (connected.root(i) != connected.root(0))
      return Error{"Magnetic domain must be connected"};
  if (bnodes.empty())
    return Error{"Magnetic mesh has no exterior boundary"};
  const int root = *bnodes.begin();
  for (int v : bnodes) {
    if (boundary.root(v) != boundary.root(root))
      return Error{"Disconnected exterior boundaries are not supported by the "
                   "magnetic gauge"};
    tree.join(root, v);
  }
  if (static_cast<long>(bnodes.size()) - static_cast<long>(bedges.size()) +
              bf !=
          2 ||
      static_cast<long>(mesh.nodes.size()) -
              static_cast<long>(mesh.edges.size()) +
              static_cast<long>(mesh.faces.size()) -
              static_cast<long>(mesh.cells.size()) !=
          1)
    return Error{
        "Magnetic gauge requires a simply connected, filled volume mesh"};
  for (size_t i = 0; i < mesh.edges.size(); ++i) {
    const auto e = mesh.edges[i];
    if (bedges.contains(e))
      gauge.boundary[i] = true;
    else if (!tree.join(e[0], e[1]))
      gauge.equations[i] = gauge.size++;
  }
  return gauge;
}

SpatialIndex::SpatialIndex(const Mesh &mesh) : order_(mesh.cells.size()) {
  std::iota(order_.begin(), order_.end(), 0);
  build(mesh, 0, static_cast<int>(order_.size()));
}
int SpatialIndex::build(const Mesh &mesh, int begin, int end) {
  const int id = static_cast<int>(tree_.size());
  Node n;
  n.begin = begin;
  n.end = end;
  n.lo = Vector::Constant(std::numeric_limits<double>::infinity());
  n.hi = -n.lo;
  for (int i = begin; i < end; ++i) {
    n.lo = n.lo.cwiseMin(mesh.cells[order_[i]].minimum);
    n.hi = n.hi.cwiseMax(mesh.cells[order_[i]].maximum);
  }
  tree_.push_back(n);
  if (end - begin > 8) {
    Eigen::Index axis;
    (n.hi - n.lo).maxCoeff(&axis);
    const int middle = (begin + end) / 2;
    std::nth_element(
        order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
        [&](int a, int b) {
          return mesh.cells[a].minimum[axis] + mesh.cells[a].maximum[axis] <
                 mesh.cells[b].minimum[axis] + mesh.cells[b].maximum[axis];
        });
    int left = build(mesh, begin, middle), right = build(mesh, middle, end);
    tree_[id].left = left;
    tree_[id].right = right;
  }
  return id;
}
std::vector<int> SpatialIndex::query(const Vector &lo, const Vector &hi) const {
  std::vector<int> result, stack{0};
  while (!stack.empty()) {
    int id = stack.back();
    stack.pop_back();
    const auto &n = tree_[id];
    if ((n.lo.array() > hi.array()).any() || (n.hi.array() < lo.array()).any())
      continue;
    if (n.left < 0)
      for (int i = n.begin; i < n.end; ++i)
        result.push_back(order_[i]);
    else {
      stack.push_back(n.left);
      stack.push_back(n.right);
    }
  }
  return result;
}

std::vector<Sample> intersection_quadrature(const Mesh &a, const Cell &ca,
                                            const Mesh &b, const Cell &cb) {
  using Polygon = std::vector<Vector>;
  std::vector<Polygon> faces;
  for (int i = 0; i < 4; ++i) {
    Polygon p;
    for (int j = 0; j < 4; ++j)
      if (j != i)
        p.push_back(a.nodes[ca.nodes[j]]);
    faces.push_back(std::move(p));
  }
  const double scale = std::max((ca.maximum - ca.minimum).norm(),
                                (cb.maximum - cb.minimum).norm());
  const double tolerance =
      64 * std::numeric_limits<double>::epsilon() *
      std::max(scale, std::max(ca.origin.norm(), cb.origin.norm()));
  for (int plane = 0; plane < 4 && !faces.empty(); ++plane) {
    const auto &face = b.faces[cb.faces[plane]];
    const Vector normal = cb.face_sign[plane] * face.normal;
    std::vector<Polygon> clipped;
    Polygon cap;
    for (const auto &poly : faces) {
      Polygon next;
      for (size_t i = 0; i < poly.size(); ++i) {
        const Vector &p = poly[i];
        const Vector &q = poly[(i + 1) % poly.size()];
        double dp = normal.dot(p - face.center),
               dq = normal.dot(q - face.center);
        if (std::abs(dp) < tolerance)
          dp = 0;
        if (std::abs(dq) < tolerance)
          dq = 0;
        if (dp <= 0)
          next.push_back(p);
        if ((dp < 0 && dq > 0) || (dp > 0 && dq < 0)) {
          Vector x = p + dp / (dp - dq) * (q - p);
          next.push_back(x);
          cap.push_back(x);
        } else if (dp == 0)
          cap.push_back(p);
      }
      Polygon clean;
      std::copy_if(next.begin(), next.end(), std::back_inserter(clean),
                   [&](const Vector &p) {
                     return clean.empty() ||
                            (p - clean.back()).norm() > tolerance;
                   });
      if (clean.size() > 1 &&
          (clean.front() - clean.back()).norm() <= tolerance)
        clean.pop_back();
      if (clean.size() >= 3)
        clipped.push_back(std::move(clean));
    }
    Polygon unique;
    std::copy_if(cap.begin(), cap.end(), std::back_inserter(unique),
                 [&](const Vector &p) {
                   return std::none_of(unique.begin(), unique.end(),
                                       [&](const Vector &q) {
                                         return (p - q).norm() <= tolerance;
                                       });
                 });
    // Only introduce a cap if the plane actually cut the polyhedron. Coplanar
    // existing faces must not be duplicated (that would double integration).
    bool outside = false, inside = false;
    for (const auto &p : faces)
      for (const auto &x : p) {
        const double d = normal.dot(x - face.center);
        outside |= d > tolerance;
        inside |= d < -tolerance;
      }
    if (outside && inside && unique.size() >= 3) {
      Vector center = std::accumulate(
          unique.begin(), unique.end(), Vector(Vector::Zero()),
          [](Vector sum, const Vector &p) -> Vector { return sum + p; });
      center /= static_cast<double>(unique.size());
      Vector u = normal.unitOrthogonal(), v = normal.cross(u);
      std::sort(unique.begin(), unique.end(),
                [&](const Vector &p, const Vector &q) {
                  return std::atan2((p - center).dot(v), (p - center).dot(u)) <
                         std::atan2((q - center).dot(v), (q - center).dot(u));
                });
      clipped.push_back(std::move(unique));
    }
    faces = std::move(clipped);
  }
  std::vector<Sample> out;
  Vector center = Vector::Zero();
  size_t count = 0;
  for (const auto &f : faces)
    for (const auto &p : f) {
      center += p;
      ++count;
    }
  if (count == 0)
    return out;
  center /= static_cast<double>(count);
  for (const auto &f : faces)
    for (size_t i = 1; i + 1 < f.size(); ++i) {
      const double volume =
          std::abs(
              (f[0] - center).dot((f[i] - center).cross(f[i + 1] - center))) /
          6;
      if (volume <= 1e-16 * std::min(ca.volume, cb.volume))
        continue;
      auto samples = tetra_quadrature({center, f[0], f[i], f[i + 1]}, volume);
      out.insert(out.end(), samples.begin(), samples.end());
    }
  return out;
}
} // namespace vibestran::magnetic
