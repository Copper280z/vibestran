#include "magnetic/mesh.hpp"
#include "magnetic/air_gap.hpp"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
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
                       const std::vector<NodeList> &cells, bool planar,
                       const Vector &normal) {
  Mesh mesh;
  mesh.nodes = std::move(nodes);
  mesh.planar = planar;
  mesh.normal = normal;
  std::map<std::array<int, 2>, int> edges;
  std::set<NodeList> unique;
  for (const auto &indices : cells) {
    Cell c;
    c.nodes = indices;
    c.planar = planar;
    c.normal = normal;
    c.id = static_cast<int>(mesh.cells.size()) + 1;
    auto key = indices;
    std::sort(key.begin(), key.end());
    const size_t n = key.size();
    if ((planar ? (n != 3 && n != 4) : (n != 4 && n != 6 && n != 8)) ||
        key.front() < 0 || key.back() >= static_cast<int>(mesh.nodes.size()) ||
        std::adjacent_find(key.begin(), key.end()) != key.end() ||
        !unique.insert(key).second)
      return Error{
          std::format("Invalid or duplicate magnetic element {}", c.id)};
    c.origin = mesh.nodes[indices[0]];
    c.minimum = c.maximum = c.origin;
    Vector center = Vector::Zero();
    for (int v : indices) {
      c.points.push_back(mesh.nodes[v]);
      c.minimum = c.minimum.cwiseMin(mesh.nodes[v]);
      c.maximum = c.maximum.cwiseMax(mesh.nodes[v]);
      center += mesh.nodes[v];
    }
    center /= static_cast<double>(n);
    const double length = (c.maximum - c.minimum).norm();
    if (!(length > 0))
      return Error{"Degenerate magnetic element"};
    if (planar)
      for (const auto &x : c.points)
        if (std::abs(normal.dot(x - mesh.nodes.front())) > 1e-10 * length)
          return Error{"MAG2D elements must lie in a common plane normal to "
                       "the specified CID z axis"};
    initialize_mapping(c);
    const auto [mapped, jac] = mapping(c, Vector::Zero());
    c.map_origin = mapped;
    c.jacobian = jac;
    c.inverse = jac.inverse();
    c.affine = true;
    const double determinant = jac.determinant();
    const double min_det = 1e-14 * std::pow(length, planar ? 2 : 3);
    if (!jac.allFinite() || std::abs(determinant) <= min_det)
      return Error{std::format("Degenerate magnetic element {}", c.id)};
    for (const auto &q : reference_quadrature(c)) {
      const auto [x, j] = mapping(c, q.position);
      if (!j.allFinite() ||
          j.determinant() * std::copysign(1., determinant) <= min_det)
        return Error{
            std::format("Folded or degenerate magnetic element {}", c.id)};
      c.volume += q.weight * std::abs(j.determinant());
      c.affine &= (j - jac).norm() <=
                  1e-12 * (planar ? jac.leftCols<2>().norm() : jac.norm());
    }
    if (!valid_mapping(c, min_det))
      return Error{std::format("Magnetic element {} has a folded, degenerate "
                               "or uncertifiable Jacobian",
                               c.id)};
    const auto reference = reference_vertices(c);
    Vector reference_center = Vector::Zero();
    for (const auto &r : reference)
      reference_center += r / static_cast<double>(reference.size());
    const auto topology = cell_edges(c);
    for (auto ij : topology) {
      std::array<int, 2> edge{indices[ij[0]], indices[ij[1]]};
      c.edge_sign.push_back(edge[0] < edge[1] ? 1 : -1);
      std::sort(edge.begin(), edge.end());
      const auto [it, inserted] =
          edges.emplace(edge, static_cast<int>(edges.size()));
      if (inserted)
        mesh.edges.push_back(edge);
      c.edges.push_back(it->second);
    }
    if (planar)
      c.edges = indices; // scalar A_normal nodal degrees of freedom
    if (!planar && n == 4) {
      auto shape = reference_shape(c, Vector::Zero());
      for (const auto &g : shape.gradients)
        c.gradient.push_back(c.inverse.transpose() * g);
      for (size_t e = 0; e < topology.size(); ++e)
        c.curl.push_back(
            2.0 * c.edge_sign[e] *
            c.gradient[topology[e][0]].cross(c.gradient[topology[e][1]]));
    }
    for (auto face : cell_faces(c)) {
      for (int &v : face)
        v = indices[v];
      // Canonical cyclic order, rather than sorted order, is essential for
      // quads.
      std::rotate(face.begin(), std::min_element(face.begin(), face.end()),
                  face.end());
      if (face.size() > 2 && face[1] > face.back())
        std::reverse(face.begin() + 1, face.end());
      auto sorted = face;
      std::sort(sorted.begin(), sorted.end());
      const auto [it, inserted] =
          mesh.face_index.emplace(sorted, static_cast<int>(mesh.faces.size()));
      Face candidate;
      candidate.nodes = face;
      for (int v : face)
        candidate.center += mesh.nodes[v];
      candidate.center /= static_cast<double>(face.size());
      if (planar) {
        candidate.normal =
            (mesh.nodes[face[1]] - mesh.nodes[face[0]]).cross(normal);
        candidate.area = candidate.normal.norm();
      } else {
        for (size_t i = 0; i < face.size(); ++i)
          candidate.normal +=
              (mesh.nodes[face[i]] - candidate.center)
                  .cross(mesh.nodes[face[(i + 1) % face.size()]] -
                         candidate.center) /
              2;
        candidate.area = candidate.normal.norm();
      }
      if (candidate.area <= 1e-14 * std::pow(length, planar ? 1 : 2))
        return Error{"Degenerate magnetic face"};
      candidate.normal.normalize();
      int sign = candidate.normal.dot(candidate.center - center) > 0 ? 1 : -1;
      if (!planar) {
        std::vector<Vector> r;
        Vector mid = Vector::Zero(), area = Vector::Zero();
        for (int v : face) {
          r.push_back(reference[std::find(indices.begin(), indices.end(), v) -
                                indices.begin()]);
          mid += r.back() / static_cast<double>(face.size());
        }
        for (size_t i = 0; i < r.size(); ++i)
          area += (r[i] - mid).cross(r[(i + 1) % r.size()] - mid);
        sign = area.dot(mid - reference_center) * determinant > 0 ? 1 : -1;
      }
      for (int v : face)
        if (std::abs(candidate.normal.dot(mesh.nodes[v] - candidate.center)) >
            1e-10 * length)
          candidate.flat = false;
      c.polyhedral &= candidate.flat;
      if (!candidate.flat) {
        double previous_area = 0;
        for (int order : {5, 9, 13}) {
          candidate.area = 0;
          for (const auto &q : face_quadrature(mesh, candidate, order))
            candidate.area += q.weight;
          if (std::abs(candidate.area - previous_area) <=
              1e-12 * candidate.area)
            break;
          previous_area = candidate.area;
        }
      }
      for (const auto &x : c.points)
        if (sign * candidate.normal.dot(x - candidate.center) >
            1e-10 * length) {
          if (planar)
            return Error{
                std::format("Nonconvex planar magnetic element {}", c.id)};
          c.polyhedral = false;
        }
      if (inserted)
        mesh.faces.push_back(candidate);
      auto &f = mesh.faces[it->second];
      if (f.nodes != face)
        return Error{"Inconsistent cyclic ordering on shared magnetic face"};
      if (f.cells[1] >= 0)
        return Error{"Non-manifold magnetic face"};
      if (f.cells[0] < 0)
        f.cells[0] = static_cast<int>(mesh.cells.size());
      else {
        const auto &previous = mesh.cells[f.cells[0]];
        const auto prior =
            std::find(previous.faces.begin(), previous.faces.end(), it->second);
        if (previous.face_sign[prior - previous.faces.begin()] == sign)
          return Error{"Overlapping cells on the same side of a shared face"};
        f.cells[1] = static_cast<int>(mesh.cells.size());
      }
      c.faces.push_back(it->second);
      c.face_sign.push_back(sign);
    }
    mesh.cells.push_back(std::move(c));
  }
  if (mesh.cells.empty())
    return Error{"Magnetic mesh contains no elements"};
  return mesh;
}

Result<Mesh> make_mesh(const Model &model) {
  // Elements of an AGE air gap (PAIRGAP) are not field cells: the air-gap
  // element replaces them. MESH gaps were made ordinary air by the parser.
  std::vector<const Element *> field;
  for (const auto &c : model.elements)
    if (!model.air_gaps.contains(c.property))
      field.push_back(&c);
  if (field.empty())
    return Error{"Magnetic mesh contains no elements outside the air gap"};
  std::vector<Vector> nodes;
  std::map<int, int> index;
  std::set<int> active;
  for (const auto *c : field)
    for (int n : c->nodes)
      active.insert(n);
  for (int id : active) {
    auto it = model.nodes.find(id);
    if (it == model.nodes.end())
      return Error{std::format("Undefined GRID {}", id)};
    index[id] = static_cast<int>(nodes.size());
    nodes.push_back(it->second.position);
  }
  std::vector<NodeList> cells;
  for (const auto *c : field) {
    NodeList ids(c->nodes.size());
    std::transform(c->nodes.begin(), c->nodes.end(), ids.begin(),
                   [&index](int n) { return index.at(n); });
    cells.push_back(ids);
  }
  Vector normal = Vector::UnitZ();
  if (model.options.plane_coordinates) {
    const auto &cs = model.coordinates.at(model.options.plane_coordinates);
    normal = (cs.b - cs.a).normalized();
  }
  auto result =
      make_mesh(std::move(nodes), cells, model.options.planar, normal);
  if (auto *e = std::get_if<Error>(&result))
    return *e;
  auto mesh = std::get<Mesh>(std::move(result));
  mesh.depth = model.options.planar ? model.options.depth : 1;
  mesh.node_index = index;
  mesh.node_ids.assign(active.begin(), active.end());
  for (size_t i = 0; i < mesh.cells.size(); ++i) {
    mesh.cells[i].id = field[i]->id;
    const auto prop = model.properties.find(field[i]->property);
    if (prop == model.properties.end())
      return Error{"Magnetic element references undefined property"};
    mesh.cells[i].property = prop->first;
    mesh.cells[i].material = prop->second.material;
    if (!mesh.element_index.emplace(mesh.cells[i].id, static_cast<int>(i))
             .second)
      return Error{"Duplicate magnetic element id"};
  }
  if (model.period) {
    auto period = make_periodicity(model, mesh);
    if (auto *e = std::get_if<Error>(&period))
      return *e;
    mesh.period = std::get<Periodicity>(std::move(period));
  }
  if (!model.air_gaps.empty()) {
    auto gap = make_air_gap(mesh, model, model.air_gaps.begin()->first);
    if (auto *e = std::get_if<Error>(&gap))
      return *e;
    mesh.air_gap = std::get<AirGap>(std::move(gap));
    AirGap &g = *mesh.air_gap;
    if (g.band_points > 0) {
      g.band_first = static_cast<int>(mesh.nodes.size());
      mesh.band_unknowns =
          static_cast<size_t>((g.band_layers - 1) * (g.band_points / g.sectors));
    }
  }
  if (mesh.planar) {
    // A planar field mesh must be one piece, or two pieces joined by the
    // air gap; name an element that is cut off.
    Disjoint parts(static_cast<int>(mesh.nodes.size()));
    for (const auto &c : mesh.cells)
      for (size_t k = 1; k < c.nodes.size(); ++k)
        parts.join(c.nodes[0], c.nodes[k]);
    if (mesh.air_gap)
      parts.join(mesh.air_gap->inner.nodes.front(),
                 mesh.air_gap->outer.nodes.front());
    const int root = parts.root(mesh.cells.front().nodes.front());
    const auto apart =
        std::find_if(mesh.cells.begin(), mesh.cells.end(), [&](const Cell &c) {
          return parts.root(c.nodes.front()) != root;
        });
    if (apart != mesh.cells.end())
      return Error{
          mesh.air_gap
              ? std::format("Element {} is not joined to element {} by elements "
                            "or by the PAIRGAP {} air gap; a MAG2D mesh must be "
                            "one connected domain",
                            apart->id, mesh.cells.front().id,
                            mesh.air_gap->property)
              : std::format(
                    "Element {} is not joined to element {} by any chain of "
                    "elements; a MAG2D mesh must be one connected domain. If "
                    "they are a rotor and a stator separated by an unmeshed "
                    "air gap, mesh the air gap between them and give it a "
                    "PAIRGAP property, or join the parts",
                    apart->id, mesh.cells.front().id)};
  }
  return mesh;
}

Result<Periodicity> make_periodicity(const Model &model, const Mesh &mesh) {
  const Periodic &card = *model.period;
  Periodicity out;
  out.id = card.id;
  out.sectors = card.sectors;
  out.anti = card.anti;
  if (card.coordinates) {
    const auto &cs = model.coordinates.at(card.coordinates);
    out.origin = cs.a;
    out.axis = coordinate_axes(cs).col(2);
  }
  // The faces and grids of each side.
  std::array<std::set<int>, 2> grids;
  std::array<std::set<int>, 2> faces;
  for (int side = 0; side < 2; ++side) {
    const int set = side == 0 ? card.first : card.second;
    auto selected = selected_faces(model, mesh, set);
    if (auto *e = std::get_if<Error>(&selected))
      return Error{std::format("MPERIOD {}: {}", card.id, e->message)};
    for (auto [f, sign] : std::get<std::vector<std::pair<int, int>>>(selected)) {
      (void)sign;
      const Face &face = mesh.faces[f];
      if (face.cells[1] >= 0)
        return Error{std::format(
            "MPERIOD {}: FSET {} selects edge ({},{}) of element {}, which is "
            "inside the field mesh; the periodic sides are on its exterior",
            card.id, set, mesh.node_ids[face.nodes[0]], mesh.node_ids[face.nodes[1]],
            mesh.cells[face.cells[0]].id)};
      faces[side].insert(f);
      grids[side].insert(face.nodes.begin(), face.nodes.end());
    }
  }
  for (int f : faces[0])
    if (faces[1].contains(f))
      return Error{std::format(
          "MPERIOD {}: edge ({},{}) is in both FSET {} and FSET {}", card.id,
          mesh.node_ids[mesh.faces[f].nodes[0]], mesh.node_ids[mesh.faces[f].nodes[1]],
          card.first, card.second)};
  const double angle = 360. / card.sectors;
  const Matrix turn =
      Eigen::AngleAxisd(2 * std::numbers::pi / card.sectors, out.axis).toRotationMatrix();
  const auto from_axis = [&](const Vector &x) {
    const Vector d = x - out.origin;
    return (d - d.dot(out.axis) * out.axis).norm();
  };
  double scale = 0;
  for (const auto &side : grids)
    for (int v : side)
      scale = std::max(scale, from_axis(mesh.nodes[v]));
  const double tolerance = 1e-6 * scale;
  std::vector<int> image(mesh.nodes.size(), -1), preimage(mesh.nodes.size(), -1);
  for (int v : grids[0]) {
    const Vector target = out.origin + turn * (mesh.nodes[v] - out.origin);
    int nearest = -1;
    double distance = std::numeric_limits<double>::infinity();
    for (int w : grids[1])
      if (const double d = (mesh.nodes[w] - target).norm(); d < distance) {
        distance = d;
        nearest = w;
      }
    if (nearest < 0 || distance > tolerance)
      return Error{std::format(
          "MPERIOD {}: grid {} of the first side (FSET {}) turned by {:.6g} "
          "degrees about the axis lands at ({:.6g}, {:.6g}, {:.6g}), {:.3e} from "
          "the nearest grid of the second side (FSET {}){}. The second side must "
          "be the first turned counterclockwise by 360/SECTORS about the z axis "
          "of CID {}, grid for grid",
          card.id, mesh.node_ids[v], card.first, angle, target.x(), target.y(),
          target.z(), distance, card.second,
          nearest < 0 ? "" : std::format(", grid {}", mesh.node_ids[nearest]),
          card.coordinates)};
    if (preimage[nearest] >= 0)
      return Error{std::format("MPERIOD {}: grids {} and {} of the first side "
                               "both turn onto grid {}",
                               card.id, mesh.node_ids[preimage[nearest]],
                               mesh.node_ids[v], mesh.node_ids[nearest])};
    image[v] = nearest;
    preimage[nearest] = v;
    if (nearest == v)
      out.axial.push_back(v);
    else
      out.pairs.push_back({v, nearest});
  }
  if (const auto stray = std::ranges::find_if(grids[1], [&](int w) { return preimage[w] < 0; });
      stray != grids[1].end())
    return Error{std::format(
        "MPERIOD {}: grid {} of the second side (FSET {}) is not the image of "
        "any grid of the first side (FSET {})",
        card.id, mesh.node_ids[*stray], card.second, card.first)};
  // The images of the first side's edges are the second side's edges, so
  // the sector and its turned copy meet conformingly.
  for (int f : faces[0]) {
    NodeList turned{image[mesh.faces[f].nodes[0]], image[mesh.faces[f].nodes[1]]};
    std::sort(turned.begin(), turned.end());
    const auto found = mesh.face_index.find(turned);
    if (found == mesh.face_index.end() || !faces[1].contains(found->second))
      return Error{std::format(
          "MPERIOD {}: edge ({},{}) of the first side turns onto grids ({},{}), "
          "which are not an edge of the second side; mesh both sides alike",
          card.id, mesh.node_ids[mesh.faces[f].nodes[0]],
          mesh.node_ids[mesh.faces[f].nodes[1]], mesh.node_ids[turned[0]],
          mesh.node_ids[turned[1]])};
  }
  if (faces[0].size() != faces[1].size())
    return Error{std::format("MPERIOD {}: FSET {} has {} edges and FSET {} has "
                             "{}; each edge of the second side is the image of "
                             "one of the first",
                             card.id, card.first, faces[0].size(), card.second,
                             faces[1].size())};
  out.faces.assign(faces[0].begin(), faces[0].end());
  out.faces.insert(out.faces.end(), faces[1].begin(), faces[1].end());
  return out;
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
    if (ci == mesh.element_index.end() && mesh.air_gap &&
        std::ranges::find(mesh.air_gap->elements, s.element) !=
            mesh.air_gap->elements.end())
      return Error{std::format(
          "FSET {} references air-gap element {} (PAIRGAP {}); air-gap "
          "elements are not field cells and have no faces of their own",
          selection, s.element, mesh.air_gap->property)};
    if (ci == mesh.element_index.end())
      return Error{
          std::format("FSET {} references undefined magnetic element {}",
                      selection, s.element)};
    NodeList ids(s.nodes.size());
    for (size_t i = 0; i < s.nodes.size(); ++i) {
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
      return Error{std::format("FSET {} is not a magnetic element face/edge",
                               selection)};
    const auto &face = mesh.faces[f->second];
    if (face.cells[0] != ci->second && face.cells[1] != ci->second)
      return Error{"FSET face does not belong to its specified element"};
    if (!seen.insert(f->second).second)
      return Error{"Duplicate geometric face in FSET"};
    auto canonical = ids;
    int sign = 1;
    if (canonical.size() == 2) {
      if (canonical[0] > canonical[1]) {
        std::reverse(canonical.begin(), canonical.end());
        sign = -1;
      }
    } else {
      std::rotate(canonical.begin(),
                  std::min_element(canonical.begin(), canonical.end()),
                  canonical.end());
      if (canonical[1] > canonical.back()) {
        std::reverse(canonical.begin() + 1, canonical.end());
        sign = -1;
      }
    }
    if (canonical != face.nodes)
      return Error{"FSET face nodes must be in cyclic order"};
    out.emplace_back(f->second, sign);
  }
  return out;
}

Eigen::VectorXd Gauge::reduce(const Eigen::VectorXd &v) const {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(size);
  for (size_t e = 0; e < equations.size(); ++e)
    if (equations[e] >= 0)
      out[equations[e]] += sign[e] * v[static_cast<Eigen::Index>(e)];
  return out;
}

void Gauge::expand(const Eigen::VectorXd &x, Eigen::VectorXd &v,
                   double scale) const {
  for (size_t e = 0; e < equations.size(); ++e)
    if (equations[e] >= 0)
      v[static_cast<Eigen::Index>(e)] += scale * sign[e] * x[equations[e]];
}

Result<Gauge> make_gauge(const Mesh &mesh, const std::vector<bool> &natural) {
  const int n = static_cast<int>(mesh.nodes.size());
  Disjoint connected(n), boundary(n), tree(n);
  Gauge gauge;
  if (mesh.planar) {
    const auto unknowns = mesh.dofs();
    gauge.equations.assign(unknowns, -1);
    gauge.boundary.assign(unknowns, false);
    gauge.sign.assign(unknowns, 1);
    // Air-gap faces are coupled by the air-gap element and periodic faces
    // by their images, NORMAL faces take the natural condition: none is
    // prescribed. The gap joins the parts it separates.
    std::vector<bool> open(mesh.faces.size(), false);
    if (mesh.air_gap) {
      for (int f : mesh.air_gap->faces)
        open[f] = true;
      connected.join(mesh.air_gap->inner.nodes.front(),
                     mesh.air_gap->outer.nodes.front());
    }
    if (mesh.period)
      for (int f : mesh.period->faces)
        open[f] = true;
    for (size_t f = 0; f < natural.size(); ++f)
      if (natural[f])
        open[f] = true;
    for (size_t f = 0; f < mesh.faces.size(); ++f)
      if (mesh.faces[f].cells[1] < 0 && !open[f])
        for (int v : mesh.faces[f].nodes)
          gauge.boundary[v] = true;
    std::vector<int> image_of(n, -1);
    if (const auto &period = mesh.period) {
      for (auto [first, second] : period->pairs) {
        if (gauge.boundary[first] != gauge.boundary[second])
          return Error{std::format(
              "MPERIOD {}: grid {} and its image grid {} must both be on the "
              "prescribed (ZERO or UNIFORM) exterior or both off it, but grid "
              "{} is on it and grid {} is not. Give the faces at both ends of "
              "the periodic sides the same boundary condition",
              period->id, mesh.node_ids[first], mesh.node_ids[second],
              mesh.node_ids[gauge.boundary[first] ? first : second],
              mesh.node_ids[gauge.boundary[first] ? second : first])};
        image_of[second] = first;
      }
      // An anti-periodic potential changes sign at a grid on the axis,
      // its own image, so it is zero there.
      if (period->anti)
        for (int v : period->axial)
          gauge.boundary[v] = true;
    }
    for (const auto &e : mesh.edges)
      connected.join(e[0], e[1]);
    for (int i = 0; i < n; ++i) {
      if (connected.root(i) != connected.root(0))
        return Error{"Magnetic domain must be connected"};
      if (!gauge.boundary[i] && image_of[i] < 0)
        gauge.equations[i] = gauge.size++;
    }
    for (int i = 0; i < n; ++i)
      if (image_of[i] >= 0) {
        gauge.equations[i] = gauge.equations[image_of[i]];
        gauge.sign[i] = mesh.period->sign();
      }
    // A sliding band's interior rings are free; their images in a sector
    // model are folded by the band itself.
    for (size_t i = mesh.nodes.size(); i < unknowns; ++i)
      gauge.equations[i] = gauge.size++;
    // An anti-periodic potential has no free constant.
    if (!(mesh.period && mesh.period->anti) &&
        std::none_of(gauge.boundary.begin(), gauge.boundary.end(),
                     [](bool b) { return b; }))
      return Error{
          "The planar field has no prescribed potential: every exterior face "
          "is periodic or NORMAL, so A is known only up to a constant. Leave "
          "part of the exterior to the ZERO or UNIFORM MBC row"};
    return gauge;
  }
  gauge.equations.assign(mesh.edges.size(), -1);
  gauge.boundary.assign(mesh.edges.size(), false);
  gauge.sign.assign(mesh.edges.size(), 1);
  std::set<std::array<int, 2>> bedges;
  std::set<int> bnodes;
  int bf = 0;
  for (const auto &f : mesh.faces)
    if (f.cells[1] < 0) {
      ++bf;
      for (size_t i = 0; i < f.nodes.size(); ++i) {
        int a = f.nodes[i], b = f.nodes[(i + 1) % f.nodes.size()];
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
                                            const Mesh &b, const Cell &cb,
                                            int order) {
  // The spatial index returns whole leaves. Most pairs can be excluded
  // before allocating polygons; contact alone has no volume contribution.
  if ((ca.maximum.array() <= cb.minimum.array()).any() ||
      (cb.maximum.array() <= ca.minimum.array()).any())
    return {};
  using Polygon = std::vector<Vector>;
  std::vector<Polygon> faces;
  for (int fi : ca.faces) {
    Polygon p;
    for (int v : a.faces[fi].nodes)
      p.push_back(a.nodes[v]);
    faces.push_back(std::move(p));
  }
  const double scale = std::max((ca.maximum - ca.minimum).norm(),
                                (cb.maximum - cb.minimum).norm());
  const double tolerance =
      64 * std::numeric_limits<double>::epsilon() *
      std::max(scale, std::max(ca.origin.norm(), cb.origin.norm()));
  for (size_t plane = 0; plane < cb.faces.size() && !faces.empty(); ++plane) {
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
  if (faces.empty())
    return {};
  std::vector<double> gauss_x(order), gauss_w(order);
  if (order > 3)
    for (int i = 0; i < order; ++i) {
      double z = std::cos(std::acos(-1.) * (i + .75) / (order + .5)),
             derivative = 0;
      for (int iteration = 0; iteration < 20; ++iteration) {
        double p0 = 1, p1 = z;
        for (int k = 2; k <= order; ++k) {
          const double p = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
          p0 = p1;
          p1 = p;
        }
        derivative = order * (z * p1 - p0) / (z * z - 1);
        const double step = p1 / derivative;
        z -= step;
        if (std::abs(step) < 1e-15)
          break;
      }
      gauss_x[i] = (z + 1) / 2;
      gauss_w[i] = 1 / ((1 - z * z) * derivative * derivative);
    }
  std::vector<Sample> out;
  size_t pieces = 0;
  for (const auto &f : faces)
    pieces += f.size() - 2;
  out.reserve(pieces * (order <= 2   ? 4
                        : order == 3 ? 5
                                     : order * order * order));
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
      if (order <= 2) {
        auto samples = tetra_quadrature({center, f[0], f[i], f[i + 1]}, volume);
        out.insert(out.end(), samples.begin(), samples.end());
      } else if (order == 3) {
        // Degree-three tetrahedral rule: exact for affine RT0 currents
        // paired with affine tet, hex or wedge edge basis functions.
        const std::array<Vector, 4> p{center, f[0], f[i], f[i + 1]};
        const Vector sum = p[0] + p[1] + p[2] + p[3];
        out.push_back({sum / 4, -.8 * volume});
        for (const auto &x : p)
          out.push_back({sum / 6 + x / 3, .45 * volume});
      } else {
        // Duffy-mapped Gauss integration on each intersection tetrahedron.
        // Mapped Piola bases are rational on nonaffine cells; the old
        // degree-two tetrahedron rule cannot conserve the transferred current.
        for (int u = 0; u < order; ++u)
          for (int v = 0; v < order; ++v)
            for (int t = 0; t < order; ++t) {
              const double r = gauss_x[u], s = (1 - r) * gauss_x[v],
                           z = (1 - r) * (1 - gauss_x[v]) * gauss_x[t];
              out.push_back({center + r * (f[0] - center) +
                                 s * (f[i] - center) + z * (f[i + 1] - center),
                             6 * volume * gauss_w[u] * gauss_w[v] * gauss_w[t] *
                                 (1 - r) * (1 - r) * (1 - gauss_x[v])});
            }
      }
    }
  return out;
}
} // namespace vibestran::magnetic
