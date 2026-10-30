#include "magnetic/solver.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <numeric>
#include <set>

namespace vibestran::magnetic {
namespace {
struct Section {
  std::vector<Eigen::Vector2d> points;
  std::vector<NodeList> quads, triangles;
};
Section cross_section(const WireProperty &p, int sides, int layers) {
  Section s;
  const int k = p.rectangular ? layers : (sides + 3) / 4;
  const double width = p.rectangular ? p.width : .45 * p.width;
  const double height = p.rectangular ? p.height : width;
  for (int j = 0; j <= k; ++j)
    for (int i = 0; i <= k; ++i)
      s.points.emplace_back(width * (static_cast<double>(i) / k - .5),
                            height * (static_cast<double>(j) / k - .5));
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < k; ++i) {
      const int a = j * (k + 1) + i;
      s.quads.push_back({a, a + 1, a + k + 2, a + k + 1});
    }
  if (!p.rectangular) {
    // A square core and quadrilateral annuli avoid collapsed cells on the axis.
    NodeList perimeter;
    for (int i = 0; i < k; ++i)
      perimeter.push_back(i);
    for (int j = 0; j < k; ++j)
      perimeter.push_back(j * (k + 1) + k);
    for (int i = k; i > 0; --i)
      perimeter.push_back(k * (k + 1) + i);
    for (int j = k; j > 0; --j)
      perimeter.push_back(j * (k + 1));
    const auto core = perimeter;
    sides = 4 * k;
    for (int layer = 1; layer <= layers; ++layer) {
      NodeList ring;
      for (int i = 0; i < sides; ++i) {
        const double theta =
            -3 * std::numbers::pi / 4 + 2 * std::numbers::pi * i / sides;
        const Eigen::Vector2d outer =
            .5 * p.width * Eigen::Vector2d(std::cos(theta), std::sin(theta));
        const double fraction = static_cast<double>(layer) / layers;
        ring.push_back(static_cast<int>(s.points.size()));
        s.points.push_back((1 - fraction) * s.points[core[i]] +
                           fraction * outer);
      }
      for (int i = 0; i < sides; ++i) {
        const int next = (i + 1) % sides;
        s.quads.push_back({perimeter[i], ring[i], ring[next], perimeter[next]});
      }
      perimeter = std::move(ring);
    }
  }
  for (const auto &q : s.quads) {
    s.triangles.push_back({q[0], q[1], q[2]});
    s.triangles.push_back({q[0], q[2], q[3]});
  }
  return s;
}
Result<Source> sweep(const Model &model, const std::vector<int> &vertices,
                     const std::vector<int> &elements,
                     const std::vector<int> &signs, bool closed) {
  const auto &property =
      model.wire_properties.at(model.wires[elements[0]].property);
  const size_t count = elements.size();
  std::vector<Vector> points, tangents, axes;
  std::transform(vertices.begin(), vertices.end(), std::back_inserter(points),
                 [&](int n) { return model.nodes.at(n).position; });
  for (size_t i = 0; i < count; ++i) {
    const auto &wire = model.wires[elements[i]];
    const auto &p = model.wire_properties.at(wire.property);
    if (p.rectangular != property.rectangular || p.width != property.width ||
        p.height != property.height || p.material != property.material)
      return Error{"A connected CWIRE path requires the same section and "
                   "conductor material"};
    Vector tangent = points[i + 1] - points[i];
    if (tangent.norm() <= 1e-12 * property.width)
      return Error{"Zero-length CWIRE segment"};
    tangent.normalize();
    tangents.push_back(tangent);
    Vector axis;
    if (property.rectangular) {
      axis = wire.orientation ? *wire.orientation
                              : model.nodes.at(wire.orientation_node).position -
                                    model.nodes.at(wire.first).position;
      axis -= axis.dot(tangent) * tangent;
      if (axis.norm() <= 1e-12)
        return Error{"CWIRE orientation is parallel to its axis"};
      axis.normalize();
    } else if (i == 0)
      axis = tangent.unitOrthogonal();
    else {
      if (tangents[i - 1].dot(tangent) < -1 + 1e-8)
        return Error{"CWIRE makes a reversing bend"};
      axis = Eigen::Quaterniond::FromTwoVectors(tangents[i - 1], tangent) *
             axes.back();
      axis.normalize();
    }
    axes.push_back(axis);
  }
  if (closed && !property.rectangular) {
    Vector returned =
        Eigen::Quaterniond::FromTwoVectors(tangents.back(), tangents.front()) *
        axes.back();
    const double mismatch =
        std::atan2(tangents.front().dot(returned.cross(axes.front())),
                   returned.dot(axes.front()));
    // Distribute the frame closure twist; the shared end ring is exact.
    for (size_t i = 1; i < count; ++i)
      axes[i] = Eigen::AngleAxisd(mismatch * static_cast<double>(i) / count,
                                  tangents[i]) *
                axes[i];
  }
  const Section section = cross_section(property, model.options.wire_sides,
                                        model.options.wire_layers);
  const int ns = static_cast<int>(section.points.size());
  std::vector<std::vector<Vector>> joints(vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i) {
    const size_t next = i == count ? 0 : i,
                 previous = i == 0 ? count - 1 : i - 1;
    const bool endpoint = !closed && (i == 0 || i == count);
    const size_t reference = i == count ? count - 1 : i;
    Vector t = tangents[reference], u = axes[reference], normal = t;
    if (!endpoint) {
      const Vector sum = tangents[previous] + tangents[next];
      if (sum.norm() < 1e-6)
        return Error{"CWIRE reversing bend cannot be swept"};
      normal = sum.normalized();
      if (property.rectangular) {
        Vector transported = Eigen::Quaterniond::FromTwoVectors(
                                 tangents[previous], tangents[next]) *
                             axes[previous];
        if (transported.dot(axes[next]) < 1 - 1e-8)
          return Error{"Rectangular CWIRE orientations disagree across a bend"};
      }
    }
    Vector v = t.cross(u);
    for (const auto &p : section.points) {
      Vector offset = p.x() * u + p.y() * v;
      offset -= t * (normal.dot(offset) / normal.dot(t));
      joints[i].push_back(points[i] + offset);
    }
  }
  if (closed)
    joints.back() = joints.front();
  std::vector<Vector> nodes = joints.front();
  // First node of the lower and upper ring of each axial slice. Cells are
  // generated afterwards, only for the element type that is meshed.
  std::vector<std::array<int, 2>> slice_rings;
  int previous_ring = 0, last_ring = 0;
  const double step = model.options.wire_step > 0
                          ? model.options.wire_step
                          : 2 * std::max(property.width, property.height);
  for (size_t segment = 0; segment < count; ++segment) {
    double length = (points[segment + 1] - points[segment]).norm();
    if (length / step > 100000)
      return Error{"CWIRE refinement exceeds 100000 slices per segment"};
    // Rigid motion can move an integer ratio by a few ulps. Preserve topology
    // at that boundary instead of creating an extra axial slice.
    const double ratio = length / step;
    const int slices =
        std::max(1, static_cast<int>(std::ceil(
                        ratio - 32 * std::numeric_limits<double>::epsilon() *
                                    std::max(1., ratio))));
    for (int slice = 1; slice <= slices; ++slice) {
      int ring = 0;
      if (!(closed && segment + 1 == count && slice == slices)) {
        ring = static_cast<int>(nodes.size());
        double fraction = static_cast<double>(slice) / slices;
        for (int j = 0; j < ns; ++j)
          nodes.push_back((1 - fraction) * joints[segment][j] +
                          fraction * joints[segment + 1][j]);
      }
      slice_rings.push_back({previous_ring, ring});
      previous_ring = ring;
      last_ring = ring;
    }
  }
  const auto hexa = [&] {
    std::vector<NodeList> cells;
    cells.reserve(slice_rings.size() * section.quads.size());
    for (const auto [lower, upper] : slice_rings)
      for (const auto &q : section.quads) {
        NodeList cell;
        for (int v : q)
          cell.push_back(lower + v);
        for (int v : q)
          cell.push_back(upper + v);
        cells.push_back(std::move(cell));
      }
    return cells;
  };
  const auto tetra = [&] {
    std::vector<NodeList> cells;
    cells.reserve(3 * slice_rings.size() * section.triangles.size());
    for (const auto [lower, upper] : slice_rings)
      for (auto tri : section.triangles) {
        std::sort(tri.begin(), tri.end());
        const int a = lower + tri[0], b = lower + tri[1], c = lower + tri[2],
                  d = upper + tri[0], e = upper + tri[1], f = upper + tri[2];
        cells.push_back({a, b, c, f});
        cells.push_back({a, b, e, f});
        cells.push_back({a, d, e, f});
      }
    return cells;
  };
  bool tetrahedral = model.options.wire_mesh == WireMesh::Tet;
  auto built = make_mesh(std::move(nodes), tetrahedral ? tetra() : hexa());
  if (auto *mesh = std::get_if<Mesh>(&built);
      mesh && std::any_of(mesh->cells.begin(), mesh->cells.end(),
                          [](const Cell &c) { return !c.polyhedral; })) {
    if (model.options.wire_mesh == WireMesh::Hex)
      return Error{"WIREMESH=HEX requires planar-faced convex CWIRE hexes; "
                   "use WIREMESH=AUTO or TET for this path"};
    // Twisted spatial paths may have warped loft faces. Fall back for the whole
    // component so no nonconforming quad/triangle interfaces are introduced.
    // make_mesh stores the nodes unchanged, so reuse them.
    built = make_mesh(std::move(mesh->nodes), tetra());
    tetrahedral = true;
  }
  if (auto *e = std::get_if<Error>(&built))
    return Error{"Invalid CWIRE sweep: " + e->message};
  Source source;
  source.wire = true;
  source.closed = closed;
  source.id = model.wires[elements[0]].id;
  source.mesh = std::get<Mesh>(std::move(built));
  for (size_t i = 0; i < count; ++i)
    source.wire_elements.emplace_back(model.wires[elements[i]].id, signs[i]);
  // A miter can fold a short segment or meet another segment. Detect positive
  // volume intersections, including between components below, before solving.
  SpatialIndex index(source.mesh);
  for (size_t i = 0; i < source.mesh.cells.size(); ++i) {
    const auto &a = source.mesh.cells[i];
    for (int j : index.query(a.minimum, a.maximum))
      if (j > static_cast<int>(i)) {
        const auto &b = source.mesh.cells[j];
        if ((a.maximum.array() <= b.minimum.array()).any() ||
            (b.maximum.array() <= a.minimum.array()).any())
          continue;
        const auto samples =
            intersection_quadrature(source.mesh, a, source.mesh, b);
        const double overlap = std::accumulate(
            samples.begin(), samples.end(), 0.0,
            [](double sum, const Sample &q) { return sum + q.weight; });
        if (overlap > 1e-9 * std::min(a.volume, b.volume))
          return Error{"CWIRE swept conductor self-intersects; refine the path "
                       "or reduce the section"};
      }
  }
  std::vector<std::pair<int, int>> cut;
  std::vector<int> terminals;
  double area = 0;
  for (auto tri : (tetrahedral ? section.triangles : section.quads)) {
    std::sort(tri.begin(), tri.end());
    int f = source.mesh.face_index.at(tri);
    cut.emplace_back(f,
                     source.mesh.faces[f].normal.dot(tangents[0]) > 0 ? 1 : -1);
    area += source.mesh.faces[f].area;
    if (!closed) {
      terminals.push_back(f);
      std::transform(tri.begin(), tri.end(), tri.begin(),
                     [last_ring](int n) { return n + last_ring; });
      terminals.push_back(source.mesh.face_index.at(tri));
    }
  }
  std::vector<double> rho(source.mesh.cells.size(),
                          1 / model.conductivity.at(property.material));
  auto flux = conserved_flux(source.mesh, cut, terminals, false, area, rho);
  if (auto *e = std::get_if<Error>(&flux))
    return *e;
  source.flux = std::get<Eigen::VectorXd>(std::move(flux));
  return source;
}
} // namespace

Result<std::vector<Source>> prepare_wires(const Model &model) {
  std::map<int, std::vector<int>> incident;
  for (size_t i = 0; i < model.wires.size(); ++i) {
    incident[model.wires[i].first].push_back(static_cast<int>(i));
    incident[model.wires[i].second].push_back(static_cast<int>(i));
  }
  for (const auto &[node, edges] : incident)
    if (edges.size() > 2)
      return Error{std::format(
          "CWIRE branches at GRID {}; branching circuits are not supported",
          node)};
  std::set<int> unused;
  for (size_t i = 0; i < model.wires.size(); ++i)
    unused.insert(static_cast<int>(i));
  std::vector<Source> sources;
  while (!unused.empty()) {
    const int initial = *unused.begin();
    int start = model.wires[initial].first, node = start, previous = -1;
    // Walk to an endpoint, or establish that this component is a loop.
    do {
      const auto &connected = incident.at(node);
      if (connected.size() == 1) {
        start = node;
        break;
      }
      int edge = connected[0] == previous ? connected[1] : connected[0];
      const auto &w = model.wires[edge];
      previous = edge;
      node = w.first == node ? w.second : w.first;
    } while (node != start);
    const bool closed = incident.at(start).size() == 2;
    std::vector<int> vertices{start}, elements, signs;
    node = start;
    for (;;) {
      int edge = -1;
      for (int candidate : incident.at(node))
        if (unused.contains(candidate)) {
          edge = candidate;
          break;
        }
      if (edge < 0)
        break;
      unused.erase(edge);
      const auto &w = model.wires[edge];
      elements.push_back(edge);
      signs.push_back(w.first == node ? 1 : -1);
      node = w.first == node ? w.second : w.first;
      vertices.push_back(node);
      if (node == start)
        break;
    }
    if (closed && elements.size() < 3)
      return Error{"Closed CWIRE path needs at least three segments"};
    auto source = sweep(model, vertices, elements, signs, closed);
    if (auto *e = std::get_if<Error>(&source))
      return *e;
    sources.push_back(std::get<Source>(std::move(source)));
  }
  std::vector<SpatialIndex> indexes;
  indexes.reserve(sources.size());
  for (const auto &source : sources)
    indexes.emplace_back(source.mesh);
  for (size_t i = 0; i < sources.size(); ++i)
    for (size_t j = 0; j < i; ++j) {
      const auto &a = sources[i].mesh;
      const auto &b = sources[j].mesh;
      for (const auto &cell : a.cells)
        for (int ci : indexes[j].query(cell.minimum, cell.maximum)) {
          const auto samples = intersection_quadrature(a, cell, b, b.cells[ci]);
          const double volume = std::accumulate(
              samples.begin(), samples.end(), 0.0,
              [](double sum, const Sample &q) { return sum + q.weight; });
          if (volume > 1e-9 * std::min(cell.volume, b.cells[ci].volume))
            return Error{"Distinct CWIRE conductors overlap"};
        }
    }
  return sources;
}
} // namespace vibestran::magnetic
