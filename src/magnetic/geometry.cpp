#include "magnetic/mesh.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <limits>

namespace vibestran::magnetic {
std::vector<Vector> reference_vertices(const Cell &c) {
  if (!c.planar && c.nodes.size() == 4)
    return {Vector::Zero(), Vector::UnitX(), Vector::UnitY(), Vector::UnitZ()};
  if (c.nodes.size() == 3)
    return {Vector::Zero(), Vector::UnitX(), Vector::UnitY()};
  if (c.nodes.size() == 6)
    return {Vector(0, 0, -1), Vector(1, 0, -1), Vector(0, 1, -1),
            Vector(0, 0, 1),  Vector(1, 0, 1),  Vector(0, 1, 1)};
  if (c.planar)
    return {Vector(-1, -1, 0), Vector(1, -1, 0), Vector(1, 1, 0),
            Vector(-1, 1, 0)};
  return {Vector(-1, -1, -1), Vector(1, -1, -1), Vector(1, 1, -1),
          Vector(-1, 1, -1),  Vector(-1, -1, 1), Vector(1, -1, 1),
          Vector(1, 1, 1),    Vector(-1, 1, 1)};
}

bool valid_mapping(const Cell &c, double minimum) {
  const double sign = std::copysign(1., c.jacobian.determinant());
  auto determinant = [&](const Vector &r) {
    return sign * mapping(c, r).second.determinant();
  };
  for (const auto &r : reference_vertices(c))
    if (!(determinant(r) > minimum))
      return false;
  if (c.affine || c.planar)
    return true;
  if (c.nodes.size() == 6) {
    // The wedge determinant is P1 in the triangle and P2 along its axis.
    std::function<bool(double, double, int)> certify = [&](double a, double b,
                                                           int depth) {
      bool bounded = true;
      for (const Vector &p :
           {Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0)}) {
        const double v0 = determinant(Vector(p.x(), p.y(), a));
        const double vm = determinant(Vector(p.x(), p.y(), (a + b) / 2));
        const double v1 = determinant(Vector(p.x(), p.y(), b));
        if (!(std::min({v0, vm, v1}) > minimum))
          return false;
        bounded &= 2 * vm - (v0 + v1) / 2 > minimum;
      }
      return bounded || (depth < 16 && certify(a, (a + b) / 2, depth + 1) &&
                         certify((a + b) / 2, b, depth + 1));
    };
    return certify(-1, 1, 0);
  }
  // det(J) for a trilinear hex is triquadratic. Bernstein coefficients bound
  // the entire reference cell, unlike checks at corners or Gauss points alone.
  // Subdivide only boxes whose coefficient bounds cannot establish positivity.
  int boxes = 0;
  std::function<bool(const Vector &, const Vector &, int)> certify =
      [&](const Vector &lo, const Vector &hi, int depth) {
        if (++boxes > 65536)
          return false;
        std::array<double, 27> coefficients{};
        for (int k = 0; k < 3; ++k)
          for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
              Vector r = lo + (hi - lo).cwiseProduct(Vector(i, j, k)) / 2;
              double d = determinant(r);
              if (!(d > minimum) || !std::isfinite(d))
                return false;
              coefficients[i + 3 * j + 9 * k] = d;
            }
        for (int stride : {1, 3, 9})
          for (int i = 0; i < 27; ++i)
            if ((i / stride) % 3 == 0)
              coefficients[i + stride] =
                  2 * coefficients[i + stride] -
                  (coefficients[i] + coefficients[i + 2 * stride]) / 2;
        if (*std::min_element(coefficients.begin(), coefficients.end()) >
            minimum)
          return true;
        if (depth >= 12)
          return false;
        const Vector mid = (lo + hi) / 2;
        for (int mask = 0; mask < 8; ++mask) {
          Vector a, b;
          for (int axis = 0; axis < 3; ++axis) {
            a[axis] = mask & (1 << axis) ? mid[axis] : lo[axis];
            b[axis] = mask & (1 << axis) ? hi[axis] : mid[axis];
          }
          if (!certify(a, b, depth + 1))
            return false;
        }
        return true;
      };
  return certify(Vector::Constant(-1), Vector::Ones(), 0);
}

std::vector<std::pair<double, double>> gauss_rule(int order) {
  std::vector<std::pair<double, double>> rule;
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
    rule.emplace_back((z + 1) / 2, 1 / ((1 - z * z) * derivative * derivative));
  }
  return rule;
}

std::vector<SurfaceSample> face_quadrature(const Mesh &m, const Face &f,
                                           int order) {
  std::vector<SurfaceSample> out;
  if (m.planar) {
    const Vector mid = (m.nodes[f.nodes[0]] + m.nodes[f.nodes[1]]) / 2;
    const Vector half = (m.nodes[f.nodes[1]] - m.nodes[f.nodes[0]]) / 2;
    const double a = std::sqrt(.6);
    return {{mid - a * half, f.normal, f.area * 5 / 18},
            {mid, f.normal, f.area * 4 / 9},
            {mid + a * half, f.normal, f.area * 5 / 18}};
  }
  const Vector a = m.nodes[f.nodes[0]], b = m.nodes[f.nodes[1]] - a;
  if (f.nodes.size() == 3) {
    const Vector d = m.nodes[f.nodes[2]] - a;
    Cell triangle;
    triangle.planar = true;
    triangle.nodes.resize(3);
    for (const auto &q : reference_quadrature(triangle))
      out.push_back({a + q.position.x() * b + q.position.y() * d, f.normal,
                     q.weight * 2 * f.area});
  } else {
    const Vector d = m.nodes[f.nodes[3]] - a,
                 e = m.nodes[f.nodes[2]] - a - b - d;
    const auto rule = gauss_rule(order);
    for (auto [u, wu] : rule)
      for (auto [v, wv] : rule) {
        const Vector cross = (b + v * e).cross(d + u * e);
        const double area = cross.norm();
        out.push_back(
            {a + u * b + v * d + u * v * e, cross / area, wu * wv * area});
      }
  }
  return out;
}

bool point_on_face(const Mesh &mesh, const Face &face, const Vector &x,
                   double tolerance) {
  const auto &cell = mesh.cells[face.cells[0]];
  if (!contains(cell, x, tolerance))
    return false;
  const auto shape = reference_shape(cell, reference_coordinates(cell, x));
  for (size_t i = 0; i < cell.nodes.size(); ++i)
    if (std::find(face.nodes.begin(), face.nodes.end(), cell.nodes[i]) ==
            face.nodes.end() &&
        std::abs(shape.values[i]) > tolerance)
      return false;
  return true;
}

std::vector<double> line_face_intersections(const Mesh &m, const Face &face,
                                            const Vector &first,
                                            const Vector &last) {
  const Vector direction = last - first;
  const double length = direction.norm();
  std::vector<double> roots;
  if (!(length > 0))
    return roots;
  const Vector p = m.nodes[face.nodes[0]], b = m.nodes[face.nodes[1]] - p;
  const Vector c = m.nodes[face.nodes.back()] - p;
  const double scale = std::max({b.norm(), c.norm(), length});
  constexpr double tolerance = 2e-11;
  auto append = [&](const Vector &point) {
    const double t = (point - first).dot(direction) / direction.squaredNorm();
    if (t >= -tolerance && t <= 1 + tolerance &&
        (first + t * direction - point).norm() <= tolerance * scale)
      roots.push_back(std::clamp(t, 0., 1.));
  };
  if (face.nodes.size() == 3) {
    const double denominator = face.normal.dot(direction);
    if (std::abs(denominator) > 1e-14 * length) {
      const Vector x =
          first + face.normal.dot(p - first) / denominator * direction;
      const Vector cross = b.cross(c);
      const double u = (x - p).cross(c).dot(cross) / cross.squaredNorm();
      const double v = b.cross(x - p).dot(cross) / cross.squaredNorm();
      if (u >= -tolerance && v >= -tolerance && u + v <= 1 + tolerance)
        append(x);
      return roots;
    }
  } else {
    const Vector d = m.nodes[face.nodes[2]] - p - b - c;
    const Vector u = direction.unitOrthogonal(),
                 v = direction.cross(u) / length;
    // Use the face scale for the resultant: a long source segment must not
    // turn a small field face into a numerically degenerate projection.
    const double face_scale = std::max({b.norm(), c.norm(), d.norm()});
    auto project = [&](const Vector &x) -> Eigen::Vector2d {
      return {x.dot(u) / face_scale, x.dot(v) / face_scale};
    };
    const auto a2 = project(p - first), b2 = project(b), c2 = project(c),
               d2 = project(d);
    auto cross = [](const Eigen::Vector2d &a, const Eigen::Vector2d &z) {
      return a.x() * z.y() - a.y() * z.x();
    };
    const double aa = cross(b2, d2), bb = cross(a2, d2) + cross(b2, c2),
                 cc = cross(a2, c2);
    const double largest = std::max({std::abs(aa), std::abs(bb), std::abs(cc)});
    if (largest > 1e-14) {
      std::vector<double> parameters;
      if (std::abs(aa) < 1e-14 * largest) {
        if (std::abs(bb) > 1e-14 * largest)
          parameters.push_back(-cc / bb);
      } else {
        const double discriminant = bb * bb - 4 * aa * cc;
        if (discriminant >= -1e-14 * largest * largest) {
          const double q =
              -.5 *
              (bb + std::copysign(std::sqrt(std::max(0., discriminant)), bb));
          if (q != 0) {
            parameters.push_back(q / aa);
            parameters.push_back(cc / q);
          } else
            parameters.push_back(-bb / (2 * aa));
        }
      }
      bool ruled = false;
      for (double r : parameters)
        if (r >= -tolerance && r <= 1 + tolerance) {
          const Eigen::Vector2d denominator = c2 + r * d2,
                                numerator = a2 + r * b2;
          const int axis =
              std::abs(denominator.x()) > std::abs(denominator.y()) ? 0 : 1;
          if (std::abs(denominator[axis]) < 1e-14) {
            ruled |= numerator.norm() < 1e-12;
            continue;
          }
          const double s = -numerator[axis] / denominator[axis];
          if (s >= -tolerance && s <= 1 + tolerance)
            append(p + r * b + s * c + r * s * d);
        }
      if (!ruled)
        return roots;
    }
  }
  // A line contained in a face (or along a ruled direction) is split at its
  // boundary edges. Shared-face intervals are handled once by the integrator.
  for (size_t i = 0; i < face.nodes.size(); ++i) {
    const Vector a = m.nodes[face.nodes[i]],
                 e = m.nodes[face.nodes[(i + 1) % face.nodes.size()]] - a;
    const Vector n = direction.cross(e);
    if (n.squaredNorm() > 1e-28 * direction.squaredNorm() * e.squaredNorm()) {
      const double s = (a - first).cross(direction).dot(n) / n.squaredNorm();
      if (s >= -tolerance && s <= 1 + tolerance)
        append(a + s * e);
    } else {
      append(a);
      append(a + e);
    }
  }
  return roots;
}

namespace {
double contact_tolerance(const Cell &poly) {
  return std::max(1e-11 * (poly.maximum - poly.minimum).norm(),
                  64 * std::numeric_limits<double>::epsilon() *
                      poly.origin.norm());
}
// For multilinear maps, each plane functional is bounded by its values at
// the parameter-box corners. This detects even small overlaps that quadrature
// samples can miss, and treats contact separately from positive penetration.
template <class Map>
Result<bool> interior_intersection(const Mesh &mesh, const Cell &poly,
                                   int dimensions, const Map &map) {
  struct Box {
    Vector lo, hi;
    int depth;
  };
  std::vector<Box> pending{{Vector::Zero(), Vector::Ones(), 0}};
  const double tolerance = contact_tolerance(poly);
  int boxes = 0;
  while (!pending.empty()) {
    const auto box = pending.back();
    pending.pop_back();
    std::vector<Vector> points;
    for (int mask = 0; mask < (1 << dimensions); ++mask) {
      Vector r = Vector::Zero();
      for (int axis = 0; axis < dimensions; ++axis)
        r[axis] = mask & (1 << axis) ? box.hi[axis] : box.lo[axis];
      points.push_back(map(r));
    }
    bool separated = false;
    std::vector<bool> inside(points.size(), true);
    for (size_t i = 0; i < poly.faces.size(); ++i) {
      const auto &face = mesh.faces[poly.faces[i]];
      double minimum = std::numeric_limits<double>::infinity();
      for (size_t k = 0; k < points.size(); ++k) {
        const double d =
            poly.face_sign[i] * face.normal.dot(points[k] - face.center);
        minimum = std::min(minimum, d);
        inside[k] = inside[k] && d < -tolerance;
      }
      if (minimum >= -tolerance) {
        separated = true;
        break;
      }
    }
    if (separated)
      continue;
    if (std::find(inside.begin(), inside.end(), true) != inside.end())
      return true;
    const Vector center = map((box.lo + box.hi) / 2);
    bool center_inside = true;
    for (size_t i = 0; i < poly.faces.size(); ++i) {
      const auto &face = mesh.faces[poly.faces[i]];
      center_inside &=
          poly.face_sign[i] * face.normal.dot(center - face.center) <
          -tolerance;
    }
    if (center_inside)
      return true;
    if (++boxes > 65536 || box.depth >= 48)
      return Error{std::format("Unable to resolve geometric contact near "
                               "source cell {}; refine the touching geometry",
                               poly.id)};
    const int axis = box.depth % dimensions;
    Box low = box, high = box;
    low.hi[axis] = high.lo[axis] = (box.lo[axis] + box.hi[axis]) / 2;
    low.depth = high.depth = box.depth + 1;
    pending.push_back(low);
    pending.push_back(high);
  }
  return false;
}
} // namespace
Result<bool> cell_intersects_interior(const Mesh &mesh, const Cell &poly,
                                      const Cell &other) {
  const double tolerance = contact_tolerance(poly);
  if ((poly.maximum.array() <= other.minimum.array() + tolerance).any() ||
      (other.maximum.array() <= poly.minimum.array() + tolerance).any())
    return false;
  // A point strictly inside both volumes proves overlap. This also avoids
  // spending the subdivision budget on a shared boundary before reaching
  // the interior of a thin conductor contained in a much larger field cell.
  Vector center = Vector::Zero();
  for (const auto &point : poly.points)
    center += point / static_cast<double>(poly.points.size());
  if (contains(other, center)) {
    const auto shape =
        reference_shape(other, reference_coordinates(other, center));
    if (*std::min_element(shape.values.begin(), shape.values.end()) > 1e-10)
      return true;
  }
  return interior_intersection(mesh, poly, 3, [&](const Vector &u) -> Vector {
    Vector r = 2 * u - Vector::Ones();
    if (other.nodes.size() == 6)
      r = Vector(u.x(), (1 - u.x()) * u.y(), 2 * u.z() - 1);
    if (other.nodes.size() == 4)
      r = Vector(u.x(), (1 - u.x()) * u.y(), (1 - u.x()) * (1 - u.y()) * u.z());
    return mapping(other, r).first;
  });
}
Result<bool> face_intersects_interior(const Mesh &poly_mesh, const Cell &poly,
                                      const Mesh &mesh, const Face &face) {
  if (face.flat) {
    // A terminal tetrahedron may touch the exterior at only an edge/vertex.
    // Its own face planes cannot necessarily separate that contact from a
    // large exterior face, but the exterior plane is an exact separating axis.
    double minimum = std::numeric_limits<double>::infinity(),
           maximum = -minimum;
    for (const auto &point : poly.points) {
      const double distance = face.normal.dot(point - face.center);
      minimum = std::min(minimum, distance);
      maximum = std::max(maximum, distance);
    }
    const double tolerance = contact_tolerance(poly);
    if (minimum >= -tolerance || maximum <= tolerance)
      return false;
  }
  const Vector p = mesh.nodes[face.nodes[0]], b = mesh.nodes[face.nodes[1]] - p;
  const Vector c = mesh.nodes[face.nodes.back()] - p;
  Vector d = Vector::Zero();
  if (face.nodes.size() == 4)
    d = mesh.nodes[face.nodes[2]] - p - b - c;
  return interior_intersection(
      poly_mesh, poly, 2, [&](const Vector &r) -> Vector {
        if (face.nodes.size() == 3)
          return p + r.x() * b + (1 - r.x()) * r.y() * c;
        return p + r.x() * b + r.y() * c + r.x() * r.y() * d;
      });
}
} // namespace vibestran::magnetic
