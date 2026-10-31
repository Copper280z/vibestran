#include "magnetic/air_gap.hpp"
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <numeric>

namespace vibestran::magnetic {
namespace {
constexpr double pi = std::numbers::pi;

struct Parts {
  std::vector<int> parent;
  explicit Parts(int n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
  int root(int i) {
    while (parent[i] != i)
      i = parent[i] = parent[parent[i]];
    return i;
  }
  void join(int a, int b) { parent[root(a)] = root(b); }
};

// A closed chain of boundary faces, its grids in order.
struct Loop {
  int part{0};
  std::vector<int> nodes, faces;
  Eigen::Vector2d center{Eigen::Vector2d::Zero()};
  double radius{0}, deviation{std::numeric_limits<double>::infinity()};
  int worst{0}; // node of largest deviation
};

// Least-squares circle through the loop's points (Kasa fit about the
// centroid), and the largest relative radial deviation of a grid from it.
void fit_circle(Loop &loop, const std::vector<Eigen::Vector2d> &p) {
  const auto n = static_cast<Eigen::Index>(loop.nodes.size());
  const Eigen::Vector2d mean =
      std::accumulate(loop.nodes.begin(), loop.nodes.end(), Eigen::Vector2d(0, 0),
                      [&p](const Eigen::Vector2d &sum, int i) { return Eigen::Vector2d(sum + p[i]); }) /
      static_cast<double>(n);
  Eigen::MatrixXd a(n, 3);
  Eigen::VectorXd b(n);
  for (Eigen::Index k = 0; k < n; ++k) {
    const Eigen::Vector2d q = p[loop.nodes[k]] - mean;
    a.row(k) << q.x(), q.y(), 1;
    b[k] = -q.squaredNorm();
  }
  const Eigen::Vector3d x = a.colPivHouseholderQr().solve(b);
  loop.center = mean - x.head<2>() / 2;
  loop.radius = std::sqrt(std::max(x.head<2>().squaredNorm() / 4 - x[2], 0.0));
  loop.deviation = 0;
  for (int i : loop.nodes) {
    const double d =
        std::abs((p[i] - loop.center).norm() - loop.radius) / loop.radius;
    if (d > loop.deviation) {
      loop.deviation = d;
      loop.worst = i;
    }
  }
}

GapCircle circle(const Loop &loop, const Eigen::Vector2d &center,
                 const std::vector<Eigen::Vector2d> &p) {
  GapCircle c;
  c.radius = loop.radius;
  std::vector<std::pair<double, int>> order;
  for (int i : loop.nodes) {
    const Eigen::Vector2d q = p[i] - center;
    order.emplace_back(std::atan2(q.y(), q.x()), i);
  }
  std::sort(order.begin(), order.end());
  for (auto [angle, i] : order) {
    c.angles.push_back(angle);
    c.nodes.push_back(i);
  }
  return c;
}

// The potential's coefficients on both circles, for field evaluation.
struct Harmonics {
  Eigen::VectorXd inner, outer;
};
Harmonics coefficients(const AirGap &gap, const Eigen::VectorXd &a) {
  auto values = [&a](const GapCircle &c) {
    Eigen::VectorXd out(static_cast<Eigen::Index>(c.nodes.size()));
    for (size_t k = 0; k < c.nodes.size(); ++k)
      out[static_cast<Eigen::Index>(k)] = a[c.nodes[k]];
    return out;
  };
  return {gap_transform(gap.inner, gap.harmonics) * values(gap.inner),
          gap_transform(gap.outer, gap.harmonics) * values(gap.outer)};
}
} // namespace

Result<std::optional<AirGap>> find_air_gap(const Mesh &mesh) {
  if (!mesh.planar)
    return std::nullopt;
  const int n = static_cast<int>(mesh.nodes.size());
  Parts parts(n);
  for (const auto &c : mesh.cells)
    for (int v : c.nodes)
      parts.join(v, c.nodes.front());
  std::vector<int> roots;
  for (int i = 0; i < n; ++i)
    if (parts.root(i) == i)
      roots.push_back(i);
  if (roots.size() == 1)
    return std::nullopt;

  // In-plane frame with u x v = normal, and plane coordinates of the grids.
  AirGap gap;
  const Vector &normal = mesh.normal;
  gap.u = normal.cross(std::abs(normal.x()) < .9 ? Vector::UnitX()
                                                 : Vector::UnitY())
              .normalized();
  gap.v = normal.cross(gap.u);
  const Vector origin = mesh.nodes.front();
  std::vector<Eigen::Vector2d> p(n);
  for (int i = 0; i < n; ++i)
    p[i] = {(mesh.nodes[i] - origin).dot(gap.u), (mesh.nodes[i] - origin).dot(gap.v)};
  auto grid = [&mesh](int i) { return mesh.node_ids.empty() ? i + 1 : mesh.node_ids[i]; };
  auto element_of_part = [&](int root) {
    const auto c = std::find_if(mesh.cells.begin(), mesh.cells.end(), [&](const Cell &cell) {
      return parts.root(cell.nodes.front()) == root;
    });
    return c == mesh.cells.end() ? 0 : c->id;
  };

  // Boundary loops; a grid shared by more than two boundary faces (a pinch)
  // ends its loop's eligibility.
  std::vector<std::vector<int>> boundary_faces(n);
  for (size_t f = 0; f < mesh.faces.size(); ++f)
    if (mesh.faces[f].cells[1] < 0)
      for (int v : mesh.faces[f].nodes)
        boundary_faces[v].push_back(static_cast<int>(f));
  std::vector<bool> visited(mesh.faces.size(), false);
  std::vector<Loop> loops;
  for (size_t start = 0; start < mesh.faces.size(); ++start) {
    if (mesh.faces[start].cells[1] >= 0 || visited[start])
      continue;
    Loop loop;
    bool simple = boundary_faces[mesh.faces[start].nodes[0]].size() == 2;
    int f = static_cast<int>(start), at = mesh.faces[start].nodes[1];
    loop.nodes.push_back(mesh.faces[start].nodes[0]);
    while (!visited[f]) {
      visited[f] = true;
      loop.faces.push_back(f);
      if (at == loop.nodes.front())
        break;
      loop.nodes.push_back(at);
      if (boundary_faces[at].size() != 2) {
        simple = false;
        break;
      }
      f = boundary_faces[at][0] == f ? boundary_faces[at][1] : boundary_faces[at][0];
      const auto &next = mesh.faces[f].nodes;
      at = next[0] == at ? next[1] : next[0];
    }
    // Any three points lie on a circle, so a gap circle needs four grids.
    if (!simple || loop.nodes.size() < 4)
      continue;
    loop.part = parts.root(loop.nodes.front());
    fit_circle(loop, p);
    loops.push_back(std::move(loop));
  }

  // Gap candidates: circular loops on different parts, concentric, with no
  // grid strictly between them. Pairs that fail are kept as diagnostics:
  // a third part inside the annulus, then the most nearly concentric pair of
  // circles, then the pair whose worse boundary is closest to a circle.
  // (Grids of a pair's own parts between its circles mean the pair is not a
  // gap at all, as for a rotor and a stator's outer boundary.)
  struct Candidate {
    int inner, outer;
    Eigen::Vector2d center;
  };
  std::vector<Candidate> candidates;
  std::string occupied, offset_reason, shape_reason;
  double best_offset = std::numeric_limits<double>::infinity(),
         best_shape = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < loops.size(); ++i)
    for (size_t j = i + 1; j < loops.size(); ++j) {
      const Loop *a = &loops[i], *b = &loops[j];
      if (a->part == b->part)
        continue;
      if (a->radius > b->radius)
        std::swap(a, b);
      const double scale = b->radius;
      if (a->deviation > kAirGapTolerance || b->deviation > kAirGapTolerance) {
        const Loop *worse = a->deviation > b->deviation ? a : b;
        if (worse->deviation < best_shape) {
          best_shape = worse->deviation;
          shape_reason = std::format(
              "the boundary through grid {} on the part containing element {} "
              "deviates from a circle by {:.2e} of its radius at grid {} "
              "(tolerance {:.0e}; write GRID coordinates in large-field or "
              "free-field format if they lose digits in small fields)",
              grid(worse->nodes.front()), element_of_part(worse->part),
              worse->deviation, grid(worse->worst), kAirGapTolerance);
        }
        continue;
      }
      if (b->radius - a->radius <= kAirGapTolerance * scale)
        continue;
      const double offset = (a->center - b->center).norm();
      const Eigen::Vector2d center = (a->center + b->center) / 2;
      const Vector center3 = origin + center.x() * gap.u + center.y() * gap.v;
      if (offset > kAirGapTolerance * scale) {
        if (offset / scale < best_offset) {
          best_offset = offset / scale;
          offset_reason = std::format(
              "the boundary circles through grid {} (radius {:.6g}) and grid {} "
              "(radius {:.6g}) are not concentric: their centers are {:.3e} apart",
              grid(a->nodes.front()), a->radius, grid(b->nodes.front()), b->radius,
              offset);
        }
        continue;
      }
      int inside = -1, stranger = -1;
      for (int k = 0; k < n && stranger < 0; ++k) {
        const double r = (p[k] - center).norm();
        if (r > a->radius * (1 + kAirGapTolerance) &&
            r < b->radius * (1 - kAirGapTolerance)) {
          inside = k;
          if (parts.root(k) != a->part && parts.root(k) != b->part)
            stranger = k;
        }
      }
      if (stranger >= 0 && occupied.empty())
        occupied = std::format(
            "grid {} lies inside the annulus between the circles of radius "
            "{:.6g} and {:.6g} about ({:.6g}, {:.6g}, {:.6g})",
            grid(stranger), a->radius, b->radius, center3.x(), center3.y(),
            center3.z());
      if (inside < 0)
        candidates.push_back({static_cast<int>(a - loops.data()),
                              static_cast<int>(b - loops.data()), center});
    }
  const std::string reason = !occupied.empty()        ? occupied
                             : !offset_reason.empty() ? offset_reason
                             : !shape_reason.empty()  ? shape_reason
                                 : "no two parts have circular boundaries";

  if (candidates.size() > 1) {
    std::string list;
    for (const auto &c : candidates)
      list += std::format("{}{:.6g} to {:.6g}", list.empty() ? "" : "; ",
                          loops[c.inner].radius, loops[c.outer].radius);
    return Error{std::format("The magnetic mesh has {} candidate air gaps (radii {}); "
                             "only one air gap is supported",
                             candidates.size(), list)};
  }
  if (candidates.empty()) {
    return Error{std::format(
        "The magnetic mesh has {} parts not joined by elements, and no air gap "
        "joins them. An air gap is an unmeshed annulus between a circle of "
        "boundary grids on one part and a concentric circle on another, with no "
        "grid inside it; {}",
        roots.size(), reason)};
  }

  const auto &chosen = candidates.front();
  const Loop &inner = loops[chosen.inner], &outer = loops[chosen.outer];
  parts.join(inner.part, outer.part);
  const auto stray = std::find_if(roots.begin(), roots.end(), [&](int root) {
    return parts.root(root) != parts.root(inner.part);
  });
  if (stray != roots.end())
    return Error{std::format(
        "Element {} is not joined to the rest of the magnetic mesh by elements "
        "or by the air gap between radii {:.6g} and {:.6g}",
        element_of_part(*stray), inner.radius, outer.radius)};
  gap.center = origin + chosen.center.x() * gap.u + chosen.center.y() * gap.v;
  gap.inner = circle(inner, chosen.center, p);
  gap.outer = circle(outer, chosen.center, p);
  // Radii about the common center.
  for (GapCircle *c : {&gap.inner, &gap.outer})
    c->radius = std::accumulate(c->nodes.begin(), c->nodes.end(), 0.0,
                                [&](double sum, int i) {
                                  return sum + (p[i] - chosen.center).norm();
                                }) /
                static_cast<double>(c->nodes.size());
  gap.faces = inner.faces;
  gap.faces.insert(gap.faces.end(), outer.faces.begin(), outer.faces.end());
  gap.deviation = std::max(inner.deviation, outer.deviation);
  gap.harmonics = static_cast<int>(
      std::min(gap.inner.nodes.size(), gap.outer.nodes.size()) / 2);
  return gap;
}

Eigen::MatrixXd gap_transform(const GapCircle &c, int harmonics) {
  const auto n = static_cast<int>(c.nodes.size());
  Eigen::MatrixXd t = Eigen::MatrixXd::Zero(1 + 2 * harmonics, n);
  for (int k = 0; k < n; ++k) {
    // Segment [u, w] from node k to node k + 1, on which the function is
    // linear in angle: f = f_k (w - t) / d + f_{k+1} (t - u) / d.
    const int next = (k + 1) % n;
    const double u = c.angles[k],
                 w = next ? c.angles[next] : c.angles[0] + 2 * pi, d = w - u;
    t(0, k) += d / (4 * pi);
    t(0, next) += d / (4 * pi);
    for (int m = 1; m <= harmonics; ++m) {
      // Integrals of each end's weight times exp(i m t) over the segment.
      const std::complex<double> eu = std::polar(1.0, m * u),
                                 ew = std::polar(1.0, m * w),
                                 jump = (ew - eu) / double(m * m),
                                 i_over_m(0, 1.0 / m);
      const std::complex<double> start = (i_over_m * d * eu - jump) / d,
                                 end = (-i_over_m * d * ew + jump) / d;
      t(2 * m - 1, k) += start.real() / pi;
      t(2 * m, k) += start.imag() / pi;
      t(2 * m - 1, next) += end.real() / pi;
      t(2 * m, next) += end.imag() / pi;
    }
  }
  return t;
}

Eigen::MatrixXd air_gap_stiffness(const AirGap &gap) {
  const int h = gap.harmonics;
  const Eigen::MatrixXd ti = gap_transform(gap.inner, h),
                        to = gap_transform(gap.outer, h);
  // Energy per unit depth of each harmonic (cos or sin) with amplitudes
  // alpha inside and beta outside is (1/2) [alpha beta] S [alpha beta]^T.
  // The mean term is ln-radial: S = 2 pi / (mu0 L) [1 -1; -1 1].
  const double log_ratio = std::log(gap.inner.radius / gap.outer.radius); // ln rho < 0
  Eigen::VectorXd self(1 + 2 * h), mutual(1 + 2 * h);
  self[0] = -2 * pi / (mu0 * log_ratio);
  mutual[0] = -self[0];
  for (int m = 1; m <= h; ++m) {
    const double q = std::exp(m * log_ratio), // rho^m
        denominator = -std::expm1(2 * m * log_ratio), scale = pi * m / mu0;
    for (int row : {2 * m - 1, 2 * m}) {
      self[row] = scale * (1 + q * q) / denominator;
      mutual[row] = -scale * 2 * q / denominator;
    }
  }
  const auto ni = ti.cols(), no = to.cols();
  Eigen::MatrixXd k(ni + no, ni + no);
  k.topLeftCorner(ni, ni) = ti.transpose() * self.asDiagonal() * ti;
  k.topRightCorner(ni, no) = ti.transpose() * mutual.asDiagonal() * to;
  k.bottomLeftCorner(no, ni) = k.topRightCorner(ni, no).transpose();
  k.bottomRightCorner(no, no) = to.transpose() * self.asDiagonal() * to;
  return k;
}

bool in_air_gap(const AirGap &gap, const Vector &x) {
  const Vector d = x - gap.center;
  const double r = std::hypot(d.dot(gap.u), d.dot(gap.v));
  const double out_of_plane = std::abs(d.dot(gap.u.cross(gap.v)));
  return out_of_plane <= kAirGapTolerance * gap.outer.radius &&
         r >= gap.inner.radius && r <= gap.outer.radius;
}

Vector air_gap_field(const AirGap &gap, const Eigen::VectorXd &a, const Vector &x) {
  const auto [alpha, beta] = coefficients(gap, a);
  const Vector d = x - gap.center;
  const double px = d.dot(gap.u), py = d.dot(gap.v), r = std::hypot(px, py),
               theta = std::atan2(py, px);
  const double r1 = gap.inner.radius, total = std::log(gap.outer.radius / r1);
  const double from_inner = std::clamp(std::log(r / r1), 0.0, total),
               from_outer = total - from_inner;
  double da_dr = (beta[0] - alpha[0]) / (r * total), da_dtheta = 0;
  for (int m = 1; m <= gap.harmonics; ++m) {
    // sinh(m y)/sinh(m L) and cosh(m y)/sinh(m L), y = from_outer, and the
    // same with from_inner, written to stay finite for large m L.
    const double den = -std::expm1(-2 * m * total),
                 ex = std::exp(-2 * m * from_inner), ey = std::exp(-2 * m * from_outer),
                 sinh_y = std::exp(-m * from_inner) * (1 - ey) / den,
                 sinh_x = std::exp(-m * from_outer) * (1 - ex) / den,
                 cosh_y = std::exp(-m * from_inner) * (1 + ey) / den,
                 cosh_x = std::exp(-m * from_outer) * (1 + ex) / den;
    const double cos_part = alpha[2 * m - 1] * sinh_y + beta[2 * m - 1] * sinh_x,
                 sin_part = alpha[2 * m] * sinh_y + beta[2 * m] * sinh_x,
                 dcos = m / r * (-alpha[2 * m - 1] * cosh_y + beta[2 * m - 1] * cosh_x),
                 dsin = m / r * (-alpha[2 * m] * cosh_y + beta[2 * m] * cosh_x);
    const double c = std::cos(m * theta), s = std::sin(m * theta);
    da_dr += dcos * c + dsin * s;
    da_dtheta += m * (-cos_part * s + sin_part * c);
  }
  const Vector radial = std::cos(theta) * gap.u + std::sin(theta) * gap.v,
               tangential = -std::sin(theta) * gap.u + std::cos(theta) * gap.v;
  const Vector gradient = da_dr * radial + da_dtheta / r * tangential;
  return gradient.cross(gap.u.cross(gap.v));
}
} // namespace vibestran::magnetic
