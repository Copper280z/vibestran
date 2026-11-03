#include "magnetic/air_gap.hpp"
#include <Eigen/QR>
#include <unsupported/Eigen/SpecialFunctions>
#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <map>
#include <numbers>
#include <set>
#include <tuple>
#include <numeric>

namespace vibestran::magnetic {
namespace {
constexpr double pi = std::numbers::pi;

// A closed chain of boundary faces, its grids in order.
struct Loop {
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
    c.sign.push_back(1);
  }
  return c;
}

// The arc of a sector model's gap unfolded around the machine (a line of a
// model that repeats along it, over the gap's length): the chain runs from
// a grid of the first periodic side to its image on the second, and copy j
// of each grid (but the last, which is the first's image) stands at its
// angle plus j times the chain's sweep, with sign (-1)^j when the model is
// anti-periodic. angle_of gives a grid's angle (a line's phase); around a
// circle it wraps, and successive angles are unwrapped.
Result<GapCircle> unfold(const Loop &chain, const auto &angle_of, const Mesh &mesh,
                         int pid, const char *which) {
  const Periodicity &period = *mesh.period;
  const bool line = period.translation;
  const int copies = line ? (period.anti ? 2 : 1) : period.sectors;
  const int first = chain.nodes.front(), last = chain.nodes.back();
  const auto paired = [&period](int a, int b) {
    return std::ranges::find(period.pairs, std::array<int, 2>{a, b}) !=
           period.pairs.end();
  };
  if (!paired(first, last) && !paired(last, first))
    return Error{std::format(
        "PAIRGAP {}: the {} {} of the air gap runs from grid {} to grid {}, "
        "which are not a grid of the first side of MPERIOD {} and its image. "
        "In a periodic model the gap's sides run from one periodic side to the "
        "other",
        pid, which, line ? "line" : "arc", mesh.node_ids[first], mesh.node_ids[last],
        period.id)};
  std::vector<int> nodes = chain.nodes;
  if (!paired(first, last))
    std::reverse(nodes.begin(), nodes.end());
  std::vector<double> theta{angle_of(nodes[0])};
  for (size_t t = 1; t < nodes.size(); ++t)
    theta.push_back(line ? angle_of(nodes[t])
                         : theta.back() + std::remainder(angle_of(nodes[t]) -
                                                             angle_of(nodes[t - 1]),
                                                         2 * pi));
  const double sweep = theta.back() - theta.front(), sector = 2 * pi / copies;
  if (std::abs(std::abs(sweep) - sector) > kAirGapTolerance * sector)
    return Error{
        line ? std::format("PAIRGAP {}: the {} line of the air gap runs {:.9g} of "
                           "the pitch along the shift from grid {} to grid {}; "
                           "it must run one pitch of MPERIOD {}",
                           pid, which, std::abs(sweep) / sector,
                           mesh.node_ids[nodes.front()], mesh.node_ids[nodes.back()],
                           period.id)
             : std::format("PAIRGAP {}: the {} arc of the air gap spans {:.9g} "
                           "degrees from grid {} to grid {}; MPERIOD {} has {} "
                           "sectors of {:.9g} degrees",
                           pid, which, std::abs(sweep) * 180 / pi,
                           mesh.node_ids[nodes.front()], mesh.node_ids[nodes.back()],
                           period.id, period.sectors, sector * 180 / pi)};
  for (size_t t = 1; t < nodes.size(); ++t)
    if ((theta[t] - theta[t - 1]) * sweep <= 0)
      return Error{std::format("PAIRGAP {}: the {} {} of the air gap turns back "
                               "at grid {}",
                               pid, which, line ? "line" : "arc",
                               mesh.node_ids[nodes[t]])};
  std::vector<std::tuple<double, int, double>> entries;
  for (int j = 0; j < copies; ++j)
    for (size_t t = 0; t + 1 < nodes.size(); ++t)
      entries.emplace_back(theta[t] + j * sweep, nodes[t],
                           period.anti && j % 2 ? -1. : 1.);
  std::sort(entries.begin(), entries.end());
  GapCircle c;
  for (const auto &[a, i, sign] : entries) {
    c.angles.push_back(a);
    c.nodes.push_back(i);
    c.sign.push_back(sign);
  }
  return c;
}

// A strip's or tube's sides from the two open chains of a model that
// repeats along a line: each a line along the shift (a tube's, along the
// axis at one radius), unfolded over the gap's length.
Status line_sides(AirGap &gap, std::vector<Loop> &loops, const Mesh &mesh, int pid) {
  const Periodicity &period = *mesh.period;
  const double pitch = period.shift.norm();
  const bool tube = mesh.axisymmetric;
  gap.shape = tube ? GapShape::Tube : GapShape::Strip;
  gap.sectors = period.anti ? 2 : 1;
  gap.length = gap.sectors * pitch;
  if (tube) {
    gap.center = mesh.axis_origin;
    gap.u = mesh.axis;
    gap.v = mesh.radial;
  } else {
    gap.center = mesh.nodes.front();
    gap.u = period.shift / pitch;
    gap.v = mesh.normal.cross(gap.u);
  }
  // Each chain's mean distance along v (radius) and its largest deviation,
  // over the pitch (or the radius, if larger).
  for (auto &loop : loops) {
    loop.radius = std::accumulate(loop.nodes.begin(), loop.nodes.end(), 0.0,
                                  [&](double sum, int i) {
                                    return sum + (mesh.nodes[i] - gap.center).dot(gap.v);
                                  }) /
                  static_cast<double>(loop.nodes.size());
    const double scale = tube ? std::max(pitch, loop.radius) : pitch;
    loop.deviation = 0;
    for (int i : loop.nodes) {
      const double d =
          std::abs((mesh.nodes[i] - gap.center).dot(gap.v) - loop.radius) / scale;
      if (d > loop.deviation) {
        loop.deviation = d;
        loop.worst = i;
      }
    }
  }
  if (loops[0].radius > loops[1].radius)
    std::swap(loops[0], loops[1]);
  const Loop &inner = loops[0], &outer = loops[1];
  for (const Loop *loop : {&inner, &outer})
    if (loop->deviation > kAirGapTolerance)
      return Error{
          tube ? std::format("PAIRGAP {}: the {} side of the tubular air gap is not "
                             "at one radius: grid {} is {:.2e} of the larger of the "
                             "pitch and the mean radius {:.6g} from it; the "
                             "tolerance is {:.0e}. Place the grids on a line along "
                             "the axis",
                             pid, loop == &inner ? "inner" : "outer",
                             mesh.node_ids[loop->worst], loop->deviation, loop->radius,
                             kAirGapTolerance)
               : std::format("PAIRGAP {}: the {} side of the air gap is not a line "
                             "along the shift of MPERIOD {}: grid {} is {:.2e} of the "
                             "pitch off it; the tolerance is {:.0e}. Place the grids "
                             "on a line along the shift",
                             pid, loop == &inner ? "inner" : "outer", period.id,
                             mesh.node_ids[loop->worst], loop->deviation,
                             kAirGapTolerance)};
  const double scale = tube ? std::max(pitch, outer.radius) : pitch;
  if (outer.radius - inner.radius <= kAirGapTolerance * scale)
    return Error{std::format("PAIRGAP {}: the air gap has no width (both sides at "
                             "{:.6g})",
                             pid, inner.radius)};
  if (tube && inner.radius <= kAirGapTolerance * scale)
    return Error{std::format("PAIRGAP {}: the inner side of the tubular air gap "
                             "is on the axis; the gap lies between two parts, off "
                             "the axis",
                             pid)};
  const auto phase = [&](int i) {
    return 2 * pi * (mesh.nodes[i] - gap.center).dot(gap.u) / gap.length;
  };
  for (auto [loop, c, which] : {std::tuple{&inner, &gap.inner, "inner"},
                                std::tuple{&outer, &gap.outer, "outer"}}) {
    auto unfolded = unfold(*loop, phase, mesh, pid, which);
    if (auto *e = std::get_if<Error>(&unfolded))
      return *e;
    *c = std::get<GapCircle>(std::move(unfolded));
    c->radius = loop->radius;
  }
  gap.faces = inner.faces;
  gap.faces.insert(gap.faces.end(), outer.faces.begin(), outer.faces.end());
  gap.deviation = std::max(inner.deviation, outer.deviation);
  gap.inner_chord = gap.inner.radius;
  return std::nullopt;
}

} // namespace

HarmonicStiffness harmonic_stiffness(const AirGap &gap, int m) {
  const double r1 = gap.inner.radius, r2 = gap.outer.radius;
  if (gap.shape == GapShape::Tube) {
    if (m == 0) {
      // A = a r + b / r: a uniform axial field, and 1/r, which has none.
      const double k = 4 * pi * gap.length / (mu0 * (r2 * r2 - r1 * r1));
      return {k * r1 * r1, k * r2 * r2, -k * r1 * r2};
    }
    // A = a I1(kappa r) + b K1(kappa r) times cos or sin(kappa s). With
    // x = kappa r at the sides, E = I1(x2) K1(x1) - I1(x1) K1(x2) and the
    // Wronskian I0 K1 + I1 K0 = 1/x, the energy (pi length / mu0) times
    // [x1 P1 alpha^2 + x2 P2 beta^2 - 2 alpha beta] / (2 E), with
    // P1 = I1(x2) K0(x1) + K1(x2) I0(x1) and P2 = K1(x1) I0(x2) + I1(x1) K0(x2).
    // In the exponentially scaled functions (I e^-x, K e^x) E and P1, P2
    // carry e^(x2 - x1), which cancels: no overflow at any harmonic.
    using Eigen::numext::bessel_i0e, Eigen::numext::bessel_i1e,
        Eigen::numext::bessel_k0e, Eigen::numext::bessel_k1e;
    const double kappa = 2 * pi * m / gap.length, x1 = kappa * r1, x2 = kappa * r2,
                 e = std::exp(-2 * (x2 - x1));
    const double i0_1 = bessel_i0e(x1), i1_1 = bessel_i1e(x1), k0_1 = bessel_k0e(x1),
                 k1_1 = bessel_k1e(x1), i0_2 = bessel_i0e(x2), i1_2 = bessel_i1e(x2),
                 k0_2 = bessel_k0e(x2), k1_2 = bessel_k1e(x2);
    const double den = i1_2 * k1_1 - e * i1_1 * k1_2, scale = pi * gap.length / mu0;
    return {scale * x1 * (i1_2 * k0_1 + e * k1_2 * i0_1) / den,
            scale * x2 * (k1_1 * i0_2 + e * i1_1 * k0_2) / den,
            -scale * std::exp(-(x2 - x1)) / den};
  }
  // An annulus in log-polar coordinates (ln r, theta) and a strip scaled by
  // 2 pi / length are both a rectangle of period 2 pi and thickness ell in
  // which A is harmonic with the same energy (the maps are conformal).
  const double log_ratio = gap.shape == GapShape::Annulus
                               ? std::log(r1 / r2)
                               : -2 * pi * (r2 - r1) / gap.length; // -ell
  if (m == 0) {
    const double k = -2 * pi / (mu0 * log_ratio);
    return {k, k, -k};
  }
  const double q = std::exp(m * log_ratio), // rho^m
      denominator = -std::expm1(2 * m * log_ratio), scale = pi * m / mu0;
  const double self = scale * (1 + q * q) / denominator;
  return {self, self, -scale * 2 * q / denominator};
}

namespace {
// The rate at which a move of the inner part turns harmonic m's (cos, sin)
// pair: per radian around an annulus, per unit length along a strip or tube.
double harmonic_rate(const AirGap &gap, int m) {
  return gap.shape == GapShape::Annulus ? m : 2 * pi * m / gap.length;
}

// A tube's radial functions of harmonic m at radius r: the potential with
// unit amplitude on the inner side and none on the outer (in), the reverse
// (out), and of each (1/r) d(r f)/dr, the axial field per unit amplitude.
struct Radial {
  double in{0}, out{0}, curl_in{0}, curl_out{0};
};
Radial tube_radial(const AirGap &gap, int m, double r) {
  const double r1 = gap.inner.radius, r2 = gap.outer.radius;
  if (m == 0) {
    const double area = r2 * r2 - r1 * r1;
    return {r1 * (r2 * r2 / r - r) / area, r2 * (r - r1 * r1 / r) / area, -2 * r1 / area,
            2 * r2 / area};
  }
  using Eigen::numext::bessel_i0e, Eigen::numext::bessel_i1e, Eigen::numext::bessel_k0e,
      Eigen::numext::bessel_k1e;
  const double kappa = 2 * pi * m / gap.length, x1 = kappa * r1, x2 = kappa * r2,
               x = kappa * r;
  const double i1_1 = bessel_i1e(x1), k1_1 = bessel_k1e(x1), i1_2 = bessel_i1e(x2),
               k1_2 = bessel_k1e(x2), i0 = bessel_i0e(x), i1 = bessel_i1e(x),
               k0 = bessel_k0e(x), k1 = bessel_k1e(x);
  const double den = i1_2 * k1_1 - std::exp(-2 * (x2 - x1)) * i1_1 * k1_2;
  // Each exponent is at most zero for r in [r1, r2].
  const double from_inner = std::exp(x1 - x), from_outer = std::exp(x - x2),
               to_outer = std::exp(-2 * (x2 - x)), to_inner = std::exp(-2 * (x - x1));
  return {from_inner * (i1_2 * k1 - to_outer * k1_2 * i1) / den,
          from_outer * (k1_1 * i1 - to_inner * i1_1 * k1) / den,
          -kappa * from_inner * (to_outer * k1_2 * i0 + i1_2 * k0) / den,
          kappa * from_outer * (k1_1 * i0 + to_inner * i1_1 * k0) / den};
}

// The potential's coefficients on both circles, for field evaluation.
struct Harmonics {
  Eigen::VectorXd inner, outer;
};
int prime_at_least(int n) {
  for (;; ++n) {
    bool prime = n >= 2;
    for (int d = 2; prime && d * d <= n; ++d)
      prime = n % d != 0;
    if (prime)
      return n;
  }
}

Harmonics coefficients(const AirGap &gap, const Eigen::VectorXd &a, int harmonics) {
  auto values = [&a](const GapCircle &c) {
    Eigen::VectorXd out(static_cast<Eigen::Index>(c.nodes.size()));
    for (size_t k = 0; k < c.nodes.size(); ++k)
      out[static_cast<Eigen::Index>(k)] = c.sign[k] * a[c.nodes[k]];
    return out;
  };
  return {gap_transform(gap.inner, harmonics) * values(gap.inner),
          gap_transform(gap.outer, harmonics) * values(gap.outer)};
}

Result<AirGap> band_and_harmonics(AirGap gap, const Mesh &mesh, const Model &model,
                                  int pid);
} // namespace

int own_harmonics(const GapCircle &c) { return static_cast<int>(c.nodes.size() / 2); }

int coupled_harmonics(const AirGap &gap) {
  return std::min(own_harmonics(gap.inner), own_harmonics(gap.outer));
}

Result<AirGap> make_air_gap(const Mesh &mesh, const Model &model, int pid) {
  AirGap gap;
  gap.property = pid;
  // Edges of the gap elements used once are the boundary of the gap region.
  struct Use {
    int count{0}, element{0};
  };
  std::map<std::array<int, 2>, Use> uses; // grid IDs, sorted
  for (const auto &e : model.elements) {
    if (e.property != pid)
      continue;
    gap.elements.push_back(e.id);
    for (size_t k = 0; k < e.nodes.size(); ++k) {
      std::array<int, 2> edge{e.nodes[k], e.nodes[(k + 1) % e.nodes.size()]};
      std::sort(edge.begin(), edge.end());
      auto &use = uses[edge];
      ++use.count;
      use.element = e.id;
    }
  }
  // Every boundary edge must be an edge of the field mesh (then its own
  // boundary, the gap being on the other side).
  std::map<int, std::vector<int>> boundary; // grid ID -> other grid IDs
  std::map<std::array<int, 2>, int> faces;  // grid IDs -> field face
  for (const auto &[edge, use] : uses) {
    if (use.count != 1)
      continue;
    const auto a = mesh.node_index.find(edge[0]), b = mesh.node_index.find(edge[1]);
    const auto face = a == mesh.node_index.end() || b == mesh.node_index.end()
                          ? mesh.face_index.end()
                          : mesh.face_index.find(NodeList{std::min(a->second, b->second),
                                                          std::max(a->second, b->second)});
    // In a sector model the gap elements' edges on the periodic sides
    // border nothing; the arcs between them are the gap's boundary.
    if (mesh.period && (face == mesh.face_index.end() ||
                        mesh.faces[face->second].cells[1] >= 0))
      continue;
    if (face == mesh.face_index.end() || mesh.faces[face->second].cells[1] >= 0)
      return Error{std::format(
          "PAIRGAP {}: edge ({},{}) of air-gap element {} is not shared with a field "
          "element. The air gap must lie between two meshed parts, the rotor and "
          "the stator: its elements' inner edges on one part's boundary and their "
          "outer edges on the other's",
          pid, edge[0], edge[1], use.element)};
    faces[edge] = face->second;
    boundary[edge[0]].push_back(edge[1]);
    boundary[edge[1]].push_back(edge[0]);
  }
  for (const auto &[grid, others] : boundary)
    if (others.size() > 2 || (others.size() != 2 && !mesh.period))
      return Error{std::format("PAIRGAP {}: the boundary of the air-gap elements "
                               "meets itself at grid {}",
                               pid, grid)};
  if (mesh.period)
    for (const auto &[edge, f] : faces)
      if (std::ranges::find(mesh.period->faces, f) != mesh.period->faces.end())
        return Error{std::format(
            "PAIRGAP {}: edge ({},{}) borders the air gap and is on a periodic "
            "side of MPERIOD {}; the periodic sides end at the gap",
            pid, edge[0], edge[1], mesh.period->id)};

  // Closed loops of boundary grids, in mesh indices; in a sector model,
  // the open arcs, walked from their ends.
  std::vector<Loop> loops;
  std::set<int> visited;
  if (mesh.period) {
    for (const auto &[first, others] : boundary) {
      if (others.size() != 1 || visited.contains(first))
        continue;
      Loop loop;
      int previous = -1, at = first;
      while (true) {
        visited.insert(at);
        loop.nodes.push_back(mesh.node_index.at(at));
        const auto &next = boundary.at(at);
        if (previous >= 0 && next.size() == 1)
          break;
        const int step = next[0] != previous ? next[0] : next[1];
        std::array<int, 2> edge{at, step};
        std::sort(edge.begin(), edge.end());
        loop.faces.push_back(faces.at(edge));
        previous = at;
        at = step;
      }
      loops.push_back(std::move(loop));
    }
    const auto closed = std::find_if(boundary.begin(), boundary.end(), [&](const auto &b) {
      return !visited.contains(b.first);
    });
    if (closed != boundary.end() || loops.size() != 2)
      return Error{std::format(
          "PAIRGAP {}: {} the air-gap elements must form {}, but their "
          "boundary has {} open {}{}",
          pid,
          mesh.period->translation
              ? std::format("in a model that repeats along a line (MPERIOD {})",
                            mesh.period->id)
              : std::format("in a sector model (MPERIOD {})", mesh.period->id),
          mesh.period->translation
              ? std::format("one pitch of a {}, bounded by a line on each of two "
                            "parts between the periodic sides",
                            mesh.axisymmetric ? "tube" : "strip")
              : "one sector of an annulus, bounded by an arc on the rotor and an "
                "arc on the stator between the periodic sides",
          loops.size(), loops.size() == 1 ? "side" : "sides",
          closed != boundary.end()
              ? std::format(" and a closed loop through grid {}", closed->first)
              : "")};
  }
  if (mesh.period && mesh.period->translation) {
    if (auto e = line_sides(gap, loops, mesh, pid))
      return *e;
    return band_and_harmonics(std::move(gap), mesh, model, pid);
  }
  for (const auto &[first, others] : boundary) {
    if (mesh.period)
      break;
    if (visited.contains(first))
      continue;
    Loop loop;
    int previous = -1, at = first;
    do {
      visited.insert(at);
      loop.nodes.push_back(mesh.node_index.at(at));
      const auto &next = boundary.at(at);
      const int step = next[0] != previous ? next[0] : next[1];
      std::array<int, 2> edge{at, step};
      std::sort(edge.begin(), edge.end());
      loop.faces.push_back(faces.at(edge));
      previous = at;
      at = step;
    } while (at != first);
    loops.push_back(std::move(loop));
  }
  if (loops.size() != 2)
    return Error{std::format(
        "PAIRGAP {}: the air-gap elements must form one annulus, bounded by a "
        "circle on the rotor and a circle on the stator, but their boundary has "
        "{} closed {}{}",
        pid, loops.size(), loops.size() == 1 ? "loop" : "loops",
        loops.size() == 1 ? " (a sector or a disk, not a full ring; a sector "
                            "model of the machine needs MPERIOD)"
                          : " (holes, or separate pieces)")};

  // In-plane frame with u x v = normal, and plane coordinates of the grids.
  const Vector &normal = mesh.normal;
  gap.u = normal.cross(std::abs(normal.x()) < .9 ? Vector::UnitX() : Vector::UnitY())
              .normalized();
  gap.v = normal.cross(gap.u);
  const Vector origin = mesh.nodes.front();
  std::vector<Eigen::Vector2d> p(mesh.nodes.size());
  for (size_t i = 0; i < mesh.nodes.size(); ++i)
    p[i] = {(mesh.nodes[i] - origin).dot(gap.u), (mesh.nodes[i] - origin).dot(gap.v)};
  for (auto &loop : loops)
    fit_circle(loop, p);
  if (loops[0].radius > loops[1].radius)
    std::swap(loops[0], loops[1]);
  const Loop &inner = loops[0], &outer = loops[1];
  for (const Loop *loop : {&inner, &outer})
    if (loop->deviation > kAirGapTolerance)
      return Error{std::format(
          "PAIRGAP {}: the {} boundary of the air gap is not a circle: grid {} "
          "deviates from the best-fit circle (radius {:.6g}) by {:.2e} of its "
          "radius; the tolerance is {:.0e}. Place the grids on a circle, and "
          "write GRID coordinates in large-field or free-field format if small "
          "fields lose digits",
          pid, loop == &inner ? "inner" : "outer", mesh.node_ids[loop->worst],
          loop->radius, loop->deviation, kAirGapTolerance)};
  const double offset = (inner.center - outer.center).norm();
  if (offset > kAirGapTolerance * outer.radius)
    return Error{std::format(
        "PAIRGAP {}: the inner and outer circles of the air gap (radii {:.6g} and "
        "{:.6g}) are not concentric: their centers are {:.3e} apart",
        pid, inner.radius, outer.radius, offset)};
  if (outer.radius - inner.radius <= kAirGapTolerance * outer.radius)
    return Error{std::format("PAIRGAP {}: the air gap has no width (radius {:.6g})",
                             pid, inner.radius)};

  const Eigen::Vector2d center = (inner.center + outer.center) / 2;
  gap.center = origin + center.x() * gap.u + center.y() * gap.v;
  if (mesh.period) {
    const Vector from_origin = gap.center - mesh.period->origin;
    const double distance =
        (from_origin - from_origin.dot(mesh.period->axis) * mesh.period->axis).norm();
    if (distance > kAirGapTolerance * outer.radius)
      return Error{std::format(
          "PAIRGAP {}: the center of the air gap's arcs is {:.3e} from the "
          "axis of MPERIOD {}; the sectors turn about the gap's center",
          pid, distance, mesh.period->id)};
    gap.sectors = mesh.period->sectors;
    const auto angle = [&](int i) {
      const Eigen::Vector2d q = p[i] - center;
      return std::atan2(q.y(), q.x());
    };
    for (auto [loop, c, which] : {std::tuple{&inner, &gap.inner, "inner"},
                                  std::tuple{&outer, &gap.outer, "outer"}}) {
      auto unfolded = unfold(*loop, angle, mesh, pid, which);
      if (auto *e = std::get_if<Error>(&unfolded))
        return *e;
      *c = std::get<GapCircle>(std::move(unfolded));
    }
  } else {
    gap.inner = circle(inner, center, p);
    gap.outer = circle(outer, center, p);
  }
  for (GapCircle *c : {&gap.inner, &gap.outer})
    c->radius = std::accumulate(c->nodes.begin(), c->nodes.end(), 0.0,
                                [&](double sum, int i) {
                                  return sum + (p[i] - center).norm();
                                }) /
                static_cast<double>(c->nodes.size());
  gap.faces = inner.faces;
  gap.faces.insert(gap.faces.end(), outer.faces.begin(), outer.faces.end());
  gap.deviation = std::max(inner.deviation, outer.deviation);
  // Points between the inner circle and its chords lie in the gap too.
  double widest = 0;
  for (size_t k = 0; k < gap.inner.angles.size(); ++k)
    widest = std::max(widest, (k + 1 < gap.inner.angles.size()
                                   ? gap.inner.angles[k + 1]
                                   : gap.inner.angles[0] + 2 * pi) -
                                  gap.inner.angles[k]);
  gap.inner_chord = gap.inner.radius * std::cos(std::min(widest, pi) / 2);

  return band_and_harmonics(std::move(gap), mesh, model, pid);
}

namespace {
// The gap's sliding band and NHARM, from its circles (lines).
Result<AirGap> band_and_harmonics(AirGap gap, const Mesh &mesh, const Model &model,
                                  int pid) {
  const auto fine = std::max(gap.inner.nodes.size(), gap.outer.nodes.size());
  const AirGapProperty &property = model.air_gaps.at(pid);
  const bool band =
      model.options.air_gap_method.value_or(property.method) == AirGapMethod::Band;
  if (band && gap.shape != GapShape::Annulus)
    return Error{std::format("PAIRGAP {}: the sliding band (BAND) of a gap along a "
                             "line is not available yet; METHOD=AGE solves it",
                             pid)};
  if (band) {
    // Blank NBAND: per sector, the smallest prime count of at least the
    // finer circle's. The band's error repeats with its point spacing, and
    // a count with a factor in common with the poles of a turning part
    // puts every pole at the same phase of it, so their errors add instead
    // of cancelling (a motor's cogging grew threefold on a symmetric mesh
    // whose finer circle had a multiple of the 14 poles).
    gap.band_points = property.band_points > 0
                          ? property.band_points
                          : gap.sectors * prime_at_least(static_cast<int>(
                                              (fine + gap.sectors - 1) / gap.sectors));
    // Blank NLAYER: about square elements, the layer's thickness in ln r
    // no more than an element's angle.
    const double thickness = std::log(gap.outer.radius / gap.inner.radius);
    gap.band_layers = property.band_layers > 0
                          ? property.band_layers
                          : std::max(1, static_cast<int>(std::ceil(
                                            thickness * gap.band_points / (2 * pi) - 1e-9)));
    if (mesh.period && mesh.period->anti)
      gap.image_sign = -1;
    // Each sector then samples its traces at the same points, so the
    // unfolded band is the sector's, repeated.
    if (gap.band_points % gap.sectors != 0)
      return Error{std::format(
          "PAIRGAP {}: NBAND {} is not a multiple of the {} sectors of the "
          "unfolded gap",
          pid, gap.band_points, gap.sectors)};
  }
  const int requested = property.harmonics;
  if (requested > static_cast<int>(fine / 2))
    return Error{std::format(
        "PAIRGAP {}: NHARM {} exceeds half the grid count of the finer circle "
        "({} grids), beyond which neither circle resolves a harmonic",
        pid, requested, fine)};
  gap.harmonics = requested > 0 ? requested : static_cast<int>(fine / 2);
  return gap;
}
} // namespace

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
  // Each circle's self terms run to the harmonics its own trace resolves;
  // the coupling runs to the coarser circle's. A self term |alpha_m|^2 is
  // the same at any turn of its circle, but the coarser circle's content
  // above its own resolution is an alias (harmonic n - m of a trace of n
  // grids turns as m, not n - m), and coupling it would make a round rotor
  // cog at its mesh pitch. Cutting the finer circle's self terms at the
  // coarser circle's resolution instead (NHARM's old blank) left its
  // shorter wavelengths with no gap stiffness, as if the gap were a free
  // boundary to them: on a motor mesh graded toward the stator that raised
  // the cogging torque 5.5%.
  const int hi = own_harmonics(gap.inner), ho = own_harmonics(gap.outer),
            hc = coupled_harmonics(gap), h = std::max(hi, ho);
  const Eigen::MatrixXd ti = gap_transform(gap.inner, hi),
                        to = gap_transform(gap.outer, ho);
  // Energy per unit depth of each harmonic (cos or sin) with amplitudes
  // alpha inside and beta outside is (1/2) [alpha beta] S [alpha beta]^T.
  Eigen::VectorXd self_inner(1 + 2 * hi), self_outer(1 + 2 * ho), mutual(1 + 2 * hc);
  for (int m = 0; m <= h; ++m) {
    const HarmonicStiffness s = harmonic_stiffness(gap, m);
    for (int row : m == 0 ? std::vector<int>{0} : std::vector<int>{2 * m - 1, 2 * m}) {
      if (m <= hi)
        self_inner[row] = s.inner;
      if (m <= ho)
        self_outer[row] = s.outer;
      if (m <= hc)
        mutual[row] = s.mutual;
    }
  }
  const Eigen::Index ni = ti.cols(), no = to.cols(), rc = 1 + 2 * hc;
  Eigen::MatrixXd k(ni + no, ni + no);
  k.topLeftCorner(ni, ni) = ti.transpose() * self_inner.asDiagonal() * ti;
  k.topRightCorner(ni, no) =
      ti.topRows(rc).transpose() * mutual.asDiagonal() * to.topRows(rc);
  k.bottomLeftCorner(no, ni) = k.topRightCorner(ni, no).transpose();
  k.bottomRightCorner(no, no) = to.transpose() * self_outer.asDiagonal() * to;
  return k;
}

bool in_air_gap(const AirGap &gap, const Vector &x) {
  const Vector d = x - gap.center;
  const double off_plane = std::abs(d.dot(gap.u.cross(gap.v)));
  if (gap.shape != GapShape::Annulus) {
    const double across = d.dot(gap.v),
                 scale = gap.shape == GapShape::Tube
                             ? std::max(gap.length, gap.outer.radius)
                             : gap.length;
    return off_plane <= kAirGapTolerance * scale && across >= gap.inner.radius &&
           across <= gap.outer.radius;
  }
  const double r = std::hypot(d.dot(gap.u), d.dot(gap.v));
  return off_plane <= kAirGapTolerance * gap.outer.radius &&
         r >= gap.inner_chord && r <= gap.outer.radius;
}

AirGapSeries::AirGapSeries(const AirGap &gap, const Eigen::VectorXd &a) : gap_(&gap) {
  // Each circle's harmonics to its own resolution, at most NHARM: the
  // stiffness's (above the coarser circle's, only the finer circle has any).
  const int h = std::max(own_harmonics(gap.inner), own_harmonics(gap.outer));
  auto [inner, outer] = coefficients(gap, a, h);
  for (auto [c, values] : {std::pair{&gap.inner, &inner}, std::pair{&gap.outer, &outer}})
    values->tail(2 * (h - std::min(own_harmonics(*c), gap.harmonics))).setZero();
  inner_ = std::move(inner);
  outer_ = std::move(outer);
}

// The gap's solution at x: A and its derivatives.
AirGapSeries::Point AirGapSeries::evaluate(const Vector &x) const {
  const Vector d = x - gap_->center;
  Point p;
  const double r1 = gap_->inner.radius, r2 = gap_->outer.radius;
  if (gap_->shape == GapShape::Tube) {
    // Clamped to the tube: a probe at a side is on it to round-off.
    p.r = std::clamp(d.dot(gap_->v), r1, r2);
    p.e_across = gap_->v;
    p.e_along = gap_->u;
    const double s = d.dot(gap_->u);
    double curl = 0; // (1/r) d(r A)/dr
    for (int m = 0; 2 * m < inner_.size(); ++m) {
      const Radial f = tube_radial(*gap_, m, p.r);
      if (m == 0) {
        p.a += inner_[0] * f.in + outer_[0] * f.out;
        curl += inner_[0] * f.curl_in + outer_[0] * f.curl_out;
        continue;
      }
      const double kappa = 2 * pi * m / gap_->length, c = std::cos(kappa * s),
                   sn = std::sin(kappa * s);
      const double cos_part = inner_[2 * m - 1] * f.in + outer_[2 * m - 1] * f.out,
                   sin_part = inner_[2 * m] * f.in + outer_[2 * m] * f.out;
      p.a += cos_part * c + sin_part * sn;
      curl += (inner_[2 * m - 1] * f.curl_in + outer_[2 * m - 1] * f.curl_out) * c +
              (inner_[2 * m] * f.curl_in + outer_[2 * m] * f.curl_out) * sn;
      p.along += kappa * (-cos_part * sn + sin_part * c);
    }
    p.across = curl - p.a / p.r;
    return p;
  }
  // The rectangle the annulus (in log-polar coordinates) or strip maps to:
  // xi from 0 on the inner side to total on the outer, theta of period 2 pi.
  double theta = 0, from_inner = 0, total = 0, metric = 0; // metric: length per unit
  if (gap_->shape == GapShape::Strip) {
    const double scale = 2 * pi / gap_->length;
    theta = scale * d.dot(gap_->u);
    total = scale * (r2 - r1);
    from_inner = std::clamp(scale * (d.dot(gap_->v) - r1), 0.0, total);
    metric = 1 / scale;
    p.e_across = gap_->v;
    p.e_along = gap_->u;
  } else {
    const double px = d.dot(gap_->u), py = d.dot(gap_->v);
    p.r = std::hypot(px, py);
    theta = std::atan2(py, px);
    total = std::log(r2 / r1);
    // Slightly negative between the inner circle and its chords, where the
    // annulus solution continues smoothly.
    from_inner = std::min(std::log(p.r / r1), total);
    metric = p.r;
    p.e_across = std::cos(theta) * gap_->u + std::sin(theta) * gap_->v;
    p.e_along = -std::sin(theta) * gap_->u + std::cos(theta) * gap_->v;
  }
  const double from_outer = total - from_inner;
  p.a = inner_[0] + (outer_[0] - inner_[0]) * from_inner / total;
  double d_xi = (outer_[0] - inner_[0]) / total, d_theta = 0;
  for (int m = 1; 2 * m < inner_.size(); ++m) {
    // sinh(m y)/sinh(m L) and cosh(m y)/sinh(m L), y = from_outer, and the
    // same with from_inner, written to stay finite for large m L.
    const double den = -std::expm1(-2 * m * total),
                 ex = std::exp(-2 * m * from_inner), ey = std::exp(-2 * m * from_outer),
                 sinh_y = std::exp(-m * from_inner) * (1 - ey) / den,
                 sinh_x = std::exp(-m * from_outer) * (1 - ex) / den,
                 cosh_y = std::exp(-m * from_inner) * (1 + ey) / den,
                 cosh_x = std::exp(-m * from_outer) * (1 + ex) / den;
    const double cos_part = inner_[2 * m - 1] * sinh_y + outer_[2 * m - 1] * sinh_x,
                 sin_part = inner_[2 * m] * sinh_y + outer_[2 * m] * sinh_x,
                 dcos = m * (-inner_[2 * m - 1] * cosh_y + outer_[2 * m - 1] * cosh_x),
                 dsin = m * (-inner_[2 * m] * cosh_y + outer_[2 * m] * cosh_x);
    const double c = std::cos(m * theta), s = std::sin(m * theta);
    p.a += cos_part * c + sin_part * s;
    d_xi += dcos * c + dsin * s;
    d_theta += m * (-cos_part * s + sin_part * c);
  }
  p.across = d_xi / metric;
  p.along = d_theta / metric;
  return p;
}

Vector AirGapSeries::field(const Vector &x) const {
  const auto p = evaluate(x);
  const Vector gradient = p.across * p.e_across + p.along * p.e_along;
  Vector b = gradient.cross(gap_->u.cross(gap_->v));
  // A tube's A is A_phi: curl(A e_phi) adds A / r along the axis.
  if (gap_->shape == GapShape::Tube)
    b += p.a / p.r * gap_->u;
  return b;
}

double AirGapSeries::potential(const Vector &x) const { return evaluate(x).a; }

double air_gap_torque(const AirGap &gap, const Eigen::VectorXd &a) {
  // Rotating the inner circle by delta turns its (cos, sin) pair of harmonic
  // m at rate (-m s, m c). Only the mutual term of W changes, by
  // m M_m (alpha_c beta_s - alpha_s beta_c) per radian, with M_m the mutual
  // coefficient of air_gap_stiffness. Along a strip or tube a move delta
  // turns them at rate kappa = 2 pi m / length instead.
  const int h = coupled_harmonics(gap);
  const auto [alpha, beta] = coefficients(gap, a, h);
  double torque = 0;
  for (int m = 1; m <= h; ++m)
    torque -= harmonic_rate(gap, m) * harmonic_stiffness(gap, m).mutual *
              (alpha[2 * m - 1] * beta[2 * m] - alpha[2 * m] * beta[2 * m - 1]);
  return torque;
}

Eigen::VectorXd air_gap_torque_gradient(const AirGap &gap, const Eigen::VectorXd &a) {
  // The torque is bilinear in the two circles' harmonics: its derivative
  // with respect to alpha is linear in beta and the reverse, mapped back to
  // the grids through each circle's transform.
  const int h = coupled_harmonics(gap);
  const auto [alpha, beta] = coefficients(gap, a, h);
  Eigen::VectorXd d_alpha = Eigen::VectorXd::Zero(alpha.size()),
                  d_beta = Eigen::VectorXd::Zero(beta.size());
  for (int m = 1; m <= h; ++m) {
    const double rate = -harmonic_rate(gap, m) * harmonic_stiffness(gap, m).mutual;
    const int c = 2 * m - 1, s = 2 * m;
    d_alpha[c] = rate * beta[s];
    d_alpha[s] = -rate * beta[c];
    d_beta[s] = rate * alpha[c];
    d_beta[c] = -rate * alpha[s];
  }
  Eigen::VectorXd out = Eigen::VectorXd::Zero(a.size());
  for (const auto &[circle, d] :
       {std::pair{&gap.inner, &d_alpha}, std::pair{&gap.outer, &d_beta}}) {
    const Eigen::VectorXd nodal =
        gap_transform(*circle, h).transpose() * *d;
    for (size_t k = 0; k < circle->nodes.size(); ++k)
      out[circle->nodes[k]] += circle->sign[k] * nodal[static_cast<Eigen::Index>(k)];
  }
  return out;
}
} // namespace vibestran::magnetic

namespace vibestran::magnetic {
namespace {
// The band grid's values on one circle. Value k is the trace f averaged
// over band point k's hat function h_k (width 2 Delta, integral Delta):
// g_k = (1/Delta) integral h_k f. Point samples of a piecewise-linear trace
// on a grid that does not line up with its grids alias the trace's
// grid-scale content into the band and stall convergence; the average
// integrates the trace exactly and damps that content. For a turn delta of
// the circle's part, f(psi - delta), dg_k/d(delta) = (1/Delta) integral
// h_k' f, the trace's integrals over the two band elements beside point k
// over Delta^2. Rows are weights on the circle's entries.
struct BandRows {
  Eigen::SparseMatrix<double> value, turn;
};
BandRows band_rows(const AirGap &gap, const GapCircle &c) {
  const int n = gap.band_points;
  const auto m = static_cast<Eigen::Index>(c.nodes.size());
  const double h = 2 * pi / n;
  std::vector<Eigen::Triplet<double>> value, turn;
  // Segment j runs from entry j to entry j + 1 (the last wraps around).
  for (Eigen::Index j = 0; j < m; ++j) {
    const Eigen::Index next = (j + 1) % m;
    const double u = c.angles[static_cast<size_t>(j)],
                 w = next ? c.angles[static_cast<size_t>(next)] : c.angles[0] + 2 * pi;
    // The segment's pieces within each band element [k h, (k + 1) h],
    // unrolled from psi = 0 by whole turns.
    for (double shift : {-2 * pi, 0.0, 2 * pi}) {
      const double from = std::max(u + shift, 0.0), to = std::min(w + shift, 2 * pi);
      if (to <= from)
        continue;
      for (int k = static_cast<int>(from / h); k < n && k * h < to; ++k) {
        const double a = std::max(from, k * h), b = std::min(to, (k + 1) * h);
        if (b <= a)
          continue;
        // Simpson's rule on [a, b], exact for the products of the linear
        // node functions and hats; the node functions alone integrate
        // exactly at the midpoint.
        const double points[3] = {a, (a + b) / 2, b}, weights[3] = {1, 4, 1};
        double left[2]{}, right[2]{}, plain[2];
        for (int q = 0; q < 3; ++q) {
          const double psi = points[q], wq = weights[q] * (b - a) / 6;
          const double node[2] = {(w + shift - psi) / (w - u), (psi - u - shift) / (w - u)};
          const double rising = (psi - k * h) / h; // h_{k+1}; h_k is 1 - rising
          for (int e = 0; e < 2; ++e) {
            left[e] += wq * (1 - rising) * node[e];
            right[e] += wq * rising * node[e];
          }
        }
        const double middle = (a + b) / 2;
        plain[0] = (b - a) * (w + shift - middle) / (w - u);
        plain[1] = (b - a) * (middle - u - shift) / (w - u);
        const int here = k, there = (k + 1) % n;
        const Eigen::Index entry[2] = {j, next};
        for (int e = 0; e < 2; ++e) {
          value.emplace_back(here, entry[e], left[e] / h);
          value.emplace_back(there, entry[e], right[e] / h);
          // h_k' = -1/Delta on this element, h_{k+1}' = +1/Delta.
          turn.emplace_back(here, entry[e], -plain[e] / (h * h));
          turn.emplace_back(there, entry[e], plain[e] / (h * h));
        }
      }
    }
  }
  BandRows out{Eigen::SparseMatrix<double>(n, m), Eigen::SparseMatrix<double>(n, m)};
  out.value.setFromTriplets(value.begin(), value.end());
  out.turn.setFromTriplets(turn.begin(), turn.end());
  return out;
}

// The band's stiffness per unit depth over its (M + 1) N point values,
// ring r (0 on the inner circle, M on the outer) at r N + k. The energy
// density in (xi, theta) is (1/(2 mu0)) ((1/L) A_xi^2 + L A_theta^2),
// L = ln(r2/r1), and each element is the bilinear rectangle
// [r/M, (r+1)/M] x [psi_k, psi_k+1].
Eigen::SparseMatrix<double> ring_stiffness(const AirGap &gap) {
  const int n = gap.band_points, m = gap.band_layers;
  const double l = std::log(gap.outer.radius / gap.inner.radius), h = 2 * pi / n;
  // 1-D stiffness [1 -1; -1 1] / w and mass w [1/3 1/6; 1/6 1/3] on an
  // interval of width w: 1/M radially, h in angle.
  const double stiff[2][2] = {{1, -1}, {-1, 1}}, mass[2][2] = {{1. / 3, 1. / 6}, {1. / 6, 1. / 3}};
  std::vector<Eigen::Triplet<double>> t;
  t.reserve(16 * static_cast<size_t>(n) * static_cast<size_t>(m));
  for (int ring = 0; ring < m; ++ring)
    for (int k = 0; k < n; ++k) {
      const int around[2] = {k, (k + 1) % n};
      for (int ri = 0; ri < 2; ++ri)
        for (int ai = 0; ai < 2; ++ai)
          for (int rj = 0; rj < 2; ++rj)
            for (int aj = 0; aj < 2; ++aj) {
              const double value = (stiff[ri][rj] * m * mass[ai][aj] * h / l +
                                    l * mass[ri][rj] / m * stiff[ai][aj] / h) /
                                   mu0;
              t.emplace_back((ring + ri) * n + around[ai], (ring + rj) * n + around[aj],
                             value);
            }
    }
  Eigen::SparseMatrix<double> k((m + 1) * n, (m + 1) * n);
  k.setFromTriplets(t.begin(), t.end());
  return k;
}

// The band's map from gap_entries to its point values, and the derivative
// of those values for a turn of the inner part (its interior rings, fixed
// in space, held).
struct Band {
  Eigen::SparseMatrix<double> map, turn;
};
Band band_map(const AirGap &gap) {
  const BandRows inner = band_rows(gap, gap.inner), outer = band_rows(gap, gap.outer);
  const int n = gap.band_points, m = gap.band_layers;
  const auto ni = static_cast<int>(gap.inner.nodes.size()),
             no = static_cast<int>(gap.outer.nodes.size());
  const int columns = ni + no + (m - 1) * n;
  std::vector<Eigen::Triplet<double>> map, turn;
  for (int k = 0; k < inner.value.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(inner.value, k); it; ++it)
      map.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
  for (int k = 0; k < outer.value.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(outer.value, k); it; ++it)
      map.emplace_back(m * n + static_cast<int>(it.row()), ni + static_cast<int>(it.col()),
                       it.value());
  for (int p = 0; p < (m - 1) * n; ++p)
    map.emplace_back(n + p, ni + no + p, 1.0);
  for (int k = 0; k < inner.turn.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(inner.turn, k); it; ++it)
      turn.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
  Band out{Eigen::SparseMatrix<double>((m + 1) * n, columns),
           Eigen::SparseMatrix<double>((m + 1) * n, columns)};
  out.map.setFromTriplets(map.begin(), map.end());
  out.turn.setFromTriplets(turn.begin(), turn.end());
  return out;
}

// The values of gap_entries for the unknowns a.
Eigen::VectorXd entries(const AirGap &gap, const Eigen::VectorXd &a) {
  const auto list = gap_entries(gap);
  Eigen::VectorXd out(static_cast<Eigen::Index>(list.size()));
  for (size_t k = 0; k < list.size(); ++k)
    out[static_cast<Eigen::Index>(k)] = list[k].second * a[list[k].first];
  return out;
}

// Maps a gradient over gap_entries to the unknowns.
Eigen::VectorXd to_unknowns(const AirGap &gap, const Eigen::VectorXd &d, Eigen::Index size) {
  const auto list = gap_entries(gap);
  Eigen::VectorXd out = Eigen::VectorXd::Zero(size);
  for (size_t k = 0; k < list.size(); ++k)
    out[list[k].first] += list[k].second * d[static_cast<Eigen::Index>(k)];
  return out;
}
} // namespace

std::vector<std::pair<int, double>> gap_entries(const AirGap &gap) {
  std::vector<std::pair<int, double>> out;
  for (const GapCircle *c : {&gap.inner, &gap.outer})
    for (size_t k = 0; k < c->nodes.size(); ++k)
      out.emplace_back(c->nodes[k], c->sign[k]);
  if (gap.band_points == 0)
    return out;
  const int per = gap.band_points / gap.sectors;
  for (int ring = 1; ring < gap.band_layers; ++ring)
    for (int k = 0; k < gap.band_points; ++k)
      out.emplace_back(gap.band_first + (ring - 1) * per + k % per,
                       (k / per) % 2 ? gap.image_sign : 1.0);
  return out;
}

Eigen::SparseMatrix<double> band_stiffness(const AirGap &gap) {
  const Band band = band_map(gap);
  return Eigen::SparseMatrix<double>(band.map.transpose() * ring_stiffness(gap) * band.map);
}

double band_torque(const AirGap &gap, const Eigen::VectorXd &a) {
  // W = (1/2) g^T K g with g = map x; turning the inner part changes g at
  // rate turn x, so -dW/d(delta) = -(K g) . (turn x).
  const Band band = band_map(gap);
  const Eigen::VectorXd x = entries(gap, a), g = band.map * x;
  return -(ring_stiffness(gap) * g).dot(band.turn * x);
}

Eigen::VectorXd band_torque_gradient(const AirGap &gap, const Eigen::VectorXd &a) {
  const Band band = band_map(gap);
  const Eigen::SparseMatrix<double> k = ring_stiffness(gap);
  const Eigen::VectorXd x = entries(gap, a);
  const Eigen::VectorXd d =
      -(band.map.transpose() * (k * (band.turn * x)) + band.turn.transpose() * (k * (band.map * x)));
  return to_unknowns(gap, d, a.size());
}
} // namespace vibestran::magnetic
