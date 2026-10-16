// src/elements/shell_directors.cpp
// Shared shell grid directors; see shell_directors.hpp.

#include "elements/shell_directors.hpp"
#include <cmath>
#include <format>
#include <numbers>

namespace vibestran {

namespace {

/// Smallest accepted cosine between a SNORM and the element normal (about
/// 89.4 degrees, the limit MYSTRAN applies).
constexpr double kMinSnormCosine = 0.01;

bool is_shell(ElementType t) {
  return t == ElementType::CQUAD4 || t == ElementType::CTRIA3;
}

Eigen::Vector3d position(const Model &model, NodeId id) {
  const Vec3 &p = model.node(id).position;
  return {p.x, p.y, p.z};
}

std::shared_ptr<const Model::ShellAdjacency> adjacency(const Model &model) {
  return model.shell_adjacency_cache.get(model.elements.size(), [&model] {
    Model::ShellAdjacency adj;
    for (std::size_t e = 0; e < model.elements.size(); ++e) {
      const ElementData &el = model.elements[e];
      if (!is_shell(el.type))
        continue;
      const int n = el.type == ElementType::CQUAD4 ? 4 : 3;
      for (int k = 0; k < n && k < static_cast<int>(el.nodes.size()); ++k)
        adj[el.nodes[static_cast<std::size_t>(k)]].emplace_back(e, k);
    }
    return adj;
  });
}

/// SNORM normal in basic, unit length, sign-aligned with the element's own
/// corner normal.
Eigen::Vector3d snorm_director(const Model &model, const ElementData &elem,
                               const Snorm &sn, const Eigen::Vector3d &own) {
  Eigen::Vector3d n(sn.n.x, sn.n.y, sn.n.z);
  if (sn.cid.value != 0) {
    const auto cs = model.coord_systems.find(sn.cid);
    if (cs == model.coord_systems.end())
      throw SolverError(std::format("SNORM for grid {}: coordinate system {} is not "
                                    "defined",
                                    sn.grid.value, sn.cid.value));
    const Mat3 R = rotation_matrix(cs->second, model.node(sn.grid).position);
    const Vec3 b = apply_rotation(R, sn.n);
    n = {b.x, b.y, b.z};
  }
  n.normalize();
  double c = n.dot(own);
  if (c < 0.0) { // the element is numbered the other way round
    n = -n;
    c = -c;
  }
  // The director carries the thickness t / (n . V); a normal nearly in the
  // element's plane is not a usable director.
  if (c < kMinSnormCosine)
    throw SolverError(std::format(
        "Shell element {}: SNORM at grid {} is {:.2f} deg from the element normal "
        "(at most {:.2f} deg is allowed)",
        elem.id.value, sn.grid.value, std::acos(c) * 180.0 / std::numbers::pi,
        std::acos(kMinSnormCosine) * 180.0 / std::numbers::pi));
  return n;
}

} // namespace

double shell_fold_angle(const Model &model) {
  constexpr double kDefaultDegrees = 20.0;
  double deg = kDefaultDegrees;
  if (const auto it = model.params.find("SNORM"); it != model.params.end()) {
    try {
      const double v = std::stod(it->second);
      if (v > 0.0)
        deg = v;
    } catch (...) {
      // Non-numeric SNORM: keep the default.
    }
  }
  return deg * std::numbers::pi / 180.0;
}

Eigen::Vector3d shell_corner_normal(const Model &model, const ElementData &elem,
                                    int corner) {
  Eigen::Vector3d n;
  if (elem.type == ElementType::CQUAD4) {
    static constexpr std::array<double, 4> xi{-1.0, 1.0, 1.0, -1.0};
    static constexpr std::array<double, 4> eta{-1.0, -1.0, 1.0, 1.0};
    Eigen::Vector3d xr = Eigen::Vector3d::Zero(), xs = Eigen::Vector3d::Zero(),
                    xd = Eigen::Vector3d::Zero();
    for (int k = 0; k < 4; ++k) {
      const Eigen::Vector3d p = position(model, elem.nodes[static_cast<std::size_t>(k)]);
      xr += 0.25 * xi[static_cast<std::size_t>(k)] * p;
      xs += 0.25 * eta[static_cast<std::size_t>(k)] * p;
      xd += 0.25 * xi[static_cast<std::size_t>(k)] * eta[static_cast<std::size_t>(k)] * p;
    }
    const auto c = static_cast<std::size_t>(corner);
    n = (xr + eta[c] * xd).cross(xs + xi[c] * xd);
  } else {
    const Eigen::Vector3d a = position(model, elem.nodes[0]);
    const Eigen::Vector3d b = position(model, elem.nodes[1]);
    const Eigen::Vector3d c = position(model, elem.nodes[2]);
    n = (b - a).cross(c - a);
  }
  const double len = n.norm();
  if (!(len > 0.0))
    throw SolverError(std::format("Shell element {}: zero normal at corner {}",
                                  elem.id.value, corner + 1));
  return n / len;
}

Eigen::Vector3d shell_grid_director(const Model &model, const ElementData &elem,
                                    int corner, const Eigen::Vector3d &own) {
  const NodeId grid = elem.nodes[static_cast<std::size_t>(corner)];
  if (const auto sn = model.snorms.find(grid); sn != model.snorms.end())
    return snorm_director(model, elem, sn->second, own);
  const auto adj = adjacency(model);
  const auto it = adj->find(grid);
  if (it == adj->end())
    return own;
  const double cos_fold = std::cos(shell_fold_angle(model));
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  for (const auto &[e, k] : it->second) {
    Eigen::Vector3d nk = shell_corner_normal(model, model.elements[e], k);
    double c = nk.dot(own);
    if (c < 0.0) { // opposite element orientation
      nk = -nk;
      c = -c;
    }
    if (c >= cos_fold)
      sum += nk;
  }
  const double len = sum.norm();
  return len > 0.0 ? Eigen::Vector3d(sum / len) : own;
}

} // namespace vibestran
