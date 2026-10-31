#include "magnetic/motion.hpp"
#include "magnetic/air_gap.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <format>
#include <numbers>

namespace vibestran::magnetic {

Matrix MotionAxis::rotation(double degrees) const {
  return Eigen::AngleAxisd(degrees * std::numbers::pi / 180, direction)
      .toRotationMatrix();
}

Vector MotionAxis::rotate(const Vector &x, double degrees) const {
  return origin + rotation(degrees) * (x - origin);
}

MotionAxis motion_axis(const Model &model, const Motion &motion) {
  if (!motion.coordinates)
    return {};
  const auto &cs = model.coordinates.at(motion.coordinates);
  return {cs.a, coordinate_axes(cs).col(2)};
}

Result<MovingPart> moving_part(const Model &model, const Mesh &mesh,
                               const Motion &motion) {
  MovingPart part;
  part.cells.assign(mesh.cells.size(), false);
  std::vector<int> mover(mesh.nodes.size(), -1); // a moving cell at each grid
  for (int id : model.element_sets.at(motion.set)) {
    const int ci = mesh.element_index.at(id);
    part.cells[ci] = true;
    part.elements.insert(id);
    for (int n : mesh.cells[ci].nodes) {
      mover[n] = ci;
      part.grids.insert(mesh.node_ids[n]);
    }
  }
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    if (part.cells[ci])
      continue;
    const auto &nodes = mesh.cells[ci].nodes;
    const auto shared = std::find_if(nodes.begin(), nodes.end(),
                                     [&](int n) { return mover[n] >= 0; });
    if (shared != nodes.end())
      return Error{std::format(
          "MOTION {}: grid {} belongs to moving element {} (ESET {}) and to "
          "fixed element {}; the moving and fixed parts may meet only across "
          "the air gap",
          motion.id, mesh.node_ids[*shared], mesh.cells[mover[*shared]].id,
          motion.set, mesh.cells[ci].id)};
  }
  const auto &gap = *mesh.air_gap;
  // Every circle grid is on a field cell, so it is moving or fixed.
  auto moving_grids = [&](const GapCircle &c) {
    return std::count_if(c.nodes.begin(), c.nodes.end(),
                         [&](int n) { return mover[n] >= 0; });
  };
  for (const auto *c : {&gap.inner, &gap.outer}) {
    const auto count = moving_grids(*c);
    if (count == 0 || count == static_cast<std::ptrdiff_t>(c->nodes.size()))
      continue;
    const auto moved = std::find_if(c->nodes.begin(), c->nodes.end(),
                                    [&](int n) { return mover[n] >= 0; });
    const auto fixed = std::find_if(c->nodes.begin(), c->nodes.end(),
                                    [&](int n) { return mover[n] < 0; });
    return Error{std::format(
        "MOTION {}: the {} circle of PAIRGAP {} (radius {:.6g}) borders both "
        "moving and fixed elements, e.g. grid {} of the moving ESET {} and "
        "grid {}; the moving part must lie wholly on one side of the air gap",
        motion.id, c == &gap.inner ? "inner" : "outer", gap.property, c->radius,
        mesh.node_ids[*moved], motion.set, mesh.node_ids[*fixed])};
  }
  const bool inner = moving_grids(gap.inner) > 0,
             outer = moving_grids(gap.outer) > 0;
  if (inner == outer)
    return Error{std::format(
        "MOTION {}: ESET {} borders {} circle of PAIRGAP {}; the moving part "
        "must border exactly one of them, the other side of the air gap being "
        "fixed",
        motion.id, motion.set, inner ? "both the inner and the outer" : "neither",
        gap.property)};
  part.inner = inner;
  const auto axis = motion_axis(model, motion);
  const Vector offset = gap.center - axis.origin;
  const double distance =
      (offset - offset.dot(axis.direction) * axis.direction).norm();
  if (distance > kAirGapTolerance * gap.outer.radius)
    return Error{std::format(
        "MOTION {}: the rotation axis, the z axis of CID {} through ({:.6g}, "
        "{:.6g}, {:.6g}), passes {:.3e} from the center of PAIRGAP {}; the air "
        "gap's circles must be centered on the axis",
        motion.id, motion.coordinates, axis.origin.x(), axis.origin.y(),
        axis.origin.z(), distance, gap.property)};
  part.sign = axis.direction.dot(gap.u.cross(gap.v)) > 0 ? 1 : -1;
  return part;
}

Model moved_model(const Model &model, const Motion &motion,
                  const MovingPart &part, double degrees) {
  Model out = model;
  if (degrees == 0)
    return out;
  const auto axis = motion_axis(model, motion);
  for (int id : part.grids)
    out.nodes.at(id).position = axis.rotate(model.nodes.at(id).position, degrees);
  int next_frame = model.coordinates.empty() ? 1 : model.coordinates.rbegin()->first + 1;
  int next_property = std::max(model.properties.empty() ? 0 : model.properties.rbegin()->first,
                               model.air_gaps.empty() ? 0 : model.air_gaps.rbegin()->first) +
                      1;
  std::map<int, int> moved; // magnetized property -> property of moving cells
  for (auto &e : out.elements) {
    if (!part.elements.contains(e.id))
      continue;
    const auto &property = model.properties.at(e.property);
    if (model.materials.at(property.material).br == 0)
      continue;
    auto [it, added] = moved.emplace(e.property, e.property);
    if (added) {
      // Basic is the rectangular frame at the origin with the basic axes.
      CoordinateSystem frame = property.coordinates
                                   ? model.coordinates.at(property.coordinates)
                                   : CoordinateSystem{};
      for (Vector *point : {&frame.a, &frame.b, &frame.c})
        *point = axis.rotate(*point, degrees);
      Property rotated = property;
      rotated.coordinates = next_frame;
      out.coordinates[next_frame++] = frame;
      const bool shared =
          std::any_of(model.elements.begin(), model.elements.end(),
                      [&](const Element &other) {
                        return other.property == e.property &&
                               !part.elements.contains(other.id);
                      });
      if (shared)
        it->second = next_property++;
      out.properties[it->second] = rotated;
    }
    e.property = it->second;
  }
  return out;
}
} // namespace vibestran::magnetic
