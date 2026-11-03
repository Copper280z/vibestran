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

Vector MotionAxis::move(const Vector &x, double position) const {
  if (translate)
    return x + position * direction;
  return origin + rotation(position) * (x - origin);
}

MotionAxis motion_axis(const Model &model, const Motion &motion) {
  // A translation is along the MPERIOD shift's axis: x in MAG2D, z in MAGAXI.
  const int column = motion.translate && !model.options.axisymmetric ? 0 : 2;
  if (!motion.coordinates)
    return {Vector::Zero(), Vector::Unit(column), motion.translate};
  const auto &cs = model.coordinates.at(motion.coordinates);
  return {cs.a, coordinate_axes(cs).col(column), motion.translate};
}

namespace {
// What a side of the gap is: a circle, a line or a cylinder.
const char *side_word(const AirGap &gap) {
  switch (gap.shape) {
  case GapShape::Strip:
    return "line";
  case GapShape::Tube:
    return "cylinder";
  case GapShape::Annulus:
    break;
  }
  return "circle";
}
// A side of the gap as an error names it.
std::string side_name(const AirGap &gap, const GapCircle &side) {
  const auto name = std::format("{} {} of PAIRGAP {}", &side == &gap.inner ? "inner" : "outer",
                                side_word(gap), gap.property);
  return gap.shape == GapShape::Strip ? name
                                      : std::format("{} (radius {:.6g})", name, side.radius);
}
} // namespace

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
  if (const auto &period = mesh.period)
    for (auto [first, second] : period->pairs)
      if ((mover[first] >= 0) != (mover[second] >= 0))
        return Error{std::format(
            "MOTION {}: grid {} and its MPERIOD {} image grid {} are on different "
            "sides of the air gap, one moving (ESET {}) and one fixed; each "
            "part's periodic sides must pair within the part",
            motion.id, mesh.node_ids[first], period->id, mesh.node_ids[second],
            motion.set)};
  const auto &gap = *mesh.air_gap;
  // The parser pairs TRANSLATE with a gap along a line (MPERIOD SHIFT) and
  // ROTATE with an annulus.
  // Every side grid is on a field cell, so it is moving or fixed.
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
        "MOTION {}: the {} borders both moving and fixed elements, e.g. grid {} "
        "of the moving ESET {} and grid {}; the moving part must lie wholly on "
        "one side of the air gap",
        motion.id, side_name(gap, *c), mesh.node_ids[*moved], motion.set,
        mesh.node_ids[*fixed])};
  }
  const bool inner = moving_grids(gap.inner) > 0,
             outer = moving_grids(gap.outer) > 0;
  if (inner == outer)
    return Error{std::format(
        "MOTION {}: ESET {} borders {} {} of PAIRGAP {}; the moving part "
        "must border exactly one of them, the other side of the air gap being "
        "fixed",
        motion.id, motion.set, inner ? "both the inner and the outer" : "neither",
        side_word(gap), gap.property)};
  part.inner = inner;
  const auto axis = motion_axis(model, motion);
  if (motion.translate) {
    // The gap's lines run along u, the shift; the AGE slides only along them.
    // Round-off prints as zero.
    const auto shown = [](double x) { return std::abs(x) < 1e-12 ? 0.0 : x; };
    const Vector &d = axis.direction;
    if (d.cross(gap.u).norm() > 1e-9)
      return Error{std::format(
          "MOTION {}: the direction of travel, the {} axis of CID {} ({:.6g}, "
          "{:.6g}, {:.6g}), is not along PAIRGAP {}'s lines ({:.6g}, {:.6g}, "
          "{:.6g}), the MPERIOD shift; the part can only slide along the gap",
          motion.id, model.options.axisymmetric ? "z" : "x", motion.coordinates,
          shown(d.x()), shown(d.y()), shown(d.z()), gap.property, shown(gap.u.x()),
          shown(gap.u.y()), shown(gap.u.z()))};
    part.sign = d.dot(gap.u) > 0 ? 1 : -1;
    return part;
  }
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
                  const MovingPart &part, double position) {
  Model out = model;
  if (position == 0)
    return out;
  const auto axis = motion_axis(model, motion);
  for (int id : part.grids)
    out.nodes.at(id).position = axis.move(model.nodes.at(id).position, position);
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
        *point = axis.move(*point, position);
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
