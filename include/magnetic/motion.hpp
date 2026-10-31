#pragma once
// MOTION for planar MAGSTAT: rigid rotation of a set of cells about an axis,
// with the air-gap element between them and the fixed cells. Each position
// is solved as the static model with the moving grids, and the remanence
// frames of moving magnets, rotated to it: the deck an analyst would write
// for that position.
#include "magnetic/mesh.hpp"
#include <set>

namespace vibestran::magnetic {
// The axis of a MOTION: the z axis of its CID through the CID origin.
struct MotionAxis {
  Vector origin{Vector::Zero()}, direction{Vector::UnitZ()};
  [[nodiscard]] Matrix rotation(double degrees) const;
  // x turned rigidly by degrees about the axis.
  [[nodiscard]] Vector rotate(const Vector &x, double degrees) const;
};
[[nodiscard]] MotionAxis motion_axis(const Model &model, const Motion &motion);

// The cells a MOTION moves, in the mesh as given.
struct MovingPart {
  std::vector<bool> cells; // by field cell index
  std::set<int> grids;     // IDs of the moving cells' grids
  std::set<int> elements;  // IDs of the moving cells
  bool inner{true};        // the moving cells border the gap's inner circle
  // +1 when the axis points along the gap's plane normal u x v, else -1.
  double sign{1};
};
// The moving cells may share no grid with a fixed cell, must border exactly
// one circle of the air gap, and the axis must pass through the gap's
// center. Errors name the grids, elements and distances that fail.
[[nodiscard]] Result<MovingPart> moving_part(const Model &model, const Mesh &mesh,
                                             const Motion &motion);

// The static model at degrees: the moving grids rotated, and each moving
// magnet's remanence frame rotated as a new coordinate system. A magnetized
// property that fixed cells also use is split into a new property for the
// moving cells, so nothing fixed changes. Zero degrees returns the model
// unchanged.
[[nodiscard]] Model moved_model(const Model &model, const Motion &motion,
                                const MovingPart &part, double degrees);
} // namespace vibestran::magnetic
