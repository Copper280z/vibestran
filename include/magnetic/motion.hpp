#pragma once
// MOTION for planar and axisymmetric MAGSTAT: rigid rotation of a set of
// cells about an axis, or translation along a line, with the air-gap element
// between them and the fixed cells. Each position is solved as the static
// model with the moving grids, and the remanence frames of moving magnets,
// moved to it: the deck an analyst would write for that position.
#include "magnetic/mesh.hpp"
#include <set>

namespace vibestran::magnetic {
// The axis of a MOTION: for ROTATE the z axis of its CID through the CID
// origin; for TRANSLATE the direction of travel, the CID x axis (MAG2D) or z
// axis (MAGAXI).
struct MotionAxis {
  Vector origin{Vector::Zero()}, direction{Vector::UnitZ()};
  bool translate{false};
  [[nodiscard]] Matrix rotation(double degrees) const;
  // x moved rigidly to position: turned by that many degrees about the
  // axis, or moved that many metres along it.
  [[nodiscard]] Vector move(const Vector &x, double position) const;
};
[[nodiscard]] MotionAxis motion_axis(const Model &model, const Motion &motion);

// The cells a MOTION moves, in the mesh as given.
struct MovingPart {
  std::vector<bool> cells; // by field cell index
  std::set<int> grids;     // IDs of the moving cells' grids
  std::set<int> elements;  // IDs of the moving cells
  bool inner{true};        // the moving cells border the gap's inner side
  // +1 when the axis points along the gap's plane normal u x v (a
  // translation along u), else -1.
  double sign{1};
};
// The moving cells may share no grid with a fixed cell and must border
// exactly one side of the air gap. A rotation's axis must pass through the
// gap's center; a translation must run along the gap's lines. Errors name
// the grids, elements and distances that fail.
[[nodiscard]] Result<MovingPart> moving_part(const Model &model, const Mesh &mesh,
                                             const Motion &motion);

// The static model at position: the moving grids moved, and each moving
// magnet's remanence frame moved with them as a new coordinate system. A
// magnetized property that fixed cells also use is split into a new property
// for the moving cells, so nothing fixed changes. Position zero returns the
// model unchanged.
[[nodiscard]] Model moved_model(const Model &model, const Motion &motion,
                                const MovingPart &part, double position);
} // namespace vibestran::magnetic
