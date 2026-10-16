#pragma once
// include/elements/shell_directors.hpp
// Shared grid directors for the continuum-mechanics-based shell elements.
//
// Each shell grid carries one director (unit normal) shared by all shell
// elements that meet there smoothly: the average of the elements' own corner
// normals, taken over the elements whose normal is within the fold angle of
// the element in question.  Elements meeting at a fold (a T-junction, a
// stiffener, a box corner) keep separate directors.  Shared directors make
// the through-thickness kinematics compatible between elements; with
// element-local normals a curved mesh develops a near-mechanism about the
// weakly stabilised drill rotation (the Raasch hook diverged with
// refinement, while shared directors converge to the reference).
//
// A SNORM entry at a grid replaces the averaged director there for every
// shell element at the grid (sign-aligned with each element's normal); a
// SNORM more than about 89.4 degrees from an element's normal is rejected.
// At a SNORM grid the elements keep the shell thickness normal to the
// element surface: the director length is t / (V . n), as in MYSTRAN.
//
// The fold angle is PARAM,SNORM (degrees) when positive, otherwise 20
// degrees.  Nastran reads SNORM <= 0 as "no grid normals"; vibestran keeps
// shared directors regardless because element-local normals give wrong
// results on curved meshes.

#include "core/model.hpp"
#include <Eigen/Dense>

namespace vibestran {

/// Fold angle (radians) separating smooth from creased shell junctions.
[[nodiscard]] double shell_fold_angle(const Model &model);

/// Unit normal of a CQUAD4 (bilinear surface) or CTRIA3 (flat) element at
/// its local corner `corner`, oriented by the element's grid order.
[[nodiscard]] Eigen::Vector3d shell_corner_normal(const Model &model,
                                                  const ElementData &elem,
                                                  int corner);

/// Shared director at `corner` of `elem`: the grid's SNORM if one is given,
/// otherwise the normalised sum of the corner
/// normals of the shell elements at that grid within the fold angle of
/// `own` (the element's own corner normal), each sign-aligned with `own`.
[[nodiscard]] Eigen::Vector3d shell_grid_director(const Model &model,
                                                  const ElementData &elem,
                                                  int corner,
                                                  const Eigen::Vector3d &own);

} // namespace vibestran
