#pragma once
// include/solver/near_nullspace.hpp
// The low-energy modes an algebraic multigrid must represent on every level.
//
// Aggregation with one constant per aggregate represents a scalar field's
// nullspace, but not a structure's: a rigid rotation is linear in position and
// couples translations to rotations. Without it, the iterations of a structural
// model grow at the sqrt(cond) rate of a one-level preconditioner.

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vibestran {

class Model;
class MpcHandler;

/// Vectors of the reduced system that a preconditioner should treat as nearly
/// singular, and the grid point that owns each equation.
struct NearNullspace {
    int cols{0};
    /// equations x cols, row by row.
    std::vector<double> modes;
    /// The grid point number of each equation; numbers run from 0 without gaps.
    std::vector<int> point;
    int points{0};
    /// The last columns that exist on part of the model only. An aggregate
    /// carries one only where it is not zero and not in the span of the columns
    /// before it, so the rest of the model pays nothing for it.
    int optional_cols{0};

    [[nodiscard]] int rows() const noexcept { return static_cast<int>(point.size()); }
};

/// Which near-nullspace structural solves give their AMG preconditioner.
enum class AmgModes {
    Constant, ///< one constant per aggregate, with aggregates of single equations
    Rigid,    ///< rigid_body_modes(), with aggregates of whole grid points
    Drilling  ///< drilling_modes(), with aggregates of whole grid points
};

/// --amg-modes and PARAM,AMGMODES values: CONSTANT, RIGID or DRILLING in any case.
[[nodiscard]] std::optional<AmgModes> parse_amg_modes(const std::string& text);

/// The six rigid-body motions of the model, about its centroid, on the
/// equations of the reduced system. The equations are in basic axes whatever a
/// grid's displacement frame (CD affects only constraints and output), and each
/// motion that satisfies the model's MPCs and RBEs is the image of its
/// independent DOFs under the elimination, so restricting it to them is exact:
/// K_reduced * mode vanishes away from single-point constraints, grounded
/// elements and springs to scalar points, whose rows of the modes are zero.
/// Grid points are numbered in ascending ID order.
[[nodiscard]] std::shared_ptr<const NearNullspace>
rigid_body_modes(const Model& model, const MpcHandler& mpc_handler);

/// rigid_body_modes() plus the two patterns of rotation about the shell normal
/// that a CQUAD4 hardly resists, as optional columns. Its membrane strain depends
/// only on differences of drilling rotation between grids, and on a rectangle a
/// uniform rotation and an alternating one (+,-,+,-) around each element leave
/// those strains at zero, so only the drilling penalty holds them. Neither is a
/// rigid motion, since a rigid rotation about the normal also moves the grids.
/// The columns are zero on grids with no CQUAD4 and at folds, where there is no
/// one normal. The normals of a connected shell are aligned with each other, so
/// the element orientation does not matter. A model with no CQUAD4 gets exactly
/// rigid_body_modes().
[[nodiscard]] std::shared_ptr<const NearNullspace>
drilling_modes(const Model& model, const MpcHandler& mpc_handler);

} // namespace vibestran
