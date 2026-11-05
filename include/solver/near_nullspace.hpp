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

    [[nodiscard]] int rows() const noexcept { return static_cast<int>(point.size()); }
};

/// Which near-nullspace structural solves give their AMG preconditioner.
enum class AmgModes {
    Constant, ///< one constant per aggregate, with aggregates of single equations
    Rigid     ///< rigid_body_modes(), with aggregates of whole grid points
};

/// --amg-modes and PARAM,AMGMODES values: CONSTANT or RIGID in any case.
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

} // namespace vibestran
