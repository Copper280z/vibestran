// src/solver/near_nullspace.cpp
// See include/solver/near_nullspace.hpp.

#include "solver/near_nullspace.hpp"
#include "core/exceptions.hpp"
#include "core/model.hpp"
#include "core/mpc_handler.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <format>

namespace vibestran {

// cppcheck-suppress unusedFunction -- called from linear_static.cpp and main.cpp
std::optional<AmgModes> parse_amg_modes(const std::string& text) {
    std::string upper = text;
    std::ranges::transform(upper, upper.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (upper == "CONSTANT")
        return AmgModes::Constant;
    if (upper == "RIGID")
        return AmgModes::Rigid;
    return std::nullopt;
}

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
std::shared_ptr<const NearNullspace> rigid_body_modes(const Model& model,
                                                      const MpcHandler& mpc_handler) {
    constexpr int kCols = 6;
    std::vector<NodeId> grids;
    grids.reserve(model.nodes.size());
    Vec3 centre{0, 0, 0};
    double located = 0;
    for (const auto& [id, grid] : model.nodes) {
        grids.push_back(id);
        if (grid.scalar)
            continue;
        centre.x += grid.position.x;
        centre.y += grid.position.y;
        centre.z += grid.position.z;
        located += 1;
    }
    std::ranges::sort(grids, [](NodeId a, NodeId b) { return a.value < b.value; });
    const auto count = std::max<double>(1.0, located);
    centre.x /= count;
    centre.y /= count;
    centre.z /= count;

    const auto n = static_cast<std::size_t>(mpc_handler.num_reduced());
    auto out = std::make_shared<NearNullspace>();
    out->cols = kCols;
    out->modes.assign(n * kCols, 0.0);
    out->point.assign(n, -1);
    const DofMap& full = mpc_handler.full_dof_map();
    for (const NodeId id : grids) {
        const GridPoint& grid = model.nodes.at(id);
        if (grid.scalar) {
            // A scalar point takes no part in rigid motion: its row of every
            // mode stays zero, so only the smoother and a coarse aggregate that
            // already holds the modes act on it.
            const EqIndex eq = mpc_handler.reduced_index(full.eq_index(id, 0));
            if (eq != CONSTRAINED_DOF)
                out->point[static_cast<std::size_t>(eq)] = out->points++;
            continue;
        }
        const Vec3& p = grid.position;
        const double x = p.x - centre.x, y = p.y - centre.y, z = p.z - centre.z;
        // Columns T1 T2 T3 R1 R2 R3 of the motion of this grid, by component.
        const std::array<std::array<double, kCols>, kCols> motion{{
            {1, 0, 0, 0, z, -y},
            {0, 1, 0, -z, 0, x},
            {0, 0, 1, y, -x, 0},
            {0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 1},
        }};
        bool used = false;
        for (int c = 0; c < kCols; ++c) {
            const EqIndex eq = mpc_handler.reduced_index(full.eq_index(id, c));
            if (eq == CONSTRAINED_DOF)
                continue;
            const auto row = static_cast<std::size_t>(eq);
            out->point[row] = out->points;
            std::ranges::copy(motion[static_cast<std::size_t>(c)],
                              out->modes.begin() + static_cast<std::ptrdiff_t>(row * kCols));
            used = true;
        }
        if (used)
            ++out->points;
    }
    if (const auto missing = std::ranges::find(out->point, -1); missing != out->point.end())
        throw SolverError(std::format(
            "rigid_body_modes: equation {} of {} belongs to no grid point ({} grids in the model)",
            std::distance(out->point.begin(), missing), n, grids.size()));
    return out;
}

} // namespace vibestran
