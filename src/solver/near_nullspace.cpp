// src/solver/near_nullspace.cpp
// See include/solver/near_nullspace.hpp.

#include "solver/near_nullspace.hpp"
#include "core/exceptions.hpp"
#include "core/model.hpp"
#include "core/mpc_handler.hpp"
#include "elements/shell_directors.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <unordered_map>

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
    if (upper == "DRILLING")
        return AmgModes::Drilling;
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

namespace {

/// What the drilling columns need to know about one grid of a CQUAD4.
struct DrillingGrid {
    Eigen::Vector3d normal{0, 0, 0}; ///< sum of the directors at the grid
    bool fold{false};
    std::vector<int> neighbours; ///< across CQUAD4 edges, as indices of the grids
    int sign{0};                 ///< colour of the alternating pattern; 0 until visited
};

/// Every grid that a CQUAD4 uses, in ascending ID order so that the result does
/// not depend on the storage order of the model.
std::pair<std::vector<NodeId>, std::vector<DrillingGrid>> drilling_grids(const Model& model) {
    std::vector<NodeId> ids;
    for (const auto& elem : model.elements)
        if (elem.type == ElementType::CQUAD4)
            ids.insert(ids.end(), elem.nodes.begin(), elem.nodes.end());
    std::ranges::sort(ids, [](NodeId a, NodeId b) { return a.value < b.value; });
    const auto [last, end] = std::ranges::unique(ids, [](NodeId a, NodeId b) { return a.value == b.value; });
    ids.erase(last, end);

    std::unordered_map<int, int> index;
    index.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i)
        index[ids[i].value] = static_cast<int>(i);

    std::vector<DrillingGrid> grids(ids.size());
    const double smooth = std::cos(shell_fold_angle(model));
    std::array<int, 4> corner_grid{};
    for (const auto& elem : model.elements) {
        const bool quad = elem.type == ElementType::CQUAD4;
        if (!quad && elem.type != ElementType::CTRIA3)
            continue;
        const auto corners = elem.nodes.size();
        for (std::size_t c = 0; c < corners; ++c) {
            const auto found = index.find(elem.nodes[c].value);
            corner_grid[c] = found == index.end() ? -1 : found->second;
        }
        for (std::size_t c = 0; c < corners; ++c) {
            if (corner_grid[c] < 0)
                continue;
            DrillingGrid& grid = grids[static_cast<std::size_t>(corner_grid[c])];
            // Summed over a smooth grid's elements, the corner normals give the
            // direction of shell_grid_director(), which would average them all
            // again at every corner.
            const int corner = static_cast<int>(c);
            Eigen::Vector3d director = shell_corner_normal(model, elem, corner);
            if (model.snorms.contains(elem.nodes[c]))
                director = shell_grid_director(model, elem, corner, director);
            if (grid.normal.squaredNorm() > 0) {
                const Eigen::Vector3d seen = grid.normal.normalized();
                if (seen.dot(director) < 0)
                    director = -director;
                grid.fold = grid.fold || seen.dot(director) < smooth;
            }
            grid.normal += director;
            if (quad) {
                grid.neighbours.push_back(corner_grid[(c + 1) % corners]);
                grid.neighbours.push_back(corner_grid[(c + corners - 1) % corners]);
            }
        }
    }
    return {std::move(ids), std::move(grids)};
}

} // namespace

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
std::shared_ptr<const NearNullspace> drilling_modes(const Model& model,
                                                    const MpcHandler& mpc_handler) {
    const auto rigid = rigid_body_modes(model, mpc_handler);
    auto [ids, grids] = drilling_grids(model);
    if (ids.empty())
        return rigid;

    // One walk per connected shell. Each grid's normal is turned to agree with
    // the grid it was reached from, and the colour alternates. A mesh that cannot
    // be two-coloured, such as a closed ring of an odd number of elements, keeps
    // whatever the walk reached first: the pattern is then a little stiffer along
    // that seam, which costs iterations there and nothing else.
    std::vector<int> queue;
    for (std::size_t root = 0; root < grids.size(); ++root) {
        if (grids[root].sign != 0)
            continue;
        grids[root].sign = 1;
        queue.assign(1, static_cast<int>(root));
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const DrillingGrid& from = grids[static_cast<std::size_t>(queue[head])];
            for (const int next : from.neighbours) {
                DrillingGrid& to = grids[static_cast<std::size_t>(next)];
                if (to.sign != 0)
                    continue;
                to.sign = -from.sign;
                // A fold has no normal of its own, so it passes on the direction
                // it was reached with.
                if (to.fold)
                    to.normal = from.normal;
                else if (to.normal.dot(from.normal) < 0)
                    to.normal = -to.normal;
                queue.push_back(next);
            }
        }
    }

    constexpr int kExtra = 2;
    const auto n = static_cast<std::size_t>(rigid->rows());
    const auto base = static_cast<std::size_t>(rigid->cols);
    const auto cols = base + kExtra;
    auto out = std::make_shared<NearNullspace>();
    out->cols = static_cast<int>(cols);
    out->optional_cols = kExtra;
    out->point = rigid->point;
    out->points = rigid->points;
    out->modes.assign(n * cols, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        std::copy_n(rigid->modes.begin() + static_cast<std::ptrdiff_t>(i * base), base,
                    out->modes.begin() + static_cast<std::ptrdiff_t>(i * cols));
    const DofMap& full = mpc_handler.full_dof_map();
    for (std::size_t g = 0; g < grids.size(); ++g) {
        if (grids[g].fold)
            continue;
        const Eigen::Vector3d axis = grids[g].normal.normalized();
        for (int c = 0; c < 3; ++c) {
            const EqIndex eq = mpc_handler.reduced_index(full.eq_index(ids[g], 3 + c));
            if (eq == CONSTRAINED_DOF)
                continue;
            const auto row = static_cast<std::size_t>(eq) * cols + base;
            out->modes[row] = axis[c];
            out->modes[row + 1] = grids[g].sign * axis[c];
        }
    }
    return out;
}

} // namespace vibestran
