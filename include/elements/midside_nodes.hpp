#pragma once
// include/elements/midside_nodes.hpp
// Omitted midside grids of the quadratic solids (CTETRA10, CHEXA20).
//
// Nastran lets any midside grid of a CTETRA or CHEXA be left blank; the
// parser stores it as NodeId{0}.  The element then interpolates that edge
// linearly: the omitted grid's shape function is folded into its two corners
// (N_a += N_m / 2, N_b += N_m / 2, N_m = 0), which keeps the partition of
// unity and linear completeness, so the patch test still holds.  For the
// geometry this is the same as placing the omitted grid at the midpoint of
// its edge.  The omitted grid has no DOFs (the DOF map has no entry for
// NodeId{0}, so its equations are CONSTRAINED_DOF).

#include "core/model.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace vibestran {

/// The two corners (0-based) of the edge of CTETRA10 midside grid 4..9:
/// G5 on G1-G2, G6 on G2-G3, G7 on G3-G1, G8 on G1-G4, G9 on G2-G4, G10 on G3-G4.
inline constexpr std::array<std::array<int, 2>, 6> kTetra10Edges{{
    {0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3},
}};

/// The first N grids of `nodes` as an array (NodeId has no default).
template <std::size_t N, class Range>
[[nodiscard]] std::array<NodeId, N> element_node_array(const Range &nodes) {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::array<NodeId, N>{nodes[I]...};
  }(std::make_index_sequence<N>{});
}

[[nodiscard]] inline bool is_omitted_grid(NodeId n) noexcept { return n.value == 0; }

/// Grid positions with each omitted midside grid at the midpoint of its edge.
/// The last E of the N grids are the midside grids of `edges`.
template <std::size_t N, std::size_t E>
[[nodiscard]] std::array<Vec3, N>
quadratic_solid_positions(const Model &model, const std::array<NodeId, N> &nodes,
                          const std::array<std::array<int, 2>, E> &edges) {
  constexpr std::size_t first = N - E;
  std::array<Vec3, N> x{};
  for (std::size_t a = 0; a < first; ++a)
    x[a] = model.node(nodes[a]).position;
  for (std::size_t k = 0; k < E; ++k) {
    const std::size_t m = first + k;
    x[m] = is_omitted_grid(nodes[m])
               ? (x[static_cast<std::size_t>(edges[k][0])] +
                  x[static_cast<std::size_t>(edges[k][1])]) *
                     0.5
               : model.node(nodes[m]).position;
  }
  return x;
}

/// Folds the values (a shape function or one of its derivatives) of each
/// omitted midside grid into the two corners of its edge.
template <std::size_t N, std::size_t E>
void fold_omitted_midside(const std::array<NodeId, N> &nodes,
                          const std::array<std::array<int, 2>, E> &edges,
                          std::array<double, N> &v) {
  constexpr std::size_t first = N - E;
  for (std::size_t k = 0; k < E; ++k) {
    const std::size_t m = first + k;
    if (!is_omitted_grid(nodes[m]))
      continue;
    v[static_cast<std::size_t>(edges[k][0])] += 0.5 * v[m];
    v[static_cast<std::size_t>(edges[k][1])] += 0.5 * v[m];
    v[m] = 0.0;
  }
}

} // namespace vibestran
