#pragma once
// include/elements/hexa20_shape.hpp
// Serendipity shape functions of the 20-node hexahedron in Nastran's CHEXA
// grid order:
//   G1..G8   corners at (xi, eta, zeta) = (-1,-1,-1), (1,-1,-1), (1,1,-1),
//            (-1,1,-1), (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1);
//   G9..G12  midside nodes of the bottom edges G1-G2, G2-G3, G3-G4, G4-G1;
//   G13..G16 midside nodes of the vertical edges G1-G5, G2-G6, G3-G7, G4-G8;
//   G17..G20 midside nodes of the top edges G5-G6, G6-G7, G7-G8, G8-G5.

#include <array>

namespace vibestran {

struct Hexa20ShapeData {
  std::array<double, 20> N{};
  /// dN[a][k] = dN_a / dxi_k.
  std::array<std::array<double, 3>, 20> dN{};
};

/// Natural coordinates of the 20 grids (corners, then the midside nodes).
inline constexpr std::array<std::array<double, 3>, 20> kHexa20Nodes{{
    {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
    {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1},
    {0, -1, -1},  {1, 0, -1},  {0, 1, -1}, {-1, 0, -1},
    {-1, -1, 0},  {1, -1, 0},  {1, 1, 0},  {-1, 1, 0},
    {0, -1, 1},   {1, 0, 1},   {0, 1, 1},  {-1, 0, 1},
}};

/// The two corners (0-based) of the edge of midside node 8..19.
inline constexpr std::array<std::array<int, 2>, 12> kHexa20Edges{{
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
}};

[[nodiscard]] inline Hexa20ShapeData hexa20_shape_functions(double xi, double eta,
                                                            double zeta) {
  const std::array<double, 3> x{xi, eta, zeta};
  Hexa20ShapeData s;
  for (std::size_t a = 0; a < 20; ++a) {
    const auto &c = kHexa20Nodes[a];
    if (a < 8) {
      // N = (1 + x.c)_k products / 8 * (c.x - 2)
      std::array<double, 3> f{};
      for (std::size_t k = 0; k < 3; ++k)
        f[k] = 1.0 + x[k] * c[k];
      const double t = x[0] * c[0] + x[1] * c[1] + x[2] * c[2] - 2.0;
      const double p = f[0] * f[1] * f[2];
      s.N[a] = p * t / 8.0;
      for (std::size_t k = 0; k < 3; ++k) {
        const double others = f[(k + 1) % 3] * f[(k + 2) % 3];
        s.dN[a][k] = c[k] * (others * t + p) / 8.0;
      }
    } else {
      // Midside node along axis z: N = (1 - x_z^2) prod_{k != z}(1 + x_k c_k) / 4
      std::size_t z = 0;
      while (c[z] != 0.0)
        ++z;
      std::array<double, 3> g{};
      std::array<double, 3> dg{};
      for (std::size_t k = 0; k < 3; ++k) {
        g[k] = k == z ? 1.0 - x[k] * x[k] : 1.0 + x[k] * c[k];
        dg[k] = k == z ? -2.0 * x[k] : c[k];
      }
      s.N[a] = g[0] * g[1] * g[2] / 4.0;
      s.dN[a][0] = dg[0] * g[1] * g[2] / 4.0;
      s.dN[a][1] = g[0] * dg[1] * g[2] / 4.0;
      s.dN[a][2] = g[0] * g[1] * dg[2] / 4.0;
    }
  }
  return s;
}

} // namespace vibestran
