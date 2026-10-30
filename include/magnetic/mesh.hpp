#pragma once
#include "magnetic/model.hpp"
#include <Eigen/SparseCore>
#include <span>

namespace vibestran::magnetic {
inline constexpr std::array<std::array<int, 2>, 6> local_edges{
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
struct Cell {
  int id{0}, material{0};
  std::array<int, 4> nodes{}, faces{};
  std::array<int, 6> edges{};
  std::array<int, 4> face_sign{};
  std::array<int, 6> edge_sign{};
  std::array<Vector, 4> gradient;
  std::array<Vector, 6> curl;
  Matrix inverse;
  Vector origin, minimum, maximum;
  double volume{0};
};
struct Face {
  std::array<int, 3> nodes{};
  std::array<int, 2> cells{-1, -1};
  Vector normal{Vector::Zero()}, center{Vector::Zero()};
  double area{0};
};
struct Mesh {
  std::vector<Vector> nodes;
  std::vector<int> node_ids;
  std::vector<Cell> cells;
  std::vector<std::array<int, 2>> edges;
  std::vector<Face> faces;
  std::map<int, int> element_index, node_index;
  std::map<std::array<int, 3>, int> face_index;
};
struct Sample {
  Vector position;
  double weight;
};
[[nodiscard]] Result<Mesh>
make_mesh(std::vector<Vector> nodes,
          const std::vector<std::array<int, 4>> &cells);
[[nodiscard]] Result<Mesh> make_mesh(const Model &model);
[[nodiscard]] std::array<double, 4> barycentric(const Cell &cell,
                                                const Vector &x);
[[nodiscard]] std::array<Vector, 6> edge_basis(const Cell &cell,
                                               const Vector &x);
[[nodiscard]] std::array<Vector, 4>
flux_basis(const Mesh &mesh, const Cell &cell, const Vector &x);
[[nodiscard]] std::array<Sample, 4> quadrature(const Mesh &mesh,
                                               const Cell &cell);
[[nodiscard]] Result<std::vector<std::pair<int, int>>>
selected_faces(const Model &model, const Mesh &mesh, int selection);
[[nodiscard]] std::vector<Sample> intersection_quadrature(const Mesh &a,
                                                          const Cell &ca,
                                                          const Mesh &b,
                                                          const Cell &cb);
class SpatialIndex {
public:
  explicit SpatialIndex(const Mesh &mesh);
  [[nodiscard]] std::vector<int> query(const Vector &minimum,
                                       const Vector &maximum) const;

private:
  struct Node {
    Vector lo, hi;
    int left{-1}, right{-1}, begin{0}, end{0};
  };
  std::vector<Node> tree_;
  std::vector<int> order_;
  int build(const Mesh &mesh, int begin, int end);
};
struct Gauge {
  std::vector<int> equations;
  std::vector<bool> boundary;
  int size{0};
};
[[nodiscard]] Result<Gauge> make_gauge(const Mesh &mesh);
} // namespace vibestran::magnetic
