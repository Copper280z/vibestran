#pragma once
#include "magnetic/model.hpp"
#include <Eigen/SparseCore>
#include <span>

namespace vibestran::magnetic {
inline constexpr std::array<std::array<int, 2>, 6> local_edges{
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
struct Cell {
  int id{0}, material{0};
  NodeList nodes, faces, edges, face_sign, edge_sign;
  std::vector<Vector> gradient, curl, points;
  std::array<Vector, 8> map_coefficients;
  Matrix inverse{Matrix::Identity()}, jacobian{Matrix::Identity()};
  Vector origin, map_origin, minimum, maximum;
  Vector normal{Vector::UnitZ()};
  bool planar{false}, affine{false};
  double volume{0};
};
struct Face {
  NodeList nodes;
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
  std::map<NodeList, int> face_index;
  bool planar{false};
  double depth{1};
  Vector normal{Vector::UnitZ()};
  [[nodiscard]] size_t dofs() const {
    return planar ? nodes.size() : edges.size();
  }
};
struct Sample {
  Vector position;
  double weight;
};
[[nodiscard]] Result<Mesh> make_mesh(std::vector<Vector> nodes,
                                     const std::vector<NodeList> &cells,
                                     bool planar = false,
                                     const Vector &normal = Vector::UnitZ());
template <size_t N>
[[nodiscard]] Result<Mesh>
make_mesh(std::vector<Vector> nodes,
          const std::vector<std::array<int, N>> &cells) {
  return make_mesh(std::move(nodes),
                   std::vector<NodeList>(cells.begin(), cells.end()));
}
[[nodiscard]] Result<Mesh> make_mesh(const Model &model);
// Tetrahedral barycentric coordinates.
[[nodiscard]] std::array<double, 4> barycentric(const Cell &cell,
                                                const Vector &x);
[[nodiscard]] bool contains(const Cell &cell, const Vector &x,
                            double tolerance = 1e-10);
[[nodiscard]] std::vector<Vector> edge_basis(const Cell &cell, const Vector &x);
[[nodiscard]] std::vector<Vector> edge_curls(const Cell &cell, const Vector &x);
[[nodiscard]] std::vector<Vector> flux_basis(const Mesh &mesh, const Cell &cell,
                                             const Vector &x);
[[nodiscard]] Vector flux_value(const Cell &cell, const Eigen::VectorXd &flux,
                                const Vector &x);
[[nodiscard]] std::vector<Sample> quadrature(const Mesh &mesh,
                                             const Cell &cell);
[[nodiscard]] std::vector<Sample> face_quadrature(const Mesh &mesh,
                                                  const Face &face);
// Reference-cell geometry and mapping used by mesh validation and basis
// evaluation.
struct Shape {
  std::vector<double> values;
  std::vector<Vector> gradients;
};
[[nodiscard]] Shape reference_shape(const Cell &cell, const Vector &r);
[[nodiscard]] std::pair<Vector, Matrix> mapping(const Cell &cell,
                                                const Vector &r);
void initialize_mapping(Cell &cell);
[[nodiscard]] std::vector<Sample> reference_quadrature(const Cell &cell);
[[nodiscard]] std::vector<std::array<int, 2>> cell_edges(const Cell &cell);
[[nodiscard]] std::vector<NodeList> cell_faces(const Cell &cell);
[[nodiscard]] Result<std::vector<std::pair<int, int>>>
selected_faces(const Model &model, const Mesh &mesh, int selection);
[[nodiscard]] std::vector<Sample>
intersection_quadrature(const Mesh &a, const Cell &ca, const Mesh &b,
                        const Cell &cb, int order = 2);
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
