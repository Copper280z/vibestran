#pragma once
#include "magnetic/model.hpp"
#include <Eigen/SparseCore>
#include <optional>
#include <span>

namespace vibestran::magnetic {
inline constexpr std::array<std::array<int, 2>, 6> local_edges{
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
struct Cell {
  int id{0}, material{0}, property{0};
  NodeList nodes, faces, edges, face_sign, edge_sign;
  std::vector<Vector> gradient, curl, points;
  std::array<Vector, 8> map_coefficients;
  Matrix inverse{Matrix::Identity()}, jacobian{Matrix::Identity()};
  Vector origin, map_origin, minimum, maximum;
  Vector normal{Vector::UnitZ()};
  bool planar{false}, affine{false};
  bool polyhedral{true}; // Convex with flat faces; eligible for plane clipping.
  double volume{0};
};
struct Face {
  NodeList nodes;
  std::array<int, 2> cells{-1, -1};
  // Canonical normal at the face center. Use SurfaceSample.normal for
  // integration on bilinear faces.
  Vector normal{Vector::Zero()}, center{Vector::Zero()};
  double area{0};
  bool flat{true};
};
// One circle of an air gap: boundary grids on a circle about AirGap::center,
// in increasing angle.
struct GapCircle {
  double radius{0};
  std::vector<int> nodes;     // mesh node indices
  std::vector<double> angles; // radians in [angles[0], angles[0] + 2 pi)
};
// An unmeshed air annulus of a planar mesh between a circle of boundary
// grids on one part of the mesh and a concentric circle on another, coupled
// analytically by the air-gap element (air_gap.hpp).
struct AirGap {
  Vector center{Vector::Zero()};
  Vector u{Vector::UnitX()}, v{Vector::UnitY()}; // in-plane, u x v = normal
  GapCircle inner, outer;
  std::vector<int> faces; // boundary faces on either circle
  double deviation{0};    // largest |distance - radius| / radius of a grid
  int harmonics{0};
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
  std::optional<AirGap> air_gap; // planar only
  [[nodiscard]] size_t dofs() const {
    return planar ? nodes.size() : edges.size();
  }
};
struct Sample {
  Vector position;
  double weight;
};
struct SurfaceSample {
  Vector position, normal;
  double weight;
};
[[nodiscard]] Result<Mesh> make_mesh(std::vector<Vector> nodes,
                                     const std::vector<NodeList> &cells,
                                     bool planar = false,
                                     const Vector &normal = Vector::UnitZ());
// Fixed-size connectivity convenience overload, currently used only by tests.
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
// Cache the RT0 coefficients for repeated evaluation of one source cell.
// weighted_reference returns J * j_reference, i.e. physical current times
// the volume Jacobian, without an inverse coordinate mapping.
class CurrentField {
public:
  CurrentField(const Cell &cell, const Eigen::VectorXd &flux);
  [[nodiscard]] Vector value(const Vector &x) const;
  [[nodiscard]] Vector weighted_reference(const Vector &r,
                                          const Matrix &jacobian) const;

private:
  const Cell &cell_;
  Vector offset_{Vector::Zero()}, slope_{Vector::Zero()},
      affine_offset_{Vector::Zero()};
  Matrix affine_slope_{Matrix::Zero()};
};
[[nodiscard]] std::vector<Sample> quadrature(const Mesh &mesh,
                                             const Cell &cell);
[[nodiscard]] std::vector<SurfaceSample>
face_quadrature(const Mesh &mesh, const Face &face, int order = 5);
[[nodiscard]] std::vector<Vector> reference_vertices(const Cell &cell);
[[nodiscard]] Vector reference_coordinates(const Cell &cell, const Vector &x);
[[nodiscard]] bool valid_mapping(const Cell &cell, double minimum_determinant);
[[nodiscard]] bool point_on_face(const Mesh &mesh, const Face &face,
                                 const Vector &x, double tolerance = 1e-9);
[[nodiscard]] std::vector<double> line_face_intersections(const Mesh &mesh,
                                                          const Face &face,
                                                          const Vector &first,
                                                          const Vector &last);
[[nodiscard]] std::vector<std::pair<double, double>> gauss_rule(int order);
// Test positive penetration into a convex, planar-faced source cell. Contact
// is excluded; unresolved bounds return an error instead of missing an overlap.
[[nodiscard]] Result<bool> cell_intersects_interior(const Mesh &poly_mesh,
                                                    const Cell &poly,
                                                    const Cell &other);
[[nodiscard]] Result<bool> face_intersects_interior(const Mesh &poly_mesh,
                                                    const Cell &poly,
                                                    const Mesh &mesh,
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
// Requires both cells to be convex and planar-faced (Cell::polyhedral).
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
