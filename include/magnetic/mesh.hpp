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
  // MAGAXI: the symmetry axis through axis_origin along axis, and the
  // radial direction of the meridian half-plane; curl(A e_phi) adds
  // (A / r) axis to the planar curl, normal being e_phi.
  Vector axis_origin{Vector::Zero()}, axis{Vector::UnitZ()}, radial{Vector::UnitX()};
  bool planar{false}, affine{false}, axisymmetric{false};
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
// In a periodic sector model the circle is the arc of the sector unfolded
// around the machine: each copy of an arc grid stands for its image in
// another sector, with the factor sign (-1 in an odd sector of an
// anti-periodic model).
struct GapCircle {
  double radius{0};
  std::vector<int> nodes;     // mesh node indices
  std::vector<double> angles; // radians in [angles[0], angles[0] + 2 pi)
  std::vector<double> sign;   // potential at angles[k] = sign[k] a[nodes[k]]
};
// The air annulus of a PAIRGAP property between a circle of boundary grids
// on one part of the field mesh and a concentric circle on another, coupled
// analytically by the air-gap element (air_gap.hpp).
struct AirGap {
  int property{0};
  std::vector<int> elements; // PAIRGAP element IDs, not field cells
  Vector center{Vector::Zero()};
  Vector u{Vector::UnitX()}, v{Vector::UnitY()}; // in-plane, u x v = normal
  GapCircle inner, outer;
  std::vector<int> faces; // boundary faces on either circle
  double deviation{0};    // largest |distance - radius| / radius of a grid
  double inner_chord{0};  // nearest approach of the inner circle's edges
  int harmonics{0}; // the gap field series' (NHARM); the AGE's energy sums all the circles resolve
  int band_points{0}; // METHOD=BAND: the sliding band's grid; 0 for the AGE
  int band_layers{1}; // the band's element rings, between M + 1 point rings
  // The band's interior point rings (1..M-1) are unknowns after the mesh's
  // nodes, from band_first: ring r's point k is unknown band_first +
  // (r - 1) N/sectors + k mod N/sectors, times image_sign^(k div N/sectors)
  // in a sector model.
  int band_first{0};
  double image_sign{1};
  int sectors{1}; // the circles are unfolded from one of this many sectors
};
// The two sides of an MPERIOD on the field mesh: each grid of the first
// side with its image on the second, turned by 2 pi / sectors
// counterclockwise about the axis.
struct Periodicity {
  int id{0}, sectors{1};
  bool anti{false};
  Vector origin{Vector::Zero()}, axis{Vector::UnitZ()};
  std::vector<int> faces;                // boundary faces on either side
  std::vector<std::array<int, 2>> pairs; // node indices: first side, its image
  std::vector<int> axial;                // grids on both sides, on the axis
  // The factor from a grid's potential to its image's.
  [[nodiscard]] double sign() const { return anti ? -1 : 1; }
};
struct Mesh {
  std::vector<Vector> nodes;
  std::vector<int> node_ids;
  std::vector<Cell> cells;
  std::vector<std::array<int, 2>> edges;
  std::vector<Face> faces;
  std::map<int, int> element_index, node_index;
  std::map<NodeList, int> face_index;
  bool planar{false}, axisymmetric{false}; // axisymmetric: MAGAXI, planar too
  double depth{1};                         // MAG2D; 1 otherwise
  Vector normal{Vector::UnitZ()};
  Vector axis_origin{Vector::Zero()}, axis{Vector::UnitZ()}, radial{Vector::UnitX()};
  std::optional<AirGap> air_gap; // planar only
  std::optional<Periodicity> period; // planar only
  // A planar mesh's unknowns: its nodes' potentials, then a sliding band's
  // interior rings (AirGap::band_first).
  size_t band_unknowns{0};
  [[nodiscard]] size_t dofs() const {
    return planar ? nodes.size() + band_unknowns : edges.size();
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
// Distance from the symmetry axis of x in a MAGAXI cell's meridian plane.
[[nodiscard]] inline double radius(const Cell &c, const Vector &x) {
  return (x - c.axis_origin).dot(c.radial);
}
// The factor that turns an integral over the mesh's plane at x into one over
// the field: DEPTH for MAG2D, 2 pi r for MAGAXI, 1 in 3D.
[[nodiscard]] double out_of_plane(const Mesh &mesh, const Vector &x);
// A MAGAXI exterior face on the symmetry axis, where A_phi = 0.
[[nodiscard]] bool on_axis(const Mesh &mesh, const Face &face);
// make_mesh(model) for a model whose elements and their grids are those of
// the model base was made from, some grids moved (rigidly, as by MOTION) and
// properties possibly changed. The topology (edges, faces, their indices)
// is base's, so only the cells and faces with a moved grid, the
// periodicity and the air gap are recomputed.
[[nodiscard]] Result<Mesh> make_mesh(const Model &model, const Mesh &base);
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
// The curls of a cell's basis functions on its reference cell at r, before
// its edge signs and curl_map: the reference gradients of a planar cell's
// nodal functions, the reference curls of a solid's edge functions. They
// depend only on the element kind and r.
[[nodiscard]] std::vector<Vector> reference_curls(const Cell &cell,
                                                  const Vector &r);
// The matrix taking reference_curls to the cell's curls where its Jacobian is
// j: J / det J for a solid's edge functions; for a planar cell's nodal
// functions N, the curl of N n is grad N x n with grad N = J^-T g.
[[nodiscard]] Matrix curl_map(const Cell &cell, const Matrix &j);
// The geometry of a mesh's cells at their quadrature points, computed once
// and reused by every assembly on the mesh: the quadrature weights, and the
// maps that take each element kind's reference curls to a cell's curls (one
// for an affine cell, one per point otherwise). Positions are mapped on
// demand. It keeps no reference to the mesh, which is passed back in.
class CellGeometry {
public:
  CellGeometry() = default;
  explicit CellGeometry(const Mesh &mesh);
  [[nodiscard]] int points(int cell) const;
  // The reference weight times |det J|, as quadrature() gives it.
  [[nodiscard]] double weight(int cell, int point) const;
  // out_of_plane at the point: weight times measure integrates over the
  // field (DEPTH, 2 pi r for MAGAXI, 1 in 3D).
  [[nodiscard]] double measure(int cell, int point) const;
  // The curls of the cell's basis functions at the point, as edge_curls
  // gives them at its position; out holds one per basis function.
  void curls(const Mesh &mesh, int cell, int point, std::span<Vector> out) const;
  [[nodiscard]] Vector position(const Mesh &mesh, int cell, int point) const;

private:
  struct Rule {
    std::vector<Vector> points;
    std::vector<double> weights;
    std::vector<Vector> curls; // point-major, functions per point
    std::vector<double> values; // shape values, the same layout (planar)
    int functions{0};
  };
  struct Entry {
    int rule{0};
    bool affine{false};
    size_t map{0}, weight{0}; // first in maps_, weights_
  };
  std::vector<Rule> rules_;
  std::vector<Entry> cells_;
  std::vector<Matrix> maps_;
  std::vector<double> weights_, measures_;
  std::vector<double> inverse_radii_; // 1 / r per point of a MAGAXI mesh
};
// The most basis functions of any cell (a hexahedron's twelve edges).
inline constexpr int max_cell_functions = 12;
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
// Unknown i of the field is sign[i] times equation equations[i], or
// prescribed when equations[i] < 0: the image grid of an MPERIOD pair takes
// its partner's equation, with sign -1 across an anti-periodic boundary.
struct Gauge {
  std::vector<int> equations;
  std::vector<bool> boundary;
  std::vector<double> sign;
  int size{0};
  // The reduced vector P^T v of a vector v on the unknowns.
  [[nodiscard]] Eigen::VectorXd reduce(const Eigen::VectorXd &v) const;
  // v += P x for a reduced vector x.
  void expand(const Eigen::VectorXd &x, Eigen::VectorXd &v,
              double scale = 1) const;
};
// natural flags the faces of the planar exterior that take n x H = 0 (MBC
// NORMAL) instead of a prescribed potential.
[[nodiscard]] Result<Gauge> make_gauge(const Mesh &mesh,
                                       const std::vector<bool> &natural = {});
// The MPERIOD sides of model on its field mesh. Errors name the grids and
// edges whose images do not match.
[[nodiscard]] Result<Periodicity> make_periodicity(const Model &model,
                                                   const Mesh &mesh);
} // namespace vibestran::magnetic
