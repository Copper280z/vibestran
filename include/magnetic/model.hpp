#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace vibestran::magnetic {

using Vector = Eigen::Vector3d;
using Matrix = Eigen::Matrix3d;
inline constexpr double mu0 = 1.2566370614359172954e-6;
struct Error {
  std::string message;
};
template <class T> using Result = std::variant<T, Error>;
using Status = std::optional<Error>;

// MATMAG: constitutive data only. A table and MUR are mutually exclusive;
// br is the remanence magnitude, directed by the referencing Property.
struct Material {
  double mur{1};
  int bh_table{0};
  double br{0};
};
// PSOLID, PMAG or PMAG2D. Remanence points along sign(axis) * e_|axis| of the
// coordinate system at each point. oriented records that the deck stated a
// frame (coordinates 0 is basic); a magnetized material requires it.
struct Property {
  int material{0}, coordinates{0}, axis{1};
  bool oriented{false};
};
struct Response {
  Vector h{Vector::Zero()};
  Matrix tangent{Matrix::Zero()};
  double energy{0};
  bool extrapolated{false};
};
struct CoordinateSystem {
  int parent{0};
  char type{'R'};
  Vector a{Vector::Zero()}, b{Vector::UnitZ()}, c{Vector::UnitX()};
};
struct Grid {
  Vector position{Vector::Zero()};
  int coordinates{0};
};
struct NodeList : std::vector<int> {
  using std::vector<int>::vector;
  template <size_t N>
  NodeList(const std::array<int, N> &a)
      : std::vector<int>(a.begin(), a.end()) {}
};
struct Element {
  int id{0}, property{0};
  NodeList nodes = NodeList(4);
};
struct FaceSelection {
  int element{0};
  NodeList nodes = NodeList(3);
};
struct Coil {
  int id{0}, elements{0};
  double turns{1};
  int start{0}, end{0};
  // COIL CUT: a closed coil's cut is derived from the half-plane z = 0, x > 0
  // of a rectangular system, with positive circulation along +z.
  bool plane_cut{false};
  int cut_coordinates{0};
};
struct PlanarCoil {
  int id{0}, elements{0};
  double turns{1}; // signed turns: positive current along plane normal
};
struct WireProperty {
  int material{0};
  bool rectangular{false};
  double width{0}, height{0}; // diameter for ROUND; width/height for RECT
};
struct Wire {
  int id{0}, property{0}, first{0}, second{0}, orientation_node{0};
  std::optional<Vector> orientation;
};
// Names the CWIRE path containing element; positive current flows from the
// element's G1 to G2.
struct WirePath {
  int id{0}, element{0};
};
// Series connection of sources (COIL, COIL2D or WIREPATH IDs) with polarity
// +/-1, representing `parallel` identical branches. Circuit and source IDs
// share one namespace; an unlisted source is its own implicit circuit.
struct Circuit {
  int id{0}, parallel{1};
  std::vector<std::pair<int, int>> members;
};
// ICIRC: terminal current of a circuit, or of a source's implicit circuit.
struct Current {
  int set{0}, target{0};
  double amperes{0};
};
struct Combination {
  double scale{1};
  std::vector<std::pair<double, int>> terms;
};
struct Subcase {
  int id{1}, load{0}, boundary{0};
  std::string label;
};
struct Boundary {
  int faces{0};
  Vector field{Vector::Zero()};
}; // A = B x x / 2
// MPROBE: position in basic once parsed (CP is resolved like GRID CP); the
// field is also reported in the local axes of output_coordinates (CD).
struct Probe {
  int id{0};
  Vector position{Vector::Zero()};
  int coordinates{0}, output_coordinates{0};
};
// MFORCE: FACES integrates an explicit FSET; BODY derives the surface around
// an ESET through `layers` air layers. Torque is about the origin of
// coordinates and is also reported in its axes.
struct ForceSurface {
  int id{0}, set{0};
  bool body{false};
  int coordinates{0}, layers{1};
};
enum class WireMesh { Auto, Hex, Tet };
struct Options {
  double length_scale{1}, nonlinear_tolerance{1e-8};
  int max_iterations{80}, wire_sides{16}, wire_layers{2};
  bool planar{false};
  int plane_coordinates{0};
  double depth{1};
  double wire_step{0}; // zero: automatic from section size
  WireMesh wire_mesh{WireMesh::Auto};
};
struct Model {
  std::map<int, Grid> nodes;
  std::map<int, CoordinateSystem> coordinates;
  std::vector<Element> elements;
  std::map<int, Property> properties;
  std::map<int, Material> materials;
  std::map<int, std::vector<std::pair<double, double>>> bh_tables;
  std::map<int, double> conductivity;
  std::map<int, std::vector<int>> element_sets;
  std::map<int, std::vector<FaceSelection>> face_sets;
  std::vector<Coil> coils;
  std::vector<PlanarCoil> planar_coils;
  std::map<int, WireProperty> wire_properties;
  std::vector<Wire> wires;
  std::vector<WirePath> wire_paths;
  std::map<int, Circuit> circuits;
  std::vector<Current> currents;
  std::map<int, Combination> combinations;
  std::map<int, Boundary> boundaries;
  std::vector<Subcase> subcases;
  std::vector<Probe> probes;
  std::vector<ForceSurface> forces;
  Options options;
};

[[nodiscard]] Result<Model> parse_file(const std::filesystem::path &path);
// In-memory deck entry point, currently used only by tests.
[[nodiscard]] Result<Model> parse_string(const std::string &text);
[[nodiscard]] Result<bool> is_magnetic_deck(const std::filesystem::path &path);
[[nodiscard]] Status validate(const Model &model);
// Columns are the local x, y, z axes, in basic, of a resolved (parent 0)
// coordinate system.
[[nodiscard]] Matrix coordinate_axes(const CoordinateSystem &cs);
// Columns are the unit vectors of the local coordinate directions at basic
// point x: (x, y, z), (r, theta, z) or (r, theta, phi) for CORD2R/C/S.
[[nodiscard]] Matrix local_axes(const CoordinateSystem &cs, const Vector &x);
// Remanence enters as B_r in basic at the evaluation point.
[[nodiscard]] Response
material_response(const Material &material,
                  const std::vector<std::pair<double, double>> &curve,
                  const Vector &b, const Vector &remanence = Vector::Zero());
[[nodiscard]] Result<std::map<int, double>> load_factors(const Model &model,
                                                         int set);

} // namespace vibestran::magnetic
