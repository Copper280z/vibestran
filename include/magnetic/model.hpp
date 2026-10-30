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

struct Material {
  double mur{1};
  int bh_table{0};
  Vector remanence{Vector::Zero()};
  int coordinates{0};
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
struct Current {
  int set{0};
  bool wire{false};
  double amperes{0};
  std::vector<int> targets;
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
struct Probe {
  int id{0};
  Vector position{Vector::Zero()};
};
struct ForceSurface {
  int id{0}, faces{0};
  Vector origin{Vector::Zero()};
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
  std::map<int, int> properties;
  std::map<int, Material> materials;
  std::map<int, std::vector<std::pair<double, double>>> bh_tables;
  std::map<int, double> conductivity;
  std::map<int, std::vector<int>> element_sets;
  std::map<int, std::vector<FaceSelection>> face_sets;
  std::vector<Coil> coils;
  std::vector<PlanarCoil> planar_coils;
  std::map<int, WireProperty> wire_properties;
  std::vector<Wire> wires;
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
[[nodiscard]] Response
material_response(const Material &material,
                  const std::vector<std::pair<double, double>> &curve,
                  const Vector &b);
[[nodiscard]] Result<std::map<int, double>> load_factors(const Model &model,
                                                         int set);

} // namespace vibestran::magnetic
