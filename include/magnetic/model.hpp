#pragma once

#include "core/result.hpp"
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
// The magnetic API reports failures with the core error values.
using vibestran::Error;
using vibestran::Result;
using vibestran::Status;

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
  // The frozen (secant) reluctivity at this B: H = secant (B - Br). Equal to
  // tangent for a linear material.
  Matrix secant{Matrix::Zero()};
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
// Case-control output requests. BFIELD writes all, no, or one ESET's cells
// to the subcase's VTKHDF block; MPROBE controls the subcase's probe rows.
struct FieldOutput {
  bool enabled{true};
  int set{0};   // 0: every cell
  int every{1}; // BFIELD(EVERY=n): every n-th MOTION position, from the first
};
struct Subcase {
  int id{1}, load{0}, boundary{0};
  int motion{0}; // MOTION ID; 0 for a single static solve
  int drive{0};  // DRIVE ID: phase currents from the MOTION position
  std::string label;
  FieldOutput field;
  bool probes{true};
  bool steps{false}; // MSTEP = ALL: per-position tables for a MOTION subcase
  // LDQ = APPARENT (default) adds the frozen-permeability L_dq to the
  // incremental one in a DRIVE subcase; LDQ = INCREMENTAL omits it.
  bool apparent{true};
};
// MOTION: rigid rotation of the cells of ESET set about the z axis of
// coordinates, through its origin, to the positions start + k * step degrees
// (k = 0 .. count - 1) relative to the mesh as given. Positive angles turn
// the cells counterclockwise looking down the z axis.
struct Motion {
  int id{0}, set{0}, coordinates{0}, count{1};
  double start{0}, step{0}; // degrees
  // SPEED (rpm, positive toward increasing positions): scales the back-EMF
  // dlambda/dphi to volts. 0 when blank.
  double speed{0};
  [[nodiscard]] double position(int k) const { return start + k * step; }
};
// DRIVE: phase currents set from the electrical angle
// theta_e = poles (phi - theta0) at each MOTION position phi. Phase k
// (k = 0 .. m - 1, circuit phases[k]) carries
// i_d cos(theta_k) - i_q sin(theta_k), theta_k = theta_e - 2 pi k / m, with
// i_d = -peak sin(gamma) and i_q = peak cos(gamma) for SINE; SIX is the
// 120-degree six-step waveform with the same fundamental phase.
struct Drive {
  int id{0};
  bool six{false};
  int poles{1};              // pole pairs
  double peak{0}, gamma{0};  // A, electrical degrees from +q towards -d
  std::optional<double> theta0; // degrees; empty for AUTO
  std::vector<int> phases;   // circuit IDs in phase sequence
};
// MBC rows of one SID. The ZERO or UNIFORM row prescribes A on faces (an
// FSET, or 0 for the rest of the exterior): A = B x x / 2 with B = field.
// NORMAL rows (planar only) name FSETs whose faces take the natural
// condition n x H = 0, where the flux crosses the boundary normally.
struct Boundary {
  int faces{0};
  Vector field{Vector::Zero()};
  std::vector<int> normal;
};
// MPERIOD: the field mesh is one of `sectors` identical sectors about the z
// axis of coordinates, through its origin. Side second is side first turned
// by 360 / sectors degrees counterclockwise; across it the potential repeats
// (PERIODIC) or changes sign (ANTI).
struct Periodic {
  int id{0}, first{0}, second{0}, coordinates{0}, sectors{0};
  bool anti{false};
};
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
// PAIRGAP: the elements of an air gap. AGE replaces them by the harmonic
// air-gap element, BAND by a sliding band of elements on a uniform grid;
// MESH solves them as ordinary air (mu_r = 1).
enum class AirGapMethod { Age, Band, Mesh };
struct AirGapProperty {
  AirGapMethod method{AirGapMethod::Age};
  int harmonics{0}; // NHARM, the reported gap field's; 0 for half the finer circle's grid count
  int band_points{0}; // NBAND; 0 for the finer circle's grid count
  int band_layers{0}; // NLAYER; 0 for about square band elements
};
struct Options {
  double length_scale{1}, nonlinear_tolerance{1e-8};
  int max_iterations{80}, wire_sides{16}, wire_layers{2};
  // PARAM,MAGPCG: CG iterations per solve with an earlier tangent's factor
  // before refactoring; 0 factors every tangent.
  int tangent_reuse{0};
  // MAG2D and MAGAXI are both planar: nodal potentials on a mesh in a
  // plane. MAGAXI is axisymmetric about the z axis of plane_coordinates, the
  // mesh in its x-z half-plane (x = r >= 0), with A = A_phi e_phi.
  bool planar{false}, axisymmetric{false};
  int plane_coordinates{0};
  double depth{1}; // MAG2D only
  double wire_step{0}; // zero: automatic from section size
  WireMesh wire_mesh{WireMesh::Auto};
  std::optional<AirGapMethod> air_gap_method; // PARAM,AIRGAP overrides PAIRGAP
  bool source_output{true}; // MSOURCE: coil/wire VTU files
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
  // PAIRGAP properties solved by the air-gap element. A deck's METHOD=MESH
  // gaps (or all of them under PARAM,AIRGAP,MESH) are resolved on parsing
  // into ordinary air properties, and kept here only to be reported.
  std::map<int, AirGapProperty> air_gaps;
  std::map<int, std::string> meshed_air_gaps; // PID -> what selected MESH
  std::map<int, Motion> motions;
  std::map<int, Drive> drives;
  std::optional<Periodic> period;
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
