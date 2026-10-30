#pragma once
#include "magnetic/mesh.hpp"

namespace vibestran::magnetic {
struct Source {
  int id{0};
  bool wire{false}, closed{true};
  double turns{1}; // signed for a planar region: positive along the normal
  std::vector<std::pair<int, int>>
      wire_elements; // element id, direction along path
  Mesh mesh;
  Eigen::VectorXd flux;
  std::vector<Vector>
      planar_density;       // axial current density per circuit ampere
  Eigen::VectorXd coupling; // integral N_edge . J_per_ampere dV
  double conservation_error{0}, transfer_error{0};
};
struct SourceResults {
  int id{0};
  bool wire{false}, closed{true};
  double current{0}, linkage{0};
};
// A resolved circuit: explicit CIRCUIT cards and the implicit circuit of every
// unlisted source, in ID order. Members index Results::sources.
struct CircuitDefinition {
  int id{0}, parallel{1};
  std::vector<std::pair<size_t, int>> members; // source index, polarity
  bool closed{true}; // linkage and inductance are defined
};
struct CircuitResults {
  int id{0};
  double current{0}, linkage{0};
};
struct ForceComponent {
  Vector force{Vector::Zero()}, torque{Vector::Zero()};
};
struct ForceResult {
  int id{0};
  Vector force{Vector::Zero()}, torque{Vector::Zero()}; // basic axes
  // BODY requests: one entry per connected piece of the enclosed region.
  std::vector<ForceComponent> components;
};
struct CaseResults {
  int id{0}, iterations{0};
  double residual{0}, energy{0}, coenergy{0};
  bool extrapolated{false};
  Eigen::VectorXd potential;
  std::vector<Vector> b, h;
  std::vector<SourceResults> sources;
  Eigen::MatrixXd differential_inductance;
  std::vector<CircuitResults> circuits;
  Eigen::MatrixXd circuit_inductance; // circuit order of Results::circuits
  std::vector<Vector> probe_b, probe_h;
  std::vector<ForceResult> forces;
};
struct Results {
  Mesh mesh;
  std::vector<Source> sources;
  std::vector<CircuitDefinition> circuits;
  std::vector<CaseResults> cases;
};
[[nodiscard]] Result<std::vector<Source>> prepare_sources(const Model &model,
                                                          const Mesh &mesh);
[[nodiscard]] Result<std::vector<Source>> prepare_wires(const Model &model);
[[nodiscard]] Result<Eigen::VectorXd>
conserved_flux(const Mesh &mesh, std::span<const std::pair<int, int>> section,
               std::span<const int> terminal_faces, bool stranded,
               double section_area, std::span<const double> resistivity);
// The transfer functions take a SpatialIndex built from mesh so one index
// serves every source and quadrature order.
[[nodiscard]] Result<Eigen::VectorXd>
couple_source(const Source &source, const Mesh &mesh, const SpatialIndex &index);
// Clip-based transfer for convex planar-faced source/background cells.
[[nodiscard]] Result<Eigen::VectorXd>
couple_polyhedral_source(const Source &source, const Mesh &mesh,
                         const SpatialIndex &index);
// Transfer a polyhedral wire source into mapped field cells. forbidden lists
// field-cell indices occupied by magnetic material or volumetric coils.
[[nodiscard]] Result<Eigen::VectorXd>
couple_curved_source(const Source &source, const Mesh &mesh,
                     const SpatialIndex &index,
                     std::span<const int> forbidden = {});
[[nodiscard]] Result<std::vector<CircuitDefinition>>
resolve_circuits(const Model &model, std::span<const Source> sources);
// Terminal current of each circuit for a LOAD set.
[[nodiscard]] Result<std::vector<double>>
circuit_currents(const Model &model,
                 std::span<const CircuitDefinition> circuits, int set);
// Current per turn in each source: polarity * I_circuit / parallel.
[[nodiscard]] Result<std::vector<double>>
source_currents(const Model &model, std::span<const Source> sources, int set);
[[nodiscard]] Result<Results> solve(const Model &model);
[[nodiscard]] Status write_results(const Model &model, const Results &results,
                                   const std::filesystem::path &report);
} // namespace vibestran::magnetic
