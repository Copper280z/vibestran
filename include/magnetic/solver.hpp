#pragma once
#include "io/f06_header.hpp"
#include "magnetic/mesh.hpp"
#include <complex>
#include <limits>
#include <optional>

namespace vibestran {
class SolverBackend;
}

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
// The dq quantities of a DRIVE position: the amplitude-invariant Park
// transform (2/m) C^T of the phase circuits' values (see park_columns), with
// d first. Inductances are (2/m) C^T L C of the phase circuits' L.
struct DriveState {
  double electrical{0}; // theta_e, degrees
  Eigen::Vector2d current{Eigen::Vector2d::Zero()}, linkage{Eigen::Vector2d::Zero()};
  Eigen::Matrix2d incremental{Eigen::Matrix2d::Zero()};
  // LDQ = APPARENT: the inductance of the loaded secant permeability with
  // the magnets off, and the linkage the currents do not make at that
  // permeability (the magnets', and any MBC field's). NaN under
  // LDQ = INCREMENTAL.
  Eigen::Matrix2d apparent{
      Eigen::Matrix2d::Constant(std::numeric_limits<double>::quiet_NaN())};
  Eigen::Vector2d magnet{
      Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN())};
};
// How a DRIVE subcase set its electrical angle.
struct DriveSummary {
  int id{0};
  double theta0{0}; // degrees (a translation's: m)
  // THETA0 AUTO: found by the no-load pre-sweep at alignment_positions,
  // with phase 1's fundamental no-load linkage (Wb) and each phase's lag
  // behind phase 1 (electrical degrees).
  bool automatic{false};
  double linkage{0};
  std::vector<double> lag;
  std::vector<int> phases; // indices into Results::circuits
};
struct EddyStep {
  double time{0}, loss{0}; // seconds, watts
  double electrical_input{0}, winding_loss{0}; // watts
  std::vector<double> voltage; // V, in circuit order
  std::vector<double> property_loss; // ordered as TransientSummary::properties
  std::vector<Vector> current_density; // cell average, A/m², when BFIELD writes
  std::vector<double> loss_density; // cell average, W/m³, when BFIELD writes
};
struct TransientSummary {
  double dt{0}, cycle_residual{0}, power_residual{0};
  double mean_load{0}, peak_to_peak{0}, mean_loss{0}; // N or N·m, W
  double mean_electrical_input{0}, mean_winding_loss{0}; // W
  int steps_per_cycle{0}, cycles{0}, symbolic_analyses{0}, numeric_factorizations{0};
  bool converged{false};
  std::vector<int> properties;
  std::vector<double> mean_property_loss;
};
struct TerminalResult {
  int id{0};
  std::complex<double> voltage, current; // peak V and A, current into conductor
};
struct HarmonicResult {
  double frequency{0}, mean_loss{0}, mean_input{0}, power_residual{0};
  std::vector<int> properties;
  std::vector<double> property_loss, loss_density;
  Eigen::VectorXd potential_imag;
  std::vector<Vector> b_imag, h_imag, current_real, current_imag;
  std::vector<std::complex<double>> winding_current, linkage;
  std::vector<TerminalResult> terminals;
};
// One solved state: a static subcase, or one position of a MOTION subcase.
struct CaseResults {
  // Why this state or sweep can't be trusted, for the fatal message; empty when all is well.
  std::string failure;
  // False when a failed solve left nothing to report: the rest of this case is empty.
  bool has_results{true};
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
  // With an air-gap element: the torque on the part inside the gap about the
  // gap's plane normal u x v (N m); NaN without one, or for a strip or tube.
  double gap_torque{std::numeric_limits<double>::quiet_NaN()};
  // With a strip or tube air-gap element: the force on the part on its inner
  // side along the gap's u (N, the whole stack's); NaN otherwise.
  double gap_force{std::numeric_limits<double>::quiet_NaN()};
  // With an air-gap element: each circuit's dlambda/d(delta) at fixed
  // currents (Wb per radian, the back-EMF per unit speed), for the same
  // rotation as gap_torque, or in a MOTION step about the MOTION axis; for a
  // strip or tube, for the move of gap_force, or in a MOTION step along the
  // direction of travel (Wb per m). It is dT/di (dF/di)
  // by reciprocity, from the inductance solves. NaN for open circuits.
  Eigen::VectorXd linkage_rate;
  // A MOTION subcase keeps one state per position in steps, in order; its
  // own fields then hold only the total Newton iterations, the largest
  // residual and whether any position extrapolated a B-H curve.
  std::vector<CaseResults> steps;
  // In a step: the position (degrees, or m for TRANSLATE) and the torque on
  // the moving part about the MOTION axis (N m), or for TRANSLATE the force
  // on it along the direction of travel (N), toward increasing positions.
  // Only positions that BFIELD writes keep their potential, b and h, with
  // the grid positions and the air gap there.
  double position{0}, torque{std::numeric_limits<double>::quiet_NaN()};
  std::vector<Vector> nodes;
  std::optional<AirGap> air_gap;
  std::optional<DriveState> drive;          // a DRIVE step
  std::optional<DriveSummary> drive_summary; // a DRIVE subcase
  std::optional<EddyStep> eddy;               // a MAGTRANS time step
  std::optional<TransientSummary> transient; // a MAGTRANS subcase
  std::optional<HarmonicResult> harmonic;    // a MAGHARM subcase
};
struct Results {
  Mesh mesh;
  std::vector<Source> sources;
  std::vector<CircuitDefinition> circuits;
  std::vector<CaseResults> cases;
  SolveSummary summary; // F06 header sections
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
// Solves with the CPU direct backend.
[[nodiscard]] Result<Results> solve(const Model &model);
// Solves every tangent system (Newton steps, differential inductances) with
// backend, requiring positive definite factorizations.
[[nodiscard]] Result<Results> solve(const Model &model, SolverBackend &backend);
// False when the build has no HDF5, which MAGSTAT results require.
[[nodiscard]] bool output_available();
// Writes the F06 report and name.mag.vtkhdf: VTKHDF fields that ParaView
// opens directly, plus the /MAGSTAT results group.
// run fills the header's RUN section; null omits it.
[[nodiscard]] Status write_results(const Model &model, const Results &results,
                                   const std::filesystem::path &report,
                                   const F06RunInfo *run = nullptr);
} // namespace vibestran::magnetic
