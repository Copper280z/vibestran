#pragma once
// include/io/results.hpp
// Solution results and output writers (F06, OP2, CSV).

#include "io/f06_header.hpp"
#include "core/model.hpp"
#include "core/types.hpp"
#include <array>
#include <filesystem>
#include <optional>
#include <ostream>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vibestran {

// ── Result data ───────────────────────────────────────────────────────────────

/// 6-DOF displacement at a node [T1,T2,T3,R1,R2,R3]
struct NodeDisplacement {
  constexpr NodeDisplacement() = default;
  NodeId node{0};
  std::array<double, 6> d{}; // indexed 0-5, in the grid's CD frame
  CoordId coord{0};          // frame of d: the grid's displacement system
  bool scalar{false};        // a scalar point: d[0] is its one value
};

/// Total shell strains at a point in the element frame (engineering shear).
/// The strain at fibre distance z is membrane - z * curvature, so a positive
/// curvature (and moment) puts the -z (Z1) fibre in tension.
struct ShellStrain {
  constexpr ShellStrain() = default;
  double ex{0}, ey{0}, exy{0}; // mid-surface membrane strains
  double kx{0}, ky{0}, kxy{0}; // curvatures (kxy is the engineering twist)
  double gxz{0}, gyz{0};       // transverse shear strains (section average)
};

/// Element stress at centroid (for plate elements: CQUAD4, CTRIA3)
/// Thickness and output fibre distances (Z1, Z2) of a shell element, from
/// its PSHELL or, for PCOMP, the laminate's mid-plane faces.
struct ShellFibres {
  double t{0.0};
  double z1{0.0};
  double z2{0.0};
};

struct PlateStressPoint {
  constexpr PlateStressPoint() = default;
  NodeId node{0};
  double sx{0}, sy{0}, sxy{0};
  double mx{0}, my{0}, mxy{0};
  double von_mises{0};
  double nx{0}, ny{0}, nxy{0}; // membrane force resultants per unit length
  double qx{0}, qy{0};         // transverse shear resultants per unit length
  ShellStrain strain;
};

struct PlateStress {
  constexpr PlateStress() = default;
  ElementId eid{0};
  ElementType etype{ElementType::CQUAD4};
  double sx{0}, sy{0}, sxy{0};  // membrane stresses
  double mx{0}, my{0}, mxy{0};  // moments per unit length (plate bending)
  double von_mises{0}; // derived
  double nx{0}, ny{0}, nxy{0};  // membrane force resultants per unit length
  double qx{0}, qy{0};          // transverse shear resultants per unit length
  ShellStrain strain;
  std::vector<PlateStressPoint> nodal;
};

/// Element stress for solid elements (at centroid)
struct SolidStressPoint {
  constexpr SolidStressPoint() = default;
  NodeId node{0};
  double sx{0}, sy{0}, sz{0};
  double sxy{0}, syz{0}, szx{0};
  double von_mises{0};
};

struct SolidStress {
  constexpr SolidStress() = default;
  ElementId eid{0};
  ElementType etype{ElementType::CHEXA8};
  double sx{0}, sy{0}, sz{0};
  double sxy{0}, syz{0}, szx{0};
  double von_mises{0};
  std::vector<SolidStressPoint> nodal;
};

/// Element strain for solid elements (at centroid and corners)
struct SolidStrainPoint {
  constexpr SolidStrainPoint() = default;
  NodeId node{0};
  double ex{0}, ey{0}, ez{0};
  double exy{0}, eyz{0}, ezx{0};
  double von_mises{0};
};

struct SolidStrain {
  constexpr SolidStrain() = default;
  ElementId eid{0};
  ElementType etype{ElementType::CHEXA8};
  double ex{0}, ey{0}, ez{0};
  double exy{0}, eyz{0}, ezx{0};
  double von_mises{0};
  std::vector<SolidStrainPoint> nodal;
};

struct LineStressEnd {
  constexpr LineStressEnd() = default;
  NodeId node{0};
  std::array<double, 4> s{};
  double axial{0};
  double smax{0};
  double smin{0};
};

/// CBAR/CBEAM engineering forces in the element frame (Nastran sign
/// convention): plane 1/2 bending moments at ends A and B, plane 1/2 shear,
/// axial force (tension positive), and torque.
struct LineForces {
  double m1a{0}, m2a{0}, m1b{0}, m2b{0};
  double v1{0}, v2{0};
  double axial{0}, torque{0};
};

struct LineStress {
  constexpr LineStress() = default;
  ElementId eid{0};
  ElementType etype{ElementType::CBAR};
  LineStressEnd end_a;
  LineStressEnd end_b;
  LineForces forces;
};

/// CROD and CONROD forces, stresses and elastic strains.  The axial force is tension
/// positive and the torque follows the right-handed twist of end B relative to
/// end A.  Margins of safety are empty without a MAT1 allowable.
struct RodResult {
  constexpr RodResult() = default;
  ElementId eid{0};
  ElementType type{ElementType::CROD};
  double axial_force{0}, torque{0};
  double axial_stress{0}, torsional_stress{0};
  double axial_strain{0}, torsional_strain{0};
  std::optional<double> axial_margin, torsional_margin;
};

/// Nodal temperature (SOL 153 heat-transfer output).
struct NodeTemperature {
  NodeId node{0};
  double temperature{0};
};

/// Element centroidal heat flux  q = -k·∇T  (1, 2, or 3 components).
struct ElementHeatFlux {
  ElementId eid{0};
  ElementType etype{ElementType::CHEXA8};
  /// Spatial dimension: 1 (line), 2 (shell), 3 (solid).
  int dim{3};
  /// q components in element-local frame (size dim).
  std::array<double, 3> q{0.0, 0.0, 0.0};
  /// Magnitude |q|.
  double magnitude{0.0};
};

/// Constraint reaction at an SPC'd grid, in the grid's global (displacement,
/// CD) coordinate system, as Nastran reports SPC forces.
struct SpcForce {
  constexpr SpcForce() = default;
  NodeId node{0};
  std::array<double, 6> f{}; // indexed 0-5 (T1..T3, R1..R3)
  CoordId coord{0};          // frame of f: the grid's displacement system
};

/// One contribution to the force balance at a grid (GPFORCE), in the grid's
/// global (CD) system.  Rows follow MYSTRAN's layout: APPLIED holds every
/// equivalent grid load including the thermal one, which the THERMAL row
/// ("MINUS EQUIV THERM LOAD") then backs out; ELEMENT rows are the force each
/// element exerts on the grid, -(K_e u_e - f_e,thermal).  Rows sum to zero.
struct GridForceContribution {
  enum class Kind { Applied, Thermal, Spc, Mpc, Element };
  Kind kind{Kind::Applied};
  ElementType etype{ElementType::CBAR}; ///< Element rows only
  ElementId eid{0};                     ///< Element rows only
  std::array<double, 6> f{};
};

struct GridPointForceBalance {
  NodeId node{0};
  CoordId coord{0};
  std::vector<GridForceContribution> rows;
  std::array<double, 6> totals{}; ///< Sum of rows; zero up to roundoff
};

struct SubCaseResults {
  int id{1};
  std::string label;

  std::vector<NodeDisplacement> displacements;
  std::vector<SpcForce> spc_forces;
  /// Forces of multipoint constraint (MPC/RBE) at the grids they connect,
  /// in each grid's global (CD) system.
  std::vector<SpcForce> mpc_forces;
  std::vector<GridPointForceBalance> gp_forces;
  /// Nonempty when constraint force recovery failed; spc_forces, mpc_forces
  /// and gp_forces are then empty although they were requested.
  std::string constraint_force_error;
  std::vector<LineStress> line_stresses;
  std::vector<RodResult> rod_results;
  std::vector<PlateStress> plate_stresses;
  std::vector<SolidStress> solid_stresses;
  std::vector<SolidStrain> solid_strains;

  // Heat-transfer (SOL 153) outputs
  std::vector<NodeTemperature> temperatures;
  std::vector<ElementHeatFlux> heat_fluxes;
};

struct SolverResults {
  std::vector<SubCaseResults> subcases;
  SolveSummary summary; ///< F06 header SOLUTION, SOLVER and NOTICES
};

// ── Modal result data ─────────────────────────────────────────────────────────

struct ModeResult {
  int mode_number{0};
  double eigenvalue{0};       ///< λ = ω² (rad²/s²)
  double radians_per_sec{0};  ///< ω = sqrt(max(λ,0))
  double cycles_per_sec{0};   ///< f = ω/(2π)
  double gen_mass{0};         ///< φᵀ M φ of `shape` (1 under NORM=MASS)
  std::vector<NodeDisplacement> shape; ///< mode shape as nodal displacements
};

struct ModalSubCaseResults {
  int id{1};
  std::string label;
  std::vector<ModeResult> modes;
  bool eigvec_print{false};
  bool eigvec_plot{false};
};

struct ModalSolverResults {
  std::vector<ModalSubCaseResults> subcases;
  SolveSummary summary; ///< F06 header SOLUTION, SOLVER and NOTICES
};

// ── Principal stress helpers ──────────────────────────────────────────────────

/// Compute 2-D principal stresses and angle from membrane stress components.
/// angle_deg: rotation angle in degrees from x-axis to major principal axis.
void compute_principal_2d(double sx, double sy, double sxy,
                          double &major, double &minor, double &angle_deg);

/// Compute 3-D principal stresses and direction-cosine matrix for a symmetric
/// stress tensor.  Eigenvalues returned as p1 >= p2 >= p3 (major → minor).
/// v[i][j] = j-th component of i-th eigenvector (Jacobi method).
void compute_principal_3d(double sx, double sy, double sz,
                          double sxy, double syz, double szx,
                          double p[3], double v[3][3]);

// ── F06 writer ────────────────────────────────────────────────────────────────

class F06Writer {
public:
  // Every writer starts with the uniform run header (f06_header.hpp) from
  // results.summary; run adds its RUN section and may be null.

  /// Write linear-static results to an F06 file (respects SubCase output flags)
  static void write(const SolverResults &results, const Model &model,
                    const std::filesystem::path &path,
                    const F06RunInfo *run = nullptr);

  /// Write linear-static results to stream (for testing)
  static void write(const SolverResults &results, const Model &model,
                    std::ostream &out, const F06RunInfo *run = nullptr);

  /// Write modal results to an F06 file
  static void write_modal(const ModalSolverResults &results, const Model &model,
                          const std::filesystem::path &path,
                          const F06RunInfo *run = nullptr);

  /// Write modal results to stream (for testing)
  static void write_modal(const ModalSolverResults &results, const Model &model,
                          std::ostream &out, const F06RunInfo *run = nullptr);

  /// Write heat-transfer (SOL 153) results to an F06 file.
  static void write_thermal(const SolverResults &results, const Model &model,
                            const std::filesystem::path &path,
                            const F06RunInfo *run = nullptr);
  static void write_thermal(const SolverResults &results, const Model &model,
                            std::ostream &out, const F06RunInfo *run = nullptr);

private:
  /// " OUTPUT FOR SUBCASE " + %8d line required by the MYSTRAN validation
  /// suite parser before every results block.
  static void write_subcase_header(std::ostream& out, int subcase_id);
  static void write_displacement_table(const SubCaseResults& sc,
                                       std::ostream& out);
  static void write_constraint_force_table(
      const SubCaseResults& sc, std::ostream& out,
      const std::vector<SpcForce>& forces, std::string_view title,
      std::string_view totals_label);
  static void write_mpc_force_table(const SubCaseResults& sc,
                                    std::ostream& out);
  static void write_gpforce_table(const SubCaseResults& sc,
                                  std::ostream& out);
  static void write_spc_force_table(const SubCaseResults& sc,
                                    std::ostream& out);
  static void write_shell_stress_table(
      const SubCaseResults& sc, std::ostream& out, ElementType etype,
      bool corners,
      const std::unordered_map<ElementId, ShellFibres>& shell_props);
  static void write_shell_strain_table(
      const SubCaseResults& sc, std::ostream& out, ElementType etype,
      bool corners, bool fiber, bool max_shear,
      const std::unordered_map<ElementId, ShellFibres>& shell_props);
  static void write_solid_stress_table(const SubCaseResults& sc,
                                       std::ostream& out, ElementType etype,
                                       bool corners);
  static void write_solid_strain_table(const SubCaseResults& sc,
                                       std::ostream& out, ElementType etype,
                                       bool corners);
  static void write_bar_force_table(const SubCaseResults& sc,
                                    std::ostream& out);
  static void write_shell_force_table(const SubCaseResults& sc,
                                      std::ostream& out, ElementType etype,
                                      bool corners);
  static void write_rod_force_table(const SubCaseResults& sc,
                                    std::ostream& out);
  static void write_rod_stress_table(const SubCaseResults& sc,
                                     std::ostream& out, bool strain);
  static void write_bar_stress_table(const SubCaseResults& sc,
                                     std::ostream& out);
  static void write_quad4_gpstress_table(const SubCaseResults& sc,
                                         std::ostream& out);
  static void write_tria3_gpstress_table(const SubCaseResults& sc,
                                         std::ostream& out);
  static void write_solid_gpstress_table(const SubCaseResults& sc,
                                         std::ostream& out, ElementType etype);
  static void write_eigenvalue_table(const ModalSubCaseResults& msc,
                                     std::ostream& out);
  static void write_eigenvector_table(const ModeResult& mode, int subcase_id,
                                      const std::string& label,
                                      std::ostream& out);
};

// ── OP2 writer ────────────────────────────────────────────────────────────────

class Op2Writer {
public:
  /// Write linear-static results to an OP2 binary file.
  /// Respects SubCase disp_plot / stress_plot flags (PLOT modifier).
  static void write(const SolverResults &results, const Model &model,
                    const std::filesystem::path &path);

  /// Write modal results to an OP2 binary file.
  /// Writes OUGV1 (eigenvectors, one table per mode, if eigvec_plot).
  static void write_modal(const ModalSolverResults &results, const Model &model,
                          const std::filesystem::path &path);
};

// ── CSV writer ────────────────────────────────────────────────────────────────

class CsvWriter {
public:
  /// Write nodal and elemental result CSV files.
  ///
  /// Outputs:
  ///   <stem>.node.csv  – one row per (subcase, node)
  ///   <stem>.elem.csv  – one row per (subcase, element)
  ///
  /// The header line starts with '#'.  Column order is documented in the
  /// header.  Fields not applicable to an element type are written as 0.0.
  static void write(const SolverResults &results, const Model &model,
                    const std::filesystem::path &stem);
};

} // namespace vibestran
