#pragma once
// include/io/results.hpp
// Solution results and output writers (F06, OP2, CSV).

#include "core/model.hpp"
#include "core/types.hpp"
#include <array>
#include <filesystem>
#include <optional>
#include <ostream>
#include <unordered_map>
#include <vector>

namespace vibestran {

// ── Result data ───────────────────────────────────────────────────────────────

/// 6-DOF displacement at a node [T1,T2,T3,R1,R2,R3]
struct NodeDisplacement {
  constexpr NodeDisplacement() = default;
  NodeId node{0};
  std::array<double, 6> d{}; // indexed 0-5
};

/// Element stress at centroid (for plate elements: CQUAD4, CTRIA3)
struct PlateStressPoint {
  constexpr PlateStressPoint() = default;
  NodeId node{0};
  double sx{0}, sy{0}, sxy{0};
  double mx{0}, my{0}, mxy{0};
  double von_mises{0};
};

struct PlateStress {
  constexpr PlateStress() = default;
  ElementId eid{0};
  ElementType etype{ElementType::CQUAD4};
  double sx{0}, sy{0}, sxy{0};  // membrane stresses
  double mx{0}, my{0}, mxy{0};  // moments (plate bending)
  double von_mises{0}; // derived
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

struct LineStressEnd {
  constexpr LineStressEnd() = default;
  NodeId node{0};
  std::array<double, 4> s{};
  double axial{0};
  double smax{0};
  double smin{0};
};

struct LineStress {
  constexpr LineStress() = default;
  ElementId eid{0};
  ElementType etype{ElementType::CBAR};
  LineStressEnd end_a;
  LineStressEnd end_b;
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

/// Constraint reaction force at an SPC'd DOF set (global/basic frame).
struct SpcForce {
  constexpr SpcForce() = default;
  NodeId node{0};
  std::array<double, 6> f{}; // indexed 0-5 (T1,R1...)
};

struct SubCaseResults {
  int id{1};
  std::string label;

  std::vector<NodeDisplacement> displacements;
  std::vector<SpcForce> spc_forces;
  std::vector<LineStress> line_stresses;
  std::vector<PlateStress> plate_stresses;
  std::vector<SolidStress> solid_stresses;

  // Heat-transfer (SOL 153) outputs
  std::vector<NodeTemperature> temperatures;
  std::vector<ElementHeatFlux> heat_fluxes;
};

struct SolverResults {
  std::vector<SubCaseResults> subcases;
};

// ── Modal result data ─────────────────────────────────────────────────────────

struct ModeResult {
  int mode_number{0};
  double eigenvalue{0};       ///< λ = ω² (rad²/s²)
  double radians_per_sec{0};  ///< ω = sqrt(max(λ,0))
  double cycles_per_sec{0};   ///< f = ω/(2π)
  double gen_mass{0};         ///< φᵀ M φ (≈ 1 after mass normalisation)
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
  /// Write linear-static results to an F06 file (respects SubCase output flags)
  static void write(const SolverResults &results, const Model &model,
                    const std::filesystem::path &path);

  /// Write linear-static results to stream (for testing)
  static void write(const SolverResults &results, const Model &model,
                    std::ostream &out);

  /// Write modal results to an F06 file
  static void write_modal(const ModalSolverResults &results, const Model &model,
                          const std::filesystem::path &path);

  /// Write modal results to stream (for testing)
  static void write_modal(const ModalSolverResults &results, const Model &model,
                          std::ostream &out);

  /// Write heat-transfer (SOL 153) results to an F06 file.
  static void write_thermal(const SolverResults &results, const Model &model,
                            const std::filesystem::path &path);
  static void write_thermal(const SolverResults &results, const Model &model,
                            std::ostream &out);

private:
  static void write_header(std::ostream& out);
  static void write_modal_header(std::ostream& out);
  /// " OUTPUT FOR SUBCASE " + %8d line required by the MYSTRAN validation
  /// suite parser before every results block.
  static void write_subcase_header(std::ostream& out, int subcase_id);
  static void write_displacement_table(const SubCaseResults& sc,
                                       std::ostream& out);
  static void write_spc_force_table(const SubCaseResults& sc,
                                    std::ostream& out);
  static void write_shell_stress_table(
      const SubCaseResults& sc, std::ostream& out, ElementType etype,
      bool corners,
      const std::unordered_map<ElementId, const PShell*>& shell_props);
  static void write_solid_stress_table(const SubCaseResults& sc,
                                       std::ostream& out, ElementType etype,
                                       bool corners);
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
