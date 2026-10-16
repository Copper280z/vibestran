#pragma once
// include/elements/shell_tria3.hpp
// CTRIA3: MITC3+ continuum-mechanics-based 3-node shell with six DOFs per
// node.
//
// References:
//   Lee Y, Lee PS, Bathe KJ. The MITC3+ shell element and its performance.
//     Comput Struct 2014;138:12-23.                         [LLB]
//   Hughes TJR, Brezzi F. On drilling degrees of freedom. Comput Methods
//     Appl Mech Eng 1989;72:105-121.
//
// Geometry is flat; grids carry shared directors (shell_directors.hpp) and
// thicknesses.  Bending and transverse shear use the MITC3+ kinematics: the
// rotations are enriched by a cubic bubble at the barycentre (two rotation
// DOFs, statically condensed; LLB Eqs. 6-8), and the transverse shear
// strain is the MITC3+ assumed field tied at the six interior points of LLB
// Table 1 (Eqs. 13-17, d = 1/10000).  The membrane is the constant-strain
// triangle, as in MITC3+.  The section, recovery conventions and thermal
// treatment are those of the CQUAD4 (shell_quad4.hpp).
//
// Drill rotations: MITC3+ is a 5-DOF element.  An Allman-type drill
// membrane strain (as in the CQUAD4) was tried and rejected: measured as
// theta . n on a flat triangle with shared, tilted directors it picks up
// bending rotations and locked thin curved shells (thin twisted beam 22%,
// hemisphere 13% too stiff), and measured about the grid directors it
// breaks rigid-body motion on curved meshes.  Each corner's drill rotation
// is instead tied to the in-plane rotation of the translations by a small
// penalty (Hughes & Brezzi 1989), much like Nastran's K6ROT spring.

// Output frame: Nastran's CTRIA3 element frame (x along G1-G2, z normal).

#include "elements/element_base.hpp"
#include "elements/shell_quad4.hpp" // ShellPointResponse
#include <array>
#include <span>

namespace vibestran {

class ShellTria3 : public ElementBase {
public:
  static constexpr int NUM_NODES = 3;
  static constexpr int NUM_DOFS = 18;

  ShellTria3(const ElementData &data, const Model &model);

  [[nodiscard]] ElementType type() const noexcept override {
    return ElementType::CTRIA3;
  }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override { return NUM_DOFS; }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), NUM_NODES};
  }

  /// Response at the centroid (index 0) and the corners G1..G3 (1..3).
  [[nodiscard]] std::array<ShellPointResponse, 4>
  recover(std::span<const double> global_displacements,
          std::span<const double> temperatures, double t_ref) const;

  /// Drill stabilisation as a fraction of the in-plane shear stiffness.
  static constexpr double kDrillStabilisation = 1.0e-6;

private:
  ElementId eid_;
  std::array<NodeId, NUM_NODES> nodes_;
  const Model &model_;
  ElementData data_;
};

} // namespace vibestran
