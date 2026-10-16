#pragma once
// include/elements/shell_quad4.hpp
// CQUAD4: MITC4+/D continuum-mechanics-based 4-node shell with six DOFs per
// node.
//
// References:
//   Ko Y, Bathe KJ, Zhang X. Continuum mechanics-based shell elements with
//     six degrees of freedom at each node - the MITC4/D and MITC4+/D
//     elements. Comput Struct 2025;308:107622.             [KBZ]
//   Ko Y, Lee PS, Bathe KJ. A new MITC4+ shell element. Comput Struct
//     2017;182:404-418.                                     [KLB]
//   Dvorkin EN, Bathe KJ. A continuum mechanics based four-node shell
//     element for general nonlinear analysis. Eng Comput 1984;1:77-88.
//
// Geometry and kinematics (KBZ Eqs. 1-4): the element is not flat; each
// grid carries a director V_i (unit normal of the element surface at that
// corner) and a thickness a_i.  Rotations are the grid's global rotation
// vectors, so u = u_m + t/2 sum a_i h_i (theta_i x V_i).
//
// Strains:
//  - membrane: the MITC4+/D assumed field of KBZ Eqs. 24-25 (tying at A-E
//    and the 1/sqrt(3) points), plus the drill-membrane strain from Allman
//    type normal edge displacements driven by the drill rotations
//    theta . V^D about the centre-plane normal V^D (KBZ Eqs. 10-21).  The
//    edge displacement sign follows KBZ Eqs. 12 and 16, which agree with
//    rigid-rotation compatibility; KBZ Eq. 19 / A.5 print the opposite
//    pairing of the edge nodes.
//  - bending: displacement-based (KBZ Eq. 7c), first order in t.
//  - transverse shear: the MITC4 assumed covariant field (Eq. 20).
// The covariant strains are mapped to a local Cartesian frame and used with
// the resultant section of elements/shell_section (PSHELL, PCOMP), which is
// what lets PSHELL's independent MID1..MID4 and PCOMP's laminate stiffness
// apply.  The t^2 bending term and the through-thickness variation of the
// metric are neglected, i.e. curvature is the first-order strain gradient.
//
// Output frame: Nastran's CQUAD4 element frame (z = d13 x d24, x bisecting
// the diagonals); for a flat element it coincides with the local frame.

#include "elements/element_base.hpp"
#include "elements/shell_section.hpp"
#include <array>
#include <span>

namespace vibestran {

/// Shell response at one point, in the element output frame.
/// Strains are total (thermal expansion included); N, M, Q are resultants
/// per unit length with Nastran signs (see shell_section.hpp).
struct ShellPointResponse {
  Eigen::Vector3d membrane_strain{Eigen::Vector3d::Zero()};
  Eigen::Vector3d curvature{Eigen::Vector3d::Zero()};
  Eigen::Vector2d transverse_shear_strain{Eigen::Vector2d::Zero()};
  Eigen::Vector3d membrane_resultant{Eigen::Vector3d::Zero()};
  Eigen::Vector3d bending_moment{Eigen::Vector3d::Zero()};
  Eigen::Vector2d transverse_shear{Eigen::Vector2d::Zero()};
  double thickness{0.0};
};

class ShellQuad4 : public ElementBase {
public:
  static constexpr int NUM_NODES = 4;
  static constexpr int NUM_DOFS = 24;

  ShellQuad4(const ElementData &data, const Model &model);

  [[nodiscard]] ElementType type() const noexcept override {
    return ElementType::CQUAD4;
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

  /// Response at the centre (index 0) and the corners G1..G4 (1..4).  Corner
  /// values are the bilinear extrapolation of the 2x2 Gauss-point values
  /// (Nastran's BILIN).  `temperatures` are grid temperatures.
  [[nodiscard]] std::array<ShellPointResponse, 5>
  recover(std::span<const double> global_displacements,
          std::span<const double> temperatures, double t_ref) const;

  /// Output frame axes (Nastran element frame) in basic coordinates.
  [[nodiscard]] std::array<Vec3, 3> element_frame() const;

  /// Drill-rotation stabilisation stiffness as a fraction of the in-plane
  /// shear stiffness (see shell_quad4.cpp).  Small enough not to interfere
  /// with the drill-membrane strain (1e-3 stiffened in-plane bending by 3%
  /// and the thin twisted beam by 15%; 1e-5 and below do not).
  static constexpr double kDrillStabilisation = 1.0e-6;

private:
  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, NUM_NODES> nodes_;
  const Model &model_;
  ElementData data_;
};

} // namespace vibestran
