#pragma once
// include/elements/solid_hexa8.hpp
// CHEXA8: 8-node hexahedron with the Q1/E21 enhanced assumed strain
// formulation (PSOLID ISOP blank, REDUCED or EAS) or the standard
// displacement element (ISOP = FULL).
//
// References:
//   Simo JC, Rifai MS. A class of mixed assumed strain methods and the
//     method of incompatible modes. IJNME 1990;29:1595-1638.
//   Andelfinger U, Ramm E. EAS-elements for two-dimensional,
//     three-dimensional, plate and shell structures and their equivalence
//     to HR-elements. IJNME 1993;36:1311-1337.
//
// The strain is eps = B u + M alpha with 21 internal strain parameters
// alpha, condensed at the element level:
//
//   M(xi) = det(J0) / det(J(xi)) * T0 * Mnat(xi)
//   G = int M' D B dV,   H = int M' D M dV,   K = int B' D B dV - G' H^-1 G
//
// T0 maps natural-frame (xi, eta, zeta) engineering strains to Cartesian
// ones with the centroid Jacobian J0.  The natural-frame modes are
//
//   e_xixi     : xi,   xi eta,   xi zeta
//   e_etaeta   : eta,  eta xi,   eta zeta
//   e_zetazeta : zeta, zeta xi,  zeta eta
//   g_xieta    : xi,   eta,      xi zeta,  eta zeta
//   g_etazeta  : eta,  zeta,     eta xi,   zeta xi
//   g_zetaxi   : zeta, xi,       zeta eta, xi eta
//
// Every mode integrates to zero over the parent cube, so with the
// det(J0)/det(J) scaling the enhanced strains are orthogonal to constant
// stress on any shape (patch test).  Each normal mode contains its own
// coordinate, so no compatible strain can be cancelled and the condensed
// stiffness has only the six rigid-body zero modes.  The set is symmetric
// in xi, eta and zeta and the push-forward is covariant, so the element is
// invariant to rotation and to orientation-preserving node relabelling.
// Modes 1-9 are the Wilson incompatible modes (bending); the bilinear
// normal-strain modes relieve volumetric locking.
//
// The same condensed operator is used for the stiffness, the thermal load
// and stress/strain recovery: with temperatures the internal modes are
// alpha = H^-1 (q - G u), q = int M' D eps_th dV, so the thermal load is
// int B' D eps_th dV - G' H^-1 q and the recovered strain is
// (B - M H^-1 G) u + M H^-1 q.  H is solved through a Cholesky factor of
// the diagonally equilibrated S H S, S = diag(H)^-1/2.
//
// Recovery at the centre and the corners; at a corner whose scaled
// Jacobian is not positive (a concave or collapsed corner) the recovery
// operator is extrapolated trilinearly from the 2x2x2 Gauss points, which
// is exact for constant strain and for parallelepiped elements.
//
// This follows MYSTRAN's CHEXA8 (MYSTRAN/dev_docs/chexa_formulation.md),
// which was verified against ComLab and Simcenter Nastran.

#include "elements/element_base.hpp"
#include <array>
#include <span>

namespace vibestran {

class SolidHexa8 : public ElementBase {
public:
  static constexpr int NUM_NODES = 8;
  static constexpr int NUM_DOFS = 24;
  static constexpr int NUM_MODES = 21;

  enum class Formulation { EnhancedStrain, Full };

  SolidHexa8(ElementId eid, PropertyId pid, std::array<NodeId, NUM_NODES> node_ids,
             const Model &model);

  [[nodiscard]] ElementType type() const noexcept override { return ElementType::CHEXA8; }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override { return NUM_DOFS; }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex> global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return {nodes_.data(), NUM_NODES};
  }

  [[nodiscard]] Formulation formulation() const noexcept { return formulation_; }

  /// Total strain (B u including the thermal expansion, engineering shear,
  /// order xx yy zz xy yz zx) and stress at one recovery point, in basic.
  struct PointResponse {
    Eigen::Matrix<double, 6, 1> strain{Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Matrix<double, 6, 1> stress{Eigen::Matrix<double, 6, 1>::Zero()};
  };

  /// Response at the centre (index 0) and the corners G1..G8 (1..8) for the
  /// element displacements (24 values, T1 T2 T3 per grid).  `temperatures`
  /// holds the eight grid temperatures (empty: none) and t_ref the material
  /// reference temperature.
  [[nodiscard]] std::array<PointResponse, 9>
  recover(std::span<const double> displacements, std::span<const double> temperatures,
          double t_ref) const;

private:
  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, NUM_NODES> nodes_;
  const Model &model_;
  Formulation formulation_;
  int order_;
};

} // namespace vibestran
