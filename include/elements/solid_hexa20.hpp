#pragma once
// include/elements/solid_hexa20.hpp
// CHEXA20: 20-node serendipity hexahedron with a projected assumed strain
// (PSOLID ISOP blank, REDUCED or EAS) or the standard displacement element
// (ISOP = FULL), both integrated with 3x3x3 Gauss points.  See
// CHEXA20_PLAN.md for the derivation and the benchmarks that selected it.
//
// The assumed strain replaces the compatible strain B (6x60, Cartesian
// engineering Voigt) by
//
//   Bm     = (1/V) int B dV                          (mean strain)
//   A(F)   = C [ Pi_S1(TF)_11, Pi_S2(TF)_22, Pi_S3(TF)_33,
//                Pi_P8(TF)_12, Pi_P8(TF)_23, Pi_P8(TF)_31 ]
//   Bs     = Bm + A(B - Bm) - (1/V) int A(B - Bm) dV
//   Bbar   = Bs + m/3 (Pi_P8(m'Bs) - m'Bs),   m = (1, 1, 1, 0, 0, 0)
//
// T(xi) maps Cartesian strains to the natural components e_ii = t_i.eps.t_i,
// g_ij = 2 t_i.eps.t_j in the local frame t_i = (dx/dxi_i) / |dx/dxi_i|, and
// C = T^-1.  Pi_S is the det(J)-weighted L2 projection over the element
// onto a polynomial space in (xi_1, xi_2, xi_3):
//   S_i = monomials of degree <= 2 in each coordinate and <= 1 in xi_i (18);
//   P8  = trilinear monomials (8).
//
// * The normal-strain projection removes the part of e_ii that is quadratic
//   along xi_i.  On a curved element that is the parasitic membrane strain
//   of inextensional bending (membrane locking); on an affine element e_ii
//   has no such part, so the projection changes nothing there.
// * The shear projection onto P8 relieves transverse-shear locking, and the
//   B-bar dilatation relieves volumetric locking (equivalent to 2x2x2
//   selective integration of the volumetric energy for isotropic material).
// * Bm and the zero-mean correction make the element reproduce constant
//   strain exactly on any shape and do no work against constant stress, so
//   it passes the patch test with curved, off-centre midside nodes.
// * The frames are covariant and the spaces map onto each other under the 24
//   orientation-preserving node relabellings: no preferred direction.
// * K = int Bbar' D Bbar dV is positive semidefinite for any material.
//
// Bbar is used for the stiffness, the thermal load and stress/strain
// recovery; the mass uses the compatible shape functions.  On affine
// elements the formulation equals MYSTRAN's CHEXA20 (feat/hexa-modern,
// MYSTRAN/dev_docs/chexa_formulation.md), whose shear and dilatation
// projections it shares.
//
// Recovery is at the centre and the eight corners, as in Nastran.  At a
// corner whose scaled Jacobian is not positive the recovery operator is
// extrapolated trilinearly from the 2x2x2 Gauss points.
//
// All 20 grids are required (Nastran allows omitted midside nodes; that is
// not implemented for the structural element).

#include "elements/element_base.hpp"
#include <array>
#include <span>
#include <utility>

namespace vibestran {

/// The first 20 grids of `nodes` as an array (NodeId has no default).
template <class Range> [[nodiscard]] std::array<NodeId, 20> hexa20_node_array(const Range &nodes) {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::array<NodeId, 20>{nodes[I]...};
  }(std::make_index_sequence<20>{});
}

class SolidHexa20 : public ElementBase {
public:
  static constexpr int NUM_NODES = 20;
  static constexpr int NUM_DOFS = 60;

  enum class Formulation { AssumedStrain, Full };

  SolidHexa20(ElementId eid, PropertyId pid, std::array<NodeId, NUM_NODES> node_ids,
              const Model &model);

  [[nodiscard]] ElementType type() const noexcept override { return ElementType::CHEXA20; }
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

  /// Total strain (Bbar u, engineering shear, order xx yy zz xy yz zx) and
  /// stress D (strain - thermal strain) at one recovery point, in basic.
  struct PointResponse {
    Eigen::Matrix<double, 6, 1> strain{Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Matrix<double, 6, 1> stress{Eigen::Matrix<double, 6, 1>::Zero()};
  };

  /// Response at the centre (index 0) and the corners G1..G8 (1..8) for the
  /// element displacements (60 values, T1 T2 T3 per grid).  `temperatures`
  /// holds the 20 grid temperatures (empty: none) and t_ref the material
  /// reference temperature.
  [[nodiscard]] std::array<PointResponse, 9>
  recover(std::span<const double> displacements, std::span<const double> temperatures,
          double t_ref) const;

  /// The strain operator at the 27 Gauss points (Bbar, or B for ISOP = FULL),
  /// for tests.
  [[nodiscard]] std::vector<Eigen::Matrix<double, 6, NUM_DOFS>> gauss_point_strain_operators() const;

private:
  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, NUM_NODES> nodes_;
  const Model &model_;
  Formulation formulation_;
};

} // namespace vibestran
