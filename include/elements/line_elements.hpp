#pragma once
// include/elements/line_elements.hpp
// 1-D structural and scalar element families:
//   - CBAR / CBEAM
//   - CROD, CONROD
//   - CBUSH
//   - CELAS1 / CELAS2
//   - CMASS1 / CMASS2

#include "elements/element_base.hpp"
#include <array>
#include <optional>

namespace vibestran {

class CBarBeamElement : public ElementBase {
public:
  static constexpr int NUM_NODES = 2;
  static constexpr int DOF_PER_NODE = 6;
  static constexpr int NUM_DOFS = NUM_NODES * DOF_PER_NODE;

  /// Builds the element from its parsed card data, including orientation,
  /// pin flags (PA/PB), end offsets (WA/WB), and OFFT.
  CBarBeamElement(const ElementData &data, const Model &model);

  [[nodiscard]] ElementType type() const noexcept override {
    return data_.type;
  }
  [[nodiscard]] ElementId id() const noexcept override { return data_.id; }
  [[nodiscard]] int num_dofs() const noexcept override { return NUM_DOFS; }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  /// Lumped mass: (rho*A + NSM)*L/2 on the translations of each element end
  /// (at the offset points), no rotational inertia, mapped to the grids
  /// through the same offset and pin-flag transforms as the stiffness.
  [[nodiscard]] LocalKe lumped_mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  struct StressRecoveryEnd {
    std::array<double, 4> s{};
    double axial{0.0};
    double smax{0.0};
    double smin{0.0};
  };
  /// Element force resultants in the element frame, Nastran sign
  /// convention: tension-positive axial force; plane-1 (x-y) and plane-2
  /// (x-z) bending moments positive when they produce positive curvature,
  /// i.e. compression at +y / +z.  Shears are taken at end A.
  struct EndForces {
    double m1a{0.0};
    double m2a{0.0};
    double m1b{0.0};
    double m2b{0.0};
    double v1{0.0};
    double v2{0.0};
    double axial{0.0};
    double torque{0.0};
  };
  struct StressRecovery {
    StressRecoveryEnd end_a;
    StressRecoveryEnd end_b;
    EndForces forces;
  };
  /// Recover end forces and stresses at the PBAR C/D/E/F points from the
  /// element's grid displacements (basic frame).  End forces are computed as
  /// K*u minus the fixed-end loads of everything applied within the span, so
  /// they honour pin flags, offsets, shear flexibility, and I12.
  /// `distributed_fixed_end_local` (optional, 12 entries) adds span loads such
  /// as PLOAD1, as returned by compute_pload1_fixed_end_local.
  [[nodiscard]] StressRecovery
  recover_stress(std::span<const double> global_displacements,
                 double average_temperature, double t_ref,
                 std::span<const double> distributed_fixed_end_local = {}) const;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return std::span<const NodeId>{nodes_.data(), NUM_NODES};
  }

private:
  ElementData data_;
  std::array<NodeId, NUM_NODES> nodes_{NodeId{0}, NodeId{0}};
  const Model &model_;
};

/// CROD and CONROD: axial and torsional stiffness only, with linear displacement and twist
/// along the element.  It has no shear or bending, so its grids need something
/// else, or AUTOSPC, to hold the directions it leaves free
/// (analysis_support.hpp).  Matrices are in basic axes; the element axis is the
/// only direction it needs.
class CRodElement : public ElementBase {
public:
  static constexpr int NUM_NODES = 2;
  static constexpr int NUM_DOFS = 12;

  CRodElement(const ElementData &data, const Model &model);

  [[nodiscard]] ElementType type() const noexcept override { return type_; }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override { return NUM_DOFS; }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  /// Consistent mass: (rho*A + NSM)*L/6 * [2 1; 1 2] on all three translations,
  /// so a transverse motion carries the whole mass, and rho*J*L/6 * [2 1; 1 2]
  /// about the axis.  PROD has no polar inertia, so J stands in for it, exactly
  /// for a circular section.
  [[nodiscard]] LocalKe mass_matrix() const override;
  /// Lumped mass: half of (rho*A + NSM)*L on each end's translations, with no
  /// rotational inertia.
  [[nodiscard]] LocalKe lumped_mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return std::span<const NodeId>{nodes_.data(), NUM_NODES};
  }

  /// Tension-positive axial force, and torque positive for a right-handed twist
  /// of end B relative to end A about the axis from A to B, with the stresses
  /// and elastic (thermal-free) strains they produce.  The torsional stress is
  /// C*T/J, and it and its strain are zero when C or J is.  A margin of safety
  /// is allowable / |stress| - 1, with MAT1 ST or SC by the sign of the axial
  /// stress and SS for torsion, and is empty for no stress or no allowable.
  struct Recovery {
    double axial_force{0.0};
    double torque{0.0};
    double axial_stress{0.0};
    double torsional_stress{0.0};
    double axial_strain{0.0};
    double torsional_strain{0.0};
    std::optional<double> axial_margin;
    std::optional<double> torsional_margin;
  };
  [[nodiscard]] Recovery recover(std::span<const double> global_displacements,
                                 double average_temperature) const;

private:
  ElementId eid_;
  ElementType type_;
  std::array<NodeId, NUM_NODES> nodes_{NodeId{0}, NodeId{0}};
  Eigen::Vector3d axis_{Eigen::Vector3d::Zero()};
  double length_{0.0};
  double E_{0.0};
  double G_{0.0};
  double rho_{0.0};
  double alpha_{0.0};
  double ref_temp_{0.0};
  double st_{0.0};
  double sc_{0.0};
  double ss_{0.0};
  double A_{0.0};
  double J_{0.0};
  double C_{0.0};
  double nsm_{0.0};
};

class CBushElement : public ElementBase {
public:
  static constexpr int NUM_NODES = 2;
  static constexpr int DOF_PER_NODE = 6;
  static constexpr int NUM_DOFS = NUM_NODES * DOF_PER_NODE;

  CBushElement(ElementId eid, PropertyId pid,
               std::array<NodeId, NUM_NODES> node_ids, const Model &model,
               std::optional<Vec3> orientation,
               std::optional<NodeId> g0,
               std::optional<CoordId> cid);

  [[nodiscard]] ElementType type() const noexcept override {
    return ElementType::CBUSH;
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
    return std::span<const NodeId>{nodes_.data(), NUM_NODES};
  }

private:
  ElementId eid_;
  PropertyId pid_;
  std::array<NodeId, NUM_NODES> nodes_;
  const Model &model_;
  std::optional<Vec3> orientation_;
  std::optional<NodeId> g0_;
  std::optional<CoordId> cid_;
};

/// One basic component of a CELAS/CMASS connection, 0-based.
struct ScalarConnectionDof {
  NodeId node{0};
  int component{0};
  double weight{0.0};
};

/// The relative motion a scalar element acts on, u_1 - u_2 in the components its
/// card names, as a sum of weight * u over basic components. Those components
/// are in each grid's displacement frame, so on a grid with CD != 0 one of them
/// reaches up to three basic components. Otherwise each end is the one component.
[[nodiscard]] std::vector<ScalarConnectionDof>
scalar_connection_dofs(std::span<const NodeId> nodes,
                       const std::array<int, 2> &components, const Model &model);

class ScalarSpringElement : public ElementBase {
public:
  ScalarSpringElement(ElementType type, ElementId eid, PropertyId pid,
                      std::vector<NodeId> nodes, std::array<int, 2> components,
                      double value, const Model &model,
                      double stress_coefficient = 0.0,
                      bool first_end_grounded = false);

  [[nodiscard]] ElementType type() const noexcept override { return type_; }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override {
    return static_cast<int>(dofs_.size());
  }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return nodes_;
  }

  /// The spring force k (u1 - u2), with a grounded end at zero, from displacements
  /// ordered as global_dof_indices().
  [[nodiscard]] double
  force(std::span<const double> global_displacements) const;
  /// S, which gives the stress S * force: from PELAS for a CELAS1.
  [[nodiscard]] double stress_coefficient() const;

private:
  [[nodiscard]] double stiffness_value() const;

  ElementType type_;
  ElementId eid_;
  PropertyId pid_;
  std::vector<NodeId> nodes_;
  std::vector<ScalarConnectionDof> dofs_;
  double value_;
  double stress_coefficient_;
  bool first_end_grounded_;
  const Model &model_;
};

/// CONM2 concentrated mass with CG offset and inertia tensor.  It adds no
/// stiffness; its 6x6 grid mass is the rigid transfer of the mass and inertia
/// from the centre of gravity to the grid (basic frame).  A point mass is
/// already lumped, so lumped_mass_matrix() equals mass_matrix().
class Conm2Element : public ElementBase {
public:
  Conm2Element(const ElementData &data, const Model &model);

  [[nodiscard]] ElementType type() const noexcept override {
    return ElementType::CONM2;
  }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override { return 6; }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  [[nodiscard]] LocalKe lumped_mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return std::span<const NodeId>{&node_, 1};
  }

private:
  ElementId eid_;
  NodeId node_;
  ElementData::ConcentratedMass data_;
  const Model &model_;
};

class ScalarMassElement : public ElementBase {
public:
  ScalarMassElement(ElementType type, ElementId eid, PropertyId pid,
                    std::vector<NodeId> nodes, std::array<int, 2> components,
                    double value, const Model &model);

  [[nodiscard]] ElementType type() const noexcept override { return type_; }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override {
    return static_cast<int>(dofs_.size());
  }

  [[nodiscard]] LocalKe stiffness_matrix() const override;
  [[nodiscard]] LocalKe mass_matrix() const override;
  [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                     double t_ref) const override;
  [[nodiscard]] std::vector<EqIndex>
  global_dof_indices(const DofMap &dof_map) const override;
  [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
    return nodes_;
  }

private:
  [[nodiscard]] double mass_value() const;

  ElementType type_;
  ElementId eid_;
  PropertyId pid_;
  std::vector<NodeId> nodes_;
  std::vector<ScalarConnectionDof> dofs_;
  double value_;
  const Model &model_;
};

/// Fixed-end loads of one PLOAD1 in the element frame at the (offset)
/// element ends, before pin-flag condensation.  Used for stress recovery.
[[nodiscard]] LocalFe compute_pload1_fixed_end_local(const ElementData &elem,
                                                     const Model &model,
                                                     const Pload1Load &load);

/// Equivalent grid loads (basic frame) of one PLOAD1, including offsets and
/// pin-flag condensation.
[[nodiscard]] LocalFe compute_pload1_equivalent_load(const ElementData &elem,
                                                     const Model &model,
                                                     const Pload1Load &load);

} // namespace vibestran
