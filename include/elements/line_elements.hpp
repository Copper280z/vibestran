#pragma once
// include/elements/line_elements.hpp
// 1-D structural and scalar element families:
//   - CBAR / CBEAM
//   - CBUSH
//   - CELAS1 / CELAS2
//   - CMASS1 / CMASS2

#include "elements/element_base.hpp"
#include <array>

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

class ScalarSpringElement : public ElementBase {
public:
  ScalarSpringElement(ElementType type, ElementId eid, PropertyId pid,
                      std::vector<NodeId> nodes, std::array<int, 2> components,
                      double value, const Model &model);

  [[nodiscard]] ElementType type() const noexcept override { return type_; }
  [[nodiscard]] ElementId id() const noexcept override { return eid_; }
  [[nodiscard]] int num_dofs() const noexcept override {
    return static_cast<int>(nodes_.size());
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
  [[nodiscard]] double stiffness_value() const;

  ElementType type_;
  ElementId eid_;
  PropertyId pid_;
  std::vector<NodeId> nodes_;
  std::array<int, 2> components_;
  double value_;
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
    return static_cast<int>(nodes_.size());
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
  std::array<int, 2> components_;
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
