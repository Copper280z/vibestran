#pragma once
// include/elements/solid_elements.hpp
// CHEXA8: 8-node isoparametric hexahedral solid element
// CTETRA4: 4-node linear tetrahedral element
//
// Both use 3 DOF per node (T1, T2, T3 translations only).
//
// CHEXA8 formulation:
//   - Trilinear isoparametric, 2×2×2 Gauss quadrature
//   - Selectively reduced integration for volumetric locking prevention
//
// CTETRA4 formulation:
//   - Constant strain tetrahedron, closed-form integration

#include "elements/element_base.hpp"
#include <array>

namespace nastran {

// ── CHEXA8 ───────────────────────────────────────────────────────────────────

class CHexa8 : public ElementBase {
public:
    static constexpr int NUM_NODES    = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOFS     = NUM_NODES * DOF_PER_NODE; // 24

    CHexa8(ElementId eid,
           PropertyId pid,
           std::array<NodeId, NUM_NODES> node_ids,
           const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CHEXA8; }
    [[nodiscard]] ElementId   id()       const noexcept override { return eid_; }
    [[nodiscard]] int         num_dofs() const noexcept override { return NUM_DOFS; }

    [[nodiscard]] LocalKe stiffness_matrix() const override;
    [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                        double t_ref) const override;
    [[nodiscard]] std::vector<EqIndex> global_dof_indices(const DofMap&) const override;
    [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
        return std::span<const NodeId>{nodes_.data(), NUM_NODES};
    }

    // Exposed for stress recovery in linear_static.cpp
    struct ShapeData {
        std::array<double, 8> N;
        std::array<double, 8> dNdxi;
        std::array<double, 8> dNdeta;
        std::array<double, 8> dNdzeta;
    };
    static ShapeData shape_functions(double xi, double eta, double zeta) noexcept;

private:
    ElementId   eid_;
    PropertyId  pid_;
    std::array<NodeId, NUM_NODES> nodes_;
    const Model& model_;

    std::array<Vec3, 8> node_coords() const;
    Eigen::Matrix<double,6,6> constitutive_D() const;
    const Mat1& material() const;
    const PSolid& psolid() const;
};

// ── CTETRA4 ──────────────────────────────────────────────────────────────────

class CTetra4 : public ElementBase {
public:
    static constexpr int NUM_NODES    = 4;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOFS     = NUM_NODES * DOF_PER_NODE; // 12

    CTetra4(ElementId eid,
            PropertyId pid,
            std::array<NodeId, NUM_NODES> node_ids,
            const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CTETRA4; }
    [[nodiscard]] ElementId   id()       const noexcept override { return eid_; }
    [[nodiscard]] int         num_dofs() const noexcept override { return NUM_DOFS; }

    [[nodiscard]] LocalKe stiffness_matrix() const override;
    [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                        double t_ref) const override;
    [[nodiscard]] std::vector<EqIndex> global_dof_indices(const DofMap&) const override;
    [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
        return std::span<const NodeId>{nodes_.data(), NUM_NODES};
    }

private:
    ElementId   eid_;
    PropertyId  pid_;
    std::array<NodeId, NUM_NODES> nodes_;
    const Model& model_;

    std::array<Vec3, 4> node_coords() const;
    Eigen::Matrix<double,6,6> constitutive_D() const;
    const Mat1& material() const;
    const PSolid& psolid() const;
};

} // namespace nastran
