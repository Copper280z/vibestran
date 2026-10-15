#pragma once
// include/elements/ctria3.hpp
// CTRIA3: 3-node constant-strain triangular shell element.
//
// Formulation:
//   - Membrane: CST (Constant Strain Triangle), plane-stress
//   - Bending:  DKT (Discrete Kirchhoff Triangle): quadratic normal slopes
//     with Kirchhoff constraints at the corners and along the edges, exact
//     3-point integration of the linear curvature field
//   - 6 DOF per node: [u, v, w, θx, θy, θz]
//
// Reference:
//   Batoz, Bathe, Ho, "A study of three-node triangular plate bending
//   elements", IJNME 15 (1980); Batoz, "An explicit formulation for an
//   efficient triangular plate-bending element", IJNME 18 (1982).

#include "elements/element_base.hpp"
#include <array>
#include <span>

namespace vibestran {

class CTria3 : public ElementBase {
public:
    static constexpr int NUM_NODES = 3;
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOFS = NUM_NODES * DOF_PER_NODE; // 18

    CTria3(ElementId eid,
           PropertyId pid,
           std::array<NodeId, NUM_NODES> node_ids,
           const Model& model);

    [[nodiscard]] ElementType type()     const noexcept override { return ElementType::CTRIA3; }
    [[nodiscard]] ElementId   id()       const noexcept override { return eid_; }
    [[nodiscard]] int         num_dofs() const noexcept override { return NUM_DOFS; }

    [[nodiscard]] LocalKe stiffness_matrix() const override;
    [[nodiscard]] LocalKe mass_matrix() const override;
    [[nodiscard]] LocalFe thermal_load(std::span<const double> temperatures,
                                        double t_ref) const override;
    [[nodiscard]] std::vector<EqIndex> global_dof_indices(const DofMap&) const override;
    [[nodiscard]] std::span<const NodeId> node_ids() const noexcept override {
        return std::span<const NodeId>{nodes_.data(), NUM_NODES};
    }

    /// Plate response at area coordinates (xi, eta) of the element (node 1 at
    /// (0,0), node 2 at (1,0), node 3 at (0,1)), in the element frame:
    /// membrane stress (CST, constant) and bending moments per unit length
    /// from the DKT curvature, with the same sign conventions as CQUAD4.
    /// DKT is a Kirchhoff element, so transverse_shear is always zero.
    struct PlateResponse {
        Eigen::Vector3d membrane_stress{Eigen::Vector3d::Zero()};
        Eigen::Vector3d membrane_resultant{Eigen::Vector3d::Zero()};
        Eigen::Vector3d curvature{Eigen::Vector3d::Zero()};
        Eigen::Vector3d bending_moment{Eigen::Vector3d::Zero()};
        Eigen::Vector2d transverse_shear{Eigen::Vector2d::Zero()};
    };
    [[nodiscard]] PlateResponse
    recover_response(std::span<const double> global_displacements, double xi,
                     double eta, double temperature, double t_ref) const;

    /// DKT curvature-displacement matrix (3x9) at (xi, eta) for local nodal
    /// coordinates, acting on [w, w_x, w_y] per node (the CQUAD4 "slope"
    /// rotation convention); curvature = [sx_x, sy_y, sx_y + sy_x].
    [[nodiscard]] static Eigen::Matrix<double, 3, 9>
    dkt_curvature_B(const std::array<double, 3>& xl,
                    const std::array<double, 3>& yl, double xi, double eta);

private:
    ElementId   eid_;
    PropertyId  pid_;
    std::array<NodeId, NUM_NODES>  nodes_;
    const Model& model_;

    std::array<Vec3, 3> node_coords() const;
    Eigen::Matrix3d plane_stress_D() const;
    /// Bending rigidity (12I/t^3) t^3/12 Q(MID2, defaulting to MID1).
    Eigen::Matrix3d bending_D() const;
    double thickness() const;
    const Mat1& material() const;
    const PShell& pshell() const;

    /// Compute membrane stiffness (CST) from local in-plane coordinates xl[3], yl[3]
    Eigen::MatrixXd membrane_stiffness(const std::array<double,3>& xl,
                                        const std::array<double,3>& yl) const;

    /// Compute bending stiffness (DKT) from local in-plane coordinates xl[3], yl[3]
    Eigen::MatrixXd bending_stiffness(const std::array<double,3>& xl,
                                       const std::array<double,3>& yl) const;
};

} // namespace vibestran
