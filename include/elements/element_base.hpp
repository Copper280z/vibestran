#pragma once
// include/elements/element_base.hpp
// Abstract interface for all finite elements.
//
// Each concrete element implements:
//   1. stiffness_matrix()  — local Ke (ndof × ndof)
//   2. thermal_load()      — local fe due to thermal strain
//   3. global_dof_indices() — maps local DOFs to global equation indices
//
// The design is deliberately free of virtual dispatch for performance
// (the concrete types are used directly in the assembler via templates),
// but an abstract base is provided for use in containers / tests.

#include "core/types.hpp"
#include "core/model.hpp"
#include "core/dof_map.hpp"
#include <vector>
#include <span>
#include <Eigen/Dense>

namespace vibestran {

/// Dense local stiffness matrix (heap-allocated, row-major)
using LocalKe = Eigen::MatrixXd;

/// Dense local force vector
using LocalFe = Eigen::VectorXd;

/// Abstract element interface
class ElementBase {
public:
    virtual ~ElementBase() = default;

    /// Element type tag
    [[nodiscard]] virtual ElementType type() const noexcept = 0;

    /// Element ID in the BDF
    [[nodiscard]] virtual ElementId id() const noexcept = 0;

    /// Number of DOFs for this element
    [[nodiscard]] virtual int num_dofs() const noexcept = 0;

    /// Compute local stiffness matrix Ke (num_dofs × num_dofs)
    [[nodiscard]] virtual LocalKe stiffness_matrix() const = 0;

    /// Compute local consistent mass matrix Me (num_dofs × num_dofs)
    [[nodiscard]] virtual LocalKe mass_matrix() const = 0;

    /// Lumped (diagonal, translational) mass matrix, used when the model
    /// selects lumped mass (PARAM,COUPMASS <= 0).  The default applies HRZ
    /// diagonal scaling to the translational block of mass_matrix(); see
    /// hrz_lumped_mass().  Elements with a natural lumped form override it.
    [[nodiscard]] virtual LocalKe lumped_mass_matrix() const;

    /// Compute local thermal load vector fe (num_dofs), given nodal temperatures
    /// and reference temperature.
    /// temperatures[i] = temperature at local node i
    [[nodiscard]] virtual LocalFe thermal_load(std::span<const double> temperatures,
                                                double t_ref) const = 0;

    /// Global equation indices for each local DOF (CONSTRAINED_DOF = -1)
    [[nodiscard]] virtual std::vector<EqIndex> global_dof_indices(
        const DofMap& dof_map) const = 0;

    /// Node IDs in local ordering
    [[nodiscard]] virtual std::span<const NodeId> node_ids() const noexcept = 0;
};

/// HRZ (Hinton-Rock-Zienkiewicz) lumping of a consistent element mass.
/// For each translational direction d the diagonal entries of that direction
/// are scaled so they sum to the direction's total mass 1^T M_dd 1; rotational
/// DOFs and all off-diagonal terms are dropped.  Unlike row-sum lumping it
/// stays positive for quadratic elements (e.g. CTETRA10 corners).  Requires a
/// uniform layout of `dofs_per_node` DOFs per node with translations first;
/// layouts with fewer than 3 DOFs per node are returned unchanged.
[[nodiscard]] LocalKe hrz_lumped_mass(const LocalKe &consistent,
                                      int dofs_per_node);

/// The element mass matrix selected by the model's mass formulation.
[[nodiscard]] LocalKe element_mass_matrix(const ElementBase &element,
                                          const Model &model);

} // namespace vibestran
