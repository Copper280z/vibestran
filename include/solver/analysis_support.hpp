#pragma once

#include "core/dof_map.hpp"
#include "core/mpc_handler.hpp"
#include "core/model.hpp"

#include <array>
#include <span>
#include <vector>

namespace vibestran {

/// Whether AUTOSPC is on: it is unless PARAM,AUTOSPC,NO turns it off.
[[nodiscard]] bool autospc_enabled(const Model &model);

/// The single-point constraints of a structural subcase: Model::structural_spcs
/// plus the AUTOSPC of grid point singularities.
///
/// For the translation and rotation block of each grid, the component nearest the
/// weakest principal direction, in the grid's CD system, is constrained until the
/// remaining components' smallest principal stiffness is at least PARAM,EPZERO
/// (1e-8) of the block's largest. These are the same zero-valued constraints
/// Nastran's AUTOSPC adds. Only components an element declares are considered, and
/// a scalar element or MPC naming one component of a CD grid declares all three of
/// its block. Components no element declares never reach K. Components that an SPC,
/// MPC or RBE already controls are left out, an RBE counting for every CD axis with
/// a part along its basic components. Reactions at these components appear in
/// SPCFORCE like those of any SPC, so a load on a singular direction is reported
/// rather than lost. PARAM,AUTOSPC,NO disables it and a singular K is then reported
/// by the solver.
///
/// A solid's stiffness at its own grid is positive definite in translation, and a
/// shell's is in both blocks when its section has membrane, bending and transverse
/// shear stiffness, so those blocks are not analysed. That keeps the largest decks
/// from computing every element's stiffness an extra time. It also keeps a shell's
/// drill stiffness, which is weak beside its bending when the in-plane shear modulus
/// is small, from being taken for a singularity: the formulation provides it by
/// design.
///
/// This computes the stiffness of the other elements, so a solver calls it once per
/// subcase and passes the result on.
[[nodiscard]] std::vector<Spc> analysis_spcs(const Model &model,
                                             const SubCase &sc);

/// Rotates a basic-frame grid 6-vector (displacements, forces, moments) into the
/// grid's global (CD) system.
[[nodiscard]] std::array<double, 6>
to_grid_frame(const Model &model, NodeId nid, const std::array<double, 6> &v);

[[nodiscard]] DofMap build_analysis_dof_map(const Model &model, const SubCase &sc,
                                            std::span<const Spc> spcs);

/// With the constraints of analysis_spcs. Only used in tests.
[[nodiscard]] DofMap build_analysis_dof_map(const Model &model, const SubCase &sc);

void build_analysis_mpc_system(const Model &model, const SubCase &sc,
                               std::span<const Spc> spcs, DofMap &dof_map,
                               MpcHandler &mpc_handler, bool use_spc_values);

} // namespace vibestran
