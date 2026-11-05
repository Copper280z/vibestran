#pragma once

#include "core/dof_map.hpp"
#include "core/mpc_handler.hpp"
#include "core/model.hpp"

#include <vector>

namespace vibestran {

/// The single-point constraints of a structural subcase: Model::structural_spcs
/// plus the AUTOSPC of grid point singularities at grids that connect a CROD.
///
/// A rod stiffens only its own axis, so a truss grid has zero rotational
/// stiffness and, unless its rods span space, singular translations too; those
/// directions are the same zero-valued constraints Nastran's AUTOSPC adds. For
/// each grid and each of its translation and rotation blocks, the component
/// nearest the weakest principal direction, in the grid's CD system, is
/// constrained until the remaining components' smallest principal stiffness is
/// at least PARAM,EPZERO (1e-8) of the block's largest. Components that an SPC,
/// MPC or RBE already controls are left out, and on a CD grid an MPC or RBE
/// removes the whole block, since those work on basic components. Reactions at
/// these components appear in SPCFORCE like those of any SPC, so a load on a
/// singular direction is reported rather than lost. PARAM,AUTOSPC,NO disables
/// it and a singular K is then reported by the solver. Other elements declare
/// the components they stiffen (their active DOFs), so grids without a rod are
/// not analysed.
[[nodiscard]] std::vector<Spc> analysis_spcs(const Model &model,
                                             const SubCase &sc);

[[nodiscard]] DofMap build_analysis_dof_map(const Model &model,
                                            const SubCase &sc);

void build_analysis_mpc_system(const Model &model, const SubCase &sc,
                               DofMap &dof_map, MpcHandler &mpc_handler,
                               bool use_spc_values);

} // namespace vibestran
