#pragma once
// include/solver/linear_static.hpp
// Orchestrates the linear static analysis pipeline:
//   1. Build DOF map (number free DOFs, apply SPCs)
//   2. Build MPC system (RBE2/RBE3, explicit MPCs, CD-frame SPCs)
//   3. Assemble global stiffness matrix K (MPC-transformed)
//   4. Assemble global force vector F (point loads + pressure loads + thermal loads)
//   5. Solve K_red * u_red = F_red
//   6. Recover full displacements (free + dep DOFs)
//   7. Recover element stresses
//   8. Return SubCaseResults

#include "core/dof_map.hpp"
#include "core/mpc_handler.hpp"
#include "core/model.hpp"
#include "core/sparse_matrix.hpp"
#include "io/results.hpp"
#include "solver/solver_backend.hpp"
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vibestran {

/// Per-DOF state for SPC force recovery: whether the DOF is constrained by
/// the active SPC set and the enforced basic-frame displacement value.
struct SpcDofState {
  bool active{false};
  double value{0.0};
};

/// Map each local DOF index of an element to its (node, component) identity.
/// Scalar elements (CELAS/CMASS) reach the basic components of their
/// scalar_connection_dofs; shell/line elements have 6 DOFs per node; solids have 3.
std::vector<std::pair<NodeId, int>> local_dof_identity(const ElementData& elem,
                                                       const Model& model);

class LinearStaticSolver {
public:
    explicit LinearStaticSolver(std::unique_ptr<SolverBackend> backend);

    /// Solve all subcases in the model's analysis case.
    [[nodiscard]] SolverResults solve(const Model& model);

    /// Write each subcase's reduced K, F, u and DOF map in Matrix Market
    /// format with this path prefix (io/system_dump.hpp); empty disables.
    void set_system_dump(std::string prefix) { dump_prefix_ = std::move(prefix); }

    /// The near-nullspace for an AMG preconditioner, over PARAM,AMGMODES and
    /// the default (--amg-modes). See solver/near_nullspace.hpp.
    // cppcheck-suppress unusedFunction -- called from main.cpp
    void set_amg_modes(std::optional<AmgModes> modes) { amg_modes_ = modes; }

private:
    std::unique_ptr<SolverBackend> backend_;
    std::string dump_prefix_;
    std::optional<AmgModes> amg_modes_;

    SubCaseResults solve_subcase(const Model& model, const SubCase& sc,
                                 SolveSummary& summary);

    /// Build DofMap with SPC constraints only (no MPC elimination yet).
    DofMap build_dof_map(const Model& model, const SubCase& sc,
                         std::span<const Spc> spcs);

    /// Build MpcHandler: CD-frame SPCs, RBE2/RBE3, explicit MPCs.
    /// Modifies dof_map: constrains dependent DOFs.
    void build_mpc_system(const Model& model, const SubCase& sc,
                          std::span<const Spc> spcs, DofMap& dof_map,
                          MpcHandler& mpc_handler);

    /// Assemble K and F using the pre-MPC dof_map (from mpc_handler.full_dof_map()).
    void assemble(const Model& model,
                  const SubCase& sc,
                  const MpcHandler& mpc_handler,
                  SparseMatrixBuilder& K_builder,
                  std::vector<double>& F);

    /// Apply concentrated loads (FORCE, MOMENT) to F via MpcHandler
    void apply_point_loads(const Model& model,
                           const SubCase& sc,
                           const MpcHandler& mpc_handler,
                           std::vector<double>& F);

    /// Apply pressure loads (PLOAD, PLOAD1, PLOAD2, PLOAD4) to F via MpcHandler
    void apply_pressure_loads(const Model& model,
                              const SubCase& sc,
                              const MpcHandler& mpc_handler,
                              std::vector<double>& F);

    /// Apply inertial loads (GRAV, ACCEL1) to F via MpcHandler
    void apply_inertial_loads(const Model& model,
                              const SubCase& sc,
                              const MpcHandler& mpc_handler,
                              std::vector<double>& F);

    /// Apply thermal loads to F via MpcHandler
    void apply_thermal_loads(const Model& model,
                             const SubCase& sc,
                             const MpcHandler& mpc_handler,
                             SparseMatrixBuilder& K_builder,
                             std::vector<double>& F);

    /// Recover stresses and displacements from full (pre-MPC) dof_map + u_free
    SubCaseResults recover_results(const Model& model,
                                   const SubCase& sc,
                                   const DofMap& dof_map,
                                   const std::vector<double>& u_free);

    /// Grid point force balance and multipoint-constraint forces.
    struct GridForceOutput {
        std::vector<SpcForce> spc_forces;
        std::vector<SpcForce> mpc_forces;
        std::vector<GridPointForceBalance> gp_forces;
        /// Set when MPC force recovery failed; all force lists are empty.
        std::optional<std::string> error;
    };

    /// Recover Ku-F, physical MPC/RBE forces, and support reactions once.
    /// SPCFORCES, MPCFORCES and GPFORCE share the same basic-frame recovery;
    /// results are rotated to each grid's CD frame only for output. A failed
    /// recovery is returned in `error` rather than thrown.
    GridForceOutput
    compute_grid_forces(const Model& model, const SubCase& sc,
                        std::span<const Spc> spcs,
                        const MpcHandler& mpc_handler,
                        const std::vector<double>& u_free,
                        bool want_spc, bool want_mpc, bool want_gp) const;

    /// Subtract pressure-type equivalent nodal loads (PLOAD, PLOAD1, PLOAD2,
    /// PLOAD4) at the given active DOFs from `react` (basic frame), so that
    /// react = K u - F like the other load-subtraction helpers.  Defined in
    /// linear_static_pressure.cpp so it can reuse the face-integration
    /// helpers.
    void compute_spc_pressure_loads(
        const Model& model, const SubCase& sc, const DofMap& dof_map,
        const std::unordered_map<NodeId, std::array<SpcDofState, 6>>& spc,
        std::unordered_map<NodeId, std::array<double, 6>>& react) const;
};

} // namespace vibestran
