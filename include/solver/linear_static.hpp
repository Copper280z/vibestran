#pragma once
// include/solver/linear_static.hpp
// Orchestrates the linear static analysis pipeline:
//   1. Build DOF map (number free DOFs, apply SPCs)
//   2. Assemble global stiffness matrix K
//   3. Assemble global force vector F (point loads + thermal loads)
//   4. Solve K * u = F
//   5. Recover element stresses
//   6. Return SubCaseResults

#include "core/model.hpp"
#include "core/dof_map.hpp"
#include "solver/solver_backend.hpp"
#include "io/results.hpp"
#include <memory>

namespace nastran {

class LinearStaticSolver {
public:
    explicit LinearStaticSolver(std::unique_ptr<SolverBackend> backend);

    /// Solve all subcases in the model's analysis case.
    [[nodiscard]] SolverResults solve(const Model& model);

private:
    std::unique_ptr<SolverBackend> backend_;

    SubCaseResults solve_subcase(const Model& model,
                                  const SubCase& sc);

    /// Build DofMap, then apply SPC constraints
    DofMap build_dof_map(const Model& model, const SubCase& sc);

    /// Assemble K and F for a subcase
    void assemble(const Model& model,
                  const SubCase& sc,
                  const DofMap& dof_map,
                  SparseMatrixBuilder& K_builder,
                  std::vector<double>& F);

    /// Apply point loads (FORCE, MOMENT) to F
    void apply_point_loads(const Model& model,
                            const SubCase& sc,
                            const DofMap& dof_map,
                            std::vector<double>& F);

    /// Apply thermal loads to F (Fth = ∫ Bᵀ D α ΔT dV)
    void apply_thermal_loads(const Model& model,
                              const SubCase& sc,
                              const DofMap& dof_map,
                              SparseMatrixBuilder& K_builder,
                              std::vector<double>& F);

    /// Recover stresses from solution vector
    SubCaseResults recover_results(const Model& model,
                                    const SubCase& sc,
                                    const DofMap& dof_map,
                                    const std::vector<double>& u_free);
};

} // namespace nastran
