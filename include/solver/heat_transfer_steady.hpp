#pragma once
// include/solver/heat_transfer_steady.hpp
// Linear steady-state heat conduction (SOL 153).
//
//   [K]{T} = {Q}
//
// where [K] is the thermal conductance assembled from MAT4/MAT5 + element
// geometry, and {Q} is the heat-load vector from QVOL/QHBDY/QBDY1/QBDY2 and
// CHBDY ambient-temperature convection.
//
// Boundary conditions: TEMP cards in the case's TEMP(LOAD) set act as Dirichlet
// constraints (prescribed nodal temperature).  TEMPD provides a default for
// nodes referenced in the same set but not given an individual TEMP card.
// SPCs are unused in this solver (NASTRAN uses component-0 SPC for thermal BC;
// we keep that for a later pass).

#include "core/model.hpp"
#include "io/results.hpp"
#include "solver/solver_backend.hpp"
#include <memory>

namespace vibestran {

class HeatTransferSteadySolver {
public:
  explicit HeatTransferSteadySolver(std::unique_ptr<SolverBackend> backend);

  /// Solve every subcase in `model.analysis.subcases`.
  [[nodiscard]] SolverResults solve(const Model &model);

private:
  [[nodiscard]] SubCaseResults solve_subcase(const Model &model,
                                             const SubCase &sc);

  std::unique_ptr<SolverBackend> backend_;
};

} // namespace vibestran
