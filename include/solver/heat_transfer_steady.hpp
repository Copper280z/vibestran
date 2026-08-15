#pragma once
// include/solver/heat_transfer_steady.hpp
// Linear steady-state heat conduction (SOL 153).
//
//   [K]{T} = {Q}
//
// where [K] is the thermal conductance assembled from MAT4/MAT5 + element
// geometry, and {Q} is the heat-load vector from QVOL/QHBDY/QBDY1/QBDY2 and
// CHBDY convection.
//
// Boundary conditions: SPC/SPC1 entries in the case's selected SPC set prescribe
// nodal temperatures. Component 0 and component 1 are accepted for the scalar
// thermal degree of freedom. TEMP/TEMPD remain temperature-field data and are
// not constraints in a heat-transfer solve.

#include "core/model.hpp"
#include "io/results.hpp"
#include "solver/solver_backend.hpp"
#include <memory>
#include <optional>
#include <span>

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

/// Return the heat-result index belonging to the latest heat subcase preceding
/// `target_subcase` in deck order. Returns nullopt when no heat subcase precedes
/// the target. The result ordering must match the heat subcases' deck ordering.
[[nodiscard]] std::optional<size_t>
preceding_heat_result_index(std::span<const SubCase> subcases,
                            size_t target_subcase);

} // namespace vibestran
