#pragma once
// include/elements/thermal_element_base.hpp
// Abstract interface for thermal finite elements (1 scalar temperature DOF per
// node).  Kept separate from ElementBase so structural elements need not grow
// stub thermal methods.
//
// Each concrete element implements:
//   1. conductance_matrix() — local Ke (n × n), n = num_nodes
//   2. capacity_matrix()    — local Be (n × n) for transient analysis
//                             (stubbed to return empty for steady-state-only
//                             elements; the steady-state solver never calls it)
//   3. volumetric_heat_load() — consistent integral of N·q_vol dV
//   4. heat_flux()          — q = -k·∇T at the element centroid
//
// Extension seam: a future nonlinear path can override conductance_matrix()
// with a temperature-dependent overload; the steady-state solver only calls
// the no-argument version.

#include "core/dof_map.hpp"
#include "core/model.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <span>
#include <vector>

namespace vibestran {

class ThermalElement {
public:
  virtual ~ThermalElement() = default;

  [[nodiscard]] virtual ElementId id() const noexcept = 0;
  [[nodiscard]] virtual ElementType type() const noexcept = 0;
  [[nodiscard]] virtual int num_nodes() const noexcept = 0;
  [[nodiscard]] virtual std::span<const NodeId> node_ids() const noexcept = 0;

  /// Element conductance matrix Ke = ∫ Bᵀ k B dV  (num_nodes × num_nodes).
  [[nodiscard]] virtual Eigen::MatrixXd conductance_matrix() const = 0;

  /// Element capacity matrix Be = ∫ ρ·cp·Nᵀ N dV.  Stubbed to empty for the
  /// steady-state path; transient solver will override.
  [[nodiscard]] virtual Eigen::MatrixXd capacity_matrix() const {
    return Eigen::MatrixXd(0, 0);
  }

  /// Consistent volumetric-heat load vector P = ∫ N·q_vol dV (num_nodes).
  [[nodiscard]] virtual Eigen::VectorXd
  volumetric_heat_load(double q_vol) const = 0;

  /// Centroidal heat flux q = -k·∇T  (size = element spatial dimension).
  [[nodiscard]] virtual Eigen::VectorXd
  heat_flux(std::span<const double> nodal_temps) const = 0;

};

} // namespace vibestran
