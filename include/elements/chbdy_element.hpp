#pragma once
// include/elements/chbdy_element.hpp
// CHBDY boundary element (surface convection, applied flux, ambient coupling).
//
// MVP geometry types: POINT (1 node), AREA3 (triangle), AREA4 (quad).
// LINE / REV / ELCYL / FTUBE are recognized at parse-time but throw at
// element-construction time.

#include "core/dof_map.hpp"
#include "core/model.hpp"
#include <Eigen/Dense>
#include <span>
#include <vector>

namespace vibestran {

class ChbdyElementImpl {
public:
  ChbdyElementImpl(const ChbdyElement &data, const Model &model);

  [[nodiscard]] std::span<const NodeId> surface_nodes() const noexcept {
    return {data_.nodes.data(), data_.nodes.size()};
  }
  [[nodiscard]] std::span<const NodeId> ambient_nodes() const noexcept {
    return {data_.ambient_nodes.data(), data_.ambient_nodes.size()};
  }
  /// Outward unit normal (only meaningful for AREA3/AREA4; zero for POINT).
  /// Currently only used in tests.
  [[nodiscard]] const Vec3 &outward_normal() const noexcept { return normal_; }
  /// Total surface area.  Currently only used in tests.
  [[nodiscard]] double area() const noexcept { return area_; }

  /// Consistent convection conductance block (surface × surface), n × n.
  /// Always positive-semidefinite; couples surface nodes among themselves
  /// (heat flow at node i is H·integral(N_i·(T_i − T_amb))).
  [[nodiscard]] Eigen::MatrixXd convection_conductance() const;

  /// Distributed flux load (QBDY1/QHBDY/QBDY2) onto surface nodes.
  /// q is either a scalar (uniform) or one value per node.
  [[nodiscard]] Eigen::VectorXd applied_flux_load(double q_uniform) const;
  [[nodiscard]] Eigen::VectorXd
  applied_flux_load_per_node(std::span<const double> q_per_node) const;

  /// QVECT contribution: projects flux vector E = q0·direction onto the
  /// inward normal (−n̂) and distributes by tributary area. Returns zero when
  /// the face is parallel to E or oriented away from it.
  [[nodiscard]] Eigen::VectorXd
  directional_flux_load(double q0, const Vec3 &direction) const;

private:
  ChbdyElement data_;
  const Model &model_;
  double area_{0.0};
  double film_h_{0.0};
  double absorptivity_{0.0};
  std::vector<double> area_frac_; // per-surface-node tributary area
  Eigen::MatrixXd conv_;          // consistent convection matrix (surface part)
  Eigen::VectorXd N_int_;         // ∫ N_i dA  (= area_frac_ × something)
  Vec3 normal_{0, 0, 0};          // outward unit normal (AREA3/AREA4)

  void compute_geometry();
};

} // namespace vibestran
