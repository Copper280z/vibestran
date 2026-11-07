#pragma once
#include "magnetic/solver.hpp"

namespace vibestran::magnetic {

[[nodiscard]] Eigen::SparseMatrix<double>
reduce_conductivity(const Eigen::SparseMatrix<double> &mass, const Gauge &gauge, double scale);

// Conductivity in the material meshes. Each floating planar conductor adds
// one axial electric-field offset, in units of A after multiplication by dt.
// Rigid motion leaves this matrix and its quadrature invariant.
class Diffusion {
public:
  Diffusion(const Model &model, const Mesh &mesh, const CellGeometry &geometry);
  [[nodiscard]] Eigen::SparseMatrix<double> tangent(const Gauge &gauge, double dt) const;
  [[nodiscard]] Eigen::VectorXd increment(const Eigen::VectorXd &a,
                                          const Eigen::VectorXd &previous) const;
  [[nodiscard]] Eigen::VectorXd residual(const Eigen::VectorXd &increment,
                                         const Gauge &gauge, double dt) const;
  [[nodiscard]] EddyStep sample(const Mesh &mesh, const CellGeometry &geometry,
                               const Eigen::VectorXd &increment, double dt,
                               bool fields) const;
  [[nodiscard]] const std::vector<int> &properties() const { return properties_; }
  [[nodiscard]] bool empty() const { return properties_.empty(); }
  [[nodiscard]] const Eigen::SparseMatrix<double> &mass() const { return mass_; }
  [[nodiscard]] Gauge augmented_gauge(const Gauge &field) const;

private:
  struct ConductorCell {
    double sigma{0};
    int component{-1}, property{-1};
    double sign{1}; // sign of the component's offset across an MPERIOD seam
  };
  Eigen::SparseMatrix<double> mass_;
  Eigen::VectorXd component_mass_;
  std::vector<ConductorCell> cells_;
  std::vector<int> properties_;
};

} // namespace vibestran::magnetic
