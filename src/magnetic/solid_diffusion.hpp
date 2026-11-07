#pragma once
#include "magnetic/solver.hpp"

namespace vibestran::magnetic {

struct SolidElectrode {
  int id{0};
  std::complex<double> voltage;
  std::vector<int> equations; // indices in the full A/phi vector
};
struct SolidConditions {
  Gauge gauge;
  std::vector<SolidElectrode> electrodes;
};

// A-V conductivity operator on edge potentials and conducting nodal
// scalars phi = V/(i omega). The physical electric field is
// -i omega (A + grad phi); charge conservation is the scalar row.
class SolidDiffusion {
public:
  SolidDiffusion(const Model &model, const Mesh &mesh);
  [[nodiscard]] const Eigen::SparseMatrix<double> &mass() const { return mass_; }
  [[nodiscard]] Result<SolidConditions> conditions(const Model &model, const Mesh &mesh,
                                                   const Gauge &field, int set) const;
  void sample(const Mesh &mesh, const Eigen::VectorXd &real,
              const Eigen::VectorXd &imag, double omega, HarmonicResult &out) const;

private:
  struct Solid {
    double sigma{0};
    int component{-1}, property{-1};
    std::vector<int> scalar;
  };
  Eigen::SparseMatrix<double> mass_;
  std::vector<Solid> cells_;
  std::vector<int> references_, properties_;
};

} // namespace vibestran::magnetic
