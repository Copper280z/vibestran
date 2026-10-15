// src/elements/element_base.cpp

#include "elements/element_base.hpp"

namespace vibestran {

LocalKe ElementBase::lumped_mass_matrix() const {
  const auto nodes = static_cast<int>(node_ids().size());
  const int dofs_per_node = (nodes > 0) ? num_dofs() / nodes : 0;
  return hrz_lumped_mass(mass_matrix(), dofs_per_node);
}

LocalKe hrz_lumped_mass(const LocalKe &consistent, const int dofs_per_node) {
  if (dofs_per_node < 3 || consistent.rows() == 0)
    return consistent;
  const Eigen::Index n = consistent.rows();
  const Eigen::Index nodes = n / dofs_per_node;
  LocalKe lumped = LocalKe::Zero(n, n);
  for (int d = 0; d < 3; ++d) {
    double total = 0.0;
    double diagonal = 0.0;
    for (Eigen::Index a = 0; a < nodes; ++a) {
      const Eigen::Index i = a * dofs_per_node + d;
      diagonal += consistent(i, i);
      for (Eigen::Index b = 0; b < nodes; ++b)
        total += consistent(i, b * dofs_per_node + d);
    }
    if (diagonal <= 0.0)
      continue;
    const double scale = total / diagonal;
    for (Eigen::Index a = 0; a < nodes; ++a) {
      const Eigen::Index i = a * dofs_per_node + d;
      lumped(i, i) = consistent(i, i) * scale;
    }
  }
  return lumped;
}

LocalKe element_mass_matrix(const ElementBase &element, const Model &model) {
  return model.use_lumped_mass() ? element.lumped_mass_matrix()
                                 : element.mass_matrix();
}

} // namespace vibestran
