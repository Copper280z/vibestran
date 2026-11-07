#include "diffusion.hpp"

namespace vibestran::magnetic {

Diffusion::Diffusion(const Model &model, const Mesh &mesh, const CellGeometry &geometry)
    : cells_(mesh.cells.size()) {
  std::map<int, int> properties;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto &cell = mesh.cells[c];
    if (auto it = model.conductivity.find(cell.material); it != model.conductivity.end()) {
      cells_[c].sigma = it->second;
      properties.emplace(cell.property, 0);
    }
  }
  for (auto &[pid, index] : properties) {
    index = static_cast<int>(properties_.size());
    properties_.push_back(pid);
  }

  int count = 0;
  if (!mesh.axisymmetric) {
    // Common edges and their periodic images represent electrical contact.
    // An anti-periodic image reverses the electric field offset. A component
    // connected to its own sign-reversed image therefore has zero offset.
    std::vector<std::vector<std::pair<int, double>>> adjacent(cells_.size());
    const auto connect = [&](int i, int j, double sign) {
      if (i >= 0 && j >= 0 && cells_[i].sigma > 0 && cells_[j].sigma > 0) {
        adjacent[i].emplace_back(j, sign);
        adjacent[j].emplace_back(i, sign);
      }
    };
    for (const auto &face : mesh.faces)
      connect(face.cells[0], face.cells[1], 1.);
    if (mesh.period)
      for (const auto [first, second] : mesh.period->face_pairs)
        connect(mesh.faces[first].cells[0], mesh.faces[second].cells[0], mesh.period->sign());
    std::vector<double> signs(cells_.size());
    for (size_t seed = 0; seed < cells_.size(); ++seed) {
      if (signs[seed] != 0 || cells_[seed].sigma == 0)
        continue;
      signs[seed] = 1;
      std::vector<int> members{static_cast<int>(seed)};
      bool forced_zero = false;
      for (size_t p = 0; p < members.size(); ++p) {
        const int i = members[p];
        for (const auto [j, sign] : adjacent[i]) {
          const double image = sign * signs[i];
          if (signs[j] == 0) {
            signs[j] = image;
            members.push_back(j);
          } else if (signs[j] != image)
            forced_zero = true;
        }
      }
      if (forced_zero)
        continue;
      for (const int i : members) {
        cells_[i].component = count;
        cells_[i].sign = signs[i];
      }
      ++count;
    }
  }
  const auto n = static_cast<int>(mesh.dofs());
  std::vector<Eigen::Triplet<double>> entries;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    auto &conductor = cells_[c];
    if (conductor.sigma == 0)
      continue;
    const auto &cell = mesh.cells[c];
    conductor.property = properties.at(cell.property);
    const int ci = static_cast<int>(c);
    Eigen::Matrix4d local = Eigen::Matrix4d::Zero();
    Eigen::Vector4d coupling = Eigen::Vector4d::Zero();
    double total = 0;
    for (int q = 0; q < geometry.points(ci); ++q) {
      const auto basis = geometry.nodal_values(ci, q);
      const double weight = conductor.sigma * geometry.weight(ci, q) * geometry.measure(ci, q);
      total += weight;
      for (size_t i = 0; i < basis.size(); ++i) {
        coupling[i] -= weight * basis[i];
        for (size_t j = 0; j < basis.size(); ++j)
          local(i, j) += weight * basis[i] * basis[j];
      }
    }
    for (size_t i = 0; i < cell.nodes.size(); ++i) {
      for (size_t j = 0; j < cell.nodes.size(); ++j)
        entries.emplace_back(cell.nodes[i], cell.nodes[j], local(i, j));
      if (conductor.component >= 0) {
        const int offset = n + conductor.component;
        entries.emplace_back(cell.nodes[i], offset, conductor.sign * coupling[i]);
        entries.emplace_back(offset, cell.nodes[i], conductor.sign * coupling[i]);
      }
    }
    if (conductor.component >= 0) {
      const int offset = n + conductor.component;
      entries.emplace_back(offset, offset, total);
    }
  }
  mass_.resize(n + count, n + count);
  mass_.setFromTriplets(entries.begin(), entries.end());
  component_mass_ = mass_.diagonal().tail(count);
}

Eigen::SparseMatrix<double>
reduce_conductivity(const Eigen::SparseMatrix<double> &mass, const Gauge &gauge, double scale) {
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<size_t>(mass.nonZeros()));
  for (int col = 0; col < mass.outerSize(); ++col)
    for (Eigen::SparseMatrix<double>::InnerIterator it(mass, col); it; ++it) {
      const auto row = it.row();
      const int i = gauge.equations[row], j = gauge.equations[col];
      if (i >= 0 && j >= 0)
        entries.emplace_back(i, j, it.value() * gauge.sign[row] * gauge.sign[col] * scale);
    }
  Eigen::SparseMatrix<double> out(gauge.size, gauge.size);
  out.setFromTriplets(entries.begin(), entries.end());
  return out;
}

Gauge Diffusion::augmented_gauge(const Gauge &field) const {
  auto gauge = field;
  for (Eigen::Index i = 0; i < component_mass_.size(); ++i) {
    gauge.equations.push_back(gauge.size++);
    gauge.sign.push_back(1.);
  }
  return gauge;
}

Eigen::SparseMatrix<double> Diffusion::tangent(const Gauge &gauge, double dt) const {
  return reduce_conductivity(mass_, augmented_gauge(gauge), 1 / dt);
}

Eigen::VectorXd Diffusion::increment(const Eigen::VectorXd &a,
                                      const Eigen::VectorXd &previous) const {
  Eigen::VectorXd delta(mass_.rows());
  delta.head(a.size()) = a - previous;
  if (component_mass_.size()) {
    // Compute the offsets exactly for each line-search trial. The Newton
    // tangent still uses the sparse augmented system with the offsets.
    const auto columns = mass_.topRightCorner(a.size(), component_mass_.size());
    delta.tail(component_mass_.size()) =
        -(columns.transpose() * delta.head(a.size())).cwiseQuotient(component_mass_);
  }
  return delta;
}

Eigen::VectorXd Diffusion::residual(const Eigen::VectorXd &delta,
                                     const Gauge &gauge, double dt) const {
  const Eigen::VectorXd full = (mass_ * delta) / dt;
  Eigen::VectorXd out(gauge.size + component_mass_.size());
  out.head(gauge.size) = gauge.reduce(full);
  out.tail(component_mass_.size()) = full.tail(component_mass_.size());
  return out;
}

EddyStep Diffusion::sample(const Mesh &mesh, const CellGeometry &geometry,
                          const Eigen::VectorXd &delta, double dt, bool fields) const {
  EddyStep out;
  out.property_loss.resize(properties_.size());
  if (fields) {
    out.current_density.resize(mesh.cells.size(), Vector::Zero());
    out.loss_density.resize(mesh.cells.size());
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto &conductor = cells_[c];
    if (conductor.sigma == 0)
      continue;
    const auto &cell = mesh.cells[c];
    const double offset = conductor.component >= 0
                              ? conductor.sign * delta[mesh.dofs() + conductor.component]
                              : 0;
    double current = 0, loss = 0, volume = 0;
    const int ci = static_cast<int>(c);
    for (int q = 0; q < geometry.points(ci); ++q) {
      const auto basis = geometry.nodal_values(ci, q);
      // Interpolate the differences from the offset to preserve cancellation
      // even when the potential changes uniformly in a floating component.
      double field = 0;
      for (size_t i = 0; i < basis.size(); ++i)
        field += basis[i] * (offset - delta[cell.nodes[i]]);
      field /= dt;
      const double dv = geometry.weight(ci, q) * geometry.measure(ci, q);
      current += dv * conductor.sigma * field;
      loss += dv * conductor.sigma * field * field;
      volume += dv;
    }
    out.loss += loss;
    out.property_loss[conductor.property] += loss;
    if (fields) {
      out.current_density[c] = (current / volume) * mesh.normal;
      out.loss_density[c] = loss / volume;
    }
  }
  if (mesh.period) {
    out.loss *= mesh.period->sectors;
    for (double &loss : out.property_loss)
      loss *= mesh.period->sectors;
  }
  return out;
}

} // namespace vibestran::magnetic
