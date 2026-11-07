#include "solid_diffusion.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <format>
#include <numeric>
#include <set>

namespace vibestran::magnetic {
namespace {
// The scalar gradient and the Nédélec basis use the same covariant mapping.
template <class Function> void quadrature_basis(const Cell &cell, const Function &visit) {
  for (const auto &q : reference_quadrature(cell)) {
    const auto [point, jacobian] = mapping(cell, q.position);
    auto basis = edge_basis(cell, point);
    const Matrix gradient = jacobian.inverse().transpose();
    for (const auto &value : reference_shape(cell, q.position).gradients)
      basis.push_back(gradient * value);
    visit(basis, q.weight * std::abs(jacobian.determinant()));
  }
}
} // namespace

SolidDiffusion::SolidDiffusion(const Model &model, const Mesh &mesh)
    : cells_(mesh.cells.size()) {
  std::map<int, int> properties;
  for (size_t i = 0; i < cells_.size(); ++i) {
    const auto &cell = mesh.cells[i];
    if (const auto it = model.conductivity.find(cell.material); it != model.conductivity.end()) {
      cells_[i].sigma = it->second;
      properties.emplace(cell.property, 0);
    }
  }
  for (auto &[pid, index] : properties) {
    index = static_cast<int>(properties_.size());
    properties_.push_back(pid);
  }
  std::vector<int> parent(cells_.size());
  std::iota(parent.begin(), parent.end(), 0);
  const auto root = [&](int i) {
    while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  for (const auto &face : mesh.faces) {
    const auto [i, j] = face.cells;
    if (j >= 0 && cells_[i].sigma > 0 && cells_[j].sigma > 0)
      parent[root(j)] = root(i);
  }
  // Distinct solids can touch at a point or an edge without sharing an
  // electrical unknown. Only a shared face is a conducting connection.
  std::map<int, int> components;
  std::map<std::pair<int, int>, int> nodes;
  int count = static_cast<int>(mesh.dofs());
  std::vector<Eigen::Triplet<double>> entries;
  for (size_t i = 0; i < cells_.size(); ++i) {
    auto &solid = cells_[i];
    if (solid.sigma == 0)
      continue;
    const auto &cell = mesh.cells[i];
    const auto [component, added] =
        components.try_emplace(root(static_cast<int>(i)), static_cast<int>(components.size()));
    if (added)
      references_.push_back(count);
    solid.component = component->second;
    solid.property = properties.at(cell.property);
    for (int node : cell.nodes) {
      const auto [index, inserted] = nodes.try_emplace(std::pair{solid.component, node}, count);
      count += inserted ? 1 : 0;
      solid.scalar.push_back(index->second);
    }
    std::vector<int> dofs(cell.edges.begin(), cell.edges.end());
    dofs.insert(dofs.end(), solid.scalar.begin(), solid.scalar.end());
    Eigen::MatrixXd local = Eigen::MatrixXd::Zero(dofs.size(), dofs.size());
    quadrature_basis(cell, [&](const std::vector<Vector> &basis, double weight) {
      for (size_t j = 0; j < dofs.size(); ++j)
        for (size_t k = 0; k < dofs.size(); ++k)
          local(j, k) += solid.sigma * weight * basis[j].dot(basis[k]);
    });
    for (size_t j = 0; j < dofs.size(); ++j)
      for (size_t k = 0; k < dofs.size(); ++k)
        entries.emplace_back(dofs[j], dofs[k], local(j, k));
  }
  mass_.resize(count, count);
  mass_.setFromTriplets(entries.begin(), entries.end());
}

Result<SolidConditions> SolidDiffusion::conditions(const Model &model, const Mesh &mesh,
                                                   const Gauge &field, int set) const {
  SolidConditions out{field, {}};
  std::vector<bool> anchored(references_.size());
  std::set<int> fixed;
  for (const auto &value : model.terminal_voltages) {
    if (value.set != set)
      continue;
    auto faces = selected_faces(model, mesh, model.terminals.at(value.terminal));
    if (const auto *e = std::get_if<Error>(&faces))
      return Error{std::format("MTERM {}: {}", value.terminal, e->message)};
    std::set<int> nodes;
    for (const auto &selection : std::get<std::vector<std::pair<int, int>>>(faces)) {
      const auto &face = mesh.faces[selection.first];
      const int ci = face.cells[0];
      const auto &solid = cells_[ci];
      if (face.cells[1] >= 0 || solid.sigma == 0)
        return Error{std::format("MTERM {} must select exterior field faces of conducting cells "
                                 "(element {})", value.terminal, mesh.cells[ci].id)};
      anchored[solid.component] = true;
      for (int v : face.nodes) {
        const auto &cell_nodes = mesh.cells[ci].nodes;
        const auto local = std::find(cell_nodes.begin(), cell_nodes.end(), v) - cell_nodes.begin();
        nodes.insert(solid.scalar[local]);
      }
    }
    if (nodes.empty())
      return Error{std::format("MTERM {} selects no conducting faces", value.terminal)};
    for (int node : nodes)
      if (!fixed.insert(node).second)
        return Error{std::format("MTERM {} shares a conducting grid with another active terminal",
                                 value.terminal)};
    out.electrodes.push_back({value.terminal, {value.real, value.imag},
                              {nodes.begin(), nodes.end()}});
  }
  for (size_t i = 0; i < references_.size(); ++i)
    if (!anchored[i])
      fixed.insert(references_[i]);
  for (auto i = static_cast<int>(mesh.dofs()); i < mass_.rows(); ++i) {
    out.gauge.equations.push_back(fixed.contains(i) ? -1 : out.gauge.size++);
    out.gauge.sign.push_back(1.);
  }
  return out;
}

void SolidDiffusion::sample(const Mesh &mesh, const Eigen::VectorXd &real,
                            const Eigen::VectorXd &imag, double omega,
                            HarmonicResult &out) const {
  out.properties = properties_;
  out.property_loss.resize(properties_.size());
  out.current_real.resize(cells_.size(), Vector::Zero());
  out.current_imag.resize(cells_.size(), Vector::Zero());
  out.loss_density.resize(cells_.size());
  for (size_t i = 0; i < cells_.size(); ++i) {
    const auto &solid = cells_[i];
    if (solid.sigma == 0)
      continue;
    const auto &cell = mesh.cells[i];
    std::vector<int> dofs(cell.edges.begin(), cell.edges.end());
    dofs.insert(dofs.end(), solid.scalar.begin(), solid.scalar.end());
    double loss = 0, volume = 0;
    quadrature_basis(cell, [&](const std::vector<Vector> &basis, double weight) {
      Vector jr = Vector::Zero(), ji = Vector::Zero();
      for (size_t k = 0; k < dofs.size(); ++k) {
        jr += imag[dofs[k]] * basis[k];
        ji -= real[dofs[k]] * basis[k];
      }
      jr *= solid.sigma * omega;
      ji *= solid.sigma * omega;
      volume += weight;
      out.current_real[i] += weight * jr;
      out.current_imag[i] += weight * ji;
      loss += weight * (jr.squaredNorm() + ji.squaredNorm()) / (2 * solid.sigma);
    });
    out.current_real[i] /= volume;
    out.current_imag[i] /= volume;
    out.loss_density[i] = loss / volume;
    out.property_loss[solid.property] += loss;
    out.mean_loss += loss;
  }
}

} // namespace vibestran::magnetic
