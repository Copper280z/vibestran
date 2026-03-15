// src/core/sparse_matrix.cpp
#include "core/sparse_matrix.hpp"
#include "core/dof_map.hpp"
#include <Eigen/Sparse>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace nastran {

void SparseMatrixBuilder::add_element_stiffness(
    std::span<const int32_t> global_dofs, std::span<const double> ke) {
  const int n = static_cast<int>(global_dofs.size());
  if (static_cast<int>(ke.size()) != n * n)
    throw std::invalid_argument("ke size mismatch");

  for (int i = 0; i < n; ++i) {
    if (global_dofs[i] == CONSTRAINED_DOF)
      continue;
    for (int j = 0; j < n; ++j) {
      if (global_dofs[j] == CONSTRAINED_DOF)
        continue;
      double v = ke[static_cast<size_t>(i * n + j)];
      if (v != 0.0)
        triplets_.push_back({global_dofs[i], global_dofs[j], v});
    }
  }
}

void SparseMatrixBuilder::add_element_force(
    std::span<const int32_t> global_dofs, std::span<const double> fe,
    std::vector<double> &F) {
  const int n = static_cast<int>(global_dofs.size());
  for (int i = 0; i < n; ++i) {
    if (global_dofs[i] == CONSTRAINED_DOF)
      continue;
    if (global_dofs[i] >= static_cast<int>(F.size()))
      throw std::out_of_range("Force vector index out of range");
    F[static_cast<size_t>(global_dofs[i])] += fe[static_cast<size_t>(i)];
  }
}

SparseMatrixBuilder::CsrData SparseMatrixBuilder::build_csr() const {
  // Use Eigen to sum duplicates and compress into CSR
  using ESM = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  std::vector<Eigen::Triplet<double>> eigen_trips;
  eigen_trips.reserve(triplets_.size());
  for (const auto &t : triplets_)
    eigen_trips.emplace_back(t.row, t.col, t.value);

  ESM mat(size_, size_);
  mat.setFromTriplets(eigen_trips.begin(), eigen_trips.end());
  mat.makeCompressed();

  const int nnz = static_cast<int>(mat.nonZeros());
  CsrData csr;
  csr.n = size_;
  csr.nnz = nnz;
  csr.row_ptr.assign(mat.outerIndexPtr(), mat.outerIndexPtr() + size_ + 1);
  csr.col_ind.assign(mat.innerIndexPtr(), mat.innerIndexPtr() + nnz);
  csr.values.assign(mat.valuePtr(), mat.valuePtr() + nnz);
  return csr;
}

} // namespace nastran
