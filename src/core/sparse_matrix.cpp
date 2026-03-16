// src/core/sparse_matrix.cpp
#include "core/sparse_matrix.hpp"
#include "core/dof_map.hpp"
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

SparseMatrixBuilder::CsrData SparseMatrixBuilder::build_csr() {
  // Sort in-place by (row, col) so duplicates are adjacent.
  std::sort(triplets_.begin(), triplets_.end(), [](const Triplet &a, const Triplet &b) {
    return a.row != b.row ? a.row < b.row : a.col < b.col;
  });

  // Deduplicate in-place: sum values for identical (row, col).
  // Write deduplicated entries back into triplets_ without any extra allocation.
  size_t out = 0;
  for (size_t i = 0; i < triplets_.size(); ++i) {
    if (out > 0 && triplets_[i].row == triplets_[out - 1].row &&
        triplets_[i].col == triplets_[out - 1].col) {
      triplets_[out - 1].value += triplets_[i].value;
    } else {
      triplets_[out++] = triplets_[i];
    }
  }
  triplets_.resize(out);

  // Build CSR directly from the sorted, deduplicated triplets.
  const int nnz = static_cast<int>(triplets_.size());
  CsrData csr;
  csr.n   = size_;
  csr.nnz = nnz;
  csr.row_ptr.assign(static_cast<size_t>(size_ + 1), 0);
  csr.col_ind.resize(static_cast<size_t>(nnz));
  csr.values.resize(static_cast<size_t>(nnz));

  // Histogram: count entries per row.
  for (const auto &t : triplets_)
    ++csr.row_ptr[static_cast<size_t>(t.row + 1)];

  // Prefix sum to get row_ptr.
  std::partial_sum(csr.row_ptr.begin(), csr.row_ptr.end(), csr.row_ptr.begin());

  // Fill col_ind and values.
  for (int i = 0; i < nnz; ++i) {
    csr.col_ind[static_cast<size_t>(i)] = triplets_[static_cast<size_t>(i)].col;
    csr.values[static_cast<size_t>(i)]  = triplets_[static_cast<size_t>(i)].value;
  }

  // Release source memory now that CSR is built.
  triplets_.clear();
  triplets_.shrink_to_fit();

  return csr;
}

} // namespace nastran
