// src/core/sparse_matrix.cpp
#include "core/sparse_matrix.hpp"
#include "core/dof_map.hpp"
#include <algorithm>
#include <iterator>
#include <numeric>
#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif
#include <stdexcept>

namespace vibestran {

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
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
      if (global_dofs[i] < global_dofs[j])
        continue;
      double v = ke[static_cast<size_t>(i * n + j)];
      if (v != 0.0)
        triplets_.push_back({global_dofs[i], global_dofs[j], v});
    }
  }
}

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
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

void SparseMatrixBuilder::merge_from(SparseMatrixBuilder&& other) {
  if (size_ != other.size_)
    throw std::invalid_argument("SparseMatrixBuilder size mismatch in merge");

  triplets_.insert(triplets_.end(),
                   std::make_move_iterator(other.triplets_.begin()),
                   std::make_move_iterator(other.triplets_.end()));
  other.triplets_.clear();
  other.triplets_.shrink_to_fit();
}

SparseMatrixBuilder::CsrData SparseMatrixBuilder::build_csr() {
  const auto n_trips = static_cast<int>(triplets_.size());
  bool lower_only = true;
  for (const auto &t : triplets_) {
    if (t.row < t.col) {
      lower_only = false;
      break;
    }
  }

  // Pass 1: histogram of row indices → row-bucket boundaries.
  // Two O(n) sequential passes over compact data instead of one O(n log n)
  // global sort over 9+ GB — the global sort is memory-bandwidth-bound and
  // does not scale with core count.
  std::vector<int> bucket_ptr(static_cast<size_t>(size_ + 1), 0);
  for (const auto &t : triplets_)
    ++bucket_ptr[static_cast<size_t>(t.row + 1)];
  std::partial_sum(bucket_ptr.begin(), bucket_ptr.end(), bucket_ptr.begin());

  // Pass 2: scatter triplets into row-ordered staging buffer using cursor
  // positions derived from bucket_ptr.
  std::vector<int> cursor(bucket_ptr.begin(), bucket_ptr.begin() + size_);
  std::vector<Triplet> staged(static_cast<size_t>(n_trips));
  for (const auto &t : triplets_)
    staged[static_cast<size_t>(cursor[static_cast<size_t>(t.row)]++)] = t;

  // Free source memory now that scatter is complete.
  triplets_.clear();
  triplets_.shrink_to_fit();

  // Pass 3: sort within each row by col.
  // Each row has O(elements_per_node²) entries (~85 for CHEXA8) — tiny arrays
  // that fit in L1 cache. Parallel dispatch across rows via TBB when available;
  // std::execution::par_unseq is not supported by AppleClang.
  auto sort_row = [&](int row) {
    auto beg = staged.begin() + bucket_ptr[static_cast<size_t>(row)];
    auto end = staged.begin() + bucket_ptr[static_cast<size_t>(row + 1)];
    std::sort(beg, end, [](const Triplet &a, const Triplet &b) {
      return a.col < b.col;
    });
  };
#ifdef HAVE_TBB
  tbb::parallel_for(tbb::blocked_range<int>(0, size_),
                    [&](const tbb::blocked_range<int> &r) {
                      for (int row = r.begin(); row != r.end(); ++row)
                        sort_row(row);
                    });
#else
  for (int row = 0; row < size_; ++row)
    sort_row(row);
#endif

  // Pass 4: dedup within each row (staged is now sorted by col within rows),
  // and write to final CSR arrays.
  CsrData csr;
  csr.n = size_;
  csr.row_ptr.resize(static_cast<size_t>(size_ + 1));
  csr.col_ind.resize(static_cast<size_t>(n_trips)); // upper bound; shrunk below
  csr.values.resize(static_cast<size_t>(n_trips));
  csr.row_ptr[0] = 0;

  int out = 0;
  for (int row = 0; row < size_; ++row) {
    const int beg = bucket_ptr[static_cast<size_t>(row)];
    const int end = bucket_ptr[static_cast<size_t>(row + 1)];
    for (int i = beg; i < end; ++i) {
      if (i > beg &&
          staged[static_cast<size_t>(i)].col ==
              staged[static_cast<size_t>(i - 1)].col) {
        csr.values[static_cast<size_t>(out - 1)] +=
            staged[static_cast<size_t>(i)].value;
      } else {
        csr.col_ind[static_cast<size_t>(out)] =
            staged[static_cast<size_t>(i)].col;
        csr.values[static_cast<size_t>(out)] =
            staged[static_cast<size_t>(i)].value;
        ++out;
      }
    }
    csr.row_ptr[static_cast<size_t>(row + 1)] = out;
  }

  csr.nnz = out;
  csr.col_ind.resize(static_cast<size_t>(out));
  csr.values.resize(static_cast<size_t>(out));
  csr.symmetry = lower_only ? SymmetryStorage::Lower : SymmetryStorage::Full;

  return csr;
}

std::vector<double>
SparseMatrixBuilder::CsrData::multiply(std::span<const double> x) const {
  if (static_cast<int>(x.size()) != n)
    throw std::invalid_argument("CSR matvec size mismatch");

  std::vector<double> y(static_cast<size_t>(n), 0.0);
  if (stores_lower_triangle_only()) {
    for (int row = 0; row < n; ++row) {
      for (int idx = row_ptr[static_cast<size_t>(row)];
           idx < row_ptr[static_cast<size_t>(row + 1)]; ++idx) {
        const int col = col_ind[static_cast<size_t>(idx)];
        const double value = values[static_cast<size_t>(idx)];
        y[static_cast<size_t>(row)] += value * x[static_cast<size_t>(col)];
        if (col != row)
          y[static_cast<size_t>(col)] += value * x[static_cast<size_t>(row)];
      }
    }
    return y;
  }

  for (int row = 0; row < n; ++row) {
    for (int idx = row_ptr[static_cast<size_t>(row)];
         idx < row_ptr[static_cast<size_t>(row + 1)]; ++idx) {
      y[static_cast<size_t>(row)] +=
          values[static_cast<size_t>(idx)] *
          x[static_cast<size_t>(col_ind[static_cast<size_t>(idx)])];
    }
  }
  return y;
}

SparseMatrixBuilder::CsrData SparseMatrixBuilder::CsrData::expanded_symmetric() const {
  if (!stores_lower_triangle_only())
    return *this;

  CsrData full;
  full.n = n;
  full.symmetry = SymmetryStorage::Full;

  int offdiag = 0;
  for (int row = 0; row < n; ++row) {
    for (int idx = row_ptr[static_cast<size_t>(row)];
         idx < row_ptr[static_cast<size_t>(row + 1)]; ++idx) {
      if (col_ind[static_cast<size_t>(idx)] != row)
        ++offdiag;
    }
  }

  full.nnz = nnz + offdiag;
  full.row_ptr.assign(static_cast<size_t>(n + 1), 0);
  for (int row = 0; row < n; ++row) {
    for (int idx = row_ptr[static_cast<size_t>(row)];
         idx < row_ptr[static_cast<size_t>(row + 1)]; ++idx) {
      const int col = col_ind[static_cast<size_t>(idx)];
      ++full.row_ptr[static_cast<size_t>(row + 1)];
      if (col != row)
        ++full.row_ptr[static_cast<size_t>(col + 1)];
    }
  }
  std::partial_sum(full.row_ptr.begin(), full.row_ptr.end(), full.row_ptr.begin());

  full.col_ind.resize(static_cast<size_t>(full.nnz));
  full.values.resize(static_cast<size_t>(full.nnz));
  std::vector<int> cursor(full.row_ptr.begin(), full.row_ptr.begin() + n);
  for (int row = 0; row < n; ++row) {
    for (int idx = row_ptr[static_cast<size_t>(row)];
         idx < row_ptr[static_cast<size_t>(row + 1)]; ++idx) {
      const int col = col_ind[static_cast<size_t>(idx)];
      const double value = values[static_cast<size_t>(idx)];

      int out = cursor[static_cast<size_t>(row)]++;
      full.col_ind[static_cast<size_t>(out)] = col;
      full.values[static_cast<size_t>(out)] = value;

      if (col != row) {
        out = cursor[static_cast<size_t>(col)]++;
        full.col_ind[static_cast<size_t>(out)] = row;
        full.values[static_cast<size_t>(out)] = value;
      }
    }
  }

  return full;
}

} // namespace vibestran
