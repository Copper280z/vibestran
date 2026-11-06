// src/core/sparse_matrix.cpp
#include "core/sparse_matrix.hpp"
#include "core/dof_map.hpp"
#include <algorithm>
#include <format>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
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

  // Order is kept: this builder's triplets, then the other's.
  if (!triplets_.empty())
    merged_.push_back(std::exchange(triplets_, {}));
  std::ranges::move(other.merged_, std::back_inserter(merged_));
  other.merged_.clear();
  if (!other.triplets_.empty())
    merged_.push_back(std::exchange(other.triplets_, {}));
}

namespace {

template <class Body> void for_each_index(size_t n, const Body &body) {
#ifdef HAVE_TBB
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i != r.end(); ++i)
      body(i);
  });
#else
  for (size_t i = 0; i < n; ++i)
    body(i);
#endif
}

} // namespace

SparseMatrixBuilder::CsrData SparseMatrixBuilder::build_csr() {
  // A stable counting sort by row. Contiguous groups of triplets each count their
  // own rows, which tells each group where in a row its triplets go, so they scatter
  // at the same time and still arrive in insertion order, however many groups there
  // are: the sums below, and so K, do not depend on the thread count.
  if (!triplets_.empty())
    merged_.push_back(std::exchange(triplets_, {}));
  const size_t total = std::accumulate(merged_.begin(), merged_.end(), size_t(0),
                                       [](size_t sum, const auto &part) { return sum + part.size(); });

  constexpr size_t kTripletsForThreads = size_t(1) << 16;
  size_t group_count = 1;
#ifdef HAVE_TBB
  if (total >= kTripletsForThreads)
    group_count = std::clamp<size_t>(static_cast<size_t>(tbb::this_task_arena::max_concurrency()), 1,
                                     std::max<size_t>(merged_.size(), 1));
#endif
  std::vector<size_t> group_parts{0};
  for (size_t g = 1, seen = 0, p = 0; g <= group_count; ++g) {
    while (p < merged_.size() && (g == group_count || seen * group_count < total * g))
      seen += merged_[p++].size();
    group_parts.push_back(p);
  }

  const auto n = static_cast<size_t>(size_);
  // Positions are size_t: duplicates are not summed yet, so there can be more
  // triplets than an int counts, on a matrix whose nonzeros still fit one.
  std::vector<std::vector<size_t>> cursor(group_count);
  std::vector<char> has_upper(group_count, 0);
  for_each_index(group_count, [&](size_t g) {
    cursor[g].assign(n, 0);
    for (size_t p = group_parts[g]; p < group_parts[g + 1]; ++p)
      for (const Triplet &t : merged_[p]) {
        ++cursor[g][static_cast<size_t>(t.row)];
        if (t.row < t.col)
          has_upper[g] = 1;
      }
  });
  std::vector<size_t> row_start(n + 1, 0);
  for_each_index(n, [&](size_t row) {
    size_t count = 0;
    for (size_t g = 0; g < group_count; ++g)
      count += cursor[g][row];
    row_start[row + 1] = count;
  });
  std::partial_sum(row_start.begin(), row_start.end(), row_start.begin());
  for_each_index(n, [&](size_t row) {
    size_t at = row_start[row];
    for (size_t g = 0; g < group_count; ++g)
      at += std::exchange(cursor[g][row], at);
  });
  // Not zeroed: the scatter touches every entry, in parallel.
  const auto staged = std::make_unique_for_overwrite<Triplet[]>(total);
  for_each_index(group_count, [&](size_t g) {
    for (size_t p = group_parts[g]; p < group_parts[g + 1]; ++p) {
      for (const Triplet &t : merged_[p])
        staged[cursor[g][static_cast<size_t>(t.row)]++] = t;
      merged_[p] = {};
    }
    cursor[g] = {};
  });
  merged_.clear();

  // Each row's entries by column, duplicates summed in insertion order. Rows hold a
  // few hundred triplets at most, so sorting them takes little memory bandwidth.
  CsrData csr;
  csr.n = size_;
  csr.row_ptr.assign(n + 1, 0);
  for_each_index(n, [&](size_t row) {
    Triplet *const begin = staged.get() + row_start[row];
    Triplet *const end = staged.get() + row_start[row + 1];
    std::stable_sort(begin, end, [](const Triplet &a, const Triplet &b) { return a.col < b.col; });
    int unique = 0;
    for (const Triplet *t = begin; t != end; ++t)
      if (t == begin || t->col != (t - 1)->col)
        ++unique;
    csr.row_ptr[row + 1] = unique;
  });
  const size_t nnz = std::accumulate(csr.row_ptr.begin(), csr.row_ptr.end(), size_t(0));
  if (nnz > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::length_error(std::format("SparseMatrixBuilder: a matrix of {} rows has {} nonzeros, more than CSR's int indices can hold",
                                        n, nnz));
  std::partial_sum(csr.row_ptr.begin(), csr.row_ptr.end(), csr.row_ptr.begin());
  csr.nnz = static_cast<int>(nnz);
  csr.col_ind.resize(nnz);
  csr.values.resize(nnz);
  for_each_index(n, [&](size_t row) {
    auto out = static_cast<size_t>(csr.row_ptr[row]);
    const Triplet *const begin = staged.get() + row_start[row];
    const Triplet *const end = staged.get() + row_start[row + 1];
    for (const Triplet *t = begin; t != end; ++t) {
      if (t != begin && t->col == (t - 1)->col) {
        csr.values[out - 1] += t->value;
      } else {
        csr.col_ind[out] = t->col;
        csr.values[out] = t->value;
        ++out;
      }
    }
  });
  const bool lower_only = std::ranges::none_of(has_upper, [](char upper) { return upper != 0; });
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

SparseMatrixBuilder::CsrData SparseMatrixBuilder::CsrData::lower_triangle() const {
  if (stores_lower_triangle_only())
    return *this;

  CsrData lower;
  lower.n = n;
  lower.symmetry = SymmetryStorage::Lower;
  lower.row_ptr.assign(static_cast<size_t>(n + 1), 0);

  int lower_nnz = 0;
  for (int row = 0; row < n; ++row) {
    for (int idx = row_ptr[static_cast<size_t>(row)];
         idx < row_ptr[static_cast<size_t>(row + 1)]; ++idx) {
      if (col_ind[static_cast<size_t>(idx)] <= row)
        ++lower_nnz;
    }
    lower.row_ptr[static_cast<size_t>(row + 1)] = lower_nnz;
  }

  lower.nnz = lower_nnz;
  lower.col_ind.resize(static_cast<size_t>(lower_nnz));
  lower.values.resize(static_cast<size_t>(lower_nnz));

  int out = 0;
  for (int row = 0; row < n; ++row) {
    for (int idx = row_ptr[static_cast<size_t>(row)];
         idx < row_ptr[static_cast<size_t>(row + 1)]; ++idx) {
      const int col = col_ind[static_cast<size_t>(idx)];
      if (col > row)
        continue;
      lower.col_ind[static_cast<size_t>(out)] = col;
      lower.values[static_cast<size_t>(out)] = values[static_cast<size_t>(idx)];
      ++out;
    }
  }

  return lower;
}

} // namespace vibestran
