#pragma once

#include "core/mpc_handler.hpp"
#include "core/model.hpp"
#include "core/sparse_matrix.hpp"
#include "elements/element_base.hpp"
#include "elements/element_factory.hpp"
#include <cstddef>
#include <utility>
#include <algorithm>
#include <vector>

#ifdef HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#endif

namespace vibestran::detail {

inline int element_num_dofs(ElementType type) {
  switch (type) {
  case ElementType::CQUAD4:
    return 24;
  case ElementType::CTRIA3:
    return 18;
  case ElementType::CHEXA8:
    return 24;
  case ElementType::CHEXA20:
    return 60;
  case ElementType::CTETRA4:
    return 12;
  case ElementType::CTETRA10:
    return 30;
  case ElementType::CPENTA6:
    return 18;
  case ElementType::CBAR:
  case ElementType::CBEAM:
  case ElementType::CROD:
  case ElementType::CONROD:
  case ElementType::CBUSH:
    return 12;
  case ElementType::CELAS1:
  case ElementType::CELAS2:
  case ElementType::CMASS1:
  case ElementType::CMASS2:
    return 2;
  case ElementType::CONM2:
    return 6;
  case ElementType::CHBDY:
    return 0;
  }
  return 0;
}

// Triplets for one triangle of each element in [first, last). Constraints can add
// more, so a builder may still grow past this.
inline size_t triplet_capacity(const Model& model, size_t first, size_t last) {
  size_t total = 0;
  for (size_t i = first; i < last; ++i) {
    const auto ndof = static_cast<size_t>(element_num_dofs(model.elements[i].type));
    total += ndof * (ndof + 1) / 2;
  }
  return total;
}

inline std::vector<double> to_row_major(const LocalKe& matrix) {
  const int rows = static_cast<int>(matrix.rows());
  const int cols = static_cast<int>(matrix.cols());
  std::vector<double> values(static_cast<size_t>(rows * cols));
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      values[static_cast<size_t>(r * cols + c)] = matrix(r, c);
  return values;
}

// Contiguous chunks of elements. Merging what each chunk produced in chunk order
// gives element order whatever the thread count or scheduling, so sums and lists
// built from it are bitwise reproducible.
inline constexpr size_t kElementsPerChunk = 512;

inline size_t element_chunk_count(size_t n_elems) {
  return (n_elems + kElementsPerChunk - 1) / kElementsPerChunk;
}

/// body(chunk, first element, one past the last) for each chunk of n_elems.
template <typename Body>
inline void for_element_chunks(size_t n_elems, Body&& body) {
  const size_t n_chunks = element_chunk_count(n_elems);
  const auto one = [&](size_t c) {
    body(c, c * kElementsPerChunk, std::min(n_elems, (c + 1) * kElementsPerChunk));
  };
#ifdef HAVE_TBB
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n_chunks, 1),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t c = range.begin(); c != range.end(); ++c)
                        one(c);
                    });
#else
  for (size_t c = 0; c < n_chunks; ++c)
    one(c);
#endif
}

template <typename MatrixFn>
inline void assemble_element_matrix(const Model& model,
                                    const MpcHandler& mpc_handler,
                                    SparseMatrixBuilder& builder,
                                    MatrixFn&& matrix_fn) {
  const DofMap& dof_map = mpc_handler.full_dof_map();

#ifdef HAVE_TBB
  // Each chunk into its own builder: the triplets then arrive in element order,
  // so duplicate entries are summed in the same order every run and K is bitwise
  // reproducible (with thread-local builders it varied in the last bits between
  // runs, which moved the solution of an ill-conditioned model by ~cond(K) eps).
  const size_t n_elems = model.elements.size();
  const size_t n_chunks = element_chunk_count(n_elems);
  std::vector<SparseMatrixBuilder> chunk_builders;
  chunk_builders.reserve(n_chunks);
  for (size_t c = 0; c < n_chunks; ++c)
    chunk_builders.emplace_back(builder.size(), 0);

  for_element_chunks(n_elems, [&](size_t c, size_t first, size_t last) {
    auto& chunk_builder = chunk_builders[c];
    chunk_builder.reserve_triplets(triplet_capacity(model, first, last));
    for (size_t i = first; i < last; ++i) {
      const auto& elem_data = model.elements[i];
      auto elem = make_element(elem_data, model);
      auto gdofs = elem->global_dof_indices(dof_map);
      LocalKe matrix = matrix_fn(*elem);
      auto matrix_row_major = to_row_major(matrix);
      mpc_handler.apply_to_stiffness(gdofs, matrix_row_major, chunk_builder);
    }
  });

  for (auto& chunk_builder : chunk_builders)
    builder.merge_from(std::move(chunk_builder));
#else
  builder.reserve_triplets(triplet_capacity(model, 0, model.elements.size()));
  for (const auto& elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    auto gdofs = elem->global_dof_indices(dof_map);
    LocalKe matrix = matrix_fn(*elem);
    auto matrix_row_major = to_row_major(matrix);
    mpc_handler.apply_to_stiffness(gdofs, matrix_row_major, builder);
  }
#endif
}

} // namespace vibestran::detail
