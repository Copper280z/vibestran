#pragma once
// src/solver/linear_factor_checks.hpp
// Shape errors shared by every LinearFactor implementation.

#include "core/result.hpp"
#include "core/sparse_matrix.hpp"
#include <cstddef>
#include <format>
#include <span>

namespace vibestran::detail {

[[nodiscard]] inline Status check_matrix(const SparseMatrixBuilder::CsrData &K) {
  if (K.n == 0)
    return Error{"Matrix is empty — no free DOFs"};
  if (K.row_ptr.size() != static_cast<std::size_t>(K.n) + 1 ||
      K.col_ind.size() != static_cast<std::size_t>(K.nnz) ||
      K.values.size() != static_cast<std::size_t>(K.nnz))
    return Error{std::format("Malformed CSR matrix: n={}, nnz={}, row_ptr {}, "
                             "col_ind {}, values {}",
                             K.n, K.nnz, K.row_ptr.size(), K.col_ind.size(),
                             K.values.size())};
  return std::nullopt;
}

[[nodiscard]] inline Status check_rhs(int n, std::span<const double> B, int nrhs) {
  if (nrhs < 1 || B.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs))
    return Error{std::format("Right-hand side has {} values; expected {} x {}",
                             B.size(), n, nrhs)};
  return std::nullopt;
}

} // namespace vibestran::detail
