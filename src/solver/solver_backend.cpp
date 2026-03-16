// src/solver/solver_backend.cpp
#include "solver/solver_backend.hpp"
#include "core/types.hpp"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <format>

namespace nastran {

std::vector<double>
EigenSolverBackend::solve(const SparseMatrixBuilder::CsrData &K_csr,
                          const std::vector<double> &F) {
  const int n = K_csr.n;
  if (n == 0)
    throw SolverError("Stiffness matrix is empty — no free DOFs");
  if (static_cast<int>(F.size()) != n)
    throw SolverError(
        std::format("Force vector size {} != matrix size {}", F.size(), n));

  // Map CsrData arrays directly into an Eigen RowMajor sparse matrix.
  // No copies: Eigen::Map references the existing vectors in-place.
  using ESMR = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  Eigen::Map<const ESMR> K_row(
      n, n, K_csr.nnz,
      K_csr.row_ptr.data(),
      K_csr.col_ind.data(),
      K_csr.values.data());

  // SimplicialLLT/LDLT expect ColMajor; convert with a single structural copy.
  using ESM = Eigen::SparseMatrix<double>;
  ESM K(K_row);

  // Map force vector
  Eigen::Map<const Eigen::VectorXd> F_eigen(F.data(), n);

  // Solve with SimplicialLLT (sparse Cholesky, positive-definite symmetric)
  Eigen::SimplicialLLT<ESM> solver;
  solver.compute(K);

  if (solver.info() != Eigen::Success) {
    // Fall back to LDLT which is more robust for near-singular systems
    Eigen::SimplicialLDLT<ESM> ldlt;
    ldlt.compute(K);
    if (ldlt.info() != Eigen::Success)
      throw SolverError(
          "Stiffness matrix factorization failed — check boundary conditions");
    Eigen::VectorXd u = ldlt.solve(F_eigen);
    if (ldlt.info() != Eigen::Success)
      throw SolverError("Back-substitution failed");
    return std::vector<double>(u.data(), u.data() + n);
  }

  Eigen::VectorXd u = solver.solve(F_eigen);
  if (solver.info() != Eigen::Success)
    throw SolverError("Back-substitution failed");

  return std::vector<double>(u.data(), u.data() + n);
}

} // namespace nastran
