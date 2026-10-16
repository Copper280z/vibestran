#pragma once
// include/io/system_dump.hpp
// Writes assembled linear systems in Matrix Market format (--dump-system),
// for inspection with SciPy (scipy.io.mmread), MATLAB, Julia, etc.
//
// For a prefix P and subcase N:
//   P_scN_K.mtx  stiffness of the reduced (post-SPC, post-MPC) system;
//                symmetric storage writes the lower triangle as
//                "coordinate real symmetric"
//   P_scN_M.mtx  mass (modal subcases)
//   P_scN_F.mtx  load vector, P_scN_u.mtx solution (static subcases),
//                "array real general"
//   P_scN_dofs.txt  equation (1-based, as in the .mtx files) -> grid and
//                component (1-6 = T1 T2 T3 R1 R2 R3); MPC-dependent and
//                constrained DOFs have no equation

#include "core/model.hpp"
#include "core/mpc_handler.hpp"
#include "core/sparse_matrix.hpp"
#include <filesystem>
#include <span>
#include <string>

namespace vibestran {

void write_matrix_market(const std::filesystem::path &path, const SparseMatrixBuilder::CsrData &A);

void write_matrix_market(const std::filesystem::path &path, std::span<const double> v);

void write_dof_map(const std::filesystem::path &path, const Model &model,
                   const MpcHandler &mpc_handler);

/// P_scN_<name> for the prefix and subcase.
[[nodiscard]] std::filesystem::path system_dump_path(const std::string &prefix, int subcase,
                                                     const std::string &name);

} // namespace vibestran
