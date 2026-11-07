// src/io/system_dump.cpp
// See include/io/system_dump.hpp.

#include "io/system_dump.hpp"
#include "core/exceptions.hpp"
#include <array>
#include <cstdio>
#include <format>
#include <memory>
#include <vector>

namespace vibestran {

namespace {

struct FileCloser {
  void operator()(std::FILE *f) const noexcept {
    if (f != nullptr)
      std::fclose(f);
  }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

File open_for_writing(const std::filesystem::path &path) {
  File f(std::fopen(path.string().c_str(), "w"));
  if (!f)
    throw SolverError(std::format("--dump-system: cannot open '{}' for writing", path.string()));
  return f;
}

void check_written(std::FILE *f, const std::filesystem::path &path) {
  if (std::ferror(f) != 0)
    throw SolverError(std::format("--dump-system: error writing '{}'", path.string()));
}

} // namespace

void write_matrix_market(const std::filesystem::path &path,
                         const SparseMatrixBuilder::CsrData &A) {
  File f = open_for_writing(path);
  const bool sym = A.stores_lower_triangle_only();
  std::fprintf(f.get(), "%%%%MatrixMarket matrix coordinate real %s\n",
               sym ? "symmetric" : "general");
  std::fprintf(f.get(), "%% written by vibestran --dump-system\n");
  std::fprintf(f.get(), "%d %d %d\n", A.n, A.n, A.nnz);
  for (int i = 0; i < A.n; ++i)
    for (int p = A.row_ptr[static_cast<size_t>(i)]; p < A.row_ptr[static_cast<size_t>(i) + 1];
         ++p)
      std::fprintf(f.get(), "%d %d %.17g\n", i + 1, A.col_ind[static_cast<size_t>(p)] + 1,
                   A.values[static_cast<size_t>(p)]);
  check_written(f.get(), path);
}

void write_matrix_market(const std::filesystem::path &path, std::span<const double> v) {
  File f = open_for_writing(path);
  std::fprintf(f.get(), "%%%%MatrixMarket matrix array real general\n");
  std::fprintf(f.get(), "%% written by vibestran --dump-system\n");
  std::fprintf(f.get(), "%zu 1\n", v.size());
  for (double x : v)
    std::fprintf(f.get(), "%.17g\n", x);
  check_written(f.get(), path);
}

std::vector<EquationDof> reduced_equation_dofs(const Model &model, const MpcHandler &mpc_handler) {
  std::vector<EquationDof> dofs;
  const DofMap &full = mpc_handler.full_dof_map();
  for (const auto &[id, g] : model.nodes)
    for (int c = 0; c < 6; ++c) {
      const EqIndex red = mpc_handler.reduced_index(full.eq_index(id, c));
      if (red == CONSTRAINED_DOF)
        continue;
      const auto eq = static_cast<std::size_t>(red);
      if (eq >= dofs.size())
        dofs.resize(eq + 1);
      dofs[eq] = {id.value, c + 1};
    }
  return dofs;
}

std::string equation_dof_name(const EquationDof &dof) {
  static constexpr std::array<const char *, 6> kComponents{"T1", "T2", "T3", "R1", "R2", "R3"};
  return std::format("grid {} {}", dof.grid, kComponents.at(static_cast<std::size_t>(dof.component - 1)));
}

void write_dof_map(const std::filesystem::path &path, const Model &model,
                   const MpcHandler &mpc_handler) {
  const std::vector<EquationDof> dofs = reduced_equation_dofs(model, mpc_handler);
  File f = open_for_writing(path);
  std::fprintf(f.get(), "# equation grid component (1-6 = T1 T2 T3 R1 R2 R3)\n");
  for (std::size_t i = 0; i < dofs.size(); ++i)
    std::fprintf(f.get(), "%zu %d %d\n", i + 1, dofs[i].grid, dofs[i].component);
  check_written(f.get(), path);
}

std::filesystem::path system_dump_path(const std::string &prefix, int subcase,
                                       const std::string &name) {
  return std::filesystem::path(std::format("{}_sc{}_{}", prefix, subcase, name));
}

} // namespace vibestran
