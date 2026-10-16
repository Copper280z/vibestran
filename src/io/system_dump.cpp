// src/io/system_dump.cpp
// See include/io/system_dump.hpp.

#include "io/system_dump.hpp"
#include "core/exceptions.hpp"
#include <algorithm>
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

void write_dof_map(const std::filesystem::path &path, const Model &model,
                   const MpcHandler &mpc_handler) {
  std::vector<NodeId> grids;
  grids.reserve(model.nodes.size());
  for (const auto &[id, g] : model.nodes)
    grids.push_back(id);
  std::sort(grids.begin(), grids.end(),
            [](NodeId a, NodeId b) { return a.value < b.value; });
  struct Row {
    EqIndex eq;
    int grid, comp;
  };
  std::vector<Row> rows;
  const DofMap &full = mpc_handler.full_dof_map();
  for (NodeId g : grids)
    for (int c = 0; c < 6; ++c) {
      const EqIndex red = mpc_handler.reduced_index(full.eq_index(g, c));
      if (red != CONSTRAINED_DOF)
        rows.push_back({red, g.value, c + 1});
    }
  std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.eq < b.eq; });
  File f = open_for_writing(path);
  std::fprintf(f.get(), "# equation grid component (1-6 = T1 T2 T3 R1 R2 R3)\n");
  for (const Row &r : rows)
    std::fprintf(f.get(), "%d %d %d\n", r.eq + 1, r.grid, r.comp);
  check_written(f.get(), path);
}

std::filesystem::path system_dump_path(const std::string &prefix, int subcase,
                                       const std::string &name) {
  return std::filesystem::path(std::format("{}_sc{}_{}", prefix, subcase, name));
}

} // namespace vibestran
