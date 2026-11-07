// tests/unit/test_system_dump.cpp
// --dump-system output (io/system_dump.hpp): Matrix Market headers, 1-based
// indices, exact round trip of the values, and the DOF map of a solved
// model.

#include <gtest/gtest.h>
#include "io/bdf_parser.hpp"
#include "io/system_dump.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace vibestran;

namespace {

std::vector<std::string> lines(const std::filesystem::path &p) {
  std::ifstream in(p);
  std::vector<std::string> out;
  for (std::string l; std::getline(in, l);)
    out.push_back(l);
  return out;
}

} // namespace

TEST(SystemDump, MatrixMarketFilesRoundTripExactly) {
  const auto dir = std::filesystem::temp_directory_path() / "vibestran_dump_test";
  std::filesystem::create_directories(dir);
  SparseMatrixBuilder::CsrData K;
  K.n = 2;
  K.symmetry = SymmetryStorage::Lower;
  K.row_ptr = {0, 1, 3};
  K.col_ind = {0, 0, 1};
  K.values = {0.1, -1.0 / 3.0, 2e-300};
  K.nnz = 3;
  write_matrix_market(dir / "K.mtx", K);
  const auto k = lines(dir / "K.mtx");
  ASSERT_GE(k.size(), 6u);
  EXPECT_EQ(k[0], "%%MatrixMarket matrix coordinate real symmetric");
  EXPECT_EQ(k[2], "2 2 3");
  std::istringstream e(k[4]);
  int i = 0, j = 0;
  double v = 0.0;
  e >> i >> j >> v;
  EXPECT_EQ(i, 2);
  EXPECT_EQ(j, 1);
  EXPECT_EQ(v, -1.0 / 3.0); // %.17g round-trips every double
  const std::vector<double> u{1e-17, 3.0};
  write_matrix_market(dir / "u.mtx", u);
  const auto uv = lines(dir / "u.mtx");
  ASSERT_EQ(uv.size(), 5u);
  EXPECT_EQ(uv[0], "%%MatrixMarket matrix array real general");
  EXPECT_EQ(uv[2], "2 1");
  EXPECT_EQ(std::stod(uv[3]), 1e-17);
  std::filesystem::remove_all(dir);
}

TEST(SystemDump, StaticSolveWritesTheReducedSystemAndItsDofMap) {
  // A two-grid spring chain: grid 1 fixed, grid 2 free in T1 only, so the
  // reduced system is 1x1 with k = 100, F = 5, u = 0.05.
  const auto dir = std::filesystem::temp_directory_path() / "vibestran_dump_test2";
  std::filesystem::create_directories(dir);
  Model model = BdfParser::parse_string(R"(SOL 101
CEND
SPC = 1
LOAD = 1
BEGIN BULK
GRID,1,,0.,0.,0.
GRID,2,,1.,0.,0.
CELAS2,1,100.,1,1,2,1
SPC1,1,123456,1
SPC1,1,23456,2
FORCE,1,2,,5.,1.,0.,0.
ENDDATA
)");
  LinearStaticSolver solver(std::make_unique<LdltSolverBackend>());
  solver.set_system_dump((dir / "chain").string());
  (void)solver.solve(model);
  const auto k = lines(dir / "chain_sc1_K.mtx");
  ASSERT_GE(k.size(), 4u);
  EXPECT_EQ(k[2], "1 1 1");
  EXPECT_EQ(k[3], "1 1 100");
  EXPECT_EQ(lines(dir / "chain_sc1_F.mtx").at(3), "5");
  EXPECT_DOUBLE_EQ(std::stod(lines(dir / "chain_sc1_u.mtx").at(3)), 0.05);
  const auto dofs = lines(dir / "chain_sc1_dofs.txt");
  ASSERT_EQ(dofs.size(), 2u);
  EXPECT_EQ(dofs[1], "1 2 1"); // equation 1 is grid 2, T1
  std::filesystem::remove_all(dir);
}
