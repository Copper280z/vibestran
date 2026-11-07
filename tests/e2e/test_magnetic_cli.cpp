// Analytical end-to-end checks of the SOL MAGSTAT command line. Decks come
// from the standard-library Python generators in tools/, the vibestran
// executable solves them, and the HDF5 results are read through the C API.
// No external reference solver or third-party Python package is used.
//
// Meson sets VIBESTRAN (the executable), VIBESTRAN_PYTHON and VIBESTRAN_TOOLS.
#include "../common/hdf5_results.hpp"
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <numbers>
#include <set>
#include <sstream>

using namespace vibestran::test;
namespace fs = std::filesystem;

namespace {
constexpr double mu0 = 4e-7 * std::numbers::pi;

std::string quoted(const fs::path &path) { return "'" + path.string() + "'"; }

std::string read(const fs::path &path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

std::string replace_all(std::string text, const std::string &from,
                        const std::string &to) {
  for (size_t at = text.find(from); at != std::string::npos;
       at = text.find(from, at + to.size()))
    text.replace(at, from.size(), to);
  return text;
}

std::set<int> integers(const Table &t) {
  return {t.values.begin(), t.values.end()};
}

class MagneticCli : public ::testing::Test {
protected:
  void SetUp() override {
    for (const char *name : {"VIBESTRAN", "VIBESTRAN_PYTHON", "VIBESTRAN_TOOLS"})
      if (!std::getenv(name))
        FAIL() << name << " is not set; run this test through meson test";
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    directory_ = fs::temp_directory_path() / "vibestran_magnetic_cli" /
                 info->name();
    fs::remove_all(directory_);
    fs::create_directories(directory_);
  }
  void TearDown() override {
    if (!HasFailure())
      fs::remove_all(directory_);
  }
  // Deck text from tools/<script> with the given generator arguments.
  std::string generate(const std::string &script, const std::string &args) {
    const auto deck = directory_ / "generated.bdf";
    const auto command = quoted(std::getenv("VIBESTRAN_PYTHON")) + " " +
                         quoted(fs::path(std::getenv("VIBESTRAN_TOOLS")) /
                                script) +
                         " " + args + " " + quoted(deck) + " > " +
                         quoted(directory_ / "generator.log") + " 2>&1";
    EXPECT_EQ(std::system(command.c_str()), 0)
        << command << "\n"
        << read(directory_ / "generator.log");
    return read(deck);
  }
  // Writes the deck, runs the solver with extra command-line options and
  // returns the result file. The log is kept in `log` so a failing case can
  // show it.
  fs::path solve(const std::string &name, const std::string &text,
                 bool expect_success = true, const std::string &options = "") {
    const auto deck = directory_ / (name + ".bdf");
    std::ofstream(deck) << text;
    const auto log_path = directory_ / (name + ".log");
    const auto command = quoted(std::getenv("VIBESTRAN")) + " " + options +
                         (options.empty() ? "" : " ") + quoted(deck) + " > " +
                         quoted(log_path) + " 2>&1";
    const int status = std::system(command.c_str());
    exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    log = read(log_path);
    if (expect_success)
      EXPECT_EQ(status, 0) << log;
    else
      EXPECT_NE(status, 0) << log;
    return directory_ / (name + ".mag.vtkhdf");
  }
  std::string log;
  int exit_code{-1};
  [[nodiscard]] const fs::path &directory() const { return directory_; }

private:
  fs::path directory_;
};

// Every row of an [n, 3] dataset equals `expected` within `tolerance`.
void expect_rows(const Table &t, const std::vector<double> &expected,
                 double tolerance, double relative = 0) {
  ASSERT_EQ(t.dims.size(), 2U);
  ASSERT_GT(t.rows(), 0U);
  for (size_t i = 0; i < t.rows(); ++i)
    for (size_t k = 0; k < 3; ++k)
      EXPECT_NEAR(t.at(i, k), expected[k],
                  tolerance + relative * std::abs(expected[k]))
          << "row " << i << " component " << k;
}
} // namespace

TEST_F(MagneticCli, UniformFieldEnergyFieldsAndProbes) {
  const auto file =
      solve("uniform", generate("magnetic_examples.py", "uniform --n 3 --extent 2"));
  const double energy = 64 * (.01 + .04 + .09) / (2 * mu0);
  EXPECT_NEAR(h5_attribute(file, "/MAGSTAT/subcases/1", "energy_J"), energy,
              1e-12 * energy);
  expect_rows(h5_table(file, "/VTKHDF/SUBCASE_1/CellData/B_T"), {.1, .2, .3},
              1e-12);
  const auto probes = h5_table(file, "/MAGSTAT/subcases/1/probes/B_T");
  ASSERT_EQ(probes.rows(), 2U);
  for (size_t i = 0; i < probes.rows(); ++i)
    EXPECT_NEAR(probes.at(i, 2), .3, 1e-12);

  // The F06 opens with the same header as the structural solutions, its
  // sections in order, and the run's input deck.
  const std::string f06 = read(fs::path(file).replace_extension("").replace_extension(".F06"));
  size_t at = 0;
  for (const char *line : {" VIBESTRAN Version ", "S O L   M A G S T A T", "\n RUN\n",
                           "   Input .................... ", "\n BUILD\n", "\n SOLUTION\n",
                           "   Formulation .............. first-order edge elements",
                           "\n SOLVER\n", "   Backend .................. ",
                           "   Indefinite fallback ...... none, positive definite required",
                           "   Tangent reuse ............ ", "\nSUBCASE 1\n"}) {
    const size_t next = f06.find(line, at);
    ASSERT_NE(next, std::string::npos) << "'" << line << "' after offset " << at << "\n" << f06;
    at = next;
  }
  EXPECT_NE(f06.find((fs::path(file).parent_path() / "uniform.bdf").string()), std::string::npos);
}

TEST_F(MagneticCli, TransientReportsSettledBrakingAndDecimatesOnlyFields) {
  const std::string group = "/MAGTRANS/subcases/1";
  const auto deck = generate("magnetic_planar_examples.py", "transient");
  const auto file = solve("transient", deck);
  ASSERT_EQ(exit_code, 0) << log;
  const auto time = h5_table(file, group + "/time_s");
  const auto angle = h5_table(file, group + "/angle_deg");
  const auto loss = h5_table(file, group + "/eddy_loss_W");
  const auto torque = h5_table(file, group + "/torque_Nm");
  const auto regional = h5_table(file, group + "/property_loss_W");
  ASSERT_GT(time.rows(), 80U);
  ASSERT_EQ(angle.rows(), time.rows());
  ASSERT_EQ(loss.rows(), time.rows());
  EXPECT_EQ(h5_attribute(file, group, "converged"), 1.);
  EXPECT_EQ(h5_attribute(file, group, "speed_rpm"), -600.);
  EXPECT_NEAR(angle.at(46), -360., 1e-12);
  EXPECT_NEAR(time.at(46), .1, 1e-12);
  EXPECT_EQ(regional.values, loss.values);
  EXPECT_GT(h5_attribute(file, group, "mean_eddy_loss_W"), 0.);
  EXPECT_GT(h5_attribute(file, group, "mean_torque_Nm"), 0.);
  const std::string block = "/VTKHDF/SUBCASE_1";
  const auto fields = h5_table(file, block + "/Steps/Values");
  ASSERT_GT(fields.rows(), 1U);
  EXPECT_EQ(fields.at(1), time.at(5));
  EXPECT_TRUE(h5_exists(file, block + "/CellData/J_eddy_A_per_m2"));
  EXPECT_TRUE(h5_exists(file, block + "/Steps/CellDataOffsets/eddy_loss_W_per_m3"));

  const auto scalar_only = solve("scalar", replace_all(deck, "BFIELD(EVERY=5) = ALL",
                                                       "BFIELD = NONE"));
  EXPECT_EQ(h5_table(scalar_only, group + "/torque_Nm").values, torque.values);
  EXPECT_EQ(h5_table(scalar_only, group + "/eddy_loss_W").values, loss.values);
  EXPECT_FALSE(h5_exists(scalar_only, block));
}

TEST_F(MagneticCli, TransientRejectsStaticMotionControlsAndReportsUnsettledRuns) {
  const auto deck = generate("magnetic_planar_examples.py", "transient");
  solve("invalid", replace_all(deck, "ROTATE,0.", "ROTATE,0.,1.,2"), false);
  EXPECT_NE(exit_code, 0);
  EXPECT_NE(log.find("blank STEP and COUNT"), std::string::npos);

  // High conductivity makes the initial equilibrium a poor approximation to
  // the settled state. Leave only two revolutions for the currents to decay.
  auto short_run = replace_all(deck, "TSTEP,1,.0022,5", "TSTEP,1,.0022,2");
  short_run = replace_all(short_run, "MATCOND,3,6.E5", "MATCOND,3,6.E10");
  const auto file = solve("unsettled", short_run, false);
  ASSERT_EQ(exit_code, 5) << log;
  EXPECT_EQ(h5_attribute(file, "/MAGTRANS/subcases/1", "converged"), 0.);
  EXPECT_GT(h5_table(file, "/MAGTRANS/subcases/1/torque_Nm").rows(), 1U);
  EXPECT_NE(read(directory() / "unsettled.F06").find("periodic steady state"),
            std::string::npos);
}

TEST_F(MagneticCli, HarmonicSolidElectrodesWriteComplexCurrentAndMeanLoss) {
  const auto file = solve("terminals", R"(SOL MAGHARM
CEND
FREQUENCY = 60
VOLTAGE = 1
BEGIN BULK
GRID,1,,0,0,0
GRID,2,,1,0,0
GRID,3,,1,1,0
GRID,4,,0,1,0
GRID,5,,0,0,1
GRID,6,,1,0,1
GRID,7,,1,1,1
GRID,8,,0,1,1
CHEXA,1,1,1,2,3,4,5,6
,7,8
PSOLID,1,1
MATMAG,1,1
MATCOND,1,5
FSET,1,1,1,4,8,5
FSET,2,1,2,3,7,6
MTERM,1,1
MTERM,2,2
VTERM,1,1,1,2
VTERM,1,2,0,0
ENDDATA
)");
  ASSERT_EQ(exit_code, 0) << log;
  const std::string group = "/MAGHARM/subcases/1";
  EXPECT_NEAR(h5_attribute(file, group, "mean_eddy_loss_W"), 12.5, 1e-11);
  EXPECT_NEAR(h5_attribute(file, group, "power_residual_W"), 0., 1e-11);
  const auto terminal = h5_table(file, group + "/terminal_current_A");
  ASSERT_EQ(terminal.dims, (std::vector<hsize_t>{2, 2}));
  EXPECT_NEAR(terminal.at(0, 0), 5., 1e-11);
  EXPECT_NEAR(terminal.at(0, 1), 10., 1e-11);
  EXPECT_NEAR(terminal.at(0, 0) + terminal.at(1, 0), 0., 1e-11);
  EXPECT_NEAR(terminal.at(0, 1) + terminal.at(1, 1), 0., 1e-11);
  const std::string fields = "/VTKHDF/SUBCASE_1/CellData";
  expect_rows(h5_table(file, fields + "/J_eddy_real_A_per_m2"), {5., 0., 0.}, 1e-11);
  expect_rows(h5_table(file, fields + "/J_eddy_imag_A_per_m2"), {10., 0., 0.}, 1e-11);
}

TEST_F(MagneticCli, NativeAndWarpedSolidsReproduceUniformFields) {
  const double energy = 64 * (.01 + .04 + .09) / (2 * mu0);
  for (const auto &[element, vtk_type] :
       {std::pair{std::string("hex"), 12}, std::pair{std::string("wedge"), 13}}) {
    SCOPED_TRACE(element);
    const auto text = generate("magnetic_examples.py",
                               "uniform --n 3 --extent 2 --element " + element);
    auto file = solve("solid_" + element, text);
    EXPECT_NEAR(h5_attribute(file, "/MAGSTAT/subcases/1", "energy_J"), energy,
                1e-12 * energy);
    EXPECT_EQ(integers(h5_table(file, "/VTKHDF/SUBCASE_1/Types")),
              std::set<int>{vtk_type});
    // Volume-preserving bilinear shear: warped native solids must reproduce
    // an exactly representable uniform field and energy.
    const std::vector<double> field =
        element == "hex" ? std::vector<double>{0, 0, .3}
                         : std::vector<double>{.3, 0, 0};
    std::stringstream lines(text);
    std::string warped;
    int moved = 0;
    for (std::string line; std::getline(lines, line);) {
      if (line.starts_with("GRID,")) {
        std::vector<std::string> parts;
        std::stringstream fields(line);
        for (std::string part; std::getline(fields, part, ',');)
          parts.push_back(part);
        const double x = std::stod(parts[3]), y = std::stod(parts[4]),
                     z = std::stod(parts[5]);
        const std::vector<double> p =
            element == "hex" ? std::vector<double>{x, y, z + .2 * x * y}
                             : std::vector<double>{x + .2 * y * z, y, z};
        line = std::format("{},{},{},{:.16g},{:.16g},{:.16g}", parts[0],
                           parts[1], parts[2], p[0], p[1], p[2]);
        moved += x * y != 0 || y * z != 0;
      } else if (line.starts_with("MBC,")) {
        line = std::format("MBC,1,0,UNIFORM,{},{},{}", field[0], field[1],
                           field[2]);
      }
      warped += line + "\n";
    }
    // Without moved grids the check below would pass on the unwarped mesh.
    ASSERT_GT(moved, 0);
    file = solve("warped_" + element, warped);
    const double expected = 64 * .09 / (2 * mu0);
    EXPECT_NEAR(h5_attribute(file, "/MAGSTAT/subcases/1", "energy_J"),
                expected, 1e-12 * expected);
    expect_rows(h5_table(file, "/VTKHDF/SUBCASE_1/CellData/B_T"), field,
                1e-12);
    EXPECT_EQ(integers(h5_table(file, "/VTKHDF/SUBCASE_1/Types")),
              std::set<int>{vtk_type});
  }
}

TEST_F(MagneticCli, PlanarElementsCarryDimensionDepthAndPotential) {
  for (const auto &[element, types] :
       {std::pair{std::string("tri"), std::set<int>{5}},
        std::pair{std::string("quad"), std::set<int>{9}},
        std::pair{std::string("mixed"), std::set<int>{5, 9}}}) {
    SCOPED_TRACE(element);
    const auto file =
        solve("planar_" + element,
              generate("magnetic_planar_examples.py",
                       "uniform --n 4 --depth .02 --element " + element));
    EXPECT_EQ(h5_attribute(file, "/MAGSTAT", "dimension"), 2);
    EXPECT_EQ(h5_attribute(file, "/MAGSTAT", "depth_m"), .02);
    const double energy = 4 * .02 * .05 / (2 * mu0);
    EXPECT_NEAR(h5_attribute(file, "/MAGSTAT/subcases/1", "energy_J"), energy,
                1e-12 * energy);
    EXPECT_EQ(integers(h5_table(file, "/VTKHDF/SUBCASE_1/Types")), types);
    // The nodal potential of B = (.1, .2) is exact for these elements.
    const auto points = h5_table(file, "/VTKHDF/SUBCASE_1/Points");
    const auto potential =
        h5_table(file, "/VTKHDF/SUBCASE_1/PointData/A_normal_Wb_per_m");
    ASSERT_EQ(potential.rows(), points.rows());
    for (size_t i = 0; i < points.rows(); ++i)
      EXPECT_NEAR(potential.at(i),
                  .1 * (points.at(i, 1) + 1) - .2 * (points.at(i, 0) + 1),
                  1e-12)
          << "point " << i;
  }
}

TEST_F(MagneticCli, NonlinearSteelMatchesTheIntegratedBHCurve) {
  // A uniform field is exact even with nonlinear steel: integrate the BH
  // curve independently and check exported H and the extrapolation flag.
  for (const bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "planar" : "solid");
    auto text = planar ? generate("magnetic_planar_examples.py",
                                  "uniform --n 4 --element quad --depth .02")
                       : generate("magnetic_examples.py",
                                  "uniform --n 3 --extent 2 --element hex");
    text = replace_all(text, "MATMAG,1,1",
                       "MATMAG,1,,1\nTABLEBH,1\n,0,0,.1,100,.2,2000,ENDT");
    const auto file = solve(planar ? "nonlinear2d" : "nonlinear3d", text);
    EXPECT_EQ(h5_attribute(file, "/MAGSTAT/subcases/1", "bh_extrapolated"), 1);
    const std::vector<double> field =
        planar ? std::vector<double>{.1, .2, 0} : std::vector<double>{.1, .2, .3};
    const double magnitude = std::hypot(field[0], field[1], field[2]);
    const double extra = magnitude - .2;
    const double density = 110 + 2000 * extra + extra * extra / (2 * mu0);
    const double energy = (planar ? 4 * .02 : 64) * density;
    EXPECT_NEAR(h5_attribute(file, "/MAGSTAT/subcases/1", "energy_J"), energy,
                1e-11 * energy);
    std::vector<double> h(3);
    std::transform(field.begin(), field.end(), h.begin(), [&](double b) {
      return (2000 + extra / mu0) * b / magnitude;
    });
    expect_rows(h5_table(file, "/VTKHDF/SUBCASE_1/CellData/H_A_per_m"), h,
                1e-9, 1e-11);
  }
}

TEST_F(MagneticCli, BackendOptionSelectsTheMagneticSolver) {
  // A nonlinear uniform field needs several Newton steps. The PCG backend
  // solves each one through the generic factor adapter and must reach the
  // same energy as the default direct solver.
  auto text = generate("magnetic_examples.py", "uniform --n 3 --extent 2");
  text = replace_all(text, "MATMAG,1,1",
                     "MATMAG,1,,1\nTABLEBH,1\n,0,0,.1,100,.2,2000,ENDT");
  const auto direct = solve("direct", text);
  // The direct backend's name depends on the platform (Accelerate, CHOLMOD).
  EXPECT_NE(log.find("Solver backend: "), std::string::npos) << log;
  EXPECT_EQ(log.find("PCG"), std::string::npos) << log;
  const double energy = h5_attribute(direct, "/MAGSTAT/subcases/1", "energy_J");
  const auto iterative = solve("pcg", text, true, "--backend=cpu-pcg");
  EXPECT_NE(log.find("Solver backend: CPU PCG"), std::string::npos) << log;
  EXPECT_NEAR(h5_attribute(iterative, "/MAGSTAT/subcases/1", "energy_J"),
              energy, 1e-9 * energy);
  // Structural solver controls stay unavailable.
  solve("refine", text, false, "--refine");
  EXPECT_NE(log.find("--refine"), std::string::npos) << log;
}

TEST_F(MagneticCli, PlanarWindingCircuitLinkageAndAmpereTurns) {
  const auto file = solve(
      "winding", generate("magnetic_planar_examples.py",
                          "winding --n 12 --element mixed --depth .02"));
  // Each leg alone is unbalanced; circuit 1 joins them.
  EXPECT_EQ(h5_table(file, "/MAGSTAT/sources/closed").values,
            (std::vector<double>{0, 0}));
  EXPECT_EQ(h5_table(file, "/MAGSTAT/circuits/id").values,
            std::vector<double>{1});
  EXPECT_EQ(h5_table(file, "/MAGSTAT/circuits/parallel").values,
            std::vector<double>{1});
  EXPECT_EQ(h5_table(file, "/MAGSTAT/circuits/closed").values,
            std::vector<double>{1});
  EXPECT_EQ(h5_table(file, "/MAGSTAT/circuits/members").values,
            (std::vector<double>{11, 1, 12, 1}));
  EXPECT_EQ(h5_table(file, "/MAGSTAT/circuits/member_offsets").values,
            (std::vector<double>{0, 2}));
  const std::string subcase = "/MAGSTAT/subcases/1";
  const double energy = h5_attribute(file, subcase, "energy_J");
  const double current = h5_table(file, subcase + "/circuits/current_A").at(0);
  const double linkage =
      h5_table(file, subcase + "/circuits/linkage_Wb_turn").at(0);
  // Linear energy and reciprocity identities for the closed circuit.
  EXPECT_NEAR(current * linkage, 2 * energy, 1e-10 * energy);
  EXPECT_NEAR(h5_table(file, subcase + "/circuits/inductance_H").at(0, 0) *
                  current,
              linkage, 1e-10 * std::abs(linkage));
  // A single planar region has no linkage of its own.
  for (double v : h5_table(file, subcase + "/sources/linkage_Wb_turn").values)
    EXPECT_TRUE(std::isnan(v));
  EXPECT_EQ(h5_children(file, "/VTKHDF"),
            (std::vector<std::string>{"Assembly", "SUBCASE_1", "COIL_11",
                                      "COIL_12"}));
  // ParaView's block tree comes from the assembly links.
  EXPECT_TRUE(h5_exists(file, "/VTKHDF/Assembly/fields/SUBCASE_1"));
  EXPECT_TRUE(h5_exists(file, "/VTKHDF/Assembly/sources/COIL_11"));
  EXPECT_TRUE(h5_exists(file, "/VTKHDF/Assembly/sources/COIL_12"));
  // The exported per-ampere densities integrate to +/-100 ampere-turns over
  // the two legs.
  double positive = 0, negative = 0;
  for (const std::string block : {"/VTKHDF/COIL_11", "/VTKHDF/COIL_12"}) {
    const auto density = h5_table(file, block + "/CellData/J_per_ampere_per_m2");
    const auto points = h5_table(file, block + "/Points");
    const auto connectivity = h5_table(file, block + "/Connectivity");
    const auto offsets = h5_table(file, block + "/Offsets");
    for (size_t c = 0; c + 1 < offsets.values.size(); ++c) {
      double area = 0;
      const auto first = static_cast<size_t>(offsets.at(c)),
                 last = static_cast<size_t>(offsets.at(c + 1));
      for (size_t k = first; k < last; ++k) {
        const auto p = static_cast<size_t>(connectivity.at(k));
        const auto q = static_cast<size_t>(
            connectivity.at(k + 1 < last ? k + 1 : first));
        area += points.at(p, 0) * points.at(q, 1) -
                points.at(p, 1) * points.at(q, 0);
      }
      EXPECT_EQ(density.at(c, 0), 0);
      EXPECT_EQ(density.at(c, 1), 0);
      const double amps = std::abs(area) / 2 * density.at(c, 2);
      (amps > 0 ? positive : negative) += amps;
    }
  }
  EXPECT_NEAR(positive, 100, 1e-12 * 100);
  EXPECT_NEAR(negative, -100, 1e-12 * 100);
}

TEST_F(MagneticCli, WireMeshSchemesAndUnsupportedCards) {
  // Deck selection through the CLI: the source topology of each formulation
  // plus the linear energy/linkage identity.
  const auto loop =
      generate("magnetic_examples.py", "loop --n 4 --extent 2");
  for (const auto &[scheme, vtk_type] :
       {std::pair{std::string("AUTO"), 12}, std::pair{std::string("HEX"), 12},
        std::pair{std::string("TET"), 10}}) {
    SCOPED_TRACE(scheme);
    const auto file =
        solve("wire_" + scheme,
              replace_all(loop, "BEGIN BULK",
                          "BEGIN BULK\nPARAM,WIREMESH," + scheme));
    const std::string subcase = "/MAGSTAT/subcases/1";
    const double energy = h5_attribute(file, subcase, "energy_J");
    EXPECT_GT(energy, 0);
    const double current = h5_table(file, subcase + "/sources/current_A").at(0);
    const double linkage =
        h5_table(file, subcase + "/sources/linkage_Wb_turn").at(0);
    EXPECT_NEAR(current * linkage, 2 * energy, 1e-10 * energy);
    const auto blocks = h5_children(file, "/VTKHDF");
    std::vector<std::string> wires;
    std::copy_if(blocks.begin(), blocks.end(), std::back_inserter(wires),
                 [](const std::string &name) { return name.starts_with("WIRE_"); });
    ASSERT_EQ(wires.size(), 1U);
    EXPECT_EQ(integers(h5_table(file, "/VTKHDF/" + wires[0] + "/Types")),
              std::set<int>{vtk_type});
  }
  solve("unsupported", replace_all(loop, "CWIRE,", "UNSUPPORTED,"), false);
  EXPECT_NE(log.find("unsupported MAGSTAT bulk card"), std::string::npos)
      << log;
}
