// tests/integration/test_fatal_messages.cpp
// Results that can't be trusted are still written, under a fatal message at the top of the
// F06: a singular K, a K_ii/D_i past its limit, a solution that does not carry the load, and
// a failed solve. PARAM,BAILOUT,-1 turns what it covers into warnings.

#include "core/exceptions.hpp"
#include "io/bdf_parser.hpp"
#include "io/op2_writer.hpp"
#include "io/results.hpp"
#include "solver/heat_transfer_steady.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

using namespace vibestran;

namespace {

const std::string kHeader = "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nDISPLACEMENT = ALL\nBEGIN BULK\n";

/// A rod along x, clamped in translation at grid 1 and pulled at grid 2. With J and no
/// rotation held, it can spin about its axis: R1 at the two grids is a free torsion spring.
std::string spinning_rod(const std::string &extra) {
    return kHeader + extra +
           "MAT1,1,1.+7,,0.3\nPROD,1,1,2.,1.\nGRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\n"
           "CROD,1,1,1,2\nSPC1,1,123,1\nSPC1,1,23,2\nFORCE,1,2,0,100.,1.,0.,0.\n";
}
const std::string kTorque = "MOMENT,1,2,0,5.,1.,0.,0.\n";

/// Grid 1 hangs from grid 2 by a spring of stiffness s, and grid 2 is grounded by a spring
/// of stiffness 1, so K_ii/D_i is s + 1.
std::string stiff_on_soft(double s, const std::string &extra = "") {
    return kHeader + extra +
           std::format("GRID,1,,0.,0.,0.,,23456\nGRID,2,,1.,0.,0.,,23456\nCELAS2,1,{:.1f},1,1,2,1\n"
                       "CELAS2,2,1.,2,1\nFORCE,1,1,0,3.,1.,0.,0.\nENDDATA\n",
                       s);
}

SolverResults solve(const std::string &deck, std::unique_ptr<SolverBackend> backend = nullptr) {
    Model model = BdfParser::parse_string(deck);
    LinearStaticSolver solver(backend ? std::move(backend) : std::make_unique<LdltSolverBackend>());
    return solver.solve(model);
}

bool any_contains(const std::vector<std::string> &messages, const std::string &text) {
    return std::ranges::any_of(messages, [&](const std::string &m) { return m.find(text) != std::string::npos; });
}

double displacement(const SubCaseResults &r, int grid, int component) {
    const auto it = std::ranges::find(r.displacements, NodeId{grid}, &NodeDisplacement::node);
    if (it == r.displacements.end())
        throw std::runtime_error(std::format("no displacement at grid {}", grid));
    return it->d[component];
}

std::string read(const std::filesystem::path &path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

} // namespace

TEST(FatalMessages, AHealthyModelHasNone) {
    const auto results = solve(stiff_on_soft(10.0));
    EXPECT_TRUE(results.summary.fatals.empty());
    EXPECT_TRUE(results.summary.notices.empty());
    EXPECT_NEAR(displacement(results.subcases.at(0), 1, 0), 3.0 / 10.0 + 3.0, 1e-12);
}

TEST(FatalMessages, AnUnloadedMechanismIsFatalButItsResultsAreRightAndWritten) {
    const auto results = solve(spinning_rod("") + "ENDDATA\n");
    ASSERT_EQ(results.summary.fatals.size(), 1u) << results.summary.fatals[0];
    const std::string &fatal = results.summary.fatals[0];
    EXPECT_NE(fatal.find("Subcase 1"), std::string::npos) << fatal;
    EXPECT_NE(fatal.find("singular"), std::string::npos) << fatal;
    EXPECT_TRUE(fatal.find("grid 1 R1") != std::string::npos || fatal.find("grid 2 R1") != std::string::npos)
        << fatal;
    // The stretch does not depend on the free spin: u = F L / (E A).
    const auto &r = results.subcases.at(0);
    EXPECT_NEAR(displacement(r, 2, 0), 100.0 * 10.0 / (1e7 * 2.0), 1e-12);
    EXPECT_DOUBLE_EQ(displacement(r, 1, 3), 0.0);
    EXPECT_DOUBLE_EQ(displacement(r, 2, 3), 0.0);
}

TEST(FatalMessages, ALoadedMechanismIsAlsoReportedAsNotCarryingTheLoad) {
    const auto results = solve(spinning_rod(kTorque) + "ENDDATA\n");
    EXPECT_TRUE(any_contains(results.summary.fatals, "singular"));
    EXPECT_TRUE(any_contains(results.summary.fatals, "does not equilibrate the load"));
}

TEST(FatalMessages, BailoutTurnsSingularityAndResidualIntoWarnings) {
    const auto results = solve(spinning_rod("PARAM,BAILOUT,-1\n" + kTorque) + "ENDDATA\n");
    EXPECT_TRUE(results.summary.fatals.empty());
    EXPECT_TRUE(any_contains(results.summary.notices, "singular"));
    EXPECT_TRUE(any_contains(results.summary.notices, "does not equilibrate the load"));
}

TEST(FatalMessages, RatioIsAWarningAboveMaxratioAndFatalAboveTheSingularLimit) {
    const auto warned = solve(stiff_on_soft(1e9));
    EXPECT_TRUE(warned.summary.fatals.empty());
    ASSERT_TRUE(any_contains(warned.summary.notices, "K_ii/D_i"));
    EXPECT_TRUE(any_contains(warned.summary.notices, "MAXRATIO"));
    // Which grid carries the ratio depends on the elimination order, but it is one of these.
    EXPECT_TRUE(any_contains(warned.summary.notices, "grid 1 T1") || any_contains(warned.summary.notices, "grid 2 T1"));

    const auto fatal = solve(stiff_on_soft(1e12));
    ASSERT_EQ(fatal.summary.fatals.size(), 1u);
    EXPECT_NE(fatal.summary.fatals[0].find("K_ii/D_i"), std::string::npos) << fatal.summary.fatals[0];
    // Not singular, only past the limit: the answer is still written, and right.
    EXPECT_NEAR(displacement(fatal.subcases.at(0), 2, 0), 3.0, 1e-2);

    const auto bailed = solve(stiff_on_soft(1e12, "PARAM,BAILOUT,-1\n"));
    EXPECT_TRUE(bailed.summary.fatals.empty());
    EXPECT_TRUE(any_contains(bailed.summary.notices, "K_ii/D_i"));
}

TEST(FatalMessages, MaxratioInTheDeckIsWhereTheDeckStops) {
    const auto lower = solve(stiff_on_soft(1e5, "PARAM,MAXRATIO,1.+3\n"));
    EXPECT_TRUE(any_contains(lower.summary.fatals, "K_ii/D_i"));
    const auto higher = solve(stiff_on_soft(1e12, "PARAM,MAXRATIO,1.+13\n"));
    EXPECT_TRUE(higher.summary.fatals.empty());
    EXPECT_FALSE(any_contains(higher.summary.notices, "K_ii/D_i"));
}

TEST(FatalMessages, PcgOnALoadedMechanismIsFatalThoughItsErrorEstimateCannotSeeIt) {
    const auto results = solve(spinning_rod(kTorque) + "ENDDATA\n", std::make_unique<CpuPCGSolverBackend>());
    EXPECT_FALSE(results.summary.fatals.empty());
}

namespace {

/// Throws on its first solve, like a direct solver that stops.
class FailsOnce final : public SolverBackend {
public:
    std::vector<double> solve(const SparseMatrixBuilder::CsrData &K, const std::vector<double> &F,
                              const FactorRatioCheckPolicy *policy) override {
        if (!failed_) {
            failed_ = true;
            throw SolverError("the factorization stopped");
        }
        return inner_.solve(K, F, policy);
    }
    std::string_view name() const noexcept override { return "fails once"; }

private:
    bool failed_{false};
    LdltSolverBackend inner_;
};

} // namespace

TEST(FatalMessages, AFailedSolveEndsItsSubcaseNotTheRunAndTheFilesAreStillWritten) {
    Model model = BdfParser::parse_string(
        "SOL 101\nCEND\nSPC = 1\nDISPLACEMENT = ALL\nSUBCASE 1\n  LOAD = 1\nSUBCASE 2\n  LOAD = 2\nBEGIN BULK\n"
        "GRID,1,,0.,0.,0.,,123456\nGRID,2,,1.,0.,0.,,23456\nCELAS2,1,1000.,1,1,2,1\n"
        "FORCE,1,2,0,10.,1.,0.,0.\nFORCE,2,2,0,20.,1.,0.,0.\nENDDATA\n");
    LinearStaticSolver solver(std::make_unique<FailsOnce>());
    const SolverResults results = solver.solve(model);
    ASSERT_EQ(results.summary.fatals, std::vector<std::string>{"Subcase 1: the factorization stopped"});
    ASSERT_EQ(results.subcases.size(), 2u);
    EXPECT_TRUE(results.subcases[0].displacements.empty());
    EXPECT_NEAR(displacement(results.subcases[1], 2, 0), 0.02, 1e-12);

    const auto dir = std::filesystem::temp_directory_path() / "vibestran_fatal_messages";
    std::filesystem::create_directories(dir);
    const auto f06 = dir / "failed.f06";
    F06Writer::write(results, model, f06);
    Op2Writer::write(results, model, dir / "failed.op2");
    const std::string text = read(f06);
    const auto fatal = text.find("*** FATAL: Subcase 1: the factorization stopped");
    ASSERT_NE(fatal, std::string::npos);
    EXPECT_LT(fatal, text.find(" BUILD"));
    EXPECT_LT(fatal, text.find(" SOLVER"));
    EXPECT_NE(text.find("D I S P L A C E M E N T"), std::string::npos);
    std::filesystem::remove_all(dir);
}

TEST(FatalMessages, HeatLoadOnAConductanceWithNoPrescribedTemperatureIsFatal) {
    const std::string bulk = "BEGIN BULK\nMAT4,1,50.\nPROD,1,1,2.0\nGRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\n"
                             "SLOAD,1,2,100.\nENDDATA\n";
    Model model = BdfParser::parse_string("SOL 153\nCEND\nLOAD = 1\n" + bulk);
    HeatTransferSteadySolver solver(std::make_unique<LdltSolverBackend>());
    const SolverResults results = solver.solve(model);
    EXPECT_TRUE(any_contains(results.summary.fatals, "conductance matrix"));
    EXPECT_TRUE(any_contains(results.summary.fatals, "does not balance the heat load"));
}
