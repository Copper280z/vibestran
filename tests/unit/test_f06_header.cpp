// tests/unit/test_f06_header.cpp
// The uniform F06 header: section order and layout, wrapping, notice
// collection, and that a solve carries its backend's details and notices
// into the F06.

#include "io/bdf_parser.hpp"
#include "io/f06_header.hpp"
#include "io/results.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace vibestran;

namespace {

using Row = std::pair<std::string, std::string>;

std::vector<std::string> lines_of(const std::string &text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);)
    lines.push_back(line);
  return lines;
}

std::string header_text(const SolveSummary &summary, const F06RunInfo *run) {
  std::ostringstream out;
  write_f06_header(out, "T E S T   T I T L E", summary, run);
  return out.str();
}

// Index of the line equal to text; fails the test when it is absent or
// repeated.
size_t only_line(const std::vector<std::string> &lines, const std::string &text) {
  const auto count = std::count(lines.begin(), lines.end(), text);
  EXPECT_EQ(count, 1) << "'" << text << "'";
  return static_cast<size_t>(std::find(lines.begin(), lines.end(), text) - lines.begin());
}

// Words of the value of the row starting with label, joined from its first
// line and every continuation line indented to the value column.
std::string row_value(const std::vector<std::string> &lines, const std::string &label,
                      size_t column) {
  auto it = std::find_if(lines.begin(), lines.end(), [&](const std::string &l) {
    return l.rfind(label, 0) == 0;
  });
  EXPECT_NE(it, lines.end()) << label;
  if (it == lines.end())
    return {};
  std::string value = it->substr(column);
  const std::string indent(column, ' ');
  for (++it; it != lines.end() && it->rfind(indent, 0) == 0 && it->size() > column &&
             (*it)[column] != ' ';
       ++it)
    value += " " + it->substr(column);
  return value;
}

SolveSummary sample_summary() {
  SolveSummary s;
  s.solution.add("Solution", "SOL 101 linear static");
  s.solution.add("Subcase 1", "12 equations");
  s.solver.add("Backend", "stub");
  s.solver.add("Precision", "fp64");
  s.notices.push_back("Subcase 1: something worth knowing");
  return s;
}

} // namespace

TEST(F06Header, SectionsAppearOnceInOrderWithValuesInOneColumn) {
  F06RunInfo run;
  run.input = "/decks/model.bdf";
  run.command_line = "vibestran /decks/model.bdf";
  run.started = "2026-10-31 12:00:00";
  run.elapsed_seconds = 1.25;
  run.cpu_threads = 6;
  const auto lines = lines_of(header_text(sample_summary(), &run));

  ASSERT_GE(lines.size(), 4U);
  EXPECT_EQ(lines[0].rfind("1V I B E S T R A N", 0), 0U);
  // The validation parser recognises the file by its third line.
  EXPECT_EQ(lines[2].rfind(" VIBESTRAN Version ", 0), 0U);
  EXPECT_NE(lines[3].find("T E S T   T I T L E"), std::string::npos);

  const size_t run_at = only_line(lines, " RUN"), build_at = only_line(lines, " BUILD"),
               solution_at = only_line(lines, " SOLUTION"),
               solver_at = only_line(lines, " SOLVER"),
               notices_at = only_line(lines, " NOTICES");
  EXPECT_LT(run_at, build_at);
  EXPECT_LT(build_at, solution_at);
  EXPECT_LT(solution_at, solver_at);
  EXPECT_LT(solver_at, notices_at);

  // Dot leaders start every value of every section 30 characters in.
  EXPECT_EQ(only_line(lines, "   Input .................... /decks/model.bdf"), run_at + 1);
  only_line(lines, "   Wall time ................ 1.250 s");
  only_line(lines, "   CPU threads .............. 6");
  EXPECT_EQ(only_line(lines, "   Solution ................. SOL 101 linear static"),
            solution_at + 1);
  EXPECT_EQ(only_line(lines, "   Backend .................. stub"), solver_at + 1);
  EXPECT_EQ(only_line(lines, "   *** WARNING: Subcase 1: something worth knowing"),
            notices_at + 1);
  for (size_t i = build_at + 1; i < solution_at - 1; ++i)
    EXPECT_EQ(lines[i].substr(28, 2), ". ") << lines[i];

  for (const auto &line : lines)
    EXPECT_LE(line.size(), 132U) << line;
}

TEST(F06Header, LongValuesAndNoticesWrapAtWordsUnderTheirColumn) {
  std::string value, notice;
  for (int i = 0; i < 40; ++i) {
    value += (i ? " " : "") + std::format("value{:04}", i);
    notice += (i ? " " : "") + std::format("notice{:04}", i);
  }
  SolveSummary s;
  s.solver.add("Backend", value);
  // A name wider than the dot-leader column keeps one space before its value.
  s.solver.add("An unusually long row name here", "short");
  s.notices.push_back(notice);
  const auto lines = lines_of(header_text(s, nullptr));

  // Rejoining the wrapped lines recovers the text: no word is split or lost.
  EXPECT_EQ(row_value(lines, "   Backend ..", 30), value);
  EXPECT_EQ(row_value(lines, "   *** WARNING: ", 16), notice);
  const auto first = std::find_if(lines.begin(), lines.end(), [](const std::string &l) {
    return l.rfind("   Backend ..", 0) == 0;
  });
  ASSERT_NE(first, lines.end());
  EXPECT_EQ((first + 1)->rfind(std::string(30, ' ') + "value", 0), 0U)
      << "continuation lines start in the value column";
  only_line(lines, "   An unusually long row name here short");
  for (const auto &line : lines)
    EXPECT_LE(line.size(), 132U) << line;
}

TEST(F06Header, RunNoticesAndEmptySectionsAreOmitted) {
  SolveSummary s;
  s.solution.add("Solution", "SOL 103 normal modes");
  const std::string text = header_text(s, nullptr);
  const auto lines = lines_of(text);
  EXPECT_EQ(std::count(lines.begin(), lines.end(), " RUN"), 0);
  EXPECT_EQ(std::count(lines.begin(), lines.end(), " SOLVER"), 0);
  EXPECT_EQ(std::count(lines.begin(), lines.end(), " NOTICES"), 0);
  only_line(lines, " BUILD");
  only_line(lines, " SOLUTION");
  // Nothing in the header may look like a result block to the validation
  // parser (MYSTRAN_Validation/f06_query.py).
  for (const char *block : {"D I S P L A C E M E N T", "E I G E N V E C T O R",
                            "S P C   F O R C E S", "M P C   F O R C E S",
                            "A P P L I E D", "OUTPUT FOR SUBCASE"})
    EXPECT_EQ(text.find(block), std::string::npos) << block;
}

TEST(F06Header, FatalMessagesComeBeforeEverySectionAndWrapUnderTheirText) {
  SolveSummary s;
  s.solution.add("Solution", "SOL 101 linear static");
  s.notices.push_back("a warning");
  std::string long_message;
  for (int i = 0; i < 60; ++i)
    long_message += "word" + std::to_string(i) + " ";
  collect_fatals(s, {long_message + "tail", "second"}, "Subcase 2");
  ASSERT_EQ(s.fatals.size(), 2u);
  EXPECT_EQ(s.fatals[1], "Subcase 2: second");
  const std::string text = header_text(s, nullptr);
  const auto lines = lines_of(text);
  only_line(lines, " FATAL MESSAGES");
  const auto title = text.find("FATAL MESSAGES");
  EXPECT_LT(title, text.find(" BUILD"));
  EXPECT_LT(title, text.find(" SOLUTION"));
  EXPECT_LT(title, text.find(" NOTICES"));
  const size_t first = only_line(lines, "   *** FATAL: Subcase 2: second") ;
  // The long message came first and wrapped under its own text, not under the prefix.
  EXPECT_GT(first, 4u);
  EXPECT_EQ(lines.at(first - 1).find("tail"), lines.at(first - 1).size() - 4);
  EXPECT_EQ(lines.at(first - 1).find_first_not_of(' '), std::string_view("   *** FATAL: ").size());
  // Line 3 stays first for the validation parser, and the title stays under it.
  EXPECT_EQ(lines.at(2).rfind(" VIBESTRAN Version", 0), 0u);
  for (const auto &line : lines)
    EXPECT_LE(line.size(), 132u) << line;
}

TEST(F06Header, CollectNoticesPrefixesContextAndDropsImmediateRepeats) {
  SolveSummary s;
  collect_notices(s, {"a", "a", "b"}, "Subcase 1");
  collect_notices(s, {"b"}, "Subcase 1"); // the same notice from a later step
  collect_notices(s, {"b"}, "Subcase 2");
  collect_notices(s, {"c"}, "");
  const std::vector<std::string> expected{"Subcase 1: a", "Subcase 1: b", "Subcase 2: b",
                                          "c"};
  EXPECT_EQ(s.notices, expected);
}

TEST(F06Header, RowSetReplacesAnExistingRowOrAppends) {
  F06Section s{"SOLVER", {}};
  s.add("Method", "Cholesky");
  s.add("Indefinite fallback", "LDL^T");
  s.set("Indefinite fallback", "none");
  s.set("Tangent reuse", "1 symbolic analysis");
  ASSERT_EQ(s.rows.size(), 3U);
  EXPECT_EQ(s.rows[1], Row("Indefinite fallback", "none"));
  EXPECT_EQ(s.rows[2].first, "Tangent reuse");
}

namespace {

// A direct backend that reports details and records a notice, the same one
// twice, on every solve.
class NoticingBackend final : public SolverBackend {
public:
  std::vector<double> solve(const SparseMatrixBuilder::CsrData &K, const std::vector<double> &F,
                            const FactorRatioCheckPolicy *policy) override {
    notices().add("stub solver notice");
    notices().add("stub solver notice");
    return inner_.solve(K, F, policy);
  }
  std::string_view name() const noexcept override { return "noticing stub"; }
  std::vector<std::pair<std::string, std::string>> details() const override {
    return {{"Method", "stub direct"}, {"Precision", "fp64"}};
  }

private:
  LdltSolverBackend inner_;
};

constexpr const char *kSpringDeck = R"(SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC = 1
  DISPLACEMENT = ALL
SUBCASE 2
  LOAD = 2
  SPC = 1
  DISPLACEMENT = ALL
BEGIN BULK
PARAM,WTMASS,1.
PARAM,POST,-1
GRID,1,,0.,0.,0.
GRID,2,,1.,0.,0.
CELAS2,1,1000.,1,1,2,1
SPC1,1,123456,1
SPC1,1,23456,2
FORCE,1,2,0,10.,1.,0.,0.
FORCE,2,2,0,20.,1.,0.,0.
ENDDATA
)";

} // namespace

TEST(F06Header, LinearStaticCarriesBackendDetailsAndPerSubcaseNoticesIntoTheF06) {
  const Model model = BdfParser::parse_string(kSpringDeck);
  LinearStaticSolver solver(std::make_unique<NoticingBackend>());
  const SolverResults results = solver.solve(model);
  ASSERT_EQ(results.subcases.size(), 2U);

  const auto &solver_rows = results.summary.solver.rows;
  ASSERT_EQ(solver_rows.size(), 3U);
  EXPECT_EQ(solver_rows[0], Row("Backend", "noticing stub"));
  EXPECT_EQ(solver_rows[1], Row("Method", "stub direct"));
  EXPECT_EQ(solver_rows[2], Row("Precision", "fp64"));

  // One reduced equation (x translation of grid 2) in each subcase,
  // and the deck's PARAMs sorted by name.
  const auto &solution_rows = results.summary.solution.rows;
  auto value_of = [&](const std::string &name) {
    for (const auto &[row, value] : solution_rows)
      if (row == name)
        return value;
    ADD_FAILURE() << "no SOLUTION row " << name;
    return std::string();
  };
  EXPECT_EQ(value_of("Subcases"), "2");
  EXPECT_EQ(value_of("Subcase 1").rfind("1 equation;", 0), 0U) << value_of("Subcase 1");
  EXPECT_EQ(value_of("Subcase 2").rfind("1 equation;", 0), 0U) << value_of("Subcase 2");
  const std::string params = value_of("PARAM settings");
  EXPECT_LT(params.find("POST="), params.find("WTMASS=")) << params;

  // The notice is kept once per subcase, under that subcase.
  const std::vector<std::string> expected{"Subcase 1: stub solver notice",
                                          "Subcase 2: stub solver notice"};
  EXPECT_EQ(results.summary.notices, expected);

  std::ostringstream out;
  F06Writer::write(results, model, out);
  const std::string f06 = out.str();
  const auto warning = f06.find("   *** WARNING: Subcase 2: stub solver notice\n");
  ASSERT_NE(warning, std::string::npos);
  EXPECT_LT(f06.find(" NOTICES\n"), warning);
  EXPECT_LT(warning, f06.find("OUTPUT FOR SUBCASE"));
  // The solve itself is unaffected: u = F / k.
  ASSERT_FALSE(results.subcases[1].displacements.empty());
  for (const auto &d : results.subcases[1].displacements)
    if (d.node == NodeId{2})
      EXPECT_NEAR(d.d[0], 0.02, 1e-14);
}
