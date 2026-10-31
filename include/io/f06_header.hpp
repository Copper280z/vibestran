#pragma once
// include/io/f06_header.hpp
// The run header every F06 starts with, whatever the solution type:
//
//   banner, version line, solution title
//   RUN       input file, start time, wall time, command line (from main)
//   BUILD     version and revision, build type, compiler, components
//   SOLUTION  what was analysed: model size, settings, per-subcase results
//   SOLVER    how the equations were solved: backend, method, precision
//   NOTICES   warnings about the solve (fallbacks, MAXRATIO, accuracy)
//
// The analysis fills SOLUTION, SOLVER and NOTICES (SolveSummary); main
// supplies RUN (F06RunInfo).

#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vibestran {

class Model;

/// A titled group of "name : value" rows.
struct F06Section {
  std::string title;
  std::vector<std::pair<std::string, std::string>> rows;

  void add(std::string name, std::string value) {
    rows.emplace_back(std::move(name), std::move(value));
  }
  /// Replaces the value of the row called name, or adds the row.
  void set(std::string_view name, std::string value) {
    for (auto &[row, text] : rows)
      if (row == name) {
        text = std::move(value);
        return;
      }
    add(std::string(name), std::move(value));
  }
};

/// What an analysis reports about its run for the header.
struct SolveSummary {
  F06Section solution{"SOLUTION", {}};
  F06Section solver{"SOLVER", {}};
  /// Warnings about the solve, already logged; prefixed with their subcase.
  std::vector<std::string> notices;
};

/// Run information that only the driver knows.
struct F06RunInfo {
  std::filesystem::path input;
  std::string command_line;
  std::string started;          ///< local date and time
  double elapsed_seconds{-1.0}; ///< negative when unknown
  int cpu_threads{0};           ///< Eigen/OpenMP threads; 0 when unknown
};

/// Writes the header. Line 3 starts with " VIBESTRAN Version", which the
/// MYSTRAN validation parser uses to recognise the file. run may be null
/// (unit tests), and then the RUN section is omitted.
void write_f06_header(std::ostream &out, std::string_view title,
                      const SolveSummary &summary, const F06RunInfo *run = nullptr);

/// SOLUTION rows common to the Nastran-deck solutions: the solution, grid
/// points, elements, subcases and the deck's PARAM settings.
[[nodiscard]] F06Section solution_section(const Model &model, std::string_view solution);

/// SOLVER rows: the backend's name, then its details().
[[nodiscard]] F06Section
solver_section(std::string_view backend,
               std::vector<std::pair<std::string, std::string>> details);
/// SOLVER rows for a linear-system or eigensolver backend.
template <typename Backend>
[[nodiscard]] F06Section solver_section(const Backend &backend) {
  return solver_section(backend.name(), backend.details());
}

/// Moves the backend's notices into summary, prefixed with context
/// (e.g. "Subcase 2"), dropping one identical to the previous notice.
void collect_notices(SolveSummary &summary, std::vector<std::string> notices,
                     std::string_view context);

/// Current local date and time, "YYYY-MM-DD HH:MM:SS".
[[nodiscard]] std::string local_time_text();

} // namespace vibestran
