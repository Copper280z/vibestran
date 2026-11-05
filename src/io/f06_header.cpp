// src/io/f06_header.cpp
#include "io/f06_header.hpp"
#include "core/build_info.hpp"
#include "core/model.hpp"
#include <algorithm>
#include <ctime>
#include <format>
#include <ostream>

namespace vibestran {

namespace {

constexpr std::size_t kPageWidth = 132;
constexpr std::size_t kNameWidth = 26; // dot leaders end at this column

/// Splits text into lines of at most width characters at spaces; a word
/// longer than width gets a line of its own.
std::vector<std::string> wrap(std::string_view text, std::size_t width) {
  std::vector<std::string> lines;
  std::string line;
  std::size_t start = 0;
  while (start < text.size()) {
    std::size_t end = text.find(' ', start);
    if (end == std::string_view::npos)
      end = text.size();
    const std::string_view word = text.substr(start, end - start);
    if (!line.empty() && line.size() + 1 + word.size() > width) {
      lines.push_back(std::move(line));
      line.clear();
    }
    if (!line.empty())
      line += ' ';
    line += word;
    start = end + 1;
  }
  if (!line.empty())
    lines.push_back(std::move(line));
  return lines;
}

void write_section(std::ostream &out, const F06Section &section) {
  if (section.rows.empty())
    return;
  out << " " << section.title << "\n";
  for (const auto &[name, value] : section.rows) {
    std::string label = "   " + name + " ";
    if (label.size() < kNameWidth + 3)
      label.append(kNameWidth + 3 - label.size(), '.').append(" ");
    const std::string indent(label.size(), ' ');
    const auto lines = wrap(value, kPageWidth - indent.size());
    out << label << (lines.empty() ? std::string() : lines.front()) << "\n";
    for (std::size_t i = 1; i < lines.size(); ++i)
      out << indent << lines[i] << "\n";
  }
  out << "\n";
}

F06Section build_section() {
  const BuildInfo &b = build_info();
  F06Section s{"BUILD", {}};
  s.add("Version", std::format("{} (revision {})", b.version, b.revision));
  s.add("Build type", std::string(b.build_type));
  s.add("Compiler", std::string(b.compiler));
  s.add("Platform", std::string(b.platform));
  s.add("Components", std::string(b.components));
  return s;
}

std::string centered(std::string_view text) {
  if (text.size() >= kPageWidth)
    return std::string(text);
  return std::string((kPageWidth - text.size()) / 2, ' ') + std::string(text);
}

} // namespace

std::string local_time_text() {
  const std::time_t now = std::time(nullptr);
  char text[32];
  std::strftime(text, sizeof(text), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  return text;
}

void write_f06_header(std::ostream &out, std::string_view title,
                      const SolveSummary &summary, const F06RunInfo *run) {
  const std::time_t now = std::time(nullptr);
  char date[32];
  std::strftime(date, sizeof(date), "%B %e, %Y", std::localtime(&now));
  const std::string left = "V I B E S T R A N";
  const std::size_t width = kPageWidth - 1;
  out << "1" << left
      << std::string(width > left.size() + std::string_view(date).size()
                         ? width - left.size() - std::string_view(date).size()
                         : 1,
                     ' ')
      << date << "\n";
  out << "0" << std::string(width, ' ') << "\n";
  // Line 3 must start with " VIBESTRAN Version" for the validation suite
  // parser to recognise the file (see MYSTRAN_Validation/f06_query.py).
  out << " VIBESTRAN Version " << build_info().version
      << "   F06 MYSTRAN-COMPATIBLE OUTPUT\n";
  out << centered(title) << "\n\n";

  if (run != nullptr) {
    F06Section r{"RUN", {}};
    r.add("Input", run->input.string());
    if (!run->command_line.empty())
      r.add("Command line", run->command_line);
    if (!run->started.empty())
      r.add("Started", run->started);
    if (run->elapsed_seconds >= 0)
      r.add("Wall time", std::format("{:.3f} s", run->elapsed_seconds));
    if (run->cpu_threads > 0)
      r.add("CPU threads", std::to_string(run->cpu_threads));
    write_section(out, r);
  }
  write_section(out, build_section());
  write_section(out, summary.solution);
  write_section(out, summary.solver);
  if (!summary.notices.empty()) {
    out << " NOTICES\n";
    for (const auto &notice : summary.notices) {
      const auto lines = wrap(notice, kPageWidth - 17);
      for (std::size_t i = 0; i < lines.size(); ++i)
        out << (i == 0 ? "   *** WARNING: " : "                ") << lines[i] << "\n";
    }
    out << "\n";
  }
}

F06Section solution_section(const Model &model, std::string_view solution) {
  F06Section s{"SOLUTION", {}};
  s.add("Solution", std::string(solution));
  const std::size_t scalar_points = model.scalar_point_count();
  s.add("Grid points", std::to_string(model.nodes.size() - scalar_points));
  if (scalar_points != 0)
    s.add("Scalar points", std::to_string(scalar_points));
  s.add("Elements", std::to_string(model.elements.size()));
  s.add("Subcases", std::to_string(model.analysis.subcases.size()));
  std::vector<std::pair<std::string, std::string>> params(model.params.begin(),
                                                           model.params.end());
  std::sort(params.begin(), params.end());
  std::string text;
  for (const auto &[name, value] : params)
    text += (text.empty() ? "" : ", ") + name + "=" + value;
  s.add("PARAM settings", text.empty() ? "none" : text);
  return s;
}

F06Section solver_section(std::string_view backend,
                          std::vector<std::pair<std::string, std::string>> details) {
  F06Section s{"SOLVER", {}};
  s.add("Backend", std::string(backend));
  for (auto &[name, value] : details)
    s.add(std::move(name), std::move(value));
  return s;
}

void collect_notices(SolveSummary &summary, std::vector<std::string> notices,
                     std::string_view context) {
  for (auto &notice : notices) {
    std::string text =
        context.empty() ? std::move(notice) : std::format("{}: {}", context, notice);
    if (summary.notices.empty() || summary.notices.back() != text)
      summary.notices.push_back(std::move(text));
  }
}

} // namespace vibestran
