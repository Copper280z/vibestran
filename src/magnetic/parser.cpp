#include "magnetic/model.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <functional>
#include <numbers>
#include <set>
#include <sstream>

namespace vibestran::magnetic {
namespace {
std::string trim(const std::string &s) {
  auto a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
  return a == std::string::npos ? std::string{} : s.substr(a, b - a + 1);
}
std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return s;
}
struct Line {
  std::string text, location;
};
std::optional<std::string> solution_name(const std::string &text) {
  if (!text.starts_with("SOL") || text.size() <= 3 ||
      !std::isspace(static_cast<unsigned char>(text[3])))
    return {};
  return trim(text.substr(3));
}
// File name of a comment-stripped, trimmed INCLUDE statement, unquoted.
std::string include_filename(const std::string &statement) {
  std::string filename = trim(statement.substr(7));
  if (filename.size() >= 2 &&
      (filename.front() == '\'' || filename.front() == '\"') &&
      filename.back() == filename.front())
    filename = filename.substr(1, filename.size() - 2);
  return filename;
}
Status read_lines(std::istream &in, const std::filesystem::path &base,
                  const std::string &name, std::vector<Line> &lines,
                  std::set<std::filesystem::path> &stack) {
  std::string text;
  int n = 0;
  while (std::getline(in, text)) {
    ++n;
    auto comment = text.find('$');
    if (comment != std::string::npos)
      text.resize(comment);
    const auto stripped = trim(text);
    if (stripped.empty())
      continue;
    const std::string location = std::format("{}:{}", name, n);
    if (upper(stripped).starts_with("INCLUDE")) {
      const std::string filename = include_filename(stripped);
      if (filename.empty())
        return Error{location + ": empty INCLUDE filename"};
      std::error_code ec;
      auto path = std::filesystem::weakly_canonical(base / filename, ec);
      if (ec)
        return Error{location + ": cannot resolve INCLUDE: " + ec.message()};
      if (stack.size() >= 64 || !stack.insert(path).second)
        return Error{location + ": cyclic/deep INCLUDE"};
      std::ifstream child(path);
      if (!child)
        return Error{location + ": cannot open INCLUDE " + path.string()};
      if (auto e = read_lines(child, path.parent_path(), path.string(), lines,
                              stack))
        return e;
      stack.erase(path);
    } else
      lines.push_back({text, location});
  }
  if (in.bad())
    return Error{name + ": input read failed"};
  return {};
}
std::vector<std::string> fields(const std::string &line) {
  std::vector<std::string> f;
  if (line.find(',') != std::string::npos) {
    size_t begin = 0;
    for (;;) {
      auto end = line.find(',', begin);
      f.push_back(trim(
          line.substr(begin, end == std::string::npos ? end : end - begin)));
      if (end == std::string::npos)
        break;
      begin = end + 1;
    }
    if (!f.empty() && (f.back() == "+" ||
                       (f.back().size() > 1 && f.back()[0] == '+' &&
                        std::isalpha(static_cast<unsigned char>(f.back()[1])))))
      f.pop_back();
  } else {
    f.push_back(trim(line.substr(0, 8)));
    int width = f[0].find('*') != std::string::npos ? 16 : 8;
    for (int i = 8; i < 72; i += width)
      f.push_back(
          i < static_cast<int>(line.size()) ? trim(line.substr(i, width)) : "");
  }
  return f;
}
struct Reader {
  std::vector<std::string> f;
  std::string where;
  Status error;
  std::string get(size_t i) const { return i < f.size() ? f[i] : ""; }
  int integer(size_t i, int fallback = 0) {
    auto s = get(i);
    if (s.empty())
      return fallback;
    if (s[0] == '+')
      s.erase(0, 1);
    int value = 0;
    auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size())
      error = Error{where + ": invalid integer '" + s + "'"};
    return value;
  }
  double real(size_t i, double fallback = 0) {
    auto s = upper(get(i));
    if (s.empty())
      return fallback;
    std::replace(s.begin(), s.end(), 'D', 'E');
    if (s.find('E') == std::string::npos)
      for (size_t j = 1; j < s.size(); ++j)
        if (s[j] == '+' || s[j] == '-') {
          s.insert(j, "E");
          break;
        }
    if (s[0] == '+')
      s.erase(0, 1);
    double value = 0;
    auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size() ||
        !std::isfinite(value))
      error = Error{where + ": invalid finite real '" + s + "'"};
    return value;
  }
  Vector vector(size_t first) {
    const double x = real(first), y = real(first + 1), z = real(first + 2);
    return {x, y, z};
  }
  void last(size_t n) {
    for (size_t i = n + 1; i < f.size(); ++i)
      if (!f[i].empty())
        error = Error{where + ": unexpected field " + std::to_string(i)};
  }
  std::vector<int> list(size_t start) {
    std::vector<int> out;
    for (size_t i = start; i < f.size(); ++i) {
      if (f[i].empty())
        continue;
      if (upper(f[i]) == "THRU") {
        if (out.empty() || i + 1 >= f.size()) {
          error = Error{where + ": malformed THRU"};
          break;
        }
        const int last_id = integer(++i), first_id = out.back();
        if (last_id < first_id ||
            static_cast<long long>(last_id) - first_id > 10000000) {
          error = Error{where + ": invalid THRU range"};
          break;
        }
        for (int id = first_id; id < last_id;)
          out.push_back(++id);
      } else
        out.push_back(integer(i));
    }
    return out;
  }
};
template <class Map, class T>
void insert_unique(Map &map, int id, T value, Reader &r) {
  if (id <= 0 || !map.emplace(id, std::move(value)).second)
    r.error =
        Error{r.where + ": nonpositive or duplicate id " + std::to_string(id)};
}
Vector transform(const CoordinateSystem &cs, Vector p) {
  const double degree = std::numbers::pi / 180;
  if (cs.type == 'C')
    p = Vector(p.x() * std::cos(p.y() * degree),
               p.x() * std::sin(p.y() * degree), p.z());
  else if (cs.type == 'S')
    p = Vector(p.x() * std::sin(p.y() * degree) * std::cos(p.z() * degree),
               p.x() * std::sin(p.y() * degree) * std::sin(p.z() * degree),
               p.x() * std::cos(p.y() * degree));
  return cs.a + coordinate_axes(cs) * p;
}
Status resolve_coordinates(Model &m) {
  std::map<int, int> state;
  std::function<Status(int)> resolve = [&](int id) -> Status {
    if (id == 0)
      return {};
    auto it = m.coordinates.find(id);
    if (it == m.coordinates.end())
      return Error{"Undefined coordinate system " + std::to_string(id)};
    if (state[id] == 2)
      return {};
    if (state[id] == 1)
      return Error{"Cyclic coordinate system references"};
    state[id] = 1;
    auto &cs = it->second;
    if (cs.parent) {
      if (auto e = resolve(cs.parent))
        return e;
      const auto &parent = m.coordinates.at(cs.parent);
      cs.a = transform(parent, cs.a);
      cs.b = transform(parent, cs.b);
      cs.c = transform(parent, cs.c);
      cs.parent = 0;
    }
    if ((cs.b - cs.a).norm() == 0 ||
        (cs.b - cs.a).cross(cs.c - cs.a).norm() <=
            1e-12 * (cs.b - cs.a).norm() * (cs.c - cs.a).norm())
      return Error{"Degenerate coordinate system " + std::to_string(id)};
    state[id] = 2;
    return {};
  };
  for (const auto &[id, cs] : m.coordinates) {
    (void)cs;
    if (auto e = resolve(id))
      return e;
  }
  for (auto &[id, g] : m.nodes) {
    (void)id;
    if (g.coordinates) {
      if (!m.coordinates.contains(g.coordinates))
        return Error{"GRID uses undefined coordinates"};
      g.position = transform(m.coordinates.at(g.coordinates), g.position);
      g.coordinates = 0;
    }
    g.position *= m.options.length_scale;
  }
  for (auto &p : m.probes) {
    if (p.coordinates) {
      if (!m.coordinates.contains(p.coordinates))
        return Error{std::format("MPROBE {} uses undefined coordinates {}",
                                 p.id, p.coordinates)};
      p.position = transform(m.coordinates.at(p.coordinates), p.position);
      p.coordinates = 0;
    }
    p.position *= m.options.length_scale;
  }
  for (auto &[id, cs] : m.coordinates) {
    (void)id;
    cs.a *= m.options.length_scale;
    cs.b *= m.options.length_scale;
    cs.c *= m.options.length_scale;
  }
  for (auto &[id, p] : m.wire_properties) {
    (void)id;
    p.width *= m.options.length_scale;
    p.height *= m.options.length_scale;
  }
  m.options.wire_step *= m.options.length_scale;
  m.options.depth *= m.options.length_scale;
  return {};
}
// A PAIRGAP solved as meshed air becomes exactly what the deck would say
// without it: an ordinary property of a mu_r = 1 material (a new MATMAG ID).
// The solver then sees no gap at all; the PID is kept only for reporting.
Status resolve_air_gaps(Model &m) {
  if (!m.air_gaps.empty() && !m.options.planar)
    return Error{std::format("PAIRGAP {} requires MAG2D; a 3D model meshes its "
                             "air gap with ordinary air elements",
                             m.air_gaps.begin()->first)};
  for (auto it = m.air_gaps.begin(); it != m.air_gaps.end();) {
    const auto [id, gap] = *it;
    if (m.properties.contains(id))
      return Error{std::format("PAIRGAP {} duplicates the ID of a PSOLID, PMAG or "
                               "PMAG2D property",
                               id)};
    if (std::none_of(m.elements.begin(), m.elements.end(),
                     [id](const Element &c) { return c.property == id; }))
      return Error{std::format("PAIRGAP {} has no elements; mesh the air gap and "
                               "give its elements PID {}",
                               id, id)};
    const auto method = m.options.air_gap_method.value_or(gap.method);
    if (method != AirGapMethod::Mesh) {
      ++it;
      continue;
    }
    const int air = m.materials.empty() ? 1 : m.materials.rbegin()->first + 1;
    m.materials[air] = Material{};
    m.properties[id] = Property{air};
    m.meshed_air_gaps[id] = m.options.air_gap_method ? "PARAM,AIRGAP,MESH"
                                                     : "PAIRGAP METHOD=MESH";
    it = m.air_gaps.erase(it);
  }
  return {};
}
// Appends the elements of each ESET PROP row, in element order, then rejects
// any set that selects an element twice.
Status expand_property_sets(Model &m,
                            const std::map<int, std::vector<int>> &rows) {
  for (const auto &[sid, properties] : rows) {
    auto &out = m.element_sets[sid];
    for (int pid : properties) {
      const size_t before = out.size();
      for (const auto &c : m.elements)
        if (c.property == pid)
          out.push_back(c.id);
      if (out.size() == before)
        return Error{std::format("ESET {} PROP {} selects no magnetic "
                                 "elements",
                                 sid, pid)};
    }
  }
  for (const auto &[sid, ids] : m.element_sets) {
    std::set<int> seen;
    for (int id : ids)
      if (!seen.insert(id).second)
        return Error{
            std::format("ESET {} selects element {} more than once", sid, id)};
  }
  return {};
}
Result<Model> parse_lines(const std::vector<Line> &lines) {
  Model m;
  std::map<int, std::vector<int>> property_rows;
  std::set<int> prescribed_mbc; // SIDs with a ZERO or UNIFORM row
  std::vector<int> element_dimensions;
  Subcase global, current;
  bool bulk = false, in_case = false, sol = false;
  for (size_t i = 0; i < lines.size(); ++i) {
    const auto text = upper(trim(lines[i].text));
    if (const auto solution = solution_name(text)) {
      if (*solution != "MAGSTAT")
        return Error{lines[i].location + ": expected SOL MAGSTAT"};
      sol = true;
      continue;
    }
    if (text == "CEND")
      continue;
    if (text.starts_with("BEGIN") && trim(text.substr(5)) == "BULK") {
      bulk = true;
      continue;
    }
    if (text == "ENDDATA")
      break;
    if (!bulk) {
      if (text.starts_with("SUBCASE")) {
        if (in_case)
          m.subcases.push_back(current);
        current = global;
        Reader r{{"", trim(text.substr(7))}, lines[i].location, {}};
        current.id = r.integer(1);
        if (r.error)
          return *r.error;
        in_case = true;
      } else {
        // A describer in parentheses, as in BFIELD(EVERY=2) = ALL, holds its
        // own '='.
        const auto open = text.find('('), first = text.find('=');
        const bool described = open < first;
        const auto close = described ? text.find(')', open) : std::string::npos;
        if (described && close == std::string::npos)
          return Error{lines[i].location + ": unclosed describer parenthesis"};
        auto equals = text.find('=', described ? close : 0);
        if (equals == std::string::npos)
          return Error{lines[i].location +
                       ": unsupported executive/case control"};
        auto key = trim(text.substr(0, described ? open : equals)),
             value = trim(text.substr(equals + 1));
        const auto describer =
            described ? trim(text.substr(open + 1, close - open - 1)) : "";
        if (described && (key != "BFIELD" || !trim(text.substr(close + 1, equals - close - 1)).empty()))
          return Error{lines[i].location + ": " + key +
                       (key == "BFIELD" ? " describer must be (EVERY=n)"
                                        : " takes no describer")};
        auto &sc = in_case ? current : global;
        Reader r{{key, value}, lines[i].location, {}};
        if (key == "LOAD")
          sc.load = r.integer(1);
        else if (key == "MAGBC")
          sc.boundary = r.integer(1);
        else if (key == "MOTION")
          sc.motion = r.integer(1);
        else if (key == "DRIVE")
          sc.drive = r.integer(1);
        else if (key == "LDQ") {
          if (value != "APPARENT" && value != "INCREMENTAL")
            return Error{lines[i].location + ": LDQ must be APPARENT or INCREMENTAL"};
          sc.apparent = value == "APPARENT";
        }
        else if (key == "MSTEP") {
          if (value != "ALL" && value != "NONE")
            return Error{lines[i].location + ": MSTEP must be ALL or NONE"};
          sc.steps = value == "ALL";
        } else if (key == "BFIELD") {
          sc.field = {value != "NONE", 0, 1};
          if (value != "ALL" && value != "NONE")
            sc.field.set = r.integer(1);
          if (described) {
            const auto split = describer.find('=');
            Reader every{{"", split == std::string::npos ? "" : trim(describer.substr(split + 1))},
                         lines[i].location, {}};
            sc.field.every = every.integer(1);
            if (split == std::string::npos || trim(describer.substr(0, split)) != "EVERY" ||
                every.error || sc.field.every < 1)
              return Error{lines[i].location +
                           ": BFIELD describer must be (EVERY=n) with n >= 1"};
          }
        } else if (key == "MPROBE" || key == "MSOURCE") {
          if (value != "ALL" && value != "NONE")
            return Error{lines[i].location + ": " + key +
                         " must be ALL or NONE"};
          if (key == "MPROBE")
            sc.probes = value == "ALL";
          else if (in_case)
            return Error{lines[i].location +
                         ": MSOURCE is a global request, above any SUBCASE"};
          else
            m.options.source_output = value == "ALL";
        }
        else if (key == "LABEL")
          // equals indexes the trimmed text; take the label from the
          // trimmed original so its case survives.
          sc.label = trim(trim(lines[i].text).substr(equals + 1));
        else if (key != "TITLE" && key != "SUBTITLE")
          return Error{lines[i].location +
                       ": unsupported MAGSTAT case control " + key};
        if (r.error)
          return *r.error;
      }
      continue;
    }
    Reader r{fields(lines[i].text), lines[i].location, {}};
    while (i + 1 < lines.size()) {
      const auto &next = lines[i + 1].text;
      auto f = fields(next);
      if (f.empty() ||
          !(f[0].empty() || f[0].starts_with('+') || f[0].starts_with('*')))
        break;
      const bool large = r.f[0].find('*') != std::string::npos;
      const size_t width = large ? 4 : 8;
      while ((r.f.size() - 1) % width != 0)
        r.f.push_back("");
      r.f.insert(r.f.end(), f.begin() + 1, f.end());
      ++i;
    }
    auto key = upper(r.get(0));
    key.erase(std::remove(key.begin(), key.end(), '*'), key.end());
    if (key == "GRID") {
      Grid g;
      int id = r.integer(1);
      g.coordinates = r.integer(2);
      g.position = r.vector(3);
      if (r.integer(6) != 0)
        r.error = Error{r.where + ": magnetic GRID requires CD=0"};
      r.last(6);
      insert_unique(m.nodes, id, g, r);
    } else if (key == "CTETRA" || key == "CHEXA" || key == "CPENTA" ||
               key == "CTRIA3" || key == "CQUAD4") {
      Element c;
      c.id = r.integer(1);
      c.property = r.integer(2);
      const int count = key == "CHEXA"    ? 8
                        : key == "CPENTA" ? 6
                        : key == "CTRIA3" ? 3
                                          : 4;
      c.nodes.resize(count);
      for (int k = 0; k < count; ++k)
        c.nodes[k] = r.integer(k + 3);
      r.last(count + 2);
      m.elements.push_back(c);
      // Keep card semantics explicit: four-node solid and planar cells differ.
      element_dimensions.push_back(key == "CTRIA3" || key == "CQUAD4" ? 2 : 3);
    } else if (key == "PSOLID" || key == "PMAG" || key == "PMAG2D") {
      // PSOLID CORDM orients remanence along its first axis; PMAG/PMAG2D
      // give the frame and a signed axis. A blank frame is not basic.
      Property p;
      const int id = r.integer(1);
      p.material = r.integer(2);
      p.oriented = !r.get(3).empty();
      p.coordinates = r.integer(3);
      if (key != "PSOLID")
        p.axis = r.integer(4, 1);
      r.last(key == "PSOLID" ? 3 : 4);
      insert_unique(m.properties, id, p, r);
    } else if (key == "PAIRGAP") {
      AirGapProperty gap;
      const int id = r.integer(1);
      const auto method = upper(r.get(2));
      if (method == "MESH")
        gap.method = AirGapMethod::Mesh;
      else if (method == "BAND")
        gap.method = AirGapMethod::Band;
      else if (!method.empty() && method != "AGE")
        return Error{r.where + ": PAIRGAP METHOD must be AGE, BAND or MESH"};
      gap.harmonics = r.integer(3);
      if (gap.harmonics < 0)
        return Error{r.where + ": PAIRGAP NHARM must not be negative (blank for the default)"};
      gap.band_points = r.integer(4);
      if (gap.band_points < 0 || (gap.band_points > 0 && gap.band_points < 3))
        return Error{r.where + ": PAIRGAP NBAND must be at least 3 (blank for the default)"};
      gap.band_layers = r.integer(5);
      if (gap.band_layers < 0)
        return Error{r.where + ": PAIRGAP NLAYER must be at least 1 (blank for the default)"};
      r.last(5);
      insert_unique(m.air_gaps, id, gap, r);
    } else if (key == "MOTION") {
      Motion motion;
      motion.id = r.integer(1);
      motion.set = r.integer(2);
      motion.coordinates = r.integer(3);
      const auto type = upper(r.get(4));
      if (type == "TRANSLATE")
        return Error{r.where + ": MOTION TRANSLATE is not supported yet; it needs "
                               "the linear air-gap element. Use ROTATE"};
      if (type != "ROTATE")
        return Error{r.where + ": MOTION TYPE must be ROTATE"};
      motion.start = r.real(5);
      motion.step = r.real(6);
      motion.count = r.integer(7, 1);
      motion.speed = r.real(9);
      r.last(9);
      insert_unique(m.motions, motion.id, motion, r);
    } else if (key == "DRIVE") {
      Drive drive;
      drive.id = r.integer(1);
      const auto wave = upper(r.get(2));
      if (wave != "SINE" && wave != "SIX")
        return Error{r.where + ": DRIVE WAVE must be SINE or SIX"};
      drive.six = wave == "SIX";
      drive.poles = r.integer(3);
      drive.peak = r.real(4);
      drive.gamma = r.real(5);
      if (const auto theta0 = upper(r.get(6)); !theta0.empty() && theta0 != "AUTO")
        drive.theta0 = r.real(6);
      if (!r.get(7).empty() || !r.get(8).empty())
        return Error{r.where + ": DRIVE fields 7 and 8 must be blank; the phase "
                               "circuits start on the continuation"};
      drive.phases = r.list(9);
      insert_unique(m.drives, drive.id, std::move(drive), r);
    } else if (key == "MPERIOD") {
      Periodic period;
      period.id = r.integer(1);
      const auto type = upper(r.get(2));
      if (type != "PERIODIC" && type != "ANTI")
        return Error{r.where + ": MPERIOD TYPE must be PERIODIC or ANTI"};
      period.anti = type == "ANTI";
      period.first = r.integer(3);
      period.second = r.integer(4);
      period.coordinates = r.integer(5);
      period.sectors = r.integer(6);
      r.last(6);
      if (m.period)
        return Error{std::format("{}: MPERIOD {} after MPERIOD {}; a model is one "
                                 "sector, with one pair of periodic sides",
                                 r.where, period.id, m.period->id)};
      m.period = period;
    } else if (key == "MAG2D" || key == "MAGAXI") {
      if (m.options.planar)
        r.error = Error{r.where + ": a deck takes one MAG2D or MAGAXI card"};
      m.options.planar = true;
      m.options.plane_coordinates = r.integer(1);
      if (key == "MAGAXI") {
        m.options.axisymmetric = true;
        r.last(1);
      } else {
        m.options.depth = r.real(2);
        r.last(2);
      }
    } else if (key == "COIL2D") {
      PlanarCoil c;
      c.id = r.integer(1);
      c.elements = r.integer(2);
      c.turns = r.real(3);
      r.last(3);
      m.planar_coils.push_back(c);
    } else if (key == "MATMAG") {
      Material a;
      int id = r.integer(1);
      a.mur = r.real(2, 1);
      a.bh_table = r.integer(3);
      a.br = r.real(4);
      if (a.bh_table && !r.get(2).empty())
        return Error{r.where + ": MATMAG MUR and BHTID are mutually "
                               "exclusive; leave MUR blank with a table"};
      r.last(4);
      insert_unique(m.materials, id, a, r);
    } else if (key == "TABLEBH") {
      // TABLEM1 layout: B,H pairs on continuations, ending with ENDT. Blank
      // fields are padding.
      const int id = r.integer(1);
      const auto interpolation = upper(r.get(2));
      if (!interpolation.empty() && interpolation != "LINEAR")
        return Error{r.where + ": TABLEBH interpolation must be LINEAR"};
      std::vector<std::string> values;
      bool ended = false;
      for (size_t k = 3; k < r.f.size(); ++k) {
        if (r.f[k].empty())
          continue;
        if (ended)
          return Error{r.where + ": TABLEBH data follows ENDT"};
        if (upper(r.f[k]) == "ENDT")
          ended = true;
        else
          values.push_back(r.f[k]);
      }
      if (!ended || values.size() % 2 != 0)
        return Error{r.where + ": TABLEBH requires B,H pairs ending in ENDT"};
      std::vector<std::pair<double, double>> curve;
      Reader pairs{{""}, r.where, {}};
      pairs.f.insert(pairs.f.end(), values.begin(), values.end());
      for (size_t k = 1; k < pairs.f.size(); k += 2)
        curve.emplace_back(pairs.real(k), pairs.real(k + 1));
      if (pairs.error)
        return *pairs.error;
      insert_unique(m.bh_tables, id, std::move(curve), r);
    } else if (key == "MATCOND") {
      int id = r.integer(1);
      double sigma = r.real(2);
      r.last(2);
      insert_unique(m.conductivity, id, sigma, r);
    } else if (key == "ESET") {
      int id = r.integer(1);
      if (upper(r.get(2)) == "PROP") {
        auto values = r.list(3);
        auto &out = property_rows[id];
        out.insert(out.end(), values.begin(), values.end());
        m.element_sets[id];
      } else {
        auto values = r.list(2);
        auto &out = m.element_sets[id];
        out.insert(out.end(), values.begin(), values.end());
      }
    } else if (key == "FSET") {
      int id = r.integer(1);
      FaceSelection f;
      f.element = r.integer(2);
      f.nodes.clear();
      for (size_t k = 3; k < r.f.size(); ++k)
        if (!r.get(k).empty())
          f.nodes.push_back(r.integer(k));
      if (f.nodes.size() < 2 || f.nodes.size() > 4)
        r.error = Error{r.where + ": FSET requires 2, 3 or 4 face nodes"};
      m.face_sets[id].push_back(f);
    } else if (key == "COIL") {
      Coil c;
      c.id = r.integer(1);
      c.elements = r.integer(2);
      c.turns = r.real(3, 1);
      if (upper(r.get(4)) == "CUT") {
        c.plane_cut = true;
        c.cut_coordinates = r.integer(5);
      } else {
        c.start = r.integer(4);
        c.end = r.integer(5);
      }
      r.last(5);
      m.coils.push_back(c);
    } else if (key == "PWIRE") {
      int id = r.integer(1);
      WireProperty p;
      p.material = r.integer(2);
      auto shape = upper(r.get(3));
      p.rectangular = shape == "RECT";
      if (shape != "ROUND" && shape != "RECT")
        r.error = Error{r.where + ": PWIRE shape must be ROUND or RECT"};
      p.width = r.real(4);
      p.height = r.real(5);
      r.last(5);
      insert_unique(m.wire_properties, id, p, r);
    } else if (key == "CWIRE") {
      Wire w;
      w.id = r.integer(1);
      w.property = r.integer(2);
      w.first = r.integer(3);
      w.second = r.integer(4);
      if (!r.get(6).empty() || !r.get(7).empty() ||
          r.get(5).find_first_of(".EeDd") != std::string::npos)
        w.orientation = r.vector(5);
      else
        w.orientation_node = r.integer(5);
      r.last(7);
      m.wires.push_back(w);
    } else if (key == "WIREPATH") {
      WirePath w;
      w.id = r.integer(1);
      w.element = r.integer(2);
      r.last(2);
      m.wire_paths.push_back(w);
    } else if (key == "CIRCUIT") {
      Circuit c;
      c.id = r.integer(1);
      c.parallel = r.integer(2, 1);
      for (size_t k = 3; k < r.f.size(); k += 2) {
        if (r.get(k).empty() && r.get(k + 1).empty())
          continue;
        c.members.emplace_back(r.integer(k), r.integer(k + 1));
      }
      insert_unique(m.circuits, c.id, std::move(c), r);
    } else if (key == "ICIRC") {
      Current c;
      c.set = r.integer(1);
      c.target = r.integer(2);
      c.amperes = r.real(3);
      r.last(3);
      m.currents.push_back(c);
    } else if (key == "LOAD") {
      int id = r.integer(1);
      Combination c;
      c.scale = r.real(2);
      for (size_t k = 3; k < r.f.size(); k += 2) {
        if (r.get(k).empty() && r.get(k + 1).empty())
          continue;
        c.terms.emplace_back(r.real(k), r.integer(k + 1));
      }
      insert_unique(m.combinations, id, std::move(c), r);
    } else if (key == "MBC") {
      // Rows of one SID combine: one ZERO or UNIFORM row, and any number
      // of NORMAL rows.
      const int id = r.integer(1), faces = r.integer(2);
      const auto kind = upper(r.get(3));
      if (id <= 0)
        return Error{r.where + ": nonpositive MBC id " + std::to_string(id)};
      auto &bc = m.boundaries[id];
      if (kind == "NORMAL") {
        if (faces == 0)
          return Error{r.where + ": MBC NORMAL needs an FSET of the faces the "
                                 "flux crosses normally"};
        bc.normal.push_back(faces);
        r.last(3);
      } else if (kind == "ZERO" || kind == "UNIFORM") {
        if (!prescribed_mbc.insert(id).second)
          return Error{std::format("{}: MBC {} has a second ZERO or UNIFORM row; "
                                   "one row prescribes A (FSET 0 for the rest "
                                   "of the exterior), and NORMAL rows add "
                                   "flux-normal faces",
                                   r.where, id)};
        bc.faces = faces;
        if (kind == "UNIFORM")
          bc.field = r.vector(4);
        r.last(6);
      } else {
        r.error = Error{r.where + ": MBC type must be ZERO, UNIFORM or NORMAL"};
      }
    } else if (key == "MPROBE") {
      Probe p;
      p.id = r.integer(1);
      p.coordinates = r.integer(2);
      p.position = r.vector(3);
      p.output_coordinates = r.integer(6);
      r.last(6);
      m.probes.push_back(p);
    } else if (key == "MFORCE") {
      ForceSurface f;
      f.id = r.integer(1);
      const auto kind = upper(r.get(2));
      if (kind != "BODY" && kind != "FACES")
        return Error{r.where + ": MFORCE is ID,FACES,FSET,CID or "
                               "ID,BODY,ESET,CID,LAYER"};
      f.body = kind == "BODY";
      f.set = r.integer(3);
      f.coordinates = r.integer(4);
      if (f.body)
        f.layers = r.integer(5, 1);
      r.last(f.body ? 5 : 4);
      m.forces.push_back(f);
    } else if (key == "CORD2R" || key == "CORD2C" || key == "CORD2S") {
      int id = r.integer(1);
      CoordinateSystem c;
      c.type = key.back();
      c.parent = r.integer(2);
      c.a = r.vector(3);
      c.b = r.vector(6);
      c.c = r.vector(9);
      r.last(11);
      insert_unique(m.coordinates, id, c, r);
    } else if (key == "PARAM") {
      auto option = upper(r.get(1));
      if (option == "MAGLEN")
        m.options.length_scale = r.real(2);
      else if (option == "MAGTOL")
        m.options.nonlinear_tolerance = r.real(2);
      else if (option == "MAGITER")
        m.options.max_iterations = r.integer(2);
      else if (option == "MAGPCG")
        m.options.tangent_reuse = r.integer(2);
      else if (option == "WIRESIDE")
        m.options.wire_sides = r.integer(2);
      else if (option == "WIRELAY")
        m.options.wire_layers = r.integer(2);
      else if (option == "WIRESTEP")
        m.options.wire_step = r.real(2);
      else if (option == "WIREMESH") {
        const auto value = upper(r.get(2));
        if (value == "AUTO")
          m.options.wire_mesh = WireMesh::Auto;
        else if (value == "HEX")
          m.options.wire_mesh = WireMesh::Hex;
        else if (value == "TET")
          m.options.wire_mesh = WireMesh::Tet;
        else
          return Error{r.where + ": WIREMESH must be AUTO, HEX or TET"};
      } else if (option == "AIRGAP") {
        const auto value = upper(r.get(2));
        if (value == "AGE")
          m.options.air_gap_method = AirGapMethod::Age;
        else if (value == "BAND")
          m.options.air_gap_method = AirGapMethod::Band;
        else if (value == "MESH")
          m.options.air_gap_method = AirGapMethod::Mesh;
        else
          return Error{r.where + ": AIRGAP must be AGE, BAND or MESH"};
      } else
        return Error{r.where + ": unsupported MAGSTAT parameter " + option};
      r.last(2);
    } else
      return Error{r.where + ": unsupported MAGSTAT bulk card " + key};
    if (r.error)
      return *r.error;
  }
  for (int dimension : element_dimensions)
    if (dimension != (m.options.planar ? 2 : 3))
      return Error{"MAG2D and MAGAXI require CTRIA3/CQUAD4 only; 3D MAGSTAT "
                   "requires CTETRA4/CHEXA8/CPENTA6"};
  if (!sol)
    return Error{"Missing SOL MAGSTAT"};
  if (in_case)
    m.subcases.push_back(current);
  else
    m.subcases.push_back(global);
  if (!(m.options.length_scale > 0))
    return Error{"MAGLEN must be positive"};
  if (auto e = resolve_air_gaps(m))
    return *e;
  if (auto e = expand_property_sets(m, property_rows))
    return *e;
  if (auto e = resolve_coordinates(m))
    return *e;
  if (auto e = validate(m))
    return *e;
  return m;
}
struct SolutionScan {
  std::optional<std::string> solution;
  bool finished{false}; // SOL or BEGIN reached
};
// Finds the SOL statement ahead of BEGIN BULK, following INCLUDEs. Deck type
// detection must not reject structural decks, so an include that cannot be
// opened, or that cycles, is skipped here; parse_file reports it if the deck
// turns out to be magnetic.
SolutionScan scan_solution(std::istream &in, const std::filesystem::path &base,
                           std::set<std::filesystem::path> &stack) {
  std::string raw;
  while (std::getline(in, raw)) {
    const auto stripped = trim(raw.substr(0, raw.find('$')));
    const auto text = upper(stripped);
    if (auto solution = solution_name(text))
      return {std::move(solution), true};
    if (text.starts_with("BEGIN"))
      return {std::nullopt, true};
    if (!text.starts_with("INCLUDE"))
      continue;
    const std::string filename = include_filename(stripped);
    if (filename.empty())
      continue;
    std::error_code ec;
    auto path = std::filesystem::weakly_canonical(base / filename, ec);
    if (ec || stack.size() >= 64 || !stack.insert(path).second)
      continue;
    std::ifstream child(path);
    auto scan = child ? scan_solution(child, path.parent_path(), stack)
                      : SolutionScan{};
    stack.erase(path);
    if (scan.finished)
      return scan;
  }
  return {};
}
} // namespace

Matrix coordinate_axes(const CoordinateSystem &cs) {
  Vector z = (cs.b - cs.a).normalized(), y = z.cross(cs.c - cs.a).normalized();
  Matrix r;
  r.col(0) = y.cross(z);
  r.col(1) = y;
  r.col(2) = z;
  return r;
}
Matrix local_axes(const CoordinateSystem &cs, const Vector &x) {
  const Matrix frame = coordinate_axes(cs);
  if (cs.type == 'R')
    return frame;
  const Vector p = frame.transpose() * (x - cs.a);
  Matrix local;
  if (cs.type == 'C') {
    const double theta = std::atan2(p.y(), p.x());
    local = Eigen::AngleAxisd(theta, Vector::UnitZ()).toRotationMatrix();
  } else {
    const double theta = std::atan2(std::hypot(p.x(), p.y()), p.z()),
                 phi = std::atan2(p.y(), p.x());
    local.col(0) = Vector(std::sin(theta) * std::cos(phi),
                          std::sin(theta) * std::sin(phi), std::cos(theta));
    local.col(1) = Vector(std::cos(theta) * std::cos(phi),
                          std::cos(theta) * std::sin(phi), -std::sin(theta));
    local.col(2) = Vector(-std::sin(phi), std::cos(phi), 0);
  }
  return frame * local;
}
Result<Model> parse_file(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in)
    return Error{"Cannot open magnetic deck " + path.string()};
  std::vector<Line> lines;
  std::set<std::filesystem::path> stack;
  std::error_code ec;
  stack.insert(std::filesystem::weakly_canonical(path, ec));
  if (auto e = read_lines(in, path.parent_path(), path.string(), lines, stack))
    return *e;
  return parse_lines(lines);
}
Result<Model> parse_string(const std::string &text) {
  std::istringstream in(text);
  std::vector<Line> lines;
  std::set<std::filesystem::path> stack;
  if (auto e = read_lines(in, ".", "<string>", lines, stack))
    return *e;
  return parse_lines(lines);
}
Result<bool> is_magnetic_deck(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in)
    return Error{"Cannot open input " + path.string()};
  std::set<std::filesystem::path> stack;
  std::error_code ec;
  stack.insert(std::filesystem::weakly_canonical(path, ec));
  return scan_solution(in, path.parent_path(), stack).solution == "MAGSTAT";
}

Status validate(const Model &m) {
  if (m.elements.empty())
    return Error{"MAGSTAT requires a magnetic mesh"};
  if (m.subcases.empty())
    return Error{"MAGSTAT requires at least one subcase"};
  // Periodic sectors and NORMAL faces are planar; say so before anything
  // else about a 3D deck that uses them.
  // The tubular air gap, axial periodicity and motion are not built yet.
  if (m.options.axisymmetric) {
    const auto missing = [](std::string_view what) {
      return Error{std::format("{} is not available for MAGAXI models yet", what)};
    };
    if (!m.air_gaps.empty() && m.meshed_air_gaps.size() != m.air_gaps.size())
      return missing(std::format("PAIRGAP {} (AGE or BAND)", m.air_gaps.begin()->first));
    if (m.period)
      return missing(std::format("MPERIOD {}", m.period->id));
    if (!m.motions.empty())
      return missing(std::format("MOTION {}", m.motions.begin()->first));
    if (!m.drives.empty())
      return missing(std::format("DRIVE {}", m.drives.begin()->first));
  }
  if (m.period && !m.options.planar)
    return Error{std::format("MPERIOD {} is for MAG2D models; 3D periodic "
                             "sectors are not implemented",
                             m.period->id)};
  for (const auto &[id, bc] : m.boundaries)
    if (!bc.normal.empty() && !m.options.planar)
      return Error{std::format("MBC {} NORMAL is for MAG2D models; the 3D "
                               "exterior takes zero tangential A",
                               id)};
  if (m.options.tangent_reuse < 0 || m.options.tangent_reuse > 1000)
    return Error{std::format("PARAM,MAGPCG,{} must be from 0 (factor every "
                             "tangent) to 1000 CG iterations",
                             m.options.tangent_reuse)};
  if (m.options.max_iterations < 1 ||
      !(m.options.nonlinear_tolerance > 0 &&
        m.options.nonlinear_tolerance < 1) ||
      m.options.wire_sides < 8 || m.options.wire_sides > 256 ||
      m.options.wire_layers < 1 || m.options.wire_layers > 32 ||
      m.options.wire_step < 0 || !std::isfinite(m.options.wire_step))
    return Error{"Invalid magnetic solver/source refinement options"};
  for (const auto &[id, curve] : m.bh_tables) {
    if (id <= 0 || curve.size() < 2 ||
        curve.front() != std::pair<double, double>{0, 0})
      return Error{std::format("TABLEBH {} must begin at (0,0) and contain at "
                               "least two points",
                               id)};
    for (size_t i = 1; i < curve.size(); ++i)
      if (!(curve[i].first > curve[i - 1].first &&
            curve[i].second > curve[i - 1].second) ||
          !std::isfinite(curve[i].first) || !std::isfinite(curve[i].second))
        return Error{std::format(
            "TABLEBH {} B and H must both be strictly increasing", id)};
  }
  for (const auto &[id, mat] : m.materials) {
    if (!(mat.mur > 0) || !std::isfinite(mat.mur) || !(mat.br >= 0) ||
        !std::isfinite(mat.br))
      return Error{std::format("Invalid MATMAG {}: MUR must be positive and BR "
                               "a nonnegative magnitude",
                               id)};
    if (mat.bh_table && (!m.bh_tables.contains(mat.bh_table) ||
                         mat.br != 0 || mat.mur != 1))
      return Error{std::format("MATMAG {} nonlinear curve must exist and "
                               "cannot be combined with MUR or remanence",
                               id)};
  }
  // The plane's normal: the z axis of a MAG2D CID; the y axis of a MAGAXI
  // CID, whose z axis is the symmetry axis.
  const Matrix plane_axes =
      m.options.plane_coordinates && m.coordinates.contains(m.options.plane_coordinates)
          ? coordinate_axes(m.coordinates.at(m.options.plane_coordinates))
          : Matrix::Identity();
  const Vector plane_normal = plane_axes.col(m.options.axisymmetric ? 1 : 2);
  const Vector plane_origin =
      m.options.plane_coordinates && m.coordinates.contains(m.options.plane_coordinates)
          ? m.coordinates.at(m.options.plane_coordinates).a
          : Vector::Zero();
  for (const auto &[id, p] : m.properties) {
    auto mat = m.materials.find(p.material);
    if (mat == m.materials.end())
      return Error{std::format("Magnetic property {} references undefined "
                               "MATMAG {}",
                               id, p.material)};
    if (p.axis == 0 || std::abs(p.axis) > 3)
      return Error{std::format("Magnetic property {} AXIS must be 1, 2 or 3, "
                               "optionally negative",
                               id)};
    if (p.coordinates && !m.coordinates.contains(p.coordinates))
      return Error{std::format("Magnetic property {} references undefined "
                               "coordinates {}",
                               id, p.coordinates)};
    if (mat->second.br == 0)
      continue;
    if (!p.oriented)
      return Error{std::format(
          "Magnetic property {} references magnetized MATMAG {} without an "
          "orientation; set PMAG/PMAG2D CID or PSOLID CORDM (0 for basic)",
          id, p.material)};
    if (!m.options.planar)
      continue;
    // Planar magnets must stay in plane everywhere, so check the frame once.
    const auto *cs = p.coordinates ? &m.coordinates.at(p.coordinates) : nullptr;
    const Matrix frame = cs ? coordinate_axes(*cs) : Matrix::Identity();
    const char type = cs ? cs->type : 'R';
    if (m.options.axisymmetric) {
      // In the meridian plane: a rectangular axis in it, or the radial or
      // axial direction of a cylindrical system about the symmetry axis.
      const Vector axis = plane_axes.col(2);
      const bool meridian =
          type == 'R'
              ? std::abs(frame.col(std::abs(p.axis) - 1).dot(plane_normal)) <= 1e-12
              : type == 'C' && std::abs(p.axis) != 2 &&
                    frame.col(2).cross(axis).norm() <= 1e-12 &&
                    (cs->a - plane_origin).cross(axis).norm() <=
                        1e-12 * std::max(1.0, (cs->a - plane_origin).norm());
      if (!meridian)
        return Error{std::format(
            "MAGAXI remanence must lie in the meridian (x-z) plane (property "
            "{}): use a rectangular axis in it, or AXIS 1 (radial) or 3 "
            "(axial) of a cylindrical system about the symmetry axis",
            id)};
      continue;
    }
    const bool in_plane =
        type == 'R'
            ? std::abs(frame.col(std::abs(p.axis) - 1).dot(plane_normal)) <=
                  1e-12
            : type == 'C' && std::abs(p.axis) < 3 &&
                  frame.col(2).cross(plane_normal).norm() <= 1e-12;
    if (!in_plane)
      return Error{std::format(
          "MAG2D remanence must lie in the simulation plane (property {}): use "
          "an in-plane rectangular axis, or AXIS 1 or 2 of a cylindrical "
          "system about the plane normal",
          id)};
  }
  for (const auto &[id, sigma] : m.conductivity)
    if (!(sigma > 0) || !std::isfinite(sigma))
      return Error{"MATCOND conductivity must be positive: " +
                   std::to_string(id)};
  if (!(m.options.depth > 0) || !std::isfinite(m.options.depth))
    return Error{"MAG2D depth must be positive and finite"};
  if (m.options.plane_coordinates &&
      (!m.coordinates.contains(m.options.plane_coordinates) ||
       m.coordinates.at(m.options.plane_coordinates).type != 'R'))
    return Error{m.options.axisymmetric ? "MAGAXI CID must reference CORD2R"
                                        : "MAG2D CID must reference CORD2R"};
  if (m.options.planar && (!m.wires.empty() || !m.coils.empty()))
    return Error{"MAG2D and MAGAXI use COIL2D regions; CWIRE and volumetric "
                 "COIL require 3D"};
  if (!m.options.planar && !m.planar_coils.empty())
    return Error{"COIL2D requires MAG2D or MAGAXI"};
  if (m.air_gaps.size() > 1)
    return Error{std::format("PAIRGAP {} and {} both use the air-gap element; "
                             "one air gap per model is supported",
                             m.air_gaps.begin()->first,
                             std::next(m.air_gaps.begin())->first)};
  // Air-gap elements mark the gap; under the air-gap element they are not
  // field cells, so sets that act on field cells may not contain them.
  std::set<int> ids, gap_elements;
  for (const auto &c : m.elements) {
    if (c.id <= 0 || !ids.insert(c.id).second)
      return Error{"Invalid/duplicate magnetic element id"};
    if (m.air_gaps.contains(c.property))
      gap_elements.insert(c.id);
    else if (!m.properties.contains(c.property))
      return Error{std::format("Magnetic element {} references undefined "
                               "property {}",
                               c.id, c.property)};
  }
  auto gap_member = [&](int set, const std::string &use) -> Status {
    for (int id : m.element_sets.at(set))
      if (gap_elements.contains(id))
        return Error{std::format("{} ESET {} contains air-gap element {} "
                                 "(PAIRGAP {}); air-gap elements are not field "
                                 "cells",
                                 use, set, id,
                                 std::find_if(m.elements.begin(), m.elements.end(),
                                              [id](const Element &c) {
                                                return c.id == id;
                                              })->property)};
    return {};
  };
  std::set<int> wireids;
  for (const auto &w : m.wires) {
    if (w.id <= 0 || ids.contains(w.id) || !wireids.insert(w.id).second)
      return Error{"CWIRE ids must be unique across volume and wire elements"};
    if (!m.nodes.contains(w.first) || !m.nodes.contains(w.second) ||
        w.first == w.second)
      return Error{"CWIRE requires two distinct existing grids"};
    auto it = m.wire_properties.find(w.property);
    if (it == m.wire_properties.end())
      return Error{"CWIRE references missing PWIRE"};
    const auto &p = it->second;
    if (!(p.width > 0) || (p.rectangular && !(p.height > 0)) ||
        (!p.rectangular && p.height != 0) || !std::isfinite(p.width) ||
        !std::isfinite(p.height) || !m.conductivity.contains(p.material))
      return Error{"PWIRE requires valid section dimensions and MATCOND"};
    if (w.orientation_node && !m.nodes.contains(w.orientation_node))
      return Error{"CWIRE orientation grid does not exist"};
    if (p.rectangular && !w.orientation_node && !w.orientation)
      return Error{"Rectangular CWIRE requires G0 or an orientation vector"};
  }
  // Sources and circuits share one ID namespace.
  std::set<int> sources, circuits;
  auto unique_source = [&](int id, const char *card) -> Status {
    if (id <= 0 || !sources.insert(id).second)
      return Error{std::format("{} ID {} must be positive and unique among "
                               "COIL, COIL2D, WIREPATH and CIRCUIT IDs",
                               card, id)};
    return {};
  };
  for (const auto &c : m.coils) {
    if (auto e = unique_source(c.id, "COIL"))
      return e;
    if (!(c.turns > 0) || !std::isfinite(c.turns) ||
        !m.element_sets.contains(c.elements) ||
        (!c.plane_cut && !m.face_sets.contains(c.start)) ||
        (c.end && !m.face_sets.contains(c.end)))
      return Error{"Invalid COIL definition"};
    if (c.plane_cut &&
        (c.start || c.end ||
         (c.cut_coordinates &&
          (!m.coordinates.contains(c.cut_coordinates) ||
           m.coordinates.at(c.cut_coordinates).type != 'R'))))
      return Error{std::format("COIL {} CUT requires a closed coil and a "
                               "CORD2R (or 0 for basic)",
                               c.id)};
    std::set<int> selected;
    for (int id : m.element_sets.at(c.elements))
      if (!ids.contains(id) || !selected.insert(id).second)
        return Error{"COIL ESET contains missing or duplicate elements"};
  }
  for (const auto &c : m.planar_coils) {
    if (c.id > 0 && sources.contains(c.id) &&
        std::count_if(m.planar_coils.begin(), m.planar_coils.end(),
                      [&](const PlanarCoil &o) { return o.id == c.id; }) > 1)
      return Error{std::format("COIL2D {} is repeated; COIL2D IDs are unique, "
                               "group legs with CIRCUIT",
                               c.id)};
    if (auto e = unique_source(c.id, "COIL2D"))
      return e;
    if (c.turns == 0 || !std::isfinite(c.turns) ||
        !m.element_sets.contains(c.elements))
      return Error{"Invalid COIL2D definition"};
    if (auto e = gap_member(c.elements, std::format("COIL2D {}", c.id)))
      return e;
    std::set<int> selected;
    for (int id : m.element_sets.at(c.elements))
      if (!ids.contains(id) || !selected.insert(id).second)
        return Error{"COIL2D ESET contains missing or duplicate elements"};
  }
  for (const auto &w : m.wire_paths) {
    if (auto e = unique_source(w.id, "WIREPATH"))
      return e;
    if (!wireids.contains(w.element))
      return Error{std::format("WIREPATH {} references undefined CWIRE {}",
                               w.id, w.element)};
  }
  std::set<int> members;
  for (const auto &[id, c] : m.circuits) {
    if (auto e = unique_source(id, "CIRCUIT"))
      return e;
    circuits.insert(id);
    if (c.parallel < 1 || c.members.empty())
      return Error{std::format("CIRCUIT {} requires PARALLEL >= 1 and at "
                               "least one member",
                               id)};
  }
  for (const auto &[id, c] : m.circuits)
    for (auto [source, polarity] : c.members) {
      if (!sources.contains(source) || circuits.contains(source))
        return Error{std::format("CIRCUIT {} member {} is not a COIL, COIL2D "
                                 "or WIREPATH",
                                 id, source)};
      if (std::abs(polarity) != 1)
        return Error{std::format("CIRCUIT {} polarity of member {} must be "
                                 "+1 or -1",
                                 id, source)};
      if (!members.insert(source).second)
        return Error{std::format("Source {} belongs to more than one "
                                 "CIRCUIT or is listed twice",
                                 source)};
    }
  for (const auto &c : m.currents) {
    if (c.set <= 0 || !std::isfinite(c.amperes))
      return Error{"ICIRC requires a positive SID and a finite current"};
    if (members.contains(c.target))
      return Error{std::format("ICIRC {} targets source {}, which belongs to "
                               "a CIRCUIT; give the current on the circuit",
                               c.set, c.target)};
    if (!sources.contains(c.target))
      return Error{std::format("ICIRC {} references undefined circuit or "
                               "source {}",
                               c.set, c.target)};
  }
  ids.clear();
  for (const auto &probe : m.probes) {
    if (probe.id <= 0 || !ids.insert(probe.id).second ||
        !probe.position.allFinite() || probe.coordinates != 0)
      return Error{"MPROBE requires a unique positive ID and finite, "
                   "resolved coordinates"};
    if (probe.output_coordinates &&
        !m.coordinates.contains(probe.output_coordinates))
      return Error{std::format("MPROBE {} CD references undefined "
                               "coordinates {}",
                               probe.id, probe.output_coordinates)};
  }
  std::set<int> element_ids;
  for (const auto &c : m.elements)
    element_ids.insert(c.id);
  ids.clear();
  for (const auto &force : m.forces) {
    if (force.id <= 0 || !ids.insert(force.id).second)
      return Error{"MFORCE requires a unique positive ID"};
    if (force.body ? !m.element_sets.contains(force.set)
                   : !m.face_sets.contains(force.set))
      return Error{std::format("MFORCE {} references undefined {} {}",
                               force.id, force.body ? "ESET" : "FSET",
                               force.set)};
    if (force.coordinates && !m.coordinates.contains(force.coordinates))
      return Error{std::format("MFORCE {} references undefined coordinates {}",
                               force.id, force.coordinates)};
    if (force.layers < 1 || force.layers > 64)
      return Error{std::format("MFORCE {} LAYER must be 1 to 64", force.id)};
    if (force.body)
      if (auto e = gap_member(force.set, std::format("MFORCE {} body", force.id)))
        return e;
    if (force.body)
      for (int id : m.element_sets.at(force.set))
        if (!element_ids.contains(id))
          return Error{std::format("MFORCE {} body ESET references undefined "
                                   "element {}",
                                   force.id, id)};
  }
  for (const auto &[id, motion] : m.motions) {
    if (!m.options.planar)
      return Error{std::format("MOTION {} requires MAG2D; 3D motion is not "
                               "supported",
                               id)};
    if (!m.element_sets.contains(motion.set))
      return Error{std::format("MOTION {} references undefined ESET {}", id,
                               motion.set)};
    for (int element : m.element_sets.at(motion.set))
      if (!element_ids.contains(element))
        return Error{std::format("MOTION {} ESET {} references undefined "
                                 "element {}",
                                 id, motion.set, element)};
    if (auto e = gap_member(motion.set, std::format("MOTION {}", id)))
      return e;
    if (motion.coordinates && !m.coordinates.contains(motion.coordinates))
      return Error{std::format("MOTION {} references undefined coordinates {}",
                               id, motion.coordinates)};
    if (motion.count < 1 || (motion.count > 1 && motion.step == 0))
      return Error{std::format("MOTION {} requires COUNT >= 1, and a nonzero "
                               "STEP when COUNT > 1",
                               id)};
    const Vector axis =
        motion.coordinates
            ? Vector(coordinate_axes(m.coordinates.at(motion.coordinates)).col(2))
            : Vector(Vector::UnitZ());
    if (axis.cross(plane_normal).norm() > 1e-9)
      return Error{std::format("MOTION {}: the z axis of CID {} must be parallel "
                               "to the MAG2D plane normal, the axis of rotation",
                               id, motion.coordinates)};
    if (m.air_gaps.empty() && !m.meshed_air_gaps.empty())
      return Error{std::format(
          "MOTION {} needs the air-gap element between the moving and fixed "
          "parts, but PAIRGAP {} is solved as meshed air by {}; meshed air "
          "cannot slide. To check one position, solve the static deck with the "
          "moving grids rotated to it",
          id, m.meshed_air_gaps.begin()->first, m.meshed_air_gaps.begin()->second)};
    if (m.air_gaps.empty())
      return Error{std::format("MOTION {} needs a PAIRGAP air gap between the "
                               "moving and fixed parts",
                               id)};
  }
  if (const auto &period = m.period) {
    if (period->sectors < 2)
      return Error{std::format("MPERIOD {} requires SECTORS >= 2: the number of "
                               "sectors in the whole machine",
                               period->id)};
    if (period->anti && period->sectors % 2 != 0)
      return Error{std::format(
          "MPERIOD {} ANTI with {} sectors: the potential would change sign "
          "an odd number of times around the machine. Anti-periodic sectors "
          "come in an even number",
          period->id, period->sectors)};
    for (int set : {period->first, period->second})
      if (!m.face_sets.contains(set))
        return Error{std::format("MPERIOD {} references undefined FSET {}",
                                 period->id, set)};
    if (period->first == period->second)
      return Error{std::format("MPERIOD {} names FSET {} for both sides",
                               period->id, period->first)};
    if (period->coordinates && !m.coordinates.contains(period->coordinates))
      return Error{std::format("MPERIOD {} references undefined coordinates {}",
                               period->id, period->coordinates)};
    const Vector axis =
        period->coordinates
            ? Vector(coordinate_axes(m.coordinates.at(period->coordinates)).col(2))
            : Vector(Vector::UnitZ());
    if (axis.cross(plane_normal).norm() > 1e-9)
      return Error{std::format("MPERIOD {}: the z axis of CID {} must be parallel "
                               "to the MAG2D plane normal, the sector axis",
                               period->id, period->coordinates)};
  }
  for (const auto &[id, bc] : m.boundaries) {
    if (const auto missing = std::ranges::find_if(
            bc.normal, [&](int set) { return !m.face_sets.contains(set); });
        missing != bc.normal.end())
      return Error{std::format("MBC {} NORMAL references undefined FSET {}", id,
                               *missing)};
    if (bc.faces && !m.face_sets.contains(bc.faces))
      return Error{std::format("MBC {} references undefined FSET {}", id,
                               bc.faces)};
  }
  for (const auto &[id, drive] : m.drives) {
    const auto count = drive.phases.size();
    if (drive.poles < 1)
      return Error{std::format("DRIVE {} requires POLES >= 1 pole pairs", id)};
    if (drive.peak < 0)
      return Error{std::format("DRIVE {} requires IPEAK >= 0", id)};
    if (count < 3)
      return Error{std::format("DRIVE {} lists {} phase circuits; it needs at "
                               "least 3, in phase sequence",
                               id, count)};
    if (drive.six && count != 3)
      return Error{std::format("DRIVE {} SIX is the three-phase six-step "
                               "waveform and lists {} phase circuits",
                               id, count)};
    std::set<int> listed;
    for (int phase : drive.phases) {
      if (!listed.insert(phase).second)
        return Error{std::format("DRIVE {} lists circuit {} twice", id, phase)};
      if (members.contains(phase))
        return Error{std::format("DRIVE {} phase {} is a source that belongs to "
                                 "a CIRCUIT; name the circuit",
                                 id, phase)};
      if (!sources.contains(phase))
        return Error{std::format("DRIVE {} references undefined circuit or "
                                 "source {}",
                                 id, phase)};
    }
  }
  ids.clear();
  for (const auto &sc : m.subcases) {
    if (sc.id <= 0 || !ids.insert(sc.id).second)
      return Error{"Subcase ids must be unique and positive"};
    if (sc.motion && !m.motions.contains(sc.motion))
      return Error{std::format("Subcase {} references undefined MOTION {}",
                               sc.id, sc.motion)};
    if (sc.boundary && !m.boundaries.contains(sc.boundary))
      return Error{"Subcase references undefined MBC"};
    if (sc.field.set && !m.element_sets.contains(sc.field.set))
      return Error{std::format("Subcase {} BFIELD references undefined ESET {}",
                               sc.id, sc.field.set)};
    if (sc.field.set)
      if (auto e = gap_member(sc.field.set, std::format("Subcase {} BFIELD", sc.id)))
        return e;
    if (sc.field.set)
      for (int id : m.element_sets.at(sc.field.set))
        if (!element_ids.contains(id))
          return Error{std::format("Subcase {} BFIELD ESET {} references "
                                   "undefined field element {}",
                                   sc.id, sc.field.set, id)};
    auto r = load_factors(m, sc.load);
    if (auto *e = std::get_if<Error>(&r))
      return *e;
    if (!sc.drive)
      continue;
    if (!m.drives.contains(sc.drive))
      return Error{std::format("Subcase {} references undefined DRIVE {}", sc.id,
                               sc.drive)};
    if (!sc.motion)
      return Error{std::format("Subcase {} DRIVE {} needs MOTION in the same "
                               "subcase: the phase currents follow the rotor "
                               "position",
                               sc.id, sc.drive)};
    const auto &drive = m.drives.at(sc.drive);
    const auto &factors = std::get<std::map<int, double>>(r);
    const auto loaded = std::find_if(m.currents.begin(), m.currents.end(),
                                     [&](const Current &c) {
                                       return factors.contains(c.set) &&
                                              std::find(drive.phases.begin(),
                                                        drive.phases.end(),
                                                        c.target) != drive.phases.end();
                                     });
    if (loaded != m.currents.end())
      return Error{std::format("Subcase {}: circuit {} is a phase of DRIVE {} and "
                               "also has an ICIRC current in LOAD {} (set {}); a "
                               "DRIVE phase takes its current from DRIVE alone",
                               sc.id, loaded->target, sc.drive, sc.load, loaded->set)};
  }
  for (const auto &[id, g] : m.nodes)
    if (id <= 0 || !g.position.allFinite())
      return Error{"Invalid GRID"};
  return {};
}

Result<std::map<int, double>> load_factors(const Model &m, int set) {
  std::map<int, double> factors;
  std::set<int> stack;
  std::function<Status(int, double)> visit = [&](int id,
                                                 double scale) -> Status {
    if (id == 0)
      return {};
    if (!stack.insert(id).second)
      return Error{"Cyclic LOAD combination at SID " + std::to_string(id)};
    auto it = m.combinations.find(id);
    if (it != m.combinations.end()) {
      if (it->second.terms.empty())
        return Error{"Empty LOAD combination"};
      for (auto [factor, child] : it->second.terms) {
        if (child <= 0)
          return Error{"LOAD references invalid SID"};
        if (auto e = visit(child, scale * it->second.scale * factor))
          return e;
      }
    } else {
      if (std::none_of(m.currents.begin(), m.currents.end(),
                       [&](const Current &c) { return c.set == id; }))
        return Error{"Undefined current LOAD set " + std::to_string(id)};
      factors[id] += scale;
    }
    stack.erase(id);
    return {};
  };
  if (auto e = visit(set, 1))
    return *e;
  return factors;
}
} // namespace vibestran::magnetic
