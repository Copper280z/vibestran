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
      std::string filename = trim(stripped.substr(7));
      if (filename.size() >= 2 &&
          (filename.front() == '\'' || filename.front() == '\"') &&
          filename.back() == filename.front())
        filename = filename.substr(1, filename.size() - 2);
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
Matrix axes(const CoordinateSystem &cs) {
  Vector z = (cs.b - cs.a).normalized(), y = z.cross(cs.c - cs.a).normalized();
  Matrix r;
  r.col(0) = y.cross(z);
  r.col(1) = y;
  r.col(2) = z;
  return r;
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
  return cs.a + axes(cs) * p;
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
  for (auto &p : m.probes)
    p.position *= m.options.length_scale;
  for (auto &f : m.forces)
    f.origin *= m.options.length_scale;
  m.options.wire_step *= m.options.length_scale;
  m.options.depth *= m.options.length_scale;
  return {};
}
Result<Model> parse_lines(const std::vector<Line> &lines) {
  Model m;
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
        auto equals = text.find('=');
        if (equals == std::string::npos)
          return Error{lines[i].location +
                       ": unsupported executive/case control"};
        auto key = trim(text.substr(0, equals)),
             value = trim(text.substr(equals + 1));
        auto &sc = in_case ? current : global;
        Reader r{{key, value}, lines[i].location, {}};
        if (key == "LOAD")
          sc.load = r.integer(1);
        else if (key == "MAGBC")
          sc.boundary = r.integer(1);
        else if (key == "LABEL")
          sc.label = trim(lines[i].text.substr(equals + 1));
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
    } else if (key == "PSOLID" || key == "PMAG2D") {
      int id = r.integer(1), mid = r.integer(2);
      if (r.integer(3) != 0)
        r.error = Error{r.where + ": use MATMAG CID for magnetic orientation"};
      r.last(key == "PSOLID" ? 3 : 2);
      insert_unique(m.properties, id, mid, r);
    } else if (key == "MAG2D") {
      if (m.options.planar)
        r.error = Error{r.where + ": duplicate MAG2D"};
      m.options.planar = true;
      m.options.plane_coordinates = r.integer(1);
      m.options.depth = r.real(2);
      r.last(2);
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
      a.remanence = r.vector(4);
      a.coordinates = r.integer(7);
      r.last(7);
      insert_unique(m.materials, id, a, r);
    } else if (key == "BHPOINT") {
      int id = r.integer(1);
      double b = r.real(2), h = r.real(3);
      r.last(3);
      m.bh_tables[id].emplace_back(b, h);
    } else if (key == "MATCOND") {
      int id = r.integer(1);
      double sigma = r.real(2);
      r.last(2);
      insert_unique(m.conductivity, id, sigma, r);
    } else if (key == "ESET") {
      int id = r.integer(1);
      auto values = r.list(2);
      auto &out = m.element_sets[id];
      out.insert(out.end(), values.begin(), values.end());
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
      c.start = r.integer(4);
      c.end = r.integer(5);
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
    } else if (key == "ICOIL" || key == "IWIRE") {
      Current c;
      c.wire = key == "IWIRE";
      c.set = r.integer(1);
      if (c.wire) {
        c.amperes = r.real(2);
        c.targets = r.list(3);
      } else {
        c.targets = {r.integer(2)};
        c.amperes = r.real(3);
        r.last(3);
      }
      m.currents.push_back(std::move(c));
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
      int id = r.integer(1);
      Boundary bc;
      bc.faces = r.integer(2);
      auto kind = upper(r.get(3));
      if (kind == "UNIFORM")
        bc.field = r.vector(4);
      else if (kind != "ZERO")
        r.error = Error{r.where + ": MBC type must be ZERO or UNIFORM"};
      r.last(6);
      insert_unique(m.boundaries, id, bc, r);
    } else if (key == "MPROBE") {
      Probe p;
      p.id = r.integer(1);
      p.position = r.vector(2);
      r.last(4);
      m.probes.push_back(p);
    } else if (key == "MFORCE") {
      ForceSurface f;
      f.id = r.integer(1);
      f.faces = r.integer(2);
      f.origin = r.vector(3);
      r.last(5);
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
      return Error{"MAG2D requires CTRIA3/CQUAD4 only; 3D MAGSTAT requires "
                   "CTETRA4/CHEXA8/CPENTA6"};
  if (!sol)
    return Error{"Missing SOL MAGSTAT"};
  if (in_case)
    m.subcases.push_back(current);
  else
    m.subcases.push_back(global);
  if (!(m.options.length_scale > 0))
    return Error{"MAGLEN must be positive"};
  if (auto e = resolve_coordinates(m))
    return *e;
  if (auto e = validate(m))
    return *e;
  return m;
}
} // namespace

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
  std::string s;
  while (std::getline(in, s)) {
    s = upper(trim(s.substr(0, s.find('$'))));
    if (const auto solution = solution_name(s))
      return *solution == "MAGSTAT";
    if (s.starts_with("INCLUDE")) {
      in.clear();
      in.seekg(0);
      std::vector<Line> lines;
      std::set<std::filesystem::path> stack;
      std::error_code ec;
      stack.insert(std::filesystem::weakly_canonical(path, ec));
      if (auto error =
              read_lines(in, path.parent_path(), path.string(), lines, stack))
        return *error;
      for (const auto &line : lines)
        if (const auto solution = solution_name(upper(trim(line.text))))
          return *solution == "MAGSTAT";
      return false;
    }
    if (s.starts_with("BEGIN"))
      break;
  }
  return false;
}

Status validate(const Model &m) {
  if (m.elements.empty())
    return Error{"MAGSTAT requires a magnetic mesh"};
  if (m.subcases.empty())
    return Error{"MAGSTAT requires at least one subcase"};
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
      return Error{
          "BHPOINT table must begin at (0,0) and contain at least two points"};
    for (size_t i = 1; i < curve.size(); ++i)
      if (!(curve[i].first > curve[i - 1].first &&
            curve[i].second > curve[i - 1].second) ||
          !std::isfinite(curve[i].first) || !std::isfinite(curve[i].second))
        return Error{
            "BHPOINT B and H must both be strictly increasing in deck order"};
  }
  for (const auto &[id, mat] : m.materials) {
    if (!(mat.mur > 0) || !std::isfinite(mat.mur) || !mat.remanence.allFinite())
      return Error{"Invalid MATMAG " + std::to_string(id)};
    if (mat.bh_table && (!m.bh_tables.contains(mat.bh_table) ||
                         mat.remanence.squaredNorm() != 0))
      return Error{"MATMAG nonlinear curve must exist and cannot be combined "
                   "with remanence"};
    if (mat.coordinates && !m.coordinates.contains(mat.coordinates))
      return Error{"MATMAG references missing coordinates"};
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
    return Error{"MAG2D CID must reference CORD2R"};
  if (m.options.planar && (!m.wires.empty() || !m.coils.empty()))
    return Error{"MAG2D uses axial COIL2D regions; CWIRE and volumetric COIL "
                 "require 3D"};
  if (!m.options.planar && !m.planar_coils.empty())
    return Error{"COIL2D requires MAG2D"};
  std::set<int> ids;
  for (const auto &c : m.elements) {
    if (c.id <= 0 || !ids.insert(c.id).second)
      return Error{"Invalid/duplicate magnetic element id"};
    auto p = m.properties.find(c.property);
    if (p == m.properties.end() || !m.materials.contains(p->second))
      return Error{
          "Magnetic element must reference a property with a MATMAG material"};
  }
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
  std::set<int> coilids;
  for (const auto &c : m.coils) {
    if (c.id <= 0 || !coilids.insert(c.id).second || !(c.turns > 0) ||
        !std::isfinite(c.turns) || !m.element_sets.contains(c.elements) ||
        !m.face_sets.contains(c.start) ||
        (c.end && !m.face_sets.contains(c.end)))
      return Error{"Invalid COIL definition"};
    std::set<int> selected;
    for (int id : m.element_sets.at(c.elements))
      if (!ids.contains(id) || !selected.insert(id).second)
        return Error{"COIL ESET contains missing or duplicate elements"};
  }
  for (const auto &c : m.planar_coils) {
    if (c.id <= 0 || c.turns == 0 || !std::isfinite(c.turns) ||
        !m.element_sets.contains(c.elements))
      return Error{"Invalid COIL2D definition"};
    coilids.insert(c.id);
    std::set<int> selected;
    for (int id : m.element_sets.at(c.elements))
      if (!ids.contains(id) || !selected.insert(id).second)
        return Error{"COIL2D ESET contains missing or duplicate elements"};
  }
  for (const auto &c : m.currents) {
    if (c.set <= 0 || c.targets.empty() || !std::isfinite(c.amperes))
      return Error{"Current requires a positive SID and targets"};
    std::set<int> targets;
    for (int target : c.targets)
      if (!(c.wire ? wireids.contains(target) : coilids.contains(target)) ||
          !targets.insert(target).second)
        return Error{"Current references missing/duplicate target"};
  }
  ids.clear();
  for (const auto &probe : m.probes)
    if (probe.id <= 0 || !ids.insert(probe.id).second ||
        !probe.position.allFinite())
      return Error{
          "MPROBE requires a unique positive ID and finite coordinates"};
  ids.clear();
  for (const auto &force : m.forces)
    if (force.id <= 0 || !ids.insert(force.id).second ||
        !force.origin.allFinite() || !m.face_sets.contains(force.faces))
      return Error{"MFORCE requires a unique positive ID, finite origin and "
                   "defined FSET"};
  ids.clear();
  for (const auto &sc : m.subcases) {
    if (sc.id <= 0 || !ids.insert(sc.id).second)
      return Error{"Subcase ids must be unique and positive"};
    if (sc.boundary && !m.boundaries.contains(sc.boundary))
      return Error{"Subcase references undefined MBC"};
    auto r = load_factors(m, sc.load);
    if (auto *e = std::get_if<Error>(&r))
      return *e;
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
