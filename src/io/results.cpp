// src/io/results.cpp
#include "io/results.hpp"
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <cmath>
#include <numbers>
#include <format>
#include <algorithm>
#include <unordered_map>

namespace vibestran {

namespace {

constexpr int kF06PageWidth = 132;

std::string center_text(std::string_view text, int width = kF06PageWidth) {
    if (static_cast<int>(text.size()) >= width) return std::string(text);
    const int pad = (width - static_cast<int>(text.size())) / 2;
    return std::string(static_cast<std::size_t>(pad), ' ') + std::string(text);
}

std::string page_line(std::string_view text) {
    return center_text(text) + "\n";
}

std::string banner_line(std::string_view left, std::string_view right,
                        int width = kF06PageWidth) {
    if (static_cast<int>(left.size() + right.size() + 1) >= width)
        return std::string(left) + " " + std::string(right) + "\n";

    const int gap = width - static_cast<int>(left.size()) - static_cast<int>(right.size());
    return std::string(left) + std::string(static_cast<std::size_t>(gap), ' ')
         + std::string(right) + "\n";
}

const SubCase* find_model_subcase(const Model& model, int subcase_id) {
    const auto it = std::find_if(
        model.analysis.subcases.begin(), model.analysis.subcases.end(),
        [&](const SubCase& sc) { return sc.id == subcase_id; });
    return (it == model.analysis.subcases.end()) ? nullptr : &*it;
}

bool contains_plate_stress_type(const SubCaseResults& sc, ElementType etype) {
    return std::any_of(
        sc.plate_stresses.begin(), sc.plate_stresses.end(),
        [&](const auto& ps) { return ps.etype == etype; });
}

bool contains_solid_stress_type(const SubCaseResults& sc, ElementType etype) {
    return std::any_of(
        sc.solid_stresses.begin(), sc.solid_stresses.end(),
        [&](const auto& ss) { return ss.etype == etype; });
}

bool contains_line_stresses(const SubCaseResults& sc) {
    return !sc.line_stresses.empty();
}

bool contains_plate_nodal_stress_type(const SubCaseResults& sc, ElementType etype) {
    return std::any_of(
        sc.plate_stresses.begin(), sc.plate_stresses.end(),
        [&](const auto& ps) { return ps.etype == etype && !ps.nodal.empty(); });
}

bool contains_solid_nodal_stress_type(const SubCaseResults& sc, ElementType etype) {
    return std::any_of(
        sc.solid_stresses.begin(), sc.solid_stresses.end(),
        [&](const auto& ss) { return ss.etype == etype && !ss.nodal.empty(); });
}

int plate_vertex_count(const ElementType etype) {
    return (etype == ElementType::CQUAD4) ? 4 : 3;
}

int solid_vertex_count(const ElementType etype) {
    switch (etype) {
    case ElementType::CHEXA8:
        return 8;
    case ElementType::CTETRA4:
        return 4;
    case ElementType::CTETRA10:
        return 4;
    case ElementType::CPENTA6:
        return 6;
    default:
        return 0;
    }
}

// ── Fixed-column row builder for MYSTRAN-format tables ───────────────────────
//
// The MYSTRAN validation suite parser (f06_query.py) reads values from
// hard-coded 1-based column windows, so these tables must place text at exact
// columns.

class F06Row {
  public:
    /// Right-justify text in the window [col, col+width-1] (1-based).
    F06Row& put(int col, int width, std::string_view text) {
        const int end = col + width;
        if (static_cast<int>(s_.size()) < end)
            s_.resize(static_cast<std::size_t>(end), ' ');
        if (static_cast<int>(text.size()) > width) text = text.substr(0, width);
        int start = (col - 1) + (width - static_cast<int>(text.size()));
        s_.replace(static_cast<std::size_t>(start), text.size(), text);
        return *this;
    }
    /// Left-justify text starting at column col (overwrites).
    F06Row& put_left(int col, std::string_view text) {
        const std::size_t at = static_cast<std::size_t>(col - 1);
        if (s_.size() < at) s_.resize(at, ' ');
        if (at + text.size() > s_.size())
            s_.resize(at + text.size(), ' ');
        s_.replace(at, text.size(), text);
        return *this;
    }
    [[nodiscard]] const std::string& str() const { return s_; }

  private:
    std::string s_;
};

/// 5-decimal scientific (6 significant digits), uppercase E, e.g.
/// "1.00000E+02" (11 chars, 12 with sign). Widths match MYSTRAN so negative
/// values still fit the parser's 12-wide column windows.
std::string fmt_e(const double v) {
    return std::format("{:.5E}", v);
}

/// 2-decimal fixed, e.g. "-90.00".
std::string fmt_angle(const double v) {
    return std::format("{:.2f}", v);
}

/// Bending stress factor for fiber distance z and thickness t:
/// sigma(z) = sigma_membrane + factor*M with factor = 12*z/t^3 (0 if t<=0).
double fiber_factor(double z, double t) {
    return (t > 0.0) ? 12.0 * z / (t * t * t) : 0.0;
}

} // namespace

// ── Principal stress helpers ──────────────────────────────────────────────────

void compute_principal_2d(double sx, double sy, double sxy,
                          double &major, double &minor, double &angle_deg) {
    double avg  = (sx + sy) * 0.5;
    double diff = (sx - sy) * 0.5;
    double R    = std::sqrt(diff * diff + sxy * sxy);
    major     = avg + R;
    minor     = avg - R;
    // angle of major principal axis from x in degrees
    angle_deg = 0.5 * std::atan2(2.0 * sxy, sx - sy) * (180.0 / std::numbers::pi);
}

// Jacobi eigenvalue algorithm for 3×3 symmetric matrix.
// On entry:  a[3][3] symmetric, v = identity.
// On exit:   a[i][i] = eigenvalue i, v[i] = eigenvector i (row-wise).
static void jacobi3(double a[3][3], double v[3][3]) {
    // Initialise v to identity
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            v[i][j] = (i == j) ? 1.0 : 0.0;

    for (int iter = 0; iter < 50; ++iter) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        double maxVal = std::abs(a[0][1]);
        if (std::abs(a[0][2]) > maxVal) { p = 0; q = 2; maxVal = std::abs(a[0][2]); }
        if (std::abs(a[1][2]) > maxVal) { p = 1; q = 2; maxVal = std::abs(a[1][2]); }
        if (maxVal < 1e-14) break;

        // Compute rotation angle
        double dif = a[q][q] - a[p][p];
        if (std::abs(a[p][q]) > 1e-300) {
            double theta = dif / (2.0 * a[p][q]);
            double t = 1.0 / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
            if (theta < 0.0) t = -t;
            double c = 1.0 / std::sqrt(t * t + 1.0);
            double s = t * c;

            // Update a
            a[p][p] -= t * a[p][q];
            a[q][q] += t * a[p][q];
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for (int r = 0; r < 3; ++r) {
                if (r != p && r != q) {
                    double ap = c * a[r][p] - s * a[r][q];
                    double aq = s * a[r][p] + c * a[r][q];
                    a[r][p] = ap; a[p][r] = ap;
                    a[r][q] = aq; a[q][r] = aq;
                }
            }

            // Update eigenvectors
            for (int r = 0; r < 3; ++r) {
                double vp = c * v[p][r] - s * v[q][r];
                double vq = s * v[p][r] + c * v[q][r];
                v[p][r] = vp;
                v[q][r] = vq;
            }
        }
    }
}

void compute_principal_3d(double sx, double sy, double sz,
                          double sxy, double syz, double szx,
                          double p[3], double v[3][3]) {
    double a[3][3] = {
        {sx,  sxy, szx},
        {sxy, sy,  syz},
        {szx, syz, sz }
    };
    jacobi3(a, v);

    // Extract eigenvalues and sort descending (major → minor)
    p[0] = a[0][0];
    p[1] = a[1][1];
    p[2] = a[2][2];

    // Sort by descending eigenvalue; keep eigenvectors in sync
    // Simple 3-element insertion sort
    for (int i = 1; i < 3; ++i) {
        for (int j = i; j > 0 && p[j] > p[j-1]; --j) {
            std::swap(p[j], p[j-1]);
            for (int k = 0; k < 3; ++k)
                std::swap(v[j][k], v[j-1][k]);
        }
    }
}

// ── F06 writer ────────────────────────────────────────────────────────────────
//
// The layout of the results blocks follows MYSTRAN's F06 output so the
// MYSTRAN_Validation suite parser (f06_query.py) can read our files.

void F06Writer::write(const SolverResults& results, const Model& model,
                      const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) throw SolverError(std::format("Cannot write F06: {}", path.string()));
    write(results, model, f);
}

void F06Writer::write(const SolverResults& results, const Model& model,
                      std::ostream& out) {
    write_header(out);

    // Map element ID to shell property for fiber-distance stress recovery.
    std::unordered_map<ElementId, const PShell*> shell_props;
    for (const auto& elem : model.elements) {
        const auto it = model.properties.find(elem.pid);
        if (it == model.properties.end()) continue;
        if (const auto* ps = std::get_if<PShell>(&it->second))
            shell_props.emplace(elem.id, ps);
    }

    for (const auto& sc : results.subcases) {
        const SubCase* msc = find_model_subcase(model, sc.id);
        const bool do_disp = (msc != nullptr) && msc->disp_print;
        const bool do_stress_any =
            (msc != nullptr) &&
            (msc->stress_print || msc->stress_corner_print);
        const bool do_corner = (msc != nullptr) && msc->stress_corner_print;
        const bool do_gpstress = (msc != nullptr) && msc->gpstress_print;
        const bool do_spc_force = (msc != nullptr) && msc->spc_force_print;

        if (do_disp) write_displacement_table(sc, out);
        if (do_spc_force) write_spc_force_table(sc, out);
        if (do_stress_any) {
            write_bar_stress_table(sc, out);
            write_shell_stress_table(sc, out, ElementType::CQUAD4, do_corner,
                                     shell_props);
            write_shell_stress_table(sc, out, ElementType::CTRIA3, do_corner,
                                     shell_props);
            write_solid_stress_table(sc, out, ElementType::CHEXA8, do_corner);
            write_solid_stress_table(sc, out, ElementType::CTETRA4, do_corner);
            write_solid_stress_table(sc, out, ElementType::CTETRA10,
                                     do_corner);
            write_solid_stress_table(sc, out, ElementType::CPENTA6, do_corner);
        }
        if (do_gpstress) {
            write_quad4_gpstress_table(sc, out);
            write_tria3_gpstress_table(sc, out);
            write_solid_gpstress_table(sc, out, ElementType::CHEXA8);
            write_solid_gpstress_table(sc, out, ElementType::CTETRA4);
            write_solid_gpstress_table(sc, out, ElementType::CTETRA10);
            write_solid_gpstress_table(sc, out, ElementType::CPENTA6);
        }
    }
    out << "\n\n                     * * * END OF JOB * * *\n\n";
}

void F06Writer::write_subcase_header(std::ostream& out, const int subcase_id) {
    // Matches MYSTRAN FORMAT(' OUTPUT FOR SUBCASE ',I8). The validation
    // suite parser reads the subcase number from columns 21-28 and expects
    // this line before (nearly) every results block.
    out << " OUTPUT FOR SUBCASE " << std::setw(8) << subcase_id << "\n";
}

void F06Writer::write_header(std::ostream& out) {
    // Date/time for the header
    std::time_t t = std::time(nullptr);
    char date_buf[32];
    std::strftime(date_buf, sizeof(date_buf), "%B %e, %Y", std::localtime(&t));

    out << "1" << banner_line("V I B E S T R A N", date_buf, kF06PageWidth - 1);
    out << "0" << std::string(static_cast<std::size_t>(kF06PageWidth - 1), ' ') << "\n";
    // Line 3 must start with " VIBESTRAN Version" for the validation suite
    // parser to recognise the file (see MYSTRAN_Validation/f06_query.py).
    out << " VIBESTRAN Version 1.0.0   F06 MYSTRAN-COMPATIBLE OUTPUT\n";
    out << page_line("S O L   1 0 1   L I N E A R   S T A T I C   A N A L Y S I S");
    out << "\n";
}

void F06Writer::write_displacement_table(const SubCaseResults& sc,
                                         std::ostream& out) {
    if (sc.displacements.empty()) return;

    write_subcase_header(out, sc.id);
    if (!sc.label.empty())
        out << " " << sc.label << "\n";
    out << "\n";
    out << "                                                       D I S P L A C E M E N T S\n";
    out << "                                              (in global coordinate system at each grid)\n";
    out << "           GRID     COORD      T1            T2            T3            R1            R2            R3\n";
    out << "                     SYS\n";

    for (const auto& nd : sc.displacements) {
        out << std::setw(15) << nd.node.value;
        out << std::setw(9) << 0; // global coordinate system
        out << std::uppercase;
        for (int i = 0; i < 6; ++i) {
            const double v = nd.d[i];
            if (v == 0.0) {
                out << "  0.0         ";
            } else {
                out << std::setw(14) << std::setprecision(6) << std::scientific << v;
            }
        }
        out << std::nouppercase;
        out << "\n";
    }
    out << "\n";
}

void F06Writer::write_spc_force_table(const SubCaseResults& sc,
                                         std::ostream& out) {
    if (sc.spc_forces.empty()) return;

    // MYSTRAN layout (the validation suite parser reads the value row
    // windows: GRID cols 8-15, COORD SYS cols 16-24, then T1/R1... at cols
    // 26/40/54/68/82/96, each 13 wide).  A single space separates the coord
    // column from the first value field, as in MYSTRAN's FORMAT 9902.
    write_subcase_header(out, sc.id);
    if (!sc.label.empty())
        out << " " << sc.label << "\n";
    out << "\n";
    out << "                                                          S P C   F O R C E S\n";
    out << "                                              (in global coordinate system at each grid)\n";
    out << "           GRID     COORD      T1            T2            T3            R1            R2            R3\n";
    out << "                     SYS\n";

    // Each value occupies 14 columns (13-wide right-justified value plus a
    // trailing space), so the validation parser's 13-wide windows, which
    // start at every 14th column, capture exactly the value.
    auto put_value = [&](const double v) {
        out << std::right;
        if (v == 0.0) {
            out << " 0.0          ";
        } else {
            out << std::uppercase << std::setw(13) << std::setprecision(6)
                << std::scientific << v << ' ';
        }
    };

    std::array<double, 6> sum{}, max{}, min{};
    std::fill(max.begin(), max.end(), -std::numeric_limits<double>::infinity());
    std::fill(min.begin(), min.end(), std::numeric_limits<double>::infinity());

    for (const auto& sf : sc.spc_forces) {
        out << std::right << std::setw(15) << sf.node.value;
        out << std::setw(9) << 0; // global coordinate system
        out << " ";
        for (int i = 0; i < 6; ++i)
            put_value(sf.f[static_cast<size_t>(i)]);
        out << std::nouppercase;
        out << "\n";
        for (int i = 0; i < 6; ++i) {
            const double v = sf.f[static_cast<size_t>(i)];
            sum[static_cast<size_t>(i)] += v;
            max[static_cast<size_t>(i)] = std::max(max[static_cast<size_t>(i)], v);
            min[static_cast<size_t>(i)] = std::min(min[static_cast<size_t>(i)], v);
        }
    }

    auto write_summary_row = [&](std::string_view label,
                                 const std::array<double, 6>& vals) {
        out << "                " << std::right << std::setw(6) << label << ":  ";
        for (int i = 0; i < 6; ++i)
            put_value(vals[static_cast<size_t>(i)]);
        out << std::nouppercase << "\n";
    };

    std::array<double, 6> abs{};
    for (int i = 0; i < 6; ++i)
        abs[static_cast<size_t>(i)] =
            std::max(std::abs(max[static_cast<size_t>(i)]),
                     std::abs(min[static_cast<size_t>(i)]));

    out << "                         ------------- ------------- ------------- ------------- ------------- -------------\n";
    write_summary_row("MAX*", max);
    write_summary_row("MIN*", min);
    out << "\n";
    write_summary_row("ABS*", abs);
    out << "                *for output set\n";
    out << "                         ------------- ------------- ------------- ------------- ------------- -------------\n";
    out << "     SPC FORCE TOTALS:  ";
    for (int i = 0; i < 6; ++i)
        put_value(sum[static_cast<size_t>(i)]);
    out << std::nouppercase << "\n";
    out << "     (for output set)\n";
    out << "\n";
}

void F06Writer::write_bar_stress_table(const SubCaseResults& sc,
                                       std::ostream& out) {
    if (!contains_line_stresses(sc)) return;

    // MYSTRAN "ELEMENT STRESSES IN LOCAL ELEMENT COORDINATE SYSTEM / FOR
    // ELEMENT TYPE BAR" layout: one element per two rows (end-A stresses
    // then end-B). Parser windows: eid cols 2-9, SA/SB1-4 at 11/25/39/53,
    // axial at 67, each 13 wide.
    write_subcase_header(out, sc.id);
    if (!sc.label.empty())
        out << " " << sc.label << "\n";
    out << "\n";
    out << "                                E L E M E N T   S T R E S S E S   I N   L O C A L   E L E M E N T   C O O R D I N A T E   S Y S T E M\n";
    out << "                                 F O R   E L E M E N T   T Y P E   B A R\n";
    out << "    Elem         End-A Bending Stresses                                Axial\n";
    out << "     ID          Sa1           Sa2           Sa3           Sa4      Stress\n";
    out << "                 Sb1           Sb2           Sb3           Sb4\n";

    for (const auto& ls : sc.line_stresses) {
        F06Row row_a;
        row_a.put(2, 8, std::to_string(ls.eid.value));
        row_a.put(11, 13, fmt_e(ls.end_a.s[0]));
        row_a.put(25, 13, fmt_e(ls.end_a.s[1]));
        row_a.put(39, 13, fmt_e(ls.end_a.s[2]));
        row_a.put(53, 13, fmt_e(ls.end_a.s[3]));
        row_a.put(67, 13, fmt_e(ls.end_a.axial));
        out << row_a.str() << "\n";
        F06Row row_b;
        row_b.put(11, 13, fmt_e(ls.end_b.s[0]));
        row_b.put(25, 13, fmt_e(ls.end_b.s[1]));
        row_b.put(39, 13, fmt_e(ls.end_b.s[2]));
        row_b.put(53, 13, fmt_e(ls.end_b.s[3]));
        out << row_b.str() << "\n";
    }
    out << "                          ------------- ------------- ------------- ------------- -------------\n";
    out << "\n";
}

namespace {

// Column windows used by the validation suite parser for the MYSTRAN-format
// shell stress tables (f06_query.py).
struct ShellCols {
    int xx, yy, sxy, angle, angle_w, major, minor, vm, zx, yz;
    int fiber; // fiber-distance column (not parsed, display only)
};

constexpr ShellCols kQuad4Cols{35, 48, 61, 75, 6, 82, 95, 108, 121, 134, 24};
constexpr ShellCols kTria3Cols{38, 51, 64, 78, 7, 86, 99, 112, 125, 138, 25};

struct FiberStress {
    double sx, sy, sxy, von_mises, major, minor, angle;
};

FiberStress fiber_stress_at(double sx, double sy, double sxy,
                            double mx, double my, double mxy,
                            double z, double t) {
    const double f = fiber_factor(z, t);
    FiberStress fs;
    fs.sx = sx + f * mx;
    fs.sy = sy + f * my;
    fs.sxy = sxy + f * mxy;
    compute_principal_2d(fs.sx, fs.sy, fs.sxy, fs.major, fs.minor, fs.angle);
    fs.von_mises = std::sqrt(fs.sx * fs.sx - fs.sx * fs.sy + fs.sy * fs.sy +
                             3.0 * fs.sxy * fs.sxy);
    return fs;
}

// Writes one fiber-stress line. `prefix` carries the element/location fields
// already positioned in their columns; the fiber and stress values are placed
// at absolute columns within the same line.
void write_fiber_row(std::ostream& out, const ShellCols& c, F06Row row,
                     const FiberStress& fs, double z, bool with_transverse) {
    row.put(c.fiber, 11, std::format("{:.4E}", z));
    row.put(c.xx, 12, fmt_e(fs.sx));
    row.put(c.yy, 12, fmt_e(fs.sy));
    row.put(c.sxy, 12, fmt_e(fs.sxy));
    row.put(c.angle, c.angle_w, fmt_angle(fs.angle));
    row.put(c.major, 12, fmt_e(fs.major));
    row.put(c.minor, 12, fmt_e(fs.minor));
    row.put(c.vm, 12, fmt_e(fs.von_mises));
    if (with_transverse) {
        // Transverse shear is not currently recovered; report zeros.
        row.put(c.zx, 12, fmt_e(0.0));
        row.put(c.yz, 12, fmt_e(0.0));
    }
    out << row.str() << "\n";
}

} // namespace

void F06Writer::write_shell_stress_table(
    const SubCaseResults& sc, std::ostream& out, const ElementType etype,
    const bool corners,
    const std::unordered_map<ElementId, const PShell*>& shell_props) {
    const bool is_quad4 = (etype == ElementType::CQUAD4);
    if (!contains_plate_stress_type(sc, etype) &&
        !(corners && contains_plate_nodal_stress_type(sc, etype)))
        return;

    write_subcase_header(out, sc.id);
    if (!sc.label.empty())
        out << " " << sc.label << "\n";
    out << "\n";
    out << "                                          E L E M E N T   S T R E S S E S   I N   L O C A L   E L E M E N T   C O O R D I N A T E   S Y S T E M\n";
    out << "                                 F O R   E L E M E N T   T Y P E   "
        << (is_quad4 ? "Q U A D 4" : "T R I A 3") << "\n";
    if (is_quad4) {
        out << "    Elem  Location        Fiber      Stresses In Element Coord System     Principal Stresses (Zero Shear)                Transverse   Transverse   % Poly\n";
        out << "     ID                  Distance   Normal-X     Normal-Y     Shear-XY     Angle     Major        Minor      von Mises    Shear-XZ     Shear-YZ    Fit Err\n";
        out << "                                                                                                                         (max through thickness)\n";
    } else {
        out << "  Element    Location       Fiber       Stresses In Element Coord System      Principal Stresses (Zero Shear)                Transverse   Transverse\n";
        out << "     ID                    Distance    Normal-X     Normal-Y     Shear-XY      Angle     Major        Minor      von Mises    Shear-XZ     Shear-YZ\n";
        out << "                                                                                                                             (max through thickness)\n";
    }
    out << "\n";

    const ShellCols& c = is_quad4 ? kQuad4Cols : kTria3Cols;

    for (const auto& ps : sc.plate_stresses) {
        if (ps.etype != etype) continue;

        double t = 0.0;
        double z1 = 0.0, z2 = 0.0;
        if (const auto it = shell_props.find(ps.eid); it != shell_props.end()) {
            t = it->second->t;
            z1 = std::isnan(it->second->z1) ? -t * 0.5 : it->second->z1;
            z2 = std::isnan(it->second->z2) ? t * 0.5 : it->second->z2;
        }

        if (is_quad4) {
            // CENTER row (corner 0 in the parser's numbering).
            F06Row cen;
            cen.put(2, 8, std::to_string(ps.eid.value));
            cen.put_left(12, "CENTER");
            write_fiber_row(out, c, cen,
                            fiber_stress_at(ps.sx, ps.sy, ps.sxy,
                                            ps.mx, ps.my, ps.mxy, z1, t),
                            z1, /*with_transverse=*/true);
            write_fiber_row(out, c, F06Row{},
                            fiber_stress_at(ps.sx, ps.sy, ps.sxy,
                                            ps.mx, ps.my, ps.mxy, z2, t),
                            z2, /*with_transverse=*/false);
            out << "\n";
            if (corners) {
                const int limit = std::min<int>(
                    plate_vertex_count(etype),
                    static_cast<int>(ps.nodal.size()));
                for (int i = 0; i < limit; ++i) {
                    const auto& pt = ps.nodal[static_cast<std::size_t>(i)];
                    F06Row grd;
                    grd.put_left(12, "GRD");
                    grd.put(15, 8, std::to_string(pt.node.value));
                    write_fiber_row(out, c, grd,
                                    fiber_stress_at(pt.sx, pt.sy, pt.sxy,
                                                    pt.mx, pt.my, pt.mxy,
                                                    z1, t),
                                    z1, true);
                    write_fiber_row(out, c, F06Row{},
                                    fiber_stress_at(pt.sx, pt.sy, pt.sxy,
                                                    pt.mx, pt.my, pt.mxy,
                                                    z2, t),
                                    z2, false);
                }
            }
        } else {
            // TRIA3: element-level rows only ("Anywhere" / "in elem").
            F06Row anywhere;
            anywhere.put(2, 8, std::to_string(ps.eid.value));
            anywhere.put_left(14, "Anywhere");
            write_fiber_row(out, c, anywhere,
                            fiber_stress_at(ps.sx, ps.sy, ps.sxy,
                                            ps.mx, ps.my, ps.mxy, z1, t),
                            z1, true);
            F06Row in_elem;
            in_elem.put_left(14, "in elem");
            write_fiber_row(out, c, in_elem,
                            fiber_stress_at(ps.sx, ps.sy, ps.sxy,
                                            ps.mx, ps.my, ps.mxy, z2, t),
                            z2, false);
        }
    }
    out << "                                  ------------ ------------ ------------         ------------ ------------ ------------ ------------\n";
    out << "\n";
}

void F06Writer::write_solid_stress_table(const SubCaseResults& sc,
                                         std::ostream& out,
                                         const ElementType etype,
                                         const bool corners) {
    if (!contains_solid_stress_type(sc, etype) &&
        !(corners && contains_solid_nodal_stress_type(sc, etype)))
        return;

    const char* type_name = "S O L I D";
    switch (etype) {
    case ElementType::CHEXA8:   type_name = "H E X A";   break;
    case ElementType::CTETRA4:  type_name = "T E T R A"; break;
    case ElementType::CTETRA10: type_name = "T E T R A"; break;
    case ElementType::CPENTA6:  type_name = "P E N T A"; break;
    default: break;
    }

    write_subcase_header(out, sc.id);
    if (!sc.label.empty())
        out << " " << sc.label << "\n";
    out << "\n";
    out << "                                E L E M E N T   S T R E S S E S   I N   M A T E R I A L   C O O R D I N A T E   S Y S T E M\n";
    out << "                                 F O R   E L E M E N T   T Y P E   " << type_name << "\n";
    out << "    Elem  Location            Sigma-xx      Sigma-yy      Sigma-zz       Tau-xy        Tau-yz        Tau-zx       von Mises\n";
    out << "     ID\n";

    auto write_stress_values = [&](const F06Row& loc, double xx, double yy,
                                   double zz, double xy, double yz, double zx,
                                   double vm) {
        F06Row row = loc;
        row.put(29, 13, fmt_e(xx));
        row.put(43, 13, fmt_e(yy));
        row.put(57, 13, fmt_e(zz));
        row.put(71, 13, fmt_e(xy));
        row.put(85, 13, fmt_e(yz));
        row.put(99, 13, fmt_e(zx));
        row.put(113, 13, fmt_e(vm));
        out << row.str() << "\n";
    };

    for (const auto& ss : sc.solid_stresses) {
        if (ss.etype != etype) continue;

        F06Row cen;
        cen.put(2, 8, std::to_string(ss.eid.value));
        cen.put_left(12, "CENTER");
        write_stress_values(cen, ss.sx, ss.sy, ss.sz, ss.sxy, ss.syz,
                            ss.szx, ss.von_mises);

        if (corners) {
            const int limit = std::min<int>(solid_vertex_count(etype),
                                            static_cast<int>(ss.nodal.size()));
            for (int i = 0; i < limit; ++i) {
                const auto& pt = ss.nodal[static_cast<std::size_t>(i)];
                F06Row grd;
                grd.put_left(12, "GRD");
                grd.put(15, 8, std::to_string(pt.node.value));
                write_stress_values(grd, pt.sx, pt.sy, pt.sz, pt.sxy,
                                    pt.syz, pt.szx, pt.von_mises);
            }
        }
    }
    out << "                            ------------- ------------- ------------- ------------- ------------- ------------- -------------\n";
    out << "\n";
}

void F06Writer::write_quad4_gpstress_table(const SubCaseResults& sc,
                                           std::ostream& out) {
    if (!contains_plate_nodal_stress_type(sc, ElementType::CQUAD4)) return;

    out << "\n                  G R I D   P O I N T   S T R E S S E S   I N   Q U A D R I L A T E R A L   E L E M E N T S   ( C Q U A D 4 )\n\n";
    out << "  ELEMENT-ID  GRID-ID    NORMAL-X       NORMAL-Y      SHEAR-XY       MOMENT-X       MOMENT-Y      MOMENT-XY       VON MISES\n";

    for (const auto& ps : sc.plate_stresses) {
        if (ps.etype != ElementType::CQUAD4) continue;
        for (const auto& point : ps.nodal) {
            out << std::setw(12) << ps.eid.value;
            out << std::setw(9) << point.node.value;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sx;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sy;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sxy;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.mx;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.my;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.mxy;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.von_mises;
            out << "\n";
        }
    }
}

void F06Writer::write_tria3_gpstress_table(const SubCaseResults& sc,
                                           std::ostream& out) {
    if (!contains_plate_nodal_stress_type(sc, ElementType::CTRIA3)) return;

    out << "\n                      G R I D   P O I N T   S T R E S S E S   I N   T R I A N G U L A R   E L E M E N T S   ( C T R I A 3 )\n\n";
    out << "  ELEMENT-ID  GRID-ID    NORMAL-X       NORMAL-Y      SHEAR-XY       MOMENT-X       MOMENT-Y      MOMENT-XY       VON MISES\n";

    for (const auto& ps : sc.plate_stresses) {
        if (ps.etype != ElementType::CTRIA3) continue;
        for (const auto& point : ps.nodal) {
            out << std::setw(12) << ps.eid.value;
            out << std::setw(9) << point.node.value;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sx;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sy;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sxy;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.mx;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.my;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.mxy;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.von_mises;
            out << "\n";
        }
    }
}

void F06Writer::write_solid_gpstress_table(const SubCaseResults& sc,
                                           std::ostream& out,
                                           ElementType etype) {
    if (!contains_solid_nodal_stress_type(sc, etype)) return;

    const char* title = "S O L I D";
    if (etype == ElementType::CHEXA8) {
        title  = "H E X A H E D R O N   E L E M E N T S   ( C H E X A )";
    } else if (etype == ElementType::CTETRA4) {
        title  = "T E T R A H E D R O N   E L E M E N T S   ( C T E T R A )";
    } else if (etype == ElementType::CTETRA10) {
        title  = "T E T R A H E D R O N   E L E M E N T S   ( C T E T R A 1 0 )";
    } else if (etype == ElementType::CPENTA6) {
        title  = "P E N T A H E D R O N   E L E M E N T S   ( C P E N T A )";
    }

    out << "\n                       G R I D   P O I N T   S T R E S S E S   I N   " << title << "\n\n";
    out << "  ELEMENT-ID  GRID-ID    NORMAL-X       NORMAL-Y       NORMAL-Z      SHEAR-XY       SHEAR-YZ       SHEAR-ZX       VON MISES\n";

    for (const auto& ss : sc.solid_stresses) {
        if (ss.etype != etype) continue;
        for (const auto& point : ss.nodal) {
            out << std::setw(12) << ss.eid.value;
            out << std::setw(7)  << point.node.value;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sx;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sy;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sz;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.sxy;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.syz;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.szx;
            out << std::setw(15) << std::setprecision(6) << std::scientific << point.von_mises;
            out << "\n";
        }
    }
}

// ── F06 modal output ──────────────────────────────────────────────────────────

void F06Writer::write_modal(const ModalSolverResults& results, const Model& model,
                            const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) throw SolverError(std::format("Cannot write F06: {}", path.string()));
    write_modal(results, model, f);
}

void F06Writer::write_modal(const ModalSolverResults& results, const Model& /*model*/,
                            std::ostream& out) {
    write_modal_header(out);
    for (const auto& msc : results.subcases) {
        const bool do_vec = msc.eigvec_print;

        write_eigenvalue_table(msc, out);

        if (do_vec)
            for (const auto& mode : msc.modes)
                write_eigenvector_table(mode, msc.id, msc.label, out);
    }
    out << "\n\n                     * * * END OF JOB * * *\n\n";
}

void F06Writer::write_modal_header(std::ostream& out) {
    std::time_t t = std::time(nullptr);
    char date_buf[32];
    std::strftime(date_buf, sizeof(date_buf), "%B %e, %Y", std::localtime(&t));

    out << "1" << banner_line("V I B E S T R A N", date_buf, kF06PageWidth - 1);
    out << "0" << std::string(static_cast<std::size_t>(kF06PageWidth - 1), ' ') << "\n";
    // Line 3 must start with " VIBESTRAN Version" for the validation suite
    // parser to recognise the file (see MYSTRAN_Validation/f06_query.py).
    out << " VIBESTRAN Version 1.0.0   F06 MYSTRAN-COMPATIBLE OUTPUT\n";
    out << page_line("S O L   1 0 3   N O R M A L   M O D E S   A N A L Y S I S");
    out << "\n";
}

void F06Writer::write_eigenvalue_table(const ModalSubCaseResults& msc,
                                       std::ostream& out) {
    if (msc.modes.empty()) return;

    write_subcase_header(out, msc.id);
    if (!msc.label.empty())
        out << " " << msc.label << "\n";
    out << "\n";
    out << "                                            R E A L   E I G E N V A L U E S\n";
    out << "   MODE  EXTRACTION      EIGENVALUE           RADIANS              CYCLES            GENERALIZED         GENERALIZED        \n";
    out << "  NUMBER   ORDER                                                                        MASS              STIFFNESS\n\n";

    for (const auto& mode : msc.modes) {
        double gen_stiff = mode.eigenvalue * mode.gen_mass;
        out << std::setw(9)  << mode.mode_number;
        out << std::setw(8)  << mode.mode_number; // extraction order = mode number
        out << std::uppercase;
        out << std::setw(20) << std::setprecision(6) << std::scientific << mode.eigenvalue;
        out << std::setw(20) << std::setprecision(6) << std::scientific << mode.radians_per_sec;
        out << std::setw(20) << std::setprecision(6) << std::scientific << mode.cycles_per_sec;
        out << std::setw(20) << std::setprecision(6) << std::scientific << mode.gen_mass;
        out << std::setw(20) << std::setprecision(6) << std::scientific << gen_stiff;
        out << std::nouppercase;
        out << "\n";
    }
    out << "\n";
}

void F06Writer::write_eigenvector_table(const ModeResult& mode, int subcase_id,
                                        const std::string& label,
                                        std::ostream& out) {
    // Order matters to the validation suite parser: the line immediately
    // before " OUTPUT FOR EIGENVECTOR" must be " OUTPUT FOR SUBCASE" (mode
    // number in cols 25-32, subcase in cols 21-28 of their respective lines).
    write_subcase_header(out, subcase_id);
    // Matches MYSTRAN FORMAT(' OUTPUT FOR EIGENVECTOR ',I8).
    out << " OUTPUT FOR EIGENVECTOR " << std::setw(8) << mode.mode_number << "\n";
    if (!label.empty())
        out << " " << label << "\n";
    out << "\n";
    out << "                                                         E I G E N V E C T O R\n";
    out << "                                              (in global coordinate system at each grid)\n";
    out << "           GRID     COORD      T1            T2            T3            R1            R2            R3\n";
    out << "                     SYS\n";

    for (const auto& nd : mode.shape) {
        out << std::setw(15) << nd.node.value;
        out << std::setw(9)  << 0; // global coord sys
        out << std::uppercase;
        for (int i = 0; i < 6; ++i) {
            double v = nd.d[i];
            if (v == 0.0) {
                out << "  0.0         ";
            } else {
                out << std::setw(14) << std::setprecision(6) << std::scientific << v;
            }
        }
        out << std::nouppercase;
        out << "\n";
    }
    out << "\n";
}

// ── Heat-transfer (SOL 153) F06 output ───────────────────────────────────────

void F06Writer::write_thermal(const SolverResults& results, const Model& model,
                              const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) throw SolverError(std::format("Cannot write F06: {}", path.string()));
    write_thermal(results, model, f);
}

void F06Writer::write_thermal(const SolverResults& results, const Model& model,
                              std::ostream& out) {
    // Header (heat-transfer banner instead of SOL 101)
    std::time_t t = std::time(nullptr);
    char date_buf[32];
    std::strftime(date_buf, sizeof(date_buf), "%B %e, %Y", std::localtime(&t));
    out << "1" << banner_line("V I B E S T R A N", date_buf, kF06PageWidth - 1);
    out << "0" << std::string(static_cast<std::size_t>(kF06PageWidth - 1), ' ') << "\n";
    out << " VIBESTRAN Version 1.0.0   F06 MYSTRAN-COMPATIBLE OUTPUT\n";
    out << page_line("S O L   1 5 3   L I N E A R   H E A T   T R A N S F E R");
    out << "\n";

    for (const auto& sc : results.subcases) {
        const SubCase* msc = find_model_subcase(model, sc.id);
        const bool do_temp  = (msc != nullptr) && msc->disp_print;
        const bool do_flux  = (msc != nullptr) && msc->stress_print;

        out << "\n OUTPUT FOR SUBCASE" << std::setw(9) << sc.id << "\n";
        if (!sc.label.empty())
            out << " " << sc.label << "\n";

        if (do_temp && !sc.temperatures.empty()) {
            out << "\n";
            out << "                                                          T E M P E R A T U R E S\n";
            out << "      POINT ID.                  TEMPERATURE\n";
            for (const auto& nt : sc.temperatures) {
                out << "    " << std::setw(10) << nt.node.value
                    << "    " << std::scientific << std::setprecision(6)
                    << std::setw(18) << nt.temperature << "\n";
            }
        }

        if (do_flux && !sc.heat_fluxes.empty()) {
            out << "\n";
            out << "                                            E L E M E N T   H E A T   F L U X\n";
            out << "    ELEMENT ID.   TYPE      QX             QY             QZ             |Q|\n";
            for (const auto& ef : sc.heat_fluxes) {
                const char* tn = "?";
                switch (ef.etype) {
                  case ElementType::CHEXA8:  tn = "CHEXA"; break;
                  case ElementType::CTETRA4: tn = "CTETRA"; break;
                  case ElementType::CTETRA10:tn = "CTETRA10"; break;
                  case ElementType::CPENTA6: tn = "CPENTA"; break;
                  case ElementType::CBAR:    tn = "CBAR"; break;
                  case ElementType::CBEAM:   tn = "CBEAM"; break;
                  default: break;
                }
                out << "    " << std::setw(10) << ef.eid.value
                    << "    " << std::setw(8) << tn
                    << std::scientific << std::setprecision(6)
                    << std::setw(16) << ef.q[0]
                    << std::setw(16) << ef.q[1]
                    << std::setw(16) << ef.q[2]
                    << std::setw(16) << ef.magnitude << "\n";
            }
        }
    }
    out << "\n\n                     * * * END OF JOB * * *\n\n";
}

} // namespace vibestran
