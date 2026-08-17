// tests/unit/test_output_writers.cpp
// Tests for F06, CSV output writers and result helper functions.
//
// Coverage:
//   - compute_principal_2d: known analytical results
//   - compute_principal_3d: diagonal (axis-aligned) and off-diagonal tensors
//   - F06Writer: displacement table, CQUAD4 and CTRIA3 separate tables,
//                principal stress columns in plate table, output flag suppression
//   - CsvWriter: header format, column order, per-subcase row emission,
//                output flag suppression
//   - BdfParser: DISPLACEMENT=NONE / STRESS=NONE case-control flags
//   - BdfParser: PARAM,CSVOUT,YES

#include <gtest/gtest.h>
#include "io/results.hpp"
#include "io/bdf_parser.hpp"
#include "core/model.hpp"
#include <algorithm>
#include <sstream>
#include <string>
#include <cmath>
#include <numbers>
#include <filesystem>
#include <fstream>
#include <bit>
#include <cstdint>
#include <vector>

using namespace vibestran;

// ── Helpers ───────────────────────────────────────────────────────────────────

static SubCaseResults make_sc_with_disps() {
    SubCaseResults sc;
    sc.id = 1;
    NodeDisplacement nd;
    nd.node = NodeId{5};
    nd.d = {1.0, 2.0, 3.0, 0.1, 0.2, 0.3};
    sc.displacements.push_back(nd);
    return sc;
}

static PlateStress make_quad4_stress(ElementId eid, double sx, double sy, double sxy) {
    PlateStress ps;
    ps.eid   = eid;
    ps.etype = ElementType::CQUAD4;
    ps.sx = sx; ps.sy = sy; ps.sxy = sxy;
    ps.von_mises = std::sqrt(sx*sx - sx*sy + sy*sy + 3.0*sxy*sxy);
    return ps;
}

static PlateStress make_tria3_stress(ElementId eid, double sx, double sy, double sxy) {
    PlateStress ps = make_quad4_stress(eid, sx, sy, sxy);
    ps.etype = ElementType::CTRIA3;
    return ps;
}

static Model make_empty_model_sc1() {
    Model m;
    m.analysis.sol = SolutionType::LinearStatic;
    SubCase sc;
    sc.id = 1;
    sc.disp_print   = true;
    sc.stress_print = true;
    m.analysis.subcases.push_back(sc);
    return m;
}

static LineStress make_line_stress(ElementId eid) {
    LineStress ls;
    ls.eid = eid;
    ls.etype = ElementType::CBAR;
    ls.end_a.node = NodeId{101};
    ls.end_a.s = {1.25, -2.5, 3.75, -4.5};
    ls.end_a.axial = 5.125;
    ls.end_a.smax = 6.25;
    ls.end_a.smin = -7.5;
    ls.end_b.node = NodeId{102};
    ls.end_b.s = {8.5, -9.25, 10.75, -11.5};
    ls.end_b.axial = 5.125;
    ls.end_b.smax = 12.125;
    ls.end_b.smin = -13.5;
    return ls;
}

static PlateStressPoint make_plate_point(int node_id, double scale) {
    PlateStressPoint pt;
    pt.node = NodeId{node_id};
    pt.sx = 10.0 * scale;
    pt.sy = 20.0 * scale;
    pt.sxy = 30.0 * scale;
    pt.mx = 40.0 * scale;
    pt.my = 50.0 * scale;
    pt.mxy = 60.0 * scale;
    pt.von_mises = 70.0 * scale;
    return pt;
}

static SolidStress make_solid_stress(ElementId eid) {
    SolidStress ss;
    ss.eid = eid;
    ss.etype = ElementType::CTETRA4;
    ss.sx = 1.0;
    ss.sy = 2.0;
    ss.sz = 3.0;
    ss.sxy = 4.0;
    ss.syz = 5.0;
    ss.szx = 6.0;
    ss.von_mises = 7.0;
    auto add_point = [&](int node_id, double sx, double sy, double sz,
                         double sxy, double syz, double szx, double vm) {
        SolidStressPoint pt;
        pt.node = NodeId{node_id};
        pt.sx = sx;
        pt.sy = sy;
        pt.sz = sz;
        pt.sxy = sxy;
        pt.syz = syz;
        pt.szx = szx;
        pt.von_mises = vm;
        ss.nodal.push_back(pt);
    };
    add_point(9001, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0);
    add_point(9002, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0);
    add_point(9003, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0);
    add_point(9004, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0);
    return ss;
}

static SolidStress make_ctetra10_solid_stress(ElementId eid) {
    SolidStress ss;
    ss.eid = eid;
    ss.etype = ElementType::CTETRA10;
    ss.sx = 1.0;
    ss.sy = 2.0;
    ss.sz = 3.0;
    ss.sxy = 4.0;
    ss.syz = 5.0;
    ss.szx = 6.0;
    ss.von_mises = 7.0;
    for (int i = 0; i < 10; ++i) {
        const double base = 10.0 * static_cast<double>(i + 1);
        SolidStressPoint pt;
        pt.node = NodeId{9101 + i};
        pt.sx = base + 1.0;
        pt.sy = base + 2.0;
        pt.sz = base + 3.0;
        pt.sxy = base + 4.0;
        pt.syz = base + 5.0;
        pt.szx = base + 6.0;
        pt.von_mises = base + 7.0;
        ss.nodal.push_back(pt);
    }
    return ss;
}

static std::vector<std::byte> run_op2_writer(const SolverResults& res,
                                             const Model& model) {
    namespace fs = std::filesystem;
    static int counter = 0;
    const fs::path path = fs::temp_directory_path() /
        ("vibestran_op2_test_" + std::to_string(++counter) + ".op2");

    Op2Writer::write(res, model, path);

    std::ifstream f(path, std::ios::binary);
    std::vector<char> chars((std::istreambuf_iterator<char>(f)), {});
    fs::remove(path);

    std::vector<std::byte> bytes(chars.size());
    std::transform(chars.begin(), chars.end(), bytes.begin(),
                   [](char c) { return static_cast<std::byte>(c); });
    return bytes;
}

static std::vector<int32_t> bytes_to_words(const std::vector<std::byte>& bytes) {
    EXPECT_EQ(bytes.size() % sizeof(int32_t), 0u);
    std::vector<int32_t> words;
    words.reserve(bytes.size() / sizeof(int32_t));
    for (std::size_t i = 0; i + sizeof(int32_t) <= bytes.size(); i += sizeof(int32_t)) {
        int32_t word = 0;
        std::memcpy(&word, bytes.data() + i, sizeof(int32_t));
        words.push_back(word);
    }
    return words;
}

static bool contains_bytes(const std::vector<std::byte>& bytes, std::string_view needle) {
    const auto* begin = reinterpret_cast<const char*>(bytes.data());
    return std::string_view(begin, bytes.size()).find(needle) != std::string_view::npos;
}

static bool contains_word_sequence(const std::vector<int32_t>& words,
                                   const std::vector<int32_t>& needle) {
    if (needle.empty() || needle.size() > words.size()) return false;
    for (std::size_t i = 0; i + needle.size() <= words.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needle.size(); ++j) {
            if (words[i + j] != needle[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

static bool contains_word(const std::vector<int32_t>& words, int32_t needle) {
    return std::find(words.begin(), words.end(), needle) != words.end();
}

static int32_t as_word(float value) {
    return std::bit_cast<int32_t>(value);
}

// ── compute_principal_2d ─────────────────────────────────────────────────────

TEST(PrincipalStress2D, UniaxialX) {
    // sx = 100, sy = 0, sxy = 0 → principal = {100, 0}, angle = 0°
    double major, minor, angle;
    compute_principal_2d(100.0, 0.0, 0.0, major, minor, angle);
    EXPECT_NEAR(major, 100.0, 1e-10);
    EXPECT_NEAR(minor,   0.0, 1e-10);
    EXPECT_NEAR(angle,   0.0, 1e-10);
}

TEST(PrincipalStress2D, PureShear) {
    // sx = 0, sy = 0, sxy = τ → principal = {τ, -τ}, angle = 45°
    double tau = 50.0;
    double major, minor, angle;
    compute_principal_2d(0.0, 0.0, tau, major, minor, angle);
    EXPECT_NEAR(major,  tau, 1e-10);
    EXPECT_NEAR(minor, -tau, 1e-10);
    EXPECT_NEAR(std::abs(angle), 45.0, 1e-6);
}

TEST(PrincipalStress2D, EqualBiaxial) {
    // sx = sy = p, sxy = 0 → both principals = p
    double p = 75.0;
    double major, minor, angle;
    compute_principal_2d(p, p, 0.0, major, minor, angle);
    EXPECT_NEAR(major, p, 1e-10);
    EXPECT_NEAR(minor, p, 1e-10);
}

TEST(PrincipalStress2D, GeneralTensor) {
    // sx=100, sy=50, sxy=30
    // avg = 75, diff = 25, R = sqrt(625+900) = sqrt(1525)
    double sx = 100.0, sy = 50.0, sxy = 30.0;
    double R = std::sqrt(25.0*25.0 + sxy*sxy);
    double major, minor, angle;
    compute_principal_2d(sx, sy, sxy, major, minor, angle);
    EXPECT_NEAR(major, 75.0 + R, 1e-8);
    EXPECT_NEAR(minor, 75.0 - R, 1e-8);
    // Check that the tensor is recovered: major and minor are eigenvalues
    EXPECT_GT(major, minor);
    // Verify trace and determinant are preserved
    EXPECT_NEAR(major + minor, sx + sy, 1e-8);
    EXPECT_NEAR(major * minor, sx*sy - sxy*sxy, 1e-6);
}

// ── compute_principal_3d ─────────────────────────────────────────────────────

TEST(PrincipalStress3D, DiagonalTensor) {
    // Diagonal stress tensor: eigenvalues are the diagonal elements
    double p[3];
    double v[3][3];
    compute_principal_3d(30.0, 20.0, 10.0, 0.0, 0.0, 0.0, p, v);
    EXPECT_NEAR(p[0], 30.0, 1e-8); // sorted descending
    EXPECT_NEAR(p[1], 20.0, 1e-8);
    EXPECT_NEAR(p[2], 10.0, 1e-8);
}

TEST(PrincipalStress3D, UniaxialStress) {
    // Only sx nonzero → one eigenvalue = sx, two = 0
    double p[3];
    double v[3][3];
    compute_principal_3d(100.0, 0.0, 0.0, 0.0, 0.0, 0.0, p, v);
    EXPECT_NEAR(p[0], 100.0, 1e-8);
    EXPECT_NEAR(p[1],   0.0, 1e-8);
    EXPECT_NEAR(p[2],   0.0, 1e-8);
}

TEST(PrincipalStress3D, TracePreserved) {
    // The trace (sum of eigenvalues) must equal sx+sy+sz
    double sx=80, sy=60, sz=40, sxy=15, syz=10, szx=5;
    double p[3];
    double v[3][3];
    compute_principal_3d(sx, sy, sz, sxy, syz, szx, p, v);
    EXPECT_NEAR(p[0]+p[1]+p[2], sx+sy+sz, 1e-6);
    // Must be sorted descending
    EXPECT_GE(p[0], p[1]);
    EXPECT_GE(p[1], p[2]);
}

TEST(PrincipalStress3D, EigenvectorsOrthonormal) {
    // Eigenvectors from Jacobi should be orthonormal
    double sx=80, sy=60, sz=40, sxy=15, syz=10, szx=5;
    double p[3];
    double v[3][3];
    compute_principal_3d(sx, sy, sz, sxy, syz, szx, p, v);
    for (int i = 0; i < 3; ++i) {
        double dot_ii = 0.0;
        for (int k = 0; k < 3; ++k) dot_ii += v[i][k]*v[i][k];
        EXPECT_NEAR(dot_ii, 1.0, 1e-10) << "row " << i << " not unit length";
        for (int j = i+1; j < 3; ++j) {
            double dot_ij = 0.0;
            for (int k = 0; k < 3; ++k) dot_ij += v[i][k]*v[j][k];
            EXPECT_NEAR(dot_ij, 0.0, 1e-10) << "rows " << i << "," << j << " not orthogonal";
        }
    }
}

// ── F06Writer ─────────────────────────────────────────────────────────────────

TEST(F06Writer, HeaderContainsNastranTitle) {
    SolverResults res;
    res.subcases.push_back({});
    res.subcases[0].id = 1;
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("V I B E S T R A N"), std::string::npos);
    EXPECT_NE(out.find("S O L   1 0 1"), std::string::npos);
}

TEST(F06Writer, HeaderLine3IdentifiesVibestran) {
    // The MYSTRAN validation suite parser (f06_query.py) requires the 3rd
    // line of the file to start with " VIBESTRAN Version".
    SolverResults res;
    res.subcases.push_back({});
    res.subcases[0].id = 1;
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::istringstream iss(oss.str());
    std::string line0, line1, line2;
    std::getline(iss, line0);
    std::getline(iss, line1);
    std::getline(iss, line2);
    EXPECT_EQ(line2.rfind(" VIBESTRAN Version ", 0), 0);
}

TEST(F06Writer, DisplacementTable) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("D I S P L A C E M E N T S"), std::string::npos);
    EXPECT_NE(out.find("(in global coordinate system at each grid)"), std::string::npos);
    EXPECT_NE(out.find("GRID     COORD"), std::string::npos);
    // Node 5 should appear
    EXPECT_NE(out.find("5"), std::string::npos);
}

TEST(F06Writer, DisplacementRowMatchesParserColumns) {
    // Parser windows: gid cols 8-15, TX 26-38, TY 40-52, RZ 96-108.
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    NodeDisplacement nd;
    nd.node = NodeId{5};
    nd.d = {1.5, -2.5, 0.0, 0.0, 0.0, 3.25e-4};
    sc.displacements.push_back(nd);
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);

    std::istringstream iss(oss.str());
    std::string line;
    while (std::getline(iss, line)) {
        // First data row of the displacement table: gid 5 right-justified
        // in cols 2-15 followed by the coordinate-system id.
        if (line.rfind("              5        0", 0) == 0) break;
        line.clear();
    }
    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.substr(7, 8), "       5");
    EXPECT_DOUBLE_EQ(std::stod(line.substr(25, 13)), 1.5);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(39, 13)), -2.5);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(95, 13)), 3.25e-4);
}

TEST(F06Writer, SpcForceTableEmittedWhenRequested) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    SpcForce sf;
    sf.node = NodeId{7};
    sf.f = {1.5, -2.5, 0.0, 0.0, 0.0, 3.25e-4};
    sc.spc_forces.push_back(sf);
    res.subcases.push_back(sc);

    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].spc_force_print = true;

    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("S P C   F O R C E S"), std::string::npos);
    EXPECT_NE(out.find("SPC FORCE TOTALS"), std::string::npos);
}

TEST(F06Writer, SpcForceRowMatchesParserColumns) {
    // Parser windows: gid cols 8-15, TX 26-38, TY 40-52, TZ 54-66,
    // RX 68-80, RY 82-94, RZ 96-108.
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    SpcForce sf;
    sf.node = NodeId{7};
    sf.f = {1.5, -2.5, 0.0, 0.0, 0.0, 3.25e-4};
    sc.spc_forces.push_back(sf);
    res.subcases.push_back(sc);

    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].spc_force_print = true;

    std::ostringstream oss;
    F06Writer::write(res, m, oss);

    std::istringstream iss(oss.str());
    std::string line;
    while (std::getline(iss, line)) {
        if (line.rfind("              7        0", 0) == 0) break;
        line.clear();
    }
    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.substr(7, 8), "       7");
    EXPECT_DOUBLE_EQ(std::stod(line.substr(25, 13)), 1.5);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(39, 13)), -2.5);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(95, 13)), 3.25e-4);
}

TEST(F06Writer, SpcForceTableSuppressedWithoutRequest) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    SpcForce sf;
    sf.node = NodeId{7};
    sf.f = {1.5, -2.5, 0.0, 0.0, 0.0, 3.25e-4};
    sc.spc_forces.push_back(sf);
    res.subcases.push_back(sc);

    Model m = make_empty_model_sc1();  // spc_force_print defaults false

    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    EXPECT_EQ(oss.str().find("S P C   F O R C E S"), std::string::npos);
}

TEST(F06Writer, StaticSubcaseHeaderMatchesParserFormat) {
    SolverResults res;
    SubCaseResults sc = make_sc_with_disps();
    sc.label = "CQUAD4 CANTILEVER BENDING";
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();

    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("OUTPUT FOR SUBCASE        1"), std::string::npos);
    EXPECT_NE(out.find("\n CQUAD4 CANTILEVER BENDING\n"), std::string::npos);
}

TEST(F06Writer, Quad4StressTablePresent) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{1}, 100.0, 50.0, 30.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("E L E M E N T   S T R E S S E S   I N   L O C A L   E L E M E N T   C O O R D I N A T E   S Y S T E M"), std::string::npos);
    EXPECT_NE(out.find("F O R   E L E M E N T   T Y P E   Q U A D 4"), std::string::npos);
    EXPECT_NE(out.find("Major"), std::string::npos);
    EXPECT_NE(out.find("Minor"), std::string::npos);
    EXPECT_NE(out.find("Angle"), std::string::npos);
    EXPECT_NE(out.find("von Mises"), std::string::npos);
}

TEST(F06Writer, Tria3StressTablePresent) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_tria3_stress(ElementId{2}, 80.0, 20.0, 10.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("F O R   E L E M E N T   T Y P E   T R I A 3"), std::string::npos);
}

TEST(F06Writer, Quad4AndTria3TablesSeparate) {
    // When both element types are present, both tables should appear and be distinct
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{1}, 100.0, 0.0, 0.0));
    sc.plate_stresses.push_back(make_tria3_stress(ElementId{2}, 50.0,  0.0, 0.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_NE(out.find("T Y P E   Q U A D 4"), std::string::npos);
    EXPECT_NE(out.find("T Y P E   T R I A 3"), std::string::npos);
    // The two table headers should be at different positions
    auto pos_q = out.find("T Y P E   Q U A D 4");
    auto pos_t = out.find("T Y P E   T R I A 3");
    EXPECT_NE(pos_q, pos_t);
}

TEST(F06Writer, OnlyTria3NoQuad4Table) {
    // If only CTRIA3 stresses are present, no CQUAD4 table should be emitted
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_tria3_stress(ElementId{3}, 20.0, 5.0, 2.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_EQ(out.find("T Y P E   Q U A D 4"), std::string::npos);
    EXPECT_NE(out.find("T Y P E   T R I A 3"), std::string::npos);
}

TEST(F06Writer, OutputFlagSuppressesDisplacement) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].disp_print = false;
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_EQ(out.find("D I S P L A C E M E N T"), std::string::npos);
}

TEST(F06Writer, OutputFlagSuppressesStress) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{1}, 100.0, 0.0, 0.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = false;
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    EXPECT_EQ(out.find("T Y P E   Q U A D 4"), std::string::npos);
}

TEST(F06Writer, Quad4PrincipalStressValues) {
    // Pure uniaxial: sx=100, sy=0, sxy=0 → major=100, minor=0, angle=0
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    PlateStress ps;
    ps.eid   = ElementId{7};
    ps.etype = ElementType::CQUAD4;
    ps.sx = 100.0; ps.sy = 0.0; ps.sxy = 0.0;
    ps.von_mises = 100.0;
    sc.plate_stresses.push_back(ps);
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    std::string out = oss.str();
    // major = 100, should appear in scientific notation with leading digits "1.000000"
    EXPECT_NE(out.find("1.00000E+02"), std::string::npos);
}

TEST(F06Writer, Quad4CenterRowMatchesParserColumns) {
    // Parser windows for QUAD4 stress rows: eid cols 2-9, CENTER at 12-17,
    // XX 35-46, YY 48-59, XY 61-72, angle 75-80, major 82-93, minor 95-106,
    // von Mises 108-119, ZX 121-132, YZ 134-145.
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    PlateStress ps;
    ps.eid   = ElementId{7};
    ps.etype = ElementType::CQUAD4;
    ps.sx = 100.0; ps.sy = -50.0; ps.sxy = 25.0;
    ps.mx = 0.0; ps.my = 0.0; ps.mxy = 0.0; // no thickness info → membrane only
    ps.von_mises = 0.0;
    sc.plate_stresses.push_back(ps);
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write(res, m, oss);

    std::istringstream iss(oss.str());
    std::string line;
    while (std::getline(iss, line)) {
        if (line.size() >= 17 && line.substr(11, 6) == "CENTER") break;
        line.clear();
    }
    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.substr(1, 8), "       7");
    EXPECT_DOUBLE_EQ(std::stod(line.substr(34, 12)), 100.0);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(47, 12)), -50.0);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(60, 12)), 25.0);
    double major, minor, angle;
    compute_principal_2d(100.0, -50.0, 25.0, major, minor, angle);
    EXPECT_NEAR(std::stod(line.substr(74, 6)), angle, 0.005);
    EXPECT_NEAR(std::stod(line.substr(81, 12)), major, std::abs(major) * 1e-5);
    EXPECT_NEAR(std::stod(line.substr(94, 12)), minor, std::abs(minor) * 1e-5);
}

TEST(F06Writer, Quad4FiberStressesFromBendingMoment) {
    // With thickness t from PSHELL, fiber stresses are
    // sigma(z) = membrane + (12*z/t^3)*moment, z1=-t/2, z2=+t/2.
    Model m = make_empty_model_sc1();
    PShell ps_prop;
    ps_prop.pid = PropertyId{10};
    ps_prop.t = 0.1; // z1 = -0.05, z2 = +0.05
    m.properties[PropertyId{10}] = ps_prop;
    ElementData elem;
    elem.id = ElementId{7};
    elem.type = ElementType::CQUAD4;
    elem.pid = PropertyId{10};
    elem.nodes = {NodeId{1}, NodeId{2}, NodeId{3}, NodeId{4}};
    m.elements.push_back(elem);

    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    PlateStress ps;
    ps.eid   = ElementId{7};
    ps.etype = ElementType::CQUAD4;
    ps.sx = 10.0; ps.sy = 0.0; ps.sxy = 0.0;
    ps.mx = 1.0; ps.my = 0.0; ps.mxy = 0.0;
    sc.plate_stresses.push_back(ps);
    res.subcases.push_back(sc);

    std::ostringstream oss;
    F06Writer::write(res, m, oss);

    // 12*z/t^3 at z=-0.05, t=0.1: 12*(-0.05)/0.001 = -600 → XX(Z1) = 10 - 600
    // At z=+0.05: XX(Z2) = 10 + 600.
    std::istringstream iss(oss.str());
    std::string z1_line, z2_line;
    while (std::getline(iss, z1_line)) {
        if (z1_line.size() >= 17 && z1_line.substr(11, 6) == "CENTER") {
            std::getline(iss, z2_line);
            break;
        }
    }
    ASSERT_FALSE(z1_line.empty());
    EXPECT_NEAR(std::stod(z1_line.substr(34, 12)), 10.0 - 600.0,
                std::abs(10.0 - 600.0) * 1e-5);
    EXPECT_NEAR(std::stod(z2_line.substr(34, 12)), 10.0 + 600.0,
                std::abs(10.0 + 600.0) * 1e-5);
}

TEST(F06Writer, BarStressTableMatchesParserColumns) {
    // Parser windows: eid cols 2-9, SA/SB1-4 at 11/25/39/53, axial at 67,
    // each 13 wide.
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.line_stresses.push_back(make_line_stress(ElementId{42}));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();

    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    const std::string out = oss.str();
    EXPECT_NE(out.find("F O R   E L E M E N T   T Y P E   B A R"), std::string::npos);

    std::istringstream iss(out);
    std::string row_a, row_b;
    while (std::getline(iss, row_a)) {
        if (row_a.size() >= 9 && row_a.substr(1, 8) == "      42") break;
        row_a.clear();
    }
    ASSERT_FALSE(row_a.empty());
    ASSERT_TRUE(std::getline(iss, row_b));
    EXPECT_EQ(row_a.substr(1, 8), "      42");
    EXPECT_DOUBLE_EQ(std::stod(row_a.substr(10, 13)), 1.25);
    EXPECT_DOUBLE_EQ(std::stod(row_a.substr(24, 13)), -2.5);
    EXPECT_DOUBLE_EQ(std::stod(row_a.substr(38, 13)), 3.75);
    EXPECT_DOUBLE_EQ(std::stod(row_a.substr(52, 13)), -4.5);
    EXPECT_DOUBLE_EQ(std::stod(row_a.substr(66, 13)), 5.125);
    EXPECT_DOUBLE_EQ(std::stod(row_b.substr(10, 13)), 8.5);
    EXPECT_DOUBLE_EQ(std::stod(row_b.substr(52, 13)), -11.5);
}

TEST(F06Writer, Quad4CornerRowsInMystranTable) {
    // With STRESS(CORNER), the QUAD4 table contains CENTER plus 4 GRD rows
    // with grid IDs in cols 15-22.
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;

    PlateStress ps = make_quad4_stress(ElementId{7}, 100.0, 50.0, 30.0);
    ps.nodal.push_back(make_plate_point(11, 1.0));
    ps.nodal.push_back(make_plate_point(12, 2.0));
    ps.nodal.push_back(make_plate_point(13, 3.0));
    ps.nodal.push_back(make_plate_point(14, 4.0));
    sc.plate_stresses.push_back(ps);
    res.subcases.push_back(sc);

    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = true;
    m.analysis.subcases[0].stress_corner_print = true;
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("Principal Stresses (Zero Shear)"), std::string::npos);
    // GRD rows: location at cols 12-14, gid right-justified in cols 15-22.
    EXPECT_NE(out.find("           GRD      11"), std::string::npos);
    EXPECT_NE(out.find("           GRD      14"), std::string::npos);
}

TEST(F06Writer, SolidStressTableMatchesParserColumns) {
    // Parser windows: eid cols 2-9, CENTER 12-17, GRD gid 15-22, values at
    // 29/43/57/71/85/99/113 each 13 wide.
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.solid_stresses.push_back(make_solid_stress(ElementId{8}));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();

    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    const std::string out = oss.str();
    EXPECT_NE(out.find("E L E M E N T   S T R E S S E S   I N   M A T E R I A L   C O O R D I N A T E   S Y S T E M"), std::string::npos);
    EXPECT_NE(out.find("F O R   E L E M E N T   T Y P E   T E T R A"), std::string::npos);

    std::istringstream iss(out);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.size() >= 17 && line.substr(11, 6) == "CENTER") break;
        line.clear();
    }
    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.substr(1, 8), "       8");
    EXPECT_DOUBLE_EQ(std::stod(line.substr(28, 13)), 1.0);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(42, 13)), 2.0);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(56, 13)), 3.0);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(70, 13)), 4.0);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(84, 13)), 5.0);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(98, 13)), 6.0);
    EXPECT_DOUBLE_EQ(std::stod(line.substr(112, 13)), 7.0);

    // First GRD corner row (no corner output requested here → none present).
    // With stress_corner_print the rows follow; covered in the test below.
}

TEST(F06Writer, SolidCornerRowsWithGids) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.solid_stresses.push_back(make_solid_stress(ElementId{8}));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = true;
    m.analysis.subcases[0].stress_corner_print = true;

    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    const std::string out = oss.str();
    // GRD rows carry grid id in cols 15-22 and first component in 29-41.
    EXPECT_NE(out.find("           GRD    9001"), std::string::npos);
    EXPECT_NE(out.find("           GRD    9004"), std::string::npos);

    std::istringstream iss(out);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.size() >= 14 && line.substr(11, 3) == "GRD") break;
        line.clear();
    }
    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.substr(14, 8), "    9001");
    EXPECT_DOUBLE_EQ(std::stod(line.substr(28, 13)), 11.0);
}

TEST(F06Writer, ModalEigenvalueAndEigenvectorBlocks) {
    // The parser requires " OUTPUT FOR EIGENVECTOR" (leading space, mode
    // number in cols 25-32) immediately preceded by " OUTPUT FOR SUBCASE",
    // and eigenvalue rows with mode at cols 2-9, eigenvalue at 25-37 and
    // cycles at 65-77.
    ModalSolverResults res;
    ModalSubCaseResults msc;
    msc.id = 1;
    msc.eigvec_print = true;
    ModeResult mode;
    mode.mode_number = 1;
    mode.eigenvalue = 2.5e6;
    mode.radians_per_sec = std::sqrt(2.5e6);
    mode.cycles_per_sec = std::sqrt(2.5e6) / (2.0 * std::numbers::pi);
    mode.gen_mass = 1.0;
    NodeDisplacement nd;
    nd.node = NodeId{5};
    nd.d = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mode.shape.push_back(nd);
    msc.modes.push_back(mode);
    res.subcases.push_back(msc);

    Model m = make_empty_model_sc1();
    std::ostringstream oss;
    F06Writer::write_modal(res, m, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("R E A L   E I G E N V A L U E S"), std::string::npos);
    EXPECT_NE(out.find("E I G E N V E C T O R"), std::string::npos);

    std::istringstream iss(out);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.size() >= 25 && line.rfind(" OUTPUT FOR EIGENVECTOR", 0) == 0)
            break;
        line.clear();
    }
    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.substr(24, 8), "       1");

    // The immediately preceding line must be the subcase header.
    const auto ev_pos = out.find(" OUTPUT FOR EIGENVECTOR");
    const auto sc_pos = out.rfind(" OUTPUT FOR SUBCASE", ev_pos);
    ASSERT_NE(sc_pos, std::string::npos);
    EXPECT_EQ(out.substr(sc_pos, ev_pos - sc_pos),
              " OUTPUT FOR SUBCASE        1\n");

    // Eigenvalue table data row.
    std::istringstream iss2(out);
    std::string ev_row;
    while (std::getline(iss2, ev_row)) {
        if (ev_row.size() >= 40 && ev_row.substr(1, 8) == "       1" &&
            ev_row.find("E+") != std::string::npos)
            break;
        ev_row.clear();
    }
    ASSERT_FALSE(ev_row.empty());
    EXPECT_DOUBLE_EQ(std::stod(ev_row.substr(24, 13)), 2.5e6);
    EXPECT_NEAR(std::stod(ev_row.substr(64, 13)),
                std::sqrt(2.5e6) / (2.0 * std::numbers::pi), 1e-4);
}

TEST(F06Writer, CtetrA10GpstressTableIncludesMidsideRows) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.solid_stresses.push_back(make_ctetra10_solid_stress(ElementId{8}));
    res.subcases.push_back(sc);

    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = false;
    m.analysis.subcases[0].gpstress_print = true;
    std::ostringstream oss;
    F06Writer::write(res, m, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("G R I D   P O I N T   S T R E S S E S"), std::string::npos);
    EXPECT_NE(out.find("9101"), std::string::npos);
    EXPECT_NE(out.find("9104"), std::string::npos);
    EXPECT_NE(out.find("9105"), std::string::npos);
    EXPECT_NE(out.find("9110"), std::string::npos);
}

TEST(Op2Writer, UsesMystranBarStressTableAndPayload) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.label = "BAR STRESS";
    sc.line_stresses.push_back(make_line_stress(ElementId{42}));
    res.subcases.push_back(sc);

    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = false;
    m.analysis.subcases[0].stress_plot = true;

    const auto bytes = run_op2_writer(res, m);
    const auto words = bytes_to_words(bytes);

    EXPECT_TRUE(contains_bytes(bytes, "OES1X   "));
    EXPECT_TRUE(contains_word_sequence(words, {
        11, 5, 34, 1, 1, 0, 0, 1, 1, 16, 1
    }));
    EXPECT_TRUE(contains_word_sequence(words, {
        42 * 10 + 1,
        as_word(1.25f), as_word(-2.5f), as_word(3.75f), as_word(-4.5f),
        as_word(5.125f), as_word(6.25f), as_word(-7.5f), as_word(0.0f),
        as_word(8.5f), as_word(-9.25f), as_word(10.75f), as_word(-11.5f),
        as_word(12.125f), as_word(-13.5f), as_word(0.0f)
    }));
}

TEST(Op2Writer, UsesMystranCornerStressTableAndSolidNodeIds) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.label = "SOLID STRESS";

    PlateStress ps = make_quad4_stress(ElementId{7}, 100.0, 50.0, 30.0);
    ps.nodal.push_back(make_plate_point(11, 1.0));
    ps.nodal.push_back(make_plate_point(12, 2.0));
    ps.nodal.push_back(make_plate_point(13, 3.0));
    ps.nodal.push_back(make_plate_point(14, 4.0));
    sc.plate_stresses.push_back(ps);
    sc.solid_stresses.push_back(make_solid_stress(ElementId{8}));
    res.subcases.push_back(sc);

    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = false;
    m.analysis.subcases[0].stress_plot = false;
    m.analysis.subcases[0].stress_corner_plot = true;
    m.analysis.subcases[0].gpstress_plot = true;

    const auto bytes = run_op2_writer(res, m);
    const auto words = bytes_to_words(bytes);

    EXPECT_TRUE(contains_bytes(bytes, "OES1X1  "));
    EXPECT_TRUE(contains_word_sequence(words, {
        11, 5, 144, 1, 1, 0, 0, 1, 1, 87, 1
    }));
    EXPECT_TRUE(contains_word_sequence(words, {
        11, 5, 39, 1, 1, 0, 0, 1, 1, 109, 1
    }));
    EXPECT_TRUE(contains_word(words, 9001));
    EXPECT_TRUE(contains_word(words, 9002));
    EXPECT_TRUE(contains_word(words, 9003));
    EXPECT_TRUE(contains_word(words, 9004));
}

TEST(Op2Writer, UsesGridPointStressTableForCtetra10MidsideNodes) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.label = "CTETRA10 GPSTRESS";
    sc.solid_stresses.push_back(make_ctetra10_solid_stress(ElementId{12}));
    res.subcases.push_back(sc);

    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = false;
    m.analysis.subcases[0].stress_plot = false;
    m.analysis.subcases[0].gpstress_plot = true;

    const auto bytes = run_op2_writer(res, m);
    const auto words = bytes_to_words(bytes);

    EXPECT_TRUE(contains_bytes(bytes, "OES1X1  "));
    EXPECT_TRUE(contains_word_sequence(words, {
        11, 5, 99, 1, 1, 0, 0, 1, 1, 235, 1
    }));
    EXPECT_FALSE(contains_word_sequence(words, {
        11, 5, 39, 1, 1, 0, 0, 1, 1, 109, 1
    }));
    EXPECT_TRUE(contains_word(words, 9101));
    EXPECT_TRUE(contains_word(words, 9104));
    EXPECT_TRUE(contains_word(words, 9105));
    EXPECT_TRUE(contains_word(words, 9110));
}

// ── CsvWriter ─────────────────────────────────────────────────────────────────

// Helper: write CSV and return the two file contents via ostringstreams
// by re-implementing the logic inline using temporary files.
// Since CsvWriter writes to filesystem paths, we use a tmpdir approach.
static std::pair<std::string,std::string> run_csv_writer(
    const SolverResults& res, const Model& m)
{
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "nastran_test_csv_XXXXXX";
    // create a unique stem name
    static int counter = 0;
    auto stem = fs::temp_directory_path() / ("nastran_test_csv_" + std::to_string(++counter));
    CsvWriter::write(res, m, stem);
    auto node_path = stem.string() + ".node.csv";
    auto elem_path = stem.string() + ".elem.csv";
    auto read_all = [](const std::string& p) {
        std::ifstream f(p);
        return std::string(std::istreambuf_iterator<char>(f), {});
    };
    std::string node_csv = read_all(node_path);
    std::string elem_csv = read_all(elem_path);
    fs::remove(node_path);
    fs::remove(elem_path);
    return {node_csv, elem_csv};
}

TEST(CsvWriter, NodeCsvHeaderStartsWithHash) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    EXPECT_EQ(node_csv[0], '#');
}

TEST(CsvWriter, ElemCsvHeaderStartsWithHash) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{10}, 100.0, 0.0, 0.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    EXPECT_EQ(elem_csv[0], '#');
}

TEST(CsvWriter, NodeCsvContainsExpectedColumns) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    // Header should mention T1..R3
    EXPECT_NE(node_csv.find("T1"), std::string::npos);
    EXPECT_NE(node_csv.find("R3"), std::string::npos);
}

TEST(CsvWriter, NodeCsvDataRow) {
    // Node 5 with T1=1.0; verify node_id and subcase_id appear on data row
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    // Data row: "5, 1, ..."
    EXPECT_NE(node_csv.find("5, 1"), std::string::npos);
}

TEST(CsvWriter, ElemCsvContainsId) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{42}, 100.0, 50.0, 30.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    EXPECT_NE(elem_csv.find("42"), std::string::npos);
    // No element type strings in CSV output
    EXPECT_EQ(elem_csv.find("CQUAD4"), std::string::npos);
}

TEST(CsvWriter, ElemCsvNoElementTypeString) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_tria3_stress(ElementId{99}, 10.0, 5.0, 2.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    EXPECT_NE(elem_csv.find("99"), std::string::npos);
    EXPECT_EQ(elem_csv.find("CTRIA3"), std::string::npos);
}

TEST(CsvWriter, NodeCsvOutputFlagSuppressesDisplacement) {
    SolverResults res;
    res.subcases.push_back(make_sc_with_disps());
    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].disp_print = false;
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    // Only the header line, no data rows (no node id "5")
    std::string without_header = node_csv.substr(node_csv.find('\n') + 1);
    EXPECT_TRUE(without_header.empty() || without_header.find_first_not_of("\n") == std::string::npos);
}

TEST(CsvWriter, ElemCsvOutputFlagSuppressesStress) {
    SolverResults res;
    SubCaseResults sc;
    sc.id = 1;
    sc.plate_stresses.push_back(make_quad4_stress(ElementId{1}, 100.0, 0.0, 0.0));
    res.subcases.push_back(sc);
    Model m = make_empty_model_sc1();
    m.analysis.subcases[0].stress_print = false;
    auto [node_csv, elem_csv] = run_csv_writer(res, m);
    std::string without_header = elem_csv.substr(elem_csv.find('\n') + 1);
    EXPECT_TRUE(without_header.empty() || without_header.find_first_not_of("\n") == std::string::npos);
}

// ── BdfParser: case-control output flags ─────────────────────────────────────

// ── BdfParser: case-control output flag parsing ───────────────────────────────

TEST(BdfParser, DefaultFlagsAllFalse) {
    // No output cards → all four flags default to false
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
    EXPECT_FALSE(sc.stress_print);
    EXPECT_FALSE(sc.stress_plot);
}

TEST(BdfParser, DisplacementAllSetsPrintOnly) {
    // No modifier → PRINT (F06/CSV) only; OP2 remains off
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
}

TEST(BdfParser, DisplacementPrintAllSetsPrintOnly) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PRINT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
}

TEST(BdfParser, DisplacementPlotAllSetsPlotOnly) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.disp_print);
    EXPECT_TRUE(sc.disp_plot);
}

TEST(BdfParser, DisplacementPrintPlotAllSetsBoth) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PRINT,PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.disp_print);
    EXPECT_TRUE(sc.disp_plot);
}

TEST(BdfParser, DisplacementNoneClearsBoth) {
    // NONE clears both PRINT and PLOT
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PRINT,PLOT) = ALL
  DISPLACEMENT = NONE
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
}

TEST(BdfParser, StressAllSetsPrintOnly) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.stress_print);
    EXPECT_FALSE(sc.stress_plot);
    EXPECT_FALSE(sc.stress_corner_print);
    EXPECT_FALSE(sc.stress_corner_plot);
}

TEST(BdfParser, StressPlotAllSetsPlotOnly) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.stress_print);
    EXPECT_TRUE(sc.stress_plot);
    EXPECT_FALSE(sc.stress_corner_print);
    EXPECT_FALSE(sc.stress_corner_plot);
}

TEST(BdfParser, StressPrintPlotAllSetsBoth) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS(PRINT,PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.stress_print);
    EXPECT_TRUE(sc.stress_plot);
    EXPECT_FALSE(sc.stress_corner_print);
    EXPECT_FALSE(sc.stress_corner_plot);
}

TEST(BdfParser, StressCornerDefaultsToPrintAndSetsCornerFlags) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS(CORNER) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.stress_print);
    EXPECT_FALSE(sc.stress_plot);
    EXPECT_TRUE(sc.stress_corner_print);
    EXPECT_FALSE(sc.stress_corner_plot);
}

TEST(BdfParser, StressPlotCornerSetsPlotAndCornerFlags) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS(PLOT,CORNER) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.stress_print);
    EXPECT_TRUE(sc.stress_plot);
    EXPECT_FALSE(sc.stress_corner_print);
    EXPECT_TRUE(sc.stress_corner_plot);
}

TEST(BdfParser, SpcForceAllSetsPrintOnly) {
    // No modifier → PRINT (F06) only; OP2 remains off
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  SPCFORCES = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.spc_force_print);
    EXPECT_FALSE(sc.spc_force_plot);
}

TEST(BdfParser, SpcForcePrintPlotAllSetsBoth) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  SPCFORCE(PRINT,PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.spc_force_print);
    EXPECT_TRUE(sc.spc_force_plot);
}

TEST(BdfParser, SpcForceNoneClearsBoth) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  SPCFORCE(PRINT) = ALL
  SPCFORCE = NONE
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.spc_force_print);
    EXPECT_FALSE(sc.spc_force_plot);
}

TEST(BdfParser, SpcForceDoesNotClobberSpcSet) {
    // SPCFORCES is a different case-control card than SPC; the SPC set must
    // still be parsed.
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 5
  SPCFORCES = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_EQ(sc.spc_set, SpcSetId{5});
    EXPECT_TRUE(sc.spc_force_print);
}


TEST(BdfParser, GpstressDefaultsToPrint) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  GPSTRESS = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.stress_print);
    EXPECT_FALSE(sc.stress_plot);
    EXPECT_TRUE(sc.gpstress_print);
    EXPECT_FALSE(sc.gpstress_plot);
}

TEST(BdfParser, GpstressPrintPlotSetsBothFlags) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  GPSTRESS(PRINT,PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.gpstress_print);
    EXPECT_TRUE(sc.gpstress_plot);
}

TEST(BdfParser, StressNoneClearsBoth) {
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  STRESS = NONE
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.stress_print);
    EXPECT_FALSE(sc.stress_plot);
    EXPECT_FALSE(sc.stress_corner_print);
    EXPECT_FALSE(sc.stress_corner_plot);
}

TEST(BdfParser, MultipleSubcasesIndependentFlags) {
    // SC1: disp PRINT only; SC2: stress PLOT only
    const std::string bdf = R"(
SOL 101
CEND
SUBCASE 1
  LOAD = 1
  SPC  = 1
  DISPLACEMENT(PRINT) = ALL
SUBCASE 2
  LOAD = 2
  SPC  = 1
  STRESS(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.analysis.subcases.size(), 2u);
    const auto& sc1 = m.analysis.subcases[0];
    const auto& sc2 = m.analysis.subcases[1];
    EXPECT_TRUE(sc1.disp_print);
    EXPECT_FALSE(sc1.disp_plot);
    EXPECT_FALSE(sc1.stress_print);
    EXPECT_FALSE(sc1.stress_plot);
    EXPECT_FALSE(sc2.disp_print);
    EXPECT_FALSE(sc2.disp_plot);
    EXPECT_FALSE(sc2.stress_print);
    EXPECT_TRUE(sc2.stress_plot);
}

TEST(BdfParser, GlobalOutputFlagsInheritedByExplicitSubcases) {
    const std::string bdf = R"(
SOL 101
CEND
DISPLACEMENT(PRINT,SORT1,REAL) = ALL
STRESS(PRINT,SORT1,REAL,VONMISES,BILIN) = ALL
SUBCASE 1
  LOAD = 1
  SPC  = 1
SUBCASE 2
  LOAD = 2
  SPC  = 1
  STRESS = NONE
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.analysis.subcases.size(), 2u);

    const auto& sc1 = m.analysis.subcases[0];
    EXPECT_TRUE(sc1.disp_print);
    EXPECT_FALSE(sc1.disp_plot);
    EXPECT_TRUE(sc1.stress_print);
    EXPECT_FALSE(sc1.stress_plot);
    EXPECT_TRUE(sc1.stress_corner_print);
    EXPECT_FALSE(sc1.stress_corner_plot);

    const auto& sc2 = m.analysis.subcases[1];
    EXPECT_TRUE(sc2.disp_print);
    EXPECT_FALSE(sc2.disp_plot);
    EXPECT_FALSE(sc2.stress_print);
    EXPECT_FALSE(sc2.stress_plot);
    EXPECT_FALSE(sc2.stress_corner_print);
    EXPECT_FALSE(sc2.stress_corner_plot);
}

TEST(BdfParser, GlobalLoadAndSpcDefaultsAreInheritedBySubcases) {
    const std::string bdf = R"(
SOL 101
CEND
LOAD = 7
SPC  = 9
DISPLACEMENT = ALL
SUBCASE 1
  LABEL = INHERITS DEFAULTS
SUBCASE 2
  LABEL = OVERRIDES LOAD ONLY
  LOAD = 11
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.analysis.subcases.size(), 2u);

    const auto& sc1 = m.analysis.subcases[0];
    EXPECT_EQ(sc1.load_set.value, 7);
    EXPECT_EQ(sc1.spc_set.value, 9);
    EXPECT_TRUE(sc1.disp_print);

    const auto& sc2 = m.analysis.subcases[1];
    EXPECT_EQ(sc2.load_set.value, 11);
    EXPECT_EQ(sc2.spc_set.value, 9);
    EXPECT_TRUE(sc2.disp_print);
}

// ── BdfParser: PARAM,CSVOUT,YES ───────────────────────────────────────────────

TEST(BdfParser, ParamCsvoutYes) {
    const std::string bdf = R"(
BEGIN BULK
PARAM,CSVOUT,YES
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    auto it = m.params.find("CSVOUT");
    ASSERT_NE(it, m.params.end());
    EXPECT_EQ(it->second, "YES");
}

TEST(BdfParser, ParamCsvoutAbsent) {
    const std::string bdf = R"(
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    EXPECT_EQ(m.params.find("CSVOUT"), m.params.end());
}

// ── BdfParser: SOL 103 case-control EIGENVECTOR / DISPLACEMENT flags ──────────

TEST(BdfParser, Sol103EigenvectorPrintSetsEigvecPrint) {
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  EIGENVECTOR(PRINT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.eigvec_print);
    EXPECT_FALSE(sc.eigvec_plot);
}

TEST(BdfParser, Sol103EigenvectorPlotSetsEigvecPlot) {
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  EIGENVECTOR(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.eigvec_print);
    EXPECT_TRUE(sc.eigvec_plot);
}

TEST(BdfParser, Sol103EigenvectorNoModifierSetsPrintOnly) {
    // EIGENVECTOR = ALL with no modifier is equivalent to PRINT
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  EIGENVECTOR = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.eigvec_print);
    EXPECT_FALSE(sc.eigvec_plot);
}

TEST(BdfParser, Sol103DisplacementPlotSetsDispPlot) {
    // DISPLACEMENT(PLOT)=ALL in SOL 103 sets disp_plot; the modal solver
    // maps disp_plot → eigvec_print (F06) and eigvec_plot (OP2).
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  DISPLACEMENT(PLOT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_FALSE(sc.disp_print);
    EXPECT_TRUE(sc.disp_plot);
    // eigvec flags remain false at parse time; mapping happens in ModalSolver
    EXPECT_FALSE(sc.eigvec_print);
    EXPECT_FALSE(sc.eigvec_plot);
}

TEST(BdfParser, Sol103DisplacementPrintSetsDispPrint) {
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 1
  DISPLACEMENT(PRINT) = ALL
BEGIN BULK
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    const auto& sc = m.analysis.subcases[0];
    EXPECT_TRUE(sc.disp_print);
    EXPECT_FALSE(sc.disp_plot);
}

TEST(BdfParser, Sol103EigrlMethodParsed) {
    const std::string bdf = R"(
SOL 103
CEND
SUBCASE 1
  METHOD = 5
  EIGENVECTOR = ALL
BEGIN BULK
EIGRL,5,0.0,,10
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_FALSE(m.analysis.subcases.empty());
    EXPECT_EQ(m.analysis.subcases[0].eigrl_id, 5);
    ASSERT_TRUE(m.eigrls.count(5));
    EXPECT_EQ(m.eigrls.at(5).nd, 10);
    EXPECT_DOUBLE_EQ(m.eigrls.at(5).v1, 0.0);
}
