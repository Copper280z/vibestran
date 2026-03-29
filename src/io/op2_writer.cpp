// src/io/op2_writer.cpp
// MSC Nastran OP2 binary writer, little-endian, post=-1 format.
//
// Format reference: pyNastran op2_writer.py / table_object.py / oes_*.py
//
// Record encoding: Fortran unformatted sequential I/O
//   [int32 nbytes] [nbytes of data] [int32 nbytes]
//
// File structure:
//   File header (MSC post=-1 tape-code block)
//   For each result table (OUGV1, OES1X per element-type):
//     Table header:  [4,2,4][8,name8,8][4,-1,4][4,7,4][28,magic7,28]
//                    [4,-2,4][4,1,4][4,0,4][4,7,4][28,subtable_date,28]
//     For each subcase (static → one time-step):
//       Table-3 record (584-byte metadata header)
//       Table-4 record (float32 result data)
//     Inter-result footer markers
//   Table close: [4,0,4]
//   File close:  [4,0,4]

#include "io/op2_writer.hpp"
#include <algorithm>
#include <cstring>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vibestran {

// ── Primitive I/O helpers ─────────────────────────────────────────────────────

static constexpr int32_t kMystranDeviceCode = 1;
static constexpr int32_t kRealFormatCode = 1;
static constexpr int32_t kVonMisesStressCode = 1;

static int32_t f32_as_i32(float v) {
    int32_t i;
    std::memcpy(&i, &v, 4);
    return i;
}

static void write_i32(std::ostream& f, int32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}

static void write_f32(std::ostream& f, float v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}

/// Write a Fortran unformatted record: [nbytes][data...][nbytes]
static void write_record(std::ostream& f, const void* data, int32_t nbytes) {
    write_i32(f, nbytes);
    f.write(reinterpret_cast<const char*>(data), nbytes);
    write_i32(f, nbytes);
}

/// Write a single int32 as a Fortran record of 4 bytes.
static void write_record_i32(std::ostream& f, int32_t v) {
    write_record(f, &v, 4);
}

/// Write a series of int32 Fortran-record markers: [4,v,4] per value.
template<typename... Vs>
static void write_markers(std::ostream& f, Vs... vs) {
    const int32_t vals[] = {static_cast<int32_t>(vs)...};
    for (int32_t v : vals) write_record_i32(f, v);
}

enum class SolidNodeOutput {
    Corner,
    AllGridPoints,
};

static const SubCase* find_model_subcase(const Model& model, int sc_id) {
    const auto it = std::find_if(
        model.analysis.subcases.begin(), model.analysis.subcases.end(),
        [&](const SubCase& sc) { return sc.id == sc_id; });
    return (it == model.analysis.subcases.end()) ? nullptr : &*it;
}

static bool has_plot_displacement_output(const Model& model, int sc_id) {
    const SubCase* sc = find_model_subcase(model, sc_id);
    return (sc != nullptr) && sc->disp_plot;
}

static bool has_standard_stress_plot_output(const Model& model, int sc_id) {
    const SubCase* sc = find_model_subcase(model, sc_id);
    return (sc != nullptr) && sc->stress_plot;
}

static bool has_line_stress_plot_output(const Model& model, int sc_id) {
    const SubCase* sc = find_model_subcase(model, sc_id);
    return (sc != nullptr) && (sc->stress_plot || sc->stress_corner_plot);
}

static bool has_corner_stress_plot_output(const Model& model, int sc_id) {
    const SubCase* sc = find_model_subcase(model, sc_id);
    return (sc != nullptr) && sc->stress_corner_plot;
}

static bool has_gpstress_plot_output(const Model& model, int sc_id) {
    const SubCase* sc = find_model_subcase(model, sc_id);
    return (sc != nullptr) && sc->gpstress_plot;
}

static bool solid_has_extra_grid_points(const ElementType etype) {
    return etype == ElementType::CTETRA10;
}

static bool has_plate_stress_type(const SubCaseResults& sc, ElementType etype) {
    return std::any_of(
        sc.plate_stresses.begin(), sc.plate_stresses.end(),
        [&](const auto& ps) { return ps.etype == etype; });
}

static bool has_solid_stress_type(const SubCaseResults& sc, ElementType etype) {
    return std::any_of(
        sc.solid_stresses.begin(), sc.solid_stresses.end(),
        [&](const auto& ss) { return ss.etype == etype; });
}

static bool has_line_stresses(const SubCaseResults& sc) {
    return !sc.line_stresses.empty();
}

// ── OP2 table header ──────────────────────────────────────────────────────────
// MYSTRAN writes the second 28-byte record as seven integers, not an OUG1 tag.
// Layout:
//   [4,2,4][8,name8,8]
//   [4,-1,4][4,7,4]
//   [28, 102,0,0,0,512,0,0, 28]
//   [4,-2,4][4,1,4][4,0,4]
//   [4,7,4]
//   [28, 0,1,month,day,dyear,0,1, 28]

static void write_table_header(std::ostream& f, const char name8[8],
                                int month, int day, int dyear) {
    // [4][2][4] [8][name8][8]
    {
        const int32_t m[3] = {4, 2, 4};
        f.write(reinterpret_cast<const char*>(m), 12);
        int32_t r = 8;
        f.write(reinterpret_cast<const char*>(&r), 4);
        f.write(name8, 8);
        f.write(reinterpret_cast<const char*>(&r), 4);
    }
    // [4,-1,4] [4,7,4]
    {
        const int32_t m1[3] = {4, -1, 4};
        const int32_t m2[3] = {4,  7, 4};
        f.write(reinterpret_cast<const char*>(m1), 12);
        f.write(reinterpret_cast<const char*>(m2), 12);
    }
    // [28][102,0,0,0,512,0,0][28]
    {
        const int32_t rec[9] = {28, 102, 0, 0, 0, 512, 0, 0, 28};
        f.write(reinterpret_cast<const char*>(rec), 36);
    }
    // [4,-2,4][4,1,4][4,0,4]
    {
        const int32_t m[9] = {4,-2,4, 4,1,4, 4,0,4};
        f.write(reinterpret_cast<const char*>(m), 36);
    }
    // [4,7,4][28][0,1,month,day,dyear,0,1][28]
    {
        const int32_t m3[3] = {4, 7, 4};
        f.write(reinterpret_cast<const char*>(m3), 12);
        const int32_t rec[9] = {28, 0, 1, month, day, dyear, 0, 1, 28};
        f.write(reinterpret_cast<const char*>(rec), 36);
    }
}

// ── TABLE-3 record (584-byte subcase metadata) ────────────────────────────────

static void write_table3(std::ostream& f, bool new_result, int itable,
                         int analysis_code, int device_code,
                         int table_code, int element_type, int isubcase,
                         int field5_int, float field6, float field7,
                         int field8_int, int num_wide, int code11_int,
                         const std::string& title,
                         const std::string& subtitle,
                         const std::string& label) {
    // Prefix markers before the 584-byte record
    if (new_result && itable != -3) {
        const int32_t m[3] = {4, 146, 4};
        f.write(reinterpret_cast<const char*>(m), 12);
    } else {
        const int32_t m[12] = {4, itable, 4, 4, 1, 4, 4, 0, 4, 4, 146, 4};
        f.write(reinterpret_cast<const char*>(m), 48);
    }

    // Build the 584-byte payload: 50 × int32 + 3 × 128-char strings
    static constexpr int N_INTS = 50;
    static constexpr int STR_LEN = 128;
    int32_t ints[N_INTS] = {};
    ints[0]  = analysis_code * 10 + device_code; // aCode
    ints[1]  = table_code;      // tCode
    ints[2]  = element_type;    // element_type (0 for OUGV1)
    ints[3]  = isubcase;        // isubcase
    ints[4]  = field5_int;      // mode/subcase/load-step field
    ints[5]  = f32_as_i32(field6);
    ints[6]  = f32_as_i32(field7);
    ints[7]  = field8_int;      // random_code for OUG, load_set for OES
    ints[8]  = kRealFormatCode; // format_code = real
    ints[9]  = num_wide;        // num_wide
    ints[10] = code11_int;      // oCode for OUG / stress_code for OES
    ints[11] = 0;               // acoustic flag
    ints[22] = 0;               // thermal

    char strings[3][STR_LEN];
    auto fill_str = [&](char dst[STR_LEN], const std::string& src) {
        std::memset(dst, ' ', STR_LEN);
        size_t n = std::min(src.size(), (size_t)STR_LEN);
        std::memcpy(dst, src.data(), n);
    };
    fill_str(strings[0], title);
    fill_str(strings[1], subtitle);
    fill_str(strings[2], label);

    int32_t rec_len = N_INTS * 4 + 3 * STR_LEN; // = 584
    f.write(reinterpret_cast<const char*>(&rec_len), 4);
    f.write(reinterpret_cast<const char*>(ints), N_INTS * 4);
    f.write(strings[0], STR_LEN);
    f.write(strings[1], STR_LEN);
    f.write(strings[2], STR_LEN);
    f.write(reinterpret_cast<const char*>(&rec_len), 4);
}

// ── File header (MSC post=-1) ─────────────────────────────────────────────────

static void write_file_header(std::ostream& f, int month, int day, int dyear) {
    // [4,3,4]
    {
        const int32_t m[3] = {4, 3, 4};
        f.write(reinterpret_cast<const char*>(m), 12);
    }
    // [12, day, month, dyear, 12][4,7,4][28, tape_code, 28]
    {
        const int32_t date_rec[5] = {12, day, month, dyear, 12};
        f.write(reinterpret_cast<const char*>(date_rec), 20);

        const int32_t m2[3] = {4, 7, 4};
        f.write(reinterpret_cast<const char*>(m2), 12);

        const char tape[] = "NASTRAN FORT TAPE ID CODE - ";  // 28 chars
        int32_t r = 28;
        f.write(reinterpret_cast<const char*>(&r), 4);
        f.write(tape, 28);
        f.write(reinterpret_cast<const char*>(&r), 4);
    }
    // [4,2,4][8, 'XXXXXXXX', 8]
    {
        const int32_t m3[3] = {4, 2, 4};
        f.write(reinterpret_cast<const char*>(m3), 12);
        int32_t r = 8;
        f.write(reinterpret_cast<const char*>(&r), 4);
        const char ver[] = "XXXXXXXX";
        f.write(ver, 8);
        f.write(reinterpret_cast<const char*>(&r), 4);
    }
    // [4,-1,4][4,0,4]
    {
        const int32_t m4[6] = {4,-1,4, 4,0,4};
        f.write(reinterpret_cast<const char*>(m4), 24);
    }
}

// ── int32 viewed as float32 helper ───────────────────────────────────────────

static float i32_as_f32(int32_t v) {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

// ── Write one displacement TABLE-4 data record ───────────────────────────────
// OUGV1: each node → [node_id*10+device_code, grid_type, T1..R3] as float32.

static int write_ougv1(std::ostream& f,
                       const SubCaseResults& sc,
                       bool new_result, int itable,
                       const std::string& subtitle) {
    if (sc.displacements.empty()) return itable;

    const int analysis_code = 1;
    const int table_code    = 1;
    const int num_wide      = 8;
    const int nn = static_cast<int>(sc.displacements.size());
    const int ntotal = nn * num_wide; // total 32-bit words in data record

    // TABLE-3
    write_table3(f, new_result, itable,
                 analysis_code, kMystranDeviceCode,
                 table_code, /*element_type=*/0, sc.id,
                 /*field5_int=*/1, /*field6=*/0.0f, /*field7=*/0.0f,
                 /*field8_int=*/0, num_wide, /*code11_int=*/0,
                 /*title=*/"", subtitle, sc.label);
    itable--;

    // TABLE-4 header markers
    {
        const int32_t m[13] = {4, itable, 4,
                               4, 1, 4,
                               4, 0, 4,
                               4, ntotal, 4,
                               4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    // TABLE-4 data: [node_id*10+device_code (as f32), grid_type (as f32), T1..R3 (f32)] per node
    for (const auto& nd : sc.displacements) {
        int32_t id_dev = static_cast<int32_t>(nd.node.value) * 10 + kMystranDeviceCode;
        write_f32(f, i32_as_f32(id_dev));
        write_f32(f, i32_as_f32(1));   // grid_type G=1
        for (int i = 0; i < 6; ++i)
            write_f32(f, static_cast<float>(nd.d[i]));
    }

    // TABLE-4 trailing length marker
    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    itable--;
    return itable;
}

// ── Write one OES1X plate stress TABLE-4 ─────────────────────────────────────
// CQUAD4 (element_type=33) or CTRIA3 (element_type=74), centroid only.
// num_wide=17 per element:
//   [eid_device, fd1, sx1,sy1,txy1,angle1,major1,minor1,vm1,
//                fd2, sx2,sy2,txy2,angle2,major2,minor2,vm2]

static int write_oes1x_plate(std::ostream& f,
                              const SubCaseResults& sc,
                              bool new_result, int itable,
                              ElementType etype,
                              const std::string& subtitle) {
    // Count matching elements
    const int nel = static_cast<int>(std::count_if(
        sc.plate_stresses.begin(), sc.plate_stresses.end(),
        [&](const auto& ps) { return ps.etype == etype; }));
    if (nel == 0) return itable;

    const int analysis_code = 1;
    const int table_code    = 5;
    const int num_wide      = 17;
    const int etype_id = (etype == ElementType::CQUAD4) ? 33 : 74;
    const int ntotal = nel * num_wide;

    write_table3(f, new_result, itable,
                 analysis_code, kMystranDeviceCode,
                 table_code, etype_id, sc.id,
                 /*field5_int=*/sc.id, /*field6=*/0.0f, /*field7=*/0.0f,
                 /*field8_int=*/1, num_wide, kVonMisesStressCode,
                 "", subtitle, sc.label);
    itable--;

    {
        const int32_t m[13] = {4, itable, 4,
                               4, 1, 4,
                               4, 0, 4,
                               4, ntotal, 4,
                               4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    for (const auto& ps : sc.plate_stresses) {
        if (ps.etype != etype) continue;
        double major, minor, angle_deg;
        compute_principal_2d(ps.sx, ps.sy, ps.sxy, major, minor, angle_deg);

        int32_t eid_dev = static_cast<int32_t>(ps.eid.value) * 10 + kMystranDeviceCode;
        // Layer 1 (fiber_dist = 0, same membrane stress at both layers)
        write_f32(f, i32_as_f32(eid_dev));
        write_f32(f, 0.0f);  // fiber dist
        write_f32(f, static_cast<float>(ps.sx));
        write_f32(f, static_cast<float>(ps.sy));
        write_f32(f, static_cast<float>(ps.sxy));
        write_f32(f, static_cast<float>(angle_deg));
        write_f32(f, static_cast<float>(major));
        write_f32(f, static_cast<float>(minor));
        write_f32(f, static_cast<float>(ps.von_mises));
        // Layer 2 (same as layer 1 for membrane-only)
        write_f32(f, 0.0f);
        write_f32(f, static_cast<float>(ps.sx));
        write_f32(f, static_cast<float>(ps.sy));
        write_f32(f, static_cast<float>(ps.sxy));
        write_f32(f, static_cast<float>(angle_deg));
        write_f32(f, static_cast<float>(major));
        write_f32(f, static_cast<float>(minor));
        write_f32(f, static_cast<float>(ps.von_mises));
    }

    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    itable--;
    return itable;
}

// ── Write one OES1X1 CQUAD4 corner/grid-point stress TABLE-4 ────────────────
// MYSTRAN CQUAD4 corner output uses element_type=144, num_wide=87:
//   [eid_device, 'CEN/', 4,
//    centroid upper/lower 16 words,
//    node1 + upper/lower 16 words,
//    node2 + upper/lower 16 words,
//    node3 + upper/lower 16 words,
//    node4 + upper/lower 16 words]

static int write_oes1x_cquad4_corner(std::ostream& f,
                                     const SubCaseResults& sc,
                                     bool new_result, int itable,
                                     const std::string& subtitle) {
    const int nel = static_cast<int>(std::count_if(
        sc.plate_stresses.begin(), sc.plate_stresses.end(),
        [&](const auto& ps) { return ps.etype == ElementType::CQUAD4; }));
    if (nel == 0) return itable;

    const int analysis_code = 1;
    const int table_code = 5;
    const int element_type = 144;
    const int num_wide = 87;
    const int ntotal = nel * num_wide;

    write_table3(f, new_result, itable,
                 analysis_code, kMystranDeviceCode,
                 table_code, element_type, sc.id,
                 /*field5_int=*/sc.id, /*field6=*/0.0f, /*field7=*/0.0f,
                 /*field8_int=*/1, num_wide, kVonMisesStressCode,
                 "", subtitle, sc.label);
    itable--;

    {
        const int32_t m[13] = {4, itable, 4,
                               4, 1, 4,
                               4, 0, 4,
                               4, ntotal, 4,
                               4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    int32_t cen_bytes;
    {
        const char cen[] = {'C','E','N','/'};
        std::memcpy(&cen_bytes, cen, 4);
    }

    auto write_point = [&](double sx, double sy, double sxy, double vm) {
        double major, minor, angle_deg;
        compute_principal_2d(sx, sy, sxy, major, minor, angle_deg);
        write_f32(f, 0.0f);
        write_f32(f, static_cast<float>(sx));
        write_f32(f, static_cast<float>(sy));
        write_f32(f, static_cast<float>(sxy));
        write_f32(f, static_cast<float>(angle_deg));
        write_f32(f, static_cast<float>(major));
        write_f32(f, static_cast<float>(minor));
        write_f32(f, static_cast<float>(vm));
        write_f32(f, 0.0f);
        write_f32(f, static_cast<float>(sx));
        write_f32(f, static_cast<float>(sy));
        write_f32(f, static_cast<float>(sxy));
        write_f32(f, static_cast<float>(angle_deg));
        write_f32(f, static_cast<float>(major));
        write_f32(f, static_cast<float>(minor));
        write_f32(f, static_cast<float>(vm));
    };

    for (const auto& ps : sc.plate_stresses) {
        if (ps.etype != ElementType::CQUAD4) continue;

        const int32_t eid_dev =
            static_cast<int32_t>(ps.eid.value) * 10 + kMystranDeviceCode;
        write_f32(f, i32_as_f32(eid_dev));
        write_f32(f, i32_as_f32(cen_bytes));
        write_f32(f, i32_as_f32(4));
        write_point(ps.sx, ps.sy, ps.sxy, ps.von_mises);

        for (int n = 0; n < 4; ++n) {
            if (n < static_cast<int>(ps.nodal.size())) {
                const auto& point = ps.nodal[static_cast<std::size_t>(n)];
                write_f32(f, i32_as_f32(static_cast<int32_t>(point.node.value)));
                write_point(point.sx, point.sy, point.sxy, point.von_mises);
            } else {
                write_f32(f, i32_as_f32(static_cast<int32_t>(n + 1)));
                write_point(ps.sx, ps.sy, ps.sxy, ps.von_mises);
            }
        }
    }

    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    itable--;
    return itable;
}

// ── Write one OES1X bar stress TABLE-4 ───────────────────────────────────────
// MYSTRAN uses a 16-word CBAR/CBEAM record:
//   [eid_device,
//    s1a,s2a,s3a,s4a,axial,smaxa,smina,MS_tension,
//    s1b,s2b,s3b,s4b,smaxb,sminb,MS_compression]

static int write_oes1x_line(std::ostream& f,
                            const SubCaseResults& sc,
                            bool new_result, int itable,
                            const std::string& subtitle) {
    const int nel = static_cast<int>(sc.line_stresses.size());
    if (nel == 0) return itable;

    const int analysis_code = 1;
    const int table_code = 5;
    const int element_type = 34;
    const int num_wide = 16;
    const int ntotal = nel * num_wide;

    write_table3(f, new_result, itable,
                 analysis_code, kMystranDeviceCode,
                 table_code, element_type, sc.id,
                 /*field5_int=*/sc.id, /*field6=*/0.0f, /*field7=*/0.0f,
                 /*field8_int=*/1, num_wide, kVonMisesStressCode,
                 "", subtitle, sc.label);
    itable--;

    {
        const int32_t m[13] = {4, itable, 4,
                               4, 1, 4,
                               4, 0, 4,
                               4, ntotal, 4,
                               4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    for (const auto& ls : sc.line_stresses) {
        const int32_t eid_dev =
            static_cast<int32_t>(ls.eid.value) * 10 + kMystranDeviceCode;
        write_f32(f, i32_as_f32(eid_dev));

        for (double s : ls.end_a.s) write_f32(f, static_cast<float>(s));
        write_f32(f, static_cast<float>(ls.end_a.axial));
        write_f32(f, static_cast<float>(ls.end_a.smax));
        write_f32(f, static_cast<float>(ls.end_a.smin));
        write_f32(f, 0.0f); // MYSTRAN margin-of-safety placeholder

        for (double s : ls.end_b.s) write_f32(f, static_cast<float>(s));
        write_f32(f, static_cast<float>(ls.end_b.smax));
        write_f32(f, static_cast<float>(ls.end_b.smin));
        write_f32(f, 0.0f); // MYSTRAN margin-of-safety placeholder
    }

    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    itable--;
    return itable;
}

// ── Write one OES1X solid stress TABLE-4 ─────────────────────────────────────
// CTETRA4 (element_type=39, num_wide=109) or CHEXA8 (element_type=67, num_wide=193).
// We have centroid data only; corner nodes are filled with the centroid values.
//
// Header (4 words per element): [eid_device, cid=0, 'CEN/', ngrids]
// Per node (21 words):
//   [node_id, oxx, txy, o1, dir1y, dir1z, dir1x, p, vm,
//             oyy, tyz, o2, dir2y, dir2z, dir2x,
//             ozz, txz, o3, dir3y, dir3z, dir3x]

static int write_oes1x_solid(std::ostream& f,
                              const SubCaseResults& sc,
                              bool new_result, int itable,
                              ElementType etype,
                              SolidNodeOutput node_output,
                              const std::string& subtitle) {
    const int nel = static_cast<int>(std::count_if(
        sc.solid_stresses.begin(), sc.solid_stresses.end(),
        [&](const auto& ss) { return ss.etype == etype; }));
    if (nel == 0) return itable;

    int etype_id, ngrids;
    if (etype == ElementType::CTETRA4) {
        etype_id = 39;
        ngrids   = 4;
    } else if (etype == ElementType::CTETRA10) {
        if (node_output == SolidNodeOutput::Corner) {
            etype_id = 39;
            ngrids   = 4;
        } else {
            etype_id = 99;
            ngrids   = 10;
        }
    } else if (etype == ElementType::CPENTA6) {
        etype_id = 68;
        ngrids   = 6;
    } else {
        etype_id = 67;  // CHEXA8
        ngrids   = 8;
    }

    // nnodes_per_element = ngrids + 1 (centroid)
    const int nnodes_per_elem = ngrids + 1;
    const int num_wide        = 4 + 21 * nnodes_per_elem;
    const int ntotal          = nel * num_wide;

    const int analysis_code = 1;
    const int table_code    = 5;

    write_table3(f, new_result, itable,
                 analysis_code, kMystranDeviceCode,
                 table_code, etype_id, sc.id,
                 /*field5_int=*/sc.id, /*field6=*/0.0f, /*field7=*/0.0f,
                 /*field8_int=*/1, num_wide, kVonMisesStressCode,
                 "", subtitle, sc.label);
    itable--;

    {
        const int32_t m[13] = {4, itable, 4,
                               4, 1, 4,
                               4, 0, 4,
                               4, ntotal, 4,
                               4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    // 'CEN/' as 4 bytes interpreted as int32
    int32_t cen_bytes;
    {
        const char cen[] = {'C','E','N','/'};
        std::memcpy(&cen_bytes, cen, 4);
    }

    for (const auto& ss : sc.solid_stresses) {
        if (ss.etype != etype) continue;

        // Element header: [eid_device, cid=0, 'CEN/', ngrids]
        int32_t eid_dev =
            static_cast<int32_t>(ss.eid.value) * 10 + kMystranDeviceCode;
        write_f32(f, i32_as_f32(eid_dev));
        write_f32(f, i32_as_f32(0));        // cid
        write_f32(f, i32_as_f32(cen_bytes)); // 'CEN/'
        write_f32(f, i32_as_f32(ngrids));    // number of corner nodes

        auto write_node_block = [&](int32_t node_id,
                                    double sx, double sy, double sz,
                                    double sxy, double syz, double szx,
                                    double vm) {
            double p[3], v[3][3];
            compute_principal_3d(sx, sy, sz, sxy, syz, szx, p, v);
            const double pressure = -(p[0] + p[1] + p[2]) / 3.0;
            // [node_id, oxx, txy, o1, dir1y, dir1z, dir1x, p, vm,
            //           oyy, tyz, o2, dir2y, dir2z, dir2x,
            //           ozz, txz, o3, dir3y, dir3z, dir3x]
            write_f32(f, i32_as_f32(node_id));
            write_f32(f, static_cast<float>(sx));
            write_f32(f, static_cast<float>(sxy));
            write_f32(f, static_cast<float>(p[0]));
            write_f32(f, static_cast<float>(v[0][1])); // dir1y
            write_f32(f, static_cast<float>(v[0][2])); // dir1z
            write_f32(f, static_cast<float>(v[0][0])); // dir1x
            write_f32(f, static_cast<float>(pressure));
            write_f32(f, static_cast<float>(vm));

            write_f32(f, static_cast<float>(sy));
            write_f32(f, static_cast<float>(syz));
            write_f32(f, static_cast<float>(p[1]));
            write_f32(f, static_cast<float>(v[1][1]));
            write_f32(f, static_cast<float>(v[1][2]));
            write_f32(f, static_cast<float>(v[1][0]));

            write_f32(f, static_cast<float>(sz));
            write_f32(f, static_cast<float>(szx));
            write_f32(f, static_cast<float>(p[2]));
            write_f32(f, static_cast<float>(v[2][1]));
            write_f32(f, static_cast<float>(v[2][2]));
            write_f32(f, static_cast<float>(v[2][0]));
        };

        write_node_block(0, ss.sx, ss.sy, ss.sz, ss.sxy, ss.syz, ss.szx, ss.von_mises);
        for (int n = 0; n < ngrids; ++n) {
            if (n < static_cast<int>(ss.nodal.size())) {
                const auto& point = ss.nodal[static_cast<size_t>(n)];
                write_node_block(static_cast<int32_t>(point.node.value),
                                 point.sx, point.sy, point.sz,
                                 point.sxy, point.syz, point.szx,
                                 point.von_mises);
            } else {
                write_node_block(static_cast<int32_t>(n + 1),
                                 ss.sx, ss.sy, ss.sz,
                                 ss.sxy, ss.syz, ss.szx,
                                 ss.von_mises);
            }
        }
    }

    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    itable--;
    return itable;
}

// LAMA table removed — eigenvalue data is embedded in each OUGV1 TABLE-3 record.

// ── Write OUGV1 eigenvector records (one full table per mode, mystran format) ──
// approach_code=21: analysis_code=2 (real modes), device_code=1 (print).
// TABLE-3 ints[5] = eigenvalue as float32 bitcast, ints[6] = radians as float32 bitcast.
// One complete table (header + TABLE-3/4 pair) is written per mode, matching the
// mystran output format expected by community tools (pyNastran, Femap, etc.).

static void write_ougv1_modal_mode(std::ostream& f,
                                   const ModalSubCaseResults& msc,
                                   const ModeResult& mr,
                                   int month, int day, int dyear) {
    if (mr.shape.empty()) return;

    const int analysis_code = 2;
    const int table_code    = 7;
    const int num_wide      = 8;
    const int nn     = static_cast<int>(mr.shape.size());
    const int ntotal = nn * num_wide;

    // Write fresh table header for this mode
    const char name8[8] = {'O','U','G','V','1',' ',' ',' '};
    write_table_header(f, name8, month, day, dyear);

    write_table3(f, /*new_result=*/false, /*itable=*/-3,
                 analysis_code, kMystranDeviceCode,
                 table_code, /*element_type=*/0, msc.id,
                 /*field5_int=*/mr.mode_number,
                 static_cast<float>(mr.eigenvalue),
                 static_cast<float>(mr.radians_per_sec),
                 /*field8_int=*/0, num_wide, /*code11_int=*/0,
                 "", "", msc.label);

    // TABLE-4 header markers: [-4, 1, 0, ntotal]
    {
        const int32_t m[13] = {4, -4, 4,
                               4, 1, 4,
                               4, 0, 4,
                               4, ntotal, 4,
                               4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    // TABLE-4 data: one record per node [node_id*10+device_code (as f32), grid_type=1, T1..R3]
    for (const auto& nd : mr.shape) {
        int32_t id_dev = static_cast<int32_t>(nd.node.value) * 10 + kMystranDeviceCode;
        write_f32(f, i32_as_f32(id_dev));
        write_f32(f, i32_as_f32(1)); // grid type G=1
        for (int i = 0; i < 6; ++i)
            write_f32(f, static_cast<float>(nd.d[i]));
    }

    // TABLE-4 trailing length marker + footer
    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    // Footer: [-5, 1, 0] then table close [0]
    {
        const int32_t footer[9] = {4, -5, 4, 4, 1, 4, 4, 0, 4};
        f.write(reinterpret_cast<const char*>(footer), 36);
    }
    write_markers(f, 0); // table close
}

// ── Op2Writer::write_modal ────────────────────────────────────────────────────

void Op2Writer::write_modal(const ModalSolverResults& results, const Model& model,
                            const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw SolverError(std::format("Cannot write OP2: {}", path.string()));

    std::time_t now = std::time(nullptr);
    const std::tm* lt = std::localtime(&now);
    int month = lt->tm_mon + 1;
    int day   = lt->tm_mday;
    int dyear = lt->tm_year - 100;

    write_file_header(f, month, day, dyear);

    // ── OUGV1 (eigenvectors) ──────────────────────────────────────────────────
    // One complete OUGV1 table is written per mode, matching the mystran format.
    // eigvec_plot flag is taken from ModalSubCaseResults (which already applies
    // the DISPLACEMENT→EIGENVECTOR alias performed in the modal solver).
    for (const auto& msc : results.subcases) {
        if (!msc.eigvec_plot || msc.modes.empty()) continue;
        for (const auto& mr : msc.modes)
            write_ougv1_modal_mode(f, msc, mr, month, day, dyear);
    }

    (void)model;

    write_markers(f, 0); // file end
}

// ── Op2Writer::write ──────────────────────────────────────────────────────────

void Op2Writer::write(const SolverResults& results, const Model& model,
                      const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw SolverError(std::format("Cannot write OP2: {}", path.string()));

    // Date for header
    std::time_t now = std::time(nullptr);
    const std::tm* lt = std::localtime(&now);
    int month = lt->tm_mon + 1;
    int day   = lt->tm_mday;
    int dyear = lt->tm_year - 100; // years since 2000

    write_file_header(f, month, day, dyear);

    // ── OUGV1 table (displacements) ───────────────────────────────────────────
    {
        const bool any_disp = std::any_of(
            results.subcases.begin(), results.subcases.end(),
            [&](const auto& sc) {
                return has_plot_displacement_output(model, sc.id) &&
                       !sc.displacements.empty();
            });
        if (any_disp) {
            const char name8[8] = {'O','U','G','V','1',' ',' ',' '};
            write_table_header(f, name8, month, day, dyear);

            int itable = -1;
            bool new_result = true;
            for (const auto& sc : results.subcases) {
                if (!has_plot_displacement_output(model, sc.id) ||
                    sc.displacements.empty())
                    continue;
                if (itable == -1) { itable = -3; }
                const int prev_itable = itable;
                itable = write_ougv1(f, sc, new_result, itable, sc.label);
                if (itable == prev_itable) continue;
                new_result = false;
                // Inter-result footer
                const int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
                f.write(reinterpret_cast<const char*>(footer), 36);
            }
            // Table close
            write_markers(f, 0);
        }
    }

    // ── OES1X table for CBAR/CBEAM line stresses ─────────────────────────────
    {
        const bool any = std::any_of(
            results.subcases.begin(), results.subcases.end(),
            [&](const auto& sc) {
                return has_line_stress_plot_output(model, sc.id) &&
                       has_line_stresses(sc);
            });
        if (any) {
            const char name8[8] = {'O','E','S','1','X',' ',' ',' '};
            write_table_header(f, name8, month, day, dyear);

            int itable = -3;
            bool new_result = true;
            for (const auto& sc : results.subcases) {
                if (!has_line_stress_plot_output(model, sc.id)) continue;
                const int prev_itable = itable;
                itable = write_oes1x_line(f, sc, new_result, itable, sc.label);
                if (itable == prev_itable) continue;
                if (!new_result || itable < -3) new_result = false;
                const int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
                f.write(reinterpret_cast<const char*>(footer), 36);
            }
            write_markers(f, 0);
        }
    }

    // ── OES1X1 table for CQUAD4 centroid stresses ────────────────────────────
    {
        const bool any = std::any_of(
            results.subcases.begin(), results.subcases.end(),
            [&](const auto& sc) {
                return has_standard_stress_plot_output(model, sc.id) &&
                       !has_corner_stress_plot_output(model, sc.id) &&
                       has_plate_stress_type(sc, ElementType::CQUAD4);
            });
        if (any) {
            const char name8[8] = {'O','E','S','1','X','1',' ',' '};
            write_table_header(f, name8, month, day, dyear);

            int itable = -3;
            bool new_result = true;
            for (const auto& sc : results.subcases) {
                if (!has_standard_stress_plot_output(model, sc.id) ||
                    has_corner_stress_plot_output(model, sc.id))
                    continue;
                const int prev_itable = itable;
                itable = write_oes1x_plate(f, sc, new_result, itable,
                                           ElementType::CQUAD4, sc.label);
                if (itable == prev_itable) continue;
                if (!new_result || itable < -3) new_result = false;
                const int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
                f.write(reinterpret_cast<const char*>(footer), 36);
            }
            write_markers(f, 0);
        }
    }

    // ── OES1X1 table for CQUAD4 corner/grid-point stresses ───────────────────
    {
        const bool any = std::any_of(
            results.subcases.begin(), results.subcases.end(),
            [&](const auto& sc) {
                return (has_corner_stress_plot_output(model, sc.id) ||
                        has_gpstress_plot_output(model, sc.id)) &&
                       has_plate_stress_type(sc, ElementType::CQUAD4);
            });
        if (any) {
            const char name8[8] = {'O','E','S','1','X','1',' ',' '};
            write_table_header(f, name8, month, day, dyear);

            int itable = -3;
            bool new_result = true;
            for (const auto& sc : results.subcases) {
                if (!has_corner_stress_plot_output(model, sc.id) &&
                    !has_gpstress_plot_output(model, sc.id))
                    continue;
                const int prev_itable = itable;
                itable = write_oes1x_cquad4_corner(f, sc, new_result, itable,
                                                   sc.label);
                if (itable == prev_itable) continue;
                if (!new_result || itable < -3) new_result = false;
                const int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
                f.write(reinterpret_cast<const char*>(footer), 36);
            }
            write_markers(f, 0);
        }
    }

    // ── OES1X1 table for CTRIA3 ───────────────────────────────────────────────
    {
        const bool any = std::any_of(
            results.subcases.begin(), results.subcases.end(),
            [&](const auto& sc) {
                return (has_standard_stress_plot_output(model, sc.id) ||
                        has_corner_stress_plot_output(model, sc.id) ||
                        has_gpstress_plot_output(model, sc.id)) &&
                       has_plate_stress_type(sc, ElementType::CTRIA3);
            });
        if (any) {
            const char name8[8] = {'O','E','S','1','X','1',' ',' '};
            write_table_header(f, name8, month, day, dyear);

            int itable = -3;
            bool new_result = true;
            for (const auto& sc : results.subcases) {
                if (!has_standard_stress_plot_output(model, sc.id) &&
                    !has_corner_stress_plot_output(model, sc.id) &&
                    !has_gpstress_plot_output(model, sc.id))
                    continue;
                const int prev_itable = itable;
                itable = write_oes1x_plate(f, sc, new_result, itable,
                                           ElementType::CTRIA3, sc.label);
                if (itable == prev_itable) continue;
                if (!new_result || itable < -3) new_result = false;
                const int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
                f.write(reinterpret_cast<const char*>(footer), 36);
            }
            write_markers(f, 0);
        }
    }

    // ── OES1X1 tables for solid corner stresses ──────────────────────────────
    for (ElementType etype : {ElementType::CTETRA4, ElementType::CTETRA10,
                              ElementType::CHEXA8, ElementType::CPENTA6}) {
        const bool any = std::any_of(
            results.subcases.begin(), results.subcases.end(),
            [&](const auto& sc) {
                const bool request_corner_like =
                    has_standard_stress_plot_output(model, sc.id) ||
                    has_corner_stress_plot_output(model, sc.id) ||
                    (has_gpstress_plot_output(model, sc.id) &&
                     !solid_has_extra_grid_points(etype));
                return request_corner_like && has_solid_stress_type(sc, etype);
            });
        if (!any) continue;

        const char name8[8] = {'O','E','S','1','X','1',' ',' '};
        write_table_header(f, name8, month, day, dyear);

        int itable = -3;
        bool new_result = true;
        for (const auto& sc : results.subcases) {
            const bool request_corner_like =
                has_standard_stress_plot_output(model, sc.id) ||
                has_corner_stress_plot_output(model, sc.id) ||
                (has_gpstress_plot_output(model, sc.id) &&
                 !solid_has_extra_grid_points(etype));
            if (!request_corner_like) continue;
            const int prev_itable = itable;
            itable = write_oes1x_solid(f, sc, new_result, itable,
                                       etype, SolidNodeOutput::Corner, sc.label);
            if (itable == prev_itable) continue;
            if (!new_result || itable < -3) new_result = false;
            const int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
            f.write(reinterpret_cast<const char*>(footer), 36);
        }
        write_markers(f, 0);
    }

    // ── OES1X1 tables for solid grid-point stresses ──────────────────────────
    for (ElementType etype : {ElementType::CTETRA10}) {
        const bool any = std::any_of(
            results.subcases.begin(), results.subcases.end(),
            [&](const auto& sc) {
                return has_gpstress_plot_output(model, sc.id) &&
                       has_solid_stress_type(sc, etype);
            });
        if (!any) continue;

        const char name8[8] = {'O','E','S','1','X','1',' ',' '};
        write_table_header(f, name8, month, day, dyear);

        int itable = -3;
        bool new_result = true;
        for (const auto& sc : results.subcases) {
            if (!has_gpstress_plot_output(model, sc.id)) continue;
            const int prev_itable = itable;
            itable = write_oes1x_solid(f, sc, new_result, itable,
                                       etype, SolidNodeOutput::AllGridPoints,
                                       sc.label);
            if (itable == prev_itable) continue;
            if (!new_result || itable < -3) new_result = false;
            const int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
            f.write(reinterpret_cast<const char*>(footer), 36);
        }
        write_markers(f, 0);
    }

    // File end marker
    write_markers(f, 0);
}

} // namespace vibestran
