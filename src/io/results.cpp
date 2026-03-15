// src/io/results.cpp
#include "io/results.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <chrono>
#include <format>

namespace nastran {

void F06Writer::write(const SolverResults& results, const Model& model,
                      const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) throw SolverError(std::format("Cannot write F06: {}", path.string()));
    write(results, model, f);
}

void F06Writer::write(const SolverResults& results, const Model& model, std::ostream& out) {
    write_header(out);
    for (const auto& sc : results.subcases) {
        out << "\n1                                             SUBCASE " << sc.id << "\n";
        if (!sc.label.empty())
            out << "                                              " << sc.label << "\n";
        write_displacement_table(sc, out);
        write_plate_stress_table(sc, out);
        write_solid_stress_table(sc, out);
    }
    out << "\n\n                     * * * END OF JOB * * *\n\n";
}

void F06Writer::write_header(std::ostream& out) {
    out << "1                           N A S T R A N - C O M P A T I B L E   S O L V E R\n";
    out << "0                                                                         \n";
    out << "         S O L   1 0 1   L I N E A R   S T A T I C   A N A L Y S I S\n";
    out << "\n";
}

void F06Writer::write_displacement_table(const SubCaseResults& sc, std::ostream& out) {
    if (sc.displacements.empty()) return;

    out << "\n                                             D I S P L A C E M E N T   V E C T O R\n\n";
    out << "      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3\n";

    for (const auto& nd : sc.displacements) {
        out << std::setw(12) << nd.node.value << "        G   ";
        for (int i = 0; i < 6; ++i)
            out << std::setw(15) << std::setprecision(6) << std::scientific << nd.d[i];
        out << "\n";
    }
}

void F06Writer::write_plate_stress_table(const SubCaseResults& sc, std::ostream& out) {
    if (sc.plate_stresses.empty()) return;

    out << "\n                                  S T R E S S E S   I N   Q U A D R I L A T E R A L   E L E M E N T S\n\n";
    out << "      ELEMENT ID         SX             SY             SXY            MX             MY             MXY          VON MISES\n";

    for (const auto& ps : sc.plate_stresses) {
        out << std::setw(14) << ps.eid.value;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.sx;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.sy;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.sxy;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.mx;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.my;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.mxy;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.von_mises;
        out << "\n";
    }
}

void F06Writer::write_solid_stress_table(const SubCaseResults& sc, std::ostream& out) {
    if (sc.solid_stresses.empty()) return;

    out << "\n                                       S T R E S S E S   I N   S O L I D   E L E M E N T S\n\n";
    out << "      ELEMENT ID         SX             SY             SZ             SXY            SYZ            SZX        VON MISES\n";

    for (const auto& ss : sc.solid_stresses) {
        out << std::setw(14) << ss.eid.value;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.sx;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.sy;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.sz;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.sxy;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.syz;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.szx;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.von_mises;
        out << "\n";
    }
}

} // namespace nastran
