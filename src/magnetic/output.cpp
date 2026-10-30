#include "magnetic/solver.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace vibestran::magnetic {
namespace {
void vector_json(std::ostream &out, const Vector &v) {
  out << '[' << v.x() << ',' << v.y() << ',' << v.z() << ']';
}
void number_json(std::ostream &out, double value) {
  if (std::isfinite(value))
    out << value;
  else
    out << "null";
}
Status vtu(const std::filesystem::path &path, const Mesh &mesh,
           const std::vector<Vector> &field, const std::string &name,
           const std::vector<Vector> &second = {}) {
  std::ofstream out(path);
  if (!out)
    return Error{"Cannot write " + path.string()};
  out << std::setprecision(17);
  out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" "
         "version=\"0.1\" byte_order=\"LittleEndian\"><UnstructuredGrid><Piece "
         "NumberOfPoints=\""
      << mesh.nodes.size() << "\" NumberOfCells=\"" << mesh.cells.size()
      << "\">\n<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" "
         "format=\"ascii\">\n";
  for (const auto &x : mesh.nodes)
    out << x.transpose() << '\n';
  out << "</DataArray></Points><Cells><DataArray type=\"Int64\" "
         "Name=\"connectivity\" format=\"ascii\">\n";
  for (const auto &c : mesh.cells)
    out << c.nodes[0] << ' ' << c.nodes[1] << ' ' << c.nodes[2] << ' '
        << c.nodes[3] << '\n';
  out << "</DataArray><DataArray type=\"Int64\" Name=\"offsets\" "
         "format=\"ascii\">\n";
  for (size_t i = 1; i <= mesh.cells.size(); ++i)
    out << 4 * i << ' ';
  out << "</DataArray><DataArray type=\"UInt8\" Name=\"types\" "
         "format=\"ascii\">\n";
  for (size_t i = 0; i < mesh.cells.size(); ++i)
    out << "10 ";
  out << "</DataArray></Cells><CellData><DataArray type=\"Int64\" "
         "Name=\"element_id\" format=\"ascii\">\n";
  for (const auto &c : mesh.cells)
    out << c.id << ' ';
  out << "</DataArray><DataArray type=\"Float64\" Name=\"" << name
      << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (const auto &x : field)
    out << x.transpose() << '\n';
  out << "</DataArray>\n";
  if (!second.empty()) {
    out << "<DataArray type=\"Float64\" Name=\"H_A_per_m\" "
           "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto &x : second)
      out << x.transpose() << '\n';
    out << "</DataArray>\n";
  }
  out << "</CellData></Piece></UnstructuredGrid></VTKFile>\n";
  out.close();
  if (!out)
    return Error{"Failed writing " + path.string()};
  return {};
}
} // namespace
Status write_results(const Model &model, const Results &results,
                     const std::filesystem::path &report) {
  auto base = report;
  base.replace_extension();
  const auto prefix = base.string();
  std::ofstream f06(report), json(prefix + ".mag.json"),
      csv(prefix + ".probes.csv");
  if (!f06 || !json || !csv)
    return Error{"Cannot open magnetic result files at " + prefix};
  f06 << std::setprecision(12);
  json << std::setprecision(17);
  csv << std::setprecision(17);
  f06 << "VIBESTRAN SOL MAGSTAT — SI UNITS\n3D tetrahedral edge elements; CPU "
         "sparse Cholesky\n";
  f06 << "Magnetic volume cells: " << results.mesh.cells.size()
      << "  edges: " << results.mesh.edges.size() << "\n";
  json << "{\"schema\":1,\"units\":\"SI\",\"sources\":[";
  for (size_t i = 0; i < results.sources.size(); ++i) {
    const auto &source = results.sources[i];
    if (i)
      json << ',';
    json << "{\"id\":" << source.id
         << ",\"wire\":" << (source.wire ? "true" : "false")
         << ",\"closed\":" << (source.closed ? "true" : "false")
         << ",\"conservation_error\":" << source.conservation_error
         << ",\"transfer_error\":" << source.transfer_error
         << ",\"wire_elements\":[";
    for (size_t j = 0; j < source.wire_elements.size(); ++j) {
      if (j)
        json << ',';
      json << '[' << source.wire_elements[j].first << ','
           << source.wire_elements[j].second << ']';
    }
    json << "]}";
    f06 << (source.wire ? "WIRE PATH " : "COIL ") << source.id
        << "  conservation error " << source.conservation_error
        << "  transfer error " << source.transfer_error << '\n';
    std::vector<Vector> density;
    for (const auto &c : source.mesh.cells) {
      Vector center = std::accumulate(c.nodes.begin(), c.nodes.end(),
                                      Vector(Vector::Zero()),
                                      [&](Vector sum, int n) -> Vector {
                                        return sum + source.mesh.nodes[n] / 4;
                                      });
      const auto basis = flux_basis(source.mesh, c, center);
      Vector j = Vector::Zero();
      for (int k = 0; k < 4; ++k)
        j += basis[k] * source.flux[c.faces[k]];
      density.push_back(j);
    }
    if (auto e = vtu(prefix + (source.wire ? "_wire" : "_coil") +
                         std::to_string(source.id) + ".vtu",
                     source.mesh, density, "J_per_ampere_per_m2"))
      return e;
  }
  json << "],\"cases\":[";
  csv << "subcase,probe,x_m,y_m,z_m,Bx_T,By_T,Bz_T,Hx_A_per_m,Hy_A_per_m,Hz_A_"
         "per_m\n";
  for (size_t ci = 0; ci < results.cases.size(); ++ci) {
    const auto &c = results.cases[ci];
    if (ci)
      json << ',';
    json << "{\"id\":" << c.id << ",\"iterations\":" << c.iterations
         << ",\"residual\":" << c.residual << ",\"energy_J\":" << c.energy
         << ",\"coenergy_J\":" << c.coenergy
         << ",\"bh_extrapolated\":" << (c.extrapolated ? "true" : "false")
         << ",\"sources\":[";
    f06 << "\nSUBCASE " << c.id << "\nNewton iterations: " << c.iterations
        << "  relative residual: " << c.residual << "\nEnergy (J): " << c.energy
        << "  source Legendre transform (J): " << c.coenergy << '\n';
    if (c.extrapolated)
      f06 << "BH curve exceeded: vacuum-slope extrapolation used.\n";
    for (size_t i = 0; i < c.sources.size(); ++i) {
      const auto &s = c.sources[i];
      if (i)
        json << ',';
      json << "{\"id\":" << s.id << ",\"current_A\":" << s.current
           << ",\"linkage_Wb_turn\":";
      number_json(json, s.linkage);
      json << '}';
      f06 << (s.wire ? "WIRE PATH " : "COIL ") << s.id << " I(A)=" << s.current;
      if (s.closed)
        f06 << " linkage(Wb-turn)=" << s.linkage;
      else
        f06 << " linkage unavailable for open path";
      f06 << '\n';
    }
    json << "],\"differential_inductance_H\":[";
    for (Eigen::Index i = 0; i < c.differential_inductance.rows(); ++i) {
      if (i)
        json << ',';
      json << '[';
      for (Eigen::Index j = 0; j < c.differential_inductance.cols(); ++j) {
        if (j)
          json << ',';
        number_json(json, c.differential_inductance(i, j));
      }
      json << ']';
    }
    json << "],\"forces\":[";
    for (size_t i = 0; i < c.forces.size(); ++i) {
      const auto &f = c.forces[i];
      if (i)
        json << ',';
      json << "{\"id\":" << f.id << ",\"force_N\":";
      vector_json(json, f.force);
      json << ",\"torque_Nm\":";
      vector_json(json, f.torque);
      json << '}';
      f06 << "SURFACE " << f.id << " force(N): " << f.force.transpose()
          << " torque(Nm): " << f.torque.transpose() << '\n';
    }
    json << "],\"probes\":[";
    for (size_t i = 0; i < model.probes.size(); ++i) {
      if (i)
        json << ',';
      json << "{\"id\":" << model.probes[i].id << ",\"B_T\":";
      vector_json(json, c.probe_b[i]);
      json << ",\"H_A_per_m\":";
      vector_json(json, c.probe_h[i]);
      json << '}';
      csv << c.id << ',' << model.probes[i].id;
      for (const Vector *v :
           {&model.probes[i].position, &c.probe_b[i], &c.probe_h[i]})
        for (int k = 0; k < 3; ++k)
          csv << ',' << (*v)[k];
      csv << '\n';
    }
    json << "]}";
    if (auto e = vtu(prefix + "_sc" + std::to_string(c.id) + ".vtu",
                     results.mesh, c.b, "B_T", c.h))
      return e;
  }
  json << "]}\n";
  f06.close();
  json.close();
  csv.close();
  if (!f06 || !json || !csv)
    return Error{"Failed writing magnetic results at " + prefix};
  return {};
}
} // namespace vibestran::magnetic
