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
void matrix_json(std::ostream &out, const Eigen::MatrixXd &m) {
  out << '[';
  for (Eigen::Index i = 0; i < m.rows(); ++i) {
    if (i)
      out << ',';
    out << '[';
    for (Eigen::Index j = 0; j < m.cols(); ++j) {
      if (j)
        out << ',';
      number_json(out, m(i, j));
    }
    out << ']';
  }
  out << ']';
}
// Writes the selected cells (every cell when `cells` is empty) with their
// own compacted points.
Status vtu(const std::filesystem::path &path, const Mesh &mesh,
           const std::vector<Vector> &field, const std::string &name,
           const std::vector<Vector> &second = {},
           const Eigen::VectorXd &potential = {},
           std::vector<int> cells = {}) {
  if (cells.empty()) {
    cells.resize(mesh.cells.size());
    std::iota(cells.begin(), cells.end(), 0);
  }
  std::vector<int> point(mesh.nodes.size(), -1), points;
  for (int c : cells)
    for (int n : mesh.cells[c].nodes)
      if (point[n] < 0) {
        point[n] = static_cast<int>(points.size());
        points.push_back(n);
      }
  std::ofstream out(path);
  if (!out)
    return Error{"Cannot write " + path.string()};
  out << std::setprecision(17);
  out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" "
         "version=\"0.1\" byte_order=\"LittleEndian\"><UnstructuredGrid><Piece "
         "NumberOfPoints=\""
      << points.size() << "\" NumberOfCells=\"" << cells.size()
      << "\">\n<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" "
         "format=\"ascii\">\n";
  for (int n : points)
    out << mesh.nodes[n].transpose() << '\n';
  out << "</DataArray></Points><Cells><DataArray type=\"Int64\" "
         "Name=\"connectivity\" format=\"ascii\">\n";
  for (int c : cells) {
    for (int n : mesh.cells[c].nodes)
      out << point[n] << ' ';
    out << '\n';
  }
  out << "</DataArray><DataArray type=\"Int64\" Name=\"offsets\" "
         "format=\"ascii\">\n";
  size_t offset = 0;
  for (int c : cells) {
    offset += mesh.cells[c].nodes.size();
    out << offset << ' ';
  }
  out << "</DataArray><DataArray type=\"UInt8\" Name=\"types\" "
         "format=\"ascii\">\n";
  for (int c : cells) {
    const size_t nodes = mesh.cells[c].nodes.size();
    out << (mesh.planar ? (nodes == 3 ? 5 : 9)
                        : (nodes == 4   ? 10
                           : nodes == 6 ? 13
                                        : 12))
        << ' ';
  }
  out << "</DataArray></Cells><CellData><DataArray type=\"Int64\" "
         "Name=\"element_id\" format=\"ascii\">\n";
  for (int c : cells)
    out << mesh.cells[c].id << ' ';
  out << "</DataArray><DataArray type=\"Float64\" Name=\"" << name
      << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int c : cells)
    out << field[c].transpose() << '\n';
  out << "</DataArray>\n";
  if (!second.empty()) {
    out << "<DataArray type=\"Float64\" Name=\"H_A_per_m\" "
           "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int c : cells)
      out << second[c].transpose() << '\n';
    out << "</DataArray>\n";
  }
  out << "</CellData>";
  if (mesh.planar &&
      potential.size() == static_cast<Eigen::Index>(mesh.nodes.size())) {
    out << "<PointData><DataArray type=\"Float64\" Name=\"A_normal_Wb_per_m\" "
           "format=\"ascii\">\n";
    for (int n : points)
      out << potential[n] << ' ';
    out << "</DataArray></PointData>";
  }
  out << "</Piece></UnstructuredGrid></VTKFile>\n";
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
  f06 << "VIBESTRAN SOL MAGSTAT — SI UNITS\nFirst-order magnetic elements; CPU "
         "sparse Cholesky\n";
  f06 << "Magnetic volume cells: " << results.mesh.cells.size()
      << "  degrees of freedom: " << results.mesh.dofs() << "\n";
  json << "{\"schema\":1,\"units\":\"SI\",\"dimension\":"
       << (results.mesh.planar ? 2 : 3) << ",\"depth_m\":" << results.mesh.depth
       << ",\"sources\":[";
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
        << "  transfer error " << source.transfer_error
        << "  preprocessing cells " << source.mesh.cells.size() << "  faces "
        << source.mesh.faces.size() << '\n';
    std::vector<Vector> density;
    if (source.mesh.planar)
      density = source.planar_density;
    else
      for (const auto &c : source.mesh.cells) {
        Vector center = std::accumulate(
            c.nodes.begin(), c.nodes.end(), Vector(Vector::Zero()),
            [&](Vector sum, int n) -> Vector {
              return sum +
                     source.mesh.nodes[n] / static_cast<double>(c.nodes.size());
            });
        const auto basis = flux_basis(source.mesh, c, center);
        Vector j = Vector::Zero();
        for (size_t k = 0; k < c.faces.size(); ++k)
          j += basis[k] * source.flux[c.faces[k]];
        density.push_back(j);
      }
    if (model.options.source_output)
      if (auto e = vtu(prefix + (source.wire ? "_wire" : "_coil") +
                           std::to_string(source.id) + ".vtu",
                       source.mesh, density, "J_per_ampere_per_m2"))
        return e;
  }
  json << "],\"circuits\":[";
  for (size_t i = 0; i < results.circuits.size(); ++i) {
    const auto &c = results.circuits[i];
    if (i)
      json << ',';
    json << "{\"id\":" << c.id << ",\"parallel\":" << c.parallel
         << ",\"closed\":" << (c.closed ? "true" : "false")
         << ",\"members\":[";
    f06 << "CIRCUIT " << c.id << "  parallel paths " << c.parallel
        << "  members";
    for (size_t j = 0; j < c.members.size(); ++j) {
      const auto [source, polarity] = c.members[j];
      if (j)
        json << ',';
      json << '[' << results.sources[source].id << ',' << polarity << ']';
      f06 << ' ' << (polarity > 0 ? '+' : '-') << results.sources[source].id;
    }
    json << "]}";
    f06 << '\n';
  }
  json << "],\"cases\":[";
  csv << "subcase,probe,x_m,y_m,z_m,Bx_T,By_T,Bz_T,Hx_A_per_m,Hy_A_per_m,Hz_A_"
         "per_m,cd,B1_T,B2_T,B3_T,H1_A_per_m,H2_A_per_m,H3_A_per_m\n";
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
        f06 << " linkage unavailable for open path or single planar region";
      f06 << '\n';
    }
    json << "],\"differential_inductance_H\":";
    matrix_json(json, c.differential_inductance);
    json << ",\"circuits\":[";
    for (size_t i = 0; i < c.circuits.size(); ++i) {
      const auto &circuit = c.circuits[i];
      if (i)
        json << ',';
      json << "{\"id\":" << circuit.id << ",\"current_A\":" << circuit.current
           << ",\"linkage_Wb_turn\":";
      number_json(json, circuit.linkage);
      json << '}';
      f06 << "CIRCUIT " << circuit.id << " I(A)=" << circuit.current;
      if (std::isfinite(circuit.linkage))
        f06 << " linkage(Wb-turn)=" << circuit.linkage
            << " differential inductance(H)=" << c.circuit_inductance(i, i);
      else
        f06 << " linkage unavailable (open or unbalanced circuit)";
      f06 << '\n';
    }
    json << "],\"circuit_inductance_H\":";
    matrix_json(json, c.circuit_inductance);
    json << ",\"forces\":[";
    for (size_t i = 0; i < c.forces.size(); ++i) {
      const auto &f = c.forces[i];
      const auto &request = model.forces[i];
      // Torque is about the CID origin; the local components use its axes.
      const Matrix axes =
          request.coordinates
              ? coordinate_axes(model.coordinates.at(request.coordinates))
              : Matrix::Identity();
      if (i)
        json << ',';
      json << "{\"id\":" << f.id << ",\"cid\":" << request.coordinates
           << ",\"force_N\":";
      vector_json(json, f.force);
      json << ",\"torque_Nm\":";
      vector_json(json, f.torque);
      json << ",\"force_local_N\":";
      vector_json(json, axes.transpose() * f.force);
      json << ",\"torque_local_Nm\":";
      vector_json(json, axes.transpose() * f.torque);
      json << ",\"components\":[";
      for (size_t k = 0; k < f.components.size(); ++k) {
        if (k)
          json << ',';
        json << "{\"force_N\":";
        vector_json(json, f.components[k].force);
        json << ",\"torque_Nm\":";
        vector_json(json, f.components[k].torque);
        json << '}';
      }
      json << "]}";
      f06 << (request.body ? "BODY " : "SURFACE ") << f.id
          << " force(N): " << f.force.transpose()
          << " torque(Nm): " << f.torque.transpose();
      if (request.coordinates)
        f06 << "  in CID " << request.coordinates
            << " force(N): " << (axes.transpose() * f.force).transpose()
            << " torque(Nm): " << (axes.transpose() * f.torque).transpose();
      f06 << '\n';
      for (size_t k = 0; k < f.components.size() && f.components.size() > 1;
           ++k)
        f06 << "  piece " << k + 1
            << " force(N): " << f.components[k].force.transpose()
            << " torque(Nm): " << f.components[k].torque.transpose() << '\n';
    }
    json << "],\"probes\":[";
    for (size_t i = 0; i < model.probes.size(); ++i) {
      const auto &probe = model.probes[i];
      // CD components use the local directions at the probe, e.g. radial
      // and tangential for a CORD2C.
      const Matrix axes =
          probe.output_coordinates
              ? local_axes(model.coordinates.at(probe.output_coordinates),
                           probe.position)
              : Matrix::Identity();
      const Vector b = axes.transpose() * c.probe_b[i],
                   h = axes.transpose() * c.probe_h[i];
      if (i)
        json << ',';
      json << "{\"id\":" << probe.id << ",\"cd\":" << probe.output_coordinates
           << ",\"B_T\":";
      vector_json(json, c.probe_b[i]);
      json << ",\"H_A_per_m\":";
      vector_json(json, c.probe_h[i]);
      json << ",\"B_local_T\":";
      vector_json(json, b);
      json << ",\"H_local_A_per_m\":";
      vector_json(json, h);
      json << '}';
      if (!model.subcases[ci].probes)
        continue;
      csv << c.id << ',' << probe.id;
      for (const Vector *v : {&probe.position, &c.probe_b[i], &c.probe_h[i]})
        for (int k = 0; k < 3; ++k)
          csv << ',' << (*v)[k];
      csv << ',' << probe.output_coordinates;
      for (const Vector *v : {&b, &h})
        for (int k = 0; k < 3; ++k)
          csv << ',' << (*v)[k];
      csv << '\n';
    }
    json << "]}";
    const auto &request = model.subcases[ci].field;
    std::vector<int> cells;
    if (request.set)
      for (int id : model.element_sets.at(request.set))
        cells.push_back(results.mesh.element_index.at(id));
    if (request.enabled)
      if (auto e = vtu(prefix + "_sc" + std::to_string(c.id) + ".vtu",
                       results.mesh, c.b, "B_T", c.h, c.potential,
                       std::move(cells)))
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
