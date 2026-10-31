#include "magnetic/solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <utility>
#ifdef VIBESTRAN_HAS_HDF5
#include <hdf5.h>
#endif

namespace vibestran::magnetic {
namespace {
void write_f06(std::ostream &f06, const Model &model, const Results &results,
               const F06RunInfo *run) {
  write_f06_header(f06,
                   "S O L   M A G S T A T   M A G N E T O S T A T I C   A N A "
                   "L Y S I S",
                   results.summary, run);
  f06 << std::setprecision(12);
  for (const auto &source : results.sources)
    f06 << (source.wire ? "WIRE PATH " : "COIL ") << source.id
        << "  conservation error " << source.conservation_error
        << "  transfer error " << source.transfer_error
        << "  preprocessing cells " << source.mesh.cells.size() << "  faces "
        << source.mesh.faces.size() << '\n';
  for (const auto &c : results.circuits) {
    f06 << "CIRCUIT " << c.id << "  parallel paths " << c.parallel
        << "  members";
    for (const auto [source, polarity] : c.members)
      f06 << ' ' << (polarity > 0 ? '+' : '-') << results.sources[source].id;
    f06 << '\n';
  }
  for (const auto &c : results.cases) {
    f06 << "\nSUBCASE " << c.id << "\nNewton iterations: " << c.iterations
        << "  relative residual: " << c.residual << "\nEnergy (J): " << c.energy
        << "  source Legendre transform (J): " << c.coenergy << '\n';
    if (c.extrapolated)
      f06 << "BH curve exceeded: vacuum-slope extrapolation used.\n";
    for (const auto &s : c.sources) {
      f06 << (s.wire ? "WIRE PATH " : "COIL ") << s.id << " I(A)=" << s.current;
      if (s.closed)
        f06 << " linkage(Wb-turn)=" << s.linkage;
      else
        f06 << " linkage unavailable for open path or single planar region";
      f06 << '\n';
    }
    for (size_t i = 0; i < c.circuits.size(); ++i) {
      const auto &circuit = c.circuits[i];
      f06 << "CIRCUIT " << circuit.id << " I(A)=" << circuit.current;
      if (std::isfinite(circuit.linkage))
        f06 << " linkage(Wb-turn)=" << circuit.linkage
            << " differential inductance(H)=" << c.circuit_inductance(i, i);
      else
        f06 << " linkage unavailable (open or unbalanced circuit)";
      f06 << '\n';
    }
    for (size_t i = 0; i < c.forces.size(); ++i) {
      const auto &f = c.forces[i];
      const auto &request = model.forces[i];
      // Torque is about the CID origin; the local components use its axes.
      const Matrix axes =
          request.coordinates
              ? coordinate_axes(model.coordinates.at(request.coordinates))
              : Matrix::Identity();
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
  }
}

#ifdef VIBESTRAN_HAS_HDF5
// Current density per ampere at each source cell: the planar region density,
// or the RT0 flux evaluated at the cell centroid.
std::vector<Vector> source_density(const Source &source) {
  if (source.mesh.planar)
    return source.planar_density;
  std::vector<Vector> density;
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
  return density;
}

// Owns one HDF5 identifier.
class Handle {
public:
  Handle() = default;
  Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
  Handle(Handle &&other) noexcept
      : id_(std::exchange(other.id_, H5I_INVALID_HID)), close_(other.close_) {}
  Handle &operator=(Handle &&other) noexcept {
    if (this != &other) {
      reset();
      id_ = std::exchange(other.id_, H5I_INVALID_HID);
      close_ = other.close_;
    }
    return *this;
  }
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  ~Handle() { reset(); }
  [[nodiscard]] hid_t get() const { return id_; }
  [[nodiscard]] bool valid() const { return id_ >= 0; }
  // Closes now; false when HDF5 reports a failure (e.g. on the final flush).
  bool reset() {
    bool ok = true;
    if (id_ >= 0 && close_)
      ok = close_(id_) >= 0;
    id_ = H5I_INVALID_HID;
    return ok;
  }

private:
  hid_t id_{H5I_INVALID_HID};
  herr_t (*close_)(hid_t){nullptr};
};

template <class T> struct Hdf5Type;
template <> struct Hdf5Type<double> {
  static hid_t memory() { return H5T_NATIVE_DOUBLE; }
  static hid_t file() { return H5T_IEEE_F64LE; }
};
template <> struct Hdf5Type<int64_t> {
  static hid_t memory() { return H5T_NATIVE_INT64; }
  static hid_t file() { return H5T_STD_I64LE; }
};
template <> struct Hdf5Type<uint8_t> {
  static hid_t memory() { return H5T_NATIVE_UINT8; }
  static hid_t file() { return H5T_STD_U8LE; }
};

// Writes one HDF5 file. The first failure is recorded with the object path
// and HDF5's own error description, and later calls do nothing, so the
// writing code reads straight through and finish() reports that one error.
// Groups track link creation order: the VTKHDF reader iterates by it, and it
// keeps subcases in deck order.
class Hdf5File {
public:
  explicit Hdf5File(std::string path) : path_(std::move(path)) {
    // Failures are returned as values; don't let HDF5 print its stack too.
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    Handle plist(H5Pcreate(H5P_FILE_CREATE), H5Pclose);
    H5Pset_link_creation_order(plist.get(),
                               H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
    file_ = Handle(
        H5Fcreate(path_.c_str(), H5F_ACC_TRUNC, plist.get(), H5P_DEFAULT),
        H5Fclose);
    if (!file_.valid())
      fail("cannot create the file", "/");
  }
  [[nodiscard]] hid_t root() const { return file_.get(); }

  Handle group(hid_t parent, const std::string &name) {
    if (error_)
      return {};
    Handle plist(H5Pcreate(H5P_GROUP_CREATE), H5Pclose);
    H5Pset_link_creation_order(plist.get(),
                               H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
    Handle g(H5Gcreate2(parent, name.c_str(), H5P_DEFAULT, plist.get(),
                        H5P_DEFAULT),
             H5Gclose);
    if (!g.valid())
      fail("cannot create group", where(parent, name));
    return g;
  }
  // A fixed-length ASCII string, the form the VTKHDF reader expects.
  void attribute(hid_t object, const std::string &name,
                 const std::string &text) {
    if (error_)
      return;
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(type.get(), std::max<size_t>(text.size(), 1));
    H5Tset_strpad(type.get(), H5T_STR_NULLTERM);
    H5Tset_cset(type.get(), H5T_CSET_ASCII);
    write_attribute(object, name, type.get(), type.get(), {}, text.data());
  }
  template <class T>
  void attribute(hid_t object, const std::string &name, T value) {
    if (!error_)
      write_attribute(object, name, Hdf5Type<T>::file(),
                      Hdf5Type<T>::memory(), {}, &value);
  }
  void attribute(hid_t object, const std::string &name,
                 const std::vector<int64_t> &values) {
    if (!error_)
      write_attribute(object, name, Hdf5Type<int64_t>::file(),
                      Hdf5Type<int64_t>::memory(), {values.size()},
                      values.data());
  }
  template <class T>
  void values(hid_t parent, const std::string &name,
              const std::vector<T> &data) {
    dataset(parent, name, data.data(), {data.size()});
  }
  // Row i holds the vector of item i.
  void vectors(hid_t parent, const std::string &name,
               const std::vector<Vector> &data) {
    std::vector<double> flat;
    flat.reserve(3 * data.size());
    for (const auto &v : data)
      flat.insert(flat.end(), {v.x(), v.y(), v.z()});
    dataset(parent, name, flat.data(), {data.size(), 3});
  }
  // Row-major, so row i is item i; NaN marks unavailable entries.
  void matrix(hid_t parent, const std::string &name, const Eigen::MatrixXd &m) {
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
        rows = m;
    dataset(parent, name, rows.data(),
            {static_cast<hsize_t>(m.rows()), static_cast<hsize_t>(m.cols())});
  }
  // Item i is the pair in row i.
  void pairs(hid_t parent, const std::string &name,
             const std::vector<std::pair<int64_t, int64_t>> &data) {
    std::vector<int64_t> flat;
    flat.reserve(2 * data.size());
    for (const auto &[a, b] : data)
      flat.insert(flat.end(), {a, b});
    dataset(parent, name, flat.data(), {data.size(), 2});
  }
  void link(hid_t parent, const std::string &name, const std::string &target) {
    if (!error_ && H5Lcreate_soft(target.c_str(), parent, name.c_str(),
                                  H5P_DEFAULT, H5P_DEFAULT) < 0)
      fail("cannot link to " + target, where(parent, name));
  }
  [[nodiscard]] Status finish() {
    if (!file_.reset() && !error_)
      fail("cannot flush and close the file", "/");
    return error_;
  }

private:
  template <class T>
  void dataset(hid_t parent, const std::string &name, const T *data,
               std::vector<hsize_t> dims) {
    if (error_)
      return;
    Handle space(H5Screate_simple(static_cast<int>(dims.size()), dims.data(),
                                  nullptr),
                 H5Sclose);
    Handle set(H5Dcreate2(parent, name.c_str(), Hdf5Type<T>::file(),
                          space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
               H5Dclose);
    if (!set.valid())
      return fail("cannot create dataset", where(parent, name));
    const bool empty = std::find(dims.begin(), dims.end(), hsize_t{0}) !=
                       dims.end();
    if (!empty && H5Dwrite(set.get(), Hdf5Type<T>::memory(), H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, data) < 0)
      fail("cannot write dataset", where(parent, name));
  }
  void write_attribute(hid_t object, const std::string &name, hid_t file_type,
                       hid_t memory_type, std::vector<hsize_t> dims,
                       const void *data) {
    Handle space(dims.empty() ? H5Screate(H5S_SCALAR)
                              : H5Screate_simple(static_cast<int>(dims.size()),
                                                 dims.data(), nullptr),
                 H5Sclose);
    Handle attribute(H5Acreate2(object, name.c_str(), file_type, space.get(),
                                H5P_DEFAULT, H5P_DEFAULT),
                     H5Aclose);
    if (!attribute.valid() ||
        H5Awrite(attribute.get(), memory_type, data) < 0)
      fail("cannot write attribute " + name, where(object, ""));
  }
  static std::string where(hid_t parent, const std::string &name) {
    const auto length = H5Iget_name(parent, nullptr, 0);
    std::string path(length > 0 ? static_cast<size_t>(length) : 0, '\0');
    if (length > 0)
      H5Iget_name(parent, path.data(), path.size() + 1);
    if (name.empty())
      return path.empty() ? "/" : path;
    return (path.empty() || path == "/" ? "/" : path + "/") + name;
  }
  void fail(const std::string &what, const std::string &object) {
    std::string detail;
    H5Ewalk2(
        H5E_DEFAULT, H5E_WALK_DOWNWARD,
        [](unsigned, const H5E_error2_t *entry, void *text) -> herr_t {
          auto &out = *static_cast<std::string *>(text);
          if (entry->desc && *entry->desc)
            out += std::string(out.empty() ? "" : "; ") + entry->desc;
          return 0;
        },
        &detail);
    H5Eclear2(H5E_DEFAULT);
    error_ = Error{"HDF5 output " + path_ + ": " + what + " at " + object +
                   (detail.empty() ? "" : " (" + detail + ")")};
  }

  std::string path_;
  Handle file_;
  Status error_;
};

std::vector<int64_t> ids(const auto &items) {
  std::vector<int64_t> out;
  for (const auto &item : items)
    out.push_back(item.id);
  return out;
}

// One UnstructuredGrid block of the VTKHDF collection: the selected cells
// with their own compacted points, element IDs and per-cell vector fields.
// The planar potential, when given, is written as point data.
void write_grid(Hdf5File &h5, hid_t vtkhdf, const std::string &name,
                int64_t index, const Mesh &mesh, std::vector<int> cells,
                const std::vector<std::pair<std::string,
                                            const std::vector<Vector> *>>
                    &fields,
                const Eigen::VectorXd *potential = nullptr) {
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
  std::vector<Vector> coordinates;
  for (int n : points)
    coordinates.push_back(mesh.nodes[n]);
  std::vector<int64_t> connectivity, offsets{0}, element_ids;
  std::vector<uint8_t> types;
  for (int c : cells) {
    const auto &cell = mesh.cells[c];
    for (int n : cell.nodes)
      connectivity.push_back(point[n]);
    offsets.push_back(static_cast<int64_t>(connectivity.size()));
    const size_t nodes = cell.nodes.size();
    // VTK triangle, quad, tetra, wedge and hexahedron.
    types.push_back(mesh.planar ? (nodes == 3 ? 5 : 9)
                                : (nodes == 4   ? 10
                                   : nodes == 6 ? 13
                                                : 12));
    element_ids.push_back(cell.id);
  }
  auto block = h5.group(vtkhdf, name);
  h5.attribute(block.get(), "Version", std::vector<int64_t>{2, 5});
  h5.attribute(block.get(), "Type", std::string("UnstructuredGrid"));
  h5.attribute(block.get(), "Index", index);
  h5.values(block.get(), "NumberOfPoints",
            std::vector<int64_t>{static_cast<int64_t>(points.size())});
  h5.values(block.get(), "NumberOfCells",
            std::vector<int64_t>{static_cast<int64_t>(cells.size())});
  h5.values(block.get(), "NumberOfConnectivityIds",
            std::vector<int64_t>{static_cast<int64_t>(connectivity.size())});
  h5.vectors(block.get(), "Points", coordinates);
  h5.values(block.get(), "Connectivity", connectivity);
  h5.values(block.get(), "Offsets", offsets);
  h5.values(block.get(), "Types", types);
  auto cell_data = h5.group(block.get(), "CellData");
  h5.values(cell_data.get(), "element_id", element_ids);
  for (const auto &[field, values] : fields) {
    std::vector<Vector> selected;
    for (int c : cells)
      selected.push_back((*values)[c]);
    h5.vectors(cell_data.get(), field, selected);
  }
  if (potential && mesh.planar &&
      potential->size() == static_cast<Eigen::Index>(mesh.nodes.size())) {
    auto point_data = h5.group(block.get(), "PointData");
    std::vector<double> selected;
    for (int n : points)
      selected.push_back((*potential)[n]);
    h5.values(point_data.get(), "A_normal_Wb_per_m", selected);
  }
}

Status write_hdf5(const std::string &path, const Model &model,
                  const Results &results) {
  Hdf5File h5(path);
  // VTKHDF fields: a PartitionedDataSetCollection with a "fields" assembly
  // node (one block per subcase BFIELD request) and a "sources" node (one
  // block per coil or wire source mesh when MSOURCE is on).
  auto vtkhdf = h5.group(h5.root(), "VTKHDF");
  h5.attribute(vtkhdf.get(), "Version", std::vector<int64_t>{2, 5});
  h5.attribute(vtkhdf.get(), "Type",
               std::string("PartitionedDataSetCollection"));
  auto assembly = h5.group(vtkhdf.get(), "Assembly");
  int64_t index = 0;
  Handle fields_node, sources_node;
  for (size_t ci = 0; ci < results.cases.size(); ++ci) {
    const auto &c = results.cases[ci];
    const auto &request = model.subcases[ci].field;
    if (!request.enabled)
      continue;
    std::vector<int> cells;
    if (request.set)
      for (int id : model.element_sets.at(request.set))
        cells.push_back(results.mesh.element_index.at(id));
    const auto name = "SUBCASE_" + std::to_string(c.id);
    write_grid(h5, vtkhdf.get(), name, index++, results.mesh, std::move(cells),
               {{"B_T", &c.b}, {"H_A_per_m", &c.h}}, &c.potential);
    if (!fields_node.valid())
      fields_node = h5.group(assembly.get(), "fields");
    h5.link(fields_node.get(), name, "/VTKHDF/" + name);
  }
  if (model.options.source_output)
    for (const auto &source : results.sources) {
      const auto name =
          (source.wire ? "WIRE_" : "COIL_") + std::to_string(source.id);
      const auto density = source_density(source);
      write_grid(h5, vtkhdf.get(), name, index++, source.mesh, {},
                 {{"J_per_ampere_per_m2", &density}});
      if (!sources_node.valid())
        sources_node = h5.group(assembly.get(), "sources");
      h5.link(sources_node.get(), name, "/VTKHDF/" + name);
    }

  // Scalar results. Rows of each definition table line up with the rows of
  // the same-named group in every subcase.
  auto magstat = h5.group(h5.root(), "MAGSTAT");
  h5.attribute(magstat.get(), "schema", int64_t{1});
  h5.attribute(magstat.get(), "units", std::string("SI"));
  h5.attribute(magstat.get(), "dimension",
               int64_t{results.mesh.planar ? 2 : 3});
  h5.attribute(magstat.get(), "depth_m", results.mesh.depth);
  {
    auto sources = h5.group(magstat.get(), "sources");
    std::vector<uint8_t> wire, closed;
    std::vector<double> conservation, transfer;
    std::vector<int64_t> offsets{0};
    std::vector<std::pair<int64_t, int64_t>> elements;
    for (const auto &s : results.sources) {
      wire.push_back(s.wire);
      closed.push_back(s.closed);
      conservation.push_back(s.conservation_error);
      transfer.push_back(s.transfer_error);
      for (const auto &[element, direction] : s.wire_elements)
        elements.emplace_back(element, direction);
      offsets.push_back(static_cast<int64_t>(elements.size()));
    }
    h5.values(sources.get(), "id", ids(results.sources));
    h5.values(sources.get(), "wire", wire);
    h5.values(sources.get(), "closed", closed);
    h5.values(sources.get(), "conservation_error", conservation);
    h5.values(sources.get(), "transfer_error", transfer);
    h5.values(sources.get(), "wire_element_offsets", offsets);
    h5.pairs(sources.get(), "wire_elements", elements);
  }
  {
    auto circuits = h5.group(magstat.get(), "circuits");
    std::vector<int64_t> parallel, offsets{0};
    std::vector<uint8_t> closed;
    std::vector<std::pair<int64_t, int64_t>> members;
    for (const auto &c : results.circuits) {
      parallel.push_back(c.parallel);
      closed.push_back(c.closed);
      for (const auto [source, polarity] : c.members)
        members.emplace_back(results.sources[source].id, polarity);
      offsets.push_back(static_cast<int64_t>(members.size()));
    }
    h5.values(circuits.get(), "id", ids(results.circuits));
    h5.values(circuits.get(), "parallel", parallel);
    h5.values(circuits.get(), "closed", closed);
    h5.values(circuits.get(), "member_offsets", offsets);
    h5.pairs(circuits.get(), "members", members);
  }
  {
    auto forces = h5.group(magstat.get(), "forces");
    std::vector<int64_t> cid;
    std::vector<uint8_t> body;
    for (const auto &f : model.forces) {
      cid.push_back(f.coordinates);
      body.push_back(f.body);
    }
    h5.values(forces.get(), "id", ids(model.forces));
    h5.values(forces.get(), "body", body);
    h5.values(forces.get(), "cid", cid);
  }
  {
    auto probes = h5.group(magstat.get(), "probes");
    std::vector<int64_t> cd;
    std::vector<Vector> position;
    for (const auto &p : model.probes) {
      cd.push_back(p.output_coordinates);
      position.push_back(p.position);
    }
    h5.values(probes.get(), "id", ids(model.probes));
    h5.values(probes.get(), "cd", cd);
    h5.vectors(probes.get(), "position_m", position);
  }
  h5.values(magstat.get(), "subcase_ids", ids(results.cases));
  auto subcases = h5.group(magstat.get(), "subcases");
  for (size_t ci = 0; ci < results.cases.size(); ++ci) {
    const auto &c = results.cases[ci];
    auto group = h5.group(subcases.get(), std::to_string(c.id));
    h5.attribute(group.get(), "id", int64_t{c.id});
    h5.attribute(group.get(), "iterations", int64_t{c.iterations});
    h5.attribute(group.get(), "residual", c.residual);
    h5.attribute(group.get(), "energy_J", c.energy);
    h5.attribute(group.get(), "coenergy_J", c.coenergy);
    h5.attribute(group.get(), "bh_extrapolated", uint8_t{c.extrapolated});
    {
      auto sources = h5.group(group.get(), "sources");
      std::vector<double> current, linkage;
      for (const auto &s : c.sources) {
        current.push_back(s.current);
        linkage.push_back(s.closed ? s.linkage : std::nan(""));
      }
      h5.values(sources.get(), "current_A", current);
      h5.values(sources.get(), "linkage_Wb_turn", linkage);
      h5.matrix(sources.get(), "differential_inductance_H",
                c.differential_inductance);
    }
    {
      auto circuits = h5.group(group.get(), "circuits");
      std::vector<double> current, linkage;
      for (const auto &circuit : c.circuits) {
        current.push_back(circuit.current);
        linkage.push_back(circuit.linkage);
      }
      h5.values(circuits.get(), "current_A", current);
      h5.values(circuits.get(), "linkage_Wb_turn", linkage);
      h5.matrix(circuits.get(), "inductance_H", c.circuit_inductance);
    }
    {
      auto forces = h5.group(group.get(), "forces");
      std::vector<Vector> force, torque, force_local, torque_local,
          piece_force, piece_torque;
      std::vector<int64_t> offsets{0};
      for (size_t i = 0; i < c.forces.size(); ++i) {
        const auto &f = c.forces[i];
        const auto &request = model.forces[i];
        // Torque is about the CID origin; the local components use its axes.
        const Matrix axes =
            request.coordinates
                ? coordinate_axes(model.coordinates.at(request.coordinates))
                : Matrix::Identity();
        force.push_back(f.force);
        torque.push_back(f.torque);
        force_local.push_back(axes.transpose() * f.force);
        torque_local.push_back(axes.transpose() * f.torque);
        for (const auto &piece : f.components) {
          piece_force.push_back(piece.force);
          piece_torque.push_back(piece.torque);
        }
        offsets.push_back(static_cast<int64_t>(piece_force.size()));
      }
      h5.vectors(forces.get(), "force_N", force);
      h5.vectors(forces.get(), "torque_Nm", torque);
      h5.vectors(forces.get(), "force_local_N", force_local);
      h5.vectors(forces.get(), "torque_local_Nm", torque_local);
      h5.values(forces.get(), "piece_offsets", offsets);
      h5.vectors(forces.get(), "piece_force_N", piece_force);
      h5.vectors(forces.get(), "piece_torque_Nm", piece_torque);
    }
    if (model.subcases[ci].probes) {
      auto probes = h5.group(group.get(), "probes");
      std::vector<Vector> b_local, h_local;
      for (size_t i = 0; i < model.probes.size(); ++i) {
        const auto &probe = model.probes[i];
        // CD components use the local directions at the probe, e.g. radial
        // and tangential for a CORD2C.
        const Matrix axes =
            probe.output_coordinates
                ? local_axes(model.coordinates.at(probe.output_coordinates),
                             probe.position)
                : Matrix::Identity();
        b_local.push_back(axes.transpose() * c.probe_b[i]);
        h_local.push_back(axes.transpose() * c.probe_h[i]);
      }
      h5.vectors(probes.get(), "B_T", c.probe_b);
      h5.vectors(probes.get(), "H_A_per_m", c.probe_h);
      h5.vectors(probes.get(), "B_local_T", b_local);
      h5.vectors(probes.get(), "H_local_A_per_m", h_local);
    }
  }
  // Groups must close before the file so the flush covers everything.
  fields_node.reset();
  sources_node.reset();
  assembly.reset();
  vtkhdf.reset();
  subcases.reset();
  magstat.reset();
  return h5.finish();
}
#endif
} // namespace

bool output_available() {
#ifdef VIBESTRAN_HAS_HDF5
  return true;
#else
  return false;
#endif
}

Status write_results(const Model &model, const Results &results,
                     const std::filesystem::path &report,
                     const F06RunInfo *run) {
  auto base = report;
  base.replace_extension();
  std::ofstream f06(report);
  if (!f06)
    return Error{"Cannot open magnetic report " + report.string()};
  write_f06(f06, model, results, run);
  f06.close();
  if (!f06)
    return Error{"Failed writing magnetic report " + report.string()};
#ifdef VIBESTRAN_HAS_HDF5
  return write_hdf5(base.string() + ".mag.vtkhdf", model, results);
#else
  return Error{"SOL MAGSTAT results are written as HDF5, but this build has "
               "no HDF5; reconfigure with HDF5 installed (-Dhdf5=enabled)"};
#endif
}
} // namespace vibestran::magnetic
