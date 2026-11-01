#include "magnetic/solver.hpp"
#include "magnetic/air_gap.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <numeric>
#include <set>
#include <utility>
#ifdef VIBESTRAN_HAS_HDF5
#include <hdf5.h>
#endif

namespace vibestran::magnetic {
namespace {
// Torque on the moving part over a MOTION subcase's positions.
struct TorqueSummary {
  double mean{0}, peak_to_peak{0};
  // Percent of |mean|. NaN for a zero mean, and without current at any
  // position: cogging alone averages to zero over its period by energy
  // balance, so its mean is sampling residue and no base for a percentage.
  double ripple{std::numeric_limits<double>::quiet_NaN()};
  bool current_free{true}; // no current at any position: cogging
};
TorqueSummary torque_summary(const CaseResults &c) {
  TorqueSummary out;
  double low = std::numeric_limits<double>::infinity(), high = -low;
  for (const auto &step : c.steps) {
    out.mean += step.torque / static_cast<double>(c.steps.size());
    low = std::min(low, step.torque);
    high = std::max(high, step.torque);
    for (const auto &s : step.sources)
      out.current_free &= s.current == 0;
  }
  out.peak_to_peak = high - low;
  if (out.mean != 0 && !out.current_free)
    out.ripple = 100 * out.peak_to_peak / std::abs(out.mean);
  return out;
}
// The whole electrical periods a DRIVE subcase's positions sample
// uniformly (COUNT STEP POLES a multiple of 360 degrees), or 0.
int electrical_periods(const Model &model, const Subcase &subcase) {
  if (!subcase.drive)
    return 0;
  const auto &motion = model.motions.at(subcase.motion);
  const double span =
      motion.count * std::abs(motion.step) * model.drives.at(subcase.drive).poles;
  const auto periods = static_cast<int>(std::lround(span / 360));
  return periods >= 1 && 2 * periods < motion.count &&
                 std::abs(span - 360. * periods) <= 1e-9 * span
             ? periods
             : 0;
}
// The fundamental amplitude and total harmonic distortion (percent, over the
// electrical harmonics below the Nyquist rate) of samples covering periods
// whole electrical periods.
struct Spectrum {
  double fundamental{0}, distortion{0};
};
Spectrum spectrum(const std::vector<double> &samples, int periods) {
  const auto n = static_cast<double>(samples.size());
  auto amplitude = [&](int bin) {
    std::complex<double> sum = 0;
    for (size_t j = 0; j < samples.size(); ++j)
      sum += samples[j] *
             std::polar(1., -2 * std::numbers::pi * bin * static_cast<double>(j) / n);
    return 2 * std::abs(sum) / n;
  };
  Spectrum out;
  out.fundamental = amplitude(periods);
  double harmonics = 0;
  for (int h = 2; 2 * h * periods < static_cast<int>(samples.size()); ++h)
    harmonics += std::pow(amplitude(h * periods), 2);
  out.distortion = out.fundamental > 0 ? 100 * std::sqrt(harmonics) / out.fundamental
                                       : std::numeric_limits<double>::quiet_NaN();
  return out;
}
// Means of the dq quantities over a DRIVE subcase's positions.
DriveState drive_means(const CaseResults &c) {
  DriveState out;
  out.apparent.setZero();
  out.magnet.setZero();
  const auto n = static_cast<double>(c.steps.size());
  for (const auto &step : c.steps) {
    const auto &d = *step.drive;
    out.current += d.current / n;
    out.linkage += d.linkage / n;
    out.incremental += d.incremental / n;
    out.apparent += d.apparent / n;
    out.magnet += d.magnet / n;
  }
  return out;
}
// A circuit's dlambda/dphi at each position of a MOTION subcase.
std::vector<double> linkage_rates(const CaseResults &c, int circuit) {
  std::vector<double> out(c.steps.size());
  std::transform(c.steps.begin(), c.steps.end(), out.begin(),
                 [&](const CaseResults &step) { return step.linkage_rate[circuit]; });
  return out;
}
// The rotor speed in rad/s, which turns dlambda/dphi into volts.
double angular_speed(const Motion &motion) {
  return motion.speed * 2 * std::numbers::pi / 60;
}
void write_drive_f06(std::ostream &f06, const Model &model,
                     const Subcase &subcase, const CaseResults &c,
                     const std::vector<CircuitResults> &circuits) {
  const auto &drive = model.drives.at(subcase.drive);
  const auto &summary = *c.drive_summary;
  f06 << "DRIVE " << drive.id << ": " << (drive.six ? "six-step" : "sinusoidal")
      << " currents, " << drive.peak << " A peak, gamma " << drive.gamma
      << " electrical degrees, " << drive.poles
      << (drive.poles == 1 ? " pole pair" : " pole pairs") << ", THETA0 "
      << summary.theta0 << " degrees";
  if (summary.automatic)
    f06 << " (AUTO: phase 1 no-load fundamental " << summary.linkage << " Wb)";
  f06 << '\n';
  const auto mean = drive_means(c);
  f06 << "Mean over positions (amplitude-invariant dq): i_d(A) " << mean.current[0]
      << "  i_q(A) " << mean.current[1] << "  lambda_d(Wb) " << mean.linkage[0]
      << "  lambda_q(Wb) " << mean.linkage[1] << '\n'
      << "Incremental inductance, mean (H): L_d " << mean.incremental(0, 0)
      << "  L_q " << mean.incremental(1, 1) << "  L_dq " << mean.incremental(0, 1)
      << '\n';
  if (subcase.apparent)
    f06 << "Apparent inductance (frozen permeability, magnets off), mean (H): L_d "
        << mean.apparent(0, 0) << "  L_q " << mean.apparent(1, 1) << "  L_dq "
        << mean.apparent(0, 1) << "  magnet linkage lambda_m,d(Wb) "
        << mean.magnet[0] << "  lambda_m,q(Wb) " << mean.magnet[1] << '\n';
  const auto &motion = model.motions.at(subcase.motion);
  if (const int periods = electrical_periods(model, subcase)) {
    for (size_t k = 0; k < summary.phases.size(); ++k) {
      const auto s = spectrum(linkage_rates(c, summary.phases[k]), periods);
      f06 << "Phase " << k + 1 << " (circuit " << circuits[summary.phases[k]].id
          << ") back-EMF over " << periods << " electrical "
          << (periods == 1 ? "period" : "periods")
          << ": fundamental constant (V s/rad, peak) " << s.fundamental;
      if (motion.speed != 0)
        f06 << "  fundamental RMS at " << motion.speed << " rpm (V) "
            << s.fundamental * std::abs(angular_speed(motion)) / std::sqrt(2.);
      f06 << "  THD (%) " << s.distortion << '\n';
    }
  }
}
void write_motion_f06(std::ostream &f06, const Model &model,
                      const Subcase &subcase, const CaseResults &c) {
  const auto &motion = model.motions.at(subcase.motion);
  const auto summary = torque_summary(c);
  f06 << "\nSUBCASE " << c.id << "\nMOTION " << motion.id << ": "
      << c.steps.size() << (c.steps.size() == 1 ? " position" : " positions")
      << " from " << c.steps.front().position << " to "
      << c.steps.back().position << " degrees about the z axis of CID "
      << motion.coordinates << "\nNewton iterations: " << c.iterations
      << "  largest relative residual: " << c.residual << '\n';
  if (c.extrapolated)
    f06 << "BH curve exceeded: vacuum-slope extrapolation used.\n";
  f06 << "Torque on ESET " << motion.set
      << " about the MOTION axis (Nm): mean " << summary.mean
      << "  peak-to-peak " << summary.peak_to_peak << "  ripple ";
  if (std::isfinite(summary.ripple))
    f06 << summary.ripple << " %\n";
  else
    f06 << (summary.current_free ? "undefined (no current: cogging)\n"
                                 : "undefined (zero mean)\n");
  if (summary.current_free)
    f06 << "No current at any position: cogging torque peak-to-peak (Nm) "
        << summary.peak_to_peak << '\n';
  if (c.drive_summary)
    write_drive_f06(f06, model, subcase, c, c.steps.front().circuits);
  if (!subcase.steps)
    return;
  // MSTEP = ALL: one row per position, with every closed circuit's linkage.
  f06 << "\n  STEP   POSITION(deg)          TORQUE(Nm)           ENERGY(J)"
         "         COENERGY(J)  ITERATIONS     RESIDUAL";
  const auto &first = c.steps.front();
  for (const auto &circuit : first.circuits)
    if (std::isfinite(circuit.linkage))
      f06 << "  LINKAGE " << std::setw(6) << circuit.id << "(Wb-turn)";
  f06 << '\n';
  for (size_t k = 0; k < c.steps.size(); ++k) {
    const auto &step = c.steps[k];
    f06 << std::setw(6) << k + 1 << std::setw(16) << step.position
        << std::setw(20) << step.torque << std::setw(20) << step.energy
        << std::setw(20) << step.coenergy << std::setw(12) << step.iterations
        << std::setw(13) << std::setprecision(4) << step.residual
        << std::setprecision(12);
    for (const auto &circuit : step.circuits)
      if (std::isfinite(circuit.linkage))
        f06 << std::setw(25) << circuit.linkage;
    f06 << '\n';
  }
  if (c.drive_summary) {
    f06 << "\n  STEP   THETA_E(deg)            I_D(A)            I_Q(A)"
           "      LAMBDA_D(Wb)      LAMBDA_Q(Wb)      LD_INC(H)      LQ_INC(H)";
    if (subcase.apparent)
      f06 << "      LD_APP(H)      LQ_APP(H)";
    f06 << '\n';
    for (size_t k = 0; k < c.steps.size(); ++k) {
      const auto &d = *c.steps[k].drive;
      f06 << std::setw(6) << k + 1 << std::setw(15) << d.electrical << std::setw(18)
          << d.current[0] << std::setw(18) << d.current[1] << std::setw(18)
          << d.linkage[0] << std::setw(18) << d.linkage[1] << std::setw(15)
          << d.incremental(0, 0) << std::setw(15) << d.incremental(1, 1);
      if (subcase.apparent)
        f06 << std::setw(15) << d.apparent(0, 0) << std::setw(15) << d.apparent(1, 1);
      f06 << '\n';
    }
  }
  if (!first.linkage_rate.array().isFinite().any())
    return;
  // Back-EMF per unit speed, and in volts with SPEED.
  const bool volts = motion.speed != 0;
  f06 << "\n  STEP   POSITION(deg)";
  for (Eigen::Index i = 0; i < first.linkage_rate.size(); ++i)
    if (std::isfinite(first.linkage_rate[i]))
      f06 << (volts ? "         EMF " : "  DLAMBDA/DPHI ") << std::setw(6)
          << first.circuits[i].id << (volts ? "(V)" : "(V s/rad)");
  f06 << '\n';
  for (size_t k = 0; k < c.steps.size(); ++k) {
    const auto &step = c.steps[k];
    f06 << std::setw(6) << k + 1 << std::setw(16) << step.position;
    for (Eigen::Index i = 0; i < step.linkage_rate.size(); ++i)
      if (std::isfinite(first.linkage_rate[i]))
        f06 << std::setw(volts ? 25 : 31)
            << step.linkage_rate[i] * (volts ? angular_speed(motion) : 1);
    f06 << '\n';
  }
}

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
  for (size_t ci = 0; ci < results.cases.size(); ++ci) {
    const auto &c = results.cases[ci];
    if (!c.steps.empty()) {
      write_motion_f06(f06, model, model.subcases[ci], c);
      continue;
    }
    f06 << "\nSUBCASE " << c.id << "\nNewton iterations: " << c.iterations
        << "  relative residual: " << c.residual << "\nEnergy (J): " << c.energy
        << "  source Legendre transform (J): " << c.coenergy << '\n';
    if (c.extrapolated)
      f06 << "BH curve exceeded: vacuum-slope extrapolation used.\n";
    if (std::isfinite(c.gap_torque))
      f06 << "Air-gap torque on the part inside the gap, about the plane "
             "normal (Nm): "
          << c.gap_torque << '\n';
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
  // Row-major data of the given shape.
  template <class T>
  void array(hid_t parent, const std::string &name, const std::vector<T> &data,
             std::vector<hsize_t> dims) {
    dataset(parent, name, data.data(), std::move(dims));
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

// The cells of one UnstructuredGrid block of the VTKHDF collection and the
// points they use: the selected mesh cells with their own compacted points,
// then any air-gap elements, drawn with the deck's grids where the deck puts
// them (the air-gap element does not use them, and under MOTION they stay
// in the fixed frame).
struct BlockTopology {
  std::vector<int> cells, nodes; // mesh cells, and the mesh node of each point
  std::vector<int> gap_grids;    // grid ID of each gap point, after the nodes
  std::vector<Vector> gap_centroids;
  std::vector<int64_t> connectivity, offsets{0}, element_ids;
  std::vector<uint8_t> types;
};
BlockTopology block_topology(const Mesh &mesh, std::vector<int> cells,
                             const Model *model = nullptr,
                             const AirGap *gap = nullptr) {
  BlockTopology t;
  if (cells.empty()) {
    cells.resize(mesh.cells.size());
    std::iota(cells.begin(), cells.end(), 0);
  }
  std::vector<int> point(mesh.nodes.size(), -1);
  for (int c : cells) {
    const auto &cell = mesh.cells[c];
    for (int n : cell.nodes) {
      if (point[n] < 0) {
        point[n] = static_cast<int>(t.nodes.size());
        t.nodes.push_back(n);
      }
      t.connectivity.push_back(point[n]);
    }
    t.offsets.push_back(static_cast<int64_t>(t.connectivity.size()));
    const size_t nodes = cell.nodes.size();
    // VTK triangle, quad, tetra, wedge and hexahedron.
    t.types.push_back(mesh.planar ? (nodes == 3 ? 5 : 9)
                                  : (nodes == 4   ? 10
                                     : nodes == 6 ? 13
                                                  : 12));
    t.element_ids.push_back(cell.id);
  }
  t.cells = std::move(cells);
  if (!gap)
    return t;
  std::map<int, int64_t> grid_point;
  for (const auto &e : model->elements) {
    if (e.property != gap->property)
      continue;
    Vector centroid = Vector::Zero();
    for (int id : e.nodes) {
      auto [it, added] = grid_point.emplace(
          id, static_cast<int64_t>(t.nodes.size() + t.gap_grids.size()));
      if (added)
        t.gap_grids.push_back(id);
      t.connectivity.push_back(it->second);
      centroid += model->nodes.at(id).position / static_cast<double>(e.nodes.size());
    }
    t.offsets.push_back(static_cast<int64_t>(t.connectivity.size()));
    t.types.push_back(e.nodes.size() == 3 ? 5 : 9);
    t.element_ids.push_back(e.id);
    t.gap_centroids.push_back(centroid);
  }
  return t;
}
// One state of a block: its points and its per-cell and per-point fields.
struct BlockState {
  std::vector<Vector> points;
  std::vector<std::pair<std::string, std::vector<Vector>>> fields;
  std::vector<double> potential; // empty for none
};
// A subcase state: B and H on the cells, the planar potential at the
// points, and on the gap cells the air-gap element's own field (B at each
// centroid, H = B / mu0, A at each point) from the potential of that state.
BlockState field_state(const BlockTopology &t, const Model &model,
                       const Mesh &mesh, const std::vector<Vector> &nodes,
                       const CaseResults &state, const AirGap *gap) {
  BlockState out;
  for (int n : t.nodes)
    out.points.push_back(nodes[n]);
  std::vector<Vector> b, h;
  for (int c : t.cells) {
    b.push_back(state.b[c]);
    h.push_back(state.h[c]);
  }
  std::optional<AirGapSeries> series;
  if (gap)
    series.emplace(*gap, state.potential);
  for (const auto &x : t.gap_centroids) {
    b.push_back(series->field(x));
    h.push_back(b.back() / mu0);
  }
  out.fields = {{"B_T", std::move(b)}, {"H_A_per_m", std::move(h)}};
  if (mesh.planar &&
      state.potential.size() >= static_cast<Eigen::Index>(mesh.nodes.size())) {
    for (int n : t.nodes)
      out.potential.push_back(state.potential[n]);
    for (int id : t.gap_grids)
      out.potential.push_back(series->potential(model.nodes.at(id).position));
  }
  for (int id : t.gap_grids)
    out.points.push_back(model.nodes.at(id).position);
  return out;
}
// Writes a block. Without times it is a single state. With times (degrees)
// it is a VTKHDF time series: the topology is written once and time j shows
// state state_at[j], so a block with fewer states, or a static one, repeats
// them without copies.
void write_block(Hdf5File &h5, hid_t vtkhdf, const std::string &name,
                 int64_t index, const BlockTopology &t,
                 const std::vector<BlockState> &states,
                 const std::vector<double> &times,
                 const std::vector<int64_t> &state_at) {
  const auto points = static_cast<int64_t>(states.front().points.size()),
             cells = static_cast<int64_t>(t.types.size());
  auto block = h5.group(vtkhdf, name);
  h5.attribute(block.get(), "Version", std::vector<int64_t>{2, 5});
  h5.attribute(block.get(), "Type", std::string("UnstructuredGrid"));
  h5.attribute(block.get(), "Index", index);
  h5.values(block.get(), "NumberOfPoints", std::vector<int64_t>{points});
  h5.values(block.get(), "NumberOfCells", std::vector<int64_t>{cells});
  h5.values(block.get(), "NumberOfConnectivityIds",
            std::vector<int64_t>{static_cast<int64_t>(t.connectivity.size())});
  std::vector<Vector> coordinates;
  for (const auto &state : states)
    coordinates.insert(coordinates.end(), state.points.begin(), state.points.end());
  h5.vectors(block.get(), "Points", coordinates);
  h5.values(block.get(), "Connectivity", t.connectivity);
  h5.values(block.get(), "Offsets", t.offsets);
  h5.values(block.get(), "Types", t.types);
  auto cell_data = h5.group(block.get(), "CellData");
  h5.values(cell_data.get(), "element_id", t.element_ids);
  for (size_t f = 0; f < states.front().fields.size(); ++f) {
    std::vector<Vector> values;
    for (const auto &state : states)
      values.insert(values.end(), state.fields[f].second.begin(),
                    state.fields[f].second.end());
    h5.vectors(cell_data.get(), states.front().fields[f].first, values);
  }
  const bool potential = !states.front().potential.empty();
  if (potential) {
    auto point_data = h5.group(block.get(), "PointData");
    std::vector<double> values;
    for (const auto &state : states)
      values.insert(values.end(), state.potential.begin(), state.potential.end());
    h5.values(point_data.get(), "A_normal_Wb_per_m", values);
  }
  if (times.empty())
    return;
  auto steps = h5.group(block.get(), "Steps");
  const auto count = static_cast<int64_t>(times.size());
  h5.attribute(steps.get(), "NSteps", count);
  h5.values(steps.get(), "Values", times);
  const std::vector<int64_t> zeros(times.size(), 0), ones(times.size(), 1);
  auto scaled = [&](int64_t size) {
    std::vector<int64_t> out;
    for (int64_t k : state_at)
      out.push_back(k * size);
    return out;
  };
  h5.values(steps.get(), "PartOffsets", zeros);
  h5.values(steps.get(), "NumberOfParts", ones);
  h5.values(steps.get(), "PointOffsets", scaled(points));
  h5.values(steps.get(), "CellOffsets", zeros);
  h5.values(steps.get(), "ConnectivityIdOffsets", zeros);
  auto cell_offsets = h5.group(steps.get(), "CellDataOffsets");
  h5.values(cell_offsets.get(), "element_id", zeros);
  for (const auto &[field, values] : states.front().fields)
    h5.values(cell_offsets.get(), field, scaled(cells));
  if (potential) {
    auto point_offsets = h5.group(steps.get(), "PointDataOffsets");
    h5.values(point_offsets.get(), "A_normal_Wb_per_m", scaled(points));
  }
}

// The per-subcase source, circuit, force and probe tables of one or more
// states. stepped gives every dataset a leading axis over the states (MOTION
// positions); a single static state is written without it.
void write_tables(Hdf5File &h5, hid_t group, const Model &model,
                  const Subcase &subcase,
                  const std::vector<const CaseResults *> &states, bool stepped) {
  auto shape = [&](std::vector<hsize_t> dims) {
    if (stepped)
      dims.insert(dims.begin(), states.size());
    return dims;
  };
  auto flat = [](std::vector<double> &out, const Vector &v) {
    out.insert(out.end(), {v.x(), v.y(), v.z()});
  };
  auto rows = [](std::vector<double> &out, const Eigen::MatrixXd &m) {
    for (Eigen::Index i = 0; i < m.rows(); ++i)
      for (Eigen::Index j = 0; j < m.cols(); ++j)
        out.push_back(m(i, j));
  };
  const auto &first = *states.front();
  const hsize_t ns = first.sources.size(), nc = first.circuits.size(),
                nf = first.forces.size(), np = model.probes.size();
  {
    auto sources = h5.group(group, "sources");
    std::vector<double> current, linkage, inductance;
    for (const auto *c : states) {
      for (const auto &s : c->sources) {
        current.push_back(s.current);
        linkage.push_back(s.closed ? s.linkage : std::nan(""));
      }
      rows(inductance, c->differential_inductance);
    }
    h5.array(sources.get(), "current_A", current, shape({ns}));
    h5.array(sources.get(), "linkage_Wb_turn", linkage, shape({ns}));
    h5.array(sources.get(), "differential_inductance_H", inductance, shape({ns, ns}));
  }
  {
    auto circuits = h5.group(group, "circuits");
    std::vector<double> current, linkage, inductance;
    for (const auto *c : states) {
      for (const auto &circuit : c->circuits) {
        current.push_back(circuit.current);
        linkage.push_back(circuit.linkage);
      }
      rows(inductance, c->circuit_inductance);
    }
    h5.array(circuits.get(), "current_A", current, shape({nc}));
    h5.array(circuits.get(), "linkage_Wb_turn", linkage, shape({nc}));
    h5.array(circuits.get(), "inductance_H", inductance, shape({nc, nc}));
    // With an air-gap element: dlambda/d(delta), the back-EMF per unit speed.
    if (first.linkage_rate.array().isFinite().any()) {
      std::vector<double> rate;
      for (const auto *c : states)
        rate.insert(rate.end(), c->linkage_rate.data(),
                    c->linkage_rate.data() + c->linkage_rate.size());
      h5.array(circuits.get(), "linkage_rate_Wb_per_rad", rate, shape({nc}));
    }
  }
  {
    auto forces = h5.group(group, "forces");
    std::vector<double> force, torque, force_local, torque_local, piece_force,
        piece_torque;
    std::vector<int64_t> offsets{0};
    for (const auto &f : first.forces)
      offsets.push_back(offsets.back() + static_cast<int64_t>(f.components.size()));
    for (const auto *c : states)
      for (size_t i = 0; i < c->forces.size(); ++i) {
        const auto &f = c->forces[i];
        const auto &request = model.forces[i];
        // Torque is about the CID origin; the local components use its axes.
        const Matrix axes =
            request.coordinates
                ? coordinate_axes(model.coordinates.at(request.coordinates))
                : Matrix::Identity();
        flat(force, f.force);
        flat(torque, f.torque);
        flat(force_local, axes.transpose() * f.force);
        flat(torque_local, axes.transpose() * f.torque);
        for (const auto &piece : f.components) {
          flat(piece_force, piece.force);
          flat(piece_torque, piece.torque);
        }
      }
    const auto pieces = static_cast<hsize_t>(offsets.back());
    h5.array(forces.get(), "force_N", force, shape({nf, 3}));
    h5.array(forces.get(), "torque_Nm", torque, shape({nf, 3}));
    h5.array(forces.get(), "force_local_N", force_local, shape({nf, 3}));
    h5.array(forces.get(), "torque_local_Nm", torque_local, shape({nf, 3}));
    h5.values(forces.get(), "piece_offsets", offsets);
    h5.array(forces.get(), "piece_force_N", piece_force, shape({pieces, 3}));
    h5.array(forces.get(), "piece_torque_Nm", piece_torque, shape({pieces, 3}));
  }
  if (subcase.probes) {
    auto probes = h5.group(group, "probes");
    std::vector<double> b, h, b_local, h_local;
    for (const auto *c : states)
      for (size_t i = 0; i < model.probes.size(); ++i) {
        const auto &probe = model.probes[i];
        // CD components use the local directions at the probe, e.g. radial
        // and tangential for a CORD2C.
        const Matrix axes =
            probe.output_coordinates
                ? local_axes(model.coordinates.at(probe.output_coordinates),
                             probe.position)
                : Matrix::Identity();
        flat(b, c->probe_b[i]);
        flat(h, c->probe_h[i]);
        flat(b_local, axes.transpose() * c->probe_b[i]);
        flat(h_local, axes.transpose() * c->probe_h[i]);
      }
    h5.array(probes.get(), "B_T", b, shape({np, 3}));
    h5.array(probes.get(), "H_A_per_m", h, shape({np, 3}));
    h5.array(probes.get(), "B_local_T", b_local, shape({np, 3}));
    h5.array(probes.get(), "H_local_A_per_m", h_local, shape({np, 3}));
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
  // With MOTION fields the collection is a time series whose time is the
  // position in degrees: every position BFIELD writes in any subcase. Each
  // block holds, at each time, its state at the largest of its own positions
  // not above it (its first position before that); static blocks hold their
  // one state throughout. VTK requires one step count for all blocks.
  std::vector<double> times;
  for (size_t ci = 0; ci < results.cases.size(); ++ci)
    if (model.subcases[ci].field.enabled)
      for (const auto &step : results.cases[ci].steps)
        if (!step.b.empty())
          times.push_back(step.position);
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  auto held = [&times](const std::vector<double> &positions) {
    std::vector<int64_t> out;
    if (times.empty())
      return out;
    for (double t : times) {
      int64_t best = -1;
      for (size_t k = 0; k < positions.size(); ++k)
        if (positions[k] <= t && (best < 0 || positions[k] >= positions[best]))
          best = static_cast<int64_t>(k);
      if (best < 0)
        best = std::min_element(positions.begin(), positions.end()) - positions.begin();
      out.push_back(best);
    }
    return out;
  };
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
    // The air gap is drawn with every cell; an ESET holds field cells only.
    const AirGap *gap =
        request.set == 0 && results.mesh.air_gap ? &*results.mesh.air_gap : nullptr;
    const auto topology = block_topology(results.mesh, std::move(cells), &model, gap);
    std::vector<BlockState> states;
    std::vector<double> positions;
    if (c.steps.empty())
      states.push_back(field_state(topology, model, results.mesh,
                                   results.mesh.nodes, c, gap));
    for (const auto &step : c.steps) {
      if (step.b.empty())
        continue;
      states.push_back(field_state(topology, model, results.mesh, step.nodes, step,
                                   step.air_gap ? &*step.air_gap : nullptr));
      positions.push_back(step.position);
    }
    const auto name = "SUBCASE_" + std::to_string(c.id);
    write_block(h5, vtkhdf.get(), name, index++, topology, states, times,
                positions.empty() ? std::vector<int64_t>(times.size(), 0)
                                  : held(positions));
    if (!fields_node.valid())
      fields_node = h5.group(assembly.get(), "fields");
    h5.link(fields_node.get(), name, "/VTKHDF/" + name);
  }
  if (model.options.source_output)
    for (const auto &source : results.sources) {
      const auto name =
          (source.wire ? "WIRE_" : "COIL_") + std::to_string(source.id);
      const auto topology = block_topology(source.mesh, {});
      BlockState state;
      for (int n : topology.nodes)
        state.points.push_back(source.mesh.nodes[n]);
      const auto density = source_density(source);
      std::vector<Vector> selected;
      for (int c : topology.cells)
        selected.push_back(density[c]);
      state.fields = {{"J_per_ampere_per_m2", std::move(selected)}};
      write_block(h5, vtkhdf.get(), name, index++, topology, {state}, times,
                  std::vector<int64_t>(times.size(), 0));
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
  h5.attribute(magstat.get(), "axisymmetric", int64_t{results.mesh.axisymmetric ? 1 : 0});
  if (const auto &gap = results.mesh.air_gap) {
    auto group = h5.group(magstat.get(), "air_gap");
    h5.attribute(group.get(), "property", int64_t{gap->property});
    h5.attribute(group.get(), "inner_radius_m", gap->inner.radius);
    h5.attribute(group.get(), "outer_radius_m", gap->outer.radius);
    h5.attribute(group.get(), "harmonics", int64_t{gap->harmonics});
    h5.attribute(group.get(), "band_points", int64_t{gap->band_points});
    h5.attribute(group.get(), "band_layers", int64_t{gap->band_points > 0 ? gap->band_layers : 0});
    h5.attribute(group.get(), "radial_deviation", gap->deviation);
    h5.vectors(group.get(), "center_m", {gap->center});
    h5.attribute(group.get(), "sectors", int64_t{gap->sectors});
    // A sector model's circles are unfolded around the machine: each grid
    // once, in angle order.
    for (const auto &[name, circle] :
         {std::pair{"inner_grids", &gap->inner}, std::pair{"outer_grids", &gap->outer}}) {
      std::vector<int64_t> grids;
      std::set<int> seen;
      for (int n : circle->nodes)
        if (seen.insert(n).second)
          grids.push_back(results.mesh.node_ids[n]);
      h5.values(group.get(), name, grids);
    }
  }
  if (const auto &period = results.mesh.period) {
    auto group = h5.group(magstat.get(), "periodic");
    h5.attribute(group.get(), "id", int64_t{period->id});
    h5.attribute(group.get(), "type", std::string(period->anti ? "ANTI" : "PERIODIC"));
    h5.attribute(group.get(), "sectors", int64_t{period->sectors});
    h5.attribute(group.get(), "sector_angle_deg", 360. / period->sectors);
    h5.vectors(group.get(), "origin_m", {period->origin});
    h5.vectors(group.get(), "axis", {period->axis});
    std::vector<int64_t> first, second, axial;
    for (auto [a, b] : period->pairs) {
      first.push_back(results.mesh.node_ids[a]);
      second.push_back(results.mesh.node_ids[b]);
    }
    for (int v : period->axial)
      axial.push_back(results.mesh.node_ids[v]);
    h5.values(group.get(), "first_grids", first);
    h5.values(group.get(), "second_grids", second);
    h5.values(group.get(), "axial_grids", axial);
  }
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
    const auto &subcase = model.subcases[ci];
    auto group = h5.group(subcases.get(), std::to_string(c.id));
    h5.attribute(group.get(), "id", int64_t{c.id});
    h5.attribute(group.get(), "iterations", int64_t{c.iterations});
    h5.attribute(group.get(), "residual", c.residual);
    h5.attribute(group.get(), "bh_extrapolated", uint8_t{c.extrapolated});
    if (c.steps.empty()) {
      h5.attribute(group.get(), "energy_J", c.energy);
      h5.attribute(group.get(), "coenergy_J", c.coenergy);
      if (std::isfinite(c.gap_torque))
        h5.attribute(group.get(), "air_gap_torque_Nm", c.gap_torque);
      write_tables(h5, group.get(), model, subcase, {&c}, false);
      continue;
    }
    // A MOTION subcase: the summary always, and with MSTEP = ALL every
    // per-position value, with a leading position axis.
    const auto &motion = model.motions.at(subcase.motion);
    const auto summary = torque_summary(c);
    h5.attribute(group.get(), "motion", int64_t{motion.id});
    h5.attribute(group.get(), "positions", static_cast<int64_t>(c.steps.size()));
    h5.attribute(group.get(), "start_deg", motion.start);
    h5.attribute(group.get(), "step_deg", motion.step);
    h5.attribute(group.get(), "torque_mean_Nm", summary.mean);
    h5.attribute(group.get(), "torque_peak_to_peak_Nm", summary.peak_to_peak);
    h5.attribute(group.get(), "torque_ripple_percent", summary.ripple);
    h5.attribute(group.get(), "current_free", uint8_t{summary.current_free});
    if (motion.speed != 0)
      h5.attribute(group.get(), "speed_rpm", motion.speed);
    if (c.drive_summary) {
      const auto &drive = model.drives.at(subcase.drive);
      const auto &found = *c.drive_summary;
      const auto mean = drive_means(c);
      auto attributes = h5.group(group.get(), "drive");
      const hid_t d = attributes.get();
      h5.attribute(d, "id", int64_t{drive.id});
      h5.attribute(d, "wave", std::string(drive.six ? "SIX" : "SINE"));
      h5.attribute(d, "pole_pairs", int64_t{drive.poles});
      h5.attribute(d, "current_peak_A", drive.peak);
      h5.attribute(d, "gamma_deg", drive.gamma);
      h5.attribute(d, "theta0_deg", found.theta0);
      h5.attribute(d, "theta0_auto", uint8_t{found.automatic});
      if (found.automatic) {
        h5.attribute(d, "alignment_linkage_Wb", found.linkage);
        h5.values(d, "alignment_lag_deg", found.lag);
      }
      std::vector<int64_t> phases;
      for (int k : found.phases)
        phases.push_back(c.steps.front().circuits[k].id);
      h5.values(d, "phase_circuits", phases);
      h5.attribute(d, "i_d_mean_A", mean.current[0]);
      h5.attribute(d, "i_q_mean_A", mean.current[1]);
      h5.attribute(d, "lambda_d_mean_Wb", mean.linkage[0]);
      h5.attribute(d, "lambda_q_mean_Wb", mean.linkage[1]);
      h5.attribute(d, "L_d_incremental_mean_H", mean.incremental(0, 0));
      h5.attribute(d, "L_q_incremental_mean_H", mean.incremental(1, 1));
      if (subcase.apparent) {
        h5.attribute(d, "L_d_apparent_mean_H", mean.apparent(0, 0));
        h5.attribute(d, "L_q_apparent_mean_H", mean.apparent(1, 1));
        h5.attribute(d, "lambda_magnet_d_mean_Wb", mean.magnet[0]);
        h5.attribute(d, "lambda_magnet_q_mean_Wb", mean.magnet[1]);
      }
      if (const int periods = electrical_periods(model, subcase)) {
        std::vector<double> constant, distortion;
        for (int k : found.phases) {
          const auto spectral = spectrum(linkage_rates(c, k), periods);
          constant.push_back(spectral.fundamental);
          distortion.push_back(spectral.distortion);
        }
        h5.attribute(d, "electrical_periods", int64_t{periods});
        h5.values(d, "emf_constant_fundamental_V_s_per_rad", constant);
        h5.values(d, "emf_thd_percent", distortion);
      }
      if (subcase.steps) {
        std::vector<double> angle, id, iq, ld, lq, incremental, apparent, md, mq;
        for (const auto &step : c.steps) {
          const auto &s = *step.drive;
          angle.push_back(s.electrical);
          id.push_back(s.current[0]);
          iq.push_back(s.current[1]);
          ld.push_back(s.linkage[0]);
          lq.push_back(s.linkage[1]);
          md.push_back(s.magnet[0]);
          mq.push_back(s.magnet[1]);
          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
              incremental.push_back(s.incremental(i, j));
              apparent.push_back(s.apparent(i, j));
            }
        }
        const hsize_t steps = c.steps.size();
        h5.values(d, "theta_e_deg", angle);
        h5.values(d, "i_d_A", id);
        h5.values(d, "i_q_A", iq);
        h5.values(d, "lambda_d_Wb", ld);
        h5.values(d, "lambda_q_Wb", lq);
        h5.array(d, "L_dq_incremental_H", incremental, {steps, 2, 2});
        if (subcase.apparent) {
          h5.array(d, "L_dq_apparent_H", apparent, {steps, 2, 2});
          h5.values(d, "lambda_magnet_d_Wb", md);
          h5.values(d, "lambda_magnet_q_Wb", mq);
        }
      }
    }
    if (!subcase.steps)
      continue;
    std::vector<const CaseResults *> states;
    std::vector<double> position, residual, energy, coenergy, torque;
    std::vector<int64_t> iterations;
    std::vector<uint8_t> extrapolated;
    for (const auto &step : c.steps) {
      states.push_back(&step);
      position.push_back(step.position);
      residual.push_back(step.residual);
      energy.push_back(step.energy);
      coenergy.push_back(step.coenergy);
      torque.push_back(step.torque);
      iterations.push_back(step.iterations);
      extrapolated.push_back(step.extrapolated);
    }
    h5.values(group.get(), "positions_deg", position);
    h5.values(group.get(), "iterations", iterations);
    h5.values(group.get(), "residual", residual);
    h5.values(group.get(), "energy_J", energy);
    h5.values(group.get(), "coenergy_J", coenergy);
    h5.values(group.get(), "bh_extrapolated", extrapolated);
    auto motion_group = h5.group(group.get(), "motion");
    h5.values(motion_group.get(), "torque_Nm", torque);
    if (motion.speed != 0 && c.steps.front().linkage_rate.array().isFinite().any()) {
      std::vector<double> emf;
      for (const auto &step : c.steps)
        for (Eigen::Index i = 0; i < step.linkage_rate.size(); ++i)
          emf.push_back(step.linkage_rate[i] * angular_speed(motion));
      h5.array(motion_group.get(), "emf_V", emf,
               {c.steps.size(), static_cast<hsize_t>(c.steps.front().linkage_rate.size())});
    }
    write_tables(h5, group.get(), model, subcase, states, true);
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
