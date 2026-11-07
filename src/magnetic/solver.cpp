#include "magnetic/solver.hpp"
#include "diffusion.hpp"
#include "solid_diffusion.hpp"
#include "core/notices.hpp"
#include "magnetic/air_gap.hpp"
#include "magnetic/drive.hpp"
#include "magnetic/motion.hpp"
#include "solver/solver_backend.hpp"
#include "solver/tangent_solver.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <format>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#ifdef HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

namespace vibestran::magnetic {
namespace {
using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
SparseMatrixBuilder::CsrData lower_csr(const Sparse &matrix) {
  const Eigen::SparseMatrix<double, Eigen::RowMajor> lower =
      matrix.triangularView<Eigen::Lower>();
  SparseMatrixBuilder::CsrData csr;
  csr.n = static_cast<int>(lower.rows());
  csr.nnz = static_cast<int>(lower.nonZeros());
  csr.row_ptr.assign(lower.outerIndexPtr(), lower.outerIndexPtr() + csr.n + 1);
  csr.col_ind.assign(lower.innerIndexPtr(), lower.innerIndexPtr() + csr.nnz);
  csr.values.assign(lower.valuePtr(), lower.valuePtr() + csr.nnz);
  csr.symmetry = SymmetryStorage::Lower;
  return csr;
}
// The gauged tangent, solved with the backend's factorization. Newton
// iterations, continuation stages, subcases and inductances all assemble the
// same pattern, so its symbolic analysis is done once. Every factorization
// must be positive definite: a failure means a gauge, mesh or material
// error, never a reason to fall back to LDLT. With PARAM,MAGPCG the factor is
// reused across tangents as a CG preconditioner (TangentSolver).
class Tangent {
public:
  // planar: order with minimum degree (FactorOptions::minimum_degree).
  // dense_block: the model has an air gap, whose harmonic coupling is a
  // dense block in every tangent (FactorOptions::dense_block). reuse: CG
  // iterations with an earlier tangent's factor before refactoring; 0
  // factors every tangent.
  Tangent(SolverBackend &backend, bool planar, bool dense_block, int reuse)
      : solver_(backend, options(planar, dense_block, reuse)), reuse_(reuse > 0) {}
  Status update(const Sparse &stiffness) {
    return solver_.update(lower_csr(stiffness));
  }
  // Solves for every column of rhs with the last update()'s tangent, each to
  // a relative residual of tolerance (0: a direct solve).
  Result<Eigen::MatrixXd> solve(const Eigen::MatrixXd &rhs, double tolerance) {
    auto x = solver_.solve({rhs.data(), static_cast<size_t>(rhs.size())},
                           static_cast<int>(rhs.cols()), tolerance);
    if (auto *e = std::get_if<Error>(&x))
      return *e;
    const auto &values = std::get<std::vector<double>>(x);
    Eigen::MatrixXd out =
        Eigen::Map<const Eigen::MatrixXd>(values.data(), rhs.rows(), rhs.cols());
    // Checked here too: this file keeps IEEE semantics, the backend may not.
    if (!out.allFinite())
      return Error{"Magnetic linear solve produced a nonfinite solution"};
    return out;
  }
  // Whether solves may be inexact (PARAM,MAGPCG).
  [[nodiscard]] bool reuses() const { return reuse_; }

  // The last analysis's fill-reducing ordering; empty for an adapter.
  [[nodiscard]] std::string_view ordering() const { return solver_.ordering(); }
  [[nodiscard]] const TangentSolverStats &stats() const { return solver_.stats(); }
  // Work done so far, for the F06 header.
  [[nodiscard]] std::string usage() const {
    const auto &s = solver_.stats();
    auto text = std::format(
        "{} symbolic {}, {} numeric {}, {} right-hand {}", s.symbolic_analyses,
        s.symbolic_analyses == 1 ? "analysis" : "analyses",
        s.numeric_factorizations,
        s.numeric_factorizations == 1 ? "factorization" : "factorizations",
        s.right_hand_sides, s.right_hand_sides == 1 ? "side" : "sides");
    if (reuse_)
      text += std::format(", {} by CG with an earlier factor in {} iterations "
                          "({} more stopped CG and refactored)",
                          s.iterative_solves, s.iterations, s.failed_iterative);
    return text;
  }

private:
  static TangentSolverOptions options(bool planar, bool dense_block, int reuse) {
    TangentSolverOptions o;
    o.factor.require_spd = true;
    o.factor.minimum_degree = planar;
    o.factor.dense_block = dense_block;
    o.reuse_factor = reuse > 0;
    o.max_iterations = std::max(reuse, 1);
    return o;
  }
  TangentSolver solver_;
  bool reuse_;
};
struct Assembly {
  Sparse stiffness;
  Eigen::VectorXd residual;
  std::vector<Vector> b, h;
  double energy{0}, force_scale{0};
  bool extrapolated{false};
};
// Property data resolved once per solve. Remanence is constant for a basic or
// rectangular frame; curvilinear frames are evaluated at each point.
struct MaterialData {
  Material material;
  const std::vector<std::pair<double, double>> *curve{nullptr};
  const CoordinateSystem *coordinates{nullptr};
  int axis{1};
  Vector remanence{Vector::Zero()};
};
using Materials = std::map<int, MaterialData>; // by property ID
Materials resolve_materials(const Model &model) {
  static const std::vector<std::pair<double, double>> empty;
  Materials out;
  for (const auto &[id, p] : model.properties) {
    MaterialData d;
    d.material = model.materials.at(p.material);
    d.curve = d.material.bh_table ? &model.bh_tables.at(d.material.bh_table)
                                  : &empty;
    d.axis = p.axis;
    if (d.material.br != 0) {
      const auto *cs =
          p.coordinates ? &model.coordinates.at(p.coordinates) : nullptr;
      if (cs && cs->type != 'R')
        d.coordinates = cs;
      else
        d.remanence = (p.axis > 0 ? 1 : -1) * d.material.br *
                      (cs ? coordinate_axes(*cs) : Matrix::Identity())
                          .col(std::abs(p.axis) - 1);
    }
    out.emplace(id, d);
  }
  return out;
}
// Remanence in basic at x.
Vector remanence_at(const MaterialData &d, const Vector &x) {
  if (!d.coordinates)
    return d.remanence;
  return (d.axis > 0 ? 1 : -1) * d.material.br *
         local_axes(*d.coordinates, x).col(std::abs(d.axis) - 1);
}
Response response_at(const MaterialData &d, const Vector &x, const Vector &b) {
  return material_response(d.material, *d.curve, b, remanence_at(d, x));
}
// The air-gap element's dense block over its grids, scaled by the depth.
// It is linear, so it is the same in every tangent.
struct GapCoupling {
  std::vector<int> nodes; // the unknowns it couples in increasing order (the
                          // circles' grids, a band's interior points), which
                          // turning a part does not change
  // Over nodes: every entry for the AGE (a dense block), a sliding band's
  // nonzeros, which move as a part turns.
  Eigen::SparseMatrix<double> stiffness;
};
GapCoupling gap_coupling(const Mesh &mesh) {
  GapCoupling out;
  if (!mesh.air_gap)
    return out;
  const auto &gap = *mesh.air_gap;
  const Eigen::SparseMatrix<double> unfolded =
      mesh.depth * (gap.band_points > 0 ? band_stiffness(gap)
                                        : Eigen::MatrixXd(air_gap_stiffness(gap)).sparseView(0, 0));
  // Each entry is sign times an unknown: a circle grid's potential, or a
  // band's interior point. In a sector model the unfolded gap is the whole
  // machine's, and the sector's share of its energy is 1/sectors.
  const auto entries = gap_entries(gap); // unknown, sign
  for (auto [node, sign] : entries)
    out.nodes.push_back(node);
  std::ranges::sort(out.nodes);
  out.nodes.erase(std::unique(out.nodes.begin(), out.nodes.end()), out.nodes.end());
  std::vector<Eigen::Index> slot; // of each entry's grid
  for (auto [node, sign] : entries)
    slot.push_back(std::ranges::lower_bound(out.nodes, node) - out.nodes.begin());
  const auto n = static_cast<Eigen::Index>(out.nodes.size());
  std::vector<Eigen::Triplet<double>> t;
  t.reserve(static_cast<size_t>(unfolded.nonZeros()));
  for (Eigen::Index j = 0; j < unfolded.outerSize(); ++j)
    for (Eigen::SparseMatrix<double>::InnerIterator it(unfolded, j); it; ++it) {
      const auto i = static_cast<size_t>(it.row()), k = static_cast<size_t>(j);
      t.emplace_back(slot[i], slot[k],
                     entries[i].second * entries[k].second * it.value() / gap.sectors);
    }
  out.stiffness.resize(n, n);
  out.stiffness.setFromTriplets(t.begin(), t.end());
  return out;
}
bool on_air_gap(const Mesh &mesh, int face) {
  return mesh.air_gap && std::find(mesh.air_gap->faces.begin(),
                                   mesh.air_gap->faces.end(),
                                   face) != mesh.air_gap->faces.end();
}
// The gauged stiffness's sparsity: the cells' couplings and the air gap's
// (the AGE's dense block, a band's local couplings). Where each local entry
// lands in it is found once, so an assembly only adds values into place. It
// depends only on the topology, so MOTION positions share it while the gap's
// couplings stay put.
struct Pattern {
  Sparse matrix;           // the structure, with zero values
  std::vector<size_t> local, functions; // per cell (and one past): first entry
                                        // of its ne x ne block, of its ne
                                        // residuals
  std::vector<int> slots;  // value index of each local entry; -1 prescribed
  std::vector<int> gap;    // of each nonzero of the gap's stiffness, in its
                           // storage order
};
Pattern make_pattern(const Mesh &mesh, const Gauge &gauge, const GapCoupling &gap) {
  Pattern p;
  std::vector<Triplet> entries;
  p.local.reserve(mesh.cells.size() + 1);
  p.functions.reserve(mesh.cells.size() + 1);
  p.local.push_back(0);
  p.functions.push_back(0);
  const auto couple = [&](int i, int j) {
    const int row = gauge.equations[i], col = gauge.equations[j];
    if (row >= 0 && col >= 0)
      entries.emplace_back(row, col, 0.0);
  };
  for (const auto &cell : mesh.cells) {
    p.local.push_back(p.local.back() + cell.edges.size() * cell.edges.size());
    p.functions.push_back(p.functions.back() + cell.edges.size());
    for (int i : cell.edges)
      for (int j : cell.edges)
        couple(i, j);
  }
  const auto &gs = gap.stiffness;
  for (Eigen::Index j = 0; j < gs.outerSize(); ++j)
    for (Eigen::SparseMatrix<double>::InnerIterator it(gs, j); it; ++it)
      couple(gap.nodes[static_cast<size_t>(it.row())], gap.nodes[static_cast<size_t>(j)]);
  p.matrix.resize(gauge.size, gauge.size);
  p.matrix.setFromTriplets(entries.begin(), entries.end());
  const auto slot = [&](int i, int j) {
    const int row = gauge.equations[i], col = gauge.equations[j];
    if (row < 0 || col < 0)
      return -1;
    const int *rows = p.matrix.innerIndexPtr();
    const int *first = rows + p.matrix.outerIndexPtr()[col];
    const int *last = rows + p.matrix.outerIndexPtr()[col + 1];
    return static_cast<int>(std::lower_bound(first, last, row) - rows);
  };
  p.slots.reserve(p.local.back());
  for (const auto &cell : mesh.cells)
    for (int i : cell.edges)
      for (int j : cell.edges)
        p.slots.push_back(slot(i, j));
  for (Eigen::Index j = 0; j < gs.outerSize(); ++j)
    for (Eigen::SparseMatrix<double>::InnerIterator it(gs, j); it; ++it)
      p.gap.push_back(
          slot(gap.nodes[static_cast<size_t>(it.row())], gap.nodes[static_cast<size_t>(j)]));
  return p;
}
// What solving a model needs from its geometry: built once for the model as
// given, and again for each MOTION position's moved model.
struct Setup {
  Mesh mesh;
  CellGeometry geometry;
  Materials materials;
  Gauge gauge;
  GapCoupling gap;
  std::shared_ptr<const Pattern> pattern;
  // The gap block's entries of the gauged stiffness, aligned with
  // pattern->gap (zero where prescribed).
  std::vector<double> gap_values;
  std::vector<Source> sources;
  std::vector<CircuitDefinition> circuits;
  // Circuit coupling g_c = sum P_i c_i / a_c, so circuit linkage is g_c . A
  // and circuit inductance needs one tangent solve per closed circuit.
  std::vector<Eigen::VectorXd> circuit_coupling;
  std::optional<SpatialIndex> index;
  // The MBC whose NORMAL faces the gauge leaves free (0 for none), and
  // those faces.
  int normal_boundary{0};
  std::vector<bool> natural;
};
// Calls work(c) for each cell, in parallel when TBB is available.
template <class Work> void for_each_cell(size_t count, const Work &work) {
#ifdef HAVE_TBB
  tbb::parallel_for(tbb::blocked_range<size_t>(0, count, 256),
                    [&](const tbb::blocked_range<size_t> &range) {
                      for (size_t c = range.begin(); c != range.end(); ++c)
                        work(c);
                    });
#else
  for (size_t c = 0; c < count; ++c)
    work(c);
#endif
}
// frozen assembles the secant (frozen-permeability) stiffness in place of
// the tangent; the residual and fields are the same. The cells are
// integrated in parallel, each into its own entries, and summed into the
// system in cell order, so the result does not depend on the thread count.
Assembly assemble(const Setup &setup, const Eigen::VectorXd &a,
                  const Eigen::VectorXd &load, bool frozen = false) {
  const Mesh &mesh = setup.mesh;
  const Gauge &gauge = setup.gauge;
  const Pattern &pattern = *setup.pattern;
  const size_t count = mesh.cells.size();
  std::vector<double> local(pattern.local.back(), 0.0),
      residual(pattern.functions.back(), 0.0), energy(count, 0.0);
  std::vector<char> extrapolated(count, 0);
  Assembly out;
  out.b.resize(count);
  out.h.resize(count);
  for_each_cell(count, [&](size_t index) {
    const int c = static_cast<int>(index);
    const Cell &cell = mesh.cells[index];
    const auto ne = static_cast<int>(cell.edges.size());
    const MaterialData &material = setup.materials.at(cell.property);
    double *k = local.data() + pattern.local[index];
    double *r = residual.data() + pattern.functions[index];
    std::array<Vector, max_cell_functions> curl, applied;
    Vector average_b = Vector::Zero(), average_h = Vector::Zero();
    for (int q = 0; q < setup.geometry.points(c); ++q) {
      setup.geometry.curls(mesh, c, q, curl);
      const double w = setup.geometry.weight(c, q);
      Vector b = Vector::Zero();
      for (int e = 0; e < ne; ++e)
        b += a[cell.edges[e]] * curl[e];
      // Only a curvilinear remanence frame needs the point's position.
      const auto response = response_at(
          material,
          material.coordinates ? setup.geometry.position(mesh, c, q) : cell.origin,
          b);
      average_b += w * b;
      average_h += w * response.h;
      const double weight = w * setup.geometry.measure(c, q);
      energy[index] += weight * response.energy;
      extrapolated[index] |= static_cast<char>(response.extrapolated);
      const Matrix &stiffness = frozen ? response.secant : response.tangent;
      for (int j = 0; j < ne; ++j)
        applied[j] = stiffness * curl[j];
      for (int i = 0; i < ne; ++i) {
        r[i] += weight * curl[i].dot(response.h);
        for (int j = 0; j < ne; ++j)
          k[i * ne + j] += weight * curl[i].dot(applied[j]);
      }
    }
    out.b[index] = average_b / cell.volume;
    out.h[index] = average_h / cell.volume;
  });
  out.residual = Eigen::VectorXd::Zero(gauge.size);
  out.stiffness = pattern.matrix;
  double *values = out.stiffness.valuePtr();
  for (size_t c = 0; c < count; ++c) {
    const Cell &cell = mesh.cells[c];
    const size_t ne = cell.edges.size();
    out.energy += energy[c];
    out.extrapolated |= extrapolated[c] != 0;
    const double *k = local.data() + pattern.local[c];
    const int *slots = pattern.slots.data() + pattern.local[c];
    const double *r = residual.data() + pattern.functions[c];
    for (size_t i = 0; i < ne; ++i) {
      const int row = gauge.equations[cell.edges[i]];
      if (row < 0)
        continue;
      const double si = gauge.sign[cell.edges[i]];
      out.residual[row] += si * r[i];
      out.force_scale += r[i] * r[i];
      for (size_t j = 0; j < ne; ++j)
        if (slots[i * ne + j] >= 0)
          values[slots[i * ne + j]] +=
              si * gauge.sign[cell.edges[j]] * k[i * ne + j];
    }
  }
  const GapCoupling &gap = setup.gap;
  if (!gap.nodes.empty()) {
    const auto n = static_cast<Eigen::Index>(gap.nodes.size());
    Eigen::VectorXd potentials(n);
    for (Eigen::Index i = 0; i < n; ++i)
      potentials[i] = a[gap.nodes[i]];
    const Eigen::VectorXd gap_residual = gap.stiffness * potentials;
    out.energy += .5 * potentials.dot(gap_residual);
    for (Eigen::Index i = 0; i < n; ++i) {
      const int row = gauge.equations[gap.nodes[i]];
      if (row < 0)
        continue;
      out.residual[row] += gauge.sign[gap.nodes[i]] * gap_residual[i];
      out.force_scale += gap_residual[i] * gap_residual[i];
    }
    for (size_t k = 0; k < pattern.gap.size(); ++k)
      if (pattern.gap[k] >= 0)
        values[pattern.gap[k]] += setup.gap_values[k];
  }
  out.residual -= gauge.reduce(load);
  return out;
}
std::string face_grids(const Mesh &mesh, const Face &f) {
  std::string out;
  for (int n : f.nodes)
    out += (out.empty() ? "" : ",") + std::to_string(mesh.node_ids[n]);
  return out;
}
// The faces of the planar exterior that MBC boundary's NORMAL rows name,
// where the flux crosses normally (n x H = 0, the natural condition).
Result<std::vector<bool>> natural_faces(const Model &model, const Mesh &mesh,
                                        int boundary) {
  std::vector<bool> out(mesh.faces.size(), false);
  if (!boundary)
    return out;
  for (int set : model.boundaries.at(boundary).normal) {
    auto faces = selected_faces(model, mesh, set);
    if (auto *e = std::get_if<Error>(&faces))
      return Error{std::format("MBC {} NORMAL: {}", boundary, e->message)};
    for (auto [f, sign] : std::get<std::vector<std::pair<int, int>>>(faces)) {
      (void)sign;
      const auto &face = mesh.faces[f];
      const auto where = std::format("MBC {} NORMAL FSET {} selects face ({}) of "
                                     "element {}",
                                     boundary, set, face_grids(mesh, face),
                                     mesh.cells[face.cells[0]].id);
      if (face.cells[1] >= 0)
        return Error{where + ", which is inside the field mesh"};
      if (on_air_gap(mesh, static_cast<int>(f)))
        return Error{where + " on the air gap, which takes no boundary condition"};
      if (on_axis(mesh, face))
        return Error{where + " on the symmetry axis, where A_phi = 0"};
      if (mesh.period && std::ranges::find(mesh.period->faces, f) !=
                             mesh.period->faces.end())
        return Error{std::format("{} on a periodic side of MPERIOD {}", where,
                                 mesh.period->id)};
      if (out[f])
        return Error{where + " twice"};
      out[f] = true;
    }
  }
  return out;
}
// The prescribed potential of the subcase's MBC row on the exterior faces
// that are neither natural, periodic nor on the air gap (the potential is
// free elsewhere; only the gauge's prescribed values are used).
Result<Eigen::VectorXd> boundary_potential(const Model &model, const Mesh &mesh,
                                           const Gauge &gauge,
                                           const std::vector<bool> &natural,
                                           const Subcase &subcase) {
  Eigen::VectorXd a = Eigen::VectorXd::Zero(mesh.dofs());
  if (!subcase.boundary)
    return a;
  const auto &bc = model.boundaries.at(subcase.boundary);
  if (bc.faces) {
    auto faces = selected_faces(model, mesh, bc.faces);
    if (auto *e = std::get_if<Error>(&faces))
      return *e;
    // The air gap is part of the domain, not its exterior; periodic faces
    // meet their images, and NORMAL faces are free.
    std::vector<bool> open(mesh.faces.size(), false);
    if (mesh.air_gap)
      for (int f : mesh.air_gap->faces)
        open[f] = true;
    if (mesh.period)
      for (int f : mesh.period->faces)
        open[f] = true;
    for (size_t f = 0; f < natural.size(); ++f)
      if (natural[f])
        open[f] = true;
    // The axis takes A_phi = 0 from the formulation, not from an MBC.
    for (size_t f = 0; f < mesh.faces.size(); ++f)
      if (on_axis(mesh, mesh.faces[f]))
        open[f] = true;
    size_t count = 0;
    for (size_t f = 0; f < mesh.faces.size(); ++f)
      if (mesh.faces[f].cells[1] < 0 && !open[f])
        ++count;
    const auto &selected = std::get<std::vector<std::pair<int, int>>>(faces);
    for (auto [f, sign] : selected) {
      (void)sign;
      if (mesh.faces[f].cells[1] >= 0)
        return Error{"MBC must select exterior faces"};
      if (open[f])
        return Error{std::format(
            "MBC FSET selects face ({}) of element {} on the {}",
            face_grids(mesh, mesh.faces[f]), mesh.cells[mesh.faces[f].cells[0]].id,
            on_air_gap(mesh, f)
                ? "air gap; the air gap is coupled by the air-gap element and "
                  "takes no boundary condition"
            : on_axis(mesh, mesh.faces[f])
                ? "symmetry axis, where A_phi = 0 without an MBC"
            : static_cast<size_t>(f) < natural.size() && natural[f]
                ? "flux-normal (NORMAL) boundary, which takes no prescribed "
                  "potential"
                : "periodic boundary, which meets its image instead of taking "
                  "a prescribed potential")};
    }
    if (selected.size() != count)
      return Error{"MBC FSET must cover the entire magnetic exterior that is not "
                   "on the air gap, periodic, NORMAL or the symmetry axis (or "
                   "use FSET=0)"};
  }
  if (mesh.planar) {
    if (mesh.axisymmetric) {
      // A uniform axial field B0: A_phi = B0 r / 2.
      if ((bc.field - mesh.axis * mesh.axis.dot(bc.field)).norm() >
          1e-12 * bc.field.norm())
        return Error{std::format("MAGAXI MBC {} field must be along the symmetry "
                                 "axis; a radial or tangential uniform field is "
                                 "not axisymmetric",
                                 subcase.boundary)};
      for (size_t i = 0; i < mesh.nodes.size(); ++i)
        a[static_cast<Eigen::Index>(i)] = mesh.axis.dot(bc.field) / 2 *
                                          (mesh.nodes[i] - mesh.axis_origin).dot(mesh.radial);
    } else {
      if (std::abs(bc.field.dot(mesh.normal)) > 1e-12 * bc.field.norm())
        return Error{"MAG2D MBC field must lie in the simulation plane"};
      for (size_t i = 0; i < mesh.nodes.size(); ++i)
        a[i] = bc.field.cross(mesh.nodes[i] - mesh.nodes.front()).dot(mesh.normal);
    }
    if (const auto &period = mesh.period) {
      // A prescribed grid and its prescribed image must agree with the
      // period: a uniform field is periodic only in special cases.
      double scale = 0;
      for (size_t i = 0; i < mesh.nodes.size(); ++i)
        if (gauge.equations[i] < 0)
          scale = std::max(scale, std::abs(a[static_cast<Eigen::Index>(i)]));
      auto pairs = period->pairs;
      std::ranges::transform(period->axial, std::back_inserter(pairs),
                             [](int v) { return std::array<int, 2>{v, v}; });
      for (auto [first, second] : pairs)
        if (gauge.equations[first] < 0 &&
            std::abs(a[second] - period->sign() * a[first]) > 1e-9 * scale)
          return Error{std::format(
              "MBC {} UNIFORM prescribes A = {:.6e} at grid {} and {:.6e} at "
              "its {} image grid {}, which MPERIOD {} requires to be {} it",
              subcase.boundary, a[first], mesh.node_ids[first], a[second],
              period->anti ? "anti-periodic" : "periodic", mesh.node_ids[second],
              period->id, period->anti ? "minus" : "equal to")};
    }
    return a;
  }
  for (size_t e = 0; e < mesh.dofs(); ++e) {
    const Vector &x = mesh.nodes[mesh.edges[e][0]];
    const Vector &y = mesh.nodes[mesh.edges[e][1]];
    a[e] = 0.5 * bc.field.cross((x + y) / 2).dot(y - x);
  }
  return a;
}
// The loosest relative residual of a Newton step's linear solve with a
// reused factor.
constexpr double kMaxForcing = 0.1;
struct Equilibrium {
  Eigen::VectorXd a;
  Assembly assembly;
  int iterations{0};
  double residual{0};
};
struct ImplicitStep {
  const Diffusion &diffusion;
  const Sparse &tangent;
  const Eigen::VectorXd &previous;
  double dt;
};
// Eliminating the voltage-driven currents adds G C^-1 G^T to the
// tangent. Solve that low-rank update without assembling a dense winding
// block. C is (dt R + Lend) divided by the number of modeled periods.
struct VoltageStep {
  Eigen::MatrixXd full, reduced;
  Eigen::VectorXd impedance, history;

  [[nodiscard]] Eigen::VectorXd currents(const Eigen::VectorXd &a) const {
    return (history - full.transpose() * a).cwiseQuotient(impedance);
  }

  [[nodiscard]] Result<Eigen::VectorXd> solve(Tangent &tangent,
                                               const Eigen::VectorXd &rhs) const {
    const auto fields = reduced.rows(), count = reduced.cols();
    Eigen::MatrixXd columns = Eigen::MatrixXd::Zero(rhs.size(), 1 + count);
    columns.col(0) = rhs;
    columns.topRightCorner(fields, count) = reduced;
    auto solved = tangent.solve(columns, 0);
    if (const auto *e = std::get_if<Error>(&solved))
      return *e;
    const auto &x = std::get<Eigen::MatrixXd>(solved);
    const auto responses = x.rightCols(count);
    Eigen::MatrixXd schur = reduced.transpose() * responses.topRows(fields);
    schur.diagonal() += impedance;
    const Eigen::LLT<Eigen::MatrixXd> factor(schur);
    if (factor.info() != Eigen::Success)
      return Error{"MAGTRANS voltage circuit Schur complement is not positive definite"};
    Eigen::VectorXd out = x.col(0) - responses *
        factor.solve(reduced.transpose() * x.col(0).head(fields));
    if (!out.allFinite())
      return Error{"MAGTRANS voltage circuit update produced nonfinite values"};
    return out;
  }
};
Result<Equilibrium> equilibrium(const Model &model, const Setup &setup,
                        Tangent &tangent, Eigen::VectorXd a,
                        const Eigen::VectorXd &load,
                        const ImplicitStep *implicit = nullptr,
                        const VoltageStep *voltage = nullptr) {
  const Mesh &mesh = setup.mesh;
  const Gauge &gauge = setup.gauge;
  const auto evaluate = [&](const Eigen::VectorXd &potential) {
    auto state = assemble(setup, potential, load);
    if (implicit) {
      const auto delta = implicit->diffusion.increment(potential, implicit->previous);
      const auto eddy = implicit->diffusion.residual(delta, gauge, implicit->dt);
      state.residual.conservativeResize(eddy.size());
      state.residual.tail(eddy.size() - gauge.size).setZero();
      state.residual += eddy;
      state.stiffness.conservativeResize(eddy.size(), eddy.size());
      state.stiffness += implicit->tangent;
    }
    if (voltage)
      state.residual.head(gauge.size).noalias() -= voltage->reduced * voltage->currents(potential);
    return state;
  };
  auto state = evaluate(a);
  double load_norm = 0;
  for (size_t e = 0; e < mesh.dofs(); ++e)
    if (gauge.equations[e] >= 0)
      load_norm += load[e] * load[e];
  const double reference =
      std::max({std::sqrt(load_norm), std::sqrt(state.force_scale),
                state.residual.norm(), 1e-30});
  const double target = model.options.nonlinear_tolerance * reference;
  // With a reused factor each step is inexact (Eisenstat and Walker's second
  // forcing term): solved to a relative residual eta, loose while Newton is
  // far from converging and tightening with the residual's decrease, so it
  // keeps Newton superlinear. It is never tighter than half the remaining
  // distance to the target, which no step needs to beat.
  double eta = kMaxForcing, previous = 0;
  int iteration = 0;
  for (; iteration <= model.options.max_iterations; ++iteration) {
    const double norm = state.residual.norm();
    if (norm / reference <= model.options.nonlinear_tolerance ||
        state.residual.size() == 0)
      return Equilibrium{std::move(a), std::move(state), iteration,
                         norm / reference};
    if (iteration == model.options.max_iterations)
      break;
    if (iteration > 0) {
      const double ratio = norm / previous;
      const double safeguard = 0.9 * eta * eta;
      eta = 0.9 * ratio * ratio;
      if (safeguard > 0.1)
        eta = std::max(eta, safeguard);
      eta = std::min(eta, kMaxForcing);
    }
    eta = std::max(eta, 0.5 * target / norm);
    previous = norm;
    if (auto e = tangent.update(state.stiffness))
      return Error{std::format("Magnetic tangent factorization failed ({}); "
                               "check mesh, gauge and material data", e->message)};
    const auto direction = [&]() -> Result<Eigen::VectorXd> {
      if (voltage)
        return voltage->solve(tangent, -state.residual);
      auto solved = tangent.solve(-state.residual, tangent.reuses() ? eta : 0);
      if (const auto *e = std::get_if<Error>(&solved))
        return *e;
      return Eigen::VectorXd(std::get<Eigen::MatrixXd>(std::move(solved)));
    }();
    if (const auto *e = std::get_if<Error>(&direction))
      return Error{std::format("Magnetic linear solve failed ({})", e->message)};
    const auto &step = std::get<Eigen::VectorXd>(direction);
    bool accepted = false;
    double alpha = 1;
    for (int trial = 0; trial < 30; ++trial) {
      Eigen::VectorXd candidate = a;
      gauge.expand(step, candidate, alpha);
      auto next = evaluate(candidate);
      if (next.residual.allFinite() &&
          next.residual.norm() <= (1 - 1e-4 * alpha) * norm) {
        a = std::move(candidate);
        state = std::move(next);
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted)
      return Error{std::format("Magnetic Newton line search failed at "
                               "iteration {} (relative residual {})",
                               iteration, norm / reference)};
  }
  return Error{std::format("Magnetic Newton did not converge in {} iterations "
                           "(relative residual {})",
                           iteration, state.residual.norm() / reference)};
}
Vector field_at(const Cell &c, const Eigen::VectorXd &a, const Vector &x) {
  const auto curls = edge_curls(c, x);
  Vector b = Vector::Zero();
  for (size_t i = 0; i < curls.size(); ++i)
    b += a[c.edges[i]] * curls[i];
  return b;
}
// Field cells inside a COIL or COIL2D winding, mapped to their source ID.
std::map<int, int> winding_cells(const Model &model, const Mesh &mesh) {
  std::map<int, int> out;
  for (const auto &c : model.coils)
    for (int id : model.element_sets.at(c.elements))
      out[mesh.element_index.at(id)] = c.id;
  for (const auto &c : model.planar_coils)
    for (int id : model.element_sets.at(c.elements))
      out[mesh.element_index.at(id)] = c.id;
  return out;
}
// Positive-area overlap of a flat face of b with the flat face fa, when they
// are coplanar. Both faces are convex polygons.
bool coplanar_overlap(const Mesh &a, const Face &fa, const Mesh &b,
                      const Face &fb) {
  const double length = std::sqrt(fa.area);
  if (!fa.flat || !fb.flat || std::abs(fa.normal.dot(fb.normal)) < 1 - 1e-12)
    return false;
  std::vector<Vector> polygon;
  for (int n : fb.nodes) {
    if (std::abs(fa.normal.dot(b.nodes[n] - fa.center)) > 1e-10 * length)
      return false;
    polygon.push_back(b.nodes[n]);
  }
  for (size_t i = 0; i < fa.nodes.size() && !polygon.empty(); ++i) {
    const Vector x0 = a.nodes[fa.nodes[i]],
                 x1 = a.nodes[fa.nodes[(i + 1) % fa.nodes.size()]];
    Vector inward = fa.normal.cross(x1 - x0);
    if (inward.dot(fa.center - x0) < 0)
      inward = -inward;
    std::vector<Vector> clipped;
    for (size_t k = 0; k < polygon.size(); ++k) {
      const Vector &p = polygon[k], &q = polygon[(k + 1) % polygon.size()];
      const double dp = inward.dot(p - x0), dq = inward.dot(q - x0);
      if (dp >= 0)
        clipped.push_back(p);
      if ((dp < 0) != (dq < 0))
        clipped.push_back(p + dp / (dp - dq) * (q - p));
    }
    polygon = std::move(clipped);
  }
  Vector area = Vector::Zero();
  for (size_t k = 2; k < polygon.size(); ++k)
    area += (polygon[k - 1] - polygon[0]).cross(polygon[k] - polygon[0]) / 2;
  return area.norm() > 1e-10 * std::min(fa.area, fb.area);
}
// Maxwell stress gives the force on everything a surface encloses, so a face
// may bound a winding but must not cut through current-carrying volume.
Status current_free(const Mesh &mesh, const Face &f, int request,
                    const std::map<int, int> &windings,
                    std::span<const Source> sources,
                    std::span<const SpatialIndex> wire_indexes) {
  const auto first = windings.find(f.cells[0]),
             second = windings.find(f.cells[1]);
  if (first != windings.end() && second != windings.end())
    return Error{std::format(
        "MFORCE {}: surface face ({}) of element {} lies inside current "
        "source {}; move the surface clear of the winding or enclose it",
        request, face_grids(mesh, f), mesh.cells[f.cells[0]].id,
        first->second)};
  if (mesh.planar)
    return {};
  Vector lo = mesh.nodes[f.nodes[0]], hi = lo;
  for (int n : f.nodes) {
    lo = lo.cwiseMin(mesh.nodes[n]);
    hi = hi.cwiseMax(mesh.nodes[n]);
  }
  size_t wire = 0;
  for (const auto &source : sources) {
    if (!source.wire)
      continue;
    for (int ci : wire_indexes[wire].query(lo, hi)) {
      auto crossing = face_intersects_interior(
          source.mesh, source.mesh.cells[ci], mesh, f);
      if (auto *e = std::get_if<Error>(&crossing))
        return *e;
      // The conductor partition places internal faces on its centerline, so
      // a surface through the conductor can lie on them without entering any
      // single cell.
      const auto &cell = source.mesh.cells[ci];
      const bool on_internal_face = std::any_of(
          cell.faces.begin(), cell.faces.end(), [&](int g) {
            const auto &face = source.mesh.faces[g];
            return face.cells[1] >= 0 &&
                   coplanar_overlap(mesh, f, source.mesh, face);
          });
      if (std::get<bool>(crossing) || on_internal_face)
        return Error{std::format(
            "MFORCE {}: surface face ({}) of element {} crosses CWIRE path "
            "{}; move the surface clear of the conductor",
            request, face_grids(mesh, f), mesh.cells[f.cells[0]].id,
            source.id)};
    }
    ++wire;
  }
  return {};
}
bool magnetic(const Material &m) {
  return m.mur != 1 || m.bh_table != 0 || m.br != 0;
}
// The cell across face f on a periodic side of a translation, in the
// neighbouring pitch: the cell on f's image (f on the first side) or
// preimage (on the second). first says which.
struct Across {
  int cell{-1};
  bool first{true};
};
std::optional<Across> across_period(const Mesh &mesh, int f) {
  if (!mesh.period)
    return std::nullopt;
  for (auto [a, b] : mesh.period->face_pairs) {
    if (a == f)
      return Across{mesh.faces[b].cells[0], true};
    if (b == f)
      return Across{mesh.faces[a].cells[0], false};
  }
  return std::nullopt;
}
// B at x on a periodic side's face, from the cell across it: that cell is
// the image (or preimage) of the copy across the side, whose potential is
// the period's sign times its own.
Vector field_across(const Mesh &mesh, const Across &across,
                    const Eigen::VectorXd &potential, const Vector &x) {
  const auto &period = *mesh.period;
  const auto &cell = mesh.cells[across.cell];
  return period.sign() *
         field_at(cell, potential, across.first ? period.image(x) : period.preimage(x));
}
// A MFORCE BODY surface: every face with exactly one side in R, the body
// plus `layers` grid-adjacent layers of nonmagnetic, current-free cells,
// oriented out of R. Only the selected body may contain current sources.
// It includes cavity walls, so an annular body excludes what it surrounds.
// component[k] is the connected piece of R that face k bounds. In a model
// that repeats along a line (MPERIOD with SHIFT) R continues across the
// periodic sides into the neighbouring pitches (grids are adjacent to their
// images), and a face on a side bounds R when the cell across it,
// across[k], is outside R; the surface is then the unfolded body's, one
// pitch of it. In a sector model the copies across the sides are turned, so
// a surface that crossed them would sum tractions of pieces of different
// bodies: it must close within the sector.
struct BodySurface {
  std::vector<std::pair<int, int>> faces;
  std::vector<int> component;
  std::vector<std::optional<Across>> across;
  int components{0};
};
Result<BodySurface> body_surface(const Model &model, const Mesh &mesh,
                                 const std::vector<std::vector<int>> &incident,
                                 const ForceSurface &request,
                                 const std::map<int, int> &windings,
                                 std::span<const Source> sources,
                                 std::span<const SpatialIndex> wire_indexes) {
  const int nc = static_cast<int>(mesh.cells.size());
  std::vector<int> layer(nc, -1); // 0 for the body, k for layer k
  std::vector<int> frontier;
  for (int id : model.element_sets.at(request.set)) {
    const int ci = mesh.element_index.at(id);
    if (layer[ci] < 0)
      frontier.push_back(ci);
    layer[ci] = 0;
  }
  auto is_magnetic = [&](int ci) {
    return magnetic(model.materials.at(mesh.cells[ci].material));
  };
  for (int k = 1; k <= request.layers; ++k) {
    std::vector<int> next;
    for (int ci : frontier)
      for (int n : mesh.cells[ci].nodes)
        for (int neighbor : incident[n])
          if (layer[neighbor] < 0) {
            if (is_magnetic(neighbor))
              return Error{
                  k == 1 ? std::format(
                               "MFORCE {}: magnetic element {} touches body "
                               "ESET {}; add it to the body set or mesh a "
                               "nonmagnetic gap between them",
                               request.id, mesh.cells[neighbor].id, request.set)
                         : std::format(
                               "MFORCE {}: magnetic element {} lies within {} "
                               "layers of body ESET {}; use a smaller LAYER or "
                               "more air layers",
                               request.id, mesh.cells[neighbor].id, k,
                               request.set)};
            const auto &cell = mesh.cells[neighbor];
            if (auto it = windings.find(neighbor); it != windings.end())
              return Error{std::format(
                  "MFORCE {}: layer {} of body ESET {} includes element {} "
                  "of current source {}; add the winding to the body set "
                  "or use a smaller LAYER or more air layers",
                  request.id, k, request.set, cell.id, it->second)};
            size_t wire = 0;
            for (const auto &source : sources) {
              if (!source.wire)
                continue;
              for (int ci : wire_indexes[wire].query(cell.minimum, cell.maximum)) {
                const auto &conductor = source.mesh.cells[ci];
                Result<bool> overlap = false;
                if (cell.polyhedral) {
                  const auto samples = intersection_quadrature(
                      source.mesh, conductor, mesh, cell);
                  const double volume = std::accumulate(
                      samples.begin(), samples.end(), 0.0,
                      [](double sum, const Sample &q) { return sum + q.weight; });
                  overlap = volume > 1e-10 * conductor.volume;
                } else
                  overlap = cell_intersects_interior(source.mesh, conductor,
                                                     cell);
                if (auto *e = std::get_if<Error>(&overlap))
                  return Error{std::format(
                      "MFORCE {}: checking layer {} of body ESET {}, element "
                      "{}, against CWIRE path {}: {}",
                      request.id, k, request.set, cell.id, source.id,
                      e->message)};
                if (std::get<bool>(overlap))
                  return Error{std::format(
                      "MFORCE {}: layer {} of body ESET {} includes element "
                      "{} containing CWIRE path {}; add all field cells "
                      "containing the conductor to the body set or use a "
                      "smaller LAYER or more air layers",
                      request.id, k, request.set, cell.id, source.id)};
              }
              ++wire;
            }
            layer[neighbor] = k;
            next.push_back(neighbor);
          }
    frontier = std::move(next);
  }
  // Connected pieces of R through shared grids.
  std::vector<int> piece(nc, -1);
  BodySurface out;
  for (int seed = 0; seed < nc; ++seed) {
    if (layer[seed] < 0 || piece[seed] >= 0)
      continue;
    std::vector<int> stack{seed};
    piece[seed] = out.components;
    while (!stack.empty()) {
      const int ci = stack.back();
      stack.pop_back();
      for (int n : mesh.cells[ci].nodes)
        for (int neighbor : incident[n])
          if (layer[neighbor] >= 0 && piece[neighbor] < 0) {
            piece[neighbor] = out.components;
            stack.push_back(neighbor);
          }
    }
    ++out.components;
  }
  for (int ci = 0; ci < nc; ++ci) {
    if (layer[ci] < 0)
      continue;
    const auto &cell = mesh.cells[ci];
    for (size_t k = 0; k < cell.faces.size(); ++k) {
      const auto &f = mesh.faces[cell.faces[k]];
      const int other = f.cells[0] == ci ? f.cells[1] : f.cells[0];
      // A surface of revolution closes on the axis: its faces there take no
      // stress (r = 0).
      if (other < 0 && on_axis(mesh, f))
        continue;
      if (other < 0 && on_air_gap(mesh, cell.faces[k]))
        return Error{std::format(
            "MFORCE {}: the surface around body ESET {} reaches the air gap at "
            "element {}; force surfaces may not touch the air gap",
            request.id, request.set, cell.id)};
      if (other < 0 && mesh.period && !mesh.period->translation &&
          std::ranges::find(mesh.period->faces, cell.faces[k]) != mesh.period->faces.end())
        return Error{std::format(
            "MFORCE {}: the surface around body ESET {} reaches a periodic side "
            "of MPERIOD {} at element {}; in a sector model a force surface must "
            "close within the sector, since across the sides it would hold parts "
            "of turned copies. For the machine's torque use the air-gap "
            "element's, which is the whole machine's",
            request.id, request.set, mesh.period->id, cell.id)};
      if (other < 0)
        if (const auto across = across_period(mesh, cell.faces[k])) {
          if (layer[across->cell] >= 0)
            continue;
          if (is_magnetic(across->cell))
            return Error{std::format(
                "MFORCE {}: magnetic element {}, across periodic side of MPERIOD "
                "{}, lies within {} layers of body ESET {}; use a smaller LAYER "
                "or more air layers",
                request.id, mesh.cells[across->cell].id, mesh.period->id,
                request.layers + 1, request.set)};
          out.faces.emplace_back(cell.faces[k], cell.face_sign[k]);
          out.component.push_back(piece[ci]);
          out.across.push_back(across);
          continue;
        }
      if (other < 0)
        return Error{std::format(
            "MFORCE {}: body ESET {} is within {} cells of the domain "
            "boundary at element {}; request the force on the opposite body",
            request.id, request.set, request.layers, cell.id)};
      if (layer[other] >= 0)
        continue;
      if (is_magnetic(other))
        return Error{std::format(
            "MFORCE {}: magnetic element {} lies within {} layers of body "
            "ESET {}; use a smaller LAYER or more air layers",
            request.id, mesh.cells[other].id, request.layers + 1,
            request.set)};
      out.faces.emplace_back(cell.faces[k], cell.face_sign[k]);
      out.component.push_back(piece[ci]);
      out.across.emplace_back();
    }
  }
  return out;
}
// Validates an explicit FACES surface: closed, consistently and outwardly
// oriented, strictly inside the domain.
Status explicit_surface(const Mesh &mesh,
                        std::span<const std::pair<int, int>> selected) {
  std::map<std::array<int, 2>, std::pair<int, int>> boundary;
  for (auto [fi, sign] : selected) {
    const auto &f = mesh.faces[fi];
    if (on_air_gap(mesh, fi))
      return Error{std::format("MFORCE surface face ({}) lies on the air gap; "
                               "force surfaces may not touch the air gap",
                               face_grids(mesh, f))};
    if (f.cells[1] < 0)
      return Error{
          "MFORCE surface must lie strictly inside the magnetic domain"};
    // A planar contour closes at its end grids; a 3D surface at its edges.
    if (mesh.planar) {
      auto &first = boundary[{f.nodes[0], f.nodes[0]}];
      auto &second = boundary[{f.nodes[1], f.nodes[1]}];
      ++first.first;
      ++second.first;
      first.second -= sign;
      second.second += sign;
      continue;
    }
    for (size_t i = 0; i < f.nodes.size(); ++i) {
      int a = f.nodes[i], b = f.nodes[(i + 1) % f.nodes.size()],
          direction = sign;
      if (a > b) {
        std::swap(a, b);
        direction = -direction;
      }
      auto &count = boundary[{a, b}];
      ++count.first;
      count.second += direction;
    }
  }
  for (const auto &[edge, count] : boundary) {
    // A MAGAXI contour may end on the axis, where its surface of
    // revolution closes.
    const auto on_axis_grid = [&](int v) {
      return std::abs((mesh.nodes[static_cast<size_t>(v)] - mesh.axis_origin)
                          .dot(mesh.radial)) <= 1e-10 * (mesh.cells.front().maximum -
                                                          mesh.cells.front().minimum)
                                                             .norm();
    };
    if (mesh.axisymmetric && count.first == 1 && on_axis_grid(edge[0]))
      continue;
    if (count.first != 2 || count.second != 0)
      return Error{mesh.planar ? "MFORCE planar FSET must be a closed, "
                                 "consistently oriented contour"
                               : "MFORCE FSET must be a closed, consistently "
                                 "oriented surface"};
  }
  // Positive signed enclosed volume distinguishes outward from inward
  // normals.
  double volume = 0;
  for (auto [fi, sign] : selected)
    for (const auto &q : face_quadrature(mesh, mesh.faces[fi]))
      volume += sign * q.weight * q.normal.dot(q.position - mesh.axis_origin) /
                (mesh.planar ? 2 : 3);
  if (!(volume > 0))
    return Error{"MFORCE normals must point outward"};
  return {};
}
Result<std::vector<ForceResult>> forces(const Model &model, const Mesh &mesh,
                                        std::span<const Source> sources,
                                        const Eigen::VectorXd &potential) {
  std::vector<ForceResult> results;
  if (model.forces.empty())
    return results;
  const auto windings = winding_cells(model, mesh);
  std::vector<SpatialIndex> wire_indexes;
  for (const auto &source : sources)
    if (source.wire)
      wire_indexes.emplace_back(source.mesh);
  std::vector<std::vector<int>> incident(mesh.nodes.size());
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci)
    for (int n : mesh.cells[ci].nodes)
      incident[n].push_back(static_cast<int>(ci));
  // Across the periodic sides of a translation a grid's neighbours include
  // its image's.
  if (mesh.period && mesh.period->translation)
    for (auto [a, b] : mesh.period->pairs) {
      auto both = incident[a];
      both.insert(both.end(), incident[b].begin(), incident[b].end());
      incident[a] = incident[b] = std::move(both);
    }
  for (const auto &request : model.forces) {
    std::vector<std::pair<int, int>> faces;
    std::vector<int> component;
    std::vector<std::optional<Across>> across;
    ForceResult result;
    result.id = request.id;
    if (request.body) {
      auto derived = body_surface(model, mesh, incident, request, windings,
                                  sources, wire_indexes);
      if (auto *e = std::get_if<Error>(&derived))
        return *e;
      auto &surface = std::get<BodySurface>(derived);
      faces = std::move(surface.faces);
      component = std::move(surface.component);
      across = std::move(surface.across);
      result.components.resize(surface.components);
    } else {
      auto selected = selected_faces(model, mesh, request.set);
      if (auto *e = std::get_if<Error>(&selected))
        return *e;
      faces = std::get<std::vector<std::pair<int, int>>>(std::move(selected));
      if (auto e = explicit_surface(mesh, faces))
        return *e;
    }
    const Vector origin = request.coordinates
                              ? model.coordinates.at(request.coordinates).a
                              : Vector::Zero();
    for (size_t k = 0; k < faces.size(); ++k) {
      const auto [fi, sign] = faces[k];
      const auto &f = mesh.faces[fi];
      // A face on a periodic side has its other side in the next copy.
      const auto side = k < across.size() ? across[k] : std::nullopt;
      const std::array<int, 2> cells{f.cells[0], side ? side->cell : f.cells[1]};
      for (int ci : cells)
        if (magnetic(model.materials.at(mesh.cells[ci].material)))
          return Error{std::format(
              "MFORCE {}: surface face ({}) touches magnetic element {}; "
              "both sides of every face must be mu_r=1 without remanence",
              request.id, face_grids(mesh, f), mesh.cells[ci].id)};
      if (side && windings.contains(cells[0]) && windings.contains(cells[1]))
        return Error{std::format(
            "MFORCE {}: surface face ({}) on a periodic side lies inside current "
            "source {}; move the surface clear of the winding or enclose it",
            request.id, face_grids(mesh, f), windings.at(cells[0]))};
      if (auto e = current_free(mesh, f, request.id, windings, sources,
                                wire_indexes))
        return *e;
      for (const auto &q : face_quadrature(mesh, f)) {
        const Vector normal = sign * q.normal;
        const Vector b =
            .5 * (field_at(mesh.cells[cells[0]], potential, q.position) +
                  (side ? field_across(mesh, *side, potential, q.position)
                        : field_at(mesh.cells[cells[1]], potential, q.position)));
        Vector force = q.weight * out_of_plane(mesh, q.position) / mu0 *
                       (b * b.dot(normal) - .5 * b.squaredNorm() * normal);
        // Around the axis the radial stress cancels, and there is no torque.
        if (mesh.axisymmetric)
          force = mesh.axis * mesh.axis.dot(force);
        const Vector torque = mesh.axisymmetric ? Vector::Zero().eval()
                                                : (q.position - origin).cross(force);
        result.force += force;
        result.torque += torque;
        if (!component.empty()) {
          result.components[component[k]].force += force;
          result.components[component[k]].torque += torque;
        }
      }
    }
    results.push_back(std::move(result));
  }
  return results;
}

// The MBC that decides a subcase's gauge: its own when it has NORMAL rows.
int normal_boundary(const Model &model, const Subcase &subcase) {
  return subcase.boundary && !model.boundaries.at(subcase.boundary).normal.empty()
             ? subcase.boundary
             : 0;
}
// Whether a and b's stiffness has the same pattern: the same cells on the
// same unknowns, gauge and gap grids.
bool same_topology(const Setup &a, const Setup &b) {
  const auto pattern = [](const Eigen::SparseMatrix<double> &m) {
    return std::pair{std::vector<int>(m.outerIndexPtr(), m.outerIndexPtr() + m.outerSize() + 1),
                     std::vector<int>(m.innerIndexPtr(), m.innerIndexPtr() + m.nonZeros())};
  };
  return a.gauge.equations == b.gauge.equations && a.gap.nodes == b.gap.nodes &&
         pattern(a.gap.stiffness) == pattern(b.gap.stiffness) &&
         std::ranges::equal(a.mesh.cells, b.mesh.cells,
                            [](const Cell &x, const Cell &y) { return x.edges == y.edges; });
}
// shared, when given, is a setup of the same model moved (a MOTION
// position's base): the mesh keeps its topology, and the pattern is shared
// when the gauge and the air gap's couplings agree too.
Result<Setup> prepare(const Model &model, int normal_boundary = 0,
                      const Setup *shared = nullptr) {
  Setup s;
  auto built = shared ? make_mesh(model, shared->mesh) : make_mesh(model);
  if (auto *e = std::get_if<Error>(&built))
    return *e;
  s.mesh = std::get<Mesh>(std::move(built));
  s.geometry = CellGeometry(s.mesh);
  s.materials = resolve_materials(model);
  s.normal_boundary = normal_boundary;
  auto natural = natural_faces(model, s.mesh, normal_boundary);
  if (auto *e = std::get_if<Error>(&natural))
    return *e;
  s.natural = std::get<std::vector<bool>>(std::move(natural));
  auto gauged = make_gauge(s.mesh, s.natural);
  if (auto *e = std::get_if<Error>(&gauged))
    return *e;
  s.gauge = std::get<Gauge>(std::move(gauged));
  s.gap = gap_coupling(s.mesh);
  s.pattern = shared && same_topology(*shared, s)
                  ? shared->pattern
                  : std::make_shared<const Pattern>(make_pattern(s.mesh, s.gauge, s.gap));
  const auto &gs = s.gap.stiffness;
  s.gap_values.reserve(static_cast<size_t>(gs.nonZeros()));
  for (Eigen::Index j = 0; j < gs.outerSize(); ++j)
    for (Eigen::SparseMatrix<double>::InnerIterator it(gs, j); it; ++it) {
      const int ni = s.gap.nodes[static_cast<size_t>(it.row())],
                nj = s.gap.nodes[static_cast<size_t>(j)];
      const int row = s.gauge.equations[ni], col = s.gauge.equations[nj];
      s.gap_values.push_back(row < 0 || col < 0
                                 ? 0.0
                                 : s.gauge.sign[ni] * s.gauge.sign[nj] * it.value());
    }
  // Planar windings that did not move keep their sources: the same cells in
  // the same places give the same coupling.
  const auto fixed = [&](const PlanarCoil &coil) {
    return std::ranges::all_of(model.element_sets.at(coil.elements), [&](int id) {
      const auto i = static_cast<size_t>(s.mesh.element_index.at(id));
      return s.mesh.cells[i].points == shared->mesh.cells[i].points;
    });
  };
  if (shared && s.mesh.planar && std::ranges::all_of(model.planar_coils, fixed)) {
    s.sources = shared->sources;
  } else {
    auto prepared = prepare_sources(model, s.mesh);
    if (auto *e = std::get_if<Error>(&prepared))
      return *e;
    s.sources = std::get<std::vector<Source>>(std::move(prepared));
  }
  auto resolved = resolve_circuits(model, s.sources);
  if (auto *e = std::get_if<Error>(&resolved))
    return *e;
  s.circuits = std::get<std::vector<CircuitDefinition>>(std::move(resolved));
  for (const auto &c : s.circuits) {
    Eigen::VectorXd g = Eigen::VectorXd::Zero(s.mesh.dofs());
    for (auto [i, polarity] : c.members)
      g += polarity * s.sources[i].coupling;
    s.circuit_coupling.push_back(g / c.parallel);
  }
  s.index.emplace(s.mesh);
  return s;
}
// What a DRIVE position adds to its subcase's LOAD.
struct StateRequest {
  std::vector<double> driven; // terminal current added to each circuit
  // Circuits (indices) whose frozen-permeability inductance is wanted.
  std::vector<int> frozen;
};
// One equilibrium of subcase's load on setup and the results that follow
// from it. warm, when given, is a nearby solution to start Newton from (the
// previous MOTION position); continued reports current continuation.
// frozen_inductance receives the request's frozen-permeability inductance.
Result<CaseResults> solve_state(const Model &model, const Setup &setup,
                                const Subcase &subcase, Tangent &tangent,
                                const Eigen::VectorXd *warm, bool &continued,
                                const StateRequest *request = nullptr,
                                Eigen::MatrixXd *frozen_inductance = nullptr) {
  const auto &mesh = setup.mesh;
  const auto &gauge = setup.gauge;
  const auto &circuits = setup.circuits;
  const auto &sources = setup.sources;
  const int ns = static_cast<int>(sources.size()),
            nc = static_cast<int>(circuits.size());
  continued = false;
  auto terminal = circuit_currents(model, circuits, subcase.load);
  if (auto *e = std::get_if<Error>(&terminal))
    return *e;
  auto circuit_amps = std::get<std::vector<double>>(terminal);
  if (request && !request->driven.empty())
    for (int c = 0; c < nc; ++c)
      circuit_amps[c] += request->driven[c];
  std::vector<double> amps(ns, 0);
  for (int c = 0; c < nc; ++c)
    for (auto [i, polarity] : circuits[c].members)
      amps[i] = polarity * circuit_amps[c] / circuits[c].parallel;
  Eigen::VectorXd load = Eigen::VectorXd::Zero(mesh.dofs());
  for (size_t i = 0; i < amps.size(); ++i)
    load += amps[i] * sources[i].coupling;
  auto imposed = boundary_potential(model, mesh, gauge, setup.natural, subcase);
  if (auto *e = std::get_if<Error>(&imposed))
    return *e;
  const auto &initial = std::get<Eigen::VectorXd>(imposed);
  Result<Equilibrium> solved = Error{""};
  if (warm) {
    // Free values from the warm start; prescribed ones as imposed here.
    Eigen::VectorXd start = initial;
    for (size_t e = 0; e < mesh.dofs(); ++e)
      if (gauge.equations[e] >= 0)
        start[e] = (*warm)[e];
    solved = equilibrium(model, setup, tangent, start, load);
  }
  if (!warm || std::holds_alternative<Error>(solved))
    solved = equilibrium(model, setup, tangent, initial, load);
  if (std::holds_alternative<Error>(solved) && !model.bh_tables.empty()) {
    // Retry with fixed current continuation. Each stage must satisfy the same
    // convergence criterion; a failed stage is never returned as a solution.
    Eigen::VectorXd a = initial;
    int iterations = 0;
    for (int stage = 0; stage <= 8; ++stage) {
      solved = equilibrium(model, setup, tangent, a, load * (stage / 8.0));
      if (auto *e = std::get_if<Error>(&solved))
        return Error{std::format("continuation stage {}: {}", stage, e->message)};
      const auto &eq = std::get<Equilibrium>(solved);
      iterations += eq.iterations;
      a = eq.a;
    }
    std::get<Equilibrium>(solved).iterations = iterations;
    continued = true;
  }
  if (auto *e = std::get_if<Error>(&solved))
    return *e;
  auto eq = std::get<Equilibrium>(std::move(solved));
  CaseResults output;
  output.id = subcase.id;
  output.iterations = eq.iterations;
  output.residual = eq.residual;
  output.energy = eq.assembly.energy;
  output.extrapolated = eq.assembly.extrapolated;
  output.potential = eq.a;
  output.b = eq.assembly.b;
  output.h = eq.assembly.h;
  output.coenergy = -output.energy;
  constexpr double unavailable = std::numeric_limits<double>::quiet_NaN();
  for (int i = 0; i < ns; ++i) {
    const auto &source = sources[i];
    const double linkage = source.coupling.dot(eq.a);
    output.coenergy += amps[i] * linkage;
    output.sources.push_back({source.id, source.wire, source.closed, amps[i],
                              source.closed ? linkage : unavailable});
  }
  for (int c = 0; c < nc; ++c)
    output.circuits.push_back({circuits[c].id, circuit_amps[c],
                               circuits[c].closed
                                   ? setup.circuit_coupling[c].dot(eq.a)
                                   : unavailable});
  // Differential inductances g_i^T K^-1 g_j, from the converged tangent,
  // among closed sources and among closed circuits; others are unavailable.
  output.differential_inductance =
      Eigen::MatrixXd::Constant(ns, ns, unavailable);
  output.circuit_inductance = Eigen::MatrixXd::Constant(nc, nc, unavailable);
  output.linkage_rate = Eigen::VectorXd::Constant(nc, unavailable);
  if (gauge.size > 0) {
    // Reduced couplings of the closed sources, then of the closed circuits,
    // solved together with one factorization of the converged tangent.
    std::vector<int> closed_sources, closed_circuits;
    std::vector<const Eigen::VectorXd *> couplings;
    for (int i = 0; i < ns; ++i)
      if (sources[i].closed) {
        closed_sources.push_back(i);
        couplings.push_back(&sources[i].coupling);
      }
    for (int c = 0; c < nc; ++c)
      if (circuits[c].closed) {
        closed_circuits.push_back(c);
        couplings.push_back(&setup.circuit_coupling[c]);
      }
    if (!couplings.empty()) {
      Eigen::MatrixXd reduced(gauge.size,
                              static_cast<Eigen::Index>(couplings.size()));
      for (size_t k = 0; k < couplings.size(); ++k)
        reduced.col(static_cast<Eigen::Index>(k)) = gauge.reduce(*couplings[k]);
      // Direct, also with a reused factor: CG to the accuracy these need
      // costs more than refactoring.
      if (auto e = tangent.update(eq.assembly.stiffness))
        return Error{std::format(
            "Differential inductance factorization failed ({})", e->message)};
      auto responses = tangent.solve(reduced, 0);
      if (auto *e = std::get_if<Error>(&responses))
        return Error{std::format("Differential inductance solve failed ({})",
                                 e->message)};
      const Eigen::MatrixXd linkage =
          reduced.transpose() * std::get<Eigen::MatrixXd>(responses);
      const auto first_circuit =
          static_cast<Eigen::Index>(closed_sources.size());
      for (size_t i = 0; i < closed_sources.size(); ++i)
        for (size_t j = 0; j < closed_sources.size(); ++j)
          output.differential_inductance(closed_sources[i], closed_sources[j]) =
              linkage(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
      for (size_t c = 0; c < closed_circuits.size(); ++c)
        for (size_t d = 0; d < closed_circuits.size(); ++d)
          output.circuit_inductance(closed_circuits[c], closed_circuits[d]) =
              linkage(first_circuit + static_cast<Eigen::Index>(c),
                      first_circuit + static_cast<Eigen::Index>(d));
      if (mesh.air_gap) {
        // dlambda_c/d(delta) = dT/di_c: the torque's gradient against the
        // potential's response to the circuit's current.
        const Eigen::VectorXd gradient =
            mesh.depth * (mesh.air_gap->band_points > 0
                              ? band_torque_gradient(*mesh.air_gap, eq.a)
                              : air_gap_torque_gradient(*mesh.air_gap, eq.a));
        const Eigen::VectorXd reduced_gradient = gauge.reduce(gradient);
        const Eigen::VectorXd rates =
            std::get<Eigen::MatrixXd>(responses).transpose() * reduced_gradient;
        for (size_t c = 0; c < closed_circuits.size(); ++c)
          output.linkage_rate[closed_circuits[c]] =
              rates[first_circuit + static_cast<Eigen::Index>(c)];
      }
    }
    if (request && !request->frozen.empty() && frozen_inductance) {
      // The secant stiffness of the converged state: the linear problem
      // whose solution with the magnets on is this state, so its response
      // to the phase currents alone is the apparent inductance.
      const auto frozen =
          assemble(setup, eq.a, Eigen::VectorXd::Zero(mesh.dofs()), true);
      const auto count = static_cast<Eigen::Index>(request->frozen.size());
      Eigen::MatrixXd reduced(gauge.size, count);
      for (Eigen::Index k = 0; k < count; ++k)
        reduced.col(k) = gauge.reduce(setup.circuit_coupling[request->frozen[k]]);
      if (auto e = tangent.update(frozen.stiffness))
        return Error{std::format("Frozen-permeability factorization failed ({})",
                                 e->message)};
      auto responses = tangent.solve(reduced, 0);
      if (auto *e = std::get_if<Error>(&responses))
        return Error{std::format("Frozen-permeability solve failed ({})",
                                 e->message)};
      *frozen_inductance = reduced.transpose() * std::get<Eigen::MatrixXd>(responses);
    }
  } else {
    for (int i = 0; i < ns; ++i)
      for (int j = 0; j < ns; ++j)
        if (sources[i].closed && sources[j].closed)
          output.differential_inductance(i, j) = 0;
    for (int c = 0; c < nc; ++c)
      for (int d = 0; d < nc; ++d)
        if (circuits[c].closed && circuits[d].closed)
          output.circuit_inductance(c, d) = 0;
  }
  // The gap's harmonics, computed at the first probe in the gap and reused.
  std::optional<AirGapSeries> gap_series;
  for (const auto &probe : model.probes) {
    Vector b = Vector::Zero(), h = Vector::Zero();
    int count = 0;
    for (int ci : setup.index->query(probe.position, probe.position)) {
      const auto &cell = mesh.cells[ci];
      if (contains(cell, probe.position)) {
        const Vector field = field_at(cell, eq.a, probe.position);
        b += field;
        h += response_at(setup.materials.at(cell.property), probe.position, field)
                 .h;
        ++count;
      }
    }
    if (count == 0 && mesh.air_gap && in_air_gap(*mesh.air_gap, probe.position)) {
      // The gap is air: H = B / mu0.
      if (!gap_series)
        gap_series.emplace(*mesh.air_gap, eq.a);
      b = gap_series->field(probe.position);
      h = b / mu0;
      count = 1;
    }
    if (count == 0)
      return Error{"MPROBE lies outside the magnetic domain: " +
                   std::to_string(probe.id)};
    output.probe_b.push_back(b / count);
    output.probe_h.push_back(h / count);
  }
  auto force = forces(model, mesh, sources, eq.a);
  if (auto *e = std::get_if<Error>(&force))
    return *e;
  output.forces = std::get<std::vector<ForceResult>>(std::move(force));
  if (mesh.air_gap) {
    const double gap_load = mesh.depth * (mesh.air_gap->band_points > 0
                                              ? band_torque(*mesh.air_gap, eq.a)
                                              : air_gap_torque(*mesh.air_gap, eq.a));
    (mesh.air_gap->shape == GapShape::Annulus ? output.gap_torque : output.gap_force) =
        gap_load;
  }
  if (mesh.period) {
    // A sector stands for the whole machine: its cells' energy, and each
    // source's linkage, are one sector's share. A source is in series with
    // its images in the other sectors, which carry its current times the
    // period's sign and link the same flux. The gap torque and the linkage
    // rates come from the unfolded gap, the whole machine's already. A
    // strip's or tube's unfolded lines are one or two pitches.
    const double n = mesh.period->sectors;
    if (mesh.air_gap && mesh.air_gap->shape != GapShape::Annulus) {
      output.gap_force *= n / mesh.air_gap->sectors;
      output.linkage_rate *= n / mesh.air_gap->sectors;
    }
    output.energy *= n;
    output.coenergy *= n;
    for (auto &source : output.sources)
      source.linkage *= n;
    for (auto &circuit : output.circuits)
      circuit.linkage *= n;
    output.differential_inductance *= n;
    output.circuit_inductance *= n;
    if (frozen_inductance && request && !request->frozen.empty())
      *frozen_inductance *= n;
  }
  return output;
}
std::string iterations_text(int iterations) {
  return std::format("{} Newton {}", iterations,
                     iterations == 1 ? "iteration" : "iterations");
}
// The circuit index of each DRIVE phase; every phase must be closed.
Result<std::vector<int>> phase_indices(const Drive &drive,
                                       std::span<const CircuitDefinition> circuits) {
  std::vector<int> out;
  for (size_t k = 0; k < drive.phases.size(); ++k) {
    const auto it = std::find_if(circuits.begin(), circuits.end(),
                                 [&](const CircuitDefinition &c) {
                                   return c.id == drive.phases[k];
                                 });
    if (it == circuits.end())
      return Error{std::format("DRIVE {} phase {} (circuit {}) is not a circuit",
                               drive.id, k + 1, drive.phases[k])};
    if (!it->closed)
      return Error{std::format("DRIVE {} phase {} (circuit {}) is an open "
                               "circuit, which has no linkage",
                               drive.id, k + 1, drive.phases[k])};
    out.push_back(static_cast<int>(it - circuits.begin()));
  }
  return out;
}
// The dq quantities of a DRIVE step; frozen is the frozen-permeability
// inductance of the phases, empty under LDQ = INCREMENTAL.
DriveState drive_state(double theta_e, const CaseResults &step,
                       const std::vector<int> &phases,
                       const Eigen::MatrixXd &frozen) {
  const auto m = static_cast<Eigen::Index>(phases.size());
  const Eigen::MatrixXd c = park_columns(static_cast<int>(m), theta_e),
                        park = 2. / static_cast<double>(m) * c.transpose();
  Eigen::VectorXd current(m), linkage(m);
  Eigen::MatrixXd inductance(m, m);
  for (Eigen::Index j = 0; j < m; ++j) {
    current[j] = step.circuits[phases[j]].current;
    linkage[j] = step.circuits[phases[j]].linkage;
    for (Eigen::Index k = 0; k < m; ++k)
      inductance(j, k) = step.circuit_inductance(phases[j], phases[k]);
  }
  DriveState out;
  out.electrical = theta_e * 180 / std::numbers::pi;
  out.current = park * current;
  out.linkage = park * linkage;
  out.incremental = park * inductance * c;
  if (frozen.size()) {
    out.apparent = park * frozen * c;
    out.magnet = park * (linkage - frozen * current);
  }
  return out;
}
// THETA0 AUTO: the no-load linkage of each phase at alignment_positions,
// with subcase's boundary condition and no current, fitted by align_drive.
// iterations receives the Newton iterations of the pre-sweep.
Result<DriveSummary> auto_alignment(const Model &model, const Setup &base,
                                    const Motion &motion, const MovingPart &part,
                                    const Drive &drive,
                                    const std::vector<int> &phases,
                                    Subcase subcase, Tangent &tangent,
                                    int &iterations) {
  subcase.load = 0;
  std::vector<std::vector<double>> linkages;
  Eigen::VectorXd previous;
  iterations = 0;
  for (double angle : alignment_positions(drive, motion)) {
    const Model moved = moved_model(model, motion, part, angle);
    std::optional<Setup> own;
    if (angle != 0) {
      auto built = prepare(moved, base.normal_boundary, &base);
      if (auto *e = std::get_if<Error>(&built))
        return Error{std::format("at {:.6g} {}: {}", angle, motion.unit(), e->message)};
      own = std::get<Setup>(std::move(built));
    }
    bool continued = false;
    auto state = solve_state(moved, own ? *own : base, subcase, tangent,
                             previous.size() ? &previous : nullptr, continued);
    if (auto *e = std::get_if<Error>(&state))
      return Error{std::format("at {:.6g} {}: {}", angle, motion.unit(), e->message)};
    const auto &solved = std::get<CaseResults>(state);
    iterations += solved.iterations;
    previous = solved.potential;
    std::vector<double> row(phases.size());
    std::transform(phases.begin(), phases.end(), row.begin(),
                   [&](int c) { return solved.circuits[c].linkage; });
    linkages.push_back(std::move(row));
  }
  auto aligned = align_drive(drive, motion, linkages);
  if (auto *e = std::get_if<Error>(&aligned))
    return *e;
  const auto &a = std::get<Alignment>(aligned);
  return DriveSummary{drive.id, a.theta0, true, a.linkage, a.lag, phases};
}

// The right endpoints of the last cycle sample each position once and
// provide the backward Euler quadrature for dissipated power.
void summarize_cycle(CaseResults &output, double state_change, double speed,
                     double cycle_distance) {
  auto &summary = *output.transient;
  const auto count = static_cast<size_t>(summary.steps_per_cycle);
  const size_t first = output.steps.size() - count;
  const bool compare = summary.cycles > 1;
  double torque = 0, loss = 0, energy = 0, minimum = output.steps[first].torque;
  double maximum = minimum, torque_change = 0, loss_change = 0, torque_scale = 0;
  double electrical = 0, winding = 0, current_change = 0, current_scale = 0;
  std::ranges::fill(summary.mean_property_loss, 0.);
  for (size_t i = first; i < output.steps.size(); ++i) {
    const auto &step = output.steps[i];
    torque += step.torque;
    loss += step.eddy->loss;
    electrical += step.eddy->electrical_input;
    winding += step.eddy->winding_loss;
    energy = std::max(energy, step.energy);
    minimum = std::min(minimum, step.torque);
    maximum = std::max(maximum, step.torque);
    torque_scale = std::max(torque_scale, std::abs(step.torque));
    for (size_t p = 0; p < summary.properties.size(); ++p)
      summary.mean_property_loss[p] += step.eddy->property_loss[p];
    if (compare) {
      const auto &prior = output.steps[i - count];
      torque_change = std::max(torque_change, std::abs(step.torque - prior.torque));
      loss_change += std::abs(step.eddy->loss - prior.eddy->loss);
      torque_scale = std::max(torque_scale, std::abs(prior.torque));
      for (size_t c = 0; c < step.circuits.size(); ++c) {
        const double i_now = step.circuits[c].current, i_prior = prior.circuits[c].current;
        current_scale = std::max({current_scale, std::abs(i_now), std::abs(i_prior)});
        current_change = std::max(current_change, std::abs(i_now - i_prior));
      }
    }
  }
  summary.mean_load = torque / count;
  summary.peak_to_peak = maximum - minimum;
  summary.mean_loss = loss / count;
  summary.mean_electrical_input = electrical / count;
  summary.mean_winding_loss = winding / count;
  for (double &power : summary.mean_property_loss)
    power /= count;
  // The floors prevent round-off in a torque or loss approaching zero
  // from keeping a settled electromagnetic state from passing the check.
  const double torque_floor = 1e-5 * energy / cycle_distance;
  const double power_floor = std::abs(speed) * torque_floor;
  summary.cycle_residual = std::max(
      {state_change, torque_change / std::max({torque_scale, torque_floor, 1e-30}),
       (loss_change / count) / std::max({summary.mean_loss, power_floor, 1e-30}),
       current_change / std::max(current_scale, 1e-12)});
  summary.power_residual = summary.mean_loss + summary.mean_winding_loss +
                           speed * summary.mean_load - summary.mean_electrical_input;
}

struct TransientExcitation {
  const Drive *drive{nullptr};
  double theta0{0};
  std::vector<int> phases;
  std::vector<double> fixed_current;
  std::vector<CircuitData> circuit_data;
  std::vector<std::pair<size_t, const Voltage *>> applied_voltage;
  VoltageStep voltage;
};

Result<TransientExcitation> transient_excitation(const Model &model, const Setup &base,
                                                 const Subcase &subcase, const Motion &motion,
                                                 const MovingPart &part, Tangent &tangent,
                                                 double repeat, double dt, int steps) {
  TransientExcitation out;
  auto prescribed = circuit_currents(model, base.circuits, subcase.load);
  if (const auto *e = std::get_if<Error>(&prescribed))
    return *e;
  out.fixed_current = std::get<std::vector<double>>(std::move(prescribed));
  if (subcase.drive) {
    out.drive = &model.drives.at(subcase.drive);
    const double periods = motion.translate ? repeat / (2 * out.drive->pitch)
                                            : out.drive->poles;
    if (periods < 1 || std::abs(periods - std::round(periods)) > 1e-9 ||
        2 * periods >= steps)
      return Error{"MAGTRANS DRIVE needs whole electrical periods in the motion cycle "
                   "and more than two time steps per electrical period"};
    auto indices = phase_indices(*out.drive, base.circuits);
    if (const auto *e = std::get_if<Error>(&indices))
      return *e;
    out.phases = std::get<std::vector<int>>(std::move(indices));
    if (out.drive->theta0)
      out.theta0 = *out.drive->theta0;
    else {
      int iterations = 0;
      auto aligned = auto_alignment(model, base, motion, part, *out.drive,
                                    out.phases, subcase, tangent, iterations);
      if (const auto *e = std::get_if<Error>(&aligned))
        return Error{"MAGTRANS DRIVE THETA0 AUTO: " + e->message};
      out.theta0 = std::get<DriveSummary>(aligned).theta0;
    }
  }
  for (size_t i = 0; i < base.circuits.size(); ++i) {
    const auto &circuit = base.circuits[i];
    const auto data = model.circuit_data.find(circuit.id);
    out.circuit_data.push_back(data == model.circuit_data.end() ? CircuitData{} : data->second);
    for (const auto &v : model.voltages)
      if (subcase.voltage == v.set && circuit.id == v.target) {
        if (v.peak > 0 && 2. * v.order >= steps)
          return Error{std::format("VCIRC {} circuit {} harmonic {} needs more time steps",
                                   v.set, v.target, v.order)};
        out.applied_voltage.emplace_back(i, &v);
      }
    if (!circuit.closed &&
        (out.fixed_current[i] != 0 ||
         std::ranges::find(out.phases, static_cast<int>(i)) != out.phases.end() ||
         (!out.applied_voltage.empty() && out.applied_voltage.back().first == i)))
      return Error{std::format("MAGTRANS excitation requires closed circuit {}", circuit.id)};
  }
  const auto count = static_cast<Eigen::Index>(out.applied_voltage.size());
  out.voltage.full.resize(base.mesh.dofs(), count);
  out.voltage.reduced.resize(base.gauge.size, count);
  out.voltage.impedance.resize(count);
  out.voltage.history.resize(count);
  const int sectors = base.mesh.period ? base.mesh.period->sectors : 1;
  for (Eigen::Index p = 0; p < count; ++p) {
    const auto i = out.applied_voltage[p].first;
    out.voltage.full.col(p) = base.circuit_coupling[i];
    out.voltage.reduced.col(p) = base.gauge.reduce(base.circuit_coupling[i]);
    const auto &data = out.circuit_data[i];
    out.voltage.impedance[p] = (dt * data.resistance + data.end_inductance) / sectors;
    if (!(out.voltage.impedance[p] > 0) || !std::isfinite(out.voltage.impedance[p]))
      return Error{std::format("CIRCDAT {} gives an unrepresentable time-step impedance",
                               base.circuits[i].id)};
  }
  return out;
}

Result<CaseResults> solve_transient(const Model &model, const Setup &base,
                                    const Subcase &subcase, const Motion &motion,
                                    const MovingPart &part, Tangent &tangent) {
  const auto &control = model.time_steps.at(subcase.time_step);
  const auto &base_gap = *base.mesh.air_gap;
  const double repeat = motion.translate ? base_gap.length : 360.;
  const double distance = motion.translate ? repeat : 2 * std::numbers::pi;
  const char *speed_unit = motion.translate ? "m/s" : "rpm";
  const double period = (motion.translate ? repeat : 60.) / std::abs(motion.speed);
  const double steps = std::max(4., std::ceil(period / control.maximum_dt));
  if (!std::isfinite(period * control.max_cycles) ||
      steps * control.max_cycles >= std::numeric_limits<int>::max())
    return Error{std::format("TSTEP {} at SPEED {} {} produces an unrepresentable time grid",
                             subcase.time_step, motion.speed, speed_unit)};
  const double dt = period / steps;
  const double sweep = std::copysign(repeat, motion.speed);
  const double speed = std::copysign(distance / period, motion.speed);
  if (!(dt > 0) || !std::isfinite(speed))
    return Error{std::format("TSTEP {} at SPEED {} {} produces an unrepresentable time step",
                             subcase.time_step, motion.speed, speed_unit)};
  const double sectors = base.mesh.period ? base.mesh.period->sectors : 1;
  const double gap_scale = base.mesh.depth * (motion.translate ? sectors / base_gap.sectors : 1);
  const int steps_per_cycle = static_cast<int>(steps);
  const int last_step = steps_per_cycle * control.max_cycles;
  const Diffusion diffusion(model, base.mesh, base.geometry);
  const auto conductivity = diffusion.tangent(base.gauge, dt);
  auto excitation = transient_excitation(model, base, subcase, motion, part, tangent,
                                          repeat, dt, steps_per_cycle);
  if (const auto *e = std::get_if<Error>(&excitation))
    return *e;
  auto &drive = std::get<TransientExcitation>(excitation);
  CaseResults output;
  output.id = subcase.id;
  auto &summary = output.transient.emplace();
  summary.dt = dt;
  summary.steps_per_cycle = steps_per_cycle;
  summary.properties = diffusion.properties();
  summary.mean_property_loss.resize(summary.properties.size());
  Eigen::VectorXd previous, cycle_start;
  const double start = std::remainder(motion.start, repeat);
  for (int k = 0; k <= last_step; ++k) {
    const double cycle_fraction = (k % steps_per_cycle) / steps;
    const double pose = start + sweep * cycle_fraction;
    const double time = k * dt;
    const auto at = std::format("MOTION {} at t={:.8g} s ({:.8g} {})",
                                motion.id, time, pose, motion.unit());
    const Model moved = moved_model(model, motion, part, pose);
    std::optional<Setup> own;
    if (pose != 0) {
      auto built = prepare(moved, base.normal_boundary, &base);
      if (const auto *e = std::get_if<Error>(&built))
        return Error{std::format("{}: {}", at, e->message)};
      own = std::get<Setup>(std::move(built));
    }
    const auto &setup = own ? *own : base;
    if (setup.gauge.equations != base.gauge.equations || setup.gauge.sign != base.gauge.sign)
      return Error{at + ": magnetic gauge changed during rigid motion"};
    auto prescribed = boundary_potential(moved, setup.mesh, setup.gauge, setup.natural, subcase);
    if (const auto *e = std::get_if<Error>(&prescribed))
      return Error{std::format("{}: {}", at, e->message)};
    auto initial = std::get<Eigen::VectorXd>(std::move(prescribed));
    if (k)
      for (size_t i = 0; i < setup.gauge.equations.size(); ++i)
        if (setup.gauge.equations[i] >= 0)
          initial[i] = previous[i];
    auto currents = drive.fixed_current;
    if (drive.drive) {
      const auto phase = phase_currents(*drive.drive,
          electrical_angle(*drive.drive, motion, drive.theta0, pose));
      for (size_t i = 0; i < phase.size(); ++i)
        currents[drive.phases[i]] = phase[i];
    }
    Eigen::VectorXd load = Eigen::VectorXd::Zero(base.mesh.dofs());
    for (size_t i = 0; i < currents.size(); ++i)
      if (currents[i] != 0)
        load.noalias() += currents[i] * base.circuit_coupling[i];
    const auto voltage_count = static_cast<Eigen::Index>(drive.applied_voltage.size());
    for (Eigen::Index p = 0; k && p < voltage_count; ++p) {
      const auto [i, card] = drive.applied_voltage[p];
      const double phase = std::remainder(card->phase, 360.) * std::numbers::pi / 180;
      const double turns = std::copysign(cycle_fraction, motion.speed) * card->order;
      const double applied = card->dc + card->peak * std::cos(2 * std::numbers::pi * turns + phase);
      const auto &prior = output.steps.back().circuits[i];
      drive.voltage.history[p] = (dt * applied +
          drive.circuit_data[i].end_inductance * prior.current + prior.linkage) / sectors;
    }
    const ImplicitStep implicit{diffusion, conductivity, previous, dt};
    auto attempted = equilibrium(moved, setup, tangent, std::move(initial), load,
                                 k && !diffusion.empty() ? &implicit : nullptr,
                                 k && voltage_count ? &drive.voltage : nullptr);
    if (const auto *e = std::get_if<Error>(&attempted)) {
      output.failure = std::format("{}: {}", at, e->message);
      break;
    }
    auto &solved = std::get<Equilibrium>(attempted);
    const auto &gap = *setup.mesh.air_gap;
    const double direction = part.sign * (part.inner ? 1 : -1);
    CaseResults step;
    step.position = motion.start + sweep * k / steps;
    step.energy = sectors * solved.assembly.energy;
    step.torque = direction * gap_scale * air_gap_torque(gap, solved.a);
    step.residual = solved.residual;
    step.extrapolated = solved.assembly.extrapolated;
    const bool fields = subcase.field.enabled && k % subcase.field.every == 0;
    const auto delta = diffusion.increment(solved.a, k ? previous : solved.a);
    step.eddy = diffusion.sample(base.mesh, base.geometry, delta, dt, fields);
    step.eddy->time = time;
    if (k && voltage_count) {
      const auto induced = drive.voltage.currents(solved.a);
      for (Eigen::Index p = 0; p < voltage_count; ++p)
        currents[drive.applied_voltage[p].first] = induced[p];
    }
    const double unavailable = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < currents.size(); ++i) {
      const auto &circuit = base.circuits[i];
      const double current = currents[i];
      const double linkage = circuit.closed ? sectors * base.circuit_coupling[i].dot(solved.a)
                                            : unavailable;
      step.circuits.push_back({circuit.id, current, linkage});
      double voltage = unavailable;
      if (k && circuit.closed) {
        const auto &prior = output.steps.back().circuits[i];
        const auto &data = drive.circuit_data[i];
        voltage = data.resistance * current +
                  (data.end_inductance * (current - prior.current) + linkage - prior.linkage) / dt;
        step.eddy->electrical_input += voltage * current;
        step.eddy->winding_loss += data.resistance * current * current;
      }
      step.eddy->voltage.push_back(voltage);
    }
    if (fields) {
      step.nodes = setup.mesh.nodes;
      step.air_gap = gap;
      step.potential = solved.a;
      step.b = std::move(solved.assembly.b);
      step.h = std::move(solved.assembly.h);
    }
    previous = std::move(solved.a);
    output.residual = std::max(output.residual, step.residual);
    output.extrapolated |= step.extrapolated;
    output.steps.push_back(std::move(step));
    if (k % steps_per_cycle == 0) {
      if (k) {
        ++summary.cycles;
        const double state_change = (previous - cycle_start).norm() /
            std::max({previous.norm(), cycle_start.norm(), 1e-30});
        summarize_cycle(output, state_change, speed, distance);
        constexpr double cycle_tolerance = 1e-6;
        if (summary.cycles > 1 && summary.cycle_residual <= cycle_tolerance) {
          summary.converged = true;
          break;
        }
      }
      cycle_start = previous;
    }
  }
  output.has_results = !output.steps.empty();
  summary.symbolic_analyses = tangent.stats().symbolic_analyses;
  summary.numeric_factorizations = tangent.stats().numeric_factorizations;
  if (!summary.converged && output.failure.empty())
    output.failure = std::format("TSTEP {} did not reach periodic steady state in {} cycles "
                                  "(relative cycle residual {:.3e})",
                                  subcase.time_step, summary.cycles, summary.cycle_residual);
  return output;
}

// Reorder the real and imaginary equations into a real symmetric
// indefinite system. This keeps sparse pivoting in SolverBackend.
Result<Eigen::VectorXd> harmonic_solve(const Sparse &stiffness, const Sparse &loss,
                                       const Eigen::VectorXd &rhs, SolverBackend &backend) {
  const int n = static_cast<int>(loss.rows());
  if (!n)
    return Eigen::VectorXd{};
  std::vector<Triplet> entries;
  entries.reserve(2 * static_cast<size_t>(stiffness.nonZeros() + loss.nonZeros()));
  for (int col = 0; col < stiffness.outerSize(); ++col)
    for (Sparse::InnerIterator it(stiffness, col); it; ++it) {
      entries.emplace_back(it.row(), col, it.value());
      entries.emplace_back(n + it.row(), n + col, -it.value());
    }
  for (int col = 0; col < loss.outerSize(); ++col)
    for (Sparse::InnerIterator it(loss, col); it; ++it) {
      entries.emplace_back(it.row(), n + col, -it.value());
      entries.emplace_back(n + it.row(), col, -it.value());
    }
  Sparse matrix(2 * n, 2 * n);
  matrix.setFromTriplets(entries.begin(), entries.end());
  auto factored = backend.factor(lower_csr(matrix), FactorOptions{.indefinite = true});
  if (const auto *e = std::get_if<Error>(&factored))
    return Error{"MAGHARM requires a pivoted symmetric indefinite factorization: " + e->message};
  const auto &factor = std::get<std::unique_ptr<LinearFactor>>(factored);
  auto result = factor->solve({rhs.data(), static_cast<size_t>(rhs.size())});
  if (const auto *e = std::get_if<Error>(&result))
    return *e;
  const auto &solution = std::get<std::vector<double>>(result);
  Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(solution.data(), rhs.size());
  const Eigen::VectorXd applied = matrix * x;
  const double relative = (applied - rhs).norm() / std::max({rhs.norm(), applied.norm(), 1e-30});
  if (!x.allFinite() || !std::isfinite(relative) || relative > 1e-8)
    return Error{std::format("MAGHARM linear solve failed residual check ({:.6g})", relative)};
  return x;
}

Result<CaseResults> harmonic_case(const Model &model, const Setup &setup,
                                   const Subcase &sc, const Diffusion *planar,
                                   const SolidDiffusion *solid, SolverBackend &backend) {
  using Complex = std::complex<double>;
  const double omega = 2 * std::numbers::pi * sc.frequency;
  if (!std::isfinite(omega) || !std::isfinite(1 / omega))
    return Error{"MAGHARM angular frequency is outside the numerical range"};
  const auto factors = load_factors(model, sc.load);
  if (const auto *e = std::get_if<Error>(&factors))
    return *e;
  const auto &loads = std::get<std::map<int, double>>(factors);
  const auto field_dofs = static_cast<Eigen::Index>(setup.mesh.dofs());
  const int field_rows = setup.gauge.size;
  CaseResults result;
  result.id = sc.id;
  auto &out = result.harmonic.emplace();
  out.frequency = sc.frequency;
  out.winding_current.resize(setup.circuits.size());
  Eigen::VectorXd load_real = Eigen::VectorXd::Zero(field_dofs), load_imag = load_real;
  for (const auto &source : model.currents)
    if (const auto factor = loads.find(source.set); factor != loads.end())
      for (size_t i = 0; i < setup.circuits.size(); ++i)
        if (setup.circuits[i].id == source.target) {
          const double phase = std::remainder(source.phase, 360.) * std::numbers::pi / 180;
          const Complex current = (factor->second * source.amperes) * std::exp(Complex(0, phase));
          if (current != Complex{} && !setup.circuits[i].closed)
            return Error{std::format("MAGHARM ICIRC requires closed circuit {}", source.target)};
          out.winding_current[i] += current;
          load_real.noalias() += current.real() * setup.circuit_coupling[i];
          load_imag.noalias() += current.imag() * setup.circuit_coupling[i];
        }

  SolidConditions conditions;
  if (solid) {
    auto made = solid->conditions(model, setup.mesh, setup.gauge, sc.voltage);
    if (const auto *e = std::get_if<Error>(&made))
      return *e;
    conditions = std::get<SolidConditions>(std::move(made));
  } else
    conditions.gauge = planar->augmented_gauge(setup.gauge);
  const auto &gauge = conditions.gauge;
  const auto &mass = solid ? solid->mass() : planar->mass();
  auto prescribed = boundary_potential(model, setup.mesh, setup.gauge, setup.natural, sc);
  if (const auto *e = std::get_if<Error>(&prescribed))
    return *e;
  Eigen::VectorXd real = Eigen::VectorXd::Zero(mass.rows()), imag = real;
  real.head(field_dofs) = std::get<Eigen::VectorXd>(prescribed);
  const bool boundary_power = real.head(field_dofs).squaredNorm() != 0;
  for (const auto &electrode : conditions.electrodes) {
    const Complex phi = electrode.voltage * Complex(0, -1 / omega);
    for (int dof : electrode.equations) {
      real[dof] = phi.real();
      imag[dof] = phi.imag();
    }
  }
  auto ar = assemble(setup, real.head(field_dofs), load_real);
  auto ai = assemble(setup, imag.head(field_dofs), load_imag);
  Eigen::VectorXd rhs(2 * gauge.size);
  rhs.head(gauge.size) = omega * gauge.reduce(mass * imag);
  rhs.tail(gauge.size) = omega * gauge.reduce(mass * real);
  rhs.head(field_rows) -= ar.residual;
  rhs.segment(gauge.size, field_rows) += ai.residual;
  auto solved = harmonic_solve(ar.stiffness, reduce_conductivity(mass, gauge, omega), rhs, backend);
  if (const auto *e = std::get_if<Error>(&solved))
    return *e;
  const auto &x = std::get<Eigen::VectorXd>(solved);
  gauge.expand(x.head(gauge.size), real);
  gauge.expand(x.tail(gauge.size), imag);

  const Eigen::VectorXd field_real = real.head(field_dofs), field_imag = imag.head(field_dofs);
  if (sc.field.enabled) {
    result.potential = field_real;
    out.potential_imag = field_imag;
    ar = assemble(setup, field_real, load_real);
    ai = assemble(setup, field_imag, load_imag);
    result.b = std::move(ar.b);
    result.h = std::move(ar.h);
    out.b_imag = std::move(ai.b);
    out.h_imag = std::move(ai.h);
  }
  if (solid)
    solid->sample(setup.mesh, real, imag, omega, out);
  else {
    const auto r = planar->sample(setup.mesh, setup.geometry, real, 1 / omega, sc.field.enabled);
    const auto i = planar->sample(setup.mesh, setup.geometry, imag, 1 / omega, sc.field.enabled);
    out.mean_loss = .5 * (r.loss + i.loss);
    out.properties = planar->properties();
    for (size_t k = 0; k < out.properties.size(); ++k)
      out.property_loss.push_back(.5 * (r.property_loss[k] + i.property_loss[k]));
    for (size_t k = 0; k < r.current_density.size(); ++k) {
      out.current_real.push_back(-i.current_density[k]);
      out.current_imag.push_back(r.current_density[k]);
      out.loss_density.push_back(.5 * (r.loss_density[k] + i.loss_density[k]));
    }
  }
  const int sectors = setup.mesh.period ? setup.mesh.period->sectors : 1;
  for (size_t i = 0; i < setup.circuits.size(); ++i) {
    if (!setup.circuits[i].closed) {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      out.linkage.emplace_back(nan, nan);
      continue;
    }
    const auto &coupling = setup.circuit_coupling[i];
    const auto linkage = static_cast<double>(sectors) *
        Complex(coupling.dot(field_real), coupling.dot(field_imag));
    out.linkage.push_back(linkage);
    out.mean_input += .5 * (Complex(0, omega) * linkage * std::conj(out.winding_current[i])).real();
  }
  if (!conditions.electrodes.empty()) {
    const Eigen::VectorXd reaction_real = mass * real, reaction_imag = mass * imag;
    for (const auto &electrode : conditions.electrodes) {
      Complex current;
      for (int node : electrode.equations)
        current += omega * Complex(-reaction_imag[node], reaction_real[node]);
      out.terminals.push_back({electrode.id, electrode.voltage, current});
      out.mean_input += .5 * (electrode.voltage * std::conj(current)).real();
    }
  }
  out.power_residual = boundary_power ? std::numeric_limits<double>::quiet_NaN()
                                      : out.mean_loss - out.mean_input;
  return result;
}

Result<Results> solve_harmonic(const Model &input, SolverBackend &backend) {
  Model model = input;
  // Remanence is DC in this fixed-geometry linear solve.
  for (auto &material : model.materials)
    material.second.br = 0;
  auto prepared = prepare(model);
  if (const auto *e = std::get_if<Error>(&prepared))
    return *e;
  auto &base = std::get<Setup>(prepared);
  std::optional<Diffusion> planar;
  std::optional<SolidDiffusion> solid;
  if (base.mesh.planar)
    planar.emplace(model, base.mesh, base.geometry);
  else
    solid.emplace(model, base.mesh);
  Results out;
  out.summary.solution.add("Solution", "SOL MAGHARM fixed-geometry magnetics, peak phasors");
  out.summary.solution.add("Field cells", std::to_string(base.mesh.cells.size()));
  out.summary.solver = solver_section(backend);
  for (const auto &sc : model.subcases) {
    std::optional<Setup> separate;
    if (const int normal = normal_boundary(model, sc)) {
      auto made = prepare(model, normal, &base);
      if (const auto *e = std::get_if<Error>(&made))
        return *e;
      separate = std::get<Setup>(std::move(made));
    }
    const auto &setup = separate ? *separate : base;
    auto solved = harmonic_case(model, setup, sc, planar ? &*planar : nullptr,
                                 solid ? &*solid : nullptr, backend);
    const auto context = std::format("Subcase {}", sc.id);
    if (const auto *e = std::get_if<Error>(&solved))
      return Error{std::format("{}: {}", context, e->message)};
    collect_fatals(out.summary, backend.notices().take_fatals(), context);
    collect_notices(out.summary, backend.notices().take(), context);
    out.cases.push_back(std::get<CaseResults>(std::move(solved)));
  }
  out.mesh = std::move(base.mesh);
  out.sources = std::move(base.sources);
  out.circuits = std::move(base.circuits);
  return out;
}
} // namespace

Result<Results> solve(const Model &model) {
  LdltSolverBackend backend;
  return solve(model, backend);
}

Result<Results> solve(const Model &model, SolverBackend &backend) {
  if (auto e = validate(model))
    return *e;
  if (model.options.harmonic)
    return solve_harmonic(model, backend);
  auto prepared = prepare(model);
  if (auto *e = std::get_if<Error>(&prepared))
    return *e;
  const auto &base = std::get<Setup>(prepared);
  const auto &mesh = base.mesh;
  const int ns = static_cast<int>(base.sources.size()),
            nc = static_cast<int>(base.circuits.size());
  Results result;
  auto &solution = result.summary.solution;
  solution.add("Solution", model.options.transient
                               ? "SOL MAGTRANS transient magnetics, SI units"
                               : "SOL MAGSTAT magnetostatics, SI units");
  solution.add("Formulation",
               mesh.axisymmetric
                   ? std::string("axisymmetric (MAGAXI), nodal A_phi on "
                                 "first-order elements, r-weighted")
               : mesh.planar
                   ? std::format("planar, nodal vector potential on first-order "
                                 "elements, depth {:.6g} m",
                                 mesh.depth)
                   : "first-order edge elements, tree gauge");
  solution.add("Field cells", std::to_string(mesh.cells.size()));
  if (mesh.air_gap && mesh.air_gap->shape != GapShape::Annulus) {
    const auto &g = *mesh.air_gap;
    const bool tube = g.shape == GapShape::Tube;
    const int finer = std::max(own_harmonics(g.inner), own_harmonics(g.outer));
    const bool band = g.band_points > 0;
    solution.add(
        "Air gap",
        std::format("PAIRGAP {}, {} in place of its {} elements: "
                    "{}, {} + {} grids over a {:.6g} m period, {}, largest deviation "
                    "{:.1e} of the {}",
                    g.property, band ? "sliding band (BAND)" : "air-gap element (AGE)",
                    g.elements.size(),
                    tube ? std::format("the tube from radius {:.6g} to {:.6g} m about "
                                       "the axis",
                                       g.inner.radius, g.outer.radius)
                         : std::format("the strip between two lines {:.6g} m apart "
                                       "along ({:.6g}, {:.6g}, {:.6g})",
                                       g.outer.radius - g.inner.radius, g.u.x(), g.u.y(),
                                       g.u.z()),
                    g.inner.nodes.size(), g.outer.nodes.size(), g.length,
                    band ? std::format("{} band points in {} layer{}", g.band_points,
                                       g.band_layers, g.band_layers == 1 ? "" : "s")
                         : std::format("{} coupled harmonics, {} on the finer side{}",
                                       coupled_harmonics(g), finer,
                                       g.harmonics < finer
                                           ? std::format(", the gap field's to {}",
                                                         g.harmonics)
                                           : ""),
                    g.deviation, tube ? "larger of the pitch and the radius" : "pitch"));
    if (g.sectors > 1)
      solution.add("Air gap",
                   std::format("the pitch's lines ({} + {} grids) continued over two "
                               "pitches, changing sign in the second{}",
                               g.inner.nodes.size() / 2 + 1, g.outer.nodes.size() / 2 + 1,
                               band ? "" : ": even harmonics of the period vanish"));
  } else if (mesh.air_gap) {
    const auto &g = *mesh.air_gap;
    // Round-off in the fitted center prints as zero.
    const auto shown = [&g](double x) {
      return std::abs(x) < 1e-12 * g.outer.radius ? 0.0 : x;
    };
    const bool band = g.band_points > 0;
    solution.add("Air gap",
                 std::format("PAIRGAP {}, {} in place of its {} "
                             "elements: radius {:.6g} to {:.6g} m about ({:.6g}, "
                             "{:.6g}, {:.6g}), {} + {} grids, {}, largest "
                             "radial deviation {:.1e} of the radius",
                             g.property,
                             band ? "sliding band (BAND)" : "air-gap element (AGE)",
                             g.elements.size(), g.inner.radius,
                             g.outer.radius, shown(g.center.x()), shown(g.center.y()),
                             shown(g.center.z()), g.inner.nodes.size(),
                             g.outer.nodes.size(),
                             band ? std::format("{} band points in {} layer{}", g.band_points,
                                                g.band_layers, g.band_layers == 1 ? "" : "s")
                                  : std::format("{} coupled harmonics, {} on the finer "
                                                "circle{}",
                                                coupled_harmonics(g),
                                                std::max(own_harmonics(g.inner),
                                                         own_harmonics(g.outer)),
                                                g.harmonics < std::max(own_harmonics(g.inner),
                                                                       own_harmonics(g.outer))
                                                    ? std::format(", the gap field's to {}",
                                                                  g.harmonics)
                                                    : ""),
                             g.deviation));
    if (g.sectors > 1)
      solution.add("Air gap",
                   std::format("the sector's arcs ({} + {} grids) unfolded around "
                               "{} sectors{}{}",
                               g.inner.nodes.size() / g.sectors + 1,
                               g.outer.nodes.size() / g.sectors + 1, g.sectors,
                               mesh.period->anti ? ", changing sign in alternate sectors"
                                                 : "",
                               band ? std::string()
                               : mesh.period->anti
                                   ? std::format(": harmonics that are not odd multiples "
                                                 "of {} vanish",
                                                 g.sectors / 2)
                                   : std::format(": harmonics that are not multiples of "
                                                 "{} vanish",
                                                 g.sectors)));
  }
  if (const auto &period = mesh.period) {
    const auto &card = *model.period;
    const std::string kind =
        period->translation
            ? std::format("{} pitch of {} along the {} axis of CID {}, {:.6g} long; "
                          "FSET {} moves onto FSET {}",
                          period->anti ? "anti-periodic" : "periodic", period->sectors,
                          mesh.axisymmetric ? "z" : "x", card.coordinates, card.shift,
                          card.first, card.second)
            : std::format("{} sector of {} about the z axis of CID {} through "
                          "({:.6g}, {:.6g}, {:.6g}), {:.6g} degrees; FSET {} turns onto "
                          "FSET {}",
                          period->anti ? "anti-periodic" : "periodic", period->sectors,
                          card.coordinates, period->origin.x(), period->origin.y(),
                          period->origin.z(), 360. / period->sectors, card.first,
                          card.second);
    const std::string part = period->translation ? "pitch" : "sector";
    solution.add(
        "Periodic",
        std::format("MPERIOD {}: {} ({} grid pairs{}). Energy, coenergy, linkage, "
                    "inductance, back-EMF and air-gap {} are the whole "
                    "machine's: {} times the {}'s, its sources in series with their "
                    "images. Fields, probes and MFORCE are the modeled {}'s",
                    period->id, kind, period->pairs.size(),
                    period->axial.empty()
                        ? std::string()
                        : std::format(", {} on the axis{}", period->axial.size(),
                                      period->anti ? " held at A = 0" : ""),
                    period->translation ? "force" : "torque", period->sectors, part,
                    part));
  }
  for (const auto &[pid, reason] : model.meshed_air_gaps)
    solution.add("Air gap",
                 std::format("PAIRGAP {} solved as meshed air (relative "
                             "permeability 1) by {}",
                             pid, reason));
  // The moving part of each MOTION a subcase uses, checked on the mesh as
  // given.
  std::map<int, MovingPart> parts;
  for (const auto &subcase : model.subcases) {
    if (!subcase.motion || parts.contains(subcase.motion))
      continue;
    const auto &motion = model.motions.at(subcase.motion);
    auto part = moving_part(model, mesh, motion);
    if (auto *e = std::get_if<Error>(&part))
      return *e;
    const auto &moving = std::get<MovingPart>(part);
    const auto axis = motion_axis(model, motion);
    const auto &d = axis.direction;
    const char *side = moving.inner ? "inner" : "outer";
    solution.add(
        "Motion",
        motion.translate
            ? std::format("MOTION {} moves ESET {} ({} elements, {} grids) along "
                          "({:.6g}, {:.6g}, {:.6g}), the {} axis of CID {}, sliding "
                          "the {} side of PAIRGAP {}: {} {} from {:.6g} to {:.6g} m",
                          motion.id, motion.set, moving.elements.size(),
                          moving.grids.size(), d.x(), d.y(), d.z(),
                          mesh.axisymmetric ? "z" : "x", motion.coordinates, side,
                          mesh.air_gap->property, motion.count,
                          motion.count == 1 ? "position" : "positions",
                          motion.position(0), motion.position(motion.count - 1))
            : std::format("MOTION {} turns ESET {} ({} elements, {} grids) about the "
                          "z axis of CID {} through ({:.6g}, {:.6g}, {:.6g}), moving "
                          "the {} circle of PAIRGAP {}: {} {} from {:.6g} to {:.6g} "
                          "degrees",
                          motion.id, motion.set, moving.elements.size(),
                          moving.grids.size(), motion.coordinates, axis.origin.x(),
                          axis.origin.y(), axis.origin.z(), side,
                          mesh.air_gap->property, motion.count,
                          motion.count == 1 ? "position" : "positions",
                          motion.position(0), motion.position(motion.count - 1)));
    parts.emplace(subcase.motion, moving);
  }
  std::set<int> drives_used;
  for (const auto &subcase : model.subcases) {
    if (!subcase.drive || !drives_used.insert(subcase.drive).second)
      continue;
    const auto &drive = model.drives.at(subcase.drive);
    const auto &motion = model.motions.at(subcase.motion);
    std::string phases;
    for (int c : drive.phases)
      phases += (phases.empty() ? "" : ", ") + std::to_string(c);
    solution.add("Drive",
                 std::format("DRIVE {}: {} currents, {}, {:.6g} A peak, "
                             "gamma {:.6g} electrical degrees, phase circuits {}, "
                             "THETA0 {}",
                             drive.id, drive.six ? "six-step" : "sinusoidal",
                             motion.translate
                                 ? std::format("pole pitch {:.6g} m", drive.pitch)
                                 : std::format("{} pole {}", drive.poles,
                                               drive.poles == 1 ? "pair" : "pairs"),
                             drive.peak, drive.gamma, phases,
                             drive.theta0
                                 ? std::format("{:.6g} {}", *drive.theta0, motion.unit())
                                 : std::string("AUTO")));
  }
  solution.add("Equations",
               std::format("{} ({} degrees of freedom before the gauge and "
                           "boundary conditions)",
                           base.gauge.size, mesh.dofs()));
  solution.add("Sources", std::format("{} ({} circuit{})", ns, nc, nc == 1 ? "" : "s"));
  solution.add("Subcases", std::to_string(model.subcases.size()));
  solution.add("Newton tolerance",
               std::format("relative residual {:.3g}, at most {} iterations",
                           model.options.nonlinear_tolerance,
                           model.options.max_iterations));
  result.summary.solver = solver_section(backend);
  result.summary.solver.set("Indefinite fallback",
                            "none, positive definite required");
  // Only the AGE's gap is a dense block; a sliding band's is sparse.
  Tangent tangent(backend, base.mesh.planar,
                  base.mesh.air_gap && base.mesh.air_gap->band_points == 0,
                  model.options.tangent_reuse);
  // THETA0 AUTO results by DRIVE, MOTION and MBC.
  std::map<std::tuple<int, int, int>, DriveSummary> alignments;
  // The setups of MBCs with NORMAL faces, whose gauge leaves them free.
  std::map<int, Setup> normal_bases;
  for (const auto &subcase : model.subcases) {
    Notices notices;
    const std::string context = std::format("Subcase {}", subcase.id);
    const int gauge_key = normal_boundary(model, subcase);
    if (gauge_key && !normal_bases.contains(gauge_key)) {
      auto built = prepare(model, gauge_key);
      if (auto *e = std::get_if<Error>(&built))
        return Error{std::format("{}: {}", context, e->message)};
      normal_bases.emplace(gauge_key, std::get<Setup>(std::move(built)));
    }
    const Setup &case_base = gauge_key ? normal_bases.at(gauge_key) : base;
    if (model.options.transient) {
      const auto &motion = model.motions.at(subcase.motion);
      auto state = solve_transient(model, case_base, subcase, motion,
                                   parts.at(subcase.motion), tangent);
      if (const auto *e = std::get_if<Error>(&state))
        return Error{std::format("{}: {}", context, e->message)};
      auto output = std::get<CaseResults>(std::move(state));
      if (!output.failure.empty())
        notices.add_fatal(output.failure);
      if (output.extrapolated)
        notices.add("The B-H curve was exceeded; its vacuum-slope extrapolation was used");
      solution.add(context, std::format("MOTION {} at {:.6g} {}, {} accepted time steps",
                                        motion.id, motion.speed, motion.translate ? "m/s" : "rpm",
                                        output.has_results ? output.steps.size() - 1 : 0));
      result.cases.push_back(std::move(output));
    } else if (!subcase.motion) {
      bool continued = false;
      auto state = solve_state(model, case_base, subcase, tangent, nullptr, continued);
      if (auto *e = std::get_if<Error>(&state))
        return Error{std::format("{}: {}", context, e->message)};
      auto output = std::get<CaseResults>(std::move(state));
      if (continued)
        notices.add("Newton did not converge at full current; solved by "
                    "current continuation in 8 stages");
      if (output.extrapolated)
        notices.add("The B-H curve was exceeded; its vacuum-slope "
                    "extrapolation was used");
      solution.add(context,
                   std::format("{}{}, relative residual {:.3e}",
                               iterations_text(output.iterations),
                               continued ? " with current continuation" : "",
                               output.residual));
      result.cases.push_back(std::move(output));
    } else {
      // Each position is the static model moved to it, solved from the
      // previous position's potential.
      const auto &motion = model.motions.at(subcase.motion);
      const auto &part = parts.at(subcase.motion);
      CaseResults output;
      output.id = subcase.id;
      output.residual = 0;
      const Drive *drive = subcase.drive ? &model.drives.at(subcase.drive) : nullptr;
      if (drive) {
        auto indexed = phase_indices(*drive, case_base.circuits);
        if (auto *e = std::get_if<Error>(&indexed))
          return Error{std::format("{}: {}", context, e->message)};
        const auto &phases = std::get<std::vector<int>>(indexed);
        if (drive->theta0) {
          output.drive_summary = DriveSummary{drive->id, *drive->theta0, false, 0,
                                              {}, phases};
        } else {
          const std::tuple key{drive->id, motion.id, subcase.boundary};
          if (!alignments.contains(key)) {
            int iterations = 0;
            auto aligned = auto_alignment(model, case_base, motion, part, *drive, phases,
                                          subcase, tangent, iterations);
            if (auto *e = std::get_if<Error>(&aligned))
              return Error{std::format("{}, DRIVE {} THETA0 AUTO with MOTION {}: {}",
                                       context, drive->id, motion.id, e->message)};
            const auto &found = std::get<DriveSummary>(aligned);
            std::string lags;
            for (size_t k = 1; k < found.lag.size(); ++k)
              lags += std::format("{}{:.2f}", lags.empty() ? "" : ", ", found.lag[k]);
            const auto positions = alignment_positions(*drive, motion);
            solution.add(
                "Drive",
                std::format("DRIVE {} THETA0 AUTO with MOTION {}{}: no-load "
                            "linkage at {} positions from 0 to {:.6g} {} ({}); "
                            "THETA0 {:.9g} {}, phase 1 fundamental {:.6g} "
                            "Wb, later phases lag by {} electrical degrees",
                            drive->id, motion.id,
                            subcase.boundary ? std::format(" and MAGBC {}",
                                                           subcase.boundary)
                                             : "",
                            positions.size(), positions.back(), motion.unit(),
                            iterations_text(iterations), found.theta0, motion.unit(),
                            found.linkage, lags));
            alignments.emplace(key, found);
          }
          output.drive_summary = alignments.at(key);
        }
      }
      Eigen::VectorXd previous;
      int continued_at = 0, fewest = std::numeric_limits<int>::max(), most = 0;
      std::optional<double> first_continued, first_extrapolated;
      for (int k = 0; k < motion.count; ++k) {
        const double angle = motion.position(k);
        const Model moved = moved_model(model, motion, part, angle);
        std::optional<Setup> own;
        if (angle != 0) {
          auto built = prepare(moved, gauge_key, &case_base);
          if (auto *e = std::get_if<Error>(&built))
            return Error{std::format("{}, MOTION {} at {:.6g} {}: {}", context,
                                     motion.id, angle, motion.unit(), e->message)};
          own = std::get<Setup>(std::move(built));
        }
        const Setup &setup = own ? *own : case_base;
        bool continued = false;
        StateRequest request;
        double theta_e = 0;
        if (drive) {
          const auto &phases = output.drive_summary->phases;
          theta_e = electrical_angle(*drive, motion, output.drive_summary->theta0, angle);
          const auto currents = phase_currents(*drive, theta_e);
          request.driven.assign(case_base.circuits.size(), 0);
          for (size_t j = 0; j < phases.size(); ++j)
            request.driven[phases[j]] = currents[j];
          if (subcase.apparent)
            request.frozen = phases;
        }
        Eigen::MatrixXd frozen;
        auto state = solve_state(moved, setup, subcase, tangent,
                                 k ? &previous : nullptr, continued,
                                 drive ? &request : nullptr, &frozen);
        if (auto *e = std::get_if<Error>(&state))
          return Error{std::format("{}, MOTION {} at {:.6g} {}: {}", context,
                                   motion.id, angle, motion.unit(), e->message)};
        auto step = std::get<CaseResults>(std::move(state));
        step.position = angle;
        // The gap's load is on its inner side, along u or about u x v.
        const double turn = part.sign * (part.inner ? 1 : -1);
        step.torque = turn * (motion.translate ? step.gap_force : step.gap_torque);
        step.linkage_rate *= turn;
        if (drive)
          step.drive =
              drive_state(theta_e, step, output.drive_summary->phases, frozen);
        if (continued) {
          ++continued_at;
          first_continued = first_continued.value_or(angle);
        }
        if (step.extrapolated)
          first_extrapolated = first_extrapolated.value_or(angle);
        output.iterations += step.iterations;
        output.residual = std::max(output.residual, step.residual);
        output.extrapolated |= step.extrapolated;
        fewest = std::min(fewest, step.iterations);
        most = std::max(most, step.iterations);
        previous = step.potential;
        // Fields are kept for the positions BFIELD writes, with the grids
        // and the air gap where they were.
        if (subcase.field.enabled && k % subcase.field.every == 0) {
          step.nodes = setup.mesh.nodes;
          step.air_gap = setup.mesh.air_gap;
        } else {
          step.potential.resize(0);
          step.b.clear();
          step.h.clear();
        }
        output.steps.push_back(std::move(step));
      }
      if (continued_at)
        notices.add(std::format("Newton did not converge at full current at {} "
                                "of {} positions (the first at {:.6g} {}); "
                                "they were solved by current continuation in 8 "
                                "stages",
                                continued_at, motion.count, *first_continued,
                                motion.unit()));
      if (first_extrapolated)
        notices.add(std::format("The B-H curve was exceeded (first at {:.6g} "
                                "{}); its vacuum-slope extrapolation was "
                                "used",
                                *first_extrapolated, motion.unit()));
      solution.add(context,
                   std::format("MOTION {}: {} {}, {} ({} to {} per position), "
                               "largest relative residual {:.3e}",
                               motion.id, motion.count,
                               motion.count == 1 ? "position" : "positions",
                               iterations_text(output.iterations), fewest, most,
                               output.residual));
      result.cases.push_back(std::move(output));
    }
    if (model.options.transient) {
      collect_fatals(result.summary, backend.notices().take_fatals(), context);
      collect_fatals(result.summary, notices.take_fatals(), context);
    }
    collect_notices(result.summary, backend.notices().take(), context);
    collect_notices(result.summary, notices.take(), context);
  }
  result.summary.solver.add("Tangent reuse", tangent.usage());
  if (!tangent.ordering().empty())
    result.summary.solver.set("Ordering", std::string(tangent.ordering()));
  auto &owned = std::get<Setup>(prepared);
  result.mesh = std::move(owned.mesh);
  result.sources = std::move(owned.sources);
  result.circuits = std::move(owned.circuits);
  return result;
}
} // namespace vibestran::magnetic
