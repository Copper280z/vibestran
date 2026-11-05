// src/solver/linear_static.cpp
// Linear static analysis pipeline.
// K * u = F_ext + F_thermal

#include "solver/linear_static.hpp"
#include "elements/shell_quad4.hpp"
#include "elements/shell_tria3.hpp"
#include "elements/shell_section.hpp"
#include "core/coord_sys.hpp"
#include "core/factor_ratio_check.hpp"
#include "core/logger.hpp"
#include "core/mpc_handler.hpp"
#include "elements/element_factory.hpp"
#include "elements/line_elements.hpp"
#include "elements/rbe_constraints.hpp"
#include "elements/solid_elements.hpp"
#include "elements/solid_hexa8.hpp"
#include "elements/solid_hexa20.hpp"
#include "elements/midside_nodes.hpp"
#include "io/system_dump.hpp"
#include "assembly_parallel.hpp"
#include "solver/analysis_support.hpp"
#include <Eigen/Dense>
#include <format>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <set>
#include <unordered_set>
#include <spdlog/spdlog.h>

namespace vibestran {

/// Map each local DOF index of an element to its (node, component) identity.
/// Scalar elements (CELAS/CMASS) have one DOF per node on an explicit
/// component; shell/line elements have 6 DOFs per node; solids have 3.
std::vector<std::pair<NodeId, int>> local_dof_identity(const ElementData &elem) {
  const bool scalar = elem.type == ElementType::CELAS1 ||
                      elem.type == ElementType::CELAS2 ||
                      elem.type == ElementType::CMASS1 ||
                      elem.type == ElementType::CMASS2;
  const bool solid = elem.type == ElementType::CHEXA8 ||
                     elem.type == ElementType::CHEXA20 ||
                     elem.type == ElementType::CTETRA4 ||
                     elem.type == ElementType::CTETRA10 ||
                     elem.type == ElementType::CPENTA6;
  std::vector<std::pair<NodeId, int>> result;
  if (scalar) {
    result.reserve(elem.nodes.size());
    for (size_t i = 0; i < elem.nodes.size(); ++i)
      result.emplace_back(elem.nodes[i], elem.components[static_cast<size_t>(i)] - 1);
    return result;
  }
  const int dpn = solid ? 3 : 6;
  result.reserve(elem.nodes.size() * static_cast<size_t>(dpn));
  for (NodeId nid : elem.nodes)
    for (int d = 0; d < dpn; ++d)
      result.emplace_back(nid, d);
  return result;
}

namespace {

/// Reference temperature of a shell element's section (PSHELL MID1's TREF,
/// any of MAT1/MAT2/MAT8, or PCOMP TREF).
[[nodiscard]] double shell_reference_temperature(const Model &model,
                                                 const ElementData &elem) {
  return build_shell_section(model, elem.pid, elem.id,
                             elem.type == ElementType::CTRIA3 ? "CTRIA3" : "CQUAD4")
      .ref_temp;
}

[[nodiscard]] MaterialId property_material_id(const Property &prop) {
  if (std::holds_alternative<PShell>(prop) || std::holds_alternative<PComp>(prop))
    return MaterialId{0}; // shells: see shell_reference_temperature
  if (const auto *psolid = std::get_if<PSolid>(&prop))
    return psolid->mid;
  if (const auto *pbar = std::get_if<PBar>(&prop))
    return pbar->mid;
  if (const auto *pbarl = std::get_if<PBarL>(&prop))
    return pbarl->mid;
  if (const auto *pbeam = std::get_if<PBeam>(&prop))
    return pbeam->mid;
  if (const auto *prod = std::get_if<PRod>(&prop))
    return prod->mid;
  return MaterialId{0};
}

/// An SLOAD as a basic force: on a scalar point it stands in for the point's one
/// component, and on a grid it acts on T1 of the displacement frame.
[[nodiscard]] Vec3 sload_force(const Model &model, const SloadLoad &load,
                               const double scale) {
  const Vec3 along_t1{scale * load.value, 0.0, 0.0};
  const GridPoint &gp = model.node(load.node);
  if (gp.cd == CoordId{0})
    return along_t1;
  const auto cs_it = model.coord_systems.find(gp.cd);
  if (cs_it == model.coord_systems.end())
    return along_t1;
  return apply_rotation(rotation_matrix(cs_it->second, gp.position), along_t1);
}

[[nodiscard]] double component_value(const Vec3 &v, const int component) {
  switch (component) {
  case 1:
    return v.x;
  case 2:
    return v.y;
  case 3:
    return v.z;
  default:
    return 0.0;
  }
}

[[nodiscard]] double wtmass_scale(const Model &model) {
  return model.param_real("WTMASS").value_or(1.0);
}

[[nodiscard]] Vec3 load_direction_in_basic(const Model &model, const CoordId cid,
                                           const Vec3 &direction,
                                           const Vec3 &position) {
  if (cid.value == 0)
    return direction;
  const auto cs_it = model.coord_systems.find(cid);
  if (cs_it == model.coord_systems.end()) {
    throw SolverError(std::format(
        "Coordinate system {} not found for inertial load direction",
        cid.value));
  }
  return apply_rotation(rotation_matrix(cs_it->second, position), direction);
}

[[nodiscard]] std::unordered_map<NodeId, double>
build_nodal_temperature_map(const Model &model, const SubCase &sc) {
  std::unordered_map<NodeId, double> nodal_temps;
  int temp_set = sc.temp_load_set;
  if (temp_set == 0)
    temp_set = sc.load_set.value;

  // TEMP/TEMPD are always looked up by direct SID match against the temperature
  // set — they do not participate in LOAD card combination.
  for (const auto &load : model.loads) {
    if (const TempLoad *tl = std::get_if<TempLoad>(&load)) {
      if (tl->sid == LoadSetId(temp_set))
        nodal_temps[tl->node] = tl->temperature;
    }
  }

  const auto tempd_it = model.tempd.find(temp_set);
  if (tempd_it != model.tempd.end()) {
    const double t_default = tempd_it->second;
    for (const auto &[nid, _] : model.nodes) {
      if (!nodal_temps.contains(nid))
        nodal_temps[nid] = t_default;
    }
  }
  return nodal_temps;
}

[[nodiscard]] double
average_temperature(const std::vector<NodeId> &nodes,
                    const std::unordered_map<NodeId, double> &nodal_temps,
                    const double default_temp) {
  if (nodes.empty())
    return default_temp;
  double sum = 0.0;
  for (NodeId nid : nodes) {
    const auto it = nodal_temps.find(nid);
    sum += (it != nodal_temps.end()) ? it->second : default_temp;
  }
  return sum / static_cast<double>(nodes.size());
}

[[nodiscard]] Eigen::Matrix<double, 6, 6>
solid_constitutive_matrix(const Mat1 &mat) {
  const double lam =
      mat.E * mat.nu / ((1.0 + mat.nu) * (1.0 - 2.0 * mat.nu));
  const double mu = mat.E / (2.0 * (1.0 + mat.nu));
  Eigen::Matrix<double, 6, 6> D = Eigen::Matrix<double, 6, 6>::Zero();
  D(0, 0) = lam + 2.0 * mu;
  D(0, 1) = lam;
  D(0, 2) = lam;
  D(1, 0) = lam;
  D(1, 1) = lam + 2.0 * mu;
  D(1, 2) = lam;
  D(2, 0) = lam;
  D(2, 1) = lam;
  D(2, 2) = lam + 2.0 * mu;
  D(3, 3) = mu;
  D(4, 4) = mu;
  D(5, 5) = mu;
  return D;
}

[[nodiscard]] double solid_von_mises(const Eigen::Matrix<double, 6, 1> &sigma) {
  const double s1 = sigma(0);
  const double s2 = sigma(1);
  const double s3 = sigma(2);
  const double t12 = sigma(3);
  const double t23 = sigma(4);
  const double t31 = sigma(5);
  return std::sqrt(0.5 * ((s1 - s2) * (s1 - s2) + (s2 - s3) * (s2 - s3) +
                          (s3 - s1) * (s3 - s1) +
                          6.0 * (t12 * t12 + t23 * t23 + t31 * t31)));
}

[[nodiscard]] SolidStressPoint
make_solid_stress_point(const NodeId node,
                        const Eigen::Matrix<double, 6, 1> &sigma) {
  SolidStressPoint point;
  point.node = node;
  point.sx = sigma(0);
  point.sy = sigma(1);
  point.sz = sigma(2);
  point.sxy = sigma(3);
  point.syz = sigma(4);
  point.szx = sigma(5);
  point.von_mises = solid_von_mises(sigma);
  return point;
}

[[nodiscard]] SolidStrainPoint
make_solid_strain_point(const NodeId node,
                        const Eigen::Matrix<double, 6, 1> &eps) {
  SolidStrainPoint point;
  point.node = node;
  point.ex = eps(0);
  point.ey = eps(1);
  point.ez = eps(2);
  point.exy = eps(3);
  point.eyz = eps(4);
  point.ezx = eps(5);
  point.von_mises = solid_von_mises(eps);
  return point;
}

[[nodiscard]] PlateStressPoint
make_plate_stress_point(const NodeId node,
                        const Eigen::Vector3d &membrane_stress,
                        const Eigen::Vector3d &bending_moment) {
  PlateStressPoint point;
  point.node = node;
  point.sx = membrane_stress(0);
  point.sy = membrane_stress(1);
  point.sxy = membrane_stress(2);
  point.mx = bending_moment(0);
  point.my = bending_moment(1);
  point.mxy = bending_moment(2);
  point.von_mises =
      std::sqrt(point.sx * point.sx - point.sx * point.sy +
                point.sy * point.sy + 3.0 * point.sxy * point.sxy);
  return point;
}

[[nodiscard]] ShellStrain
make_shell_strain(const Eigen::Vector3d &membrane_strain,
                  const Eigen::Vector3d &curvature,
                  const Eigen::Vector2d &transverse_shear_strain) {
  ShellStrain strain;
  strain.ex = membrane_strain(0);
  strain.ey = membrane_strain(1);
  strain.exy = membrane_strain(2);
  strain.kx = curvature(0);
  strain.ky = curvature(1);
  strain.kxy = curvature(2);
  strain.gxz = transverse_shear_strain(0);
  strain.gyz = transverse_shear_strain(1);
  return strain;
}

/// Plate output point from a shell element response (membrane stress is
/// N / t; moments, resultants and strains as recovered).
void fill_shell_point(PlateStressPoint &pt, const ShellPointResponse &r) {
  pt = make_plate_stress_point(pt.node, r.membrane_resultant / r.thickness,
                               r.bending_moment);
  pt.nx = r.membrane_resultant(0);
  pt.ny = r.membrane_resultant(1);
  pt.nxy = r.membrane_resultant(2);
  pt.qx = r.transverse_shear(0);
  pt.qy = r.transverse_shear(1);
  pt.strain = make_shell_strain(r.membrane_strain, r.curvature,
                                r.transverse_shear_strain);
}

/// Mechanical stress and total strain at one evaluation point of a solid
/// element.  Strain is total strain (B*u, including thermal expansion) to
/// match MYSTRAN, which reports STRAIN = B*U; the free thermal strain is
/// subtracted for stress only.
struct SolidStressStrain {
  Eigen::Matrix<double, 6, 1> sigma{Eigen::Matrix<double, 6, 1>::Zero()};
  Eigen::Matrix<double, 6, 1> eps{Eigen::Matrix<double, 6, 1>::Zero()};
};

[[nodiscard]] SolidStressStrain
solid_stress_strain_from_B(const Eigen::Matrix<double, 6, 6> &D,
                           const Eigen::MatrixXd &B,
                           const Eigen::VectorXd &ue, const double alpha,
                           const double temperature,
                           const double reference_temperature) {
  Eigen::Matrix<double, 6, 1> eps_th;
  eps_th << alpha * (temperature - reference_temperature),
      alpha * (temperature - reference_temperature),
      alpha * (temperature - reference_temperature), 0.0, 0.0, 0.0;
  SolidStressStrain out;
  out.eps = B * ue;
  out.sigma = D * (out.eps - eps_th);
  return out;
}

} // namespace

LinearStaticSolver::LinearStaticSolver(std::unique_ptr<SolverBackend> backend)
    : backend_(std::move(backend)) {}

SolverResults LinearStaticSolver::solve(const Model &model) {
  model.validate();

  SolverResults results;
  results.summary.solution = solution_section(model, "SOL 101 linear static");
  results.summary.solver = solver_section(*backend_);
  for (const auto &sc : model.analysis.subcases) {
    results.subcases.push_back(solve_subcase(model, sc, results.summary));
  }
  return results;
}

namespace {

/// The command line over PARAM,AMGMODES over the default, the rigid-body modes.
AmgModes structural_amg_modes(const Model &model, std::optional<AmgModes> override_modes) {
  if (override_modes)
    return *override_modes;
  const auto it = model.params.find("AMGMODES");
  if (it == model.params.end())
    return AmgModes::Rigid;
  const auto parsed = parse_amg_modes(it->second);
  if (!parsed)
    throw SolverError(std::format("PARAM,AMGMODES,{}: expected CONSTANT or RIGID", it->second));
  return *parsed;
}

} // namespace

SubCaseResults LinearStaticSolver::solve_subcase(const Model &model,
                                                 const SubCase &sc,
                                                 SolveSummary &summary) {
  // Warnings for the F06 NOTICES, with the backend's, under this subcase.
  Notices notices;
  using Clock = std::chrono::steady_clock;
  using Ms = std::chrono::duration<double, std::milli>;
  const auto t0 = Clock::now();

  // 1. Build DOF map and apply SPC boundary conditions
  DofMap dof_map = build_dof_map(model, sc);
  const auto t1 = Clock::now();
  spdlog::debug("[subcase {}] build_dof_map: {:.3f} ms  ({} free DOFs)",
                sc.id, Ms(t1 - t0).count(), dof_map.num_free_dofs());

  // 2. Build MPC handler (CD-frame SPCs + explicit MPCs + RBE2/RBE3)
  MpcHandler mpc_handler;
  build_mpc_system(model, sc, dof_map, mpc_handler);
  const int n = mpc_handler.num_reduced();
  const auto t2 = Clock::now();
  spdlog::debug("[subcase {}] build_mpc_system: {:.3f} ms  ({} reduced DOFs)",
                sc.id, Ms(t2 - t1).count(), n);

  // 3. Assemble global K and F using pre-MPC dof_map
  SparseMatrixBuilder K_builder(n);
  std::vector<double> F(static_cast<size_t>(n), 0.0);

  assemble(model, sc, mpc_handler, K_builder, F);
  const auto t3 = Clock::now();
  spdlog::debug("[subcase {}] assemble K: {:.3f} ms", sc.id, Ms(t3 - t2).count());

  apply_point_loads(model, sc, mpc_handler, F);
  const auto t4 = Clock::now();
  spdlog::debug("[subcase {}] apply_point_loads: {:.3f} ms", sc.id, Ms(t4 - t3).count());

  apply_pressure_loads(model, sc, mpc_handler, F);
  const auto t4b = Clock::now();
  spdlog::debug("[subcase {}] apply_pressure_loads: {:.3f} ms", sc.id,
                Ms(t4b - t4).count());

  apply_inertial_loads(model, sc, mpc_handler, F);
  const auto t4c = Clock::now();
  spdlog::debug("[subcase {}] apply_inertial_loads: {:.3f} ms", sc.id,
                Ms(t4c - t4b).count());

  apply_thermal_loads(model, sc, mpc_handler, K_builder, F);
  const auto t5 = Clock::now();
  spdlog::debug("[subcase {}] apply_thermal_loads: {:.3f} ms", sc.id,
                Ms(t5 - t4c).count());

  // 4. Build CSR
  auto csr = K_builder.build_csr();
  const auto t5b = Clock::now();
  spdlog::debug("[subcase {}] build_csr: {:.3f} ms  ({} nnz)", sc.id, Ms(t5b - t5).count(), csr.nnz);

  const SparseMatrixBuilder::CsrData* solve_csr = &csr;
  SparseMatrixBuilder::CsrData expanded_csr;
  if (backend_->requires_full_symmetric_csr()) {
    expanded_csr = csr.expanded_symmetric();
    solve_csr = &expanded_csr;
  }

  const FactorRatioCheckPolicy factor_ratio_policy =
      build_factor_ratio_check_policy(
          model, std::format("[subcase {}]", sc.id), "stiffness matrix");
  backend_->set_near_nullspace(
      backend_->uses_near_nullspace(n) && structural_amg_modes(model, amg_modes_) == AmgModes::Rigid
          ? rigid_body_modes(model, mpc_handler)
          : nullptr);
  std::vector<double> u_reduced =
      backend_->solve(*solve_csr, F, &factor_ratio_policy);
  const auto t6 = Clock::now();
  spdlog::debug("[subcase {}] linear solve: {:.3f} ms", sc.id, Ms(t6 - t5b).count());
  if (!dump_prefix_.empty()) {
    write_matrix_market(system_dump_path(dump_prefix_, sc.id, "K.mtx"), csr);
    write_matrix_market(system_dump_path(dump_prefix_, sc.id, "F.mtx"), F);
    write_matrix_market(system_dump_path(dump_prefix_, sc.id, "u.mtx"), u_reduced);
    write_dof_map(system_dump_path(dump_prefix_, sc.id, "dofs.txt"), model, mpc_handler);
    spdlog::info("[subcase {}] wrote K, F, u and the DOF map to {}_sc{}_*", sc.id, dump_prefix_,
                 sc.id);
  }

  // Log iterative solver convergence info when available (PCG backends).
  {
    int iters = backend_->last_iteration_count();
    if (iters >= 0)
      spdlog::debug("[subcase {}] PCG: {} iterations, estimated residual = {:.3e}",
                    sc.id, iters, backend_->last_estimated_error());
  }

  // Solution accuracy.  Without refinement (the default for direct solves)
  // the solution is assessed with a compensated residual, as Nastran does:
  // the normwise backward error (round-off for any stable solve) and the
  // load-sequence epsilon u . (F - K u) / u . F, which roughly tracks the
  // forward error and is O(1) for a mechanism.  The relative residual
  // ||F - K u|| / ||F|| is logged but is not an accuracy measure: for an
  // ill-conditioned K it is ~ eps ||K|| ||u|| / ||F|| even for the exact
  // solution rounded to double.  Refined solves (--refine, PARAM,REFINE,YES,
  // and all PCG backends) report the refinement's error estimate instead.
  {
    std::optional<RefinementReport> rep = backend_->last_refinement();
    const bool refined = rep.has_value();
    if (!refined) {
      std::vector<double> u_probe = u_reduced;
      RefinementOptions probe;
      probe.max_corrections = 0;
      rep = refine_solution(csr, F, u_probe, nullptr, probe);
      spdlog::info("[subcase {}] epsilon {:.3e}, relative residual ||F - K*u|| / ||F|| = {:.3e}, "
                   "backward error {:.3e}",
                   sc.id, rep->epsilon, rep->relative_residual, rep->backward_error);
    } else {
      spdlog::info("[subcase {}] solution accuracy: estimated relative error {:.2e} after {} "
                   "refinement correction(s) (unrefined {:.2e}); epsilon {:.3e}, relative "
                   "residual {:.3e}, backward error {:.3e}",
                   sc.id, rep->estimated_error, rep->corrections, rep->initial_error,
                   rep->epsilon, rep->relative_residual, rep->backward_error);
    }
    summary.solution.add(
        std::format("Subcase {}", sc.id),
        refined ? std::format("{} equation{}; estimated relative error {:.2e} after {} "
                              "refinement correction(s), epsilon {:.2e}",
                              csr.n, csr.n == 1 ? "" : "s", rep->estimated_error,
                              rep->corrections, rep->epsilon)
                : std::format("{} equation{}; epsilon {:.2e}, backward error {:.2e}", csr.n,
                              csr.n == 1 ? "" : "s", rep->epsilon, rep->backward_error));
    constexpr double kAccuracyWarning = 1e-6;
    constexpr double kBackwardErrorWarning = 1e-10;
    if (indicates_mechanism(*rep, refined)) {
      notices.add(std::format(
          "the solution does not equilibrate the load ({}): the "
          "stiffness matrix is singular or too ill-conditioned for double "
          "precision; the model likely has an unrestrained mechanism loaded by the "
          "applied forces, and results are not meaningful",
          refined ? std::format("estimated relative error {:.2e} after refinement",
                                rep->estimated_error)
                  : std::format("epsilon {:.2e}", rep->epsilon)));
    } else if (refined && !rep->converged && rep->estimated_error > kAccuracyWarning) {
      notices.add(std::format("the stiffness matrix is very ill-conditioned: the solution "
                              "is only accurate to about {:.0e} (relative)",
                              rep->estimated_error));
    } else if (!refined && std::abs(rep->epsilon) > kAccuracyWarning) {
      notices.add(std::format("the stiffness matrix is very ill-conditioned (epsilon "
                              "{:.1e}): the solution may only be accurate to about that "
                              "(relative); --refine (or PARAM,REFINE,YES) refines it to "
                              "full accuracy",
                              rep->epsilon));
    } else if (!refined && rep->backward_error > kBackwardErrorWarning) {
      notices.add(std::format("the solve was not backward stable (backward error {:.2e})",
                              rep->backward_error));
    }
  }

  // 5. Recover full displacement vector (free + dep DOFs)
  int n_full = mpc_handler.full_dof_map().num_free_dofs();
  std::vector<double> u_free(static_cast<size_t>(n_full), 0.0);
  mpc_handler.recover_dependent_dofs(u_free, u_reduced);

  // 6. Recover results (displacements in CD frame + element stresses)
  SubCaseResults result = recover_results(model, sc,
                                         mpc_handler.full_dof_map(),
                                         u_free);
  const auto t7 = Clock::now();
  spdlog::debug("[subcase {}] recover_results: {:.3f} ms", sc.id, Ms(t7 - t6).count());

  // Recover the physical residual and constraint forces once, independently
  // of which combination of force outputs was requested.
  const bool want_spc = sc.spc_force_print || sc.spc_force_plot;
  const bool want_mpc = sc.mpc_force_print || sc.mpc_force_plot;
  const bool want_gp = sc.gpforce_print || sc.gpforce_plot;
  if (want_spc || want_mpc || want_gp) {
    GridForceOutput forces = compute_grid_forces(
        model, sc, mpc_handler, u_free, want_spc, want_mpc, want_gp);
    if (forces.error) {
      notices.add(std::format("constraint forces unavailable: {}", *forces.error));
      result.constraint_force_error = std::move(*forces.error);
    }
    result.spc_forces = std::move(forces.spc_forces);
    result.mpc_forces = std::move(forces.mpc_forces);
    result.gp_forces = std::move(forces.gp_forces);
    const auto t8 = Clock::now();
    spdlog::debug("[subcase {}] recover_forces: {:.3f} ms", sc.id,
                  Ms(t8 - t7).count());
  }

  spdlog::debug("[subcase {}] total: {:.3f} ms", sc.id, Ms(t7 - t0).count());

  const std::string context = std::format("Subcase {}", sc.id);
  collect_notices(summary, backend_->notices().take(), context);
  collect_notices(summary, notices.take(), context);
  return result;
}

DofMap LinearStaticSolver::build_dof_map(const Model &model,
                                         const SubCase &sc) {
  return build_analysis_dof_map(model, sc);
}

void LinearStaticSolver::build_mpc_system(const Model &model,
                                           const SubCase &sc,
                                           DofMap &dof_map,
                                           MpcHandler &mpc_handler) {
  build_analysis_mpc_system(model, sc, dof_map, mpc_handler,
                            /*use_spc_values=*/true);
}

void LinearStaticSolver::assemble(const Model &model, const SubCase & /*sc*/,
                                  const MpcHandler &mpc_handler,
                                  SparseMatrixBuilder &K_builder,
                                  std::vector<double> &F) {
  K_builder.reserve_triplets(detail::estimate_triplet_capacity(model));
  detail::assemble_element_matrix(
      model, mpc_handler, K_builder,
      [](const ElementBase &elem) { return elem.stiffness_matrix(); });

  if (mpc_handler.has_affine_offsets()) {
    const DofMap& full_dof_map = mpc_handler.full_dof_map();
    for (const auto& elem_data : model.elements) {
      auto elem = make_element(elem_data, model);
      const auto gdofs = elem->global_dof_indices(full_dof_map);
      const auto ke = detail::to_row_major(elem->stiffness_matrix());
      mpc_handler.apply_prescribed_displacement_load(gdofs, ke, F);
    }
  }
}

void LinearStaticSolver::apply_point_loads(const Model &model,
                                           const SubCase &sc,
                                           const MpcHandler &mpc_handler,
                                           std::vector<double> &F) {
  const DofMap &dof_map = mpc_handler.full_dof_map();
  for (const auto &[lp, load_scale] : model.loads_for_set(sc.load_set)) {
    std::visit(
        [&](const auto &load) {
          using T = std::decay_t<decltype(load)>;

          if constexpr (std::is_same_v<T, ForceLoad>) {
            // Rotate force from CID to basic if CID ≠ 0
            Vec3 force{load_scale * load.scale * load.direction.x,
                       load_scale * load.scale * load.direction.y,
                       load_scale * load.scale * load.direction.z};
            if (load.cid.value != 0) {
              auto cs_it = model.coord_systems.find(load.cid);
              if (cs_it != model.coord_systems.end()) {
                // Force in local CID frame → basic frame
                const Vec3& node_pos = model.node(load.node).position;
                Mat3 T3 = rotation_matrix(cs_it->second, node_pos);
                force = apply_rotation(T3, force);
              }
            }

            std::array<EqIndex, 6> full_eqs;
            dof_map.global_indices(load.node, full_eqs);
            std::vector<EqIndex> gdofs(full_eqs.begin(), full_eqs.end());
            double fe[6] = {force.x, force.y, force.z, 0, 0, 0};
            mpc_handler.apply_to_force(gdofs,
                std::span<const double>(fe, 6), F);

          } else if constexpr (std::is_same_v<T, SloadLoad>) {
            const Vec3 force = sload_force(model, load, load_scale);
            std::array<EqIndex, 6> full_eqs;
            dof_map.global_indices(load.node, full_eqs);
            std::vector<EqIndex> gdofs(full_eqs.begin(), full_eqs.end());
            double fe[6] = {force.x, force.y, force.z, 0, 0, 0};
            mpc_handler.apply_to_force(gdofs,
                std::span<const double>(fe, 6), F);

          } else if constexpr (std::is_same_v<T, MomentLoad>) {
            Vec3 moment{load_scale * load.scale * load.direction.x,
                        load_scale * load.scale * load.direction.y,
                        load_scale * load.scale * load.direction.z};
            if (load.cid.value != 0) {
              auto cs_it = model.coord_systems.find(load.cid);
              if (cs_it != model.coord_systems.end()) {
                const Vec3& node_pos = model.node(load.node).position;
                Mat3 T3 = rotation_matrix(cs_it->second, node_pos);
                moment = apply_rotation(T3, moment);
              }
            }
            std::array<EqIndex, 6> full_eqs;
            dof_map.global_indices(load.node, full_eqs);
            std::vector<EqIndex> gdofs(full_eqs.begin(), full_eqs.end());
            double fe[6] = {0, 0, 0, moment.x, moment.y, moment.z};
            mpc_handler.apply_to_force(gdofs,
                std::span<const double>(fe, 6), F);
          }
        },
        *lp);
  }
}

void LinearStaticSolver::apply_inertial_loads(const Model &model,
                                              const SubCase &sc,
                                              const MpcHandler &mpc_handler,
                                              std::vector<double> &F) {
  const double wtmass = wtmass_scale(model);
  std::unordered_map<NodeId, Vec3> nodal_accels;
  bool has_inertial_load = false;

  auto add_accel_to_node = [&](NodeId nid, const Vec3 &accel) {
    nodal_accels[nid] = nodal_accels[nid] + accel;
  };

  for (const auto &[lp, load_scale] : model.loads_for_set(sc.load_set)) {
    std::visit(
        [&](const auto &load) {
          using T = std::decay_t<decltype(load)>;
          if constexpr (std::is_same_v<T, GravLoad>) {
            has_inertial_load = true;
            for (const auto &[nid, gp] : model.nodes) {
              if (gp.scalar)
                continue; // rigid-body acceleration does not move a scalar point
              const Vec3 accel =
                  load_direction_in_basic(model, load.cid,
                                          load.direction * (load_scale * load.scale),
                                          gp.position);
              add_accel_to_node(nid, accel);
            }
          } else if constexpr (std::is_same_v<T, Accel1Load>) {
            has_inertial_load = true;
            for (NodeId nid : load.nodes) {
              const Vec3 accel = load_direction_in_basic(
                  model, load.cid, load.direction * (load_scale * load.scale),
                  model.node(nid).position);
              add_accel_to_node(nid, accel);
            }
          } else if constexpr (std::is_same_v<T, AccelLoad>) {
            throw SolverError(std::format(
                "ACCEL load set {} is parsed but not implemented",
                load.sid.value));
          }
        },
        *lp);
  }

  if (!has_inertial_load)
    return;

  const DofMap &dof_map = mpc_handler.full_dof_map();
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    LocalKe mass = element_mass_matrix(*elem, model);
    if (mass.rows() == 0)
      continue;
    mass *= wtmass;
    if (mass.cwiseAbs().maxCoeff() < 1e-30)
      continue;

    LocalFe accel = LocalFe::Zero(elem->num_dofs());
    switch (elem_data.type) {
    case ElementType::CQUAD4:
    case ElementType::CTRIA3:
    case ElementType::CBAR:
    case ElementType::CBEAM:
    case ElementType::CROD:
    case ElementType::CBUSH:
    case ElementType::CONM2:
      for (size_t i = 0; i < elem_data.nodes.size(); ++i) {
        const Vec3 a = nodal_accels[elem_data.nodes[i]];
        accel(6 * static_cast<int>(i) + 0) = a.x;
        accel(6 * static_cast<int>(i) + 1) = a.y;
        accel(6 * static_cast<int>(i) + 2) = a.z;
      }
      break;
    case ElementType::CHEXA8:
    case ElementType::CHEXA20:
    case ElementType::CTETRA4:
    case ElementType::CTETRA10:
    case ElementType::CPENTA6:
      for (size_t i = 0; i < elem_data.nodes.size(); ++i) {
        const Vec3 a = nodal_accels[elem_data.nodes[i]];
        accel(3 * static_cast<int>(i) + 0) = a.x;
        accel(3 * static_cast<int>(i) + 1) = a.y;
        accel(3 * static_cast<int>(i) + 2) = a.z;
      }
      break;
    case ElementType::CELAS1:
    case ElementType::CELAS2:
      break;
    case ElementType::CMASS1:
    case ElementType::CMASS2:
      for (size_t i = 0; i < elem_data.nodes.size(); ++i) {
        const auto it = nodal_accels.find(elem_data.nodes[i]);
        if (it == nodal_accels.end())
          continue;
        accel(static_cast<int>(i)) =
            component_value(it->second, elem_data.components[i]);
      }
      break;
    case ElementType::CHBDY:
      continue;  // CHBDY boundary elements have no inertial load contribution
    }

    const LocalFe fe = mass * accel;
    if (fe.cwiseAbs().maxCoeff() < 1e-30)
      continue;
    const auto gdofs = elem->global_dof_indices(dof_map);
    std::vector<double> fe_vec(fe.data(), fe.data() + fe.size());
    mpc_handler.apply_to_force(gdofs, fe_vec, F);
  }
}

void LinearStaticSolver::apply_thermal_loads(
    const Model &model, const SubCase &sc,
    const MpcHandler &mpc_handler,
    SparseMatrixBuilder & /*K_builder*/, std::vector<double> &F) {
  // Build nodal temperature map from TEMP cards and/or TEMPD defaults.
  // TEMPERATURE(LOAD) selects the temperature set; if not specified, fall back
  // to the structural load set for backward compatibility.
  std::unordered_map<NodeId, double> nodal_temps;

  int temp_set = sc.temp_load_set;
  if (temp_set == 0) temp_set = sc.load_set.value; // backward compat

  // TEMP/TEMPD are always looked up by direct SID match against the temperature
  // set — they do not participate in LOAD card combination.
  for (const auto &load : model.loads) {
    if (const TempLoad *tl = std::get_if<TempLoad>(&load)) {
      if (tl->sid == LoadSetId(temp_set))
        nodal_temps[tl->node] = tl->temperature;
    }
  }

  // TEMPD (default temperature for all nodes not covered by individual TEMP cards)
  auto tempd_it = model.tempd.find(temp_set);
  if (tempd_it != model.tempd.end()) {
    double T_default = tempd_it->second;
    for (const auto& [nid, _] : model.nodes) {
      if (nodal_temps.find(nid) == nodal_temps.end())
        nodal_temps[nid] = T_default;
    }
  }

  if (nodal_temps.empty())
    return;

  const DofMap &dof_map = mpc_handler.full_dof_map();
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    auto node_ids = elem->node_ids();
    const int nn = static_cast<int>(node_ids.size());

    // Reference temperature from the element's material (MAT1 TREF field)
    MaterialId mid{0};
    if (elem_data.type != ElementType::CELAS2 &&
        elem_data.type != ElementType::CMASS2 &&
        elem_data.type != ElementType::CONM2) {
      const auto &prop = model.property(elem_data.pid);
      mid = property_material_id(prop);
    }
    const bool is_shell = elem_data.type == ElementType::CQUAD4 ||
                          elem_data.type == ElementType::CTRIA3;
    double elem_t_ref =
        is_shell ? shell_reference_temperature(model, elem_data)
                 : ((mid.value != 0) ? model.material(mid).ref_temp : 0.0);

    std::vector<double> temps(static_cast<size_t>(nn));
    for (int i = 0; i < nn; ++i) {
      auto it = nodal_temps.find(node_ids[i]);
      temps[static_cast<size_t>(i)] =
          (it != nodal_temps.end()) ? it->second : elem_t_ref;
    }

    LocalFe fe = elem->thermal_load(temps, elem_t_ref);
    auto gdofs = elem->global_dof_indices(dof_map);
    std::vector<double> fe_vec(fe.data(), fe.data() + fe.size());

    mpc_handler.apply_to_force(gdofs, fe_vec, F);
  }
}

namespace {

/// The (node, component) -> enforced-displacement map for the active SPC set.
/// Component is 0-based (0=T1 .. 5=R3).  Values are in the basic frame.
std::unordered_map<NodeId, std::array<SpcDofState, 6>>
active_spc_state(const Model &model, const SubCase &sc) {
  std::unordered_map<NodeId, std::array<SpcDofState, 6>> spc;
  for (const Spc &spc_c : analysis_spcs(model, sc)) {
    auto node_it = model.nodes.find(spc_c.node);
    if (node_it == model.nodes.end())
      continue;
    const GridPoint &gp = node_it->second;

    // Basic-frame enforced displacement vector at this node.
    std::array<double, 6> u_spc{};
    for (int d = 0; d < 6; ++d)
      if (spc_c.dofs.has(d + 1))
        u_spc[static_cast<size_t>(d)] = spc_c.value;

    if (gp.cd.value != 0) {
      auto cs_it = model.coord_systems.find(gp.cd);
      if (cs_it != model.coord_systems.end()) {
        // u_basic = R * u_cd for translations and rotations separately.
        const Mat3 T3 = rotation_matrix(cs_it->second, gp.position);
        std::array<double, 6> u_basic{};
        for (int comp = 0; comp < 3; ++comp) {
          double t = 0.0;
          for (int j = 0; j < 3; ++j)
            t += T3(comp, j) * u_spc[static_cast<size_t>(j)];
          u_basic[static_cast<size_t>(comp)] = t;
          t = 0.0;
          for (int j = 0; j < 3; ++j)
            t += T3(comp, j) * u_spc[static_cast<size_t>(j + 3)];
          u_basic[static_cast<size_t>(comp + 3)] = t;
        }
        u_spc = u_basic;
      }
    }

    std::array<SpcDofState, 6> &state = spc[spc_c.node];
    if (gp.cd.value != 0) {
      // SPC components are in the grid's CD frame, not basic: accumulate the
      // full basic residual K u - F at this grid (every basic component), and
      // let compute_spc_forces rotate it to CD and keep the constrained
      // components.  Basic enforced values R u_cd are exact for any basic
      // DOF the constraint fixes directly.
      for (int d = 0; d < 6; ++d) {
        auto &st = state[static_cast<size_t>(d)];
        if (!st.active)
          st.value = u_spc[static_cast<size_t>(d)];
        else
          st.value += u_spc[static_cast<size_t>(d)];
        st.active = true;
      }
      continue;
    }
    for (int d = 0; d < 6; ++d) {
      if (spc_c.dofs.has(d + 1)) {
        state[static_cast<size_t>(d)].active = true;
        state[static_cast<size_t>(d)].value = u_spc[static_cast<size_t>(d)];
      }
    }
  }
  return spc;
}

/// Subtract applied point loads (FORCE/MOMENT) at SPC'd DOFs from `react`.
/// Memory: `react` is keyed by (node, component).
void subtract_spc_point_loads(const Model &model, const SubCase &sc,
                              const std::unordered_map<NodeId,
                                  std::array<SpcDofState, 6>> &spc,
                              std::unordered_map<NodeId,
                                  std::array<double, 6>> &react) {
  for (const auto &[lp, load_scale] : model.loads_for_set(sc.load_set)) {
    std::visit(
        [&](const auto &load) {
          using T = std::decay_t<decltype(load)>;
          if constexpr (std::is_same_v<T, ForceLoad>) {
            Vec3 force{load_scale * load.scale * load.direction.x,
                       load_scale * load.scale * load.direction.y,
                       load_scale * load.scale * load.direction.z};
            if (load.cid.value != 0) {
              auto cs_it = model.coord_systems.find(load.cid);
              if (cs_it != model.coord_systems.end()) {
                const Vec3 &node_pos = model.node(load.node).position;
                force = apply_rotation(rotation_matrix(cs_it->second, node_pos),
                                       force);
              }
            }
            auto spc_it = spc.find(load.node);
            if (spc_it == spc.end())
              return;
            for (int d = 0; d < 3; ++d) {
              if (!spc_it->second[static_cast<size_t>(d)].active)
                continue;
              const double v = (d == 0) ? force.x : ((d == 1) ? force.y : force.z);
              react[load.node][static_cast<size_t>(d)] -= v;
            }
          } else if constexpr (std::is_same_v<T, SloadLoad>) {
            const Vec3 force = sload_force(model, load, load_scale);
            auto spc_it = spc.find(load.node);
            if (spc_it == spc.end())
              return;
            for (int d = 0; d < 3; ++d) {
              if (!spc_it->second[static_cast<size_t>(d)].active)
                continue;
              const double v = (d == 0) ? force.x : ((d == 1) ? force.y : force.z);
              react[load.node][static_cast<size_t>(d)] -= v;
            }
          } else if constexpr (std::is_same_v<T, MomentLoad>) {
            Vec3 moment{load_scale * load.scale * load.direction.x,
                        load_scale * load.scale * load.direction.y,
                        load_scale * load.scale * load.direction.z};
            if (load.cid.value != 0) {
              auto cs_it = model.coord_systems.find(load.cid);
              if (cs_it != model.coord_systems.end()) {
                const Vec3 &node_pos = model.node(load.node).position;
                moment = apply_rotation(rotation_matrix(cs_it->second, node_pos),
                                        moment);
              }
            }
            auto spc_it = spc.find(load.node);
            if (spc_it == spc.end())
              return;
            for (int d = 3; d < 6; ++d) {
              if (!spc_it->second[static_cast<size_t>(d)].active)
                continue;
              const double v = (d == 3) ? moment.x
                               : ((d == 4) ? moment.y : moment.z);
              react[load.node][static_cast<size_t>(d)] -= v;
            }
          }
        },
        *lp);
  }
}

/// Subtract inertial (GRAV/ACCEL1) loads at SPC'd DOFs from `react`.
void subtract_spc_inertial_loads(const Model &model, const SubCase &sc,
                                 const std::unordered_map<NodeId,
                                     std::array<SpcDofState, 6>> &spc,
                                 std::unordered_map<NodeId,
                                     std::array<double, 6>> &react) {
  const double wtmass = wtmass_scale(model);
  std::unordered_map<NodeId, Vec3> nodal_accels;
  bool has_inertial_load = false;

  auto add_accel = [&](NodeId nid, const Vec3 &accel) {
    nodal_accels[nid] = nodal_accels[nid] + accel;
  };

  for (const auto &[lp, load_scale] : model.loads_for_set(sc.load_set)) {
    std::visit(
        [&](const auto &load) {
          using T = std::decay_t<decltype(load)>;
          if constexpr (std::is_same_v<T, GravLoad>) {
            has_inertial_load = true;
            for (const auto &[nid, gp] : model.nodes) {
              if (gp.scalar)
                continue; // rigid-body acceleration does not move a scalar point
              const Vec3 accel = load_direction_in_basic(
                  model, load.cid, load.direction * (load_scale * load.scale),
                  gp.position);
              add_accel(nid, accel);
            }
          } else if constexpr (std::is_same_v<T, Accel1Load>) {
            has_inertial_load = true;
            for (NodeId nid : load.nodes) {
              const Vec3 accel = load_direction_in_basic(
                  model, load.cid, load.direction * (load_scale * load.scale),
                  model.node(nid).position);
              add_accel(nid, accel);
            }
          }
        },
        *lp);
  }
  if (!has_inertial_load)
    return;

  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    LocalKe mass = element_mass_matrix(*elem, model);
    if (mass.rows() == 0)
      continue;
    mass *= wtmass;
    if (mass.cwiseAbs().maxCoeff() < 1e-30)
      continue;

    LocalFe accel = LocalFe::Zero(elem->num_dofs());
    switch (elem_data.type) {
    case ElementType::CQUAD4:
    case ElementType::CTRIA3:
    case ElementType::CBAR:
    case ElementType::CBEAM:
    case ElementType::CROD:
    case ElementType::CBUSH:
    case ElementType::CONM2:
      for (size_t i = 0; i < elem_data.nodes.size(); ++i) {
        const Vec3 a = nodal_accels[elem_data.nodes[i]];
        accel(6 * static_cast<int>(i) + 0) = a.x;
        accel(6 * static_cast<int>(i) + 1) = a.y;
        accel(6 * static_cast<int>(i) + 2) = a.z;
      }
      break;
    case ElementType::CHEXA8:
    case ElementType::CHEXA20:
    case ElementType::CTETRA4:
    case ElementType::CTETRA10:
    case ElementType::CPENTA6:
      for (size_t i = 0; i < elem_data.nodes.size(); ++i) {
        const Vec3 a = nodal_accels[elem_data.nodes[i]];
        accel(3 * static_cast<int>(i) + 0) = a.x;
        accel(3 * static_cast<int>(i) + 1) = a.y;
        accel(3 * static_cast<int>(i) + 2) = a.z;
      }
      break;
    case ElementType::CELAS1:
    case ElementType::CELAS2:
      break;
    case ElementType::CMASS1:
    case ElementType::CMASS2:
      for (size_t i = 0; i < elem_data.nodes.size(); ++i) {
        const auto it = nodal_accels.find(elem_data.nodes[i]);
        if (it == nodal_accels.end())
          continue;
        accel(static_cast<int>(i)) =
            component_value(it->second, elem_data.components[i]);
      }
      break;
    case ElementType::CHBDY:
      continue;
    }

    const LocalFe fe = mass * accel;
    if (fe.cwiseAbs().maxCoeff() < 1e-30)
      continue;

    const auto identity = local_dof_identity(elem_data);
    for (int li = 0; li < elem->num_dofs(); ++li) {
      const auto [nid, comp] = identity[static_cast<size_t>(li)];
      const auto spc_it = spc.find(nid);
      if (spc_it == spc.end() ||
          !spc_it->second[static_cast<size_t>(comp)].active)
        continue;
      react[nid][static_cast<size_t>(comp)] -=
          fe(static_cast<Eigen::Index>(li));
    }
  }
}

/// Subtract thermal element loads at SPC'd DOFs from `react`.
/// Nodal temperatures of a subcase's thermal load set (TEMP, with TEMPD as
/// the default for unlisted grids); empty when the subcase has none.
std::unordered_map<NodeId, double>
subcase_thermal_temperatures(const Model &model, const SubCase &sc) {
  int temp_set = sc.temp_load_set;
  if (temp_set == 0)
    temp_set = sc.load_set.value;

  std::unordered_map<NodeId, double> nodal_temps;
  for (const auto &load : model.loads) {
    if (const TempLoad *tl = std::get_if<TempLoad>(&load)) {
      if (tl->sid == LoadSetId(temp_set))
        nodal_temps[tl->node] = tl->temperature;
    }
  }
  auto tempd_it = model.tempd.find(temp_set);
  if (tempd_it != model.tempd.end()) {
    const double T_default = tempd_it->second;
    for (const auto &[nid, _] : model.nodes) {
      if (!nodal_temps.contains(nid))
        nodal_temps[nid] = T_default;
    }
  }
  return nodal_temps;
}

/// Equivalent thermal load vector of one element (element DOF order, basic
/// frame) for the given nodal temperatures.
LocalFe element_thermal_vector(const ElementBase &elem,
                               const ElementData &elem_data, const Model &model,
                               const std::unordered_map<NodeId, double> &nodal_temps) {
  auto node_ids = elem.node_ids();
  const int nn = static_cast<int>(node_ids.size());

  MaterialId mid{0};
  if (elem_data.type != ElementType::CELAS2 &&
      elem_data.type != ElementType::CMASS2 &&
      elem_data.type != ElementType::CONM2) {
    const auto &prop = model.property(elem_data.pid);
    mid = property_material_id(prop);
  }
  const bool is_shell = elem_data.type == ElementType::CQUAD4 ||
                        elem_data.type == ElementType::CTRIA3;
  const double elem_t_ref =
      is_shell ? shell_reference_temperature(model, elem_data)
               : ((mid.value != 0) ? model.material(mid).ref_temp : 0.0);

  std::vector<double> temps(static_cast<size_t>(nn));
  for (int i = 0; i < nn; ++i) {
    auto it = nodal_temps.find(node_ids[static_cast<size_t>(i)]);
    temps[static_cast<size_t>(i)] =
        (it != nodal_temps.end()) ? it->second : elem_t_ref;
  }
  return elem.thermal_load(temps, elem_t_ref);
}

void subtract_spc_thermal_loads(const Model &model, const SubCase &sc,
                                const std::unordered_map<NodeId,
                                    std::array<SpcDofState, 6>> &spc,
                                std::unordered_map<NodeId,
                                    std::array<double, 6>> &react) {
  const auto nodal_temps = subcase_thermal_temperatures(model, sc);
  if (nodal_temps.empty())
    return;

  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    const LocalFe fe =
        element_thermal_vector(*elem, elem_data, model, nodal_temps);
    const auto identity = local_dof_identity(elem_data);
    for (int li = 0; li < elem->num_dofs(); ++li) {
      const auto [nid, comp] = identity[static_cast<size_t>(li)];
      const auto spc_it = spc.find(nid);
      if (spc_it == spc.end() ||
          !spc_it->second[static_cast<size_t>(comp)].active)
        continue;
      react[nid][static_cast<size_t>(comp)] -=
          fe(static_cast<Eigen::Index>(li));
    }
  }
}

/// Rotate a basic-frame grid 6-vector (forces, moments) into the grid's
/// global (CD) system.
std::array<double, 6> to_grid_frame(const Model &model, const NodeId nid,
                                    const std::array<double, 6> &v) {
  const GridPoint &gp = model.node(nid);
  if (gp.cd.value == 0)
    return v;
  const auto cs_it = model.coord_systems.find(gp.cd);
  if (cs_it == model.coord_systems.end()) {
    throw SolverError(std::format(
        "GRID {}: displacement coordinate system {} not found", nid.value,
        gp.cd.value));
  }
  const Mat3 T3 = rotation_matrix(cs_it->second, gp.position);
  std::array<double, 6> out{};
  for (int block = 0; block < 2; ++block)
    for (int i = 0; i < 3; ++i) {
      double x = 0.0;
      for (int j = 0; j < 3; ++j)
        x += T3(j, i) * v[static_cast<size_t>(3 * block + j)];
      out[static_cast<size_t>(3 * block + i)] = x;
    }
  return out;
}

} // namespace

LinearStaticSolver::GridForceOutput LinearStaticSolver::compute_grid_forces(
    const Model &model, const SubCase &sc, const MpcHandler &mpc_handler,
    const std::vector<double> &u_free, const bool want_spc,
    const bool want_mpc, const bool want_gp) const {
  GridForceOutput out;
  const DofMap &dof_map = mpc_handler.full_dof_map();

  // Enforced values for constrained DOFs, and an "every component of every
  // grid" map so the load helpers assemble the full applied load vector.
  const auto spc = active_spc_state(model, sc);
  std::unordered_map<NodeId, std::array<SpcDofState, 6>> all;
  all.reserve(model.nodes.size());
  for (const auto &[nid, _] : model.nodes) {
    std::array<SpcDofState, 6> st{};
    const auto it = spc.find(nid);
    for (int d = 0; d < 6; ++d) {
      st[static_cast<size_t>(d)].active = true;
      if (it != spc.end() && it->second[static_cast<size_t>(d)].active)
        st[static_cast<size_t>(d)].value = it->second[static_cast<size_t>(d)].value;
    }
    all.emplace(nid, st);
  }

  const auto nodal_temps = subcase_thermal_temperatures(model, sc);

  // Element internal forces K_e u_e - f_e,thermal (basic), per element and
  // grid, and their sum per grid.
  struct ElementForceAtGrid {
    ElementType etype;
    ElementId eid;
    std::array<double, 6> f{};
  };
  std::unordered_map<NodeId, std::vector<ElementForceAtGrid>> elem_rows;
  std::unordered_map<NodeId, std::array<double, 6>> internal;
  std::vector<const ElementData *> sorted_elements;
  sorted_elements.reserve(model.elements.size());
  for (const auto &e : model.elements)
    sorted_elements.push_back(&e);
  std::sort(sorted_elements.begin(), sorted_elements.end(),
            [](const ElementData *a, const ElementData *b) { return a->id < b->id; });

  for (const ElementData *ep : sorted_elements) {
    const ElementData &elem_data = *ep;
    auto elem = make_element(elem_data, model);
    const LocalKe ke = elem->stiffness_matrix();
    if (ke.size() == 0 || ke.cwiseAbs().maxCoeff() == 0.0)
      continue; // mass-only elements carry no internal force
    const auto gdofs = elem->global_dof_indices(dof_map);
    const auto identity = local_dof_identity(elem_data);
    const int ndof = elem->num_dofs();

    Eigen::VectorXd ue = Eigen::VectorXd::Zero(ndof);
    for (int li = 0; li < ndof; ++li) {
      const EqIndex eq = gdofs[static_cast<size_t>(li)];
      if (eq != CONSTRAINED_DOF && eq < static_cast<EqIndex>(u_free.size())) {
        ue(li) = u_free[static_cast<size_t>(eq)];
      } else {
        const auto [nid, comp] = identity[static_cast<size_t>(li)];
        const auto it = spc.find(nid);
        if (it != spc.end() && it->second[static_cast<size_t>(comp)].active)
          ue(li) = it->second[static_cast<size_t>(comp)].value;
      }
    }
    Eigen::VectorXd fe = ke * ue;
    if (!nodal_temps.empty())
      fe -= element_thermal_vector(*elem, elem_data, model, nodal_temps);

    std::unordered_map<NodeId, std::array<double, 6>> per_grid;
    for (int li = 0; li < ndof; ++li) {
      const auto [nid, comp] = identity[static_cast<size_t>(li)];
      if (is_omitted_grid(nid))
        continue; // an omitted midside grid of a CTETRA10/CHEXA20
      per_grid[nid][static_cast<size_t>(comp)] += fe(li);
      internal[nid][static_cast<size_t>(comp)] += fe(li);
    }
    if (want_gp) {
      for (const NodeId nid : elem_data.nodes) {
        const auto it = per_grid.find(nid);
        if (it == per_grid.end())
          continue;
        elem_rows[nid].push_back({elem_data.type, elem_data.id, it->second});
        per_grid.erase(it); // a grid listed twice gets one row
      }
    }
  }

  // Applied equivalent loads (mechanical) and thermal equivalent loads, per
  // grid in basic.  The helpers subtract into their map, so negate.
  std::unordered_map<NodeId, std::array<double, 6>> neg_applied, neg_thermal;
  subtract_spc_point_loads(model, sc, all, neg_applied);
  subtract_spc_inertial_loads(model, sc, all, neg_applied);
  compute_spc_pressure_loads(model, sc, dof_map, all, neg_applied);
  subtract_spc_thermal_loads(model, sc, all, neg_thermal);

  auto get = [](const std::unordered_map<NodeId, std::array<double, 6>> &m,
                const NodeId nid) {
    const auto it = m.find(nid);
    return (it != m.end()) ? it->second : std::array<double, 6>{};
  };

  // Grids coupled by multipoint constraints of this subcase.
  std::set<NodeId> mpc_grids;
  if (sc.mpc_set.value != 0)
    for (const Mpc *mpc : model.mpcs_for_set(sc.mpc_set))
      for (const auto &t : mpc->terms)
        mpc_grids.insert(t.node);
  for (const auto &rbe2 : model.rbe2s) {
    mpc_grids.insert(rbe2.gn);
    mpc_grids.insert(rbe2.gm.begin(), rbe2.gm.end());
  }
  for (const auto &rbe3 : model.rbe3s) {
    mpc_grids.insert(rbe3.ref_node);
    for (const auto &wg : rbe3.weight_groups)
      mpc_grids.insert(wg.nodes.begin(), wg.nodes.end());
  }

  // Physical nodal residual before constraint elimination. Thermal loads
  // are already included in internal; neg_applied contains mechanical loads.
  NodalForces residuals = internal;
  for (const auto& [nid, force] : neg_applied)
    for (size_t d = 0; d < 6; ++d)
      residuals[nid][d] += force[d];
  const auto recovery = mpc_handler.recover_mpc_forces(residuals);
  if (const auto* error = std::get_if<MpcForceRecoveryError>(&recovery)) {
    // Without valid MPC forces neither the SPC reactions nor the force
    // balance can be attributed. Report no constraint forces rather than
    // partial ones; the displacement solution is unaffected.
    out.error = error->message;
    return out;
  }
  const auto& mpc_forces = std::get<NodalForces>(recovery);

  std::unordered_map<NodeId, DofSet> constrained;
  for (const Spc &c : analysis_spcs(model, sc))
    constrained[c.node].mask |= c.dofs.mask;
  // AUTOSPC adds a component at every rod grid, so its grids are reported only
  // when they carry a reaction.
  std::unordered_set<NodeId> user_constrained;
  for (const Spc &c : model.structural_spcs(sc.spc_set))
    user_constrained.insert(c.node);

  std::vector<NodeId> grids;
  grids.reserve(model.nodes.size());
  for (const auto &[nid, _] : model.nodes)
    grids.push_back(nid);
  std::sort(grids.begin(), grids.end());

  auto any_nonzero = [](const std::array<double, 6> &v) {
    return std::any_of(v.begin(), v.end(), [](double x) { return x != 0.0; });
  };

  for (const NodeId nid : grids) {
    // R = K u - F = sum(K_e u_e - f_th) - F_applied (the thermal part of F
    // is already inside the element terms).
    const auto f_app = get(neg_applied, nid);  // = -F_applied
    const auto f_th = get(neg_thermal, nid);   // = -F_thermal
    const auto residual = get(residuals, nid);
    const auto mpc_basic = get(mpc_forces, nid);
    SpcForce sf;
    const auto c_it = constrained.find(nid);
    const bool has_spc = c_it != constrained.end();
    if (has_spc) {
      sf.node = nid;
      sf.coord = model.node(nid).cd;
      for (size_t d = 0; d < 6; ++d)
        sf.f[d] = residual[d] - mpc_basic[d];
      sf.f = to_grid_frame(model, nid, sf.f);
      for (int d = 0; d < 6; ++d)
        if (!c_it->second.has(d + 1))
          sf.f[static_cast<size_t>(d)] = 0.0;
      if (want_spc && (user_constrained.contains(nid) || any_nonzero(sf.f)))
        out.spc_forces.push_back(sf);
    }

    std::array<double, 6> mpc_force{};
    bool has_mpc = false;
    if (mpc_grids.contains(nid)) {
      mpc_force = to_grid_frame(model, nid, mpc_basic);
      has_mpc = true;
      if (want_mpc) {
        SpcForce mf;
        mf.node = nid;
        mf.f = mpc_force;
        mf.coord = model.node(nid).cd;
        out.mpc_forces.push_back(mf);
      }
    }

    if (!want_gp)
      continue;
    GridPointForceBalance gb;
    gb.node = nid;
    gb.coord = model.node(nid).cd;
    auto add_row = [&](GridForceContribution::Kind kind,
                       const std::array<double, 6> &f, ElementType etype,
                       ElementId eid) {
      gb.rows.push_back({kind, etype, eid, f});
      for (size_t d = 0; d < 6; ++d)
        gb.totals[d] += f[d];
    };
    // APPLIED holds every equivalent load including the thermal one, which
    // the THERMAL row then backs out (MYSTRAN's layout).
    std::array<double, 6> applied{};
    std::array<double, 6> minus_thermal{};
    for (size_t d = 0; d < 6; ++d) {
      applied[d] = -f_app[d] - f_th[d];
      minus_thermal[d] = f_th[d];
    }
    if (any_nonzero(applied))
      add_row(GridForceContribution::Kind::Applied,
              to_grid_frame(model, nid, applied), ElementType::CBAR, ElementId{0});
    if (any_nonzero(minus_thermal))
      add_row(GridForceContribution::Kind::Thermal,
              to_grid_frame(model, nid, minus_thermal), ElementType::CBAR,
              ElementId{0});
    if (has_spc)
      add_row(GridForceContribution::Kind::Spc, sf.f, ElementType::CBAR,
              ElementId{0});
    if (has_mpc && any_nonzero(mpc_force))
      add_row(GridForceContribution::Kind::Mpc, mpc_force, ElementType::CBAR,
              ElementId{0});
    if (const auto it = elem_rows.find(nid); it != elem_rows.end()) {
      for (const auto &row : it->second) {
        std::array<double, 6> on_grid{};
        for (size_t d = 0; d < 6; ++d)
          on_grid[d] = -row.f[d];
        add_row(GridForceContribution::Kind::Element,
                to_grid_frame(model, nid, on_grid), row.etype, row.eid);
      }
    }
    if (!gb.rows.empty())
      out.gp_forces.push_back(std::move(gb));
  }
  return out;
}

SubCaseResults
LinearStaticSolver::recover_results(const Model &model, const SubCase &sc,
                                    const DofMap &dof_map,
                                    const std::vector<double> &u_free) {
  SubCaseResults res;
  res.id = sc.id;
  res.label = sc.label;

  // ── Recover displacements ─────────────────────────────────────────────────
  std::vector<NodeId> sorted_nodes;
  sorted_nodes.reserve(model.nodes.size());
  for (const auto &[nid, _] : model.nodes)
    sorted_nodes.push_back(nid);
  std::sort(sorted_nodes.begin(), sorted_nodes.end());

  for (NodeId nid : sorted_nodes) {
    NodeDisplacement nd;
    nd.node = nid;
    nd.scalar = model.node(nid).scalar;
    for (int d = 0; d < 6; ++d) {
      EqIndex eq = dof_map.eq_index(nid, d);
      nd.d[d] = (eq != CONSTRAINED_DOF && eq < static_cast<int>(u_free.size()))
                    ? u_free[static_cast<size_t>(eq)]
                    : 0.0;
    }
    // Rotate displacements to CD frame for output
    const GridPoint &gp = model.node(nid);
    if (gp.cd.value != 0) {
      auto cs_it = model.coord_systems.find(gp.cd);
      if (cs_it != model.coord_systems.end()) {
        Mat3 T3 = rotation_matrix(cs_it->second, gp.position);
        // v_cd = T3^T * v_basic
        // T3^T[i][j] = T3[j][i]
        Vec3 u_basic{nd.d[0], nd.d[1], nd.d[2]};
        // v_cd[i] = sum_j T3[j][i] * v_basic[j]
        Vec3 u_cd{
            T3(0,0)*u_basic.x + T3(1,0)*u_basic.y + T3(2,0)*u_basic.z,
            T3(0,1)*u_basic.x + T3(1,1)*u_basic.y + T3(2,1)*u_basic.z,
            T3(0,2)*u_basic.x + T3(1,2)*u_basic.y + T3(2,2)*u_basic.z,
        };
        nd.d[0] = u_cd.x; nd.d[1] = u_cd.y; nd.d[2] = u_cd.z;

        Vec3 rot_basic{nd.d[3], nd.d[4], nd.d[5]};
        Vec3 rot_cd{
            T3(0,0)*rot_basic.x + T3(1,0)*rot_basic.y + T3(2,0)*rot_basic.z,
            T3(0,1)*rot_basic.x + T3(1,1)*rot_basic.y + T3(2,1)*rot_basic.z,
            T3(0,2)*rot_basic.x + T3(1,2)*rot_basic.y + T3(2,2)*rot_basic.z,
        };
        nd.d[3] = rot_cd.x; nd.d[4] = rot_cd.y; nd.d[5] = rot_cd.z;
        nd.coord = gp.cd;
      }
    }
    res.displacements.push_back(nd);
  }

  // ── Build nodal temperature map for thermal stress correction ────────────
  const std::unordered_map<NodeId, double> nodal_temps_rec =
      build_nodal_temperature_map(model, sc);

  if (sc.has_any_stress_output() || sc.has_any_strain_output()) {
    std::set<std::string> unsupported_types;
    for (const auto &elem_data : model.elements) {
      switch (elem_data.type) {
      case ElementType::CBUSH:
        unsupported_types.insert("CBUSH");
        break;
      case ElementType::CELAS1:
        unsupported_types.insert("CELAS1");
        break;
      case ElementType::CELAS2:
        unsupported_types.insert("CELAS2");
        break;
      case ElementType::CMASS1:
        unsupported_types.insert("CMASS1");
        break;
      case ElementType::CMASS2:
        unsupported_types.insert("CMASS2");
        break;
      default:
        break;
      }
    }

    if (!unsupported_types.empty()) {
      std::string families;
      for (const auto &name : unsupported_types) {
        if (!families.empty())
          families += ", ";
        families += name;
      }
      spdlog::warn(
          "[subcase {}] stress recovery is not implemented for element "
          "families [{}]; those results were skipped",
          sc.id, families);
    }
  }

  // ── Span loads (PLOAD1) per line element, for end-force recovery ────────
  std::unordered_map<ElementId, LocalFe> line_span_loads;
  {
    std::unordered_map<ElementId, const ElementData *> line_elements;
    for (const auto &elem : model.elements)
      if (elem.type == ElementType::CBAR || elem.type == ElementType::CBEAM)
        line_elements.emplace(elem.id, &elem);
    for (const auto &[lp, load_scale] : model.loads_for_set(sc.load_set)) {
      const auto *pload1 = std::get_if<Pload1Load>(lp);
      if (pload1 == nullptr)
        continue;
      const auto it = line_elements.find(pload1->element);
      if (it == line_elements.end())
        continue; // invalid targets are reported when loads are applied
      const LocalFe f =
          load_scale * compute_pload1_fixed_end_local(*it->second, model, *pload1);
      auto [slot, inserted] = line_span_loads.try_emplace(pload1->element, f);
      if (!inserted)
        slot->second += f;
    }
  }

  // ── Recover element stresses ──────────────────────────────────────────────
  for (const auto &elem_data : model.elements) {
    auto elem = make_element(elem_data, model);
    auto gdofs = elem->global_dof_indices(dof_map);

    const int nd_ = static_cast<int>(gdofs.size());
    Eigen::VectorXd ue = Eigen::VectorXd::Zero(nd_);
    for (int i = 0; i < nd_; ++i) {
      EqIndex eq = gdofs[i];
      if (eq != CONSTRAINED_DOF && eq < static_cast<int>(u_free.size()))
        ue(i) = u_free[static_cast<size_t>(eq)];
    }

    if (elem_data.type == ElementType::CBAR ||
        elem_data.type == ElementType::CBEAM) {
      const MaterialId mid = property_material_id(model.property(elem_data.pid));
      const Mat1 &mat_ = model.material(mid);
      const double avg_temp =
          average_temperature(elem_data.nodes, nodal_temps_rec, mat_.ref_temp);

      std::array<double, CBarBeamElement::NUM_DOFS> line_u{};
      for (int i = 0; i < CBarBeamElement::NUM_DOFS; ++i)
        line_u[static_cast<size_t>(i)] = ue(i);

      const CBarBeamElement line_elem(elem_data, model);
      const auto span_it = line_span_loads.find(elem_data.id);
      const std::span<const double> span_load =
          (span_it != line_span_loads.end())
              ? std::span<const double>(span_it->second.data(),
                                        static_cast<size_t>(span_it->second.size()))
              : std::span<const double>{};
      const auto recovery =
          line_elem.recover_stress(line_u, avg_temp, mat_.ref_temp, span_load);

      LineStress ls;
      ls.eid = elem_data.id;
      ls.etype = elem_data.type;
      ls.end_a.node = elem_data.nodes[0];
      ls.end_a.s = recovery.end_a.s;
      ls.end_a.axial = recovery.end_a.axial;
      ls.end_a.smax = recovery.end_a.smax;
      ls.end_a.smin = recovery.end_a.smin;
      ls.end_b.node = elem_data.nodes[1];
      ls.end_b.s = recovery.end_b.s;
      ls.end_b.axial = recovery.end_b.axial;
      ls.end_b.smax = recovery.end_b.smax;
      ls.end_b.smin = recovery.end_b.smin;
      ls.forces = {recovery.forces.m1a, recovery.forces.m2a,
                   recovery.forces.m1b, recovery.forces.m2b,
                   recovery.forces.v1,  recovery.forces.v2,
                   recovery.forces.axial, recovery.forces.torque};
      res.line_stresses.push_back(ls);
    } else if (elem_data.type == ElementType::CROD) {
      const CRodElement rod(elem_data, model);
      const MaterialId mid = property_material_id(model.property(elem_data.pid));
      const double avg_temp = average_temperature(
          elem_data.nodes, nodal_temps_rec, model.material(mid).ref_temp);
      const auto recovered = rod.recover(
          std::span<const double>(ue.data(), static_cast<size_t>(ue.size())),
          avg_temp);
      RodResult rr;
      rr.eid = elem_data.id;
      rr.axial_force = recovered.axial_force;
      rr.torque = recovered.torque;
      rr.axial_stress = recovered.axial_stress;
      rr.torsional_stress = recovered.torsional_stress;
      rr.axial_strain = recovered.axial_strain;
      rr.torsional_strain = recovered.torsional_strain;
      rr.axial_margin = recovered.axial_margin;
      rr.torsional_margin = recovered.torsional_margin;
      res.rod_results.push_back(rr);
    } else if (elem_data.type == ElementType::CQUAD4 ||
               elem_data.type == ElementType::CTRIA3) {
      PlateStress ps;
      ps.eid   = elem_data.id;
      ps.etype = elem_data.type;

      if (elem_data.type == ElementType::CQUAD4) {
        const ShellQuad4 quad(elem_data, model);
        const double t_ref = shell_reference_temperature(model, elem_data);
        std::array<double, 4> temps{};
        for (size_t n = 0; n < 4; ++n) {
          const auto it = nodal_temps_rec.find(elem_data.nodes[n]);
          temps[n] = (it != nodal_temps_rec.end()) ? it->second : t_ref;
        }
        const auto points = quad.recover(
            std::span<const double>(ue.data(), static_cast<size_t>(ue.size())),
            temps, t_ref);
        PlateStressPoint centroid;
        fill_shell_point(centroid, points[0]);
        ps.sx = centroid.sx;
        ps.sy = centroid.sy;
        ps.sxy = centroid.sxy;
        ps.mx = centroid.mx;
        ps.my = centroid.my;
        ps.mxy = centroid.mxy;
        ps.von_mises = centroid.von_mises;
        ps.nx = centroid.nx;
        ps.ny = centroid.ny;
        ps.nxy = centroid.nxy;
        ps.qx = centroid.qx;
        ps.qy = centroid.qy;
        ps.strain = centroid.strain;
        for (size_t n = 0; n < 4; ++n) {
          PlateStressPoint pt;
          pt.node = elem_data.nodes[n];
          fill_shell_point(pt, points[n + 1]);
          ps.nodal.push_back(pt);
        }
      } else {
        const ShellTria3 tria(elem_data, model);
        const double t_ref = shell_reference_temperature(model, elem_data);
        std::array<double, 3> temps{};
        for (size_t n = 0; n < 3; ++n) {
          const auto it = nodal_temps_rec.find(elem_data.nodes[n]);
          temps[n] = (it != nodal_temps_rec.end()) ? it->second : t_ref;
        }
        const auto points = tria.recover(
            std::span<const double>(ue.data(), static_cast<size_t>(ue.size())),
            temps, t_ref);
        PlateStressPoint centroid;
        fill_shell_point(centroid, points[0]);
        ps.sx = centroid.sx;
        ps.sy = centroid.sy;
        ps.sxy = centroid.sxy;
        ps.mx = centroid.mx;
        ps.my = centroid.my;
        ps.mxy = centroid.mxy;
        ps.von_mises = centroid.von_mises;
        ps.nx = centroid.nx;
        ps.ny = centroid.ny;
        ps.nxy = centroid.nxy;
        ps.qx = centroid.qx;
        ps.qy = centroid.qy;
        ps.strain = centroid.strain;
        for (size_t n = 0; n < 3; ++n) {
          PlateStressPoint pt;
          pt.node = elem_data.nodes[n];
          fill_shell_point(pt, points[n + 1]);
          ps.nodal.push_back(pt);
        }
      }
      res.plate_stresses.push_back(ps);
    } else if (elem_data.type == ElementType::CHEXA8 ||
               elem_data.type == ElementType::CHEXA20 ||
               elem_data.type == ElementType::CTETRA4 ||
               elem_data.type == ElementType::CTETRA10 ||
               elem_data.type == ElementType::CPENTA6) {
      SolidStress ss;
      ss.eid   = elem_data.id;
      ss.etype = elem_data.type;

      SolidStrain strain;
      strain.eid   = elem_data.id;
      strain.etype = elem_data.type;

      const auto &psol_ = std::get<PSolid>(model.property(elem_data.pid));
      const Mat1 &mat_ = model.material(psol_.mid);
      const Eigen::Matrix<double, 6, 6> D_ = solid_constitutive_matrix(mat_);

      const bool want_strain = sc.has_any_strain_output();

      SolidStressStrain center;
      auto eval_ctetra4 = [&]() {
        auto nc = [&]() -> std::array<Vec3, 4> {
          std::array<Vec3, 4> c;
          for (int i = 0; i < 4; ++i)
            c[i] = model.node(elem_data.nodes[static_cast<size_t>(i)]).position;
          return c;
        }();
        const double x1 = nc[0].x, y1 = nc[0].y, z1 = nc[0].z;
        const double x2 = nc[1].x, y2 = nc[1].y, z2 = nc[1].z;
        const double x3 = nc[2].x, y3 = nc[2].y, z3 = nc[2].z;
        const double x4 = nc[3].x, y4 = nc[3].y, z4 = nc[3].z;
        Eigen::Matrix4d A4;
        A4 << 1, x1, y1, z1, 1, x2, y2, z2, 1, x3, y3, z3, 1, x4, y4, z4;
        const double V6 = A4.determinant();
        Eigen::Matrix4d cofA = Eigen::Matrix4d::Zero();
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            Eigen::Matrix3d m3;
            int ri = 0;
            for (int r = 0; r < 4; ++r) {
              if (r == j)
                continue;
              int ci = 0;
              for (int c = 0; c < 4; ++c) {
                if (c == i)
                  continue;
                m3(ri, ci++) = A4(r, c);
              }
              ++ri;
            }
            cofA(i, j) = std::pow(-1.0, i + j) * m3.determinant();
          }
        }
        Eigen::MatrixXd B(6, 12);
        B.setZero();
        for (int n = 0; n < 4; ++n) {
          const double bx = cofA(1, n) / V6;
          const double by = cofA(2, n) / V6;
          const double bz = cofA(3, n) / V6;
          const int c0 = 3 * n;
          B(0, c0) = bx;
          B(1, c0 + 1) = by;
          B(2, c0 + 2) = bz;
          B(3, c0) = by;
          B(3, c0 + 1) = bx;
          B(4, c0 + 1) = bz;
          B(4, c0 + 2) = by;
          B(5, c0) = bz;
          B(5, c0 + 2) = bx;
        }
        const double temp =
            average_temperature(elem_data.nodes, nodal_temps_rec, mat_.ref_temp);
        return solid_stress_strain_from_B(D_, B, ue, mat_.A, temp,
                                         mat_.ref_temp);
      };

      auto eval_ctetra10 = [&](const double L1, const double L2,
                               const double L3) {
        // The element's own geometry and shape functions, with omitted
        // midside grids folded into their corners.
        const CTetra10 tet(elem_data.id, elem_data.pid,
                           element_node_array<10>(elem_data.nodes), model);
        const auto nc = tet.node_coords();
        std::array<double, 10> temp{};
        for (int i = 0; i < 10; ++i) {
          const auto it =
              nodal_temps_rec.find(elem_data.nodes[static_cast<size_t>(i)]);
          temp[static_cast<size_t>(i)] =
              (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        const auto sd = tet.folded_shape(L1, L2, L3);
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 10; ++n) {
          J(0, 0) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(0, 1) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(0, 2) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(1, 0) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(1, 1) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(1, 2) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(2, 0) += sd.dNdL3[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(2, 1) += sd.dNdL3[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(2, 2) += sd.dNdL3[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
        }
        const Eigen::Matrix3d Jinv = J.inverse();
        Eigen::MatrixXd B(6, 30);
        B.setZero();
        for (int n = 0; n < 10; ++n) {
          const double dnx =
              Jinv(0, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(0, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(0, 2) * sd.dNdL3[static_cast<size_t>(n)];
          const double dny =
              Jinv(1, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(1, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(1, 2) * sd.dNdL3[static_cast<size_t>(n)];
          const double dnz =
              Jinv(2, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(2, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(2, 2) * sd.dNdL3[static_cast<size_t>(n)];
          const int c0 = 3 * n;
          B(0, c0) = dnx;
          B(1, c0 + 1) = dny;
          B(2, c0 + 2) = dnz;
          B(3, c0) = dny;
          B(3, c0 + 1) = dnx;
          B(4, c0 + 1) = dnz;
          B(4, c0 + 2) = dny;
          B(5, c0) = dnz;
          B(5, c0 + 2) = dnx;
        }
        double point_temp = 0.0;
        for (int n = 0; n < 10; ++n)
          point_temp += sd.N[static_cast<size_t>(n)] * temp[static_cast<size_t>(n)];
        return solid_stress_strain_from_B(D_, B, ue, mat_.A, point_temp,
                                         mat_.ref_temp);
      };

      auto eval_cpenta6 = [&](const double L1, const double L2,
                              const double zeta) {
        auto nc = [&]() -> std::array<Vec3, 6> {
          std::array<Vec3, 6> c;
          for (int i = 0; i < 6; ++i)
            c[i] = model.node(elem_data.nodes[static_cast<size_t>(i)]).position;
          return c;
        }();
        std::array<double, 6> temp{};
        for (int i = 0; i < 6; ++i) {
          const auto it =
              nodal_temps_rec.find(elem_data.nodes[static_cast<size_t>(i)]);
          temp[static_cast<size_t>(i)] =
              (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        const auto sd = CPenta6::shape_functions(L1, L2, zeta);
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 6; ++n) {
          J(0, 0) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(0, 1) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(0, 2) += sd.dNdL1[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(1, 0) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(1, 1) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(1, 2) += sd.dNdL2[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(2, 0) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(2, 1) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(2, 2) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
        }
        const Eigen::Matrix3d Jinv = J.inverse();
        Eigen::MatrixXd B(6, 18);
        B.setZero();
        for (int n = 0; n < 6; ++n) {
          const double dnx =
              Jinv(0, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(0, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(0, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const double dny =
              Jinv(1, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(1, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(1, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const double dnz =
              Jinv(2, 0) * sd.dNdL1[static_cast<size_t>(n)] +
              Jinv(2, 1) * sd.dNdL2[static_cast<size_t>(n)] +
              Jinv(2, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const int c0 = 3 * n;
          B(0, c0) = dnx;
          B(1, c0 + 1) = dny;
          B(2, c0 + 2) = dnz;
          B(3, c0) = dny;
          B(3, c0 + 1) = dnx;
          B(4, c0 + 1) = dnz;
          B(4, c0 + 2) = dny;
          B(5, c0) = dnz;
          B(5, c0 + 2) = dnx;
        }
        double point_temp = 0.0;
        for (int n = 0; n < 6; ++n)
          point_temp += sd.N[static_cast<size_t>(n)] * temp[static_cast<size_t>(n)];
        return solid_stress_strain_from_B(D_, B, ue, mat_.A, point_temp,
                                         mat_.ref_temp);
      };

      auto eval_chexa8 = [&](const double xi, const double eta,
                             const double zeta) {
        auto nc = [&]() -> std::array<Vec3, 8> {
          std::array<Vec3, 8> c;
          for (int i = 0; i < 8; ++i)
            c[i] = model.node(elem_data.nodes[static_cast<size_t>(i)]).position;
          return c;
        }();
        std::array<double, 8> temp{};
        for (int i = 0; i < 8; ++i) {
          const auto it =
              nodal_temps_rec.find(elem_data.nodes[static_cast<size_t>(i)]);
          temp[static_cast<size_t>(i)] =
              (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        const auto sd = CHexa8::shape_functions(xi, eta, zeta);
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
          J(0, 0) += sd.dNdxi[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(0, 1) += sd.dNdxi[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(0, 2) += sd.dNdxi[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(1, 0) += sd.dNdeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(1, 1) += sd.dNdeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(1, 2) += sd.dNdeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
          J(2, 0) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].x;
          J(2, 1) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].y;
          J(2, 2) += sd.dNdzeta[static_cast<size_t>(n)] * nc[static_cast<size_t>(n)].z;
        }
        const Eigen::Matrix3d Jinv = J.inverse();
        Eigen::MatrixXd B(6, 24);
        B.setZero();
        for (int n = 0; n < 8; ++n) {
          const double dnx =
              Jinv(0, 0) * sd.dNdxi[static_cast<size_t>(n)] +
              Jinv(0, 1) * sd.dNdeta[static_cast<size_t>(n)] +
              Jinv(0, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const double dny =
              Jinv(1, 0) * sd.dNdxi[static_cast<size_t>(n)] +
              Jinv(1, 1) * sd.dNdeta[static_cast<size_t>(n)] +
              Jinv(1, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const double dnz =
              Jinv(2, 0) * sd.dNdxi[static_cast<size_t>(n)] +
              Jinv(2, 1) * sd.dNdeta[static_cast<size_t>(n)] +
              Jinv(2, 2) * sd.dNdzeta[static_cast<size_t>(n)];
          const int c0 = 3 * n;
          B(0, c0) = dnx;
          B(1, c0 + 1) = dny;
          B(2, c0 + 2) = dnz;
          B(3, c0) = dny;
          B(3, c0 + 1) = dnx;
          B(4, c0 + 1) = dnz;
          B(4, c0 + 2) = dny;
          B(5, c0) = dnz;
          B(5, c0 + 2) = dnx;
        }
        double point_temp = 0.0;
        for (int n = 0; n < 8; ++n)
          point_temp += sd.N[static_cast<size_t>(n)] * temp[static_cast<size_t>(n)];
        return solid_stress_strain_from_B(D_, B, ue, mat_.A, point_temp,
                                         mat_.ref_temp);
      };

      if (elem_data.type == ElementType::CHEXA20) {
        // The element's assumed-strain operator at the centre and corners.
        std::array<double, 20> temps{};
        const auto nids = hexa20_node_array(elem_data.nodes);
        for (size_t n = 0; n < 20; ++n) {
          const auto it = nodal_temps_rec.find(elem_data.nodes[n]);
          temps[n] = (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        const SolidHexa20 hexa(elem_data.id, elem_data.pid, nids, model);
        const auto points = hexa.recover(
            std::span<const double>(ue.data(), static_cast<size_t>(ue.size())), temps,
            mat_.ref_temp);
        center.eps = points[0].strain;
        center.sigma = points[0].stress;
        for (size_t n = 0; n < 8; ++n) {
          ss.nodal.push_back(make_solid_stress_point(elem_data.nodes[n], points[n + 1].stress));
          if (want_strain)
            strain.nodal.push_back(
                make_solid_strain_point(elem_data.nodes[n], points[n + 1].strain));
        }
      } else if (elem_data.type == ElementType::CTETRA4) {
        center = eval_ctetra4();
        for (NodeId nid : elem_data.nodes) {
          ss.nodal.push_back(make_solid_stress_point(nid, center.sigma));
          if (want_strain)
            strain.nodal.push_back(
                make_solid_strain_point(nid, center.eps));
        }
      } else if (elem_data.type == ElementType::CTETRA10) {
        center = eval_ctetra10(0.25, 0.25, 0.25);
        static constexpr std::array<std::array<double, 3>, 10> kTet10Coords{{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {0.0, 0.0, 0.0},
            {0.5, 0.5, 0.0},
            {0.0, 0.5, 0.5},
            {0.5, 0.0, 0.5},
            {0.5, 0.0, 0.0},
            {0.0, 0.5, 0.0},
            {0.0, 0.0, 0.5},
        }};
        for (int n = 0; n < 10; ++n) {
          if (is_omitted_grid(elem_data.nodes[static_cast<size_t>(n)]))
            continue; // no grid to report at an omitted midside node
          const auto point =
              eval_ctetra10(kTet10Coords[static_cast<size_t>(n)][0],
                            kTet10Coords[static_cast<size_t>(n)][1],
                            kTet10Coords[static_cast<size_t>(n)][2]);
          ss.nodal.push_back(
              make_solid_stress_point(elem_data.nodes[static_cast<size_t>(n)],
                                      point.sigma));
          if (want_strain)
            strain.nodal.push_back(make_solid_strain_point(
                elem_data.nodes[static_cast<size_t>(n)], point.eps));
        }
      } else if (elem_data.type == ElementType::CPENTA6) {
        center = eval_cpenta6(1.0 / 3.0, 1.0 / 3.0, 0.0);
        static constexpr std::array<std::array<double, 3>, 6> kPentaCoords{{
            {1.0, 0.0, -1.0},
            {0.0, 1.0, -1.0},
            {0.0, 0.0, -1.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0},
            {0.0, 0.0, 1.0},
        }};
        for (int n = 0; n < 6; ++n) {
          const auto point =
              eval_cpenta6(kPentaCoords[static_cast<size_t>(n)][0],
                           kPentaCoords[static_cast<size_t>(n)][1],
                           kPentaCoords[static_cast<size_t>(n)][2]);
          ss.nodal.push_back(
              make_solid_stress_point(elem_data.nodes[static_cast<size_t>(n)],
                                      point.sigma));
          if (want_strain)
            strain.nodal.push_back(make_solid_strain_point(
                elem_data.nodes[static_cast<size_t>(n)], point.eps));
        }
      } else if (psol_.isop != SolidFormulation::SRI) {
        // Q1/E21 (or ISOP = FULL) CHEXA8: the element's own condensed
        // strain operator, with extrapolation at degenerate corners.
        std::array<double, 8> temps{};
        for (size_t n = 0; n < 8; ++n) {
          const auto it = nodal_temps_rec.find(elem_data.nodes[n]);
          temps[n] = (it != nodal_temps_rec.end()) ? it->second : mat_.ref_temp;
        }
        const SolidHexa8 hexa(elem_data.id, elem_data.pid,
                              {elem_data.nodes[0], elem_data.nodes[1], elem_data.nodes[2],
                               elem_data.nodes[3], elem_data.nodes[4], elem_data.nodes[5],
                               elem_data.nodes[6], elem_data.nodes[7]},
                              model);
        const auto points = hexa.recover(
            std::span<const double>(ue.data(), static_cast<size_t>(ue.size())), temps,
            mat_.ref_temp);
        center.eps = points[0].strain;
        center.sigma = points[0].stress;
        for (size_t n = 0; n < 8; ++n) {
          ss.nodal.push_back(make_solid_stress_point(elem_data.nodes[n], points[n + 1].stress));
          if (want_strain)
            strain.nodal.push_back(
                make_solid_strain_point(elem_data.nodes[n], points[n + 1].strain));
        }
      } else {
        center = eval_chexa8(0.0, 0.0, 0.0);
        static constexpr std::array<std::array<double, 3>, 8> kHexCoords{{
            {-1.0, -1.0, -1.0},
            {1.0, -1.0, -1.0},
            {1.0, 1.0, -1.0},
            {-1.0, 1.0, -1.0},
            {-1.0, -1.0, 1.0},
            {1.0, -1.0, 1.0},
            {1.0, 1.0, 1.0},
            {-1.0, 1.0, 1.0},
        }};
        for (int n = 0; n < 8; ++n) {
          const auto point =
              eval_chexa8(kHexCoords[static_cast<size_t>(n)][0],
                          kHexCoords[static_cast<size_t>(n)][1],
                          kHexCoords[static_cast<size_t>(n)][2]);
          ss.nodal.push_back(
              make_solid_stress_point(elem_data.nodes[static_cast<size_t>(n)],
                                      point.sigma));
          if (want_strain)
            strain.nodal.push_back(make_solid_strain_point(
                elem_data.nodes[static_cast<size_t>(n)], point.eps));
        }
      }

      ss.sx = center.sigma(0);
      ss.sy = center.sigma(1);
      ss.sz = center.sigma(2);
      ss.sxy = center.sigma(3);
      ss.syz = center.sigma(4);
      ss.szx = center.sigma(5);
      ss.von_mises = solid_von_mises(center.sigma);
      res.solid_stresses.push_back(ss);

      if (want_strain) {
        strain.ex = center.eps(0);
        strain.ey = center.eps(1);
        strain.ez = center.eps(2);
        strain.exy = center.eps(3);
        strain.eyz = center.eps(4);
        strain.ezx = center.eps(5);
        strain.von_mises = solid_von_mises(center.eps);
        res.solid_strains.push_back(strain);
      }
    }
  }

  return res;
}

} // namespace vibestran
