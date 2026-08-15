// src/solver/heat_transfer_steady.cpp

#include "solver/heat_transfer_steady.hpp"

#include "core/dof_map.hpp"
#include "core/sparse_matrix.hpp"
#include "elements/chbdy_element.hpp"
#include "elements/thermal_elements.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <span>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace vibestran {

namespace {

// Build set of nodes that participate in any thermal element (volume/line or
// CHBDY surface).  Used to constrain orphan nodes so K is non-singular.
[[nodiscard]] std::unordered_set<NodeId>
thermal_active_nodes(const Model &model) {
  std::unordered_set<NodeId> active;
  for (const auto &elem : model.elements) {
    if (make_thermal_element(elem, model) == nullptr)
      continue;
    for (NodeId nid : elem.nodes)
      active.insert(nid);
  }
  for (const auto &c : model.chbdy_elements) {
    for (NodeId nid : c.nodes)
      active.insert(nid);
    if (c.ambient_node)
      active.insert(*c.ambient_node);
  }
  return active;
}

// Collect prescribed nodal temperatures: TEMP cards whose SID matches the
// case's TEMP(LOAD) set, plus an optional TEMPD default for those same nodes.
// IMPORTANT: TEMPD alone is *not* a Dirichlet BC — it's a default value,
// matching NASTRAN convention.  Only nodes that appear on a TEMP card are
// constrained.
[[nodiscard]] std::unordered_map<NodeId, double>
collect_prescribed_temperatures(const Model &model, const SubCase &sc) {
  std::unordered_map<NodeId, double> out;
  const int temp_set = (sc.temp_load_set != 0) ? sc.temp_load_set
                                               : sc.load_set.value;
  if (temp_set == 0)
    return out;
  for (const auto &load : model.loads) {
    if (const TempLoad *tl = std::get_if<TempLoad>(&load))
      if (tl->sid == LoadSetId(temp_set))
        out[tl->node] = tl->temperature;
  }
  return out;
}

// Build the thermal DOF map: one scalar T per node.  Constrain (a) nodes that
// no thermal element touches, and (b) nodes with prescribed temperature
// (those values are imposed via Dirichlet partitioning after assembly).
[[nodiscard]] DofMap build_thermal_dof_map(
    const Model &model,
    const std::unordered_set<NodeId> &active,
    const std::unordered_map<NodeId, double> &prescribed) {
  DofMap dmap;
  dmap.build(model.nodes, /*default_dofs_per_node=*/1);
  std::vector<std::pair<NodeId, int>> to_constrain;
  to_constrain.reserve(model.nodes.size());
  for (const auto &[nid, _] : model.nodes) {
    if (!active.count(nid) || prescribed.count(nid))
      to_constrain.emplace_back(nid, 0);
  }
  dmap.constrain_batch(to_constrain);
  return dmap;
}

// Apply Dirichlet contribution: for each prescribed node i with value Ti,
// move K_{*,i}·Ti to the RHS of every *free* equation.  Since the prescribed
// nodes are already constrained out of the dof_map, the corresponding columns
// of K are *not* assembled — we need to recompute their contribution element-
// wise.  This is the standard "static condensation of known values" pattern.
void apply_prescribed_temperatures(
    const Model &model, const DofMap &dof_map,
    const std::unordered_map<NodeId, double> &prescribed,
    std::vector<double> &Q) {
  if (prescribed.empty()) return;

  auto contribute = [&](std::span<const NodeId> elem_nodes,
                        const Eigen::MatrixXd &Ke) {
    const int n = static_cast<int>(elem_nodes.size());
    for (int i = 0; i < n; ++i) {
      const EqIndex row = dof_map.eq_index(elem_nodes[i], 0);
      if (row == CONSTRAINED_DOF) continue;
      double accum = 0.0;
      for (int j = 0; j < n; ++j) {
        auto it = prescribed.find(elem_nodes[j]);
        if (it == prescribed.end()) continue;
        accum += Ke(i, j) * it->second;
      }
      if (accum != 0.0)
        Q[static_cast<size_t>(row)] -= accum;
    }
  };

  for (const auto &elem : model.elements) {
    auto te = make_thermal_element(elem, model);
    if (!te) continue;
    const Eigen::MatrixXd Ke = te->conductance_matrix();
    contribute(te->node_ids(), Ke);
  }
  for (const auto &c : model.chbdy_elements) {
    ChbdyElementImpl cb(c, model);
    const Eigen::MatrixXd Ke = cb.convection_conductance();
    contribute(cb.surface_nodes(), Ke);
  }
}

// Assemble conductance K (free × free) and applied-flux RHS Q (free), ignoring
// the prescribed-Dirichlet partition (which is handled separately).
void assemble_system(const Model &model, const SubCase &sc,
                     const DofMap &dof_map,
                     SparseMatrixBuilder &K_builder,
                     std::vector<double> &Q) {
  auto add_block = [&](std::span<const NodeId> elem_nodes,
                       const Eigen::MatrixXd &Ke) {
    const int n = static_cast<int>(elem_nodes.size());
    for (int i = 0; i < n; ++i) {
      const EqIndex ri = dof_map.eq_index(elem_nodes[i], 0);
      if (ri == CONSTRAINED_DOF) continue;
      for (int j = 0; j < n; ++j) {
        const EqIndex rj = dof_map.eq_index(elem_nodes[j], 0);
        if (rj == CONSTRAINED_DOF) continue;
        K_builder.add(ri, rj, Ke(i, j));
      }
    }
  };

  auto add_rhs = [&](std::span<const NodeId> elem_nodes,
                     const Eigen::VectorXd &Pe) {
    for (int i = 0; i < static_cast<int>(elem_nodes.size()); ++i) {
      const EqIndex r = dof_map.eq_index(elem_nodes[i], 0);
      if (r == CONSTRAINED_DOF) continue;
      Q[static_cast<size_t>(r)] += Pe(i);
    }
  };

  // 1) Volume / line conductance
  for (const auto &elem : model.elements) {
    auto te = make_thermal_element(elem, model);
    if (!te) continue;
    add_block(te->node_ids(), te->conductance_matrix());
  }

  // 2) CHBDY convection (couples surface nodes; also surface↔ambient_node
  //    when present)
  for (const auto &c : model.chbdy_elements) {
    ChbdyElementImpl cb(c, model);
    const Eigen::MatrixXd Ks = cb.convection_conductance();
    add_block(cb.surface_nodes(), Ks);
    if (c.ambient_node) {
      // Augmented block: K_sf = -row_sum_per_row distributed to ambient node.
      // Simpler form: convection couples surface node i and ambient node f as
      //   q_i = sum_j K_ij (T_i - T_f)  (consistent matrix)
      // → add K_ii contribution to (i,f) as -K_ij rowsum.
      const int n = static_cast<int>(cb.surface_nodes().size());
      const NodeId f = *c.ambient_node;
      const EqIndex rf = dof_map.eq_index(f, 0);
      for (int i = 0; i < n; ++i) {
        const EqIndex ri = dof_map.eq_index(cb.surface_nodes()[i], 0);
        if (ri == CONSTRAINED_DOF) continue;
        double rowsum = 0.0;
        for (int j = 0; j < n; ++j) rowsum += Ks(i, j);
        // -K_ij rowsum on (ri, rf) and (rf, ri); +rowsum on (rf, rf).
        if (rf != CONSTRAINED_DOF) {
          K_builder.add(ri, rf, -rowsum);
          K_builder.add(rf, ri, -rowsum);
          K_builder.add(rf, rf, rowsum);
        }
      }
    } else if (cb.t_amb() != 0.0) {
      // Fixed ambient T from PHBDY: contributes to RHS only.
      add_rhs(cb.surface_nodes(), cb.ambient_rhs());
    }
  }

  // 3) Apply load-set Q*: QVOL, QHBDY, QBDY1, QBDY2
  for (const auto &[lp, scale] : model.loads_for_set(sc.load_set)) {
    std::visit(
        [&](const auto &load) {
          using T = std::decay_t<decltype(load)>;
          if constexpr (std::is_same_v<T, QvolLoad>) {
            for (ElementId eid : load.elements) {
              auto it = std::find_if(model.elements.begin(),
                                     model.elements.end(),
                                     [eid](const ElementData &e) {
                                       return e.id == eid;
                                     });
              if (it == model.elements.end()) continue;
              auto te = make_thermal_element(*it, model);
              if (!te) continue;
              add_rhs(te->node_ids(),
                      te->volumetric_heat_load(scale * load.q_vol));
            }
          } else if constexpr (std::is_same_v<T, Qbdy1Load>) {
            for (ElementId eid : load.elements) {
              auto it = std::find_if(model.chbdy_elements.begin(),
                                     model.chbdy_elements.end(),
                                     [eid](const ChbdyElement &e) {
                                       return e.eid == eid;
                                     });
              if (it == model.chbdy_elements.end()) continue;
              ChbdyElementImpl cb(*it, model);
              add_rhs(cb.surface_nodes(),
                      cb.applied_flux_load(scale * load.q0));
            }
          } else if constexpr (std::is_same_v<T, Qbdy2Load>) {
            auto it = std::find_if(model.chbdy_elements.begin(),
                                   model.chbdy_elements.end(),
                                   [&load](const ChbdyElement &e) {
                                     return e.eid == load.element;
                                   });
            if (it == model.chbdy_elements.end()) return;
            ChbdyElementImpl cb(*it, model);
            std::array<double, 4> q{};
            for (int i = 0; i < 4; ++i) q[i] = scale * load.q[i];
            add_rhs(cb.surface_nodes(),
                    cb.applied_flux_load_per_node(
                        std::span<const double>(q.data(),
                                                cb.surface_nodes().size())));
          } else if constexpr (std::is_same_v<T, QvectLoad>) {
            for (ElementId eid : load.elements) {
              auto it = std::find_if(model.chbdy_elements.begin(),
                                     model.chbdy_elements.end(),
                                     [eid](const ChbdyElement &e) {
                                       return e.eid == eid;
                                     });
              if (it == model.chbdy_elements.end()) continue;
              ChbdyElementImpl cb(*it, model);
              add_rhs(cb.surface_nodes(),
                      cb.directional_flux_load(scale * load.q0,
                                               load.direction));
            }
          } else if constexpr (std::is_same_v<T, QhbdyLoad>) {
            // QHBDY defines its own geometry on the fly; build a temporary
            // CHBDY surface element to reuse the area integration.
            ChbdyElement tmp;
            tmp.eid = ElementId(-1);
            tmp.pid = PropertyId(0);
            tmp.geom = load.geom;
            tmp.mid = MaterialId(0);
            tmp.nodes = load.nodes;
            ChbdyElementImpl cb(tmp, model);
            add_rhs(cb.surface_nodes(),
                    cb.applied_flux_load(scale * load.q0 * load.af));
          }
        },
        *lp);
  }
}

} // namespace

HeatTransferSteadySolver::HeatTransferSteadySolver(
    std::unique_ptr<SolverBackend> backend)
    : backend_(std::move(backend)) {}

SolverResults HeatTransferSteadySolver::solve(const Model &model) {
  model.validate();
  SolverResults results;
  for (const auto &sc : model.analysis.subcases)
    results.subcases.push_back(solve_subcase(model, sc));
  return results;
}

SubCaseResults HeatTransferSteadySolver::solve_subcase(const Model &model,
                                                       const SubCase &sc) {
  using Clock = std::chrono::steady_clock;
  using Ms = std::chrono::duration<double, std::milli>;
  const auto t0 = Clock::now();

  const auto active = thermal_active_nodes(model);
  const auto prescribed = collect_prescribed_temperatures(model, sc);
  DofMap dof_map = build_thermal_dof_map(model, active, prescribed);
  const int n_free = dof_map.num_free_dofs();
  spdlog::debug("[heat subcase {}] active nodes: {}  prescribed: {}  free DOFs: {}",
                sc.id, active.size(), prescribed.size(), n_free);

  std::vector<double> T_free;
  if (n_free > 0) {
    SparseMatrixBuilder K_builder(n_free);
    std::vector<double> Q(static_cast<size_t>(n_free), 0.0);

    assemble_system(model, sc, dof_map, K_builder, Q);
    apply_prescribed_temperatures(model, dof_map, prescribed, Q);

    auto csr = K_builder.build_csr();
    const auto *solve_csr = &csr;
    SparseMatrixBuilder::CsrData expanded;
    if (backend_->requires_full_symmetric_csr()) {
      expanded = csr.expanded_symmetric();
      solve_csr = &expanded;
    }
    T_free = backend_->solve(*solve_csr, Q, nullptr);
    const auto t1 = Clock::now();
    spdlog::debug("[heat subcase {}] solve: {:.3f} ms", sc.id, Ms(t1 - t0).count());
  }

  // ── Recover full nodal temperature vector ────────────────────────────────
  std::unordered_map<NodeId, double> all_temps = prescribed;
  for (const auto &[nid, _] : model.nodes) {
    if (all_temps.count(nid)) continue;
    const EqIndex eq = dof_map.eq_index(nid, 0);
    all_temps[nid] =
        (eq != CONSTRAINED_DOF && eq < n_free) ? T_free[static_cast<size_t>(eq)]
                                               : 0.0;
  }

  SubCaseResults res;
  res.id = sc.id;
  res.label = sc.label;

  // Sorted temperature output
  std::vector<NodeId> sorted;
  sorted.reserve(model.nodes.size());
  for (const auto &[nid, _] : model.nodes) sorted.push_back(nid);
  std::sort(sorted.begin(), sorted.end());
  for (NodeId nid : sorted)
    res.temperatures.push_back({nid, all_temps.at(nid)});

  // Element heat-flux recovery (centroidal)
  for (const auto &elem : model.elements) {
    auto te = make_thermal_element(elem, model);
    if (!te) continue;
    std::vector<double> tn;
    tn.reserve(elem.nodes.size());
    std::transform(elem.nodes.begin(), elem.nodes.end(),
                   std::back_inserter(tn),
                   [&](NodeId nid) { return all_temps[nid]; });
    Eigen::VectorXd q = te->heat_flux(tn);
    ElementHeatFlux ef;
    ef.eid = elem.id;
    ef.etype = elem.type;
    ef.dim = static_cast<int>(q.size());
    for (int i = 0; i < ef.dim && i < 3; ++i) ef.q[i] = q(i);
    double m2 = 0.0;
    for (int i = 0; i < ef.dim; ++i) m2 += q(i) * q(i);
    ef.magnitude = std::sqrt(m2);
    res.heat_fluxes.push_back(ef);
  }
  return res;
}

} // namespace vibestran
