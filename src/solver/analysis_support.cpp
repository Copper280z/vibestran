// src/solver/analysis_support.cpp

#include "solver/analysis_support.hpp"

#include "core/coord_sys.hpp"
#include "elements/element_factory.hpp"
#include "elements/rbe_constraints.hpp"
#include "solver/linear_static.hpp"
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <iterator>
#include <map>
#include <unordered_map>

namespace vibestran {

namespace {

using ActiveMask = std::array<bool, 6>;

[[nodiscard]] ActiveMask empty_mask() {
  ActiveMask mask{};
  mask.fill(false);
  return mask;
}

void mark_component(ActiveMask &mask, const int component) {
  if (component >= 1 && component <= 6)
    mask[static_cast<size_t>(component - 1)] = true;
}

void mark_all(ActiveMask &mask) { mask.fill(true); }

/// Marks a component that a card names on a grid. With CD != 0 it lies along a
/// displacement axis, generally oblique to basic, so its whole block is active.
void mark_grid_component(const Model &model, const NodeId nid, const int component,
                         std::unordered_map<NodeId, ActiveMask> &node_masks) {
  const auto grid = model.nodes.find(nid);
  ActiveMask &mask = node_masks[nid];
  if (grid == model.nodes.end() || grid->second.cd == CoordId{0} ||
      component < 1 || component > 6) {
    mark_component(mask, component);
    return;
  }
  const int first = component <= 3 ? 1 : 4;
  for (int basic = first; basic < first + 3; ++basic)
    mark_component(mask, basic);
}

void mark_basic_translations(ActiveMask &mask) {
  mask[0] = true;
  mask[1] = true;
  mask[2] = true;
}

[[nodiscard]] bool is_scalar_element(const ElementData &elem) {
  return elem.type == ElementType::CELAS1 || elem.type == ElementType::CELAS2 ||
         elem.type == ElementType::CMASS1 || elem.type == ElementType::CMASS2;
}

void mark_element_dofs(const ElementData &elem, const Model &model,
                       std::unordered_map<NodeId, ActiveMask> &node_masks) {
  switch (elem.type) {
  case ElementType::CQUAD4:
  case ElementType::CTRIA3:
  case ElementType::CBAR:
  case ElementType::CBEAM:
  case ElementType::CROD:
  case ElementType::CBUSH:
    for (NodeId nid : elem.nodes)
      mark_all(node_masks[nid]);
    break;
  case ElementType::CHEXA8:
  case ElementType::CHEXA20:
  case ElementType::CTETRA4:
  case ElementType::CTETRA10:
  case ElementType::CPENTA6:
    for (NodeId nid : elem.nodes)
      if (nid.value != 0) // NodeId{0}: an omitted CTETRA10/CHEXA20 midside grid
        mark_basic_translations(node_masks[nid]);
    break;
  case ElementType::CELAS1:
  case ElementType::CELAS2:
  case ElementType::CMASS1:
  case ElementType::CMASS2:
    for (size_t end = 0; end < elem.nodes.size(); ++end)
      mark_grid_component(model, elem.nodes[end], elem.components[end], node_masks);
    break;
  case ElementType::CONM2: {
    // Translations always carry the mass.  Rotations carry inertia only with
    // an inertia tensor or a CG offset; activating them otherwise would
    // create stiffness-less DOFs at grids that only solids connect.
    if (elem.nodes.empty())
      break;
    ActiveMask &mask = node_masks[elem.nodes[0]];
    mark_basic_translations(mask);
    if (elem.conm2.has_value()) {
      const auto &c = *elem.conm2;
      const bool has_inertia = std::any_of(c.inertia.begin(), c.inertia.end(),
                                           [](double v) { return v != 0.0; });
      const bool has_offset =
          c.cid == -1 || c.offset.x != 0.0 || c.offset.y != 0.0 || c.offset.z != 0.0;
      if (has_inertia || has_offset)
        mark_all(mask);
    }
    break;
  }
  case ElementType::CHBDY:
    break;  // CHBDY elements are stored in model.chbdy_elements, never here
  }
}

void mark_rigid_element_dofs(const Model &model,
                             std::unordered_map<NodeId, ActiveMask> &node_masks) {
  for (const auto &rbe2 : model.rbe2s) {
    for (int dof = 1; dof <= 6; ++dof) {
      if (!rbe2.cm.has(dof))
        continue;
      mark_component(node_masks[rbe2.gn], dof);
      for (NodeId nid : rbe2.gm)
        mark_component(node_masks[nid], dof);
    }
  }

  for (const auto &rbe3 : model.rbe3s) {
    for (int dof = 1; dof <= 6; ++dof)
      if (rbe3.refc.has(dof))
        mark_grid_component(model, rbe3.ref_node, dof, node_masks);
    for (const auto &group : rbe3.weight_groups) {
      for (int dof = 1; dof <= 6; ++dof) {
        if (!group.component.has(dof))
          continue;
        for (NodeId nid : group.nodes)
          mark_grid_component(model, nid, dof, node_masks);
      }
    }
  }
}

void mark_explicit_mpc_dofs(const Model &model, const SubCase &sc,
                            std::unordered_map<NodeId, ActiveMask> &node_masks) {
  if (sc.mpc_set.value == 0)
    return;
  for (const Mpc *mpc : model.mpcs_for_set(sc.mpc_set)) {
    for (const auto &term : mpc->terms)
      mark_grid_component(model, term.node, term.dof, node_masks);
  }
}

/// An explicit MPC on basic components, the only ones the DOF map has. A term on
/// a grid with CD != 0 names a component of that grid's displacement frame, and
/// u_cd[i] = sum_j axes(j, i) u_basic[j]. The terms on one block of one grid are
/// summed first, so components that cancel in basic leave no trace, then written
/// where the first of them stood with the largest first: MpcHandler prefers the
/// earliest term as dependent, and this keeps it the strongest basic component.
[[nodiscard]] Mpc mpc_on_basic_components(const Mpc &mpc, const Model &model) {
  const auto in_grid_frame = [&](const MpcTerm &term) {
    const auto grid = model.nodes.find(term.node);
    return grid != model.nodes.end() && grid->second.cd != CoordId{0} &&
           term.dof >= 1 && term.dof <= 6;
  };
  const auto block_of = [](const MpcTerm &term) {
    return std::pair{term.node.value, term.dof <= 3 ? 0 : 3};
  };

  std::map<std::pair<int, int>, Eigen::Vector3d> blocks;
  double largest = 0.0;
  for (const MpcTerm &term : mpc.terms) {
    largest = std::max(largest, std::abs(term.coeff));
    if (!in_grid_frame(term))
      continue;
    const auto key = block_of(term);
    const Mat3 axes = model.displacement_axes(term.node);
    const int axis = term.dof - 1 - key.second;
    const auto [block, unused] = blocks.try_emplace(key, Eigen::Vector3d::Zero());
    (void)unused;
    for (int basic = 0; basic < 3; ++basic)
      block->second(basic) += term.coeff * axes(basic, axis);
  }
  if (blocks.empty())
    return mpc;

  Mpc result = mpc;
  result.terms.clear();
  for (const MpcTerm &term : mpc.terms) {
    if (!in_grid_frame(term)) {
      result.terms.push_back(term);
      continue;
    }
    const auto block = blocks.find(block_of(term));
    if (block == blocks.end())
      continue; // written with the block's first term
    std::array<int, 3> order{0, 1, 2};
    std::ranges::stable_sort(order, std::greater{}, [&](const int basic) {
      return std::abs(block->second(basic));
    });
    for (const int basic : order)
      if (std::abs(block->second(basic)) > 1e-14 * largest)
        result.terms.push_back(
            {term.node, block->first.second + basic + 1, block->second(basic)});
    blocks.erase(block);
  }
  return result;
}

void reduce_3x3_constraints(NodeId nid, const Mat3 &T3,
                            const std::array<int, 3> &cd_dofs,
                            const std::array<double, 3> &cd_values,
                            const int count, const int base_offset,
                            std::vector<Mpc> &all_mpcs,
                            std::vector<std::pair<NodeId, int>> &direct_constraints) {
  if (count == 0)
    return;
  if (count == 3) {
    // A complete CD block specifies a unique basic-frame vector. Preserve
    // its nonzero components as affine equations so attached RBEs/MPCs and
    // the reduced load vector receive the prescribed motion.
    for (int j = 0; j < 3; ++j) {
      double value = 0.0;
      for (int i = 0; i < count; ++i)
        value += T3(j, cd_dofs[i]) * cd_values[i];
      if (value == 0.0) {
        direct_constraints.emplace_back(nid, base_offset + j);
      } else {
        Mpc prescribed;
        prescribed.is_spc = true;
        prescribed.terms.push_back({nid, base_offset + j + 1, 1.0});
        prescribed.rhs = value;
        all_mpcs.push_back(std::move(prescribed));
      }
    }
    return;
  }

  double A[3][3] = {};
  double rhs[3] = {};
  int col_perm[3] = {0, 1, 2};
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < 3; ++j)
      A[i][j] = T3(j, cd_dofs[i]);
  for (int i = 0; i < count; ++i)
    rhs[i] = cd_values[i];

  for (int row = 0; row < count; ++row) {
    int best_col = row;
    double best_val = std::abs(A[row][row]);
    for (int col = row + 1; col < 3; ++col) {
      if (std::abs(A[row][col]) > best_val) {
        best_val = std::abs(A[row][col]);
        best_col = col;
      }
    }
    if (best_col != row) {
      for (int i = 0; i < count; ++i)
        std::swap(A[i][row], A[i][best_col]);
      std::swap(col_perm[row], col_perm[best_col]);
    }
    if (std::abs(A[row][row]) < 1e-14)
      continue;
    for (int i = row + 1; i < count; ++i) {
      const double factor = A[i][row] / A[row][row];
      for (int j = row; j < 3; ++j)
        A[i][j] -= factor * A[row][j];
      rhs[i] -= factor * rhs[row];
    }
  }

  for (int row = 0; row < count; ++row) {
    if (std::abs(A[row][row]) < 1e-14)
      continue;

    int nnz = 0;
    for (int col = row; col < 3; ++col)
      if (std::abs(A[row][col]) > 1e-14)
        ++nnz;

    if (nnz == 1 && std::abs(rhs[row]) < 1e-14) {
      direct_constraints.emplace_back(nid, base_offset + col_perm[row]);
      continue;
    }

    Mpc mpc;
    mpc.sid = MpcSetId{0};
    mpc.is_spc = true;
    mpc.rhs = rhs[row];
    for (int col = row; col < 3; ++col) {
      if (std::abs(A[row][col]) > 1e-14)
        mpc.terms.push_back({nid, base_offset + col_perm[col] + 1, A[row][col]});
    }
    if (!mpc.terms.empty())
      all_mpcs.push_back(std::move(mpc));
  }
}

/// Removes from `free` (positions in a block) the components nearest each
/// principal direction of `block` whose stiffness is below tolerance, and
/// returns them.
std::vector<int> singular_components(const Eigen::Matrix3d &block,
                                     std::vector<int> free,
                                     const double tolerance) {
  std::vector<int> singular;
  while (!free.empty()) {
    const auto n = static_cast<Eigen::Index>(free.size());
    Eigen::MatrixXd sub(n, n);
    for (Eigen::Index i = 0; i < n; ++i)
      for (Eigen::Index j = 0; j < n; ++j)
        sub(i, j) = block(free[static_cast<size_t>(i)], free[static_cast<size_t>(j)]);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(sub);
    if (eigen.eigenvalues()(0) > tolerance)
      break;
    // The largest entry of the weakest direction; the first of those that tie,
    // up to roundoff, so a rod at 45 degrees picks the same component each run.
    Eigen::Index nearest = 0;
    for (Eigen::Index i = 1; i < n; ++i)
      if (std::abs(eigen.eigenvectors()(i, 0)) >
          std::abs(eigen.eigenvectors()(nearest, 0)) + 1e-9)
        nearest = i;
    singular.push_back(free[static_cast<size_t>(nearest)]);
    free.erase(free.begin() + nearest);
  }
  return singular;
}

} // namespace

std::vector<Spc> analysis_spcs(const Model &model, const SubCase &sc) {
  std::vector<Spc> result = model.structural_spcs(sc.spc_set);
  const auto autospc = model.params.find("AUTOSPC");
  if (autospc != model.params.end() && !autospc->second.empty() &&
      std::toupper(static_cast<unsigned char>(autospc->second[0])) == 'N')
    return result;

  // The translation and rotation blocks to analyse at each grid.
  std::unordered_map<NodeId, std::array<bool, 2>> analysed;
  for (const auto &elem : model.elements)
    if (elem.type == ElementType::CROD)
      for (const NodeId nid : elem.nodes)
        analysed[nid] = {true, true};

  // Components named on a CD grid by a scalar element or MPC, which activate
  // their whole block. Nothing else may stiffen the rest of it.
  std::vector<std::pair<NodeId, int>> named_on_cd;
  const auto note_named = [&](const NodeId nid, const int component) {
    const auto grid = model.nodes.find(nid);
    if (grid != model.nodes.end() && grid->second.cd != CoordId{0} &&
        component >= 1 && component <= 6)
      named_on_cd.emplace_back(nid, component);
  };
  for (const auto &elem : model.elements)
    if (is_scalar_element(elem))
      for (size_t end = 0; end < elem.nodes.size(); ++end)
        note_named(elem.nodes[end], elem.components[end]);
  std::unordered_map<NodeId, ActiveMask> by_explicit;
  if (sc.mpc_set.value != 0)
    for (const Mpc *mpc : model.mpcs_for_set(sc.mpc_set))
      for (const auto &term : mpc->terms) {
        mark_component(by_explicit[term.node], term.dof);
        note_named(term.node, term.dof);
      }
  std::unordered_map<NodeId, ActiveMask> by_rigid;
  mark_rigid_element_dofs(model, by_rigid);
  if (!named_on_cd.empty()) {
    std::unordered_map<NodeId, ActiveMask> declared = by_rigid;
    for (const auto &elem : model.elements)
      if (!is_scalar_element(elem) && elem.type != ElementType::CONM2)
        mark_element_dofs(elem, model, declared);
    for (const auto &[nid, component] : named_on_cd) {
      const size_t block = component <= 3 ? 0 : 1;
      const auto mask = declared.find(nid);
      bool whole = mask != declared.end();
      for (size_t c = 0; whole && c < 3; ++c)
        whole = mask->second[3 * block + c];
      if (!whole)
        analysed[nid][block] = true;
    }
  }
  if (analysed.empty())
    return result;

  std::unordered_map<NodeId, Eigen::Matrix<double, 6, 6>> blocks;
  for (const auto &[nid, _] : analysed)
    blocks.emplace(nid, Eigen::Matrix<double, 6, 6>::Zero());

  // Each grid's own 6x6 stiffness from every element on it, in basic axes.
  for (const auto &elem : model.elements) {
    if (std::none_of(elem.nodes.begin(), elem.nodes.end(),
                     [&](NodeId nid) { return blocks.contains(nid); }))
      continue;
    const LocalKe ke = make_element(elem, model)->stiffness_matrix();
    const auto identity = local_dof_identity(elem, model);
    for (Eigen::Index i = 0; i < ke.rows(); ++i) {
      const auto [node, row] = identity[static_cast<size_t>(i)];
      const auto block = blocks.find(node);
      if (block == blocks.end())
        continue;
      for (Eigen::Index j = 0; j < ke.cols(); ++j) {
        const auto [other, col] = identity[static_cast<size_t>(j)];
        if (other == node)
          block->second(row, col) += ke(i, j);
      }
    }
  }

  // Components already controlled at these grids.
  std::unordered_map<NodeId, ActiveMask> by_spc;
  for (const Spc &spc : result)
    for (int d = 1; d <= 6; ++d)
      if (spc.dofs.has(d))
        mark_component(by_spc[spc.node], d);

  const double epzero = model.param_real("EPZERO").value_or(1e-8);
  std::vector<NodeId> grids;
  grids.reserve(analysed.size());
  for (const auto &[nid, _] : analysed)
    grids.push_back(nid);
  std::sort(grids.begin(), grids.end());

  for (const NodeId nid : grids) {
    const Eigen::Matrix<double, 6, 6> &k = blocks.at(nid);
    const std::array<bool, 2> &analyse = analysed.at(nid);
    const Mat3 T3 = model.displacement_axes(nid);
    Eigen::Matrix3d T;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        T(i, j) = T3(i, j);
    const auto spc_mask = by_spc.find(nid);
    const auto rigid_mask = by_rigid.find(nid);
    const auto explicit_mask = by_explicit.find(nid);
    const auto marked = [](const auto &masks, const auto &it, const int d) {
      return it != masks.end() && it->second[static_cast<size_t>(d)];
    };

    Spc added;
    added.sid = sc.spc_set;
    added.node = nid;
    for (const int base : {0, 3}) {
      if (!analyse[base == 0 ? 0 : 1])
        continue;
      // RBE components are basic, so one controls each axis it has a part of.
      const auto rigid_on_axis = [&](const int c) {
        for (int j = 0; j < 3; ++j)
          if (marked(by_rigid, rigid_mask, base + j) && std::abs(T(j, c)) > 1e-9)
            return true;
        return false;
      };
      std::vector<int> free;
      for (int c = 0; c < 3; ++c)
        if (!marked(by_spc, spc_mask, base + c) &&
            !marked(by_explicit, explicit_mask, base + c) && !rigid_on_axis(c))
          free.push_back(c);
      const Eigen::Matrix3d block = T.transpose() * k.block<3, 3>(base, base) * T;
      const double largest =
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(block, Eigen::EigenvaluesOnly)
              .eigenvalues()(2);
      for (const int c : singular_components(block, std::move(free), epzero * largest))
        added.dofs.mask |= static_cast<uint8_t>(1U << (base + c));
    }
    if (added.dofs.mask != 0)
      result.push_back(added);
  }
  return result;
}

DofMap build_analysis_dof_map(const Model &model, const SubCase &sc) {
  DofMap dmap;
  dmap.build(model.nodes, 6);

  std::unordered_map<NodeId, ActiveMask> node_masks;
  for (const auto &[nid, _] : model.nodes)
    node_masks.emplace(nid, empty_mask());

  for (const auto &elem : model.elements)
    mark_element_dofs(elem, model, node_masks);
  mark_rigid_element_dofs(model, node_masks);
  mark_explicit_mpc_dofs(model, sc, node_masks);

  std::vector<std::pair<NodeId, int>> inactive_constraints;
  for (const auto &[nid, mask] : node_masks) {
    for (int d = 0; d < 6; ++d) {
      if (!mask[static_cast<size_t>(d)])
        inactive_constraints.emplace_back(nid, d);
    }
  }
  dmap.constrain_batch(inactive_constraints);

  std::vector<std::pair<NodeId, int>> spc_constraints;
  for (const Spc &spc : analysis_spcs(model, sc)) {
    auto node_it = model.nodes.find(spc.node);
    const bool has_cd =
        (node_it != model.nodes.end() && node_it->second.cd.value != 0);
    for (int d = 0; d < 6; ++d) {
      if (!spc.dofs.has(d + 1))
        continue;
      if (has_cd || spc.value != 0.0)
        continue;
      spc_constraints.emplace_back(spc.node, d);
    }
  }
  dmap.constrain_batch(spc_constraints);

  return dmap;
}

// cppcheck-suppress unusedFunction -- referenced from linear_static.cpp and modal.cpp
void build_analysis_mpc_system(const Model &model, const SubCase &sc,
                               DofMap &dof_map, MpcHandler &mpc_handler,
                               const bool use_spc_values) {
  std::vector<Mpc> all_mpcs;

  struct SpcValues {
    DofSet dofs;
    std::array<double, 6> values{};
  };
  std::unordered_map<NodeId, SpcValues> node_spcs;
  for (const Spc &spc : analysis_spcs(model, sc)) {
    SpcValues& values = node_spcs[spc.node];
    values.dofs.mask |= spc.dofs.mask;
    for (int d = 0; d < 6; ++d) {
      if (spc.dofs.has(d + 1))
        values.values[static_cast<size_t>(d)] =
            use_spc_values ? spc.value : 0.0;
    }
  }

  {
    std::vector<std::pair<NodeId, int>> direct_constraints;
    for (const auto &[nid, gp] : model.nodes) {
      if (gp.cd == CoordId{0})
        continue;
      const auto spc_it = node_spcs.find(nid);
      if (spc_it == node_spcs.end())
        continue;
      const auto cs_it = model.coord_systems.find(gp.cd);
      if (cs_it == model.coord_systems.end())
        continue;

      const Mat3 T3 = rotation_matrix(cs_it->second, gp.position);
      const SpcValues& spc_values = spc_it->second;

      std::array<int, 3> trans{};
      std::array<double, 3> trans_values{};
      int n_trans = 0;
      for (int d = 0; d < 3; ++d) {
        if (spc_values.dofs.has(d + 1) && dof_map.is_free(nid, d)) {
          trans[n_trans] = d;
          trans_values[n_trans++] = spc_values.values[static_cast<size_t>(d)];
        }
      }
      reduce_3x3_constraints(nid, T3, trans, trans_values, n_trans, 0, all_mpcs,
                             direct_constraints);

      std::array<int, 3> rot{};
      std::array<double, 3> rot_values{};
      int n_rot = 0;
      for (int d = 0; d < 3; ++d) {
        if (spc_values.dofs.has(d + 4) && dof_map.is_free(nid, d + 3)) {
          rot[n_rot] = d;
          rot_values[n_rot++] =
              spc_values.values[static_cast<size_t>(d + 3)];
        }
      }
      reduce_3x3_constraints(nid, T3, rot, rot_values, n_rot, 3, all_mpcs,
                             direct_constraints);
    }

    if (!direct_constraints.empty())
      dof_map.constrain_batch(direct_constraints);
  }

  // Basic-frame nonzero SPCs remain in the pre-MPC map and are represented as
  // one-term affine equations. Zero SPCs were already removed by the DOF map.
  for (const auto& [nid, spc_values] : node_spcs) {
    const auto node_it = model.nodes.find(nid);
    if (node_it == model.nodes.end() || node_it->second.cd.value != 0)
      continue;
    for (int d = 0; d < 6; ++d) {
      const double value = spc_values.values[static_cast<size_t>(d)];
      if (!spc_values.dofs.has(d + 1) || !dof_map.is_free(nid, d))
        continue;
      Mpc prescribed;
      prescribed.is_spc = true;
      prescribed.terms.push_back({nid, d + 1, 1.0});
      prescribed.rhs = value;
      all_mpcs.push_back(std::move(prescribed));
    }
  }

  for (const auto &rbe2 : model.rbe2s)
    expand_rbe2(rbe2, model, all_mpcs);
  for (const auto &rbe3 : model.rbe3s)
    expand_rbe3(rbe3, model, all_mpcs);
  if (sc.mpc_set.value != 0) {
    for (const Mpc *mpc : model.mpcs_for_set(sc.mpc_set))
      all_mpcs.push_back(mpc_on_basic_components(*mpc, model));
  }

  std::vector<const Mpc *> mpc_ptrs;
  mpc_ptrs.reserve(all_mpcs.size());
  std::transform(all_mpcs.begin(), all_mpcs.end(), std::back_inserter(mpc_ptrs),
                 [](const Mpc &mpc) { return &mpc; });
  mpc_handler.build(mpc_ptrs, dof_map);
}

} // namespace vibestran
