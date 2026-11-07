// src/solver/analysis_support.cpp

#include "solver/analysis_support.hpp"

#include "assembly_parallel.hpp"
#include "core/coord_sys.hpp"
#include "core/factor_ratio_check.hpp"
#include "core/notices.hpp"
#include "elements/element_factory.hpp"
#include "elements/rbe_constraints.hpp"
#include "elements/shell_section.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iterator>
#include <map>
#include <type_traits>
#include <unordered_map>

namespace vibestran {

void report_pivots(const PivotReport &pivots, const FactorRatioCheckPolicy &policy,
                   const std::function<std::string(int)> &equation_name, Notices &notices) {
  const auto report = [&](std::string text, bool fatal) {
    if (fatal)
      notices.add_fatal(std::move(text));
    else
      notices.add(std::move(text));
  };
  if (!pivots.held_rows.empty()) {
    constexpr std::size_t kListed = 8;
    std::string names;
    for (std::size_t i = 0; i < std::min(pivots.held_rows.size(), kListed); ++i)
      names += (i == 0 ? "" : ", ") + equation_name(pivots.held_rows[i]);
    if (pivots.held_rows.size() > kListed)
      names += std::format(" and {} more", pivots.held_rows.size() - kListed);
    report(std::format("the {} is singular: {} pivot(s) came out exactly zero, and these "
                       "unknowns were held at zero: {}. The model has an unconstrained "
                       "mechanism there",
                       policy.matrix_name, pivots.held_rows.size(), names),
           policy.fatal);
  }
  if (pivots.max_ratio > policy.maxratio && pivots.max_ratio_row >= 0) {
    const bool past_fatal = pivots.max_ratio > policy.fatal_ratio;
    report(std::format("the {} has K_ii/D_i = {:.3e} at {} (K_ii {:.3e}, D_i {:.3e}), above "
                       "{} = {:.1e}{}",
                       policy.matrix_name, pivots.max_ratio, equation_name(pivots.max_ratio_row),
                       pivots.max_ratio_diagonal, pivots.max_ratio_pivot,
                       past_fatal ? "the limit for a singular matrix" : "MAXRATIO",
                       past_fatal ? policy.fatal_ratio : policy.maxratio,
                       past_fatal ? ": it is numerically singular, so the model likely has a "
                                    "mechanism or a stiffness too small to resolve there"
                                  : ""),
           past_fatal && policy.fatal);
  }
}

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
  case ElementType::CONROD:
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

/// The blocks at its grids that an element's own stiffness keeps nonsingular, so
/// that AUTOSPC need not compute that stiffness.
struct SureBlocks {
  bool translation{false};
  bool rotation{false};
};

[[nodiscard]] SureBlocks
sure_blocks(const ElementData &elem, const Model &model,
            std::unordered_map<PropertyId, bool> &complete_sections) {
  switch (elem.type) {
  case ElementType::CHEXA8:
  case ElementType::CHEXA20:
  case ElementType::CTETRA4:
  case ElementType::CTETRA10:
  case ElementType::CPENTA6:
    return {.translation = true, .rotation = false};
  case ElementType::CQUAD4:
  case ElementType::CTRIA3: {
    const auto known = complete_sections.find(elem.pid);
    if (known != complete_sections.end())
      return {.translation = known->second, .rotation = known->second};
    const ShellSection section = build_shell_section(
        model, elem.pid, elem.id, elem.type == ElementType::CQUAD4 ? "CQUAD4" : "CTRIA3");
    const auto positive_definite = [](const auto &m) {
      return Eigen::LLT<std::decay_t<decltype(m)>>(m).info() == Eigen::Success;
    };
    const bool complete = positive_definite(section.A) && positive_definite(section.D) &&
                          positive_definite(section.T);
    complete_sections.emplace(elem.pid, complete);
    return {.translation = complete, .rotation = complete};
  }
  default:
    return {};
  }
}

} // namespace

bool autospc_enabled(const Model &model) {
  const auto autospc = model.params.find("AUTOSPC");
  return autospc == model.params.end() || autospc->second.empty() ||
         std::toupper(static_cast<unsigned char>(autospc->second[0])) != 'N';
}

std::vector<Spc> analysis_spcs(const Model &model, const SubCase &sc) {
  std::vector<Spc> result = model.structural_spcs(sc.spc_set);
  if (!autospc_enabled(model))
    return result;

  // Only the elements that can leave a block singular need their stiffness.
  std::unordered_map<PropertyId, bool> complete_sections;
  const auto sure_of = [&](const ElementData &elem) {
    return sure_blocks(elem, model, complete_sections);
  };
  std::vector<const ElementData *> unsure;
  // What those elements declare, which is all that can be singular here: a
  // component an RBE or MPC names is controlled by it.
  std::unordered_map<NodeId, ActiveMask> declared;
  for (const auto &elem : model.elements) {
    const SureBlocks sure = sure_of(elem);
    if (sure.translation || sure.rotation)
      continue;
    unsure.push_back(&elem);
    mark_element_dofs(elem, model, declared);
  }
  std::unordered_map<NodeId, ActiveMask> by_explicit;
  if (sc.mpc_set.value != 0)
    for (const Mpc *mpc : model.mpcs_for_set(sc.mpc_set))
      for (const auto &term : mpc->terms) {
        mark_component(by_explicit[term.node], term.dof);
        // A component named on a CD grid activates all three basic components of
        // its block, and the MPC controls only the one it names.
        const auto grid = model.nodes.find(term.node);
        if (grid != model.nodes.end() && grid->second.cd != CoordId{0})
          mark_grid_component(model, term.node, term.dof, declared);
      }

  // The translation and rotation blocks to analyse at each grid.
  std::unordered_map<NodeId, std::array<bool, 2>> analysed;
  analysed.reserve(declared.size());
  for (const auto &[nid, mask] : declared)
    if (model.nodes.contains(nid)) // a grounded end has no grid
      analysed.emplace(nid, std::array<bool, 2>{mask[0] || mask[1] || mask[2],
                                                mask[3] || mask[4] || mask[5]});
  if (analysed.empty())
    return result;
  for (const auto &elem : model.elements) {
    const SureBlocks sure = sure_of(elem);
    if (!sure.translation && !sure.rotation)
      continue;
    for (const NodeId nid : elem.nodes) {
      const auto grid = analysed.find(nid);
      if (grid == analysed.end())
        continue;
      grid->second[0] = grid->second[0] && !sure.translation;
      grid->second[1] = grid->second[1] && !sure.rotation;
    }
  }

  // A grid's own translation and rotation stiffness, in basic axes.
  using GridBlocks = std::array<Eigen::Matrix3d, 2>;
  std::unordered_map<NodeId, GridBlocks> blocks;
  blocks.reserve(analysed.size());
  for (const auto &[nid, analyse] : analysed)
    if (analyse[0] || analyse[1])
      blocks.emplace(nid, GridBlocks{Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()});
  if (blocks.empty())
    return result;

  // From every element on each grid. A sure element has no part in a block that is
  // still analysed.
  std::vector<std::vector<std::pair<NodeId, GridBlocks>>> parts(
      detail::element_chunk_count(unsure.size()));
  detail::for_element_chunks(unsure.size(), [&](const size_t chunk, const size_t first,
                                                const size_t last) {
    auto &part = parts[chunk];
    for (size_t e = first; e < last; ++e) {
      const ElementData &elem = *unsure[e];
      if (std::none_of(elem.nodes.begin(), elem.nodes.end(),
                       [&](NodeId nid) { return blocks.contains(nid); }))
        continue;
      const LocalKe ke = make_element(elem, model)->stiffness_matrix();
      // Compared with NaN every direction would look singular and be held, which
      // would hide the failure as a solution of zeros.
      if (!ke.allFinite())
        throw SolverError(std::format("AUTOSPC: element {} has a non-finite stiffness matrix",
                                      elem.id.value));
      const auto identity = local_dof_identity(elem, model);
      const size_t element_start = part.size();
      for (Eigen::Index i = 0; i < ke.rows(); ++i) {
        const auto [node, row] = identity[static_cast<size_t>(i)];
        if (!blocks.contains(node))
          continue;
        auto entry = std::find_if(part.begin() + static_cast<std::ptrdiff_t>(element_start),
                                  part.end(), [&](const auto &p) { return p.first == node; });
        if (entry == part.end()) {
          part.emplace_back(node, GridBlocks{Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()});
          entry = part.end() - 1;
        }
        for (Eigen::Index j = 0; j < ke.cols(); ++j) {
          const auto [other, col] = identity[static_cast<size_t>(j)];
          if (other == node && col / 3 == row / 3)
            entry->second[static_cast<size_t>(row / 3)](row % 3, col % 3) += ke(i, j);
        }
      }
    }
  });
  for (const auto &part : parts)
    for (const auto &[nid, add] : part) {
      GridBlocks &sum = blocks.at(nid);
      sum[0] += add[0];
      sum[1] += add[1];
    }

  std::unordered_map<NodeId, ActiveMask> by_rigid;
  mark_rigid_element_dofs(model, by_rigid);
  // Components already controlled at these grids.
  std::unordered_map<NodeId, ActiveMask> by_spc;
  for (const Spc &spc : result)
    for (int d = 1; d <= 6; ++d)
      if (spc.dofs.has(d))
        mark_component(by_spc[spc.node], d);

  const double epzero = model.param_real("EPZERO").value_or(1e-8);
  std::vector<NodeId> grids;
  grids.reserve(blocks.size());
  for (const auto &[nid, _] : blocks)
    grids.push_back(nid);
  std::sort(grids.begin(), grids.end());

  std::vector<uint8_t> held(grids.size(), 0);
  detail::for_element_chunks(grids.size(), [&](size_t, const size_t first, const size_t last) {
    for (size_t g = first; g < last; ++g) {
      const NodeId nid = grids[g];
      const GridBlocks &k = blocks.at(nid);
      const std::array<bool, 2> &analyse = analysed.at(nid);
      const Mat3 T3 = model.displacement_axes(nid);
      Eigen::Matrix3d T;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          T(i, j) = T3(i, j);
      const auto spc_mask = by_spc.find(nid);
      const auto rigid_mask = by_rigid.find(nid);
      const auto explicit_mask = by_explicit.find(nid);
      const auto declared_mask = declared.find(nid);
      const auto marked = [](const auto &masks, const auto &it, const int d) {
        return it != masks.end() && it->second[static_cast<size_t>(d)];
      };

      for (const int base : {0, 3}) {
        const size_t index = base == 0 ? 0 : 1;
        if (!analyse[index])
          continue;
        // Masks of basic components mark each CD axis with a part along them, so an
        // RBE controls each such axis.
        const auto on_axis = [&](const auto &masks, const auto &it, const int c) {
          for (int j = 0; j < 3; ++j)
            if (marked(masks, it, base + j) && std::abs(T(j, c)) > 1e-9)
              return true;
          return false;
        };
        std::vector<int> free;
        for (int c = 0; c < 3; ++c)
          if (on_axis(declared, declared_mask, c) && !marked(by_spc, spc_mask, base + c) &&
              !marked(by_explicit, explicit_mask, base + c) &&
              !on_axis(by_rigid, rigid_mask, c))
            free.push_back(c);
        const Eigen::Matrix3d block = T.transpose() * k[index] * T;
        const double largest =
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(block, Eigen::EigenvaluesOnly)
                .eigenvalues()(2);
        for (const int c : singular_components(block, std::move(free), epzero * largest))
          held[g] |= static_cast<uint8_t>(1U << (base + c));
      }
    }
  });
  for (size_t g = 0; g < grids.size(); ++g)
    if (held[g] != 0) {
      Spc added;
      added.sid = sc.spc_set;
      added.node = grids[g];
      added.dofs.mask = held[g];
      result.push_back(added);
    }
  return result;
}

std::array<double, 6> to_grid_frame(const Model &model, const NodeId nid,
                                    const std::array<double, 6> &v) {
  if (model.node(nid).cd.value == 0)
    return v;
  const Mat3 T3 = model.displacement_axes(nid);
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

DofMap build_analysis_dof_map(const Model &model, const SubCase &sc) {
  return build_analysis_dof_map(model, sc, analysis_spcs(model, sc));
}

DofMap build_analysis_dof_map(const Model &model, const SubCase &sc,
                              const std::span<const Spc> spcs) {
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
  for (const Spc &spc : spcs) {
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
                               const std::span<const Spc> spcs, DofMap &dof_map,
                               MpcHandler &mpc_handler, const bool use_spc_values) {
  std::vector<Mpc> all_mpcs;

  struct SpcValues {
    DofSet dofs;
    std::array<double, 6> values{};
  };
  std::unordered_map<NodeId, SpcValues> node_spcs;
  for (const Spc &spc : spcs) {
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
