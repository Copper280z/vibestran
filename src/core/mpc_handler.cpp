// src/core/mpc_handler.cpp
// Master-slave elimination for multi-point constraints.

#include "core/mpc_handler.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace vibestran {

void MpcHandler::build(std::span<const Mpc* const> mpcs, DofMap& dof_map) {
    eliminations_.clear();
    dep_to_elim_.clear();
    index_map_.clear();

    // Save pre-MPC dof_map
    full_dof_map_ = dof_map;
    n_full_ = dof_map.num_free_dofs();

    if (mpcs.empty()) {
        // No MPCs: trivial identity mapping
        index_map_.resize(static_cast<size_t>(n_full_));
        for (int i = 0; i < n_full_; ++i)
            index_map_[static_cast<size_t>(i)] = static_cast<EqIndex>(i);
        n_reduced_ = n_full_;
        return;
    }

    // Build eliminations using pre-MPC eq indices
    std::vector<std::pair<NodeId, int>> dep_dofs_to_constrain;
    std::unordered_set<EqIndex> selected_dep_eqs;
    std::vector<EqIndex> selected_dep_order;
    std::vector<std::map<EqIndex, double>> constraint_rows;
    std::vector<double> constraint_right_hand_sides;

    for (const Mpc* mpc : mpcs) {
        if (mpc->terms.empty())
            continue;

        std::map<EqIndex, double> row_coefficients;
        std::unordered_map<EqIndex, std::pair<NodeId, int>> dof_identity;
        std::vector<EqIndex> candidates;
        for (const MpcTerm& term : mpc->terms) {
            const EqIndex eq = dof_map.eq_index(term.node, term.dof - 1);
            if (eq == CONSTRAINED_DOF)
                continue;
            if (!row_coefficients.contains(eq)) {
                candidates.push_back(eq);
                dof_identity.emplace(eq,
                    std::pair<NodeId, int>{term.node, term.dof - 1});
            }
            row_coefficients[eq] += term.coeff;
        }
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                [&](const EqIndex eq) {
                    return std::abs(row_coefficients.at(eq)) < 1e-30;
                }),
            candidates.end());

        // Nastran MPC, RBE2, and RBE3 rows designate the first term as the
        // preferred dependent. If another equation already uses that pivot,
        // retain both equations by pivoting this row on its next available
        // free term. Choosing by coefficient magnitude is not valid because it
        // reverses rigid-element dependencies when a lever arm exceeds one.
        EqIndex dep_eq = CONSTRAINED_DOF;
        for (const EqIndex candidate : candidates) {
            if (selected_dep_eqs.contains(candidate))
                continue;

            // A column absent from all preceding rows extends the pivot matrix
            // triangularly. Otherwise use pivoted LU so coupled RBE3 blocks do
            // not select a singular dependent-column set.
            const bool appears_in_previous_row = std::any_of(
                constraint_rows.begin(), constraint_rows.end(),
                [&](const auto& previous_row) {
                    const auto value = previous_row.find(candidate);
                    return value != previous_row.end() &&
                           std::abs(value->second) >= 1e-30;
                });
            bool full_rank = !appears_in_previous_row;
            if (appears_in_previous_row) {
                const size_t new_size = selected_dep_order.size() + 1;
                Eigen::MatrixXd pivot_matrix = Eigen::MatrixXd::Zero(
                    static_cast<Eigen::Index>(new_size),
                    static_cast<Eigen::Index>(new_size));
                for (size_t row = 0; row < constraint_rows.size(); ++row) {
                    for (size_t col = 0; col < selected_dep_order.size(); ++col) {
                        const auto value = constraint_rows[row].find(
                            selected_dep_order[col]);
                        if (value != constraint_rows[row].end()) {
                            pivot_matrix(static_cast<Eigen::Index>(row),
                                         static_cast<Eigen::Index>(col)) =
                                value->second;
                        }
                    }
                    const auto value = constraint_rows[row].find(candidate);
                    if (value != constraint_rows[row].end()) {
                        pivot_matrix(static_cast<Eigen::Index>(row),
                                     static_cast<Eigen::Index>(new_size - 1)) =
                            value->second;
                    }
                }
                for (size_t col = 0; col < selected_dep_order.size(); ++col) {
                    const auto value = row_coefficients.find(selected_dep_order[col]);
                    if (value != row_coefficients.end()) {
                        pivot_matrix(static_cast<Eigen::Index>(new_size - 1),
                                     static_cast<Eigen::Index>(col)) = value->second;
                    }
                }
                pivot_matrix(static_cast<Eigen::Index>(new_size - 1),
                             static_cast<Eigen::Index>(new_size - 1)) =
                    row_coefficients.at(candidate);
                Eigen::FullPivLU<Eigen::MatrixXd> pivot_check(pivot_matrix);
                pivot_check.setThreshold(1e-12);
                full_rank = pivot_check.rank() ==
                    static_cast<Eigen::Index>(new_size);
            }
            if (full_rank) {
                dep_eq = candidate;
                break;
            }
        }
        if (dep_eq == CONSTRAINED_DOF) {
            if (constraint_rows.empty()) {
                if (std::abs(mpc->rhs) < 1e-30)
                    continue;
                throw SolverError(std::format(
                    "MPC set {} has a nonzero right-hand side but no free pivot DOF",
                    mpc->sid.value));
            }

            const size_t size = selected_dep_order.size();
            Eigen::MatrixXd pivot_matrix(size, size);
            Eigen::VectorXd current_row(size);
            for (size_t row = 0; row < size; ++row) {
                for (size_t col = 0; col < size; ++col) {
                    const auto value = constraint_rows[row].find(
                        selected_dep_order[col]);
                    pivot_matrix(static_cast<Eigen::Index>(row),
                                 static_cast<Eigen::Index>(col)) =
                        value == constraint_rows[row].end() ? 0.0 : value->second;
                }
                const auto value = row_coefficients.find(selected_dep_order[row]);
                current_row(static_cast<Eigen::Index>(row)) =
                    value == row_coefficients.end() ? 0.0 : value->second;
            }
            const Eigen::VectorXd combination =
                pivot_matrix.transpose().fullPivLu().solve(current_row);
            double implied_rhs = 0.0;
            for (size_t row = 0; row < size; ++row)
                implied_rhs += combination(static_cast<Eigen::Index>(row)) *
                               constraint_right_hand_sides[row];
            const double rhs_scale = std::max({1.0, std::abs(mpc->rhs),
                                               std::abs(implied_rhs)});
            if (std::abs(mpc->rhs - implied_rhs) <= 1e-12 * rhs_scale)
                continue;
            throw SolverError(std::format(
                "MPC set {} contains inconsistent dependent equations "
                "(right-hand side {:.6e}, implied {:.6e})",
                mpc->sid.value, mpc->rhs, implied_rhs));
        }
        selected_dep_eqs.insert(dep_eq);
        selected_dep_order.push_back(dep_eq);
        constraint_rows.push_back(row_coefficients);
        constraint_right_hand_sides.push_back(mpc->rhs);

        // Build elimination with pre-MPC eq indices
        MpcElimination elim;
        elim.dep = dep_eq;
        const double a_dep = row_coefficients.at(dep_eq);
        elim.offset = mpc->rhs / a_dep;
        for (const auto& [eq, coefficient] : row_coefficients) {
            if (eq == dep_eq || std::abs(coefficient) < 1e-30)
                continue;
            elim.terms.emplace_back(eq, -coefficient / a_dep);
        }
        eliminations_.push_back(std::move(elim));
        dep_dofs_to_constrain.push_back(dof_identity.at(dep_eq));
    }

    if (eliminations_.empty()) {
        // All MPCs skipped
        index_map_.resize(static_cast<size_t>(n_full_));
        for (int i = 0; i < n_full_; ++i)
            index_map_[static_cast<size_t>(i)] = static_cast<EqIndex>(i);
        n_reduced_ = n_full_;
        return;
    }

    // Reduce the dependency graph by strongly connected component. Acyclic
    // chains are ordinary substitution. A coupled component is a small linear
    // system (I-C)x=r, which is valid when full rank; singular coupled
    // definitions remain an error. Redundant rows were removed above.
    std::unordered_map<EqIndex, size_t> raw_dep_to_elim;
    for (size_t i = 0; i < eliminations_.size(); ++i) {
        if (!raw_dep_to_elim.emplace(eliminations_[i].dep, i).second) {
            throw SolverError(std::format(
                "MPC dependent DOF equation {} is defined more than once",
                eliminations_[i].dep));
        }
    }

    const std::vector<MpcElimination> raw_eliminations = eliminations_;
    const size_t count = eliminations_.size();
    std::vector<int> graph_index(count, -1);
    std::vector<int> low_link(count, -1);
    std::vector<bool> on_stack(count, false);
    std::vector<size_t> stack;
    std::vector<std::vector<size_t>> components;
    std::vector<size_t> component_of(count, 0);
    int next_index = 0;

    std::function<void(size_t)> find_components = [&](const size_t current) {
        graph_index[current] = next_index;
        low_link[current] = next_index++;
        stack.push_back(current);
        on_stack[current] = true;

        for (const auto& [ind_eq, unused] : raw_eliminations[current].terms) {
            (void)unused;
            const auto child_it = raw_dep_to_elim.find(ind_eq);
            if (child_it == raw_dep_to_elim.end())
                continue;
            const size_t child = child_it->second;
            if (graph_index[child] == -1) {
                find_components(child);
                low_link[current] = std::min(low_link[current], low_link[child]);
            } else if (on_stack[child]) {
                low_link[current] = std::min(low_link[current], graph_index[child]);
            }
        }

        if (low_link[current] != graph_index[current])
            return;
        const size_t component_id = components.size();
        components.emplace_back();
        while (true) {
            const size_t member = stack.back();
            stack.pop_back();
            on_stack[member] = false;
            component_of[member] = component_id;
            components.back().push_back(member);
            if (member == current)
                break;
        }
    };
    for (size_t i = 0; i < count; ++i)
        if (graph_index[i] == -1)
            find_components(i);

    std::vector<int> component_state(components.size(), 0);
    std::function<void(size_t)> resolve_component = [&](const size_t component_id) {
        if (component_state[component_id] == 2)
            return;
        if (component_state[component_id] == 1)
            throw SolverError("Internal MPC component graph cycle");
        component_state[component_id] = 1;

        const auto& members = components[component_id];
        std::unordered_map<size_t, size_t> local_index;
        for (size_t local = 0; local < members.size(); ++local)
            local_index[members[local]] = local;

        std::vector<std::map<EqIndex, double>> root_terms(members.size());
        std::vector<double> offsets(members.size(), 0.0);
        Eigen::MatrixXd system = Eigen::MatrixXd::Identity(
            static_cast<Eigen::Index>(members.size()),
            static_cast<Eigen::Index>(members.size()));

        for (size_t row = 0; row < members.size(); ++row) {
            const MpcElimination& raw = raw_eliminations[members[row]];
            offsets[row] = raw.offset;
            for (const auto& [ind_eq, coeff] : raw.terms) {
                const auto child_it = raw_dep_to_elim.find(ind_eq);
                if (child_it == raw_dep_to_elim.end()) {
                    root_terms[row][ind_eq] += coeff;
                    continue;
                }
                const size_t child = child_it->second;
                const size_t child_component = component_of[child];
                if (child_component == component_id) {
                    system(static_cast<Eigen::Index>(row),
                           static_cast<Eigen::Index>(local_index.at(child))) -= coeff;
                    continue;
                }
                resolve_component(child_component);
                const MpcElimination& resolved_child = eliminations_[child];
                offsets[row] += coeff * resolved_child.offset;
                for (const auto& [root_eq, root_coeff] : resolved_child.terms)
                    root_terms[row][root_eq] += coeff * root_coeff;
            }
        }

        std::map<EqIndex, size_t> root_columns;
        for (const auto& row_terms : root_terms)
            for (const auto& [root_eq, unused] : row_terms) {
                (void)unused;
                if (!root_columns.contains(root_eq))
                    root_columns[root_eq] = root_columns.size();
            }
        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(members.size()),
            static_cast<Eigen::Index>(1 + root_columns.size()));
        for (size_t row = 0; row < members.size(); ++row) {
            rhs(static_cast<Eigen::Index>(row), 0) = offsets[row];
            for (const auto& [root_eq, coeff] : root_terms[row])
                rhs(static_cast<Eigen::Index>(row),
                    static_cast<Eigen::Index>(1 + root_columns.at(root_eq))) = coeff;
        }

        Eigen::FullPivLU<Eigen::MatrixXd> decomposition(system);
        decomposition.setThreshold(1e-12);
        if (decomposition.rank() < static_cast<Eigen::Index>(members.size())) {
            throw SolverError("MPC dependency graph contains a singular cycle; "
                              "check redundant or circular constraint equations");
        }
        const Eigen::MatrixXd solution = decomposition.solve(rhs);
        std::vector<EqIndex> roots(root_columns.size());
        for (const auto& [root_eq, column] : root_columns)
            roots[column] = root_eq;
        for (size_t row = 0; row < members.size(); ++row) {
            MpcElimination resolved;
            resolved.dep = raw_eliminations[members[row]].dep;
            resolved.offset = solution(static_cast<Eigen::Index>(row), 0);
            for (size_t col = 0; col < roots.size(); ++col) {
                const double coeff = solution(
                    static_cast<Eigen::Index>(row),
                    static_cast<Eigen::Index>(col + 1));
                if (std::abs(coeff) > 1e-14)
                    resolved.terms.emplace_back(roots[col], coeff);
            }
            eliminations_[members[row]] = std::move(resolved);
        }
        component_state[component_id] = 2;
    };
    for (size_t component = 0; component < components.size(); ++component)
        resolve_component(component);

    // Constrain dep DOFs in the main dof_map
    dof_map.constrain_batch(dep_dofs_to_constrain);

    // Build index_map_: pre-MPC eq → post-MPC eq
    std::unordered_set<EqIndex> dep_set;
    for (const auto& e : eliminations_)
        dep_set.insert(e.dep);

    std::vector<EqIndex> sorted_deps(dep_set.begin(), dep_set.end());
    std::sort(sorted_deps.begin(), sorted_deps.end());

    index_map_.resize(static_cast<size_t>(n_full_), CONSTRAINED_DOF);
    int removed = 0;
    size_t dp = 0;
    for (int i = 0; i < n_full_; ++i) {
        // Advance dep pointer
        while (dp < sorted_deps.size() && sorted_deps[dp] < static_cast<EqIndex>(i))
            ++dp;
        if (dp < sorted_deps.size() && sorted_deps[dp] == static_cast<EqIndex>(i)) {
            index_map_[static_cast<size_t>(i)] = CONSTRAINED_DOF;
            ++removed;
        } else {
            index_map_[static_cast<size_t>(i)] = static_cast<EqIndex>(i - removed);
        }
    }
    n_reduced_ = n_full_ - removed;

    // Update elim.terms from pre-MPC to post-MPC (reduced) eq indices
    for (auto& elim : eliminations_) {
        for (auto& [ind_eq, c] : elim.terms) {
            if (ind_eq >= 0 && ind_eq < n_full_)
                ind_eq = index_map_[static_cast<size_t>(ind_eq)];
        }
        elim.terms.erase(
            std::remove_if(elim.terms.begin(), elim.terms.end(),
                           [](const auto& p) { return p.first == CONSTRAINED_DOF; }),
            elim.terms.end());
    }

    // Build O(1) lookup from dep eq → elimination index
    dep_to_elim_.clear();
    dep_to_elim_.reserve(eliminations_.size());
    for (int i = 0; i < static_cast<int>(eliminations_.size()); ++i)
        dep_to_elim_[eliminations_[i].dep] = i;
}

bool MpcHandler::has_affine_offsets() const noexcept {
    return std::any_of(eliminations_.begin(), eliminations_.end(),
                       [](const MpcElimination& elim) {
                           return std::abs(elim.offset) > 1e-30;
                       });
}

EqIndex MpcHandler::reduced_index(EqIndex full) const {
    if (full < 0 || full >= static_cast<EqIndex>(index_map_.size()))
        return CONSTRAINED_DOF;
    return index_map_[static_cast<size_t>(full)];
}

std::vector<std::pair<EqIndex, double>>
MpcHandler::t_column(EqIndex full_eq) const {
    if (full_eq == CONSTRAINED_DOF)
        return {};
    // Check if it's a dep DOF (O(1) lookup)
    auto it = dep_to_elim_.find(full_eq);
    if (it != dep_to_elim_.end())
        return eliminations_[it->second].terms;
    // Free non-dep DOF
    EqIndex r = reduced_index(full_eq);
    if (r == CONSTRAINED_DOF)
        return {};
    return {{r, 1.0}};
}

double MpcHandler::t_offset(const EqIndex full_eq) const {
    const auto it = dep_to_elim_.find(full_eq);
    return it == dep_to_elim_.end() ? 0.0 : eliminations_[it->second].offset;
}

// cppcheck-suppress unusedFunction -- called from linear_static.cpp
void MpcHandler::apply_to_stiffness(std::span<const EqIndex> gdofs_full,
                                     std::span<const double> ke,
                                     SparseMatrixBuilder& K_builder) const {
    int ndof = static_cast<int>(gdofs_full.size());

    // Fast path: if no element DOF is a dependent MPC DOF, use direct assembly
    // with simple index remapping (avoids dense T^T*Ke*T transformation).
    {
        bool any_dep = false;
        if (has_constraints()) {
            for (int i = 0; i < ndof && !any_dep; ++i) {
                EqIndex full = gdofs_full[i];
                if (full != CONSTRAINED_DOF && dep_to_elim_.count(full))
                    any_dep = true;
            }
        }
        if (!any_dep) {
            std::vector<int32_t> gd32(static_cast<size_t>(ndof));
            for (int i = 0; i < ndof; ++i) {
                EqIndex full = gdofs_full[i];
                gd32[static_cast<size_t>(i)] =
                    (full == CONSTRAINED_DOF || full < 0 ||
                     full >= static_cast<EqIndex>(index_map_.size()))
                        ? CONSTRAINED_DOF
                        : index_map_[static_cast<size_t>(full)];
            }
            K_builder.add_element_stiffness(gd32, ke);
            return;
        }
    }

    // Build compact T: collect unique active reduced column indices.
    // For each element DOF i, t_column(gdofs_full[i]) yields (reduced_eq, coeff) pairs.
    // active_cols = union of all reduced_eq values across all element DOFs.
    std::vector<EqIndex> active_cols;
    active_cols.reserve(static_cast<size_t>(ndof) * 2);

    // Store column data per element row: T_cols[i] = list of (col_in_active, coeff)
    using ColCoeff = std::vector<std::pair<int, double>>;
    std::vector<ColCoeff> row_entries(static_cast<size_t>(ndof));

    for (int i = 0; i < ndof; ++i) {
        auto tc = t_column(gdofs_full[i]);
        for (const auto& [r, c] : tc) {
            if (r == CONSTRAINED_DOF || r < 0) continue;
            // Find or add to active_cols
            int col_idx = -1;
            for (int j = 0; j < static_cast<int>(active_cols.size()); ++j)
                if (active_cols[j] == r) { col_idx = j; break; }
            if (col_idx < 0) {
                col_idx = static_cast<int>(active_cols.size());
                active_cols.push_back(r);
            }
            row_entries[static_cast<size_t>(i)].emplace_back(col_idx, c);
        }
    }

    int na = static_cast<int>(active_cols.size());
    if (na == 0) return; // all DOFs constrained

    // Build dense T_compact: ndof × na
    std::vector<double> T(static_cast<size_t>(ndof * na), 0.0);
    for (int i = 0; i < ndof; ++i)
        for (const auto& [ci, c] : row_entries[static_cast<size_t>(i)])
            T[static_cast<size_t>(i * na + ci)] += c;

    // tmp = Ke * T_compact  (ndof × na)
    std::vector<double> tmp(static_cast<size_t>(ndof * na), 0.0);
    for (int i = 0; i < ndof; ++i)
        for (int j = 0; j < na; ++j)
            for (int k = 0; k < ndof; ++k)
                tmp[static_cast<size_t>(i * na + j)] +=
                    ke[static_cast<size_t>(i * ndof + k)] *
                    T[static_cast<size_t>(k * na + j)];

    // k_red = T_compact^T * tmp  (na × na)
    std::vector<double> k_red(static_cast<size_t>(na * na), 0.0);
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < na; ++j)
            for (int k = 0; k < ndof; ++k)
                k_red[static_cast<size_t>(i * na + j)] +=
                    T[static_cast<size_t>(k * na + i)] *
                    tmp[static_cast<size_t>(k * na + j)];

    // Scatter k_red into K_builder using active_cols as global indices
    std::vector<int32_t> rdofs(static_cast<size_t>(na));
    for (int i = 0; i < na; ++i)
        rdofs[static_cast<size_t>(i)] = static_cast<int32_t>(active_cols[i]);

    K_builder.add_element_stiffness(rdofs, k_red);
}

void MpcHandler::apply_to_force(std::span<const EqIndex> gdofs_full,
                                 std::span<const double> fe,
                                 std::vector<double>& F) const {
    // Fast path: if no element DOF is dependent, use direct index mapping
    bool any_dep = false;
    if (has_constraints()) {
        for (size_t i = 0; i < gdofs_full.size() && !any_dep; ++i) {
            EqIndex full = gdofs_full[i];
            if (full != CONSTRAINED_DOF && dep_to_elim_.count(full))
                any_dep = true;
        }
    }
    if (!any_dep) {
        for (size_t i = 0; i < gdofs_full.size(); ++i) {
            EqIndex full = gdofs_full[i];
            if (full == CONSTRAINED_DOF || full < 0 ||
                full >= static_cast<EqIndex>(index_map_.size()))
                continue;
            EqIndex r = index_map_[static_cast<size_t>(full)];
            if (r >= 0 && r < static_cast<EqIndex>(F.size()))
                F[static_cast<size_t>(r)] += fe[i];
        }
        return;
    }
    for (size_t i = 0; i < gdofs_full.size(); ++i) {
        for (const auto& [r, c] : t_column(gdofs_full[i])) {
            if (r >= 0 && r < static_cast<EqIndex>(F.size()))
                F[static_cast<size_t>(r)] += c * fe[i];
        }
    }
}

void MpcHandler::apply_prescribed_displacement_load(
    std::span<const EqIndex> gdofs_full, std::span<const double> ke,
    std::vector<double>& F) const {
    if (!has_affine_offsets())
        return;

    const int ndof = static_cast<int>(gdofs_full.size());
    std::vector<double> ke_u0(static_cast<size_t>(ndof), 0.0);
    for (int i = 0; i < ndof; ++i) {
        for (int j = 0; j < ndof; ++j) {
            ke_u0[static_cast<size_t>(i)] +=
                ke[static_cast<size_t>(i * ndof + j)] *
                t_offset(gdofs_full[static_cast<size_t>(j)]);
        }
    }
    for (int i = 0; i < ndof; ++i) {
        for (const auto& [reduced_eq, coeff] :
             t_column(gdofs_full[static_cast<size_t>(i)])) {
            if (reduced_eq >= 0 && reduced_eq < static_cast<EqIndex>(F.size()))
                F[static_cast<size_t>(reduced_eq)] -=
                    coeff * ke_u0[static_cast<size_t>(i)];
        }
    }
}

void MpcHandler::recover_dependent_dofs(std::vector<double>& u_free_full,
                                         const std::vector<double>& u_reduced) const {
    // Fill free (non-dep) entries from u_reduced
    for (size_t i = 0; i < index_map_.size(); ++i) {
        EqIndex r = index_map_[i];
        if (r != CONSTRAINED_DOF && r < static_cast<EqIndex>(u_reduced.size()))
            u_free_full[i] = u_reduced[static_cast<size_t>(r)];
    }
    // Compute dep DOF values: u_dep = sum_j c_j * u_reduced[j]
    for (const auto& elim : eliminations_) {
        double val = elim.offset;
        for (const auto& [r, c] : elim.terms) {
            if (r >= 0 && r < static_cast<EqIndex>(u_reduced.size()))
                val += c * u_reduced[static_cast<size_t>(r)];
        }
        if (elim.dep >= 0 && elim.dep < static_cast<EqIndex>(u_free_full.size()))
            u_free_full[static_cast<size_t>(elim.dep)] = val;
    }
}

} // namespace vibestran
