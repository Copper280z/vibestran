// src/elements/rbe_constraints.cpp
// Expand RBE2 / RBE3 elements into MPC equations.

#include "elements/rbe_constraints.hpp"
#include "core/coord_sys.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <map>

namespace vibestran {

// RBE2 rigid-body constraint: gm moves with gn as a rigid body.
void expand_rbe2(const Rbe2& rbe, const Model& model, std::vector<Mpc>& out) {
    const Vec3 pos_n = model.node(rbe.gn).position;

    for (NodeId gm : rbe.gm) {
        const Vec3 pos_m = model.node(gm).position;
        Vec3 r{pos_m.x - pos_n.x, pos_m.y - pos_n.y, pos_m.z - pos_n.z};
        MpcSetId sid{0};

        auto make_mpc = [&](std::initializer_list<MpcTerm> terms) {
            Mpc mpc;
            mpc.sid = sid;
            mpc.terms = std::vector<MpcTerm>(terms);
            out.push_back(std::move(mpc));
        };

        if (rbe.cm.has(1)) {
            make_mpc({{gm, 1, +1.0}, {rbe.gn, 1, -1.0},
                      {rbe.gn, 5, -r.z}, {rbe.gn, 6, +r.y}});
        }
        if (rbe.cm.has(2)) {
            make_mpc({{gm, 2, +1.0}, {rbe.gn, 2, -1.0},
                      {rbe.gn, 4, +r.z}, {rbe.gn, 6, -r.x}});
        }
        if (rbe.cm.has(3)) {
            make_mpc({{gm, 3, +1.0}, {rbe.gn, 3, -1.0},
                      {rbe.gn, 4, -r.y}, {rbe.gn, 5, +r.x}});
        }
        if (rbe.cm.has(4))
            make_mpc({{gm, 4, +1.0}, {rbe.gn, 4, -1.0}});
        if (rbe.cm.has(5))
            make_mpc({{gm, 5, +1.0}, {rbe.gn, 5, -1.0}});
        if (rbe.cm.has(6))
            make_mpc({{gm, 6, +1.0}, {rbe.gn, 6, -1.0}});
    }
}

namespace {

Eigen::Matrix3d displacement_axes(const Model& model, const NodeId node) {
    const GridPoint& grid = model.node(node);
    if (grid.cd == CoordId{0})
        return Eigen::Matrix3d::Identity();

    const auto coord_it = model.coord_systems.find(grid.cd);
    if (coord_it == model.coord_systems.end()) {
        throw SolverError(std::format(
            "RBE3 grid {} references unresolved displacement coordinate system {}",
            node.value, grid.cd.value));
    }
    const Mat3 axes = rotation_matrix(coord_it->second, grid.position);
    Eigen::Matrix3d result;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            result(row, col) = axes(row, col);
    return result;
}

struct Rbe3Column {
    NodeId node{0};
    int component{0};
    Eigen::Matrix3d axes{Eigen::Matrix3d::Identity()};
};

Eigen::MatrixXd select_matrix(const Eigen::MatrixXd& matrix,
                              const std::vector<int>& rows,
                              const std::vector<int>& cols) {
    Eigen::MatrixXd selected(rows.size(), cols.size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < cols.size(); ++j)
            selected(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                matrix(rows[i], cols[j]);
    return selected;
}

Eigen::MatrixXd select_rows(const Eigen::MatrixXd& matrix,
                            const std::vector<int>& rows) {
    Eigen::MatrixXd selected(rows.size(), matrix.cols());
    for (size_t i = 0; i < rows.size(); ++i)
        selected.row(static_cast<Eigen::Index>(i)) = matrix.row(rows[i]);
    return selected;
}

} // namespace

// Weighted least-squares fit of independent scalar DOFs to the six rigid-body
// motions at the reference grid. A = sum(w H H^T), B_j = -w H. Components
// omitted from REFC are eliminated with a generalized Schur complement.
void expand_rbe3(const Rbe3& rbe, const Model& model, std::vector<Mpc>& out) {
    const GridPoint& reference = model.node(rbe.ref_node);
    const Eigen::Vector3d reference_position{
        reference.position.x, reference.position.y, reference.position.z};
    const Eigen::Matrix3d reference_axes =
        displacement_axes(model, rbe.ref_node);

    Eigen::Matrix<double, 6, 6> a =
        Eigen::Matrix<double, 6, 6>::Zero();
    std::vector<Eigen::Matrix<double, 6, 1>> b_columns;
    std::vector<Rbe3Column> columns;

    for (const auto& group : rbe.weight_groups) {
        for (const NodeId node : group.nodes) {
            const GridPoint& independent = model.node(node);
            const Eigen::Vector3d position{
                independent.position.x, independent.position.y,
                independent.position.z};
            const Eigen::Vector3d relative =
                reference_axes.transpose() * (position - reference_position);
            const Eigen::Matrix3d independent_axes =
                displacement_axes(model, node);
            const Eigen::Matrix3d axes_in_reference =
                reference_axes.transpose() * independent_axes;

            for (int component = 1; component <= 6; ++component) {
                if (!group.component.has(component))
                    continue;
                Eigen::Matrix<double, 6, 1> h =
                    Eigen::Matrix<double, 6, 1>::Zero();
                const Eigen::Vector3d direction = axes_in_reference.col(
                    component <= 3 ? component - 1 : component - 4);
                if (component <= 3) {
                    h.head<3>() = direction;
                    h.tail<3>() = relative.cross(direction);
                } else {
                    h.tail<3>() = direction;
                }
                a.noalias() += group.weight * h * h.transpose();
                b_columns.push_back(-group.weight * h);
                columns.push_back({node, component, independent_axes});
            }
        }
    }

    if (columns.empty())
        throw SolverError(std::format(
            "RBE3 {} has no active independent DOFs", rbe.eid.value));

    Eigen::MatrixXd b(6, static_cast<Eigen::Index>(b_columns.size()));
    for (size_t col = 0; col < b_columns.size(); ++col)
        b.col(static_cast<Eigen::Index>(col)) = b_columns[col];

    std::vector<int> retained;
    std::vector<int> discarded;
    for (int component = 0; component < 6; ++component) {
        if (rbe.refc.has(component + 1))
            retained.push_back(component);
        else
            discarded.push_back(component);
    }
    if (retained.empty())
        throw SolverError(std::format(
            "RBE3 {} has no active REFC components", rbe.eid.value));

    Eigen::MatrixXd a_reduced = select_matrix(a, retained, retained);
    Eigen::MatrixXd b_reduced = select_rows(b, retained);
    if (!discarded.empty()) {
        const Eigen::MatrixXd a_dd = select_matrix(a, discarded, discarded);
        const Eigen::MatrixXd a_dr = select_matrix(a, discarded, retained);
        const Eigen::MatrixXd a_rd = select_matrix(a, retained, discarded);
        const Eigen::MatrixXd b_d = select_rows(b, discarded);
        Eigen::MatrixXd rhs(a_dr.rows(), a_dr.cols() + b_d.cols());
        rhs << a_dr, b_d;

        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> decomposition(a_dd);
        decomposition.setThreshold(1e-12);
        const Eigen::MatrixXd eliminated = decomposition.solve(rhs);
        const double residual = (a_dd * eliminated - rhs).cwiseAbs().maxCoeff();
        const double scale = std::max(1.0, rhs.cwiseAbs().maxCoeff());
        if (residual > 1e-10 * scale) {
            throw SolverError(std::format(
                "RBE3 {} has an inconsistent rank-deficient omitted-component "
                "system (residual {:.3e})", rbe.eid.value, residual));
        }
        a_reduced.noalias() -= a_rd * eliminated.leftCols(
            static_cast<Eigen::Index>(retained.size()));
        b_reduced.noalias() -= a_rd * eliminated.rightCols(b_d.cols());
    }

    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> rank_check(a_reduced);
    rank_check.setThreshold(1e-12);
    if (rank_check.rank() < static_cast<Eigen::Index>(retained.size())) {
        throw SolverError(std::format(
            "RBE3 {} cannot determine all {} requested REFC components; rank is {}",
            rbe.eid.value, retained.size(), rank_check.rank()));
    }

    for (size_t row = 0; row < retained.size(); ++row) {
        std::map<std::pair<NodeId, int>, double> coefficients;
        for (size_t col = 0; col < retained.size(); ++col) {
            const int local_component = retained[col];
            const int offset = local_component < 3 ? 0 : 3;
            const Eigen::Vector3d axis =
                reference_axes.col(local_component % 3);
            for (int basic = 0; basic < 3; ++basic) {
                coefficients[{rbe.ref_node, offset + basic + 1}] +=
                    a_reduced(static_cast<Eigen::Index>(row),
                              static_cast<Eigen::Index>(col)) * axis(basic);
            }
        }
        for (size_t col = 0; col < columns.size(); ++col) {
            const Rbe3Column& source = columns[col];
            const int offset = source.component <= 3 ? 0 : 3;
            const Eigen::Vector3d axis = source.axes.col(
                source.component <= 3 ? source.component - 1
                                      : source.component - 4);
            for (int basic = 0; basic < 3; ++basic) {
                coefficients[{source.node, offset + basic + 1}] +=
                    b_reduced(static_cast<Eigen::Index>(row),
                              static_cast<Eigen::Index>(col)) * axis(basic);
            }
        }

        const double scale = std::max(
            a_reduced.row(static_cast<Eigen::Index>(row)).cwiseAbs().maxCoeff(),
            b_reduced.row(static_cast<Eigen::Index>(row)).cwiseAbs().maxCoeff());
        const double tolerance =
            10.0 * std::numeric_limits<double>::epsilon() * scale;
        Mpc mpc;
        mpc.sid = MpcSetId{0};

        auto preferred = coefficients.end();
        for (auto it = coefficients.begin(); it != coefficients.end(); ++it) {
            if (it->first.first != rbe.ref_node ||
                std::abs(it->second) <= tolerance)
                continue;
            if (preferred == coefficients.end() ||
                std::abs(it->second) > std::abs(preferred->second))
                preferred = it;
        }
        if (preferred == coefficients.end())
            throw SolverError(std::format(
                "RBE3 {} reduced equation {} has no reference-grid pivot",
                rbe.eid.value, row + 1));
        mpc.terms.push_back(
            {preferred->first.first, preferred->first.second, preferred->second});
        coefficients.erase(preferred);
        for (const auto& [key, coefficient] : coefficients) {
            if (std::abs(coefficient) > tolerance)
                mpc.terms.push_back({key.first, key.second, coefficient});
        }
        out.push_back(std::move(mpc));
    }
}

} // namespace vibestran
