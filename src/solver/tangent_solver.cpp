// src/solver/tangent_solver.cpp
#include "solver/tangent_solver.hpp"
#include <cmath>
#include <format>
#include <numeric>
#include <spdlog/spdlog.h>
#include <utility>

namespace vibestran {

namespace {
double dot(std::span<const double> x, std::span<const double> y) {
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
}
} // namespace

TangentSolver::TangentSolver(SolverBackend& backend, const TangentSolverOptions& options)
    : backend_(backend), options_(options) {}

std::string_view TangentSolver::method() const noexcept {
    return factor_ ? factor_->method() : std::string_view{};
}

std::string_view TangentSolver::ordering() const noexcept {
    return factor_ ? factor_->ordering() : std::string_view{};
}

Status TangentSolver::update(SparseMatrixBuilder::CsrData K) {
    K_ = std::move(K);
    current_ = false;
    const bool same_pattern =
        factor_ && K_.row_ptr == row_ptr_ && K_.col_ind == col_ind_;
    if (options_.reuse_factor && same_pattern)
        return {};
    return factor_current();
}

Status TangentSolver::factor_current() {
    ++stats_.numeric_factorizations;
    if (factor_ && K_.row_ptr == row_ptr_ && K_.col_ind == col_ind_) {
        if (auto e = factor_->refactor(K_)) {
            factor_.reset();
            return e;
        }
        current_ = true;
        return {};
    }
    factor_.reset();
    ++stats_.symbolic_analyses;
    auto made = backend_.factor(K_, options_.factor);
    if (auto* e = std::get_if<Error>(&made))
        return *e;
    factor_ = std::get<std::unique_ptr<LinearFactor>>(std::move(made));
    row_ptr_ = K_.row_ptr;
    col_ind_ = K_.col_ind;
    current_ = true;
    return {};
}

Result<bool> TangentSolver::iterate(std::span<const double> b, double tolerance,
                                    std::span<double> x) {
    const double target = tolerance * std::sqrt(dot(b, b));
    std::ranges::fill(x, 0.0);
    if (target == 0)
        return true; // b = 0
    std::vector<double> r(b.begin(), b.end()), z, p;
    double rz = 0;
    // (Re)starts the recurrences from the residual r of x.
    auto restart = [&]() -> Status {
        auto solved = factor_->solve(r);
        if (auto* e = std::get_if<Error>(&solved))
            return *e;
        z = std::get<std::vector<double>>(std::move(solved));
        p = z;
        rz = dot(r, z);
        return {};
    };
    if (auto e = restart())
        return *e;
    for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
        // A nonpositive r.z or p.Kp means the factor or the tangent is not
        // positive definite (or round-off has the same effect): CG does not
        // apply.
        if (!(rz > 0))
            return false;
        ++stats_.iterations;
        const std::vector<double> q = K_.multiply(p);
        const double pq = dot(p, q);
        if (!(pq > 0))
            return false;
        const double alpha = rz / pq;
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        if (std::sqrt(dot(r, r)) <= target) {
            // The recurrence's residual drifts from the true one by round-off;
            // accept only the true residual.
            const std::vector<double> Kx = K_.multiply(x);
            for (size_t i = 0; i < r.size(); ++i)
                r[i] = b[i] - Kx[i];
            if (std::sqrt(dot(r, r)) <= target)
                return true;
            if (auto e = restart())
                return *e;
            continue;
        }
        auto solved = factor_->solve(r);
        if (auto* e = std::get_if<Error>(&solved))
            return *e;
        z = std::get<std::vector<double>>(std::move(solved));
        const double next = dot(r, z);
        const double beta = next / rz;
        rz = next;
        for (size_t i = 0; i < p.size(); ++i)
            p[i] = z[i] + beta * p[i];
    }
    return false;
}

Result<std::vector<double>> TangentSolver::solve(std::span<const double> B, int nrhs,
                                                 double tolerance) {
    if (!factor_)
        return Error{"TangentSolver::solve before a successful update()"};
    const auto n = static_cast<size_t>(K_.n);
    if (nrhs < 0 || B.size() != n * static_cast<size_t>(nrhs))
        return Error{std::format("TangentSolver::solve: {} values for {} right-hand "
                                 "sides of {} equations",
                                 B.size(), nrhs, n)};
    stats_.right_hand_sides += nrhs;
    if (!current_ && tolerance > 0) {
        std::vector<double> X(B.size());
        bool converged = true;
        const int before = stats_.iterations;
        for (int j = 0; j < nrhs && converged; ++j) {
            const auto column = static_cast<size_t>(j) * n;
            auto done = iterate(B.subspan(column, n), tolerance,
                                std::span(X).subspan(column, n));
            if (auto* e = std::get_if<Error>(&done))
                return *e;
            converged = std::get<bool>(done);
        }
        spdlog::debug("[tangent] CG with an earlier factor: {} right-hand side(s) "
                      "to {:.2e}, {} iterations, {}",
                      nrhs, tolerance, stats_.iterations - before,
                      converged ? "converged" : "refactoring");
        if (converged) {
            stats_.iterative_solves += nrhs;
            return X;
        }
        ++stats_.failed_iterative;
    }
    if (!current_)
        if (auto e = factor_current())
            return *e;
    return factor_->solve(B, nrhs);
}

} // namespace vibestran
