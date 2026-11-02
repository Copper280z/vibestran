#pragma once
#include <Eigen/SparseCholesky>
#include "core/exceptions.hpp"
#include <amgcl/backend/interface.hpp>
#include <format>

namespace vibestran::detail {
// Only the small coarse problem is factored directly. The original fine
// matrix remains iterative; no full-system CPU factorization is performed.
class AmgCoarseFactor {
public:
    // cppcheck-suppress unusedFunction -- invoked by AMGCL's templated parameter constructor
    static unsigned coarse_enough() { return 256; }
    template<class Matrix> explicit AmgCoarseFactor(const Matrix& a) {
        const auto n = static_cast<int>(amgcl::backend::rows(a));
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(amgcl::backend::nonzeros(a));
        for (int i = 0; i < n; ++i)
            for (auto j = amgcl::backend::row_begin(a, i); j; ++j)
                entries.emplace_back(i, static_cast<int>(j.col()), double(j.value()));
        Eigen::SparseMatrix<double> matrix(n, n);
        matrix.setFromTriplets(entries.begin(), entries.end());
        factor_.compute(matrix);
        if (factor_.info() != Eigen::Success || !(factor_.vectorD().minCoeff() > 0))
            throw SolverError(std::format("AMG coarse LDLT failed: n={}, nnz={}, Eigen status={}",
                n, matrix.nonZeros(), int(factor_.info())));
    }
    template<class Matrix, class Params>
    AmgCoarseFactor(const std::shared_ptr<Matrix>& a, const Params&) : AmgCoarseFactor(*a) {}
    template<class X, class Y> void operator()(const X& rhs, Y& solution) const {
        const auto n = static_cast<Eigen::Index>(rhs.size());
        Eigen::Map<Eigen::VectorXd>(solution.data(), n) = factor_.solve(Eigen::Map<const Eigen::VectorXd>(rhs.data(), n));
    }
    std::size_t bytes() const { return std::size_t(factor_.matrixL().nestedExpression().nonZeros()) * (sizeof(int) + sizeof(double)); }
private:
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor_;
};
} // namespace vibestran::detail
