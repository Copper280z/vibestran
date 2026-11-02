#pragma once
#include "amg_pcg.hpp"
#include "amg_coarse.hpp"
#include <functional>

namespace vibestran::detail {
// Fine-grid products may use a fixed device row prefix; all coarse operators
// and vectors remain in host RAM. The hierarchy is reused across corrections.
class AmgSpmv {
public:
    virtual ~AmgSpmv() = default;
    virtual void multiply(std::span<const double> x, std::span<double> y) = 0;
};
using AmgSpmvFactory = std::function<std::unique_ptr<AmgSpmv>(const SparseMatrixBuilder::CsrData&)>;

struct HybridAmgMatrix {
    using value_type = double;
    using col_type = int;
    using ptr_type = int;
    using HostMatrix = amgcl::backend::builtin<double, int, int>::matrix;
    std::shared_ptr<HostMatrix> host;
    SparseMatrixBuilder::CsrData fine;
    mutable std::unique_ptr<AmgSpmv> gpu;
    mutable std::vector<double> product;

    HybridAmgMatrix(std::shared_ptr<HostMatrix> a, AmgSpmvFactory* factory) : host(std::move(a)) {
        if (factory && *factory) {
            fine.n = static_cast<int>(host->nrows); fine.nnz = static_cast<int>(host->nnz);
            fine.row_ptr.assign(host->ptr, host->ptr + host->nrows + 1);
            fine.col_ind.assign(host->col, host->col + host->nnz);
            fine.values.assign(host->val, host->val + host->nnz);
            gpu = (*factory)(fine);
            *factory = {}; // only the finest square operator consumes device storage
            product.resize(fine.n);
        }
    }
    std::size_t rows() const { return host->nrows; }
    std::size_t cols() const { return host->ncols; }
    std::size_t nonzeros() const { return host->nnz; }
    std::size_t bytes() const { return amgcl::backend::bytes(*host) + product.size() * sizeof(double); }
};

struct HybridAmgBackend : amgcl::backend::builtin<double, int, int> {
    using matrix = HybridAmgMatrix;
    using direct_solver = AmgCoarseFactor;
    struct provides_row_iterator : std::false_type {};
    struct params : amgcl::backend::builtin<double, int, int>::params { AmgSpmvFactory* factory{nullptr}; };
    // cppcheck-suppress unusedFunction -- AMGCL invokes this backend hook when building the hierarchy
    static std::shared_ptr<matrix> copy_matrix(std::shared_ptr<HybridAmgMatrix::HostMatrix> a, const params& p) {
        return std::make_shared<matrix>(std::move(a), p.factory);
    }
    // cppcheck-suppress unusedFunction -- AMGCL invokes this backend hook for the coarse factor
    static std::shared_ptr<direct_solver> create_solver(std::shared_ptr<HybridAmgMatrix::HostMatrix> a, const params& p) {
        return std::make_shared<direct_solver>(a, p);
    }
};
} // namespace vibestran::detail

namespace amgcl::backend {
template<class A, class X, class B, class Y>
struct spmv_impl<A, vibestran::detail::HybridAmgMatrix, X, B, Y> {
    static void apply(A alpha, const vibestran::detail::HybridAmgMatrix& a, const X& x, B beta, Y& y) {
        if (!a.gpu) { spmv(alpha, *a.host, x, beta, y); return; }
        a.gpu->multiply(std::span<const double>(x.data(), x.size()), a.product);
        const auto n = static_cast<std::ptrdiff_t>(a.rows());
        #pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < n; ++i)
            y[i] = alpha * a.product[i] + (beta == 0 ? 0 : beta * y[i]);
    }
};
template<class X, class Y, class Z>
struct residual_impl<vibestran::detail::HybridAmgMatrix, X, Y, Z> {
    static void apply(const X& rhs, const vibestran::detail::HybridAmgMatrix& a, const Y& x, Z& r) {
        copy(rhs, r); spmv(-1.0, a, x, 1.0, r);
    }
};
} // namespace amgcl::backend

namespace vibestran::detail {
inline AmgPcgResult solve_hybrid_amg(const SparseMatrixBuilder::CsrData& k, const std::vector<double>& f,
    double tolerance, int max_iterations, AmgSpmvFactory factory) {
    HybridAmgBackend::params p;
    p.factory = &factory;
    return solve_amg_pcg<HybridAmgBackend>(k, f, tolerance, max_iterations, p,
        [](const auto& host, auto& work) { std::copy(host.begin(), host.end(), work.data()); },
        [](const auto& work, auto& host) { std::copy(work.data(), work.data() + work.size(), host.begin()); });
}
} // namespace vibestran::detail
