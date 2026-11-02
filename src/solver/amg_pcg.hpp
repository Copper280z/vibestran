#pragma once
// Smoothed-aggregation PCG with diagonal equilibration and the same compensated
// refinement/forward-error acceptance used by the other iterative paths.
#define AMGCL_NO_BOOST
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include "core/sparse_matrix.hpp"
#include "core/exceptions.hpp"
#include "solver/iterative_refinement.hpp"
#include <chrono>
#include <cmath>
#include <type_traits>
#include <format>
#include "core/logger.hpp"

namespace vibestran::detail {
struct ScaledAmgMatrix {
    SparseMatrixBuilder::CsrData matrix;
    std::vector<double> scale;

    explicit ScaledAmgMatrix(const SparseMatrixBuilder::CsrData& original)
        : matrix(original.expanded_symmetric()), scale(original.n) {
        auto& k = matrix;
        int removed = 0;
        for (int i = 0; i < k.n; ++i) {
            const int start = k.row_ptr[i];
            k.row_ptr[i] -= removed;
            for (int j = start; j < k.row_ptr[i + 1]; ++j) {
                if (k.col_ind[j] == i) {
                    if (!(k.values[j] > 0) || !std::isfinite(k.values[j]))
                        throw SolverError(std::format("AMG PCG: invalid diagonal {} at row {} (n={})",
                            k.values[j], i, k.n));
                    scale[i] = 1 / std::sqrt(k.values[j]);
                }
                if (k.values[j] == 0) { ++removed; continue; }
                k.col_ind[j - removed] = k.col_ind[j];
                k.values[j - removed] = k.values[j];
            }
            if (scale[i] == 0)
                throw SolverError(std::format("AMG PCG: missing diagonal at row {} (n={})", i, k.n));
        }
        k.row_ptr[k.n] -= removed;
        k.nnz -= removed;
        k.col_ind.resize(k.nnz); k.values.resize(k.nnz);
        for (int i = 0; i < k.n; ++i)
            for (int j = k.row_ptr[i]; j < k.row_ptr[i + 1]; ++j)
                k.values[j] *= scale[i] * scale[k.col_ind[j]];
    }
};

struct AmgPcgResult {
    std::vector<double> solution;
    RefinementReport refinement;
    int iterations{0};
    // A device backend may decline the planned workspace before solving.
    bool workspace_available{true};
};

// Backend-specific transfer callbacks keep this driver independent of the
// device vector representation (CUDA, Vulkan, or the partitioned host path).
template<class Backend>
using PcgAmg = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;

template<class Backend, class Upload, class Download, class Amg = PcgAmg<Backend>>
AmgPcgResult solve_amg_pcg(const SparseMatrixBuilder::CsrData& original,
    const std::vector<double>& rhs, double tolerance, int max_iterations,
    const typename Backend::params& bp, Upload&& upload, Download&& download,
    std::type_identity<Amg> = {}) {
    const auto start = std::chrono::steady_clock::now();
    ScaledAmgMatrix scaled(original);
    const auto& k = scaled.matrix;
    typename Amg::params ap;
    ap.coarse_enough = 256;
    Amg preconditioner(std::tie(k.n, k.row_ptr, k.col_ind, k.values), ap, bp);
    using Cg = amgcl::solver::cg<Backend>;
    typename Cg::params cp;
    cp.maxiter = max_iterations > 0 ? max_iterations : 10000;
    cp.tol = std::min(tolerance, 1e-8);
    Cg cg(k.n, cp, bp);
    auto b = Backend::create_vector(k.n, bp), x = Backend::create_vector(k.n, bp);
    AmgPcgResult result;
    if constexpr (requires { Backend::prepare(bp); }) {
        if (!Backend::prepare(bp)) {
            result.workspace_available = false;
            return result;
        }
    }
    const auto ready = std::chrono::steady_clock::now();
    bool first = true;
    const auto inner = [&](const std::vector<double>& f) {
        std::vector<double> host(k.n);
        double norm = 0;
        for (int i = 0; i < k.n; ++i) {
            host[i] = f[i] * scaled.scale[i];
            norm = std::max(norm, std::abs(host[i]));
        }
        if (norm == 0) return std::vector<double>(k.n, 0);
        std::transform(host.begin(), host.end(), host.begin(), [norm](double v) { return v / norm; });
        upload(host, *b);
        amgcl::backend::clear(*x);
        cg.prm.tol = first ? std::min(tolerance, 1e-8) : 1e-3;
        first = false;
        const auto [iters, residual] = cg(preconditioner, *b, *x);
        result.iterations += static_cast<int>(iters);
        download(*x, host);
        for (int i = 0; i < k.n; ++i) host[i] *= norm * scaled.scale[i];
        log_debug(std::format("[amg-pcg] inner iterations={}, scaled residual={:.3e}", iters, residual));
        return host;
    };
    result.solution = inner(rhs);
    RefinementOptions options;
    options.tolerance = tolerance;
    options.max_corrections = 20;
    result.refinement = refine_solution(original, rhs, result.solution, inner, options);
    const auto end = std::chrono::steady_clock::now();
    log_debug(std::format("[amg-pcg] n={}, nnz={}, setup={:.3f} ms, iteration/refinement={:.3f} ms, iterations={}, estimated error={:.3e}",
        k.n, k.nnz, std::chrono::duration<double, std::milli>(ready - start).count(),
        std::chrono::duration<double, std::milli>(end - ready).count(), result.iterations,
        result.refinement.estimated_error));
    return result;
}
} // namespace vibestran::detail
