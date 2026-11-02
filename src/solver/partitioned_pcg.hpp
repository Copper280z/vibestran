#pragma once
#include "core/sparse_matrix.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>
#ifdef HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

namespace vibestran::detail {
// Select a persistent CSR row prefix using actual nonzero counts. Device
// storage is the matrix prefix, a complete input vector and a prefix output.
// Reserve 64 KiB for allocation alignment/driver bookkeeping.
inline int pcg_prefix_rows(const SparseMatrixBuilder::CsrData& k,
                          std::size_t budget, std::size_t scalar_bytes,
                          std::size_t storage_limit = std::numeric_limits<std::size_t>::max()) {
    if (std::size_t(k.n) * scalar_bytes > storage_limit) return 0;
    const std::size_t fixed = std::size_t(k.n) * scalar_bytes + 65536;
    int rows = 0;
    for (int end = 1; end <= k.n; ++end) {
        const std::size_t required = fixed + std::size_t(end) * scalar_bytes +
            (std::size_t(end) + 1) * sizeof(int) +
            std::size_t(k.row_ptr[end]) * (sizeof(int) + scalar_bytes);
        if (required > budget) break;
        if ((std::size_t(end) + 1) * sizeof(int) > storage_limit ||
            std::size_t(end) * scalar_bytes > storage_limit ||
            std::size_t(k.row_ptr[end]) * std::max(sizeof(int), scalar_bytes) > storage_limit) break;
        rows = end;
    }
    return rows;
}

template<class T>
void pcg_cpu_rows(const SparseMatrixBuilder::CsrData& k, std::span<const T> values,
                  std::span<const T> p, std::span<double> ap, int first) {
    const auto multiply = [&](int begin, int end) {
        for (int row = begin; row < end; ++row) {
            double sum = 0;
            for (int j = k.row_ptr[row]; j < k.row_ptr[row + 1]; ++j)
                sum += double(values[j]) * double(p[k.col_ind[j]]);
            ap[row] = sum;
        }
    };
#ifdef HAVE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(first, k.n, 2048),
        [&](const tbb::blocked_range<int>& range) { multiply(range.begin(), range.end()); });
#else
    multiply(first, k.n);
#endif
}

// Host vector updates with a heterogeneous matvec. Jacobi is applied to the
// complete system, preserving cross-partition couplings and PCG symmetry.
// Refinement checks the result against the original double-precision matrix.
template<class Apply>
std::vector<double> partitioned_pcg(std::span<const double> diagonal,
    const std::vector<double>& rhs, Apply&& apply, double tolerance, int max_iters,
    int& iterations, double& residual, int stagnation_window = 0,
    double stagnation_threshold = 0.01) {
    const Eigen::Index n = static_cast<Eigen::Index>(rhs.size());
    const Eigen::Map<const Eigen::ArrayXd> diag(diagonal.data(), n);
    Eigen::VectorXd r = Eigen::Map<const Eigen::VectorXd>(rhs.data(), n);
    Eigen::VectorXd z = r.array() / diag;
    Eigen::VectorXd p = z, ap(n), x = Eigen::VectorXd::Zero(n);
    const double norm = r.norm();
    iterations = 0;
    residual = norm > 0 ? 1.0 : 0.0;
    if (norm == 0) return std::vector<double>(rhs.size(), 0.0);
    double rz = r.dot(z), progress = 0, checkpoint = 0;
    for (int iter = 0; iter < max_iters; ++iter) {
        apply(std::span<const double>(p.data(), rhs.size()), std::span<double>(ap.data(), rhs.size()));
        const double pap = p.dot(ap);
        if (!(pap > 0) || !std::isfinite(pap)) break;
        const double alpha = rz / pap;
        x.noalias() += alpha * p;
        r.noalias() -= alpha * ap;
        iterations = iter + 1;
        residual = r.norm() / norm;
        if (!std::isfinite(residual) || residual < tolerance) break;
        z = r.array() / diag;
        const double next_rz = r.dot(z);
        progress += rz * rz / pap;
        if (stagnation_window > 0 && iterations % stagnation_window == 0) {
            if (progress > 0 && progress - checkpoint < stagnation_threshold * progress) break;
            checkpoint = progress;
        }
        p = z + (next_rz / rz) * p;
        rz = next_rz;
    }
    return std::vector<double>(x.data(), x.data() + n);
}
} // namespace vibestran::detail
