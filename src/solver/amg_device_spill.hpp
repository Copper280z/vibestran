#pragma once
#include "amg_pcg.hpp"
#include <array>
#include <optional>
#include <span>
#ifdef HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

namespace vibestran::detail {
using AmgHostCsr = amgcl::backend::builtin<double, int, int>::matrix;

inline std::size_t amg_csr_bytes(const AmgHostCsr& a, int rows) {
    return rows ? std::size_t(a.ptr[rows]) * 12 + (std::size_t(rows) + 1) * 4 : 0;
}

// Krylov vectors and the coarse hierarchy stay on the device. Only the fine
// operator's CPU suffix needs a gathered input and an output upload. All
// couplings, including nonlocal edges back into resident rows, are preserved.
struct AmgDevicePartition {
    int gpu_rows{0};
    std::size_t device_bytes{0};
    std::vector<int> input_columns, cpu_columns;
    // The budgets that keep 50%, 75% and 90% of the fine rows resident.
    std::array<std::size_t, 3> row_fraction_budgets{};

    static std::optional<AmgDevicePartition> plan(const AmgHostCsr& a,
        std::size_t budget, std::size_t fixed_bytes,
        std::size_t buffer_limit = std::numeric_limits<std::size_t>::max()) {
        if (a.nrows > std::size_t(std::numeric_limits<int>::max()) ||
            a.ncols > std::size_t(std::numeric_limits<int>::max()) ||
            a.nnz > std::size_t(std::numeric_limits<int>::max())) return {};
        const int rows = static_cast<int>(a.nrows);
        std::vector<int> references(a.ncols, 0);
        for (std::size_t j = 0; j < a.nnz; ++j) ++references[a.col[j]];
        std::size_t inputs = std::count_if(references.begin(), references.end(), [](int n) { return n != 0; });
        int best = -1;
        std::size_t best_bytes = 0;
        std::array<std::size_t, 3> fractions{};
        for (int row = 0; row <= rows; ++row) {
            const std::size_t tail = a.nrows - row;
            const std::size_t bytes = fixed_bytes + amg_csr_bytes(a, row) + inputs * 12 + tail * 8;
            if (row == rows * 50 / 100) fractions[0] = bytes;
            if (row == rows * 75 / 100) fractions[1] = bytes;
            if (row == rows * 90 / 100) fractions[2] = bytes;
            if (bytes <= budget && std::size_t(a.ptr[row]) * 8 <= buffer_limit &&
                (std::size_t(row) + 1) * 4 <= buffer_limit && inputs * 8 <= buffer_limit && tail * 8 <= buffer_limit) {
                best = row; best_bytes = bytes;
            }
            if (row < rows)
                for (int j = a.ptr[row]; j < a.ptr[row + 1]; ++j)
                    if (--references[a.col[j]] == 0) --inputs;
        }
        if (best < 0) return {};
        AmgDevicePartition out;
        out.gpu_rows = best; out.device_bytes = best_bytes; out.row_fraction_budgets = fractions;
        std::fill(references.begin(), references.end(), -1);
        for (std::size_t j = a.ptr[best]; j < a.nnz; ++j) references[a.col[j]] = 0;
        for (std::size_t col = 0; col < a.ncols; ++col)
            if (references[col] == 0) {
                references[col] = static_cast<int>(out.input_columns.size());
                out.input_columns.push_back(static_cast<int>(col));
            }
        out.cpu_columns.reserve(a.nnz - a.ptr[best]);
        for (std::size_t j = a.ptr[best]; j < a.nnz; ++j) out.cpu_columns.push_back(references[a.col[j]]);
        return out;
    }

    static std::size_t budget_for_rows(const AmgHostCsr& a, int rows, std::size_t fixed_bytes) {
        std::vector<bool> used(a.ncols, false);
        for (std::size_t j = a.ptr[rows]; j < a.nnz; ++j) used[a.col[j]] = true;
        const auto inputs = std::count(used.begin(), used.end(), true);
        return fixed_bytes + amg_csr_bytes(a, rows) + std::size_t(inputs) * 12 + std::size_t(a.nrows - rows) * 8;
    }

    void multiply_cpu(const AmgHostCsr& a, std::span<const double> input, std::span<double> output) const {
        const int offset = a.ptr[gpu_rows];
        const auto multiply = [&](int begin, int end) {
            for (int row = begin; row < end; ++row) {
                double sum = 0;
                for (int j = a.ptr[row]; j < a.ptr[row + 1]; ++j)
                    sum += a.val[j] * input[cpu_columns[j - offset]];
                output[row - gpu_rows] = sum;
            }
        };
#ifdef HAVE_TBB
        tbb::parallel_for(tbb::blocked_range<int>(gpu_rows, static_cast<int>(a.nrows), 2048),
            [&](const tbb::blocked_range<int>& range) { multiply(range.begin(), range.end()); });
#else
        multiply(gpu_rows, static_cast<int>(a.nrows));
#endif
    }
};

inline void log_amg_device_plan(const char* backend, const AmgHostCsr& a,
    const AmgDevicePartition& p, std::size_t fixed_bytes, std::size_t budget) {
    log_debug(std::format("[{} device AMG] {} GPU rows, {} CPU rows, {} gathered inputs, {} vector bytes/SpMV; fixed={} planned={} budget={} bytes",
        backend, p.gpu_rows, a.nrows - p.gpu_rows, p.input_columns.size(),
        8 * (p.input_columns.size() + a.nrows - p.gpu_rows), fixed_bytes, p.device_bytes, budget));
    log_debug(std::format("[{} device AMG] row budgets: 50%={} 75%={} 90%={} bytes", backend,
        p.row_fraction_budgets[0], p.row_fraction_budgets[1], p.row_fraction_budgets[2]));
}
} // namespace vibestran::detail
