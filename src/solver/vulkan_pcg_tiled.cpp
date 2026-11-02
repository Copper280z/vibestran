// When the matrix exceeds the GPU budget, retain a fixed row prefix on the
// device and compute the remaining rows on the CPU concurrently. Only vectors
// cross PCIe per iteration; no matrix tiles are repeatedly uploaded.
#ifdef HAVE_VULKAN
#include "vulkan_pcg_internal.hpp"
#include "vulkan_commands.hpp"
#include "partitioned_pcg.hpp"
#include "amg_hybrid.hpp"
#include <spdlog/spdlog.h>
#include <type_traits>

namespace vibestran {
namespace {
VulkanBuffer make_buffer(const VulkanContext& ctx, std::size_t bytes, bool host = false) {
    auto b = host ? VulkanBuffer::create_host(ctx, bytes) : VulkanBuffer::create(ctx, bytes);
    if (!b) throw SolverError(std::format("Vulkan partitioned PCG: {} allocation of {} bytes failed", host ? "host" : "device", bytes));
    return std::move(*b);
}

template<class T> class PartitionedPcg final : public VulkanPcgWorkspace, public detail::AmgSpmv {
public:
    PartitionedPcg(VulkanContext& ctx, Pipelines& pl, const SparseMatrixBuilder::CsrData& k, std::size_t budget)
        : ctx_(ctx), pl_(pl), k_(k), rows_(detail::pcg_prefix_rows(k, budget, sizeof(T), ctx.device_info().max_storage_buffer_range)),
          values_(k.values.begin(), k.values.end()), cpu_input_(k.n), diagonal_(k.n), commands_(ctx) {
        for (int row = 0; row < k.n; ++row)
            for (int j = k.row_ptr[row]; j < k.row_ptr[row + 1]; ++j)
                if (k.col_ind[j] == row) diagonal_[row] = double(values_[j]);
        if (rows_) {
            const auto nnz = std::size_t(k.row_ptr[rows_]);
            row_ptr_ = make_buffer(ctx, (std::size_t(rows_) + 1) * sizeof(int));
            col_ind_ = make_buffer(ctx, nnz * sizeof(int));
            matrix_ = make_buffer(ctx, nnz * sizeof(T));
            p_ = make_buffer(ctx, std::size_t(k.n) * sizeof(T));
            ap_ = make_buffer(ctx, std::size_t(rows_) * sizeof(T));
            input_ = make_buffer(ctx, p_.size_bytes(), true);
            output_ = make_buffer(ctx, ap_.size_bytes(), true);
            row_ptr_.upload<int>(ctx, std::span<const int>(k.row_ptr.data(), rows_ + 1));
            col_ind_.upload<int>(ctx, std::span<const int>(k.col_ind.data(), nnz));
            matrix_.upload<T>(ctx, std::span<const T>(values_.data(), nnz));
            ds_ = pl.alloc_set(std::is_same_v<T, double> ? PL_SPMV_D : PL_SPMV);
            uint32_t binding = 0;
            for (const auto* b : {&row_ptr_, &col_ind_, &matrix_, &p_, &ap_})
                Pipelines::bind_buffer(ctx.device(), ds_, binding++, b->handle(), b->size_bytes());
        }
        spdlog::debug("[vulkan] partitioned PCG: {} GPU rows, {} CPU rows, {} matrix bytes uploaded once, {} vector bytes/SpMV",
            rows_, k.n - rows_, std::size_t(k.row_ptr[rows_]) * (sizeof(int) + sizeof(T)),
            rows_ ? (std::size_t(k.n) + rows_) * sizeof(T) : 0);
    }
    ~PartitionedPcg() override {
        commands_.drain();
        if (ds_) vkFreeDescriptorSets(ctx_.device(), pl_.pool, 1, &ds_);
    }
    std::vector<double> solve(const std::vector<double>& rhs, const VulkanSolverConfig& cfg,
                             int& iterations, double& residual) override {
        return detail::partitioned_pcg(diagonal_, rhs,
            [&](std::span<const double> p, std::span<double> ap) { multiply(p, ap); },
            cfg.tolerance, cfg.max_iterations, iterations, residual,
            cfg.stagnation_window, cfg.stagnation_threshold);
    }
    void multiply(std::span<const double> p, std::span<double> ap) override {
        auto* input = cpu_input_.data();
        std::transform(p.begin(), p.end(), input, [](double x) { return T(x); });
        if (rows_) {
            std::memcpy(input_.mapped_data(), input, p_.size_bytes());
            auto cmd = commands_.begin();
            VulkanCommands::copy(cmd, input_, p_, p_.size_bytes());
            VulkanCommands::barrier(cmd);
            const auto max_rows = ctx_.device_info().max_workgroup_count_x;
            for (uint32_t row = 0; row < uint32_t(rows_); row += max_rows) {
                const auto count = std::min(max_rows, uint32_t(rows_) - row);
                if constexpr (std::is_same_v<T, double>) cmd_spmv_d(cmd, pl_, ds_, {row, count, uint32_t(k_.n)}, count);
                else cmd_spmv(cmd, pl_, ds_, {row, count, uint32_t(k_.n)}, count);
            }
            VulkanCommands::barrier(cmd);
            VulkanCommands::copy(cmd, ap_, output_, ap_.size_bytes());
            commands_.submit();
        }
        detail::pcg_cpu_rows<T>(k_, values_, std::span<const T>(input, k_.n), ap, rows_);
        if (rows_) {
            commands_.wait();
            const auto* output = static_cast<const T*>(output_.mapped_data());
            std::copy(output, output + rows_, ap.begin());
        }
    }
private:
    VulkanContext& ctx_;
    Pipelines& pl_;
    const SparseMatrixBuilder::CsrData& k_;
    int rows_;
    std::vector<T> values_, cpu_input_;
    std::vector<double> diagonal_;
    VulkanBuffer row_ptr_, col_ind_, matrix_, p_, ap_, input_, output_;
    VkDescriptorSet ds_{VK_NULL_HANDLE};
    VulkanCommands commands_;
};
} // namespace

std::unique_ptr<VulkanPcgWorkspace> make_partitioned_pcg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& K, bool use_double, std::size_t budget) {
    if (use_double) return std::make_unique<PartitionedPcg<double>>(ctx, pl, K, budget);
    return std::make_unique<PartitionedPcg<float>>(ctx, pl, K, budget);
}

std::vector<double> solve_vulkan_hybrid_amg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& k, const std::vector<double>& rhs,
    std::size_t budget, double tolerance, int max_iterations, int& iterations, RefinementReport& report) {
    auto result = detail::solve_hybrid_amg(k, rhs, tolerance, max_iterations,
        [&](const auto& matrix) { return std::make_unique<PartitionedPcg<double>>(ctx, pl, matrix, budget); });
    iterations = result.iterations;
    report = result.refinement;
    return std::move(result.solution);
}
} // namespace vibestran
#endif
