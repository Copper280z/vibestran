// Resident PCG: two submissions/readbacks per iteration. Matrix, descriptors,
// coherent readback storage and command buffers are reused across refinement.
#ifdef HAVE_VULKAN
#include "vulkan_pcg_internal.hpp"
#include "vulkan_commands.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <type_traits>

namespace vibestran {
namespace {
VulkanBuffer buffer(const VulkanContext& ctx, std::size_t bytes, bool host = false) {
    auto b = host ? VulkanBuffer::create_host(ctx, bytes) : VulkanBuffer::create(ctx, bytes);
    if (!b) throw SolverError(std::format("Vulkan PCG: allocation of {} bytes ({}) failed", bytes, host ? "host" : "device"));
    return std::move(*b);
}

template<class T> class FullPcg final : public VulkanPcgWorkspace {
public:
    FullPcg(VulkanContext& ctx, Pipelines& pl, const SparseMatrixBuilder::CsrData& k)
        : ctx_(ctx), pl_(pl), n_(k.n), groups_((n_ + 255) / 256),
          rows_(buffer(ctx, (std::size_t(n_) + 1) * sizeof(int))),
          cols_(buffer(ctx, std::size_t(k.nnz) * sizeof(int))),
          values_(buffer(ctx, std::size_t(k.nnz) * sizeof(T))),
          diag_(buffer(ctx, std::size_t(n_) * sizeof(T))),
          x_(buffer(ctx, std::size_t(n_) * sizeof(T))),
          r_(buffer(ctx, std::size_t(n_) * sizeof(T))),
          z_(buffer(ctx, std::size_t(n_) * sizeof(T))),
          p_(buffer(ctx, std::size_t(n_) * sizeof(T))),
          ap_(buffer(ctx, std::size_t(n_) * sizeof(T))),
          partial_(buffer(ctx, std::size_t(groups_) * sizeof(T))),
          partial2_(buffer(ctx, std::size_t(groups_) * sizeof(T))),
          readback_(buffer(ctx, std::size_t(groups_) * 2 * sizeof(T), true)), commands_(ctx) {
        std::vector<T> values(k.values.begin(), k.values.end()), diag(n_);
        for (int row = 0; row < k.n; ++row)
            for (int j = k.row_ptr[row]; j < k.row_ptr[row + 1]; ++j)
                if (k.col_ind[j] == row) diag[row] = T(k.values[j]);
        rows_.upload<int>(ctx_, k.row_ptr);
        cols_.upload<int>(ctx_, k.col_ind);
        values_.upload<T>(ctx_, values);
        diag_.upload<T>(ctx_, diag);
        sets_[0] = set(PL_SPMV, {&rows_, &cols_, &values_, &p_, &ap_});
        sets_[1] = set(PL_DOT_REDUCE, {&p_, &ap_, &partial_});
        sets_[2] = set(PL_AXPBY, {&p_, &x_});
        sets_[3] = set(PL_AXPBY, {&ap_, &r_});
        sets_[4] = set(PL_JACOBI, {&diag_, &r_, &z_});
        sets_[5] = set(PL_DOT_REDUCE, {&r_, &z_, &partial_});
        sets_[6] = set(PL_DOT_REDUCE, {&r_, &r_, &partial2_});
        sets_[7] = set(PL_AXPBY, {&z_, &p_});
    }
    ~FullPcg() override {
        commands_.drain();
        vkFreeDescriptorSets(ctx_.device(), pl_.pool, static_cast<uint32_t>(sets_.size()), sets_.data());
    }
    std::vector<double> solve(const std::vector<double>& rhs, const VulkanSolverConfig& cfg,
                             int& iterations, double& residual) override {
        const std::vector<T> b(rhs.begin(), rhs.end());
        r_.upload<T>(ctx_, b);
        auto cmd = commands_.begin();
        vkCmdFillBuffer(cmd, x_.handle(), 0, x_.size_bytes(), 0);
        vkCmdFillBuffer(cmd, p_.handle(), 0, p_.size_bytes(), 0);
        jacobi(cmd);
        VulkanCommands::barrier(cmd);
        dots(cmd);
        download_dots(cmd);
        double rz = sum(0), norm_b = std::sqrt(std::max(0.0, sum(1)));
        iterations = 0;
        residual = norm_b > 0 ? 1.0 : 0.0;
        if (norm_b == 0) return std::vector<double>(n_, 0.0);
        double beta = 0, progress = 0, checkpoint = 0;
        for (int iter = 0; iter < cfg.max_iterations; ++iter) {
            cmd = commands_.begin();
            axpby(cmd, sets_[7], T(1), T(beta));
            VulkanCommands::barrier(cmd);
            const auto max_rows = ctx_.device_info().max_workgroup_count_x;
            for (uint32_t row = 0; row < n_; row += max_rows) {
                const uint32_t count = std::min(max_rows, n_ - row);
                if constexpr (std::is_same_v<T, double>) cmd_spmv_d(cmd, pl_, sets_[0], {row, count, n_}, count);
                else cmd_spmv(cmd, pl_, sets_[0], {row, count, n_}, count);
            }
            VulkanCommands::barrier(cmd);
            dot(cmd, sets_[1]);
            VulkanCommands::barrier(cmd);
            VulkanCommands::copy(cmd, partial_, readback_, partial_.size_bytes());
            commands_.submit(); commands_.wait();
            const double pap = sum(0);
            if (!(pap > 0) || !std::isfinite(pap)) {
                if (iter > 0 && !cfg.throw_on_no_convergence) break;
                throw SolverError(std::format("Vulkan PCG: non-positive/nonfinite p^T A p={} at iteration {} (n={})", pap, iter, n_));
            }
            const T alpha = T(rz / pap);
            cmd = commands_.begin();
            axpby(cmd, sets_[2], alpha, T(1));
            axpby(cmd, sets_[3], -alpha, T(1));
            VulkanCommands::barrier(cmd);
            jacobi(cmd);
            VulkanCommands::barrier(cmd);
            dots(cmd);
            download_dots(cmd);
            const double next_rz = sum(0);
            residual = std::sqrt(std::abs(sum(1))) / norm_b;
            iterations = iter + 1;
            if (!std::isfinite(residual)) break;
            if (residual < cfg.tolerance) break;
            progress += rz * rz / pap;
            if (cfg.stagnation_window > 0 && iterations % cfg.stagnation_window == 0) {
                if (progress > 0 && progress - checkpoint < cfg.stagnation_threshold * progress) break;
                checkpoint = progress;
            }
            beta = next_rz / rz;
            rz = next_rz;
        }
        if (cfg.throw_on_no_convergence && !(residual < cfg.tolerance))
            throw SolverError(std::format("Vulkan PCG: failed to converge (n={}, iterations={}, residual={}, tolerance={})",
                n_, iterations, residual, cfg.tolerance));
        const auto x = x_.download<T>(ctx_, n_);
        return std::vector<double>(x.begin(), x.end());
    }
private:
    static constexpr int offset = std::is_same_v<T, double> ? 4 : 0;
    VkDescriptorSet set(int id, std::initializer_list<const VulkanBuffer*> buffers) {
        const auto ds = pl_.alloc_set(id + offset);
        uint32_t i = 0;
        for (const auto* b : buffers) Pipelines::bind_buffer(ctx_.device(), ds, i++, b->handle(), b->size_bytes());
        return ds;
    }
    void dot(VkCommandBuffer cmd, VkDescriptorSet ds) {
        if constexpr (std::is_same_v<T, double>) cmd_dot_reduce_d(cmd, pl_, ds, n_);
        else cmd_dot_reduce(cmd, pl_, ds, n_);
    }
    void dots(VkCommandBuffer cmd) { dot(cmd, sets_[5]); dot(cmd, sets_[6]); }
    void jacobi(VkCommandBuffer cmd) {
        if constexpr (std::is_same_v<T, double>) cmd_jacobi_d(cmd, pl_, sets_[4], n_);
        else cmd_jacobi(cmd, pl_, sets_[4], n_);
    }
    void axpby(VkCommandBuffer cmd, VkDescriptorSet ds, T a, T b) {
        if constexpr (std::is_same_v<T, double>) cmd_axpby_d(cmd, pl_, ds, {a, b, n_, 0}, n_);
        else cmd_axpby(cmd, pl_, ds, {a, b, n_}, n_);
    }
    void download_dots(VkCommandBuffer cmd) {
        VulkanCommands::barrier(cmd);
        VulkanCommands::copy(cmd, partial_, readback_, partial_.size_bytes());
        VulkanCommands::copy(cmd, partial2_, readback_, partial2_.size_bytes(), 0, partial_.size_bytes());
        commands_.submit(); commands_.wait();
    }
    double sum(int slot) const {
        const auto* ptr = static_cast<const T*>(readback_.mapped_data()) + slot * groups_;
        return std::accumulate(ptr, ptr + groups_, 0.0);
    }
    VulkanContext& ctx_;
    Pipelines& pl_;
    uint32_t n_, groups_;
    VulkanBuffer rows_, cols_, values_, diag_, x_, r_, z_, p_, ap_, partial_, partial2_, readback_;
    std::array<VkDescriptorSet, 8> sets_{};
    VulkanCommands commands_;
};
} // namespace

std::unique_ptr<VulkanPcgWorkspace> make_full_pcg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& K, bool use_double) {
    if (use_double) return std::make_unique<FullPcg<double>>(ctx, pl, K);
    return std::make_unique<FullPcg<float>>(ctx, pl, K);
}
} // namespace vibestran
#endif
