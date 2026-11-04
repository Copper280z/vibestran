// Vulkan AMG PCG: the counterpart of cuda_amg.hpp. Every CG scalar stays on
// the device, so an iteration is recorded once into command buffers and
// replayed; the host only polls convergence. Products apply the cycle's
// vector updates in the same dispatch. With a device budget the fine
// operator keeps a GPU row prefix; each fine product then ends a recorded
// segment, the CPU computes the suffix rows from the gathered inputs while
// the prefix product runs, and the next segment applies the CPU rows.
#ifdef HAVE_VULKAN
#include "amg_factor.hpp"
#include "amg_device_spill.hpp"
#include "vulkan_pcg_internal.hpp"
#include "vulkan_commands.hpp"
#include "amg_spmv_d_spv.h"
#include "amg_ops_d_spv.h"
#include "amg_cg_d_spv.h"
#include "amg_scalar_d_spv.h"
#include "amg_dense_d_spv.h"
#include <algorithm>
#include <map>

namespace vibestran::detail {
namespace {
// std430 layout of amg_state.glsl.
struct CgState {
    double rho, alpha, beta, norm_b, target, rr;
    std::int32_t flags, iterations, max_iterations, pad;
};
static_assert(sizeof(CgState) == 64);

constexpr std::uint32_t kGroups = 256;   // fixed reduction grid (amg_cg_d.comp)
constexpr int kPollIterations = 8;
enum Mode : std::uint32_t { kSet, kAdd, kResidual, kSmooth };
enum Kernel { kSpmv4, kSpmv8, kSpmv16, kSpmv32, kOps, kCg, kScalar, kDense, kKernels };

struct PcSpmv { std::uint32_t row_start, rows, mode; };
struct PcOps { std::uint32_t n, op, offset, mode; };
struct PcCg { std::uint32_t n, op; };
struct PcScalar { std::uint32_t op; };
struct PcDense { std::uint32_t n; };

void check(VkResult status, const char* action) {
    if (status != VK_SUCCESS)
        throw SolverError(std::format("Vulkan AMG: {} failed (VkResult={})", action, static_cast<int>(status)));
}

// Subgroup shuffles reduce a row without shared memory or barriers. They
// need consecutive invocations in one subgroup, which these vendors use for
// one-dimensional workgroups; other devices take the shared-memory reduction.
std::uint32_t shuffle_lanes(const VulkanContext& ctx) {
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(ctx.physical_device(), &props);
    const auto vendor = props.properties.vendorID;
    const bool linear = vendor == 0x10DE || vendor == 0x1002;
    const bool shuffle = (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT);
    return linear && shuffle ? subgroup.subgroupSize : 0;
}

class AmgPipelines {
public:
    explicit AmgPipelines(const VulkanContext& ctx) : device_(ctx.device()) {
        struct Spec { const unsigned char* code; unsigned len; std::uint32_t bindings, push; };
        const Spec specs[kKernels] = {
            {amg_spmv_d_spv, amg_spmv_d_spv_len, 8, sizeof(PcSpmv)},
            {amg_spmv_d_spv, amg_spmv_d_spv_len, 8, sizeof(PcSpmv)},
            {amg_spmv_d_spv, amg_spmv_d_spv_len, 8, sizeof(PcSpmv)},
            {amg_spmv_d_spv, amg_spmv_d_spv_len, 8, sizeof(PcSpmv)},
            {amg_ops_d_spv, amg_ops_d_spv_len, 6, sizeof(PcOps)},
            {amg_cg_d_spv, amg_cg_d_spv_len, 6, sizeof(PcCg)},
            {amg_scalar_d_spv, amg_scalar_d_spv_len, 2, sizeof(PcScalar)},
            {amg_dense_d_spv, amg_dense_d_spv_len, 4, sizeof(PcDense)},
        };
        const auto subgroup = shuffle_lanes(ctx);
        for (int k = 0; k < kKernels; ++k) {
            const auto& s = specs[k];
            std::vector<std::uint32_t> words((s.len + 3) / 4);
            std::memcpy(words.data(), s.code, s.len);
            VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            mi.codeSize = s.len; mi.pCode = words.data();
            check(vkCreateShaderModule(device_, &mi, nullptr, &modules_[k]), "create shader module");
            std::vector<VkDescriptorSetLayoutBinding> b(s.bindings);
            for (std::uint32_t i = 0; i < s.bindings; ++i)
                b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            di.bindingCount = s.bindings; di.pBindings = b.data();
            check(vkCreateDescriptorSetLayout(device_, &di, nullptr, &dsl_[k]), "create descriptor layout");
            const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, s.push};
            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            li.setLayoutCount = 1; li.pSetLayouts = &dsl_[k];
            li.pushConstantRangeCount = 1; li.pPushConstantRanges = &range;
            check(vkCreatePipelineLayout(device_, &li, nullptr, &layout_[k]), "create pipeline layout");
            // Sparse products specialize LANES and SUBGROUP.
            const std::uint32_t lanes = 4u << k;
            const std::array<std::uint32_t, 2> constants{lanes, subgroup >= lanes ? 1u : 0u};
            const std::array<VkSpecializationMapEntry, 2> entries{{{0, 0, 4}, {1, 4, 4}}};
            const VkSpecializationInfo si{2, entries.data(), sizeof(constants), constants.data()};
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.layout = layout_[k];
            ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                VK_SHADER_STAGE_COMPUTE_BIT, modules_[k], "main", k <= kSpmv32 ? &si : nullptr};
            check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline_[k]), "create pipeline");
        }
    }
    ~AmgPipelines() {
        for (int k = 0; k < kKernels; ++k) {
            if (pipeline_[k]) vkDestroyPipeline(device_, pipeline_[k], nullptr);
            if (layout_[k]) vkDestroyPipelineLayout(device_, layout_[k], nullptr);
            if (dsl_[k]) vkDestroyDescriptorSetLayout(device_, dsl_[k], nullptr);
            if (modules_[k]) vkDestroyShaderModule(device_, modules_[k], nullptr);
        }
    }
    AmgPipelines(const AmgPipelines&) = delete;
    AmgPipelines& operator=(const AmgPipelines&) = delete;

    VkDescriptorSetLayout layout(int k) const { return dsl_[k]; }

    template<class Push>
    void dispatch(VkCommandBuffer cmd, int k, VkDescriptorSet ds, const Push& push, std::uint32_t groups) const {
        VulkanCommands::barrier(cmd);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_[k]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_[k], 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, layout_[k], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        vkCmdDispatch(cmd, groups, 1, 1);
    }

private:
    VkDevice device_;
    std::array<VkShaderModule, kKernels> modules_{};
    std::array<VkDescriptorSetLayout, kKernels> dsl_{};
    std::array<VkPipelineLayout, kKernels> layout_{};
    std::array<VkPipeline, kKernels> pipeline_{};
};

// Descriptor sets for one solver's buffers, freed with it.
class DescriptorCache {
public:
    DescriptorCache(VkDevice device, const AmgPipelines& kernels) : device_(device), kernels_(kernels) {
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 1024; pi.poolSizeCount = 1; pi.pPoolSizes = &size;
        check(vkCreateDescriptorPool(device_, &pi, nullptr, &pool_), "create descriptor pool");
    }
    ~DescriptorCache() { vkDestroyDescriptorPool(device_, pool_, nullptr); }
    DescriptorCache(const DescriptorCache&) = delete;
    DescriptorCache& operator=(const DescriptorCache&) = delete;

    VkDescriptorSet get(int k, std::initializer_list<const VulkanBuffer*> buffers) {
        std::vector<const void*> key{reinterpret_cast<const void*>(std::uintptr_t(k))};
        key.insert(key.end(), buffers.begin(), buffers.end());
        if (const auto it = sets_.find(key); it != sets_.end()) return it->second;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        const auto layout = kernels_.layout(k);
        ai.descriptorPool = pool_; ai.descriptorSetCount = 1; ai.pSetLayouts = &layout;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        check(vkAllocateDescriptorSets(device_, &ai, &ds), "allocate descriptor set");
        std::uint32_t binding = 0;
        for (const auto* b : buffers) Pipelines::bind_buffer(device_, ds, binding++, b->handle(), b->size_bytes());
        sets_.emplace(std::move(key), ds);
        return ds;
    }

private:
    VkDevice device_;
    const AmgPipelines& kernels_;
    VkDescriptorPool pool_{VK_NULL_HANDLE};
    std::map<std::vector<const void*>, VkDescriptorSet> sets_;
};

std::uint32_t div_up(std::size_t n, std::size_t d) { return static_cast<std::uint32_t>((n + d - 1) / d); }

struct DeviceCsr {
    int rows{0};
    std::uint32_t lanes{8};
    VulkanBuffer rp, ci, val;
};

class Solver {
public:
    // Device bytes other than the fine operator, for the budget planner.
    static std::size_t fixed_bytes(const AmgHierarchy& h) {
        const auto n = h.levels.front().A->nrows;
        std::size_t bytes = 5 * n * 8 + kGroups * 8 + sizeof(CgState) + 65536;
        for (std::size_t l = 0; l < h.levels.size(); ++l) {
            const auto& level = h.levels[l];
            bytes += (l == 0 ? 1 : 3) * level.A->nrows * 8 + level.smoother.size() * 8;
            if (l) bytes += amg_csr_bytes(*level.A, int(level.A->nrows));
            if (level.P) bytes += amg_csr_bytes(*level.P, int(level.P->nrows)) + amg_csr_bytes(*level.R, int(level.R->nrows));
        }
        return bytes + h.coarse_inverse.size() * 8;
    }
    // Every buffer must fit one storage-buffer descriptor.
    static bool bindable(const AmgHierarchy& h, std::size_t limit) {
        const auto fits = [limit](const AmgHostCsr* a) {
            return !a || (a->nnz * 8 <= limit && (a->nrows + 1) * 8 <= limit);
        };
        return std::ranges::all_of(h.levels, [&](const AmgLevel& level) {
                   return fits(level.A.get()) && fits(level.P.get()) && fits(level.R.get());
               }) && h.coarse_inverse.size() * 8 <= limit;
    }

    Solver(VulkanContext& ctx, Pipelines& pl, const AmgHierarchy& h, std::optional<AmgDevicePartition> partition)
        : ctx_(ctx), pl_(pl), kernels_(kernels(ctx, pl)), sets_(ctx.device(), kernels_), h_(h), n_(int(h.levels.front().A->nrows)), partition_(std::move(partition)),
          r_(buffer(std::size_t(n_) * 8)), u_(buffer(std::size_t(n_) * 8)), p_(buffer(std::size_t(n_) * 8)),
          q_(buffer(std::size_t(n_) * 8)), z_(buffer(std::size_t(n_) * 8)),
          partial_(buffer(kGroups * 8)), state_(buffer(sizeof(CgState))),
          state_host_(host(sizeof(CgState))), staging_(host(std::size_t(n_) * 8)) {
        for (std::size_t l = 0; l < h.levels.size(); ++l) {
            const auto& level = h.levels[l];
            const auto count = level.A->nrows;
            Level d;
            if (l) { d.f = buffer(count * 8); d.y = buffer(count * 8); }
            d.x = buffer(count * 8);
            d.m = buffer(std::max<std::size_t>(level.smoother.size(), 1) * 8);
            if (!level.smoother.empty()) upload(d.m, level.smoother);
            d.a = csr(*level.A, l == 0 && partition_ ? partition_->gpu_rows : int(count));
            if (level.P) { d.p = csr(*level.P, int(level.P->nrows)); d.r = csr(*level.R, int(level.R->nrows)); }
            levels_.push_back(std::move(d));
        }
        coarse_ = buffer(std::max<std::size_t>(h.coarse_inverse.size(), 1) * 8);
        if (h.coarse_direct()) upload(coarse_, h.coarse_inverse);
        if (partitioned()) {
            const auto count = partition_->input_columns.size();
            const auto tail = std::size_t(n_ - partition_->gpu_rows);
            columns_ = buffer(std::max<std::size_t>(count, 1) * 4);
            if (count) columns_.upload<int>(ctx_, partition_->input_columns);
            gathered_ = buffer(std::max<std::size_t>(count, 1) * 8);
            tail_ = buffer(tail * 8);
            input_ = host(std::max<std::size_t>(count, 1) * 8);
            output_ = host(tail * 8);
        }
        record();
    }
    ~Solver() {
        vkDeviceWaitIdle(ctx_.device());
        for (auto& s : segments_) release(s);
        release(start_); release(finish_);
        for (const auto& [handle, ds] : gather_sets_) vkFreeDescriptorSets(ctx_.device(), pl_.pool, 1, &ds);
    }
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    bool fully_resident() const { return !partitioned(); }

    // Same patterns, new values (AmgHierarchy::rebuild).
    void update_values() {
        for (std::size_t l = 0; l < h_.levels.size(); ++l) {
            const auto& level = h_.levels[l];
            auto& d = levels_[l];
            if (!level.smoother.empty()) upload(d.m, level.smoother);
            if (d.a.rows) d.a.val.upload<double>(ctx_, {level.A->val, std::size_t(level.A->ptr[d.a.rows])});
        }
        if (h_.coarse_direct()) upload(coarse_, h_.coarse_inverse);
    }

    std::pair<int, double> solve(const std::vector<double>& b, double tolerance, int max_iterations,
                                 std::vector<double>& x) {
        std::memcpy(staging_.mapped_data(), b.data(), std::size_t(n_) * 8);
        CgState initial{};
        initial.target = tolerance;
        initial.max_iterations = max_iterations;
        std::memcpy(state_host_.mapped_data(), &initial, sizeof(initial));
        run(start_);
        const auto& state = *static_cast<const CgState*>(state_host_.mapped_data());
        while (!state.flags) {
            for (std::size_t i = 0; i < segments_.size(); ++i) {
                submit(segments_[i]);
                if (!segments_[i].gather) continue;
                // The prefix product runs while the CPU computes the suffix.
                submit(segments_[++i]);
                wait(segments_[i - 1]);
                partition_->multiply_cpu(*h_.levels.front().A,
                    {static_cast<const double*>(input_.mapped_data()), partition_->input_columns.size()},
                    {static_cast<double*>(output_.mapped_data()), std::size_t(n_ - partition_->gpu_rows)});
            }
            wait(segments_.back());
        }
        run(finish_);
        std::memcpy(x.data(), staging_.mapped_data(), std::size_t(n_) * 8);
        return {state.iterations, state.norm_b > 0 ? std::sqrt(state.rr) / state.norm_b : 0.0};
    }

    // V-cycle operations (amg_vcycle), recorded into the current segment.
    void smooth_zero(std::size_t l) { ops(0, rows(l), 0, kSet, f(l), levels_[l].x, f(l), levels_[l].m, f(l)); }
    void residual(std::size_t l) { product(l, kResidual, levels_[l].x, y(l), f(l), f(l)); }
    void restriction(std::size_t l) { spmv(levels_[l].r, kSet, y(l), f(l + 1), y(l), y(l)); }
    void coarse(std::size_t l) {
        if (!h_.coarse_direct()) { smooth_zero(l); smooth_post(l); return; }
        const auto n = std::uint32_t(rows(l));
        kernels_.dispatch(cmd_, kDense, sets_.get(kDense, {&coarse_, &f(l), &y(l), &state_}), PcDense{n}, n);
    }
    void prolongation(std::size_t l) { spmv(levels_[l].p, kAdd, y(l + 1), levels_[l].x, y(l + 1), y(l + 1)); }
    void smooth_post(std::size_t l) { product(l, kSmooth, levels_[l].x, y(l), f(l), levels_[l].m); }

private:
    struct Level { VulkanBuffer f, x, y, m; DeviceCsr a, p, r; };
    struct Segment { VkCommandBuffer cmd{VK_NULL_HANDLE}; VkFence fence{VK_NULL_HANDLE}; bool gather{false}; };

    bool partitioned() const { return partition_ && partition_->gpu_rows < n_; }
    std::size_t rows(std::size_t l) const { return h_.levels[l].A->nrows; }
    VulkanBuffer& f(std::size_t l) { return l ? levels_[l].f : r_; }
    VulkanBuffer& y(std::size_t l) { return l ? levels_[l].y : z_; }

    VulkanBuffer buffer(std::size_t bytes) {
        auto b = VulkanBuffer::create(ctx_, std::max<std::size_t>(bytes, 8));
        if (!b) throw SolverError(std::format("Vulkan AMG: device allocation of {} bytes failed (n={})", bytes, n_));
        return std::move(*b);
    }
    VulkanBuffer host(std::size_t bytes) {
        auto b = VulkanBuffer::create_host(ctx_, std::max<std::size_t>(bytes, 8));
        if (!b) throw SolverError(std::format("Vulkan AMG: host-visible allocation of {} bytes failed (n={})", bytes, n_));
        return std::move(*b);
    }
    void upload(VulkanBuffer& b, std::span<const double> values) { b.upload<double>(ctx_, values); }
    DeviceCsr csr(const AmgHostCsr& a, int prefix) {
        DeviceCsr d;
        d.rows = prefix;
        const double average = a.nrows ? double(a.nnz) / double(a.nrows) : 0.0;
        d.lanes = average <= 6 ? 4 : average <= 12 ? 8 : average <= 24 ? 16 : 32;
        const auto nnz = std::size_t(a.ptr[prefix]);
        d.rp = buffer((std::size_t(prefix) + 1) * 4); d.ci = buffer(nnz * 4); d.val = buffer(nnz * 8);
        if (prefix) {
            d.rp.upload<int>(ctx_, {a.ptr, std::size_t(prefix) + 1});
            d.ci.upload<int>(ctx_, {a.col, nnz});
            d.val.upload<double>(ctx_, {a.val, nnz});
        }
        return d;
    }

    void spmv(const DeviceCsr& a, Mode mode, const VulkanBuffer& x, VulkanBuffer& out,
              const VulkanBuffer& rhs, const VulkanBuffer& m) {
        const int kernel = a.lanes == 4 ? kSpmv4 : a.lanes == 8 ? kSpmv8 : a.lanes == 16 ? kSpmv16 : kSpmv32;
        const std::size_t per_group = 256 / a.lanes;
        const std::size_t chunk = std::size_t(ctx_.device_info().max_workgroup_count_x) * per_group;
        for (std::size_t start = 0; start < std::size_t(a.rows); start += chunk) {
            const auto count = std::min(chunk, std::size_t(a.rows) - start);
            kernels_.dispatch(cmd_, kernel, sets_.get(kernel, {&a.rp, &a.ci, &a.val, &x, &out, &rhs, &m, &state_}),
                PcSpmv{std::uint32_t(start), std::uint32_t(count), mode}, div_up(count, per_group));
        }
    }
    void ops(std::uint32_t op, std::size_t n, std::uint32_t offset, Mode mode, const VulkanBuffer& x, VulkanBuffer& out,
             const VulkanBuffer& rhs, const VulkanBuffer& m, const VulkanBuffer& product) {
        kernels_.dispatch(cmd_, kOps, sets_.get(kOps, {&x, &out, &rhs, &m, &product, &state_}),
            PcOps{std::uint32_t(n), op, offset, mode}, div_up(n, 256));
    }
    void cg(std::uint32_t op, const VulkanBuffer& a, const VulkanBuffer& b) {
        kernels_.dispatch(cmd_, kCg, sets_.get(kCg, {&a, &b, &u_, &r_, &partial_, &state_}), PcCg{std::uint32_t(n_), op}, kGroups);
    }
    void scalar(std::uint32_t op) { kernels_.dispatch(cmd_, kScalar, sets_.get(kScalar, {&partial_, &state_}), PcScalar{op}, 1); }

    // Level products; a partitioned fine product splits the current segment.
    void product(std::size_t l, Mode mode, const VulkanBuffer& x, VulkanBuffer& out,
                 const VulkanBuffer& rhs, const VulkanBuffer& m) {
        if (l || !partitioned()) { spmv(levels_[l].a, mode, x, out, rhs, m); return; }
        const auto count = partition_->input_columns.size();
        if (count) {
            const auto ds = gather_set(x);
            VulkanCommands::barrier(cmd_);
            const PCDot pc{std::uint32_t(count)};
            vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pl_.pipeline[PL_AMG_GATHER_D]);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pl_.pl_layout[PL_AMG_GATHER_D], 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd_, pl_.pl_layout[PL_AMG_GATHER_D], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd_, div_up(count, 256), 1, 1);
            VulkanCommands::barrier(cmd_);
            VulkanCommands::copy(cmd_, gathered_, input_, count * 8);
        }
        end_segment(true);
        spmv(levels_[0].a, mode, x, out, rhs, m);
        end_segment(false);
        const auto tail = std::size_t(n_ - partition_->gpu_rows);
        VulkanCommands::copy(cmd_, output_, tail_, tail * 8);
        ops(1, tail, std::uint32_t(partition_->gpu_rows), mode, x, out, rhs, m, tail_);
    }
    VkDescriptorSet gather_set(const VulkanBuffer& x) {
        if (const auto it = gather_sets_.find(&x); it != gather_sets_.end()) return it->second;
        const auto ds = pl_.alloc_set(PL_AMG_GATHER_D);
        if (!ds) throw SolverError("Vulkan AMG: gather descriptor allocation failed");
        Pipelines::bind_buffer(ctx_.device(), ds, 0, x.handle(), x.size_bytes());
        Pipelines::bind_buffer(ctx_.device(), ds, 1, columns_.handle(), columns_.size_bytes());
        Pipelines::bind_buffer(ctx_.device(), ds, 2, gathered_.handle(), gathered_.size_bytes());
        gather_sets_.emplace(&x, ds);
        return ds;
    }

    VkCommandBuffer begin() {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = ctx_.command_pool(); ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cmd), "allocate command buffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cmd, &bi), "begin command buffer");
        // Orders this buffer after earlier submissions and host writes.
        VulkanCommands::barrier(cmd);
        return cmd;
    }
    Segment finish(VkCommandBuffer cmd, bool fenced) {
        VulkanCommands::barrier(cmd);
        check(vkEndCommandBuffer(cmd), "end command buffer");
        Segment s{cmd};
        if (fenced) {
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check(vkCreateFence(ctx_.device(), &fi, nullptr, &s.fence), "create fence");
        }
        return s;
    }
    void end_segment(bool gather) {
        auto s = finish(cmd_, gather);
        s.gather = gather;
        segments_.push_back(s);
        cmd_ = begin();
    }
    void release(Segment& s) {
        if (s.fence) vkDestroyFence(ctx_.device(), s.fence, nullptr);
        if (s.cmd) vkFreeCommandBuffers(ctx_.device(), ctx_.command_pool(), 1, &s.cmd);
        s = {};
    }
    void submit(const Segment& s) {
        if (s.fence) check(vkResetFences(ctx_.device(), 1, &s.fence), "reset fence");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &s.cmd;
        check(vkQueueSubmit(ctx_.compute_queue(), 1, &si, s.fence), "submit");
    }
    void wait(const Segment& s) { check(vkWaitForFences(ctx_.device(), 1, &s.fence, VK_TRUE, UINT64_MAX), "wait"); }
    void run(const Segment& s) { submit(s); wait(s); }

    void iteration() {
        amg_vcycle(*this, h_.levels.size());
        cg(0, r_, z_);
        scalar(1);
        ops(2, std::size_t(n_), 0, kSet, z_, p_, z_, z_, z_);
        product(0, kSet, p_, q_, p_, p_);
        cg(0, p_, q_);
        scalar(2);
        cg(1, p_, q_);
        scalar(3);
    }

    void record() {
        cmd_ = begin();
        VulkanCommands::copy(cmd_, staging_, r_, std::size_t(n_) * 8);
        vkCmdFillBuffer(cmd_, u_.handle(), 0, VK_WHOLE_SIZE, 0);
        VulkanCommands::copy(cmd_, state_host_, state_, sizeof(CgState));
        cg(0, r_, r_);
        scalar(0);
        VulkanCommands::barrier(cmd_);
        VulkanCommands::copy(cmd_, state_, state_host_, sizeof(CgState));
        start_ = finish(cmd_, true);

        // A partitioned product waits for the host each iteration anyway, so
        // the host polls every iteration there.
        cmd_ = begin();
        for (int i = 0; i < (partitioned() ? 1 : kPollIterations); ++i) iteration();
        VulkanCommands::barrier(cmd_);
        VulkanCommands::copy(cmd_, state_, state_host_, sizeof(CgState));
        segments_.push_back(finish(cmd_, true));

        cmd_ = begin();
        VulkanCommands::copy(cmd_, u_, staging_, std::size_t(n_) * 8);
        finish_ = finish(cmd_, true);
        cmd_ = VK_NULL_HANDLE;
    }

    static AmgPipelines& kernels(const VulkanContext& ctx, Pipelines& pl) {
        if (!pl.amg) pl.amg = {new AmgPipelines(ctx), [](void* p) { delete static_cast<AmgPipelines*>(p); }};
        return *static_cast<AmgPipelines*>(pl.amg.get());
    }

    VulkanContext& ctx_;
    Pipelines& pl_;
    const AmgPipelines& kernels_;
    DescriptorCache sets_;
    const AmgHierarchy& h_;
    int n_;
    std::optional<AmgDevicePartition> partition_;
    VulkanBuffer r_, u_, p_, q_, z_, partial_, state_, state_host_, staging_;
    VulkanBuffer coarse_, columns_, gathered_, tail_, input_, output_;
    std::vector<Level> levels_;
    std::map<const VulkanBuffer*, VkDescriptorSet> gather_sets_;
    VkCommandBuffer cmd_{VK_NULL_HANDLE};
    std::vector<Segment> segments_;
    Segment start_, finish_;
};
} // namespace
} // namespace vibestran::detail

namespace vibestran {
namespace {
std::unique_ptr<detail::AmgSystem> make_vulkan_amg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& k, double tolerance, int max_iterations, std::function<std::size_t()> budget,
    bool& fully_resident, std::optional<AmgDeviceReport>& report, std::shared_ptr<const NearNullspace> modes = nullptr) {
    using namespace detail;
    const int max = max_iterations > 0 ? max_iterations : 10000;
    class Engine final : public AmgEngine {
    public:
        Engine(VulkanContext& c, Pipelines& p, const AmgHierarchy& h, std::optional<AmgDevicePartition> part, int m)
            : solver_(c, p, h, std::move(part)), max_(m) {}
        std::pair<int, double> solve(const std::vector<double>& b, double tol, std::vector<double>& x) override {
            return solver_.solve(b, tol, max_, x);
        }
        void values_changed() override { solver_.update_values(); }
        bool fully_resident() const { return solver_.fully_resident(); }
    private:
        Solver solver_;
        int max_;
    };
    return std::make_unique<AmgSystem>(k, tolerance,
        [&ctx, &pl, budget = std::move(budget), max, &fully_resident, &report](const ScaledAmgMatrix& scaled, const AmgHierarchy& h)
            -> std::unique_ptr<AmgEngine> {
            using Report = AmgDeviceReport;
            const auto& fine = *h.levels.front().A;
            const auto rows = int(fine.nrows);
            const auto limit = ctx.device_info().max_storage_buffer_range;
            const auto fixed = Solver::fixed_bytes(h);
            // Queried now: the previous hierarchy of a refactor is gone and setup took time.
            const auto available = budget();
            std::optional<AmgDevicePartition> partition;
            if (available >= fine.nrows * 80 && fine.nrows * 8 <= limit &&
                (fine.nrows + 255) / 256 <= ctx.device_info().max_workgroup_count_x && Solver::bindable(h, limit))
                partition = AmgDevicePartition::plan(fine, available, fixed, limit);
            fully_resident = false;
            if (!partition) {
                log_debug(std::format("[vulkan device AMG] workspace does not fit: fixed={} budget={} bytes; using host hierarchy", fixed, available));
                report = Report{Report::Execution::Host, 0, rows, 0, available, 0};
                return std::make_unique<HostAmgEngine>(scaled, h, vulkan_partitioned_spmv(ctx, pl, available), max);
            }
            log_amg_device_plan("vulkan", fine, *partition, fixed, available);
            report = Report{partition->gpu_rows == rows ? Report::Execution::Resident : Report::Execution::FineRowSpill,
                partition->gpu_rows, rows - partition->gpu_rows, partition->device_bytes, available, 0};
            auto engine = std::make_unique<Engine>(ctx, pl, h, std::move(partition), max);
            fully_resident = engine->fully_resident();
            return engine;
        }, std::move(modes));
}
} // namespace

std::vector<double> solve_vulkan_amg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& k, const std::vector<double>& rhs, double tolerance,
    int max_iterations, const std::function<std::size_t()>& budget, bool& fully_resident,
    std::optional<AmgDeviceReport>& report, const VulkanAmgAccept& accept,
    std::shared_ptr<const NearNullspace> modes) {
    auto result = make_vulkan_amg(ctx, pl, k, tolerance, max_iterations, budget, fully_resident, report, modes)->solve(k, rhs);
    if (auto e = accept(result)) throw SolverError(e->message);
    return std::move(result.solution);
}

std::unique_ptr<LinearFactor> make_vulkan_amg_factor(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& k, double tolerance, int max_iterations,
    std::function<std::size_t()> budget, bool& fully_resident,
    std::optional<AmgDeviceReport>& report, VulkanAmgAccept accept, std::shared_ptr<const NearNullspace> modes) {
    return std::make_unique<detail::AmgFactor>(k,
        [&ctx, &pl, tolerance, max_iterations, budget = std::move(budget), &fully_resident, &report, modes = std::move(modes)](const SparseMatrixBuilder::CsrData& matrix) {
            return make_vulkan_amg(ctx, pl, matrix, tolerance, max_iterations, budget, fully_resident, report, modes);
        },
        std::move(accept));
}
} // namespace vibestran
#endif
