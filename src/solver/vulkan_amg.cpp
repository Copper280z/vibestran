#ifdef HAVE_VULKAN
#include "amg_pcg.hpp"
#include "vulkan_pcg_internal.hpp"
#include "vulkan_commands.hpp"
#include "amg_coarse.hpp"
#include "amg_device_spill.hpp"
#include <map>

namespace vibestran::detail {
struct VulkanAmgMatrix;
struct VulkanAmgContext {
    const VulkanContext& ctx;
    Pipelines& pipelines;
    VulkanCommands commands;
    VkCommandBuffer recording{VK_NULL_HANDLE};
    std::size_t budget, allocated{0};
    bool declined{false};
    bool fully_resident{false};
    std::vector<VulkanAmgMatrix*> matrices;
    VulkanBuffer partial, readback;
    std::map<std::array<std::uint64_t, 6>, VkDescriptorSet> sets;

    VulkanAmgContext(const VulkanContext& c, Pipelines& p, int n, std::size_t bytes)
        : ctx(c), pipelines(p), commands(c), budget(bytes),
          partial(allocate((std::size_t(n) + 255) / 256 * sizeof(double))),
          readback(allocate(partial.size_bytes(), true)) {}
    ~VulkanAmgContext() {
        commands.drain();
        for (const auto& [key, ds] : sets) vkFreeDescriptorSets(ctx.device(), pipelines.pool, 1, &ds);
    }
    bool prepare();
    VulkanBuffer allocate(std::size_t bytes, bool host = false) {
        if (!host) {
            allocated += bytes;
            if (allocated > budget || bytes > ctx.device_info().max_storage_buffer_range) {
                declined = true;
                return {};
            }
        }
        auto b = host ? VulkanBuffer::create_host(ctx, bytes) : VulkanBuffer::create(ctx, bytes);
        if (!b) throw SolverError(std::format("Vulkan AMG: {} allocation of {} bytes failed", host ? "host" : "device", bytes));
        return std::move(*b);
    }
    VkCommandBuffer begin() {
        if (!recording) recording = commands.begin();
        VulkanCommands::barrier(recording);
        return recording;
    }
    void submit() {
        if (!recording) return;
        commands.submit(); recording = VK_NULL_HANDLE;
    }
    void flush() { submit(); commands.wait(); }
    VkDescriptorSet descriptors(int pipeline, std::initializer_list<const VulkanBuffer*> buffers) {
        std::array<std::uint64_t, 6> key{};
        key[0] = static_cast<std::uint64_t>(pipeline);
        std::size_t i = 1;
        for (const auto* b : buffers) {
            const auto handle = b->handle();
            static_assert(sizeof(handle) <= sizeof(std::uint64_t));
            std::memcpy(&key[i++], &handle, sizeof(handle));
        }
        if (const auto it = sets.find(key); it != sets.end()) return it->second;
        const auto ds = pipelines.alloc_set(pipeline);
        if (!ds) throw SolverError(std::format("Vulkan AMG: descriptor allocation failed ({} cached sets)", sets.size()));
        std::uint32_t binding = 0;
        for (const auto* b : buffers) Pipelines::bind_buffer(ctx.device(), ds, binding++, b->handle(), b->size_bytes());
        sets.emplace(key, ds);
        return ds;
    }
};

struct VulkanAmgVector {
    using value_type = double;
    VulkanAmgContext& context;
    std::size_t count;
    mutable VulkanBuffer storage;
    VulkanAmgVector(VulkanAmgContext& c, std::size_t n) : context(c), count(n), storage(c.allocate(n * sizeof(double))) {}
    std::size_t size() const { return count; }
    std::size_t bytes() const { return storage.size_bytes(); }
    void upload(std::span<const double> values) {
        if (!storage.handle()) return; // setup will decline an insufficient device budget
        context.flush(); storage.upload<double>(context.ctx, values);
    }
    void download(std::vector<double>& values) const {
        context.flush(); values = storage.download<double>(context.ctx, count);
    }
};

void amg_vector(double alpha, const VulkanAmgVector& x, double beta,
                const VulkanAmgVector& y, double gamma, VulkanAmgVector& z, unsigned operation = 0) {
    auto& c = z.context;
    const auto ds = c.descriptors(PL_AMG_VECTOR_D, {&x.storage, &y.storage, &z.storage});
    const auto cmd = c.begin();
    const auto& pl = c.pipelines;
    const PCAmgVector pc{alpha, beta, gamma, static_cast<std::uint32_t>(z.size()), operation};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline[PL_AMG_VECTOR_D]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pl_layout[PL_AMG_VECTOR_D], 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl.pl_layout[PL_AMG_VECTOR_D], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (pc.n + 255) / 256, 1, 1);
}

struct VulkanAmgMatrix {
    using value_type = double;
    using col_type = int;
    using ptr_type = int;
    VulkanAmgContext& context;
    std::size_t nr, nc, nnz;
    std::shared_ptr<AmgHostCsr> host;
    VulkanBuffer row_ptr, col_ind, values, columns, gathered, input, output;
    mutable VulkanAmgVector product;
    std::optional<AmgDevicePartition> partition;
    int gpu_rows{0};
    explicit VulkanAmgMatrix(VulkanAmgContext& c, std::shared_ptr<AmgHostCsr> a)
        : context(c), nr(a->nrows), nc(a->ncols), nnz(a->nnz), host(std::move(a)), product(c, nr) {
        if (nnz > std::size_t(std::numeric_limits<int>::max()))
            throw SolverError(std::format("Vulkan AMG: {} nonzeros exceeds the supported 32-bit CSR range", nnz));
        c.matrices.push_back(this);
    }
    void allocate() {
        auto& c = context;
        gpu_rows = partition ? partition->gpu_rows : int(nr);
        c.flush();
        if (gpu_rows) {
            const auto entries = std::size_t(host->ptr[gpu_rows]);
            row_ptr = c.allocate((std::size_t(gpu_rows) + 1) * 4);
            col_ind = c.allocate(entries * 4); values = c.allocate(entries * 8);
            row_ptr.upload<int>(c.ctx, std::span<const int>(host->ptr, gpu_rows + 1));
            col_ind.upload<int>(c.ctx, std::span<const int>(host->col, entries));
            values.upload<double>(c.ctx, std::span<const double>(host->val, entries));
        }
        if (std::size_t(gpu_rows) < nr) {
            const auto count = partition->input_columns.size();
            columns = c.allocate(count * 4); gathered = c.allocate(count * 8);
            input = c.allocate(count * 8, true); output = c.allocate((nr - gpu_rows) * 8, true);
            columns.upload<int>(c.ctx, partition->input_columns);
        }
    }
    std::size_t rows() const { return nr; }
    std::size_t cols() const { return nc; }
    std::size_t nonzeros() const { return nnz; }
    std::size_t bytes() const { return row_ptr.size_bytes() + col_ind.size_bytes() + values.size_bytes() + product.bytes(); }
    void multiply(double alpha, const VulkanAmgVector& x, double beta, VulkanAmgVector& y) const {
        auto& c = context;
        if (input.handle()) {
            const auto ds = c.descriptors(PL_AMG_GATHER_D, {&x.storage, &columns, &gathered});
            auto cmd = c.begin();
            const PCDot pc{static_cast<uint32_t>(partition->input_columns.size())};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipelines.pipeline[PL_AMG_GATHER_D]);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipelines.pl_layout[PL_AMG_GATHER_D], 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, c.pipelines.pl_layout[PL_AMG_GATHER_D], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (pc.n + 255) / 256, 1, 1);
            VulkanCommands::barrier(cmd);
            VulkanCommands::copy(cmd, gathered, input, input.size_bytes());
            c.flush();
        }
        if (gpu_rows) {
            const auto ds = c.descriptors(PL_SPMV_D, {&row_ptr, &col_ind, &values, &x.storage, &product.storage});
            const auto cmd = c.begin();
            const auto max_rows = c.ctx.device_info().max_workgroup_count_x;
            for (std::uint32_t row = 0; row < std::uint32_t(gpu_rows); row += max_rows) {
                const auto count = std::min(max_rows, std::uint32_t(gpu_rows) - row);
                cmd_spmv_d(cmd, c.pipelines, ds, {row, count, static_cast<std::uint32_t>(nc)}, count);
            }
        }
        if (input.handle()) {
            c.submit(); // GPU prefix can run while the CPU computes the suffix.
            partition->multiply_cpu(*host,
                {static_cast<const double*>(input.mapped_data()), partition->input_columns.size()},
                {static_cast<double*>(output.mapped_data()), nr - gpu_rows});
            const auto cmd = c.begin();
            VulkanCommands::copy(cmd, output, product.storage, output.size_bytes(), 0, std::size_t(gpu_rows) * 8);
        }
        amg_vector(alpha, product, 0, product, beta, y);
    }
};

bool VulkanAmgContext::prepare() {
    if (declined) return false;
    const auto limit = ctx.device_info().max_storage_buffer_range;
    std::size_t fixed = allocated + 65536;
    for (std::size_t i = 1; i < matrices.size(); ++i) {
        const auto& a = *matrices[i]->host;
        if (std::size_t(a.nnz) * 8 > limit || (std::size_t(a.nrows) + 1) * 4 > limit) return false;
        fixed += amg_csr_bytes(a, a.nrows);
    }
    auto& fine = *matrices.front();
    fine.partition = AmgDevicePartition::plan(*fine.host, budget, fixed, limit);
    if (!fine.partition) {
        log_debug(std::format("[vulkan device AMG] workspace does not fit: fixed={} budget={} bytes; using host hierarchy", fixed, budget));
        return false;
    }
    log_amg_device_plan("vulkan", *fine.host, *fine.partition, fixed, budget);
    fully_resident = std::size_t(fine.partition->gpu_rows) == fine.host->nrows;
    for (auto* matrix : matrices) matrix->allocate();
    return true;
}

struct VulkanAmgCoarse : AmgCoarseFactor {
    VulkanAmgContext& context;
    mutable std::vector<double> rhs, solution;
    VulkanBuffer input, output;
    template<class Matrix, class Params>
    VulkanAmgCoarse(const Matrix& a, const Params& p)
        : AmgCoarseFactor(*a), context(*p.context), rhs(amgcl::backend::rows(*a)), solution(rhs.size()),
          input(context.allocate(rhs.size() * sizeof(double), true)), output(context.allocate(input.size_bytes(), true)) {}
    // cppcheck-suppress constParameterReference -- the recorded GPU copy overwrites x's device storage
    void operator()(const VulkanAmgVector& b, VulkanAmgVector& x) const {
        auto cmd = context.begin();
        VulkanCommands::copy(cmd, b.storage, input, input.size_bytes());
        context.flush();
        std::memcpy(rhs.data(), input.mapped_data(), input.size_bytes());
        AmgCoarseFactor::operator()(rhs, solution);
        std::memcpy(output.mapped_data(), solution.data(), output.size_bytes());
        cmd = context.begin();
        VulkanCommands::copy(cmd, output, x.storage, output.size_bytes());
    }
};

struct VulkanAmgBackend {
    using value_type = double;
    using col_type = int;
    using ptr_type = int;
    using vector = VulkanAmgVector;
    using matrix = VulkanAmgMatrix;
    using matrix_diagonal = vector;
    using direct_solver = VulkanAmgCoarse;
    struct provides_row_iterator : std::false_type {};
    struct params { VulkanAmgContext* context{nullptr}; };
    static std::string name() { return "vulkan"; }
    static bool prepare(const params& p) { return p.context->prepare(); }
    static std::shared_ptr<matrix> copy_matrix(std::shared_ptr<amgcl::backend::builtin<double, int, int>::matrix> a, const params& p) {
        return std::make_shared<matrix>(*p.context, std::move(a));
    }
    static std::shared_ptr<vector> create_vector(std::size_t n, const params& p) {
        return std::make_shared<vector>(*p.context, n);
    }
    static std::shared_ptr<vector> copy_vector(const amgcl::backend::builtin<double>::vector& host, const params& p) {
        auto out = create_vector(host.size(), p); out->upload(std::span<const double>(host.data(), host.size())); return out;
    }
    static std::shared_ptr<vector> copy_vector(std::shared_ptr<amgcl::backend::builtin<double>::vector> host, const params& p) {
        return copy_vector(*host, p);
    }
    static std::shared_ptr<direct_solver> create_solver(std::shared_ptr<amgcl::backend::builtin<double, int, int>::matrix> a, const params& p) {
        return std::make_shared<direct_solver>(a, p);
    }
};
} // namespace vibestran::detail

namespace amgcl::backend {
using VV = vibestran::detail::VulkanAmgVector;
using VM = vibestran::detail::VulkanAmgMatrix;
template<class A, class B> struct spmv_impl<A, VM, VV, B, VV> {
    static void apply(A a, const VM& m, const VV& x, B b, VV& y) { m.multiply(a, x, b, y); }
};
template<> struct residual_impl<VM, VV, VV, VV> {
    static void apply(const VV& b, const VM& a, const VV& x, VV& r) {
        vibestran::detail::amg_vector(1, b, 0, b, 0, r); a.multiply(-1, x, 1, r);
    }
};
template<> struct clear_impl<VV> {
    static void apply(VV& x) { vibestran::detail::amg_vector(0, x, 0, x, 0, x); }
};
template<> struct copy_impl<VV, VV> {
    static void apply(const VV& x, VV& y) { vibestran::detail::amg_vector(1, x, 0, x, 0, y); }
};
template<> struct inner_product_impl<VV, VV> {
    static double get(const VV& x, const VV& y) {
        auto& c = x.context;
        const auto ds = c.descriptors(vibestran::PL_DOT_REDUCE_D, {&x.storage, &y.storage, &c.partial});
        const auto cmd = c.begin();
        vibestran::cmd_dot_reduce_d(cmd, c.pipelines, ds, static_cast<std::uint32_t>(x.size()));
        vibestran::VulkanCommands::barrier(cmd);
        const auto groups = (x.size() + 255) / 256;
        vibestran::VulkanCommands::copy(cmd, c.partial, c.readback, groups * sizeof(double));
        c.flush();
        const auto* data = static_cast<const double*>(c.readback.mapped_data());
        return std::accumulate(data, data + groups, 0.0);
    }
};
template<class A, class B> struct axpby_impl<A, VV, B, VV> {
    static void apply(A a, const VV& x, B b, VV& y) { vibestran::detail::amg_vector(a, x, 0, x, b, y); }
};
template<class A, class B, class C> struct axpbypcz_impl<A, VV, B, VV, C, VV> {
    static void apply(A a, const VV& x, B b, const VV& y, C c, VV& z) { vibestran::detail::amg_vector(a, x, b, y, c, z); }
};
template<class A, class B> struct vmul_impl<A, VV, VV, B, VV> {
    static void apply(A a, const VV& x, const VV& y, B b, VV& z) { vibestran::detail::amg_vector(a, x, 0, y, b, z, 1); }
};
} // namespace amgcl::backend

namespace vibestran {
std::vector<double> solve_vulkan_amg(VulkanContext& ctx, Pipelines& pl,
    const SparseMatrixBuilder::CsrData& k, const std::vector<double>& rhs,
    double tolerance, int max_iterations, int& iterations, RefinementReport& report, std::size_t budget, bool& fully_resident) {
    fully_resident = false;
    if (budget >= std::size_t(k.n) * 80 &&
        std::size_t(k.n) * 8 <= ctx.device_info().max_storage_buffer_range &&
        (std::size_t(k.n) + 255) / 256 <= ctx.device_info().max_workgroup_count_x) {
        detail::VulkanAmgContext context(ctx, pl, k.n, budget);
        auto result = detail::solve_amg_pcg<detail::VulkanAmgBackend>(k, rhs, tolerance, max_iterations, {&context},
            [](const auto& host, auto& device) { device.upload(host); },
            [](const auto& device, auto& host) { device.download(host); });
        context.flush();
        if (result.workspace_available) {
            fully_resident = context.fully_resident;
            iterations = result.iterations;
            report = result.refinement;
            return std::move(result.solution);
        }
    }
    return solve_vulkan_hybrid_amg(ctx, pl, k, rhs, budget, tolerance, max_iterations, iterations, report);
}
} // namespace vibestran
#endif
