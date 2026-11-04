#pragma once
// Included after the CUDA PCG context/buffer definitions, inside vibestran.
template<class T> struct PcgPinnedBuffer {
    T* ptr{nullptr};
    explicit PcgPinnedBuffer(std::size_t count) {
        if (count) throw_if_cuda_failed(cudaMallocHost(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "allocate PCG pinned buffer");
    }
    ~PcgPinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
    PcgPinnedBuffer(const PcgPinnedBuffer&) = delete;
    PcgPinnedBuffer& operator=(const PcgPinnedBuffer&) = delete;
};

template<class T>
__global__ static void partitioned_spmv_kernel(const int* rp, const int* ci,
    const T* values, const T* p, T* ap, int rows) {
    const int row = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;
    T sum = 0;
    for (int j = rp[row] + lane; j < rp[row + 1]; j += 32) sum += values[j] * p[ci[j]];
    for (int stride = 16; stride; stride /= 2) sum += __shfl_down_sync(0xffffffff, sum, stride);
    if (lane == 0) ap[row] = sum;
}

class CudaAmgSpmv final : public detail::AmgSpmv {
public:
    CudaAmgSpmv(const SparseMatrixBuilder::CsrData& k, std::size_t budget)
        : k_(k), rows_(detail::pcg_prefix_rows(k, budget, sizeof(double))),
          rp_(rows_ ? std::size_t(rows_) + 1 : 0), ci_(k.row_ptr[rows_]),
          matrix_(k.row_ptr[rows_]), p_(rows_ ? k.n : 0), ap_(rows_),
          host_p_(rows_ ? k.n : 0), host_ap_(rows_) {
        if (rows_) {
            rp_.upload(k.row_ptr.data(), std::size_t(rows_) + 1);
            ci_.upload(k.col_ind.data(), k.row_ptr[rows_]);
            matrix_.upload(k.values.data(), k.row_ptr[rows_]);
        }
        log_debug("[cuda-amg] partitioned finest matrix: " + std::to_string(rows_) +
            " GPU rows, " + std::to_string(k.n - rows_) + " CPU rows");
    }
    void multiply(std::span<const double> x, std::span<double> y) override {
        if (rows_) {
            std::copy(x.begin(), x.end(), host_p_.ptr);
            throw_if_cuda_failed(cudaMemcpyAsync(p_.ptr, host_p_.ptr, x.size_bytes(), cudaMemcpyHostToDevice), "AMG input upload");
            partitioned_spmv_kernel<double><<<(rows_ + 3) / 4, 128>>>(rp_.ptr, ci_.ptr, matrix_.ptr, p_.ptr, ap_.ptr, rows_);
            throw_if_cuda_failed(cudaGetLastError(), "AMG partitioned SpMV");
            throw_if_cuda_failed(cudaMemcpyAsync(host_ap_.ptr, ap_.ptr, std::size_t(rows_) * sizeof(double), cudaMemcpyDeviceToHost), "AMG output download");
        }
        detail::pcg_cpu_rows<double>(k_, k_.values, x, y, rows_);
        if (rows_) {
            throw_if_cuda_failed(cudaStreamSynchronize(nullptr), "AMG partitioned SpMV completion");
            std::copy(host_ap_.ptr, host_ap_.ptr + rows_, y.begin());
        }
    }
private:
    const SparseMatrixBuilder::CsrData& k_;
    int rows_;
    PCGDeviceBuffer<int> rp_, ci_;
    PCGDeviceBuffer<double> matrix_, p_, ap_;
    PcgPinnedBuffer<double> host_p_, host_ap_;
};

// Fixed GPU row prefix and CPU suffix. cudaMallocHost buffers persist for all
// iterations/refinement solves; CPU work overlaps the asynchronous GPU work.
template<class T>
static std::vector<double> solve_cuda_partitioned(CudaPCGContext& ctx,
    const SparseMatrixBuilder::CsrData& original, const std::vector<double>& rhs,
    std::size_t budget) {
    const auto full = original.expanded_symmetric();
    const int rows = detail::pcg_prefix_rows(full, budget, sizeof(T));
    const int nnz = full.row_ptr[rows];
    const auto n = std::size_t(full.n);
    const std::vector<T> values(full.values.begin(), full.values.end());
    const auto diagonal = build_jacobi_diagonal(full);
    PCGDeviceBuffer<int> rp(rows ? std::size_t(rows) + 1 : 0), ci(nnz);
    PCGDeviceBuffer<T> matrix(nnz), p(rows ? n : 0), ap(rows);
    PcgPinnedBuffer<T> host_p(rows ? n : 0), host_ap(rows);
    std::vector<T> cpu_p(n);
    if (rows) {
        rp.upload(full.row_ptr.data(), std::size_t(rows) + 1);
        ci.upload(full.col_ind.data(), nnz);
        matrix.upload(values.data(), nnz);
    }
    ctx.notices.push_back("CUDA PCG: matrix exceeds the device budget; using Jacobi PCG with " +
        std::to_string(rows) + " GPU rows and " + std::to_string(full.n - rows) + " CPU rows");
    log_debug("[cuda-pcg] partitioned: matrix bytes uploaded once=" +
        std::to_string(std::size_t(nnz) * (sizeof(T) + sizeof(int))) +
        ", vector bytes/iteration=" + std::to_string(rows ? (n + rows) * sizeof(T) : 0));
    const auto multiply = [&](std::span<const double> x, std::span<double> y) {
        std::transform(x.begin(), x.end(), cpu_p.begin(), [](double v) { return T(v); });
        if (rows) {
            std::copy(cpu_p.begin(), cpu_p.end(), host_p.ptr);
            throw_if_cuda_failed(cudaMemcpyAsync(p.ptr, host_p.ptr, n * sizeof(T), cudaMemcpyHostToDevice), "partitioned vector upload");
            partitioned_spmv_kernel<T><<<(rows + 3) / 4, 128>>>(rp.ptr, ci.ptr, matrix.ptr, p.ptr, ap.ptr, rows);
            throw_if_cuda_failed(cudaGetLastError(), "partitioned SpMV kernel");
            throw_if_cuda_failed(cudaMemcpyAsync(host_ap.ptr, ap.ptr, std::size_t(rows) * sizeof(T), cudaMemcpyDeviceToHost), "partitioned vector download");
        }
        detail::pcg_cpu_rows<T>(full, values, cpu_p, y, rows);
        if (rows) {
            throw_if_cuda_failed(cudaStreamSynchronize(nullptr), "partitioned SpMV completion");
            std::copy(host_ap.ptr, host_ap.ptr + rows, y.begin());
        }
    };
    int total = 0;
    const auto inner = [&](const std::vector<double>& b) {
        double scale = 0;
        for (double v : b) scale = std::max(scale, std::abs(v));
        if (scale == 0) return std::vector<double>(n, 0.0);
        std::vector<double> normalized(n);
        std::transform(b.begin(), b.end(), normalized.begin(), [=](double v) { return v / scale; });
        int iterations = 0;
        double residual = 0;
        auto x = detail::partitioned_pcg(diagonal, normalized, multiply, 1e-3, ctx.max_iters, iterations, residual);
        total += iterations;
        for (double& v : x) v *= scale;
        return x;
    };
    auto solution = inner(rhs);
    RefinementOptions options;
    options.tolerance = ctx.tolerance;
    options.max_corrections = kCudaPcgMaxCorrections;
    ctx.last_refinement = refine_solution(original, rhs, solution, inner, options);
    ctx.last_iters = total;
    ctx.last_rel_res = ctx.last_refinement->relative_residual;
    require_refined(ctx, "cuda-pcg partitioned");
    return solution;
}

// Free device memory less headroom for what the solver does not count, within
// any explicit budget. Reported free memory is not all allocatable, and kernel
// code loads on first launch, which a fraction of a nearly full device would not
// cover: with 100 MiB reported free, 39 MiB allocations failed on a Titan V.
static std::size_t cuda_device_budget(const CudaPCGContext& ctx) {
    constexpr std::size_t kMinimumHeadroom = std::size_t(256) * 1024 * 1024;
    std::size_t free_bytes = 0, total_bytes = 0;
    throw_if_cuda_failed(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    const auto headroom = std::max(free_bytes / 5, kMinimumHeadroom);
    const auto available = free_bytes > headroom ? free_bytes - headroom : 0;
    return ctx.device_memory_budget ? std::min(ctx.device_memory_budget, available) : available;
}

// Below the AMG size. AMG plans from its hierarchy instead (cuda_amg_engines).
template<class T>
static std::optional<std::size_t> cuda_pcg_spill_budget(const CudaPCGContext& ctx,
    const SparseMatrixBuilder::CsrData& k) {
    const auto budget = cuda_device_budget(ctx);
    // Conservative allowance for matrix, IC0, SpMV/SpSV scratch and vectors.
    const std::size_t estimate = std::size_t(k.nnz) * (4 * sizeof(T) + 4 * sizeof(int)) +
        std::size_t(k.n) * (12 * sizeof(T) + 8 * sizeof(int)) + 1048576;
    if (estimate > budget) return budget;
    return {};
}
