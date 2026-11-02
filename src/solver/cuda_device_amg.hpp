#pragma once
// Included at namespace scope after the CUDA buffer/context/coarse definitions.
#include "amg_device_spill.hpp"

namespace vibestran {
__global__ static void amg_gather_kernel(const double* x, const int* columns, double* packed, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) packed[i] = x[columns[i]];
}
__global__ static void amg_device_spmv_kernel(const int* rp, const int* ci, const double* values,
    const double* x, double* y, int rows, double alpha, double beta) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
    const int lane = threadIdx.x % 8;
    if (row >= rows) return;
    double sum = 0;
    for (int j = rp[row] + lane; j < rp[row + 1]; j += 8) sum += values[j] * x[ci[j]];
    for (int d = 4; d; d /= 2) sum += __shfl_down_sync(__activemask(), sum, d, 8);
    if (lane == 0) y[row] = alpha * sum + (beta == 0 ? 0 : beta * y[row]);
}
__global__ static void amg_tail_update_kernel(const double* product, double* y, int n, double alpha, double beta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * product[i] + (beta == 0 ? 0 : beta * y[i]);
}

struct CudaDeviceAmgMatrix;
struct CudaDeviceAmgContext {
    std::size_t budget, vector_bytes{0};
    std::vector<CudaDeviceAmgMatrix*> matrices;
    explicit CudaDeviceAmgContext(std::size_t bytes) : budget(bytes) {}
    bool prepare();
};

struct CudaDeviceAmgMatrix {
    using value_type = double;
    std::shared_ptr<detail::AmgHostCsr> host;
    PCGDeviceBuffer<int> rp, ci, columns;
    PCGDeviceBuffer<double> values, gathered, tail;
    std::optional<detail::AmgDevicePartition> partition;
    std::unique_ptr<PcgPinnedBuffer<double>> input, output;
    cudaEvent_t input_ready{nullptr};
    int gpu_rows{0};

    CudaDeviceAmgMatrix(std::shared_ptr<detail::AmgHostCsr> a, CudaDeviceAmgContext& c) : host(std::move(a)) {
        c.matrices.push_back(this);
    }
    ~CudaDeviceAmgMatrix() { if (input_ready) cudaEventDestroy(input_ready); }
    std::size_t rows() const { return host->nrows; }
    std::size_t cols() const { return host->ncols; }
    std::size_t nonzeros() const { return host->nnz; }
    std::size_t bytes() const { return detail::amg_csr_bytes(*host, gpu_rows); }
    void allocate() {
        gpu_rows = partition ? partition->gpu_rows : int(host->nrows);
        if (gpu_rows) {
            rp = PCGDeviceBuffer<int>(gpu_rows + 1); ci = PCGDeviceBuffer<int>(host->ptr[gpu_rows]);
            values = PCGDeviceBuffer<double>(host->ptr[gpu_rows]);
            rp.upload(host->ptr, gpu_rows + 1); ci.upload(host->col, host->ptr[gpu_rows]);
            values.upload(host->val, host->ptr[gpu_rows]);
        }
        if (std::size_t(gpu_rows) < host->nrows) {
            const auto count = partition->input_columns.size();
            columns = PCGDeviceBuffer<int>(count); columns.upload(partition->input_columns.data(), count);
            gathered = PCGDeviceBuffer<double>(count); tail = PCGDeviceBuffer<double>(host->nrows - gpu_rows);
            input = std::make_unique<PcgPinnedBuffer<double>>(count);
            output = std::make_unique<PcgPinnedBuffer<double>>(host->nrows - gpu_rows);
            throw_if_cuda_failed(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming), "create AMG gather event");
        }
    }
    void multiply(double alpha, const thrust::device_vector<double>& x, double beta, thrust::device_vector<double>& y) const {
        const double* xp = thrust::raw_pointer_cast(x.data());
        double* yp = thrust::raw_pointer_cast(y.data());
        const int count = input ? int(partition->input_columns.size()) : 0;
        if (input) {
            if (count) {
                amg_gather_kernel<<<(count + 255) / 256, 256>>>(xp, columns.ptr, gathered.ptr, count);
                throw_if_cuda_failed(cudaMemcpyAsync(input->ptr, gathered.ptr, std::size_t(count) * 8, cudaMemcpyDeviceToHost), "AMG gathered input download");
            }
            throw_if_cuda_failed(cudaEventRecord(input_ready), "record AMG gathered input");
        }
        if (gpu_rows)
            amg_device_spmv_kernel<<<(gpu_rows + 31) / 32, 256>>>(rp.ptr, ci.ptr, values.ptr, xp, yp, gpu_rows, alpha, beta);
        if (input) {
            throw_if_cuda_failed(cudaEventSynchronize(input_ready), "wait for AMG gathered input");
            const int n = host->nrows - gpu_rows;
            partition->multiply_cpu(*host, {input->ptr, std::size_t(count)}, {output->ptr, std::size_t(n)});
            throw_if_cuda_failed(cudaMemcpyAsync(tail.ptr, output->ptr, std::size_t(n) * 8, cudaMemcpyHostToDevice), "AMG CPU output upload");
            amg_tail_update_kernel<<<(n + 255) / 256, 256>>>(tail.ptr, yp + gpu_rows, n, alpha, beta);
        }
        throw_if_cuda_failed(cudaGetLastError(), "device-owned AMG product kernels");
    }
};

inline bool CudaDeviceAmgContext::prepare() {
    std::size_t fixed = vector_bytes + 65536;
    for (std::size_t i = 1; i < matrices.size(); ++i) fixed += detail::amg_csr_bytes(*matrices[i]->host, matrices[i]->rows());
    auto& fine = *matrices.front();
    fine.partition = detail::AmgDevicePartition::plan(*fine.host, budget, fixed);
    if (!fine.partition) {
        log_debug(std::format("[cuda device AMG] workspace does not fit: fixed={} budget={} bytes; using host hierarchy", fixed, budget));
        return false;
    }
    detail::log_amg_device_plan("cuda", *fine.host, *fine.partition, fixed, budget);
    for (auto* matrix : matrices) matrix->allocate();
    return true;
}

struct CudaDeviceAmgBackend : amgcl::backend::cuda<double, CudaAmgCoarse> {
    using Base = amgcl::backend::cuda<double, CudaAmgCoarse>;
    using matrix = CudaDeviceAmgMatrix;
    using col_type = int;
    using ptr_type = int;
    struct params { CudaDeviceAmgContext* context; };
    static std::shared_ptr<matrix> copy_matrix(std::shared_ptr<detail::AmgHostCsr> a, const params& p) {
        return std::make_shared<matrix>(std::move(a), *p.context);
    }
    static std::shared_ptr<vector> create_vector(std::size_t n, const params& p) {
        p.context->vector_bytes += n * sizeof(double);
        // Setup can decline the device budget without an allocation failure.
        return std::make_shared<vector>(p.context->vector_bytes <= p.context->budget ? n : 0);
    }
    static std::shared_ptr<vector> copy_vector(const amgcl::backend::builtin<double>::vector& host, const params& p) {
        auto out = create_vector(host.size(), p);
        if (out->size()) thrust::copy(host.data(), host.data() + host.size(), out->begin());
        return out;
    }
    static std::shared_ptr<vector> copy_vector(std::shared_ptr<amgcl::backend::builtin<double>::vector> host, const params& p) {
        return copy_vector(*host, p);
    }
    static std::shared_ptr<direct_solver> create_solver(std::shared_ptr<detail::AmgHostCsr> a, const params& p) {
        return std::make_shared<direct_solver>(a, p);
    }
    static bool prepare(const params& p) { return p.context->prepare(); }
};
} // namespace vibestran

namespace amgcl::backend {
template<class A, class B> struct spmv_impl<A, vibestran::CudaDeviceAmgMatrix, thrust::device_vector<double>, B, thrust::device_vector<double>> {
    static void apply(A alpha, const vibestran::CudaDeviceAmgMatrix& a, const thrust::device_vector<double>& x,
        B beta, thrust::device_vector<double>& y) { a.multiply(alpha, x, beta, y); }
};
template<> struct residual_impl<vibestran::CudaDeviceAmgMatrix, thrust::device_vector<double>, thrust::device_vector<double>, thrust::device_vector<double>> {
    static void apply(const thrust::device_vector<double>& b, const vibestran::CudaDeviceAmgMatrix& a,
        const thrust::device_vector<double>& x, thrust::device_vector<double>& r) {
        thrust::copy(b.begin(), b.end(), r.begin()); a.multiply(-1, x, 1, r);
    }
};
} // namespace amgcl::backend

namespace vibestran {
static std::vector<double> solve_cuda_device_amg(CudaPCGContext& ctx, const SparseMatrixBuilder::CsrData& k,
    const std::vector<double>& rhs, std::size_t budget) {
    if (budget >= std::size_t(k.n) * 80) {
        CudaDeviceAmgContext context(budget);
        auto result = detail::solve_amg_pcg<CudaDeviceAmgBackend>(k, rhs, ctx.tolerance, ctx.max_iters, {&context},
            [](const auto& host, auto& device) { thrust::copy(host.begin(), host.end(), device.begin()); },
            [](const auto& device, auto& host) { thrust::copy(device.begin(), device.end(), host.begin()); });
        if (result.workspace_available) {
            ctx.last_iters = result.iterations; ctx.last_refinement = result.refinement;
            ctx.last_rel_res = result.refinement.relative_residual;
            require_refined(ctx, "cuda-pcg device AMG");
            return std::move(result.solution);
        }
    }
    return solve_cuda_hybrid_amg(ctx, k, rhs, budget);
}
} // namespace vibestran
