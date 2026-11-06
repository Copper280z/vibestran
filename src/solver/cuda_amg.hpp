#pragma once
// Included at namespace scope inside vibestran, after the CUDA PCG buffer,
// context and pinned-buffer definitions.
//
// Device AMG PCG. The V-cycle and CG run on one stream with every scalar on
// the device, so an iteration has no host synchronization and is replayed as
// a CUDA graph; the host polls convergence every kPollIterations (every
// iteration when CPU rows would otherwise do discarded work). Sparse
// products apply their vector update in the same kernel (residual, smoothing,
// prolongation). With a device budget the fine operator keeps a GPU row
// prefix; its CPU rows run in a stream host callback on a second stream,
// concurrently with the prefix product, and are captured in the same graph.
#include "amg_device_spill.hpp"

namespace vibestran::cuda_amg {
constexpr int kBlock = 256;
constexpr int kDotBlocks = 256;
constexpr int kPollIterations = 8;
constexpr int kFlagConverged = 1, kFlagBreakdown = 2, kFlagLimit = 4;

enum Epilogue : int { kSet, kAdd, kResidual, kSmooth, kSubtract, kCorrect };

struct CgState {
    double rho, alpha, beta, norm_b, target, rr;
    int flags, iterations, max_iterations, pad;
};

template<class T, int Mode>
__device__ __forceinline__ T epilogue(T sum, int row, const T* y, const T* f, const T* x, const T* m) {
    if constexpr (Mode == kSet) return sum;
    else if constexpr (Mode == kAdd) return y[row] + sum;
    else if constexpr (Mode == kResidual) return f[row] - sum;
    else if constexpr (Mode == kSmooth) return x[row] + m[row] * (f[row] - sum);
    else if constexpr (Mode == kSubtract) return y[row] - sum;
    else return y[row] + f[row] - m[row] * sum;
}

// Lanes threads per row; all threads reach the shuffles.
template<class T, int Lanes, int Mode>
__global__ static void spmv_kernel(const int* __restrict__ rp, const int* __restrict__ ci,
    const T* __restrict__ v, const T* __restrict__ x, T* __restrict__ y, const T* __restrict__ f,
    const T* __restrict__ m, int rows, const CgState* __restrict__ state) {
    if (state->flags) return;
    const int row = int((blockIdx.x * blockDim.x + threadIdx.x) / Lanes);
    const int lane = int(threadIdx.x % Lanes);
    T sum = 0;
    if (row < rows)
        for (int j = rp[row] + lane; j < rp[row + 1]; j += Lanes) sum += v[j] * x[ci[j]];
    for (int d = Lanes / 2; d; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d, Lanes);
    if (lane == 0 && row < rows) y[row] = epilogue<T, Mode>(sum, row, y, f, x, m);
}

// CPU rows of a partitioned product: rows offset..offset+count of y.
template<class T, int Mode>
__global__ static void tail_kernel(const T* __restrict__ product, T* __restrict__ y, const T* __restrict__ f,
    const T* __restrict__ x, const T* __restrict__ m, int offset, int count, const CgState* __restrict__ state) {
    if (state->flags) return;
    const int k = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (k < count) y[offset + k] = epilogue<T, Mode>(product[k], offset + k, y, f, x, m);
}

template<class T>
__global__ static void gather_kernel(const T* __restrict__ x, const int* __restrict__ columns,
    T* __restrict__ packed, int n, const CgState* __restrict__ state) {
    if (state->flags) return;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) packed[i] = x[columns[i]];
}

template<class T>
__global__ static void scale_kernel(const T* __restrict__ m, const T* __restrict__ f, T* __restrict__ x,
    int n, const CgState* __restrict__ state) {
    if (state->flags) return;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) x[i] = m[i] * f[i];
}

// y = T e from the patches of AmgTentative: a thread per row.
template<class T>
__global__ static void tentative_kernel(const T* __restrict__ values, const int* __restrict__ equation,
    const int* __restrict__ row_patch, const int* __restrict__ patches, const T* __restrict__ e, T* __restrict__ y,
    int rows, const CgState* __restrict__ state) {
    if (state->flags) return;
    const int row = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    const int* patch = patches + 4 * row_patch[row];
    const int first_col = patch[2], width = patch[6] - first_col;
    const T* block = values + patch[1] + (row - patch[0]) * width;
    T sum = 0;
    for (int k = 0; k < width; ++k) sum += block[k] * e[first_col + k];
    y[equation[row]] = sum;
}

// y = T^T x: Lanes threads per coarse equation.
template<class T, int Lanes>
__global__ static void tentative_transpose_kernel(const T* __restrict__ values, const int* __restrict__ equation,
    const int* __restrict__ col_patch, const int* __restrict__ patches, const T* __restrict__ x, T* __restrict__ y,
    int cols, const CgState* __restrict__ state) {
    if (state->flags) return;
    const int col = int((blockIdx.x * blockDim.x + threadIdx.x) / Lanes);
    const int lane = int(threadIdx.x % Lanes);
    T sum = 0;
    if (col < cols) {
        const int* patch = patches + 4 * col_patch[col];
        const int first_row = patch[0], rows = patch[4] - first_row, width = patch[6] - patch[2];
        const T* column = values + patch[1] + (col - patch[2]);
        for (int r = lane; r < rows; r += Lanes) sum += column[r * width] * x[equation[first_row + r]];
    }
    for (int d = Lanes / 2; d; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d, Lanes);
    if (lane == 0 && col < cols) y[col] = sum;
}

// y = C f for the dense coarse inverse: one warp per row.
template<class T>
__global__ static void dense_kernel(const T* __restrict__ c, const T* __restrict__ f, T* __restrict__ y,
    int n, const CgState* __restrict__ state) {
    if (state->flags) return;
    const int row = int((blockIdx.x * blockDim.x + threadIdx.x) / 32);
    const int lane = int(threadIdx.x % 32);
    T sum = 0;
    if (row < n)
        for (int j = lane; j < n; j += 32) sum += c[std::size_t(row) * n + j] * f[j];
    for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d);
    if (lane == 0 && row < n) y[row] = sum;
}

template<class From, class To>
__global__ static void convert_kernel(const From* __restrict__ in, To* __restrict__ out, int n,
    const CgState* __restrict__ state) {
    if (state->flags) return;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = To(in[i]);
}

__device__ __forceinline__ double block_sum(double v) {
    __shared__ double shared[kBlock];
    shared[threadIdx.x] = v;
    __syncthreads();
    for (int s = kBlock / 2; s; s /= 2) {
        if (int(threadIdx.x) < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    return shared[0];
}

// Fixed grid, so the summation order (and the iteration history) is
// independent of timing.
__global__ static void dot_kernel(const double* __restrict__ a, const double* __restrict__ b, int n,
    double* __restrict__ partial, const CgState* __restrict__ state) {
    if (state->flags) return;
    double sum = 0;
    for (int i = int(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += kDotBlocks * kBlock) sum += a[i] * b[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

__device__ __forceinline__ double partial_total(const double* partial) {
    return block_sum(threadIdx.x < kDotBlocks ? partial[threadIdx.x] : 0.0);
}

__global__ static void init_kernel(const double* __restrict__ partial, CgState* __restrict__ state) {
    const double rr = partial_total(partial);
    if (threadIdx.x) return;
    state->norm_b = std::sqrt(rr);
    state->rr = rr;
    state->iterations = 0;
    state->flags = state->max_iterations <= 0 || !(state->norm_b > state->target * state->norm_b) ? kFlagConverged : 0;
}

__global__ static void rho_kernel(const double* __restrict__ partial, CgState* __restrict__ state) {
    if (state->flags) return;
    const double rho = partial_total(partial);
    if (threadIdx.x) return;
    state->beta = state->iterations ? rho / state->rho : 0.0;
    state->rho = rho;
}

__global__ static void direction_kernel(const double* __restrict__ z, double* __restrict__ p, int n,
    const CgState* __restrict__ state) {
    if (state->flags) return;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) p[i] = state->iterations ? z[i] + state->beta * p[i] : z[i];
}

__global__ static void alpha_kernel(const double* __restrict__ partial, CgState* __restrict__ state) {
    if (state->flags) return;
    const double pq = partial_total(partial);
    if (threadIdx.x) return;
    if (!(pq > 0)) { state->flags |= kFlagBreakdown; return; }
    state->alpha = state->rho / pq;
}

// u += alpha p, r -= alpha q, and the partial sums of r.r.
__global__ static void update_kernel(const double* __restrict__ p, const double* __restrict__ q,
    double* __restrict__ u, double* __restrict__ r, int n, double* __restrict__ partial,
    const CgState* __restrict__ state) {
    if (state->flags) return;
    const double alpha = state->alpha;
    double sum = 0;
    for (int i = int(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += kDotBlocks * kBlock) {
        u[i] += alpha * p[i];
        const double ri = r[i] - alpha * q[i];
        r[i] = ri;
        sum += ri * ri;
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

__global__ static void check_kernel(const double* __restrict__ partial, CgState* __restrict__ state) {
    if (state->flags) return;
    const double rr = partial_total(partial);
    if (threadIdx.x) return;
    ++state->iterations;
    state->rr = rr;
    if (!(std::sqrt(rr) > state->target * state->norm_b)) state->flags |= kFlagConverged;
    else if (state->iterations >= state->max_iterations) state->flags |= kFlagLimit;
}

inline int blocks(std::size_t threads) { return int((threads + kBlock - 1) / kBlock); }

// Lanes per row from the average row length.
inline int lanes_for(const detail::AmgHostCsr& a) {
    const double average = a.nrows ? double(a.nnz) / double(a.nrows) : 0.0;
    return average <= 6 ? 4 : average <= 12 ? 8 : average <= 24 ? 16 : 32;
}

template<class T> struct DeviceCsr {
    int rows{0}, lanes{8};
    PCGDeviceBuffer<int> rp, ci;
    PCGDeviceBuffer<T> val;

    DeviceCsr() = default;
    DeviceCsr(const detail::AmgHostCsr& a, int prefix) : rows(prefix), lanes(lanes_for(a)) {
        if (!rows) return;
        const auto nnz = std::size_t(a.ptr[rows]);
        rp = PCGDeviceBuffer<int>(std::size_t(rows) + 1); ci = PCGDeviceBuffer<int>(nnz);
        val = PCGDeviceBuffer<T>(nnz);
        rp.upload(a.ptr, std::size_t(rows) + 1); ci.upload(a.col, nnz);
        upload_values(a);
    }
    void upload_values(const detail::AmgHostCsr& a) {
        if (!rows) return;
        const auto nnz = std::size_t(a.ptr[rows]);
        if constexpr (std::is_same_v<T, double>) val.upload(a.val, nnz);
        else {
            const std::vector<T> converted(a.val, a.val + nnz);
            val.upload(converted.data(), nnz);
        }
    }
    static std::size_t bytes(const detail::AmgHostCsr& a, int prefix) {
        return prefix ? std::size_t(a.ptr[prefix]) * (sizeof(T) + sizeof(int)) + (std::size_t(prefix) + 1) * sizeof(int) : 0;
    }

    template<int Mode>
    void launch(cudaStream_t s, const T* x, T* y, const T* f, const T* m, const CgState* state) const {
        if (!rows) return;
        const auto grid = blocks(std::size_t(rows) * lanes);
        switch (lanes) {
        case 4: spmv_kernel<T, 4, Mode><<<grid, kBlock, 0, s>>>(rp.ptr, ci.ptr, val.ptr, x, y, f, m, rows, state); break;
        case 8: spmv_kernel<T, 8, Mode><<<grid, kBlock, 0, s>>>(rp.ptr, ci.ptr, val.ptr, x, y, f, m, rows, state); break;
        case 16: spmv_kernel<T, 16, Mode><<<grid, kBlock, 0, s>>>(rp.ptr, ci.ptr, val.ptr, x, y, f, m, rows, state); break;
        default: spmv_kernel<T, 32, Mode><<<grid, kBlock, 0, s>>>(rp.ptr, ci.ptr, val.ptr, x, y, f, m, rows, state); break;
        }
    }
};

// AmgTentative on the device.
template<class T> struct DeviceTentative {
    static constexpr int kLanes = 32;
    int rows{0}, cols{0};
    PCGDeviceBuffer<int> equation, row_patch, col_patch, patches;
    PCGDeviceBuffer<T> values;

    DeviceTentative() = default;
    explicit DeviceTentative(const detail::AmgTentative& t)
        : rows(int(t.equation.size())), cols(int(t.col_patch.size())) {
        equation = PCGDeviceBuffer<int>(t.equation.size()); equation.upload(t.equation.data(), t.equation.size());
        row_patch = PCGDeviceBuffer<int>(t.row_patch.size()); row_patch.upload(t.row_patch.data(), t.row_patch.size());
        col_patch = PCGDeviceBuffer<int>(t.col_patch.size()); col_patch.upload(t.col_patch.data(), t.col_patch.size());
        patches = PCGDeviceBuffer<int>(t.patches.size() * 4); patches.upload(t.patches.front().data(), t.patches.size() * 4);
        values = PCGDeviceBuffer<T>(t.values.size());
        if constexpr (std::is_same_v<T, double>) values.upload(t.values.data(), t.values.size());
        else {
            const std::vector<T> converted(t.values.begin(), t.values.end());
            values.upload(converted.data(), converted.size());
        }
    }
    void apply(cudaStream_t s, const T* e, T* y, const CgState* state) const {
        tentative_kernel<T><<<blocks(std::size_t(rows)), kBlock, 0, s>>>(values.ptr, equation.ptr, row_patch.ptr, patches.ptr, e, y, rows, state);
    }
    void apply_transpose(cudaStream_t s, const T* x, T* y, const CgState* state) const {
        tentative_transpose_kernel<T, kLanes><<<blocks(std::size_t(cols) * kLanes), kBlock, 0, s>>>(values.ptr, equation.ptr, col_patch.ptr, patches.ptr, x, y, cols, state);
    }
};

// Uploads into existing storage, so captured graphs keep valid pointers.
template<class T> void upload_converted(PCGDeviceBuffer<T>& out, std::span<const double> host) {
    if constexpr (std::is_same_v<T, double>) out.upload(host.data(), host.size());
    else {
        const std::vector<T> converted(host.begin(), host.end());
        out.upload(converted.data(), host.size());
    }
}
template<class T> PCGDeviceBuffer<T> device_copy(std::span<const double> host) {
    PCGDeviceBuffer<T> out(host.size());
    upload_converted(out, host);
    return out;
}

// The fine operator with its optional CPU row suffix.
class FineOperator {
public:
    FineOperator(const detail::AmgHostCsr& a, std::optional<detail::AmgDevicePartition> partition)
        : host_(&a), partition_(std::move(partition)),
          gpu_(a, partition_ ? partition_->gpu_rows : int(a.nrows)) {
        if (!partition_ || std::size_t(partition_->gpu_rows) == a.nrows) return;
        const auto count = partition_->input_columns.size();
        tail_rows_ = int(a.nrows) - partition_->gpu_rows;
        columns_ = PCGDeviceBuffer<int>(count); columns_.upload(partition_->input_columns.data(), count);
        gathered_ = PCGDeviceBuffer<double>(count); tail_ = PCGDeviceBuffer<double>(tail_rows_);
        input_ = std::make_unique<PcgPinnedBuffer<double>>(count);
        output_ = std::make_unique<PcgPinnedBuffer<double>>(tail_rows_);
        throw_if_cuda_failed(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking), "create AMG CPU-row stream");
        throw_if_cuda_failed(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming), "create AMG fork event");
        throw_if_cuda_failed(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming), "create AMG join event");
    }
    ~FineOperator() {
        if (fork_) cudaEventDestroy(fork_);
        if (join_) cudaEventDestroy(join_);
        if (side_) cudaStreamDestroy(side_);
    }
    FineOperator(const FineOperator&) = delete;
    FineOperator& operator=(const FineOperator&) = delete;

    void update_values(const detail::AmgHostCsr& a) { host_ = &a; gpu_.upload_values(a); }

    template<int Mode>
    void apply(cudaStream_t s, const double* x, double* y, const double* f, const double* m, const CgState* state) {
        if (tail_rows_) {
            const int count = int(partition_->input_columns.size());
            // Gather before the fork: on a branch of its own the small gather
            // waits for SMs behind the prefix product, so the CPU rows would
            // start only after it instead of alongside it.
            if (count)
                gather_kernel<double><<<blocks(count), kBlock, 0, s>>>(x, columns_.ptr, gathered_.ptr, count, state);
            throw_if_cuda_failed(cudaEventRecord(fork_, s), "fork AMG CPU rows");
            throw_if_cuda_failed(cudaStreamWaitEvent(side_, fork_), "wait AMG fork");
            if (count)
                throw_if_cuda_failed(cudaMemcpyAsync(input_->ptr, gathered_.ptr, std::size_t(count) * 8,
                    cudaMemcpyDeviceToHost, side_), "AMG gathered input download");
            throw_if_cuda_failed(cudaLaunchHostFunc(side_, &FineOperator::cpu_rows, this), "AMG CPU-row callback");
            throw_if_cuda_failed(cudaMemcpyAsync(tail_.ptr, output_->ptr, std::size_t(tail_rows_) * 8,
                cudaMemcpyHostToDevice, side_), "AMG CPU output upload");
            throw_if_cuda_failed(cudaEventRecord(join_, side_), "join AMG CPU rows");
        }
        gpu_.template launch<Mode>(s, x, y, f, m, state);
        if (tail_rows_) {
            throw_if_cuda_failed(cudaStreamWaitEvent(s, join_), "wait AMG CPU rows");
            tail_kernel<double, Mode><<<blocks(tail_rows_), kBlock, 0, s>>>(
                tail_.ptr, y, f, x, m, partition_->gpu_rows, tail_rows_, state);
        }
    }
    int gpu_rows() const { return gpu_.rows; }
    bool partitioned() const { return tail_rows_ > 0; }

private:
    // Runs on a CUDA driver thread; no CUDA calls. An inactive solve still
    // computes a (discarded) product: the host cannot read device flags here.
    static void CUDART_CB cpu_rows(void* self) {
        auto& op = *static_cast<FineOperator*>(self);
        op.partition_->multiply_cpu(*op.host_, {op.input_->ptr, op.partition_->input_columns.size()},
            {op.output_->ptr, std::size_t(op.tail_rows_)});
    }

    const detail::AmgHostCsr* host_;
    std::optional<detail::AmgDevicePartition> partition_;
    DeviceCsr<double> gpu_;
    int tail_rows_{0};
    PCGDeviceBuffer<int> columns_;
    PCGDeviceBuffer<double> gathered_, tail_;
    std::unique_ptr<PcgPinnedBuffer<double>> input_, output_;
    cudaStream_t side_{nullptr};
    cudaEvent_t fork_{nullptr}, join_{nullptr};
};

// T is the hierarchy precision. With T = float, CG keeps an fp64 fine
// operator and fp64 vectors; the cycle converts its input and output.
template<class T>
class Solver {
public:
    // Device bytes other than the fine operator, for the budget planner. With
    // computed_transfers the finest P and R are applied from A, M and T.
    static std::size_t fixed_bytes(const detail::AmgHierarchy& h, bool computed_transfers) {
        const auto n = h.levels.front().A->nrows;
        std::size_t bytes = 5 * n * sizeof(double) + kDotBlocks * sizeof(double) + sizeof(CgState) + 65536;
        for (std::size_t l = 0; l < h.levels.size(); ++l) {
            const auto& level = h.levels[l];
            const auto rows = level.A->nrows;
            // f, x, y per level; fp64 level 0 uses r and z for f and y.
            bytes += (std::is_same_v<T, double> && l == 0 ? 1 : 3) * rows * sizeof(T);
            bytes += level.smoother.size() * sizeof(T);
            if (l || !std::is_same_v<T, double>) bytes += DeviceCsr<T>::bytes(*level.A, int(rows));
            if (computed_transfers && l == 0) bytes += level.tentative->bytes(sizeof(T)) + rows * sizeof(T);
            else if (level.P) bytes += DeviceCsr<T>::bytes(*level.P, int(level.P->nrows)) + DeviceCsr<T>::bytes(*level.R, int(level.R->nrows));
        }
        return bytes + h.coarse_inverse.size() * sizeof(T);
    }

    Solver(const detail::AmgHierarchy& h, std::optional<detail::AmgDevicePartition> partition, bool computed_transfers)
        : h_(h), n_(int(h.levels.front().A->nrows)), fine_(*h.levels.front().A, std::move(partition)),
          r_(n_), u_(n_), p_(n_), q_(n_), z_(n_), partial_(kDotBlocks), state_(1),
          host_state_(std::make_unique<PcgPinnedBuffer<CgState>>(1)) {
        throw_if_cuda_failed(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create AMG stream");
        for (std::size_t l = 0; l < h.levels.size(); ++l) {
            const auto& level = h.levels[l];
            const auto rows = level.A->nrows;
            Level d;
            const bool alias = std::is_same_v<T, double> && l == 0;
            if (!alias) { d.f = PCGDeviceBuffer<T>(rows); d.y = PCGDeviceBuffer<T>(rows); }
            d.x = PCGDeviceBuffer<T>(rows);
            if (!level.smoother.empty()) d.m = device_copy<T>(level.smoother);
            if (l || !std::is_same_v<T, double>) d.a = DeviceCsr<T>(*level.A, int(rows));
            if (computed_transfers && l == 0) { d.tentative = DeviceTentative<T>(*level.tentative); d.work = PCGDeviceBuffer<T>(rows); }
            else if (level.P) { d.p = DeviceCsr<T>(*level.P, int(level.P->nrows)); d.r = DeviceCsr<T>(*level.R, int(level.R->nrows)); }
            levels_.push_back(std::move(d));
        }
        if (h.coarse_direct()) coarse_ = device_copy<T>(h.coarse_inverse);
    }
    ~Solver() {
        if (graph_) cudaGraphExecDestroy(graph_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    // Same patterns, new values (AmgHierarchy::rebuild).
    void update_values() {
        sync();
        fine_.update_values(*h_.levels.front().A);
        for (std::size_t l = 0; l < h_.levels.size(); ++l) {
            const auto& level = h_.levels[l];
            auto& d = levels_[l];
            if (!level.smoother.empty()) upload_converted(d.m, level.smoother);
            if (d.a.rows) d.a.upload_values(*level.A);
            if (level.tentative && d.p.rows) { d.p.upload_values(*level.P); d.r.upload_values(*level.R); }
        }
        if (h_.coarse_direct()) upload_converted(coarse_, h_.coarse_inverse);
    }

    std::pair<int, double> solve(const std::vector<double>& b, double tolerance, int max_iterations,
                                 std::vector<double>& x) {
        throw_if_cuda_failed(cudaMemcpyAsync(r_.ptr, b.data(), std::size_t(n_) * 8, cudaMemcpyHostToDevice, stream_), "AMG rhs upload");
        throw_if_cuda_failed(cudaMemsetAsync(u_.ptr, 0, std::size_t(n_) * 8, stream_), "AMG solution reset");
        CgState initial{};
        initial.target = tolerance;
        initial.max_iterations = max_iterations;
        *host_state_->ptr = initial;
        throw_if_cuda_failed(cudaMemcpyAsync(state_.ptr, host_state_->ptr, sizeof(CgState), cudaMemcpyHostToDevice, stream_), "AMG state upload");
        dot_kernel<<<kDotBlocks, kBlock, 0, stream_>>>(r_.ptr, r_.ptr, n_, partial_.ptr, state_.ptr);
        init_kernel<<<1, kBlock, 0, stream_>>>(partial_.ptr, state_.ptr);
        throw_if_cuda_failed(cudaGetLastError(), "AMG CG start");
        if (!graph_) capture();
        CgState& state = *host_state_->ptr;
        for (;;) {
            for (int i = 0; i < (fine_.partitioned() ? 1 : kPollIterations); ++i)
                throw_if_cuda_failed(cudaGraphLaunch(graph_, stream_), "AMG iteration graph");
            throw_if_cuda_failed(cudaMemcpyAsync(host_state_->ptr, state_.ptr, sizeof(CgState), cudaMemcpyDeviceToHost, stream_), "AMG state download");
            sync();
            if (state.flags) break;
        }
        throw_if_cuda_failed(cudaMemcpyAsync(x.data(), u_.ptr, std::size_t(n_) * 8, cudaMemcpyDeviceToHost, stream_), "AMG solution download");
        sync();
        return {state.iterations, state.norm_b > 0 ? std::sqrt(state.rr) / state.norm_b : 0.0};
    }

    int fine_gpu_rows() const { return fine_.gpu_rows(); }

private:
    struct Level {
        PCGDeviceBuffer<T> f, x, y, m, work;
        DeviceCsr<T> a, p, r;
        DeviceTentative<T> tentative;
    };

    void sync() { throw_if_cuda_failed(cudaStreamSynchronize(stream_), "AMG stream synchronize"); }

    T* f(std::size_t l) { return std::is_same_v<T, double> && l == 0 ? reinterpret_cast<T*>(r_.ptr) : levels_[l].f.ptr; }
    T* y(std::size_t l) { return std::is_same_v<T, double> && l == 0 ? reinterpret_cast<T*>(z_.ptr) : levels_[l].y.ptr; }

    template<int Mode> void level_product(std::size_t l, const T* x, T* out, const T* rhs, const T* m) {
        if constexpr (std::is_same_v<T, double>)
            if (l == 0) { fine_.template apply<Mode>(stream_, x, out, rhs, m, state_.ptr); return; }
        levels_[l].a.template launch<Mode>(stream_, x, out, rhs, m, state_.ptr);
    }

public:
    // V-cycle operations (detail::amg_vcycle).
    void smooth_zero(std::size_t l) {
        const auto rows = h_.levels[l].A->nrows;
        scale_kernel<T><<<blocks(rows), kBlock, 0, stream_>>>(levels_[l].m.ptr, f(l), levels_[l].x.ptr, int(rows), state_.ptr);
    }
    void residual(std::size_t l) { level_product<kResidual>(l, levels_[l].x.ptr, y(l), f(l), nullptr); }
    // R y = T^T (y - A M y). y is not read again before smooth_post writes it.
    void restriction(std::size_t l) {
        auto& d = levels_[l];
        if (!d.tentative.rows) { d.r.template launch<kSet>(stream_, y(l), f(l + 1), nullptr, nullptr, state_.ptr); return; }
        const auto rows = h_.levels[l].A->nrows;
        scale_kernel<T><<<blocks(rows), kBlock, 0, stream_>>>(d.m.ptr, y(l), d.work.ptr, int(rows), state_.ptr);
        level_product<kSubtract>(l, d.work.ptr, y(l), nullptr, nullptr);
        d.tentative.apply_transpose(stream_, y(l), f(l + 1), state_.ptr);
    }
    void coarse(std::size_t l) {
        if (!h_.coarse_direct()) { smooth_zero(l); smooth_post(l); return; }
        const int rows = int(h_.levels[l].A->nrows);
        dense_kernel<T><<<blocks(std::size_t(rows) * 32), kBlock, 0, stream_>>>(coarse_.ptr, f(l), y(l), rows, state_.ptr);
    }
    // x += P e = T e - M A T e.
    void prolongation(std::size_t l) {
        auto& d = levels_[l];
        if (!d.tentative.rows) { d.p.template launch<kAdd>(stream_, y(l + 1), d.x.ptr, nullptr, nullptr, state_.ptr); return; }
        d.tentative.apply(stream_, y(l + 1), d.work.ptr, state_.ptr);
        level_product<kCorrect>(l, d.work.ptr, d.x.ptr, d.work.ptr, d.m.ptr);
    }
    void smooth_post(std::size_t l) { level_product<kSmooth>(l, levels_[l].x.ptr, y(l), f(l), levels_[l].m.ptr); }

private:
    void iteration() {
        const auto s = stream_;
        if constexpr (!std::is_same_v<T, double>)
            convert_kernel<double, T><<<blocks(n_), kBlock, 0, s>>>(r_.ptr, levels_[0].f.ptr, n_, state_.ptr);
        detail::amg_vcycle(*this, h_.levels.size());
        if constexpr (!std::is_same_v<T, double>)
            convert_kernel<T, double><<<blocks(n_), kBlock, 0, s>>>(levels_[0].y.ptr, z_.ptr, n_, state_.ptr);
        dot_kernel<<<kDotBlocks, kBlock, 0, s>>>(r_.ptr, z_.ptr, n_, partial_.ptr, state_.ptr);
        rho_kernel<<<1, kBlock, 0, s>>>(partial_.ptr, state_.ptr);
        direction_kernel<<<blocks(n_), kBlock, 0, s>>>(z_.ptr, p_.ptr, n_, state_.ptr);
        fine_.template apply<kSet>(s, p_.ptr, q_.ptr, nullptr, nullptr, state_.ptr);
        dot_kernel<<<kDotBlocks, kBlock, 0, s>>>(p_.ptr, q_.ptr, n_, partial_.ptr, state_.ptr);
        alpha_kernel<<<1, kBlock, 0, s>>>(partial_.ptr, state_.ptr);
        update_kernel<<<kDotBlocks, kBlock, 0, s>>>(p_.ptr, q_.ptr, u_.ptr, r_.ptr, n_, partial_.ptr, state_.ptr);
        check_kernel<<<1, kBlock, 0, s>>>(partial_.ptr, state_.ptr);
    }

    void capture() {
        cudaGraph_t graph = nullptr;
        throw_if_cuda_failed(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin AMG graph capture");
        iteration();
        throw_if_cuda_failed(cudaStreamEndCapture(stream_, &graph), "end AMG graph capture");
        const auto status = cudaGraphInstantiate(&graph_, graph, 0);
        cudaGraphDestroy(graph);
        throw_if_cuda_failed(status, "instantiate AMG graph");
    }

    const detail::AmgHierarchy& h_;
    int n_;
    FineOperator fine_;
    PCGDeviceBuffer<double> r_, u_, p_, q_, z_, partial_;
    PCGDeviceBuffer<CgState> state_;
    std::unique_ptr<PcgPinnedBuffer<CgState>> host_state_;
    std::vector<Level> levels_;
    PCGDeviceBuffer<T> coarse_;
    cudaStream_t stream_{nullptr};
    cudaGraphExec_t graph_{nullptr};
};
} // namespace vibestran::cuda_amg
