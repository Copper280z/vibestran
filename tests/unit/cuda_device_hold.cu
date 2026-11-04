// tests/unit/cuda_device_hold.cu
// Device memory held on purpose, to test plans under real memory pressure. Built
// with nvcc, the only compiler here that always finds the CUDA runtime's headers.
#include <cstddef>
#include <vector>

// Device memory until leave bytes are reported free, or the driver refuses more.
// Reported free memory is not all allocatable, so the last chunks can fail, which
// must not stay behind as the thread's last CUDA error.
void* cuda_test_hold_all_but(std::size_t leave) {
    constexpr std::size_t kChunk = std::size_t(16) * 1024 * 1024;
    auto* held = new std::vector<void*>;
    std::size_t free_bytes = 0, total_bytes = 0;
    while (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes > leave + kChunk) {
        void* chunk = nullptr;
        if (cudaMalloc(&chunk, kChunk) != cudaSuccess) {
            (void)cudaGetLastError();
            break;
        }
        held->push_back(chunk);
    }
    return held;
}

void cuda_test_release(void* held) {
    auto* chunks = static_cast<std::vector<void*>*>(held);
    for (void* chunk : *chunks) cudaFree(chunk);
    delete chunks;
}
