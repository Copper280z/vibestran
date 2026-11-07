#define HAVE_CUDA 1

#include "solver/cuda_mixed_pcg_solver_backend.hpp"
#include "cuda_pcg_backend_impl.hpp"
#include <iomanip>
#include <sstream>
#include <utility>

namespace vibestran {

CudaMixedPCGSolverBackend::CudaMixedPCGSolverBackend(
    std::unique_ptr<CudaPCGContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaMixedPCGSolverBackend::~CudaMixedPCGSolverBackend() {
    destroy_cuda_pcg_context(ctx_);
}

CudaMixedPCGSolverBackend::CudaMixedPCGSolverBackend(
    CudaMixedPCGSolverBackend&&) noexcept = default;
CudaMixedPCGSolverBackend& CudaMixedPCGSolverBackend::operator=(
    CudaMixedPCGSolverBackend&&) noexcept = default;

std::optional<CudaMixedPCGSolverBackend>
CudaMixedPCGSolverBackend::try_create(double tolerance, int max_iters, std::size_t device_memory_budget) noexcept {
    auto ctx = try_create_cuda_pcg_context(
        tolerance > 0.0 ? tolerance : 1e-6, max_iters);
    if (!ctx)
        return std::nullopt;
    ctx->device_memory_budget = pcg_device_budget(device_memory_budget);
    return CudaMixedPCGSolverBackend(std::move(ctx));
}

std::vector<double>
CudaMixedPCGSolverBackend::solve(const SparseMatrixBuilder::CsrData& K,
                                 const std::vector<double>& F,
                                 const FactorRatioCheckPolicy*) {
    ctx_->near_nullspace = near_nullspace();
    auto u = solve_cuda_pcg_mixed(*ctx_, K, F);
    notices().keep(std::exchange(ctx_->notices, {}));
    for (auto& fatal : std::exchange(ctx_->fatals, {}))
        notices().add_fatal(std::move(fatal));
    return u;
}

Result<std::unique_ptr<LinearFactor>>
CudaMixedPCGSolverBackend::factor(const SparseMatrixBuilder::CsrData& K, const FactorOptions& options) {
    if (K.n < 50000) return SolverBackend::factor(K, options);
    try {
        return make_cuda_amg_factor(*ctx_, K, true, "cuda-pcg mixed AMG", near_nullspace());
    } catch (const std::exception& e) {
        return Error{std::format("{}: {}", name(), e.what())};
    }
}

std::vector<std::pair<std::string, std::string>> CudaMixedPCGSolverBackend::details() const {
    return {{"Device", ctx_->device_name},
            {"Method", "conjugate gradient; smoothed-aggregation AMG for large systems, IC0/ILU0 for small resident systems"},
            {"Tolerance", [this] {
                 std::ostringstream text;
                 text << std::scientific << std::setprecision(1) << ctx_->tolerance
                      << " (estimated relative error)";
                 return text.str();
             }()},
            {"Precision", "fp32 AMG with fp64 Krylov operator; small IC0 solves in fp32; spill AMG in fp64; compensated refinement"},
            {"Iterative refinement", "always"}};
}

std::string_view CudaMixedPCGSolverBackend::name() const noexcept {
    return "CUDA mixed-precision PCG + iterative refinement (GPU)";
}

int CudaMixedPCGSolverBackend::last_iteration_count() const noexcept {
    return ctx_->last_iters;
}

double CudaMixedPCGSolverBackend::last_estimated_error() const noexcept {
    return ctx_->last_refinement ? ctx_->last_refinement->estimated_error : ctx_->last_rel_res;
}

std::optional<RefinementReport> CudaMixedPCGSolverBackend::last_refinement() const noexcept {
    return ctx_->last_refinement;
}

std::optional<AmgDeviceReport> CudaMixedPCGSolverBackend::last_amg_device_report() const noexcept {
    return ctx_->last_amg_device;
}

std::string_view CudaMixedPCGSolverBackend::device_name() const noexcept {
    return ctx_->device_name;
}

} // namespace vibestran
