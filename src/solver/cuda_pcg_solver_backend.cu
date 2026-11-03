#define HAVE_CUDA 1

#include "solver/cuda_pcg_solver_backend.hpp"
#include "cuda_pcg_backend_impl.hpp"
#include <iomanip>
#include <sstream>
#include <utility>

namespace vibestran {

CudaPCGSolverBackend::CudaPCGSolverBackend(std::unique_ptr<CudaPCGContext> ctx) noexcept
    : ctx_(std::move(ctx)) {}

CudaPCGSolverBackend::~CudaPCGSolverBackend() {
    destroy_cuda_pcg_context(ctx_);
}

CudaPCGSolverBackend::CudaPCGSolverBackend(CudaPCGSolverBackend&&) noexcept = default;
CudaPCGSolverBackend& CudaPCGSolverBackend::operator=(CudaPCGSolverBackend&&) noexcept = default;

std::optional<CudaPCGSolverBackend>
CudaPCGSolverBackend::try_create(double tolerance, int max_iters, std::size_t device_memory_budget) noexcept {
    auto ctx = try_create_cuda_pcg_context(
        tolerance > 0.0 ? tolerance : 1e-8, max_iters);
    if (!ctx)
        return std::nullopt;
    ctx->device_memory_budget = pcg_device_budget(device_memory_budget);
    return CudaPCGSolverBackend(std::move(ctx));
}

std::string_view CudaPCGSolverBackend::name() const noexcept {
    return "CUDA PCG float64 (GPU)";
}

int CudaPCGSolverBackend::last_iteration_count() const noexcept {
    return ctx_->last_iters;
}

double CudaPCGSolverBackend::last_relative_residual() const noexcept {
    return ctx_->last_rel_res;
}

double CudaPCGSolverBackend::last_estimated_error() const noexcept {
    return ctx_->last_refinement ? ctx_->last_refinement->estimated_error : ctx_->last_rel_res;
}

std::optional<RefinementReport> CudaPCGSolverBackend::last_refinement() const noexcept {
    return ctx_->last_refinement;
}

std::string_view CudaPCGSolverBackend::device_name() const noexcept {
    return ctx_->device_name;
}

std::vector<double>
CudaPCGSolverBackend::solve(const SparseMatrixBuilder::CsrData& K,
                            const std::vector<double>& F,
                            const FactorRatioCheckPolicy*) {
    auto u = solve_cuda_pcg_stable(*ctx_, K, F);
    notices().keep(std::exchange(ctx_->notices, {}));
    return u;
}

Result<std::unique_ptr<LinearFactor>>
CudaPCGSolverBackend::factor(const SparseMatrixBuilder::CsrData& K, const FactorOptions& options) {
    if (K.n < 50000) return SolverBackend::factor(K, options);
    try {
        return make_cuda_amg_factor(*ctx_, K, false, "cuda-pcg AMG");
    } catch (const std::exception& e) {
        return Error{std::format("{}: {}", name(), e.what())};
    }
}

std::vector<std::pair<std::string, std::string>> CudaPCGSolverBackend::details() const {
    return {{"Device", ctx_->device_name},
            {"Method", "conjugate gradient; smoothed-aggregation AMG for large systems, IC0/ILU0 for small resident systems"},
            {"Tolerance", [this] {
                 std::ostringstream text;
                 text << std::scientific << std::setprecision(1) << ctx_->tolerance
                      << " (estimated relative error)";
                 return text.str();
             }()},
            {"Precision", "fp64"},
            {"Iterative refinement", "always"}};
}

} // namespace vibestran
