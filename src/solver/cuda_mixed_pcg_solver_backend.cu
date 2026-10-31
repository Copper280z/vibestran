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
CudaMixedPCGSolverBackend::try_create(double tolerance, int max_iters) noexcept {
    auto ctx = try_create_cuda_pcg_context(
        tolerance > 0.0 ? tolerance : 1e-6, max_iters);
    if (!ctx)
        return std::nullopt;
    return CudaMixedPCGSolverBackend(std::move(ctx));
}

std::vector<double>
CudaMixedPCGSolverBackend::solve(const SparseMatrixBuilder::CsrData& K,
                                 const std::vector<double>& F,
                                 const FactorRatioCheckPolicy*) {
    auto u = solve_cuda_pcg_mixed(*ctx_, K, F);
    notices().keep(std::exchange(ctx_->notices, {}));
    return u;
}

std::vector<std::pair<std::string, std::string>> CudaMixedPCGSolverBackend::details() const {
    return {{"Device", ctx_->device_name},
            {"Method", "conjugate gradient, IC0 preconditioner (ILU0, then Jacobi, when IC0 fails)"},
            {"Tolerance", [this] {
                 std::ostringstream text;
                 text << std::scientific << std::setprecision(1) << ctx_->tolerance
                      << " (estimated relative error)";
                 return text.str();
             }()},
            {"Precision", "fp32 inner solves, refined in fp64"},
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

std::string_view CudaMixedPCGSolverBackend::device_name() const noexcept {
    return ctx_->device_name;
}

} // namespace vibestran
