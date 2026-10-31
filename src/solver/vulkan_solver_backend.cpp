// src/solver/vulkan_solver_backend.cpp
#ifdef HAVE_VULKAN

#include "solver/vulkan_solver_backend.hpp"
#include "solver/vulkan_context.hpp"
#include "solver/solver_backend.hpp"
#include "vulkan_pcg_internal.hpp"
#include "vulkan_pipelines.hpp"
#include "core/exceptions.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>

namespace vibestran {

// ── Constructor / destructor ──────────────────────────────────────────────────

VulkanSolverBackend::VulkanSolverBackend(VulkanContext ctx, const VulkanSolverConfig& cfg)
    : ctx_(std::move(ctx)), cfg_(cfg) {}

VulkanSolverBackend::~VulkanSolverBackend() = default;

VulkanSolverBackend::VulkanSolverBackend(VulkanSolverBackend&&) noexcept = default;
VulkanSolverBackend& VulkanSolverBackend::operator=(VulkanSolverBackend&&) noexcept = default;

// ── Factory ───────────────────────────────────────────────────────────────────

std::optional<VulkanSolverBackend>
VulkanSolverBackend::try_create(const VulkanSolverConfig& cfg) {
    auto ctx = VulkanContext::create();
    if (!ctx) return std::nullopt;
    return VulkanSolverBackend(std::move(*ctx), cfg);
}

// ── Accessors ─────────────────────────────────────────────────────────────────

std::string_view VulkanSolverBackend::name() const noexcept {
    return "Vulkan PCG (Jacobi preconditioner)";
}

std::vector<std::pair<std::string, std::string>> VulkanSolverBackend::details() const {
    return {{"Device", ctx_.device_info().device_name},
            {"Method", std::format("conjugate gradient, Jacobi preconditioner (CPU direct "
                                   "below {} equations)", cfg_.min_dofs_for_gpu)},
            {"Tolerance", std::format("{:.1e} (estimated relative error)", cfg_.tolerance)},
            {"Precision", cfg_.use_double ? "fp64"
                                          : "fp32 inner solves, refined in fp64 (fp64 "
                                            "retry when fp32 cannot converge)"},
            {"Iterative refinement", "always"}};
}

bool   VulkanSolverBackend::last_solve_was_full_gpu() const noexcept { return last_full_gpu_; }
int    VulkanSolverBackend::last_iteration_count()    const noexcept { return last_iters_; }
double VulkanSolverBackend::last_residual_norm()      const noexcept { return last_residual_; }
double VulkanSolverBackend::last_estimated_error()    const noexcept {
    return last_refinement_ ? last_refinement_->estimated_error : last_residual_;
}

// ── VRAM checks ───────────────────────────────────────────────────────────────

// Returns estimated bytes needed for a float32 full-GPU solve.
static size_t vram_needed_f32(const SparseMatrixBuilder::CsrData& K) noexcept {
    size_t matrix = static_cast<size_t>(K.nnz)   * sizeof(float)
                  + static_cast<size_t>(K.nnz)   * sizeof(int)
                  + static_cast<size_t>(K.n + 1) * sizeof(int);
    size_t vecs   = 6ULL * static_cast<size_t>(K.n) * sizeof(float); // x,r,z,p,Ap,diag
    size_t parts  = static_cast<size_t>((static_cast<uint32_t>(K.n) + 255u) / 256u) * sizeof(float);
    return matrix + vecs + parts;
}

// Returns estimated bytes needed for a float64 full-GPU solve.
static size_t vram_needed_f64(const SparseMatrixBuilder::CsrData& K) noexcept {
    size_t matrix = static_cast<size_t>(K.nnz)   * sizeof(double)
                  + static_cast<size_t>(K.nnz)   * sizeof(int)
                  + static_cast<size_t>(K.n + 1) * sizeof(int);
    size_t vecs   = 6ULL * static_cast<size_t>(K.n) * sizeof(double);
    size_t parts  = static_cast<size_t>((static_cast<uint32_t>(K.n) + 255u) / 256u) * sizeof(double);
    return matrix + vecs + parts;
}

bool VulkanSolverBackend::fits_in_vram(const SparseMatrixBuilder::CsrData& K) const noexcept {
    size_t needed    = cfg_.use_double ? vram_needed_f64(K) : vram_needed_f32(K);
    size_t available = static_cast<size_t>(
        static_cast<double>(ctx_.device_info().vram_bytes) * (1.0 - cfg_.vram_headroom));
    return needed <= available;
}

// ── Logging helpers ───────────────────────────────────────────────────────────

static std::string mib_str(size_t bytes) {
    return std::format("{} MiB", bytes / (1024 * 1024));
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// ── Lazy pipeline init ─────────────────────────────────────────────────────────

static Pipelines* ensure_pipelines(const VulkanContext& ctx,
                                    std::unique_ptr<Pipelines>& cache) {
    if (cache) return cache.get();
    // build_pipelines throws SolverError on failure
    cache.reset(build_pipelines(ctx));
    return cache.get();
}

// ── solve ─────────────────────────────────────────────────────────────────────

std::vector<double>
VulkanSolverBackend::solve(const SparseMatrixBuilder::CsrData& K,
                            const std::vector<double>& F,
                            const FactorRatioCheckPolicy* factor_ratio_policy) {
    const int n = K.n;
    if (n == 0)
        throw SolverError("Vulkan solver: stiffness matrix is empty — no free DOFs");
    if (static_cast<int>(F.size()) != n)
        throw SolverError(std::format(
            "Vulkan solver: force vector size {} != matrix size {}", F.size(), n));

    auto t_diag = std::chrono::steady_clock::now();

    // Check for missing diagonal (structural singularity / missing BCs).
    // The Jacobi preconditioner would divide by zero otherwise.
    for (int row = 0; row < n; ++row) {
        bool found = false;
        for (int idx = K.row_ptr[row]; idx < K.row_ptr[row + 1]; ++idx)
            if (K.col_ind[idx] == row) { found = true; break; }
        if (!found)
            throw SolverError(std::format(
                "Vulkan solver: zero diagonal at row {} — "
                "stiffness matrix appears singular. Check boundary conditions.", row));
    }
    spdlog::debug("[vulkan] diagonal scan: {:.3f} ms", ms_since(t_diag));

    // ── Path selection ────────────────────────────────────────────────────

    // Small problems: GPU kernel launch and PCIe readback overhead per iteration
    // dominates actual compute time. Fall back to the CPU direct solver.
    last_refinement_.reset();
    if (n < cfg_.min_dofs_for_gpu) {
        spdlog::debug("[vulkan] {} DOFs → CPU fallback (below min_dofs_for_gpu={})",
                      n, cfg_.min_dofs_for_gpu);
        last_full_gpu_ = false;
        last_iters_    = 1;
        notices().add(std::format("Vulkan backend: {} equations is below the GPU "
                                  "threshold of {}; solved with the CPU direct solver",
                                  n, cfg_.min_dofs_for_gpu));
        EigenSolverBackend eigen;
        eigen.set_refinement(refine_direct_);
        auto u = eigen.solve(K, F, factor_ratio_policy);
        notices().keep(eigen.notices().take());
        last_refinement_ = eigen.last_refinement();
        last_residual_   = last_refinement_ ? last_refinement_->relative_residual : 0.0;
        return u;
    }

    // use_double requires shaderFloat64 device feature
    if (cfg_.use_double && !ctx_.device_info().supports_float64)
        throw SolverError(std::format(
            "Vulkan solver: use_double=true but GPU '{}' does not support shaderFloat64. "
            "Use --cpu for double-precision solve.",
            ctx_.device_info().device_name));

    auto t_pipelines = std::chrono::steady_clock::now();
    Pipelines* pl = ensure_pipelines(ctx_, pipelines_);
    spdlog::debug("[vulkan] pipeline init: {:.3f} ms", ms_since(t_pipelines));

    const size_t vram_total = ctx_.device_info().vram_bytes;
    const size_t available  = static_cast<size_t>(
        static_cast<double>(vram_total) * (1.0 - cfg_.vram_headroom));

    enum class Path { Float32, Float64, Tiled };
    Path path;
    if (!cfg_.force_tiled && fits_in_vram(K)) {
        path = cfg_.use_double ? Path::Float64 : Path::Float32;
        spdlog::debug("[vulkan] {} DOFs → full-GPU {} ({} / {} VRAM)", n,
                      cfg_.use_double ? "float64" : "float32",
                      mib_str(cfg_.use_double ? vram_needed_f64(K) : vram_needed_f32(K)),
                      mib_str(vram_total));
    } else {
        if (cfg_.use_double)
            throw SolverError(std::format(
                "Vulkan solver: use_double=true but K ({}) exceeds available VRAM ({}) "
                "for a full-GPU float64 solve. Use --cpu.",
                mib_str(vram_needed_f64(K)), mib_str(available)));
        path = Path::Tiled;
        spdlog::debug("[vulkan] {} DOFs → tiled float32 (matrix {} exceeds VRAM budget {})",
                      n, mib_str(vram_needed_f32(K)), mib_str(available));
    }
    last_full_gpu_ = path != Path::Tiled;

    // Each inner PCG solve stops at inner_tolerance (or its precision floor)
    // and returns its iterate; refinement with compensated double-precision
    // residuals takes the solution to `tolerance`.  The right-hand side is
    // scaled to unit max-norm so fp32 keeps the range of tiny residuals.
    VulkanSolverConfig inner = cfg_;
    inner.tolerance = cfg_.inner_tolerance;
    inner.throw_on_no_convergence = false;
    int total_iters = 0;
    const auto refine_on = [&](Path p, RefinementReport& rep) {
        const auto pcg = [&](const std::vector<double>& rhs) {
            double scale = std::accumulate(rhs.begin(), rhs.end(), 0.0,
                [](double m, double v) { return std::max(m, std::abs(v)); });
            if (scale == 0.0 || !std::isfinite(scale))
                scale = 1.0;
            std::vector<double> b(rhs.size());
            std::transform(rhs.begin(), rhs.end(), b.begin(),
                           [scale](double v) { return v / scale; });
            int iters = 0;
            double res = 0.0;
            std::vector<double> x =
                p == Path::Float64 ? solve_full_gpu_double(ctx_, *pl, K, b, inner, iters, res)
              : p == Path::Float32 ? solve_full_gpu(ctx_, *pl, K, b, inner, iters, res)
                                   : solve_tiled(ctx_, *pl, K, b, inner,
                                                 static_cast<uint64_t>(available), iters, res);
            total_iters += iters;
            std::transform(x.begin(), x.end(), x.begin(),
                           [scale](double v) { return v * scale; });
            return x;
        };
        std::vector<double> u = pcg(F);
        RefinementOptions opts;
        opts.tolerance = cfg_.tolerance;
        opts.max_corrections = 20;
        rep = refine_solution(K, F, u, pcg, opts);
        spdlog::debug("[vulkan] iterative refinement: {} correction(s), estimated relative "
                      "error {:.2e} -> {:.2e}, backward error {:.2e}{}",
                      rep.corrections, rep.initial_error, rep.estimated_error,
                      rep.backward_error, rep.converged ? "" : " (not converged)");
        return u;
    };

    auto t_solve = std::chrono::steady_clock::now();
    RefinementReport rep;
    std::vector<double> u = refine_on(path, rep);
    // float32 too imprecise for this K (cond(K) beyond about 1e7 with Jacobi
    // PCG): retry in float64 when the device supports it and K fits.
    if (!rep.converged && path == Path::Float32 && ctx_.device_info().supports_float64 &&
        vram_needed_f64(K) <= available) {
        notices().add(std::format("Vulkan fp32 PCG could not reach the target accuracy "
                                  "(estimated relative error {:.2e}); solved again in fp64",
                                  rep.estimated_error));
        u = refine_on(Path::Float64, rep);
    }
    last_iters_      = total_iters;
    last_residual_   = rep.relative_residual;
    last_refinement_ = rep;
    if (!rep.converged)
        throw SolverError(std::format(
            "Vulkan PCG could not reach the target relative error {:.1e} (estimated {:.2e} "
            "after {} refinement correction(s), {} PCG iterations; n = {}): the stiffness "
            "matrix is too ill-conditioned for Jacobi PCG{}, or singular (check SPCs). "
            "Use --backend=cpu for a direct solve.",
            cfg_.tolerance, rep.estimated_error, rep.corrections, total_iters, n,
            path == Path::Float64 ? "" : " in float32"));
    spdlog::info("[vulkan] converged: {} PCG iterations, {} refinement correction(s), "
                 "estimated relative error {:.2e}, solve = {:.3f} ms",
                 last_iters_, rep.corrections, rep.estimated_error, ms_since(t_solve));
    return u;
}

} // namespace vibestran

#endif // HAVE_VULKAN
