// src/solver/vulkan_solver_backend.cpp
#ifdef HAVE_VULKAN

#include "solver/vulkan_solver_backend.hpp"
#include "solver/vulkan_context.hpp"
#include "solver/solver_backend.hpp"
#include "solver/pcg_memory.hpp"
#include "vulkan_pcg_internal.hpp"
#include "amg_result.hpp"
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
    : ctx_(std::move(ctx)), cfg_(cfg) {
    cfg_.device_memory_budget = pcg_device_budget(cfg_.device_memory_budget);
}

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
    return "Vulkan PCG";
}

std::vector<std::pair<std::string, std::string>> VulkanSolverBackend::details() const {
    return {{"Device", ctx_.device_info().device_name},
            {"Method", std::format("conjugate gradient; AMG for large fp64 systems, Jacobi otherwise (CPU direct "
                                   "below {} equations)", cfg_.min_dofs_for_gpu)},
            {"Tolerance", std::format("{:.1e} (estimated relative error)", cfg_.tolerance)},
            {"Precision", cfg_.use_double ? "fp64"
                                          : "fp64 AMG when supported; otherwise fp32 Jacobi with fp64 refinement/retry"},
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
    return matrix + vecs + 2 * parts;
}

// Returns estimated bytes needed for a float64 full-GPU solve.
static size_t vram_needed_f64(const SparseMatrixBuilder::CsrData& K) noexcept {
    size_t matrix = static_cast<size_t>(K.nnz)   * sizeof(double)
                  + static_cast<size_t>(K.nnz)   * sizeof(int)
                  + static_cast<size_t>(K.n + 1) * sizeof(int);
    size_t vecs   = 6ULL * static_cast<size_t>(K.n) * sizeof(double);
    size_t parts  = static_cast<size_t>((static_cast<uint32_t>(K.n) + 255u) / 256u) * sizeof(double);
    return matrix + vecs + 2 * parts;
}

bool VulkanSolverBackend::fits_in_vram(const SparseMatrixBuilder::CsrData& K, std::size_t available) const noexcept {
    const auto scalar_bytes = cfg_.use_double ? sizeof(double) : sizeof(float);
    const auto limit = ctx_.device_info().max_storage_buffer_range;
    if (std::size_t(K.nnz) * std::max(sizeof(int), scalar_bytes) > limit ||
        std::size_t(K.n) * scalar_bytes > limit || (std::size_t(K.n) + 1) * sizeof(int) > limit ||
        (std::size_t(K.n) + 255) / 256 > ctx_.device_info().max_workgroup_count_x) return false;
    size_t needed    = cfg_.use_double ? vram_needed_f64(K) : vram_needed_f32(K);
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

Status VulkanSolverBackend::accept_amg(const detail::AmgPcgResult& result) {
    last_iters_ = result.iterations;
    last_residual_ = result.refinement.relative_residual;
    last_refinement_ = result.refinement;
    if (result.refinement.converged) return std::nullopt;
    return Error{std::format("Vulkan AMG PCG: estimated error {:.3e} exceeds {:.3e} after {} iterations and {} corrections (n={})",
        result.refinement.estimated_error, cfg_.tolerance, result.iterations, result.refinement.corrections,
        result.solution.size())};
}

std::size_t VulkanSolverBackend::device_budget() const {
    auto available = static_cast<std::size_t>(
        static_cast<double>(ctx_.available_memory()) * (1.0 - cfg_.vram_headroom));
    if (cfg_.device_memory_budget) available = std::min(available, cfg_.device_memory_budget);
    return available;
}

Result<std::unique_ptr<LinearFactor>>
VulkanSolverBackend::factor(const SparseMatrixBuilder::CsrData& K, const FactorOptions& options) {
    if (K.n < 50000 || K.n < cfg_.min_dofs_for_gpu || !ctx_.device_info().supports_float64)
        return SolverBackend::factor(K, options);
    try {
        Pipelines* pl = ensure_pipelines(ctx_, pipelines_);
        return make_vulkan_amg_factor(ctx_, *pl, K, cfg_.tolerance, cfg_.max_iterations,
            [this] { return device_budget(); }, last_full_gpu_,
            [this](const detail::AmgPcgResult& result) { return accept_amg(result); });
    } catch (const std::exception& e) {
        return Error{std::format("{}: {}", name(), e.what())};
    }
}

// cppcheck-suppress unusedFunction -- called through SolverBackend from linear_static.cpp
bool VulkanSolverBackend::uses_near_nullspace(int n) const noexcept {
    return n >= 50000 && ctx_.device_info().supports_float64;
}

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
    const size_t available = device_budget();

    // A multilevel preconditioner removes the low-frequency error that Jacobi
    // leaves behind on refined meshes. Leave conservative hierarchy headroom.
    if (n >= 50000 && ctx_.device_info().supports_float64) {
        return solve_vulkan_amg(ctx_, *pl, K, F, cfg_.tolerance, cfg_.max_iterations, available, last_full_gpu_,
            [this](const detail::AmgPcgResult& result) { return accept_amg(result); }, near_nullspace());
    }

    enum class Path { Float32, Float64, Partitioned32, Partitioned64 };
    Path path;
    if (!cfg_.force_tiled && fits_in_vram(K, available)) {
        path = cfg_.use_double ? Path::Float64 : Path::Float32;
        spdlog::debug("[vulkan] {} DOFs → full-GPU {} ({} / {} VRAM)", n,
                      cfg_.use_double ? "float64" : "float32",
                      mib_str(cfg_.use_double ? vram_needed_f64(K) : vram_needed_f32(K)),
                      mib_str(vram_total));
    } else {
        path = cfg_.use_double ? Path::Partitioned64 : Path::Partitioned32;
        spdlog::debug("[vulkan] {} DOFs → partitioned PCG (matrix {} exceeds VRAM budget {})",
                      n, mib_str(vram_needed_f32(K)), mib_str(available));
    }
    last_full_gpu_ = path == Path::Float32 || path == Path::Float64;

    // Each inner PCG solve stops at inner_tolerance (or its precision floor)
    // and returns its iterate; refinement with compensated double-precision
    // residuals takes the solution to `tolerance`.  The right-hand side is
    // scaled to unit max-norm so fp32 keeps the range of tiny residuals.
    VulkanSolverConfig inner = cfg_;
    inner.tolerance = cfg_.inner_tolerance;
    inner.throw_on_no_convergence = false;
    int total_iters = 0;
    const auto refine_on = [&](Path p, RefinementReport& rep) {
        auto workspace = (p == Path::Partitioned32 || p == Path::Partitioned64)
            ? make_partitioned_pcg(ctx_, *pl, K, p == Path::Partitioned64, available)
            : make_full_pcg(ctx_, *pl, K, p == Path::Float64);
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
            std::vector<double> x = workspace->solve(b, inner, iters, res);
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
    if (!rep.converged && (path == Path::Float32 || path == Path::Partitioned32) &&
        ctx_.device_info().supports_float64) {
        notices().add(std::format("Vulkan fp32 PCG could not reach the target accuracy "
                                  "(estimated relative error {:.2e}); solved again in fp64",
                                  rep.estimated_error));
        const auto limit = ctx_.device_info().max_storage_buffer_range;
        const bool fits = vram_needed_f64(K) <= available &&
            std::size_t(K.nnz) * sizeof(double) <= limit && std::size_t(K.n) * sizeof(double) <= limit &&
            (std::size_t(K.n) + 255) / 256 <= ctx_.device_info().max_workgroup_count_x;
        path = fits && !cfg_.force_tiled ? Path::Float64 : Path::Partitioned64;
        last_full_gpu_ = path == Path::Float64;
        u = refine_on(path, rep);
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
            (path == Path::Float64 || path == Path::Partitioned64) ? "" : " in float32"));
    spdlog::info("[vulkan] converged: {} PCG iterations, {} refinement correction(s), "
                 "estimated relative error {:.2e}, solve = {:.3f} ms",
                 last_iters_, rep.corrections, rep.estimated_error, ms_since(t_solve));
    return u;
}

} // namespace vibestran

#endif // HAVE_VULKAN
