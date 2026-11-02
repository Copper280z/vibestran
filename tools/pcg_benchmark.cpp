// Repeatable linear-solver benchmark. Timings include setup and refinement,
// exclude input construction and independent solution verification.
#include "core/sparse_matrix.hpp"
#include "solver/solver_backend.hpp"
#include "solver/iterative_refinement.hpp"
#include "solver/heat_transfer_steady.hpp"
#include "io/bdf_parser.hpp"
#include "io/system_dump.hpp"
#include "magnetic/model.hpp"
#include "magnetic/solver.hpp"
#ifdef HAVE_VULKAN
#include "solver/vulkan_solver_backend.hpp"
#include "solver/vulkan_context.hpp"
#endif
#ifdef HAVE_CUDA
#include "solver/cuda_pcg_solver_backend.hpp"
#include "solver/cuda_mixed_pcg_solver_backend.hpp"
#endif
#include <spdlog/spdlog.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace vibestran;
using Csr = SparseMatrixBuilder::CsrData;

namespace {
struct Options {
    std::string backend{"cpu-pcg"}, matrix, rhs, reference, capture_deck, capture_prefix;
    int side{24}, repeats{3}, warmups{1};
    double contrast{1.0};
    double budget_mib{0};
    bool tiled{false}, fp64{false}, profile{false};
};

template<class T> bool number(std::string_view s, T& out) {
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return ec == std::errc{} && end == s.data() + s.size();
}

std::optional<Options> parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        if (a == "--tiled") { o.tiled = true; continue; }
        if (a == "--fp64") { o.fp64 = true; continue; }
        if (a == "--profile") { o.profile = true; continue; }
        if (++i == argc) return {};
        const std::string_view v(argv[i]);
        if (a == "--backend") o.backend = v;
        else if (a == "--matrix") o.matrix = v;
        else if (a == "--rhs") o.rhs = v;
        else if (a == "--reference") o.reference = v;
        else if (a == "--capture-deck") o.capture_deck = v;
        else if (a == "--capture-prefix") o.capture_prefix = v;
        else if (a == "--side") { if (!number(v, o.side)) return {}; }
        else if (a == "--repeats") { if (!number(v, o.repeats)) return {}; }
        else if (a == "--warmups") { if (!number(v, o.warmups)) return {}; }
        else if (a == "--contrast") { if (!number(v, o.contrast)) return {}; }
        else if (a == "--budget-mib") { if (!number(v, o.budget_mib)) return {}; }
        else return {};
    }
    if (o.side < 2 || o.side > 400 || o.repeats < 1 || o.warmups < 0 ||
        !std::isfinite(o.contrast) || o.contrast < 1.0 ||
        o.rhs.empty() != o.reference.empty() ||
        o.capture_deck.empty() != o.capture_prefix.empty() ||
        !std::isfinite(o.budget_mib) || o.budget_mib < 0 ||
        o.budget_mib >= double(std::numeric_limits<std::size_t>::max() / 1048576)) return {};
    return o;
}

// Conservative 3D diffusion discretization with homogeneous Dirichlet outer
// boundary and a material interface. Harmonic face conductivities preserve
// symmetry and flux continuity. Store the lower triangle, as FE assembly does.
Csr diffusion(int side, double contrast) {
    Csr a;
    a.n = side * side * side;
    a.symmetry = SymmetryStorage::Lower;
    a.row_ptr.reserve(a.n + 1);
    a.col_ind.reserve(static_cast<std::size_t>(a.n) * 4);
    a.values.reserve(static_cast<std::size_t>(a.n) * 4);
    auto conductivity = [=](int x) { return x < side / 2 ? 1.0 : contrast; };
    for (int z = 0; z < side; ++z) for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
            const int row = x + side * (y + side * z);
            a.row_ptr.push_back(static_cast<int>(a.values.size()));
            double diag = 0;
            for (int axis = 0; axis < 3; ++axis) for (int sign : {-1, 1}) {
                const int xx = x + (axis == 0 ? sign : 0);
                const int yy = y + (axis == 1 ? sign : 0);
                const int zz = z + (axis == 2 ? sign : 0);
                const double k = conductivity(x);
                const bool inside = xx >= 0 && xx < side && yy >= 0 && yy < side && zz >= 0 && zz < side;
                const double other = inside ? conductivity(xx) : k;
                const double face = 2.0 * k / (1.0 + k / other);
                diag += face;
                const int col = xx + side * (yy + side * zz);
                if (inside && col < row) {
                    a.col_ind.push_back(col);
                    a.values.push_back(-face);
                }
            }
            // The three preceding neighbours were visited in x,y,z order.
            const auto first = a.row_ptr.back();
            std::reverse(a.col_ind.begin() + first, a.col_ind.end());
            std::reverse(a.values.begin() + first, a.values.end());
            a.col_ind.push_back(row);
            a.values.push_back(diag);
        }
    a.nnz = static_cast<int>(a.values.size());
    a.row_ptr.push_back(a.nnz);
    return a;
}

std::optional<Csr> read_matrix(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line) || line != "%%MatrixMarket matrix coordinate real symmetric") {
        std::cerr << "Expected a real symmetric coordinate Matrix Market file: " << path << '\n';
        return {};
    }
    while (std::getline(in, line) && (line.empty() || line[0] == '%')) {}
    std::istringstream dims(line);
    int n = 0, cols = 0;
    std::size_t nnz = 0;
    if (!(dims >> n >> cols >> nnz) || n <= 0 || n != cols || nnz > INT32_MAX) return {};
    SparseMatrixBuilder builder(n, nnz);
    for (std::size_t k = 0; k < nnz; ++k) {
        int row, col; double value;
        if (!(in >> row >> col >> value) || row < 1 || col < 1 || row > n || col > n ||
            !std::isfinite(value)) return {};
        if (row < col) std::swap(row, col);
        builder.add(row - 1, col - 1, value);
    }
    return builder.build_csr();
}

std::optional<std::vector<double>> read_vector(const std::string& path, int n) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line) || line != "%%MatrixMarket matrix array real general") return {};
    while (std::getline(in, line) && (line.empty() || line[0] == '%')) {}
    std::istringstream dims(line);
    int rows = 0, cols = 0;
    if (!(dims >> rows >> cols) || rows != n || cols != 1) return {};
    std::vector<double> values(n);
    for (auto& x : values) if (!(in >> x) || !std::isfinite(x)) return {};
    return values;
}

// Capture the actual reduced thermal system or each magnetic Newton/response
// system through the normal assembly path, without changing production output.
class CaptureBackend final : public SolverBackend {
public:
    explicit CaptureBackend(std::string prefix) : prefix_(std::move(prefix)) {}
    std::string_view name() const noexcept override { return "CPU direct benchmark capture"; }
    std::vector<double> solve(const Csr& k, const std::vector<double>& f,
                             const FactorRatioCheckPolicy* policy) override {
        const auto start = std::chrono::steady_clock::now();
        auto u = cpu_.solve(k, f, policy);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto path = prefix_ + "_" + std::to_string(++count_);
        write_matrix_market(path + "_K.mtx", k.lower_triangle());
        write_matrix_market(path + "_F.mtx", f);
        write_matrix_market(path + "_u.mtx", u);
        std::cout << path << ',' << k.n << ',' << k.nnz << ',' << seconds << std::endl;
        return u;
    }
private:
    EigenSolverBackend cpu_;
    std::string prefix_;
    int count_{0};
};

int capture(const Options& o) {
    auto magnetic_deck = magnetic::is_magnetic_deck(o.capture_deck);
    if (const auto* error = std::get_if<Error>(&magnetic_deck)) {
        std::cerr << error->message << '\n'; return 1;
    }
    auto recorder = std::make_unique<CaptureBackend>(o.capture_prefix);
    std::cout << "system,n,nnz,direct_seconds\n";
    if (std::get<bool>(magnetic_deck)) {
        auto parsed = magnetic::parse_file(o.capture_deck);
        if (const auto* error = std::get_if<Error>(&parsed)) {
            std::cerr << error->message << '\n'; return 1;
        }
        auto results = magnetic::solve(std::get<magnetic::Model>(parsed), *recorder);
        if (const auto* error = std::get_if<Error>(&results)) {
            std::cerr << error->message << '\n'; return 1;
        }
    } else {
        const auto model = BdfParser::parse_file(o.capture_deck);
        HeatTransferSteadySolver solver(std::move(recorder));
        const auto results = solver.solve(model);
        if (results.subcases.empty()) return 1;
    }
    return 0;
}

std::unique_ptr<SolverBackend> backend(const Options& o) {
    if (o.backend == "cpu") return std::make_unique<EigenSolverBackend>();
    if (o.backend == "cpu-pcg") return std::make_unique<EigenPCGSolverBackend>();
#ifdef HAVE_VULKAN
    if (o.backend == "vulkan") {
        VulkanSolverConfig cfg;
        cfg.min_dofs_for_gpu = 0;
        cfg.force_tiled = o.tiled;
        cfg.use_double = o.fp64;
        cfg.device_memory_budget = static_cast<std::size_t>(o.budget_mib * 1048576);
        auto b = VulkanSolverBackend::try_create(cfg);
        if (b) return std::make_unique<VulkanSolverBackend>(std::move(*b));
    }
#endif
#ifdef HAVE_CUDA
    if (o.backend == "cuda-pcg") {
        auto b = CudaPCGSolverBackend::try_create(1e-8, 0, static_cast<std::size_t>(o.budget_mib * 1048576));
        if (b) return std::make_unique<CudaPCGSolverBackend>(std::move(*b));
    }
    if (o.backend == "cuda-pcg-mixed") {
        auto b = CudaMixedPCGSolverBackend::try_create(1e-8, 0, static_cast<std::size_t>(o.budget_mib * 1048576));
        if (b) return std::make_unique<CudaMixedPCGSolverBackend>(std::move(*b));
    }
#endif
    return {};
}
} // namespace

int main(int argc, char** argv) {
    const auto opts = parse(argc, argv);
    if (!opts) {
        std::cerr << "Usage: pcg_benchmark --backend cpu|cpu-pcg|vulkan|cuda-pcg|cuda-pcg-mixed "
                     "[--side N] [--contrast K] [--matrix symmetric.mtx] [--repeats N] "
                     "[--rhs load.mtx --reference solution.mtx] "
                     "[--warmups N] [--budget-mib N] [--tiled] [--fp64] [--profile]\n"
                     "       pcg_benchmark --capture-deck thermal_or_magnetic.bdf --capture-prefix path\n";
        return 2;
    }
    spdlog::set_level(opts->profile ? spdlog::level::debug : spdlog::level::warn);
    if (!opts->capture_deck.empty()) return capture(*opts);
    auto matrix = opts->matrix.empty() ? std::optional<Csr>(diffusion(opts->side, opts->contrast)) : read_matrix(opts->matrix);
    if (!matrix) { std::cerr << "Could not read matrix\n"; return 2; }
    auto& k = *matrix;
    std::vector<double> expected(k.n);
    // Multiple spatial frequencies plus deterministic irregular excitation:
    // unlike a single Laplacian eigenmode this exercises Krylov convergence.
    for (int i = 0; i < k.n; ++i)
        expected[i] = std::sin(0.017 * i) + 0.3 * std::cos(0.071 * i) +
            0.1 * static_cast<double>((static_cast<std::uint64_t>(i) * 48271) % 101) / 101.0;
    auto f = k.multiply(expected);
    if (!opts->rhs.empty()) {
        auto load = read_vector(opts->rhs, k.n), solution = read_vector(opts->reference, k.n);
        if (!load || !solution) { std::cerr << "Could not read load/reference vectors\n"; return 2; }
        f = std::move(*load);
        expected = std::move(*solution);
    }
    auto solver = backend(*opts);
    if (!solver) { std::cerr << "Backend unavailable: " << opts->backend << '\n'; return 77; }
    if (solver->requires_full_symmetric_csr()) k = k.expanded_symmetric();
    std::cerr << "Backend: " << solver->name() << "; n=" << k.n << "; nnz=" << k.nnz << '\n';
    std::cout << "backend,n,nnz,budget_mib,run,seconds,iterations,corrections,relative_error,relative_residual,estimated_error\n";
    try {
        for (int run = -opts->warmups; run < opts->repeats; ++run) {
            const auto start = std::chrono::steady_clock::now();
            auto u = solver->solve(k, f);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const auto residual = compensated_residual(k, u, f);
            double error = 0, scale = 0, rr = 0, ff = 0;
            for (int i = 0; i < k.n; ++i) {
                if (!std::isfinite(u[i])) return 1;
                error = std::max(error, std::abs(u[i] - expected[i]));
                scale = std::max(scale, std::abs(expected[i]));
                rr += residual[i] * residual[i]; ff += f[i] * f[i];
            }
            scale = std::max(scale, 1e-300);
            if (run >= 0) {
                const auto refinement = solver->last_refinement();
                std::cout << std::setprecision(12) << opts->backend << ',' << k.n << ',' << k.nnz << ','
                          << opts->budget_mib << ',' << run << ',' << seconds << ',' << solver->last_iteration_count()
                          << ',' << (refinement ? refinement->corrections : 0) << ',' << error / scale
                          << ',' << std::sqrt(rr / std::max(ff, 1e-300)) << ',' << solver->last_estimated_error() << std::endl;
            }
            if (error / scale > 1e-6) {
                std::cerr << "Reference-solution error exceeds 1e-6: " << error / scale << '\n';
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Benchmark solve failed: " << e.what() << '\n';
        return 1;
    }
}
