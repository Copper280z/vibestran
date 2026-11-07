#pragma once
// A structural system as the analysis hands it to a backend, with its
// near-nullspace, for tests of AMG on structures.
#include "core/exceptions.hpp"
#include "io/bdf_parser.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace vibestran::test {

// Records what the analysis hands the backend, then solves directly.
class RecordingBackend final : public SolverBackend {
public:
    struct Record {
        SparseMatrixBuilder::CsrData k;
        std::vector<double> f, u;
        std::shared_ptr<const NearNullspace> modes;
    };
    explicit RecordingBackend(Record& record, bool solve_it = true) : record_(record), solve_it_(solve_it) {}
    std::vector<double> solve(const SparseMatrixBuilder::CsrData& k, const std::vector<double>& f,
                              const FactorRatioCheckPolicy*) override {
        record_.k = k;
        record_.f = f;
        record_.modes = near_nullspace();
        record_.u = solve_it_ ? direct_.solve(k, f) : std::vector<double>(f.size(), 0.0);
        return record_.u;
    }
    [[nodiscard]] std::string_view name() const noexcept override { return "recording"; }
    [[nodiscard]] bool uses_near_nullspace(int) const noexcept override { return true; }

private:
    Record& record_;
    bool solve_it_;
    LdltSolverBackend direct_;
};

// CG iterations of the host cycle on a recorded system; in amg_host_iterations.cpp.
int host_amg_iterations(const RecordingBackend::Record& r);

// New values on the same pattern for a refactor, with loads for the same solution:
// K + D K D with D in bands of equations. Equilibration hides a uniform factor or a
// diagonal congruence from the hierarchy, while this really changes M A, and so
// P = (I - M A) T. Assembling a changed deck does not do: a uniform mesh cancels some
// entries to zero, so the pattern differs and the hierarchy is built again.
struct ChangedSystem {
    SparseMatrixBuilder::CsrData k;
    std::vector<double> f;
};
inline ChangedSystem with_stiff_bands(const SparseMatrixBuilder::CsrData& k, const std::vector<double>& u) {
    ChangedSystem out{k, {}};
    const auto d = [](int i) { return i / 96 % 2 == 1 ? 5.0 : 1.0; };
    for (int i = 0; i < k.n; ++i)
        for (int j = k.row_ptr[std::size_t(i)]; j < k.row_ptr[std::size_t(i) + 1]; ++j)
            out.k.values[std::size_t(j)] *= 1 + d(i) * d(k.col_ind[std::size_t(j)]);
    out.f = out.k.multiply(u);
    return out;
}

inline RecordingBackend::Record assemble(const std::string& bulk, bool solve_it = true,
                                  std::optional<AmgModes> modes = AmgModes::Rigid,
                                  const std::string& params = {}) {
    std::istringstream deck("SOL 101\nCEND\nSPC = 1\nLOAD = 2\nBEGIN BULK\n" + params + bulk + "ENDDATA\n");
    Model model = BdfParser::parse_stream(deck);
    RecordingBackend::Record record;
    LinearStaticSolver solver(std::make_unique<RecordingBackend>(record, solve_it));
    solver.set_amg_modes(modes);
    (void)solver.solve(model);
    return record;
}

inline double largest_magnitude(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0,
                           [](double top, double v) { return std::max(top, std::abs(v)); });
}

inline const std::string kSteel = "MAT1,1,2.1+11,,0.3\nPSOLID,1,1\nPSHELL,2,1,0.01,1,,1\nPBAR,3,1,1.-3,2.-6,2.-6,3.-6\n";

// A clamped CHEXA8 bar with a shear load on its free end.
inline std::string hex_bar(int n) {
    std::ostringstream out;
    out << kSteel;
    const int nx = 4 * n;
    const auto node = [&](int i, int j, int k) { return 1 + i + (nx + 1) * (j + (n + 1) * k); };
    for (int k = 0; k <= n; ++k)
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i <= nx; ++i)
                out << std::format("GRID,{},,{},{},{}\n", node(i, j, k), 4.0 * i / nx, double(j) / n, double(k) / n);
    int eid = 1;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < nx; ++i)
                out << std::format("CHEXA,{},1,{},{},{},{},{},{},+\n+,{},{}\n", eid++, node(i, j, k), node(i + 1, j, k),
                                   node(i + 1, j + 1, k), node(i, j + 1, k), node(i, j, k + 1), node(i + 1, j, k + 1),
                                   node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1));
    for (int k = 0; k <= n; ++k)
        for (int j = 0; j <= n; ++j) {
            out << "SPC1,1,123," << node(0, j, k) << "\n";
            out << std::format("FORCE,2,{},0,1000.,0.,0.,-1.\n", node(nx, j, k));
        }
    return out.str();
}

// A thin CQUAD4 cantilever sheet bending under an edge load.
// cppcheck-suppress unusedFunction -- called from test_near_nullspace.cpp and test_amg_vcycle.cpp, which Eigen stops cppcheck on
inline std::string shell_sheet(int n) {
    std::ostringstream out;
    out << kSteel;
    const auto node = [&](int i, int j) { return 1 + i + (n + 1) * j; };
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) out << std::format("GRID,{},,{},{},0.\n", node(i, j), double(i) / n, double(j) / n);
    int eid = 1;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            out << std::format("CQUAD4,{},2,{},{},{},{}\n", eid++, node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1));
    for (int j = 0; j <= n; ++j) {
        out << "SPC1,1,123456," << node(0, j) << "\n";
        out << std::format("FORCE,2,{},0,1.,0.,0.,-1.\n", node(n, j));
    }
    return out.str();
}

// shell_sheet with its load in the plane of the sheet, which excites the
// rotations about the normal that drilling_modes() describes.
// cppcheck-suppress unusedFunction -- called from test_near_nullspace.cpp, test_cuda_pcg_amg.cpp and test_vulkan_solver.cpp, which Eigen stops cppcheck on
inline std::string in_plane_sheet(int n) {
    std::ostringstream out;
    out << kSteel;
    const auto node = [&](int i, int j) { return 1 + i + (n + 1) * j; };
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) out << std::format("GRID,{},,{},{},0.\n", node(i, j), double(i) / n, double(j) / n);
    int eid = 1;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            out << std::format("CQUAD4,{},2,{},{},{},{}\n", eid++, node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1));
    for (int i = 0; i <= n; ++i) {
        out << "SPC1,1,123456," << node(i, 0) << "\n";
        out << std::format("FORCE,2,{},0,1000.,0.,1.,0.\n", node(i, n));
    }
    return out.str();
}

// CHEXA8 blocks tied to a clamped CQUAD4 plate with RBE2 spiders: mixed grids,
// partial constraints and dense rows from one system.
// cppcheck-suppress unusedFunction -- called from test_near_nullspace.cpp and test_amg_vcycle.cpp, which Eigen stops cppcheck on
inline std::string mixed_assembly(int n) {
    std::ostringstream out;
    out << kSteel;
    const auto plate = [&](int i, int j) { return 1 + i + (n + 1) * j; };
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) out << std::format("GRID,{},,{},{},0.\n", plate(i, j), 2.0 * i / n, 2.0 * j / n);
    int eid = 1;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            out << std::format("CQUAD4,{},2,{},{},{},{}\n", eid++, plate(i, j), plate(i + 1, j), plate(i + 1, j + 1), plate(i, j + 1));
    for (int i = 0; i <= n; ++i) out << "SPC1,1,123456," << plate(i, 0) << "," << plate(i, n) << "\n";
    for (int j = 1; j < n; ++j) out << "SPC1,1,3," << plate(0, j) << "\n";
    const int m = n / 2;
    int first = plate(n, n) + 1;
    int rbe = 1000000;
    for (int b = 0; b < 2; ++b) {
        const auto node = [&](int i, int j, int k) { return first + i + (m + 1) * (j + (m + 1) * k); };
        for (int k = 0; k <= m; ++k)
            for (int j = 0; j <= m; ++j)
                for (int i = 0; i <= m; ++i)
                    out << std::format("GRID,{},,{},{},{}\n", node(i, j, k), .25 + b + .5 * i / m, .75 + .5 * j / m, .1 + .5 * k / m);
        for (int k = 0; k < m; ++k)
            for (int j = 0; j < m; ++j)
                for (int i = 0; i < m; ++i)
                    out << std::format("CHEXA,{},1,{},{},{},{},{},{},+\n+,{},{}\n", eid++, node(i, j, k), node(i + 1, j, k),
                                       node(i + 1, j + 1, k), node(i, j + 1, k), node(i, j, k + 1), node(i + 1, j, k + 1),
                                       node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1));
        const int anchor = plate((1 + 2 * b) * n / 4, n / 2);
        for (int j = 0; j <= m; ++j)
            for (int i = 0; i <= m; ++i)
                out << std::format("RBE2,{},{},123,{}\n", rbe++, anchor, node(i, j, 0));
        out << std::format("FORCE,2,{},0,1000.,0.,0.,-1.\n", node(m / 2, m / 2, m));
        first = node(m, m, m) + 1;
    }
    return out.str();
}

} // namespace vibestran::test
