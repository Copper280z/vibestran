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
    EigenSolverBackend direct_;
};

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

} // namespace vibestran::test
