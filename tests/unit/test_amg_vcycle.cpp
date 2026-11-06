// tests/unit/test_amg_vcycle.cpp
// The preconditioner conjugate gradients actually applies: the whole V-cycle.
//
// With R = P', A_c = P'A P of full rank, rho(M A) < 2 on every level and an SPD
// coarse solve, the cycle B is SPD and the eigenvalues of B A lie in (0, 1].
// Small assembled systems are checked by forming B explicitly, across the kinds
// of model and frames of reference that could break a link in that argument: partly
// constrained grids, lines of grids whose modes are rank deficient on an aggregate,
// material interfaces, RBEs, and shifted, rotated or rescaled coordinates.

#include <gtest/gtest.h>
#include "amg_cpu.hpp"
#include "structural_system.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

using namespace vibestran;
using vibestran::test::assemble;

namespace {

// Where a deck's geometry sits, and in what length unit: a new deck of the same
// structure. Sections and moduli follow the unit, so stresses keep their meaning.
struct Frame {
    std::string name;
    std::array<double, 3> shift{0, 0, 0};
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    double unit = 1;

    [[nodiscard]] Eigen::Vector3d direction(double x, double y, double z) const { return rotation * Eigen::Vector3d(x, y, z); }
    [[nodiscard]] std::string grid(int id, double x, double y, double z) const {
        const Eigen::Vector3d p = unit * direction(x, y, z);
        return std::format("GRID,{},,{:.14e},{:.14e},{:.14e}\n", id, p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]);
    }
    [[nodiscard]] std::string properties() const {
        const double e = 2.1e11 / (unit * unit);
        return std::format("MAT1,1,{:.8e},,0.3\nMAT1,2,{:.8e},,0.3\nPSOLID,1,1\nPSOLID,2,2\nPSHELL,3,1,{:.8e},1,,1\n"
                           "PBAR,4,1,{:.8e},{:.8e},{:.8e},{:.8e}\n",
                           e, 1e-6 * e, 0.01 * unit, 1e-3 * unit * unit, 2e-6 * std::pow(unit, 4), 2e-6 * std::pow(unit, 4),
                           3e-6 * std::pow(unit, 4));
    }
};

Eigen::Matrix3d skew_rotation() {
    return (Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(-1.1, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX())).toRotationMatrix();
}

const std::vector<Frame>& frames() {
    static const std::vector<Frame> all{
        {"basic"},
        {"shifted", {1.0e4, -2.0e4, 3.0e4}},
        {"rotated", {0, 0, 0}, skew_rotation()},
        {"millimetres", {0, 0, 0}, Eigen::Matrix3d::Identity(), 1000.0},
    };
    return all;
}

void hex_cards(const Frame& f, int nx, int ny, int nz, int first_grid, int first_element, double z0,
                      const std::function<int(int)>& property, std::ostringstream& out) {
    const auto node = [=](int i, int j, int k) { return first_grid + i + (nx + 1) * (j + (ny + 1) * k); };
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i)
                out << f.grid(node(i, j, k), double(i) / ny, double(j) / ny, z0 + double(k) / ny);
    int eid = first_element;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                out << std::format("CHEXA,{},{},{},{},{},{},{},{},+\n+,{},{}\n", eid++, property(i), node(i, j, k),
                                   node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k), node(i, j, k + 1),
                                   node(i + 1, j, k + 1), node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1));
}

// Clamped end, and grids along the far end held in one direction only.
std::string constrained_solid(const Frame& f) {
    std::ostringstream out;
    out << f.properties();
    constexpr int nx = 8, ny = 3;
    hex_cards(f, nx, ny, ny, 1, 1, 0.0, [](int) { return 1; }, out);
    const auto node = [](int i, int j, int k) { return 1 + i + (nx + 1) * (j + (ny + 1) * k); };
    for (int k = 0; k <= ny; ++k)
        for (int j = 0; j <= ny; ++j) out << "SPC1,1,123," << node(0, j, k) << "\n";
    for (int j = 0; j <= ny; ++j) out << "SPC1,1,3," << node(nx, j, 0) << "\n";
    const Eigen::Vector3d load = f.direction(0, 0, -1);
    out << std::format("FORCE,2,{},0,1000.,{:.12e},{:.12e},{:.12e}\n", node(nx, ny, ny), load[0], load[1], load[2]);
    return out.str();
}

// A stiff half joined to one softer by 10^6.
std::string material_interface(const Frame& f) {
    std::ostringstream out;
    out << f.properties();
    constexpr int nx = 8, ny = 3;
    hex_cards(f, nx, ny, ny, 1, 1, 0.0, [](int i) { return i < nx / 2 ? 1 : 2; }, out);
    const auto node = [](int i, int j, int k) { return 1 + i + (nx + 1) * (j + (ny + 1) * k); };
    for (int k = 0; k <= ny; ++k)
        for (int j = 0; j <= ny; ++j) out << "SPC1,1,123," << node(0, j, k) << "\n";
    const Eigen::Vector3d load = f.direction(0, 0, -1);
    out << std::format("FORCE,2,{},0,1.,{:.12e},{:.12e},{:.12e}\n", node(nx, ny, ny), load[0], load[1], load[2]);
    return out.str();
}

void sheet_cards(const Frame& f, int n, std::ostringstream& out) {
    const auto node = [n](int i, int j) { return 1 + i + (n + 1) * j; };
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) out << f.grid(node(i, j), double(i) / n, double(j) / n, 0.0);
    int eid = 1;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            out << std::format("CQUAD4,{},3,{},{},{},{}\n", eid++, node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1));
}

// A clamped edge and a far edge supported out of plane only.
std::string constrained_shell(const Frame& f) {
    std::ostringstream out;
    out << f.properties();
    constexpr int n = 10;
    sheet_cards(f, n, out);
    const auto node = [](int i, int j) { return 1 + i + (n + 1) * j; };
    for (int j = 0; j <= n; ++j) {
        out << "SPC1,1,123456," << node(0, j) << "\n";
        out << "SPC1,1,3," << node(n, j) << "\n";
    }
    const Eigen::Vector3d load = f.direction(0, 0, -1);
    out << std::format("FORCE,2,{},0,1.,{:.12e},{:.12e},{:.12e}\n", node(n / 2, n / 2), load[0], load[1], load[2]);
    return out.str();
}

// CBARs offset from a shell's midplane, as stiffeners are.
std::string stiffened_shell(const Frame& f) {
    std::ostringstream out;
    out << f.properties();
    constexpr int n = 10;
    sheet_cards(f, n, out);
    const auto node = [](int i, int j) { return 1 + i + (n + 1) * j; };
    const Eigen::Vector3d normal = f.direction(0, 0, 1), offset = f.unit * f.direction(0, 0, 0.03);
    int eid = 1000;
    for (const int j : {3, 7})
        for (int i = 0; i < n; ++i)
            out << std::format("CBAR,{},4,{},{},{:.10e},{:.10e},{:.10e},,+\n+,,,{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}\n",
                               eid++, node(i, j), node(i + 1, j), normal[0], normal[1], normal[2], offset[0], offset[1],
                               offset[2], offset[0], offset[1], offset[2]);
    for (int j = 0; j <= n; ++j) out << "SPC1,1,123456," << node(0, j) << "\n";
    const Eigen::Vector3d load = f.direction(0, 0, -1);
    out << std::format("FORCE,2,{},0,1.,{:.12e},{:.12e},{:.12e}\n", node(n, n / 2), load[0], load[1], load[2]);
    return out.str();
}

// A solid hung from a shell by an RBE2 and loaded through an RBE3: 3- and 6-DOF
// grids, dense rows, and constraint elimination in one matrix.
std::string rbe_coupled(const Frame& f) {
    std::ostringstream out;
    out << f.properties();
    constexpr int n = 8, m = 2;
    sheet_cards(f, n, out);
    const auto plate = [](int i, int j) { return 1 + i + (n + 1) * j; };
    for (int i = 0; i <= n; ++i) out << "SPC1,1,123456," << plate(i, 0) << "," << plate(i, n) << "\n";
    const int first = plate(n, n) + 1;
    hex_cards(f, m, m, m, first, 1000, 0.05, [](int) { return 1; }, out);
    const auto block = [first](int i, int j, int k) { return first + i + (m + 1) * (j + (m + 1) * k); };
    int rbe = 100000;
    for (int j = 0; j <= m; ++j)
        for (int i = 0; i <= m; ++i) out << std::format("RBE2,{},{},123,{}\n", rbe++, plate(n / 2, n / 2), block(i, j, 0));
    const int reference = block(m, m, m) + 1;
    out << f.grid(reference, 0.5, 0.5, 2.0);
    out << std::format("RBE3,{},,{},123456,1.,123,{},{},+\n+,{},{}\n", rbe, reference, block(0, 0, m), block(m, 0, m),
                       block(m, m, m), block(0, m, m));
    const Eigen::Vector3d load = f.direction(1, 0, -1);
    out << std::format("FORCE,2,{},0,1.,{:.12e},{:.12e},{:.12e}\n", reference, load[0], load[1], load[2]);
    return out.str();
}

// Lines of grids: on a straight line of bars an aggregate's modes lose rank.
std::string bar_lines(const Frame& f) {
    std::ostringstream out;
    out << f.properties();
    constexpr int n = 40;
    const auto node = [](int line, int i) { return 1 + i + (n + 1) * line; };
    const Eigen::Vector3d normal = f.direction(0, 0, 1);
    int eid = 1;
    for (int line = 0; line < 3; ++line) {
        for (int i = 0; i <= n; ++i) out << f.grid(node(line, i), double(i) / n, 0.2 * line, 0.0);
        for (int i = 0; i < n; ++i)
            out << std::format("CBAR,{},4,{},{},{:.10e},{:.10e},{:.10e}\n", eid++, node(line, i), node(line, i + 1), normal[0],
                               normal[1], normal[2]);
        out << "SPC1,1,123456," << node(line, 0) << "\n";
    }
    for (int i = 10; i <= n; i += 10) {
        out << std::format("CBAR,{},4,{},{},{:.10e},{:.10e},{:.10e}\n", eid++, node(0, i), node(1, i), normal[0], normal[1], normal[2]);
        out << std::format("CBAR,{},4,{},{},{:.10e},{:.10e},{:.10e}\n", eid++, node(1, i), node(2, i), normal[0], normal[1], normal[2]);
    }
    const Eigen::Vector3d load = f.direction(0, 0, -1);
    out << std::format("FORCE,2,{},0,1.,{:.12e},{:.12e},{:.12e}\n", node(1, n), load[0], load[1], load[2]);
    return out.str();
}

using RecordingBackendRecord = test::RecordingBackend::Record;

struct ModelKind {
    std::string name;
    std::function<std::string(const Frame&)> deck;
};

const std::vector<ModelKind>& models() {
    static const std::vector<ModelKind> all{
        {"constrained solid", constrained_solid}, {"material interface", material_interface},
        {"constrained shell", constrained_shell}, {"stiffened shell", stiffened_shell},
        {"RBE-coupled solid and shell", rbe_coupled}, {"lines of bars", bar_lines},
    };
    return all;
}

Eigen::MatrixXd dense(const detail::AmgHostCsr& a) {
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(Eigen::Index(a.nrows), Eigen::Index(a.ncols));
    for (std::size_t i = 0; i < a.nrows; ++i)
        for (int j = a.ptr[i]; j < a.ptr[i + 1]; ++j) out(Eigen::Index(i), a.col[j]) = a.val[j];
    return out;
}

// B of any cycle, from its action on the unit vectors. z = apply(f).
template<class Apply> Eigen::MatrixXd formed_cycle(std::size_t n, Apply&& apply) {
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(Eigen::Index(n), Eigen::Index(n));
    std::vector<double> unit(n, 0.0), z(n);
    for (std::size_t c = 0; c < n; ++c) {
        unit[c] = 1;
        apply(unit, z);
        b.col(Eigen::Index(c)) = Eigen::Map<const Eigen::VectorXd>(z.data(), Eigen::Index(n));
        unit[c] = 0;
    }
    return b;
}

Eigen::MatrixXd applied_cycle(const detail::AmgHierarchy& h) {
    detail::HostAmg amg(h, nullptr);
    return formed_cycle(h.levels.front().A->nrows, [&](const std::vector<double>& f, std::vector<double>& z) { z = amg.cycle(f); });
}

Eigen::MatrixXd applied_cpu_cycle(const detail::AmgHierarchy& h) {
    detail::AmgCpuCycleOf<double> cycle(h);
    return formed_cycle(h.levels.front().A->nrows, [&](const std::vector<double>& f, std::vector<double>& z) { cycle.apply(f, z); });
}

struct CycleSpectrum {
    double asymmetry, low, high;
};

// The eigenvalues of B A, from the similar symmetric L'B L with A = L L'.
CycleSpectrum spectrum_of(const detail::AmgHierarchy& h, const Eigen::MatrixXd& b) {
    const Eigen::MatrixXd a = dense(*h.levels.front().A);
    const Eigen::LLT<Eigen::MatrixXd> llt(a);
    EXPECT_EQ(llt.info(), Eigen::Success);
    const Eigen::MatrixXd l = llt.matrixL();
    const Eigen::MatrixXd s = l.transpose() * b * l;
    const Eigen::VectorXd values =
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(0.5 * (s + s.transpose()), Eigen::EigenvaluesOnly).eigenvalues();
    return {(b - b.transpose()).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(), values.minCoeff(), values.maxCoeff()};
}

CycleSpectrum spectrum(const detail::AmgHierarchy& h) { return spectrum_of(h, applied_cycle(h)); }

double smoother_top(const detail::AmgHostCsr& a, const std::vector<double>& m) {
    const Eigen::VectorXd root = Eigen::Map<const Eigen::VectorXd>(m.data(), Eigen::Index(m.size())).cwiseSqrt();
    const Eigen::MatrixXd scaled = root.asDiagonal() * dense(a) * root.asDiagonal();
    return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(0.5 * (scaled + scaled.transpose()), Eigen::EigenvaluesOnly).eigenvalues().maxCoeff();
}

void expect_every_link_holds(const detail::AmgHierarchy& h) {
    for (std::size_t l = 0; l + 1 < h.levels.size(); ++l) {
        const auto& level = h.levels[l];
        ASSERT_TRUE(level.P && level.R);
        EXPECT_LE(smoother_top(*level.A, level.smoother), detail::kAmgSmootherBound * (1 + 1e-12)) << "level " << l;
        const Eigen::MatrixXd p = dense(*level.P);
        EXPECT_LE((dense(*level.R) - p.transpose()).cwiseAbs().maxCoeff(), 1e-13 * p.cwiseAbs().maxCoeff()) << "R != P' at level " << l;
        const Eigen::MatrixXd galerkin = p.transpose() * dense(*level.A) * p;
        const Eigen::MatrixXd coarse = dense(*h.levels[l + 1].A);
        EXPECT_LE((coarse - galerkin).cwiseAbs().maxCoeff(), 1e-11 * galerkin.cwiseAbs().maxCoeff()) << "A_c != P'A P at level " << l;
        EXPECT_EQ(Eigen::LLT<Eigen::MatrixXd>(coarse).info(), Eigen::Success) << "A_c not SPD at level " << l;
    }
}

void expect_valid_cycle(const detail::AmgHierarchy& h) {
    expect_every_link_holds(h);
    const auto s = spectrum(h);
    EXPECT_LE(s.asymmetry, 1e-10);
    EXPECT_GT(s.low, 0.0);
    EXPECT_LE(s.high, 1 + 1e-9);
    const auto probe = detail::probe_amg_cycle(h);
    EXPECT_TRUE(probe.valid()) << std::format("asymmetry {:.2e} energy {:.3e}..{:.9f} contraction {:.9f}", probe.asymmetry,
                                              probe.min_energy, probe.max_energy, probe.contraction);
    // The probe agrees with the spectrum: E has A-norm 1 - lambda_min(B A) and
    // z'A z / f'z with z = B f lies within the spectrum of B A.
    EXPECT_LE(probe.contraction, 1 - s.low + 1e-9);
    EXPECT_LE(probe.max_energy, s.high + 1e-9);
    EXPECT_GE(probe.max_energy, s.low - 1e-9);
}

} // namespace

// Three or more levels, so the cycle's recursion is what is being checked, with
// and without modes, in every frame.
TEST(AmgCycle, IsSymmetricPositiveDefiniteWithSpectrumInUnitIntervalAcrossModelsAndFrames) {
    for (const auto& model : models())
        for (const auto& frame : frames()) {
            const auto r = assemble(model.deck(frame), false);
            ASSERT_TRUE(r.modes);
            const detail::ScaledAmgMatrix scaled(r.k);
            for (const bool with_modes : {true, false}) {
                SCOPED_TRACE(std::format("{}, {} coordinates, {}, n={}", model.name, frame.name, with_modes ? "rigid" : "constant", r.k.n));
                const detail::AmgHierarchy h(scaled.matrix, with_modes ? r.modes.get() : nullptr, scaled.scale, 40);
                EXPECT_GE(h.levels.size(), 2u);
                expect_valid_cycle(h);
            }
        }
}

// The CPU's Gauss-Seidel cycle meets the same bound with no weight to bound: any
// hierarchy of the same links, so even where the coarsest level is smoothed, not solved.
TEST(AmgCycle, GaussSeidelOnTheCpuIsSymmetricPositiveDefiniteWithSpectrumInUnitInterval) {
    for (const auto& model : models()) {
        const auto r = assemble(model.deck(frames().front()), false);
        const detail::ScaledAmgMatrix scaled(r.k);
        for (const bool with_modes : {true, false})
            for (const unsigned coarse_enough : {40u, 1u}) {
                SCOPED_TRACE(std::format("{}, {}, coarse_enough={}, n={}", model.name, with_modes ? "rigid" : "constant",
                                         coarse_enough, r.k.n));
                const detail::AmgHierarchy h(scaled.matrix, with_modes ? r.modes.get() : nullptr, scaled.scale, coarse_enough);
                const auto s = spectrum_of(h, applied_cpu_cycle(h));
                EXPECT_LE(s.asymmetry, 1e-10);
                EXPECT_GT(s.low, 0.0);
                EXPECT_LE(s.high, 1 + 1e-9);
            }
    }
}

// The coarsest level with nothing left to aggregate is smoothed, not solved: two
// smoothing steps, still SPD.
TEST(AmgCycle, StaysPositiveDefiniteWithASmoothedCoarsestLevel) {
    for (const auto& model : models()) {
        SCOPED_TRACE(model.name);
        const auto r = assemble(model.deck(frames().front()), false);
        const detail::ScaledAmgMatrix scaled(r.k);
        for (const bool with_modes : {true, false}) {
            const detail::AmgHierarchy h(scaled.matrix, with_modes ? r.modes.get() : nullptr, scaled.scale, 1);
            // The scalar hierarchy can reach one row, which is then solved.
            if (with_modes) { ASSERT_FALSE(h.coarse_direct()); }
            const auto s = spectrum(h);
            EXPECT_LE(s.asymmetry, 1e-10);
            EXPECT_GT(s.low, 0.0);
            EXPECT_LE(s.high, 1 + 1e-9);
        }
    }
}

// Without the bound, these checks must fail: SPAI-0 leaves the cycle indefinite on
// a structure, and the probe sees it on a system too large to form.
TEST(AmgCycle, TheChecksCatchUnboundedWeights) {
    const auto r = assemble(constrained_solid(frames().front()), false);
    const detail::ScaledAmgMatrix scaled(r.k);
    detail::AmgHierarchy h(scaled.matrix, nullptr, scaled.scale, 40);
    for (std::size_t l = 0; l + 1 < h.levels.size(); ++l) h.levels[l].smoother = detail::spai0_weights(*h.levels[l].A);
    const auto s = spectrum(h);
    EXPECT_LT(s.low, 0.0) << "SPAI-0 no longer breaks this cycle, so this test proves nothing";
    EXPECT_FALSE(detail::probe_amg_cycle(h).valid());
}

// Shifting a model or changing its length unit changes no physics and, after
// equilibration, nothing numerically: not the hierarchy, and not the solution.
// Iterations are compared loosely, since rounding and the order of threaded sums
// move CG by tens of iterations on the ill-conditioned material interface.
TEST(AmgCycle, IterationsIgnoreTheOriginAndTheLengthUnit) {
    for (const auto& model : models()) {
        SCOPED_TRACE(model.name);
        const auto base = assemble(model.deck(frames()[0]));
        const auto reference = detail::solve_hybrid_amg(base.k, base.f, 1e-9, 5000, {}, base.modes);
        for (const std::size_t f : {std::size_t(1), std::size_t(3)}) {
            SCOPED_TRACE(frames()[f].name);
            const auto r = assemble(model.deck(frames()[f]));
            ASSERT_EQ(r.k.n, base.k.n);
            // Rows only: rounding turns some exact zeros of K into tiny entries.
            const auto shape = [](const RecordingBackendRecord& record) {
                const detail::ScaledAmgMatrix scaled(record.k);
                const detail::AmgHierarchy h(scaled.matrix, record.modes.get(), scaled.scale, 40);
                std::vector<std::size_t> rows(h.levels.size());
                std::ranges::transform(h.levels, rows.begin(), [](const auto& level) { return level.A->nrows; });
                return rows;
            };
            EXPECT_EQ(shape(r), shape(base));
            const auto solved = detail::solve_hybrid_amg(r.k, r.f, 1e-9, 5000, {}, r.modes);
            EXPECT_TRUE(solved.refinement.converged);
            EXPECT_LE(solved.iterations, 2 * reference.iterations + 10);
            EXPECT_LE(reference.iterations, 2 * solved.iterations + 10);
            const double top = test::largest_magnitude(r.u);
            for (std::size_t i = 0; i < r.u.size(); ++i) ASSERT_NEAR(solved.solution[i], r.u[i], 1e-6 * top) << i;
        }
        // Rotating changes which couplings are strong, since the point graph uses
        // the largest entry of each block, which is not invariant, so aggregates
        // differ. A rigid-body method still must not depend on the axes in any
        // large way. Lines of bars come closest to this limit, at 2.1 times.
        const auto rotated = assemble(model.deck(frames()[2]));
        const auto solved = detail::solve_hybrid_amg(rotated.k, rotated.f, 1e-9, 5000, {}, rotated.modes);
        EXPECT_TRUE(solved.refinement.converged);
        EXPECT_LE(solved.iterations, 3 * reference.iterations + 10);
    }
}

// Systems too large to form B, by the bilinear forms alone.
TEST(AmgCycle, ProbesHoldOnLargerSystems) {
    const std::vector<std::string> decks{test::hex_bar(8), test::shell_sheet(48), test::mixed_assembly(16)};
    for (const auto& bulk : decks) {
        const auto r = assemble(bulk, false);
        const detail::ScaledAmgMatrix scaled(r.k);
        for (const bool with_modes : {true, false}) {
            const detail::AmgHierarchy h(scaled.matrix, with_modes ? r.modes.get() : nullptr, scaled.scale);
            const auto probe = detail::probe_amg_cycle(h);
            EXPECT_TRUE(probe.valid()) << std::format("n={} modes={} asymmetry {:.2e} energy {:.3e}..{:.9f} contraction {:.9f}",
                r.k.n, with_modes, probe.asymmetry, probe.min_energy, probe.max_energy, probe.contraction);
        }
    }
}
