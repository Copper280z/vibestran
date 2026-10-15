// src/solver/eigensolver_backend.cpp
// CPU eigensolver using Spectra shift-and-invert Lanczos.
//
// Solves: K φ = λ M φ
// Transform: (K - σM)^{-1} M φ = ν φ,  ν = 1/(λ - σ)
// Spectra finds ν → returns original λ = σ + 1/ν.
//
// compute() takes two SortRule arguments:
//   selection: used during Lanczos restarts; must match Lanczos convergence.
//              Lanczos naturally converges to LARGEST |ν| (modes closest to σ),
//              so use LargestMagn.
//   sorting:   applied to the final λ values (after ν→λ conversion);
//              SmallestAlge gives ascending eigenvalue order (lowest freq first).

#include "solver/eigensolver_backend.hpp"
#include "core/exceptions.hpp"
#include "core/factor_ratio_check.hpp"

#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <format>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#ifdef HAVE_ACCELERATE
#include <Eigen/AccelerateSupport>
#endif
#ifdef EIGEN_CHOLMOD_SUPPORT
#include <Eigen/CholmodSupport>
#endif

namespace vibestran {

namespace {

#if !defined(HAVE_ACCELERATE)
template <typename DiagonalExpr>
std::vector<double> copy_diagonal(const DiagonalExpr& diag_expr) {
    Eigen::VectorXd dense = diag_expr;
    return std::vector<double>(dense.data(), dense.data() + dense.size());
}

[[maybe_unused]] void
report_factor_ratio_result(const FactorRatioCheckPolicy& policy,
                           const FactorRatioCheckResult& result,
                           std::string_view backend_label) {
    if (result.status == FactorRatioCheckStatus::Ok)
        return;

    const std::string matrix_name =
        policy.matrix_name.empty() ? "matrix" : policy.matrix_name;
    if (result.status == FactorRatioCheckStatus::NonPositiveFactorDiagonal) {
        throw SolverError(std::format(
            "{} {} has non-positive or near-zero factor diagonal at row {} "
            "(matrix diag {:.3e}, factor diag {:.3e}, backend {})",
            policy.context, matrix_name, result.row_1based,
            result.matrix_diag, result.factor_diag, backend_label));
    }

    const std::string message = std::format(
        "{} {} diagonal / factor diagonal ratio {:.3e} exceeds "
        "MAXRATIO={:.3e} at row {} (matrix diag {:.3e}, factor diag {:.3e}, "
        "backend {})",
        policy.context, matrix_name, result.max_ratio, policy.maxratio,
        result.row_1based, result.matrix_diag, result.factor_diag,
        backend_label);
    if (policy.fatal)
        throw SolverError(message);
    spdlog::warn("{}", message);
}
#endif

#if defined(HAVE_ACCELERATE)

struct AccelerateOrderConfig {
    SparseOrder_t order;
    const char* label;
};

AccelerateOrderConfig accelerate_order_config() {
    const char* env = std::getenv("VIBESTRAN_ACCELERATE_ORDER");
    if (env == nullptr || env[0] == '\0')
        return {SparseOrderMetis, "metis"};
    if (std::strcmp(env, "default") == 0)
        return {SparseOrderDefault, "default"};
    if (std::strcmp(env, "amd") == 0)
        return {SparseOrderAMD, "amd"};
    if (std::strcmp(env, "metis") == 0)
        return {SparseOrderMetis, "metis"};

    spdlog::warn("Ignoring invalid VIBESTRAN_ACCELERATE_ORDER='{}' "
                 "(valid: default, amd, metis)",
                 env);
    return {SparseOrderMetis, "metis"};
}

#endif

template <typename Solver>
void apply_solve(const Solver& solver, std::string_view backend_label,
                 const double* x_in, double* y_out, Eigen::Index n) {
    using Vector = Eigen::VectorXd;
    using MapConstVec = Eigen::Map<const Vector>;
    using MapVec = Eigen::Map<Vector>;

    MapConstVec x(x_in, n);
    MapVec y(y_out, n);
    y.noalias() = solver.solve(x);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(std::format(
            "shift-invert back-substitution failed using {}", backend_label));
    }
}

class SpectraDirectShiftInvert {
public:
    using Scalar = double;
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;
    using Index = Eigen::Index;

    SpectraDirectShiftInvert(const SparseMatrix& K, const SparseMatrix& M,
                             const FactorRatioCheckPolicy* factor_ratio_policy)
        : K_(K), M_(M), n_(K.rows()),
          factor_ratio_policy_(factor_ratio_policy) {
        if (n_ != K.cols() || n_ != M.rows() || n_ != M.cols()) {
            throw std::invalid_argument(
                "SpectraDirectShiftInvert: K and M must be square matrices of the same size");
        }
    }

    [[nodiscard]] Index rows() const { return n_; }
    [[nodiscard]] Index cols() const { return n_; }
    [[nodiscard]] std::string_view backend_label() const noexcept {
        return backend_label_;
    }

    void set_shift(const Scalar& sigma) {
        // Spectra calls this from every solver constructor; reuse the
        // factorization when a solve is retried with the same shift.
        if (active_backend_ != ActiveBackend::None && factored_sigma_ == sigma)
            return;
        factored_sigma_ = sigma;
        shift_ = SparseMatrix(K_.template triangularView<Eigen::Lower>());
        SparseMatrix M_lower(M_.template triangularView<Eigen::Lower>());
        shift_ -= sigma * M_lower;
        shift_.makeCompressed();

#if defined(HAVE_ACCELERATE)
        const auto order_cfg = accelerate_order_config();

        accelerate_llt_.setOrder(order_cfg.order);
        accelerate_llt_.compute(shift_);
        if (accelerate_llt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::AccelerateLLT;
            backend_label_ = std::format("accelerate-llt(order={})", order_cfg.label);
            if (factor_ratio_policy_ != nullptr) {
                spdlog::debug(
                    "[cpu-eig] skipping MAXRATIO check for {} because factor "
                    "diagonal is unavailable",
                    backend_label_);
            }
            return;
        }

        accelerate_ldlt_.setOrder(order_cfg.order);
        accelerate_ldlt_.compute(shift_);
        if (accelerate_ldlt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::AccelerateLDLT;
            backend_label_ = std::format("accelerate-ldlt(order={})", order_cfg.label);
            if (factor_ratio_policy_ != nullptr) {
                spdlog::debug(
                    "[cpu-eig] skipping MAXRATIO check for {} because factor "
                    "diagonal is unavailable",
                    backend_label_);
            }
            return;
        }

        active_backend_ = ActiveBackend::None;
        backend_label_ = std::format("accelerate(order={})", order_cfg.label);
        throw std::runtime_error(std::format(
            "Accelerate factorization failed for K - sigma*M with sigma={:.16g}",
            sigma));
#elif defined(EIGEN_CHOLMOD_SUPPORT)
        cholmod_.compute(shift_);
        if (cholmod_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::Cholmod;
            backend_label_ = "cholmod";
            if (factor_ratio_policy_ != nullptr) {
                spdlog::debug(
                    "[cpu-eig] skipping MAXRATIO check for {} because factor "
                    "diagonal is unavailable",
                    backend_label_);
            }
            return;
        }

        simplicial_ldlt_.compute(shift_);
        if (simplicial_ldlt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::SimplicialLDLT;
            backend_label_ = "simplicial-ldlt";
            if (factor_ratio_policy_ != nullptr) {
                report_factor_ratio_result(
                    *factor_ratio_policy_,
                    evaluate_factor_ratio_check(
                        copy_diagonal(shift_.diagonal()),
                        copy_diagonal(simplicial_ldlt_.vectorD()),
                        factor_ratio_policy_->maxratio),
                    backend_label_);
            }
            return;
        }

        active_backend_ = ActiveBackend::None;
        backend_label_ = "cholmod+ldlt";
        throw std::runtime_error(std::format(
            "CHOLMOD factorization failed for K - sigma*M with sigma={:.16g}; "
            "SimplicialLDLT fallback also failed",
            sigma));
#else
        simplicial_llt_.compute(shift_);
        if (simplicial_llt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::SimplicialLLT;
            backend_label_ = "simplicial-llt";
            if (factor_ratio_policy_ != nullptr) {
                report_factor_ratio_result(
                    *factor_ratio_policy_,
                    evaluate_factor_ratio_check(
                        copy_diagonal(shift_.diagonal()),
                        copy_diagonal(simplicial_llt_.matrixL().diagonal()),
                        factor_ratio_policy_->maxratio),
                    backend_label_);
            }
            return;
        }

        simplicial_ldlt_.compute(shift_);
        if (simplicial_ldlt_.info() == Eigen::Success) {
            active_backend_ = ActiveBackend::SimplicialLDLT;
            backend_label_ = "simplicial-ldlt";
            if (factor_ratio_policy_ != nullptr) {
                report_factor_ratio_result(
                    *factor_ratio_policy_,
                    evaluate_factor_ratio_check(
                        copy_diagonal(shift_.diagonal()),
                        copy_diagonal(simplicial_ldlt_.vectorD()),
                        factor_ratio_policy_->maxratio),
                    backend_label_);
            }
            return;
        }

        active_backend_ = ActiveBackend::None;
        backend_label_ = "eigen-llt+ldlt";
        throw std::runtime_error(std::format(
            "Eigen sparse factorization failed for K - sigma*M with sigma={:.16g}; "
            "both SimplicialLLT and SimplicialLDLT failed",
            sigma));
#endif
    }

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        switch (active_backend_) {
#if defined(HAVE_ACCELERATE)
        case ActiveBackend::AccelerateLLT:
            apply_solve(accelerate_llt_, backend_label_, x_in, y_out, n_);
            return;
        case ActiveBackend::AccelerateLDLT:
            apply_solve(accelerate_ldlt_, backend_label_, x_in, y_out, n_);
            return;
#endif
#if defined(EIGEN_CHOLMOD_SUPPORT)
        case ActiveBackend::Cholmod:
            apply_solve(cholmod_, backend_label_, x_in, y_out, n_);
            return;
#endif
        case ActiveBackend::SimplicialLLT:
            apply_solve(simplicial_llt_, backend_label_, x_in, y_out, n_);
            return;
        case ActiveBackend::SimplicialLDLT:
            apply_solve(simplicial_ldlt_, backend_label_, x_in, y_out, n_);
            return;
        case ActiveBackend::None:
            break;
        }

        throw std::logic_error(
            "SpectraDirectShiftInvert::perform_op called before successful factorization");
    }

private:
    enum class ActiveBackend {
        None,
#if defined(HAVE_ACCELERATE)
        AccelerateLLT,
        AccelerateLDLT,
#endif
#if defined(EIGEN_CHOLMOD_SUPPORT)
        Cholmod,
#endif
        SimplicialLLT,
        SimplicialLDLT,
    };

    const SparseMatrix& K_;
    const SparseMatrix& M_;
    const Index n_;
    SparseMatrix shift_;
    ActiveBackend active_backend_{ActiveBackend::None};
    Scalar factored_sigma_{0.0};
    std::string backend_label_{"uninitialized"};
    const FactorRatioCheckPolicy* factor_ratio_policy_{nullptr};

#if defined(HAVE_ACCELERATE)
    Eigen::AccelerateLLT<SparseMatrix> accelerate_llt_;
    Eigen::AccelerateLDLT<SparseMatrix> accelerate_ldlt_;
#endif
#if defined(EIGEN_CHOLMOD_SUPPORT)
    Eigen::CholmodDecomposition<SparseMatrix> cholmod_;
#endif
    Eigen::SimplicialLLT<SparseMatrix> simplicial_llt_;
    Eigen::SimplicialLDLT<SparseMatrix> simplicial_ldlt_;
};

} // namespace

std::vector<int> massed_dof_indices(const Eigen::SparseMatrix<double>& M) {
    const auto n = static_cast<int>(M.rows());
    Eigen::VectorXd diag = Eigen::VectorXd::Zero(n);
    for (int col = 0; col < M.outerSize(); ++col)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, col); it; ++it)
            if (it.row() == it.col())
                diag(it.row()) += it.value();
    std::vector<int> massed;
    massed.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        if (diag(i) > 0.0)
            massed.push_back(i);
    return massed;
}

namespace {

/// Every finite eigenpair of K phi = lambda M phi by dense algebra: massless
/// DOFs (index set 0) are condensed exactly, K_red = K_mm - K_m0 K_00^-1 K_0m,
/// the definite problem K_red x = lambda M_mm x is solved, and the massless
/// components are recovered as phi_0 = -K_00^-1 K_0m x.  Vectors are returned
/// mass-normalized and sorted by ascending eigenvalue.
std::vector<EigenPair> solve_condensed_dense(const Eigen::SparseMatrix<double>& K,
                                             const Eigen::SparseMatrix<double>& M,
                                             const std::vector<int>& massed) {
    const auto n = static_cast<int>(K.rows());
    const Eigen::MatrixXd Kd =
        Eigen::MatrixXd(Eigen::SparseMatrix<double>(K.selfadjointView<Eigen::Lower>()));
    const Eigen::MatrixXd Md =
        Eigen::MatrixXd(Eigen::SparseMatrix<double>(M.selfadjointView<Eigen::Lower>()));
    std::vector<int> massless;
    {
        std::vector<char> is_massed(static_cast<size_t>(n), 0);
        for (const int i : massed)
            is_massed[static_cast<size_t>(i)] = 1;
        for (int i = 0; i < n; ++i)
            if (!is_massed[static_cast<size_t>(i)])
                massless.push_back(i);
    }
    const auto nm = static_cast<Eigen::Index>(massed.size());
    const auto n0 = static_cast<Eigen::Index>(massless.size());
    auto sub = [](const Eigen::MatrixXd& A, const std::vector<int>& rows,
                  const std::vector<int>& cols) {
        Eigen::MatrixXd S(static_cast<Eigen::Index>(rows.size()),
                          static_cast<Eigen::Index>(cols.size()));
        for (size_t i = 0; i < rows.size(); ++i)
            for (size_t j = 0; j < cols.size(); ++j)
                S(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                    A(rows[i], cols[j]);
        return S;
    };
    const Eigen::MatrixXd Kmm = sub(Kd, massed, massed);
    const Eigen::MatrixXd Mmm = sub(Md, massed, massed);
    Eigen::MatrixXd Kred = Kmm;
    Eigen::MatrixXd recover; // phi_0 = recover * x
    if (n0 > 0) {
        const Eigen::MatrixXd K00 = sub(Kd, massless, massless);
        const Eigen::MatrixXd K0m = sub(Kd, massless, massed);
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(K00);
        if (ldlt.info() != Eigen::Success || !ldlt.isPositive() ||
            ldlt.vectorD().cwiseAbs().minCoeff() <=
                1e-14 * ldlt.vectorD().cwiseAbs().maxCoeff()) {
            throw SolverError(std::format(
                "Eigensolver: the {} massless DOFs have a singular stiffness "
                "block (an unrestrained mechanism without mass)", n0));
        }
        recover = -ldlt.solve(K0m);
        Kred += K0m.transpose() * recover;
    }
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        0.5 * (Kred + Kred.transpose()), Mmm);
    if (ges.info() != Eigen::Success)
        throw SolverError(std::format(
            "Eigensolver: dense generalized eigensolve failed (n={}, massed={})",
            n, nm));
    spdlog::info("Eigensolver: all {} finite eigenvalues requested; solved the "
                 "condensed problem densely", nm);

    std::vector<EigenPair> results;
    results.reserve(static_cast<size_t>(nm));
    for (Eigen::Index k = 0; k < nm; ++k) {
        const Eigen::VectorXd x = ges.eigenvectors().col(k); // M_mm-normalized
        EigenPair ep;
        ep.eigenvalue = ges.eigenvalues()(k);
        ep.eigenvector = Eigen::VectorXd::Zero(n);
        for (Eigen::Index i = 0; i < nm; ++i)
            ep.eigenvector(massed[static_cast<size_t>(i)]) = x(i);
        if (n0 > 0) {
            const Eigen::VectorXd phi0 = recover * x;
            for (Eigen::Index i = 0; i < n0; ++i)
                ep.eigenvector(massless[static_cast<size_t>(i)]) = phi0(i);
        }
        results.push_back(std::move(ep));
    }
    return results;
}

} // namespace

std::vector<EigenPair> SpectraEigensolverBackend::solve(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int nd, double sigma,
    const FactorRatioCheckPolicy* factor_ratio_policy)
{
    const int n = static_cast<int>(K.rows());
    if (n < 1)
        throw SolverError("Eigensolver: system has no free DOFs");
    if (nd < 1)
        throw SolverError("Eigensolver: nd must be >= 1");

    // The Lanczos iteration itself is mostly serial, but the shift-invert
    // factorization and triangular solves dominate runtime for large models.
    // Use the same sparse direct backend priority as the static CPU solver:
    // Apple Accelerate on macOS, CHOLMOD when available, otherwise Eigen's
    // simplicial factorizations. This avoids Spectra's default SparseLU path.
    //
    // Massless DOFs.  Shift-invert Lanczos requires a positive definite M.
    // With lumped mass, rotations carry no inertia and M has zero rows; a
    // Lanczos run in that semi-definite M inner product produces spurious
    // Ritz pairs and misses genuine ones.  Those DOFs are condensed out
    // exactly instead: with M_00 = 0 and M_0m = 0 the reduced problem is
    //   K_red phi_m = lambda M_mm phi_m,  K_red = K_mm - K_m0 K_00^-1 K_0m,
    // and (K_red - sigma M_mm)^-1 is precisely the massed-DOF block of
    // (K - sigma M)^-1.  Lanczos therefore runs on the massed DOFs using the
    // full factorization (no Schur-complement fill-in), and each full
    // eigenvector is recovered exactly as (lambda - sigma)(K - sigma M)^-1 M phi.
    // Without massless DOFs this is the ordinary shift-invert solve.
    using FullOp = SpectraDirectShiftInvert;
    FullOp full_op(K, M, factor_ratio_policy);

    const std::vector<int> massed = massed_dof_indices(M);
    const int nm = static_cast<int>(massed.size());
    if (nm < 1) {
        throw SolverError(std::format(
            "Eigensolver: none of the {} free DOFs carries mass", n));
    }
    // The number of finite eigenvalues equals the massed DOFs.  Spectra needs
    // nd < nm; when every finite eigenvalue is requested, solve the condensed
    // problem densely instead (this only happens for small models).
    constexpr int kMaxDenseMassed = 2000;
    if (nd >= nm && nm <= kMaxDenseMassed)
        return solve_condensed_dense(K, M, massed);
    if (nd >= nm)
        nd = nm - 1;
    const bool reduced = (nm < n);

    // M restricted to the massed DOFs (lower triangle, like M).
    Eigen::SparseMatrix<double> M_mm;
    if (reduced) {
        std::vector<int> position(static_cast<size_t>(n), -1);
        for (int i = 0; i < nm; ++i)
            position[static_cast<size_t>(massed[static_cast<size_t>(i)])] = i;
        std::vector<Eigen::Triplet<double>> trips;
        for (int col = 0; col < M.outerSize(); ++col) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(M, col); it; ++it) {
                const int r = position[static_cast<size_t>(it.row())];
                const int c = position[static_cast<size_t>(it.col())];
                if (r >= 0 && c >= 0 && r >= c)
                    trips.emplace_back(r, c, it.value());
            }
        }
        M_mm.resize(nm, nm);
        M_mm.setFromTriplets(trips.begin(), trips.end());
        M_mm.makeCompressed();
    }

    /// Spectra operation y = (K_red - sigma M_mm)^-1 x on the massed DOFs,
    /// evaluated as the massed block of the full shift-invert solve.
    class ReducedShiftInvert {
    public:
        using Scalar = double;
        ReducedShiftInvert(FullOp &full, const std::vector<int> &massed, int n)
            : full_(full), massed_(massed), n_(n),
              rhs_(Eigen::VectorXd::Zero(n)), sol_(n) {}
        [[nodiscard]] Eigen::Index rows() const {
            return static_cast<Eigen::Index>(massed_.size());
        }
        [[nodiscard]] Eigen::Index cols() const { return rows(); }
        void set_shift(const Scalar &sigma) { full_.set_shift(sigma); }
        void perform_op(const Scalar *x_in, Scalar *y_out) const {
            rhs_.setZero();
            for (size_t i = 0; i < massed_.size(); ++i)
                rhs_(massed_[i]) = x_in[i];
            full_.perform_op(rhs_.data(), sol_.data());
            for (size_t i = 0; i < massed_.size(); ++i)
                y_out[i] = sol_(massed_[i]);
        }
    private:
        FullOp &full_;
        const std::vector<int> &massed_;
        int n_;
        mutable Eigen::VectorXd rhs_;
        mutable Eigen::VectorXd sol_;
    };

    const auto run = [&](auto &op, const Eigen::SparseMatrix<double> &B,
                         const int size) {
        using OpT = std::remove_reference_t<decltype(op)>;
        using BOpType = Spectra::SparseSymMatProd<double>;
        BOpType Bop(B);
        // Lanczos basis size: must satisfy nd < ncv <= size
        const int ncv = std::min(size, std::max(2 * nd + 10, 3 * nd));
        std::optional<Spectra::SymGEigsShiftSolver<
            OpT, BOpType, Spectra::GEigsMode::ShiftInvert>> solver;
        try {
            solver.emplace(op, Bop, nd, ncv, sigma);
        } catch (const std::exception& e) {
            throw SolverError(std::format(
                "Spectra eigensolver setup failed for sigma={:.16g}: {}",
                sigma, e.what()));
        }
        spdlog::info("Spectra shift-invert factorization backend: {}",
                     full_op.backend_label());
        solver->init();
        // LargestMagn selection: keep Ritz pairs with largest |nu| during
        // restarts, matching the natural convergence of the shift-inverted
        // Lanczos.  SmallestAlge sorting: after nu -> lambda conversion,
        // sort final results ascending by lambda.
        const int nconv = solver->compute(Spectra::SortRule::LargestMagn, 1000,
                                          1e-10, Spectra::SortRule::SmallestAlge);
        if (solver->info() != Spectra::CompInfo::Successful || nconv < 1)
            throw SolverError(std::format(
                "Spectra eigensolver: converged only {}/{} eigenvalues (info={})",
                nconv, nd, static_cast<int>(solver->info())));
        return std::pair<Eigen::VectorXd, Eigen::MatrixXd>{
            solver->eigenvalues(), solver->eigenvectors()};
    };

    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
    if (reduced) {
        spdlog::info("Spectra: {} of {} DOFs are massless and are condensed "
                     "out of the eigenproblem", n - nm, n);
        ReducedShiftInvert op(full_op, massed, n);
        std::tie(eigenvalues, eigenvectors) = run(op, M_mm, nm);
    } else {
        std::tie(eigenvalues, eigenvectors) = run(full_op, M, n);
    }

    std::vector<EigenPair> results;
    results.reserve(static_cast<size_t>(eigenvalues.size()));
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        EigenPair ep;
        ep.eigenvalue = eigenvalues(i);
        if (!reduced) {
            ep.eigenvector = eigenvectors.col(i);
        } else {
            // Expand: phi = (lambda - sigma)(K - sigma M)^-1 M phi, where
            // M phi only involves the massed components.  The caller
            // mass-normalizes the vector.
            Eigen::VectorXd phi_m_full = Eigen::VectorXd::Zero(n);
            for (int k = 0; k < nm; ++k)
                phi_m_full(massed[static_cast<size_t>(k)]) = eigenvectors(k, i);
            const Eigen::VectorXd m_phi =
                M.selfadjointView<Eigen::Lower>() * phi_m_full;
            Eigen::VectorXd phi(n);
            full_op.perform_op(m_phi.data(), phi.data());
            ep.eigenvector = phi * (eigenvalues(i) - sigma);
        }
        results.push_back(std::move(ep));
    }
    return results;
}

} // namespace vibestran
