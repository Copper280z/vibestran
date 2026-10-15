#pragma once
// include/solver/eigensolver_backend.hpp
// Abstract interface for generalized eigenvalue solvers: K φ = λ M φ
//
// Mirrors SolverBackend to allow future GPU eigensolver backends.

#include <Eigen/SparseCore>
#include <string>
#include <vector>

namespace vibestran {

struct FactorRatioCheckPolicy;

/// One mode shape result (single eigenvalue/eigenvector pair)
struct EigenPair {
    double eigenvalue;          ///< λ = ω² (rad²/s²), unshifted
    Eigen::VectorXd eigenvector; ///< φ, mass-normalised (φᵀMφ = 1)
};

/// Abstract interface for generalized eigensolver backends.
/// Solves: K φ = λ M φ  for the nd eigenvalues nearest to sigma.
/// Indices of the DOFs that carry mass, i.e. have a positive diagonal in M
/// (stored as its lower triangle or in full).  Since M is positive
/// semi-definite, a zero diagonal implies a zero row: such DOFs are massless
/// (e.g. rotations under lumped mass) and are condensed out of the
/// eigenproblem by the backends.  The number of finite eigenvalues equals the
/// number of massed DOFs.
[[nodiscard]] std::vector<int> massed_dof_indices(const Eigen::SparseMatrix<double>& M);

class EigensolverBackend {
public:
    virtual ~EigensolverBackend() = default;

    /// Solve the generalised eigenvalue problem K φ = λ M φ.
    /// @param K     Stiffness matrix (symmetric positive semi-definite)
    /// @param M     Mass matrix (symmetric positive semi-definite; massless
    ///              DOFs, i.e. zero rows, are condensed out exactly)
    /// @param nd    Number of desired eigenpairs
    /// @param sigma Shift: find eigenvalues closest to this value
    /// @param factor_ratio_policy Optional PARAM,MAXRATIO enforcement policy
    ///                            for backends that expose factor diagonals
    /// @return      Up to nd EigenPairs sorted by eigenvalue (ascending)
    [[nodiscard]] virtual std::vector<EigenPair> solve(
        const Eigen::SparseMatrix<double>& K,
        const Eigen::SparseMatrix<double>& M,
        int nd, double sigma,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

/// CPU implementation using Spectra (shift-and-invert Lanczos).
class SpectraEigensolverBackend : public EigensolverBackend {
public:
    [[nodiscard]] std::vector<EigenPair> solve(
        const Eigen::SparseMatrix<double>& K,
        const Eigen::SparseMatrix<double>& M,
        int nd, double sigma,
        const FactorRatioCheckPolicy* factor_ratio_policy = nullptr) override;

    [[nodiscard]] std::string name() const override {
#if defined(HAVE_ACCELERATE)
        return "Spectra + Apple Accelerate (CPU shift-invert Lanczos)";
#elif defined(EIGEN_CHOLMOD_SUPPORT)
        return "Spectra + SuiteSparse CHOLMOD (CPU shift-invert Lanczos)";
#else
        return "Spectra + Eigen sparse direct (CPU shift-invert Lanczos)";
#endif
    }
};

} // namespace vibestran
