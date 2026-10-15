#pragma once
// tests/unit/lumped_beam_problem.hpp
// Shared eigenproblem with massless DOFs for the CPU and CUDA eigensolver
// tests: a fixed-free Euler-Bernoulli beam of `elements` equal elements with
// DOFs (w, theta) per free node and lumped translational mass, so every
// rotation is massless (M has zero rows).  The exact discrete eigenvalues are
// obtained densely by condensing the rotations out of K (Schur complement)
// and solving the reduced definite problem.

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <vector>

namespace vibestran::test {

struct LumpedBeamProblem {
  Eigen::SparseMatrix<double> K; ///< lower triangle stored
  Eigen::SparseMatrix<double> M; ///< lower triangle stored (diagonal)
  Eigen::VectorXd reference_eigenvalues; ///< ascending, one per massed DOF
};

[[nodiscard]] inline LumpedBeamProblem
make_lumped_beam_problem(const int elements, const double EI = 2.0,
                         const double length = 1.0, const double mass_per_length = 3.0) {
  const double L = length / elements;
  const int nodes = elements;            // free nodes (node 0 is clamped)
  const int n = 2 * nodes;               // DOFs: w_i at 2i, theta_i at 2i+1
  Eigen::MatrixXd Kd = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd Md = Eigen::MatrixXd::Zero(n, n);
  Eigen::Matrix4d ke;
  ke << 12, 6 * L, -12, 6 * L,
        6 * L, 4 * L * L, -6 * L, 2 * L * L,
        -12, -6 * L, 12, -6 * L,
        6 * L, 2 * L * L, -6 * L, 4 * L * L;
  ke *= EI / (L * L * L);
  for (int e = 0; e < elements; ++e) {
    // Element DOFs: node e (w, theta) and node e+1; global node k -> 2(k-1).
    const int dofs[4] = {2 * (e - 1), 2 * (e - 1) + 1, 2 * e, 2 * e + 1};
    for (int a = 0; a < 4; ++a) {
      if (dofs[a] < 0)
        continue;
      for (int b = 0; b < 4; ++b)
        if (dofs[b] >= 0)
          Kd(dofs[a], dofs[b]) += ke(a, b);
    }
    // Lumped translational mass: half an element at each end.
    const double half = 0.5 * mass_per_length * L;
    if (e > 0)
      Md(2 * (e - 1), 2 * (e - 1)) += half;
    Md(2 * e, 2 * e) += half;
  }

  std::vector<Eigen::Triplet<double>> kt, mt;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j) {
      if (Kd(i, j) != 0.0)
        kt.emplace_back(i, j, Kd(i, j));
      if (Md(i, j) != 0.0)
        mt.emplace_back(i, j, Md(i, j));
    }
  LumpedBeamProblem p;
  p.K.resize(n, n);
  p.M.resize(n, n);
  p.K.setFromTriplets(kt.begin(), kt.end());
  p.M.setFromTriplets(mt.begin(), mt.end());

  // Reference: K_red = K_ww - K_wt K_tt^-1 K_tw, then K_red x = lambda M_ww x.
  Eigen::MatrixXd Kww(nodes, nodes), Kwt(nodes, nodes), Ktt(nodes, nodes);
  Eigen::MatrixXd Mww = Eigen::MatrixXd::Zero(nodes, nodes);
  for (int a = 0; a < nodes; ++a) {
    Mww(a, a) = Md(2 * a, 2 * a);
    for (int b = 0; b < nodes; ++b) {
      Kww(a, b) = Kd(2 * a, 2 * b);
      Kwt(a, b) = Kd(2 * a, 2 * b + 1);
      Ktt(a, b) = Kd(2 * a + 1, 2 * b + 1);
    }
  }
  const Eigen::MatrixXd Kred = Kww - Kwt * Ktt.ldlt().solve(Kwt.transpose());
  const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(Kred, Mww);
  p.reference_eigenvalues = ges.eigenvalues();
  return p;
}

} // namespace vibestran::test
