// src/solver/iterative_refinement.cpp
// See include/solver/iterative_refinement.hpp.

#include "solver/iterative_refinement.hpp"
#include "core/exceptions.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>

namespace vibestran {

namespace {

/// max |v_i|, or NaN when any entry is NaN (std::max would drop it).
double inf_norm(std::span<const double> v) {
  double m = 0.0;
  for (double x : v) {
    if (std::isnan(x))
      return x;
    m = std::max(m, std::abs(x));
  }
  return m;
}

double two_norm(std::span<const double> v) {
  return std::sqrt(
      std::accumulate(v.begin(), v.end(), 0.0, [](double s, double x) { return s + x * x; }));
}

/// ||K||_inf, reading lower-triangular storage as symmetric.
double matrix_inf_norm(const SparseMatrixBuilder::CsrData &K) {
  std::vector<double> rows(static_cast<std::size_t>(K.n), 0.0);
  const bool lower = K.stores_lower_triangle_only();
  for (int i = 0; i < K.n; ++i)
    for (int p = K.row_ptr[static_cast<std::size_t>(i)];
         p < K.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
      const int j = K.col_ind[static_cast<std::size_t>(p)];
      const double a = std::abs(K.values[static_cast<std::size_t>(p)]);
      rows[static_cast<std::size_t>(i)] += a;
      if (lower && j != i)
        rows[static_cast<std::size_t>(j)] += a;
    }
  return inf_norm(rows);
}

} // namespace

// The error-free transformations need IEEE arithmetic without reassociation
// or FMA contraction (s - a * x fused into an FMA double-counts the product's
// rounding error).  meson.build compiles this file without the release
// build's -ffast-math; RefinementTest.CompensatedResidualIsExact checks it.

namespace {

/// s -= a x exactly, with the rounding error added to c: p + pe = a x
/// (TwoProd), t + te = s - p (TwoSum).
inline void subtract_product(double &s, double &c, double a, double x) {
  const double p = a * x;
  const double pe = std::fma(a, x, -p);
  const double t = s - p;
  const double z = t - s;
  const double te = (s - (t - z)) + (-p - z);
  s = t;
  c += te - pe;
}

} // namespace

std::vector<double> compensated_residual(const SparseMatrixBuilder::CsrData &K,
                                         std::span<const double> x,
                                         std::span<const double> F) {
  const auto n = static_cast<std::size_t>(K.n);
  if (x.size() != n || F.size() != n)
    throw SolverError(std::format("compensated_residual: sizes {} / {} do not match the "
                                  "{}x{} matrix",
                                  x.size(), F.size(), n, n));
  std::vector<double> s(F.begin(), F.end()); // running sums
  std::vector<double> c(n, 0.0);             // accumulated rounding errors
  const bool lower = K.stores_lower_triangle_only();
  for (std::size_t i = 0; i < n; ++i)
    for (int p = K.row_ptr[i]; p < K.row_ptr[i + 1]; ++p) {
      const auto j = static_cast<std::size_t>(K.col_ind[static_cast<std::size_t>(p)]);
      const double a = K.values[static_cast<std::size_t>(p)];
      subtract_product(s[i], c[i], a, x[j]);
      if (lower && j != i)
        subtract_product(s[j], c[j], a, x[i]);
    }
  for (std::size_t i = 0; i < n; ++i)
    s[i] += c[i];
  return s;
}


RefinementReport refine_solution(const SparseMatrixBuilder::CsrData &K, std::span<const double> F,
                                 std::vector<double> &u, const CorrectionSolve &solve,
                                 const RefinementOptions &options) {
  RefinementReport rep;
  const double normK = matrix_inf_norm(K);
  const double normF_inf = inf_norm(F);
  const double normF_2 = two_norm(F);
  const auto backward_error = [&](const std::vector<double> &r, std::span<const double> x) {
    return inf_norm(r) / (normK * inf_norm(x) + normF_inf);
  };
  auto finish = [&](const std::vector<double> &r) {
    rep.backward_error = backward_error(r, u);
    rep.relative_residual = normF_2 > 0.0 ? two_norm(r) / normF_2 : two_norm(r);
    const double uF = std::inner_product(u.begin(), u.end(), F.begin(), 0.0);
    const double ur = std::inner_product(u.begin(), u.end(), r.begin(), 0.0);
    rep.epsilon = uF != 0.0 ? ur / uF : (ur == 0.0 ? 0.0 : 1.0);
    return rep;
  };
  std::vector<double> r = compensated_residual(K, u, F);
  if (!std::isfinite(inf_norm(r))) { // the solve produced Inf or NaN
    rep.initial_error = rep.estimated_error = std::numeric_limits<double>::quiet_NaN();
    return finish(r);
  }
  if (normF_inf == 0.0 || inf_norm(r) == 0.0) {
    rep.converged = true;
    return finish(r);
  }
  double bw = backward_error(r, u);
  // Contraction is judged on the absolute size of the corrections: an
  // inner solver that overshoots inflates u as well, so relative sizes can
  // shrink while the error grows.
  double previous = 0.0;
  std::vector<double> next(u.size());
  for (int k = 0; k < options.max_corrections; ++k) {
    std::vector<double> d = solve(r);
    if (d.size() != u.size())
      throw SolverError(std::format("refine_solution: correction has {} entries, expected {}",
                                    d.size(), u.size()));
    const double dn = inf_norm(d);
    const double un = inf_norm(u);
    const double size = un > 0.0 ? dn / un : dn;
    if (!std::isfinite(size)) {
      rep.estimated_error = size;
      return finish(r);
    }
    // r != 0 here, so K d = r has no zero solution: a zero correction is an
    // inner solver that broke down without progress (e.g. a PCG stopped at
    // its first iteration), not convergence.
    if (dn == 0.0) {
      if (k == 0)
        rep.initial_error = rep.estimated_error = std::numeric_limits<double>::infinity();
      return finish(r);
    }
    if (k == 0)
      rep.initial_error = size;
    rep.estimated_error = size;
    // A growing correction (a singular or hopelessly ill-conditioned K, or
    // an inner solver that failed): keep the current iterate.
    if (k > 0 && dn > previous)
      return finish(r);
    for (std::size_t i = 0; i < u.size(); ++i)
      next[i] = u[i] + d[i];
    std::vector<double> r_next = compensated_residual(K, next, F);
    const double bw_next = backward_error(r_next, next);
    // A correction that makes the backward error grow beyond rounding noise
    // made the solution worse, however small ||d|| / ||u|| looks: an inner
    // solver too imprecise for K (fp32 with cond(K) eps_32 > 1) can inflate
    // u, and with it the denominator of the estimate.  Keep the current
    // iterate.
    if (!(bw_next <= std::max(4.0 * bw, options.backward_error_floor)))
      return finish(r);
    u.swap(next);
    r = std::move(r_next);
    bw = bw_next;
    ++rep.corrections;
    if (size <= options.tolerance || inf_norm(r) == 0.0) {
      rep.converged = true;
      return finish(r);
    }
    if (k > 0 && dn > options.min_contraction * previous)
      return finish(r); // stagnated
    previous = dn;
  }
  return finish(r);
}
bool indicates_mechanism(const RefinementReport &r, bool refined) {
  if (refined)
    return !r.converged && !(r.estimated_error <= 1e-2);
  return !(std::abs(r.epsilon) <= kMechanismEpsilon);
}

} // namespace vibestran
