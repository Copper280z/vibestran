// src/solver/iterative_refinement.cpp
// See include/solver/iterative_refinement.hpp.

#include "solver/iterative_refinement.hpp"
#include "core/exceptions.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#ifdef HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

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

template <class Body> void for_rows(std::size_t n, const Body &body) {
#ifdef HAVE_TBB
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n, 4096),
                    [&](const tbb::blocked_range<std::size_t> &r) {
                      for (std::size_t i = r.begin(); i != r.end(); ++i)
                        body(i);
                    });
#else
  for (std::size_t i = 0; i < n; ++i)
    body(i);
#endif
}

/// The rows of K as stored and, for lower-triangular storage, its strictly
/// lower entries by column: row i of the symmetric matrix is row i as stored
/// followed by column i below the diagonal.  Each row is then accumulated by
/// one thread in the order a serial sweep of the stored rows would use, so
/// the parallel results equal the serial ones bit for bit.
class SymmetricRows {
public:
  explicit SymmetricRows(const SparseMatrixBuilder::CsrData &K) : K_(K) {
    if (!K.stores_lower_triangle_only())
      return;
    const auto n = static_cast<std::size_t>(K.n);
    std::vector<int> count(n + 1, 0);
    for_rows(n, [&](std::size_t i) {
      for (int p = K.row_ptr[i]; p < K.row_ptr[i + 1]; ++p) {
        const auto j = static_cast<std::size_t>(K.col_ind[static_cast<std::size_t>(p)]);
        if (j != i)
          std::atomic_ref<int>(count[j + 1]).fetch_add(1, std::memory_order_relaxed);
      }
    });
    std::partial_sum(count.begin(), count.end(), count.begin());
    start_ = count;
    // Not zeroed: the fill below touches every entry, in parallel.
    below_ = std::make_unique_for_overwrite<Entry[]>(static_cast<std::size_t>(start_[n]));
    for_rows(n, [&](std::size_t i) {
      for (int p = K.row_ptr[i]; p < K.row_ptr[i + 1]; ++p) {
        const auto j = static_cast<std::size_t>(K.col_ind[static_cast<std::size_t>(p)]);
        if (j != i) {
          const int at = std::atomic_ref<int>(count[j]).fetch_add(1, std::memory_order_relaxed);
          below_[static_cast<std::size_t>(at)] = {static_cast<int>(i), p};
        }
      }
    });
    // Positions increase with the row, as in a serial sweep.  A column holds
    // a few dozen entries at most, so insertion sort.
    for_rows(n, [&](std::size_t j) {
      const auto first = static_cast<std::size_t>(start_[j]);
      const auto last = static_cast<std::size_t>(start_[j + 1]);
      for (std::size_t e = first + 1; e < last; ++e)
        for (std::size_t f = e; f > first && below_[f - 1].position > below_[f].position; --f)
          std::swap(below_[f - 1], below_[f]);
    });
  }

  /// f(value, column) for each entry of row i of the symmetric matrix.
  template <class F> void row(std::size_t i, const F &f) const {
    for (int p = K_.row_ptr[i]; p < K_.row_ptr[i + 1]; ++p)
      f(K_.values[static_cast<std::size_t>(p)],
        static_cast<std::size_t>(K_.col_ind[static_cast<std::size_t>(p)]));
    if (start_.empty())
      return;
    for (int q = start_[i]; q < start_[i + 1]; ++q) {
      const Entry &e = below_[static_cast<std::size_t>(q)];
      f(K_.values[static_cast<std::size_t>(e.position)], static_cast<std::size_t>(e.row));
    }
  }

  const SparseMatrixBuilder::CsrData &matrix() const { return K_; }

private:
  struct Entry {
    int row;
    int position;
  };
  const SparseMatrixBuilder::CsrData &K_;
  std::vector<int> start_;
  std::unique_ptr<Entry[]> below_;
};

/// ||K||_inf, reading lower-triangular storage as symmetric.
double matrix_inf_norm(const SymmetricRows &K) {
  std::vector<double> rows(static_cast<std::size_t>(K.matrix().n), 0.0);
  for_rows(rows.size(), [&](std::size_t i) {
    double sum = 0.0;
    K.row(i, [&](double a, std::size_t) { sum += std::abs(a); });
    rows[i] = sum;
  });
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

namespace {

std::vector<double> compensated_residual(const SymmetricRows &rows, std::span<const double> x,
                                         std::span<const double> F) {
  const auto n = static_cast<std::size_t>(rows.matrix().n);
  if (x.size() != n || F.size() != n)
    throw SolverError(std::format("compensated_residual: sizes {} / {} do not match the "
                                  "{}x{} matrix",
                                  x.size(), F.size(), n, n));
  std::vector<double> r(n);
  for_rows(n, [&](std::size_t i) {
    double s = F[i]; // running sum
    double c = 0.0;  // accumulated rounding errors
    rows.row(i, [&](double a, std::size_t j) { subtract_product(s, c, a, x[j]); });
    r[i] = s + c;
  });
  return r;
}

} // namespace

std::vector<double> compensated_residual(const SparseMatrixBuilder::CsrData &K,
                                         std::span<const double> x,
                                         std::span<const double> F) {
  return compensated_residual(SymmetricRows(K), x, F);
}

RefinementReport refine_solution(const SparseMatrixBuilder::CsrData &K, std::span<const double> F,
                                 std::vector<double> &u, const CorrectionSolve &solve,
                                 const RefinementOptions &options) {
  RefinementReport rep;
  const SymmetricRows rows(K);
  const double normK = matrix_inf_norm(rows);
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
  std::vector<double> r = compensated_residual(rows, u, F);
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
    std::vector<double> r_next = compensated_residual(rows, next, F);
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
  if (!(r.relative_residual <= kMechanismResidual))
    return true;
  if (refined)
    return !r.converged && !(r.estimated_error <= 1e-2);
  return !(std::abs(r.epsilon) <= kMechanismEpsilon);
}

} // namespace vibestran
