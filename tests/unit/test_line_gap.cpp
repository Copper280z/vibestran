// tests/unit/test_line_gap.cpp
// The air-gap element of a model that repeats along a line (MPERIOD SHIFT):
// the strip between two lines of a MAG2D model and the tube between two
// cylinders of a MAGAXI one. The stiffness and the gap's field series are
// checked against the field energy of the series, exact harmonic fields
// (Bessel functions in the tube) and the limits of the tube's coefficients;
// solves against the same gap meshed as air, a stack moved by whole rows and
// the derivatives of the coenergy and linkage as the mover moves.

#include "magnetic/air_gap.hpp"
#include "tube_deck.hpp"
#ifdef VIBESTRAN_HAS_HDF5
#include "../common/hdf5_results.hpp"
#endif
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/SpecialFunctions>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

using namespace tube_deck;

namespace {

// A strip or tube gap built directly: sides at r1 and r2 of n_inner and
// n_outer grids over one period (unknowns 0.. and n_inner..), graded so that
// the spacing varies by 30%. The strip runs along x with y across it; the
// tube along z with r along x, as in a MAGAXI model in the basic x-z plane.
AirGap line_gap(GapShape shape, double r1, double r2, double length, int n_inner,
                int n_outer) {
  AirGap g;
  g.shape = shape;
  g.length = length;
  if (shape == GapShape::Tube) {
    g.u = Vector::UnitZ();
    g.v = Vector::UnitX();
  }
  auto side = [](double r, int n, int first, double phase) {
    GapCircle c;
    c.radius = r;
    for (int k = 0; k < n; ++k) {
      c.nodes.push_back(first + k);
      c.sign.push_back(1);
      const double s = 2 * pi * (k + phase) / n;
      c.angles.push_back(-pi + s + .3 * std::sin(s));
    }
    return c;
  };
  g.inner = side(r1, n_inner, 0, .1);
  g.outer = side(r2, n_outer, n_inner, .45);
  g.harmonics = std::max(n_inner, n_outer) / 2;
  return g;
}

// The point of grid k of side c.
Vector at(const AirGap &g, const GapCircle &c, size_t k) {
  return g.center + c.radius * g.v + c.angles[k] * g.length / (2 * pi) * g.u;
}

// Nodal values of A at the gap grids for a potential function of x.
Eigen::VectorXd sample(const AirGap &g, auto potential) {
  Eigen::VectorXd a(static_cast<Eigen::Index>(g.inner.nodes.size() + g.outer.nodes.size()));
  for (const GapCircle *c : {&g.inner, &g.outer})
    for (size_t k = 0; k < c->nodes.size(); ++k)
      a[c->nodes[k]] = potential(at(g, *c, k));
  return a;
}

// Gauss-Legendre points and weights on [a, b] (Golub-Welsch).
std::vector<std::pair<double, double>> gauss(int n, double a, double b) {
  Eigen::MatrixXd j = Eigen::MatrixXd::Zero(n, n);
  for (int k = 1; k < n; ++k)
    j(k, k - 1) = j(k - 1, k) = k / std::sqrt(4. * k * k - 1);
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(j);
  std::vector<std::pair<double, double>> out;
  for (int k = 0; k < n; ++k)
    out.emplace_back(a + (b - a) * (solver.eigenvalues()[k] + 1) / 2,
                     (b - a) * solver.eigenvectors()(0, k) * solver.eigenvectors()(0, k));
  return out;
}

// The energy of the series' field over the gap, per unit depth for a strip
// and in all for a tube: Gauss-Legendre across, the trapezoidal rule (exact
// for the trigonometric products) along one period.
double field_energy(const AirGap &g, const AirGapSeries &series) {
  double energy = 0;
  const int along = 512;
  for (auto [t, w] : gauss(24, g.inner.radius, g.outer.radius))
    for (int k = 0; k < along; ++k) {
      const Vector x = g.center + t * g.v + g.length * (k + .5) / along * g.u;
      const double measure = g.shape == GapShape::Tube ? 2 * pi * t : 1;
      energy += series.field(x).squaredNorm() / (2 * mu0) * measure * w * g.length / along;
    }
  return energy;
}

double i0(double x) { return Eigen::numext::bessel_i0e(x) * std::exp(x); }
double i1(double x) { return Eigen::numext::bessel_i1e(x) * std::exp(x); }
double k0(double x) { return Eigen::numext::bessel_k0e(x) * std::exp(-x); }
double k1(double x) { return Eigen::numext::bessel_k1e(x) * std::exp(-x); }

// An exact field in the gap and its potential: a uniform field (axial in a
// tube), a growing harmonic and a decaying one.
struct Exact {
  std::function<double(const Vector &)> potential;
  std::function<Vector(const Vector &)> field;
};
Exact exact_field(const AirGap &g) {
  const double b0 = .8, c1 = .3, c3 = .02, k1_ = 2 * pi / g.length, k3 = 3 * k1_;
  if (g.shape == GapShape::Tube) {
    // A_phi = b0 r / 2 + c1 I1(k1 r) cos(k1 z) + c3 K1(k3 r) sin(k3 z) / K1(k3 r1),
    // B_z = (1/r) d(r A)/dr = b0 + c1 k1 I0 cos - c3 k3 K0 sin / K1(k3 r1),
    // B_r = -dA/dz.
    const double scale = c3 / k1(k3 * g.inner.radius), grow = c1 / i1(k1_ * g.outer.radius);
    return {[=](const Vector &x) {
              const double r = x.x(), z = x.z();
              return b0 * r / 2 + grow * i1(k1_ * r) * std::cos(k1_ * z) +
                     scale * k1(k3 * r) * std::sin(k3 * z);
            },
            [=](const Vector &x) {
              const double r = x.x(), z = x.z();
              const double bz = b0 + grow * k1_ * i0(k1_ * r) * std::cos(k1_ * z) -
                                scale * k3 * k0(k3 * r) * std::sin(k3 * z);
              const double br = grow * k1_ * i1(k1_ * r) * std::sin(k1_ * z) -
                                scale * k3 * k1(k3 * r) * std::cos(k3 * z);
              return Vector(br, 0, bz);
            }};
  }
  // A = b0 y + c1 cosh(k1 y) cos(k1 x) + c3 sinh(k3 y) sin(k3 x), each harmonic
  // scaled to its value at the outer side; B = (dA/dy, -dA/dx).
  const double r2 = g.outer.radius, grow = c1 / std::cosh(k1_ * r2),
               other = c3 / std::sinh(k3 * r2);
  return {[=](const Vector &x) {
            return b0 * x.y() + grow * std::cosh(k1_ * x.y()) * std::cos(k1_ * x.x()) +
                   other * std::sinh(k3 * x.y()) * std::sin(k3 * x.x());
          },
          [=](const Vector &x) {
            const double dy = b0 + grow * k1_ * std::sinh(k1_ * x.y()) * std::cos(k1_ * x.x()) +
                              other * k3 * std::cosh(k3 * x.y()) * std::sin(k3 * x.x()),
                         dx = -grow * k1_ * std::cosh(k1_ * x.y()) * std::sin(k1_ * x.x()) +
                              other * k3 * std::sinh(k3 * x.y()) * std::cos(k3 * x.x());
            return Vector(dy, -dx, 0);
          }};
}

const char *name(GapShape shape) { return shape == GapShape::Tube ? "tube" : "strip"; }

} // namespace

TEST(LineGapStiffness, StripIsSymmetricWithConstantNullSpaceAndExactMeanTerm) {
  // Different constants on the two sides are a uniform field across the
  // strip, represented exactly: W = length (alpha - beta)^2 / (2 mu0 g).
  const double r1 = .002, r2 = .003, length = .04;
  const AirGap g = line_gap(GapShape::Strip, r1, r2, length, 40, 57);
  const Eigen::MatrixXd k = air_gap_stiffness(g);
  ASSERT_EQ(k.rows(), 97);
  EXPECT_LT((k - k.transpose()).norm(), 1e-12 * k.norm());
  EXPECT_LT((k * Eigen::VectorXd::Ones(97)).norm(), 1e-10 * k.norm());
  const Eigen::VectorXd eigenvalues =
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(k).eigenvalues();
  EXPECT_GT(eigenvalues.minCoeff(), -1e-10 * eigenvalues.maxCoeff());
  Eigen::VectorXd a(97);
  a.head(40).setConstant(.7);
  a.tail(57).setConstant(-.2);
  const double expected = length * .81 / (2 * mu0 * (r2 - r1));
  EXPECT_NEAR(.5 * a.dot(k * a), expected, 1e-12 * expected);
}

TEST(LineGapStiffness, TubeNullSpaceIsOneOverRAndAUniformAxialFieldIsExact) {
  // In a tube a constant A_phi is not field-free: c / r is. A uniform axial
  // field B0 is A_phi = B0 r / 2, constant on each side, so the element
  // holds it exactly: W = B0^2 / (2 mu0) pi (r2^2 - r1^2) length.
  const double r1 = .00675, r2 = .00775, length = .04, b0 = 1.3;
  const AirGap g = line_gap(GapShape::Tube, r1, r2, length, 40, 57);
  const Eigen::MatrixXd k = air_gap_stiffness(g);
  EXPECT_LT((k - k.transpose()).norm(), 1e-12 * k.norm());
  Eigen::VectorXd inverse(97), uniform(97), constant = Eigen::VectorXd::Ones(97);
  inverse.head(40).setConstant(1 / r1);
  inverse.tail(57).setConstant(1 / r2);
  uniform.head(40).setConstant(b0 * r1 / 2);
  uniform.tail(57).setConstant(b0 * r2 / 2);
  EXPECT_LT((k * inverse).norm(), 1e-12 * k.norm() * inverse.norm());
  // A constant c is the mean term alone, A = a r + b / r with
  // W = 2 pi length c^2 (r2 - r1) / (mu0 (r1 + r2)).
  const double constant_energy = 2 * pi * length * (r2 - r1) / (mu0 * (r1 + r2));
  EXPECT_NEAR(.5 * constant.dot(k * constant), constant_energy, 1e-12 * constant_energy);
  const Eigen::VectorXd eigenvalues =
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(k).eigenvalues();
  EXPECT_GT(eigenvalues.minCoeff(), -1e-10 * eigenvalues.maxCoeff());
  const double expected = b0 * b0 / (2 * mu0) * pi * (r2 * r2 - r1 * r1) * length;
  EXPECT_NEAR(.5 * uniform.dot(k * uniform), expected, 1e-12 * expected);
}

TEST(LineGapStiffness, EnergyIsTheFieldEnergyOfTheSeries) {
  // The stiffness comes from each harmonic's boundary flux (the Wronskian of
  // I1 and K1 in a tube); the series' field from the radial functions
  // themselves. For any traces the energy (1/2) a^T K a must be the volume
  // integral of B^2 / (2 mu0) of the series' field, harmonics, mean terms
  // and couplings alike. The traces are not harmonic solutions and the two
  // sides have different grids.
  for (GapShape shape : {GapShape::Strip, GapShape::Tube}) {
    SCOPED_TRACE(name(shape));
    const AirGap g = line_gap(shape, .00675, .00775, .04, 40, 57);
    const Eigen::VectorXd a = sample(g, [&](const Vector &x) {
      const double s = 2 * pi * x.dot(g.u) / g.length, t = x.dot(g.v);
      return std::exp(std::sin(s)) * (1 + 40 * t) + .3 * std::cos(7 * s + 200 * t);
    });
    const double energy = .5 * a.dot(air_gap_stiffness(g) * a);
    EXPECT_NEAR(field_energy(g, AirGapSeries(g, a)), energy, 1e-9 * energy);
  }
}

TEST(LineGapStiffness, TubeTakesItsLimitsFarFromTheAxisAndAcrossAThickGap) {
  // Far from the axis a tube is a strip wrapped around it: its coefficients
  // per unit circumference (2 pi r at each side, the geometric mean for the
  // coupling) approach the strip's with an error of order gap / radius.
  const double length = .04, gap = .001;
  for (int m : {0, 1, 5, 20}) {
    SCOPED_TRACE(m);
    const auto strip =
        harmonic_stiffness(line_gap(GapShape::Strip, 0, gap, length, 4, 4), m);
    double previous = 1;
    for (double radius : {.1, 1., 10.}) {
      const auto tube = harmonic_stiffness(
          line_gap(GapShape::Tube, radius, radius + gap, length, 4, 4), m);
      const double c = 2 * pi * radius, error = std::max(
          {std::abs(tube.inner / c - strip.inner) / strip.inner,
           std::abs(tube.outer / (c + 2 * pi * gap) - strip.outer) / strip.outer,
           std::abs(tube.mutual / std::sqrt(c * (c + 2 * pi * gap)) - strip.mutual) /
               std::abs(strip.mutual)});
      EXPECT_LT(error, 2 * gap / radius) << radius;
      EXPECT_LT(error, previous / 5) << radius;
      previous = error;
    }
  }
  // Across a gap many wavelengths thick the sides decouple, and each sees
  // the field of a lone cylinder: inside a cylinder at r2 the growing
  // I1 (Dirichlet-to-Neumann x I0/I1), outside one at r1 the decaying K1
  // (x K0/K1), times pi length / mu0. Unscaled, I1 at the outer side would
  // overflow (x2 up to 6e4).
  const double r1 = .001, r2 = .5, thick = .005;
  const AirGap g = line_gap(GapShape::Tube, r1, r2, thick, 4, 4);
  for (int m : {1, 10, 100}) {
    SCOPED_TRACE(m);
    const auto s = harmonic_stiffness(g, m);
    const double kappa = 2 * pi * m / thick, x1 = kappa * r1, x2 = kappa * r2,
                 scale = pi * thick / mu0;
    using Eigen::numext::bessel_i0e, Eigen::numext::bessel_i1e,
        Eigen::numext::bessel_k0e, Eigen::numext::bessel_k1e;
    const double inner = scale * x1 * bessel_k0e(x1) / bessel_k1e(x1),
                 outer = scale * x2 * bessel_i0e(x2) / bessel_i1e(x2);
    ASSERT_TRUE(std::isfinite(s.inner) && std::isfinite(s.outer) && std::isfinite(s.mutual));
    EXPECT_NEAR(s.inner, inner, 1e-12 * inner);
    EXPECT_NEAR(s.outer, outer, 1e-12 * outer);
    EXPECT_LT(std::abs(s.mutual), 1e-100 * inner);
  }
}

TEST(LineGapSeries, PotentialIsTheSeriesWhoseCurlIsTheField) {
  // The field the series returns must be the curl of the potential it
  // returns, by central differences: a strip's B = (dA/dy, -dA/dx), a
  // tube's B_r = -dA/dz and B_z = (1/r) d(r A)/dr.
  for (GapShape shape : {GapShape::Strip, GapShape::Tube}) {
    SCOPED_TRACE(name(shape));
    const AirGap g = line_gap(shape, .00675, .00775, .04, 48, 64);
    const Eigen::VectorXd a = sample(g, [&](const Vector &x) {
      const double s = 2 * pi * x.dot(g.u) / g.length, t = x.dot(g.v);
      return .01 * (std::exp(std::sin(s)) * (1 + 40 * t) + .3 * std::cos(5 * s));
    });
    const AirGapSeries series(g, a);
    double worst = 0;
    for (double s : {.001, .013, .029})
      for (double t : {.25, .5, .75}) {
        const double r = g.inner.radius + t * (g.outer.radius - g.inner.radius),
                     h = 1e-7;
        const Vector x = r * g.v + s * g.u;
        const auto potential = [&](const Vector &dx) { return series.potential(x + dx); };
        Vector curl;
        if (shape == GapShape::Strip) {
          curl = Vector((potential(h * g.v) - potential(-h * g.v)) / (2 * h),
                        -(potential(h * g.u) - potential(-h * g.u)) / (2 * h), 0);
        } else {
          const double d_rz = ((r + h) * potential(h * g.v) - (r - h) * potential(-h * g.v)) /
                              (2 * h * r);
          curl = -(potential(h * g.u) - potential(-h * g.u)) / (2 * h) * g.v + d_rz * g.u;
        }
        const Vector b = series.field(x);
        worst = std::max(worst, (curl - b).norm() / b.norm());
      }
    EXPECT_LT(worst, 1e-6);
  }
}

TEST(LineGapSeries, ConvergesToAnExactFieldAtSecondOrder) {
  // Traces sampled from an exact field (a tube's in I1 and K1): the series'
  // potential and field inside the gap, and the element's energy, converge
  // to the exact ones at O(h^2), the piecewise-linear trace's error. The
  // exact energy is the volume integral of the exact field.
  for (GapShape shape : {GapShape::Strip, GapShape::Tube}) {
    SCOPED_TRACE(name(shape));
    const AirGap reference = line_gap(shape, .00675, .00775, .04, 8, 8);
    const Exact exact = exact_field(reference);
    double exact_energy = 0;
    for (auto [t, w] : gauss(24, reference.inner.radius, reference.outer.radius))
      for (int k = 0; k < 512; ++k) {
        const Vector x = t * reference.v + reference.length * (k + .5) / 512 * reference.u;
        exact_energy += exact.field(x).squaredNorm() / (2 * mu0) *
                        (shape == GapShape::Tube ? 2 * pi * t : 1) * w *
                        reference.length / 512;
      }
    double previous[3] = {1, 1, 1};
    for (int n : {32, 64, 128}) {
      SCOPED_TRACE(n);
      const AirGap g = line_gap(shape, .00675, .00775, .04, n, n * 3 / 2);
      const Eigen::VectorXd a = sample(g, exact.potential);
      const AirGapSeries series(g, a);
      double potential = 0, field = 0;
      for (double s : {.001, .013, .029})
        for (double t : {.25, .5, .75}) {
          const Vector x = (g.inner.radius + t * (g.outer.radius - g.inner.radius)) * g.v +
                           s * g.u;
          potential = std::max(potential, std::abs(series.potential(x) - exact.potential(x)) /
                                              std::abs(exact.potential(x)));
          field = std::max(field, (series.field(x) - exact.field(x)).norm() /
                                      exact.field(x).norm());
        }
      const double energy =
          std::abs(.5 * a.dot(air_gap_stiffness(g) * a) - exact_energy) / exact_energy;
      EXPECT_LT(potential, previous[0] / 3.5);
      EXPECT_LT(field, previous[1] / 3.5);
      EXPECT_LT(energy, previous[2] / 3.5);
      previous[0] = potential;
      previous[1] = field;
      previous[2] = energy;
    }
    EXPECT_LT(previous[1], 2e-3);
    EXPECT_LT(previous[2], 2e-3);
  }
}

TEST(LineGapForce, IsMinusTheEnergysDerivativeForAMoveOfTheInnerSide) {
  // Moving the inner side's grids by delta along the gap, their potentials
  // held, changes the energy at the rate -F. The force is bilinear in the
  // potentials, so its gradient . a is 2 F.
  for (GapShape shape : {GapShape::Strip, GapShape::Tube}) {
    SCOPED_TRACE(name(shape));
    const AirGap g = line_gap(shape, .00675, .00775, .04, 40, 57);
    const Eigen::VectorXd a = sample(g, [&](const Vector &x) {
      const double s = 2 * pi * x.dot(g.u) / g.length, t = x.dot(g.v);
      return .01 * (std::sin(s + 300 * t) + .4 * std::cos(3 * s - 100 * t));
    });
    const double force = air_gap_torque(g, a), delta = 1e-6;
    auto energy = [&](double move) {
      AirGap moved = g;
      std::ranges::transform(moved.inner.angles, moved.inner.angles.begin(),
                             [&](double angle) { return angle + 2 * pi * move / g.length; });
      return .5 * a.dot(air_gap_stiffness(moved) * a);
    };
    const double derivative = (energy(delta) - energy(-delta)) / (2 * delta);
    EXPECT_GT(std::abs(force), 1e-3);
    EXPECT_NEAR(force, -derivative, 1e-6 * std::abs(force));
    EXPECT_NEAR(air_gap_torque_gradient(g, a).dot(a), 2 * force, 1e-12 * std::abs(force));
  }
}

TEST(LineGapSolve, AgeConvergesToTheMeshedGap) {
  // The tubular and flat stacks, loaded, with the gap solved by the element
  // and as meshed air (refined with the rest of the mesh): energy, linkage
  // and force (the element's on the mover; MFORCE's on the stator, through
  // the meshed gap) agree, and their differences fall at second order.
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    double previous[3] = {1, 1, 1};
    for (int refine : {1, 2}) {
      Tube age{.rows = 48 * refine, .offset = 5 * refine, .layers = 0, .current = 3,
               .gap = true, .planar = planar, .refine = refine};
      Tube mesh = age;
      mesh.mesh = true;
      mesh.layers = 2;
      mesh.stator_body = true;
      const auto a = solved(parsed(tube(age)));
      const auto b = solved(parsed(tube(mesh)));
      ASSERT_EQ(a.cases.size(), 1U);
      ASSERT_EQ(b.cases.size(), 1U);
      ASSERT_TRUE(a.mesh.air_gap.has_value());
      EXPECT_EQ(a.mesh.air_gap->shape, planar ? GapShape::Strip : GapShape::Tube);
      EXPECT_FALSE(b.mesh.air_gap.has_value());
      const auto &ca = a.cases[0], &cb = b.cases[0];
      const double energy = relative(ca.coenergy, cb.coenergy),
                   linkage = relative(ca.circuits[0].linkage, cb.circuits[0].linkage),
                   force = relative(ca.gap_force, -cb.forces[0].force.z() * age.machine);
      EXPECT_LT(energy, (refine == 1 ? 1 : .35) * previous[0]);
      EXPECT_LT(linkage, (refine == 1 ? 1 : .35) * previous[1]);
      EXPECT_LT(force, (refine == 1 ? 1 : .35) * previous[2]);
      previous[0] = energy;
      previous[1] = linkage;
      previous[2] = force;
    }
    EXPECT_LT(previous[0], 5e-5);
    EXPECT_LT(previous[1], 3e-5);
    EXPECT_LT(previous[2], 2e-4);
  }
}

TEST(LineGapSolve, MovingTheMoverByARowIsTheStackOffsetByARow) {
  // The element sees the mover only through its side's trace. Moving the
  // mover by one row (its grids, the gap's elements sheared) puts its trace
  // on the same phases as the stack whose magnets start a row later, the
  // grids past the top side standing for their images: the same discrete
  // problem, so energy, linkage and force agree to round-off.
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    Tube moved{.offset = 5, .layers = 0, .current = 3, .gap = true, .planar = planar};
    moved.move = pitch / moved.rows;
    Tube offset = moved;
    offset.move = 0;
    offset.offset = 6;
    const auto a = solved(parsed(tube(moved)));
    const auto b = solved(parsed(tube(offset)));
    ASSERT_EQ(a.cases.size(), 1U);
    ASSERT_EQ(b.cases.size(), 1U);
    const auto &ca = a.cases[0], &cb = b.cases[0];
    EXPECT_NEAR(ca.coenergy, cb.coenergy, 1e-10 * std::abs(cb.coenergy));
    EXPECT_NEAR(ca.gap_force, cb.gap_force, 1e-9 * std::abs(cb.gap_force));
    for (size_t c = 0; c < 3; ++c)
      EXPECT_NEAR(ca.circuits[c].linkage, cb.circuits[c].linkage,
                  1e-10 * std::abs(cb.circuits[0].linkage));
  }
}

TEST(LineGapSolve, ForceAndLinkageRateAreTheDerivativesOfCoenergyAndLinkage) {
  // At fixed currents, moving the mover by dz changes the whole stack's
  // coenergy at the rate of the element's force on it, and each circuit's
  // linkage at its linkage rate (the back-EMF per unit speed, from the same
  // solve by reciprocity). Central differences over decks with the mover
  // moved by +-dz; the gap's elements, which the element does not use,
  // shear. The derivative of the discrete coenergy is exactly the element's
  // force, so they agree to the difference's error, not the mesh's.
  constexpr double dz = 1e-6;
  for (bool planar : {false, true}) {
    SCOPED_TRACE(planar ? "strip" : "tube");
    Tube t{.offset = 5, .layers = 0, .current = 3, .gap = true, .planar = planar};
    std::vector<Results> runs;
    for (int shift : {-1, 0, 1}) {
      t.move = shift * dz;
      runs.push_back(solved(parsed(tube(t))));
      ASSERT_EQ(runs.back().cases.size(), 1U);
    }
    const auto &c = runs[1].cases[0];
    const double force = (runs[2].cases[0].coenergy - runs[0].cases[0].coenergy) / (2 * dz);
    EXPECT_GT(std::abs(force), 1.);
    EXPECT_NEAR(c.gap_force, force, 1e-6 * std::abs(force));
    for (size_t k = 0; k < 3; ++k) {
      const double rate = (runs[2].cases[0].circuits[k].linkage -
                           runs[0].cases[0].circuits[k].linkage) /
                          (2 * dz);
      EXPECT_NEAR(c.linkage_rate[static_cast<Eigen::Index>(k)], rate,
                  1e-6 * std::abs(c.linkage_rate[0]))
          << k;
    }
  }
}

TEST(LineGapSolve, ResultsRecordTheGap) {
  Tube t{.offset = 5, .layers = 0, .current = 3, .gap = true};
  const auto m = parsed(tube(t));
  const auto results = solved(m);
  ASSERT_TRUE(results.mesh.air_gap.has_value());
  const auto &gap = *results.mesh.air_gap;
  EXPECT_EQ(gap.shape, GapShape::Tube);
  EXPECT_NEAR(gap.length, 2 * pitch, 1e-15);
  EXPECT_EQ(gap.sectors, 2);
  EXPECT_NEAR(gap.inner.radius, .00675, 1e-15);
  EXPECT_NEAR(gap.outer.radius, .00775, 1e-15);
  // The pitch's 48 rows, unfolded over two pitches.
  EXPECT_EQ(gap.inner.nodes.size(), 96U);
  EXPECT_TRUE(std::isnan(results.cases[0].gap_torque));
#ifdef VIBESTRAN_HAS_HDF5
  const auto directory = std::filesystem::temp_directory_path() / "vibestran_line_gap";
  std::filesystem::create_directories(directory);
  ASSERT_FALSE(write_results(m, results, directory / "tube.F06").has_value());
  const auto file = directory / "tube.mag.vtkhdf";
  using vibestran::test::h5_attribute, vibestran::test::h5_exists;
  EXPECT_NEAR(h5_attribute(file, "/MAGSTAT/air_gap", "length_m"), 2 * pitch, 1e-15);
  EXPECT_NEAR(h5_attribute(file, "/MAGSTAT/air_gap", "inner_radius_m"), .00675, 1e-15);
  EXPECT_NEAR(h5_attribute(file, "/MAGSTAT/subcases/1", "air_gap_force_N"),
              results.cases[0].gap_force, 1e-12 * std::abs(results.cases[0].gap_force));
  EXPECT_TRUE(h5_exists(file, "/MAGSTAT/subcases/1/circuits/linkage_rate_Wb_per_m"));
  EXPECT_FALSE(h5_exists(file, "/MAGSTAT/subcases/1/circuits/linkage_rate_Wb_per_rad"));
  std::ifstream in(directory / "tube.F06");
  std::stringstream f06;
  f06 << in.rdbuf();
  for (const char *text :
       {"PAIRGAP 9, air-gap element (AGE) in place of its 192 elements: the tube from "
        "radius 0.00675 to 0.00775",
        "96 + 96 grids over a 0.04 m period, 48 coupled harmonics",
        "the pitch's lines (49 + 49 grids) continued over two pitches, changing sign",
        "back-EMF and air-gap force are the whole",
        "Air-gap force on the part on the gap's inner side, along the axis (N): "})
    EXPECT_NE(f06.str().find(text), std::string::npos) << text << "\n" << f06.str();
  std::filesystem::remove_all(directory);
#else
  GTEST_SKIP() << "MAGSTAT results need HDF5";
#endif
}

TEST(LineGapCards, ChecksNameWhatFails) {
  const auto message = [](const std::string &text) {
    return error_of(vibestran::magnetic::parse_string(text));
  };
  const auto solve_message = [](const std::string &text) {
    auto m = vibestran::magnetic::parse_string(text);
    if (const auto *e = std::get_if<vibestran::Error>(&m))
      return e->message;
    return error_of(solve(std::get<Model>(m)));
  };
  Tube t{.offset = 5, .layers = 0, .gap = true};
  const std::string base = deck(tube(t));
  // A MAGAXI gap needs the model to repeat along the axis.
  const std::string unperiodic =
      std::regex_replace(base, std::regex("MPERIOD,[^\n]*\n"), "");
  EXPECT_NE(message(unperiodic).find("PAIRGAP 9 in a MAGAXI model is a tube along the "
                                     "axis"),
            std::string::npos)
      << message(unperiodic);
  // The sliding band along a line is not built yet.
  const std::string band =
      std::regex_replace(base, std::regex("PAIRGAP,9\n"), "PAIRGAP,9,BAND\n");
  EXPECT_NE(message(band).find("the sliding band (BAND) of PAIRGAP 9 is not available"),
            std::string::npos)
      << message(band);
  // A grid of the outer side (ring 11, at 7.75 mm) off the cylinder.
  const int rings = static_cast<int>(tube(t).r.size()), grid = 12 + rings * 10;
  const std::string bent = std::regex_replace(
      base, std::regex(std::format("GRID,{},,[^,]*,", grid)),
      std::format("GRID,{},,.0077501,", grid));
  EXPECT_NE(solve_message(bent).find(std::format(
                "PAIRGAP 9: the outer side of the tubular air gap is not at one radius: "
                "grid {}",
                grid)),
            std::string::npos)
      << solve_message(bent);
}
