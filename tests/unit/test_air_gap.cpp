// tests/unit/test_air_gap.cpp
// Air-gap element (AGE) for planar MAGSTAT: the harmonic transforms and
// stiffness, the PAIRGAP property and the validation of its region, and
// solutions against analytic fields and the same gap solved as meshed air.
// Models are written as deck text and parsed, as a user would run them.

#include "magnetic/air_gap.hpp"
#include "magnetic/solver.hpp"
#include "machine_deck.hpp"
#include "solver/solver_backend.hpp"
#ifdef VIBESTRAN_HAS_HDF5
#include "../common/hdf5_results.hpp"
#endif
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numbers>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace vibestran::magnetic;
using namespace machine_deck;
using vibestran::Error;

namespace {

// A gap built directly: circles of n_inner and n_outer grids (indices
// 0.. and n_inner..) about the origin, smoothly graded so that the spacing
// varies by 30% around each circle.
AirGap direct_gap(double r1, double r2, int n_inner, int n_outer) {
  AirGap g;
  auto circle = [](double r, int n, int first, double phase) {
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
  g.inner = circle(r1, n_inner, 0, .1);
  g.outer = circle(r2, n_outer, n_inner, .45);
  g.harmonics = std::min(n_inner, n_outer) / 2;
  return g;
}

// Nodal values of A at the gap grids for a potential function of x.
Eigen::VectorXd sample(const AirGap &g, auto potential) {
  Eigen::VectorXd a(static_cast<Eigen::Index>(g.inner.nodes.size() + g.outer.nodes.size()));
  for (const GapCircle *c : {&g.inner, &g.outer})
    for (size_t k = 0; k < c->nodes.size(); ++k)
      a[c->nodes[k]] = potential(
          Vector(c->radius * std::cos(c->angles[k]), c->radius * std::sin(c->angles[k]), 0));
  return a;
}

} // namespace

TEST(AirGapStiffness, TransformIsTheFourierSeriesOfTheInterpolant) {
  const AirGap g = direct_gap(1, 1.1, 13, 17);
  const auto &c = g.inner;
  const int harmonics = 6;
  const Eigen::MatrixXd t = gap_transform(c, harmonics);
  Eigen::VectorXd values(13);
  for (int k = 0; k < 13; ++k)
    values[k] = std::cos(2.1 * k) + .3 * k;
  const Eigen::VectorXd coefficients = t * values;
  // The piecewise-linear interpolant in angle, integrated by the midpoint
  // rule on a fine grid over one period from the first grid.
  auto f = [&](double angle) {
    double x = angle;
    while (x >= c.angles[0] + 2 * pi)
      x -= 2 * pi;
    size_t k = 0;
    while (k + 1 < c.angles.size() && c.angles[k + 1] <= x)
      ++k;
    const size_t next = (k + 1) % c.angles.size();
    const double w = next ? c.angles[next] : c.angles[0] + 2 * pi;
    const double s = (x - c.angles[k]) / (w - c.angles[k]);
    return (1 - s) * values[static_cast<Eigen::Index>(k)] +
           s * values[static_cast<Eigen::Index>(next)];
  };
  const int steps = 400000;
  Eigen::VectorXd expected = Eigen::VectorXd::Zero(1 + 2 * harmonics);
  for (int i = 0; i < steps; ++i) {
    const double x = c.angles[0] + 2 * pi * (i + .5) / steps, w = 2 * pi / steps;
    expected[0] += f(x) * w / (2 * pi);
    for (int m = 1; m <= harmonics; ++m) {
      expected[2 * m - 1] += f(x) * std::cos(m * x) * w / pi;
      expected[2 * m] += f(x) * std::sin(m * x) * w / pi;
    }
  }
  EXPECT_LT((coefficients - expected).norm(), 1e-8 * expected.norm())
      << coefficients.transpose() << "\n" << expected.transpose();
}

TEST(AirGapStiffness, SymmetricWithConstantNullSpaceAndExactMeanTerm) {
  const double r1 = .02, r2 = .021;
  const AirGap g = direct_gap(r1, r2, 40, 57);
  const Eigen::MatrixXd k = air_gap_stiffness(g);
  ASSERT_EQ(k.rows(), 97);
  EXPECT_LT((k - k.transpose()).norm(), 1e-12 * k.norm());
  // A constant potential carries no field.
  EXPECT_LT((k * Eigen::VectorXd::Ones(97)).norm(), 1e-10 * k.norm());
  const Eigen::VectorXd eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(k).eigenvalues();
  EXPECT_GT(eigenvalues.minCoeff(), -1e-10 * eigenvalues.maxCoeff());
  // A different constant on each circle is a logarithmic field, represented
  // exactly: W = pi (alpha - beta)^2 / (mu0 ln(r2/r1)).
  Eigen::VectorXd a(97);
  a.head(40).setConstant(.7);
  a.tail(57).setConstant(-.2);
  const double expected = pi * .81 / (mu0 * std::log(r2 / r1));
  EXPECT_NEAR(.5 * a.dot(k * a), expected, 1e-12 * expected);
}

TEST(AirGapStiffness, UniformFieldEnergyAndGapFieldConvergeAtSecondOrder) {
  // A = B0 . (y, -x) gives B = (B0, 0.4 B0); its energy in the annulus is
  // |B|^2 / (2 mu0) * pi (r2^2 - r1^2). The traces are interpolated
  // piecewise-linearly, so the AGE converges at O(h^2). The constant is
  // large: with different divisions on the two circles their interpolation
  // errors differ, and a thin gap's energy depends on the difference of the
  // traces, amplifying it by about r / gap = 20. (In a solve the meshed
  // regions adjust the traces instead; see the magnetized rotor test.)
  const double r1 = .02, r2 = .021, b0 = .8;
  const Vector field(b0, .4 * b0, 0);
  const double exact = field.squaredNorm() / (2 * mu0) * pi * (r2 * r2 - r1 * r1);
  double previous_energy = 1, previous_field = 1;
  for (int n : {32, 64, 128}) {
    const AirGap g = direct_gap(r1, r2, n, n * 3 / 2);
    const Eigen::VectorXd a =
        sample(g, [&](const Vector &x) { return field.x() * x.y() - field.y() * x.x(); });
    const double energy = .5 * a.dot(air_gap_stiffness(g) * a);
    double field_error = 0;
    // Inside the gap: on the circles the tangential derivative of a
    // piecewise-linear trace is only first order.
    for (double t : {0., 1., 2.5, 4.})
      for (double r : {r1 + .25 * (r2 - r1), (r1 + r2) / 2, r1 + .75 * (r2 - r1)})
        field_error = std::max(
            field_error,
            (AirGapSeries(g, a).field(Vector(r * std::cos(t), r * std::sin(t), 0)) - field)
                    .norm() /
                field.norm());
    const double energy_error = std::abs(energy - exact) / exact;
    EXPECT_LT(energy_error, previous_energy / 3.5) << n;
    EXPECT_LT(field_error, previous_field / 3.5) << n;
    previous_energy = energy_error;
    previous_field = field_error;
  }
  EXPECT_LT(previous_energy, 3e-3);
  EXPECT_LT(previous_field, 4e-3);
}

TEST(AirGapStiffness, FinerCircleKeepsTheStiffnessOfWavelengthsTheCoarserCannotResolve) {
  // The outer circle carries harmonic 40, which it resolves (1024 grids)
  // and inner circles of 16 to 64 grids cannot; the inner trace is zero.
  // With the inner circle held at zero the annulus energy is the outer
  // self term alone, pi n / (2 mu0) coth(n L) per unit amplitude squared,
  // whatever the inner circle's grid count. Cutting the sum at the coarser
  // circle's resolution (half its grid count, NHARM's old blank) gave the
  // outer circle no stiffness at all at this wavelength, and a motor mesh
  // graded toward the stator 5.5% too much cogging.
  const double r1 = .02, r2 = .021, l = std::log(r2 / r1);
  const int n = 40;
  const double exact = pi * n / (2 * mu0) / std::tanh(n * l);
  std::vector<double> energies;
  for (int inner : {16, 32, 64}) {
    SCOPED_TRACE(inner);
    const AirGap g = direct_gap(r1, r2, inner, 1024);
    ASSERT_GT(2 * n, inner);
    const Eigen::VectorXd a = sample(g, [&](const Vector &x) {
      return x.head<2>().norm() < (r1 + r2) / 2 ? 0. : std::cos(n * std::atan2(x.y(), x.x()));
    });
    energies.push_back(.5 * a.dot(air_gap_stiffness(g) * a));
    // The piecewise-linear trace of cos(40 t) on 1024 graded grids is 1.1%
    // short of the exact energy.
    EXPECT_NEAR(energies.back(), exact, 2e-2 * exact);
    EXPECT_NEAR(energies.back(), energies.front(), 1e-12 * exact);
  }
}

TEST(AirGapStiffness, PotentialIsTheSeriesWhoseCurlIsTheGapField) {
  // A harmonic potential with a mean (log) term, a growing and a decaying
  // harmonic: A = b0 [r1 ln(r/r1) + (x^2 - y^2) / (2 r1) + r1^2 y / r^2].
  // The series' potential must be the one whose curl its field returns
  // (B = (dA/dy, -dA/dx), checked by central differences), and it must
  // converge to the exact A at O(h^2) inside the gap.
  const double r1 = .02, r2 = .021, b0 = .8;
  auto exact = [&](const Vector &x) {
    return b0 * (r1 * std::log(x.head<2>().norm() / r1) +
                 (x.x() * x.x() - x.y() * x.y()) / (2 * r1) +
                 r1 * r1 * x.y() / x.head<2>().squaredNorm());
  };
  double previous = 1;
  for (int n : {32, 64, 128}) {
    const AirGap g = direct_gap(r1, r2, n, n * 3 / 2);
    const Eigen::VectorXd a = sample(g, exact);
    const AirGapSeries series(g, a);
    double curl_error = 0, error = 0;
    for (double t : {0., 1., 2.5, 4.})
      for (double r : {r1 + .25 * (r2 - r1), (r1 + r2) / 2, r1 + .75 * (r2 - r1)}) {
        const Vector x(r * std::cos(t), r * std::sin(t), 0);
        const double h = 1e-6 * (r2 - r1);
        auto potential = [&](double dx, double dy) {
          return series.potential(x + Vector(dx, dy, 0));
        };
        const Vector curl((potential(0, h) - potential(0, -h)) / (2 * h),
                          -(potential(h, 0) - potential(-h, 0)) / (2 * h), 0);
        const Vector b = series.field(x);
        curl_error = std::max(curl_error, (curl - b).norm() / b.norm());
        error = std::max(error, std::abs(series.potential(x) - exact(x)) / (b0 * r1));
      }
    SCOPED_TRACE(n);
    EXPECT_LT(curl_error, 1e-6);
    EXPECT_LT(error, previous / 3.5);
    previous = error;
  }
  EXPECT_LT(previous, 5e-4); // the r / gap amplification of the test above
}

// The sliding band (METHOD=BAND) of direct_gap's circles on n points.
AirGap band_gap(double r1, double r2, int n_inner, int n_outer, int n) {
  AirGap g = direct_gap(r1, r2, n_inner, n_outer);
  g.band_points = n;
  return g;
}

TEST(AirGapBand, SparseSymmetricWithConstantNullSpaceAndExactMeanTerm) {
  const double r1 = .02, r2 = .021;
  const AirGap g = band_gap(r1, r2, 40, 57, 90);
  const Eigen::SparseMatrix<double> sparse = band_stiffness(g);
  const Eigen::MatrixXd k(sparse);
  ASSERT_EQ(k.rows(), 97);
  EXPECT_LT((k - k.transpose()).norm(), 1e-12 * k.norm());
  EXPECT_LT((k * Eigen::VectorXd::Ones(97)).norm(), 1e-10 * k.norm());
  const Eigen::VectorXd eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(k).eigenvalues();
  EXPECT_GT(eigenvalues.minCoeff(), -1e-10 * eigenvalues.maxCoeff());
  // Local: a grid's trace reaches the band points within one element of
  // its two segments, which couple to their neighbours' averages on both
  // circles: under 20 grids, where the harmonic AGE couples all 97.
  for (int j = 0; j < sparse.outerSize(); ++j)
    EXPECT_LE(sparse.col(j).nonZeros(), 20) << j;
  // The radial functions are linear in ln r, so the logarithmic field of a
  // constant on each circle is exact, as in the AGE.
  Eigen::VectorXd a(97);
  a.head(40).setConstant(.7);
  a.tail(57).setConstant(-.2);
  const double expected = pi * .81 / (mu0 * std::log(r2 / r1));
  EXPECT_NEAR(.5 * a.dot(k * a), expected, 1e-12 * expected);
}

TEST(AirGapBand, HarmonicEnergyErrorIsTheOneLayerTruncation) {
  // A trace cos(n theta) on the inner circle and 0.6 cos(n theta + 0.4) on
  // the outer, sampled on fine circles and a fine band so that only the
  // radial (one-layer) error is left. Exactly, the self energy per circle is
  // n coth(nL) ~ 1/L + n^2 L/3 - n^4 L^3/45 and the coupling
  // n / sinh(nL) ~ 1/L - n^2 L/6 + 7 n^4 L^3/360; the bilinear band keeps
  // the first two terms of each, an error of order (nL)^4 / 45 relative to
  // the mean term, against the exact annulus energy of the AGE.
  const double r1 = .02, r2 = .021, l = std::log(r2 / r1);
  for (int n : {4, 16, 48}) {
    SCOPED_TRACE(n);
    AirGap g = band_gap(r1, r2, 2048, 2048, 4096);
    g.harmonics = 1024;
    const Eigen::VectorXd a = sample(g, [&](const Vector &x) {
      const double t = std::atan2(x.y(), x.x());
      return x.head<2>().norm() < (r1 + r2) / 2 ? std::cos(n * t) : .6 * std::cos(n * t + .4);
    });
    const double band = .5 * a.dot(band_stiffness(g) * a),
                 exact = .5 * a.dot(air_gap_stiffness(g) * a);
    const double nl = n * l, error = std::abs(band - exact) / exact;
    EXPECT_GT(band, exact); // too stiff: a linear radial profile
    EXPECT_LT(error, std::max(.5 * std::pow(nl, 4), 1e-6)) << band << " " << exact;
    if (nl > .5)
      EXPECT_GT(error, std::pow(nl, 4) / 450);
  }
}

TEST(AirGapBand, LayersReduceTheTruncationAsTheirSquare) {
  // Harmonic 24 across a gap with n L = 1.17: one layer is far too stiff.
  // The interior rings are unknowns; with them condensed out (the energy
  // minimized over them for the given traces) the band's radial error falls
  // as 1/M^2, the O(h^2) energy error of linear elements (measured 4.1, 4.1
  // and 4.2 times per doubling of the layers). Each band is
  // compared with a 32-layer band on the same angular grid, which leaves
  // the grid's own (angular) error out; the one-layer band is also checked
  // against the exact annulus.
  const double r1 = .02, r2 = .021;
  const int n = 24;
  auto energy = [&](int layers, double *exact) {
    AirGap g = band_gap(r1, r2, 512, 512, 1024);
    g.harmonics = 256;
    g.band_layers = layers;
    g.band_first = 1024;
    const Eigen::VectorXd traces = sample(g, [&](const Vector &x) {
      const double t = std::atan2(x.y(), x.x());
      return x.head<2>().norm() < (r1 + r2) / 2 ? std::cos(n * t) : .6 * std::cos(n * t + .4);
    });
    if (exact)
      *exact = .5 * traces.dot(air_gap_stiffness(g) * traces);
    const Eigen::SparseMatrix<double> k = band_stiffness(g);
    const Eigen::Index circles = 1024, interior = k.rows() - circles;
    EXPECT_EQ(interior, (layers - 1) * 1024);
    double band = .5 * traces.dot(k.topLeftCorner(circles, circles) * traces);
    if (interior > 0) {
      const Eigen::SparseMatrix<double> kii = k.bottomRightCorner(interior, interior),
                                        kic = k.bottomLeftCorner(interior, circles);
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solve(kii);
      const Eigen::VectorXd rhs = kic * traces, inside = solve.solve(rhs);
      band -= .5 * rhs.dot(inside);
    }
    return band;
  };
  double exact = 0;
  const double one = energy(1, &exact), converged = energy(32, nullptr);
  // One layer: (n L)^4 / 45 relative to the mean term, here about 6%.
  EXPECT_GT((one - exact) / exact, .03);
  EXPECT_LT(std::abs(converged - exact) / exact, 1e-2); // the grid's error
  double previous = (one - converged) / converged;
  for (int layers : {2, 4, 8}) {
    SCOPED_TRACE(layers);
    const double error = (energy(layers, nullptr) - converged) / converged;
    EXPECT_GT(error, 0);
    EXPECT_GT(error, previous / 4.5);
    EXPECT_LT(error, previous / 3.5);
    previous = error;
  }
}

TEST(AirGapBand, TorqueIsMinusTheEnergysDerivativeForATurnOfTheInnerPart) {
  // The energy for the inner grids turned by delta (their potentials held),
  // by central differences: the band averages the turned trace, so this is
  // the torque band_torque returns.
  const AirGap g = band_gap(.02, .021, 37, 52, 60);
  Eigen::VectorXd a(89);
  for (int i = 0; i < 89; ++i)
    a[i] = std::sin(1.3 * i) + .2 * std::cos(.4 * i);
  auto energy = [&](double delta) {
    AirGap turned = g;
    for (double &angle : turned.inner.angles)
      angle += delta;
    return .5 * a.dot(band_stiffness(turned) * a);
  };
  const double h = 1e-5, torque = band_torque(g, a);
  const double expected = -(energy(h) - energy(-h)) / (2 * h);
  EXPECT_NEAR(torque, expected, 1e-6 * std::abs(expected));
  // Its gradient in a, by central differences, with gradient . a = 2 T.
  const Eigen::VectorXd gradient = band_torque_gradient(g, a);
  EXPECT_NEAR(gradient.dot(a), 2 * torque, 1e-10 * std::abs(torque));
  for (int i : {0, 5, 36, 37, 60, 88}) {
    Eigen::VectorXd up = a, down = a;
    up[i] += 1e-4;
    down[i] -= 1e-4;
    EXPECT_NEAR(gradient[i], (band_torque(g, up) - band_torque(g, down)) / 2e-4,
                1e-6 * gradient.norm())
        << i;
  }
}

TEST(AirGapBand, TorqueConvergesToTheAnnulusTorque) {
  // Traces cos(3 theta) inside and cos(3 theta - 0.5) outside: the annulus
  // torque is -3 M_3 sin(0.5), M_3 = -(3 pi / mu0) 2 rho^3 / (1 - rho^6).
  // The band's torque converges to it at O(h^2), smoothly: the averaged
  // traces move continuously under the band as a part turns.
  const double r1 = .02, r2 = .021, q = std::pow(r1 / r2, 3);
  const double exact = -3 * -(3 * pi / mu0) * 2 * q / (1 - q * q) * std::sin(.5);
  double previous = 1;
  for (int n : {48, 96, 192, 384}) {
    SCOPED_TRACE(n);
    const AirGap g = band_gap(r1, r2, n, n * 3 / 2, n * 3 / 2);
    const Eigen::VectorXd a = sample(g, [&](const Vector &x) {
      const double t = std::atan2(x.y(), x.x());
      return x.head<2>().norm() < (r1 + r2) / 2 ? std::cos(3 * t) : std::cos(3 * t - .5);
    });
    const double error = std::abs(band_torque(g, a) - exact) / std::abs(exact);
    EXPECT_LT(error, previous / 3.5);
    previous = error;
  }
  EXPECT_LT(previous, 1e-3);
}

TEST(AirGapParser, PairgapAndAirgapParameterCards) {
  const Geometry geo = geometry(Machine{});
  auto message = [&](const std::string &gap) {
    Machine g;
    g.gap = gap;
    return error_of(parse(g, geo));
  };
  for (const auto &[cards, expected] : std::vector<std::pair<std::string, std::string>>{
           {"PAIRGAP,9,FEM\n", "PAIRGAP METHOD must be AGE, BAND or MESH"},
           {"PAIRGAP,9,AGE,-2\n", "PAIRGAP NHARM must not be negative"},
           {"PAIRGAP,9,BAND,,2\n", "PAIRGAP NBAND must be at least 3"},
           {"PAIRGAP,9,BAND,,-4\n", "PAIRGAP NBAND must be at least 3"},
           {"PAIRGAP,9,BAND,,96,-1\n", "PAIRGAP NLAYER must be at least 1"},
           {"PAIRGAP,9,AGE,4,70,1,2\n", "unexpected field 6"},
           {"PAIRGAP,9\nPARAM,AIRGAP,FEM\n", "AIRGAP must be AGE, BAND or MESH"},
           {"PAIRGAP,9\nPMAG2D,9,1\n", "PAIRGAP 9 duplicates the ID of a PSOLID, PMAG or PMAG2D"},
           {"PAIRGAP,9\nPAIRGAP,9\n", "duplicate id 9"},
           {"PAIRGAP,9\nPAIRGAP,8\n", "PAIRGAP 8 has no elements; mesh the air gap and give "
                                      "its elements PID 8"},
           {"", "references undefined property 9"}}) {
    SCOPED_TRACE(cards);
    const auto text = message(cards);
    EXPECT_FALSE(text.empty());
    EXPECT_NE(text.find(expected), std::string::npos) << text;
  }
  // METHOD is AGE when blank.
  Machine g;
  g.gap = "PAIRGAP,9,,12\n";
  auto m = parsed(g, geo);
  ASSERT_EQ(m.air_gaps.size(), 1U);
  EXPECT_EQ(m.air_gaps.at(9).method, AirGapMethod::Age);
  EXPECT_EQ(m.air_gaps.at(9).harmonics, 12);
  EXPECT_FALSE(m.properties.contains(9));
  // METHOD=BAND with its grid count, and PARAM,AIRGAP,BAND over AGE.
  g.gap = "PAIRGAP,9,BAND,,96\n";
  m = parsed(g, geo);
  EXPECT_EQ(m.air_gaps.at(9).method, AirGapMethod::Band);
  EXPECT_EQ(m.air_gaps.at(9).band_points, 96);
  g.gap = "PAIRGAP,9,BAND,,96,3\n";
  m = parsed(g, geo);
  EXPECT_EQ(m.air_gaps.at(9).band_layers, 3);
  g.gap = "PAIRGAP,9\nPARAM,AIRGAP,BAND\n";
  m = parsed(g, geo);
  EXPECT_EQ(m.options.air_gap_method, AirGapMethod::Band);
  EXPECT_FALSE(m.properties.contains(9));
  // A PAIRGAP in a 3D model.
  const auto solid = parse_string(
      "SOL MAGSTAT\nCEND\nBEGIN BULK\nPAIRGAP,9\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\n"
      "GRID,3,,0.,1.,0.\nGRID,4,,0.,0.,1.\nCTETRA,1,9,1,2,3,4\nENDDATA\n");
  EXPECT_NE(error_of(solid).find("PAIRGAP 9 requires MAG2D"), std::string::npos)
      << error_of(solid);
}

TEST(AirGapParser, MeshMethodIsAnOrdinaryAirProperty) {
  // METHOD=MESH, or PARAM,AIRGAP,MESH over METHOD=AGE, leaves no gap: PID 9
  // is a property of a new mu_r = 1 material, and the model records why.
  const Geometry geo = geometry(Machine{});
  for (const std::string gap : {"PAIRGAP,9,MESH\n", "PAIRGAP,9\nPARAM,AIRGAP,MESH\n",
                                "PAIRGAP,9,AGE,8\nPARAM,AIRGAP,MESH\n"}) {
    SCOPED_TRACE(gap);
    Machine g;
    g.gap = gap;
    const auto m = parsed(g, geo);
    EXPECT_TRUE(m.air_gaps.empty());
    ASSERT_TRUE(m.properties.contains(9));
    EXPECT_EQ(m.properties.at(9).material, 2);
    const auto &air = m.materials.at(m.properties.at(9).material);
    EXPECT_EQ(air.mur, 1);
    EXPECT_EQ(air.bh_table, 0);
    EXPECT_EQ(air.br, 0);
    EXPECT_FALSE(m.properties.at(9).oriented);
    EXPECT_EQ(m.meshed_air_gaps.at(9), gap.starts_with("PAIRGAP,9,MESH")
                                           ? "PAIRGAP METHOD=MESH"
                                           : "PARAM,AIRGAP,MESH");
  }
  // PARAM,AIRGAP,AGE overrides METHOD=MESH.
  Machine g;
  g.gap = "PAIRGAP,9,MESH\nPARAM,AIRGAP,AGE\n";
  const auto m = parsed(g, geo);
  EXPECT_TRUE(m.air_gaps.contains(9));
  EXPECT_FALSE(m.properties.contains(9));
}

TEST(AirGapRegion, BuildsTheAnnulusFromThePairgapElements) {
  // Nonconforming, unevenly spaced circles about an offset center, with the
  // gap meshed in triangles: the gap elements are not field cells, and the
  // AGE takes the circles from the boundary of their region.
  Machine g;
  g.center = Vector(.3, -.2, 0);
  g.rotor_div = 48;
  g.stator_div = 72;
  g.stator_phase = .37;
  g.jitter = .2;
  const Geometry geo = geometry(g);
  ASSERT_EQ(geo.gap_elements.size(), 120U);
  const auto m = parsed(g, geo);
  const Mesh mesh = mesh_of(m);
  ASSERT_TRUE(mesh.air_gap.has_value());
  const auto &gap = *mesh.air_gap;
  EXPECT_EQ(mesh.cells.size(), m.elements.size() - 120);
  for (int id : geo.gap_elements)
    EXPECT_FALSE(mesh.element_index.contains(id));
  EXPECT_EQ(gap.property, 9);
  EXPECT_EQ(gap.elements, geo.gap_elements);
  EXPECT_LT((gap.center - g.center).norm(), 1e-12);
  EXPECT_NEAR(gap.inner.radius, g.rotor, 1e-14);
  EXPECT_NEAR(gap.outer.radius, g.gap_outer, 1e-14);
  EXPECT_EQ(gap.inner.nodes.size(), 48U);
  EXPECT_EQ(gap.outer.nodes.size(), 72U);
  EXPECT_EQ(gap.faces.size(), 120U);
  // Blank NHARM: half the finer circle's grid count.
  EXPECT_EQ(gap.harmonics, 36);
  EXPECT_LT(gap.deviation, 1e-12);
  // The polygon of the rotor's edges comes no closer to the center than
  // the midpoint of its widest chord.
  double nearest = g.rotor;
  for (size_t k = 0; k < gap.inner.nodes.size(); ++k) {
    const int a = gap.inner.nodes[k], b = gap.inner.nodes[(k + 1) % gap.inner.nodes.size()];
    nearest = std::min(nearest, ((mesh.nodes[a] + mesh.nodes[b]) / 2 - g.center).norm());
  }
  EXPECT_NEAR(gap.inner_chord, nearest, 1e-12);
  auto grid_ids = [&mesh](const GapCircle &c) {
    std::vector<int> ids(c.nodes.size());
    std::transform(c.nodes.begin(), c.nodes.end(), ids.begin(),
                   [&mesh](int v) { return mesh.node_ids[v]; });
    return ids;
  };
  std::vector<int> inner_ids = grid_ids(gap.inner), outer_ids = grid_ids(gap.outer);
  std::sort(inner_ids.begin(), inner_ids.end());
  std::sort(outer_ids.begin(), outer_ids.end());
  EXPECT_EQ(inner_ids, geo.rotor_edge);
  EXPECT_EQ(outer_ids, geo.stator_edge);
  for (const GapCircle *c : {&gap.inner, &gap.outer}) {
    EXPECT_TRUE(std::is_sorted(c->angles.begin(), c->angles.end()));
    EXPECT_LT(c->angles.back(), c->angles.front() + 2 * pi);
    for (size_t k = 0; k < c->nodes.size(); ++k) {
      const Vector d = mesh.nodes[c->nodes[k]] - gap.center;
      EXPECT_NEAR(std::atan2(d.dot(gap.v), d.dot(gap.u)), c->angles[k], 1e-12);
    }
  }
  // Gap grids are unknowns; the stator's outer boundary stays prescribed.
  auto gauged = make_gauge(mesh);
  ASSERT_TRUE(std::holds_alternative<Gauge>(gauged)) << error_of(gauged);
  const auto &gauge = std::get<Gauge>(gauged);
  for (const GapCircle *c : {&gap.inner, &gap.outer})
    for (int v : c->nodes)
      EXPECT_GE(gauge.equations[v], 0);
  int prescribed = 0;
  for (size_t v = 0; v < mesh.nodes.size(); ++v)
    if (gauge.equations[v] < 0) {
      ++prescribed;
      EXPECT_NEAR((mesh.nodes[v] - g.center).norm(), g.stator, 1e-12);
    }
  EXPECT_EQ(prescribed, 72);

  // NHARM sets the harmonics.
  g.gap = "PAIRGAP,9,,10\n";
  EXPECT_EQ(mesh_of(parsed(g, geo)).air_gap->harmonics, 10);
  // Solved as meshed air, the gap elements are field cells and there is no
  // air-gap element.
  g.gap = "PAIRGAP,9,MESH\n";
  const Mesh meshed = mesh_of(parsed(g, geo));
  EXPECT_FALSE(meshed.air_gap.has_value());
  EXPECT_EQ(meshed.cells.size(), m.elements.size());

  // Grids inside a layered gap belong to no field cell and are left out.
  Machine layered;
  layered.gap_layers = 3;
  const Geometry layers = geometry(layered);
  const Mesh dropped = mesh_of(parsed(layered, layers));
  EXPECT_EQ(dropped.nodes.size(), layers.grids.size() - 2 * 48);
  EXPECT_EQ(dropped.air_gap->elements.size(), 3U * 48);
}

TEST(AirGapRegion, ErrorsNameTheGridsElementsAndRadiiThatFail) {
  const Machine g;
  auto message = [](const Machine &machine, const Geometry &geo) {
    const auto out = parse(machine, geo);
    if (const auto *e = std::get_if<Error>(&out))
      return e->message;
    return error_of(make_mesh(std::get<Model>(out)));
  };
  auto expect = [](const std::string &text, const std::string &part) {
    EXPECT_NE(text.find(part), std::string::npos) << text;
  };

  // One rotor grid off the circle by 1e-4 of the radius. The least-squares
  // circle absorbs a little of the offset.
  auto off = geometry(g);
  const int moved = off.rotor_edge[5];
  off.grids.at(moved) *= 1 + 1e-4;
  auto text = message(g, off);
  expect(text, std::format("PAIRGAP 9: the inner boundary of the air gap is not a "
                           "circle: grid {} deviates from the best-fit circle (radius",
                           moved));
  expect(text, "by 9.");
  expect(text, "e-05 of its radius; the tolerance is 1e-06");

  // The stator shifted off the rotor's center.
  auto shifted = geometry(g);
  for (auto &[id, x] : shifted.grids)
    if (x.norm() > g.rotor * 1.01)
      x += Vector(1e-5, 0, 0);
  expect(message(g, shifted), "PAIRGAP 9: the inner and outer circles of the air gap "
                              "(radii 0.02 and 0.021) are not concentric: their "
                              "centers are 1.000e-05 apart");

  // PAIRGAP on a sector of the band, the rest of it ordinary air.
  auto sector = geometry(g);
  for (size_t k = 0; k < 20; ++k)
    sector.elements[static_cast<size_t>(sector.gap_elements[k] - 1)].property = 1;
  expect(message(g, sector), "PAIRGAP 9: the air-gap elements must form one annulus, "
                             "bounded by a circle on the rotor and a circle on the "
                             "stator, but their boundary has 1 closed loop (a sector");

  // A field cell inside a three-layer gap.
  Machine layered = g;
  layered.gap_layers = 3;
  auto island = geometry(layered);
  island.elements[static_cast<size_t>(island.gap_elements[48 + 7] - 1)].property = 1;
  expect(message(layered, island), "boundary has 3 closed loops (holes, or separate pieces)");

  // Gap elements on the model's exterior: the outer stator ring.
  auto exterior = geometry(g);
  for (size_t k = exterior.elements.size() - 48; k < exterior.elements.size(); ++k)
    exterior.elements[k].property = 9;
  text = message(g, exterior);
  expect(text, "is not shared with a field element. The air gap must lie between two "
               "meshed parts");

  // A missing gap element leaves its neighbors' edges unshared.
  auto removed = geometry(g);
  removed.elements.erase(removed.elements.begin() + removed.gap_elements[3] - 1);
  expect(message(g, removed), "is not shared with a field element");

  // NHARM beyond what the circles resolve.
  Machine many = g;
  many.gap = "PAIRGAP,9,AGE,25\n";
  expect(message(many, geometry(g)), "PAIRGAP 9: NHARM 25 exceeds half the grid count "
                                     "of the finer circle (48 grids)");

  // Two AGE gaps.
  auto two = geometry(g);
  for (size_t k = two.elements.size() - 48; k < two.elements.size(); ++k)
    two.elements[k].property = 8;
  Machine both = g;
  both.gap = "PAIRGAP,9\nPAIRGAP,8\n";
  expect(message(both, two), "PAIRGAP 8 and 9 both use the air-gap element; one air "
                             "gap per model is supported");
  // As meshed air they are two ordinary properties.
  both.gap += "PARAM,AIRGAP,MESH\n";
  EXPECT_EQ(message(both, two), "");

  // Sets that act on field cells may not hold gap elements.
  const int member = geometry(g).gap_elements[4];
  for (const auto &[cards, part] : std::vector<std::pair<std::string, std::string>>{
           {std::format("ESET,5,{}\nCOIL2D,11,5,10.\n", member), "COIL2D 11 ESET 5"},
           {std::format("ESET,5,{}\nMFORCE,1,BODY,5\n", member), "MFORCE 1 body ESET 5"}}) {
    Machine with = g;
    with.cards = cards;
    expect(message(with, geometry(g)),
           part + std::format(" contains air-gap element {} (PAIRGAP 9); air-gap "
                              "elements are not field cells",
                              member));
  }
  {
    auto bfield = deck(g, geometry(g));
    bfield.replace(bfield.find("CEND\n"), 5, "CEND\nBFIELD=5\n");
    bfield.replace(bfield.find("ENDDATA"), 7, std::format("ESET,5,{}\nENDDATA", member));
    expect(error_of(parse_string(bfield)),
           std::format("BFIELD ESET 5 contains air-gap element {}", member));
  }
  {
    // An FSET on a gap element's edge.
    auto m = model(g);
    const auto &e = m.elements[static_cast<size_t>(member - 1)];
    m.face_sets[7] = {{member, {e.nodes[0], e.nodes[1]}}};
    m.boundaries[1] = {7, Vector::Zero(), {}};
    m.subcases[0].boundary = 1;
    expect(error_of(solve(m)), std::format("FSET 7 references air-gap element {} "
                                           "(PAIRGAP 9)",
                                           member));
  }

  // Without the gap elements and PAIRGAP, rotor and stator are apart.
  auto apart = geometry(g);
  std::erase_if(apart.elements, [](const Element &e) { return e.property == 9; });
  Machine bare = g;
  bare.gap = "";
  text = message(bare, apart);
  expect(text, std::format("Element {} is not joined to element 1 by any chain of "
                           "elements",
                           apart.elements[6 * 48].id));
  expect(text, "mesh the air gap between them and give it a PAIRGAP property, or join "
               "the parts");

  // A third part that the gap does not reach.
  auto stray = geometry(g);
  for (int k = 0; k < 3; ++k)
    stray.grids[100000 + k] = Vector(.1 + .01 * (k == 1), .1 + .01 * (k == 2), 0);
  stray.elements.push_back({90000, 1, {100000, 100001, 100002}});
  expect(message(g, stray), "Element 90000 is not joined to element 1 by elements or by "
                            "the PAIRGAP 9 air gap");
}

TEST(AirGapSolve, MeshMethodIsIdenticalToPlainAir) {
  // The analyst's cross-check: the deck with PAIRGAP,9,MESH, the deck with
  // no PAIRGAP and PID 9 an ordinary air property, and the deck with
  // PARAM,AIRGAP,MESH over the AGE are one model, through a saturating
  // Newton solve with a magnet, a winding, probes and a force.
  Machine g;
  g.rotor_div = 40;
  g.stator_div = 60;
  g.stator_phase = .25;
  g.gap_layers = 2;
  g.rotor_property = 2;
  g.stator_property = 3;
  g.cards = magnetized_rotor(.8, .3) + "MATMAG,3,,1\nPMAG2D,3,3\n" + iron_table;
  const Geometry geo = geometry(g);
  auto run = [&](const std::string &gap, vibestran::SolverBackend &backend,
                 bool force = true) {
    Machine h = g;
    h.gap = gap;
    auto m = parsed(h, geo);
    wind(m, 50, .4);
    m.options.nonlinear_tolerance = 1e-10;
    if (force) {
      m.element_sets[5] = elements_within(m, 0, g.rotor);
      m.forces = {{1, 5, true}};
    }
    const double r = (g.rotor + g.gap_outer) / 2;
    m.probes = {{1, Vector(r * std::cos(.7), r * std::sin(.7), 0)},
                {2, Vector(.005, .003, 0)},
                {3, Vector(.03, -.01, 0)}};
    auto out = solve(m, backend);
    EXPECT_TRUE(std::holds_alternative<Results>(out)) << error_of(out);
    return std::holds_alternative<Results>(out) ? std::get<Results>(std::move(out))
                                                : Results{};
  };
  const std::vector<std::string> decks{"PAIRGAP,9,MESH\n", "MATMAG,8,1.\nPMAG2D,9,8\n",
                                       "PAIRGAP,9,AGE,7\nPARAM,AIRGAP,MESH\n"};

  // With a deterministic linear solver every result is bitwise equal: the
  // solver receives the same equations.
  vibestran::CpuPCGSolverBackend pcg(1e-12);
  std::vector<Results> exact(decks.size());
  std::transform(decks.begin(), decks.end(), exact.begin(),
                 [&](const std::string &gap) { return run(gap, pcg); });
  ASSERT_EQ(exact[0].cases.size(), 1U);
  const auto &a = exact[0].cases[0];
  EXPECT_GT(a.iterations, 2);
  ASSERT_FALSE(a.b.empty());
  for (size_t k = 1; k < exact.size(); ++k) {
    SCOPED_TRACE(decks[k]);
    ASSERT_EQ(exact[k].cases.size(), 1U);
    const auto &b = exact[k].cases[0];
    EXPECT_FALSE(exact[k].mesh.air_gap.has_value());
    EXPECT_EQ(exact[k].mesh.cells.size(), exact[0].mesh.cells.size());
    EXPECT_TRUE(a.potential == b.potential);
    EXPECT_EQ(a.energy, b.energy);
    EXPECT_EQ(a.coenergy, b.coenergy);
    EXPECT_EQ(a.iterations, b.iterations);
    EXPECT_EQ(a.residual, b.residual);
    ASSERT_EQ(a.b.size(), b.b.size());
    for (size_t i = 0; i < a.b.size(); ++i) {
      EXPECT_EQ(a.b[i], b.b[i]) << i;
      EXPECT_EQ(a.h[i], b.h[i]) << i;
    }
    ASSERT_EQ(b.probe_b.size(), 3U);
    for (size_t i = 0; i < a.probe_b.size(); ++i)
      EXPECT_EQ(a.probe_b[i], b.probe_b[i]) << i;
    ASSERT_EQ(b.circuits.size(), 1U);
    EXPECT_EQ(a.circuits[0].linkage, b.circuits[0].linkage);
    EXPECT_TRUE(a.circuit_inductance == b.circuit_inductance);
    ASSERT_EQ(b.forces.size(), 1U);
    EXPECT_EQ(a.forces[0].force, b.forces[0].force);
    EXPECT_EQ(a.forces[0].torque, b.forces[0].torque);
  }
  // The F06 says which card made the gap meshed air.
  EXPECT_EQ(gap_rows(exact[0]), std::vector<std::string>{"PAIRGAP 9 solved as meshed air "
                                                         "(relative permeability 1) by "
                                                         "PAIRGAP METHOD=MESH"});
  EXPECT_TRUE(gap_rows(exact[1]).empty());
  EXPECT_EQ(gap_rows(exact[2]), std::vector<std::string>{"PAIRGAP 9 solved as meshed air "
                                                         "(relative permeability 1) by "
                                                         "PARAM,AIRGAP,MESH"});

  // A multithreaded sparse factorization (the default backend's) sums in
  // an order that varies from run to run, so even one deck solved twice
  // differs in the last bits; across the three decks the results agree to
  // that round-off and no more.
  vibestran::LdltSolverBackend direct;
  std::vector<Results> runs(decks.size());
  std::transform(decks.begin(), decks.end(), runs.begin(),
                 [&](const std::string &gap) { return run(gap, direct); });
  for (size_t k = 1; k < runs.size(); ++k) {
    SCOPED_TRACE(decks[k]);
    ASSERT_EQ(runs[k].cases.size(), 1U);
    const auto &x = runs[0].cases[0], &y = runs[k].cases[0];
    EXPECT_LT((x.potential - y.potential).norm(), 1e-12 * x.potential.norm());
    EXPECT_NEAR(x.energy, y.energy, 1e-13 * x.energy);
    EXPECT_NEAR(x.circuits[0].linkage, y.circuits[0].linkage,
                1e-12 * std::abs(x.circuits[0].linkage));
    EXPECT_LT((x.forces[0].torque - y.forces[0].torque).norm(),
              1e-11 * x.forces[0].torque.norm());
    EXPECT_EQ(x.iterations, y.iterations);
  }
  EXPECT_NEAR(runs[0].cases[0].energy, a.energy, 1e-11 * a.energy);

  // The AGE is a different discretization of the gap: close, not equal.
  const Results age = run("PAIRGAP,9\n", pcg, false);
  ASSERT_TRUE(age.mesh.air_gap.has_value());
  ASSERT_EQ(age.cases.size(), 1U);
  EXPECT_NE(age.cases[0].energy, a.energy);
  EXPECT_NEAR(age.cases[0].energy, a.energy, 1e-2 * a.energy);
}

TEST(AirGapSolve, MagnetizedRotorMatchesTheAnalyticFieldAtSecondOrder) {
  // Nonconforming circles (the stator has 1.5 times the rotor's divisions,
  // offset by a third of a division): the AGE couples them without error of
  // its own, so the polygonal magnet boundary sets the O(h^2) error.
  const double br = 1.0;
  double previous_gap = 1, previous_inside = 1;
  for (int div : {24, 48, 96}) {
    SCOPED_TRACE(div);
    Machine g;
    g.rotor_div = div;
    g.stator_div = div * 3 / 2;
    g.stator_phase = 1. / 3;
    g.rotor_rings = div / 8;
    g.stator_rings = div / 6;
    g.rotor_property = 2;
    g.cards = magnetized_rotor(br);
    auto m = model(g);
    const double gap = g.gap_outer - g.rotor;
    std::vector<Vector> in_gap;
    for (double t : {.3, 2.2, 4.1})
      for (double depth : {.1, .5, .9}) {
        const double r = g.rotor + depth * gap;
        in_gap.emplace_back(r * std::cos(t), r * std::sin(t), 0);
      }
    m.probes.clear();
    for (size_t i = 0; i < in_gap.size(); ++i)
      m.probes.push_back({static_cast<int>(i) + 1, in_gap[i]});
    m.probes.push_back({100, Vector(.005, .003, 0)});
    // Between the rotor's outermost chord and the circle: outside the
    // polygonal magnet, in no field cell, and in the gap.
    const double mid = pi / div, sliver = g.rotor * (1 + std::cos(mid)) / 2;
    m.probes.push_back({101, Vector(sliver * std::cos(mid), sliver * std::sin(mid), 0)});
    const auto results = solved(m);
    ASSERT_EQ(results.cases.size(), 1U);
    const auto &c = results.cases[0];
    double gap_error = 0;
    for (size_t i = 0; i < in_gap.size(); ++i) {
      const Vector expected = magnet_field_outside(g, br, in_gap[i]);
      gap_error = std::max(gap_error, (c.probe_b[i] - expected).norm() / expected.norm());
      EXPECT_LT((c.probe_h[i] - c.probe_b[i] / mu0).norm(), 1e-9 * c.probe_h[i].norm());
    }
    const Vector inside(br / 2 * (1 - g.rotor * g.rotor / (g.stator * g.stator)), 0, 0);
    const double inside_error = (c.probe_b[in_gap.size()] - inside).norm() / inside.norm();
    EXPECT_LT(gap_error, previous_gap / 3.5);
    EXPECT_LT(inside_error, previous_inside / 3.5);
    previous_gap = gap_error;
    previous_inside = inside_error;
    const Vector edge = magnet_field_outside(g, br, m.probes.back().position);
    EXPECT_LT((c.probe_b.back() - edge).norm(), 2e-2 * edge.norm());
    const auto rows = gap_rows(results);
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_NE(rows[0].find(std::format("PAIRGAP 9, air-gap element (AGE) in place of its "
                                       "{} elements",
                                       div * 5 / 2)),
              std::string::npos)
        << rows[0];
    EXPECT_NE(rows[0].find(std::format("{} + {} grids, {} coupled harmonics, {} on the "
                                       "finer circle",
                                       div, div * 3 / 2, div / 2, div * 3 / 4)),
              std::string::npos)
        << rows[0];
  }
  EXPECT_LT(previous_gap, 1e-3);
  EXPECT_LT(previous_inside, 5e-4);
}

TEST(AirGapSolve, BandMatchesTheAnalyticMagnetizedRotorAtSecondOrder) {
  // The magnetized rotor of the AGE test above with METHOD=BAND. The field
  // is harmonic 1, for which the one-layer band's error (n ln(r2/r1))^4 / 45
  // is 1e-7, so the sampling of the traces and the meshed parts set an
  // O(h^2) error.
  const double br = 1.0;
  double previous_gap = 1, previous_inside = 1;
  for (int div : {24, 48, 96}) {
    SCOPED_TRACE(div);
    Machine g;
    g.rotor_div = div;
    g.stator_div = div * 3 / 2;
    g.stator_phase = 1. / 3;
    g.rotor_rings = div / 8;
    g.stator_rings = div / 6;
    g.rotor_property = 2;
    g.gap = "PAIRGAP,9,BAND\n";
    g.cards = magnetized_rotor(br);
    auto m = model(g);
    const double gap = g.gap_outer - g.rotor;
    std::vector<Vector> in_gap;
    for (double t : {.3, 2.2, 4.1})
      for (double depth : {.1, .5, .9}) {
        const double r = g.rotor + depth * gap;
        in_gap.emplace_back(r * std::cos(t), r * std::sin(t), 0);
      }
    m.probes.clear();
    for (size_t i = 0; i < in_gap.size(); ++i)
      m.probes.push_back({static_cast<int>(i) + 1, in_gap[i]});
    m.probes.push_back({100, Vector(.005, .003, 0)});
    const auto results = solved(m);
    ASSERT_EQ(results.cases.size(), 1U);
    ASSERT_TRUE(results.mesh.air_gap);
    // NBAND blank: the smallest prime count of at least the finer circle's
    // grid count.
    const int band = std::map<int, int>{{24, 37}, {48, 73}, {96, 149}}.at(div);
    EXPECT_EQ(results.mesh.air_gap->band_points, band);
    const auto &c = results.cases[0];
    double gap_error = 0;
    for (size_t i = 0; i < in_gap.size(); ++i) {
      const Vector expected = magnet_field_outside(g, br, in_gap[i]);
      gap_error = std::max(gap_error, (c.probe_b[i] - expected).norm() / expected.norm());
    }
    const Vector inside(br / 2 * (1 - g.rotor * g.rotor / (g.stator * g.stator)), 0, 0);
    const double inside_error = (c.probe_b[in_gap.size()] - inside).norm() / inside.norm();
    EXPECT_LT(gap_error, previous_gap / 3.5);
    EXPECT_LT(inside_error, previous_inside / 3.5);
    previous_gap = gap_error;
    previous_inside = inside_error;
    const auto rows = gap_rows(results);
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_NE(rows[0].find(std::format("PAIRGAP 9, sliding band (BAND) in place of its "
                                       "{} elements",
                                       div * 5 / 2)),
              std::string::npos)
        << rows[0];
    EXPECT_NE(rows[0].find(std::format("{} + {} grids, {} band points", div, div * 3 / 2,
                                       band)),
              std::string::npos)
        << rows[0];
  }
  EXPECT_LT(previous_gap, 2e-3);
  EXPECT_LT(previous_inside, 1e-3);
}

TEST(AirGapSolve, BandConvergesToTheAgeAroundAnIronStator) {
  // One deck solved with the AGE and with the band: two discretizations of
  // the same gap, whose energies and gap fields approach each other at about
  // second order as the circles and the band are refined together (measured
  // 1.6e-3, 2.5e-4, 7.3e-5 in energy and 5.9e-4, 8.9e-5, 4.2e-5 in the
  // field: 21 and 14 times over two doublings, not evenly per step).
  std::vector<double> energies, fields;
  for (int div : {48, 96, 192}) {
    SCOPED_TRACE(div);
    Machine g;
    g.rotor_div = div;
    g.stator_div = div * 3 / 2;
    g.stator_phase = .3;
    g.rotor_rings = div / 8;
    g.stator_rings = div / 6;
    g.rotor_property = 2;
    g.stator_property = 3;
    g.cards = magnetized_rotor(1.0) + "MATMAG,3,1000.\nPMAG2D,3,3\n";
    const Geometry geo = geometry(g);
    const double r = (g.rotor + g.gap_outer) / 2;
    const Vector probe(r * std::cos(.7), r * std::sin(.7), 0);
    double energy[2]{};
    Vector field[2]{Vector::Zero(), Vector::Zero()};
    for (int band : {0, 1}) {
      Machine h = g;
      h.gap = band ? "PAIRGAP,9,BAND\n" : "PAIRGAP,9\n";
      auto m = parsed(h, geo);
      m.probes = {{1, probe}};
      const auto results = solved(m);
      ASSERT_EQ(results.cases.size(), 1U);
      energy[band] = results.cases[0].energy;
      field[band] = results.cases[0].probe_b[0];
    }
    energies.push_back(std::abs(energy[0] - energy[1]) / energy[0]);
    fields.push_back((field[0] - field[1]).norm() / field[0].norm());
  }
  EXPECT_LT(energies[2], energies[0] / 12);
  EXPECT_LT(fields[2], fields[0] / 10);
  EXPECT_LT(energies[2], 1e-4);
  EXPECT_LT(fields[2], 6e-5);
}

TEST(AirGapSolve, AgeConvergesToTheMeshedGapAroundAnIronStator) {
  // The same deck with the gap meshed in four layers, solved by the AGE
  // and as meshed air: both discretizations approach one solution, so the
  // energy difference shrinks at O(h^2) and the gap field agrees.
  double previous = 1, previous_field = 1;
  for (int div : {48, 96, 192}) {
    SCOPED_TRACE(div);
    Machine g;
    g.rotor_div = g.stator_div = div;
    g.rotor_rings = div / 8;
    g.stator_rings = div / 6;
    g.gap_layers = 4;
    g.rotor_property = 2;
    g.stator_property = 3;
    g.cards = magnetized_rotor(1.0) + "MATMAG,3,1000.\nPMAG2D,3,3\n";
    const Geometry geo = geometry(g);
    const double r = (g.rotor + g.gap_outer) / 2;
    const Vector probe(r * std::cos(.7), r * std::sin(.7), 0);
    double energy[2]{};
    Vector field[2]{Vector::Zero(), Vector::Zero()};
    for (int meshed : {0, 1}) {
      Machine h = g;
      h.gap = meshed ? "PAIRGAP,9,MESH\n" : "PAIRGAP,9\n";
      auto m = parsed(h, geo);
      m.probes = {{1, probe}};
      const auto results = solved(m);
      ASSERT_EQ(results.cases.size(), 1U);
      EXPECT_EQ(results.mesh.air_gap.has_value(), !meshed);
      energy[meshed] = results.cases[0].energy;
      field[meshed] = results.cases[0].probe_b[0];
    }
    const double difference = std::abs(energy[0] - energy[1]) / energy[1];
    EXPECT_LT(difference, previous / 3.4);
    previous = difference;
    // The meshed gap's B is the gradient of a piecewise-linear potential,
    // first-order accurate, so the fields agree at first order.
    const double field_difference = (field[0] - field[1]).norm() / field[1].norm();
    EXPECT_LT(field_difference, previous_field / 1.4);
    previous_field = field_difference;
  }
  EXPECT_LT(previous, 2e-4);
  EXPECT_LT(previous_field, 1e-2);
}

TEST(AirGapSolve, RigidRotationLeavesEnergyAndRotatedFieldsUnchanged) {
  // Rotating the whole model, magnetization included, rotates every
  // harmonic of the traces exactly, so the solution is the same to round-off.
  Machine g;
  g.rotor_div = 40;
  g.stator_div = 54;
  g.stator_phase = .2;
  g.jitter = .15;
  g.rotor_property = 2;
  g.stator_property = 3;
  const double angle = .4;
  const double r = (g.rotor + g.gap_outer) / 2;
  const Vector probe(r * std::cos(1.1), r * std::sin(1.1), 0);
  const Eigen::Matrix3d rotate =
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Results results[2];
  for (int turned : {0, 1}) {
    Machine h = g;
    h.rotation = turned ? angle : 0;
    h.cards = magnetized_rotor(1.0, h.rotation) + "MATMAG,3,500.\nPMAG2D,3,3\n";
    auto m = model(h);
    m.probes = {{1, turned ? Vector(rotate * probe) : probe}};
    results[turned] = solved(m);
  }
  ASSERT_EQ(results[0].cases.size(), 1U);
  ASSERT_EQ(results[1].cases.size(), 1U);
  const auto &a = results[0].cases[0], &b = results[1].cases[0];
  EXPECT_NEAR(a.energy, b.energy, 1e-10 * a.energy);
  EXPECT_LT((rotate * a.probe_b[0] - b.probe_b[0]).norm(), 1e-9 * a.probe_b[0].norm());
  ASSERT_EQ(a.b.size(), b.b.size());
  for (size_t i = 0; i < a.b.size(); ++i)
    EXPECT_LT((rotate * a.b[i] - b.b[i]).norm(), 1e-9 * (a.b[i].norm() + 1e-3));
}

TEST(AirGapSolve, ExteriorBoundaryConditionSkipsTheGap) {
  // All air in a uniform applied field: the field inside the rotor is the
  // applied field, up to the O(h^2) interpolation of the gap traces.
  Machine g;
  g.rotor_div = 64;
  g.stator_div = 80;
  auto m = model(g);
  const Vector applied(.3, -.1, 0);
  m.boundaries[1] = {0, applied, {}};
  m.subcases[0].boundary = 1;
  m.probes = {{1, Vector(.004, -.006, 0)},
              {2, Vector((g.rotor + g.gap_outer) / 2, 0, 0)}};
  const auto whole = solved(m);
  ASSERT_EQ(whole.cases.size(), 1U);
  const auto &c = whole.cases[0];
  EXPECT_LT((c.probe_b[0] - applied).norm(), 2e-3 * applied.norm());
  EXPECT_LT((c.probe_b[1] - applied).norm(), 2e-3 * applied.norm());

  // An explicit FSET of the outer boundary is the same condition.
  const Mesh mesh = mesh_of(m);
  ASSERT_TRUE(mesh.air_gap.has_value());
  std::vector<bool> on_gap(mesh.faces.size(), false);
  for (int f : mesh.air_gap->faces)
    on_gap[f] = true;
  std::vector<FaceSelection> outer, with_gap_face;
  for (size_t f = 0; f < mesh.faces.size(); ++f) {
    const auto &face = mesh.faces[f];
    if (face.cells[1] >= 0)
      continue;
    const FaceSelection s{mesh.cells[face.cells[0]].id,
                          {mesh.node_ids[face.nodes[0]], mesh.node_ids[face.nodes[1]]}};
    if (!on_gap[f])
      outer.push_back(s);
    else if (with_gap_face.empty())
      with_gap_face.push_back(s);
  }
  ASSERT_EQ(outer.size(), 80U);
  m.face_sets[7] = outer;
  m.boundaries[1].faces = 7;
  const auto explicit_set = solved(m);
  ASSERT_EQ(explicit_set.cases.size(), 1U);
  EXPECT_LT((explicit_set.cases[0].potential - c.potential).norm(),
            1e-12 * c.potential.norm());

  // The gap takes no boundary condition.
  m.face_sets[7].push_back(with_gap_face.front());
  const auto text = error_of(solve(m));
  EXPECT_NE(text.find("on the air gap; the air gap is coupled by the air-gap element"),
            std::string::npos)
      << text;
}

TEST(AirGapSolve, SaturatedIronAndWindingMatchTheMeshedGap) {
  // Iron rotor and stator (saturating B-H curve) with a two-slot winding:
  // Newton runs through the AGE's constant block, and the circuit's linkage
  // and differential inductance agree with the gap solved as meshed air.
  Machine g;
  g.rotor_div = g.stator_div = 96;
  g.rotor_rings = 12;
  g.stator_rings = 16;
  g.gap_layers = 4;
  g.rotor_property = g.stator_property = 2;
  g.cards = "MATMAG,2,,1\nPMAG2D,2,2\n" + iron_table;
  const Geometry geo = geometry(g);
  Results results[2];
  for (int meshed : {0, 1}) {
    Machine h = g;
    h.gap = meshed ? "PAIRGAP,9,MESH\n" : "PAIRGAP,9\n";
    auto m = parsed(h, geo);
    m.options.nonlinear_tolerance = 1e-11;
    wind(m, 100, .35);
    results[meshed] = solved(m);
  }
  ASSERT_EQ(results[0].cases.size(), 1U);
  ASSERT_EQ(results[1].cases.size(), 1U);
  const auto &age = results[0].cases[0], &mesh = results[1].cases[0];
  // Past the end of the B-H table: well into saturation.
  EXPECT_TRUE(age.extrapolated);
  EXPECT_GT(age.iterations, 3);
  EXPECT_LE(age.residual, 1e-11);
  EXPECT_LE(age.iterations, mesh.iterations + 2);
  EXPECT_NEAR(age.energy, mesh.energy, 2e-4 * mesh.energy);
  ASSERT_EQ(age.circuits.size(), 1U);
  EXPECT_NEAR(age.circuits[0].linkage, mesh.circuits[0].linkage,
              2e-4 * std::abs(mesh.circuits[0].linkage));
  EXPECT_NEAR(age.circuit_inductance(0, 0), mesh.circuit_inductance(0, 0),
              5e-4 * mesh.circuit_inductance(0, 0));
}

TEST(AirGapSolve, ForceSurfacesMayNotTouchTheGap) {
  Machine g;
  auto m = model(g);
  const Mesh mesh = mesh_of(m);
  ASSERT_TRUE(mesh.air_gap.has_value());
  // The body around the rotor's elements reaches the gap at once.
  m.element_sets[5] = elements_within(m, 0, g.rotor);
  m.forces = {{1, 5, true}};
  auto text = error_of(solve(m));
  EXPECT_NE(text.find("MFORCE 1: the surface around body ESET 5 reaches the air gap"),
            std::string::npos)
      << text;
  // An explicit surface on the gap's faces.
  std::vector<FaceSelection> faces;
  for (int f : mesh.air_gap->faces) {
    const auto &face = mesh.faces[f];
    faces.push_back({mesh.cells[face.cells[0]].id,
                     {mesh.node_ids[face.nodes[0]], mesh.node_ids[face.nodes[1]]}});
  }
  m.face_sets[6] = faces;
  m.forces = {{2, 6, false}};
  text = error_of(solve(m));
  EXPECT_NE(text.find("lies on the air gap; force surfaces may not touch the air gap"),
            std::string::npos)
      << text;
}

TEST(AirGapSolve, ResultsRecordTheGap) {
  Machine g;
  g.rotor_div = 40;
  g.stator_div = 60;
  g.rotor_property = 2;
  g.cards = magnetized_rotor(1.0);
  const auto m = model(g);
  const auto results = solved(m);
  ASSERT_TRUE(results.mesh.air_gap.has_value());
#ifdef VIBESTRAN_HAS_HDF5
  const auto &gap = *results.mesh.air_gap;
  const auto directory = std::filesystem::temp_directory_path() / "vibestran_air_gap";
  std::filesystem::create_directories(directory);
  ASSERT_FALSE(write_results(m, results, directory / "gap.F06").has_value());
  const auto file = directory / "gap.mag.vtkhdf";
  EXPECT_EQ(vibestran::test::h5_attribute(file, "/MAGSTAT/air_gap", "property"), 9);
  EXPECT_NEAR(vibestran::test::h5_attribute(file, "/MAGSTAT/air_gap", "inner_radius_m"),
              g.rotor, 1e-15);
  EXPECT_NEAR(vibestran::test::h5_attribute(file, "/MAGSTAT/air_gap", "outer_radius_m"),
              g.gap_outer, 1e-15);
  EXPECT_EQ(vibestran::test::h5_attribute(file, "/MAGSTAT/air_gap", "harmonics"), 30);
  // Grid IDs in angle order: the deck's own IDs, so a user can find them.
  const auto inner = vibestran::test::h5_table(file, "/MAGSTAT/air_gap/inner_grids");
  const auto outer = vibestran::test::h5_table(file, "/MAGSTAT/air_gap/outer_grids");
  ASSERT_EQ(inner.values.size(), 40U);
  ASSERT_EQ(outer.values.size(), 60U);
  for (size_t k = 0; k < inner.values.size(); ++k)
    EXPECT_EQ(inner.values[k], results.mesh.node_ids[gap.inner.nodes[k]]);
  for (double id : outer.values)
    EXPECT_NEAR(m.nodes.at(static_cast<int>(id)).position.norm(), g.gap_outer, 1e-15);
  std::ifstream in(directory / "gap.F06");
  std::stringstream f06;
  f06 << in.rdbuf();
  EXPECT_NE(f06.str().find("   Air gap .................. PAIRGAP 9, air-gap element (AGE) "
                           "in place of its 100 elements: radius 0.02 to 0.021 m about "
                           "(0, 0, 0),\n"
                           "                              40 + 60 grids, 20 coupled harmonics, "
                           "30 on the finer circle"),
            std::string::npos)
      << f06.str();
  std::filesystem::remove_all(directory);
#else
  GTEST_SKIP() << "MAGSTAT results need HDF5";
#endif
}
