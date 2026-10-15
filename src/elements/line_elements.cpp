// src/elements/line_elements.cpp

#include "elements/line_elements.hpp"
#include "core/coord_sys.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numbers>

namespace vibestran {

namespace {

constexpr double kFrameTolerance = 1e-12;

struct LineSection {
  double E{0.0};
  double G{0.0};
  double rho{0.0};
  double alpha{0.0};
  double ref_temp{0.0};
  double A{0.0};
  double I1{0.0};
  double I2{0.0};
  double I12{0.0};
  double J{0.0};
  double nsm{0.0};
  /// Transverse shear area factors; K*A*G is the shear stiffness of the
  /// plane.  Zero means rigid in shear (Euler-Bernoulli in that plane).
  double K1{0.0};
  double K2{0.0};
};

struct LineFrame {
  double length{0.0};
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
  Eigen::Matrix<double, 12, 12> lambda{Eigen::Matrix<double, 12, 12>::Identity()};
};

struct HermiteData {
  std::array<double, 4> N{};
  std::array<double, 4> dN_dx{};
  std::array<double, 4> d2N_dx2{};
};

[[nodiscard]] std::string uppercase_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  return s;
}

[[nodiscard]] Vec3 normalized_or_throw(const Vec3 &v, const std::string &ctx) {
  const double n = v.norm();
  if (n < kFrameTolerance)
    throw SolverError(std::format("{}: direction vector has near-zero norm",
                                  ctx));
  return v * (1.0 / n);
}

[[nodiscard]] Vec3 choose_reference_axis(const Vec3 &x_axis) {
  Vec3 ref{0.0, 0.0, 1.0};
  if (std::abs(x_axis.dot(ref)) > 0.95)
    ref = Vec3{0.0, 1.0, 0.0};
  if (std::abs(x_axis.dot(ref)) > 0.95)
    ref = Vec3{1.0, 0.0, 0.0};
  return ref;
}

[[nodiscard]] Vec3 orientation_vector_in_basic(const Model &model, const Vec3 &v,
                                               const CoordId cid,
                                               const Vec3 &position,
                                               const std::string &ctx) {
  if (cid.value == 0)
    return v;
  const auto cs_it = model.coord_systems.find(cid);
  if (cs_it == model.coord_systems.end()) {
    throw SolverError(std::format(
        "{}: coordinate system {} not found for orientation vector", ctx,
        cid.value));
  }
  return apply_rotation(rotation_matrix(cs_it->second, position), v);
}

[[nodiscard]] LineFrame build_line_frame(
    const std::array<NodeId, 2> &nodes, const Model &model,
    const std::optional<Vec3> &orientation, const std::optional<NodeId> &g0,
    const std::optional<CoordId> &cid, const std::string &ctx,
    const bool allow_coincident, const bool coincident_orientation_is_x,
    const std::array<Vec3, 2> &end_offsets_basic = {}) {
  // The element axis runs between the (possibly offset) element ends; the
  // orientation vector is anchored at grid GA, as in Nastran.
  const Vec3 ga = model.node(nodes[0]).position;
  const Vec3 x1 = ga + end_offsets_basic[0];
  const Vec3 x2 = model.node(nodes[1]).position + end_offsets_basic[1];
  Vec3 axis = x2 - x1;
  double length = axis.norm();

  Vec3 ref_vec;
  if (g0.has_value()) {
    ref_vec = model.node(*g0).position - ga;
  } else if (orientation.has_value()) {
    ref_vec = orientation_vector_in_basic(model, *orientation, cid.value_or(CoordId{0}),
                                          ga, ctx);
  }

  if (length < kFrameTolerance) {
    if (!allow_coincident) {
      throw SolverError(std::format(
          "{}: coincident grid points are not supported", ctx));
    }
    if (ref_vec.norm() < kFrameTolerance) {
      throw SolverError(std::format(
          "{}: coincident grids require an explicit orientation", ctx));
    }
    axis = coincident_orientation_is_x ? ref_vec : choose_reference_axis(Vec3{1.0, 0.0, 0.0});
    length = 1.0;
    if (coincident_orientation_is_x)
      ref_vec = choose_reference_axis(normalized_or_throw(axis, ctx));
  }

  const Vec3 ex = normalized_or_throw(axis, ctx);

  if (length >= kFrameTolerance) {
    if (ref_vec.norm() < kFrameTolerance)
      ref_vec = choose_reference_axis(ex);
  } else {
    ref_vec = choose_reference_axis(ex);
  }

  Vec3 ey_candidate = ref_vec - ex * ex.dot(ref_vec);
  if (ey_candidate.norm() < kFrameTolerance) {
    ref_vec = choose_reference_axis(ex);
    ey_candidate = ref_vec - ex * ex.dot(ref_vec);
  }
  const Vec3 ey = normalized_or_throw(ey_candidate, ctx);
  const Vec3 ez = normalized_or_throw(ex.cross(ey), ctx);
  const Vec3 ey_fixed = normalized_or_throw(ez.cross(ex), ctx);

  LineFrame frame;
  frame.length = length;
  frame.rotation << ex.x, ey_fixed.x, ez.x, ex.y, ey_fixed.y, ez.y, ex.z,
      ey_fixed.z, ez.z;
  frame.lambda.setZero();
  for (int node = 0; node < 2; ++node) {
    const int offset = 6 * node;
    frame.lambda.block<3, 3>(offset, offset) = frame.rotation.transpose();
    frame.lambda.block<3, 3>(offset + 3, offset + 3) =
        frame.rotation.transpose();
  }
  return frame;
}

[[nodiscard]] HermiteData hermite_shape(const double x, const double L) {
  const double s = x / L;
  HermiteData h;
  h.N[0] = 1.0 - 3.0 * s * s + 2.0 * s * s * s;
  h.N[1] = L * (s - 2.0 * s * s + s * s * s);
  h.N[2] = 3.0 * s * s - 2.0 * s * s * s;
  h.N[3] = L * (-s * s + s * s * s);

  h.dN_dx[0] = (-6.0 * s + 6.0 * s * s) / L;
  h.dN_dx[1] = 1.0 - 4.0 * s + 3.0 * s * s;
  h.dN_dx[2] = (6.0 * s - 6.0 * s * s) / L;
  h.dN_dx[3] = -2.0 * s + 3.0 * s * s;

  h.d2N_dx2[0] = (-6.0 + 12.0 * s) / (L * L);
  h.d2N_dx2[1] = (-4.0 + 6.0 * s) / L;
  h.d2N_dx2[2] = (6.0 - 12.0 * s) / (L * L);
  h.d2N_dx2[3] = (-2.0 + 6.0 * s) / L;
  return h;
}

[[nodiscard]] std::array<double, 2> linear_shape(const double x,
                                                 const double L) {
  return {1.0 - x / L, x / L};
}

[[nodiscard]] LineSection section_from_bar_property(const PBar &prop,
                                                    const Mat1 &mat) {
  return {
      .E = mat.E,
      .G = (mat.G != 0.0) ? mat.G : mat.E / (2.0 * (1.0 + mat.nu)),
      .rho = mat.rho,
      .alpha = mat.A,
      .ref_temp = mat.ref_temp,
      .A = prop.A,
      .I1 = prop.I1,
      .I2 = prop.I2,
      .I12 = prop.I12,
      .J = prop.J,
      .nsm = prop.nsm,
      .K1 = prop.K1,
      .K2 = prop.K2,
  };
}

[[nodiscard]] LineSection section_from_barl_property(const PBarL &prop,
                                                     const Mat1 &mat) {
  return {
      .E = mat.E,
      .G = (mat.G != 0.0) ? mat.G : mat.E / (2.0 * (1.0 + mat.nu)),
      .rho = mat.rho,
      .alpha = mat.A,
      .ref_temp = mat.ref_temp,
      .A = prop.A,
      .I1 = prop.I1,
      .I2 = prop.I2,
      .I12 = 0.0,
      .J = prop.J,
      .nsm = prop.nsm,
  };
}

[[nodiscard]] LineSection section_from_beam_property(const PBeam &prop,
                                                     const Mat1 &mat) {
  return {
      .E = mat.E,
      .G = (mat.G != 0.0) ? mat.G : mat.E / (2.0 * (1.0 + mat.nu)),
      .rho = mat.rho,
      .alpha = mat.A,
      .ref_temp = mat.ref_temp,
      .A = prop.A,
      .I1 = prop.I1,
      .I2 = prop.I2,
      .I12 = prop.I12,
      .J = prop.J,
      .nsm = prop.nsm,
  };
}

[[nodiscard]] LineSection resolve_line_section(const Model &model,
                                               const PropertyId pid,
                                               const ElementType type,
                                               const ElementId eid) {
  const Property &prop = model.property(pid);
  if (type == ElementType::CBAR) {
    if (const auto *p = std::get_if<PBar>(&prop))
      return section_from_bar_property(*p, model.material(p->mid));
    if (const auto *p = std::get_if<PBarL>(&prop))
      return section_from_barl_property(*p, model.material(p->mid));
    throw SolverError(std::format(
        "CBAR {}: property {} must be PBAR or PBARL", eid.value, pid.value));
  }

  if (const auto *p = std::get_if<PBeam>(&prop))
    return section_from_beam_property(*p, model.material(p->mid));
  throw SolverError(std::format("CBEAM {}: property {} must be PBEAM",
                                eid.value, pid.value));
}

using RecoveryPoints = std::array<std::array<double, 2>, 4>;

/// Stress recovery points C, D, E, F as (y, z) in the element frame.
/// PBAR supplies them explicitly; PBARL derives them from the section
/// dimensions; otherwise they are all at the neutral axis (zero bending
/// stress), which is the Nastran behaviour for blank C1..F2.
[[nodiscard]] RecoveryPoints recovery_points_from_property(const Property &prop) {
  if (const auto *pbar = std::get_if<PBar>(&prop))
    return pbar->recovery_points;
  if (const auto *pbarl = std::get_if<PBarL>(&prop)) {
    const std::string type = uppercase_copy(pbarl->section_type);
    if (type == "BAR" && pbarl->dimensions.size() == 2) {
      // DIM1 spans element z and DIM2 spans element y, consistent with
      // I1 = DIM1*DIM2^3/12 (plane 1 bends about z).
      const double half_y = 0.5 * pbarl->dimensions[1];
      const double half_z = 0.5 * pbarl->dimensions[0];
      return {{{half_y, half_z},
               {-half_y, half_z},
               {-half_y, -half_z},
               {half_y, -half_z}}};
    }
    if (type == "ROD" && pbarl->dimensions.size() == 1) {
      const double r = 0.5 * pbarl->dimensions[0];
      return {{{r, 0.0}, {0.0, r}, {-r, 0.0}, {0.0, -r}}};
    }
    if (type == "TUBE" && pbarl->dimensions.size() == 2) {
      const double r = 0.5 * pbarl->dimensions[0];
      return {{{r, 0.0}, {0.0, r}, {-r, 0.0}, {0.0, -r}}};
    }
  }
  return {};
}

/// Bending stress at section point (y, z) for plane moments M1, M2 (Nastran
/// sign convention, see add_coupled_bending_stiffness):
///   sigma = -[y (I2 M1 - I12 M2) + z (I1 M2 - I12 M1)] / (I1 I2 - I12^2)
/// which reduces to -M1 y / I1 - M2 z / I2 for I12 = 0.  A plane with zero
/// inertia carries no bending stress.
[[nodiscard]] double bending_stress(const LineSection &section, const double y,
                                    const double z, const double m1,
                                    const double m2) {
  if (section.I12 != 0.0) {
    const double den = section.I1 * section.I2 - section.I12 * section.I12;
    return -(y * (section.I2 * m1 - section.I12 * m2) +
             z * (section.I1 * m2 - section.I12 * m1)) /
           den;
  }
  double sigma = 0.0;
  if (section.I1 > 0.0)
    sigma -= m1 * y / section.I1;
  if (section.I2 > 0.0)
    sigma -= m2 * z / section.I2;
  return sigma;
}

[[nodiscard]] CBarBeamElement::StressRecoveryEnd
recover_line_end_stress(const LineSection &section,
                        const RecoveryPoints &points, const double axial_force,
                        const double m1, const double m2) {
  CBarBeamElement::StressRecoveryEnd end;
  if (section.A > 0.0)
    end.axial = axial_force / section.A;

  for (size_t i = 0; i < 4; ++i)
    end.s[i] = bending_stress(section, points[i][0], points[i][1], m1, m2);

  const auto [smin_it, smax_it] =
      std::minmax_element(end.s.begin(), end.s.end());
  end.smax = end.axial + *smax_it;
  end.smin = end.axial + *smin_it;
  return end;
}

[[nodiscard]] Eigen::Matrix3d skew(const Vec3 &w) {
  Eigen::Matrix3d S;
  S << 0.0, -w.z, w.y,
       w.z, 0.0, -w.x,
       -w.y, w.x, 0.0;
  return S;
}

/// Element frame plus the map from grid DOFs (basic) to element-end DOFs
/// (local frame): u_end_local = T * u_grid_basic.  T combines the rigid end
/// offsets (u_end = u_grid + theta x W) with the global-to-local rotation.
struct BarGeometry {
  LineFrame frame;
  Eigen::Matrix<double, 12, 12> T{Eigen::Matrix<double, 12, 12>::Identity()};
};

[[nodiscard]] std::string line_element_context(const ElementData &data) {
  return std::format("{} {}",
                     (data.type == ElementType::CBAR) ? "CBAR" : "CBEAM",
                     data.id.value);
}

[[nodiscard]] BarGeometry build_bar_geometry(const ElementData &data,
                                             const Model &model) {
  const std::string ctx = line_element_context(data);
  const std::array<NodeId, 2> nodes{data.nodes[0], data.nodes[1]};

  // Offsets are given in the displacement coordinate system of each grid.
  std::array<Vec3, 2> offsets_basic{};
  for (size_t end = 0; end < 2; ++end) {
    const GridPoint &gp = model.node(nodes[end]);
    offsets_basic[end] = orientation_vector_in_basic(
        model, data.offsets[end], gp.cd, gp.position, ctx);
  }
  // The orientation vector is in the displacement system of GA unless OFFT
  // selects the basic system.
  const CoordId orientation_cid =
      data.orientation_in_basic ? CoordId{0} : model.node(nodes[0]).cd;

  BarGeometry geom;
  geom.frame = build_line_frame(nodes, model, data.orientation, data.g0,
                                orientation_cid, ctx, false, false,
                                offsets_basic);
  Eigen::Matrix<double, 12, 12> offset_map =
      Eigen::Matrix<double, 12, 12>::Identity();
  for (int end = 0; end < 2; ++end)
    offset_map.block<3, 3>(6 * end, 6 * end + 3) =
        -skew(offsets_basic[static_cast<size_t>(end)]);
  geom.T = geom.frame.lambda * offset_map;
  return geom;
}

/// Static condensation of pin-flag releases (PA/PB) on the local element-end
/// DOFs.  For released set r and retained set k,
///   u_r = K_rr^+ (f_r - K_rk u_k)
/// so the full end displacement is u_full = C u + D f, the condensed
/// stiffness is C^T K C, and an element load f enters the assembly as
/// C^T f.  K_rr^+ is a minimum-norm pseudo-inverse: releasing a component at
/// both ends (e.g. torsion) leaves K_rr singular, and because K is
/// positive semi-definite the range of K_rk lies in the range of K_rr, so the
/// released DOFs still carry zero force.
struct PinCondensation {
  Eigen::Matrix<double, 12, 12> C{Eigen::Matrix<double, 12, 12>::Identity()};
  Eigen::Matrix<double, 12, 12> D{Eigen::Matrix<double, 12, 12>::Zero()};
};

[[nodiscard]] PinCondensation
condense_pin_flags(const LocalKe &K, const std::array<std::uint8_t, 2> &pins) {
  PinCondensation pc;
  std::vector<int> released;
  std::vector<int> retained;
  for (int dof = 0; dof < 12; ++dof) {
    const std::uint8_t mask = pins[static_cast<size_t>(dof / 6)];
    if ((mask >> (dof % 6)) & 1u)
      released.push_back(dof);
    else
      retained.push_back(dof);
  }
  if (released.empty())
    return pc;

  const auto nr = static_cast<Eigen::Index>(released.size());
  const auto nk = static_cast<Eigen::Index>(retained.size());
  Eigen::MatrixXd K_rr(nr, nr);
  Eigen::MatrixXd K_rk(nr, nk);
  for (Eigen::Index i = 0; i < nr; ++i) {
    for (Eigen::Index j = 0; j < nr; ++j)
      K_rr(i, j) = K(released[static_cast<size_t>(i)],
                     released[static_cast<size_t>(j)]);
    for (Eigen::Index j = 0; j < nk; ++j)
      K_rk(i, j) = K(released[static_cast<size_t>(i)],
                     retained[static_cast<size_t>(j)]);
  }
  const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(K_rr);
  const Eigen::MatrixXd K_rr_pinv = cod.pseudoInverse();
  const Eigen::MatrixXd slave = -K_rr_pinv * K_rk;

  for (const int r : released)
    pc.C(r, r) = 0.0;
  for (Eigen::Index i = 0; i < nr; ++i) {
    const int r = released[static_cast<size_t>(i)];
    for (Eigen::Index j = 0; j < nk; ++j)
      pc.C(r, retained[static_cast<size_t>(j)]) = slave(i, j);
    for (Eigen::Index j = 0; j < nr; ++j)
      pc.D(r, released[static_cast<size_t>(j)]) = K_rr_pinv(i, j);
  }
  return pc;
}

// cppcheck-suppress constParameterReference -- Eigen matrices are mutated via operator()
void add_linear_2x2(LocalKe &K, const int i0, const int i1,
                    const Eigen::Matrix2d &sub) {
  K(i0, i0) += sub(0, 0);
  K(i0, i1) += sub(0, 1);
  K(i1, i0) += sub(1, 0);
  K(i1, i1) += sub(1, 1);
}

// cppcheck-suppress constParameterReference -- Eigen matrices are mutated via operator()
void add_plane_4x4(LocalKe &K, const std::array<int, 4> &idx,
                   const Eigen::Matrix4d &sub) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      K(idx[i], idx[j]) += sub(i, j);
}

/// Bending stiffness of one plane in (v1, theta1, v2, theta2) order, with
/// theta = dv/dx.  Timoshenko form: shear_stiffness = K*A*G; zero gives the
/// Euler-Bernoulli matrix (phi = 0).
[[nodiscard]] Eigen::Matrix4d timoshenko_plane_stiffness(const double EI,
                                                         const double shear_stiffness,
                                                         const double L) {
  const double phi =
      (shear_stiffness > 0.0) ? 12.0 * EI / (shear_stiffness * L * L) : 0.0;
  const double c = EI / (L * L * L * (1.0 + phi));
  const double L2 = L * L;
  Eigen::Matrix4d k;
  k << 12.0, 6.0 * L, -12.0, 6.0 * L,
       6.0 * L, (4.0 + phi) * L2, -6.0 * L, (2.0 - phi) * L2,
       -12.0, -6.0 * L, 12.0, -6.0 * L,
       6.0 * L, (2.0 - phi) * L2, -6.0 * L, (4.0 + phi) * L2;
  return c * k;
}

/// Coupled bending stiffness for a section with a nonzero product of
/// inertia, from the exact flexibility of a cantilever fixed at end A.
///
/// Sign convention (Nastran): bending stress is
///   sigma = -E (y kappa1 + z kappa2),  kappa1 = v'',  kappa2 = w''
/// so the plane moments are [M1; M2] = E * Ibar * [kappa1; kappa2] with
/// Ibar = [[I1, I12], [I12, I2]].  For tip loads p = [Vy, Vz, My, Mz] at
/// end B the moments along the span are [M1; M2](x) = H(x) p with
///   H = [[L-x, 0, 0, 1], [0, L-x, -1, 0]],
/// and the conjugate tip displacements are [v, w, theta_y, theta_z].  The
/// tip flexibility is F = int_0^L H^T (E Ibar)^-1 H dx, plus L/(K1*A*G) and
/// L/(K2*A*G) on the v and w diagonals for transverse shear.  F^-1 is
/// expanded to both ends through the rigid-body relation between end A and
/// the tip, d = R u.
void add_coupled_bending_stiffness(LocalKe &K, const LineSection &section,
                                   const double L, const std::string &ctx) {
  const double den = section.I1 * section.I2 - section.I12 * section.I12;
  if (!(den > 0.0) || section.E <= 0.0) {
    throw SolverError(std::format(
        "{}: coupled bending requires E > 0 and I1*I2 > I12^2 (E={}, I1={}, "
        "I2={}, I12={})",
        ctx, section.E, section.I1, section.I2, section.I12));
  }
  Eigen::Matrix2d compliance;
  compliance << section.I2, -section.I12, -section.I12, section.I1;
  compliance /= section.E * den;

  // H^T C H is quadratic in x; two-point Gauss-Legendre is exact.
  Eigen::Matrix4d F = Eigen::Matrix4d::Zero();
  const double g = 1.0 / std::sqrt(3.0);
  for (const double xi : {-g, g}) {
    const double x = 0.5 * L * (xi + 1.0);
    Eigen::Matrix<double, 2, 4> H;
    H << L - x, 0.0, 0.0, 1.0,
         0.0, L - x, -1.0, 0.0;
    F += H.transpose() * compliance * H * (0.5 * L);
  }
  const double shear1 = section.K1 * section.A * section.G;
  const double shear2 = section.K2 * section.A * section.G;
  if (shear1 > 0.0)
    F(0, 0) += L / shear1;
  if (shear2 > 0.0)
    F(1, 1) += L / shear2;

  // Tip deformation relative to the rigid-body motion of end A.
  Eigen::Matrix<double, 4, 12> R = Eigen::Matrix<double, 4, 12>::Zero();
  R(0, 7) = 1.0;  R(0, 1) = -1.0; R(0, 5) = -L;   // v_B - v_A - L*theta_zA
  R(1, 8) = 1.0;  R(1, 2) = -1.0; R(1, 4) = L;    // w_B - w_A + L*theta_yA
  R(2, 10) = 1.0; R(2, 4) = -1.0;                 // theta_yB - theta_yA
  R(3, 11) = 1.0; R(3, 5) = -1.0;                 // theta_zB - theta_zA

  const Eigen::Matrix4d K_tip = F.llt().solve(Eigen::Matrix4d::Identity());
  K += R.transpose() * K_tip * R;
}

/// Local 12x12 stiffness at the element ends (before pin-flag condensation).
/// Plane 1 (element x-y, DOFs v/theta_z) bends with I1 and shear factor K1;
/// plane 2 (element x-z, DOFs w/theta_y) bends with I2 and K2.
[[nodiscard]] LocalKe build_beam_local_stiffness(const LineSection &section,
                                                 const double L,
                                                 const std::string &ctx) {
  LocalKe K = LocalKe::Zero(12, 12);

  const Eigen::Matrix2d axial =
      section.E * section.A / L *
      (Eigen::Matrix2d() << 1.0, -1.0, -1.0, 1.0).finished();
  add_linear_2x2(K, 0, 6, axial);

  const Eigen::Matrix2d torsion =
      section.G * section.J / L *
      (Eigen::Matrix2d() << 1.0, -1.0, -1.0, 1.0).finished();
  add_linear_2x2(K, 3, 9, torsion);

  if (section.I12 != 0.0) {
    add_coupled_bending_stiffness(K, section, L, ctx);
    return K;
  }

  add_plane_4x4(K, {1, 5, 7, 11},
                timoshenko_plane_stiffness(section.E * section.I1,
                                           section.K1 * section.A * section.G,
                                           L));
  // Plane 2 uses theta_y = -dw/dx, so flip the rotation rows/columns.
  const Eigen::Vector4d flip{1.0, -1.0, 1.0, -1.0};
  add_plane_4x4(K, {2, 4, 8, 10},
                flip.asDiagonal() *
                    timoshenko_plane_stiffness(section.E * section.I2,
                                               section.K2 * section.A * section.G,
                                               L) *
                    flip.asDiagonal());
  return K;
}

[[nodiscard]] LocalKe build_beam_local_mass(const LineSection &section,
                                            const double L) {
  LocalKe M = LocalKe::Zero(12, 12);
  const double mu = section.rho * section.A + section.nsm;
  if (mu == 0.0 && section.rho == 0.0)
    return M;

  static constexpr std::array<double, 4> gauss_pts{
      -0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
      0.8611363115940526};
  static constexpr std::array<double, 4> gauss_wts{
      0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
      0.3478548451374538};

  Eigen::Matrix2d maxial = Eigen::Matrix2d::Zero();
  Eigen::Matrix2d mtorsion = Eigen::Matrix2d::Zero();
  Eigen::Matrix4d mv = Eigen::Matrix4d::Zero();
  Eigen::Matrix4d mw = Eigen::Matrix4d::Zero();
  const double torsional_mu = section.rho * (section.I1 + section.I2);

  for (int gp = 0; gp < 4; ++gp) {
    const double x = 0.5 * L * (gauss_pts[gp] + 1.0);
    const double jac = 0.5 * L * gauss_wts[gp];
    const auto nlin = linear_shape(x, L);
    const HermiteData h = hermite_shape(x, L);

    Eigen::Matrix<double, 2, 1> n_ax;
    n_ax << nlin[0], nlin[1];
    maxial += mu * (n_ax * n_ax.transpose()) * jac;
    mtorsion += torsional_mu * (n_ax * n_ax.transpose()) * jac;

    Eigen::Matrix<double, 4, 1> nv;
    nv << h.N[0], h.N[1], h.N[2], h.N[3];
    mv += mu * (nv * nv.transpose()) * jac;

    Eigen::Matrix<double, 4, 1> nw;
    nw << h.N[0], -h.N[1], h.N[2], -h.N[3];
    mw += mu * (nw * nw.transpose()) * jac;
  }

  add_linear_2x2(M, 0, 6, maxial);
  add_linear_2x2(M, 3, 9, mtorsion);
  add_plane_4x4(M, {1, 5, 7, 11}, mv);
  add_plane_4x4(M, {2, 4, 8, 10}, mw);
  return M;
}

[[nodiscard]] double scalar_stiffness_from_property(const Model &model,
                                                    const PropertyId pid,
                                                    const ElementType type,
                                                    const ElementId eid) {
  const Property &prop = model.property(pid);
  if (const auto *pelas = std::get_if<PElas>(&prop))
    return pelas->k;
  throw SolverError(std::format(
      "{} {}: property {} must be PELAS",
      (type == ElementType::CELAS1) ? "CELAS1" : "CELAS2", eid.value,
      pid.value));
}

[[nodiscard]] double scalar_mass_from_property(const Model &model,
                                               const PropertyId pid,
                                               const ElementType type,
                                               const ElementId eid) {
  const Property &prop = model.property(pid);
  if (const auto *pmass = std::get_if<PMass>(&prop))
    return pmass->mass;
  throw SolverError(std::format(
      "{} {}: property {} must be PMASS",
      (type == ElementType::CMASS1) ? "CMASS1" : "CMASS2", eid.value,
      pid.value));
}

[[nodiscard]] const PBush &lookup_pbush(const Model &model, const PropertyId pid,
                                        const ElementId eid) {
  const Property &prop = model.property(pid);
  if (!std::holds_alternative<PBush>(prop))
    throw SolverError(std::format("CBUSH {}: property {} is not PBUSH",
                                  eid.value, pid.value));
  return std::get<PBush>(prop);
}

[[nodiscard]] LocalFe zero_force(const int ndof) {
  return LocalFe::Zero(ndof);
}

[[nodiscard]] std::vector<EqIndex> line_global_dofs(const std::array<NodeId, 2> &nodes,
                                                    const DofMap &dof_map) {
  std::vector<EqIndex> result;
  result.reserve(12);
  for (const NodeId nid : nodes) {
    const auto &blk = dof_map.block(nid);
    for (int d = 0; d < 6; ++d)
      result.push_back(blk.eq[d]);
  }
  return result;
}

/// Decoded PLOAD1 TYPE/SCALE (Nastran QRG): FX..MZ act in basic directions,
/// FXE..MZE in element directions; LE/FR give X1/X2 as length or fraction of
/// the element, and the PR variants give the intensity per unit length
/// projected normal to the load direction (basic-direction types only).
struct Pload1Kind {
  bool moment{false};
  bool element_axes{false};
  int axis{0}; // 0 = x, 1 = y, 2 = z
  bool fractional{false};
  bool projected{false};
};

[[nodiscard]] Pload1Kind decode_pload1(const Pload1Load &load) {
  const std::string type = uppercase_copy(load.load_type);
  const std::string scale = uppercase_copy(load.scale_type);
  Pload1Kind kind;
  const bool valid_type =
      (type.size() == 2 || (type.size() == 3 && type[2] == 'E')) &&
      (type[0] == 'F' || type[0] == 'M') &&
      (type[1] == 'X' || type[1] == 'Y' || type[1] == 'Z');
  if (!valid_type) {
    throw SolverError(std::format(
        "PLOAD1 on element {}: load type '{}' is not supported (expected "
        "FX, FY, FZ, MX, MY, MZ or the element-axis forms FXE..MZE)",
        load.element.value, load.load_type));
  }
  kind.moment = (type[0] == 'M');
  kind.axis = type[1] - 'X';
  kind.element_axes = (type.size() == 3);
  if (scale == "LE" || scale == "FR" || scale == "LEPR" || scale == "FRPR") {
    kind.fractional = scale.starts_with("FR");
    kind.projected = scale.ends_with("PR");
  } else {
    throw SolverError(std::format(
        "PLOAD1 on element {}: scale type '{}' is not supported (expected "
        "LE, FR, LEPR or FRPR)",
        load.element.value, load.scale_type));
  }
  if (kind.projected && kind.element_axes) {
    throw SolverError(std::format(
        "PLOAD1 on element {}: projected scale '{}' requires a basic-direction "
        "load type, got '{}'",
        load.element.value, load.scale_type, load.load_type));
  }
  return kind;
}

/// Add the consistent nodal loads of a force vector and a moment vector
/// (element frame) acting at station x, scaled by `weight`.  Axial force and
/// torque use linear shapes; transverse loads use the Hermite shapes of the
/// bending stiffness, with theta_z = dv/dx and theta_y = -dw/dx.  For a
/// point load these are the exact Euler-Bernoulli fixed-end forces.
void add_station_load(LocalFe &fe, const double x, const double L,
                      const Eigen::Vector3d &force,
                      const Eigen::Vector3d &moment, const double weight) {
  const auto nlin = linear_shape(x, L);
  const HermiteData h = hermite_shape(x, L);
  const std::array<int, 4> v_dofs{1, 5, 7, 11};
  const std::array<int, 4> w_dofs{2, 4, 8, 10};
  const std::array<double, 4> v_force{h.N[0], h.N[1], h.N[2], h.N[3]};
  const std::array<double, 4> w_force{h.N[0], -h.N[1], h.N[2], -h.N[3]};
  const std::array<double, 4> v_moment{h.dN_dx[0], h.dN_dx[1], h.dN_dx[2],
                                       h.dN_dx[3]};
  const std::array<double, 4> w_moment{-h.dN_dx[0], h.dN_dx[1], -h.dN_dx[2],
                                       h.dN_dx[3]};

  fe(0) += nlin[0] * force(0) * weight;
  fe(6) += nlin[1] * force(0) * weight;
  fe(3) += nlin[0] * moment(0) * weight;
  fe(9) += nlin[1] * moment(0) * weight;
  for (size_t i = 0; i < 4; ++i) {
    fe(v_dofs[i]) += (v_force[i] * force(1) + v_moment[i] * moment(2)) * weight;
    fe(w_dofs[i]) += (w_force[i] * force(2) + w_moment[i] * moment(1)) * weight;
  }
}

} // namespace

CBarBeamElement::CBarBeamElement(const ElementData &data, const Model &model)
    : data_(data), model_(model) {
  if (data_.nodes.size() != NUM_NODES) {
    throw SolverError(std::format("{} needs 2 nodes, got {}",
                                  line_element_context(data_),
                                  data_.nodes.size()));
  }
  nodes_ = {data_.nodes[0], data_.nodes[1]};
}

namespace {

/// Everything needed to map between grid DOFs and local end forces for one
/// CBAR/CBEAM: geometry, unreleased local stiffness, and pin condensation.
struct BarSystem {
  LineSection section;
  BarGeometry geom;
  LocalKe K_local;
  PinCondensation pins;
};

[[nodiscard]] BarSystem build_bar_system(const ElementData &data,
                                         const Model &model) {
  BarSystem sys;
  sys.section = resolve_line_section(model, data.pid, data.type, data.id);
  sys.geom = build_bar_geometry(data, model);
  sys.K_local = build_beam_local_stiffness(sys.section, sys.geom.frame.length,
                                           line_element_context(data));
  sys.pins = condense_pin_flags(sys.K_local, data.pin_flags);
  return sys;
}

/// Local fixed-end load vector for a uniform temperature change: the axial
/// thermal force E*A*alpha*dT pushes the ends apart.
[[nodiscard]] LocalFe thermal_fixed_end_local(const LineSection &section,
                                              const double dT) {
  LocalFe f = LocalFe::Zero(12);
  const double axial_force = section.E * section.A * section.alpha * dT;
  f(0) = -axial_force;
  f(6) = axial_force;
  return f;
}

} // namespace

LocalKe CBarBeamElement::stiffness_matrix() const {
  const BarSystem sys = build_bar_system(data_, model_);
  const Eigen::Matrix<double, 12, 12> map = sys.pins.C * sys.geom.T;
  return map.transpose() * sys.K_local * map;
}

LocalKe CBarBeamElement::mass_matrix() const {
  const BarSystem sys = build_bar_system(data_, model_);
  const LocalKe m_local =
      build_beam_local_mass(sys.section, sys.geom.frame.length);
  // Released DOFs follow the same static slaving as the stiffness (Guyan).
  const Eigen::Matrix<double, 12, 12> map = sys.pins.C * sys.geom.T;
  return map.transpose() * m_local * map;
}

LocalKe CBarBeamElement::lumped_mass_matrix() const {
  const BarSystem sys = build_bar_system(data_, model_);
  const double half_mass = 0.5 * sys.geom.frame.length *
                           (sys.section.rho * sys.section.A + sys.section.nsm);
  LocalKe m_local = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
  for (const int dof : {0, 1, 2, 6, 7, 8})
    m_local(dof, dof) = half_mass;
  const Eigen::Matrix<double, 12, 12> map = sys.pins.C * sys.geom.T;
  return map.transpose() * m_local * map;
}

LocalFe CBarBeamElement::thermal_load(std::span<const double> temperatures,
                                      double t_ref) const {
  const BarSystem sys = build_bar_system(data_, model_);
  if (sys.section.alpha == 0.0)
    return zero_force(NUM_DOFS);
  const double dT = 0.5 * (temperatures[0] + temperatures[1]) - t_ref;
  const LocalFe f_local = thermal_fixed_end_local(sys.section, dT);
  return sys.geom.T.transpose() * (sys.pins.C.transpose() * f_local);
}

// cppcheck-suppress unusedFunction
CBarBeamElement::StressRecovery CBarBeamElement::recover_stress(
    std::span<const double> global_displacements,
    const double average_temperature, const double t_ref,
    std::span<const double> distributed_fixed_end_local) const {
  const std::string ctx = line_element_context(data_);
  if (global_displacements.size() != static_cast<size_t>(NUM_DOFS)) {
    throw SolverError(std::format(
        "{}: expected {} displacement entries for stress recovery, got {}",
        ctx, NUM_DOFS, global_displacements.size()));
  }
  if (!distributed_fixed_end_local.empty() &&
      distributed_fixed_end_local.size() != static_cast<size_t>(NUM_DOFS)) {
    throw SolverError(std::format(
        "{}: expected {} fixed-end load entries for stress recovery, got {}",
        ctx, NUM_DOFS, distributed_fixed_end_local.size()));
  }

  const BarSystem sys = build_bar_system(data_, model_);
  const RecoveryPoints points =
      recovery_points_from_property(model_.property(data_.pid));

  Eigen::Matrix<double, 12, 1> u_global;
  for (int i = 0; i < NUM_DOFS; ++i)
    u_global(i) = global_displacements[static_cast<size_t>(i)];

  // Fixed-end loads of everything applied inside the span (thermal +
  // distributed), in the local end frame before pin condensation.
  LocalFe f_fixed = thermal_fixed_end_local(
      sys.section, average_temperature - t_ref);
  for (size_t i = 0; i < distributed_fixed_end_local.size(); ++i)
    f_fixed(static_cast<Eigen::Index>(i)) += distributed_fixed_end_local[i];

  // End forces acting on the element: p = K u_full - f, where u_full
  // restores the released DOFs.  Released components come out as zero.
  const Eigen::Matrix<double, 12, 1> u_end = sys.geom.T * u_global;
  const Eigen::Matrix<double, 12, 1> u_full =
      sys.pins.C * u_end + sys.pins.D * f_fixed;
  const Eigen::Matrix<double, 12, 1> p = sys.K_local * u_full - f_fixed;

  // Nastran force resultants: tension-positive axial force, plane moments
  // positive when they produce positive curvature (compression at +y/+z).
  StressRecovery recovery;
  EndForces &fr = recovery.forces;
  fr.axial = -p(0);
  fr.torque = -p(3);
  fr.v1 = -p(1);
  fr.v2 = -p(2);
  fr.m1a = -p(5);
  fr.m2a = p(4);
  fr.m1b = p(11);
  fr.m2b = -p(10);

  recovery.end_a =
      recover_line_end_stress(sys.section, points, fr.axial, fr.m1a, fr.m2a);
  // Axial stress is reported at end A in Nastran; with no axial span load
  // both ends agree.
  recovery.end_b =
      recover_line_end_stress(sys.section, points, fr.axial, fr.m1b, fr.m2b);
  return recovery;
}

std::vector<EqIndex>
CBarBeamElement::global_dof_indices(const DofMap &dof_map) const {
  return line_global_dofs(nodes_, dof_map);
}

CBushElement::CBushElement(ElementId eid, PropertyId pid,
                           std::array<NodeId, 2> node_ids, const Model &model,
                           std::optional<Vec3> orientation,
                           std::optional<NodeId> g0,
                           std::optional<CoordId> cid)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model),
      orientation_(std::move(orientation)), g0_(g0), cid_(cid) {}

LocalKe CBushElement::stiffness_matrix() const {
  const PBush &prop = lookup_pbush(model_, pid_, eid_);
  const Vec3 x1 = model_.node(nodes_[0]).position;
  const Vec3 x2 = model_.node(nodes_[1]).position;
  const bool coincident = (x2 - x1).norm() < kFrameTolerance;
  const LineFrame frame = build_line_frame(
      nodes_, model_, orientation_, g0_, cid_,
      std::format("CBUSH {}", eid_.value), true, true);

  LocalKe k_local = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
  for (int comp = 0; comp < 6; ++comp) {
    const double k = prop.k[comp];
    k_local(comp, comp) += k;
    k_local(comp, comp + 6) -= k;
    k_local(comp + 6, comp) -= k;
    k_local(comp + 6, comp + 6) += k;
  }

  if (coincident && !orientation_.has_value() && !g0_.has_value()) {
    throw SolverError(std::format(
        "CBUSH {}: coincident grids require an explicit orientation",
        eid_.value));
  }

  return frame.lambda.transpose() * k_local * frame.lambda;
}

LocalKe CBushElement::mass_matrix() const {
  return LocalKe::Zero(NUM_DOFS, NUM_DOFS);
}

LocalFe CBushElement::thermal_load(std::span<const double> /*temperatures*/,
                                   double /*t_ref*/) const {
  return zero_force(NUM_DOFS);
}

std::vector<EqIndex>
CBushElement::global_dof_indices(const DofMap &dof_map) const {
  return line_global_dofs(nodes_, dof_map);
}

ScalarSpringElement::ScalarSpringElement(ElementType type, ElementId eid,
                                         PropertyId pid,
                                         std::vector<NodeId> nodes,
                                         std::array<int, 2> components,
                                         double value, const Model &model)
    : type_(type), eid_(eid), pid_(pid), nodes_(std::move(nodes)),
      components_(components), value_(value), model_(model) {}

double ScalarSpringElement::stiffness_value() const {
  if (type_ == ElementType::CELAS1)
    return scalar_stiffness_from_property(model_, pid_, type_, eid_);
  return value_;
}

LocalKe ScalarSpringElement::stiffness_matrix() const {
  const double k = stiffness_value();
  if (nodes_.size() == 1) {
    LocalKe K(1, 1);
    K(0, 0) = k;
    return K;
  }
  LocalKe K(2, 2);
  K << k, -k, -k, k;
  return K;
}

LocalKe ScalarSpringElement::mass_matrix() const {
  return LocalKe::Zero(num_dofs(), num_dofs());
}

LocalFe ScalarSpringElement::thermal_load(std::span<const double> /*temperatures*/,
                                          double /*t_ref*/) const {
  return zero_force(num_dofs());
}

std::vector<EqIndex>
ScalarSpringElement::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> result;
  result.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i)
    result.push_back(dof_map.eq_index(nodes_[i], components_[i] - 1));
  return result;
}

ScalarMassElement::ScalarMassElement(ElementType type, ElementId eid,
                                     PropertyId pid, std::vector<NodeId> nodes,
                                     std::array<int, 2> components,
                                     double value, const Model &model)
    : type_(type), eid_(eid), pid_(pid), nodes_(std::move(nodes)),
      components_(components), value_(value), model_(model) {}

double ScalarMassElement::mass_value() const {
  if (type_ == ElementType::CMASS1)
    return scalar_mass_from_property(model_, pid_, type_, eid_);
  return value_;
}

LocalKe ScalarMassElement::stiffness_matrix() const {
  return LocalKe::Zero(num_dofs(), num_dofs());
}

LocalKe ScalarMassElement::mass_matrix() const {
  const double m = mass_value();
  if (nodes_.size() == 1) {
    LocalKe M(1, 1);
    M(0, 0) = m;
    return M;
  }
  LocalKe M(2, 2);
  M << m, -m, -m, m;
  return M;
}

LocalFe ScalarMassElement::thermal_load(std::span<const double> /*temperatures*/,
                                        double /*t_ref*/) const {
  return zero_force(num_dofs());
}

std::vector<EqIndex>
ScalarMassElement::global_dof_indices(const DofMap &dof_map) const {
  std::vector<EqIndex> result;
  result.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i)
    result.push_back(dof_map.eq_index(nodes_[i], components_[i] - 1));
  return result;
}

LocalFe compute_pload1_fixed_end_local(const ElementData &elem,
                                       const Model &model,
                                       const Pload1Load &load) {
  if (elem.type != ElementType::CBAR && elem.type != ElementType::CBEAM) {
    throw SolverError(std::format(
        "PLOAD1 on element {} is only supported for CBAR and CBEAM",
        elem.id.value));
  }
  const Pload1Kind kind = decode_pload1(load);

  // The load acts along the element span between the (offset) ends; X is
  // measured from end A.
  const LineFrame frame = build_bar_geometry(elem, model).frame;
  const double L = frame.length;

  // Unit load direction in the element frame.  Basic directions are
  // rotated in; frame.rotation has the element axes as columns.
  Eigen::Vector3d dir = Eigen::Vector3d::Zero();
  dir(kind.axis) = 1.0;
  if (!kind.element_axes)
    dir = frame.rotation.transpose() * dir;
  // Projected intensity: per unit length of the element's projection onto
  // the plane normal to the load, i.e. scaled by |sin| of the angle between
  // the element axis and the load direction.
  const double intensity_scale =
      kind.projected ? std::sqrt(std::max(0.0, 1.0 - dir(0) * dir(0))) : 1.0;
  const Eigen::Vector3d force_dir =
      kind.moment ? Eigen::Vector3d::Zero() : Eigen::Vector3d(dir);
  const Eigen::Vector3d moment_dir =
      kind.moment ? Eigen::Vector3d(dir) : Eigen::Vector3d::Zero();

  const double station_scale = kind.fractional ? L : 1.0;
  const double x1 = load.x1 * station_scale;
  const bool concentrated =
      !load.x2.has_value() || *load.x2 == load.x1;
  const double x2 = concentrated ? x1 : *load.x2 * station_scale;
  const double tol = 1e-9 * L;
  if (x2 < x1 || x1 < -tol || x2 > L + tol) {
    throw SolverError(std::format(
        "PLOAD1 on element {}: stations X1 = {:.6g}, X2 = {:.6g} must satisfy "
        "0 <= X1 <= X2 <= {} (element length {:.6g}, scale {})",
        load.element.value, x1, x2, kind.fractional ? "1" : "L", L,
        load.scale_type));
  }

  LocalFe fe_local = LocalFe::Zero(12);
  if (concentrated) {
    add_station_load(fe_local, std::clamp(x1, 0.0, L), L, force_dir,
                     moment_dir, load.p1 * intensity_scale);
    return fe_local;
  }

  // Linear intensity times cubic shapes is quintic: 4-point Gauss is exact.
  static constexpr std::array<double, 4> gauss_pts{
      -0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
      0.8611363115940526};
  static constexpr std::array<double, 4> gauss_wts{
      0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
      0.3478548451374538};
  const double p2 = load.p2.value_or(load.p1);
  for (size_t gp = 0; gp < 4; ++gp) {
    const double xi = 0.5 * (gauss_pts[gp] + 1.0);
    const double x = std::clamp(x1 + (x2 - x1) * xi, 0.0, L);
    const double q = load.p1 + (p2 - load.p1) * xi;
    add_station_load(fe_local, x, L, force_dir, moment_dir,
                     q * intensity_scale * 0.5 * (x2 - x1) * gauss_wts[gp]);
  }
  return fe_local;
}

LocalFe compute_pload1_equivalent_load(const ElementData &elem,
                                       const Model &model,
                                       const Pload1Load &load) {
  const LocalFe f_local = compute_pload1_fixed_end_local(elem, model, load);
  const BarSystem sys = build_bar_system(elem, model);
  return sys.geom.T.transpose() * (sys.pins.C.transpose() * f_local);
}

} // namespace vibestran
