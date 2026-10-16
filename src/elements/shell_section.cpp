// src/elements/shell_section.cpp
// Shell section resultants for PSHELL and PCOMP; see shell_section.hpp.

#include "elements/shell_section.hpp"
#include <cmath>
#include <format>
#include <numbers>

namespace vibestran {

Eigen::Matrix3d strain_rotation(double phi) {
  const double c = std::cos(phi), s = std::sin(phi);
  Eigen::Matrix3d T;
  T << c * c, s * s, c * s,          //
      s * s, c * c, -c * s,          //
      -2.0 * c * s, 2.0 * c * s, c * c - s * s;
  return T;
}

namespace {

Eigen::Matrix2d shear_rotation(double phi) {
  const double c = std::cos(phi), s = std::sin(phi);
  Eigen::Matrix2d R;
  R << c, s, -s, c;
  return R;
}

} // namespace

PlaneStressMaterial plane_stress_material(const Model &model, MaterialId mid,
                                          ElementId eid, std::string_view owner) {
  PlaneStressMaterial pm;
  if (const auto it = model.materials.find(mid); it != model.materials.end()) {
    const Mat1 &m = it->second;
    double E = m.E, nu = m.nu, G = m.G;
    if (G <= 0.0 && E > 0.0)
      G = E / (2.0 * (1.0 + nu));
    if (E <= 0.0 && G > 0.0)
      E = 2.0 * G * (1.0 + nu);
    const double c = E / (1.0 - nu * nu);
    pm.Q << c, c * nu, 0.0, c * nu, c, 0.0, 0.0, 0.0, G;
    pm.alpha << m.A, m.A, 0.0;
    pm.shear << G, 0.0, 0.0, G;
    pm.rho = m.rho;
    pm.ref_temp = m.ref_temp;
    return pm;
  }
  if (const auto it = model.mat2_materials.find(mid);
      it != model.mat2_materials.end()) {
    const Mat2 &m = it->second;
    pm.Q << m.g11, m.g12, m.g13, m.g12, m.g22, m.g23, m.g13, m.g23, m.g33;
    pm.alpha << m.a1, m.a2, m.a12;
    // Used as a transverse shear material (PSHELL MID3), MAT2's G11, G12
    // and G22 are the 2x2 shear moduli.
    pm.shear << m.g11, m.g12, m.g12, m.g22;
    pm.rho = m.rho;
    pm.ref_temp = m.ref_temp;
    return pm;
  }
  if (const auto it = model.mat8_materials.find(mid);
      it != model.mat8_materials.end()) {
    const Mat8 &m = it->second;
    if (m.e1 <= 0.0 || m.e2 <= 0.0)
      throw SolverError(std::format("{} {}: MAT8 {} needs positive E1 and E2",
                                    owner, eid.value, mid.value));
    const double nu21 = m.nu12 * m.e2 / m.e1;
    const double den = 1.0 - m.nu12 * nu21;
    if (den <= 0.0)
      throw SolverError(std::format(
          "{} {}: MAT8 {} Poisson ratios give a non-positive stiffness "
          "(1 - nu12 nu21 = {:.6g})",
          owner, eid.value, mid.value, den));
    pm.Q << m.e1 / den, m.nu12 * m.e2 / den, 0.0, //
        m.nu12 * m.e2 / den, m.e2 / den, 0.0,     //
        0.0, 0.0, m.g12;
    pm.alpha << m.a1, m.a2, 0.0;
    // A blank G1Z/G2Z leaves the transverse shear flexibility undefined;
    // the in-plane shear modulus is used as a finite substitute.
    pm.shear << (m.g1z > 0.0 ? m.g1z : m.g12), 0.0, 0.0,
        (m.g2z > 0.0 ? m.g2z : m.g12);
    pm.rho = m.rho;
    pm.ref_temp = m.ref_temp;
    return pm;
  }
  if (const char *card = model.material_card_name(mid))
    throw SolverError(std::format(
        "{} {}: material {} is a {} card, which is not a shell material "
        "(use MAT1, MAT2 or MAT8)",
        owner, eid.value, mid.value, card));
  throw SolverError(
      std::format("{} {}: material {} not found", owner, eid.value, mid.value));
}

ShellSection ShellSection::rotated(double phi) const {
  ShellSection out = *this;
  if (phi == 0.0)
    return out;
  // Material-frame strain = Te * target-frame strain; energy invariance
  // gives C_target = Te^T C_mat Te and resultants N_target = Te^T N_mat.
  const Eigen::Matrix3d Te = strain_rotation(phi);
  out.A = Te.transpose() * A * Te;
  out.B = Te.transpose() * B * Te;
  out.D = Te.transpose() * D * Te;
  out.thermal_N = Te.transpose() * thermal_N;
  out.thermal_M = Te.transpose() * thermal_M;
  const Eigen::Matrix2d R = shear_rotation(phi);
  out.T = R.transpose() * T * R;
  return out;
}

ShellSection ShellSection::scaled_thickness(double r) const {
  ShellSection out = *this;
  out.A *= r;
  out.B *= r * r;
  out.D *= r * r * r;
  out.T *= r;
  out.thermal_N *= r;
  out.thermal_M *= r * r;
  out.thickness *= r;
  out.z_bottom *= r;
  out.z_top *= r;
  out.z1 *= r;
  out.z2 *= r;
  out.mass_per_area *= r;
  out.mass_first_moment *= r * r;
  out.mass_second_moment *= r * r * r;
  return out;
}

namespace {

ShellSection pshell_section(const Model &model, const PShell &ps, ElementId eid,
                            std::string_view owner) {
  if (ps.t <= 0.0)
    throw SolverError(std::format("{} {}: PSHELL {} has non-positive thickness {}",
                                  owner, eid.value, ps.pid.value, ps.t));
  const PlaneStressMaterial m1 = plane_stress_material(model, ps.mid1, eid, owner);
  // A blank MID2 means a membrane-only shell in Nastran.  vibestran has no
  // AUTOSPC to remove the resulting rotational singularities, so a blank
  // MID2 falls back to MID1 and a blank MID3 to the bending material.
  const MaterialId mid2 = ps.mid2.value != 0 ? ps.mid2 : ps.mid1;
  const MaterialId mid3 = ps.mid3.value != 0 ? ps.mid3 : mid2;
  const PlaneStressMaterial m2 = plane_stress_material(model, mid2, eid, owner);
  const PlaneStressMaterial m3 = plane_stress_material(model, mid3, eid, owner);

  ShellSection s;
  const double t = ps.t;
  s.thickness = t;
  s.z_bottom = -0.5 * t;
  s.z_top = 0.5 * t;
  s.z1 = std::isnan(ps.z1) ? -0.5 * t : ps.z1;
  s.z2 = std::isnan(ps.z2) ? 0.5 * t : ps.z2;
  s.A = t * m1.Q;
  s.D = (ps.twelveI_t3 * t * t * t / 12.0) * m2.Q;
  s.T = (ps.tst * t) * m3.shear;
  if (ps.mid4.value != 0)
    s.B = t * t * plane_stress_material(model, ps.mid4, eid, owner).Q;
  s.thermal_N = s.A * m1.alpha;
  s.ref_temp = m1.ref_temp;
  s.mass_per_area = m1.rho * t + ps.nsm;
  s.mass_second_moment = m1.rho * t * t * t / 12.0;
  return s;
}

ShellSection pcomp_section(const Model &model, const PComp &pc, ElementId eid,
                           std::string_view owner) {
  if (pc.lam == LaminateOption::SmCore)
    throw SolverError(std::format(
        "{} {}: PCOMP {} LAM=SMCORE is not supported", owner, eid.value,
        pc.pid.value));
  ShellSection s;
  s.is_composite = true;
  s.thickness = pc.total_thickness();
  s.z_bottom = pc.bottom_z();
  s.z_top = s.z_bottom + s.thickness;
  s.z1 = s.z_bottom;
  s.z2 = s.z_top;
  s.ref_temp = pc.tref;

  // Physical laminate integrals (z upwards from the reference surface).
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero(), Bp = Eigen::Matrix3d::Zero(),
                  D = Eigen::Matrix3d::Zero();
  Eigen::Matrix2d T = Eigen::Matrix2d::Zero();
  Eigen::Vector3d NT = Eigen::Vector3d::Zero(), MTp = Eigen::Vector3d::Zero();
  double z = s.z_bottom;
  for (const PCompPly &ply : pc.plies) {
    const PlaneStressMaterial m = plane_stress_material(model, ply.mid, eid, owner);
    const double th = ply.theta * std::numbers::pi / 180.0;
    // Ply axes are at +theta from the laminate material axes.
    const Eigen::Matrix3d Te = strain_rotation(th);
    const Eigen::Matrix3d Qbar = Te.transpose() * m.Q * Te;
    const Eigen::Vector3d abar = Te.inverse() * m.alpha;
    const Eigen::Matrix2d R = shear_rotation(th);
    const Eigen::Matrix2d Gbar = R.transpose() * m.shear * R;
    const double z0 = z, z1 = z + ply.t;
    const double h1 = z1 - z0, h2 = 0.5 * (z1 * z1 - z0 * z0),
                 h3 = (z1 * z1 * z1 - z0 * z0 * z0) / 3.0;
    A += Qbar * h1;
    Bp += Qbar * h2;
    D += Qbar * h3;
    T += Gbar * h1;
    NT += Qbar * abar * h1;
    MTp += Qbar * abar * h2;
    s.mass_per_area += m.rho * h1;
    s.mass_first_moment += m.rho * h2;
    s.mass_second_moment += m.rho * h3;
    z = z1;
  }
  s.mass_per_area += pc.nsm;
  // First-order shear deformation theory with the 5/6 correction factor.
  T *= 5.0 / 6.0;

  switch (pc.lam) {
  case LaminateOption::Mem:
    s.A = A;
    s.thermal_N = NT;
    break;
  case LaminateOption::Bend:
    // Bending only, about the laminate's own centroid.
    {
      const double zc = 0.5 * (s.z_bottom + s.z_top);
      s.D = D - 2.0 * zc * Bp + zc * zc * A;
      s.T = T;
      s.thermal_M = -(MTp - zc * NT);
    }
    break;
  case LaminateOption::Smear:
    // Stacking sequence ignored: smeared membrane stiffness with uniform
    // bending, D = t^2/12 A, no coupling.
    s.A = A;
    s.D = (s.thickness * s.thickness / 12.0) * A;
    s.T = T;
    s.thermal_N = NT;
    break;
  default:
    s.A = A;
    s.B = -Bp; // Nastran curvature sign, see header
    s.D = D;
    s.T = T;
    s.thermal_N = NT;
    s.thermal_M = -MTp;
    break;
  }
  return s;
}

} // namespace

ShellSection build_shell_section(const Model &model, PropertyId pid,
                                 ElementId eid, std::string_view owner) {
  const Property &prop = model.property(pid);
  if (const auto *ps = std::get_if<PShell>(&prop))
    return pshell_section(model, *ps, eid, owner);
  if (const auto *pc = std::get_if<PComp>(&prop))
    return pcomp_section(model, *pc, eid, owner);
  throw SolverError(std::format("{} {}: property {} is not PSHELL or PCOMP",
                                owner, eid.value, pid.value));
}

} // namespace vibestran
