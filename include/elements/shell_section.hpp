#pragma once
// include/elements/shell_section.hpp
// Resultant constitutive description of a shell section (PSHELL or PCOMP),
// independent of the element formulation.
//
// Sign conventions (Nastran): N, M, Q are force and moment resultants per
// unit length.  The curvature kappa is defined so that a positive moment puts
// the -z (Z1) fibre in tension; in terms of the physical through-thickness
// strain gradient chi = d(eps)/dz, kappa = -chi and M = -integral(sigma z dz).
// With that convention
//
//     [N]   [A  B] [eps0 ]   [N_T]          Q = T gamma,
//     [M] = [B  D] [kappa] - [M_T] dT,
//
// where eps0 is the reference-surface strain (engineering shear) and dT the
// uniform temperature rise above the section reference temperature.  For
// PSHELL, B = T^2 G4 (MID4) as in Nastran; for PCOMP, B = -(physical
// laminate coupling) so the same convention holds.
//
// The matrices are expressed in the section's material frame.  Elements
// rotate them into their local frame with `rotated()`.

#include "core/model.hpp"
#include <Eigen/Dense>
#include <string_view>

namespace vibestran {

struct ShellSection {
  Eigen::Matrix3d A{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d B{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d D{Eigen::Matrix3d::Zero()};
  Eigen::Matrix2d T{Eigen::Matrix2d::Zero()};

  /// Thermal resultants per unit uniform temperature rise.
  Eigen::Vector3d thermal_N{Eigen::Vector3d::Zero()};
  Eigen::Vector3d thermal_M{Eigen::Vector3d::Zero()};
  double ref_temp{0.0};

  /// Total thickness and the reference-surface offsets of its faces.
  double thickness{0.0};
  double z_bottom{0.0};
  double z_top{0.0};
  /// Output fibre distances (PSHELL Z1/Z2; PCOMP bottom/top surfaces).
  double z1{0.0};
  double z2{0.0};

  /// Mass per unit area (including NSM) and its first and second moments
  /// about the reference surface (for offsets and rotary inertia).
  double mass_per_area{0.0};
  double mass_first_moment{0.0};
  double mass_second_moment{0.0};

  bool is_composite{false};

  /// The section in another in-plane frame, where `phi` is the angle
  /// (radians, counter-clockwise about the normal) of the material x axis
  /// measured from that frame's x axis.
  [[nodiscard]] ShellSection rotated(double phi) const;

  /// Section with every thickness-dependent quantity scaled for a local
  /// thickness ratio t_local / t_nominal (PSHELL grid thicknesses): A, T,
  /// N_T and mass scale with r, B and M_T with r^2, D with r^3.
  [[nodiscard]] ShellSection scaled_thickness(double r) const;
};

/// Builds the section of shell property `pid` for element `eid` (named in
/// error messages as `owner`).  Supports PSHELL (MID1-MID4 on MAT1, MAT2 or
/// MAT8) and PCOMP (classical lamination theory, LAM blank/SYM/MEM/BEND/
/// SMEAR).  Throws SolverError for missing or unsupported data.
[[nodiscard]] ShellSection build_shell_section(const Model &model, PropertyId pid,
                                               ElementId eid,
                                               std::string_view owner);

/// Engineering-strain rotation: eps_new = T(phi) eps_old for in-plane strain
/// [exx, eyy, gxy] when the new axes are at angle phi from the old ones.
[[nodiscard]] Eigen::Matrix3d strain_rotation(double phi);

/// Plane-stress stiffness of one material (MAT1, MAT2 or MAT8) in its own
/// material frame, and its thermal expansion vector [a1, a2, a12].
struct PlaneStressMaterial {
  Eigen::Matrix3d Q{Eigen::Matrix3d::Zero()};
  Eigen::Vector3d alpha{Eigen::Vector3d::Zero()};
  Eigen::Matrix2d shear{Eigen::Matrix2d::Zero()}; // transverse shear moduli
  double rho{0.0};
  double ref_temp{0.0};
};
[[nodiscard]] PlaneStressMaterial plane_stress_material(const Model &model,
                                                       MaterialId mid,
                                                       ElementId eid,
                                                       std::string_view owner);

} // namespace vibestran
