#pragma once
// include/elements/thermal_elements.hpp
// Concrete thermal element classes for linear steady-state heat conduction.
//
// Coverage (MVP):
//   - ThermalLine   — 1D axial conduction (CROD / CBAR / CTUBE / CONROD)
//   - ThermalTetra4 — constant-gradient tetrahedron (closed-form)
//   - ThermalHexa8  — trilinear 8-node hex, 2×2×2 Gauss
//   - ThermalTetra10 — quadratic 10-node tet, 4-point Gauss
//   - ThermalPenta6 — linear 6-node wedge, 6-point Gauss
//
// All elements share the same scalar-T-per-node DOF model.

#include "elements/thermal_element_base.hpp"

namespace vibestran {

// Build a thermal element from existing ElementData + Model.  Returns nullptr
// if the element type has no thermal counterpart.
std::unique_ptr<ThermalElement>
make_thermal_element(const ElementData &data, const Model &model);

// Resolve the thermal material (MAT4 isotropic k, or MAT5 conductivity tensor)
// referenced by an element's property.  Throws if neither is present.
struct ThermalMaterial {
  bool isotropic{true};
  double k{0.0};                       // isotropic
  Eigen::Matrix3d k_tensor{Eigen::Matrix3d::Zero()}; // anisotropic
  double cp{0.0};
  double rho{0.0};                     // ρ × cp lumped into cp by NASTRAN MAT4;
                                       // we treat MAT4.cp as the volumetric ρcp
};

ThermalMaterial resolve_thermal_material(const Model &model, MaterialId mid);

// Extract the MID referenced by a property card (PSOLID/PSHELL/PBAR-like).
MaterialId property_thermal_mid(const Model &model, PropertyId pid);

} // namespace vibestran
