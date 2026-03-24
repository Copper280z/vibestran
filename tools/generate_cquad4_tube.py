#!/usr/bin/env python3
"""Generate a CQUAD4 cantilevered square tube BDF file.

Geometry
--------
Square hollow tube aligned along the X-axis with fixed physical dimensions:
  - Length         L = 1.0 m    (X extent)
  - Cross-section  W = 20 mm    (square outer dimension, Y and Z extent)
  - Wall thickness t = 0.5 mm   (default, configurable via --thickness)

The mesh is parameterized by the target element count.  Elements are sized to
be approximately square on the tube wall surface:
  - Element size d = sqrt(4 * L * W / n_target)
  - NC = max(1, round(W / d))  elements per wall across the cross-section
  - NX = max(1, round(L / d))  elements along the length
  - Actual element count = NX * 4 * NC

Node layout
-----------
Nodes are numbered as: node_id = ix * (4*NC) + ic + 1
  ix : 0 .. NX   (along X-axis)
  ic : 0 .. 4*NC-1  (around the closed cross-section perimeter)

The perimeter traversal is counter-clockwise when viewed from +X (free end):
  face 0 (bottom, z=0) : y from 0 to W,  z = 0
  face 1 (right,  y=W) : y = W,          z from 0 to W
  face 2 (top,    z=W) : y from W to 0,  z = W
  face 3 (left,   y=0) : y = 0,          z from W to 0

Element ordering G1=(ix,ic), G2=(ix,ic+1), G3=(ix+1,ic+1), G4=(ix+1,ic)
gives outward-pointing normals on all four faces.

Material (aluminum)
-------------------
  E     = 70.0e9  Pa
  nu    = 0.33
  alpha = 2.3e-5  /°C
  rho   = 2700.0  kg/m³

Load cases  (select with --load-case)
--------------------------------------
bending  (default)
  Cantilever: fixed end at x=0 (all 6 DOFs clamped), transverse load F=1000 N
  in +Z at the free end x=L, distributed uniformly over the 4*NC perimeter
  nodes at x=L.

  Analytical (Euler-Bernoulli, valid for L/W >> 1):
    I             = (W_o^4 - W_i^4) / 12    (second moment of area)
    delta_z_tip   = F * L^3 / (3 * E * I)
    sigma_xx_max  = F * L * (W/2) / I        (fixed-end top/bottom fibers)

combined
  Same cantilever setup, but with equal load components in +Y and +Z
  (F/2 in each direction).  By symmetry I_y = I_z so deflections are equal.

modal
  Fixed end at x=0 (all 6 DOFs clamped).  SOL 103 free vibration, first 10
  modes extracted by the Lanczos method.

  Approximate analytical first bending frequency (Euler-Bernoulli cantilever):
    f1 = (1.8751^2 / (2*pi*L^2)) * sqrt(E*I / (rho*A))

thermal
  Uniaxially constrained: x=0 face fully clamped, x=L face locked in T1 only
  to prevent axial thermal expansion.  Uniform temperature rise dT=100 °C.

  Analytical (uniaxial constraint):
    sigma_xx = -E * alpha * dT
    (lateral and rotational DOFs at x=L are free for Poisson contraction)

Usage
-----
  python3 generate_cquad4_tube.py N [--thickness T] [--load-case CASE] [-o FILE]

Examples
--------
  python3 generate_cquad4_tube.py 500  --load-case bending  -o tube_bending.bdf
  python3 generate_cquad4_tube.py 2000 --load-case modal
  python3 generate_cquad4_tube.py 1000 --thickness 0.001 --load-case thermal
  python3 generate_cquad4_tube.py 800  --load-case combined -o tube_combined.bdf
"""

import argparse
import math
import sys

# ── Material defaults (aluminum) ─────────────────────────────────────────────
E_DEFAULT = 70.0e9   # Pa    Young's modulus
NU_DEFAULT = 0.33     # –     Poisson's ratio
ALPHA_DEFAULT = 2.3e-5   # /°C   coefficient of thermal expansion
RHO_DEFAULT = 2700.0   # kg/m³ density

# ── Fixed tube geometry ───────────────────────────────────────────────────────
L_DEFAULT = 1.0      # m    tube length        (X extent)
W_DEFAULT = 0.020    # m    outer side length   (square cross-section)
T_DEFAULT = 0.0005   # m    wall thickness      (default 0.5 mm)

# ── Load magnitudes ───────────────────────────────────────────────────────────
F_TOTAL = 100.0   # N    total applied force (bending / combined cases)
DT = 100.0    # °C   temperature rise    (thermal case)


# ── Mesh helpers ──────────────────────────────────────────────────────────────

def node_id(ix: int, ic: int, nc: int) -> int:
    """Return 1-based node ID for mesh indices (ix along X, ic around perimeter).

    ix : 0 .. NX   (axial position)
    ic : 0 .. 4*NC-1  (circumferential position, CCW from +X)
    nc : elements per wall face around the cross-section
    """
    return ix * (4 * nc) + ic + 1


def cross_section_pos(ic: int, nc: int, w: float) -> tuple[float, float]:
    """Return (y, z) coordinates for perimeter index ic.

    Traversal order when viewed from +X (free end):
      face 0 (bottom, z=0) : y increases  0 → W
      face 1 (right,  y=W) : z increases  0 → W
      face 2 (top,    z=W) : y decreases  W → 0
      face 3 (left,   y=0) : z decreases  W → 0

    This CCW ordering makes the element normal point *outward* for the standard
    CQUAD4 connectivity G1=(ix,ic), G2=(ix,ic+1), G3=(ix+1,ic+1), G4=(ix+1,ic).
    """
    dw = w / nc
    face = ic // nc
    pos = ic % nc
    if face == 0:    # bottom (z = 0)
        return pos * dw, 0.0
    elif face == 1:  # right (y = W)
        return w, pos * dw
    elif face == 2:  # top (z = W)
        return w - pos * dw, w
    else:            # left (y = 0)
        return 0.0, w - pos * dw


# ── BDF card writers ──────────────────────────────────────────────────────────

def fmt_float(v: float) -> str:
    """Format a float for BDF output (scientific notation, 6 sig-figs)."""
    return f"{v:.6E}"


def write_cquad4(out, eid: int, pid: int, g1: int, g2: int, g3: int, g4: int) -> None:
    """Write a CQUAD4 free-field card."""
    out.write(f"CQUAD4,{eid},{pid},{g1},{g2},{g3},{g4}\n")


# ── Main generator ────────────────────────────────────────────────────────────

def generate(n_target: int, thickness: float, load_case: str, out_path: str) -> None:
    # ── Geometry ──────────────────────────────────────────────────────────────
    L = L_DEFAULT
    W = W_DEFAULT
    t = thickness

    # ── Mesh sizing ───────────────────────────────────────────────────────────
    # Unwrap the four walls into a flat sheet of area 4*L*W.
    # For approximately square elements of side d:
    #   n_target ≈ NX * 4 * NC  where  NX = L/d  and  NC = W/d
    # Solving:  d = sqrt(4*L*W / n_target)
    d = math.sqrt(4.0 * L * W / max(1, n_target))
    NC = max(1, round(W / d))
    NX = max(1, round(L / d))

    n_perimeter = 4 * NC
    n_nodes = (NX + 1) * n_perimeter
    n_elements = NX * n_perimeter

    # ── Material ──────────────────────────────────────────────────────────────
    E = E_DEFAULT
    nu = NU_DEFAULT
    alpha = ALPHA_DEFAULT
    rho = RHO_DEFAULT

    # ── Analytical quantities ─────────────────────────────────────────────────
    W_i = W - 2.0 * t
    # second moment of area (bending about centroidal axis)
    I = (W**4 - W_i**4) / 12.0
    A = W**2 - W_i**2             # cross-sectional area

    # Pre-compute per-load-case values (used in header comments and summary).
    delta_z_tip = 0.0
    delta_y_tip = 0.0
    sigma_xx_surf = 0.0
    sigma_xx_thermal = 0.0
    f1_hz = 0.0
    slenderness = L / W

    if load_case in ("bending", "combined"):
        F_z = F_TOTAL if load_case == "bending" else F_TOTAL / 2.0
        F_y = 0.0 if load_case == "bending" else F_TOTAL / 2.0
        delta_z_tip = F_z * L**3 / (3.0 * E * I)
        delta_y_tip = F_y * L**3 / (3.0 * E * I)
        sigma_xx_surf = F_z * L * (W / 2.0) / I
    elif load_case == "modal":
        beta1L = 1.8751   # first root of characteristic equation for cantilever
        f1_hz = (beta1L**2 / (2.0 * math.pi * L**2)) * \
            math.sqrt(E * I / (rho * A))
    elif load_case == "thermal":
        sigma_xx_thermal = -E * alpha * DT

    # ── Open output ───────────────────────────────────────────────────────────
    if out_path == "-":
        out = sys.stdout
    else:
        out = open(out_path, "w")

    def wr(s: str = "") -> None:
        out.write(s + "\n")

    # ── File header ───────────────────────────────────────────────────────────
    wr("$ CQUAD4 cantilevered square tube — generated by generate_cquad4_tube.py")
    wr(f"$ Target elements : {n_target}")
    wr(f"$ Actual elements : {n_elements}  ({
       NX} axial x {NC} per face x 4 faces)")
    wr(f"$ Total nodes     : {n_nodes}")
    wr(f"$ Geometry        : L = {L:.4g} m,  W = {
       W*1e3:.4g} mm x {W*1e3:.4g} mm,  t = {t*1e3:.4g} mm")
    wr(f"$ Element size    : d ~ {d*1e3:.4g} mm  (NX={NX}, NC={NC} per wall)")
    wr(f"$ Material (Al)   : E = {E:.4g} Pa,  nu = {
       nu},  alpha = {alpha:.4g} /deg,  rho = {rho:.4g} kg/m3")
    wr("$")
    wr("$ ── Analytical reference values ─────────────────────────────────────────")
    wr(f"$   Hollow square section : W_o = {
       W*1e3:.4g} mm,  W_i = {W_i*1e3:.4g} mm")
    wr(f"$   I = (W_o^4 - W_i^4)/12 = {I:.6G} m^4")
    wr(f"$   A = W_o^2 - W_i^2     = {A:.6G} m^2")
    if load_case == "bending":
        wr(f"$   Load case    : BENDING  (F = {
           F_TOTAL} N in +Z, cantilever at x=0)")
        wr(f"$   L/W ratio    : {slenderness:.1f}  (Euler-Bernoulli valid for L/W >> 1)")
        wr(f"$   delta_z_tip  = F*L^3/(3*E*I) = {delta_z_tip:.6G} m")
        wr(f"$   sigma_xx_max = F*L*(W/2)/I   = {
           sigma_xx_surf:.6G} Pa  (fixed-end top/bottom fiber)")
    elif load_case == "combined":
        wr(f"$   Load case    : COMBINED  (F = {
           F_TOTAL} N split equally in +Y and +Z)")
        wr(f"$   L/W ratio    : {slenderness:.1f}")
        wr(f"$   delta_y_tip = delta_z_tip = {
           delta_z_tip:.6G} m  (I_y = I_z by symmetry)")
        wr(f"$   sigma_xx_max from Z-bending = {
           sigma_xx_surf:.6G} Pa  (fixed-end top/bottom fiber)")
    elif load_case == "modal":
        wr(f"$   Load case : MODAL  (free vibration, first 10 modes)")
        wr(f"$   Approx first bending mode (Euler-Bernoulli cantilever):")
        wr(f"$     f1 = (1.8751^2/(2*pi*L^2)) * sqrt(E*I/(rho*A)) = {
           f1_hz:.6G} Hz")
    elif load_case == "thermal":
        wr(f"$   Load case : THERMAL  (dT = {
           DT} deg, uniaxially constrained tube)")
        wr(f"$   BCs       : x=0 fully clamped; x=L locked in T1 only")
        wr(f"$   sigma_xx  = -E * alpha * dT = {sigma_xx_thermal:.6G} Pa")
    wr("$")

    # ── Case control section ──────────────────────────────────────────────────
    wr("SOL 103" if load_case == "modal" else "SOL 101")
    wr("CEND")
    wr("SUBCASE 1")
    labels = {
        "bending":  "CQUAD4 CANTILEVER BENDING",
        "combined": "CQUAD4 CANTILEVER COMBINED BENDING",
        "modal":    "CQUAD4 CANTILEVER MODAL",
        "thermal":  "CQUAD4 THERMAL UNIAXIALLY CONSTRAINED",
    }
    wr(f"  LABEL = {labels[load_case]}")
    wr("  DISPLACEMENT(PRINT) = ALL")
    if load_case == "modal":
        wr("  METHOD = 1")
    else:
        wr("  STRESS(PRINT) = ALL")
        wr("  LOAD  = 1")
    wr("  SPC   = 1")
    wr("BEGIN BULK")

    # ── Material (aluminum) ───────────────────────────────────────────────────
    wr("$")
    wr("$ ── Material (aluminum) ────────────────────────────────────────────────")
    # MAT1 fields: MID, E, G(blank), NU, RHO, A(alpha), TREF
    wr(f"MAT1,1,{fmt_float(E)},,{fmt_float(nu)},{
       fmt_float(rho)},{fmt_float(alpha)},0.0")

    # ── Shell property ────────────────────────────────────────────────────────
    wr("$")
    wr("$ ── Property ───────────────────────────────────────────────────────────")
    # PSHELL fields: PID, MID1, T  (MID2 defaults to MID1 for bending)
    wr(f"PSHELL,1,1,{fmt_float(t)}")

    # ── Eigensolver (modal only) ──────────────────────────────────────────────
    if load_case == "modal":
        wr("$")
        wr("$ ── Eigenvalue method ──────────────────────────────────────────────────")
        wr("EIGRL,1,0.0,,10")

    # ── Nodes ─────────────────────────────────────────────────────────────────
    wr("$")
    wr("$ ── Nodes ──────────────────────────────────────────────────────────────")
    dx = L / NX
    for ix in range(NX + 1):
        x = ix * dx
        for ic in range(n_perimeter):
            y, z = cross_section_pos(ic, NC, W)
            nid = node_id(ix, ic, NC)
            wr(f"GRID,{nid},,{fmt_float(x)},{fmt_float(y)},{fmt_float(z)}")

    # ── Elements ──────────────────────────────────────────────────────────────
    wr("$")
    wr("$ ── Elements ───────────────────────────────────────────────────────────")
    # Element normal convention: CCW traversal (ic increasing) gives outward normal.
    # Verified analytically for all four faces; see module docstring.
    eid = 1
    for ix in range(NX):
        for ic in range(n_perimeter):
            ic_next = (ic + 1) % n_perimeter
            g1 = node_id(ix,     ic,      NC)
            g2 = node_id(ix,     ic_next, NC)
            g3 = node_id(ix + 1, ic_next, NC)
            g4 = node_id(ix + 1, ic,      NC)
            write_cquad4(out, eid, 1, g1, g2, g3, g4)
            eid += 1

    # ── Boundary conditions ───────────────────────────────────────────────────
    wr("$")
    wr("$ ── Boundary Conditions ────────────────────────────────────────────────")

    # Perimeter nodes at x=0 and x=L are contiguous (ic = 0..n_perimeter-1).
    x0_first = node_id(0,  0,              NC)
    x0_last = node_id(0,  n_perimeter - 1, NC)
    xL_first = node_id(NX, 0,              NC)
    xL_last = node_id(NX, n_perimeter - 1, NC)

    if load_case == "thermal":
        # Uniaxial constraint: fully clamp x=0, lock T1 at x=L.
        wr(f"$ x=0 face: all 6 DOFs clamped.")
        wr(f"SPC1,1,123456,{x0_first},THRU,{x0_last}")
        wr(f"$ x=L face: T1 locked to prevent axial thermal expansion.")
        wr(f"SPC1,1,1,{xL_first},THRU,{xL_last}")
    else:
        # Standard cantilever: fully clamp x=0 face (all 6 DOFs).
        wr(f"$ Fixed end (x=0): all 6 DOFs clamped.")
        wr(f"SPC1,1,123456,{x0_first},THRU,{x0_last}")

    # ── Loads ─────────────────────────────────────────────────────────────────
    if load_case != "modal":
        wr("$")
        wr("$ ── Loads ──────────────────────────────────────────────────────────────")

    if load_case == "bending":
        fz_each = F_TOTAL / n_perimeter
        wr(f"$ Transverse force: {F_TOTAL} N in +Z distributed over {n_perimeter} free-end nodes"
           f" ({fz_each:.6G} N each).")
        for ic in range(n_perimeter):
            nid = node_id(NX, ic, NC)
            wr(f"FORCE,1,{nid},0,{fmt_float(fz_each)},0.0,0.0,1.0")

    elif load_case == "combined":
        fy_each = (F_TOTAL / 2.0) / n_perimeter
        fz_each = (F_TOTAL / 2.0) / n_perimeter
        wr(f"$ Combined bending: {
           F_TOTAL/2:.0f} N in +Y and {F_TOTAL/2:.0f} N in +Z,")
        wr(f"$ split over {
           n_perimeter} free-end nodes ({fy_each:.6G} N/node each direction).")
        for ic in range(n_perimeter):
            nid = node_id(NX, ic, NC)
            wr(f"FORCE,1,{nid},0,{fmt_float(fy_each)},0.0,1.0,0.0")
            wr(f"FORCE,1,{nid},0,{fmt_float(fz_each)},0.0,0.0,1.0")

    elif load_case == "thermal":
        wr("$ Reference temperature (stress-free state): 0 deg.")
        wr("TEMPD,1,0.0")
        wr(f"$ Nodal temperatures: all {n_nodes} nodes at dT = {DT} deg.")
        all_nids = list(range(1, n_nodes + 1))
        for chunk_start in range(0, len(all_nids), 3):
            chunk = all_nids[chunk_start:chunk_start + 3]
            pairs = ",".join(f"{nid},{DT:.1f}" for nid in chunk)
            wr(f"TEMP,1,{pairs}")

    wr("ENDDATA")

    # ── Close file and print summary ──────────────────────────────────────────
    if out_path != "-":
        out.close()

    summary = [
        f"Written: {out_path if out_path != '-' else '(stdout)'}",
        f"  Mesh     : {NX} axial x {NC} per face x 4 faces = {
            n_elements} CQUAD4 elements,"
        f" {n_nodes} nodes",
        f"  Geometry : L = {L:.4g} m,  W = {W*1e3:.4g} mm,  t = {t*1e3:.4g} mm"
        f"  (W_i = {W_i*1e3:.4g} mm)",
        f"  I = {I:.6G} m^4,  A = {A:.6G} m^2",
    ]
    if load_case == "bending":
        summary += [
            f"  Expected : delta_z_tip   = {delta_z_tip:.6G} m",
            f"           : sigma_xx_max  = {
                sigma_xx_surf:.6G} Pa  (fixed-end top/bottom fiber)",
        ]
    elif load_case == "combined":
        summary += [
            f"  Expected : delta_y_tip = delta_z_tip = {delta_z_tip:.6G} m",
            f"           : sigma_xx_max (Z-bending)  = {sigma_xx_surf:.6G} Pa",
        ]
    elif load_case == "modal":
        summary.append(f"  Expected : f1 ~ {
                       f1_hz:.6G} Hz  (E-B cantilever bending)")
    elif load_case == "thermal":
        summary.append(f"  Expected : sigma_xx = {
                       sigma_xx_thermal:.6G} Pa  (uniaxial constraint)")
    print("\n".join(summary), file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a CQUAD4 cantilevered square tube BDF file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "n_elements",
        type=int,
        help="Target number of CQUAD4 elements.  Elements are sized to be approximately "
             "square on the tube wall surface.",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=T_DEFAULT,
        metavar="T",
        help=f"Wall thickness in metres (default: {T_DEFAULT} m = 0.5 mm).",
    )
    parser.add_argument(
        "--load-case",
        choices=["bending", "combined", "modal", "thermal"],
        default="bending",
        help="Load case to generate (default: bending).",
    )
    parser.add_argument(
        "-o", "--output",
        default="-",
        metavar="FILE",
        help="Output BDF file path.  Use '-' for stdout (default).",
    )
    args = parser.parse_args()

    if args.n_elements < 1:
        parser.error("n_elements must be at least 1")
    if args.thickness <= 0.0:
        parser.error("--thickness must be a positive value")

    generate(args.n_elements, args.thickness, args.load_case, args.output)


if __name__ == "__main__":
    main()
