# Shell Decks Against Refined CHEXA8 Solids: vibestran vs MYSTRAN

Date: 2026-10-16. vibestran at commit 2540fc2 plus the orthotropic thermal
fix described below. MYSTRAN is the local build
`MYSTRAN/Binaries/mystran`, version 19.0.0, with the Q1/E21 enhanced-strain
CHEXA8. That binary is newer than every Fortran source file. The suite's
stored MYSTRAN references (`MYSTRAN_Validation/reference_mystran`) came from
MYSTRAN 18.0.0 and are listed as "MYSTRAN v18".

The open questions were the shell decks whose correct answer was in doubt:
the hemisphere-with-cutout decks, the thick twisted plate, the Raasch hook,
the diamond-mesh cylinder under pressure, and the two free-thermal-expansion
cylinders. For each one there is a CHEXA8 model of the same body, refined
until it converges, run in both solvers.

Generators and drivers are in `tools/solid_reference/`:

| File | Purpose |
|---|---|
| `cases.py` | One builder per CHEXA8 reference model |
| `run_solids.py` | Refinement series in both solvers, written to `solid_results.json` |
| `run_shells.py` | The original shell decks in both solvers |
| `refine_cylinder_deck.py` | Splits the diamond cylinder deck's shell mesh 1:4 on the true cylinder |

The decks use fixed-format fields, since MYSTRAN rejects free-field values
longer than 16 characters.

```bash
cd tools/solid_reference && python3 run_solids.py /tmp/solid_refs
```

## 1. Which CHEXA8 is a trustworthy reference

The refined hex8 results in both solvers:

| Model | Mesh | vibestran CHEXA8 | MYSTRAN CHEXA8 |
|---|---|---:|---:|
| Twisted plate t = 0.6, corner TZ | 8x8x2 | -1.6010e-4 | -1.5400e-4 |
| | 16x16x4 | -1.6584e-4 | -1.6386e-4 |
| | 32x32x6 | -1.7166e-4 | -1.7032e-4 |
| | 48x48x8 | -1.7278e-4 | -1.7201e-4 |
| | 64x64x10 | -1.7367e-4 | **-1.7308e-4** |
| Twisted plate t = 0.2 | 16x16x2 / 32x32x4 / 64x64x4 | -1.6482 / -1.6502 / -1.6645e-3 | -1.6057 / -1.6392 / **-1.6415e-3** |
| Twisted plate t = 0.06 | 16 / 32 / 64 / 96 (x2) | -1.1314 / -1.1615 / -1.1865 / -1.2080e-2 | -1.1342 / -1.1448 / -1.1490 / **-1.1499e-2** |
| Hemisphere t = 0.04, TX at load | 16 / 32 / 64 / 96 (x2) | 0.00146 / 0.00619 / 0.0198 / 0.0361 | 0.09253 / 0.09345 / 0.09354 / **0.09357** |
| Hemisphere t = 0.004 | 16 / 32 / 64 / 96 / 128 (x2) | 1.4e-5 / 5.6e-5 / 2.5e-4 / 5.7e-4 / 9.9e-4 | 0.0600 / 0.09016 / 0.09146 / 0.09148 / **0.09154** |
| Raasch hook, free-end TY (m) | 72x10x2 / 144x20x2 / 288x40x4 / 432x60x4 | 0.1443 / 0.1436 / 0.1310 / 0.1308 | 0.12464 / 0.12603 / 0.12712 / **0.12724** |
| Cylinder pressure, TY at y = 25 | 48x14 / 96x28 / 192x56 (x2) / 288x84x4 | -6.3191 / -6.3048 / -6.3012 / -6.3005e-6 | -6.3190 / -6.3048 / -6.3012 / **-6.3005e-6** |
| Half cylinder, uniform dT | 24x16x2, 48x32x2 | exact (u = alpha dT x) | exact |

MYSTRAN's Q1/E21 CHEXA8 converges cleanly in every case. It is exactly
symmetric where the problem is (hemisphere TX = -TY), and its Raasch value
(0.1272) matches the published C3D20R solid (0.1275 m, 5.02 in). It is the
reference used below.

**vibestran's CHEXA8 is not reliable on thin or curved geometry.** It uses
9-mode Wilson-Taylor incompatible modes.
- **Hemisphere:** it locks almost completely, below 40% of the answer even
  at 96x96.
- **Symmetry:** it breaks the hemisphere's x-y symmetry (TX 0.0361 against
  |TY| 0.0420).
- **Raasch hook and thin twisted plate:** it converges to a wrong, too
  flexible value (Raasch 0.1308, +2.8%; twisted plate t = 0.06 still drifting
  at 96x96, +5%).

These are the classic signs of an incompatible-mode element that fails the
patch test on non-parallelepiped shapes. It agrees with MYSTRAN only where
the elements are nearly undistorted: the thick twisted plate (0.3%), the
cylinder (1e-5), and thermal expansion.

The twisted-plate "solid" column in an earlier version of
`SHELL_BENCHMARKS.md` came from vibestran's CHEXA8. It is corrected there to
the MYSTRAN values.

## 2. The shell decks

"Solid" is the converged MYSTRAN CHEXA8. Errors are relative to it.

### Pinched hemisphere with 18-degree hole

Each deck holds five meshes. The values are TX at the +x load of the finest
mesh (grid 1485).

| Deck | Solid | vibestran | MYSTRAN v19 | MYSTRAN v18 |
|---|---:|---:|---:|---:|
| t = 0.04, `QUAD4TYP MITC4` | 0.0936 | 0.09346 (-0.1%) | 0.09470 (+1.2%) | 0.09470 |
| t = 0.04, MITC4+ | 0.0936 | 0.09346 (-0.1%) | 0.09517 (+1.7%) | 0.09517 |
| t = 0.04, MITC4+ SNORM | 0.0936 | 0.09342 (-0.2%) | 0.09340 (-0.2%) | 0.09338 |
| t = 0.004, MITC4+ | 0.0916 | 0.09135 (-0.3%) | 0.09142 (-0.2%) | 0.09142 |
| t = 0.004, MITC4+ SNORM | 0.0916 | 0.09130 (-0.3%) | 0.09130 (-0.3%) | 0.09129 |
| t = 0.004, SNORM K6ROT | 0.0916 | 0.09130 (-0.3%) | 0.09127 (-0.4%) | 0.09125 |

On the finest meshes both solvers are within 0.4% of the solid, except
MYSTRAN without SNORM at t = 0.04. That run is 1.2-1.7% flexible and not
symmetric: TX 0.09517 against TY -0.09456.

On coarse meshes the formulations differ a lot. At t = 0.04 with MITC4+, the
five meshes give:
- vibestran: 0.0228 / 0.0803 / 0.0919 / 0.0931 / 0.0935
- MYSTRAN: 0.0752 / 0.0907 / 0.0928 / 0.0937 / 0.0952

So MYSTRAN is closer on the coarsest mesh, and vibestran converges
monotonically to the solid. At t = 0.004, vibestran is the stiffer of the two
on the two coarsest meshes (0.0004 / 0.0081 against 0.026 / 0.078). That is
the remaining membrane locking of a 4-node shell at R/t = 2500 on 4x4 and
8x8 meshes.

**Verdict:** equivalent and both correct on converged meshes. MYSTRAN is
better on very coarse thin meshes; vibestran is better without SNORM at
t = 0.04.

### Twisted plate, t/L = 0.6 (`NAS S30 twisted plate mit4+`)

Corner TZ on the deck's six meshes, finest last:

| | Solid | vibestran | MYSTRAN v19 | MYSTRAN v18 (stored) |
|---|---:|---:|---:|---:|
| Finest mesh | -1.731e-4 | -2.360e-4 (+36%) | -3.712e-4 (+114%) | -1.887e-4 (+9%) |

The thickness sweep below uses a 32x32 vibestran shell mesh of the same
plate:

| t / L | Solid | vibestran (first-order metric) | vibestran before it |
|---|---:|---:|---:|
| 0.06 | -1.1499e-2 | +0.50% | +1.3% |
| 0.2 | -1.6415e-3 | +4.8% | +12% |
| 0.6 | -1.731e-4 | +35% | +114% |

**Verdict:** MYSTRAN v18 was the most accurate here, because its shell
integrates the 3D metric exactly. The current MYSTRAN v19 build is no better
than vibestran before the metric fix: v18's -1.887e-4 became -3.712e-4.
vibestran's remaining error falls roughly as t^2. Exact through-thickness
integration (the proposed follow-up) should close it.

### Raasch hook

Free-end corner TY (m). The deck holds four meshes, 1, 3, 10 and 20 elements
across the width.

| Deck | vibestran | MYSTRAN v19 | MSC Nastran (stored) |
|---|---|---|---|
| K6ROT = 10 | 0.1175 / 0.1218 / 0.1257 / **0.1271** | 0.1173 / 0.1302 / 0.2402 / **0.3859** | 0.1213 / 0.1352 / 0.2416 / **0.3866** |
| K6ROT = 1000 | same | 0.1166 / 0.1242 / 0.1332 / **0.1358** | 0.1200 / 0.1285 / 0.1340 / **0.1361** |

The solid is 0.1272; the published value is 0.1275.

**Verdict:** vibestran is clearly the most accurate, within 0.1% of the
solid, and independent of K6ROT.
- With K6ROT = 10, MYSTRAN and MSC Nastran diverge with refinement (+203%).
- With K6ROT = 1000 they converge about 7% too flexible.

The suite's Raasch criteria are MSC Nastran's numbers, so they reward a
K6ROT-dependent answer.

### Cylinder under pressure, diamond mesh (`diamond cylinder pressure PLOAD2/PLOAD4`)

TY at grid 106, y = 25. The exact value is -6.300e-6. The solid with the
deck's supports, including the radial pin at grid 56, gives -6.3005e-6. With
statically determinate supports it gives -6.3011e-6.

Where the pressure acts matters at the 1.7% level:
- mid-surface (split between the faces): -6.301e-6;
- inner face: -6.196e-6;
- outer face: -6.406e-6.

Shells load the mid-surface.

The deck's mesh, then the same mesh split 1:4 one, two and three times:

| Mesh | vibestran | MYSTRAN v19 |
|---|---:|---:|
| Deck mesh | -6.009e-6 (-4.6%) | -6.063e-6 (-3.8%) |
| 1 split | -6.220e-6 | -6.206e-6 |
| 2 splits | -6.278e-6 | -6.276e-6 |
| 3 splits | -6.2943e-6 | -6.2939e-6 |

The refiner rewrites the deck's quads and triangles in small-field form
without THETA, which is irrelevant for MAT1. That rewrite alone (level 0)
moves vibestran's value by 0.01% (-6.0086e-6) and MYSTRAN's by 0.6%
(-6.0985e-6).

**Verdict:** both shells converge to the exact value at the same rate. The
deck's 3-5% is coarse-mesh error in both codes; MYSTRAN is 0.8 points closer
on the coarse mesh. The suite's criterion (-6.21e-6, 3%) is from neither
MYSTRAN build: v18 and v19 both give -6.063e-6.

### Free thermal expansion, half cylinders (`thermal strain free rectangle / diamond cylinder`)

Uniform dT = 100 with supports that allow free expansion. The exact answer is
zero stress and u = alpha dT (x - x0); the CHEXA8 models reproduce it exactly
in both solvers.

| Deck | Quantity | Exact | vibestran | MYSTRAN v19 | MYSTRAN v18 (stored; the suite's criteria) |
|---|---|---:|---:|---:|---:|
| Rectangle, MAT1 | TX grids 56, 66 | -0.18 | -0.18 | -0.18 | 8e-14 |
| | TZ grids 50, 59 | 0.09 | 0.09 | 0.09 | 0.141 |
| Diamond, MAT8 (alpha1 7e-4 axial, alpha2 2e-4 hoop) | TY grids 23, 45 | 0.21 | 0.2099 | 0.2099 | 0.2098 |
| | TZ grids 63, 200 | 0.06 | 0.0604 (was 0.0177) | 0.0600 | 0.0944 |
| | TX grids 58, 66 | -0.12 | -0.1184 (was -0.267) | -0.1201 | 0.0003 |

The vibestran "was" values are before the fix below.

**Verdict:** vibestran (after the fix) and MYSTRAN v19 match the exact
solution. MYSTRAN v18, which the suite's criteria encode, is wrong for both
decks.

Both codes leave residual membrane stress on the coarse diamond mesh, where
the exact answer is zero:

| Element | vibestran | MYSTRAN |
|---|---:|---:|
| 190 (coarse cylinder) | 6.0e7 | 5.8e7 |
| 322 (fine cylinder) | 3.3e6 | 3.6e6 |

That is discretization of the warped facets.

## 3. Bug found and fixed: orthotropic thermal expansion of curved shells

The curved-shell thermal correction builds the free-expansion field
u = E (x - x0) to which the element's own strain operator is applied. It used
the mean in-plane alpha for E. For MAT8/MAT2/PCOMP with alpha1 != alpha2 that
is not the free expansion.

On the diamond deck, whose material axes are 45 degrees to the element axes,
the hoop expansion came out 120% too large (TX -0.267 instead of -0.12).
E is now the section's in-plane free-strain tensor. Its normal component is
the strain along the curved directions, tr(b e)/tr(b), with b the director
curvature: a cylinder grows radially with its hoop strain. Isotropic
expansion is unchanged.

Tests: `ShellQuad4/ShellTria3.OrthotropicFreeExpansionOfACylinderIsExact`, a
36x4 MAT8 cylinder.
- **Before the fix:** 0.97% / 0.62% error.
- **After:** exact to 1e-11.

## 4. Summary

| Case | Closest to the solid |
|---|---|
| Hemisphere, converged meshes | Both (within 0.4%); MYSTRAN better on 4x4/8x8 thin meshes |
| Twisted plate t/L = 0.6 | MYSTRAN v18 (+9%), then vibestran (+36%), then MYSTRAN v19 (+114%) |
| Raasch hook | vibestran (-0.1%); MYSTRAN and MSC are 7% off or diverge, depending on K6ROT |
| Diamond cylinder pressure | Both converge identically; MYSTRAN 0.8 points closer on the coarse deck mesh |
| Free thermal expansion (MAT1, MAT8) | vibestran and MYSTRAN v19 exact; MYSTRAN v18 (the suite's references) wrong |
| CHEXA8 itself | MYSTRAN's Q1/E21 is accurate; vibestran's incompatible-mode hex8 locks and converges wrongly on curved geometry |
