# CHEXA20 formulation proposal

Status (2026-10-16): asnl2 is implemented in vibestran (`SolidHexa20`, `src/elements/solid_hexa20.cpp`);
the MYSTRAN port is to follow. The numbers below come from a numpy prototype (`tools/hex20_prototype/`,
run with `MYSTRAN/.venv/bin/python`) on MYSTRAN's own comparison decks (`MYSTRAN/hexa_msc_comparison/`,
branch `feat/hexa-modern`); vibestran reproduces them to 2e-6 or better
(`tests/integration/test_hexa20_reference.cpp`).

## 1. The problem

MYSTRAN's current CHEXA20 ("assumed strain": shear projected onto the trilinear space in the centroid
natural frame, plus a B-bar dilatation) is consistent, invariant and free of volumetric locking. However, it
locks in membrane action on thin curved solids. One element through the thickness, shell-theory reference:

| deck | MYSTRAN now | Simcenter 2412 | ComLab |
|---|---:|---:|---:|
| pinched hemisphere 8x8 | 55.1% | 96.6% | 98.7% |
| pinched hemisphere 12x12 | 85.2% | 99.0% | 99.5% |
| pinched cylinder 8x8 | 88.0% | 93.0% | 96.9% |
| pinched cylinder 12x12 | 96.6% | 98.3% | 99.1% |
| Lame cylinder, nu = 0.4999 | 100.0% | 93.6% | 99.8% |

ComLab gets its thin-shell accuracy by giving up the curved-edge patch test (0.9% displacement error) and
connectivity invariance. Simcenter keeps both, but locks volumetrically. The target in
`MYSTRAN/dev_docs/chexa_formulation.md` is at least 95% at 8x8 and about 99% at 12x12. It must keep the patch
test, invariance, full rank, and the flat-plate and incompressible accuracy.

vibestran has no CHEXA20: it parses the card but rejects it in the solver.

## 2. Diagnosis

The current element leaves the three **normal** strains untouched. On a curved element they carry the
parasitic membrane strain of inextensional bending. For a curved quadratic edge, the strain along the edge
picks up a component quadratic in the edge coordinate (the w/R term). The quadratic displacement field
cannot cancel that component, so it locks. This is the same mechanism that MITC9-type shells and
2-point reduced integration remove.

On a straight-edged (affine) element, the compatible normal strain e_ii is already at most linear in its
own coordinate xi_i: d/dxi_i of the serendipity space has no xi_i^2 term. So removing the xi_i^2 component
of e_ii changes nothing on flat or affine elements. It acts only where the geometry is curved, which is
exactly where the locking is.

The prototype confirms this. Plain 2x2x2 integration ("red") gives 99.1% on the 8x8 hemisphere, which
places the locking in the strain field rather than the mesh. That element has 6 spurious modes and fails
the curved patch test by 0.9%, the same figure as ComLab.

## 3. Proposed formulation

A projected assumed-strain element with **directional normal-strain spaces** and **local natural frames**.
A two-sided mean correction keeps it exactly consistent. There are no internal parameters, no
stabilization constants and no preferred direction.

Notation: serendipity shape functions N_a(xi), xi = (xi_1, xi_2, xi_3). 3x3x3 Gauss points g with
dV_g = w_g det J_g. Compatible B_g (6x60, Cartesian engineering Voigt). V = sum_g dV_g.

1. **Mean strain.** B_m = (1/V) sum_g dV_g B_g.
2. **Local natural frame** at each point: t_i = J e_i / |J e_i| (columns of J scaled to unit length).
   T_g maps Cartesian Voigt strain to the natural components (e_11, e_22, e_33, g_12, g_23, g_31), with
   e_ii = t_i.eps.t_i and g_ij = 2 t_i.eps.t_j. C_g = inv(T_g).
3. **Spaces.** For each normal strain e_ii, S_i = {xi^a : a_k <= 2, a_i <= 1} (18 functions: linear in
   xi_i, quadratic in the other two). For the shear strains and the dilatation, the trilinear space
   P8 = {xi^a : a_k <= 1}. Pi_S is the det(J)-weighted L2 projection onto S over the element.
4. **Assumed-strain operator**, applied to a strain field F_g:
   A(F)_g = C_g [ Pi_S1(TF)_1, Pi_S2(TF)_2, Pi_S3(TF)_3, Pi_P8(TF)_4, Pi_P8(TF)_5, Pi_P8(TF)_6 ]_g.
5. **Two-sided consistency correction.** D_g = A(B - B_m)_g, then
   Bs_g = B_m + D_g - (1/V) sum_h dV_h D_h.
6. **B-bar dilatation** (unchanged from MYSTRAN): Bbar_g = Bs_g + (m/3) (Pi_P8(m'Bs) - m'Bs)_g,
   m = (1,1,1,0,0,0).
7. K = sum_g dV_g Bbar' D Bbar, at 3x3x3. The same Bbar is used for the thermal load
   sum_g dV_g Bbar' D eps_th and for recovery, sigma = D (Bbar u - eps_th). Every projected field is a
   polynomial in xi, so Bbar can be evaluated at the centroid and corners directly, without extrapolation
   (Section 6 covers bad corners). Mass and differential stiffness use the compatible N.

### Properties

* **Constant strain is exact on any shape.** For a linear u, B u = B_m u = eps_0, so D = 0 and
  Bbar u = eps_0.
* **No work against constant stress.** For a constant sigma_0, sum_g dV_g (Bbar - B)' sigma_0 = 0.
  The mean correction makes D zero-mean, and every projection space contains 1. Together with the
  previous property this is the patch test in any mesh, including curved, off-centre midside nodes.
  Without the second property, a local-frame element passes a single-element energy check but fails the
  mesh patch test by 5.8% (variant "asnl" below).
* **Invariance.** The frames are covariant. The 24 orientation-preserving node relabelings permute xi
  together with S_1, S_2, S_3, and P8 is symmetric, so the element does not depend on node order or
  orientation.
* **Material-agnostic and semidefinite.** The projections do not involve D, so MAT9 anisotropy and
  thermal coupling need no special treatment, and K = sum Bbar' D Bbar >= 0 for any admissible D.
* **Affine elements are unchanged.** With T constant and e_ii already in S_i, the element reduces exactly
  to MYSTRAN's current assumed-strain element on parallelepipeds. Flat plates, the twisted beam, the
  regular modal plates and the Lame cylinder keep their current accuracy.
* **Rank.** Full rank (54 positive and 6 rigid modes) is not proven; it is checked numerically below. A
  projection that cannot be rank-deficient is not available for this space; that is why a rank test is part
  of acceptance (Section 5).

A simpler fallback, **"asn"**, keeps the current fixed centroid frame and adds only the directional
normal-strain projection (no mean correction needed). It is about 2 points worse on the coarse
cylinders.

## 4. Prototype results

Variants:
* **as** is MYSTRAN's current element. The prototype reproduces it to every printed digit
  (hemisphere 8x8 0.0517536, cylinder 8x8 -1.60669e-5, distorted cantilever 1.29920e-3).
* **asn** is the fallback above.
* **asnl2** is the proposal.
* **mitc** uses MITC9-style shear spaces (linear in both shear directions, quadratic in the third)
  instead of P8; it was slightly worse.
* **red** is plain 2x2x2.

| deck (% of reference) | as (now) | asn | **asnl2** | red (2x2x2) | Simcenter | ComLab |
|---|---:|---:|---:|---:|---:|---:|
| hemisphere 4x4 | 9.0 | 64.9 | **64.5** | 77.6 | 51.7 | 74.5 |
| hemisphere 8x8 | 55.1 | 97.6 | **97.9** | 99.1 | 96.6 | 98.7 |
| hemisphere 12x12 | 85.2 | 99.2 | **99.3** | 99.6 | 99.0 | 99.5 |
| hemisphere 16x16 | 94.5 | 99.5 | **99.6** | 99.7 | 99.5 | 99.6 |
| cylinder 4x4 | 61.4 | 67.9 | **79.8** | 86.8 | 64.8 | 86.4 |
| cylinder 8x8 | 88.0 | 94.0 | **96.4** | 97.1 | 93.0 | 96.9 |
| cylinder 12x12 | 96.6 | 98.5 | **99.1** | 99.1 | 98.3 | 99.1 |
| cylinder 16x16 | 99.1 | 99.8 | **100.0** | 100.0 | 99.8 | 100.0 |
| Scordelis-Lo roof | 99.0 | 99.6 | **99.7** | 99.7 | 99.6 | 99.7 |
| twisted beam, in-plane | 99.3 | 99.3 | **99.3** | 99.3 | 99.3 | 99.3 |
| twisted beam, out-of-plane | 99.9 | 99.9 | **99.9** | 99.9 | 99.8 | 99.9 |
| plate, simply supported | 100.8 | 100.8 | **100.8** | 100.8 | 100.8 | 100.8 |
| plate, clamped | 97.5 | 97.5 | **97.5** | 97.5 | 97.5 | 97.5 |
| Lame cylinder, nu = 0.3 | 100.0 | 100.0 | **99.9** | 100.0 | 100.0 | 100.0 |
| Lame cylinder, nu = 0.4999 | 100.0 | 100.0 | **99.9** | n/a | 93.6 | 99.8 |
| curved patch, max u error / 1e-3 | 7e-15 | 1.5e-14 | **8.9e-15** | 9.0e-3 | 9.5e-8 | 9.0e-3 |
| aspect 1e3 rotated, % exact | 100.0 | 100.0 | **100.0** | - | 100.0 | ~0.07 (1500x too flexible) |

Element-level checks for asnl2 (`tools/hex20_prototype/check2.py`, `rank.py`):
* Curved, off-centre distorted element and a hemisphere-like shell segment (R/t = 250, aspect ratio about
  50): energy of the six constant-strain fields exact to 3e-15 (1.7e-10 at the extreme aspect ratio).
  Nodal forces of constant stress equal the compatible element's to 8e-15.
* Rigid-body forces at round-off (<= 9e-16 of max |K|).
* The lowest non-rigid eigenvalue is >= 0.02 of the full element's at nu = 0.3, and >= 2.6e-3 at
  nu = 0.4999 (volumetric relief, as for the current element). Shapes: cube, curved-distorted, 20:1:1
  needle, 10:10:1 slab, and real 8x8 hemisphere elements. No spurious modes.
* 24 relabelings and an arbitrary rotation: K unchanged to 2.5e-15.

asnl2 meets the doc's targets: hemisphere and cylinder at least 95% at 8x8 and about 99% at 12x12, with
the patch test, invariance, rank, flat-plate and incompressible accuracy retained. It matches or beats
Simcenter on every row except the nu = 0.3 Lame cylinder (99.9% vs 100.0%). It is within 0.2 points of
ComLab from 12x12 on, and slightly behind at 8x8 (97.9% vs 98.7%, 96.4% vs 96.9%), without ComLab's
patch-test and invariance defects.

## 5. Acceptance tests (both solvers)

These are the MYSTRAN doc's criteria, made concrete:
1. Single-element spectra: exactly 6 rigid modes, with a scale-aware test. The lowest deformation
   eigenvalue must be at least 1e-3 of the full element's. Shapes: cube, randomly distorted with curved
   off-centre midside nodes, needle, slab, trapezoid, a thin shell segment. nu = 0.3 and 0.4999.
2. Six independent constant-strain fields exact at every Gauss point, the centroid and the corners. The
   patch deck `patch_20_default` exact.
3. Nodal forces of constant stress equal to the compatible element's (the two-sided condition).
4. Rotation and 24 relabelings: K unchanged to 1e-12 relative.
5. Free thermal expansion, and linear temperature on affine patches (`gradient_20_*`): exact, with zero
   stress at all recovery points. MAT9 cases (`ortho_20_*`) as in MYSTRAN.
6. Benchmarks: hemisphere and cylinder at least 97% / 96% at 8x8 and 99% at 12x12. Flat plates and
   twisted beam within 0.1 point of the current element. Lame nu = 0.4999 at least 99.5%. Aspect 1e3 exact.
7. Cost: element stiffness time against the full 3x3x3 element. Expected under 2x, since the dominant
   cost, the 60x60x27 energy sum, is the same.

## 6. Implementation plan

**First in vibestran.** The MYSTRAN tree is read-only for me in this work. vibestran also has no
CHEXA20 to replace, and the Q1/E21 CHEXA8 port gives the template: `SolidHexa8` and its tests, and
`tests/integration/test_hexa8_reference.cpp`. The Fortran port to MYSTRAN's `SOLID_ELEMENTS.f03`
(`HEXA`) is a direct transcription. Its existing B-bar and natural-frame code covers steps 2-3 and 6, and
the mean correction and normal-strain spaces are small additions. I can do it once you open the MYSTRAN
tree, or you can, using the prototype as the reference.

vibestran steps:
1. `SolidHexa20` (`include/elements/solid_hexa20.hpp`, `src/elements/solid_hexa20.cpp`), with
   stiffness, thermal load, recovery at the centroid and 8 corners (Nastran CHEXA20 output points), and
   consistent mass.
   * PSOLID semantics as in MYSTRAN: ISOP blank/REDUCED gives this element; ISOP FULL gives the
     displacement element at 3x3x3.
   * IN = TWO is promoted to THREE with a one-time warning, because 2x2x2 has 6 spurious modes.
   * At a corner with a non-positive Jacobian, fall back to the centroid frame for T at that point, or
     extrapolate as MYSTRAN does. To be decided when the patch with element 3 is run.
2. Wire it into the factory, SOL 101/103/153 assembly and output. Add 8-node-face PLOAD4 and GRAV
   consistent loads for CHEXA20; `linear_static_pressure.cpp` handles CTETRA10 faces but not 8-node quad
   faces.
3. Unit tests for Section 5 items 1-5, ported from `test_solid_hexa8.cpp`. An integration test runs the
   comparison decks with exact references (patch, Lame, aspect, convergence thresholds). Data is copied
   into `tests/integration/data/hexa20_reference` as for the HEXA8.
4. MYSTRAN validation suite run. CHEXA20 is currently one cause of the `unsupported_input` decks, and
   some of these will start running. Update the docs, then commit.

## 7. Alternatives considered

* **asn (fixed centroid frame).** Simpler, and meets the hemisphere target, but the cylinder 8x8 is at
  94.0% (target 95%). Kept as the fallback if the local-frame version raises any robustness issue.
* **MITC9-style shear spaces.** Slightly stiffer than P8 on every curved case (hemisphere 8x8 96.1% vs
  97.6% with the fixed frame). Not pursued.
* **Plain or stabilized 2x2x2 (Sze 1994).** 2x2x2 is the accuracy ceiling here (99.1% / 97.1%), but it
  has 6 spurious modes and fails the curved patch test. Stabilization would need derived assumed-stress
  terms tuned not to restore the locking. The projection already reaches within 1.2 points of 2x2x2 at
  8x8 without either problem.
* **EAS or hybrid stress for the normal strains.** Equivalent in principle to the projection, through
  the Hu-Washizu equivalence, but with condensed internal parameters, thermal forcing in condensation and
  recovery, and conditioning to manage at aspect ratio 50. Not needed given the results above.

## 8. Open questions for you

1. Adopt asnl2 (recommended) or the simpler asn?
2. Implement in vibestran first, then port to MYSTRAN, as proposed? Who does the MYSTRAN port?
3. Scope of the first vibestran commit: is SOL 101 + SOL 103 + thermal + PLOAD4/GRAV, as listed, right?
   Differential stiffness is out of scope, since vibestran has no buckling.
