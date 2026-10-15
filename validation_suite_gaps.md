# Validation Suite Coverage Gaps

Features that vibestran implements (or is implementing) but that no deck in
`MYSTRAN_Validation/decks` exercises. Candidates for new validation decks.

Each entry records how absence was checked (a search over every file under
`MYSTRAN_Validation/decks`) and what a useful deck should test. Prefer decks
with a closed-form or independently verified answer over decks whose only
reference is a MYSTRAN run.

## Case control

| Feature | Check | Suggested deck |
|---|---|---|
| Abbreviated commands other than `DISP`, `STRE`, `STRA`, `SPCF`, `LABE` (e.g. `METH`, `VECT`, `EIGE`, `GPST`, `ANAL`, `THER`, 5+ character forms such as `DISPL`) | No case-control line matching `^<abbr>\s*[(=]` | One modal and one static deck that use only abbreviated commands; results must equal the long-form deck |
| `ELSTRESS` alias for `STRESS` | Not found | Same deck as above |

## Bulk data syntax

| Feature | Check | Suggested deck |
|---|---|---|
| Free-field continuation lines that start with `,` (blank continuation label), including short segments | No deck line starts with `,` followed by data | Any free-field deck whose PBAR/CORD2R continuations use `,` and stop before the 8th field; results must equal the fixed-field deck |

## Grid defaults

| Feature | Check | Suggested deck |
|---|---|---|
| BAROR OFFT field (`BGG` default for CBARs with blank OFFT) | 14 decks use BAROR, none set its OFFT | BAROR with `BGG` and a GA displacement system that differs from basic, so the orientation-vector frame matters |

## CBAR / PBAR

| Feature | Check | Suggested deck |
|---|---|---|
| CBAR `OFFT` field (any of `GGG`, `BGG`, ...) | No CBAR continuation contains an OFFT string | Offset bar with `BGG` and a GA displacement coordinate system that differs from basic, so the orientation-vector frame matters |
| PBAR with both a nonzero I12 and K1/K2 shear factors | No deck (2 decks use I12, 13 use K1/K2, none both) | Cantilever with a rotated rectangular section described by I1/I2/I12 and K1 = K2; must equal the same bar described in principal axes (vibestran unit test `CBar.ProductOfInertiaMatchesRotatedPrincipalSection` does this; MYSTRAN's BAR1 drops shear flexibility when I12 != 0, so it cannot serve as the reference) |
| CBAR pin flags combined with end offsets | No deck (18 use pin flags, 10 use offsets, none both) | Offset bar with a moment release at the offset end; the released moment must be zero at the offset point, not at the grid |
| PLOAD1 (any type) | No `PLOAD1` card | Simply supported and cantilever bars with uniform and linearly varying `FYE`/`FZE` and basic-direction `FY` loads; closed-form end moments and deflections |
| PLOAD1 basic-direction types on inclined bars, projected scales (`LEPR`/`FRPR`), concentrated loads and moments (X2 blank) | No `PLOAD1` card | Inclined cantilever under basic `FY` with `LE` and `LEPR` (total load q L vs q L sin(angle)); point load at mid-span with blank X2 (fixed-end forces P/2, PL/8) |
| PLOAD1 combined with pin flags or offsets | No `PLOAD1` card | Propped cantilever (pin flag at one end) under uniform load: reaction 3wL/8 and fixed-end moment wL^2/8 |
| PLOAD1 under a `LOAD` combination with a scale factor | No `PLOAD1` card | Same as above with `LOAD` scale != 1 |

## CBEAM / PBEAM

| Feature | Check | Suggested deck |
|---|---|---|
| CBEAM / PBEAM (any use) | No `CBEAM` or `PBEAM` card | Cantilever with tip loads in both planes and a nonzero I12; compare against the closed-form unsymmetric bending solution |

## Mass and modal analysis

| Feature | Check | Suggested deck |
|---|---|---|
| `PARAM,COUPMASS` (lumped mass; vibestran defaults to consistent mass) | No deck sets COUPMASS | Copy the `V30 beam released dof *` decks with `PARAM,COUPMASS,-1`: the pin-flag and MPC-hinge structures must then give identical frequency pairs (vibestran reproduces MYSTRAN's `released dof y-n` frequencies exactly this way) |
| Modal analysis with massless DOFs (lumped mass, rotations without inertia) | No deck exercises it | Same decks; checks that no spurious modes appear and none are missed |
| CONM2 with CID = -1 (X is the basic location of the CG) | 19 decks use CONM2, none with CID = -1 | Same offset mass given once with CID = 0 and an offset, once with CID = -1 and the absolute CG location; results must be identical |
| Modal analysis requesting all finite modes (ND >= massed DOFs, e.g. a one-DOF spring-mass) | Not checked by any deck | Grounded CELAS2 + CONM2: lambda = k/m |
| CTETRA10 consistent mass in modal or inertial-load analysis | 12 decks use CTETRA10, none in SOL 103 | Free-free or cantilevered tet10 block in SOL 103, compared with a refined CHEXA/CTETRA4 reference, plus a GRAV case checking total reaction = rho*V*g |
