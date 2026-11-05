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

## Rods and scalar points

| Feature | How absence was checked | Suggested deck |
|---|---|---|
| CONROD | No deck contains the card | Any CROD deck with each CROD and PROD pair rewritten as one CONROD: identical results. 35 suite decks were checked this way, but that is not a comparison against another solver |
| PBARL CHAN, T, I1, BOX, ROD and TUBE shear factors | The only PBARL deck with a transverse load uses BAR | Cantilever of each section under a tip force, compared with MSC Nastran's numerical conversion |
| PBARL properties in SOL 103 and with a CD frame or offsets | Both PBARL decks are SOL 101 without offsets | Cantilever bending modes of a CHAN against the closed-form frequencies |
| CELAS stress in OP2 | The suite reads F06 only | Read OES1X types 11 and 12 with pyNastran |
| CMASS3, CMASS4 | No deck contains either card | Spring-mass chain of scalar points: lambda_j = (k/m)(2 - 2 cos((2j-1) pi / (2n+1))) |
| Scalar points in SOL 103 and SOL 153 | The six SPOINT and CELAS3/4 decks are all SOL 101 | Scalar oscillator on a rod tip (two-DOF eigenproblem); scalar ambient point on a CHBDY (series resistance) |
| SLOAD on a grid with a CD frame | SLOAD appears in 3 decks, all on scalar points | Rod along a rotated CD X axis, loaded on its T1 |
| Implicit scalar points | Every deck with a blank or zero CELAS/CMASS component on an undefined point also has SPOINT | The same chain with the SPOINT card removed must give identical results |
| CROD torsion with C, and rod margins on torsion | No PROD sets C with nonzero torque on a suite-checked record | Skew rod with a moment along its axis: T L / (G J), tau = C T / J, margin SS/tau - 1 |
| AUTOSPC of a skew rod on a CD grid | No deck has a singular rod block on a CD grid without an MPC | Rod along a rotated CD X axis: displacement F L / (E A) along T1 and exactly zero on T2 and T3 |

## MPC sets and frames

| Feature | How absence was checked | Suggested deck |
|---|---|---|
| MPCADD naming its own SID or another MPCADD, and over MPC cards of its SID | The only 2 MPCADD decks name plain MPC sets of other SIDs | A rod chain with the same ties given each way, as in `MpcAdd.*` |
| MPC on a cylindrical CD grid, or on rotation components of a CD grid | CD MPC appears only in `SB/SS-RODLOAD-MPC-FOR-SING`, on translations in a rectangular frame | `CdMpc.CylindricalFrameFollowsTheGrid` and `CdMpc.RotationsTurnWithTheSameAxes` |
| SOL 103 on a grid with a CD frame | No SOL 103 deck has a GRID with nonzero CD | A radial rod with a mass in a cylindrical CD frame: the shape is all T1, against the basic split between X and Y, with NORM=MASS and NORM=MAX, as in `CdModeShape.*` |
| CELAS and CMASS components on a CD grid | The CELAS grids of `vic/6/NAS S30 twisted plate mit4+.bdf`, the only deck with both, have no CD frame | A grounded spring on T1 of a skew frame (the load's projection, with the rest reacted on T2), and a mass on a skew axis between unequal rods, as in `CdScalar.*` |

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
| CBEAM / PBEAM (any use) | No `CBEAM` or `PBEAM` card | Cantilever with tip loads in both planes and a nonzero I12; compare against the closed-form unsymmetric bending solution, with `FORCE = ALL` so the element end forces (written in the BAR force layout) are checked too |

## Mass and modal analysis

| Feature | Check | Suggested deck |
|---|---|---|
| `PARAM,COUPMASS` (lumped mass; vibestran defaults to consistent mass) | No deck sets COUPMASS | Copy the `V30 beam released dof *` decks with `PARAM,COUPMASS,-1`: the pin-flag and MPC-hinge structures must then give identical frequency pairs (vibestran reproduces MYSTRAN's `released dof y-n` frequencies exactly this way) |
| Modal analysis with massless DOFs (lumped mass, rotations without inertia) | No deck exercises it | Same decks; checks that no spurious modes appear and none are missed |
| CONM2 with CID = -1 (X is the basic location of the CG) | 19 decks use CONM2, none with CID = -1 | Same offset mass given once with CID = 0 and an offset, once with CID = -1 and the absolute CG location; results must be identical |
| Modal analysis requesting all finite modes (ND >= massed DOFs, e.g. a one-DOF spring-mass) | Not checked by any deck | Grounded CELAS2 + CONM2: lambda = k/m |
| CTETRA10 consistent mass in modal or inertial-load analysis | 12 decks use CTETRA10, none in SOL 103 (this hid a bug: the mass rejected Nastran's vertex orientation, fixed with `CTetra10.MassIsTheSameForEitherVertexOrientation`) | Free-free or cantilevered tet10 block in SOL 103, compared with a refined CHEXA/CTETRA4 reference, plus a GRAV case checking total reaction = rho*V*g |
| Generalized mass and stiffness under NORM=MAX | 12 decks set EIGRL MAX, but `f06_query.py` reads only EIGENVALUE and CYCLES from the eigenvalue table | Two equal masses and springs from ground: m (1 + 1 / g^2) for both modes, g the golden ratio, as in `GeneralizedMass.*`, with the comparator extended to those columns |

## Solids

| Feature | Check | Suggested deck |
|---|---|---|
| PLOAD4 on a CHEXA20 (8-node quadratic face) | 38 decks use 20-node CHEXA, none with PLOAD4 | A curved CHEXA20 shell segment (cylinder panel) under uniform and corner-varying PLOAD4, against the equivalent consistent FORCE set and a refined reference; plus a block under top-face pressure with sliding supports (exact uniform stress, as in `Hexa20Reference.Pload4OnAQuadraticFaceGivesUniformStress`) |
| Omitted midside grids of a structural CHEXA20 / CTETRA10 | No CHEXA20 or CTETRA10 with a blank midside slot in the 64 of 79 CHEXA/CTETRA decks pyNastran reads (the other 15 it cannot parse) | A CHEXA20 mesh joined to a CHEXA8 mesh (the shared faces' midside grids omitted) and a CTETRA10/CTETRA4 transition, in SOL 101 under PLOAD4 on a face with an omitted grid and in SOL 103, against the same mesh with the grids at edge midpoints tied by MPCs (as in `OmittedMidside.*`) |

## Shells (CQUAD4 / CTRIA3)

| Feature | Check | Suggested deck |
|---|---|---|
| PCOMP `LAM` = `MEM`, `BEND`, `SMEAR`, `SMCORE` | 67 PCOMP cards: LAM blank (48) or `SYM` (19) | One laminate with each option under membrane and bending loads; MEM must have no bending stiffness from the plies (and BEND none in membrane), SMEAR no B coupling, SMCORE the honeycomb-core equivalent of the listed face plies |
| SNORM with CID != 0 | Every SNORM card (about 21,700) has CID 0 | A cylindrical patch whose SNORMs are given as (1, 0, 0) in a CORD2C about the cylinder axis; results must equal the basic-system deck |
| `PARAM,SNORM` > 0 (shared-director fold angle) | Only `PARAM,SNORM,0.0` (2 decks) | A folded plate (box section or T-junction) with folds of 10 and 30 degrees, run with SNORM = 20 and 5: the 30-degree fold must keep separate directors in both, the 10-degree one only with 5 |
| Free thermal expansion of a doubly curved shell | Only faceted cylinders (`thermal strain free rectangle/diamond cylinder`) | Faceted hemisphere under uniform dT with statically determinate supports: zero stress and radial growth alpha dT R (vibestran benchmark `ShellBenchmark.ThermalFacetedRing*` does the ring) |
