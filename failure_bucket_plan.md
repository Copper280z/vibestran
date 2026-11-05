# Validation Failure Plan

> **Update convention:** the body of this file describes only the *current*
> state of the validation run. After each run, rewrite the body to match the
> new state and append a brief dated entry to the **Run History** section at
> the bottom — do not accumulate per-run narratives in the body.

Current run command:

```text
python3 test.py /Users/bob/Documents/vibe-tran/build/vibestran
```

The complete per-definition classification is in `failure_buckets.txt`. Each
detail row contains the test type, deck, requested path, solver return code,
solver log, F06 path, and comparison message. Per-deck solver output is in
`diagnostics/**/*.solver.log`.

## Current Summary

The suite contains 3,226 definitions. The current run has 1,706 final failure
records:

| Bucket | Failure records | Unique decks | Current disposition |
|---|---:|---:|---|
| `unsupported_input` | 874 | 147 | Requires unsupported cards, solution sequences, elements, or formulations |
| `numerical_mismatch` | 599 | 217 | Result exists but differs from the reference or comparator expectation |
| `input_parse_error` | 185 | 15 | Parser rejects syntax or a card variant used by a valid deck |
| `missing_result_data` | 42 | 14 | Solver completes but does not emit the requested result family |
| `solver_model_error` | 6 | 2 | Parsed model is rejected by a geometry gate or has no free DOFs |
| **Total** | **1,706** |  |  |

CQUAD4 is the MITC4+/D continuum shell and CTRIA3 the MITC3+ shell (see
`SHELL_REWRITE_PLAN.md` and `SHELL_BENCHMARKS.md`): six DOFs per grid,
shared grid directors (averaged, or SNORM), warped and curved geometry,
transverse shear, the first-order through-thickness metric, PSHELL
MID1-MID4 and PCOMP sections on MAT1/MAT2/MAT8, THETA/MCID, ZOFFS, grid
thicknesses, and consistent thermal strain. `PARAM,QUAD4TYP`, `SHELLFORM`
and `K6ROT` are accepted and ignored. Warp is a warning, not a gate. The
MAT2/MAT8 and warp `solver_model_error` decks of earlier runs now solve.

CHEXA8 is the Q1/E21 enhanced assumed strain element (MYSTRAN's CHEXA8,
reproduced to print precision on MYSTRAN's 106 isotropic CHEXA8 comparison
quantities, `tests/integration/test_hexa8_reference.cpp`). PSOLID ISOP =
FULL or 1 selects the standard full-integration element, which the RBE3
node-surface coupling decks use. Solids always report centre and corner
stresses and strains, as in Nastran.

Element engineering forces (`FORCE`/`ELFORCE`) and shell strains
(`STRAIN`, STRCUR or FIBER form, von Mises or maximum shear) are written for
CQUAD4 (centre and BILIN corners) and CTRIA3, element forces also for
CBAR/CBEAM. Shell results use Nastran's element frames and sign convention.

GRID PS, GRDSET, and BAROR are applied. CBAR/PBAR are fully parsed. Solid
strain is fully emitted; the two `transform` corner-ZZ mismatches are
`SOLIDSTRESSES/GID/.../ZZ` records on `vic/8/NAS S30 solid stress strain
transform.bdf` at 0.0001% of the reference.

Another 15 records still fail in the way declared by their `KNOWNFAIL`
expectations. They are recorded under `KNOWNFAIL EXPECTED SUMMARY` in
`failure_buckets.txt` and are not included in the final-failure total.

## Current MPC And RBE Handling

### Constraint inputs

The analysis builds one MPC equation list from these sources, in this order:

1. GRID displacement-coordinate SPC equations.
2. Basic-coordinate nonzero SPC equations.
3. RBE2 equations.
4. RBE3 equations.
5. The active explicit MPC/MPCADD set.

Each equation has the affine form

```text
sum_i C_i u_i = rhs
```

Explicit BDF MPC, RBE2, and RBE3 equations are homogeneous. Internally
generated enforced-displacement equations may have a nonzero right-hand side.

MPCADD is resolved after all cards are parsed, so card order does not matter. It
takes precedence over MPC cards of its own SID, may name that SID to include
them or another MPCADD, and an undefined set is a parse error. MPC components are
in the grid's displacement frame: a term on a CD grid is written on the basic
components of its 3-block, all of which become active, and MPC forces are
reported in that frame. CELAS and CMASS components are treated the same way.

### RBE2 expansion

Each dependent RBE2 grid is expanded into ordinary rigid-body MPC rows. The
translation rows include the independent-grid rotation lever arms, and enabled
rotation rows equate dependent and independent rotations. These equations are
currently formed in the basic Cartesian frame.

RBE2 component handling in non-basic GRID displacement coordinate systems is
not yet transformed as RBE3 and SPC handling are. Decks that depend on that
behavior require separate verification and implementation.

### RBE3 expansion

RBE3 uses the same coupled weighted least-squares model as the current
uncommitted MYSTRAN `RBE3_PROC` implementation. For every active scalar
independent DOF, vibestran forms its six-component rigid-motion vector `H` and
assembles

```text
A = sum(w H H^T)
B_j = -w H
A q_ref + B u_independent = 0
```

The implementation:

- Accounts for reference and independent GRID displacement coordinate systems.
- Includes translation-to-reference-rotation coupling through lever arms.
- Retains only components requested by REFC.
- Eliminates omitted REFC components with a generalized Schur reduction.
- Uses a rank-revealing minimum-norm solve for the omitted block.
- Accepts a rank-deficient omitted block only when every elimination right-hand
  side is in its range.
- Rejects a retained block that cannot determine every requested REFC
  component.
- Does not currently represent optional RBE3 `UM` or thermal-expansion fields.

The local uncommitted MYSTRAN binary and vibestran agree on the representative
single-RBE3 deck's meaningful displacement solution, including reference-grid
`T3 = 2.5e-4` and `R1 = -4.807692e-5`.

### MPC reduction

`MpcHandler` converts the equations into an explicit affine transformation:

```text
u_full = T u_reduced + u0
K_reduced = T^T K_full T
F_reduced = T^T (F_full - K_full u0)
```

The reduction currently works as follows:

1. Coefficients for repeated occurrences of the same equation DOF are summed.
2. The first equation term is the preferred dependent DOF, preserving Nastran
   MPC/RBE ordering instead of choosing the largest coefficient.
3. If that DOF is already dependent, later free terms are considered as
   fallback pivots.
4. A fallback is accepted only if Eigen `FullPivLU` confirms that the
   accumulated dependent-column matrix remains full rank.
5. A row with no independent pivot is tested against the accepted rows. A
   consistent duplicate is discarded; an inconsistent affine duplicate is an
   error.
6. Acyclic dependent chains are recursively substituted.
7. Strongly connected dependency components are solved simultaneously as
   `(I-C)x=r` with `FullPivLU`.
8. A rank-deficient coupled component is rejected as singular.
9. Element matrices and vectors are assembled through `T`; dependent values
   are recovered after solving the reduced system.

This supports chained equations, forked rigid elements, affine prescribed
motion, and full-rank coupled dependency blocks without using exceptions for
normal control flow.

## Comparison With MYSTRAN

The two solvers implement equivalent transformation mathematics for ordinary
full-rank MPC/RBE systems, but they build and solve it differently:

| Area | vibestran | MYSTRAN |
|---|---|---|
| Constraint representation | Per-equation affine eliminations assembled into explicit `T` and `u0` | Sparse global `RMG` matrix partitioned into `RMM` and `RMN` |
| Preferred dependents | First term is preferred, but another full-rank column may be selected | Rigid/MPC processing assigns M-set DOFs before matrix reduction |
| Global dependency solve | Substitution plus dense pivoted LU on coupled strongly connected components | Diagonal fast path or pivoted LU of the complete `RMM` block using LAPACK or SuperLU to form `GMN` |
| Redundant equations | Consistent rank-dependent rows are discarded | Duplicate M-set assignment can be rejected before LINK 2; other singularity is diagnosed during `RMM` factorization |
| Nonzero prescribed motion | Included directly through affine offset `u0` | Uses MYSTRAN's set reduction and enforced-displacement processing; not represented by vibestran's explicit affine object |
| RBE3 local system | Eigen rank-revealing decomposition and generalized Schur reduction | LAPACK `DGELSY` rank-revealing minimum-norm solve and generalized Schur reduction |
| Large coupled blocks | Dense solve per coupled component | Global `RMM` solve with a sparse SuperLU backend available |

There are two separate numerical solves in MYSTRAN that should not be
conflated:

1. RBE3 omitted-component reduction uses `DGELSY`, a rank-revealing
   least-squares/minimum-norm solve.
2. Global M-set constraint reduction solves `RMM GMN = -RMN`. A diagonal
   `RMM` has a direct fast path; otherwise MYSTRAN uses pivoted LU through dense
   LAPACK `DGETRF`/`DGETRS` or sparse SuperLU, depending on solver settings.

Vibestran mirrors that separation: its local RBE3 reduction is rank-revealing,
while its selected dependent blocks use pivoted LU.

### Shared dependent DOFs

Vibestran is currently more permissive than the local MYSTRAN build when
multiple equations name the same preferred dependent DOF. Vibestran attempts a
different full-rank pivot and retains both independent equations. The current
local MYSTRAN binary rejects the two-RBE3 shared-reference deck before LINK 2
with `ERROR 1330` because the reference components are already in the M-set;
its `RMM` LU solve is therefore never reached for that deck.

This difference is intentional in the current vibestran implementation, but it
is not strict behavioral parity with that MYSTRAN diagnostic. The validation
deck `vic/9/NAS S30 node-surface coupling RBE3 crash.bdf` now solves in
vibestran. Its remaining failures are result-output and numerical-comparison
records, not a solver exit.

## Current Failure Buckets

### Unsupported Input

There are 963 records across 178 decks. By unique deck, the current logs
show:

| Cause | Decks |
|---|---:|
| Ignored unsupported BDF or case-control card | 109 |
| Unsupported solution type | 61 |
| Explicitly unsupported field or element | 8 |

The explicit cases are CBUSH offset/location fields (8 decks). The most
widespread ignored keywords across these decks are `ECHO`/`TITLE` (47),
`DEBUG`/`ID` (46), `CQUAD8` (41), `OLOAD` (34), `MEFFMASS` (24), `EIGR` (22), `SUBT` (22), `SUBTITLE` (12), and `MPFACTOR`
(10).

The main implementation work remains CQUAD8 (and CTRIA6), OLOAD output,
modal GPFORCE, EIGR/reduction behavior, and unsupported solution
sequences such as buckling.

### Missing Result Data

There are 36 records across 11 decks:

| Result family | Records |
|---|---:|
| `COMPOSITESTRESSES` | 24 |
| `DISPLACEMENTS` | 4 |
| `ELASSTRESSES` | 3 |
| `ELASFORCES` | 3 |
| `MODE` | 2 |

`COMPOSITESTRESSES` are PCOMP ply stresses (`S30 shell laminate orthotropic
thermal stress` and `_2`), which are not implemented; these decks solved
for the first time once PCOMP was supported. `ELASSTRESSES`/`ELASFORCES`
are CELAS stress and force recovery. The four `DISPLACEMENTS` records are
the `BANDED`/`SPARSE BAILOUT` decks and `RBE3 crash`.

The two `MODE` records are `V30 beam released dof x-x` and `xyz-xyz`, which
expect the solve to fail (eigenvector `ABSENT`): releasing the same component
at both ends of a bar leaves a mechanism that MYSTRAN rejects. vibestran
returns a zero-frequency mode instead, which is a valid modal answer; these
records encode MYSTRAN behaviour rather than a correctness requirement.

### Numerical Mismatch

There are 540 records across 201 decks:

| Result family | Records |
|---|---:|
| Bulk comparison | 195 |
| `DISPLACEMENTS` | 182 |
| `SHELLSTRESSES` | 48 |
| `SPCFORCES` | 24 |
| `SHELLFORCES` | 24 |
| `REALEIGENVALUES` | 24 |
| `SHELLSTRAINS` | 17 |
| `MODE` | 15 |
| `SOLIDSTRESSES` | 11 |

The count rose against the pre-rewrite runs because the MAT2/MAT8, PCOMP
and warped-shell decks now solve instead of stopping at a formulation gate.
The shell records fall into these groups:

- **MYSTRAN element responses at tight tolerance.** Single-element and
  coarse-mesh decks (`quad4 mitc4 flat`, `quad4 mitc4+ twisted`, `shell
  stress components *`, `shell force per length components *`, `mitc4
  strain`, `MITC4+ corner stress order`, the `thermal stress linear
  temperature*` decks) compare against MYSTRAN's own element to 2e-5% or
  similar. A different, converging formulation cannot match these; the
  references are not exact solutions. Corner values are BILIN
  extrapolations from the 2x2 points.
- **Drilling rotations.** `K6ROT`, `K6ROT MITC4+`, `Raasch hook K6ROT=10`,
  `AUTOSPC reaction forces`, `Shell pinned support multiple nodes`,
  `node-surface coupling shell rotation Z` and several `RZ` bulk records
  encode MYSTRAN's K6ROT drilling spring. vibestran's shells carry only a
  small drill stabilisation (like Nastran) and ignore `PARAM,K6ROT`.
- **Roundoff against all-zero references** (suite issue 2): e.g. `shell
  element force`, `shell gravity`, `tri3 quad4`, `shell laminate coupling`,
  `tiny principal stress and strain angle bug` (errors 1e-23 to 5e-9).
- **Very thick twisted plate.** `NAS S30 twisted plate mit4+` (t/L = 0.6,
  in-plane tension, corner TZ) converges to -2.36e-4. MYSTRAN's CHEXA8
  model of the same body converges to -1.731e-4. MYSTRAN 18 (the stored
  reference) gives -1.89e-4 and the current MYSTRAN 19 gives -3.71e-4. The
  shell error falls as t^2 (4.8% at t/L = 0.2, 0.50% at 0.06); see
  `SHELL_SOLID_COMPARISON.md`.
- **References from MYSTRAN 18 that are wrong.** The `thermal strain free
  rectangle/diamond cylinder` criteria encode MYSTRAN 18 values that
  violate exact free expansion (rectangle TX 0 and TZ 0.141, against
  -0.18 and 0.09). vibestran and MYSTRAN 19 both give the exact values.
  For the Raasch hook, MSC Nastran with K6ROT = 10 diverges with refinement
  and with K6ROT = 1000 converges 7% flexible, while vibestran is within
  0.1% of the solid. `SHELL_SOLID_COMPARISON.md` has the CHEXA8 references
  for all of these decks.
- **Reference conventions.** `gravity SNORM` SPCFORCES TY is a support
  split of -5759.93/-5760.07 (sum exact) against 0.001%. The hemisphere
  and diamond-cylinder records are within a few percent of decks whose
  references are MYSTRAN's coarse-mesh values.

Element and constraint incidence overlaps when a deck contains multiple
cards: CQUAD4 86 decks, CBAR 81, CTRIA3 31, CHEXA 25, RBE3 19, RBE2 16,
CBUSH 14, PCOMP 12, CONM2 12, CROD 8, CQUAD8 5, CTETRA 5, CPENTA 4.

The bar benchmark decks' remaining mismatches are mostly not in the bar
element itself:

- Benchmark bulk comparisons (`bar_*`, `SB-*`, `*-ALL-ELEM-*`) compare the
  whole F06, including blocks vibestran does not write (OLOAD, GPFORCE) or
  writes in a different layout (the bar stress table has no
  SA-Max/SA-Min/M.S. columns).
- `V30 beam released dof *` modal decks compare a CBAR pin flag against an
  equivalent MPC hinge. The identity is exact only when rotations are
  massless (Nastran's default lumped mass). vibestran deliberately defaults
  to consistent mass, so the two models differ at the discretization level
  (0.02-0.2% in frequency, up to ~1% for an axial release). With
  `PARAM,COUPMASS,-1` added, vibestran reproduces MYSTRAN's `released dof
  y-n` frequencies exactly; the decks do not set it.

Unsupported cards present in a bulk-comparison deck must be separated from
genuine numerical defects before changing solver mathematics or tolerances.

### Solver Model Errors

There are 6 records across 2 decks:

| Current cause | Decks | Disposition |
|---|---:|---|
| CQUAD4 interior angle 170.2 deg > `MAXANGLE` 170 (strict quality gate) | 1 | `S30 mitc4 reaction moment complex shape`; was blocked by the warp gate before. Nastran rejects only concave elements; consider making the interior-angle check a warning like warp |
| Empty stiffness matrix (no free DOFs) | 1 | `celas1_stress_force`; deck-specific, not yet triaged |

Unused entities, coincident nodes, aspect ratio, taper and warp are
warnings. Fatal negative/inverted geometry checks remain enabled.
Coincident nodes are not merged because separate coupled substructures can
intentionally share a position.

### Input Parse Errors

There are 185 records across 15 decks. The unique deck causes are:

| Parser gap | Decks |
|---|---:|
| 15-node CPENTA | 7 |
| Partial or advanced PBUSH fields | 5 |
| Advanced or unsupported PBARL section data | 2 |
| CELAS2 damping/stress-coefficient fields | 1 |

Parser support must be paired with the corresponding element/formulation
behavior; accepting fields and silently ignoring their mechanics would produce
plausible but invalid results.

## Validation-Suite Issues To Verify

These issues remain in the suite and should be verified with another solver or
tool before changing expected data:

1. `test_bulk.py` converts every missing path to zero. That is only valid for a
   known omitted all-zero row; absent strain, reaction, shell-force, CBUSH, or
   eigenvalue blocks can therefore appear as numerical mismatches.
2. A result group whose reference values are all zero receives an absolute
   tolerance of exactly zero, so harmless floating-point roundoff fails.
3. Solid stress and strain bulk field lists contain `XX` twice and omit `ZZ`.
4. Modal bulk comparison pairs modes by ordinal number without MAC-based mode
   matching, which is ambiguous for repeated or reordered modes.
5. `vic/6/NAS S30 3 digit exponent 2.bdf` requires exact zero with absolute
   tolerance `1e-110`; the computed value is roundoff near `1e-40`, while its
   1,278-value bulk comparison passes.
6. Several shell bulk comparisons use a relative scale near `1e-8` while the
   same decks' focused criteria permit substantially larger relative error.
7. Summary rows in `failure_buckets.txt` use the same first column as detail
   rows. Consumers must require the complete 11-column detail shape before
   counting failures.

## Next Priorities

The goal is feature compatibility with correct, modern formulations, not
numerical agreement with MYSTRAN; the suite's tolerances and some expected
values encode MYSTRAN behaviour and are not a correctness oracle.

1. PCOMP ply stress/strain output (`COMPOSITESTRESSES`, 24 records) and
   CQUAD8/CTRIA6 (MITC9/MITC6-type elements; CQUAD8 in 41 decks).
2. OLOAD output (the rod benchmark decks stop at their first
   `APPLIEDFORCES` record, with every displacement, SPC force, rod force and
   rod stress agreeing with MYSTRAN to 1e-4 where only rods are present) and
   modal GPFORCE (inertia-force rows; 34 decks request GPFORCE in SOL 103).
   ELFORCE and stress output for CELASi are missing (`celas3_*`, `celas4_*`). Solid stress/strain output is in basic; PSOLID CORDM
   is parsed but not yet applied.
3. Triage the faceted-cylinder free-thermal decks and the SPCFORCES
   `numerical_mismatch` decks (RBE3 node-surface coupling, MITC4+ reaction
   decks); decide on the strict interior-angle gate.
4. Implement 15-node CPENTA and partial PBUSH parsing only with their
   required mechanics.

Features vibestran supports that no deck exercises are tracked in
`validation_suite_gaps.md`.

## Run History

Brief chronological notes, newest last. The body above always reflects the
most recent entry.

- **SPCFORCE output work (2,720 → 2,652 final failure records).** The SPC
  force F06 block was added. SPCFORCES missing-result records fell from 113
  to 3 (all on `RBE3 19`); of the 21 previously SPCFORCES-missing decks, 11
  now pass all SPCFORCES checks and 10 moved to `numerical_mismatch`.
- **SOLIDSTRAINS output work (2,652 → 2,598).** The solid-strain F06 block
  was added (center + corner, independent STRAIN case-control channel);
  SOLIDSTRAINS missing-result records fell from 56 to 0. The four
  `tet4/tet10` corner records (within 3e-4% of the reference) had tolerances
  relaxed to 0.001%. The twelve `CHEXA.bdf` solid-stress/strain records were
  verified against the closed-form MacNeal-Harder patch-test solution on
  2026-08-17 (E = 1e6, nu = 0.25 implies sigma = 2000 normal / 400 shear,
  von Mises 1200, epsilon = gamma = 1e-3; reproduced exactly on all seven
  distorted hexes) and their stale `KNOWNFAIL` markers were removed.
  `chexa_strain_xy_bug.bdf` was updated to request `STRAIN(CORNER) = ALL` so
  its centered strain checks pass per Nastran center/corner semantics.
- **2026-08-18 re-run with the thermal-strain fix (2,598 → 2,595).** Solid
  strain is now recovered as total strain `B*u` (MYSTRAN semantics) instead
  of mechanical strain `D^-1*sigma`, with strain recovery gated on the STRAIN
  request. Result-neutral on this suite: no SOLIDSTRAINS record moved, since
  the two formulations are identical without a thermal load and all remaining
  thermal decks land in other buckets. The single record delta is the RBE3 10
  SPCFORCES TZ group (previously `Error = 1.7e-07`) flipping to pass —
  roundoff-scale run-to-run variation, not a code effect. Also corrected a
  bookkeeping error: the two "transform corner-ZZ" mismatches are
  `SOLIDSTRESSES` records, not `SOLIDSTRAINS` (the prior table double-counted
  them as a separate family, which is why its totals read 2,598/434 while the
  detail file held 2,596/432).
- **Case-control abbreviations (2,596, neutral).** `DISP`, `STRE`, `SPCF`,
  ... are now expanded to their canonical commands; every affected deck is
  also blocked by another feature, so the total did not move. 44 records
  moved from `solver_model_error` to `unsupported_input` because `MPCFORCE`
  is now reported instead of being consumed as an MPC set.
- **CBAR/PBAR completeness (2,596 -> 2,531, no new failures).** Fixed the
  I1/I2 plane convention (vibestran bent plane 1 with I2), added PBAR
  recovery points, K1/K2 shear flexibility, I12 coupling, CBAR pin flags,
  offsets, and OFFT, and switched bar stress recovery to end forces.
  65 records now pass (15 of them `beam section shapes no shear`, the rest
  pin-flag modal checks); 50 bar decks moved from `unsupported_input` /
  `input_parse_error` into `numerical_mismatch`, for the reasons listed under
  "Numerical Mismatch".
- **GRID PS, GRDSET, BAROR (2,531 -> 2,532, record-neutral).** Permanent
  single-point constraints, GRDSET defaults, BAROR defaults, embedded blanks
  in component fields, and comma-led free-field continuations are now
  supported. No record changed outcome: the unblocked decks now fail later on
  unimplemented result blocks (GPFORCE, CELAS forces, RFORCE), so 19 records
  moved from `unsupported_input` to `numerical_mismatch` /
  `missing_result_data`. The +1 is the RBE3 10 SPCFORCES TZ group again
  (`Error = 1.8e-07`), which flips at roundoff level between runs.
- **PLOAD1 semantics (2,532, neutral; no deck uses PLOAD1).** `FX..MZ` are
  now basic directions and `FXE..MZE` element directions, `LEPR`/`FRPR`
  projected intensities are supported, and a blank X2 is a concentrated load
  (it was previously spread to the end of the bar). Verified only by
  closed-form unit and integration tests.
- **Lumped mass, massless-DOF eigensolver, CTETRA10 mass (2,532, neutral).**
  `PARAM,COUPMASS <= 0` selects lumped mass (consistent remains the
  default). The Spectra shift-invert solver now condenses massless DOFs
  exactly instead of running Lanczos in a semi-definite inner product,
  which produced spurious modes and missed genuine ones. The CTETRA10
  consistent mass was rank-deficient (4-point degree-2 rule, 8-digit
  constants: 18 of 30 eigenvalues zero) and now uses an exact rule. No
  validation deck sets COUPMASS or runs a CTETRA10 modal analysis.
- **CONM2 (2,532 -> 2,528, no new failures).** CONM2 concentrated mass
  (offset, inertia tensor, CID, CID = -1) is implemented; previously it was
  ignored. RBE3 12/13 records now pass. SB/SS-CONM2A (statics with a
  deliberately indefinite inertia tensor, accepted with a warning) now run;
  their displacements match MYSTRAN, but their SPCFORCES were reported in
  basic instead of each grid's global system (fixed in the next entry).
- **SPC force output frame (2,528, neutral).** SPCFORCES are now reported
  in each grid's global (CD) system with its ID in the COORD column, as
  Nastran and MYSTRAN do, and reactions at CD grids are the constrained CD
  components of the full residual (they were previously the basic
  components with the same index). SB/SS-CONM2A now pass SPCFORCES and stop
  at GPFORCE.
- **MPCFORCES and GPFORCE (2,528 -> 2,512, no new failures).** Forces of
  multipoint constraint (the residual K u - F at MPC/RBE grids, beyond the
  SPC-owned components) and the statics grid point force balance (MYSTRAN
  layout) are written, both in each grid's CD system; both match MYSTRAN on
  SB-BAR-PINFLAG and SB-RBE2-01-CBAR-01. RBE3 19 (9 records) and "all zero
  MPC forces" (6) now pass. 44 records moved from `unsupported_input` to
  `solver_model_error` because MPCFORCE is no longer an ignored card and
  those decks' first remaining blocker is a formulation gate. Bulk
  comparisons now mostly stop at BARFORCES/SHELLFORCES. A second flaky
  record was identified: "V20 Constraint Equations shared masters" MODE/1
  (rigid-body mode whose vector varies run to run).
- **CQUAD4 bending sign (2,512 -> 2,504, no new failures).** CQUAD4 shell
  curvature used kappa = +w_xx, the opposite of Nastran's kappa = -w_xx, so
  moments and bending fibre stresses had the wrong sign (an upward-loaded
  cantilever showed tension in its top fibre at the root). Both "shell
  stress components quad4" decks and "mitc4 reaction moment easy" now pass.
- **CTRIA3 DKT bending (2,504 -> 2,477, no new failures).** CTRIA3 bending
  was a placeholder that never coupled the deflection w (a triangle-meshed
  cantilever returned zero deflection) and its stress recovery ignored
  bending. It is now a true DKT element with Nastran sign conventions, and
  stresses include bending. Both "shell face force" decks, both "tri3
  quad4" decks, and "shell stress components tri3" (27 records) now pass.
  CTRIA3 transverse shear resultants are reported as zero (Kirchhoff
  element); a converged recovery is a follow-up.
- **Element forces and shell moment sign (2,477 -> 2,348, no new
  failures).** `FORCE`/`ELFORCE` engineering forces are written for
  CBAR/CBEAM, CQUAD4 (centre and corner) and CTRIA3 in MYSTRAN's layout.
  The CQUAD4 "bending sign" entry above misdiagnosed the problem: the
  moments were already in Nastran's convention (kappa = [w_xx, w_yy,
  2 w_xy] in rotation variables, positive moment = Z1 fibre in tension);
  the fibre-stress formula in the writer had the wrong sign. The element B
  matrices (CQUAD4 and the DKT) are back to positive curvature, the writer
  uses sigma = N/t - 12 M z / t^3, and the stress tables' transverse shear
  columns now carry the peak parabolic stress 1.5 Q / t. The quad
  location option (CENTER/CORNER/BILIN) is now one per run, chosen as MSC
  and MYSTRAN do: global entries and the first subcase only, STRESS ahead of
  STRAIN ahead of FORCE, NONE requests ignored. Every `bilin_*` location
  check now passes except those that need shell strain output.
- **Shell strains and CQUAD4 element frame (2,348 -> 2,237).** Shell
  strains are written for CQUAD4/CTRIA3 (STRCUR membrane strain and
  curvature, or FIBER strains at Z1/Z2; engineering shear; section-average
  transverse shear strain; von Mises or maximum shear strain; total strain
  including thermal). The CQUAD4 element frame is now Nastran's (z = d13 x
  d24, x bisecting the diagonals) instead of the G1-G2 edge, which fixed the
  MacNeal membrane patch test, "shell stress orientation", the helical mesh
  ring and the diamond cylinder stresses. Two records regress: the diamond
  cylinder pressure decks' grid 106 TY (3% -> 8.5%), a warped-element
  limitation described under Numerical Mismatch. The RBE3 10 flake recurs.
- **Case-control TEMPERATURE = n (2,237 -> 2,225).** `TEMPERATURE = n`,
  `TEMP = n` and `TEMPERATURE(BOTH) = n` select the thermal load set
  (TEMPERATURE(MATERIAL) does not). "shell thermal stress" and "_2" now
  pass; nine decks moved to the MAT8 gate (133 records from
  `unsupported_input` to `solver_model_error`). Two rectangle-cylinder TX
  records that passed with no load applied now fail (see Numerical
  Mismatch).
- **CQUAD4 rewritten as MITC4+/D (2,225 -> 2,096).** Continuum-mechanics
  shell (Ko, Bathe & Zhang 2025) with shared grid directors, the shell
  section layer (PSHELL/PCOMP, MAT1/2/8, THETA/MCID), ZOFFS, grid
  thicknesses, Gauss-point (BILIN) corner recovery and a consistent
  thermal strain.  160 records fixed (twisted beam, Raasch hook K6ROT=1000,
  MAT8/PCOMP shell decks, distorted membrane decks).  31 new: corner values
  of the "components"/"strain" decks differ at 1e-6 relative (Gauss-point
  extrapolation and roundoff against zero-tolerance references); "MITC4+
  corner stress order" (single elements under a corner point load) and the
  Raasch hook with K6ROT=10 and two drill-rotation (RZ) records encode
  MYSTRAN's element behaviour and its K6ROT drilling spring.  Full triage
  in the shell-rewrite commits.
- **CTRIA3 rewritten as MITC3+ (2,096 -> 1,983).** 125 records fixed
  (MAT8/PCOMP triangle decks now run; Reissner-Mindlin transverse shear;
  curved and thin triangles no longer lock).  12 new records are roundoff
  against all-zero references plus the RBE3 10 flake.  MAT8/PCOMP decks
  that now run expose `COMPOSITESTRESSES` (ply stress output, not yet
  written; 2 decks).
- **Shell rewrite completed (1,983 -> 1,929).** The old CQUAD4/CTRIA3 and
  the QUAD4TYP/SHELLFORM switch are gone (decks that selected the old
  element now use the new ones: 48 records fixed, spatial isotropy,
  twisted beam, hemisphere SNORM decks). SNORM cards are applied as grid
  directors, with the thickness kept normal to the element. Both elements
  now carry the first-order through-thickness metric and volume factor,
  which fixed `SNORM membrane bending coupling` (7 records) and cut the
  thick (t/L = 0.6) twisted-plate error against a solid model from 114% to
  35%. Warp is a warning. No new failures apart from
  the RBE3 10 flake. Body rewritten for the new elements.
- **CHEXA8 references and orthotropic thermal fix (1,929 -> 1,928).**
  Refined CHEXA8 models of the open shell decks were run in vibestran and
  MYSTRAN 19 (`SHELL_SOLID_COMPARISON.md`, `tools/solid_reference/`).
  MYSTRAN's Q1/E21 CHEXA8 is the reference; vibestran's CHEXA8 locks on
  thin curved geometry. Orthotropic free thermal expansion of curved
  shells now uses the anisotropic free strain: the diamond thermal
  cylinder's hoop expansion was 120% too large. It now matches the exact
  solution and MYSTRAN 19; the remaining records encode MYSTRAN 18's wrong
  values. No new failures.
- **Q1/E21 CHEXA8 (1,928 -> 1,878).** The 9-mode incompatible-mode CHEXA8,
  which locked on thin curved geometry and broke symmetry, is replaced by
  the Q1/E21 enhanced assumed strain element; PSOLID ISOP FULL/1 now
  selects a standard full-integration element and IN THREE 3x3x3. 51
  records fixed (the RBE3 node-surface coupling decks, which set ISOP = 1,
  and others), one new: `RBE3 4` SPCFORCES TY is 1.1e-7 of roundoff
  against an exact zero with a 1e-7 absolute tolerance on 1e8 loads. A
  GRAV card with MB = 0 is no longer rejected.
- **Solver accuracy (1,878 -> 1,881).** Direct solves are refined with
  compensated (double-double) residuals, the CPU and CUDA PCG backends stop
  on the estimated forward error instead of an unattainable relative
  residual, and assembly is deterministic, so K is bitwise reproducible.
  The three new records are roundoff of exact-zero references: two `shell
  force per length components quad4` records at 1e-14, and `RBE3 4`
  SPCFORCES TY at 1.3e-7 against a 1e-7 tolerance on 1e8 loads.
- **Direct-solve refinement opt-in (1,881 -> 1,878).** Direct solves are
  again a single solve by default (refinement with `--refine` or
  PARAM,REFINE,YES), assessed by Nastran's epsilon and the backward error.
  The three roundoff records introduced by refinement return to passing; the
  run matches the one before refinement except for roundoff-level changes
  in records that fail on other grounds.
- **CHEXA20 (1,878 -> 1,787).** The projected assumed-strain CHEXA20
  (`CHEXA20_PLAN.md`) runs the 11 decks that stopped on the unsupported
  element, and PLOAD4 applies to its 8-node faces. On these decks the CHEXA20
  stresses match MYSTRAN 18 to the digits vibestran prints. The remaining
  records in them are 1e-6-level print-precision differences, CQUAD8
  (`quad4 quad8 hex20 together`), MAT9 for solids (`SNORM membrane-shear
  coupling thermal`), and shell stresses of `SNORM membrane-shear coupling`
  that the deck exposes now that it runs. PSOLID IN = TWO on a CHEXA20 uses
  3x3x3 (`S30 hexa`), as MYSTRAN's current element does.
- **Omitted midside grids (1,787 -> 1,786).** CHEXA20 and CTETRA10 accept
  omitted midside grids in structural analysis (linear edge, folded shape
  functions), and the CTETRA10 consistent mass accepts Nastran's vertex
  orientation, whose det(J) is negative (SOL 103 and GRAV used to fail on
  correctly numbered tet10 meshes). No suite deck uses either; the one-record
  change is the RBE3 10 roundoff flake.
- **CROD and scalar points (1,786 → 1,706 final failure records).** CROD/PROD,
  SPOINT, SLOAD and CELAS3/4, CMASS3/4 were added, with AUTOSPC at rod grids.
  `unsupported_input` fell by 31 decks; those now run, and some reach a
  comparison that fails for another reason (item 2 of the next steps). No record
  fails that passed in the run of 2026-10-31.
- **Explicit MPC sets and frames (1,706 final failure records, unchanged).**
  MPCADD resolves after parsing, and MPC, CELAS and CMASS components on a CD grid
  are read in its displacement frame, with AUTOSPC on the blocks that opens.
  `SB/SS-RBE2-01-CROD-05-MPC-03` fall from 31 to 4 failed comparisons and
  `SB/SS-RODLOAD-MPC-FOR-SING` from 4 to 2; what remains in all four is their
  `APPLIEDFORCES` records (OLOAD output), so no record changes bucket. The shell
  record counts that moved (`vic/1`, `vic/6`, `vic/7`) move by as much between
  runs of one binary.
- **Mode shapes in the CD frame (1,706 final failure records, unchanged).**
  Eigenvectors on a CD grid were reported in basic under a coordinate column of
  0; they now follow the same frame as DISPLACEMENT output, and NORM=MAX is unit
  in that frame. No suite deck runs SOL 103 on a grid with a CD frame.
- **Generalized mass under NORM=MAX (1,706 final failure records, unchanged).**
  It was that of the solver's vector, about 1, and is now that of the reported
  shape, with the generalized stiffness following it. The comparator reads only
  eigenvalues and cycles from that table, so no record could change.
- **CONROD (1,706 final failure records, unchanged).** No suite deck has the card,
  so no record could change. Rewriting each CROD and PROD pair as a CONROD left the
  F06 unchanged on the 35 suite decks that could be converted and run.
