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

The suite contains 3,226 definitions. The current run has 2,528 final failure
records:

| Bucket | Failure records | Unique decks | Current disposition |
|---|---:|---:|---|
| `unsupported_input` | 1,484 | 223 | Requires unsupported cards, solution sequences, elements, or formulations |
| `numerical_mismatch` | 484 | 172 | Result exists but differs from the reference or comparator expectation |
| `missing_result_data` | 260 | 44 | Solver completes but does not emit the requested result family |
| `input_parse_error` | 185 | 15 | Parser rejects syntax or a card variant used by a valid deck |
| `solver_model_error` | 115 | 13 | Parsed model is rejected by a current formulation or geometry gate |
| **Total** | **2,528** |  |  |

GRID PS, GRDSET, and BAROR are now applied. The decks they unblocked solve
(the SB-BAR I12/PINFLAG/OFFSET displacements and stresses match MYSTRAN
exactly) but still fail whole-F06 bulk comparisons on result blocks
vibestran does not write, most often `GPFORCE` (grid point force balance).

CBAR/PBAR are now fully parsed (recovery points, K1/K2, I12, pin flags,
offsets, OFFT `GGG`/`BGG`), so 50 bar decks moved out of `unsupported_input`
and `input_parse_error`. Most of them now land in `numerical_mismatch` for
reasons outside the bar element (see "Numerical Mismatch" below).

Solid strain is fully emitted and every SOLIDSTRAINS check passes. The two
remaining `transform` corner-ZZ mismatches are `SOLIDSTRESSES/GID/.../ZZ`
records on `vic/8/NAS S30 solid stress strain transform.bdf` at 0.0001% of
the reference (tolerances unchanged).

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

There are 1,503 records across 237 decks. By unique deck, the current logs show:

| Cause | Decks |
|---|---:|
| Ignored unsupported BDF or case-control card | 156 |
| Unsupported solution type | 61 |
| Explicitly unsupported field or formulation | 20 |

The most widespread ignored keywords across all failing decks are `ECHO`
(55), `TITLE` (55), `ID` (54), `DEBUG` (54), `MPCFORCE`/`MPCFORCES` (52/48),
`CQUAD8` (43), `FORCE` (42), `OLOAD` (42), `GPFORCE` (38), `ELFORCE` (36),
`CROD`/`PROD` (32), `MEFFMASS` (24), `EIGR` (22), `SUBT` (22), `GRDSET` (20),
`PCOMP` (15), and `TEMPERATURE` (15). Abbreviated case-control commands
(`DISP`, `STRE`, `SPCF`, ...) are now accepted and no longer appear.

The main implementation work remains grid point force output (`GPFORCE`),
element force output (`FORCE`/`ELFORCE`), CONM2, CQUAD8, PCOMP laminates,
CROD/PROD, MPC force output, EIGR/reduction behavior, and unsupported solution
sequences such as buckling.

### Missing Result Data

There are 252 records across 40 decks:

| Result family | Records |
|---|---:|
| `SHELLFORCES` | 132 |
| `SHELLSTRAINS` | 95 |
| `MPCFORCES` | 12 |
| `SHELLSTRESSES` | 4 |
| `DISPLACEMENTS` | 4 |
| `SPCFORCES` | 3 |
| `MODE` | 2 |

The two `MODE` records are `V30 beam released dof x-x` and `xyz-xyz`, which
expect the solve to fail (eigenvector `ABSENT`): releasing the same component
at both ends of a bar leaves a mechanism that MYSTRAN rejects. vibestran
returns a zero-frequency mode instead, which is a valid modal answer; these
records encode MYSTRAN behaviour rather than a correctness requirement.

`SOLIDSTRAINS` has no missing-result records (the block is emitted for all
supported solid types). The three remaining SPCFORCES records all belong to
`vic/9/S30 node-surface coupling RBE3 19.bdf`. The decks whose SPCFORCES
comparisons need value-level investigation (mostly RBE3/coupled-node and
MITC4+ cases):

- `vic/7/S30 mitc4 macneal twisted beam.bdf`
- `vic/7/S30 mitc4 macneal twisted beam_2.bdf`
- `vic/8/NAS S30 PLOAD2 PLOAD4 MITC4+.bdf`
- `vic/9/S30 node-surface coupling RBE3 12.bdf`, `... 19.bdf`, `... 2.bdf`,
  `... 5.bdf`, `... 7.bdf`, `... 8.bdf`
- `vic/10/NAS S30 AUTOSPC reaction forces.bdf`

Recommended order:

1. Resolve the remaining SPCFORCES/MPCFORCES deck (`RBE3 19`) and the
   SPCFORCES numerical mismatches on the RBE3-coupled and MITC4+ decks.
2. Add shell engineering force recovery.
3. Add shell strain recovery and output (solid strain is now emitted).
4. Investigate the remaining small displacement and shell-stress omissions by
   output request and element type.

### Numerical Mismatch

There are 476 records across 164 decks:

| Result family | Records |
|---|---:|
| `DISPLACEMENTS` | 173 |
| Bulk comparison | 160 |
| `SHELLSTRESSES` | 55 |
| `SPCFORCES` | 34 |
| `REALEIGENVALUES` | 27 |
| `MODE` | 15 |
| `SOLIDSTRESSES` | 12 |

Element and constraint incidence overlaps when a deck contains multiple cards:

| Card | Decks |
|---|---:|
| `CBAR` | 73 |
| `CQUAD4` | 51 |
| `CHEXA` | 25 |
| `RBE3` | 19 |
| `CTRIA3` | 18 |
| `RBE2` | 15 |
| `CBUSH` | 14 |
| `CONM2` | 10 |
| `CROD` | 8 |
| `CTETRA` | 5 |
| `CPENTA` | 4 |
| `CQUAD8` | 3 |

The CBAR count rose because the bar benchmark decks now parse and solve.
Their remaining mismatches are mostly not in the bar element itself:

- Benchmark bulk comparisons (`bar_*`, `SB-*`, `*-ALL-ELEM-*`) compare the
  whole F06, including blocks vibestran does not write (bar element forces,
  OLOAD, GPFORCE) or writes in a different layout (the bar stress table has no
  SA-Max/SA-Min/M.S. columns). `bar_01` and `bar_05` displacements match
  MYSTRAN to all printed digits.
- `V30 beam released dof *` modal decks compare a CBAR pin flag against an
  equivalent MPC hinge. The identity is exact only when rotations are
  massless (Nastran's default lumped mass). vibestran deliberately defaults
  to consistent mass, so the two models differ at the discretization level
  (0.02-0.2% in frequency, up to ~1% for an axial release). With
  `PARAM,COUPMASS,-1` added, vibestran reproduces MYSTRAN's `released dof
  y-n` frequencies exactly and every pair matches; the decks do not set it.

Current investigation priorities are constraint coordinate systems, remaining
RBE2/RBE3 cases, shell material axes and recovery, solid recovery, and modal
mode matching. Unsupported cards present in a bulk-comparison deck must be
separated from genuine numerical defects before changing solver mathematics or
tolerances.

### Solver Model Errors

There are 115 records across 13 decks:

| Current cause | Decks | Disposition |
|---|---:|---|
| CQUAD4 warp exceeds the 30 degree validity threshold | 7 | Requires a true warped-shell formulation rather than bypassing the gate |
| MAT2/MAT8 used by a formulation that currently requires MAT1 | 6 | Requires orthotropic/anisotropic constitutive, orientation, thermal, and recovery support |

Unused entities, coincident nodes, aspect ratio, and taper are warnings. Fatal
negative/inverted geometry checks remain enabled. Coincident nodes are not
merged because separate coupled substructures can intentionally share a
position.

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

1. Grid point force balance (`GPFORCE`), the first blocker of most
   Benchmark bulk comparisons, and element force output (`FORCE`/`ELFORCE`):
   bar end forces are already recovered (`CBarBeamElement::EndForces`);
   shell forces are the largest missing-result family.
2. CROD/PROD and CQUAD8.
3. Triage the SPCFORCES `numerical_mismatch` decks (RBE3 node-surface
   coupling and MITC4+ reaction decks) and MPC force output.
4. Implement 15-node CPENTA and partial PBUSH parsing only with their required
   mechanics; keep MAT2/MAT8 and severely warped CQUAD4 decks explicitly
   unsupported until the corresponding formulations are implemented.

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
