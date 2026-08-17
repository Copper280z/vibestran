# Validation Failure Plan

Current run command:

```text
python3 test.py /Users/bob/Documents/vibe-tran/build/vibestran
```

This document describes only the current result of that run. The complete
per-definition classification is in `failure_buckets.txt`. Each detail row
contains the test type, deck, requested path, solver return code, solver log,
F06 path, and comparison message. Per-deck solver output is in
`diagnostics/**/*.solver.log`.

## Current Summary

The suite contains 3,226 definitions. The current run has 2,652 final failure
records (down 68 from the previous 2,720, driven by the SPCFORCE output work):

| Bucket | Failure records | Unique decks | Current disposition |
|---|---:|---:|---|
| `unsupported_input` | 1,570 | 277 | Requires unsupported cards, solution sequences, elements, or formulations |
| `numerical_mismatch` | 431 | 137 | Result exists but differs from the reference or comparator expectation; SPCFORCES records moved here from `missing_result_data` once the block is emitted |
| `missing_result_data` | 301 | 45 | Solver completes but does not emit the requested result family |
| `input_parse_error` | 185 | 16 | Parser rejects syntax or a card variant used by a valid deck |
| `solver_model_error` | 159 | 19 | Parsed model is rejected by a current formulation or geometry gate |
| `knownfail_regression` | 6 | 2 | A result marked `KNOWNFAIL` now passes and its expectation needs review |
| **Total** | **2,652** |  |  |

The only result-family that changed buckets is SPCFORCES: `missing_result_data`
dropped from 113 to 3 records (all three remaining belong to
`vic/9/S30 node-surface coupling RBE3 19.bdf`). Of the 21 previously
SPCFORCES-missing decks, 11 now pass all their SPCFORCES checks and 10 show
`numerical_mismatch` (block emitted but values differ from the MYSTRAN
reference; RBE3 19 has records in both buckets).

Another 21 records still fail in the way declared by their `KNOWNFAIL`
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

There are 1,570 records across 270 decks. By unique deck, the current logs show:

| Cause | Decks |
|---|---:|
| Ignored unsupported BDF or case-control card | 138 |
| Explicitly unsupported field or formulation | 71 |
| Unsupported solution type | 61 |

The most widespread ignored keywords are `DISP` (48 decks), `CQUAD8` (43),
`ECHO` (41), `FORCE` (41), `TITLE` (41), `DEBUG` (40), `ID` (40), `GPFORCE`
(30), `OLOAD` (28), `ELFORCE` (28), `PROD` (26), `CROD` (26), `SUBT` (22),
`GRDSET` (20), `MEFFMASS` (18), `EIGR` (16), `PCOMP` (15), and `TEMPERATURE`
(15). A deck may contain more than one ignored keyword.

The main implementation work remains CQUAD8, PCOMP laminates, CROD/PROD,
advanced case-control output, EIGR/reduction behavior, and unsupported solution
sequences such as buckling. These are feature additions, not fixes to make as
part of failure-bucket classification.

### Missing Result Data

There are 301 records across 45 decks:

| Result family | Records |
|---|---:|
| `SHELLFORCES` | 132 |
| `SHELLSTRAINS` | 95 |
| `SOLIDSTRAINS` | 56 |
| `MPCFORCES` | 15 |
| `SHELLSTRESSES` | 5 |
| `DISPLACEMENTS` | 4 |
| `SPCFORCES` | 3 |

`SPCFORCES` missing-result records fell from 113 to 3 after the SPC force
output work; the remaining three are all for
`vic/9/S30 node-surface coupling RBE3 19.bdf`. The decks whose SPCFORCES
moved to `numerical_mismatch` need value-level investigation (mostly
RBE3/coupled-node and MITC4+ cases):

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
3. Add shell and solid strain recovery and output.
4. Investigate the remaining small displacement and shell-stress omissions by
   output request and element type.

### Numerical Mismatch

There are 431 records across 137 decks:

The SPCFORCES additions account for 34 of these records (10 decks); the rest are
unchanged from the previous run.

| Result family | Records |
|---|---:|
| `DISPLACEMENTS` | 189 |
| Bulk comparison | 128 |
| `SHELLSTRESSES` | 55 |
| `SOLIDSTRESSES` | 12 |
| `REALEIGENVALUES` | 12 |
| `MODE` | 3 |

Element and constraint incidence overlaps when a deck contains multiple cards:

| Card | Decks |
|---|---:|
| `CQUAD4` | 49 |
| `CBAR` | 37 |
| `CHEXA` | 26 |
| `RBE3` | 20 |
| `CTRIA3` | 16 |
| `RBE2` | 15 |
| `CBUSH` | 11 |
| `CONM2` | 8 |
| `CTETRA` | 5 |
| `CROD` | 5 |
| `CPENTA` | 4 |
| `CQUAD8` | 3 |

Current investigation priorities are constraint coordinate systems, remaining
RBE2/RBE3 cases, shell material axes and recovery, solid recovery, and modal
mode matching. Unsupported cards present in a bulk-comparison deck must be
separated from genuine numerical defects before changing solver mathematics or
tolerances.

### Solver Model Errors

There are 159 records across 19 decks:

| Current cause | Decks | Disposition |
|---|---:|---|
| MAT8 used by a formulation that currently requires MAT1 | 6 | Requires orthotropic constitutive, orientation, thermal, and recovery support |
| CQUAD4 warp exceeds the 30 degree validity threshold | 10 | Requires a true warped-shell formulation rather than bypassing the gate |
| MAT2 MID roles used by a formulation that currently requires MAT1 | 2 | Requires anisotropic MID1/MID2/MID3 support |

Unused entities, coincident nodes, aspect ratio, and taper are warnings. Fatal
negative/inverted geometry checks remain enabled. Coincident nodes are not
merged because separate coupled substructures can intentionally share a
position.

### Input Parse Errors

There are 185 records across 16 decks. The unique deck causes are:

| Parser gap | Decks |
|---|---:|
| 15-node CPENTA | 7 |
| Partial or advanced PBUSH fields | 5 |
| Advanced or unsupported PBARL section data | 2 |
| CELAS2 damping/stress-coefficient fields | 1 |

Parser support must be paired with the corresponding element/formulation
behavior; accepting fields and silently ignoring their mechanics would produce
plausible but invalid results.

### Knownfail Regression

The six records are the six solid-stress components for `CHEXA.bdf`. They are
marked `KNOWNFAIL` but now pass. Verify the values, then remove or update the
stale expectation rather than changing the solver to restore the failure.

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

1. Verify and clear the six stale `KNOWNFAIL` expectations.
2. Triage the 9 SPCFORCES `numerical_mismatch` decks (RBE3 node-surface
   coupling and MITC4+ reaction decks) and close `RBE3 19`'s remaining
   SPCFORCES/MPCFORCES gap.
3. Finish MPC force output to match SPCFORCES, then shell-force and strain
   result families.
4. Triage numerical mismatches by constraint frame, formulation, and result
   recovery family without masking unsupported inputs.
5. Implement 15-node CPENTA and partial PBUSH parsing only with their required
   mechanics.
6. Keep MAT2/MAT8 and severely warped CQUAD4 decks explicitly unsupported until
   the corresponding formulations are implemented.
