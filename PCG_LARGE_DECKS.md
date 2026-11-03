# PCG tests on decks with more than 200000 grids

CUDA fp64, CUDA mixed and Vulkan pass the large physical matrix replays with
GPU-owned iterative state. After the AMG engine rewrite below, the
1.58-million-equation plate solves in **4.29 s CUDA / 4.82 s Vulkan** resident
and **5.38 / 5.72 s** with 90% of its fine-matrix rows on the GPU, versus
**6.44 / 8.28 s** and **7.95 / 9.41 s** before it. Independent solution error
remains below **1.86e-9** at the unchanged **1e-8** target. The complete
nonlinear motor now runs faster than CPU direct on every GPU backend.

Measured on 2026-11-02 on bob-desktop.local: Ryzen 7 5700, eight physical cores
(`taskset -c 0-7`, `OMP_NUM_THREADS=8`), 32 GiB RAM, TITAN V 12 GiB,
driver 580.178.04, CUDA 12.8, Release Meson build. cuDSS is unchanged.

## Physical decks

| Deck | GRID nodes | Elements | Free equations |
|---|---:|---:|---:|
| Full radial magnetic motor | 218,579 | 436,258 CTRIA3 | 225,529 |
| Thermal conduction with internal heat generation | 226,981 | 216,000 CHEXA8 | 219,539 |
| Existing structural plate | 263,169 | 262,144 CQUAD4 | 1,575,936 |

The motor comes from `Magnetic_Validation/mesh_study.py` with `tri:0.225`:
the full 12-slot, 14-pole machine, nonlinear steel B–H data, permanent magnets,
24 winding regions and three phase circuits. The benchmark keeps the full
geometry and materials, selecting one rated zero-angle operating point from
the generator's 78-position sweep. Phase currents are 2.58819, 7.07107 and
−9.65926 A. The existing `PARAM,AIRGAP,BAND` setting is retained; its internal
air-gap remeshing accounts for the equation count exceeding the input GRID
count. The first and tenth Newton tangent systems are replayed with their
actual assembled right-hand sides, and the complete nonlinear deck is also run.

The motor's individual direct tangent factorizations take 0.29–0.50 seconds,
plus approximately 25 ms per right-hand side. The **complete nonlinear CPU
analysis takes 16.72 seconds**, including assembly, 11 Newton iterations,
circuit responses and output. Thus the individual direct solves meet the
requested couple-second scale; the full refined nonlinear analysis does not.
All iterative runs have explicit process time limits.

The thermal case is a 60×60×60 physical conduction mesh with a 100:1
conductivity interface, prescribed hot/cold faces and a localized 1000 W/m³
heat source crossing that interface. It extends the repository's thermal
generator. The structural case is the existing `plate262k.bdf`, unchanged.
All replay loads come from FE assembly; none use manufactured right-hand sides.

## AMG engine rewrite and factor reuse

The hierarchy is still AMGCL 1.5.0's smoothed aggregation with SPAI-0
smoothing (coarse_enough 256), but AMGCL now only builds it. Each backend runs
its own V-cycle and CG:

- **Zero-guess cycle.** The first pre-smoothing step of every level starts
  from zero, so it is a diagonal scaling rather than a product: one fewer
  sparse product per level and cycle. Iterates match AMGCL's own
  amg + cg to rounding (`PcgAmgTest.ZeroGuessVcycleIsAmgclsPreconditioner`);
  iteration counts are unchanged.
- **CUDA.** All CG scalars stay in device memory, with deterministic
  fixed-grid reductions. Residual, smoothing and accumulation are fused into
  SpMV epilogues, with 4–32 lanes per row chosen from the average row length.
  One iteration is a captured CUDA graph, launched eight times per host poll
  when resident and once per iteration when spilling. CPU rows run in a
  captured host node on a side branch.
- **Vulkan.** Pre-recorded command buffers are replayed, with device-side CG
  state and specialised SpMV pipelines. Subgroup reduction is used where the
  device supports relative shuffles; otherwise shared memory. Pipelines are
  cached per context. Spill solves split the recording at each fine product
  so the CPU rows run between submissions.
- **Factor reuse.** `factor()` on the CUDA PCG, CUDA mixed and Vulkan backends
  (n ≥ 50,000) returns a factor that keeps the equilibrated matrix, hierarchy
  and device workspace across right-hand sides. `refactor()` with an unchanged
  pattern keeps the aggregates and transfer operators and recomputes only the
  Galerkin operators, smoothers and coarse inverse (AMGCL's rebuild). A new
  pattern, or a rebuilt hierarchy needing more than twice the iterations of
  its fresh build, builds again. The nonlinear motor builds once and rebuilds
  values for the other 11 tangents.
- **CUDA spill ordering.** The gather of CPU-row inputs runs before the fork
  to the CPU branch. As its own branch it waited for SMs behind the large
  prefix product, so the CPU rows effectively ran after that product instead of
  alongside it. On the plate at 90% this cut the CPU-callback gaps from
  4.24 s to 2.19 s and the solve from 7.58 s to 5.50 s.
- **Refinement.** The compensated residual and ‖K‖∞ are evaluated
  row-parallel. Lower-triangular storage uses a transposed index built once
  per refinement. Every row is summed in the serial order, so results are
  bitwise identical; on the plate, 243 ms of serial work becomes about 130 ms.

Times are **median seconds of three solves**, measured as before (affinity
0–7, eight threads, no warmup, setup and refinement included). Row fractions
use the production planner's exact 50/75/90% budgets. The logged GPU/CPU split
of every repeat is checked against the target. Arrows show commit `f13d5cc`
→ this change, both measured in the same session:

| System | Backend | 50% GPU rows | 75% GPU rows | 90% GPU rows | Resident |
|---|---|---:|---:|---:|---:|
| Thermal | CUDA fp64 | 0.880 → 0.609 | 0.417 → 0.265 | 0.377 → 0.193 | 0.360 → 0.192 |
| Thermal | CUDA mixed | | | | 0.402 → 0.180 |
| Thermal | Vulkan | 1.019 → 0.665 | 0.570 → 0.339 | 0.551 → 0.304 | 0.500 → 0.278 |
| Motor, first tangent | CUDA fp64 | 0.869 → 0.618 | 0.780 → 0.455 | 0.539 → 0.356 | 0.471 → 0.199 |
| Motor, first tangent | CUDA mixed | | | | 0.440 → 0.220 |
| Motor, first tangent | Vulkan | 1.136 → 0.698 | 0.937 → 0.567 | 0.781 → 0.485 | 0.603 → 0.359 |
| Motor, late tangent | CUDA fp64 | 0.568 → 0.414 | 0.464 → 0.307 | 0.317 → 0.207 | 0.287 → 0.145 |
| Motor, late tangent | CUDA mixed | | | | 0.296 → 0.132 |
| Motor, late tangent | Vulkan | 0.781 → 0.487 | 0.633 → 0.398 | 0.543 → 0.321 | 0.429 → 0.259 |
| Plate | CUDA fp64 | 28.816 → 20.916 | 15.801 → 11.236 | 7.945 → 5.385 | 6.443 → 4.294 |
| Plate | CUDA mixed | | | | 6.163 → 4.269 |
| Plate | Vulkan | 30.981 → 21.330 | 17.361 → 11.610 | 9.407 → 5.723 | 8.283 → 4.824 |

CUDA mixed spills through the fp64 engine, so its spill times are CUDA
fp64's. The 50% plate remains bound by the CPU rows: about 5.9 ms per
product for 788k rows, close to host memory bandwidth, plus 12.7 MB of
transfers. The plate keeps **872 iterations** (mixed resident: 1016). All 72
timed configurations, old and new, pass the 1e-8 reference check; the
maximum reference error is **1.856e-9**. The new code's fixed device state
is the same for CUDA and Vulkan, so their row budgets now coincide (plate
90%: 677.98 MiB).

Factor reuse on the late motor tangent (median of five runs):

| Backend | Solve, with setup | Factor solve | Same-pattern refactor + solve |
|---|---:|---:|---:|
| CUDA fp64 | 0.149 | 0.106 | 0.123 |
| Vulkan | 0.279 | 0.150 | 0.177 |

The refactored hierarchy is built from the first tangent's aggregates and
needs 277 iterations, against 279 for a fresh one. Each further plate
right-hand side takes 3.69 s CUDA / 3.95 s Vulkan.

Measured and rejected (plate, CUDA resident, single runs; 4.44 s with the defaults):

| Variant | Time (s) |
|---|---:|
| Node block aggregation, block size 6 | 5.96 |
| Block 6 with a rigid-body near-nullspace | 14.80 |
| Block size 3 | 23.10 |
| Spectral radius estimate | 4.88 |
| eps_strong 0.02 | 9.66 |
| First inner tolerance 1e-10 | 4.58 |
| First inner tolerance 1e-12 | 4.37 |
| Correction tolerance 1e-2 | 5.04 |
| Correction tolerance 1e-4 | 5.28 |
| Blocking-sync poll wait while spilling, 90% | 5.53 (vs 5.50) |

Block size 2 failed coarse Cholesky. The 1e-12 first tolerance is within
noise of 1e-8 and was not adopted.

All **18 complete GPU analyses pass** (single runs, seconds; spill budgets are
the new 90% budgets rounded up to whole MiB: 78, 52 and 678). Arrows show the
previous section's runs → this change:

| Deck | CPU direct | CUDA resident | CUDA spill | Mixed resident | Mixed spill | Vulkan resident | Vulkan spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| Thermal | 10.02 | 3.84 → 3.63 | 3.92 → 3.62 | 3.80 → 3.60 | 3.77 → 3.59 | 3.99 → 3.87 | 4.04 → 3.78 |
| Full nonlinear motor | 16.72 | 18.21 → 14.88 | 18.81 → 15.66 | 16.73 → 14.54 | 17.31 → 14.99 | 20.46 → 15.88 | 22.75 → 17.14 |
| Plate | 26.71 | 21.58 → 19.62 | 23.00 → 20.50 | 21.45 → 19.05 | 23.10 → 20.56 | 24.15 → 20.24 | 25.55 → 21.20 |

The six motor runs converge in **11 Newton iterations**. Compared with CPU
direct, their fields and circuits have maximum relative error
**2.357e-12**, and torque differs by at most **3.065e-14**. All pass
`tools/verify_pcg_motor_outputs.py` at 1e-8.

New tests cover the zero-guess cycle, same-pattern and changed-pattern
refactor, multi-column factor solves, and far-off-diagonal transposed
entries in the parallel residual. CUDA and Vulkan factor tests run resident,
with about half the fine rows on the CPU (13 MiB) and with the host-hierarchy
fallback (1 MiB). `unit_tests` (680), `cuda_tests` and `vulkan_tests` pass on
the GPU host. Run concurrently with the GPU suites, `unit_tests` exceeds
meson's 60 s limit; alone it takes 37 s. The local suite passes.
`ninja cppcheck` reports no errors: 18 existing unused-function findings
locally, and 101 findings on the GPU host, the same count as before this
change. The host build also analyses the CUDA and CHOLMOD code, which the
local Accelerate build compiles out. The earlier Vulkan AMG test's 8 MiB budget fell below the
fixed device state and so tested the host fallback; it now uses 13 MiB.

[Samples, baseline runs, experiments, complete analyses and motor comparisons](benchmarks/pcg/titan-v-amg-engine-20261102.json)
are retained. `tools/pcg_benchmark --factor` and `--refactor-from FILE` time
factor reuse.

## Device-owned spill: previous implementation

Large CUDA/Vulkan spill solves keep PCG vectors, reductions, smoothing
updates, restriction/prolongation and the coarse AMG operators on the GPU.
Only the small terminal coarse factor/solve remains on the CPU. The fine
matrix is partitioned after reserving this working state. CPU rows download
the unique input entries they reference and upload only their output rows;
the GPU prefix runs concurrently with the CPU product. Couplings across the
partition remain exact. Initial and refinement CG solves reuse one workspace
with their original tolerances, removing four duplicate full-size vectors.

**All 176 replay solves pass**: 108 row-controlled solves, 36 resident controls,
24 old-version equal-budget controls and eight calibration solves. All use
1e-8 acceptance and independent reference checks; the maximum relative
infinity-norm solution error is **1.856e-9**. No tolerance was loosened.

Times below are **median seconds of three solves**, without a discarded warmup.
They include setup, hierarchy construction, transfers and refinement; matrix
loading, backend creation and independent verification are excluded. The same
TITAN V, affinity 0–7 and eight OpenMP threads are used; runs are serialized.
Every spill repeat verifies the actual GPU/CPU row count.

| System | Backend | 50% GPU rows | 75% GPU rows | 90% GPU rows | Resident |
|---|---|---:|---:|---:|---:|
| Thermal | CUDA fp64 | 0.873 | 0.417 | 0.379 | 0.361 |
| Thermal | CUDA mixed | 0.884 | 0.410 | 0.378 | 0.381 |
| Thermal | Vulkan | 1.040 | 0.615 | 0.556 | 0.540 |
| Motor, first tangent | CUDA fp64 | 0.865 | 0.778 | 0.538 | 0.410 |
| Motor, first tangent | CUDA mixed | 0.866 | 0.781 | 0.539 | 0.406 |
| Motor, first tangent | Vulkan | 1.143 | 0.925 | 0.770 | 0.595 |
| Motor, late tangent | CUDA fp64 | 0.578 | 0.447 | 0.315 | 0.288 |
| Motor, late tangent | CUDA mixed | 0.578 | 0.444 | 0.322 | 0.277 |
| Motor, late tangent | Vulkan | 0.756 | 0.675 | 0.573 | 0.446 |
| Plate | CUDA fp64 | 28.719 | 15.817 | 7.896 | 6.416 |
| Plate | CUDA mixed | 28.717 | 15.761 | 7.882 | 6.126 |
| Plate | Vulkan | 31.215 | 17.237 | 9.341 | 8.169 |

At 90%, plate solve times are **7.896 s CUDA / 9.341 s Vulkan**, compared
with **48.071 / 63.000 s** in the preceding row-residency sweep. Both are faster
than the retained **11.893 s CPU-direct** median. At 50%, substantial CPU
matrix work remains and both GPU paths are still slower than CPU direct.
The plate keeps **872 iterations** at all three fractions. Motor iteration
counts can vary with the partition and reduction order; individual counts
are retained rather than assuming every timing change is bandwidth alone.

### Equal-budget controls and memory accounting

**Matching row fractions does not match total GPU memory.** The new method
reserves GPU vectors and hierarchy storage before placing the fine matrix.
The byte budgets below come from the production planner; calibration is
separate from the repeated timings. CUDA mixed uses the same fp64 spill path
and budgets as CUDA fp64. Its unbudgeted resident path retains fp32 AMG.

| System | Backend | 50% budget (MiB) | 75% budget (MiB) | 90% budget (MiB) |
|---|---|---:|---:|---:|
| Thermal | CUDA fp64 | 57.362 | 73.112 | 82.561 |
| Thermal | Vulkan | 60.851 | 76.600 | 86.050 |
| Motor, first tangent | CUDA fp64 | 50.138 | 54.210 | 56.282 |
| Motor, first tangent | Vulkan | 54.231 | 58.303 | 60.375 |
| Motor, late tangent | CUDA fp64 | 50.472 | 54.543 | 56.616 |
| Motor, late tangent | Vulkan | 54.594 | 58.666 | 60.738 |
| Plate | CUDA fp64 | 569.570 | 659.799 | 714.020 |
| Plate | Vulkan | 598.083 | 688.312 | 742.533 |

The old executable was also rerun at each new 90% budget. With that budget
it places **100% of the fine rows** on the GPU but retains host vectors and
a host hierarchy. This isolates the benefit of using the available budget
for the rest of the solver:

| System | Backend | Old, same budget (s) | New, 90% rows (s) | Speedup |
|---|---|---:|---:|---:|
| Thermal | CUDA fp64 | 0.744 | 0.379 | 1.96× |
| Thermal | Vulkan | 0.931 | 0.556 | 1.67× |
| Motor, first tangent | CUDA fp64 | 2.100 | 0.538 | 3.90× |
| Motor, first tangent | Vulkan | 2.107 | 0.770 | 2.73× |
| Motor, late tangent | CUDA fp64 | 1.174 | 0.315 | 3.72× |
| Motor, late tangent | Vulkan | 1.377 | 0.573 | 2.40× |
| Plate | CUDA fp64 | 44.008 | 7.896 | 5.57× |
| Plate | Vulkan | 59.898 | 9.341 | 6.41× |

The budget accounts for solver arrays and a 64 KiB allowance; it is not a
hard cap on driver/context/library allocation. In the CUDA 90% plate trace,
Nsight records a **722.11 MiB peak** of device/device-static allocations versus
a **714.02 MiB solver budget**. Those events include existing library storage
and temporary allocations but exclude driver/context memory and pinned host RAM.
Physical 12 GiB VRAM was not exhausted in these tests.

### Profiles

Paired Nsight Systems traces of the plate compare the old and new CUDA paths
at the same 714.02 MiB solver budget. These are instrumented single runs,
kept separate from the timing medians:

| Measurement | Old, fine rows 100% | New, fine rows 90% |
|---|---:|---:|
| Host/device copy volume, including setup | 88.44 GB | 11.80 GB |
| Actual host/device copy duration | 6.781 s | 0.936 s |
| Sum of GPU kernel durations | 3.410 s | 4.737 s |
| First-to-last-kernel interval | 41.093 s | 7.481 s |

The new GPU performs more of the solve, and the long gaps between its work
shrink. Device-to-device copies are excluded from PCIe totals. API wait time
is not counted as transfer time. Sparse products across all AMG levels account
for **88.9% of new CUDA kernel time**. One sampled fine-product kernel reaches
**89.00% active-warp occupancy** and **86.99% sustained DRAM throughput**;
these are kernel counters, not whole-program utilization.

At 90%, the fine operator transfers **3.179 MB/product** (239,772 input entries
and 157,594 output rows), versus 23.954 MB in the earlier 90% path. Vulkan
GPU batches occupy **6.440 s in an 8.805 s first-to-last-batch interval**.
Input gathering uses the actual sparsity graph while row ownership remains
an original-equation-order prefix; no couplings are dropped or approximated.

### Complete-deck validation

All **18 complete GPU analyses pass**. Times below are single end-to-end runs
in seconds, including parsing, assembly, nonlinear iteration and output.
Spill budgets are the measured 90% budgets rounded up to integer MiB; the motor
uses the larger budget of the two captured tangents. Actual residency can vary
with the hierarchy during other Newton steps. CPU direct is the retained baseline.

| Deck | CPU direct | CUDA resident | CUDA spill | Mixed resident | Mixed spill | Vulkan resident | Vulkan spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| Thermal | 10.02 | 3.84 | 3.92 | 3.80 | 3.77 | 3.99 | 4.04 |
| Full nonlinear motor | 16.72 | 18.21 | 18.81 | 16.73 | 17.31 | 20.46 | 22.75 |
| Plate | 26.71 | 21.58 | 23.00 | 21.45 | 23.10 | 24.15 | 25.55 |

All six motor outputs retain identical geometry, converge in **11 Newton
iterations**, and use no B–H extrapolation. Independent field/circuit comparisons
have maximum relative error **4.934e-12**; torque differs by at most
**3.011e-14**. Unavailable all-NaN winding quantities are checked for matching
masks and excluded from numeric error maxima. Complete CLI runs keep their
existing defaults (1e-8 stable CUDA/Vulkan; 1e-6 experimental mixed); matrix
replays explicitly request 1e-8 for all three backends.

CUDA Compute Sanitizer memcheck reports **zero errors** on the first motor
at 90% residency. All **28 PCG and 19 Vulkan tests pass** on the GPU host.
The added partition test checks independently assembled interface fluxes,
nonlocal couplings, exact budget selection and descriptor limits. `ninja cppcheck`
reports no errors locally or on the GPU host. Local findings match the preceding
19 unused-function warnings; the GPU host reports 101 findings, none in the new
spill implementation. cuDSS source and behavior are unchanged.

### Remaining limits and reproduction

If the GPU cannot hold the vectors and coarse hierarchy plus the exchange
workspace, the solver retains the host-based AMG fallback. Very small budgets
still pass correctness tests but do not receive the speedups above. The full
matrix/hierarchy construction and refinement must fit host RAM. This change
does not add disk-backed assembly or solving beyond system RAM.

Build in `build` with `ninja` and `ninja pcg_benchmark`. Using the captured
physical systems described below:

```sh
OMP_NUM_THREADS=8 python3 tools/run_device_spill_benchmarks.py \
  --binary build/pcg_benchmark --matrices build/pcg-large-cases \
  --output build/pcg-large-results/device-spill --repeats 3 --timeout 300
```

Use `--baseline PATH` to include equal-budget controls from a preserved earlier
executable. `tools/run_large_pcg_analyses.py --spill-budgets FILE` accepts a JSON
mapping from case to backend to integer MiB for complete analyses.
`tools/verify_pcg_motor_outputs.py` compares HDF5 physics outputs with CPU direct.

[Replay samples, budgets, commands and complete analyses](benchmarks/pcg/titan-v-device-spill-20261102.json),
[profiler counters and GPU tests](benchmarks/pcg/titan-v-device-spill-profile-20261102.json), and
[independent motor comparisons](benchmarks/pcg/titan-v-device-spill-motor-accuracy-20261102.json)
are retained. Full traces and logs are under
`/tmp/vibestran-pcg-20261102/large/device-spill-validation-20261102` on bob-desktop.local.
The earlier executable is preserved under `baseline-row-residency/` in that checkout.

## Earlier forced-budget matrix timings and accuracy

Times are seconds, including setup, preconditioner construction, transfers,
iterations and compensated refinement. Matrix-file loading, backend creation
and independent answer verification are excluded. Resident GPU numbers are
medians of three solves with no discarded warmup; each rebuilds its hierarchy.
CPU direct medians are the retained three-solve baseline on this host. Spill
numbers are single bounded runs, not statistically stable speedup estimates.
All entries below successfully reach the unchanged accuracy target.

| System | CPU direct | CUDA fp64 resident | CUDA mixed resident | Vulkan resident | CUDA fp64 spill | Vulkan spill |
|---|---:|---:|---:|---:|---:|---:|
| Thermal | 6.719 | 0.365 | 0.381 | 0.505 | 2.678 | 2.284 |
| Motor, first tangent | 0.543 | 0.409 | 0.438 | 0.740 | 2.812 | 3.021 |
| Motor, late tangent | 0.533 | 0.298 | 0.280 | 0.445 | 1.820 | 2.016 |
| Plate | 11.893 | 6.480 | 6.254 | 8.349 | 67.553 | 69.150 |

Mixed CUDA spill uses the same fp64 hybrid AMG path: thermal 2.386 s, first
motor 2.863 s, late motor 1.859 s, plate 66.590 s. The resident mixed path uses
an fp32 preconditioner with an fp64 operator and reductions. Its speed relative
to fp64 varies; it is not uniformly faster on this TITAN V.

Forced budgets are **16 MiB thermal, 8 MiB motor, 128 MiB plate**. These exercise
real row partitioning and vector transfers over PCIe, but the decks themselves
fit in the card's physical 12 GiB. They are not measurements of exhausting
physical VRAM or system RAM. Driver/library allocations are outside the budget
estimate. The full matrix, hierarchy and work arrays must still fit in host RAM.

The unchanged CPU PCG baseline takes 1.862 s for thermal, 27.638/22.989 s for
the motor tangents, and times out on the plate at the 75-second process limit.
CPU PCG has not been improved in this CUDA/Vulkan change; it also remains short
of the requested all-backend performance expectation.

## Earlier host-based row-residency sweep: 50%, 75%, and 90%

This follow-up measures the same four physical matrices at **50%, 75%, and
90% of fine-matrix rows on the GPU**. All **108 solves (36 configurations,
three repeats each) pass** the unchanged 1e-8 target; maximum independent
relative infinity-norm solution error against CPU direct is **1.856e-09**.
Every repeat reports the expected GPU/CPU row counts. Fractions are rounded
down to whole rows.

Times below are median seconds including setup, transfers, hierarchy building,
iterations and refinement. There is no discarded warmup. Matrix loading,
backend creation and independent verification are excluded. CPU affinity remains
0–7 and `OMP_NUM_THREADS=8`; all configurations run sequentially. Solver code is
unchanged from the preceding AMG measurements.

| System | Backend | 50% GPU rows | 75% GPU rows | 90% GPU rows |
|---|---|---:|---:|---:|
| Thermal | CUDA fp64 | 1.573 | 1.036 | 0.768 |
| Thermal | CUDA mixed | 1.566 | 1.040 | 0.787 |
| Thermal | Vulkan | 1.711 | 1.153 | 0.961 |
| Motor, first tangent | CUDA fp64 | 2.383 | 1.841 | 2.178 |
| Motor, first tangent | CUDA mixed | 2.432 | 1.838 | 2.151 |
| Motor, first tangent | Vulkan | 3.176 | 2.108 | 2.533 |
| Motor, late tangent | CUDA fp64 | 1.565 | 1.205 | 1.189 |
| Motor, late tangent | CUDA mixed | 1.564 | 1.206 | 1.192 |
| Motor, late tangent | Vulkan | 1.745 | 1.417 | 1.441 |
| Plate | CUDA fp64 | 58.590 | 53.123 | 48.071 |
| Plate | CUDA mixed | 58.799 | 53.609 | 48.619 |
| Plate | Vulkan | 64.098 | 56.363 | 63.000 |

For the plate, moving from 50% to 90% reduces CUDA fp64 time by **18.0%** and
Vulkan time by **1.7%**. Both still use 872 iterations at every fraction.
Vulkan is fastest at 75%; its 90% median is **11.8% slower** than at 75%.
The three Vulkan plate samples at 90% are 59.412–63.320 seconds, versus
56.214–56.894 seconds at 75%. Unlike the first motor tangent, this slowdown
does not come from additional iterations; this timing sweep does not
identify its cause. These spill times remain well above the retained
11.893-second CPU-direct and 6.480/8.349-second resident CUDA/Vulkan controls.

The byte budgets are computed from the **actual full CSR after exact-zero
removal**, using the original equation-order row prefix. They include that
prefix’s values, column indices and row pointers, a complete input vector,
a prefix output vector, and the selector’s 64 KiB reserve. They are not
measurements of total device allocation or percentages of the card’s 12 GiB.

| System | 50% budget (MiB) | 75% budget (MiB) | 90% budget (MiB) |
|---|---:|---:|---:|
| Thermal | 35.799 | 53.014 | 63.343 |
| Motor, both tangents | 12.614 | 17.812 | 21.247 |
| Plate | 217.063 | 319.299 | 381.075 |

The budget for `r` GPU rows is `12*nnz_prefix + 4*(r+1) + 8*(n+r) + 65536`
bytes. Runtime row-count verification guards against differences between this
calculation and the actual solver’s sparsity/layout. Raw results also record
the nonzero fraction, exact bytes, individual samples, iterations, errors,
file/binary hashes, peak host RSS and commands.

**More GPU rows do not guarantee fewer seconds.** The first motor tangent
takes 440 CUDA iterations at 75% but 524 at 90%, while keeping the same
accuracy target. The late tangent has little further gain above 75%.
Changing the partition changes the floating-point summation path; the
recorded iteration counts distinguish this from a pure bandwidth comparison.

These remain **hybrid AMG** solves: the coarse hierarchy and vector operations
stay in host RAM at all three fractions. A fully resident solve also moves
those operations onto the GPU, so the preceding fully resident results are
a different execution mode, not simply the 100% endpoint of this sweep.
Each fine product transfers the full input vector plus the GPU output
prefix: `8*(n+r)` bytes. Increasing resident rows reduces CPU matrix work
but increases the output-vector transfer; 90% residency does not mean
only 10% of vector traffic crosses PCIe.
CUDA mixed uses the same fp64 hybrid path here; its resident fp32
preconditioner is not used. Physical VRAM was not exhausted.

[Raw sweep and retained controls](benchmarks/pcg/titan-v-row-residency-20261102.json).
Reproduce with:

```sh
OMP_NUM_THREADS=8 python3 tools/run_pcg_residency_benchmarks.py \
  --binary build/pcg_benchmark --matrices build/pcg-large-cases \
  --output build/pcg-large-results/row-residency --repeats 3 --timeout 300
```

`ninja cppcheck` completes with no errors or new findings. The same 19
existing local unused-function warnings remain. No solver code or
mathematical acceptance criteria changed for this sweep.

## Earlier AMG changes and profiles

- Large systems use diagonally equilibrated smoothed-aggregation multigrid
  instead of IC0/Jacobi alone. The AMGCL 1.5.0 dependency is pinned by a Meson
  wrap. The original fp64 matrix and compensated refinement decide acceptance.
- Resident CUDA/Vulkan keep the hierarchy and Krylov operations on the GPU.
  Only the small coarse problem (at most 256 equations) is factored on the CPU;
  there is no full-system CPU direct fallback in these replay measurements.
- Spill mode retains a fixed fine-matrix row prefix on the GPU. Host rows run
  concurrently with the device product; the coarse hierarchy and vectors are
  in host RAM. Each V-cycle needs multiple fine products, so vectors can cross
  PCIe several times per PCG iteration. Matrix tiles are not repeatedly copied.
- Vulkan SpMV now assigns eight lanes per CSR row and sixteen rows per
  workgroup, with three reduction stages instead of six. This avoids wasting
  a whole 64-thread group on the motor's approximately eight entries per row.
- CUDA uses full CSR for these products, eliminating the separate transpose
  SpMV required by symmetric-lower storage. Small resident systems retain the
  previously optimized colored IC0 path.
- The optional energy-progress cutoff is disabled by default. It stopped some
  slowly converging magnetic solves prematurely; final acceptance is unchanged.

Before AMG, the plate CUDA profile spent **56.4% of GPU kernel time in triangular
solves**, **21.8% in transpose SpMV**, and **12.5% in normal SpMV**. A sampled
triangular kernel had **12.5% active-warp occupancy** and **71.16% of sustained
DRAM throughput** (80 blocks, 256 threads/block). These are kernel-level
measurements, not whole-program utilization. They support the utilization and
memory diagnosis; the TITAN V's fp64 arithmetic rate was not the missing fix.
The large `cudaMemcpy` API time in that resident trace largely waited for queued
GPU work and should not be read as PCIe transfer time.

In the updated plate profile, **87.5% of kernel time is SpMV**. A sampled
fine-grid SpMV reaches **61.15% active-warp occupancy** and **90.33% of sustained
DRAM throughput**. This is a different kernel from the original triangular
solve; the comparison reflects the algorithm change, not a tuning of that
triangular kernel.

On the profiled motor spill solve, actual CUDA host/device copies total
**0.329 s** for 4.085 GB, while the first-to-last-copy interval is **2.881 s**.
Vulkan GPU batches occupy 0.366 s in a 2.813 s activity interval, versus 0.368 s
in 0.623 s when resident. Thus the remaining spill gap is mostly outside GPU
work and actual PCIe-copy duration, in the host-side hierarchy/products and
coordination. These are instrumented trace intervals, not benchmark solve times
or whole-program occupancy. The retained Nsight summaries and counters support
these measurements.

The initial plate resident CUDA implementation took 23.384 s; the updated
median is 6.479 s. Vulkan and spill paths that previously failed or timed out
now converge. Thermal is a tradeoff: the previous specialized IC0/Jacobi
resident times were 0.144 s CUDA and 0.269 s Vulkan, faster than AMG's current
0.365/0.505 s. Thermal remains much faster than CPU direct. A cheap adaptive
preconditioner selection policy is still needed to preserve that advantage.

On the forced-spill plate, setup is about 1.3 s but iterations/refinement take
about 66 s over 872 iterations. Better data placement alone has not closed the
gap: iteration count and repeated host-memory products dominate. Bounded
experiments with a larger exact coarse solve, graph reordering, stronger host
smoothing, physical rigid-body modes and alternate interpolation settings did
not produce a sufficiently faster accurate replacement and were not integrated.

## Earlier complete-deck validation

All **18 GPU analyses pass**, including parsing, assembly, all nonlinear solves,
postprocessing and output. Times are single end-to-end runs in seconds with a
240-second cap; CPU direct numbers are the retained unchanged baseline.

| Deck | CPU direct | CUDA fp64 resident | CUDA fp64 spill | CUDA mixed resident | CUDA mixed spill | Vulkan resident | Vulkan spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| Thermal | 10.02 | 3.90 | 5.66 | 3.83 | 5.42 | 4.07 | 6.02 |
| Full nonlinear motor | 16.72 | 18.19 | 50.36 | 16.91 | 39.70 | 20.50 | 52.82 |
| Plate | 26.71 | 21.70 | 81.25 | 21.22 | 82.03 | 23.71 | 85.77 |

These use existing CLI defaults: CUDA fp64 and Vulkan target 1e-8; experimental
mixed CUDA retains its 1e-6 default. The replay table above requests 1e-8 for
every backend, including mixed. No CLI default was loosened in this change.

All six motor outputs have identical geometry and element order to CPU direct,
converge in **11 Newton iterations**, and use no B–H extrapolation. Independent
comparisons of magnetic potential, B/H fields, probes, circuit linkages and
inductances have maximum relative difference **4.88e-12** across all datasets
and modes. Torque differs by at most **3.38e-14** relative (CPU: 4.89736534 Nm).
All-NaN open-winding quantities remain unavailable in both outputs and are not
counted as accuracy measurements. Exact dataset comparisons and file hashes are
in [the full motor verification](benchmarks/pcg/titan-v-amg-motor-accuracy-20261102.json).

## Reproduction and retained results

[Deck counts, generation commands and hashes](benchmarks/pcg/large-decks-20261102.json),
[earlier samples, failures and diagnostic logs](benchmarks/pcg/titan-v-large-20261102.json),
and [earlier motor comparisons](benchmarks/pcg/titan-v-large-motor-accuracy-20261102.json)
are retained in the repository. Generated decks, matrices and full outputs are
under `/tmp/vibestran-pcg-20261102/large` on bob-desktop.local. The unchanged plate
input is under that checkout's `measurements/plate262k.bdf`. The separate
baseline checkout is `/tmp/vibestran-pcg-baseline-20261102`.

Generate the motor using the repository's Gmsh environment (measured with
Gmsh 4.15.2), then prepare the static operating point and thermal deck:

```sh
.venv/bin/python Magnetic_Validation/mesh_study.py decks build/pcg-large-cases --mesh tri:0.225
python3 tools/prepare_large_pcg_cases.py \
  --motor build/pcg-large-cases/tri_0.225/motor.bdf --output build/pcg-large-cases
```

Build in `build` with `ninja` and `ninja pcg_benchmark`. On the GPU host, with
the CUDA/cuDSS library directories on `LD_LIBRARY_PATH`, capture physical
systems and run the bounded replay:

```sh
export OMP_NUM_THREADS=8
for name in thermal227k motor218k; do
  timeout 120s taskset -c 0-7 build/pcg_benchmark \
    --capture-deck build/pcg-large-cases/$name.bdf \
    --capture-prefix build/pcg-large-cases/$name
done
timeout 180s taskset -c 0-7 build/vibestran --backend=cpu \
  --dump-system=build/pcg-large-cases/plate plate262k.bdf build/pcg-large-cases/plate.f06
python3 tools/run_large_pcg_benchmarks.py --binary build/pcg_benchmark \
  --matrices build/pcg-large-cases --output build/pcg-large-results/screen \
  --backends cpu,cuda-pcg,cuda-pcg-mixed,vulkan --timeout 120
python3 tools/run_large_pcg_benchmarks.py --binary build/pcg_benchmark \
  --matrices build/pcg-large-cases --output build/pcg-large-results/repeated \
  --successful-from build/pcg-large-results/screen/updated-results.json \
  --backends cuda-pcg,cuda-pcg-mixed,vulkan --only resident --repeats 3 --timeout 120
```

The screening script returns nonzero if any configuration fails, while retaining
every result and continuing through the suite. For the baseline executable use
`--version baseline --backends cuda-pcg,vulkan`; its benchmark adapter must read
the same physical RHS/reference and map the original Vulkan headroom setting to
the requested budget, as in the preserved baseline checkout.

For complete-deck runs, use `build/vibestran --backend=BACKEND deck.bdf result.f06`
under the same affinity and timeout. Set `VIBESTRAN_PCG_DEVICE_MEMORY_MIB` to the
case budget for spill runs, and unset it for resident runs.


The preceding host-based row-residency samples are in
[the sweep record](benchmarks/pcg/titan-v-row-residency-20261102.json).
The preceding resident measurements and profiler evidence are in
[the AMG benchmark record](benchmarks/pcg/titan-v-amg-20261102.json).
The earlier pre-AMG results are retained in
[the original large-deck record](benchmarks/pcg/titan-v-large-20261102.json).

For the complete-deck sequence (240-second limit per process):

```sh
OMP_NUM_THREADS=8 python3 tools/run_large_pcg_analyses.py \
  --binary build/vibestran --decks build/pcg-large-cases \
  --plate plate262k.bdf --output build/pcg-large-results/analyses
```

The complete local suite passes. On the TITAN V host, unit, integration,
CLI/end-to-end, CUDA and Vulkan tests pass; the Python VTK output test is skipped
because its optional module is unavailable. New tests cover material-interface
flux, equation units differing by twelve orders of magnitude, both symmetric
storage formats, non-workgroup-multiple sizes, zero loads, and device budgets
smaller than a vector. The existing diagnostics test was updated because
Vulkan's automatic large-system path now selects fp64 AMG; no mathematical
acceptance criterion was relaxed.

Cppcheck reports existing warnings. Its new AMG backend-hook unused-function
findings are false positives: the third-party template library invokes those
hooks, exercised by the mathematical tests. They have narrow inline
suppressions; a similar suppression marks the Vulkan coarse output as mutable
because a recorded GPU copy writes it. Final `ninja cppcheck` completes on both hosts with **no errors and no new
unsuppressed findings** (19 existing local unused-function warnings and 59
unique existing remote warnings).
