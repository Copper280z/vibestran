# CUDA and Vulkan PCG: resident and partitioned solves

The subsequent AMG implementation and >=200k-GRID results are documented in
[PCG_LARGE_DECKS.md](PCG_LARGE_DECKS.md). The measurements below retain the
earlier IC0/Jacobi implementation for historical comparison.

Measured on 2026-11-02 against commit `ac147f6`, on bob-desktop.local:
Ryzen 7 5700, eight physical CPU cores (`taskset -c 0-7`), 32 GiB RAM,
TITAN V 12 GiB, driver 580.178.04, CUDA 12.8, Release Meson build,
SuiteSparse CHOLMOD CPU direct backend. cuDSS implementation and configuration
are unchanged.

Follow-up tests on [physical decks with more than 200,000 GRID nodes](PCG_LARGE_DECKS.md)
include the full radial motor and a 1.58-million-equation plate. Initial runs
exposed accuracy failures and timeouts on the larger motor and plate; the later
AMG and device-owned spill implementations documented there resolve them.
The small-deck measurements below describe the earlier implementation.

[Raw samples and accuracy results](benchmarks/pcg/titan-v-20261102.json) and
[Nsight Systems excerpts](benchmarks/pcg/titan-v-20261102-profiles.txt) accompany
this report. Times below are medians of three solves after one warmup. They
include solver setup, preconditioner construction, host/device transfers,
iteration and compensated refinement. Assembly, backend/context creation and
independent solution verification are excluded. Vulkan's small-system direct
fallback is disabled in the benchmark so these measurements exercise GPU PCG.
Desktop GPU activity and the small sample count limit precision.

## Results

On the 262,144-equation 3D diffusion system, wall seconds were:

| Backend and mode | Before | After |
|---|---:|---:|
| CPU direct | — | 5.369 |
| CPU PCG | 1.821 | 1.821 |
| CUDA fp64, resident | 0.151 | 0.101 |
| CUDA mixed, resident | — | 0.097 |
| CUDA fp64, 8 MiB budget | unavailable | 0.840 |
| CUDA mixed, 8 MiB budget | unavailable | 0.620 |
| Vulkan, resident | 1.360 | 0.302 |
| Vulkan, 8 MiB budget | 10.562 | 0.598 |

Thus Vulkan improved 4.5× resident and 17.7× with the reduced budget. CUDA
fp64 improved 1.5× resident. Both new budgeted implementations beat CPU PCG
on this case, although partitioning remains slower than resident GPU solving.
The original CUDA PCG implementation allocated the entire device system; it
had no explicit budgeted spill implementation, so there is no equivalent
baseline CUDA spill timing.

Actual thermal and magnetic matrices were captured through normal FE assembly,
including the physical load vectors. Every replay is checked against its CPU
direct solution. Times in this table are **milliseconds**; arrows show
before → after:

| System | Equations | CPU direct | CUDA fp64 resident | Vulkan resident | Vulkan budgeted | CUDA fp64 budgeted |
|---|---:|---:|---:|---:|---:|---:|
| CHEXA8 heat conduction, 100:1 conductivity interface | 14,375 | 135.49 | 36.32 → 9.94 | 268.40 → 31.07 | 925.52 → 17.90 | 6.65 |
| Planar mixed triangle/quad winding | 3,969 | 4.62 | 55.29 → 26.34 | 714.34 → 49.98 | 1,626.54 → 24.92 | 17.48 |
| Nonlinear planar motor, first tangent | 3,969 | 4.67 | 49.61 → 7.22 | 936.85 → 61.64 | 2,162.27 → 31.16 | 12.18 |
| Same motor, late tangent | 3,969 | 4.59 | 46.20 → 7.23 | 719.44 → 49.57 | 1,940.24 → 22.84 | 18.34 |
| 3D tetrahedral coil | 5,401 | 11.17 | 139.89 → 84.09 | 2,520.26 → 132.61 | 9,592.92 → 52.61 | 40.65 |

The thermal budget is 1 MiB; magnetic budgets are 256 KiB. These deliberately
small budgets force CPU/GPU partitioning on modest matrices. Complete magnetic
CPU-direct CLI analyses took 0.16 s (winding), 0.17 s (motor), and 0.18 s
(coil), comfortably below two seconds. The motor uses a synthetic B–H curve,
50 A excitation and the model's existing vacuum-slope extrapolation at high
field. First and late Newton tangents exercise its changing material stiffness.
Small magnetic systems still favor CPU direct solves.

CUDA mixed resident also improved on these physical systems: thermal
39.48 → 13.10 ms, winding 72.83 → 33.80 ms, first motor tangent
108.48 → 20.05 ms, late tangent 69.12 → 13.05 ms, coil
154.64 → 101.06 ms. Mixed precision remains experimental and may fail the
accuracy target on more ill-conditioned matrices.

Additional coverage includes diffusion with 10,000:1 coefficient contrast and
an 18,150-equation CHEXA8 bending matrix. These use a manufactured solution
with multiple frequencies and irregular excitation. For the bending matrix,
Vulkan resident improved 2.834 → 0.192 s and its 4 MiB path improved
15.393 → 0.161 s. Original CUDA fp64 failed the unchanged accuracy target
on this case; the updated solver succeeds in 0.147 s. A failed solve is not
counted as a performance comparison.

All successful recorded solutions have relative infinity-norm forward error
below 2e-11. The solver's requested estimated error remains 1e-8; neither
convergence tolerances nor refinement checks were relaxed. Accuracy against a
direct solution is a comparison, while the manufactured cases also have a
known independent exact solution.

## What the profiles showed

The old resident Vulkan implementation repeatedly allocated staging buffers
and created fences inside PCG. A 32³ diffusion trace contained 939 allocation
and free pairs, plus 3,187 queue submissions and fence create/destroy pairs.
Allocation/free consumed 44% of recorded Vulkan API time and fence
create/destroy another 26%. The new trace has 24 allocation/free pairs,
586 submissions, and 13 fence create/destroy pairs, including setup and
teardown. The iteration now records two command batches, uses persistent
mapped reduction readback, and reuses matrix storage, descriptors and command
resources across refinement corrections.

CUDA's main resident cost was IC0's triangular dependency chains. On 64³
diffusion, triangular solves consumed about 75% of GPU kernel time, averaging
373 microseconds each. Symmetric graph coloring groups independent equations,
reducing the average to 56 microseconds and the share to 39%. The iteration
count increases from 121 to 182, but total solve time falls. Coloring changes
the incomplete preconditioner, not the original operator; a failed refinement
attempt is retried with the original ordering. At most 32 colors are accepted.
An environment switch allows direct comparison with natural ordering.

CUDA also fuses the solution/residual update and polls the device convergence
state once per eight iterations. Terminal state is sticky, so extra queued
iterations cannot change a converged solution. Diagnostics count actual
iterations, not the padded batch. Nsight timings include that normal batching;
the optional per-phase CUDA instrumentation polls each iteration and should
not be used for headline timing comparisons.

The old Vulkan tiled path uploaded matrix bands on every iteration and
submitted/waited separately for multiple operations per band. The replacement
keeps a fixed CSR row prefix on the GPU and computes the suffix on CPU while
GPU work is pending. Persistent pinned/coherent buffers exchange only the
search vector and GPU output rows. CPU updates and a full-system Jacobi
preconditioner preserve couplings across the partition. CPU arithmetic uses
cached host memory; doing it directly through uncached mapped staging memory
was itself a severe bottleneck during development.

At 8 MiB on the 64³ case, Vulkan retains 115,081 rows on GPU and computes
147,063 on CPU. It uploads about 6.35 MB of matrix indices/values once and
transfers 1.51 MB of vectors per iteration. CUDA fp64 uploads about 5.44 MB
of matrix indices/values once and transfers 2.62 MB per iteration; Nsight
records 941 MB H→D and 235 MB D→H over 446 iterations, including setup.
Transfers are still significant. Much of their latency overlaps CPU work;
`cudaMemcpy` API duration must not be mistaken for pure PCIe bandwidth cost.

Automatic path selection uses CUDA free memory or the Vulkan driver's
[current memory budget](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMemoryBudgetPropertiesEXT.html),
with headroom. Devices without that Vulkan extension use total heap size.
Vulkan descriptor range and dispatch limits also constrain the resident path.

## Reproduction

Build the optional harness with the project's usual build directory:

```sh
meson setup build --buildtype=release -Dhdf5=enabled
cd build
ninja
ninja pcg_benchmark
cd ..
python3 tools/generate_pcg_cases.py build/pcg-cases
```

On Linux, time the direct magnetic analyses first. The generated examples
above meet the two-second limit on this host; recheck after changing their
mesh size or on another machine:

```sh
for name in magnetic_winding magnetic_motor magnetic_coil; do
  /usr/bin/time -f 'wall_seconds=%e' timeout 10s taskset -c 0-7 \
    build/vibestran --backend=cpu build/pcg-cases/$name.bdf \
    build/pcg-cases/$name.f06
done
for name in thermal_interface magnetic_winding magnetic_motor magnetic_coil; do
  timeout 25s taskset -c 0-7 build/pcg_benchmark \
    --capture-deck build/pcg-cases/$name.bdf \
    --capture-prefix build/pcg-cases/$name
done
python3 tools/run_pcg_benchmarks.py --binary build/pcg_benchmark \
  --matrices build/pcg-cases --output build/pcg-results --cpu-affinity 0-7
```

Each suite invocation allows 60 seconds for a warmup plus three timed solves,
records failures/timeouts, and continues to the next case. In the initial
30-second batch run the old coil spill implementation completed only two
samples; a bounded 60-second replacement supplied all three reported samples.
The capture tool writes every magnetic Newton/response system; the suite
selects the first tangent and motor tangent 7. It does not time complete
nonlinear PCG analyses or change their production output format.

```sh
taskset -c 0-7 build/pcg_benchmark --backend cuda-pcg --side 64
taskset -c 0-7 build/pcg_benchmark --backend vulkan --side 64 --budget-mib 8
taskset -c 0-7 build/pcg_benchmark --backend cuda-pcg --side 32 --contrast 10000
VIBESTRAN_CUDA_PCG_NATURAL_ORDER=1 taskset -c 0-7 \
  build/pcg_benchmark --backend cuda-pcg --side 64
nsys profile --sample=none --cpuctxsw=none --trace=cuda,vulkan,osrt \
  -o build/pcg-profile taskset -c 0-7 build/pcg_benchmark \
  --backend cuda-pcg --side 64 --budget-mib 8 --warmups 0 --repeats 1 --profile
```

The preserved baseline executable and complete `.nsys-rep` traces are in
`/tmp/vibestran-pcg-20261102` on bob-desktop.local. Its separate baseline
source/build is `/tmp/vibestran-pcg-baseline-20261102`; the solver sources
were hash-checked against `ac147f6`. Only the benchmark adapter was updated
there to read physical loads and map the old Vulkan headroom setting to the
same budget. The baseline CUDA path has no budget parameter.

## Validation and limits

The complete local Meson suite passed (7 targets). The GPU-host suite passed
6 targets, including CUDA and Vulkan; the optional Python VTKHDF test skipped
because its Python dependencies are absent there and passed locally. New
tests check cross-partition material-interface fluxes, true solution accuracy
in both precisions, zero loads, original/permuted equation mapping, budget
selection, and descriptor limits. `ninja cppcheck` completed locally and on
the GPU host with no errors or new diagnostics. The GPU host's 101 existing
static-analysis diagnostics are identical to the baseline, including
existing unused-function reports; no unrelated suppressions were added.
An earlier local cppcheck pass also reported existing false positives such as
`Model::resolve_coordinates` (called by the BDF parser) and
`TangentSolver::factor_is_current` (covered by tangent-solver tests). The
large-deck follow-up cppcheck pass again reported only existing unused-function
false positives and no errors.

The reduced-budget runs exercise real host/device transfers and partitioned
execution, but the tested matrices do **not** physically exceed 12 GiB.
Exhausted-VRAM behavior under competing applications, other GPU architectures,
and systems larger than host RAM are not validated by these measurements.
The budget is a working-storage estimate, not a hard cap on driver/library
allocations. Host CSR storage, precision conversions and working vectors must
fit in RAM. Disk-backed assembly/solving would require a separate design.
Jacobi convergence and host vector bandwidth remain limits in partitioned
mode, and IC0 is still rebuilt for each new solve. These results should not
be extrapolated to every structural or magnetic problem.
