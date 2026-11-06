# PCG tests on decks with more than 200000 grids

CUDA fp64, CUDA mixed and Vulkan pass the large physical matrix replays with
GPU-owned iterative state. After the AMG engine rewrite below, the
1.58-million-equation plate solves in **4.29 s CUDA / 4.82 s Vulkan** resident
and **5.38 / 5.72 s** with 90% of its fine-matrix rows on the GPU, versus
**6.44 / 8.28 s** and **7.95 / 9.41 s** before it. Independent solution error
remains below **1.86e-9** at the unchanged **1e-8** target. The complete
nonlinear motor now runs faster than CPU direct on every GPU backend.

The CPU backend takes about a minute on the largest decks, which is 1.4-1.5x the TITAN V
over a complete deck. See [CPU PCG](#cpu-pcg).

Measured on 2026-11-02 on bob-desktop.local: Ryzen 7 5700, eight physical cores
(`taskset -c 0-7`, `OMP_NUM_THREADS=8`), 32 GiB RAM, TITAN V 12 GiB,
driver 580.178.04, CUDA 12.8, Release Meson build. cuDSS is unchanged.

## Structural AMG and the model-class suite

Measured on 2026-11-03 on the same machine. The timings in the rest of this
file predate this section and use the scalar configuration,
`--amg-modes=constant`, the default when they were taken. `rigid` became the
structural default on 2026-11-04 and `drilling` on 2026-11-05: see "The default"
below.

Tuning on one deck overfits, so `tools/amg_model_suite.py` judges the AMG setup
by how iterations grow with refinement in each class of model, iterations ~
h^-p. Raw results are in `benchmarks/pcg/amg-model-suite-20261103.json`.

Structural solves (SOL 101, 50000 equations or more, GPU PCG backends) use
`drilling`, the rigid-body modes plus two columns on CQUAD4 grids, unless
`--amg-modes=rigid` or `--amg-modes=constant` (`PARAM,AMGMODES,...`) selects otherwise.
Thermal and magnetic solves
always use the scalar hierarchy, with unchanged iteration counts. Three changes
were each necessary; none pays off alone:

- **A stable smoother.** SPAI-0 weights a_ii / sum a_ij^2 suit a scalar Laplacian,
  but on a system many small couplings keep the denominator small while
  lambda_max grows. On a hexahedral bar m lambda_max reached 2.9, above the limit
  of 2, so the smoother amplified high frequencies. The V-cycle was then indefinite
  (eigenvalues of M A down to -0.67), and CG needs it positive definite. The
  replacement is `l1_jacobi_weights()`, m_i = w / g_i with g_i = sum_j |a_ij|; see
  the certified bound under "Validity and the Priority 0 baseline". Engines needed
  no change: they read a per-level diagonal.
- **Rigid-body modes on aggregates of whole grid points.** Modes come from the
  reduced index and grid positions, in basic axes (K is assembled in basic axes
  whatever the grid's CD), and K * mode vanishes on free-free models of every
  element family and RBE tested. AMGCL's block aggregation needs uniform blocks,
  which a partial SPC breaks, so `ModeAggregation` groups by the model's grids.
- **P smoothed with the whole matrix.** AMGCL's filter lumps weak couplings into
  the diagonal. That keeps the constant vector but not a rotation, because the
  lumped row gains sum a_ij (B_j - B_i), which grows with the lever arm. On the
  bar this alone took 79 iterations to 33.

CUDA fp64, resident. Each cell lists the mesh levels; "fail" means the solve ended
above the accuracy target. Constant ran only the two coarsest levels, within
120 s each. p uses the two finest levels of the rigid run.

| Class | Equations | Constant iterations | Rigid iterations | p | Rigid iterate s | Rigid op. complexity |
|---|---:|---:|---:|---:|---:|---:|
| solid-hex | 66k / 227k / 807k | 100000 / fail | 39 / 40 / 36 | -0.25 | 0.04 / 0.1 / 0.5 | 2.4-2.5 |
| solid-tet10 | 55k / 180k / 694k | 6155 / 22693 | 63 / 60 / 57 | -0.11 | 0.1 / 0.2 / 0.6 | 2.5-2.8 |
| solid-contrast (10^6:1) | 66k / 227k / 807k | fail / fail | 70 / 71 / 98 | 0.76 | 0.1 / 0.2 / 0.9 | 2.8-2.9 |
| solid-thin (flat hexahedra) | 58k / 232k / 924k | fail / fail | 16873 / 4856 / 2820 | -0.79 | 6.8 / 6.9 / 17 | 1.9-2.2 |
| shell-thin (t/L 0.001) | 59k / 238k / 955k | 29598 / 58023 | 723 / 612 / 567 | -0.11 | 0.2 / 0.5 / 2.0 | 2.7 |
| shell-thick (t/L 0.1) | 59k / 238k / 955k | 145 / 243 | 30 / 31 / 35 | 0.17 | 0.01 / 0.04 / 0.2 | 2.8 |
| shell-foundation | 61k / 242k / 965k | 1969 / 7077 | 104 / 101 / 131 | 0.38 | 0.03 / 0.1 / 0.5 | 2.7 |
| shell-cylinder-cd | 62k / 249k / 995k | 4258 / 5269 | 580 / 833 / 1451 | 0.80 | 0.2 / 1.1 / 7.2 | 1.8 |
| stiffened-panel | 63k / 251k / 999k | 100000 / fail | 309 / 644 / 1412 | 1.14 | 0.1 / 0.7 / 5.4 | 3.0-3.7 |
| rbe-assembly | 55k / 319k / 939k | fail / fail | 1409 / 2616 / 4260 | 0.90 | 0.5 / 5.6 / 29 | 2.2-2.6 |
| **shell-membrane (t/L 10)** | 61k / 241k / 962k | **202 / 353** | **634 / 1213 / 2167** | 0.84 | 0.2 / 1.1 / 6.7 | 2.8 |
| **cubic-lattice** | 61k / 229k / 876k | **80 / 97** | **133 / 147 / 160** | 0.19 | 0.04 / 0.1 / 0.5 | 7.5-8.0 |

This table predates the certified smoother bound, and its p column predates
refinement-parameter exponents. The current numbers and the stricter comparison with
a direct solve are under "Validity and the Priority 0 baseline".

**The last two rows regress.** Plate262k is `shell-membrane`: 902 iterations in
3.43 s with constant, 2556 in 12.27 s with rigid. Its ν = 0 helps constant, because
the weak-coupling filter that harms rotations helps nearly decoupled fields: removing
it (threshold 0) takes constant's 120 / 199 iterations to 459 / 833 at n = 48 / 96,
worse than rigid's 331 / 625. Rigid's own iterations have another cause, measured on
2026-11-05 at n = 400: t/L does not matter, freezing bending leaves 1658 iterations,
and freezing the drilling rotation leaves 34 with the same answer. See "Drilling
modes" below, which fixes this class. `PARAM,AMGMODES,CONSTANT` also remains.

Realistic shell assemblies, the WebMesh rover (CQUAD4, RBE2, RBE3, CBUSH, CONM2;
two subcases, per subcase, iterate time from the solver log):

| Deck | Equations | Before: iterations | Rigid: iterations | Before: iterate s | Rigid: iterate s | Rigid complexity |
|---|---:|---:|---:|---:|---:|---:|
| Complete rover, CUDA | 60,606 | 6867 / 9041 | 331 / 332 | 1.82 / 2.39 | 0.119 / 0.119 | 2.5 |
| Complete rover, Vulkan | 60,606 | 6891 / 7432 | 331 / 332 | 2.77 / 3.03 | 0.191 / 0.172 | 2.5 |
| Four-wheel rover, CUDA | 68,730 | 2933 / 2930 | 398 / 411 | 0.80 / 0.80 | 0.150 / 0.153 | 2.6 |
| Four-wheel rover, Vulkan | 68,730 | 2933 / 2930 | 398 / 411 | 1.14 / 1.13 | 0.230 / 0.232 | 2.6 |

The previous binary and `--amg-modes=constant` give identical CUDA iteration
counts on both decks. Rigid displacements of both subcases match CHOLMOD within
the F06's printed precision, 7.9e-9 or better.

**Device memory.** The hierarchy holds more, so fewer equations stay resident.
At 8.49 GB of usable Titan V memory (planned bytes per equation at about 1M
equations; the plan's capacity is a linear extrapolation):

| Class | Constant B/DOF | Rigid B/DOF | Rigid resident capacity |
|---|---:|---:|---:|
| Shells | 502 | 885-907 | about 9.5M DOF (17M with constant) |
| Cubic lattice | 495 | 1109 | about 7.7M DOF (17M with constant) |
| RBE assembly | not measured | 2032 | about 4.2M DOF |
| Solid hexahedra | not measured | 2416 | about 3.5M DOF |
| CTETRA10 | not measured | 2840 | about 3.0M DOF |

The 15.4M-equation stress deck above would no longer be fully resident with
rigid modes: it would spill by default, and stays resident with `constant`.

**Scalar hierarchies have a separate problem, not changed here.** `thermal-hex`
grows at p = 0.97 with operator complexity 1.04-1.09: 122 / 200 / 308 iterations
at 66k / 266k / 1.0M. On a uniform Q1 hexahedral Laplacian the off-diagonal
ratios are 0, 1/16 and 1/32, below `eps_strong = 0.08`, so nearly every coupling
counts as weak and the coarse levels barely exist. A threshold of 0.04 took 65k
equations from 126 iterations to 22 on the host cycle. The membrane shows the
threshold is class-dependent, so it needs the suite's magnetic classes before it
changes: `magnetic-motor` 79 / 112 / 147 and `magnetic-coil`
1991 / 2601 / 3273 per solve are this configuration's baseline.

Reproduce, on the benchmark machine:

```bash
python3 tools/amg_model_suite.py run --binary build/vibestran --work /tmp/amg-suite --results r.jsonl --modes rigid,constant --direct --require-solved rigid
python3 tools/amg_model_suite.py report r.jsonl
```

## Validity and the Priority 0 baseline

Measured on 2026-11-03 on the Titan V in fp64. The script and manifest (commit, compiler
flags and dependency versions, and the hashes of the binary and the working-tree
patch) are in `benchmarks/pcg/amg-baseline-20261103/`. Per-run results with deck and
matrix hashes, commands and budgets, the patch, the logs, the full-precision solutions
and the reference K and F stay in `~/pcg-perf/baseline-20261103/` on the benchmark
machine: the repository holds no binaries or test data. Timings are single runs, and
failures and timeouts are recorded as outcomes.

**A certified smoother bound, on every hierarchy.** A V(1,1) cycle with R = P',
A_c = P'A P of full rank, rho(M A) < 2 on every level and an SPD coarse solve is
SPD, with the eigenvalues of B A in (0, 1]. For SPD A and g_i = sum_j |a_ij|,
rho(G^-1 A) <= 1, so any diagonal M with m_i <= w / g_i has rho(M A) <= w. The bound
`kAmgSmootherBound` is w = 1.9, and the weights respect it for any matrix:

- Rigid: w = min(1.9, (4/3) / rho), with rho from the power iterations. They now
  only tune w, so a wrong estimate cannot break stability. The 2 x 2 matrix on which
  they report 0.5 for a true 1 would otherwise give rho(M A) = 2.67.
- Constant: SPAI-0, limited to 1.9 / g_i. On a diagonally dominant row SPAI-0 is
  already below that, so scalar fields barely change.

Other rules were measured and not adopted. w = 4/3 (rho = 1, no estimate) costs about
50% more iterations: shell-thin 723 to 1133, solid-hex 39 to 60. A limit of 1.98 gains
almost nothing over 1.9. A tighter certified estimate from the squared iteration
matrix recovers iterations but costs setup: on the largest CTETRA10 deck it took
67 iterations to 59 and setup from 5.1 s to 11.2 s.

Iterations per solve against the commit before the bound: the coarsest two levels
before, and all three levels now. "fail" means the iteration limit was reached.

| Class | Constant before | Constant now | Rigid before | Rigid now |
|---|---:|---:|---:|---:|
| solid-hex | 100000 / fail | 407 / 617 / 949 | 39 / 40 | 41 / 41 / 41 |
| solid-tet10 | 6155 / 22693 | 686 / 1019 / 2030 | 63 / 60 | 69 / 68 / 67 |
| solid-contrast | fail / fail | 596 / 889 / 1624 | 70 / 71 | 70 / 71 / 98 |
| solid-thin | fail / fail | fail / fail / fail | 16873 / 4856 | 17138 / 6106 / 3457 |
| shell-thin | 29598 / 58023 | 39413 / 69161 / timeout | 723 / 612 | 722 / 612 / 567 |
| shell-thick | 145 / 243 | 147 / 252 / 521 | 30 / 31 | 30 / 31 / 35 |
| shell-membrane | 202 / 353 | 206 / 358 / 710 | 634 / 1213 | 634 / 1213 / 2167 |
| shell-cylinder-cd | 4258 / 5269 | 3751 / 5764 / 11460 | 580 / 833 | 581 / 1159 / 1452 |
| shell-foundation | 1969 / 7077 | 1982 / 7671 / 6380 | 104 / 101 | 106 / 100 / 130 |
| stiffened-panel | 100000 / fail | 100000 / fail / fail | 309 / 644 | 309 / 645 / 1413 |
| cubic-lattice | 80 / 97 | 79 / 97 / 115 | 133 / 147 | 134 / 148 / 160 |
| rbe-assembly | fail / fail | 19343 / 12880 / 29154 | 1409 / 2616 | 1413 / 2614 / 4234 |
| Complete rover (mean of subcases) | 7954 | 3749 | 332 | 329 |
| Four-wheel rover (mean of subcases) | 2932 | 3465 | 404 | 404 |
| thermal-hex | 122 / 200 | 117 / 197 / 283 | | |
| magnetic-motor (mean per solve) | 79 / 112 | 79 / 112 / 147 | | |
| magnetic-coil (per solve) | 1991 / 2601 | 1990 / 2597 / 3253 | | |

The bound helps the scalar hierarchy far more than it costs: solids go from failing
or tens of thousands of iterations to hundreds or low thousands, at no memory cost.
It costs iterations where SPAI-0 was above the limit yet happened to work: shell-thin
+19-33% and the four-wheel rover +18%. On the rigid hierarchy the cost is up to 13% on
CTETRA10 and 26% on solid-thin, and 39% at one cylinder level, whose neighbours do
not move (580 to 581 coarser, and 1451 to 1452 finer).

**Why `constant` is not robust.** It needs less memory and wins some classes, but
it fails solid-thin at
every level and stiffened-panel above the coarsest, where it needs 100000 iterations,
and shell-thin at 955k equations exceeded the 150 s limit. Total solver time (setup
plus iterations) and planned device memory at the finest level of each class:

| Class | Equations | Constant s | Rigid s | Constant MiB | Rigid MiB |
|---|---:|---:|---:|---:|---:|
| solid-hex | 807k | 5.5 | 4.1 | 941 | 1870 |
| solid-tet10 | 694k | 9.9 | 5.9 | 865 | 1881 |
| solid-contrast | 807k | 8.5 | 6.2 | 919 | 2085 |
| solid-thin | 924k | fail | 23.6 | 832 | 1541 |
| shell-thin | 955k | timeout | 3.0 | - | 826 |
| shell-thick | 955k | 1.7 | 1.3 | 445 | 820 |
| shell-membrane | 962k | 2.1 | 7.8 | 460 | 813 |
| shell-membrane (plate262k) | 1.58M | 4.1 | 14.0 | 715 | 1292 |
| shell-cylinder-cd | 995k | 49 | 9.0 | 873 | 1178 |
| shell-foundation | 965k | 16 | 1.7 | 453 | 833 |
| stiffened-panel | 999k | fail | 6.7 | 504 | 957 |
| cubic-lattice | 876k | 0.7 | 1.9 | 414 | 927 |
| rbe-assembly | 939k | 133 | 33 | 923 | 1820 |
| Complete rover (both subcases) | 61k | 2.0 | 0.44 | 34 | 62 |
| Four-wheel rover (both subcases) | 69k | 2.0 | 0.55 | 39 | 70 |

Rigid solved all 39 structural runs; constant solved 33, failed 5 and timed out once.
Rigid is at worst 3.6 times slower than constant here and needs about twice the
memory. These are resident times. The equal-budget section below compares the two under
memory pressure.

**The applied cycle is tested, not only its parts.** `test_amg_vcycle.cpp` forms B
explicitly on small assembled systems, with and without modes, in basic, shifted,
rotated and millimetre coordinates: a partly constrained solid, a 10^6:1 material
interface, a partly constrained shell, offset stiffeners, a solid and shell coupled
by RBE2 and RBE3, and lines of bars, whose modes lose rank on an aggregate. It checks
R = P', the Galerkin product, SPD coarse operators, rho(M A) <= 1.9 on each level, B
symmetric, and the eigenvalues of B A in (0, 1]. Unbounded SPAI-0 fails these checks.
Larger systems use bilinear-form probes. `VIBESTRAN_AMG_PROBE=1` logs them, and the
suite ran them on every hierarchy: all valid, with relative asymmetry at most 1.1e-12
and ||E||_A at most 0.973. Three things are not checked directly: the device cycles,
which are tied to the host cycle by iteration count; the fp32 cycle of the mixed
hierarchy, which has only unit tests; and Vulkan and mixed precision in this baseline,
which also have only unit tests.

Rotating a deck can change iterations, by up to 2.1 times on lines of bars, because
the point graph uses each block's largest entry, which depends on the axes.

**Every hierarchy has a plan.** After it is built, against the memory free at that
moment, each hierarchy is planned from its exact storage as resident, fine-row spill
or host. The earlier estimate, four times the fine nonzeros, could skip this, and a
cubic lattice's matrix-plus-transfer complexity reaches 8. Across the 86 device runs,
the memory measured after the first solve was within +48 / -21 MiB of the plan, up to
2.1 GiB. The headroom is now at least 256 MiB, because a fraction of free memory alone
fails on a nearly full device: with about 100 MiB reported free, 39 MiB allocations
failed on the Titan V. An allocation probe before the buffers turns a race with
another process into a change of plan. Starting a device context takes about 300 MiB,
so below that the backend does not start and says so. The CUDA tests hold real device
memory to force each transition. Vulkan plans the same way, without the probe, and a
failed buffer there is a reported error.

**Acceptance now means something.** `tools/amg_model_suite.py` requires complete
matching equations and reactions, an identical assembled K and F (by hash), finite
values, and explicit limits, separately for translations and rotations, on strain
energy, on the residual and on reactions. It exits nonzero when a required check
fails. `tools/test_amg_model_suite.py` feeds it wrong answers.

The first run of these checks failed solid-thin and stiffened-panel against a plain
direct solve. An extended-precision refinement showed the PCG solutions were right,
5.8e-10 and 4.8e-13 from the refined answer, and the direct solve wrong, by 1.1e-6
and 1.9e-8. Agreement "to F06 precision" could not have shown this. The reference is
now a direct solve with `--refine`. Against it, every solved run at the coarsest
levels agrees within the 1e-8 limit: at worst 1.7e-9 for constant and 4.3e-11 for
rigid on displacements. On an ill-conditioned K even an accurate solution has a large
residual (8.9e-6 relative for rigid on solid-thin), so the reference's own residual
sets the floor of that limit.

Growth exponents now come from the refinement parameter, and classes that are not
h-refinement say so: cubic-lattice grows its domain at fixed bay length, and solid-thin's
aspect ratio improves as it refines. rbe-assembly's p is 1.19 by this measure.

## Equal-budget crossover (Priority 1)

Measured on 2026-11-04 on the Titan V at the commit and working-tree patch in the
manifest. Rigid needs about twice the device memory, so resident times alone flatter
it. `tools/amg_budget_study.py` runs both methods at identical absolute budgets: both
resident, rigid at 90, 75 and 50% of its fine rows, rigid's fixed state not fitting,
and neither fitting. The scripts and manifest are in `benchmarks/pcg/amg-budget-20261104/`;
every sample, log, the patch and the binaries stay in `~/pcg-perf/budget-20261104/` on
the benchmark machine. `report` on those files reproduces the tables here.

Criteria, fixed before the measurements: a default must solve every class at every
budget where either method solves, and at equal budgets stay within 2x of the better
method's complete-deck time and 4x of its solver time. Budget conditions where neither
solves within the limit are reported and do not count.

All 498 runs, phase timings included, recorded the execution their budget predicted,
once the two tool defects noted below were corrected. Every solved complete deck agreed
with its class's first solved run to at most 8e-7 of the largest displacement, at the
F06's printed precision.
Timeouts were 300 s, and 1200 s for the large decks.

### About a million equations, CUDA fp64

Complete-deck seconds, median of the samples; `n` is the number of samples behind the
rigid column at each repeated cell, and a cell with no `n` is one run. Constant stays
resident in every column but the last, so its time is one state repeated.

| Class | Method | Both resident | Rigid 90% | Rigid 75% | Rigid 50% | Rigid on host | Both on host |
|---|---|---:|---:|---:|---:|---:|---:|
| solid-hex | constant | 19.14 | 19.08 | 19.08 | 19.15 | 19.15 | 94.40 |
| | rigid | 18.03 (n=6) | 18.07 | 18.56 | 19.17 (n=6) | 20.27 (n=6) | 20.32 |
| solid-tet10 | constant | 15.44 | 15.36 | 15.36 | 15.45 | 15.43 | 121.82 |
| | rigid | 11.59 (n=6) | 11.71 | 12.22 | 13.34 (n=6) | 15.20 (n=6) | 15.23 |
| shell-thin | constant | fails | fails | fails | fails | fails | timeout |
| | rigid | 8.00 (n=3) | 8.21 | 10.78 | 15.04 | 24.80 | 24.80 |
| rbe-assembly | constant | 147.80 | 147.80 | 147.77 | 147.71 | 147.80 | timeout |
| | rigid | 47.78 | 59.44 | 97.03 (n=4) | 182.49 (n=6) | 256.48 | 256.48 |
| shell-membrane | constant | 6.78 | 6.76 | 6.79 | 6.76 | 6.79 | 28.95 |
| | rigid | 12.45 (n=6) | 13.21 (n=6) | 22.60 | 38.50 (n=6) | 77.26 | 77.26 |
| cubic-lattice | constant | 4.66 | 4.64 | 4.61 | 4.68 | 4.61 | 7.68 |
| | rigid | 5.92 (n=6) | 5.92 (n=6) | 6.25 | 6.95 (n=6) | 12.25 | 12.25 |

Every pair's rigid / constant ratio of complete-deck time was within 2% of its cell's
median, so these verdicts do not rest on noise:

| Cell | Median | Verdict |
|---|---:|---|
| solid-hex both resident | 0.94 | rigid |
| solid-hex rigid 50% | 1.00 | tie |
| solid-hex rigid on host | 1.06 | constant |
| solid-tet10 both resident | 0.75 | rigid |
| solid-tet10 rigid 50% | 0.86 | rigid |
| solid-tet10 rigid on host | 0.98 | rigid |
| rbe-assembly rigid 75% | 0.66 | rigid |
| rbe-assembly rigid 50% | 1.24 | constant |
| shell-membrane both resident | 1.84 | constant |
| shell-membrane rigid 90% | 1.96 | constant |
| shell-membrane rigid 50% | 5.70 | constant |
| cubic-lattice both resident | 1.27 | constant |
| cubic-lattice rigid 90% | 1.28 | constant |
| cubic-lattice rigid 50% | 1.48 | constant |

Constant fails shell-thin deterministically: three runs at 100000 iterations and 258 s,
each ending with an estimated error of 1.18e-8 against the 1e-8 target.

Spilling costs in proportion to the iterations, since each one pays the gather and the
host product. At 41 iterations solid-hex barely notices losing half its rows; at 4279
rbe-assembly goes from 3 times faster than constant to 24% slower. Neither method meets
the criteria:

- Constant fails or times out on shell-thin at every budget and times out on
  rbe-assembly on the host. It is 3.1x slower on rbe-assembly when both are resident,
  and 4.6x and 8.0x slower on the solids when both are on the host. The Priority 0
  baseline adds solid-thin and stiffened-panel failures.
- Rigid solves every cell, but is up to 11.4x slower complete and 33x in the solver on
  shell-membrane, and 2.7x and 11.7x on cubic-lattice with its hierarchy on the host.

### Realistic decks

The two rover shell assemblies, about 60000 and 69000 equations with two subcases, are
small enough that their budgets are forced. Rigid meets the criteria on both and
constant does not. These budgets are near one MiB of granularity, so "50%" holds 55 and
54% of the rows.

| Deck | Method | Both resident | Rigid 50% | Rigid on host | Both on host |
|---|---|---:|---:|---:|---:|
| rover-complete | constant | 7.94 | 7.93 | 7.95 | 24.00 |
| | rigid | 6.30 | 6.59 | 7.17 | 7.18 |
| | paired ratio | 0.79 | 0.83 | 0.90 | |
| rover-static | constant | 8.02 | 8.01 | 8.04 | 24.93 |
| | rigid | 6.59 | 7.10 | 7.83 | 7.76 |
| | paired ratio | 0.82 | 0.88 | 0.97, tie | |

### Phases

`pcg_benchmark --structural-deck` times one equation system's phases, n = 5, in seconds.
The retained solve has a different right-hand side, and the refactor scales K to
D K D, which the unchanged rigid modes fit only approximately.

| Class | Budget | Method | Factor | First solve | Retained solve | Refactor | Refactored solve |
|---|---|---|---:|---:|---:|---:|---:|
| solid-hex | resident | constant | 1.01 | 4.57 | 6.85 | 0.66 | 4.56 |
| | | rigid | 3.91 | 0.48 | 0.54 | 2.74 | 0.49 |
| | rigid 50% | rigid | 3.94 | 1.62 | 1.72 | 2.69 | 1.62 |
| solid-tet10 | resident | constant | 1.15 | 8.60 | 7.95 | 0.74 | 7.51 |
| | | rigid | 5.45 | 0.67 | 0.73 | 4.25 | 0.66 |
| | rigid 50% | rigid | 5.54 | 2.36 | 2.52 | 4.22 | 2.35 |
| shell-membrane | resident | constant | 0.42 | 1.76 | 1.75 | 0.24 | 1.72 |
| | | rigid | 1.14 | 6.67 | 6.92 | 0.65 | 6.28 |
| | rigid 50% | rigid | 1.16 | 32.69 | 33.91 | 0.64 | 30.77 |
| cubic-lattice | resident | constant | 0.43 | 0.29 | 0.33 | 0.30 | 0.29 |
| | | rigid | 1.40 | 0.53 | 0.57 | 0.94 | 0.51 |
| | rigid 50% | rigid | 1.39 | 1.53 | 1.63 | 0.95 | 1.51 |

On the solids rigid repays its setup within the first solve when resident. At 50% of
the rows the first solve breaks even on solid-hex and still pays on solid-tet10, and
every later solve is about 3 to 13 times faster. A retained factor would widen this, but
SOL 101 still sets up again for each subcase.

### Large decks at the card's own limit

No forced budget: the backend's own limit, 80% of the memory the card reports free,
8181 to 8202 MiB with the desktop's 1.4 GiB in use. Two pairs, except one pair at
shell-membrane 1320. Solid-hex 70 and shell-membrane 1320 were limited by host memory,
since rigid's RSS approaches the 26 GiB this study allows.

| Deck | Equations | Method | Execution | Complete s | Solver s | Iterations | Measured MiB | Peak VRAM | Peak RSS MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| solid-hex 65 | 3.40M | constant | resident | 93.0 | 34.6 | 1534 | 4028 | 4352 | 12304 |
| | | rigid | resident | 78.2 | 19.1 | 52 | 8044 | 8368 | 16454 |
| solid-hex 70 | 4.23M | constant | resident | 119.0 | 46.1 | 1654 | 5048 | 5372 | 15282 |
| | | rigid | 49% spill | 105.2 | 31.7 | 50 | 8242 | 8566 | 20602 |
| shell-membrane 1258 | 9.50M | constant | resident | 103.8 | 59.1 | 2160 | 4778-4805 | 5101-5129 | 11033 |
| | | rigid | 98% spill | 192.1 | 146.9 | 4248 | 8224 | 8547 | 16056 |
| shell-membrane 1320 | 10.46M | constant | resident | 130.0 | 80.3 | 2702 | 5274 | 5598 | 12311 |
| | | rigid | 70% spill | 493.6 | 443.7 | 3825 | 8238 | 8562 | 17848 |

- The forced budgets predicted these runs. The crossover moves toward rigid with size
  on solids, because its iterations stay flat (41, 52, 50) while constant's grow (949,
  1534, 1654): 49% of the rows wins by 12% at 4.2M equations where 50% tied at 0.8M.
  Membrane does not improve: 3.8x at 70% of the rows.
- Peak VRAM was measured bytes plus 323 or 324 MiB in all 14 runs, from 3.9 to 8.0 GiB.
  That overhead is a constant, and no transient showed at 200 ms sampling. Measured bytes
  exceeded the plan by 17 to 70 MiB, at most 1.5%.
- Allocation never failed and the device never ran out: the 20% headroom keeps about
  2 GiB in reserve. Nothing here tests the behavior at physical exhaustion.
- Host memory is a real limit: rigid uses 4.1 to 5.4 GiB more RSS than constant here.

### Other backends

Screens on solid-hex, shell-thin, shell-membrane and rbe-assembly, with five pairs on
both resident and rigid 50% for solid-hex and shell-membrane. Rigid / constant
complete-deck ratio, paired where repeated:

| Cell | CUDA fp64 | Vulkan fp64 | CUDA mixed |
|---|---:|---:|---:|
| solid-hex both resident | 0.94 | 0.96 | 0.87 |
| solid-hex rigid 50% | 1.00 | 1.01 | 0.91 |
| shell-membrane both resident | 1.84 | 1.85 | 1.79 |
| shell-membrane rigid 50% | 5.70 | 5.61 | 6.18 |
| rbe-assembly both resident | 0.32 | 0.32 | 0.33 |
| rbe-assembly rigid 50% | 1.24 | 1.29 | 1.44 |

- Vulkan fp64 reproduces CUDA fp64, 3 to 9% slower with fine rows on the device. Rigid's
  host runs are 10 to 25% slower, so it times out on rbe-assembly on the host at 300 s
  (CUDA 256 s). That is the only budget at which rigid fails where constant solves.
- Mixed accepts a solution at an estimated error of 1e-6 where fp64 asks for 1e-8, so its
  times are not at the same accuracy as the other two columns' and only its ratios
  compare. This changes one verdict. Constant's shell-thin, which fp64 misses at 1.18e-8,
  passes on mixed at 6.5e-7, in 50000 iterations and 115 s against rigid's 7.8 s at
  6.0e-8: from failure to 14.8x slower.
- Mixed needs more device memory than fp64, not less, because the fp32 hierarchy adds to
  the fp64 fine operator refinement needs. Plans in MiB on solid-hex: constant 1366
  against 941, rigid 1985 against 1870. Below the fp32 plan it uses fp64, so a constant
  mixed solve can be resident at budgets its fp32 hierarchy could not fit.
- At that looser target mixed is 8 to 14% faster than fp64 on rbe-assembly and
  shell-membrane and 2.5% faster with rigid on shell-thin. On solid-hex it is 10% slower
  with constant (21.08 s and 1648 iterations against 19.14 s and 949) and 1.6% slower
  with rigid, so fp32 gives no general speedup here.

### A tube chassis

`cubic-lattice` is not what a beam model usually looks like, so `tube-chassis` adds one:
a rigid-jointed, triangulated frame of 182 square and round tubes, with the proportions
of a Locost (wheelbase 88 in, 42 in wide, 12 in from floor to top rail) and a tube layout
of its own, since the book's coordinates are not published. It is not a replica: it has
more triangulation than the book's frame and double wishbones at the rear in place of a
live axle. 39 tubes (21%) lie on an axis, all of them cross-members, and joints have 2 to 9
tubes. Each corner is a rigid upright (RBE2) on six pinned links, one of them a 40 kN/m
spring. Wheel forces for braking into a corner over a bump are applied at the contact
patches, and three statically determinate supports stand in for the masses, so they
cannot stiffen the frame.

The model behaves physically. Reactions balance the wheel forces, and its torsional
stiffness at the frame is 6935 N m/deg, identical at 10, 20 and 80 elements per metre. That
is above the 2000 N m/deg or so reported for a book chassis and within the 4000 to 6000+
reported for stiffened ones, as builders' figures from varied test setups, which fits its
extra triangulation.

| Equations | Elements per m | Element length | Direct s | Rigid s | Rigid iterations | Constant |
|---:|---:|---:|---:|---:|---:|---|
| 62k | 135 | 7.4 mm | 0.42 | 0.90 | 729 | fails, error 1.2e-2 |
| 249k | 540 | 1.9 mm | 1.35 | 3.12 | 1671 | fails, error 7.7e-3 |
| 999k | 2170 | 0.46 mm | 5.29 | 19.1 | 4624 | fails, error 1.6e-3 |

Times are the complete deck, from the suite, with the probes on. Rigid agrees with the
refined direct solve to 1.6e-12 or better in displacement. Constant is not defeated by the
support or the springs: at 249k equations it still fails with stiff links in place of the
springs, with the rear pickups clamped as on a torsion rig and wheel loads at the front
only, and with both, where rigid takes 1531 to 1671 iterations.

At equal budgets at 999k equations, constant fails at every budget on all three backends,
including mixed's looser target, and times out with both on the host, so rigid meets the
criteria by default. These are single runs without probes, since there are no pairs to
form. Complete-deck seconds for rigid:

| Backend | Resident | 90% | 75% | 50% | Host |
|---|---:|---:|---:|---:|---:|
| CUDA fp64 | 15.7 | 16.7 | 24.4 | 46.2 | 166.7 |
| Vulkan | 17.2 | 19.3 | 25.6 | 47.2 | 203.6 |
| Mixed (1e-6 target) | 12.5 | 12.5 (fp32) | 12.5 (fp32) | 42.7 | 154.4 |

Three limits apply. A real chassis model uses a few elements per tube, about 1000 to 10000
equations, which is below the 50000 where AMG starts, so it goes to a direct solve and none
of this applies to it. To reach AMG the class must divide tubes into elements far shorter
than their 25 mm width, which a CBAR does not need: one element already represents a tube
exactly, so refinement changes the system and not the answer. And a direct solve wins at
every size here, 3.6x over rigid at 999k equations, because chains of elements cause almost
no fill. A deck dominated by beams is better sent to `--backend=cpu`. Scripts and manifest
are in `benchmarks/pcg/chassis-20261104/`, and results in `~/pcg-perf/chassis-20261104/` on
the benchmark machine.

### The default

`drilling` has been the structural default since 2026-11-05, by decision after the
results under "Drilling modes", which also lists what it costs on decks that don't need
it. It adds to the following decision.

`rigid` became the structural default on 2026-11-04, by decision after these results
and not because it passed: neither method meets the criteria on the synthetic classes.
Constant does not solve shell-thin on CUDA fp64 or Vulkan and times out on rbe-assembly
with both on the host. Rigid's failures are slowness on shell-membrane and cubic-lattice,
apart from its Vulkan timeout on rbe-assembly with its fixed state on the host, the one
budget where it fails and constant solves. Rigid meets the criteria on both realistic
rover decks, and constant does not.

The costs are about twice the device memory, 4.1-5.4 GiB more host memory on the large
decks, and up to 11.4x complete-deck time on membrane-like decks under memory pressure.
`--amg-modes=constant` or `PARAM,AMGMODES,CONSTANT` selects the scalar hierarchy for such
decks, and "Drilling modes" below is the better option on CQUAD4 membranes. Nothing changes below 50000 equations, on the CPU backends, or on Vulkan devices
without fp64, which cannot use the modes.

### Drilling modes

`--amg-modes=drilling` (`PARAM,AMGMODES,DRILLING`) adds two columns to the rigid-body
modes: a uniform rotation about the shell normal, and one that alternates in sign
between neighbouring grids. A CQUAD4's membrane strain depends only on differences of
drilling rotation, and on a rectangle both patterns leave those strains at zero, so
only the drilling penalty resists them. An aggregate carries a column only where it
isn't zero and isn't in the span of the columns before it, so grids with no CQUAD4 pay
nothing. A model with no CQUAD4 gets exactly the rigid-body modes. Neither column
changes the solution, which refinement accepts against the assembled stiffness. It has
been the default since 2026-11-05; `--amg-modes=rigid` leaves the columns out.

CUDA fp64 on the TITAN V with no budget, one run each, measured 2026-11-05. Scripts are
in `benchmarks/pcg/amg-drilling-20261105/`, and results and logs are in
`~/pcg-perf/drilling-20261105/` on that machine. Iterations are at the suite's three
levels, and time is the complete deck at the largest:

| Class | Equations | Rigid iterations | Drilling iterations | Rigid s | Drilling s |
|---|---|---|---|---:|---:|
| shell-membrane | 61k / 241k / 962k | 634 / 1213 / 2167 | **30 / 28 / 38** | 12.4 | **6.5** |
| shell-cylinder-cd | 62k / 249k / 995k | 581 / 1159 / 1452 | **429 / 431 / 395** | 14.8 | **10.4** |
| rbe-assembly | 55k / 319k / 939k | 1413 / 2614 / 4234 | **826 / 1988 / 2186** | 47.5 | **33.8** |
| stiffened-panel | 63k / 251k / 999k | **309** / 645 / **1413** | 350 / **643** / 1416 | **11.6** | 13.1 |
| shell-foundation | 61k / 242k / 965k | 106 / **100 / 130** | 106 / 101 / 133 | **6.6** | 7.2 |
| shell-thin | 59k / 238k / 955k | **722 / 612 / 567** | 736 / 664 / 626 | **8.0** | 9.0 |
| shell-thick | 59k / 238k / 955k | 30 / 31 / 35 | 30 / 31 / 35 | **6.1** | 6.7 |

The solid and beam classes have no CQUAD4, so both options run the same hierarchy. Every class's coarsest level agrees with a refined direct solve.

Decks outside the suite, at about a million equations unless noted:

| Deck | Rigid iterations | Drilling iterations | Rigid s | Drilling s |
|---|---:|---:|---:|---:|
| Sheet, elements randomly reversed, in-plane load | 1809 | **33** | 11.6 | **6.3** |
| Sheet, 5% of cells split into triangles, in-plane load | 1773 | **33** | 11.8 | **6.6** |
| Sheet, half quads and half triangles, in-plane load | 1347 | **45** | 12.2 | **8.5** |
| Sheet, 32x32 checkerboard of quads and triangles, in-plane load | 4597 | **392** | 22.0 | **9.9** |
| Sheet, 8x8 checkerboard of quads and triangles, in-plane load | 3640 | **1299** | 19.5 | **13.3** |
| Unstructured all-quad, in-plane load, 64k / 252k / 988k | 834 / 1799 / 3786 | **125 / 226 / 402** | 19.9 | **8.7** |
| Unstructured all-quad, transverse load | **363** | 467 | **7.8** | 9.0 |
| Unstructured quads and triangles, in-plane load, 227k / 902k | 1431 / 3511 | **361 / 1834** | 19.7 | **16.4** |
| Unstructured quads and triangles, transverse load | 383 | 384 | **9.6** | 11.0 |
| Unstructured triangles, in-plane load | 43 | 43 | 11.0 | 11.0 |
| Closed cylinder, axial load, 576 around (two-colourable) | 1993 | **498** | 17.2 | **10.8** |
| Closed cylinder, axial load, 577 around (not two-colourable) | 1987 | **495** | 17.3 | **10.8** |
| Rover complete, 61k | **330** | 379 | **6.4** | 6.7 |
| Rover four-wheel, 69k | 398 | **390** | **6.5** | 6.7 |

`shell-membrane` at 9.5M equations, under the planner's default budget, goes from 4248
iterations in 189 s to 36 in 69 s, with peak host memory from 16.0 to 19.2 GiB. The
larger hierarchy puts 7.78M fine rows on the CPU instead of 88k, and it is still faster.

What this shows:

- **Membrane action on regular meshes gains the most.** Iterations stay nearly flat
  with refinement there. On the unstructured all-quad mesh they fall about ninefold but
  still grow, so the two patterns don't capture everything on an irregular mesh.
- **Bending-dominated decks pay.** They get no gain and spend about 10-13% more time,
  with up to 29% more iterations on the unstructured all-quad deck.
- **Memory.** The hierarchy's plan grows 27% on an all-CQUAD4 deck (852 to 1080 MB on
  `shell-membrane` at 962k), and 3% on `rbe-assembly`, whose solids carry no extra
  columns.
- **Where triangles meet quads.** Neither pattern has a best set of grids. On the
  checkerboards, leaving the alternating column off grids that touch a triangle gives
  432 and 285 iterations against 1299 and 392. On the scattered and half decks it gives
  232 and 75 against 33 and 45. Both columns on every CQUAD4 grid is the simpler rule
  and beats rigid on all of them, so that is what's implemented.
- **It doesn't help `stiffened-panel`,** which is limited by something else.

### Computed transfers

A hierarchy with modes builds its finest transfer as `P = (I - M A) T`, where `T` is the
tentative prolongation from each patch of grids, `A` the fine matrix and `M` the
smoother's weights, and `R = P^T`. A device already holds `A` and `M`, so it can hold `T`
instead of `P` and `R`:

```text
restriction    y -= A (M y)        f_coarse = T^T y
prolongation   w = T e             x += w - M (A w)
```

This is the same preconditioner in less memory, for one more product with `A` each way.
Patches keep `T` as one dense block of values, a row per equation, stored once for `T`
and `T^T` and without column indices (`AmgTentative`). The AMGCL scalar hierarchy filters
`A` when it smooths, so it isn't of this form and keeps its stored transfers, which covers
all thermal and magnetic solves.

No option selects this. The plan takes the first of these that fits the device budget:

1. Everything resident with stored transfers, which is fastest.
2. Everything resident with computed transfers.
3. Computed transfers with a suffix of the fine rows on the CPU.
4. The host hierarchy.

Mixed precision tries the fp32 hierarchy with stored then computed transfers before that
list. The log reports the kind on the plan line, and `pcg_benchmark` as
`computed_transfers`. Only placement differs, so plans remain free to change with memory
without changing the method. A refactor smooths the finest `P` again, because a stored `P`
left with old values would be a different operator from one computed from the new `A`.

CUDA fp64 on the TITAN V, measured 2026-11-05, default modes, one run each, largest level
of each class, with the budget set just under the stored plan to force computed. Scripts
are in `benchmarks/pcg/amg-computed-transfers-20261105/`, results and logs in
`~/pcg-perf/implicit-20261105/` on that machine. ms/iter is iterate time including
refinement over iterations:

| Class | Equations | Iterations | Plan MiB, stored to computed | ms/iter, stored to computed |
|---|---:|---:|---:|---:|
| solid-hex | 806,880 | 41, 41 | 1870 to 1249 (-33%) | 11.74 to 13.29 (+13%) |
| solid-tet10 | 693,576 | 67, 67 | 1881 to 1314 (-30%) | 9.83 to 11.38 (+16%) |
| solid-contrast | 806,880 | 98, 98 | 2085 to 1457 (-30%) | 9.50 to 11.18 (+18%) |
| solid-thin | 924,480 | 3457, 3457 | 1541 to 1057 (-31%) | 6.06 to 7.72 (+27%) |
| shell-thin | 955,206 | 626, 626 | 1042 to 628 (-40%) | 3.86 to 4.67 (+21%) |
| shell-thick | 955,206 | 35, 35 | 1036 to 622 (-40%) | 6.20 to 7.10 (+15%) |
| shell-membrane | 962,400 | 38, 38 | 1030 to 612 (-41%) | 5.88 to 6.51 (+11%) |
| shell-cylinder-cd | 994,752 | 395, 395 | 1402 to 971 (-31%) | 5.69 to 7.46 (+31%) |
| shell-foundation | 964,800 | 133, 133 | 1050 to 632 (-40%) | 4.37 to 5.17 (+18%) |
| stiffened-panel | 998,784 | 1416, 1417 | 1241 to 783 (-37%) | 4.41 to 5.20 (+18%) |
| cubic-lattice | 876,408 | 160, 160 | 927 to 546 (-41%) | 3.22 to 3.17 (-2%) |
| tube-chassis | 998,934 | 4624, 4635 | 680 to 509 (-25%) | 2.72 to 3.35 (+23%) |
| rbe-assembly | 938,886 | 2186, 2188 | 1872 to 1250 (-33%) | 7.01 to 8.48 (+21%) |
| rover-complete | 60,606 | 379, 379 | 79 to 54 (-32%) | 0.41 to 0.48 (+17%) |
| rover-static | 68,730 | 390, 390 | 90 to 61 (-32%) | 0.44 to 0.50 (+14%) |

Vulkan on the same card, in the same plan sizes: shell-membrane 5.51 to 6.08 ms/iter and
solid-hex 10.47 to 10.55 ms/iter, with the same iterations.

Where it matters is decks that used to spill. Default budget, drilling modes, stored
transfers before and computed now:

| Deck | Equations | CPU rows | ms/iter | Iterate s | Complete deck s |
|---|---:|---:|---:|---:|---:|
| shell-membrane-1258 | 9,502,932 | 8.08M to 0 | 290 to 82 | 10.45 to 2.94 | 68.0 to 60.1 |
| solid-hex-70 | 4,234,440 | 2.18M to 0 | 210 to 67 | 10.50 to 3.35 | 101.7 to 94.4 |

With budgets for a share of the fine rows on the GPU, plan MiB and ms/iter:

| Deck | 100% | 90% | 75% | 50% |
|---|---:|---:|---:|---:|
| shell-membrane-1258 | 6236, 82 | 5971, 98 | 5573, 171 | 4910, 289 |
| solid-hex-70 | 6795, 67 | 6432, 87 | 5861, 170 | 4908, 312 |

The old plan couldn't reach 50% of the rows of shell-membrane-1258 on this card. On
solid-hex-70 it held 49% in 8242 MiB at 210 ms/iter; computed transfers at 50% are slower,
312 ms/iter, because five products reach the CPU rows per iteration instead of three, but
need 4908 MiB, below the 6268 MiB of the old plan's fixed state alone.

The one measured loss is a deck that only just spilled. shell-membrane-1258 with `rigid`
needed just 0.39M CPU rows with stored transfers, and computing them instead took it from
31.5 to 40.5 ms/iter and from 187 to 225 s for the deck, measured with the prototype.
Computing is preferred anyway, since keeping every row on the GPU gains far more on the
decks above.

Host memory rises by the patches, about 3% on these two decks (19.2 to 19.9 GiB and
20.6 to 21.2 GiB of peak RSS). Stored plans hold them too, since a refactor needs `T`.

#### Kernel profile

`nsys --cuda-graph-trace=node`, one iteration in the middle of a solve, computed transfers:

| Part of the iteration | shell-membrane-400, us | solid-hex-40, us |
|---|---:|---:|
| Five products with the fine `A` | 2883 (68%) | 6636 (76%) |
| Level 1 products | 443 (10%) | 1281 (15%) |
| `T` | 231 (5%) | 119 (1%) |
| `T^T` | 146 (3%) | 93 (1%) |
| Fine `M` scalings | 78 (2%) | 66 (1%) |
| CG vector operations | 181 (4%) | 154 (2%) |
| Everything else, coarser levels | 288 (7%) | 341 (4%) |
| Total | 4250 | 8690 |

With stored transfers the same iteration of shell-membrane-400 takes 3525 us, so the true
per-iteration cost there is +21%, not the +11% in the table above, whose ms/iter also
holds about 70 ms per solve outside the iterations, in refinement.

Nothing here was wasted. Between kernels the graph has gaps of 0.3-1.4 us, 14-17 us per
iteration. The fine product reads 12 bytes per nonzero at about 480 GB/s on the shell and
570 GB/s on the solid, against the card's rated 652 GB/s, so it is limited by memory
bandwidth and not by the kernel. `T^T` with 8, 16 or 32 threads per coarse equation takes
the same time to within noise. The new kernels hold no large share, and the product
modes they added cost the same as the old ones.

That bandwidth limit says where the rest of the time is: less to read per nonzero. A
measurement of the same product in other formats, on matrices from these hierarchies,
is in `~/pcg-perf/nnz-20261105/` on that machine and shows the gain depends on the level.
Rows of a coarse patch share one column list, so fp32 values with shared columns take
about 37% of the memory and 53% of the time of CSR on the unstructured and rover decks.
On the fine `K`, shared or 16-bit indices save 15-25% of the memory but at most 7% of the
time on those decks. The uniform synthetic meshes show much more, up to 29%, because their
rows repeat one pattern throughout, so they overstate what a real mesh would gain.

### Limits of these results

- `cubic-lattice` fills its whole volume with rigid-jointed CBARs between nearest
  neighbours, with no diagonals, so a lateral load is carried by member bending. It is
  not a space frame, whose triangulated members carry mostly axial load, so its results
  should not be read as evidence about those structures. It was named `space-frame` when
  measured, and the stored results and the scripts under `benchmarks/pcg/` use that name.
- Clear-cut cells were not repeated, including constant's host failures and rigid's
  host runs on membrane, frame and rbe. The rbe-assembly 75% cell had 3 pairs and
  shell-thin both resident had 2.
- Phases exclude rbe-assembly, whose constant solves would pass the 300 s limit, and
  shell-thin, which constant does not solve.
- Mixed and Vulkan were not run on the large decks.
- Per-level A/P/R bytes in the results are CSR estimates from the logged nonzeros, not
  device measurements. Transfers are the logged vector bytes per SpMV.
- No selection policy exists yet, so no selection overhead was measured.

Two defects in the study tool itself gave false alarms and were corrected: a mixed
solve's plan needs a second run under its fp32 plan to find its fp64 plan, and a spilled
run logs the partition it chose rather than what staying resident would need. The first
mixed attempt, which stopped there, is kept in `cuda-pcg-mixed/first-attempt/`; `stage4.sh`
replaces the mixed part of `stage3.sh`.

Measuring phases also exposed that `factor()` ignored the near-nullspace on CUDA, mixed
and Vulkan, so a retained factor silently ran constant. Tests on each backend now check
a factored and refactored solve against the host's iteration count. Logs now flush each
second, since a run killed at its time limit otherwise left an empty log.

## CPU PCG

`--backend=cpu-pcg` was Eigen's CG with incomplete Cholesky, which was slow because of
its iterations rather than its time per iteration, so threading it would not have
helped. Since 2026-11-06 it is CG on K in fp64, preconditioned by the same AMG
hierarchy as the GPU backends through a cycle built for the CPU in
`src/solver/amg_cpu.hpp`. Incomplete Cholesky is gone.

Complete-deck seconds on bob-desktop, 8 threads pinned to `taskset -c 0-7`, one run
each. Incomplete Cholesky runs stopped at 240 s:

| Deck | Equations | Incomplete Cholesky | CPU PCG | TITAN V `cuda-pcg` | CPU direct |
|---|---:|---:|---:|---:|---:|
| shell-membrane-200 | 241200 | 18.9 | 1.3 | 1.2 | 1.4 |
| solid-hex-26 | 227448 | 17.5 | 3.2 | 2.5 | 10.9 |
| thermal-hex-64 | 266175 | 5.1 | 3.5 | 2.4 | 10.8 |
| shell-thin-200 | 237606 | timeout | 5.0 | 1.9 | 1.5 |
| stiffened-panel-144 | 250560 | timeout | 5.2 | 2.2 | 2.7 |
| rbe-assembly-128 | 318726 | timeout | 40.1 | 7.1 | 5.4 |
| rover-static | 68730 | 40.6 | 3.1 | 1.8 | 1.3 |

Below the old 50000-equation threshold AMG was equal or better on all 18 decks tried,
so there is no size switch. Direct is often still the fastest at small sizes and on
thin shells.

The two largest decks now take about a minute, not the hours first feared:

| Deck | Backend | Complete s | Setup s | Iterate s | Iterations | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|
| solid-hex-70 (4234440 equations, 335M nonzeros) | CPU PCG | 63.3 | 27.4 | 13.6 | 36 | 19.6 |
| | `cuda-pcg` | 45.9 | 20.5 | 3.4 | 50 | 17.6 |
| shell-membrane-1258 (9502932 equations, 244M nonzeros) | CPU PCG | 53.5 | 20.0 | 13.8 | 28 | 21.2 |
| | `cuda-pcg` | 35.9 | 13.8 | 3.0 | 36 | 18.4 |

The two backends' displacements agree to 8e-10 and 2e-8 of the largest value, within
the F06's printed precision.

### Why the cycle is built this way

**Bytes read.** This Ryzen 7 5700 saturates its memory at about 35-40 GB/s by 2-4
threads, so threads help the kernels only as far as they read less. On `solid-hex-40`:

| Threads | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Complete deck s | 40.5 | 23.9 | 15.3 | 11.2 | 11.3 |
| Iterate s | 5.23 | 3.05 | 2.16 | 2.07 | 2.46 |

Past 8, the extra threads share the same cores; an earlier 16-thread run took 2.99 s to
iterate and 11.8 s in all. Nothing caps the default, which is OpenMP's 16 unless
`OMP_NUM_THREADS` is set. Since iterations hardly change, the cycle runs entirely in
fp32 and stores one column list for consecutive rows with identical columns, which
took about 30% off iterating before the change below. CG and the final accuracy check
stay in fp64 against K. Rows share a list only when their columns really match: 1.8-2.0
rows per list on hex and the rover, 1.02 on `shell-membrane-200`, where a uniform mesh
cancels entries to exact zero, and 1.0 on thermal decks, which have one equation per
grid.

**Gauss-Seidel.** The GPU cycle smooths with weighted Jacobi, which updates every row
from the previous values, so that rows can run in any order at once. Gauss-Seidel uses
each new value as soon as it exists, and reduces error faster per sweep, but a CPU has
few enough threads to afford the ordering it needs. Here it is exact Gauss-Seidel in a
thread-shareable order: rows are split into 16 blocks of equal nonzeros; rows with no
coupling to another block's remaining rows go first, in parallel; the rest are coloured
by block with at most 4 colours, each a stage; and what's left repeats with half the
blocks. With M = D + L in that order, M + M' - A = D is positive definite whatever the
order is, so the symmetric sweep is a valid preconditioner for CG, with eigenvalues of
BA in (0, 1] and no weight to tune. A test forms the cycle on small models to check
this. An iteration costs more, 74 ms against 61 ms on `solid-hex-40`, which the fewer
iterations more than repay. The residual after a forward sweep needs only the upper
half of the product, which keeps that cost down. An earlier hybrid, Gauss-Seidel inside
a block and Jacobi between, lost the gain: 44-50 iterations on hex against 29 for
serial Gauss-Seidel and 41 for Jacobi.

Largest level of each class, both cycles in fp32 at 8 threads. Jacobi comes from a
prototype of the same code, one run each:

| Class | Equations | Jacobi iterations | Gauss-Seidel iterations | Jacobi iterate s | Gauss-Seidel iterate s |
|---|---:|---:|---:|---:|---:|
| shell-membrane | 962400 | 38 | 27 | 1.33 | 1.08 |
| shell-thick | 955206 | 35 | 24 | 1.23 | 0.93 |
| solid-hex | 806880 | 41 | 28 | 2.49 | 2.06 |
| solid-tet10 | 693576 | 67 | 44 | 3.65 | 3.26 |
| solid-contrast | 806880 | 98 | 59 | 6.03 | 4.85 |
| thermal-hex | 1009899 | 283 | 239 | 7.74 | 7.46 |
| shell-foundation | 964800 | 133 | 87 | 4.45 | 3.32 |
| cubic-lattice | 876408 | 160 | 115 | 3.86 | 3.28 |
| shell-thin | 955206 | 626 | 407 | 20.27 | 14.78 |
| shell-cylinder-cd | 994752 | 395 | 220 | 17.42 | 10.67 |
| stiffened-panel | 998784 | 1350 | 756 | 48.62 | 30.66 |
| solid-thin | 924480 | 3673 | 2369 | 174.92 | 143.35 |
| tube-chassis | 998934 | 4407 | 2410 | 90.59 | 54.88 |
| rbe-assembly | 938886 | 2357 | 1992 | 128.45 | 125.80 |

Gauss-Seidel also won on the magnetic coil, tied on the motor, and lost on one deck
found: the rover took 0.88 s against 0.75 s, with 229 iterations against 390 at twice
the time per iteration, probably because its numbering leaves few rows that can run at
once. Reordering rows breadth-first gained little on most decks and made the panel worse
(408 iterations), so it was dropped. Flexible CG gained nothing either.

No deck tried has failed in a way that points to fp32, so there is no fp64 fallback. The
hardest found are `solid-thin` at 58320 equations, 13126 iterations in 44 s, and MYSTRAN's
`hemisphere with cutout 4e-4`, 33475 equations and about 11900 iterations in 22 s on the
Mac, where direct takes 0.5 s. Both reached their accuracy, though one CG run stops at
10000 iterations, so a harder deck would be rejected.

The solution doesn't depend on the thread count, bit for bit: blocks are fixed, and dot
products sum the blocks in a fixed order.

### Time outside the solver

Assembly and recovery were most of a deck, on every backend. On `solid-hex-40`:

| Stage | Before, s | After, s |
|---|---:|---:|
| Assemble K | 1.33 | 0.69 |
| `build_csr` | 1.59 | 0.63 |
| Linear solve | 8.0 | 7.1 |
| `recover_results` | 4.03 | 0.57 |
| `recover_forces` | 4.59 | 0.71 |
| Complete deck | 20.9 | 11.2 |

- `build_csr` is a parallel stable counting sort, and `merge_from` moves each chunk's
  triplets rather than copying them.
- Element recovery and grid forces run in chunks of 512 elements and merge in chunk
  order, so results are in element order and the same to the bit whatever the thread
  count.
- `MpcHandler::build` factored a dense matrix of every kept row for each row whose
  pivot appeared in an earlier row, which is quartic in the rows, and took 2.28 s per
  subcase on the rover. Only the rows a new row can reach through shared pivot columns
  decide its rank, which took it to 3 ms. Judging a row's rank on its own block also
  fixes a defect: a stiff row with nothing in common could raise the rank threshold
  and move the dependent to the wrong term.
- Thermal assembly is threaded the same way, and a QVOL, QBDY1, QBDY2 or QVECT naming
  every element no longer searches the model for each.
- Assembly reserved space for every triplet in a builder it never used, about 12.6 GB
  of untouched address space on hex-70, which is gone.

The GPU backends gained the same: solid-hex-70 went from 94.4 s to 45.9 s,
shell-membrane-1258 from 60.1 s to 35.9 s and the rover from 6.5 s to 1.8 s.

### Verification

- The model-class suite solved all 48 levels of 16 classes, and all 13 comparisons
  against a refined direct solve passed on the structural classes, the worst being a
  residual of 9.0e-6 on solid-thin, whose limit is 1e-6 with a floor for its
  conditioning. The rest were 2.9e-8 or less.
- All 23 e2e cases pass on `--backend=cpu-pcg`.
- On MYSTRAN's 573 validation decks, run on the Mac, AMG completes every deck direct does
  except two, and rejects those with an error. Both load a direction with no stiffness,
  so `K u = F` has no solution as the deck stands. `RBE3 26` loads grids held only by a
  CQUAD8, which is ignored as unsupported. Accelerate's direct solve leaves those rows at
  zero and warns, with a relative residual of 0.89, CHOLMOD on Linux stops with an error,
  and incomplete Cholesky returned a residual near 1e149. In `AUTOSPC reaction forces` a
  moment drives a rigid in-plane rotation, since `PARAM,K6ROT` is ignored. Direct warns
  and returns displacements of order 1e9 that differ between Mac and Linux, and
  incomplete Cholesky also fails. Validation records number 1656 failures against
  direct's 1648. Of the 8 extra, 5 are checks `RBE3 26` passed on direct, 1 is a reaction
  `AUTOSPC reaction forces` reports on direct, and 2 are shear forces of about 1e-15
  against an expected zero. Failures on both differ only by noise near zero.
- Rows with no stiffness get a unit diagonal, and a singular coarsest level uses a
  pseudo-inverse, so a deck that only has unloaded singular directions still solves, on
  every AMG backend.

### Limits

- Times are one run each, and rbe-assembly, solid-thin and tube-chassis still take
  thousands of iterations. Direct took 5.4 s against 40.1 s on `rbe-assembly-128`.
- Setup is now the largest part of a hex deck: on solid-hex-40 AMGCL's matrix products
  are about 20% of the run's CPU time and Eigen's QR about 10%. The cycle's own
  preparation is 7.3 s on hex-70 and the fp64 hierarchy stays in memory during iterating,
  so there is room to cut both.

Scripts are in `benchmarks/pcg/cpu-pcg-20261106/`, and results stay in
`~/pcg-perf/cpucycle-20261106/` on that machine.

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
