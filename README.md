# Vibestran

A finite element solver written in modern C++20 that attempts to be
Nastran/Mystran-compatible: linear statics, normal modes and steady heat transfer
from Nastran bulk data, plus a magnetics extension.

## Why

I was curious how feasible it would be to create something this complex with an AI agent. It's a well documented field, but it also requires that the math is precisely correct. I also wanted to experiment with seeing how fast a FEM solver could be when written in a language I understand (well, within the limits of one's ability to understand C++), and using GPU acceleration. I was prepared to do the benchmarking and optimization myself, but it turned out that Claude was quite capable with some guidance.

Vibestran draws significant inspiration from the [Mystran](https://github.com/MYSTRANsolver/MYSTRAN)
project. If you're unfamiliar with the Mystran project you should go look at it first. It should be considered the reference for open source Nastran-compatible solvers. Without it, as an introduction to OSS FEM Solvers, there's zero chance I would have tried this.

The OpenJFEM project, which is another solver, written in Julia, was also an inspiration for trying this with an agent.

## Performance

An 84k-element, 170k-node hex mesh model solves in 2.6 s on an M2 MacBook Air, 3.6 s on
an RTX 3060 12 GB with a Ryzen 5700, and 4.3 s on the Ryzen 5700 alone. GPU and large-deck
benchmarks are in [PCG_PERFORMANCE.md](PCG_PERFORMANCE.md) and
[PCG_LARGE_DECKS.md](PCG_LARGE_DECKS.md).

## Solutions

| `SOL` | Analysis |
|---|---|
| 101 | Linear static |
| 103 | Normal modes (`EIGRL`) |
| 153 | Steady heat transfer; `ANALYSIS = STATICS` subcases then run a linear static with the computed temperatures |
| `MAGSTAT` | Nonlinear magnetostatics, 3D, planar and axisymmetric, with motion and phase-current sweeps |
| `MAGTRANS` | Eddy currents in planar and axisymmetric motors at constant speed |
| `MAGHARM` | Harmonic eddy currents on fixed, linear geometry |

The magnetic solutions use Vibestran's own cards. [MAGNETICS.md](MAGNETICS.md) documents
them with examples, and [Magnetic_Validation](Magnetic_Validation/README.md) holds the
independent comparisons.

## Supported input

- **Elements**: CQUAD4, CTRIA3, CHEXA8, CHEXA20, CTETRA4, CTETRA10, CPENTA6, CBAR, CBEAM,
  CROD, CONROD, CBUSH, CELAS1-4, CMASS1-4, CONM2 (CG offset, inertia tensor, CID / CID = -1),
  and CHBDY for heat transfer.
- **Shells**: CQUAD4 is the MITC4+/D continuum shell (Ko, Bathe & Zhang 2025) and CTRIA3
  the MITC3+ shell (Lee, Lee & Bathe 2014), with six DOFs per grid, shared grid directors
  (averaged within the `PARAM,SNORM` fold angle, default 20°, or given by SNORM cards),
  warped and curved geometry, transverse shear, ZOFFS, T1-T4 and TFLAG, THETA/MCID material
  axes, and thermal loads. See
  [SHELL_BENCHMARKS.md](SHELL_BENCHMARKS.md).
- **Solids**: CHEXA8 is the Q1/E21 enhanced assumed strain hexahedron, the same element as
  MYSTRAN's, which it reproduces to print precision; `PSOLID ISOP = FULL` selects full
  integration. CHEXA20 is a projected assumed-strain serendipity hexahedron without
  membrane or volumetric locking ([CHEXA20_PLAN.md](CHEXA20_PLAN.md)). Midside grids of
  CHEXA20 and CTETRA10 may be left blank, as in Nastran, which makes that edge linear.
- **Materials**: MAT1 (all elements), MAT2 and MAT8 (shells), MAT4 and MAT5 (heat transfer);
  MAT3 and MAT6 are parsed.
- **Properties**: PSHELL (MID1-MID4, 12I/T³, TS/T, NSM, Z1/Z2), PCOMP (classical lamination;
  LAM blank, SYM, MEM, BEND, SMEAR, SMCORE), PSOLID, PBAR, PBARL (`ROD`, `TUBE`, `BAR`, `BOX`,
  `CHAN`, `T`, `I1`), PBEAM, PROD, PBUSH (`K`), PELAS, PMASS, PHBDY.
- **CBAR**: pin flags PA/PB, offsets WA/WB with OFFT `GGG`/`BGG`, BAROR defaults.
- **Scalar points**: SPOINT, or implied by a CELASi/CMASSi with a blank component; usable in
  springs, masses, SPC, MPC, SLOAD, TEMP and heat transfer.
- **Loads**: FORCE, MOMENT, SLOAD, TEMP/TEMPD, PLOAD, PLOAD1 (CBAR/CBEAM), PLOAD2, PLOAD4,
  GRAV, ACCEL1; QBDY1, QBDY2, QHBDY, QVECT and QVOL in heat transfer.
- **Constraints**: SPC, SPC1, GRID PS, MPC/MPCADD, RBE2, RBE3, and Nastran's AUTOSPC at every
  grid (`PARAM,EPZERO`, `PARAM,AUTOSPC,NO`).
- **Coordinate systems**: CORD1R/C/S, CORD2R/C/S; GRDSET defaults.
- **Mass**: consistent by default, unlike Nastran; `PARAM,COUPMASS,-1` gives lumped mass (HRZ
  for continuum elements). Massless DOFs are condensed exactly in the eigensolver.
- **Input formats**: Nastran BDF, and CalculiX/Abaqus `.inp` (experimental: C3D8, C3D4,
  C3D10, C3D6, S4, S3 and their R variants; see `include/io/inp_parser.hpp`).

## Output

- F06 and OP2 always; nodal and element CSV with `--csv` or `PARAM,CSVOUT,YES`.
- Displacements, SPCFORCE and MPCFORCE in each grid's CD system.
- Stresses and strains; shell strains as STRCUR (default) or FIBER, at the centre or CORNER.
- Element forces (FORCE/ELFORCE) for CBAR, CBEAM, CROD, CONROD, CELAS1-4, CQUAD4 and CTRIA3.
- GPFORCE grid point force balance in linear statics.
- The magnetic solutions write F06 tables and VTKHDF fields.

The SPCFORCE, MPCFORCE, element force and GPFORCE blocks follow MYSTRAN's layout.

## Limitations

- Not implemented: CTUBE, CDAMPi, CQUAD8, CTRIA6, the 15-node CPENTA, MAT9, superelements
  (SEID), PCOMP ply stresses and strains, and CBAR offsets in the element system
  (OFFT `xGO`/`xOG`/`xOO`).
- `ACCEL` is parsed but not applied, and PLOAD1 applies only to CBAR and CBEAM.
- CELAS3/4 and CMASS3/4 are reported as CELAS1/2 and CMASS1/2.
- There is no stress recovery for CBUSH or CMASS1-4, and no element forces for CBUSH or
  solids. Solid stresses and strains are in the basic system, ignoring PSOLID CORDM.
- GPFORCE is written for linear statics only.
- Shell drill rotations carry only a small stabilisation, as Nastran's K6ROT does;
  `PARAM,K6ROT`, `QUAD4TYP` and `SHELLFORM` are accepted and ignored. Second-order thickness
  terms are dropped, so very thick shells (t/L around 0.5) can be well off a 3D solution.
- A PSHELL with a blank MID2 is a membrane, as in Nastran. A blank MID1 is rejected.
- PBARL's J is the closed-form thin-wall value, up to 2% off MSC Nastran's. CELAS GE has no
  effect, since no solution computes a damped response.
- AUTOSPC looks at one grid at a time, as Nastran's does, so it misses a mechanism spanning
  grids, such as a rod spinning freely about its axis. The default backend reports one as a
  [fatal message](#fatal-messages-and-exit-codes) naming the grids; PCG and cuDSS only do
  when a load reaches it.
- `EIGRL` `V2` isn't applied yet.

## Building

```bash
meson setup build
cd build
ninja
meson test
ninja cppcheck
```

Optional backends are detected at configure time.

| Dependency | Required | Purpose |
|---|---|---|
| C++20 compiler (GCC 11+, Clang 13+) and a C11 compiler | Yes | |
| Eigen 3.4+, spdlog | Yes | Linear algebra and logging (Meson wraps) |
| METIS and AMD (`libmetis-dev`, `libsuitesparse-dev`) | Yes | Orderings for the LDLᵀ library |
| BLAS with a C interface (Accelerate, OpenBLAS) | Yes | Dense kernels for the LDLᵀ library |
| OpenMP | Yes | CPU threads |
| Spectra | Yes | CPU eigensolver for SOL 103 (bundled in `third_party/`) |
| AMGCL 1.5.0 | Yes | Downloaded by a Meson wrap |
| Google Test | Tests | Meson wrap |
| HDF5 | For magnetics | VTKHDF results; without it the magnetic solutions are unavailable |
| CHOLMOD (SuiteSparse) | No | `cpu-eigen` backend off macOS |
| TBB | No | Parallel execution |
| CUDA toolkit ≥ 11, cuDSS ≥ 0.7 | No | CUDA backends |
| Vulkan SDK, `glslc`, `xxd` | No | Vulkan backend |

On Ubuntu 24.04, install cuDSS from NVIDIA's local repo deb:

```bash
sudo dpkg -i cudss-local-repo-ubuntu2404-*.deb
sudo apt update
sudo apt install libcudss0-cuda-12 libcudss0-dev-cuda-12
```

## Usage

```
vibestran [--backend=<cpu|cpu-eigen|cpu-pcg|vulkan|cuda|cuda-pcg|cuda-pcg-mixed>]
          [--cuda-precision=<fp32|fp64>] [--csv] [--log-file=<path>]
          [--checkmode=lenient] [--no-quality-checks] [--dump-system=<prefix>]
          [--refine] [--amg-modes=<constant|rigid|drilling>]
          <input.bdf|input.inp> [output.f06]
```

The format comes from the file extension, and the output defaults to `<input>.f06`.
`vibestran` with no arguments prints what each option does.

`--dump-system=<prefix>` writes each subcase's reduced K (and M, F, u) in Matrix Market
format with an `equation grid component` map of its DOFs, as `<prefix>_sc<N>_K.mtx` and so on.

CPU backends use one thread per physical core, since a second hyperthread adds no memory
bandwidth and slows these solvers down. `OMP_NUM_THREADS` overrides it.

## Solver backends

| Backend | Method | Use |
|---|---|---|
| `cpu` (default) | Multifrontal LDLᵀ, vendored in `src/solver/ldlt` | Statics, heat transfer and magnetics; SOL 103 uses the `cpu-eigen` factorizations for now. Reports pivots, inertia and the grids of a mechanism |
| `cpu-eigen` | Apple Accelerate on macOS, else CHOLMOD, else Eigen simplicial | The factorizations used before the library. CHOLMOD and Eigen stop at a zero pivot |
| `cpu-pcg` | CG with smoothed-aggregation AMG, OpenMP | O(nnz) memory; usually faster than direct on 3D solids, slower on shells |
| `cuda` | cuDSS sparse Cholesky, LU fallback, optional fp32 factor | SOL 101 and 103. Factors larger than the GPU run in cuDSS hybrid mode from host RAM |
| `cuda-pcg` | GPU PCG with AMG (from 50,000 equations) or IC0 | SOL 101 and 103 |
| `cuda-pcg-mixed` | fp32 AMG preconditioner, fp64 Krylov | SOL 101 |
| `vulkan` | Vulkan compute PCG, experimental | SOL 101. Below 50,000 equations uses the CPU direct solver |

The PCG backends partition a matrix larger than the GPU between device and host
(`VIBESTRAN_PCG_DEVICE_MEMORY_MIB` sets the budget), and compute the finest AMG transfers
during the solve when the hierarchy doesn't fit. `--amg-modes` picks the AMG near-nullspace:
`drilling` (default) adds rotation about the shell normal on CQUAD4 grids to the rigid-body
modes, `rigid` leaves those out, and `constant` uses half the memory but fails on thin
shells and solids. It changes speed and memory, never the solution. cuDSS's device limit is
`VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB`. See [PCG_LARGE_DECKS.md](PCG_LARGE_DECKS.md).

### Solution accuracy

A direct solve is one backward-stable factorization and solve, as in Nastran. Each SOL 101
subcase logs Nastran's epsilon, u · (F − K u) / u · F, computed in compensated arithmetic,
along with the backward error. `--refine` (or `PARAM,REFINE,YES`) refines direct solves to
round-off for typically 5-10% of the solve time, which removes the cond(K)·eps error of a
single solve. PCG always refines, to an estimated relative error of 1e-8. An fp32 cuDSS
factor refines to double accuracy when cond(K) is below about 1e7, and otherwise the solve
is repeated in fp64 with a warning. Assembly is deterministic, so K is identical bit for bit between runs.

### Fatal messages and exit codes

A run that reaches a solution always writes it, since a partial answer helps find what went
wrong. What can't be trusted goes in a `FATAL MESSAGES` block at the top of the F06, and the
run exits with code 5.

| Condition | Message |
|---|---|
| A pivot comes out exactly zero | Fatal, naming the unknowns held at zero by grid and component |
| K_ii/D_i above 1e11 | Fatal: numerically singular |
| K_ii/D_i above `PARAM,MAXRATIO` (default 1e7) | Warning |
| ‖F − K u‖ / ‖F‖ above 1e-3, or \|epsilon\| above 1e-2 | Fatal: the solution does not equilibrate the load |
| An iterative solve doesn't converge, or a solve fails | Fatal |

`PARAM,MAXRATIO` sets both ratio limits. `PARAM,BAILOUT,-1` turns the pivot, ratio and
residual fatals into warnings; non-convergence and failed solves stay fatal. The pivot
checks need the default backend, since the others don't expose pivots. The modal and
magnetic solutions don't use this block yet: their failures stop the run with exit code 3.

| Exit code | Meaning |
|---|---|
| 0 | Solved |
| 1 | Bad command line, or the backend isn't available |
| 2 | The input couldn't be parsed |
| 3 | Solver error before any results |
| 4 | Any other error |
| 5 | Results written under a fatal message |

### Normal modes

The CPU eigensolver is Spectra's shift-and-invert Lanczos, factoring with Apple Accelerate
on macOS (ordering set by `VIBESTRAN_ACCELERATE_ORDER=metis|amd|default`, default `metis`)
and otherwise CHOLMOD or Eigen. CUDA uses implicitly restarted Lanczos on cuDSS with
Rayleigh-Ritz refinement. With `EIGRL` `V1` blank the shift is −1; with `V1` set it is
(2π·V1)², and the `ND` modes nearest it are reported. Each mode's residual
‖Kφ − λMφ‖ / ‖Kφ‖ is logged, and the largest goes in the F06's `SOLUTION` section.

## Architecture

```
vibestran/
├── include/, src/
│   ├── core/        # Model, DOFs, coordinate systems, quality checks
│   ├── elements/    # Shell, solid, beam, rod, bush, spring, mass and thermal elements
│   ├── io/          # BDF and INP parsers; F06, OP2 and CSV writers
│   ├── magnetic/    # Magnetic model, parser, solver and output
│   └── solver/      # Backends (src/solver/ldlt is the LDLᵀ library) and solution sequences
├── tests/
│   ├── unit/        # Per-module tests
│   ├── integration/ # Analytic and reference solutions
│   └── e2e/         # BDF decks run through the vibestran binary
└── third_party/     # Spectra
```

New element types implement `ElementBase` and new solvers `SolverBackend`. Parser, model,
solver and output are separate layers, and errors are values rather than exceptions.
