# Vibestran

A finite element solver written in modern C++20 that attempts to be
Nastran/Mystran-compatible.

## Why

I was curious how feasible it would be to create something this complex with an AI agent. It's a well documented field, but it also requires that the math is precisely correct. I also wanted to experiment with seeing how fast a FEM solver could be when written in a language I understand (well, within the limits of one's ability to understand C++), and using GPU acceleration. I was prepared to do the benchmarking and optimization myself, but it turned out that Claude was quite capable with some guidance. 

Vibestran draws significant inspiration from the [Mystran](https://github.com/MYSTRANsolver/MYSTRAN)
project. If you're unfamiliar with the Mystran project you should go look at it first. It should be considered the reference for open source Nastran-compatible solvers. Without it, as an introduction to OSS FEM Solvers, there's zero chance I would have tried this.

The OpenJFEM project, which is another solver, written in Julia, was also an inspiration for trying this with an agent.

## Performance

More coming soon. 84k elem/170k node hex mesh model solves in 2.6sec on M2 macbook air, 3.6sec on an rtx3060 12gb+ryzen 5700, and 4.3s on the ryzen 5700 alone. 

## Architecture

```
vibestran/
├── include/
│   ├── core/           # Matrix, DOF, Mesh data structures
│   ├── elements/       # Shell, solid, beam, bush, spring, and lumped-mass elements
│   ├── io/             # BDF/INP parsers, F06/OP2/CSV writers
│   ├── solver/         # Solver backends and static/modal analysis
│   └── utils/          # Logging, error handling
├── src/                # Implementations
├── tests/
│   ├── unit/           # Unit tests per module
│   ├── integration/    # Integration tests with hand-calculated solutions
│   └── e2e/            # End-to-end analysis cases (BDF + expected JSON, run against vibestran binary)
└── third_party/        # Bundled third-party headers (for example Spectra)
```

## Design Principles

- **Separation of concerns**: Parser, FE model, solver, and output are independent layers
- **Extensibility**: New element types implement `ElementBase`; new solvers implement `SolverBackend`
- **Modern C++20**: spans, ranges, strong types, error values over exceptions throughout
- **Performance**: Hash maps are wonderful. Parallel assembly and sparse solves where available, with CUDA, Vulkan, and Apple Accelerate-backed sparse factorizations.

## Supported Features

- **Static magnetics**: `SOL MAGSTAT`, tetrahedral edge elements, nonlinear B–H materials, permanent magnets, volumetric coil preprocessing and finite-section `CWIRE` sources. Includes field/linkage/inductance/force output and prepared-angle sweeps; see [MAGNETICS.md](MAGNETICS.md) for the dialect, examples and first-pass limitations. Independent comparisons stay in [Magnetic_Validation](Magnetic_Validation/README.md), outside the test suite.
- **Elements**: CQUAD4, CTRIA3, CHEXA8, CHEXA20, CTETRA4, CTETRA10, CPENTA6, CBAR, CBEAM, CBUSH, CELAS1, CELAS2, CMASS1, CMASS2, CONM2 (CG offset, inertia tensor, CID / CID = -1)
- **Shells**: CQUAD4 is the MITC4+/D continuum shell (Ko, Bathe & Zhang 2025) and CTRIA3 the MITC3+ shell (Lee, Lee & Bathe 2014), both with six DOFs per grid, shared grid directors (averaged within the `PARAM,SNORM` fold angle, default 20 degrees, or given by SNORM cards), warped and curved geometry, transverse shear, ZOFFS, grid thicknesses (T1-T4, TFLAG), THETA/MCID material axes, and thermal loads consistent with the element kinematics (free expansion of curved shells is stress-free). See `SHELL_BENCHMARKS.md`.
- **Solids**: CHEXA8 is the Q1/E21 enhanced assumed strain hexahedron (Simo & Rifai 1990, Andelfinger & Ramm 1993; the same element as MYSTRAN's CHEXA8, which it reproduces to print precision on MYSTRAN's CHEXA comparison decks): patch test on distorted meshes, no spurious modes, no preferred direction, relieved bending and volumetric locking, condensed thermal load and recovery. PSOLID ISOP = FULL (or 1) selects the standard full-integration element, IN = THREE 3x3x3 integration.
- **CHEXA20** is a projected assumed-strain serendipity hexahedron (`CHEXA20_PLAN.md`): the natural normal strains are projected onto spaces linear in their own direction (removing membrane locking on curved elements), the shears onto the trilinear space, and the dilatation by B-bar, with local natural frames and a zero-mean correction that keeps the patch test exact with curved, off-centre midside nodes. One element through the thickness gives 97.9% / 99.3% of the pinched-hemisphere reference on 8x8 / 12x12 meshes (Simcenter Nastran 96.6% / 99.0%), with no spurious modes, no preferred direction and no volumetric locking (Lame cylinder at nu = 0.4999: 99.9%). ISOP = FULL selects the standard element; CHEXA20 always uses 3x3x3 Gauss points (PSOLID IN is not used, since 2x2x2 has six spurious modes per element). PLOAD4 applies to its 8-node faces.
- **Omitted midside grids**: any midside grid of a CHEXA20 or CTETRA10 may be left blank, as in Nastran. That edge is then linear (the grid's shape function is folded into the edge's corners, `elements/midside_nodes.hpp`), so CHEXA20/CHEXA8 and CTETRA10/CTETRA4 transitions are compatible; PLOAD4 on a face with an omitted grid folds its load the same way.
- **Materials**: MAT1 (all elements), MAT2 and MAT8 (shells), plus parser/model support for MAT3, MAT4, MAT5, and MAT6
- **Properties**: PSHELL (MID1-MID4, 12I/T^3, TS/T, NSM, Z1/Z2), PCOMP (classical lamination theory; LAM blank, SYM, MEM, BEND, SMEAR, SMCORE), PSOLID, PBAR (stress recovery points C-F, K1/K2 transverse shear, I12), PBARL (`ROD`, `TUBE`, `BAR`), PBEAM, PBUSH (`K` section), PELAS, PMASS
- **CBAR end data**: pin flags (PA/PB, statically condensed), rigid end offsets (WA/WB), OFFT `GGG`/`BGG`
- **Loads**: FORCE, MOMENT, TEMP/TEMPD, PLOAD, PLOAD1 (CBAR/CBEAM), PLOAD2, PLOAD4, GRAV, ACCEL1
- **Constraints**: SPC, SPC1, GRID PS (permanent SPCs, in the grid's CD frame), MPC, RBE2, RBE3
- **Defaults**: GRDSET (CP/CD/PS for blank GRID fields), BAROR (PID/orientation/OFFT for blank CBAR fields)
- **Reaction output**: SPCFORCE(S) and MPCFORCE(S) in F06, in each grid's global (CD) system (MYSTRAN-compatible blocks)
- **Shell strains**: STRAIN for CQUAD4/CTRIA3 in STRCUR (membrane strain + curvature, default) or FIBER form, von Mises or SHEAR/MAXS, centre and CORNER
- **Element forces**: FORCE/ELFORCE for CBAR/CBEAM, CQUAD4 (centre and CORNER/BILIN) and CTRIA3, Nastran sign convention (MYSTRAN-compatible blocks)
- **Grid point force balance**: GPFORCE for linear statics (applied, thermal, SPC, MPC, and per-element rows; MYSTRAN layout)
- **Coordinate systems**: CORD1R/C/S, CORD2R/C/S
- **Input formats**: Nastran BDF (`.bdf`) and CalculiX/Abaqus (`.inp`, experimental)
- **Solutions**: SOL 101 (Linear Static), SOL 103 (Normal Modes / Modal Analysis)
- **Mass**: consistent (default) or lumped with `PARAM,COUPMASS` <= 0 (HRZ diagonal lumping for continuum elements; translational end masses for CBAR/CBEAM); massless DOFs are condensed exactly in the eigensolver

Notable current limitations:

- `ACCEL` is parsed but not yet applied in the static solver.
- PLOAD1 support is limited to CBAR and CBEAM.
- Stress recovery is not yet implemented for CBUSH, CELAS1/2, and CMASS1/2.
- CBAR offsets in the element system (OFFT `xGO`/`xOG`/`xOO`) are not yet supported.
- GRID and GRDSET SEID (superelements) are not supported.
- Element forces are not yet written for CBUSH, CELAS, or solids; solid stress/strain output ignores PSOLID CORDM (always basic).
- GPFORCE is written for linear statics only (no modal inertia-force rows yet).
- Shell drill rotations carry only a small stabilisation (like Nastran's K6ROT); `PARAM,K6ROT`, `QUAD4TYP`, and `SHELLFORM` are accepted and ignored.
- PCOMP ply stresses and strains (COMPOSITESTRESSES), CQUAD8, and CTRIA6 are not implemented; PCOMP is not used in heat transfer.
- Shell sections are resultant (ABD) sections with the first-order through-thickness metric (z/R) terms; second-order thickness terms are dropped, so very thick shells (t/L around 0.5) can be well off a 3D solution (see `SHELL_BENCHMARKS.md`).
- Solids accept only MAT1 (no MAT9 anisotropy). The 15-node CPENTA is not implemented.
- Mass is consistent by default (unlike Nastran, whose default is lumped); `PARAM,COUPMASS,-1` selects lumped mass.

## SOL 103 — Normal Modes (Modal Analysis)

Compute natural frequencies and mode shapes using a generalized eigensolver. Activated
by `SOL 103` in the BDF case control deck.

```
vibestran model.bdf
```

Eigensolvers are selected via `--backend`:

| Backend | Eigensolver | Notes |
|---|---|---|
| `cpu` (default) | Spectra shift-and-invert + sparse direct solve | Uses Apple Accelerate on macOS when available, otherwise CHOLMOD/Eigen |
| `cuda` / `cuda-pcg` | Implicitly Restarted Lanczos (IRL) with Rayleigh-Ritz refinement | GPU-accelerated; requires CUDA + cuDSS |

On macOS, the CPU modal path prefers Apple Accelerate for the shift-invert
factorization and triangular solves used inside Spectra. The Accelerate sparse
ordering can be controlled with `VIBESTRAN_ACCELERATE_ORDER=metis|amd|default`;
the default is `metis`.

The CUDA eigensolver uses cuDSS for the shift-and-invert linear solves inside
the Lanczos iteration, then applies a Rayleigh-Ritz post-refinement step to
improve accuracy of the converged eigenpairs.

## Building

```bash
meson setup build
cd build
ninja
meson test
ninja cppcheck
```

Optional solver backends are detected automatically at configure time.

## Usage

```
vibestran [--backend=<cpu|cpu-pcg|vulkan|cuda|cuda-pcg|cuda-pcg-mixed>]
         [--cuda-precision=<fp32|fp64>] [--csv]
         [--log-file=<path>] [--dump-system=<prefix>] [--refine]
         <input.bdf|input.inp> [output.f06]
```

Both Nastran BDF (`.bdf`) and CalculiX/Abaqus (`.inp`) input files are supported.
The format is detected by file extension. If no output path is given, it defaults
to `<input>.f06`.

### CalculiX/Abaqus `.inp` support (experimental)

The `.inp` parser accepts standard CalculiX input files and maps them to the same
internal model used by the BDF parser. Supported keywords include `*NODE`,
`*ELEMENT`, `*MATERIAL`/`*ELASTIC`/`*DENSITY`/`*EXPANSION`, `*SOLID SECTION`,
`*SHELL SECTION`, `*BOUNDARY`, `*CLOAD`, `*TEMPERATURE`, `*STEP`/`*STATIC`/`*END STEP`,
and output request keywords (`*NODE FILE`, `*EL FILE`, `*NODE PRINT`, `*EL PRINT`).

Supported element types: C3D8, C3D4, C3D10, C3D6, S4, S3 (and their R variants).
C3D20 is not currently supported.

See `include/io/inp_parser.hpp` for full details on design decisions and limitations.

### Output files

| File | Format | Written when |
|---|---|---|
| `<stem>.f06` | F06 text | Always |
| `<stem>.op2` | OP2 binary | Always |
| `<stem>.node.csv` | Nodal results CSV | `--csv` flag or `PARAM,CSVOUT,YES` in BDF |
| `<stem>.elem.csv` | Element results CSV | `--csv` flag or `PARAM,CSVOUT,YES` in BDF |

### Dumping the linear system

`--dump-system=<prefix>` writes each subcase's reduced system in Matrix Market
format (1-based, 17 significant digits) for inspection in SciPy, MATLAB or
Julia:

| File | Contents |
|---|---|
| `<prefix>_sc<N>_K.mtx` | Reduced stiffness (`coordinate real symmetric`, lower triangle) |
| `<prefix>_sc<N>_F.mtx` | Reduced load vector (SOL 101) |
| `<prefix>_sc<N>_u.mtx` | Reduced solution (SOL 101) |
| `<prefix>_sc<N>_M.mtx` | Reduced mass (SOL 103) |
| `<prefix>_sc<N>_dofs.txt` | `equation grid component` map of the reduced DOFs |

SPC'd and MPC-dependent DOFs are not in the reduced system.

## Solver Backends

### Solution accuracy

A direct solve is a single backward-stable factorization and solve, as in
Nastran, and is assessed with a residual computed in compensated
(double-double) arithmetic. Each SOL 101 subcase logs:

- **epsilon** = u · (F − K u) / u · F, Nastran's load-sequence epsilon. It roughly
  tracks the relative error of the solution: below 1e-10 for ordinary models,
  ~1e-4 for a thin CHEXA8 shell with cond(K) ~ 6e14, and O(1) for a loaded
  mechanism. |epsilon| > 1e-2 is reported as a mechanism (results not
  meaningful), |epsilon| > 1e-6 as an ill-conditioned model.
- the normwise **backward error**, which is round-off for any stable solve;
- the relative residual ||F − K u|| / ||F||, which is not an accuracy measure:
  for an ill-conditioned K it is ~ eps ||K|| ||u|| / ||F|| even for the exact
  solution rounded to double.

`--refine` (or `PARAM,REFINE,YES` in the bulk data) refines direct solves with
their factorization and compensated residuals until the corrections reach
round-off, which removes the cond(K) * eps error of a single solve (thin
hemisphere: 4e-4 to 1e-14). It costs a few triangular solves, typically 5-10%
of the solve time, and the log then reports the estimated relative error.
The PCG backends and fp32 factorizations always refine: that is how they reach
double accuracy at all. Assembly is deterministic, so K is identical bit for
bit between runs.

### CPU — Cholesky (default)

Sparse direct CPU solve. Always available, no extra dependencies. Uses Apple
Accelerate on macOS when available, SuiteSparse CHOLMOD when available on other
platforms, and otherwise falls back to Eigen's simplicial factorizations.

```bash
vibestran --backend=cpu model.bdf
```

On macOS, the Accelerate sparse ordering can be selected with:

```bash
VIBESTRAN_ACCELERATE_ORDER=metis vibestran --backend=cpu model.bdf
```

The default is METIS, except for a MAGSTAT model with an air-gap element
(PAIRGAP AGE), whose dense block every direct backend orders with minimum
degree (AMD): nested dissection cuts through the sparse rest and roughly
triples the factorization's work. CHOLMOD also stops setting the block's
dense rows aside in AMD, which otherwise inflates AMD's fill estimate and
makes CHOLMOD pick METIS; cuDSS uses its AMD reordering. The F06 header's
Ordering row reports the ordering a MAGSTAT solve used.

### CPU — PCG

Eigen Preconditioned Conjugate Gradient with Incomplete Cholesky preconditioning.
Iterative solver with O(nnz) memory — suitable for very large systems where the
direct Cholesky fill-in would exhaust RAM. This is slower than the direct (Cholesky) solution.
The solve is refined until the estimated relative error is below 1e-8; a model
too ill-conditioned for incomplete-Cholesky PCG (e.g. a thin solid shell, with
cond(K) ~ 1e14) is rejected with an error rather than returned inaccurate.

```bash
vibestran --backend=cpu-pcg model.bdf
```

### CUDA — cuDSS

Uses [NVIDIA cuDSS](https://developer.nvidia.com/cudss) for GPU-resident sparse
direct solve. Requires CUDA toolkit ≥ 11 and cuDSS ≥ 0.7. Also used as the
shift-and-invert linear solver inside the IRL eigensolver for SOL 103.

**Solver strategy:**
1. Sparse Cholesky (`CUDSS_MTYPE_SPD`) — optimal for SPD FEM stiffness matrices
2. Sparse LU (`CUDSS_MTYPE_GENERAL`) fallback when the Cholesky factorization
   fails or its solution cannot be refined
3. With `--refine`, the factorization is kept and the solution refined on the
   host; a singular stiffness (whose LU solution does not equilibrate the
   load) is an error

**Memory management:**
- After symbolic analysis, cuDSS memory estimates are logged (device/host stable and peak).
- If the estimated device peak exceeds 85% of free GPU memory, hybrid host/device memory
  mode is enabled automatically so factor storage spills to host RAM.
- If analysis itself fails with an allocation error, hybrid mode is retried before giving up.
- In theory Hybrid mode allows for very large systems to be solved, but in practice that hasn't been achieved (yet).

```bash
vibestran --backend=cuda model.bdf
```

The cuDSS direct backend also supports a float32 factorization for reduced
VRAM usage. Refinement with double-precision residuals recovers double accuracy
when cond(K) is below about 1e7; for a more ill-conditioned model (most shell
and thin-solid models) the solve is repeated in fp64 with a warning:

```bash
vibestran --backend=cuda --cuda-precision=fp32 model.bdf
```

#### Installing cuDSS (Ubuntu 24.04)

Download the deb from Nvidia, then install:
```bash
# Add the local repo and install
sudo dpkg -i cudss-local-repo-ubuntu2404-*.deb
sudo apt update
sudo apt install libcudss0-cuda-12 libcudss0-dev-cuda-12
```

### CUDA — PCG

GPU Preconditioned Conjugate Gradient with IC0 → ILU0 → Jacobi fallback preconditioning
(IC0 is retried on K + α diag(K) with a growing shift α when it meets a
non-positive pivot, as shell stiffness matrices often cause),
using cuSPARSE SpSV for triangular solves and cuBLAS for vector operations.
O(nnz) device memory — no factorization fill-in — making it suitable for systems too large
for cuDSS even with hybrid host/device memory mode. This is slower than the cuDSS direct solve for most cases.
Like the CPU PCG it is refined to an estimated relative error of 1e-8 and fails
with an error, not an inaccurate answer, when it cannot get there.
`--backend=cuda-pcg-mixed` runs the PCG in fp32 inside the same refinement, which
only converges for cond(K) below about 1e7.

```bash
vibestran --backend=cuda-pcg model.bdf
```

Requires CUDA toolkit ≥ 11 with cuBLAS and cuSPARSE (both included in the standard toolkit).

[CUDA_PCG_PERFORMANCE.md](CUDA_PCG_PERFORMANCE.md) profiles this backend
against the direct ones: the IC0 triangular solves dominate, and each solve
rebuilds the preconditioner.

### Vulkan - Experimental

PCG iterative solver using Vulkan compute shaders. Requires Vulkan SDK, `glslc`, and `xxd`. Unfortunately performance is terrible right now, it's orders of magnitude slower than the base Eigen Cholesky CPU solver.
The fp32 Jacobi PCG is refined with double-precision residuals to an estimated
relative error of 1e-8, retried in fp64 if that fails (when the GPU supports
it). Systems below 50,000 DOFs use the CPU direct solver.

```bash
vibestran --backend=vulkan model.bdf
```

## Dependencies

| Dependency | Required | Purpose |
|---|---|---|
| C++20 compiler (GCC 11+, Clang 13+) | Yes | |
| Eigen 3.4+ | Yes | Core linear algebra and sparse CPU backends |
| Spectra | Yes | CPU generalized eigensolver for SOL 103 |
| TBB | No | Optional parallel execution support |
| Google Test | Yes (tests) | Fetched by Meson WrapDB |
| CUDA toolkit ≥ 11 + cuDSS ≥ 0.7 | No | CUDA backend |
| Vulkan SDK + glslc + xxd | No | Vulkan backend |
