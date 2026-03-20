# NastranSolver

A Nastran-compatible finite element solver written in modern C++20.

## Architecture

```
nastran_solver/
├── include/
│   ├── core/           # Matrix, DOF, Mesh data structures
│   ├── elements/       # Element formulations (CQUAD4, CTRIA3, CHEXA, CTETRA, CPENTA)
│   ├── io/             # BDF/INP parsers, F06/OP2/CSV writers
│   ├── solver/         # Solver backends and linear static analysis
│   └── utils/          # Logging, error handling
├── src/                # Implementations
├── tests/
│   ├── unit/           # Unit tests per module
│   └── integration/    # End-to-end analysis cases
└── third_party/        # Eigen (header-only)
```

## Design Principles

- **Separation of concerns**: Parser, FE model, solver, and output are independent layers
- **Extensibility**: New element types implement `ElementBase`; new solvers implement `SolverBackend`
- **Modern C++20**: spans, ranges, strong types, error values over exceptions throughout

## Supported Features

- **Elements**: CQUAD4, CTRIA3, CHEXA8, CHEXA20, CTETRA4, CTETRA10, CPENTA6
- **Materials**: MAT1 (isotropic)
- **Properties**: PSHELL, PSOLID
- **Loads**: FORCE, MOMENT, TEMP (thermal)
- **Constraints**: SPC, SPC1, MPC, RBE2, RBE3
- **Coordinate systems**: CORD1R/C/S, CORD2R/C/S
- **Input formats**: Nastran BDF (`.bdf`) and CalculiX/Abaqus (`.inp`, experimental)
- **Solution**: SOL 101 (Linear Static)

## Building

```bash
meson setup build
cd build
ninja
meson test
```

Optional backends are detected automatically at configure time.

## Usage

```
nastran_solver [--backend=<cpu|cpu-pcg|vulkan|cuda|cuda-pcg>]
               [--cuda-single-precision] [--csv]
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
C3D20 is not supported due to midside node ordering differences.

See `include/io/inp_parser.hpp` for full details on design decisions and limitations.

### Output files

| File | Format | Written when |
|---|---|---|
| `<stem>.f06` | F06 text | Always |
| `<stem>.op2` | OP2 binary | Set via case control deck |
| `<stem>.node.csv` | Nodal results CSV | `--csv` flag or `PARAM,CSVOUT,YES` in BDF |
| `<stem>.elem.csv` | Element results CSV | `--csv` flag or `PARAM,CSVOUT,YES` in BDF |

## Solver Backends

### CPU — Cholesky (default)

Eigen sparse Cholesky. Always available, no extra dependencies.

```bash
nastran_solver --backend=cpu model.bdf
```

### CPU — PCG

Eigen Preconditioned Conjugate Gradient with Incomplete Cholesky preconditioning.
Iterative solver with O(nnz) memory — suitable for very large systems where the
direct Cholesky fill-in would exhaust RAM.

```bash
nastran_solver --backend=cpu-pcg model.bdf
```

### CUDA — cuDSS (recommended for large models)

Uses [NVIDIA cuDSS](https://developer.nvidia.com/cudss) for GPU-resident sparse
direct solve. Requires CUDA toolkit ≥ 11 and cuDSS ≥ 0.7.

**Solver strategy:**
1. Sparse Cholesky (`CUDSS_MTYPE_SPD`) — optimal for SPD FEM stiffness matrices
2. Sparse LU (`CUDSS_MTYPE_GENERAL`) fallback for non-SPD or ill-conditioned matrices

**Memory management:**
- After symbolic analysis, cuDSS memory estimates are logged (device/host stable and peak).
- If the estimated device peak exceeds 85% of free GPU memory, hybrid host/device memory
  mode is enabled automatically so factor storage spills to host RAM.
- If analysis itself fails with an allocation error, hybrid mode is retried before giving up.

```bash
nastran_solver --backend=cuda model.bdf
```

**Single-precision mode** halves GPU memory usage by downcasting to float32 before the
solve and upcasting the result. Useful for very large models that exhaust device memory
even with hybrid mode. Applies to both `--backend=cuda` and `--backend=cuda-pcg`:

```bash
nastran_solver --backend=cuda --cuda-single-precision model.bdf
```

#### Installing cuDSS (Ubuntu 24.04)

```bash
# Add the local repo and install
sudo dpkg -i cudss-local-repo-ubuntu2404-*.deb
sudo apt update
sudo apt install libcudss0-cuda-12 libcudss0-dev-cuda-12
```

### CUDA — PCG

GPU Preconditioned Conjugate Gradient with IC0 → ILU0 → Jacobi fallback preconditioning,
using cuSPARSE SpSV for triangular solves and cuBLAS for vector operations.
O(nnz) device memory — no factorization fill-in — making it suitable for systems too large
for cuDSS even with hybrid host/device memory mode.

```bash
nastran_solver --backend=cuda-pcg model.bdf
```

**Single-precision mode** halves VRAM usage by performing the entire PCG solve in float32:

```bash
nastran_solver --backend=cuda-pcg --cuda-single-precision model.bdf
```

Requires CUDA toolkit ≥ 11 with cuBLAS and cuSPARSE (both included in the standard toolkit).

### Vulkan

PCG iterative solver using Vulkan compute shaders. Requires Vulkan SDK, `glslc`, and `xxd`.

```bash
nastran_solver --backend=vulkan model.bdf
```

## Dependencies

| Dependency | Required | Purpose |
|---|---|---|
| C++20 compiler (GCC 11+, Clang 13+) | Yes | |
| Eigen 3.4 | Yes | CPU sparse Cholesky, matrix assembly |
| TBB | Yes | Parallel element assembly |
| Google Test | Yes (tests) | Fetched by Meson WrapDB |
| CUDA toolkit ≥ 11 + cuDSS ≥ 0.7 | No | CUDA backend |
| Vulkan SDK + glslc + xxd | No | Vulkan backend |
