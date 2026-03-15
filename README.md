# NastranSolver

A Nastran-compatible finite element solver written in modern C++20.

## Architecture

```
nastran_solver/
├── include/
│   ├── core/           # Matrix, DOF, Mesh data structures
│   ├── elements/       # Element formulations (CQUAD4, CTRIA3, CHEXA, CTETRA)
│   ├── io/             # BDF parser, F06/OP2 writer
│   ├── solver/         # Linear solver, load assembly
│   └── utils/          # Logging, error handling
├── src/                # Implementations
├── tests/
│   ├── unit/           # Unit tests per module
│   └── integration/    # End-to-end analysis cases
└── third_party/        # Eigen (header-only)
```

## Design Principles

- **Separation of concerns**: Parser, FE model, solver, and output are independent layers
- **Extensibility**: New element types implement `ElementBase` interface; new solvers implement `SolverBase`
- **CUDA/Vulkan readiness**: Matrix assembly and solve are abstracted behind interfaces; memory layouts use contiguous storage suitable for GPU transfer
- **Modern C++20**: Concepts, spans, ranges, strong types throughout

## Supported Features (v1)

- **Elements**: CQUAD4, CTRIA3, CHEXA8, CTETRA4
- **Materials**: MAT1 (isotropic)
- **Properties**: PSHELL, PSOLID
- **Loads**: FORCE, MOMENT, TEMP (thermal)
- **Constraints**: SPC, SPC1
- **Solution**: SOL 101 (Linear Static)

## Building

```bash
meson setup build
cd build
meson compile
meson test
```

## Dependencies

- C++20 compiler (GCC 11+, Clang 13+, MSVC 2022+)
- Eigen 3.4 (header-only, included in third_party/)
- Google Test (fetched by Meson WrapDB)

## Future Acceleration

The `SolverBackend` interface is designed for:
- **CUDA**: Replace `EigenSolverBackend` with `CudaSolverBackend` using cuSPARSE/cuSOLVER
- **Vulkan Compute**: Replace with `VulkanSolverBackend` using compute shaders for matrix assembly
