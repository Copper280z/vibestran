# CUDA PCG performance review (2026-11-01)

`--backend=cuda-pcg` (`src/solver/cuda_pcg_backend_impl.hpp`) against the
direct backends, measured on a MAGSTAT tangent. The findings are about the
backend's structure, not the physics, so they should apply to structural,
thermal and MAGSTAT solves alike. Only the MAGSTAT matrix was profiled;
structural and thermal matrices may change the numbers, not the order of the
costs.

## Setup

- bob-desktop: Ryzen 7 5700 (8 cores, pinned with `taskset -c 0-7`), NVIDIA
  TITAN V, CUDA 12, cuDSS 0.7.1, SuiteSparse CHOLMOD 5.2 with OpenBLAS.
- The coarse whole motor of the xfemm comparison
  (`Magnetic_Validation/run_xfemm.py`) with `PARAM,AIRGAP,BAND` and each
  MOTION cut to two positions: 6 positions, 75 Newton solves of a tangent
  with n = 12,509 and 58,195 lower nonzeros (high permeability contrast:
  steel against air and magnets).
- `VIBESTRAN_CUDA_PCG_PROFILE=1` prints per-phase GPU timings for each solve.

## Results

| backend | wall time |
|---|---:|
| `cpu` (CHOLMOD) | 1.8 s |
| `cuda` (cuDSS) | 1.7 s |
| `cuda-pcg` (IC0) | 148 s |
| `cuda-pcg`, Jacobi forced (temporary patch) | 14.6 s |

One IC0 solve to the 1e-8 tolerance, 350 iterations:

| phase | per iteration |
|---|---:|
| IC0 forward triangular solve (`cusparseSpSV`) | 2.05 ms |
| IC0 backward triangular solve (`cusparseSpSV`, transposed) | 2.01 ms |
| SpMV, lower triangle | 0.026 ms |
| SpMV, transposed lower triangle | 0.015 ms |
| each dot product, axpy or scalar kernel | 0.005-0.008 ms |

IC0 setup per solve: 53 ms (7 ms factor analysis, 5 ms factor, 26 + 8 ms
triangular-solve analyses).

## Findings

1. **The IC0 triangular solves are 98% of each iteration.** cuSPARSE's
   SpSV runs a triangular solve level by level. An incomplete factor in the
   mesh's own numbering has long dependency chains, so a 12.5k-row solve is
   mostly sequential: 2 ms against 0.026 ms for a multiply. Forced Jacobi
   needs 3 times the iterations (1,081 against 350) but each costs 0.13 ms
   instead of 4.2 ms, so the run is 10 times faster. IC0 is only worthwhile
   once its triangular solves parallelize:
   - a multicolor (or other level-reducing) reordering before IC0, which
     trades some convergence for a handful of levels; or
   - a preconditioner applied by SpMV-like kernels: algebraic multigrid
     (for example AmgX), a polynomial (Chebyshev) preconditioner, or an
     approximate inverse.
2. **Every solve starts from scratch.** The backend has no `factor()` of its
   own, so `SolverBackend::factor()` adapts it: each right-hand side is a
   full `solve(K, b)` that uploads K, rebuilds IC0 (53 ms here) and redoes
   the SpSV analyses. A nonlinear solve (MAGSTAT Newton, a future geometric
   nonlinear structural solve) factors the same pattern many times. A
   `CudaPCGFactor` should keep the device matrix and the analyses and
   refactor numerically only.
3. **Each solve runs about 1.8 times the iterations it needs.**
   `solve_cuda_pcg_stable` solves to the tolerance and then refines with
   corrections to a loose inner tolerance; here 350 + 20 + 270 iterations
   for one right-hand side. A Newton step needs only an inexact solve
   (Eisenstat-Walker forcing): early iterations can stop at 1e-2 relative
   residual. The linear-solver interface should take the tolerance from the
   caller.
4. **About 0.1 ms per iteration is overhead.** Jacobi's GPU-accounted time
   is 95 us per iteration for a few microseconds of arithmetic: about 12
   kernel launches and one host synchronization (the convergence flag
   download). At n = 10^4-10^5 the GPU is latency-bound. Fusing the vector
   updates and dot products into a few kernels, checking convergence every
   k iterations, or capturing an iteration in a CUDA graph would remove most
   of it.
5. **Lower-triangle storage needs a transposed SpMV.** K p is computed as
   L p + L^T p - D p. The transposed multiply is cheap here (0.015 ms), but
   for large matrices storing the full matrix (twice the memory) or a CSC
   copy avoids cuSPARSE's slower transposed algorithm.

## What this means

At the sizes of planar MAGSTAT models (10^4-10^5 unknowns), even an ideal
GPU PCG is unlikely to beat a direct factorization, which takes tens of
milliseconds once the matrix is sparse (`PAIRGAP METHOD=BAND`). PCG pays
off where factoring is the cost: large 3D models, or many factorizations of
slowly changing matrices. The second case is the reused-factor PCG: CG on
the current tangent, preconditioned by a direct factorization of an earlier
one, refactoring only when CG slows down (`TangentSolver`,
`include/solver/tangent_solver.hpp`, `PARAM,MAGPCG` for MAGSTAT). Its CUDA
form can start from this backend's device-side CG loop (device pointer mode,
device-resident scalars, one flag download per iteration) with cuDSS's solve
as the preconditioner instead of IC0; today `TangentSolver` runs CG on the
host and calls the factor's solve, a round trip to the device per iteration.
