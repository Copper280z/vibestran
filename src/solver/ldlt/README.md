<!-- Vendored from the ldlt library, branch perf-pass, commit 95a6ad4. Only the sources and the
     tests that need no fixtures are here; src/solver/ldlt/meson.build replaces its build. -->
# ldlt - supernodal sparse LDL^T solver (C11)

Three-phase API: `ldlt_analyze` -> `ldlt_factorize` -> `ldlt_solve`. The analysis can be reused
for any matrix with the same pattern. Input is the lower triangle in CSC, with strictly
increasing rows in each column.

## How it works

The factorization is multifrontal. Each supernode assembles a dense front from its entries of A
and its children's contribution blocks, eliminates its columns, and passes the rest of the front
to its parent. The analysis never forms the pattern of L: column counts come from the
elimination tree, and each supernode's rows from its children's.

Parallelism has two phases. Independent subtrees run on OpenMP threads with a single-threaded
BLAS; the few large fronts at the top of the tree then run one at a time with a threaded BLAS.
The library sets the BLAS thread count itself for OpenBLAS (globally) and Accelerate (per thread,
macOS 15 or later). With any other threaded BLAS, limit it to one thread yourself or the subtree
phase will oversubscribe.

### Without pivoting (`LDLT_PIVOT_NONE`, the default)

For matrices that don't need pivoting, positive definite ones above all. Each front tries Cholesky
and falls back to an unpivoted LDL^T if that fails, so this also handles matrices that are
indefinite but stable without pivoting. Nothing checks that stability, so use the pivoting path
if you don't know: an unstable factorization can return wrong signs in D and a bad solution
without any error. `require_pd` returns `LDLT_ERR_INDEFINITE` on any pivot that isn't positive.

A pivot that comes out exactly zero is held: that unknown is fixed at zero, as if constrained,
and everything else is solved exactly as if its row and column had been deleted. `ldlt_get_held`
lists those rows. Holding a pivot says nothing about whether a load on that unknown is carried, so
check the residual of the solve if the matrix may be singular. A pivot that is only nearly zero
isn't held: look at `max_ratio` from `ldlt_get_info`, the largest |A_ii / D_i|.

### With pivoting (`LDLT_PIVOT_THRESHOLD`)

For indefinite matrices. The matrix is scaled symmetrically, then each pivot must keep entries of L
below 1/u, where u is `pivot_threshold` (default 0.01). Columns go through in blocks that are tried
without pivoting and checked afterwards. A block that fails takes 1x1 and 2x2 pivots one at a time.
A column with no acceptable pivot in its front is delayed to the parent's front, and at a root the
largest remaining entries are used. A column that is negligible throughout is held.

Delays add fill that the analysis can't predict, so `ldlt_lnz` is only the unpivoted figure; use
`lnz` from `ldlt_get_info` for what was stored. The smaller u is, the less fill and time and the
worse the backward error. Entries of L up to 100 can cost a couple of digits, so check the residual
and refine if you need them back. Positive definite matrices also work, at some extra cost, and
can take delays they don't need if their diagonal varies widely.

### Results

`ldlt_get_info` gives inertia (by Sylvester's law, the signs of the matrix's eigenvalues), counts of
held, 2x2 and delayed pivots, stored entries, and the largest |A_ii / D_i| with its row.
`ldlt_get_perm` gives the elimination order and `ldlt_get_d` gives D by row of A, with scaling
undone and each 2x2 pivot's partner.

## Meson Build

```sh
meson setup build
meson compile -C build
```

Tests and benchmarks are opt-in:

```sh
meson setup build-dev -Dwith_tests=true -Dwith_bench=true
meson compile -C build-dev
meson test -C build-dev
build-dev/bench/ldlt_bench_spd
build-dev/bench/ldlt_bench_pivot
```

Options (`meson configure build -Dkey=value`):
- `blas=auto|system|reference` - default auto (Accelerate on macOS, then cblas/openblas/blas; reference fallback).
- `ordering=auto|amd|metis|internal` - which of SuiteSparse AMD and METIS to look for. `internal` builds with neither. At run time, `LDLT_ORDER_AUTO` (the default) uses METIS, then AMD, then natural order, and asking for one that wasn't built returns `LDLT_ERR_UNSUPPORTED`. `ldlt_ordering_used` says which ran.
- `with_openmp=true|false`.
- `with_fortran_interface=true|false` - builds `ldlt_fortran`, the QDLDL-compatible Fortran bridge.
- `with_tests=true|false`.
- `with_bench=true|false`.
- `large_matrices=true` - enable heavier SuiteSparse fixtures.

## CMake Build

```sh
cmake -S . -B build-cmake -DLDLT_BLAS=AUTO -DLDLT_ORDERING=AUTO
cmake --build build-cmake
```

Useful options:
- `LDLT_BLAS=AUTO|SYSTEM|REFERENCE`.
- `LDLT_ORDERING=AUTO|AMD|METIS|INTERNAL`.
- `LDLT_ENABLE_OPENMP=ON|OFF`.
- `LDLT_BUILD_FORTRAN_INTERFACE=ON|OFF`.
- `LDLT_BUILD_TESTS=ON|OFF`.
- `LDLT_BUILD_BENCH=ON|OFF`.

## Test fixtures

`python3 matrices/fetch_suitesparse.py` downloads .mtx files listed in `matrices/manifest.txt`.
Set `LDLT_FIXTURE_DIR=$PWD/matrices` so the parameterized SuiteSparse tests pick them up.
