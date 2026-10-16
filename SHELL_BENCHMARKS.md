# Shell Benchmarks

`tests/integration/test_shell_benchmarks.cpp` generates the standard shell
benchmark decks (MacNeal & Harder 1985; Ko, Lee, Lee & Bathe 2017,
Comput Struct 193:187-206) and checks each result against the published
reference. Meshes and tolerances follow the MITC4+/MITC3+ tables in Ko et al.
Run with `./integration_tests --gtest_filter='ShellBenchmark.*'`; each test
prints a `[benchmark]` line, and `SHELL_BENCH_DUMP=<dir>` writes the decks.

## Baseline: pre-rewrite elements (2026-10-16)

Flat CQUAD4 ("MITC4" adaptation) and CST + DKT CTRIA3. Error relative to the
reference. Tests that fail on this baseline were committed `DISABLED_` until
the new elements replace these.

| Benchmark | Mesh | CQUAD4 | CTRIA3 | Reference |
|---|---|---:|---:|---|
| Scordelis-Lo roof | 16×16 q / 32×32 t | 0.75% | 1.53% | 0.3024 |
| Pinched cylinder | 32×32 | 0.82% | 0.08% | 1.8248e-5 |
| Pinched hemisphere (18° hole) | 16×16 | 3.15% | 7.04% | 0.093 |
| Twisted beam t = 0.32, in-plane | 2×12 q / 8×48 t | 61.4% | 39.1% | 5.424e-3 |
| Twisted beam t = 0.32, out-of-plane | 2×12 q / 8×48 t | 47.5% | 13.7% | 1.754e-3 |
| Twisted beam t = 0.0032, in-plane | 2×12 q / 8×48 t | 99.99% (locked) | 77.7% | 5.256e-3 |
| Twisted beam t = 0.0032, out-of-plane | 2×12 q / 8×48 t | 99.98% (locked) | 71.9% | 1.294e-3 |
| Clamped plate, a/t = 1000 | 8×8 quarter | 0.42% | 0.54% | 1.267e-3 pL⁴/D |
| Simply supported plate, a/t = 1000 | 8×8 quarter | 0.12% | 0.34% | 4.062e-3 pL⁴/D |
| Cook's membrane | 16×16 | 1.30% | 0.62% | 23.96 |
| Free thermal expansion, faceted ring | 24 facets | exact | exact | zero stress |

## CQUAD4 = MITC4+/D (2026-10-16)

| Benchmark | Mesh | Before | MITC4+/D | Ko et al. MITC4+ on that mesh |
|---|---|---:|---:|---:|
| Scordelis-Lo roof | 16×16 | 0.75% | 0.93% | 0.27% |
| Pinched cylinder | 32×32 | 0.82% | 1.21% | 1.29% |
| Pinched hemisphere | 16×16 | 3.15% | 0.02% | 0.3% |
| Twisted beam t = 0.32, in-plane | 2×12 | 61.4% | 0.49% | 0.58% |
| Twisted beam t = 0.32, out-of-plane | 2×12 | 47.5% | 1.69% | 1.68% |
| Twisted beam t = 0.0032, in-plane | 2×12 | 99.99% | 0.34% | 0.3% |
| Twisted beam t = 0.0032, out-of-plane | 2×12 | 99.98% | 1.28% | 1.26% |
| Clamped plate, a/t = 1000 | 8×8 | 0.42% | 0.42% | |
| Simply supported plate, a/t = 1000 | 8×8 | 0.12% | 0.12% | |
| Cook's membrane (mid right edge) | 16×16 | 1.30%* | 0.46% | 0.72% (2D-MITC4) |
| Free thermal expansion, faceted ring | 24 facets | exact | exact | |

\* The baseline table measured Cook's membrane at the top corner; the
reference 23.96 is the mid-point of the right edge (the plain Q4 values of
Ko et al. 2017, C&S 192, Table 5, are reproduced exactly there), and the
benchmark now measures there.

The MYSTRAN Raasch hook deck (`raasch_hook_k6rot_1000.0`) now converges to
the published 0.1275 m (5.02 in): 0.1175, 0.1218, 0.1258, 0.1272 m for 9,
18, 72 and 144 elements along the hook.

## Through-thickness metric (2026-10-16)

Both elements map the strain to the local Cartesian frame with the metric
of each fibre, to first order in the thickness coordinate: the curvature
includes the mid-surface strain carried through the changing contravariant
basis, and the section includes the volume factor (1 + 2Hz).  This is the
first-order content of the 3D-integrated formulations of Ko et al.; without
it the hoop curvature of an expanding cylinder has the wrong sign and
membrane-bending coupling of curved and twisted shells is lost.  The tables
above include it (the thin benchmarks moved by at most 0.12 points).

A twisted plate under in-plane tension (MYSTRAN `NAS S30 twisted plate
mit4+`, corner TZ, 1 x 1, twist +-0.1) compared with an EAS CHEXA8 model of
the same body (48 x 48 x 8, converging):

| t / L | Shell before | Shell now | Solid |
|---|---:|---:|---:|
| 0.06 | +0.9% | +0.17% | -1.154e-2 |
| 0.2 | +11% | +3.7% | -1.658e-3 |
| 0.6 | +114% | +35% | -1.73e-4 |

The remaining error falls as t^2 (second-order thickness terms and shell
theory itself; MYSTRAN, which integrates the 3D metric exactly, is 9% above
the solid at t / L = 0.6).

## CTRIA3 = MITC3+ (2026-10-16)

| Benchmark | Mesh | Before (CST+DKT) | MITC3+ | Ko et al. MITC3+ on that mesh |
|---|---|---:|---:|---:|
| Scordelis-Lo roof | 32×32 | 1.53% | 1.28% | 1.49% |
| Pinched cylinder | 64×64 (was 32×32) | 0.08% | 0.06% | 0.2% |
| Pinched hemisphere | 16×16 | 7.04% | 0.92% | 0.4% |
| Twisted beam t = 0.32, in-plane | 8×48 | 39.1% | 0.37% | 0.53% |
| Twisted beam t = 0.32, out-of-plane | 8×48 | 13.7% | 2.17% | 2.6% |
| Twisted beam t = 0.0032, in-plane | 8×48 | 77.7% | 0.62% | 0.68% |
| Twisted beam t = 0.0032, out-of-plane | 8×48 | 71.9% | 0.46% | 0.49% |
| Clamped plate, a/t = 1000 | 8×8 | 0.54% | 2.18% | |
| Simply supported plate, a/t = 1000 | 8×8 | 0.34% | 0.85% | |
| Cook's membrane (mid right edge) | 16×16 | 0.62%* | 3.49% | |
| Free thermal expansion, faceted ring | 24 facets | exact | exact | |

\* corner value in the baseline table.  The triangle membrane is the
constant-strain triangle (as in MITC3+), so it is stiffer in in-plane
bending (Cook's) than the old element on that point; plates and curved
shells are far better.  An Allman drill membrane was tried and rejected:
it locked thin curved shells (thin twisted beam 22%, hemisphere 13%).
