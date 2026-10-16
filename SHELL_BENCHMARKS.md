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
