# CHEXA8 reference decks

Copies of CHEXA8 decks from MYSTRAN's `hexa_msc_comparison` set
(`MYSTRAN/hexa_msc_comparison`, MSC Nastran input written by its
`export_msc_decks.py`), used by `tests/integration/test_hexa8_reference.cpp`.

| folder | decks |
|---|---|
| `macneal_beams` | MacNeal-Harder straight (regular, trapezoidal, parallelogram) and curved cantilevers, 6x1x1 |
| `patch` | MacNeal-Harder distorted 7-element patch, nu = 0.25 and 0.495 |
| `shell` | Scordelis-Lo roof, pinched cylinder and hemisphere, twisted beam, SS and clamped plates, one element through the thickness |
| `modal` | SS and cantilever plate frequencies, MacNeal beam frequencies |
| `robustness` | free-free single-element spectra, the patch with ISOP = FULL and in micrometre units, free thermal expansion, linear temperature gradients, rotated and relabelled distorted cantilevers, Lame cylinders, 1e3:1 and 1e20:1 elements, IN options |

The expected values in `tests/integration/hexa8_reference_checks.inc` are the
results of MYSTRAN 19.0.0's Q1/E21 CHEXA8 for these decks (7 printed digits);
the published references and Simcenter Nastran 2412's values are in the
comments, from `MYSTRAN/hexa_msc_comparison/results.md`.
