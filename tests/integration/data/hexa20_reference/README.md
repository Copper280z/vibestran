# CHEXA20 reference decks

Copies of CHEXA20 decks from MYSTRAN's `hexa_msc_comparison` set
(`MYSTRAN/hexa_msc_comparison`, MSC Nastran input written by its
`export_msc_decks.py`), used by `tests/integration/test_hexa20_reference.cpp`.
They are flat here; the source folder of each is:

| source folder | decks |
|---|---|
| `convergence` | pinched hemisphere and cylinder, 8x8x1 and 12x12x1 |
| `shell` | Scordelis-Lo roof, twisted beam (both loads), simply supported plate |
| `modal` | simply supported plate frequencies, trapezoidal MacNeal cantilever |
| `robustness` | curved-midside patch, free thermal expansion, linear temperature gradients, rotated and relabelled distorted cantilevers, Lame cylinders at nu = 0.3 and 0.4999, a rotated 1e3:1 element |

The benchmark expectations are the values of the numpy prototype
(`tools/hex20_prototype/ref_values.py`); the others are exact solutions. The
published references and the values of MYSTRAN, Simcenter Nastran 2412 and
ComLab are in the test comments, from `MYSTRAN/hexa_msc_comparison/results.md`.
