# CHEXA20 formulation prototype

numpy prototype of the CHEXA20 formulations compared in `CHEXA20_PLAN.md`,
run on MYSTRAN's `hexa_msc_comparison` decks (needs pyNastran, numpy, scipy;
`MYSTRAN/.venv/bin/python` has them). `hex20.py` holds the element variants:
`as` (MYSTRAN's current CHEXA20), `asn`, `asnl2` (the formulation implemented
in `src/elements/solid_hexa20.cpp`), `mitc`, `red` (2x2x2) and `full`.

* `decks.py as,asnl2` - benchmark table (shells, convergence, patch, Lame).
* `check2.py`, `rank.py` - element patch, consistency, invariance and rank checks.
* `ref_values.py` - the prototype values that `tests/integration/test_hexa20_reference.cpp`
  checks vibestran against.
