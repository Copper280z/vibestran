# Drilling modes, 2026-11-05

Run on bob-desktop with `vibestran` copied to `~/pcg-perf/drilling-20261105/` and the rover decks beside it:

- `full.py suite`: the model-class suite with `rigid,drilling`, verified at the coarsest levels.
- `full.py custom`: unstructured, reversed-element, cylinder and rover decks, and the other PCG backends.
- `variants.py`: which grids carry each pattern. It needs a build with a temporary `VIBESTRAN_DRILL_VARIANT` switch, which was not kept.

Results stay on that machine in `runs.jsonl` and `suite.json`. Write-up: "Drilling modes" in `PCG_LARGE_DECKS.md`.
