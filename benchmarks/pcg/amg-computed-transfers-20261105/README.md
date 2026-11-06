# Computed transfers, 2026-11-05

Scripts behind "Computed transfers" in `PCG_LARGE_DECKS.md`, run on the TITAN V machine
from `~/pcg-perf/implicit-20261105/`, with the built `vibestran` copied there.

- `base.py`: one measured run with a budget, appended to `runs.jsonl`.
- `full.py`: the default plan against forced computed on the largest level of each class,
  row-fraction budgets on the two large decks with the previous binary, the rover decks,
  mixed precision and Vulkan. MAGSTAT refuses `--no-quality-checks`, so the magnetic
  classes fail here and run in `suite.sh`.
- `prof.sh`: `nsys` kernel profiles, stored and computed.
- `suite.sh`: `tools/amg_model_suite.py` against a direct solve, with no budget and at
  200 MiB.

Results and logs stay on that machine.
