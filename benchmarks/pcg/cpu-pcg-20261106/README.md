# CPU PCG, 2026-11-06

Run on bob-desktop from `~/pcg-perf/cpucycle-20261106/`, with the solver built in `~/vibestran-pcg-perf/build` and generated decks in `/tmp/cpucycle`:

- `run.py`: runs one deck on one backend, pinned to the first cores, and appends a line to a results file. It is imported by the others.
- `big2.py`: `solid-hex` 70 and `shell-membrane` 1258 on `cuda-pcg` and `cpu-pcg`, and how far apart their F06 displacements are.
- `all.py`: thread scaling on `solid-hex` 40, then `cpu-pcg`, `cuda-pcg` and `cpu` on the decks near 240k equations. The rover deck is the one from `amg-drilling-20261105`.
- `watch.sh`: kills a run that passes a resident memory limit.
- `show.py`: prints a results file.
- `validation_decks.py`: on the Mac, runs every deck in `MYSTRAN_Validation` on direct, the old incomplete Cholesky binary and AMG, and counts which complete. It takes the directory holding a build of the old code, as `wt-main/build/vibestran`.

The model-class suite ran from `tools/amg_model_suite.py`:

```sh
OMP_NUM_THREADS=8 amg_model_suite.py run --binary vibestran --work WORK --results final-suite.jsonl \
    --backend cpu-pcg --modes default --direct --prefix "taskset -c 0-7" --timeout 600
```

Results stay on that machine in `big2.jsonl`, `final.jsonl`, `final-suite.jsonl` and the older `runs.jsonl`, `gs*.jsonl` and `suite.jsonl`, which record the smoothers that were tried. Those used prototype switches that were not kept. Write-up: "CPU PCG" in `PCG_LARGE_DECKS.md`.
