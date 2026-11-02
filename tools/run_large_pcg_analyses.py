#!/usr/bin/env python3
"""Run complete >=200k-GRID analyses, with time limits and peak-memory measurements.

Use the prepared thermal227k.bdf and motor218k.bdf plus the existing plate deck.
Every failed/timeout analysis remains in the output; no accuracy setting changes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--decks", required=True, type=Path)
    parser.add_argument("--plate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backends", default="cuda-pcg,cuda-pcg-mixed,vulkan")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--only", help="substring of case/backend/budget name")
    parser.add_argument("--spill-budgets", type=Path,
                        help="JSON mapping case -> backend -> positive integer MiB budget")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("thermal227k", args.decks / "thermal227k.bdf", 16),
             ("motor218k", args.decks / "motor218k.bdf", 8),
             ("plate", args.plate, 128)]
    results = []
    spill_budgets = json.loads(args.spill_budgets.read_text()) if args.spill_budgets else {}
    for case, deck, spill_budget in cases:
        for backend in args.backends.split(","):
            selected_budget = spill_budgets.get(case, {}).get(backend, spill_budget)
            if not isinstance(selected_budget, int) or selected_budget <= 0:
                parser.error(f"invalid spill budget for {case}/{backend}: {selected_budget}")
            for budget in ([0] if backend.startswith("cpu") else [0, selected_budget]):
                tag = f"{case}-{backend}-{budget}"
                if args.only and args.only not in tag:
                    continue
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "8"
                env.pop("VIBESTRAN_PCG_DEVICE_MEMORY_MIB", None)
                if budget:
                    env["VIBESTRAN_PCG_DEVICE_MEMORY_MIB"] = str(budget)
                resource = (args.output / f"{tag}.resources.json").resolve()
                command = ["/usr/bin/time", "-f", '{"wall_seconds":%e,"peak_rss_kib":%M}',
                           "-o", str(resource), "timeout", f"{args.timeout}s", "taskset", "-c", "0-7",
                           str(args.binary.resolve()), f"--backend={backend}", str(deck.resolve()),
                           str((args.output / f"{tag}.f06").resolve())]
                print("Running", tag, flush=True)
                log = args.output / f"{tag}.log"
                with log.open("w") as output:
                    code = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT).returncode
                usage = json.loads(next(line for line in resource.read_text().splitlines() if line.startswith("{")))
                results.append({"case": case, "backend": backend, "budget_mib": budget,
                                "exit": code, "timeout_seconds": args.timeout, "command": command,
                                "omp_threads": 8, **usage, "diagnostic_tail": log.read_text()[-12000:]})
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(tag, "exit", code, usage, flush=True)
    return int(any(row["exit"] != 0 for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
