#!/usr/bin/env python3
"""Replay captured physical PCG systems, preserving failures and raw timings.

Generate decks with generate_pcg_cases.py and capture each with pcg_benchmark
--capture-deck. Magnetic decks must pass the CPU direct time gate first.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60,
                        help="seconds for the warmup plus three timed solves")
    parser.add_argument("--cpu-affinity", help="Linux taskset CPU list, e.g. 0-7")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("thermal_interface_1", 1), ("magnetic_winding_1", .25),
             ("magnetic_motor_1", .25), ("magnetic_motor_7", .25),
             ("magnetic_coil_1", .25)]
    versions = [("updated", args.binary)]
    if args.baseline:
        versions.insert(0, ("baseline", args.baseline))
    statuses = []
    for version, executable in versions:
        for case, budget in cases:
            for backend in ("cpu", "cpu-pcg", "cuda-pcg", "cuda-pcg-mixed", "vulkan"):
                if version == "baseline" and backend.startswith("cpu"):
                    continue
                # Original CUDA allocated a complete device system: it had no
                # explicit partitioned implementation to compare at this budget.
                modes = ["resident"]
                if backend == "vulkan" or (version == "updated" and backend.startswith("cuda")):
                    modes.append("spill")
                for mode in modes:
                    name = f"{version}-physical-{case}-{backend}-{mode}"
                    command = [str(executable.resolve()), "--backend", backend,
                               "--matrix", str((args.matrices / f"{case}_K.mtx").resolve()),
                               "--rhs", str((args.matrices / f"{case}_F.mtx").resolve()),
                               "--reference", str((args.matrices / f"{case}_u.mtx").resolve())]
                    if args.cpu_affinity:
                        command[:0] = ["taskset", "-c", args.cpu_affinity]
                    if mode == "spill":
                        command += ["--budget-mib", str(budget)]
                    with (args.output / f"{name}.csv").open("w") as out, \
                            (args.output / f"{name}.err").open("w") as err:
                        start = time.monotonic()
                        try:
                            code = subprocess.run(command, stdout=out, stderr=err,
                                                  timeout=args.timeout).returncode
                        except subprocess.TimeoutExpired:
                            code = 124
                    row = {"name": name, "exit": code, "wall_seconds": time.monotonic()-start,
                           "command": command}
                    statuses.append(row)
                    print(name, code, round(row["wall_seconds"], 3), flush=True)
                    (args.output / "physical-status.json").write_text(json.dumps(statuses, indent=2)+"\n")
    return int(any(row["exit"] != 0 for row in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
