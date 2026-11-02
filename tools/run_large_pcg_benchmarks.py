#!/usr/bin/env python3
"""Bounded PCG replay of the >=200k-grid physical deck captures.

Each process records its actual solve timing, convergence error, wall time,
and peak host RSS. Failed/timeout runs remain in the JSON output. Initial
screening uses one cold solve; follow successful cases with repeated runs.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="updated")
    parser.add_argument("--timeout", type=float, default=75)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", help="substring of case/backend/mode name")
    parser.add_argument("--backends", default="cpu,cpu-pcg,cuda-pcg,cuda-pcg-mixed,vulkan")
    parser.add_argument("--successful-from", type=Path,
                        help="repeat only successful configurations from a screening JSON")
    args = parser.parse_args()
    if args.timeout <= 0 or args.repeats < 1 or args.warmups < 0:
        parser.error("invalid timeout or repeat counts")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("thermal227k_1", 16), ("motor218k_1", 8),
             ("motor218k_10", 8), ("plate_sc1", 128)]
    results = []
    successful = None
    if args.successful_from:
        successful = {(row["case"], row["backend"], row["mode"])
                      for row in json.loads(args.successful_from.read_text())
                      if row["exit"] == 0 and row["samples"]}
    for case, budget in cases:
        for backend in args.backends.split(","):
            if args.version == "baseline" and backend.startswith("cpu"):
                continue
            modes = ["resident"]
            if backend == "vulkan" or (args.version != "baseline" and backend.startswith("cuda")):
                modes.append("spill")
            for mode in modes:
                if successful is not None and (case, backend, mode) not in successful:
                    continue
                tag = f"{args.version}-{case}-{backend}-{mode}"
                if args.only and args.only not in tag:
                    continue
                command = [str(args.binary.resolve()), "--backend", backend,
                           "--matrix", str((args.matrices/f"{case}_K.mtx").resolve()),
                           "--rhs", str((args.matrices/f"{case}_F.mtx").resolve()),
                           "--reference", str((args.matrices/f"{case}_u.mtx").resolve()),
                           "--warmups", str(args.warmups), "--repeats", str(args.repeats), "--profile"]
                if mode == "spill":
                    command += ["--budget-mib", str(budget)]
                resource = (args.output/f"{tag}.resources.json").resolve()
                invocation = ["/usr/bin/time", "-f", '{"wall_seconds":%e,"peak_rss_kib":%M}',
                              "-o", str(resource), "timeout", f"{args.timeout}s",
                              "taskset", "-c", "0-7", *command]
                output, error = args.output/f"{tag}.log", args.output/f"{tag}.err"
                print("Running", tag, flush=True)
                with output.open("w") as out, error.open("w") as err:
                    code = subprocess.run(invocation, stdout=out, stderr=err).returncode
                text = output.read_text()
                lines = [line for line in text.splitlines()
                         if line.startswith("backend,n,") or line.startswith(backend+",")]
                samples = [{k: v if k == "backend" else float(v) for k, v in row.items()}
                           for row in csv.DictReader(lines)]
                resource_text = resource.read_text()
                usage = json.loads(next(line for line in resource_text.splitlines() if line.startswith("{")))
                results.append({"case": case, "backend": backend, "mode": mode,
                                "version": args.version, "exit": code, "timeout_seconds": args.timeout,
                                "warmups": args.warmups, "repeats": args.repeats,
                                "budget_mib": budget if mode == "spill" else 0,
                                "command": command, "samples": samples, **usage,
                                "diagnostic_tail": text[-6000:]+error.read_text()[-6000:]})
                (args.output/f"{args.version}-results.json").write_text(json.dumps(results, indent=2)+"\n")
                print(tag, "exit", code, usage, samples, flush=True)
    return int(any(row["exit"] != 0 for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
