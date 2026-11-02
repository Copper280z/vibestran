#!/usr/bin/env python3
"""Benchmark device-owned AMG state with measured 50/75/90% fine-row budgets.

Calibration obtains the exact budgets from the production memory planner.
Every timed spill solve must report the requested row count and pass the same
independent solution-error limit. Includes resident and equal-budget controls.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="thermal227k_1,motor218k_1,motor218k_10,plate_sc1")
    parser.add_argument("--backends", default="cuda-pcg,cuda-pcg-mixed,vulkan")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("repeats and timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    env.pop("VIBESTRAN_PCG_DEVICE_MEMORY_MIB", None)
    record = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "binary_sha256": sha256(args.binary), "baseline_sha256": sha256(args.baseline) if args.baseline else None,
              "protocol": {"cpu_affinity": "0-7", "omp_threads": 8, "repeats": args.repeats,
                           "warmups": 0, "tolerance": 1e-8, "reference_error_limit": 1e-8,
                           "timing": "Solve including setup/hierarchy/transfers/refinement; excludes file loading and verification.",
                           "budget_scope": "Device vectors, hierarchy, resident fine CSR and compact exchange workspace plus 64 KiB; excludes driver/library/context overhead."},
              "matrices": {}, "runs": []}

    def run(case, backend, budget, mode, percentage=None, repeats=None, binary=None):
        tag = f"{case}-{backend}-{mode}" + (f"-{percentage}pct" if percentage else "")
        repeats = args.repeats if repeats is None else repeats
        command = [str((binary or args.binary).resolve()), "--backend", backend,
                   "--matrix", str((args.matrices / f"{case}_K.mtx").resolve()),
                   "--rhs", str((args.matrices / f"{case}_F.mtx").resolve()),
                   "--reference", str((args.matrices / f"{case}_u.mtx").resolve()),
                   "--budget-mib", format(budget / 1048576, ".17g"),
                   "--warmups", "0", "--repeats", str(repeats), "--profile"]
        resource = (args.output / f"{tag}.resources.json").resolve()
        invocation = ["/usr/bin/time", "-f", '{"wall_seconds":%e,"peak_rss_kib":%M}', "-o", str(resource),
                      "timeout", f"{args.timeout}s", "taskset", "-c", "0-7", *command]
        print("Running", tag, "budget bytes", budget, flush=True)
        stdout, stderr = args.output / f"{tag}.log", args.output / f"{tag}.err"
        with stdout.open("w") as out, stderr.open("w") as err:
            code = subprocess.run(invocation, env=env, stdout=out, stderr=err).returncode
        output = stdout.read_text() + stderr.read_text()
        lines = [line for line in output.splitlines() if line.startswith("backend,n,") or line.startswith(backend + ",")]
        samples = [{k: v if k == "backend" else float(v) for k, v in row.items()} for row in csv.DictReader(lines)]
        partitions = [(int(g), int(c)) for g, c in re.findall(r"device AMG\] (\d+) GPU rows, (\d+) CPU rows", output)]
        plans = re.findall(r"row budgets: 50%=(\d+) 75%=(\d+) 90%=(\d+) bytes", output)
        accuracy = len(samples) == repeats and all(
            math.isfinite(s["relative_error"]) and s["relative_error"] <= 1e-8 and
            math.isfinite(s["estimated_error"]) and s["estimated_error"] <= 1e-8 for s in samples)
        partition_ok = True
        if mode == "rows":
            n = record["matrices"][case]["n"]
            partition_ok = len(partitions) == repeats and all(g == n * percentage // 100 and g + c == n for g, c in partitions)
        usage = json.loads(next(line for line in resource.read_text().splitlines() if line.startswith("{")))
        result = {"case": case, "backend": backend, "mode": mode, "percentage": percentage,
                  "budget_bytes": budget, "budget_mib": budget / 1048576, "repeats": repeats,
                  "exit": code, "passed": code == 0 and accuracy and partition_ok,
                  "accuracy_verified": accuracy, "partition_verified": partition_ok,
                  "device_gpu_cpu_rows": partitions, "row_budgets": [list(map(int, p)) for p in plans],
                  "command": command, "invocation": invocation, "samples": samples,
                  "diagnostics": output, **usage}
        if samples:
            seconds = [s["seconds"] for s in samples]
            result.update(median_seconds=statistics.median(seconds), min_seconds=min(seconds), max_seconds=max(seconds))
        record["runs"].append(result)
        (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
        print(tag, "PASS" if result["passed"] else "FAIL", [s["seconds"] for s in samples], partitions, flush=True)
        return result

    for case in args.cases.split(","):
        matrix = args.matrices / f"{case}_K.mtx"
        with matrix.open() as stream:
            n, _, nnz = map(int, next(line for line in stream if line.strip() and not line.startswith("%")).split())
        record["matrices"][case] = {"n": n, "stored_nnz": nnz,
            "sha256": {suffix: sha256(args.matrices / f"{case}_{suffix}.mtx") for suffix in ("K", "F", "u")}}
        budgets = {}
        backends = args.backends.split(",")
        for backend in backends:
            family = "cuda-pcg" if backend.startswith("cuda") else backend
            if family in budgets:
                continue
            # Calibration is separate from all repeated timing configurations.
            calibration = run(case, family, (2048 if n > 1000000 else 256) * 1048576, "calibration", repeats=1)
            if not calibration["passed"] or not calibration["row_budgets"]:
                print("Calibration failed; retained diagnostics", flush=True)
                return 1
            budgets[family] = dict(zip((50, 75, 90), calibration["row_budgets"][0]))
        for percentage in (50, 75, 90):
            for backend in backends:
                family = "cuda-pcg" if backend.startswith("cuda") else backend
                run(case, backend, budgets[family][percentage], "rows", percentage)
        for backend in backends:
            run(case, backend, 0, "resident")
        if args.baseline:
            for backend in (b for b in ("cuda-pcg", "vulkan") if b in backends):
                run(case, backend, budgets[backend][90], "baseline_same_budget", 90, binary=args.baseline)
    return int(any(not r["passed"] for r in record["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
