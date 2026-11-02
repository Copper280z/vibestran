#!/usr/bin/env python3
"""Measure spill PCG at exact fine-matrix row residency fractions.

Derive byte budgets from the captured matrix's full, zero-pruned CSR pattern,
matching ScaledAmgMatrix and pcg_prefix_rows. Verify the actual GPU/CPU row
counts printed by every solve. No solver implementation or tolerance changes.
"""
import argparse
from array import array
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


def matrix_plan(path, percentages):
    """Read the canonical lower-triangle dump; reject duplicates or reordering."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        header = next(stream)
        digest.update(header)
        if header.strip() != b"%%MatrixMarket matrix coordinate real symmetric":
            raise ValueError(f"Unsupported Matrix Market header: {path}")
        for line in stream:
            digest.update(line)
            if line.strip() and not line.startswith(b"%"):
                n, columns, stored_nnz = map(int, line.split())
                break
        else:
            raise ValueError(f"Missing matrix dimensions: {path}")
        if n < 50000 or columns != n:
            raise ValueError(f"Expected a square AMG-sized system, got {n}x{columns}: {path}")
        counts = array("I", [0]) * n
        previous = (0, 0)
        entries = 0
        for line in stream:
            digest.update(line)
            fields = line.split()
            if not fields or line.startswith(b"%"):
                continue
            row, col = int(fields[0]), int(fields[1])
            value = float(fields[2])
            if not (1 <= col <= row <= n) or (row, col) <= previous or not math.isfinite(value):
                raise ValueError(f"Invalid/noncanonical entry {entries + 1} in {path}: {line!r}")
            previous = (row, col)
            entries += 1
            if value != 0:
                counts[row - 1] += 1
                if col != row:
                    counts[col - 1] += 1
        if entries != stored_nnz:
            raise ValueError(f"Expected {stored_nnz} entries, read {entries}: {path}")

    full_nnz = sum(counts)
    fractions = []
    for percentage in percentages:
        rows = n * percentage // 100
        nnz = sum(counts[:rows])
        # fp64 values + int32 columns; full input and prefix output vectors;
        # int32 row pointers and the selector's fixed 64 KiB reserve.
        matrix_bytes = 12 * nnz + 4 * (rows + 1)
        vector_bytes = 8 * (n + rows)
        budget = matrix_bytes + vector_bytes + 65536
        next_budget = budget + 12 * counts[rows] + 12
        if int(float(format(budget / 1048576, ".17g")) * 1048576) != budget:
            raise ValueError(f"Budget does not round-trip exactly through --budget-mib: {budget}")
        fractions.append({"percentage": percentage, "gpu_rows": rows, "cpu_rows": n - rows,
                          "actual_row_fraction": rows / n, "gpu_nnz": nnz,
                          "actual_nnz_fraction": nnz / full_nnz,
                          "matrix_bytes": matrix_bytes, "vector_bytes": vector_bytes,
                          "budget_bytes": budget, "budget_mib": budget / 1048576,
                          "next_row_budget_bytes": next_budget})
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "n": n,
            "stored_nnz": stored_nnz, "full_nonzero_nnz": full_nnz, "fractions": fractions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--matrices", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--percentages", default="50,75,90")
    parser.add_argument("--backends", default="cuda-pcg,cuda-pcg-mixed,vulkan")
    parser.add_argument("--cases", default="thermal227k_1,motor218k_1,motor218k_10,plate_sc1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    percentages = [int(p) for p in args.percentages.split(",")]
    backends = args.backends.split(",")
    if not percentages or any(p <= 0 or p >= 100 for p in percentages):
        parser.error("percentages must be strictly between 0 and 100")
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    if any(b not in {"cuda-pcg", "cuda-pcg-mixed", "vulkan"} for b in backends):
        parser.error("this sweep supports CUDA/Vulkan PCG backends only")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    env.pop("VIBESTRAN_PCG_DEVICE_MEMORY_MIB", None)
    record = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "binary": str(args.binary.resolve()),
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "protocol": {"percentages": percentages, "repeats": args.repeats, "warmups": 0,
                           "cpu_affinity": "0-7", "omp_threads": 8, "estimated_error_target": 1e-8,
                           "independent_reference_error_limit": 1e-8,
                           "timing": "Setup, upload, hierarchy and refinement included; input loading and independent verification excluded.",
                           "partition": "Original equation-order prefix of full zero-pruned fine CSR. Coarse hierarchy and vector operations remain on host.",
                           "budget_scope": "Accounted fine CSR and vectors plus 64 KiB; not a cap on driver/library allocations."},
              "matrices": {}, "runs": []}
    results_path = args.output / "results.json"
    for case in args.cases.split(","):
        print("Planning exact row budgets:", case, flush=True)
        plan = matrix_plan(args.matrices / f"{case}_K.mtx", percentages)
        record["matrices"][case] = plan
        for fraction in plan["fractions"]:
            for backend in backends:
                tag = f"{case}-{backend}-{fraction['percentage']}pct"
                command = [str(args.binary.resolve()), "--backend", backend,
                           "--matrix", plan["path"],
                           "--rhs", str((args.matrices / f"{case}_F.mtx").resolve()),
                           "--reference", str((args.matrices / f"{case}_u.mtx").resolve()),
                           "--budget-mib", format(fraction["budget_mib"], ".17g"),
                           "--warmups", "0", "--repeats", str(args.repeats), "--profile"]
                resource = (args.output / f"{tag}.resources.json").resolve()
                invocation = ["/usr/bin/time", "-f", '{"wall_seconds":%e,"peak_rss_kib":%M}',
                              "-o", str(resource), "timeout", f"{args.timeout}s", "taskset", "-c", "0-7", *command]
                stdout, stderr = args.output / f"{tag}.log", args.output / f"{tag}.err"
                print("Running", tag, f"budget={fraction['budget_mib']:.6f} MiB",
                      f"GPU rows={fraction['gpu_rows']}/{plan['n']}", flush=True)
                with stdout.open("w") as out, stderr.open("w") as err:
                    code = subprocess.run(invocation, env=env, stdout=out, stderr=err).returncode
                output = stdout.read_text()
                diagnostics = output + stderr.read_text()
                lines = [line for line in output.splitlines()
                         if line.startswith("backend,n,") or line.startswith(backend + ",")]
                samples = [{key: value if key == "backend" else float(value) for key, value in row.items()}
                           for row in csv.DictReader(lines)]
                observed = [(int(gpu), int(cpu)) for gpu, cpu in
                            re.findall(r"(\d+) GPU rows, (\d+) CPU rows", diagnostics)]
                expected = (fraction["gpu_rows"], fraction["cpu_rows"])
                partition_ok = len(observed) == args.repeats and all(pair == expected for pair in observed)
                accuracy_ok = len(samples) == args.repeats and all(
                    math.isfinite(s["relative_error"]) and s["relative_error"] <= 1e-8 and
                    math.isfinite(s["estimated_error"]) and s["estimated_error"] <= 1e-8 and
                    s["n"] == plan["n"] for s in samples)
                usage = json.loads(next(line for line in resource.read_text().splitlines() if line.startswith("{")))
                result = {"case": case, "backend": backend, **fraction, "exit": code,
                          "partition_verified": partition_ok, "observed_gpu_cpu_rows": observed,
                          "accuracy_verified": accuracy_ok, "passed": code == 0 and partition_ok and accuracy_ok,
                          "timeout_seconds": args.timeout, "command": command, "invocation": invocation,
                          "samples": samples, **usage, "diagnostic_tail": diagnostics[-10000:]}
                if samples:
                    seconds = [s["seconds"] for s in samples]
                    result.update(median_seconds=statistics.median(seconds), min_seconds=min(seconds), max_seconds=max(seconds))
                record["runs"].append(result)
                results_path.write_text(json.dumps(record, indent=2) + "\n")
                print(tag, "PASS" if result["passed"] else "FAIL", "seconds=",
                      [s["seconds"] for s in samples], "partition=", observed, flush=True)
    return int(any(not r["passed"] for r in record["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
