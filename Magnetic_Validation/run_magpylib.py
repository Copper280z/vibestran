#!/usr/bin/env python3
"""Optional external comparison; deliberately absent from Meson and CI.

Requires Magpylib >=5 and h5py. A thin rectangular loop is compared to an independent
filament solution at probes away from the conductor. The finite wire section,
field mesh and truncated air domain each contribute error; this is not a
reference for finite-section self-inductance or force.
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from magnetic_examples import deck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vibestran", type=Path, default=Path("build/vibestran"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[12, 16, 24])
    parser.add_argument("--extents", type=float, nargs="+", default=[3, 5])
    parser.add_argument("--tolerance", type=float, default=.01)
    args = parser.parse_args()
    try:
        import magpylib
    except ImportError:
        parser.error("Install Magpylib in a separate validation environment: pip install 'magpylib>=5,<7'")
    version = importlib.metadata.version("magpylib")
    if int(version.split(".")[0]) < 5:
        parser.error("Magpylib >=5 is required for SI units")
    args.output.mkdir(parents=True, exist_ok=True)
    reference = magpylib.current.Polyline(current=1, vertices=[[-.6, -.6, 0], [.6, -.6, 0], [.6, .6, 0], [-.6, .6, 0], [-.6, -.6, 0]])
    probes = [[0, 0, 0], [0, 0, .3]]
    expected = reference.getB(probes).tolist()
    runs = []
    for extent in args.extents:
        for n in args.levels:
            name = f"loop_n{n}_extent{extent:g}"
            path = args.output/(name+".bdf")
            path.write_text(deck("loop", n, extent))
            start = time.monotonic()
            with (args.output/(name+".log")).open("w") as log:
                subprocess.run([str(args.vibestran.resolve()), str(path.resolve())],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            with h5py.File(path.with_suffix(".mag.vtkhdf"), "r") as result:
                case = result["MAGSTAT/subcases/1"]
                observed = case["probes/B_T"][()].tolist()
                linkage = float(case["sources/linkage_Wb_turn"][0])
            errors = [math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))/math.sqrt(sum(y*y for y in b))
                      for a, b in zip(observed, expected)]
            record = dict(n=n, extent=extent, elapsed_s=time.monotonic()-start,
                          observed_T=observed, relative_errors=errors,
                          linkage_Wb_turn=linkage)
            runs.append(record)
            print(f"{name}: max field error {max(errors):.3%}", flush=True)
    finest = max(runs, key=lambda row: (row["extent"], row["n"]))
    report = dict(reference_tool="Magpylib", version=version, reference_T=expected,
                  tolerance=args.tolerance, runs=runs,
                  finest_pass=max(finest["relative_errors"]) <= args.tolerance)
    (args.output/"comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    return 0 if report["finest_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
