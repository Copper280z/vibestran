#!/usr/bin/env python3
"""Compare independently exported solver scalars outside the test suite.

Both JSON files contain {"tool":..., "version":..., "metrics":{
  "gap_B_T":{"value":..., "abs_tol":..., "rel_tol":.01}, ...}}
Tolerances belong to the reference manifest and must describe converged meshes.
Use identical SI units, signs, phase currents, angles and integration surfaces.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    comparisons = []
    if not reference["metrics"]:
        parser.error("Reference has no metrics")
    for name, metric in reference["metrics"].items():
        expected = float(metric["value"])
        observed = float(candidate["metrics"][name]["value"])
        absolute = float(metric["abs_tol"])
        relative = float(metric["rel_tol"])
        if not all(math.isfinite(x) for x in (expected, observed, absolute, relative)) or min(absolute, relative) < 0:
            parser.error("Metrics and tolerances must be finite; tolerances must be nonnegative")
        difference = abs(observed-expected)
        tolerance = absolute+relative*abs(expected)
        comparisons.append(dict(metric=name, expected=expected, observed=observed,
                                error=difference, tolerance=tolerance, passed=difference <= tolerance))
    report = dict(reference={k: reference[k] for k in ("tool", "version")},
                  candidate={k: candidate[k] for k in ("tool", "version")},
                  comparisons=comparisons, passed=all(row["passed"] for row in comparisons))
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
