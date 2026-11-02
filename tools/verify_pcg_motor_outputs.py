#!/usr/bin/env python3
"""Compare physical motor fields, circuits, energy and torque with CPU direct."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


FIELDS = [
    "VTKHDF/SUBCASE_1/PointData/A_normal_Wb_per_m",
    "VTKHDF/SUBCASE_1/CellData/B_T", "VTKHDF/SUBCASE_1/CellData/H_A_per_m",
    *[f"MAGSTAT/subcases/1/probes/{name}" for name in ("B_T", "B_local_T", "H_A_per_m", "H_local_A_per_m")],
    *[f"MAGSTAT/subcases/1/circuits/{name}" for name in ("current_A", "inductance_H", "linkage_Wb_turn", "linkage_rate_Wb_per_rad")],
    *[f"MAGSTAT/subcases/1/sources/{name}" for name in ("current_A", "differential_inductance_H", "linkage_Wb_turn")],
]


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("tolerance must be finite and positive")
    record = {"reference": str(args.reference), "reference_sha256": file_hash(args.reference),
              "relative_error_definition": "max(abs(candidate-reference))/max(abs(reference)) per complete dataset",
              "tolerance": args.tolerance, "comparisons": []}
    with h5py.File(args.reference) as reference:
        for candidate in args.candidates:
            result = {"file": str(candidate), "sha256": file_hash(candidate), "datasets": {}, "attributes": {}, "passed": True}
            with h5py.File(candidate) as actual:
                geometry = ("NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds", "Points", "Connectivity", "Offsets", "Types")
                result["identical_geometry"] = all(np.array_equal(reference[f"VTKHDF/SUBCASE_1/{name}"][...], actual[f"VTKHDF/SUBCASE_1/{name}"][...]) for name in geometry)
                result["passed"] &= result["identical_geometry"]
                for path in FIELDS:
                    a, b = reference[path][...], actual[path][...]
                    masks_match = (a.shape == b.shape and not np.isinf(a).any() and not np.isinf(b).any()
                                   and np.array_equal(np.isfinite(a), np.isfinite(b))
                                   and np.array_equal(np.isnan(a), np.isnan(b)))
                    finite = np.isfinite(a)
                    if not masks_match:
                        result["datasets"][path] = {"shape_and_finite_masks_match": False}
                        result["passed"] = False
                        continue
                    if not finite.any():
                        result["datasets"][path] = {"all_unavailable": True, "shape_and_finite_masks_match": True}
                        continue
                    scale = float(np.max(np.abs(a[finite])))
                    error = float(np.max(np.abs(a[finite] - b[finite])))
                    relative = error / max(scale, 1e-300)
                    result["datasets"][path] = {"max_abs_reference": scale, "max_abs_error": error, "relative_error": relative}
                    result["passed"] &= relative <= args.tolerance
                ref_attrs, attrs = reference["MAGSTAT/subcases/1"].attrs, actual["MAGSTAT/subcases/1"].attrs
                for name in ("air_gap_torque_Nm", "energy_J", "coenergy_J"):
                    a, b = float(ref_attrs[name]), float(attrs[name])
                    error = abs(a - b) / max(abs(a), 1e-300)
                    result["attributes"][name] = {"cpu": a, "candidate": b, "relative_error": error}
                    result["passed"] &= bool(np.isfinite(b)) and error <= args.tolerance
                result.update(iterations=int(attrs["iterations"]), nonlinear_residual=float(attrs["residual"]), bh_extrapolated=int(attrs["bh_extrapolated"]))
                result["passed"] &= result["bh_extrapolated"] == int(ref_attrs["bh_extrapolated"])
                result["max_dataset_relative_error"] = max((x.get("relative_error", 0) for x in result["datasets"].values()), default=0)
            record["comparisons"].append(result)
            print(candidate.name, "PASS" if result["passed"] else "FAIL", result["max_dataset_relative_error"])
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    return int(any(not r["passed"] for r in record["comparisons"]))


if __name__ == "__main__":
    raise SystemExit(main())
