#!/usr/bin/env python3
"""Run explicitly prepared rotor-angle decks and collect scalar results.

Manifest: {"cases": [{"angle_deg": 0, "deck": "angle0.bdf"}, ...]}
Deck paths are relative to the manifest. This tool does not remesh or rotate data.
Results are read from each run's name.mag.vtkhdf (/MAGSTAT group); needs h5py.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess

import h5py


def run(executable, manifest, output):
    entries = json.loads(manifest.read_text())["cases"]
    if not entries:
        raise ValueError("Sweep manifest contains no cases")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, entry in enumerate(entries):
        angle = float(entry["angle_deg"])
        if not math.isfinite(angle):
            raise ValueError("Rotor angles must be finite")
        deck = (manifest.parent / entry["deck"]).resolve()
        report = output / f"angle_{index:04d}.F06"
        with (output / f"angle_{index:04d}.log").open("w") as log:
            subprocess.run([str(executable), "--backend=cpu", str(deck), str(report)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        with h5py.File(report.with_suffix(".mag.vtkhdf"), "r") as results:
            magstat = results["MAGSTAT"]
            for subcase in magstat["subcase_ids"]:
                case = magstat[f"subcases/{subcase}"]
                attrs = case.attrs
                row = dict(angle_deg=angle, subcase=int(subcase), energy_J=float(attrs["energy_J"]),
                           coenergy_J=float(attrs["coenergy_J"]), residual=float(attrs["residual"]),
                           bh_extrapolated=bool(attrs["bh_extrapolated"]))
                for i, surface in enumerate(magstat["forces/id"]):
                    for field in ("force_N", "torque_Nm"):
                        for axis, value in zip("xyz", case["forces"][field][i]):
                            row[f"surface{surface}_{field}_{axis}"] = float(value)
                for i, circuit in enumerate(magstat["circuits/id"]):
                    name = f"circuit{circuit}"
                    row[name+"_current_A"] = float(case["circuits/current_A"][i])
                    row[name+"_linkage_Wb_turn"] = float(case["circuits/linkage_Wb_turn"][i])
                rows.append(row)
    # At fixed currents, d(I.lambda-W)/d(theta) is the virtual-work torque.
    # Three adjacent prepared angles are needed; endpoints stay blank.
    for subcase in {row["subcase"] for row in rows}:
        group = sorted((row for row in rows if row["subcase"] == subcase),
                       key=lambda row: row["angle_deg"])
        if any(a["angle_deg"] == b["angle_deg"] for a, b in zip(group, group[1:])):
            raise ValueError("Repeated rotor angle for the same subcase")
        currents = [{k: v for k, v in row.items() if k.endswith("_current_A")} for row in group]
        for row in group:
            row["virtual_work_torque_Nm"] = None
        if any(current != currents[0] for current in currents):
            continue
        for i in range(1, len(group)-1):
            a, b, c = group[i-1:i+2]
            left = math.radians(b["angle_deg"]-a["angle_deg"])
            right = math.radians(c["angle_deg"]-b["angle_deg"])
            b["virtual_work_torque_Nm"] = (
                -right/(left*(left+right))*a["coenergy_J"] +
                (right-left)/(left*right)*b["coenergy_J"] +
                left/(right*(left+right))*c["coenergy_J"])
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output/"sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--vibestran", type=Path, default=Path("build/vibestran"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.vibestran.resolve(), args.manifest.resolve(), args.output.resolve())
    print(args.output/"sweep.csv")


if __name__ == "__main__":
    main()
