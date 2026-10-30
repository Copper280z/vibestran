#!/usr/bin/env python3
"""Run explicitly prepared rotor-angle decks and collect scalar results.

Manifest: {"cases": [{"angle_deg": 0, "deck": "angle0.bdf"}, ...]}
Deck paths are relative to the manifest. This tool does not remesh or rotate data.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess


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
        results = json.loads(report.with_suffix(".mag.json").read_text())
        for case in results["cases"]:
            row = dict(angle_deg=angle, subcase=case["id"], energy_J=case["energy_J"],
                       coenergy_J=case["coenergy_J"], residual=case["residual"],
                       bh_extrapolated=case["bh_extrapolated"])
            for surface in case["forces"]:
                for field, values in (("force_N", surface["force_N"]), ("torque_Nm", surface["torque_Nm"])):
                    for axis, value in zip("xyz", values):
                        row[f"surface{surface['id']}_{field}_{axis}"] = value
            for source, definition in zip(case["sources"], results["sources"]):
                name = ("wire" if definition["wire"] else "coil") + str(source["id"])
                row[name+"_current_A"] = source["current_A"]
                row[name+"_linkage_Wb_turn"] = source["linkage_Wb_turn"]
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
