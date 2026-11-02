#!/usr/bin/env python3
"""Prepare >=200k-grid physical PCG cases using the repository generators.

First generate the full radial motor with:
  .venv/bin/python Magnetic_Validation/mesh_study.py decks build/pcg-large-cases --mesh tri:0.225
Then run this script on its motor.bdf. Retain the full geometry/materials and
select the rated zero-angle operating point instead of the 78-position sweep.
The existing plate262k.bdf supplies the structural case unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from generate_pcg_cases import thermal_deck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thermal-side", type=int, default=60)
    args = parser.parse_args()
    if args.thermal_side < 58:
        parser.error("thermal-side must be at least 58 for >=200k grids")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.motor.read_text()
    bulk = source.split("BEGIN BULK\n", 1)[1].splitlines()
    kept, skipping = [], False
    for line in bulk:
        if line.startswith(("MOTION,", "DRIVE,")):
            skipping = True
            continue
        if skipping and line.startswith(","):
            continue
        skipping = False
        if line != "ENDDATA":
            kept.append(line)
    # The validation motor's rated phase currents at rotor position zero:
    # peak 10 A, theta0 = 15 electrical degrees, three phases 120 deg apart.
    currents = [-10 * math.sin(math.radians(-15-120*k)) for k in range(3)]
    kept += [f"ICIRC,1,{k+1},{current:.17g}" for k, current in enumerate(currents)]
    motor = "\n".join(["SOL MAGSTAT", "CEND", "SUBCASE 1",
                        "LABEL = FULL RADIAL MOTOR RATED AT ZERO DEGREES",
                        "LOAD = 1", "BFIELD = ALL", "MPROBE = ALL", "BEGIN BULK",
                        *kept, "ENDDATA", ""])
    n = args.thermal_side
    heat = thermal_deck(n).replace("SPC=1\n", "SPC=1\nLOAD=2\n", 1)
    # A localized heat-generating region crossing the material interface makes
    # this a three-dimensional conduction load, not only a linear end gradient.
    heated = [1+i+n*(j+n*k) for k in range(n) for j in range(n) for i in range(n)
              if all(.35 < (v+.5)/n < .65 for v in (i,j,k))]
    heat = heat.replace("ENDDATA", "QVOL,2,1000,"+",".join(map(str, heated))+"\nENDDATA")
    decks = {"motor218k": motor, "thermal227k": heat}
    metadata = {"motor_source": str(args.motor),
                "motor_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "motor_phase_currents_A": currents, "thermal_side": n, "cases": {}}
    for name, text in decks.items():
        path = args.output / f"{name}.bdf"
        path.write_text(text)
        counts = {card: sum(line.startswith(card+",") for line in text.splitlines())
                  for card in ("GRID", "CTRIA3", "CHEXA")}
        if counts["GRID"] < 200000:
            parser.error(f"{name} has only {counts['GRID']} grids; refine the source mesh")
        metadata["cases"][name] = {"path": str(path), "counts": counts,
                                    "sha256": hashlib.sha256(text.encode()).hexdigest()}
        print(name, counts)
    (args.output / "large-cases.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
