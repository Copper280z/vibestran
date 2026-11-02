#!/usr/bin/env python3
"""Small, varied physical decks for PCG profiling (no external mesher).

Magnetic sizes are deliberately modest. Check CPU direct wall time on the
benchmark machine before increasing them; always time-limit iterative runs.
"""
import argparse
from pathlib import Path

from magnetic_examples import deck as magnetic_deck
from magnetic_planar_examples import planar_deck


def thermal_deck(n=24):
    """CHEXA8 conduction across a 100:1 conductivity interface, hot/cold ends."""
    lines = ["SOL 153", "CEND", "SUBCASE 1", "SPC=1", "THERMAL=ALL",
             "FLUX=ALL", "BEGIN BULK", "MAT4,1,1", "MAT4,2,100",
             "PSOLID,1,1", "PSOLID,2,2"]
    def node(i, j, k):
        return 1 + i + (n+1)*(j + (n+1)*k)
    for k in range(n+1):
        for j in range(n+1):
            for i in range(n+1):
                v = node(i, j, k)
                lines.append(f"GRID,{v},,{i/n:.12g},{j/n:.12g},{k/n:.12g}")
                if i in (0, n):
                    lines.append(f"SPC,1,{v},0,{100 if i == 0 else 0}")
    eid = 0
    for k in range(n):
        for j in range(n):
            for i in range(n):
                eid += 1
                vertices = [node(i,j,k), node(i+1,j,k), node(i+1,j+1,k), node(i,j+1,k),
                            node(i,j,k+1), node(i+1,j,k+1), node(i+1,j+1,k+1), node(i,j+1,k+1)]
                lines += [f"CHEXA,{eid},{1 if i < n//2 else 2}," +
                          ",".join(map(str, vertices[:6])) + ",+",
                          "+," + ",".join(map(str, vertices[6:]))]
    return "\n".join(lines + ["ENDDATA", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {"thermal_interface": thermal_deck(),
             "magnetic_winding": planar_deck("winding", n=64, element="mixed"),
             "magnetic_motor": planar_deck("motor", n=64, element="quad", current=50),
             "magnetic_coil": magnetic_deck("coil", n=10, element="tet")}
    for name, deck in cases.items():
        if deck is None:
            parser.error(f"Could not generate {name}")
        (args.output / f"{name}.bdf").write_text(deck)


if __name__ == "__main__":
    main()
