"""Runs the original MYSTRAN-suite shell decks in vibestran and MYSTRAN and
writes the compared quantities to a JSON file.

usage: python3 run_shells.py OUTDIR"""

import json
import os
import shutil
import sys
import deckio

DECKS = "/Users/bob/Documents/vibe-tran/MYSTRAN_Validation/decks"

HEMI = {"mesh 1 (coarsest) TX": ([1], 0), "mesh 2 TX": ([26], 0), "mesh 3 TX": ([107], 0),
        "mesh 4 TX": ([396], 0), "mesh 5 (finest) TX": ([1485], 0),
        "mesh 5 (finest) TY at -y load": ([5645], 1)}
CASES = {
    "hemisphere t=0.04 (QUAD4TYP MITC4)": ("vic/6/NAS S30 hemisphere with cutout 4e-3 mitc4.bdf", HEMI),
    "hemisphere t=0.04 (MITC4+)": ("vic/6/NAS S30 hemisphere with cutout 4e-3 mitc4+.bdf", HEMI),
    "hemisphere t=0.04 (MITC4+ SNORM)": ("vic/6/NAS S30 hemisphere with cutout 4e-3 mitc4+ SNORM.bdf", HEMI),
    "hemisphere t=0.004 (MITC4+)": ("vic/6/NAS S30 hemisphere with cutout 4e-4 mitc4+.bdf", HEMI),
    "hemisphere t=0.004 (MITC4+ SNORM)": ("vic/6/NAS S30 hemisphere with cutout 4e-4 mitc4+ SNORM.bdf", HEMI),
    "hemisphere t=0.004 (MITC4+ SNORM K6ROT)":
        ("vic/6/NAS S30 hemisphere with cutout 4e-4 mitc4+ SNORM K6ROT.bdf", HEMI),
    "twisted plate t=0.6": ("vic/6/NAS S30 twisted plate mit4+.bdf",
                            {f"mesh {i} TZ": ([g], 2) for i, g in
                             enumerate((4, 8, 12, 16, 123, 412), start=1)}),
    "diamond cylinder pressure PLOAD4": ("vic/8/NAS S30 mitc4+ diamond cylinder pressure PLOAD4.bdf",
                                         {"TY grid 106": ([106], 1)}),
    "diamond cylinder pressure PLOAD2": ("vic/8/NAS S30 mitc4+ diamond cylinder pressure PLOAD2.bdf",
                                         {"TY grid 106": ([106], 1)}),
    "thermal rectangle cylinder": ("vic/7/S30 mitc4 thermal strain free rectangle cylinder.bdf",
                                   {"TX 56,66": ([56, 66], 0), "TZ 50,59": ([50, 59], 2),
                                    "TY 45,56,141,76": ([45, 56, 141, 76], 1)}),
    "thermal diamond cylinder": ("vic/7/S30 mitc4 thermal strain free diamond cylinder.bdf",
                                 {"TY 23,45": ([23, 45], 1), "TZ 63,200": ([63, 200], 2),
                                  "TX 58,66": ([58, 66], 0), "TX 243,259": ([243, 259], 0)}),
}


def raasch_corners(path):
    """Free-end corner grids of each Raasch mesh (the FORCE grids with the
    least and greatest y in each mesh), coarse to fine."""
    import re
    lines = open(path).read().splitlines()

    def num(s):
        s = s.strip()
        m = re.match(r"^([+-]?[\d.]+)([+-]\d+)$", s)
        return float(m.group(1) + "e" + m.group(2)) if m else float(s or 0)
    y, forces = {}, []
    for i, l in enumerate(lines):
        if l.startswith("GRID*"):
            y[int(l[8:24])] = num(l[56:72])
        if l.startswith("FORCE*"):
            forces.append(int(l[24:40]))
    # The meshes are 0.508 wide and placed every 1.016 along y.
    groups = {}
    for g in sorted(set(forces), key=lambda g: y[g]):
        groups.setdefault(int(y[g] / 1.016 + 1e-6), []).append(g)
    return {f"mesh {i} ({len(gr) - 1} el. across) TY corners": ([gr[0], gr[-1]], 1)
            for i, (_, gr) in enumerate(sorted(groups.items()), start=1)}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for k in ("10.0", "1000.0"):
        deck = f"raasch_hook_k6rot_{k}.bdf"
        CASES[f"Raasch hook K6ROT={k}"] = (deck, raasch_corners(os.path.join(DECKS, deck)))
    results = {}
    for name, (deck, q) in CASES.items():
        local = os.path.join(out, os.path.basename(deck).replace(" ", "_"))
        shutil.copy(os.path.join(DECKS, deck), local)
        entry = results.setdefault(name, {"deck": deck})
        for solver in ("vibestran", "mystran"):
            try:
                f06, dt = deckio.run(solver, local)
                entry[solver] = {"values": deckio.measure(f06, q), "seconds": round(dt, 2)}
            except Exception as e:
                entry[solver] = {"error": str(e)[:500]}
            print(name, solver, entry[solver], flush=True)
        json.dump(results, open(os.path.join(out, "shell_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
