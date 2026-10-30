"""Runs the CHEXA8 reference models at increasing refinement in vibestran
and MYSTRAN and writes the measured quantities to a JSON file.

usage: python3 run_solids.py OUTDIR [case ...]"""

import json
import os
import sys
import cases
import deckio

SERIES = {
    "twisted_plate": [(cases.twisted_plate, dict(n=n, nw=w)) for n, w in
                      ((8, 2), (16, 4), (32, 6), (48, 8), (64, 10))],
    "twisted_plate_t0.2": [(cases.twisted_plate, dict(n=n, nw=w, t=0.2, F=9.90272e7 * 0.2 / 0.6))
                           for n, w in ((16, 2), (32, 4), (64, 4))],
    "twisted_plate_t0.06": [(cases.twisted_plate, dict(n=n, nw=w, t=0.06, F=9.90272e7 * 0.06 / 0.6))
                            for n, w in ((16, 2), (32, 2), (64, 2), (96, 2))],
    "hemisphere_t0.04": [(cases.hemisphere, dict(n=n, nw=2, t=0.04, F=1.0)) for n in
                         (16, 32, 64, 96)],
    "hemisphere_t0.004": [(cases.hemisphere, dict(n=n, nw=2, t=0.004, F=1.0e-3)) for n in
                          (16, 32, 64, 96, 128)],
    "raasch_hook": [(cases.raasch_hook, dict(nu=a, nv=b, nw=c)) for a, b, c in
                    ((72, 10, 2), (144, 20, 2), (288, 40, 4), (432, 60, 4))],
    "cylinder_pressure_free_split": [(cases.cylinder_pressure,
                                      dict(nth=a, ny=b, nw=2, where="split", supports="free"))
                                     for a, b in ((48, 8), (96, 16), (192, 32))],
    "cylinder_pressure_free_inner": [(cases.cylinder_pressure,
                                      dict(nth=192, ny=32, nw=2, where="inner", supports="free"))],
    "cylinder_pressure_free_outer": [(cases.cylinder_pressure,
                                      dict(nth=192, ny=32, nw=2, where="outer", supports="free"))],
    "cylinder_pressure_deck_split": [(cases.cylinder_pressure,
                                      dict(nth=a, ny=b, nw=c, where="split", supports="deck"))
                                     for a, b, c in ((48, 14, 2), (96, 28, 2), (192, 56, 2),
                                                     (288, 84, 4))],
    "half_cylinder_thermal": [(cases.half_cylinder_thermal, dict(nth=a, ny=b, nw=2)) for a, b in
                              ((24, 16), (48, 32))],
}


def main():
    out = sys.argv[1]
    names = sys.argv[2:] or list(SERIES)
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(out, "solid_results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for name in names:
        for builder, kw in SERIES[name]:
            tag = "_".join(f"{k}{v}" for k, v in kw.items())
            path = os.path.join(out, f"{name}__{tag}.bdf")
            q = builder(path, **kw)
            entry = results.setdefault(name, {}).setdefault(tag, {"mesh": kw})
            for solver in ("vibestran", "mystran"):
                try:
                    f06, dt = deckio.run(solver, path)
                    entry[solver] = {"values": deckio.measure(f06, q), "seconds": round(dt, 2)}
                except Exception as e:  # report and continue
                    entry[solver] = {"error": str(e)[:500]}
                print(name, tag, solver, entry[solver], flush=True)
            json.dump(results, open(res_path, "w"), indent=1)


if __name__ == "__main__":
    main()
