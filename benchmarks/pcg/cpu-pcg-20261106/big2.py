# Whole-deck times on both backends, and how far apart their displacements are.
import os, re, sys, run as R
R.RES = "big2.jsonl"
real_remove = os.remove
def table(path):
    out, on = {}, False
    number = re.compile(r"^\s*(\d+)\s+0\s+(.*)$")
    with open(path, errors="replace") as f:
        for line in f:
            if "D I S P L A C E M E N T S" in line: on = True; continue
            if "S P C" in line or "F O R C E" in line: on = False
            if on:
                m = number.match(line)
                if m: out[int(m[1])] = [float(v) for v in m[2].split()]
    return out
for name, size in [("solid-hex", 70), ("shell-membrane", 1258)]:
    deck = R.deck_for(name, size)
    tables = {}
    for backend in ("cuda-pcg", "cpu-pcg"):
        kept = {}
        def keep(path, kept=kept):
            if path.endswith(".f06"): kept["f06"] = path
            else: real_remove(path)
        os.remove = keep
        R.run(f"{name}-{size}", deck, {}, threads=8, timeout=1200, backend=backend)
        os.remove = real_remove
        tables[backend] = table(kept["f06"]); real_remove(kept["f06"])
    a, b = tables["cuda-pcg"], tables["cpu-pcg"]
    print(name, size, "grids", len(a), len(b), "same grids", a.keys() == b.keys(), flush=True)
    for label, cols in (("translation", range(0, 3)), ("rotation", range(3, 6))):
        top = max((abs(v[c]) for v in a.values() for c in cols), default=0.0)
        gap = max((abs(a[g][c] - b[g][c]) for g in a if g in b for c in cols), default=0.0)
        print(f"  {label}: largest {top:.6e}, largest difference {gap:.3e}, ratio {gap / top if top else 0:.2e} (the F06 prints 7 digits)", flush=True)
