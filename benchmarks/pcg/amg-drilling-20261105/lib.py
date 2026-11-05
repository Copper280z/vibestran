# Shared by the drilling-mode studies: deck writers and one measured run. No PCG budget.
import json, math, os, random, re, sys
import numpy as np
from scipy.spatial import Delaunay
sys.path.insert(0, os.path.expanduser("~/vibestran-pcg-perf/tools"))
import amg_model_suite as suite
OUT = os.path.expanduser("~/pcg-perf/drilling-20261105")
TMP = "/tmp/drill"
LIB = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12"
os.makedirs(TMP, exist_ok=True)
MAT = "MAT1,1,1.+9,,0.3\n"
def maxdisp(path):
    if not os.path.exists(path): return None
    top, on = 0.0, False
    for line in open(path, errors="replace"):
        if "D I S P L A C E M E N T" in line: on = True; continue
        if on:
            if "F O R C E" in line or "S T R E S S" in line: break
            p = line.split()
            if len(p) == 8 and p[0].isdigit(): top = max(top, *(abs(float(x)) for x in p[2:5]))
    return top
def run(tag, deck, modes, binary, variant=None, timeout=900, backend="cuda-pcg"):
    env = {**os.environ, "LD_LIBRARY_PATH": LIB, "OMP_NUM_THREADS": "8"}
    for v in ("VIBESTRAN_PCG_DEVICE_MEMORY_MIB", "VIBESTRAN_CUDSS_DEVICE_MEMORY_MIB", "VIBESTRAN_DRILL_VARIANT"): env.pop(v, None)
    if variant: env["VIBESTRAN_DRILL_VARIANT"] = variant
    name = f"{tag}-{modes}" + (f"-{variant}" if variant else "")
    log, f06 = f"{OUT}/{name}.log", f"{TMP}/{name}.f06"
    if os.path.exists(log): os.remove(log)
    outcome, detail, code, wall, rss = suite.run_and_measure(
        ["taskset", "-c", "0-7", binary, f"--backend={backend}", f"--amg-modes={modes}", "--no-quality-checks", f"--log-file={log}", deck, f06], env, timeout)
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    s = re.search(r"setup=([\d.]+) ms, iteration/refinement=([\d.]+) ms, iterations=(\d+), estimated error=([\d.e+-]+)", text)
    split = re.search(r"(\d+) GPU rows, (\d+) CPU rows.*fixed=(\d+) planned=(\d+)", text)
    lv = re.search(r"\[amg-pcg\] (\d+) levels: (\d+)x\d+ \((\d+) nnz, P (\d+) nnz, R \d+ nnz\), (\d+)x\d+ \((\d+) nnz", text)
    row = {"tag": tag, "modes": modes, "variant": variant, "outcome": outcome, "exit": code, "wall_s": round(wall, 1),
        "setup_s": round(float(s[1]) / 1000, 2) if s else None, "iter_s": round(float(s[2]) / 1000, 2) if s else None,
        "iterations": int(s[3]) if s else None, "error": s[4] if s else None,
        "gpu_rows": int(split[1]) if split else None, "cpu_rows": int(split[2]) if split else None,
        "hier_mib": round(int(split[4]) / 2**20) if split else None,
        "levels": int(lv[1]) if lv else None, "n": int(lv[2]) if lv else None, "p_nnz": int(lv[4]) if lv else None, "n1": int(lv[5]) if lv else None,
        "max_disp": maxdisp(f06), "rss_mib": round(rss / 2**20) if rss else None}
    with open(f"{OUT}/runs.jsonl", "a") as out: out.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    if os.path.exists(f06): os.remove(f06)
    return row
def write_shell(name, xyz, quads, trias, clamp, loaded, load="inplane", t=0.01):
    path = f"{TMP}/{name}.bdf"
    with open(path, "w") as out:
        out.write(suite.STATIC_HEADER + MAT + f"PSHELL,1,1,{t:.6g},1,,1\n")
        out.write("".join(f"GRID,{i + 1},,{p[0]:.12g},{p[1]:.12g},{p[2]:.12g}\n" for i, p in enumerate(xyz)))
        e = 1
        for q in quads: out.write(f"CQUAD4,{e},1,{q[0] + 1},{q[1] + 1},{q[2] + 1},{q[3] + 1}\n"); e += 1
        for q in trias: out.write(f"CTRIA3,{e},1,{q[0] + 1},{q[1] + 1},{q[2] + 1}\n"); e += 1
        suite.spc1(out, 123456, [int(i) + 1 for i in clamp])
        suite.forces(out, [int(i) + 1 for i in loaded], 1e4 if load == "inplane" else 1.0, (0, 1, 0) if load == "inplane" else (0, 0, -1))
        out.write("ENDDATA\n")
    return path
def grid_sheet(name, n, is_quad, load="inplane"):
    nid = lambda i, j: i + (n + 1) * j
    xyz = [(i / n, j / n, 0.0) for j in range(n + 1) for i in range(n + 1)]
    quads, trias = [], []
    for j in range(n):
        for i in range(n):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            if is_quad(i, j): quads.append((a, b, c, d))
            else: trias += [(a, b, c), (a, c, d)]
    return write_shell(name, xyz, quads, trias, [nid(i, 0) for i in range(n + 1)], [nid(i, n) for i in range(n + 1)], load)
def delaunay(points_per_side, seed=7):
    # Random interior points kept apart by a jittered lattice, a plain boundary, and a hole.
    rng = np.random.default_rng(seed)
    m = points_per_side; h = 1.0 / m
    pts = []
    for j in range(1, m):
        for i in range(1, m):
            x = (i + 0.5 * (j % 2)) * h + rng.uniform(-.3, .3) * h; y = j * h * 0.98 + 0.01 + rng.uniform(-.3, .3) * h
            if 0.02 < x < 0.98 and math.hypot(x - .5, y - .5) > 0.15 + .5 * h: pts.append((x, y))
    for i in range(m + 1):
        pts += [(i * h, 0.0), (i * h, 1.0)]
        if 0 < i < m: pts += [(0.0, i * h), (1.0, i * h)]
    k = int(2 * math.pi * .15 / h)
    pts += [(.5 + .15 * math.cos(2 * math.pi * a / k), .5 + .15 * math.sin(2 * math.pi * a / k)) for a in range(k)]
    pts = np.array(pts)
    tri = Delaunay(pts).simplices
    centre = pts[tri].mean(axis=1)
    tri = tri[np.hypot(centre[:, 0] - .5, centre[:, 1] - .5) > .15]
    return pts, tri
def oriented(pts, t):
    a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
    return tuple(t) if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0 else (t[0], t[2], t[1])
def unstructured_allquad(name, side, load="inplane"):
    # Each Delaunay triangle splits into three quadrilaterals: a conforming all-quad mesh of irregular valence.
    pts, tri = delaunay(side)
    xyz = [tuple(p) + (0.0,) for p in pts]
    mid = {}
    def edge(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            mid[key] = len(xyz); xyz.append(tuple((pts[a] + pts[b]) / 2) + (0.0,))
        return mid[key]
    quads = []
    for t in tri:
        a, b, c = oriented(pts, t)
        ab, bc, ca = edge(a, b), edge(b, c), edge(c, a)
        o = len(xyz); xyz.append(tuple((pts[a] + pts[b] + pts[c]) / 3) + (0.0,))
        quads += [(a, ab, o, ca), (b, bc, o, ab), (c, ca, o, bc)]
    arr = np.array(xyz)
    return write_shell(name, xyz, quads, [], np.where(arr[:, 1] < 1e-12)[0], np.where(arr[:, 1] > 1 - 1e-12)[0], load)
def unstructured_merged(name, side, load="inplane", min_quality=.5):
    # Greedy merging of Delaunay triangle pairs into convex quadrilaterals, best corners first; the rest stay triangles.
    pts, tri = delaunay(side)
    tris = [oriented(pts, t) for t in tri]
    by_edge = {}
    for f, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)): by_edge.setdefault((min(u, v), max(u, v)), []).append(f)
    def quality(q):
        worst = 1.0
        for i in range(4):
            p, a, b = pts[q[i]], pts[q[(i + 1) % 4]], pts[q[i - 1]]
            u, v = a - p, b - p
            cross = u[0] * v[1] - u[1] * v[0]
            if cross <= 1e-14: return -1
            angle = math.atan2(cross, u @ v)
            worst = min(worst, 1 - abs(angle - math.pi / 2) / (math.pi / 2))
        return worst
    cand = []
    for (u, v), fs in by_edge.items():
        if len(fs) != 2: continue
        a = next(x for x in tris[fs[0]] if x not in (u, v)); b = next(x for x in tris[fs[1]] if x not in (u, v))
        t = tris[fs[0]]; i = t.index(a)
        q = (a, t[(i + 1) % 3], b, t[(i + 2) % 3])
        s = quality(q)
        if s >= min_quality: cand.append((s, fs[0], fs[1], q))
    cand.sort(key=lambda r: -r[0])
    used, quads = set(), []
    for s, f, g, q in cand:
        if f in used or g in used: continue
        used |= {f, g}; quads.append(q)
    trias = [t for f, t in enumerate(tris) if f not in used]
    xyz = [tuple(p) + (0.0,) for p in pts]
    print(name, "grids", len(xyz), "quads", len(quads), "trias", len(trias), flush=True)
    return write_shell(name, xyz, quads, trias, np.where(pts[:, 1] < 1e-12)[0], np.where(pts[:, 1] > 1 - 1e-12)[0], load)
