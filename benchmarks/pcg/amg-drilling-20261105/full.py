import os, random, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
BIN = f"{OUT}/vibestran"
stage = sys.argv[1]
if stage == "suite":
    env = {**os.environ, "LD_LIBRARY_PATH": LIB, "OMP_NUM_THREADS": "8"}
    subprocess.run([sys.executable, os.path.expanduser("~/vibestran-pcg-perf/tools/amg_model_suite.py"), "run",
        "--binary", BIN, "--work", f"{TMP}/suite", "--results", f"{OUT}/suite.json", "--backend", "cuda-pcg",
        "--modes", "rigid,drilling", "--no-probe", "--direct", "--prefix", "taskset -c 0-7", "--timeout", "900"], env=env, check=True)
else:
    def pair(tag, deck, remove=True):
        run(tag, deck, "rigid", BIN); run(tag, deck, "drilling", BIN)
        if remove: os.remove(deck)
    for side in (44, 88, 175):
        pair(f"allquad-inplane-{side}", unstructured_allquad(f"aq{side}", side))
    pair("allquad-transverse-175", unstructured_allquad("aqt", 175, "transverse"))
    pair("merged-inplane-205", unstructured_merged("m205", 205))
    pair("merged-inplane-410", unstructured_merged("m410", 410))
    pair("merged-transverse-410", unstructured_merged("mt", 410, "transverse"))
    pts, tri = delaunay(410)
    arr = np.c_[pts, np.zeros(len(pts))]
    pair("alltri-inplane-410", write_shell("at", arr, [], [oriented(pts, t) for t in tri], np.where(pts[:, 1] < 1e-12)[0], np.where(pts[:, 1] > 1 - 1e-12)[0]))
    # A sheet whose elements have random orientation.
    n = 400; r = random.Random(3); nid = lambda i, j: i + (n + 1) * j
    quads = []
    for j in range(n):
        for i in range(n):
            q = (nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1))
            quads.append(q[::-1] if r.random() < .5 else q)
    pair("flipped-inplane-400", write_shell("flip", [(i / n, j / n, 0.0) for j in range(n + 1) for i in range(n + 1)], quads, [], [nid(i, 0) for i in range(n + 1)], [nid(i, n) for i in range(n + 1)]))
    # Closed cylinders: even and odd numbers of elements around.
    for nt in (576, 577):
        nz = 288
        node = lambda i, j: i % nt + nt * j
        xyz = [(math.cos(2 * math.pi * i / nt), math.sin(2 * math.pi * i / nt), 3 * j / nz) for j in range(nz + 1) for i in range(nt)]
        q = [(node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)) for j in range(nz) for i in range(nt)]
        path = f"{TMP}/cyl{nt}.bdf"
        with open(path, "w") as out:
            out.write(suite.STATIC_HEADER + MAT + "PSHELL,1,1,0.01,1,,1\n")
            out.write("".join(f"GRID,{i + 1},,{p[0]:.12g},{p[1]:.12g},{p[2]:.12g}\n" for i, p in enumerate(xyz)))
            for e, c in enumerate(q): out.write(f"CQUAD4,{e + 1},1,{c[0] + 1},{c[1] + 1},{c[2] + 1},{c[3] + 1}\n")
            suite.spc1(out, 123456, [node(i, 0) + 1 for i in range(nt)])
            suite.forces(out, [node(i, nz) + 1 for i in range(nt)], 1e4, (0, 0, 1))
            out.write("ENDDATA\n")
        pair(f"cylinder-axial-{nt}", path)
    pair("rover-complete", f"{OUT}/rover-complete.bdf", False)
    pair("rover-static", f"{OUT}/rover-static.bdf", False)
    # The other PCG backends, on one deck.
    deck = f"{TMP}/membrane.bdf"; suite.write_deck("shell-membrane", 200, deck)
    for backend in ("cuda-pcg-mixed", "vulkan"):
        for modes in ("rigid", "drilling"):
            run(f"membrane-200-{backend}", deck, modes, BIN, backend=backend)
    os.remove(deck)
print("finished", stage, flush=True)
