#!/usr/bin/env python3
"""Judge the AMG preconditioner by model class, not by one deck.

A change to the AMG setup is judged by how the iterations of each class of
model grow with mesh refinement: iterations ~ h^-p, where p = 0 is a
mesh-independent preconditioner and p = 1 is the sqrt(cond) rate of a
preconditioner that only removes high frequencies from a second-order problem.
A change is acceptable only if no class regresses.

    amg_model_suite.py list
    amg_model_suite.py run --binary vibestran --work DIR --results R.jsonl \\
        --backend cuda-pcg --modes constant,rigid [--classes a,b] [--direct]
    amg_model_suite.py report R.jsonl

Every level of a class has at least 50000 equations, below which the device
backends use a different preconditioner, and generated decks and outputs are
deleted after each level, so the suite needs little disk. Run it on the
benchmark machine; levels reach about 1M equations.
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from generate_pcg_cases import thermal_deck
from magnetic_examples import deck as magnetic_deck
from magnetic_planar_examples import planar_deck

STATIC_HEADER = "SOL 101\nCEND\nDISPLACEMENT = ALL\nSPC = 1\nLOAD = 2\nBEGIN BULK\n"
STEEL = "MAT1,1,2.1+11,,0.3\n"


def num(v):
    return f"{v:.12g}"


def chunks(values, size):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def continued(first, values, per_line=8):
    """A free-field card whose list of values runs over continuation lines."""
    head = list(first)
    room = 9 - len(head)
    lines = [",".join(map(str, head + list(values[:room])))]
    lines += ["+," + ",".join(map(str, part)) for part in chunks(list(values[room:]), per_line)]
    return "\n".join(lines) + "\n"


def spc1(out, comps, grids):
    for part in chunks(list(grids), 400):
        out.write(continued(["SPC1", 1, comps], part))


def forces(out, grids, total, direction):
    """Equal nodal forces summing to total along direction, in basic axes."""
    grids = list(grids)
    each = num(total / len(grids))
    d = ",".join(num(v) for v in direction)
    out.write("".join(f"FORCE,2,{g},0,{each},{d}\n" for g in grids))


def hex_card(eid, pid, v):
    return f"CHEXA,{eid},{pid},{v[0]},{v[1]},{v[2]},{v[3]},{v[4]},{v[5]},+\n+,{v[6]},{v[7]}\n"


def hex_block(out, nx, ny, nz, size, origin=(0, 0, 0), first_grid=1, first_eid=1, pid=lambda i, j, k: 1):
    """A structured CHEXA8 block; returns its node(i, j, k) and next ids."""
    def node(i, j, k):
        return first_grid + i + (nx + 1) * (j + (ny + 1) * k)
    lines = []
    for k in range(nz + 1):
        z = num(origin[2] + size[2] * k / nz)
        for j in range(ny + 1):
            y = num(origin[1] + size[1] * j / ny)
            for i in range(nx + 1):
                lines.append(f"GRID,{node(i, j, k)},,{num(origin[0] + size[0] * i / nx)},{y},{z}\n")
        out.write("".join(lines))
        lines.clear()
    eid = first_eid
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                lines.append(hex_card(eid, pid(i, j, k), [
                    node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k),
                    node(i, j, k + 1), node(i + 1, j, k + 1), node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)]))
                eid += 1
        out.write("".join(lines))
        lines.clear()
    return node, node(nx, ny, nz) + 1, eid


def quad_sheet(out, nx, ny, point, first_grid=1, first_eid=1, pid=1, cd=None):
    """A structured CQUAD4 sheet over point(i, j) -> (x, y, z), j rows of nx+1."""
    def node(i, j):
        return first_grid + i + (nx + 1) * j
    suffix = f",{cd}" if cd else ""
    lines = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y, z = point(i, j)
            lines.append(f"GRID,{node(i, j)},,{num(x)},{num(y)},{num(z)}{suffix}\n")
        out.write("".join(lines))
        lines.clear()
    eid = first_eid
    for j in range(ny):
        for i in range(nx):
            lines.append(f"CQUAD4,{eid},{pid},{node(i, j)},{node(i + 1, j)},{node(i + 1, j + 1)},{node(i, j + 1)}\n")
            eid += 1
        out.write("".join(lines))
        lines.clear()
    return node, eid


def solid_cantilever(out, n):
    """A 4:1:1 CHEXA8 bar clamped at one end with a shear load on the other."""
    out.write(STATIC_HEADER + STEEL + "PSOLID,1,1\n")
    node, _, _ = hex_block(out, 4 * n, n, n, (4, 1, 1))
    spc1(out, 123, (node(0, j, k) for k in range(n + 1) for j in range(n + 1)))
    forces(out, (node(4 * n, j, k) for k in range(n + 1) for j in range(n + 1)), 1e6, (0, 0, -1))
    out.write("ENDDATA\n")


def solid_tet10(out, n):
    """The same bar in CTETRA10: six tetrahedra per cell (Kuhn's split)."""
    out.write(STATIC_HEADER + STEEL + "PSOLID,1,1\n")
    nx, ny, nz = 8 * n, 2 * n, 2 * n
    def node(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)
    out.write("".join(f"GRID,{node(i, j, k)},,{num(4 * i / nx)},{num(j / ny)},{num(k / nz)}\n"
                      for k in range(nz + 1) for j in range(ny + 1) for i in range(nx + 1)))
    # Paths from the (0,0,0) corner to the (1,1,1) corner, in units of two
    # lattice steps; those with an odd permutation swap two corners so every
    # tetrahedron has a positive volume.
    paths = [((1, 0, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)), ((0, 0, 1), (1, 0, 1)),
             ((1, 1, 0), (0, 1, 0)), ((0, 1, 1), (0, 0, 1)), ((1, 0, 1), (1, 0, 0))]
    eid = 1
    lines = []
    for k in range(0, nz, 2):
        for j in range(0, ny, 2):
            for i in range(0, nx, 2):
                for b, c in paths:
                    corners = [(0, 0, 0), b, c, (1, 1, 1)]
                    def mid(p, q):
                        return node(i + p[0] + q[0], j + p[1] + q[1], k + p[2] + q[2])
                    v = [mid(p, p) for p in corners]
                    m = [mid(corners[a], corners[d]) for a, d in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))]
                    lines.append(f"CTETRA,{eid},1,{v[0]},{v[1]},{v[2]},{v[3]},{m[0]},{m[1]},+\n+,{m[2]},{m[3]},{m[4]},{m[5]}\n")
                    eid += 1
        out.write("".join(lines))
        lines.clear()
    spc1(out, 123, (node(0, j, k) for k in range(nz + 1) for j in range(ny + 1)))
    forces(out, (node(nx, j, k) for k in range(nz + 1) for j in range(ny + 1)), 1e6, (0, 0, -1))
    out.write("ENDDATA\n")


def solid_contrast(out, n):
    """The hexahedral bar as a 10^6:1 stiffness checkerboard of a fixed pattern."""
    out.write(STATIC_HEADER + "MAT1,1,1.+9,,0.3\nMAT1,2,1.+3,,0.3\nPSOLID,1,1\nPSOLID,2,2\n")
    cell = max(1, n // 4)
    node, _, _ = hex_block(out, 4 * n, n, n, (4, 1, 1),
                           pid=lambda i, j, k: 1 + (i // cell + j // cell + k // cell) % 2)
    spc1(out, 123, (node(0, j, k) for k in range(n + 1) for j in range(n + 1)))
    forces(out, (node(4 * n, j, k) for k in range(n + 1) for j in range(n + 1)), 1e3, (0, 0, -1))
    out.write("ENDDATA\n")


def solid_plate(out, n):
    """A thin plate meshed with two layers of flat hexahedra: high aspect ratio."""
    out.write(STATIC_HEADER + STEEL + "PSOLID,1,1\n")
    node, _, _ = hex_block(out, n, n, 2, (1, 1, 0.002))
    spc1(out, 123, (node(0, j, k) for k in range(3) for j in range(n + 1)))
    forces(out, (node(i, j, 2) for j in range(n + 1) for i in range(1, n + 1)), 100, (0, 0, -1))
    out.write("ENDDATA\n")


def shell_bending(thickness):
    def write(out, n):
        """A clamped square CQUAD4 plate under a uniform transverse load."""
        out.write(STATIC_HEADER + STEEL + f"PSHELL,1,1,{num(thickness)},1,,1\n")
        node, _ = quad_sheet(out, n, n, lambda i, j: (i / n, j / n, 0))
        edge = sorted({node(i, j) for j in range(n + 1) for i in (0, n)} |
                      {node(i, j) for i in range(n + 1) for j in (0, n)})
        spc1(out, 123456, edge)
        forces(out, (node(i, j) for j in range(1, n) for i in range(1, n)), 1000 * thickness ** 3, (0, 0, -1))
        out.write("ENDDATA\n")
    return write


def shell_membrane(out, n):
    """A CQUAD4 sheet in tension: the deck behind the first AMG measurements."""
    out.write(STATIC_HEADER + "MAT1,1,1.+9,5.+8,0.\nPSHELL,1,1,0.01,1,,1\n")
    node, _ = quad_sheet(out, n, n, lambda i, j: (0, j * 1e-3 / n, i * 1e-3 / n))
    spc1(out, 123456, (node(i, 0) for i in range(n + 1)))
    forces(out, (node(i, n) for i in range(n + 1)), 1e4, (0, 1, 0))
    out.write("ENDDATA\n")


def shell_cylinder(out, n):
    """A closed thin cylinder, pinched at one end, with its grids in a cylindrical
    displacement frame. A ring held only radially at mid-length is a partial
    constraint in that frame, so the solver eliminates it through MPCs."""
    out.write(STATIC_HEADER + STEEL + "PSHELL,1,1,0.01,1,,1\n")
    out.write("CORD2C,1,0,0.,0.,0.,0.,0.,1.,+\n+,1.,0.,0.\n")
    nt, nz = 2 * n, n
    def node(i, j):
        return 1 + i % nt + nt * j
    lines = []
    for j in range(nz + 1):
        z = num(3 * j / nz)
        lines += [f"GRID,{node(i, j)},,{num(math.cos(2 * math.pi * i / nt))},{num(math.sin(2 * math.pi * i / nt))},{z},1\n"
                  for i in range(nt)]
    out.write("".join(lines))
    out.write("".join(f"CQUAD4,{i + nt * j + 1},1,{node(i, j)},{node(i + 1, j)},{node(i + 1, j + 1)},{node(i, j + 1)}\n"
                      for j in range(nz) for i in range(nt)))
    spc1(out, 123456, (node(i, 0) for i in range(nt)))
    spc1(out, 1, (node(i, nz // 2) for i in range(nt)))
    forces(out, [node(0, nz)], 1e3, (-1, 0, 0))
    forces(out, [node(n, nz)], 1e3, (1, 0, 0))
    out.write("ENDDATA\n")


def stiffened_panel(out, n):
    """A CQUAD4 skin with offset CBAR stringers and frames of a fixed pattern."""
    out.write(STATIC_HEADER + STEEL + "PSHELL,1,1,0.002,1,,1\nPBAR,2,1,1.-3,2.-6,2.-6,3.-6\n")
    nx, ny = 2 * n, n
    node, eid = quad_sheet(out, nx, ny, lambda i, j: (2 * i / nx, j / ny, 0))
    bars = []
    step_y, step_x = max(1, n // 8), max(1, n // 4)
    for j in range(0, ny + 1, step_y):
        bars += [(node(i, j), node(i + 1, j), (0, 0, 1)) for i in range(nx)]
    for i in range(0, nx + 1, step_x):
        bars += [(node(i, j), node(i, j + 1), (0, 0, 1)) for j in range(ny)]
    for ga, gb, v in bars:
        out.write(f"CBAR,{eid},2,{ga},{gb},{v[0]}.,{v[1]}.,{v[2]}.,,+\n+,,,0.,0.,-0.02,0.,0.,-0.02\n")
        eid += 1
    spc1(out, 123456, (node(0, j) for j in range(ny + 1)))
    forces(out, (node(nx, j) for j in range(ny + 1)), 100, (0, 0, -1))
    out.write("ENDDATA\n")


def space_frame(out, n):
    """A rigid-jointed CBAR lattice, 1 m bays: a structure growing in size."""
    out.write(STATIC_HEADER + STEEL + "PBAR,1,1,1.-2,1.-5,1.-5,2.-5\n")
    row = n + 1
    def node(i, j, k):
        return 1 + i + row * (j + row * k)
    out.write("".join(f"GRID,{node(i, j, k)},,{i}.,{j}.,{k}.\n"
                      for k in range(row) for j in range(row) for i in range(row)))
    eid = 1
    lines = []
    for k in range(row):
        for j in range(row):
            for i in range(row):
                # The orientation vector must not lie along the member.
                if i < n:
                    lines.append(f"CBAR,{eid},1,{node(i, j, k)},{node(i + 1, j, k)},0.,0.,1.\n")
                    eid += 1
                if j < n:
                    lines.append(f"CBAR,{eid},1,{node(i, j, k)},{node(i, j + 1, k)},0.,0.,1.\n")
                    eid += 1
                if k < n:
                    lines.append(f"CBAR,{eid},1,{node(i, j, k)},{node(i, j, k + 1)},1.,0.,0.\n")
                    eid += 1
        out.write("".join(lines))
        lines.clear()
    spc1(out, 123456, (node(i, j, 0) for j in range(row) for i in range(row)))
    forces(out, (node(i, j, n) for j in range(row) for i in range(row)), 1e5, (1, 0, 0))
    out.write("ENDDATA\n")


def foundation_plate(out, n):
    """A CQUAD4 plate on grounded CELAS2 springs: no rigid-body nullspace to speak of."""
    out.write(STATIC_HEADER + STEEL + "PSHELL,1,1,0.01,1,,1\n")
    node, eid = quad_sheet(out, n, n, lambda i, j: (i / n, j / n, 0))
    k = num(1e7 / (n * n))
    lines = []
    for j in range(n + 1):
        for i in range(n + 1):
            lines.append(f"CELAS2,{eid},{k},{node(i, j)},3\n")
            eid += 1
    out.write("".join(lines))
    # The solver rejects a model with no constraints at all.
    spc1(out, 123456, [node(0, 0)])
    forces(out, [node(n // 2, n // 2)], 1000, (0, 0, -1))
    out.write("ENDDATA\n")


def rbe_assembly(out, n):
    """Sixteen CHEXA8 blocks on a clamped CQUAD4 plate: each block's base is an
    RBE2 spider to one plate grid and its load an RBE3 from a reference grid.
    The reduced system mixes 3- and 6-DOF grids and has dense rows."""
    n = 8 * max(1, round(n / 8))
    m = n // 8
    out.write(STATIC_HEADER + STEEL + "PSHELL,1,1,0.02,1,,1\nPSOLID,2,1\n")
    node, eid = quad_sheet(out, n, n, lambda i, j: (4 * i / n, 4 * j / n, 0))
    spc1(out, 123456, sorted({node(i, j) for j in range(n + 1) for i in (0, n)} |
                             {node(i, j) for i in range(n + 1) for j in (0, n)}))
    grid = node(n, n) + 1
    rbe = 10_000_000
    for bj in range(4):
        for bi in range(4):
            cx, cy = bi + 0.5, bj + 0.5
            block, grid, eid = hex_block(out, m, m, m, (0.5, 0.5, 0.5), (cx - 0.25, cy - 0.25, 0.1),
                                         grid, eid, lambda i, j, k: 2)
            anchor = node((2 * bi + 1) * n // 8, (2 * bj + 1) * n // 8)
            out.write(continued(["RBE2", rbe, anchor, 123],
                                [block(i, j, 0) for j in range(m + 1) for i in range(m + 1)]))
            rbe += 1
            ref = grid
            grid += 1
            out.write(f"GRID,{ref},,{num(cx)},{num(cy)},0.6\n")
            out.write(continued(["RBE3", rbe, "", ref, 123456, 1.0, 123],
                                [block(i, j, m) for j in range(m + 1) for i in range(m + 1)]))
            rbe += 1
            out.write(f"FORCE,2,{ref},0,1.+4,0.,0.,-1.\n")
    out.write("ENDDATA\n")


def text_deck(make):
    def write(out, n):
        out.write(make(n))
    return write


# name: (generator, refinement parameters, spatial dimension of the refinement)
CLASSES = {
    "thermal-hex": (text_deck(thermal_deck), [40, 64, 100], 3),
    "magnetic-motor": (text_deck(lambda n: planar_deck("motor", n=n, element="quad", current=50)), [256, 512, 1024], 2),
    "magnetic-coil": (text_deck(lambda n: magnetic_deck("coil", n=n, element="tet")), [32, 40, 48], 3),
    "solid-hex": (solid_cantilever, [17, 26, 40], 3),
    "solid-tet10": (solid_tet10, [8, 12, 19], 3),
    "solid-contrast": (solid_contrast, [17, 26, 40], 3),
    "solid-thin": (solid_plate, [80, 160, 320], 2),
    "shell-thin": (shell_bending(0.001), [100, 200, 400], 2),
    "shell-thick": (shell_bending(0.1), [100, 200, 400], 2),
    "shell-membrane": (shell_membrane, [100, 200, 400], 2),
    "shell-cylinder-cd": (shell_cylinder, [72, 144, 288], 2),
    "shell-foundation": (foundation_plate, [100, 200, 400], 2),
    "stiffened-panel": (stiffened_panel, [72, 144, 288], 2),
    "space-frame": (space_frame, [21, 33, 52], 3),
    "rbe-assembly": (rbe_assembly, [64, 128, 192], 2),
}


def write_deck(name, n, path):
    with open(path, "w") as out:
        CLASSES[name][0](out, n)


def parse_log(text):
    """What a run's debug log says about its AMG solves."""
    solves = re.findall(r"\[amg-pcg\] n=(\d+), nnz=(\d+), setup=([\d.]+) ms, "
                        r"iteration/refinement=([\d.]+) ms, iterations=(\d+), estimated error=([^\s]+)", text)
    hierarchy = re.search(r"\[amg-pcg\] (\d+) levels: .*?operator\+transfer nnz/fine nnz=([\d.]+)", text)
    gpu_rows = re.findall(r"device AMG\] (\d+) GPU rows, (\d+) CPU rows", text)
    out = {
        "n": int(solves[0][0]) if solves else None,
        "nnz": int(solves[0][1]) if solves else None,
        "solves": len(solves),
        "setup_s": sum(float(s[2]) for s in solves) / 1e3,
        "iterate_s": sum(float(s[3]) for s in solves) / 1e3,
        "iterations": [int(s[4]) for s in solves],
        "estimated_error": max((float(s[5]) for s in solves), default=None),
        "levels": int(hierarchy.group(1)) if hierarchy else None,
        "complexity": float(hierarchy.group(2)) if hierarchy else None,
    }
    if gpu_rows:
        out["cpu_rows"] = int(gpu_rows[-1][1])
    return out


def read_displacements(f06):
    """Grid -> six values from the F06 displacement table; empty when it has none."""
    values, active = {}, False
    with open(f06) as f:
        for line in f:
            if "D I S P L A C E M E N T" in line:
                active = True
            elif active and (" S P C " in line or "F O R C E" in line or "S T R E S S" in line):
                active = False
            elif active:
                t = line.split()
                if len(t) == 8 and t[0].isdigit():
                    values[int(t[0])] = [float(x) for x in t[2:]]
    return values


def relative_difference(a, b):
    """max |a - b| / max |b| over the components both tables hold."""
    top = max((abs(x) for row in b.values() for x in row), default=0.0)
    worst = max((abs(x - y) for g, row in a.items() if g in b for x, y in zip(row, b[g])), default=0.0)
    return worst / top if top > 0 else worst


def run_once(binary, deck, backend, modes, env_prefix, timeout):
    stem = deck.with_name(f"{deck.stem}-{backend}-{modes}")
    log, f06 = stem.with_suffix(".log"), stem.with_suffix(".f06")
    command = [*env_prefix, str(binary), f"--backend={backend}", f"--log-file={log}"]
    if modes != "default":
        command.append(f"--amg-modes={modes}")
    command += [str(deck), str(f06)]
    start = time.monotonic()
    status = "ok"
    try:
        done = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if done.returncode != 0:
            status = f"exit {done.returncode}: {done.stderr[-300:]}"
    except subprocess.TimeoutExpired:
        status = "timeout"
    record = {"status": status, "wall_s": time.monotonic() - start}
    if log.exists():
        text = log.read_text()
        record.update(parse_log(text))
        if status != "ok":
            record["log_tail"] = text[-600:]
    displacements = read_displacements(f06) if f06.exists() else {}
    for path in deck.parent.glob(stem.name + "*"):
        path.unlink()
    return record, displacements


def cmd_list(_):
    for name, (_, levels, dim) in CLASSES.items():
        print(f"{name:20s} {dim}D refinements {levels}")


def cmd_run(args):
    args.work.mkdir(parents=True, exist_ok=True)
    wanted = args.classes.split(",") if args.classes else list(CLASSES)
    unknown = [name for name in wanted if name not in CLASSES]
    if unknown:
        sys.exit(f"unknown classes: {unknown}")
    modes = args.modes.split(",")
    with open(args.results, "a") as results:
        for name in wanted:
            levels = [int(v) for v in args.levels.split(",")] if args.levels else CLASSES[name][1]
            for index, n in enumerate(levels):
                level_dir = args.work / f"{name}-{n}"
                level_dir.mkdir(parents=True, exist_ok=True)
                deck = level_dir / "deck.bdf"
                start = time.monotonic()
                write_deck(name, n, deck)
                base = {"class": name, "refinement": n, "dim": CLASSES[name][2],
                        "deck_mb": deck.stat().st_size / 1e6, "generate_s": time.monotonic() - start}
                reference = None
                if args.direct and index == 0:
                    record, reference = run_once(args.binary, deck, args.direct_backend, "default",
                                                 args.prefix.split(), args.timeout)
                    results.write(json.dumps({**base, **record, "backend": args.direct_backend, "modes": "direct"}) + "\n")
                    results.flush()
                first = None
                for mode in modes:
                    record, displacements = run_once(args.binary, deck, args.backend, mode,
                                                     args.prefix.split(), args.timeout)
                    if reference:
                        record["difference_from_direct"] = relative_difference(displacements, reference)
                    if first is None:
                        first = displacements
                    elif first and displacements:
                        record["difference_from_first_mode"] = relative_difference(displacements, first)
                    row = {**base, **record, "backend": args.backend, "modes": mode}
                    results.write(json.dumps(row) + "\n")
                    results.flush()
                    print(f"{name} n={n} {mode}: n={row.get('n')} iterations={row.get('iterations')} "
                          f"setup={row.get('setup_s')} iterate={row.get('iterate_s')} "
                          f"complexity={row.get('complexity')} status={row['status'][:80]}", flush=True)
                shutil.rmtree(level_dir)


def growth(rows):
    """The exponent p of iterations ~ h^-p from the finest two levels with results."""
    usable = [r for r in rows if r.get("iterations") and r.get("n")]
    if len(usable) < 2:
        return None
    a, b = usable[-2], usable[-1]
    ratio = (b["n"] / a["n"]) ** (1 / b["dim"])
    mean = lambda r: sum(r["iterations"]) / len(r["iterations"])
    return math.log(mean(b) / mean(a)) / math.log(ratio) if ratio > 1 and mean(a) > 0 else None


def cmd_report(args):
    rows = [json.loads(line) for line in open(args.results)]
    table = {}
    for r in rows:
        if r["modes"] != "direct":
            table.setdefault(r["class"], {}).setdefault(r["modes"], {})[r["refinement"]] = r
    print("| class | modes | equations | iterations (mean per solve) | p | setup s | iterate s | op. complexity | worst diff |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, by_mode in table.items():
        for mode, levels in by_mode.items():
            ordered = [levels[k] for k in sorted(levels)]
            join = lambda f: " / ".join(f(r) for r in ordered)
            def iters(r):
                if r["status"] != "ok" and not r.get("iterations"):
                    return r["status"].split(":")[0]
                return f"{sum(r['iterations']) / max(1, len(r['iterations'])):.0f}" + ("" if r["status"] == "ok" else "!")
            p = growth([r for r in ordered if r["status"] == "ok"])
            diffs = [v for r in ordered for v in (r.get("difference_from_direct"), r.get("difference_from_first_mode")) if v is not None]
            print(f"| {name} | {mode} | {join(lambda r: str(r.get('n')))} | {join(iters)} | "
                  f"{'' if p is None else format(p, '.2f')} | "
                  f"{join(lambda r: format(r.get('setup_s') or 0, '.1f'))} | "
                  f"{join(lambda r: format(r.get('iterate_s') or 0, '.1f'))} | "
                  f"{join(lambda r: format(r.get('complexity') or 0, '.2f'))} | "
                  f"{format(max(diffs), '.1e') if diffs else ''} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    run = sub.add_parser("run")
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--results", type=Path, required=True)
    run.add_argument("--backend", default="cuda-pcg")
    run.add_argument("--modes", default="default", help="comma-separated --amg-modes values, or default")
    run.add_argument("--classes")
    run.add_argument("--levels", help="override the refinement parameters, comma separated")
    run.add_argument("--direct", action="store_true", help="solve the coarsest level directly for reference")
    run.add_argument("--direct-backend", default="cpu")
    run.add_argument("--prefix", default="", help="command prefix, such as taskset -c 0-7")
    run.add_argument("--timeout", type=float, default=1200)
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report")
    report.add_argument("results", type=Path)
    report.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
