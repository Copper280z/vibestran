#!/usr/bin/env python3
"""Judge the AMG preconditioner by model class, not by one deck.

A change to the AMG setup is judged by how the iterations of each class of
model grow as it is refined: iterations ~ h^-p, where p = 0 is a
mesh-independent preconditioner and p = 1 is the sqrt(cond) rate of a
preconditioner that only removes high frequencies from a second-order problem.
Failures and timeouts are recorded outcomes, not gaps in the table.

    amg_model_suite.py list
    amg_model_suite.py run --binary vibestran --work DIR --results R.jsonl \\
        --backend cuda-pcg --modes constant,rigid [--classes a,b] [--direct] \\
        [--archive DIR] [--manifest M.json --source-commit SHA --source-patch P]
    amg_model_suite.py report R.jsonl

With --direct, each class's coarsest level (or --verify-levels of them) is
verified against a refined direct solve in full precision: same assembled K and F,
the same equations, finite values, translations and rotations separately against
explicit tolerances, the residual on the assembled K, and strain energy. Reactions
are compared in the F06's printed precision over complete tables. The process
exits nonzero if a required comparison fails, or if a solve fails in a mode named
by --require-solved.

Every level of a class has at least 50000 equations, below which the device
backends use a different preconditioner, and generated decks and outputs are
deleted after each level unless archived, so the suite needs little disk. Run it
on the benchmark machine; levels reach about 1M equations.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from generate_pcg_cases import thermal_deck
from magnetic_examples import deck as magnetic_deck
from magnetic_planar_examples import planar_deck

STATIC_HEADER = "SOL 101\nCEND\nDISPLACEMENT = ALL\nSPCFORCES = ALL\nSPC = 1\nLOAD = 2\nBEGIN BULK\n"
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


# name: (generator, refinement parameters, what the parameter does). The parameter
# is proportional to 1 / h, except where the note says otherwise; "structural"
# classes have a displacement solution to verify.
H = "h-refinement"
CLASSES = {
    "thermal-hex": (text_deck(thermal_deck), [40, 64, 100], H),
    "magnetic-motor": (text_deck(lambda n: planar_deck("motor", n=n, element="quad", current=50)), [256, 512, 1024], H),
    "magnetic-coil": (text_deck(lambda n: magnetic_deck("coil", n=n, element="tet")), [32, 40, 48], H),
    "solid-hex": (solid_cantilever, [17, 26, 40], H),
    "solid-tet10": (solid_tet10, [8, 12, 19], H),
    "solid-contrast": (solid_contrast, [17, 26, 40], H),
    "solid-thin": (solid_plate, [80, 160, 320], "in-plane h only: two layers, so the aspect ratio improves from 12.5 to 3.1"),
    "shell-thin": (shell_bending(0.001), [100, 200, 400], H),
    "shell-thick": (shell_bending(0.1), [100, 200, 400], H),
    "shell-membrane": (shell_membrane, [100, 200, 400], H),
    "shell-cylinder-cd": (shell_cylinder, [72, 144, 288], H),
    "shell-foundation": (foundation_plate, [100, 200, 400], H),
    "stiffened-panel": (stiffened_panel, [72, 144, 288], H),
    "space-frame": (space_frame, [21, 33, 52], "domain growth at fixed bay length: size scaling, not h"),
    "rbe-assembly": (rbe_assembly, [64, 128, 192], H),
}
SCALAR_CLASSES = {"thermal-hex", "magnetic-motor", "magnetic-coil"}

# Required agreement with the reference, a direct solve with --refine. Without
# refinement a direct solve carries about cond(K) * eps: 1.1e-6 on solid-thin, where
# the PCG solution was within 6e-10 of the refined answer. The displacement limits
# are the solver's own accuracy target.
TOLERANCES = {
    "translation": 1e-8,   # max |du| / max |u| over T1, T2, T3
    "rotation": 1e-8,      # the same over R1, R2, R3
    "residual": 1e-6,      # ||F - K u|| / ||F||: see verify_against for its floor
    "energy": 1e-8,        # |u'K u - u_ref'K u_ref| / u_ref'K u_ref
    "reaction": 1e-5,      # the F06's SPC forces, in printed precision
}


def write_deck(name, n, path):
    with open(path, "w") as out:
        CLASSES[name][0](out, n)


def add_external_deck(specification):
    """--deck NAME=PATH: one level, a realistic deck rather than a generated family."""
    name, _, source = specification.partition("=")
    text = Path(source).read_text()
    CLASSES[name] = (lambda out, _: out.write(text), [1], "external deck: " + Path(source).name)


def parse_log(text):
    """What a run's debug log says about its AMG solves."""
    solves = re.findall(r"\[amg-pcg\] n=(\d+), nnz=(\d+), setup=([\d.]+) ms, "
                        r"iteration/refinement=([\d.]+) ms, iterations=(\d+), estimated error=([^\s]+)", text)
    hierarchy = re.search(r"\[amg-pcg\] (\d+) levels: .*?operator\+transfer nnz/fine nnz=([\d.]+); "
                          r"operator complexity=([\d.]+), transfer complexity=([\d.]+), grid complexity=([\d.]+)", text)
    if not hierarchy:  # binaries from before these were logged
        hierarchy = re.search(r"\[amg-pcg\] (\d+) levels: .*?operator\+transfer nnz/fine nnz=([\d.]+)", text)
    gpu_rows = re.findall(r"device AMG\] (?:fp32 hierarchy resident: )?(\d+) GPU rows, (\d+) CPU rows", text)
    plan = re.findall(r"device AMG\].*?fixed=(\d+) planned=(\d+) budget=(\d+) bytes", text)
    measured = re.findall(r"device AMG\] planned=\d+ measured=(\d+) bytes", text)
    host = "using host hierarchy" in text
    probes = re.findall(r"cycle probe: valid=(\w+) asymmetry=(\S+) min_energy=(\S+) max_energy=(\S+) "
                        r"contraction=(\S+) last_contraction=(\S+)", text)
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
    if hierarchy and hierarchy.lastindex >= 5:
        out.update(operator_complexity=float(hierarchy.group(3)), transfer_complexity=float(hierarchy.group(4)),
                   grid_complexity=float(hierarchy.group(5)))
    if gpu_rows:
        out["cpu_rows"] = int(gpu_rows[-1][1])
        out["execution"] = "spill" if out["cpu_rows"] else "resident"
    if host:
        out["execution"] = "host"
    if plan:
        out.update(fixed_bytes=int(plan[-1][0]), planned_bytes=int(plan[-1][1]), budget_bytes=int(plan[-1][2]))
    if measured:
        out["measured_bytes"] = int(measured[-1])
    if probes:
        out["cycle_probe"] = {
            "valid": all(p[0] == "true" for p in probes),
            "asymmetry": max(float(p[1]) for p in probes),
            "min_energy": min(float(p[2]) for p in probes),
            "max_energy": max(float(p[3]) for p in probes),
            "contraction": max(float(p[4]) for p in probes),
            "last_contraction": max(float(p[5]) for p in probes),
        }
    return out


def read_table(f06, title):
    """Grid -> six values from every instance of an F06 table, by (subcase order, grid)."""
    values, active, instance = {}, False, -1
    with open(f06) as f:
        for line in f:
            if re.search(r"[A-Z] [A-Z] [A-Z]", line):  # the F06's spaced-out titles
                active = title in line
                if active:
                    instance += 1
            elif active:
                t = line.split()
                if len(t) == 8 and t[0].isdigit():
                    try:
                        values[(instance, int(t[0]))] = [float(x) for x in t[2:]]
                    except ValueError:
                        pass
    return values


def read_displacements(f06):
    return read_table(f06, "D I S P L A C E M E N T")


def relative_difference(a, b):
    """max |a - b| / max |b|; a table's entries must all be in the other's, or it is inf."""
    if set(a) != set(b):
        return math.inf
    top = max((abs(x) for row in b.values() for x in row), default=0.0)
    worst = max((abs(x - y) for g, row in a.items() for x, y in zip(row, b[g])), default=0.0)
    return worst / top if top > 0 else worst


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_market_values(path, columns):
    """The body of a Matrix Market file as an array, in full precision."""
    import numpy as np
    with open(path) as f:
        for line in f:
            if not line.startswith("%"):
                shape = [int(v) for v in line.split()]
                break
        return shape, np.loadtxt(f, ndmin=2).reshape(-1, columns)


class AssembledMatrix:
    """K from --dump-system, for products; lower-triangle or full storage."""

    def __init__(self, path):
        import numpy as np
        shape, entries = matrix_market_values(path, 3)
        self.n = shape[0]
        self.rows, self.cols = entries[:, 0].astype(np.int64) - 1, entries[:, 1].astype(np.int64) - 1
        # Extended where the platform has it: a double product's rounding can pass
        # the residual of an ill-conditioned K. The decimals are parsed as doubles
        # first, since they round-trip only there, and K must not be perturbed.
        self.values = entries[:, 2].astype(np.longdouble)
        with open(path) as f:
            self.symmetric = "symmetric" in f.readline() or not (self.rows < self.cols).any()

    def product(self, x):
        import numpy as np
        x = x.astype(np.longdouble)
        y = np.zeros(self.n, dtype=np.longdouble)
        np.add.at(y, self.rows, self.values * x[self.cols])
        if self.symmetric:
            lower = self.rows != self.cols
            np.add.at(y, self.cols[lower], self.values[lower] * x[self.rows[lower]])
        return y


def load_system(prefix, with_matrix):
    """{subcase: dict} from --dump-system files; the matrix is loaded only if asked for."""
    import numpy as np
    out = {}
    for path in sorted(prefix.parent.glob(prefix.name + "_sc*_K.mtx")):
        stem = str(path)[:-len("_K.mtx")]
        out[stem.rsplit("_sc", 1)[1]] = {
            "k_sha256": sha256(path),
            "f_sha256": sha256(stem + "_F.mtx"),
            "k": AssembledMatrix(path) if with_matrix else None,
            "f": matrix_market_values(stem + "_F.mtx", 1)[1].ravel(),
            "u": matrix_market_values(stem + "_u.mtx", 1)[1].ravel(),
            "dofs": np.loadtxt(stem + "_dofs.txt", dtype=np.int64, ndmin=2),
        }
    return out


def verify_against(system, reference, reactions, reference_reactions):
    """The checks of one run against a direct reference: a list of failures and the measures."""
    import numpy as np
    failures, measures, floors = [], {}, {}
    if set(system) != set(reference) or not system:
        return [f"subcases {sorted(system)} differ from the reference's {sorted(reference)}"], measures
    for sc, run in system.items():
        ref = reference[sc]
        u, u_ref = run["u"], ref["u"]
        if len(u) != len(u_ref) or not np.array_equal(run["dofs"], ref["dofs"]):
            failures.append(f"subcase {sc}: equations differ from the reference's")
            continue
        if run["k_sha256"] != ref["k_sha256"] or run["f_sha256"] != ref["f_sha256"]:
            failures.append(f"subcase {sc}: assembled K or F differs from the reference's")
            continue
        if not np.isfinite(u).all():
            failures.append(f"subcase {sc}: non-finite solution")
            continue
        translations = ref["dofs"][:, 2] <= 3
        difference = np.abs(u - u_ref)
        for name, mask in (("translation", translations), ("rotation", ~translations)):
            top = np.abs(u_ref[mask]).max() if mask.any() else 0.0
            value = float(difference[mask].max() / top) if top > 0 else 0.0
            measures[name] = max(measures.get(name, 0.0), value)
        f = ref["f"].astype(np.longdouble)
        relative_residual = lambda ku: float(np.linalg.norm((f - ku).astype(float)) / np.linalg.norm(ref["f"]))
        ku, ku_ref = ref["k"].product(u), ref["k"].product(u_ref)
        # Even the exact solution rounded to double leaves about eps ||K|| ||u||,
        # which on an ill-conditioned K exceeds any fixed limit, so the reference's
        # own residual sets a floor.
        floors["residual"] = max(floors.get("residual", 0.0), 10 * relative_residual(ku_ref))
        measures["residual"] = max(measures.get("residual", 0.0), relative_residual(ku))
        energy_ref = float(u_ref @ ku_ref)
        measures["energy"] = max(measures.get("energy", 0.0), abs(float(u @ ku) - energy_ref) / abs(energy_ref))
    if not failures:
        if not reference_reactions:
            failures.append("the reference's F06 has no SPC forces")
        measures["reaction"] = relative_difference(reactions, reference_reactions)
        for name, value in measures.items():
            limit = max(TOLERANCES[name], floors.get(name, 0.0))
            if not value <= limit:  # also catches NaN
                failures.append(f"{name} difference {value:.3e} exceeds {limit:.1e}")
    return failures, measures


def run_once(binary, deck, backend, modes, args, dump=False):
    """One solve. Outputs are returned and deleted; the log can be archived."""
    stem = deck.with_name(f"{deck.stem}-{backend}-{modes}")
    log, f06 = stem.with_suffix(".log"), stem.with_suffix(".f06")
    dump_prefix = stem.with_name(stem.name + "-dump")
    command = [*args.prefix.split(), str(binary), f"--backend={backend}", f"--log-file={log}"]
    if modes == "direct":
        command.append("--refine")
    elif modes != "default":
        command.append(f"--amg-modes={modes}")
    if dump:
        command.append(f"--dump-system={dump_prefix}")
    command += [str(deck), str(f06)]
    env = dict(os.environ, VIBESTRAN_AMG_PROBE="1")
    if args.budget_mib and modes != "direct":
        env["VIBESTRAN_PCG_DEVICE_MEMORY_MIB"] = str(args.budget_mib)
    start = time.monotonic()
    outcome, detail = "solved", ""
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
    try:
        _, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        _, stderr = process.communicate()
        outcome = "timeout"
    else:
        if process.returncode != 0:
            outcome, detail = ("crashed" if process.returncode < 0 else "failed"), f"exit {process.returncode}: {stderr[-300:]}"
    record = {"outcome": outcome, "exit": process.returncode, "wall_s": time.monotonic() - start,
              "command": " ".join(command), "budget_mib": args.budget_mib}
    record["status"] = "ok" if outcome == "solved" else (detail or outcome)
    if log.exists():
        text = log.read_text()
        record.update(parse_log(text))
        if outcome != "solved":
            record["log_tail"] = text[-600:]
    system = load_system(dump_prefix, modes == "direct") if dump and outcome == "solved" else {}
    tables = {}
    if f06.exists():
        tables = {"displacements": read_displacements(f06), "reactions": read_table(f06, "S P C   F O R C E S")}
    if system:
        record["system_sha256"] = {sc: {"k": v["k_sha256"], "f": v["f_sha256"]} for sc, v in system.items()}
    if args.archive:
        archive = args.archive / deck.parent.name
        archive.mkdir(parents=True, exist_ok=True)
        tag = f"{backend}-{modes}"
        keep = [log] + ([f06] if dump else []) + sorted(deck.parent.glob(f"{dump_prefix.name}_*"))
        # Only the reference keeps K and F; the checks above show the others match.
        keep = [p for p in keep if p.exists() and (modes == "direct" or not p.name.endswith(("_K.mtx", "_F.mtx")))]
        for path in keep:
            short = path.name.split("-dump_")[-1] if "-dump_" in path.name else path.suffix.lstrip(".")
            with open(path, "rb") as source, gzip.open(archive / f"{tag}-{short}.gz", "wb", compresslevel=6) as out:
                shutil.copyfileobj(source, out)
    for path in deck.parent.glob(stem.name + "*"):
        path.unlink()
    return record, system, tables


def cmd_list(_):
    for name, (_, levels, kind) in CLASSES.items():
        print(f"{name:20s} {levels} {kind}")


def tool_version(*command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def write_manifest(args):
    """What a baseline needs to be reproduced."""
    manifest = {
        "date": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "source_commit": args.source_commit,
        "source_patch": args.source_patch.name if args.source_patch else None,
        "source_patch_sha256": sha256(args.source_patch) if args.source_patch else None,
        "binary_sha256": sha256(args.binary),
        "suite_sha256": sha256(__file__),
        "tolerances": TOLERANCES,
        "budget_mib": args.budget_mib,
        "prefix": args.prefix,
        "timeout_s": args.timeout,
        "machine": {
            "platform": platform.platform(),
            "cpu": tool_version("sh", "-c", "grep -m1 'model name' /proc/cpuinfo"),
            "gpu": tool_version("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        },
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "LD_LIBRARY_PATH")},
    }
    if args.build_dir:
        introspect = lambda what: json.loads(tool_version("meson", "introspect", str(args.build_dir), what) or "null")
        manifest["compilers"] = {
            name: {"id": c.get("id"), "version": c.get("version"), "exelist": c.get("exelist")}
            for name, c in (introspect("--compilers").get("host") or {}).items()
        }
        manifest["dependencies"] = [{"name": d.get("name"), "version": d.get("version")} for d in introspect("--dependencies") or []]
        manifest["options"] = {o["name"]: o["value"] for o in introspect("--buildoptions")
                               if o["name"] in ("buildtype", "optimization", "debug", "b_ndebug", "b_lto", "cpp_std", "cpp_args")}
        commands = json.loads((args.build_dir / "compile_commands.json").read_text())
        manifest["compile_flags"] = sorted({" ".join(flag for flag in c["command"].split() if flag.startswith(("-O", "-m", "-f", "-D", "--std", "-std")))
                                            for c in commands if c["file"].endswith("linear_static.cpp")})
        wrap = args.build_dir.parent / "subprojects" / "amgcl.wrap"
        manifest["amgcl_wrap"] = wrap.read_text() if wrap.exists() else None
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


def cmd_run(args):
    args.work.mkdir(parents=True, exist_ok=True)
    wanted = args.classes.split(",") if args.classes else (list(CLASSES) if not args.deck else [])
    for specification in args.deck:
        add_external_deck(specification)
        wanted.append(specification.split("=")[0])
    unknown = [name for name in wanted if name not in CLASSES]
    if unknown:
        sys.exit(f"unknown classes: {unknown}")
    if args.manifest:
        write_manifest(args)
    modes = args.modes.split(",")
    required = set(args.require_solved.split(",")) if args.require_solved else set()
    problems = []
    with open(args.results, "a") as results:
        for name in wanted:
            levels = [int(v) for v in args.levels.split(",")] if args.levels and CLASSES[name][0] and name in (args.classes or "").split(",") else CLASSES[name][1]
            for index, n in enumerate(levels):
                level_dir = args.work / f"{name}-{n}"
                level_dir.mkdir(parents=True, exist_ok=True)
                deck = level_dir / "deck.bdf"
                start = time.monotonic()
                write_deck(name, n, deck)
                base = {"class": name, "refinement": n, "kind": CLASSES[name][2], "deck_sha256": sha256(deck),
                        "deck_mb": deck.stat().st_size / 1e6, "generate_s": time.monotonic() - start}
                # Scalar classes have no displacements or reactions to compare.
                verify = args.direct and index < args.verify_levels and name not in SCALAR_CLASSES
                reference, reference_tables = {}, {}
                if verify:
                    record, reference, reference_tables = run_once(args.binary, deck, args.direct_backend, "direct", args, dump=True)
                    results.write(json.dumps({**base, **record, "backend": args.direct_backend, "modes": "direct"}) + "\n")
                    results.flush()
                    if record["outcome"] != "solved" or not reference:
                        problems.append(f"{name} n={n}: no direct reference ({record['status'][:100]})")
                        verify = False
                first = None
                for mode in modes:
                    record, system, tables = run_once(args.binary, deck, args.backend, mode, args, dump=verify)
                    solved = record["outcome"] == "solved"
                    if solved and name not in SCALAR_CLASSES and not tables.get("displacements"):
                        problems.append(f"{name} n={n} {mode}: solved, but the F06 has no displacements")
                    if verify and solved:
                        failures, measures = verify_against(system, reference, tables.get("reactions", {}), reference_tables["reactions"])
                        record["comparison"] = {"measures": measures, "failures": failures}
                        problems += [f"{name} n={n} {mode}: {failure}" for failure in failures]
                    if solved and tables.get("displacements"):
                        if first is None:
                            first = tables["displacements"]
                        else:
                            record["difference_from_first_mode"] = relative_difference(tables["displacements"], first)
                    probe = record.get("cycle_probe")
                    if probe and not probe["valid"] and mode in required:
                        problems.append(f"{name} n={n} {mode}: the applied cycle failed its bilinear-form checks: {probe}")
                    if mode in required and not solved:
                        problems.append(f"{name} n={n} {mode}: {record['status'][:100]}")
                    row = {**base, **record, "backend": args.backend, "modes": mode}
                    results.write(json.dumps(row) + "\n")
                    results.flush()
                    print(f"{name} n={n} {mode}: {record['outcome']} n={row.get('n')} iterations={row.get('iterations')} "
                          f"setup={row.get('setup_s')} iterate={row.get('iterate_s')} complexity={row.get('complexity')} "
                          f"execution={row.get('execution')} planned={row.get('planned_bytes')} measured={row.get('measured_bytes')} "
                          f"probe={'valid' if probe and probe['valid'] else probe} "
                          f"comparison={record.get('comparison')}", flush=True)
                shutil.rmtree(level_dir)
    if problems:
        print("\nREQUIRED CHECKS FAILED:", *problems, sep="\n  ", file=sys.stderr)
        sys.exit(1)


def growth(rows):
    """The exponent p of iterations ~ parameter^p from the finest two levels with results."""
    usable = [r for r in rows if r.get("iterations") and r.get("n")]
    if len(usable) < 2:
        return None
    a, b = usable[-2], usable[-1]
    mean = lambda r: sum(r["iterations"]) / len(r["iterations"])
    ratio = b["refinement"] / a["refinement"]
    return math.log(mean(b) / mean(a)) / math.log(ratio) if ratio > 1 and mean(a) > 0 else None


def cmd_report(args):
    rows = [json.loads(line) for line in open(args.results)]
    for r in rows:
        if r["class"] not in CLASSES:
            CLASSES[r["class"]] = (None, [], r.get("kind", H))
    table = {}
    for r in rows:
        if r["modes"] != "direct":
            table.setdefault(r["class"], {}).setdefault(r["modes"], {})[r["refinement"]] = r
    print("| class | modes | equations | iterations (mean per solve) | p | setup s | iterate s | op+transfer complexity | device B/DOF | execution | cycle rate | worst comparison |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, by_mode in table.items():
        for mode, levels in by_mode.items():
            ordered = [levels[k] for k in sorted(levels)]
            join = lambda f: " / ".join(f(r) for r in ordered)
            def iters(r):
                if r["status"] != "ok" and not r.get("iterations"):
                    return r.get("outcome", r["status"].split(":")[0])
                return f"{sum(r['iterations']) / max(1, len(r['iterations'])):.0f}" + ("" if r["status"] == "ok" else "!")
            def bytes_per_dof(r):
                return format(r["planned_bytes"] / r["n"], ".0f") if r.get("planned_bytes") and r.get("n") else "-"
            def rate(r):
                return format(r["cycle_probe"]["last_contraction"], ".3f") if r.get("cycle_probe") else "-"
            def worst(r):
                comparison = r.get("comparison")
                if not comparison:
                    return None
                if comparison["failures"]:
                    return "FAILED"
                return max(comparison["measures"].values(), default=0.0)
            p = growth([r for r in ordered if r["status"] == "ok"])
            comparisons = [v for v in map(worst, ordered) if v is not None]
            shown = "FAILED" if "FAILED" in comparisons else (format(max(comparisons), ".1e") if comparisons else "")
            print(f"| {name} | {mode} | {join(lambda r: str(r.get('n')))} | {join(iters)} | "
                  f"{'' if p is None else format(p, '.2f')} | "
                  f"{join(lambda r: format(r.get('setup_s') or 0, '.2f'))} | "
                  f"{join(lambda r: format(r.get('iterate_s') or 0, '.2f'))} | "
                  f"{join(lambda r: format(r.get('complexity') or 0, '.2f'))} | "
                  f"{join(bytes_per_dof)} | {join(lambda r: r.get('execution') or '-')} | {join(rate)} | {shown} |")
    notes = {name: CLASSES[name][2] for name in table if CLASSES[name][2] != H}
    for name, note in notes.items():
        print(f"\n{name}: {note}")


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
    run.add_argument("--deck", action="append", default=[], help="NAME=PATH: also run this deck as a class of its own")
    run.add_argument("--levels", help="override the refinement parameters, comma separated")
    run.add_argument("--direct", action="store_true", help="verify the coarsest levels against a direct solve")
    run.add_argument("--verify-levels", type=int, default=1, help="how many of the coarsest levels to verify")
    run.add_argument("--require-solved", help="comma-separated modes whose failure to solve fails the run")
    run.add_argument("--budget-mib", type=int, default=0, help="device memory budget for every PCG run")
    run.add_argument("--archive", type=Path, help="keep logs, full-precision solutions and the reference K and F here, gzipped")
    run.add_argument("--manifest", type=Path, help="write what reproduces this run: source, build, machine, tolerances")
    run.add_argument("--source-commit")
    run.add_argument("--source-patch", type=Path, help="the working-tree patch on top of the commit")
    run.add_argument("--build-dir", type=Path, help="the Meson build directory of the binary, for the manifest")
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
