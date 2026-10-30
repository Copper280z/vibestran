"""Refines the MYSTRAN 'diamond cylinder pressure PLOAD4' shell deck by
splitting every CQUAD4 and CTRIA3 into four, with the new grids on the cylinder
(radius 3 about the y axis), so the diamond pattern is kept.  Original
supports are kept; new grids on the y = 11 end get TY and on the y = 25 end
R1 R3, as the original end grids.

usage: python3 refine_cylinder_deck.py IN.bdf OUT.bdf LEVELS"""

import math
import re
import sys
import numpy as np


def num(s):
    s = s.strip()
    if not s:
        return 0.0
    m = re.match(r"^([+-]?[\d.]+)([+-]\d+)$", s)
    return float(m.group(1) + "e" + m.group(2)) if m else float(s)


def main():
    src, dst, levels = sys.argv[1], sys.argv[2], int(sys.argv[3])
    lines = open(src).read().splitlines()
    head, grids, quads, keep = [], {}, [], []
    pressure = {}
    i = 0
    in_bulk = False
    while i < len(lines):
        l = lines[i]
        if l.startswith("BEGIN BULK"):
            in_bulk = True
            head.append(l)
        elif not in_bulk:
            head.append(l)
        elif l.startswith("GRID*"):
            a = [l[k:k + 16] for k in range(8, 72, 16)]
            b = [lines[i + 1][k:k + 16] for k in range(8, 72, 16)]
            grids[int(a[0])] = np.array([num(a[2]), num(a[3]), num(b[0])])
            i += 1
        elif l.startswith("CQUAD4*"):
            a = [l[k:k + 16] for k in range(8, 72, 16)]
            b = [lines[i + 1][k:k + 16] for k in range(8, 72, 16)]
            quads.append((int(a[0]), int(a[1]), [int(a[2]), int(a[3]), int(b[0]), int(b[1])]))
            i += 1
        elif l.startswith("CTRIA3*"):
            a = [l[k:k + 16] for k in range(8, 72, 16)]
            b = [lines[i + 1][k:k + 16] for k in range(8, 72, 16)]
            quads.append((int(a[0]), int(a[1]), [int(a[2]), int(a[3]), int(b[0])]))
            i += 1
        elif l.startswith("PLOAD4"):
            f = [l[k:k + 8] for k in range(0, 80, 8)]
            pressure[int(f[2])] = (int(f[1]), num(f[3]))
        elif l.startswith("ENDDATA"):
            pass
        else:
            keep.append(l)
        i += 1
    R = 3.0

    def on_cylinder(p):
        r = math.hypot(p[0], p[2])
        return np.array([p[0] * R / r, p[1], p[2] * R / r])

    next_id = max(grids) + 1
    edge_mid = {}

    def mid(a, b):
        nonlocal next_id
        key = tuple(sorted((a, b)))
        if key not in edge_mid:
            grids[next_id] = on_cylinder(0.5 * (grids[a] + grids[b]))
            edge_mid[key] = next_id
            next_id += 1
        return edge_mid[key]

    for _ in range(levels):
        edge_mid.clear()
        out = []
        new_pressure = {}
        for eid, pid, n in quads:
            if len(n) == 4:
                c0, c1, c2, c3 = n
                m01, m12, m23, m30 = mid(c0, c1), mid(c1, c2), mid(c2, c3), mid(c3, c0)
                c = next_id
                grids[c] = on_cylinder(0.25 * (grids[c0] + grids[c1] + grids[c2] + grids[c3]))
                next_id += 1
                children = ((c0, m01, c, m30), (m01, c1, m12, c), (c, m12, c2, m23),
                            (m30, c, m23, c3))
            else:
                c0, c1, c2 = n
                m01, m12, m20 = mid(c0, c1), mid(c1, c2), mid(c2, c0)
                children = ((c0, m01, m20), (m01, c1, m12), (m20, m12, c2), (m01, m12, m20))
            for kids in children:
                out.append((len(out) + 1, pid, list(kids)))
                new_pressure[len(out)] = pressure[eid]
        quads, pressure = out, new_pressure
    with open(dst, "w") as fh:
        fh.write("\n".join(head) + "\n")
        for gid, p in sorted(grids.items()):
            fh.write(f"{'GRID*':<8s}{gid:<16d}{'':16s}{p[0]:16.9E}{p[1]:16.9E}*\n")
            fh.write(f"{'*':<8s}{p[2]:16.9E}\n")
        for eid, pid, n in quads:
            card = "CQUAD4" if len(n) == 4 else "CTRIA3"
            fh.write("".join(f"{x!s:<8s}" for x in [card, eid, pid, *n]) + "\n")
            sid, p = pressure[eid]
            fh.write("".join(f"{x!s:<8s}" for x in ["PLOAD4", sid, eid]) + f"{p:<8.1E}\n")
        for l in keep:
            fh.write(l + "\n")
        original = {int(l[16:24]) for l in keep if l.startswith("SPC     ")}
        for gid, p in sorted(grids.items()):
            if gid in original:
                continue
            if abs(p[1] - 11.0) < 1e-9:
                fh.write(f"SPC1    1       2       {gid}\n")
            elif abs(p[1] - 25.0) < 1e-9:
                fh.write(f"SPC1    1       46      {gid}\n")
        fh.write("ENDDATA\n")


if __name__ == "__main__":
    main()
