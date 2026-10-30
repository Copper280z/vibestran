#!/usr/bin/env python3
"""Generate small MAGSTAT demonstrations without an external mesher.

The motor is a deliberately simple four-tooth stator and two-pole block rotor.
Its dimensions, synthetic steel curve and coarse mesh are illustrative.
"""
import argparse
import itertools
import math
from pathlib import Path


def structured_mesh(axis):
    nodes = list(itertools.product(axis, repeat=3))
    size = len(axis)

    def index(p):
        return 1 + (p[0] * size + p[1]) * size + p[2]

    cells = []
    for voxel in itertools.product(range(size - 1), repeat=3):
        for order in itertools.permutations(range(3)):
            point = list(voxel)
            tetra = [index(point)]
            for direction in order:
                point[direction] += 1
                tetra.append(index(point))
            cells.append(tetra)
    return nodes, cells


def center(nodes, cell):
    return tuple(sum(nodes[n - 1][k] for n in cell) / len(cell) for k in range(3))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def oriented_face(nodes, face, interior):
    p, q, r = (nodes[n - 1] for n in face)
    normal = cross(tuple(q[k] - p[k] for k in range(3)),
                   tuple(r[k] - p[k] for k in range(3)))
    if sum(normal[k] * (p[k] - interior[k]) for k in range(3)) < 0:
        face = [face[0], face[2], face[1]]
    return face


def deck(kind="loop", n=8, extent=2.0, angle=0.0, current=1.0):
    if n < 2 or extent <= 1:
        raise ValueError("n must be >=2 and extent >1")
    if kind == "motor":
        if abs(angle) > 20:
            raise ValueError("Demo mesh deformation supports angles within +/-20 degrees")
        landmarks = [-extent, -1.5, -1.25, -1, -.75, -.625, -.5, -.25,
                     0, .25, .5, .625, .75, 1, 1.25, 1.5, extent]
        if extent <= 1.5:
            raise ValueError("Motor extent must exceed 1.5")
        axis = []
        for a, b in zip(landmarks, landmarks[1:]):
            parts = max(1, math.ceil((b-a)*n/(2*extent)))
            axis.extend(a+(b-a)*i/parts for i in range(parts))
        axis.append(extent)
    elif kind == "loop":
        # Resolve the loop independently of the exterior boundary distance.
        # Keep the probes and source centerlines on the same mesh planes at all
        # levels, and grade the outer air instead of one very stretched layer.
        outer = [extent]
        distance = 1.0
        while distance < extent:
            outer.append(distance)
            distance *= 1.5
        axis = sorted(set([-x for x in outer] + [-.6, -.3, .3, .6] +
                          [-1+2*i/n for i in range(n+1)] + outer))
    else:
        axis = [-extent+2*extent*i/n for i in range(n+1)]
    original, cells = structured_mesh(axis)
    nodes = original[:]
    materials = [1] * len(cells)
    inside = set()
    radians = math.radians(angle)
    if kind == "motor":
        for i, (x, y, z) in enumerate(original):
            weight = max(0.0, min(1.0, (.75-max(abs(x), abs(y)))/.25))
            theta = radians*weight
            nodes[i] = (x*math.cos(theta)-y*math.sin(theta),
                        x*math.sin(theta)+y*math.cos(theta), z)
        for i, cell in enumerate(cells):
            x, y, z = center(original, cell)
            ax, ay, az = abs(x), abs(y), abs(z)
            if az < .5:
                if (1.25 < max(ax, ay) < 1.5 or
                    (.75 < ax < 1.25 and ay < .25) or
                    (.75 < ay < 1.25 and ax < .25) or
                    (ax < .25 and ay < .25)):
                    materials[i] = 2
                if .25 < ax < .5 and ay < .25:
                    materials[i] = 3
            if max(ax, ay, az) < .625:
                inside.add(i)
        nodes = [tuple(.01*x for x in point) for point in nodes]
    lines = ["SOL MAGSTAT", "CEND", "TITLE = " + kind + " magnetic example"]
    lines += ["MAGBC=1"] if kind == "uniform" else ["LOAD=1"]
    lines += ["BEGIN BULK", "MATMAG,1,1", "PSOLID,1,1"]
    if kind == "motor":
        lines += ["MATMAG,2,1,1", "PSOLID,2,2",
                  f"MATMAG,3,1.05,,{1.2*math.cos(radians):.16g},{1.2*math.sin(radians):.16g},0",
                  "PSOLID,3,3"]
        lines += [f"BHPOINT,1,{b},{h}" for b, h in
                  [(0, 0), (1, 200), (1.5, 1200), (1.8, 8000), (2, 50000)]]
    for i, p in enumerate(nodes, 1):
        lines.append(f"GRID,{i},," + ",".join(f"{x:.16g}" for x in p))
    for i, (cell, material) in enumerate(zip(cells, materials), 1):
        lines.append(f"CTETRA,{i},{material}," + ",".join(map(str, cell)))
    if kind == "uniform":
        lines += ["MBC,1,0,UNIFORM,0.1,0.2,0.3"]
    if kind in ("loop", "motor"):
        scale = .01 if kind == "motor" else 1
        lines += ["MATCOND,10,5.8e7", f"PWIRE,10,10,ROUND,{.06*scale}",
                  "PARAM,WIRESIDE,12", "PARAM,WIRELAY,1", f"PARAM,WIRESTEP,{.25*scale}"]
        paths = [[(-.6, -.6, 0), (.6, -.6, 0), (.6, .6, 0), (-.6, .6, 0)]]
        if kind == "motor":
            paths = [[(x, -.375, -.625), (x, .375, -.625),
                      (x, .375, .625), (x, -.375, .625)] for x in (-1, 1)]
        next_grid = len(nodes)+1
        next_element = len(cells)+1
        for path in paths:
            element_ids = []
            for i, p in enumerate(path):
                lines.append(f"GRID,{next_grid+i},," + ",".join(f"{scale*x:.16g}" for x in p))
                eid = next_element+i
                element_ids.append(eid)
                lines.append(f"CWIRE,{eid},10,{next_grid+i},{next_grid+(i+1)%4}")
            lines.append(f"IWIRE,1,{current}," + ",".join(map(str, element_ids)))
            next_grid += 4
            next_element += 4
    if kind in ("coil", "motor"):
        faces = {}
        selected = set()
        if kind == "coil":
            for i, cell in enumerate(cells):
                # Select complete structured voxels, avoiding disconnected
                # corner tetrahedra when dimensions fall between grid planes.
                x, y, z = ((min(nodes[n-1][k] for n in cell) +
                            max(nodes[n-1][k] for n in cell))/2 for k in range(3))
                if .4 < max(abs(x), abs(y)) < 1.1 and abs(z) < .4:
                    selected.add(i)
                    lines.append(f"ESET,1,{i+1}")
        for ci, cell in enumerate(cells):
            for face in itertools.combinations(cell, 3):
                faces.setdefault(tuple(sorted(face)), []).append(ci)
        for face, adjacent in faces.items():
            if kind == "coil" and all(ci in selected for ci in adjacent) and len(adjacent) == 2:
                p = center(nodes, face)
                if abs(p[1]) < 1e-12 and p[0] > .4:
                    oriented = oriented_face(nodes, list(face), (p[0], p[1]-1, p[2]))
                    lines.append(f"FSET,1,{adjacent[0]+1}," + ",".join(map(str, oriented)))
            elif kind == "motor" and len(adjacent) == 2 and ((adjacent[0] in inside) != (adjacent[1] in inside)):
                owner = adjacent[0] if adjacent[0] in inside else adjacent[1]
                oriented = oriented_face(nodes, list(face), center(nodes, cells[owner]))
                lines.append(f"FSET,1,{owner+1}," + ",".join(map(str, oriented)))
        if kind == "coil":
            if not selected:
                raise ValueError("Mesh too coarse to represent coil; use n=12 or finer")
            lines += ["COIL,1,1,20,1", f"ICOIL,1,1,{current}"]
        else:
            lines += ["MFORCE,1,1,0,0,0"]
    lines += ["MPROBE,1,0,0,0", "MPROBE,2,0,0," + ("0.008" if kind == "motor" else "0.3"), "ENDDATA"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["uniform", "loop", "coil", "motor"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--extent", type=float, default=2)
    parser.add_argument("--angle", type=float, default=0)
    parser.add_argument("--current", type=float, default=1)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(deck(args.kind, args.n, args.extent, args.angle, args.current))
    print(args.output)


if __name__ == "__main__":
    main()
