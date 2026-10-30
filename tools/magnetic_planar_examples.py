#!/usr/bin/env python3
"""Planar MAGSTAT examples; no third-party mesher or reference solver needed."""
import argparse
import math
from pathlib import Path


def planar_deck(kind="winding", n=12, element="quad", depth=.02, angle=0., current=1.):
    """Return a deck, or None when the parameters cannot describe the example."""
    if n < 2 or depth <= 0 or abs(angle) > 20:
        return None
    motor = kind == "motor"
    if motor:
        landmarks = [-2,-1.5,-1.25,-1,-.75,-.625,-.5,-.25,0,.25,.5,.625,.75,1,1.25,1.5,2]
        axis = []
        for a, b in zip(landmarks,landmarks[1:]):
            parts = max(1,math.ceil((b-a)*n/4))
            axis += [a+(b-a)*i/parts for i in range(parts)]
        axis.append(2)
    else:
        axis = [-1+2*i/n for i in range(n+1)]
    size = len(axis)
    scale = .01 if motor else 1
    radians = math.radians(angle)
    original = [(x,y) for y in axis for x in axis]
    nodes = []
    for x,y in original:
        weight = max(0.,min(1.,(.75-max(abs(x),abs(y)))/.25)) if motor else 0
        a = radians*weight
        nodes.append((scale*(x*math.cos(a)-y*math.sin(a)),scale*(x*math.sin(a)+y*math.cos(a))))
    cells = []
    for j in range(size-1):
        for i in range(size-1):
            a = 1+i+size*j
            q = [a,a+1,a+1+size,a+size]
            if element == "tri" or (element == "mixed" and i < (size-1)//2):
                cells += [q[:3],[q[0],q[2],q[3]]]
            else:
                cells.append(q)
    lines = ["SOL MAGSTAT","CEND", "MAGBC=1" if kind == "uniform" else "LOAD=1", "BEGIN BULK",
             f"MAG2D,0,{depth:.16g}","MATMAG,1,1","PMAG2D,1,1"]
    if motor:
        # Rotor frame 30 turns with the rotor; the magnets follow its x axis.
        lines += ["MATMAG,2,,1","PMAG2D,2,2",
                  f"CORD2R,30,,0,0,0,0,0,1,{math.cos(radians):.16g},{math.sin(radians):.16g},0",
                  "MATMAG,3,1.05,,1.2","PMAG2D,3,3,30,1"]
        lines += ["TABLEBH,1", ",0,0,1,200,1.5,1200", ",1.8,8000,2,50000,ENDT"]
    for i,(x,y) in enumerate(nodes,1): lines.append(f"GRID,{i},,{x:.16g},{y:.16g},0")
    inside, regions, edges = set(), {}, {}
    for eid,c in enumerate(cells,1):
        x,y = (sum(original[v-1][k] for v in c)/len(c) for k in (0,1))
        ax,ay = abs(x),abs(y)
        material, region = 1, 0
        if motor:
            if (1.25 < max(ax,ay) < 1.5 or (.75 < ax < 1.25 and ay < .25) or
                (.75 < ay < 1.25 and ax < .25) or (ax < .25 and ay < .25)): material = 2
            if .25 < ax < .5 and ay < .25: material = 3
            if .75 < ax < 1.25 and .25 < ay < .5:
                region = (1 if x*y > 0 else 2)
            if max(ax,ay) < .625: inside.add(eid)
        elif kind == "winding" and .25 < ax < .75 and ay < .5:
            region = 1 if x < 0 else 2
        if region: regions.setdefault(region,[]).append(eid)
        card = "CTRIA3" if len(c) == 3 else "CQUAD4"
        lines.append(f"{card},{eid},{material},"+",".join(map(str,c)))
        for a,b in zip(c,c[1:]+c[:1]): edges.setdefault(tuple(sorted((a,b))),[]).append((eid,a,b))
    if kind == "uniform": lines.append("MBC,1,0,UNIFORM,.1,.2,0")
    else:
        if set(regions) != {1,2}: return None
        # Forward and return legs are sources 11 and 12 of circuit 1.
        for region,ids in regions.items():
            for eid in ids: lines.append(f"ESET,{region},{eid}")
            lines.append(f"COIL2D,{10+region},{region},{100 if region == 1 else -100}")
        lines += ["CIRCUIT,1,,11,1,12,1", f"ICIRC,1,1,{current:.16g}"]
    if motor:
        for adjacent in edges.values():
            selected = [e for e in adjacent if e[0] in inside]
            if len(adjacent) == 2 and len(selected) == 1:
                eid,a,b = selected[0]
                lines.append(f"FSET,1,{eid},{a},{b}")
        lines.append("MFORCE,1,FACES,1,0")
        lines.append("MPROBE,1,0,.006,.001,0")
    else: lines.append("MPROBE,1,0,0,0,0")
    return "\n".join(lines+["ENDDATA",""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=["uniform","winding","motor"])
    parser.add_argument("output",type=Path)
    parser.add_argument("--n",type=int,default=12)
    parser.add_argument("--element",choices=["tri","quad","mixed"],default="quad")
    parser.add_argument("--depth",type=float,default=.02)
    parser.add_argument("--angle",type=float,default=0)
    parser.add_argument("--current",type=float,default=1)
    args = parser.parse_args()
    text = planar_deck(args.kind,args.n,args.element,args.depth,args.angle,args.current)
    if text is None:
        parser.error(f"Cannot generate {args.kind}: n={args.n}, element={args.element}, "
                     f"depth={args.depth}, angle={args.angle}; require n>=2, positive "
                     "depth, angle within +/-20 degrees, and a mesh resolving both winding regions")
    args.output.write_text(text)


if __name__ == "__main__":
    main()
