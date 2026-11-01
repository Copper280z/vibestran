#!/usr/bin/env python3
"""Mesh efficiency study of the xfemm comparison's whole motor
(run_xfemm.py), solved by vibestran alone with the sliding band
(PARAM,AIRGAP,BAND): triangles against quadrilaterals at several sizes.

Each mesh is the motor geometry meshed by gmsh with run_xfemm's size field
times a scale (1 is the comparison's coarse level, 0.5 its fine level):
tri, gmsh's triangles; quad, the same triangles recombined by gmsh (a few
triangles remain); fq, gmsh's full-quad recombination (Blossom, then
the leftover triangles split); qm, all quadrilaterals from WebMesh's quadmesh inside
gmsh's boundary nodes (run_xfemm.quadmesh_surfaces). Every mesh
is solved with the comparison's sweeps (rated and no-load over an electrical
period, cogging over a cogging period), and its figures are compared with
a reference mesh's: mean torque, torque ripple and cogging peak to peak, the
phase linkage's fundamental at no load and rated current, and the gap's
radial B fundamental.

    decks OUT --mesh tri:1 tri:0.5 qm:1/0.5 ... write OUT/<kind>_<scale>[-<near>]/motor.bdf
                                                (scale/near: run_xfemm.motor_geometry's
                                                far-field scale and gap scale)
    report OUT --reference tri:0.25             compare every solved mesh
                                                (motor.mag.vtkhdf, and
                                                motor.time: wall seconds)
    plot OUT/<mesh> mesh.png                    draw a deck's mesh

The decks are solved outside the script, so that they can be timed on any
machine and backend; report reads the results and the times.
"""

import argparse
import json
import pathlib

import h5py
import numpy as np

import run_xfemm as rx


def name_of(spec):
    return spec.replace(":", "_").replace("/", "-")


def write_decks(out, specs):
    for spec in specs:
        kind, scale = spec.split(":")
        scale, _, near = scale.partition("/")
        quads = {"tri": False, "quad": True, "fq": "full", "qm": "quadmesh"}[kind]
        g = rx.motor_geometry(float(scale), False, quads=quads, near=float(near) if near else None)
        deck, angles = rx.motor_deck(g, False)
        deck = deck.replace("BEGIN BULK\n", "BEGIN BULK\nPARAM,AIRGAP,BAND\n", 1)
        workdir = out / name_of(spec)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "motor.bdf").write_text(deck)
        nodes = len({n for _, cell in g.cells for n in cell})
        quads = sum(len(cell) == 4 for _, cell in g.cells)
        (workdir / "mesh.json").write_text(json.dumps(
            {"cells": len(g.cells), "quads": quads, "nodes": nodes, "angles": angles.tolist()}))
        print(f"{name_of(spec)}: {nodes} grids, {len(g.cells)} cells ({quads} quadrilaterals)")


def figures(workdir):
    mesh = json.loads((workdir / "mesh.json").read_text())
    with h5py.File(workdir / "motor.mag.vtkhdf", "r") as h5:
        v = rx.motor_results(h5, np.array(mesh["angles"]), False)
    rated = np.array(v["rated"]["torque"])
    cogging = np.array(v["cogging"]["torque"])
    linkage = {k: np.array(v[k]["linkage"])[:, 0] for k in ("rated", "noload")}
    br = np.array(v["rated"]["gap_br"])
    return {
        "mean torque (Nm)": rated.mean(),
        "ripple p-p (Nm)": np.ptp(rated),
        "cogging p-p (Nm)": np.ptp(cogging),
        "no-load linkage fundamental (Wb-turn)": rx.harmonic(linkage["noload"], 1),
        "rated linkage fundamental (Wb-turn)": rx.harmonic(linkage["rated"], 1),
        "gap Br fundamental (T)": rx.harmonic(br, rx.motor.POLE_PAIRS),
    }


def read_deck(workdir):
    """The deck's grid coordinates and cells (lists of coordinates)."""
    grids, cells = {}, []
    for line in (workdir / "motor.bdf").read_text().splitlines():
        f = line.split(",")
        if f[0] == "GRID":
            grids[f[1]] = (float(f[3]), float(f[4]))
        elif f[0] in ("CTRIA3", "CQUAD4"):
            cells.append(f[3:])
    return [np.array([grids[n] for n in c]) for c in cells]


def corner_angles(cell):
    """Interior angles in degrees, negative at a reflex (inverting) corner."""
    a = cell - np.roll(cell, 1, axis=0)
    b = np.roll(cell, -1, axis=0) - cell
    return 180 - np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)))


def quality(workdir):
    worst = np.array([corner_angles(c).min() for c in read_deck(workdir)])
    largest = np.array([corner_angles(c).max() for c in read_deck(workdir)])
    return {"min angle": worst.min(), "max angle": largest.max(), "under 20": int((worst < 20).sum())}


def report(out, reference):
    ref = figures(out / name_of(reference))
    rows = []
    for workdir in sorted(p.parent for p in out.glob("*/motor.mag.vtkhdf")):
        mesh = json.loads((workdir / "mesh.json").read_text())
        mesh.update(quality(workdir))
        time = (workdir / "motor.time")
        f = figures(workdir)
        rows.append((workdir.name, mesh, float(time.read_text().split()[0]) if time.exists() else None,
                     {k: abs(f[k] - ref[k]) / abs(ref[k]) for k in ref}, f))
    names = list(ref)
    lines = [f"Reference: {name_of(reference)}", "",
             "| mesh | grids | cells (quads) | angles (cells < 20°) | wall (s) | " + " | ".join(names) + " |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(names)]
    for name, mesh, time, err, _ in rows:
        lines.append(f"| {name} | {mesh['nodes']} | {mesh['cells']} ({mesh['quads']}) | "
                     f"{mesh['min angle']:.0f}–{mesh['max angle']:.0f}° ({mesh['under 20']}) | "
                     f"{'' if time is None else f'{time:.1f}'} | "
                     + " | ".join(f"{100 * err[k]:.3f}%" for k in names) + " |")
    lines += ["", "Reference values: " + ", ".join(f"{k} {ref[k]:.6g}" for k in names)]
    text = "\n".join(lines) + "\n"
    (out / "mesh_study.md").write_text(text)
    print(text)


def plot(workdir, path):
    """The deck's mesh, whole and over one slot pitch at the gap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    cells = read_deck(workdir)
    polys = [c * 1e3 for c in cells]
    quads = np.array([len(c) == 4 for c in cells])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    pitch = 2 * np.pi / rx.motor.SLOTS
    for ax, (lim, lw) in zip(axes, [(None, 0.15), (pitch, 0.4)]):
        ax.add_collection(PolyCollection(polys, facecolors=np.where(quads[:, None], [[0.85, 0.9, 1.0, 1]],
                                                                      [[1.0, 0.85, 0.7, 1]]),
                                         edgecolors="0.2", linewidths=lw))
        ax.set_aspect("equal")
        if lim is None:
            r = rx.motor.R_OUTER * 1e3 * 1.02
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
        else:
            t = np.linspace(-lim / 2, lim / 2, 2)
            r0, r1 = rx.motor.R_YOKE * 1e3 - 1, rx.motor.R_SLOT_BOTTOM * 1e3 + 1
            ax.set_xlim(r0 * np.cos(t[0]) - 0.5, r1 + 0.5)
            ax.set_ylim(r1 * np.sin(t[0]), r1 * np.sin(t[1]))
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    mesh = json.loads((workdir / "mesh.json").read_text())
    axes[0].set_title(f"{workdir.name}: {mesh['nodes']} grids, {mesh['quads']} quadrilaterals (blue), "
                      f"{mesh['cells'] - mesh['quads']} triangles (orange)")
    axes[1].set_title("one slot pitch, rotor yoke to slot bottom")
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("decks")
    d.add_argument("out", type=pathlib.Path)
    d.add_argument("--mesh", nargs="+", required=True, help="kind:scale[/near], kind tri, quad, fq or qm")
    r = sub.add_parser("report")
    r.add_argument("out", type=pathlib.Path)
    r.add_argument("--reference", required=True, help="kind:scale of the reference mesh")
    p = sub.add_parser("plot")
    p.add_argument("workdir", type=pathlib.Path)
    p.add_argument("png", type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "decks":
        write_decks(args.out, args.mesh)
    elif args.command == "plot":
        plot(args.workdir, args.png)
    else:
        report(args.out, args.reference)


if __name__ == "__main__":
    main()
