#!/usr/bin/env python3
"""A 12-slot, 14-pole (7 pole pair) surface-magnet motor for SOL MAGSTAT.

Builds the cross section with gmsh, writes the MAGSTAT deck, runs vibestran
and plots the results from name.mag.vtkhdf:

- motor_field.png: |B| and flux lines at load;
- motor_field_detail.png: the same, closer, over a quarter of the motor;
- motor_curves.png: torque at load and cogging against rotor position, the
  no-load back-EMF of the three phases, and the air-gap flux density;

Run from the repository root with the venv (tools/requirements-magnetic.txt
plus gmsh and matplotlib):

    .venv/bin/python tools/magnetic_motor_example.py OUTPUT_DIR

MAGNETICS.md ("Worked example: a 12-slot, 14-pole motor") describes the
subcases and the checks the results satisfy.
"""

import argparse
import math
import pathlib
import subprocess
import sys

import gmsh
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

# Dimensions in metres.
R_SHAFT, R_YOKE, R_MAGNET, R_BORE, R_OUTER = .008, .026, .0295, .0305, .045
R_TIP, R_SLOT_TOP, R_SLOT_BOTTOM = .032, .0335, .038
TOOTH_WIDTH, OPENING = .006, .002
SLOTS, POLE_PAIRS, MAGNET_ARC = 12, 7, .85  # pole arc as a fraction of the pole pitch
DEPTH, TURNS, PEAK = .05, 30, 10.0           # m, turns per coil, A
BR, MU_MAGNET = 1.2, 1.05
SPEED = 1000.0                               # rpm
GAP_PROBES = 720                             # mid-gap probes, every half degree

# PIDs.
AIR, ROTOR_IRON, STATOR_IRON, MAGNET_OUT, MAGNET_IN, ROTOR_AIR, GAP = 1, 2, 3, 4, 5, 6, 9

# Electrical steel, B (T) against H (A/m).
STEEL = [(0, 0), (.5, 60), (1.0, 120), (1.3, 250), (1.5, 800), (1.6, 1800),
         (1.7, 4500), (1.8, 10000), (1.9, 20000), (2.0, 40000), (2.1, 80000)]


def polar(r, angle):
    return r * math.cos(angle), r * math.sin(angle)


def rotate(p, angle):
    c, s = math.cos(angle), math.sin(angle)
    return c * p[0] - s * p[1], s * p[0] + c * p[1]


class Builder:
    """Closed outlines of points, lines and arcs about the origin."""

    def __init__(self):
        self.occ = gmsh.model.occ
        self.center = self.occ.addPoint(0, 0, 0)

    def surface(self, outline):
        """outline: [(x, y, arc_to_next)], arc_to_next true for an arc about
        the origin to the next point."""
        points = [self.occ.addPoint(x, y, 0) for x, y, _ in outline]
        curves = []
        for k, (_, _, arc) in enumerate(outline):
            a, b = points[k], points[(k + 1) % len(points)]
            curves.append(self.occ.addCircleArc(a, self.center, b) if arc
                          else self.occ.addLine(a, b))
        return self.occ.addPlaneSurface([self.occ.addCurveLoop(curves)])


def slot_outlines(sigma):
    """The slot centered at angle sigma: the opening (air) under the tooth
    tips, and the two coil halves, clockwise half first."""
    half = math.pi / SLOTS

    def side(r, sign):
        # The flank of the tooth at sigma + sign * half that faces the slot,
        # at radius r: half a tooth width from the tooth's center line,
        # towards the slot's.
        u = (math.cos(sign * half), math.sin(sign * half))
        n = (math.sin(half), -sign * math.cos(half))
        s = math.sqrt(r * r - (TOOTH_WIDTH / 2) ** 2)
        return rotate((s * u[0] + TOOTH_WIDTH / 2 * n[0], s * u[1] + TOOTH_WIDTH / 2 * n[1]),
                      sigma)

    def at(x, y):
        return rotate((x, y), sigma)

    o = OPENING / 2
    bore = [at(math.sqrt(R_BORE ** 2 - o * o), -o), at(math.sqrt(R_BORE ** 2 - o * o), o)]
    tip = [at(math.sqrt(R_TIP ** 2 - o * o), -o), at(math.sqrt(R_TIP ** 2 - o * o), o)]
    flank = [side(R_SLOT_TOP, -1), side(R_SLOT_TOP, 1)]
    bottom = [side(R_SLOT_BOTTOM, -1), side(R_SLOT_BOTTOM, 1)]
    middle = ((flank[0][0] + flank[1][0]) / 2, (flank[0][1] + flank[1][1]) / 2)
    floor = polar(R_SLOT_BOTTOM, sigma)
    opening = [(*bore[0], False), (*tip[0], False), (*flank[0], False), (*middle, False),
               (*flank[1], False), (*tip[1], False), (*bore[1], True)]
    clockwise = [(*flank[0], False), (*bottom[0], True), (*floor, False), (*middle, False)]
    counter = [(*middle, False), (*floor, True), (*bottom[1], False), (*flank[1], False)]
    return opening, clockwise, counter


def coil_layout():
    """Each tooth coil's phase and polarity, from the star of slots: tooth k's
    linkage has electrical angle p theta_k. Adjacent teeth are 30 electrical
    degrees apart, so each phase takes two pairs whose phasors lie 15 degrees
    either side of its axis, at 15 + 120 k for phase k (0, 1, 2 for A, B, C,
    each lagging the last by 120 electrical degrees)."""
    out = []
    for tooth in range(SLOTS):
        angle = POLE_PAIRS * 360.0 * tooth / SLOTS

        def distance(choice):
            phase, sign = choice
            axis = 15 + 120 * phase + (0 if sign > 0 else 180)
            return abs((angle - axis + 180) % 360 - 180)
        out.append(min(((phase, sign) for phase in range(3) for sign in (1, -1)),
                       key=distance))
    return out


def build_mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("motor")
    occ = gmsh.model.occ
    b = Builder()
    roles = []  # (dimtag, priority, role)
    for radius, role, priority in [(R_SHAFT, ("pid", ROTOR_AIR), 5),
                                   (R_YOKE, ("pid", ROTOR_IRON), 4),
                                   (R_MAGNET, ("pid", ROTOR_AIR), 3),
                                   (R_BORE, ("pid", GAP), 2),
                                   (R_OUTER, ("pid", STATOR_IRON), 1)]:
        roles.append(((2, occ.addDisk(0, 0, 0, radius, radius)), priority, role))
    pitch = math.pi / POLE_PAIRS
    for j in range(2 * POLE_PAIRS):
        a0, a1 = (j - MAGNET_ARC / 2) * pitch, (j + MAGNET_ARC / 2) * pitch
        tag = b.surface([(*polar(R_YOKE, a0), False), (*polar(R_MAGNET, a0), True),
                         (*polar(R_MAGNET, a1), False), (*polar(R_YOKE, a1), True)])
        roles.append(((2, tag), 6, ("pid", MAGNET_OUT if j % 2 == 0 else MAGNET_IN)))
    for slot in range(SLOTS):
        sigma = 2 * math.pi * (slot + .5) / SLOTS
        opening, clockwise, counter = slot_outlines(sigma)
        roles.append(((2, b.surface(opening)), 7, ("pid", AIR)))
        # The clockwise half belongs to the tooth at slot, the other half to
        # the tooth after it; a tooth coil's counterclockwise side is in the
        # slot after the tooth.
        roles.append(((2, b.surface(clockwise)), 8, ("coil", slot, +1)))
        roles.append(((2, b.surface(counter)), 8, ("coil", (slot + 1) % SLOTS, -1)))
    occ.synchronize()
    entities = [r[0] for r in roles]
    _, children = occ.fragment(entities[:1], entities[1:])
    occ.synchronize()
    best = {}
    for (dimtag, priority, role), kids in zip(roles, children):
        for kid in kids:
            if kid not in best or best[kid][0] < priority:
                best[kid] = (priority, role)
    field = gmsh.model.mesh.field
    size = field.add("MathEval")
    field.setString(size, "F", f"Min(1.4e-3, 3.2e-4 + 0.12 * Fabs(Sqrt(x*x + y*y) - {R_BORE}))")
    field.setAsBackgroundMesh(size)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = dict(zip(node_tags.astype(int), coords.reshape(-1, 3)[:, :2]))
    elements = []  # (pid or coil key, [node ids])
    for kid, (_, role) in best.items():
        types, _, connectivity = gmsh.model.mesh.getElements(*kid)
        for t, conn in zip(types, connectivity):
            assert t == 2, "triangles only"
            for tri in conn.reshape(-1, 3).astype(int):
                elements.append((role, list(tri)))
    gmsh.finalize()
    return nodes, elements


def write_deck(path, nodes, elements):
    layout = coil_layout()
    pitch = 360.0 / POLE_PAIRS
    rated = 120          # positions over one electrical period
    cogging = 40         # positions over one cogging period
    cogging_period = 360.0 / math.lcm(SLOTS, 2 * POLE_PAIRS)
    lines = [
        "SOL MAGSTAT", "CEND",
        "TITLE = 12-SLOT 14-POLE SURFACE-MAGNET MOTOR",
        "SUBCASE 1", "  LABEL = RATED", "  MOTION = 1", "  DRIVE = 10", "  MSTEP = ALL",
        f"  BFIELD(EVERY={rated // 4}) = ALL",
        "SUBCASE 2", "  LABEL = NO LOAD", "  MOTION = 1", "  DRIVE = 11", "  MSTEP = ALL",
        f"  BFIELD(EVERY={rated}) = ALL", "  LDQ = INCREMENTAL",
        "SUBCASE 3", "  LABEL = COGGING", "  MOTION = 2", "  MSTEP = ALL", "  BFIELD = NONE",
        "  MPROBE = NONE",
        "BEGIN BULK",
        f"MAG2D,0,{DEPTH}",
        "CORD2C,10,,0.,0.,0.,0.,0.,1.", ",1.,0.,0.",
        "MATMAG,1,1.", f"PMAG2D,{AIR},1", f"PMAG2D,{ROTOR_AIR},1",
        "MATMAG,2,,1", f"PMAG2D,{ROTOR_IRON},2", f"PMAG2D,{STATOR_IRON},2",
        f"MATMAG,3,{MU_MAGNET},,{BR}",
        f"PMAG2D,{MAGNET_OUT},3,10,1", f"PMAG2D,{MAGNET_IN},3,10,-1",
        f"PAIRGAP,{GAP}",
        "TABLEBH,1",
    ]
    flat = [f"{v:g}" for pair in STEEL for v in pair] + ["ENDT"]
    for k in range(0, len(flat), 8):
        lines.append("," + ",".join(flat[k:k + 8]))
    lines += [
        f"ESET,20,PROP,{ROTOR_IRON},{MAGNET_OUT},{MAGNET_IN},{ROTOR_AIR}",
        f"MOTION,1,20,0,ROTATE,0.,{pitch / rated:.12g},{rated}", f",{SPEED}",
        f"MOTION,2,20,0,ROTATE,0.,{cogging_period / cogging:.12g},{cogging}",
        f"DRIVE,10,SINE,{POLE_PAIRS},{PEAK},0.", ",1,2,3",
        f"DRIVE,11,SINE,{POLE_PAIRS},0.,0.", ",1,2,3",
    ]
    # Probes at mid-gap, fixed in space: the AGE gives the field there.
    middle = (R_MAGNET + R_BORE) / 2
    for k in range(GAP_PROBES):
        x, y = polar(middle, 2 * math.pi * k / GAP_PROBES)
        lines.append(f"MPROBE,{k + 1},0,{x:.15g},{y:.15g},0.")
    used = sorted({n for _, tri in elements for n in tri})
    for n in used:
        x, y = nodes[n]
        lines.append(f"GRID,{n},,{x:.15g},{y:.15g},0.")
    coils = {}
    for eid, (role, tri) in enumerate(elements, start=1):
        if role[0] == "coil":
            coils.setdefault((role[1], role[2]), []).append(eid)
            pid = AIR
        else:
            pid = role[1]
        lines.append(f"CTRIA3,{eid},{pid},{tri[0]},{tri[1]},{tri[2]}")
    members = {0: [], 1: [], 2: []}
    for index, ((tooth, side), eids) in enumerate(sorted(coils.items()), start=1):
        phase, polarity = layout[tooth]
        lines.append(f"ESET,{100 + index}," + ",".join(map(str, eids)))
        # Positive coil current runs along the plane normal on the tooth's
        # counterclockwise side, making a field outward along the tooth.
        lines.append(f"COIL2D,{100 + index},{100 + index},{side * polarity * TURNS}")
        members[phase].append(100 + index)
    for phase, ids in members.items():
        lines.append(f"CIRCUIT,{phase + 1},1," + ",".join(f"{i},1" for i in ids))
    lines.append("ENDDATA")
    path.write_text("\n".join(lines) + "\n")
    return layout


def read_results(path):
    with h5py.File(path, "r") as f:
        subcases = f["MAGSTAT/subcases"]
        out = {}
        for key in subcases:
            g = subcases[key]
            out[int(key)] = {
                "positions": g["positions_deg"][()],
                "torque": g["motion/torque_Nm"][()],
                "linkage": g["circuits/linkage_Wb_turn"][()],
                "emf": g["motion/emf_V"][()] if "motion/emf_V" in g else None,
                "probe_b": g["probes/B_T"][()] if "probes/B_T" in g else None,
                "drive": {k: g["drive"][k][()] for k in g["drive"]} if "drive" in g else None,
                "attrs": dict(g.attrs),
                "drive_attrs": dict(g["drive"].attrs) if "drive" in g else None,
            }
        out["fields"] = {k: read_field(f[f"VTKHDF/SUBCASE_{k}"]) for k in (1, 2)}
    return out


def read_field(block):
    """The first time step of a field block: points, triangles, B per cell
    and A per point."""
    steps = block["Steps"]
    points = int(block["NumberOfPoints"][0])
    cells = int(block["NumberOfCells"][0])
    p0 = int(steps["PointOffsets"][0])
    c0 = int(steps["CellDataOffsets/B_T"][0])
    a0 = int(steps["PointDataOffsets/A_normal_Wb_per_m"][0])
    connectivity = block["Connectivity"][()]
    offsets = block["Offsets"][()]
    types = block["Types"][()]
    field = {
        "points": block["Points"][p0:p0 + points, :2],
        "triangles": np.array([connectivity[offsets[k]:offsets[k + 1]]
                               for k in range(cells) if types[k] == 5]),
        "triangle_cells": np.array([k for k in range(cells) if types[k] == 5]),
        "b": block["CellData/B_T"][c0:c0 + cells, :2],
        "a": block["PointData/A_normal_Wb_per_m"][a0:a0 + points],
        "element": block["CellData/element_id"][()],
        "time": float(steps["Values"][0]),
    }
    return field


def outline_segments(nodes, elements):
    """Edges between regions (iron, magnets, air, each coil side) and on the
    outside, for drawing outlines."""
    owners = {}
    for role, tri in elements:
        region = region_of(role)
        for k in range(3):
            edge = tuple(sorted((tri[k], tri[(k + 1) % 3])))
            owners.setdefault(edge, []).append(region)
    return np.array([[nodes[a], nodes[b]] for (a, b), owner in owners.items()
                     if len(owner) == 1 or owner[0] != owner[1]])


def region_of(role):
    regions = {ROTOR_AIR: "air", GAP: "air", AIR: "air", MAGNET_IN: "magnet",
               MAGNET_OUT: "magnet"}
    return role if role[0] == "coil" else regions.get(role[1], role[1])


def smooth_field(field, nodes, elements):
    """A triangulation with each region's own copy of its grids, so |B|
    averaged to the grids is smooth within a region and sharp between them.
    Element k of the deck is elements[k - 1]."""
    points = field["points"]
    triangles = field["triangles"]
    cells = field["triangle_cells"]
    magnitude = np.linalg.norm(field["b"][cells], axis=1)
    copies, xy, value, weight, out = {}, [], [], [], []
    for t, c in zip(triangles, cells):
        key = region_of(elements[field["element"][c] - 1][0])
        corners = []
        for n in t:
            index = copies.setdefault((n, key), len(xy))
            if index == len(xy):
                xy.append(points[n])
                value.append(0.0)
                weight.append(0.0)
            corners.append(index)
        a, b, d = (points[n] for n in t)
        area = abs((b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])) / 2
        for index in corners:
            value[index] += area * magnitude[len(out)]
            weight[index] += area
        out.append(corners)
    xy = np.array(xy)
    return (mtri.Triangulation(xy[:, 0] * 1e3, xy[:, 1] * 1e3, np.array(out)),
            np.array(value) / np.array(weight))


def plot_field(field, nodes, elements, filename, window=None, title=""):
    from matplotlib.collections import LineCollection
    smooth, magnitude = smooth_field(field, nodes, elements)
    lines = mtri.Triangulation(field["points"][:, 0] * 1e3, field["points"][:, 1] * 1e3,
                               field["triangles"])
    # No flux lines where there is next to no flux (the rotor's core), where
    # A is flat and a level would only trace round-off.
    lines.set_mask(np.linalg.norm(field["b"][field["triangle_cells"]], axis=1) < .03)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=220)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    shading = ax.tripcolor(smooth, magnitude, shading="gouraud", cmap="inferno", vmin=0,
                           vmax=2.0, rasterized=True)
    # Flux lines: levels offset half a step from A = 0 (the outer boundary,
    # and the rotor's flux-free core) so no line wanders where A is flat.
    a = field["a"]
    step = (a.max() - a.min()) / 34
    levels = np.arange(np.floor(a.min() / step) + .5, np.ceil(a.max() / step), 1) * step
    ax.tricontour(lines, a, levels=levels, colors="white", linewidths=.5, alpha=.75,
                  linestyles="solid")
    ax.add_collection(LineCollection(outline_segments(nodes, elements) * 1e3, colors="#c9d1d9",
                                     linewidths=.4, alpha=.55))
    ax.set_aspect("equal")
    window = window or (-46, 46, -46, 46)
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.axis("off")
    bar = fig.colorbar(shading, ax=ax, fraction=.035, pad=.01)
    bar.set_label("|B| (T)", color="white")
    bar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(bar.ax.axes, "yticklabels"), color="white")
    ax.set_title(title, color="white", fontsize=11)
    fig.savefig(filename, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_curves(results, filename):
    rated, noload, cog = results[1], results[2], results[3]
    colors = ["#e4572e", "#29335c", "#f3a712"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=160)
    ax = axes[0, 0]
    ax.plot(rated["positions"], rated["torque"], color=colors[1], lw=1.6)
    mean = rated["attrs"]["torque_mean_Nm"]
    ax.axhline(mean, color=colors[0], lw=1, ls="--", label=f"mean {mean:.2f} N·m")
    ax.set_title(f"Torque at {PEAK:g} A, γ = 0: ripple "
                 f"{rated['attrs']['torque_ripple_percent']:.1f} %")
    ax.set_xlabel("rotor position (°)")
    ax.set_ylabel("torque (N·m)")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[0, 1]
    ax.plot(cog["positions"], cog["torque"] * 1e3, color=colors[1], lw=1.6, marker="o", ms=2.5)
    ax.set_title(f"Cogging: {cog['attrs']['torque_peak_to_peak_Nm'] * 1e3:.0f} mN·m "
                 "peak-to-peak")
    ax.set_xlabel("rotor position (°)")
    ax.set_ylabel("torque (mN·m)")
    ax = axes[1, 0]
    for k in range(3):
        ax.plot(noload["positions"], noload["emf"][:, k], color=colors[k], lw=1.6,
                label="ABC"[k])
    ax.set_title(f"No-load back-EMF at {SPEED:g} rpm")
    ax.set_xlabel("rotor position (°)")
    ax.set_ylabel("EMF (V)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0),
              title="phase", title_fontsize=8, fontsize=8)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    ax = axes[1, 1]
    angle = np.arange(GAP_PROBES) * 360 / GAP_PROBES
    radial = np.column_stack([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    for subcase, color, label in ((noload, colors[1], "no load"),
                                  (rated, colors[0], f"{PEAK:g} A")):
        b = subcase["probe_b"][0, :, :2]
        ax.plot(angle, np.sum(b * radial, axis=1), color=color, lw=1.1, label=label)
    ax.set_xlim(0, 360 / POLE_PAIRS * 2)
    ax.set_title("Radial flux density at mid-gap, rotor at 0°")
    ax.set_xlabel("angle (°)")
    ax.set_ylabel("B_r (T)")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    for ax in axes.flat:
        ax.grid(alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("12-slot, 14-pole surface-magnet motor: SOL MAGSTAT with MOTION and DRIVE",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--vibestran", default="build/vibestran")
    parser.add_argument("--reuse", action="store_true",
                        help="plot the results already in OUTPUT without solving")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deck = args.output / "motor.bdf"
    nodes, elements = build_mesh()
    print(f"{len(elements)} triangles, {len({n for _, t in elements for n in t})} grids")
    write_deck(deck, nodes, elements)
    if not args.reuse:
        subprocess.run([args.vibestran, str(deck)], check=True, cwd=args.output)
    results = read_results(args.output / "motor.mag.vtkhdf")
    field = results["fields"][1]
    plot_field(field, nodes, elements, args.output / "motor_field.png",
               title=f"12-slot, 14-pole surface-magnet motor at {PEAK:g} A, rotor at "
                     f"{field['time']:g}°\n|B| and flux lines, SOL MAGSTAT")
    plot_field(field, nodes, elements, args.output / "motor_field_detail.png",
               window=(4, 46, -4, 38),
               title=f"Teeth, slots and magnets at {PEAK:g} A: |B| and flux lines")
    plot_curves(results, args.output / "motor_curves.png")
    print("wrote", *(args.output / f for f in ("motor_field.png", "motor_field_detail.png",
                                                 "motor_curves.png")))


if __name__ == "__main__":
    sys.exit(main())
