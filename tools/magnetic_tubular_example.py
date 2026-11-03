#!/usr/bin/env python3
"""A tubular linear motor for SOL MAGSTAT: one pole pitch of the endless stack.

Builds the r-z section with gmsh, writes the MAGAXI deck, runs vibestran and
plots the results from tubular.mag.vtkhdf:

- tubular_field.png: |B| and flux lines at rated current, the pitch tiled
  over three pole pitches and mirrored about the axis;
- tubular_curves.png: thrust at rated current and cogging against the
  mover's position, the no-load back-EMF of the three phases, and the flux
  density at mid-gap;
- tubular_motion.gif: the rated sweep over one electrical period as a
  looping animation, in the magnets' frame, the coils sliding past.

The design: 12 mm diameter, 20 mm long N35 magnets alternating along the
axis ([NS][SN][NS]) in an aluminium tube with a 0.75 mm wall, a 1 mm air
gap, 6.67 mm coils of 88 turns (24 AWG) to a 25 mm OD, 1 mm steel teeth
between the coils and a 1.5 mm steel sleeve around them. The model is one
pole pitch, anti-periodic along the axis (MPERIOD SHIFT), its gap solved by
the tube air-gap element (PAIRGAP), the mover slid by MOTION TRANSLATE and
the phases driven by DRIVE with the pole pitch.

Run from the repository root with the venv (tools/requirements-magnetic.txt
plus gmsh and matplotlib):

    .venv/bin/python tools/magnetic_tubular_example.py OUTPUT_DIR

MAGNETICS.md ("Worked example: a tubular linear motor") describes the
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
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# Dimensions in metres: radii from the axis, the pole pitch along it.
POLE = .020                                    # magnet length, the pole pitch tau
R_MAGNET, WALL, GAP = .006, .00075, .001       # magnet radius, tube wall, air gap
R_COIL = .0125                                 # coils' outside radius (25 mm OD)
TOOTH, SLEEVE = .001, .0015                    # steel tooth width, sleeve thickness
R_OUTER = .030                                 # air to here, held at A = 0
TURNS, PEAK, SPEED = 88, 15.0, 1.0             # turns per coil, A peak (i_q), m/s
BR, MU_MAGNET = 1.2, 1.05                      # N35
PITCHES = 3                                    # the stack the results stand for
RATED, COGGING, PROBES = 96, 40, 81            # positions and mid-gap probes
FRAMES = 48                                    # animation frames: every other rated position
R_TUBE, R_GAP = R_MAGNET + WALL, R_MAGNET + WALL + GAP
Z0 = -POLE / 2                                 # the pitch: Z0 to Z0 + POLE

# PIDs: air, the tube's aluminium (air to the field), steel, the pitch's
# magnet (its anti-periodic copies are the others), the gap's air (the
# air-gap element's).
AIR, TUBE, STEEL, MAGNET, GAP_PID = 1, 2, 3, 4, 9
# Electrical steel, B (T) against H (A/m).
STEEL_BH = [(0, 0), (.5, 60), (1.0, 120), (1.3, 250), (1.5, 800), (1.6, 1800),
            (1.7, 4500), (1.8, 10000), (1.9, 20000), (2.0, 40000), (2.1, 80000)]
# The coils of the pitch in order along +z: the sequence A, -C, B, -A, C, -B
# from z = Z0 - 3 POLE / 2, of which the pitch holds -A, C, -B.
COILS = [("A", -1), ("C", 1), ("B", -1)]
PHASES = "ABC"  # circuits 1, 2, 3


def build_mesh():
    """Triangles of the pitch, each with its role: ("pid", PID) or
    ("coil", k). The sides z = Z0 and Z0 + POLE are meshed alike."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("tubular")
    occ = gmsh.model.occ

    def box(r0, z0, r1, z1):
        return occ.addRectangle(r0, z0, 0, r1 - r0, z1 - z0)

    width = POLE / 3
    regions = [(box(0, Z0, R_OUTER, Z0 + POLE), 1, ("pid", AIR)),
               (box(0, Z0, R_MAGNET, Z0 + POLE), 2, ("pid", MAGNET)),
               (box(R_MAGNET, Z0, R_TUBE, Z0 + POLE), 2, ("pid", TUBE)),
               (box(R_TUBE, Z0, R_GAP, Z0 + POLE), 2, ("pid", GAP_PID)),
               (box(R_GAP, Z0, R_COIL + SLEEVE, Z0 + POLE), 2, ("pid", STEEL))]
    for k in range(3):
        regions.append((box(R_GAP, Z0 + k * width + TOOTH / 2, R_COIL,
                            Z0 + (k + 1) * width - TOOTH / 2), 3, ("coil", k)))
    occ.synchronize()
    tags = [(2, t) for t, _, _ in regions]
    _, children = occ.fragment(tags[:1], tags[1:])
    occ.synchronize()
    best = {}
    for (_, priority, role), kids in zip(regions, children):
        for kid in kids:
            if kid not in best or best[kid][0] < priority:
                best[kid] = (priority, role)
    field = gmsh.model.mesh.field
    f = field.add("MathEval")
    # 0.25 mm at the gap, coarser away from it (x is r).
    field.setString(f, "F", f"Min(2.5e-3, 2.5e-4 + 0.15 * Max(0, Max({R_MAGNET} - x, x - {R_GAP})))")
    field.setAsBackgroundMesh(f)
    for option in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                   "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(option, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # Each curve on the top side is meshed as its image on the bottom, moved
    # up a pitch, so that every grid of one side has its image on the other.
    sides = {}
    for _, c in gmsh.model.getEntities(1):
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        ends = [gmsh.model.getValue(1, c, [t]) for t in (lo[0], hi[0])]
        for side, z in ((0, Z0), (1, Z0 + POLE)):
            if all(abs(p[1] - z) < 1e-12 for p in ends):
                sides[c] = (side, tuple(sorted(round(p[0], 12) for p in ends)))
    shift = [1, 0, 0, 0, 0, 1, 0, POLE, 0, 0, 1, 0, 0, 0, 0, 1]
    for c, (side, span) in sides.items():
        if side == 0:
            image = [d for d, (s, e) in sides.items() if s == 1 and e == span]
            gmsh.model.mesh.setPeriodic(1, image, [c], shift)
    gmsh.model.mesh.generate(2)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = dict(zip(node_tags.astype(int), coords.reshape(-1, 3)[:, :2]))
    elements = []
    for kid, (_, role) in best.items():
        types, _, connectivity = gmsh.model.mesh.getElements(*kid)
        for t, conn in zip(types, connectivity):
            assert t == 2, "triangles only"
            for tri in conn.reshape(-1, 3).astype(int):
                elements.append((role, [int(n) for n in tri]))
    gmsh.finalize()
    return nodes, elements


def write_deck(path, nodes, elements):
    """The MAGAXI deck: grids at (r, 0, z), the movers' cells first."""
    elements.sort(key=lambda e: e[0] not in (("pid", MAGNET), ("pid", TUBE)))
    mover = [eid for eid, (role, _) in enumerate(elements, start=1)
             if role in (("pid", MAGNET), ("pid", TUBE))]
    tooth_period = POLE / 3
    lines = [
        "SOL MAGSTAT", "CEND",
        "TITLE = TUBULAR LINEAR MOTOR, ONE POLE PITCH OF THE ENDLESS STACK",
        "MAGBC = 1",
        "SUBCASE 1", "  LABEL = RATED", "  MOTION = 1", "  DRIVE = 10", "  MSTEP = ALL",
        f"  BFIELD(EVERY={RATED // FRAMES}) = ALL",
        "SUBCASE 2", "  LABEL = NO LOAD", "  MOTION = 1", "  DRIVE = 11", "  MSTEP = ALL",
        f"  BFIELD(EVERY={RATED}) = ALL", "  LDQ = INCREMENTAL",
        "SUBCASE 3", "  LABEL = COGGING", "  MOTION = 2", "  MSTEP = ALL", "  BFIELD = NONE",
        "  MPROBE = NONE",
        "BEGIN BULK",
        "MAGAXI,0",
        "MATMAG,1,1.", f"PMAG2D,{AIR},1", f"PMAG2D,{TUBE},1",
        "MATMAG,2,,1", f"PMAG2D,{STEEL},2",
        f"MATMAG,3,{MU_MAGNET},,{BR}",
        # Magnetized along -z (basic z is the symmetry axis); its
        # anti-periodic neighbours point +z.
        f"PMAG2D,{MAGNET},3,0,-3",
        f"PAIRGAP,{GAP_PID}",
        "MBC,1,0,ZERO",
        f"MPERIOD,1,ANTI,31,32,0,{PITCHES},{POLE:.12g}",
        "TABLEBH,1",
    ]
    flat = [f"{v:g}" for pair in STEEL_BH for v in pair] + ["ENDT"]
    for k in range(0, len(flat), 8):
        lines.append("," + ",".join(flat[k:k + 8]))
    lines += [
        f"ESET,20,{mover[0]},THRU,{mover[-1]}",
        f"MOTION,1,20,0,TRANSLATE,0.,{2 * POLE / RATED:.12g},{RATED}", f",{SPEED}",
        f"MOTION,2,20,0,TRANSLATE,0.,{tooth_period / COGGING:.12g},{COGGING}",
        f"DRIVE,10,SINE,{POLE},{PEAK},0.", ",1,2,3",
        f"DRIVE,11,SINE,{POLE},0.,0.", ",1,2,3",
    ]
    # Probes at mid-gap, fixed in space: the air-gap element gives the field.
    for k in range(PROBES):
        lines.append(f"MPROBE,{k + 1},0,{(R_TUBE + R_GAP) / 2:.15g},0.,"
                     f"{Z0 + POLE * k / (PROBES - 1):.15g}")
    used = sorted({n for _, tri in elements for n in tri})
    for n in used:
        r, z = nodes[n]
        lines.append(f"GRID,{n},,{r:.15g},0.,{z:.15g}")
    coils = {}
    owner = {}  # each element's edges, for the periodic sides
    for eid, (role, tri) in enumerate(elements, start=1):
        pid = AIR if role[0] == "coil" else role[1]
        if role[0] == "coil":
            coils.setdefault(role[1], []).append(eid)
        lines.append(f"CTRIA3,{eid},{pid},{tri[0]},{tri[1]},{tri[2]}")
        if pid != GAP_PID:  # the gap continues across the sides by itself
            for k in range(3):
                owner[frozenset((tri[k], tri[(k + 1) % 3]))] = eid
    # The periodic sides: field faces on z = Z0 (FSET 31) and Z0 + POLE (32).
    for edge, eid in owner.items():
        a, b = sorted(edge)
        for sid, z in ((31, Z0), (32, Z0 + POLE)):
            if abs(nodes[a][1] - z) < 1e-12 and abs(nodes[b][1] - z) < 1e-12:
                lines.append(f"FSET,{sid},{eid},{a},{b}")
    members = {p: [] for p in PHASES}
    for k, eids in sorted(coils.items()):
        phase, sign = COILS[k]
        lines.append(f"ESET,{100 + k}," + ",".join(map(str, eids)))
        lines.append(f"COIL2D,{100 + k},{100 + k},{sign * TURNS}")
        members[phase].append(100 + k)
    for c, phase in enumerate(PHASES, start=1):
        lines.append(f"CIRCUIT,{c},1," + ",".join(f"{i},1" for i in members[phase]))
    lines.append("ENDDATA")
    path.write_text("\n".join(lines) + "\n")
    return elements


def read_results(path):
    with h5py.File(path, "r") as f:
        out = {}
        for key, g in f["MAGSTAT/subcases"].items():
            out[int(key)] = {
                "positions": g["positions_m"][()],
                "force": g["motion/force_N"][()],
                "linkage": g["circuits/linkage_Wb_turn"][()],
                "emf": g["motion/emf_V"][()] if "motion/emf_V" in g else None,
                "probe_b": g["probes/B_T"][()] if "probes/B_T" in g else None,
                "attrs": dict(g.attrs),
                "drive": dict(g["drive"].attrs) if "drive" in g else None,
                "drive_data": {k: g["drive"][k][()] for k in g["drive"]} if "drive" in g else None,
            }
        out["fields"] = read_fields(f["VTKHDF/SUBCASE_1"])
    return out


def read_fields(block):
    """Every time step of a field block: (z, r) points in mm, triangles, |B|
    per cell, r A_phi per point (constant along a flux line), cell element
    IDs and the step's position."""
    steps = block["Steps"]
    points, cells = block["NumberOfPoints"][()], block["NumberOfCells"][()]
    connectivity, offsets = block["Connectivity"][()], block["Offsets"][()]
    out = []
    for k in range(int(steps.attrs["NSteps"])):
        p0, c0 = int(steps["PointOffsets"][k]), int(steps["CellOffsets"][k])
        part0 = int(steps["PartOffsets"][k])
        np_, nc = int(points[part0]), int(cells[part0])
        o0 = int(steps["ConnectivityIdOffsets"][k])
        b0 = int(steps["CellDataOffsets/B_T"][k])
        a0 = int(steps["PointDataOffsets/A_normal_Wb_per_m"][k])
        e0 = int(steps["CellDataOffsets/element_id"][k])
        xyz = block["Points"][p0:p0 + np_]
        tris = connectivity[o0:o0 + 3 * nc].reshape(-1, 3)
        assert np.all(np.diff(offsets[c0 + k:c0 + k + nc + 1]) == 3), "triangles only"
        out.append({
            "points": xyz[:, [2, 0]] * 1e3,
            "triangles": tris,
            "b": np.linalg.norm(block["CellData/B_T"][b0:b0 + nc], axis=1),
            "flux": block["PointData/A_normal_Wb_per_m"][a0:a0 + np_] * xyz[:, 0],
            "element": block["CellData/element_id"][e0:e0 + nc],
            "position": float(steps["Values"][k]),
        })
    return out


def region_of(role):
    """The region a cell's |B| is smoothed over: air (the tube's aluminium
    and the gap too), steel, the magnet, or one coil."""
    if role[0] == "coil":
        return role
    return {AIR: "air", TUBE: "air", GAP_PID: "air", STEEL: "steel"}.get(role[1], "magnet")


def smoothed(field, elements):
    """|B| averaged to each region's own copy of its grids: smooth inside a
    region, sharp between regions."""
    pts, tris = field["points"], field["triangles"]
    p = pts[tris]
    areas = np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                   - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])) / 2
    copies, xy, value, weight, out = {}, [], [], [], []
    for t, e, m, area in zip(tris, field["element"], field["b"], areas):
        key = region_of(elements[e - 1][0])
        corners = []
        for v in t:
            k = copies.setdefault((v, key), len(xy))
            if k == len(xy):
                xy.append(pts[v])
                value.append(0.0)
                weight.append(0.0)
            value[k] += area * m
            weight[k] += area
            corners.append(k)
        out.append(corners)
    return np.array(xy), np.array(out), np.array(value) / np.array(weight)


def outlines(field, elements):
    """Edges between regions; the pitch's sides and the box are left out."""
    owners = {}
    for t, e in zip(field["triangles"], field["element"]):
        for k in range(3):
            owners.setdefault(tuple(sorted((t[k], t[(k + 1) % 3]))), []).append(
                region_of(elements[e - 1][0]))
    pts = field["points"]
    return np.array([[pts[u], pts[v]] for (u, v), o in owners.items()
                     if len(o) == 2 and o[0] != o[1]])


def coil_boxes():
    """Each coil of the pitch: (z0, r0, z1, r1) in mm, its phase and sign."""
    width = POLE / 3
    return [((Z0 + k * width + TOOTH / 2) * 1e3, R_GAP * 1e3,
             (Z0 + (k + 1) * width - TOOTH / 2) * 1e3, R_COIL * 1e3, phase, sign)
            for k, (phase, sign) in enumerate(COILS)]


def draw_field(ax, field, elements, levels, currents, frame_shift=0.0, window=(-30, 30)):
    """The pitch's field tiled into its anti-periodic copies over the
    window, mirrored about the axis. frame_shift (mm) moves the drawing, the
    coils with it: minus the mover's position draws the magnets' frame."""
    xy, tris, smooth = smoothed(field, elements)
    edges = outlines(field, elements)
    halo = [pe.withStroke(linewidth=2.2, foreground="#0d1117")]
    pitch = POLE * 1e3
    shading = None
    for k in range(int(math.floor((window[0] - frame_shift) / pitch)) - 2,
                   int(math.ceil((window[1] - frame_shift) / pitch)) + 2):
        shift, sign = k * pitch + frame_shift, (-1) ** k
        for side in (1, -1):
            s = mtri.Triangulation(xy[:, 0] + shift, side * xy[:, 1], tris)
            shading = ax.tripcolor(s, smooth, shading="gouraud", cmap="inferno", vmin=0,
                                   vmax=2.0, rasterized=True)
            lines = mtri.Triangulation(field["points"][:, 0] + shift,
                                       side * field["points"][:, 1], field["triangles"])
            lines.set_mask(field["b"] < .03)
            ax.tricontour(lines, sign * field["flux"], levels=levels, colors="white",
                          linewidths=.5, alpha=.75, linestyles="solid")
            ax.add_collection(LineCollection(edges * [1, side] + [shift, 0], colors="#c9d1d9",
                                             linewidths=.4, alpha=.55))
            for z0, r0, z1, r1, phase, wound in coil_boxes():
                z0, z1, wound = z0 + shift, z1 + shift, wound * sign
                if z1 < window[0] - 8 or z0 > window[1] + 8:
                    continue
                y0 = r0 if side > 0 else -r1
                ax.add_patch(Rectangle((z0, y0), z1 - z0, r1 - r0, facecolor="#b87333",
                                       alpha=.22, edgecolor="none"))
                ax.add_patch(Rectangle((z0, y0), z1 - z0, r1 - r0, facecolor="none",
                                       edgecolor="#e8a165", lw=.9, alpha=.9))
                # Current along +phi comes out of the page above the axis.
                i = currents[phase]
                mark = "" if abs(i) < .05 * PEAK else "⊙" if wound * i * side > 0 else "⊗"
                ax.text((z0 + z1) / 2, side * (r0 + r1) / 2,
                        f"{phase}{'+' if wound > 0 else '−'}\n{mark}", color="white",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        path_effects=halo, linespacing=1.1, clip_on=True)
    ax.axhline(0, color="#c9d1d9", lw=.4, alpha=.35, ls=(0, (6, 4)))  # the axis
    ax.set_aspect("equal")
    ax.set_xlim(*window)
    ax.set_ylim(-(R_COIL + SLEEVE) * 1e3 - 3, (R_COIL + SLEEVE) * 1e3 + 3)
    ax.axis("off")
    return shading


def dark_figure(width, height, dpi):
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("#0d1117")
    return fig


def colorbar(fig, shading, cax):
    bar = fig.colorbar(shading, cax=cax)
    bar.set_label("|B| (T)", color="white")
    bar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(bar.ax.axes, "yticklabels"), color="white")


def flux_levels(fields):
    """Levels shared by every frame, symmetric about zero, so that the
    anti-periodic copies (the flux negated) continue each other's lines."""
    top = max(np.abs(f["flux"]).max() for f in fields)
    step = 2 * top / 28
    return (np.arange(-14, 14) + .5) * step


def plot_field(results, elements, filename):
    rated, field = results[1], results["fields"][0]
    fig = dark_figure(12, 5, 220)
    ax = fig.add_axes([0, 0, .9, 1])
    shading = draw_field(ax, field, elements, flux_levels([field]),
                         dict(zip(PHASES, rated["phase_currents"][0])), window=(-35, 35))
    bar = fig.colorbar(shading, ax=ax, fraction=.02, pad=.01)
    bar.set_label("|B| (T)", color="white")
    bar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(bar.ax.axes, "yticklabels"), color="white")
    ax.set_title(f"Tubular linear motor at {PEAK:g} A on q, mover at "
                 f"{field['position'] * 1e3:g} mm: |B| and flux lines, SOL MAGSTAT",
                 color="white", fontsize=11)
    fig.savefig(filename, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def plot_animation(results, elements, filename):
    """The rated sweep, one frame per position, drawn in the magnets' frame."""
    from PIL import Image
    rated, fields = results[1], results["fields"]
    levels = flux_levels(fields)
    pngs = []
    # The window is 60 by 34 mm: axes 8.8 in wide and 4.99 in tall.
    for n, field in enumerate(fields):
        k = int(round(field["position"] / (2 * POLE / RATED)))  # the rated step
        fig = dark_figure(10, 6.24, 100)
        ax = fig.add_axes([.01, .1, .88, .8])
        cax = fig.add_axes([.905, .1, .015, .8])
        x = field["position"] * 1e3
        currents = dict(zip(PHASES, rated["phase_currents"][k]))
        shading = draw_field(ax, field, elements, levels, currents, frame_shift=-x)
        colorbar(fig, shading, cax)
        ax.set_title("Tubular linear motor, coils sliding past the magnets: |B| and flux lines",
                     color="white", fontsize=11)
        ax.text(.5, -.04, f"coils at {-x:+5.1f} mm    i_A {currents['A']:+6.2f}  i_B "
                f"{currents['B']:+6.2f}  i_C {currents['C']:+6.2f} A    thrust "
                f"{rated['force'][k]:+6.1f} N", transform=ax.transAxes, ha="center", va="top",
                color="#c9d1d9", fontsize=9, family="monospace")
        png = filename.parent / f"frame{n:03d}.png"
        fig.savefig(png, facecolor=fig.get_facecolor())
        plt.close(fig)
        pngs.append(png)
    images = [Image.open(p).convert("RGB") for p in pngs]
    palette = images[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=70, loop=0,
                   optimize=True, disposal=1)
    for p in pngs:
        p.unlink()


def plot_curves(results, filename):
    rated, noload, cog = results[1], results[2], results[3]
    colors = ["#e4572e", "#29335c", "#f3a712"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=160)
    ax = axes[0, 0]
    ax.plot(rated["positions"] * 1e3, rated["force"], color=colors[1], lw=1.6)
    mean = rated["attrs"]["force_mean_N"]
    ax.axhline(mean, color=colors[0], lw=1, ls="--", label=f"mean {mean:.1f} N")
    ax.set_title(f"Thrust at {PEAK:g} A on q: ripple {rated['attrs']['force_ripple_percent']:.1f} %")
    ax.set_xlabel("mover position (mm)")
    ax.set_ylabel("thrust (N)")
    ax.legend(frameon=False, loc="upper center")
    ax.set_ylim(top=ax.get_ylim()[1] + .25 * np.ptp(ax.get_ylim()))
    ax = axes[0, 1]
    ax.plot(cog["positions"] * 1e3, cog["force"], color=colors[1], lw=1.6, marker="o", ms=2.5)
    ax.set_title(f"Cogging: {cog['attrs']['force_peak_to_peak_N']:.2f} N peak-to-peak")
    ax.set_xlabel("mover position (mm)")
    ax.set_ylabel("force (N)")
    ax = axes[1, 0]
    for k in range(3):
        ax.plot(noload["positions"] * 1e3, noload["emf"][:, k], color=colors[k], lw=1.6,
                label=PHASES[k])
    ax.set_title(f"No-load back-EMF at {SPEED:g} m/s")
    ax.set_xlabel("mover position (mm)")
    ax.set_ylabel("EMF (V)")
    ax.legend(frameon=False, ncol=3, loc="upper center", title="phase", title_fontsize=8,
              fontsize=8)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax = axes[1, 1]
    z = (Z0 + POLE * np.arange(PROBES) / (PROBES - 1)) * 1e3
    for subcase, style, label in ((noload, "--", "no load"), (rated, "-", f"{PEAK:g} A")):
        b = subcase["probe_b"][0]
        ax.plot(z, b[:, 0], color=colors[0], ls=style, lw=1.2, label=f"B_r, {label}")
        ax.plot(z, b[:, 2], color=colors[1], ls=style, lw=1.2, label=f"B_z, {label}")
    ax.set_title("Flux density at mid-gap, mover at 0")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("B (T)")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=7)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    for ax in axes.flat:
        ax.grid(alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tubular linear motor: SOL MAGSTAT with MOTION TRANSLATE and DRIVE",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def checks(results):
    """The relations the results satisfy, printed for MAGNETICS.md."""
    rated, noload, cog = results[1], results[2], results[3]
    d = rated["drive"]
    dq = 1.5 * math.pi / POLE * (d["lambda_d_mean_Wb"] * d["i_q_mean_A"]
                                 - d["lambda_q_mean_Wb"] * d["i_d_mean_A"])
    n = noload["drive"]
    emf = noload["drive_data"]["emf_constant_fundamental_V_s_per_m"]
    print(f"rated thrust {rated['attrs']['force_mean_N']:.7g} N; "
          f"(3/2)(pi/tau)(lambda_d i_q - lambda_q i_d) {dq:.7g} N")
    print(f"no-load back-EMF constants {', '.join(f'{v:.6g}' for v in emf)} V s/m; "
          f"(pi/tau) lambda_pm {math.pi / POLE * n['alignment_linkage_Wb']:.6g}")
    print(f"THETA0 {n['theta0_m'] * 1e3:.6g} mm, lags "
          f"{', '.join(f'{v:.2f}' for v in noload['drive_data']['alignment_lag_deg'][1:])}")
    print(f"cogging: no-load sweep {np.ptp(noload['force']):.4g} N, cogging sweep "
          f"{cog['attrs']['force_peak_to_peak_N']:.4g} N peak-to-peak, mean "
          f"{cog['attrs']['force_mean_N']:.2g} N")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--vibestran", default="build/vibestran")
    parser.add_argument("--reuse", action="store_true",
                        help="plot the results already in OUTPUT without solving")
    parser.add_argument("--no-gif", action="store_true", help="skip the animation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deck = args.output / "tubular.bdf"
    nodes, elements = build_mesh()
    print(f"{len(elements)} triangles, {len({n for _, t in elements for n in t})} grids")
    elements = write_deck(deck, nodes, elements)
    if not args.reuse:
        subprocess.run([str(pathlib.Path(args.vibestran).resolve()), deck.name], check=True,
                       cwd=args.output, stdout=subprocess.DEVNULL)
    results = read_results(args.output / "tubular.mag.vtkhdf")
    rated = results[1]
    with h5py.File(args.output / "tubular.mag.vtkhdf", "r") as f:
        currents = f["MAGSTAT/subcases/1/circuits/current_A"][()]
    rated["phase_currents"] = currents
    checks(results)
    plot_field(results, elements, args.output / "tubular_field.png")
    plot_curves(results, args.output / "tubular_curves.png")
    written = ["tubular_field.png", "tubular_curves.png"]
    if not args.no_gif:
        plot_animation(results, elements, args.output / "tubular_motion.gif")
        written.append("tubular_motion.gif")
    print("wrote", *(args.output / f for f in written))


if __name__ == "__main__":
    sys.exit(main())
