#!/usr/bin/env python3
"""A finite tubular linear motor for SOL MAGSTAT: one set of three coils
sliding along a rod of magnets.

The motor:

- the rod: five 12 mm diameter, 20 mm long N35 magnets in a 0.75 mm
  aluminium tube, magnetized along the axis and alternating ([NS][SN]...);
- a 1 mm air gap;
- the stator: one set of three coils (phases A, -C, B), each 6.67 mm long
  with 88 turns, out to a 25 mm OD. With steel (the default) there is a 1 mm
  steel tooth on either side of every coil and a 1.5 mm steel shell around
  them; --no-steel leaves the coils in air.

How it is modelled. vibestran slides a part along a gap only in a model
that repeats along that line (MPERIOD with SHIFT), so the motor is placed
in a 200 mm long slice of space that repeats along the axis. The motor is
far shorter than that, so each copy hardly feels its neighbours: the result
is one finite motor with its rod ends and its end coils. The air gap is
the tube air-gap element (PAIRGAP), which runs the full 200 mm. Everything
inside the gap (magnets, tube, and the air beyond the rod's ends) is the
moving part, and MOTION TRANSLATE slides it. DRIVE sets the phase currents
from the position, with the pole pitch (20 mm) in its POLES field.

Two subcases sweep the rod over one electrical period (two pole pitches,
40 mm, centred on the stator): no load (back-EMF, and the force the steel
and the rod's ends pull with) and 15 A peak on the q axis (thrust).

The script writes tubular_finite.bdf, runs vibestran, and draws:

- field_<subcase>_<position>.png: |B| and flux lines at the mover positions
  -20 mm and 0, the section through the axis;
- curves.png: thrust and no-load force, and the no-load back-EMF.

Run from the repository root with the venv (tools/requirements-magnetic.txt
plus gmsh and matplotlib):

    .venv/bin/python tools/magnetic_tubular_finite_example.py OUTPUT_DIR [--no-steel]

MAGNETICS.md ("Worked example: a finite tubular motor") describes it.
"""

import argparse
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
from matplotlib.collections import LineCollection  # noqa: E402

# ---------------------------------------------------------------------------
# The design. Lengths are in metres; r is the distance from the axis and z
# the position along it.
# ---------------------------------------------------------------------------
POLE = .020           # magnet length: one pole pitch
MAGNETS = 5           # magnets on the rod
R_MAGNET = .006       # magnet radius (12 mm diameter)
WALL = .00075         # aluminium tube wall
GAP = .001            # air gap
R_COIL = .0125        # outside radius of the coils (25 mm OD)
TOOTH = .001          # steel tooth width, and the space between coils
SHELL = .0015         # steel shell thickness
TURNS = 88            # turns per coil (24 AWG)
PEAK = 15.0           # rated current, A peak, all on the q axis
SPEED = 1.0           # m/s, to turn the back-EMF into volts
BR, MU_MAGNET = 1.2, 1.05   # N35 remanence (T) and relative permeability

PERIOD = .200         # the slice of space that repeats along the axis
R_OUTER = .050        # the air around the motor, held at A = 0 out here
POSITIONS = 48        # mover positions over one electrical period

# Derived radii: the tube's outside, and the gap's outside (the coils' inside).
R_TUBE = R_MAGNET + WALL
R_GAP = R_TUBE + GAP
COIL_PITCH = POLE / 3  # each coil and the space beside it

# Property IDs. The tube's aluminium and the air inside the gap behave as
# air, but get their own IDs so the moving part can be named by property.
AIR, ROD_AIR, TUBE, STEEL, MAGNET_UP, MAGNET_DOWN, GAP_ID = 1, 2, 3, 4, 5, 6, 9
MOVING = (ROD_AIR, TUBE, MAGNET_UP, MAGNET_DOWN)

# The three coils from -z to +z: their phase, and +1 or -1 for the way they
# are wound. Phase A, then C wound backwards, then B: the start of the
# usual A, -C, B, -A, C, -B sequence of a three-phase linear winding.
COILS = [("A", 1), ("C", -1), ("B", 1)]

# A simple electrical steel, B (T) against H (A/m).
STEEL_BH = [(0, 0), (.5, 60), (1.0, 120), (1.3, 250), (1.5, 800), (1.6, 1800),
            (1.7, 4500), (1.8, 10000), (1.9, 20000), (2.0, 40000), (2.1, 80000)]


def make_mesh(steel):
    """Mesh the r-z section with gmsh: a list of triangles, each with its
    property ID (or ("coil", k) for coil k), and the grid coordinates.

    The section is drawn as overlapping rectangles. Where rectangles
    overlap, the one with the higher priority decides what the area is (a
    coil wins over the air around it, for example)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    occ = gmsh.model.occ

    def rectangle(r0, z0, r1, z1):
        return occ.addRectangle(r0, z0, 0, r1 - r0, z1 - z0)

    z_bottom, z_top = -PERIOD / 2, PERIOD / 2
    rod = MAGNETS * POLE / 2          # the rod runs from -rod to +rod
    stator = 1.5 * COIL_PITCH         # the coils run from -stator to +stator
    shapes = [  # (rectangle, priority, what it is)
        (rectangle(0, z_bottom, R_OUTER, z_top), 1, AIR),
        (rectangle(0, z_bottom, R_TUBE, z_top), 2, ROD_AIR),
        (rectangle(R_TUBE, z_bottom, R_GAP, z_top), 3, GAP_ID),
        (rectangle(R_MAGNET, -rod, R_TUBE, rod), 3, TUBE),
    ]
    for j in range(MAGNETS):
        z0 = -rod + j * POLE
        shapes.append((rectangle(0, z0, R_MAGNET, z0 + POLE), 4,
                       MAGNET_UP if j % 2 == 0 else MAGNET_DOWN))
    if steel:
        # The shell and the teeth: one block of steel from the gap out to
        # the shell's outside, which the coils then cut their windows from.
        shapes.append((rectangle(R_GAP, -stator - TOOTH / 2, R_COIL + SHELL,
                                 stator + TOOTH / 2), 3, STEEL))
    for k in range(3):
        z0 = -stator + k * COIL_PITCH + TOOTH / 2
        shapes.append((rectangle(R_GAP, z0, R_COIL, z0 + COIL_PITCH - TOOTH), 4, ("coil", k)))

    # Cut the rectangles into non-overlapping pieces, and give each piece
    # the role of the highest-priority rectangle it came from.
    occ.synchronize()
    tags = [(2, tag) for tag, _, _ in shapes]
    _, pieces = occ.fragment(tags[:1], tags[1:])
    occ.synchronize()
    role_of = {}
    for (_, priority, role), parts in zip(shapes, pieces):
        for part in parts:
            if part not in role_of or role_of[part][0] < priority:
                role_of[part] = (priority, role)

    # Element size: 0.25 mm in and around the gap, growing away from it.
    field = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(
        field, "F", f"Min(5e-3, 2.5e-4 + 0.15 * Max(0, Max({R_MAGNET} - x, x - {R_GAP})))")
    gmsh.model.mesh.field.setAsBackgroundMesh(field)
    for option in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                   "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(option, 0)

    # The bottom and top edges of the slice must have matching grids, so
    # that the solver can join them: mesh each top edge as a copy of the
    # bottom edge below it, moved up one period.
    bottom, top = {}, {}
    for _, curve in gmsh.model.getEntities(1):
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, curve)
        span = (round(x0, 9), round(x1, 9))
        if abs(y0 - z_bottom) < 1e-9 and abs(y1 - z_bottom) < 1e-9:
            bottom[span] = curve
        elif abs(y0 - z_top) < 1e-9 and abs(y1 - z_top) < 1e-9:
            top[span] = curve
    move_up = [1, 0, 0, 0, 0, 1, 0, PERIOD, 0, 0, 1, 0, 0, 0, 0, 1]
    for span, curve in bottom.items():
        gmsh.model.mesh.setPeriodic(1, [top[span]], [curve], move_up)

    gmsh.model.mesh.generate(2)
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    grids = dict(zip(node_tags.astype(int), coordinates.reshape(-1, 3)[:, :2]))
    triangles = []
    for part, (_, role) in role_of.items():
        _, _, nodes = gmsh.model.mesh.getElements(*part)
        for corners in nodes[0].reshape(-1, 3).astype(int):
            triangles.append((role, [int(n) for n in corners]))
    gmsh.finalize()
    # The moving part's elements first, so the deck can name them as a range.
    triangles.sort(key=lambda t: t[0] not in MOVING)
    return grids, triangles


def write_deck(path, grids, triangles):
    """Write the MAGSTAT deck. Element k of the deck is triangles[k - 1]."""
    moving = sum(1 for role, _ in triangles if role in MOVING)
    step = 2 * POLE / POSITIONS
    deck = [
        "SOL MAGSTAT",
        "CEND",
        "TITLE = FINITE TUBULAR LINEAR MOTOR",
        "MAGBC = 1",
        "MSTEP = ALL",
        f"BFIELD(EVERY={POSITIONS // 2}) = ALL",   # the field at -20 mm and at 0
        "SUBCASE 1",
        "  LABEL = NO LOAD",
        "  MOTION = 1",
        "  DRIVE = 11",
        "SUBCASE 2",
        "  LABEL = RATED",
        "  MOTION = 1",
        "  DRIVE = 10",
        "BEGIN BULK",
        "MAGAXI,0",                                 # axisymmetric: x is r, z is z
        # Materials: air, steel with its B-H table, and the magnets.
        "MATMAG,1,1.",
        "MATMAG,2,,1",
        f"MATMAG,3,{MU_MAGNET},,{BR}",
        f"PMAG2D,{AIR},1", f"PMAG2D,{ROD_AIR},1", f"PMAG2D,{TUBE},1", f"PMAG2D,{STEEL},2",
        # Magnets point along +z or -z (the z axis is the symmetry axis).
        f"PMAG2D,{MAGNET_UP},3,0,3",
        f"PMAG2D,{MAGNET_DOWN},3,0,-3",
        # The air gap, solved exactly by the tube air-gap element. (BAND
        # instead of AGE solves it about three times faster, a little less
        # exactly.)
        f"PAIRGAP,{GAP_ID},AGE",
        # A = 0 on the outside of the air.
        "MBC,1,0,ZERO",
        # The slice repeats every PERIOD along the axis (FSET 31 is its
        # bottom edge, FSET 32 its top). One copy: results are one motor's.
        f"MPERIOD,1,PERIODIC,31,32,0,1,{PERIOD}",
        # The moving part, and the sweep: from -20 mm, one electrical period.
        f"ESET,20,1,THRU,{moving}",
        f"MOTION,1,20,0,TRANSLATE,{-POLE},{step:.12g},{POSITIONS}",
        f",{SPEED}",
        # Phase currents from the position: pole pitch POLE, PEAK amps on
        # the q axis (DRIVE 10) or none (DRIVE 11), the magnets' d axis found
        # from the no-load flux (THETA0 blank: AUTO).
        f"DRIVE,10,SINE,{POLE},{PEAK},0.",
        ",1,2,3",
        f"DRIVE,11,SINE,{POLE},0.,0.",
        ",1,2,3",
        "TABLEBH,1",
    ]
    numbers = [f"{value:g}" for point in STEEL_BH for value in point] + ["ENDT"]
    for k in range(0, len(numbers), 8):
        deck.append("," + ",".join(numbers[k:k + 8]))

    # Grids at (r, 0, z), and the triangles.
    for n in sorted({n for _, corners in triangles for n in corners}):
        r, z = grids[n]
        deck.append(f"GRID,{n},,{r:.15g},0.,{z:.15g}")
    for k, (role, corners) in enumerate(triangles, start=1):
        pid = AIR if isinstance(role, tuple) else role   # a coil is air too
        deck.append(f"CTRIA3,{k},{pid},{corners[0]},{corners[1]},{corners[2]}")

    # The bottom and top edges of the slice, as faces of the elements on
    # them. The gap's own elements are left out: the air-gap element joins
    # the gap across the slice's edges by itself.
    for k, (role, corners) in enumerate(triangles, start=1):
        if role == GAP_ID:
            continue
        for a, b in zip(corners, corners[1:] + corners[:1]):
            for fset, z in ((31, -PERIOD / 2), (32, PERIOD / 2)):
                if abs(grids[a][1] - z) < 1e-12 and abs(grids[b][1] - z) < 1e-12:
                    deck.append(f"FSET,{fset},{k},{a},{b}")

    # The coils: each coil's elements, its turns and winding direction, and
    # the three phase circuits (circuit 1 is A, 2 is B, 3 is C).
    phases = {"A": [], "B": [], "C": []}
    for k, (phase, sign) in enumerate(COILS):
        elements = [str(e) for e, (role, _) in enumerate(triangles, start=1) if role == ("coil", k)]
        deck.append(f"ESET,{100 + k}," + ",".join(elements))
        deck.append(f"COIL2D,{100 + k},{100 + k},{sign * TURNS}")
        phases[phase].append(100 + k)
    for circuit, phase in enumerate("ABC", start=1):
        deck.append(f"CIRCUIT,{circuit},1," + ",".join(f"{c},1" for c in phases[phase]))
    deck.append("ENDDATA")
    path.write_text("\n".join(deck) + "\n")


def read_field(block, step):
    """One saved field of a subcase: the grids as (z, r) in mm, the
    triangles, |B| on each triangle, r A (whose contours are flux lines),
    each triangle's element ID and the mover's position."""
    steps = block["Steps"]
    part = int(steps["PartOffsets"][step])
    n_points = int(block["NumberOfPoints"][part])
    n_cells = int(block["NumberOfCells"][part])
    p0 = int(steps["PointOffsets"][step])
    c0 = int(steps["ConnectivityIdOffsets"][step])
    b0 = int(steps["CellDataOffsets/B_T"][step])
    a0 = int(steps["PointDataOffsets/A_normal_Wb_per_m"][step])
    e0 = int(steps["CellDataOffsets/element_id"][step])
    points = block["Points"][p0:p0 + n_points]            # (r, 0, z)
    return {
        "zr": points[:, [2, 0]] * 1e3,
        "triangles": block["Connectivity"][c0:c0 + 3 * n_cells].reshape(-1, 3),
        "b": np.linalg.norm(block["CellData/B_T"][b0:b0 + n_cells], axis=1),
        "flux": block["PointData/A_normal_Wb_per_m"][a0:a0 + n_points] * points[:, 0],
        "element": block["CellData/element_id"][e0:e0 + n_cells],
        "position": float(steps["Values"][step]),
    }


def plot_field(field, triangles, title, path):
    """|B| and flux lines on the section through the axis: the r-z half
    plane and its mirror image below the axis."""
    zr, tri = field["zr"], field["triangles"]
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=160)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    # Flux lines are contours of r A, at levels shared by both halves.
    top = np.abs(field["flux"]).max()
    levels = np.linspace(-top, top, 41)[1:-1]
    # The outlines of the parts: triangle edges whose two sides differ.
    owner = {}
    for t, e in zip(tri, field["element"]):
        for a, b in zip(t, np.roll(t, -1)):
            owner.setdefault(tuple(sorted((a, b))), []).append(triangles[e - 1][0])
    edges = np.array([[zr[a], zr[b]] for (a, b), roles in owner.items()
                      if len(roles) == 2 and roles[0] != roles[1]])
    for side in (1, -1):   # above the axis, then its mirror image
        mesh = mtri.Triangulation(zr[:, 0], side * zr[:, 1], tri)
        shading = ax.tripcolor(mesh, facecolors=field["b"], cmap="inferno", vmin=0, vmax=2.0,
                               rasterized=True)
        ax.tricontour(mesh, field["flux"], levels=levels, colors="white", linewidths=.45,
                      alpha=.7, linestyles="solid")
        ax.add_collection(LineCollection(edges * [1, side], colors="#c9d1d9", linewidths=.4,
                                         alpha=.6))
    ax.axhline(0, color="#c9d1d9", lw=.4, alpha=.35, ls=(0, (6, 4)))   # the axis
    ax.set_aspect("equal")
    ax.set_xlim(-PERIOD / 2 * 1e3, PERIOD / 2 * 1e3)
    ax.set_ylim(-R_OUTER * 1e3, R_OUTER * 1e3)
    ax.set_xlabel("z (mm)", color="white")
    ax.set_ylabel("r (mm)", color="white")
    ax.tick_params(colors="white")
    bar = fig.colorbar(shading, ax=ax, fraction=.02, pad=.01)
    bar.set_label("|B| (T)", color="white")
    bar.ax.tick_params(colors="white")
    ax.set_title(title, color="white")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_curves(subcases, path):
    """Force against position for both subcases, and the no-load back-EMF."""
    no_load, rated = subcases[1], subcases[2]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    left.plot(rated["position"], rated["force"], label=f"{PEAK:g} A on q", color="#29335c")
    left.plot(no_load["position"], no_load["force"], label="no load", color="#e4572e")
    left.set_xlabel("mover position (mm)")
    left.set_ylabel("force on the rod (N)")
    left.set_title(f"Thrust: mean {rated['mean']:.1f} N at {PEAK:g} A")
    left.legend(frameon=False)
    for k, (phase, color) in enumerate(zip("ABC", ["#e4572e", "#29335c", "#f3a712"])):
        right.plot(no_load["position"], no_load["emf"][:, k], label=phase, color=color)
    right.set_xlabel("mover position (mm)")
    right.set_ylabel("EMF (V)")
    right.set_title(f"No-load back-EMF at {SPEED:g} m/s")
    right.legend(frameon=False, title="phase")
    for ax in (left, right):
        ax.grid(alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--no-steel", action="store_true",
                        help="leave out the steel teeth and shell: the coils in air")
    parser.add_argument("--vibestran", default="build/vibestran")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Mesh, write the deck and solve it.
    grids, triangles = make_mesh(steel=not args.no_steel)
    deck = args.output / "tubular_finite.bdf"
    write_deck(deck, grids, triangles)
    print(f"{len(triangles)} triangles; solving {deck}")
    subprocess.run([str(pathlib.Path(args.vibestran).resolve()), deck.name], check=True,
                   cwd=args.output, stdout=subprocess.DEVNULL)

    # Read each subcase's sweep and its saved fields.
    subcases = {}
    with h5py.File(args.output / "tubular_finite.mag.vtkhdf", "r") as results:
        for k, name in ((1, "no load"), (2, "rated")):
            group = results[f"MAGSTAT/subcases/{k}"]
            subcases[k] = {
                "position": group["positions_m"][()] * 1e3,
                "force": group["motion/force_N"][()],
                "emf": group["motion/emf_V"][()],
                "mean": group.attrs["force_mean_N"],
                "theta0": group["drive"].attrs["theta0_m"],
            }
            block = results[f"VTKHDF/SUBCASE_{k}"]
            for step in range(int(block["Steps"].attrs["NSteps"])):
                field = read_field(block, step)
                where = f"{round(field['position'] * 1e3) + 0:+d}mm"
                plot_field(field, triangles,
                           f"Finite tubular motor, {name}{'' if k == 1 else f' ({PEAK:g} A on q)'},"
                           f" mover at {where}: |B| and flux lines",
                           args.output / f"field_{name.replace(' ', '_')}_{where}.png")
    plot_curves(subcases, args.output / "curves.png")

    rated, no_load = subcases[2], subcases[1]
    print(f"THETA0 (AUTO): {no_load['theta0'] * 1e3:.3f} mm")
    print(f"thrust at {PEAK:g} A: mean {rated['mean']:.2f} N, "
          f"{rated['force'].min():.2f} to {rated['force'].max():.2f} N")
    print(f"no-load force: {no_load['force'].min():.2f} to {no_load['force'].max():.2f} N")
    print("wrote the field images and curves.png in", args.output)


if __name__ == "__main__":
    sys.exit(main())
