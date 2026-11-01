#!/usr/bin/env python3
"""External comparison of SOL MAGSTAT with xfemm (the FEMM 4.2 solver).

Each case is one geometry built with gmsh. vibestran solves the gmsh mesh.
xfemm gets the same geometry as FEMM nodes, segments, arcs and block labels,
and meshes it with its own mesher. The two discretizations are independent;
only the geometry, materials, sources and boundary conditions are shared.
Both are solved at two mesh levels (sizes scaled by 1 and 1/2), so each
solver's own refinement change can be set beside the difference between
the solvers. The motor has a third level, graded: quadrilaterals for
vibestran, fine at the gap (MOTOR_LEVELS), and FEMM's triangles from the
same size field. --airgap AGE BAND solves vibestran's motor both ways.

Cases:

- core: a U-core electromagnet and its armature, 200 turns around the
  yoke. Linear iron at one current and saturating steel at a current
  sweep: coil linkage, force on the armature, |B| at the gap.
- magnet: the same core with a magnet in the yoke, saturating steel, the
  coil at -4, 0 and +4 A.
- motor: the 12-slot, 14-pole surface-magnet motor of
  tools/magnetic_motor_example.py with dense saturating steel: rated
  torque and phase linkage over an electrical period, cogging over a
  cogging period, no-load phase linkage, and the radial flux density in
  the gap. The whole machine with the air-gap element (vibestran PAIRGAP,
  FEMM periodic AGE), and its anti-periodic half (MPERIOD ANTI; FEMM
  anti-periodic boundaries and anti-periodic AGE).

vibestran runs locally; femmcli runs on --femm-host over ssh. Results go to
OUTPUT/results.json, OUTPUT/summary.md and plots. femmcli must be built with
xfemm-validation.patch (see README.md): without it the post-processor reads
an uninitialized flag and crashes or misreports.

    .venv/bin/python Magnetic_Validation/run_xfemm.py OUTPUT [--cases core magnet motor]
        [--femm-host bob@bob-desktop.local] [--femmcli ~/xfemm-patched/cfemm/bin/femmcli]
"""

import argparse
import dataclasses
import json
import math
import pathlib
import hashlib
import subprocess
import sys
import time

import gmsh
import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import magnetic_motor_example as motor  # noqa: E402

MU0 = 4e-7 * math.pi
LEVELS = {"coarse": 1.0, "fine": 0.5}
# The motor's levels: motor_geometry's arguments. graded is the mesh study's
# fq:1.5/0.35 (README.md, "Motor mesh efficiency"): quadrilaterals for
# vibestran, fine at the gap; FEMM meshes the same size field in triangles.
MOTOR_LEVELS = {"coarse": {"scale": 1.0}, "fine": {"scale": 0.5},
                "graded": {"scale": 1.5, "near": 0.35, "quads": "full"}}


def steel_curve():
    """A smooth saturating steel, sampled densely so that FEMM's interpolant
    and vibestran's piecewise-linear one differ little: B = mu0 H +
    Js (2/pi) atan(pi (mu_i - 1) mu0 H / (2 Js)), Js = 1.8 T, mu_i = 5000."""
    js, mu_i = 1.8, 5000.0
    hs = [0.0] + list(np.geomspace(5.0, 1.0e6, 70))
    return [(MU0 * h + js * 2 / math.pi * math.atan(math.pi * (mu_i - 1) * MU0 * h / (2 * js)), h)
            for h in hs]


@dataclasses.dataclass(frozen=True)
class Role:
    """What a region of the geometry is. material: air, iron (linear,
    MUR 1000), steel (the curve above), magnet or gap (the air gap of the
    air-gap element). magnet: 'radial+', 'radial-' or a fixed angle in
    degrees. circuit/turns: a coil side. moving: turns with MOTION. body:
    the MFORCE body."""
    material: str
    region: str
    circuit: str = ""
    turns: float = 0.0
    magnet: object = None
    moving: bool = False
    body: bool = False


# --------------------------------------------------------------------------
# Geometry: gmsh fragments with roles, the mesh, and the FEMM drawing.
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Geometry:
    nodes: dict          # gmsh node tag -> (x, y)
    cells: list          # (role, [node tags]) triangles, or quadrilaterals with quads
    points: dict         # FEMM nodes: key -> (x, y)
    segments: list       # (key a, key b, kind) kind: exterior, age, gapside, inner, side1, side2
    arcs: list           # (key a, key b, angle deg, radius, mid (x, y), kind), counterclockwise a -> b
    labels: list         # (x, y, role, mesh size)
    side_edges: list     # half model: (side, node a, node b) mesh edges on the periodic sides
    size: object = None  # size(x, y): the target element size the mesh was built with


class Builder:
    """Closed outlines of points, lines and arcs about the origin."""

    def __init__(self):
        self.occ = gmsh.model.occ
        self.center = self.occ.addPoint(0, 0, 0)

    def surface(self, outline):
        points = [self.occ.addPoint(x, y, 0) for x, y, _ in outline]
        curves = []
        for k, (_, _, arc) in enumerate(outline):
            a, b = points[k], points[(k + 1) % len(points)]
            curves.append(self.occ.addCircleArc(a, self.center, b) if arc else self.occ.addLine(a, b))
        return self.occ.addPlaneSurface([self.occ.addCurveLoop(curves)])

    def rectangle(self, x0, y0, x1, y1):
        return self.occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)


def build(entities, size, half=False, quads=False):
    """entities: [(surface tag, priority, Role)]; size(x, y) and its gmsh
    MathEval expression in x, y: the target element size. half keeps
    y >= 0 and meshes the two halves of the cut periodically. quads
    recombines the triangles into quadrilaterals (Frontal-Delaunay for
    quads, Blossom matching), leaving a triangle only where a surface
    needs one; quads="full" recombines into quadrilaterals only (Blossom
    full-quad); quads="quadmesh" meshes every surface all-quad with
    WebMesh's quadmesh instead (see quadmesh_surfaces)."""
    occ = gmsh.model.occ
    if half:
        box = occ.addRectangle(-1, 0, 0, 2, 1)
        cut = []
        for tag, priority, role in entities:
            out, _ = occ.intersect([(2, tag)], [(2, box)], removeObject=True, removeTool=False)
            cut += [(t, priority, role) for d, t in out if d == 2]
        occ.remove([(2, box)], recursive=True)
        entities = cut
    occ.synchronize()
    tags = [(2, t) for t, _, _ in entities]
    # In a half model, a point at the center splits a cut line through it
    # into two, one per periodic side.
    center = [(0, occ.addPoint(0, 0, 0))] if half else []
    _, children = occ.fragment(tags[:1], tags[1:] + center)
    occ.synchronize()
    best = {}
    for (_, priority, role), kids in zip(entities, children):
        for kid in kids:
            if kid not in best or best[kid][0] < priority:
                best[kid] = (priority, role)
    # Remove what no role claims (none expected), then the drawing.
    roles = {kid[1]: role for kid, (_, role) in best.items()}
    field = gmsh.model.mesh.field
    f = field.add("MathEval")
    field.setString(f, "F", size[1])
    field.setAsBackgroundMesh(f)
    for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(opt, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8 if quads else 6)
    if quads:
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3 if quads == "full" else 1)
    curves = [c for _, c in gmsh.model.getEntities(1)]
    on_cut = {}
    if half:
        for c in curves:
            lo, hi = gmsh.model.getParametrizationBounds(1, c)
            ends = [gmsh.model.getValue(1, c, [t]) for t in (lo[0], hi[0])]
            middle = gmsh.model.getValue(1, c, [(lo[0] + hi[0]) / 2])
            # On the cut: both ends and the middle (the half circles end on it too).
            if all(abs(p[1]) < 1e-12 for p in ends + [middle]):
                on_cut[c] = ends
        first = [c for c, e in on_cut.items() if min(p[0] for p in e) > -1e-12]
        rotation = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        for c in first:
            image = sorted(-p[0] for p in on_cut[c])
            match = [d for d, e in on_cut.items()
                     if d not in first and np.allclose(sorted(p[0] for p in e), image, atol=1e-12)]
            assert len(match) == 1, (c, on_cut[c])
            gmsh.model.mesh.setPeriodic(1, [match[0]], [c], rotation)
    if quads == "quadmesh":
        assert not half, "quadmesh meshes whole models only"
        nodes, by_surface = quadmesh_surfaces(roles, size[0])
        cells = [(roles[s], cell) for s in roles for cell in by_surface[s]]
        geometry = drawing(roles, by_surface, nodes, size[0], on_cut)
        geometry.nodes, geometry.cells, geometry.side_edges = nodes, cells, []
        geometry.size = size[0]
        return geometry
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = dict(zip(tags.astype(int), coords.reshape(-1, 3)[:, :2]))
    cells, by_surface = [], {}
    for s, role in roles.items():
        types, _, conn = gmsh.model.mesh.getElements(2, s)
        for t, c in zip(types, conn):
            assert t in (2, 3), "triangles and quadrilaterals only"
            for cell in c.reshape(-1, 3 if t == 2 else 4).astype(int):
                cells.append((role, list(cell)))
                by_surface.setdefault(s, []).append(list(cell))
    # Mesh edges on the periodic sides.
    side_edges = []
    for c, ends in on_cut.items():
        upward = gmsh.model.getAdjacencies(1, c)[0]
        if any(roles[s].material == "gap" for s in upward):
            continue
        side = 1 if min(p[0] for p in ends) > -1e-12 else 2
        types, _, conn = gmsh.model.mesh.getElements(1, c)
        for t, e in zip(types, conn):
            for a, b in e.reshape(-1, 2).astype(int):
                side_edges.append((side, a, b))
    geometry = drawing(roles, by_surface, nodes, size[0], on_cut)
    geometry.nodes, geometry.cells, geometry.side_edges = nodes, cells, side_edges
    geometry.size = size[0]
    return geometry


def drawing(roles, by_surface, nodes, size, on_cut):
    """The FEMM drawing of the fragmented geometry: its points, its curves
    (closed circles split in two), the boundary kind of each curve, and a
    block label inside each surface with the coarsest target size over it."""
    points = {}

    def key(p):
        k = (round(p[0], 9), round(p[1], 9))
        points.setdefault(k, (float(p[0]), float(p[1])))
        return k

    segments, arcs = [], []
    for _, c in gmsh.model.getEntities(1):
        upward = list(gmsh.model.getAdjacencies(1, c)[0])
        kinds = [roles[s].material for s in upward]
        if c in on_cut:
            kind = "gapside" if "gap" in kinds else (
                "side1" if min(p[0] for p in on_cut[c]) > -1e-12 else "side2")
        elif "gap" in kinds:
            kind = "age" if len(upward) == 2 else "gapside"
        elif len(upward) == 1:
            kind = "exterior"
        else:
            kind = "inner"
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        t0, t1 = lo[0], hi[0]
        value = lambda t: np.array(gmsh.model.getValue(1, c, [t])[:2])  # noqa: E731
        if gmsh.model.getType(1, c) == "Line":
            segments.append((key(value(t0)), key(value(t1)), kind))
            continue
        pieces = [(t0, t1)]
        if np.linalg.norm(value(t0) - value(t1)) < 1e-12:  # a whole circle
            pieces = [(t0, (t0 + t1) / 2), ((t0 + t1) / 2, t1)]
        for u0, u1 in pieces:
            s, m, e = value(u0), value((u0 + u1) / 2), value(u1)
            # Circle through s, m, e, and the counterclockwise sweep.
            ax, ay, bx, by, cx, cy = *s, *m, *e
            d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
                  + (cx * cx + cy * cy) * (ay - by)) / d
            uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
                  + (cx * cx + cy * cy) * (bx - ax)) / d
            center = np.array([ux, uy])
            radius = float(np.linalg.norm(s - center))
            ccw = (m[0] - s[0]) * (e[1] - s[1]) - (m[1] - s[1]) * (e[0] - s[0]) > 0
            a, b = (s, e) if ccw else (e, s)
            sweep = (math.atan2(*(b - center)[::-1]) - math.atan2(*(a - center)[::-1])) % (2 * math.pi)
            arcs.append((key(a), key(b), math.degrees(sweep), radius, tuple(m), kind))
    labels = []
    for s, tris in by_surface.items():
        best = max(tris, key=lambda t: area([nodes[n] for n in t]))
        inside = np.mean([nodes[n] for n in best], axis=0)
        coarsest = max(size(*nodes[n]) for t in tris for n in t)
        labels.append((float(inside[0]), float(inside[1]), roles[s], coarsest))
    return Geometry({}, [], points, segments, arcs, labels, [])


def area(p):
    return abs((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0])) / 2


# --------------------------------------------------------------------------
# The cases.
# --------------------------------------------------------------------------

def core_geometry(scale, magnet):
    """U-core (60 x 50 mm, 15 mm legs and yoke), a 60 x 15 mm armature 1.5 mm
    above the legs, a 200-turn coil around the yoke, and a 240 mm air box. With
    magnet, a 10 mm magnet magnetized along +x replaces the middle of the
    yoke."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("core")
    b = Builder()
    mm = 1e-3
    iron = "steel"
    e = [(b.rectangle(-120 * mm, -120 * mm, 120 * mm, 120 * mm), 1, Role("air", "air"))]
    e.append((b.rectangle(-30 * mm, 0, 30 * mm, 50 * mm), 2, Role(iron, "core")))
    e.append((b.rectangle(-15 * mm, 15 * mm, 15 * mm, 50 * mm), 3, Role("air", "air")))
    e.append((b.rectangle(-30 * mm, 51.5 * mm, 30 * mm, 66.5 * mm), 2, Role(iron, "armature", body=True)))
    e.append((b.rectangle(-12 * mm, 17 * mm, 12 * mm, 29 * mm), 4, Role("air", "coil+", "coil", 200)))
    e.append((b.rectangle(-12 * mm, -14 * mm, 12 * mm, -2 * mm), 4, Role("air", "coil-", "coil", -200)))
    if magnet:
        e.append((b.rectangle(-5 * mm, 0, 5 * mm, 15 * mm), 5, Role("magnet", "magnet", magnet=0.0)))
    expr = (f"{scale} * Min(6e-3, 2.5e-4 + 0.15 * Max(0, Sqrt(Max(Abs(x) - 0.03, 0)^2"
            " + (y - 0.05075)^2) - 0.001))")
    g = build(e, (lambda x, y: core_size(x, y, scale), expr))
    gmsh.finalize()
    return g


QUADMESH = pathlib.Path.home() / "Documents" / "WebMesh-Editor"


def quadmesh_surfaces(roles, size):
    """Every surface meshed all-quad by quadmesh (WebMesh-Editor's
    meshers/quadmesh: its constrained recombination path, with smoothing),
    inside gmsh's nodes on its boundary curves, fixed, so that neighbouring
    surfaces conform. An all-quad surface needs an even number of boundary
    edges: curves gain a node, in the middle of their longest edge, along a
    tree from the exterior to each surface with an odd count (a curve's node
    changes the count of both its surfaces). Returns the nodes and each
    surface's quadrilaterals."""
    sys.path.insert(0, str(QUADMESH / "meshers" / "quadmesh" / "bindings" / "python"))
    from quadmesh import Mesher
    mesher = Mesher(QUADMESH / "zig-out" / "lib" / "libquadmesh_capi.dylib")
    gmsh.model.mesh.generate(1)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = dict(zip(tags.astype(int), coords.reshape(-1, 3)[:, :2]))
    # Each curve's nodes and parameters, first to last.
    curve = {}
    for _, c in gmsh.model.getEntities(1):
        inner, _, t = gmsh.model.mesh.getNodes(1, c, returnParametricCoord=True)
        ends = [gmsh.model.mesh.getNodes(0, p)[0][0] for _, p in gmsh.model.getBoundary([(1, c)])]
        if len(ends) == 0:  # a whole circle: its seam point is its only vertex
            ends = [gmsh.model.mesh.getNodes(0, gmsh.model.getAdjacencies(1, c)[1][0])[0][0]] * 2
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        # The vertex at the parametrization's start first.
        start = np.array(gmsh.model.getValue(1, c, [lo[0]])[:2])
        if np.linalg.norm(nodes[int(ends[0])] - start) > np.linalg.norm(nodes[int(ends[-1])] - start):
            ends = ends[::-1]
        order = np.argsort(t)
        curve[c] = ([int(ends[0])] + [int(inner[k]) for k in order] + [int(ends[-1])],
                    [lo[0]] + [float(t[k]) for k in order] + [hi[0]])
    boundary = {s: [(1 if c > 0 else -1, abs(c)) for _, c in
                    gmsh.model.getBoundary([(2, s)], combined=False, oriented=False)] for s in roles}
    # Parity: a tree from the exterior over curves shared with neighbours.
    odd = {s: sum(len(curve[c][0]) - 1 for _, c in boundary[s]) % 2 == 1 for s in roles}
    beside = {}
    for c in curve:
        up = [s for s in gmsh.model.getAdjacencies(1, c)[0] if s in roles]
        for s in up:
            beside.setdefault(s, []).append((c, next((u for u in up if u != s), None)))
    parent, order, frontier = {}, [], [s for s in roles if any(o is None for _, o in beside[s])]
    for s in frontier:
        parent[s] = (next(c for c, o in beside[s] if o is None), None)
    while frontier:
        s = frontier.pop(0)
        order.append(s)
        for c, o in beside[s]:
            if o is not None and o not in parent:
                parent[o] = (c, s)
                frontier.append(o)
    assert len(order) == len(roles)
    next_tag = max(nodes) + 1
    for s in reversed(order):
        if not odd[s]:
            continue
        c, up = parent[s]
        ids, ts = curve[c]
        k = max(range(len(ids) - 1), key=lambda k: np.linalg.norm(nodes[ids[k + 1]] - nodes[ids[k]]))
        t = (ts[k] + ts[k + 1]) / 2
        nodes[next_tag] = np.array(gmsh.model.getValue(1, c, [t])[:2])
        curve[c] = (ids[:k + 1] + [next_tag] + ids[k + 1:], ts[:k + 1] + [t] + ts[k + 1:])
        next_tag += 1
        odd[s] = False
        if up is not None:
            odd[up] = not odd[up]
    by_surface = {}
    for s in roles:
        # Chain the curves into closed loops of node tags (oriented below).
        chains = [list(curve[c][0]) for _, c in boundary[s]]
        loops = []
        while chains:
            loop = chains.pop(0)
            while loop[0] != loop[-1]:
                k = next(k for k, ch in enumerate(chains) if loop[-1] in (ch[0], ch[-1]))
                ch = chains.pop(k)
                loop += (ch if ch[0] == loop[-1] else ch[::-1])[1:]
            loops.append(loop[:-1])
        def signed_area(loop):
            p = np.array([nodes[n] for n in loop])
            return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        loops.sort(key=lambda loop: -abs(signed_area(loop)))
        loops = [loop if (signed_area(loop) > 0) == (k == 0) else loop[::-1] for k, loop in enumerate(loops)]
        # In millimetres: quadmesh's tolerances suit unit-sized coordinates.
        target = 1e3 * max(size(*nodes[n]) for loop in loops for n in loop)
        def mesh(jitter):
            rng = np.random.default_rng(s)
            spec = []
            for k, loop in enumerate(loops):
                points = [tuple(1e3 * nodes[n] + jitter * rng.standard_normal(2)) for n in loop]
                spec.append({"points": points, "role": "outer" if k == 0 else "hole",
                             "fixed_runs": [{"nodes": points, "closed": True}]})
            return mesher.mesh(spec, target_size=target, finite_element_sizing=1, enable_smoothing=1)
        try:
            r = mesh(0)
        except ValueError:
            # quadmesh's constrained triangulation can fail to recover an
            # edge between cocircular nodes ("cannot recover boundary edge"),
            # and meshes the same loops moved by 1e-9 mm. The boundary nodes
            # keep their exact positions below, through the fixed-node map.
            r = mesh(1e-9)
        assert r["valid"], (s, roles[s], r["issues"])
        tag = {}
        for b in r["boundary_nodes"]:
            tag[b["node"]] = loops[b["loop"]][b["fixed_node"]]
        for k, p in enumerate(r["nodes"]):
            if k not in tag:
                nodes[next_tag] = np.array(p) * 1e-3
                tag[k] = next_tag
                next_tag += 1
        by_surface[s] = [[tag[n] for n in q] for q in r["quads"]]
    return nodes, by_surface


def motor_geometry(scale, half, quads=False, near=None):
    """tools/magnetic_motor_example.py's motor, with the PAIRGAP annulus
    between the magnets' outer radius and the bore. The element size is
    scale times min(1.4 mm, 0.32 mm + 0.12 |r - R_BORE|); with near, it is
    min(scale 1.4 mm, near 0.32 mm + 0.12 |r - R_BORE|) instead: the gap's
    size set apart from the far field's."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("motor")
    occ = gmsh.model.occ
    b = Builder()
    e = []
    for radius, role, priority in [(motor.R_SHAFT, Role("air", "shaft", moving=True), 5),
                                   (motor.R_YOKE, Role("steel", "rotor", moving=True), 4),
                                   (motor.R_MAGNET, Role("air", "interpole", moving=True), 3),
                                   (motor.R_BORE, Role("gap", "gap"), 2),
                                   (motor.R_OUTER, Role("steel", "stator"), 1)]:
        e.append((occ.addDisk(0, 0, 0, radius, radius), priority, role))
    if quads == "quadmesh":
        # The stator in two surfaces, teeth and yoke, split at the slot
        # bottoms: quadmesh inverts quadrilaterals at the reflex corners
        # where a slot's bottom meets its sides.
        e.append((occ.addDisk(0, 0, 0, motor.R_SLOT_BOTTOM, motor.R_SLOT_BOTTOM), 1.5,
                  Role("steel", "stator")))
    pitch = math.pi / motor.POLE_PAIRS
    for j in range(2 * motor.POLE_PAIRS):
        a0, a1 = (j - motor.MAGNET_ARC / 2) * pitch, (j + motor.MAGNET_ARC / 2) * pitch
        tag = b.surface([(*motor.polar(motor.R_YOKE, a0), False), (*motor.polar(motor.R_MAGNET, a0), True),
                         (*motor.polar(motor.R_MAGNET, a1), False), (*motor.polar(motor.R_YOKE, a1), True)])
        e.append((tag, 6, Role("magnet", f"magnet{j}", magnet="radial+" if j % 2 == 0 else "radial-",
                               moving=True)))
    layout = motor.coil_layout()
    for slot in range(motor.SLOTS):
        sigma = 2 * math.pi * (slot + .5) / motor.SLOTS
        opening, clockwise, counter = motor.slot_outlines(sigma)
        e.append((b.surface(opening), 7, Role("air", f"opening{slot}")))
        for outline, tooth, side in ((clockwise, slot, 1), (counter, (slot + 1) % motor.SLOTS, -1)):
            phase, polarity = layout[tooth]
            e.append((b.surface(outline), 8, Role("air", f"coil{slot}{'ab'[side < 0]}", "ABC"[phase],
                                                  side * polarity * motor.TURNS)))
    if near is None:
        expr = f"{scale} * Min(1.4e-3, 3.2e-4 + 0.12 * Abs(Sqrt(x*x + y*y) - {motor.R_BORE}))"

        def size(x, y):
            return scale * min(1.4e-3, 3.2e-4 + 0.12 * abs(math.hypot(x, y) - motor.R_BORE))
    else:
        expr = f"Min({scale * 1.4e-3}, {near * 3.2e-4} + 0.12 * Abs(Sqrt(x*x + y*y) - {motor.R_BORE}))"

        def size(x, y):
            return min(scale * 1.4e-3, near * 3.2e-4 + 0.12 * abs(math.hypot(x, y) - motor.R_BORE))
    g = build(e, (size, expr), half=half, quads=quads)
    gmsh.finalize()
    return g


# --------------------------------------------------------------------------
# vibestran decks and results.
# --------------------------------------------------------------------------

def table_cards(tid):
    flat = [f"{v:.10g}" for pair in steel_curve() for v in pair] + ["ENDT"]
    out = [f"TABLEBH,{tid}"]
    for k in range(0, len(flat), 8):
        out.append("," + ",".join(flat[k:k + 8]))
    return out


def vibestran_deck(g, control, bulk, motion_sets=False, period=None):
    """The deck for geometry g: grids and elements, a PID per material and
    magnetization, ESETs of coils, the body and the moving cells, COIL2D and
    CIRCUIT cards. Returns the text and the element IDs of each region."""
    pids, lines = {}, []
    materials = {"air": "MATMAG,1,1.", "iron": "MATMAG,2,1000.", "steel": "MATMAG,3,,1",
                 "magnet": f"MATMAG,4,1.05,,{motor.BR}"}
    mids = {"air": 1, "iron": 2, "steel": 3, "magnet": 4}
    cells = sorted(g.cells, key=lambda c: not c[0].moving)  # moving cells first
    regions = {}
    for eid, (role, tri) in enumerate(cells, start=1):
        regions.setdefault(role.region, []).append(eid)
        key = (role.material, role.magnet)
        if key not in pids:
            pid = 9 if role.material == "gap" else 10 + len(pids)
            pids[key] = pid
            if role.material == "gap":
                lines.append("PAIRGAP,9")
            elif role.magnet is None:
                lines.append(f"PMAG2D,{pid},{mids[role.material]}")
            elif role.magnet in ("radial+", "radial-"):
                lines.append(f"PMAG2D,{pid},4,100,{1 if role.magnet == 'radial+' else -1}")
            else:
                lines.append(f"PMAG2D,{pid},4,0,1")
    text = ["SOL MAGSTAT", "CEND", *control, "BEGIN BULK", "MAG2D,0,{depth}",
            "CORD2C,100,,0.,0.,0.,0.,0.,1.", ",1.,0.,0.", *materials.values(), *lines,
            *table_cards(1), *bulk]
    used = sorted({n for _, t in cells for n in t})
    for n in used:
        x, y = g.nodes[n]
        text.append(f"GRID,{n},,{x:.17g},{y:.17g},0.")
    for eid, (role, cell) in enumerate(cells, start=1):
        card = "CTRIA3" if len(cell) == 3 else "CQUAD4"
        text.append(f"{card},{eid},{pids[(role.material, role.magnet)]}," + ",".join(map(str, cell)))
    circuits = {}
    set_id = 1000
    for role in {c[0] for c in cells}:
        if role.circuit:
            set_id += 1
            text.append(f"ESET,{set_id}," + ",".join(map(str, regions[role.region])))
            text.append(f"COIL2D,{set_id},{set_id},{role.turns:.17g}")
            circuits.setdefault(role.circuit, []).append(set_id)
    names = sorted(circuits)
    for k, name in enumerate(names, start=1):
        text.append(f"CIRCUIT,{k},1," + ",".join(f"{s},1" for s in circuits[name]))
    body = [eid for eid, (role, _) in enumerate(cells, start=1) if role.body]
    if body:
        text.append("ESET,30," + ",".join(map(str, body)))
    if motion_sets:
        moving = [eid for eid, (role, _) in enumerate(cells, start=1) if role.moving]
        text.append(f"ESET,20,{moving[0]},THRU,{moving[-1]}")
        assert moving == list(range(moving[0], moving[-1] + 1))
    if period:
        # Each side edge with an element that owns it.
        owner = {}
        for eid, (_, cell) in enumerate(cells, start=1):
            for k in range(len(cell)):
                owner[frozenset((cell[k], cell[(k + 1) % len(cell)]))] = eid
        for side, a, b in g.side_edges:
            text.append(f"FSET,{30 + side},{owner[frozenset((a, b))]},{a},{b}")
        text.append("MPERIOD,1,ANTI,31,32,0,2")
    text.append("ENDDATA")
    return "\n".join(text) + "\n", {name: k for k, name in enumerate(names, start=1)}


def run_vibestran(deck_path, executable):
    start = time.perf_counter()
    subprocess.run([executable, deck_path.name], check=True, cwd=deck_path.parent,
                   stdout=subprocess.DEVNULL)
    (deck_path.parent / "seconds.txt").write_text(f"{time.perf_counter() - start:.2f}\n")
    return h5py.File(deck_path.with_suffix(".mag.vtkhdf"), "r")


# --------------------------------------------------------------------------
# FEMM Lua.
# --------------------------------------------------------------------------

def femm_age_band(g, r_in, r_out, half, size):
    """FEMM's air-gap element assumes evenly spaced nodes on both of its
    circles, the same number on each; it gives every AGE arc one side length,
    so circles made of arcs of unequal span (magnet faces and interpoles,
    tooth faces and slot openings) misregister. The usual FEMM drawing is
    used instead: meshed air rings over the outer and inner thirds of the gap
    and the AGE between two whole circles. The field problem is the same; the
    gap is air either way."""
    r1, r2 = r_in + (r_out - r_in) / 3, r_out - (r_out - r_in) / 3
    points = dict(g.points)

    def key(x, y):
        k = (round(x, 9), round(y, 9))
        points.setdefault(k, (x, y))
        return k
    segments = []
    for s in g.segments:
        if s[2] != "gapside":
            segments.append(s)
            continue
        # A cut of the half model across the gap: ring, no-mesh, ring pieces.
        sign = 1 if g.points[s[0]][0] + g.points[s[1]][0] > 0 else -1
        radii = [r_in, r1, r2, r_out]
        kinds = ["side1" if sign > 0 else "side2", "gapside", "side1" if sign > 0 else "side2"]
        for k in range(3):
            segments.append((key(sign * radii[k], 0.0), key(sign * radii[k + 1], 0.0), kinds[k]))
    arcs = [(a, b, angle, radius, mid, "inner" if kind == "age" else kind)
            for a, b, angle, radius, mid, kind in g.arcs]
    for r in (r1, r2):
        arcs.append((key(r, 0.0), key(-r, 0.0), 180.0, r, (0.0, r), "age"))
        if not half:
            arcs.append((key(-r, 0.0), key(r, 0.0), 180.0, r, (0.0, -r), "age"))
    labels = [lab for lab in g.labels if lab[2].material != "gap"]
    gap = next(lab[2] for lab in g.labels if lab[2].material == "gap")
    labels += [(0.0, (r1 + r2) / 2, gap, 0.0),
               (0.0, (r_in + r1) / 2, Role("air", "rotor gap ring"), min(size(0, r1), r1 - r_in)),
               (0.0, (r2 + r_out) / 2, Role("air", "stator gap ring"), min(size(0, r2), r_out - r2))]
    return dataclasses.replace(g, points=points, segments=segments, arcs=arcs, labels=labels)


def femm_drawing(g, depth, half, size):
    """Lua that draws g, with its materials, circuits and boundaries."""
    # Only the materials in use.
    used = {lab[2].material for lab in g.labels}
    definitions = {
        "air": 'mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)',
        "iron": 'mi_addmaterial("Iron", 1000, 1000, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)',
        "steel": 'mi_addmaterial("Steel", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)',
        "magnet": f'mi_addmaterial("Magnet", 1.05, 1.05, {motor.BR / (MU0 * 1.05):.17g}, 0, 0, 0, 0, 1, '
                  '0, 0, 0, 0, 0)'}
    out = [f'newdocument(0)', f'mi_probdef(0, "meters", "planar", 1e-9, {depth}, 30)',
           *(v for k, v in definitions.items() if k in used),
           'mi_addboundprop("A0", 0, 0, 0, 0, 0, 0, 0, 0, 0)',
           f'mi_addboundprop("AGE", 0, 0, 0, 0, 0, 0, 0, 0, {7 if half else 6})']
    if "steel" in used:
        out += [f'mi_addbhpoint("Steel", {bh:.17g}, {hh:.17g})' for bh, hh in steel_curve()[1:]]
    circuits = sorted({lab[2].circuit for lab in g.labels if lab[2].circuit})
    out += [f'mi_addcircprop("{c}", 0, 1)' for c in circuits]
    for x, y in g.points.values():
        out.append(f"mi_addnode({x:.17g}, {y:.17g})")
    pairs = []
    if half:
        # Each segment of the first side (x > 0 on y = 0) with its image
        # turned by 180 degrees: an anti-periodic boundary of its own.
        def span(seg):
            return sorted(round(g.points[k][0], 9) for k in seg[:2])
        firsts = [t for t in g.segments if t[2] == "side1"]
        for k, s in enumerate(firsts):
            image = sorted(-x for x in span(s))
            match = [t for t in g.segments if t[2] == "side2" and span(t) == image]
            assert len(match) == 1, s
            out.append(f'mi_addboundprop("ap{k}", 0, 0, 0, 0, 0, 0, 0, 0, 5)')
            pairs += [(s, f"ap{k}"), (match[0], f"ap{k}")]
    named = {id(s): name for s, name in pairs}
    for s in g.segments:
        (ax, ay), (bx, by) = g.points[s[0]], g.points[s[1]]
        out.append(f"mi_addsegment({ax:.17g}, {ay:.17g}, {bx:.17g}, {by:.17g})")
    for a, b, angle, radius, mid, kind in g.arcs:
        (ax, ay), (bx, by) = g.points[a], g.points[b]
        out.append(f"mi_addarc({ax:.17g}, {ay:.17g}, {bx:.17g}, {by:.17g}, {angle:.17g}, 1)")
    # Boundary conditions and the discretization of each curve.
    for s in g.segments:
        (ax, ay), (bx, by) = g.points[s[0]], g.points[s[1]]
        mx, my = (ax + bx) / 2, (ay + by) / 2
        prop = {"exterior": "A0"}.get(s[2], named.get(id(s), "<None>"))
        out += [f"mi_selectsegment({mx:.17g}, {my:.17g})",
                f'mi_setsegmentprop("{prop}", {size(mx, my):.6g}, 0, 0, 0)', "mi_clearselected()"]
    for a, b, angle, radius, mid, kind in g.arcs:
        prop = {"exterior": "A0", "age": "AGE"}.get(kind, "<None>")
        step = math.degrees(size(*mid) / radius)
        out += [f"mi_selectarcsegment({mid[0]:.17g}, {mid[1]:.17g})",
                f'mi_setarcsegmentprop({min(step, 5):.6g}, "{prop}", 0, 0)', "mi_clearselected()"]
    for x, y, role, mesh in g.labels:
        out.append(f"mi_addblocklabel({x:.17g}, {y:.17g})")
        out.append(f"mi_selectlabel({x:.17g}, {y:.17g})")
        material = {"air": "Air", "iron": "Iron", "steel": "Steel", "magnet": "Magnet"}.get(role.material)
        if material is None:
            out.append('mi_setblockprop("<No Mesh>", 1, 0, "<None>", 0, 0, 0)')
        else:
            direction = {"radial+": '"theta"', "radial-": '"theta+180"'}.get(role.magnet, role.magnet or 0)
            out.append(f'mi_setblockprop("{material}", 0, {mesh:.6g}, "{role.circuit or "<None>"}", '
                       f'{direction}, {1 if role.body else 0}, {role.turns:.17g})')
        out.append("mi_clearselected()")
    out.append('mi_saveas("model.fem")')
    return out


LUA_PRELUDE = r"""
results = openfile("results.txt", "w")
function put(name, ...)
  write(results, name)
  for k = 1, getn(arg) do write(results, format(" %.17g", arg[k])) end
  write(results, "\n")
end
function solve()
  mi_analyze(1)
  mi_loadsolution()
end
"""


def run_femm(lua, workdir, args):
    """Runs the script with femmcli on the remote host; returns the parsed
    results.txt (name -> list of rows of numbers) and femmcli's wall time in
    seconds. A script solved before, in any case of OUTPUT, is not solved
    again: OUTPUT/femm_cache holds each script's results and time."""
    text = "\n".join(lua) + "\n"
    (workdir / "case.lua").write_text(text)
    cache = workdir.parent / "femm_cache" / hashlib.sha256(text.encode()).hexdigest()[:16]
    if not (cache / "results.txt").exists():
        remote = f"~/xfemm-validation/{workdir.name}"
        subprocess.run(["ssh", args.femm_host, f"mkdir -p {remote}"], check=True)
        subprocess.run(["scp", "-q", str(workdir / "case.lua"), f"{args.femm_host}:{remote}/"], check=True)
        # femmcli's own wall time, measured on the host.
        subprocess.run(["ssh", args.femm_host,
                        f"cd {remote} && rm -f results.txt && s=$(date +%s.%N) && "
                        f"{args.femmcli} -q --lua-script=case.lua > femm.log 2>&1 && "
                        f"echo \"$(date +%s.%N) - $s\" | bc > seconds.txt"], check=True)
        cache.mkdir(parents=True, exist_ok=True)
        for name in ("results.txt", "seconds.txt"):
            subprocess.run(["scp", "-q", f"{args.femm_host}:{remote}/{name}", str(cache / name)], check=True)
    (workdir / "femm_results.txt").write_text((cache / "results.txt").read_text())
    seconds = float((cache / "seconds.txt").read_text())
    out = {}
    for line in (workdir / "femm_results.txt").read_text().splitlines():
        name, *values = line.split()
        out.setdefault(name, []).append([float(v) for v in values])
    return out, seconds


# --------------------------------------------------------------------------
# Core and magnet cases.
# --------------------------------------------------------------------------

CORE_CURRENTS = {"core": [("linear", 2.0)] + [("steel", i) for i in (0.5, 1, 2, 4, 8, 16)],
                 "magnet": [("steel", i) for i in (-4.0, 0.0, 4.0)]}
GAP_POINT = (-0.0225, 0.05075)  # middle of the left leg's gap


def core_case(name, level, out, args):
    scale = LEVELS[level]
    g = core_geometry(scale, magnet=name == "magnet")
    results = {"vibestran": [], "femm": []}
    variants = CORE_CURRENTS[name]
    # One deck and one FEMM script per material model: "linear" has MUR
    # 1000 iron in place of the steel curve.
    for material in sorted({m for m, _ in variants}):
        workdir = out / f"{name}_{level}_{material}"
        workdir.mkdir(parents=True, exist_ok=True)
        currents = [i for m, i in variants if m == material]
        swap = (lambda r: dataclasses.replace(r, material="iron") if r.material == "steel" else r) \
            if material == "linear" else (lambda r: r)
        geo = dataclasses.replace(g, cells=[(swap(r), t) for r, t in g.cells],
                                  labels=[(x, y, swap(r), m) for x, y, r, m in g.labels])
        control = ["MPROBE = ALL"]
        bulk = ["PARAM,MAGTOL,1e-10", "MFORCE,1,BODY,30,0,1",
                f"MPROBE,1,0,{GAP_POINT[0]},{GAP_POINT[1]},0."]
        for k, current in enumerate(currents, start=1):
            control += [f"SUBCASE {k}", f"  LOAD = {k if current else 0}"]
            if current:
                bulk.append(f"ICIRC,{k},1,{current:.17g}")
        deck, _ = vibestran_deck(geo, control, bulk)
        path = workdir / "core.bdf"
        path.write_text(deck.replace("{depth}", "0.1"))
        with run_vibestran(path, args.vibestran) as h5:
            for k, current in enumerate(currents, start=1):
                s = h5[f"MAGSTAT/subcases/{k}"]
                b = s["probes/B_T"][()][0]
                force = s["forces/force_N"][()][0]
                results["vibestran"].append({
                    "material": material, "current": current,
                    "linkage": float(s["circuits/linkage_Wb_turn"][()][0]),
                    "force_x": float(force[0]), "force_y": float(force[1]),
                    "b_gap": float(np.hypot(b[0], b[1])), "energy": float(s.attrs["energy_J"]),
                    "cells": len(geo.cells)})
        lua = femm_drawing(geo, 0.1, False, lambda x, y: core_size(x, y, scale))
        lua.append(LUA_PRELUDE)
        for current in currents:
            lua += [f'mi_modifycircprop("coil", 1, {current:.17g})', "solve()",
                    'ic, vc, flux = mo_getcircuitproperties("coil")',
                    "mo_groupselectblock(1)", "fx = mo_blockintegral(18)", "fy = mo_blockintegral(19)",
                    "mo_clearblock()",
                    f"a, b1, b2 = mo_getpointvalues({GAP_POINT[0]}, {GAP_POINT[1]})",
                    "mo_groupselectblock()", "w = mo_blockintegral(2)", "mo_clearblock()",
                    f'put("state", {current:.17g}, flux, fx, fy, b1, b2, w, mo_numelements())',
                    "mo_close()"]
        lua += ["closefile(results)", "quit()"]
        for row in run_femm(lua, workdir, args)[0]["state"]:
            current, linkage, fx, fy, b1, b2, energy, elements = row
            results["femm"].append({"material": material, "current": current, "linkage": linkage,
                                    "force_x": fx, "force_y": fy, "b_gap": math.hypot(b1, b2),
                                    "energy": energy, "cells": int(elements)})
    return results


def core_size(x, y, scale):
    dx = max(abs(x) - 0.03, 0)
    return scale * min(6e-3, 2.5e-4 + 0.15 * max(0, math.hypot(dx, y - 0.05075) - 0.001))


# --------------------------------------------------------------------------
# Motor.
# --------------------------------------------------------------------------

ELECTRICAL = 360.0 / motor.POLE_PAIRS        # mechanical degrees per electrical period
COGGING = 360.0 / math.lcm(motor.SLOTS, 2 * motor.POLE_PAIRS)
THETA0 = 15.0 / motor.POLE_PAIRS             # phase A's d axis (15 electrical degrees)
SWEEPS = {"rated": (ELECTRICAL / 24, 24, motor.PEAK), "noload": (ELECTRICAL / 24, 24, 0.0),
          "cogging": (COGGING / 30, 30, 0.0)}
GAP_ANGLES = np.arange(0, 360, 2.0)
# Where B is sampled in the gap: in the middle of FEMM's meshed stator-side
# air ring (see femm_age_band), inside vibestran's air-gap element. xfemm's
# mo_getgapb is not used: it returns the AGE field scaled by 1/100 (and,
# without xfemm-validation.patch, only B_t).
GAP_RADIUS = motor.R_BORE - (motor.R_BORE - motor.R_MAGNET) / 6


def phase_currents(position, peak):
    """DRIVE SINE at gamma = 0: i_k = -I sin(theta_e - 2 pi k / 3)."""
    theta = math.radians(motor.POLE_PAIRS * (position - THETA0))
    return [-peak * math.sin(theta - 2 * math.pi * k / 3) for k in range(3)]


def motor_deck(g, half):
    """The motor deck for geometry g: the SWEEPS as MOTION subcases (DRIVE
    for the loaded ones) and gap probes at GAP_RADIUS. Returns the deck and
    the probe angles."""
    control, bulk = [], ["PARAM,MAGTOL,1e-10"]
    for k, (name, (step, count, peak)) in enumerate(SWEEPS.items(), start=1):
        control += [f"SUBCASE {k}", f"  LABEL = {name.upper()}", f"  MOTION = {k}", "  MSTEP = ALL",
                    "  BFIELD = NONE", f"  MPROBE = {'ALL' if name != 'cogging' else 'NONE'}"]
        bulk.append(f"MOTION,{k},20,0,ROTATE,0.,{step:.17g},{count}")
        if name != "cogging":
            control.append(f"  DRIVE = {k}")
            bulk += [f"DRIVE,{k},SINE,{motor.POLE_PAIRS},{peak:.17g},0.,{THETA0:.17g}", ",1,2,3"]
    # A half model is sampled over its half and extended anti-periodically.
    angles = GAP_ANGLES[GAP_ANGLES < 180] if half else GAP_ANGLES
    for k, angle in enumerate(angles, start=1):
        x, y = GAP_RADIUS * math.cos(math.radians(angle)), GAP_RADIUS * math.sin(math.radians(angle))
        bulk.append(f"MPROBE,{k},0,{x:.17g},{y:.17g},0.")
    deck, circuit_ids = vibestran_deck(g, control, bulk, motion_sets=True, period=half)
    assert circuit_ids == {"A": 1, "B": 2, "C": 3}
    return deck.replace("{depth}", str(motor.DEPTH)), angles


def motor_results(h5, angles, half):
    """Torque, phase linkage and the gap's radial B of each sweep."""
    vib = {}
    for k, name in enumerate(SWEEPS, start=1):
        s = h5[f"MAGSTAT/subcases/{k}"]
        vib[name] = {"positions": s["positions_deg"][()].tolist(),
                     "torque": s["motion/torque_Nm"][()].tolist(),
                     "linkage": s["circuits/linkage_Wb_turn"][()].tolist()}
        if "probes/B_T" in s:
            b = s["probes/B_T"][()][0]
            t = np.radians(angles)
            br = b[:, 0] * np.cos(t) + b[:, 1] * np.sin(t)
            vib[name]["gap_br"] = (np.concatenate([br, -br]) if half else br).tolist()
    return vib


def motor_case(variant, level, out, args):
    """vibestran with each air-gap method of --airgap (AGE: "vibestran",
    BAND: "vibestran_band") and FEMM with its AGE, on the level's mesh."""
    half = variant == "half"
    workdir = out / f"motor_{variant}_{level}"
    workdir.mkdir(parents=True, exist_ok=True)
    g = motor_geometry(half=half, **MOTOR_LEVELS[level])
    deck, angles = motor_deck(g, half)
    result = {}
    for method in args.airgap:
        rundir = workdir if method == "AGE" else workdir / method.lower()
        rundir.mkdir(exist_ok=True)
        path = rundir / "motor.bdf"
        path.write_text(deck if method == "AGE" else
                        deck.replace("BEGIN BULK\n", f"BEGIN BULK\nPARAM,AIRGAP,{method}\n", 1))
        with run_vibestran(path, args.vibestran) as h5:
            vib = motor_results(h5, angles, half)
        vib["cells"] = len(g.cells)
        vib["quads"] = sum(len(c) == 4 for _, c in g.cells)
        vib["seconds"] = float((rundir / "seconds.txt").read_text())
        result["vibestran" if method == "AGE" else f"vibestran_{method.lower()}"] = vib

    def whole_circle(br):
        return np.concatenate([br, -br]) if half else br
    radius = GAP_RADIUS
    size = g.size
    lua = femm_drawing(femm_age_band(g, motor.R_MAGNET, motor.R_BORE, half, size), motor.DEPTH, half, size)
    lua.append(LUA_PRELUDE)
    for name, (step, count, peak) in SWEEPS.items():
        for k in range(count):
            position = k * step
            currents = phase_currents(position, peak)
            lua += [f'mi_modifyboundprop("AGE", 10, {position:.17g})']
            lua += [f'mi_modifycircprop("{p}", 1, {i:.17g})' for p, i in zip("ABC", currents)]
            lua += ["solve()", 'i1, v1, fa = mo_getcircuitproperties("A")',
                    'i2, v2, fb = mo_getcircuitproperties("B")', 'i3, v3, fc = mo_getcircuitproperties("C")',
                    f'put("{name}", {position:.17g}, mo_gapintegral("AGE", 0), fa, fb, fc, mo_numelements())']
            if k == 0 and name != "cogging":
                for a in angles:
                    x, y = radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a))
                    lua += [f"a, b1, b2 = mo_getpointvalues({x:.17g}, {y:.17g})",
                            f'put("{name}_gap", {a:g}, b1, b2)']
            lua.append("mo_close()")
    lua += ["closefile(results)", "quit()"]
    femm_raw, seconds = run_femm(lua, workdir, args)
    femm = {"cells": int(femm_raw["rated"][0][5]), "seconds": seconds}
    for name in SWEEPS:
        rows = np.array(femm_raw[name])
        femm[name] = {"positions": rows[:, 0].tolist(), "torque": rows[:, 1].tolist(),
                      "linkage": rows[:, 2:5].tolist()}
        if f"{name}_gap" in femm_raw:
            rows = np.array(femm_raw[f"{name}_gap"])
            t = np.radians(rows[:, 0])
            femm[name]["gap_br"] = whole_circle(rows[:, 1] * np.cos(t) + rows[:, 2] * np.sin(t)).tolist()
    return {**result, "femm": femm}


# --------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------

def rel(a, b):
    """a against b, in percent of |b|."""
    return 100 * (a - b) / abs(b)


def harmonic(values, order):
    """Amplitude of harmonic `order` of samples spread uniformly over one
    period of the sampled function."""
    v = np.asarray(values, dtype=float)
    k = np.arange(len(v))
    return 2 * abs(np.sum(v * np.exp(-2j * np.pi * order * k / len(v)))) / len(v)


def core_rows(data, case, level):
    d = data.get(f"{case}_{level}")
    return [] if d is None else list(zip(d["vibestran"], d["femm"]))


CORE_METRICS = [("linkage", "linkage (Wb-turn)"), ("force_y", "armature F_y (N)"),
                ("b_gap", "gap B magnitude (T)"), ("energy", "energy (J)")]


def core_report(data, case):
    lines = []
    for level in LEVELS:
        rows = core_rows(data, case, level)
        if not rows:
            continue
        v0, f0 = rows[0]
        lines += [f"### {case}, {level} mesh ({v0['cells']} vibestran triangles, "
                  f"{f0['cells']} FEMM triangles)", "",
                  "| material | I (A) | " + " | ".join(f"{t}: vibestran / FEMM / diff %" for _, t in CORE_METRICS)
                  + " | F_x / abs F_y % (vib, FEMM) |",
                  "|---|---:|" + "---:|" * (len(CORE_METRICS) + 1)]
        for v, f in rows:
            cells = [f"{v[k]:.5g} / {f[k]:.5g} / {rel(v[k], f[k]):+.2f}" for k, _ in CORE_METRICS]
            fx = f"{100 * v['force_x'] / abs(v['force_y']):+.3f}, {100 * f['force_x'] / abs(f['force_y']):+.3f}"
            lines.append(f"| {v['material']} | {v['current']:g} | " + " | ".join(cells) + f" | {fx} |")
        lines.append("")
    coarse, fine = core_rows(data, case, "coarse"), core_rows(data, case, "fine")
    if coarse and fine:
        lines += [f"### {case}, coarse to fine change (%)", "",
                  "| material | I (A) | " + " | ".join(f"{t}: vibestran / FEMM" for _, t in CORE_METRICS) + " |",
                  "|---|---:|" + "---:|" * len(CORE_METRICS)]
        for (vc, fc), (vf, ff) in zip(coarse, fine):
            cells = [f"{rel(vf[k], vc[k]):+.2f} / {rel(ff[k], fc[k]):+.2f}" for k, _ in CORE_METRICS]
            lines.append(f"| {vc['material']} | {vc['current']:g} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def motor_metrics(r, half_femm_scale=1.0):
    """Scalar measures of one solver's motor results. half_femm_scale
    multiplies FEMM's linkage in a half model (its circuits hold half the
    coils)."""
    p = motor.POLE_PAIRS
    rated, noload, cog = r["rated"], r["noload"], r["cogging"]
    la = np.asarray(rated["linkage"])[:, 0] * half_femm_scale
    l0 = np.asarray(noload["linkage"])[:, 0] * half_femm_scale
    out = {"rated mean torque (N m)": float(np.mean(rated["torque"])),
           "rated phase A linkage, fundamental (Wb-turn)": harmonic(la, 1),
           "no-load phase A linkage, fundamental (Wb-turn)": harmonic(l0, 1),
           "cogging torque p-p (N m)": float(np.ptp(cog["torque"]))}
    if "gap_br" in noload:
        out[f"no-load gap B_r, harmonic {p} (T)"] = harmonic(noload["gap_br"], p)
    return out


def vibestran_runs(r):
    """The vibestran results of one motor case: (label, results), the AGE's
    first."""
    return [(k.replace("vibestran_", "").upper() if k != "vibestran" else "AGE", v)
            for k, v in sorted(r.items(), key=lambda kv: kv[0] != "vibestran")
            if k.startswith("vibestran")]


def motor_report(data):
    lines = []
    keys = [k for k in (f"motor_{v}_{lvl}" for v in ("whole", "half") for lvl in MOTOR_LEVELS) if k in data]
    if not keys:
        return lines
    table = {}
    for key in keys:
        half = "_half_" in key
        r = data[key]
        table[key] = (motor_metrics(r["vibestran"]), motor_metrics(r["femm"], 2.0 if half else 1.0),
                      r["vibestran"]["cells"], r["femm"]["cells"])
    names = list(next(iter(table.values()))[0])
    lines += ["vibestran's cells are triangles, or quadrilaterals at the graded level; FEMM's are "
              "triangles.", "",
              "| case | vibestran gap | cells (vib / FEMM) | " + " | ".join(f"{n}: vibestran / FEMM / diff %" for n in names)
              + " |", "|---|---|---:|" + "---:|" * len(names)]
    for key in keys:
        r, f = data[key], table[key][1]
        for method, run in vibestran_runs(r):
            v = motor_metrics(run)
            cells = [f"{v[n]:.5g} / {f[n]:.5g} / {rel(v[n], f[n]):+.2f}" for n in names]
            lines.append(f"| {key[6:]} | {method} | {run['cells']} / {r['femm']['cells']} | "
                         + " | ".join(cells) + " |")
    lines.append("")
    timed = [k for k in keys if "seconds" in data[k]["femm"]]
    if timed:
        lines += ["Wall time of the 78 positions (s): vibestran on the machine running this script, "
                  "femmcli (single-threaded; each position remeshes, solves, writes and reloads its "
                  "solution) on --femm-host:", "",
                  "| case | " + " | ".join(f"vibestran {m}" for m, _ in vibestran_runs(data[timed[0]]))
                  + " | FEMM |", "|---|" + "---:|" * (len(vibestran_runs(data[timed[0]])) + 1)]
        for key in timed:
            lines.append(f"| {key[6:]} | " + " | ".join(f"{run['seconds']:.1f}" for _, run in vibestran_runs(data[key]))
                         + f" | {data[key]['femm']['seconds']:.1f} |")
        lines.append("")
    # Pointwise curve differences, as a fraction of the curve's scale.
    lines += ["Largest pointwise difference between the solvers, as a percentage of the FEMM curve's "
              "largest magnitude:", "",
              "| case | rated torque | rated phase A linkage | no-load phase A linkage | cogging torque "
              "| no-load gap B_r |", "|---|---:|---:|---:|---:|---:|"]
    for key in keys:
        r, s = data[key], 2.0 if "_half_" in key else 1.0

        def worst(a, b):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            return f"{100 * np.max(np.abs(a - b)) / np.max(np.abs(b)):.2f}"
        v, f = r["vibestran"], r["femm"]
        lines.append(f"| {key[6:]} | {worst(v['rated']['torque'], f['rated']['torque'])} | "
                     f"{worst(np.asarray(v['rated']['linkage'])[:, 0], s * np.asarray(f['rated']['linkage'])[:, 0])} | "
                     f"{worst(np.asarray(v['noload']['linkage'])[:, 0], s * np.asarray(f['noload']['linkage'])[:, 0])} | "
                     f"{worst(v['cogging']['torque'], f['cogging']['torque'])} | "
                     f"{worst(v['noload']['gap_br'], f['noload']['gap_br'])} |")
    lines.append("")
    # Each solver's half model against its whole model.
    pairs = [lvl for lvl in MOTOR_LEVELS if f"motor_whole_{lvl}" in table and f"motor_half_{lvl}" in table]
    if pairs:
        lines += ["Half model against the whole model, each solver (%):", "",
                  "| level | " + " | ".join(f"{n}: vibestran / FEMM" for n in names) + " |",
                  "|---|" + "---:|" * len(names)]
        for lvl in pairs:
            vw, fw, _, _ = table[f"motor_whole_{lvl}"]
            vh, fh, _, _ = table[f"motor_half_{lvl}"]
            lines.append(f"| {lvl} | " + " | ".join(f"{rel(vh[n], vw[n]):+.3f} / {rel(fh[n], fw[n]):+.3f}"
                                                   for n in names) + " |")
        lines.append("")
    return lines


# Categorical slots 1 and 2 of the validated default palette, and its ink.
COLORS = {"vibestran": "#2a78d6", "FEMM": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e4e3dd", "#fcfcfb"


def styled_axes(fig, rows, cols):
    axes = fig.subplots(rows, cols, squeeze=False)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK)
    return axes


def series(ax, x, y, solver, label=None):
    style = dict(color=COLORS[solver], linewidth=2, label=label or solver)
    if solver == "FEMM":
        ax.plot(x, y, linestyle="--", marker="s", markersize=4, markerfacecolor="none", **style)
    else:
        ax.plot(x, y, marker="o", markersize=4, **style)


def plots(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK, "axes.titlecolor": INK,
                         "legend.frameon": False})
    written = []
    level = "fine" if "core_fine" in data else "coarse"
    cases = [c for c in ("core", "magnet") if f"{c}_{level}" in data]
    if cases:
        fig = plt.figure(figsize=(9, 3.2 * len(cases)), facecolor=SURFACE)
        axes = styled_axes(fig, len(cases), 2)
        for row, case in zip(axes, cases):
            rows = [(v, f) for v, f in core_rows(data, case, level) if v["material"] == "steel"]
            current = [v["current"] for v, _ in rows]
            for ax, key, title in ((row[0], "force_y", "armature force F_y (N)"),
                                   (row[1], "linkage", "coil linkage (Wb-turn)")):
                series(ax, current, [v[key] for v, _ in rows], "vibestran")
                series(ax, current, [f[key] for _, f in rows], "FEMM")
                ax.set_title(f"{case}, steel, {level} mesh: {title}", loc="left")
                ax.set_xlabel("coil current (A)")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / "core.png", dpi=150)
        written.append("core.png")
        plt.close(fig)
    key = next((k for k in (f"motor_whole_{lvl}" for lvl in ("graded", "fine", "coarse")) if k in data), None)
    if key:
        r = data[key]
        fig = plt.figure(figsize=(9, 6.4), facecolor=SURFACE)
        axes = styled_axes(fig, 2, 2)
        panels = [(axes[0, 0], "rated", "torque", "rated torque (N m)", "rotor position (deg)"),
                  (axes[0, 1], "cogging", "torque", "cogging torque (N m)", "rotor position (deg)"),
                  (axes[1, 0], "rated", "linkage", "rated phase A linkage (Wb-turn)", "rotor position (deg)")]
        for ax, sweep, quantity, title, xlabel in panels:
            for solver, name in (("vibestran", "vibestran"), ("femm", "FEMM")):
                s = r[solver][sweep]
                y = np.asarray(s[quantity])
                series(ax, s["positions"], y[:, 0] if y.ndim == 2 else y, name)
            ax.set_title(title, loc="left")
            ax.set_xlabel(xlabel)
        ax = axes[1, 1]
        series(ax, GAP_ANGLES, r["vibestran"]["noload"]["gap_br"], "vibestran")
        series(ax, GAP_ANGLES, r["femm"]["noload"]["gap_br"], "FEMM")
        ax.set_title(f"no-load gap B_r at r = {1e3 * GAP_RADIUS:.2f} mm, position 0 (T)", loc="left")
        ax.set_xlabel("angle (deg)")
        ax.set_xlim(0, 360 / motor.POLE_PAIRS * 2)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2)
        fig.suptitle(f"12-slot, 14-pole motor, whole machine, {key.split('_')[-1]} mesh",
                     x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / "motor.png", dpi=150)
        written.append("motor.png")
        plt.close(fig)
    return written


def report(data, out):
    lines = ["# vibestran SOL MAGSTAT against xfemm", "",
             "Generated by Magnetic_Validation/run_xfemm.py. Differences are vibestran against FEMM, "
             "in percent of the FEMM value.", ""]
    for case, title in (("core", "U-core electromagnet"), ("magnet", "U-core with a magnet in the yoke")):
        body = core_report(data, case)
        if body:
            lines += [f"## {title}", "", *body]
    body = motor_report(data)
    if body:
        lines += ["## 12-slot, 14-pole motor", "",
                  "Half-model FEMM linkages are doubled: the half model's circuits hold half of each "
                  "phase's coils. Torques are the whole machine's in both solvers. The rated sweep's "
                  "step is half the cogging period, so it does not resolve the torque ripple; the "
                  "cogging sweep does.", "", *body]
    for name in plots(data, out):
        lines += [f"![{name}]({name})", ""]
    (out / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--report-only", action="store_true",
                        help="rewrite summary.md and the plots from OUTPUT/results.json")
    parser.add_argument("--cases", nargs="+", default=["core", "magnet", "motor"])
    parser.add_argument("--levels", nargs="+", default=list(MOTOR_LEVELS),
                        help="mesh levels; graded is the motor's only (the core cases skip it)")
    parser.add_argument("--airgap", nargs="+", default=["AGE"], choices=["AGE", "BAND"],
                        help="vibestran's motor air-gap methods (FEMM always uses its AGE)")
    parser.add_argument("--vibestran", default=str(pathlib.Path("build/vibestran").resolve()))
    parser.add_argument("--femm-host", default="bob@bob-desktop.local")
    parser.add_argument("--femmcli", default="~/xfemm-patched/cfemm/bin/femmcli")
    parser.add_argument("--motor-variants", nargs="+", default=["whole", "half"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store = args.output / "results.json"
    data = json.loads(store.read_text()) if store.exists() else {}
    if args.report_only:
        report(data, args.output)
        print("wrote", args.output / "summary.md")
        return 0
    for case in args.cases:
        for level in args.levels:
            if case == "motor":
                for variant in args.motor_variants:
                    print(f"motor {variant} {level}", flush=True)
                    data[f"motor_{variant}_{level}"] = motor_case(variant, level, args.output, args)
                    store.write_text(json.dumps(data, indent=1))
            elif level in LEVELS:
                print(f"{case} {level}", flush=True)
                data[f"{case}_{level}"] = core_case(case, level, args.output, args)
                store.write_text(json.dumps(data, indent=1))
    report(data, args.output)
    print("wrote", store, "and", args.output / "summary.md")


if __name__ == "__main__":
    sys.exit(main())
