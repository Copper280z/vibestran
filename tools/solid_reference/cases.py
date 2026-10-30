"""CHEXA8 reference models of the MYSTRAN validation shell decks whose
correct answer is in question.  Each builder writes a deck for one mesh
refinement and returns the quantities to measure as
{name: (grid list, component index 0-5)}; a quantity is the mean of the
listed grids' component."""

import math
import numpy as np
from deckio import Deck
from mesh import Block, face_loads, add_loads


def _material(deck, E, nu, alpha=None, tref=None):
    G = E / (2.0 * (1.0 + nu))
    if alpha is None:
        deck.line(f"MAT1*   {1:<16d}{E:16.9E}{G:16.9E}{nu:16.9E}*")
        deck.line("*")
    else:
        deck.line(f"MAT1*   {1:<16d}{E:16.9E}{G:16.9E}{nu:16.9E}*")
        deck.line(f"*       {0.0:16.9E}{alpha:16.9E}{tref:16.9E}")
    deck.line("PSOLID  1       1")


def _total(loads):
    return sum(np.asarray(f) for f in loads.values())


def hemisphere(path, n, nw=2, t=0.04, F=1.0):
    """MacNeal-Harder pinched hemisphere with an 18-degree hole, quarter
    model: R = 10, E = 6.825e7, nu = 0.3, +x / -y loads F at the equator."""
    R = 10.0

    def P(u, v, w):
        lam = 0.5 * math.pi * u
        th = math.radians(90.0 - 72.0 * v)
        r = R + 0.5 * w * t
        return (r * math.sin(th) * math.cos(lam), r * math.sin(th) * math.sin(lam), r * math.cos(th))

    b = Block(P, n, n, nw)
    d = Deck(f"hemisphere t={t} hex8 {n}x{n}x{nw}", case_extra=["SPC = 1", "LOAD = 1"])
    _material(d, 6.825e7, 0.3)
    b.write(d)
    d.spc1(1, 2, b.layer_ids("u", 0))          # y = 0 symmetry plane
    d.spc1(1, 1, b.layer_ids("u", n))          # x = 0 symmetry plane
    d.spc1(1, 3, [b.nid(n // 2, 0, nw // 2)])  # rigid z translation
    # Line loads through the thickness at the two equator points.
    for i, direction in ((0, (1.0, 0.0, 0.0)), (n, (0.0, -1.0, 0.0))):
        for k in range(nw + 1):
            wgt = (0.5 if k in (0, nw) else 1.0) / nw
            d.force(1, b.nid(i, 0, k), *(F * wgt * np.asarray(direction)))
    d.write(path)
    return {"TX at +x load (mid-thickness)": ([b.nid(0, 0, nw // 2)], 0),
            "TY at -y load (mid-thickness)": ([b.nid(n, 0, nw // 2)], 1)}


def twisted_plate(path, n, nw=4, t=0.6, F=9.90272e7):
    """MYSTRAN 'twisted plate mit4+': unit square, corners z = +-0.1
    (bilinear), clamped at y = 0, in-plane tension F along +y on y = 1."""

    def mid(x, y):
        return np.array([x, y, -0.1 + 0.2 * x + 0.2 * y - 0.4 * x * y])

    def normal(x, y):
        nv = np.array([-(0.2 - 0.4 * y), -(0.2 - 0.4 * x), 1.0])
        return nv / np.linalg.norm(nv)

    def P(u, v, w):
        return mid(u, v) + 0.5 * w * t * normal(u, v)

    b = Block(P, n, n, nw)
    d = Deck(f"twisted plate t={t} hex8 {n}x{n}x{nw}", case_extra=["SPC = 1", "LOAD = 1"])
    _material(d, 2.1e11, 0.0)
    b.write(d)
    d.spc1(1, 123, b.layer_ids("v", 0))
    loads, _ = face_loads(b.face("v", n), traction=lambda x: np.array([0.0, 1.0, 0.0]))
    add_loads(d, 1, loads, F / _total(loads)[1])
    d.write(path)
    return {"TZ at corner (0,1), mid-thickness": ([b.nid(0, n, nw // 2)], 2)}


def raasch_hook(path, nu, nv, nw=2):
    """Raasch hook (SI): 60-degree arc R = 0.3556 then 150-degree arc
    R = 1.1684 in the x-z plane, width 0.508 along y, t = 0.0508,
    E = 2.2752699e7, nu = 0.35, clamped at the small-arc end, 1 lbf
    (4.4482216 N) shear along +y on the free end."""
    R1, R2, W, t = 0.3556, 1.1684, 0.508, 0.0508
    c1, c2 = np.array([-R1, 0.0]), np.array([R2, 0.0])
    L1, L2 = R1 * math.pi / 3.0, R2 * 5.0 * math.pi / 6.0
    n1 = max(2, round(nu * L1 / (L1 + L2)))
    f1 = n1 / nu

    def P(u, v, w):
        z = 0.5 * w * t
        if u <= f1 + 1e-12:
            th = math.radians(-60.0) + (u / f1) * (math.pi / 3.0)
            p = c1 + (R1 + z) * np.array([math.cos(th), math.sin(th)])
        else:
            th = math.pi - ((u - f1) / (1.0 - f1)) * (5.0 * math.pi / 6.0)
            p = c2 + (R2 - z) * np.array([math.cos(th), math.sin(th)])
        return (p[0], v * W, p[1])

    b = Block(P, nu, nv, nw)
    d = Deck(f"Raasch hook hex8 {nu}x{nv}x{nw}", case_extra=["SPC = 1", "LOAD = 1"])
    _material(d, 2.2752699067e7, 0.35)
    b.write(d)
    d.spc1(1, 123, b.layer_ids("u", 0))
    loads, _ = face_loads(b.face("u", nu), traction=lambda x: np.array([0.0, 1.0, 0.0]))
    add_loads(d, 1, loads, 4.4482216 / _total(loads)[1])
    d.write(path)
    return {"TY free-end corner y=0 (mid)": ([b.nid(nu, 0, nw // 2)], 1),
            "TY free-end corner y=W (mid)": ([b.nid(nu, nv, nw // 2)], 1),
            "TY free-end mean (mid)": ([b.nid(nu, j, nw // 2) for j in range(nv + 1)], 1)}


def cylinder_pressure(path, nth, ny, nw=2, where="split", supports="deck"):
    """MYSTRAN 'diamond cylinder pressure': R = 3, t = 0.1, y from 11 to 25,
    E = 2e11, nu = 0.3, outward pressure 1e4 on the mid-surface; open ends,
    TY = 0 on the y = 11 end and, as in the deck (grid 56), TX = TZ = 0 on
    the thickness line at theta = 0, y = 15 (supports="deck"; ny must be a
    multiple of 7), or statically determinate supports ("free").  The
    mid-surface load is applied as the
    same radial force per unit area of mid-surface on the inner face
    ('inner'), the outer face ('outer') or half on each ('split')."""
    R, t, p = 3.0, 0.1, 1.0e4

    def P(u, v, w):
        th = 2.0 * math.pi * u
        r = R + 0.5 * w * t
        return (r * math.cos(th), 11.0 + 14.0 * v, r * math.sin(th))

    b = Block(P, nth, ny, nw, periodic_u=True)
    d = Deck(f"cylinder pressure ({where}, {supports}) hex8 {nth}x{ny}x{nw}",
             case_extra=["SPC = 1", "LOAD = 1"])
    _material(d, 2.0e11, 0.3)
    b.write(d)
    d.spc1(1, 2, b.layer_ids("v", 0))
    mid = nw // 2
    if supports == "deck":
        assert ny % 7 == 0, "y = 15 must be a node layer"
        d.spc1(1, 13, [b.nid(0, 2 * ny // 7, k) for k in range(nw + 1)])
    else:
        d.spc1(1, 3, [b.nid(0, 0, mid), b.nid(nth // 2, 0, mid)])
        d.spc1(1, 1, [b.nid(nth // 4, 0, mid)])
    total_radial = p * R * 2.0 * math.pi * 14.0          # mid-surface force
    shares = {"inner": {0: 1.0}, "outer": {nw: 1.0}, "split": {0: 0.5, nw: 0.5}}[where]
    for k, share in shares.items():
        radial = lambda x: np.array([x[0], 0.0, x[2]]) / math.hypot(x[0], x[2])
        loads, area = face_loads(b.face("w", k), traction=radial)
        mag = sum(np.dot(f, radial(xyz)) for gid, f in loads.items()
                  for xyz in [b.xyz(*_ijk(b, gid))])
        add_loads(d, 1, loads, share * total_radial / mag)
    d.write(path)
    th120 = round(nth / 3)
    return {"TY at y=25, theta=120 deg (mid)": ([b.nid(th120, ny, mid)], 1),
            "TY at y=25, mean (mid)": ([b.nid(i, ny, mid) for i in range(nth)], 1)}


def _ijk(b, gid):
    g = gid - 1
    i = g % b.nu_nodes
    g //= b.nu_nodes
    j = g % (b.nv + 1)
    return i, j, g // (b.nv + 1)


def half_cylinder_thermal(path, nth, ny, nw=2):
    """MYSTRAN 'thermal strain free rectangle cylinder': half cylinder
    R = 3, t = 0.1, y in [0, 16], E = 2e11, nu = 0.3, alpha = 3e-4,
    uniform dT = 100 from 273.15; supports that allow free expansion about
    the origin (TZ = 0 at z = 0, TX = 0 on x = 0, TY = 0 at y = 0)."""
    R, t = 3.0, 0.1

    def P(u, v, w):
        th = math.pi * u
        r = R + 0.5 * w * t
        return (r * math.cos(th), 16.0 * v, r * math.sin(th))

    b = Block(P, nth, ny, nw)
    d = Deck(f"half cylinder thermal hex8 {nth}x{ny}x{nw}",
             case_extra=["SPC = 1", "TEMPERATURE = 4", "STRESS = ALL"])
    _material(d, 2.0e11, 0.3, alpha=3.0e-4, tref=273.15)
    nodes = b.write(d)
    d.spc1(1, 3, b.layer_ids("u", 0) + b.layer_ids("u", nth))
    d.spc1(1, 1, b.layer_ids("u", nth // 2))
    d.spc1(1, 2, b.layer_ids("v", 0))
    d.line(f"TEMPD   4       {273.15}")
    for gid, _ in b.nodes():
        d.line(f"TEMP    4       {gid:<8d}{373.15}")
    d.write(path)
    mid = nw // 2
    return {"TZ at top (theta=90), y=16 (mid)": ([b.nid(nth // 2, ny, mid)], 2),
            "TY at y=16 (mid)": ([b.nid(nth // 2, ny, mid)], 1),
            "TX at theta=0 (mid)": ([b.nid(0, ny // 2, mid)], 0)}
