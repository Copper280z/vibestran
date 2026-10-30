"""Structured CHEXA8 meshes of a mapped block and consistent face loads."""

import numpy as np
from deckio import Deck


class Block:
    """Hexahedral mesh of P(u, v, w), u, v in [0, 1], w in [-1, 1] (w is the
    thickness coordinate), with nu x nv x nw elements."""

    def __init__(self, P, nu, nv, nw, periodic_u=False):
        self.P, self.nu, self.nv, self.nw, self.per = P, nu, nv, nw, periodic_u
        self.nu_nodes = nu if periodic_u else nu + 1

    def nid(self, i, j, k):
        if self.per:
            i %= self.nu
        return 1 + i + self.nu_nodes * (j + (self.nv + 1) * k)

    def xyz(self, i, j, k):
        return np.asarray(self.P(i / self.nu, j / self.nv, -1.0 + 2.0 * k / self.nw), float)

    def nodes(self):
        for k in range(self.nw + 1):
            for j in range(self.nv + 1):
                for i in range(self.nu_nodes):
                    yield self.nid(i, j, k), self.xyz(i, j, k)

    def write(self, deck, pid=1):
        for gid, p in self.nodes():
            deck.grid(gid, *p)
        # Orientation from the first element's Jacobian sign.
        a, b, c, d = (self.xyz(0, 0, 0), self.xyz(1, 0, 0), self.xyz(0, 1, 0), self.xyz(0, 0, 1))
        flip = np.dot(np.cross(b - a, c - a), d - a) < 0.0
        eid = 1
        for k in range(self.nw):
            for j in range(self.nv):
                for i in range(self.nu):
                    lo = [self.nid(i, j, k), self.nid(i + 1, j, k), self.nid(i + 1, j + 1, k),
                          self.nid(i, j + 1, k)]
                    hi = [self.nid(i, j, k + 1), self.nid(i + 1, j, k + 1),
                          self.nid(i + 1, j + 1, k + 1), self.nid(i, j + 1, k + 1)]
                    deck.chexa(eid, pid, (hi + lo) if flip else (lo + hi))
                    eid += 1
        return eid - 1

    def face(self, which, index):
        """Quads of the node layer i = index ('u'), j = index ('v') or k = index
        ('w'), as lists of (ids, positions)."""
        quads = []
        if which == "u":
            for k in range(self.nw):
                for j in range(self.nv):
                    ijk = [(index, j, k), (index, j + 1, k), (index, j + 1, k + 1), (index, j, k + 1)]
                    quads.append(ijk)
        elif which == "v":
            for k in range(self.nw):
                for i in range(self.nu):
                    ijk = [(i, index, k), (i + 1, index, k), (i + 1, index, k + 1), (i, index, k + 1)]
                    quads.append(ijk)
        else:
            for j in range(self.nv):
                for i in range(self.nu):
                    ijk = [(i, j, index), (i + 1, j, index), (i + 1, j + 1, index), (i, j + 1, index)]
                    quads.append(ijk)
        return [([self.nid(*q) for q in quad], [self.xyz(*q) for q in quad]) for quad in quads]

    def layer_ids(self, which, index):
        ids = set()
        for quad_ids, _ in self.face(which, index):
            ids.update(quad_ids)
        return sorted(ids)


_G = np.polynomial.legendre.leggauss(2)


def face_loads(quads, traction=None, pressure=None, sign=1.0):
    """Consistent nodal forces of bilinear quads: a traction vector field
    t(x) or a pressure along the quad normal (sign chooses the side).
    Returns ({gid: force}, total area)."""
    loads, area = {}, 0.0
    g, w = _G
    for ids, X in quads:
        for r, wr in zip(g, w):
            for s, ws in zip(g, w):
                N = [0.25 * (1 - r) * (1 - s), 0.25 * (1 + r) * (1 - s),
                     0.25 * (1 + r) * (1 + s), 0.25 * (1 - r) * (1 + s)]
                dr = 0.25 * (-(1 - s) * X[0] + (1 - s) * X[1] + (1 + s) * X[2] - (1 + s) * X[3])
                ds = 0.25 * (-(1 - r) * X[0] - (1 + r) * X[1] + (1 + r) * X[2] + (1 - r) * X[3])
                nA = np.cross(dr, ds) * wr * ws
                dA = np.linalg.norm(nA)
                x = sum(n * xi for n, xi in zip(N, X))
                f = traction(x) * dA if traction is not None else sign * pressure * nA
                area += dA
                for gid, n in zip(ids, N):
                    loads[gid] = loads.get(gid, 0.0) + n * f
    return loads, area


def add_loads(deck, sid, loads, scale=1.0):
    for gid, f in sorted(loads.items()):
        deck.force(sid, gid, *(scale * np.asarray(f)))
