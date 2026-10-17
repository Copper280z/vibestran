"""Prototype 20-node hexahedron formulations (research only).

Variants (strain treatments; all integrate the energy with 3x3x3 Gauss
points unless noted):
  full    standard displacement element
  red     plain 2x2x2 integration
  as      MYSTRAN feat/hexa-modern "assumed strain": shear strains of the
          centroid natural frame projected onto the trilinear space P8, then
          a B-bar dilatation onto P8
  asn     as + each natural normal strain e_ii projected onto
          {1, xi_i} x Q2(other two)   (directional membrane reduction)
  mitc    asn, but shear g_ij projected onto {1,xi_i}{1,xi_j}{1,xi_k,xi_k^2}
  mitcn   mitc without the B-bar
Projections are det(J)-weighted L2 projections over the element.
"""
import itertools
import numpy as np

# ---- serendipity shape functions, Nastran CHEXA20 node order -------------
CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7),
         (4, 5), (5, 6), (6, 7), (7, 4)]
NAT = np.vstack([CORNERS, [(CORNERS[a] + CORNERS[b]) / 2 for a, b in EDGES]])


def shape(x):
    """N (20,), dN/dxi (20,3) at natural point x."""
    N = np.zeros(20)
    dN = np.zeros((20, 3))
    for i, c in enumerate(NAT):
        if i < 8:
            f = [1 + x[k] * c[k] for k in range(3)]
            s = x @ c - 2
            N[i] = f[0] * f[1] * f[2] * s / 8
            for k in range(3):
                o = [f[m] for m in range(3) if m != k]
                dN[i, k] = (c[k] * o[0] * o[1] * s + f[0] * f[1] * f[2] * c[k]) / 8
        else:
            z = int(np.flatnonzero(c == 0)[0])  # direction along the edge
            g = [1 - x[k] ** 2 if k == z else 1 + x[k] * c[k] for k in range(3)]
            N[i] = g[0] * g[1] * g[2] / 4
            for k in range(3):
                dg = -2 * x[k] if k == z else c[k]
                o = [g[m] for m in range(3) if m != k]
                dN[i, k] = dg * o[0] * o[1] / 4
    return N, dN


def gauss(n):
    p, w = np.polynomial.legendre.leggauss(n)
    pts, wts = [], []
    for i, j, k in itertools.product(range(n), repeat=3):
        pts.append([p[i], p[j], p[k]])
        wts.append(w[i] * w[j] * w[k])
    return np.array(pts), np.array(wts)


def D_iso(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[:3, :3] += 2 * G * np.eye(3)
    D[3:, 3:] = G * np.eye(3)
    return D


def Bmat(dNdx):
    B = np.zeros((6, 60))
    for a in range(20):
        dx, dy, dz = dNdx[a]
        c = 3 * a
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c], B[3, c + 1] = dy, dx          # gamma_xy
        B[4, c + 1], B[4, c + 2] = dz, dy      # gamma_yz
        B[5, c], B[5, c + 2] = dz, dx          # gamma_zx
    return B


def voigt_to_tensor(v):
    return np.array([[v[0], v[3] / 2, v[5] / 2], [v[3] / 2, v[1], v[4] / 2],
                     [v[5] / 2, v[4] / 2, v[2]]])


def natural_frame(J0):
    """TN: Cartesian engineering Voigt -> natural (11,22,33,12,23,31)."""
    t = J0 / np.linalg.norm(J0, axis=0)
    TN = np.zeros((6, 6))
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
    for c in range(6):
        e = np.zeros(6)
        e[c] = 1.0
        T = voigt_to_tensor(e)
        for r, (i, j) in enumerate(pairs):
            v = t[:, i] @ T @ t[:, j]
            TN[r, c] = v if i == j else 2 * v
    return TN


# ---- projection spaces (functions of the natural point) ------------------
def mono(x, exps):
    return np.array([x[0] ** a * x[1] ** b * x[2] ** c for a, b, c in exps])


P8 = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def dir_space(i):
    """{1, xi_i} x Q2(other two) for the normal strain e_ii."""
    out = []
    for e in itertools.product(range(3), repeat=3):
        if e[i] <= 1:
            out.append(e)
    return out


def shear_space(i, j):
    k = 3 - i - j
    out = []
    for e in itertools.product(range(3), repeat=3):
        if e[i] <= 1 and e[j] <= 1:
            out.append(e)
    return out


SPACES = {
    "P8": P8,
    "N0": dir_space(0), "N1": dir_space(1), "N2": dir_space(2),
    "S01": shear_space(0, 1), "S12": shear_space(1, 2), "S20": shear_space(2, 0),
}


class Element:
    def __init__(self, X, E, nu, variant="as", ngp=3):
        self.X = np.asarray(X, float)
        self.D = D_iso(E, nu)
        self.variant = variant
        self.ngp = 2 if variant == "red" else ngp

    def stiffness(self):
        pts, wts = gauss(self.ngp)
        X = self.X
        _, dN0 = shape(np.zeros(3))
        J0 = X.T @ dN0
        TN = natural_frame(J0)
        TC = np.linalg.inv(TN)
        Bs, dV = [], []
        for x, w in zip(pts, wts):
            _, dN = shape(x)
            J = X.T @ dN
            dNdx = dN @ np.linalg.inv(J)
            Bs.append(Bmat(dNdx))
            dV.append(w * np.linalg.det(J))
        Bs = np.array(Bs)
        dV = np.array(dV)
        if np.any(dV <= 0):
            raise ValueError("non-positive Jacobian")
        v = self.variant
        if v in ("asnl", "asnl2"):
            # local natural frame at each point, normal strains onto
            # {1, xi_i} x Q2, shears onto P8; constant strain restored by the
            # mean-strain correction A(B) + Bm - A(Bm); then B-bar
            def proj(rows, space):
                S = np.array([mono(x, SPACES[space]) for x in pts])
                W = S.T @ (dV[:, None] * S)
                return S @ np.linalg.solve(W, S.T @ (dV[:, None] * rows))
            TNs, TCs = [], []
            for x in pts:
                _, dN = shape(x)
                T = natural_frame(X.T @ dN)
                TNs.append(T); TCs.append(np.linalg.inv(T))
            TNs, TCs = np.array(TNs), np.array(TCs)
            def A(Bf):
                Nat = np.einsum("gij,gjk->gik", TNs, Bf)
                out = Nat.copy()
                for r in (0, 1, 2):
                    out[:, r, :] = proj(Nat[:, r, :], "N%d" % r)
                for r in (3, 4, 5):
                    out[:, r, :] = proj(Nat[:, r, :], "P8")
                return np.einsum("gij,gjk->gik", TCs, out)
            Bm = np.einsum("gik,g->ik", Bs, dV) / dV.sum()
            Bmg = np.repeat(Bm[None], len(pts), axis=0)
            if v == "asnl":
                Bb = A(Bs) + Bmg - A(Bmg)
            else:  # asnl2: reproduces constant strain and is work-free against constant stress
                Dv = A(Bs - Bmg)
                Dm = np.einsum("gik,g->ik", Dv, dV) / dV.sum()
                Bb = Dv - Dm[None] + Bmg
            m = np.array([1, 1, 1, 0, 0, 0.0])
            d = np.einsum("i,gik->gk", m, Bb)
            dp = proj(d, "P8")
            Bb = Bb + m[None, :, None] * ((dp - d) / 3)[:, None, :]
            K = np.einsum("gji,jk,gkl,g->il", Bb, self.D, Bb, dV)
            return 0.5 * (K + K.T)
        if v in ("full", "red"):
            Bb = Bs
        else:
            Nat = np.einsum("ij,gjk->gik", TN, Bs)
            def proj(rows, space):
                S = np.array([mono(x, SPACES[space]) for x in pts])  # g x m
                W = S.T @ (dV[:, None] * S)
                C = np.linalg.solve(W, S.T @ (dV[:, None] * rows))
                return S @ C
            out = Nat.copy()
            if v == "as":
                for r in (3, 4, 5):
                    out[:, r, :] = proj(Nat[:, r, :], "P8")
            if v in ("asn", "mitc", "mitcn"):
                for r in (0, 1, 2):
                    out[:, r, :] = proj(Nat[:, r, :], "N%d" % r)
                for r, sp in zip((3, 4, 5), ("S01", "S12", "S20")):
                    out[:, r, :] = proj(Nat[:, r, :], "P8" if v == "asn" else sp)
            Bb = np.einsum("ij,gjk->gik", TC, out)
            if v != "mitcn":
                m = np.array([1, 1, 1, 0, 0, 0.0])
                d = np.einsum("i,gik->gk", m, Bb)
                dp = proj(d, "P8")
                Bb = Bb + m[None, :, None] * ((dp - d) / 3)[:, None, :]
        K = np.einsum("gji,jk,gkl,g->il", Bb, self.D, Bb, dV)
        return 0.5 * (K + K.T)
