"""Solve a SOL 101 CHEXA20 deck with a prototype formulation."""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pyNastran.bdf.bdf import read_bdf
from hex20 import Element, shape, gauss


def solve(path, variant):
    model = read_bdf(path, xref=True, debug=None)
    nids = sorted(model.nodes)
    for n in nids:
        if model.nodes[n].cd not in (0, None):
            raise NotImplementedError("CD != 0")
    idx = {n: i for i, n in enumerate(nids)}
    xyz = np.array([model.nodes[n].get_position() for n in nids])
    ndof = 3 * len(nids)
    rows, cols, vals = [], [], []
    for eid, e in model.elements.items():
        if e.type != "CHEXA" or len(e.node_ids) != 20 or None in e.node_ids:
            raise NotImplementedError(e.type)
        prop = e.pid_ref
        mat = prop.mid_ref
        el = Element(xyz[[idx[n] for n in e.node_ids]], mat.e, mat.nu, variant)
        K = el.stiffness()
        dofs = np.array([3 * idx[n] + k for n in e.node_ids for k in range(3)])
        r, c = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(K.ravel())
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ndof, ndof))
    sc = model.subcases[1] if 1 in model.subcases else model.subcases[0]
    F = np.zeros(ndof)
    if "LOAD" in sc:
        lid = sc["LOAD"][0]
        for ld in model.loads.get(lid, []):
            if ld.type == "FORCE":
                f = ld.cid_ref.transform_vector_to_global(ld.scaled_vector)
                F[3 * idx[ld.node] : 3 * idx[ld.node] + 3] += f
            elif ld.type == "GRAV":
                g = ld.cid_ref.transform_vector_to_global(ld.scale * np.asarray(ld.N, float))
                pts, wts = gauss(3)
                for e in model.elements.values():
                    Xe = xyz[[idx[n] for n in e.node_ids]]
                    rho = e.pid_ref.mid_ref.rho
                    for x, w in zip(pts, wts):
                        N, dN = shape(x)
                        dv = w * np.linalg.det(Xe.T @ dN)
                        for a, n in enumerate(e.node_ids):
                            F[3 * idx[n] : 3 * idx[n] + 3] += rho * dv * N[a] * g
            else:
                raise NotImplementedError(ld.type)
    fixed = {}
    sid = sc["SPC"][0]
    for card in model.spcs.get(sid, []):
        if card.type == "SPC1":
            for n in card.nodes:
                for ch in str(card.components):
                    if ch in "123":
                        fixed[3 * idx[n] + int(ch) - 1] = 0.0
        elif card.type == "SPC":
            for n, comp, en in zip(card.nodes, card.components, card.enforced):
                for ch in str(comp):
                    if ch in "123":
                        fixed[3 * idx[n] + int(ch) - 1] = en
        else:
            raise NotImplementedError(card.type)
    u = np.zeros(ndof)
    fd = np.array(sorted(fixed), int)
    u[fd] = [fixed[d] for d in fd]
    free = np.setdiff1d(np.arange(ndof), fd)
    rhs = F[free] - K[free][:, fd] @ u[fd]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return model, idx, u.reshape(-1, 3), K


if __name__ == "__main__":
    path, variant = sys.argv[1], sys.argv[2]
    model, idx, U, _ = solve(path, variant)
    for n in map(int, sys.argv[3].split(",")):
        print(n, U[idx[n]])
