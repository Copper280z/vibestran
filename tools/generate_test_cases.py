"""
generate_test_cases.py
=====================
Generate Nastran SOL 101 / SOL 103 BDF test cases for validating element
formulations and solver robustness.

All cases have analytical reference solutions embedded as comments in the
generated BDF and printed to stdout when the script runs.

Usage
-----
    python generate_test_cases.py          # write all cases, n=2 (medium mesh)
    python generate_test_cases.py --n 4    # finer mesh
    python generate_test_cases.py --cases cantilever_quad cylinder_pressure
    python generate_test_cases.py --list   # show available case names

No external dependencies – pure Python 3.6+.

Output
------
One .bdf file per case in ./generate_cases/
One reference_solutions.txt summarising analytical targets.
"""

import os
import re
import sys
import math
import argparse
from collections import namedtuple
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Minimal BDF writer
# ---------------------------------------------------------------------------


class BDFWriter:
    """Builds a Nastran BDF file as a list of formatted lines."""

    FIELD = 8  # small-field format width

    def __init__(self, title: str, sol: int):
        self.title = title
        self.sol = sol
        self._lines: List[str] = []
        self._next_nid = 1
        self._next_eid = 1
        self._next_pid = 1
        self._next_mid = 1
        self._next_sid = 1  # set/load/bc ids share a namespace here

    # --- id dispensers ---
    def new_nid(self) -> int:
        v = self._next_nid
        self._next_nid += 1
        return v

    def new_eid(self) -> int:
        v = self._next_eid
        self._next_eid += 1
        return v

    def new_pid(self) -> int:
        v = self._next_pid
        self._next_pid += 1
        return v

    def new_mid(self) -> int:
        v = self._next_mid
        self._next_mid += 1
        return v

    def new_sid(self) -> int:
        v = self._next_sid
        self._next_sid += 1
        return v

    # --- formatting helpers ---
    @staticmethod
    def _f(v) -> str:
        """Format a number into an 8-char Nastran small field.

        Nastran distinguishes integers from reals by the presence of a decimal
        point or exponent.  A Python float must always produce one or the other,
        even for whole numbers (e.g. 200.0 must become '200.' not '200').
        """
        if isinstance(v, int):
            return str(v).rjust(8)
        for fmt in ("{:.6g}", "{:.5g}", "{:.4g}", "{:.3g}", "{:.2g}", "{:.1g}"):
            s = fmt.format(v)
            if "." not in s and "e" not in s and "E" not in s:
                s += "."  # ensure Nastran reads it as REAL, not INTEGER
            if len(s) <= 8:
                return s.rjust(8)
        s = "{:.3E}".format(v)
        s = re.sub(r"E([+-])0*(\d+)", lambda m: f"E{m.group(1)}{m.group(2)}", s)
        if len(s) <= 8:
            return s.rjust(8)
        s = "{:.2E}".format(v)
        s = re.sub(r"E([+-])0*(\d+)", lambda m: f"E{m.group(1)}{m.group(2)}", s)
        return s.rjust(8)[:8]

    def _card(self, *fields) -> str:
        """Build one 80-char small-field card from field values."""
        parts = []
        for i, f in enumerate(fields):
            if i == 0:
                parts.append(str(f).ljust(8))
            else:
                parts.append(self._f(f) if not isinstance(f, str) else str(f).rjust(8))
            if i > 0 and (i % 8 == 0):
                parts.append("\n        ")  # continuation
        return "".join(parts)

    def raw(self, line: str):
        self._lines.append(line)

    def comment(self, text: str = ""):
        self._lines.append(f"${' ' + text if text else ''}")

    def blank(self):
        self._lines.append("$")

    # --- executive / case control ---
    def _header(self) -> str:
        lines = []
        lines.append(f"SOL {self.sol}")
        lines.append("CEND")
        lines.append(f"TITLE = {self.title}")
        lines.append("ECHO = NONE")
        if self.sol == 101:
            lines.append("DISPLACEMENT(PRINT,SORT1,REAL) = ALL")
            lines.append("STRESS(PRINT,SORT1,REAL,VONMISES,BILIN) = ALL")
            lines.append("FORCE(SORT1,REAL) = ALL")
            lines.append("SPCFORCE = ALL")
        elif self.sol == 103:
            lines.append("DISPLACEMENT(PRINT,SORT1,REAL) = ALL")
            lines.append("METHOD = 1")
        return "\n".join(lines)

    def _eigrl(self, sid: int, nd: int = 20) -> str:
        # EIGRL  SID  V1  V2  ND
        return f"EIGRL   {sid:8d}{'':8}{'':8}{nd:8d}"

    # --- bulk data cards ---
    def grid(self, nid: int, x: float, y: float, z: float):
        self._lines.append(
            f"GRID    {nid:8d}{'':8}{self._f(x)}{self._f(y)}{self._f(z)}"
        )

    def cquad4(self, eid: int, pid: int, n1, n2, n3, n4):
        self._lines.append(f"CQUAD4  {eid:8d}{pid:8d}{n1:8d}{n2:8d}{n3:8d}{n4:8d}")

    def ctria3(self, eid: int, pid: int, n1, n2, n3):
        self._lines.append(f"CTRIA3  {eid:8d}{pid:8d}{n1:8d}{n2:8d}{n3:8d}")

    def chexa(self, eid: int, pid: int, nodes: List[int]):
        assert len(nodes) == 8
        n = nodes
        line1 = f"CHEXA   {eid:8d}{pid:8d}{n[0]:8d}{n[1]:8d}{n[2]:8d}{n[3]:8d}{n[4]:8d}{n[5]:8d}"
        line2 = f"        {n[6]:8d}{n[7]:8d}"
        self._lines.append(line1)
        self._lines.append(line2)

    def cpenta(self, eid: int, pid: int, nodes: List[int]):
        assert len(nodes) == 6
        n = nodes
        line1 = f"CPENTA  {eid:8d}{pid:8d}{n[0]:8d}{n[1]:8d}{n[2]:8d}{n[3]:8d}{n[4]:8d}{n[5]:8d}"
        self._lines.append(line1)

    def pshell(self, pid: int, mid: int, t: float, mid2: int = None):
        m2 = mid2 if mid2 is not None else mid
        self._lines.append(f"PSHELL  {pid:8d}{mid:8d}{self._f(t)}{m2:8d}")

    def psolid(self, pid: int, mid: int):
        self._lines.append(f"PSOLID  {pid:8d}{mid:8d}")

    def mat1(self, mid: int, E: float, G_or_blank, nu: float, rho: float = 0.0):
        g_str = self._f(G_or_blank) if G_or_blank is not None else "        "
        self._lines.append(
            f"MAT1    {mid:8d}{self._f(E)}{g_str}{self._f(nu)}{self._f(rho)}"
        )

    def spc1(self, sid: int, dofs: str, *nids):
        """SPC1  SID  C  G1  G2 ..."""
        nid_list = list(nids)
        # first card: up to 6 nodes
        chunk = nid_list[:6]
        fields = ["SPC1", sid, dofs] + chunk
        line = f"SPC1    {sid:8d}{str(dofs):>8}"
        for n in chunk:
            line += f"{n:8d}"
        self._lines.append(line)
        # continuation cards
        idx = 6
        while idx < len(nid_list):
            chunk = nid_list[idx : idx + 8]
            line = "        "
            for n in chunk:
                line += f"{n:8d}"
            self._lines.append(line)
            idx += 8

    def spc(self, sid: int, nid: int, dofs: str, disp: float = 0.0):
        self._lines.append(f"SPC     {sid:8d}{nid:8d}{str(dofs):>8}{self._f(disp)}")

    def force(
        self, sid: int, nid: int, cid: int, mag: float, fx: float, fy: float, fz: float
    ):
        self._lines.append(
            f"FORCE   {sid:8d}{nid:8d}{cid:8d}{self._f(mag)}"
            f"{self._f(fx)}{self._f(fy)}{self._f(fz)}"
        )

    def moment(
        self, sid: int, nid: int, cid: int, mag: float, mx: float, my: float, mz: float
    ):
        self._lines.append(
            f"MOMENT  {sid:8d}{nid:8d}{cid:8d}{self._f(mag)}"
            f"{self._f(mx)}{self._f(my)}{self._f(mz)}"
        )

    def pload2(self, sid: int, pressure: float, *eids):
        """Uniform normal pressure on shell elements."""
        eid_list = list(eids)
        chunk = eid_list[:4]
        line = f"PLOAD2  {sid:8d}{self._f(pressure)}"
        for e in chunk:
            line += f"{e:8d}"
        self._lines.append(line)
        idx = 4
        while idx < len(eid_list):
            chunk = eid_list[idx : idx + 8]
            line = "        "
            for e in chunk:
                line += f"{e:8d}"
            self._lines.append(line)
            idx += 8

    def pload4(self, sid: int, eid: int, p: float):
        """Pressure on solid face (simplified: uniform on one face)."""
        self._lines.append(f"PLOAD4  {sid:8d}{eid:8d}{self._f(p)}")

    def rbe2(self, eid: int, gn: int, cm: str, *gmi):
        """RBE2 rigid element."""
        nodes = list(gmi)
        chunk = nodes[:5]
        line = f"RBE2    {eid:8d}{gn:8d}{str(cm):>8}"
        for n in chunk:
            line += f"{n:8d}"
        self._lines.append(line)
        idx = 5
        while idx < len(nodes):
            chunk = nodes[idx : idx + 8]
            line = "        "
            for n in chunk:
                line += f"{n:8d}"
            self._lines.append(line)
            idx += 8

    def rbe3(
        self,
        eid: int,
        refgrid: int,
        refc: str,
        weight: float,
        ci: str,
        nodes: List[int],
    ):
        """RBE3 interpolation element (single weight group)."""
        chunk = nodes[:4]
        line = (
            f"RBE3    {eid:8d}{'':8}{refgrid:8d}{str(refc):>8}"
            f"{self._f(weight)}{str(ci):>8}"
        )
        for n in chunk:
            line += f"{n:8d}"
        self._lines.append(line)
        idx = 4
        while idx < len(nodes):
            chunk = nodes[idx : idx + 8]
            line = "        "
            for n in chunk:
                line += f"{n:8d}"
            self._lines.append(line)
            idx += 8

    def subcase(self, sid: int, label: str, load_sid: int, spc_sid: int):
        """Inject a SUBCASE block into the case-control section."""
        # stored separately, inserted during write
        self._subcases = getattr(self, "_subcases", [])
        self._subcases.append((sid, label, load_sid, spc_sid))

    def write(self, path: str):
        subcases = getattr(self, "_subcases", [])
        with open(path, "w") as f:
            f.write(f"SOL {self.sol}\n")
            f.write("CEND\n")
            f.write(f"TITLE = {self.title}\n")
            f.write("ECHO = NONE\n")
            if self.sol == 101:
                f.write("DISPLACEMENT(PRINT,SORT1,REAL) = ALL\n")
                f.write("STRESS(PRINT,SORT1,REAL,VONMISES,BILIN) = ALL\n")
                f.write("SPCFORCE = ALL\n")
                if subcases:
                    for sid, label, lsid, ssid in subcases:
                        f.write(f"SUBCASE {sid}\n")
                        f.write(f"  LABEL = {label}\n")
                        f.write(f"  LOAD = {lsid}\n")
                        f.write(f"  SPC = {ssid}\n")
                else:
                    f.write("LOAD = 1\n")
                    f.write("SPC = 2\n")
            elif self.sol == 103:
                f.write("DISPLACEMENT(PRINT,SORT1,REAL) = ALL\n")
                f.write("METHOD = 1\n")
                if subcases:
                    for sid, label, lsid, ssid in subcases:
                        f.write(f"SUBCASE {sid}\n")
                        f.write(f"  LABEL = {label}\n")
                        f.write(f"  SPC = {ssid}\n")
                else:
                    f.write("SPC = 2\n")
            f.write("BEGIN BULK\n")
            if self.sol == 103:
                f.write(f"EIGRL   {'1':>7}{'':8}{'':8}{'20':>8}\n")
            for line in self._lines:
                f.write(line + "\n")
            f.write("ENDDATA\n")


# ---------------------------------------------------------------------------
# Analytical reference solutions
# ---------------------------------------------------------------------------


class RefSolutions:
    results = []

    @classmethod
    def add(cls, case: str, quantity: str, value: float, unit: str, formula: str):
        cls.results.append((case, quantity, value, unit, formula))

    @classmethod
    def report(cls) -> str:
        lines = ["=" * 78, "ANALYTICAL REFERENCE SOLUTIONS", "=" * 78]
        cur_case = None
        for case, qty, val, unit, formula in cls.results:
            if case != cur_case:
                lines.append(f"\n--- {case} ---")
                cur_case = case
            lines.append(f"  {qty:<35s} = {val:>14.6g}  [{unit}]")
            lines.append(f"    formula: {formula}")
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def write(cls, path: str):
        with open(path, "w") as f:
            f.write(cls.report())


# ---------------------------------------------------------------------------
# Material / section constants (consistent units: N, mm, MPa)
# ---------------------------------------------------------------------------

E = 200000.0  # MPa  (steel)
NU = 0.3
RHO = 7.85e-9  # tonne/mm^3  (steel, so f in Hz with mm/N/tonne)
G = E / (2 * (1 + NU))


# ===========================================================================
# CASE 1 — Cantilever Beam, Tip Point Load  (CQUAD4)
# ===========================================================================


def case_cantilever_quad(n: int, out_dir: str):
    """
    Cantilever beam modelled with CQUAD4 shell elements.
    Beam: length L, width b, thickness h.
    Tip load P in -Z direction.
    Analytical tip deflection: delta = P*L^3 / (3*E*I)
    Root bending stress:       sigma = M*c/I = P*L*(h/2)/I
    """
    L = 200.0
    b = 20.0
    h = 10.0
    P = 1000.0
    I = b * h**3 / 12.0
    delta_ref = P * L**3 / (3 * E * I)
    sigma_ref = P * L * (h / 2) / I

    name = "cantilever_quad"
    RefSolutions.add(name, "Tip deflection (delta_z)", delta_ref, "mm", "P*L^3/(3*E*I)")
    RefSolutions.add(
        name, "Root bending stress (sigma_x)", sigma_ref, "MPa", "P*L*(h/2)/I"
    )

    bdf = BDFWriter(f"Cantilever CQUAD4 n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Cantilever beam: L={L} b={b} h={h} P={P}")
    bdf.comment(f"ANALYTICAL: delta_tip = {delta_ref:.4f} mm")
    bdf.comment(f"ANALYTICAL: sigma_root = {sigma_ref:.4f} MPa")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.pshell(pid, mid, h)
    bdf.blank()

    nx = 4 * n  # along length
    ny = 2 * n  # along width
    dx = L / nx
    dy = b / ny

    nids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = bdf.new_nid()
            nids[(i, j)] = nid
            bdf.grid(nid, i * dx, j * dy, 0.0)

    eids = []
    for i in range(nx):
        for j in range(ny):
            eid = bdf.new_eid()
            eids.append(eid)
            bdf.cquad4(
                eid,
                pid,
                nids[(i, j)],
                nids[(i + 1, j)],
                nids[(i + 1, j + 1)],
                nids[(i, j + 1)],
            )

    # SPC: clamp all DOF at x=0
    spc_sid = 2
    root_nodes = [nids[(0, j)] for j in range(ny + 1)]
    bdf.spc1(spc_sid, "123456", *root_nodes)

    # Load: tip force at x=L, distributed over tip nodes
    load_sid = 1
    tip_nodes = [nids[(nx, j)] for j in range(ny + 1)]
    p_per_node = P / len(tip_nodes)
    for nid in tip_nodes:
        bdf.force(load_sid, nid, 0, p_per_node, 0.0, 0.0, -1.0)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, delta_ref


# ===========================================================================
# CASE 2 — Cantilever Beam, Tip Point Load  (CHEXA solid)
# ===========================================================================


def case_cantilever_hexa(n: int, out_dir: str):
    """
    Same cantilever, now with CHEXA solid elements.
    Exposes shear locking with coarse meshes.
    """
    L = 200.0
    b = 20.0
    h = 10.0
    P = 1000.0
    I = b * h**3 / 12.0
    delta_ref = P * L**3 / (3 * E * I)
    sigma_ref = P * L * (h / 2) / I

    name = "cantilever_hexa"
    RefSolutions.add(name, "Tip deflection (delta_z)", delta_ref, "mm", "P*L^3/(3*E*I)")
    RefSolutions.add(
        name, "Root bending stress (sigma_x)", sigma_ref, "MPa", "P*L*(h/2)/I"
    )

    bdf = BDFWriter(f"Cantilever CHEXA n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Cantilever beam: L={L} b={b} h={h} P={P}")
    bdf.comment(f"ANALYTICAL: delta_tip = {delta_ref:.4f} mm")
    bdf.comment(f"ANALYTICAL: sigma_root = {sigma_ref:.4f} MPa")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.psolid(pid, mid)
    bdf.blank()

    nx = 4 * n
    ny = n
    nz = n
    dx = L / nx
    dy = b / ny
    dz = h / nz

    nids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                nid = bdf.new_nid()
                nids[(i, j, k)] = nid
                bdf.grid(nid, i * dx, j * dy, k * dz)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                eid = bdf.new_eid()
                # CHEXA connectivity: two faces, bottom then top
                bot = [
                    nids[(i, j, k)],
                    nids[(i + 1, j, k)],
                    nids[(i + 1, j + 1, k)],
                    nids[(i, j + 1, k)],
                ]
                top = [
                    nids[(i, j, k + 1)],
                    nids[(i + 1, j, k + 1)],
                    nids[(i + 1, j + 1, k + 1)],
                    nids[(i, j + 1, k + 1)],
                ]
                bdf.chexa(eid, pid, bot + top)

    spc_sid = 2
    root_nodes = [nids[(0, j, k)] for j in range(ny + 1) for k in range(nz + 1)]
    bdf.spc1(spc_sid, "123456", *root_nodes)

    load_sid = 1
    tip_nodes = [nids[(nx, j, k)] for j in range(ny + 1) for k in range(nz + 1)]
    p_per_node = P / len(tip_nodes)
    for nid in tip_nodes:
        bdf.force(load_sid, nid, 0, p_per_node, 0.0, 0.0, -1.0)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, delta_ref


# ===========================================================================
# CASE 3 — Simply Supported Beam, Uniform Pressure (CQUAD4)
# ===========================================================================


def case_ss_beam_pressure(n: int, out_dir: str):
    """
    Simply supported beam, uniform line load w (force/length) applied as
    pressure on the top face.
    Analytical midspan deflection: delta = 5*w*L^4 / (384*E*I)
    """
    L = 300.0
    b = 20.0
    h = 10.0
    w = 10.0  # N/mm (line load)
    q = w / b  # pressure on top face
    I = b * h**3 / 12.0
    delta_ref = 5 * w * L**4 / (384 * E * I)
    sigma_ref = w * L**2 / (8 * b * h**2 / 6)  # M/(b*h^2/6)

    name = "ss_beam_pressure"
    RefSolutions.add(
        name, "Midspan deflection (delta_z)", delta_ref, "mm", "5*w*L^4/(384*E*I)"
    )
    RefSolutions.add(
        name,
        "Midspan bending stress (sigma_x)",
        sigma_ref,
        "MPa",
        "M_max*c/I, M_max=wL^2/8",
    )

    bdf = BDFWriter(f"SS Beam Pressure CQUAD4 n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Simply supported beam: L={L} b={b} h={h} q={q:.4f} MPa")
    bdf.comment(f"ANALYTICAL: delta_mid = {delta_ref:.4f} mm")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.pshell(pid, mid, h)
    bdf.blank()

    nx = 4 * n
    ny = 2 * n
    dx = L / nx
    dy = b / ny

    nids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = bdf.new_nid()
            nids[(i, j)] = nid
            bdf.grid(nid, i * dx, j * dy, 0.0)

    eids = []
    for i in range(nx):
        for j in range(ny):
            eid = bdf.new_eid()
            eids.append(eid)
            bdf.cquad4(
                eid,
                pid,
                nids[(i, j)],
                nids[(i + 1, j)],
                nids[(i + 1, j + 1)],
                nids[(i, j + 1)],
            )

    # SPC: pin left (1,2,3), roller right (2,3 only, free in x)
    spc_sid = 2
    left_nodes = [nids[(0, j)] for j in range(ny + 1)]
    right_nodes = [nids[(nx, j)] for j in range(ny + 1)]
    bdf.spc1(spc_sid, "123", *left_nodes)
    bdf.spc1(spc_sid, "23", *right_nodes)

    # Uniform pressure load (PLOAD2)
    load_sid = 1
    bdf.pload2(load_sid, -q, *eids)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, delta_ref


# ===========================================================================
# CASE 4 — Simply Supported Square Plate, Uniform Pressure  (CQUAD4)
# ===========================================================================


def case_ss_plate_pressure(n: int, out_dir: str):
    """
    Simply supported square plate a x a, uniform pressure q.
    Navier solution (1st term): w_max = 0.00406*q*a^4/D
      where D = E*t^3/(12*(1-nu^2))
    Valid for thin plate, nu=0.3.
    """
    a = 200.0
    t = 4.0
    q = 0.1  # MPa
    D = E * t**3 / (12 * (1 - NU**2))
    w_ref = 0.00406 * q * a**4 / D

    name = "ss_plate_pressure"
    RefSolutions.add(
        name, "Max deflection w_max", w_ref, "mm", "0.00406*q*a^4/D  (Navier, nu=0.3)"
    )

    bdf = BDFWriter(f"SS Square Plate n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Simply supported square plate: a={a} t={t} q={q}")
    bdf.comment(f"D = {D:.2f} N.mm")
    bdf.comment(f"ANALYTICAL: w_max = {w_ref:.4f} mm")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.pshell(pid, mid, t)
    bdf.blank()

    nn = 4 * n  # elements per side
    ds = a / nn

    nids = {}
    for i in range(nn + 1):
        for j in range(nn + 1):
            nid = bdf.new_nid()
            nids[(i, j)] = nid
            bdf.grid(nid, i * ds, j * ds, 0.0)

    eids = []
    for i in range(nn):
        for j in range(nn):
            eid = bdf.new_eid()
            eids.append(eid)
            bdf.cquad4(
                eid,
                pid,
                nids[(i, j)],
                nids[(i + 1, j)],
                nids[(i + 1, j + 1)],
                nids[(i, j + 1)],
            )

    # SS BCs: simply supported = no out-of-plane displacement on all edges
    # Also prevent rigid-body translation/rotation in-plane
    spc_sid = 2
    edges = []
    for i in range(nn + 1):
        edges += [nids[(i, 0)], nids[(i, nn)], nids[(0, i)], nids[(nn, i)]]
    edges = list(set(edges))
    bdf.spc1(spc_sid, "3", *edges)
    # Corner node: pin in-plane to kill RBM
    bdf.spc1(spc_sid, "12", nids[(0, 0)])

    load_sid = 1
    bdf.pload2(load_sid, -q, *eids)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, w_ref


# ===========================================================================
# CASE 5 — Plate with Central Hole, Uniaxial Tension  (CQUAD4)
# ===========================================================================


def case_plate_hole(n: int, out_dir: str):
    """
    Rectangular plate with central circular hole, far-field tension S.
    Kirsch solution: hoop stress at hole equator = 3*S (stress concentration Kt=3).
    Model quarter-plate with symmetry BCs.
    Hole radius r, plate half-width W.  Valid for r/W << 1.
    """
    W = 100.0
    H = 100.0
    r = 10.0
    t = 4.0
    S = 100.0  # MPa far-field

    name = "plate_hole"
    RefSolutions.add(
        name,
        "Hoop stress at hole equator",
        3.0 * S,
        "MPa",
        "Kt * S = 3*S  (Kirsch, r/W->0)",
    )
    RefSolutions.add(name, "Hoop stress at hole pole", -S, "MPa", "-S  (Kirsch)")

    bdf = BDFWriter(f"Plate with Hole n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Quarter plate with hole: W={W} H={H} r={r} S={S} MPa")
    bdf.comment(f"ANALYTICAL: sigma_hoop(equator) = {3 * S:.1f} MPa  [Kirsch Kt=3]")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.pshell(pid, mid, t)
    bdf.blank()

    # Build a simple mapped quarter mesh
    # Radial lines from r to W in x, r to H in y (quarter model, x>=0, y>=0)
    # Use polar-to-cartesian mapping
    nr = 3 * n  # radial divisions
    nt = 3 * n  # circumferential divisions (0 to pi/2)

    # theta goes from 0 (equator, y=0) to pi/2 (pole, x=0)
    # rho goes from r (hole) to ~W
    rho_out = min(W, H)

    def rho_at(k):
        # geometric grading: fine near hole
        frac = k / nr
        return r + (rho_out - r) * frac**1.5

    nids = {}
    for j in range(nt + 1):
        theta = math.pi / 2 * j / nt
        for k in range(nr + 1):
            rho = rho_at(k)
            x = rho * math.cos(theta)
            y = rho * math.sin(theta)
            nid = bdf.new_nid()
            nids[(j, k)] = nid
            bdf.grid(nid, x, y, 0.0)

    eids = []
    for j in range(nt):
        for k in range(nr):
            eid = bdf.new_eid()
            eids.append(eid)
            bdf.cquad4(
                eid,
                pid,
                nids[(j, k)],
                nids[(j, k + 1)],
                nids[(j + 1, k + 1)],
                nids[(j + 1, k)],
            )

    # Symmetry BCs
    spc_sid = 2
    # y=0 symmetry plane (theta=0): fix UY (dof 2)
    y0_nodes = [nids[(0, k)] for k in range(nr + 1)]
    bdf.spc1(spc_sid, "2", *y0_nodes)
    # x=0 symmetry plane (theta=pi/2): fix UX (dof 1)
    x0_nodes = [nids[(nt, k)] for k in range(nr + 1)]
    bdf.spc1(spc_sid, "1", *x0_nodes)
    # Fix UZ everywhere (plane stress)
    all_nids = list(nids.values())
    bdf.spc1(spc_sid, "3", *all_nids)
    # One more node to fix rotation about Z (prevent in-plane RBM)
    bdf.spc1(spc_sid, "6", nids[(0, 0)])

    # Applied load: uniform tension S on right edge (theta=0, k=nr outer arc)
    load_sid = 1
    # Right edge: x=rho_out*cos(theta), y=rho_out*sin(theta) for theta near 0
    # Use FORCE on outer edge at theta=0 nodes (k=nr, j=0)
    # Apply as nodal forces equivalent to S*t*dy
    right_edge = [(0, nr)]  # only one node at theta=0, k=nr
    # Better: distribute along outer boundary at y=0 face
    # Load the right edge (j=0, k from 0 to nr outer) doesn't quite work for
    # arbitrary mesh; instead apply far-field force resultant to outer nodes at j=0
    outer_nodes_j0 = [nids[(0, nr)]]  # corner of outer edge at equator
    # For simplicity apply the full resultant at the outer boundary nodes at theta=0
    # Resultant = S * t * H  on half-model, distributed over right-edge outer node
    # This is a thin-strip model so we apply force to the strip end
    force_total = S * t * H  # total force on quarter model right edge (y=0 to H)
    # Nodes at right boundary: j=0, all k... but outer boundary at y=0 is just j=0
    # The "right" edge in the plate problem is x=W, but in this polar mesh
    # the outer boundary is a circular arc; apply equivalent load to x-direction
    # on the outer arc nodes at theta=0 (j=0 row)
    outer_arc_nodes = [nids[(0, k)] for k in range(1, nr + 1)]

    # Compute y-extent for each outer arc node (half the distance to neighbors)
    def arc_y_extent(k):
        y_cur = rho_at(k) * math.sin(0)  # theta=0, so y=0 for all j=0...
        # Actually at theta=0 all nodes have y=0, so this reduces to 1D:
        # distribute by rho spacing
        rho_lo = rho_at(k - 0.5) if k > 0 else rho_at(0)
        rho_hi = rho_at(k + 0.5) if k < nr else rho_at(nr)
        return rho_hi - rho_lo

    rho_extents = [
        rho_at(k + 0.5) - rho_at(k - 0.5)
        if 0 < k < nr
        else (rho_at(0.5) - rho_at(0) if k == 0 else rho_at(nr) - rho_at(nr - 0.5))
        for k in range(1, nr + 1)
    ]
    rho_total = sum(rho_extents)
    for idx, nid in enumerate(outer_arc_nodes):
        fx = S * t * rho_extents[idx]
        bdf.force(load_sid, nid, 0, 1.0, fx, 0.0, 0.0)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name


# ===========================================================================
# CASE 6 — Thick-Walled Cylinder, Internal Pressure (CHEXA)
# ===========================================================================


def case_cylinder_pressure(n: int, out_dir: str):
    """
    Thick-walled cylinder: inner radius Ri, outer radius Ro, internal pressure p.
    Lamé solution:
      sigma_r(r)   = p*Ri^2/(Ro^2-Ri^2) * (1 - Ro^2/r^2)
      sigma_th(r)  = p*Ri^2/(Ro^2-Ri^2) * (1 + Ro^2/r^2)
    Model as a 15-degree wedge sector (saves elements) with symmetry BCs.
    """
    Ri = 50.0
    Ro = 100.0
    p_int = 100.0  # MPa
    Lz = 20.0  # axial length (short, plane-strain-like)
    c = Ri**2 / (Ro**2 - Ri**2)
    sig_th_inner = p_int * c * (1 + Ro**2 / Ri**2)  # max hoop
    sig_r_inner = -p_int  # equals -p at inner surface (should be)
    sig_th_outer = p_int * c * 2  # hoop at outer surface
    u_inner = (p_int * Ri / E) * (
        Ri**2 / (Ro**2 - Ri**2) * (1 + NU) + (1 - NU) * Ro**2 / (Ro**2 - Ri**2)
    )
    # Simplified radial disp at inner wall (plane stress):
    u_inner_ps = (
        (p_int * Ri / E) * ((1 - NU) * Ri**2 + (1 + NU) * Ro**2) / (Ro**2 - Ri**2)
    )

    name = "cylinder_pressure"
    RefSolutions.add(
        name,
        "Hoop stress at inner wall",
        sig_th_inner,
        "MPa",
        "p*Ri^2/(Ro^2-Ri^2)*(1+Ro^2/Ri^2)",
    )
    RefSolutions.add(
        name, "Radial stress at inner wall", sig_r_inner, "MPa", "-p_int  (BC)"
    )
    RefSolutions.add(
        name, "Hoop stress at outer wall", sig_th_outer, "MPa", "p*Ri^2/(Ro^2-Ri^2)*2"
    )
    RefSolutions.add(
        name,
        "Radial disp at inner wall (plane stress)",
        u_inner_ps,
        "mm",
        "p*Ri/E*((1-nu)*Ri^2+(1+nu)*Ro^2)/(Ro^2-Ri^2)",
    )

    bdf = BDFWriter(f"Thick-Walled Cylinder n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Thick cylinder: Ri={Ri} Ro={Ro} p={p_int} Lz={Lz}")
    bdf.comment(f"ANALYTICAL: sigma_hoop(inner) = {sig_th_inner:.4f} MPa")
    bdf.comment(f"ANALYTICAL: sigma_hoop(outer) = {sig_th_outer:.4f} MPa")
    bdf.comment(f"ANALYTICAL: u_r(inner) = {u_inner_ps:.6f} mm  (plane stress)")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.psolid(pid, mid)
    bdf.blank()

    # Wedge: theta from 0 to 15 degrees, r from Ri to Ro, z from 0 to Lz
    theta_span = math.radians(15.0)
    nr = 3 * n
    ntheta = 2 * n
    nz = n

    def node_xyz(ir, it, iz):
        frac_r = ir / nr
        r = Ri + (Ro - Ri) * frac_r
        theta = theta_span * it / ntheta
        z = Lz * iz / nz
        return r * math.cos(theta), r * math.sin(theta), z

    nids = {}
    for iz in range(nz + 1):
        for it in range(ntheta + 1):
            for ir in range(nr + 1):
                nid = bdf.new_nid()
                nids[(ir, it, iz)] = nid
                x, y, z = node_xyz(ir, it, iz)
                bdf.grid(nid, x, y, z)

    inner_face_eids = []
    for iz in range(nz):
        for it in range(ntheta):
            for ir in range(nr):
                eid = bdf.new_eid()
                # Bottom face (iz), then top face (iz+1) — CHEXA ordering
                bot = [
                    nids[(ir, it, iz)],
                    nids[(ir + 1, it, iz)],
                    nids[(ir + 1, it + 1, iz)],
                    nids[(ir, it + 1, iz)],
                ]
                top = [
                    nids[(ir, it, iz + 1)],
                    nids[(ir + 1, it, iz + 1)],
                    nids[(ir + 1, it + 1, iz + 1)],
                    nids[(ir, it + 1, iz + 1)],
                ]
                bdf.chexa(eid, pid, bot + top)
                if ir == 0:
                    inner_face_eids.append(eid)

    # BCs
    spc_sid = 2
    # Symmetry: theta=0 plane -> fix tangential (UY = dof 2 in global, but need
    # the direction perpendicular to the face).  For simplicity fix the component
    # in the direction normal to each symmetry plane.
    # theta=0 face (it=0): normal is +Y, fix UY
    theta0_nodes = [nids[(ir, 0, iz)] for ir in range(nr + 1) for iz in range(nz + 1)]
    bdf.spc1(spc_sid, "2", *theta0_nodes)
    # theta=15 face (it=ntheta): normal perpendicular to that face
    # For a 15 deg wedge this is approximately -sin(15)*X + cos(15)*Y
    # In Nastran we'd need a local coord sys; for simplicity fix equivalent
    # by constraining the in-plane tangential component with SPC.
    # Here we just prevent rigid body rotation about Z by fixing the node
    # at the outer radius of the theta=max face in the tangential direction.
    thetatop_nodes = [
        nids[(ir, ntheta, iz)] for ir in range(nr + 1) for iz in range(nz + 1)
    ]
    bdf.spc1(spc_sid, "2", *thetatop_nodes)  # approximate for small wedge angle
    # Axial: fix UZ at z=0 to prevent rigid body translation
    z0_nodes = [nids[(ir, it, 0)] for ir in range(nr + 1) for it in range(ntheta + 1)]
    bdf.spc1(spc_sid, "3", *z0_nodes)
    # Fix UX at a node to prevent rigid translation in X
    bdf.spc1(spc_sid, "1", nids[(nr, 0, 0)])

    # Internal pressure on inner face (PLOAD4 on each CHEXA inner face)
    # PLOAD4: pressure on element face
    load_sid = 1
    for eid in inner_face_eids:
        bdf.pload4(load_sid, eid, p_int)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, sig_th_inner


# ===========================================================================
# CASE 7 — Hollow Cylinder in Torsion (CHEXA)
# ===========================================================================


def case_cylinder_torsion(n: int, out_dir: str):
    """
    Hollow cylinder in pure torsion: inner radius Ri, outer radius Ro, length L.
    Analytical shear stress: tau = T*r/J
      J = pi/2*(Ro^4 - Ri^4)
    Applied as tangential nodal forces at both ends (couple).
    """
    Ri = 30.0
    Ro = 50.0
    L = 200.0
    T = 1e6  # N.mm
    J = math.pi / 2 * (Ro**4 - Ri**4)
    tau_inner = T * Ri / J
    tau_outer = T * Ro / J
    phi_ref = T * L / (G * J)  # twist angle in radians

    name = "cylinder_torsion"
    RefSolutions.add(name, "Shear stress at inner wall", tau_inner, "MPa", "T*Ri/J")
    RefSolutions.add(name, "Shear stress at outer wall", tau_outer, "MPa", "T*Ro/J")
    RefSolutions.add(name, "Total twist angle phi", phi_ref, "rad", "T*L/(G*J)")

    bdf = BDFWriter(f"Cylinder Torsion n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Hollow cylinder torsion: Ri={Ri} Ro={Ro} L={L} T={T}")
    bdf.comment(f"J = {J:.2f} mm^4")
    bdf.comment(f"ANALYTICAL: tau_outer = {tau_outer:.4f} MPa")
    bdf.comment(f"ANALYTICAL: phi = {phi_ref:.6f} rad")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.psolid(pid, mid)
    bdf.blank()

    # Full 360 model divided into ntheta sectors
    nr = 2 * n
    ntheta = 8 * n
    nz = 4 * n
    dtheta = 2 * math.pi / ntheta

    nids = {}
    for iz in range(nz + 1):
        for it in range(ntheta):  # periodic: it=ntheta wraps to it=0
            for ir in range(nr + 1):
                nid = bdf.new_nid()
                nids[(ir, it, iz)] = nid
                r = Ri + (Ro - Ri) * ir / nr
                theta = dtheta * it
                z = L * iz / nz
                bdf.grid(nid, r * math.cos(theta), r * math.sin(theta), z)

    def get_nid(ir, it, iz):
        return nids[(ir, it % ntheta, iz)]

    for iz in range(nz):
        for it in range(ntheta):
            for ir in range(nr):
                eid = bdf.new_eid()
                bot = [
                    get_nid(ir, it, iz),
                    get_nid(ir + 1, it, iz),
                    get_nid(ir + 1, it + 1, iz),
                    get_nid(ir, it + 1, iz),
                ]
                top = [
                    get_nid(ir, it, iz + 1),
                    get_nid(ir + 1, it, iz + 1),
                    get_nid(ir + 1, it + 1, iz + 1),
                    get_nid(ir, it + 1, iz + 1),
                ]
                bdf.chexa(eid, pid, bot + top)

    # BCs: fix z=0 face (all DOF)
    spc_sid = 2
    z0_nodes = [nids[(ir, it, 0)] for ir in range(nr + 1) for it in range(ntheta)]
    bdf.spc1(spc_sid, "123456", *z0_nodes)

    # Apply torque at z=L: tangential forces on outer ring nodes
    load_sid = 1
    ztop_nodes = [(ir, it) for ir in range(nr + 1) for it in range(ntheta)]
    # Only outer ring (ir=nr) for efficiency, equally spaced
    outer_top = [nids[(nr, it, nz)] for it in range(ntheta)]
    F_tang = T / (Ro * ntheta)  # tangential force per node
    for it, nid in enumerate(outer_top):
        theta = dtheta * it + dtheta / 2
        # tangential direction: (-sin(theta), cos(theta), 0)
        fx = -math.sin(dtheta * it) * F_tang
        fy = math.cos(dtheta * it) * F_tang
        bdf.force(load_sid, nid, 0, 1.0, fx, fy, 0.0)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, tau_outer


# ===========================================================================
# CASE 8 — RBE2 Cantilever: Offset Tip Load via Rigid Spider
# ===========================================================================


def case_rbe2_cantilever(n: int, out_dir: str):
    """
    Cantilever beam with tip load applied via RBE2 spider.
    The RBE2 reference node (independent) sits at beam centerline;
    dependent nodes are at the tip cross-section.
    Deflection must match the direct-load cantilever.
    """
    L = 200.0
    b = 20.0
    h = 10.0
    P = 1000.0
    I = b * h**3 / 12.0
    delta_ref = P * L**3 / (3 * E * I)

    name = "rbe2_cantilever"
    RefSolutions.add(
        name,
        "Tip deflection (delta_z) via RBE2",
        delta_ref,
        "mm",
        "P*L^3/(3*E*I)  - same as direct load",
    )

    bdf = BDFWriter(f"RBE2 Cantilever n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Cantilever with RBE2 tip spider: L={L} b={b} h={h} P={P}")
    bdf.comment(f"ANALYTICAL: delta_tip = {delta_ref:.4f} mm")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.pshell(pid, mid, h)
    bdf.blank()

    nx = 4 * n
    ny = 2 * n
    dx = L / nx
    dy = b / ny

    nids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = bdf.new_nid()
            nids[(i, j)] = nid
            bdf.grid(nid, i * dx, j * dy, 0.0)

    for i in range(nx):
        for j in range(ny):
            eid = bdf.new_eid()
            bdf.cquad4(
                eid,
                pid,
                nids[(i, j)],
                nids[(i + 1, j)],
                nids[(i + 1, j + 1)],
                nids[(i, j + 1)],
            )

    # RBE2 master node at tip centerline (offset from mesh)
    master_nid = bdf.new_nid()
    bdf.grid(master_nid, L, b / 2, 0.0)

    tip_nodes = [nids[(nx, j)] for j in range(ny + 1)]
    rbe2_eid = bdf.new_eid()
    bdf.rbe2(rbe2_eid, master_nid, "123456", *tip_nodes)

    # SPC: clamp root
    spc_sid = 2
    root_nodes = [nids[(0, j)] for j in range(ny + 1)]
    bdf.spc1(spc_sid, "123456", *root_nodes)

    # Load on master node
    load_sid = 1
    bdf.force(load_sid, master_nid, 0, P, 0.0, 0.0, -1.0)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, delta_ref


# ===========================================================================
# CASE 9 — RBE3 Load Distribution Check
# ===========================================================================


def case_rbe3_distribution(n: int, out_dir: str):
    """
    Square plate, one edge clamped, opposite edge has RBE3 reference node.
    Point load applied to RBE3 reference node; verify it distributes as
    weighted average to the edge nodes (statically equivalent to direct load).
    Compare tip deflection to cantilever plate analytical approximation.
    """
    L = 200.0
    b = 100.0
    h = 5.0
    P = 5000.0
    # Approximate cantilever plate: use beam formula with I = b*h^3/12
    I = b * h**3 / 12.0
    delta_ref = P * L**3 / (3 * E * I)

    name = "rbe3_distribution"
    RefSolutions.add(
        name,
        "Tip deflection (delta_z) via RBE3",
        delta_ref,
        "mm",
        "P*L^3/(3*E*I) approx (plate cantilever)",
    )

    bdf = BDFWriter(f"RBE3 Distribution n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Plate with RBE3 tip load: L={L} b={b} h={h} P={P}")
    bdf.comment(f"ANALYTICAL (approx): delta_tip = {delta_ref:.4f} mm")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.pshell(pid, mid, h)
    bdf.blank()

    nx = 4 * n
    ny = 2 * n
    dx = L / nx
    dy = b / ny

    nids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = bdf.new_nid()
            nids[(i, j)] = nid
            bdf.grid(nid, i * dx, j * dy, 0.0)

    for i in range(nx):
        for j in range(ny):
            eid = bdf.new_eid()
            bdf.cquad4(
                eid,
                pid,
                nids[(i, j)],
                nids[(i + 1, j)],
                nids[(i + 1, j + 1)],
                nids[(i, j + 1)],
            )

    # RBE3 reference node at tip center
    ref_nid = bdf.new_nid()
    bdf.grid(ref_nid, L, b / 2, 0.0)

    tip_nodes = [nids[(nx, j)] for j in range(ny + 1)]
    rbe3_eid = bdf.new_eid()
    bdf.rbe3(rbe3_eid, ref_nid, "123456", 1.0, "123456", tip_nodes)

    spc_sid = 2
    root_nodes = [nids[(0, j)] for j in range(ny + 1)]
    bdf.spc1(spc_sid, "123456", *root_nodes)

    load_sid = 1
    bdf.force(load_sid, ref_nid, 0, P, 0.0, 0.0, -1.0)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, delta_ref


# ===========================================================================
# CASE 10 — Free-Free Beam, Natural Frequencies (SOL 103, CQUAD4)
# ===========================================================================


def case_ff_beam_modes(n: int, out_dir: str):
    """
    Free-free beam modelled with CQUAD4 shells.
    Euler-Bernoulli natural frequencies:
      f_n = (beta_n*L)^2 / (2*pi) * sqrt(E*I / (rho*A*L^4))
    Free-free coefficients (beta_n*L): 4.730, 7.853, 10.996, 14.137
    First 6 eigenvalues should be near-zero (rigid body modes).
    """
    L = 300.0
    b = 20.0
    h = 10.0
    A = b * h
    I = b * h**3 / 12.0
    beta_L = [4.7300, 7.8532, 10.9956, 14.1372]
    freqs = []
    for bl in beta_L:
        f = bl**2 / (2 * math.pi) * math.sqrt(E * I / (RHO * A * L**4))
        freqs.append(f)

    name = "ff_beam_modes"
    for i, f in enumerate(freqs):
        RefSolutions.add(
            name,
            f"Mode {i + 7} frequency (bending {i + 1})",
            f,
            "Hz",
            f"(beta_n*L)^2/(2pi)*sqrt(EI/rhoA/L^4), beta_n*L={beta_L[i]:.4f}",
        )

    bdf = BDFWriter(f"Free-Free Beam SOL103 n={n}", sol=103)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Free-free beam: L={L} b={b} h={h}")
    for i, f in enumerate(freqs):
        bdf.comment(f"ANALYTICAL: mode {i + 7} (bending {i + 1}) = {f:.4f} Hz")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.pshell(pid, mid, h)
    bdf.blank()

    nx = 8 * n
    ny = 2 * n
    dx = L / nx
    dy = b / ny

    nids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = bdf.new_nid()
            nids[(i, j)] = nid
            bdf.grid(nid, i * dx, j * dy, 0.0)

    for i in range(nx):
        for j in range(ny):
            eid = bdf.new_eid()
            bdf.cquad4(
                eid,
                pid,
                nids[(i, j)],
                nids[(i + 1, j)],
                nids[(i + 1, j + 1)],
                nids[(i, j + 1)],
            )

    # SPC: just prevent numerical zero-energy modes in z-translation for the
    # free-free run; keep truly free so 6 RBMs appear.
    # Constrain one node to prevent translational RBM in Z (out of plane)
    # but leave all bending DOF free.
    spc_sid = 2
    mid_node = nids[(nx // 2, ny // 2)]
    bdf.spc1(spc_sid, "3", mid_node)
    bdf.spc1(spc_sid, "16", nids[(0, 0)])  # stop in-plane rigid body

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, freqs[0]


# ===========================================================================
# CASE 11 — Simply Supported Beam, Natural Frequencies (SOL 103, CQUAD4)
# ===========================================================================


def case_ss_beam_modes(n: int, out_dir: str):
    """
    Simply supported beam, first 4 bending modes.
    f_n = n^2 * pi / (2*L^2) * sqrt(E*I / (rho*A))
    """
    L = 300.0
    b = 20.0
    h = 10.0
    A = b * h
    I = b * h**3 / 12.0
    freqs = []
    for mode_n in range(1, 5):
        f = mode_n**2 * math.pi / (2 * L**2) * math.sqrt(E * I / (RHO * A))
        freqs.append(f)

    name = "ss_beam_modes"
    for i, f in enumerate(freqs):
        RefSolutions.add(
            name,
            f"Mode {i + 1} frequency",
            f,
            "Hz",
            f"n^2*pi/(2L^2)*sqrt(EI/rhoA), n={i + 1}",
        )

    bdf = BDFWriter(f"SS Beam SOL103 n={n}", sol=103)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Simply supported beam modes: L={L} b={b} h={h}")
    for i, f in enumerate(freqs):
        bdf.comment(f"ANALYTICAL: mode {i + 1} = {f:.4f} Hz")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.pshell(pid, mid, h)
    bdf.blank()

    nx = 8 * n
    ny = 2 * n
    dx = L / nx
    dy = b / ny

    nids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = bdf.new_nid()
            nids[(i, j)] = nid
            bdf.grid(nid, i * dx, j * dy, 0.0)

    for i in range(nx):
        for j in range(ny):
            eid = bdf.new_eid()
            bdf.cquad4(
                eid,
                pid,
                nids[(i, j)],
                nids[(i + 1, j)],
                nids[(i + 1, j + 1)],
                nids[(i, j + 1)],
            )

    spc_sid = 2
    # Simply supported: UZ=0 at both ends
    left = [nids[(0, j)] for j in range(ny + 1)]
    right = [nids[(nx, j)] for j in range(ny + 1)]
    bdf.spc1(spc_sid, "3", *left)
    bdf.spc1(spc_sid, "3", *right)
    # Suppress in-plane rigid body modes
    bdf.spc1(spc_sid, "12", nids[(0, 0)])
    bdf.spc1(spc_sid, "2", nids[(nx, 0)])

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, freqs[0]


# ===========================================================================
# CASE 12 — Clamped Square Plate, Natural Frequencies (SOL 103, CQUAD4)
# ===========================================================================


def case_clamped_plate_modes(n: int, out_dir: str):
    """
    Fully clamped square plate, fundamental frequency.
    Leissa (1969): f_11 = lambda_11^2 / (2*pi*a^2) * sqrt(D / (rho*t))
    lambda_11 = 35.99  for CCCC plate, square.
    """
    a = 200.0
    t = 4.0
    D = E * t**3 / (12 * (1 - NU**2))
    lambda11 = 35.99
    f_ref = lambda11 / (2 * math.pi * a**2) * math.sqrt(D / (RHO * t))

    name = "clamped_plate_modes"
    RefSolutions.add(
        name,
        "Fundamental frequency f_11",
        f_ref,
        "Hz",
        "lambda_11^2/(2*pi*a^2)*sqrt(D/rho/t), lambda_11=35.99 [Leissa]",
    )

    bdf = BDFWriter(f"Clamped Plate SOL103 n={n}", sol=103)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"CCCC square plate modes: a={a} t={t}")
    bdf.comment(f"D = {D:.2f} N.mm")
    bdf.comment(f"ANALYTICAL: f_11 = {f_ref:.4f} Hz  (Leissa, lambda_11=35.99)")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.pshell(pid, mid, t)
    bdf.blank()

    nn = 4 * n
    ds = a / nn

    nids = {}
    for i in range(nn + 1):
        for j in range(nn + 1):
            nid = bdf.new_nid()
            nids[(i, j)] = nid
            bdf.grid(nid, i * ds, j * ds, 0.0)

    for i in range(nn):
        for j in range(nn):
            eid = bdf.new_eid()
            bdf.cquad4(
                eid,
                pid,
                nids[(i, j)],
                nids[(i + 1, j)],
                nids[(i + 1, j + 1)],
                nids[(i, j + 1)],
            )

    spc_sid = 2
    edge_nodes = []
    for i in range(nn + 1):
        edge_nodes += [nids[(i, 0)], nids[(i, nn)], nids[(0, i)], nids[(nn, i)]]
    edge_nodes = list(set(edge_nodes))
    bdf.spc1(spc_sid, "123456", *edge_nodes)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, f_ref


# ===========================================================================
# CASE 13 — Pressurized Sphere Octant (CHEXA, SOL 101)
# ===========================================================================


def case_sphere_pressure(n: int, out_dir: str):
    """
    Thick-walled sphere: inner radius Ri, outer radius Ro, internal pressure p.
    Lamé (spherical):
      sigma_r(r)   = A - 2B/r^3
      sigma_th(r)  = A + B/r^3
      where A = p*Ri^3/(Ro^3-Ri^3), B = p*Ri^3*Ro^3/(2*(Ro^3-Ri^3))
    Octant model with three symmetry planes.

    Mesh parameterisation
    ---------------------
    Spherical coordinates with:
      r     in [Ri, Ro]           -- radial (ir index, G1-G4/G5-G8 axis)
      theta in [0, pi/2]          -- azimuth  (it index, XZ->YZ)
      phi   in [phi_start, pi/2]  -- polar from z-axis (ip index)

    phi starts one full phi-step away from the z-axis pole
    (phi_start = pi/2/nphi) to avoid the degenerate case where
    sin(phi)=0 collapses all nodes on the z-axis to the same point.
    The ip=0 face (near-pole cap) carries the z-axis symmetry BC (UX=UY=0).

    CHEXA winding: radial direction is the element "extrusion" axis.
      G1-G4: face at ir   (inner), in order (it,ip),(it,ip+1),(it+1,ip+1),(it+1,ip)
      G5-G8: face at ir+1 (outer), same order
    This gives positive Jacobians throughout.

    Symmetry BCs:
      it=0      (theta=0, y=0, XZ plane):        fix UY (dof 2)
      it=ntheta (theta=pi/2, x=0, YZ plane):     fix UX (dof 1)
      ip=nphi   (phi=pi/2, z=0, equatorial plane): fix UZ (dof 3)
      ip=0      (near-pole cap):                  fix UX and UY (dof 12)
    """
    Ri = 50.0
    Ro = 80.0
    p_int = 50.0
    A_lame = p_int * Ri**3 / (Ro**3 - Ri**3)
    B_lame = p_int * Ri**3 * Ro**3 / (2 * (Ro**3 - Ri**3))
    sig_th_inner = A_lame + B_lame / Ri**3
    sig_r_inner = A_lame - 2 * B_lame / Ri**3  # should equal -p_int
    sig_th_outer = A_lame + B_lame / Ro**3

    name = "sphere_pressure"
    RefSolutions.add(
        name,
        "Hoop stress at inner surface",
        sig_th_inner,
        "MPa",
        "A + B/Ri^3  (Lamé spherical)",
    )
    RefSolutions.add(
        name,
        "Hoop stress at outer surface",
        sig_th_outer,
        "MPa",
        "A + B/Ro^3  (Lamé spherical)",
    )
    RefSolutions.add(
        name,
        "Radial stress at inner surface",
        sig_r_inner,
        "MPa",
        "A - 2B/Ri^3 = -p_int  (BC check)",
    )

    bdf = BDFWriter(f"Pressurized Sphere Octant n={n}", sol=101)
    pid = bdf.new_pid()
    mid = bdf.new_mid()
    bdf.comment(f"Sphere octant: Ri={Ri} Ro={Ro} p={p_int}")
    bdf.comment(f"ANALYTICAL: sigma_hoop(inner) = {sig_th_inner:.4f} MPa")
    bdf.comment(f"ANALYTICAL: sigma_hoop(outer) = {sig_th_outer:.4f} MPa")
    bdf.comment(f"Mesh: radial extrusion CHEXA, phi avoids z-axis pole")
    bdf.blank()

    bdf.mat1(mid, E, None, NU, RHO)
    bdf.psolid(pid, mid)
    bdf.blank()

    nr = 2 * n  # radial layers
    ntheta = 3 * n  # azimuth divisions  (theta: 0 -> pi/2)
    nphi = 3 * n  # polar divisions    (phi: phi_start -> pi/2)

    # phi range: start one step from the pole to avoid sin(phi)=0 degeneracy.
    # phi_start = pi/2/nphi  (one full grid spacing from pole)
    phi_start = math.pi / 2 / nphi
    phi_end = math.pi / 2  # equatorial plane (z=0)

    def sph2cart(r, theta, phi):
        return (
            r * math.sin(phi) * math.cos(theta),
            r * math.sin(phi) * math.sin(theta),
            r * math.cos(phi),
        )

    nids = {}
    for ir in range(nr + 1):
        r = Ri + (Ro - Ri) * ir / nr
        for it in range(ntheta + 1):
            theta = math.pi / 2 * it / ntheta
            for ip in range(nphi + 1):
                phi = phi_start + (phi_end - phi_start) * ip / nphi
                nid = bdf.new_nid()
                nids[(ir, it, ip)] = nid
                x, y, z = sph2cart(r, theta, phi)
                bdf.grid(nid, x, y, z)

    # CHEXA elements: radial direction is the extrusion axis (G1-G4 at ir, G5-G8 at ir+1)
    # Within each constant-r face, wind (it,ip) -> (it,ip+1) -> (it+1,ip+1) -> (it+1,ip)
    # This ordering gives positive Jacobians for the sphere geometry.
    inner_eids = []
    for ir in range(nr):
        for it in range(ntheta):
            for ip in range(nphi):
                eid = bdf.new_eid()
                bdf.chexa(
                    eid,
                    pid,
                    [
                        nids[(ir, it, ip)],  # G1
                        nids[(ir, it, ip + 1)],  # G2
                        nids[(ir, it + 1, ip + 1)],  # G3
                        nids[(ir, it + 1, ip)],  # G4
                        nids[(ir + 1, it, ip)],  # G5
                        nids[(ir + 1, it, ip + 1)],  # G6
                        nids[(ir + 1, it + 1, ip + 1)],  # G7
                        nids[(ir + 1, it + 1, ip)],  # G8
                    ],
                )
                if ir == 0:
                    inner_eids.append(eid)

    spc_sid = 2

    # theta=0 face (it=0): XZ plane, y=0 -> fix UY
    bdf.spc1(
        spc_sid,
        "2",
        *[nids[(ir, 0, ip)] for ir in range(nr + 1) for ip in range(nphi + 1)],
    )

    # theta=pi/2 face (it=ntheta): YZ plane, x=0 -> fix UX
    bdf.spc1(
        spc_sid,
        "1",
        *[nids[(ir, ntheta, ip)] for ir in range(nr + 1) for ip in range(nphi + 1)],
    )

    # phi=pi/2 face (ip=nphi): equatorial plane, z=0 -> fix UZ
    bdf.spc1(
        spc_sid,
        "3",
        *[nids[(ir, it, nphi)] for ir in range(nr + 1) for it in range(ntheta + 1)],
    )

    # Near-pole cap face (ip=0): closest face to z-axis (phi=phi_start).
    # By symmetry of the full sphere about the z-axis, UX=UY=0 on this face.
    bdf.spc1(
        spc_sid,
        "12",
        *[nids[(ir, it, 0)] for ir in range(nr + 1) for it in range(ntheta + 1)],
    )

    # Internal pressure on inner radial face (ir=0)
    load_sid = 1
    for eid in inner_eids:
        bdf.pload4(load_sid, eid, p_int)

    bdf.write(os.path.join(out_dir, f"{name}_n{n}.bdf"))
    return name, sig_th_inner


# ===========================================================================
# Registry and main
# ===========================================================================

CASES = {
    "cantilever_quad": case_cantilever_quad,
    "cantilever_hexa": case_cantilever_hexa,
    "ss_beam_pressure": case_ss_beam_pressure,
    "ss_plate_pressure": case_ss_plate_pressure,
    "plate_hole": case_plate_hole,
    "cylinder_pressure": case_cylinder_pressure,
    "cylinder_torsion": case_cylinder_torsion,
    "rbe2_cantilever": case_rbe2_cantilever,
    "rbe3_distribution": case_rbe3_distribution,
    "ff_beam_modes": case_ff_beam_modes,
    "ss_beam_modes": case_ss_beam_modes,
    "clamped_plate_modes": case_clamped_plate_modes,
    "sphere_pressure": case_sphere_pressure,
}


def main():
    parser = argparse.ArgumentParser(description="Generate Nastran BDF test cases")
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Mesh refinement factor (1=coarse, 4=fine, default=2)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Subset of case names to generate (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available case names and exit"
    )
    parser.add_argument(
        "--out",
        default="test_cases",
        help="Output directory (default: test_cases/)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for name in CASES:
            print(f"  {name}")
        return

    selected = args.cases if args.cases else list(CASES.keys())
    unknown = [c for c in selected if c not in CASES]
    if unknown:
        print(f"Unknown cases: {unknown}")
        print("Run with --list to see available cases.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print(f"Generating {len(selected)} case(s) with n={args.n} into '{args.out}/'")
    print()

    for name in selected:
        fn = CASES[name]
        try:
            fn(args.n, args.out)
            bdf_path = os.path.join(args.out, f"{name}_n{args.n}.bdf")
            print(f"  [OK]  {bdf_path}")
        except Exception as exc:
            print(f"  [ERR] {name}: {exc}")
            raise

    # Write reference solutions
    ref_path = os.path.join(args.out, "reference_solutions.txt")
    RefSolutions.write(ref_path)
    print()
    print(f"Reference solutions written to: {ref_path}")
    print()
    print(RefSolutions.report())


if __name__ == "__main__":
    main()
