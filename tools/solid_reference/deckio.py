"""Deck writing, solver runs and F06 parsing shared by the solid reference
generators.  Decks use fixed-format fields (large field for real values)
so that both vibestran and MYSTRAN read them."""

import os
import re
import subprocess
import time

VIBESTRAN = os.environ.get("VIBESTRAN", "/Users/bob/Documents/vibe-tran/build/vibestran")
MYSTRAN = os.environ.get("MYSTRAN", "/Users/bob/Documents/vibe-tran/MYSTRAN/Binaries/mystran")


def f16(v):
    """A real in a 16-character field."""
    s = f"{v:16.9E}"
    assert len(s) == 16, s
    return s


def f8(v):
    s = str(v)
    assert len(s) <= 8, s
    return f"{s:<8s}"


class Deck:
    def __init__(self, title, sol=101, case_extra=()):
        self.case = [f"SOL {sol}", "CEND", f"TITLE = {title}", "ECHO = NONE",
                     "DISPLACEMENT = ALL", *case_extra]
        self.bulk = []

    def line(self, s):
        self.bulk.append(s)

    def grid(self, gid, x, y, z):
        self.bulk.append(f"{'GRID*':<8s}{gid:<16d}{'':16s}{f16(x)}{f16(y)}*")
        self.bulk.append(f"{'*':<8s}{f16(z)}")

    def force(self, sid, gid, fx, fy, fz):
        self.bulk.append(f"{'FORCE*':<8s}{sid:<16d}{gid:<16d}{0:<16d}{f16(1.0)}*")
        self.bulk.append(f"{'*':<8s}{f16(fx)}{f16(fy)}{f16(fz)}")

    def chexa(self, eid, pid, n):
        self.bulk.append("".join(f8(x) for x in ["CHEXA", eid, pid, *n[:6]]))
        self.bulk.append("".join(f8(x) for x in ["", *n[6:]]))

    def spc1(self, sid, comps, gids):
        gids = list(gids)
        for i in range(0, len(gids), 6):
            self.bulk.append("".join(f8(x) for x in ["SPC1", sid, comps, *gids[i:i + 6]]))

    def write(self, path, case_extra=()):
        with open(path, "w") as fh:
            fh.write("\n".join(self.case + list(case_extra) + ["BEGIN BULK"] + self.bulk +
                               ["ENDDATA"]) + "\n")


def run(solver, path, timeout=3600):
    """Runs a solver on `path` (in its directory); returns (f06 path, seconds)."""
    exe = VIBESTRAN if solver == "vibestran" else MYSTRAN
    src_dir, name = os.path.split(os.path.abspath(path))
    d = os.path.join(src_dir, solver)
    os.makedirs(d, exist_ok=True)
    with open(path) as src, open(os.path.join(d, name), "w") as dst:
        dst.write(src.read())
    t0 = time.time()
    r = subprocess.run([exe, name], cwd=d, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    stem = os.path.splitext(name)[0]
    for ext in (".F06", ".f06"):
        f06 = os.path.join(d, stem + ext)
        if os.path.exists(f06):
            return f06, dt
    raise RuntimeError(f"{solver} produced no F06 for {path}:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")


_ROW = re.compile(r"^\s+(\d+)\s+(?:0|G)\s+([-+0-9.E]+)\s+([-+0-9.E]+)\s+([-+0-9.E]+)"
                  r"\s+([-+0-9.E]+)\s+([-+0-9.E]+)\s+([-+0-9.E]+)\s*$")


def displacements(f06, subcase_index=0):
    """{grid: (T1, T2, T3, R1, R2, R3)} from the first displacement table
    (both vibestran's and MYSTRAN's F06 layouts)."""
    txt = open(f06, errors="ignore").read()
    if "FATAL" in txt and "D I S P L A C E M E N T" not in txt:
        raise RuntimeError(f"fatal errors in {f06}")
    out = {}
    blocks = re.split(r"D I S P L A C E M E N T", txt)[1:]
    if not blocks:
        raise RuntimeError(f"no displacement table in {f06}")
    # Tables may span pages; collect rows from every block of the subcase.
    for b in blocks:
        for line in b.splitlines()[:100000]:
            m = _ROW.match(line)
            if m:
                gid = int(m.group(1))
                if gid not in out:
                    out[gid] = tuple(float(m.group(k)) for k in range(2, 8))
            elif out and ("S T R E S S" in line or "F O R C E" in line or "E L E M" in line):
                break
    return out


def measure(f06, quantities):
    """Mean of each quantity's grids' component."""
    D = displacements(f06)
    return {name: sum(D[g][comp] for g in grids) / len(grids)
            for name, (grids, comp) in quantities.items()}
