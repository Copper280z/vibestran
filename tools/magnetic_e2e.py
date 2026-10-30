#!/usr/bin/env python3
"""Internal analytical CLI verification. Uses no external reference solver."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from magnetic_examples import deck
from magnetic_sweep import run


def main():
    executable = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="vibestran-magnetic-") as directory:
        output = Path(directory)
        path = output/"uniform.bdf"
        path.write_text(deck("uniform", 3, 2))
        subprocess.run([str(executable), str(path)], check=True)
        result = json.loads(path.with_suffix(".mag.json").read_text())
        energy = 64*(.1**2+.2**2+.3**2)/(2*4*math.pi*1e-7)
        assert math.isclose(result["cases"][0]["energy_J"], energy, rel_tol=1e-12)
        tree = ET.parse(output/"uniform_sc1.vtu")
        values = list(map(float, tree.find(".//CellData/DataArray[@Name='B_T']").text.split()))
        assert all(math.isclose(value, [.1, .2, .3][i % 3], abs_tol=1e-12) for i, value in enumerate(values))
        with (output/"uniform.probes.csv").open() as stream:
            for probe in csv.DictReader(stream):
                assert math.isclose(float(probe["Bz_T"]), .3, abs_tol=1e-12)
        manifest = output/"sweep.json"
        manifest.write_text(json.dumps({"cases": [dict(angle_deg=a, deck=path.name) for a in [-1, 0, 1]]}))
        rows = run(executable, manifest, output/"sweep")
        assert abs(rows[1]["virtual_work_torque_Nm"]) < 1e-8
        assert rows[0]["virtual_work_torque_Nm"] is None
        path.write_text(deck("loop", 4, 2).replace("CWIRE,", "UNSUPPORTED,"))
        invalid = subprocess.run([str(executable), str(path)], capture_output=True, text=True)
        assert invalid.returncode != 0 and "unsupported MAGSTAT bulk card" in invalid.stdout+invalid.stderr
    print("Magnetic CLI, analytical fields, exports and sweep passed")


if __name__ == "__main__":
    main()
