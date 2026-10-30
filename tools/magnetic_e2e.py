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
from magnetic_planar_examples import planar_deck
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
        for element in ("hex","wedge"):
            path = output/f"solid_{element}.bdf"
            path.write_text(deck("uniform",3,2,element=element))
            subprocess.run([str(executable),str(path)],check=True)
            result = json.loads(path.with_suffix(".mag.json").read_text())
            assert math.isclose(result["cases"][0]["energy_J"],energy,rel_tol=1e-12)
            tree = ET.parse(output/f"solid_{element}_sc1.vtu")
            types = set(map(int,tree.find(".//Cells/DataArray[@Name='types']").text.split()))
            assert types == ({12} if element == "hex" else {13})
        for element in ("tri", "quad", "mixed"):
            path = output/f"planar_{element}.bdf"
            path.write_text(planar_deck("uniform", 4, element, .02))
            subprocess.run([str(executable), str(path)], check=True)
            result = json.loads(path.with_suffix(".mag.json").read_text())
            assert result["dimension"] == 2 and result["depth_m"] == .02
            assert math.isclose(result["cases"][0]["energy_J"], 4*.02*.05/(2*4*math.pi*1e-7), rel_tol=1e-12)
            tree = ET.parse(output/f"planar_{element}_sc1.vtu")
            types = set(map(int, tree.find(".//Cells/DataArray[@Name='types']").text.split()))
            assert types == ({5} if element == "tri" else {9} if element == "quad" else {5, 9})
            potential = list(map(float, tree.find(".//PointData/DataArray").text.split()))
            coordinates = list(map(float, tree.find(".//Points/DataArray").text.split()))
            for i, value in enumerate(potential):
                x, y = coordinates[3*i:3*i+2]
                assert math.isclose(value, .1*(y+1)-.2*(x+1), abs_tol=1e-12)
        path = output/"planar_winding.bdf"
        path.write_text(planar_deck("winding", 12, "mixed", .02))
        subprocess.run([str(executable), str(path)], check=True)
        result = json.loads(path.with_suffix(".mag.json").read_text())
        case = result["cases"][0]
        assert result["sources"][0]["closed"]
        assert math.isclose(case["sources"][0]["linkage_Wb_turn"], 2*case["energy_J"], rel_tol=1e-10)
        tree = ET.parse(output/"planar_winding_coil1.vtu")
        density = list(map(float, tree.find(".//CellData/DataArray[@Name='J_per_ampere_per_m2']").text.split()))
        xyz = list(map(float, tree.find(".//Points/DataArray").text.split()))
        nodes = [xyz[i:i+3] for i in range(0,len(xyz),3)]
        connectivity = list(map(int, tree.find(".//Cells/DataArray[@Name='connectivity']").text.split()))
        offsets = list(map(int, tree.find(".//Cells/DataArray[@Name='offsets']").text.split()))
        start, positive, negative = 0, 0., 0.
        for i,end in enumerate(offsets):
            polygon = [nodes[v] for v in connectivity[start:end]]
            area = abs(sum(p[0]*q[1]-p[1]*q[0] for p,q in zip(polygon,polygon[1:]+polygon[:1])))/2
            assert density[3*i] == 0 and density[3*i+1] == 0
            amps = area*density[3*i+2]
            positive += max(amps,0)
            negative += min(amps,0)
            start = end
        assert math.isclose(positive,100,rel_tol=1e-12)
        assert math.isclose(negative,-100,rel_tol=1e-12)
        path.write_text(deck("loop", 4, 2).replace("CWIRE,", "UNSUPPORTED,"))
        invalid = subprocess.run([str(executable), str(path)], capture_output=True, text=True)
        assert invalid.returncode != 0 and "unsupported MAGSTAT bulk card" in invalid.stdout+invalid.stderr
    print("Magnetic CLI, analytical fields, exports and sweep passed")


if __name__ == "__main__":
    main()
