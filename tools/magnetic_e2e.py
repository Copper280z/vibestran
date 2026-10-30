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
        # Bad field-output selections must fail before solving or opening
        # result files, with the ESET and element identified in the error.
        path = output/"invalid_bfield.bdf"
        path.write_text(deck("uniform", 3, 2).replace(
            "BEGIN BULK", "BFIELD=99\nBEGIN BULK\nESET,99,1,999999"))
        invalid = subprocess.run([str(executable), str(path)], capture_output=True, text=True)
        diagnostic = invalid.stdout+invalid.stderr
        assert invalid.returncode != 0
        assert "BFIELD ESET 99" in diagnostic and "undefined field element 999999" in diagnostic
        assert not path.with_suffix(".F06").exists()
        assert not path.with_suffix(".mag.json").exists()
        for element in ("hex","wedge"):
            path = output/f"solid_{element}.bdf"
            path.write_text(deck("uniform",3,2,element=element))
            subprocess.run([str(executable),str(path)],check=True)
            result = json.loads(path.with_suffix(".mag.json").read_text())
            assert math.isclose(result["cases"][0]["energy_J"],energy,rel_tol=1e-12)
            tree = ET.parse(output/f"solid_{element}_sc1.vtu")
            types = set(map(int,tree.find(".//Cells/DataArray[@Name='types']").text.split()))
            assert types == ({12} if element == "hex" else {13})
            # Volume-preserving bilinear shear: warped native solids must
            # reproduce an exactly representable uniform field and energy.
            warped = []
            field = [0., 0., .3] if element == "hex" else [.3, 0., 0.]
            for line in deck("uniform", 3, 2, element=element).splitlines():
                if line.startswith("GRID,"):
                    parts = line.split(",")
                    x, y, z = map(float, parts[3:6])
                    position = (x, y, z+.2*x*y) if element == "hex" else (x+.2*y*z, y, z)
                    line = ",".join(parts[:3]+[f"{v:.16g}" for v in position])
                elif line.startswith("MBC,"):
                    line = "MBC,1,0,UNIFORM,"+",".join(map(str, field))
                warped.append(line)
            path = output/f"warped_{element}.bdf"
            path.write_text("\n".join(warped)+"\n")
            subprocess.run([str(executable), str(path)], check=True)
            result = json.loads(path.with_suffix(".mag.json").read_text())
            expected = 64*.3**2/(2*4*math.pi*1e-7)
            assert math.isclose(result["cases"][0]["energy_J"], expected, rel_tol=1e-12)
            tree = ET.parse(output/f"warped_{element}_sc1.vtu")
            values = list(map(float, tree.find(".//CellData/DataArray[@Name='B_T']").text.split()))
            assert all(math.isclose(v, field[i % 3], abs_tol=1e-12) for i, v in enumerate(values))
            types = set(map(int, tree.find(".//Cells/DataArray[@Name='types']").text.split()))
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
        # A uniform field is exact even with nonlinear steel: independently
        # integrate the BH curve and verify exported H and extrapolation status.
        for planar in (False, True):
            path = output/("nonlinear2d.bdf" if planar else "nonlinear3d.bdf")
            text = planar_deck("uniform", 4, "quad", .02) if planar else deck("uniform", 3, 2, element="hex")
            text = text.replace("MATMAG,1,1", "MATMAG,1,,1\nTABLEBH,1\n,0,0,.1,100,.2,2000,ENDT")
            path.write_text(text)
            subprocess.run([str(executable), str(path)], check=True)
            result = json.loads(path.with_suffix(".mag.json").read_text())
            case = result["cases"][0]
            assert case["bh_extrapolated"]
            field = [.1, .2, 0] if planar else [.1, .2, .3]
            magnitude = math.sqrt(sum(b*b for b in field))
            extra = magnitude-.2
            mu0 = 4*math.pi*1e-7
            energy_density = 110+2000*extra+extra*extra/(2*mu0)
            volume = 4*.02 if planar else 64
            assert math.isclose(case["energy_J"], volume*energy_density, rel_tol=1e-11)
            expected_h = [(2000+extra/mu0)*b/magnitude for b in field]
            tree = ET.parse(path.with_name(path.stem+"_sc1.vtu"))
            values = list(map(float, tree.find(".//CellData/DataArray[@Name='H_A_per_m']").text.split()))
            assert all(math.isclose(v, expected_h[i % 3], rel_tol=1e-11, abs_tol=1e-9) for i, v in enumerate(values))
        path = output/"planar_winding.bdf"
        path.write_text(planar_deck("winding", 12, "mixed", .02))
        subprocess.run([str(executable), str(path)], check=True)
        result = json.loads(path.with_suffix(".mag.json").read_text())
        case = result["cases"][0]
        # Each leg alone is unbalanced; circuit 1 joins them.
        assert not result["sources"][0]["closed"]
        assert result["circuits"] == [{"id": 1, "parallel": 1, "closed": True, "members": [[11, 1], [12, 1]]}]
        circuit = case["circuits"][0]
        assert math.isclose(circuit["current_A"]*circuit["linkage_Wb_turn"], 2*case["energy_J"], rel_tol=1e-10)
        assert math.isclose(case["circuit_inductance_H"][0][0]*circuit["current_A"], circuit["linkage_Wb_turn"], rel_tol=1e-10)
        positive, negative = 0., 0.
        for leg in (11, 12):
            tree = ET.parse(output/f"planar_winding_coil{leg}.vtu")
            density = list(map(float, tree.find(".//CellData/DataArray[@Name='J_per_ampere_per_m2']").text.split()))
            xyz = list(map(float, tree.find(".//Points/DataArray").text.split()))
            nodes = [xyz[i:i+3] for i in range(0,len(xyz),3)]
            connectivity = list(map(int, tree.find(".//Cells/DataArray[@Name='connectivity']").text.split()))
            offsets = list(map(int, tree.find(".//Cells/DataArray[@Name='offsets']").text.split()))
            start = 0
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
        # Exercise deck selection through the CLI, and verify the actual source
        # topology plus the linear energy/linkage identity for each formulation.
        for scheme, vtk_type in (("AUTO", 12), ("HEX", 12), ("TET", 10)):
            path = output/f"wire_{scheme.lower()}.bdf"
            text = deck("loop", 4, 2).replace("BEGIN BULK", f"BEGIN BULK\nPARAM,WIREMESH,{scheme}")
            path.write_text(text)
            subprocess.run([str(executable), str(path)], check=True)
            result = json.loads(path.with_suffix(".mag.json").read_text())
            case = result["cases"][0]
            assert case["energy_J"] > 0
            source = case["sources"][0]
            assert math.isclose(source["current_A"]*source["linkage_Wb_turn"], 2*case["energy_J"], rel_tol=1e-10)
            files = list(output.glob(f"wire_{scheme.lower()}_wire*.vtu"))
            assert len(files) == 1
            tree = ET.parse(files[0])
            types = set(map(int, tree.find(".//Cells/DataArray[@Name='types']").text.split()))
            assert types == {vtk_type}
        path.write_text(deck("loop", 4, 2).replace("CWIRE,", "UNSUPPORTED,"))
        invalid = subprocess.run([str(executable), str(path)], capture_output=True, text=True)
        assert invalid.returncode != 0 and "unsupported MAGSTAT bulk card" in invalid.stdout+invalid.stderr
    print("Magnetic CLI, analytical fields, exports and sweep passed")


if __name__ == "__main__":
    main()
