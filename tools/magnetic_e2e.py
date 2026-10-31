#!/usr/bin/env python3
"""Internal analytical CLI verification. Uses no external reference solver."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import h5py
    import numpy as np
    import vtk
except ImportError as missing:
    # Meson reports exit code 77 as a skipped test.
    print(f"magnetic_e2e needs h5py, numpy and the VTK Python module: {missing}")
    sys.exit(77)

from magnetic_examples import deck
from magnetic_planar_examples import planar_deck
from magnetic_sweep import run


def solve(executable, path, text):
    path.write_text(text)
    subprocess.run([str(executable), str(path)], check=True)
    return h5py.File(path.with_suffix(".mag.vtkhdf"), "r")


def vtk_array(data, name):
    array = data.GetArray(name)
    return np.array([array.GetTuple(i) for i in range(array.GetNumberOfTuples())])


def check_paraview(result):
    """Read the fields through VTK's own VTKHDF reader, as ParaView does. Every
    block must load with the same cells and cell fields as the raw datasets."""
    reader = vtk.vtkHDFReader()
    reader.SetFileName(result.filename)
    reader.Update()
    collection = reader.GetOutputDataObject(0)
    blocks = {}
    for i in range(collection.GetNumberOfPartitionedDataSets()):
        name = collection.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
        blocks[name] = collection.GetPartition(i, 0)
    assert sorted(blocks) == sorted(k for k in result["VTKHDF"] if k != "Assembly")
    for name, grid in blocks.items():
        raw = result["VTKHDF"][name]
        assert grid.GetNumberOfCells() == raw["NumberOfCells"][0]
        assert grid.GetNumberOfPoints() == raw["NumberOfPoints"][0]
        for field in raw["CellData"]:
            values = vtk_array(grid.GetCellData(), field)
            assert np.array_equal(values.reshape(raw["CellData"][field].shape), raw["CellData"][field][()])
    return blocks


def main():
    executable = Path(sys.argv[1]).resolve()
    mu0 = 4*math.pi*1e-7
    with tempfile.TemporaryDirectory(prefix="vibestran-magnetic-") as directory:
        output = Path(directory)
        path = output/"uniform.bdf"
        energy = 64*(.1**2+.2**2+.3**2)/(2*mu0)
        with solve(executable, path, deck("uniform", 3, 2)) as result:
            assert math.isclose(result["MAGSTAT/subcases/1"].attrs["energy_J"], energy, rel_tol=1e-12)
            check_paraview(result)
            assert np.allclose(result["VTKHDF/SUBCASE_1/CellData/B_T"][()], [.1, .2, .3], rtol=0, atol=1e-12)
            assert np.allclose(result["MAGSTAT/subcases/1/probes/B_T"][:, 2], .3, rtol=0, atol=1e-12)
        manifest = output/"sweep.json"
        manifest.write_text(json.dumps({"cases": [dict(angle_deg=a, deck=path.name) for a in [-1, 0, 1]]}))
        rows = run(executable, manifest, output/"sweep")
        assert abs(rows[1]["virtual_work_torque_Nm"]) < 1e-8
        assert rows[0]["virtual_work_torque_Nm"] is None
        for element in ("hex", "wedge"):
            vtk_type = 12 if element == "hex" else 13
            path = output/f"solid_{element}.bdf"
            with solve(executable, path, deck("uniform", 3, 2, element=element)) as result:
                assert math.isclose(result["MAGSTAT/subcases/1"].attrs["energy_J"], energy, rel_tol=1e-12)
                assert set(result["VTKHDF/SUBCASE_1/Types"][()]) == {vtk_type}
                assert check_paraview(result)["SUBCASE_1"].GetCellType(0) == vtk_type
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
            with solve(executable, path, "\n".join(warped)+"\n") as result:
                expected = 64*.3**2/(2*mu0)
                assert math.isclose(result["MAGSTAT/subcases/1"].attrs["energy_J"], expected, rel_tol=1e-12)
                assert np.allclose(result["VTKHDF/SUBCASE_1/CellData/B_T"][()], field, rtol=0, atol=1e-12)
                assert set(result["VTKHDF/SUBCASE_1/Types"][()]) == {vtk_type}
        for element in ("tri", "quad", "mixed"):
            path = output/f"planar_{element}.bdf"
            with solve(executable, path, planar_deck("uniform", 4, element, .02)) as result:
                magstat = result["MAGSTAT"]
                assert magstat.attrs["dimension"] == 2 and magstat.attrs["depth_m"] == .02
                assert math.isclose(magstat["subcases/1"].attrs["energy_J"], 4*.02*.05/(2*mu0), rel_tol=1e-12)
                block = result["VTKHDF/SUBCASE_1"]
                assert set(block["Types"][()]) == ({5} if element == "tri" else {9} if element == "quad" else {5, 9})
                grid = check_paraview(result)["SUBCASE_1"]
                potential = vtk_array(grid.GetPointData(), "A_normal_Wb_per_m")[:, 0]
                assert np.array_equal(potential, block["PointData/A_normal_Wb_per_m"][()])
                for value, (x, y, _) in zip(potential, block["Points"][()]):
                    assert math.isclose(value, .1*(y+1)-.2*(x+1), abs_tol=1e-12)
        # A uniform field is exact even with nonlinear steel: independently
        # integrate the BH curve and verify exported H and extrapolation status.
        for planar in (False, True):
            path = output/("nonlinear2d.bdf" if planar else "nonlinear3d.bdf")
            text = planar_deck("uniform", 4, "quad", .02) if planar else deck("uniform", 3, 2, element="hex")
            text = text.replace("MATMAG,1,1", "MATMAG,1,,1\nTABLEBH,1\n,0,0,.1,100,.2,2000,ENDT")
            with solve(executable, path, text) as result:
                case = result["MAGSTAT/subcases/1"].attrs
                assert case["bh_extrapolated"]
                field = [.1, .2, 0] if planar else [.1, .2, .3]
                magnitude = math.sqrt(sum(b*b for b in field))
                extra = magnitude-.2
                energy_density = 110+2000*extra+extra*extra/(2*mu0)
                volume = 4*.02 if planar else 64
                assert math.isclose(case["energy_J"], volume*energy_density, rel_tol=1e-11)
                expected_h = [(2000+extra/mu0)*b/magnitude for b in field]
                values = result["VTKHDF/SUBCASE_1/CellData/H_A_per_m"][()]
                assert np.allclose(values, expected_h, rtol=1e-11, atol=1e-9)
        path = output/"planar_winding.bdf"
        with solve(executable, path, planar_deck("winding", 12, "mixed", .02)) as result:
            magstat = result["MAGSTAT"]
            case = magstat["subcases/1"]
            # Each leg alone is unbalanced; circuit 1 joins them.
            assert not magstat["sources/closed"][0]
            circuits = magstat["circuits"]
            assert list(circuits["id"]) == [1] and list(circuits["parallel"]) == [1]
            assert list(circuits["closed"]) == [1]
            assert circuits["members"][()].tolist() == [[11, 1], [12, 1]]
            assert list(circuits["member_offsets"]) == [0, 2]
            current = case["circuits/current_A"][0]
            linkage = case["circuits/linkage_Wb_turn"][0]
            assert math.isclose(current*linkage, 2*case.attrs["energy_J"], rel_tol=1e-10)
            assert math.isclose(case["circuits/inductance_H"][0, 0]*current, linkage, rel_tol=1e-10)
            # A single planar region has no linkage of its own.
            assert all(math.isnan(v) for v in case["sources/linkage_Wb_turn"])
            assert set(check_paraview(result)) == {"SUBCASE_1", "COIL_11", "COIL_12"}
            positive, negative = 0., 0.
            for leg in (11, 12):
                block = result[f"VTKHDF/COIL_{leg}"]
                density = block["CellData/J_per_ampere_per_m2"][()]
                nodes = block["Points"][()].tolist()
                connectivity = block["Connectivity"][()]
                offsets = block["Offsets"][()]
                for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
                    polygon = [nodes[v] for v in connectivity[start:end]]
                    area = abs(sum(p[0]*q[1]-p[1]*q[0] for p, q in zip(polygon, polygon[1:]+polygon[:1])))/2
                    assert density[i, 0] == 0 and density[i, 1] == 0
                    amps = area*density[i, 2]
                    positive += max(amps, 0)
                    negative += min(amps, 0)
            assert math.isclose(positive, 100, rel_tol=1e-12)
            assert math.isclose(negative, -100, rel_tol=1e-12)
        # Exercise deck selection through the CLI, and verify the actual source
        # topology plus the linear energy/linkage identity for each formulation.
        for scheme, vtk_type in (("AUTO", 12), ("HEX", 12), ("TET", 10)):
            path = output/f"wire_{scheme.lower()}.bdf"
            text = deck("loop", 4, 2).replace("BEGIN BULK", f"BEGIN BULK\nPARAM,WIREMESH,{scheme}")
            with solve(executable, path, text) as result:
                case = result["MAGSTAT/subcases/1"]
                assert case.attrs["energy_J"] > 0
                current = case["sources/current_A"][0]
                linkage = case["sources/linkage_Wb_turn"][0]
                assert math.isclose(current*linkage, 2*case.attrs["energy_J"], rel_tol=1e-10)
                wires = [name for name in result["VTKHDF"] if name.startswith("WIRE_")]
                assert len(wires) == 1
                assert set(result["VTKHDF"][wires[0]]["Types"][()]) == {vtk_type}
                assert check_paraview(result)[wires[0]].GetCellType(0) == vtk_type
        path.write_text(deck("loop", 4, 2).replace("CWIRE,", "UNSUPPORTED,"))
        invalid = subprocess.run([str(executable), str(path)], capture_output=True, text=True)
        assert invalid.returncode != 0 and "unsupported MAGSTAT bulk card" in invalid.stdout+invalid.stderr
    print("Magnetic CLI, analytical fields, VTKHDF exports and sweep passed")


if __name__ == "__main__":
    main()
