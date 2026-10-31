#!/usr/bin/env python3
"""Optional MAGSTAT output checks that need third-party Python packages.

The analytical CLI checks live in tests/e2e/test_magnetic_cli.cpp and need
no packages. This script covers what does:
- every VTKHDF block loads through VTK's own vtkHDFReader, the reader
  ParaView uses, with the same cells and fields as the raw HDF5 datasets;
- tools/magnetic_sweep.py reads the /MAGSTAT results with h5py.

The packages are listed in tools/requirements-magnetic.txt. Without them on
the current interpreter, the script re-runs itself with the repository's
.venv interpreter if there is one, and otherwise exits 77, which Meson
reports as a skipped test.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import h5py
    import numpy as np
    import vtk
except ImportError as missing:
    root = Path(__file__).resolve().parents[1]
    venv = root/".venv"/("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv.exists() and Path(sys.prefix).resolve() != (root/".venv").resolve():
        os.execv(str(venv), [str(venv), __file__, *sys.argv[1:]])
    print(f"Skipping: {missing}. Create the venv with\n"
          "  python3 -m venv .venv && .venv/bin/pip install -r tools/requirements-magnetic.txt")
    sys.exit(77)

from magnetic_examples import deck
from magnetic_planar_examples import planar_deck
from magnetic_sweep import run


def solve(executable, path, text):
    path.write_text(text)
    subprocess.run([str(executable), str(path)], check=True, capture_output=True)
    return h5py.File(path.with_suffix(".mag.vtkhdf"), "r")


def vtk_array(data, name):
    array = data.GetArray(name)
    return np.array([array.GetTuple(i) for i in range(array.GetNumberOfTuples())])


def check_paraview(result):
    """Read the fields as ParaView does. Every block must load with the raw
    datasets' cell and point counts, cell types and cell and point fields."""
    reader = vtk.vtkHDFReader()
    reader.SetFileName(result.filename)
    reader.Update()
    collection = reader.GetOutputDataObject(0)
    assert collection.IsA("vtkPartitionedDataSetCollection")
    blocks = {}
    for i in range(collection.GetNumberOfPartitionedDataSets()):
        name = collection.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
        blocks[name] = collection.GetPartition(i, 0)
    assert sorted(blocks) == sorted(k for k in result["VTKHDF"] if k != "Assembly")
    for name, grid in blocks.items():
        raw = result["VTKHDF"][name]
        assert grid.GetNumberOfCells() == raw["NumberOfCells"][0]
        assert grid.GetNumberOfPoints() == raw["NumberOfPoints"][0]
        types = [grid.GetCellType(i) for i in range(grid.GetNumberOfCells())]
        assert np.array_equal(types, raw["Types"][()])
        for group, data in (("CellData", grid.GetCellData()), ("PointData", grid.GetPointData())):
            for field in raw.get(group, {}):
                values = vtk_array(data, field).reshape(raw[group][field].shape)
                assert np.array_equal(values, raw[group][field][()]), (name, field)
    # The assembly groups blocks into fields and sources.
    assembly = collection.GetDataAssembly()
    nodes = {assembly.GetNodeName(n): assembly.GetDataSetIndices(n)
             for n in assembly.GetChildNodes(0)}
    fields = [n for n in blocks if n.startswith("SUBCASE_")]
    assert len(nodes.get("fields", [])) == len(fields)
    assert len(nodes.get("sources", [])) == len(blocks)-len(fields)
    return blocks


def main():
    executable = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="vibestran-vtkhdf-") as directory:
        output = Path(directory)
        cases = {
            "uniform_tet": deck("uniform", 3, 2),
            "uniform_hex": deck("uniform", 3, 2, element="hex"),
            "uniform_wedge": deck("uniform", 3, 2, element="wedge"),
            "planar_mixed": planar_deck("uniform", 4, "mixed", .02),
            "winding": planar_deck("winding", 12, "mixed", .02),
            "wire_tet": deck("loop", 4, 2).replace("BEGIN BULK", "BEGIN BULK\nPARAM,WIREMESH,TET"),
        }
        expected = {
            "uniform_tet": {"SUBCASE_1"}, "uniform_hex": {"SUBCASE_1"},
            "uniform_wedge": {"SUBCASE_1"}, "planar_mixed": {"SUBCASE_1"},
            "winding": {"SUBCASE_1", "COIL_11", "COIL_12"},
        }
        for name, text in cases.items():
            with solve(executable, output/f"{name}.bdf", text) as result:
                blocks = set(check_paraview(result))
                if name in expected:
                    assert blocks == expected[name], (name, blocks)
                else:
                    assert len([b for b in blocks if b.startswith("WIRE_")]) == 1, blocks
        # Fixed-current virtual work from the prepared-deck sweep: zero torque
        # for a uniform field, blank at the endpoints.
        manifest = output/"sweep.json"
        manifest.write_text(json.dumps({"cases": [dict(angle_deg=a, deck="uniform_tet.bdf") for a in [-1, 0, 1]]}))
        rows = run(executable, manifest, output/"sweep")
        assert abs(rows[1]["virtual_work_torque_Nm"]) < 1e-8
        assert rows[0]["virtual_work_torque_Nm"] is None
    print("VTKHDF blocks load through vtkHDFReader; sweep tool reads the HDF5 results")


if __name__ == "__main__":
    main()
