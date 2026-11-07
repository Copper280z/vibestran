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
from magnetic_planar_examples import machine_deck, planar_deck, transient_deck
from magnetic_sweep import run


def solve(executable, path, text):
    path.write_text(text)
    subprocess.run([str(executable), str(path)], check=True, capture_output=True)
    return h5py.File(path.with_suffix(".mag.vtkhdf"), "r")


def vtk_array(data, name):
    array = data.GetArray(name)
    return np.array([array.GetTuple(i) for i in range(array.GetNumberOfTuples())])


def read_step(filename, step):
    """The collection at one step, from a fresh reader as ParaView loads it."""
    reader = vtk.vtkHDFReader()
    reader.SetFileName(filename)
    reader.UpdateInformation()
    if step is not None:
        reader.SetStep(step)
    reader.Update()
    collection = reader.GetOutputDataObject(0)
    assert collection.IsA("vtkPartitionedDataSetCollection")
    blocks = {}
    for i in range(collection.GetNumberOfPartitionedDataSets()):
        name = collection.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
        blocks[name] = collection.GetPartition(i, 0)
    return reader, collection, blocks


def check_paraview(result):
    """Read the fields as ParaView does. Every block must load with the raw
    datasets' cell and point counts, cell types and cell and point fields.
    A time series must report its step values, and at each step every block
    must hold the raw data its step offsets select."""
    reader, collection, blocks = read_step(result.filename, None)
    assert sorted(blocks) == sorted(k for k in result["VTKHDF"] if k != "Assembly")
    times = reader.GetOutputInformation(0).Get(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())
    steps = [None] if times is None else range(len(times))
    for step in steps:
        _, _, blocks = read_step(result.filename, step)
        for name, grid in blocks.items():
            raw = result["VTKHDF"][name]
            if step is not None:
                assert np.array_equal(raw["Steps/Values"][()], times), name
            points, cells = raw["NumberOfPoints"][0], raw["NumberOfCells"][0]
            assert grid.GetNumberOfCells() == cells
            assert grid.GetNumberOfPoints() == points
            types = [grid.GetCellType(i) for i in range(grid.GetNumberOfCells())]
            assert np.array_equal(types, raw["Types"][()])
            first = raw["Steps/PointOffsets"][step] if step is not None else 0
            coordinates = np.array([grid.GetPoint(i) for i in range(points)])
            assert np.array_equal(coordinates, raw["Points"][first:first + points]), (name, step)
            for group, data, size in (("CellData", grid.GetCellData(), cells),
                                      ("PointData", grid.GetPointData(), points)):
                for field in raw.get(group, {}):
                    offsets = f"Steps/{group}Offsets/{field}"
                    first = raw[offsets][step] if step is not None else 0
                    expected = raw[group][field][first:first + size]
                    values = vtk_array(data, field).reshape(expected.shape)
                    assert np.array_equal(values, expected), (name, field, step)
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
            # Two sweeps and a static subcase: one time axis of every written
            # position, each block holding its own states.
            "motion": machine_deck(["SUBCASE 1", "LOAD=1", "MOTION=1", "BFIELD(EVERY=2)=ALL",
                                    "SUBCASE 2", "LOAD=1", "SUBCASE 3", "LOAD=1", "MOTION=2"],
                                   "MOTION,1,20,0,ROTATE,0.,10.,4\nMOTION,2,20,0,ROTATE,5.,20.,2"),
            "transient": transient_deck(),
        }
        expected = {
            "uniform_tet": {"SUBCASE_1"}, "uniform_hex": {"SUBCASE_1"},
            "uniform_wedge": {"SUBCASE_1"}, "planar_mixed": {"SUBCASE_1"},
            "winding": {"SUBCASE_1", "COIL_11", "COIL_12"},
            "motion": {"SUBCASE_1", "SUBCASE_2", "SUBCASE_3", "COIL_11", "COIL_12"},
            "transient": {"SUBCASE_1"},
        }
        for name, text in cases.items():
            with solve(executable, output/f"{name}.bdf", text) as result:
                blocks = set(check_paraview(result))
                if name in expected:
                    assert blocks == expected[name], (name, blocks)
                else:
                    assert len([b for b in blocks if b.startswith("WIRE_")]) == 1, blocks
                if name == "motion":
                    reader, _, _ = read_step(result.filename, None)
                    times = reader.GetOutputInformation(0).Get(
                        vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())
                    assert list(times) == [0, 5, 20, 25], times
                if name == "transient":
                    reader, _, _ = read_step(result.filename, None)
                    times = reader.GetOutputInformation(0).Get(
                        vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())
                    case = result["MAGTRANS/subcases/1"]
                    assert np.array_equal(times, case["time_s"][::5])
                    assert np.all(np.diff(case["angle_deg"][:]) < 0)
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
