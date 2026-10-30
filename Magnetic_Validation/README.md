# Independent magnetic validation

This directory is deliberately separate from Meson tests and ordinary CI, like
the MYSTRAN validation workflow. External executables, Python packages, fetched
models and external reference results are **not** test-suite dependencies or
golden regression data. Reports are generated only when explicitly running a
validation command.

## Runnable field comparison

The runner generates the same finite-wire square loop at three magnetic mesh
levels and two exterior distances, runs Vibestran, and compares B at the center
and an axial probe with Magpylib's independent filament solution. The conductor
is thin relative to the loop and probes are away from it; this comparison does
not validate self-inductance or local fields at the conductor surface.
[Magpylib's Polyline documentation](https://magpylib.readthedocs.io/en/stable/_autogen/magpylib.current.html)
describes the line-current model and SI units.

```sh
python3 -m venv /tmp/magnetic-validation-env
/tmp/magnetic-validation-env/bin/pip install 'magpylib>=5,<7'
/tmp/magnetic-validation-env/bin/python Magnetic_Validation/run_magpylib.py \
  --vibestran build/vibestran --output /tmp/magnetic-comparison
```

Default mesh levels are 12, 16, 24 subdivisions in the inner interval and exterior
half-widths are 3 and 5. Source and probe planes stay fixed across levels; the
outer air is graded. The report records actual package version, B vectors,
relative vector errors, linkage, and timing. Exit status is nonzero if the
finest/largest case exceeds the default 1% field tolerance. Review the complete
refinement and boundary study, not just that final status. Refine section sides,
layers and axial slices separately for finite-conductor quantities.

## Other independent references

| Tool / case | Intended comparison | Setup requirements |
|---|---|---|
| [GetDP 3D magnetostatics](https://github.com/getdp-project/getdp/blob/master/templates/Interactive_Magnetostatics.pro) | 3D B, linkage, energy, PMs and nonlinear steel | Match geometry, source model, B–H interpolation, outer boundary and SI units. Prepare an independent mesh and formulation. |
| [Elmer steady coils](https://github.com/ElmerCSC/elmerfem/blob/devel/fem/tests/mgdyn_steady_coils/case.sif) | Coil preprocessing and 3D fields | Compare current cross-section integrals before comparing magnetic results. |
| [Elmer air-gap force](https://github.com/ElmerCSC/elmerfem/blob/devel/fem/tests/mgdyn_airgap_force/case.sif) | Maxwell force and torque | Use converged source-free air surfaces and matching force/torque conventions. |
| [FEMM](https://www.femm.info/doku/doku.php?id=MagneticsTutorial) | Planar or axisymmetric reductions | FEMM is 2D/axisymmetric; account for extrusion depth/end effects. It is not a reference for an arbitrary finite 3D motor. |
| [TEAM 20](https://www.compumag.org/wp/wp-content/uploads/2018/06/problem20.pdf) | Nonlinear 3D static force | Use the published geometry, excitation cases and steel data; compare fields as well as force. |

GetDP/Elmer/FEMM adapters and TEAM benchmark decks are **not yet provided**.
The table is the next validation work, not a claim that those comparisons ran.
TEAM 13 is also a suitable nonlinear field benchmark once its published geometry
and data are independently transcribed and reviewed.

For these comparisons, export scalar measurements from both tools into JSON:

```json
{
  "tool": "independent solver",
  "version": "record actual version",
  "metrics": {
    "gap_B_T": {"value": 0.5, "abs_tol": 0.000001, "rel_tol": 0.01},
    "torque_Nm": {"value": 0.2, "abs_tol": 0.000001, "rel_tol": 0.03}
  }
}
```

The numbers above illustrate the schema, **not reference measurements**.
Candidate metrics require `value`; tolerances are taken from the independently
reviewed reference manifest. Near-zero quantities need a physically chosen
absolute tolerance. Run:

```sh
python3 Magnetic_Validation/compare_scalars.py reference.json candidate.json \
  --output comparison.json
```

Use at least three meshes, separate winding refinement and outer-domain
enlargement. Aim for 1% in fields/linkage/energy and 3% in nonzero force/torque
after convergence; report any materially different source or end-effect model.
For motors, compare several fixed-current angles, a zero-current cogging case,
Maxwell torque on multiple air surfaces, and an angle-step study of fixed-current
coenergy derivatives. Do not use a torque result that depends strongly on which
air surface was selected.

## First implementation checks (2026-10-30)

Magpylib **6.0.1**, square loop, maximum relative vector error across two probes:

| Inner subdivisions | Air half-width 3 | Air half-width 5 |
|---:|---:|---:|
| 12 | 2.548% | 2.493% |
| 16 | 2.542% | 1.852% |
| 24 | 1.123% | **0.699%** |

This passes the final 1% field criterion. It is not external validation of the
nonlinear material, inductance or force implementations. Those have internal
mathematical/analytical checks but still need the independent comparisons above.

The deliberately coarse motor example at 10 degrees and 50 A converged, but its
stress torque was approximately -0.01181 N m while a 9/10/11-degree fixed-current
coenergy difference gave -0.008712 N m. That disagreement is not accepted force
validation. Refine the air gap, change the integration surface and reduce the
angle step before interpreting motor torque. The solver reports both results
without concealing their discretization sensitivity.


## First-order element extension checks

The ordinary suite verifies CTETRA4/CHEXA8/CPENTA6 and planar CTRIA3/CQUAD4
against analytical and manufactured solutions. External comparisons remain
optional commands in this directory, with no Meson or CI registration.

After the hex wire-preprocessing extension, the existing Magpylib 6.0.1 loop
comparison was rerun at `--levels 24 --extents 5`: maximum probe vector error
was **0.699%**, below the 1% threshold. This checks the finite-section source
against the independent filament field away from the conductor. It does not
validate finite-section self-inductance, nonlinear materials or force. Source
transfer took more work for mapped round-wire hexes; fewer preprocessing cells
do not imply a universal speedup.

For the planar formulation, [FEMM's magnetostatic tutorial](https://www.femm.info/doku/doku.php?id=MagneticsTutorial)
is a useful independent comparison workflow. Generate a planar winding or motor
with `tools/magnetic_planar_examples.py`, reproduce the same in-plane geometry,
MATMAG/BHPOINT data, axial depth, signed turns and exterior A=0 condition in FEMM,
and compare B probes, energy, circuit linkage and torque. Start with the air
winding case, then add steel and PMs. Refine both meshes and enlarge the exterior
before accepting agreement. There is not yet an automated FEMM adapter or a
claimed independent validation of planar motor torque.

For native 3D hex/wedge comparisons with GetDP or Elmer, use
`tools/magnetic_examples.py uniform --element hex` (with an output path), then
`coil` and `loop`, and export scalar reference data using the existing comparison
schema. Compare the same physical geometry with independent meshes. Include a
separate distorted-cell refinement study, including warped CHEXA8/CPENTA6
faces. The internal tests now exercise these mapped geometries, curved-face
stress integration, matched coils and independent CWIRE transfer. There is not
yet a completed third-party comparison specifically for warped background
meshes; the Magpylib result above used the earlier background geometry.
