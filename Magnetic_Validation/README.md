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
/tmp/magnetic-validation-env/bin/pip install 'magpylib>=5,<7' h5py
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

GetDP/Elmer adapters and TEAM benchmark decks are **not yet provided**. The
table is the next validation work, not a claim that those comparisons ran. The
planar FEMM comparison has run: see [Planar comparison with xfemm](#planar-comparison-with-xfemm-2026-10-31).
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

For the planar formulation, `run_xfemm.py` compares against FEMM's solver; see
[Planar comparison with xfemm](#planar-comparison-with-xfemm-2026-10-31) below.

For native 3D hex/wedge comparisons with GetDP or Elmer, use
`tools/magnetic_examples.py uniform --element hex` (with an output path), then
`coil` and `loop`, and export scalar reference data using the existing comparison
schema. Compare the same physical geometry with independent meshes. Include a
separate distorted-cell refinement study, including warped CHEXA8/CPENTA6
faces. The internal tests now exercise these mapped geometries, curved-face
stress integration, matched coils and independent CWIRE transfer. There is not
yet a completed third-party comparison specifically for warped background
meshes; the Magpylib result above used the earlier background geometry.

## Planar comparison with xfemm (2026-10-31)

`run_xfemm.py` compares planar SOL MAGSTAT with [xfemm](https://github.com/crobarcro/xfemm),
the FEMM 4.2 solver, at commit `34ac6cc`. Each case is built once with gmsh.
vibestran solves the gmsh mesh. xfemm gets the same geometry as FEMM nodes,
segments, arcs and block labels and meshes it with its own mesher. Only the
geometry, materials, sources and boundary conditions are shared, so the two
discretizations are independent. Both solvers run at two mesh levels (target
sizes scaled by 1 and 1/2). Each solver's coarse-to-fine change shows how far
its result is from converged. The steel curve is a smooth saturating
B-H curve (J_s = 1.8 T, initial mu_r = 5000) sampled at 70 points so that the
interpolants differ little; the exterior is A = 0 on a box or circle.

| case | what is compared |
|---|---|
| core | U-core and armature, a 200-turn coil on the yoke: linear iron at 2 A and steel from 0.5 A into deep saturation at 16 A. Coil linkage, armature force, \|B\| in the gap, energy. |
| magnet | the core with a magnet (B_r 1.2 T) in the yoke, steel, coil at -4, 0, +4 A |
| tubular | the static tubular stack (MAGAXI against FEMM axi): axial force on the mover, phase linkages, energy and gap B at six mover offsets, without and with current. |
| motor | the 12-slot, 14-pole motor of `tools/magnetic_motor_example.py`: rated torque and phase linkage over an electrical period (DRIVE), no-load linkage and gap B_r, cogging over a cogging period. The whole machine (PAIRGAP; FEMM periodic AGE) and its anti-periodic half (MPERIOD ANTI; FEMM anti-periodic boundaries and AGE). |

```sh
.venv/bin/python Magnetic_Validation/run_xfemm.py /tmp/xfemm-comparison \
  --femm-host user@host --femmcli ~/xfemm-patched/cfemm/bin/femmcli
.venv/bin/python Magnetic_Validation/run_xfemm.py /tmp/xfemm-comparison --report-only
```

The Python side needs gmsh, h5py, numpy and matplotlib; femmcli runs over ssh on
`--femm-host`. The run writes `results.json`, `summary.md` (every state, both
levels, each solver's refinement change) and `core.png` / `motor.png`. Most
of the run time is FEMM's 312 nonlinear motor solves.

### xfemm needs `xfemm-validation.patch`

The run uses xfemm with the two fixes in `xfemm-validation.patch`:

- `CMMaterialProp`'s constructor leaves `MuMax` uninitialized, and its copy
  constructor drops it. The post-processor reads `MuMax > 0` as "incremental
  permeability", so `mi_loadsolution` segfaults or uses the wrong
  permeability, depending on what the memory held.
- `mo_getgapb` pushes B_r and B_t but returns 1, so Lua gets only B_t.

`mo_getgapb` also returns the air-gap field scaled by 1/100, while
`mo_gapintegral` torque is right. That is not patched; the runner does not use
`mo_getgapb`.

Drawing the FEMM motor also has a requirement. FEMM's air-gap element assumes
equally spaced nodes on both circles, the same number on each, and it gives
every AGE arc one side length. A circle made of arcs of unequal span (magnet
faces and interpoles, or tooth faces and slot openings) breaks that
assumption. The two rings then misregister: no-load torque came out at 1.7 N m
where it should be zero. The FEMM drawing therefore meshes air rings over the
outer and inner thirds of the gap and puts the AGE between two whole circles.
This is the usual FEMM practice, and the field problem is unchanged. vibestran
keeps its air-gap element over the whole gap. Gap B_r is sampled in the middle
of the stator-side ring (r = 30.33 mm) in both solvers.

### Results

Fine mesh: vibestran 68.7k / FEMM 29.7k triangles for the core, 90.7k / 62.3k
for the whole motor. Differences are vibestran against FEMM.

| case | quantity | vibestran | FEMM | difference | coarse to fine, vibestran / FEMM |
|---|---|---:|---:|---:|---:|
| core, linear, 2 A | linkage (Wb-turn) | 0.074061 | 0.07388 | +0.24% | +0.34% / +0.50% |
| | armature F_y (N) | -31.333 | -31.338 | +0.01% | -0.37% / -0.39% |
| core, steel, 8 A | armature F_y (N) | -557.9 | -557.96 | +0.01% | -0.29% / -0.33% |
| core, steel, 16 A | linkage (Wb-turn) | 0.53655 | 0.53696 | -0.08% | +0.36% / +0.92% |
| | armature F_y (N) | -1659.5 | -1671.3 | +0.71% | -0.48% / -1.54% |
| | gap \|B\| (T) | 1.1367 | 1.1405 | -0.33% | +0.19% / +0.74% |
| magnet, 0 A | linkage (Wb-turn) | 0.27378 | 0.27403 | -0.09% | -0.69% / -0.30% |
| | armature F_y (N) | -420.12 | -422.37 | +0.53% | +1.57% / +1.06% |
| magnet, -4 A | armature F_y (N) | -312.01 | -314.28 | +0.72% | +2.54% / +1.56% |
| motor, whole | rated mean torque (N m) | 4.8888 | 4.8893 | -0.01% | -0.08% / -0.07% |
| | rated phase A linkage, fundamental (Wb-turn) | 0.050404 | 0.050402 | +0.00% | +0.10% / +0.05% |
| | no-load phase A linkage, fundamental (Wb-turn) | 0.047373 | 0.047378 | -0.01% | -0.04% / -0.05% |
| | no-load gap B_r, 7th harmonic (T) | 0.96697 | 0.9659 | +0.11% | +0.09% / +0.20% |
| | cogging torque, peak to peak (N m) | 0.052199 | 0.053057 | -1.62% | -16.2% / -7.3% |

- The core and motor agree to a few tenths of a percent or better. The
  difference is about the size of each solver's own coarse-to-fine change, and
  the difference shrinks with refinement. In the motor, the largest pointwise
  difference over the sweeps is 0.13% of the rated torque and 0.03% of the
  linkage at the fine level.
- The 16 A point in deep saturation differs by 0.7% in force, while FEMM moves
  1.5% between levels.
- The magnet case's mesh refines only near the air gap, not at the magnet's
  corners. Its force changes 1.6-2.5% between levels in vibestran and 1.1-1.6%
  in FEMM, and the two differ by 0.4-0.7% at the fine level.
- Cogging torque is the least converged quantity. At the coarse level the two
  solvers differ by 9%; at the fine level by 1.6%. vibestran's value moves 16%
  between levels, against FEMM's 7%, so vibestran's coarse mesh overestimates
  cogging more.
- Sector models agree with the whole machine in each solver, on their own
  meshes. vibestran's half model against its whole: torque, linkage and gap
  B_r within 0.002%, cogging 0.1%. FEMM's: within 0.006%, cogging 1%.
- FEMM's half-model circuits hold only the modeled half of each phase's coils,
  so its linkage is doubled for the comparison. vibestran's sector results are
  already the whole machine's.

#### Graded quadrilateral mesh (2026-11-01)

`--levels graded` solves the motor on the mesh study's graded full-quad mesh
(fq:1.5/0.35, below): 33,548 quadrilaterals for vibestran, and FEMM's
triangles from the same size field (46,166). `--airgap AGE BAND` solves
vibestran both ways against one FEMM run. Differences against FEMM:

| case | vibestran gap | mean torque | rated linkage | no-load linkage | cogging p-p | no-load gap B_r |
|---|---|---:|---:|---:|---:|---:|
| whole | AGE | -0.05% | +0.02% | -0.04% | +1.11% | +0.08% |
| whole | BAND | -0.05% | +0.01% | -0.04% | -0.24% | +0.04% |
| half | AGE | -0.05% | -0.01% | -0.05% | +3.39% | +0.07% |
| half | BAND | -0.05% | -0.01% | -0.05% | +0.12% | +0.03% |

FEMM's half model is 2.1% below its whole in cogging. vibestran's half is
0.15% above its whole with the AGE and 1.7% below with the band. The half is
its own mesh (844 and 1,732 grids unfolded, against 842 and 1,730) with its
own band (1,754 points against 1,733), and the band's cogging moves by about
that much with its point count: 0.0586 and 0.0574 N m at 839 and 841 points
on one mesh in the mesh study. Against FEMM's whole machine the half AGE is
+1.3%.

Wall time of the whole motor's 78 positions on one machine (Ryzen 7 5700,
TITAN V; vibestran pinned to 8 threads; femmcli single-threaded, remeshing,
writing and reloading its solution at every position):

| solver | time | against FEMM |
|---|---:|---:|
| FEMM (its AGE) | 380 s | |
| vibestran BAND, cuDSS | 38.8 s | 9.8x |
| vibestran BAND, CHOLMOD | 56.9 s | 6.7x |
| vibestran AGE, cuDSS | 104.9 s | 3.6x |
| vibestran AGE, CHOLMOD | 177.5 s | 2.1x |

On the M-series Mac (Accelerate) vibestran takes 24.6 s with the band and
100.8 s with the AGE. The AGE's dense block, one row and column per grid on
the gap's circles (2,572 here), dominates its time on this mesh, whose
circles are finer than the triangle levels'.

This level found an AGE defect, fixed on 2026-11-01: the AGE's sum was cut
at the coarser circle's resolution (NHARM's old blank), which left the finer
circle's shorter wavelengths with no gap stiffness. On this mesh (842 and
1,730 grids on the circles) the AGE's cogging was 5.5% above FEMM's (7.7% in
the half model). Each circle's self terms now run to its own resolution
(MAGNETICS.md, "METHOD=AGE").

#### Tubular stack, axisymmetric (2026-11-01)

`--cases tubular` compares MAGAXI with FEMM's axi problem on the static
tubular reference motor of MAGNETICS_ROADMAP.md: three 20 mm N35 magnets
alternating along the axis in a 0.75 mm aluminium tube, a 1 mm gap, nine
6.67 mm coils of 88 turns (A, -C, B, ...), and a 1.5 mm steel sleeve
around the coils (the roadmap leaves back iron open; the sleeve makes the
comparison saturate), in an A = 0 box. Each of six mover offsets over a
pole pitch is its own mesh, solved without current and with A = 3 A,
B = C = -1.5 A. Fine mesh (151.6k vibestran / 41.3k FEMM triangles):

| quantity | vibestran against FEMM | coarse to fine, vibestran / FEMM |
|---|---:|---:|
| phase linkages | within 0.15%; phase C at offset 12 mm, where it is near zero, 0.23% (0.0011 Wb-turn, no load) and 4.2% (4.6e-5 Wb-turn, 1.8e-6 apart, loaded) | up to 0.15% / 0.37% |
| energy | -0.14 to -0.16% | up to 0.20% / 0.34% |
| loaded axial force (2 to 14 N) | within 1.7% | up to 5.7% / 2.9% |
| no-load (detent) force (up to 1.7 N) | within 0.07 N | |
| B along the gap, RMS difference | 2.0% (3.4% coarse) | |

- Linkage and energy agree to about a tenth of a percent and converge
  together.
- The forces agree within each solver's own refinement change.
  vibestran's force is Maxwell stress on one surface around the mover
  (MFORCE BODY, LAYER 3) and moves up to 5.7% (0.3 N) between levels; FEMM's
  weighted stress tensor up to 2.9%. At offset 0, where the exact no-load
  force is zero by symmetry, vibestran gives -0.0003 N and FEMM +0.07 N.
- Pointwise B differs most at the magnet junctions, where B_r peaks and B_z
  crosses zero steeply near the magnets' corners (6.6% of the largest |B|);
  the RMS difference along the gap falls from 3.4% to 2.0% with refinement,
  first-order like pointwise B itself.

These are comparisons of the same 2D (planar or axisymmetric) problem. They
do not cover end effects beyond the modeled domain or the 3D formulation.

## Motor mesh efficiency (2026-11-01)

`mesh_study.py` meshes the xfemm comparison's whole motor several ways,
solves each with vibestran alone (`PARAM,AIRGAP,BAND`, the comparison's 78
positions) and compares it with a triangle mesh at a quarter of the coarse
size (177k grids). Kinds: `tri` (gmsh triangles), `quad` (gmsh
recombination, a few triangles left), `fq` (gmsh full-quad recombination)
and `qm` (all-quad from WebMesh's quadmesh, `run_xfemm.quadmesh_surfaces`).
A spec `kind:scale` scales the whole size field; `kind:scale/near` caps the
far field at scale × 1.4 mm and sets the gap size to near × 0.32 mm.

```sh
.venv/bin/python Magnetic_Validation/mesh_study.py decks OUT --mesh tri:0.25 tri:0.5 fq:1.5/0.35
# solve each OUT/<mesh>/motor.bdf with vibestran; write its wall seconds to OUT/<mesh>/motor.time
.venv/bin/python Magnetic_Validation/mesh_study.py report OUT --reference tri:0.25
.venv/bin/python Magnetic_Validation/mesh_study.py plot OUT/fq_1.5-0.35 mesh.png
```

Errors against the reference, wall time on an M-series Mac (Accelerate):

| mesh | grids | corner angles (cells < 20°) | wall | mean torque | ripple p-p | cogging p-p | linkage, no load / rated | gap B_r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tri:1 (coarse level) | 12,077 | 12–104° (48) | 6.4 s | 0.092% | 22.1% | 14.0% | 0.085 / 0.086% | 1.39% |
| quad:1 | 10,985 | 12–179° (49) | 6.5 s | 0.065% | 22.3% | 13.7% | 0.067 / 0.061% | 1.32% |
| qm:1 | 13,123 | 6–174° (144) | 6.2 s | 0.046% | 21.2% | 14.7% | 0.067 / 0.091% | 1.33% |
| tri:0.5 (fine level) | 45,528 | 12–111° (48) | 32.0 s | 0.017% | 0.16% | 1.54% | 0.014 / 0.032% | 0.10% |
| quad:0.5 | 40,297 | 12–179° (48) | 24.3 s | 0.009% | 1.9% | 2.5% | 0.004 / 0.031% | 0.08% |
| tri:1/0.35 | 34,874 | 12–107° (48) | 24.3 s | 0.018% | 2.7% | 1.9% | 0.026 / 0.050% | 0.11% |
| quad:1/0.35 | 30,900 | 12–179° (48) | 19.1 s | 0.008% | 1.2% | 1.4% | 0.014 / 0.043% | 0.08% |
| qm:1/0.35 | 28,153 | 6–173° (274) | 16.7 s | 0.052% | 3.9% | 2.0% | 0.003 / 0.140% | 0.08% |
| **fq:1.5/0.35** | 33,626 | 12–165° (24) | 20.2 s | 0.014% | 0.52% | 0.89% | 0.009 / 0.048% | 0.08% |

Reference: mean torque 4.8886 N m, ripple 0.0343, cogging 0.0527 N m.
The gap B_r column predates the 2026-11-01 AGE fix, which also changed the
reported gap field: each circle's harmonics now run to its own resolution,
not the coarser circle's. On the graded mesh that moved the band's gap B_r
against FEMM from +0.14% to +0.04%; the other columns are unchanged.
fq:1.5/0.35 also took 53.0 s with CHOLMOD and 35.5 s with cuDSS on the Linux
box, against tri:0.5's 69.1 and 47.8 s (8 threads, TITAN V).

- **fq:1.5/0.35 replaces the fine triangles** (tri:0.5): every figure as
  close to the reference or closer, except ripple (0.5% against 0.16%; every
  other mesh at this size is 1-4%, so tri:0.5's ripple looks like a lucky
  cancellation) and rated linkage (0.048% against 0.032%). It has 26% fewer
  grids, all quadrilaterals, and is 1.6× faster on the Mac and 1.3-1.35× on
  Linux. The smaller gain on Linux is the band: its finer gap circles give
  1,733 band points against 1,213, and the band is a new analysis at every
  position.
- **The element type matters little; the spacing at the gap matters most.**
  Quadrilaterals beat triangles of the same grid count on the smooth figures
  (bilinear and linear elements converge at the same rate, the quad with a
  better constant), but ripple and cogging follow the size at the air gap and
  slot openings for every kind. Refining only there (near 0.35, far field
  unchanged or coarser) gets the uniform fine mesh's ripple and cogging with
  fewer grids.
- **quadmesh** (qm) meshes the motor all-quad, but leaves 6° corners at the
  slot openings' 12° corners and needs two workarounds (see WebMesh's
  `meshers/quadmesh/webmesh_faces/motor/README.md`). gmsh's full-quad
  recombination gives better angles here.
- **NBAND and symmetric meshes.** Before the blank NBAND became prime, quad:0.75
  (840 band points, a multiple of the 14 poles) had a cogging sawtooth three
  times the true value; see the NBAND row and the band section of MAGNETICS.md.
