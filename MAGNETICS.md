# Static magnetics

`SOL MAGSTAT` adds a separate 3D magnetostatic model, parser, edge DOFs, current
preprocessor and results. Structural SOL 101/103 and thermal SOL 153 keep their
existing paths. The first implementation supports linear isotropic materials,
monotone nonlinear steel, reversible permanent magnets, volumetric stranded
coils and finite-section `CWIRE` paths. It is a foundation for static motor
analysis; the demonstration motor is not a validated engineering design.

## Run an example

Build with `ninja` from `build`, then from the repository root:

```sh
build/vibestran example_cases/magnetic_uniform.bdf /tmp/magnetic_uniform.F06

python3 tools/magnetic_examples.py loop /tmp/magnetic/loop.bdf --n 12
build/vibestran /tmp/magnetic/loop.bdf

python3 tools/magnetic_examples.py coil /tmp/magnetic/coil.bdf --n 12 --extent 1.5
build/vibestran /tmp/magnetic/coil.bdf

python3 tools/magnetic_examples.py motor /tmp/magnetic/motor.bdf --n 4 --angle 10 --current 50
build/vibestran /tmp/magnetic/motor.bdf
```

The generator uses Python's standard library and supplies the entire magnetic
volume, including air. Production `CWIRE` processing has no meshing-tool
dependency. The motor example has a four-tooth square stator, two block magnets
on a steel rotor, two finite conductor loops and an air torque surface. Its
synthetic B–H curve is for demonstrating the solve. `--current` is the current
in each single conductor loop, not an assumed turn count. The motor dimensions
are in meters (the outer box is 40 mm wide with the default extent).

`--backend=cpu` or automatic selection uses Accelerate, CHOLMOD, or Eigen sparse
Cholesky, according to the configured build. GPU/PCG and structural quality or
refinement overrides are not supported for MAGSTAT. Magnetic validity checks
are mandatory. The numerical implementation is built without fast-math.

## Deck dialect

Free, 8-column and 16-column fields, continuations, `D`/implicit exponents and
relative `INCLUDE` paths are supported. Unknown cards and unsupported fields are
errors. These magnetic cards are Vibestran extensions, not standard Nastran cards.

```text
SOL MAGSTAT
CEND
SUBCASE 1
  LOAD = 100
BEGIN BULK
$ GRID, CTETRA and magnetic materials/properties, then sources
...
ENDDATA
```

SI is the default: m, A, T, A/m, S/m, J, N, N m, Wb-turn, H. `PARAM,MAGLEN,s`
multiplies all deck coordinates, section dimensions, probe positions, torque
origins and `WIRESTEP` by `s`. It does not rescale current, conductivity, B–H
data or remanence. Use `s=0.001` for millimeter geometry. Lengths are converted
once on parsing; the C++ magnetic model API takes resolved SI coordinates.

| Card | Fields after the card name |
|---|---|
| `GRID` | `ID,CP,X,Y,Z,CD` (CD blank/zero; no structural PS/SEID) |
| `CORD2R/C/S` | `CID,RID,A1,A2,A3,B1,B2,B3,C1,C2,C3` |
| `CTETRA` | `EID,PID,G1,G2,G3,G4` (four nodes only) |
| `PSOLID` | `PID,MID,CORDM` (CORDM blank/zero; orientation belongs to MATMAG) |
| `MATMAG` | `MID,MUR,BHTID,BR1,BR2,BR3,CID` |
| `BHPOINT` | `TID,B,H` (repeat in strictly increasing B and H order) |
| `MATCOND` | `MID,SIGMA` (positive conductivity in S/m) |
| `ESET` | `SID,EID...` (repeatable, `THRU` allowed) |
| `FSET` | `SID,EID,G1,G2,G3` (repeat for each mesh triangle) |
| `COIL` | `CID,ESET,N,START_FSET,END_FSET` |
| `PWIRE` | `PID,MID,ROUND,DIAMETER` or `PID,MID,RECT,WIDTH,HEIGHT` |
| `CWIRE` | `EID,PID,G1,G2,G0` or `EID,PID,G1,G2,X1,X2,X3` |
| `ICOIL` | `SID,CID,CURRENT` |
| `IWIRE` | `SID,CURRENT,EID...` (`THRU` allowed) |
| `LOAD` | `SID,S,S1,L1,S2,L2,...` |
| `MBC` | `SID,FSET,ZERO` or `SID,FSET,UNIFORM,BX,BY,BZ` |
| `MPROBE` | `ID,X,Y,Z` |
| `MFORCE` | `ID,FSET,ORIGIN_X,ORIGIN_Y,ORIGIN_Z` |

Case control accepts `LOAD`, `MAGBC`, `LABEL`, `TITLE`, `SUBTITLE` and `SUBCASE`.
Global LOAD/MAGBC settings are inherited by subcases. Unexcited subcases can
solve permanent magnets alone. Nested LOAD combinations are additive; cycles
and undefined sets fail. Positive source current follows the source's stated
orientation; reversing the load reverses it.

### Volumetric windings

Select a connected, approximately constant-section winding pack using `ESET`.
Its cells use a `MATMAG` with MUR=1 and no remanence or nonlinear table. Turns
are on the coil; the current is per turn. The normal of an FSET triangle follows
the right-hand rule for its three grids.

For a closed winding, omit END_FSET and select one complete internal cross
section with START_FSET. Its normal defines positive circulation. The material
mesh stays connected across the cut; no duplicated grids are necessary.

```text
ESET,40,1001,THRU,1400
$ Repeat FSET for every triangle covering the one internal cut.
FSET,41,1001,11,12,13
COIL,50,40,80,41
ICOIL,100,50,3.5
```

For an open winding, START_FSET is the entry cap (normals into the conductor)
and END_FSET is the exit cap (normals out). Both caps must lie on the magnetic
domain exterior; all other conductor boundary faces are insulating. Internal
source/sink endpoints are not admissible. Coil volumes cannot overlap.

This is a homogenized stranded approximation. The preprocessor finds a flow
direction and projects a constant-magnitude target into a conservative flux
space with the prescribed ampere-turns. It does not represent individual
strands, arbitrary branching, fill-factor losses or a circuit network.

### Wires

```text
MATCOND,20,5.8e7
PWIRE,30,20,ROUND,0.001
CWIRE,101,30,1,2
CWIRE,102,30,2,3
CWIRE,103,30,3,4
CWIRE,104,30,4,1
IWIRE,100,4.0,101,102,103,104
```

Positive IWIRE flows from each CWIRE's G1 to G2. If an element is reversed,
its signed current must be reversed too. All segments of a connected path
must have the same circulating current after LOAD combinations. Segments not
selected by a current card carry zero, which is an error if their neighbors
carry nonzero current. A complete zero-current path is allowed.

Rectangular section orientation is the width axis, projected perpendicular to
the path. G0 supplies a direction from G1; vector components are in basic axes.
Specify real components such as `0.,0.,1.` to distinguish a vector from G0.
Orientations must agree by parallel transport across bends. A connected path
must use the same section dimensions and conductor material.

Closed unbranched loops and unbranched paths terminating on the domain exterior
are supported. The swept finite conductor must fit entirely inside the supplied
magnetic domain and overlap only nonmagnetic, non-coil volume. Changing section,
branching, joined volumetric/wire sources, overlapping conductors, reversing
bends and folded miters are rejected. Wire sections provide magnetic geometry;
no structural stiffness, thermal solve or magnetomechanical coupling is added.

Round sections are inscribed polygons. Increase WIRESIDE/WIRELAY and refine
WIRESTEP independently of the background magnetic mesh. Interior points of
a segment interpolate between shared miter sections. Extreme bends and rapidly
twisting paths need shorter segments and a section-refinement study.

| Parameter | Default | Meaning |
|---|---:|---|
| `MAGLEN` | 1 | Deck length to meters |
| `MAGTOL` | 1e-8 | Relative equilibrium residual |
| `MAGITER` | 80 | Newton iteration limit per continuation stage |
| `WIRESIDE` | 16 | Round section sides (8–256) |
| `WIRELAY` | 2 | Radial layers or rectangular subdivisions (1–32) |
| `WIRESTEP` | 0 | Maximum axial slice length; zero uses twice section size |

### Materials, boundary and gauge

Linear/PM materials use `H=(B-Br)/(mu0*MUR)`. Br is in tesla. CID rotates the
remanence; cylindrical and spherical bases are evaluated at integration points.
The PM model is reversible and has no irreversible demagnetization or hysteresis.

BHPOINT starts at `(0,0)` and defines piecewise-linear H(B), with positive slopes.
Above the final point the tangent is `1/mu0`; results explicitly report this
extrapolation. A BH table and remanence cannot be combined on one material.

The magnetic domain must be a connected, filled, simply connected tetrahedral
volume with one closed exterior surface. Mesh the air inside conductor holes;
the *conductor* may be a torus but the *magnetic domain* must be filled. The
default exterior condition is zero tangential A. Increase the air domain to
check truncation error. Periodic, symmetry, infinite elements and multiply
connected magnetic domains are not implemented.

`MAGBC=SID` selects MBC. FSET=0 means the whole exterior; an explicit FSET must
cover the whole exterior. UNIFORM prescribes the tangential trace of
`A = B cross x / 2`, useful for applied uniform fields and exact patch tests.
The source perturbation still has zero tangential exterior trace.

## Numerical formulation

Lowest-order Nedelec edge elements represent A; B is their elementwise constant
curl. A boundary-contracted spanning tree removes the gradient nullspace. The
reduced curl-curl system is SPD for positive reluctivity. Nonlinear solves use
the consistent material tangent, residual line search and, on failure, an
eight-increment current continuation retry. Failure to meet the requested
residual is an error, not a partial result.

Each current source uses lowest-order Raviart–Thomas face fluxes. Cell balance,
insulating side walls and unit cut flux constrain a minimum-resistive-energy
solve. Closed-source preprocessing removes the redundant global cell balance.
The stranded projection retains these constraints. N multiplies the resulting
unit-current density exactly once. Wire and magnetic meshes need not match:
tetrahedron intersections and degree-two quadrature transfer `integral N.J dV`
without point sampling. Checks cover per-cell current conservation, source
volume coverage and orthogonality of the transferred load to interior nodal
gradients. Conservation residual tolerances are 1e-10, transfer 1e-9, and relative
coverage 1e-8.

Closed-source linkage is the same coupling vector dotted with A; differential
inductance uses the converged tangent solve. This ensures discrete energy and
reciprocity consistency. Open-source linkage and inductance are unavailable
(`null` in JSON) because they depend on an unspecified return path.

MFORCE integrates Maxwell stress on a closed outward-oriented mesh-face surface
with air on both sides, reporting force and torque about the stated origin.
Choose a surface clear of every current source and material interface. The
field is averaged from its two adjacent cells before stress evaluation. Vary
the surface and refine the mesh: force and torque generally converge more
slowly than field values. Thin air gaps need several element layers.

## Results and angle sweeps

For `name.F06`, the solver also writes:

- `name.mag.json`: SI scalar results, residuals, source orientation maps,
  conservation checks, linkage, differential inductance, force/torque and probes.
- `name_sc<ID>.vtu`: unsmoothed cell B and H plus element IDs.
- `name_coil<ID>.vtu` / `name_wire<ID>.vtu`: source meshes and J per ampere.
- `name.probes.csv`: field at requested probes. On an element boundary this is
  the arithmetic mean of the incident cell traces, including different materials;
  place probes within the intended material when a one-sided value is needed.

JSON matrix order matches the source array. A wire path ID is its first traversed
element; `wire_elements` records each constituent element and traversal sign.
No magnetic OP2 tables are emitted. The JSON `coenergy_J` is the source Legendre
transform `sum(I*lambda)-W`, retaining PM energy constants; it is not generally
`W` when permanent magnets or imposed exterior fields are present.

Supply a valid mesh and rotated PM directions for each angle. The sweep runner
does not remesh or silently rotate selected material regions:

```json
{"cases": [
  {"angle_deg": 9, "deck": "motor9.bdf"},
  {"angle_deg": 10, "deck": "motor10.bdf"},
  {"angle_deg": 11, "deck": "motor11.bdf"}
]}
```

```sh
python3 tools/magnetic_sweep.py manifest.json --vibestran build/vibestran --output /tmp/magnetic/sweep
```

`sweep.csv` includes source currents/linkages and surface force/torque. At interior
angles it computes the three-point derivative of coenergy with respect to angle
in **radians** when source IDs and currents match. This is the fixed-current
virtual-work torque about the rotation axis. It is blank at endpoints or when
currents change. Check mesh and angle-step convergence before comparing it with
the corresponding component of stress torque. Do not differentiate a commutated
current sequence as if the currents were fixed.

The example generator deforms the air mesh between the rigid rotor and fixed
stator and restricts its demonstration angles to +/-20 degrees. Arbitrary
prepared meshes can be used with the solver. A full BLDC sweep requires suitable
meshes and phase-current subcases; no circuit, transient or sliding-band model
is implied.

## Verification

The ordinary test suite contains only internal/analytical checks: oriented edge
and face basis identities, exact intersection quadrature, material energy and
tangent derivatives, straight-coil flux/turn scaling, closed cuts, finite-section
wire Biot–Savart comparison, conservative nonmatching transfer, invalid geometry
and currents, exact uniform fields/energy, PM reversal, nonlinear differential
inductance, nonconvergence errors, mutual reciprocity, superposition, gauge
renumbering, rigid rotation, and loop torque approaching `m cross B` within 3%.
The CLI test checks actual JSON/VTU/CSV values and the prepared-deck sweep.

Run `meson test --print-errorlogs` and `ninja cppcheck` from `build`.
Independent tool comparisons, their dependencies and their outputs live in
[Magnetic_Validation](Magnetic_Validation/README.md); none are registered in
Meson or ordinary CI. Detailed external force and nonlinear benchmark validation
remains necessary before relying on a real motor result.
