# Static magnetics

`SOL MAGSTAT` adds separate 3D and planar 2D magnetostatic models, a parser, current
preprocessing and results. Structural SOL 101/103 and thermal SOL 153 keep their
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
multiplies all deck coordinates (including coordinate-system origins, so
probe positions and torque origins), section dimensions, `MAG2D` depth and
`WIRESTEP` by `s`. It does not rescale current, conductivity, B–H
data or remanence. Use `s=0.001` for millimeter geometry. Lengths are converted
once on parsing; the C++ magnetic model API takes resolved SI coordinates.
`PARAM,AIRGAP,AGE`, `PARAM,AIRGAP,BAND` or `PARAM,AIRGAP,MESH` sets the METHOD of every `PAIRGAP`
in the deck, overriding the cards ([Air gap](#air-gap-pairgap)).

| Card | Fields after the card name |
|---|---|
| `GRID` | `ID,CP,X,Y,Z,CD` (CD blank/zero; no structural PS/SEID) |
| `CORD2R/C/S` | `CID,RID,A1,A2,A3,B1,B2,B3,C1,C2,C3` |
| `CTETRA` | `EID,PID,G1,G2,G3,G4` (four nodes only) |
| `CHEXA` | `EID,PID,G1,...,G8` (standard corner order; first order only) |
| `CPENTA` | `EID,PID,G1,...,G6` (triangular ends G1–G3 and G4–G6) |
| `CTRIA3` / `CQUAD4` | `EID,PID,G1,...` (three/four corners, planar mode only) |
| `MAG2D` | `CID,DEPTH` (one card activates planar mode; CID=0 uses basic z) |
| `PMAG2D` | `PID,MID,CID,AXIS` (planar magnetic property; orientation as PMAG) |
| `PAIRGAP` | `PID,METHOD,NHARM,NBAND,NLAYER` (planar air-gap property; METHOD `AGE`, `BAND` or `MESH`; see [Air gap](#air-gap-pairgap)) |
| `MOTION` | `ID,ESET,CID,ROTATE,START,STEP,COUNT` + `,SPEED` (planar rotation of ESET through the air gap, degrees; SPEED in rpm; see [Motion](#motion-motion)) |
| `DRIVE` | `ID,WAVE,POLES,IPEAK,GAMMA,THETA0` + `,C1,C2,...` (phase currents from the rotor position; WAVE `SINE` or `SIX`; see [Phase currents](#phase-currents-drive)) |
| `COIL2D` | `ID,ESET,SIGNED_TURNS` (one planar winding region; ID unique) |
| `PSOLID` | `PID,MID,CORDM` (CORDM orients remanence along its axis 1) |
| `PMAG` | `PID,MID,CID,AXIS` (3D magnetic property with a signed orientation axis) |
| `MATMAG` | `MID,MUR,BHTID,BR` (MUR and BHTID exclusive; BR is a magnitude) |
| `TABLEBH` | `TID,INTERP` then `B1,H1,B2,H2,...,ENDT` on continuations |
| `MATCOND` | `MID,SIGMA` (positive conductivity in S/m) |
| `ESET` | `SID,EID...` or `SID,PROP,PID...` (repeatable rows, `THRU` allowed) |
| `FSET` | `SID,EID,G1,G2[,G3[,G4]]` (2D edge or 3D triangular/quadrilateral face) |
| `COIL` | `ID,ESET,N,START_FSET,END_FSET` or `ID,ESET,N,CUT,CID` (closed coil) |
| `PWIRE` | `PID,MID,ROUND,DIAMETER` or `PID,MID,RECT,WIDTH,HEIGHT` |
| `CWIRE` | `EID,PID,G1,G2,G0` or `EID,PID,G1,G2,X1,X2,X3` |
| `WIREPATH` | `ID,EID` (names the CWIRE path containing EID; positive G1 to G2 of EID) |
| `CIRCUIT` | `ID,PARALLEL,S1,P1,S2,P2,...` (series sources with polarity ±1) |
| `ICIRC` | `SID,ID,CURRENT` (terminal current of a circuit or unlisted source) |
| `LOAD` | `SID,S,S1,L1,S2,L2,...` |
| `MBC` | `SID,FSET,ZERO` or `SID,FSET,UNIFORM,BX,BY,BZ` |
| `MPROBE` | `ID,CP,X1,X2,X3,CD` (GRID layout; field also reported in CD axes) |
| `MFORCE` | `ID,FACES,FSET,CID` or `ID,BODY,ESET,CID,LAYER` (torque about CID origin) |

Case control accepts `LOAD`, `MAGBC`, `BFIELD`, `MPROBE`, `MSOURCE`, `MOTION`,
`DRIVE`, `MSTEP`, `LDQ`, `LABEL`, `TITLE`, `SUBTITLE` and `SUBCASE`; `BFIELD` takes the
describer `(EVERY=n)` for MOTION subcases. Global settings are inherited by
subcases. Unexcited subcases can
solve permanent magnets alone. Nested LOAD combinations are additive; cycles
and undefined sets fail. Positive source current follows the source's stated
orientation; reversing the load reverses it.

`ESET,SID,PROP,PID...` selects every magnetic element with those properties,
which suits preprocessors that export regions as property IDs (Gmsh physical
groups, for example). Rows with one SID accumulate; a property no element uses,
or an element selected twice, is an error.

### Circuits

`COIL`, `COIL2D` and `WIREPATH` IDs name current sources. They share one ID
namespace with `CIRCUIT`. `CIRCUIT,ID,PARALLEL,S1,P1,...` connects sources in
series with polarity Pi = ±1. PARALLEL (default 1) is the number of identical
parallel branches the listed sources represent, so each source carries
`Pi * I / PARALLEL` per turn. A source belongs to at most one circuit. A source
not listed in any circuit is its own circuit with polarity +1, so single-coil
decks need no CIRCUIT card.

`ICIRC,SID,ID,I` sets the terminal current I of a circuit, or of an unlisted
source. Current on a member of an explicit circuit is an error. Circuit
linkage and differential inductance are

```text
lambda_c = sum_i P_i lambda_i / a_c
L_cd     = sum_(i in c) sum_(j in d) P_i P_j L_ij / (a_c a_d)
```

computed from the circuit coupling vector `sum P_i c_i / a`, so they cost one
tangent solve per circuit. They are available when every 3D member is closed,
or when a planar circuit's polarity-weighted signed turns sum to zero. The F06
and HDF5 results report circuits in ID order, including implicit ones;
per-source values remain in the HDF5 `sources` groups.

### Volumetric windings

Select a connected, approximately constant-section winding pack using `ESET`.
Its cells use a `MATMAG` with MUR=1 and no remanence or nonlinear table. Turns
are on the coil; the current is per turn. The normal of an FSET face follows
the right-hand rule for its cyclic grid order (three or four grids).

For a closed winding, omit END_FSET and select one complete internal cross
section with START_FSET. Its normal defines positive circulation. The material
mesh stays connected across the cut; no duplicated grids are necessary.

```text
ESET,40,1001,THRU,1400
$ Repeat FSET for every face covering the one internal cut.
FSET,41,1001,11,12,13
COIL,50,40,80,41
ICIRC,100,50,3.5
```

`COIL,ID,ESET,N,CUT,CID` derives the cut instead, from the half-plane z = 0,
x > 0 of the CORD2R CID (0 for basic), with positive circulation along +z.
Place the CID origin on the coil axis with z along the current where it
crosses, and x pointing at the crossing to use. The cut is the set of
internal faces separating coil cells whose centroids have local z < 0 from
those with z >= 0, keeping faces with local x > 0. It may zig-zag through
cells. Since a conserved current has the same flux through any separating
surface, the source is the same as with a flat cut. Its stranded section area
is the cut's area projected onto z. Any closed-coil cut, FSET or derived,
must form one complete cross section that the winding crosses once;
otherwise the error says the cut is incomplete or crosses the winding more
than once.

```text
CORD2R,60,,0.,0.,0.,0.,1.,0.     $ z along +y where the coil crosses x > 0
,1.,0.,0.
COIL,50,40,80,CUT,60
```

For an open winding, START_FSET is the entry cap (normals into the conductor)
and END_FSET is the exit cap (normals out). Both caps must lie on the magnetic
domain exterior; all other conductor boundary faces are insulating. Internal
source/sink endpoints are not admissible. Coil volumes cannot overlap.

This is a homogenized stranded approximation. The preprocessor finds a flow
direction and projects a constant-magnitude target into a conservative flux
space with the prescribed ampere-turns. It does not represent individual
strands, arbitrary branching or fill-factor losses.

### Wires

```text
MATCOND,20,5.8e7
PWIRE,30,20,ROUND,0.001
CWIRE,101,30,1,2
CWIRE,102,30,2,3
CWIRE,103,30,3,4
CWIRE,104,30,4,1
WIREPATH,60,101
ICIRC,100,60,4.0
```

Every connected CWIRE path is named by exactly one `WIREPATH`, which may name
any of its elements. Positive current flows from G1 to G2 of the named element.
The solver propagates the direction along the path, so reversed segments need
no sign bookkeeping; `/MAGSTAT/sources/wire_elements` records each element
with its traversal sign relative to the path. A path that carries no current is simply
not in any load set.

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

`PARAM,WIREMESH,AUTO` (the default) prefers native hexahedra. Rectangular
sections use a regular quad grid; round sections use a square quad core and
quadrilateral annuli, without collapsed cells on the centerline. Each lofted
quad makes one hex. If a spatial path produces warped or nonconvex loft faces,
AUTO falls back to conforming tetrahedra for its whole connected component.
Folded/invalid sweeps still fail validation.

Use `PARAM,WIREMESH,HEX` to require hexahedra, or `PARAM,WIREMESH,TET` to split
each lofted quad prism into six conforming tetrahedra. Both choices use the
same section partition and axial refinement. HEX returns a diagnostic if the
sweep needs the geometric fallback. These restrictions apply only to internal
CWIRE cells; user CHEXA/CPENTA faces may be warped. The `WIRE_<ID>` source
block exposes the actual cell types and the report includes preprocessing mesh
sizes.

Hexes reduce the current-preprocessing system, while tets have cheaper polynomial
transfer on planar-faced backgrounds. The faster choice depends on the source
and field meshes; compare both and check refinement. Integration failure returns
a diagnostic rather than silently changing the requested element type.

Round sections are inscribed polygons. WIRESIDE rounds upward to a multiple of
four for the quad core. Increase WIRESIDE/WIRELAY and refine
WIRESTEP independently of the background magnetic mesh. Interior points of
a segment interpolate between shared miter sections. Extreme bends and rapidly
twisting paths need shorter segments and a section-refinement study.

| Parameter | Default | Meaning |
|---|---:|---|
| `MAGLEN` | 1 | Deck length to meters |
| `MAGTOL` | 1e-8 | Relative equilibrium residual |
| `MAGITER` | 80 | Newton iteration limit per continuation stage |
| `MAGPCG` | 0 | CG iterations per Newton step with an earlier tangent's factor before refactoring (0–1000); 0 factors every tangent. See [Reusing a factor](#reusing-a-factor-parammagpcg) |
| `WIREMESH` | AUTO | Internal CWIRE mesh: AUTO (hex with geometric tet fallback), HEX (required), TET (required) |
| `WIRESIDE` | 16 | Requested round sides (8–256), rounded up to a multiple of four |
| `WIRELAY` | 2 | Annular layers outside the square core, or rectangular subdivisions (1–32) |
| `WIRESTEP` | 0 | Maximum axial slice length; zero uses twice section size |

### Materials, boundary and gauge

Linear/PM materials use `H=(B-Br)/(mu0*MUR)`. `BR` on MATMAG is the remanence
magnitude in tesla; the property that references the material gives its
direction. `PMAG,PID,MID,CID,AXIS` (or `PMAG2D` in planar mode) magnetizes
along `sign(AXIS) * e_|AXIS|` of CID, with AXIS in ±1, ±2, ±3 (default +1).
`PSOLID,PID,MID,CORDM` is the same with AXIS=+1. Cylindrical and spherical
axes are evaluated at integration points, so `CORD2C` AXIS=1 is radial and
AXIS=2 tangential. North and south poles share one material by using
opposite AXIS signs:

```text
CORD2R,100,,0,0,0,0,0,1          $ rotor frame: rotating it moves every magnet
,0.9848,0.1736,0
CORD2R,101,100,0,0,0,0,0,1       $ pole frame, given in rotor coordinates
,1,0,0
MATMAG,5,1.05,,1.2
PMAG,11,5,101,1                  $ north pole
PMAG,12,5,101,-1                 $ south pole
```

A magnetized material whose property leaves CID/CORDM blank is an error; an
explicit 0 means basic axes. In planar mode each magnetization must stay in
the plane: use an in-plane axis of a rectangular system, or AXIS 1 or 2 of a
cylindrical system whose axis is the plane normal. The PM model is reversible
and has no irreversible demagnetization or hysteresis.

`TABLEBH,TID,INTERP` lists `B,H` pairs (tesla, A/m) on continuations in the
`TABLEM1` layout, ending with `ENDT`; blank fields are padding. INTERP is blank
or LINEAR. The table starts at `(0,0)` and defines piecewise-linear H(B), with
B and H strictly increasing. Above the final point the tangent is `1/mu0`;
results explicitly report this extrapolation. A material with BHTID must leave
MUR blank and has no remanence.

```text
MATMAG,2,,7
TABLEBH,7
,0.,0.,1.,200.,1.5,1200.
,1.8,8000.,2.,50000.,ENDT
```

The 3D magnetic domain must be a connected, filled, simply connected volume
with one closed exterior surface. CTETRA4, CHEXA8 and CPENTA6 can be mixed at
conforming faces. A quad face cannot meet two separate triangle faces: use
wedges to transition between compatible triangle/quad faces, or choose a mesh
with matching face topologies. User CHEXA8 and CPENTA6 elements may have warped
bilinear quadrilateral faces. Their native isoparametric geometry and magnetic
basis are retained; background solids are not split into tetrahedra. The
Jacobian must remain nonsingular with consistent orientation throughout each
cell. Folded, degenerate or numerically uncertifiable cells are rejected with
an element diagnostic. Pyramid and quadratic magnetic elements are not supported. Mesh the air inside conductor holes;
the *conductor* may be a torus but the *magnetic domain* must be filled. The
default exterior condition is zero tangential A. Increase the air domain to
check truncation error. In 3D, periodic and symmetry boundaries, infinite
elements and multiply connected magnetic domains are not implemented; planar
models have [periodic sectors and flux-normal faces](#sector-models-mperiod-and-flux-normal-faces-mbc-normal).

`MAGBC=SID` selects MBC. Its rows are `MBC,SID,FSET,TYPE,BX,BY,BZ`, and the
rows of one SID combine: one ZERO or UNIFORM row prescribes A, and, in planar
models, any number of NORMAL rows name faces the flux crosses normally (see
[below](#sector-models-mperiod-and-flux-normal-faces-mbc-normal)). On the
prescribing row FSET=0 means the rest of the exterior; an explicit FSET must
cover it. A SID with only NORMAL rows prescribes zero on the rest. A planar
[air gap](#air-gap-pairgap) solved by the air-gap element is not exterior:
FSET=0 leaves its circles out, and an FSET that selects an edge on them is an
error. Periodic sides and NORMAL faces are left out the same way. UNIFORM
prescribes the tangential trace of `A = B cross x / 2`, useful for applied
uniform fields and exact patch tests. The source perturbation still has zero
tangential exterior trace.

## Planar motor cross sections

`MAG2D,CID,DEPTH` activates a separate 2D formulation: `A = A_n n`, where `n` is
the z axis of the specified CORD2R (basic z for CID=0). All mesh nodes must be
coplanar normal to that axis; the plane may be translated or rotated. CTRIA3
and CQUAD4 can be mixed on conforming edges, using `PMAG2D` and the same MATMAG
materials. Distorted convex planar quads are supported. The solver uses scalar
nodal P1/Q1 potentials and `B = grad(A_n) cross n`; there is no vector-potential
gradient nullspace. All exterior nodes have prescribed A_n, zero by default,
except those on the circles of an [air gap](#air-gap-pairgap) solved by the
air-gap element.
UNIFORM uses `A_n = (B cross (x-x0)).n`, with x0 the first active mesh node.
Applied B and remanence must lie in the plane.

DEPTH is the uniform out-of-plane motor length, in deck length units. It
multiplies energy, linkage, inductance, force and torque, and does not change B
or H. This is an infinite-extrusion field approximation; end-turn fields and end
effects are absent. PSHELL thickness and mechanical shell kinematics do not enter
this formulation, so an explicit PMAG2D property is used.

A winding is a collection of signed axial current regions. For each region,
`J_n = SIGNED_TURNS * I / region_area`; winding area is computed from the mesh.
Each `COIL2D` is one region with a unique ID; a `CIRCUIT` joins forward and
return legs. For example:

```text
MAG2D,0,0.05
PMAG2D,10,1
MATMAG,1,1
$ CQUAD4/CTRIA3 elements reference property 10; ESETs select winding slots.
COIL2D,21,101,40
COIL2D,22,102,-40
CIRCUIT,20,,21,1,22,1
ICIRC,7,20,5
```

Each leg carries 200 ampere-turns, in opposite directions. `LOAD=7` selects the
5 A circuit current. A circuit may contain multiple disconnected slot regions;
the product of turn sign and polarity defines axial direction. Overlapping
regions are rejected. A circuit whose polarity-weighted signed turns sum to
zero has gauge-independent linkage and inductance. A single region, or any
unbalanced circuit, models an unspecified return at infinity; the field is
solved with the finite exterior boundary, and linkage/inductance are
unavailable.
A volumetric COIL cut or an in-plane CWIRE path does not define an axial current;
those cards remain exclusive to 3D.

For 2D MFORCE FACES, each FSET row contains two edge grids. Order the contour
so that `tangent cross n` points outward (counterclockwise when viewed from
+n). Use a closed contour strictly inside air; MFORCE BODY derives it instead.
Forces and torque vectors use global axes, and CID axes when given; torque
about n includes DEPTH. Both nonlinear steel and reversible PMs work in
planar mode.

```sh
python3 tools/magnetic_planar_examples.py winding /tmp/winding2d.bdf --element mixed
build/vibestran /tmp/winding2d.bdf
python3 tools/magnetic_planar_examples.py motor /tmp/motor2d.bdf --n 24 --angle 10 --current 5 --depth .02
build/vibestran /tmp/motor2d.bdf
python3 tools/magnetic_examples.py uniform /tmp/hex.bdf --element hex
build/vibestran /tmp/hex.bdf
```

The planar motor demonstration uses the same illustrative steel/PM geometry
as the 3D example, with signed slot regions and an air torque contour. Its
synthetic steel and coarse mesh do not constitute engineering validation.

### Axisymmetric models (MAGAXI)

`MAGAXI,CID` selects the axisymmetric formulation, for solenoids, voice coils
and tubular machines. The symmetry axis is the z axis of the CORD2R `CID`
(basic for 0); the mesh lies in that frame's x–z half-plane with x = r ≥ 0,
Nastran's convention for axisymmetric elements. A grid at negative radius is
an error. CTRIA3 and CQUAD4 with `PMAG2D`, MATMAG materials and `COIL2D`
windings work as in planar models; `MAGAXI` and `MAG2D` are exclusive.

```text
MAGAXI,0
PMAG2D,10,1
MATMAG,1,1.
$ GRID,id,,r,0.,z for every grid; CQUAD4/CTRIA3 in the x-z plane
COIL2D,21,101,88        $ 88 turns around the axis over ESET 101
ICIRC,7,21,2.
```

The unknown is the azimuthal potential, A = A_φ e_φ, nodal on first-order
elements (e_φ is the frame's y axis in the meridian plane), and

    B = curl(A_φ e_φ) = grad A_φ × e_φ + (A_φ / r) e_z.

Every integral (energy, residual, tangent, coil coupling, Maxwell stress)
carries 2πr in place of the planar DEPTH, so results are for the whole
body of revolution. A_φ itself, not the flux function r A_φ, is the
unknown, so a uniform axial field (A_φ = B0 r / 2) is represented exactly by
linear elements. Grids on the axis take A_φ = 0 from the formulation: an MBC
does not select axis faces (neither an FSET of a ZERO or UNIFORM row nor a
NORMAL row), and an FSET-restricted MBC covers the exterior apart from them.
`MBC ... UNIFORM` takes an axial field only (A_φ = B0 r / 2); a radial or
tangential uniform field is not axisymmetric. On the axis, where A_φ / r is
0/0, probes use its limit dA_φ/dr.

- **Windings.** `COIL2D` gives J_φ = N I / area over its meridian region.
  Each turn is a closed loop around the axis, so every winding's linkage and
  inductance are defined, a single region included (planar models need a
  balanced circuit).
- **Magnets.** Remanence must lie in the meridian plane: a rectangular axis in
  the x–z plane (AXIS 3 of basic for axial magnetization), or AXIS 1 (radial)
  or 3 (axial) of a CORD2C about the symmetry axis. Azimuthal remanence is not
  axisymmetric and is an error.
- **Forces.** MFORCE reports the axial force; radial stress cancels around the
  axis and there is no torque. A FACES contour may end on the axis, where its
  surface of revolution closes, and a BODY surface closes there by itself.
- **Not yet available** with MAGAXI: the tubular air gap (`PAIRGAP` AGE and
  BAND; METHOD=MESH works, as ordinary air), `MOTION`, `MPERIOD` and `DRIVE`.
  They are the next steps toward tubular linear motors.

HDF5 `/MAGSTAT` carries `axisymmetric = 1` (`dimension` is 2, `depth_m` 1),
and the planar blocks' nodal potential is A_φ.

### Air gap (PAIRGAP)

The air gap between a rotor and a stator is meshed like any other air: a band
of CTRIA3/CQUAD4 elements between the rotor's outer boundary and the stator's
inner boundary. Give those elements a property ID of their own and define it
with `PAIRGAP` instead of `PMAG2D`:

```text
PAIRGAP,PID,METHOD,NHARM,NBAND,NLAYER
```

| Field | Meaning |
|---|---|
| `PID` | Property ID of the gap elements. It must not also be a PSOLID, PMAG or PMAG2D ID, and at least one element must use it. |
| `METHOD` | `AGE` (blank is AGE): replace the gap elements by the air-gap element. `BAND`: replace them by a sliding band of elements (sparse; see below). `MESH`: solve them as ordinary meshed air. |
| `NHARM` | AGE harmonics, and the harmonics of the gap field reported for AGE and BAND. Blank uses half the grid count of the coarser circle; at most half the grid count of the finer one. Ignored by MESH. |
| `NBAND` | BAND: points around the band, at least 3 and a multiple of a sector model's sectors. Blank uses the finer circle's grid count (unfolded in a sector model). Ignored otherwise. |
| `NLAYER` | BAND: rings of band elements across the gap, at least 1. Blank makes the elements about square: ⌈NBAND ln(r₂/r₁) / 2π⌉. Ignored otherwise. |

There is no material field: a gap is air (relative permeability 1, no
remanence, no B–H table). Under the AGE the gap elements are not field cells,
so a COIL2D, MFORCE BODY or BFIELD ESET may not contain them and an FSET may
not name them; under MESH they are ordinary air elements. Keep such sets off
the gap if the deck is to run both ways. `PARAM,AIRGAP,AGE|BAND|MESH` overrides
METHOD on every PAIRGAP in the deck. PAIRGAP requires MAG2D.

**METHOD=MESH** is resolved when the deck is read. The PID becomes an
ordinary property of a new MUR=1 material, exactly what the deck would say
with the PAIRGAP card replaced by

```text
MATMAG,MID,1.
PMAG2D,PID,MID
```

The solver then sees no gap at all. The F06 SOLUTION section records the
substitution, e.g. `Air gap ... PAIRGAP 9 solved as meshed air (relative
permeability 1) by PAIRGAP METHOD=MESH` (or `by PARAM,AIRGAP,MESH`).

**METHOD=AGE** removes the gap elements from the field mesh and couples the
two circles that bound them with an air-gap element: the exact Laplace field
of an air annulus that matches the potential on both circles, expanded in
harmonics up to NHARM. The gap elements only mark the region. Their shape,
their number of layers and the grids inside the band do not enter the
solution; one layer of triangles between two different grid counts is enough.
The region is validated, and each failure is an error naming the PAIRGAP
and the grid, element or radii involved:

1. Every boundary edge of the gap elements is an edge of a field element.
   The gap lies between two meshed parts, not on the model's exterior.
2. The boundary does not touch itself at a grid.
3. The boundary is exactly two closed loops: one full annulus, not a sector,
   with no holes and no field elements inside it.
4. Each loop lies on a circle: every grid is within 1e-6 of the best-fit
   radius. Write GRID coordinates in large-field or free-field format when
   small fields would lose digits.
5. The circles are concentric to 1e-6 of the outer radius, and the gap has
   nonzero width.
6. NHARM, when given, is at most half the finer circle's grid count.

The two circles may have different grid counts and uneven spacing. A model
has at most one AGE gap; any number of PAIRGAPs may be solved as MESH. A
planar mesh must be one connected domain, through its elements and the AGE.
A rotor and stator with no elements between them are rejected, with a hint
to mesh the gap and give it a PAIRGAP property.

**How the AGE works.** The potential on each circle is the piecewise-linear
trace of its grids' nodal values. Closed-form transforms of the hat
functions give the trace's Fourier coefficients exactly: α on the inner
circle and β on the outer. Uneven spacing is allowed. Inside the annulus the
field solves Laplace's equation with those traces. Its energy per unit depth
separates by harmonic, with ρ = r₁/r₂ < 1:

- **Mean term:** W₀ = π (α₀ − β₀)² / (μ₀ ln(r₂/r₁)).
- **Harmonic n ≥ 1:**
  W_n = πn / (2μ₀ (1 − ρ²ⁿ)) · [(1 + ρ²ⁿ)(α_c² + α_s² + β_c² + β_s²) − 4ρⁿ (α_c β_c + α_s β_s)],
  where c and s are the harmonic's cosine and sine coefficients.

The first bracket is each circle's self energy. It depends only on each
circle's amplitude in that harmonic, √(α_c² + α_s²), not its phase. The
second is the inner-to-outer coupling, which depends on the two circles'
relative phase. Summed to NHARM and mapped back through the transforms,
these terms form a dense, symmetric stiffness block on the circle grids. It
is linear, so it adds the same term to every Newton tangent. The energy is
exact for the gap given the traces, so discretization error stays in the
meshed regions. The torque on the inner part is −dW/dδ for a rigid turn δ of
its trace, which acts only on the coupling term (see Results below).

**METHOD=BAND** removes the gap elements like the AGE and couples the two
circles through a sliding band instead: NLAYER rings of NBAND elements on a
uniform angular grid fixed in the plane. In log-polar coordinates
ξ = ln(r/r₁)/ln(r₂/r₁) the annulus is a rectangle with constant
coefficients, and each element is bilinear in (ξ, θ), so the mean (ln r)
field is exact. On each circle a band point takes the circle's
piecewise-linear trace averaged over the point's hat function. The average
is exact for the trace, damps its grid-scale content (point samples on a
grid that does not line up with the circle's alias it into the band), and
moves continuously as a part turns. The band's interior point rings are
unknowns of their own, after the grids. Each band point couples only to its
neighbours, so the gap adds a few nonzeros per grid instead of the AGE's
dense block, and the factorization stays as sparse as the meshed parts. In
the 12-slot, 14-pole motor of the xfemm comparison (39,629 unknowns) that is
170,058 lower-triangle nonzeros instead of 2,316,091.

The band is a discretization of the gap, not the exact annulus: a harmonic
n of the traces is too stiff in one layer by about (n ln(r₂/r₁))⁴/45 of the
mean term, and M layers divide that by about M². Slot and cogging harmonics
need several layers in a thin gap: cogging in the motor is harmonic 84,
with n ln(r₂/r₁) = 2.8, and a one-layer band gave 0.097 N m peak to peak for
the AGE's 0.052. The blank NLAYER, about square elements, is the guard
against this. When a part turns, the averaged traces move under the band
and the band's couplings change, so each MOTION position is a new symbolic
analysis; the factorization is cheap enough that this costs little.

Fields at points in the gap (probes, drawn gap cells) are, as with the
AGE, the annulus field of the solved traces up to NHARM harmonics: the
band's own bilinear field is only first-order accurate. The air-gap torque
is the derivative of the band's energy for a turn of the inner part's
traces with the band held, so it is the derivative of the coenergy that the
band solves.

**What each method guarantees.** These hold for every deck, and the test
suite (`tests/unit/test_air_gap.cpp`, `tests/unit/test_motion.cpp`) checks
each on a magnet rotor in a saturating stator with a winding:

| Comparison | Result |
|---|---|
| `PAIRGAP,PID,MESH`, `PARAM,AIRGAP,MESH`, and the deck with PAIRGAP replaced by MATMAG + PMAG2D air | The same equations. Every result (potentials, energies, B, H, probes, linkage, inductance, force, torque, Newton iterations) is bitwise identical with a deterministic linear solver. The default multithreaded sparse factorization varies in its last bits from run to run on any deck, so with it the three agree to round-off (measured 6e-16 in energy). |
| AGE against MESH on the same gap mesh | Two discretizations of the same gap, not identical. Both converge to one solution, and their difference falls at O(h²) in energy, linkage and inductance. |
| AGE against an analytic magnetized rotor | O(h²) in the gap field and in the magnet. The AGE adds no error of its own for the piecewise-linear traces it receives; the meshed parts set the error. |
| AGE torque against Maxwell stress on the same gap solved as MESH | Two discretizations of one torque; with a well-formed meshed gap they agree to 7e-4 at 48 divisions and converge faster than O(h²) (see the torque cross-check under [Motion](#motion-motion)). |
| MOTION at angle φ against the static deck rotated by φ | The same model, solved from a different Newton start: round-off with linear materials, Newton's tolerance with B–H curves. See [Motion](#motion-motion). |

Measured convergence, for a 20 mm rotor in a 1 mm gap with a 40 mm stator:

| Rotor divisions | AGE vs analytic, gap B | AGE vs analytic, magnet B | AGE vs MESH (4 layers), energy |
|---|---|---|---|
| 24 | 9.6e-3 | 4.6e-3 | |
| 48 | 2.4e-3 | 1.1e-3 | 1.4e-3 |
| 96 | 6.0e-4 | 2.9e-4 | 3.6e-4 |
| 192 | | | 9.8e-5 |

With saturated iron (past the end of the B–H table) and a two-slot winding at
96 divisions, AGE and MESH energies, linkages and differential inductances
agree to 2.4e-5, 2.3e-5 and 3.4e-5, in the same 9 Newton iterations. A meshed
gap's B is the gradient of a piecewise-linear potential and only first-order
accurate inside the gap cells; compare energies, linkages, inductances and
torques rather than B in gap cells.

**Cross-checking a deck.** Run it as written (AGE), then add
`PARAM,AIRGAP,MESH` and run again. The second run is the deck you would have
written without the AGE: an ordinary meshed air gap. The difference between
the two is the discretization error of the gap, and it shrinks as the
circles are refined. A meshed gap needs several element layers to be
accurate (see MFORCE below); the AGE does not, so a deck meant for the AGE
can carry a single layer and be refined in the gap only for the cross-check.
To cross-check torque, give the MESH run an `MFORCE BODY` on the rotor whose
surface runs between two rings of quads in the middle of the gap (four gap
layers and `LAYER=2`, for example). Maxwell stress on a surface against a
band of distorted triangles is poor: with two gap layers, the second a band
of triangles between different grid counts, it was 52% off the AGE torque at
48 divisions and converged only at first order, while four layers agreed to
7e-4.
In a thin gap, give the two circles similar grid spacing: the energy depends
on the difference between the two traces, amplified by about radius / gap.

```text
MAG2D,0,0.05
MATMAG,1,1.
MATMAG,2,,10
TABLEBH,10
,0.,0.,1.,200.,1.6,4000.,2.,60000.
,ENDT
PMAG2D,1,1           $ slots and other air
PMAG2D,2,2           $ rotor and stator iron
PAIRGAP,9            $ the gap band: CQUAD4/CTRIA3 elements with PID 9
$ PARAM,AIRGAP,MESH  $ uncomment to solve the same band as meshed air
```

**Results.** Probes in the annulus report the analytic gap field (H = B/μ0),
including probes between the rotor's outermost element edges and its circle.
The F06 SOLUTION section has an `Air gap` row: the PAIRGAP ID, the number of
gap elements replaced, the radii, center, grid counts, harmonics and worst
radial deviation. The HDF5 results have an `air_gap` group.

Each subcase reports the **air-gap torque**: the torque on the part inside
the gap about the plane normal through the gap's center, N m including
DEPTH (`Air-gap torque on the part inside the gap` in the F06,
`air_gap_torque_Nm` in the HDF5 subcase). The part outside takes the
opposite torque. It is the derivative of the gap energy with respect to a
rigid turn of the inner part, closed-form in the harmonics, and equals the
Maxwell stress torque on any circle in the annulus.

A `BFIELD=ALL` VTKHDF block draws the gap elements after the field cells,
with the AGE's own field: B at each element's centroid, H = B/μ0, and A at
its grids from the harmonic series. On the circles that series is the
truncated harmonics of the nodal trace, so it differs slightly from the
neighbouring cells' nodal values. An ESET block holds field cells only.
Under MESH the gap elements are ordinary field cells.

**Limits.** MBC does not apply to the AGE's circles, and MFORCE surfaces may
not touch them; the AGE gives the rotor torque instead. A model has one AGE
gap. 3D models mesh their air gap with ordinary air elements; PAIRGAP is
planar only.

### Motion (MOTION)

A planar model with an AGE air gap can turn its rotor through a sweep of
positions in one run. Name the moving elements with an ESET, describe the
sweep with `MOTION`, and set `MOTION = ID` in a subcase:

```text
MOTION,ID,ESET,CID,TYPE,START,STEP,COUNT
,SPEED
```

| Field | Meaning |
|---|---|
| `ID` | Referenced by the case control `MOTION = ID`. |
| `ESET` | The moving elements: the rotor, or the stator of an outer-rotor machine. `ESET,SID,PROP,PID...` suits meshes exported by physical group. It may not hold air-gap elements. |
| `CID` | The axis is the z axis of CID through its origin; 0 is basic z through the basic origin. It must be parallel to the MAG2D plane normal, in either direction. |
| `TYPE` | `ROTATE`. `TRANSLATE` is reserved for linear motion along the CID x axis and is an error for now (see below). |
| `START`, `STEP`, `COUNT` | Positions `START + k*STEP` degrees for k = 0 … COUNT−1, relative to the mesh as given. Positive angles turn counterclockwise looking down the CID z axis. COUNT ≥ 1 (blank is 1); STEP must be nonzero when COUNT > 1. |
| `SPEED` | Optional rotor speed in rpm, positive toward increasing positions. It only converts the back-EMF per unit speed into volts (see Back-EMF below); every position is still a static solve. |

**A reserved type.** `TRANSLATE` is part of the card so decks keep one
layout when linear motion arrives, but it is an error until it can take
effect. A value that is accepted but changes nothing would leave a deck that
reads as solved but was not. `TRANSLATE` needs a different air-gap element:
the AGE is the exact field of an annulus, and a gap that slides in a
straight line needs the field of a periodic strip.

**What moves.** Every grid of a moving element turns rigidly about the axis.
Everything defined on the moving elements turns with them: magnetization
(each moving magnet's PMAG2D frame, rectangular or cylindrical, turns with
the magnet), COIL2D regions and MFORCE surfaces. Probes stay where they are
in space, so a gap probe sees the rotor go by. The stator, the exterior
boundary and MBC stay fixed. The gap elements are not moved: they only mark
the region, and the AGE does not use their shape, so they may shear or fold.

**What is checked.** Each failure is an error naming the MOTION and the
grid, element, circle or distance involved:

1. The model is MAG2D and has a PAIRGAP solved by the AGE. Under
   `PARAM,AIRGAP,MESH` or `METHOD=MESH` the gap is meshed air, which cannot
   slide, and MOTION is an error that says so.
2. No grid belongs to both a moving and a fixed element: the moving and
   fixed parts meet only across the air gap.
3. The moving elements border exactly one of the gap's circles (the inner
   one for a rotor, the outer one for an outer rotor), and the fixed
   elements the other.
4. The axis passes through the gap's center, to 1e-6 of the outer radius,
   and is parallel to the plane normal.

**How a position is solved.** Each position is a static model: the deck
the analyst would write by hand for that angle. For position φ the solver
copies the model and makes two changes:
- it turns every grid of the moving elements by φ about the axis;
- it turns each moving magnet's PMAG2D frame by φ, as a new coordinate
  system.

If fixed elements also use the magnetized property, the moving elements get
a new PID with the turned frame, and the fixed ones keep the original. The
new coordinate systems and PIDs exist only inside the solver: results report
the deck's own IDs. The turned model is then solved by the same code as a
static subcase. Sources, probes, MFORCE surfaces and the AGE need no
motion-specific path.

**How the air-gap element sees the rotation.** Only through the moving
circle. Turning that circle's grids by φ adds φ to each grid's angle. The
hat-function transform of the trace then turns each harmonic's (cos, sin)
pair by nφ. In the harmonic energy W_n (under "How the AGE works" in
[Air gap](#air-gap-pairgap)), the self terms depend only on each circle's
amplitude, which a rotation does not change, so only the inner-to-outer
coupling term changes. Solving the turned deck is
therefore the same as holding the mesh still and rotating each harmonic's
coupling by nφ. Building the turned model makes this hold by construction,
with no second implementation of the AGE to keep consistent. Every position
has the same elements and connectivity, so each position's mesh is the first
one's with only the cells and faces on moved grids, and the air gap, placed
again; windings that do not move keep their source couplings. With the AGE
the sparsity pattern is the same too: the tangent only refactors at a new
position, and the factorization's symbolic analysis runs once per run (the
F06 `Tangent reuse` row counts it). A sliding band's pattern changes with the
position, so each position has its own analysis, which the CPU backends
order with minimum degree (AMD) in a few milliseconds.

**Where Newton starts, and what it means for the result.** Each position
starts Newton from the previous position's potential. Neighbouring
positions have nearby solutions, so a saturated sweep needs fewer
iterations. The start is the only difference between a MOTION position and
the same deck turned by hand, which starts from zero. It sets how closely
the two agree:
- **Linear materials:** the tangent is the system matrix, so Newton's first
  step solves it exactly from any start. They agree to round-off.
- **B–H curves:** Newton stops once the relative residual is below `MAGTOL`.
  Two starts stop at two points, both within that tolerance of the
  solution. They agree to Newton's tolerance, not to round-off, and a
  smaller `MAGTOL` brings them closer.

A position that does not converge from the previous solution is retried
from zero, then by current continuation, as a static subcase is.

**Torque.** At each position the AGE gives the torque on the moving part
about the MOTION axis, positive in the direction of positive positions:
the air-gap torque above, with its sign taken for the moving side and the
axis direction. It is exactly the derivative of the solved model's coenergy
with respect to the position at fixed current.

**Back-EMF.** At each position the solver reports each closed circuit's
dλ/dφ at fixed currents, in Wb per radian. That is V·s/rad, the back-EMF
per unit speed: at a speed ω the circuit sees e = ω·dλ/dφ, and with `SPEED`
the volts are reported too. It comes from the torque, by reciprocity. The
coenergy W'(i, φ) has ∂W'/∂i_c = λ_c and ∂W'/∂φ = T, so

  ∂λ_c/∂φ = ∂²W'/∂φ∂i_c = ∂T/∂i_c = (∂T/∂a) · (∂a/∂i_c).

The AGE torque is bilinear in the two circles' harmonics, so ∂T/∂a is
closed-form. ∂a/∂i_c is the solution's response to the circuit's current,
which the inductance solve at that position has already computed. So the
back-EMF costs no extra solve, and it is exact for the solved model, not a
difference between positions. It is the motional EMF at the position's
currents. When the currents also change in time (a `DRIVE` sweep at speed),
the terminal voltage adds the incremental inductance times di/dt, and the
winding's resistive drop. A static subcase with a PAIRGAP reports the same
rate for a turn of the part inside the gap about the plane normal.
| Case control | Values | Default | Meaning |
|---|---|---|---|
| `MOTION` | ID | none | The subcase solves every position of MOTION ID, with its LOAD and MAGBC at each. |
| `MSTEP` | `ALL` / `NONE` | `NONE` | Per-position tables in the F06 and HDF5. Summaries are always written. |
| `BFIELD(EVERY=n)` | `ALL` / `NONE` / ESET | every position | Field output at every n-th position, from the first. |

All three may be global or set per subcase, like the other requests;
`MSTEP` and `EVERY` affect only MOTION subcases.

**What MOTION guarantees.** Take the MOTION and PAIRGAP cards out and the
deck still runs: a static deck rotated by hand to any position, with the
gap as meshed air or with PAIRGAP restored, is the analyst's check on what
MOTION did. The test suite checks each of these:

| Comparison | Result |
|---|---|
| MOTION at angle φ against the static deck with the rotor grids and magnet frame turned by φ and no MOTION card | The same model. The only difference is Newton's start (the previous position against zero), so they agree to round-off with linear materials (measured 1e-13 in energy, coenergy, linkage, inductance, torque and probes) and to Newton's tolerance with B–H curves (measured at most 2e-8 relative in differential inductance, 4e-10 in energy and 3e-9 in torque, with a 1e-8 relative residual). |
| Torque against the coenergy | The torque is the derivative of the coenergy at fixed current: a central difference over ±0.05° agrees to 1.3e-7 with linear iron (the difference's own truncation) and 3.5e-6 with saturated iron (Newton's tolerance in the coenergy). |
| Back-EMF against the linkage | dλ/dφ from reciprocity against a central difference of each phase's linkage: 1.3e-7 of the largest rate over ±0.05° with an air stator (the difference's own truncation), 5.3e-9 over ±0.01° with saturated iron and `MAGTOL` 1e-12. A piecewise-linear B–H table makes the linkage's derivative kink wherever an element crosses a breakpoint, so a wider difference is only first order there; the reported rate is exact at the position. |
| A magnet rotor (Br 1 T, μr 1) in a uniform 0.3 T field held by MBC, over a full turn | The analytic torque m × B0 at every position, to 7.9e-4 of its amplitude at 48 rotor divisions and 2.0e-4 at 96: O(h²). |
| AGE torque against Maxwell stress (`PARAM,AIRGAP,MESH` and MFORCE BODY on the rotor, four gap layers, `LAYER=2`) | 7.4e-4, 1.4e-4 and 1.1e-5 apart at 48, 96 and 192 rotor divisions with a wound iron stator. |
| A round, unmagnetized iron rotor in a wound stator | Every position the same to round-off and zero torque, between mesh pitches too: the rotor polygon's shape is above the gap's harmonics. |
| A magnet rotor in a two-slot stator without current | Cogging torque and energy repeat every 180° and the winding's linkage reverses, to round-off: the magnet frame turns with the rotor. |

```text
SUBCASE 1
  LABEL = COGGING
  MOTION = 1
  MSTEP = ALL
SUBCASE 2
  LABEL = 5 A
  LOAD = 7
  MOTION = 1
  BFIELD(EVERY=4) = ALL
BEGIN BULK
MAG2D,0,0.05
PAIRGAP,9                           $ the meshed gap band, PID 9
ESET,20,PROP,2,3                    $ rotor iron and magnets
MOTION,1,20,0,ROTATE,0.,.5,121      $ 0 to 60 degrees in 0.5 degree steps
...
```

**Results.** The F06 SOLUTION section gains a `Motion` row (the moving
ESET's elements and grids, the axis, which circle turns and the positions)
and, per MOTION subcase, the positions, total Newton iterations, their range
per position and the largest residual. The subcase block gives the torque's
mean, peak-to-peak and ripple (peak-to-peak as a percentage of the mean's
magnitude). When no source carries current at any position it gives the
cogging peak-to-peak instead of a ripple: cogging averages to zero over its
period, so its mean is only sampling residue and no base for a percentage. With `MSTEP = ALL` it adds a table of position, torque,
energy, coenergy, Newton iterations, residual and each closed circuit's
linkage. The HDF5 subcase group holds the same summary (see
[Results](#results-and-angle-sweeps)), and with `MSTEP = ALL` every
per-position value.

In `name.mag.vtkhdf`, MOTION fields make the collection a time series whose
time is the position in degrees, which ParaView animates with its time
controls: the moving points turn, and the gap elements show the AGE field
in the fixed frame.

**The time axis.** A VTKHDF file has one time axis. VTK's reader takes the
time values from one block, and every block must have its own `Steps` group
with the same number of steps. A block without one breaks the axis. So every
subcase's block uses one shared axis: every position written by any
subcase, in increasing order. At each time, a block shows its state at the
largest of its own written positions at or before that time, or its first
position before any of them. In effect, a block holds its last state
between its own positions. Some consequences:
- A static subcase, and each source block, show their one state at every
  time.
- Two subcases on one MOTION share their times exactly.
- Subcases on different sweeps each change only at their own positions.
- A position that `BFIELD(EVERY=n)` skips is not on the axis unless another
  subcase writes it.

A held state is not copied. The topology is written once, and each time
step's offsets point at the state it shows.

**Limits.** One moving part (one AGE gap); rotation only; planar only.
Currents are fixed by LOAD at every position unless `DRIVE` sets the phase
currents from the position (next section). A probe in the moving part stays
in space while the rotor's cells pass under it.

### Phase currents (DRIVE)

A `DRIVE` gives the phase circuits of a MOTION subcase currents that follow
the rotor, as a current-controlled drive does, and reports the machine in dq
axes. Set `DRIVE = ID` in a subcase that also has `MOTION`:

```text
DRIVE,ID,WAVE,POLES,IPEAK,GAMMA,THETA0
,C1,C2,C3,...
```

| Field | Meaning |
|---|---|
| `ID` | Referenced by the case control `DRIVE = ID`. |
| `WAVE` | `SINE` for sinusoidal currents, or `SIX` for 120° six-step (block) currents, which needs three phases. |
| `POLES` | Pole pairs p, at least 1. |
| `IPEAK` | Peak phase current I in A, at least 0. For `SIX` it is the block's amplitude. |
| `GAMMA` | Current angle γ in electrical degrees, from +q towards −d: i_d = −I sin γ, i_q = I cos γ. γ = 0 is pure q-axis current; γ > 0 weakens the field. Blank is 0. |
| `THETA0` | The rotor position, in degrees, at which the rotor's d axis lines up with phase 1's axis. Blank or `AUTO` finds it (see below). |
| `C1 … Cm` | The phase circuits in phase sequence, at least three: CIRCUIT IDs, or COIL2D IDs that belong to no CIRCUIT. Each must be closed. |

Fields 7 and 8 of the first line must be blank; the phases start on the
continuation.

**The currents.** At position φ the electrical angle is θe = p(φ − THETA0).
Phase k (k = 1 … m) is at θ_k = θe − 2π(k−1)/m and carries the terminal
current

  i_k = i_d cos θ_k − i_q sin θ_k            (`SINE`)

The same current is I cos(θ_k + γ + 90°). `SIX` conducts +I while that
sinusoid is within 60 electrical degrees of its positive peak, −I within 60°
of its negative peak, and nothing in the 60° between. So at every angle two
phases conduct, in the same sequence, and the fundamental of a phase's
current is (2√3/π) I ≈ 1.103 I, in phase with `SINE`'s. A phase circuit
takes its current from the DRIVE alone: an ICIRC on it in the subcase's LOAD
is an error. Other circuits, such as a field winding, keep their LOAD
current at every position.

**THETA0 AUTO.** The d axis is where a phase links the most magnet flux.
With `THETA0` blank, the solver finds it from a no-load pre-sweep:

1. It solves eight positions over one electrical period, 360·j/(8p) degrees
   (j = 0 … 7) from the mesh as given, with no current in any circuit and
   the subcase's MAGBC. These are ordinary MOTION positions.
2. It fits each phase's no-load linkage with its first harmonic,
   λ_k ≈ λ_pm cos(pφ − ψ_k), by the discrete Fourier transform of the eight
   samples.
3. THETA0 = ψ_1/p, in [0, 360/p) degrees, so phase 1's no-load linkage
   peaks at θe = 0.
4. It checks the phase sequence: each phase k must lag phase 1 by
   360(k−1)/m electrical degrees, to within half the phase spacing.

If the phases lag the other way, the phase sequence opposes the direction
of increasing position, and the error says so, with "swap C2 and C3" for
three phases. Phases out of sequence in any other way are an error that
lists each measured lag. A phase 1 with no position-dependent no-load
linkage has no d axis to find: a rotor without magnets, for example. That
is an error too, asking for THETA0.

The F06 `Drive` row reports the THETA0 found (nine digits), phase 1's
fundamental λ_pm and each phase's lag. The pre-sweep runs once for each
DRIVE, MOTION and MAGBC, and every subcase that needs it reuses it.
Eight samples alias the 7th and 9th harmonics of the linkage onto the
fundamental, which shifts the THETA0 found by their share. A winding whose
linkage carries those harmonics may want its THETA0 given instead. Either
way the dq results use exactly the THETA0 reported, so giving that value
reproduces the subcase.

**The dq quantities.** Every DRIVE position reports the amplitude-invariant
Park transform of the phase values. With C the m × 2 matrix of rows
(cos θ_k, −sin θ_k), so that the phase values of d and q components are
C·(d, q):

- currents: (i_d, i_q) = (2/m) Cᵀ i;
- linkages: (λ_d, λ_q) = (2/m) Cᵀ λ, using the phase circuits' linkages;
- **incremental inductance:** L_dq = (2/m) Cᵀ L C, from the phase circuits'
  differential (tangent) inductance L at the position. It is the dq
  response to a small change in current.
- **apparent inductance** (`LDQ = APPARENT`, the default): the same
  transform of the frozen-permeability inductance. Every element's
  permeability is frozen at its secant value at the loaded state,
  μ = |B|/|H| (a linear material keeps its own), and the magnets are
  switched off.

  The loaded state solves this frozen linear problem exactly: at the
  converged B every element's H is the secant times B. So by superposition
  its linkage is the frozen inductance times the currents plus what the
  magnets (and any MBC field) link at that permeability. The solver reports
  that rest as the magnet linkage, (λ_m,d, λ_m,q) = λ_dq − L_app,dq i_dq.
  This is the usual frozen-permeability split, λ_d = λ_m,d + L_d i_d +
  L_dq i_q, with L_d and L_q defined at every current angle, γ = 0 and 90°
  included. The cost is one extra factorization of the secant stiffness
  per position, solved for the m phases together. `LDQ = INCREMENTAL`
  skips it.

With linear materials the secant is the tangent, so the apparent and
incremental inductances are the same.

**Torque in dq.** The AGE torque at each position is the reference. In a
machine whose phases link exact sinusoids and whose inductances do not
depend on position, it equals (m/2) p (λ_d i_q − λ_q i_d). Slotting,
saturation and winding harmonics make the two differ, and the AGE torque
includes them.

**Back-EMF over an electrical period.** When the positions sample whole
electrical periods uniformly (COUNT·|STEP|·p a multiple of 360° and at least
two samples per period), each phase's back-EMF per unit speed (see
[Motion](#motion-motion)) is Fourier analysed over the sweep. The summary
gives its fundamental, the back-EMF constant in V·s/rad (peak), and the
total harmonic distortion of the electrical harmonics below the sampling
Nyquist rate. With `SPEED` it also gives the fundamental's RMS in volts.
Use IPEAK 0 for the no-load (open-circuit) back-EMF; at load the same figures
are the motional EMF at the drive's currents.

| Case control | Values | Default | Meaning |
|---|---|---|---|
| `DRIVE` | ID | none | Phase currents from DRIVE ID at each MOTION position. The subcase needs `MOTION`. |
| `LDQ` | `APPARENT` / `INCREMENTAL` | `APPARENT` | Whether a DRIVE subcase also reports the frozen-permeability L_dq and magnet linkage. |

**What DRIVE guarantees.** Take the DRIVE card out and the deck still runs:
each position as a single-position MOTION subcase, with ICIRC currents equal
to the phase currents the DRIVE set there. The test suite
(`tests/unit/test_drive.cpp`) checks each of these:

| Comparison | Result |
|---|---|
| A DRIVE position against the static deck at that position with the same ICIRC phase currents | The same model, from a different Newton start: round-off with linear materials, Newton's tolerance with B–H curves. Measured with an air stator 5e-16 in energy, 2e-13 in torque, 2e-14 in linkage and 2e-16 in inductance; with saturated iron 7e-11 in energy, 9e-9 in torque, 5e-8 in linkage (relative to 1 mWb), 4e-9 in inductance and 9e-10 in back-EMF. |
| THETA0 AUTO against the THETA0 it reports | The same subcase (3e-15 in energy). |
| THETA0 AUTO on a uniformly magnetized two-pole rotor in an air stator | The magnet's axis to round-off, and lags of exactly 120° and 240°. A uniformly magnetized 48-gon rotor of unit permeability has no harmonics between 2 and NHARM on the gap circle, so each phase links an exact first harmonic. |
| The same sinusoidal machine with SINE currents at γ = 0°, 30° and 90° | To 1e-13 at every position: the AGE torque equals (3/2) p (λ_d i_q − λ_q i_d) and is constant at (3/2) p λ_pm I cos γ; L_d = L_q; the apparent inductance equals the incremental one; the magnet linkage is λ_pm on d; the back-EMF per unit speed is −p λ_pm sin θe. Over one period its fundamental constant is λ_pm with 3e-13 % distortion. |
| SIX currents in the same machine | Two phases at ±I at every position, and the AGE torque is (3/2) p λ_pm i_q of the six-step currents (3e-13). |
| Apparent inductance, saturated iron, against a linear deck with every iron element given its own MATMAG at its loaded secant permeability and the magnets off | The same inductance (3e-16, on a triangle mesh, where B is uniform in each element). With the magnets on and the DRIVE currents that deck reproduces the loaded linkage (1e-10), and without current its linkage is the reported magnet linkage (7e-11). |

```text
SUBCASE 1
  LABEL = NO LOAD
  MOTION = 1
  DRIVE = 11
SUBCASE 2
  LABEL = RATED
  MOTION = 1
  DRIVE = 10
  MSTEP = ALL
BEGIN BULK
ESET,20,PROP,2,3                    $ rotor iron and magnets
MOTION,1,20,0,ROTATE,0.,1.25,72     $ one electrical period of a 4-pole-pair rotor
,3000.                              $ rpm, for the back-EMF in volts
DRIVE,10,SINE,4,20.,30.             $ 20 A peak, gamma 30 deg, THETA0 AUTO
,1,2,3                              $ phase circuits in sequence
DRIVE,11,SINE,4,0.                  $ no load: back-EMF and magnet linkage
,1,2,3
...
```

**Results.** The F06 SOLUTION section gains a `Drive` row per DRIVE (wave,
pole pairs, current, γ, phases, THETA0) and, for THETA0 AUTO, a row with the
pre-sweep and what it found. The subcase block adds the DRIVE line, the mean
i_d, i_q, λ_d and λ_q over the positions, the mean incremental and apparent
L_d, L_q and L_dq, the mean magnet linkage, and each phase's back-EMF
constant and distortion when the sweep covers whole electrical periods.
With `MSTEP = ALL` there is also a table of θe, i_d, i_q, λ_d, λ_q and the
inductances at each position. Every MOTION subcase with a closed circuit
also gets a table of each circuit's dλ/dφ, or EMF in volts with `SPEED`.
The HDF5 subcase group gains a `drive` group (see
[Results](#results-and-angle-sweeps)).

### Worked example: a 12-slot, 14-pole motor

`tools/magnetic_motor_example.py` is the reference workflow from an external
mesher to plots. It builds a surface-magnet motor with a gmsh cross section:
12 tooth-wound slots, 7 pole pairs, 1.2 T magnets and the nonlinear steel table
from the script. It writes the MAGSTAT deck (PAIRGAP air gap, MOTION, DRIVE,
720 mid-gap MPROBEs), runs vibestran, and reads `motor.mag.vtkhdf` for the
plots:

```sh
.venv/bin/pip install -r tools/requirements-magnetic.txt gmsh matplotlib
.venv/bin/python tools/magnetic_motor_example.py /tmp/motor
```

There are three subcases:

1. Rated: 10 A peak at γ = 0, over one electrical period in 120 positions.
2. No load: `DRIVE` at 0 A, for THETA0 AUTO's d axis, the back-EMF at
   1000 rpm and the incremental L_d, L_q.
3. Cogging: 40 positions over one cogging period, 360/lcm(12, 14) degrees.

The script reads only `element_id`, B and A from the VTKHDF file. It reads
torque, linkage, EMF and probes from the HDF5 subcase groups. `--reuse`
replots an earlier run without solving. On the mesh it makes (20 500 cells, the
3450 in the gap replaced by the AGE), the run takes about 2.5 minutes on 8 threads.

The results are self-consistent where the theory gives an exact relation:

| Check | Result |
|---|---|
| Rated torque against (3/2) p (λ_d i_q − λ_q i_d) from the mean dq quantities | 4.9201 N·m from the AGE; 4.9200 N·m from the dq relation |
| No-load back-EMF constant against p λ_pm | 0.33200 V·s/rad against 7 × 0.047422 = 0.33195 |
| Phase balance | The phases' back-EMF constants agree to 2e-5; the lags are 120.00° and 240.00° |
| Cogging from the no-load sweep (120 positions per electrical period) against the cogging sweep (40 per cogging period) | 0.0598 and 0.0607 N·m peak-to-peak; the coarser sweep falls between the peaks. The cogging sweep's mean is −1.2e-4 N·m, 0.2 % of its peak-to-peak |

These checks show that the AGE torque, the linkages and the dq post-processing
are consistent with each other. They are not a validation against an
independent solver or a test bench; the dimensions and steel are illustrative.

### Sector models (MPERIOD) and flux-normal faces (MBC NORMAL)

Most machines repeat around the shaft. In a 12-slot, 14-pole motor, turning
the cross section by 180° takes every pole to one of opposite polarity, and
every tooth coil to a coil of the same phase wound the other way. The field
is then anti-periodic, A(turned x) = −A(x). In a 12-slot, 8-pole motor it
repeats every 90°. One sector carries all the information. `MPERIOD` says
which sector the mesh is and how it repeats:

```text
MPERIOD,ID,TYPE,FSET1,FSET2,CID,SECTORS
```

| Field | Meaning |
|---|---|
| `ID` | Identifies the card in messages and results. A deck has at most one MPERIOD. |
| `TYPE` | `PERIODIC`: A is the same at a grid and its image. `ANTI`: A changes sign. |
| `FSET1`, `FSET2` | The two sides of the sector, as edges of field elements on the exterior. FSET2 must be FSET1 turned counterclockwise by 360/SECTORS degrees about the axis, grid for grid and edge for edge. |
| `CID` | The axis is the z axis of CID through its origin (blank or 0: basic z through the basic origin). It must be parallel to the MAG2D plane normal. |
| `SECTORS` | The number of sectors in the whole machine, at least 2; even for ANTI. |

MPERIOD requires MAG2D. The sides need no particular shape: they are
usually radial lines, but a skewed or stepped side works as long as its
turned copy is the other side.

**What the sector stands for.** The sector model is the whole machine: N
turned copies of the sector's mesh, meshed alike at the sides, with each
magnet's remanence and each COIL2D's turns multiplied by the period's sign
in the odd copies of an ANTI model. Each CIRCUIT is in series with its images
in every copy. That deck is the plain equivalent: removing MPERIOD from the
sector deck does not give it. Without the card the sides become ordinary
exterior, with A = 0, and the deck describes a different machine.

**How it is solved.** Each grid of FSET2 is the image of a grid of FSET1,
and its potential is the image's sign (+1 or −1) times that grid's. The
gauge gives the pair one unknown: the image grid's rows and columns are added
to its partner's, times the sign. The sides take no prescribed potential and
no load of their own. A grid on both sides lies on the axis and is its own
image: PERIODIC leaves it free, and ANTI holds it at A = 0. Grids where the
sides meet the prescribed exterior (the stator's outer circle, say) are
prescribed on both sides.

**The air gap in a sector.** The gap elements form a sector of the annulus,
and their arcs run from a grid of FSET1 to its image on FSET2. The AGE
unfolds each arc around the machine: copy j of each arc grid stands at its
angle plus j sectors, with the factor (−1)^j for ANTI. It then builds the
whole machine's gap from these circles exactly as for a full model. The
unfolded trace repeats, so its Fourier coefficients vanish except at
multiples of SECTORS (PERIODIC) or odd multiples of SECTORS/2 (ANTI): the
sector carries only the harmonics the period allows. The default NHARM is
the whole machine's, half the unfolded coarser circle's grid count. The
whole machine's gap stiffness is folded back onto the sector's grids, with
the signs, and divided by SECTORS: the sector's share of the gap energy.

**Results are the whole machine's.**

| Quantity | In a sector model |
|---|---|
| Energy, coenergy | SECTORS times the sector's |
| Circuit and source linkage, differential inductance, apparent and incremental L_dq | SECTORS times the sector's. A source is in series with its images, which carry its current times the sign and link the same flux. |
| Air-gap torque, linkage rates (back-EMF) | From the unfolded gap: the whole machine's directly |
| DRIVE dq quantities, THETA0 AUTO | From the whole machine's linkages and inductances |
| B, H, potentials, VTKHDF fields | The modeled sector's |
| MPROBE | Points in the sector's cells; in the gap, any angle (the unfolded gap's field) |
| MFORCE | The named body in the sector, unscaled. Its surface may not reach a periodic side, which is on the mesh's exterior. Use the AGE torque for the machine. |

A COIL2D region on its own, or a circuit whose turns do not balance, is
open in an ordinary planar model: its return is unspecified, and linkage is
unavailable (see [Planar motor cross sections](#planar-motor-cross-sections)).
In an ANTI model its images alternate in sign, so the whole machine's
series circuit balances, and linkage and inductance are defined. A phase of
a half model of a 12-slot, 14-pole motor holds the go sides of its coils in
one half and the return sides in the other; it is closed. In a PERIODIC
model the images carry the same turns, so a circuit closes only if it
balances within the sector.

**MOTION in a sector.** Each part's sides pair within the part: the rotor's
first side with the rotor's second, the stator's with the stator's. MOTION
turns the rotor's sides with it, and the pairing is rebuilt at each position
from the turned grids. A grid paired across the air gap, moving with fixed,
is an error. DRIVE and THETA0 AUTO need nothing further: the electrical
angle and dq quantities use the whole machine's linkages.

**What is checked.** Each failure is an error that names the grid, edge,
element or angle involved:

1. Each grid of FSET1, turned by 360/SECTORS, lands within 1e-6 (of the
   largest distance of a side grid from the axis) of a grid of FSET2. Each
   FSET2 grid is the image of exactly one FSET1 grid, and each FSET1 edge
   turns onto an FSET2 edge.
2. The sides are on the exterior, share no edge, and do not touch the air
   gap. An MBC row may not select them.
3. A grid and its image are both on the prescribed exterior or both off it.
   Where both are prescribed, the prescribed values agree with the period.
   ZERO always does. UNIFORM does only when the uniform field is periodic
   itself, as about the origin under ANTI with two sectors.
4. The gap's arcs run from FSET1 to FSET2, span exactly one sector, and are
   centered on the axis.
5. Some exterior face is prescribed, unless the model is ANTI: otherwise A
   would be fixed only up to a constant.

**Flux-normal faces (MBC NORMAL).** The exterior condition is otherwise
flux-parallel (B·n = 0, A prescribed). It is the right condition on a mirror
line across which A is odd. Where A is even, as on the line between two
like poles or through the middle of a magnet magnetized across it, the flux
crosses the line normally. There n × H = 0, the natural condition of the A
formulation: leave A free. List such edges in an FSET and add a NORMAL row
to the subcase's MBC:

```text
MAGBC = 1
...
MBC,1,0,ZERO                         $ prescribe A = 0 on the rest
MBC,1,7,NORMAL                       $ FSET 7: the mirror line, flux normal
```

NORMAL faces take no prescribed value and no boundary load, and the gauge
leaves their grids free. Subcases whose MBCs have different NORMAL faces
get gauges of their own. A NORMAL row needs an explicit FSET, and its
faces must be exterior, off the air gap and off periodic sides. NORMAL is a
boundary condition, not a statement about the rest of the domain: results
are the modeled part's, unscaled. Its plain equivalent is the whole
mirrored domain, and the half model's energy and linkage are half of that
domain's.

**Guarantees.** Each is a unit test in `tests/unit/test_periodic.cpp`, on
models whose plain equivalent is built from the same grids and elements:

| Sector model | Against | Measured |
|---|---|---|
| ANTI half (2 sectors) of a 12-slot, 14-pole surface-magnet motor, linear iron, three phase currents | The whole machine | Potential 8e-14, B 3e-14, energy and coenergy 1e-15, AGE torque 1e-14, circuit linkage 4e-15, inductance 4e-15, back-EMF per unit speed 4e-15, and gap probes in every sector 6e-15 |
| The same, saturating steel driven to 2.1 T, MAGTOL 1e-12 | The whole machine | Potential 6e-13, B 5e-13, energy 9e-15, torque 1e-15, linkage 3e-15, inductance 4e-15 |
| PERIODIC quarter (4 sectors) and half of a 12-slot, 8-pole motor | The whole machine | Every quantity above within 1.3e-13; torque 2e-14 |
| ANTI half with MOTION (6 positions) and DRIVE THETA0 AUTO | The whole machine's sweep | THETA0 7e-15 degrees; torque 1e-14, dq linkages and both L_dq 3e-15, and back-EMF 5e-15 at every position |
| Upper half of a mirror-symmetric domain (coils, iron, a magnet across the line) with MBC NORMAL on the mirror line | The whole domain | Potential 2e-15, B 6e-16, twice the energy 3e-15, twice the linkage 2e-16 |

Agreement is to round-off with linear materials and to the Newton tolerance
with B–H curves. The discrete sector problem is the whole machine's problem
restricted to fields with the symmetry. The whole machine's discrete
solution has that symmetry, because its mesh and sources do.

**Results.** The F06 SOLUTION section gains a `Periodic` row: type,
sectors, axis, sector angle, the number of grid pairs and axial grids, and
which quantities are the whole machine's. In a sector model the `Air gap`
row adds the arcs' grid counts and the harmonics that remain. The HDF5 file
gains `MAGSTAT/periodic` (see [Results](#results-and-angle-sweeps)).

Full 3D magnetic shells are a separate physical model: a thin magnetic layer or
surface-current sheet coupled to the surrounding 3D field. Mechanical shell
elements cannot simply be given the planar A_n equation on an arbitrary curved
surface. That thin-layer/sheet formulation is not implemented here. For an
example of the distinction, see COMSOL's
[magnetic shielding thin-layer formulation](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_multiphysics.20.71.html).

## Numerical formulation

Lowest-order [Nedelec edge elements](https://defelement.org/elements/nedelec1.html)
represent 3D A, with covariant Piola mapping and consistently mapped curls.
Tetrahedral B is constant; hex/wedge B varies within a cell. Material response,
energy and tangent are integrated at volume quadrature points. Probes evaluate
the actual local field. Face quadrature uses the bilinear surface, local area
measure and pointwise normal, including in Maxwell stress and enclosed-volume
checks. Jacobian validation bounds the determinant over the whole reference
cell using Bernstein coefficients and subdivision (see
[Johnen, Weill and Remacle](https://arxiv.org/abs/1706.01613)), rather than relying
only on corner or Gauss-point samples. Exported cell
fields are volume averages. A boundary-contracted spanning tree removes the gradient nullspace. The
reduced curl-curl system is SPD for positive reluctivity. Nonlinear solves use
the consistent material tangent, residual line search and, on failure, an
eight-increment current continuation retry. Failure to meet the requested
residual is an error, not a partial result.

The tangent systems go through the structural solver backend (`--backend`).
One factorization serves the whole run: its symbolic analysis is done once
per sparsity pattern, and Newton steps, continuation stages, subcases and the
differential inductances only refactor values. The pattern holds across
MOTION positions with the AGE; a sliding band couples each grid to the band
points under it, so the pattern, and the analysis, change at every position.
The CPU backends order planar tangents with minimum degree (AMD): at their
sizes its fill is within about 15% of nested dissection's and it orders
several times faster. cuDSS keeps nested dissection, which its GPU
factorization needs. Each subcase's inductances are a single
solve with one right-hand side per closed source and circuit. Every
factorization must be positive definite: a failed Cholesky is an error, never
an LU fallback. `--backend=cuda` (cuDSS) reuses its analysis the same way; in
fp32 each solve is refined in double, and the factor switches to fp64 when
that cannot converge. The PCG backends work but converge poorly on
high-contrast reluctivity. `--refine`, `--checkmode`, `--no-quality-checks` and
`--dump-system` are structural controls and are rejected.

Each 3D current source uses lowest-order
[Raviart–Thomas face fluxes](https://defelement.org/elements/raviart-thomas.html),
with contravariant Piola mapping for hexes/wedges. Cell balance,
insulating side walls and unit cut flux constrain a minimum-resistive-energy
solve. Closed-source preprocessing removes the redundant global cell balance.
The stranded projection retains these constraints. N multiplies the resulting
unit-current density exactly once. Matched coil regions integrate directly on
their reference cells, including warped cells and cuts.

Wire and magnetic meshes need not match. Affine wire cells on convex planar-faced
backgrounds use polyhedral intersections to transfer `integral N.J dV`. Disjoint
bounding boxes are rejected before clipping. Polynomial products use exact
four-point degree-two (tet field) or five-point degree-three (affine hex/wedge
field) tetrahedral rules. The latter includes a negative centroid weight.
Nonaffine field cells on this clipping path retain adaptive Gauss orders 6, 8,
10, 12, then 14, checking current continuity and relative coupling change (1e-8).

For nonaffine wire cells or backgrounds containing warped cells, source-reference
integration lines are split at the actual field faces, including both crossings
of a saddle face. Nested adaptive Gauss integration also splits the transverse
coordinates at source/field boundary and edge crossings. Tetrahedral integration
rays follow the cell edge best aligned with its current, avoiding tangencies
introduced by arbitrary subdivision diagonals. The RT0 current coefficients are
cached once per cell; reference-domain integration applies the Piola map directly
without solving for source coordinates again at every sample.

Adaptive error estimates, coverage checks and discrete current continuity retain
their existing tolerances. Transfer retries more tightly on a continuity failure;
it does not project the load to conceal one. Geometric interval bounds detect
material overlaps and exterior crossings even when quadrature misses them.
Curved integration/contact has finite refinement budgets; unresolved cases return
a diagnostic requesting refinement. An independent clipping implementation remains
available for cross-checking ray integration on planar-faced meshes.

Checks cover per-cell current conservation, source
volume coverage and orthogonality of the transferred load to interior nodal
gradients. Conservation residual tolerances are 1e-10, transfer 1e-9, and relative
coverage 1e-8.

Closed-source linkage is the same coupling vector dotted with A; differential
inductance uses the converged tangent solve. This ensures discrete energy and
reciprocity consistency. Open-source linkage and inductance are unavailable
(NaN in the HDF5 results) because they depend on an unspecified return path.

MFORCE integrates Maxwell stress on a closed outward-oriented mesh-face surface
with mu_r=1, remanence-free material on both sides, reporting force and torque
about the origin of CID (basic for 0), in basic and in CID axes. The result is
the force on everything the surface encloses. `MFORCE,ID,FACES,FSET,CID`
integrates an explicit FSET surface.

`MFORCE,ID,BODY,ESET,CID,LAYER` derives the surface from a body element set.
Layer 1 is every cell sharing a grid with the body, and each further layer
(LAYER defaults to 1) the cells sharing a grid with the previous one. The
surface is every face with exactly one side in the body plus its layers,
oriented outward. It includes cavity walls, so a stator's surface has a
wall in the air gap and excludes the rotor. Every layer cell, and every cell
just outside the surface, must be nonmagnetic (MUR=1, no table, no
remanence); plastic and potting qualify. Added layer cells must also be free
of COIL/COIL2D regions and CWIRE conductor volume, so growing the surface
cannot include the force on an unrelated winding. To include a winding's
force, select its elements in the body ESET; for CWIRE, select all field cells
containing the swept conductor. Winding checks apply even when the source
has zero current in a subcase. A winding outside the surface may touch it,
but may not extend into the added layers. A magnetic cell sharing a
grid with the body is touching it: Maxwell stress cannot separate the force
between touching parts, so add it to the body set (rotor = shaft + core +
magnets) or mesh a gap. A magnetic cell further out means the gap is too
thin for LAYER, and a body within LAYER cells of the domain exterior is
rejected; request the opposite body instead. The body need not be connected
or manifold. A disconnected set reports the total and, in the HDF5 `piece_*`
datasets, the force on each connected piece.

```text
ESET,70,PROP,21,22,23       $ shaft, core and magnet properties
MFORCE,1,BODY,70,100,2      $ torque about the rotor frame origin
```
 A face may bound a winding but must not cut through current: a face
with COIL or COIL2D cells on both sides, or one crossing a CWIRE conductor
(including along its internal section faces), is rejected with the face,
element and source named. Keep the surface clear of material interfaces. The
field is averaged from its two adjacent cells before stress evaluation. Vary
the surface and refine the mesh: force and torque generally converge more
slowly than field values. Thin air gaps need several element layers.

### Reusing a factor (PARAM,MAGPCG)

`PARAM,MAGPCG,N` solves a Newton step whose tangent has the pattern of the
factored one by conjugate gradients, preconditioned with that factor, instead
of factoring it (`TangentSolver`, `include/solver/tangent_solver.hpp`). A
step is inexact: CG stops at a relative residual eta, the second forcing term
of Eisenstat and Walker, at most 0.1 and tightening with the residual's
decrease so that Newton stays superlinear, but never tighter than half the
remaining distance to `MAGTOL`. Newton's convergence test is unchanged, so
the results are those of factoring every tangent to `MAGTOL`. A step whose CG
does not converge in N iterations (or breaks down) refactors with its own
tangent and solves directly; a new pattern is always factored. The
differential inductances and linkage rates stay direct solves. The F06
`Tangent reuse` row counts the CG solves, their iterations and the steps
that stopped CG and refactored.

It pays only where a numeric refactorization costs many solves with the
factor. On planar models it hardly does: a stale factor needs about 10 CG
iterations per step (saturation changes the reluctivity of many cells from
one step to the next, which is not a low-rank change), each a solve and a
product with the tangent, while a refactorization with the analysis reused
costs 2 (Accelerate) to 12 (CHOLMOD) solves. The whole motor of the xfemm
comparison with `PARAM,AIRGAP,BAND` (MAGTOL 1e-10, 78 positions):

| model, backend | factor every tangent | MAGPCG,20 |
|---|---:|---:|
| coarse (n = 12,509), Accelerate (M-series Mac) | 6.1 s | 8.3 s |
| coarse, CHOLMOD (Ryzen 7 5700, 8 threads) | 15.0 s | 13.7 s |
| coarse, cuDSS (TITAN V) | 13.5 s | 15.0 s |
| fine (n = 46,901), CHOLMOD | 64.4 s | 69.1 s |
| fine, cuDSS | 47.0 s | 51.7 s |

The torques agree to 1e-10 (relative). On the fine model 352 steps were solved
by CG in 4,540 iterations, 13 each, and 80 more stopped CG and refactored;
factorizations fell from 586 to 284.

It is meant for models where factoring dominates: 3D MAGSTAT, whose factors
fill much more than planar ones, and, through the same interface, future
geometric nonlinear structural solves.

## Results and angle sweeps

For `name.F06`, the solver also writes `name.mag.vtkhdf`, a single HDF5
file. SOL MAGSTAT needs a build with HDF5 (the meson `hdf5` feature, on
automatically when `libhdf5-dev` or Homebrew `hdf5` is installed); without it,
structural solutions are unaffected and MAGSTAT decks stop with an error before
solving. The file has two top-level groups:

- `/VTKHDF`: fields in the VTKHDF format, which ParaView opens directly
  (File > Open; the `.vtkhdf` extension selects its VTKHDF reader). It is a
  PartitionedDataSetCollection whose assembly has a `fields` node with one
  `SUBCASE_<ID>` block per subcase, and a `sources` node with one `COIL_<ID>`
  or `WIRE_<ID>` block per source mesh. Field blocks carry unsmoothed cell
  `B_T`, `H_A_per_m` and `element_id`; planar blocks also carry nodal
  `A_normal_Wb_per_m`. Source blocks carry `J_per_ampere_per_m2`.
- `/MAGSTAT`: SI scalar results, ignored by ParaView. Attributes give `schema`,
  `units`, `dimension`, `depth_m` and `axisymmetric`. Definition tables, one row per item in
  ID order:
  - `sources/`: `id`, `wire`, `closed`, `conservation_error`,
    `transfer_error`, and `wire_elements` (element ID, traversal sign) with
    `wire_element_offsets` giving each source's rows.
  - `circuits/`: `id`, `parallel`, `closed`, and `members` (source ID,
    polarity) with `member_offsets`.
  - `forces/`: `id`, `body`, `cid`. `probes/`: `id`, `cd`, `position_m`.
  - `air_gap/` (planar models with a PAIRGAP solved by the AGE or BAND):
    attributes `property`, `inner_radius_m`, `outer_radius_m`, `harmonics`,
    `band_points` and `band_layers` (0 for the AGE), `radial_deviation`;
    `center_m`; `sectors` (1, or MPERIOD's SECTORS); and `inner_grids`,
    `outer_grids`, the grid IDs on each circle in increasing angle (in a
    sector model, each arc grid once, without the last, which is the image
    of the first).
  - `periodic/` (with MPERIOD): attributes `id`, `type` (`PERIODIC` or
    `ANTI`), `sectors`, `sector_angle_deg`; `origin_m` and `axis`; and
    `first_grids`, `second_grids` (row k pairs a grid with its image) and
    `axial_grids`. Energies, linkages, inductances, back-EMF and the AGE
    torque in the subcase groups are the whole machine's.
  - `subcase_ids`, and `subcases/<ID>/` per subcase, with attributes `id`,
    `iterations`, `residual`, `energy_J`, `coenergy_J`, `bh_extrapolated`,
    and `air_gap_torque_Nm` with an AGE or BAND gap.
    Its `sources/`, `circuits/`, `forces/` and `probes/` groups line up row
    for row with the definition tables:
    - `sources/current_A`, `linkage_Wb_turn`, `differential_inductance_H`
      (square, source order);
    - `circuits/current_A`, `linkage_Wb_turn`, `inductance_H` (square,
      circuit order), and with an AGE gap `linkage_rate_Wb_per_rad`, each
      circuit's dλ/dφ at fixed current (NaN for open circuits);
    - `forces/force_N`, `torque_Nm`, `force_local_N`, `torque_local_Nm`, and
      per connected piece `piece_force_N`, `piece_torque_Nm` with
      `piece_offsets`;
    - `probes/B_T`, `H_A_per_m` in basic components, and `B_local_T`,
      `H_local_A_per_m` in the local directions of CD at the probe (e.g.
      radial and tangential for a CORD2C). On an element boundary a probe
      value is the arithmetic mean of the incident cell traces, including
      different materials; place probes within the intended material when a
      one-sided value is needed.

  - A MOTION subcase's group has instead the attributes `id`, `iterations`
    (the total), `residual` (the largest), `bh_extrapolated` (at any
    position), `motion`, `positions`, `start_deg`, `step_deg`,
    `torque_mean_Nm`, `torque_peak_to_peak_Nm`, `torque_ripple_percent` (NaN
    for a zero mean or a current-free sweep) and `current_free`. With `MSTEP = ALL` it also has the
    datasets `positions_deg`, `iterations`, `residual`, `energy_J`,
    `coenergy_J`, `bh_extrapolated` and `motion/torque_Nm`, one row per
    position, and the `sources/`, `circuits/`, `forces/` and `probes/`
    tables above with a leading position axis (e.g. `forces/torque_Nm` is
    [positions, surfaces, 3]). With `SPEED` the group has the attribute
    `speed_rpm`, and with `MSTEP = ALL` `motion/emf_V` [positions, circuits].
  - A DRIVE subcase's group also has a `drive/` group with the attributes
    `id`, `wave` (`SINE` or `SIX`), `pole_pairs`, `current_peak_A`,
    `gamma_deg`, `theta0_deg`, `theta0_auto` and the means over the
    positions `i_d_mean_A`, `i_q_mean_A`, `lambda_d_mean_Wb`,
    `lambda_q_mean_Wb`, `L_d_incremental_mean_H`, `L_q_incremental_mean_H`,
    and under `LDQ = APPARENT` `L_d_apparent_mean_H`, `L_q_apparent_mean_H`,
    `lambda_magnet_d_mean_Wb`, `lambda_magnet_q_mean_Wb`. Its datasets are
    `phase_circuits`; for THETA0 AUTO the attribute `alignment_linkage_Wb`
    (phase 1's no-load fundamental) and the dataset `alignment_lag_deg`;
    and when the sweep covers whole electrical periods the attribute
    `electrical_periods` with the per-phase datasets
    `emf_constant_fundamental_V_s_per_rad` and `emf_thd_percent`. With
    `MSTEP = ALL` it adds per position `theta_e_deg`, `i_d_A`, `i_q_A`,
    `lambda_d_Wb`, `lambda_q_Wb` and `L_dq_incremental_H` [positions, 2, 2]
    (d first), and under `LDQ = APPARENT` `L_dq_apparent_H`,
    `lambda_magnet_d_Wb` and `lambda_magnet_q_Wb`.

Vectors are rows of length 3. Unavailable linkage and inductance entries are
NaN. Groups keep creation order, so subcases appear in deck order. `h5py`
reads everything, e.g.
`h5py.File("name.mag.vtkhdf")["MAGSTAT/subcases/1"].attrs["energy_J"]`.

Output requests select what is written. `BFIELD=ALL` (default), `NONE` or an
ESET ID writes the subcase field block for every cell, none, or only that set
(points are compacted to the set's cells). `BFIELD(EVERY=n)` writes a MOTION
subcase's fields at every n-th position; with MOTION fields the VTKHDF
collection is a time series in degrees ([Motion](#motion-motion)). A selected ESET must reference
existing field elements; CWIRE IDs are not field elements. `MPROBE=ALL|NONE`
keeps or drops the subcase's `probes` group. Both may be global or set per
subcase; a subcase value overrides the global one. `MSOURCE=ALL|NONE` is
global only and controls the source blocks. The other scalar results are
always written.

No magnetic OP2 tables are emitted. `coenergy_J` is the source Legendre
transform `sum(I*lambda)-W`, retaining PM energy constants; it is not generally
`W` when permanent magnets or imposed exterior fields are present.

A planar model with an AGE air gap sweeps its rotor in one run with
[MOTION](#motion-motion). For other models, supply a valid mesh and rotated
PM directions for each angle; the sweep runner does not remesh or silently
rotate selected material regions:

```json
{"cases": [
  {"angle_deg": 9, "deck": "motor9.bdf"},
  {"angle_deg": 10, "deck": "motor10.bdf"},
  {"angle_deg": 11, "deck": "motor11.bdf"}
]}
```

```sh
.venv/bin/python tools/magnetic_sweep.py manifest.json --vibestran build/vibestran --output /tmp/magnetic/sweep
```

The sweep runner reads each run's `/MAGSTAT` results with `h5py`, so it runs
from the `.venv` described under the tests below.

`sweep.csv` includes circuit currents/linkages and surface force/torque. At interior
angles it computes the three-point derivative of coenergy with respect to angle
in **radians** when circuit IDs and currents match. This is the fixed-current
virtual-work torque about the rotation axis. It is blank at endpoints or when
currents change. Check mesh and angle-step convergence before comparing it with
the corresponding component of stress torque. Do not differentiate a commutated
current sequence as if the currents were fixed.

The example generator deforms the air mesh between the rigid rotor and fixed
stator and restricts its demonstration angles to +/-20 degrees. Arbitrary
prepared meshes can be used with the solver. A full BLDC sweep requires suitable
meshes and phase-current subcases (one CIRCUIT per phase); no transient,
voltage-driven or sliding-band model is implied.

## Verification

The ordinary test suite contains only internal/analytical checks: oriented edge
and face basis identities, exact intersection quadrature, material energy and
tangent derivatives, straight-coil flux/turn scaling, closed cuts, finite-section
wire Biot–Savart comparison, conservative nonmatching transfer, invalid geometry
and currents, exact uniform fields/energy, PM reversal, nonlinear differential
inductance, nonconvergence errors, mutual reciprocity, superposition, gauge
renumbering, rigid rotation, and loop torque approaching `m cross B` within 3%.
Additional checks cover tapered/reversed hex and wedge edge/flux moments,
finite-difference curls and the discrete curl-gradient identity; native mixed
solid meshes and conserved coils; driven hex/wedge field refinement; planar
tri/quad manufactured-solution convergence, nonlinear differential inductance,
PM reversal, rotated distorted meshes, depth scaling and nonzero winding torque
against its analytical axial-current moment. Warped hex/wedge checks cover
bilinear-face flux moments, mixed-interface
continuity, uniform fields and Maxwell stress, folded mappings missed by
volume quadrature, tiny curved overlaps versus contact, warped coil cuts,
open-wire endpoints, and independent wire transfer/current moments and energy.
Ray transfer is also compared with polyhedral intersection integration on a
shared analytical fixtures. Both HEX and TET are checked for straight-wire
current, volume, far field,
warped-background transfer, open terminals and material exclusion. Cubic monomial
integrals verify the five-point rule; arbitrary face fluxes verify cached currents
on affine, warped, reflected and translated cells. Tet transfer is invariant to
local vertex order and rigid motion. The spatial-loop AUTO fallback and strict
HEX rejection are checked against the current moment.
Axisymmetric (MAGAXI) checks: a uniform axial field is exact on graded quads
and triangles (cell and on-axis probe B, energy); an infinitely long iron tube
in an axial field and a thick solenoid, against their exact piecewise
solutions, converge at second order in energy and inductance; the axial
MFORCE on an iron ring (BODY and an axis-ended contour) matches the
coenergy's derivative to 1%; and the card checks (meridian remanence, axial
uniform field, axis faces, negative radius, features not yet available).

The `magnetic_cli` test (`tests/e2e/test_magnetic_cli.cpp`) runs the
`vibestran` executable on decks from the standard-library generators in
`tools/` and reads the HDF5 results through the C API. It checks analytical
energies and fields, VTK cell types, planar nodal potentials, nonlinear BH
energy and H, circuit linkage and inductance identities, exported ampere-turn
integrals and wire-mesh schemes. It needs no third-party Python packages.

The air-gap and motion tests (`tests/unit/test_air_gap.cpp`,
`tests/unit/test_motion.cpp`) check the guarantee tables under
[Air gap](#air-gap-pairgap) and [Motion](#motion-motion) on generated rotor
and stator decks, parsed as a user's deck would be.

The optional `magnetic_vtkhdf` test (`tools/magnetic_vtkhdf_check.py`) loads
every block through VTK's own `vtkHDFReader`, as ParaView does, and checks it
against the raw datasets, at every time step of a MOTION time series; it also
runs the prepared-deck sweep. It needs the
packages in `tools/requirements-magnetic.txt`, which it takes from the
repository `.venv` when the current interpreter lacks them, and is reported as
skipped without them:

```sh
python3 -m venv .venv
.venv/bin/pip install -r tools/requirements-magnetic.txt
```

Run `meson test --print-errorlogs` and `ninja cppcheck` from `build`.
Independent tool comparisons, their dependencies and their outputs live in
[Magnetic_Validation](Magnetic_Validation/README.md); none are registered in
Meson or ordinary CI. Detailed external force and nonlinear benchmark validation
remains necessary before relying on a real motor result.

## CWIRE performance measurements (2026-10-30)

Build and run the optional benchmark from `build`:

```sh
meson configure -Dbuildtype=release
ninja magnetic_wire_benchmark
./magnetic_wire_benchmark 3 > /tmp/magnetic-wire-timings.csv
```

The second argument can select `flat` or one case name from the table below.
The target is not built by default or registered as a timing test. It uses the
production WIREMESH selector, alternates HEX/TET run order, and emits stage times,
mesh/system sizes, current continuity, volume, moment and center Biot–Savart field.
All cases use linear unit-current preprocessing. No magnetic field solve is timed;
field DOFs below are the reduced system size a subsequent solve would use.
Preparation includes source meshing, validity/overlap checks and the current solve;
transfer integrates the source against the background edge basis. Background
construction, diagnostics and file I/O are outside these stage timers.

Measurements used Apple arm64, Apple clang 17, Accelerate sparse direct solves,
Meson Release (`-O3 -DNDEBUG -march=native`), and precise magnetic kernels
(`-fno-fast-math -ffp-contract=off`). Baseline is commit `3be86d5` with the existing
tet fallback forced for comparison; the new selectable TET uses that same split.
Times below are milliseconds, median of three uninstrumented runs. Old warped
HEX times marked * are approximate single instrumented profile runs; do not infer
precise speedup ratios from those two entries.

| Case | Field cells / reduced DOFs | Wire cells HEX / TET | Current system HEX / TET | HEX old → new ms | TET old → new ms |
|---|---:|---:|---:|---:|---:|
| `round_tet` | 1296 tets / 1081 | 144 / 864 | 528 / 2496 | 2659.7 → 112.8 | 114.8 → 64.8 |
| `rect_tet` | 1296 tets / 1081 | 12 / 72 | 24 / 168 | 55.1 → 22.7 | 10.3 → 6.4 |
| `rect_hex` | 512 hexes / 833 | 12 / 72 | 24 / 168 | 72.3 → 17.3 | 11.4 → 4.7 |
| `rect_warped_hex` | 8 hexes / 5 | 12 / 72 | 24 / 168 | ~7690* → 44.7 | refinement-budget failure → 311.6 |
| `rect_warped_wedge` | 16 wedges / 9 | 12 / 72 | 24 / 168 | ~9185* → 49.3 | refinement-budget failure → 411.0 |

The current system includes face-flux unknowns and constraint multipliers;
closed paths have one fewer independent cell-balance constraint and one cut
constraint. All loops have side length 1.2 m, diameter 0.08 m or rectangular
section 0.08 by 0.06 m, WIRESTEP=0.4, WIRESIDE=8 and WIRELAY=1. Flat backgrounds
have half-width 1.5 m. Warped backgrounds have half-width 1 m before the shear
`z += 0.35*x*y` (hex) or `x += 0.35*y*z` (wedge, loop in the yz plane).
These small fixtures isolate transfer costs; they are not motor-scale solve
benchmarks or a source-mesh convergence study.

On `round_tet`, median preparation/transfer times are 4.3/108.5 ms for HEX and
35.1/29.7 ms for TET. Hex preprocessing remains smaller and faster; tetrahedral
transfer still wins overall here. Tets improve 1.6–2.4 times on the three flat
cases, and hexes improve 2.4–23.6 times. Hexes win the two warped cases.

Separate temporary operation counters identified the costs, without adding
instrumentation overhead to the timings above:

| Operation / case | Before | After |
|---|---:|---:|
| Current evaluations, round HEX on tets | 20,570,112 | 914,112 |
| Inverse-map Newton steps, round HEX on tets | 78,027,151 | 79,972 |
| Polygon clips after broad-phase query, round TET | 43,533 | 10,800 |
| Integration points, rectangular TET on hexes | 53,568 | 9,920 |
| Rays, HEX on warped hexes | 854,720 | 6,912 |
| Line quadrature points, HEX on warped hexes | 21,496,608 | 165,888 |

The optimized ray and independent clipping algorithms agree to relative 1e-9
in regression tests for the same source, including round/rectangular sections
and both source element types. Across these benchmark fixtures, HEX versus TET
load vectors differ by at most 0.229% and center Biot–Savart fields by 0.0141%.
Those are different current discretizations, not an equal-error comparison.
The largest current-continuity residual is 5.1e-12, below the unchanged 1e-9
limit. No acceptance tolerance was relaxed. At this profiling checkpoint, all
six Meson suites passed with 42 magnetic tests; the full unit suite decreased
from 44.77 to 12.78 seconds, including the new cross-checks. Third-party
validation remains separate.


## Nonlinear verification coverage

The prior single-winding planar test stayed below its first 0.1 T B–H knot.
Its replacement requires quadrature points in both the 0.1–0.3 T and 0.3–1 T
segments, and more than one Newton iteration. The original tetrahedral nonlinear
test now also asserts which B–H range it reaches. These checks inspect the
reconstructed field at integration points, not the exported cell averages.

The new shared B–H table is `(B,H) = (0,0), (0.1,100), (0.3,2000), (1,100000)`.
Each planar case has two independent 100-turn circuits at 4000 A and 2500 A;
each 3D case has one rectangular CWIRE loop at 100000 A. Every fixture solves
an operating point, a 1% low-current linear limit, a 20x extrapolation case,
current reversal, and +/-0.01% perturbations of each circuit. These are
synthetic constitutive stress cases, not representative motor operating points.

Independent current perturbations verify every entry of the differential
inductance matrix, including nonzero planar mutual coupling, to relative 1e-6.
They also verify `dW'/dI_i = lambda_i` and
`dW/dI_i = sum_j I_j * d(lambda_j)/dI_i`. The tests check reciprocity, positive
differential inductance, saturation's departure from secant inductance and
quadratic energy, the low-field linear identities, and field/linkage reversal.
The solver residual tolerance is 1e-12 in these derivative fixtures; the
production default remains 1e-8. The high-current solutions must actually
exceed the final B–H knot and report extrapolation.

Representative Release timings on the same Apple arm64/Accelerate build as
above, measured for the whole test (all subcases, preprocessing and assertions),
not per Newton iteration:

| Fixture | Field cells | Free field DOFs | Subcases | Operating Newton iterations | Maximum steel B (T) | Test time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Planar CQUAD4 | 144 | 121 | 8 | 5 | 0.643 | 19 |
| Planar CTRIA3 | 288 | 121 | 8 | 5 | 0.506 | 18 |
| Planar mixed | 216 | 121 | 8 | 5 | 0.632 | 20 |
| CTETRA4 | 384 | 289 | 6 | 6 | 0.330 | 53 |
| CHEXA8 | 64 | 81 | 6 | 3 | 0.369 | 34 |
| CPENTA6 | 128 | 129 | 6 | 5 | 0.323 | 45 |
| Warped CHEXA8 | 64 | 81 | 6 | 3 | 0.383 | 93 |
| Warped CPENTA6 | 128 | 129 | 6 | 5 | 0.335 | 102 |

The planar meshes are distorted in-plane. Each 3D source uses 12 internal
hexes and a 24-unknown current system; the field meshes above are independent.
These eight tests take about 0.39 s together in this run. Sizes, field maxima
and Newton counts are also recorded as Google Test XML properties.

Additional checks cover:

- B–H knot continuity and one-sided radial/transverse material tangents,
  including the vacuum-slope extrapolation and zero-field limit.
- Rejection of missing origins, duplicate/unsorted B values, and zero or
  negative differential slopes; positive slopes need not themselves increase.
- Successful continuation in both 2D and 3D. A per-stage iteration cap forces
  the retry; its full-current fields, energy, linkage and tangent must match
  an unrestricted solve. The hex fixture uses 21 continuation iterations
  versus 4 direct; the planar fixture uses 32 versus 5. A tighter cap must
  return an error instead of a partial solution. Together these take ~106 ms.
- Permanent magnets driving nonlinear steel across both interior knots on
  planar triangles and quads, including simultaneous magnet reversal (~11 ms).
- CLI decks with TABLEBH data: exact uniform-field nonlinear energy, exported
  H and the extrapolation flag in both 2D and 3D.

Run just the new driven cases from `build` with
`./unit_tests --gtest_filter='MagneticElements/MagneticNonlinear*'`.
The full magnetic suite uses `--gtest_filter='Magnetic*'` and now has 54 tests.
All these checks are internal/analytical; third-party nonlinear validation
remains in the separate workflow described above.
