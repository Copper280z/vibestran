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
`PARAM,AIRGAP,AGE` or `PARAM,AIRGAP,MESH` sets the METHOD of every `PAIRGAP`
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
| `PAIRGAP` | `PID,METHOD,NHARM` (planar air-gap property; METHOD `AGE` or `MESH`; see [Air gap](#air-gap-pairgap)) |
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

Case control accepts `LOAD`, `MAGBC`, `BFIELD`, `MPROBE`, `MSOURCE`, `LABEL`,
`TITLE`, `SUBTITLE` and `SUBCASE`. Global LOAD/MAGBC settings are inherited by
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
check truncation error. Periodic, symmetry, infinite elements and multiply
connected magnetic domains are not implemented.

`MAGBC=SID` selects MBC. FSET=0 means the whole exterior; an explicit FSET must
cover the whole exterior. A planar [air gap](#air-gap-pairgap) solved by
the air-gap element is not exterior: FSET=0 leaves its circles out, and an
FSET that selects an edge on them is an error. UNIFORM prescribes the tangential trace of
`A = B cross x / 2`, useful for applied uniform fields and exact patch tests.
The source perturbation still has zero tangential exterior trace.

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

### Air gap (PAIRGAP)

The air gap between a rotor and a stator is meshed like any other air: a band
of CTRIA3/CQUAD4 elements between the rotor's outer boundary and the stator's
inner boundary. Give those elements a property ID of their own and define it
with `PAIRGAP` instead of `PMAG2D`:

```text
PAIRGAP,PID,METHOD,NHARM
```

| Field | Meaning |
|---|---|
| `PID` | Property ID of the gap elements. It must not also be a PSOLID, PMAG or PMAG2D ID, and at least one element must use it. |
| `METHOD` | `AGE` (blank is AGE): replace the gap elements by the air-gap element. `MESH`: solve them as ordinary meshed air. |
| `NHARM` | AGE harmonics. Blank uses half the grid count of the coarser circle; at most half the grid count of the finer one. Ignored by MESH. |

There is no material field: a gap is air (relative permeability 1, no
remanence, no B–H table). Under the AGE the gap elements are not field cells,
so a COIL2D, MFORCE BODY or BFIELD ESET may not contain them and an FSET may
not name them; under MESH they are ordinary air elements. Keep such sets off
the gap if the deck is to run both ways. `PARAM,AIRGAP,AGE|MESH` overrides
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

**What each method guarantees.** These hold for every deck, and the test
suite (`tests/unit/test_air_gap.cpp`) checks each on a magnet rotor in a
saturating stator with a winding:

| Comparison | Result |
|---|---|
| `PAIRGAP,PID,MESH`, `PARAM,AIRGAP,MESH`, and the deck with PAIRGAP replaced by MATMAG + PMAG2D air | The same equations. Every result (potentials, energies, B, H, probes, linkage, inductance, force, torque, Newton iterations) is bitwise identical with a deterministic linear solver. The default multithreaded sparse factorization varies in its last bits from run to run on any deck, so with it the three agree to round-off (measured 6e-16 in energy). |
| AGE against MESH on the same gap mesh | Two discretizations of the same gap, not identical. Both converge to one solution, and their difference falls at O(h²) in energy, linkage and inductance. |
| AGE against an analytic magnetized rotor | O(h²) in the gap field and in the magnet. The AGE adds no error of its own for the piecewise-linear traces it receives; the meshed parts set the error. |
| MOTION at angle φ (planned) against a static solve of the rotor rotated by φ | The same to round-off. |

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
radial deviation. The HDF5 results have an `air_gap` group. Under the AGE the
VTKHDF field blocks contain the field cells only; the gap elements are not in
them. Under MESH the gap elements are ordinary field cells.

**Limits.** MBC does not apply to the AGE's circles, and MFORCE surfaces may
not touch them, so rotor torque is not yet available with the AGE (use MESH
for torque until MOTION reports it). The planned `MOTION` card will rotate
the rotor through the AGE and report its torque. 3D models mesh their air
gap with ordinary air elements; PAIRGAP is planar only.

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
One factorization serves the whole run: its symbolic analysis is done once,
and Newton steps, continuation stages, subcases and the differential
inductances only refactor values. Each subcase's inductances are a single
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
  `units`, `dimension` and `depth_m`. Definition tables, one row per item in
  ID order:
  - `sources/`: `id`, `wire`, `closed`, `conservation_error`,
    `transfer_error`, and `wire_elements` (element ID, traversal sign) with
    `wire_element_offsets` giving each source's rows.
  - `circuits/`: `id`, `parallel`, `closed`, and `members` (source ID,
    polarity) with `member_offsets`.
  - `forces/`: `id`, `body`, `cid`. `probes/`: `id`, `cd`, `position_m`.
  - `air_gap/` (planar models with a PAIRGAP solved by the AGE): attributes
    `property`, `inner_radius_m`, `outer_radius_m`, `harmonics`,
    `radial_deviation`;
    `center_m`; and `inner_grids`, `outer_grids`, the grid IDs on each
    circle in increasing angle.
  - `subcase_ids`, and `subcases/<ID>/` per subcase, with attributes `id`,
    `iterations`, `residual`, `energy_J`, `coenergy_J`, `bh_extrapolated`.
    Its `sources/`, `circuits/`, `forces/` and `probes/` groups line up row
    for row with the definition tables:
    - `sources/current_A`, `linkage_Wb_turn`, `differential_inductance_H`
      (square, source order);
    - `circuits/current_A`, `linkage_Wb_turn`, `inductance_H` (square,
      circuit order);
    - `forces/force_N`, `torque_Nm`, `force_local_N`, `torque_local_Nm`, and
      per connected piece `piece_force_N`, `piece_torque_Nm` with
      `piece_offsets`;
    - `probes/B_T`, `H_A_per_m` in basic components, and `B_local_T`,
      `H_local_A_per_m` in the local directions of CD at the probe (e.g.
      radial and tangential for a CORD2C). On an element boundary a probe
      value is the arithmetic mean of the incident cell traces, including
      different materials; place probes within the intended material when a
      one-sided value is needed.

Vectors are rows of length 3. Unavailable linkage and inductance entries are
NaN. Groups keep creation order, so subcases appear in deck order. `h5py`
reads everything, e.g.
`h5py.File("name.mag.vtkhdf")["MAGSTAT/subcases/1"].attrs["energy_J"]`.

Output requests select what is written. `BFIELD=ALL` (default), `NONE` or an
ESET ID writes the subcase field block for every cell, none, or only that set
(points are compacted to the set's cells). A selected ESET must reference
existing field elements; CWIRE IDs are not field elements. `MPROBE=ALL|NONE`
keeps or drops the subcase's `probes` group. Both may be global or set per
subcase; a subcase value overrides the global one. `MSOURCE=ALL|NONE` is
global only and controls the source blocks. The other scalar results are
always written.

No magnetic OP2 tables are emitted. `coenergy_J` is the source Legendre
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

The `magnetic_cli` test (`tests/e2e/test_magnetic_cli.cpp`) runs the
`vibestran` executable on decks from the standard-library generators in
`tools/` and reads the HDF5 results through the C API. It checks analytical
energies and fields, VTK cell types, planar nodal potentials, nonlinear BH
energy and H, circuit linkage and inductance identities, exported ampere-turn
integrals and wire-mesh schemes. It needs no third-party Python packages.

The optional `magnetic_vtkhdf` test (`tools/magnetic_vtkhdf_check.py`) loads
every block through VTK's own `vtkHDFReader`, as ParaView does, and checks it
against the raw datasets; it also runs the prepared-deck sweep. It needs the
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
