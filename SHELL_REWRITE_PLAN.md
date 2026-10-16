# Proposal: Rebuilding vibestran's Shell Elements

Status: **proposal, awaiting approval** (2026-10-16). Nothing is implemented yet.

## 1. Why a rewrite

vibestran's shells grew incrementally: a flat Mindlin quad, then an in-house
"MITC4+" adaptation on a flat projection, a CST + DKT triangle, and several
patches this week (moment sign, element frame, strains). Problems in the
current code (`src/elements/cquad4.cpp`, `src/elements/ctria3.cpp`):

| # | Problem | Evidence |
|---|---|---|
| 1 | Both elements are **flat**. A warped quad is projected onto one plane, and nothing accounts for the warp. | MacNeal twisted beam 51–61% too stiff. Results on warped meshes depend on the projection plane (the diamond cylinder moved 8.5% when the plane changed). Warp > 30° is a fatal gate (10 decks). |
| 2 | **Drilling DOF is a penalty**: 1e-6 × the largest rotational (quad) or overall (tria) diagonal, with an ad hoc drill mass. There's no real in-plane rotation stiffness. | Shell–beam junctions and shells meeting at large angles rely on a numerically arbitrary spring. |
| 3 | The quad's "MITC4" option is an **unpublished adaptation** of MITC4+ to a flat local frame. MINDLIN is a second, older formulation. | No literature formulation to verify against. Two code paths to maintain. |
| 4 | **CTRIA3**: CST membrane (locks under in-plane bending), Kirchhoff DKT (no transverse shear), MID3 and MID4 ignored. | Q = 0 for every triangle (`S30 mitc3 strain` fails). Stiff membranes. |
| 5 | **Section/material coverage**: MAT1 only (MAT2/MAT8 are fatal, 17 decks). PCOMP isn't parsed (20 decks). THETA/MCID are ignored. ZOFFS is parsed but **silently ignored**. CQUAD4/CTRIA3 corner thicknesses aren't read. Rotary inertia is always included. | `failure_bucket_plan.md`. |
| 6 | **Thermal**: membrane load only; no through-thickness gradient (thermal moment), no MID4 coupling term. | Free thermal expansion of a faceted cylinder is 36% wrong. |
| 7 | **Recovery**: "corner" values come from evaluating B at the corner, not Nastran's BILIN extrapolation from the Gauss points. | Corner stresses differ from Nastran semantics on distorted elements. |
| 8 | **No higher-order shells**: CQUAD8/CTRIA6 don't exist. | 47 failing decks use CQUAD8. |

Shells appear in **186 of the 419 failing validation decks (1,395 of 2,225
records)**. Not all of those are element defects, but shells are the largest
area of the suite.

## 2. Chosen formulations (published research)

The goal is low-order shells that are **continuum-mechanics based** (so they
handle curved and warped geometry natively), free of shear and membrane
locking on distorted meshes, provably pass the basic tests (zero-energy
modes, isotropy, patch tests), and carry **six DOFs per node**, because every
Nastran grid has six.

### 2.1 CQUAD4 → MITC4+/D (Ko, Bathe & Zhang 2025)

This is the formulation in the paper the user supplied:
*Continuum mechanics-based shell elements with six degrees of freedom at
each node – the MITC4/D and MITC4+/D elements*, Comput Struct 308 (2025)
107622. It extends the Bathe group's MITC4+ element:

- **Geometry**: a 4-node continuum shell with nodal directors V_n^i and
  nodal thicknesses a_i (Eq. 1). It isn't flat; warped and curved
  geometry are represented exactly at the nodes.
- **Transverse shear**: the classical MITC4 assumed covariant shear field,
  tied at the four edge midpoints (Dvorkin & Bathe 1984; Eq. 20).
- **Membrane**: the MITC4+ assumed membrane strain field, which removes
  membrane locking in distorted and curved meshes (Ko, Lee & Bathe 2017,
  Comput Struct 182:404–418). The improved version embeds the 2D-MITC4
  in-plane field (Ko, Lee & Bathe 2017, Comput Struct 192:34–49). The /D
  paper gives the explicit tying relations (Eqs. 21–24, Appendix B).
- **Drilling rotation**: a true drill stiffness from Allman-type normal
  edge displacements at fictitious mid-side nodes. All drill rotations are
  measured about the single normal V^D of the element's centre plane P
  (Eqs. 5–19). That choice keeps rigid-body modes exact even for non-flat
  elements. The element passes the patch tests **with the drill rotations
  free**, with no artificial drilling factor. That is exactly what a 6-DOF
  Nastran grid needs.

Useful consequences:
- V^D = x_r × x_s is parallel to d13 × d24, the Nastran CQUAD4 element
  normal already in use since commit 504d524. The Nastran element frame
  and the formulation's base plane coincide.
- Corner thicknesses (CQUAD4 T1–T4) map directly onto the nodal thicknesses
  a_i.

### 2.2 CTRIA3 → MITC3+ with a drilling extension

- **MITC3+** (Lee, Lee & Bathe 2014, Comput Struct 138:12–23): a flat
  3-node continuum shell with a cubic bubble on the rotations (condensed
  out at element level) and an assumed transverse shear field tied at six
  points. It passes the basic tests and shows near-optimal convergence. It
  is a Reissner–Mindlin element, so **CTRIA3 gets real transverse shear**.
  The full formulation is restated in the freely available Jeon, Lee, Lee
  & Bathe 2015 (Comput Struct 153:148–164), Eqs. 6–9 and Table 1.
- **Drilling**: MITC3+ has 5 DOFs per node, and there is no published
  "MITC3+/D". I propose the same construction as MITC4+/D: Allman-type
  normal edge displacements from the corner drill rotations, measured about
  the (flat) element normal. That is Allman's original 1984 triangle idea,
  added to the MITC3+ membrane. Because a triangle is flat, this is simpler
  than the quad case. **It is our extension, not a published element**, so
  it must pass the same basic tests (six zero eigenvalues, isotropy,
  membrane/bending/shear patch tests with drill rotations free) before it
  replaces anything. Fallback: MITC3+ with a documented, Nastran-style
  `K6ROT` drilling penalty.

### 2.3 Later phase: CQUAD8 / CTRIA6

These are separate follow-ups, not part of the core rewrite: a quadratic
MITC element for CQUAD8 (MITC8/MITC9 family: Bathe & Dvorkin 1986; Bathe,
Lee & Hiller 2003) and MITC6 for CTRIA6 (Lee & Bathe 2004). They will reuse
the section, recovery and Nastran-semantics layers built here.

## 3. Nastran semantics layer (shared by all shells)

The formulation gives stiffness; Nastran compatibility lives in a shared
layer that is independent of the element type and tested separately.

1. **Section (`ShellSection`)**: resultant constitutive
   `[N; M] = [A B; Bᵀ D] [ε0; κ] − thermal`, `Q = T γ`.
   - PSHELL: MID1 membrane (×T), MID2 bending (×12I/T³·T³/12), MID3
     transverse shear (×TS/T·T), MID4 coupling (×T²), NSM, Z1/Z2.
   - Materials: MAT1, MAT2 (anisotropic), MAT8 (orthotropic, including the
     transverse shear G1Z/G2Z), each rotated by the material angle.
   - PCOMP/PCOMPG: classical lamination theory (ABD + transverse shear),
     LAM options SYM/MEM/BEND/SMEAR, ply recovery as a later step.
   - Material orientation: CQUAD4/CTRIA3 THETA or MCID, projected per the
     Nastran QRG definition (to be checked against the QRG text during
     implementation, not against MYSTRAN).
2. **Resultant vs. through-thickness integration**: the continuum element
   gives covariant strains `e = e_m + t·e_b1 + t²·e_b2` (Eq. 8). Mapped to
   the local Cartesian frame at each in-plane Gauss point, these give the
   mid-surface strain ε0, curvature κ and shear γ, which feed the resultant
   section above. This is what makes PSHELL's independent MID2/12I/T³ and
   PCOMP's ABD possible. It is exact for homogeneous flat sections; the
   neglected `t²` term is O(t/R) small for thin curved shells. The verified
   benchmark table will show whether that matters in practice.
3. **Offsets**: ZOFFS (element) and PSHELL/PCOMP Z0 implemented as a rigid
   offset of the reference surface along the element normal. They are
   currently ignored, which is a silent error.
4. **Directors**: by default each element uses its own corner normals
   (Nastran's default behaviour). `PARAM,SNORM,angle` switches to averaged
   grid normals within the angle, which MYSTRAN's "SNORM" decks exercise.
5. **Mass**: consistent by default (the user's stated choice), lumped with
   `PARAM,COUPMASS ≤ 0`. Rotary inertia follows Nastran (off by default,
   included only on request; to be confirmed against the QRG). The drill
   rotation gets the inertia the formulation implies rather than an ad hoc
   value.
6. **Loads**: thermal (see 3.1 below); PLOAD2/PLOAD4 stay in
   `linear_static_pressure.cpp`, unchanged.
7. **Recovery**: stresses, strains and forces at the 2×2 Gauss points,
   centre, and corners. CORNER/BILIN corners use Nastran's bilinear
   extrapolation from the Gauss points. Output stays in the Nastran element
   frame (the diagonal bisector for CQUAD4, G1→G2 for CTRIA3), with fibre
   strains and stresses at Z1/Z2 and transverse shear from the assumed field.
   The F06 writers don't change; they already consume `PlateStress`.
8. **Quality checks**: warp becomes a warning, not a fatal gate, because the
   element handles warped geometry.

### 3.1 Thermal stress and heat transfer

**Heat transfer (SOL 153 / ANALYSIS=HEAT) is not changed.** Shell
conduction is a separate set of elements in `thermal_elements.cpp`, with its
own frame code, reading only the PSHELL thickness and material
conductivity. They don't use the structural shell classes. The one coupling
is data: once PCOMP exists, a PCOMP-shell heat model needs the laminate
thickness. That is a small follow-up, and today those decks fail at
PCOMP anyway.

**Thermal stress (structural thermal loads) is reworked**, and that is where
the results change:

- **Inputs.** Today: TEMP/TEMPD grid temperatures (mid-surface) and HEAT →
  STATICS chaining, which also feeds mid-surface grid temperatures.
  Added: TEMPP1 (mean temperature plus through-thickness gradient, or top
  and bottom temperatures). The parser doesn't read it now.
- **Section thermal resultants** come from the same ABD layer:
  `N_T = A·α_m·ΔT` (MID1 α), `M_T = D·α_b·∇T` (MID2 α, from the gradient),
  plus the MID4/B coupling terms. PCOMP uses ply-by-ply CLT, so an
  unsymmetric laminate bends under uniform ΔT, as it physically should.
  Today only the membrane term exists.
- **Consistency.** The element thermal load is `∫ Bᵀ [N_T; M_T]` with the
  **same assumed-strain B matrices as the stiffness** (MITC4+ membrane,
  drill-membrane, MITC shear). A free body under uniform ΔT, flat or curved,
  then expands with zero stress. The current element breaks that on curved
  facet meshes, which is the 36% faceted-cylinder error. This becomes a
  required benchmark.
- **Recovery.** Mechanical stress is `σ = Q(ε − αΔT(z))` at each fibre;
  reported strains stay total, as since 504d524.
- **HEAT → STATICS.** Conduction temperatures are mid-surface only, so a
  chained run gets the membrane term (and B coupling for unsymmetric
  sections), the same as an equivalent TEMP deck.

## 4. Architecture

```
src/elements/shell/
  shell_section.{hpp,cpp}     PSHELL/PCOMP + MAT1/2/8 → ABD, T, thermal, mass
  shell_geometry.{hpp,cpp}    directors, V^D plane, element frame, offsets
  mitc4plus_d.{hpp,cpp}       CQUAD4 stiffness / mass / loads / recovery
  mitc3plus_d.{hpp,cpp}       CTRIA3 (bubble condensation, drill extension)
  shell_recovery.{hpp,cpp}    Gauss → centre/corner, fibre values, frames
```

- `CQuad4` and `CTria3` keep their `ElementBase` interface, so assembly,
  solvers, CUDA/Vulkan paths and the writers don't change. Recovery in
  `linear_static.cpp` calls one shell recovery API instead of the two
  element-specific ones it calls today.
- The old flat MINDLIN/"MITC4" quad and CST+DKT triangle stay selectable
  only during the transition (for A/B comparison in tests). They are
  **deleted** once the new elements pass verification, so only one shell
  formulation is maintained. `PARAM,QUAD4TYP` (MYSTRAN-specific) is then
  accepted and ignored.
- Thermal (SOL 153) shell elements in `thermal_elements.cpp` are out of
  scope.

## 5. Verification

MYSTRAN is a cross-check, not the oracle. Every element must first pass
tests whose answers come from theory or the published literature.

**Element-level (unit tests)**
- Zero-energy modes: exactly 6 zero eigenvalues of the free 24×24 / 18×18
  stiffness (flat, warped, curved, distorted, varying thickness); none
  spurious, including the drill.
- Rigid-body motion: zero strain for all six rigid modes on warped and
  curved elements.
- Isotropy: results invariant to node renumbering and in-plane rotation.
- Patch tests (membrane, bending, transverse shear) on the
  MacNeal / Ko et al. distorted 5-element patch, with drill rotations free.
- Section: ABD from PCOMP checked against hand CLT, MAT8 rotation against
  an independent tensor rotation, and offsets against a statically
  equivalent model.

**Benchmarks (integration tests with checked-in decks; values from the
cited papers and MacNeal & Harder 1985)**
- Scordelis–Lo roof, pinched cylinder, pinched hemisphere (with and without
  hole), Raasch hook, MacNeal twisted beam (thick and thin), straight and
  curved cantilever, clamped and simply supported plates (Timoshenko),
  Morley skew plate, Cook's membrane, the hyperboloid and clamped/free
  cylinder s-norm problems (Ko et al. 2017, C&S 193). Each test asserts the
  converged-mesh value within a stated tolerance, plus an observed
  convergence rate on refinement where the paper gives one.
- Thermal: free expansion of flat and faceted curved shells (zero stress,
  exact αΔT growth), and a bimetal strip under a gradient.
- Shell–beam junction and a T-junction of plates (drill stiffness in use).

**System**: full unit/integration suites, cppcheck, then the MYSTRAN suite.
Each newly failing record gets triaged as either our defect (fix it) or
MYSTRAN behaviour (document it in `failure_bucket_plan.md`).

## 6. Phases and commits

Each phase is its own commit (or a few), with tests and a validation run.

| Phase | Content | Exit criterion |
|---|---|---|
| 0 | Benchmark decks + harness; record current results as the baseline | Baseline table committed |
| 1 | `ShellSection`: PSHELL (full), MAT1/2/8, THETA/MCID, PCOMP/PCOMPG, nodal thickness, offsets data | Section unit tests pass; parser reads PCOMP, T1–T4 |
| 2 | MITC4+/D CQUAD4 (stiffness, mass, thermal, recovery) behind a switch | Element tests + quad benchmarks pass |
| 3 | MITC3+ with drill extension for CTRIA3 (penalty fallback if the extension fails its tests) | Element tests + tria benchmarks pass |
| 4 | Switch defaults, ZOFFS, SNORM, BILIN recovery, warp → warning; delete old elements; MYSTRAN triage; docs | No unexplained validation regressions |
| 5 (later) | CQUAD8/CTRIA6, differential stiffness for buckling | Separate proposals |

## 7. Risks

- **Drill-extended MITC3+ is unpublished.** It is gated on the basic tests,
  with a documented penalty fallback.
- **Resultant mapping on curved elements.** The O(t²) term is neglected for
  PSHELL semantics. Benchmarks with R/t ≥ 100 will show any effect; an exact
  through-thickness path for homogeneous MAT1 sections is possible if needed.
- **Paywalled detail.** The MITC4+ 2017 (C&S 182) paper isn't in hand.
  The /D paper and the open C&S 192 paper restate the membrane field with
  its coefficients, which should suffice. Any gap will be reported, not
  guessed.
- **Expected numerical shifts.** Results on curved and warped meshes will
  move. Some MYSTRAN records will newly fail where MYSTRAN's elements are
  less accurate; each one is triaged and documented.
- **Licence note.** The /D paper's licence forbids text and data mining and
  AI training. It is used here only as an engineering reference: equations
  are implemented and cited by number, and no text is copied into the repo.

## 8. Decisions requested

1. Approve MITC4+/D for CQUAD4 and MITC3+ (+ drill extension, with penalty
   fallback) for CTRIA3.
2. Approve the resultant-section approach (needed for PSHELL MID2/12I/T³
   and PCOMP semantics).
3. Approve deleting the old shell formulations after verification (rather
   than keeping them selectable).
4. If available, PDFs of Lee, Lee & Bathe 2014 (MITC3+, C&S 138:12–23) and
   Ko, Lee & Bathe 2017 (MITC4+, C&S 182:404–418) would remove the main
   source risk; neither is required to start phases 0–2.

## Sources

- Ko Y, Bathe KJ, Zhang X. Continuum mechanics-based shell elements with
  six degrees of freedom at each node – the MITC4/D and MITC4+/D elements.
  Comput Struct 2025;308:107622. (user-supplied PDF)
- Ko Y, Lee PS, Bathe KJ. A new MITC4+ shell element. Comput Struct
  2017;182:404–18.
- Ko Y, Lee PS, Bathe KJ. A new 4-node MITC element for analysis of
  two-dimensional solids and its formulation in a shell element. Comput
  Struct 2017;192:34–49. https://web.mit.edu/kjb/www/Principal_Publications/A_new_4_node_MITC_element_for_analysis_of_2D_solids_and_its_formulation_in_a_shell_element.pdf
- Ko Y, Lee Y, Lee PS, Bathe KJ. Performance of the MITC3+ and MITC4+ shell
  elements in widely-used benchmark problems. Comput Struct
  2017;193:187–206. https://web.mit.edu/kjb/www/Principal_Publications/Performance_of_the_MITC3+_and_MITC4+_shell_elements_in_widely_used_benchmark_problems.pdf
- Lee Y, Lee PS, Bathe KJ. The MITC3+ shell element and its performance.
  Comput Struct 2014;138:12–23.
- Jeon HM, Lee Y, Lee PS, Bathe KJ. The modal behavior of the MITC3+
  triangular shell element. Comput Struct 2015;153:148–64.
  https://web.mit.edu/kjb/www/Principal_Publications/The_modal_behavior_of_the_MITC3_plus_triangular_shell_element.pdf
- Dvorkin EN, Bathe KJ. A continuum mechanics based four-node shell
  element for general nonlinear analysis. Eng Comput 1984;1(1):77–88.
- Allman DJ. A compatible triangular element including vertex rotations for
  plane elasticity analysis. Comput Struct 1984;19(1–2):1–8.
- MacNeal RH, Harder RL. A proposed standard set of problems to test finite
  element accuracy. Finite Elem Anal Des 1985;1(1):3–20.
- Chapelle D, Bathe KJ. The finite element analysis of shells –
  fundamentals. 2nd ed. Springer; 2011.
