# S8 implementation plan for speed-dependent magnetics

The sections below record the original scope and its validation criteria. The user guide describes the implemented deck controls and result paths. [Magnetic analysis](MAGNETICS.md#eddy-currents-and-speed-dependent-cogging)

## First deliverable

Predict how the open-circuit electromagnetic torque waveform of a permanent-magnet rotary motor changes with prescribed RPM. Start with a full-circle `MAG2D` model, `PAIRGAP` with the air-gap element (AGE), and explicitly conducting magnets, sleeve, or solid back iron. Run the same geometry at several speeds in both directions and compare against its zero-current `SOL MAGSTAT` cogging sweep. The broader S8 roadmap also covers linear motion, general 3D conductors, and voltage-driven circuits.

At fixed geometry and material properties, static cogging is a waveform in rotor angle. Mapping it to time changes its frequency. Eddy currents can change the amplitude and phase of the ripple and produce a mean braking torque. Report those effects separately. For the symmetric 12-slot, 14-pole example, the first cogging order is 84 per revolution, or 1.4 kHz at 1,000 rpm. Use the frequency content seen by each conductor to assess eddy-current resolution. For example, the rotor sees slot passage at 200 Hz at that speed. [Existing rotary example](tools/magnetic_motor_example.py) · [Rotor-frame harmonic analysis](https://doi.org/10.1049/iet-epa.2014.0300)

## Proposed deck and results

Add a transient magnetic solution through the existing magnetic deck and CLI entry points. Use `SOL MAGTRANS` to select it. Reuse `MOTION` for the moving ESET, axis, start angle, and signed speed in RPM. In this solution, require its position `STEP` and `COUNT` fields to be blank and let `TSTEP` control time integration. The following deck excerpt shows the proposed controls.

```text
SOL MAGTRANS
CEND
SUBCASE 1
  MOTION = 10
  TSTEP = 20
  BFIELD(EVERY=20) = ALL
BEGIN BULK
MAG2D,0,0.025
MOTION,10,90,0,ROTATE,0.
,1000.
TSTEP,20,2.0E-5,12
MATMAG,5,1.05,,1.2
MATCOND,5,6.0E5
$ Mesh, PMAG2D properties, moving ESET 90, PAIRGAP, and boundaries
ENDDATA
```

- `TSTEP,ID,DTMAX,MAXCYCLES` gives the maximum step in seconds and the limit on complete mechanical revolutions for convergence. Subdivide each revolution into an integer number of equal steps. Initialize at the static open-circuit solution at the specified angle and compare successive revolutions at matched angles. Use `SOL MAGSTAT` for the zero-speed reference.
- In `MAGTRANS`, associate `MATCOND` in S/m with the matching field material ID. A field material without `MATCOND` is insulating. A material ID can be shared by several electrically separate regions. Use physical conductivity for bulk solids and leave laminated regions insulating in the first motor example. Review the actual material and insulation layout before using the example for a design estimate.
- Keep the first deck open-circuit. Reject prescribed winding currents and `DRIVE`, unsupported periodic reductions, and conductivity on an air-gap property with an input error naming the card or region. Use error values for invalid inputs, factorization failures, and incomplete runs.
- Write an F06 summary with signed RPM, mean and peak-to-peak torque in N·m after settling, mean eddy watts by property and total, and the cycle and power-balance residuals. Also record the actual time step, number of revolutions, and convergence status. Compute mechanical-order amplitudes and phases from the step data in the speed-sweep script. Reference phase to the deck's rotor angle, including for reverse rotation. If an order is below the numerical resolution, omit its phase.
- Write a `/MAGTRANS` HDF5 group with time in seconds, rotor angle, signed torque, and instantaneous losses at every accepted step. Export cell current density in A/m² and Joule loss density in W/m³ through VTKHDF. Use physical seconds on its shared time axis. `BFIELD(EVERY=n)` controls field output cadence independently of integration and torque sampling. Keep one transient run per speed initially and aggregate the results in the example script. [Existing magnetic output conventions](MAGNETICS.md#results-and-angle-sweeps)

## Discrete formulation

Use magnetoquasistatic magnetic diffusion with rigid prescribed motion. Store the potential history on nodes attached to each body. For `MAG2D`, the potential is $A_z$ and the body motion is in the plane, so the induced electric field uses the time derivative at a material point. In lab coordinates the same induction follows from $\mathbf E+\mathbf u\times\mathbf B$. With the body-attached derivative, adding a separate motional term would count the motion twice. Stationary conductors use the same assembly with zero body velocity. [Moving-conductor formulation and examples](https://doi.org/10.3311/PPee.8202)

For a floating planar conductor $C$, solve for one uniform axial electric-field offset $e_C$ by imposing zero net current.

$$
J_z = \sigma\left(e_C - \dot A_z\big|_{\text{material}}\right),
\qquad \int_C J_z\,\mathrm dS = 0.
$$

Build conductor components from electrically connected elements, with shared edges representing contact in the initial conforming mesh. A point contact or an insulating separation does not join components. The offset allows opposing axial currents with zero net current per isolated section. It represents the return constraint of the 2D model. Actual axial return paths and axial segmentation can change the losses and require 3D verification. [Planar current-return benchmark](https://femm.info/doku/doku.php?id=tubeexample)

Assemble the conductivity mass $M_\sigma$, the component columns $G_{iC}=\int_C \sigma N_i\,\mathrm dV$, and the diagonal $S_{CC}=\int_C\sigma\,\mathrm dV$, using the same planar depth and quadrature conventions as the field solve. Retain a scalar $b_C=\Delta t\,e_C$ per conductor to keep the assembly sparse. Backward Euler gives the following step for linear materials before boundary elimination.

$$
\begin{bmatrix}
K(q_{n+1})+M_\sigma/\Delta t & -G/\Delta t \\
-G^T/\Delta t & S/\Delta t
\end{bmatrix}
\begin{bmatrix}a_{n+1}\\b\end{bmatrix}
=
\begin{bmatrix}f(q_{n+1})+M_\sigma a_n/\Delta t\\-G^T a_n/\Delta t\end{bmatrix}.
$$

Here $q$ is the signed mechanical position and $f$ includes permanent-magnet excitation. Eliminating the offsets adds the positive semidefinite term $(M_\sigma-GS^{-1}G^T)/\Delta t$ to the gauged stiffness. Assemble the augmented sparse system directly to avoid forming dense conductor blocks. Reuse the existing magnetic residual and material tangent when adding nonlinear B–H support. Compute gap torque from the instantaneous solved field using the existing air-gap stress calculation and integrate $J^2/\sigma$ at conductor quadrature points for loss. Squaring the cell-averaged current would miss variation within a cell.

## Implementation sequence

1. **Establish the static baseline.** Use the zero-current cogging subcase in `tools/magnetic_motor_example.py`. Refine angle sampling and the gap trace mesh independently. Record torque versus angle, mean torque over a full period, and amplitude and phase at the first cogging order and its significant multiples. Use AGE as the reference. Keep a small linear deck for automated regressions and the 12-slot, 14-pole motor for the worked example. The existing validation shows that air-gap discretization alone can materially change cogging. [Air-gap validation](MAGNETICS.md#air-gap-pairgap) · [Independent FEMM comparison](Magnetic_Validation/README.md)
2. **Implement conductor assembly and one implicit step.** Add an internal transient solver under `src/magnetic/`. Reuse `Model`, magnetic mesh and quadrature, the gauge reduction, and `SolverBackend::factor`. Expose only the mesh and assembly pieces needed by both solve paths. Test isolated conductor constraints, uniform potential-rate cancellation, and magnetic diffusion before wiring in rotary motion. Use the pivot and residual checks appropriate to the real symmetric gauged system. [Magnetic assembly](src/magnetic/solver.cpp) · [Solver backend](include/solver/solver_backend.hpp)
3. **Advance a full-circle rotary model.** Reuse the moving-part map and AGE coupling. At each step rotate the rotor and remanence by the prescribed angle, update gap coupling, and preserve the node-to-node correspondence of the previous potential on each body. Cache conductivity assembly and reuse symbolic analysis when the sparsity pattern is fixed. Refactor when the gap or material tangent changes. Begin with a linear reference case, then use the existing B–H tangent for the worked motor where saturation affects the static baseline. Compare the conductive state at matching angles on successive revolutions in a consistent gauge, along with torque harmonics and mean loss. Use the full mechanical revolution as the initial repeat interval. The net cogging waveform can repeat sooner than the fields inside each conductor. [Motion implementation](src/magnetic/motion.cpp)
4. **Add the CLI, report, and worked speed sweep.** Teach the magnetic parser and dispatch to recognize `MAGTRANS` and validate its controls. Reuse the existing magnetic F06 and VTKHDF infrastructure. The example should run the static reference once and produce torque versus angle, the change from the static waveform at each speed, and a summary of mean braking torque, cogging-order amplitude and phase, and regional watts versus signed RPM. Report failed or unsettled speeds distinctly from converged periodic results. Record wall time, peak memory, and factorization counts for the worked example before enabling larger models. [Magnetic parser](src/magnetic/parser.cpp) · [Output implementation](src/magnetic/output.cpp)

## Acceptance checks

| Check | Expected result |
|---|---|
| Conductivity omitted from the rotary deck | The transient entry point reproduces the static torque waveform at matching angles to the solve tolerance. Eddy current and loss are zero. |
| Isolated planar conductors, including two disjoint regions sharing a material ID | Each region independently integrates to zero axial current. A spatially uniform change in potential on an isolated region leaves its induced current zero. |
| Conducting cylinder or annulus with one prescribed travelling harmonic | Field attenuation, phase, and integrated loss converge against the corresponding diffusion solution as mesh and time step are refined. Exercise the same relative excitation with moving and stationary descriptions to catch sign and frame errors. [Cylinder reference](https://www.aemjournal.org/index.php/AEM/article/view/719) · [Rotary diffusion reference](https://doi.org/10.1049/iet-epa.2014.0300) |
| Passive open-circuit rotary case at both signs of speed | The mean electromagnetic torque opposes rotation. Over a settled revolution, $P_{\mathrm{eddy}}$ approaches $-\Omega\bar T$ with time and mesh refinement when eddy dissipation is the only physical loss. Account for backward Euler's numerical dissipation when interpreting a finite-step imbalance. |
| Cogging speed sweep | Refine the time step and conductor and air-gap meshes separately. A reported amplitude or phase change must exceed the observed numerical variation. Compare at fixed mechanical orders and a common angular origin. Verify that decimating field output leaves scalar results unchanged. |
| CLI and output | A small complete transient deck writes physical seconds, torque and regional loss with the documented units. Invalid options and runs that reach the cycle cap without settling produce actionable diagnostics and a non-success status. |

Keep the physics checks in the magnetic unit suites, the deck and output checks in `tests/e2e/test_magnetic_cli.cpp`, and the larger independent comparisons under `Magnetic_Validation/`. Build with `ninja` from `build`, run the relevant Meson tests and the full suite before submission, then run `ninja cppcheck` from `build`.

## Follow-on phases

1. **Linear motion and periodic models.** Add `TRANSLATE` with signed m/s, using the existing strip and tube air-gap couplings. Flat `MAG2D` uses the planar return constraint. Tubular `MAGAXI` uses $A_\phi$, $2\pi r$ integration, and closed azimuthal currents. Verify material history across periodic wraps and anti-periodic sign changes before using reduced models. Compare tubular drag and dissipation against a magnet translating through a conducting tube. [Tubular benchmark](https://arxiv.org/abs/1407.0875)
2. **Prescribed currents and voltage drive.** Apply `DRIVE` at the instantaneous position and add time waveforms as needed. Use the reserved `CIRCDAT` for winding resistance and end-turn inductance when coupling terminal voltage to current and field linkage. Validate power balance with electrical input, mechanical output, stored energy, and all modeled losses.
3. **General 3D and harmonic analysis.** Add the conductor electric scalar potential to the edge-element field formulation, with insulating boundaries, terminals, and a compatible gauge. Validate current closure and losses in geometries with axial return paths or segmented sleeves and magnets. A harmonic solution can serve fixed-geometry AC cases and explicitly defined relative harmonics after the transient benchmarks establish the reference behavior.

## Implementation checks

`MAGTRANS` covers rotary, flat-linear, and tubular motion at prescribed speed, with open, current-driven, or voltage-driven windings. `MAGHARM` covers linear fixed-geometry problems, including 3D solid conductors with exterior electrodes. Automated checks include a transverse cylindrical diffusion solution, an axisymmetric sleeve in the weak reaction limit, periodic reductions against explicit full models, a shorted winding, and charge and power balance for 3D terminals.

The worked rotary example settled in three revolutions at both +600 and −600 rpm. The following +600 rpm runs use the same 12-slot, 14-pole geometry and illustrate independent time and mesh refinement. Timings cover the transient solve on two CPU threads. Peak RSS is the maximum across the solver processes in each run.

| Triangles | Steps per revolution | Mean torque (N·m) | Eddy loss (W) | Order-84 phase shift (deg) | Numeric factorizations | Wall time (s) | Peak RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2,553 | 336 | −0.000233 | 0.01534 | −0.0217 | 4,022 | 10.5 | 125 |
| 2,553 | 672 | −0.000248 | 0.01559 | −0.0238 | 6,866 | 20.0 | 135 |
| 3,682 | 672 | −0.000184 | 0.01163 | −0.0019 | 7,258 | 39.1 | 172 |
| 7,195 | 672 | −0.000169 | 0.01070 | −0.0034 | 7,670 | 89.6 | 304 |

The mean torque reverses with rotation. The coarser meshes give phase estimates that differ by more than the shift on the finest mesh, and mean loss differs by about 8% between the two finest meshes. These runs demonstrate the workflow. Quantifying the speed-dependent phase and loss requires additional mesh refinement.

## Open validation

- Establish a numerical floor for changes in cogging amplitude, phase, and loss by refining the conductor mesh and air-gap traces independently. The worked runs above vary the whole mesh together.
- Compare against the independent benchmark of a permanent magnet translating through a conducting tube. The automated tubular test covers the weak reaction limit for a sleeve translating through a coil field. [Translating-magnet benchmark](https://arxiv.org/abs/1407.0875)
- The implemented 3D A–V path uses fixed geometry and linear harmonic analysis. Prescribed motion with time-dependent fields uses the planar and axisymmetric paths.
