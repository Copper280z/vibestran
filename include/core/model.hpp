#pragma once
// include/core/model.hpp
// The FE model: nodes, elements, materials, properties, loads, SPCs.
// This is a pure data structure layer — no computation here.

#include "core/types.hpp"
#include "core/coord_sys.hpp"
#include "core/numbers.hpp"
#include <algorithm>
#include <cmath>
#include <array>
#include <iterator>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace vibestran {

// ── Grid point ───────────────────────────────────────────────────────────────

struct GridPoint {
  NodeId id{0};
  CoordId cp{0}; // input coordinate system
  Vec3 position{0, 0, 0};
  CoordId cd{0}; // displacement coordinate system
  /// Permanent single-point constraints (GRID field 8, PS), applied in every
  /// structural subcase in the displacement coordinate system `cd`.
  DofSet ps;
  /// A scalar point (SPOINT): one DOF, stored as component 1 (T1), with no
  /// location or displacement frame.  It shares the grid ID space.
  bool scalar{false};
};

// ── Materials ────────────────────────────────────────────────────────────────

struct Mat1 {
  MaterialId id{0};
  double E{0};        // Young's modulus
  double G{0};        // Shear modulus (0 = derived from E,nu)
  double nu{0};       // Poisson's ratio
  double rho{0};      // density
  double A{0};        // thermal expansion coefficient
  double ref_temp{0}; // reference temperature
  double st{0};       // tension allowable (0 = none)
  double sc{0};       // compression allowable (0 = none)
  double ss{0};       // shear allowable (0 = none)
};

struct Mat2 {
  MaterialId id{0};
  double g11{0};
  double g12{0};
  double g13{0};
  double g22{0};
  double g23{0};
  double g33{0};
  double rho{0};
  double a1{0};
  double a2{0};
  double a12{0};
  double ref_temp{0};
  double ge{0};
  double st{0};
  double sc{0};
  double ss{0};
  CoordId mcsid{0};
};

struct Mat3Material {
  MaterialId id{0};
  double ex{0};
  double ey{0};
  double ez{0};
  double nuxy{0};
  double nuyz{0};
  double nuzx{0};
  double rho{0};
  double gxy{0};
  double gyz{0};
  double gzx{0};
  double ax{0};
  double ay{0};
  double az{0};
  double ref_temp{0};
  double ge{0};
};

struct Mat4 {
  MaterialId id{0};
  double k{0};
  double cp{0};
};

struct Mat5 {
  MaterialId id{0};
  double kxx{0};
  double kxy{0};
  double kxz{0};
  double kyy{0};
  double kyz{0};
  double kzz{0};
  double cp{0};
};

struct Mat6 {
  MaterialId id{0};
  double g11{0};
  double g12{0};
  double g13{0};
  double g14{0};
  double g15{0};
  double g16{0};
  double g22{0};
  double g23{0};
  double g24{0};
  double g25{0};
  double g26{0};
  double g33{0};
  double g34{0};
  double g35{0};
  double g36{0};
  double g44{0};
  double g45{0};
  double g46{0};
  double g55{0};
  double g56{0};
  double g66{0};
  double rho{0};
  double axx{0};
  double ayy{0};
  double azz{0};
  double axy{0};
  double ayz{0};
  double azx{0};
  double ref_temp{0};
  double ge{0};
};

struct Mat8 {
  MaterialId id{0};
  double e1{0};
  double e2{0};
  double nu12{0};
  double g12{0};
  double g1z{0};
  double g2z{0};
  double rho{0};
  double a1{0};
  double a2{0};
  double ref_temp{0};
  double xt{0};
  double xc{0};
  double yt{0};
  double yc{0};
  double s{0};
  double ge{0};
  double f12{0};
};

/// Lazily built, thread-safe cache attached to a Model.  Copies and moves
/// start empty, so a cache never outlives the data it was built from; `key`
/// lets the owner detect a rebuilt model (e.g. a changed element count).
template <class T> class LazyModelCache {
public:
  LazyModelCache() = default;
  LazyModelCache(const LazyModelCache &) {}
  LazyModelCache(LazyModelCache &&) noexcept {}
  LazyModelCache &operator=(const LazyModelCache &) {
    reset();
    return *this;
  }
  LazyModelCache &operator=(LazyModelCache &&) noexcept {
    reset();
    return *this;
  }
  ~LazyModelCache() = default;

  template <class Build>
  [[nodiscard]] std::shared_ptr<const T> get(std::size_t key, Build &&build) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!value_ || key_ != key) {
      value_ = std::make_shared<const T>(build());
      key_ = key;
    }
    return value_;
  }

private:
  void reset() {
    const std::lock_guard<std::mutex> lock(mutex_);
    value_.reset();
    key_ = 0;
  }
  mutable std::mutex mutex_;
  mutable std::shared_ptr<const T> value_;
  mutable std::size_t key_{0};
};

// ── Properties ───────────────────────────────────────────────────────────────

/// PSOLID ISOP: EAS (blank, REDUCED or EAS) is the default enhanced-strain
/// element (Q1/E21 for CHEXA8), FULL the standard displacement element with
/// full integration, SRI vibestran's selectively reduced integration.
enum class SolidFormulation { SRI, EAS, Full };

struct PShell {
  PropertyId pid{0};
  MaterialId mid1{0};     // membrane material
  double t{0};            // thickness
  MaterialId mid2{0};     // bending material (0 = same as mid1)
  double twelveI_t3{1.0}; // 12I/t^3 bending stiffness factor
  MaterialId mid3{0};     // transverse shear material
  double tst{0.833333};   // transverse shear thickness ratio
  double nsm{0.0};        // non-structural mass / unit area
  double z1{std::numeric_limits<double>::quiet_NaN()}; // stress recovery fiber distance 1
  double z2{std::numeric_limits<double>::quiet_NaN()}; // stress recovery fiber distance 2
  MaterialId mid4{0};     // membrane-bending coupling
};

/// One ply of a PCOMP laminate, listed from the bottom (Z0) upwards.
struct PCompPly {
  MaterialId mid{0};
  double t{0.0};     // ply thickness
  double theta{0.0}; // fibre angle (degrees) from the element material axis
  bool sout{false};  // ply stress/strain output requested
};

/// PCOMP laminate options (LAM field).
enum class LaminateOption { Default, Sym, Mem, Bend, Smear, SmCore };

/// PCOMP composite shell property.  `plies` holds the full stack: a SYM
/// laminate's listed half is mirrored by the parser.
struct PComp {
  PropertyId pid{0};
  /// Distance from the reference surface to the bottom of the laminate
  /// (NaN = default, -total_thickness / 2).
  double z0{std::numeric_limits<double>::quiet_NaN()};
  double nsm{0.0};
  double sb{0.0};          // allowable interlaminar shear (failure; unused)
  std::string ft;          // failure theory (unused)
  double tref{0.0};        // reference temperature
  double ge{0.0};          // structural damping (unused)
  LaminateOption lam{LaminateOption::Default};
  std::vector<PCompPly> plies;

  [[nodiscard]] double total_thickness() const {
    return std::accumulate(plies.begin(), plies.end(), 0.0,
                           [](double t, const PCompPly &p) { return t + p.t; });
  }
  [[nodiscard]] double bottom_z() const {
    return std::isnan(z0) ? -0.5 * total_thickness() : z0;
  }
};

struct PSolid {
  PropertyId pid{0};
  MaterialId mid{0};
  int cordm{0}; // material coordinate system
  SolidFormulation isop{SolidFormulation::EAS};
  /// PSOLID IN: Gauss points per direction (2 for blank, BUBBLE, 0 or TWO;
  /// 3 for THREE).
  int integration_order{2};
};

struct PBar {
  PropertyId pid{0};
  MaterialId mid{0};
  double A{0.0};
  double I1{0.0};  ///< Bending inertia in plane 1 (element x-y plane), Izz
  double I2{0.0};  ///< Bending inertia in plane 2 (element x-z plane), Iyy
  double J{0.0};
  double nsm{0.0};
  /// Stress recovery points C, D, E, F as (y, z) in the element frame.
  std::array<std::array<double, 2>, 4> recovery_points{};
  /// Transverse shear area factors (shear stiffness K*A*G).  Zero (blank on
  /// the card) means the bar is rigid in transverse shear.
  double K1{0.0};
  double K2{0.0};
  double I12{0.0}; ///< Area product of inertia, integral of y*z dA
};

struct PBarL {
  PropertyId pid{0};
  MaterialId mid{0};
  std::string section_type;
  std::vector<double> dimensions;
  double A{0.0};
  double I1{0.0};
  double I2{0.0};
  double J{0.0};
  double nsm{0.0};
};

struct PBeam {
  PropertyId pid{0};
  MaterialId mid{0};
  double A{0.0};
  double I1{0.0};
  double I2{0.0};
  double I12{0.0};
  double J{0.0};
  double nsm{0.0};
};

/// PROD: section of a CROD, which carries axial force and torque only.
struct PRod {
  PropertyId pid{0};
  MaterialId mid{0};
  double A{0.0};
  double J{0.0};
  double C{0.0};   ///< Torsional stress recovery coefficient, tau = C * T / J
  double nsm{0.0};
};

struct PBush {
  PropertyId pid{0};
  std::array<double, 6> k{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

struct PElas {
  PropertyId pid{0};
  double k{0.0};
  double ge{0.0};
  double s{0.0};
};

struct PMass {
  PropertyId pid{0};
  double mass{0.0};
};

using Property = std::variant<PShell, PSolid, PBar, PBarL, PBeam, PBush,
                              PElas, PMass, PComp, PRod>;

// ── Elements ─────────────────────────────────────────────────────────────────

enum class ElementType {
  CQUAD4,
  CTRIA3,
  CHEXA8,
  CHEXA20,
  CTETRA4,
  CTETRA10,
  CPENTA6,
  CBAR,
  CBEAM,
  CROD,
  CBUSH,
  CELAS1,
  CELAS2,
  CMASS1,
  CMASS2,
  CONM2,
  CHBDY,
};

/// CHBDY geometry types (FLAG field on the CHBDY card).
/// Mirrors NASTRAN-95's HBDY geometry codes; LINE/REV/ELCYL/FTUBE are reserved
/// for future use.
enum class ChbdyType {
  POINT,
  LINE,
  REV,
  AREA3,
  AREA4,
  ELCYL,
  FTUBE,
};

/// PHBDY card: properties for CHBDY boundary element (NASTRAN PHBDY).
/// AF is the area factor for POINT/LINE/ELCYL/FTUBE geometries.
struct PHBDY {
  PropertyId pid{0};
  MaterialId mid{0}; ///< MAT4 carrying film coefficient and surface capacity
  double af{1.0};    ///< area factor for POINT/LINE/ELCYL/FTUBE
  double emissivity{0.0};
  double absorptivity{0.0}; ///< defaults to emissivity when blank
  double r1{0.0};
  double r2{0.0};
};

/// CHBDY boundary element (parsed at top level, stored separately from solid
/// elements because its DOF/connectivity model is different).
struct ChbdyElement {
  ElementId eid{0};
  PropertyId pid{0};      ///< references a PHBDY
  ChbdyType geom{ChbdyType::AREA3};
  std::vector<NodeId> nodes;     ///< surface nodes (1, 2, 3, or 4)
  /// Ambient grids GA1..GA4 correspond positionally to the surface grids.
  /// NodeId{0} denotes a blank field, i.e. a fixed zero-temperature ambient.
  std::vector<NodeId> ambient_nodes;
  Vec3 orientation{0.0, 0.0, 0.0};
};

struct ElementData {
  ElementId id{0};
  ElementType type{0};
  PropertyId pid{0};
  std::vector<NodeId> nodes;
  std::optional<double> theta{std::nullopt};
  std::optional<CoordId> mcid{std::nullopt};
  double zoffs{0.0};
  /// CQUAD4/CTRIA3 grid-point membrane thicknesses T1..Tn (continuation
  /// fields); empty when not given.  With TFLAG = 1 they are factors on the
  /// property thickness, otherwise actual thicknesses; a blank Ti means the
  /// property thickness.
  std::vector<std::optional<double>> corner_thickness;
  int tflag{0};
  std::optional<Vec3> orientation{std::nullopt};
  std::optional<NodeId> g0{std::nullopt};
  std::optional<CoordId> cid{std::nullopt};
  std::array<int, 2> components{0, 0};
  double value{0.0};
  /// CBAR/CBEAM pin flags (PA, PB): released element-frame DOFs at ends A
  /// and B, as a bitmask where bit k releases component k+1.
  std::array<std::uint8_t, 2> pin_flags{0, 0};
  /// CBAR/CBEAM rigid end offsets (WA, WB) from GA/GB to the element ends,
  /// expressed in the displacement coordinate system of the respective grid.
  std::array<Vec3, 2> offsets{};
  /// CBAR OFFT first character 'B': the orientation vector is in the basic
  /// system instead of the displacement coordinate system of GA.
  bool orientation_in_basic{false};

  /// CONM2 data: mass, offset X of the centre of gravity, and inertia
  /// {I11, I21, I22, I31, I32, I33} about the CG.  With cid >= 0 the offset
  /// is measured from the grid in system `cid` and the inertia is expressed in
  /// `cid`; with cid = -1 the offset is the basic location of the CG and the
  /// inertia is in basic.  Off-diagonal inertia terms enter the inertia matrix
  /// with a minus sign (Nastran convention).
  struct ConcentratedMass {
    double mass{0.0};
    Vec3 offset{};
    std::array<double, 6> inertia{};
    int cid{0};
  };
  std::optional<ConcentratedMass> conm2{std::nullopt};
};

// ── Loads
// ─────────────────────────────────────────────────────────────────────

struct ForceLoad {
  LoadSetId sid{0};
  NodeId node{0};
  CoordId cid{0};
  double scale{1.0};
  Vec3 direction;
  // Effective force = scale * direction (direction includes magnitude in
  // Nastran)
};

struct MomentLoad {
  LoadSetId sid{0};
  NodeId node{0};
  CoordId cid{0};
  double scale{1.0};
  Vec3 direction;
};

/// Nodal temperature (TEMP card)
struct TempLoad {
  LoadSetId sid{0};
  NodeId node{0};
  double temperature{0};
};

/// Temperature field reference for elements (not needed at nodes — handled via
/// TEMP)
struct TempRefLoad {
  LoadSetId sid{0};
  double t_ref{0}; // from TEMPD or analysis case
};

/// SLOAD card: a load on one scalar point, or on component T1 of a grid in its
/// displacement frame.  In heat transfer it is a power into the point.
struct SloadLoad {
  LoadSetId sid{0};
  NodeId node{0};
  double value{0.0};
};

/// PLOAD card: pressure on 3 or 4 explicitly listed grid points.
struct PloadLoad {
  LoadSetId sid{0};
  double pressure{0.0};
  std::vector<NodeId> nodes;
};

/// PLOAD1 card: line pressure on bar/beam-type elements.
/// This is currently parsed and carried through the model, but no supported
/// element types consume it yet.
struct Pload1Load {
  LoadSetId sid{0};
  ElementId element{0};
  std::string load_type;
  std::string scale_type;
  double x1{0.0};
  double p1{0.0};
  std::optional<double> x2;
  std::optional<double> p2;
};

/// PLOAD2 card: uniform pressure on a 2-D element.
struct Pload2Load {
  LoadSetId sid{0};
  ElementId element{0};
  double pressure{0.0};
};

/// PLOAD4 card: pressure on an element face, optionally with an explicit
/// direction vector.
struct Pload4Load {
  LoadSetId sid{0};
  ElementId element{0};
  std::array<double, 4> pressures{0.0, 0.0, 0.0, 0.0};
  bool use_vector{false};
  CoordId cid{0};
  Vec3 direction{0.0, 0.0, 0.0};
  std::optional<NodeId> face_node1;
  std::optional<NodeId> face_node34;
};

/// GRAV card: uniform translational acceleration applied to all mass.
struct GravLoad {
  LoadSetId sid{0};
  CoordId cid{0};
  double scale{0.0};
  Vec3 direction{0.0, 0.0, 0.0};
};

/// ACCEL1 card: uniform translational acceleration applied to selected grids.
struct Accel1Load {
  LoadSetId sid{0};
  CoordId cid{0};
  double scale{0.0};
  Vec3 direction{0.0, 0.0, 0.0};
  std::vector<NodeId> nodes;
};

/// ACCEL card stub. Parsed so unsupported usage can fail explicitly.
struct AccelLoad {
  LoadSetId sid{0};
  CoordId cid{0};
  double scale{0.0};
  Vec3 direction{0.0, 0.0, 0.0};
};

/// QVOL — volumetric heat generation rate on structural elements (W/volume).
struct QvolLoad {
  LoadSetId sid{0};
  double q_vol{0.0};
  std::vector<ElementId> elements;
};

/// QHBDY — distributed heat flux on a directly-listed boundary patch
/// (does not reference a CHBDY element; carries its own geometry tag and grids).
struct QhbdyLoad {
  LoadSetId sid{0};
  ChbdyType geom{ChbdyType::AREA3};
  double q0{0.0};
  double af{1.0};
  std::vector<NodeId> nodes;
};

/// QBDY1 — uniform heat flux on listed CHBDY elements.
struct Qbdy1Load {
  LoadSetId sid{0};
  double q0{0.0};
  std::vector<ElementId> elements;
};

/// QBDY2 — per-node heat flux on a single CHBDY element.
struct Qbdy2Load {
  LoadSetId sid{0};
  ElementId element{0};
  std::array<double, 4> q{0.0, 0.0, 0.0, 0.0};
};

/// QVECT — directional heat flux vector applied to CHBDY elements (e.g. solar
/// irradiance). Only the inward-facing component (−E·n̂ > 0) is loaded; faces
/// pointed away from the source receive no flux (matches NASTRAN-95).
struct QvectLoad {
  LoadSetId sid{0};
  double q0{0.0};           ///< magnitude scale
  Vec3 direction{0,0,0};    ///< flux propagation direction (basic frame)
  std::vector<ElementId> elements;
};

using Load = std::variant<ForceLoad, MomentLoad, SloadLoad, TempLoad, PloadLoad,
                          Pload1Load, Pload2Load, Pload4Load, GravLoad,
                          Accel1Load, AccelLoad,
                          QvolLoad, QhbdyLoad, Qbdy1Load, Qbdy2Load,
                          QvectLoad>;

/// LOAD bulk data card: combines multiple load sets with scale factors.
/// The effective load = overall_scale * sum(Si * loads_with_SID_Li)
struct LoadCombination {
  LoadSetId sid{0};
  double overall_scale{1.0};
  std::vector<std::pair<double, LoadSetId>> entries; // (Si, Li)
};

// ── Single point constraints
// ──────────────────────────────────────────────────

struct Spc {
  SpcSetId sid{0};
  NodeId node{0};
  DofSet dofs;
  double value{0}; // enforced displacement (0 = fixed)
};

// ── Multi-point constraints
// ─────────────────────────────────────────────────

/// One term in an MPC equation: coeff * u[node, dof]
struct MpcTerm {
  NodeId node{0};
  int dof{0};    // 1-based DOF (1-6)
  double coeff{0.0};
};

/// One MPC equation: sum_i coeff_i * u[node_i, dof_i] = rhs.
/// Nastran defines the first term as the dependent DOF. rhs is zero for BDF
/// MPC cards and nonzero for internally generated enforced-displacement rows.
struct Mpc {
  MpcSetId sid{0};
  std::vector<MpcTerm> terms;
  double rhs{0.0};
  /// Internally generated SPC equation; its force belongs to SPCFORCES.
  bool is_spc{false};
};

// ── Rigid elements
// ──────────────────────────────────────────────────────────

/// RBE2 — rigid body element with independent node GN and dependent nodes GM
struct Rbe2 {
  ElementId eid{0};
  NodeId gn{0};       // independent node
  DofSet cm;          // constrained DOFs (on dependent nodes)
  std::vector<NodeId> gm; // dependent nodes
};

/// One weight group in an RBE3
struct Rbe3WeightGroup {
  double weight{1.0};
  DofSet component;         // active DOF components for this group
  std::vector<NodeId> nodes;
};

/// RBE3 — interpolation constraint element
struct Rbe3 {
  ElementId eid{0};
  NodeId ref_node{0};   // reference (dependent) node
  DofSet refc;          // constrained DOFs on reference node
  std::vector<Rbe3WeightGroup> weight_groups;
};

// ── SNORM card (user shell grid normal) ─────────────────────────────────────

/// SNORM: the shell normal at a grid, components in coordinate system `cid`.
/// It replaces the averaged director of every CQUAD4/CTRIA3 at the grid
/// (shell_directors.hpp).
struct Snorm {
  NodeId grid{0};
  CoordId cid{0};
  Vec3 n;
};

// ── EIGRL card (modal analysis) ──────────────────────────────────────────────

struct EigRL {
  int sid{0};
  double v1{0.0};   // lower frequency bound (Hz); 0 = unconstrained
  double v2{1e30};  // upper frequency bound (Hz)
  int nd{10};       // number of desired eigenvalues
  enum class Norm { Mass, Max } norm{Norm::Mass};
};

// ── Analysis case
// ─────────────────────────────────────────────────────────────

/// Subcase-scoped ANALYSIS keyword.  Used by the SOL 153 driver to mix heat
/// and statics passes in a single deck (one-way coupling).
enum class SubCaseAnalysis {
  Default,  ///< inherit from solution sequence (SOL 101 → Statics, SOL 153 → Heat)
  Heat,
  Statics,
};

struct SubCase {
  int id{1};
  std::string label;
  LoadSetId load_set{0};
  SpcSetId spc_set{0};
  MpcSetId mpc_set{0}; // 0 = no MPCs
  int temp_load_set{0}; // TEMPERATURE(LOAD) / TEMP(LOAD) set ID
  /// True when TEMP(LOAD)=THERMAL was given: structural pass consumes the
  /// most recent heat subcase's temperature field instead of TEMP cards.
  bool temp_from_heat{false};
  double t_ref{0}; // reference temperature for thermal load
  SubCaseAnalysis analysis_type{SubCaseAnalysis::Default};

  // Output selection (case control deck).
  // PRINT → text output (F06, CSV); PLOT → binary output (OP2).
  // Both default to false; set by DISPLACEMENT/STRESS case control entries.
  // No output card → no output.  NONE → clears both for that result type.
  bool disp_print{false};   // DISPLACEMENT(PRINT)=ALL  or  DISPLACEMENT=ALL
  bool disp_plot{false};    // DISPLACEMENT(PLOT)=ALL
  bool stress_print{false}; // STRESS(PRINT)=ALL  or  STRESS=ALL
  bool stress_plot{false};  // STRESS(PLOT)=ALL
  bool stress_corner_print{false}; // STRESS(PRINT,CORNER)=ALL
  bool stress_corner_plot{false};  // STRESS(PLOT,CORNER)=ALL
  bool strain_print{false};        // STRAIN(PRINT)=ALL  or  STRAIN=ALL
  bool strain_corner_print{false}; // STRAIN(PRINT,CORNER)=ALL
  bool strain_fiber{false};        // STRAIN(FIBER): shell strains at Z1/Z2
                                   // instead of membrane strain + curvature
  bool strain_max_shear{false};    // STRAIN(SHEAR|MAXS): max shear strain
                                   // instead of von Mises strain
  bool gpstress_print{false};      // GPSTRESS(PRINT)=ALL
  bool gpstress_plot{false};       // GPSTRESS(PLOT)=ALL
  bool spc_force_print{false};     // SPCFORCE(PRINT)=ALL or SPCFORCE=ALL
  bool spc_force_plot{false};      // SPCFORCE(PLOT)=ALL
  bool mpc_force_print{false};     // MPCFORCE(PRINT)=ALL or MPCFORCE=ALL
  bool mpc_force_plot{false};      // MPCFORCE(PLOT)=ALL
  bool force_print{false};         // FORCE/ELFORCE(PRINT)=ALL (engineering)
  bool force_plot{false};          // FORCE/ELFORCE(PLOT)=ALL
  bool force_corner{false};        // FORCE(CORNER|BILIN): shell corner rows
  bool gpforce_print{false};       // GPFORCE(PRINT)=ALL or GPFORCE=ALL
  bool gpforce_plot{false};        // GPFORCE(PLOT)=ALL

  // Modal analysis (SOL 103) output selection
  int eigrl_id{0};        // METHOD = <sid>  references an EIGRL card
  bool eigvec_print{false}; // EIGENVECTOR(PRINT)=ALL
  bool eigvec_plot{false};  // EIGENVECTOR(PLOT)=ALL

  [[nodiscard]] bool has_any_stress_print() const noexcept {
    return stress_print || stress_corner_print || gpstress_print;
  }

  [[nodiscard]] bool has_any_stress_plot() const noexcept {
    return stress_plot || stress_corner_plot || gpstress_plot;
  }

  [[nodiscard]] bool has_any_stress_output() const noexcept {
    return has_any_stress_print() || has_any_stress_plot();
  }

  [[nodiscard]] bool has_any_strain_output() const noexcept {
    return strain_print || strain_corner_print;
  }
};

struct AnalysisCase {
  SolutionType sol{SolutionType::LinearStatic};
  std::vector<SubCase> subcases;
};

// ── The Model
// ─────────────────────────────────────────────────────────────────

class Model {
public:
  // Nodes
  std::unordered_map<NodeId, GridPoint> nodes;

  // Elements
  std::vector<ElementData> elements;

  // Materials
  std::unordered_map<MaterialId, Mat1> materials;
  std::unordered_map<MaterialId, Mat2> mat2_materials;
  std::unordered_map<MaterialId, Mat3Material> mat3_materials;
  std::unordered_map<MaterialId, Mat4> mat4_materials;
  std::unordered_map<MaterialId, Mat5> mat5_materials;
  std::unordered_map<MaterialId, Mat6> mat6_materials;
  std::unordered_map<MaterialId, Mat8> mat8_materials;

  // Properties
  std::unordered_map<PropertyId, Property> properties;

  // PHBDY properties (separate map; not part of Property variant since they
  // are only used by CHBDY boundary elements).
  std::unordered_map<PropertyId, PHBDY> phbdy_properties;

  // CHBDY boundary elements (kept separate from `elements` since they have a
  // different DOF model and only participate in thermal analyses).
  std::vector<ChbdyElement> chbdy_elements;

  // Loads (keyed by set id for quick lookup)
  std::vector<Load> loads;

  // LOAD bulk data card combinations (SID → combination)
  std::unordered_map<LoadSetId, LoadCombination> load_combinations;

  // SPCs
  std::vector<Spc> spcs;

  // MPCs
  std::vector<Mpc> mpcs;

  // Rigid elements
  std::vector<Rbe2> rbe2s;
  std::vector<Rbe3> rbe3s;

  // Coordinate systems (CoordId{0} = basic, never stored here)
  std::unordered_map<CoordId, CoordSys> coord_systems;

  // Default temperatures per set (TEMPD cards): set_id → temperature
  std::unordered_map<int, double> tempd;

  // Analysis
  AnalysisCase analysis;

  // SNORM cards (user shell grid normals), keyed by grid
  std::unordered_map<NodeId, Snorm> snorms;

  // PARAM entries (name → value as string)
  std::unordered_map<std::string, std::string> params;

  /// A numeric PARAM, or nullopt when it is absent or blank. The BDF parser
  /// rejects malformed numeric PARAMs with the card's line, so a malformed
  /// value here means a model built elsewhere; that is reported as an error
  /// rather than replaced by a default.
  [[nodiscard]] std::optional<double> param_real(const std::string &name) const {
    const auto it = params.find(name);
    if (it == params.end() || detail::trim_blanks(it->second).empty())
      return std::nullopt;
    if (const auto value = parse_real(it->second))
      return value;
    throw SolverError("PARAM," + name + " value '" + it->second +
                      "' is not a real number");
  }
  [[nodiscard]] std::optional<int> param_integer(const std::string &name) const {
    const auto it = params.find(name);
    if (it == params.end() || detail::trim_blanks(it->second).empty())
      return std::nullopt;
    if (const auto value = parse_integer(it->second))
      return value;
    throw SolverError("PARAM," + name + " value '" + it->second +
                      "' is not an integer");
  }

  // EIGRL cards (keyed by SID)
  std::unordered_map<int, EigRL> eigrls;

  /// Shell grid adjacency: for each grid, the (element index, corner) pairs
  /// of the CQUAD4/CTRIA3 elements that use it.  Built on first use by the
  /// shell elements to form shared grid directors; see shell_directors.hpp.
  using ShellAdjacency =
      std::unordered_map<NodeId, std::vector<std::pair<std::size_t, int>>>;
  mutable LazyModelCache<ShellAdjacency> shell_adjacency_cache;

  // ── Accessors ────────────────────────────────────────────────────────────

  const GridPoint &node(NodeId id) const {
    auto it = nodes.find(id);
    if (it == nodes.end())
      throw SolverError(std::format("Node {} not found in model", id.value));
    return it->second;
  }

  [[nodiscard]] const char *structural_material_card_name(MaterialId id) const {
    if (materials.count(id))
      return "MAT1";
    if (mat2_materials.count(id))
      return "MAT2";
    if (mat3_materials.count(id))
      return "MAT3";
    if (mat6_materials.count(id))
      return "MAT6";
    if (mat8_materials.count(id))
      return "MAT8";
    return nullptr;
  }

  [[nodiscard]] const char *material_card_name(MaterialId id) const {
    if (const char *card = structural_material_card_name(id))
      return card;
    if (mat4_materials.count(id))
      return "MAT4";
    if (mat5_materials.count(id))
      return "MAT5";
    return nullptr;
  }

  [[nodiscard]] bool has_structural_material(MaterialId id) const {
    return structural_material_card_name(id) != nullptr;
  }

  [[nodiscard]] size_t material_count() const {
    return materials.size() + mat2_materials.size() + mat3_materials.size() +
           mat4_materials.size() + mat5_materials.size() +
           mat6_materials.size() + mat8_materials.size();
  }

  const Mat1 &material(MaterialId id) const {
    auto it = materials.find(id);
    if (it != materials.end())
      return it->second;
    if (const char *card = structural_material_card_name(id)) {
      throw SolverError(std::format(
          "Material {} is defined on {}, but current element formulations "
          "require MAT1",
          id.value, card));
    }
    if (const char *card = material_card_name(id)) {
      throw SolverError(std::format(
          "Material {} is defined on {}, which is not a structural material "
          "card for the current element formulations",
          id.value, card));
    }
    throw SolverError(std::format("Material {} not found", id.value));
  }

  const Property &property(PropertyId id) const {
    auto it = properties.find(id);
    if (it == properties.end())
      throw SolverError(std::format("Property {} not found", id.value));
    return it->second;
  }

  /// Retrieve all loads for a given set, along with the combined scale factor
  /// from any LOAD bulk data card indirection. If the SID refers to a LOAD
  /// card, loads from each referenced sub-set are returned with scale =
  /// overall_scale * Si. Otherwise each load is returned with scale = 1.0.
  std::vector<std::pair<const Load *, double>> loads_for_set(LoadSetId sid) const {
    std::vector<std::pair<const Load *, double>> result;

    auto it = load_combinations.find(sid);
    if (it != load_combinations.end()) {
      // Resolve through LOAD card
      const LoadCombination &combo = it->second;
      for (const auto &[entry_scale, entry_sid] : combo.entries) {
        const double combined = combo.overall_scale * entry_scale;
        for (const auto &load : loads) {
          auto has_sid = [&](const auto &l) { return l.sid == entry_sid; };
          if (std::visit(has_sid, load))
            result.emplace_back(&load, combined);
        }
      }
    } else {
      // Direct SID lookup — no LOAD card
      for (const auto &load : loads) {
        auto has_sid = [&](const auto &l) { return l.sid == sid; };
        if (std::visit(has_sid, load))
          result.emplace_back(&load, 1.0);
      }
    }
    return result;
  }

  /// Retrieve all SPCs for a given set
  std::vector<const Spc *> spcs_for_set(SpcSetId sid) const {
    std::vector<const Spc *> result;
    for (const auto &spc : spcs) {
      if (spc.sid == sid)
        result.push_back(&spc);
    }
    return result;
  }

  /// Single-point constraints active in a structural subcase: the subcase's
  /// SPC set plus one zero-valued constraint per grid with GRID PS
  /// components.  Heat-transfer analyses use spcs_for_set() only, since PS
  /// constrains structural components.
  // cppcheck-suppress unusedFunction -- called from analysis_support.cpp, linear_static.cpp and quality_checks.cpp
  std::vector<Spc> structural_spcs(SpcSetId sid) const {
    // Permanent constraints come first so that an explicit SPC with an
    // enforced value on the same component (merged later by consumers)
    // takes precedence over the zero-valued PS entry.
    std::vector<Spc> result;
    for (const auto &[nid, gp] : nodes) {
      if (gp.ps.mask == 0)
        continue;
      Spc permanent;
      permanent.sid = sid;
      permanent.node = nid;
      permanent.dofs = gp.ps;
      result.push_back(permanent);
    }
    std::copy_if(spcs.begin(), spcs.end(), std::back_inserter(result),
                 [sid](const Spc &spc) { return spc.sid == sid; });
    return result;
  }

  /// Mass formulation.  PARAM,COUPMASS <= 0 selects lumped mass (Nastran's
  /// default); a positive value, or no PARAM, selects consistent mass, which
  /// is vibestran's default.
  [[nodiscard]] bool use_lumped_mass() const {
    const auto coupmass = param_real("COUPMASS");
    return coupmass && *coupmass <= 0.0;
  }

  /// Number of entries of `nodes` that are scalar points.
  [[nodiscard]] std::size_t scalar_point_count() const {
    return static_cast<std::size_t>(
        std::count_if(nodes.begin(), nodes.end(),
                      [](const auto &entry) { return entry.second.scalar; }));
  }

  /// Retrieve all MPCs for a given set
  std::vector<const Mpc *> mpcs_for_set(MpcSetId sid) const {
    std::vector<const Mpc *> result;
    for (const auto &mpc : mpcs) {
      if (mpc.sid == sid)
        result.push_back(&mpc);
    }
    return result;
  }

  /// After parsing, transform all GridPoint positions from their CP frame to
  /// basic Cartesian.  Also updates GridPoint.cp to CoordId::basic() after
  /// transformation so position is always in basic once called.
  /// Must be called after all GRID and CORDxx cards have been parsed.
  void resolve_coordinates();

  /// Validate consistency of the model (throws on error)
  void validate() const;
};

} // namespace vibestran
