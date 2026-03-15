#pragma once
// include/io/results.hpp + f06_writer.hpp combined
// Solution results and output in Nastran F06 format.

#include "core/model.hpp"
#include "core/types.hpp"
#include <filesystem>
#include <optional>
#include <ostream>
#include <unordered_map>

namespace nastran {

// ── Result data
// ───────────────────────────────────────────────────────────────

/// 6-DOF displacement at a node [T1,T2,T3,R1,R2,R3]
struct NodeDisplacement {
  constexpr NodeDisplacement() = default;
  NodeId node{0};
  std::array<double, 6> d{}; // indexed 0-5
};

/// Element stress at centroid (for plate elements)
struct PlateStress {
  constexpr PlateStress() = default;
  ElementId eid{0};
  double sx{0}, sy{0}, sxy{0};  // membrane stresses
  double mx{0}, my{0}, mxy{0};  // moments (plate bending)
  double von_mises{0}; // derived
};

/// Element stress for solid elements (at centroid)
struct SolidStress {
  constexpr SolidStress() = default;
  ElementId eid{0};
  double sx{0}, sy{0}, sz{0};
  double sxy{0}, syz{0}, szx{0};
  double von_mises{0};
};

struct SubCaseResults {
  int id{1};
  std::string label;

  std::vector<NodeDisplacement> displacements;
  std::vector<PlateStress> plate_stresses;
  std::vector<SolidStress> solid_stresses;
};

struct SolverResults {
  std::vector<SubCaseResults> subcases;
};

// ── F06 writer
// ────────────────────────────────────────────────────────────────

class F06Writer {
public:
  /// Write results to an F06 file
  static void write(const SolverResults &results, const Model &model,
                    const std::filesystem::path &path);

  /// Write to stream (for testing)
  static void write(const SolverResults &results, const Model &model,
                    std::ostream &out);

private:
  static void write_header(std::ostream &out);
  static void write_displacement_table(const SubCaseResults &sc,
                                       std::ostream &out);
  static void write_plate_stress_table(const SubCaseResults &sc,
                                       std::ostream &out);
  static void write_solid_stress_table(const SubCaseResults &sc,
                                       std::ostream &out);
};

} // namespace nastran
