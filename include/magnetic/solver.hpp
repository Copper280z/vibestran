#pragma once
#include "magnetic/mesh.hpp"

namespace vibestran::magnetic {
struct Source {
  int id{0};
  bool wire{false}, closed{true};
  double turns{1};
  std::vector<std::pair<int, int>>
      wire_elements; // element id, direction along path
  Mesh mesh;
  Eigen::VectorXd flux;
  Eigen::VectorXd coupling; // integral N_edge . J_per_ampere dV
  double conservation_error{0}, transfer_error{0};
};
struct SourceResults {
  int id{0};
  bool wire{false}, closed{true};
  double current{0}, linkage{0};
};
struct ForceResult {
  int id{0};
  Vector force{Vector::Zero()}, torque{Vector::Zero()};
};
struct CaseResults {
  int id{0}, iterations{0};
  double residual{0}, energy{0}, coenergy{0};
  bool extrapolated{false};
  Eigen::VectorXd potential;
  std::vector<Vector> b, h;
  std::vector<SourceResults> sources;
  Eigen::MatrixXd differential_inductance;
  std::vector<Vector> probe_b, probe_h;
  std::vector<ForceResult> forces;
};
struct Results {
  Mesh mesh;
  std::vector<Source> sources;
  std::vector<CaseResults> cases;
};
[[nodiscard]] Result<std::vector<Source>> prepare_sources(const Model &model,
                                                          const Mesh &mesh);
[[nodiscard]] Result<std::vector<Source>> prepare_wires(const Model &model);
[[nodiscard]] Result<Eigen::VectorXd>
conserved_flux(const Mesh &mesh, std::span<const std::pair<int, int>> section,
               std::span<const int> terminal_faces, bool stranded,
               double section_area, std::span<const double> resistivity);
[[nodiscard]] Result<Eigen::VectorXd> couple_source(const Source &source,
                                                    const Mesh &mesh);
[[nodiscard]] Result<std::vector<double>>
source_currents(const Model &model, std::span<const Source> sources, int set);
[[nodiscard]] Result<Results> solve(const Model &model);
[[nodiscard]] Status write_results(const Model &model, const Results &results,
                                   const std::filesystem::path &report);
} // namespace vibestran::magnetic
