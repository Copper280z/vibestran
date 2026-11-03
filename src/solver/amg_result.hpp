#pragma once
// The outcome of one AMG PCG solve, kept apart from amg_pcg.hpp so backends
// can report it without including AMGCL.
#include "solver/iterative_refinement.hpp"
#include <vector>

namespace vibestran::detail {
struct AmgPcgResult {
    std::vector<double> solution;
    RefinementReport refinement;
    int iterations{0};
};
} // namespace vibestran::detail
