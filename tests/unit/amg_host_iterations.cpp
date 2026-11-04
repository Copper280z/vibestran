// The host AMG solve the device tests compare against, in a file of its own:
// amg_factor.hpp brings in Eigen, whose headers stop cppcheck on the including
// file, so including it in a device test would hide that whole test from analysis.
#include "structural_system.hpp"
#include "amg_factor.hpp"

namespace vibestran::test {

int host_amg_iterations(const RecordingBackend::Record& r) {
    return detail::solve_hybrid_amg(r.k, r.f, 1e-8, 5000, {}, r.modes).iterations;
}

} // namespace vibestran::test
