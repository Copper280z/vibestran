#pragma once
// include/core/threads.hpp

namespace vibestran {

/// The physical cores this process may run on: its CPU affinity with hyperthread siblings
/// counted once, where the platform shows them, and otherwise the logical CPU count. At least 1.
[[nodiscard]] int physical_core_count();

} // namespace vibestran
