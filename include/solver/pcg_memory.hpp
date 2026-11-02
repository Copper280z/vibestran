#pragma once
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace vibestran {
// Shared optional device budget for PCG, in bytes. An explicit configuration
// wins over the environment. Invalid environment values leave automatic sizing.
inline std::size_t pcg_device_budget(std::size_t configured) noexcept {
    if (configured) return configured;
    const char* env = std::getenv("VIBESTRAN_PCG_DEVICE_MEMORY_MIB");
    if (!env) return 0;
    const std::string_view text(env);
    std::size_t mib = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), mib);
    if (error != std::errc{} || end != text.data() + text.size() ||
        mib > std::numeric_limits<std::size_t>::max() / 1048576) return 0;
    return mib * 1048576;
}
}
