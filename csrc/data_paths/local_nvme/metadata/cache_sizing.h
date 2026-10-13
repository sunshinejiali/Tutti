#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace tutti::data_paths::local_nvme {

inline constexpr std::uint32_t kAutoPrpCacheCapacity =
    std::numeric_limits<std::uint32_t>::max();

// Bound cache backing to 64 MiB per DataPath, shared across controllers.
// Each sub-IO needs at most one LIST page; terminal unreleased ops retain pins.
// This is a cache budget, not a guarantee that every registered address fits.
inline std::uint32_t auto_prp_cache_capacity(std::uint32_t max_entries,
                                           std::uint32_t max_inflight,
                                           std::uint64_t page_size,
                                           std::uint64_t controllers) {
    if (page_size == 0 || controllers == 0) return 0;
    constexpr std::uint64_t budget = 64ull * 1024 * 1024;
    const auto pages = budget / controllers / page_size;
    if (max_entries == 0 || max_inflight == 0) return 0;
    // Saturate before multiplying: only demand up to the bounded budget matters.
    const std::uint64_t slots = 2ull * max_inflight;
    if (slots > pages / max_entries) return static_cast<std::uint32_t>(pages);
    return static_cast<std::uint32_t>(std::min(pages, slots * max_entries));
}

} // namespace tutti::data_paths::local_nvme
