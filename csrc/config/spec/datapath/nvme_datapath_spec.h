#pragma once

#include <cstdint>

namespace tutti::config {

struct NvmeDataPathTuning {
    // Internal admission/workspace bounds, not GPU utilization targets.
    // Native callers without workload hints use 16 operations / 256 entries.
    // Model adapters resolve their own budgets before DataPath initialization.
    // The arena preallocates 2 * operations slots; every slot holds entries.
    // Large values multiply HBM usage, so there is no cheap unlimited maximum.
    // Zero explicitly selects the same finite defaults. Advanced callers may
    // override for known workloads; oversized batches must be split by callers.
    std::uint64_t max_in_flight_operations = 16;
    std::uint64_t max_batch_entries = 256;
};

} // namespace tutti::config
