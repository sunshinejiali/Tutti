#pragma once
// csrc/data_paths/striped_local_nvme/async_submit_kernel.cuh
//
// Kernels of StripedAsyncDataPath (design: doc/design/async-nvme-completion.md):
//   * submit-only kernel: the worker pool of fused_submit_kernel_pool, but
//     each entry is issued with submit_no_wait (no completion wait);
//   * wait_batches kernel: one warp that drains every queue of the DataPath
//     until the listed batches have finished; a deadline reports timeout but
//     is not permission to retire a kernel before the IO has completed.

#include <cstdint>

#include "csrc/data_paths/local_nvme/io/nvme_async_primitives.cuh"
#include "csrc/data_paths/striped_local_nvme/fused_submit_kernel.cuh"

namespace tutti::data_paths::striped_local_nvme {

using tutti::data_paths::local_nvme::async_io::CompletionTables;

// Batches one wait_batches kernel can wait for; longer lists launch several.
constexpr std::uint32_t kWaitBatchesPerKernel = 16;

struct WaitList {
    std::uint32_t count = 0;
    std::uint32_t slot[kWaitBatchesPerKernel] = {};
    std::uint32_t batch_id[kWaitBatchesPerKernel] = {};
};

cudaError_t launch_submit_no_wait(const StripedDeviceSubmitEntry* d_entries,
                                  EntryCompletionStatus* d_status,
                                  const DeviceTargetHandle* const* d_dev_table,
                                  std::uint32_t count,
                                  std::uint32_t num_devs,
                                  void* stream,
                                  std::uint32_t pool_workers,
                                  unsigned int* d_task_counter,
                                  const CompletionTables& tables,
                                  std::uint32_t slot,
                                  std::uint32_t batch_id);

cudaError_t launch_wait_batches(const CompletionTables& tables,
                                const WaitList& list,
                                unsigned long long timeout_ns,
                                void* stream);

// sizeof(QueuePair): the libnvm type is only complete under nvcc.
std::size_t queue_pair_bytes();

} // namespace tutti::data_paths::striped_local_nvme
