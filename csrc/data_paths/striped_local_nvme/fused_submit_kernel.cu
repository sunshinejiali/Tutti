// csrc/data_paths/striped_local_nvme/fused_submit_kernel.cu
//
// Host launcher for the fused multi-device submit kernel.
// Compiled by nvcc; links against libnvm + CUDA runtime.

#include "csrc/data_paths/striped_local_nvme/fused_submit_kernel.cuh"

#include <tutti/cuda_like.h>

namespace tutti::data_paths::striped_local_nvme {

cudaError_t launch_fused_submit(
    const StripedDeviceSubmitEntry* d_entries,
    EntryCompletionStatus*          d_status,
    const DeviceTargetHandle* const* d_dev_table,
    std::uint32_t                   count,
    std::uint32_t                   num_devs,
    std::uint32_t                   cq_poll_budget,
    std::uint32_t                   inject_flag,
    unsigned long long*             d_timing,
    void*                           stream,
    std::uint32_t                   pool_workers,
    unsigned int*                   d_task_counter)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    constexpr std::uint32_t kBlock = local_nvme::kSubmitBlockThreads;

    if (pool_workers == 0 || d_task_counter == nullptr) {
        // Legacy model: one thread per entry.
        const std::uint32_t blocks = count == 0 ? 1 : 1 + (count - 1) / kBlock;
        fused_submit_kernel<<<blocks, kBlock, 0, s>>>(
            d_entries, d_status, d_dev_table, count, num_devs,
            cq_poll_budget, inject_flag, d_timing);
        return cudaGetLastError();
    }

    // Worker-pool model: a fixed set of worker threads per launch,
    // independent of how many entries the batch carries -- never more than
    // the batch has entries -- in one-warp blocks. 256 workers drive 4
    // drives at the same ~12 GiB/s as 2048. The task cursor is zero on entry
    // (arena init, then reset by the previous launch's last worker), so no
    // memset precedes the launch.
    const std::uint32_t workers = count < pool_workers ? (count ? count : 1u) : pool_workers;
    const std::uint32_t tpb = workers < kBlock ? workers : kBlock;
    const std::uint32_t blocks = 1 + (workers - 1) / tpb;
    fused_submit_kernel_pool<<<blocks, tpb, 0, s>>>(
        d_entries, d_status, d_dev_table, count, num_devs,
        cq_poll_budget, inject_flag, d_timing, d_task_counter, workers);
    return cudaGetLastError();
}

} // namespace tutti::data_paths::striped_local_nvme

// Submit-only and wait kernels of StripedAsyncDataPath (same translation unit
// so the kernels defined in fused_submit_kernel.cuh exist once).
#include "csrc/data_paths/striped_local_nvme/async_submit_kernel_impl.cuh"
