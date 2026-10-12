#pragma once
// csrc/data_paths/striped_local_nvme/async_submit_kernel_impl.cuh
//
// Definitions for async_submit_kernel.cuh. Compiled as part of
// fused_submit_kernel.cu (included at its end): fused_submit_kernel.cuh
// defines its kernels in the header, so a second translation unit that
// includes it would define them twice at device-link time.

#include "csrc/data_paths/striped_local_nvme/async_submit_kernel.cuh"

#include <tutti/cuda_like.h>

namespace tutti::data_paths::striped_local_nvme {

namespace async_io = tutti::data_paths::local_nvme::async_io;
using tutti::data_paths::local_nvme::AddressDescriptor;

namespace {

// Same worker pool as fused_submit_kernel_pool; each entry is issued and the
// worker moves on. Every entry is finished exactly once (here on failure
// before a command exists, otherwise by whoever drains its completion).
TUTTI_GLOBAL
void submit_no_wait_kernel_pool(const StripedDeviceSubmitEntry* entries,
                                EntryCompletionStatus* status,
                                const DeviceTargetHandle* const* dev_table,
                                std::uint32_t count,
                                std::uint32_t num_devs,
                                unsigned int* next_task,
                                std::uint32_t total_workers,
                                CompletionTables tables,
                                std::uint32_t slot,
                                std::uint32_t batch_id) {
    const std::uint32_t tid = TUTTI_THREAD_IDX_X + TUTTI_BLOCK_IDX_X * TUTTI_BLOCK_DIM_X;
    if (tid >= total_workers) return;
    for (;;) {
        const std::uint32_t idx = atomicAdd(next_task, 1u);
        if (idx >= count) {
            if (idx == count + total_workers - 1u) atomicExch(next_task, 0u);
            return;
        }
        const StripedDeviceSubmitEntry e = entries[idx];
        if (e.dev_idx >= num_devs) {
            status[idx].result = 1;
            status[idx].nvme_status_dword3 = 0;
            async_io::finish_entry(tables, slot);
            continue;
        }
        const AddressDescriptor* desc = e.prp_entry;
        async_io::submit_no_wait(
            tables, dev_table[e.dev_idx], desc->prp1, desc->prp2, e.shard_offset,
            desc->data_length,
            static_cast<std::uint8_t>(e.direction == 0 ? NVM_IO_READ : NVM_IO_WRITE),
            slot, batch_id, idx, &status[idx], 0);
    }
}

TUTTI_GLOBAL
void wait_batches_kernel(CompletionTables tables, WaitList list,
                         unsigned long long timeout_ns) {
    const std::uint32_t lane = TUTTI_THREAD_IDX_X;
    const unsigned long long start = TUTTI_GLOBALTIMER_NS();
    unsigned int ns = 32;
    bool timeout_reported = false;
    for (;;) {
        for (std::uint32_t q = lane; q < tables.num_queues; q += 32) {
            async_io::drain_completions(tables, q, 256);
        }
        bool done = true;
        if (lane < list.count) {
            done = *reinterpret_cast<volatile std::uint32_t*>(
                       &tables.finished_batch_id[list.slot[lane]]) == list.batch_id[lane];
        }
        if (__all_sync(0xffffffffu, done)) return;
        if (!timeout_reported && TUTTI_GLOBALTIMER_NS() - start > timeout_ns) {
            if (!done) {
                __threadfence_system();
                *reinterpret_cast<volatile std::uint32_t*>(
                    &tables.timed_out_host[list.slot[lane]]) = 1u;
            }
            timeout_reported = true;
        }
        __nanosleep(ns);
        if (ns < 1024) ns *= 2;
    }
}

} // namespace

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
                                  std::uint32_t batch_id) {
    constexpr std::uint32_t kBlock = local_nvme::kSubmitBlockThreads;
    const std::uint32_t cap = pool_workers != 0 ? pool_workers : 256u;
    const std::uint32_t workers = count < cap ? (count != 0 ? count : 1u) : cap;
    const std::uint32_t tpb = workers < kBlock ? workers : kBlock;
    const std::uint32_t blocks = 1 + (workers - 1) / tpb;
    submit_no_wait_kernel_pool<<<blocks, tpb, 0, static_cast<cudaStream_t>(stream)>>>(
        d_entries, d_status, d_dev_table, count, num_devs, d_task_counter, workers,
        tables, slot, batch_id);
    return cudaGetLastError();
}

std::size_t queue_pair_bytes() { return sizeof(QueuePair); }

cudaError_t launch_wait_batches(const CompletionTables& tables,
                                const WaitList& list,
                                unsigned long long timeout_ns,
                                void* stream) {
    wait_batches_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        tables, list, timeout_ns);
    return cudaGetLastError();
}

} // namespace tutti::data_paths::striped_local_nvme
