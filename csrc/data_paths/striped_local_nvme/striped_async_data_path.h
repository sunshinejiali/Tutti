#pragma once
// csrc/data_paths/striped_local_nvme/striped_async_data_path.h
//
// StripedAsyncDataPath — StripedDataPath whose IO kernel only submits.
// Design: doc/design/async-nvme-completion.md.
//
// Same targets, memory registration, arena, PRP handling and request
// fan-out as StripedDataPath (inherited unchanged). Differences:
//   * the submit kernel issues each command and exits: no IO kernel stays
//     resident while the IO is in flight;
//   * completion records are processed by whoever drains a queue: a
//     wait_batches kernel enqueued with enqueue_wait() on the consuming
//     stream, or a submit thread that finds its queue full;
//   * an op is terminal once every entry has finished (pending count of its
//     arena slot reached zero), observed by the host through a host-mapped
//     mirror of finished_batch_id.
//
// Every queue of this DataPath runs in this mode only.

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tutti/spi/async_completion.h>

#include "csrc/data_paths/local_nvme/io/nvme_async_primitives.cuh"
#include "csrc/data_paths/striped_local_nvme/striped_data_path.h"

namespace tutti::testing { struct AsyncCompletionTestAccess; }

namespace tutti::data_paths::striped_local_nvme {

class StripedAsyncDataPath final : public StripedDataPath,
                                   public tutti::AsyncCompletion {
public:
    using StripedDataPath::StripedDataPath;
    ~StripedAsyncDataPath() override;

    // Experimental switch, read where striped DataPaths are built (YAML
    // factory and preset): TUTTI_ASYNC_COMPLETION=1 builds this class.
    static bool requested();

    Status initialize(const DataPathConfig& config,
                      ResourceProvider& resources) override;
    Status shutdown(std::uint64_t timeout_ns) override;
    Result<ProgressResult> progress(ProgressBudget budget) override;
    Status release(DataPathOp op) override;
    Status release_retired(DataPathOp op, bool& retired) override;
    SubmitOutcome submit(const DataPathRequest* requests, std::size_t count,
                         const HostSubmitContext& ctx) override;
    Result<DataPathTarget> open(const ResolvedTarget& target) override;
    Result<DataPathMemory> register_memory(const DataPathMemoryView& view,
                                           const RegistrationDomainKey& domain) override;

    Status enqueue_wait(const DataPathOp* ops, std::size_t count,
                        void* stream) override;

    // A wait kernel reports a timeout but drains until actual completion.
    static constexpr unsigned long long kWaitTimeoutNs = 60ull * 1000 * 1000 * 1000;
    // An op nobody enqueued a wait for gets one on an internal stream once its
    // submit has finished this long ago (host-only callers, warm-up reads).
    static constexpr std::uint64_t kUnwaitedGraceNs = 100ull * 1000 * 1000;

protected:
    void handle_submit_event_failure_(OpEntry& op) override;

    int launch_io_kernel_(const StripedDeviceSubmitEntry* d_entries,
                          EntryCompletionStatus* d_status,
                          const tutti::data_paths::local_nvme::DeviceTargetHandle* const* d_dev_table,
                          std::uint32_t entry_count,
                          std::uint32_t dev_table_rows,
                          unsigned long long* d_timing,
                          void* stream,
                          unsigned int* d_task_counter,
                          std::uint32_t arena_slot) override;

private:
    friend struct tutti::testing::AsyncCompletionTestAccess;
    using SlotState = tutti::data_paths::local_nvme::async_io::SlotState;
    using CompletionTables = tutti::data_paths::local_nvme::async_io::CompletionTables;

    struct WaitFence {
        void* stream = nullptr;
        void* event = nullptr;
        ~WaitFence();
    };

    struct OpWaitState {
        std::vector<std::shared_ptr<WaitFence>> waits;
        bool submit_retired = false;
        std::uint64_t submit_seen_done_ns = 0;
    };

    Status create_tables_();
    void destroy_tables_();
    Status wait_fences_finished_(const OpWaitState& state) const;
    // Enqueue a wait for `ops` on `stream` (gate held by the caller).
    Status enqueue_wait_locked_(const std::vector<OpEntry*>& ops, void* stream);

    CompletionTables tables_{};
    // Owned allocations behind tables_.
    SlotState* d_slots_ = nullptr;
    std::uint32_t* d_finished_ = nullptr;
    std::uint32_t* h_finished_ = nullptr;     // host-mapped
    std::uint32_t* h_timed_out_ = nullptr;    // host-mapped
    unsigned long long* d_cid_owner_ = nullptr;
    QueuePair** d_queues_ = nullptr;
    QueuePair** d_device_queue_base_ = nullptr;
    std::uint32_t* d_device_queue_offset_ = nullptr;
    void* fallback_stream_ = nullptr;         // cudaStream_t

    unsigned long long wait_timeout_ns_ = kWaitTimeoutNs;
    bool stopping_ = false;
    Status device_failure_;
    std::vector<SlotState> host_slot_states_;
    std::uint32_t next_batch_id_ = 1;
    std::vector<std::uint32_t> slot_batch_id_;  // host copy, per arena slot
    std::unordered_map<std::uint32_t, OpWaitState> wait_state_;  // per arena slot
    bool fallback_logged_ = false;
};

} // namespace tutti::data_paths::striped_local_nvme
