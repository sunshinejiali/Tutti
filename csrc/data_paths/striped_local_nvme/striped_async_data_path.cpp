// csrc/data_paths/striped_local_nvme/striped_async_data_path.cpp
//
// See striped_async_data_path.h and doc/design/async-nvme-completion.md.

#include "csrc/data_paths/striped_local_nvme/striped_async_data_path.h"

#include "csrc/data_paths/local_nvme/io/nvme_queue_group.h"

#include "csrc/data_paths/striped_local_nvme/async_submit_kernel.cuh"

#include <tutti/accelerator_device_guard.h>
#include <tutti/cuda_like.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace tutti::data_paths::striped_local_nvme {

namespace async_io = tutti::data_paths::local_nvme::async_io;

namespace {

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

Status cuda_status(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return Status::Ok();
    cudaGetLastError();
    return Status(StatusCode::DEVICE_ERROR,
                  std::string(what) + ": " + cudaGetErrorString(e));
}

std::uint32_t volatile_load(const std::uint32_t* p) {
    return *reinterpret_cast<const volatile std::uint32_t*>(p);
}

} // namespace

StripedAsyncDataPath::~StripedAsyncDataPath() {
    // Explicit shutdown is bounded and retryable. Destruction cannot safely
    // release resources still used by a controller or another CUDA stream.
    while (initialized_) {
        const Status stopped = shutdown(1000000000ULL);
        if (stopped.ok()) break;
        std::fprintf(stderr, "[tutti] async destruction waiting for safe drain: %s\n",
                     stopped.message().c_str());
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

StripedAsyncDataPath::WaitFence::~WaitFence() {
    if (event) cudaEventDestroy(static_cast<cudaEvent_t>(event));
}

Status StripedAsyncDataPath::wait_fences_finished_(const OpWaitState& state) const {
    for (const auto& fence : state.waits) {
        if (!fence->event) {
            // Recording failed after launch: retry a fence on the same
            // stream, never treat a missing event as completed work.
            cudaEvent_t retry = nullptr;
            cudaError_t error = cudaEventCreateWithFlags(&retry, cudaEventDisableTiming);
            if (error != cudaSuccess) return cuda_status(error, "async retirement retry create");
            error = cudaEventRecord(retry, static_cast<cudaStream_t>(fence->stream));
            if (error != cudaSuccess) {
                cudaEventDestroy(retry);
                return cuda_status(error, "async retirement retry record");
            }
            fence->event = retry;
        }
        const cudaError_t result = cudaEventQuery(static_cast<cudaEvent_t>(fence->event));
        if (result == cudaErrorNotReady) return Status(StatusCode::BUSY, "async wait still running");
        if (result != cudaSuccess) return cuda_status(result, "async wait retirement");
    }
    return Status::Ok();
}

bool StripedAsyncDataPath::requested() {
    const char* v = std::getenv("TUTTI_ASYNC_COMPLETION");
    return v != nullptr && std::strcmp(v, "1") == 0;
}

// ---------------------------------------------------------------------------
// initialize / shutdown
// ---------------------------------------------------------------------------

Status StripedAsyncDataPath::initialize(const DataPathConfig& config,
                                        ResourceProvider& resources) {
    Status st = StripedDataPath::initialize(config, resources);
    if (!st.ok()) return st;
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    st = create_tables_();
    if (!st.ok()) {
        destroy_tables_();
        StripedDataPath::shutdown(0);
        return st;
    }
    return guard.restore();
}

Status StripedAsyncDataPath::create_tables_() {
    const std::uint32_t num_slots = arena_.capacity();
    if (num_slots > 65536) return Status(StatusCode::INVALID_ARGUMENT, "async slot encoding exceeded");
    stopping_ = false;
    device_failure_ = Status::Ok();
    host_slot_states_.assign(num_slots, SlotState{});
    std::vector<QueuePair*> queues;
    std::vector<QueuePair*> device_base;
    std::vector<std::uint32_t> device_offset;
    for (const auto& dev : devices_) {
        if (!dev.queue_group) {
            return Status(StatusCode::INVALID_ARGUMENT, "device without queue group");
        }
        QueuePair* base = dev.queue_group->d_qps();
        device_base.push_back(base);
        device_offset.push_back(static_cast<std::uint32_t>(queues.size()));
        for (std::uint32_t q = 0; q < dev.queue_group->n_qps(); ++q) {
            queues.push_back(reinterpret_cast<QueuePair*>(
                reinterpret_cast<char*>(base) + q * queue_pair_bytes()));
        }
    }
    if (queues.empty() || num_slots == 0) {
        return Status(StatusCode::INVALID_ARGUMENT, "no queues or arena slots");
    }

    const std::size_t owner_bytes =
        queues.size() * static_cast<std::size_t>(async_io::kCidSpace) * sizeof(unsigned long long);
    Status st = cuda_status(cudaMalloc(&d_cid_owner_, owner_bytes), "cid_owner");
    if (st.ok()) st = cuda_status(cudaMemset(d_cid_owner_, 0xff, owner_bytes), "cid_owner init");
    if (st.ok()) st = cuda_status(cudaMalloc(&d_slots_, num_slots * sizeof(SlotState)), "slots");
    if (st.ok()) st = cuda_status(cudaMemset(d_slots_, 0, num_slots * sizeof(SlotState)), "slots init");
    if (st.ok()) st = cuda_status(cudaMalloc(&d_finished_, num_slots * sizeof(std::uint32_t)), "finished");
    if (st.ok()) st = cuda_status(cudaMemset(d_finished_, 0, num_slots * sizeof(std::uint32_t)), "finished init");
    if (st.ok()) st = cuda_status(cudaHostAlloc(reinterpret_cast<void**>(&h_finished_),
                                                num_slots * sizeof(std::uint32_t),
                                                cudaHostAllocMapped), "finished mirror");
    if (st.ok()) st = cuda_status(cudaHostAlloc(reinterpret_cast<void**>(&h_timed_out_),
                                                num_slots * sizeof(std::uint32_t),
                                                cudaHostAllocMapped), "timed_out");
    if (st.ok()) {
        std::memset(h_finished_, 0, num_slots * sizeof(std::uint32_t));
        std::memset(h_timed_out_, 0, num_slots * sizeof(std::uint32_t));
    }
    if (st.ok()) st = cuda_status(cudaMalloc(&d_queues_, queues.size() * sizeof(QueuePair*)), "queues");
    if (st.ok()) st = cuda_status(cudaMemcpy(d_queues_, queues.data(), queues.size() * sizeof(QueuePair*),
                                             cudaMemcpyHostToDevice), "queues H2D");
    if (st.ok()) st = cuda_status(cudaMalloc(&d_device_queue_base_, device_base.size() * sizeof(QueuePair*)),
                                  "device queue base");
    if (st.ok()) st = cuda_status(cudaMemcpy(d_device_queue_base_, device_base.data(),
                                             device_base.size() * sizeof(QueuePair*),
                                             cudaMemcpyHostToDevice), "device queue base H2D");
    if (st.ok()) st = cuda_status(cudaMalloc(&d_device_queue_offset_, device_offset.size() * sizeof(std::uint32_t)),
                                  "device queue offset");
    if (st.ok()) st = cuda_status(cudaMemcpy(d_device_queue_offset_, device_offset.data(),
                                             device_offset.size() * sizeof(std::uint32_t),
                                             cudaMemcpyHostToDevice), "device queue offset H2D");
    cudaStream_t s = nullptr;
    if (st.ok()) st = cuda_status(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "fallback stream");
    if (!st.ok()) return st;
    fallback_stream_ = s;

    std::uint32_t* finished_host_dev = nullptr;
    std::uint32_t* timed_out_dev = nullptr;
    st = cuda_status(cudaHostGetDevicePointer(reinterpret_cast<void**>(&finished_host_dev), h_finished_, 0),
                     "finished mirror device pointer");
    if (st.ok()) st = cuda_status(cudaHostGetDevicePointer(reinterpret_cast<void**>(&timed_out_dev), h_timed_out_, 0),
                                  "timed_out device pointer");
    if (!st.ok()) return st;

    tables_.slots = d_slots_;
    tables_.finished_batch_id = d_finished_;
    tables_.finished_batch_id_host = finished_host_dev;
    tables_.timed_out_host = timed_out_dev;
    tables_.cid_owner = d_cid_owner_;
    tables_.queues = d_queues_;
    tables_.device_queue_base = d_device_queue_base_;
    tables_.device_queue_offset = d_device_queue_offset_;
    tables_.num_queues = static_cast<std::uint32_t>(queues.size());
    tables_.num_slots = num_slots;
    tables_.num_devices = static_cast<std::uint32_t>(device_base.size());
    slot_batch_id_.assign(num_slots, 0);
    std::fprintf(stderr,
                 "[tutti] StripedAsyncDataPath: %u queues on %u devices, %u slots "
                 "(submit-only IO kernel; completions drained by enqueued waits)\n",
                 tables_.num_queues, tables_.num_devices, num_slots);
    return Status::Ok();
}

void StripedAsyncDataPath::destroy_tables_() {
    if (fallback_stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(fallback_stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(fallback_stream_));
        fallback_stream_ = nullptr;
    }
    if (d_cid_owner_) cudaFree(d_cid_owner_);
    if (d_slots_) cudaFree(d_slots_);
    if (d_finished_) cudaFree(d_finished_);
    if (h_finished_) cudaFreeHost(h_finished_);
    if (h_timed_out_) cudaFreeHost(h_timed_out_);
    if (d_queues_) cudaFree(d_queues_);
    if (d_device_queue_base_) cudaFree(d_device_queue_base_);
    if (d_device_queue_offset_) cudaFree(d_device_queue_offset_);
    d_cid_owner_ = nullptr;
    d_slots_ = nullptr;
    d_finished_ = nullptr;
    h_finished_ = nullptr;
    h_timed_out_ = nullptr;
    d_queues_ = nullptr;
    d_device_queue_base_ = nullptr;
    d_device_queue_offset_ = nullptr;
    tables_ = CompletionTables{};
    wait_state_.clear();
}

Status StripedAsyncDataPath::shutdown(std::uint64_t timeout_ns) {
    if (!initialized_) return Status::Ok();
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    stopping_ = true;
    const std::uint64_t start = now_ns();
    std::vector<OpEntry*> pending;
    for (auto& [token, op] : ops_) {
        if (op.state == IoState::IN_FLIGHT) pending.push_back(&op);
    }
    if (!pending.empty()) {
        Status st = enqueue_wait_locked_(pending, fallback_stream_);
        if (!st.ok()) return st;
    }
    for (;;) {
        auto advanced = progress(ProgressBudget{max_in_flight_operations_, 1000000});
        if (!advanced.ok()) return advanced.status();
        bool inflight = false;
        for (const auto& [token, op] : ops_) {
            inflight |= op.state == IoState::IN_FLIGHT;
        }
        if (!inflight) break;
        if (timeout_ns == 0 || now_ns() - start >= timeout_ns) {
            return Status(StatusCode::TIMEOUT, "async shutdown: resources retained for drain retry");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    // progress publishes terminal only after every waiter fence retires.
    Status st = StripedDataPath::shutdown(0);
    if (!st.ok()) return st;
    destroy_tables_();
    return guard.restore();
}

SubmitOutcome StripedAsyncDataPath::submit(const DataPathRequest* requests,
                                           std::size_t count,
                                           const HostSubmitContext& ctx) {
    if (stopping_ || !device_failure_.ok()) {
        SubmitOutcome out;
        out.status = device_failure_.ok()
            ? Status(StatusCode::NOT_READY, "async DataPath is draining") : device_failure_;
        out.initial_states.resize(count);
        for (auto& state : out.initial_states) {
            state.state = IoRequestState::REJECTED;
            state.status = out.status;
        }
        return out;
    }
    return StripedDataPath::submit(requests, count, ctx);
}

Result<DataPathTarget> StripedAsyncDataPath::open(const ResolvedTarget& target) {
    if (stopping_ || !device_failure_.ok()) {
        return Result<DataPathTarget>::Failure(
            Status(StatusCode::NOT_READY, "async DataPath is draining or faulted"));
    }
    return StripedDataPath::open(target);
}

Result<DataPathMemory> StripedAsyncDataPath::register_memory(
    const DataPathMemoryView& view, const RegistrationDomainKey& domain) {
    if (stopping_ || !device_failure_.ok()) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::NOT_READY, "async DataPath is draining or faulted"));
    }
    return StripedDataPath::register_memory(view, domain);
}

// ---------------------------------------------------------------------------
// submit: the IO kernel only submits
// ---------------------------------------------------------------------------

int StripedAsyncDataPath::launch_io_kernel_(
    const StripedDeviceSubmitEntry* d_entries,
    EntryCompletionStatus* d_status,
    const tutti::data_paths::local_nvme::DeviceTargetHandle* const* d_dev_table,
    std::uint32_t entry_count,
    std::uint32_t dev_table_rows,
    unsigned long long* /*d_timing*/,
    void* stream,
    unsigned int* d_task_counter,
    std::uint32_t arena_slot) {
    if (arena_slot >= tables_.num_slots) return static_cast<int>(cudaErrorInvalidValue);
    if (next_batch_id_ == 0) return static_cast<int>(cudaErrorInvalidValue);
    const std::uint32_t batch_id = next_batch_id_++;
    slot_batch_id_[arena_slot] = batch_id;
    wait_state_[arena_slot] = OpWaitState{};
    // The slot's previous batch has finished and been released (the arena
    // hands out a slot only then), so resetting its host mirrors is safe.
    h_finished_[arena_slot] = 0;
    h_timed_out_[arena_slot] = 0;

    SlotState& state = host_slot_states_[arena_slot];
    state.status = d_status;
    state.pending_entries = entry_count;
    state.batch_id = batch_id;
    cudaError_t e = cudaMemcpyAsync(&d_slots_[arena_slot], &state, sizeof(state),
                                    cudaMemcpyHostToDevice,
                                    static_cast<cudaStream_t>(stream));
    if (e != cudaSuccess) return static_cast<int>(e);
    if (entry_count == 0) {
        // Nothing to submit: the batch is finished as of now.
        e = cudaMemcpyAsync(&d_finished_[arena_slot], &state.batch_id, sizeof(state.batch_id),
                            cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
        h_finished_[arena_slot] = batch_id;
        return static_cast<int>(e);
    }
    return static_cast<int>(launch_submit_no_wait(
        d_entries, d_status,
        reinterpret_cast<const DeviceTargetHandle* const*>(d_dev_table),
        entry_count, dev_table_rows, stream, pool_workers_, d_task_counter,
        tables_, arena_slot, batch_id));
}

void StripedAsyncDataPath::handle_submit_event_failure_(OpEntry& op) {
    auto& state = wait_state_[op.arena_slot];
    const cudaError_t result = cudaStreamSynchronize(static_cast<cudaStream_t>(op.stream));
    if (result == cudaSuccess) {
        state.submit_retired = true;
    } else {
        device_failure_ = cuda_status(result, "async submit stream retirement");
    }
    // The commands may still be on the device even after the stream exits.
    op.state = IoState::IN_FLIGHT;
    op.status = Status::Ok();
}

// ---------------------------------------------------------------------------
// enqueue_wait
// ---------------------------------------------------------------------------

Status StripedAsyncDataPath::enqueue_wait_locked_(const std::vector<OpEntry*>& ops,
                                                  void* stream) {
    WaitList list;
    auto flush = [&]() -> Status {
        if (list.count == 0) return Status::Ok();
        auto fence = std::make_shared<WaitFence>();
        fence->stream = stream;
        cudaEvent_t event = nullptr;
        Status st = cuda_status(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                                "async retirement event");
        if (!st.ok()) return st;
        fence->event = event;
        st = cuda_status(launch_wait_batches(tables_, list, wait_timeout_ns_, stream),
                         "wait_batches launch");
        if (!st.ok()) return st;
        st = cuda_status(cudaEventRecord(event, static_cast<cudaStream_t>(stream)),
                         "async retirement record");
        if (!st.ok()) {
            cudaEventDestroy(event);
            fence->event = nullptr;
            device_failure_ = st;
        }
        // A successful launch is retained even if recording its fence fails.
        for (std::uint32_t i = 0; i < list.count; ++i) {
            wait_state_[list.slot[i]].waits.push_back(fence);
        }
        list = WaitList{};
        return st;
    };
    for (OpEntry* op : ops) {
        if (op == nullptr || op->state != IoState::IN_FLIGHT) continue;
        if (op->arena_slot >= tables_.num_slots) continue;
        auto& state = wait_state_[op->arena_slot];
        if (std::any_of(state.waits.begin(), state.waits.end(),
                        [&](const auto& fence) { return fence->stream == stream; })) continue;
        bool already_listed = false;
        for (std::uint32_t i = 0; i < list.count; ++i) already_listed |= list.slot[i] == op->arena_slot;
        if (already_listed) continue;
        if (!state.submit_retired) {
            Status st = cuda_status(
                cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
                                    static_cast<cudaEvent_t>(op->event), 0),
                "wait for submit");
            if (!st.ok()) return st;
        }
        list.slot[list.count] = op->arena_slot;
        list.batch_id[list.count] = slot_batch_id_[op->arena_slot];
        ++list.count;
        if (list.count == kWaitBatchesPerKernel) {
            const Status st = flush();
            if (!st.ok()) return st;
        }
    }
    return flush();
}

Status StripedAsyncDataPath::enqueue_wait(const DataPathOp* ops, std::size_t count,
                                          void* stream) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    std::vector<OpEntry*> found;
    found.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        OpEntry* op = find_op_(ops[i]);
        if (op == nullptr) {
            return Status(StatusCode::NOT_FOUND, "enqueue_wait: op not found");
        }
        found.push_back(op);
    }
    Status st = enqueue_wait_locked_(found, stream);
    Status restored = guard.restore();
    return st.ok() ? restored : st;
}

// ---------------------------------------------------------------------------
// progress / release
// ---------------------------------------------------------------------------

Result<ProgressResult> StripedAsyncDataPath::progress(ProgressBudget budget) {
    if (budget.max_work_units == 0 || budget.timeout_ns == 0) {
        ProgressResult skipped{};
        skipped.more_work_likely = !ops_.empty();
        return Result<ProgressResult>::Success(skipped);
    }
    const std::uint64_t started = now_ns();
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<ProgressResult>::Failure(guard.status());

    ProgressResult result{};
    std::uint64_t work = 0;
    std::vector<OpEntry*> unwaited;
    for (auto& [tok, op] : ops_) {
        if (op.state != IoState::IN_FLIGHT) continue;
        if (work >= budget.max_work_units || now_ns() - started >= budget.timeout_ns) {
            result.more_work_likely = true;
            break;
        }
        ++work;
        // The submit kernel must have exited before the slot can be reused.
        const std::uint32_t slot = op.arena_slot;
        auto& ws = wait_state_[slot];
        const cudaError_t submitted = ws.submit_retired
            ? cudaSuccess : cudaEventQuery(static_cast<cudaEvent_t>(op.event));
        if (submitted == cudaErrorNotReady) {
            ++result.operations_advanced;
            continue;
        }
        if (submitted != cudaSuccess) {
            device_failure_ = cuda_status(submitted, "async submit retirement");
            return Result<ProgressResult>::Failure(device_failure_);
        }
        bool timed_out = volatile_load(&h_timed_out_[slot]) != 0;
        if (timed_out && device_failure_.ok()) {
            device_failure_ = Status(StatusCode::TIMEOUT, "async completion deadline exceeded; draining");
        }
        if (volatile_load(&h_finished_[slot]) == slot_batch_id_[slot]) {
            Status retired = wait_fences_finished_(ws);
            if (retired.code() == StatusCode::BUSY) {
                ++result.operations_advanced;
                continue;
            }
            if (!retired.ok()) {
                device_failure_ = retired;
                return Result<ProgressResult>::Failure(retired);
            }
            timed_out = volatile_load(&h_timed_out_[slot]) != 0;
            aggregate_completion_status_(op);
            if (timed_out) {
                op.state = IoState::FAILED;
                op.status = Status(StatusCode::TIMEOUT, "async deadline exceeded; DMA safely drained");
                op.completion_detail.timeout_seen = true;
                op.completion_detail.failure_kind = IoFailureKind::CQ_TIMEOUT;
                op.completion_detail.failure_scope = IoFailureScope::WHOLE_OPERATION;
            }
            ++result.operations_terminal;
            continue;
        }
        ++result.operations_advanced;
        if (ws.waits.empty()) {
            const std::uint64_t t = now_ns();
            if (ws.submit_seen_done_ns == 0) {
                ws.submit_seen_done_ns = t;
            } else if (t - ws.submit_seen_done_ns > kUnwaitedGraceNs) {
                unwaited.push_back(&op);
            }
        }
    }
    if (!unwaited.empty() && now_ns() - started < budget.timeout_ns) {
        if (!fallback_logged_) {
            fallback_logged_ = true;
            std::fprintf(stderr,
                         "[tutti] StripedAsyncDataPath: %zu op(s) had no enqueued wait "
                         "%.0f ms after submit; draining them on an internal stream "
                         "(logged once)\n",
                         unwaited.size(), kUnwaitedGraceNs / 1e6);
        }
        Status st = enqueue_wait_locked_(unwaited, fallback_stream_);
        if (!st.ok()) return Result<ProgressResult>::Failure(std::move(st));
    }
    result.work_units_consumed = work;
    if (!result.more_work_likely) {
        for (const auto& [tok, op] : ops_) {
            if (op.state == IoState::IN_FLIGHT) { result.more_work_likely = true; break; }
        }
    }
    Status restored = guard.restore();
    if (!restored.ok()) return Result<ProgressResult>::Failure(std::move(restored));
    return Result<ProgressResult>::Success(std::move(result));
}

Status StripedAsyncDataPath::release(DataPathOp op) {
    bool retired = false;
    return release_retired(op, retired);
}

Status StripedAsyncDataPath::release_retired(DataPathOp op, bool& retired) {
    retired = false;
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    const OpEntry* entry = find_op_(op);
    const std::uint32_t slot = entry ? entry->arena_slot : UINT32_MAX;
    if (entry && entry->state != IoState::IN_FLIGHT && slot != UINT32_MAX) {
        const Status fence_status = wait_fences_finished_(wait_state_[slot]);
        if (!fence_status.ok()) return fence_status;
    }
    const Status st = release_impl_(op);
    if (st.ok()) {
        retired = true;
        if (slot != UINT32_MAX) wait_state_.erase(slot);
    }
    const Status restored = guard.restore();
    return st.ok() ? restored : st;
}

} // namespace tutti::data_paths::striped_local_nvme
