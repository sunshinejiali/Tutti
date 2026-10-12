#include "csrc/data_paths/striped_local_nvme/striped_async_data_path.h"

#include <tutti/accelerator_device_guard.h>
#include <tutti/cuda_like.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <algorithm>

#include "csrc/data_paths/striped_local_nvme/async_submit_kernel.cuh"

using namespace tutti;
using AsyncPath = data_paths::striped_local_nvme::StripedAsyncDataPath;
using Tables = data_paths::local_nvme::async_io::CompletionTables;

void publish_test_completion(Tables tables, std::uint32_t slot, unsigned long long delay_ns,
                             cudaStream_t stream);
void test_synthetic_completion_queue();
void gate_test_stream(const std::uint32_t* release, cudaStream_t stream);

struct StreamGate {
    std::uint32_t* host = nullptr;
    std::uint32_t* device = nullptr;
    StreamGate() {
        if (cudaHostAlloc(&host, sizeof(*host), cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess) {
            throw std::runtime_error("stream gate allocation failed");
        }
        *host = 0;
    }
    void release() { *reinterpret_cast<volatile std::uint32_t*>(host) = 1; }
    ~StreamGate() { cudaFreeHost(host); }
};

struct StreamGates {
    StreamGate completion;
    StreamGate second_wait;
    ~StreamGates() {
        completion.release();
        second_wait.release();
        cudaDeviceSynchronize();
    }
};

#define REQUIRE(test) do { if (!(test)) throw std::runtime_error(#test); } while (0)
#define CUDA_CHECK(call) REQUIRE((call) == cudaSuccess)

namespace tutti::testing {

struct AsyncCompletionTestAccess {
    static void initialize(AsyncPath& path) {
        REQUIRE(path.arena_.init({24, 1, 4096, 0, 1}));
        const auto n = path.arena_.capacity();
        CUDA_CHECK(cudaMalloc(&path.d_slots_, n * sizeof(AsyncPath::SlotState)));
        CUDA_CHECK(cudaMalloc(&path.d_finished_, n * sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMemset(path.d_finished_, 0, n * sizeof(std::uint32_t)));
        CUDA_CHECK(cudaHostAlloc(&path.h_finished_, n * sizeof(std::uint32_t), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&path.h_timed_out_, n * sizeof(std::uint32_t), cudaHostAllocMapped));
        std::fill_n(path.h_finished_, n, 0);
        std::fill_n(path.h_timed_out_, n, 0);
        std::uint32_t *finished, *timeout;
        CUDA_CHECK(cudaHostGetDevicePointer(&finished, path.h_finished_, 0));
        CUDA_CHECK(cudaHostGetDevicePointer(&timeout, path.h_timed_out_, 0));
        path.tables_.slots = path.d_slots_;
        path.tables_.finished_batch_id = path.d_finished_;
        path.tables_.finished_batch_id_host = finished;
        path.tables_.timed_out_host = timeout;
        path.tables_.num_slots = n;
        cudaStream_t fallback;
        CUDA_CHECK(cudaStreamCreateWithFlags(&fallback, cudaStreamNonBlocking));
        path.fallback_stream_ = fallback;
        path.host_slot_states_.resize(n);
        path.slot_batch_id_.assign(n, 0);
        path.wait_timeout_ns_ = 1000000;
        path.initialized_ = true;
    }

    static DataPathOp batch(AsyncPath& path, cudaStream_t submit) {
        data_paths::striped_local_nvme::StripedArena::Lease lease;
        REQUIRE(path.arena_.acquire(lease));
        const std::uint32_t slot = lease.slot_index;
        const std::uint32_t id = path.next_batch_id_++;
        path.slot_batch_id_[slot] = id;
        path.h_finished_[slot] = 0;
        path.h_timed_out_[slot] = 0;
        path.wait_state_[slot] = AsyncPath::OpWaitState{};
        auto& state = path.host_slot_states_[slot];
        state.status = lease.d_status;
        state.pending_entries = 1;
        state.batch_id = id;
        CUDA_CHECK(cudaMemcpyAsync(path.d_slots_ + slot, &state, sizeof(state),
                                    cudaMemcpyHostToDevice, submit));
        CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(lease.event), submit));
        AsyncPath::OpEntry op;
        op.arena_slot = slot;
        op.event = lease.event;
        op.stream = submit;
        op.d_status = lease.d_status;
        op.entry_count = 1;
        op.entry_lengths = {4096};
        op.op_token = path.next_op_token_++;
        op.op_generation = 1;
        op.total_bytes = 4096;
        const auto token = op.op_token;
        path.ops_.emplace(token, std::move(op));
        return detail::SpiIdentityMint::mint<detail::DataPathOpTag>(token, 1);
    }

    static void publish(AsyncPath& path, DataPathOp op, cudaStream_t stream,
                        unsigned long long delay_ns) {
        publish_test_completion(path.tables_, path.find_op_(op)->arena_slot, delay_ns, stream);
    }

    static bool finished(AsyncPath& path, DataPathOp op) {
        const auto slot = path.find_op_(op)->arena_slot;
        return *reinterpret_cast<volatile std::uint32_t*>(path.h_finished_ + slot)
                   == path.slot_batch_id_[slot];
    }

    static bool timed_out(AsyncPath& path, DataPathOp op) {
        const auto slot = path.find_op_(op)->arena_slot;
        return *reinterpret_cast<volatile std::uint32_t*>(path.h_timed_out_ + slot) != 0;
    }

    static std::uint32_t available(AsyncPath& path) { return path.arena_.available(); }
    static bool resources_alive(AsyncPath& path) {
        return path.initialized_ && path.arena_.initialized() && path.d_slots_ && path.h_finished_;
    }
    static void lose_wait_fence(AsyncPath& path, DataPathOp op) {
        auto fence = path.wait_state_[path.find_op_(op)->arena_slot].waits.front();
        CUDA_CHECK(cudaEventDestroy(static_cast<cudaEvent_t>(fence->event)));
        fence->event = nullptr;
    }
    static void lose_submit_event(AsyncPath& path, DataPathOp op) {
        path.handle_submit_event_failure_(*path.find_op_(op));
    }
};

} // namespace tutti::testing

using Access = tutti::testing::AsyncCompletionTestAccess;

static void wait_terminal(AsyncPath& path, DataPathOp op) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (path.query(op).value().state == IoState::IN_FLIGHT) {
        REQUIRE(path.progress({32, 1000000}).ok());
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

static void test_timeout_and_wait_retirement(cudaStream_t submit, cudaStream_t producer,
                                              cudaStream_t consumer_a, cudaStream_t consumer_b) {
    AsyncPath path({}, 0);
    Access::initialize(path);
    StreamGates gates;
    publish_test_completion({}, 0, 0, producer);
    CUDA_CHECK(cudaStreamSynchronize(producer));
    auto& completion_gate = gates.completion;
    auto& second_wait_gate = gates.second_wait;
    const auto op = Access::batch(path, submit);
    CUDA_CHECK(cudaStreamSynchronize(submit));
    // Load all test kernels before blocking any stream (CUDA lazy loading).
    completion_gate.release();
    gate_test_stream(completion_gate.device, producer);
    CUDA_CHECK(cudaStreamSynchronize(producer));
    *completion_gate.host = 0;
    data_paths::striped_local_nvme::WaitList empty;
    CUDA_CHECK(data_paths::striped_local_nvme::launch_wait_batches({}, empty, 1, consumer_a));
    CUDA_CHECK(cudaStreamSynchronize(consumer_a));
    gate_test_stream(completion_gate.device, producer);
    Access::publish(path, op, producer, 0);
    REQUIRE(path.enqueue_wait(&op, 1, consumer_a).ok());
    gate_test_stream(second_wait_gate.device, consumer_b);
    REQUIRE(path.enqueue_wait(&op, 1, consumer_b).ok());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!Access::timed_out(path, op) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    REQUIRE(Access::timed_out(path, op));
    REQUIRE(path.progress({32, 1000000}).ok());
    REQUIRE(path.query(op).value().state == IoState::IN_FLIGHT);
    REQUIRE(path.release(op).code() == StatusCode::BUSY);
    REQUIRE(Access::available(path) == 23);
    completion_gate.release();
    CUDA_CHECK(cudaStreamSynchronize(producer));
    REQUIRE(Access::finished(path, op));
    const auto idle_budget = path.progress({32, 0});
    REQUIRE(idle_budget.ok() && idle_budget.value().work_units_consumed == 0);
    REQUIRE(path.query(op).value().state == IoState::IN_FLIGHT);
    REQUIRE(path.progress({0, 1000000}).value().work_units_consumed == 0);
    REQUIRE(path.progress({32, 1000000}).ok());
    REQUIRE(path.query(op).value().state == IoState::IN_FLIGHT);
    REQUIRE(path.shutdown(1000000).code() == StatusCode::TIMEOUT);
    REQUIRE(Access::resources_alive(path));
    REQUIRE(path.submit(nullptr, 0, {}).status.code() != StatusCode::OK);
    REQUIRE(!path.open(ResolvedTarget{}).ok());
    second_wait_gate.release();
    wait_terminal(path, op);
    REQUIRE(path.query(op).value().state == IoState::FAILED);
    REQUIRE(path.release(op).ok());
    REQUIRE(Access::available(path) == 24);
    REQUIRE(path.shutdown(1000000000).ok());
    REQUIRE(!Access::resources_alive(path));
}

static void test_submit_and_wait_fence_recovery(cudaStream_t submit, cudaStream_t producer,
                                                cudaStream_t consumer) {
    AsyncPath path({}, 0);
    Access::initialize(path);
    auto op = Access::batch(path, submit);
    CUDA_CHECK(cudaStreamSynchronize(submit));
    Access::lose_submit_event(path, op);
    REQUIRE(path.query(op).value().state == IoState::IN_FLIGHT);
    Access::publish(path, op, producer, 10000000);
    REQUIRE(path.enqueue_wait(&op, 1, consumer).ok());
    Access::lose_wait_fence(path, op);
    wait_terminal(path, op);
    REQUIRE(path.release(op).ok());
    REQUIRE(path.shutdown(1000000000).ok());
}

static void test_wait_list_groups(cudaStream_t submit, cudaStream_t producer,
                                  cudaStream_t consumer) {
    AsyncPath path({}, 0);
    Access::initialize(path);
    std::vector<DataPathOp> ops;
    for (int i = 0; i < 20; ++i) ops.push_back(Access::batch(path, submit));
    CUDA_CHECK(cudaStreamSynchronize(submit));
    for (auto op : ops) Access::publish(path, op, producer, 100000);
    REQUIRE(path.enqueue_wait(ops.data(), ops.size(), consumer).ok());
    for (auto op : ops) {
        wait_terminal(path, op);
        REQUIRE(path.release(op).ok());
    }
    REQUIRE(Access::available(path) == 24);
    REQUIRE(path.shutdown(1000000000).ok());
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t submit, producer, a, b;
    CUDA_CHECK(cudaStreamCreateWithFlags(&submit, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    test_synthetic_completion_queue();
    test_timeout_and_wait_retirement(submit, producer, a, b);
    test_submit_and_wait_fence_recovery(submit, producer, a);
    test_wait_list_groups(submit, producer, a);
    CUDA_CHECK(cudaStreamDestroy(submit));
    CUDA_CHECK(cudaStreamDestroy(producer));
    CUDA_CHECK(cudaStreamDestroy(a));
    CUDA_CHECK(cudaStreamDestroy(b));
    std::puts("async lifecycle GPU-only contracts passed (no NVMe IO)");
}
