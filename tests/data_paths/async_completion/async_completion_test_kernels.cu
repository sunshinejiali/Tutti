#include "csrc/data_paths/local_nvme/io/nvme_async_primitives.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

using Tables = tutti::data_paths::local_nvme::async_io::CompletionTables;
using Slot = tutti::data_paths::local_nvme::async_io::SlotState;
using EntryStatus = tutti::data_paths::local_nvme::EntryCompletionStatus;
namespace aio = tutti::data_paths::local_nvme::async_io;

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void publish_completion(Tables tables, std::uint32_t slot,
                                    unsigned long long delay_ns) {
    if (tables.num_slots == 0) return;
    const auto start = TUTTI_GLOBALTIMER_NS();
    while (TUTTI_GLOBALTIMER_NS() - start < delay_ns) __nanosleep(256);
    tables.slots[slot].status[0] = {0, 0};
    aio::finish_entry(tables, slot);
}

__global__ void wait_for_gate(const std::uint32_t* release) {
    while (*reinterpret_cast<const volatile std::uint32_t*>(release) == 0) __nanosleep(256);
}

void gate_test_stream(const std::uint32_t* release, cudaStream_t stream) {
    wait_for_gate<<<1, 1, 0, stream>>>(release);
    check(cudaGetLastError());
}

void publish_test_completion(Tables tables, std::uint32_t slot, unsigned long long delay_ns,
                             cudaStream_t stream) {
    publish_completion<<<1, 1, 0, stream>>>(tables, slot, delay_ns);
    check(cudaGetLastError());
}

__global__ void prepare_queue(QueuePair* pair, padded_struct* tickets, padded_struct* ids,
                               nvm_cpl_t* records, std::uint32_t* doorbell,
                               Tables tables, EntryStatus* status) {
    pair->sq.qs = pair->cq.qs = 4;
    pair->sq.qs_minus_1 = pair->cq.qs_minus_1 = 3;
    pair->sq.qs_log2 = pair->cq.qs_log2 = 2;
    pair->sq.tickets = tickets;
    pair->sq.cid = ids;
    pair->cq.vaddr = records;
    pair->cq.db = doorbell;
    pair->sq.head.store(0);
    pair->cq.head.store(3);  // completions straddle a phase wrap
    pair->cq.head_lock.store(0);
    ids[7].val.store(1);
    ids[9].val.store(1);
    tables.slots[0] = {status, 1, 42};
    tables.slots[1] = {status + 1, 1, 43};
    tables.cid_owner[7] = aio::pack_owner(0, 42, 0);
    tables.cid_owner[9] = aio::pack_owner(1, 43, 0);
    records[3].dword[2] = 1;
    records[3].dword[3] = (1u << 16) | 9;  // slot 1 completes before slot 0
    records[0].dword[2] = 2;
    records[0].dword[3] = (1u << 17) | 7;  // phase 0, NVMe error
    records[1].dword[3] = (1u << 16);      // not yet completed at phase 0
}

__global__ void drain_queue(Tables tables, padded_struct* tickets, padded_struct* ids,
                             std::uint32_t* doorbell, std::uint32_t* result) {
    const auto first = aio::drain_completions(tables, 0, 8);
    const auto second = aio::drain_completions(tables, 0, 8);
    result[0] = first == 2 && second == 0 &&
        tables.slots[0].pending_entries == 0 && tables.slots[1].pending_entries == 0 &&
        tables.finished_batch_id[0] == 42 && tables.finished_batch_id[1] == 43 &&
        tables.slots[0].status[0].result == 3 && tables.slots[1].status[0].result == 0 &&
        tables.queues[0]->cq.head.load() == 5 && *doorbell == 1 &&
        tables.queues[0]->sq.head.load() == 2 &&
        tickets[0].val.load() == 1 && tickets[1].val.load() == 1 &&
        ids[7].val.load() == 0 && ids[9].val.load() == 0;
}

void test_synthetic_completion_queue() {
    QueuePair* pair;
    QueuePair** queues;
    padded_struct *tickets, *ids;
    nvm_cpl_t* records;
    Slot* slots;
    EntryStatus* status;
    unsigned long long* owners;
    std::uint32_t *finished, *mirror, *doorbell, *result;
    check(cudaMalloc(&pair, sizeof(QueuePair)));
    check(cudaMalloc(&queues, sizeof(QueuePair*)));
    check(cudaMalloc(&tickets, 4 * sizeof(padded_struct)));
    check(cudaMalloc(&ids, aio::kCidSpace * sizeof(padded_struct)));
    check(cudaMalloc(&records, 4 * sizeof(nvm_cpl_t)));
    check(cudaMalloc(&slots, 2 * sizeof(Slot)));
    check(cudaMalloc(&status, 2 * sizeof(EntryStatus)));
    check(cudaMalloc(&owners, aio::kCidSpace * sizeof(unsigned long long)));
    check(cudaMalloc(&finished, 2 * sizeof(std::uint32_t)));
    check(cudaMalloc(&mirror, 2 * sizeof(std::uint32_t)));
    check(cudaMalloc(&doorbell, sizeof(std::uint32_t)));
    check(cudaMalloc(&result, sizeof(std::uint32_t)));
    check(cudaMemset(pair, 0, sizeof(QueuePair)));
    check(cudaMemset(tickets, 0, 4 * sizeof(padded_struct)));
    check(cudaMemset(records, 0, 4 * sizeof(nvm_cpl_t)));
    check(cudaMemcpy(queues, &pair, sizeof(pair), cudaMemcpyHostToDevice));
    Tables tables;
    tables.slots = slots;
    tables.finished_batch_id = finished;
    tables.finished_batch_id_host = mirror;
    tables.cid_owner = owners;
    tables.queues = queues;
    tables.num_queues = 1;
    tables.num_slots = 2;
    prepare_queue<<<1, 1>>>(pair, tickets, ids, records, doorbell, tables, status);
    drain_queue<<<1, 1>>>(tables, tickets, ids, doorbell, result);
    std::uint32_t ok = 0;
    check(cudaMemcpy(&ok, result, sizeof(ok), cudaMemcpyDeviceToHost));
    if (!ok) throw std::runtime_error("synthetic CQ ownership/wrap/retirement failed");
    check(cudaFree(pair)); check(cudaFree(queues));
    check(cudaFree(tickets)); check(cudaFree(ids)); check(cudaFree(records));
    check(cudaFree(slots)); check(cudaFree(status)); check(cudaFree(owners));
    check(cudaFree(finished)); check(cudaFree(mirror)); check(cudaFree(doorbell));
    check(cudaFree(result));
}
