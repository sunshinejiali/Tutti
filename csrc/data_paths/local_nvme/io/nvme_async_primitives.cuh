#pragma once
// csrc/data_paths/local_nvme/io/nvme_async_primitives.cuh
//
// Submit / drain split for GPU-initiated NVMe IO
// (design: doc/design/async-nvme-completion.md).
//
// The synchronous primitives (nvme_submit_primitives.cuh) have each IO thread
// wait for its own completion record, so the IO kernel stays resident for the
// whole IO. Here a submitting thread records who owns the command id, writes
// the command and moves on; whoever drains a completion queue looks the
// owner up and finishes that entry. A batch (arena slot) is complete when its
// pending_entries count reaches zero.
//
// A queue must be driven by one mode only: a synchronous thread waits for a
// completion record that a drainer would take away. Every queue of a
// DataPath that uses these primitives is used only by them.
//
// Host-visible section: SlotState / CompletionTables (plain structs).
// Device section (__CUDACC__): finish_entry, drain_completions,
// sq_enqueue_or_drain, submit_no_wait.

#include <cstdint>

#include "csrc/data_paths/local_nvme/io/nvme_submit_primitives.cuh"

struct QueuePair;

namespace tutti::data_paths::local_nvme::async_io {

// Command ids per queue: get_cid() hands out ids in [0, 65536).
constexpr std::uint32_t kCidSpace = 65536;

// Per arena slot, device memory, written by the host (H2D on the submit
// stream) before the submit kernel of the batch that uses the slot.
struct SlotState {
    EntryCompletionStatus* status = nullptr;  // the slot's status array
    std::uint32_t pending_entries = 0;        // entries of this batch not yet finished
    std::uint32_t batch_id = 0;               // this batch's id (unique, never 0)
};

// Everything the device side needs, passed to kernels by value.
struct CompletionTables {
    SlotState* slots = nullptr;                     // [num_slots], device
    std::uint32_t* finished_batch_id = nullptr;     // [num_slots], device
    std::uint32_t* finished_batch_id_host = nullptr;// [num_slots], host-mapped mirror
    std::uint32_t* timed_out_host = nullptr;        // [num_slots], host-mapped
    unsigned long long* cid_owner = nullptr;        // [num_queues * kCidSpace], device
    QueuePair* const* queues = nullptr;             // [num_queues], device
    QueuePair* const* device_queue_base = nullptr;  // [num_devices]: each device's d_qps
    const std::uint32_t* device_queue_offset = nullptr;  // [num_devices]
    std::uint32_t num_queues = 0;
    std::uint32_t num_slots = 0;
    std::uint32_t num_devices = 0;
};

// cid_owner entry: {slot:16 | low 16 bits of batch_id:16 | entry:32}.
inline constexpr unsigned long long pack_owner(std::uint32_t slot,
                                               std::uint32_t batch_id,
                                               std::uint32_t entry) {
    return (static_cast<unsigned long long>(slot & 0xffffu) << 48) |
           (static_cast<unsigned long long>(batch_id & 0xffffu) << 32) |
           static_cast<unsigned long long>(entry);
}

} // namespace tutti::data_paths::local_nvme::async_io

#if defined(__CUDACC__)

namespace tutti::data_paths::local_nvme::async_io {

TUTTI_DEVICE TUTTI_FORCEINLINE void backoff(unsigned int& ns) {
#if (__CUDA_ARCH__ >= 700)
    __nanosleep(ns);
    if (ns < 256) ns *= 2;
#endif
}

TUTTI_DEVICE TUTTI_FORCEINLINE void write_doorbell(volatile std::uint32_t* db,
                                                   std::uint32_t value) {
#if defined(TUTTI_USE_MACA)
    __threadfence_system();
    *db = value;
#else
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;" ::"l"(db), "r"(value) : "memory");
#endif
}

// One entry of `slot` is finished (its status is written). The thread that
// finishes the last entry publishes the batch id, first to device memory
// (for waiting kernels) then to the host-mapped mirror (for the host).
TUTTI_DEVICE TUTTI_FORCEINLINE void finish_entry(const CompletionTables& t,
                                                 std::uint32_t slot) {
    __threadfence();
    const unsigned int before = atomicSub(&t.slots[slot].pending_entries, 1u);
    if (before == 1u) {
        __threadfence();
        const std::uint32_t batch_id =
            *reinterpret_cast<volatile std::uint32_t*>(&t.slots[slot].batch_id);
        atomicExch(&t.finished_batch_id[slot], batch_id);
        __threadfence_system();
        *reinterpret_cast<volatile std::uint32_t*>(&t.finished_batch_id_host[slot]) = batch_id;
    }
}

// Process every completion record that has already arrived on queue
// `queue_index` (at most `max_records`). Never waits: if another thread is
// draining this queue it returns 0 at once. Returns the records processed.
TUTTI_DEVICE inline std::uint32_t drain_completions(const CompletionTables& t,
                                                    std::uint32_t queue_index,
                                                    std::uint32_t max_records) {
    QueuePair* qp = t.queues[queue_index];
    nvm_queue_t* cq = &qp->cq;
    nvm_queue_t* sq = &qp->sq;
    if (cq->head_lock.load(cuda::memory_order_relaxed) == LOCKED) return 0;
    if (cq->head_lock.fetch_or(LOCKED, cuda::memory_order_acquire) == LOCKED) return 0;

    const std::uint32_t head = cq->head.load(cuda::memory_order_relaxed);
    const nvm_cpl_t* records = (const nvm_cpl_t*)cq->vaddr;  // dwords are volatile
    std::uint32_t n = 0;
    std::uint32_t last_sq_head = 0;
    while (n < max_records) {
        const std::uint32_t position = head + n;
        const std::uint32_t loc = position & cq->qs_minus_1;
        const std::uint32_t expected_phase = (~(position >> cq->qs_log2)) & 1u;
        const std::uint32_t dw3 = records[loc].dword[3];
        if (((dw3 >> 16) & 1u) != expected_phase) break;  // not arrived yet
        const std::uint32_t dw2 = records[loc].dword[2];
        const std::uint16_t cid = static_cast<std::uint16_t>(dw3 & 0xffffu);

        // Read the owner before the cid is returned (it may be reused at once).
        const unsigned long long owner = *reinterpret_cast<volatile unsigned long long*>(
            &t.cid_owner[static_cast<std::size_t>(queue_index) * kCidSpace + cid]);
        const std::uint32_t slot = static_cast<std::uint32_t>(owner >> 48);
        const std::uint32_t batch_low = static_cast<std::uint32_t>(owner >> 32) & 0xffffu;
        const std::uint32_t entry = static_cast<std::uint32_t>(owner);
        if (slot < t.num_slots &&
            batch_low == (*reinterpret_cast<volatile std::uint32_t*>(
                              &t.slots[slot].batch_id) & 0xffffu)) {
            EntryCompletionStatus* status =
                *reinterpret_cast<EntryCompletionStatus* volatile*>(&t.slots[slot].status);
            status[entry].nvme_status_dword3 = dw3;
            status[entry].result = ((dw3 >> 17) != 0) ? 3u : 0u;
            finish_entry(t, slot);
        }
        // else: a completion for a batch that no longer owns the slot (only
        // possible after a timeout); its cid is still returned below.
        put_cid(sq, cid);
        last_sq_head = dw2 & 0xffffu;
        ++n;
    }

    if (n != 0) {
        // Free the submission-queue positions the controller has consumed
        // (same bookkeeping as move_head_cq: one ticket round per position).
        const std::uint32_t cur_sq_head = sq->head.load(cuda::memory_order_relaxed);
        std::uint32_t loc = cur_sq_head & sq->qs_minus_1;
        std::uint32_t moved = 0;
        for (; loc != last_sq_head; ++moved, loc = (loc + 1) & sq->qs_minus_1) {
            sq->tickets[loc].val.fetch_add(1, cuda::memory_order_relaxed);
        }
        if (moved != 0) sq->head.fetch_add(moved, cuda::memory_order_acq_rel);

        const std::uint32_t new_head = head + n;
        write_doorbell(cq->db, new_head & cq->qs_minus_1);
        cq->head.store(new_head, cuda::memory_order_release);
    }
    cq->head_lock.store(UNLOCKED, cuda::memory_order_release);
    return n;
}

// sq_enqueue (nvm_parallel_queue.h) with one difference: every wait for a
// free position drains this queue first instead of relying on other threads
// that wait for their own completions (there are none in this mode).
TUTTI_DEVICE inline void sq_enqueue_or_drain(const CompletionTables& t,
                                             std::uint32_t queue_index,
                                             nvm_queue_t* sq,
                                             nvm_cmd_t* cmd) {
    const std::uint32_t ticket = sq->in_ticket.fetch_add(1, cuda::memory_order_relaxed);
    const std::uint32_t pos = ticket & sq->qs_minus_1;
    const std::uint64_t id = get_id(ticket, sq->qs_log2);

    unsigned int ns = 8;
    while (sq->tickets[pos].val.load(cuda::memory_order_acquire) != id) {
        drain_completions(t, queue_index, 64);
        backoff(ns);
    }

    copy_type* queue_loc = (copy_type*)(((nvm_cmd_t*)sq->vaddr) + pos);
    const copy_type* src = reinterpret_cast<const copy_type*>(cmd->dword);
#pragma unroll
    for (std::uint32_t i = 0; i < 64 / sizeof(copy_type); ++i) queue_loc[i] = src[i];
    sq->tail_mark[pos].val.store(LOCKED, cuda::memory_order_release);

    ns = 8;
    bool cont = sq->tail_mark[pos].val.load(cuda::memory_order_relaxed) == LOCKED;
    while (cont) {
        bool busy = sq->tail_lock.load(cuda::memory_order_relaxed) == LOCKED;
        if (!busy) {
            busy = sq->tail_lock.fetch_or(LOCKED, cuda::memory_order_acquire) == LOCKED;
            if (!busy) {
                const std::uint32_t cur_tail = sq->tail.load(cuda::memory_order_relaxed);
                const std::uint32_t moved = move_tail(sq, cur_tail);
                if (moved != 0) {
                    const std::uint32_t new_tail = cur_tail + moved;
                    write_doorbell(sq->db, new_tail & sq->qs_minus_1);
                    sq->tail.store(new_tail, cuda::memory_order_release);
                }
                sq->tail_lock.store(UNLOCKED, cuda::memory_order_release);
            }
        }
        cont = sq->tail_mark[pos].val.load(cuda::memory_order_relaxed) == LOCKED;
        if (cont) {
            // The tail cannot pass the controller's head: free positions.
            drain_completions(t, queue_index, 64);
            backoff(ns);
        }
    }
    sq->tickets[pos].val.fetch_add(1, cuda::memory_order_acq_rel);
}

// Global index (into t.queues / cid_owner) of queue `local` of the device
// whose queue array is `device_qps`; UINT32_MAX if unknown.
TUTTI_DEVICE TUTTI_FORCEINLINE std::uint32_t global_queue_index(
    const CompletionTables& t, const QueuePair* device_qps, std::uint32_t local) {
    for (std::uint32_t d = 0; d < t.num_devices; ++d) {
        if (t.device_queue_base[d] == device_qps) return t.device_queue_offset[d] + local;
    }
    return UINT32_MAX;
}

// One entry: resolve the LBA, take a cid, record its owner, enqueue the
// command and return without waiting. Every path finishes the entry exactly
// once, either here (failure before the command exists) or in a drainer.
TUTTI_DEVICE inline void submit_no_wait(const CompletionTables& t,
                                        const DeviceTargetHandle* h,
                                        std::uint64_t prp1,
                                        std::uint64_t prp2,
                                        std::uint64_t logical_off,
                                        std::uint64_t nbytes,
                                        std::uint8_t opcode,
                                        std::uint32_t slot,
                                        std::uint32_t batch_id,
                                        std::uint32_t entry,
                                        EntryCompletionStatus* status,
                                        std::uint32_t inject_flag) {
    std::uint64_t starting_lba = 0;
    std::uint64_t n_blocks = 0;
    std::uint32_t queue_index = UINT32_MAX;
    std::uint32_t local = 0;
    if (resolve_lba(h, logical_off, nbytes, &starting_lba, &n_blocks, inject_flag)) {
        local = QueueAcquireHelper::acquire_queue(h->num_d_qps);
        queue_index = global_queue_index(t, h->d_qps, local);
    }
    if (queue_index == UINT32_MAX) {
        status->result = 1;
        status->nvme_status_dword3 = 0;
        finish_entry(t, slot);
        return;
    }
    QueuePair* qp = &h->d_qps[local];
    nvm_cmd_t cmd;
    nvm_cmd_clear(&cmd);
    const std::uint16_t cid = get_cid(&qp->sq);
    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    // The owner must be visible before the controller can complete the
    // command (it can only see the command after the doorbell below).
    *reinterpret_cast<volatile unsigned long long*>(
        &t.cid_owner[static_cast<std::size_t>(queue_index) * kCidSpace + cid]) =
        pack_owner(slot, batch_id, entry);
    __threadfence();
    sq_enqueue_or_drain(t, queue_index, &qp->sq, &cmd);
}

} // namespace tutti::data_paths::local_nvme::async_io

#endif // __CUDACC__
