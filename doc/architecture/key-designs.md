# Key Designs — Performance & Low Overhead

> The five designs that keep Tutti's data path fast and its CPU overhead
> near zero, as implemented in the current codebase — each with pointers
> to the code that realizes it and the measured numbers behind it. For the
> layer-by-layer architecture see [system-architecture.md](system-architecture.md).

The GPU-centric data path: the CPU appears only **O(1) times per batch,
not per I/O**.

## 1. GPU io_uring — shared submission state, one hand-off per batch, async harvest

io_uring's essence is a shared-memory ring that moves the submission
boundary: kernel and userspace both see the SQ/CQ, so a single
`io_uring_enter` carries a whole batch in, and completions are reaped
later, independently of submission. Tutti applies the same shape to the
**CPU↔GPU boundary**:

- **Shared submission state instead of per-IO crossings.** The
  batch-entry array staged in GPU memory — together with the IO handle
  that tracks it — plays the SQ/CQ role between CPU and GPU: the host
  fills the entries, ONE `cudaMemcpyAsync` hands the whole batch across
  (the analog of one `io_uring_enter`), and ONE kernel launch is the only
  boundary crossing. Everything per-IO then happens inside the GPU:
  virtual→physical LBA resolution over the file's extent list, queue pick
  (SQ slots serialized by an atomic CAS on the tail), SQE write +
  doorbell ring through the GPU-mapped BAR, and completion busy-poll on
  the CQ phase bit. No syscalls, no interrupts, no CPU involvement on the
  data path.
- **Asynchronous by stream semantics.** A submit launches on the caller's
  CUDA stream and returns an IO handle immediately; the IO kernel's
  completion *embeds* CQ polling, so downstream compute on the same
  stream is ordered after the IO data is actually resident — the stream
  itself is the completion fence. Host-side harvest goes through
  `wait()`/`progress()` on the handle, decoupled from submission — the
  same submit/reap split as io_uring's SQ/CQE pair.

Each controller exposes **16 queue pairs × MQES+1 entries** to userspace
(queue depth always takes the controller maximum, NVMe CAP.MQES + 1;
the runtime follows the controller-reported depth, userspace cannot
override it). One bounded CQ poll budget per entry makes the poll
fail-closed instead of spinning forever on a lost completion.

Code: `tutti/data_paths/local_nvme/io/submit_one.cu`,
`tutti/data_paths/local_nvme/io/nvme_submit_primitives.cuh`,
`tutti/data_paths/local_nvme/io/nvme_queue_group.cu`

## 2. Register-time precomputation — the hot path is table lookup, not arithmetic

`register_memory()` DMA-maps a buffer **exactly once** — PCI bus addresses
are controller-agnostic under IOMMU=pt, so one mapping serves every bound
NVMe controller — pre-splits the buffer into MDTS-sized IO slices, and
pre-builds every PRP1/PRP2 descriptor into a contiguous GPU-resident
array. Submit time is a slice-index table lookup, never PRP math.

Two memory optimizations make this scale to LMCache-scale registration
counts (target deployment: 180 GB KV cache in 128 KiB tensors ≈ 1.47M
registrations):

- **Sub-page PRP-list packing.** With MDTS = 128 KiB, one IO covers at
  most 32 data pages, so a PRP list needs only 31 entries (248 B). Lists
  are packed at 256 B granularity — 16 slices share one 4 KiB page
  (6% → 96% utilization), cutting the PRP-list footprint 16×
  (180 GB KV: 5.6 GiB → ~350 MiB of pinned host memory).
- **Pooled descriptor allocation.** GPU descriptor arrays are bump-
  allocated from a DataPath-level pool (256 MiB segments ≈ 11.2M
  descriptors) instead of one `cudaMalloc` per registration — 1.47M
  registrations cost ~2 segment allocations, not 1.47M `cudaMalloc` calls.

Code: `tutti/data_paths/local_nvme/local_nvme_data_path.cpp`
(`build_prebuilt_descriptors_`),
`tutti/data_paths/local_nvme/metadata/desc_pool.{h,cpp}`

## 3. Two-tier caches sized for LMCache-scale file counts

A file's on-GPU handle is **~200 bytes** — 8 inline extents plus a rare
overflow buffer, mirroring NVMe's own PRP1/PRP2 + PRP-list pattern — an
order of magnitude smaller than a naive all-inline design. Handles live in
a two-tier cache: a small GPU-resident L1 backed by a large pinned-host L2
(default 4× L1). FIEMAP is walked exactly once per file, at L2 admission;
L1 eviction is a *downgrade* (snapshot to L2 via 2×D2H), not a delete, so
a re-touch costs one memcpy restore (2×H2D) instead of a rebuild. L2 is
inclusive: L1-resident handles pin their L2 record, and a genuine L2
eviction is the only true delete.

Code: `tutti/data_paths/local_nvme/metadata/handle_workspace_cache.h`,
`tutti/data_paths/local_nvme/io/device_target.{h,cu}` (snapshot/restore),
`tutti/data_paths/local_nvme/metadata/host_slot_pool.h`

## 4. A batched, event-frugal control plane

- **One H2D per submit in steady state** — batch entries are staged
  host-side and uploaded as ONE contiguous `cudaMemcpyAsync`; PRP lists
  and handles are already resident from registration/open time.
- **Event-fenced slot reuse** — GPU arena slots are released via
  `cudaEventRecord` + `cudaEventQuery` (one shared event per stream, not
  one per slot); release is CPU-side immediate, so LRU evict-then-refill
  never blocks on a GPU sync.
- **Batch open** — `open_batch()` resolves hundreds of per-layer files
  through a parallel FIEMAP thread pool with fail-closed per-item results:
  500 files in ~10 ms vs ~52–117 ms serial (**5–12×**), each returned
  handle independently usable for IO.

Code: `tutti/include/tutti/storage_runtime.h` (`open_batch`),
`tests/data_paths/perf/batch_open/batch_open_perf.cpp`,
`tutti/data_paths/local_nvme/metadata/metadata_arena.cpp`

## 5. Multi-device placement

KV lives in fixed segment files spread over up to 4 NVMe devices. A chunk is one
slot; slot numbers rotate over the devices (`slot % N`) and each device packs
2048 of its slots into one segment file. A chunk's layers are one contiguous
region on one drive, so a single IO is never fragmented across drives; inside a
chunk, splitting happens only at the MDTS layer (contiguous LBA ranges on the
same device). Balance comes from consecutive chunks of a prompt rotating over
the drives. A single fused kernel submits to all drives: a device table of
per-target rows lets one launch fan a batch out across the whole fleet. Each
slot reserves a 32 KiB header prefix so every payload starts 16 KiB-aligned —
with a 4 KiB prefix the same reads were ~20% slower (the SSD's parallelism
follows the LBA pattern).

Measured (8-GPU HY3 TP8, 4 drives, 32 KiB per-layer reads): **26.6 GiB/s**
aggregate read, close to fio on the raw block devices.

Code: `csrc/data_paths/striped_local_nvme/fused_submit_kernel.cu`,
`csrc/data_paths/striped_local_nvme/striped_data_path.cpp`,
`csrc/storage_objects/slot_placement_policy.h`,
`csrc/resolvers/local_file/multi_mount_resolver.h`
