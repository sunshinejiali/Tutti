# Benchmarking and profiling

Runnable examples live in `doc/intergration/tutti-vllm-manual.md`; wrappers are
under `scripts/vllm/run`. Set Python, MODEL, NVMe scratch, daemon inventory,
DEVICE_GROUPS, KV_MOUNT and TP explicitly. Set `CAPACITY` or pass `--capacity`
for the per-rank payload budget; HBM baseline does not need storage capacity.
The online wrapper uses `TOTAL_CAPACITY` for the service total, divided by TP.
Scripts preserve GPU/compiler choices.

## Reuse proof

The offline driver issues A to populate, B to reuse a prefix, with distinct
first tokens across rounds to avoid cross-round hits at the same shape.
Tutti wrapper resets only local HBM prefix cache between A and B. Without it,
a fast B can be served wholly by HBM and prove nothing about NVMe offload.

Require successful commit/read evidence, external hits and matching output
before reporting performance. `--kv-load-failure-policy fail` exposes broken
loads; recompute can mask loss of offload benefit. `--baseline` is an HBM prefix
reference, not a full-recompute B reference.

Use at least three rounds; separate first-round JIT/registration/precreation
costs from steady mean. Small single-sample changes are not a performance claim.
Include model revision, TP, dtype, block/chunk size, context/reuse, load format,
concurrency, cache state, completion mode and tool versions with results.

## Size and isolation

- Queue counts per rank multiply across ranks on each device. Use real daemon
  grants/available capacity, not a fixed per-drive number from another machine.
- Pool capacity is an upper bound. Segment groups are real-zero-filled and
  FIEMAP-verified, and cold startup can be significant. Background growth can
  compete for bandwidth; admission beyond ready capacity may be trimmed.
- `POOL_TAG` only isolates checkpoint/index roots. Segment payloads are shared
  under `<mount>/r<rank>/segments`; no concurrent benchmark/live instance there.
- After interruption inspect clients and committed state. Do not blindly delete
  segments or assert that a fresh tag alone protects existing live data.
- Logs, reports, caches and pytest basetemp must remain on NVMe scratch.

## nsys

Use wrapper `--nsys` after normal correctness acceptance. Keep profiler overhead
separate from non-profiled wall-time comparisons. Reports need real request
execution and expected CUDA/NVTX activity, not only startup.

Classify submit, wait, transfer and compute kernels separately. In synchronous
mode the IO kernel submits and polls; in experimental async mode submit-only
and `wait_batches` are distinct. Direction comes from NVTX `op=read|write` and
the actual stream/path, not grid size. A waiting kernel's busy time is not bytes
transferred, and low async kernel residency does not imply SSD inactivity.

For one step across CPU/GPU asynchronous boundaries, attribute kernels by their
launch correlation and process, not simply CPU-window start/end. Merge busy
intervals to measure overlap; never add concurrent rank/kernel durations and
call that wall time. Compare all ranks: NCCL duration may include skew waiting
for a slow rank rather than transport alone.

PCIe metrics need calibration against a known byte volume. Prefer throughput
from known completed request bytes and elapsed time; do not infer absolute
bandwidth from a device-specific utilization percentage.

## Failure paths and rollout

`TUTTI_ASYNC_COMPLETION=1` is experimental, synchronous is default. Successful
normal-path/256K runs do not certify timeout, late CQE or slot/PRP lifetime.
Fault-injection acceptance remains required before enabling a production default.

Online `/metrics` and client traces complement server logs. HTTP 200 alone
neither proves external cache hits nor absence of writes to unrelated files.
