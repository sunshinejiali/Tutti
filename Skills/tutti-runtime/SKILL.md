---
name: tutti-runtime
description: Deploy, build, test, and operate Tutti GPU-direct NVMe KV cache offloading and its vLLM connector. Use for new-host PCIe/IOMMU and disk identity checks, kernel module/daemon setup, native extensions, offline prefix reuse, online API and streaming acceptance, nsys profiling, and KV offload diagnostics. Also use before changing controllers, mounts, or daemon ownership in the Tutti repository.
---

# Tutti Runtime

Tutti uses GPU-initiated NVMe IO to move KV between SSD and GPU memory. The vLLM
entry is `tutti.integration.vllm.connector.TuttiConnectorV1`; no vLLM or
FlashInfer source modifications are required.

## Start here

Locate the checkout before using any relative path. Read `doc/getting-started.md`
for the canonical install/build/host/test sequence and
`doc/intergration/tutti-vllm-manual.md` for offline reuse and online service.
Read `tests/README.md` to select tests by architecture module and dependency:
CTest adds `module_<module>` labels; Python tests live under matching modules
in `tests/python/`. Distinguish GPU-only synthetic tests from real NVMe IO.
For configuration, read `config/README.md` and
`doc/configuration-and-performance.md`: daemon, native Runtime YAML and Python
preset are separate consumers. Prefer automatic model geometry/in-flight
values and sync workload batch sizing. Preserve existing batch budgets for
cross-step async reads; shrinking those needs a matching completion/backpressure
design. Native model-free defaults are finite 16 operations / 256 entries;
review devices/capacity/queue budget explicitly. PRP cache is internally
budgeted in host pinned memory using controller page size and effective IO
limits, not GPU free memory. Do not emit cache capacity knobs or unused
allocator/reclaim/granularity options. Application factories avoid duplicate
handle L1/L2 caching of long-lived file tickets. Local/multi worker defaults
still differ; do not claim a universal optimum or auto-run sweeps.
Python users may omit wave/IO stream/queue knobs; the connector provides defaults
without changing low-level store/preset callers. User capacity uses only
`capacity: "64 GiB"` (per rank); raw byte/slot option aliases are removed.
Read `config/integration/vllm/README.md` for plain field explanations; never
guess device_groups from GPU/disk numbers. The default preset is `raid0` over
an existing md array; native examples use `config/storage_runtime/raid0.yaml`.
Verify all selected namespaces are exactly one array's members; never create
or format an array to make a config pass. Use `nvme-device-group` to describe
the shared IO controller set and `selection: device-group` to allocate that set;
neither means RAID0. Use `local` for one device. Reject the removed `multi`,
`multi-nvme` and `striped` configuration names.
If only this skill is installed, first obtain the Tutti checkout; the skill
contains instructions, not the compiled runtime or host configuration.

Follow this sequence without skipping gates:

1. Read `references/hardware.md`; identify the intended GPU/SSD PCI BDFs,
   serials, namespaces, filesystem, mount users, and P2P/IOMMU constraints.
   Ask permission before host-level changes. Prefer KV-only data disks.
2. Select the Python environment and an NVMe scratch directory. Install
   CUDA-compatible PyTorch/vLLM first; build Tutti and both native extensions
   against the same environment. Root package installation alone is insufficient.
3. Write host-specific daemon YAML and validate it. Shipped `config/local/`
   files describe the maintainer's machine, not this machine. For existing
   RAID0, configure and rehearse a safe handover before starting a daemon.
4. Run software tests; then, with explicit permission and reviewed test paths,
   run hardware byte-exact read/write and controller-identity tests.
5. Run a small offline A/B workload. Clear HBM prefix cache between A and B;
   require real external-read evidence, no IO failures, and matching output.
6. Preview service configuration, start the API, check `/health` and
   `/v1/models`, and verify SSE tokens with `[DONE]`. Only then run a trace.

## Safety boundaries

Never infer SSD identity from `nvmeN`, `snvmeN`, or `ssnvmeN`: their allocators
and bring-up order differ. Match PCI/namespace and the kernel-reported identity.

Do not automatically format disks, erase RAID superblocks, unmount busy data,
change `/etc/fstab`, bind/unbind controllers, or unload modules. An explicit
request to deploy does not authorize destroying existing host data.

Stop API and all IO workers before stopping a daemon or switching drivers.
Never use a second SIGTERM, SIGKILL, or lazy unmount as proof that controller
release is safe. A daemon's supervisor cannot guarantee rollback if it too is
killed, or if clients still hold the devices.

`POOL_TAG` isolates checkpoint/index directories, **not** the shared
`<mount>/r<rank>/segments` payload files. Never run two instances or a benchmark
and live service against the same mount/rank workspace. Do not delete segment
files while any index/service uses them.

Keep scratch, pytest basetemp, JIT caches, logs, and profiler exports on NVMe.
Preserve the caller's GPU selection, compiler, and Python. Do not overwrite
live libraries or remove the CMake build tree behind deployed extensions.

## Command selection

- `scripts/tutti-env.sh status`: read-only environment diagnosis; a failure may
  simply mean modules/daemon are intentionally stopped.
- `build` / `build-ext`: C++ stack / two in-place Python extensions.
- `test py|cpp|all`: software tests; C++ excludes `hardware` by default.
- `test cpp --hardware`: explicit hardware IO; review every selected config first.
- `daemon start|stop|restart|status [NAME...]`: configuration-specific processes.
  Without NAME, the script considers all local daemon configs. Always name the
  intended deployment on an unknown/shared machine.
- `scripts/tutti-md.sh`: supervised RAID handover. Its table currently contains
  maintainer-specific PCI sets, UUIDs and mounts; new hosts require new entries.
  `protected` handovers need `--confirm NAME`, and confirmation does not bypass
  busy-client checks. Do not use `bootstrap` blindly on a fresh machine.
- `scripts/vllm/run/serve-8gpu-striped.sh --dry-run`: real JSON/command preview,
  not model execution. The historical name does not force TP8.
- `scripts/vllm/run/bench-8gpu-striped.sh`: offline reuse / HBM reference / nsys.

Required run inputs: `MODEL`, `TUTTI_PROFILE_ROOT`, selected Python
(`TUTTI_PYTHON`), `DAEMON_CONFIG`, `DEVICE_GROUPS`, `KV_MOUNT`, and TP. Select
model-specific block/context/KV dtype/parser instead of trusting Hy3 defaults.
Serve uses unit-bearing `TOTAL_CAPACITY` (service total); bench requires
`CAPACITY` or `--capacity` (per rank). The daemon always requires `--config`
with the reviewed explicit-identity schema, with no old-path fallback.
Standard weight loading is `auto`; Phoenix is optional and not needed for KV
IO. See the operation manual for runnable commands and API checks.

## Supported layouts and experimental features

Current connector geometry is a single KV cache group with equal per-layer
page bytes. Hybrid/multi-group layouts are rejected. A model loading in vLLM
alone is not evidence of Tutti compatibility. MiniMax/DeepSeek new-version
adaptation is pending model identification and separate verification.

`TUTTI_ASYNC_COMPLETION=1` selects experimental submit/drain separation for the
striped DataPath; default is synchronous completion. Lifetime fixes now retain
resources until DMA and all waiters retire; shutdown timeout is retryable and
does not cancel IO. GPU-only synthetic CQ/late completion/waiter tests and
software release-failure tests exist, but the revised code has not rerun real
NVMe/LLM regression or controller-reset fault injection. Do not describe it as
fully certified. See `doc/design/async-nvme-completion.md`.

## Code invariants

1. Scheduler modules and package initializers must not import torch, CUDA, or
   native data-plane extensions. Worker/scheduler increments publish residency.
2. An IO handle can represent partial acceptance. Inspect `initial_states`,
   not just handle presence; rejected entries did not execute.
3. Only confirmed successful IO may publish writes as committed. Wait errors,
   invalid observations, missing results, and failed states are not success.
4. Data-file IO uses `O_DIRECT` with aligned buffers, offsets and lengths.
5. Queue depth and MDTS come from the controller/kernel. Never create a smaller
   shadow ring or assume another host's queue count.
6. `<tutti/...>` is the installed public-header namespace, not an arbitrary
   repository directory prefix. Avoid blanket path replacement.

## Read references on demand

- `references/hardware.md`: identity, prerequisites, safe host transitions.
- `references/architecture.md`: layers, index authority, segment layout.
- `references/benchmarking.md`: sizing, isolation and profiling interpretation.
- `references/diagnostics.md`: discriminating checks for failures.

Start diagnostics with read-only status/logs. Use GPU timelines to distinguish
host stalls from device waits, Python/native stacks to identify the cause, and
nsys launch correlation for cross-stream steps. Save evidence on NVMe. Do not
attribute an idle gap to the first kernel after it or infer SSD bandwidth from
summed waiting-kernel times.

Keep this skill synchronized with scripts and supported-layout guards. Prefer
encoding lessons in checks/tests over accumulating machine-specific history.
