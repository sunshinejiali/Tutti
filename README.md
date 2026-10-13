# Tutti

**A GPU-centric, SSD-backed KV cache object store for long-context LLM serving.**

**Current release: v0.1.1** — stable `StorageRuntime` public API, striped multi-device data path, and a vendor-neutral GPU framework.

Tutti (Italian for "all instruments together") makes NVMe SSDs a practical KV-cache tier for LLM inference, building on the ideas of [GeminiFS](https://www.usenix.org/conference/fast25/presentation/qiu) (FAST'25), a companion file system for GPUs. **The CPU launches I/O kernels; the GPU executes them.** Each GPU kernel moves KV cache between SSDs and HBM on its own — giving SSD-backed KV cache DRAM-like performance with hundreds of times the capacity.

## 📰 News
- **2026.8.11**   Added support for Metax GPU.
- **2026.8** — **v0.1.1**: stable `StorageRuntime` public API (open/open_batch/register/submit/wait) over replaceable resolver/binding/DataPath boundaries; striped multi-device data path reaches 25.0 GB/s on 4× NVMe (98% of near-saturated per-drive bandwidth); vendor-neutral `cuda_like` GPU framework (CUDA proven; MUSA/MACA profiles); kernel P2P layer split into symmetric nvidia/metax backends; batch open (500 files in ~10 ms).

- **2026.7** — `snvme` kernel module update: the kernel NVMe stack and GPU-side `libnvm` coexist on one drive — `tutti_daemon` mounts it once, GPU processes lease I/O queues on demand, and crashed processes are reclaimed automatically. Poll-based I/O submission from both user-space threads and GPU kernel threads.

- **2026.5** — Tutti paper released on [arXiv](https://arxiv.org/abs/2605.03375) (SSD-backed KV cache: −78.3% TTFT, 2× request rate, −27% serving cost vs. GDS-enabled LMCache).

## ✨ Product Highlights

- **GPU io_uring** — the batch-entry array and IO handle are shared CPU↔GPU submission state: one `cudaMemcpyAsync` hands a whole batch across, one kernel launch is the only boundary crossing, and completion rides the CUDA stream so submit and harvest stay decoupled. Each GPU thread resolves LBAs, writes SQEs, rings doorbells, and polls completions on its own. No CPU on the data path.
- **Register once, then just look up** — PRP descriptors are pre-built at registration; one DMA mapping serves every NVMe controller. The I/O hot path is table lookup, not arithmetic.
- **Millions of files within GPU reach** — ~200-byte GPU-resident file handles plus two-tier (GPU L1 / pinned-host L2) caching; FIEMAP is walked exactly once per file.
- **Multi-device KV storage** — chunks rotate over backing mounts and live in per-rank segment files; one fused submit drives the selected devices. A backing mount may instead be an md RAID0 array: the resolver translates its extents to member namespaces. Controller identity is verified by PCI address, never by device-number coincidence.
- **Batch open** — `open_batch` resolves hundreds of per-layer files with a parallel FIEMAP pipeline and fail-closed per-item results: 500 files in ~10 ms vs ~50–115 ms serial (5–12×).
- **snvme — one drive, kernel and GPU together** — the kernel NVMe stack and GPU-side `libnvm` are served by the same Linux kernel: the drive stays a mounted, POSIX-visible block device while GPU processes get their own user I/O queues. `tutti_daemon` mounts the NVMe driver once; every GPU process then leases queues on demand, and a crashed process has its queues reclaimed by the kernel's fd-close cascade — dynamic allocation, isolated failures.
- **Beyond NVIDIA** — three-layer `cuda_like` framework (profile selector → vendor shim → kernel primitive macros): CUDA proven in production, MUSA/MACA build profiles in place; the kernel P2P layer is split into symmetric per-vendor backends (nvidia done, metax symmetric).

## What is Tutti

Modern LLM engines (vLLM, SGLang) page the KV cache into small, scattered GPU memory blocks. Once that cache is offloaded to SSDs, memory fragmentation becomes **I/O fragmentation**: restoring a long prefix means issuing hundreds of thousands of tiny random I/Os — and in a CPU-centric design, every single one is initiated by the CPU. The result is massive GPU stalls: reusing KV cache from SSD ends up slower than recomputing it, and the penalty only grows as GPU compute keeps getting faster.

The conductor doesn't play the instruments: the CPU only cues the orchestra (launches I/O kernels); the GPU musicians read their own descriptors and drive the NVMe drives directly:

<p align="center">
  <img src="doc/tutti-arch.png" alt="Tutti architecture: the CPU conducts — launching I/O kernels only — while the GPU drives NVMe SSDs directly for AI applications" width="900">
</p>

Tutti's answer is a GPU-centric data path where the CPU appears only **O(1) times per batch, not per I/O** — detailed in [doc/architecture/system-architecture.md](doc/architecture/system-architecture.md).

## System Architecture (v0.1.1)

```text
Application  (uri / offset / size)
    │  open / register / submit / wait      ← CPU appears O(1) per batch
    ▼
StorageRuntime  — stable public API
    │  resolver + binding → StorageTarget
    ▼
DataPaths  — local_nvme / striped_local_nvme
    │  PRP prebuilt at register; GPU-resident handle cache
    ▼
libnvm + tutti_daemon  →  snvme kernel module
    │  P2P DMA: GPU threads drive NVMe directly
    ▼
NVMe SSDs  ◄──────►  GPU HBM
```

Cross-cutting: the `cuda_like` three-layer GPU framework (profile selector →
vendor shim → kernel primitive macros) covers every layer above the kernel —
CUDA proven, MUSA/MACA profiles in place; the kernel P2P layer is split into
symmetric per-vendor backends (nvidia done, metax symmetric).

**Environment requirements and module verification**

- OS: Linux. The unified `snvme/` tree selects `5.4-tlinux4`, `5.10`, `5.15`, or `6.8` from the running kernel. See [`PORTING.md`](csrc/device_manager/nvme/kernel_modules/PORTING.md) for the matrix.
- Accelerator: NVIDIA GPU + CUDA toolkit (`nvcc`); bare metal with IOMMU in passthrough mode.
- Runtime: daemon-only — `tutti_daemon` bring-up creates devices and views; queue depth always takes the controller maximum (NVMe CAP.MQES + 1).
- Host deps: CMake, protobuf / gRPC / uuid / yaml-cpp. CMake discovers system packages by default; an existing vcpkg tree is an optional fallback.
- All file I/O is opened with `O_DIRECT` (project policy).

## Deploy the vLLM KV cache connector

**v0.2.0 preparation:** the vLLM connector, deployment scripts and agent skill
are being prepared for community use. The release version/tag has not yet been
changed. Supported-layout and compatibility limits are documented explicitly.

Start with [Getting Started](doc/getting-started.md): build the C++ stack and
both native Python extensions, review host/device ownership, run software and
byte-exact hardware gates. Then use the
[vLLM operation manual](doc/intergration/tutti-vllm-manual.md) for offline reuse,
service configuration preview, `/health`, and SSE streaming acceptance.

Prefer agent-assisted setup? Install the complete
[Tutti runtime skill](Skills/README.md), including its references. It requires
host-specific device identity and explicit approval before privileged changes.
Shipped daemon configurations are examples of the maintainer's machine, not
safe defaults for arbitrary disks.

MiniMax/DeepSeek new-model adaptation is a subsequent task. Hybrid/multi-group
KV layouts are not currently supported. Experimental async completion is
opt-in (`TUTTI_ASYNC_COMPLETION=1`); fault-path validation is still incomplete
and it is not the production default.

## Manual C++ KV workload

The canonical build flow is [doc/getting-started.md](doc/getting-started.md).

`layerwise_kv_overlap` is a manual workload that requires daemon-published
`--directory` paths; its source header contains the run commands and `--help`
lists all runtime options.

## Configuration

Configuration is organized by its consumer; see the
[configuration index](config/README.md) and
[end-to-end configuration/performance guide](doc/configuration-and-performance.md).

- **`config/device_manager/nvme_daemon.yaml`** is the daemon schema template;
  reviewed host facts live in `config/local/tutti_daemon_<name>.yaml`.
- **`config/storage_runtime/raid0.yaml`** is the recommended native deployment;
  `local_nvme.yaml` / `nvme_device_group.yaml` are explicit alternatives. These describe the native
  `TuttiRuntimeSpec` resource/resolver/datapath/backend graph. Runtime device
  facts and queue grants come from the daemon allocation response.
- **`config/integration/vllm/kv_transfer.json`** is the separate Python connector
  preset schema. It does not implicitly load the native Runtime YAML.

Use the module paths directly; old template aliases and positional daemon
schemas are removed. The daemon requires an explicit `--config` pointing to
reviewed host facts. Python uses only unit-bearing `capacity`, while engines
and object stores calculate bytes and slot counts internally.

Legacy application fields (`gpu`, `nvme_service`, `nvme`, `local_nvme`,
`local_nvme_config`, `storage.backend`, and `storage.default_stripe_unit`) were
are not supported. There is no legacy configuration adapter; native Runtime
and daemon inputs must use their current schemas.

DataPath cache precedence remains programmatic override > DataPath spec
config > test-only `TUTTI_*_CACHE_CAP` environment override > built-in default.
Queue depth remains kernel-authoritative (controller maximum, NVMe CAP.MQES + 1).

The repository root is the only supported CMake entry. All component
`CMakeLists.txt` files are reached from that target graph and are not
standalone project entries. Runtime is daemon-only.

## Deep Dive

- [Getting Started](doc/getting-started.md) — 唯一默认硬件 build：CUDA、SNVMe 模块与 daemon
- [System Architecture](doc/architecture/system-architecture.md) — the as-implemented layers, IO walkthrough, and deployment topology
- [Key Designs](doc/architecture/key-designs.md) — the five performance designs behind the GPU-centric data path, with measured numbers
- [Backend SPI](doc/design/backend-spi.md) — the DataPath / Resolver / Binding semantic contracts
- [GPU Porting Guide](doc/gpu-porting-guide.md) — the `cuda_like` three-layer framework, primitive semantic contracts, and Metax integration steps
- [Kernel Portability](doc/design/kernel-portability.md) — the snvme module across kernel versions and GPU vendors
- [snvme Design Rationale](doc/architecture/snvme-design-rationale.md) — why a custom NVMe driver is required (no mainline facility exposes user-owned NVMe queues; RDMA user-doorbell analogy)
- [Extending Tutti](doc/extending_tutti.md) — adding resolvers, bindings, data paths, and SNVMe smoke tests
- [Advanced Build](doc/advanced-build.md) — driver rebuild, reload, Kbuild, and baseline maintenance
- [Contributing](CONTRIBUTING.md) — install, test, and contribution rules

## Cite

If you use Tutti in your research, please cite our paper — and the GeminiFS work it builds on:

```bibtex
@article{tutti,
  title   = {Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving},
  author  = {Qiu, Shi and Hu, Yifan and Wang, Xintao and Zhu, Wenhao and Yan, Jianqin and Chen, Hao and Xu, Kaiqiang and Chen, Kai and Zhang, Yiming},
  journal = {arXiv preprint arXiv:2605.03375},
  year    = {2026}
}

@inproceedings{geminifs,
  author    = {Shi Qiu and Weinan Liu and Yifan Hu and Jianqin Yan and Zhirong Shen and Xin Yao and Renhai Chen and Gong Zhang and Yiming Zhang},
  title     = {{GeminiFS}: A Companion File System for {GPUs}},
  booktitle = {23rd USENIX Conference on File and Storage Technologies (FAST 25)},
  year      = {2025},
  isbn      = {978-1-939133-45-8},
  address   = {Santa Clara, CA},
  pages     = {221--236},
  publisher = {USENIX Association},
  month     = feb
}
```

## Acknowledgements

Tutti's pluggable GPU-vendor framework (`cuda_like`) draws on the design patterns
pioneered by [Mooncake](https://github.com/kvcache-ai/Mooncake) (Apache 2.0),
which similarly decouples the GPU runtime API surface from the kernel that
consumes it. See [Mooncake's Apache license](https://github.com/kvcache-ai/Mooncake/blob/main/LICENSE-APACHE)
for the full license text.

## Contact & community

*Contact information to be added (maintainers / mailing list / Slack or WeChat group).* For bug reports, please use the issue template — see [CONTRIBUTING.md](CONTRIBUTING.md).

Questions, adaptation proposals, or interest in becoming a community maintainer / manager — reach out to Shi Qiu at ryeqiu@tencent.com.

## License

Apache-2.0 — see [LICENSE](LICENSE).
