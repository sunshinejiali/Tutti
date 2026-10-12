# Hardware and safe host setup

Read this before any privileged deployment. A wrong controller or DMA mapping
can corrupt unrelated files; a successful model response does not certify it.

## Host checklist

| Item | Verify |
|---|---|
| NVIDIA GPU / CUDA / torch | Driver/toolkit compatibility; torch CUDA availability in the selected Python |
| Supported running kernel | Matching headers and the verification matrix in `csrc/device_manager/nvme/kernel_modules/PORTING.md` |
| P2P route | GPU/SSD ancestry with `lspci -tv`; prefer same switch or NUMA, verify other routes experimentally |
| IOMMU | Passthrough or disabled; translated DMA addresses are not this runtime's supported assumption |
| SSD identity | PCI BDF, serial/model/capacity, namespace, filesystem and array UUID, not ordinal names |
| Ownership | No active filesystem/array/client can be disrupted by the intended handover |
| Space | Scratch/JIT/logs and first segment group have room on the intended NVMe |

`pci_topology_check.sh` prints topology and writes a JSON inventory. Its output
path is a host artifact, not automatically a safe place for large logs. The
mapping needs an operator's interpretation before selecting device groups.

MDTS and queue depth are kernel/controller authority. `ranks_per_device ×
queues_per_rank` must fit the actual granted user-QID pool, not another host's
measured capacity.

## Deployment facts

Create `config/local/tutti_daemon_<name>.yaml` from the canonical schema after
checking the physical machine. Each daemon owns a disjoint controller set and
an endpoint. Client inventories merging files require globally unique
`device_id` values. `allowed_accel_ids` must cover every GPU that uses a group.

The shipped kv0/md0 files are maintainer deployment examples, not auto-detected
configs. `scripts/tutti-md.sh` also contains maintainer-specific PCI sets, UUIDs,
mounts and `normal`/`protected` levels. Register a new array explicitly and
exercise rollback with disposable data before using it with important files.

The daemon controls ownership; clients obtain/validate actual controller and
namespace identities. Block-controller and character-device numbers may differ.
Never synthesize `/dev/ssnvmeN` from a block-controller number.

## Order and gates

Build/install does not load modules. With explicit operator permission:

1. Verify filesystem preparation on the intended KV-only devices. Formatting
   and RAID recreation are not default installation steps.
2. Load matching snvme modules (sign according to host policy if required).
3. For an existing array: stop clients, safely unmount and stop the array via
   the reviewed supervisor handover; only then bind to snvme.
4. Start the named daemon, verify endpoint, mount source/members and views.
   Devices on snvme appear only after daemon bring-up.
5. Run software tests, then reviewed hardware byte-exact tests on allowed paths.
6. Run small reuse benchmark, then online acceptance.

Use `tutti-env.sh --dry-run` before operations. Never invoke all local daemons
blindly or use `bootstrap` on a new host just because the repository contains
working maintainer configuration.

## Stop and rollback

Stop API/workers first. At every safety level, controller release is unsafe
while GPU IO clients still drive it. Protected arrays additionally refuse a
busy mount and require explicit `--confirm NAME` for handover.

The supervisor attempts return to Linux nvme and the same md superblocks on a
daemon exit. This is not an unconditional guarantee: busy clients, dead/hung
hardware, stale process state, or killing the supervisor may require manual
recovery. Do not override this with a second daemon signal or lazy unmount.
Do not modify fstab or mdadm configuration without an explicit site decision.

## Module and library maintenance

Module and userspace UAPI must match. Sources are under
`csrc/device_manager/nvme/kernel_modules/`; baseline presence is not a hardware
certification. `modules rebuild` only rebuilds modules: rebuild libnvm/userspace
and relink Python too if the shared ABI changes. Never replace live mapped
libraries or delete a deployed build tree.

Phoenix weight loading is optional. Its BAR remapping can conflict with
nvidia_peermem DMA mapping; do not load/unload it automatically to resolve RDMA
errors. Record module state and follow the site's driver procedure.

## Cleanup

`kv_root` / `POOL_TAG` isolate indexes but segment data lives at
`<mount>/r<rank>/segments`. Multiple deployments with the same mount/rank collide.
Inventory users and references before deleting anything. A new tag is not a
safe workaround for concurrent IO, and fsfreeze is not a correctness test.
Full filesystem repair/checks require an unmounted, inactive filesystem.
