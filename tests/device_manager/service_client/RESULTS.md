# T-008 attach smoke results

## Date

2026-07-30 16:16:30 +08:00

## Git HEAD

`1e0b333d9f6fc1b4f70733a791d979acf4523a26`

## Daemon binary

`/data/home/ryeqiu/Tutti/build/bin/tutti_daemon` — executable and present.

## Client binary

`/data/home/ryeqiu/Tutti/build/bin/nvmeservice_client` — executable and present.

(CMake target name is `nvmeservice_client_example`; OUTPUT_NAME is `nvmeservice_client`.)

## Kernel/module state observed (read-only commands)

Commands used: `uname -r`, `lsmod | grep -E 'snvme|phoenix'`

```text
Kernel: 5.4.241-1-tlinux4-0017.7

snvme                  73728  0
snvme_core             77824  1 snvme
phoenixfs              81920  2
```

No module state was changed by this task. The snvme driver creates block
devices (`snvmeXn1`) and character devices (`ssnvmeX`) via B3 bind+probe when
the daemon starts, and removes them when the daemon exits.

## Generated device mapping

| device_id | PCI BDF | block device | namespace | allowed GPU |
|---:|---|---|---:|---:|
| 0 | `0000:08:00.0` | `/dev/snvme0n1` (auto-detected) | 1 | 0 |
| 1 | `0000:4b:00.0` | `/dev/snvme1n1` (auto-detected) | 1 | 1 |
| 2 | `0000:57:00.0` | `/dev/snvme2n1` (auto-detected) | 1 | 2 |
| 3 | `0000:63:00.0` | `/dev/snvme3n1` (auto-detected) | 1 | 3 |

Note: block device names are auto-detected. The snvme driver creates `snvmeXn1`
after re-bind; initial module load may create `nvmeXn1`. The harness tries both.

## Raw NVMe unmounted/holders check

In execute mode, block devices do not exist before the daemon starts (the snvme
driver creates them via B3 bind+probe). The harness reports this as a warning
and proceeds. After daemon startup, the character devices `/dev/ssnvme0-3` are
verified to exist. No raw NVMe block device is mounted or has holders during
the test.

## Dry-run result

`PASS` (T-013 fix, 2026-07-30)

The four target NVMe PCI devices are in their **steady state: UNBOUND**. This
is not a fault — the devices are managed exclusively by Tutti and are only
bound by the daemon at runtime via B3 (chrdev create → bind → probe). When the
daemon is not running, the devices are UNBOUND and no block device exists.

**Correction of previous misattribution**: the T-008 round attributed the
dry-run BLOCKED to "a previous debugging SIGKILL caused the driver to unbind
the PCI devices." This attribution was **wrong**. The UNBOUND state is the
normal steady state, not an environment anomaly caused by SIGKILL. The dry-run
was structurally broken: it required block devices to exist, but those devices
only exist while the daemon is running — which is exactly when dry-run is not
used.

T-013 fix: dry-run now treats missing block devices as INFO (expected when
daemon is not running). Block devices that DO exist are still checked for
mount targets and non-empty holders — violations remain real errors.

Passed checks: GPU 0-3 queryable, PCI BDFs exist (all UNBOUND — expected),
sudo works, endpoint free, daemon/client executables present.

Exit code: `0` (PASS).

## ListDevices result

`PASS` — rc=0

```text
device_id=0 pci=0000:08:00.0 snvme=/dev/ssnvme0
device_id=1 pci=0000:4b:00.0 snvme=/dev/ssnvme1
device_id=2 pci=0000:57:00.0 snvme=/dev/ssnvme2
device_id=3 pci=0000:63:00.0 snvme=/dev/ssnvme3
```

## Attach results

### device=0 / GPU=0

- Connect: PASS (allocation_id=7e6fb33493eeac51fbcb6a6c598ebd1d)
- nvm_ctrl_attach_client: PASS (step 2)
- nvm_create_group: PASS (gid=1, max_queues=16, granted=2)
- nvm_destroy_group (skip-io): PASS (step 4)
- nvm_ctrl_free_client (skip-io): PASS (step 5)
- heartbeat: PASS (2s hold, no LEASE_REVOKED)
- Disconnect: PASS
- mount_path: under mount_work/gpus/gpu0/
- mount->: mount_work/nvmes/nvme0/GPU0
- Negative markers: clean (no Write IO, Read IO, mapped SQ/CQ, etc.)

### device=1 / GPU=1

- Connect: PASS (allocation_id=a0c3a72ca2dff3489261427b4808c43c)
- nvm_ctrl_attach_client: PASS
- nvm_create_group: PASS (gid=1, granted=2)
- nvm_destroy_group (skip-io): PASS
- nvm_ctrl_free_client (skip-io): PASS
- heartbeat: PASS (2s hold, no LEASE_REVOKED)
- Disconnect: PASS
- mount_path: under mount_work/gpus/gpu1/
- mount->: mount_work/nvmes/nvme1/GPU1
- Negative markers: clean

### device=2 / GPU=2

- Connect: PASS (allocation_id=740030d26a195e02816d5be3c5f4b9a2)
- nvm_ctrl_attach_client: PASS
- nvm_create_group: PASS (gid=1, granted=2)
- nvm_destroy_group (skip-io): PASS
- nvm_ctrl_free_client (skip-io): PASS
- heartbeat: PASS (2s hold, no LEASE_REVOKED)
- Disconnect: PASS
- mount_path: under mount_work/gpus/gpu2/
- mount->: mount_work/nvmes/nvme2/GPU2
- Negative markers: clean

### device=3 / GPU=3

- Connect: PASS (allocation_id=a4ceda0264c38fb245b30c814a5c3c88)
- nvm_ctrl_attach_client: PASS
- nvm_create_group: PASS (gid=1, granted=2)
- nvm_destroy_group (skip-io): PASS
- nvm_ctrl_free_client (skip-io): PASS
- heartbeat: PASS (2s hold, no LEASE_REVOKED)
- Disconnect: PASS
- mount_path: under mount_work/gpus/gpu3/
- mount->: mount_work/nvmes/nvme3/GPU3
- Negative markers: clean

## Symlink verification

During each attach, the daemon created symlinks:

```text
gpus/gpu0/ssnvme0 -> nvmes/nvme0/GPU0
gpus/gpu1/ssnvme1 -> nvmes/nvme1/GPU1
gpus/gpu2/ssnvme2 -> nvmes/nvme2/GPU2
gpus/gpu3/ssnvme3 -> nvmes/nvme3/GPU3
```

After daemon exit, all symlinks were cleaned up (0 remaining per GPU).

## Daemon cleanup

`PASS` — daemon received SIGTERM (delivered via `sudo -n kill` to the real
daemon PID, not the sudo wrapper). Daemon log contains:

```text
Shutting down...
tutti_daemon exited cleanly.
```

## Heartbeat verification

Each client used `--hold 2` with `heartbeat_interval_sec: 1`, `timeout_sec: 5`.
All four clients held sessions for 2 seconds with the heartbeat thread running.
No `lease revoked by daemon for allocation` notices appeared in any client log.

Limitation: the `--skip-io` path does not create user queue pairs or map data
buffers. Heartbeat verification only confirms the bidi-stream heartbeat tick
survived the hold window without lease revocation.

## Log directory

```text
/data/home/ryeqiu/Tutti/tests/device_manager/service_client/.work/logs/20260730-161630/
├── attach_config.yaml
├── daemon.log
├── client_list.log
├── client_device_0_gpu_0.log
├── client_device_1_gpu_1.log
├── client_device_2_gpu_2.log
├── client_device_3_gpu_3.log
├── harness.log
└── mount_work/
    ├── gpus/gpu0..3/
    └── nvmes/nvme0..3/
```

## Verification

```text
python3 -m py_compile tests/device_manager/service_client/generate_attach_config.py  PASS
bash -n tests/device_manager/service_client/run_attach_smoke.sh                    PASS
tests/device_manager/service_client/run_attach_smoke.sh                            PASS (T-013 fix: UNBOUND is steady state)
tests/device_manager/service_client/run_attach_smoke.sh --execute                  PASS
```

## Overall

`PASS` (execute mode)

The `--execute` run passed all success criteria:
1. Generator correctly handles absent block devices (daemon creates them via B3).
2. Dry-run PASS (T-013 fix: UNBOUND is the expected steady state, not a fault).
3. Daemon successfully listening on 127.0.0.1:50051.
4. ListDevices returned all four devices.
5. All four Connect + attach + create_group + destroy_group + disconnect succeeded.
6. All four had 2s heartbeat windows with no LEASE_REVOKED or RPC failures.
7. No block IO, user queue, or data buffer map markers in any client log.
8. Daemon exited cleanly via SIGTERM.
9. Complete logs saved in `.work/logs/20260730-161630/`.
10. No files outside the allowed list were modified.
11. No kernel module state changes; no insmod/rmmod/mount/format.
