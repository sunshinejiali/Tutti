"""preset 设备字段推导：daemon 配置是硬件事实的单一来源。

daemon YAML 的 nvmes[] 按 device_id 给出 pci_addr / backing_mount_path /
namespace_id；preset 只写 device_id（local 为单数 device，striped 为
devices 列表），由此补全为运行时组装器要求显式给出的设备字段。

两种布局共用同一推导规则，避免 store 与 metadata 两处漂移；显式给出的
字段优先（preset 可覆盖 daemon），device_id 仅作查询键、不进运行时。

daemon_config 可以是多份配置（列表，或逗号分隔的字符串）：每个盘组由独立的
daemon 管理时（如 kv0 管 NVMe 0-3、md0 管 NVMe 4-7），客户端把它们合并成一份
设备清单；device_id 在各份之间必须唯一。
"""

from __future__ import annotations

from pathlib import Path

_PCI_SYSFS = Path("/sys/bus/pci/devices")


def derive_device_fields(preset: dict, yaml) -> dict:
    """daemon_config + device_id(s) → 补全设备字段的 preset 副本。

    无 daemon_config → 原样返回。striped 按 devices 列表逐个推导；
    local 按顶层 device_id + device 推导。函数幂等：已补全的设备条目
    再次调用不改变。
    """
    daemon_paths = daemon_config_paths(preset.get("daemon_config"))
    if not daemon_paths:
        return preset
    nvmes: dict = {}
    for daemon_path in daemon_paths:
        daemon = yaml.safe_load(Path(daemon_path).read_text()) or {}
        for entry in daemon.get("nvmes", []):
            device_id = entry.get("device_id")
            if device_id in nvmes:
                raise RuntimeError(
                    f"device_id={device_id} 同时出现在多份 daemon 配置中：{daemon_paths}"
                )
            nvmes[device_id] = entry
    if preset.get("type", "raid0") in ("raid0", "nvme-device-group"):
        derived = dict(preset)
        derived["devices"] = [
            _derive_one(dict(device), nvmes)
            for device in preset.get("devices") or []
        ]
        return derived
    device_id = preset.get("device_id")
    if device_id is None:
        raise RuntimeError("preset 携带 daemon_config 时必须同时给出 device_id")
    derived = dict(preset)
    derived["device"] = _derive_one(
        dict(preset.get("device") or {}, device_id=device_id), nvmes
    )
    return derived


def _derive_one(device: dict, nvmes: dict) -> dict:
    """单个设备条目：device_id → pci_bdf / mount_path / namespace / backing。"""
    device_id = device.pop("device_id", None)
    if device_id is None:
        return device  # 字段已显式给出（无 daemon 查询键）
    entry = nvmes.get(device_id)
    if entry is None:
        raise RuntimeError(f"daemon 配置无 device_id={device_id} 的 NVMe 条目")
    namespace_id = device.get("namespace_id", entry.get("namespace_id", 1))
    device.setdefault("pci_bdf", entry["pci_addr"])
    device.setdefault("mount_path", entry["backing_mount_path"])
    device.setdefault("namespace_id", namespace_id)
    device.setdefault(
        "backing_device", _block_device_for(entry["pci_addr"], namespace_id, device_id)
    )
    return device


def daemon_config_paths(value) -> list[str]:
    """daemon_config 的取值 → 配置文件路径列表（None/空 → []）。"""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _block_device_for(pci_addr: str, namespace_id: int, device_id: int) -> str:
    """该 PCI 控制器上 namespace 的块设备。

    snvme 块设备号由内核按 bind 顺序分配，几个 daemon 各自拉起控制器时它
    不一定等于 device_id，所以按 PCI 地址从 sysfs 查；控制器尚未由 snvme
    接管时退回旧约定 /dev/snvme<device_id>n<ns>（resolver 会按 st_rdev
    再校验一次，错了 fail-closed）。
    """
    controllers = _PCI_SYSFS / str(pci_addr) / "snvme"
    try:
        for controller in sorted(controllers.iterdir()):
            disk = controller / f"{controller.name}n{namespace_id}"
            if disk.exists():
                return f"/dev/{disk.name}"
    except OSError:
        pass
    return f"/dev/snvme{device_id}n{namespace_id}"
