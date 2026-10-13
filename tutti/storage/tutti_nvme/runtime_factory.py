"""真机 runtime 装配：preset 归一 → daemon 事实推导 → 绑定调用。

本模块归一配置、验证拓扑并选择原生装配入口。默认使用已存在的md RAID0，
单盘与独立多盘需显式选择；不创建或格式化阵列，也不涉及KV key语义。
"""

from __future__ import annotations

import os
from pathlib import Path

from tutti.storage.tutti_nvme.preset_derive import derive_device_fields


def normalize_preset(preset) -> dict:
    """递归归一 preset：字符串值恰为纯十进制整数时转 int。"""
    if isinstance(preset, dict):
        return {k: normalize_preset(v) for k, v in preset.items()}
    if isinstance(preset, list):
        return [normalize_preset(v) for v in preset]
    if isinstance(preset, str) and preset.strip().isdigit():
        return int(preset)
    return preset


def prepare_preset(preset: dict, *, mounts=None) -> dict:
    """推导配置并验证拓扑；实际md级别、成员和offset仍由native resolver核验。"""
    import yaml

    preset = normalize_preset(preset)
    if not isinstance(preset, dict):
        raise RuntimeError("preset 必须是映射")
    removed = {"prp_cache_capacity", "handle_cache_capacity", "handle_cache_l2_capacity"}.intersection(preset)
    if removed:
        raise ValueError(f"cache容量由runtime管理，不支持preset参数 {sorted(removed)}")
    preset_type = preset.get("type", "raid0")
    if preset_type not in ("raid0", "nvme-device-group", "local"):
        raise RuntimeError(f"未知 preset type：{preset_type}")
    preset = dict(preset, type=preset_type)
    if "daemon_config" in preset:
        preset = derive_device_fields(preset, yaml)
    if preset_type == "raid0":
        devices = preset.get("devices") or []
        array_mounts = preset_mounts(preset)
        if len(devices) < 2 or not array_mounts or len(array_mounts) != 1:
            raise RuntimeError("raid0 必须选择至少两个成员且共用一个阵列挂载点")
        if mounts is not None and tuple(map(str, mounts)) != tuple(array_mounts):
            raise RuntimeError("raid0 options.mounts 必须与阵列挂载点一致")
    return preset


def preset_mounts(preset):
    """从多设备配置得到payload挂载点，RAID0成员共用同一个挂载点。"""
    if not isinstance(preset, dict):
        return None
    devices = preset.get("devices")
    if not isinstance(devices, (list, tuple)):
        return None
    mounts = []
    for device in devices:
        if not isinstance(device, dict) or not device.get("mount_path"):
            return None
        # 多块盘可以挂在同一个挂载点后面（md RAID0）：那是一个挂载点。
        if device["mount_path"] not in mounts:
            mounts.append(device["mount_path"])
    return mounts or None


def build_runtime(preset: dict):
    """按 preset dict 构造真机 runtime（daemon_config 推导与归一同环境变量路径）。"""
    preset = prepare_preset(preset)

    try:
        import tutti_runtime  # bindings 构建产物（需在 sys.path/PYTHONPATH）
    except ImportError as exc:
        raise RuntimeError(
            "tutti_runtime 绑定不可用：先构建 csrc/python/ 并安装该扩展"
        ) from exc

    preset = dict(preset)
    preset_type = preset.pop("type")
    preset.pop("daemon_config", None)  # 推导元键不进 runtime preset
    preset.pop("device_id", None)
    if preset_type == "raid0":
        return tutti_runtime.make_raid0_nvme_runtime(preset)
    if preset_type == "nvme-device-group":
        return tutti_runtime.make_nvme_device_group_runtime(preset)
    if preset_type == "local":
        return tutti_runtime.make_local_nvme_runtime(preset)
    raise RuntimeError(f"未知 preset type：{preset_type}")


def build_runtime_from_env():
    """按 TUTTI_NVME_PRESET 构造真机 runtime（本包私有推导）。"""
    import yaml

    raw = os.environ.get("TUTTI_NVME_PRESET", "").strip()
    if not raw:
        raise RuntimeError(
            "runtime=None 需要 TUTTI_NVME_PRESET（yaml/json 内联或文件路径）"
        )
    text = Path(raw).read_text() if os.path.isfile(raw) else raw
    preset = yaml.safe_load(text)
    if not isinstance(preset, dict):
        raise RuntimeError("TUTTI_NVME_PRESET 解析结果必须是映射")
    return build_runtime(normalize_preset(preset))
