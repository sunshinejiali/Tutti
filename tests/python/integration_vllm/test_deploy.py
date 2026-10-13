"""多副本占位符与 preset 直传的部署机制测试。"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from tutti.integration.vllm.connector import TuttiConnectorV1
from tutti.integration.vllm.geometry import (
    apply_capacity, apply_device_groups, deployment_rank, expand_placeholders,
    resolve_io_budget,
)
from tutti.storage.tutti_nvme.store import _normalize_preset
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)


def _budget_config(context=4096, sequences=1):
    return SimpleNamespace(model_config=SimpleNamespace(max_model_len=context),
                           scheduler_config=SimpleNamespace(max_num_seqs=sequences))


def _budget_extra(layers=80, load_mode="sync"):
    return {"num_layers": layers, "load_mode": load_mode, "chunk_tokens": 512,
            "max_chunks_per_wave": 512, "kv_block_tokens": 128, "kv_page_bytes": 16384}


@pytest.mark.parametrize("layers, mode, expected", [(80, "sync", 160), (80, "auto", 240), (3, "sync", 6)])
def test_io_budget_matches_layer_lifetime(layers, mode, expected):
    options = {"preset": {}}
    got = resolve_io_budget(options, _budget_extra(layers, mode), _budget_config())
    assert got["preset"]["max_in_flight_operations"] == expected
    assert options == {"preset": {}}


@pytest.mark.parametrize("context, sequences, entries", [(4096, 1, 2048), (131072, 1, 4096),
                                                         (131072, 4, 16384), (1048576, 256, 16384)])
def test_batch_budget_follows_workload_with_finite_ceiling(context, sequences, entries):
    got = resolve_io_budget({"preset": {}}, _budget_extra(), _budget_config(context, sequences))
    assert got["preset"]["max_batch_entries"] == entries


@pytest.mark.parametrize("mode", ["auto", "async"])
def test_cross_step_async_budget_is_not_shrunk_to_one_step(mode):
    got = resolve_io_budget({"preset": {}}, _budget_extra(load_mode=mode), _budget_config())
    assert got["preset"]["max_batch_entries"] == 16384
    assert got["preset"]["max_in_flight_operations"] == 240


@pytest.mark.parametrize("value", [4, 4096, 0])
def test_io_budget_preserves_explicit_limits(value):
    options = {"preset": {"max_batch_entries": value, "max_in_flight_operations": value}}
    assert resolve_io_budget(options, _budget_extra(), _budget_config()) == options


def test_io_budget_without_model_uses_bounded_existing_defaults():
    for preset_type, entries in [("local", 4096), ("raid0", 16384)]:
        got = resolve_io_budget({"preset": {"type": preset_type}}, _budget_extra(), SimpleNamespace())
        assert got["preset"]["max_batch_entries"] == entries
    options = {"capacity": "64 GiB"}
    assert resolve_io_budget(options, _budget_extra(), _budget_config()) is options


@pytest.mark.parametrize("role", ["worker", "scheduler"])
def test_factories_pass_the_same_model_budget_without_io(monkeypatch, role):
    import tutti.engine.core as core
    import tutti.engine.metadata as metadata
    import tutti.storage.registry as registry
    import tutti.storage.metadata as stores_metadata
    from tutti.integration.vllm.factory import worker_engine_for, scheduler_index_for

    captured = {}
    create = lambda _type, opts: captured.update(opts) or object()
    monkeypatch.setattr(registry, "create_store", create)
    monkeypatch.setattr(stores_metadata, "create_metadata_store", create)
    monkeypatch.setattr(core, "KVEngine", lambda *_: object())
    monkeypatch.setattr(metadata, "SchedulerMetadataIndex", lambda *_: object())
    cfg = _budget_config()
    cfg.parallel_config = SimpleNamespace(rank=0, tensor_parallel_size=1)
    extra = {**_budget_extra(), "chunk_kv_bytes": 512 * 1280, "store": {
        "type": "tutti_nvme", "options": {"capacity": "64 GiB", "root": "/unused", "preset": {}}}}
    (worker_engine_for if role == "worker" else scheduler_index_for)(cfg, extra)
    assert captured["preset"]["max_batch_entries"] == 2048
    assert captured["preset"]["max_in_flight_operations"] == 160
    assert extra["store"]["options"]["preset"] == {}


class TestLocalRankPlaceholder:
    """{LOCAL_RANK} 占位符：递归替换、缺省 0、非字符串值不动。"""

    def test_replace_in_nested_strings(self):
        spec = {
            "root": "/mnt/nvme{LOCAL_RANK}/pool",
            "num_chunks": 128,
            "preset": {
                "device_id": "{LOCAL_RANK}",
                "gpu_id": "{LOCAL_RANK}",
                "pci_bdf": "0000:4b:00.0",
            },
        }
        os.environ["LOCAL_RANK"] = "3"
        try:
            got = expand_placeholders(spec)
        finally:
            del os.environ["LOCAL_RANK"]
        assert got["root"] == "/mnt/nvme3/pool"
        assert got["num_chunks"] == 128  # 数值不动
        assert got["preset"]["device_id"] == "3"
        assert got["preset"]["pci_bdf"] == "0000:4b:00.0"

    def test_defaults_to_zero(self):
        os.environ.pop("LOCAL_RANK", None)
        assert expand_placeholders("/mnt/nvme{LOCAL_RANK}") == "/mnt/nvme0"

    def test_vllm_config_rank_takes_priority(self):
        """vllm_config.parallel_config.rank 优先于环境变量（V1 worker 无 env）。"""
        cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(rank=3),
        )
        os.environ["LOCAL_RANK"] = "1"
        try:
            assert expand_placeholders("/mnt/nvme{LOCAL_RANK}", cfg) == "/mnt/nvme3"
        finally:
            del os.environ["LOCAL_RANK"]

    def test_initialized_world_local_rank_is_authoritative(self, monkeypatch):
        """Worker config is shared; process-local world rank selects hardware."""
        import vllm.distributed.parallel_state as parallel_state

        monkeypatch.setattr(
            parallel_state,
            "get_world_group",
            lambda: SimpleNamespace(local_rank=2),
        )
        cfg = SimpleNamespace(parallel_config=SimpleNamespace(rank=0))
        assert deployment_rank(cfg, worker=True) == "2"
        assert expand_placeholders(
            {"device_id": "{LOCAL_RANK}", "gpu_id": "{LOCAL_RANK}"}, cfg,
            rank=deployment_rank(cfg, worker=True),
        ) == {"device_id": "2", "gpu_id": "2"}

    def test_env_used_when_config_absent(self):
        os.environ["LOCAL_RANK"] = "2"
        try:
            assert expand_placeholders("/mnt/nvme{LOCAL_RANK}") == "/mnt/nvme2"
        finally:
            del os.environ["LOCAL_RANK"]

    def test_no_placeholder_untouched(self):
        spec = {"root": "/mnt/nvme0", "x": ["a", 1]}
        assert expand_placeholders(spec) == spec

    def test_store_options_expanded_before_creation(self, monkeypatch):
        """engine 构造时 options 已按 LOCAL_RANK 展开。"""
        captured = {}

        import tutti.storage.metadata as metadata_mod

        def fake_create_store(type_name, options):
            captured["type"] = type_name
            captured["options"] = options
            from tutti.storage.memory import MemoryKVStore

            return MemoryKVStore(segment_bytes=4096, num_chunks=4)

        monkeypatch.setattr(metadata_mod, "create_metadata_store", fake_create_store)
        monkeypatch.setenv("LOCAL_RANK", "2")
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
                "chunk_tokens": 8,
                "chunk_kv_bytes": 12288,
                "max_chunks_per_wave": 4,
                "num_layers": 3,
                "store": {"type": "tutti_nvme", "options": {
                    "root": "/mnt/nvme{LOCAL_RANK}/kv-pool",
                }},
            }),
            cache_config=SimpleNamespace(block_size=16),
        )
        connector = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())
        connector.shutdown()
        assert captured["type"] == "tutti_nvme"
        assert captured["options"]["root"] == "/mnt/nvme2/kv-pool"
        # 调度侧没有 bind 阶段，层宽必须构造时注入：metadata store 的 scan() 在
        # layer_span 未声明时 fail-closed 返回空，漏传会让"复用已有池"的进程
        # 永远恢复不到驻留项（命中静默归零，曾实测 B 请求退化为全量重算）。
        assert captured["options"]["layer_span"] == 3

    def test_metadata_store_applies_layer_span(self, tmp_path):
        """层宽注入后成为对象几何的一半；缺省保持未声明（fail-closed）。"""
        from tutti.storage.metadata import TuttiMetadataStore

        store = TuttiMetadataStore(
            root=str(tmp_path / "pool"),
            num_chunks=4,
            segment_bytes=4096,
            layer_span=80,
        )
        assert store.layer_span == 80

        bare = TuttiMetadataStore(
            root=str(tmp_path / "bare"),
            num_chunks=4,
            segment_bytes=4096,
        )
        bare.open()
        assert bare.layer_span is None
        assert bare.scan() == []

    def test_rank_local_nvme_and_gpu_are_expanded_together(self, monkeypatch):
        """TP rank selects matching mount, daemon device, and CUDA device."""
        captured = {}
        import tutti.storage.metadata as metadata_mod

        def fake_create_store(type_name, options):
            captured.update(options)
            from tutti.storage.memory import MemoryKVStore
            return MemoryKVStore(segment_bytes=4096, num_chunks=4)

        monkeypatch.setattr(metadata_mod, "create_metadata_store", fake_create_store)
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
                "chunk_tokens": 8,
                "chunk_kv_bytes": 12288,
                "max_chunks_per_wave": 4,
                "num_layers": 3,
                "store": {"type": "tutti_nvme", "options": {
                    "root": "/mnt/nvme{LOCAL_RANK}/pool",
                    "preset": {
                        "device_id": "{LOCAL_RANK}",
                        "gpu_id": "{LOCAL_RANK}",
                    },
                }},
            }),
            cache_config=SimpleNamespace(block_size=16),
            parallel_config=SimpleNamespace(
                rank=3, tensor_parallel_size=4,
                decode_context_parallel_size=1,
            ),
        )
        connector = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())
        connector.shutdown()
        assert captured["root"] == "/mnt/nvme0/pool"
        assert captured["tp_size"] == 4
        assert [item["root"] for item in captured["rank_options"]] == [
            f"/mnt/nvme{rank}/pool" for rank in range(4)
        ]
        assert [item["preset"]["device_id"]
                for item in captured["rank_options"]] == [str(rank)
                                                          for rank in range(4)]
        assert [item["preset"]["gpu_id"]
                for item in captured["rank_options"]] == [str(rank)
                                                          for rank in range(4)]

    def test_device_groups_expand_per_rank_in_scheduler(self, monkeypatch):
        """scheduler 侧：8 卡 device_groups 按 rank 展开为各自盘组。"""
        captured = {}
        import tutti.storage.metadata as metadata_mod

        def fake_create_store(type_name, options):
            captured.update(options)
            from tutti.storage.memory import MemoryKVStore
            return MemoryKVStore(segment_bytes=4096, num_chunks=4)

        monkeypatch.setattr(metadata_mod, "create_metadata_store",
                            fake_create_store)
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
                "chunk_tokens": 8,
                "chunk_kv_bytes": 12288,
                "max_chunks_per_wave": 4,
                "num_layers": 3,
                "store": {"type": "tutti_nvme", "options": {
                    "root": "/mnt/nvme0/pool-rank{LOCAL_RANK}",
                    "preset": {
                        "type": "nvme-device-group",
                        "device_groups": [[0, 1], [2, 3]],
                        "gpu_id": "{LOCAL_RANK}",
                    },
                }},
            }),
            cache_config=SimpleNamespace(block_size=16),
            parallel_config=SimpleNamespace(
                rank=4, tensor_parallel_size=8,
                decode_context_parallel_size=1,
            ),
        )
        connector = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())
        connector.shutdown()
        rank_devices = [
            [device["device_id"] for device in item["preset"]["devices"]]
            for item in captured["rank_options"]
        ]
        assert rank_devices == [[0, 1]] * 4 + [[2, 3]] * 4
        # TP8 的 8 个 root 必须两两不同（scheduler 侧强校验）
        assert len({item["root"]
                    for item in captured["rank_options"]}) == 8
        # device_groups 不应泄漏进运行时 preset
        assert all("device_groups" not in item["preset"]
                   for item in captured["rank_options"])


@pytest.mark.parametrize("io_stream, queues", [("missing", "missing"), (None, 4), (17, 0)])
def test_worker_connector_applies_only_missing_io_defaults(monkeypatch, io_stream, queues):
    import tutti.engine.core as core
    import tutti.storage.registry as registry
    from tutti.integration.vllm.factory import worker_engine_for

    captured = {}
    options = {"capacity": "49152 B", "root": "/unused", "preset": {"type": "nvme-device-group"}}
    if io_stream != "missing":
        options["io_stream"] = io_stream
    if queues != "missing":
        options["preset"]["num_queues"] = queues
    extra = {"chunk_tokens": 8, "chunk_kv_bytes": 12288, "num_layers": 3,
             "store": {"type": "tutti_nvme", "options": options}}
    monkeypatch.setattr(registry, "create_store",
                        lambda _type, opts: captured.update(opts) or object())
    monkeypatch.setattr(core, "KVEngine", lambda config, store: SimpleNamespace(config=config))
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(rank=0, tensor_parallel_size=1))
    engine = worker_engine_for(cfg, extra)
    assert engine.config["max_chunks_per_wave"] == 512
    assert captured["io_stream"] == ("auto" if io_stream == "missing" else io_stream)
    assert captured["preset"]["num_queues"] == (8 if queues == "missing" else queues)
    assert ("io_stream" in options) == (io_stream != "missing")
    assert ("num_queues" in options["preset"]) == (queues != "missing")


@pytest.mark.parametrize("role", ["worker", "scheduler"])
def test_human_capacity_is_normalized_before_store_creation(monkeypatch, role):
    import tutti.engine.core as core
    import tutti.engine.metadata as metadata
    import tutti.storage.registry as registry
    import tutti.storage.metadata as stores_metadata
    from tutti.integration.vllm.factory import worker_engine_for, scheduler_index_for

    captured = {}
    options = {"root": "/unused-{LOCAL_RANK}", "capacity": "1 MiB"}
    extra = {"chunk_tokens": 8, "chunk_kv_bytes": 12288, "num_layers": 3,
             "store": {"type": "tutti_nvme", "options": options}}
    create = lambda _type, opts: captured.update(opts) or object()
    if role == "worker":
        monkeypatch.setattr(registry, "create_store", create)
        monkeypatch.setattr(core, "KVEngine", lambda config, store: object())
        factory = worker_engine_for
    else:
        monkeypatch.setattr(stores_metadata, "create_metadata_store", create)
        monkeypatch.setattr(metadata, "SchedulerMetadataIndex", lambda config, store: object())
        factory = scheduler_index_for
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(rank=0, tensor_parallel_size=2))
    factory(cfg, extra)
    assert captured["num_chunks"] == 1024**2 // 12288
    assert "capacity" not in captured and "capacity_bytes" not in captured
    if role == "scheduler":
        assert all(item["num_chunks"] == captured["num_chunks"]
                   and "capacity" not in item for item in captured["rank_options"])
    assert options == {"root": "/unused-{LOCAL_RANK}", "capacity": "1 MiB"}


@pytest.mark.parametrize("role", ["worker", "scheduler"])
@pytest.mark.parametrize("key", ["capacity_bytes", "num_chunks", "prp_cache_capacity", "handle_cache_capacity", "handle_cache_l2_capacity"])
def test_removed_capacity_is_rejected_on_cached_factory(monkeypatch, role, key):
    import tutti.engine.core as core
    import tutti.engine.metadata as metadata
    import tutti.storage.registry as registry
    import tutti.storage.metadata as stores_metadata
    from tutti.integration.vllm.factory import worker_engine_for, scheduler_index_for

    extra = {"chunk_tokens": 8, "chunk_kv_bytes": 12288, "num_layers": 3,
             "store": {"type": "memory", "options": {"capacity": "49152 B"}}}
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(rank=0, tensor_parallel_size=1))
    monkeypatch.setattr(registry, "create_store", lambda *_: object())
    monkeypatch.setattr(stores_metadata, "create_metadata_store", lambda *_: object())
    monkeypatch.setattr(core, "KVEngine", lambda *_: object())
    monkeypatch.setattr(metadata, "SchedulerMetadataIndex", lambda *_: object())
    factory = worker_engine_for if role == "worker" else scheduler_index_for
    factory(cfg, extra)
    extra["store"]["options"] = ({"capacity": "49152 B", "preset": {key: 4}}
                                 if "cache" in key else {key: 4})
    with pytest.raises(ValueError, match="不支持"):
        factory(cfg, extra)


class TestPresetNormalization:
    """preset 归一：纯十进制数字字符串转 int，其余原样。"""

    def test_digit_strings_become_int(self):
        preset = {
            "device_id": "2",
            "gpu_id": "2",
            "num_queues": "8",
            "pci_bdf": "0000:4b:00.0",
            "device": {"namespace_id": "1", "block_size": "4096",
                       "mount_path": "/mnt/nvme2"},
        }
        got = _normalize_preset(preset)
        assert got["device_id"] == 2
        assert got["gpu_id"] == 2
        assert got["num_queues"] == 8
        assert got["pci_bdf"] == "0000:4b:00.0"  # 非纯数字不动
        assert got["device"]["namespace_id"] == 1
        assert got["device"]["block_size"] == 4096
        assert got["device"]["mount_path"] == "/mnt/nvme2"

    def test_plain_values_untouched(self):
        preset = {"gpu_id": 0, "type": "local", "stripe_unit": 262144}
        assert _normalize_preset(preset) == preset

    def test_lists_recursed(self):
        assert _normalize_preset(["1", "abc", 2]) == [1, "abc", 2]


class TestTuttiStorePresetParam:
    """TuttiKVStore 的 preset 构造参数：dict 保存、归一后待 open 使用。"""

    def test_preset_stored_normalized(self, tmp_path):
        from tutti.storage.tutti_nvme.store import TuttiKVStore

        store = TuttiKVStore(
            tmp_path / "pool", num_chunks=4, segment_bytes=4096,
            preset={"type": "local", "device_id": "1", "gpu_id": "1"},
        )
        assert store._preset == {"type": "local", "device_id": 1, "gpu_id": 1}

    def test_no_preset_stays_none(self, tmp_path):
        from tutti.storage.tutti_nvme.store import TuttiKVStore

        store = TuttiKVStore(tmp_path / "pool", num_chunks=4, segment_bytes=4096)
        assert store._preset is None

    def test_preset_direct_overrides_env(self, tmp_path, monkeypatch):
        """preset dict 给出时 open 不再读 TUTTI_NVME_PRESET。"""
        from tutti.storage.tutti_nvme import store as store_mod
        from tutti.storage.tutti_nvme.store import TuttiKVStore

        store = TuttiKVStore(
            tmp_path / "pool", num_chunks=4, segment_bytes=4096,
            preset={"type": "local", "device_id": "1"},
        )
        monkeypatch.setenv("TUTTI_NVME_PRESET", "/nonexistent/invalid.yaml")
        called = {}

        def fake_build(preset):
            called["preset"] = preset
            raise RuntimeError("stop-here")

        monkeypatch.setattr(store_mod, "_build_runtime", fake_build)
        with pytest.raises(RuntimeError, match="stop-here"):
            store.open()
        assert called["preset"] == {"type": "local", "device_id": 1}  # 归一化后直达


def test_metadata_raid0_mounts_follow_each_rank(tmp_path, monkeypatch):
    from tutti.storage import metadata as metadata_mod
    import yaml

    config = tmp_path / "daemon.yaml"
    config.write_text(yaml.safe_dump({"nvmes": [
        {"device_id": index, "pci_addr": f"ffff:00:0{index}.0",
         "backing_mount_path": str(tmp_path / ("array-a" if index < 2 else "array-b"))}
        for index in range(4)
    ]}))
    captured = []

    class View:
        def __init__(self, options):
            captured.append(options)

        def open(self):
            pass

        def recover(self):
            return set()

        def close(self):
            pass

    monkeypatch.setattr(metadata_mod, "ObjectStore", View)
    ranks = [{"root": str(tmp_path / f"index-{rank}"), "preset": {
        "type": "raid0", "daemon_config": str(config),
        "devices": [{"device_id": index} for index in ([0, 1] if rank < 2 else [2, 3])],
    }} for rank in range(4)]
    store = metadata_mod.TuttiMetadataStore(ranks[0]["root"], 8, 4096,
                                            tp_size=4, rank_options=ranks, layer_span=3)
    store.open()
    assert [[item["mount_path"] for item in options["devices"]] for options in captured] == [
        [str(tmp_path / "array-a")], [str(tmp_path / "array-a")],
        [str(tmp_path / "array-b")], [str(tmp_path / "array-b")],
    ]
    assert [options["rank_id"] for options in captured] == list(range(4))
    store.close()


class TestDeviceGroups:
    """device_groups：多卡共用盘组（8 卡 / 每 4 个 rank 一组 2 盘条带）。"""

    def test_groups_split_by_rank_span(self):
        options = {"preset": {"type": "nvme-device-group",
                              "device_groups": [[0, 1], [2, 3]]}}
        for rank in range(4):
            got = apply_device_groups(options, rank=str(rank), tp_size=8)
            assert [d["device_id"] for d in got["preset"]["devices"]] == [0, 1]
        for rank in range(4, 8):
            got = apply_device_groups(options, rank=str(rank), tp_size=8)
            assert [d["device_id"] for d in got["preset"]["devices"]] == [2, 3]
        # 原 options 不被就地修改
        assert "device_groups" in options["preset"]
        assert "devices" not in options["preset"]

    def test_single_group_covers_all_ranks(self):
        options = {"preset": {"device_groups": [[0, 1]]}}
        got = apply_device_groups(options, rank="7", tp_size=8)
        assert [d["device_id"] for d in got["preset"]["devices"]] == [0, 1]

    def test_absent_groups_passthrough(self):
        options = {"preset": {"type": "local", "device_id": "0"}}
        assert apply_device_groups(options, rank="0", tp_size=8) is options

    def test_group_count_must_divide_tp_size(self):
        options = {"preset": {"device_groups": [[0], [1], [2]]}}
        with pytest.raises(ValueError, match="整除"):
            apply_device_groups(options, rank="0", tp_size=8)

    def test_empty_groups_rejected(self):
        options = {"preset": {"device_groups": []}}
        with pytest.raises(ValueError, match="非空"):
            apply_device_groups(options, rank="0", tp_size=8)


class TestCapacityToChunks:
    """每rank的capacity按payload几何转成内部槽数，不扣持久格式开销。"""

    # HY3 几何：80 层 × 128 KiB = 10 MiB/槽（一个文件）。
    SEGMENT_BYTES = 131072
    NUM_LAYERS = 80
    TI_BYTES = 1024 ** 4

    @staticmethod
    def _base_options():
        return {
            "preset": {
                "type": "nvme-device-group",
                "gpu_id": "{LOCAL_RANK}",
                "device_groups": [[0, 1], [2, 3]],
            },
        }

    def _convert(self, options):
        return apply_capacity(
            options, segment_bytes=self.SEGMENT_BYTES,
            num_layers=self.NUM_LAYERS,
        )

    def test_four_tib_fills_four_disks(self):
        """4 TiB → 419430 槽（每槽一个 10 MiB 文件，误差 < 一个槽位）。"""
        options = self._base_options()
        options["capacity"] = "4 TiB"
        got = self._convert(options)
        assert got["num_chunks"] == 419430
        used = got["num_chunks"] * 10 * 1024 * 1024
        assert used <= 4 * self.TI_BYTES
        assert 4 * self.TI_BYTES - used < 10 * 1024 * 1024

    def test_absent_key_leaves_options_untouched(self):
        got = self._convert(self._base_options())
        assert "num_chunks" not in got
        assert "capacity_bytes" not in got

    @pytest.mark.parametrize("key", ["capacity_bytes", "num_chunks"])
    @pytest.mark.parametrize("with_capacity", [False, True])
    def test_removed_capacity_forms_are_rejected(self, key, with_capacity):
        options = self._base_options()
        options[key] = 8
        if with_capacity:
            options["capacity"] = "1 TiB"
        with pytest.raises(ValueError, match="不支持"):
            self._convert(options)

    def test_rejects_non_positive(self):
        options = self._base_options()
        options["capacity"] = "0 B"
        with pytest.raises(ValueError, match="正整数字节"):
            self._convert(options)

    def test_rejects_capacity_smaller_than_one_slot(self):
        options = self._base_options()
        options["capacity"] = "1024 B"
        with pytest.raises(ValueError, match="小于单槽位"):
            self._convert(options)

    def test_capacity_does_not_depend_on_disk_count(self):
        """槽数仅由层payload乘积得出，不依赖盘组数量。"""
        got = self._convert({"capacity": "1 TiB"})
        assert got["num_chunks"] == (
            self.TI_BYTES // (self.SEGMENT_BYTES * self.NUM_LAYERS)
        )

    def test_single_disk_uses_whole_payload_per_slot(self):
        """单盘部署：每槽就是一个文件（10 MiB），与多盘同式。"""
        options = {
            "capacity": "1 TiB",
            "preset": {"devices": [{"device_id": 0}]},
        }
        got = self._convert(options)
        assert got["num_chunks"] == (
            self.TI_BYTES // (self.SEGMENT_BYTES * self.NUM_LAYERS)
        )
