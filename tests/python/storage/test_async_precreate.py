"""段文件预建契约：单位是文件组，写路径永不建文件。

定案（2026-09-30）：
  1. 文件组 = 每盘一个段文件（``segment_file_slots × 盘数`` 个槽位），对象层
     只整组发布；
  2. 冷池 open() 同步建出首个文件组，开服即可写；热池 open() 只收编已就绪的
     组，一个字节都不写；
  3. 写路径拿不到已发布槽位时与容量耗尽同一契约——非阻塞拒绝、调用方裁剪；
  4. 后续文件组由后台线程在最后一个已建组用过一半时建，不越过容量上限；
     失败不发布任何槽位，稍后重试。
"""

from __future__ import annotations

import logging
import os
import time

import pytest

pytest.importorskip("tutti_runtime._core")

from tutti.storage.tutti_nvme.object_layout import ObjectLayout

SPAN = 2
SEGMENT = 4096
SLOTS_PER_FILE = 8


@pytest.fixture(autouse=True)
def _neutralize_space_guard(monkeypatch):
    """把磁盘空间护栏门限归零（护栏行为由专门的用例显式覆盖）。"""
    monkeypatch.setenv("TUTTI_PRECREATE_MIN_FREE_BYTES", "0")


def _layout(tmp_path, *, capacity=32, mounts=1):
    root = tmp_path / "ns"
    root.mkdir(parents=True, exist_ok=True)
    dirs = []
    for index in range(mounts):
        mount = tmp_path / f"dev{index}"
        mount.mkdir(parents=True, exist_ok=True)
        dirs.append(str(mount))
    layout = ObjectLayout(
        root,
        SEGMENT,
        mounts=dirs,
        capacity_chunks=capacity,
        background_reclaim=False,
        namespace=b"test-async-precreate",
        segment_file_slots=SLOTS_PER_FILE,
        segment_header_bytes=32 * 1024,
    )
    layout.set_layer_span(SPAN)
    return layout


def _keys(*chunks: bytes) -> list[bytes]:
    return [
        chunk + layer.to_bytes(2, "little")
        for chunk in chunks
        for layer in range(SPAN)
    ]


def _chunk(index: int) -> bytes:
    return index.to_bytes(16, "little")


def _path(uri: str) -> str:
    return uri[len("file://"):] if uri.startswith("file://") else uri


def _segments(tmp_path, mount=0) -> list[str]:
    seg_dir = tmp_path / f"dev{mount}" / "r0" / "segments"
    if not seg_dir.exists():
        return []
    return sorted(name for name in os.listdir(seg_dir) if name.endswith(".seg"))


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_cold_open_builds_exactly_the_first_group(tmp_path):
    """冷池：open 同步建首个文件组（每盘一个文件），不多建。"""
    layout = _layout(tmp_path, capacity=64, mounts=2)
    store = layout._store
    assert store.precreated_slots() == 2 * SLOTS_PER_FILE
    for mount in (0, 1):
        assert _segments(tmp_path, mount) == ["0.seg"]
        assert (tmp_path / f"dev{mount}" / "r0" / "segments" / "0.seg.ready").exists()
    # 开服即可写：首个组内的写全部受理。
    chunks = [_chunk(i) for i in range(2 * SLOTS_PER_FILE)]
    admitted, rejected = layout.prepare_put(_keys(*chunks), capacity_chunks=64)
    assert rejected == 0 and len(admitted) == len(chunks) * SPAN


def test_warm_open_adopts_groups_without_writing(tmp_path):
    """热池：重开收编已就绪的组，不重写任何段文件。"""
    first = _layout(tmp_path)
    first.prepare_put(_keys(_chunk(0)), capacity_chunks=32)
    assert first._store.precreate_step(SLOTS_PER_FILE) == SLOTS_PER_FILE
    first.close_object_pool()
    seg_dir = tmp_path / "dev0" / "r0" / "segments"
    before = {name: os.stat(seg_dir / name).st_mtime_ns for name in os.listdir(seg_dir)}

    reopened = _layout(tmp_path)
    assert reopened._store.precreated_slots() == 2 * SLOTS_PER_FILE
    after = {name: os.stat(seg_dir / name).st_mtime_ns for name in os.listdir(seg_dir)}
    assert before == after, "复用路径不得重写段文件"


def test_write_path_refuses_beyond_published_groups(tmp_path):
    """越过已发布组的写立即拒绝——写路径绝不自己建文件。"""
    layout = _layout(tmp_path)
    chunks = [_chunk(i) for i in range(SLOTS_PER_FILE + 1)]
    started = time.monotonic()
    admitted, rejected = layout.prepare_put(_keys(*chunks), capacity_chunks=32)
    elapsed = time.monotonic() - started
    assert rejected == 1 and len(admitted) == SLOTS_PER_FILE * SPAN
    assert elapsed < 0.05, f"拒绝耗时 {elapsed * 1000:.1f}ms，写路径被预建阻塞了"
    assert _segments(tmp_path) == ["0.seg"], "写路径不得偷偷建文件"


HALF = SLOTS_PER_FILE // 2


def _use(layout, count, start=0):
    chunks = [_chunk(start + i) for i in range(count)]
    admitted, rejected = layout.prepare_put(_keys(*chunks), capacity_chunks=64)
    assert rejected == 0 and len(admitted) == count * SPAN


def test_background_grows_when_group_is_half_used(tmp_path, caplog):
    """后台线程在最后一组用过一半时建下一组，只领先一个组。"""
    caplog.set_level(logging.INFO)
    layout = _layout(tmp_path, capacity=64)
    store = layout._store
    layout.start_background_precreate()
    # 用到一半（含）之前：余量仍够，不建（建组与在线 KV 争带宽）。
    _use(layout, HALF)
    time.sleep(0.6)
    assert store.precreated_slots() == SLOTS_PER_FILE
    assert _segments(tmp_path) == ["0.seg"]
    # 越过一半 → 建第 2 组，然后停下。
    _use(layout, 1, start=HALF)
    assert _wait_for(lambda: store.precreated_slots() == 2 * SLOTS_PER_FILE)
    time.sleep(0.6)
    assert store.precreated_slots() == 2 * SLOTS_PER_FILE
    assert _segments(tmp_path) == ["0.seg", "1.seg"]
    assert any("BACKGROUND_PRECREATE_GROUP" in r.message for r in caplog.records)
    layout.stop_background_precreate()


def test_growth_never_exceeds_capacity(tmp_path):
    """容量不是文件组整数倍时，最后一组截到容量，不越界。"""
    layout = _layout(tmp_path, capacity=SLOTS_PER_FILE + 3)
    store = layout._store
    layout.start_background_precreate()
    _use(layout, SLOTS_PER_FILE)
    assert _wait_for(lambda: store.precreated_slots() == SLOTS_PER_FILE + 3)
    time.sleep(0.6)
    assert store.precreated_slots() == SLOTS_PER_FILE + 3
    assert _segments(tmp_path) == ["0.seg", "1.seg"]
    layout.stop_background_precreate()


def test_start_is_idempotent(tmp_path):
    layout = _layout(tmp_path)
    layout.start_background_precreate()
    layout.start_background_precreate()
    assert layout._precreate_stop is not None
    layout.stop_background_precreate()


def test_space_guard_pauses_growth_and_resumes(tmp_path, caplog, monkeypatch):
    """磁盘空间护栏：可用空间低于门限时暂停预建，恢复后自动继续。"""
    layout = _layout(tmp_path, capacity=64)
    store = layout._store
    _use(layout, HALF + 1)
    monkeypatch.setenv("TUTTI_PRECREATE_MIN_FREE_BYTES", str(10 ** 18))
    layout.start_background_precreate()
    time.sleep(0.6)
    assert store.precreated_slots() == SLOTS_PER_FILE, "空间不足时不得预建"
    assert any("BACKGROUND_PRECREATE_SPACE_LOW" in r.message
               for r in caplog.records), "空间不足必须留下告警"
    assert not any("BACKGROUND_PRECREATE_FAILED" in r.message
                   for r in caplog.records), "护栏不是失败：线程必须活着"
    monkeypatch.setenv("TUTTI_PRECREATE_MIN_FREE_BYTES", "0")
    assert _wait_for(lambda: store.precreated_slots() == 2 * SLOTS_PER_FILE,
                     timeout=15.0)
    layout.stop_background_precreate()


def test_grower_failure_is_retried(tmp_path, caplog, monkeypatch):
    """后台建组失败不发布任何槽位，线程存活并在之后重试成功。"""
    monkeypatch.setattr(
        "tutti.storage.tutti_nvme.object_layout._PRECREATE_SPACE_RECHECK_S", 0.05
    )
    layout = _layout(tmp_path, capacity=64)
    store = layout._store
    real_step = store.precreate_step
    calls = {"n": 0}

    def flaky(headroom):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected grower failure")
        return real_step(headroom)

    store.precreate_step = flaky
    _use(layout, HALF + 1)
    layout.start_background_precreate()
    assert _wait_for(lambda: store.precreated_slots() == 2 * SLOTS_PER_FILE)
    assert calls["n"] >= 2
    assert any("BACKGROUND_PRECREATE_FAILED" in r.message for r in caplog.records)
    layout.stop_background_precreate()
