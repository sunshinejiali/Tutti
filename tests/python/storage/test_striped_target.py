"""多盘段文件布局契约：对象层 + 文件 resolver 替身。

替身按 C++ local-file resolver 的同一语义（``file://<绝对路径>``，路径即
文件）把每个请求落到真实文件上，因此这些用例覆盖真实文件读写，却不需要
daemon 或 NVMe 硬件。

放置在对象层（C++）：槽位号在 mounts 间轮转（slot % N），每盘连续
``segment_file_slots`` 个槽位打包进 ``<mount>/r<rank_id>/segments/<id>.seg``，
槽位 i 的段 0 从 ``i × slot_bytes + header`` 开始。
"""

from __future__ import annotations

import ctypes
import os
from collections import namedtuple
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from tutti.storage.tutti_nvme.object_layout import ObjectLayout
from tutti.storage.tutti_nvme.store import TuttiKVStore


SEGMENT = 16 * 1024
MOUNTS = 3
HEADER = 32 * 1024
SLOTS_PER_FILE = 4
SPAN = 2


def _key(value, layer=0):
    return bytes([value]) * 16 + layer.to_bytes(2, "little")


SubmitResult = namedtuple(
    "SubmitResult", "status_ok status_msg io_handle initial_states rejected"
)


class RotatingFakeRuntime:
    """Host fake that maps every request onto the real slot file."""

    def __init__(self):
        self._next = 0
        self._targets = {}
        self._memories = {}

    def caps(self):
        return {"target": ["stub"], "memory": ["host"]}

    def open_batch(self, uris):
        tickets = []
        for uri in uris:
            assert uri.startswith("file://")
            path = unquote(urlsplit(uri).netloc + urlsplit(uri).path)
            self._next += 1
            self._targets[self._next] = path
            tickets.append(self._next)
        return tickets

    def close_target(self, ticket):
        self._targets.pop(ticket, None)

    def close_batch(self, tickets):
        for ticket in tickets:
            self.close_target(ticket)

    def register_memory(self, addr, size, kind, accel_id=-1, io_granularity=0):
        self._next += 1
        self._memories[self._next] = (addr, size)
        return self._next

    def submit(self, requests, **_kwargs):
        for request in requests:
            self._execute(request)
        self._next += 1
        return SubmitResult(True, "", self._next, [], [])

    def wait(self, _handle, _timeout_ms=0):
        return "OK", "COMPLETED"

    def release_io(self, _handle):
        return None

    def shutdown(self, _timeout_ms):
        return None

    def _execute(self, request):
        target, target_offset, memory, memory_offset, length, direction = request
        path = self._targets[target]
        addr, _ = self._memories[memory]
        fd = os.open(path, os.O_RDWR)
        try:
            if direction == "write":
                data = ctypes.string_at(addr + memory_offset, length)
                os.pwrite(fd, data, target_offset)
            else:
                data = os.pread(fd, length, target_offset)
                ctypes.memmove(addr + memory_offset, data, len(data))
        finally:
            os.close(fd)


def _object_layout(tmp_path, *, mounts, layers=SPAN, segment=SEGMENT,
                   capacity=8, rank_id=0, rank_count=1):
    root = tmp_path / "meta-root"
    root.mkdir(parents=True, exist_ok=True)
    for mount in mounts:
        Path(mount).mkdir(parents=True, exist_ok=True)
    layout = ObjectLayout(
        root, segment, mounts=[str(mount) for mount in mounts],
        capacity_chunks=capacity,
        rank_id=rank_id, rank_count=rank_count,
        namespace=b"rotating-contract",
        segment_file_slots=SLOTS_PER_FILE, segment_header_bytes=HEADER,
    )
    layout.set_layer_span(layers)
    return layout


def _uri_path(uri):
    parsed = urlsplit(uri)
    return Path(unquote(parsed.netloc + parsed.path))


def test_rotating_store_roundtrip_and_drop(tmp_path):
    """多盘 store 端到端：写读回、按对象提交、drop 使整个对象失效。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    store = TuttiKVStore(
        tmp_path / "meta-root", 2, SEGMENT, runtime=RotatingFakeRuntime(),
        mounts=mounts, segment_file_slots=SLOTS_PER_FILE,
    )
    store.open()
    store.set_layer_span(SPAN)
    source = bytearray(SEGMENT * SPAN)
    source[:SEGMENT] = bytes(range(256)) * (SEGMENT // 256)
    source[SEGMENT:] = b"z" * SEGMENT
    source_id = store.register_buffer(source, SEGMENT)
    key0, key1 = _key(7, 0), _key(7, 1)
    store.put_batch([(key0, source_id, 0), (key1, source_id, SEGMENT)]).wait()

    assert store.scan() == sorted([key0, key1])
    destination = bytearray(SEGMENT * SPAN)
    destination_id = store.register_buffer(destination, SEGMENT)
    store.get_batch(
        [(key0, destination_id, 0), (key1, destination_id, SEGMENT)]
    ).wait()
    assert destination == source

    # drop 命中对象任一 key ⇒ 整个对象（各层）一起失效
    store.drop([key0])
    assert not store.has(key0) and not store.has(key1)
    assert store.scan() == []
    store.close()


def test_rotating_uri_and_segment_files(tmp_path):
    """URI 是 file://<mount>/r<rank>/segments/<id>.seg；段文件整组预建。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(MOUNTS)]
    layout = _object_layout(tmp_path, mounts=mounts, layers=MOUNTS, capacity=16)
    slot_bytes = HEADER + MOUNTS * SEGMENT
    chunks = [_key(value)[:16] for value in range(MOUNTS + 1)]
    admitted, rejected = layout.prepare_put(
        [chunk + (1).to_bytes(2, "little") for chunk in chunks], capacity_chunks=16
    )
    assert len(admitted) == len(chunks) and rejected == 0

    # 槽位 0..2 轮转到三块盘的 0.seg 开头；槽位 3 回到盘 0、同文件第二个槽位。
    for slot, chunk in enumerate(chunks):
        path = _uri_path(layout.target_uri(chunk))
        assert path == mounts[slot % MOUNTS] / "r0" / "segments" / "0.seg"
        assert layout.target_offset(chunk) == (slot // MOUNTS) * slot_bytes + HEADER
        assert layout.target_size(chunk) == MOUNTS * SEGMENT
    for mount in mounts:
        seg = mount / "r0" / "segments" / "0.seg"
        assert seg.stat().st_size == SLOTS_PER_FILE * slot_bytes
    layout.close_object_pool()


def test_rotating_slots_spread_over_mounts(tmp_path):
    """槽位号在 mounts 间轮转：连续槽位把请求摊到每块盘。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    layout = _object_layout(tmp_path, mounts=mounts, capacity=8)
    per_mount = {0: 0, 1: 0}
    for value in range(8):
        layout.prepare_put([_key(value, 0)], capacity_chunks=8)
        uri = layout.target_uri(_key(value)[:16])
        path = Path(unquote(urlsplit(uri).netloc + urlsplit(uri).path))
        mount_index = next(
            i for i, mount in enumerate(mounts)
            if str(path).startswith(str(mount))
        )
        per_mount[mount_index] += 1
    # 8 槽 × 2 盘 = 每盘 4（首个文件组恰好 8 槽）：轮转必须均匀。
    assert per_mount == {0: 4, 1: 4}
    layout.close_object_pool()


def test_rotating_slot_identity_is_stable(tmp_path):
    """同一 chunk 的槽位 URI 恒定；回收后槽位可被新 chunk 复用。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    layout = _object_layout(tmp_path, mounts=mounts, capacity=2)
    first = _key(3)[:16]
    layout.prepare_put([_key(3, 0), _key(3, 1)], capacity_chunks=2)
    uri = layout.target_uri(first)
    # 重复预留（同一步重放）不改 URI
    layout.prepare_put([_key(3, 0)], capacity_chunks=2)
    assert layout.target_uri(first) == uri

    layout.commit_layers([_key(3, 0), _key(3, 1)])
    assert layout.is_committed(first)
    layout.release_chunks([first])
    assert not layout.is_committed(first)
    assert layout.committed_chunks() == set()

    # 回收后的槽位可再次分配并完成提交
    second = _key(4)[:16]
    admitted, rejected = layout.prepare_put(
        [_key(4, 0), _key(4, 1)], capacity_chunks=2
    )
    assert len(admitted) == 2 and rejected == 0
    layout.commit_layers([_key(4, 0), _key(4, 1)])
    assert layout.is_committed(second)
    layout.close_object_pool()


def test_rotating_capacity_trims_unadmitted_chunks(tmp_path):
    """容量耗尽按对象层契约裁剪：被拒 chunk 没有槽位、也不会被提交。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    layout = _object_layout(tmp_path, mounts=mounts, capacity=1)
    admitted, rejected = layout.prepare_put(
        [_key(9, 0), _key(10, 0)], capacity_chunks=1
    )
    assert len(admitted) == 1 and rejected == 1
    kept = next(iter(admitted.values()))[0]
    dropped = _key(10)[:16] if kept == _key(9)[:16] else _key(9)[:16]
    with pytest.raises(KeyError):
        layout.target_uri(dropped)
    assert not layout.is_committed(dropped)
    layout.close_object_pool()


def test_rotating_rank_slots_are_isolated(tmp_path):
    """多 rank 共用同一组盘：槽位文件按 rank 子目录隔离。

    否则不同 rank 会写同名文件（互相覆盖），槽位校验失败且数据互相污染
    ——8 卡共享一组盘时的实测故障。分片文件在 ``<mount>/r<rank_id>/`` 下，
    这里把隔离钉住。
    """
    mounts = [tmp_path / "nvme0", tmp_path / "nvme1"]
    rank0 = _object_layout(tmp_path / "rank0", mounts=mounts, rank_id=0)
    rank4 = _object_layout(tmp_path / "rank4", mounts=mounts, rank_id=4)
    rank0.prepare_put([_key(7, 0)], capacity_chunks=8)
    rank4.prepare_put([_key(7, 0)], capacity_chunks=8)
    path0 = Path(unquote(urlsplit(
        rank0.target_uri(_key(7)[:16])).netloc +
        urlsplit(rank0.target_uri(_key(7)[:16])).path))
    path4 = Path(unquote(urlsplit(
        rank4.target_uri(_key(7)[:16])).netloc +
        urlsplit(rank4.target_uri(_key(7)[:16])).path))
    # 同一槽位号、同一组盘，但 rank 子目录不同 ⇒ 互不覆盖
    assert path0.parent != path4.parent
    assert path0.parent.parent.name == "r0"
    assert path4.parent.parent.name == "r4"
    assert path0.exists() and path4.exists()
    assert path0 != path4
    rank0.close_object_pool()
    rank4.close_object_pool()


def test_invalid_segment_options(tmp_path):
    """段文件几何非法时构造即拒绝；旧布局选择开关已不存在。"""
    for kwargs in ({"segment_file_slots": 0}, {"segment_header_bytes": 1000},
                   {"segment_header_bytes": 0}):
        with pytest.raises(ValueError):
            TuttiKVStore(tmp_path, 1, SEGMENT, runtime=RotatingFakeRuntime(), **kwargs)
    with pytest.raises(TypeError):
        TuttiKVStore(tmp_path, 1, SEGMENT, runtime=RotatingFakeRuntime(),
                     layout="striped")
