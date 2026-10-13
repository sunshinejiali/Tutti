"""对象层支撑的 KV 布局：Python 侧不做任何文件系统操作。

`Layout` / `StripedLayout` 的公开面（store.py 与调度侧只依赖这些方法）在这里
全部由 C++ ``StorageObjectStore`` 承担：

    槽位分配  → reserve()          对象有效性 → commit()（每对象一次）
    槽位路径  → placement.uri      崩溃恢复   → recover()
    容量      → usage()

与旧实现的关键语义差异（评审定案）：

* **按对象提交，不按层**。一个 chunk 的全部层段共用一个对象，``commit`` 只写
  一次对象头；某层没写完在结构上就等于"整个对象未提交"，读侧永远看不到半截
  chunk。Python 侧只记 ``{chunk: 已写层集合}``，齐了才调一次 commit。
* **固定大段文件**：槽位号在挂载点间轮转（slot % N），每盘上连续
  ``segment_file_slots`` 个槽位打包进一个段文件
  ``<mount>/r<rank>/segments/<id>.seg``，槽位前缀（对象头）
  ``segment_header_bytes``。路径不含 chunk 身份，票据按文件长驻，分配与
  回收都不需要 rename。
* **零 marker 文件**。旧实现的 ``meta/<chunk>.<layer>.ok`` 与 manifest JSON 全
  部消失，冷启动驻留集合来自 ``recover()``。

层宽（``num_layers`` → ``segment_count``）在建布局时未知，因此对象的 open 推迟
到 :meth:`set_layer_span`（引擎在此之后立刻 ``_deferred_restore``，见
``engine/core.py``）；在那之前的 :meth:`scan` 返回空集。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

# 挂在 tutti 树下的专属通道（connector 会为该树挂 handler 放行 INFO）。
_PRECREATE_LOG = logging.getLogger("tutti.precreate")

# 段文件几何默认值（HY3 TP8：2048 槽 × ~10 MiB ≈ 20 GiB/文件）。槽位前缀
# 128 KiB：payload 起点至少 16 KiB 对齐（实测 4 KiB 前缀带宽低 20%），且每层
# 段起点 128 KiB 对齐——md RAID0 用 128 KiB 条带时单条 IO（≤128 KiB）永不
# 跨条带（32 KiB 前缀时每条 128 KiB IO 都被拆成 96K+32K，单卡 12.6 vs 14.2 GiB/s）。
DEFAULT_SEGMENT_FILE_SLOTS = 2048
DEFAULT_SEGMENT_HEADER_BYTES = 128 * 1024

# 后台预建没有工作时的轮询间隔。
_PRECREATE_IDLE_SLEEP_S = 0.5

# 磁盘空间护栏：预建线程在把盘写满之前必须停手。故障形态（2026-09-22 事故）：
# 容量配置超过物理盘（8TiB/rank × 8 rank = 64TiB 需求 > 4×5.8TB 盘），预建
# 线程一路补到 ENOSPC。护栏把增长压回"可用空间之内"：空间不足时暂停增长
# （写路径继续按非阻塞裁剪契约拒绝缺槽写），空间恢复后自动继续。
# 门限可用 TUTTI_PRECREATE_MIN_FREE_BYTES 覆盖（测试用）。
_PRECREATE_MIN_FREE_BYTES_ENV = "TUTTI_PRECREATE_MIN_FREE_BYTES"
_PRECREATE_DEFAULT_MIN_FREE_BYTES = 32 * 1024 ** 3  # 32 GiB
_PRECREATE_SPACE_RECHECK_S = 5.0

from tutti.index.chunk_index import decode_io_key as _decode
from tutti.storage.object_store import (
    ObjectPlacement,
    ObjectStore,
    SCHEME_LOCAL_NVME_FILE,
    SCHEME_STRIPED_NVME_FILE,
)

__all__ = [
    "DEFAULT_SEGMENT_FILE_SLOTS",
    "DEFAULT_SEGMENT_HEADER_BYTES",
    "ObjectLayout",
    "layout_fingerprint",
]


def layout_fingerprint(namespace, segment_file_slots, segment_header_bytes):
    """命名空间 + 段文件几何：几何变了就是另一个池，对象层 fail-closed。"""
    tag = (b"segment-files-rotating-v4-slots:" +
           str(int(segment_file_slots)).encode() + b"-header:" +
           str(int(segment_header_bytes)).encode())
    return (namespace or b"") + b"\0" + tag


class ObjectLayout:
    """由对象层驱动的布局。

    ``mounts`` 是数据盘目录（缺省为 root 本身），槽位在其间轮转、打包进
    固定大段文件。``devices`` 是给 runtime resolver 用的设备事实
    （controller/namespace），对象层只用其中的 ``mount_path``。
    """

    def __init__(
        self,
        root: str | os.PathLike,
        segment_bytes: int,
        *,
        mounts=None,
        devices=None,
        capacity_chunks: int = 0,
        rank_id: int = 0,
        rank_count: int = 1,
        namespace: bytes | str | None = None,
        segment_file_slots: int = DEFAULT_SEGMENT_FILE_SLOTS,
        segment_header_bytes: int = DEFAULT_SEGMENT_HEADER_BYTES,
    ):
        self._root = Path(root)
        self._segment_bytes = int(segment_bytes)
        self._segment_file_slots = int(segment_file_slots)
        self._segment_header_bytes = int(segment_header_bytes)
        self._mounts = [str(m) for m in (mounts if mounts else [self._root])]
        self._devices = list(devices or [])
        # 容量按 chunk 计（对象层按槽位几何折算为字节）。
        self._capacity_chunks = int(capacity_chunks or 0)
        self._rank_id = int(rank_id)
        self._rank_count = int(rank_count)
        # 后台预建线程的停止信号（None = 未启动），见 start_background_precreate。
        self._precreate_stop: threading.Event | None = None
        self._namespace = (
            namespace.encode("utf-8") if isinstance(namespace, str) else namespace
        )

        self._layer_span: int | None = None
        self._store: ObjectStore | None = None
        self._committed: set[bytes] = set()
        self._reserved: dict[bytes, ObjectPlacement] = {}
        self._written_layers: dict[bytes, set[int]] = {}

    # ---------- 几何 / 生命周期 ----------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def mounts(self) -> tuple[str, ...]:
        """数据盘挂载点（未配置时即 root 本身）。"""
        return tuple(self._mounts)

    @property
    def segment_bytes(self) -> int:
        return self._segment_bytes

    @property
    def layer_span(self) -> int | None:
        return self._layer_span

    def set_namespace(self, namespace) -> None:
        """声明 key 命名空间（打开前生效）；不一致时对象层 fail-closed。"""
        if self._store is not None:
            raise RuntimeError("命名空间须在打开对象层之前声明")
        self._namespace = (
            namespace.encode("utf-8") if isinstance(namespace, str) else namespace
        )

    def set_layer_span(self, num_layers: int) -> None:
        """定层宽并打开对象层（此后才可能做冷启动恢复）。"""
        num_layers = int(num_layers)
        if num_layers <= 0:
            raise ValueError(f"层数须为正整数，got {num_layers}")
        if self._layer_span is not None and self._layer_span != num_layers:
            raise ValueError(
                f"层宽已定案 {self._layer_span}，不能改为 {num_layers}"
            )
        self._layer_span = num_layers
        if self._store is None:
            self._open_store()

    def _config(self) -> dict:
        assert self._layer_span is not None
        devices = []
        if self._devices:
            for index, device in enumerate(self._devices):
                entry = dict(device)
                entry.setdefault(
                    "mount_path",
                    self._mounts[index] if index < len(self._mounts) else self._mounts[0],
                )
                devices.append(entry)
        else:
            devices = [{"mount_path": mount} for mount in self._mounts]
        # 段文件几何是介质命名空间的一部分：换几何必须换池。
        fingerprint = layout_fingerprint(
            self._namespace, self._segment_file_slots, self._segment_header_bytes,
        )
        return {
            "scheme": (
                SCHEME_STRIPED_NVME_FILE
                if len(self._mounts) > 1 else SCHEME_LOCAL_NVME_FILE
            ),
            "uri": str(self._root),
            # 容量按槽位声明：每槽位字节数由对象层按几何算。
            "capacity_slots": self._capacity_chunks,
            "segment_bytes": self._segment_bytes,
            "segment_count": self._layer_span,
            "segment_file_slots": self._segment_file_slots,
            "segment_header_bytes": self._segment_header_bytes,
            "namespace_fingerprint": fingerprint,
            "devices": devices,
            "rank_id": self._rank_id,
            "rank_count": self._rank_count,
        }

    def _open_store(self) -> None:
        store = ObjectStore(self._config())
        # 命名空间/几何不一致时对象层 fail-closed（不覆盖旧数据），异常直接上抛：
        # 异构池必须由人处理，不能被当成"空池"静默复用。冷池时 open 同步建出
        # 首个文件组（每盘一个段文件，HY3 TP8 约 50s），开服即可写。
        store.open()
        self._store = store
        self._committed = store.recover()

    def close_object_pool(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
        self._committed.clear()
        self._reserved.clear()
        self._written_layers.clear()

    # ---------- 恢复 ----------

    def ready_slots(self) -> list[tuple[str, int]]:
        """已物化槽位的 ``(uri, generation)`` 列表（绑定期预热用）。

        冷池与热池都非空：对象层 open() 至少建好首个文件组。绑定期拿这些 URI
        先把运行时票据开好，请求路径就只剩内存查找——首轮 resolve +
        peer-memory 注册的几百毫秒因此移出请求路径。
        """
        store = self._store
        if store is None:
            return []
        out: list[tuple[str, int]] = []
        for slot in range(store.ready_slots()):
            uri = store.slot_uri(slot)
            if uri:
                out.append((uri, store.slot_generation(slot)))
        return out

    @property
    def slot_payload_bytes(self) -> int:
        """每个对象的 payload 字节数（各槽位一致，用于预热票据的尺寸校验）。"""
        return self._segment_bytes * int(self._layer_span or 0)

    # ---------- 后台预建（大容量冷启动与按需增长都不在关键路径上）----------

    def _mounts_free_bytes(self) -> int | None:
        """所有数据盘挂载点的可用空间合计；任一 statvfs 失败返回 None。

        None 表示护栏失效（调用方按"不拦"处理）：护栏是防呆，不是正确性前提，
        探测不到空间时不该把预建卡死。
        """
        total = 0
        for mount in self._mounts:
            try:
                st = os.statvfs(mount)
            except OSError:
                return None
            total += st.f_bavail * st.f_frsize
        return total

    def _min_free_bytes(self) -> int:
        """护栏门限：默认 min(32 GiB, 挂载点总容量的 5%)。

        绝对门限保护大盘（32 GiB ≈ 5.8 TB 盘的 0.55%，足够写路径回旋）；
        百分比上限保护小盘与测试环境（20 GiB 的盘不该因为凑不够 32 GiB 而
        永远不能预建）。``TUTTI_PRECREATE_MIN_FREE_BYTES`` 直接覆盖门限
        （测试用；设 0 表示关闭护栏）。
        """
        raw = os.environ.get(_PRECREATE_MIN_FREE_BYTES_ENV)
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
        total = 0
        for mount in self._mounts:
            try:
                st = os.statvfs(mount)
            except OSError:
                total = 0
                break
            total += st.f_blocks * st.f_frsize
        if total <= 0:
            return _PRECREATE_DEFAULT_MIN_FREE_BYTES
        return min(_PRECREATE_DEFAULT_MIN_FREE_BYTES, total * 5 // 100)

    def start_background_precreate(self) -> None:
        """后台线程：最后一个已建文件组用过一半时，建下一个文件组。

        单位是**文件组**（每盘一个段文件 = ``segment_file_slots × 盘数`` 槽位），
        对象层只整组发布，写路径永不建文件：前沿追上已建末尾时 reserve 非阻塞
        拒绝、调用方裁剪，后台建好下一组后自动恢复。就绪余量取半个文件组：
        建一组是重写入（HY3 TP8 每盘 8 rank × 20 GiB，约 50s），与在线 KV 争盘
        带宽，所以不在组刚开始用时就建；剩一半（每 rank 4096 槽 ≈ 40 GiB KV）
        足以覆盖建组时间。容量是上限，不是要立即铺满的目标。

        一个线程足够：对象层同一时刻只建一个组，同盘文件创建还有挂载点锁串行。
        失败（如 ENOSPC、extent 过多）不发布任何槽位，稍后重试。幂等。
        """
        if self._precreate_stop is not None:
            return
        store = self._store_required()
        group = self._segment_file_slots * len(self._mounts)
        headroom = max(1, group // 2)
        stop = threading.Event()
        self._precreate_stop = stop

        def run() -> None:
            space_low = False
            while not stop.is_set():
                # 磁盘空间护栏（见模块头常量注释）。
                free = self._mounts_free_bytes()
                floor = self._min_free_bytes()
                if free is not None and free < floor:
                    if not space_low:
                        space_low = True
                        _PRECREATE_LOG.warning(
                            "BACKGROUND_PRECREATE_SPACE_LOW free=%dB floor=%dB "
                            "precreated=%d；暂停增长，空间恢复后继续",
                            free, floor, store.precreated_slots(),
                        )
                    stop.wait(_PRECREATE_SPACE_RECHECK_S)
                    continue
                if space_low:
                    space_low = False
                    _PRECREATE_LOG.info(
                        "BACKGROUND_PRECREATE_SPACE_RESUMED free=%dB floor=%dB",
                        free, floor,
                    )
                started = time.monotonic()
                try:
                    made = store.precreate_step(headroom)
                except Exception as exc:
                    _PRECREATE_LOG.warning(
                        "BACKGROUND_PRECREATE_FAILED err=%r；稍后重试，写路径"
                        "保持非阻塞拒绝", exc,
                    )
                    stop.wait(_PRECREATE_SPACE_RECHECK_S)
                    continue
                if made > 0:
                    _PRECREATE_LOG.info(
                        "BACKGROUND_PRECREATE_GROUP slots=%d precreated=%d "
                        "capacity=%d took_s=%.1f",
                        made, store.precreated_slots(), self._capacity_chunks,
                        time.monotonic() - started,
                    )
                    continue
                stop.wait(_PRECREATE_IDLE_SLEEP_S)

        threading.Thread(target=run, name="tutti-precreate", daemon=True).start()
        _PRECREATE_LOG.info(
            "BACKGROUND_PRECREATE_START precreated=%d capacity=%d group=%d "
            "headroom=%d",
            store.precreated_slots(), self._capacity_chunks, group, headroom,
        )

    def stop_background_precreate(self) -> None:
        """请求后台预建停止（close 时调用；线程是 daemon，不阻塞退出）。"""
        if self._precreate_stop is not None:
            self._precreate_stop.set()
            self._precreate_stop = None

    def committed_chunks(self) -> set[bytes]:
        """已提交（= 层齐全）的 chunk 集合；层宽未定案时为空集。"""
        return set(self._committed)

    def checkpoint(self) -> None:
        """把对象层内存状态（已提交 key 集合）落到检查点。

        close() 会自动落盘；显式调用用于"希望下一次冷启动立刻看到"的时点
        （例如回收/驱逐之后）。
        """
        if self._store is not None:
            self._store.checkpoint()

    def scan(self) -> set[bytes]:
        """已提交对象的 io_key 集合。

        对象有效 ⇔ 全部段都写过，因此每个已提交 chunk 展开成 layer_span
        个 io_key——与旧实现"全层标记齐备"的判定等价。
        """
        span = self._layer_span or 0
        keys: set[bytes] = set()
        for chunk in self._committed:
            if len(chunk) == 16:
                # 标准 chunk：对象有效 ⇒ 全部层段都可读
                keys.update(
                    chunk + layer.to_bytes(2, "little")
                    for layer in range(span)
                )
            else:
                # 通用短 key（通用 KV 契约）：key 本身没有层后缀，形态即原样
                keys.add(chunk)
        return keys

    # ---------- 写路径 ----------

    def prepare_put(self, io_keys, capacity_chunks: int):
        """预留对象，返回 ``(admitted, rejected_count)``。

        ``admitted`` 形如 ``{io_key: (chunk, layer)}``，只含**受理**的 key；
        容量耗尽是稳态而非故障（对象层契约：部分受理、绝不阻塞），因此这里
        不抛错——调用方按返回的集合裁剪本批写入即可，缓存未命中不该让产生它
        的请求失败。
        """
        decoded = {}
        chunks: list[bytes] = []
        for io_key in io_keys:
            chunk_id, layer = _decode(io_key)
            decoded[bytes(io_key)] = (chunk_id, layer)
            if chunk_id not in chunks:
                chunks.append(chunk_id)
        placements, rejected = self._store_required().reserve(chunks)
        self._reserved.update(placements)
        admitted = {
            io_key: value for io_key, value in decoded.items()
            if value[0] in placements
        }
        return admitted, int(rejected)

    def commit_layers(self, io_keys) -> set[bytes]:
        """逐层记账；某 chunk 的层写齐后按对象提交一次。

        返回**本次新提交**的 chunk 集合——调用方据此把"对象有效"的那批
        层记为驻留（对象有效 ⇔ 全部段都写过）。
        """
        pending: dict[bytes, set[int]] = {}
        for io_key in io_keys:
            chunk_id, layer = _decode(io_key)
            pending.setdefault(chunk_id, set()).add(layer)
        ready: list[bytes] = []
        span = self._layer_span
        for chunk_id, layers in pending.items():
            written = self._written_layers.setdefault(chunk_id, set())
            written |= layers
            # 层宽未知（未 set_layer_span）时不允许提交：全对象语义要求
            # "所有段都写过"才有效，无法判定就必须等。
            if span is not None and len(written) >= span:
                ready.append(chunk_id)
        if not ready:
            return set()
        self._store_required().commit(ready)
        for chunk_id in ready:
            self._committed.add(chunk_id)
            self._written_layers.pop(chunk_id, None)
        return set(ready)

    def abort_uncommitted(self, chunk_ids) -> None:
        """丢弃未提交的预留（已提交的对象不受影响）。"""
        pending = [
            chunk for chunk in dict.fromkeys(bytes(c) for c in chunk_ids)
            if chunk not in self._committed
        ]
        if pending:
            self._store_required().abort(pending)
        for chunk in pending:
            self._reserved.pop(chunk, None)
            self._written_layers.pop(chunk, None)

    def release_chunks(self, chunk_ids) -> int:
        chunks = list(dict.fromkeys(bytes(c) for c in chunk_ids))
        released = self._store_required().release(chunks)
        for chunk in chunks:
            self._committed.discard(chunk)
            self._reserved.pop(chunk, None)
            self._written_layers.pop(chunk, None)
        return released

    def drop(self, io_keys) -> None:
        self.release_chunks(_decode(c)[0] for c in io_keys)

    def releasable_chunks(self, io_keys) -> set[bytes]:
        """这些 key 里可回收的 chunk：只回收已提交（层齐全）的对象。"""
        return {
            chunk for chunk in (_decode(c)[0] for c in io_keys)
            if chunk in self._committed
        }

    # ---------- 定位 ----------

    def is_committed(self, chunk_id: bytes) -> bool:
        """对象是否有效（全部段都写过）。半截对象一律不可读。"""
        return bytes(chunk_id) in self._committed

    def _placement(self, chunk_id: bytes) -> ObjectPlacement | None:
        placement = self._reserved.get(chunk_id)
        if placement is not None:
            return placement
        store = self._store
        if store is None:
            return None
        placement = store.placement(bytes(chunk_id))
        if placement is not None:
            self._reserved[chunk_id] = placement
        return placement

    def target_uri(self, chunk_id: bytes) -> str:
        placement = self._placement(chunk_id)
        if placement is None:
            raise KeyError(f"chunk 未预留：{bytes(chunk_id)!r}")
        return placement.uri

    def target_offset(self, chunk_id: bytes) -> int:
        """段 0 在对象逻辑地址空间中的起点（对象头之后）。"""
        placement = self._placement(chunk_id)
        if placement is None:
            raise KeyError(f"chunk 未预留：{bytes(chunk_id)!r}")
        return int(placement.offset)

    def target_size(self, chunk_id: bytes) -> int:
        placement = self._placement(chunk_id)
        return int(placement.payload_bytes) if placement is not None else 0

    def target_generation(self, chunk_id: bytes) -> int:
        placement = self._placement(chunk_id)
        return int(placement.generation) if placement is not None else 0

    def object_pool_snapshot(self) -> dict | None:
        if self._store is None:
            return None
        snapshot = self._store.usage()
        snapshot["committed_chunks"] = len(self._committed)
        return snapshot

    # ---------- 内部 ----------

    def _store_required(self) -> ObjectStore:
        if self._store is None:
            raise RuntimeError(
                "对象层未打开：先 set_layer_span(num_layers)（层宽定案后才能定"
                "义对象几何）"
            )
        return self._store
