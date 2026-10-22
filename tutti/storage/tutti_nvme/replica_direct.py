"""Replica-group NVMe IO directly into final KV pages, without payload staging.

Main KV uses rank-local IO; index owners read into final paged index KV.
Consumers pull remote index pages with one kernel per entity. Epoch-scoped notifications
protect readiness and the final source pages until all consumers finish.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import logging
import queue
import struct
import threading
import time

from tutti.common.nvtx import range as nvtx_range
from .layerwise_direct import TuttiLayerwiseDirectBackend
from .store import DirectAdmissionError, _TuttiCompletion

_LOG = logging.getLogger(__name__)

_POLL_SECONDS = 120.0
# Notification waits block the forward thread once per cache entity, and peers
# normally publish within a few hundred microseconds.  Poll tightly at first
# and back off only after the wait has already exceeded that window.
_TIGHT_POLLS = 32
_TIGHT_POLL_SECONDS = 0.00002
_LOOSE_POLLS = 2048
_LOOSE_POLL_SECONDS = 0.0002
_SLOW_POLL_SECONDS = 0.0005
# Failure propagation is rare; probing the terminal key on every iteration
# doubles the store traffic of every wait.
_FAILURE_PROBE_SECONDS = 0.005


def owner_index(key, size):
    if size <= 0 or len(key) < 8:
        raise ValueError("replica owner needs an eight-byte key and nonempty group")
    return int.from_bytes(bytes(key)[:8], "little") % size


def shard_indices(keys, size):
    shards = [[] for _ in range(size)]
    for index, key in enumerate(keys):
        shards[owner_index(key, size)].append(index)
    return tuple(tuple(shard) for shard in shards)


def peer_page_rows(keys, members, local_rank, tables_by_rank):
    if local_rank not in members:
        raise ValueError("replica local rank is not a source member")
    local = tables_by_rank[local_rank]
    if len(local) != len(keys):
        raise ValueError("replica key and block table counts differ")
    width = len(local[0]) if local else 0
    for rank in members:
        tables = tables_by_rank[rank]
        if len(tables) != len(keys) or any(len(t) != width for t in tables):
            raise ValueError("replica block table must contain exactly one complete chunk")
        blocks = [int(b) for table in tables for b in table]
        if any(b < 0 for b in blocks):
            raise ValueError("replica block table has a negative page")
        if len(blocks) != len(set(blocks)):
            raise ValueError("replica restore destination pages overlap")
    peers = tuple(r for r in members if r != local_rank)
    rows = []
    for index, key in enumerate(keys):
        owner = members[owner_index(key, len(members))]
        if owner != local_rank:
            rows.extend([peers.index(owner), int(source), int(destination)]
                        for source, destination in zip(tables_by_rank[owner][index], local[index]))
    return rows


class _ReplicaSignals:
    """Immutable CPU notifications; no GPU stream or tensor is consulted."""

    def __init__(self, store, rank, ranks):
        self.store, self.rank, self.ranks = store, rank, tuple(ranks)
        self.epoch = 0
        self.error = None
        self.wait_seconds = 0.0
        self.wait_calls = 0
        self._parent_check = None
        self._pause_waits = False

    def begin(self, epoch, descriptor):
        if self.error is not None:
            raise RuntimeError("replica channel is poisoned; restart the worker") from self.error
        self.check()
        if epoch <= self.epoch:
            raise RuntimeError("replica request epoch must increase")
        self.epoch = epoch
        self.store.set(f"{epoch}/plan/{self.rank}", descriptor)
        keys = [f"{epoch}/plan/{r}" for r in self.ranks]
        self.wait(keys)
        if any(self.store.get(k) != descriptor for k in keys):
            error = RuntimeError("replica keys or entity geometry differ across ranks")
            self.fail(error)
            raise error

    def fail(self, error):
        if self.error is None:
            self.error = error
            self.store.compare_set("terminal", b"", f"peer {self.rank} failed: {error}".encode())
            self.store.set(f"{self.epoch}/failed/{self.rank}", str(error))

    def check(self):
        if self.error is not None:
            raise self.error
        if self._parent_check is not None:
            self._parent_check()
        state = self.store.compare_set("terminal", b"", b"")
        if state:
            self.error = RuntimeError("replica " + state.decode())
            raise self.error

    def wait(self, keys, stop=None, check_failure=True, pauseable=False):
        deadline = time.monotonic() + _POLL_SECONDS
        started = time.monotonic()
        self.wait_calls += 1
        # A per-entity wait sits directly on the forward thread, so the poll
        # cadence is the notification latency.  Peer publication is expected
        # within a fraction of a millisecond: poll tightly first and only back
        # off once the wait has already run long.  The terminal/failure probe
        # is throttled because it costs a second store round trip.
        failure_deadline = 0.0
        polls = 0
        try:
            while True:
                if check_failure:
                    now = time.monotonic()
                    if now >= failure_deadline:
                        self.check()
                        failure_deadline = now + _FAILURE_PROBE_SECONDS
                if stop is not None and stop.is_set():
                    raise RuntimeError("replica transfer stopped")
                if self.store.check(keys):
                    return
                if not pauseable and time.monotonic() >= deadline:
                    raise TimeoutError(f"replica notification timed out: {keys}")
                polls += 1
                if polls <= _TIGHT_POLLS:
                    time.sleep(_TIGHT_POLL_SECONDS)
                elif polls <= _LOOSE_POLLS:
                    time.sleep(_LOOSE_POLL_SECONDS)
                else:
                    time.sleep(_SLOW_POLL_SECONDS)
        finally:
            self.wait_seconds += time.monotonic() - started

    def ready(self, entity, handle=b"1"):
        self.check()
        self.store.set(f"{self.epoch}/ready/{entity}/{self.rank}", handle)

    def wait_ready(self, entity, members, stop):
        self.wait([f"{self.epoch}/ready/{entity}/{r}" for r in members], stop,
                  pauseable=self._pause_waits)

    def ack(self, entity, handle=b"1"):
        self.check()
        self.store.set(f"{self.epoch}/ack/{entity}/{self.rank}", handle)

    def wait_acks(self, entity, stop, members=None):
        members = self.ranks if members is None else members
        self.wait([f"{self.epoch}/ack/{entity}/{r}" for r in members], stop,
                  pauseable=self._pause_waits)

    def quiescent(self):
        self.store.set(f"{self.epoch}/quiescent/{self.rank}", b"1")
        self.wait([f"{self.epoch}/quiescent/{r}" for r in self.ranks],
                  check_failure=False)

    def closing(self):
        self.store.compare_set("terminal", b"", f"peer {self.rank} is closing".encode())
        self.store.set(f"closing/{self.rank}", b"1")
        self.wait([f"closing/{r}" for r in self.ranks], check_failure=False)

    def release(self):
        self.store.set(f"released/{self.rank}", b"1")
        self.wait([f"released/{r}" for r in self.ranks], check_failure=False)


class _ScatterCompletion(_TuttiCompletion):
    def __init__(self, runtime, event, check_failure=None):
        super().__init__(runtime, [], lambda ok: None, auto_watch=False, direction="read")
        self.fence_event = event
        self._check_failure = check_failure

    def poll(self):
        if self._check_failure is not None:
            self._check_failure()
        return super().poll() if self.fence_event.query() else None

    def query(self):
        return self.poll() is True

    def wait_result(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else None
        if self._check_failure is not None:
            self._check_failure()
        while not self.fence_event.query():
            if self._check_failure is not None:
                self._check_failure()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("replica scatter completion timed out")
            time.sleep(0.001)
        if self._check_failure is not None:
            self._check_failure()
        return super().wait_result(timeout)


class TuttiReplicaDirectBackend(TuttiLayerwiseDirectBackend):
    requires_ordered_collectives = False
    independent_read_write = True
    _binding_serial = 0

    manages_background_reads = True

    @property
    def supports_async_read(self):
        return getattr(self._store, "_read_async_stream", None) is not None

    def __init__(self, store):
        super().__init__(store)
        self._exports = []
        self._raw_layers = ()
        self._entity_names = ()
        self._prefetch_thread = None
        self._completion_thread = None
        self._scatter_thread = None
        self._submitted_entities = queue.Queue()
        self._recovered_entities = queue.Queue()
        self._owned_work = []
        self._ipc_events = []
        self._peer_events = {}
        self._prefetch_stop = threading.Event()
        self._prefetch_error = None
        self._failure_lock = threading.Lock()
        self._scatter_plans = {}
        self._entity_sequence = ()
        self._prefetch_keys = ()
        self._next_entity = 0
        self._peer_mappings = {}
        self._peer_layers = {}
        self._peer_tables = {}
        self._owned_indices = {}
        self._signals = None
        self._epoch = 0
        self._active_token = None
        self._pending_events = []
        self._retained = []
        self._recover_lock = threading.Lock()
        self._closing = False
        self._quiesced = False
        self._poisoned = False
        self._stats = dict(read_bytes=0, write_bytes=0, gather_bytes=0,
                           padding_bytes=0, read_layers=0, read_seconds=0.0)
        self._read_direction = "read"
        self._io_direction = "read_copy"
        self._read_event_lock = threading.Lock()
        self._entity_fences = {}
        self._local_completions = {}
        self._background_allowed = None
        self._background_token = None
        self._background_scope = lambda: nullcontext(True)
        self._cancel_lock = threading.RLock()
        self._cancel_requested = threading.Event()
        self._parent = None
        self._detached_readers = {}
        self._detached_serial = 0

    def prepare_detached_read(self, plan, keys, block_tables, physical_layers, allowed,
                              submit_scope=None):
        import torch.distributed as dist
        self._detached_serial += 1
        try:
            reader = type(self)(self._store)
        finally:
            self._store._direct_backend = self
        for name in ("_geometry", "_memory_ticket", "_layer_tickets", "_layer_pages",
                     "_layer_tensors", "_raw_layers", "_entity_names", "_rank", "_peer_layers"):
            setattr(reader, name, getattr(self, name))
        reader._parent = self
        reader._invalid_plan_tokens = self._invalid_plan_tokens
        reader._read_direction = reader._io_direction = "read_async"
        reader._background_allowed = allowed
        if submit_scope is not None:
            reader._background_scope = submit_scope
        reader._signals = _ReplicaSignals(
            dist.PrefixStore(f"detached/{self._detached_serial}/", self._signals.store),
            self._rank, self._signals.ranks)
        reader._signals._parent_check = self._signals.check
        reader._signals._pause_waits = True
        reader._target_plans["read"] = plan
        import torch
        reader._compute_prefix = torch.cuda.Event()
        reader._compute_prefix.record(torch.cuda.current_stream(reader.geometry.accel_id))
        reader._detached_args = (keys, block_tables, physical_layers)
        self._detached_readers[plan.plan_token] = reader

    def start_detached_read(self, plan):
        reader = self._detached_readers[plan.plan_token]
        reader.begin_prefetch(*reader._detached_args)

    def cancel_detached_read(self, plan):
        reader = self._detached_readers.get(plan.plan_token)
        if reader is not None:
            with reader._cancel_lock:
                reader._cancel_requested.set()
                reader._prefetch_stop.set()

    def end_detached_read_plan(self, plan):
        reader = self._detached_readers.get(plan.plan_token)
        if reader is not None:
            if reader._active_token is None:
                reader._epoch = reader._signals.epoch = 1
                reader._fail(RuntimeError("detached replica initialization failed"))
            reader._quiesce()
            reader._close_scatter_plans()
            self._detached_readers.pop(plan.plan_token)
        super().end_detached_read_plan(plan)
        return reader._prefetch_error if reader is not None else None

    def register_paged_caches(self, pool, **kwargs):
        import torch
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_world_size() != 8:
            raise DirectAdmissionError("replica IO requires an initialized eight-rank world")
        self._rank = dist.get_rank()
        if self._rank != self._store._rank_id:
            raise DirectAdmissionError("replica IO does not support nonlocal/global rank remapping")
        try:
            from torch.distributed.distributed_c10d import _get_default_store
            from tutti_kv_transfer import _native
            type(self)._binding_serial += 1
            control = dist.PrefixStore(f"tutti-replica/{type(self)._binding_serial}/",
                                       _get_default_store())
            self._signals = _ReplicaSignals(control, self._rank, range(8))
            result = super().register_paged_caches(pool, **kwargs)
            if len(self._store._replica_groups) != len(self._layer_pages):
                raise DirectAdmissionError("replica group and cache entity counts differ")
            for members in self._store._replica_groups:
                if (not members or len(members) != len(set(members)) or self._rank not in members
                        or any(r not in self._signals.ranks for r in members)):
                    raise DirectAdmissionError("replica source members are invalid")
            self._entity_names = tuple(pool)
            spans = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                           for t in self._layer_tensors)
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                raise DirectAdmissionError("replica cache entity byte ranges overlap")
            with torch.cuda.device(self.geometry.accel_id):
                self._raw_layers = tuple(torch.empty(0, dtype=torch.uint8, device=t.device).set_(
                    t.untyped_storage(), t.storage_offset() * t.element_size(),
                    (p.num_blocks, p.page_bytes), (p.page_bytes, 1))
                    for t, p in zip(self._layer_tensors, self._layer_pages))
                for tensor, members in zip(self._raw_layers, self._store._replica_groups):
                    self._exports.append(_native.ReplicaIpcExport(tensor) if len(members) > 1 else None)
            self._exchange_shared_views()
            _LOG.warning("REPLICA_IO_BOUND rank=%d mode=direct_pages staging_bytes=0 entities=%d",
                         self._rank, len(self._raw_layers))
        except Exception as exc:
            self._fail(exc)
            # Imports may already exist on a peer. Keep backing until a
            # coordinated close instead of hiding the binding error in cleanup.
            raise
        return result

    def _exchange_shared_views(self):
        from tutti_kv_transfer import _native
        views = [(export.handle().hex() if export is not None else "",
                  export.offset if export is not None else 0, tensor.numel(),
                  export.allocation_bytes if export is not None else tensor.numel(),
                  page.page_bytes, tuple(members), name)
                 for export, tensor, page, members, name in zip(
                     self._exports, self._raw_layers, self._layer_pages,
                     self._store._replica_groups, self._entity_names)]
        self._signals.store.set(f"handle/{self._rank}", json.dumps(views).encode())
        self._signals.wait([f"handle/{r}" for r in self._signals.ranks])
        for peer in self._signals.ranks:
            if peer == self._rank:
                continue
            views = json.loads(self._signals.store.get(f"handle/{peer}"))
            if len(views) != len(self._raw_layers):
                raise DirectAdmissionError("replica IPC peer entity count differs")
            for physical, (handle, offset, size, allocation_bytes, page_bytes, members, name) in enumerate(views):
                local_members = self._store._replica_groups[physical]
                if (name != self._entity_names[physical] or not members
                        or len(members) != len(set(members)) or peer not in members
                        or any(r not in self._signals.ranks for r in members)
                        or ((peer in local_members or self._rank in members)
                            and tuple(members) != tuple(local_members))):
                    raise DirectAdmissionError("replica IPC peer entity or source members differ")
                if (page_bytes != self._layer_pages[physical].page_bytes or size <= 0
                        or size % page_bytes or offset < 0 or offset + size > allocation_bytes):
                    raise DirectAdmissionError("replica IPC peer page geometry is invalid")
                if len(local_members) == 1:
                    if handle or len(members) != 1:
                        raise DirectAdmissionError("main KV must remain local without IPC exports")
                    continue
                key = (peer, handle)
                if key not in self._peer_mappings:
                    self._peer_mappings[key] = _native.ReplicaIpcMapping(
                        bytes.fromhex(handle), 0, allocation_bytes, self.geometry.accel_id)
                self._peer_layers[peer, physical] = (self._peer_mappings[key], offset, size)

    def _page_requests(self, layer_idx, block_id, block_ordinal, entry, plan):
        members = self._store._replica_groups[layer_idx]
        if len(members) > 1 and owner_index(entry.chunk_id, len(members)) != members.index(self._rank):
            return []
        return super()._page_requests(layer_idx, block_id, block_ordinal, entry, plan)

    def _submit(self, keys, layer_idx, block_tables, direction, target_plan=None):
        completion = super()._submit(keys, layer_idx, block_tables, direction, target_plan)
        if direction == "write":
            completion.add_done_callback(self._record_write_result)
        return completion

    def _record_write_result(self, result):
        self._stats["write_bytes"] += sum(item.confirmed_bytes for item in result.results)

    def _check_plan(self, keys, block_tables, physical_layers):
        digest = hashlib.sha256()
        digest.update(repr((self.geometry.blocks_per_chunk, self.geometry.segment_bytes,
                            self._entity_names, physical_layers, tuple(len(g) for g in self._store._replica_groups),
                            tuple(p.page_bytes for p in self._layer_pages))).encode())
        for key, blocks in zip(keys, block_tables):
            digest.update(struct.pack("<I", len(key)))
            digest.update(bytes(key))
            digest.update(struct.pack("<I", len(blocks)))
        self._signals.begin(self._epoch, digest.digest())
        prefix = f"{self._epoch}/blocks/"
        flat = [int(b) for table in block_tables for b in table]
        self._signals.store.set(prefix + str(self._rank), struct.pack(f"<{len(flat)}q", *flat))
        self._signals.wait([prefix + str(r) for r in self._signals.ranks])
        self._peer_tables = {}
        width = self.geometry.blocks_per_chunk
        for rank in self._signals.ranks:
            blob = self._signals.store.get(prefix + str(rank))
            if len(blob) != len(flat) * 8:
                raise ValueError("replica peer block table size differs")
            blocks = struct.unpack(f"<{len(flat)}q", blob)
            if any(b < 0 for b in blocks) or len(blocks) != len(set(blocks)):
                raise ValueError("replica peer block table is negative or overlapping")
            self._peer_tables[rank] = tuple(tuple(blocks[i:i + width])
                                            for i in range(0, len(blocks), width))

    def _fail(self, error):
        if self._cancel_requested.is_set() and str(error) in (
                "replica prefetch stopped", "replica background read stopped",
                "replica transfer stopped", "replica request aborted"):
            return
        with self._failure_lock:
            self._poisoned = True
            if self._prefetch_error is None:
                self._prefetch_error = error
            if self._signals is not None:
                self._signals.fail(self._prefetch_error)
        if self._parent is not None:
            self._parent._fail(error)

    def _check_read_failure(self):
        try:
            if self._parent is not None:
                self._parent._signals.check()
            self._signals.check()
        except Exception as exc:
            self._fail(exc)
            raise

    def _wait_event(self, event, cancellable=True):
        deadline = time.monotonic() + _POLL_SECONDS
        while not event.query():
            if cancellable:
                self._signals.check()
                if self._prefetch_stop.is_set():
                    raise RuntimeError("replica transfer stopped")
                if self._background_allowed is not None and self._background_allowed() is None:
                    deadline = time.monotonic() + _POLL_SECONDS
            if time.monotonic() >= deadline:
                raise TimeoutError("replica device work timed out; backing retained")
            time.sleep(0.0001)

    def _retain_device_work(self, *backing):
        import torch
        event = torch.cuda.Event(interprocess=True)
        self._ipc_events.append(event)
        # Retain before recording: even a CUDA error recording the fence must
        # not drop tensors that were already used by submitted work.
        self._retained.extend(backing)
        self._pending_events.append(event)
        event.record(self._store._stream_for(self._read_direction)[1])
        return event

    def _wait_peer_events(self, entity, members, phase):
        import torch
        for peer in members:
            if peer == self._rank:
                continue
            key = (phase, entity, peer)
            if key not in self._peer_events:
                handle = self._signals.store.get(f"{self._epoch}/{phase}/{entity}/{peer}")
                self._peer_events[key] = torch.cuda.Event.from_ipc_handle(
                    self.geometry.accel_id, handle)
            self._store._stream_for(self._read_direction)[1].wait_event(self._peer_events[key])

    def _build_scatter_plans(self, physical_layers):
        import torch
        from tutti_kv_transfer import _native
        descriptors_by_group = {}
        max_blocks = {rank: max((b for table in tables for b in table), default=-1)
                      for rank, tables in self._peer_tables.items()}
        for physical in physical_layers:
            members = tuple(self._store._replica_groups[physical])
            if len(members) == 1:
                continue
            peers = [r for r in members if r != self._rank]
            sources = [self._peer_layers[r, physical] for r in peers]
            page_bytes = self._layer_pages[physical].page_bytes
            for rank, (_, _, size) in zip(peers, sources):
                if max_blocks[rank] >= size // page_bytes:
                    raise ValueError("replica peer source page is out of bounds")
            if members not in descriptors_by_group:
                rows = peer_page_rows(self._prefetch_keys, members, self._rank, self._peer_tables)
                descriptors_by_group[members] = torch.tensor(
                    rows, dtype=torch.int64, device="cpu").reshape(-1, 3)
            descriptors = descriptors_by_group[members]
            self._scatter_plans[physical] = _native.ReplicaPeerPagePlan(
                [s[0] for s in sources], [s[1] for s in sources], [s[2] for s in sources],
                self._raw_layers[physical], descriptors)

    def _close_scatter_plans(self):
        for plan in self._scatter_plans.values():
            plan.close()
        self._scatter_plans.clear()

    def begin_prefetch(self, keys, block_tables, physical_layers):
        import torch
        if self._closing or self._poisoned:
            raise RuntimeError("replica backend is closing or poisoned; restart the worker")
        if (self._prefetch_thread is not None or self._completion_thread is not None
                or self._scatter_thread is not None or self._active_token is not None):
            raise RuntimeError("replica prefetch already active")
        plan = self._target_plans["read"]
        self._validate_target_plan(plan)
        physical_layers = tuple(physical_layers)
        if (not physical_layers or len(set(physical_layers)) != len(physical_layers)
                or any(p < 0 or p >= self.geometry.num_layers for p in physical_layers)):
            raise ValueError("replica physical layer schedule is invalid")
        self._prefetch_keys = tuple(bytes(k) for k in keys)
        self._prefetch_tables = tuple(tuple(t) for t in self.validate_block_tables(block_tables))
        if len(self._prefetch_keys) != len(self._prefetch_tables):
            raise ValueError("replica key and block table counts differ")
        blocks = [b for table in self._prefetch_tables for b in table]
        if len(blocks) != len(set(blocks)):
            raise ValueError("replica restore destination pages overlap across chunks")
        self._epoch += 1
        self._active_token = plan.plan_token
        self._quiesced = False
        self._prefetch_stop.clear()
        self._prefetch_error = None
        self._owned_indices.clear()
        try:
            if self._parent is None:
                self._compute_prefix = torch.cuda.Event()
                self._compute_prefix.record(torch.cuda.current_stream(self.geometry.accel_id))
            self._entity_fences.clear()
            self._local_completions.clear()
            self._check_plan(self._prefetch_keys, self._prefetch_tables, physical_layers)
            with self._store.stream_context(self._io_direction):
                self._store._stream_for(self._io_direction)[1].wait_event(self._compute_prefix)
            with self._store.stream_context(self._read_direction):
                self._build_scatter_plans(physical_layers)
            self._entity_sequence = physical_layers
            self._ready_entities = queue.Queue()
            self._submitted_entities = queue.Queue()
            self._recovered_entities = queue.Queue()
            self._next_entity = 0
            self._completion_thread = threading.Thread(target=self._complete_owned,
                args=(), name="tutti-replica-completion", daemon=True)
            self._prefetch_thread = threading.Thread(target=self._produce_owned,
                args=(plan,), name="tutti-replica-prefetch", daemon=True)
            # The cross-rank restore runs ahead of the consumer: the forward
            # thread only waits for the fence of a layer that is already
            # restored instead of paying the per-entity notifications.
            self._scatter_thread = threading.Thread(target=self._drive_scatter,
                args=(), name="tutti-replica-scatter", daemon=True)
            try:
                self._completion_thread.start()
            except Exception:
                self._completion_thread = self._prefetch_thread = self._scatter_thread = None
                raise
            try:
                self._prefetch_thread.start()
            except Exception:
                self._prefetch_thread = self._scatter_thread = None
                raise
            try:
                self._scatter_thread.start()
            except Exception:
                self._scatter_thread = None
                raise
        except Exception as exc:
            self._fail(exc)
            self._prefetch_stop.set()
            raise

    def _queue_get(self, channel):
        deadline = time.monotonic() + _POLL_SECONDS
        while not self._prefetch_stop.is_set():
            if self._prefetch_error is not None:
                raise self._prefetch_error
            self._signals.check()
            try:
                item = channel.get(timeout=0.01)
                if self._cancel_requested.is_set():
                    raise RuntimeError("replica prefetch stopped")
                return item
            except queue.Empty:
                if self._parent is None and time.monotonic() >= deadline:
                    raise TimeoutError("replica prefetch queue timed out")
        raise RuntimeError("replica prefetch stopped")

    def _queue_put(self, value):
        self._signals.check()
        if self._prefetch_stop.is_set():
            raise RuntimeError("replica prefetch stopped")
        self._ready_entities.put(value)

    def _admit_background_entity(self, physical, phase="read"):
        if self._background_allowed is None:
            return
        while not self._prefetch_stop.is_set():
            self._check_read_failure()
            token = self._background_allowed()
            if token is not None and token != self._background_token:
                self._background_token = token
                return
            self._prefetch_stop.wait(0.001)
        raise RuntimeError("replica background read stopped")

    @contextmanager
    def _before_background_submit(self, handles):
        if self._background_allowed is None:
            yield
            return
        paused_at = None
        while not self._prefetch_stop.is_set():
            self._check_read_failure()
            with self._cancel_lock, self._background_scope() as allowed:
                if allowed and self._background_allowed() is not None and not self._prefetch_stop.is_set():
                    if paused_at is not None:
                        self._submit_deadline = getattr(self, "_submit_deadline", paused_at) + time.monotonic() - paused_at
                    yield
                    return
            if paused_at is None:
                paused_at = time.monotonic()
            for handle in handles:
                self._store._runtime.query_io(handle.handle)
            self._prefetch_stop.wait(0.001)
        raise RuntimeError("replica background read stopped")

    def _produce_owned(self, plan):
        import torch
        try:
            torch.cuda.set_device(self.geometry.accel_id)
            for key in self._prefetch_keys:
                self._validate_plan_entry(plan.entry_for(key))
            for physical in self._entity_sequence:
                self._signals.check()
                if self._prefetch_stop.is_set():
                    return
                members = tuple(self._store._replica_groups[physical])
                if members not in self._owned_indices:
                    self._owned_indices[members] = shard_indices(
                        self._prefetch_keys, len(members))[members.index(self._rank)]
                self._admit_background_entity(physical)
                requests = []
                with self._read_event_lock, self._store.stream_context(self._io_direction), nvtx_range(
                        f"tutti.replica.read_owned|layer={physical}|rank={self._rank}|op={self._io_direction}|mode=direct_pages"):
                    for index in self._owned_indices[members]:
                        key, blocks = self._prefetch_keys[index], self._prefetch_tables[index]
                        entry = plan.entry_for(key)
                        self._validate_plan_entry(entry)
                        for ordinal, block in enumerate(blocks):
                            requests.extend(super()._page_requests(physical, block, ordinal, entry, plan))
                    self._submit_deadline = time.monotonic() + _POLL_SECONDS
                    handles = self._store._submit_retry(requests, "read", stream_direction=self._io_direction,
                                                        on_backpressure=self._wait_submit_capacity,
                                                        before_submit=self._before_background_submit)
                    completion = _TuttiCompletion(self._store._runtime, handles, lambda ok: None,
                                                  auto_watch=False, direction="read",
                                                  retain_observation_errors=self._parent is not None)
                    self._owned_work.append((completion, None))
                    self._store._track_completion(completion, self._prefetch_keys)
                    from .store import _ASYNC_COMPLETION
                    if _ASYNC_COMPLETION:
                        self._store.enqueue_compute_wait((completion,))
                    done = torch.cuda.Event(interprocess=len(members) > 1)
                    self._ipc_events.append(done)
                    if len(members) == 1:
                        completion.fence_event = done
                        self._local_completions[physical] = completion
                    self._owned_work[-1] = (completion, done)
                    done.record(self._store._stream_for(self._io_direction)[1])
                if len(members) > 1:
                    self._signals.ready(physical, done.ipc_handle())
                self._queue_put((physical, done, None))
                self._submitted_entities.put((physical, completion, done))
                if self._parent is not None:
                    while not self._prefetch_stop.is_set():
                        with self._read_event_lock:
                            ready = self._entity_fences.get(physical)
                        if ready is not None:
                            self._wait_event(ready)
                            outcome = completion.wait_result()
                            if not outcome.ok:
                                raise RuntimeError(str(outcome))
                            break
                        self._check_read_failure()
                        self._prefetch_stop.wait(0.001)
        except Exception as exc:
            retained = tuple(getattr(exc, "tutti_submitted_completions", ()))
            for completion in retained:
                self._owned_work.append((completion, None))
            self._store._deferred_completions[:] = [
                completion for completion in self._store._deferred_completions
                if completion not in retained]
            self._fail(exc)
        finally:
            self._submitted_entities.put(None)

    def _wait_submit_capacity(self, handles):
        with nvtx_range(f"tutti.replica.capacity_wait|rank={self._rank}"):
            self._signals.check()
            if self._prefetch_stop.is_set():
                raise RuntimeError("replica prefetch stopped")
            if time.monotonic() >= self._submit_deadline:
                raise TimeoutError("replica submit capacity timed out; backing retained")
            if handles:
                self._store._runtime.query_io(handles[-1].handle)
            with self._store._inflight_lock:
                active = {item for pending in self._store._inflight_by_chunk.values()
                          for item in pending}
            for completion in active:
                completion.poll()
            self._prefetch_stop.wait(0.0001)

    def _complete_owned(self):
        import torch
        try:
            torch.cuda.set_device(self.geometry.accel_id)
            while True:
                item = self._queue_get(self._submitted_entities)
                if item is None:
                    return
                physical, completion, done = item
                with nvtx_range(f"tutti.replica.owned_complete|layer={physical}|rank={self._rank}"):
                    self._wait_event(done)
                    deadline = time.monotonic() + _POLL_SECONDS
                    while completion.poll() is None:
                        self._signals.check()
                        if self._prefetch_stop.wait(0.0001):
                            return
                        if time.monotonic() >= deadline:
                            raise TimeoutError("replica owned IO timed out; backing retained")
                    outcome = completion.wait_result()
                    self._stats["read_bytes"] += sum(r.confirmed_bytes for r in outcome.results)
                    if not outcome.ok:
                        raise RuntimeError(str(outcome))
        except Exception as exc:
            self._fail(exc)

    def _drive_scatter(self):
        """Restore every cache entity ahead of the consumer.

        The per-entity cross-rank notifications (peer readiness and the source
        page release) used to run on the forward thread once per layer.  They
        now run here: compute only waits for the fence of a layer that this
        thread has already restored, so the read no longer advances at the
        pace of the forward thread.
        """
        import torch
        try:
            torch.cuda.set_device(self.geometry.accel_id)
            while self._next_entity < len(self._entity_sequence):
                self._signals.check()
                if self._prefetch_stop.is_set():
                    return
                physical = self._entity_sequence[self._next_entity]
                members = tuple(self._store._replica_groups[physical])
                page = self._layer_pages[physical]
                label = f"layer={physical}|rank={self._rank}"
                with nvtx_range(f"tutti.replica.owned_wait|{label}"):
                    item = self._queue_get(self._ready_entities)
                if item is None:
                    return
                if item[0] != physical:
                    raise RuntimeError("replica prefetch entity order differs")
                _, io_done, error = item
                if error is not None:
                    raise error
                if len(members) == 1:
                    with self._read_event_lock:
                        self._entity_fences[physical] = io_done
                    self._next_entity += 1
                    self._recovered_entities.put((physical, io_done))
                    continue
                with nvtx_range(f"tutti.replica.peer_wait|{label}"):
                    self._signals.wait_ready(physical, members, self._prefetch_stop)
                self._admit_background_entity(physical, "scatter")
                scatter_plan = self._scatter_plans[physical]
                self._retained.append(scatter_plan)
                with self._read_event_lock, self._store.stream_context(self._read_direction):
                    if io_done is not None:
                        self._store._stream_for(self._read_direction)[1].wait_event(io_done)
                    self._wait_peer_events(physical, members, "ready")
                    with self._before_background_submit(()), nvtx_range(
                            f"tutti.replica.scatter|{label}|op={self._read_direction}|mode=direct_pages"):
                        scatter_plan.launch()
                    free = self._retain_device_work(scatter_plan)
                self._signals.ack(physical, free.ipc_handle())
                with nvtx_range(f"tutti.replica.source_release|{label}"):
                    self._signals.wait_acks(physical, self._prefetch_stop, members)
                    with self._read_event_lock, self._store.stream_context(self._read_direction):
                        self._wait_peer_events(physical, members, "ack")
                        record = (self._store.record_read_async_event if self._parent is not None
                                  else self._store.record_read_event)
                        ready = record()
                        self._entity_fences[physical] = ready
                self._stats["gather_bytes"] += sum(
                    owner_index(key, len(members)) != members.index(self._rank)
                    for key in self._prefetch_keys
                ) * self.geometry.blocks_per_chunk * page.page_bytes
                self._next_entity += 1
                self._recovered_entities.put((physical, ready))
        except Exception as exc:
            self._fail(exc)
        finally:
            self._recovered_entities.put(None)

    def join_prefetch(self):
        for name in ("_prefetch_thread", "_scatter_thread"):
            thread = getattr(self, name)
            if thread is not None:
                thread.join(timeout=_POLL_SECONDS)
                if thread.is_alive():
                    raise TimeoutError("replica prefetch did not stop; backing retained")
        if self._prefetch_error is not None:
            raise self._prefetch_error

    def stop_prefetch(self):
        if (not self._poisoned and self._active_token is not None
                and self._next_entity != len(self._entity_sequence)):
            self._fail(RuntimeError("replica request aborted"))
        self._prefetch_stop.set()
        for name in ("_prefetch_thread", "_scatter_thread", "_completion_thread"):
            thread = getattr(self, name)
            if thread is not None:
                thread.join(timeout=_POLL_SECONDS)
                if thread.is_alive():
                    raise TimeoutError("replica prefetch did not stop; backing retained")
                setattr(self, name, None)

    def _quiesce(self):
        if self._quiesced:
            return
        started_poisoned = self._poisoned
        if (self._active_token is not None and not self._poisoned
                and self._next_entity == len(self._entity_sequence)):
            self.join_prefetch()
            if self._completion_thread is not None:
                self._completion_thread.join(timeout=_POLL_SECONDS)
                if self._completion_thread.is_alive():
                    raise TimeoutError("replica completion did not stop; backing retained")
        self.stop_prefetch()
        if not self._recover_lock.acquire(timeout=_POLL_SECONDS):
            raise TimeoutError("replica recovery still active; backing retained")
        try:
            if self._layer_tickets:
                if self._parent is None:
                    for direction in dict.fromkeys((self._read_direction, self._io_direction)):
                        self._wait_event(self._store._stream_for(direction)[1], cancellable=False)
                else:
                    with self._read_event_lock:
                        tail = self._store.record_read_async_event()
                    self._wait_event(tail, cancellable=False)
                if self._parent is None:
                    self._store._wait_chunk_io(self._prefetch_keys)
                else:
                    deadline = time.monotonic() + _POLL_SECONDS
                    for completion, _ in self._owned_work:
                        while completion.poll() is None:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("replica IO drain timed out; backing retained")
                            time.sleep(0.001)
                        outcome = completion.wait_result()
                        if not outcome.ok:
                            self._fail(RuntimeError(str(outcome)))
            if self._parent is None and self._store._deferred_completions:
                self._store.drain_deferred_completions()
            if self._signals is not None:
                self._signals.quiescent()
            if self._active_token is not None:
                waited = getattr(self._signals, "wait_seconds", None)
                if waited is not None:
                    self._stats["notify_wait_ms"] = round(waited * 1000.0, 3)
                    self._stats["notify_waits"] = getattr(self._signals, "wait_calls", 0)
                    self._signals.wait_seconds = 0.0
                    self._signals.wait_calls = 0
                _LOG.warning("REPLICA_IO_STATS rank=%d %s", self._rank, self._stats)
            self._pending_events.clear()
            self._retained.clear()
            self._owned_work.clear()
            self._peer_events.clear()
            self._ipc_events.clear()
            while not self._submitted_entities.empty():
                self._submitted_entities.get_nowait()
            self._quiesced = True
            if not started_poisoned and self._prefetch_error is not None:
                raise self._prefetch_error
        finally:
            self._recover_lock.release()

    def end_target_plan(self, direction):
        if direction == "read" and self._active_token is not None:
            self._quiesce()
            self._close_scatter_plans()
            self._entity_sequence = ()
            self._peer_tables = {}
            self._owned_indices.clear()
            self._local_completions.clear()
            self._active_token = None
        super().end_target_plan(direction)

    def get_paged_batch(self, keys, layer_idx, block_tables, target_plan=None):
        if target_plan is not None and target_plan.plan_token in self._detached_readers:
            return self._detached_readers[target_plan.plan_token].get_paged_batch(
                keys, layer_idx, block_tables, target_plan)
        with self._recover_lock:
            try:
                return self._recover_batch(keys, layer_idx, block_tables, target_plan)
            except Exception as exc:
                self._fail(exc)
                raise

    def _recover_batch(self, keys, layer_idx, block_tables, target_plan=None):
        from tutti.index.chunk_index import derive_io_key
        if self._closed:
            raise RuntimeError("replica direct backend is closed")
        keys = [bytes(key) for key in keys]
        block_tables = self.validate_block_tables(block_tables)
        if len(keys) != len(block_tables) or not 0 <= layer_idx < self.geometry.num_layers:
            raise ValueError("replica layer or key/block table count is invalid")
        if any(derive_io_key(key, layer_idx) not in self._store._live for key in keys):
            raise ValueError("replica read has a nonresident fragment")
        plan = target_plan or self._target_plans.get("read")
        if plan is None:
            plan = self.begin_target_plan(keys, "read")
        self._validate_target_plan(plan)
        if self._prefetch_thread is None or plan.plan_token != self._active_token:
            raise RuntimeError("replica recovery requires its active prefetch plan")
        if self._closing:
            raise RuntimeError("replica backend is closing")
        self._signals.check()
        if tuple(keys) != self._prefetch_keys or tuple(map(tuple, block_tables)) != self._prefetch_tables:
            raise RuntimeError("replica recovery differs from its prefetch plan")
        start = time.perf_counter()
        label = f"layer={layer_idx}|rank={self._rank}"
        with nvtx_range(f"tutti.replica.prefetch_wait|{label}"):
            item = self._queue_get(self._recovered_entities)
        if item is None:
            raise self._prefetch_error or RuntimeError("replica restore stopped before this layer")
        physical, ready = item
        if physical != layer_idx:
            raise RuntimeError("replica recovery callback is duplicated or out of order")
        self._stats["read_layers"] += 1
        self._stats["read_seconds"] += time.perf_counter() - start
        if layer_idx in self._local_completions:
            return self._local_completions[layer_idx]
        return _ScatterCompletion(self._store._runtime, ready, self._check_read_failure)

    def close(self):
        if self._closed:
            return
        self._closing = True
        if self._signals is not None:
            self._signals.closing()
        for reader in tuple(self._detached_readers.values()):
            reader._quiesce()
            reader._close_scatter_plans()
        self._detached_readers.clear()
        self._quiesce()
        self._close_scatter_plans()
        for mapping in self._peer_mappings.values():
            mapping.close()
        self._peer_mappings.clear()
        self._peer_layers.clear()
        if self._signals is not None:
            self._signals.release()
        for export in self._exports:
            if export is not None:
                export.close()
        self._exports.clear()
        self._raw_layers = ()
        super().close()
