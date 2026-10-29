"""Per-request choice between asynchronous and synchronous KV loads.

Synchronous (``get_num_new_matched_tokens`` returns ``False``): the request
computes in the same step as its read; per-layer fences let layer L's
attention start once layer L is in HBM. The overlap is free while the read
fits in the step. A longer read makes the whole batch -- running decodes
included -- wait for the excess.

Asynchronous (``True``): vLLM parks the request in WAITING_FOR_REMOTE_KVS,
the read runs on its own stream while the other requests keep stepping, and
the request is scheduled after every TP rank has reported the read done. It
gives up its own read/compute overlap, waits for an extra scheduling round,
and holds its blocks for the whole read without computing on them.

Async is the default. Deployments can explicitly select ``sync`` or ``auto``
when they need the old cost-gated behavior:

1. **Other requests are computing.** Otherwise nobody gains from async.
2. **Memory.** Blocks held by async loads (this one included) stay within
   ``free_ratio`` of the blocks not used by computing requests. Idle-held
   blocks cost concurrency; with a small KV pool or very long contexts the
   gate keeps loads synchronous.
3. **Cost.** With ``T = C + S`` (this request's compute plus the rest of the
   step) and read time ``R``::

       sync_cost  = max(0, R - T) * others   # every other request waits
       async_cost = min(R, T)                # this request's lost overlap

   async iff ``sync_cost > async_cost``.

``R = r * chunks``, ``C = c * tokens``, ``S``: EWMAs of measured read time
per chunk, compute time per token (prefill-sized steps) and step time (steps
without synchronous reads). Model, GPU, disks and TP enter only through these
measurements. In ``auto`` mode, until ``r`` and ``c`` are measured every load
stays synchronous.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MODE_ENV = "TUTTI_LOAD_MODE"
FREE_RATIO_ENV = "TUTTI_ASYNC_LOAD_FREE_RATIO"
MODES = ("auto", "sync", "async")
DEFAULT_FREE_RATIO = 0.5

_EWMA_ALPHA = 0.2
# Steps with fewer tokens are decode-dominated: their per-token time says
# nothing about prefill compute.
_PREFILL_SAMPLE_TOKENS = 512


def resolve_load_mode(extra: dict) -> str:
    """``TUTTI_LOAD_MODE`` env > ``load_mode`` extra config > ``async``."""
    raw = os.environ.get(MODE_ENV) or extra.get("load_mode") or "async"
    mode = str(raw).strip().lower()
    if mode not in MODES:
        raise ValueError(f"load_mode must be one of {MODES}, got {raw!r}")
    return mode


def resolve_free_ratio(extra: dict) -> float:
    """``TUTTI_ASYNC_LOAD_FREE_RATIO`` env > ``async_load_free_ratio`` > 0.5."""
    raw = os.environ.get(FREE_RATIO_ENV) or extra.get("async_load_free_ratio")
    if raw is None or raw == "":
        return DEFAULT_FREE_RATIO
    ratio = float(raw)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"async_load_free_ratio must be in (0, 1], got {raw!r}")
    return ratio


class _Ewma:
    def __init__(self, alpha: float = _EWMA_ALPHA):
        self._alpha = alpha
        self.value: float | None = None

    def add(self, x: float) -> None:
        if self.value is None:
            self.value = float(x)
        else:
            self.value += self._alpha * (float(x) - self.value)


@dataclass
class LoadDecision:
    is_async: bool
    reason: str
    read_ms: float
    compute_ms: float
    step_ms: float


class LoadModePolicy:
    """Scheduler-side decision and its online measurements (see module doc)."""

    def __init__(self, *, mode: str, free_ratio: float = DEFAULT_FREE_RATIO,
                 max_step_tokens: int | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown load mode {mode!r}")
        self.mode = mode
        self.free_ratio = float(free_ratio)
        self._max_step_tokens = int(max_step_tokens) if max_step_tokens else None
        self._read_ms_per_chunk = _Ewma()
        self._compute_ms_per_token = _Ewma()
        self._step_ms = _Ewma()
        self.decisions: dict[str, int] = {}

    # ---- measurements ----

    def observe_read(self, chunks: int, ms: float) -> None:
        if chunks > 0 and ms > 0:
            self._read_ms_per_chunk.add(ms / chunks)

    def observe_step(self, tokens: int, ms: float) -> None:
        if tokens <= 0 or ms <= 0:
            return
        self._step_ms.add(ms)
        if tokens >= _PREFILL_SAMPLE_TOKENS:
            self._compute_ms_per_token.add(ms / tokens)

    # ---- decision ----

    def decide(self, *, chunks: int, compute_tokens: int, others: int,
               held_blocks: int, need_blocks: int,
               free_blocks: int | None) -> LoadDecision:
        """``others``: computing requests besides this one. ``held_blocks``:
        blocks already held by async loads. ``need_blocks``: blocks this
        request holds once admitted. ``free_blocks``: estimated free blocks
        before admitting it (None = unknown)."""
        tokens = max(0, int(compute_tokens))
        if self._max_step_tokens:
            tokens = min(tokens, self._max_step_tokens)
        r = self._read_ms_per_chunk.value
        c = self._compute_ms_per_token.value
        step = self._step_ms.value or 0.0
        read_ms = r * chunks if r is not None else 0.0
        compute_ms = c * tokens if c is not None else 0.0
        if self.mode == "async":
            # A parked request cannot drive the detached feeder by itself.
            # Keep the layerwise foreground path for the only live request;
            # detached prefetch is enabled when another request can keep the
            # scheduler/worker heartbeat active.  Retaining its blocks still
            # obeys the HBM admission guard.
            if others <= 0:
                is_async, reason = False, "alone"
            elif free_blocks is None or held_blocks + need_blocks > (
                    self.free_ratio * (max(0, free_blocks) + held_blocks)):
                is_async, reason = False, "memory"
            else:
                is_async, reason = True, "mode=async"
        elif self.mode == "sync":
            is_async, reason = False, "mode=sync"
        elif others <= 0:
            is_async, reason = False, "alone"
        elif r is None or c is None:
            is_async, reason = False, "uncalibrated"
        elif free_blocks is None or held_blocks + need_blocks > (
                self.free_ratio * (max(0, free_blocks) + held_blocks)):
            is_async, reason = False, "memory"
        else:
            hidden = compute_ms + step
            sync_cost = max(0.0, read_ms - hidden) * others
            async_cost = min(read_ms, hidden)
            is_async = sync_cost > async_cost
            reason = "cost"
        key = ("async:" if is_async else "sync:") + reason
        self.decisions[key] = self.decisions.get(key, 0) + 1
        return LoadDecision(is_async, reason, read_ms, compute_ms, step)

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "free_ratio": self.free_ratio,
            "read_ms_per_chunk": self._read_ms_per_chunk.value,
            "compute_ms_per_token": self._compute_ms_per_token.value,
            "step_ms": self._step_ms.value,
            "decisions": dict(self.decisions),
        }
