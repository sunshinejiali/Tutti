"""同步/异步加载判定（load_policy）：默认同步，三道门槛 + 在线测量。"""

from __future__ import annotations

import pytest

from tutti.integration.vllm.load_policy import (
    LoadModePolicy,
    resolve_free_ratio,
    resolve_load_mode,
)


def _calibrated(read_ms_per_chunk=3.0, compute_ms_per_token=0.1, step_ms=50.0,
                **kw):
    p = LoadModePolicy(mode=kw.pop("mode", "auto"), **kw)
    p.observe_read(100, read_ms_per_chunk * 100)
    p.observe_step(4096, compute_ms_per_token * 4096)
    # 小步只进步时 EWMA，不污染每 token 计算耗时
    for _ in range(50):
        p.observe_step(8, step_ms)
    return p


def _decide(p, *, chunks=300, tokens=256, others=4, held=0, need=1000,
            free=100_000):
    return p.decide(chunks=chunks, compute_tokens=tokens, others=others,
                    held_blocks=held, need_blocks=need, free_blocks=free)


def test_uncalibrated_stays_sync():
    p = LoadModePolicy(mode="auto")
    d = _decide(p, chunks=10_000)
    assert not d.is_async and d.reason == "uncalibrated"


def test_alone_stays_sync_even_for_huge_reads():
    """没有别的请求在算：异步谁也帮不到，只丢重叠（离线单请求 A/B 实测 +28%）。"""
    d = _decide(_calibrated(), chunks=10_000, others=0)
    assert not d.is_async and d.reason == "alone"


def test_long_read_with_running_batch_goes_async():
    # R = 900 ms，T = 25.6 + 50 ≈ 76 ms：4 个请求各被拖 ~824 ms，远大于 76 ms。
    d = _decide(_calibrated(), chunks=300, tokens=256, others=4)
    assert d.is_async and d.reason == "cost"


def test_read_hidden_by_own_compute_stays_sync():
    # R = 300 ms，T = 0.1*8192 + 50 ≈ 870 ms：读被自己的计算盖住。
    d = _decide(_calibrated(), chunks=100, tokens=8192, others=8)
    assert not d.is_async


def test_cost_scales_with_batch_size():
    """暴露的读时间要乘上被拖住的请求数；只有一个陪跑时不值得异步。"""
    p = _calibrated()
    # R = 150 ms，T ≈ 76 ms：暴露 74 ms；异步代价 76 ms。
    assert not _decide(p, chunks=50, others=1).is_async
    assert _decide(p, chunks=50, others=2).is_async


def test_slower_gpu_or_bigger_model_keeps_more_loads_sync():
    fast = _calibrated(compute_ms_per_token=0.05)
    slow = _calibrated(compute_ms_per_token=0.5)
    # R = 600 ms，本请求算 1024 token：快卡 T≈101 ms 盖不住，慢卡 T≈562 ms 基本盖住。
    assert _decide(fast, chunks=200, tokens=1024, others=1).is_async
    assert not _decide(slow, chunks=200, tokens=1024, others=1).is_async


def test_step_budget_caps_compute_estimate():
    p = _calibrated(max_step_tokens=2048)
    # 不封顶时 100k token 的计算远超读；封顶后本步只算 2048 token。
    assert _decide(p, chunks=1000, tokens=100_000, others=2).is_async


def test_memory_gate_uses_free_block_ratio():
    p = _calibrated(free_ratio=0.5)
    # 可用 = 空闲 1000 + 已被异步占用 0；本请求要 600 > 0.5×1000 → 同步。
    d = _decide(p, need=600, free=1000)
    assert not d.is_async and d.reason == "memory"
    # 要 400 ≤ 500 → 放行。
    assert _decide(p, need=400, free=1000).is_async
    # 已有 300 被异步占着：300 + 400 > 0.5 × (700 + 300) → 同步。
    assert _decide(p, held=300, need=400, free=700).reason == "memory"


def test_unknown_pool_size_never_goes_async():
    d = _decide(_calibrated(), free=None)
    assert not d.is_async and d.reason == "memory"


def test_read_measurement_follows_drift():
    p = _calibrated(read_ms_per_chunk=1.0)
    for _ in range(30):
        p.observe_read(100, 600.0)   # 盘被分走带宽：6 ms/chunk
    assert p.snapshot()["read_ms_per_chunk"] == pytest.approx(6.0, rel=0.01)


def test_forced_modes_bypass_gates():
    assert _decide(_calibrated(mode="async"), others=0, free=None).is_async
    assert not _decide(_calibrated(mode="sync"), chunks=10_000).is_async


def test_resolve_load_mode(monkeypatch):
    monkeypatch.delenv("TUTTI_LOAD_MODE", raising=False)
    assert resolve_load_mode({}) == "auto"
    assert resolve_load_mode({"load_mode": "sync"}) == "sync"
    monkeypatch.setenv("TUTTI_LOAD_MODE", "async")
    assert resolve_load_mode({"load_mode": "sync"}) == "async"
    monkeypatch.setenv("TUTTI_LOAD_MODE", "bogus")
    with pytest.raises(ValueError):
        resolve_load_mode({})


def test_resolve_free_ratio(monkeypatch):
    monkeypatch.delenv("TUTTI_ASYNC_LOAD_FREE_RATIO", raising=False)
    assert resolve_free_ratio({}) == 0.5
    assert resolve_free_ratio({"async_load_free_ratio": 0.25}) == 0.25
    monkeypatch.setenv("TUTTI_ASYNC_LOAD_FREE_RATIO", "0.8")
    assert resolve_free_ratio({"async_load_free_ratio": 0.25}) == 0.8
    monkeypatch.setenv("TUTTI_ASYNC_LOAD_FREE_RATIO", "1.5")
    with pytest.raises(ValueError):
        resolve_free_ratio({})
