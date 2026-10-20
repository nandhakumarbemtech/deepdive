"""Latency and scaling behaviour: bounded parallelism, in-flight cap, context propagation, fetch cache, percentiles."""
import threading
import time

import pytest

from deepdive import config, graph, llm, sources
from deepdive.parallel import pmap
from deepdive.store import Store

from . import fakes


def test_pmap_keeps_order_and_propagates_errors():
    """Results come back in input order; the first failure is re-raised (so budget/verifier errors still stop a run)."""
    assert pmap(lambda x: x * 2, [3, 1, 2], 4) == [6, 2, 4]
    assert pmap(lambda x: x * 2, [3, 1, 2], 1) == [6, 2, 4]          # sequential mode is identical

    def boom(x):
        if x == 2:
            raise llm.BudgetExceeded("cap")
        return x
    with pytest.raises(llm.BudgetExceeded):
        pmap(boom, [1, 2, 3], 3)


def test_inflight_cap_limits_simultaneous_llm_calls(monkeypatch):
    """Even with 8 workers, no more than the configured number of LLM requests are in flight at once."""
    llm.reset_breakers()
    monkeypatch.setattr(llm, "_inflight", threading.BoundedSemaphore(2))
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def slow_raw(model, system, user, json_mode, temperature=0.1):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return "{}", 10, 5, False

    monkeypatch.setattr(llm, "_raw_call", slow_raw)
    pmap(lambda _: llm.call("planner", "ROLE: planner\nx", "u"), range(8), 8)
    assert 1 < state["max"] <= 2


def _timed_run(monkeypatch, tmp_path, workers):
    """One full pipeline with 50 ms simulated model latency; returns (seconds, metrics)."""
    monkeypatch.setattr(config, "PARALLEL_WORKERS", workers)
    sources.clear_cache()
    fakes.install_web(monkeypatch)
    fakes.install_raw(monkeypatch, delay=0.05)
    store = Store(str(tmp_path / f"w{workers}.db"))
    rid = store.create_run("EV battery risks?")
    t0 = time.time()
    graph.run(store, rid, "EV battery risks?")
    return time.time() - t0, store.get_run(rid)["metrics"]


def test_parallel_run_is_faster_and_gives_the_same_result(monkeypatch, tmp_path):
    """Parallel stages cut wall-clock time without changing what is verified, rejected or spent."""
    t_seq, m_seq = _timed_run(monkeypatch, tmp_path, 1)
    t_par, m_par = _timed_run(monkeypatch, tmp_path, 4)
    assert t_par < t_seq * 0.9
    for k in ("claims_verified", "claims_rejected", "llm_calls", "research_rounds"):
        assert m_par[k] == m_seq[k]
    assert m_par["cost_usd"] == pytest.approx(m_seq["cost_usd"])     # accounting survives threads (context propagated)


def test_fetch_cache_avoids_repeat_downloads(monkeypatch):
    """The same URL is downloaded once within the TTL; TTL=0 disables the cache."""
    sources.clear_cache()
    n = {"c": 0}

    def direct(url, timeout):
        n["c"] += 1
        return {"title": "T", "text": "body"}, "ok"

    monkeypatch.setattr(sources, "_fetch_direct", direct)
    a = sources.fetch("https://cache.example.org/p")
    a["text"] = "mutated"                                            # callers get a copy
    assert sources.fetch("https://cache.example.org/p")["text"] == "body" and n["c"] == 1
    monkeypatch.setattr(config, "FETCH_CACHE_TTL_S", 0)
    sources.fetch("https://cache.example.org/p")
    assert n["c"] == 2


def test_latency_percentiles_in_usage_summary(tmp_path):
    """/cost reports p50/p95/max latency of successful calls."""
    store = Store(str(tmp_path / "t.db"))
    for ms in (100, 200, 300, 400, 5000):
        store.add_usage(1, "writer", "m", 1, 1, 0.0, ms, True, False, 0, False, "")
    store.add_usage(1, "writer", "m", 0, 0, 0.0, 99999, False, False, 0, False, "boom")   # failures are excluded
    lat = store.usage_summary(1)["latency_ms"]
    assert lat == {"p50": 300, "p95": 5000, "max": 5000}
