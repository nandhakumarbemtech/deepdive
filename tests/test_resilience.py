"""Tests of the LLM gateway layer: retries, model fallback, circuit breaker, cost accounting, budget caps."""
import json

import pytest

from deepdive import config, llm
from deepdive.store import Store

from . import fakes

SYS = "ROLE: planner\nx"


def _raw(monkeypatch, behaviour):
    """Install a fake _raw_call driven by `behaviour(model, n)`; returns the list of (model) calls."""
    llm.reset_breakers()
    monkeypatch.setattr(llm, "_sleep", lambda s: None)
    calls = []

    def fake(model, system, user, json_mode, temperature=0.1):
        calls.append(model)
        return behaviour(model, len(calls))

    monkeypatch.setattr(llm, "_raw_call", fake)
    return calls


def ok(text='{"a":1}', pt=100, ct=50):
    return text, pt, ct, False


def test_retry_then_success(monkeypatch):
    """Two transient 503s are retried on the same model, then the call succeeds."""
    def b(model, n):
        if n <= 2:
            raise fakes.HTTPErr(503)
        return ok()
    calls = _raw(monkeypatch, b)
    assert llm.call("planner", SYS, "u") == '{"a":1}'
    assert len(calls) == 3 and len(set(calls)) == 1


def test_fallback_when_first_model_down(monkeypatch, tmp_path):
    """If the preferred model is down, the next model in the chain serves the call and it is recorded."""
    first = config.chain("planner")[0]

    def b(model, n):
        if model == first:
            raise fakes.HTTPErr(503)
        return ok()
    _raw(monkeypatch, b)
    store = Store(str(tmp_path / "t.db"))
    llm.begin_run(1, store)
    assert llm.call("planner", SYS, "u")
    llm.end_run()
    assert store.usage_summary(1)["fallbacks_used"] == 1


def test_circuit_breaker_skips_dead_model(monkeypatch):
    """After repeated failures the breaker opens and the dead model is no longer called at all."""
    first = config.chain("planner")[0]

    def b(model, n):
        if model == first:
            raise fakes.HTTPErr(500)
        return ok()
    calls = _raw(monkeypatch, b)
    llm.call("planner", SYS, "u")
    llm.call("planner", SYS, "u")
    before = calls.count(first)
    llm.call("planner", SYS, "u")  # breaker is open now
    assert calls.count(first) == before
    assert llm.breaker_states()[first] == "open"


def test_auth_error_is_fatal_and_not_retried(monkeypatch):
    """A 401 means the key is wrong: fail immediately, never fall back or retry."""
    calls = _raw(monkeypatch, lambda m, n: (_ for _ in ()).throw(fakes.HTTPErr(401)))
    with pytest.raises(llm.FatalLLMError):
        llm.call("planner", SYS, "u")
    assert len(calls) == 1


def test_all_models_fail(monkeypatch):
    """Every model failing raises AllModelsFailed (callers decide how to degrade)."""
    _raw(monkeypatch, lambda m, n: (_ for _ in ()).throw(fakes.HTTPErr(503)))
    with pytest.raises(llm.AllModelsFailed):
        llm.call("planner", SYS, "u")


def test_cost_calculation():
    """Cost = tokens x price per million; unknown models use the pessimistic default price."""
    assert llm.cost_usd("openai/gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.5)
    assert llm.cost_usd("mystery/model", 1_000_000, 0) == pytest.approx(config.DEFAULT_PRICE[0])


def test_budget_cap_stops_run(monkeypatch):
    """Once spend reaches the cap the next call raises BudgetExceeded instead of spending more."""
    _raw(monkeypatch, lambda m, n: ok(pt=1_000_000, ct=0))
    llm.begin_run(1, None, max_cost=0.01)
    llm.call("planner", SYS, "u")  # costs far more than the cap
    with pytest.raises(llm.BudgetExceeded):
        llm.call("planner", SYS, "u")
    llm.end_run()


def test_json_nudge_retry(monkeypatch):
    """If a model ignores the JSON format, chat_json retries once with a corrective instruction."""
    calls = _raw(monkeypatch, lambda m, n: ok("sorry, not json") if n == 1 else ok('{"ok": true}'))
    assert llm.chat_json("planner", SYS, "u") == {"ok": True}
    assert len(calls) == 2
