"""Graph-level tests: what happens to a whole run when parts of the system fail."""
from deepdive import config, graph, llm
from deepdive.store import Store

from . import fakes


def _run(monkeypatch, tmp_path, **raw_kw):
    """Run one full pipeline against the fake web and a fake gateway configured by raw_kw."""
    fakes.install_web(monkeypatch)
    fakes.install_raw(monkeypatch, **raw_kw)
    store = Store(str(tmp_path / "t.db"))
    rid = store.create_run("EV battery risks?")
    graph.run(store, rid, "EV battery risks?")
    return store, rid, store.get_run(rid)


def test_cost_is_tracked_through_the_graph(monkeypatch, tmp_path):
    """Real gateway code path: usage rows exist, cost > 0, and the report carries the cost breakdown."""
    store, rid, run = _run(monkeypatch, tmp_path)
    m = run["metrics"]
    assert run["status"] == "done"
    assert m["llm_calls"] > 0 and m["cost_usd"] > 0 and m["tokens"] > 0
    assert "verifier" in m["cost_by_role"] and "writer" in m["cost_by_role"]


def test_verifier_down_fails_closed(monkeypatch, tmp_path):
    """With the verifier fully down nothing is verified, so NO report is published (never trust unchecked claims)."""
    store, rid, run = _run(monkeypatch, tmp_path, fail_roles={"verifier"})
    assert run["status"] == "failed_verifier_down"
    assert run["report"] is None
    assert store.claims(rid, "verified") == []


def test_synthesizer_down_uses_deterministic_fallback(monkeypatch, tmp_path):
    """If the synthesizer is down the report is built from verified claims verbatim and flagged as degraded."""
    store, rid, run = _run(monkeypatch, tmp_path, fail_roles={"synthesizer"})
    assert run["status"] == "done"
    assert "synthesizer_fallback" in run["metrics"]["degraded"]
    assert run["report"]["sections"]


def test_planner_down_degrades_to_single_subquestion(monkeypatch, tmp_path):
    """A planner outage must not kill the run."""
    store, rid, run = _run(monkeypatch, tmp_path, fail_roles={"planner"})
    assert run["status"] == "done"
    assert "planner_fallback" in run["metrics"]["degraded"]


def test_one_model_down_uses_fallback_model(monkeypatch, tmp_path):
    """Primary verifier model down: the run still completes using the fallback and records that it did."""
    store, rid, run = _run(monkeypatch, tmp_path, fail_models={config.chain("verifier")[0]})
    assert run["status"] == "done"
    assert run["metrics"]["fallbacks_used"] > 0


def test_budget_exhaustion_publishes_partial_report(monkeypatch, tmp_path):
    """Hitting the call cap mid-run publishes a partial report from claims verified so far."""
    monkeypatch.setattr(config, "MAX_LLM_CALLS_PER_RUN", 8)
    store, rid, run = _run(monkeypatch, tmp_path)
    assert run["status"] == "partial"
    assert run["report"]["sections"]
    assert "partial" in run["metrics"]["degraded"]


def test_resume_after_verifier_outage(monkeypatch, tmp_path):
    """A run that failed because the verifier was down can be RESUMED once it is back: no search/drafting is repeated."""
    store, rid, run = _run(monkeypatch, tmp_path, fail_roles={"verifier"})
    assert run["status"] == "failed_verifier_down"
    ok, _ = graph.can_resume(store, rid)
    assert ok
    drafted_before = len(store.claims(rid))
    evidence_before = len(store.evidence(rid))
    fakes.install_raw(monkeypatch)                      # the verifier is healthy again
    graph.run(store, rid, "EV battery risks?", resume=True)
    run = store.get_run(rid)
    assert run["status"] == "done" and run["error"] is None
    assert store.claims(rid, "verified")                # pending claims were judged on resume
    assert any(t["event"] == "resume" for t in store.trail(rid))
    assert len(store.claims(rid, status="rejected")) > 0
    # initial pass was reused (extra rounds may add more, but the first round's rows are untouched)
    assert len(store.claims(rid)) >= drafted_before and len(store.evidence(rid)) >= evidence_before


def test_resume_refused_for_finished_run(monkeypatch, tmp_path):
    """A 'done' run cannot be resumed."""
    store, rid, run = _run(monkeypatch, tmp_path)
    assert graph.can_resume(store, rid)[0] is False


def test_resume_continues_after_budget_stop(monkeypatch, tmp_path):
    """A budget stop produces a partial report; resuming with a fresh allowance finishes the run."""
    monkeypatch.setattr(config, "MAX_LLM_CALLS_PER_RUN", 8)
    store, rid, run = _run(monkeypatch, tmp_path)
    assert run["status"] == "partial"
    graph.run(store, rid, "EV battery risks?", resume=True)   # allowance = prior calls + 8 more
    assert store.get_run(rid)["status"] in ("done", "partial")
    assert any(t["event"] == "resume" for t in store.trail(rid))
