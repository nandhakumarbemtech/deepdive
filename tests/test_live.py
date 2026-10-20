"""The live dashboard backend: step states, running-now, failures, incremental log, run list, UI wiring."""
import importlib

from fastapi.testclient import TestClient

from deepdive import config, graph, live
from deepdive.store import Store

from . import fakes


def _go(monkeypatch, tmp_path, **kw):
    """Run one full pipeline against fakes; returns (store, run_id)."""
    fakes.install_web(monkeypatch)
    fakes.install_raw(monkeypatch, **kw)
    store = Store(str(tmp_path / "t.db"))
    rid = store.create_run("EV battery risks?")
    graph.run(store, rid, "EV battery risks?")
    return store, rid


def _states(snap):
    """{step id: state}."""
    return {s["id"]: s["state"] for s in snap["steps"]}


def test_successful_run_snapshot(monkeypatch, tmp_path):
    """All steps done, loop-back visible as runs>1, counters and LLM feed populated, log lines readable."""
    store, rid = _go(monkeypatch, tmp_path)
    s = live.snapshot(store, rid)
    assert s["run"]["terminal"] and s["run"]["status"] == "done" and s["running_now"] == []
    assert set(_states(s).values()) <= {"done", "degraded"}
    assert next(x for x in s["steps"] if x["id"] == "research")["runs"] > 1
    c = s["counters"]
    assert c["claims"]["verified"] > 0 and c["claims"]["rejected"] > 0 and c["cost_usd"] > 0 and c["llm_calls"] > 0
    assert s["llm_calls"] and all(l["text"] for l in s["logs"])


def test_failed_run_snapshot(monkeypatch, tmp_path):
    """Verifier outage: verify = failed, later steps = skipped, earlier ones done, and the error is visible."""
    store, rid = _go(monkeypatch, tmp_path, fail_roles={"verifier"})
    s = live.snapshot(store, rid)
    st = _states(s)
    assert s["run"]["status"] == "failed_verifier_down"
    assert st["plan"] == "done" and st["verify"] == "failed" and st["finalize"] == "skipped"
    assert next(x for x in s["steps"] if x["id"] == "verify")["error"]
    assert any(l["level"] == "error" for l in s["logs"])


def test_degraded_step(monkeypatch, tmp_path):
    """Synthesizer outage is 'degraded' (fallback report), not 'failed'."""
    store, rid = _go(monkeypatch, tmp_path, fail_roles={"synthesizer"})
    assert _states(live.snapshot(store, rid))["synthesize"] == "degraded"


def test_running_now_and_interrupted(tmp_path):
    """A started-but-unfinished step is 'running' while the run is live, and 'interrupted' once the run is terminal."""
    store = Store(str(tmp_path / "t.db"))
    rid = store.create_run("q?")
    store.set_run(rid, status="running")
    store.log(rid, "researcher", "step_started", {"node": "research", "round": 0})
    s = live.snapshot(store, rid)
    assert s["running_now"] == ["Researcher"] and _states(s)["research"] == "running"
    store.set_run(rid, status="interrupted")
    assert _states(live.snapshot(store, rid))["research"] == "interrupted"


def test_incremental_log_endpoint_and_run_list(monkeypatch, tmp_path):
    """`since` returns only new lines; /api/runs lists runs with status and cost; the UI page uses the live endpoint."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "api.db"))
    app_module = importlib.reload(importlib.import_module("app"))
    fakes.install_web(monkeypatch)
    fakes.install_raw(monkeypatch)
    rid = app_module.store.create_run("EV battery risks?")
    graph.run(app_module.store, rid, "EV battery risks?")
    client = TestClient(app_module.app)
    first = client.get(f"/api/runs/{rid}/live").json()
    assert first["logs"] and first["next_since"] > 0
    assert client.get(f"/api/runs/{rid}/live", params={"since": first["next_since"]}).json()["logs"] == []
    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == rid and runs[0]["status"] == "done" and runs[0]["cost_usd"] > 0
    page = client.get("/").text
    assert "/live?since=" in page and "Running now" in page
