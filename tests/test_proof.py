"""The four review points are provable from stored data: distinct agents + handoffs, removed claims, citation support, trail + numbers."""
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from deepdive import config, graph, proof
from deepdive.store import Store

from . import fakes


@pytest.fixture
def env(monkeypatch, tmp_path):
    """App on a temp DB, a temp (empty) eval dir, and one finished run built from the fakes."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setattr(config, "EVAL_DIR", str(tmp_path / "evaldir"))
    app_module = importlib.reload(importlib.import_module("app"))
    fakes.install_web(monkeypatch)
    fakes.install_raw(monkeypatch)
    rid = app_module.store.create_run("EV battery risks?")
    graph.run(app_module.store, rid, "EV battery risks?")
    return TestClient(app_module.app), rid, app_module.store, tmp_path


def test_agents_are_distinct():
    """Each LLM agent has its own prompt (unique fingerprint), job and tools; the coordinator is logic, not a prompt."""
    info = proof.agents_info()
    assert len(info) >= 6
    ids = [a["prompt_id"] for a in info if a["prompt_id"]]
    assert len(ids) == len(set(ids)) >= 6                         # no two agents share a prompt
    assert all(a["job"] and a["tools"] for a in info)


def test_handoffs_show_results_passed_between_agents(env):
    """The trail of handoffs: planner -> researcher -> drafter -> verifier -> coordinator, a loop-back, then synthesis and audit."""
    c, rid, store, _ = env
    h = c.get(f"/api/runs/{rid}/handoffs").json()["handoffs"]
    pairs = [(x["from"], x["to"]) for x in h]
    for p in [("planner", "researcher"), ("researcher", "drafter"), ("drafter", "verifier"), ("verifier", "coordinator"),
              ("coordinator", "researcher"), ("synthesizer", "verifier (audit)"), ("coordinator", "report")]:
        assert p in pairs
    m = store.get_run(rid)["metrics"]
    verdicts = [x for x in h if x["artifact"] == "verdicts"]
    assert sum(int(x["detail"].split(" verified")[0]) for x in verdicts) == m["claims_verified"]


def test_removed_claims_can_be_pointed_at(env):
    """A removed claim comes with the passage it was checked against, the reason, the bad number and quote status."""
    c, rid, _, _ = env
    rem = c.get(f"/api/runs/{rid}/removed").json()["claims"]
    num = next(x for x in rem if x["stage"] == "gate:number_not_in_source")
    assert num["bad_numbers"] == ["480"] and num["passage"] and "120 billion" in num["passage"]
    fab = next(x for x in rem if x["stage"] == "gate:quote_not_found")
    assert fab["quote_found"] is False and "2040" in fab["claim"]


def test_proof_summary_and_honest_eval_state(env):
    """Proof lists rejections and trail counts; evaluation is reported as NOT measured until results exist."""
    c, rid, _, tmp = env
    p = c.get(f"/api/runs/{rid}/proof").json()
    assert p["orchestration"]["loopbacks"] >= 1 and p["orchestration"]["distinct_prompts"] >= 6
    assert p["rejection"]["rejected"] > 0 and p["rejection"]["examples"]
    assert p["citations"]["total"] > 0 and p["trail"]["events"] > 10
    assert p["eval"]["measured"] is False                                   # honest: nothing measured yet
    (tmp / "evaldir").mkdir()
    (tmp / "evaldir" / "results.json").write_text(json.dumps({"rows": [{"id": "q1"}], "average": {"coverage": 0.8}}))
    p = c.get(f"/api/runs/{rid}/proof").json()
    assert p["eval"]["measured"] is True and p["eval"]["average"] == {"coverage": 0.8}
    assert "definitions" in c.get("/api/eval").json() and c.get("/api/eval").json()["limitations"]


def test_independent_recheck_is_persisted_and_painted(env):
    """The recheck judges every sentence, stores the verdicts, and the report HTML carries matching data-loc anchors."""
    c, rid, store, _ = env
    r = c.post(f"/api/runs/{rid}/recheck").json()
    assert r["total"] == store.get_run(rid)["metrics"]["final_sentences"] and r["supported"] == r["total"]
    html = c.get(f"/api/runs/{rid}/report.html", params={"fragment": "true"}).text
    assert all(f'data-loc="{x["loc"]}"' in html for x in r["results"])
    assert c.get(f"/api/runs/{rid}/proof").json()["citations"]["recheck"]["total"] == r["total"]
    assert any(t["event"] == "recheck" for t in store.trail(rid))


def test_recheck_refuses_to_claim_when_models_are_down(env, monkeypatch):
    """If the auditor models are unavailable the endpoint says so (503); it never reports a fake pass."""
    c, rid, _, _ = env
    fakes.install_raw(monkeypatch, fail_roles={"verifier"})                 # judge prompts are ROLE: verifier
    assert c.post(f"/api/runs/{rid}/recheck").status_code == 503
