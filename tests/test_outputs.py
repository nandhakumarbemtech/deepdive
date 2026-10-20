"""Everything generated is readable: claims, evidence, log, report json, LLM prompts/answers, follow-ups, bundle, resume API."""
import importlib
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from deepdive import config, graph

from . import fakes


@pytest.fixture
def env(monkeypatch, tmp_path):
    """App with a temp DB and one finished run produced against the fakes. Returns (client, run_id, store)."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "api.db"))
    app_module = importlib.reload(importlib.import_module("app"))
    fakes.install_web(monkeypatch)
    fakes.install_raw(monkeypatch)
    rid = app_module.store.create_run("EV battery risks?")
    graph.run(app_module.store, rid, "EV battery risks?")
    return TestClient(app_module.app), rid, app_module.store


def test_claims_and_evidence_are_readable(env):
    """Claims show verdict/stage/quote/source; evidence shows passages and which claims cite them."""
    c, rid, _ = env
    claims = c.get(f"/api/runs/{rid}/claims").json()
    assert {x["status"] for x in claims} >= {"verified", "rejected"}
    assert all("source_url" in x and "stage" in x and "quote" in x for x in claims)
    ev = c.get(f"/api/runs/{rid}/evidence").json()
    assert ev and ev[0]["text"] and any(e["cited_by"] for e in ev)


def test_log_and_report_json(env):
    """The audit trail is available as plain text and the report as structured JSON."""
    c, rid, _ = env
    log = c.get(f"/api/runs/{rid}/log.txt").text
    assert "Planner started" in log and "REJECTED" in log
    rep = c.get(f"/api/runs/{rid}/report.json").json()
    assert rep["sections"] and rep["metrics"]["cost_usd"] > 0


def test_every_llm_prompt_and_answer_is_stored(env):
    """Clicking an LLM call in the UI works: the exact prompt and answer are stored for successful calls."""
    c, rid, _ = env
    calls = c.get(f"/api/runs/{rid}/live").json()["llm_calls"]
    io_row = c.get(f"/api/runs/{rid}/llm/{calls[0]['id']}").json()
    assert io_row["system_prompt"].startswith("ROLE:") and io_row["response"]
    assert c.get(f"/api/runs/{rid}/llm/999999").status_code == 404


def test_report_fragment_has_clickable_sentences(env):
    """The embeddable report wraps each sentence with its claim ids (click-to-evidence) and carries no page <style>."""
    c, rid, _ = env
    frag = c.get(f"/api/runs/{rid}/report.html", params={"fragment": "true"}).text
    assert 'class="s" data-c="' in frag and "<style>" not in frag
    assert "<style>" in c.get(f"/api/runs/{rid}/report.html").text


def test_followup_is_persisted_and_listed(env):
    """A follow-up answer is stored in the audit trail and can be read back later."""
    c, rid, _ = env
    r = c.post(f"/api/runs/{rid}/followup", json={"question": "What was the market valued at?"})
    assert r.status_code == 200
    f = c.get(f"/api/runs/{rid}/followups").json()
    assert len(f) == 1 and f[0]["question"].startswith("What was") and "answer" in f[0]


def test_bundle_contains_everything(env):
    """One zip with the report in 3 formats, claims, evidence, logs, usage, prompts, follow-ups and metrics."""
    c, rid, _ = env
    r = c.get(f"/api/runs/{rid}/bundle.zip")
    assert r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    assert {"README.txt", "report.md", "report.html", "report.json", "claims.csv", "evidence.json", "log.txt",
            "trail.jsonl", "usage.json", "llm_io.jsonl", "followups.json", "metrics.json"} <= names
    assert json.loads(z.read("usage.json"))["summary"]["llm_calls"] > 0
    assert "claim_id" in z.read("claims.csv").decode()


def test_prompt_storage_can_be_disabled(monkeypatch, tmp_path):
    """STORE_LLM_IO=0 keeps prompts out of the database (privacy option); everything else still works."""
    monkeypatch.setattr(config, "STORE_LLM_IO", False)
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "api.db"))
    app_module = importlib.reload(importlib.import_module("app"))
    fakes.install_web(monkeypatch)
    fakes.install_raw(monkeypatch)
    rid = app_module.store.create_run("EV battery risks?")
    graph.run(app_module.store, rid, "EV battery risks?")
    assert app_module.store.llm_io(rid) == []


def test_resume_endpoint_rules(env):
    """Finished runs cannot be resumed (409); unknown runs are 404; readyz lists search provider health."""
    c, rid, _ = env
    assert c.post(f"/api/runs/{rid}/resume").status_code == 409
    assert c.post("/api/runs/99999/resume").status_code == 404
    assert "search_providers" in c.get("/readyz").json()
