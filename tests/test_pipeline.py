"""Offline end-to-end tests of the verification logic (fake LLM at the chat_json level, fake web)."""
import pytest

from deepdive import graph, llm, sources
from deepdive import verify as V
from deepdive.store import Store

from . import fakes


def test_gates():
    """Quote gate and number gate catch altered facts, even when only one digit changes."""
    assert V.quote_in_passage("was valued at 120 billion dollars in 2024", fakes.TEXT)
    assert not V.quote_in_passage("was valued at 900 billion dollars in 2024", fakes.TEXT)
    assert V.ungrounded_numbers("The market was 480 billion in 2024", fakes.TEXT) == ["480"]


def test_quality_weighting():
    """Primary sources outrank blog platforms."""
    assert sources.quality("https://www.iea.org/x")[0] == 1.0
    assert sources.quality("https://someone.medium.com/x")[0] == 0.3


def test_full_run_rejects_bad_claims_and_loops(monkeypatch, tmp_path):
    """The whole graph: unsupported claims are rejected at the right gate, thin coverage triggers a loop-back."""
    fakes.install_web(monkeypatch)
    monkeypatch.setattr(llm, "chat_json", lambda role, system, user: fakes.respond(system, user))
    store = Store(str(tmp_path / "t.db"))
    rid = store.create_run("EV battery risks?")
    graph.run(store, rid, "EV battery risks?")
    run = store.get_run(rid)
    m = run["metrics"]
    assert run["status"] == "done"
    assert m["claims_rejected"] > 0 and m["claims_verified"] > 0
    assert "gate:number_not_in_source" in m["rejected_by_stage"]   # the 480 claim
    assert "gate:quote_not_found" in m["rejected_by_stage"]        # the fabricated quote
    assert m["research_rounds"] > 1                                # loop-back happened
    assert run["report"]["sections"] and run["report"]["sources"]
    assert any(t["event"] == "refine_queries" for t in store.trail(rid))
