"""Failover alternatives for search and page fetching (no network: providers and HTTP are faked)."""
import pytest

from deepdive import agents, config, sources
from deepdive.store import Store

from . import fakes


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Fresh provider breakers and a known provider order for every test."""
    sources.reset_providers()
    sources.clear_cache()
    monkeypatch.setattr(config, "SEARCH_PROVIDERS", ["ddgs", "brave", "wikipedia"])


def _providers(monkeypatch, **fns):
    """Install fake providers {name: callable or None(disabled)}; returns call counters."""
    calls = {k: 0 for k in fns}
    for name, fn in fns.items():
        def wrap(q, n, name=name, fn=fn):
            calls[name] += 1
            return fn(q, n)
        monkeypatch.setitem(sources.PROVIDERS, name, (wrap, (lambda: True) if fn else (lambda: False)))
    return calls


HIT = lambda q, n: [{"url": "https://a.org/x", "title": "t"}]


def _boom(q, n):
    raise RuntimeError("provider down")


def test_fails_over_to_next_provider(monkeypatch):
    """First provider errors -> the next one answers and the hit is tagged with who answered."""
    calls = _providers(monkeypatch, ddgs=_boom, brave=HIT, wikipedia=HIT)
    hits = sources.search("q")
    assert hits[0]["provider"] == "brave" and calls == {"ddgs": 1, "brave": 1, "wikipedia": 0}


def test_empty_answer_moves_on_without_blame(monkeypatch):
    """An EMPTY result tries the next provider but does not count as a failure of the first."""
    _providers(monkeypatch, ddgs=lambda q, n: [], brave=HIT, wikipedia=HIT)
    assert sources.search("q")[0]["provider"] == "brave"
    assert sources._pbreaker("ddgs").fails == 0


def test_disabled_provider_is_skipped(monkeypatch):
    """A provider without its API key is never called."""
    calls = _providers(monkeypatch, ddgs=_boom, brave=None, wikipedia=HIT)
    assert sources.search("q")[0]["provider"] == "wikipedia" and calls["brave"] == 0


def test_all_providers_fail_returns_empty(monkeypatch):
    """Total search outage degrades to 'no results' (never an exception)."""
    _providers(monkeypatch, ddgs=_boom, brave=_boom, wikipedia=_boom)
    assert sources.search("q") == []


def test_search_breaker_opens(monkeypatch):
    """A repeatedly failing provider is skipped entirely once its breaker opens."""
    calls = _providers(monkeypatch, ddgs=_boom, brave=HIT, wikipedia=HIT)
    for _ in range(config.BREAKER_THRESHOLD + 2):
        sources.search("q")
    assert calls["ddgs"] == config.BREAKER_THRESHOLD
    assert sources.provider_states()["ddgs"] == "open"


def test_fetch_falls_back_to_archive(monkeypatch):
    """HTTP error on the live page -> Internet Archive snapshot is used and marked."""
    seq = [(None, "status"), ({"title": "T", "text": "archived body"}, "ok")]
    monkeypatch.setattr(sources, "_fetch_direct", lambda url, timeout: seq.pop(0))
    monkeypatch.setattr(sources, "_wayback_snapshot", lambda url, timeout: "https://web.archive.org/web/1/x")
    page = sources.fetch("https://example.com/x")
    assert page["via"] == "wayback" and page["text"] == "archived body"


@pytest.mark.parametrize("reason", ["unsafe", "not_html"])
def test_no_archive_fallback_for_unsafe_or_non_html(monkeypatch, reason):
    """SSRF-blocked URLs and non-HTML files must never be retried through the archive."""
    monkeypatch.setattr(sources, "_fetch_direct", lambda url, timeout: (None, reason))
    monkeypatch.setattr(sources, "_wayback_snapshot", lambda *a: pytest.fail("fallback must not run"))
    assert sources.fetch("https://example.com/x") is None


def test_archive_fallback_can_be_disabled(monkeypatch):
    """FETCH_ARCHIVE_FALLBACK=0 turns the alternative off."""
    monkeypatch.setattr(config, "FETCH_ARCHIVE_FALLBACK", False)
    monkeypatch.setattr(sources, "_fetch_direct", lambda url, timeout: (None, "status"))
    monkeypatch.setattr(sources, "_wayback_snapshot", lambda *a: pytest.fail("fallback must not run"))
    assert sources.fetch("https://example.com/x") is None


def test_research_labels_archived_evidence_and_provider(monkeypatch, tmp_path):
    """Evidence from an archive snapshot is labelled, and the audit trail records which search provider answered."""
    monkeypatch.setattr(sources, "search", lambda q, n=6: [{"url": "https://news.example.com/a", "title": "t", "provider": "brave"}])
    monkeypatch.setattr(sources, "fetch", lambda url, timeout=None: {"title": "T", "text": fakes.TEXT, "via": "wayback"})
    store = Store(str(tmp_path / "t.db"))
    rid = store.create_run("q?")
    agents.research(store, rid, {"id": "S1", "question": "EV battery market size", "queries": ["q"]}, ["q"], 0)
    assert "(archived copy)" in store.evidence(rid)[0]["quality_reason"]
    assert next(t for t in store.trail(rid) if t["event"] == "search")["data"]["provider"] == "brave"
