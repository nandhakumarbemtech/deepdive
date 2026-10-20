"""Shared fakes for offline tests: a fake LLM (keyed on the ROLE line of each prompt) and a fake web."""
import json
import re
import time

from deepdive import llm, sources

TEXT = ("The global electric vehicle battery market was valued at 120 billion dollars in 2024 according to the "
        "industry report, and analysts expect strong growth through 2030 as demand for lithium ion cells rises "
        "across passenger vehicles and grid storage. Prices may fall further if raw material costs stay low.")


class HTTPErr(Exception):
    """Fake SDK error carrying an HTTP status code (the gateway layer classifies errors by status)."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def respond(system, user):
    """Return the JSON-able answer a well-behaved-but-imperfect model would give for each agent role."""
    role = re.match(r"ROLE: (\S+)", system).group(1)
    if role == "planner":
        return {"subquestions": [{"id": "S1", "question": "market size of EV batteries", "queries": ["ev battery size"]},
                                 {"id": "S2", "question": "EV battery risks", "queries": ["ev battery risks"]}]}
    if role == "researcher":
        return {"queries": ["ev battery market primary source report"]}
    if role == "drafter":
        eid = re.search(r"\[E(\d+)\]", user).group(1)
        quote = "was valued at 120 billion dollars in 2024 according to the industry report"
        return {"claims": [
            {"text": "The EV battery market was valued at 120 billion dollars in 2024.", "evidence_id": f"E{eid}", "quote": quote},
            {"text": "The EV battery market was valued at 480 billion dollars in 2024.", "evidence_id": f"E{eid}", "quote": quote},
            {"text": "Demand will triple by 2040.", "evidence_id": f"E{eid}",
             "quote": "demand is projected to triple by the year 2040 worldwide"}]}
    if role == "verifier":
        ids = re.findall(r"### ITEM (\S+)", user)
        return {"results": [{"id": i, "verdict": "supported", "reason": "ok"} for i in ids]}
    if role == "contradiction-detector":
        return {"conflicts": []}
    if role == "synthesizer":
        cids = sorted(set(re.findall(r"\[C(\d+)\]", user)), key=int)
        return {"title": "T", "summary": [{"text": "Summary.", "claim_ids": [f"C{cids[0]}"]}],
                "sections": [{"subq_id": "S1", "heading": "Size",
                              "sentences": [{"text": "Valued at 120 billion.", "claim_ids": [f"C{cids[0]}"]}]}]}
    raise AssertionError(role)


def install_web(monkeypatch):
    """Replace search/fetch with deterministic fakes (a different URL per query so loop-backs find new pages)."""
    monkeypatch.setattr(sources, "search",
                        lambda q, n=6, retries=2: [{"url": f"https://site{abs(hash(q)) % 999}.org/a", "title": "t"}])
    monkeypatch.setattr(sources, "fetch", lambda url, timeout=None: {"title": "T", "text": TEXT})


def install_raw(monkeypatch, fail_roles=(), fail_models=(), fail_status=503, usage=(1000, 500), delay=0.0):
    """Patch the lowest-level HTTP call so the REAL retry/fallback/breaker/accounting code runs.

    fail_roles  : prompt roles (e.g. 'verifier') that always error, simulating a full outage
    fail_models : model ids that always error, simulating one model being down
    """
    llm.reset_breakers()
    monkeypatch.setattr(llm, "_sleep", lambda s: None)  # never actually wait in tests
    calls = []

    def fake_raw(model, system, user, json_mode, temperature=0.1):
        calls.append(model)
        if delay:
            time.sleep(delay)  # simulate model latency (threads overlap, so parallel stages get faster)
        role = re.match(r"ROLE: (\S+)", system).group(1)
        if role in fail_roles or model in fail_models:
            raise HTTPErr(fail_status)
        return json.dumps(respond(system, user)), usage[0], usage[1], False

    monkeypatch.setattr(llm, "_raw_call", fake_raw)
    return calls
