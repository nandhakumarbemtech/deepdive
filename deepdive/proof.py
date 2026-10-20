"""Evidence for the four things a reviewer should be able to check, computed from what the run actually stored:

  1. a coordinator really orchestrates DISTINCT agents and passes results between them   -> agents_info(), handoffs()
  2. the verifier really rejects unsupported claims (you can point at a removed claim)    -> removed()
  3. the report's citations really support the sentences they are attached to            -> summarize()["citations"], recheck.py
  4. a visible decision trail and honest, self-measured groundedness numbers             -> summarize()["trail"/"metrics"/"eval"]

Nothing here calls an LLM: it only reads stored data, so it is free and always consistent with the run.
"""
import hashlib
import json
import os

from . import agents, config
from . import verify as V

# What each agent is for and which tools it owns. 'role' is the model-chain role it calls (None = no LLM).
_AGENTS = [
    ("planner", "planner", agents.PLANNER_SYS, "Splits the question into sub-questions and search queries", "none (reasoning only)"),
    ("researcher", "researcher", agents.REFINE_SYS, "Finds evidence: searches the web (provider failover), fetches pages safely, stores passages with "
     "source and quality weight; writes sharper queries when coverage is thin", "web search, page fetch + archive fallback, passage extraction"),
    ("drafter", "writer", agents.WRITER_SYS, "Turns passages into atomic claims, each with a verbatim supporting quote", "none (reads numbered passages)"),
    ("verifier", "verifier", V.JUDGE_SYS, "Checks every claim against the passage it cites: existence, quote, numbers, then a strict judge "
     "from a different model family", "deterministic gates + LLM judge"),
    ("contradiction-detector", "verifier", agents.CONTRA_SYS, "Flags verified claims whose sources disagree instead of silently picking one", "LLM, verified claims only"),
    ("synthesizer", "writer", agents.SYNTH_SYS, "Writes the report from verified claim ids only; sentences without valid ids are dropped in code", "none (verified claims only)"),
    ("coordinator", None, None, "LangGraph state machine: plans, delegates, passes results between agents, decides loop-backs, applies "
     "budget and failure policy", "graph edges, coverage rules, budget guard"),
]

DEFINITIONS = {
    "first_pass_groundedness": "verified claims / drafted claims, before any filtering: how often the writer was grounded on its first attempt",
    "audit_pass_rate": "final report sentences kept / (kept + dropped) by the post-synthesis audit",
    "independent_groundedness": "final sentences re-judged by a DIFFERENT model chain (auditor role): supported / total",
    "coverage": "fraction of sub-questions that reached the verification threshold (>= MIN_VERIFIED claims from >= MIN_SOURCES sources)",
    "red-team catch rate": "share of deliberately corrupted claims (number shifted, hedge upgraded, clause invented, negation) that the verifier rejects or flags",
}
LIMITATIONS = [
    "Verdicts come from LLM judges, which reduce but do not eliminate errors; there are no human-labelled gold answers.",
    "The question set is small (see eval/questions.json); web content changes between runs, so numbers are not exactly reproducible.",
    "Coverage measures verification thresholds, not whether the report is complete in the reader's sense.",
]


def agents_info():
    """The agents with their own job, tools, model chain and a fingerprint of their own prompt (proves they are distinct)."""
    out = []
    for name, role, prompt, job, tools in _AGENTS:
        out.append({"agent": name, "job": job, "tools": tools, "models": config.chain(role) if role else [],
                    "prompt_id": hashlib.sha1(prompt.encode()).hexdigest()[:8] if prompt else None,
                    "prompt_chars": len(prompt) if prompt else 0})
    return out


def _h(frm, to, artifact, count, detail, rnd):
    """One handoff record: what `frm` passed to `to`."""
    return {"from": frm, "to": to, "artifact": artifact, "count": count, "detail": detail, "round": rnd}


def handoffs(store, run_id):
    """Chronological list of results passed between agents, reconstructed from the stored evidence, claims and trail."""
    trail, out = store.trail(run_id), []
    plan = next((t for t in trail if t["event"] == "plan"), None)
    if plan:
        subs = plan["data"]["subquestions"]
        out.append(_h("planner", "researcher", "sub-questions + search queries", len(subs),
                      f"{len(subs)} sub-questions, {sum(len(s['queries']) for s in subs)} queries", 0))
    evidence, claims = store.evidence(run_id), store.claims(run_id)
    for r in sorted({e["rnd"] for e in evidence} | {c["rnd"] for c in claims}):
        refine = [t for t in trail if t["event"] == "refine_queries" and t["data"]["round"] == r]
        if r > 0 and refine:  # the loop-back: coordinator sends thin sub-questions back with sharper queries
            thin = sorted({t["data"]["subq"] for t in refine})
            out.append(_h("coordinator", "researcher", "LOOP-BACK: thin sub-questions + refined queries", len(thin),
                          f"{', '.join(thin)} had too little verified support; {sum(len(t['data']['queries']) for t in refine)} new queries", r))
        ev = [e for e in evidence if e["rnd"] == r]
        if ev:
            out.append(_h("researcher", "drafter", "source passages (with URL + quality)", len(ev),
                          f"{len(ev)} passages from {len({e['url'] for e in ev})} sources", r))
        cl = [c for c in claims if c["rnd"] == r]
        if cl:
            out.append(_h("drafter", "verifier", "claims with verbatim quotes", len(cl), f"{len(cl)} claims", r))
            n = {k: sum(c["status"] == k for c in cl) for k in ("verified", "flagged", "rejected", "pending")}
            out.append(_h("verifier", "coordinator", "verdicts", len(cl),
                          f"{n['verified']} verified · {n['flagged']} flagged · {n['rejected']} rejected" +
                          (f" · {n['pending']} not yet judged" if n["pending"] else ""), r))
    verified = [c for c in claims if c["status"] == "verified"]
    conf = next((t for t in reversed(trail) if t["event"] == "conflicts"), None)
    done = next((t for t in reversed(trail) if t["event"] == "done"), None)
    if conf:
        out.append(_h("coordinator", "contradiction-detector", "verified claims", len(verified), f"{len(verified)} claims compared", -1))
        out.append(_h("contradiction-detector", "synthesizer", "conflicts to report", conf["data"]["count"], f"{conf['data']['count']} conflicts flagged", -1))
    if done:
        m = done["data"]
        kept, dropped = m.get("final_sentences", 0), m.get("audit_dropped", 0)
        out.append(_h("synthesizer", "verifier (audit)", "report sentences with claim ids", kept + dropped, f"{kept + dropped} sentences drafted", -1))
        out.append(_h("verifier (audit)", "coordinator", "audited sentences", kept, f"{kept} kept · {dropped} dropped", -1))
        out.append(_h("coordinator", "report", "final cited report", kept, f"{m.get('sources_cited', 0)} sources cited; status {m.get('status')}", -1))
    return out


def removed(store, run_id):
    """Every claim the verifier removed or flagged, side by side with what its source actually says, plus audit-dropped sentences."""
    ev = {e["id"]: e for e in store.evidence(run_id)}
    claims = []
    for c in store.claims(run_id):
        if c["status"] not in ("rejected", "flagged"):
            continue
        e = ev.get(c["evidence_id"])
        passage = e["text"] if e else ""
        claims.append({"claim_id": c["id"], "subquestion": c["subq"], "status": c["status"], "stage": c["stage"],
                       "reason": c["reason"], "claim": c["text"], "quote": c["quote"],
                       "quote_found": bool(e) and V.quote_in_passage(c["quote"] or "", passage),
                       "bad_numbers": V.ungrounded_numbers(c["text"], passage) if e else [],
                       "passage": passage, "source_url": e["url"] if e else None, "source_quality": e["quality"] if e else None})
    sentences = [{"text": t["data"].get("text"), "verdict": t["data"].get("verdict"), "reason": t["data"].get("reason")}
                 for t in store.trail(run_id) if t["event"] == "sentence_dropped"]
    return {"claims": claims, "sentences": sentences}


def eval_results():
    """Measured evaluation numbers if the harness has been run (eval/results.json, eval/redteam.json), else an honest 'not measured'."""
    def load(name):
        try:
            with open(os.path.join(config.EVAL_DIR, name)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    res, red = load("results.json"), load("redteam.json")
    return {"measured": bool(res and res.get("rows")), "results": res, "redteam": red,
            "command": "python -m eval.run_eval   then   python -m eval.redteam <run_id>",
            "definitions": DEFINITIONS, "limitations": LIMITATIONS}


def summarize(store, run_id):
    """Everything the 'Proof' tab shows for one run."""
    run = store.get_run(run_id)
    m, trail = run["metrics"] or {}, store.trail(run_id)
    claims = store.claims(run_id)
    n = {k: sum(c["status"] == k for c in claims) for k in ("verified", "flagged", "rejected", "pending")}
    rem = removed(store, run_id)
    gate_first = sorted(rem["claims"], key=lambda c: (not str(c["stage"]).startswith("gate"), c["claim_id"]))
    hand = handoffs(store, run_id)
    recheck = next((t["data"] for t in reversed(trail) if t["event"] == "recheck"), None)
    kept, dropped = m.get("final_sentences", 0), m.get("audit_dropped", 0)
    usage = store.usage_summary(run_id)
    ev = eval_results()
    avg = (ev["results"] or {}).get("average") if ev["measured"] else None
    return {
        "run_done": bool(run["report"]),
        "orchestration": {"agents": len(_AGENTS), "llm_agents": sum(1 for a in _AGENTS if a[1]), "handoffs": len(hand),
                          "loopbacks": sum(1 for h in hand if h["artifact"].startswith("LOOP-BACK")),
                          "distinct_prompts": len({a["prompt_id"] for a in agents_info() if a["prompt_id"]})},
        "rejection": {"drafted": len(claims), **n,
                      "by_stage": {s: sum(1 for c in claims if c["status"] == "rejected" and c["stage"] == s)
                                   for s in sorted({c["stage"] for c in claims if c["status"] == "rejected"})},
                      "examples": gate_first[:3], "sentences_dropped": len(rem["sentences"])},
        "citations": {"total": kept, "kept": kept, "dropped": dropped, "audit_pass_rate": m.get("audit_pass_rate"),
                      "recheck": ({k: recheck[k] for k in ("supported", "total", "models", "results")} if recheck else None)},
        "trail": {"events": len(trail), "searches": sum(t["event"] == "search" for t in trail),
                  "pages": sum(t["event"] == "fetched" for t in trail), "llm_calls": usage["llm_calls"]},
        "metrics": {k: m.get(k) for k in ("first_pass_groundedness", "audit_pass_rate", "cost_usd", "research_rounds",
                                          "subquestions", "subquestions_fully_covered")},
        "eval": {"measured": ev["measured"], "average": avg, "questions": len((ev["results"] or {}).get("rows", [])),
                 "redteam": bool(ev["redteam"]), "command": ev["command"]},
    }
