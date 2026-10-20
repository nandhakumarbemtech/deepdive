"""Live view of a run for the dashboard: per-agent state, current action, counters and human-readable log lines.

Everything is DERIVED from the audit trail and usage tables, so the dashboard needs no extra bookkeeping and
stays correct after a restart. `describe` and `pipeline` are pure functions (easy to unit-test).
"""
import time
from urllib.parse import urlparse

from . import llm

# Terminal run statuses: once reached, nothing is still running.
TERMINAL = {"done", "partial", "budget_exceeded", "timeout", "failed_verifier_down", "failed_auth",
            "error", "interrupted"}

# The agents/steps in pipeline order. `agents` = trail agent names whose events belong to this step.
NODES = [
    {"id": "plan", "label": "Planner", "does": "Splits the question into sub-questions", "agents": ["planner"]},
    {"id": "research", "label": "Researcher", "does": "Searches the web, fetches pages safely", "agents": ["researcher"]},
    {"id": "draft", "label": "Drafter", "does": "Extracts claims with verbatim quotes", "agents": ["drafter"]},
    {"id": "verify", "label": "Verifier", "does": "Quote + number gates, then LLM judge", "agents": ["verifier"]},
    {"id": "assess", "label": "Coordinator", "does": "Checks coverage; loops back or proceeds", "agents": ["coordinator"]},
    {"id": "contradict", "label": "Conflict detector", "does": "Flags disagreeing sources",
     "agents": ["contradiction-detector"]},
    {"id": "synthesize", "label": "Synthesizer", "does": "Writes prose from verified claims only",
     "agents": ["synthesizer"]},
    {"id": "audit", "label": "Final audit", "does": "Re-verifies the finished sentences",
     "agents": ["auditor", "verifier"]},
    {"id": "finalize", "label": "Finalizer", "does": "Numbers sources, computes metrics and cost",
     "agents": ["coordinator"]},
]
# Which agent name each graph node logs under (used by graph.tracked).
NODE_AGENT = {"plan": "planner", "research": "researcher", "draft": "drafter", "verify": "verifier",
              "assess": "coordinator", "contradict": "contradiction-detector", "synthesize": "synthesizer",
              "audit": "auditor", "finalize": "coordinator"}
LABEL = {n["id"]: n["label"] for n in NODES}
STEP_EVENTS = ("step_started", "step_finished", "step_failed")


def _host(url):
    """Short host name for log lines."""
    return (urlparse(url).hostname or url or "").removeprefix("www.")


def describe(row):
    """Turn one trail row into {'level','text'}; level is info | ok | warn | error."""
    e, d = row["event"], row["data"]
    node = LABEL.get(d.get("node"), d.get("node"))
    if e == "step_started":
        return {"level": "info", "text": f"{node} started" + (f" (round {d['round'] + 1})" if d.get("round") else "")}
    if e == "step_finished":
        tail = f" — degraded: {', '.join(d['degraded'])}" if d.get("degraded") else ""
        return {"level": "warn" if d.get("degraded") else "ok", "text": f"{node} finished in {d.get('seconds', 0)}s{tail}"}
    if e == "step_failed":
        return {"level": "error", "text": f"{node} FAILED ({d.get('kind')}): {d.get('error')}"}
    if e == "plan":
        return {"level": "info", "text": f"Planned {len(d['subquestions'])} sub-questions: "
                + "; ".join(f"{x['id']} {x['question']}" for x in d["subquestions"])}
    if e == "search":
        return {"level": "info", "text": f"Searched \"{d['query']}\" for {d['subq']} → {d['results']} results"}
    if e == "fetched":
        return {"level": "ok", "text": f"Fetched {_host(d['url'])} ({d['why']}, quality {d['quality']}, "
                                       f"{d['passages']} passages)"}
    if e == "fetch_failed":
        return {"level": "warn", "text": f"Could not fetch {_host(d['url'])} (blocked, timeout or not HTML)"}
    if e == "refine_queries":
        return {"level": "info", "text": f"{d['subq']} thin → new queries (round {d['round'] + 1}): "
                + "; ".join(d["queries"])}
    if e == "draft":
        return {"level": "info", "text": f"Drafted {d['claims']} claims for {d['subq']} from {d['passages']} passages"}
    if e == "draft_failed":
        return {"level": "error", "text": f"Drafting failed for {d['subq']} (models unavailable)"}
    if e == "claim_verdict":
        lvl = {"verified": "ok", "flagged": "warn", "rejected": "error"}.get(d["status"], "info")
        return {"level": lvl, "text": f"{d['status'].upper()} [{d['stage']}] {d['claim']} — {d['reason']}"}
    if e == "sentence_dropped":
        return {"level": "warn", "text": f"Audit dropped sentence ({d['verdict']}): {d['text']}"}
    if e == "assess":
        cov = d["coverage"]
        thin = [k for k, v in cov.items() if v["thin"]]
        return {"level": "info", "text": f"Coverage check: thin={thin or 'none'} → {d['decision']}"}
    if e == "conflicts":
        return {"level": "warn" if d["count"] else "info", "text": f"Conflict check: {d['count']} conflicts found"}
    if e == "skipped":
        return {"level": "warn", "text": f"Conflict check skipped for {d['subq']} (models unavailable)"}
    if e == "draft_report":
        return {"level": "info", "text": f"Synthesizer drafted {d['sections']} sections"}
    if e == "audit_skipped":
        return {"level": "warn", "text": "Final audit skipped (verifier unavailable) → verbatim verified claims used"}
    if e == "stopped":
        return {"level": "error", "text": f"Run stopped: {d['status']} — {d['reason']}"}
    if e == "error":
        return {"level": "error", "text": f"Run error: {d['error']}"}
    if e == "done":
        return {"level": "ok", "text": f"Run complete: {d['claims_verified']} verified / {d['claims_rejected']} rejected, "
                                       f"${d['cost_usd']} spent"}
    if e == "usage":
        return {"level": "info", "text": f"Usage: {d['llm_calls']} calls, ${d['cost_usd']}, {d['failed_calls']} failed, "
                                         f"{d['fallbacks_used']} via fallback"}
    if e == "recheck":
        return {"level": "ok" if d["supported"] == d["total"] else "warn",
                "text": f"Independent re-check ({', '.join(d['models'][:1])}): {d['supported']}/{d['total']} report sentences supported by their sources"}
    if e == "followup":
        return {"level": "ok", "text": f"Follow-up answered ({len(d['answer'])} verified, {d['rejected']} rejected): {d['question']}"}
    if e == "resume":
        return {"level": "info", "text": f"Resumed from status {d['from_status']} at step '{d['entry']}' (spend so far ${d['spent_usd']})"}
    return {"level": "info", "text": f"{e} {d}"}


def pipeline(rows, terminal):
    """Fold step events into one state per agent step.

    States: pending | running | done | degraded | failed | skipped | interrupted. A step that loops back
    (research/draft/verify/assess) flips to running again and its run counter increases.
    """
    st = {n["id"]: {"id": n["id"], "label": n["label"], "does": n["does"], "state": "pending", "runs": 0,
                    "seconds": 0.0, "error": None, "degraded": [], "round": None, "last_action": None,
                    "_start": None, "_end": None, "_agents": n["agents"]} for n in NODES}
    for r in rows:
        if r["event"] not in STEP_EVENTS or r["data"].get("node") not in st:
            continue
        n, d = st[r["data"]["node"]], r["data"]
        if r["event"] == "step_started":
            n.update(state="running", runs=n["runs"] + 1, _start=r["id"], _end=None, error=None, round=d.get("round"))
        elif r["event"] == "step_finished":
            n.update(state="degraded" if d.get("degraded") else "done", _end=r["id"],
                     seconds=round(n["seconds"] + d.get("seconds", 0), 1), degraded=n["degraded"] + d.get("degraded", []))
        else:
            n.update(state="failed", _end=r["id"], error=d.get("error"), seconds=round(n["seconds"] + d.get("seconds", 0), 1))
    for n in st.values():
        if terminal and n["state"] == "running":
            n["state"] = "interrupted"  # the process died or was stopped mid-step
        elif terminal and n["state"] == "pending":
            n["state"] = "skipped"      # run ended before reaching this step
        if n["_start"]:  # current/last action = latest non-step event by this agent inside the step's window
            for r in reversed(rows):
                if r["id"] < n["_start"]:
                    break
                if (n["_end"] is None or r["id"] <= n["_end"]) and r["agent"] in n["_agents"] \
                        and r["event"] not in STEP_EVENTS:
                    n["last_action"] = describe(r)["text"]
                    break
    for n in st.values():
        for k in ("_start", "_end", "_agents"):
            n.pop(k)
    return [st[n["id"]] for n in NODES]


def snapshot(store, rid, since=0, max_lines=300):
    """One JSON-able snapshot for the dashboard; `since` makes the log incremental (only new lines)."""
    run = store.run_brief(rid)
    rows = store.trail(rid)  # all rows: step states need history; only NEW rows are returned as log lines
    terminal = run["status"] in TERMINAL
    new = [r for r in rows if r["id"] > since][:max_lines]
    usage = store.usage_summary(rid)
    end = rows[-1]["ts"] if (terminal and rows) else time.time()
    steps = pipeline(rows, terminal)
    running = [s["label"] for s in steps if s["state"] == "running"]
    return {
        "run": {"id": rid, "question": run["question"], "status": run["status"], "error": run["error"],
                "terminal": terminal, "started": run["created"], "elapsed_s": round(end - run["created"], 1)},
        "running_now": running,
        "steps": steps,
        "counters": {"claims": store.claim_counts(rid), "evidence": store.evidence_stats(rid),
                     "cost_usd": usage["cost_usd"], "tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                     "llm_calls": usage["llm_calls"], "failed_calls": usage["failed_calls"],
                     "fallbacks": usage["fallbacks_used"], "by_role": usage["by_role"]},
        "llm_calls": [{"id": u["id"], "t": round(u["ts"] - run["created"], 1), "role": u["role"], "model": u["model"],
                       "ok": bool(u["ok"]), "tokens": u["prompt_tokens"] + u["completion_tokens"],
                       "cost": round(u["cost_usd"], 5), "ms": u["latency_ms"], "fallback": bool(u["fallback"]),
                       "error": u["error"]} for u in store.usage_rows(rid)],
        "breakers": llm.breaker_states(),
        "logs": [{"id": r["id"], "t": round(r["ts"] - run["created"], 1), "agent": r["agent"], "event": r["event"],
                  **describe(r)} for r in new],
        "next_since": new[-1]["id"] if new else since,
    }
