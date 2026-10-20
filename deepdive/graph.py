"""LangGraph orchestration with an explicit loop: verify -> assess -> (thin?) -> research again.

Failure philosophy ("fail closed, degrade gracefully"):
  * An unverified claim NEVER reaches the report.
  * If a NON-critical agent is down (planner, researcher refinement, contradiction detector, synthesizer,
    final audit) the run continues in a degraded mode and says so in the report metrics.
  * If the VERIFIER is down, or the budget/deadline is hit, the run stops and publishes a PARTIAL report built
    only from claims that were already verified (or fails cleanly if there are none).
"""
import logging
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import agents, config, llm, metrics
from .parallel import pmap
from .live import NODE_AGENT
from . import verify as V

log = logging.getLogger("deepdive.graph")


class S(TypedDict, total=False):
    """State passed between graph nodes (kept small; the heavy data lives in the database)."""

    run_id: int
    question: str
    subqs: list       # sub-questions with their queries
    pending: list     # sub-question ids that still need research this round
    round: int        # current research round (0 = first pass)
    thin: list        # sub-questions that did not reach the verification threshold
    conflicts: list   # contradictions between verified claims
    report: dict
    dropped: list     # final sentences removed by the audit
    degraded: list    # degradation tags, e.g. 'synthesizer_fallback'


def _deg(s, tag):
    """Return the degraded list with one more tag (state updates must return new values)."""
    return list(s.get("degraded", [])) + [tag]


def tracked(store, node, fn):
    """Wrap a graph node so the dashboard can show it as running / finished / degraded / failed.

    Emits step_started, then step_finished (with duration and any NEW degradation tags) or step_failed
    (with the error) and re-raises, so failure handling in run() is unchanged.
    """
    agent = NODE_AGENT[node]

    def inner(s):
        rid, rnd, t0 = s["run_id"], s.get("round", 0), time.time()
        store.log(rid, agent, "step_started", {"node": node, "round": rnd})
        try:
            out = fn(s)
        except Exception as e:  # noqa: BLE001 - logged, then propagated unchanged
            store.log(rid, agent, "step_failed", {"node": node, "round": rnd, "kind": type(e).__name__,
                                                  "error": str(e)[:300], "seconds": round(time.time() - t0, 2)})
            raise
        added = [t for t in (out or {}).get("degraded", []) if t not in s.get("degraded", [])]
        store.log(rid, agent, "step_finished", {"node": node, "round": rnd, "degraded": added,
                                                "seconds": round(time.time() - t0, 2)})
        return out
    return inner


def assemble(store, run_id, rep, subqs, thin, conflicts, dropped, rnd, degraded, status="done"):
    """Number the sources, compute run metrics (quality + cost) and persist the finished report."""
    numbering = {}

    def cites(sent):
        """Attach footnote numbers to a sentence, numbering each distinct source in order of first use."""
        out = []
        for cid in sent["claim_ids"]:
            ev = store.evidence_by_id(store.claim(cid)["evidence_id"])
            if ev["url"] not in numbering:
                numbering[ev["url"]] = {"n": len(numbering) + 1, "url": ev["url"], "title": ev["title"],
                                        "quality": ev["quality"], "why": ev["quality_reason"]}
            out.append(numbering[ev["url"]]["n"])
        sent["cites"] = sorted(set(out))

    for sent in rep["summary"]:
        cites(sent)
    thin_set = set(thin)
    for sec in rep["sections"]:
        sec["limited"] = sec["subq_id"] in thin_set  # honest label instead of padding thin sections
        for sent in sec["sentences"]:
            cites(sent)

    allc = store.claims(run_id)
    by_status = {k: sum(c["status"] == k for c in allc) for k in ("verified", "flagged", "rejected")}
    stages = {}
    for c in allc:
        if c["status"] == "rejected":
            stages[c["stage"]] = stages.get(c["stage"], 0) + 1
    n_final = len(rep["summary"]) + sum(len(x["sentences"]) for x in rep["sections"])
    n_drop = len(dropped)
    usage = store.usage_summary(run_id)

    rep["sources"] = sorted(numbering.values(), key=lambda x: x["n"])
    rep["conflicts"] = [{"a": store.claim(x["a"])["text"], "b": store.claim(x["b"])["text"],
                         "explanation": x["explanation"], "prefer": x.get("prefer")} for x in conflicts]
    rep["unverified"] = [{"text": c["text"], "reason": c["reason"]} for c in allc if c["status"] == "flagged"]
    rep["rejected"] = [{"text": c["text"], "stage": c["stage"], "reason": c["reason"]}
                       for c in allc if c["status"] == "rejected"]
    rep["dropped_sentences"] = [{"text": d["text"], "reason": d["reason"]} for d in dropped]
    metrics_out = {
        "status": status, "degraded": degraded,
        "claims_drafted": len(allc), **{f"claims_{k}": v for k, v in by_status.items()},
        "first_pass_groundedness": round(by_status["verified"] / len(allc), 3) if allc else 0.0,
        "rejected_by_stage": stages, "research_rounds": rnd + 1,
        "sources_cited": len(numbering), "subquestions": len(subqs),
        "subquestions_fully_covered": len(subqs) - len(thin_set),
        "final_sentences": n_final, "audit_dropped": n_drop,
        "audit_pass_rate": round(n_final / (n_final + n_drop), 3) if (n_final + n_drop) else 0.0,
        "cost_usd": usage["cost_usd"], "tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        "llm_calls": usage["llm_calls"], "failed_llm_calls": usage["failed_calls"],
        "fallbacks_used": usage["fallbacks_used"], "cost_by_role": usage["by_role"],
    }
    rep["metrics"] = metrics_out
    store.set_run(run_id, report=rep, metrics=metrics_out, status=status)
    store.log(run_id, "coordinator", "done", metrics_out)
    return rep


def build_partial(store, run_id, question, status, reason):
    """Publish a report from already-verified claims when the run cannot continue. Needs NO LLM calls.

    Returns True if a partial report was produced, False if nothing verified exists.
    """
    store.set_run(run_id, error=reason)
    if not store.claims(run_id, "verified"):
        store.set_run(run_id, status=status)
        return False
    plan_ev = next((t for t in store.trail(run_id) if t["event"] == "plan"), None)
    if plan_ev:
        subqs = [{"id": x["id"], "question": x["question"]} for x in plan_ev["data"]["subquestions"]]
    else:  # planner never completed: group by whatever sub-question ids exist
        subqs = [{"id": i, "question": i} for i in sorted({c["subq"] for c in store.claims(run_id)})]
    by = {sq["id"]: store.claims(run_id, "verified", sq["id"]) for sq in subqs}
    rep = agents.fallback_report(question, subqs, by)
    assemble(store, run_id, rep, subqs, [sq["id"] for sq in subqs], [], [], 0, [status, "report_not_audited"], status)
    return True


def build_graph(store, entry="plan"):
    """Create the compiled LangGraph state machine for one run.

    `entry` is where execution starts: 'plan' for a new run, or 'research' / 'draft' / 'verify' when RESUMING a run
    that stopped part-way (see resume_plan).
    """

    def plan(s):
        """Planner node: decompose the question; degrade to a single sub-question if the planner is down."""
        try:
            subs = agents.plan(s["question"])
            degraded = s.get("degraded", [])
        except llm.AllModelsFailed:
            subs, degraded = agents.single_subquestion(s["question"]), _deg(s, "planner_fallback")
        store.log(s["run_id"], "planner", "plan", {"subquestions": [
            {"id": x["id"], "question": x["question"], "queries": x["queries"]} for x in subs]})
        return {"subqs": subs, "pending": [x["id"] for x in subs], "round": 0, "degraded": degraded}

    def research(s):
        """Researcher node: first round uses planner queries; later rounds ask for refined queries.

        Sub-questions are independent, so they are researched in parallel (config.PARALLEL_WORKERS).
        """
        rid, rnd = s["run_id"], s["round"]

        def one(sq):
            if rnd == 0:
                queries = sq["queries"]
            else:
                queries = agents.refine_queries(sq, store.claims(rid, "verified", sq["id"]),
                                                store.claims(rid, "rejected", sq["id"]))
                store.log(rid, "researcher", "refine_queries", {"subq": sq["id"], "round": rnd, "queries": queries})
            sq["tried"] += queries
            agents.research(store, rid, sq, queries, rnd)

        pmap(one, [sq for sq in s["subqs"] if sq["id"] in s["pending"]], config.PARALLEL_WORKERS)
        return {}

    def draft(s):
        """Drafter node: turn this round's evidence into claims (a failure only thins that sub-question).

        The LLM calls run in parallel; database writes happen afterwards in plan order so claim ids stay deterministic.
        """
        rid, rnd = s["run_id"], s["round"]

        def one(sq):
            rows = store.evidence(rid, sq["id"], rnd)
            if rnd == 0:
                rows += store.evidence(rid, "*", 0)  # user-uploaded documents apply to every sub-question
            rows = agents.pick_evidence(sq, rows)
            try:
                return sq, rows, agents.write_claims(sq["question"], rows)
            except llm.AllModelsFailed:
                store.log(rid, "drafter", "draft_failed", {"subq": sq["id"], "round": rnd})
                return sq, rows, None

        for sq, rows, claims in pmap(one, [sq for sq in s["subqs"] if sq["id"] in s["pending"]], config.PARALLEL_WORKERS):
            if claims is None:
                continue
            for c in claims:
                store.add_claim(rid, sq["id"], c["text"], c["evidence_id"], c["quote"], rnd)
            store.log(rid, "drafter", "draft", {"subq": sq["id"], "round": rnd, "passages": len(rows), "claims": len(claims)})
        return {}

    def verify(s):
        """Verifier node (CRITICAL): VerifierUnavailable is allowed to propagate and stop the run."""
        V.verify_claims(store, s["run_id"], store.claims(s["run_id"], status="pending"))
        return {}

    def assess(s):
        """Coordinator node: decide per sub-question whether evidence is thin, and whether to loop back."""
        rid, thin, cov = s["run_id"], [], {}
        for sq in s["subqs"]:
            ver = store.claims(rid, "verified", sq["id"])
            evs = [store.evidence_by_id(c["evidence_id"]) for c in ver]
            srcs = {e["url"] for e in evs}
            weight = round(sum(e["quality"] for e in evs), 2)  # source-quality-weighted support
            is_thin = len(ver) < config.MIN_VERIFIED or len(srcs) < config.MIN_SOURCES
            cov[sq["id"]] = {"verified": len(ver), "sources": len(srcs), "quality_weight": weight, "thin": is_thin}
            if is_thin:
                thin.append(sq["id"])
        within_budget = llm.budget_used_fraction() < config.BUDGET_STOP_FRACTION
        loop = bool(thin) and s["round"] < config.MAX_ROUNDS and within_budget
        degraded = s.get("degraded", [])
        if thin and not within_budget and s["round"] < config.MAX_ROUNDS:
            degraded = _deg(s, "research_stopped_for_budget")  # we wanted to loop but protected the budget
        decision = "loop back to research" if loop else ("stop: budget" if not within_budget and thin else "proceed")
        store.log(rid, "coordinator", "assess", {"round": s["round"], "coverage": cov, "decision": decision})
        return {"thin": thin, "pending": thin if loop else [], "degraded": degraded,
                "round": s["round"] + 1 if loop else s["round"]}

    def route(s):
        """Conditional edge: more research for thin sub-questions, otherwise move on."""
        return "research" if s["pending"] else "contradict"

    def contradict(s):
        """Contradiction-detector node (non-critical): flag disagreements; prefer the higher-quality source.

        One LLM call per sub-question, run in parallel.
        """
        rid, conflicts = s["run_id"], []

        def one(sq):
            try:
                return sq, agents.find_contradictions(store.claims(rid, "verified", sq["id"]))
            except llm.AllModelsFailed:
                store.log(rid, "contradiction-detector", "skipped", {"subq": sq["id"]})
                return sq, None

        for sq, pairs in pmap(one, s["subqs"], config.PARALLEL_WORKERS):
            for p in pairs or []:
                qa = store.evidence_by_id(store.claim(p["a"])["evidence_id"])
                qb = store.evidence_by_id(store.claim(p["b"])["evidence_id"])
                p["prefer"] = p["a"] if qa["quality"] > qb["quality"] else (p["b"] if qb["quality"] > qa["quality"] else None)
                conflicts.append(p)
        store.log(rid, "contradiction-detector", "conflicts", {"count": len(conflicts), "conflicts": conflicts})
        return {"conflicts": conflicts}

    def by_subq(s):
        """Verified claims grouped by sub-question id."""
        return {sq["id"]: store.claims(s["run_id"], "verified", sq["id"]) for sq in s["subqs"]}

    def synthesize(s):
        """Synthesizer node: LLM prose from verified claims; deterministic fallback if the LLM is down."""
        rid = s["run_id"]
        try:
            report, degraded = agents.synthesize(s["question"], s["subqs"], by_subq(s), s["conflicts"]), s.get("degraded", [])
        except llm.AllModelsFailed:
            report, degraded = agents.fallback_report(s["question"], s["subqs"], by_subq(s)), _deg(s, "synthesizer_fallback")
        store.log(rid, "synthesizer", "draft_report", {"sections": len(report["sections"])})
        return {"report": report, "degraded": degraded}

    def audit(s):
        """Audit node: re-verify the final sentences; if the auditor is down, fall back to verbatim claims."""
        rid, rep, dropped = s["run_id"], s["report"], []
        try:
            rep["summary"], d = V.audit_sentences(store, rid, rep["summary"])
            dropped += d
            for sec in rep["sections"]:
                sec["sentences"], d = V.audit_sentences(store, rid, sec["sentences"])
                dropped += d
            rep["sections"] = [x for x in rep["sections"] if x["sentences"]]
            return {"report": rep, "dropped": dropped}
        except llm.VerifierUnavailable:
            store.log(rid, "verifier", "audit_skipped", {"reason": "verifier unavailable"})
            return {"report": agents.fallback_report(s["question"], s["subqs"], by_subq(s)), "dropped": [],
                    "degraded": _deg(s, "audit_skipped")}

    def finalize(s):
        """Finalize node: number sources, compute metrics, persist."""
        assemble(store, s["run_id"], s["report"], s["subqs"], s.get("thin", []), s["conflicts"],
                 s.get("dropped", []), s["round"], s.get("degraded", []))
        return {}

    g = StateGraph(S)
    for name, fn in [("plan", plan), ("research", research), ("draft", draft), ("verify", verify),
                     ("assess", assess), ("contradict", contradict), ("synthesize", synthesize),
                     ("audit", audit), ("finalize", finalize)]:
        g.add_node(name, tracked(store, name, fn))
    g.add_edge(START, entry)
    g.add_edge("plan", "research")
    g.add_edge("research", "draft")
    g.add_edge("draft", "verify")
    g.add_edge("verify", "assess")
    g.add_conditional_edges("assess", route, {"research": "research", "contradict": "contradict"})  # the loop
    g.add_edge("contradict", "synthesize")
    g.add_edge("synthesize", "audit")
    g.add_edge("audit", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


# Statuses from which a run can be resumed (a finished 'done' run cannot).
RESUMABLE = {"failed_verifier_down", "budget_exceeded", "timeout", "interrupted", "partial", "error", "failed_auth"}


def resume_plan(store, run_id, question):
    """Work out where a stopped run should re-enter the graph and rebuild its state from the database.

    Returns (entry_node, state) or None if the run never got as far as planning (then a new run is the right answer).
      claims exist   -> 'verify'   (judge anything still pending, then re-assess coverage)
      evidence only  -> 'draft'    (the drafter never ran or failed)
      nothing yet    -> 'research'
    Evidence, claims and verdicts already stored are reused: no search, drafting or judging is repeated.
    """
    trail = store.trail(run_id)
    plan_ev = next((t for t in trail if t["event"] == "plan"), None)
    if not plan_ev:
        return None
    tried = {}
    for t in trail:  # remember the queries already used so refinement does not repeat them
        if t["event"] == "search":
            tried.setdefault(t["data"]["subq"], []).append(t["data"]["query"])
    subqs = [{"id": x["id"], "question": x["question"], "queries": x["queries"], "tried": tried.get(x["id"], [])}
             for x in plan_ev["data"]["subquestions"]]
    claims, ev = store.claims(run_id), store.evidence(run_id)
    rnd = max([c["rnd"] for c in claims] + [e["rnd"] for e in ev if e["subq"] != "*"] + [0])
    if claims:
        entry, pending = "verify", []
    elif ev:
        entry, pending = "draft", [s["id"] for s in subqs]
    else:
        entry, pending = "research", [s["id"] for s in subqs]
    return entry, {"run_id": run_id, "question": question, "subqs": subqs, "pending": pending,
                   "round": rnd, "degraded": []}


def can_resume(store, run_id):
    """(ok, reason). A run is resumable if it stopped in a RESUMABLE status and had completed planning."""
    run = store.run_brief(run_id)
    if run["status"] not in RESUMABLE:
        return False, f"status '{run['status']}' cannot be resumed"
    if not any(t["event"] == "plan" for t in store.trail(run_id)):
        return False, "run stopped before planning finished; start a new run instead"
    return True, ""


def run(store, run_id, question, resume=False, extra_usd=None):
    """Execute one research run end to end, translating every failure into a clear final status.

    Statuses: done | partial | budget_exceeded | timeout | failed_verifier_down | failed_auth | error
    With resume=True the run continues from where it stopped. Spend already incurred counts toward the caps, and
    `extra_usd` (default: a full MAX_COST_USD_PER_RUN) is the NEW allowance on top of it.
    """
    entry, state, start_cost, start_calls, max_cost, max_calls = "plan", {"run_id": run_id, "question": question}, 0.0, 0, None, None
    if resume:
        ok, why = can_resume(store, run_id)
        if not ok:
            raise ValueError(why)
        from_status = store.run_brief(run_id)["status"]
        entry, state = resume_plan(store, run_id, question)
        usage = store.usage_summary(run_id)
        start_cost, start_calls = usage["cost_usd"], usage["llm_calls"]
        max_cost = start_cost + (extra_usd if extra_usd is not None else config.MAX_COST_USD_PER_RUN)
        max_calls = start_calls + config.MAX_LLM_CALLS_PER_RUN
        store.log(run_id, "coordinator", "resume", {"from_status": from_status, "entry": entry, "spent_usd": start_cost})
        store.set_run(run_id, error=None)
    llm.begin_run(run_id, store, max_cost=max_cost, max_calls=max_calls, start_cost=start_cost, start_calls=start_calls)
    store.set_run(run_id, status="running")
    metrics.inc("runs_started_total")
    try:
        build_graph(store, entry).invoke(state, {"recursion_limit": config.GRAPH_RECURSION_LIMIT})
        metrics.inc("runs_finished_total", status="done")
    except (llm.BudgetExceeded, llm.RunTimeout, llm.VerifierUnavailable) as e:
        status = {"BudgetExceeded": "budget_exceeded", "RunTimeout": "timeout",
                  "VerifierUnavailable": "failed_verifier_down"}[type(e).__name__]
        store.log(run_id, "coordinator", "stopped", {"status": status, "reason": str(e)})
        partial = build_partial(store, run_id, question, "partial" if store.claims(run_id, "verified") else status, str(e))
        metrics.inc("runs_finished_total", status="partial" if partial else status)
    except llm.FatalLLMError as e:  # e.g. invalid API key: nothing can be salvaged
        store.log(run_id, "coordinator", "error", {"error": str(e)})
        store.set_run(run_id, status="failed_auth", error=str(e))
        metrics.inc("runs_finished_total", status="failed_auth")
    except Exception as e:  # noqa: BLE001 - last line of defence; keep the service alive
        log.exception("run crashed", extra={"run_id": run_id})
        store.log(run_id, "coordinator", "error", {"error": str(e)})
        build_partial(store, run_id, question, "error", str(e))
        metrics.inc("runs_finished_total", status="error")
    finally:
        store.log(run_id, "coordinator", "usage", store.usage_summary(run_id))
        llm.end_run()
