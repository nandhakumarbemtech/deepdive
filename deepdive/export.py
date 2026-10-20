"""Everything the system generates, made readable: claims with their sources, evidence passages, plain-text log,
follow-up answers, and a single downloadable bundle (zip) containing all of it.

Nothing here calls an LLM; it only reads what is already stored, so exports are free, fast and always consistent
with what the run actually did.
"""
import csv
import io
import json
import zipfile
from datetime import datetime, timezone

from . import live, render


def _iso(ts):
    """Unix time -> readable UTC timestamp."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def claims_rows(store, run_id):
    """Every claim with its verdict, the deciding stage, the quote and the source it cites."""
    ev = {e["id"]: e for e in store.evidence(run_id)}
    rows = []
    for c in store.claims(run_id, origin=None):
        e = ev.get(c["evidence_id"]) or {}
        rows.append({"claim_id": c["id"], "origin": c["origin"], "subquestion": c["subq"], "round": c["rnd"],
                     "status": c["status"], "stage": c["stage"], "reason": c["reason"], "claim": c["text"],
                     "quote": c["quote"], "evidence_id": c["evidence_id"], "source_url": e.get("url"),
                     "source_quality": e.get("quality")})
    return rows


def evidence_rows(store, run_id):
    """Every stored source passage, with the claims that cite it (so a reader can see what each source supports)."""
    cites = {}
    for c in store.claims(run_id, origin=None):
        cites.setdefault(c["evidence_id"], []).append({"claim_id": c["id"], "status": c["status"]})
    return [{"evidence_id": e["id"], "subquestion": e["subq"], "round": e["rnd"], "url": e["url"], "title": e["title"],
             "quality": e["quality"], "quality_reason": e["quality_reason"], "text": e["text"],
             "cited_by": cites.get(e["id"], [])} for e in store.evidence(run_id)]


def log_text(store, run_id):
    """The whole audit trail as a plain-text log (one line per event), suitable for reading or attaching to a bug report."""
    run = store.run_brief(run_id)
    lines = [f"DeepDive run #{run_id}  status={run['status']}  started {_iso(run['created'])} UTC",
             f"Question: {run['question']}", ""]
    for t in store.trail(run_id):
        d = live.describe(t)
        lines.append(f"{_iso(t['ts'])}  +{t['ts'] - run['created']:7.1f}s  [{d['level']:5}] {t['agent']:>22}: {d['text']}")
    return "\n".join(lines) + "\n"


def followups(store, run_id):
    """Follow-up questions asked after the run, with the verified answers given."""
    return [{"time": _iso(t["ts"]), **t["data"]} for t in store.trail(run_id) if t["event"] == "followup"]


def _csv(rows):
    """Rows (list of dicts) -> CSV text."""
    if not rows:
        return ""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


MANIFEST = """DeepDive run bundle
===================
Question: {question}
Status:   {status}

Files
-----
report.md / report.html / report.json   the final cited report (only if one was produced)
claims.csv                              every drafted claim with verdict, deciding stage, reason, quote and source
evidence.json                           every source passage retrieved, with quality weight and the claims citing it
log.txt                                 human-readable audit log of everything the agents did
trail.jsonl                             the same audit trail as machine-readable JSON lines
usage.json                              tokens, cost and latency per LLM call, plus totals by agent and model
llm_io.jsonl                            exact prompt and answer of every LLM call (absent if STORE_LLM_IO=0)
followups.json                          follow-up questions and their verified answers
metrics.json                            run metrics (groundedness, coverage, cost, degradation flags)
"""


def bundle(store, run_id):
    """Build a zip (bytes) containing every artefact of the run in readable formats."""
    run = store.get_run(run_id)
    z_io = io.BytesIO()
    with zipfile.ZipFile(z_io, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", MANIFEST.format(question=run["question"], status=run["status"]))
        if run["report"]:
            z.writestr("report.md", render.to_markdown(run["report"]))
            z.writestr("report.html", render.to_html(run["report"]))
            z.writestr("report.json", json.dumps(run["report"], indent=1))
        z.writestr("claims.csv", _csv(claims_rows(store, run_id)))
        z.writestr("evidence.json", json.dumps(evidence_rows(store, run_id), indent=1))
        z.writestr("log.txt", log_text(store, run_id))
        z.writestr("trail.jsonl", "\n".join(json.dumps(t) for t in store.trail(run_id)))
        z.writestr("usage.json", json.dumps({"summary": store.usage_summary(run_id), "calls": store.usage_all(run_id)}, indent=1))
        io_rows = store.llm_io(run_id)
        if io_rows:
            z.writestr("llm_io.jsonl", "\n".join(json.dumps(r) for r in io_rows))
        z.writestr("followups.json", json.dumps(followups(store, run_id), indent=1))
        z.writestr("metrics.json", json.dumps(run["metrics"] or {}, indent=1))
    return z_io.getvalue()
