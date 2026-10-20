"""Independent citation re-check: judge EVERY sentence of the final report again, with a different model chain.

The pipeline already verifies claims and audits the prose. This is the reviewer's button: it asks the `auditor` model
chain (not the verifier's) whether each sentence is supported by the source passages behind its citations, and stores
the per-sentence verdicts in the audit trail so the UI can show them on the report.
"""
from . import config
from . import verify as V


def sentence_locations(report):
    """[(loc, sentence)] in report order. loc strings match the data-loc attributes the HTML renderer emits."""
    out = [(f"s-{i}", s) for i, s in enumerate(report["summary"])]
    for j, sec in enumerate(report["sections"]):
        out += [(f"c-{j}-{k}", s) for k, s in enumerate(sec["sentences"])]
    return out


def run(store, run_id):
    """Re-judge all report sentences; persist and return {supported, total, models, results:[{loc, verdict, reason}]}."""
    report = store.get_run(run_id)["report"]
    if not report:
        raise ValueError("run has no report to check")
    locs = sentence_locations(report)
    res = V.judge(V.sentence_items(store, [s for _, s in locs], prefix="r"), role="auditor")
    results = [{"loc": loc, "verdict": res[f"r{i}"][0], "reason": res[f"r{i}"][1]} for i, (loc, _) in enumerate(locs)]
    data = {"role": "auditor", "models": config.chain("auditor"), "total": len(results),
            "supported": sum(r["verdict"] == "supported" for r in results), "results": results}
    store.log(run_id, "auditor", "recheck", data)
    return data
