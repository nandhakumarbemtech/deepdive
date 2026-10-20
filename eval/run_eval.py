"""Self-evaluation harness.   Usage: python -m eval.run_eval [N]   (runs the first N questions)

Metrics written to eval/results.json and eval/RESULTS.md:
 - first_pass_groundedness : verified / drafted claims (how often the writer was grounded BEFORE checking)
 - audit_pass_rate         : final sentences kept / (kept + dropped) in the post-synthesis audit
 - independent_groundedness: final sentences re-judged by a DIFFERENT model chain ('auditor' role)
 - coverage                : fraction of sub-questions that reached the verification threshold
 - cost_usd / tokens       : real spend measured from gateway usage (see deepdive/llm.py)
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

from deepdive import config, graph
from deepdive import llm
from deepdive import verify as V
from deepdive.store import Store


def main(n=None):
    """Run the evaluation questions end to end and write the results tables."""
    qs = json.load(open("eval/questions.json"))[:n]
    store, rows = Store(config.DB_PATH), []
    for q in qs:
        rid = store.create_run(q["question"])
        t = time.time()
        try:
            graph.run(store, rid, q["question"])
        except Exception as e:  # noqa: BLE001 - record the failure and keep evaluating
            rows.append({"id": q["id"], "error": str(e)})
            continue
        run = store.get_run(rid)
        if not run["report"]:  # run failed before producing any report
            rows.append({"id": q["id"], "error": run["status"]})
            continue
        m, rep = run["metrics"], run["report"]
        sents = rep["summary"] + [s for sec in rep["sections"] for s in sec["sentences"]]
        llm.begin_run(rid, store)  # account the independent audit's cost too
        res = V.judge(V.sentence_items(store, sents), role="auditor")  # independent model re-judges the prose
        llm.end_run()
        ind = sum(v[0] == "supported" for v in res.values()) / max(1, len(sents))
        rows.append({"id": q["id"], "run_id": rid, "status": run["status"], "seconds": round(time.time() - t),
                     "claims_drafted": m["claims_drafted"], "claims_rejected": m["claims_rejected"],
                     "first_pass_groundedness": m["first_pass_groundedness"], "audit_pass_rate": m["audit_pass_rate"],
                     "independent_groundedness": round(ind, 3),
                     "coverage": round(m["subquestions_fully_covered"] / max(1, m["subquestions"]), 3),
                     "sources": m["sources_cited"], "rounds": m["research_rounds"],
                     "cost_usd": m["cost_usd"], "tokens": m["tokens"]})
        print(rows[-1])
    ok = [r for r in rows if "error" not in r]
    keys = ("first_pass_groundedness", "audit_pass_rate", "independent_groundedness", "coverage", "rounds", "cost_usd")
    avg = {k: round(sum(r[k] for r in ok) / len(ok), 3) for k in keys} if ok else {}
    out_dir = config.EVAL_DIR
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"rows": rows, "average": avg, "n_questions": len(qs), "models": config.CHAINS,
               "generated_at": datetime.now(timezone.utc).isoformat()}, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    md = ["| id | drafted | rejected | first-pass | audit | independent | coverage | rounds | cost $ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        md.append(f"| {r['id']} | {r['claims_drafted']} | {r['claims_rejected']} | {r['first_pass_groundedness']} | "
                  f"{r['audit_pass_rate']} | {r['independent_groundedness']} | {r['coverage']} | {r['rounds']} | {r['cost_usd']} |")
    md.append(f"| **avg** | | | {avg.get('first_pass_groundedness')} | {avg.get('audit_pass_rate')} | "
              f"{avg.get('independent_groundedness')} | {avg.get('coverage')} | {avg.get('rounds')} | {avg.get('cost_usd')} |")
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
