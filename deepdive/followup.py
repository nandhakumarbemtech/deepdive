"""Conversational follow-up: reuses stored evidence and re-runs the SAME verification gates."""
from . import agents
from . import verify as V


def answer(store, run_id, question):
    """Answer a follow-up question using only verified claims drawn from the run's stored evidence."""
    top = agents.pick_evidence({"question": question}, store.evidence(run_id), k=8)  # best stored passages
    claims = agents.write_claims(question, top)  # draft new claims for THIS question
    ids = [store.add_claim(run_id, "FOLLOWUP", c["text"], c["evidence_id"], c["quote"], 99, origin="followup")
           for c in claims]
    V.verify_claims(store, run_id, store.claims_by_ids(ids))  # same gates as the main run
    done = store.claims_by_ids(ids)
    good = [c for c in done if c["status"] == "verified"]
    out, urls = [], {}
    for c in good:  # number sources in order of first use
        ev = store.evidence_by_id(c["evidence_id"])
        urls.setdefault(ev["url"], len(urls) + 1)
        out.append({"text": c["text"], "cite": urls[ev["url"]]})
    result = {"answer": out, "sources": [{"n": n, "url": u} for u, n in urls.items()],
              "rejected": sum(c["status"] == "rejected" for c in done),
              "note": None if out else "The stored research does not contain verified evidence for this question."}
    # Persist question + answer in the audit trail so follow-ups can be read back later (and appear in the export bundle).
    store.log(run_id, "followup", "followup", {"question": question, **result})
    return result
