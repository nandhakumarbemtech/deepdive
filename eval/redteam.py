"""Verifier red-team: does the verifier CATCH deliberately corrupted claims?   Usage: python -m eval.redteam <run_id>

Takes claims the verifier already approved in a finished run, corrupts each one in a known way, and measures the
fraction the verifier now rejects/flags (the 'catch rate'). This tests the CHECKER itself, not just the writer.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

from deepdive import config
from deepdive import verify as V
from deepdive.store import Store


def number_shift(t):
    """Attack 1: triple the first number in the claim (e.g. 120 -> 360)."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", t)
    if not m:
        return None
    v = float(m.group(0).replace(",", ""))
    return t[:m.start()] + f"{v * 3:g}" + t[m.end():]


def hedge_upgrade(t):
    """Attack 2: turn a hedge into a certainty ('may' -> 'will')."""
    n = re.sub(r"\b(may|might|could|is expected to|are expected to|is projected to)\b", "will", t, count=1)
    return n if n != t else None


def fabricated_clause(t):
    """Attack 3: append an invented causal clause that no source states."""
    return t.rstrip(".") + ", which caused global demand to fall by 40% in 2031."


def negation(t):
    """Attack 4: negate the claim ('is' -> 'is not')."""
    n = re.sub(r"\b(is|are|was|were|has|have)\b", r"\1 not", t, count=1)
    return n if n != t else None


ATTACKS = {"number_shift": number_shift, "hedge_upgrade": hedge_upgrade,
           "fabricated_clause": fabricated_clause, "negation": negation}


def main(run_id):
    """Corrupt verified claims, run them through the same gates + judge, and print the catch rate per attack."""
    store = Store(config.DB_PATH)
    rows, stats = [], {k: [0, 0] for k in ATTACKS}
    for c in store.claims(run_id, "verified"):  # start from claims known to be genuinely supported
        for name, fn in ATTACKS.items():
            bad = fn(c["text"])
            if bad:
                rows.append({"id": len(rows), "attack": name, "claim": bad, "orig": c})
    for r in rows:  # deterministic gates first (free), exactly like the real pipeline
        ev = store.evidence_by_id(r["orig"]["evidence_id"])
        r["caught"] = (not V.quote_in_passage(r["orig"]["quote"], ev["text"])) or bool(
            V.ungrounded_numbers(r["claim"], ev["text"]))
        r["passage"] = ev["text"]
    pending = [r for r in rows if not r["caught"]]  # only what slipped past the gates costs LLM tokens
    res = V.judge([{"id": r["id"], "claim": r["claim"], "passage": r["passage"]} for r in pending])
    for r in pending:
        r["caught"] = res[r["id"]][0] != "supported"
    for r in rows:
        stats[r["attack"]][0] += r["caught"]
        stats[r["attack"]][1] += 1
    out = ["| attack | corrupted | caught | catch rate |", "|---|---|---|---|"]
    for k, (c, n) in stats.items():
        out.append(f"| {k} | {n} | {c} | {c / n:.2f} |" if n else f"| {k} | 0 | 0 | n/a |")
    tot_c, tot_n = sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())
    out.append(f"| **all** | {tot_n} | {tot_c} | {tot_c / max(1, tot_n):.2f} |")
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    open(os.path.join(config.EVAL_DIR, "REDTEAM.md"), "w").write("\n".join(out))
    json.dump({"run_id": run_id, "attacks": {k: {"corrupted": n, "caught": c} for k, (c, n) in stats.items()},
               "total_corrupted": tot_n, "total_caught": tot_c, "models": config.CHAINS["verifier"],
               "generated_at": datetime.now(timezone.utc).isoformat()}, open(os.path.join(config.EVAL_DIR, "redteam.json"), "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main(int(sys.argv[1]))
