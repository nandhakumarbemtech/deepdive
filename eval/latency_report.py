"""Where did the time go?   Usage: python -m eval.latency_report <run_id>

Prints wall-clock time per pipeline step, LLM latency percentiles and per-role call counts for a finished run, so you can
measure REAL latency (the simulated benchmark in README only shows the effect of parallelism).
"""
import sys

from deepdive import config, live
from deepdive.store import Store


def main(run_id):
    """Print a latency breakdown for one run."""
    store = Store(config.DB_PATH)
    snap = live.snapshot(store, run_id)
    usage = store.usage_summary(run_id)
    print(f"run #{run_id}  status={snap['run']['status']}  elapsed={snap['run']['elapsed_s']}s")
    print(f"{'step':28} {'state':10} {'runs':>4} {'seconds':>8}")
    for s in snap["steps"]:
        print(f"{s['label']:28} {s['state']:10} {s['runs']:>4} {s['seconds']:>8}")
    lat = usage["latency_ms"]
    print(f"\nLLM calls: {usage['llm_calls']} ({usage['failed_calls']} failed)  latency p50={lat['p50']}ms p95={lat['p95']}ms max={lat['max']}ms")
    for role, b in usage["by_role"].items():
        print(f"  {role:12} calls={b['calls']:3} tokens={b['tokens']:7} cost=${b['cost_usd']}")
    slow = sorted(snap["steps"], key=lambda s: s["seconds"], reverse=True)[:2]
    print("\nSlowest steps:", ", ".join(f"{s['label']} ({s['seconds']}s)" for s in slow))
    print("Tip: a stage dominated by LLM calls speeds up with PARALLEL_WORKERS; research is bound by network and the page cache.")


if __name__ == "__main__":
    main(int(sys.argv[1]))
