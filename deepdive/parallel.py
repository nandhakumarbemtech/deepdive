"""Tiny helper for bounded parallelism that keeps the per-run accounting context.

Why not a bare ThreadPoolExecutor: the 'current run' (cost/budget accounting) lives in a ContextVar, and threads
do not inherit it automatically. Each task therefore runs inside a COPY of the caller's context. Results come back
in input order and the first exception is re-raised, so budget/verifier errors still stop the run exactly as they
do in sequential code.
"""
import contextvars
from concurrent.futures import ThreadPoolExecutor


def pmap(fn, items, workers):
    """Apply fn to every item, up to `workers` at a time; returns results in input order.

    workers <= 1 (or a single item) runs sequentially, which keeps behaviour identical for debugging.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with ThreadPoolExecutor(min(workers, len(items))) as ex:
        futures = [ex.submit(contextvars.copy_context().run, fn, i) for i in items]
        return [f.result() for f in futures]  # re-raises the first failure in input order
