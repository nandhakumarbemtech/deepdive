# Latency and scaling

## Where the time goes (one run)
`plan -> [research -> draft -> verify -> assess]x(1..3 rounds) -> conflicts -> synthesize -> audit -> assemble`

| Stage | Dominant cost | Parallel now? | Scales with |
|---|---|---|---|
| plan | 1 small LLM call | no (single call) | - |
| research | search + page downloads | yes: across sub-questions, bounded downloads, page cache | network, provider throttling |
| draft | 1 LLM call per sub-question | yes | waves = ceil(sub-questions / `PARALLEL_WORKERS`) |
| verify | claims / `JUDGE_BATCH_SIZE` judge calls | yes: batches in parallel | number of claims |
| assess | database reads | n/a | - |
| loop-back (0-2 times) | repeats research + draft + verify for thin sub-questions | yes inside each stage | rounds = `MAX_ROUNDS`; the biggest single multiplier |
| conflicts | 1 call per sub-question | yes | sub-questions |
| synthesize | 1 large call | no (single call) | report length |
| audit | sentences / batch judge calls | yes | number of sentences |

Rule of thumb: run time ~ sum over stages of (waves x per-call latency) x rounds. Measure the real values with
`python -m eval.latency_report <run_id>`; this repo does not claim real-world latency numbers.

**Simulation only** (5 sub-questions, 100 ms per model call, identical results and cost): 1 worker 3.29 s, 2 workers 2.40 s,
4 workers 1.88 s, 5 workers 1.30 s. This shows the effect of parallelism, not Mesh's real latency.

## What is in place
- Parallel per-sub-question stages (`parallel.pmap`) that keep cost accounting and propagate the first error.
- Process-wide cap on in-flight LLM requests and on page downloads (no self-inflicted rate limiting).
- Page cache (TTL + LRU), per-step timing, LLM latency percentiles, Prometheus counters at `/metrics`.
- Run deadline, cost cap and call cap; partial report instead of a lost run; resume after a stop.
- Load shedding: at most `MAX_CONCURRENT_RUNS` runs per instance, otherwise HTTP 429; per-IP rate limit.
- The dashboard streams progress, which lowers *perceived* latency even when total time is unchanged.

## Capacity of one instance
Throughput ~ `MAX_CONCURRENT_RUNS x 3600 / run_seconds` runs per hour (measure `run_seconds`). Spend per hour is bounded by
`MAX_CONCURRENT_RUNS x MAX_COST_USD_PER_RUN` per run-slot cycle. Model-request concurrency is bounded by `MAX_INFLIGHT_LLM`
regardless of how many runs are active.

## Known limits (honest)
- **SQLite + threads = one instance.** Writes are serialized by one lock; WAL mode keeps readers (the UI) fast.
- **Per-process state:** rate limiter, circuit breakers, page cache and in-flight caps are not shared across processes.
- **Search throttling:** parallel sub-questions hit the search engine together; DuckDuckGo may throttle (failover mitigates).
- **Nested concurrency:** sub-question workers x page-download workers are bounded only by `MAX_INFLIGHT_FETCH`.
- **Soft cost cap:** with parallel calls the cap can be overshot by up to `PARALLEL_WORKERS` calls already in flight.
- **Single large calls** (planner, synthesizer) cannot be parallelized; very long reports lengthen the synthesize step.

## Scale-out plan (in order of payoff)
1. **Quality/latency presets** (quick / standard / deep) that set `MAX_SUBQS`, `MAX_ROUNDS` and `MAX_COST_USD_PER_RUN`.
   The loop-back count is the largest multiplier, so a "quick" mode (0 extra rounds) is the cheapest latency win.
2. **Queue + workers:** API enqueues runs; stateless workers consume (Celery/RQ/Arq or SQS). Runs are already resumable,
   so a crashed worker's run can be re-queued with `resume`.
3. **Postgres** instead of SQLite (the `Store` interface is small), with row-level run ownership for multi-tenant use.
4. **Redis** for the shared rate limiter, circuit breakers, page/search cache and a global in-flight limiter.
5. **Autoscale workers on queue depth** (not CPU: the work is I/O and waiting on models); set per-tenant concurrency quotas.
6. **Provider-side:** request higher rate limits, use prompt caching if the gateway supports it, and keep a second model
   family warm for fallback.
7. **Observability:** export `/metrics` to Prometheus; alert on p95 run time, failed_calls, breaker-open and cost per run.
