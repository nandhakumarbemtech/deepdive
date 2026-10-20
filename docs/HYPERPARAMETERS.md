# Hyperparameter decisions

Every tunable value lives in `deepdive/config.py` with an inline decision record (WHY / RISK / TUNE).
This page is the table version. **Honest status: none of these has been tuned on data yet.** Tags:

- **JUDGMENT** - reasoned default; treat as a starting point and sweep it with the eval harness.
- **SAFETY** - deliberate guardrail; keep conservative even if it costs some recall.
- **PROVIDER** - dictated by an external limit (API, site behaviour, price list).

| Parameter | Default | Why this value | Too low / too high | Tune with | Tag |
|---|---|---|---|---|---|
| `TEMPERATURE` verifier, auditor | 0.0 | same claim must give the same verdict | higher: verdicts flip between runs | repeat a question 3x; red-team catch rate | JUDGMENT |
| `TEMPERATURE` writer | 0.1 | exact extraction, tiny variation harmless | higher: paraphrase drift | first_pass_groundedness | JUDGMENT |
| `TEMPERATURE` planner, researcher | 0.3 | diverse queries find more sources | 0: repeated queries return the same pages | coverage, sources_cited | JUDGMENT |
| `JUDGE_BATCH_SIZE` | 5 | ~5x fewer calls, each item still in focus | >8 judge skims; 1 is most accurate but 5x calls | red-team catch rate at 1 / 5 / 8; cost_usd | JUDGMENT |
| `FUZZY_QUOTE_THRESHOLD` | 0.90 | tolerates HTML/whitespace noise, rejects paraphrase | 0.8 lets reworded quotes pass; 1.0 rejects true quotes | red-team fabricated_clause + false-rejection rate | JUDGMENT |
| `MIN_QUOTE_WORDS` | 4 | a shorter quote matches anywhere | lower: gate proves nothing | - | SAFETY |
| `PASSAGE_CHARS` | 900 | qualifiers stay next to the number; 10 passages fit the prompt | small: more "partial" verdicts; large: diluted, costly | first_pass_groundedness, cost_usd | JUDGMENT |
| `PASSAGES_PER_PAGE` | 3 | covers a long article without one site dominating | 1 misses facts; >5 floods the ranking | coverage | JUDGMENT |
| `MIN_PASSAGE_CHARS` | 200 | shorter fragments are menus/captions | lower: junk passages | - | JUDGMENT |
| `EVIDENCE_PER_SUBQ` | 10 | ~2.5k prompt tokens, enough for corroboration | low: thin; high: cost | coverage vs cost_usd | JUDGMENT |
| `EVIDENCE_PER_SOURCE` | 3 | forces corroboration across sites | 1: too strict; large: one site dominates | sources_cited | JUDGMENT |
| `QUALITY_RANK_WEIGHT` | 0.5 | a primary source beats a slightly more relevant blog | 0: ignores quality; 1+: ignores relevance | spot-check cited sources | JUDGMENT |
| `QUALITY` weights | 1.0 / 0.85 / 0.8 / 0.65 / 0.5 / 0.3 | primary > uploads > news > org > unknown > user-generated | heuristic by domain; a great blog scores low | spot-check | JUDGMENT |
| `PARALLEL_WORKERS` | 5 | = MAX_SUBQS so every sub-question runs in one wave; simulated benchmark 3.29 s (1 worker) -> 1.30 s (5) | >5 gains nothing; more concurrency means more 429s | run wall-clock, failed_calls | MEASURED (simulation only) |
| `MAX_INFLIGHT_LLM` | 6 | process-wide cap; typical gateway tiers allow ~5-10 concurrent requests | low wastes parallelism; high causes self-inflicted 429s | failed_calls stays 0 | PROVIDER |
| `MAX_INFLIGHT_FETCH` | 12 | sub-question workers x fetch workers would reach 30 downloads | high: sites block | fetch_total{result=failed} | PROVIDER |
| `FETCH_CACHE_TTL_S` / `MAX` | 3600 s / 256 pages | same URL recurs across sub-questions and repeat questions; pages change slowly within a session | long TTL serves stale pages (0 disables) | fetch_total{result=cache} | JUDGMENT |
| `MAX_SUBQS` | 5 | typical decomposition (size, players, risks, events, counter-arguments) | <3 misses angles; >6 dilutes budget | coverage vs cost_usd | JUDGMENT |
| `QUERIES_PER_SUBQ` | 3 | recall without many search calls | 1: brittle | sources_cited | JUDGMENT |
| `MAX_ROUNDS` | 2 | first loop fixes most gaps, second catches stragglers | 0: no self-correction; >3: diminishing returns | coverage, research_rounds, cost_usd | JUDGMENT |
| `MIN_VERIFIED` / `MIN_SOURCES` | 3 / 2 | smallest bar giving corroboration | higher: more loops and "limited" sections | coverage, cost_usd | JUDGMENT |
| `RESULTS_PER_QUERY` | 6 | below: recall; above: ranking noise | - | sources_cited | JUDGMENT |
| `MAX_PAGES_PER_ROUND` | 8 | bounds fetch time and LLM tokens | low: thin; high: slow, costly | coverage, seconds | JUDGMENT |
| `FETCH_WORKERS` | 6 | I/O-bound parallelism | high: sites rate-limit or block | fetch_failed count | PROVIDER |
| `PAGE_TEXT_CAP` | 60000 chars | relevant text is almost always early | - | - | SAFETY |
| `LLM_TIMEOUT_S` | 60 | long synthesis can take 20-40s | low: aborts valid answers; high: hung call blocks a worker | p95 latency_ms | JUDGMENT |
| `LLM_MAX_RETRIES` | 2 | most blips clear by retry 1-2 | high: slow failover in a real outage | failed_calls | JUDGMENT |
| `BACKOFF_BASE_S` / `CAP` | 1s / 20s (+-50% jitter) | respects rate limits, avoids lock-step retries | cap too high stalls runs | - | JUDGMENT |
| `BREAKER_THRESHOLD` / `COOLDOWN` | 4 / 60s | more than one whole failed call; matches typical incident blips | low: flaps; long cooldown: recovered model unused | breaker events | JUDGMENT |
| `MAX_COST_USD_PER_RUN` | $1.50 | ~3x expected typical run | low: partial reports; high: runaway risk | p95 cost_usd x 2 | SAFETY |
| `MAX_LLM_CALLS_PER_RUN` | 250 | normal runs make 30-60; catches loops | low stops real runs | llm_calls | SAFETY |
| `RUN_TIMEOUT_S` | 900 | search + fetch + ~50 sequential LLM calls fit | - | seconds | SAFETY |
| `BUDGET_STOP_FRACTION` | 0.7 | keep budget to verify/synthesize/audit what was gathered | high: partial runs; low: thin coverage | status=partial rate | JUDGMENT |
| `GRAPH_RECURSION_LIMIT` | 60 | max ~17 node executions per pass x3 headroom | - | - | SAFETY |
| `SEARCH_PROVIDERS` | ddgs, brave, tavily, wikipedia | free first, keyed APIs next, always-up last resort | - | search_calls_total by provider | JUDGMENT |
| `SEARCH_TIMEOUT_S` | 10 | APIs answer in <2s; 10s means stuck | - | - | JUDGMENT |
| `FETCH_TIMEOUT_S` | 12 | most pages <3s | - | fetch_total{result=failed} | JUDGMENT |
| `MAX_FETCH_BYTES` / `MAX_REDIRECTS` | 2 MB / 3 | memory and DoS guard; real sites need <=2 hops | - | - | SAFETY |
| `MAX_CONCURRENT_RUNS` | 2 | bounds worst-case spend and provider rate pressure | high needs a queue + global limiter | - | SAFETY |
| `RATE_LIMIT_PER_MIN` | 10 per IP | humans start few runs per minute | - | - | SAFETY |
| `MIN/MAX_QUESTION_CHARS` | 10 / 1000 | not a question below; prompt-stuffing above | - | - | SAFETY |
| `MAX_UPLOADS` / `MAX_UPLOAD_BYTES` | 3 / 1 MB | bounds prompt cost and parsing risk | - | - | SAFETY |
| `LLM_IO_MAX_CHARS` | 20000 | bounds DB growth per stored prompt | - | DB size | SAFETY |
| Prompt limits: claim <= ~35 words, quote 8-30 words, 4-5 sub-questions, 3-5 summary sentences | see `agents.py` | atomic claims are fully supportable by one passage | long claims bundle facts and get "partial" | first_pass_groundedness | JUDGMENT |

## How to tune (one parameter at a time)
1. Run `python -m eval.run_eval` and `python -m eval.redteam <run_id>`; save both tables as the baseline.
2. Change ONE value via environment variable or `config.py`; re-run; compare first_pass_groundedness,
   independent_groundedness, coverage, cost_usd and the red-team catch rate.
3. Keep a change only if quality is equal or better at equal or lower cost. Record the result in this table
   (replace the tag JUDGMENT with the measured numbers).
4. Never loosen a SAFETY value to improve a metric; fix the cause instead.
