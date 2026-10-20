"""Central configuration. Everything tunable lives here and can be overridden by environment variables
(12-factor style), so the same code runs unchanged on a laptop, in CI and in production.

HOW TO READ THE HYPERPARAMETER COMMENTS
Each tunable value carries a short decision record:
  WHY   - the reasoning behind the default
  RISK  - what goes wrong if it is set too low / too high
  TUNE  - which self-evaluation metric to watch (python -m eval.run_eval, python -m eval.redteam)
and a tag saying how much evidence stands behind it:
  [JUDGMENT]  reasoned default, NOT yet tuned on data - treat as a starting point and sweep it
  [SAFETY]    deliberate guardrail; keep conservative even if it costs some recall
  [PROVIDER]  dictated by an external limit (API, model context, website behaviour)
A table version of this lives in docs/HYPERPARAMETERS.md.
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()  # read a local, git-ignored .env file if present


def _env_float(name, default):
    """Read a float from the environment, falling back to a default."""
    return float(os.getenv(name, default))


def _env_int(name, default):
    """Read an int from the environment, falling back to a default."""
    return int(os.getenv(name, default))


def _env_bool(name, default):
    """Read a boolean flag (1/true/yes/on) from the environment."""
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _chain(name, default):
    """Parse a comma-separated list. For model chains the first entry is preferred, the rest are fallbacks."""
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


# =============================================================================================
# Mesh API: the mandatory gateway for every LLM call
# =============================================================================================
MESH_BASE_URL = os.getenv("MESH_BASE_URL", "https://api.meshapi.ai/v1")
MESH_API_KEY = os.getenv("MESH_API_KEY")

# =============================================================================================
# LLM FAILOVER: ordered model chain per agent role
# =============================================================================================
# WHY: a cheap model for planning/query-writing (low stakes, easy task), a strong model for drafting and
#      synthesis, and a DIFFERENT FAMILY for verification so the checker does not share the writer's blind
#      spots. Each chain ends with a second model so one model outage never stops a run.
# RISK: if the verifier falls back to the writer's family, independence weakens. That is recorded in the run
#      metrics (`fallbacks_used`) instead of being hidden.  [JUDGMENT]
# TUNE: compare independent_groundedness and cost_usd in eval/RESULTS.md after swapping models.
CHAINS = {
    "planner": _chain("MODELS_PLANNER", "openai/gpt-4o-mini,openai/gpt-4o"),
    "researcher": _chain("MODELS_RESEARCHER", "openai/gpt-4o-mini,openai/gpt-4o"),
    "writer": _chain("MODELS_WRITER", "openai/gpt-4o,openai/gpt-4o-mini"),
    "verifier": _chain("MODELS_VERIFIER", "anthropic/claude-sonnet-4.5,openai/gpt-4o"),
    "auditor": _chain("MODELS_AUDITOR", "openai/gpt-4o,anthropic/claude-sonnet-4.5"),
}


def chain(role):
    """Return the ordered list of models for an agent role."""
    return CHAINS[role]


# Sampling temperature per role.
# WHY: verification and auditing must be as deterministic as possible (same claim -> same verdict), so 0.0.
#      The writer extracts/paraphrases facts, where small variation is harmless but exactness matters: 0.1.
#      Planner/researcher write search queries, where a little diversity finds more sources: 0.3.
# RISK: higher temperature on the verifier makes verdicts flip between runs (hurts reproducibility and the
#      red-team catch rate); 0.0 on the planner makes repeated queries return the same pages.  [JUDGMENT]
# TUNE: run the same question 3x and compare verdict stability; red-team catch rate for the verifier.
TEMPERATURE = {"planner": 0.3, "researcher": 0.3, "writer": 0.1, "verifier": 0.0, "auditor": 0.0}
DEFAULT_TEMPERATURE = 0.1

# --- Pricing (USD per 1M tokens: input, output) ---------------------------------------------------
# These are APPROXIMATE list prices used for budgeting. Verify against the Mesh dashboard and override with
# PRICES_JSON='{"openai/gpt-4o":[2.5,10.0]}' if they differ.  [PROVIDER]
PRICES = {
    "openai/gpt-4o": (2.5, 10.0),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-sonnet-4.5": (3.0, 15.0),
}
PRICES.update({k: tuple(v) for k, v in json.loads(os.getenv("PRICES_JSON", "{}")).items()})
# WHY pessimistic: an unknown model priced too LOW would let spend slip past the budget unnoticed; priced too HIGH
#      it only trips the guard early. RISK: a cheap unknown model looks expensive.  [SAFETY]
DEFAULT_PRICE = (5.0, 15.0)

# =============================================================================================
# RELIABILITY
# =============================================================================================
# Per-request timeout. WHY 60s: long synthesis calls can legitimately take 20-40s; beyond 60s the gateway is
# almost certainly stuck and a retry/fallback is faster. RISK: too low aborts valid long answers, too high lets a
# hung call block a worker. TUNE: p95 latency_ms in the usage table.  [JUDGMENT]
LLM_TIMEOUT_S = _env_float("LLM_TIMEOUT_S", 60)
# Retries per model for TRANSIENT errors (429/5xx/timeouts). WHY 2: most blips clear on the first or second retry;
# a third retry rarely helps and delays the fallback model. RISK: higher = slower failover during a real outage.  [JUDGMENT]
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)
# Backoff: wait base*2^attempt seconds (+/-50% jitter), capped. WHY: exponential spacing respects rate limits;
# jitter prevents parallel runs from retrying in lock-step. RISK: cap too high stalls a run.  [JUDGMENT]
BACKOFF_BASE_S = _env_float("BACKOFF_BASE_S", 1.0)
BACKOFF_CAP_S = _env_float("BACKOFF_CAP_S", 20.0)
# Circuit breaker: open after N consecutive failures, retry after the cooldown. WHY 4: with 2 retries a single
# bad call makes 3 failures, so 4 means "this model failed more than one whole call" before we stop trying it.
# Cooldown 60s matches typical provider incident blips. RISK: threshold too low flaps on noise; cooldown too
# long keeps a recovered model unused.  [JUDGMENT]
BREAKER_THRESHOLD = _env_int("BREAKER_THRESHOLD", 4)
BREAKER_COOLDOWN_S = _env_float("BREAKER_COOLDOWN_S", 60)

# =============================================================================================
# PER-RUN GUARDRAILS (cost control and runaway protection)  [SAFETY]
# =============================================================================================
# $1.50 per run: a typical run is expected to cost well under $0.50 (see README cost model); the cap is ~3x that
# so legitimate deep runs finish but a loop bug cannot burn money. RISK: too low truncates good runs into
# partial reports. TUNE: set to ~2x the p95 cost_usd of eval/RESULTS.md.
MAX_COST_USD_PER_RUN = _env_float("MAX_COST_USD_PER_RUN", 1.50)
# 250 calls: a normal run makes ~30-60 calls; this only catches runaway loops. RISK: low values stop real runs.
MAX_LLM_CALLS_PER_RUN = _env_int("MAX_LLM_CALLS_PER_RUN", 250)
# 15 minutes wall-clock: search + fetch + ~50 sequential LLM calls fit comfortably; beyond that something is wrong.
RUN_TIMEOUT_S = _env_float("RUN_TIMEOUT_S", 900)
# Stop STARTING new research rounds at 70% of any budget so there is always budget left to verify, synthesize and
# audit what was already gathered (research without verification is wasted spend). RISK: too high -> runs end
# partial; too low -> loops end early with thin coverage.  [JUDGMENT]
BUDGET_STOP_FRACTION = 0.7

# =============================================================================================
# RESEARCH BEHAVIOUR
# =============================================================================================
# Sub-questions: 5 covers the typical decomposition (size, players, risks, recent events, counter-arguments).
# More sub-questions multiply cost linearly (each gets its own search/draft/verify). RISK: <3 misses angles,
# >6 dilutes budget. TUNE: coverage vs cost_usd.  [JUDGMENT]
MAX_SUBQS = 5
QUERIES_PER_SUBQ = 3          # diverse queries per sub-question; 3 balances recall against search calls
# Extra research loops after the first pass. WHY 2: the first loop fixes most thin areas (new queries aimed at the
# gap); a second catches stragglers; beyond that returns diminish while cost grows. TUNE: coverage, research_rounds.  [JUDGMENT]
MAX_ROUNDS = 2
# A sub-question is "covered" with >= 3 verified claims from >= 2 distinct sources. WHY: one claim is an anecdote,
# two from one site is one opinion; 3 claims / 2 sources is the smallest bar that gives corroboration without
# demanding what thin topics cannot supply. RISK: raising these makes more sub-questions "limited" and triggers more
# loops (cost). TUNE: coverage and cost_usd.  [JUDGMENT]
MIN_VERIFIED = 3
MIN_SOURCES = 2
RESULTS_PER_QUERY = 6         # top-6 hits per query: below that recall suffers, above it low-ranked noise dominates
MAX_PAGES_PER_ROUND = 8       # pages fetched per sub-question per round; bounds fetch time and later LLM tokens
FETCH_WORKERS = 6             # parallel page downloads (I/O bound); more risks rate-limits/blocks from sites  [PROVIDER]

# --- Latency and concurrency ------------------------------------------------------------------
# Sub-questions are independent, so research, drafting, judging and conflict checks run in parallel across them.
# PARALLEL_WORKERS = how many tasks at once inside one stage. WHY 5 = MAX_SUBQS: all sub-questions fit in ONE wave. A
# SIMULATED benchmark (5 sub-questions, 100 ms per model call, identical results and cost) gave 3.29 s / 2.40 s / 1.88 s /
# 1.30 s for 1 / 2 / 4 / 5 workers: 4 workers needs two waves, 5 needs one. Real gateway latency will differ - re-measure
# with eval/latency_report.py. RISK: more concurrent requests means more 429s (bounded by MAX_INFLIGHT_LLM below);
# 1 = fully sequential (debugging). Keep it >= MAX_SUBQS or accept a second wave.
# TUNE: per-step seconds in the dashboard, failed_calls (429s) in /cost, run wall-clock in eval/RESULTS.md.  [MEASURED in simulation, JUDGMENT for real latency]
PARALLEL_WORKERS = _env_int("PARALLEL_WORKERS", 5)
# Hard cap on simultaneous LLM requests for the WHOLE process (all runs together): MAX_CONCURRENT_RUNS x PARALLEL_WORKERS
# could otherwise reach 8. WHY 6: typical gateway tiers allow roughly 5-10 concurrent requests; staying under that avoids
# self-inflicted 429s that would trigger retries (slower than waiting for a slot). RISK: too low wastes the parallelism,
# too high triggers rate limits. TUNE: raise only if failed_calls stays 0.  [PROVIDER]
MAX_INFLIGHT_LLM = _env_int("MAX_INFLIGHT_LLM", 6)
MAX_INFLIGHT_FETCH = _env_int("MAX_INFLIGHT_FETCH", 12)   # simultaneous page downloads: sub-question workers x FETCH_WORKERS would reach 24  [PROVIDER]
# Page cache: the same URL is often returned for several sub-questions and for repeated questions. Cached pages skip the
# network entirely. WHY 1 hour: news/market pages change slowly relative to a research session; 256 pages ~ tens of MB.
# RISK: long TTL serves stale pages (set 0 to disable). TUNE: fetch_total{result="cache"} hit rate.  [JUDGMENT]
FETCH_CACHE_TTL_S = _env_float("FETCH_CACHE_TTL_S", 3600)
FETCH_CACHE_MAX = _env_int("FETCH_CACHE_MAX", 256)

# --- Passage extraction (what the drafter and verifier actually read) -----------------------------
# 900 characters ~ 150-200 words ~ 1-2 paragraphs. WHY: long enough that a claim's qualifiers ("may", "in 2024",
# "according to ...") stay next to the number; short enough that 10 passages fit in the drafter prompt and the
# verifier judges a focused text. RISK: too small loses context (more "partial" verdicts); too large dilutes
# relevance and raises token cost. TUNE: first_pass_groundedness and cost_usd.  [JUDGMENT]
PASSAGE_CHARS = 900
MIN_PASSAGE_CHARS = 200       # shorter fragments (menus, captions) rarely contain a verifiable fact
PASSAGES_PER_PAGE = 3         # best-3 passages per page: enough for a long article, avoids one site dominating
PAGE_TEXT_CAP = 60000         # characters kept per page; the relevant part is almost always in the first 60k  [SAFETY]
# Drafter input: top-10 passages per sub-question, at most 3 from one source (diversity beats depth: it forces
# corroboration across sites). The 0.5 weight means source quality counts about as much as half a relevance point so
# a primary source beats a slightly more relevant blog. RISK: 10 passages ~ 2.5k tokens; raise only if coverage is low.  [JUDGMENT]
EVIDENCE_PER_SUBQ = 10
EVIDENCE_PER_SOURCE = 3
QUALITY_RANK_WEIGHT = 0.5

# --- Source-quality weights (see sources.quality) -----------------------------------------------
# WHY these ordinals: primary/official sources are the ground truth others quote; established news/analysts add
# editorial review; unknown sites are neutral; user-generated platforms are least reliable. User documents get 0.85:
# the user chose them, but we cannot vouch for them. They only RANK evidence and break conflicts - they never
# replace verification. RISK: domain lists are a heuristic (a great blog scores 0.3). TUNE: spot-check sources_cited.  [JUDGMENT]
QUALITY = {"primary": 1.0, "upload": 0.85, "news": 0.8, "org": 0.65, "unknown": 0.5, "low": 0.3}

# =============================================================================================
# VERIFICATION
# =============================================================================================
# Claims judged per LLM call. WHY 5: batching amortises the fixed prompt cost (~5x fewer calls) while keeping each
# item's passage in focus; big batches let the judge skim and miss subtle errors. RISK: >8 degrades verdict quality,
# 1 is the most accurate but 5x the calls. TUNE: red-team catch rate at batch 1 vs 5 vs 8.  [JUDGMENT]
JUDGE_BATCH_SIZE = 5
# A quote counts as "found" if it matches a same-length window of the source with >= 0.90 string similarity (and its
# numbers match exactly). WHY: tolerates whitespace/punctuation/typo-level differences from HTML extraction while
# still rejecting paraphrase. RISK: 0.8 lets reworded (possibly distorted) quotes through; 1.0 rejects true quotes
# broken by formatting. TUNE: red-team fabricated_clause/negation catch rate vs false rejections.  [JUDGMENT]
FUZZY_QUOTE_THRESHOLD = 0.9
# A quote of fewer than 4 words proves nothing (a name or "in 2024" matches anywhere). The drafter is told to
# produce 8-30 word quotes; 4 is the floor for the gate.  [SAFETY]
MIN_QUOTE_WORDS = 4

# =============================================================================================
# ORCHESTRATION
# =============================================================================================
# LangGraph step budget. One pass = 17 node executions at most (plan, 4 loop nodes x (1+MAX_ROUNDS), 4 tail nodes);
# 60 leaves >3x headroom while still stopping a graph that loops forever.  [SAFETY]
GRAPH_RECURSION_LIMIT = 60

# =============================================================================================
# SEARCH + FETCH FAILOVER
# =============================================================================================
# Providers are tried in order; a provider with no key is skipped; each has its own circuit breaker. DuckDuckGo needs
# no key but throttles; Brave and Tavily are keyed APIs (free tiers exist); Wikipedia needs no key and is the last
# resort for general facts. WHY this order: free first, then paid-quality APIs, then a narrow but always-up source.  [JUDGMENT]
SEARCH_PROVIDERS = _chain("SEARCH_PROVIDERS", "ddgs,brave,tavily,wikipedia")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SEARCH_TIMEOUT_S = _env_float("SEARCH_TIMEOUT_S", 10)   # search APIs answer in <2s normally; 10s means it is stuck
# If a page cannot be fetched directly (403/404/5xx/timeout) try the Internet Archive's snapshot. The evidence is
# labelled "(archived copy)". RISK: an old snapshot may differ from the live page, so quality is not raised.  [JUDGMENT]
FETCH_ARCHIVE_FALLBACK = _env_bool("FETCH_ARCHIVE_FALLBACK", True)

# --- Web fetching safety ---------------------------------------------------------------------
FETCH_TIMEOUT_S = _env_float("FETCH_TIMEOUT_S", 12)       # most pages load in <3s; 12s tolerates slow sites  [JUDGMENT]
MAX_FETCH_BYTES = _env_int("MAX_FETCH_BYTES", 2_000_000)  # never read more than 2 MB per page (memory + DoS guard)  [SAFETY]
MAX_REDIRECTS = 3                                          # each hop is re-checked for SSRF; real sites need <=2  [SAFETY]

# =============================================================================================
# TRANSPARENCY: store everything the models were asked and answered
# =============================================================================================
# Keep every prompt and response so any generated text can be read back (UI "LLM calls" -> click a row, and the
# downloadable bundle). WHY on by default: auditability is the product. RISK: prompts contain fetched web text and the
# user's question - disable (STORE_LLM_IO=0) if that is sensitive; the cap bounds database growth per call.  [SAFETY]
STORE_LLM_IO = _env_bool("STORE_LLM_IO", True)
LLM_IO_MAX_CHARS = _env_int("LLM_IO_MAX_CHARS", 20000)

# =============================================================================================
# SERVICE LIMITS  [SAFETY]
# =============================================================================================
API_KEY = os.getenv("DEEPDIVE_API_KEY")                   # if set, every /api call must send X-API-Key
# 2 concurrent runs: each run can spend up to the cost cap and issues bursts of LLM calls; 2 keeps worst-case spend and
# provider rate-limit pressure bounded on one instance. RISK: higher needs a job queue and a global limiter.
MAX_CONCURRENT_RUNS = _env_int("MAX_CONCURRENT_RUNS", 2)
RATE_LIMIT_PER_MIN = _env_int("RATE_LIMIT_PER_MIN", 10)   # per IP; a human starts few runs/min, a script abusing the API starts many
MIN_QUESTION_CHARS, MAX_QUESTION_CHARS = 10, 1000         # below 10 is not a question; above 1000 is prompt-stuffing
MAX_UPLOADS, MAX_UPLOAD_BYTES = 3, 1_000_000             # 3 files x 1 MB keeps prompt cost and parsing risk small
DB_PATH = os.getenv("DEEPDIVE_DB", "deepdive.db")
EVAL_DIR = os.getenv("DEEPDIVE_EVAL_DIR", "eval")   # where run_eval / redteam write results that the UI displays


def validate():
    """Return a list of configuration problems; an empty list means the service can start."""
    problems = []
    if not MESH_API_KEY:
        problems.append("MESH_API_KEY is not set")
    elif not MESH_API_KEY.startswith("rsk_"):
        problems.append("MESH_API_KEY does not look like a Mesh key (expected prefix rsk_)")
    for role, models in CHAINS.items():
        if not models:
            problems.append(f"model chain for {role} is empty")
    if not SEARCH_PROVIDERS:
        problems.append("SEARCH_PROVIDERS is empty")
    return problems
