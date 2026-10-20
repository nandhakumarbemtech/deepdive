"""LLM gateway layer: the ONLY place that talks to Mesh.

Production behaviours implemented here:
  * timeouts and bounded retries with exponential backoff + jitter (transient errors only)
  * ordered model fallback chain per agent role (handles a model being down)
  * per-model circuit breaker (stops hammering a failing model, auto-recovers after a cooldown)
  * token and USD cost accounting for every call, stored in the database
  * per-run guardrails: cost cap, call cap and wall-clock deadline
  * classification of errors: fatal (bad key) vs transient (retry) vs model-unavailable (fall back)
"""
import contextvars
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field

from openai import OpenAI

from . import config, metrics

log = logging.getLogger("deepdive.llm")


# ---------------------------------------------------------------- exceptions
class LLMError(Exception):
    """Base class for every error raised by this module."""


class FatalLLMError(LLMError):
    """Authentication/permission problem: retrying or falling back cannot help."""


class AllModelsFailed(LLMError):
    """Every model in the role's chain failed (outage, breakers open, or persistent errors)."""


class VerifierUnavailable(LLMError):
    """Verification could not be performed. The system must fail CLOSED, never skip checking."""


class BudgetExceeded(LLMError):
    """The run hit its cost or call cap."""


class RunTimeout(LLMError):
    """The run exceeded its wall-clock deadline."""


# ---------------------------------------------------------------- per-run context
@dataclass
class RunContext:
    """Mutable accounting for a single research run (cost, tokens, calls, deadline)."""

    run_id: int
    store: object = None
    max_cost: float = config.MAX_COST_USD_PER_RUN
    max_calls: int = config.MAX_LLM_CALLS_PER_RUN
    deadline: float = 0.0
    started: float = field(default_factory=time.time)
    cost: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


# A ContextVar keeps the "current run" without threading it through every function signature.
_ctx = contextvars.ContextVar("deepdive_run_ctx", default=None)


def begin_run(run_id, store=None, max_cost=None, max_calls=None, timeout=None, start_cost=0.0, start_calls=0):
    """Start accounting for a run and make it the current context.

    start_cost/start_calls carry spend from earlier attempts when a run is RESUMED, so the cap covers the whole run.
    """
    ctx = RunContext(run_id=run_id, store=store,
                     max_cost=max_cost if max_cost is not None else config.MAX_COST_USD_PER_RUN,
                     max_calls=max_calls if max_calls is not None else config.MAX_LLM_CALLS_PER_RUN)
    ctx.deadline = ctx.started + (timeout if timeout is not None else config.RUN_TIMEOUT_S)
    ctx.cost, ctx.calls = start_cost, start_calls
    _ctx.set(ctx)
    return ctx


def end_run():
    """Clear the current run context."""
    _ctx.set(None)


def budget_used_fraction():
    """Largest fraction used across cost, calls and time (0 when no run is active)."""
    ctx = _ctx.get()
    if not ctx:
        return 0.0
    return max(ctx.cost / ctx.max_cost if ctx.max_cost else 0,
               ctx.calls / ctx.max_calls if ctx.max_calls else 0,
               (time.time() - ctx.started) / max(1.0, ctx.deadline - ctx.started))


def _enforce(ctx):
    """Raise before a call if the run is out of budget or time (cheap guard, called each attempt)."""
    if not ctx:
        return
    if time.time() > ctx.deadline:
        raise RunTimeout(f"run {ctx.run_id} exceeded its deadline")
    if ctx.calls >= ctx.max_calls:
        raise BudgetExceeded(f"call cap {ctx.max_calls} reached")
    if ctx.cost >= ctx.max_cost:
        raise BudgetExceeded(f"cost cap ${ctx.max_cost:.2f} reached (spent ${ctx.cost:.4f})")


# ---------------------------------------------------------------- cost
def cost_usd(model, prompt_tokens, completion_tokens):
    """Compute USD cost from the price table; unknown models use a pessimistic default price."""
    price_in, price_out = config.PRICES.get(model, config.DEFAULT_PRICE)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


# ---------------------------------------------------------------- circuit breaker
class Breaker:
    """Opens after N consecutive failures; lets one trial call through after the cooldown (half-open)."""

    def __init__(self):
        self.fails, self.opened_at, self.lock = 0, 0.0, threading.Lock()

    def allow(self):
        with self.lock:
            if self.fails < config.BREAKER_THRESHOLD:
                return True
            return time.time() - self.opened_at >= config.BREAKER_COOLDOWN_S  # half-open trial

    def ok(self):
        with self.lock:
            self.fails = 0

    def bad(self):
        with self.lock:
            self.fails += 1
            if self.fails >= config.BREAKER_THRESHOLD:
                self.opened_at = time.time()


_breakers, _blk = {}, threading.Lock()
# Process-wide cap on simultaneous LLM requests (see config.MAX_INFLIGHT_LLM for the reasoning).
_inflight = threading.BoundedSemaphore(config.MAX_INFLIGHT_LLM)


def _breaker(model):
    """Get (or create) the breaker for a model."""
    with _blk:
        return _breakers.setdefault(model, Breaker())


def breaker_states():
    """Return {model: 'open'|'closed'} for the readiness endpoint."""
    with _blk:
        return {m: ("open" if b.fails >= config.BREAKER_THRESHOLD else "closed") for m, b in _breakers.items()}


def reset_breakers():
    """Forget all breaker state (used by tests)."""
    with _blk:
        _breakers.clear()


# ---------------------------------------------------------------- error classification
def _status(e):
    """HTTP status carried by an SDK exception, if any."""
    return getattr(e, "status_code", None)


def _is_fatal(e):
    """401/403 mean our credentials are wrong; no model can fix that."""
    return _status(e) in (401, 403)


def _is_transient(e):
    """Timeouts, connection problems, rate limits and 5xx are worth retrying."""
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    if type(e).__name__ in ("APITimeoutError", "APIConnectionError", "RateLimitError"):
        return True
    s = _status(e)
    return s in (408, 409, 425, 429) or (s is not None and s >= 500)


def _sleep(seconds):
    """Indirection so tests can skip real waiting."""
    time.sleep(seconds)


def _backoff(attempt):
    """Exponential backoff with jitter, capped."""
    return min(config.BACKOFF_CAP_S, config.BACKOFF_BASE_S * (2 ** attempt)) * random.uniform(0.5, 1.5)


# ---------------------------------------------------------------- raw call
_client = None


def client():
    """Lazily build the OpenAI SDK client pointed at Mesh. SDK retries are disabled: we retry ourselves."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.MESH_BASE_URL, api_key=config.MESH_API_KEY,
                         timeout=config.LLM_TIMEOUT_S, max_retries=0)
    return _client


def _raw_call(model, system, user, json_mode, temperature=0.1):
    """One HTTP call to Mesh. Returns (text, prompt_tokens, completion_tokens, usage_was_estimated)."""
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = client().chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
    text = r.choices[0].message.content or ""
    u = getattr(r, "usage", None)
    if u and getattr(u, "prompt_tokens", None) is not None:
        return text, u.prompt_tokens, u.completion_tokens or 0, False
    # Gateway did not report usage: estimate ~4 characters per token and flag it as an estimate.
    return text, (len(system) + len(user)) // 4, len(text) // 4, True


def _record(ctx, role, model, pt, ct, latency_ms, ok, estimated, attempt, fallback, error="", io=None):
    """Charge the run and persist one usage row (also feeds /metrics)."""
    c = cost_usd(model, pt, ct) if ok else 0.0
    metrics.inc("llm_calls_total", model=model, ok=str(ok).lower())
    if ok:
        metrics.inc("llm_cost_usd_total", c, model=model)
        metrics.inc("llm_tokens_total", pt + ct, model=model)
    if fallback:
        metrics.inc("llm_fallbacks_total", role=role)
    if not ctx:
        return
    with ctx.lock:
        ctx.calls += 1
        ctx.cost += c
        ctx.prompt_tokens += pt
        ctx.completion_tokens += ct
    if ctx.store:
        uid = ctx.store.add_usage(ctx.run_id, role, model, pt, ct, c, latency_ms, ok, estimated, attempt, fallback, error)
        if io and config.STORE_LLM_IO:  # keep the exact prompt and answer so every generated text can be read back
            cap = config.LLM_IO_MAX_CHARS
            ctx.store.add_llm_io(ctx.run_id, uid, role, model, *[(x[:cap] + " …[truncated]") if len(x) > cap else x for x in io])


def call(role, system, user, json_mode=False, temperature=None):
    """Call the best available model for `role`, with retries, fallback, breaker and accounting."""
    ctx = _ctx.get()
    last = None
    if temperature is None:  # per-role default (verifier 0.0 ... planner 0.3); rationale in config.TEMPERATURE
        temperature = config.TEMPERATURE.get(role, config.DEFAULT_TEMPERATURE)
    for idx, model in enumerate(config.chain(role)):
        breaker = _breaker(model)
        if not breaker.allow():
            log.warning("breaker open, skipping model", extra={"model": model, "role": role})
            metrics.inc("breaker_skips_total", model=model)
            continue
        use_json = json_mode
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            _enforce(ctx)
            t0 = time.time()
            try:
                with _inflight:  # wait for a free slot instead of provoking provider rate limits
                    text, pt, ct, est = _raw_call(model, system, user, use_json, temperature)
            except Exception as e:  # noqa: BLE001 - classified below
                last = e
                ms = int((time.time() - t0) * 1000)
                _record(ctx, role, model, 0, 0, ms, False, False, attempt, idx > 0, f"{type(e).__name__}: {e}"[:200])
                if _is_fatal(e):
                    raise FatalLLMError(f"credentials rejected by gateway: {e}") from e
                if _status(e) == 400 and use_json:
                    use_json = False  # some models reject response_format; retry once without it
                    continue
                breaker.bad()
                if _is_transient(e) and attempt < config.LLM_MAX_RETRIES:
                    _sleep(_backoff(attempt))
                    continue
                break  # persistent failure or retries exhausted: move to the next model in the chain
            breaker.ok()
            _record(ctx, role, model, pt, ct, int((time.time() - t0) * 1000), True, est, attempt, idx > 0, io=(system, user, text))
            return text
    raise AllModelsFailed(f"all models failed for role '{role}': {last}")


# ---------------------------------------------------------------- JSON helpers
def parse_json(text):
    """Parse model output as JSON, tolerating code fences and surrounding prose. Returns {} on failure."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    return {}


def chat(role, system, user, json_mode=False, temperature=None):
    """Plain text completion for a role."""
    return call(role, system, user, json_mode=json_mode, temperature=temperature)


def chat_json(role, system, user):
    """JSON completion for a role, with one corrective retry if the model ignores the format."""
    data = parse_json(call(role, system, user, json_mode=True))
    if not data:
        nudge = system + "\n\nIMPORTANT: respond with a single valid JSON object and nothing else."
        data = parse_json(call(role, nudge, user, json_mode=True))
    return data
