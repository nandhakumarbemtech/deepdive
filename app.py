"""FastAPI service: starts research runs, exposes the audit trail, exports reports, answers follow-ups.

Production features: API-key auth, input validation, upload limits, per-IP rate limiting, a concurrency cap
(protects spend and the LLM rate limit), health/readiness probes, Prometheus metrics, security headers,
JSON logs, and recovery of runs interrupted by a restart.
"""
import logging
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from deepdive import config, export, followup, graph, live, llm, logs, metrics, proof, recheck, render, sources
from deepdive.store import Store

logs.setup()
log = logging.getLogger("deepdive.app")
app = FastAPI(title="DeepDive", docs_url="/docs")
store = Store()
store.mark_stale_runs()  # runs that were mid-flight when the process died can never finish
for _problem in config.validate():
    log.warning("config problem", extra={"problem": _problem})

# Concurrency cap: at most N runs execute at once; extra requests get HTTP 429 instead of queueing forever.
_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_RUNS)
_hits = defaultdict(deque)  # client ip -> timestamps of recent requests (sliding window)
_hits_lock = threading.Lock()


class Q(BaseModel):
    """Body of a follow-up request."""

    question: str


def require_key(x_api_key: str = Header(default=None)):
    """Dependency: enforce the API key when DEEPDIVE_API_KEY is configured."""
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(401, "missing or invalid X-API-Key")


def rate_limit(request: Request):
    """Dependency: simple per-IP sliding-window limiter (use a gateway/Redis limiter for multi-instance)."""
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= config.RATE_LIMIT_PER_MIN:
            raise HTTPException(429, "rate limit exceeded; try again in a minute")
        q.append(now)


@app.middleware("http")
async def add_headers(request: Request, call_next):
    """Attach a request id and basic security headers to every response."""
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/")
def index():
    """Serve the single-page UI."""
    return FileResponse("static/index.html")


@app.get("/healthz")
def healthz():
    """Liveness probe: the process is up."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness probe: database reachable, config valid, and which model circuit breakers are open."""
    problems = config.validate()
    try:
        store.ping()
    except Exception as e:  # noqa: BLE001
        problems.append(f"database: {e}")
    body = {"ready": not problems, "problems": problems, "breakers": llm.breaker_states(),
            "search_providers": sources.provider_states()}
    return JSONResponse(body, status_code=200 if not problems else 503)


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus():
    """Prometheus metrics: LLM calls, tokens, cost, fallbacks, breaker skips, run outcomes."""
    return metrics.render()


def _run(rid):
    """Load a run or raise 404."""
    r = store.get_run(rid)
    if not r:
        raise HTTPException(404, "unknown run")
    return r


def _safe_run(rid, question, resume=False, extra_usd=None):
    """Thread target: run (or resume) the pipeline and always release the concurrency slot."""
    try:
        graph.run(store, rid, question, resume=resume, extra_usd=extra_usd)
    except Exception:  # noqa: BLE001 - graph.run already recorded the failure
        pass
    finally:
        _slots.release()


@app.post("/api/runs", dependencies=[Depends(require_key), Depends(rate_limit)])
async def start(question: str = Form(...), files: list[UploadFile] = File(default=[])):
    """Validate input, optionally ingest uploaded .txt/.md files as evidence, and start a run."""
    question = question.strip()
    if not (config.MIN_QUESTION_CHARS <= len(question) <= config.MAX_QUESTION_CHARS):
        raise HTTPException(422, f"question must be {config.MIN_QUESTION_CHARS}-{config.MAX_QUESTION_CHARS} characters")
    if len(files) > config.MAX_UPLOADS:
        raise HTTPException(422, f"at most {config.MAX_UPLOADS} files")
    uploads = []
    for f in files:
        if not (f.filename or "").lower().endswith((".txt", ".md")):
            raise HTTPException(422, "only .txt and .md uploads are supported")
        data = await f.read(config.MAX_UPLOAD_BYTES + 1)
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large")
        uploads.append((f.filename, data.decode("utf-8", errors="ignore")))
    if not _slots.acquire(blocking=False):  # no free slot: shed load instead of overspending
        raise HTTPException(429, "all research slots are busy; retry shortly")
    try:
        rid = store.create_run(question)
        for name, text in uploads:  # user documents become evidence (sub-question '*' = applies to all)
            q, why = sources.quality(f"upload://{name}")
            for p in sources.best_passages(text, question, size=900, k=12) or [text[:900]]:
                store.add_evidence(rid, "*", f"upload://{name}", name, p, q, why, 0)
        threading.Thread(target=_safe_run, args=(rid, question), daemon=True).start()
    except Exception:
        _slots.release()
        raise
    return {"run_id": rid}


@app.get("/api/runs", dependencies=[Depends(require_key)])
def list_runs():
    """Recent runs with status and cost (completed, ongoing, failed...) for the dashboard."""
    return store.list_runs(20)


@app.get("/api/runs/{rid}/live", dependencies=[Depends(require_key)])
def live_view(rid: int, since: int = 0):
    """Everything the dashboard needs in ONE cheap call: agent states, current actions, counters, new log lines."""
    _run(rid)
    return live.snapshot(store, rid, since)


@app.get("/api/runs/{rid}", dependencies=[Depends(require_key)])
def status(rid: int):
    """Run status, metrics, report (when done) and the full audit trail."""
    r = _run(rid)
    return {"status": r["status"], "error": r["error"], "question": r["question"], "report": r["report"],
            "metrics": r["metrics"], "trail": store.trail(rid)}


@app.get("/api/runs/{rid}/cost", dependencies=[Depends(require_key)])
def cost(rid: int):
    """Token and USD cost for a run, by agent role and by model."""
    _run(rid)
    return store.usage_summary(rid)


@app.get("/api/runs/{rid}/claims", dependencies=[Depends(require_key)])
def claims(rid: int):
    """Every claim with its verdict, deciding stage, reason, verbatim quote and the source it cites."""
    _run(rid)
    return export.claims_rows(store, rid)


@app.get("/api/runs/{rid}/evidence", dependencies=[Depends(require_key)])
def evidence(rid: int):
    """Every retrieved source passage with its quality weight and the claims that cite it."""
    _run(rid)
    return export.evidence_rows(store, rid)


@app.get("/api/runs/{rid}/log.txt", response_class=PlainTextResponse, dependencies=[Depends(require_key)])
def log_txt(rid: int):
    """The full audit trail as a readable plain-text log."""
    _run(rid)
    return export.log_text(store, rid)


@app.get("/api/runs/{rid}/report.json", dependencies=[Depends(require_key)])
def report_json(rid: int):
    """The structured report (sentences with claim ids and citations, sources, conflicts, metrics)."""
    r = _run(rid)
    if not r["report"]:
        raise HTTPException(409, "report not ready")
    return r["report"]


@app.get("/api/runs/{rid}/followups", dependencies=[Depends(require_key)])
def followups_list(rid: int):
    """Follow-up questions already asked, with their verified answers."""
    _run(rid)
    return export.followups(store, rid)


@app.get("/api/runs/{rid}/llm/{usage_id}", dependencies=[Depends(require_key)])
def llm_io(rid: int, usage_id: int):
    """The exact prompt and answer of one LLM call (available when STORE_LLM_IO is on and the call succeeded)."""
    _run(rid)
    rows = store.llm_io(rid, usage_id)
    if not rows:
        raise HTTPException(404, "no stored prompt/response for this call (failed call, or STORE_LLM_IO=0)")
    return rows[0]


@app.get("/api/runs/{rid}/proof", dependencies=[Depends(require_key)])
def proof_summary(rid: int):
    """The four review points with live evidence: orchestration, rejections, citation support, trail + measured numbers."""
    _run(rid)
    return proof.summarize(store, rid)


@app.get("/api/runs/{rid}/handoffs", dependencies=[Depends(require_key)])
def run_handoffs(rid: int):
    """The distinct agents (own prompt/tools/models) and the results passed between them, per research round."""
    _run(rid)
    return {"agents": proof.agents_info(), "handoffs": proof.handoffs(store, rid)}


@app.get("/api/runs/{rid}/removed", dependencies=[Depends(require_key)])
def run_removed(rid: int):
    """Claims the verifier removed or flagged (with the source passage they were checked against) and sentences dropped by the audit."""
    _run(rid)
    return proof.removed(store, rid)


@app.post("/api/runs/{rid}/recheck", dependencies=[Depends(require_key), Depends(rate_limit)])
def run_recheck(rid: int):
    """Independently re-judge every report sentence with the auditor model chain (spends a few LLM calls)."""
    r = _run(rid)
    if not r["report"]:
        raise HTTPException(409, "report not ready")
    llm.begin_run(rid, store)
    try:
        return recheck.run(store, rid)
    except llm.VerifierUnavailable:
        raise HTTPException(503, "auditor models unavailable; refusing to claim the citations were re-checked")
    except (llm.BudgetExceeded, llm.RunTimeout, llm.AllModelsFailed):
        raise HTTPException(503, "model capacity unavailable; try again shortly")
    finally:
        llm.end_run()


@app.get("/api/eval", dependencies=[Depends(require_key)])
def evaluation():
    """Self-measured evaluation results (if the harness was run) with metric definitions and limitations; honest 'not measured' otherwise."""
    return proof.eval_results()


@app.get("/api/runs/{rid}/bundle.zip", dependencies=[Depends(require_key)])
def bundle(rid: int):
    """One zip with every generated artefact of the run: report, claims, evidence, logs, usage, prompts, follow-ups."""
    _run(rid)
    return Response(export.bundle(store, rid), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="deepdive-run-{rid}.zip"'})


@app.post("/api/runs/{rid}/resume", dependencies=[Depends(require_key), Depends(rate_limit)])
def resume(rid: int, extra_usd: float = Query(default=None, ge=0, le=10)):
    """Resume a stopped run (verifier outage, budget, timeout, crash) from where it left off, reusing stored work."""
    r = _run(rid)
    ok, why = graph.can_resume(store, rid)
    if not ok:
        raise HTTPException(409, why)
    if not _slots.acquire(blocking=False):
        raise HTTPException(429, "all research slots are busy; retry shortly")
    threading.Thread(target=_safe_run, args=(rid, r["question"], True, extra_usd), daemon=True).start()
    return {"run_id": rid, "resumed": True}


@app.get("/api/runs/{rid}/report.html", response_class=HTMLResponse, dependencies=[Depends(require_key)])
def report_html(rid: int, fragment: bool = False):
    """Shareable HTML report (use the browser's Print -> Save as PDF)."""
    r = _run(rid)
    if not r["report"]:
        raise HTTPException(409, "report not ready")
    return render.to_html(r["report"], fragment=fragment)


@app.get("/api/runs/{rid}/report.md", response_class=PlainTextResponse, dependencies=[Depends(require_key)])
def report_md(rid: int):
    """Markdown report."""
    r = _run(rid)
    if not r["report"]:
        raise HTTPException(409, "report not ready")
    return render.to_markdown(r["report"])


@app.post("/api/runs/{rid}/followup", dependencies=[Depends(require_key), Depends(rate_limit)])
def ask(rid: int, q: Q):
    """Answer a follow-up using stored evidence and the same verification gates."""
    _run(rid)
    if not (config.MIN_QUESTION_CHARS <= len(q.question.strip()) <= config.MAX_QUESTION_CHARS):
        raise HTTPException(422, "question length out of range")
    llm.begin_run(rid, store)  # follow-ups are also cost-accounted and budget-capped
    try:
        return followup.answer(store, rid, q.question.strip())
    except llm.VerifierUnavailable:
        raise HTTPException(503, "verifier unavailable; refusing to answer without verification")
    except (llm.BudgetExceeded, llm.RunTimeout, llm.AllModelsFailed):
        raise HTTPException(503, "model capacity unavailable; try again shortly")
    finally:
        llm.end_run()
