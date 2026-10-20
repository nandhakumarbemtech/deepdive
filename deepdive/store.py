"""SQLite persistence: runs, evidence (with source), claims (with verdicts), usage (tokens/cost), trail.

Production notes: WAL mode lets readers (the UI polling) work while a run writes; a busy timeout avoids
"database is locked" errors; indexes keep per-run queries fast. For multi-instance deployments swap this
class for Postgres (the interface is small on purpose).
"""
import json
import sqlite3
import threading
import time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, status TEXT,
  created REAL, report TEXT, metrics TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS evidence(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INT, subq TEXT, url TEXT,
  title TEXT, text TEXT, quality REAL, quality_reason TEXT, rnd INT);
CREATE TABLE IF NOT EXISTS claims(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INT, subq TEXT, text TEXT,
  evidence_id INT, quote TEXT, status TEXT, reason TEXT, stage TEXT, rnd INT, origin TEXT);
CREATE TABLE IF NOT EXISTS trail(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INT, ts REAL, agent TEXT,
  event TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS usage(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INT, ts REAL, role TEXT,
  model TEXT, prompt_tokens INT, completion_tokens INT, cost_usd REAL, latency_ms INT, ok INT,
  estimated INT, attempt INT, fallback INT, error TEXT);
CREATE TABLE IF NOT EXISTS llm_io(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INT, usage_id INT, role TEXT,
  model TEXT, system_prompt TEXT, user_prompt TEXT, response TEXT);
CREATE INDEX IF NOT EXISTS ix_llmio_run ON llm_io(run_id, usage_id);
CREATE INDEX IF NOT EXISTS ix_evidence_run ON evidence(run_id, subq);
CREATE INDEX IF NOT EXISTS ix_claims_run ON claims(run_id, status);
CREATE INDEX IF NOT EXISTS ix_trail_run ON trail(run_id);
CREATE INDEX IF NOT EXISTS ix_usage_run ON usage(run_id);
"""


class Store:
    """Thread-safe wrapper around one SQLite connection (a lock serialises access)."""

    def __init__(self, path=None):
        self.db = sqlite3.connect(path or config.DB_PATH, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")   # readers do not block the writer
            self.db.execute("PRAGMA busy_timeout=30000")  # wait instead of failing when locked
            self.db.executescript(SCHEMA)
            try:  # migrate databases created before the `error` column existed
                self.db.execute("ALTER TABLE runs ADD COLUMN error TEXT")
            except sqlite3.OperationalError:
                pass
            self.db.commit()

    def _exec(self, sql, args=()):
        """Run a write statement and commit."""
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def _all(self, sql, args=()):
        """Run a read query and return rows as dicts."""
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def ping(self):
        """Cheap health check used by /readyz."""
        self._all("SELECT 1")
        return True

    # ------------------------------------------------------------ runs
    def create_run(self, question):
        """Create a run row in status 'queued' and return its id."""
        return self._exec("INSERT INTO runs(question,status,created) VALUES(?,?,?)",
                          (question, "queued", time.time())).lastrowid

    def set_run(self, run_id, **fields):
        """Update status/report/metrics/error. JSON fields are serialised; other keys are rejected."""
        for k, v in fields.items():
            if k not in ("status", "report", "metrics", "error"):
                raise ValueError(k)
            if k in ("report", "metrics"):
                v = json.dumps(v)
            self._exec(f"UPDATE runs SET {k}=? WHERE id=?", (v, run_id))

    def get_run(self, run_id):
        """Return one run (with report/metrics decoded) or None."""
        rows = self._all("SELECT * FROM runs WHERE id=?", (run_id,))
        if not rows:
            return None
        r = rows[0]
        r["report"] = json.loads(r["report"]) if r["report"] else None
        r["metrics"] = json.loads(r["metrics"]) if r["metrics"] else None
        return r

    def mark_stale_runs(self):
        """On startup, runs left 'running'/'queued' by a crash can never finish: mark them interrupted."""
        self._exec("UPDATE runs SET status='interrupted', error='service restarted mid-run' "
                   "WHERE status IN ('running','queued')")

    # ------------------------------------------------------------ evidence
    def add_evidence(self, run_id, subq, url, title, text, quality, why, rnd):
        """Store one source passage together with its URL and quality weight (this is what claims trace to)."""
        return self._exec(
            "INSERT INTO evidence(run_id,subq,url,title,text,quality,quality_reason,rnd) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, subq, url, title, text, quality, why, rnd)).lastrowid

    def evidence(self, run_id, subq=None, rnd=None):
        """List a run's evidence, optionally filtered by sub-question and research round."""
        sql, args = "SELECT * FROM evidence WHERE run_id=?", [run_id]
        if subq is not None:
            sql += " AND subq=?"
            args.append(subq)
        if rnd is not None:
            sql += " AND rnd=?"
            args.append(rnd)
        return self._all(sql, args)

    def evidence_by_id(self, eid):
        """Fetch one evidence row by id."""
        rows = self._all("SELECT * FROM evidence WHERE id=?", (eid,))
        return rows[0] if rows else None

    def has_url(self, run_id, subq, url):
        """True if this URL was already fetched for this sub-question (avoids duplicate work)."""
        return bool(self._all("SELECT 1 FROM evidence WHERE run_id=? AND subq=? AND url=?", (run_id, subq, url)))

    # ------------------------------------------------------------ claims
    def add_claim(self, run_id, subq, text, evidence_id, quote, rnd, origin="draft"):
        """Insert a claim in status 'pending' (awaiting verification)."""
        return self._exec(
            "INSERT INTO claims(run_id,subq,text,evidence_id,quote,status,rnd,origin) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, subq, text, evidence_id, quote, "pending", rnd, origin)).lastrowid

    def set_claim(self, cid, status, stage, reason):
        """Record a verification verdict: status (verified/flagged/rejected), the stage that decided, and why."""
        self._exec("UPDATE claims SET status=?, stage=?, reason=? WHERE id=?", (status, stage, reason, cid))

    def claims(self, run_id, status=None, subq=None, origin="draft"):
        """List claims filtered by status / sub-question / origin ('draft' vs 'followup')."""
        sql, args = "SELECT * FROM claims WHERE run_id=?", [run_id]
        for col, val in (("status", status), ("subq", subq), ("origin", origin)):
            if val is not None:
                sql += f" AND {col}=?"
                args.append(val)
        return self._all(sql + " ORDER BY id", args)

    def claim(self, cid):
        """Fetch one claim by id."""
        rows = self._all("SELECT * FROM claims WHERE id=?", (cid,))
        return rows[0] if rows else None

    def claims_by_ids(self, ids):
        """Fetch several claims by id (missing ids are skipped)."""
        return [c for c in (self.claim(i) for i in ids) if c]

    # ------------------------------------------------------------ usage (tokens and cost)
    def add_usage(self, run_id, role, model, pt, ct, cost, latency_ms, ok, estimated, attempt, fallback, error=""):
        """Persist one LLM call's tokens, cost, latency and outcome."""
        return self._exec("INSERT INTO usage(run_id,ts,role,model,prompt_tokens,completion_tokens,cost_usd,latency_ms,"
                          "ok,estimated,attempt,fallback,error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (run_id, time.time(), role, model, pt, ct, cost, latency_ms, int(ok), int(estimated),
                           attempt, int(fallback), error)).lastrowid  # id links the call to its stored prompt/response

    def add_llm_io(self, run_id, usage_id, role, model, system_prompt, user_prompt, response):
        """Store the exact prompt and answer of one LLM call so generated text can be read back and audited."""
        self._exec("INSERT INTO llm_io(run_id,usage_id,role,model,system_prompt,user_prompt,response) VALUES(?,?,?,?,?,?,?)",
                   (run_id, usage_id, role, model, system_prompt, user_prompt, response))

    def llm_io(self, run_id, usage_id=None):
        """Prompt/response records for a run (one call if usage_id is given, else all)."""
        if usage_id is not None:
            return self._all("SELECT * FROM llm_io WHERE run_id=? AND usage_id=?", (run_id, usage_id))
        return self._all("SELECT * FROM llm_io WHERE run_id=? ORDER BY id", (run_id,))

    def usage_all(self, run_id):
        """Every LLM call row of a run (oldest first), for export."""
        return self._all("SELECT * FROM usage WHERE run_id=? ORDER BY id", (run_id,))

    def usage_summary(self, run_id):
        """Aggregate a run's cost/tokens/calls, broken down by agent role and by model."""
        rows = self._all("SELECT * FROM usage WHERE run_id=?", (run_id,))
        by_role, by_model = {}, {}
        for r in rows:
            for key, bucket in ((r["role"], by_role), (r["model"], by_model)):
                b = bucket.setdefault(key, {"calls": 0, "cost_usd": 0.0, "tokens": 0})
                b["calls"] += 1
                b["cost_usd"] = round(b["cost_usd"] + r["cost_usd"], 6)
                b["tokens"] += r["prompt_tokens"] + r["completion_tokens"]
        lat = sorted(r["latency_ms"] for r in rows if r["ok"])  # latency of successful calls only

        def pct(p):
            """p-th percentile (0..1) of call latency in ms (nearest-rank)."""
            return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else 0

        return {
            "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "max": lat[-1] if lat else 0},
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "llm_calls": len(rows),
            "failed_calls": sum(1 for r in rows if not r["ok"]),
            "fallbacks_used": sum(1 for r in rows if r["fallback"] and r["ok"]),
            "estimated_usage_calls": sum(1 for r in rows if r["estimated"]),
            "by_role": by_role, "by_model": by_model,
        }

    # ------------------------------------------------------------ trail (audit log)
    def log(self, run_id, agent, event, data=None):
        """Append an audit-trail event (what an agent did and why)."""
        self._exec("INSERT INTO trail(run_id,ts,agent,event,data) VALUES(?,?,?,?,?)",
                   (run_id, time.time(), agent, event, json.dumps(data or {})))

    def trail(self, run_id, since=0):
        """Return the run's audit trail in order (only events with id > since), JSON data decoded."""
        rows = self._all("SELECT * FROM trail WHERE run_id=? AND id>? ORDER BY id", (run_id, since))
        for r in rows:
            r["data"] = json.loads(r["data"])
        return rows

    # ------------------------------------------------------------ live-dashboard helpers
    def run_brief(self, run_id):
        """Run row WITHOUT the (large) report/metrics JSON: cheap enough to poll every second."""
        rows = self._all("SELECT id,question,status,created,error FROM runs WHERE id=?", (run_id,))
        return rows[0] if rows else None

    def usage_rows(self, run_id, limit=40):
        """Most recent LLM calls for a run (oldest first), for the live 'LLM calls' feed."""
        rows = self._all("SELECT * FROM usage WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit))
        return list(reversed(rows))

    def claim_counts(self, run_id):
        """Live claim funnel: how many claims are pending/verified/flagged/rejected right now."""
        out = {"pending": 0, "verified": 0, "flagged": 0, "rejected": 0}
        for r in self._all("SELECT status, COUNT(*) n FROM claims WHERE run_id=? AND origin='draft' GROUP BY status",
                           (run_id,)):
            out[r["status"]] = r["n"]
        return out

    def evidence_stats(self, run_id):
        """Number of stored passages and distinct source URLs gathered so far."""
        r = self._all("SELECT COUNT(*) n, COUNT(DISTINCT url) u FROM evidence WHERE run_id=?", (run_id,))[0]
        return {"passages": r["n"], "sources": r["u"]}

    def list_runs(self, limit=20):
        """Recent runs with status and total cost, newest first (the 'all runs' panel)."""
        return self._all("SELECT r.id, r.question, r.status, r.created, COALESCE(SUM(u.cost_usd),0) AS cost_usd "
                         "FROM runs r LEFT JOIN usage u ON u.run_id=r.id GROUP BY r.id ORDER BY r.id DESC LIMIT ?",
                         (limit,))
