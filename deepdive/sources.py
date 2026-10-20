"""Web access and source handling: FAILOVER search, SAFE fetch (with archive fallback), passage extraction,
source-quality weighting.

Failover design
  search : providers are tried in config.SEARCH_PROVIDERS order (default ddgs -> brave -> tavily -> wikipedia).
           A provider without a key is skipped, a failing provider trips its own circuit breaker, and an EMPTY answer
           moves on to the next provider without blaming the one that answered.
  fetch  : direct download first; if that fails with an HTTP error or timeout, the Internet Archive snapshot of the
           same URL is tried (evidence is then labelled "(archived copy)").

Security: fetching arbitrary URLs from search results is an SSRF risk (a result could point at
http://169.254.169.254/ cloud metadata or an internal service). `is_safe_url` blocks private, loopback,
link-local and reserved addresses, and redirects are followed manually so each hop is re-checked.
"""
import ipaddress
import re
import socket
import threading
import time
from collections import OrderedDict
from urllib.parse import quote as urlquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from . import config, llm, metrics

# --- Source-quality tiers (visible, auditable logic; weights live in config.QUALITY) ----------
PRIMARY_SUFFIX = (".gov", ".edu", ".int", ".mil")
PRIMARY_HOSTS = {"sec.gov", "iea.org", "worldbank.org", "imf.org", "oecd.org", "arxiv.org", "nature.com",
                 "science.org", "europa.eu", "un.org", "who.int", "nih.gov"}
TIER2_HOSTS = {"reuters.com", "bloomberg.com", "ft.com", "wsj.com", "apnews.com", "economist.com",
               "nytimes.com", "bbc.com", "cnbc.com", "mckinsey.com", "bcg.com", "spglobal.com"}
LOW_HOSTS = {"medium.com", "blogspot.com", "wordpress.com", "substack.com", "quora.com", "reddit.com",
             "pinterest.com", "linkedin.com", "facebook.com", "x.com", "twitter.com", "tumblr.com"}

# Words ignored when scoring passage relevance.
STOP = set("the a an of to in and or for on with by is are was were be as at from that this it its their "
           "what which who how why when where about into over under between among can will".split())

UA = {"User-Agent": "Mozilla/5.0 (DeepDive research bot)"}


def quality(url):
    """Return (weight in [0,1], human-readable reason) for a source URL.

    Primary/official sources outrank news, which outranks unknown sites, which outrank blog platforms.
    """
    w = config.QUALITY
    if url.startswith("upload://"):
        return w["upload"], "user-supplied document"
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")

    def match(group):
        return any(host == h or host.endswith("." + h) for h in group)

    if host.endswith(PRIMARY_SUFFIX) or match(PRIMARY_HOSTS):
        return w["primary"], "primary/official source"
    if match(TIER2_HOSTS):
        return w["news"], "established news/analyst outlet"
    if match(LOW_HOSTS):
        return w["low"], "user-generated / blog platform"
    if host.endswith(".org"):
        return w["org"], "organisation site"
    return w["unknown"], "unclassified site"


def is_safe_url(url):
    """True only for public http(s) hosts. Rejects loopback, private, link-local, reserved and metadata IPs."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                    or ip.is_multicast or ip.is_unspecified):
                return False
        return True
    except Exception:  # noqa: BLE001 - any parsing/DNS problem means "do not fetch"
        return False


# ---------------------------------------------------------------- search providers (failover chain)
def _ddgs(query, n):
    """DuckDuckGo via the ddgs package: no key needed, but it throttles. One quick retry before failing over."""
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    last = None
    for attempt in range(2):
        try:
            return [{"url": r["href"], "title": r.get("title", "")} for r in DDGS().text(query, max_results=n)]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 + attempt)
    raise last


def _brave(query, n):
    """Brave Search API (needs BRAVE_API_KEY)."""
    r = httpx.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n},
                  headers={"X-Subscription-Token": config.BRAVE_API_KEY, "Accept": "application/json"},
                  timeout=config.SEARCH_TIMEOUT_S)
    r.raise_for_status()
    return [{"url": x["url"], "title": x.get("title", "")} for x in r.json().get("web", {}).get("results", [])][:n]


def _tavily(query, n):
    """Tavily search API (needs TAVILY_API_KEY)."""
    r = httpx.post("https://api.tavily.com/search", json={"query": query, "max_results": n},
                   headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"}, timeout=config.SEARCH_TIMEOUT_S)
    r.raise_for_status()
    return [{"url": x["url"], "title": x.get("title", "")} for x in r.json().get("results", [])][:n]


def _wikipedia(query, n):
    """Wikipedia search: no key, rarely down, but only useful for encyclopedic facts. The last resort."""
    r = httpx.get("https://en.wikipedia.org/w/api.php",
                  params={"action": "query", "list": "search", "srsearch": query, "srlimit": n, "format": "json"},
                  headers=UA, timeout=config.SEARCH_TIMEOUT_S)
    r.raise_for_status()
    return [{"url": "https://en.wikipedia.org/wiki/" + urlquote(x["title"].replace(" ", "_")), "title": x["title"]}
            for x in r.json().get("query", {}).get("search", [])][:n]


# name -> (function, is_enabled). A provider is enabled when it needs no key or its key is present.
PROVIDERS = {
    "ddgs": (_ddgs, lambda: True),
    "brave": (_brave, lambda: bool(config.BRAVE_API_KEY)),
    "tavily": (_tavily, lambda: bool(config.TAVILY_API_KEY)),
    "wikipedia": (_wikipedia, lambda: True),
}
_pbreakers = {}


def _pbreaker(name):
    """Circuit breaker per search provider (same class and thresholds as the LLM breakers)."""
    return _pbreakers.setdefault(name, llm.Breaker())


def provider_states():
    """{provider: 'open'|'closed'|'disabled'} for the readiness endpoint."""
    out = {}
    for name in config.SEARCH_PROVIDERS:
        spec = PROVIDERS.get(name)
        if not spec or not spec[1]():
            out[name] = "disabled"
        else:
            out[name] = "open" if _pbreaker(name).fails >= config.BREAKER_THRESHOLD else "closed"
    return out


def reset_providers():
    """Forget provider breaker state (used by tests)."""
    _pbreakers.clear()


def search(query, n=6):
    """Search with automatic failover across providers. Returns [] (never raises) if every provider fails.

    Each hit is {'url','title','provider'} so the audit trail can show who answered.
    """
    for name in config.SEARCH_PROVIDERS:
        spec = PROVIDERS.get(name)
        if not spec or not spec[1]():
            continue  # unknown provider name, or its API key is missing
        breaker = _pbreaker(name)
        if not breaker.allow():
            metrics.inc("search_breaker_skips_total", provider=name)
            continue
        try:
            hits = spec[0](query, n)
        except Exception:  # noqa: BLE001 - network/API failure: count it and fail over
            breaker.bad()
            metrics.inc("search_calls_total", provider=name, ok="false")
            continue
        breaker.ok()
        metrics.inc("search_calls_total", provider=name, ok="true")
        if hits:
            return [{**h, "provider": name} for h in hits]
        # an empty answer is not an error, but another provider may know more: keep going
    return []


# ---------------------------------------------------------------- fetch (direct, then archive fallback)
def _fetch_direct(url, timeout):
    """Download one HTML page safely. Returns (page|None, reason).

    reason: ok | unsafe (SSRF check failed) | not_html | status (HTTP error) | error (network/timeout/redirect loop).
    Safety: SSRF check on every hop, manual redirects (max N), size cap, HTML-only, hard timeout.
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, headers=UA) as c:
            cur = url
            for _ in range(config.MAX_REDIRECTS + 1):
                if not is_safe_url(cur):
                    return None, "unsafe"
                with c.stream("GET", cur) as r:
                    if r.is_redirect:  # follow ourselves so the next hop is re-validated
                        cur = urljoin(cur, r.headers.get("location", ""))
                        continue
                    if r.status_code != 200:
                        return None, "status"
                    if "html" not in r.headers.get("content-type", ""):
                        return None, "not_html"
                    buf = bytearray()
                    for chunk in r.iter_bytes():
                        buf += chunk
                        if len(buf) > config.MAX_FETCH_BYTES:  # never hold huge bodies in memory
                            break
                    html = bytes(buf).decode(r.encoding or "utf-8", errors="ignore")
                break
            else:
                return None, "error"  # too many redirects
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            t.decompose()  # drop page chrome so passages are body text
        title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
        text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
        return {"title": title, "text": text[:config.PAGE_TEXT_CAP]}, "ok"
    except Exception:  # noqa: BLE001
        return None, "error"


def _wayback_snapshot(url, timeout):
    """Ask the Internet Archive for the closest snapshot of `url`; returns its URL or None."""
    try:
        r = httpx.get("https://archive.org/wayback/available", params={"url": url}, headers=UA, timeout=timeout)
        r.raise_for_status()
        snap = r.json().get("archived_snapshots", {}).get("closest") or {}
        link = snap.get("url", "")
        # accept only archive.org hosts so a manipulated answer cannot redirect us to an arbitrary site
        if snap.get("available") and (urlparse(link).hostname or "").endswith("archive.org"):
            return link.replace("http://", "https://", 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _fetch_uncached(url, timeout=None):
    """Fetch a page: direct first, then an archived copy if the direct fetch hit an HTTP error or timeout.

    Returns {'title','text'[, 'via': 'wayback']} or None. Unsafe URLs and non-HTML pages never use the fallback.
    """
    timeout = timeout or config.FETCH_TIMEOUT_S
    page, reason = _fetch_direct(url, timeout)
    if page:
        metrics.inc("fetch_total", result="direct")
        return page
    if reason in ("status", "error") and config.FETCH_ARCHIVE_FALLBACK:
        snap = _wayback_snapshot(url, timeout)
        if snap:
            page, _ = _fetch_direct(snap, timeout)
            if page:
                page["via"] = "wayback"
                metrics.inc("fetch_total", result="wayback")
                return page
    metrics.inc("fetch_total", result="failed")
    return None


# Page cache (TTL + size bounded) and a cap on simultaneous downloads; the reasoning for both values is in config.
_cache, _cache_lock = OrderedDict(), threading.Lock()
_fetch_slots = threading.BoundedSemaphore(config.MAX_INFLIGHT_FETCH)


def clear_cache():
    """Forget all cached pages (used by tests)."""
    with _cache_lock:
        _cache.clear()


def fetch(url, timeout=None):
    """Cached, concurrency-bounded fetch. Same contract as _fetch_uncached; callers get a copy they may modify."""
    now = time.time()
    with _cache_lock:
        hit = _cache.get(url)
        if hit and now - hit[0] < config.FETCH_CACHE_TTL_S:
            _cache.move_to_end(url)
            metrics.inc("fetch_total", result="cache")
            return dict(hit[1])
    with _fetch_slots:  # never download more than MAX_INFLIGHT_FETCH pages at once, across all runs
        page = _fetch_uncached(url, timeout)
    if page:
        with _cache_lock:
            _cache[url] = (now, page)
            while len(_cache) > config.FETCH_CACHE_MAX:
                _cache.popitem(last=False)  # evict the least recently used page
        return dict(page)
    return None


# ---------------------------------------------------------------- passage extraction
def tokens(s):
    """Lowercase content words (stop-words and very short words removed) for relevance scoring."""
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2]


def best_passages(text, query, size=None, k=None):
    """Split text into ~`size`-character chunks and return the k most relevant to `query`.

    size/k default to config.PASSAGE_CHARS / config.PASSAGES_PER_PAGE (see the rationale there).
    """
    size, k = size or config.PASSAGE_CHARS, k or config.PASSAGES_PER_PAGE
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:  # greedily pack sentences into chunks
        if len(cur) + len(s) > size and cur:
            chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    q = set(tokens(query))
    scored = [(len(q & set(tokens(c))) / (1 + len(q)) ** 0.5, c) for c in chunks if len(c) >= config.MIN_PASSAGE_CHARS]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for sc, c in scored[:k] if sc > 0]
