"""Turn the stored report dict into Markdown or a self-contained HTML page (print to PDF from a browser)."""
from html import escape


def _c(s):
    """Footnote markers like [1][3] for a sentence."""
    return "".join(f"[{n}]" for n in s["cites"])


def _banner(m):
    """One-line honesty banner when the run was partial or degraded."""
    notes = []
    if m.get("status") == "partial":
        notes.append("PARTIAL report: the run stopped early; only claims verified before that point are shown")
    if m.get("degraded"):
        notes.append("degraded mode: " + ", ".join(m["degraded"]))
    return "; ".join(notes)


def to_markdown(r):
    """Render the report as Markdown (summary, sections, conflicts, sources, verification + cost summary)."""
    m = r["metrics"]
    L = [f"# {r['title']}", ""]
    if _banner(m):
        L += [f"> {_banner(m)}", ""]
    L += ["## Executive summary", " ".join(f"{s['text']} {_c(s)}" for s in r["summary"]), ""]
    for sec in r["sections"]:
        L += [f"## {sec['heading']}", ""]
        if sec["limited"]:
            L += ["> Limited evidence: this area did not reach the verification threshold.", ""]
        L += [" ".join(f"{s['text']} {_c(s)}" for s in sec["sentences"]), ""]
    if r["conflicts"]:
        L += ["## Conflicting evidence", ""] + [f"- {x['explanation']}" for x in r["conflicts"]] + [""]
    L += ["## Sources", ""] + [f"[{s['n']}] {s['url']} (quality {s['quality']}: {s['why']})" for s in r["sources"]]
    L += ["", "## Verification summary", "",
          f"Drafted {m['claims_drafted']} claims; {m['claims_verified']} verified, {m['claims_flagged']} flagged, "
          f"{m['claims_rejected']} rejected. Final-prose audit pass rate {m['audit_pass_rate']}. "
          f"Cost ${m.get('cost_usd', 0):.4f} over {m.get('llm_calls', 0)} LLM calls."]
    return "\n".join(L)


def _safe_href(url):
    """Only http(s) URLs become links (blocks javascript: and other schemes)."""
    return escape(url) if url.startswith(("http://", "https://")) else "#"


def to_html(r, fragment=False):
    """Render the report as HTML; every dynamic string is HTML-escaped.

    fragment=False -> a standalone page (print to PDF). fragment=True -> just the content, for embedding in the
    dashboard (no <style>/<doctype>, so it cannot restyle the host page). Each sentence is wrapped in
    <span class="s" data-c="claim ids"> so the dashboard can show the exact supporting quote on click.
    """
    m = r["metrics"]
    e = escape

    def sent(s, loc):
        sup = "".join(f'<sup><a href="#src{n}">[{n}]</a></sup>' for n in s["cites"])
        ids = ",".join(str(c) for c in s["claim_ids"])
        return f'<span class="s" data-c="{ids}" data-loc="{loc}">{e(s["text"])}</span>{sup} '

    style = ("<!doctype html><meta charset=utf-8><title>" + e(r["title"]) + "</title>"
             "<style>body{font:16px/1.6 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}"
             "h1{font-size:1.7rem}h2{margin-top:2rem}.lim,.ban{background:#fff6e0;padding:.5rem .8rem;border-radius:6px}"
             ".box{background:#f4f6f8;padding:1rem;border-radius:8px;margin-top:2rem}td,th{padding:.3rem .6rem;"
             "text-align:left;vertical-align:top;font-size:.9rem}table{border-collapse:collapse}small{color:#666}</style>")
    h = ["<div class=rpt>" if fragment else style, f"<h1>{e(r['title'])}</h1>"]
    if _banner(m):
        h.append(f"<p class=ban>{e(_banner(m))}</p>")
    h.append(f"<h2>Executive summary</h2><p>{''.join(sent(s, f's-{i}') for i, s in enumerate(r['summary']))}</p>")
    for j, sec in enumerate(r["sections"]):
        h.append(f"<h2>{e(sec['heading'])}</h2>")
        if sec["limited"]:
            h.append('<p class=lim>Limited evidence: this area did not reach the verification threshold.</p>')
        h.append(f"<p>{''.join(sent(s, f'c-{j}-{k}') for k, s in enumerate(sec['sentences']))}</p>")
    if r["conflicts"]:
        h.append("<h2>Conflicting evidence</h2><ul>" + "".join(
            f"<li>{e(x['explanation'])}<br><small>A: {e(x['a'])}<br>B: {e(x['b'])}</small></li>" for x in r["conflicts"]) + "</ul>")
    h.append("<h2>Sources</h2><ol>" + "".join(
        f'<li id="src{s["n"]}"><a href="{_safe_href(s["url"])}">{e(s["title"] or s["url"])}</a> '
        f'<small>quality {s["quality"]} ({e(s["why"])})</small></li>' for s in r["sources"]) + "</ol>")
    h.append(f"<div class=box><b>Verification report</b><br>Drafted {m['claims_drafted']} claims: "
             f"{m['claims_verified']} verified, {m['claims_flagged']} flagged, {m['claims_rejected']} rejected "
             f"(first-pass groundedness {m['first_pass_groundedness']}). Final-prose audit pass rate "
             f"{m['audit_pass_rate']}. Research rounds: {m['research_rounds']}. "
             f"Cost ${m.get('cost_usd', 0):.4f}, {m.get('tokens', 0)} tokens, {m.get('llm_calls', 0)} LLM calls.")
    if r["rejected"]:
        h.append("<h3>Rejected claims</h3><table><tr><th>Claim<th>Stage<th>Why</tr>" + "".join(
            f"<tr><td>{e(x['text'])}<td>{e(x['stage'])}<td>{e(x['reason'])}</tr>" for x in r["rejected"]) + "</table>")
    h.append("</div>" + ("</div>" if fragment else ""))
    return "".join(h)
