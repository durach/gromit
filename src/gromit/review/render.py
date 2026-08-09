"""Render the self-contained review page (inline CSS/JS, file:// friendly)."""

from __future__ import annotations

import html
from dataclasses import dataclass

_CSS = """
* { box-sizing: border-box; }
body { font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 1rem 1rem 6rem;
       color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.3rem; }
.row { background: #fff; border: 1px solid #ddd; border-radius: 10px;
       padding: 1rem; margin: 1rem 0; display: grid; gap: .6rem;
       grid-template-columns: 360px 1fr; }
.row video { width: 360px; border-radius: 8px; background: #000; }
.meta { display: flex; flex-direction: column; gap: .5rem; }
.tags span { font-size: .72rem; padding: .1rem .5rem; border-radius: 999px;
             background: #eee; margin-right: .3rem; }
.reading { padding: .4rem .6rem; border-radius: 6px; background: #f4f4f4; }
.reading .lbl { color: #888; font-size: .72rem; text-transform: uppercase; }
mark { background: #ffe08a; border-radius: 3px; padding: 0 2px; }
input[type=text], select { font: inherit; padding: .35rem .5rem; border: 1px solid #bbb;
       border-radius: 6px; }
.correction { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; }
.correction input[type=text] { flex: 1 1 200px; }
.unavail { color: #b00; font-weight: 600; width: 360px; display: grid; place-items: center;
           background: #fbeaea; border-radius: 8px; }
.bar { position: fixed; bottom: 0; left: 0; right: 0; padding: .8rem 1rem;
       background: #fff; border-top: 1px solid #ddd; display: flex; gap: 1rem;
       align-items: center; }
button { font: inherit; padding: .5rem 1rem; border: 0; border-radius: 8px;
         background: #2a6; color: #fff; cursor: pointer; }
"""

_JS = """
function yamlQuote(s){ return '"' + String(s).replace(/\\\\/g,'\\\\\\\\').replace(/"/g,'\\\\"') + '"'; }
function exportCorrections(){
  const rows = document.querySelectorAll('.row');
  const items = [];
  rows.forEach(r => {
    if(!r.querySelector('.add').checked) return;
    const canonical = r.querySelector('.canon').value.trim();
    const heard = r.querySelector('.heard').value.trim();
    const category = r.querySelector('.cat').value;
    if(!canonical || !heard) return;
    items.push({canonical, heard, category});
  });
  let body = 'corrections:\\n';
  if(items.length === 0){ body += '  []\\n'; }
  items.forEach(it => {
    body += '  - canonical: ' + yamlQuote(it.canonical) + '\\n';
    body += '    heard: ' + yamlQuote(it.heard) + '\\n';
    body += '    category: ' + it.category + '\\n';
  });
  const blob = new Blob([body], {type: 'text/yaml'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'corrections.yaml';
  a.click();
}
"""

_CATEGORIES = ("term", "person", "company", "product")


@dataclass(frozen=True)
class ReviewRow:
    index: int
    clip_rel: str | None
    timestamp: str
    speaker: str
    reasons: tuple[str, ...]
    meet_html: str
    gromit_html: str
    suggestion: str | None


def _plain(html_frag: str) -> str:
    """Strip <mark> tags to recover the plain (still escaped) text for prefill."""
    return html_frag.replace("<mark>", "").replace("</mark>", "")


def _row_html(row: ReviewRow) -> str:
    if row.clip_rel:
        clip = f'<video src="{html.escape(row.clip_rel)}" controls preload="none"></video>'
    else:
        clip = '<div class="unavail">clip unavailable</div>'
    tags = "".join(f"<span>{html.escape(r)}</span>" for r in row.reasons)
    speaker = f' · {html.escape(row.speaker)}' if row.speaker else ""
    suggestion = html.escape(row.suggestion or "")
    heard_prefill = _plain(row.gromit_html)
    cats = "".join(
        f'<option value="{c}"{" selected" if c == "term" else ""}>{c}</option>'
        for c in _CATEGORIES
    )
    return f"""<div class="row" data-i="{row.index}">
  {clip}
  <div class="meta">
    <div class="tags"><b>{html.escape(row.timestamp)}</b>{speaker} {tags}</div>
    <div class="reading"><span class="lbl">meet</span> {row.meet_html or "&mdash;"}</div>
    <div class="reading"><span class="lbl">gromit</span> {row.gromit_html or "&mdash;"}</div>
    <div class="correction">
      <input type="text" class="canon" placeholder="correction (canonical)" value="{suggestion}">
      <input type="text" class="heard" placeholder="heard as" value="{heard_prefill}">
      <select class="cat">{cats}</select>
      <label><input type="checkbox" class="add"> add to glossary</label>
    </div>
  </div>
</div>"""


def render_html(rows: list[ReviewRow], title: str) -> str:
    """Build the full self-contained review document."""
    body = "\n".join(_row_html(r) for r in rows)
    t = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review — {t}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Review — {t} <small>({len(rows)} spans)</small></h1>
{body}
<div class="bar">
  <button onclick="exportCorrections()">Export corrections.yaml</button>
  <span>Tick "add to glossary", fill correction + heard-as, then Export.</span>
</div>
<script>{_JS}</script>
</body>
</html>"""
