"""Build the paper markdown into a styled HTML page + a single-column DOCX.

Follows the paper-build skill's SynSmith house style (Charter/Georgia, 720px,
centered title, small-caps abstract, booktabs tables, first-line indent).

Usage:
  python build_paper.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "paper_shell_sampling.md"
HTML_OUT = ROOT / "paper.html"
DOCX_OUT = ROOT / "paper.docx"
HTML2DOC = Path(r"C:\Users\apart\.claude\skills\html2doc")

TITLE = ("Where to Query a Blackbox Anomaly Pipeline: "
         "Shell Sampling for Label-Free Distillation")
AUTHOR_HTML = (
    '<p class="authors">Alexander Apartsin<sup>1</sup>, Yehudit Aperstein<sup>2</sup></p>'
    '<p class="affil"><sup>1</sup>School of Computer Science, Faculty of Sciences, '
    'Holon Institute of Technology (HIT), Holon, Israel</p>'
    '<p class="affil"><sup>2</sup>Intelligent Systems, Afeka Academic College of '
    'Engineering, Tel-Aviv, Israel</p>')

STYLE = """
:root{
  --fg:#111418;--fg-soft:#2c3138;--muted:#5a626c;--accent:#14385c;
  --bg:#ffffff;--bg-soft:#fafafa;--rule:#d1d4d8;--line:#d1d4d8;
  --code-bg:#f4f5f7;--code-fg:#14181d;
  --font-body:"Charter","Iowan Old Style","Source Serif Pro",Georgia,"Times New Roman","Liberation Serif",serif;
  --font-title:"Charter","Source Serif Pro",Georgia,"Times New Roman",serif;
  --font-mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html{background:var(--bg);scroll-behavior:smooth}
body{font-family:var(--font-body);color:var(--fg);background:var(--bg);
  max-width:720px;margin:0 auto;padding:3rem 1.3rem 4rem;
  line-height:1.55;font-size:11pt;font-feature-settings:"lnum","kern"}
h1{font-family:var(--font-title);font-size:19pt;font-weight:600;line-height:1.22;
  margin:0 0 .8rem;letter-spacing:-.005em;color:var(--fg);text-align:center}
.authors{text-align:center;font-family:var(--font-body);font-size:12pt;color:var(--fg);margin:.6rem 0 .25rem}
.affil{text-align:center;font-size:10pt;color:var(--fg-soft);font-style:italic;margin:.1rem 0 1.8rem}
h2{font-family:var(--font-title);font-size:13pt;font-weight:700;margin:2rem 0 .6rem;
  padding-top:.8rem;border-top:none;letter-spacing:-.005em;color:var(--fg)}
h3{font-family:var(--font-title);font-size:11.5pt;font-weight:700;margin:1.3rem 0 .45rem;color:var(--fg)}
h4{font-size:10.5pt;font-weight:700;margin:1rem 0 .3rem;color:var(--fg)}
p{margin:0;text-align:justify;text-justify:inter-word;hyphens:auto;line-height:1.55;text-indent:1.4em}
h1+p,h2+p,h3+p,h4+p,figure+p,.tablewrap+p,.katex-display+p,.abstract p:first-of-type,li>p:first-child,blockquote+p{text-indent:0}
li{text-align:left}
li p{text-indent:0}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
code{font-family:var(--font-mono);font-size:.9em;background:var(--code-bg);color:var(--code-fg);padding:.08em .28em;border-radius:2px}
pre{background:var(--code-bg);color:var(--code-fg);border:1px solid var(--rule);border-radius:3px;
  padding:.7rem .9rem;overflow-x:auto;font-size:9pt;line-height:1.45;margin:.9rem 0}
pre code{background:transparent;padding:0;font-size:inherit}
table{border-collapse:collapse;width:100%;margin:1rem 0 1.2rem;font-size:9.5pt;
  font-feature-settings:"tnum";border-top:1.5px solid var(--fg);border-bottom:1.5px solid var(--fg)}
th,td{padding:.4rem .55rem;text-align:left;border-bottom:.5px solid var(--rule);vertical-align:top}
th{font-weight:700;color:var(--fg);background:var(--bg);border-bottom:1px solid var(--fg);border-top:none}
blockquote{border-left:2px solid var(--accent);margin:1rem 0;padding:.2rem 1rem;color:var(--fg-soft)}
.abstract{margin:1.5rem 0 2rem;padding:0 1.2rem}
.abstract h2{font-family:var(--font-body);font-size:10.5pt;font-weight:700;text-align:center;
  margin:0 0 .5rem;padding:0;border:none;color:var(--fg);font-variant:small-caps;letter-spacing:0}
.abstract p{margin:.55rem 0;font-size:10pt;text-align:justify;hyphens:auto;color:var(--fg);text-indent:0}
hr{border:none;border-top:.5px solid var(--rule);margin:2rem 0}
figure{margin:1.4rem auto;text-align:center;max-width:100%}
img{max-width:100%;height:auto;display:block;margin:1.4rem auto .35rem;border:.5px solid var(--rule);border-radius:0}
p:has(img){text-indent:0;text-align:center}
p:has(img)+p,figcaption{font-size:9.5pt;color:var(--fg-soft);text-align:justify;text-justify:inter-word;hyphens:auto;margin:.2rem 0 1rem;padding:0 .4rem;text-indent:0}
.docxlinks{position:fixed;top:1rem;right:1rem;display:flex;flex-direction:column;gap:.4rem;align-items:flex-end;z-index:10}
.docxlink{font-family:var(--font-body);font-size:10pt;font-weight:600;padding:.4rem .85rem;
  border:1px solid var(--accent);color:var(--accent);background:var(--bg);text-decoration:none;
  border-radius:3px;letter-spacing:.01em;white-space:nowrap;line-height:1;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.docxlink:hover{background:var(--accent);color:var(--bg);text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,.12)}
footer{margin-top:3rem;padding-top:.9rem;border-top:.5px solid var(--rule);font-size:9pt;color:var(--muted);text-align:center;font-family:var(--font-body)}
@media print{
  .docxlinks{display:none}
  body{max-width:none;font-size:10pt;padding:0;color:#000}
  @page{margin:2cm}
  h2,h3{break-after:avoid}
  img,table,figure{break-inside:avoid;page-break-inside:avoid}
  pre,blockquote{break-inside:avoid}
  a{color:#000;text-decoration:none}
}
"""

KATEX = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
  integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
  integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  integrity="sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk" crossorigin="anonymous"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false},{left:'\\\\(',right:'\\\\)',display:false},{left:'\\\\[',right:'\\\\]',display:true}]});"></script>"""

DOCXLINK = ('<div class="docxlinks">'
            '<a class="docxlink" href="paper.pdf">&#8595; Paper (.pdf)</a>'
            '<a class="docxlink" href="paper.docx" download>&#8595; Manuscript (.docx)</a>'
            '</div>')


def strip_top_matter(md: str) -> str:
    """Remove the source markdown's H1 title and author byline; they are
    replaced by templated versions in the HTML wrapper."""
    lines = md.splitlines()
    kept = []
    seen_h1 = False
    for line in lines:
        if not seen_h1 and line.startswith("# "):
            seen_h1 = True
            continue
        # drop the author byline emitted right after the H1
        if seen_h1 and line.strip().startswith("**A. Apartsin"):
            continue
        kept.append(line)
    return "\n".join(kept)


# --- Math-preservation across the markdown parser -------------------------
# Python-markdown treats `_` as italics, `*` as bold, and can also swallow
# `\{...\}`. Any of these characters inside `$...$` (KaTeX math) gets
# corrupted before KaTeX ever sees it. Fix: pre-extract math spans, replace
# with placeholders, run markdown, then re-inject the original spans so KaTeX
# auto-render (client-side) processes them intact.
_MATH_PLACEHOLDER = "\x00MATH{}\x01"
_MATH_PATTERNS = [
    (re.compile(r'\$\$(.+?)\$\$', re.DOTALL), '$$', '$$'),  # display first
    (re.compile(r'(?<!\\)\$([^\$\n]+?)(?<!\\)\$'), '$', '$'),  # inline
]


def _protect_math(md: str):
    stash: list[str] = []
    def _stash(m, opener, closer):
        stash.append(f"{opener}{m.group(1)}{closer}")
        return _MATH_PLACEHOLDER.format(len(stash) - 1)
    for pat, opener, closer in _MATH_PATTERNS:
        md = pat.sub(lambda m, o=opener, c=closer: _stash(m, o, c), md)
    return md, stash


def _restore_math(html: str, stash: list[str]) -> str:
    for i, span in enumerate(stash):
        html = html.replace(_MATH_PLACEHOLDER.format(i), span)
    return html


def wrap_abstract(html: str) -> str:
    """Wrap the Abstract H2 and its following <p> in <div class="abstract">."""
    pat = re.compile(
        r'(<h2[^>]*>\s*Abstract\s*</h2>\s*<p>.*?</p>)',
        re.DOTALL | re.IGNORECASE,
    )
    return pat.sub(lambda m: f'<div class="abstract">{m.group(1)}</div>', html, count=1)


def build_html() -> Path:
    md = strip_top_matter(SRC.read_text(encoding="utf-8"))
    md, math_stash = _protect_math(md)
    html = markdown.markdown(md, extensions=[
        "tables", "fenced_code", "toc", "sane_lists", "attr_list", "md_in_html",
    ])
    html = _restore_math(html, math_stash)
    html = wrap_abstract(html)
    page = (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'<title>{TITLE}</title>\n{KATEX}\n<style>{STYLE}</style>\n'
        f'</head>\n<body>\n'
        f'{DOCXLINK}\n'
        f'<h1>{TITLE}</h1>\n'
        f'{AUTHOR_HTML}\n'
        f'{html}\n</body>\n</html>\n'
    )
    HTML_OUT.write_text(page, encoding="utf-8")
    print(f"[html] wrote {HTML_OUT} ({len(page):,} chars)")
    return HTML_OUT


def _leftalign_code_blocks(docx_path: Path, source_md: Path) -> None:
    """apply_academic_style.py sets Body Text justified and re-tags every
    paragraph — even pandoc's Source Code ones — so the style-name test
    misses them and code lines spread across the full column with grotesque
    word gaps. Detect code paragraphs by content instead: extract every
    fenced-code block from the SOURCE markdown, then match paragraphs whose
    text is a line from any of those blocks."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    # Extract lines from every ```...``` fence in the source markdown.
    md = source_md.read_text(encoding="utf-8")
    code_lines: set[str] = set()
    for block in re.findall(r"```[^\n]*\n(.*?)```", md, re.DOTALL):
        for line in block.splitlines():
            s = line.strip()
            if s:
                code_lines.add(s)

    doc = Document(str(docx_path))
    n_fixed = 0
    for p in doc.paragraphs:
        text = p.text
        if not text.strip():
            continue
        # Pandoc emits each fenced block as ONE paragraph with literal '\n' in
        # its text. Test: does any line of the paragraph appear verbatim in the
        # source's code blocks? If yes, this whole paragraph is code.
        para_lines = {ln.strip() for ln in text.split("\n") if ln.strip()}
        if para_lines & code_lines:
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Pt(0)
            for run in p.runs:
                run.font.name = "Consolas"
                run.font.size = Pt(9)
            n_fixed += 1
    doc.save(str(docx_path))
    print(f"[docx] left-aligned {n_fixed} code paragraph(s)")


def build_docx(html_path: Path) -> Path:
    """Run the html2doc 3-stage pipeline to produce a single-column DOCX."""
    stage1 = ROOT / "_paper_mathml.html"
    stage2 = ROOT / "_paper_converted.docx"

    env = os.environ.copy()
    env["NODE_PATH"] = str(HTML2DOC / "node_modules")

    # Stage 1: KaTeX -> MathML
    print("[docx] stage 1: KaTeX -> MathML")
    subprocess.run(
        ["node", str(HTML2DOC / "scripts" / "katex_to_mathml.js"),
         "--input", str(html_path), "--output", str(stage1)],
        env=env, check=True, cwd=str(ROOT),
    )
    # Stage 2: MathML -> DOCX with OMML
    print("[docx] stage 2: pandoc HTML -> DOCX")
    subprocess.run(
        [sys.executable, str(HTML2DOC / "scripts" / "convert_to_docx.py"),
         "--input", str(stage1), "--output", str(stage2),
         "--profile", "camera-ready-generic"],
        check=True, cwd=str(ROOT),
    )
    # Stage 3: academic styling
    print("[docx] stage 3: academic styling (Georgia serif, booktabs, indent)")
    subprocess.run(
        [sys.executable, str(HTML2DOC / "scripts" / "apply_academic_style.py"),
         "--input", str(stage2), "--output", str(DOCX_OUT),
         "--profile", "camera-ready-generic",
         "--font-family", "Georgia"],
        check=True, cwd=str(ROOT),
    )
    # Post-process: fix pandoc code-block paragraphs that inherited justified.
    _leftalign_code_blocks(DOCX_OUT, SRC)
    # Cleanup intermediates.
    for p in (stage1, stage2):
        try: p.unlink()
        except FileNotFoundError: pass
    print(f"[docx] wrote {DOCX_OUT}")
    return DOCX_OUT


def canaries():
    """Content-canary check: HTML links to the DOCX; DOCX exists and non-empty;
    both figures embedded (checked by presence of img tags in HTML)."""
    html_txt = HTML_OUT.read_text(encoding="utf-8")
    assert "paper.docx" in html_txt, "HTML does not link the DOCX"
    for fig in ("figure_sampler_shape.png", "figure_complex.png", "figure_highd_scaling.png",
                "figure_realbench.png", "figure_budget.png"):
        assert fig in html_txt, f"HTML missing reference to {fig}"
    import re as _re
    for banned in (r"\bSAG\b", r"\bSafeRide\b", r"\bgap filling\b",
                   r"\bsynthetic anomaly generation\b", r"\bcovariate shift\b"):
        assert not _re.search(banned, html_txt, _re.IGNORECASE), f"deck term leaked: {banned}"
    assert DOCX_OUT.exists() and DOCX_OUT.stat().st_size > 10_000, \
        f"DOCX missing or too small: {DOCX_OUT}"
    print("[canaries] all pass")


def main():
    html_path = build_html()
    build_docx(html_path)
    canaries()


if __name__ == "__main__":
    main()
