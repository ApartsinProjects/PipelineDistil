"""Take paper.html and produce an artifact-ready standalone HTML:
  1) pre-render $..$ math to inline MathML (Artifact CSP blocks the KaTeX CDN)
  2) inline the two PNG figures as data: URIs
  3) drop the .docxlinks sidebar (download links are inert on the hosted page)
  4) drop the KaTeX <link>/<script> tags (no longer needed)

The visual identity (SynSmith serif house style) is unchanged - the artifact
is the same page, just self-contained. Light-only theme, painted background.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN = ROOT / "paper.html"
OUT = ROOT / "paper_artifact.html"
HTML2DOC = Path(r"C:\Users\apart\.claude\skills\html2doc")


def _prerender_math(inp: Path, outp: Path) -> None:
    env = os.environ.copy()
    env["NODE_PATH"] = str(HTML2DOC / "node_modules")
    subprocess.run(
        ["node", str(HTML2DOC / "scripts" / "katex_to_mathml.js"),
         "--input", str(inp), "--output", str(outp)],
        env=env, check=True, cwd=str(ROOT),
    )


def _inline_image(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def main():
    tmp_mathml = ROOT / "_paper_artifact_mathml.html"
    _prerender_math(IN, tmp_mathml)
    html = tmp_mathml.read_text(encoding="utf-8")

    # Drop the KaTeX CSS/JS tags (math is now inline MathML).
    html = re.sub(
        r'<link[^>]*katex[^>]*>\s*|<script[^>]*katex[^>]*></script>\s*',
        "", html, flags=re.IGNORECASE,
    )

    # Drop the .docxlinks sidebar (its downloads don't exist next to the artifact).
    html = re.sub(
        r'<div class="docxlinks">.*?</div>\s*',
        "", html, flags=re.DOTALL,
    )

    # Inline every relative <img src="...">.
    def _sub_img(m):
        src = m.group(1)
        p = (ROOT / src).resolve()
        if not p.exists():
            return m.group(0)
        return f'src="{_inline_image(p)}"'
    html = re.sub(r'src="([^":]+\.(?:png|jpe?g))"', _sub_img, html)

    # Paint body background explicitly (survives any host theme).
    # The existing CSS already does this via var(--bg) which is defined; belt-and-braces:
    if "body{" in html and "background:var(--bg)" in html:
        pass  # already correct
    else:
        html = html.replace("body{", "body{background:#ffffff;", 1)

    # Ensure browsers render inline MathML properly (Chromium needs no polyfill,
    # but Safari older versions want the `math` display style hint).
    if "<style>" in html and "math{" not in html:
        css_hint = (
            "math{font-family:'Cambria Math','STIX Two Math','Latin Modern Math',"
            "'Georgia',serif;font-size:1em}"
            ".katex-display,.mathml-display{display:block;margin:.6rem 0;text-align:center}"
        )
        html = html.replace("<style>", f"<style>{css_hint}", 1)

    OUT.write_text(html, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"[artifact] wrote {OUT}  ({kb:.0f} KB, all figures inlined, math as MathML)")
    try: tmp_mathml.unlink()
    except FileNotFoundError: pass


if __name__ == "__main__":
    main()
