#!/usr/bin/env python3
"""Render README.md and docs/**/*.md to the static docs site at https://osma-bench.github.io/docs/.

- README.md becomes index.html and docs/ is the site root, so docs/metrics.md -> metrics.html and
  docs/images/teaser.jpg -> images/teaser.jpg.
- Headings get GitHub-style ids; relative .md links become .html links.
- $...$ and $$...$$ are rendered by KaTeX in the browser; the math is set aside before Markdown
  parsing, so backslashes, braces and underscores reach KaTeX unchanged.
- "> [!NOTE]"-style alerts become callouts; raw HTML such as <details> passes through.
- Every page gets a sidebar with the main docs, an "On this page" list of its sections and previous/next
  links. The README's link row (the first paragraph that holds only links separated by " · ") becomes a
  button bar at the top of every page; placeholder URLs are greyed out.
- Only the main docs are listed. A .md file under docs/ that the docs link to is rendered but not listed;
  the other files under docs/ (maintainer notes) are not rendered.
- Links to anything outside README.md and docs/ (code, configs, submodule READMEs) point to the file in
  the anonymous code repo. Images and videos under docs/ are copied into the site.
- Every relative link and image is checked: the target must exist and a #fragment must be a heading id
  there. Targets inside a submodule that is not checked out (as in CI) are not checked. Any broken link
  makes the script exit with status 1.

The output is rebuilt from scratch on every run (--out is deleted first, and must lie under dist/ or
outputs/). scripts/build_anon_page.sh runs this into dist/anon-page/docs.

Usage: python3 scripts/docs_site/build_docs_site.py [--out DIR]
"""
import argparse
import html
import os
import re
import shutil
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[2]
CSS = Path(__file__).with_name("github-markdown-light.css")
# Reading order of the documentation; anything else found under docs/ is listed after it.
MAIN = ["README.md", "docs/getting_started.md", "docs/benchmark.md", "docs/metrics.md",
        "docs/policies.md", "docs/experiments.md"]
NAV_LABEL = {"README.md": "Overview"}
BRAND = "🚦 OSMA-Bench"
TITLE = "OSMA-Bench Docs"
# Copied to the preview root; the page links them from there.
ICONS = {"favicon.ico": "page/favicon.ico", "favicon-32.png": "page/static/images/favicon-32.png",
         "apple-touch-icon.png": "page/static/images/apple-touch-icon.png"}
MAX_COPY_BYTES = 50 * 2**20
# Placeholder URLs in the README link row carry this token (as in page/index.html).
PLACEHOLDER_TOKEN = "REPLACE-AFTER-SUBMISSION"
# Files outside README.md and docs/ are linked here.
CODE_URL = "https://github.com/OSMA-Bench/OSMA-Bench/blob/main"
SCHEME = re.compile(r"[a-z][a-z0-9+.-]*:", re.I)
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
KATEX_HEAD = f"""<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js" onload="renderMathInElement(document.querySelector('main'), """ + r"""{delimiters: [{left: '\\[', right: '\\]', display: true}, {left: '\\(', right: '\\)', display: false}], throwOnError: false});"></script>"""
ALERT = re.compile(r"\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
DISPLAY = re.compile(r"\$\$(.+?)\$\$")
INLINE = re.compile(r"(?<![\\$\w])\$(?![\s$])((?:\\.|[^$\\\n])+?)(?<![\s\\])\$(?![\w$])")
PLACEHOLDER = re.compile(r"MATHPH([a-z]+)ENDPH")


def slugify(text, seen):
    """github-slugger: lowercase, drop punctuation except '-' and '_', spaces to '-', dedupe with -N."""
    base = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
    slug, n = base, seen.get(base, 0)
    if n:
        slug = f"{base}-{n}"
    seen[base] = n + 1
    return slug


def letters(n):
    s = ""
    while True:
        s = chr(ord("a") + n % 26) + s
        n = n // 26 - 1
        if n < 0:
            return s


def protect_math(src):
    """Swap math for letter-only placeholders outside code; returns the new text and the math list."""
    store = []

    def hold(expr, display):
        store.append((expr, display))
        return f"MATHPH{letters(len(store) - 1)}ENDPH"

    def inline_line(line):
        parts, pos = [], 0
        for m in CODE_SPAN.finditer(line):
            parts.append(math_text(line[pos:m.start()]))
            parts.append(m.group(0))
            pos = m.end()
        parts.append(math_text(line[pos:]))
        return "".join(parts)

    def math_text(text):
        text = DISPLAY.sub(lambda m: hold(m.group(1), True), text)
        return INLINE.sub(lambda m: hold(m.group(1), False), text)

    lines, out, fence, i = src.split("\n"), [], None, 0
    while i < len(lines):
        line = lines[i]
        m = FENCE.match(line)
        if fence:
            out.append(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not line.strip().strip(fence[0]):
                fence = None
            i += 1
            continue
        if m:
            fence = m.group(1)
            out.append(line)
            i += 1
            continue
        body = line.strip()
        if body.startswith("$$") and body.count("$$") == 1:
            # Display math over several lines: collect up to the closing $$.
            j = i + 1
            while j < len(lines) and "$$" not in lines[j]:
                j += 1
            if j < len(lines):
                end = lines[j].index("$$")
                expr = "\n".join([body[2:]] + lines[i + 1:j] + [lines[j][:end]])
                indent = line[:len(line) - len(line.lstrip())]
                out.append(indent + hold(expr, True) + lines[j][end + 2:])
                i = j + 1
                continue
        out.append(inline_line(line))
        i += 1
    return "\n".join(out), store


def restore_math(body, store):
    def one(m):
        idx = 0
        for ch in m.group(1):
            idx = idx * 26 + (ord(ch) - ord("a") + 1)
        expr, display = store[idx - 1]
        expr = html.escape(expr, quote=False)
        return f"\\[{expr}\\]" if display else f"\\({expr}\\)"

    # A display formula that is a paragraph of its own becomes a centred block.
    body = re.sub(r"<p>(MATHPH[a-z]+ENDPH)</p>", lambda m: f'<div class="math-display">{PLACEHOLDER.sub(one, m.group(1))}</div>', body)
    return PLACEHOLDER.sub(one, body)


def heading_text(inline):
    return "".join(c.content for c in inline.children or [] if c.type in ("text", "code_inline"))


def convert_alerts(tokens):
    """> [!NOTE] ... blockquotes -> <div class="markdown-alert markdown-alert-note">, as on GitHub."""
    for i, t in enumerate(tokens):
        if t.type != "blockquote_open" or i + 2 >= len(tokens):
            continue
        para, inline = tokens[i + 1], tokens[i + 2]
        if para.type != "paragraph_open" or inline.type != "inline":
            continue
        m = ALERT.match(inline.content)
        if not m:
            continue
        need, kids = len(m.group(0).rstrip()), inline.children
        while kids and need > 0 and kids[0].type == "text":
            if len(kids[0].content) <= need:
                need -= len(kids[0].content)
                kids.pop(0)
            else:
                kids[0].content = kids[0].content[need:].lstrip()
                need = 0
        while kids and (kids[0].type in ("softbreak", "hardbreak") or (kids[0].type == "text" and not kids[0].content.strip())):
            kids.pop(0)
        kind = m.group(1).lower()
        t.tag = "div"
        t.attrSet("class", f"markdown-alert markdown-alert-{kind}")
        depth = 0
        for close in tokens[i:]:
            if close.type == "blockquote_open":
                depth += 1
            elif close.type == "blockquote_close":
                depth -= 1
                if depth == 0:
                    close.tag = "div"
                    break


def refs(tokens):
    """(href, token, attribute) for every link and image; attribute is None for raw HTML."""
    for t in tokens:
        if t.type == "inline":
            for c in t.children or []:
                if c.type == "link_open":
                    yield c.attrGet("href"), c, "href"
                elif c.type == "image":
                    yield c.attrGet("src"), c, "src"
                elif c.type == "html_inline":
                    for m in re.finditer(r'\b(?:href|src)="([^"]*)"', c.content):
                        yield m.group(1), c, None
        elif t.type == "html_block":
            for m in re.finditer(r'\b(?:href|src|poster)="([^"]*)"', t.content):
                yield m.group(1), t, None


def set_ref(tok, attr, old, new):
    if attr:
        tok.attrSet(attr, new)
    else:
        tok.content = tok.content.replace(f'="{old}"', f'="{new}"')


def take_link_row(tokens):
    """Remove the first paragraph before any h2 that holds only links separated by " · "; return [(href, text)]."""
    for i, t in enumerate(tokens):
        if t.type == "heading_open" and t.tag == "h2":
            break
        if t.type != "paragraph_open" or i + 2 >= len(tokens) or tokens[i + 1].type != "inline":
            continue
        row, href, text, ok = [], None, "", True
        for c in tokens[i + 1].children or []:
            if c.type == "link_open":
                href, text = c.attrGet("href"), ""
            elif c.type == "link_close":
                row.append((href, text.strip()))
                href = None
            elif href is not None and c.type == "text":
                text += c.content
            elif href is None and (c.type == "softbreak" or (c.type == "text" and c.content.strip() in ("", "·"))):
                continue
            else:
                ok = False
                break
        if ok and len(row) >= 2:
            del tokens[i:i + 3]
            return row
    return []


def out_name(rel):
    """Path in the site: README.md is the index and docs/ is the root; .md becomes .html."""
    if rel == "README.md":
        return "index.html"
    rel = rel[len("docs/"):] if rel.startswith("docs/") else rel
    return rel[:-3] + ".html" if rel.endswith(".md") else rel


def site_link(target, frag, rel):
    """Relative link from the page built from rel to target."""
    return os.path.relpath(out_name(target), os.path.dirname(out_name(rel)) or ".") + (f"#{frag}" if frag else "")


def title_of(tokens, rel):
    for i, t in enumerate(tokens):
        if t.type == "heading_open" and t.tag == "h1":
            return heading_text(tokens[i + 1]).strip()
    return rel


def submodules():
    out = subprocess.run(["git", "-C", str(ROOT), "config", "-f", ".gitmodules", "--get-regexp", r"\.path$"],
                         capture_output=True, text=True).stdout
    return [line.split(" ", 1)[1] for line in out.splitlines()]


def unchecked(target, subs):
    """True for a path inside a submodule that is not checked out."""
    for sub in subs:
        d = ROOT / sub
        if (target + "/").startswith(sub + "/") and not (d.is_dir() and any(d.iterdir())):
            return True
    return False


def git_ignored(rel):
    # 0 = ignored, 1 = not ignored, 128 = e.g. a path inside a submodule (treated as not ignored).
    return subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", rel], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "anon-page" / "docs")
    args = ap.parse_args()
    out = args.out.resolve()
    if not any((ROOT / d).resolve() in out.parents for d in ("dist", "outputs")):
        raise SystemExit(f"--out must lie under {ROOT / 'dist'} or {ROOT / 'outputs'} (it is deleted and rebuilt): {out}")

    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    others, skipped = [], []
    for p in sorted((ROOT / "docs").rglob("*.md")):
        rel = str(p.relative_to(ROOT))
        if rel in MAIN:
            continue
        (skipped if git_ignored(rel) else others).append(rel)
    files = [f for f in MAIN if (ROOT / f).exists()]

    # Pass 1: parse every file and collect its heading ids.
    parsed, anchors, maths, toc = {}, {}, {}, {}

    def parse(rel):
        text, store = protect_math((ROOT / rel).read_text(encoding="utf-8"))
        tokens = md.parse(text)
        convert_alerts(tokens)
        seen, ids, sections = {}, set(), []
        for i, t in enumerate(tokens):
            if t.type == "heading_open":
                name = heading_text(tokens[i + 1])
                slug = slugify(name, seen)
                t.attrSet("id", slug)
                ids.add(slug)
                if t.tag in ("h2", "h3"):
                    sections.append((t.tag, slug, name))
        for t in tokens:  # ids written in raw HTML, e.g. <a id="...">
            if t.type in ("html_block", "html_inline") or t.type == "inline":
                for c in [t] + list(t.children or []):
                    if c.type in ("html_block", "html_inline"):
                        ids.update(re.findall(r'\bid="([^"]+)"', c.content))
        parsed[rel], anchors[rel], maths[rel], toc[rel] = tokens, ids, store, sections

    for rel in files:
        parse(rel)
    link_row = take_link_row(parsed["README.md"]) if "README.md" in parsed else []
    # Also render the other .md files under docs/ that the docs link to.
    linked, queue = [], list(files)
    while queue:
        rel = queue.pop(0)
        for href, tok, attr in refs(parsed[rel]):
            path = (href or "").partition("#")[0]
            if not path.endswith(".md") or SCHEME.match(href):
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(rel), path))
            if target.startswith("docs/") and target not in parsed and (ROOT / target).is_file():
                parse(target)
                linked.append(target)
                queue.append(target)
    files += linked

    # Pass 2: check and rewrite links, copy assets, render.
    broken, external, copied, code = [], {href for href, _ in link_row}, set(), set()
    unverified, subs = set(), submodules()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for rel in files:
        tokens, here = parsed[rel], os.path.dirname(rel)
        for href, tok, attr in refs(tokens):
            if not href:
                continue
            if SCHEME.match(href) or href.startswith("//"):
                external.add(href)
                continue
            path, _, frag = href.partition("#")
            target = os.path.normpath(os.path.join(here, path)) if path else rel
            src = ROOT / target
            if path and target != "README.md" and not target.startswith("docs/"):
                if unchecked(target, subs):
                    unverified.add(target)
                elif not src.exists():
                    broken.append(f"{rel}: {href} -> missing {target}")
                    continue
                url = f"{CODE_URL}/{target}" + (f"#{frag}" if frag else "")
                code.add(url)
                set_ref(tok, attr, href, url)
                continue
            if path and not src.exists():
                broken.append(f"{rel}: {href} -> missing {target}")
                continue
            if target.endswith(".md"):
                if target not in anchors:
                    broken.append(f"{rel}: {href} -> {target} is not rendered")
                elif frag and frag not in anchors[target]:
                    broken.append(f"{rel}: {href} -> no id #{frag} in {target}")
                if path:
                    set_ref(tok, attr, href, site_link(target, frag, rel))
                continue
            if frag and not path:
                broken.append(f"{rel}: {href} -> #{frag} is not an id in this page")
                continue
            if src.is_dir():
                broken.append(f"{rel}: {href} -> {target} is a folder, not part of the site")
                continue
            if src.stat().st_size > MAX_COPY_BYTES:
                broken.append(f"{rel}: {href} -> {target} is larger than {MAX_COPY_BYTES >> 20} MB, not copied")
                continue
            if target not in copied:
                (out / out_name(target)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out / out_name(target))
                copied.add(target)
            set_ref(tok, attr, href, site_link(target, frag, rel))

        body = md.renderer.render(tokens, md.options, {})
        body = restore_math(body, maths[rel])
        body = re.sub(r'<div class="markdown-alert markdown-alert-(\w+)">',
                      lambda m: f'{m.group(0)}\n<p class="markdown-alert-title">{m.group(1).capitalize()}</p>', body)
        # Hover anchor next to each heading, as on GitHub.
        body = re.sub(r'<(h[1-6]) id="([^"]*)">', r'<\1 id="\2"><a class="anchor" href="#\2" aria-hidden="true">#</a>', body)

        up = "../" * out_name(rel).count("/")

        def label(f):
            return NAV_LABEL.get(f) or title_of(parsed[f], f)

        def link(f, text=None):
            cls = ' class="here" aria-current="page"' if f == rel else ""
            return f'<a{cls} href="{up}{out_name(f)}">{html.escape(text or label(f))}</a>'

        nav = [link(f) for f in files if f in MAIN]
        buttons = []
        for href, text in link_row:
            todo = PLACEHOLDER_TOKEN in href
            extra = ' todo" title="Placeholder: the URL is set after submission' if todo else ""
            buttons.append(f'<a class="btn{extra}" href="{html.escape(href)}">{html.escape(text)}</a>')
        bar = f'<nav class="doclinks" aria-label="Project links">{"".join(buttons)}</nav>' if buttons else ""
        side = ""
        if toc[rel]:
            items = "\n".join(f'<a class="toc-{tag}" href="#{slug}">{html.escape(name)}</a>' for tag, slug, name in toc[rel])
            side = f'<aside class="toc"><div class="toc-head">On this page</div>\n{items}\n</aside>'
        pager = ""
        if rel in MAIN:
            k = MAIN.index(rel)
            prev = f'<a class="prev" href="{up}{out_name(MAIN[k - 1])}"><span>Previous</span>{html.escape(label(MAIN[k - 1]))}</a>' if k > 0 else "<span></span>"
            nxt = f'<a class="next" href="{up}{out_name(MAIN[k + 1])}"><span>Next</span>{html.escape(label(MAIN[k + 1]))}</a>' if k + 1 < len(MAIN) else "<span></span>"
            pager = f'<nav class="pager" aria-label="Previous and next page">{prev}{nxt}</nav>'
        katex = KATEX_HEAD if maths[rel] else ""
        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<link rel="icon" href="{up}favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="32x32" href="{up}favicon-32.png">
<link rel="apple-touch-icon" href="{up}apple-touch-icon.png">
<link rel="stylesheet" href="{up}github-markdown-light.css">
{katex}
<style>
  :root {{ --teal: #007f7f; --teal-dark: #006666; --teal-soft: #e6f2f2; --high: #c72f2e; --moderate: #c78b11; --low: #309341;
          --ink: #1f2328; --muted: #57606a; --line: #d0d7de; --panel: #f6f8fa;
          --font: "Helvetica Neue", "Nimbus Sans", Helvetica, Arial, sans-serif; }}
  body {{ margin: 0; background: #fff; color: var(--ink); font-family: var(--font); }}
  .layout {{ display: grid; grid-template-columns: 220px minmax(0, 1fr) 200px; gap: 40px; max-width: 1400px; margin: 0 auto; padding: 28px 24px; box-sizing: border-box; }}
  .sidebar, .toc {{ position: sticky; top: 28px; align-self: start; max-height: calc(100vh - 56px); overflow-y: auto; font: 14px/1.5 var(--font); }}
  .docbar {{ border-bottom: 1px solid var(--line); }}
  .docbar-inner {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px 24px; max-width: 1400px; margin: 0 auto; padding: 12px 24px; box-sizing: border-box; }}
  .docbar .brand {{ font-weight: 700; font-size: 18px; color: var(--ink); text-decoration: none; }}
  .doclinks {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .doclinks .btn {{ display: inline-flex; align-items: center; height: 34px; padding: 0 16px; border: 1px solid var(--teal); border-radius: 999px; font: 600 14px/1 var(--font); color: #fff; background: var(--teal); text-decoration: none; }}
  .doclinks .btn:hover {{ border-color: var(--teal-dark); background: var(--teal-dark); }}
  .doclinks .btn.todo {{ border: 1px dashed var(--line); color: var(--muted); background: var(--panel); }}
  .sidebar a {{ display: block; padding: 5px 8px; border-left: 3px solid transparent; color: var(--ink); text-decoration: none; overflow-wrap: anywhere; }}
  .sidebar a:hover {{ background: var(--panel); color: var(--teal-dark); }}
  .sidebar a.here {{ border-left-color: var(--teal); background: var(--teal-soft); color: var(--teal-dark); font-weight: 600; }}
  .toc {{ font-size: 13px; }}
  .toc .toc-head {{ padding: 0 0 8px 10px; color: var(--muted); font-size: 12px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }}
  .toc a {{ display: block; padding: 3px 0 3px 10px; border-left: 2px solid var(--line); color: var(--muted); text-decoration: none; }}
  .toc a.toc-h3 {{ padding-left: 22px; }}
  .toc a:hover {{ color: var(--teal-dark); border-left-color: var(--teal); }}
  main.markdown-body {{ min-width: 0; max-width: 900px; font-family: var(--font); color: var(--ink); }}
  .markdown-body a {{ color: var(--teal); }}
  .markdown-body a:hover {{ color: var(--teal-dark); }}
  .markdown-body h1, .markdown-body h2 {{ border-bottom-color: var(--line); }}
  .markdown-body .anchor {{ float: left; margin-left: -20px; padding-right: 4px; color: var(--muted); text-decoration: none; visibility: hidden; }}
  .markdown-body :is(h1, h2, h3, h4, h5, h6):hover .anchor {{ visibility: visible; }}
  .markdown-body :is(h1, h2, h3, h4) {{ scroll-margin-top: 16px; }}
  .markdown-body table th {{ background: var(--panel); }}
  .markdown-body details {{ margin: 0 0 16px; padding: 10px 16px; border: 1px solid var(--line); border-radius: 6px; }}
  .markdown-body details[open] > summary {{ margin-bottom: 12px; }}
  .markdown-body summary {{ color: var(--teal-dark); font-weight: 600; cursor: pointer; }}
  .markdown-body img {{ border-radius: 4px; }}
  .math-display {{ margin: 0 0 16px; overflow-x: auto; overflow-y: hidden; }}
  .markdown-alert {{ margin: 0 0 16px; padding: 8px 16px; border-left: 4px solid var(--c); background: color-mix(in srgb, var(--c) 7%, #fff); border-radius: 0 6px 6px 0; }}
  .markdown-alert > :last-child {{ margin-bottom: 0; }}
  .markdown-alert-title {{ margin-bottom: 4px !important; color: var(--c); font-weight: 600; }}
  .markdown-alert-note {{ --c: var(--teal); }}
  .markdown-alert-tip {{ --c: var(--low); }}
  .markdown-alert-important {{ --c: var(--teal-dark); }}
  .markdown-alert-warning {{ --c: var(--moderate); }}
  .markdown-alert-caution {{ --c: var(--high); }}
  .pager {{ display: flex; justify-content: space-between; gap: 16px; margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line); }}
  .pager a {{ display: block; max-width: 48%; padding: 10px 14px; border: 1px solid var(--line); border-radius: 6px; color: var(--teal-dark); font-weight: 600; text-decoration: none; }}
  .pager a:hover {{ border-color: var(--teal); background: var(--teal-soft); }}
  .pager a span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 400; }}
  .pager .next {{ text-align: right; }}
  @media (max-width: 1100px) {{
    .layout {{ grid-template-columns: 200px minmax(0, 1fr); }}
    .toc {{ display: none; }}
  }}
  @media (max-width: 760px) {{
    .docbar-inner {{ padding: 10px 16px; }}
    .layout {{ display: block; padding: 16px; }}
    .sidebar {{ position: static; max-height: none; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid var(--line); }}
  }}
</style>
</head>
<body>
<header class="docbar"><div class="docbar-inner"><a class="brand" href="{up}index.html">{BRAND}</a>{bar}</div></header>
<div class="layout">
<nav class="sidebar" aria-label="Documentation">
{chr(10).join(nav)}
</nav>
<main class="markdown-body">
{body}
{pager}
</main>
{side}
</div>
</body>
</html>
"""
        dst = out / out_name(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(page, encoding="utf-8")
    shutil.copy(CSS, out / CSS.name)
    for name, src in ICONS.items():
        shutil.copy2(ROOT / src, out / name)

    print(f"rendered {len(files)} files into {out}")
    for f in files:
        print(f"  {out_name(f)}")
    print(f"link row: {' | '.join(f'{t} -> {h}' for h, t in link_row) or 'none'}")
    unlinked = [f for f in others if f not in parsed]
    if unlinked:
        print(f"not rendered (under docs/, not linked from the docs): {', '.join(unlinked)}")
    if skipped:
        print(f"skipped (git-ignored): {', '.join(skipped)}")
    print(f"copied {len(copied)} linked files: {', '.join(sorted(copied))}")
    print(f"links to the code repo: {len(code)}")
    for c in sorted(code):
        print(f"  {c}")
    if unverified:
        print(f"not checked (submodule not checked out): {', '.join(sorted(unverified))}")
    print(f"external links (not checked): {len(external)}")
    for e in sorted(external):
        print(f"  {e}")
    print(f"broken links: {len(broken)}")
    for b in broken:
        print(f"  {b}")
    if broken:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
