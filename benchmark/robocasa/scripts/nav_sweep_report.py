"""Build a single browsable HTML page over a nav_sweep output tree.

The sweep leaves three things worth eyeballing side by side:

    overlay/<LAYOUT>/<tier>/diff/overlay_<KEY>.png   colour-coded still
    overlay/<LAYOUT>/<tier>/grid/overlay_<KEY>.mp4   small-multiples clip
    stability/validation_<LAYOUT>.csv                per-episode drift

...where <KEY> is <Mode>Route<R>_<LAYOUT>_<STYLE>. This writes
<root>/index.html, which pairs the diff and the grid for every scene behind
a layout/tier picker and puts the stability verdict on its own tab.

    python scripts/nav_sweep_report.py [figures/nav_sweep]

Asset paths in the page are relative to <root>, so the tree stays movable
and the page opens straight off the filesystem (file://) with no server.
"""
import argparse
import concurrent.futures
import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Same tolerances the standalone verdict script uses, so the page and
# scripts/nav_sweep_stability.py cannot disagree about what "stable" means.
XY_TOL, Z_UP_TOL, Z_DOWN_TOL = 0.05, 0.05, 0.10

KEY_RE = re.compile(r"^overlay_((Non)?Blocking)Route([A-G])_(.+)$")

# Mirrors DIFF_PALETTE in scripts/overlay_obstacles.py. The colour an
# obstacle gets there is its index in sorted(members of that scene), so the
# legend here is only correct if both lists stay in this order.
DIFF_PALETTE_HEX = [
    "#ff0000", "#00ff00", "#4d80ff", "#ffff00", "#ff00ff",
    "#00ffff", "#ff8c00", "#b300ff", "#ffffff", "#99ff00",
]

# The sweep's tier directory names, in the order the page shows them, and
# the roster tier each one maps to.
TIER_DIRS = [("high", "High"), ("moderate", "Medium"), ("low", "Low")]


def tier_members():
    """{tier_dir: (obstacle, ...)} — from the roster, else from the sweep."""
    try:
        from robocasa.metrics.ssi import ROSTER
        by_name = {t.capitalize(): tuple(obs) for t, obs in ROSTER.items()}
        return {d: by_name[t] for d, t in TIER_DIRS}
    except Exception as exc:  # roster unavailable: fall back to the sweep
        print(f"[warn] roster import failed ({exc}); parsing nav_sweep.sh",
              file=sys.stderr)
        sh = (Path(__file__).parent / "nav_sweep.sh").read_text()
        out = {}
        for d, _ in TIER_DIRS:
            m = re.search(rf"^{d.upper()}=\((.*?)\)$", sh, re.M)
            out[d] = tuple(m.group(1).split()) if m else ()
        return out


def scene_members(videos_root, layout, members, key):
    """Which of `members` actually contributed a clip to this scene.

    overlay_obstacles.py colours by index into sorted(present members), and
    a scene can be short one member (the human is RouteF's target, so it is
    never also RouteF's obstacle). Recovering the real list per scene keeps
    the legend and the panel map honest.
    """
    d = videos_root / layout
    return [o for o in sorted(members)
            if d.is_dir() and any((d / o).glob(f"*{key}.mp4"))]


def collect_scenes(root, members_by_tier):
    videos_root = root / "videos"
    overlay_root = root / "overlay"
    layouts, scenes = [], {}
    if not overlay_root.is_dir():
        return layouts, scenes

    for layout_dir in sorted(p for p in overlay_root.iterdir() if p.is_dir()):
        layout = layout_dir.name
        for tier, _ in TIER_DIRS:
            diff_dir, grid_dir = layout_dir / tier / "diff", layout_dir / tier / "grid"
            keys = set()
            for d, ext in ((diff_dir, ".png"), (grid_dir, ".mp4")):
                if d.is_dir():
                    keys |= {p.stem for p in d.glob(f"*{ext}")}
            if not keys:
                continue

            items = []
            for stem in sorted(keys, key=lambda s: (KEY_RE.match(s).group(3),
                                                    KEY_RE.match(s).group(1))
                               if KEY_RE.match(s) else (s, "")):
                m = KEY_RE.match(stem)
                if not m:
                    continue
                mode, _, route, tail = m.groups()
                key = stem[len("overlay_"):]
                present = scene_members(videos_root, layout,
                                        members_by_tier[tier], key)
                cols = math.ceil(math.sqrt(len(present))) if present else 1
                png, mp4 = diff_dir / f"{stem}.png", grid_dir / f"{stem}.mp4"
                items.append({
                    "key": key,
                    "route": route,
                    "mode": mode,
                    "style": tail[len(layout) + 1:] if tail.startswith(layout) else tail,
                    "diff": os.path.relpath(png, root) if png.exists() else None,
                    "grid": os.path.relpath(mp4, root) if mp4.exists() else None,
                    "poster": None,
                    "_poster_src": mp4 if mp4.exists() else None,
                    "_poster_dst": root / "posters" / layout / tier / f"{stem}.jpg",
                    "members": present,
                    "cols": cols,
                    "rows": math.ceil(len(present) / cols) if present else 0,
                })
            if items:
                layouts.append(layout) if layout not in layouts else None
                scenes[f"{layout}|{tier}"] = items
    return layouts, scenes


def build_posters(root, scenes, jobs=8):
    """One JPEG per grid clip, so a card shows its scene without fetching mp4s.

    A grid sheet is 4608x2048 and a layout/tier holds 14 of them; letting the
    browser pull every clip just to paint a first frame stalls the page. The
    still is extracted once and reused on every later run.
    """
    if shutil.which("ffmpeg") is None:
        print("[warn] ffmpeg not on PATH; grid clips will show black until played",
              file=sys.stderr)
        return 0

    def one(scene):
        src, dst = scene["_poster_src"], scene["_poster_dst"]
        if src is None:
            return False
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            dst.parent.mkdir(parents=True, exist_ok=True)
            res = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-frames:v", "1",
                 "-vf", "scale=1280:-2", "-q:v", "4", str(dst)],
                capture_output=True, text=True)
            if res.returncode != 0:
                return False
        scene["poster"] = os.path.relpath(dst, root)
        return True

    todo = [s for items in scenes.values() for s in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        return sum(ex.map(one, todo))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def collect_stability(root):
    """Per-layout counts plus the worst episode per obstacle."""
    paths = sorted((root / "stability").glob("validation_*.csv"))
    rows = []
    for p in paths:
        layout = p.stem[len("validation_"):]
        for r in csv.DictReader(p.open()):
            r["_layout"] = layout
            rows.append(r)
    if not rows:
        return None

    def over(r):
        return (num(r["xy_drift_max"]) > XY_TOL
                or num(r["z_drift_up_max"]) > Z_UP_TOL
                or num(r["z_drift_down_max"]) > Z_DOWN_TOL)

    def popped(r):
        return r["pop_out"] not in ("0", "", "False")

    per_layout = defaultdict(lambda: {"n": 0, "bad": 0, "pop": 0, "over": 0})
    for r in rows:
        e = per_layout[r["_layout"]]
        e["n"] += 1
        e["bad"] += r["status"] != "success"
        e["pop"] += popped(r)
        e["over"] += over(r)

    worst = {}
    for r in rows:
        obs = r["obstacle"] or "?"
        cur = (num(r["xy_drift_max"]), num(r["z_drift_up_max"]),
               num(r["z_drift_down_max"]))
        if obs not in worst or max(cur) > max(worst[obs][:3]):
            worst[obs] = cur + (f"{r['env_name']} / {r['_layout']}",)

    failures = [{
        "env": r["env_name"], "layout": r["_layout"], "obstacle": r["obstacle"],
        "status": r["status"],
        "xy": num(r["xy_drift_max"]) * 100,
        "up": num(r["z_drift_up_max"]) * 100,
        "down": num(r["z_drift_down_max"]) * 100,
        "pop": bool(popped(r)), "error": (r["error"] or "")[:120],
    } for r in rows if r["status"] != "success" or popped(r) or over(r)]

    return {
        "episodes": len(rows),
        "layouts": len(paths),
        "bad_status": sum(r["status"] != "success" for r in rows),
        "popped": sum(popped(r) for r in rows),
        "over": sum(over(r) for r in rows),
        "per_layout": {k: v for k, v in sorted(per_layout.items())},
        "worst": [{"obstacle": o, "xy": w[0] * 100, "up": w[1] * 100,
                   "down": w[2] * 100, "cell": w[3]}
                  for o, w in sorted(worst.items(), key=lambda kv: -max(kv[1][:3]))],
        "failures": failures,
        "tol": {"xy": XY_TOL * 100, "up": Z_UP_TOL * 100, "down": Z_DOWN_TOL * 100},
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#0e1116; --panel:#161b22; --panel2:#1c232c; --line:#2a323d;
  --fg:#e6edf3; --dim:#8b949e; --accent:#4d9fff; --ok:#3fb950; --bad:#f85149;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;z-index:20;background:rgba(14,17,22,.94);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:10px 16px}
h1{margin:0 0 8px;font-size:15px;font-weight:600;letter-spacing:.2px}
h1 span{color:var(--dim);font-weight:400;margin-left:8px}
.bar{display:flex;flex-wrap:wrap;gap:14px;align-items:center}
.grp{display:flex;align-items:center;gap:6px}
.grp>b{color:var(--dim);font-weight:500;font-size:11px;text-transform:uppercase;
  letter-spacing:.6px;margin-right:2px}
button{font:inherit;color:var(--fg);background:var(--panel2);cursor:pointer;
  border:1px solid var(--line);border-radius:6px;padding:4px 10px}
button:hover{border-color:#3d4855}
button.on{background:var(--accent);border-color:var(--accent);color:#04101f;font-weight:600}
main{padding:16px;max-width:2200px;margin:0 auto}
.cards{display:grid;gap:14px;
  grid-template-columns:repeat(auto-fill,minmax(min(100%,680px),1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  overflow:hidden;display:flex;flex-direction:column}
.card>h3{margin:0;padding:8px 12px;font-size:13px;font-weight:600;
  border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}
.tag{font-size:10px;font-weight:600;letter-spacing:.5px;padding:2px 6px;
  border-radius:999px;background:var(--panel2);color:var(--dim);text-transform:uppercase}
.tag.block{background:#3a1d1d;color:#ff9d9d}
.tag.nonblock{background:#132a1b;color:#8fe0a5}
.media{display:grid;gap:1px;background:var(--line)}
.media.both{grid-template-columns:1fr}
figure{margin:0;background:#000;position:relative;min-height:80px}
figure>figcaption{position:absolute;left:6px;top:6px;font-size:10px;
  letter-spacing:.5px;text-transform:uppercase;color:#cfd8e3;
  background:rgba(0,0,0,.55);padding:2px 6px;border-radius:4px;pointer-events:none}
figure img,figure video{display:block;width:100%;height:auto;cursor:zoom-in}
.legend{display:flex;flex-wrap:wrap;gap:5px 10px;padding:8px 12px;
  border-top:1px solid var(--line)}
.chip{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--dim)}
.dot{width:9px;height:9px;border-radius:2px;box-shadow:0 0 0 1px rgba(0,0,0,.6)}
.pos{color:#5c6672;font-variant-numeric:tabular-nums}
.miss{padding:24px;color:var(--dim);text-align:center;font-size:12px}
#lightbox{position:fixed;inset:0;z-index:50;background:rgba(4,6,10,.94);
  display:none;align-items:center;justify-content:center;padding:24px;cursor:zoom-out}
#lightbox.open{display:flex}
#lightbox>*{max-width:100%;max-height:100%;object-fit:contain}
table{border-collapse:collapse;width:100%;font-size:12.5px;
  font-variant-numeric:tabular-nums}
th,td{padding:5px 10px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500;font-size:11px;text-transform:uppercase;
  letter-spacing:.5px}
td.n,th.n{text-align:right}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:600;
  font-size:12px}
.pill.ok{background:#0f2d18;color:var(--ok)} .pill.bad{background:#3a1416;color:var(--bad)}
.panels{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
  margin-top:14px}
.panels>section{background:var(--panel);border:1px solid var(--line);
  border-radius:10px;overflow:hidden}
.panels h4{margin:0;padding:8px 12px;font-size:12px;border-bottom:1px solid var(--line)}
.hide{display:none}
</style></head><body>
<header>
  <h1>__TITLE__<span id="sub"></span></h1>
  <div class="bar">
    <div class="grp"><b>view</b><span id="tabs"></span></div>
    <div class="grp" id="pick-layout"><b>layout</b><span></span></div>
    <div class="grp" id="pick-tier"><b>tier</b><span></span></div>
    <div class="grp" id="pick-mode"><b>show</b><span></span></div>
    <div class="grp" id="pick-play"><button id="playall">play visible</button>
      <button id="pauseall">pause</button></div>
  </div>
</header>
<main>
  <div id="gallery"><div class="cards" id="cards"></div></div>
  <div id="stability" class="hide"></div>
</main>
<div id="lightbox"></div>
<script>
const DATA = __DATA__;
const PALETTE = __PALETTE__;
const $ = (s, r = document) => r.querySelector(s);
const el = (t, a = {}, ...kids) => {
  const n = Object.assign(document.createElement(t), a);
  for (const k of kids) n.append(k);
  return n;
};
// The whole view lives in the URL hash, so a particular layout/tier is a
// link someone else can open on the same tree.
const state = {
  view: "gallery",
  layout: DATA.layouts[0] || null,
  tier: DATA.tiers[0],
  show: "both",
};
const ALLOWED = {
  view: ["gallery", "stability"], layout: DATA.layouts,
  tier: DATA.tiers, show: ["both", "diff", "grid"],
};
function readHash() {
  for (const part of location.hash.replace(/^#/, "").split("&")) {
    const [k, v] = part.split("=");
    if (ALLOWED[k] && ALLOWED[k].includes(decodeURIComponent(v || "")))
      state[k] = decodeURIComponent(v);
  }
}
function writeHash() {
  const h = "#" + ["view", "layout", "tier", "show"]
    .filter(k => state[k]).map(k => `${k}=${encodeURIComponent(state[k])}`).join("&");
  if (h !== location.hash) history.replaceState(null, "", h);
}
readHash();
addEventListener("hashchange", () => { readHash(); render(); });

function seg(host, values, key, labels) {
  const box = host.querySelector("span");
  box.textContent = "";
  values.forEach(v => {
    const b = el("button", { textContent: labels ? labels[v] : v });
    b.onclick = () => { state[key] = v; render(); };
    b.classList.toggle("on", state[key] === v);
    box.append(b);
  });
}

function lightbox(src, isVideo) {
  const lb = $("#lightbox");
  lb.textContent = "";
  lb.append(isVideo
    ? el("video", { src, controls: true, autoplay: true, loop: true, muted: true })
    : el("img", { src }));
  lb.classList.add("open");
}
$("#lightbox").onclick = () => {
  $("#lightbox").classList.remove("open");
  $("#lightbox").textContent = "";
};
addEventListener("keydown", e => {
  if (e.key === "Escape") $("#lightbox").click();
});

// Videos are 4608x2048 six-panel sheets; loading every one at once stalls the
// page, so a clip only fetches data once its card scrolls into view.
const io = new IntersectionObserver(entries => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const v = e.target;
    if (!v.src) { v.preload = "metadata"; v.src = v.dataset.src + "#t=0.1"; }
  }
}, { rootMargin: "300px" });

function card(s) {
  const c = el("div", { className: "card" });
  c.append(el("h3", {},
    el("span", { textContent: `Route ${s.route}` }),
    el("span", {
      className: "tag " + (s.mode === "Blocking" ? "block" : "nonblock"),
      textContent: s.mode === "Blocking" ? "blocking" : "non-blocking",
    }),
    el("span", { className: "tag", textContent: `${s.members.length} obstacles` }),
    el("span", { className: "tag", textContent: s.style })));

  const media = el("div", { className: "media " + state.show });
  if (state.show !== "grid" && s.diff) {
    const img = el("img", { src: DATA.base + s.diff, loading: "lazy", alt: s.key });
    img.onclick = () => lightbox(img.src, false);
    media.append(el("figure", {}, img, el("figcaption", { textContent: "diff" })));
  }
  if (state.show !== "diff" && s.grid) {
    const v = el("video", { muted: true, loop: true, playsInline: true,
                            controls: true, preload: "none" });
    if (s.poster) v.poster = DATA.base + s.poster;
    v.dataset.src = DATA.base + s.grid;
    v.ondblclick = () => lightbox(v.dataset.src, true);
    io.observe(v);
    media.append(el("figure", {}, v,
      el("figcaption", { textContent: `grid ${s.cols}x${s.rows}` })));
  }
  if (!media.children.length) media.append(el("div", { className: "miss",
    textContent: "no overlay rendered for this scene" }));
  c.append(media);

  const lg = el("div", { className: "legend" });
  s.members.forEach((o, i) => {
    const r = Math.floor(i / s.cols) + 1, col = (i % s.cols) + 1;
    lg.append(el("span", { className: "chip" },
      el("span", { className: "dot",
                   style: `background:${PALETTE[i % PALETTE.length]}` }),
      document.createTextNode(o),
      el("span", { className: "pos", textContent: `r${r}c${col}` })));
  });
  c.append(lg);
  return c;
}

function renderGallery() {
  const cards = $("#cards");
  cards.textContent = "";
  const list = DATA.scenes[`${state.layout}|${state.tier}`] || [];
  if (!list.length) {
    cards.append(el("div", { className: "miss",
      textContent: "nothing rendered for this layout/tier yet" }));
  }
  list.forEach(s => cards.append(card(s)));
  $("#sub").textContent =
    `${state.layout} · ${state.tier} tier · ${list.length} scenes`;
}

function tbl(head, rows, cls = []) {
  const t = el("table");
  t.append(el("thead", {}, el("tr", {}, ...head.map((h, i) =>
    el("th", { textContent: h, className: cls[i] || "" })))));
  const body = el("tbody");
  rows.forEach(r => body.append(el("tr", {}, ...r.map((v, i) =>
    el("td", { className: cls[i] || "" }, v instanceof Node ? v
      : document.createTextNode(String(v)))))));
  t.append(body);
  return t;
}

function renderStability() {
  const host = $("#stability");
  host.textContent = "";
  const st = DATA.stability;
  if (!st) {
    host.append(el("div", { className: "miss",
      textContent: "no validation CSVs found under stability/" }));
    return;
  }
  const clean = !st.bad_status && !st.popped && !st.over;
  host.append(el("div", {},
    el("span", { className: "pill " + (clean ? "ok" : "bad"),
      textContent: clean ? "all episodes stable" : "unstable cells present" }),
    el("span", { style: "margin-left:12px;color:var(--dim)",
      textContent: `${st.episodes} episodes over ${st.layouts} layouts · `
        + `tolerance xy>${st.tol.xy}cm, z_up>${st.tol.up}cm, `
        + `z_down>${st.tol.down}cm` })));

  const panels = el("div", { className: "panels" });
  const p1 = el("section", {}, el("h4", { textContent: "per layout" }));
  p1.append(tbl(["layout", "episodes", "failed", "pop_out", "over tol"],
    Object.entries(st.per_layout).map(([k, v]) =>
      [k, v.n, v.bad, v.pop, v.over]),
    ["", "n", "n", "n", "n"]));
  const p2 = el("section", {},
    el("h4", { textContent: "worst episode per obstacle (cm)" }));
  p2.append(tbl(["obstacle", "xy", "z up", "z down", "cell"],
    st.worst.map(w => [w.obstacle, w.xy.toFixed(2), w.up.toFixed(2),
                       w.down.toFixed(2), w.cell]),
    ["", "n", "n", "n", ""]));
  panels.append(p1, p2);
  host.append(panels);

  if (st.failures.length) {
    const p3 = el("section", { style: "margin-top:12px" },
      el("h4", { textContent: `flagged episodes (${st.failures.length})` }));
    p3.append(tbl(["env", "layout", "obstacle", "status", "xy", "up", "down", "pop", "error"],
      st.failures.map(f => [f.env, f.layout, f.obstacle, f.status,
        f.xy.toFixed(2), f.up.toFixed(2), f.down.toFixed(2),
        f.pop ? "yes" : "", f.error]),
      ["", "", "", "", "n", "n", "n", "", ""]));
    host.append(p3);
  }
}

function render() {
  writeHash();
  seg($("#tabs").parentElement, ["gallery", "stability"], "view");
  seg($("#pick-layout"), DATA.layouts, "layout");
  seg($("#pick-tier"), DATA.tiers, "tier");
  seg($("#pick-mode"), ["both", "diff", "grid"], "show");
  const gal = state.view === "gallery";
  $("#gallery").classList.toggle("hide", !gal);
  $("#stability").classList.toggle("hide", gal);
  ["#pick-layout", "#pick-tier", "#pick-mode", "#pick-play"].forEach(s =>
    $(s).classList.toggle("hide", !gal));
  gal ? renderGallery() : renderStability();
}

$("#playall").onclick = () => document.querySelectorAll("video").forEach(v => {
  const r = v.getBoundingClientRect();
  if (r.bottom < 0 || r.top > innerHeight) return;   // offscreen: leave it cold
  if (!v.src) v.src = v.dataset.src;
  v.play().catch(() => {});
});
$("#pauseall").onclick = () =>
  document.querySelectorAll("video").forEach(v => v.pause());

// #tabs is only an anchor for the view segment; seg() fills its parent.
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default="figures/nav_sweep", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output html (default: <root>/index.html)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-posters", action="store_true",
                    help="skip extracting a still per grid clip (the cards "
                         "then show black until a clip is played)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel ffmpeg poster extractions (default: 8)")
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"[error] root not found: {root}")
    out = args.out or root / "index.html"

    members_by_tier = tier_members()
    layouts, scenes = collect_scenes(root, members_by_tier)
    if not scenes:
        raise SystemExit(f"[error] no overlays under {root}/overlay")
    stability = collect_stability(root)
    if not args.no_posters:
        n = build_posters(root, scenes, jobs=args.jobs)
        print(f"[ok] {n} grid posters under {root / 'posters'}")
    for items in scenes.values():
        for s_ in items:
            s_.pop("_poster_src", None), s_.pop("_poster_dst", None)

    # Asset hrefs are relative to the html's own directory, so a report
    # written outside the tree still resolves.
    base = "" if out.parent.resolve() == root.resolve() else \
        os.path.relpath(root, out.parent) + "/"

    data = {
        "base": base,
        "layouts": layouts,
        "tiers": [d for d, _ in TIER_DIRS],
        "scenes": scenes,
        "stability": stability,
    }
    title = args.title or f"{root.name} — navigate_safe sweep"
    page = (PAGE.replace("__TITLE__", html.escape(title))
                .replace("__PALETTE__", json.dumps(DIFF_PALETTE_HEX))
                .replace("__DATA__", json.dumps(data)))
    out.write_text(page)

    n_scenes = sum(len(v) for v in scenes.values())
    n_diff = sum(1 for v in scenes.values() for s in v if s["diff"])
    n_grid = sum(1 for v in scenes.values() for s in v if s["grid"])
    print(f"[ok] {out}  ({len(layouts)} layouts x {len(TIER_DIRS)} tiers, "
          f"{n_scenes} scenes, {n_diff} diff stills, {n_grid} grid clips)")
    if stability:
        clean = not (stability["bad_status"] or stability["popped"]
                     or stability["over"])
        print(f"[stability] {stability['episodes']} episodes, "
              f"{'all stable' if clean else 'FLAGGED: ' + str(len(stability['failures']))}")


if __name__ == "__main__":
    main()
