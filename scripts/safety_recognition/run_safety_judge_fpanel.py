#!/usr/bin/env python3
"""VLM safety-score judge for the fpanel_cm_sweep dataset.

Same inference core as run_safety_judge_samecm.py, but instead of a manifest it
walks the on-disk panel tree

    {img_root}/cm{NN}/{n}_{Tier}/L{layout}_{Obstacle}_Route{R}.png

where cm{NN} is the obstacle-distance bucket (00,05,...,80) and the tier folder
is 1_High / 2_Medium / 3_Low.  Every panel is one "(f) Trajectory" figure; the
same task is re-rendered at all 17 cm buckets, so the judge sweeps obstacle
distance.  Results stream to JSONL; rerun with --resume to skip done panels.

Example (run inside the container, vLLM server on localhost:8006):
    docker exec robotics-safety-container python3 \
      scripts/safety_recognition/run_safety_judge_fpanel.py \
      --img-root /workspace/storyboard_overlay_cropped_final \
      --prompt   scripts/safety_recognition/prompts/safety_prompt.txt \
      --model    Qwen/Qwen3-VL-32B-Instruct \
      --base-url http://localhost:8006/v1 \
      --scale    binary \
      --workers  8 \
      --output   outputs/safety_judgments/qwen3_vl_32b_binary_v5.jsonl
"""
from __future__ import annotations

# Defensive: drop our directory from sys.path[0] before importing asyncio,
# in case some sibling module would shadow a stdlib name.
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:] = [p for p in _sys.path if _os.path.abspath(p) != _HERE]

import argparse
import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(\d)", re.IGNORECASE)
# Kimi-Thinking emits "◁think▷ ... ◁/think▷N" with the verdict as a bare digit
# right after the closing think tag (no "Final Answer:" prefix). The patterns
# below cover that and a couple of common variants other reasoning models use.
POST_THINK_RES = [
    re.compile(r"◁/think▷\s*(\d)"),       # Kimi-Thinking native
    re.compile(r"</think>\s*(\d)"),        # other reasoning models
    re.compile(r"answer[^0-9]{0,10}(\d)\s*$", re.IGNORECASE),  # "answer is N" at end
]


def _accept(n: int, scale: str) -> int | None:
    if scale == "binary":
        return n if n in (0, 1) else None        # 1 = safe, 0 = unsafe
    return n if 1 <= n <= 5 else None


_BEGIN_OF_BOX_RE = re.compile(r"<\|begin_of_box\|>\s*(\d)")


def parse_score(text: str | None, scale: str = "1-5") -> int | None:
    if not text:
        return None
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return _accept(int(m.group(1)), scale)
    for rx in POST_THINK_RES:
        m = rx.search(text)
        if m:
            return _accept(int(m.group(1)), scale)
    m = _BEGIN_OF_BOX_RE.search(text)
    if m:
        return _accept(int(m.group(1)), scale)
    stripped = text.rstrip()
    if stripped and stripped[-1].isdigit():
        return _accept(int(stripped[-1]), scale)
    return None


def discover_jobs(img_root: Path) -> list[dict]:
    """Walk cm{NN}/{n}_{Tier}/L{layout}_{Obstacle}_Route{R}.png into job dicts."""
    jobs = []
    for png in sorted(img_root.glob("cm[0-9][0-9]/[0-9]_*/*.png")):
        cm = int(png.parent.parent.name.removeprefix("cm"))
        tier = png.parent.name.split("_", 1)[1]              # 1_High -> High
        layout, obstacle, route = png.stem.split("_")        # L0_Cat_RouteA
        jobs.append({
            "cm": cm,
            "tier": tier,
            "layout": int(layout.removeprefix("L")),
            "obstacle": obstacle,
            "route": route.removeprefix("Route"),
            "img": str(png),
        })
    return jobs


async def judge_one(client, model, prompt, img_path, max_tokens, sem,
                    freq_penalty=0.0, pres_penalty=0.0, stop=None):
    async with sem:
        b64 = base64.b64encode(Path(img_path).read_bytes()).decode("utf-8")
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }]
        t0 = time.time()
        try:
            kwargs = dict(
                model=model, messages=msgs, max_tokens=max_tokens,
                temperature=0.0,
                frequency_penalty=freq_penalty,
                presence_penalty=pres_penalty,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            if stop:
                kwargs["stop"] = stop
            resp = await client.chat.completions.create(**kwargs)
            raw, err = resp.choices[0].message.content, None
        except Exception as e:
            raw, err = "", repr(e)
        return raw, err, (time.time() - t0) * 1000


async def main_async(args):
    prompt = Path(args.prompt).read_text()
    jobs = discover_jobs(Path(args.img_root))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Save a copy of the prompt next to the output JSONL for reproducibility.
    # File name mirrors the output stem so it's obvious which prompt produced
    # which JSONL: judgments_X.jsonl <-> judgments_X.prompt.txt
    try:
        prompt_copy = out.with_suffix(".prompt.txt")
        prompt_copy.write_text(prompt)
    except Exception as e:
        print(f"WARN could not save prompt copy: {e}", file=sys.stderr, flush=True)

    done: set[str] = set()
    if args.resume and out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["img"])
            except Exception:
                pass
        jobs = [j for j in jobs if j["img"] not in done]
    print(f"jobs={len(jobs)}  (resumed past {len(done)})",
          file=sys.stderr, flush=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY",
                         timeout=args.timeout)
    sem = asyncio.Semaphore(args.workers)

    async def run(job):
        raw, err, dt = await judge_one(client, args.model, prompt, job["img"],
                                       args.max_tokens, sem,
                                       freq_penalty=args.frequency_penalty,
                                       pres_penalty=args.presence_penalty,
                                       stop=args.stop or None)
        return {**job, "score": parse_score(raw, args.scale), "raw": raw,
                "scale": args.scale, "error": err, "latency_ms": round(dt, 1)}

    n_done, n_total, t0 = 0, len(jobs), time.time()
    mode = "a" if (args.resume and out.exists()) else "w"
    with out.open(mode) as fout:
        tasks = [asyncio.create_task(run(j)) for j in jobs]
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 20 == 0 or n_done == n_total:
                rate = n_done / max(time.time() - t0, 1e-6)
                eta = (n_total - n_done) / max(rate, 1e-6)
                print(f"  done {n_done}/{n_total}  {rate:.2f} req/s  "
                      f"eta {eta:.0f}s", file=sys.stderr, flush=True)
    print(f"finished {n_done}/{n_total} in {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8004/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", choices=["1-5", "binary"], default="1-5",
                    help="score scale; 'binary' parses a 0/1 safe verdict")
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="OpenAI freq penalty; >0 discourages repetitive loops")
    ap.add_argument("--presence-penalty", type=float, default=0.0,
                    help="OpenAI presence penalty")
    ap.add_argument("--stop", action="append", default=[],
                    help="stop sequence (repeatable); e.g. <|end_of_box|>")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
