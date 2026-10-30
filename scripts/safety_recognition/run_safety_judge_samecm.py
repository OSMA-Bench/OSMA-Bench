#!/usr/bin/env python3
"""VLM safety-score judge for the wp_samecm_by_cm dataset.

Same inference core as run_safety_judge.py, but reads the flat
`cm{N}/{tier}__{model}__{task_name}__{obstacle}.png` layout and its
`_manifest.csv`. Restrict to specific cost-map buckets with --cm.

Example (run inside the container, vLLM server on localhost:8004):
    docker exec robotics-safety-container python3 \
      scripts/safety_recognition/run_safety_judge_samecm.py \
      --manifest <path_to>/_manifest.csv \
      --img-root <path_to_samecm_root> \
      --prompt   scripts/safety_recognition/prompts/safety_prompt.txt \
      --cm 25 30 --model Qwen/Qwen3-VL-32B-Instruct \
      --base-url http://localhost:8004/v1 \
      --output   outputs/safety_judgments/samecm_cm2530/judgments.jsonl
"""
from __future__ import annotations

# Defensive: drop our directory from sys.path[0] before importing asyncio.
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:] = [p for p in _sys.path if _os.path.abspath(p) != _HERE]

import argparse
import asyncio
import base64
import csv
import json
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(\d)", re.IGNORECASE)


def parse_score(text: str | None) -> int | None:
    if not text:
        return None
    m = FINAL_ANSWER_RE.search(text)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 5 else None


def resolve_img(img_root: Path, row: dict) -> Path | None:
    folder = img_root / f"cm{row['cm']}"
    fn = f"{row['tier']}__{row['model']}__{row['task_name']}__{row['obstacle']}.png"
    p = folder / fn
    if p.exists():
        return p
    cand = list(folder.glob(f"*{row['task_name']}*{row['obstacle']}.png"))
    return cand[0] if len(cand) == 1 else None


async def judge_one(client, model, prompt, img_path, max_tokens, sem):
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
            resp = await client.chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            raw, err = resp.choices[0].message.content, None
        except Exception as e:
            raw, err = "", repr(e)
        return raw, err, (time.time() - t0) * 1000


async def main_async(args):
    rows = list(csv.DictReader(open(args.manifest)))
    cms = {str(c).removeprefix("cm") for c in args.cm}
    rows = [r for r in rows if r["cm"] in cms]
    prompt = Path(args.prompt).read_text()
    img_root = Path(args.img_root)

    jobs, missing = [], []
    for r in rows:
        img = resolve_img(img_root, r)
        if img is None:
            missing.append(r)
        else:
            jobs.append((r, img))
    print(f"manifest_rows={len(rows)} jobs={len(jobs)} missing={len(missing)}",
          file=sys.stderr, flush=True)
    for r in missing:
        print(f"  MISSING cm{r['cm']} {r['tier']} {r['task_name']}",
              file=sys.stderr, flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY",
                         timeout=args.timeout)
    sem = asyncio.Semaphore(args.workers)

    async def run(r, img):
        raw, err, dt = await judge_one(client, args.model, prompt, img,
                                       args.max_tokens, sem)
        return {
            "cm": int(r["cm"]),
            "rank": int(r["rank"]) if r.get("rank") else None,
            "policy_model": r["model"],
            "task_name": r["task_name"],
            "obstacle": r["obstacle"],
            "route": r.get("route"),
            "tier": r["tier"],
            "safe_gt": r["safe"],
            "samecm_partners": r.get("samecm_partners"),
            "img": str(img),
            "score": parse_score(raw),
            "raw": raw, "error": err, "latency_ms": round(dt, 1),
        }

    n_total, n_done, t0 = len(jobs), 0, time.time()
    with out.open("w") as fout:
        tasks = [asyncio.create_task(run(r, img)) for r, img in jobs]
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 10 == 0 or n_done == n_total:
                print(f"  done {n_done}/{n_total}", file=sys.stderr, flush=True)
    print(f"finished {n_done}/{n_total} in {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--cm", nargs="+", required=True, help="cost-map buckets, e.g. 25 30")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8004/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
