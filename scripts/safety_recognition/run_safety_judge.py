#!/usr/bin/env python3
"""Run a vision-language safety-score judgment over wp_follow_f episodes.

For each episode in `_manifest.json`, we locate the corresponding trajectory PNG
under `{img_root}/{safe|unsafe}/{tier}/{rank:04d}__*.png`, send (image + prompt)
to a vLLM OpenAI-compatible endpoint, parse "Final Answer: N" from the response,
and stream results to a JSONL.

Resumable: rerun with --resume to skip ranks already written.

Example (run inside the container, vLLM server on localhost:8004):
    docker exec robotics-safety-container python3 \
      scripts/safety_recognition/run_safety_judge.py \
        --manifest <path_to>/_manifest.json \
        --prompt   scripts/safety_recognition/prompts/safety_prompt.txt \
        --img-root <path_to_image_root> \
        --model    Qwen/Qwen3-VL-32B-Instruct \
        --base-url http://localhost:8004/v1 \
        --workers  8 \
        --output   outputs/safety_judgments/qwen3_vl_32b/judgments.jsonl
"""
from __future__ import annotations

# Defensive: if this script is ever invoked from a directory containing a
# stdlib-shadowing module (e.g. `select.py`), drop our directory from
# sys.path[0] before importing asyncio (which pulls in `select`).
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


def find_image(img_root: Path, rank: int, safe: str, tier: str) -> Path | None:
    folder = img_root / safe / tier
    if not folder.is_dir():
        return None
    matches = list(folder.glob(f"{rank:04d}__*.png"))
    return matches[0] if matches else None


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def parse_score(text: str | None) -> int | None:
    if not text:
        return None
    m = FINAL_ANSWER_RE.search(text)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 5 else None


async def judge_one(client, model, prompt, img_path, max_tokens, semaphore,
                    enable_thinking: bool):
    async with semaphore:
        b64 = encode_image(img_path)
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }]
        # Qwen3 family emits <think>...</think> by default even on -Instruct.
        # That's fine if max_tokens is large enough, but for tight budgets we
        # disable it via the tokenizer chat-template kwargs.
        extra: dict = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body=extra,
            )
            raw = resp.choices[0].message.content
            err = None
        except Exception as e:
            raw, err = "", repr(e)
        return raw, err, (time.time() - t0) * 1000


async def main_async(args):
    manifest = json.loads(Path(args.manifest).read_text())
    prompt = Path(args.prompt).read_text()
    img_root = Path(args.img_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY",
                        timeout=args.timeout)
    sem = asyncio.Semaphore(args.workers)

    jobs, missing = [], 0
    for ep in manifest:
        img = find_image(img_root, ep["rank"], ep["safe"], ep["tier"])
        if img is None:
            missing += 1
            continue
        jobs.append((ep, img))
    print(f"manifest={len(manifest)} jobs={len(jobs)} missing_img={missing}",
          file=sys.stderr, flush=True)

    done_ranks: set[int] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done_ranks.add(json.loads(line)["rank"])
            except Exception:
                pass
        jobs = [j for j in jobs if j[0]["rank"] not in done_ranks]
        print(f"resume: skipping {len(done_ranks)} done, {len(jobs)} remaining",
              file=sys.stderr, flush=True)

    async def run(ep, img):
        raw, err, dt = await judge_one(client, args.model, prompt, img,
                                       args.max_tokens, sem,
                                       args.enable_thinking)
        return {
            "rank": ep["rank"],
            "policy_model": ep["model"],   # the model that produced the trajectory
            "task_name": ep["task_name"],
            "obstacle": ep["obstacle"],
            "layout": ep["layout"],
            "tier": ep["tier"],
            "safe_gt": ep["safe"],
            "dev_max": ep["dev_max"],
            "dev_mean": ep["dev_mean"],
            "img": str(img),
            "score": parse_score(raw),
            "raw": raw,
            "error": err,
            "latency_ms": round(dt, 1),
        }

    n_done, n_total = 0, len(jobs)
    t0 = time.time()
    mode = "a" if (args.resume and out_path.exists()) else "w"
    with out_path.open(mode) as fout:
        tasks = [asyncio.create_task(run(ep, img)) for ep, img in jobs]
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 10 == 0 or n_done == n_total:
                rate = n_done / max(time.time() - t0, 1e-6)
                print(f"  done {n_done}/{n_total} ({rate:.2f} req/s)",
                      file=sys.stderr, flush=True)
    print(f"finished {n_done}/{n_total} in {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8004/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="Bump to 1024+ if --enable-thinking is on.")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Allow Qwen3 <think>...</think> traces. Default off "
                         "for cheaper, terser judgments.")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
