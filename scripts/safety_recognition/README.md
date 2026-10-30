# safety_recognition

VLM-based safety-recognition pipeline. Given a pre-rendered trajectory image
(storyboard overlay or similar single-panel figure), a vision-language model
judges whether the depicted robot trajectory is safe with respect to a nearby
obstacle. Results are streamed to JSONL for downstream aggregation.

## Layout

```
safety_recognition/
├── README.md
├── run_safety_judge.py          # generic judge (manifest-driven)
├── run_safety_judge_fpanel.py   # panel-tree judge (cm0..cm80 sweep)
├── run_safety_judge_samecm.py   # samecm subset judge (cm filter)
└── prompts/
    └── safety_prompt.txt        # system prompt for the safety judge
```

## Inference scripts

Each script wraps a vLLM-served VLM (any OpenAI-compatible endpoint) with
async concurrency, parses the score from the response, and streams
records to JSONL. All three accept `--resume` to skip records already
written.

| Script | Input layout | Notes |
|---|---|---|
| `run_safety_judge.py` | `{img_root}/{safe|unsafe}/{tier}/{rank:04d}__*.png` driven by a JSON `_manifest.json` | original entry point |
| `run_safety_judge_fpanel.py` | `{img_root}/cm{NN}/{n}_{Tier}/L{layout}_{Obstacle}_Route{R}.png` (panel tree) | sweeps cm0..cm80 |
| `run_safety_judge_samecm.py` | `cm{NN}/{tier}__{model}__{task_name}__{obstacle}.png` driven by `_manifest.csv` | restrict to specific `--cm` buckets |

All three support:
- `--scale {1-5,binary}` — score grammar
- `--max-tokens`, `--frequency-penalty`, `--presence-penalty`, `--stop` —
  generation knobs. Models that don't follow `Final Answer: N` (e.g. small
  multimodal models) can be coaxed with `--frequency-penalty 1.2
  --stop '<|end_of_box|>'`.

The parser tolerates `Final Answer: N`, a trailing bare digit, or
`<|begin_of_box|>N<|end_of_box|>`.

## Example

```bash
# 1. Serve a VLM (host or container):
vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8006 --max-model-len 4096

# 2. Run safety judgment on a panel-tree dataset:
docker exec robotics-safety-container python3 \
  scripts/safety_recognition/run_safety_judge_fpanel.py \
  --img-root /workspace/storyboard_overlay_cropped_final \
  --prompt   scripts/safety_recognition/prompts/safety_prompt.txt \
  --model    Qwen/Qwen3-VL-32B-Instruct \
  --base-url http://localhost:8006/v1 \
  --scale    binary \
  --workers  8 \
  --output   outputs/safety_judgments/qwen3_vl_32b.jsonl
```

Each output record schema:

```json
{
  "cm": 25, "tier": "High", "layout": 0,
  "obstacle": "Cat", "route": "B",
  "img": ".../cm25/1_High/L0_Cat_RouteB.png",
  "score": 1, "raw": "...Final Answer: 1",
  "scale": "binary", "error": null,
  "latency_ms": 4231.0
}
```
