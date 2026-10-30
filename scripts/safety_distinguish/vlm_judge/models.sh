# Model registry for the safety-judge runners. Source this file; do not run.
#
# JUDGE_SPEC: key -> "hf_model_id|default_port|jsonl_suffix|max_tokens|workers"
#   - jsonl_suffix matches the historical output naming, so --resume picks up
#     the existing judgments in outputs/qual_fig_judgments/ (blank = Qwen32B,
#     the final paper-figure model).
#   - ports are conventions from the recorded runs, not reservations; override
#     with --port when they collide.
#
# SERVE_SPEC: key -> "tensor_parallel|max_model_len|gpu_mem_util"
#   Model-specific serving quirks (GLM-FP8 conda env, DeepSeek hf-overrides,
#   Kimi-Thinking TP=1-only) are handled in serve.sh.

declare -A JUDGE_SPEC=(
  [qwen3-vl-32b]="Qwen/Qwen3-VL-32B-Instruct|8011||256|8"
  [qwen3-vl-30b-a3b]="Qwen/Qwen3-VL-30B-A3B-Instruct|8013|_qwen3-vl-30b-a3b|512|8"
  [glm-4.6v-fp8]="zai-org/GLM-4.6V-FP8|8006|_glm-4.6v-fp8|512|8"
  [glm-flash]="zai-org/GLM-4.6V-Flash|8016|_glm-4.6v-flash|512|4"
  [kimi-instruct]="moonshotai/Kimi-VL-A3B-Instruct|8006|_kimi-vl-instruct|768|8"
  [kimi-thinking]="moonshotai/Kimi-VL-A3B-Thinking-2506|8012|_kimi-vl-thinking|1536|4"
  [molmo]="allenai/Molmo-7B-D-0924|8007|_molmo-7b-d|256|8"
  [deepseek-vl2]="deepseek-ai/deepseek-vl2-small|8008|_deepseek-vl2-small|256|8"
  [cosmos-8b]="nvidia/Cosmos-Reason2-8B|8014|_cosmos-reason2-8b|512|8"
  [qwen3-vl-4b]="Qwen/Qwen3-VL-4B-Instruct|8014|_qwen3-vl-4b|512|8"
)

declare -A SERVE_SPEC=(
  [qwen3-vl-32b]="2|4096|0.90"
  [qwen3-vl-30b-a3b]="2|4096|0.90"
  [glm-4.6v-fp8]="4|8192|0.92"
  [glm-flash]="2|8192|0.88"
  [kimi-instruct]="1|4096|0.90"
  [kimi-thinking]="1|8192|0.90"
  [molmo]="1|4096|0.90"
  [deepseek-vl2]="2|4096|0.90"
  [cosmos-8b]="1|4096|0.90"
  [qwen3-vl-4b]="1|4096|0.90"
)

model_keys() { printf '%s\n' "${!JUDGE_SPEC[@]}" | sort; }

resolve_judge() {  # $1=key -> sets MODEL_ID PORT SUFFIX MAX_TOK WORKERS
  local spec=${JUDGE_SPEC[$1]:-}
  if [[ -z "$spec" ]]; then
    echo "unknown model key '$1' — available:" >&2
    model_keys >&2
    return 2
  fi
  IFS='|' read -r MODEL_ID PORT SUFFIX MAX_TOK WORKERS <<< "$spec"
}

resolve_serve() {  # $1=key -> sets TP MAX_LEN GPU_UTIL
  local spec=${SERVE_SPEC[$1]:-}
  if [[ -z "$spec" ]]; then
    echo "no serve recipe for '$1' — available:" >&2
    model_keys >&2
    return 2
  fi
  IFS='|' read -r TP MAX_LEN GPU_UTIL <<< "$spec"
}
