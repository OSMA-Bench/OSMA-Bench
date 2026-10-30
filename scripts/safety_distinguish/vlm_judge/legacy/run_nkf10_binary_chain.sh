#!/bin/bash
# Run binary safety-judge on storyboard_overlay_nkf10 (cm60-80 dataset) for all
# 6 models in the v5 2x3 figure. Serializes on GPUs 6,7 (TP=2 per model).
#
# IMG_ROOT: /workspace/storyboard_overlay_nkf10 — 387 PNGs across cm60/65/70/75/80.
# OUT_DIR : /workspace/outputs/qual_fig_judgments/storyboard_overlay_nkf10/
# Each model emits: judgments_<suffix>_binary_v5.jsonl
#
# Per-model loop: spin up vLLM on GPUs 6,7 -> wait health -> launch judge ->
# wait until JSONL has ≥ TARGET lines (or 15-min stall) -> teardown -> next.
#
# Priority order = Qwen3-VL-32B + Qwen3-VL-30B-A3B first, so the single-panel
# figures can be re-baked early.
set -u

CT=robotics-safety-container
HOST_WS=~/workspace/robotics-safety
IMG_ROOT=/workspace/storyboard_overlay_nkf10
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay_nkf10
HOST_OUT_DIR=$HOST_WS/outputs/qual_fig_judgments/storyboard_overlay_nkf10
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
PROMPT_BIN=/workspace/scripts/safety_recognition/prompts/safety_prompt_binary.txt
TARGET=387          # total PNGs in nkf10
STALL_MAX=900       # 15 min of zero growth -> abort

mkdir -p "$HOST_OUT_DIR"

# (suffix, port, max_tokens, model_id, vllm_serve_id) — vllm_serve_id is the
# pre-baked launcher key resolved inside launch_vllm() below. We keep flags
# next to model_id so model-specific quoting (deepseek's hf-overrides) lives
# in one place rather than in shell interpolation.
MODELS=(
  "qwen32b:_binary_v5:8011:256:Qwen/Qwen3-VL-32B-Instruct"
  "qwen30b:_qwen3-vl-30b-a3b_binary_v5:8013:256:Qwen/Qwen3-VL-30B-A3B-Instruct"
  "kimi-ins:_kimi-vl-instruct_binary_v5:8006:768:moonshotai/Kimi-VL-A3B-Instruct"
  "kimi-thk:_kimi-vl-thinking_binary_v5:8012:1024:moonshotai/Kimi-VL-A3B-Thinking-2506"
  "molmo:_molmo-7b-d_binary_v5:8007:256:allenai/Molmo-7B-D-0924"
  "deepseek:_deepseek-vl2-small_binary_v5:8008:256:deepseek-ai/deepseek-vl2-small"
  # 7th model added per user request — port 8016 (not 8014 = Cosmos-32B in
  # separate container; 8015 = Cosmos-Reason2-8B on GPU 4; 8011-8013 reused
  # for prior models in this same chain so collisions impossible serial).
  "glm-flash:_glm-4.6v-flash_binary_v5:8016:512:zai-org/GLM-4.6V-Flash"
)

wait_jsonl_done() {
  local jpath="$1" label="$2"
  local last=-1 stable=0
  while :; do
    local n=$(wc -l < "$jpath" 2>/dev/null || echo 0)
    if [[ "$n" -ge "$TARGET" ]]; then
      echo "[$(date +%T)] $label DONE (n=$n)"; return 0
    fi
    if [[ "$n" -eq "$last" ]]; then
      stable=$((stable + 60))
      if [[ "$stable" -ge "$STALL_MAX" ]]; then
        echo "[$(date +%T)] $label STALLED at n=$n (no growth ${STALL_MAX}s)"
        return 1
      fi
    else
      stable=0; last=$n
      echo "[$(date +%T)] $label progress: n=$n / $TARGET"
    fi
    sleep 60
  done
}

teardown_vllm() {
  # NB: VLLM::Worker_TP* / VLLM::EngineCore are prctl(PR_SET_NAME) comm names,
  # NOT command-line strings. pkill -f matches /proc/<pid>/cmdline (the
  # python invocation), which doesn't contain "VLLM::Worker" — so -f leaves
  # the workers as orphan PIDs holding GPU memory. Drop -f for VLLM:: comms;
  # keep -f only for the API server whose comm is generic "python3".
  docker exec $CT bash -lc "pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null; pkill -9 'VLLM::Worker' 2>/dev/null; pkill -9 'VLLM::EngineCore' 2>/dev/null" || true
  # Poll until both GPU 6 and GPU 7 fall below 1 GiB; if not within 60s, do
  # one more aggressive sweep. (A naive 20s sleep race-loses to slow CUDA
  # context release and gives the next launch a 4 GiB-free GPU.)
  for w in $(seq 1 30); do
    local g67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 2>/dev/null | awk '{s+=$1} END {print s}')
    if [[ -n "$g67" ]] && [[ "$g67" -lt 2048 ]]; then
      echo "[$(date +%T)] GPUs 6,7 cleared after ${w}x2s (used=${g67} MiB)"
      return 0
    fi
    if [[ "$w" -eq 15 ]]; then
      echo "[$(date +%T)] GPUs 6,7 still at ${g67} MiB — second sweep"
      docker exec $CT bash -lc "pkill -9 'VLLM::Worker' 2>/dev/null; pkill -9 'VLLM::EngineCore' 2>/dev/null; pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null" || true
    fi
    sleep 2
  done
  echo "[$(date +%T)] WARN: GPUs 6,7 not clean after 60s — proceeding anyway"
}

# Per-model launcher. Keeps shell-quoting hazards localized.
launch_vllm() {
  local key="$1" model="$2" port="$3" log="$4"
  case "$key" in
    deepseek)
      # hf-overrides JSON survives via printf %q inside the container.
      docker exec -d $CT bash -lc "CUDA_VISIBLE_DEVICES=6,7 python3 -m vllm.entrypoints.openai.api_server --model '$model' --port $port --tensor-parallel-size 2 --gpu-memory-utilization 0.88 --max-model-len 8192 --trust-remote-code --hf-overrides '{\"architectures\":[\"DeepseekVLV2ForCausalLM\"]}' > $log 2>&1"
      ;;
    *)
      docker exec -d $CT bash -lc "CUDA_VISIBLE_DEVICES=6,7 python3 -m vllm.entrypoints.openai.api_server --model '$model' --port $port --tensor-parallel-size 2 --gpu-memory-utilization 0.88 --max-model-len 8192 --trust-remote-code > $log 2>&1"
      ;;
  esac
}

for spec in "${MODELS[@]}"; do
  key=$(echo    "$spec" | cut -d: -f1)
  suffix=$(echo "$spec" | cut -d: -f2)
  port=$(echo   "$spec" | cut -d: -f3)
  tok=$(echo    "$spec" | cut -d: -f4)
  model=$(echo  "$spec" | cut -d: -f5-)   # f5- in case model id has colons

  out_bin="$OUT_DIR/judgments${suffix}.jsonl"
  host_bin="$HOST_OUT_DIR/judgments${suffix}.jsonl"

  echo "=================================================================="
  echo "[$(date +%T)] BEGIN $key ($model)  -> $out_bin"
  echo "=================================================================="

  # already done?
  if [[ -f "$host_bin" ]] && [[ "$(wc -l < "$host_bin")" -ge "$TARGET" ]]; then
    echo "[$(date +%T)] already complete ($(wc -l < "$host_bin")/$TARGET) — skip"
    continue
  fi

  teardown_vllm
  log_vllm="/workspace/outputs/vllm_nkf10${suffix}.log"
  echo "[$(date +%T)] starting vLLM (GPUs 6,7, port $port) ..."
  launch_vllm "$key" "$model" "$port" "$log_vllm"

  # Wait for health (up to 12 min)
  ready=0
  for i in $(seq 1 48); do
    if docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$port/health > /dev/null 2>&1"; then
      echo "[$(date +%T)] :$port healthy after ${i}x15s"; ready=1; break
    fi
    sleep 15
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[$(date +%T)] FATAL: :$port never came up for $model"
    docker exec $CT bash -lc "tail -100 $log_vllm" 2>&1 | tail -100
    teardown_vllm
    continue
  fi

  log_judge="/workspace/outputs/sbov_nkf10${suffix}.log"
  echo "[$(date +%T)] launching judge..."
  docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_BIN --model '$model' \
  --base-url http://localhost:$port/v1 --output $out_bin \
  --workers 8 --scale binary --max-tokens $tok --resume \
  > $log_judge 2>&1"

  if wait_jsonl_done "$host_bin" "$(basename $out_bin)"; then
    echo "[$(date +%T)] judge OK"
  else
    echo "[$(date +%T)] judge stalled — moving on"
  fi

  # Re-bake whichever plots are unlocked by available data
  echo "[$(date +%T)] re-baking plots ..."
  docker exec $CT bash -lc "python3 /workspace/scripts/qual_fig/plot_v5_2x3_binary.py" 2>&1 | tail -8
  docker exec $CT bash -lc "python3 /workspace/scripts/qual_fig/plot_v5_single_panel.py" 2>&1 | tail -8

  teardown_vllm
done

echo "=================================================================="
echo "[$(date +%T)] ALL nkf10 BINARY JUDGES DONE"
echo "=================================================================="
