#!/bin/bash
# Parameterized safety-judge runner: one script, model as an argument.
# Replaces the per-model run_sbov_* scripts (kept in legacy/).
#
#   ./run_judge.sh qwen3-vl-32b                     # binary + 1-5 on cropped_final
#   ./run_judge.sh kimi-thinking --scale 1-5 --wait # single scale, poll to completion
#   ./run_judge.sh glm-flash --img-root /workspace/storyboard_overlay_nkf10
#
# Expects a vLLM endpoint already serving the model (see serve.sh). Output
# naming follows the historical convention, so --resume continues the
# existing JSONLs under outputs/qual_fig_judgments/.
set -u
cd "$(dirname "$0")"
source ./models.sh

CT=${CT:-robotics-safety-container}
IMG_ROOT=/workspace/storyboard_overlay_cropped_final
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
PROMPT_BIN=/workspace/scripts/safety_recognition/prompts/safety_prompt_binary.txt
PROMPT_15=/workspace/scripts/safety_recognition/prompts/safety_prompt.txt
TAG=v5
SCALE=both
WAIT=0
STALL_MAX=${STALL_MAX:-1200}   # --wait: abort after this many stalled seconds

usage() {
  echo "usage: $0 MODEL_KEY [--scale both|binary|1-5] [--img-root DIR] [--out-dir DIR]"
  echo "          [--tag TAG] [--port N] [--max-tokens N] [--workers N] [--wait]"
  echo "model keys:"; model_keys | sed 's/^/  /'
  exit 2
}

[[ $# -lt 1 ]] && usage
KEY=$1; shift
resolve_judge "$KEY" || exit 2

while [[ $# -gt 0 ]]; do
  case $1 in
    --scale)      SCALE=$2; shift 2 ;;
    --img-root)   IMG_ROOT=$2; shift 2 ;;
    --out-dir)    OUT_DIR=$2; shift 2 ;;
    --tag)        TAG=$2; shift 2 ;;
    --port)       PORT=$2; shift 2 ;;
    --max-tokens) MAX_TOK=$2; shift 2 ;;
    --workers)    WORKERS=$2; shift 2 ;;
    --wait)       WAIT=1; shift ;;
    *) echo "unknown flag $1"; usage ;;
  esac
done

case $SCALE in both|binary|1-5) ;; *) echo "bad --scale '$SCALE'"; usage ;; esac

echo "[$(date +%T)] $KEY = $MODEL_ID  (:$PORT, suffix '${SUFFIX}', tok $MAX_TOK, workers $WORKERS)"
echo "[$(date +%T)] confirming :$PORT health..."
docker exec "$CT" bash -lc "curl -fsS -m 3 http://localhost:$PORT/health > /dev/null" \
  || { echo "  :$PORT NOT HEALTHY — serve the model first (see serve.sh)"; exit 1; }

docker exec "$CT" bash -lc "mkdir -p $OUT_DIR"

launch() {  # $1=scale-label(binary|1-5)
  local scale=$1 prompt out log infix=""
  if [[ "$scale" == binary ]]; then prompt=$PROMPT_BIN; infix="_binary"; else prompt=$PROMPT_15; fi
  out=$OUT_DIR/judgments${SUFFIX}${infix}_${TAG}.jsonl
  log=/workspace/outputs/judge_${KEY}${infix}_${TAG}.log
  docker exec -d "$CT" bash -lc \
    "python3 $SCRIPT --img-root $IMG_ROOT --prompt $prompt --model '$MODEL_ID' \
      --base-url http://localhost:$PORT/v1 --output $out \
      --workers $WORKERS --scale $scale --max-tokens $MAX_TOK --resume > $log 2>&1"
  echo "  [launched] $KEY $scale -> $out  (log $log)"
  LAUNCHED_OUT+=("$out")
}

LAUNCHED_OUT=()
[[ "$SCALE" == both || "$SCALE" == binary ]] && launch binary
[[ "$SCALE" == both || "$SCALE" == 1-5   ]] && launch 1-5

sleep 5
echo "[$(date +%T)] process snapshot:"
docker exec "$CT" bash -lc "ps -ef | grep run_safety_judge_fpanel | grep -v grep" || true

if [[ "$WAIT" -eq 1 ]]; then
  target=$(docker exec "$CT" bash -lc "find $IMG_ROOT -name '*.png' | wc -l")
  echo "[$(date +%T)] waiting for ${#LAUNCHED_OUT[@]} judge(s), target=$target records each"
  for out in "${LAUNCHED_OUT[@]}"; do
    last=-1; stable=0
    while :; do
      n=$(docker exec "$CT" bash -lc "wc -l < $out 2>/dev/null || echo 0")
      if [[ "$n" -ge "$target" ]]; then echo "[$(date +%T)] $out DONE (n=$n)"; break; fi
      if [[ "$n" -eq "$last" ]]; then
        stable=$((stable + 60))
        [[ "$stable" -ge "$STALL_MAX" ]] && { echo "[$(date +%T)] $out STALLED at n=$n"; break; }
      else
        stable=0; last=$n
        echo "[$(date +%T)] $out progress: $n / $target"
      fi
      sleep 60
    done
  done
fi
