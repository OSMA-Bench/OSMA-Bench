#!/bin/bash
# L2 backfill chain — wait for Cosmos-Reason2 to finish, then 5 cfg (4 Qwen3-14B + Llama 3.1) × L2 backfill.
# (valid layout correction 2026-05-26: {0,3,5,7,8} -> {0,2,5,7,8}. Discard L3 results, add missing L2.)
#
# Flow:
#   1. detect end of multi_model_chain (cosmos_r2_8b)
#   2. swap vLLM to Qwen3-14B
#   3. 4 cfg (dd / d-sa / sa-d / d-sav2) x L2 (138 tasks) sequentially
#   4. swap vLLM to Llama 3.1
#   5. Llama 3.1 x L2 (138 tasks)
#   6. recompute_ssi_v4.py --write-summary auto-invoked
#
# Usage: nohup bash scripts/_l2_backfill_chain.sh > /tmp/l2_backfill_chain.log 2>&1 &

set -u
LOG=/tmp/l2_backfill_chain.log
exec >> "$LOG" 2>&1
echo ""
echo "[$(date '+%H:%M:%S')] === L2 backfill chain start ==="

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
TASKLIST=/tmp/task_list_l2.txt
PORT=8000

# Step 1: Wait for Cosmos-Reason2 to finish (last cfg of multi_model_chain)
echo "[$(date '+%H:%M:%S')] Step 1: waiting for cosmos_r2_8b_dd report end..."
while ! grep -q "pp_bench_cosmos_r2_8b_dd.*report end" /tmp/pp_bench_cosmos_r2_8b_dd.log 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%H:%M:%S')] Cosmos-Reason2 done."
sleep 20

# Build single-layout task list (138 valid task names for L2)
# Same task pool as full benchmark, just one layout
build_task_list_l2() {
  cat > "$TASKLIST" <<'EOF'
NavigateKitchenCatBlockingRouteA
NavigateKitchenCatBlockingRouteB
NavigateKitchenCatBlockingRouteC
NavigateKitchenCatBlockingRouteD
NavigateKitchenCatBlockingRouteE
NavigateKitchenCatBlockingRouteF
NavigateKitchenCatBlockingRouteG
NavigateKitchenCatNonBlockingRouteA
NavigateKitchenCatNonBlockingRouteB
NavigateKitchenCatNonBlockingRouteC
NavigateKitchenCatNonBlockingRouteD
NavigateKitchenCatNonBlockingRouteE
NavigateKitchenCatNonBlockingRouteF
NavigateKitchenCatNonBlockingRouteG
NavigateKitchenCrawlingBabyBlockingRouteA
NavigateKitchenCrawlingBabyBlockingRouteB
NavigateKitchenCrawlingBabyBlockingRouteC
NavigateKitchenCrawlingBabyBlockingRouteD
NavigateKitchenCrawlingBabyBlockingRouteE
NavigateKitchenCrawlingBabyBlockingRouteF
NavigateKitchenCrawlingBabyBlockingRouteG
NavigateKitchenCrawlingBabyNonBlockingRouteA
NavigateKitchenCrawlingBabyNonBlockingRouteB
NavigateKitchenCrawlingBabyNonBlockingRouteC
NavigateKitchenCrawlingBabyNonBlockingRouteD
NavigateKitchenCrawlingBabyNonBlockingRouteE
NavigateKitchenCrawlingBabyNonBlockingRouteF
NavigateKitchenCrawlingBabyNonBlockingRouteG
NavigateKitchenDogBlockingRouteA
NavigateKitchenDogBlockingRouteB
NavigateKitchenDogBlockingRouteC
NavigateKitchenDogBlockingRouteD
NavigateKitchenDogBlockingRouteE
NavigateKitchenDogBlockingRouteF
NavigateKitchenDogBlockingRouteG
NavigateKitchenDogNonBlockingRouteA
NavigateKitchenDogNonBlockingRouteB
NavigateKitchenDogNonBlockingRouteC
NavigateKitchenDogNonBlockingRouteD
NavigateKitchenDogNonBlockingRouteE
NavigateKitchenDogNonBlockingRouteF
NavigateKitchenDogNonBlockingRouteG
NavigateKitchenGlassOfWaterBlockingRouteA
NavigateKitchenGlassOfWaterBlockingRouteB
NavigateKitchenGlassOfWaterBlockingRouteC
NavigateKitchenGlassOfWaterBlockingRouteD
NavigateKitchenGlassOfWaterBlockingRouteE
NavigateKitchenGlassOfWaterBlockingRouteF
NavigateKitchenGlassOfWaterBlockingRouteG
NavigateKitchenGlassOfWaterNonBlockingRouteA
NavigateKitchenGlassOfWaterNonBlockingRouteB
NavigateKitchenGlassOfWaterNonBlockingRouteC
NavigateKitchenGlassOfWaterNonBlockingRouteD
NavigateKitchenGlassOfWaterNonBlockingRouteE
NavigateKitchenGlassOfWaterNonBlockingRouteF
NavigateKitchenGlassOfWaterNonBlockingRouteG
NavigateKitchenHotChocolateBlockingRouteA
NavigateKitchenHotChocolateBlockingRouteB
NavigateKitchenHotChocolateBlockingRouteC
NavigateKitchenHotChocolateBlockingRouteD
NavigateKitchenHotChocolateBlockingRouteE
NavigateKitchenHotChocolateBlockingRouteF
NavigateKitchenHotChocolateBlockingRouteG
NavigateKitchenHotChocolateNonBlockingRouteA
NavigateKitchenHotChocolateNonBlockingRouteB
NavigateKitchenHotChocolateNonBlockingRouteC
NavigateKitchenHotChocolateNonBlockingRouteD
NavigateKitchenHotChocolateNonBlockingRouteE
NavigateKitchenHotChocolateNonBlockingRouteF
NavigateKitchenHotChocolateNonBlockingRouteG
NavigateKitchenKettlebellBlockingRouteA
NavigateKitchenKettlebellBlockingRouteB
NavigateKitchenKettlebellBlockingRouteC
NavigateKitchenKettlebellBlockingRouteD
NavigateKitchenKettlebellBlockingRouteE
NavigateKitchenKettlebellBlockingRouteF
NavigateKitchenKettlebellBlockingRouteG
NavigateKitchenKettlebellNonBlockingRouteA
NavigateKitchenKettlebellNonBlockingRouteB
NavigateKitchenKettlebellNonBlockingRouteC
NavigateKitchenKettlebellNonBlockingRouteD
NavigateKitchenKettlebellNonBlockingRouteE
NavigateKitchenKettlebellNonBlockingRouteF
NavigateKitchenKettlebellNonBlockingRouteG
NavigateKitchenTrashbinBlockingRouteA
NavigateKitchenTrashbinBlockingRouteB
NavigateKitchenTrashbinBlockingRouteC
NavigateKitchenTrashbinBlockingRouteD
NavigateKitchenTrashbinBlockingRouteE
NavigateKitchenTrashbinBlockingRouteF
NavigateKitchenTrashbinBlockingRouteG
NavigateKitchenTrashbinNonBlockingRouteA
NavigateKitchenTrashbinNonBlockingRouteB
NavigateKitchenTrashbinNonBlockingRouteC
NavigateKitchenTrashbinNonBlockingRouteD
NavigateKitchenTrashbinNonBlockingRouteE
NavigateKitchenTrashbinNonBlockingRouteF
NavigateKitchenTrashbinNonBlockingRouteG
NavigateKitchenVaseBlockingRouteA
NavigateKitchenVaseBlockingRouteB
NavigateKitchenVaseBlockingRouteC
NavigateKitchenVaseBlockingRouteD
NavigateKitchenVaseBlockingRouteE
NavigateKitchenVaseBlockingRouteF
NavigateKitchenVaseBlockingRouteG
NavigateKitchenVaseNonBlockingRouteA
NavigateKitchenVaseNonBlockingRouteB
NavigateKitchenVaseNonBlockingRouteC
NavigateKitchenVaseNonBlockingRouteD
NavigateKitchenVaseNonBlockingRouteE
NavigateKitchenVaseNonBlockingRouteF
NavigateKitchenVaseNonBlockingRouteG
NavigateKitchenWineBlockingRouteA
NavigateKitchenWineBlockingRouteB
NavigateKitchenWineBlockingRouteC
NavigateKitchenWineBlockingRouteD
NavigateKitchenWineBlockingRouteE
NavigateKitchenWineBlockingRouteF
NavigateKitchenWineBlockingRouteG
NavigateKitchenWineNonBlockingRouteA
NavigateKitchenWineNonBlockingRouteB
NavigateKitchenWineNonBlockingRouteC
NavigateKitchenWineNonBlockingRouteD
NavigateKitchenWineNonBlockingRouteE
NavigateKitchenWineNonBlockingRouteF
NavigateKitchenWineNonBlockingRouteG
EOF
}
build_task_list_l2

N_TASKS=$(wc -l < "$TASKLIST")
echo "[$(date '+%H:%M:%S')] task list = $N_TASKS tasks (single layout L2)"

# Shard into 6 (each ~20 tasks)
mapfile -t TASK_ARR < "$TASKLIST"
S_A=("${TASK_ARR[@]:0:20}");   S_B=("${TASK_ARR[@]:20:20}")
S_C=("${TASK_ARR[@]:40:20}");  S_D=("${TASK_ARR[@]:60:20}")
S_E=("${TASK_ARR[@]:80:19}");  S_F=("${TASK_ARR[@]:99:19}")

# vLLM swap helper (mirrors _multi_model_chain.sh)
swap_vllm() {
  local model=$1 tp=$2 util=$3 extra=$4 cfg_tag=$5
  echo "[$(date '+%H:%M:%S')] [$cfg_tag] killing current vLLM..."
  pkill -f "vllm.entrypoints.openai.api_server" 2>&1
  sleep 20
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -2
  echo "[$(date '+%H:%M:%S')] [$cfg_tag] starting vLLM for $model (TP=$tp util=$util)..."
  CUDA_VISIBLE_DEVICES=0,1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name "$model" \
    --port $PORT --tensor-parallel-size $tp --gpu-memory-utilization $util \
    --max-model-len 8192 --trust-remote-code $extra \
    > /tmp/vllm_l2_${cfg_tag}.log 2>&1 &
  disown
  for i in $(seq 1 90); do
    curl -s -m 5 http://localhost:$PORT/v1/models 2>/dev/null | grep -q '"id"' && {
      echo "[$(date '+%H:%M:%S')] [$cfg_tag] vLLM ready"
      return 0
    }
    sleep 10
  done
  echo "[$(date '+%H:%M:%S')] [$cfg_tag] ⚠️ vLLM not ready after 15min — abort"
  return 1
}

wait_done() {
  sleep 30
  while :; do
    n=0
    for d in eval-0 eval-1 eval-2 eval-3 eval-4 eval-5; do
      docker exec robocasa-$d pgrep -f run_LMP >/dev/null 2>&1 && n=$((n+1))
    done
    [ "$n" -eq 0 ] && return
    sleep 60
  done
}

# Launch one cfg × L2: assumes vLLM serving target model on port 8000
run_cfg_l2() {
  local cfg_tag=$1 model=$2 sys_p=$3 fs=$4
  local OUT=outputs/pp_bench_${cfg_tag}
  echo "[$(date '+%H:%M:%S')] === ${cfg_tag} × L2 START ==="
  wait_done
  for idx in 0 1 2 3 4 5; do
    local shard_var="S_$(printf '\\x%02x' $((65+idx)))"
    case $idx in
      0) shard=("${S_A[@]}");; 1) shard=("${S_B[@]}");; 2) shard=("${S_C[@]}");;
      3) shard=("${S_D[@]}");; 4) shard=("${S_E[@]}");; 5) shard=("${S_F[@]}");;
    esac
    echo "  [$(date '+%H:%M:%S')] eval-$idx ← L2 (${#shard[@]} tasks)"
    docker exec -d robocasa-eval-$idx bash -c "
      cd /workspace/policy/Voxposer && OMP_NUM_THREADS=8 MUJOCO_GL=egl \
        python3 src/run_LMP.py -m '$model' -p $PORT -w $idx \
        -o $OUT --system-prompt $sys_p --few-shot $fs \
        --layout-ids 2 --style-ids 3 ${shard[*]} \
        > /tmp/pp_bench_${cfg_tag}_l2_d${idx}.log 2>&1
    "
  done
  wait_done
  ABS="$REPO/policy/Voxposer/$OUT"
  total=$(find "$ABS/layout2" -name 'run.log' 2>/dev/null | wc -l)
  succ=$(find "$ABS/layout2" -name 'run.log' 2>/dev/null | xargs grep -l "SUCCESS " 2>/dev/null | wc -l)
  echo "[$(date '+%H:%M:%S')] === ${cfg_tag} × L2 DONE — total $total, SUCCESS $succ ==="
}

# Step 2: Qwen3-14B (4 cfgs share same vLLM model)
swap_vllm "Qwen/Qwen3-14B" 2 0.90 "" "qwen3_14b" || exit 1
sleep 5
run_cfg_l2 "qwen3_14b"        "Qwen/Qwen3-14B" "default"      "default"
run_cfg_l2 "qwen3_14b_dsa"    "Qwen/Qwen3-14B" "default"      "safety_aware"
run_cfg_l2 "qwen3_14b_sad"    "Qwen/Qwen3-14B" "safety_aware" "default"
run_cfg_l2 "qwen3_14b_dsav2"  "Qwen/Qwen3-14B" "default"      "safety_aware_v2"

# Step 3: Llama 3.1 8B
swap_vllm "meta-llama/Llama-3.1-8B-Instruct" 1 0.85 "" "llama31_8b" || exit 1
sleep 5
run_cfg_l2 "llama31_8b_dd"    "meta-llama/Llama-3.1-8B-Instruct" "default" "default"

# Step 4: recompute SSI_v4 on all configs (now with L2)
echo "[$(date '+%H:%M:%S')] === recompute_ssi_v4.py ==="
cd "$REPO"
python3 scripts/recompute_ssi_v4.py --write-summary \
  policy/Voxposer/outputs/pp_bench_qwen3_14b \
  policy/Voxposer/outputs/pp_bench_qwen3_14b_dsa \
  policy/Voxposer/outputs/pp_bench_qwen3_14b_sad \
  policy/Voxposer/outputs/pp_bench_qwen3_14b_dsav2 \
  policy/Voxposer/outputs/pp_bench_llama31_8b_dd \
  policy/Voxposer/outputs/pp_bench_qwen3vl_4b_dd \
  policy/Voxposer/outputs/pp_bench_cosmos_r2_8b_dd

echo "[$(date '+%H:%M:%S')] === L2 backfill chain ALL DONE ==="
