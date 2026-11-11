#!/usr/bin/env bash
# Train a list of configs x seeds on a multi-GPU server, one run per GPU at a time. A run whose
# train_metrics.json exists (written at the end) is skipped, so re-running resumes after a kill.
# Start it inside tmux, within your booked server time.
#   GO2WM_ROOT=/home/storage/$USER/go2wm bash scripts/server/train_queue.sh \
#       --data data/go2_e4c_train --prefix e5 --seeds "0 1 2" --gpus "0 1" \
#       dinov2s_gru dinov2s_grid_gru            # -> configs/world_model_<prefix>_<name>.yaml
set -uo pipefail
DATA=data/go2_e4c_train; PREFIX=e5; SEEDS="0 1 2"; GPUS="0 1"; MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    *) MODELS+=("$1"); shift ;;
  esac
done
: "${GO2WM_ROOT:?set GO2WM_ROOT (data/outputs root)}"
cd "$(dirname "$0")/../.."
LOG="$GO2WM_ROOT/logs/${PREFIX}_train"; mkdir -p "$LOG"
jobs_list=()
for s in $SEEDS; do for m in "${MODELS[@]}"; do jobs_list+=("$m:$s"); done; done
read -r -a gpu_arr <<< "$GPUS"
declare -A pid_gpu=()
run_one() {  # $1 = model:seed, $2 = gpu
  local m s out  # separate assignments: `local a=x b=$a` expands $a before assigning a
  m="${1%%:*}"; s="${1##*:}"; out="outputs/${PREFIX}_${m}_s${s}"
  if [[ -f "$GO2WM_ROOT/$out/train_metrics.json" ]]; then echo "[queue] skip $out"; return 0; fi
  echo "[queue] $(date -Is) gpu$2 start $out"
  CUDA_VISIBLE_DEVICES="$2" python scripts/train.py --config "configs/world_model_${PREFIX}_${m}.yaml" \
      --data "$DATA" --out "$out" --seed "$s" > "$LOG/${PREFIX}_${m}_s${s}.log" 2>&1
  echo "[queue] $(date -Is) gpu$2 end $out exit=$?"
}
i=0
for job in "${jobs_list[@]}"; do
  # wait for a free GPU
  while true; do
    for g in "${gpu_arr[@]}"; do
      busy=0
      for p in "${!pid_gpu[@]}"; do
        if kill -0 "$p" 2>/dev/null; then [[ "${pid_gpu[$p]}" == "$g" ]] && busy=1; else unset "pid_gpu[$p]"; fi
      done
      if [[ $busy == 0 ]]; then run_one "$job" "$g" & pid_gpu[$!]="$g"; continue 3; fi
    done
    sleep 5
  done
done
wait
echo "[queue] all done"
