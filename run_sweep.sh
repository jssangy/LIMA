#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
PY=${PY:-python}
SCRIPT=${SCRIPT:-train.py}
PROBLEM=${PROBLEM:-problems/cross/cross_1.json}

TOTAL_FRAMES=${TOTAL_FRAMES:-1000000}
SAVE_EVERY=${SAVE_EVERY:-0}

# 고정(원하면 환경변수로 바꿀 수 있음)
FRAMES_PER_BATCH=${FRAMES_PER_BATCH:-1024}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
NUM_EPOCHS=${NUM_EPOCHS:-10}

# 사용할 GPU 목록 (공백 구분). 예: GPUS="0 1 2 3"
GPUS_STR=${GPUS:-"0"}
read -r -a GPUS <<< "$GPUS_STR"
NGPU=${#GPUS[@]}
JOBS=${JOBS:-$NGPU}

# 스윕 규모
SIZE=${SIZE:-M}
SEEDS=(${SEEDS:-7})

# -------- sweep grids (only 5 hparams) --------
if [[ "$SIZE" == "S" ]]; then
  LRS=("1e-4" "3e-4" "5e-4")
  GAMMAS=("0.97" "0.99")
  LAMBDAS=("0.90" "0.95")
  CLIPS=("0.1" "0.2")
  ENTS=("0.0" "0.005" "0.01")
elif [[ "$SIZE" == "L" ]]; then
  LRS=("5e-5" "1e-4" "2e-4" "3e-4" "5e-4" "7e-4" "1e-3")
  GAMMAS=("0.95" "0.97" "0.99")
  LAMBDAS=("0.90" "0.92" "0.95" "0.97")
  CLIPS=("0.05" "0.1" "0.15" "0.2" "0.25" "0.3")
  ENTS=("0.0" "0.001" "0.0025" "0.005" "0.0075" "0.01" "0.015")
else # M
  LRS=("5e-5" "1e-4" "5e-4" "1e-3" "5e-3")
  GAMMAS=("0.95" "0.99")
  LAMBDAS=("0.90" "0.95")
  CLIPS=("0.1" "0.2" "0.3")
  ENTS=("0.0" "0.005" "0.01")
fi

# -------- dirs & group --------
ROOT="$(pwd)"
LOGDIR="${ROOT}/logs"
ARTIFACTS="${ROOT}/artifacts"
RUNDIR="${ROOT}/runs"
mkdir -p "$LOGDIR" "$ARTIFACTS" "$RUNDIR"

GROUP="sweep_${SIZE}_$(date +%Y%m%d_%H%M%S)"
export WANDB_RUN_GROUP="${GROUP}"

# -------- helper --------
run_one () {
  local gpu="$1" lr="$2" gamma="$3" lam="$4" clip="$5" ent="$6"

  local EXP_ID="g${gpu}_lr${lr}_gm${gamma}_lam${lam}_clip${clip}_ent${ent}"
  local RUN_PATH="${RUNDIR}/${EXP_ID}"
  local LOGPATH="${LOGDIR}/${EXP_ID}.log"

  mkdir -p "${RUN_PATH}"
  (
    cd "${RUN_PATH}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo ">> [${EXP_ID}] GPU=${gpu} start in $(pwd)"

    ${PY} "${ROOT}/${SCRIPT}" -p "${ROOT}/${PROBLEM}" \
      --total_frames "${TOTAL_FRAMES}" \
      --frames_per_batch "${FRAMES_PER_BATCH}" \
      --mini_batch_size "${MINI_BATCH_SIZE}" \
      --num_epochs "${NUM_EPOCHS}" \
      --lr "${lr}" \
      --gamma "${gamma}" \
      --lmbda "${lam}" \
      --clip_epsilon "${clip}" \
      --entropy_coeff "${ent}" \
      --save_dir "${ARTIFACTS}" \
      --exp_name "${EXP_ID}" \
      --save_every "${SAVE_EVERY}" \
      --seed "${seed}" \
      2>&1 | tee "${LOGPATH}"

    echo ">> [${EXP_ID}] done. artifacts: ${ARTIFACTS}/${EXP_ID}"
  )
}

# -------- count combos --------
COUNT=0
for s in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for gm in "${GAMMAS[@]}"; do
      for lam in "${LAMBDAS[@]}"; do
        for clip in "${CLIPS[@]}"; do
          for ent in "${ENTS[@]}"; do
            COUNT=$((COUNT+1))
          done
        done
      done
    done
  done
done
echo "GPUS: ${GPUS[*]}  (NGPU=${NGPU})"
echo "Total runs: ${COUNT}  (GROUP=${GROUP}, JOBS=${JOBS})"
echo "Fixed: FRAMES_PER_BATCH=${FRAMES_PER_BATCH}, MINI_BATCH_SIZE=${MINI_BATCH_SIZE}, NUM_EPOCHS=${NUM_EPOCHS}, TOTAL_FRAMES=${TOTAL_FRAMES}"

# -------- dispatch (round-robin GPU assignment) --------
idx=0
running=0
for s in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for gm in "${GAMMAS[@]}"; do
      for lam in "${LAMBDAS[@]}"; do
        for clip in "${CLIPS[@]}"; do
          for ent in "${ENTS[@]}"; do
            gpu="${GPUS[$((idx % NGPU))]}"
            idx=$((idx+1))

            run_one "$gpu" "$lr" "$gm" "$lam" "$clip" "$ent" &
            running=$((running+1))
            if (( running % JOBS == 0 )); then
              wait
            fi
          done
        done
      done
    done
  done
done
wait

echo "All runs finished."
