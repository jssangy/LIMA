#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
PY=${PY:-python}
SCRIPT=${SCRIPT:-train.py}
PROBLEM=${PROBLEM:-problems/cross/cross_1.json}

TOTAL_FRAMES=${TOTAL_FRAMES:-500000}
GAMMA=${GAMMA:-0.99}
SAVE_EVERY=${SAVE_EVERY:-0}

# 사용할 GPU 목록 (공백 구분). 예: GPUS="0 1 2 3"
GPUS_STR=${GPUS:-"0"}
read -r -a GPUS <<< "$GPUS_STR"
NGPU=${#GPUS[@]}

# 동시 실행 수(기본: GPU 수와 동일)
JOBS=${JOBS:-$NGPU}

# 스윕 규모 S/M/L
SIZE=${SIZE:-L}
SEEDS=(${SEEDS:-0 1})

if [[ "$SIZE" == "S" ]]; then
  LRS=("1e-4" "3e-4" "5e-4")
  CLIPS=("0.1" "0.2" "0.3")
  ENTS=("0.0" "0.005" "0.01")
  LAMBDAS=("0.90" "0.95")
  EPOCHS=("8")
  MINIBATCH=("64")
  FPB=("256")
elif [[ "$SIZE" == "L" ]]; then
  LRS=("5e-5" "1e-4" "2e-4" "3e-4" "5e-4" "7e-4" "1e-3")
  CLIPS=("0.05" "0.1" "0.15" "0.2" "0.25" "0.3")
  ENTS=("0.0" "0.001" "0.0025" "0.005" "0.0075" "0.01" "0.015")
  LAMBDAS=("0.90" "0.92" "0.95" "0.97")
  EPOCHS=("5" "8" "10")
  MINIBATCH=("32" "64" "128")
  FPB=("256" "512")
else
  LRS=("1e-4" "2e-4" "3e-4" "5e-4")
  CLIPS=("0.1" "0.15" "0.2" "0.25")
  ENTS=("0.0" "0.0025" "0.005" "0.01")
  LAMBDAS=("0.90" "0.95" "0.97")
  EPOCHS=("8" "10")
  MINIBATCH=("64" "128")
  FPB=("256" "512")
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
  local gpu="$1" seed="$2" lr="$3" clip="$4" ent="$5" lam="$6" ep="$7" mb="$8" fpb="$9"

  local EXP_ID="g${gpu}_S${seed}_lr${lr}_clip${clip}_ent${ent}_lam${lam}_ep${ep}_mb${mb}_fpb${fpb}"
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
      --frames_per_batch "${fpb}" \
      --mini_batch_size "${mb}" \
      --num_epochs "${ep}" \
      --lr "${lr}" \
      --gamma "${GAMMA}" \
      --lmbda "${lam}" \
      --clip_epsilon "${clip}" \
      --entropy_coeff "${ent}" \
      --save_dir "${ARTIFACTS}" \
      --exp_name "${EXP_ID}" \
      --save_every "${SAVE_EVERY}" \
      2>&1 | tee "${LOGPATH}"

    echo ">> [${EXP_ID}] done. artifacts: ${ARTIFACTS}/${EXP_ID}"
  )
}

# -------- count combos --------
COUNT=0
for s in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for clip in "${CLIPS[@]}"; do
      for ent in "${ENTS[@]}"; do
        for lam in "${LAMBDAS[@]}"; do
          for ep in "${EPOCHS[@]}"; do
            for mb in "${MINIBATCH[@]}"; do
              for fpb in "${FPB[@]}"; do
                COUNT=$((COUNT+1))
              done
            done
          done
        done
      done
    done
  done
done
echo "GPUS: ${GPUS[*]}  (NGPU=${NGPU})"
echo "Total runs: ${COUNT}  (GROUP=${GROUP}, JOBS=${JOBS})"

# -------- dispatch (round-robin GPU assignment) --------
idx=0
running=0
for s in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for clip in "${CLIPS[@]}"; do
      for ent in "${ENTS[@]}"; do
        for lam in "${LAMBDAS[@]}"; do
          for ep in "${EPOCHS[@]}"; do
            for mb in "${MINIBATCH[@]}"; do
              for fpb in "${FPB[@]}"; do
                gpu="${GPUS[$((idx % NGPU))]}"
                idx=$((idx+1))

                run_one "$gpu" "$s" "$lr" "$clip" "$ent" "$lam" "$ep" "$mb" "$fpb" &
                running=$((running+1))
                # 동시 실행 제한
                if (( running % JOBS == 0 )); then
                  wait
                fi
              done
            done
          done
        done
      done
    done
  done
done
wait

echo "All runs finished."
