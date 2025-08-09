#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
PY=${PY:-python}
SCRIPT=${SCRIPT:-train.py}          # train.py 경로(리포 루트 기준)
PROBLEM=${PROBLEM:-problems/cross/cross_1.json}

TOTAL_FRAMES=${TOTAL_FRAMES:-500000}
GAMMA=${GAMMA:-0.99}
SAVE_EVERY=${SAVE_EVERY:-0}         # N 프레임마다 주기 저장(0이면 비활성)

# 실행 규모: S(작게) / M(중간=기본) / L(크게)
SIZE=${SIZE:-M}

# 동시 실행 개수 (병렬)
JOBS=${JOBS:-1}

# 반복 시드(현재는 exp_name에만 반영; 추후 train.py가 seed 옵션 지원하면 전달 가능)
SEEDS=(${SEEDS:-0 1})

# -------- arrays by SIZE --------
if [[ "$SIZE" == "S" ]]; then
  LRS=("1e-4" "3e-4" "5e-4")
  CLIPS=("0.1" "0.2" "0.3")
  ENTS=("0.0" "0.005" "0.01")
  LAMBDAS=("0.90" "0.95")
  EPOCHS=("8")                 # num_epochs
  MINIBATCH=("64")             # --mini_batch_size
  FPB=("256")                  # --frames_per_batch
elif [[ "$SIZE" == "L" ]]; then
  LRS=("5e-5" "1e-4" "2e-4" "3e-4" "5e-4" "7e-4" "1e-3")
  CLIPS=("0.05" "0.1" "0.15" "0.2" "0.25" "0.3")
  ENTS=("0.0" "0.001" "0.0025" "0.005" "0.0075" "0.01" "0.015")
  LAMBDAS=("0.90" "0.92" "0.95" "0.97")
  EPOCHS=("5" "8" "10")        # UTD 조절
  MINIBATCH=("32" "64" "128")
  FPB=("256" "512")
else # M (default)
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
ARTIFACTS="${ROOT}/artifacts"   # train.py --save_dir 로 전달
RUNDIR="${ROOT}/runs"           # 작업 디렉토리(로그/중간파일)
mkdir -p "$LOGDIR" "$ARTIFACTS" "$RUNDIR"

GROUP="sweep_${SIZE}_$(date +%Y%m%d_%H%M%S)"
export WANDB_RUN_GROUP="${GROUP}"

# -------- helper: run one experiment in isolated working dir --------
run_one () {
  local seed="$1" lr="$2" clip="$3" ent="$4" lam="$5" ep="$6" mb="$7" fpb="$8"

  local EXP_ID="S${seed}_lr${lr}_clip${clip}_ent${ent}_lam${lam}_ep${ep}_mb${mb}_fpb${fpb}"
  local RUN_PATH="${RUNDIR}/${EXP_ID}"
  local LOGPATH="${LOGDIR}/${EXP_ID}.log"

  mkdir -p "${RUN_PATH}"
  (
    cd "${RUN_PATH}"
    echo ">> [${EXP_ID}] starting in $(pwd)"
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
    echo ">> [${EXP_ID}] artifacts at: ${ARTIFACTS}/${EXP_ID}"
  )
}

# -------- optionally use GNU parallel if available --------
PARALLEL_BIN="$(command -v parallel || true)"

# make vars/functions visible to parallel subshells
export ROOT LOGDIR ARTIFACTS RUNDIR PY SCRIPT PROBLEM TOTAL_FRAMES GAMMA SAVE_EVERY
export -f run_one

# -------- count combinations for info --------
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
echo "Total runs: ${COUNT}  (GROUP=${GROUP}, SIZE=${SIZE}, JOBS=${JOBS})"

# -------- dispatch runs --------
if [[ -n "$PARALLEL_BIN" && "$JOBS" -gt 1 ]]; then
  parallel -j "${JOBS}" --halt now,fail=1 run_one \
    ::: "${SEEDS[@]}" \
    ::: "${LRS[@]}" \
    ::: "${CLIPS[@]}" \
    ::: "${ENTS[@]}" \
    ::: "${LAMBDAS[@]}" \
    ::: "${EPOCHS[@]}" \
    ::: "${MINIBATCH[@]}" \
    ::: "${FPB[@]}"
else
  running=0
  for s in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do
      for clip in "${CLIPS[@]}"; do
        for ent in "${ENTS[@]}"; do
          for lam in "${LAMBDAS[@]}"; do
            for ep in "${EPOCHS[@]}"; do
              for mb in "${MINIBATCH[@]}"; do
                for fpb in "${FPB[@]}"; do
                  run_one "$s" "$lr" "$clip" "$ent" "$lam" "$ep" "$mb" "$fpb" &
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
    done
  done
  wait
fi

echo "All runs finished."
