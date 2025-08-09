#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
PY=${PY:-python}
SCRIPT=${SCRIPT:-train.py}          # train.py 경로(리포 루트 기준)
PROBLEM=${PROBLEM:-problems/cross/cross_1.json}

TOTAL_FRAMES=${TOTAL_FRAMES:-500000}
GAMMA=${GAMMA:-0.99}

# 실행 규모: S(작게) / M(중간=기본) / L(크게)
SIZE=${SIZE:-M}

# 동시 실행 개수 (병렬). train.py가 ckpt를 상대경로 "checkpoint/"에 저장하므로
# 각 런을 개별 작업 디렉토리에서 실행하여 충돌 방지함.
JOBS=${JOBS:-1}   # 먼저 1로 안전하게. 병렬 원하면 2~4로 올리세요.

# 반복 시드
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
ARTIFACTS="${ROOT}/artifacts"
RUNDIR="${ROOT}/runs"
mkdir -p "$LOGDIR" "$ARTIFACTS" "$RUNDIR"

GROUP="sweep_${SIZE}_$(date +%Y%m%d_%H%M%S)"
export WANDB_RUN_GROUP="${GROUP}"

# -------- helper: run one experiment in isolated working dir --------
run_one () {
  local seed="$1" lr="$2" clip="$3" ent="$4" lam="$5" ep="$6" mb="$7" fpb="$8"

  local EXP_ID="S${seed}_lr${lr}_clip${clip}_ent${ent}_lam${lam}_ep${ep}_mb${mb}_fpb${fpb}"
  local RUN_PATH="${RUNDIR}/${EXP_ID}"
  local LOGPATH="${LOGDIR}/${EXP_ID}.log"
  local ARTIPATH="${ARTIFACTS}/${EXP_ID}"

  mkdir -p "${RUN_PATH}" "${ARTIPATH}"
  # 개별 작업 디렉토리에서 실행 → checkpoint/ 충돌 방지
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
      2>&1 | tee "${LOGPATH}"
    # 결과 수집
    if [[ -d checkpoint ]]; then
      mv -f checkpoint/* "${ARTIPATH}/" 2>/dev/null || true
    fi
    echo ">> [${EXP_ID}] done"
  )
}

# -------- optionally use GNU parallel if available --------
PARALLEL_BIN="$(command -v parallel || true)"

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
  # GNU parallel 경로 (권장)
  export -f run_one
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
  # bash 백그라운드 + 슬랏 제한(간단 버전)
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
                  # 간단한 동시성 제한
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
