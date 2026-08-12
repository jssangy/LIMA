#!/bin/bash
# Solver comparison at the local-instance level (experiment E10).
cd "$(dirname "$0")/.." || exit 1
run() {
  local name=$1; shift
  for e in 5 10 15; do
    for n in 6 10 14; do
      ./build/lima --mode bench --bench-arms "$e,$e,$e,$e" --bench-n "$n" \
        --bench-instances 200 --seed 7 "$@" \
        --output "/tmp/sb_${name}_${e}_${n}.csv" 2>/dev/null
    done
  done
}
run ida
run idaopt --bound-step 0
run idanofp --no-fastpath
run greedy --solver greedy
python3 tools/solver_bench_report.py
