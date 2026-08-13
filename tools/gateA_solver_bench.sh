#!/bin/bash
# Gate A solver tournament grid (bench mode auto-raises max_capacity).
#
# Arms sweep: 5..20 per arm; N at 30/60/90% of the summed capacity; 200
# instances per cell, seed 7.  The 16- and 20-high rows deliberately exceed
# CPMP-benchmark tier norms (<= 10) -- the literature-flagged height sweep.
# CSVs land in results/gateA_bench/; summarize with gateA_solver_report.py.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/gateA_bench
run() {
  local name=$1 arms=$2 n=$3; shift 3
  local out="results/gateA_bench/${name}_a${arms%%,*}_n${n}.csv"
  [ -s "$out" ] && return 0
  timeout 3600 ./build/lima --mode bench --bench-arms "$arms" --bench-n "$n" \
    --bench-instances 200 --seed 7 "$@" --output "$out" 2>/dev/null \
    || echo "TIMEOUT/FAIL $name $arms n=$n"
}
grid() {
  local name=$1; shift
  run "$name" 5,5,5,5    6  "$@"
  run "$name" 5,5,5,5    12 "$@"
  run "$name" 5,5,5,5    18 "$@"
  run "$name" 10,10,10,10 12 "$@"
  run "$name" 10,10,10,10 24 "$@"
  run "$name" 10,10,10,10 36 "$@"
  run "$name" 16,16,16,16 19 "$@"
  run "$name" 16,16,16,16 38 "$@"
  run "$name" 16,16,16,16 58 "$@"
  run "$name" 20,20,20,20 24 "$@"
  run "$name" 20,20,20,20 48 "$@"
  run "$name" 20,20,20,20 72 "$@"
}
# 2M expanded nodes ~ 0.5-0.9 s/instance worst case: an order of magnitude
# beyond the online ms envelope, small enough to keep all-fail cells bounded.
# Optional args restrict which configs run (parallel lanes); default = all.
budget="--solver-nodes 2000000"
want() { [ $# -eq 1 ] && return 0; local c; for c in "${@:2}"; do [ "$c" = "$1" ] && return 0; done; return 1; }
want ida_legacy "$@" && grid ida_legacy $budget
want ida_bf     "$@" && grid ida_bf     $budget --lb-mode bf
want ida_tt     "$@" && grid ida_tt     $budget --lb-mode tt
want ida_tt_dom "$@" && grid ida_tt_dom $budget --lb-mode tt --dominance
want ida_tt_opt "$@" && grid ida_tt_opt $budget --lb-mode tt --dominance --bound-step 0 --no-fastpath
want beam       "$@" && grid beam       --solver beam
want greedy     "$@" && grid greedy     --solver greedy
echo done
