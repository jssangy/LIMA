#!/bin/bash
# Full-cell throughput runs: can a budgeted-solver variant finish a DNF cell
# inside the sweep's 2400 s wall budget?
# Usage: tools/throughput_full.sh <outdir> <jobs_file> [parallel]
# jobs line: map|scendir|agents|sidx|variant
cd ~/lima-dev
BIN=${BIN:-build_gating2/lima}
OUT=$1
JOBS=$2
PAR=${3:-1}
mkdir -p "$OUT"
export BIN OUT

run_full() {
  IFS="|" read -r map sdir agents sidx v <<< "$1"
  case "$v" in
    base) flags="" ;;
    nodes20k) flags="--solver-nodes 20000" ;;
    nodes200k) flags="--solver-nodes 200000" ;;
    nodes20k_fix) flags="--solver-nodes 20000 --pibt-arm-retreat-last --pibt-replan 8" ;;
    nodes200k_fix) flags="--solver-nodes 200000 --pibt-arm-retreat-last --pibt-replan 8" ;;
    nodes20k_fix_subset) flags="--solver-nodes 20000 --pibt-arm-retreat-last --pibt-replan 8 --subset-scheduling" ;;
    nodes20k_fix_nodisc) flags="--solver-nodes 20000 --pibt-arm-retreat-last --pibt-replan 8 --no-discharge" ;;
    fix) flags="--pibt-arm-retreat-last --pibt-replan 8" ;;
    # Straggler fix that survived the tournament (replan only, no arm retreat)
    # combined with the solver node budget.
    nodes20k_replan) flags="--solver-nodes 20000 --pibt-replan 8" ;;
    ttdom20k_replan) flags="--lb-mode tt --dominance --solver-nodes 20000 --pibt-replan 8" ;;
    ttdom20k_replan_subset) flags="--lb-mode tt --dominance --solver-nodes 20000 --pibt-replan 8 --subset-scheduling" ;;
    *) echo "unknown variant $v" >&2; return 1 ;;
  esac
  tag="${map}_a${agents}_x${sidx}_${v}"
  line="$OUT/${tag}.line"
  [ -f "$line" ] && return
  out=$(timeout "${CELL_TIMEOUT:-2400}" "$BIN" --mode solve \
    --map "data/maps/${map}.map" \
    --scenario "data/scenarios/${sdir}/${sdir}_s${sidx}.scen" \
    --agents "$agents" --planner bfs --seed "$sidx" \
    --output "/tmp/tf_${tag}.txt" $flags 2>&1 | tail -1)
  rm -f "/tmp/tf_${tag}.txt"
  [ -z "$out" ] && out="status=wall_timeout_2400"
  echo "${map}|${agents}|${sidx}|${v}|${out}" > "$line"
}
export -f run_full

xargs -P "$PAR" -I{} bash -c 'run_full "$@"' _ {} < "$JOBS"
echo DONE > "$OUT/DONE"
