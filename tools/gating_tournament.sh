#!/bin/bash
# Gate C/D straggler tournament: variant x cell matrix, parallel.
#
# Usage: tools/gating_tournament.sh <jobs_file> <outdir> [parallel]
# jobs_file lines: map|scendir|agents|sidx|variant_name
# Variant flags come from the table below.
cd ~/lima-dev
BIN=${BIN:-build_gating/lima}
JOBS=$1
OUT=$2
PAR=${3:-4}
mkdir -p "$OUT"
export BIN OUT

run_cell() {
  IFS="|" read -r map sdir agents sidx v <<< "$1"
  case "$v" in
    base) flags="" ;;
    retreat) flags="--pibt-arm-retreat" ;;
    agerate) flags="--pibt-age-rate" ;;
    replan8) flags="--pibt-replan 8" ;;
    retreat_replan8) flags="--pibt-arm-retreat --pibt-replan 8" ;;
    yield) flags="--pibt-sink-yield" ;;
    retreat_nodisc) flags="--pibt-arm-retreat --no-discharge" ;;
    retreat_drandom) flags="--pibt-arm-retreat --discharge-random" ;;
    retreat_dpartial07) flags="--pibt-arm-retreat --discharge-partial 0.7" ;;
    retreat_dallarms) flags="--pibt-arm-retreat --discharge-all-arms" ;;
    retreat_dstalled) flags="--pibt-arm-retreat --discharge-stalled-neighbor" ;;
    retreat_hyst1) flags="--pibt-arm-retreat --admit-hysteresis 1" ;;
    retreat_subset) flags="--pibt-arm-retreat --subset-scheduling" ;;
    retreat_capfix) flags="--pibt-arm-retreat --capacity-formula paper" ;;
    shuffle*) flags="--pibt-arm-retreat --shuffle-order ${v#shuffle}" ;;
    *) echo "unknown variant $v" >&2; return 1 ;;
  esac
  tag="${map}_a${agents}_x${sidx}_${v}"
  line="$OUT/${tag}.line"
  [ -f "$line" ] && return
  out=$(timeout "${CELL_TIMEOUT:-2400}" "$BIN" --mode solve \
    --map "data/maps/${map}.map" \
    --scenario "data/scenarios/${sdir}/${sdir}_s${sidx}.scen" \
    --agents "$agents" --planner bfs --seed "$sidx" \
    --output "/tmp/gt_${tag}.txt" $flags 2>&1 | tail -1)
  rm -f "/tmp/gt_${tag}.txt"
  echo "${map}|${agents}|${sidx}|${v}|${out}" > "$line"
}
export -f run_cell

xargs -P "$PAR" -I{} bash -c 'run_cell "$@"' _ {} < "$JOBS"
echo DONE > "$OUT/DONE"
