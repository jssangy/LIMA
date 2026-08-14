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
    retreatlast) flags="--pibt-arm-retreat-last" ;;
    retreatlast_replan8) flags="--pibt-arm-retreat-last --pibt-replan 8" ;;
    retreatlast_nodisc) flags="--pibt-arm-retreat-last --no-discharge" ;;
    retreatlast_drandom) flags="--pibt-arm-retreat-last --discharge-random" ;;
    retreatlast_dpartial07) flags="--pibt-arm-retreat-last --discharge-partial 0.7" ;;
    retreatlast_dallarms) flags="--pibt-arm-retreat-last --discharge-all-arms" ;;
    retreatlast_dstalled) flags="--pibt-arm-retreat-last --discharge-stalled-neighbor" ;;
    retreatlast_ddet) flags="--pibt-arm-retreat-last --discharge-deterministic" ;;
    shufflelast*) flags="--pibt-arm-retreat-last --shuffle-order ${v#shufflelast}" ;;
    # W7: winning straggler fix (replan) under a per-cycle random processing order.
    shuffle*) flags="--pibt-replan 8 --shuffle-order ${v#shuffle}" ;;
    # W7 control: same seeds without the fix, to separate order sensitivity
    # from the straggler defect itself.
    bareshuffle*) flags="--shuffle-order ${v#bareshuffle}" ;;
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
    # Discharge re-verification (Gate D) on top of the winning straggler fix.
    replan8_nodisc) flags="--pibt-replan 8 --no-discharge" ;;
    replan8_drandom) flags="--pibt-replan 8 --discharge-random" ;;
    replan8_dpartial07) flags="--pibt-replan 8 --discharge-partial 0.7" ;;
    replan8_dpartial05) flags="--pibt-replan 8 --discharge-partial 0.5" ;;
    replan8_dallarms) flags="--pibt-replan 8 --discharge-all-arms" ;;
    replan8_dstalled) flags="--pibt-replan 8 --discharge-stalled-neighbor" ;;
    replan8_ddetonly) flags="--pibt-replan 8 --discharge-random --discharge-deterministic" ;;
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
