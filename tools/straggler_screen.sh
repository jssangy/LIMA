#!/bin/bash
# Straggler-fix screening: run each variant on the cheap straggler cells.
# Usage: tools/straggler_screen.sh <outdir> <cell>...   cell = map|scendir|agents|sidx
cd ~/lima-dev
BIN=${BIN:-build_gating/lima}
OUT=$1; shift
mkdir -p "$OUT"

declare -A VARIANTS=(
  [base]=""
  [yield]="--pibt-sink-yield"
  [retreat]="--pibt-arm-retreat"
  [agerate]="--pibt-age-rate"
  [replan8]="--pibt-replan 8"
  [yield_replan8]="--pibt-sink-yield --pibt-replan 8"
  [retreat_replan8]="--pibt-arm-retreat --pibt-replan 8"
  [yield_retreat]="--pibt-sink-yield --pibt-arm-retreat"
  [all3]="--pibt-sink-yield --pibt-arm-retreat --pibt-replan 8"
)

for cell in "$@"; do
  IFS="|" read -r map sdir agents sidx <<< "$cell"
  for v in base yield retreat agerate replan8 yield_replan8 retreat_replan8 yield_retreat all3; do
    tag="${map}_a${agents}_x${sidx}_${v}"
    line="$OUT/${tag}.line"
    [ -f "$line" ] && continue
    out=$(timeout "${CELL_TIMEOUT:-2400}" "$BIN" --mode solve \
      --map "data/maps/${map}.map" \
      --scenario "data/scenarios/${sdir}/${sdir}_s${sidx}.scen" \
      --agents "$agents" --planner bfs --seed "$sidx" \
      --output "/tmp/ss_${tag}.txt" ${VARIANTS[$v]} 2>&1 | tail -1)
    rm -f "/tmp/ss_${tag}.txt"
    echo "${map}|${agents}|${sidx}|${v}|${out}" > "$line"
    echo "${map}|${agents}|${sidx}|${v}|${out}"
  done
done
