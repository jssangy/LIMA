#!/bin/bash
# Slimmed straggler-fix screening (yield variants disqualified on a279:
# completion collapse 279->198 from cascading demotions at shared sinks).
# Usage: tools/straggler_screen2.sh <outdir> <cell>...  cell = map|scendir|agents|sidx
cd ~/lima-dev
BIN=${BIN:-build_gating/lima}
OUT=$1; shift
mkdir -p "$OUT"

declare -A VARIANTS=(
  [base]=""
  [retreat]="--pibt-arm-retreat"
  [agerate]="--pibt-age-rate"
  [replan8]="--pibt-replan 8"
  [retreat_replan8]="--pibt-arm-retreat --pibt-replan 8"
)

for cell in "$@"; do
  IFS="|" read -r map sdir agents sidx <<< "$cell"
  for v in base retreat agerate replan8 retreat_replan8; do
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
