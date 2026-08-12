#!/bin/bash
# Gating-variant and solver comparison grid (experiments E2/E10/E11 support).
cd "$(dirname "$0")/.." || exit 1
OUT=~/lima-dev/results/variant_grid
mkdir -p "$OUT/logs"

CELLS=(
  "warehouse_10_20 warehouse-10-20 794 4"
  "warehouse_10_20 warehouse-10-20 794 5"
  "warehouse_10_20 warehouse-10-20 1059 0"
  "warehouse_10_20 warehouse-10-20 1059 1"
  "warehouse_10_20 warehouse-10-20 1059 2"
  "warehouse_10_20 warehouse-10-20 1589 0"
  "warehouse_10_20 warehouse-10-20 1589 1"
  "warehouse_20_40 warehouse-20-40 3149 0"
  "warehouse_20_40 warehouse-20-40 6299 0"
  "cross_3030 cross-30-30 3060 0"
  "cross_3030 cross-30-30 6120 0"
)
VARIANTS=(
  "base|"
  "resync|--gate-resync"
  "resyncrot|--gate-resync --rotation"
  "resyncopt|--gate-resync --bound-step 0"
)

run_one() {
  local map=$1 sdir=$2 agents=$3 sidx=$4 vname=$5 vflags=$6
  local tag="${map}_a${agents}_x${sidx}_${vname}"
  local line="$OUT/logs/${tag}.line"
  [ -f "$line" ] && return
  local tmp="/tmp/vg_${tag}.txt"
  local out
  out=$(timeout 1800 ./build/lima --mode solve --map "data/maps/${map}.map" \
    --scenario "data/scenarios/${sdir}/${sdir}_s${sidx}.scen" --agents "$agents" \
    --planner bfs --seed "$sidx" $vflags --output "$tmp" --validate-conflicts 2>&1 | tail -1)
  rm -f "$tmp"
  echo "${map}|${agents}|${sidx}|${vname}|${out}" > "$line"
}
export -f run_one
export OUT

JOBS=()
for cell in "${CELLS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    vname="${variant%%|*}"
    vflags="${variant#*|}"
    JOBS+=("$cell $vname|$vflags")
  done
done
printf "%s\n" "${JOBS[@]}" | xargs -P 6 -I{} bash -c 'set -- {}; a=$1 b=$2 c=$3 d=$4; rest=$5; run_one "$a" "$b" "$c" "$d" "${rest%%|*}" "${rest#*|}"'
sort "$OUT"/logs/*.line > "$OUT/summary_raw.txt"
echo DONE > "$OUT/DONE"
