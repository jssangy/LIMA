#!/bin/bash
# Gating-variant and solver comparison grid (experiments E2/E10/E11 support).
cd "$(dirname "$0")/.." || exit 1
export OUT=~/lima-dev/results/variant_grid
mkdir -p "$OUT/logs"

JOBS="$OUT/jobs.txt"
: > "$JOBS"
while read -r map sdir agents sidx; do
  echo "${map}|${sdir}|${agents}|${sidx}|base|" >> "$JOBS"
  echo "${map}|${sdir}|${agents}|${sidx}|resync|--gate-resync" >> "$JOBS"
  echo "${map}|${sdir}|${agents}|${sidx}|resyncrot|--gate-resync --rotation" >> "$JOBS"
  echo "${map}|${sdir}|${agents}|${sidx}|resyncopt|--gate-resync --bound-step 0" >> "$JOBS"
done << 'CELLS'
warehouse_10_20 warehouse-10-20 794 4
warehouse_10_20 warehouse-10-20 794 5
warehouse_10_20 warehouse-10-20 1059 0
warehouse_10_20 warehouse-10-20 1059 1
warehouse_10_20 warehouse-10-20 1059 2
warehouse_10_20 warehouse-10-20 1589 0
warehouse_10_20 warehouse-10-20 1589 1
warehouse_20_40 warehouse-20-40 3149 0
warehouse_20_40 warehouse-20-40 6299 0
cross_3030 cross-30-30 3060 0
cross_3030 cross-30-30 6120 0
CELLS

run_one() {
  IFS='|' read -r map sdir agents sidx vname vflags <<< "$1"
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

xargs -P 6 -I{} bash -c 'run_one "$@"' _ {} < "$JOBS"
sort "$OUT"/logs/*.line > "$OUT/summary_raw.txt"
echo DONE > "$OUT/DONE"
