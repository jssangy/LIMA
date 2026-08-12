#!/bin/bash
# E0v3: full instrumented grid under the frozen LIMA configuration.
cd "$(dirname "$0")/.." || exit 1
export OUT=~/lima-dev/results/E0v3
mkdir -p "$OUT/logs" "$OUT/metrics"

JOBS="$OUT/jobs.txt"
if [ ! -f "$JOBS" ]; then
  : > "$JOBS"
  while read -r map sdir counts; do
    for a in $counts; do
      for s in 0 1 2 3 4 5 6 7 8 9; do
        echo "${map}|${sdir}|${a}|${s}" >> "$JOBS"
      done
    done
  done << 'CELLS'
warehouse_10_20 warehouse-10-20 26 132 264 529 794 1059 1324 1589
warehouse_20_40 warehouse-20-40 104 524 1049 2099 3149 4199 5249 6299
cross_3030 cross-30-30 102 510 1020 2040 3060 4080 5100 6120
CELLS
fi

run_one() {
  IFS='|' read -r map sdir agents sidx <<< "$1"
  local tag="${map}_a${agents}_x${sidx}"
  local line="$OUT/logs/${tag}.line"
  [ -f "$line" ] && return
  local tmp="/tmp/e0v3_${tag}.txt"
  local out
  out=$(timeout 3600 ./build/lima --mode solve --map "data/maps/${map}.map" \
    --scenario "data/scenarios/${sdir}/${sdir}_s${sidx}.scen" --agents "$agents" \
    --planner bfs --seed "$sidx" --metrics "$OUT/metrics/${tag}" \
    --output "$tmp" --validate-conflicts 2>&1 | tail -1)
  rm -f "$tmp"
  echo "${map}|${agents}|${sidx}|${out}" > "$line"
}
export -f run_one

xargs -P 8 -I{} bash -c 'run_one "$@"' _ {} < "$JOBS"
sort "$OUT"/logs/*.line > "$OUT/summary_raw.txt"
echo DONE > "$OUT/DONE"
