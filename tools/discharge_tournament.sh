#!/bin/bash
# Discharge-variant tournament on the hard cells (frozen defaults underneath)
# plus targeted 60%-regime probes.
cd "$(dirname "$0")/.." || exit 1
export OUT=~/lima-dev/results/discharge_tournament
mkdir -p "$OUT/logs"

JOBS="$OUT/jobs.txt"
if [ ! -f "$JOBS" ]; then
  : > "$JOBS"
  while IFS='|' read -r map sdir agents sidx; do
    for v in "dbase|" "ddet|--discharge-deterministic" "davail|--discharge-avail-weighted" \
             "dallarms|--discharge-all-arms" "dstalled|--discharge-stalled-neighbor" \
             "dpartial5|--discharge-partial 0.5" "dpartial8|--discharge-partial 0.8"; do
      echo "${map}|${sdir}|${agents}|${sidx}|${v}|" >> "$JOBS"
    done
  done << 'CELLS'
warehouse_10_20|warehouse-10-20|794|4
warehouse_10_20|warehouse-10-20|794|5
warehouse_10_20|warehouse-10-20|1059|0
warehouse_10_20|warehouse-10-20|1059|1
warehouse_10_20|warehouse-10-20|1059|2
cross_3030|cross-30-30|3060|0
CELLS
  # 60%-regime probes (short horizons, completion count at fixed t)
  {
    echo "warehouse_10_20|warehouse-10-20|1589|0|p_ctrl||600"
    echo "warehouse_10_20|warehouse-10-20|1589|0|p_nodis|--subset-scheduling --no-discharge|600"
    echo "warehouse_10_20|warehouse-10-20|1589|0|p_part5|--subset-scheduling --discharge-partial 0.5|600"
    echo "warehouse_10_20|warehouse-10-20|1589|0|p_cap8|--subset-scheduling --isolation-cap 8|600"
    echo "warehouse_10_20|warehouse-10-20|1589|0|p_subs|--subset-scheduling|600"
  } >> "$JOBS"
fi

run_one() {
  IFS='|' read -r map sdir agents sidx vname vflags msteps <<< "$1"
  local tag="${map}_a${agents}_x${sidx}_${vname}"
  local line="$OUT/logs/${tag}.line"
  [ -f "$line" ] && return
  local steps_arg=""
  [ -n "$msteps" ] && steps_arg="--max-steps $msteps"
  local tmp="/tmp/dt_${tag}.txt"
  local out
  out=$(timeout 1500 ./build/lima --mode solve --map "data/maps/${map}.map" \
    --scenario "data/scenarios/${sdir}/${sdir}_s${sidx}.scen" --agents "$agents" \
    --planner bfs --seed "$sidx" $vflags $steps_arg --output "$tmp" 2>&1 | tail -1)
  rm -f "$tmp"
  echo "${map}|${agents}|${sidx}|${vname}|${out}" > "$line"
}
export -f run_one

xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} < "$JOBS"
sort "$OUT"/logs/*.line > "$OUT/summary_raw.txt"
echo DONE > "$OUT/DONE"
