#!/bin/bash
# E6: fair-budget baseline grids (LaCAM 60s, PIBT) on the paper instances.
export OUT=~/lima-dev/results/E6_baselines
mkdir -p "$OUT/logs" "$OUT/raw"
BASE=~/mapf-baselines

JOBS="$OUT/jobs.txt"
if [ ! -f "$JOBS" ]; then
  : > "$JOBS"
  while read -r map sdir counts; do
    for a in $counts; do
      for s in 0 1 2 3 4 5 6 7 8 9; do
        echo "${map}|${sdir}|${a}|${s}|lacam" >> "$JOBS"
        echo "${map}|${sdir}|${a}|${s}|pibt" >> "$JOBS"
      done
    done
  done << 'CELLS'
warehouse_10_20 warehouse-10-20 26 132 264 529 794 1059 1324 1589
warehouse_20_40 warehouse-20-40 104 524 1049 2099 3149 4199 5249 6299
cross_3030 cross-30-30 102 510 1020 2040 3060 4080 5100 6120
CELLS
fi

run_one() {
  IFS='|' read -r map sdir agents sidx algo <<< "$1"
  local tag="${map}_a${agents}_x${sidx}_${algo}"
  local line="$OUT/logs/${tag}.line"
  [ -f "$line" ] && return
  local scen=~/lima-dev/data/scenarios/${sdir}/${sdir}_s${sidx}.scen
  local mapf=$BASE/data/${map}.map
  local raw="$OUT/raw/${tag}.txt"
  if [ "$algo" = "lacam" ]; then
    timeout 90 $BASE/lacam/build/main -i "$scen" -m "$mapf" -N "$agents" -s "$sidx" -t 60 -o "$raw" > /dev/null 2>&1
  else
    timeout 400 $BASE/pibt2/build/mapf -i "$scen" -m "$mapf" -s PIBT -N "$agents" -t 300 -o "$raw" > /dev/null 2>&1
  fi
  local solved makespan comp
  solved=$(grep -o "^solved=[0-9]*" "$raw" 2>/dev/null | head -1 | cut -d= -f2)
  makespan=$(grep -o "^makespan=[0-9]*" "$raw" 2>/dev/null | head -1 | cut -d= -f2)
  comp=$(grep -o "^comp_time=[0-9]*" "$raw" 2>/dev/null | head -1 | cut -d= -f2)
  echo "${map}|${agents}|${sidx}|${algo}|solved=${solved:-0} makespan=${makespan:-} comp_ms=${comp:-}" > "$line"
  grep -v "^starts=\|^solution=\|^[0-9]*:" "$raw" > "${raw}.head" 2>/dev/null && mv "${raw}.head" "$raw"
}
export -f run_one
export BASE

xargs -P 8 -I{} bash -c 'run_one "$@"' _ {} < "$JOBS"
sort "$OUT"/logs/*.line > "$OUT/summary_raw.txt"
echo DONE > "$OUT/DONE"
