#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

root="results/diagnostics/wh10_d60_s9_recovery_audit_v1"
binary="results/revision_final/frozen_artifacts_paper_h30000_v1/lima"
map="results/revision_final/certified_inputs_v3/maps/warehouse_10_20_unique_goals.map"
scenario="results/revision_final/certified_inputs_v3/scenarios/warehouse_10_20/warehouse_10_20_paired_d60_s9.scen"
common=(
  --profile lima-default --mode solve
  --map "$map" --scenario "$scenario"
  --agents 1589 --seed 9 --max-steps 30000 --stall-threshold 30001
  --goal-behavior disappear --no-trace
)

mkdir -p "$root"/{no_discharge,exclusive_reserve,exclusive_age,composite}

"$binary" "${common[@]}" --metrics "$root/no_discharge" --no-discharge \
  >"$root/no_discharge.log" 2>&1 &
p1=$!
"$binary" "${common[@]}" --metrics "$root/exclusive_reserve" --recirc-exclusive reserve \
  >"$root/exclusive_reserve.log" 2>&1 &
p2=$!
"$binary" "${common[@]}" --metrics "$root/exclusive_age" --recirc-exclusive age \
  >"$root/exclusive_age.log" 2>&1 &
p3=$!
"$binary" "${common[@]}" --metrics "$root/composite" --discharge-policy composite \
  >"$root/composite.log" 2>&1 &
p4=$!

status=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  wait "$pid" || status=$?
done

for log in "$root"/*.log; do
  echo "=== $log"
  cat "$log"
done
exit "$status"
