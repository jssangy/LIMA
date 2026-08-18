#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
pibt2_repo="${PIBT2_REPO:-$(cd "$root/../mapf-baselines/pibt2" && pwd)}"

frozen="results/revision_final/frozen_artifacts_paper_h30000_v1"
inputs="results/revision_final/lifelong_inputs_boundary_v3/MANIFEST.json"
queue="results/revision_final/queue_lifelong_boundary_v3"
mkdir -p "$queue/logs"
export PYTHONUNBUFFERED=1

python3 tools/run_final_lifelong.py \
  --input-manifest "$inputs" \
  --binary "$frozen/lima_lifelong" \
  --variants bfs,swr,static-guidance \
  --densities 10,30,50 \
  --scenarios 0,1,2,3,4 \
  --horizon 10000 \
  --warmup 1000 \
  --jobs 4 \
  --output-dir results/revision_final/lifelong_lima_boundary_v3 \
  >"$queue/logs/lima.log" 2>&1 &
p_lima=$!

python3 tools/run_bonus_lifelong_pibt.py \
  --input-manifest "$inputs" \
  --binary "$frozen/pibt_lifelong" \
  --adapter-source "$frozen/pibt_lifelong.cpp" \
  --upstream-repo "$pibt2_repo" \
  --densities 10,30,50 \
  --scenarios 0,1,2,3,4 \
  --horizon 10000 \
  --warmup 1000 \
  --jobs 3 \
  --output-dir results/revision_final/lifelong_pibt_boundary_v3 \
  >"$queue/logs/pibt.log" 2>&1 &
p_pibt=$!

status=0
for entry in "lima:$p_lima" "pibt:$p_pibt"; do
  name="${entry%%:*}"
  pid="${entry##*:}"
  if wait "$pid"; then
    printf '%s\t0\n' "$name" >>"$queue/STATUS.tsv"
  else
    code=$?
    printf '%s\t%s\n' "$name" "$code" >>"$queue/STATUS.tsv"
    status=1
  fi
done

if (( status == 0 )); then
  python3 tools/summarize_lifelong.py \
    results/revision_final/lifelong_lima_boundary_v3 \
    >"$queue/logs/lima_summary.log" 2>&1
  python3 tools/summarize_lifelong.py \
    results/revision_final/lifelong_pibt_boundary_v3 \
    >"$queue/logs/pibt_summary.log" 2>&1
fi
exit "$status"
