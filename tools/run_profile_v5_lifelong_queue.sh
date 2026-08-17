#!/usr/bin/env bash
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

frozen="results/revision_final/frozen_artifacts_lifelong_profile_v5_v1"
binary="$frozen/lima"
pibt="$frozen/pibt_lifelong"
inputs="results/revision_final/lifelong_inputs_v2/MANIFEST.json"
queue_root="results/revision_final/queue_lifelong_profile_v5_v1"
mkdir -p "$queue_root/logs"

echo "053a79c9b021a4a9f9b49d961009fc828c35fb5647b8702389cee04385a9537d  $binary" \
  | sha256sum -c -
echo "7256c79d898768d4dfd75b259cf1497a10e8d02ec7e68a2d35f89fb82d7a4471  $pibt" \
  | sha256sum -c -

export PYTHONUNBUFFERED=1

python3 tools/run_final_lifelong.py \
  --input-manifest "$inputs" \
  --binary "$binary" \
  --variants bfs,swr,static-guidance \
  --densities 10,30,50 \
  --scenarios 0,1,2,3,4 \
  --horizon 10000 \
  --warmup 1000 \
  --jobs 3 \
  --output-dir results/revision_final/lifelong_lima_profile_v5_3route_v1 \
  >"$queue_root/logs/lima.log" 2>&1 &
lima_pid=$!

python3 tools/run_bonus_lifelong_pibt.py \
  --input-manifest "$inputs" \
  --binary "$pibt" \
  --adapter-source "$frozen/pibt_lifelong.cpp" \
  --upstream-repo /home/shlee/mapf-baselines/pibt2 \
  --densities 10,30,50 \
  --scenarios 0,1,2,3,4 \
  --horizon 10000 \
  --warmup 1000 \
  --jobs 2 \
  --output-dir results/revision_final/lifelong_pibt_profile_v5_v1 \
  >"$queue_root/logs/pibt.log" 2>&1 &
pibt_pid=$!

status=0
for entry in "lima:$lima_pid" "pibt:$pibt_pid"; do
  name="${entry%%:*}"
  pid="${entry##*:}"
  if wait "$pid"; then
    printf '%s\t0\n' "$name" >>"$queue_root/STATUS.tsv"
  else
    code=$?
    printf '%s\t%s\n' "$name" "$code" >>"$queue_root/STATUS.tsv"
    status=1
  fi
done

if (( status == 0 )); then
  python3 tools/summarize_lifelong.py \
    results/revision_final/lifelong_lima_profile_v5_3route_v1 \
    >>"$queue_root/logs/summary.log" 2>&1
  python3 tools/summarize_lifelong.py \
    results/revision_final/lifelong_pibt_profile_v5_v1 \
    >>"$queue_root/logs/summary.log" 2>&1
fi

exit "$status"
