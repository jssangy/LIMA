#!/usr/bin/env bash
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

freeze="results/revision_final/frozen_artifacts_profile_v5_widest_ratio/MANIFEST.json"
binary="results/revision_final/frozen_artifacts_profile_v5_widest_ratio/lima"
certified="results/revision_final/certified_inputs_v3/MANIFEST.json"
queue_root="results/revision_final/queue_lima_profile_v5_widest_ratio"
mkdir -p "$queue_root/logs"

export PYTHONUNBUFFERED=1

python3 tools/run_final_certified_oneshot.py \
  --algorithm lima \
  --input-manifest "$certified" \
  --freeze-manifest "$freeze" \
  --lima-binary "$binary" \
  --max-steps 100000 \
  --jobs 10 \
  --no-early-stop \
  --output-dir results/revision_final/oneshot_lima_profile_v5_widest_ratio \
  >"$queue_root/logs/oneshot.log" 2>&1 &
oneshot_pid=$!

python3 tools/run_final_stochastic.py \
  --algorithm lima \
  --input-manifest "$certified" \
  --lima "$binary" \
  --densities 10,20,30 \
  --scenarios 0-4 \
  --probabilities 0.05,0.10,0.15,0.20 \
  --max-steps 5000 \
  --jobs 6 \
  --no-early-stop \
  --output-dir results/revision_final/stochastic_lima_profile_v5_widest_ratio \
  >"$queue_root/logs/stochastic.log" 2>&1 &
stochastic_pid=$!

python3 tools/run_final_admission_ablation.py \
  --freeze-manifest "$freeze" \
  --certified-manifest "$certified" \
  --binary "$binary" \
  --max-steps 100000 \
  --jobs 2 \
  --output-dir results/revision_final/admission_ablation_profile_v5_widest_ratio \
  >"$queue_root/logs/admission.log" 2>&1 &
admission_pid=$!

status=0
for entry in \
  "oneshot:$oneshot_pid" \
  "stochastic:$stochastic_pid" \
  "admission:$admission_pid"; do
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

exit "$status"
