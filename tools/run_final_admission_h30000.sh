#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
output="results/revision_final/admission_ablation_final_h30000_v1"
mkdir -p "$output"
exec python3 tools/run_final_admission_ablation.py \
  --freeze-manifest results/revision_final/frozen_artifacts_paper_h30000_v1/MANIFEST.json \
  --certified-manifest results/revision_final/certified_inputs_v3/MANIFEST.json \
  --binary results/revision_final/frozen_artifacts_paper_h30000_v1/lima \
  --max-steps 30000 \
  --scenarios 0,1,2,3,4 \
  --jobs 1 \
  --output-dir "$output" \
  >"$output/runner.log" 2>"$output/runner.err"
