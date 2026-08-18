#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

freeze="results/revision_final/frozen_artifacts_paper_h30000_v1/MANIFEST.json"
lima="results/revision_final/frozen_artifacts_paper_h30000_v1/lima"
inputs="results/revision_final/certified_inputs_v3/MANIFEST.json"
targets="d01,d05,d10,d20,d30,d40,d50,d60,d65,d70"
queue="results/revision_final/queue_paper_h30000_v1"
mkdir -p "$queue/logs"
export PYTHONUNBUFFERED=1

run_oneshot() {
  local algorithm="$1" jobs="$2" output="$3" reuse="$4"
  python3 tools/run_final_certified_oneshot.py \
    --algorithm "$algorithm" \
    --input-manifest "$inputs" \
    --freeze-manifest "$freeze" \
    --lima-binary "$lima" \
    --targets "$targets" \
    --scenarios 0,1,2,3,4,5,6,7,8,9 \
    --max-steps 30000 \
    --cbs-max-expansions 100000 \
    --lacam-max-iterations 100000 \
    --jobs "$jobs" \
    --no-early-stop \
    --reuse-records-from "$reuse" \
    --output-dir "$output" \
    >"$queue/logs/oneshot_${algorithm}.log" 2>&1
}

run_oneshot lima 4 \
  results/revision_final/oneshot_lima_final_h30000_v1 \
  results/revision_final/oneshot_lima_profile_v5_widest_ratio &
p_lima=$!
run_oneshot cbs 2 \
  results/revision_final/oneshot_cbs_final_h30000_v1 \
  results/revision_final/oneshot_cbs_certified_step_v3 &
p_cbs=$!
run_oneshot lacam 3 \
  results/revision_final/oneshot_lacam_final_h30000_v1 \
  results/revision_final/oneshot_lacam_certified_step_v3 &
p_lacam=$!
run_oneshot pibt 3 \
  results/revision_final/oneshot_pibt_final_h30000_v1 \
  results/revision_final/oneshot_pibt_certified_step_v3 &
p_pibt=$!

status=0
for entry in "lima:$p_lima" "cbs:$p_cbs" "lacam:$p_lacam" "pibt:$p_pibt"; do
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
(( status == 0 )) || exit "$status"

run_stochastic() {
  local algorithm="$1" jobs="$2" output="$3"
  python3 tools/run_final_stochastic.py \
    --algorithm "$algorithm" \
    --input-manifest "$inputs" \
    --lima "$lima" \
    --densities 10,20,30 \
    --scenarios 0-4 \
    --probabilities 0.05,0.10,0.15,0.20 \
    --max-steps 30000 \
    --lacam-max-iterations 100000 \
    --jobs "$jobs" \
    --no-early-stop \
    --output-dir "$output" \
    >"$queue/logs/stochastic_${algorithm}.log" 2>&1
}

run_stochastic lima 4 results/revision_final/stochastic_lima_final_h30000_v1 &
p_stoch_lima=$!
run_stochastic lacam-replan 4 \
  results/revision_final/stochastic_lacam_replan_final_h30000_v1 &
p_stoch_lacam=$!
run_stochastic pibt 4 results/revision_final/stochastic_pibt_final_h30000_v1 &
p_stoch_pibt=$!

status=0
for entry in \
  "stochastic_lima:$p_stoch_lima" \
  "stochastic_lacam:$p_stoch_lacam" \
  "stochastic_pibt:$p_stoch_pibt"; do
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
exit "$status"
