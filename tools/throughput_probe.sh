#!/bin/bash
# Throughput-class probe: fixed step budget, measure wall time per variant.
# Usage: tools/throughput_probe.sh <outdir> <map> <scendir> <agents> <seed> <maxsteps> <variant>...
cd ~/lima-dev
BIN=${BIN:-build_gating/lima}
OUT=$1; map=$2; sdir=$3; agents=$4; sidx=$5; steps=$6; shift 6
mkdir -p "$OUT"

flags_for() {
  case "$1" in
    base) echo "" ;;
    nodes2m) echo "--solver-nodes 2000000" ;;
    nodes200k) echo "--solver-nodes 200000" ;;
    nodes20k) echo "--solver-nodes 20000" ;;
    ttdom) echo "--lb-mode tt --dominance" ;;
    ttdom_nodes2m) echo "--lb-mode tt --dominance --solver-nodes 2000000" ;;
    ttdom_nodes200k) echo "--lb-mode tt --dominance --solver-nodes 200000" ;;
    ttdom_nodes20k) echo "--lb-mode tt --dominance --solver-nodes 20000" ;;
    nodes20k_retreat) echo "--solver-nodes 20000 --pibt-arm-retreat" ;;
    ttdom_nodes20k_retreat) echo "--lb-mode tt --dominance --solver-nodes 20000 --pibt-arm-retreat" ;;
    ttdom_nodes20k_retreat_nodisc) echo "--lb-mode tt --dominance --solver-nodes 20000 --pibt-arm-retreat --no-discharge" ;;
    *) echo "UNKNOWN" ;;
  esac
}

for v in "$@"; do
  f=$(flags_for "$v")
  [ "$f" = "UNKNOWN" ] && { echo "unknown variant $v" >&2; continue; }
  tag="${map}_a${agents}_x${sidx}_s${steps}_${v}"
  line="$OUT/${tag}.line"
  [ -f "$line" ] && continue
  start=$(date +%s.%N)
  out=$(timeout "${CELL_TIMEOUT:-2400}" "$BIN" --mode solve \
    --map "data/maps/${map}.map" \
    --scenario "data/scenarios/${sdir}/${sdir}_s${sidx}.scen" \
    --agents "$agents" --planner bfs --seed "$sidx" --max-steps "$steps" \
    --output "/tmp/tp_${tag}.txt" $f 2>&1 | tail -1)
  end=$(date +%s.%N)
  rm -f "/tmp/tp_${tag}.txt"
  echo "${map}|${agents}|${sidx}|${steps}|${v}|wall=$(echo "$end - $start" | bc)|${out}" > "$line"
  cat "$line"
done
