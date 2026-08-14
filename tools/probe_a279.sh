#!/bin/bash
# Probe the a279 s0 straggler: after stall, dump every intersection that still
# holds members, plus the stuck agents.
cd ~/lima-dev
BIN=${BIN:-build_gating/lima}
{
  printf "step 500\n"
  for i in $(seq 0 188); do printf "intersection %d\n" "$i"; done
  printf "state\nquit\n"
} | "$BIN" --mode debug --map data/maps/warehouse_10_20.map \
    --scenario data/scenarios/warehouse-10-20/warehouse-10-20_s0.scen \
    --agents 279 --planner bfs --seed 0 --output /tmp/probe_a279_trace.txt 2>/dev/null \
  > /tmp/probe_a279_out.jsonl
python3 - <<'EOF'
import json
lines = open("/tmp/probe_a279_out.jsonl").read().strip().split("\n")
for line in lines:
    try:
        d = json.loads(line)
    except Exception:
        continue
    if "members" in d and d["members"]:
        print("INTERSECTION", json.dumps(d))
    if isinstance(d.get("agents"), list):
        for a in d["agents"]:
            if a["active"]:
                print("AGENT", json.dumps(a))
EOF
