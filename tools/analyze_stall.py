#!/usr/bin/env python3
"""Analyze a debug-REPL stall dump produced by the M10 diagnosis run."""
import json
import sys
from collections import Counter

lines = open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/m10_repl.jsonl",
             encoding="utf-8").read().splitlines()
step_res = json.loads(lines[1])
print("STALL:", step_res["stalled"], "summary:", step_res["summary"])
inv = next((l for l in lines if l.startswith('{"ok"')), "?")
print("invariants:", inv)
state = json.loads(next(l for l in lines if l.startswith('{"summary"')))
stuck = [a for a in state["agents"] if a["active"]]
print("stuck agents:", len(stuck), "wait_reasons:", Counter(a["wait_reason"] for a in stuck))
print("scheduled among stuck:", sum(1 for a in stuck if a["scheduled"]))
for a in stuck[:6]:
    print("  id={id} pos={pos} next={next} wait={wait_steps} reason={wait_reason} "
          "sched={scheduled} rem={route_remaining}".format(**a))
cells = {tuple(a["pos"]) for a in stuck}
mutual = sum(1 for a in stuck if tuple(a["next"]) in cells and a["next"] != a["pos"])
print("stuck agents whose next cell holds another stuck agent:", mutual)
inters = [json.loads(l) for l in lines[4:4 + 189] if l.startswith('{"id"')]
hot = [x for x in inters if x["members"]]
for x in hot[:10]:
    print("  I{id} center={center} members={m} cap={capacity} avail={available} "
          "active={active} waiting={waiting}".format(m=len(x["members"]), **x))
print("intersections with members:", len(hot),
      "active:", sum(1 for x in hot if x["active"]),
      "waiting:", sum(1 for x in hot if x["waiting"]))
