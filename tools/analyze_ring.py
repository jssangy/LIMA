#!/usr/bin/env python3
"""Check whether the stalled agents form a closed next-pointer cycle."""
import json
import sys

lines = open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/m10_repl.jsonl",
             encoding="utf-8").read().splitlines()
state = json.loads(next(l for l in lines if l.startswith('{"summary"')))
stuck = [a for a in state["agents"] if a["active"]]
by_pos = {tuple(a["pos"]): a for a in stuck}
print("stuck:", len(stuck))
visited = set()
for seed in stuck:
    if seed["id"] in visited:
        continue
    chain = []
    seen_local = {}
    a = seed
    while True:
        if a["id"] in visited:
            if a["id"] in seen_local:
                cycle = chain[seen_local[a["id"]]:]
                print(f"CYCLE len={len(cycle)}:",
                      [(c["id"], c["wait_reason"]) for c in cycle])
            break
        visited.add(a["id"])
        seen_local[a["id"]] = len(chain)
        chain.append(a)
        nxt = by_pos.get(tuple(a["next"]))
        if nxt is None or a["next"] == a["pos"]:
            print(f"chain from {seed['id']} DEAD-ENDS at {a['id']} "
                  f"({a['wait_reason']}, next={a['next']}, "
                  f"{'EMPTY cell' if tuple(a['next']) not in by_pos else 'self'})")
            break
        a = nxt
