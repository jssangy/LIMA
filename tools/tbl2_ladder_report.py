#!/usr/bin/env python3
"""Pair the two paper-ladder arms cell by cell and summarize completion."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "paperladder" / "logs"
TILES = {"warehouse_10_20": 2649, "warehouse_20_40": 10499, "cross_3030": 10200}
CAP = 2400.0

rows = {}
for path in sorted(LOGS.glob("*.line")):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        continue
    parts = text.split("|", 4)
    if len(parts) < 4:
        continue
    arm, map_name, agents, scen = parts[0], parts[1], int(parts[2]), int(parts[3])
    f = dict(re.findall(r"(\w+)=([^ ]+)", parts[4] if len(parts) > 4 else ""))
    rows[(map_name, agents, scen, arm)] = f or {"status": "wallclock_dnf"}


def fmt(f):
    if not f:
        return "        -"
    st = f.get("status", "?")
    done = f.get("completed", "-")
    wall = float(f.get("elapsed_seconds", CAP))
    if st == "completed":
        return f"ok {wall:8.1f}s"
    if st == "wallclock_dnf":
        return "DNF >2400s"
    got, want = (int(x) for x in done.split("/"))
    return f"-{want - got:<3d} {wall:7.1f}s"


print(f"{'map':16}{'N':>6}{'dens':>6}{'s':>3}   {'paper instances':>16}   {'our instances':>16}")
print("-" * 74)
tally = {"orig": [0, 0], "ours": [0, 0]}
for (map_name, agents, scen, arm) in sorted(rows):
    if arm != "orig":
        continue
    o = rows.get((map_name, agents, scen, "orig"))
    u = rows.get((map_name, agents, scen, "ours"))
    dens = 100.0 * agents / TILES[map_name]
    print(f"{map_name:16}{agents:6}{dens:5.0f}%{scen:3}   {fmt(o):>16}   {fmt(u):>16}")
for (map_name, agents, scen, arm), f in rows.items():
    tally[arm][0] += 1
    if f.get("status") == "completed":
        tally[arm][1] += 1
print()
for arm, label in (("orig", "paper instances"), ("ours", "our instances")):
    n, ok = tally[arm]
    print(f"{label:18}: {ok}/{n} cells fully completed")
