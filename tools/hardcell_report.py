#!/usr/bin/env python3
"""Summarize the hard-cell rediscovery sweep into a landscape table.

Reads results/hardcells/logs/*.line (one line per cell, possibly empty body
when the cell exceeded the wall-clock cap) and prints a density-ordered
landscape plus the runtime distribution used for the unified budget T.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "hardcells" / "logs"
CAP = 2400.0
TILES = {"warehouse_10_20": 5577, "warehouse_20_40": 22358, "cross_3030": 17834}


def load():
    rows = []
    for path in sorted(LOGS.glob("*.line")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parts = text.split("|", 3)
        if len(parts) < 3:
            continue
        map_name, agents, scen = parts[0], int(parts[1]), int(parts[2])
        body = parts[3] if len(parts) > 3 else ""
        fields = dict(re.findall(r"(\w+)=([^ ]+)", body))
        if not fields:
            fields = {"status": "wallclock_dnf"}
        rows.append((map_name, agents, scen, fields))
    return rows


def main():
    rows = load()
    header = f"{'map':17}{'N':>6}{'dens':>6}{'s':>3}  {'status':14}{'completed':>13}{'steps':>7}{'wall_s':>10}"
    print(header)
    print("-" * len(header))
    for map_name, agents, scen, f in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
        dens = 100.0 * agents / TILES[map_name]
        wall = float(f.get("elapsed_seconds", CAP))
        done = f.get("completed", "-")
        gap = ""
        if "/" in done:
            got, want = done.split("/")
            if got != want:
                gap = f"  (-{int(want) - int(got)})"
        print(f"{map_name:17}{agents:6}{dens:5.0f}%{scen:3}  {f.get('status', '?'):14}"
              f"{done:>13}{f.get('steps', '-'):>7}{wall:10.1f}{gap}")

    print()
    ok = [float(f["elapsed_seconds"]) for *_, f in rows if f.get("status") == "completed"]
    ok.sort()
    if ok:
        def pct(p):
            return ok[min(len(ok) - 1, int(round(p / 100.0 * (len(ok) - 1))))]
        print(f"completed cells: {len(ok)}/{len(rows)}")
        print(f"wall-clock of completed cells (s): min {ok[0]:.2f}  p50 {pct(50):.1f}  "
              f"p90 {pct(90):.1f}  p99 {pct(99):.1f}  max {ok[-1]:.1f}")
    classes = {}
    for map_name, agents, scen, f in rows:
        st = f.get("status", "?")
        if st == "completed":
            key = "completed"
        elif st == "wallclock_dnf":
            key = "wallclock_dnf (>=2400s)"
        else:
            done = f.get("completed", "0/0")
            got, want = (int(x) for x in done.split("/"))
            key = f"straggler ({st})"
            classes.setdefault("_straggler_gaps", []).append((map_name, agents, scen, want - got))
        classes[key] = classes.get(key, 0) + 1
    gaps = classes.pop("_straggler_gaps", [])
    print("\nclasses:", {k: v for k, v in sorted(classes.items())})
    if gaps:
        print("straggler cells (agents left):")
        for map_name, agents, scen, gap in sorted(gaps):
            print(f"  {map_name} a{agents} s{scen}: {gap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
