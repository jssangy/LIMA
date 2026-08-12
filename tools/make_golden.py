#!/usr/bin/env python3
"""Build the quick-regression golden file from baseline E0 grid logs.

Selects completed cells with small elapsed time across maps and densities so a
refactored binary can be checked for behavioral identity in a couple of
minutes.  Golden lines keep only deterministic summary fields.
"""
import argparse
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--logs", type=Path, default=Path.home() / "lima/results/E0_grid/logs")
LOGS = parser.parse_args().logs
OUT = Path(__file__).resolve().parent.parent / "tests/golden/e0_quick.golden"
DETERMINISTIC = ("status", "steps", "completed", "moves", "waits", "deadlocks",
                 "intersections", "validation", "vertex_conflicts", "edge_conflicts")
MAX_ELAPSED = 45.0
SEEDS_PER_CELL = 2

def main() -> int:
    if not LOGS.is_dir():
        print(f"baseline log directory not found: {LOGS}", file=sys.stderr)
        return 1
    cells = {}
    for line_file in sorted(LOGS.glob("*_run.line")):
        text = line_file.read_text(encoding="utf-8").strip()
        parts = text.split("|")
        if len(parts) < 5:
            continue
        map_name, agents, seed = parts[0], int(parts[1]), int(parts[2])
        fields = dict(re.findall(r"(\w+)=([^ ]+)", "|".join(parts[4:])))
        if fields.get("status") != "completed":
            continue
        if float(fields.get("elapsed_seconds", "1e9")) > MAX_ELAPSED:
            continue
        kept = " ".join(f"{k}={fields[k]}" for k in DETERMINISTIC if k in fields)
        cells.setdefault((map_name, agents), []).append((seed, kept))

    lines = []
    for (map_name, agents), entries in sorted(cells.items()):
        for seed, kept in sorted(entries)[:SEEDS_PER_CELL]:
            lines.append(f"{map_name}|{agents}|{seed}|{kept}")
    if not lines:
        print("no eligible baseline lines found", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"golden cells: {len(lines)} -> {OUT}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
