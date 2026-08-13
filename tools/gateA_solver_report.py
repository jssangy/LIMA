#!/usr/bin/env python3
"""Summarize results/gateA_bench/*.csv into a markdown table on stdout."""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results" / "gateA_bench"
SOLVERS = ("ida_legacy", "ida_bf", "ida_tt", "ida_tt_dom", "ida_tt_opt", "beam", "greedy")
GRID = [(5, (6, 12, 18)), (10, (12, 24, 36)), (16, (19, 38, 58)), (20, (24, 48, 72))]


def pct(sorted_values, q):
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def main():
    print("| arms | N | solver | fail | med len | med us | p99 us | med nodes | p99 nodes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, ns in GRID:
        for n in ns:
            for name in SOLVERS:
                path = ROOT / f"{name}_a{arm}_n{n}.csv"
                if not path.exists():
                    print(f"| {arm}x4 | {n} | {name} | (missing) | | | | | |")
                    continue
                rows = list(csv.DictReader(open(path)))
                fails = sum(1 for r in rows if r["outcome"] != "solved")
                ok = [r for r in rows if r["outcome"] == "solved"]
                lengths = sorted(int(r["solution_len"]) for r in ok)
                walls = sorted(int(r["wall_us"]) for r in ok)
                nodes = sorted(int(r["expanded"]) for r in ok)
                print(f"| {arm}x4 | {n} | {name} | {fails}/{len(rows)}"
                      f" | {statistics.median(lengths) if lengths else '-'}"
                      f" | {statistics.median(walls) if walls else '-'}"
                      f" | {pct(walls, 0.99)} | {statistics.median(nodes) if nodes else '-'}"
                      f" | {pct(nodes, 0.99)} |")


if __name__ == "__main__":
    main()
