#!/usr/bin/env python3
"""Summarize /tmp/sb_<solver>_<e>_<n>.csv files from solver_bench_compare.sh."""
import csv

print(f"{'solver':9}{'|e|':5}{'N':4}{'fail':6}{'med_len':9}{'med_us':8}{'p99_us':8}")
for name in ("ida", "idaopt", "idanofp", "greedy"):
    for e in (5, 10, 15):
        for n in (6, 10, 14):
            rows = list(csv.DictReader(open(f"/tmp/sb_{name}_{e}_{n}.csv")))
            fails = sum(1 for r in rows if r["outcome"] != "solved")
            ok = [r for r in rows if r["outcome"] == "solved"]
            lengths = sorted(int(r["solution_len"]) for r in ok) or [0]
            walls = sorted(int(r["wall_us"]) for r in ok) or [0]
            print(f"{name:9}{e:<5}{n:<4}{fails:<6}{lengths[len(lengths)//2]:<9}"
                  f"{walls[len(walls)//2]:<8}{walls[max(0, int(len(walls)*0.99)-1)]:<8}")
