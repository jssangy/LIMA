#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import argparse
import statistics

MAPS = ["cross-30-30", "warehouse-10-20", "warehouse-20-40"]

# planner -> results folder name (assets/{map}/{folder}/...)
PLANNER_DIR = {
    "cbs": "results_lima_cbs",
    "bfs": "results_lima_bfs",
    "bfs_highpass": "results_lima",   # 너가 말한 bfs_highpass 결과 폴더
}

PLANNERS = ["cbs", "bfs", "bfs_highpass"]
SCEN_IDXS = list(range(10))


def read_first_row_csv(path: str) -> dict | None:
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            row = next(r, None)
            return row
    except FileNotFoundError:
        return None


def to_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=int, default=10, help="target density to summarize (default: 10)")
    ap.add_argument("--out", type=str, default="throughput_summary.csv")
    ap.add_argument("--skip-missing", action="store_true",
                    help="if set, missing files are ignored silently (default: warn)")
    args = ap.parse_args()

    rows_out = []
    for map_name in MAPS:
        for planner in PLANNERS:
            folder = PLANNER_DIR[planner]
            results_dir = os.path.join("assets", map_name, folder)

            thr_list: list[float] = []
            found = 0
            missing = 0

            for scen_idx in SCEN_IDXS:
                file_name = f"{map_name}_{args.density}_s{scen_idx}.csv"
                path = os.path.join(results_dir, file_name)

                row = read_first_row_csv(path)
                if row is None:
                    missing += 1
                    if not args.skip_missing:
                        print(f"[WARN] missing: {path}")
                    continue

                thr = to_float(row.get("throughput"))
                if thr is None:
                    print(f"[WARN] invalid throughput in: {path} (value={row.get('throughput')})")
                    continue

                thr_list.append(thr)
                found += 1

            if found == 0:
                # 아무 파일도 못 읽었으면 빈 값으로 기록
                rows_out.append({
                    "map": map_name,
                    "planner": planner,
                    "throughput_mean": "",
                    "throughput_min": "",
                    "throughput_max": "",
                    "n_files": 0,
                })
                continue

            rows_out.append({
                "map": map_name,
                "planner": planner,
                "throughput_mean": f"{statistics.mean(thr_list):.6f}",
                "throughput_min": f"{min(thr_list):.6f}",
                "throughput_max": f"{max(thr_list):.6f}",
                "n_files": found,  # 참고용(원하면 제거 가능)
            })

            print(f"[{map_name} | {planner}] density={args.density} "
                  f"n={found} mean/min/max="
                  f"{statistics.mean(thr_list):.6f}/"
                  f"{min(thr_list):.6f}/"
                  f"{max(thr_list):.6f}")

    # 저장
    out_fields = ["map", "planner", "throughput_mean", "throughput_min", "throughput_max", "n_files"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
