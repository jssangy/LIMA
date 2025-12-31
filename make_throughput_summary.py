#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import argparse
import statistics

MAPS = ["cross-30-30", "warehouse-10-20", "warehouse-20-40"]

PLANNER_DIR = {
    "CBS": "results_lima_cbs",
    "BFS": "results_lima_bfs",
    "DoR": "results_lima",
}

PLANNERS = ["CBS", "BFS", "DoR"]
SCEN_IDXS = list(range(10))


def read_first_row_csv(path: str):
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            return next(r, None)
    except FileNotFoundError:
        return None


def to_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def write_csv(path: str, fieldnames: list[str], rows: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=int, default=10)
    ap.add_argument("--out-box", type=str, default="throughput_boxplot.csv")
    ap.add_argument("--out-sum", type=str, default="throughput_summary.csv")
    ap.add_argument("--skip-missing", action="store_true")
    args = ap.parse_args()

    # ✅ 박스플롯용: 시나리오 단위 원자료
    box_rows = []

    # ✅ 요약용: mean/min/max
    sum_rows = []

    for map_name in MAPS:
        for planner in PLANNERS:
            folder = PLANNER_DIR[planner]
            results_dir = os.path.join("assets", map_name, folder)

            thr_list = []
            found = 0

            for scen_idx in SCEN_IDXS:
                file_name = f"{map_name}_{args.density}_s{scen_idx}.csv"
                path = os.path.join(results_dir, file_name)

                row = read_first_row_csv(path)
                if row is None:
                    if not args.skip_missing:
                        print(f"[WARN] missing: {path}")
                    continue

                thr = to_float(row.get("throughput"))
                if thr is None:
                    print(f"[WARN] invalid throughput in: {path} (value={row.get('throughput')})")
                    continue

                # --- 박스플롯용 원자료 저장 ---
                box_rows.append({
                    "map": map_name,
                    "planner": planner,
                    "density": args.density,
                    "scen_idx": scen_idx,
                    "throughput": thr,
                })

                thr_list.append(thr)
                found += 1

            # --- 요약 저장(원하면) ---
            if found == 0:
                sum_rows.append({
                    "map": map_name,
                    "planner": planner,
                    "density": args.density,
                    "throughput_mean": "",
                    "throughput_min": "",
                    "throughput_max": "",
                    "n_files": 0,
                })
            else:
                sum_rows.append({
                    "map": map_name,
                    "planner": planner,
                    "density": args.density,
                    "throughput_mean": f"{statistics.mean(thr_list):.6f}",
                    "throughput_min": f"{min(thr_list):.6f}",
                    "throughput_max": f"{max(thr_list):.6f}",
                    "n_files": found,
                })

                print(f"[{map_name} | {planner}] density={args.density} n={found} "
                      f"mean/min/max={statistics.mean(thr_list):.6f}/{min(thr_list):.6f}/{max(thr_list):.6f}")

    # 저장
    write_csv(
        args.out_box,
        fieldnames=["map", "planner", "density", "scen_idx", "throughput"],
        rows=box_rows,
    )
    write_csv(
        args.out_sum,
        fieldnames=["map", "planner", "density", "throughput_mean", "throughput_min", "throughput_max", "n_files"],
        rows=sum_rows,
    )

    print(f"\nSaved boxplot data: {args.out_box}")
    print(f"Saved summary data : {args.out_sum}")


if __name__ == "__main__":
    main()
