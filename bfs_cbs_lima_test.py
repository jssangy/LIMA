import os
import sys
import csv
import time
import argparse


def _ensure_fixed_hash_seed(seed: str = "0"):
    if os.environ.get("PYTHONHASHSEED") == seed:
        return
    os.environ["PYTHONHASHSEED"] = seed
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_fixed_hash_seed("0")

from Environment import ENV  # noqa: E402


MAPS = ["cross-30-30", "warehouse-10-20", "warehouse-20-40"]
PLANNERS = ["bfs", "cbs"]
SCEN_IDXS = list(range(10))


def run_one(map_name: str, planner: str, density: int, scen_idx: int, args) -> dict:
    map_path = f"assets/{map_name}/{map_name}.map"
    scen_path = f"assets/{map_name}/scen/{map_name}_s{scen_idx}.scen"

    env = None
    t0 = time.time()
    try:
        env = ENV(
            map_path,
            density=density,
            num_amrs=0,
            max_steps=args.max_steps,
            planner=planner,
            workers=args.workers,
            cache_db_path=args.cache_db_path,
            task_mode="scen",
            scen_path=scen_path,
            seed=args.seed,
        )

        env.reset()

        last_info = None
        while True:
            ret = env.step()
            if ret is False:
                break
            last_info = ret

        info = last_info if last_info is not None else env.make_info()
        wall_sec = time.time() - t0

        # 요청한 metric 2개만 핵심으로 사용
        sr = float(info.get("success_rate", 0.0))
        th = float(info.get("throughput", 0.0))

        return {
            "map": map_name,
            "planner": planner,
            "density": density,
            "scen_idx": scen_idx,
            "success_rate": sr,
            "throughput": th,
            # 디버깅/추적용(원치 않으면 저장/출력에서 제외해도 됨)
            "time": info.get("time", 0),
            "wall_sec": wall_sec,
        }

    finally:
        if env is not None:
            try:
                env.scheduler_pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                env.scheduler_pool.shutdown(wait=True)


def save_per_scenario_csv(out_path: str, row: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = ["success_rate", "throughput"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def save_table_csv(out_path: str, rows: list, fields: list):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--cache-db-path", type=str, default="./cache/cache.sqlite")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--density", type=int, default=10)  # ✅ 고정하고 싶으면 여기만 10으로 두면 됨
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="bfs_cbs_density10_summary")
    args = parser.parse_args()

    density = args.density  # 요청은 10

    per_rows = []
    summary_rows = []

    total = len(MAPS) * len(PLANNERS) * len(SCEN_IDXS)
    done = 0

    for map_name in MAPS:
        for planner in PLANNERS:
            # 너가 쓰던 스타일 그대로 results_{planner} 폴더에 저장
            results_dir = f"assets/{map_name}/results_lima_{planner}"

            sr_list = []
            th_list = []

            for scen_idx in SCEN_IDXS:
                out_path = f"{results_dir}/{map_name}_{density}_s{scen_idx}.csv"

                if args.skip_existing and os.path.exists(out_path):
                    done += 1
                    print(f"[SKIP] ({done}/{total}) {out_path}")
                    # 스킵한 경우 요약에 포함시키려면 파일을 읽어야 하는데,
                    # 여기서는 단순히 "실행 스킵"만 하고 요약은 이번 실행분으로만 만들게.
                    continue

                print(f"[RUN ] ({done+1}/{total}) map={map_name} planner={planner} scen={scen_idx} density={density}", flush=True)
                row = run_one(map_name, planner, density, scen_idx, args)

                # per-scenario 파일 저장 (success_rate, throughput만)
                save_per_scenario_csv(out_path, row)

                per_rows.append(row)
                sr_list.append(row["success_rate"])
                th_list.append(row["throughput"])

                done += 1
                print(
                    f"[DONE] ({done}/{total}) -> {out_path} | "
                    f"SR={row['success_rate']:.3f} TH={row['throughput']:.3f}",
                    flush=True,
                )

            # map×planner 요약(이번 실행에서 실제로 돌린 것만)
            sr_stat = summarize(sr_list)
            th_stat = summarize(th_list)

            summary_rows.append({
                "map": map_name,
                "planner": planner,
                "density": density,
                "n_scenarios_ran": len(sr_list),
                "success_rate_mean": sr_stat["mean"],
                "success_rate_min": sr_stat["min"],
                "success_rate_max": sr_stat["max"],
                "throughput_mean": th_stat["mean"],
                "throughput_min": th_stat["min"],
                "throughput_max": th_stat["max"],
            })

            print(
                f"[SUMMARY] map={map_name} planner={planner} density={density} "
                f"SR(mean/min/max)={sr_stat['mean']:.3f}/{sr_stat['min']:.3f}/{sr_stat['max']:.3f} "
                f"TH(mean/min/max)={th_stat['mean']:.3f}/{th_stat['min']:.3f}/{th_stat['max']:.3f}",
                flush=True,
            )

    # 전체 요약 CSV 저장
    save_table_csv(
        os.path.join(args.outdir, "summary.csv"),
        summary_rows,
        fields=[
            "map", "planner", "density", "n_scenarios_ran",
            "success_rate_mean", "success_rate_min", "success_rate_max",
            "throughput_mean", "throughput_min", "throughput_max",
        ],
    )

    # per-scenario도 한 파일로 모아서 저장(원하면)
    save_table_csv(
        os.path.join(args.outdir, "per_scenario.csv"),
        per_rows,
        fields=["map", "planner", "density", "scen_idx", "success_rate", "throughput", "time", "wall_sec"],
    )

    print(f"\nSaved summary to: {args.outdir}/summary.csv")
    print(f"Saved per-scenario to: {args.outdir}/per_scenario.csv")


if __name__ == "__main__":
    main()
