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
DENSITIES = [1, 5, 10, 20, 30, 40, 50, 60]
SCEN_IDXS = list(range(10))   # ✅ FIX
SEED = 7


def run_one(map_name: str, density: int, scen_idx: int, args) -> dict:
    map_path = f"assets/{map_name}/{map_name}.map"
    scen_path = f"assets/{map_name}/scen/{map_name}_s{scen_idx}.scen"

    env = None
    try:
        env = ENV(
            map_path,
            density=density,
            num_amrs=0,
            max_steps=args.max_steps,
            planner="cbs",
            workers=args.workers,
            cache_db_path=args.cache_db_path,
            task_mode="scen",
            scen_path=scen_path,
            seed=SEED,
        )
        env.reset()

        last_info = None
        while True:
            ret = env.step()
            if ret is False:
                break
            last_info = ret

        info = last_info if last_info is not None else env.make_info()

        return {
            "success_rate": info.get("success_rate", 0.0),
            "time": info.get("time", 0),
        }

    finally:
        if env is not None:
            try:
                env.scheduler_pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                env.scheduler_pool.shutdown(wait=True)


def save_csv(out_path: str, row: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fields = ["success_rate", "time"]  # ✅ 원하는 2개만

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--cache-db-path", type=str, default="./cache/cache.sqlite")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    total = len(MAPS) * len(DENSITIES) * len(SCEN_IDXS)
    done = 0

    for map_name in MAPS:
        results_dir = f"assets/{map_name}/results_cbs"

        for scen_idx in SCEN_IDXS:
            for density in DENSITIES:
                out_path = f"{results_dir}/{map_name}_{density}_s{scen_idx}.csv"

                if args.skip_existing and os.path.exists(out_path):
                    done += 1
                    print(f"[SKIP] ({done}/{total}) {out_path}")
                    continue

                print(f"[RUN ] ({done+1}/{total}) map={map_name} scen={scen_idx} density={density}", flush=True)
                row = run_one(map_name, density, scen_idx, args)
                save_csv(out_path, row)

                done += 1
                print(
                    f"[DONE] ({done}/{total}) -> {out_path} | "
                    f"SR={row['success_rate']:.3f} T={row['time']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
