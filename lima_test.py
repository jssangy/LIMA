import os
import sys
import csv
import time
import argparse


def _ensure_fixed_hash_seed(seed: str = "0"):
    # 이미 고정돼 있으면 그대로 진행
    if os.environ.get("PYTHONHASHSEED") == seed:
        return
    # 환경변수 세팅 후, 같은 파이썬으로 프로세스 재시작
    os.environ["PYTHONHASHSEED"] = seed
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_fixed_hash_seed("0")

from Environment import ENV  # noqa: E402


MAPS = ["warehouse-10-20", "warehouse-20-40"]
DENSITIES = [10]
SCEN_IDXS = list(range(10))
SEED = 7


def run_one(map_name: str, density: int, scen_idx: int, args) -> dict:
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
            planner=args.planner,
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

        # step()가 마지막에 False로 끝나서 info가 없을 수 있으니 안전 처리
        info = last_info if last_info is not None else env.make_info()
        wall_sec = time.time() - t0

        return {
            "success_rate": info.get("success_rate", 0.0),
            "throughput": info.get("throughput", 0.0),
            "avg_path_integrity": info.get("avg_path_integrity", 0.0),
            "time": info.get("time", 0),
            # 필요하면 성능 확인용(요청엔 없지만 디버깅에 유용)
            "wall_sec": wall_sec,
        }

    finally:
        if env is not None:
            # 프로세스 풀 정리(안 하면 반복 실행 시 워커가 계속 남을 수 있음)
            try:
                env.scheduler_pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                env.scheduler_pool.shutdown(wait=True)


def save_csv(out_path: str, row: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 요청한 4개만 저장
    fields = ["success_rate", "throughput", "avg_path_integrity", "time"]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--cache-db-path", type=str, default="./cache/cache.sqlite")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--planner", choices=["bfs", "cbs"], default="bfs")
    args = parser.parse_args()

    total = len(MAPS) * len(DENSITIES) * len(SCEN_IDXS)
    done = 0

    for map_name in MAPS:
        results_dir = f"assets/{map_name}/results_lima"

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
                    f"SR={row['success_rate']:.3f} TH={row['throughput']:.3f} "
                    f"PI={row['avg_path_integrity']:.3f} T={row['time']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()