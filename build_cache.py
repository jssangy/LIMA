import argparse
import os
import time
import multiprocessing as mp
import queue as pyqueue

from Environment import ENV
from utils.schedule_cache import CacheWriter


class QueueCacheWriter:
    """워커(ENV 프로세스)는 DB에 직접 쓰지 않고 Queue로 (key, blob)만 전달한다."""
    def __init__(self, q: mp.Queue):
        self.q = q

    def put_blob(self, key: bytes, blob: bytes) -> None:
        self.q.put((key, blob), block=True)


def writer_loop(
    db_path: str,
    q: mp.Queue,
    stats_every: int = 200_000,
    batch_size: int = 500,
    flush_seconds: float = 1.0,
    dedup_in_batch: bool = True,
    insert_mode: str = "ignore",  # "ignore" or "replace"
):
    """
    DB write 전담 프로세스.
    - batch_size개 모이면 executemany + commit
    - flush_seconds마다 강제 flush (너무 오래 DB 반영 지연되는 것 방지)
    - dedup_in_batch=True면 배치 내부 중복 key는 마지막 것만 남김(쓰기량↓)
    """
    writer = CacheWriter(db_path)

    if insert_mode == "replace":
        sql = "INSERT OR REPLACE INTO cache(k,v) VALUES(?,?)"
    else:
        # 캐시 빌드에서는 동일 key 재삽입 의미가 거의 없으므로 ignore가 보통 더 빠름
        sql = "INSERT OR IGNORE INTO cache(k,v) VALUES(?,?)"

    written = 0
    t0 = time.time()
    last_flush = time.time()

    batch_map = {}   # key -> blob (dedup용)
    batch_list = []  # [(key, blob), ...] (dedup 안 할 때)

    def batch_len() -> int:
        return len(batch_map) if dedup_in_batch else len(batch_list)

    def add_item(key: bytes, blob: bytes):
        if dedup_in_batch:
            batch_map[key] = blob
        else:
            batch_list.append((key, blob))

    def flush():
        nonlocal written, last_flush
        if batch_len() == 0:
            last_flush = time.time()
            return

        rows = list(batch_map.items()) if dedup_in_batch else batch_list
        writer.conn.executemany(sql, rows)
        writer.conn.commit()

        written += len(rows)
        batch_map.clear()
        batch_list.clear()
        last_flush = time.time()

        if stats_every and written % stats_every == 0:
            dt = last_flush - t0
            rate = written / dt if dt > 0 else 0.0
            print(f"[DB] writes={written:,} ({rate:.1f}/s)")

    while True:
        try:
            item = q.get(timeout=max(0.05, float(flush_seconds)))
        except pyqueue.Empty:
            if (time.time() - last_flush) >= flush_seconds:
                flush()
            continue

        if item is None:  # sentinel
            flush()
            break

        key, blob = item
        add_item(key, blob)

        now = time.time()
        if batch_len() >= batch_size or (now - last_flush) >= flush_seconds:
            flush()

    writer.conn.close()
    print(f"[DB] writer exit. total_writes={written:,}")


def run_env_episodes(
    worker_id: int,
    episodes: int,
    prob_path: str,
    density: int,
    num_amrs: int,
    max_steps: int,
    env_workers: int,
    cache_db_path: str,
    q: mp.Queue,
    log_every: int,
):
    """
    워커 프로세스:
    - ENV를 episodes번 생성 → step() 끝날 때까지 돌림 → 다음 ENV 생성 반복
    - DB write는 하지 않고 QueueCacheWriter로 writeback만 전달
    """
    queue_writer = QueueCacheWriter(q)

    for ep in range(episodes):
        env = ENV(
            prob_path,
            density=density,
            num_amrs=num_amrs,
            max_steps=max_steps,
            workers=env_workers,  # 교차로 1개면 1 권장
            cache_db_path=cache_db_path,
        )

        # ENV가 writeback을 put_blob()로 내보내도록 연결
        env.cache_writer = queue_writer

        # Intersection에서 cache_db_path를 직접 읽는 구조면 주입(필요 없으면 무시됨)
        if hasattr(env, "intersections"):
            for I in env.intersections.values():
                setattr(I, "cache_db_path", cache_db_path)

        env.reset()

        while True:
            run = env.step()
            if not run:
                break

        if log_every and (ep + 1) % log_every == 0:
            print(f"[Worker {worker_id}] num_amrs={num_amrs} episodes {ep+1:,}/{episodes:,} done")

    print(f"[Worker {worker_id}] done. num_amrs={num_amrs}, episodes={episodes:,}")


def main():
    parser = argparse.ArgumentParser("Build schedule cache without GUI (multi num_amrs, batched writer)")

    parser.add_argument("--prob", type=str, default="warehouse_1",
                        help="problems/cross/{prob}.json 를 사용")
    parser.add_argument("--density", type=int, default=10,
                        help="num_amrs가 0일 때만 의미 있음(ENV가 density로 계산하는 경우 대비)")
    parser.add_argument("--num-amrs", type=int, nargs="+",
                        default=[13, 14],
                        help="예: --num-amrs 13 14")

    # episodes는 '각 num_amrs 그룹당' 총 에피소드 수
    parser.add_argument("--episodes", type=int, default=10_000_000)

    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--env-workers", type=int, default=1)

    # 바깥쪽 병렬 ENV 러너 총 개수
    parser.add_argument("--procs", type=int, default=8)

    parser.add_argument("--cache-db", type=str, default="./cache/schedule_cache.sqlite")
    parser.add_argument("--queue-max", type=int, default=100)

    # writer 배치 설정
    parser.add_argument("--db-stats-every", type=int, default=1000)
    parser.add_argument("--db-batch-size", type=int, default=50)
    parser.add_argument("--db-flush-seconds", type=float, default=0.2)
    parser.add_argument("--db-dedup", action="store_true")
    parser.add_argument("--db-insert-mode", type=str, default="ignore", choices=["ignore", "replace"])

    # 워커 로그
    parser.add_argument("--log-every", type=int, default=100)

    args = parser.parse_args()

    num_list = list(args.num_amrs)
    groups = len(num_list)
    if groups == 0:
        raise ValueError("--num-amrs must have at least one value")

    if args.procs < groups:
        raise ValueError(f"--procs({args.procs}) must be >= len(num_amrs)({groups})")

    # problem path 결정
    prob_path = f"problems/cross/{args.prob}.json"

    # 0) DB 파일 미리 생성(워커가 mode=ro로 CacheReader를 열 수 있게)
    os.makedirs(os.path.dirname(args.cache_db) or ".", exist_ok=True)
    tmp = CacheWriter(args.cache_db)
    tmp.conn.close()

    # seed를 안 쓰는 대신, fork를 쓰면 RNG 상태가 복제될 수 있음 → spawn 고정
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=args.queue_max)

    # 1) writer 프로세스(DB write 전담)
    writer_p = ctx.Process(
        target=writer_loop,
        args=(
            args.cache_db,
            q,
            args.db_stats_every,
            args.db_batch_size,
            args.db_flush_seconds,
            args.db_dedup,
            args.db_insert_mode,
        ),
        daemon=False,
    )
    writer_p.start()

    # 2) num_amrs별로 procs 분배
    base_procs = args.procs // groups
    rem_procs = args.procs % groups

    print(f"[MAIN] num_amrs={num_list}")
    print(f"[MAIN] total_procs={args.procs}, groups={groups}, base_procs/group={base_procs}, remainder={rem_procs}")
    print(f"[MAIN] episodes_per_group={args.episodes:,}")
    print(f"[MAIN] env_workers={args.env_workers}, max_steps={args.max_steps}")

    workers = []
    wid_counter = 0

    try:
        for gi, amrs in enumerate(num_list):
            k = base_procs + (1 if gi < rem_procs else 0)  # 이 num_amrs에 할당되는 프로세스 수
            if k <= 0:
                continue

            # 그룹당 episodes를 k개 프로세스에 나눔
            base_eps = args.episodes // k
            rem_eps = args.episodes % k

            print(f"[MAIN] num_amrs={amrs}: procs={k}, episodes={args.episodes:,} (each ~{base_eps:,})")

            for local in range(k):
                eps = base_eps + (1 if local < rem_eps else 0)
                if eps <= 0:
                    continue

                p = ctx.Process(
                    target=run_env_episodes,
                    args=(
                        wid_counter,
                        eps,
                        prob_path,
                        args.density,
                        amrs,
                        args.max_steps,
                        args.env_workers,
                        args.cache_db,
                        q,
                        args.log_every,
                    ),
                    daemon=False,
                )
                p.start()
                workers.append(p)
                wid_counter += 1

        for p in workers:
            p.join()

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt: terminating workers...")
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join()

    finally:
        # 3) writer 종료
        q.put(None)
        writer_p.join()
        print("[DONE] cache build finished.")


if __name__ == "__main__":
    main()
