#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solve_h2() inference time benchmark

핵심:
- solve_h2() 호출 구간만 perf_counter로 측정 (cache 사용 X)
- 3개 환경
- 아이템 개수별 시나리오 10개 생성 (이미 solved면 제외)
- 10개 모이면 mean/min/max(ms) 기록

환경별 item 범위:
- 기본: --min-items / --max-items
- override: --env-range ENV_NAME=MIN:MAX (repeat 가능)

"너무 쉬운 랜덤" 방지 옵션:
- --type-sampling bounded|unbounded
    bounded  : (기본) 타입 카운트가 cap[type]를 넘지 않게 샘플링 (overflow 거의 없음 → 쉬움)
    unbounded: 타입을 균등 샘플링 (overflow 타입 생김 → 더 어려움)
- --placement-mode uniform|avoid-home
    uniform   : (기본) 그냥 빈 스택 중 랜덤 배치 (자기 스택에 들어갈 확률 큼 → 쉬움)
    avoid-home: 가능하면 item t를 stack t에 넣지 않음 (초기 정답 비율↓ → 더 어려움)

쉬운 케이스 필터:
- --min-misplaced N
- --min-misplaced-ratio R
  misplaced = "item 값 != 들어있는 stack_id" 개수
  조건: misplaced >= N AND misplaced >= ceil(R * n_items)

예시(확실히 어렵게):
python bench_solve_h2_time.py --outdir bench_h2 --max-iters 1000000 \
  --type-sampling unbounded --placement-mode avoid-home --min-misplaced-ratio 0.8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from copy import deepcopy
from typing import List, Sequence, Tuple, Dict, Any, Literal


TypeSampling = Literal["bounded", "unbounded"]
PlacementMode = Literal["uniform", "avoid-home"]


def _import_project_modules():
    # Prefer user's project layout: utils/{sch.py, env.py}
    try:
        from utils.sch import StackScheduler  # type: ignore
        from utils.env import StackRearrangementEnv  # type: ignore
        return StackScheduler, StackRearrangementEnv
    except Exception:
        # Fallback: same-folder imports
        from sch import StackScheduler  # type: ignore
        from env import StackRearrangementEnv  # type: ignore
        return StackScheduler, StackRearrangementEnv


def parse_env_ranges(env_range_args: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Parse repeated args like:
      --env-range S4_cap5-5-5-5=2:14
    into dict: { "S4_cap5-5-5-5": (2,14) }
    """
    out: Dict[str, Tuple[int, int]] = {}
    for s in env_range_args:
        if "=" not in s or ":" not in s:
            raise ValueError(f"Invalid --env-range '{s}'. Expected ENV=MIN:MAX")
        env, rest = s.split("=", 1)
        lo_s, hi_s = rest.split(":", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        if lo > hi:
            raise ValueError(f"Invalid --env-range '{s}': MIN > MAX")
        out[env] = (lo, hi)
    return out


def count_misplaced(stacks: Sequence[Sequence[int]]) -> int:
    """#items not in their destination stack (dest == item value)."""
    mis = 0
    for sid, st in enumerate(stacks):
        for item in st:
            if item != sid:
                mis += 1
    return mis


def gen_random_stacks(caps, n_items, rng):
    """
    완전 랜덤:
      - 각 아이템 색: uniform(0..n-1)
      - 각 아이템이 들어갈 스택: 전체 cap 슬롯에서 무작위로 뽑기
      - 스택 내부 순서도 랜덤
    """
    n = len(caps)
    total_cap = sum(caps)
    if n_items > total_cap:
        raise ValueError(f"n_items={n_items} > total capacity={total_cap}")

    # 1) 색(타입) 랜덤 -> 색깔 개수도 자동으로 랜덤
    items = [rng.randrange(n) for _ in range(n_items)]

    # 2) 배치 랜덤: 전체 슬롯에서 n_items개 선택
    slots = []
    for sid, cap in enumerate(caps):
        slots += [sid] * cap
    rng.shuffle(slots)
    chosen_slots = slots[:n_items]

    # 3) 스택에 채우기
    rng.shuffle(items)
    stacks = [[] for _ in range(n)]
    for sid, item in zip(chosen_slots, items):
        stacks[sid].append(item)

    # 4) 각 스택 내부 순서도 랜덤(리스트는 bottom->top)
    for sid in range(n):
        rng.shuffle(stacks[sid])

    return stacks



def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="bench_h2", help="output directory")

    # ✅ global default item range / env-range 인자 제거
    # parser.add_argument("--min-items", ...)
    # parser.add_argument("--max-items", ...)
    # parser.add_argument("--env-range", ...)

    # hardness filter
    parser.add_argument("--min-misplaced", type=int, default=0)
    parser.add_argument("--min-misplaced-ratio", type=float, default=0.0)

    # benchmark settings
    parser.add_argument("--scens-per-n", type=int, default=1000, help="number of scenarios per n_items")
    parser.add_argument("--max-iters", type=int, default=1_000_000)
    parser.add_argument("--deterministic", type=int, default=1, help="use deterministic move (0/1)")
    parser.add_argument("--max-attempts", type=int, default=200_000, help="limit to avoid infinite loops")
    args = parser.parse_args()

    StackScheduler, StackRearrangementEnv = _import_project_modules()

    # ✅ 여기서 환경별 item range를 직접 설정 (lo, hi)
    # 형식: (env_name, caps, min_items, max_items)
    env_specs: List[Tuple[str, List[int], int, int]] = [
        ("S4_cap2-2-10-10", [2, 2, 10, 10], 2, 14),  # <- 여기 범위 바꾸면 됨
        ("S3_cap2-10-10",   [2, 10, 10],     2, 12),  # <- 여기 범위 바꾸면 됨
        ("S4_cap5-5-5-5",  [5,5,5,5],       2, 15),
    ]

    rng = random.Random()
    deterministic = bool(args.deterministic)

    min_misplaced = max(0, int(args.min_misplaced))
    min_misplaced_ratio = float(args.min_misplaced_ratio)
    if not (0.0 <= min_misplaced_ratio <= 1.0):
        raise ValueError("--min-misplaced-ratio must be in [0.0, 1.0]")

    per_scen_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # ✅ env_specs에서 lo, hi를 바로 받음
    for env_name, caps, lo, hi in env_specs:
        n = len(caps)

        # 불가능한 범위 자동 보정(총 용량 초과 방지)
        total_cap = sum(caps)
        if hi > total_cap:
            hi = total_cap
        if lo < 0:
            lo = 0
        if lo > hi:
            raise ValueError(f"[{env_name}] invalid item range: lo={lo} > hi={hi}")

        for n_items in range(lo, hi + 1):
            times_sec: List[float] = []

            scen_idx = 0
            attempts = 0
            while scen_idx < args.scens_per_n:
                attempts += 1
                if attempts > args.max_attempts:
                    raise RuntimeError(
                        f"[{env_name}] n_items={n_items}: exceeded max attempts while sampling scenarios."
                    )

                init_stacks = gen_random_stacks(
                    caps=caps,
                    n_items=n_items,
                    rng=rng,
                )

                env = StackRearrangementEnv(stacks=deepcopy(init_stacks), stack_capacities=caps)
                if env.is_solved():
                    continue

                mis = count_misplaced(init_stacks)
                mis_need = max(min_misplaced, int(math.ceil(min_misplaced_ratio * n_items)))
                if mis < mis_need:
                    continue

                scheduler = StackScheduler(initial_stacks=init_stacks, stack_capacities=caps)
                t0 = time.perf_counter()
                try:
                    moves = scheduler.solve_h2(max_iters=args.max_iters, use_deterministic_move=deterministic)
                except Exception:
                    continue
                t1 = time.perf_counter()

                elapsed = t1 - t0
                times_sec.append(elapsed)

                per_scen_rows.append(
                    {
                        "env": env_name,
                        "caps": json.dumps(caps, ensure_ascii=False),
                        "n_stacks": n,
                        "n_items": n_items,
                        "scenario_idx": scen_idx,
                        "time_sec": f"{elapsed:.9f}",
                        "time_ms": f"{elapsed*1000.0:.3f}",
                        "misplaced": mis,
                        "moves_len": len(moves) if isinstance(moves, list) else "",
                        "initial_stacks": json.dumps(init_stacks, ensure_ascii=False),
                    }
                )
                scen_idx += 1

            mean_ms = statistics.mean(times_sec) * 1000.0
            min_ms = min(times_sec) * 1000.0
            max_ms = max(times_sec) * 1000.0

            summary_rows.append(
                {
                    "env": env_name,
                    "caps": json.dumps(caps, ensure_ascii=False),
                    "n_stacks": n,
                    "n_items": n_items,
                    "n_scenarios": len(times_sec),
                    "min_misplaced": min_misplaced,
                    "min_misplaced_ratio": f"{min_misplaced_ratio:.3f}",
                    "mean_ms": f"{mean_ms:.3f}",
                    "min_ms": f"{min_ms:.3f}",
                    "max_ms": f"{max_ms:.3f}",
                }
            )

            print(
                f"[{env_name}] n_items={n_items:2d} | mean={mean_ms:8.3f} ms"
                f"  min={min_ms:8.3f} ms  max={max_ms:8.3f} ms"
                f"  (mis>= {max(min_misplaced, int(math.ceil(min_misplaced_ratio * n_items)))})"
            )

    outdir = args.outdir
    per_csv = os.path.join(outdir, "solve_h2_per_scenario.csv")
    sum_csv = os.path.join(outdir, "solve_h2_summary.csv")

    write_csv(
        per_csv,
        fieldnames=[
            "env",
            "caps",
            "n_stacks",
            "n_items",
            "scenario_idx",
            "time_sec",
            "time_ms",
            "misplaced",
            "moves_len",
            "initial_stacks",
        ],
        rows=per_scen_rows,
    )
    write_csv(
        sum_csv,
        fieldnames=[
            "env",
            "caps",
            "n_stacks",
            "n_items",
            "n_scenarios",
            "type_sampling",
            "placement_mode",
            "min_misplaced",
            "min_misplaced_ratio",
            "mean_ms",
            "min_ms",
            "max_ms",
        ],
        rows=summary_rows,
    )

    print("\nSaved:")
    print(f"  - {per_csv}")
    print(f"  - {sum_csv}")


if __name__ == "__main__":
    main()
