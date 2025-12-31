from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Optional, Sequence, Union
import time

from utils.env import StackRearrangementEnv 
from utils.schedule_cache import CacheReader, make_cache_key, encode_actions, decode_actions

DEBUG = False
USE_DETERMINISTIC_MOVE = True

Move = Tuple[int, int]

_CACHE_READER = None
_CACHE_READER_PATH = None

try:
    import cpp_sch
    _HAS_CPP = True
except Exception:
    _HAS_CPP = False


def _get_cache_reader(db_path: str) -> CacheReader:
    global _CACHE_READER, _CACHE_READER_PATH
    if _CACHE_READER is None or _CACHE_READER_PATH != db_path:
        _CACHE_READER_PATH = db_path
        _CACHE_READER = CacheReader(db_path)  # Reader opened in mode=ro
    return _CACHE_READER


def _normalize_caps(stack_capacities: Union[int, Sequence[int]], n: int) -> List[int]:
    if isinstance(stack_capacities, int):
        return [int(stack_capacities)] * n
    caps = [int(x) for x in stack_capacities]
    if len(caps) != n:
        raise ValueError(f"stack_capacities len {len(caps)} != stacks len {n}")
    if any(c < 0 for c in caps):
        raise ValueError(f"stack_capacities must be >=0: {caps}")
    return caps


class StackScheduler:
    """Generates a schedule (move sequence) by receiving only the initial stack state."""

    def __init__(
        self,
        initial_stacks: List[List[int]],
        stack_capacities: Union[int, Sequence[int]],
        seed: Optional[int] = None,
    ) -> None:
        n = len(initial_stacks)
        caps = _normalize_caps(stack_capacities, n)

        # env uses the per-stack capacity as is
        self.env = StackRearrangementEnv(
            stacks=deepcopy(initial_stacks),
            stack_capacities=caps,
        )
        self.caps = caps
        self.moves: List[Move] = []

    # =========================================================================
    # H2 (IDA*)
    # =========================================================================
    def solve_h2(self, max_iters=100_000, use_deterministic_move=True):
        if _HAS_CPP:
            # If based on self.caps (per-stack capacity list), pass it as is,
            # if it's still a single cap, pass it as [cap]*n
            caps = getattr(self, "caps", None)
            if caps is None:
                # For old version (single cap)
                caps = [self.env.stack_capacity] * len(self.env.stacks)
            return cpp_sch.solve_h2_base(self.env.stacks, caps, max_iters, use_deterministic_move)

        else:
            raise RuntimeError("C++ extension for H2 solver is not available.")

def schedule(
    initial_stacks: List[List[int]],
    stack_capacities: Union[int, Sequence[int]],
    per_stack_quota: Optional[Sequence[int]] = None,
    order: Optional[Sequence[int]] = None,
    *,
    cache_db_path: Optional[str] = None,
    max_iters: int = 1_000_000,
    **kwargs,
):
    """
    - Cache key: uses only (initial_stacks, stack_capacities)
    - Cache value: stores/loads only base_moves created by solve_h2()
    - per_stack_quota is not cached, applied only as post-processing every time (lightweight)

    Returns:
        (final_moves, elapsed_time, cache_writeback)
        cache_writeback: None or (key_bytes, blob_bytes)  # put only in main
    """
    n = len(initial_stacks)
    caps = _normalize_caps(stack_capacities, n)

    if order is None:
        order = list(range(n))
    else:
        order = list(order)
        if len(order) != n:
            raise ValueError(f"order len {len(order)} != stacks len {n}")

    start_time = time.time()

    base_moves = None
    writeback = None

    # 1) Cache read-only lookup
    key = None
    cache_hit = False
    if cache_db_path:
        key = make_cache_key(initial_stacks=initial_stacks, stack_capacities=caps)
        reader = _get_cache_reader(cache_db_path)
        blob = reader.get_blob(key)  # bytes or None
        if blob is not None:
            base_moves = decode_actions(blob)
            cache_hit = True

    # 2) If miss, execute solve_h2() (most expensive part)
    if base_moves is None:
        scheduler = StackScheduler(initial_stacks, stack_capacities=caps)
        base_moves = scheduler.solve_h2(max_iters=max_iters, **kwargs)
        cache_hit = False

        if cache_db_path and key is not None:
            writeback = (key, encode_actions(base_moves))  # Main saves it

    # 3) Post-processing (lightweight) — same as existing schedule()
    moves = append_explicit_overflow_moves(
        initial_stacks=initial_stacks,
        base_moves=base_moves,
        stack_capacities=caps,
        order=order,
    )

    if per_stack_quota is not None:
        moves = append_overflow_moves(
            initial_stacks=initial_stacks,
            moves=moves,
            per_stack_quota=list(per_stack_quota),
            stack_capacities=caps,
            order=order,
        )

    elapsed_time = time.time() - start_time
    return moves, elapsed_time, writeback, cache_hit


def append_explicit_overflow_moves(
    initial_stacks: List[List[int]],
    base_moves: List[Move],
    stack_capacities: Sequence[int],
    order: Sequence[int],
    debug: bool = False,
) -> List[Move]:
    n = len(initial_stacks)
    caps = _normalize_caps(stack_capacities, n)
    order = list(order)

    stacks = deepcopy(initial_stacks)

    # 1) Apply base_moves (check cap[dst])
    for (src, dst) in base_moves:
        if src == dst:
            continue
        if not stacks[src]:
            raise RuntimeError(f"Invalid base_moves: pop empty src={src}")
        if len(stacks[dst]) >= caps[dst]:
            raise RuntimeError(f"Invalid base_moves: dst full dst={dst} cap={caps[dst]}")

        ball = stacks[src].pop()
        stacks[dst].append(ball)

    # 2) Number of types (need)
    counts = Counter(item for st in stacks for item in st)
    need = [counts.get(i, 0) for i in range(n)]

    overflow_types = {i for i in range(n) if need[i] > caps[i]}
    if not overflow_types:
        return base_moves

    # 3) Calculate target_len (based on cap_i)
    lens = [0] * n
    for i in range(n):
        lens[i] = caps[i] if i in overflow_types else need[i]

    for t in range(n):
        if t not in overflow_types:
            continue
        extra = need[t] - caps[t]
        for _ in range(extra):
            cands = [j for j in range(n) if j != t and j not in overflow_types and lens[j] < caps[j]]
            if not cands:
                # If no more distribution is possible, return base_moves as is
                return base_moves
            min_len = min(lens[j] for j in cands)
            dst = next(j for j in order if j in cands and lens[j] == min_len)
            lens[dst] += 1

    target_len = lens

    # 4) Match current length to target_len (only for non-overflow stacks)
    extra_moves: List[Move] = []
    step = 0
    while True:
        surplus = [i for i in order if i not in overflow_types and len(stacks[i]) > target_len[i]]
        deficit = [i for i in order if i not in overflow_types and len(stacks[i]) < target_len[i]]
        if not surplus or not deficit:
            break

        max_over = max(len(stacks[i]) - target_len[i] for i in surplus)
        src = next(i for i in order if i in surplus and (len(stacks[i]) - target_len[i]) == max_over)

        min_len = min(len(stacks[i]) for i in deficit)
        dst = next(i for i in order if i in deficit and len(stacks[i]) == min_len)

        top = stacks[src][-1] if stacks[src] else None
        if (not stacks[src]) or (top not in overflow_types):
            # Since we don't do "digging", give up here
            return base_moves

        if len(stacks[dst]) >= caps[dst]:
            return base_moves

        ball = stacks[src].pop()
        stacks[dst].append(ball)
        extra_moves.append((src, dst))

        step += 1
        if step > 2000:
            break

    return base_moves + extra_moves


def append_overflow_moves(
    initial_stacks: List[List[int]],
    moves: List[Move],
    per_stack_quota: List[int],
    stack_capacities: Sequence[int],
    order: Sequence[int],
    debug: bool = False,
) -> List[Move]:
    n = len(initial_stacks)
    caps = _normalize_caps(stack_capacities, n)
    order = list(order)

    if len(per_stack_quota) != n:
        raise ValueError(f"per_stack_quota len {len(per_stack_quota)} != stacks len {n}")

    # Clamp quota so it doesn't exceed cap
    quota = [max(0, min(int(per_stack_quota[i]), caps[i])) for i in range(n)]

    stacks = deepcopy(initial_stacks)

    # 1) Apply moves dry-run (check cap[dst])
    for (src, dst) in moves:
        if src == dst:
            continue
        if not stacks[src]:
            continue
        if len(stacks[dst]) >= caps[dst]:
            continue
        ball = stacks[src].pop()
        stacks[dst].append(ball)

    # 2) Move quota excess to locations with large slack
    extra: List[Move] = []
    step = 0

    while True:
        overflow_list = [(i, len(stacks[i]) - quota[i]) for i in range(n) if len(stacks[i]) > quota[i]]
        if not overflow_list:
            break

        max_over = max(k for _, k in overflow_list)
        over_srcs = [i for i, k in overflow_list if k == max_over]
        src = next(i for i in order if i in over_srcs)

        candidates = [
            j for j in order
            if j != src and len(stacks[j]) < caps[j] and len(stacks[j]) < quota[j]
        ]
        if not candidates:
            break

        def slack(j: int):
            return (quota[j] - len(stacks[j]), caps[j] - len(stacks[j]))

        best_sl = max(slack(j) for j in candidates)
        best = [j for j in candidates if slack(j) == best_sl]
        dst = next(j for j in order if j in best)

        if not stacks[src]:
            break
        if len(stacks[dst]) >= caps[dst]:
            break

        ball = stacks[src].pop()
        stacks[dst].append(ball)
        extra.append((src, dst))

        step += 1
        if step > 4000:
            break

    return moves + extra
