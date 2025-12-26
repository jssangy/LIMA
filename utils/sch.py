from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Sequence, Union
import random
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
        _CACHE_READER = CacheReader(db_path)  # mode=ro로 열리는 Reader
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
    """초기 스택 상태만 받아 스케줄(move 시퀀스)을 생성."""

    def __init__(
        self,
        initial_stacks: List[List[int]],
        stack_capacities: Union[int, Sequence[int]],
        seed: Optional[int] = None,
    ) -> None:
        n = len(initial_stacks)
        caps = _normalize_caps(stack_capacities, n)

        # env는 스택별 cap을 그대로 사용
        self.env = StackRearrangementEnv(
            stacks=deepcopy(initial_stacks),
            stack_capacities=caps,
        )
        self.caps = caps
        self.moves: List[Move] = []
        self.rng = random.Random(seed)

    # =========================================================================
    # Random
    # =========================================================================
    def solve_random(self, max_iters: int = 100_000) -> List[Move]:
        iterations = 0
        while not self.env.is_solved() and iterations < max_iters:
            iterations += 1
            sources = [i for i, st in enumerate(self.env.stacks) if st]
            if not sources:
                break
            src = self.rng.choice(sources)
            dest_candidates = [
                i for i in range(self.env.num_stacks)
                if i != src and len(self.env.stacks[i]) < self.caps[i]
            ]
            if not dest_candidates:
                continue
            dst = self.rng.choice(dest_candidates)
            self._move(src, dst)

        if not self.env.is_solved():
            raise RuntimeError("무작위 이동으로 해를 찾지 못했습니다.")
        return self.moves

    # =========================================================================
    # H2 (IDA*)
    # =========================================================================
    def solve_h2(self, max_iters=100_000, use_deterministic_move=True):
        if _HAS_CPP:
            # self.caps(스택별 cap 리스트) 기반이면 그대로 넘기고,
            # 아직 단일 cap이면 [cap]*n 형태로 만들어서 넘기면 됨
            caps = getattr(self, "caps", None)
            if caps is None:
                # 구버전(단일 cap)일 때
                caps = [self.env.stack_capacity] * len(self.env.stacks)
            return cpp_sch.solve_h2_base(self.env.stacks, caps, max_iters, use_deterministic_move)

        # fallback: 기존 파이썬 구현
        return self._solve_h2_python(max_iters=max_iters, use_deterministic_move=use_deterministic_move)
    

    def _solve_h2_python(
        self,
        max_iters: int = 100_000,
        use_deterministic_move: bool = USE_DETERMINISTIC_MOVE,
    ) -> List[Move]:
        start_state = tuple(tuple(s) for s in self.env.stacks)
        initial_stacks = deepcopy(self.env.stacks)

        threshold = self._get_lower_bound_h2(start_state)
        initial_threshold = threshold

        path = [{"state": start_state, "move": None}]
        iteration = 0

        while iteration < max_iters:
            iteration += 1
            visited_costs: Dict[tuple, int] = {}

            res_f, res_path = self._dfs_h2(path, 0, threshold, visited_costs, use_deterministic_move)

            if res_path is not None:
                moves = [step["move"] for step in res_path[1:]]
                self.moves = moves
                return moves

            if res_f == float("inf"):
                if use_deterministic_move:
                    use_deterministic_move = False
                    threshold = initial_threshold
                    path = [{"state": start_state, "move": None}]
                    iteration = 0
                    continue
                else:
                    # 디버그 출력 최소화: 필요하면 DEBUG로
                    if DEBUG:
                        print("H2: 해를 찾을 수 없습니다.")
                        print("\n초기 상태:")
                        original_stacks = self.env.stacks
                        self.env.stacks = initial_stacks
                        self.env.visualize()
                        self.env.stacks = original_stacks
                    raise RuntimeError("H2: 해를 찾을 수 없습니다.")

            threshold = max(res_f, threshold + 3.0)

        raise RuntimeError("H2: 최대 반복 횟수 내에 해를 찾지 못했습니다.")

    def _move(self, src: int, dst: int) -> bool:
        if self.env.move(src, dst):
            self.moves.append((src, dst))
            return True
        return False

    def _is_sorted_h2(self, state) -> bool:
        # (1) 기본: 각 스택이 자기 색만
        if all(all(item == sid for item in st) for sid, st in enumerate(state)):
            return True

        # (2) overflow: "색 i의 총 개수 > cap[i]"
        counts = Counter(item for st in state for item in st)
        overflow_types = set()
        for t, total in counts.items():
            if 0 <= t < self.env.num_stacks and total > self.caps[t]:
                overflow_types.add(t)

        if overflow_types:
            return self._is_valid_with_overflow_h2(state, overflow_types)

        return False

    def _is_valid_with_overflow_h2(self, state, overflow_types: set[int]) -> bool:
        for sid, st in enumerate(state):
            cap_sid = self.caps[sid]
            if len(st) > cap_sid:
                return False

            if sid in overflow_types:
                # overflow 스택은 길이=cap[sid] & 자기 색만
                if len(st) != cap_sid:
                    return False
                if not all(x == sid for x in st):
                    return False
            else:
                # 아래: 자기색, 위: overflow 색들만
                idx = 0
                while idx < len(st) and st[idx] == sid:
                    idx += 1
                while idx < len(st) and st[idx] in overflow_types:
                    idx += 1
                if idx != len(st):
                    return False
        return True

    def _get_lower_bound_h2(self, state) -> float:
        score = 0.0
        for sid, st in enumerate(state):
            cap_sid = self.caps[sid]
            is_clean = True
            for depth, ball in enumerate(st):  # depth: 0(bottom) ...
                if is_clean:
                    if ball != sid:
                        is_clean = False
                        score += (cap_sid - depth) * 1.5
                else:
                    score += 1
        return score

    def _get_valid_moves_h2(
        self,
        state,
        last_src: int,
        last_dest: int,
        use_deterministic_move: bool = False,
    ) -> List[Move]:
        n = self.env.num_stacks

        # (옵션) 결정적 이동
        if use_deterministic_move:
            for src in range(n):
                if not state[src]:
                    continue
                ball = state[src][-1]
                target = ball  # ball 자체가 목표 스택 인덱스라고 가정
                if not (0 <= target < n):
                    continue

                if src != target and len(state[target]) < self.caps[target]:
                    is_target_ready = all(b == target for b in state[target])
                    if is_target_ready:
                        return [(src, target)]

        moves: List[Move] = []
        for src in range(n):
            if not state[src]:
                continue
            for dst in range(n):
                if src == dst:
                    continue
                if len(state[dst]) >= self.caps[dst]:
                    continue
                if src == last_dest and dst == last_src:
                    continue
                moves.append((src, dst))
        return moves

    def _apply_move_h2(self, state, move: Move):
        src, dst = move
        src_stack = state[src]
        dst_stack = state[dst]
        ball = src_stack[-1]
        new_src_stack = src_stack[:-1]
        new_dst_stack = dst_stack + (ball,)
        new_state = list(state)
        new_state[src] = new_src_stack
        new_state[dst] = new_dst_stack
        return tuple(new_state)

    def _dfs_h2(
        self,
        path: List[dict],
        g: int,
        threshold: float,
        visited_costs: Dict[tuple, int],
        use_deterministic_move: bool = False,
    ):
        current_state = path[-1]["state"]

        if current_state in visited_costs and visited_costs[current_state] <= g:
            return float("inf"), None
        visited_costs[current_state] = g

        f = g + self._get_lower_bound_h2(current_state)
        if f > threshold:
            return f, None

        if self._is_sorted_h2(current_state):
            return f, path

        min_surplus = float("inf")

        last_src, last_dest = -1, -1
        if len(path) > 1:
            last_src, last_dest = path[-1]["move"]

        valid_moves = self._get_valid_moves_h2(current_state, last_src, last_dest, use_deterministic_move)

        for move in valid_moves:
            next_state = self._apply_move_h2(current_state, move)
            path.append({"state": next_state, "move": move})

            res_f, res_path = self._dfs_h2(path, g + 1, threshold, visited_costs, use_deterministic_move)
            if res_path is not None:
                return res_f, res_path
            min_surplus = min(min_surplus, res_f)

            path.pop()

        return min_surplus, None


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
    - 캐시 key: (initial_stacks, stack_capacities)만 사용
    - 캐시 value: solve_h2()가 만든 base_moves만 저장/로드
    - per_stack_quota는 캐시에 안 넣고, 매번 후처리로만 적용(가벼움)

    Returns:
        (final_moves, elapsed_time, cache_writeback)
        cache_writeback: None 또는 (key_bytes, blob_bytes)  # 메인에서만 put
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

    # 1) 캐시 read-only 조회
    key = None
    cache_hit = False
    if cache_db_path:
        key = make_cache_key(initial_stacks=initial_stacks, stack_capacities=caps)
        reader = _get_cache_reader(cache_db_path)
        blob = reader.get_blob(key)  # bytes or None
        if blob is not None:
            base_moves = decode_actions(blob)
            cache_hit = True

    # 2) miss면 solve_h2() 실행 (가장 비싼 부분)
    if base_moves is None:
        scheduler = StackScheduler(initial_stacks, stack_capacities=caps)
        base_moves = scheduler.solve_h2(max_iters=max_iters, **kwargs)
        cache_hit = False

        if cache_db_path and key is not None:
            writeback = (key, encode_actions(base_moves))  # 메인이 저장

    # 3) 후처리(가벼움) — 기존 schedule()과 동일
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

    # 1) base_moves 적용 (cap[dst] 검사)
    for (src, dst) in base_moves:
        if src == dst:
            continue
        if not stacks[src]:
            raise RuntimeError(f"Invalid base_moves: pop empty src={src}")
        if len(stacks[dst]) >= caps[dst]:
            raise RuntimeError(f"Invalid base_moves: dst full dst={dst} cap={caps[dst]}")

        ball = stacks[src].pop()
        stacks[dst].append(ball)

    # 2) 타입 개수(need)
    counts = Counter(item for st in stacks for item in st)
    need = [counts.get(i, 0) for i in range(n)]

    overflow_types = {i for i in range(n) if need[i] > caps[i]}
    if not overflow_types:
        return base_moves

    # 3) target_len 계산 (cap_i 기반)
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
                # 더 이상 분배 불가면 base_moves 그대로
                return base_moves
            min_len = min(lens[j] for j in cands)
            dst = next(j for j in order if j in cands and lens[j] == min_len)
            lens[dst] += 1

    target_len = lens

    # 4) 현재 길이를 target_len으로 맞추기 (비-overflow 스택들만)
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
            # “digging”은 안 하므로 여기서 포기
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

    # quota는 cap을 넘지 않게 clamp
    quota = [max(0, min(int(per_stack_quota[i]), caps[i])) for i in range(n)]

    stacks = deepcopy(initial_stacks)

    # 1) moves 드라이런 적용 (cap[dst] 검사)
    for (src, dst) in moves:
        if src == dst:
            continue
        if not stacks[src]:
            continue
        if len(stacks[dst]) >= caps[dst]:
            continue
        ball = stacks[src].pop()
        stacks[dst].append(ball)

    # 2) quota 초과분을 slack 큰 곳으로 이동
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
