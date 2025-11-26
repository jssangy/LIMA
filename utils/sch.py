from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Optional, Dict
import random
import time

from utils.env import StackRearrangementEnv, NUM_STACKS, STACK_CAPACITY

# 디버깅 플래그: True일 때만 print 문이 출력됨
DEBUG = True

# 결정적 이동(Deterministic Move) 최적화 사용 여부 (기본값: True)
USE_DETERMINISTIC_MOVE = True


class StackScheduler:
    """초기 스택 상태만 받아 스케줄을 생성하는 간단한 스케줄러."""

    def __init__(
        self,
        initial_stacks: List[List[int]],
        stack_capacity: int = STACK_CAPACITY,
        seed: Optional[int] = None,
    ) -> None:
        num_stacks = len(initial_stacks)
        self.env = StackRearrangementEnv(
            num_stacks=num_stacks,
            stack_capacity=stack_capacity,
            stacks=deepcopy(initial_stacks),
        )
        self.moves: List[Tuple[int, int]] = []
        self.rng = random.Random(seed)

    def solve_random(self, max_iters: int = 100_000) -> List[Tuple[int, int]]:
        """무작위 move()를 반복하면서 목표 상태를 찾으려 시도한다."""
        iterations = 0
        while not self.env.is_solved() and iterations < max_iters:
            iterations += 1
            sources = [i for i, stack in enumerate(self.env.stacks) if stack]
            if not sources:
                break
            src = self.rng.choice(sources)
            dest_candidates = [
                i
                for i in range(self.env.num_stacks)
                if i != src and len(self.env.stacks[i]) < self.env.stack_capacity
            ]
            if not dest_candidates:
                continue
            dst = self.rng.choice(dest_candidates)
            self._move(src, dst)
        if not self.env.is_solved():
            raise RuntimeError("무작위 이동으로 해를 찾지 못했습니다.")
        return self.moves

    def solve_h2(self, max_iters: int = 100_000, use_deterministic_move: bool = USE_DETERMINISTIC_MOVE) -> List[Tuple[int, int]]:
        """H2 스케줄링 전략: IDA* 알고리즘 (DP 적용).
        
        Args:
            max_iters: 최대 반복 횟수
            use_deterministic_move: 결정적 이동 최적화 사용 여부 (기본값: USE_DETERMINISTIC_MOVE)
        """
        start_state = tuple(tuple(s) for s in self.env.stacks)
        initial_stacks = deepcopy(self.env.stacks)  # 초기 상태 저장
        threshold = self._get_lower_bound_h2(start_state)
        initial_threshold = threshold  # 초기 threshold 저장
        if DEBUG:
            print(f"[H2] 탐색 시작 (Initial Heuristic: {threshold})...")
        
        path = [{"state": start_state, "move": None}]
        
        iteration = 0
        while iteration < max_iters:
            iteration += 1
            
            # [중요] 새로운 threshold가 시작될 때마다 visited_costs를 초기화해야 함.
            # 이유: 이전 threshold에서 '비용이 너무 커서' 가지치기 당했던 경로가
            # threshold가 늘어난 지금은 유효한 경로일 수 있기 때문.
            # 하지만 *같은 threshold 내*에서는 중복 경로를 확실히 잡아줌.
            visited_costs: Dict[tuple, int] = {}
            
            if DEBUG:
                print(f"[H2] 심도(Threshold) {threshold} 탐색 중... (Deterministic Move: {use_deterministic_move})")
            res_f, res_path = self._dfs_h2(path, 0, threshold, visited_costs, use_deterministic_move)
            
            if res_path is not None:
                if DEBUG:
                    print(f"[H2] 🎉 최적해 발견! 총 {len(res_path)-1}회 이동.")
                moves = [step["move"] for step in res_path[1:]]
                self.moves = moves
                return moves
            
            if res_f == float("inf"):
                # 결정적 이동 최적화가 켜져있으면 끄고 처음부터 다시 탐색
                if use_deterministic_move:
                    if DEBUG:
                        print(f"[H2] res_f == inf 발생. 결정적 이동 최적화를 비활성화하고 처음부터 다시 탐색합니다.")
                    use_deterministic_move = False
                    # 처음부터 다시 시작: threshold와 path를 초기화
                    threshold = initial_threshold
                    path = [{"state": start_state, "move": None}]
                    iteration = 0  # iteration도 리셋하지 않으면 max_iters에 도달할 수 있음
                    continue  # while loop의 처음으로 돌아감
                else:
                    # 이미 최적화가 꺼져있는데도 inf가 발생하면 진짜 해가 없는 경우
                    print("H2: 해를 찾을 수 없습니다.")
                    print("\n초기 상태:")
                    # 초기 상태를 시각화하기 위해 임시로 env.stacks 복원
                    original_stacks = self.env.stacks
                    self.env.stacks = initial_stacks
                    self.env.visualize()
                    self.env.stacks = original_stacks  # 원래 상태로 복원
                    raise RuntimeError("H2: 해를 찾을 수 없습니다.")
            
            # threshold = res_f
            threshold = max(res_f, threshold + 3.0)  # 최적 대신 Threshold 공격적 증가로 속도 증가
        
        raise RuntimeError("H2: 최대 반복 횟수 내에 해를 찾지 못했습니다.")

    # ============================================================================
    # 공통 유틸리티 함수
    # ============================================================================

    def _move(self, src: int, dst: int) -> bool:
        """공통 이동 함수: env에 이동을 적용하고 moves 리스트에 기록."""
        if self.env.move(src, dst):
            self.moves.append((src, dst))
            return True
        return False

    # ============================================================================
    # Random 알고리즘
    # ============================================================================

    # Random 알고리즘은 solve_random()만으로 완성됨 (별도 헬퍼 함수 없음)

    # ============================================================================
    # H2 알고리즘 (IDA*)
    # ============================================================================

    def _is_sorted_h2(self, state) -> bool:
        """H2: env.is_solved()와 동일한 로직으로 해결 상태 확인."""
        # 기본 조건: 모든 스택이 자기 색상만 포함
        if all(all(item == stack_id for item in stack) for stack_id, stack in enumerate(state)):
            return True

        # Overflow 색상 확인: 특정 색상의 원소 수가 stack_capacity보다 많은 경우
        counts = Counter(item for stack in state for item in stack)
        overflow_types = set(
            item_type
            for item_type, total in counts.items()
            if total > self.env.stack_capacity
        )

        # Overflow 색상이 있으면 특별한 규칙 적용 (모든 overflow 색상을 동시에 고려)
        if overflow_types:
            return self._is_valid_with_overflow_h2(state, overflow_types)

        return False

    def _is_valid_with_overflow_h2(self, state, overflow_types: set[int]) -> bool:
        """H2: 여러 overflow 색상이 있을 때 해결 조건 확인."""
        for stack_id, stack in enumerate(state):
            if stack_id in overflow_types:
                # Overflow 색상의 스택은 자기 색상만 포함해야 함
                if not all(item == stack_id for item in stack):
                    return False
            else:
                # 나머지 스택은 아래에서부터 자기 색상, 그 위에 overflow 색상들만
                idx = 0
                # 자기 색상이 먼저 깔려있는지 확인
                while idx < len(stack) and stack[idx] == stack_id:
                    idx += 1
                # 그 위에 overflow 색상들만 있어야 함 (순서는 상관없음)
                while idx < len(stack) and stack[idx] in overflow_types:
                    idx += 1
                # 다른 색상이 있으면 실패
                if idx != len(stack):
                    return False
        return True

    def _get_lower_bound_h2(self, state) -> float:
        """
        H2: 휴리스틱 함수 h(n) - 깊이 가중치를 적용한 추정치.
        깊이 박혀있을수록(depth가 작을수록) 페널티를 크게 줌.
        """
        score = 0
        for stack_idx, stack in enumerate(state):
            is_clean = True
            for depth, ball in enumerate(stack):  # depth: 0(바닥) ~ N
                if is_clean:
                    if ball != stack_idx:
                        is_clean = False
                        # 깊이 박혀있을수록(depth가 작을수록) 페널티를 크게 줌
                        # 예: 높이 10인 스택에서 바닥(0)에 있으면 페널티 10, Top(9)에 있으면 1
                        score += (self.env.stack_capacity - depth) * 1.5  # 1.5는 가중치 상수
                else:
                    # 이미 더러워진 구간 위의 공들도 치워야 함
                    score += 1
        return score

    def _get_valid_moves_h2(self, state, last_src: int, last_dest: int, use_deterministic_move: bool = False) -> List[Tuple[int, int]]:
        """H2: 가능한 모든 이동을 생성 (핑퐁 방지 포함, 결정적 이동 우선 처리).
        
        Args:
            state: 현재 상태
            last_src: 직전 이동의 출발지
            last_dest: 직전 이동의 목적지
            use_deterministic_move: 결정적 이동 최적화 사용 여부
        """
        # [최적화 1] 결정적 이동(Deterministic Move) 감지 (옵션)
        # "내 목표 스택으로 바로 갈 수 있고, 거기가 받을 준비가 됐다면?" -> 딴 생각 말고 무조건 가라.
        if use_deterministic_move:
            for src in range(self.env.num_stacks):
                if not state[src]:
                    continue

                ball = state[src][-1]
                target_idx = ball % self.env.num_stacks  # 공의 숫자가 곧 목표 스택 인덱스

                # 목표 스택이 src와 다르고, 공간이 있으며
                if src != target_idx and len(state[target_idx]) < self.env.stack_capacity:
                    # 목표 스택이 'Ready(순수)' 상태인가? (바닥부터 자기 색깔만 있나)
                    is_target_ready = all(b == target_idx for b in state[target_idx])

                    if is_target_ready:
                        # 이것은 무조건 해야 하는 이동이다!
                        return [(src, target_idx)]  # 다른 move 다 버리고 이것만 리턴

        # [기존 로직] 결정적 이동이 없거나 비활성화된 경우 일반 탐색 수행
        moves = []
        for src in range(self.env.num_stacks):
            if not state[src]:  # 비어있으면 못 꺼냄
                continue

            for dst in range(self.env.num_stacks):
                if src == dst:
                    continue
                if len(state[dst]) >= self.env.stack_capacity:
                    continue  # 꽉 차서 못 넣음

                # 직전 행동 역행 방지 (Ping-Pong Pruning)
                if src == last_dest and dst == last_src:
                    continue

                moves.append((src, dst))

        return moves

    def _apply_move_h2(self, state, move: Tuple[int, int]):
        """H2: 이동 적용 (불변 튜플 반환, 튜플 슬라이싱 최적화)."""
        src, dst = move
        
        # 리스트 변환 없이 튜플 슬라이싱으로 처리 (속도 향상)
        src_stack = state[src]
        dst_stack = state[dst]
        
        ball = src_stack[-1]
        
        new_src_stack = src_stack[:-1]      # Pop
        new_dst_stack = dst_stack + (ball,) # Push
        
        # 전체 state 튜플 재조립
        # (src, dst 인덱스 위치만 바꿔치기)
        new_state = list(state)  # 얕은 복사 (내부 튜플은 그대로)
        new_state[src] = new_src_stack
        new_state[dst] = new_dst_stack
        
        return tuple(new_state)

    def _dfs_h2(self, path: List[dict], g: int, threshold: float, visited_costs: Dict[tuple, int], use_deterministic_move: bool = False) -> Tuple[float, Optional[List[dict]]]:
        """
        H2: DP 적용된 DFS 함수.
        
        Args:
            path: 현재까지의 경로 (상태들의 리스트가 아니라 이동 기록)
            g: 현재까지 이동 횟수
            threshold: 이번 탐색의 최대 깊이 한계 (f_limit)
            visited_costs: {state: min_g} 형태의 딕셔너리. 
                           해당 상태에 도달한 최소 비용을 기억함.
            use_deterministic_move: 결정적 이동 최적화 사용 여부
        """
        current_state = path[-1]["state"]
        
        # [DP 핵심 1] Memoization 체크
        # 이미 이 상태를 '현재보다 적거나 같은 비용(g)'으로 방문한 적이 있다면,
        # 이 경로는 더 볼 필요가 없음 (중복 또는 비효율적 경로).
        if current_state in visited_costs and visited_costs[current_state] <= g:
            # 이미 더 좋은 경로로 여길 와봤으므로, 이쪽 길은 가망 없음(혹은 중복).
            # 여기서 리턴하면 가지치기가 됨.
            # 단, 임계값 초과로 리턴하는 것과 구분하기 위해 아주 큰 값을 반환.
            return float("inf"), None

        # 방문 기록 업데이트 (더 적은 비용으로 왔으므로 갱신)
        visited_costs[current_state] = g

        f = g + self._get_lower_bound_h2(current_state)

        # 1. 가지치기 (Threshold check)
        if f > threshold:
            return f, None

        # 2. 목표 도달 확인
        if self._is_sorted_h2(current_state):
            return f, path

        min_surplus = float("inf")

        # 직전 이동 정보
        last_src, last_dest = -1, -1
        if len(path) > 1:
            last_src, last_dest = path[-1]["move"]

        # 3. 자식 노드 탐색
        valid_moves = self._get_valid_moves_h2(current_state, last_src, last_dest, use_deterministic_move)
        
        # (옵션) 휴리스틱 정렬: 유망한 자식을 먼저 탐색하면 가지치기 확률이 높아짐
        # valid_moves.sort(key=lambda m: ...) 

        for move in valid_moves:
            next_state = self._apply_move_h2(current_state, move)

            # [DP 핵심 2] 기존의 느린 Loop 방식 Cycle Check 삭제
            # visited_costs가 이미 Cycle Check 역할도 수행함 
            # (과거의 나: g가 더 작음 -> 현재의 나: g가 큼 -> visited_costs 조건에 걸림)
            
            new_path_node = {"state": next_state, "move": move}
            path.append(new_path_node)

            res_f, res_path = self._dfs_h2(path, g + 1, threshold, visited_costs, use_deterministic_move)

            if res_path is not None:
                return res_f, res_path

            if res_f < min_surplus:
                min_surplus = res_f

            path.pop()  # 백트래킹

        return min_surplus, None


def schedule(
    initial_stacks: List[List[int]], mode: str = "random", **kwargs
) -> Tuple[List[Tuple[int, int]], float]:
    """헬퍼 함수: 초기 스택 상태에서 필요한 move 시퀀스를 반환.
    
    Returns:
        (moves, elapsed_time): 이동 시퀀스와 소요 시간(초)의 튜플
    """
    scheduler = StackScheduler(initial_stacks)
    start_time = time.time()
    
    if mode == "random":
        moves = scheduler.solve_random(**kwargs)
    elif mode == "h2":
        moves = scheduler.solve_h2(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}")
    
    elapsed_time = time.time() - start_time
    return moves, elapsed_time


if __name__ == "__main__":
    env = StackRearrangementEnv()
    initial_stacks = deepcopy(env.stacks)  # 초기 상태 저장
    if DEBUG:
        env.visualize()
        print("\nScheduling...\n")
    try:
        moves, elapsed_time = schedule(env.stacks, mode="h2", max_iters=1_000_000)
        success = True
    except RuntimeError as err:
        print(f"스케줄 실패: {err}")
        print("\n초기 상태:")
        # 초기 상태를 시각화하기 위해 임시 env 생성
        temp_env = StackRearrangementEnv()
        temp_env.stacks = initial_stacks
        temp_env.visualize()
        moves = []
        elapsed_time = 0.0
        success = False

    if DEBUG:
        print(f"Total moves: {len(moves)}")
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    for src, dst in moves:
        env.move(src, dst)
    if DEBUG:
        print("\nSolved state:\n" if success else "\nPartial state (failure):\n")
        env.visualize()
    for idx, (src, dst) in enumerate(moves, start=1):
        if DEBUG:
            print(f"{idx:>3}: S{src} -> S{dst}")