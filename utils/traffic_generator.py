import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable, Set, Callable


class TaskSetGenerator:
    """
    한 번에 모든 태스크를 생성/스폰하는 TaskSetGenerator (배치 없음).
    - 시작: 테두리(가장자리) 전체 제외한 모든 walkable(0) 좌표에서 중복 없이 랜덤 추출
            (단, goal set에 포함된 좌표는 시작 후보에서 제외)
    - 목표: 입력 goal set 중에서 랜덤(중복 허용). goal set이 없으면 코너→테두리 순으로 폴백
    - 총 개수: num_tasks (시작 후보 수를 초과하면 자동 캡)
    - 반환: {"id": int, "start_pos": (x,y), "goal_pos": (x,y)}
    """
    def __init__(
        self,
        map_array: np.ndarray,
        num_tasks: int,
        goal_positions: Optional[Iterable[Tuple[int, int]]] = None,
    ):
        assert map_array.ndim == 2, "map_array must be 2D"
        self.map = map_array
        self.H, self.W = map_array.shape
        self.rng = np.random.default_rng()

        self.num_tasks = max(0, int(num_tasks))

        # 상태
        self.agv_id_counter = 0
        self.completed_total = 0
        self.task_set: List[Dict] = []
        self.spawned_once = False

        # 미리 집계
        self.walkable_coords = self._collect_walkables()
        self.walkable_count = len(self.walkable_coords)

        # 1) goal 후보 확정 (입력 set 우선)
        self.goal_candidates = self._resolve_goal_candidates(goal_positions)

        # 2) 시작 후보는 goal을 제외하고 테두리 전체 제외
        exclude_goals: Set[Tuple[int,int]] = set(self.goal_candidates)
        self.start_candidates = self._collect_start_candidates(exclude_goals=exclude_goals)

        if not self.goal_candidates:
            print("[TaskSetGenerator] Warning: no valid goal candidates.")
        if not self.start_candidates:
            print("[TaskSetGenerator] Warning: no valid start candidates.")

    # --- ENV에서 호출되는 API ---
    def start_new_episode(self, reset_ids: bool = True):
        if reset_ids:
            self.agv_id_counter = 0
        self.completed_total = 0
        self.task_set = []
        self.spawned_once = False

        if not self.start_candidates or not self.goal_candidates or self.walkable_count == 0:
            return

        total_agvs = min(self.num_tasks, len(self.start_candidates))
        if total_agvs == 0:
            print("[TaskSetGenerator] num_tasks is 0 or no start candidates; no AGVs generated.")
            return

        # 시작 좌표: 중복 없이 샘플
        idxs = self.rng.choice(len(self.start_candidates), size=total_agvs, replace=False)
        starts = [self.start_candidates[i] for i in idxs]

        # 목표 좌표: goal set에서 랜덤(중복 허용)
        tasks: List[Dict] = []
        for s in starts:
            g = self.goal_candidates[int(self.rng.integers(0, len(self.goal_candidates)))]
            if g == s:
                # 시작과 동일하면 다른 goal로 몇 번 교체 시도
                for _ in range(8):
                    g2 = self.goal_candidates[int(self.rng.integers(0, len(self.goal_candidates)))]
                    if g2 != s:
                        g = g2
                        break
            tasks.append({
                "id": self.agv_id_counter,
                "start_pos": tuple(s),
                "goal_pos": tuple(g),
            })
            self.agv_id_counter += 1

        self.task_set = tasks
        print(f"Generated {len(self.task_set)} AGVs (one-shot, count-based).")

    def get_next_task_pair(self, current_time: int) -> List[Dict]:
        if not self.spawned_once and self.task_set:
            self.spawned_once = True
            print(f"[Time: {current_time}] Spawning ALL {len(self.task_set)} AGVs")
            return list(self.task_set)
        return []

    def should_spawn_next(self) -> bool:
        return (not self.spawned_once) and bool(self.task_set)

    def set_arm_gate(self, *args, **kwargs):
        pass

    def complete_task(self, agv_id: int):
        self.completed_total += 1

    def is_episode_done(self) -> bool:
        if not self.task_set:
            return False
        all_spawned = self.spawned_once
        all_completed = self.completed_total >= len(self.task_set)
        return all_spawned and all_completed

    def get_progress(self) -> Dict:
        spawned_count = len(self.task_set) if self.spawned_once else 0
        active = spawned_count - self.completed_total
        return {
            "spawned_total": spawned_count,
            "completed_total": self.completed_total,
            "active_agvs": active,
            "max_agvs": len(self.task_set),
            "total_tasks": len(self.task_set),
        }

    # --- 런타임에 goal set 교체 (선택) ---
    def set_goal_positions(self, goal_positions: Iterable[Tuple[int,int]]):
        """goal set을 런타임에 갱신. 다음 episode부터 반영."""
        self.goal_candidates = self._validate_goals(goal_positions)
        exclude_goals = set(self.goal_candidates)
        self.start_candidates = self._collect_start_candidates(exclude_goals=exclude_goals)

    # --- 내부 유틸 ---
    def _collect_walkables(self) -> List[Tuple[int, int]]:
        ys, xs = np.where(self.map == 0)
        return list(zip(xs.tolist(), ys.tolist()))  # (x,y)

    def _collect_start_candidates(self, exclude_goals: Optional[Set[Tuple[int,int]]] = None) -> List[Tuple[int, int]]:
        cands: List[Tuple[int, int]] = []
        excl = exclude_goals or set()
        for x, y in self.walkable_coords:
            # 테두리 제외
            if x == 0 or x == self.W - 1 or y == 0 or y == self.H - 1:
                continue
            # goal set 제외
            if (x, y) in excl:
                continue
            cands.append((x, y))
        return cands

    def _resolve_goal_candidates(self, goal_positions: Optional[Iterable[Tuple[int,int]]]) -> List[Tuple[int,int]]:
        if goal_positions:
            valid = self._validate_goals(goal_positions)
            if valid:
                return valid
            # 입력이 모두 무효면 폴백
            print("[TaskSetGenerator] Provided goal set is empty/invalid. Falling back.")
        # 폴백: 코너 → 테두리
        goals = self._collect_goal_corners()
        if not goals:
            goals = self._collect_border_walkables()
        return goals

    def _validate_goals(self, goal_positions: Iterable[Tuple[int,int]]) -> List[Tuple[int,int]]:
        uniq: Set[Tuple[int,int]] = set()
        for p in goal_positions:
            try:
                x, y = int(p[0]), int(p[1])
            except Exception:
                continue
            if 0 <= x < self.W and 0 <= y < self.H and self.map[y][x] == 0:
                uniq.add((x, y))
        return list(uniq)

    def _collect_goal_corners(self) -> List[Tuple[int, int]]:
        corners = [(0, 0), (self.W - 1, 0), (0, self.H - 1), (self.W - 1, self.H - 1)]
        return [(x, y) for (x, y) in corners if self.map[y][x] == 0]

    def _collect_border_walkables(self) -> List[Tuple[int, int]]:
        border: List[Tuple[int, int]] = []
        for x in range(self.W):
            for y in [0, self.H - 1]:
                if self.map[y][x] == 0:
                    border.append((x, y))
        for y in range(1, self.H - 1):
            for x in [0, self.W - 1]:
                if self.map[y][x] == 0:
                    border.append((x, y))
        # 중복 제거
        seen = set(); uniq = []
        for p in border:
            if p not in seen:
                seen.add(p); uniq.append(p)
        return uniq


DIRS = ("N", "E", "S", "W")

def discover_border_arms_NxM(intersections: Dict[str, object]) -> List[Tuple[str, str]]:
    """
    [새로 추가된 함수]
    N x M 그리드에서 맵 바깥을 향하는 모든 외곽 팔을 찾아 반환.
    """
    if not intersections:
        return []

    # 1) 모든 교차로의 x, y 좌표를 수집하여 고유한 값만 정렬
    all_x = sorted(list({I.center_x for I in intersections.values()}))
    all_y = sorted(list({I.center_y for I in intersections.values()}))

    N = len(all_x)
    M = len(all_y)

    if N == 0 or M == 0:
        return []

    # 2) 좌표 값을 그리드 인덱스(0, 1, ...)로 변환하기 위한 맵 생성
    x_to_idx = {x: i for i, x in enumerate(all_x)}
    y_to_idx = {y: i for i, y in enumerate(all_y)}

    # 3) 외곽 팔 수집
    border_arms: List[Tuple[str, str]] = []
    for iid, I in intersections.items():
        ix = x_to_idx.get(I.center_x)
        iy = y_to_idx.get(I.center_y)

        if ix is None or iy is None:
            continue

        # 가장 왼쪽 열에 있는 교차로의 '서쪽(W)' 팔
        if ix == 0:
            border_arms.append((iid, "W"))
        # 가장 오른쪽 열에 있는 교차로의 '동쪽(E)' 팔
        if ix == N - 1:
            border_arms.append((iid, "E"))
        # 가장 위쪽 행에 있는 교차로의 '북쪽(N)' 팔
        if iy == 0:
            border_arms.append((iid, "N"))
        # 가장 아래쪽 행에 있는 교차로의 '남쪽(S)' 팔
        if iy == M - 1:
            border_arms.append((iid, "S"))
            
    print(f"Discovered {len(border_arms)} border arms from a {N}x{M} grid.")
    return border_arms


class TrafficGenerator:
    """
    외곽 팔에서만 스폰하는 트래픽 생성기.
    - 각 arm별로 Poisson 분포에 따라 AGV 생성.
    - [수정] goal은 항상 출발 팔의 '반대 방향'에 있는 팔 중에서 균등 샘플.
    - arm_gate 콜백이 False면 해당 팔에서 생성하지 않음.
    - max_agvs를 초과하여 생성하지 않음.
    """
    def __init__(
        self,
        arms: List[Tuple[str, str]],
        lam: float = 0.2,
        lambda_per_arm: Optional[Dict[Tuple[str, str], float]] = None,
        arm_gate: Optional[Callable[[str, str], bool]] = None,
        debug: bool = False,
        max_agvs: int = 12,
    ):
        self.rng = np.random.default_rng()
        self.arms = [(str(iid), d) for (iid, d) in arms]
        self.lam = float(lam)
        self.lambda_arm = {
            (iid, d): (lambda_per_arm.get((iid, d), self.lam) if lambda_per_arm else self.lam)
            for (iid, d) in self.arms
        }
        self.agv_id_counter = 0
        self.spawned_total = 0
        self.completed_total = 0
        self.step_count = 0
        self.max_agvs = max_agvs

        # [추가] 방향별로 팔을 분류하고, 반대 방향을 매핑
        self.arms_by_direction = {"N": [], "S": [], "E": [], "W": []}
        for iid, d in self.arms:
            if d in self.arms_by_direction:
                self.arms_by_direction[d].append((iid, d))
        
        self.opposite_direction = {"N": "S", "S": "N", "E": "W", "W": "E"}

        # [추가] 유효성 검사: 모든 방향에 대해 반대 방향 팔이 존재하는지 확인
        for direction, arm_list in self.arms_by_direction.items():
            if arm_list:  # 해당 방향에 출발 팔이 있다면
                opposite_dir = self.opposite_direction[direction]
                if not self.arms_by_direction[opposite_dir]:
                    print(f"Warning: Arms exist for direction '{direction}', but no arms found for the opposite direction '{opposite_dir}'. This may cause errors.")

        self._arm_gate = arm_gate
        self.debug = bool(debug)

    def set_arm_gate(self, fn: Callable[[str, str], bool]) -> None:
        self._arm_gate = fn

    # --- 외부 인터페이스 ---
    def start_new_episode(self, reset_ids: bool = True):
        if reset_ids:
            self.agv_id_counter = 0
            self.spawned_total = 0
            self.completed_total = 0
            self.step_count = 0

    def should_spawn_next(self) -> bool:
        return True

    def get_next_task_pair(self) -> List[Dict]:
        self.step_count += 1
        tasks: List[Dict] = []
        active_agvs = self.spawned_total - self.completed_total

        if active_agvs >= self.max_agvs:
            return []

        arms_rr = list(self.arms)
        random.shuffle(arms_rr)
        for (iid, d) in arms_rr:
            if (active_agvs + len(tasks)) >= self.max_agvs:
                break

            if self._arm_gate is not None and not self._arm_gate(iid, d):
                continue

            k = int(self.rng.poisson(self.lambda_arm.get((iid, d), self.lam)))
            if k <= 0:
                continue
            
            for _ in range(k):
                if (active_agvs + len(tasks)) >= self.max_agvs:
                    break

                # [수정] 새로운 목적지 샘플링 함수 호출
                try:
                    gid, gd = self._sample_goal_opposite((iid, d))
                    tasks.append({
                        "id": self.agv_id_counter,
                        "intersection_id": iid,
                        "start_direction": d,
                        "goal_intersection_id": gid,
                        "goal_direction": gd,
                    })
                    self.agv_id_counter += 1
                    self.spawned_total += 1
                except ValueError as e:
                    if self.debug:
                        print(f"[TG12:{self.step_count}] Goal sampling error: {e}")
                    continue # 목적지를 찾을 수 없으면 해당 AGV는 생성하지 않음

        return tasks

    def complete_task(self, agv_id: int):
        self.completed_total += 1

    def is_episode_done(self) -> bool:
        return False

    def get_progress(self) -> Dict:
        return {
            "spawned_total": self.spawned_total,
            "completed_total": self.completed_total,
            "active_agvs": self.spawned_total - self.completed_total,
            "max_agvs": self.max_agvs,
            "step": self.step_count,
            "arms": list(self.arms),
            "lambdas": {f"{iid}:{d}": self.lambda_arm.get((iid,d)) for (iid, d) in self.arms},
        }

    # --- 내부 ---
    def _sample_goal_opposite(self, src_arm: Tuple[str, str]) -> Tuple[str, str]:
        """
        [새로운 함수]
        출발 팔의 반대 방향에 있는 팔들 중에서 목적지를 균등하게 샘플링.
        """
        _, src_dir = src_arm
        
        target_dir = self.opposite_direction[src_dir]
        candidate_goals = self.arms_by_direction[target_dir]
        
        if not candidate_goals:
            raise ValueError(f"No goal arms found for opposite direction '{target_dir}'.")

        goal_idx = self.rng.integers(len(candidate_goals))
        return candidate_goals[goal_idx]