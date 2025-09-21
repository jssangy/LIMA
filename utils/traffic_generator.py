import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable, Set

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
