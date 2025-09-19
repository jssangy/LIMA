import numpy as np
from typing import Dict, List, Tuple

class TaskSetGenerator:
    """
    한 번에 모든 태스크를 생성/스폰하는 TaskSetGenerator (배치 없음).
    - 시작: 테두리(가장자리) 전체 제외한 모든 walkable(0) 좌표에서 중복 없이 랜덤 추출
    - 목표: 네 모서리 중 walkable(0)에서 랜덤(없으면 테두리 walkable로 폴백)
    - 총 개수: num_tasks (시작 후보 수를 초과하면 자동 캡)
    - 반환 포맷: {"id": int, "start_pos": (x,y), "goal_pos": (x,y)}
    """
    def __init__(
        self,
        map_array: np.ndarray,
        num_tasks: int,                   # ★ 개수 기반만 지원
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

        self.start_candidates = self._collect_start_candidates()  # 테두리 제외
        self.goal_candidates = self._collect_goal_corners()
        if not self.goal_candidates:
            # 코너가 다 막혔으면 테두리 walkable에서 폴백
            self.goal_candidates = self._collect_border_walkables()
            if not self.goal_candidates:
                print("[TaskSetGenerator] Warning: no valid goal candidates.")

    # --- ENV에서 호출되는 API ---
    def start_new_episode(self, reset_ids: bool = True):
        if reset_ids:
            self.agv_id_counter = 0
        self.completed_total = 0
        self.task_set = []
        self.spawned_once = False

        if not self.start_candidates or not self.goal_candidates or self.walkable_count == 0:
            return

        # 총 AGV 수 = num_tasks (시작 후보 수를 넘지 않도록 캡)
        total_agvs = min(self.num_tasks, len(self.start_candidates))
        if total_agvs == 0:
            print("[TaskSetGenerator] num_tasks is 0 or no start candidates; no AGVs generated.")
            return

        # 시작 좌표: 중복 없이 샘플
        idxs = self.rng.choice(len(self.start_candidates), size=total_agvs, replace=False)
        starts = [self.start_candidates[i] for i in idxs]

        # 목표 좌표: 코너 중에서 랜덤(중복 허용)
        tasks: List[Dict] = []
        for s in starts:
            g = self.goal_candidates[int(self.rng.integers(0, len(self.goal_candidates)))]
            if g == s:
                # 드물지만 동일할 수 있으니 몇 번 교체 시도
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
        # 한 번만 전부 반환
        if not self.spawned_once and self.task_set:
            self.spawned_once = True
            print(f"[Time: {current_time}] Spawning ALL {len(self.task_set)} AGVs")
            return list(self.task_set)
        return []

    def should_spawn_next(self) -> bool:
        # 첫 호출에서만 스폰
        return (not self.spawned_once) and bool(self.task_set)

    def set_arm_gate(self, *args, **kwargs):
        # 호환용
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
            "total_tasks": len(self.task_set),  # ENV.make_info() 호환
        }

    # --- 내부 유틸 ---
    def _collect_walkables(self) -> List[Tuple[int, int]]:
        ys, xs = np.where(self.map == 0)
        return list(zip(xs.tolist(), ys.tolist()))  # (x,y)

    def _collect_start_candidates(self) -> List[Tuple[int, int]]:
        # 테두리(가장자리) 전체 제외
        cands: List[Tuple[int, int]] = []
        for x, y in self.walkable_coords:
            if x == 0 or x == self.W - 1 or y == 0 or y == self.H - 1:
                continue
            cands.append((x, y))
        return cands

    def _collect_goal_corners(self) -> List[Tuple[int, int]]:
        corners = [(0, 0), (self.W - 1, 0), (0, self.H - 1), (self.W - 1, self.H - 1)]
        goals = []
        for x, y in corners:
            if 0 <= x < self.W and 0 <= y < self.H and self.map[y][x] == 0:
                goals.append((x, y))
        return goals

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