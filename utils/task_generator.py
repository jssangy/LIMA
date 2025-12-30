from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

Coord = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


@dataclass(frozen=True, slots=True)
class Task:
    id: int
    start_pos: Coord
    goal_pos: Coord

    def as_dict(self) -> Dict:
        return {"id": self.id, "start_pos": self.start_pos, "goal_pos": self.goal_pos}


class RandomGenerator:
    """
    에피소드 시작 시 한 번에 task_set을 만들고, get_next_task_pair()에서 딱 한 번 반환하는 형태.

    기본 규칙:
    - start 후보: walkable(0) 좌표 중 테두리 제외(기본), goal 좌표 제외(기본)
    - goal 후보: 입력 goal_positions 우선(유효한 것만), 없거나 전부 무효면 코너 -> 테두리 walkable로 fallback
    - start는 중복 없이 샘플링, goal은 중복 허용

    옵션:
    - start_in_goal_bbox=True: goal 후보들의 (min_x~max_x, min_y~max_y) bbox 내부에서만 start 후보를 고름
    - bbox_margin: bbox를 상하좌우로 확장(선택)
    - start_bbox: bbox를 직접 지정(있으면 goal bbox 대신 이걸 사용)
    - fallback_to_any_start=True: bbox 필터 결과가 비면 bbox 없이 다시 후보를 구성
    """

    def __init__(
        self,
        map_array: np.ndarray,
        num_tasks: int,
        goal_positions: Optional[Iterable[Coord]] = None,
        *,
        seed: Optional[int] = None,
        exclude_border: bool = True,
        exclude_goals_from_start: bool = True,
        start_in_goal_bbox: bool = True,
        bbox_margin: int = 0,
        start_bbox: Optional[BBox] = None,
        fallback_to_any_start: bool = True,
    ):
        if map_array.ndim != 2:
            raise ValueError("map_array must be 2D")
        self.map = map_array
        self.H, self.W = map_array.shape

        self.rng = np.random.default_rng(seed)
        self.num_tasks = max(0, int(num_tasks))

        # 옵션
        self.exclude_border = bool(exclude_border)
        self.exclude_goals_from_start = bool(exclude_goals_from_start)
        self.start_in_goal_bbox = bool(start_in_goal_bbox)
        self.bbox_margin = int(bbox_margin)
        self.start_bbox = start_bbox
        self.fallback_to_any_start = bool(fallback_to_any_start)

        # 상태(ENV에서 사용)
        self.agv_id_counter = 0
        self.completed_total = 0
        self.spawned_once = False
        self.task_set: List[Dict] = []

        # 미리 집계(속도/가독성)
        self.walkable_coords: List[Coord] = self._collect_walkables()
        self.walkable_count = len(self.walkable_coords)

        # goal/start 후보 구성
        self.goal_candidates: List[Coord] = self._resolve_goal_candidates(goal_positions)
        self.start_candidates: List[Coord] = self._build_start_candidates()

    # -----------------------
    # ENV에서 호출되는 API
    # -----------------------
    def start_new_episode(self, reset_ids: bool = True) -> None:
        if reset_ids:
            self.agv_id_counter = 0
        self.completed_total = 0
        self.spawned_once = False
        self.task_set = []

        if self.walkable_count == 0:
            return
        if not self.goal_candidates:
            return
        if not self.start_candidates:
            return

        self.task_set = self._generate_task_set()

    def get_next_task_pair(self, current_time: int) -> List[Dict]:
        # 한 번만 spawn
        if not self.spawned_once and self.task_set:
            self.spawned_once = True
            return list(self.task_set)
        return []

    def should_spawn_next(self) -> bool:
        return (not self.spawned_once) and bool(self.task_set)

    def set_arm_gate(self, *args, **kwargs) -> None:
        # 기존 인터페이스 유지용(사용 안 하면 비워둬도 됨)
        return

    def complete_task(self, agv_id: int) -> None:
        self.completed_total += 1

    def is_episode_done(self) -> bool:
        if not self.task_set:
            return False
        return self.spawned_once and (self.completed_total >= len(self.task_set))

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

    def set_goal_positions(self, goal_positions: Iterable[Coord]) -> None:
        """goal set을 런타임에 갱신. 다음 episode부터 반영."""
        self.goal_candidates = self._validate_walkable_coords(goal_positions)
        self.start_candidates = self._build_start_candidates()

    # -----------------------
    # 내부 로직
    # -----------------------
    def _generate_task_set(self) -> List[Dict]:
        total = min(self.num_tasks, len(self.start_candidates))
        if total <= 0:
            return []

        # start는 중복 없이 샘플
        idxs = self.rng.choice(len(self.start_candidates), size=total, replace=False)
        starts = [self.start_candidates[int(i)] for i in idxs]

        tasks: List[Dict] = []
        goals = self.goal_candidates

        for s in starts:
            g = self._sample_goal_not_equal_to_start(goals, s)
            tasks.append(Task(self.agv_id_counter, s, g).as_dict())
            self.agv_id_counter += 1

        return tasks

    def _sample_goal_not_equal_to_start(self, goals: Sequence[Coord], start: Coord) -> Coord:
        if not goals:
            return start
        if len(goals) == 1:
            return goals[0]

        # start와 같은 goal을 피하고 싶으면, 가능한 경우만 회피
        # (goal 후보가 start랑 같은 것만 있는 경우는 어차피 불가능)
        for _ in range(8):
            g = goals[int(self.rng.integers(0, len(goals)))]
            if g != start:
                return g
        return goals[int(self.rng.integers(0, len(goals)))]

    def _build_start_candidates(self) -> List[Coord]:
        exclude_goals: Set[Coord] = set(self.goal_candidates) if self.exclude_goals_from_start else set()

        bbox = self._get_start_bbox()
        cands = self._collect_start_candidates(exclude_goals=exclude_goals, bbox=bbox)

        # bbox 때문에 후보가 비면 fallback
        if not cands and bbox is not None and self.fallback_to_any_start:
            cands = self._collect_start_candidates(exclude_goals=exclude_goals, bbox=None)

        return cands

    def _get_start_bbox(self) -> Optional[BBox]:
        # 사용자가 bbox를 직접 지정하면 그걸 우선
        if self.start_bbox is not None:
            return self._clamp_bbox(self.start_bbox)

        if not self.start_in_goal_bbox:
            return None

        if not self.goal_candidates:
            return None

        # goal의 x 범위만 사용
        xs = [x for x, _ in self.goal_candidates]
        x_min = min(xs) - self.bbox_margin
        x_max = max(xs) + self.bbox_margin

        # y는 전체(0 ~ H-1)
        return self._clamp_bbox((x_min, 0, x_max, self.H - 1))

    def _clamp_bbox(self, bbox: BBox) -> BBox:
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(self.W - 1, int(x_max))
        y_max = min(self.H - 1, int(y_max))
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        return (x_min, y_min, x_max, y_max)

    def _collect_walkables(self) -> List[Coord]:
        ys, xs = np.where(self.map == 0)
        return list(zip(xs.tolist(), ys.tolist()))  # (x, y)

    def _collect_start_candidates(
        self,
        *,
        exclude_goals: Set[Coord],
        bbox: Optional[BBox],
    ) -> List[Coord]:
        cands: List[Coord] = []

        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox

        for x, y in self.walkable_coords:
            # 테두리 제외(기본)
            if self.exclude_border:
                if x == 0 or x == self.W - 1 or y == 0 or y == self.H - 1:
                    continue

            # bbox 필터: y는 포함, x는 "경계 미포함"
            if bbox is not None:
                if not (x_min < x < x_max and y_min <= y <= y_max):
                    continue

            # goal 좌표 제외(기본)
            if (x, y) in exclude_goals:
                continue

            cands.append((x, y))

        return cands


    def _resolve_goal_candidates(self, goal_positions: Optional[Iterable[Coord]]) -> List[Coord]:
        # 1) 입력 goals 우선
        if goal_positions is not None:
            valid = self._validate_walkable_coords(goal_positions)
            if valid:
                return valid

        # 2) fallback: 코너 -> 테두리
        corners = self._collect_goal_corners()
        if corners:
            return corners

        border = self._collect_border_walkables()
        return border

    def _validate_walkable_coords(self, coords: Iterable[Coord]) -> List[Coord]:
        uniq: Set[Coord] = set()
        for p in coords:
            try:
                x, y = int(p[0]), int(p[1])
            except Exception:
                continue
            if 0 <= x < self.W and 0 <= y < self.H and self.map[y][x] == 0:
                uniq.add((x, y))
        return list(uniq)

    def _collect_goal_corners(self) -> List[Coord]:
        corners = [(0, 0), (self.W - 1, 0), (0, self.H - 1), (self.W - 1, self.H - 1)]
        return [(x, y) for (x, y) in corners if self.map[y][x] == 0]

    def _collect_border_walkables(self) -> List[Coord]:
        border: Set[Coord] = set()

        # 상/하
        for x in range(self.W):
            if self.map[0][x] == 0:
                border.add((x, 0))
            if self.map[self.H - 1][x] == 0:
                border.add((x, self.H - 1))

        # 좌/우
        for y in range(self.H):
            if self.map[y][0] == 0:
                border.add((0, y))
            if self.map[y][self.W - 1] == 0:
                border.add((self.W - 1, y))

        return list(border)


class ScenGenerator:
    """
    LaCAM .scen 파일 기반 Task generator.

    .scen 포맷(예):
      version 1
      0 <map_path> <W> <H> <sx> <sy> <gx> <gy> 0

    동작:
    - start_new_episode() 시 task_set을 준비
    - get_next_task_pair() 호출 시 한 번에 전부 spawn (TaskSetGenerator와 동일한 "1회 스폰" 패턴)
    - num_tasks가 주어지면 .scen의 앞에서부터 num_tasks개만 사용 (LaCAM의 -N과 동일 개념)
    - offset을 주면 앞에서 offset개는 건너뛰고 사용
    """

    def __init__(self, scen_path: str, num_tasks: Optional[int] = None, offset: int = 0):
        self.scen_path = scen_path
        self.num_tasks = None if num_tasks is None else max(0, int(num_tasks))
        self.offset = max(0, int(offset))

        # raw records: List[(sx,sy,gx,gy)]
        self._records: List[Tuple[int, int, int, int]] = self._load_scen(self.scen_path)

        # ENV가 기대하는 상태 변수들
        self.agv_id_counter = 0
        self.completed_total = 0
        self.spawned_once = False
        self.task_set: List[Dict] = []

    def _load_scen(self, scen_path: str) -> List[Tuple[int, int, int, int]]:
        records: List[Tuple[int, int, int, int]] = []
        with open(scen_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("version"):
                    continue

                parts = line.split()
                # bucket map W H sx sy gx gy opt  -> 최소 9개
                if len(parts) < 9:
                    continue

                sx = int(parts[4])
                sy = int(parts[5])
                gx = int(parts[6])
                gy = int(parts[7])

                records.append((sx, sy, gx, gy))
        return records

    # -----------------------
    # ENV에서 호출되는 API
    # -----------------------
    def start_new_episode(self, reset_ids: bool = True) -> None:
        if reset_ids:
            self.agv_id_counter = 0
        self.completed_total = 0
        self.spawned_once = False
        self.task_set = []

        # offset/limit 적용 (LaCAM처럼 "앞에서부터 N개" 사용)
        recs = self._records[self.offset:]
        if self.num_tasks is not None:
            recs = recs[: self.num_tasks]

        # id는 0..N-1로 재부여(ENV 내부에서 쓰기 편하게)
        for i, (sx, sy, gx, gy) in enumerate(recs):
            self.task_set.append({
                "id": i,
                "start_pos": (sx, sy),
                "goal_pos": (gx, gy),
            })

    def should_spawn_next(self) -> bool:
        return (not self.spawned_once) and bool(self.task_set)

    def get_next_task_pair(self, current_time: int) -> List[Dict]:
        if self.should_spawn_next():
            self.spawned_once = True
            return list(self.task_set)
        return []

    def complete_task(self, agv_id: int) -> None:
        self.completed_total += 1

    def is_episode_done(self) -> bool:
        # task가 없으면 즉시 종료(무한루프 방지)
        if not self.task_set:
            return True
        return self.spawned_once and (self.completed_total >= len(self.task_set))

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