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
    Creates a task_set all at once at the start of an episode, and returns it exactly once in get_next_task_pair().

    Basic Rules:
    - start candidates: walkable(0) coordinates excluding borders (default) and goal coordinates (default)
    - goal candidates: input goal_positions first (only valid ones), if none or all invalid, fallback to corner -> border walkable
    - start is sampled without duplicates, goal allows duplicates

    Options:
    - start_in_goal_bbox=True: picks start candidates only within the (min_x~max_x, min_y~max_y) bbox of goal candidates
    - bbox_margin: expands the bbox in all directions (optional)
    - start_bbox: directly specifies the bbox (if present, used instead of goal bbox)
    - fallback_to_any_start=True: if bbox filtering results in an empty set, reconstruct candidates without bbox
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

        # Options
        self.exclude_border = bool(exclude_border)
        self.exclude_goals_from_start = bool(exclude_goals_from_start)
        self.start_in_goal_bbox = bool(start_in_goal_bbox)
        self.bbox_margin = int(bbox_margin)
        self.start_bbox = start_bbox
        self.fallback_to_any_start = bool(fallback_to_any_start)

        # State (used in ENV)
        self.agv_id_counter = 0
        self.completed_total = 0
        self.spawned_once = False
        self.task_set: List[Dict] = []

        # Pre-aggregation (speed/readability)
        self.walkable_coords: List[Coord] = self._collect_walkables()
        self.walkable_count = len(self.walkable_coords)

        # Configure goal/start candidates
        self.goal_candidates: List[Coord] = self._resolve_goal_candidates(goal_positions)
        self.start_candidates: List[Coord] = self._build_start_candidates()

    # -----------------------
    # API called from ENV
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
        # Spawn only once
        if not self.spawned_once and self.task_set:
            self.spawned_once = True
            return list(self.task_set)
        return []

    def should_spawn_next(self) -> bool:
        return (not self.spawned_once) and bool(self.task_set)

    def set_arm_gate(self, *args, **kwargs) -> None:
        # For maintaining existing interface (can be left empty if not used)
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
        """Update goal set at runtime. Reflected from the next episode."""
        self.goal_candidates = self._validate_walkable_coords(goal_positions)
        self.start_candidates = self._build_start_candidates()

    # -----------------------
    # Internal Logic
    # -----------------------
    def _generate_task_set(self) -> List[Dict]:
        total = min(self.num_tasks, len(self.start_candidates))
        if total <= 0:
            return []

        # Sample start without duplicates
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

        # If you want to avoid a goal that is the same as start, avoid it only when possible
        # (It's impossible anyway if the goal candidate is only the same as start)
        for _ in range(8):
            g = goals[int(self.rng.integers(0, len(goals)))]
            if g != start:
                return g
        return goals[int(self.rng.integers(0, len(goals)))]

    def _build_start_candidates(self) -> List[Coord]:
        exclude_goals: Set[Coord] = set(self.goal_candidates) if self.exclude_goals_from_start else set()

        bbox = self._get_start_bbox()
        cands = self._collect_start_candidates(exclude_goals=exclude_goals, bbox=bbox)

        # Fallback if candidates are empty due to bbox
        if not cands and bbox is not None and self.fallback_to_any_start:
            cands = self._collect_start_candidates(exclude_goals=exclude_goals, bbox=None)

        return cands

    def _get_start_bbox(self) -> Optional[BBox]:
        # If the user directly specifies a bbox, prioritize that
        if self.start_bbox is not None:
            return self._clamp_bbox(self.start_bbox)

        if not self.start_in_goal_bbox:
            return None

        if not self.goal_candidates:
            return None

        # Use only the x range of the goal
        xs = [x for x, _ in self.goal_candidates]
        x_min = min(xs) - self.bbox_margin
        x_max = max(xs) + self.bbox_margin

        # y is full range (0 ~ H-1)
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
            # Exclude border (default)
            if self.exclude_border:
                if x == 0 or x == self.W - 1 or y == 0 or y == self.H - 1:
                    continue

            # bbox filter: y is inclusive, x is "exclusive of boundaries"
            if bbox is not None:
                if not (x_min < x < x_max and y_min <= y <= y_max):
                    continue

            # Exclude goal coordinates (default)
            if (x, y) in exclude_goals:
                continue

            cands.append((x, y))

        return cands


    def _resolve_goal_candidates(self, goal_positions: Optional[Iterable[Coord]]) -> List[Coord]:
        # 1) Prioritize input goals
        if goal_positions is not None:
            valid = self._validate_walkable_coords(goal_positions)
            if valid:
                return valid

        # 2) fallback: corners -> border
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

        # Top/Bottom
        for x in range(self.W):
            if self.map[0][x] == 0:
                border.add((x, 0))
            if self.map[self.H - 1][x] == 0:
                border.add((x, self.H - 1))

        # Left/Right
        for y in range(self.H):
            if self.map[y][0] == 0:
                border.add((0, y))
            if self.map[y][self.W - 1] == 0:
                border.add((self.W - 1, y))

        return list(border)


class ScenGenerator:
    """
    Task generator based on LaCAM .scen files.

    .scen format (example):
      version 1
      0 <map_path> <W> <H> <sx> <sy> <gx> <gy> 0

    Behavior:
    - Prepares task_set during start_new_episode()
    - Spawns all at once when get_next_task_pair() is called (same "one-time spawn" pattern as TaskSetGenerator)
    - If num_tasks is given, uses only num_tasks from the beginning of the .scen (same concept as LaCAM's -N)
    - If offset is given, skips the first offset tasks
    """

    def __init__(self, scen_path: str, num_tasks: Optional[int] = None, offset: int = 0):
        self.scen_path = scen_path
        self.num_tasks = None if num_tasks is None else max(0, int(num_tasks))
        self.offset = max(0, int(offset))

        # raw records: List[(sx,sy,gx,gy)]
        self._records: List[Tuple[int, int, int, int]] = self._load_scen(self.scen_path)

        # ENV expected state variables
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
                # bucket map W H sx sy gx gy opt  -> at least 9 parts
                if len(parts) < 9:
                    continue

                sx = int(parts[4])
                sy = int(parts[5])
                gx = int(parts[6])
                gy = int(parts[7])

                records.append((sx, sy, gx, gy))
        return records

    # -----------------------
    # API called from ENV
    # -----------------------
    def start_new_episode(self, reset_ids: bool = True) -> None:
        if reset_ids:
            self.agv_id_counter = 0
        self.completed_total = 0
        self.spawned_once = False
        self.task_set = []

        # Apply offset/limit (use "N from the front" like LaCAM)
        recs = self._records[self.offset:]
        if self.num_tasks is not None:
            recs = recs[: self.num_tasks]

        # Re-assign IDs as 0..N-1 (for convenience within ENV)
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
        # If no tasks, terminate immediately (prevent infinite loop)
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