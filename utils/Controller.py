import random
from typing import Dict, Optional, Tuple, List
import numpy as np

from utils.global_planning import BFS, CBS


class BFSPlanner:
    def __init__(self, map_data, center_xs, center_ys, rng=None):
        """
        Initializes a BFS-based distance field planner based on map data.
        """
        self.planner = BFS(map_data, rng)
        self.center_xs = center_xs
        self.center_ys = center_ys

    def plan_for_new_amrs(self, amr_list):
        """
        Calculates paths only for new AMRs that do not have a path.
        Since the BFS planner is very fast, there is almost no performance degradation even if all paths are recalculated.
        """
        for amr in amr_list.values():
            if not amr.path:
                self.calculate_and_set_path(amr)

    def replan_all(self, amr_list):
        """
        Forcibly recalculates paths for all AMRs.
        """
        for amr in amr_list.values():
            self.calculate_and_set_path(amr)

    def calculate_and_set_path(self, amr):
        """
        Calculates and injects a path using the BFS (distance field) planner.
        """
        path = self.planner.plan_path_highway(amr.pos, amr.goal, self.center_xs, self.center_ys)
        amr.set_path(path)

    def plan_path(self, start, goal):
        """
        Calculates a path from start to goal using the BFS (distance field) planner.
        """
        return self.planner.plan_path(start, goal)
    

Pos = Tuple[int, int]  # (x, y)

class CBSPlanner:
    def __init__(
        self,
        map_data: np.ndarray,
        *,
        seed: int = 7,
        time_limit: float = 10.0,
        center_xs: Optional[List[int]] = None,
        center_ys: Optional[List[int]] = None,
        trim_after_goal: bool = True,
        fallback: str = "bfs",  # "bfs" or "stay"
    ):
        """
        - map_data: 0 = free, 1 = obstacle (assuming LIMA map format)
        - time_limit: CBS time limit (seconds)
        - trim_after_goal: Remove padding after reaching goal (recommended True if removed upon goal arrival in the environment)
        - fallback: Alternative if CBS fails ("bfs"=individual BFS shortest path, "stay"=stay in place)
        """
        self.map = map_data
        self.time_limit = float(time_limit)
        self.trim_after_goal = bool(trim_after_goal)
        self.fallback = fallback

        self.center_xs = center_xs or []
        self.center_ys = center_ys or []

        self.rng = random.Random(seed)
        self.bfs = BFS(map_data, rng=self.rng)  # for plan_path / for fallback on failure

        # State for debugging/benchmarking
        self.last_conflicts: Optional[int] = None
        self.last_solved: Optional[bool] = None

        # For checking "if a new AMR has entered"
        self._known_ids: set[int] = set()

    # ---- ENV compatibility (some places in LIMA use planner.plan_path) ----
    def plan_path(self, start: Pos, goal: Pos) -> List[Pos]:
        if self.center_xs and self.center_ys:
            return self.bfs.plan_path_highway(start, goal, self.center_xs, self.center_ys)
        return self.bfs.plan_path(start, goal)

    # ---- Called after ENV spawn ----
    def plan_for_new_amrs(self, amr_list: Dict[int, object]) -> None:
        """
        If a new AMR is added or there is an AMR without a path, full replanning.
        """
        if not amr_list:
            return

        cur_ids = set(amr_list.keys())
        needs_replan = (cur_ids != self._known_ids) or any(
            not getattr(amr, "path", None) for amr in amr_list.values()
        )

        if needs_replan:
            self.replan_all(amr_list)
            self._known_ids = cur_ids

    def replan_all(self, amr_list: Dict[int, object]) -> None:
        if not amr_list:
            return

        # 1) CBS input configuration: fixed order (reproducibility)
        agents_to_plan: Dict[int, Dict[str, Pos]] = {}
        for aid in sorted(amr_list.keys()):
            amr = amr_list[aid]
            sx, sy = int(amr.pos[0]), int(amr.pos[1])   # assume (x,y)
            gx, gy = int(amr.goal[0]), int(amr.goal[1])
            agents_to_plan[aid] = {"start": (sx, sy), "goal": (gx, gy)}

        # 2) Run CBS
        solver = CBS(self.map, agents_to_plan)

        # Safer to sort agent_ids inside CBS as well
        solver.agent_ids = sorted(agents_to_plan.keys())

        sol = solver.solve(time_limit=self.time_limit)
        if sol is None:
            self.last_solved = False
            self.last_conflicts = None
            self._apply_fallback(amr_list)
            return

        # 3) If timeout, a solution with remaining conflicts may be returned -> judge success/failure by conflicts
        conflicts = solver.find_all_conflicts(sol)
        self.last_conflicts = conflicts
        self.last_solved = (conflicts == 0)

        # (In benchmarks, it is recommended to record "CBS success" as True only when conflicts==0)
        # If conflicts > 0 even in the environment, it's not a 'complete CBS solution', so it's good to leave a log.
        if conflicts > 0:
            print(f"[CBSPlanner] WARNING: solution has {conflicts} conflicts (timeout/best-so-far).")

        # 4) Inject path into AMR (trim until goal arrival if necessary)
        for aid, path in sol.items():
            if aid not in amr_list:
                continue

            amr = amr_list[aid]
            goal = (int(amr.goal[0]), int(amr.goal[1]))

            new_path = self._normalize_path(path)
            if self.trim_after_goal:
                new_path = self._trim_at_first_goal(new_path, goal)

            self._set_amr_path(amr, new_path)

        # If any AMR is missing in sol, fallback
        for aid, amr in amr_list.items():
            if not getattr(amr, "path", None):
                self._set_amr_path(amr, self._fallback_path(amr))

    # ---- Internal Utils ----
    def _normalize_path(self, path: List[Pos]) -> List[Pos]:
        # Normalize to tuple(int, int) as numpy int etc. may be mixed
        return [(int(x), int(y)) for (x, y) in path] if path else []

    def _trim_at_first_goal(self, path: List[Pos], goal: Pos) -> List[Pos]:
        if not path:
            return path
        for t, p in enumerate(path):
            if p == goal:
                return path[: t + 1]
        return path

    def _set_amr_path(self, amr: object, path: List[Pos]) -> None:
        # Safer to use set_path if it exists according to the project AMR implementation
        if hasattr(amr, "set_path"):
            amr.set_path(path)
            return

        # If not, set minimum fields (adjustable to project)
        amr.path = path
        amr.path_cursor = 0
        amr.next_pos = path[1] if len(path) > 1 else (path[0] if path else tuple(amr.pos))

    def _fallback_path(self, amr: object) -> List[Pos]:
        start = (int(amr.pos[0]), int(amr.pos[1]))
        goal = (int(amr.goal[0]), int(amr.goal[1]))
        if self.fallback == "stay":
            return [start]
        p = self.plan_path(start, goal)
        return p if p else [start]

    def _apply_fallback(self, amr_list: Dict[int, object]) -> None:
        for amr in amr_list.values():
            self._set_amr_path(amr, self._fallback_path(amr))

    # CBS is a global planner, so single calculation is usually not used. Still provided for env compatibility.
    def calculate_and_set_path(self, amr: object) -> None:
        self._set_amr_path(amr, self._fallback_path(amr))