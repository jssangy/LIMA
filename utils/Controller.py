import random
from typing import Dict, Optional, Tuple, List
import numpy as np

from utils.global_planning import BFS, CBS


class BFSPlanner:
    def __init__(self, map_data, center_xs, center_ys, rng=None):
        """
        맵 데이터를 기반으로 BFS 기반 거리장 플래너를 초기화합니다.
        """
        self.planner = BFS(map_data, rng)
        self.center_xs = center_xs
        self.center_ys = center_ys

    def plan_for_new_amrs(self, amr_list):
        """
        경로가 없는 새로운 AMR에 대해서만 경로를 계산합니다.
        BFS 플래너는 매우 빠르므로, 모든 경로를 다시 계산해도 성능 저하가 거의 없습니다.
        """
        for amr in amr_list.values():
            if not amr.path:
                self.calculate_and_set_path(amr)

    def replan_all(self, amr_list):
        """
        모든 AMR의 경로를 강제로 다시 계산합니다.
        """
        for amr in amr_list.values():
            self.calculate_and_set_path(amr)

    def calculate_and_set_path(self, amr):
        """
        BFS(거리장) 플래너를 사용하여 경로를 계산하고 주입합니다.
        """
        path = self.planner.plan_path_highway(amr.pos, amr.goal, self.center_xs, self.center_ys)
        amr.set_path(path)

    def plan_path(self, start, goal):
        """
        BFS(거리장) 플래너를 사용하여 start에서 goal까지의 경로를 계산합니다.
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
        - map_data: 0 = free, 1 = obstacle (LIMA map 형식 가정)
        - time_limit: CBS 제한시간(초)
        - trim_after_goal: goal 도착 이후 padding 제거(환경에서 goal 도착 시 제거한다면 True 추천)
        - fallback: CBS 실패 시 대체 ("bfs"=각자 BFS 최단경로, "stay"=제자리)
        """
        self.map = map_data
        self.time_limit = float(time_limit)
        self.trim_after_goal = bool(trim_after_goal)
        self.fallback = fallback

        self.center_xs = center_xs or []
        self.center_ys = center_ys or []

        self.rng = random.Random(seed)
        self.bfs = BFS(map_data, rng=self.rng)  # plan_path 용/실패시 fallback 용

        # 디버깅/벤치마크용 상태
        self.last_conflicts: Optional[int] = None
        self.last_solved: Optional[bool] = None

        # "새 AMR 들어왔는지" 체크용
        self._known_ids: set[int] = set()

    # ---- ENV 호환 (LIMA에서 planner.plan_path를 쓰는 곳이 있음) ----
    def plan_path(self, start: Pos, goal: Pos) -> List[Pos]:
        if self.center_xs and self.center_ys:
            return self.bfs.plan_path_highway(start, goal, self.center_xs, self.center_ys)
        return self.bfs.plan_path(start, goal)

    # ---- ENV가 spawn 후 호출 ----
    def plan_for_new_amrs(self, amr_list: Dict[int, object]) -> None:
        """
        새 AMR이 추가되었거나, path가 없는 AMR이 있으면 전체 재계획.
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

        # 1) CBS 입력 구성: 순서 고정(재현성)
        agents_to_plan: Dict[int, Dict[str, Pos]] = {}
        for aid in sorted(amr_list.keys()):
            amr = amr_list[aid]
            sx, sy = int(amr.pos[0]), int(amr.pos[1])   # (x,y) 가정
            gx, gy = int(amr.goal[0]), int(amr.goal[1])
            agents_to_plan[aid] = {"start": (sx, sy), "goal": (gx, gy)}

        # 2) CBS 실행
        solver = CBS(self.map, agents_to_plan)

        # CBS 내부 agent_ids도 정렬해두면 더 안전
        solver.agent_ids = sorted(agents_to_plan.keys())

        sol = solver.solve(time_limit=self.time_limit)
        if sol is None:
            self.last_solved = False
            self.last_conflicts = None
            self._apply_fallback(amr_list)
            return

        # 3) timeout이면 conflicts가 남은 해를 반환할 수 있음 → conflicts로 성공/실패 판단
        conflicts = solver.find_all_conflicts(sol)
        self.last_conflicts = conflicts
        self.last_solved = (conflicts == 0)

        # (벤치마크에서 "CBS 성공"은 conflicts==0 인 경우만 True로 기록 추천)
        # 환경에서라도 conflicts>0이면 '완전한 CBS 해'가 아니니 로그 남기는 게 좋음.
        if conflicts > 0:
            print(f"[CBSPlanner] WARNING: solution has {conflicts} conflicts (timeout/best-so-far).")

        # 4) AMR에 경로 주입 (필요하면 goal 도착까지만 trim)
        for aid, path in sol.items():
            if aid not in amr_list:
                continue

            amr = amr_list[aid]
            goal = (int(amr.goal[0]), int(amr.goal[1]))

            new_path = self._normalize_path(path)
            if self.trim_after_goal:
                new_path = self._trim_at_first_goal(new_path, goal)

            self._set_amr_path(amr, new_path)

        # 혹시 sol에 누락된 AMR이 있으면 fallback
        for aid, amr in amr_list.items():
            if not getattr(amr, "path", None):
                self._set_amr_path(amr, self._fallback_path(amr))

    # ---- 내부 유틸 ----
    def _normalize_path(self, path: List[Pos]) -> List[Pos]:
        # numpy int 등이 섞일 수 있으니 tuple(int,int)로 정규화
        return [(int(x), int(y)) for (x, y) in path] if path else []

    def _trim_at_first_goal(self, path: List[Pos], goal: Pos) -> List[Pos]:
        if not path:
            return path
        for t, p in enumerate(path):
            if p == goal:
                return path[: t + 1]
        return path

    def _set_amr_path(self, amr: object, path: List[Pos]) -> None:
        # 프로젝트 AMR 구현에 따라 set_path가 있으면 그걸 쓰는 게 안전
        if hasattr(amr, "set_path"):
            amr.set_path(path)
            return

        # 없으면 최소한의 필드 세팅(프로젝트에 맞게 조정 가능)
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

    # CBS는 전역 플래너라 단일 계산은 보통 안 씀. 그래도 env 호환용으로 제공.
    def calculate_and_set_path(self, amr: object) -> None:
        self._set_amr_path(amr, self._fallback_path(amr))