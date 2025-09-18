import copy
import numpy as np
from collections import defaultdict

from utils.global_planning import AStar, PIBT

class controller():    
    def __init__(self, map_data):
        # 동적 AGV 관리를 위해 초기화 시 비워둠
        self.agv_pos = {}
        self.next_buffer = {}
        self.control_buffer = {}
        self.agv_nums = []
        self.agv_goal = {}
        self.planners = {}
        self.agv_path = {}

        # Map of the environment
        self.map = map_data

        self.push_sequence = []

        self.running_opt = 0
        self.pibt = PIBT(self.map)
        self.pibt_vanilla = PIBTVanilla(self.map)

    def reset(self):
        self.agv_pos.clear()
        self.next_buffer.clear()
        self.control_buffer.clear()
        self.agv_nums.clear()
        self.agv_goal.clear()
        self.planners.clear()
        self.agv_path.clear()
        self.push_sequence.clear()
        self.pibt.reset()
        self.pibt_vanilla.reset()

    def add_agv(self, agv_num, start_pos, goal_pos):
        """AGV를 컨트롤러에 동적으로 추가"""
        self.agv_nums.append(agv_num)
        self.agv_pos[agv_num] = start_pos
        self.agv_goal[agv_num] = goal_pos
        self.next_buffer[agv_num] = (0, 0)
        self.control_buffer[agv_num] = (0, 0)
        self.planners[agv_num] = AStar(self.map, start_pos, goal_pos)
        self.agv_path[agv_num] = []

    def remove_agv(self, agv_num):
        """완료된 AGV를 컨트롤러에서 제거"""
        if agv_num in self.agv_nums:
            self.agv_nums.remove(agv_num)
            self.agv_pos.pop(agv_num, None)
            self.agv_goal.pop(agv_num, None)
            self.next_buffer.pop(agv_num, None)
            self.control_buffer.pop(agv_num, None)
            self.planners.pop(agv_num, None)
            self.agv_path.pop(agv_num, None)

    def get_sensing(self, agv_num, pos):
        """Env로부터 AGV의 현재 위치를 업데이트"""
        if agv_num in self.agv_nums:
            self.agv_pos[agv_num] = pos

    def make_control(self):
        """모든 활성 AGV에 대한 제어 신호 생성"""
        if self.running_opt == 0:              # A*
            self.astar_rout()
        elif self.running_opt == 1:            # D*
            self.dstar_rout()
        elif self.running_opt == 2:            # PIBT 충돌 시
            self.pibt_rout(use_dstar_hint=True, on_conflict=False)
        elif self.running_opt == 3:            # CBS
            self.cbs_rout()
    
    def astar_rout(self):
        """A* 알고리즘을 사용하여 각 AGV의 경로를 계산하고 제어 신호 생성"""
        for num in self.agv_nums:
            pos = self.agv_pos.get(num)
            goal = self.agv_goal.get(num)
            planner = self.planners.get(num)

            if not all([pos, goal, planner]):
                continue

            # 목표에 도달하면 이동 멈춤 (제거는 gym_env에서 처리)
            if pos == goal:
                self.next_buffer[num] = (0, 0)
                self.control_buffer[num] = (0, 0)
                continue

            # A* 플래너의 시작점을 현재 위치로 업데이트하고 경로 재계산
            planner.start = pos
            planner.compute_shortest_path()
            path = planner.extract_path()
            self.agv_path[num] = path

            # 경로에 따라 다음 이동 방향 결정
            next_pos = path[1]
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            self.next_buffer[num] = (dx, dy)
            self.control_buffer[num] = (dx, dy)

    def pibt_rout(self, use_dstar_hint: bool = True, on_conflict: bool = False):
        """
        PIBT로 한 스텝 제어 벡터를 채운다.
        - use_dstar_hint=True: D*의 다음 칸을 후보 최우선 힌트로 사용
        - on_conflict=True: D* 제안대로 갔을 때 t+1 충돌이 예상되는 에이전트 '그룹'만 PIBT로 재결정
        """
        if not self.agv_nums:
            return

        # 스냅샷
        pos = {a: self.agv_pos[a] for a in self.agv_nums}
        goals = {a: self.agv_goal[a] for a in self.agv_nums}

        # 1) D*로 각자 다음 칸/경로 제안 (힌트로도 사용)
        dstar_next = {}
        dstar_hint = {}
        for a in self.agv_nums:
            p = pos[a]; g = goals[a]; planner = self.planners.get(a)
            if (p is None) or (g is None) or (planner is None):
                dstar_next[a] = p
                dstar_hint[a] = None
                self.agv_path[a] = [p]
                continue

            if p == g:
                dstar_next[a] = p
                dstar_hint[a] = None
                self.agv_path[a] = [p]
                continue

            planner.start = p
            planner.goal = g
            try:
                planner.compute_shortest_path()
                path = planner.extract_path()
            except Exception:
                path = [p]

            self.agv_path[a] = path
            nx = path[1] if path and len(path) > 1 else p
            dstar_next[a] = nx
            dstar_hint[a] = (nx if (use_dstar_hint and nx != p) else None)

        # 2) on_conflict 옵션: 충돌 예상 그룹만 PIBT 적용
        def _detect_conflict_set(pos_dict, nxt_dict):
            agents = list(nxt_dict.keys())
            bad = set()
            for i in range(len(agents)):
                a = agents[i]
                for j in range(i+1, len(agents)):
                    b = agents[j]
                    va, vb = nxt_dict[a], nxt_dict[b]
                    ua, ub = pos_dict[a], pos_dict[b]
                    # 1) 정점 충돌
                    if va == vb:
                        bad.update([a, b]); continue
                    # 2) 에지(스왑) 충돌
                    if va == ub and vb == ua:
                        bad.update([a, b]); continue
            return bad

        if on_conflict:
            group = _detect_conflict_set(pos, dstar_next)
            if not group:
                # 충돌 없으면 D* 제안대로 진행
                for a in self.agv_nums:
                    nx = dstar_next[a]
                    dx, dy = nx[0] - pos[a][0], nx[1] - pos[a][1]
                    self.next_buffer[a] = (dx, dy)
                    self.control_buffer[a] = (dx, dy)
                return
            # 그룹 밖은 고정 점유/엣지로 취급해 그룹만 PIBT 재결정
            non_group = set(self.agv_nums) - set(group)
            fixed_vertices = {dstar_next[b] for b in non_group}
            fixed_edges = {(pos[b], dstar_next[b]) for b in non_group}
            group_next = self.pibt.plan_one_step(
                pos=pos, goals=goals,
                dstar_hint={a: dstar_hint[a] for a in group},
                subset=group,
                fixed_vertices=fixed_vertices,
                fixed_edges=fixed_edges,
            )
            final_next = dict(dstar_next)
            final_next.update(group_next)
        else:
            # 전체를 PIBT로 한 스텝 결정 (D* 힌트는 후보 우선)
            final_next = self.pibt.plan_one_step(
                pos=pos, goals=goals, dstar_hint=dstar_hint
            )

        # 3) 제어 벡터 적용
        for a in self.agv_nums:
            nx = final_next.get(a, pos[a])
            dx, dy = nx[0] - pos[a][0], nx[1] - pos[a][1]
            self.next_buffer[a] = (dx, dy)
            self.control_buffer[a] = (dx, dy)

    def dstar_rout(self):
        """D* 알고리즘을 사용하여 각 AGV의 경로를 계산하고, 다른 AGV를 장애물로 취급하여 제어 신호 생성"""
        for num in self.agv_nums:
            pos = self.agv_pos.get(num)
            goal = self.agv_goal.get(num)
            planner = self.planners.get(num)

            if pos is None or goal is None:
                continue

            # 맵 복사 및 다른 AGV를 장애물로 표시
            # self.map이 numpy 배열 또는 2D 리스트라고 가정
            if hasattr(self.map, 'copy'):
                dynamic_map = self.map.copy()
            else:
                dynamic_map = copy.deepcopy(self.map)

            for other in self.agv_nums:
                if other != num:
                    ox, oy = self.agv_pos.get(other, (None, None))
                    if ox is not None and oy is not None:
                        try:
                            # numpy array or list
                            dynamic_map[oy][ox] = 0  # 0: obstacle
                        except Exception:
                            pass

            # 목표에 도달하면 이동 멈춤
            if pos == goal:
                self.next_buffer[num] = (0, 0)
                self.control_buffer[num] = (0, 0)
                continue

            # D* 플래너로 경로 계산
            self.planners[num] = AStar(dynamic_map, pos, goal)
            self.planners[num].compute_shortest_path()
            path = self.planners[num].extract_path()
            self.agv_path[num] = path

            if path and len(path) >= 2:
                next_pos = path[1]
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]
                self.next_buffer[num] = (dx, dy)
                self.control_buffer[num] = (dx, dy)
            else:
                self.next_buffer[num] = (0, 0)
                self.control_buffer[num] = (0, 0)

    def cbs_rout(self):
        """CBS로 한 스텝 제어 벡터를 채운다."""
        pass