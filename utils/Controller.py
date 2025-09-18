import copy
import numpy as np
from collections import defaultdict

from utils.global_planning import AStar, PIBT, CBS, BFS

class controller():    
    def __init__(self, map_data, running_opt=0):
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

        self.running_opt = running_opt
        self.pibt = PIBT(self.map)
        self.pibt_bump = defaultdict(int)
        self.bfs = BFS(self.map)


        self.cbs_planner = None
        self.agv_full_paths = {}
        self.agv_path_timestep = {}

        self.cbs_plan_generated = False
        self.cbs_planned_agents = set()  # [추가] 마지막으로 계획을 세운 AGV 목록


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
        self.pibt_bump.clear()

        self.agv_full_paths.clear()
        self.agv_path_timestep.clear()

        self.cbs_plan_generated = False
        self.cbs_planned_agents.clear() # [추가] 리셋 시 초기화

    def add_agv(self, agv_num, start_pos, goal_pos):
        """AGV를 컨트롤러에 동적으로 추가"""
        self.agv_nums.append(agv_num)
        self.agv_pos[agv_num] = start_pos
        self.agv_goal[agv_num] = goal_pos
        self.next_buffer[agv_num] = (0, 0)
        self.control_buffer[agv_num] = (0, 0)
        self.planners[agv_num] = AStar(self.map, start_pos, goal_pos)
        self.agv_path[agv_num] = []

        self.agv_path_timestep[agv_num] = 0

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


            self.agv_full_paths.pop(agv_num, None)
            self.agv_path_timestep.pop(agv_num, None)


    def get_sensing(self, agv_num, pos):
        """Env로부터 AGV의 현재 위치를 업데이트"""
        if agv_num in self.agv_nums:
            self.agv_pos[agv_num] = pos

    def make_control(self):
        """모든 활성 AGV에 대한 제어 신호 생성"""

        # if self.running_opt == 3 and not self.cbs_plan_generated:
        #     # CBS 모드인데 아직 계획이 없다면, 초기 계획을 실행
        #     self.cbs_initial_plan()
        #     self.cbs_plan_generated = True # 계획이 생성되었음을 표시
        if self.running_opt == 0:           # BFS
            self.bfs_rout()
        elif self.running_opt == 1:              # A*
            self.astar_rout()
        elif self.running_opt == 2:            # D*
            self.dstar_rout()
        elif self.running_opt == 3:            # PIBT
            self.pibt_rout(use_dstar_hint=True, on_conflict=False)
        elif self.running_opt == 4:            # CBS
            current_agents = set(self.agv_nums)
            # if current_agents != self.cbs_planned_agents:
            if not current_agents.issubset(self.cbs_planned_agents):
                # AGV 목록에 변화가 생겼으므로 재계획 수행
                self.cbs_initial_plan()
    
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
            if nx != dstar_next.get(a, pos[a]):
                self.pibt_bump[a] += 1

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



    # --- [추가] CBS ---
    def cbs_initial_plan(self):
        """
        CBS 플래너를 실행하여 모든 에이전트의 전체 경로를 미리 계산.
        시뮬레이션 시작 전에 한 번만 호출되어야 함.
        """

        # [추가] 1. 계획 시작을 알리는 로그
        print(f"============== CBS Initial Plan START ==============")

        if not self.agv_nums:
            print("CBS: No AGVs to plan for.")
            return

        print(f"CBS: Starting planning for {len(self.agv_nums)} agents.")

        agents_to_plan = {
            num: {'start': self.agv_pos[num], 'goal': self.agv_goal[num]}
            for num in self.agv_nums
        }

        self.cbs_planner = CBS(self.map, agents_to_plan)
        timeout_seconds = 10.0
        print('Trying to calculate within', timeout_seconds, 'seconds...')
        solution = self.cbs_planner.solve(time_limit=timeout_seconds)
        # solution = self.cbs_planner.solve()

        # [수정] 계획 성공 여부에 따라 planned_agents 목록을 업데이트
        if solution:
            self.agv_full_paths = solution
            self.agv_path = solution.copy()
            # 계획에 사용된 AGV 목록을 저장
            self.cbs_planned_agents = set(agents_to_plan.keys())
            print(f"Solution: {solution}")
            print("CBS: Planning ended. New plan stored.")
        else:
            # 계획 실패 시, 다음 스텝에서 다시 시도하도록 목록을 비워둠
            self.cbs_planned_agents.clear()
            print("CBS: Planning failed or timed out.")
        
        # self.agv_full_paths = solution
        # self.agv_path = solution.copy()
        # print(f"Solution: {solution}")
        # print("CBS: Planning ended. New plan stored.")
        # 모든 에이전트의 타임스텝 초기화
        for num in self.agv_nums:
            self.agv_path_timestep[num] = 0

    def cbs_rout(self):
            """
            미리 계산된 CBS 경로를 따르되, 다음 스텝의 충돌을 예측하여 선제적으로 정지시키는 로직을 포함.
            """
            if not self.agv_nums:
                return

            # --- 1단계: 모든 AGV의 '예상 다음 위치' 계산 ---
            proposed_moves = {}
            for num in self.agv_nums:
                pos = self.agv_pos.get(num)
                path = self.agv_full_paths.get(num)
                
                if not path:
                    proposed_moves[num] = pos
                    continue

                t = self.agv_path_timestep.get(num, 0)
                if t + 1 < len(path):
                    proposed_moves[num] = path[t + 1]
                else:
                    proposed_moves[num] = pos

            # --- 2단계 & 3단계: 충돌을 반복적으로 감지하여 '정지할 AGV' 목록 완성 ---
            agents_to_stop = set()
            # (이 부분은 이전과 동일하게 유지합니다)
            while True:
                newly_stopped_count = 0
                for i in range(len(self.agv_nums)):
                    for j in range(i + 1, len(self.agv_nums)):
                        agv1_id, agv2_id = self.agv_nums[i], self.agv_nums[j]
                        if agv1_id in agents_to_stop and agv2_id in agents_to_stop: continue
                        
                        pos1, pos2 = self.agv_pos[agv1_id], self.agv_pos[agv2_id]
                        next_pos1 = pos1 if agv1_id in agents_to_stop else proposed_moves[agv1_id]
                        next_pos2 = pos2 if agv2_id in agents_to_stop else proposed_moves[agv2_id]
                        
                        is_conflict = False
                        if next_pos1 == next_pos2 and next_pos1 != pos1 and next_pos1 != pos2: is_conflict = True
                        elif next_pos1 == pos2 and next_pos2 == pos1: is_conflict = True

                        if is_conflict:
                            if agv1_id not in agents_to_stop:
                                agents_to_stop.add(agv1_id); newly_stopped_count += 1
                            if agv2_id not in agents_to_stop:
                                agents_to_stop.add(agv2_id); newly_stopped_count += 1
                
                if newly_stopped_count == 0: break

            # --- 4단계: '정지 목록'을 기반으로 최종 제어 신호 생성 ---
            for num in self.agv_nums:
                if num in agents_to_stop:
                    # [수정] 충돌이 감지된 AGV는 멈추고, 타임스텝을 '증가시키지 않음'
                    # (다음 스텝에 동일한 이동을 다시 시도해야 하므로)
                    self.control_buffer[num] = (0, 0)
                else:
                    # [수정] 충돌 없는 AGV는 계획(이동 또는 대기)을 따르고, 타임스텝을 '무조건 증가시킴'
                    pos = self.agv_pos.get(num)
                    next_pos = proposed_moves[num]
                    dx = next_pos[0] - pos[0]
                    dy = next_pos[1] - pos[1]
                    self.control_buffer[num] = (dx, dy)
                    
                    # 계획된 'wait'도 경로의 한 스텝을 완료한 것이므로 타임스텝을 증가시켜야 함
                    self.agv_path_timestep[num] += 1

    def bfs_rout(self):
        """
        BFS로 모든 AGV의 '전체 경로'를 한 번에 재계산/갱신하고
        A*와 동일한 방식으로 다음 스텝 제어 신호를 만든다.
        - self.agv_full_paths[aid] = [cur, ..., goal] (풀 경로)
        - self.agv_path[aid]       = [cur, ..., goal] (Intersection 등에서 참고)
        - self.control_buffer/next_buffer에 (dx, dy) 세팅
        """
        if not self.agv_nums:
            return

        # 1) 전체 경로를 한 번에 계산 (플래너에 일괄 API가 없으면 개별 호출로 폴백)
        if hasattr(self.bfs, "plan_all_paths"):
            paths = self.bfs.plan_all_paths(self.agv_pos, self.agv_goal)  # {aid: [(x,y), ...]}
        else:
            paths = {}
            for aid in self.agv_nums:
                s = self.agv_pos.get(aid)
                g = self.agv_goal.get(aid)
                if s is None or g is None:
                    paths[aid] = [s] if s is not None else []
                else:
                    paths[aid] = self.bfs.plan_path(s, g)

        # 2) 각 AGV의 풀 경로 갱신 및 다음 이동 결정
        for aid in self.agv_nums:
            pos  = self.agv_pos.get(aid)
            goal = self.agv_goal.get(aid)
            path = paths.get(aid, [])

            # 풀 경로 캐시
            self.agv_full_paths[aid] = path

            # Intersection 등 호환을 위해 A*와 동일하게 "전체 경로"를 저장
            # (빈 경로 방지: 최소 [현재칸] 보장)
            if not path:
                safe_path = [pos] if pos is not None else []
                self.agv_path[aid] = safe_path
            else:
                self.agv_path[aid] = path

            # 목표 도달 또는 경로가 한 칸뿐이면 정지
            if pos is None or goal is None or pos == goal or len(self.agv_path[aid]) < 2:
                self.next_buffer[aid] = (0, 0)
                self.control_buffer[aid] = (0, 0)
                continue

            # 다음 스텝(= path[1])으로 제어 신호 생성
            next_pos = self.agv_path[aid][1]
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            self.next_buffer[aid] = (dx, dy)
            self.control_buffer[aid] = (dx, dy)
