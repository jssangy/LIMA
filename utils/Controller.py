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
        self.initial_paths = {}
        self.initial_visit_idx = {}

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
        
        self.initial_paths.clear()
        self.initial_visit_idx.clear()

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

        init_path = self.bfs.plan_path(start_pos, goal_pos) or [start_pos]
        self.initial_paths[agv_num] = init_path
        self.initial_visit_idx[agv_num] = 0

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

            self.initial_paths.pop(agv_num, None)
            self.initial_visit_idx.pop(agv_num, None)


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


    def cbs_initial_plan(self):
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
        timeout_seconds = 60.0  # 타임아웃 설정
        print('Trying to calculate within', timeout_seconds, 'seconds...')
        solution = self.cbs_planner.solve(time_limit=timeout_seconds)

        if solution:
            self.agv_full_paths = solution
            self.agv_path = solution.copy()
            self.cbs_planned_agents = set(agents_to_plan.keys())
            print(f"Solution: {solution}")
            print("CBS: Planning ended. New plan stored.")
        else:
            self.cbs_planned_agents.clear()
            print("CBS: Planning failed or timed out.")
        
        for num in self.agv_nums:
            self.agv_path_timestep[num] = 0

    def cbs_rout(self):
        if not self.agv_nums:
            return

        proposed_moves = {}
        for num in self.agv_nums:
            # AGV가 도중에 추가/제거될 수 있으므로, 경로가 없는 AGV는 현재 위치에 머무르도록 처리
            if num not in self.agv_full_paths:
                proposed_moves[num] = self.agv_pos.get(num)
                continue

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

        agents_to_stop = set()
        while True:
            newly_stopped_count = 0
            # 현재 활성화된 AGV 목록으로 충돌 검사
            active_agvs = [agv_id for agv_id in self.agv_nums if agv_id in self.cbs_planned_agents]
            for i in range(len(active_agvs)):
                for j in range(i + 1, len(active_agvs)):
                    agv1_id, agv2_id = active_agvs[i], active_agvs[j]
                    if agv1_id in agents_to_stop and agv2_id in agents_to_stop: continue
                    
                    pos1, pos2 = self.agv_pos[agv1_id], self.agv_pos[agv2_id]
                    next_pos1 = pos1 if agv1_id in agents_to_stop else proposed_moves[agv1_id]
                    next_pos2 = pos2 if agv2_id in agents_to_stop else proposed_moves[agv2_id]
                    
                    is_conflict = False
                    
                    # [수정] 충돌 감지 로직 개선
                    # 원인: 기존 로직은 '정지한 AGV'와 '이동하려는 AGV'의 충돌을 감지하지 못했음
                    # 1. 정점 충돌 (Vertex Conflict): 두 AGV가 다음 스텝에 '같은' 위치를 차지하려는 경우
                    if next_pos1 == next_pos2:
                        is_conflict = True
                    # 2. 교차(스와핑) 충돌 (Edge Conflict): 두 AGV가 서로의 위치를 맞바꾸려는 경우
                    elif next_pos1 == pos2 and next_pos2 == pos1:
                        is_conflict = True

                    if is_conflict:
                        if agv1_id not in agents_to_stop:
                            agents_to_stop.add(agv1_id); newly_stopped_count += 1
                        if agv2_id not in agents_to_stop:
                            agents_to_stop.add(agv2_id); newly_stopped_count += 1
            
            if newly_stopped_count == 0: break

        for num in self.agv_nums:
            if num in agents_to_stop:
                self.control_buffer[num] = (0, 0)
            else:
                pos = self.agv_pos.get(num)
                # proposed_moves에 없는 AGV(계획에 없던 AGV)는 제자리에 있도록 처리
                next_pos = proposed_moves.get(num, pos)
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]
                self.control_buffer[num] = (dx, dy)
                
                # 계획에 포함된 AGV만 타임스텝을 증가시켜 경로를 따라가도록 함
                if num in self.cbs_planned_agents:
                    self.agv_path_timestep[num] = self.agv_path_timestep.get(num, 0) + 1

    def bfs_rout(self):
        if not self.agv_nums:
            return

        for aid in self.agv_nums:
            pos  = self.agv_pos.get(aid)
            goal = self.agv_goal.get(aid)

            if pos is None or goal is None:
                self.agv_path[aid] = [pos] if pos is not None else []
                self.next_buffer[aid] = (0, 0)
                self.control_buffer[aid] = (0, 0)
                continue

            # 0) 초기 경로/포인터 보장
            ipath = self.initial_paths.get(aid)
            if not ipath:
                ipath = self.bfs.plan_path(pos, goal) or [pos]
                self.initial_paths[aid] = ipath
                self.initial_visit_idx[aid] = 0

            visit_idx = self.initial_visit_idx.get(aid, 0)
            visit_idx = max(0, min(visit_idx, len(ipath)-1))

            # 1) 현재가 "다음에 반드시 밟아야 할" 노드 위에 있으면 인덱스 전진
            if pos == ipath[visit_idx]:
                # 여러 칸 건너뛰어 붙었더라도 '정확히 그 노드'에 올랐을 때만 1칸 전진
                visit_idx = min(visit_idx + 1, len(ipath))  # len(ipath)까지 허용(끝 표시)
                self.initial_visit_idx[aid] = visit_idx

            # 2) 방문해야 할 노드가 더 없다면(=초기 경로 끝), goal 체크 후 정지
            if visit_idx >= len(ipath) or pos == goal:
                self.agv_path[aid] = [pos]
                self.next_buffer[aid] = (0, 0)
                self.control_buffer[aid] = (0, 0)
                continue

            # 3) 반드시 밟아야 할 다음 노드(target) 정의
            target = ipath[visit_idx]

            # 4) full_path 구성: pos→target 재합류 경로 + target 이후 초기경로 꼬리
            if abs(target[0]-pos[0]) + abs(target[1]-pos[1]) == 1:
                # 인접하면 바로 붙이기
                full_path = [pos] + ipath[visit_idx:]
            else:
                # 재합류 경로: target까지 BFS (실패시 뒤 인덱스로 완화)
                rejoin = self.bfs.plan_path(pos, target) or [pos]
                if len(rejoin) < 2:
                    # target이 벽/봉쇄 등으로 실패하면 뒤쪽에서 가장 가까운 reachable 노드 탐색
                    found = False
                    for j in range(visit_idx+1, len(ipath)):
                        rp = self.bfs.plan_path(pos, ipath[j]) or [pos]
                        if len(rp) >= 2:
                            tail = ipath[j+1:] if j+1 < len(ipath) else []
                            full_path = rp + tail
                            found = True
                            break
                    if not found:
                        # 최후: 그냥 goal까지
                        fallback = self.bfs.plan_path(pos, goal) or [pos]
                        full_path = fallback
                else:
                    tail = ipath[visit_idx+1:] if visit_idx+1 < len(ipath) else []
                    full_path = rejoin + tail  # rejoin 끝이 target이므로 target은 한 번만 포함

            # 5) 제어 벡터/경로 적용
            self.agv_full_paths[aid] = full_path
            if len(full_path) < 2:
                self.agv_path[aid] = [pos]
                self.next_buffer[aid] = (0, 0)
                self.control_buffer[aid] = (0, 0)
            else:
                self.agv_path[aid] = full_path
                next_pos = full_path[1]
                self.next_buffer[aid] = (next_pos[0]-pos[0], next_pos[1]-pos[1])
                self.control_buffer[aid] = self.next_buffer[aid]
