import os
import json
import math
import numpy as np
from typing import Dict

from utils.AMR import AMR
from utils.Intersection import Intersection
from utils import Funct
from utils.traffic_generator import TaskSetGenerator, discover_border_arms_NxM, TrafficGenerator
from utils.Controller import AStarPlanner, PIBTPlanner, CBSPlanner, BFSPlanner


class ENV():
    def __init__(self, prob_path, max_arm_len_h=5, max_arm_len_v=5, num_amrs=500, max_steps=1024, traffic_mode='task'):
        super().__init__()
        """환경 초기화"""
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])
        self.goal = set()

        self.time = 0
        
        self.map = self._load_map(map_path)
        self.walkable_tiles = np.count_nonzero(self.map == 0)
        self.max_arm_len_h = max_arm_len_h
        self.max_arm_len_v = max_arm_len_v
        processed_intersections = self._find_intersections_and_build_graph()
        
        self.time = 0
        self.amr_list = {}
        self.max_steps = max_steps

        self.planner = BFSPlanner(self.map)

        self.intersections: Dict[str, Intersection] = {}
        for iid, inter_info in processed_intersections.items():
            self.intersections[iid] = Intersection(
                inter_info['data'], 
                self.controller, 
                inter_info['neighbors'],
                inter_info['present_dirs'],
            )

        self.event_cells = set()
        self.cell2ix = {}
        for iid, I in self.intersections.items():
            center = (I.center_x, I.center_y)
            self.event_cells.add(center)
            self.cell2ix[center] = iid
            for d in I.dirs:
                coords = I.lane_coords[d]
                end_cell = coords[-1]
                self.event_cells.add(end_cell)
                self.cell2ix[end_cell] = iid

        self.deadlock_queue = []

        self.traffic_mode = traffic_mode

        # TaskGenerator
        if self.traffic_mode == 'task':
            self.task_generator = TaskSetGenerator(self.map, num_tasks=num_amrs, goal_positions=self.goal)

        # Traffic Generator
        elif self.traffic_mode == 'traffic':
            arms = discover_border_arms_NxM(self.intersections)
            self.traffic_generator = TrafficGenerator(arms)
            self.traffic_generator.set_arm_gate(lambda iid, d: self.is_arm_outgoing_clear(iid, d))

        # Color mapping
        self.color_map = Funct.Color_dict(6).dic

        self.use_scheduler = False

        self.completed_amr_steps = []

        self.completed_path_integrities: list[float] = []

        self.time_ms = []


    def reset(self):        
        self.time = 0
        self.amr_list.clear()
        
        if self.traffic_mode == 'task':
            self.task_generator.start_new_episode()
        elif self.traffic_mode == 'traffic':
            self.traffic_generator.start_new_episode()

        # 컨트롤러와 모든 교차로의 내부 상태 초기화
        self.controller.reset()
        for I in self.intersections.values():
            I.reset()
        self.deadlock_queue = []

        if self.traffic_mode == 'task':
            self._spawn_amrs_from_task_gen()
        elif self.traffic_mode == 'traffic':
            self._spawn_amrs_from_stream_gen()

        self.use_scheduler = False

        self.completed_amr_steps.clear()

        self.completed_path_integrities.clear()

        self.time_ms.clear()

        return


    def step(self):
        """
        actions: { "x{cx}y{cy}": action_idx, ... }
        반환: obs_next, reward_map, info_next
        """
        self.time += 1
        
        # 공통 종료 조건: 최대 스텝 도달
        if self.time >= self.max_steps:
            return False

        if self.use_scheduler:
            sorted_iids = sorted(self.intersections.keys(), key=self._inter_rank, reverse=True)

            # === 2) 교차로별 플래닝 (옵션 틱 → 액션 시작) ===
            for iid in sorted_iids:
                I = self.intersections[iid]
                if not I.is_deadlock:
                    continue

                I.begin_plan()
                I.resolve_arm_swaps_all()
                I.action_control()

            final_plan_moves = {}
            final_plan_prio = {}
            final_plan_order = {}
            final_plan_owner = {}

            for iid in sorted_iids:
                I = self.intersections[iid]
                for amr_id, prio in I._plan_prio.items():
                    prev_prio = final_plan_prio.get(amr_id, -10**9)
                    if prio >= prev_prio:
                        final_plan_prio[amr_id] = prio
                        final_plan_moves[amr_id] = I._plan_moves[amr_id]
                        final_plan_order[amr_id] = I._plan_order[amr_id]
                        final_plan_owner[amr_id] = iid

            self.controller.control_buffer.update(final_plan_moves)
            
            items = []
            for amr_id, (prio, *order) in final_plan_order.items():
                owner_iid = final_plan_owner[amr_id]
                rank = self._inter_rank(owner_iid)
                items.append((rank, -prio, tuple(order), amr_id))
            items.sort()
            
            seq = []
            seen = set()
            for _, _, _, aid in items:
                if aid not in seen:
                    seen.add(aid)
                    seq.append(aid)
            self.controller.push_sequence = seq

            # 3) Movement 커밋 (컨트롤러의 우선순위 큐 먼저 처리)
            moved = set()
            for amr_id in self.controller.push_sequence:
                amr_obj = self.amr_list.get(amr_id)
                if amr_obj:
                    sig = self.controller.control_buffer.get(amr_id, (0, 0))
                    if self._is_valid_move(amr_obj, sig):
                        amr_obj.move(sig)
                        amr_obj.action_count += 1
                        moved.add(amr_id)
            
            # 나머지 AMR 이동
            for amr_id, amr_obj in self.amr_list.items():
                if amr_id not in moved:
                    sig = self.controller.control_buffer.get(amr_id, (0, 0))
                    if self._is_valid_move(amr_obj, sig):
                        amr_obj.move(sig)

            # 사용 후 정리
            self.controller.push_sequence = []
        
        else:
            for amr_id, amr_obj in self.amr_list.items():
                sig = self.controller.control_buffer.get(amr_id, (0, 0))
                if self._is_valid_move(amr_obj, sig):
                    amr_obj.move(sig)

        # 4) 환경 변화 처리
        self._check_amr_completion()

        if self.traffic_mode == 'task':
            self._spawn_amrs_from_task_gen()
        elif self.traffic_mode == 'traffic':
            self._spawn_amrs_from_stream_gen()

        # GUI/테스트 모드: 기존 요약 반환 유지
        return self.make_info()
    

    def step(self):
        self.time += 1

        for amr_id, amr_obj in self.amr_list.items():
            if amr_obj.pos in self.event_cells:
                I = self.cell2ix[amr_obj.pos]
                


    def generate_observation(self):
        """
        관측 생성(부작용 없음):
        - sensing/make_control
        - 교차로 최신화(_update_intersections_state)
        - 상태/마스크/푸시아웃 플래그 계산만 수행
        """
        # 1) 센싱/컨트롤 업데이트
        for amr_id, amr_obj in self.amr_list.items():
            self.controller.get_sensing(amr_id, amr_obj.pos)
        self.controller.make_control()
        if self.time == 0:
            # 리셋 직후: AMR별 초기 경로 설정
            for amr_id, paths in self.controller.amr_path.items():
                self.amr_list[amr_id].set_initial_path(paths)

        # 2) 교차로 최신화 (여기서만 최신화! step에서는 하지 않음)
        self._update_intersections_state()

        # 3) deadlock 큐 갱신
        temp_info_for_queue = {iid: {"is_deadlock": I.is_deadlock} for iid, I in self.intersections.items()}
        self._update_deadlock_queue(temp_info_for_queue)

        # 4) 데드락/상태/마스크
        obs = {}
        info = {}

        for iid, I in self.intersections.items():
            state = np.asarray(I.get_state(), dtype=np.float32)
            
            action_mask = I.calculate_action_mask()
            action_mask = np.asarray(action_mask, dtype=np.bool_)

            obs[iid] = {
                "state": state,
            }
            info[iid] = {
                "is_deadlock": I.is_deadlock,
                "action_mask": action_mask,
                "macro_busy": (I.macro is not None),
            }

        return obs, info
    
    
    def _update_deadlock_queue(self, info):
        """
        데드락 큐(FIFO):
        - 새로 데드락이 '처음' 관찰되면 (iid, 발생시간) 을 맨 뒤에 추가
        - 데드락이 해소되면 해당 iid 항목을 제거
        - 더 이상 어떤 기준으로도 정렬하지 않음 (발생 시점 순서 유지)
        """
        # 현재 큐에 있는 iid 집합(빠른 조회용)
        iids_in_queue = {iid for iid, _ in self.deadlock_queue}

        for iid, meta in info.items():
            if not isinstance(meta, dict):
                continue

            is_deadlocked = bool(meta.get("is_deadlock", False))

            # 새 데드락: 큐에 없으면 append
            if is_deadlocked and iid not in iids_in_queue:
                self.deadlock_queue.append((iid, self.time))
                iids_in_queue.add(iid)

            # 해소: 큐에서 제거
            elif (not is_deadlocked) and iid in iids_in_queue:
                self.deadlock_queue = [item for item in self.deadlock_queue if item[0] != iid]

                iids_in_queue.discard(iid)


    def _inter_rank(self, iid):
        """
        [수정] 데드락 큐의 구조 변경에 맞춰 iid의 순위를 반환합니다.
        """
        try:
            # 큐는 (iid, timestamp) 튜플의 리스트이므로, iid만 추출하여 인덱스를 찾음
            iids_only = [item[0] for item in self.deadlock_queue]
            return iids_only.index(iid)
        except ValueError:
            return math.inf
    
    
    def _update_intersections_state(self):
        for I in self.intersections.values():
            I.soft_reset()

        for amr_id, amr_obj in self.amr_list.items():
            pos = amr_obj.pos
            for I in self.intersections.values():
                if pos in I.all_lane_coords:
                    I.add_amr(amr_obj)

        for I in self.intersections.values():
            I.check_deadlock()
            if I.macro is not None and not I.is_deadlock:
                I.macro = None


    def _spawn_amrs_from_task_gen(self):
        """
        [이름 변경 및 Task 모드 전용]
        TaskSetGenerator로부터 새로운 AMR을 받아 환경에 추가.
        """
        gen = self.task_generator
        if not gen or not gen.should_spawn_next():
            return

        new_tasks = gen.get_next_task_pair(current_time=self.time)
        
        for task in new_tasks:
            amr_id = task['id']

            start_pos = tuple(task['start_pos'])
            goal_pos = tuple(task['goal_pos'])

            new_amr = AMR(amr_id, start_pos, goal_pos, self.color_map[amr_id % 6])
            self.amr_list[amr_id] = new_amr
        
        self.planner.plan_for_new_amrs(self.amr_list)


    def _check_amr_completion(self):
        completed_amrs = []
        for amr_id, amr_obj in list(self.amr_list.items()):
            if amr_obj.pos == self.controller.amr_goal.get(amr_id):
                completed_amrs.append(amr_id)

        for amr_id in completed_amrs:
            amr_obj = self.amr_list[amr_id]
            if amr_obj is not None:
                pi_pct = amr_obj.path_integrity_ratio()
                self.completed_path_integrities.append(pi_pct)
                self.completed_amr_steps.append(amr_obj.steps)
            if self.traffic_mode == 'task':
                self.task_generator.complete_task(amr_id)
            elif self.traffic_mode == 'traffic':
                self.traffic_generator.complete_task(amr_id)
            del self.amr_list[amr_id]
            self.controller.remove_amr(amr_id)


    def _spawn_amrs_from_stream_gen(self):
        """
        [새로 추가된 함수 - Traffic 모드 전용]
        TrafficGenerator로부터 새로운 AMR을 받아 환경에 추가.
        """
        gen = self.traffic_generator
        if not gen or not gen.should_spawn_next():
            return
        
        # TrafficGenerator12는 current_time 인자가 없음
        new_tasks = gen.get_next_task_pair()

        for task in new_tasks:
            amr_id = task['id']
            start_iid = task['intersection_id']
            start_dir = task['start_direction']
            goal_iid = task['goal_intersection_id']
            goal_dir = task['goal_direction']

            start_pos = self._direction_to_coords(start_dir, start_iid)
            goal_pos = self._direction_to_coords(goal_dir, goal_iid)

            if start_pos is None or goal_pos is None:
                continue
            
            # AMR 생성 및 등록 (amr 생성자 인자 순서 수정)
            new_amr = AMR(amr_id, start_pos, goal_pos, self.color_map[amr_id % 6])
            self.amr_list[amr_id] = new_amr
        
        self.planner.plan_for_new_amrs(self.amr_list)


    def _direction_to_coords(self, direction, intersection_ref):
        """
        direction: 'N'|'E'|'S'|'W'
        intersection_ref: 교차로 id 문자열("x{cx}y{cy}") 또는 (cx,cy,lenN,lenE,lenS,lenW) 튜플 모두 허용
        """
        # 1) iid 문자열 → Intersection에서 스펙 가져오기
        if isinstance(intersection_ref, str):
            I = self.intersections[intersection_ref]
            # outer_entry_cells 같은 사전이 있으면 그걸 우선 사용
            if hasattr(I, "outer_entry_cells") and direction in I.outer_entry_cells:
                return I.outer_entry_cells[direction]
            center_x, center_y = I.center_x, I.center_y
            len_N, len_E, len_S, len_W = I.len_N, I.len_E, I.len_S, I.len_W

        # 2) 과거 호환: 스펙 튜플로 온 경우
        else:
            center_x, center_y, len_N, len_E, len_S, len_W = intersection_ref

        direction_map = {
            'N': (center_x, center_y - len_N - 1),
            'E': (center_x + len_E + 1, center_y),
            'S': (center_x, center_y + len_S + 1),
            'W': (center_x - len_W - 1, center_y),
        }
        return direction_map[direction]

    
    def _load_map(self, map_path):        
        if not os.path.isfile(map_path): raise FileNotFoundError(f"Map file not found: {map_path}")
        map_data = []
        with open(map_path, 'r') as f: lines = f.readlines()
        map_start = None
        for idx, line in enumerate(lines):
            if line.strip() == 'map': map_start = idx + 1; break
        if map_start is None: raise ValueError("Map data not found in file")
        for line in lines[map_start:]:
            row = []
            for c in line.strip():
                if c in ['@', 'T']: row.append(1)
                elif c in ['.', 'E', 'S']: 
                    row.append(0)
                    if c == "S":
                        self.goal.add((len(row)-1, len(map_data)))  # (x,y)
                else: raise ValueError(f"Invalid character in map file: {c}")
            if row: map_data.append(row)
        return np.array(map_data)

    def _find_intersection_center(self):
        # 3x3 패턴들: 0=도로, 1=벽
        plus4 = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])

        # T자 (팔 하나 없는 방향)
        t_noN = np.array([  # 위쪽 팔 없음 (E/W/S만 열림)
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])
        t_noE = np.array([  # 오른쪽 팔 없음 (N/W/S만 열림)
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 1]
        ])
        t_noS = np.array([  # 아래쪽 팔 없음 (N/E/W만 열림)
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1]
        ])
        t_noW = np.array([  # 왼쪽 팔 없음 (N/E/S만 열림)
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 1]
        ])

        kernels = (plus4, t_noN, t_noE, t_noS, t_noW)

        # 3x3 슬라이딩 윈도우
        windows = np.lib.stride_tricks.sliding_window_view(self.map, (3, 3))
        # 각 커널에 대해 매칭 후 OR 합치기
        match_any = np.zeros(windows.shape[:2], dtype=bool)
        for K in kernels:
            match_any |= np.all(windows == K, axis=(2, 3))

        # 윈도우 좌표 → 중심 좌표(슬라이딩 오프셋 +1)
        centers = (np.argwhere(match_any) + 1).tolist()
        return centers
        
    def _ray_len(self, r, c, dr, dc, max_len=None):
        H, W = self.map.shape
        length = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < H and 0 <= cc < W and self.map[rr][cc] == 0:
            if dr != 0:
                left_wall  = (cc - 1 < 0) or (self.map[rr][cc - 1] == 1)
                right_wall = (cc + 1 >= W) or (self.map[rr][cc + 1] == 1)
                if not (left_wall and right_wall): break
            else:
                up_wall   = (rr - 1 < 0) or (self.map[rr - 1][cc] == 1)
                down_wall = (rr + 1 >= H) or (self.map[rr + 1][cc] == 1)
                if not (up_wall and down_wall): break

            length += 1
            if max_len is not None and length >= max_len:
                break
            rr += dr
            cc += dc
        return length
    
    def _find_intersections_and_build_graph(self):
        centers_rc = self._find_intersection_center()
        centers_xy = [(c, r) for r, c in centers_rc]

        center_xy_to_data = {}
        for c, r in centers_xy:
            len_N = self._ray_len(r, c, -1, 0, max_len=self.max_arm_len_v)
            len_S = self._ray_len(r, c,  1, 0, max_len=self.max_arm_len_v)
            len_E = self._ray_len(r, c,  0, 1, max_len=self.max_arm_len_h)
            len_W = self._ray_len(r, c,  0,-1, max_len=self.max_arm_len_h)

            # ★ 사거리/삼거리 허용: 팔이 3개 이상 존재해야 교차로 인정
            present = {d for d, L in zip("NESW", [len_N, len_E, len_S, len_W]) if L > 0}
            if len(present) >= 3:
                center_xy_to_data[(c, r)] = (c, r, len_N, len_E, len_S, len_W, present)

        processed_intersections = {}
        for (c, r), tup in center_xy_to_data.items():
            c, r, len_N, len_E, len_S, len_W, present = tup
            current_iid = f'x{c}y{r}'

            # ★ 있는 팔만 이웃 계산
            neighbors_map = {}
            if 'N' in present:
                t = (c, r - len_N - 1)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['N'] = f'x{nc}y{nr}'
            if 'E' in present:
                t = (c + len_E + 1, r)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['E'] = f'x{nc}y{nr}'
            if 'S' in present:
                t = (c, r + len_S + 1)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['S'] = f'x{nc}y{nr}'
            if 'W' in present:
                t = (c - len_W - 1, r)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['W'] = f'x{nc}y{nr}'

            processed_intersections[current_iid] = {
                'data': (c, r, len_N, len_E, len_S, len_W),
                'neighbors': neighbors_map,
                # ↓ 이후 단계에서 마스크/상태 0패딩에 쓰기 좋게 전달
                'present_dirs': present,
            }
        return processed_intersections

    
    def is_arm_outgoing_clear(self, iid: str, d: str) -> bool:
        I = self.intersections[iid]

        # (선택) 삼거리 대응: 존재하지 않는 팔은 금지
        present = getattr(I, "present_dirs", set(I.lane_coords.keys()))
        if d not in present:
            return False

        # 1) 해당 팔에 '바깥으로 나가려는 흐름'이 있으면 금지
        has_outgoing = bool(getattr(I, "outgoing", {}).get(d, False))
        if has_outgoing:
            return False

        # 2) 교차로 데드락인 경우 금지  ← deadlock_queue 정규화
        dq = self.deadlock_queue or []
        if dq and isinstance(dq[0], tuple):
            dead_iids = {x for (x, _) in dq}
        else:
            dead_iids = set(dq)
        if iid in dead_iids:
            return False

        # (권장) 3) 팔 팁(outer entry)이 도로이고 비어있는지 확인
        if hasattr(I, "outer_entry_cells") and d in I.outer_entry_cells:
            tip = I.outer_entry_cells[d]
        else:
            cx, cy = I.center_x, I.center_y
            if   d == "N": tip = (cx, cy - I.len_N - 1)
            elif d == "E": tip = (cx + I.len_E + 1, cy)
            elif d == "S": tip = (cx, cy + I.len_S + 1)
            else:          tip = (cx - I.len_W - 1, cy)

        H, W = self.map.shape
        tx, ty = tip
        if not (0 <= tx < W and 0 <= ty < H):
            return False
        if self.map[ty][tx] == 1:
            return False
        if any(a.pos == tip for a in self.amr_list.values()):
            return False

        return True

    
    def _update_and_check_stagnation(self) -> bool:
        """
        최근 전역 위치 시그니처를 바탕으로 정지/진동을 감지.
        True면 조기 종료해야 함.
        """
        if self.time < self._stg_min_time:
            self._sig_hist.clear()
            return False
        if not self.amr_list:
            self._sig_hist.clear()
            return False

        # 전역 시그니처: (amr_id, x, y) 튜플을 정렬한 튜플
        sig = tuple(sorted((aid, amr.pos[0], amr.pos[1]) for aid, amr in self.amr_list.items()))
        self._sig_hist.append(sig)

        # 1) 정지: 최근 N개가 모두 동일
        idle = False
        if len(self._sig_hist) >= self._stg_idle_win:
            lastN = list(self._sig_hist)[-self._stg_idle_win:]
            idle = all(s == lastN[0] for s in lastN)

        # 2) 진동: 최근 M개가 ABABAB 형태(두 개의 시그니처가 번갈아)
        osc = False
        if len(self._sig_hist) >= self._stg_osc_win:
            w = self._stg_osc_win
            lastM = list(self._sig_hist)[-w:]
            if lastM[0] != lastM[1]:
                osc = all(lastM[i] == lastM[i % 2] for i in range(w))

        if idle or osc:
            return True

        return False
    

    # --- [GUI 연동을 위한 어댑터 함수들] ---
    def Get_AMR(self):
        """GUI가 AMR 목록을 가져갈 수 있도록 하는 함수"""
        return self.amr_list

    def get_active_tasks(self):
        """GUI가 AMR의 목표 지점을 가져갈 수 있도록 하는 함수"""
        return self.controller.amr_goal

    def make_info(self):
        """
        [수정] GUI에 필요한 모든 정보를 계산하여 반환합니다.
        'task' 모드와 'traffic' 모드를 명시적으로 구분하여 처리합니다.
        """
        # --- 2. 모드에 따라 통계 정보 계산 ---
        if self.traffic_mode == 'traffic':
            progress = self.traffic_generator.get_progress()
        elif self.traffic_mode == 'task':
            progress = self.task_generator.get_progress()
        completed_tasks = progress.get('completed_total', 0)
        total_tasks = progress.get('spawned_total', 0)

        # Success Rate 계산
        success_rate = (completed_tasks / total_tasks) if total_tasks > 0 else 0.0
        
        # 스루풋 계산 (분 단위)
        throughput = (completed_tasks / self.time * 60) if self.time > 0 else 0.0

        active_pi = []
        for amr_obj in self.amr_list.values():
            active_pi.append(amr_obj.path_integrity_ratio())
        all_pi = self.completed_path_integrities + active_pi
        avg_pi = float(np.mean(all_pi)) if all_pi else 0.0

        if self.use_scheduler:
            avg_ms = float(np.mean(self.time_ms)) if self.time_ms else 0.0
        else:
            avg_ms = float(np.mean(self.controller.time_ms)) if self.controller.time_ms else 0.0

        # --- 3. 현재 활성화된 AMR들의 상세 정보 수집 ---
        active_amr_details = {}
        for amr_id, amr_obj in self.amr_list.items():
            active_amr_details[amr_id] = {
                "steps": amr_obj.steps,
            }

        # --- 4. 최종 정보 취합하여 반환 ---
        return {
            "success_rate": success_rate,
            "throughput": throughput,
            "active_amrs": active_amr_details,
            "avg_path_integrity": avg_pi,
            "avg_inference_time": avg_ms,
            "time": self.time,
        }
