import os
import json
import math
import numpy as np
from typing import Dict
from collections import defaultdict, deque
import heapq

from utils.AMR import AMR
from utils.Intersection import Intersection
from utils import Funct
from utils.traffic_generator import discover_border_arms_NxM, TrafficGenerator, TrafficGenerator12, TaskSetGenerator
from utils.Controller import AStarPlanner, PIBTPlanner, CBSPlanner, BFSPlanner


class ENV():
    def __init__(self, prob_path):
        super().__init__()

        # Load map and Intersections
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])        
        self.map = self._load_map(map_path)
        self.walkable_tiles = np.count_nonzero(self.map == 0)
        print(f"Map loaded. Walkable tiles (value 0): {self.walkable_tiles}")
        processed_intersections = self._find_intersections_and_build_graph()
        
        self.time = 0
        self.amr_list: Dict[int, AMR] = {}
        self.max_steps = 1000

        self.planner = BFSPlanner(self.map)
        
        self.intersections: Dict[str, Intersection] = {}
        for iid, inter_info in processed_intersections.items():
            self.intersections[iid] = Intersection(
                inter_info['data'], 
                inter_info['neighbors'],
            )

        self.deadlock_queue = []
        arms = discover_border_arms_NxM(self.intersections)
        self.traffic_gen_stream = TrafficGenerator12(arms12=arms)
        self.traffic_gen_task_set = TaskSetGenerator(all_arms=arms)
        self.traffic_mode = 'traffic'
        self.traffic_generator = self.traffic_gen_stream
        self.traffic_generator.set_arm_gate(lambda iid, d: self.is_arm_outgoing_clear(iid, d))

        # Color mapping
        self.color_map = Funct.Color_dict(100).dic
        self.prev_deadlock_map: dict[str, bool] = {}
        self.use_rl = False
        self.rl_policy = None
        self.completed_amr_steps = []

    def reset(self):        
        self.time = 0
        self.amr_list.clear()
        self.traffic_generator.start_new_episode()
        for I in self.intersections.values(): I.reset()
        self.deadlock_queue = []

        self._spawn_new_amrs()

        # 리셋 시에는 초기 관찰 상태만 반환
        obs, info = self.generate_observation()
        self.prev_deadlock_map: dict[str, bool] = {}
        self.completed_amr_steps.clear()

        return obs, info

    def step(self, actions=None, train=True):
        """
        actions: { "x{cx}y{cy}": action_idx, ... }
        반환: obs_next, reward_map, info_next
        """
        self.time += 1
        if actions is None: actions = {}

        # 1. 관측 생성 및 AMR 기본 계획 수립
        #    - generate_observation 내부에서 각 AMR의 update_next_buffer가 호출됨
        obs_now, info_now = self.generate_observation()

        # 2. 교차로 개입 및 AMR 우선순위 설정
        act_to_apply = self._get_actions_to_apply(actions, obs_now, info_now, train)

        # 우선순위가 낮은 교차로부터 순회하며 계획을 수정하고 높은 교차로가 계획을 수정할 수 있도록 하고 관련 AMR의 우선순위를 갱신
        reverse_sorted_iids = sorted(self.intersections.keys(), key=self._inter_rank, reverse=True)
        for i, iid in enumerate(reverse_sorted_iids):
            I = self.intersections[iid]
            I.resolve_all_conflicts(priority=i)
            if iid in act_to_apply:
                I.action_control(act_to_apply[iid], priority=i)

        # 3. 최종 이동 계획 생성 및 실행 (점유 테이블 기반 충돌 해결)
        self.execute_moves(reverse_sorted_iids[::-1])

        # 4. 환경 변화 처리 (AMR 완료 체크 및 제거, 새로운 AMR 스폰)
        self._check_amr_completion()
        self._spawn_new_amrs()

        # GUI/테스트 모드: 기존 요약 반환 유지
        if not train:
            return self.make_info()

        # 5. 다음 관측 및 보상 계산
        obs_next, info_next = self.generate_observation()
        reward_map, info_next = self._calculate_rewards_and_update_info(info_next)

        return obs_next, reward_map, info_next

    def generate_observation(self):
        """
        관측 생성:
        - sensing/make_control
        - 교차로 최신화(_update_intersections_state)
        - 상태/마스크/푸시아웃 플래그 계산만 수행
        """
        # 1. 교차로 상태 최신화
        self._update_intersections_state()

        # 2. 데드락 큐 및 최종 관측 정보 생성
        obs, info = {}, {}
        temp_info_for_queue = {iid: {"is_deadlock": I.is_deadlock} for iid, I in self.intersections.items()}
        self._update_deadlock_queue(temp_info_for_queue)

        for iid, I in self.intersections.items():
            state = np.asarray(I.get_state(), dtype=np.float32)
            action_mask = np.asarray(I.calculate_action_mask(self.deadlock_queue), dtype=np.bool_)
            obs[iid] = {"state": state}
            info[iid] = {"is_deadlock": I.is_deadlock, "action_mask": action_mask}
            
        return obs, info
    
    def _get_actions_to_apply(self, actions, obs_now, info_now, train):
        """
        [수정] 'center_amr' 대신 'amr_intent_map'을 확인하여 중앙 AMR 존재 여부를 판단합니다.
        """
        act_to_apply = dict(actions)
        if (not train) and self.use_rl and self.rl_policy:
            for iid, meta in info_now.items():
                # 중앙에 AMR이 있는지 확인하는 로직 수정
                intersection = self.intersections[iid]
                has_center_amr = any(data['current_arm'] == 'C' for data in intersection.amr_intent_map.values())

                if isinstance(meta, dict) and meta.get("is_deadlock", False) and has_center_amr and iid not in act_to_apply:
                    action_mask = meta.get("action_mask")
                    # obs_now[iid]는 state, action_mask 등을 포함한 딕셔너리이므로 state만 전달
                    rl_action = int(self.rl_policy(obs_now[iid], action_mask))
                    act_to_apply[iid] = rl_action
        return act_to_apply
    
    def execute_moves(self, sorted_iids=None):
        """
        [전면 수정] 제어 그룹을 먼저 이동시킨 후, 자율 그룹을 이동시키는 순차적 방식 적용
        """
        # --- 1. AMR 그룹 분리 ---
        controlled_amrs = []
        free_amrs = []
        for amr in self.amr_list.values():
            if amr.priority > 0:
                controlled_amrs.append(amr)
            else:
                free_amrs.append(amr)

        # --- 2. 교차로 제어 그룹 이동 결정 및 실행 ---
        controlled_signals = {}
        occupied_pos = set()
        
        sorted_controlled_amrs = sorted(controlled_amrs, key=lambda amr: amr.priority, reverse=True)
        pos_to_amr_id = {amr.pos: amr.id for amr in self.amr_list.values()}

        for amr_obj in sorted_controlled_amrs:
            signal = amr_obj.control_buffer
            next_pos = (amr_obj.pos[0] + signal[0], amr_obj.pos[1] + signal[1])

            if next_pos in pos_to_amr_id:
                other_amr = self.amr_list[pos_to_amr_id[next_pos]]
                if other_amr in sorted_controlled_amrs:
                    other_next_pos = (other_amr.pos[0] + other_amr.control_buffer[0], other_amr.pos[1] + other_amr.control_buffer[1])
                    if other_next_pos == amr_obj.pos:
                        controlled_signals[amr_obj.id] = (0, 0)
                        occupied_pos.add(amr_obj.pos)
                        continue
            
            if next_pos in occupied_pos:
                controlled_signals[amr_obj.id] = (0, 0)
                occupied_pos.add(amr_obj.pos)
                continue

            controlled_signals[amr_obj.id] = signal
            occupied_pos.add(next_pos)

        for amr_obj in controlled_amrs:
            signal_to_move = controlled_signals.get(amr_obj.id, (0, 0))
            amr_obj.move(signal_to_move)

        # --- 3. 자율 주행 그룹 이동 결정 및 실행 ---
        # [수정] 데드락 교차로 진입 방지 로직 (set 처리)
        for amr_obj in free_amrs:
            signal = amr_obj.control_buffer
            next_pos = (amr_obj.pos[0] + signal[0], amr_obj.pos[1] + signal[1])

            # 규칙 1: 다른 AMR의 현재 위치와 충돌하는가?
            is_collision = False
            for other_amr in self.amr_list.values():
                if other_amr is not amr_obj and next_pos == other_amr.pos:
                    is_collision = True
                    break
            if is_collision:
                amr_obj.move((0, 0))
                continue

            # 규칙 2: 데드락 교차로 진입을 방지하는가?
            should_stop = False
            target_intersection_id = None
            for iid, I in self.intersections.items():
                if next_pos in I.all_lane_coords:
                    target_intersection_id = iid
                    break
            
            if target_intersection_id:
                # [수정] 현재 AMR이 속한 교차로들의 랭크 중 가장 높은 랭크(가장 낮은 숫자)를 선택
                current_iids = amr_obj.current_intersection_id
                if not current_iids:
                    current_rank = math.inf
                else:
                    current_rank = min(self._inter_rank(iid) for iid in current_iids)

                # 목표 교차로의 랭크
                target_rank = self._inter_rank(target_intersection_id)

                # 목표 교차로의 우선순위가 더 높으면(랭크 숫자가 더 작으면) 정지
                if target_rank < current_rank:
                    should_stop = True

            if should_stop:
                amr_obj.move((0, 0))
            else:
                amr_obj.move(signal)

    def _calculate_rewards_and_update_info(self, info_next):
        """
        [수정]
        보상 맵과 수정된 info 딕셔너리를 모두 반환합니다.
        """
        reward_map = {}
        for iid, meta in info_next.items():
            if not isinstance(meta, dict):
                continue

            curr = bool(meta.get("is_deadlock", False))
            prev = self.prev_deadlock_map.get(iid, False)

            # 1. 보상 계산
            r = -0.05 if curr else 0.0
            if prev and not curr:
                r += 1.0
            reward_map[iid] = r

            # 2. 최종 정보 업데이트
            meta["event_start"] = (not prev) and curr
            meta["event_end"] = prev and (not curr)

        # 3. 다음 스텝을 위해 prev_deadlock_map 갱신
        self.prev_deadlock_map = {
            iid: meta.get("is_deadlock", False) 
            for iid, meta in info_next.items() 
            if isinstance(meta, dict)
        }

        # 4. 에피소드 요약 정보 추가
        info_next["_summary"] = {
            "terminated": False,
            "truncated": (self.time >= self.max_steps),
            "time": self.time,
        }
        
        return reward_map, info_next
    
    def _update_deadlock_queue(self, info):
        """
        [수정] 데드락 큐를 (iid, amr_count, timestamp) 튜플로 관리합니다.
        1순위: 교차로 내 AMR 수 (내림차순)
        2순위: 데드락 발생 시점 (오름차순)
        """
        queue_changed = False
        
        # 현재 큐에 있는 iid와 정보를 맵으로 변환
        queue_map = {item[0]: item for item in self.deadlock_queue}
        
        # 새로운 큐를 생성
        next_deadlock_queue = []

        for iid, meta in info.items():
            if not isinstance(meta, dict):
                continue
            
            is_deadlocked = bool(meta.get("is_deadlock", False))
            
            if is_deadlocked:
                num_amrs = len(self.intersections[iid].amr_intent_map)
                if iid in queue_map:
                    # 기존 데드락: AMR 수만 업데이트하고, 발생 시간은 유지
                    _, _, timestamp = queue_map[iid]
                    next_deadlock_queue.append((iid, num_amrs, timestamp))
                else:
                    # 새로운 데드락: (iid, amr_count, timestamp) 추가
                    next_deadlock_queue.append((iid, num_amrs, self.time))
                queue_changed = True # 데드락이 하나라도 있으면 정렬 필요
            elif iid in queue_map:
                # 데드락 해소: 큐에서 제거되었으므로 변경 발생
                queue_changed = True

        # 큐에 변화가 있거나, 데드락이 하나라도 존재하면 정렬 수행
        if queue_changed or next_deadlock_queue:
            # 정렬 키: (-AMR 수, 발생 시간)
            # item[1]은 AMR 수, item[2]는 발생 시간
            next_deadlock_queue.sort(key=lambda item: (-item[1], item[2]))
            
            # 최종적으로 큐를 교체
            self.deadlock_queue = next_deadlock_queue

    def _inter_rank(self, iid):
        """
        [수정] 데드락 큐의 구조 변경에 맞춰 iid의 순위를 반환합니다.
        """
        try:
            # 큐는 (iid, amr_count, timestamp) 튜플의 리스트이므로, iid만 추출하여 인덱스를 찾음
            iids_only = [item[0] for item in self.deadlock_queue]
            return iids_only.index(iid)
        except ValueError:
            return math.inf
    
    def _update_intersections_state(self):
        """
        [전면 수정] 매 스텝 교차로를 초기화하고, 현재 AMR 위치를 기반으로 상태를 재구성합니다.
        """
        # --- 1. 모든 교차로의 내부 상태를 초기화합니다. ---
        for I in self.intersections.values():
            I.reset()

        # --- 2. 모든 AMR의 현재 위치를 기반으로 교차로에 다시 등록합니다. ---
        for amr in self.amr_list.values():
            # AMR 객체의 소속 교차로 정보도 매번 새로 계산
            amr.current_intersection_id.clear()
            for iid, I in self.intersections.items():
                if amr.pos in I.all_lane_coords:
                    I.register_amr(amr)
                    amr.current_intersection_id.add(iid)

        # --- 3. 모든 교차로에 대해 데드락을 검사합니다. ---
        for I in self.intersections.values():
            I.check_deadlock()
        
    def set_traffic_mode(self, mode: str):
        """
        [새로 추가된 함수]
        트래픽 생성 모드를 설정합니다 ('traffic' 또는 'task').
        """
        if mode == 'task':
            self.traffic_mode = 'task'
            self.traffic_generator = self.traffic_gen_task_set
            print("Traffic mode set to: 'task' (Fixed Task Set at reset)")
        elif mode == 'traffic':
            self.traffic_mode = 'traffic'
            self.traffic_generator = self.traffic_gen_stream
            self.traffic_generator.set_arm_gate(lambda iid, d: self.is_arm_outgoing_clear(iid, d))
            print("Traffic mode set to: 'traffic' (Streaming Poisson Traffic)")
        else:
            raise ValueError(f"Unknown traffic mode: '{mode}'. Choose 'traffic' or 'task'.")

    def _check_amr_completion(self):
        """
        AMR의 현재 위치가 자신의 goal 속성과 일치하는지 확인합니다.
        """
        completed_amrs = []
        for amr_id, amr_obj in self.amr_list.items():
            if amr_obj.pos == amr_obj.goal:
                completed_amrs.append(amr_id)

        for amr_id in completed_amrs:
            if amr_id in self.amr_list:
                self.completed_amr_steps.append(self.amr_list[amr_id].steps)
            self.traffic_generator.complete_task(amr_id)
            del self.amr_list[amr_id]

    def _spawn_new_amrs(self):
        """
        [통합 및 수정된 버전]
        현재 traffic_mode에 따라 새로운 AMR을 생성하고, Planner를 통해 경로를 계산하여 주입합니다.
        """
        gen = self.traffic_generator
        if not gen or not gen.should_spawn_next():
            return
        
        current_occupied_pos = {amr.pos for amr in self.amr_list.values()}

        # 1. 현재 모드에 맞는 task 생성
        if self.traffic_mode == 'task':
            new_tasks = gen.get_next_task_pair(current_time=self.time)
        else: # 'traffic' 모드
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
                print(f"Warning: Could not get start/goal position for AMR {amr_id}. Skipping.")
                continue

            if start_pos in current_occupied_pos:
                continue

            # 2. AMR 객체 생성 (goal 인자 추가)
            new_amr = AMR(start_pos, goal_pos, amr_id, self.color_map[amr_id % 100])
            
            # 3. 환경에 AMR 등록
            self.amr_list[amr_id] = new_amr

        # 4. Planner를 통해 경로 계산 및 경로 주입
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
                elif c in ['.', 'E', 'S']: row.append(0)
                else: raise ValueError(f"Invalid character in map file: {c}")
            if row: map_data.append(row)
        return np.array(map_data)

    def _find_intersection_center(self):
        kernel = np.array(
            [[1, 0, 1], 
             [0, 0, 0], 
             [1, 0, 1]])
        windows = np.lib.stride_tricks.sliding_window_view(self.map, kernel.shape)
        matches = np.all(windows == kernel, axis=(2, 3))
        centers = (np.argwhere(matches) + 1).tolist()
        return centers
        
    def _ray_len(self, r, c, dr, dc):
        H, W = self.map.shape
        length = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < H and 0 <= cc < W and self.map[rr][cc] == 0:
            if dr != 0:
                left_wall = (cc - 1 < 0) or (self.map[rr][cc - 1] == 1)
                right_wall = (cc + 1 >= W) or (self.map[rr][cc + 1] == 1)
                if not (left_wall or right_wall): break
            else:
                up_wall = (rr - 1 < 0) or (self.map[rr - 1][cc] == 1)
                down_wall = (rr + 1 >= H) or (self.map[rr + 1][cc] == 1)
                if not (up_wall or down_wall): break
            length += 1
            rr += dr
            cc += dc
        return length
    
    def _find_intersections_and_build_graph(self):
        """
        [수정] 교차로 데이터를 찾고, 각 교차로의 방향별 이웃 정보까지 계산하여
        처리된 딕셔너리를 한 번에 반환.
        """
        # 1. 모든 교차로 중심 좌표를 찾음
        centers_rc = self._find_intersection_center()
        centers_xy = [(c, r) for r, c in centers_rc]
        
        # 2. 각 중심에 대해 intersection_data 튜플을 생성하고, 좌표-데이터 맵을 만듦
        center_xy_to_data = {}
        for c, r in centers_xy:
            len_N = self._ray_len(r, c, -1, 0)
            len_E = self._ray_len(r, c, 0, 1)
            len_S = self._ray_len(r, c, 1, 0)
            len_W = self._ray_len(r, c, 0, -1)

            if min(len_N, len_E, len_S, len_W) > 0:
                current_data = (c, r, len_N, len_E, len_S, len_W)
                center_xy_to_data[(c, r)] = current_data

        # 3. 각 교차로에 대해 이웃 정보를 계산하고 최종 데이터 구조 생성
        processed_intersections = {}
        for center_coord, current_data in center_xy_to_data.items():
            c, r, len_N, len_E, len_S, len_W = current_data
            current_iid = f'x{c}y{r}'
            
            # 각 방향의 도로 끝에서 한 칸 더 나아간 '연결 예상 좌표' 계산
            target_coords = {
                'N': (c, r - len_N - 1), 'E': (c + len_E + 1, r),
                'S': (c, r + len_S + 1), 'W': (c - len_W - 1, r)
            }

            neighbors_map = {}
            # 4. 연결 예상 좌표가 다른 교차로의 중심과 일치하는지 확인
            for direction, target_coord in target_coords.items():
                if target_coord in center_xy_to_data:
                    neighbor_data = center_xy_to_data[target_coord]
                    neighbor_iid = f'x{neighbor_data[0]}y{neighbor_data[1]}'
                    neighbors_map[direction] = neighbor_iid
            
            processed_intersections[current_iid] = {
                'data': current_data,
                'neighbors': neighbors_map
            }
        
        return processed_intersections
    
    def is_arm_outgoing_clear(self, iid: str, d: str) -> bool:
        """
        [수정] 'outgoing' 속성 대신 'amr_intent_map'을 확인하여 진출 AMR 존재 여부를 판단합니다.
        """
        I = self.intersections[iid]

        # 1. 해당 팔에 나가는(outgoing) AMR이 있으면 생성 금지
        # AMR의 현재 팔과 출구 팔이 같으면 'outgoing' 상태임
        has_outgoing = any(
            data['current_arm'] == d and data['exit_arm'] == d 
            for data in I.amr_intent_map.values()
        )
        if has_outgoing:
            return False
            
        # 2. 해당 교차로가 데드락 상태이면 생성 금지
        # 데드락 큐의 아이템은 (iid, amr_count, timestamp) 튜플이므로 iid만 추출하여 확인
        deadlocked_iids = {item[0] for item in self.deadlock_queue}
        if iid in deadlocked_iids:
            return False
            
        # 두 조건 모두 통과하면 생성 허용
        return True

    # --- [GUI 연동을 위한 어댑터 함수들] ---
    def Get_AMR(self):
        """GUI가 AMR 목록을 가져갈 수 있도록 하는 함수"""
        return self.amr_list

    def get_active_tasks(self):
        """GUI가 AMR의 목표 지점을 가져갈 수 있도록 하는 함수"""
        active_goals = {}
        for amr_id, amr_obj in self.amr_list.items():
            active_goals[amr_id] = amr_obj.goal
        return active_goals

    def make_info(self):
        # [수정] 에피소드 종료 조건 확인
        terminated = False
        # 'task' 모드일 경우, 모든 AMR이 작업을 완료했는지 확인
        if self.traffic_mode == 'task':
            if self.traffic_generator.is_episode_done():
                terminated = True
        
        # 타임아웃 또는 정상 종료 시 False를 반환하여 루프 중단 신호
        if self.time >= self.max_steps or terminated:
            return False

        progress = self.traffic_generator.get_progress()
        total_pairs_done = progress['completed_total']

        # 스루풋 계산 (분 단위)
        throughput = (total_pairs_done / self.time * 60) if self.time > 0 else 0.0

        amr_states = {}
        for amr_id, amr_obj in self.amr_list.items():
            # AMR 상태를 더미 값으로 채움 (평가 스크립트에서는 사용하지 않음)
            amr_states[amr_id] = [f"Goal_{amr_id}", 0]

        # GUI가 사용하던 포맷 유지: [완료수, 스루풋, AMR상태]
        return [total_pairs_done, throughput, amr_states]


    def set_planner(self, algorithm_name: str):
        """
        [신규] 알고리즘 이름에 따라 self.planner 객체를 교체합니다.
        """
        if algorithm_name == "A*":
            self.planner = AStarPlanner(self.map)
        elif algorithm_name == "BFS":
            self.planner = BFSPlanner(self.map)
        elif algorithm_name == "CBS":
            self.planner = CBSPlanner(self.map)
        elif algorithm_name == "PIBT":
            self.planner = PIBTPlanner(self.map)

    def replan_all_paths(self):
        """
        [수정] 컨트롤러의 'replan_all' 메서드를 호출하도록 변경
        """
        if not self.amr_list:
            return
        
        print(f"Replanning all paths for {len(self.amr_list)} AMRs...")
        self.planner.replan_all(self.amr_list) # 수정된 함수 호출
        print("Replanning complete.")