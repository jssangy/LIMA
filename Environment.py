import os
import json
import random
import numpy as np
from typing import Dict
from collections import defaultdict

from utils.AGV import agv
from utils.Intersection import Intersection
from utils.DeadlockDetector import DeadlockDetector
from utils import Funct
from utils.traffic_generator import TrafficGenerator
from utils.Controller import controller


class ENV():
    def __init__(self, prob_path):
        super().__init__()
        """환경 초기화"""
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])
        
        self.map = self._load_map(map_path)
        self.intersection_data, self.full_adjacency = self._find_intersections_and_build_graph()
        if not self.intersection_data:
            raise ValueError("No intersection found in the map")
        
        self.time = 0
        self.agv_list = {}
        self.l_hop = 1
        self.max_steps = 1000

        # TrafficGenerator
        self.traffic_generator = TrafficGenerator()
        self.max_inside = 6
        self.traffic_generator.set_capacity_gate(self._spawn_gate)

        # Controller, DeadlockDetector, Intersections
        self.color_map = Funct.Color_dict(self.traffic_generator.total_tasks_in_episode).dic
        self.controller = controller(self.map)
        self.deadlock_detector = DeadlockDetector(self.controller)
        self.intersections: Dict[tuple, Intersection] = {
            inter_data: Intersection(inter_data, self.controller)
            for inter_data in self.intersection_data
        }
        self.prev_deadlock = False

        self.use_rl = False
        self.rl_policy = None
        
        self.is_push_out = False

    def reset(self):        
        self.time = 0
        self.tau = 0
        self.agv_list.clear()
        self.traffic_generator.start_new_episode()

        # 컨트롤러와 모든 교차로의 내부 상태 초기화
        self.controller.reset()
        self.intersection.reset()

        self._spawn_amrs_if_needed()

        # 리셋 시에는 초기 관찰 상태만 반환
        obs, info = self.generate_observation()
        self.prev_deadlock = False

        self.is_push_out = False

        return obs, info

    def step(self, actions=None, train=True):
        """한 스텝 '실행'에만 집중 (generate에서 최신화했으면 step에선 다시 최신화하지 않음)"""
        self.time += 1

        # 0) 현재 스냅샷(상태/마스크/푸시아웃 플래그) 1회만 생성
        obs_now, info_now = self.generate_observation()

        # 1) 액션 결정 (테스트 모드에서 RL on이면 상승엣지에만)
        act_to_apply = actions
        if act_to_apply is None and (not train) and self.use_rl and (self.rl_policy is not None):
            if obs_now is not None and info_now.get("deadlock_active", False) and self.intersection.center_agv is not None:
                is_start = (not self.prev_deadlock)
                if is_start:
                    try:
                        act_to_apply = self.rl_policy(obs_now, info_now.get("action_mask", None))
                    except Exception:
                        self.use_rl = False

        # 2) (중요) 이번 스텝 이동 계획을 한 곳에서 모아 커밋
        if hasattr(self.intersection, "begin_plan"):
            self.intersection.begin_plan()             # 플래너 시작

            # 2-1) 팔 스와핑 끌어오기 계획 (부작용은 여기서만)
            self.intersection.resolve_arm_swaps_all()

            # 2-2) 중앙 액션/푸시아웃 계획 (마찬가지로 플래너에만 추가)
            if act_to_apply is not None:
                self.intersection.action_control(int(act_to_apply), self.is_push_out)

            self.intersection.finalize_plan()          # control_buffer / push_sequence 커밋
        else:
            # 플래너가 없다면, 기존 방식 fallback (권장: 플래너 도입)
            self.intersection.resolve_arm_swaps_all()
            if act_to_apply is not None:
                self.intersection.action_control(int(act_to_apply), self.is_push_out)

        # 3) Movement: push_sequence 우선 → 일반 이동
        priority = getattr(self.controller, "push_sequence", [])
        moved = set()

        for agv_id in priority:
            agv_obj = self.agv_list.get(agv_id)
            if agv_obj is None:
                continue
            sig = self.controller.control_buffer.get(agv_id, (0, 0))
            if self._is_valid_move(agv_obj, sig):
                agv_obj.move(sig)
                moved.add(agv_id)

        for agv_id, agv_obj in list(self.agv_list.items()):
            if agv_id in moved:
                continue
            sig = self.controller.control_buffer.get(agv_id, (0, 0))
            if self._is_valid_move(agv_obj, sig):
                agv_obj.move(sig)

        # 사용 후 정리
        self.controller.push_sequence = []

        # 4) 환경 변화 처리
        self._check_amr_completion()
        self._spawn_amrs_if_needed()

        # 5) 다음 의사결정을 위한 상태 계산 (여기서만 다시 generate 호출)
        obs_next, info_next = self.generate_observation()

        if not train:
            return self.make_info()   # GUI/테스트용 요약 반환 유지

        # 6) 보상/이벤트
        curr_deadlock = bool(info_next["deadlock_active"])
        event_start = (not self.prev_deadlock) and curr_deadlock
        event_end   = self.prev_deadlock and (not curr_deadlock)

        reward = 0.0
        if self.prev_deadlock:
            reward -= 0.05
            self.tau += 1
        if event_end:
            reward += 1.0
            info_next["tau"] = self.tau
            self.tau = 0

        self.prev_deadlock = curr_deadlock

        terminated = False
        truncated  = (self.time >= self.max_steps)

        if (terminated or truncated) and self.prev_deadlock:
            info_next["event_end"] = True
            info_next["tau"] = self.tau
            self.tau = 0
        else:
            info_next["event_end"] = event_end

        info_next.update({
            "event_start": event_start,
            "terminated": terminated,
            "truncated": truncated,
        })

        return obs_next, reward, info_next


    def generate_observation(self):
        """
        관측 생성(부작용 없음):
        - sensing/make_control
        - 교차로 최신화(_update_intersections_state)
        - 상태/마스크/푸시아웃 플래그 계산만 수행
        """
        # 1) 센싱/컨트롤 업데이트
        for agv_id, agv_obj in self.agv_list.items():
            self.controller.get_sensing(agv_id, agv_obj.pos)
        self.controller.make_control()

        # 2) 교차로 최신화 (여기서만 최신화! step에서는 하지 않음)
        self._update_intersections_state()

        # 3) (중요) 계획 주입 함수는 여기서 호출하지 않음
        #    self.intersection.resolve_arm_swaps_all()  # ← 제거

        # 4) 데드락/상태/마스크
        is_deadlock = self.deadlock_detector.check_deadlock(self.intersection)

        state = np.asarray(self.intersection.get_state())
        edge_index = np.array([[0], [0]], dtype=np.int64)
        action_mask, is_push_out = self.intersection.calculate_action_mask()
        action_mask = np.asarray(action_mask, dtype=np.bool_)

        obs = {
            "state": state,
            "edge_index": edge_index,
        }
        info = {
            "deadlock_active": is_deadlock,
            "action_mask": action_mask,
            "is_push_out": is_push_out,
        }

        # step에서 사용하게 내부 플래그 저장
        self.is_push_out = is_push_out
        return obs, info


    def _is_valid_move(self, current_agv, control_signal):
        next_pos = (current_agv.pos[0] + control_signal[0], current_agv.pos[1] + control_signal[1])
        if self.map[next_pos[1]][next_pos[0]] == 1: return False
        if not (0 <= next_pos[0] < self.map.shape[1] and 0 <= next_pos[1] < self.map.shape[0]): return False
        for agv_id, other_agv in self.agv_list.items():
            if current_agv != other_agv and next_pos == other_agv.pos: return False
        return True
    
    def _update_intersections_state(self):
        self.intersection.reset()
        for agv_id, agv_obj in self.agv_list.items():
            pos = agv_obj.pos
            if pos in self.intersection.all_lane_coords:
                self.intersection.add_agv(agv_obj)

    def _spawn_amrs_if_needed(self):
        # [수정] 새로운 TrafficGenerator 로직에 맞게 변경
        if self.traffic_generator.should_spawn_next():
            task_pair = self.traffic_generator.get_next_task_pair()
            if task_pair is None: return
            
            # 두 개의 AMR을 순차적으로 생성
            for task_info in task_pair:
                start_intersection_data = random.choice(self.intersection_data)
                
                # 교차로가 2개 이상일 때만 다른 목적지를 선택
                if len(self.intersection_data) > 1:
                    possible_goals = [i for i in self.intersection_data if i != start_intersection_data]
                    goal_intersection_data = random.choice(possible_goals)
                else:
                    goal_intersection_data = start_intersection_data

                agv_id = task_info['id']
                start_pos = self._direction_to_coords(task_info['start_direction'], start_intersection_data)
                goal_pos = self._direction_to_coords(task_info['goal_direction'], goal_intersection_data)
                
                color = self.color_map.get(agv_id, (255, 0, 0))
                self.agv_list[agv_id] = agv(start_pos, agv_id, color)
                
                self.controller.add_agv(agv_id, start_pos, goal_pos)

    def _check_amr_completion(self):
        completed_agvs = []
        for agv_id, agv_obj in self.agv_list.items():
            if agv_obj.pos == self.controller.agv_goal.get(agv_id):
                completed_agvs.append(agv_id)
        
        for agv_id in completed_agvs:
            self.traffic_generator.complete_task(agv_id)
            del self.agv_list[agv_id]
            self.controller.remove_agv(agv_id)

    def _direction_to_coords(self, direction, intersection_data):
        center_x, center_y, len_N, len_E, len_S, len_W = intersection_data
        direction_map = {
            'N': (center_x, center_y - len_N - 1), 'E': (center_x + len_E + 1, center_y),
            'S': (center_x, center_y + len_S + 1), 'W': (center_x - len_W - 1, center_y)
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
        교차로 데이터 튜플을 ID로 사용하여 그래프를 생성하는 통합 함수.
        """
        intersections_data = []
        adj = defaultdict(list)
        
        # 1. 모든 교차로 중심 좌표를 찾음
        centers_rc = self._find_intersection_center()
        centers_xy = [(c, r) for r, c in centers_rc]
        
        # 2. 각 중심에 대해 intersection_data 튜플을 생성하고, 좌표-튜플ID 맵을 만듦
        center_xy_to_id = {}
        for c, r in centers_xy:
            len_N = self._ray_len(r, c, -1, 0)
            len_E = self._ray_len(r, c, 0, 1)
            len_S = self._ray_len(r, c, 1, 0)
            len_W = self._ray_len(r, c, 0, -1)

            if min(len_N, len_E, len_S, len_W) > 0:
                # intersection_data 튜플을 생성하여 ID로 사용
                current_id = (c, r, len_N, len_E, len_S, len_W)
                intersections_data.append(current_id)
                center_xy_to_id[(c, r)] = current_id

        # 3. 각 교차로에 대해 도로를 뻗어 연결성 검사
        for current_id in intersections_data:
            c, r, len_N, len_E, len_S, len_W = current_id
            
            # 각 방향의 도로 끝에서 한 칸 더 나아간 '연결 예상 좌표' 계산
            target_coords = {
                'N': (c, r - len_N - 1), 'E': (c + len_E + 1, r),
                'S': (c, r + len_S + 1), 'W': (c - len_W - 1, r)
            }

            # 4. 연결 예상 좌표가 다른 교차로의 중심과 일치하는지 확인
            for direction, target_coord in target_coords.items():
                if target_coord in center_xy_to_id:
                    neighbor_id = center_xy_to_id[target_coord]
                    # 양방향으로 연결 정보 추가
                    adj[current_id].append(neighbor_id)
                    adj[neighbor_id].append(current_id)
        
        # defaultdict를 일반 dict로 변환하고, 중복 제거 및 정렬
        final_adj = {k: sorted(list(set(v))) for k, v in adj.items()}

        return intersections_data, final_adj

    def _center_occupied_any(self) -> bool:
        """(단일 교차로 가정) 교차로 중앙 점유 여부"""
        return self.intersection.center_agv is not None

    def _count_inside_intersection(self) -> int:
        """교차로 내부(팔+중앙) AMR 수 (인덱스 사용)"""
        # agvs_in_intersection: set of AGV objects
        return len(self.intersection.agvs_in_intersection)

    def _arm_has_outgoing(self, direction: str) -> bool:
        """해당 팔에서 바깥으로 나가려는(outgoing) AMR이 하나라도 있으면 True"""
        return bool(getattr(self.intersection, 'outgoing', {}).get(direction, False))

    def _spawn_gate(self, direction: str) -> bool:
        """
        Poisson 스폰을 막는 글로벌 게이트:
        - 중앙 점유 시 전체 스폰 정지
        - 교차로 내부 AMR 수가 임계치 이상이면 정지
        - 해당 팔 점유 시 해당 방향 스폰 금지
        """
        if self._center_occupied_any():               # ★ 중앙 점유 금지
            return False
        if self._count_inside_intersection() >= self.max_inside:
            return False
        if self._arm_has_outgoing(direction):
            return False
        return True

    # --- [GUI 연동을 위한 어댑터 함수들] ---
    def Get_AGV(self):
        """GUI가 AGV 목록을 가져갈 수 있도록 하는 함수"""
        return self.agv_list

    def get_active_tasks(self):
        """GUI가 AGV의 목표 지점을 가져갈 수 있도록 하는 함수"""
        return self.controller.agv_goal

    def make_info(self):
        # 스트리밍 모드: 타임리밋으로만 종료 신호
        if self.time >= self.max_steps:
            return False  # GUI에게 에피소드 종료 신호

        progress = self.traffic_generator.get_progress()
        # 바뀐 키들:
        # 'completed_pairs_in_cycle', 'total_pairs_per_cycle', 'completed_total', 'active_agvs', 'cycle'
        total_pairs_done = progress['completed_total']

        # 단순 스루풋(쌍/스텝). 필요하면 시간 스케일 맞춰 곱해 쓰세요.
        throughput = (total_pairs_done / self.time) if self.time > 0 else 0.0

        agv_states = {}
        for agv_id, agv_obj in self.agv_list.items():
            mode = 2 if self.prev_deadlock and agv_obj in self.intersection.agvs_in_intersection else 0
            agv_states[agv_id] = [f"Goal_{agv_id}", mode]

        # GUI가 쓰던 포맷 유지: [완료수, 스루풋, AGV상태]
        return [total_pairs_done, throughput, agv_states]
