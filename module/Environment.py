import os
import json
import random
import numpy as np
from collections import defaultdict

from utils.AGV import agv
from utils.Intersection import Intersection
from utils.DeadlockDetector import DeadlockDetector
from utils import Funct
from utils.traffic_generator import TrafficGenerator
from utils.train_controller import controller


class ENV():
    def __init__(self, prob_path):
        super().__init__()
        self._init_environment(prob_path)

    def _init_environment(self, prob_path):
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

        # TrafficGenerator, Controller, DeadlockDetector, Intersections
        self.traffic_generator = TrafficGenerator()
        self.color_map = Funct.Color_dict(self.traffic_generator.total_tasks_in_episode).dic
        self.controller = controller(self.map)
        self.deadlock_detector = DeadlockDetector(self.controller)
        self.intersection = Intersection(self.intersection_data[0], self.controller)
        self.prev_deadlock = False

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
        self.prev_deadlock = bool(info['deadlock_active'])
        self.in_event = self.prev_deadlock

        return obs, info

    def step(self, actions=None, train=True):
        """한 스텝 '실행'에만 집중"""
        self.time += 1

        # 1. Action 적용: 전달받은 actions을 기반으로 제어 신호 수정
        if actions is not None:
            self.intersection.action_control(actions)

        # 2. Movement: 수정된 제어 신호에 따라 모든 AGV 이동
        for agv_id, agv_obj in list(self.agv_list.items()):
            control_sig = self.controller.control_buffer[agv_id]
            if self._is_valid_move(agv_obj, control_sig):
                agv_obj.move(control_sig)
        
        # 3. 환경 변화 처리: AGV 완료 및 신규 생성
        self._check_amr_completion()
        self._spawn_amrs_if_needed()

        # 4. 다음 의사결정을 위한 상태 계산
        obs_next, info_next = self.generate_observation()
        curr_deadlock = bool(info_next["deadlock_active"])

        if not train:
            return self.make_info()

        event_start = (not self.in_event) and curr_deadlock
        event_end = self.in_event and (not curr_deadlock)

        # 5. 보상 계산
        reward = 0.0
        if self.in_event:
            reward -= 0.01
            self.tau += 1
        if event_end:
            reward += 1.0
            info_next["tau"] = self.tau
            self.tau = 0
        else:
            self.in_event = curr_deadlock

        info_next.update({
            "event_start": event_start,
            "in_event": self.in_event,
        })
        
        self.prev_deadlock = curr_deadlock

        # 6. 종료 조건 확인
        done = self.traffic_generator.is_episode_done() and not self.agv_list

        return obs_next, reward, done, info_next

    def generate_observation(self):
        """
        GNN을 위한 Dict 형태의 관측 생성
        """
        # 물리적 상태 업데이트 및 데드락 탐지 로직 (기존과 동일)
        for agv_id, agv_obj in self.agv_list.items():
            self.controller.get_sensing(agv_id, agv_obj.pos)
        self.controller.make_control()
        self._update_intersections_state()
        is_deadlock = self.deadlock_detector.check_deadlock(self.intersection)
        
        # 이벤트가 없을 경우, 0으로 패딩된 고정 크기 Dict 반환
        if not is_deadlock:
            return None, {'deadlock_active': False}

        state = np.asarray(self.intersection.get_state())
        edge_index = np.empty((2, 0), dtype=np.int64)  # [2, E] (단일 노드→엣지 없음)
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

        return obs, info

    def _is_valid_move(self, current_agv, control_signal):
        next_pos = (current_agv.pos[0] + control_signal[0], current_agv.pos[1] + control_signal[1])
        if self.map[next_pos[1]][next_pos[0]] == 1: return False
        for agv_id, other_agv in self.agv_list.items():
            if current_agv != other_agv and next_pos == other_agv.pos: return False
        return True
    
    def _update_intersections_state(self):
        self.intersection.reset()
        for agv_id, agv_obj in self.agv_list.items():
            pos = agv_obj.pos
            if pos in self.intersection.all_lane_coords:
                self.intersection.add_agv(agv_id, pos)

    def _spawn_amrs_if_needed(self):
        # [수정] 새로운 TrafficGenerator 로직에 맞게 변경
        if self.traffic_generator.should_spawn_next():
            task_pair = self.traffic_generator.get_next_task_pair()
            if task_pair is None: return

            print(f"\nTime {self.time}: Spawning new AMR pair...")
            
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
                self.agv_list[agv_id] = agv(start_pos, color)
                
                self.controller.add_agv(agv_id, start_pos, goal_pos)
                print(f"  - AMR {agv_id} spawned at {start_pos}, heading to goal near {goal_pos} "
                      f"({task_info['start_direction']}->{task_info['goal_direction']})")

    def _check_amr_completion(self):
        completed_agvs = []
        for agv_id, agv_obj in self.agv_list.items():
            if agv_obj.pos == self.controller.agv_goal.get(agv_id):
                completed_agvs.append(agv_id)
        
        for agv_id in completed_agvs:
            print(f"Time {self.time}: AMR {agv_id} reached goal!")
            # [수정] TrafficGenerator에 작업 완료 알림
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

        print("✓ Intersections and Adjacency List Created in one pass:")
        for inter_id in intersections_data:
            print(f"  - {inter_id}: {final_adj.get(inter_id, [])}")

        return intersections_data, final_adj
    
    def close(self):
        pass

    # --- [GUI 연동을 위한 어댑터 함수들] ---
    def Get_AGV(self):
        """GUI가 AGV 목록을 가져갈 수 있도록 하는 함수"""
        return self.agv_list

    def get_active_tasks(self):
        """GUI가 AGV의 목표 지점을 가져갈 수 있도록 하는 함수"""
        return self.controller.agv_goal

    def make_info(self):
        """GUI의 State Panel 포맷에 맞게 정보를 가공하는 함수"""
        if self.traffic_generator.is_episode_done() and not self.agv_list:
            return False # 에피소드 종료 신호

        progress = self.traffic_generator.get_progress()
        total_prod = progress['completed_pairs']

        throughput = (total_prod / self.time * 3600) if self.time > 0 else 0

        agv_states = {}
        for agv_id, agv_obj in self.agv_list.items():
            # 단일 intersection만 고려
            mode = 2 if self.prev_deadlock and agv_id in self.intersection.agvs_in_intersection else 0
            agv_states[agv_id] = [f"Goal_{agv_id}", mode]

        return [total_prod, throughput, agv_states]