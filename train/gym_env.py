import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import json

from utils.AGV import agv
from utils.Intersection import Intersection
from train_controller import controller
from traffic_generator import TrafficGenerator


class GymEnv(gym.Env):    
    metadata = {"render_modes": []}

    def __init__(self, prob_path):
        super().__init__()
        
        # Environment 초기화
        self._init_environment(prob_path)
        
        # Gym spaces 정의
        self._init_gym_spaces()

    def _init_environment(self, prob_path):
        """환경 초기화"""
        # 기존 초기화 코드...
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])
        
        self.map = self.load_map(map_path)
        self.intersection_centers = self.find_intersections()
        if not self.intersection_centers:
            raise ValueError("No intersection found in the map")
        
        self.traffic_generator = TrafficGenerator()
        
        # Environment state
        self.time = 0
        self.agv_list = {}
        self.current_amr = None
        self.current_goal_direction = None
        
        # 위치 추적을 위한 변수들
        self.prev_agv_positions = {}  # 이전 스텝의 AGV 위치들
        
        # Controller and intersection
        self.controller = None
        self.intersection = None

    def _init_gym_spaces(self):
        """기존과 동일"""
        # Observation space (교차로 상태)
        low = []
        high = []
        for _ in range(4):  # 4방향
            low.extend([0] * 6)
            high.extend([1, 1, 1, 1, 1000, 1])
        low.extend([0] * 4)
        high.extend([1] * 4)

        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(28,),
            dtype=np.float32,
        )

        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4, 4])

    def reset(self, seed=None, options=None):
        """환경 리셋"""
        super().reset(seed=seed)
        
        # 환경 상태 초기화
        self.time = 0
        self.agv_list = {}
        self.current_amr = None
        self.current_goal_direction = None
        self.prev_agv_positions = {}
        
        # 새 에피소드 시작
        self.traffic_generator.start_new_episode()
        
        # Controller와 Intersection 초기화
        self._init_controller_and_intersection()
        
        # 첫 번째 AMR 생성
        self._spawn_new_amr()

        obs = self._get_observation().astype(np.float32, copy=False)
        info = {
            "agv_in_intersection": np.array(self._agv_in_intersection(), dtype=np.int8),
            "episode_progress": self.traffic_generator.get_progress()
        }
        return obs, info

    def step(self, action):
        """한 스텝 실행"""
        self.time += 1
        
        # AMR 완료 체크
        episode_done = False
        step_reward = 0
        
        if self._check_amr_completion():
            step_reward += 1.0
            
            if self.traffic_generator.has_remaining_tasks():
                self._spawn_new_amr()
            else:
                episode_done = True
                episode_metrics = self.traffic_generator.get_episode_metrics()
                print(f"Episode completed! Metrics: {episode_metrics}")
        
        elif self.current_amr is None:
            self._spawn_new_amr()
        
        # 교차로 제어 적용
        if self.current_amr and action is not None:
            self._apply_intersection_control(action)
        
        # AMR 이동 시뮬레이션
        if self.current_amr:
            old_pos = self.current_amr.pos
            self._simulate_amr_movement()
            new_pos = self.current_amr.pos
            
            # 위치 변화가 있으면 intersection에 알림
            if old_pos != new_pos:
                self._notify_position_change('A', old_pos, new_pos)
        
        # 관찰, 보상, 종료 조건
        observation = self._get_observation().astype(np.float32, copy=False)
        reward = float(step_reward + self._calculate_step_reward())
        terminated = episode_done
        truncated = False
        
        info = {
            "agv_in_intersection": np.array(self._agv_in_intersection(), dtype=np.int8),
            "episode_progress": self.traffic_generator.get_progress(),
            "time": self.time
        }

        return observation, reward, terminated, truncated, info

    def _notify_position_change(self, agv_num, old_pos, new_pos):
        """AGV 위치 변화를 intersection에 알림"""
        if hasattr(self, 'intersection') and self.intersection:
            self.intersection.on_agv_position_changed(agv_num, old_pos, new_pos)
        
        # Controller의 agv_pos도 업데이트
        if hasattr(self, 'controller') and self.controller:
            self.controller.agv_pos[agv_num] = new_pos
    
    def _spawn_new_amr(self):
        """새 AMR 생성"""
        task_info = self.traffic_generator.get_next_task()
        if task_info is None:
            return
        
        start_pos = self._direction_to_coords(task_info['start_direction'])
        self.current_amr = agv(start_pos, (255, 0, 0))
        self.current_goal_direction = task_info['goal_direction']
        self.agv_list['A'] = self.current_amr
        
        # 경로 생성 및 설정
        self._create_amr_path(task_info)
        
        # Intersection에 새 AMR 알림
        self._notify_position_change('A', None, start_pos)
        
        print(f"Time {self.time}: New AMR spawned: {task_info['start_direction']} → {task_info['goal_direction']}")

    def _create_amr_path(self, task_info):
        """AMR 경로 생성"""
        start_pos = self._direction_to_coords(task_info['start_direction'])
        center_pos = (self.intersection_centers[0][0], self.intersection_centers[0][1])
        goal_pos = self._direction_to_coords(task_info['goal_direction'])
        
        # 간단한 경로: 시작점 → 교차로 중심 → 목표점
        path = [start_pos, center_pos, goal_pos]
        
        # Controller에 경로 설정
        if hasattr(self, 'controller') and self.controller:
            self.controller.agv_path['A'] = path
            self.controller.agv_pos['A'] = start_pos

    def _check_amr_completion(self):
        """AMR 완료 여부 체크"""
        if self.current_amr is None:
            return False
        
        if self._is_goal_reached():
            # 완료 처리
            self.traffic_generator.complete_current_task(self.time, success=True)
            
            # Intersection에 AMR 제거 알림
            self._notify_position_change('A', self.current_amr.pos, None)
            
            # AMR 제거
            self.current_amr = None
            self.current_goal_direction = None
            if 'A' in self.agv_list:
                del self.agv_list['A']
            
            # Controller에서도 제거
            if hasattr(self, 'controller') and self.controller:
                self.controller.agv_pos.pop('A', None)
                self.controller.agv_path.pop('A', None)
            
            print(f"Time {self.time}: AMR reached goal!")
            return True
        
        return False

    def _simulate_amr_movement(self):
        """AMR 이동 시뮬레이션"""
        if not self.current_amr or not hasattr(self, 'controller'):
            return
        
        # Controller의 제어 신호가 있으면 적용
        control_signal = self.controller.control_buffer.get('A', None)
        if control_signal:
            dx, dy = control_signal
            current_x, current_y = self.current_amr.pos
            new_pos = (current_x + dx, current_y + dy)
            self.current_amr.pos = new_pos
            # 제어 신호 소비
            self.controller.control_buffer.pop('A', None)
        else:
            # 기본 이동 로직 (교차로 중심으로)
            center_x, center_y = self.intersection_centers[0][:2]
            current_x, current_y = self.current_amr.pos
            
            if abs(current_x - center_x) > 1:
                dx = 1 if current_x < center_x else -1
                self.current_amr.pos = (current_x + dx, current_y)
            elif abs(current_y - center_y) > 1:
                dy = 1 if current_y < center_y else -1
                self.current_amr.pos = (current_x, current_y + dy)
            else:
                # 중심에 도달했으면 목표 방향으로 이동
                goal_pos = self._direction_to_coords(self.current_goal_direction)
                dx = np.sign(goal_pos[0] - current_x)
                dy = np.sign(goal_pos[1] - current_y)
                if dx != 0 or dy != 0:
                    self.current_amr.pos = (current_x + dx, current_y + dy)

    def _init_controller_and_intersection(self):
        """Controller와 Intersection 초기화"""
        # Controller 초기화
        self.controller = controller(1, self.map, self.agv_list, [])
        
        # Intersection 초기화 (이벤트 기반)
        if self.intersection_centers:
            self.intersection = Intersection(self.intersection_centers[0], self.controller)

    def _apply_intersection_control(self, action):
        """교차로 제어 적용"""
        if hasattr(self, 'intersection') and self.intersection:
            self.intersection.action_control(action)

    def _get_observation(self):
        """관찰 상태 반환"""
        if hasattr(self, 'intersection') and self.intersection:
            state = self.intersection.get_state()
            return np.array(state, dtype=np.float32)
        else:
            return np.zeros(28, dtype=np.float32)

    def _calculate_step_reward(self):
        """스텝별 보상 계산"""
        reward = -0.01  # 시간 페널티
        
        if hasattr(self, 'intersection') and self.intersection:
            for event in self.intersection.exit_events:
                if event.get("correct", False):
                    reward += 0.1
                else:
                    reward -= 0.1
        
        return reward

    def _agv_in_intersection(self):
        """교차로 내 AMR 존재 여부"""
        if hasattr(self, 'intersection') and self.intersection:
            return 0 if self.intersection.is_empty else 1
        return 0

    # 기존 유틸리티 메서드들은 그대로 유지...
    def _is_goal_reached(self):
        """목표 지점 도달 여부 체크"""
        if not self.current_amr or not self.current_goal_direction:
            return False
        
        center_x, center_y, len_N, len_E, len_S, len_W = self.intersection_centers[0]
        current_x, current_y = self.current_amr.pos
        
        if self.current_goal_direction == 'N':
            return current_y <= center_y - len_N - 1
        elif self.current_goal_direction == 'E':
            return current_x >= center_x + len_E + 1
        elif self.current_goal_direction == 'S':
            return current_y >= center_y + len_S + 1
        elif self.current_goal_direction == 'W':
            return current_x <= center_x - len_W - 1
        
        return False

    def _direction_to_coords(self, direction):
        """방향을 실제 좌표로 변환"""
        center_x, center_y, len_N, len_E, len_S, len_W = self.intersection_centers[0]
        
        direction_map = {
            'N': (center_x, center_y - len_N),
            'E': (center_x + len_E, center_y),
            'S': (center_x, center_y + len_S),
            'W': (center_x - len_W, center_y)
        }
        
        return direction_map[direction]

    # 나머지 맵 로딩 메서드들은 기존과 동일...
    def load_map(self, map_path):        
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f"Map file not found: {map_path}")
        
        map_data = []
        with open(map_path, 'r') as f:
            lines = f.readlines()
        
        map_start = None
        for idx, line in enumerate(lines):
            if line.strip() == 'map':
                map_start = idx + 1
                break
        
        if map_start is None:
            raise ValueError("Map data not found in file")
            
        for line in lines[map_start:]:
            row = []
            for c in line.strip():
                if c in ['@', 'T']:
                    row.append(1)
                elif c in ['.', 'E', 'S']:
                    row.append(0)
                else:
                    raise ValueError(f"Invalid character in map file: {c}")
            if row:
                map_data.append(row)
            
        return np.array(map_data)

    def find_intersection_center(self):
        """교차로 중심점 찾기"""
        from numpy.lib.stride_tricks import sliding_window_view
        
        kernel = np.array([[1, 0, 1],
                           [0, 0, 0],
                           [1, 0, 1]])

        windows = sliding_window_view(self.map, kernel.shape)
        matches = np.all(windows == kernel, axis=(2, 3))
        centers = (np.argwhere(matches) + 1).tolist()

        return centers
        
    def ray_len(self, r, c, dr, dc):
        """특정 방향으로 뻗어나가는 길이 계산"""
        H, W = self.map.shape
        length = 0
        rr, cc = r + dr, c + dc

        while 0 <= rr < H and 0 <= cc < W and self.map[rr][cc] == 0:
            if dr != 0:
                left_wall = (cc - 1 < 0) or (self.map[rr][cc - 1] == 1)
                right_wall = (cc + 1 >= W) or (self.map[rr][cc + 1] == 1)
                if not (left_wall or right_wall):
                    break
            else:
                up_wall = (rr - 1 < 0) or (self.map[rr - 1][cc] == 1)
                down_wall = (rr + 1 >= H) or (self.map[rr + 1][cc] == 1)
                if not (up_wall or down_wall):
                    break
            length += 1
            rr += dr
            cc += dc
        return length
    
    def find_intersections(self):
        """교차로 정보 찾기"""
        intersections = []
        for r, c in self.find_intersection_center():
            len_N = self.ray_len(r, c, -1, 0)
            len_E = self.ray_len(r, c, 0, 1)
            len_S = self.ray_len(r, c, 1, 0)
            len_W = self.ray_len(r, c, 0, -1)

            if min(len_N, len_E, len_S, len_W) > 0:
                intersections.append((c, r, len_N, len_E, len_S, len_W))

        return intersections

    def close(self):
        """환경 종료"""
        pass