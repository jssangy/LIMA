import os
import json
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.AGV import agv
from utils.Intersection import Intersection
from utils.DeadlockDetector import DeadlockDetector
from train_controller import controller
from traffic_generator import TrafficGenerator


class GymEnv(gym.Env):    
    metadata = {"render_modes": []}

    def __init__(self, prob_path):
        super().__init__()
        self._init_environment(prob_path)
        self._init_gym_spaces()

    def _init_environment(self, prob_path):
        """환경 초기화"""
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])
        
        self.map = self._load_map(map_path)
        self.intersection_centers = self._find_intersections()
        if not self.intersection_centers:
            raise ValueError("No intersection found in the map")
        
        self.time = 0
        self.agv_list = {}
        self.traffic_generator = TrafficGenerator()

        # Controller, DeadlockDetector, Intersections
        self.controller = controller(self.map)
        self.deadlock_detector = DeadlockDetector(self.controller)
        self.intersections = [Intersection(data, self.controller) for data in self.intersection_centers]

        # [최적화] 좌표-교차로 매핑을 미리 생성
        self.coord_to_intersection_map = self._build_coord_to_intersection_map()

        self.previous_deadlock_states = {}

    def _init_gym_spaces(self):
        """다중 교차로(에이전트)를 위한 Gym spaces 정의"""
        obs_spaces = {}
        action_spaces = {}
        single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        single_action_space = spaces.MultiDiscrete([4, 4, 4, 4, 4])

        for intersection in self.intersections:
            agent_id = intersection.id
            obs_spaces[agent_id] = single_obs_space
            action_spaces[agent_id] = single_action_space
        
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Dict(action_spaces)

    def reset(self, seed=None, options=None):
        """환경 리셋"""
        super().reset(seed=seed)
        
        self.time = 0
        self.agv_list.clear()
        self.traffic_generator.start_new_episode()

        # 컨트롤러와 모든 교차로의 내부 상태 초기화
        self.controller.reset()
        for intersection in self.intersections:
            intersection.reset()

        self._spawn_amrs_if_needed()

        # 리셋 시에는 초기 관찰 상태만 반환
        observations, active_agents = self._update_observation()
        self.previous_deadlock_states = {inter.id: (inter.id in active_agents) for inter in self.intersections}
        info = self._get_info(active_agents)
        return observations, info

    def step(self, actions):
        """[재설계] 한 스텝 '실행'에만 집중"""
        self.time += 1

        # 1. Action 적용: 전달받은 actions을 기반으로 제어 신호 수정
        if actions:
            for agent_id, action in actions.items():
                target_intersection = next((inter for inter in self.intersections if inter.id == agent_id), None)
                if target_intersection:
                    target_intersection.action_control(action)

        # 2. Movement: 수정된 제어 신호에 따라 모든 AGV 이동
        for agv_id, agv_obj in list(self.agv_list.items()):
            control_sig = self.controller.control_buffer[agv_id]
            if self._is_valid_move(agv_obj, control_sig):
                agv_obj.move(control_sig)
        
        # 3. 환경 변화 처리: AGV 완료 및 신규 생성
        self._check_amr_completion()
        self._spawn_amrs_if_needed()

        # 4. 다음 의사결정을 위한 상태 계산
        observation, active_agents = self._update_observation()
        
        # 5. 보상, 종료 조건 계산
        rewards = self._calculate_step_reward(actions.keys(), active_agents)
        self.previous_deadlock_states = {inter.id: (inter.id in active_agents) for inter in self.intersections}
        episode_terminated = self.traffic_generator.is_episode_done() and not self.agv_list
        terminateds = {agent_id: episode_terminated for agent_id in active_agents}
        terminateds["__all__"] = episode_terminated
        truncated = False
        info = self._get_info(active_agents)

        return observation, rewards, terminateds, truncated, info
    
    def _is_valid_move(self, current_agv, control_signal):
        next_pos = (current_agv.pos[0] + control_signal[0], current_agv.pos[1] + control_signal[1])
        if self.map[next_pos[1]][next_pos[0]] == 1: return False
        for agv_id, other_agv in self.agv_list.items():
            if current_agv != other_agv and next_pos == other_agv.pos: return False
        return True

    def _update_observation(self):
        # 1. Sensing: 현재 AGV들의 물리적 위치를 Controller에 업데이트
        for agv_id, agv_obj in self.agv_list.items():
            self.controller.get_sensing(agv_id, agv_obj.pos)

        # 2. Planning: Controller가 최신 위치 기반으로 D* 경로 및 제어 신호 계산
        self.controller.make_control()

        # 3. State Update: 계산된 경로를 바탕으로 교차로 내부 상태 업데이트
        self._update_intersections_state()

        # 4. Observation Generation: 최종적으로 확정된 상태를 반환
        observations = {}
        active_agents = []
        for intersection in self.intersections:
            if self.deadlock_detector.check_deadlock(intersection):
                observations[intersection.id] = intersection.get_state()
                active_agents.append(intersection.id)
        return observations, active_agents
    
    def _calculate_step_reward(self, agents_who_acted, current_active_agents):
        rewards = {}
        for agent_id in agents_who_acted:
            rewards[agent_id] = -0.01  # 기본 시간 페널티

        for agent_id, was_deadlocked in self.previous_deadlock_states.items():
            is_currently_deadlocked = agent_id in current_active_agents
            if was_deadlocked and not is_currently_deadlocked:
                if agent_id in agents_who_acted:
                    rewards[agent_id] = rewards.get(agent_id, 0) + 1.0
        return rewards
    
    def _get_info(self, active_agents):
        """인자로 받은 active_agents를 사용"""
        return {
            "time": self.time,
            "episode_progress": self.traffic_generator.get_progress(),
            "active_agents": active_agents 
        }
    
    def _update_intersections_state(self):
        for intersection in self.intersections:
            intersection.clear_internal_state()
        for agv_id, agv_obj in self.agv_list.items():
            pos = agv_obj.pos
            if pos in self.coord_to_intersection_map:
                target_intersection = self.coord_to_intersection_map[pos]
                if target_intersection:
                    target_intersection.add_agv(agv_id, pos)
    
    def _build_coord_to_intersection_map(self):
        """
        모든 교차로의 좌표를 키로, 교차로 객체를 값으로 하는 딕셔너리를 생성합니다.
        이 함수는 초기화 시 한 번만 호출됩니다.
        """
        mapping = {}
        for intersection in self.intersections:
            for coord in intersection.all_lane_coords:
                mapping[coord] = intersection
        return mapping

    def _spawn_amrs_if_needed(self):
        while self.traffic_generator.should_spawn_next(self.time):
            task_info = self.traffic_generator.get_next_task()
            if task_info is None: break
            start_intersection_data = random.choice(self.intersection_centers)
            goal_intersection_data = random.choice(self.intersection_centers)
            while start_intersection_data == goal_intersection_data:
                goal_intersection_data = random.choice(self.intersection_centers)
            agv_id = task_info['id']
            start_pos = self._direction_to_coords(task_info['start_direction'], start_intersection_data)
            goal_pos = self._direction_to_coords(task_info['goal_direction'], goal_intersection_data)
            self.agv_list[agv_id] = agv(start_pos, (255, 0, 0))
            self.controller.add_agv(agv_id, start_pos, goal_pos)
            print(f"Time {self.time}: AMR {agv_id} spawned at {start_pos}, heading to goal near {goal_pos}")

    def _direction_to_coords(self, direction, intersection_data):
        center_x, center_y, len_N, len_E, len_S, len_W = intersection_data
        direction_map = {
            'N': (center_x, center_y - len_N), 'E': (center_x + len_E, center_y),
            'S': (center_x, center_y + len_S), 'W': (center_x - len_W, center_y)
        }
        return direction_map[direction]

    def _check_amr_completion(self):
        completed_agvs = []
        for agv_id, agv_obj in self.agv_list.items():
            if agv_obj.pos == self.controller.agv_goal.get(agv_id):
                completed_agvs.append(agv_id)
        for agv_id in completed_agvs:
            self.traffic_generator.complete_task(agv_id, self.time, success=True)
            del self.agv_list[agv_id]
            self.controller.remove_agv(agv_id)
            print(f"Time {self.time}: AMR {agv_id} reached goal!")
    
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
        kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
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
    
    def _find_intersections(self):
        intersections = []
        for r, c in self._find_intersection_center():
            len_N = self._ray_len(r, c, -1, 0)
            len_E = self._ray_len(r, c, 0, 1)
            len_S = self._ray_len(r, c, 1, 0)
            len_W = self._ray_len(r, c, 0, -1)
            if min(len_N, len_E, len_S, len_W) > 0:
                intersections.append((c, r, len_N, len_E, len_S, len_W))
        return intersections

    def close(self):
        pass