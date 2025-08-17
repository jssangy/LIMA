import os
import json
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict

from utils.AGV import agv
from utils.Intersection import Intersection
from utils.DeadlockDetector import DeadlockDetector
from utils import Funct
from utils.traffic_generator import TrafficGenerator
from utils.train_controller import controller


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
        self.intersections = [Intersection(data, self.controller) for data in self.intersection_data]

        # [최적화] 좌표-교차로 매핑을 미리 생성
        self.id_to_intersection = {inter.id: inter for inter in self.intersections}
        self.coord_to_intersection_map = self._build_coord_to_intersection_map()

        self.agent_ids = sorted([inter.id for inter in self.intersections])

        self.previous_deadlock_states = {}

    def _init_gym_spaces(self):
        """
        패딩을 적용한 고정 크기 관측 공간 정의
        - 모든 공간의 크기는 '서브그래프'의 최대 크기를 기준으로 합니다.
        - 실제 데이터가 어디까지인지를 알리는 'mask'를 추가합니다.
        """
        # 1. 서브그래프의 최대 크기 정의 (상황에 맞게 조절 가능)
        self.N_MAX = 5          # 서브그래프에 포함될 수 있는 최대 노드 수
        self.E_MAX = 4          # 서브그래프에 포함될 수 있는 최대 엣지 수
        self.STATE_DIM = 24     # 노드 특징 벡터의 차원

        # 2. 관측 공간(Observation Space) 정의
        # 모든 관측은 self.N_MAX 크기에 맞춰 패딩됩니다.
        self.observation_space = spaces.Dict({
            # 노드 특징: (최대 노드 수, 특징 차원)
            "nodes": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.N_MAX, self.STATE_DIM),
                dtype=np.float32
            ),
            # 엣지 인덱스: (2, 최대 엣지 수)
            "edge_index": spaces.Box(
                low=0, high=self.N_MAX - 1,
                shape=(2, self.E_MAX),
                dtype=np.int64
            ),
            # 활성 에이전트 마스크: (최대 노드 수,)
            # 서브그래프 내에서 '행동해야 하는' 에이전트(데드락 발생 주체)를 표시
            "active_mask": spaces.Box(
                low=0, high=1,
                shape=(self.N_MAX,),
                dtype=np.bool_
            ),
            # 노드 마스크: (최대 노드 수,)
            # 패딩된 데이터와 '실제' 노드 데이터를 구분하기 위한 필수 정보
            "nodes_mask": spaces.Box(
                low=0, high=1,
                shape=(self.N_MAX,),
                dtype=np.bool_
            )
        })
        
        # 3. 행동 공간(Action Space) 정의
        # 이 부분은 전체 에이전트에 대해 정의하는 것이 일반적입니다.
        # 훈련 시에는 active_mask를 통해 실제 행동을 취한 에이전트만 선별합니다.
        self.action_space = spaces.Dict({
            agent_id: spaces.MultiDiscrete([4, 4, 4, 4, 4]) for agent_id in self.agent_ids
        })

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
        observations, info = self._generate_gnn_observation()
        self.previous_deadlock_states = {inter.id: (inter.id in info["active_agents"]) for inter in self.intersections}
        return observations, info

    def step(self, actions=None, train=False):
        """한 스텝 '실행'에만 집중"""
        self.time += 1

        # 1. Action 적용: 전달받은 actions을 기반으로 제어 신호 수정
        if actions:
            for agent_id, action in actions.items():
                if agent_id in self.id_to_intersection:
                    self.id_to_intersection[agent_id].action_control(action)

        # 2. Movement: 수정된 제어 신호에 따라 모든 AGV 이동
        for agv_id, agv_obj in list(self.agv_list.items()):
            control_sig = self.controller.control_buffer[agv_id]
            if self._is_valid_move(agv_obj, control_sig):
                agv_obj.move(control_sig)
        
        # 3. 환경 변화 처리: AGV 완료 및 신규 생성
        self._check_amr_completion()
        self._spawn_amrs_if_needed()

        # 4. 다음 의사결정을 위한 상태 계산
        observation, info = self._generate_gnn_observation()

        # 5. 보상 계산
        if train:
            rewards = self._calculate_step_reward(actions.keys() if actions else [], info["active_agents"])

        self.previous_deadlock_states = {inter.id: (inter.id in info["active_agents"]) for inter in self.intersections}

        # 6. 종료 조건 확인
        episode_terminated = self.traffic_generator.is_episode_done() and not self.agv_list
        terminated = {agent_id: episode_terminated for agent_id in self.agent_ids}
        terminated["__all__"] = episode_terminated
        truncated = False

        if not train:
            return self.make_info()

        return observation, rewards, terminated, truncated, info
    
    def _generate_gnn_observation(self):
        """
        [수정] 가변 크기의 서브그래프 데이터를 패딩 없이 그대로 반환합니다.
        """
        # ... (물리적 상태 업데이트 및 데드락 탐지 로직은 동일) ...
        for agv_id, agv_obj in self.agv_list.items():
            self.controller.get_sensing(agv_id, agv_obj.pos)
        self.controller.make_control()
        self._update_intersections_state()
        event_agent_ids = {inter.id for inter in self.intersections if self.deadlock_detector.check_deadlock(inter)}

        # --- 패딩 준비: 최대 크기로 빈 배열 생성 ---
        nodes_padded = np.zeros((self.N_MAX, self.STATE_DIM), dtype=np.float32)
        edge_index_padded = np.zeros((2, self.E_MAX), dtype=np.int64)
        active_mask_padded = np.zeros(self.N_MAX, dtype=np.bool_)
        nodes_mask = np.zeros(self.N_MAX, dtype=np.bool_)
        
        info = {"active_agents": list(event_agent_ids), "subgraph_nodes": []}

        # --- 서브그래프 구성 ---
        if not event_agent_ids:
            # 이벤트가 없으면 0으로 채워진 고정 크기 관측값 반환
            observation = {
                "nodes": nodes_padded,
                "edge_index": edge_index_padded,
                "active_mask": active_mask_padded,
                "nodes_mask": nodes_mask
            }
            return observation, info

        # L-hop 서브그래프 구성
        subgraph_nodes = set(event_agent_ids)
        if self.l_hop > 0:
            for agent_id in list(event_agent_ids):
                subgraph_nodes.update(self.full_adjacency.get(agent_id, []))
        
        subgraph_agent_ids = sorted(list(subgraph_nodes))
        info['subgraph_nodes'] = subgraph_agent_ids

        num_real_nodes = len(subgraph_agent_ids)
        subgraph_map = {agent_id: i for i, agent_id in enumerate(subgraph_agent_ids)}

        # 실제 서브그래프 데이터 생성
        node_features = np.array([self.id_to_intersection[aid].get_state() for aid in subgraph_agent_ids], dtype=np.float32)
        
        subgraph_edges = []
        for u_id in subgraph_agent_ids:
            for v_id in self.full_adjacency.get(u_id, []):
                if v_id in subgraph_map:
                    subgraph_edges.append([subgraph_map[u_id], subgraph_map[v_id]])
        
        edge_index = np.array(subgraph_edges, dtype=np.int64).T if subgraph_edges else np.empty((2, 0), dtype=np.int64)
        active_mask = np.array([aid in event_agent_ids for aid in subgraph_agent_ids], dtype=np.bool_)

        # 실제 데이터를 패딩된 배열에 복사
        nodes_padded[:num_real_nodes] = node_features
        edge_index_padded[:, :edge_index.shape[1]] = edge_index
        active_mask_padded[:num_real_nodes] = active_mask
        nodes_mask[:num_real_nodes] = True

        observation = {
            "nodes": nodes_padded,
            "edge_index": edge_index_padded,
            "active_mask": active_mask_padded,
            "nodes_mask": nodes_mask
        }

        # --- [추가] 디버깅을 위한 전체 관측값 출력 ---
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            print("\n--- [DEBUG] GNN Observation ---")
            print(f"Time: {self.time}")
            print(f"Active Agents (Deadlock): {info['active_agents']}")
            print(f"Subgraph Nodes (Neighbors): {info['subgraph_nodes']}")
            for key, value in observation.items():
                print(f"  - {key}:")
                print(value)
            print("---------------------------------\n")
        # -----------------------------------------
        
        return observation, info
    
    def _is_valid_move(self, current_agv, control_signal):
        next_pos = (current_agv.pos[0] + control_signal[0], current_agv.pos[1] + control_signal[1])
        if self.map[next_pos[1]][next_pos[0]] == 1: return False
        for agv_id, other_agv in self.agv_list.items():
            if current_agv != other_agv and next_pos == other_agv.pos: return False
        return True
    
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
            intersection.reset()
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
            'N': (center_x, center_y - len_N), 'E': (center_x + len_E, center_y),
            'S': (center_x, center_y + len_S), 'W': (center_x - len_W, center_y)
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
            mode = 0 # 0: Normal
            for inter_id, is_deadlocked in self.previous_deadlock_states.items():
                if is_deadlocked:
                    intersection = next((inter for inter in self.intersections if inter.id == inter_id), None)
                    if intersection and agv_id in intersection.agvs_in_intersection:
                        mode = 2 # 2: Deadlock
                        break
            
            agv_states[agv_id] = [f"Goal_{agv_id}", mode]

        return [total_prod, throughput, agv_states]