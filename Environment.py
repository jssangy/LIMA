import os
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch

from AGV import agv
import Funct
import Network
from Controller import controller
from Intersection import Intersection


class ENV():    
    def __init__(self, prob_path):
        # agv_list[alphabet] = agv object
        self.agv_list = {}

        # Load problem path
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])
        self.agent_path = os.path.join(base_dir, data['agentFile'])
        self.task_path = os.path.join(base_dir, data['taskFile'])
        
        # number of agvs
        self.agv_num = data['teamSize']

        # import map            
        self.map = self.load_map(map_path)    

        # Find intersections in the map (x, y, len_N, len_E, len_S, len_W)
        self.intersection_centers = self.find_intersections()
        
        self.color = Funct.Color_dict(self.agv_num)        
        self.network = Network.network()

        # Import tasks [[task_pick 0, task_drop 0], ...]
        self.tasks = self.load_tasks(self.task_path)
        
        # RL 정책 관련 설정
        self.rl_policy = None
        self.use_rl = False
        
        self.init_scenario()
    
    def init_scenario(self):
        self.time = 0
        
        # Import AGV start position
        self.agv_list = self.load_agents(self.agent_path)
        
        # Set controller
        self.controller = controller(self.agv_num, self.map, self.agv_list, self.tasks, self.intersection_centers)

        # Set Intersection controller
        self.intersection = Intersection(self.intersection_centers[0], self.controller)

        return

    def reset(self):
        self.init_scenario()
    
    def set_rl_policy(self, policy):
        """RL 정책을 설정합니다. 사용 여부는 별도로 제어됩니다."""
        self.rl_policy = policy
        # use_rl은 GUI에서 체크박스로 제어하므로 여기서 설정하지 않음
    
    def get_observation_for_rl(self):
        """RL을 위한 관찰 상태를 반환합니다."""
        if not self.use_rl or not self.intersection:
            return None
        return self.intersection.get_state()
    
    def get_rl_action(self):
        """RL 정책을 사용해 액션을 생성합니다."""
        if not self.use_rl or not self.rl_policy:
            return [0, 0, 0, 0, 0]  # 기본 액션 (모든 힘 0)
        
        observation = self.get_observation_for_rl()
        if observation is None:
            return [0, 0, 0, 0, 0]
        
        # 모델이 있는 디바이스 확인
        device = next(self.rl_policy.parameters()).device
        
        with torch.no_grad():
            # 관찰값을 텐서로 변환하고 모델과 같은 디바이스로 이동
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
            
            # TensorDict 형태로 변환 (TorchRL 모델은 TensorDict를 입력으로 받음)
            from tensordict import TensorDict
            td_input = TensorDict({"observation": obs_tensor}, batch_size=[1])
            
            # 모델 실행
            td_output = self.rl_policy(td_input)
            
            # 액션 추출
            if "action" in td_output.keys():
                action_tensor = td_output["action"]
            else:
                # 키가 없는 경우 기본 액션 반환
                return [0, 0, 0, 0, 0]
            
            # [핵심 수정] 원핫 벡터를 정수 인덱스로 변환
            action_tensor = action_tensor.cpu().squeeze()  # GPU -> CPU, 배치 차원 제거
            
            # OneHot 벡터가 2차원인 경우 ([5, 4] 형태)
            if action_tensor.dim() == 2:
                # 각 행에서 최대값의 인덱스를 찾아 정수 액션으로 변환
                action_indices = torch.argmax(action_tensor, dim=1).tolist()
                return action_indices
            
            # 1차원인 경우 (이미 인덱스 형태)
            elif action_tensor.dim() == 1:
                action_indices = action_tensor.tolist()
                return action_indices
            
            else:
                return [0, 0, 0, 0, 0]

    def step(self, action=None, test_mode=False):
        # 1 time step (sec)  
        self.time += 1

        if self.controller.task_count >= len(self.tasks):
            return False
        
        # <1 Step>
        # All AGVs send the sensor signal
        for num, agv in self.agv_list.items():
            # Send the signal to controller through network 
            self.controller.get_sensing(num, self.network.send(agv.sensing()))
        
        # <2 Step>
        # Controller sends the conntrol signal through network
        self.controller.make_control()
        
        if action is not None:
            self.intersection.action_control(action)
        # GUI 모드에서 RL 사용 시에만 자체 액션 생성
        elif test_mode and self.use_rl and self.rl_policy:
            action = self.get_rl_action()
            self.intersection.action_control(action)

        control_sig = self.controller.get_control_sig()
        for num, agv in self.agv_list.items():
            agv.get_control(self.network.send([control_sig[0][num], control_sig[1][num]]))
            
        # <3 Step>
        # All AGVs interacts with ENV!
        for num, agv in self.agv_list.items():
            # Possible Move
            if (self.interact(agv, agv.next_pos()) == 0):
                agv.move()
                agv.goal = self.controller.agv_goal[num][self.controller.agv_state[num]]
            # Collision with wall or move out of line
            elif (self.interact(agv, agv.next_pos()) == 1):
                pass                
            # Collision with other AGVs
            elif (self.interact(agv, agv.next_pos()) == 2):
                pass

        return self.make_info()

    # Single Process Step
    def Run(self):
        # 1 time step (sec)  
        self.time += 1

        if self.controller.task_count >= len(self.tasks):
            print("All tasks completed.")
            return False
        
        # <1 Step>
        # All AGVs send the sensor signal
        for num, agv in self.agv_list.items():
            # Send the signal to controller through network 
            self.controller.get_sensing(num, self.network.send(agv.sensing()))
        
        # <2 Step>
        # Controller sends the conntrol signal through network
        self.controller.make_control()
        control_sig = self.controller.get_control_sig()
        for num, agv in self.agv_list.items():
            agv.get_control(self.network.send([control_sig[0][num], control_sig[1][num]]))
            
        # <3 Step>
        # All AGVs interacts with ENV!
        for num, agv in self.agv_list.items():
            # Possible Move
            if (self.interact(agv, agv.next_pos()) == 0):
                agv.move()
                agv.goal = self.controller.agv_goal[num][self.controller.agv_state[num]]
            # Collision with wall or move out of line
            elif (self.interact(agv, agv.next_pos()) == 1):
                pass                
            # Collision with other AGVs
            elif (self.interact(agv, agv.next_pos()) == 2):
                pass

        return self.make_info()
    
    def interact(self, agv, next_pos):
        if self.map[next_pos[1]][next_pos[0]] == 1:
            agv.mode = 1
            return 1

        for other_agv in self.agv_list.values():
            if (agv != other_agv and next_pos == other_agv.pos):
                agv.mode = 2
                return 2
        
        agv.mode = 0
        return 0
    
    # Get the list of object
    def Get_AGV(self):
        return self.agv_list
    
    def make_info(self):
        # Use for GUI
        if (self.time != 0):
            info_list = [self.controller.whole_product, self.controller.whole_product / self.time]
        else:
            info_list = [self.controller.whole_product, 0]
        
        # Product of AGVs
        info_list.append(self.controller.agv_info)
        
        return info_list
    
    def get_active_tasks(self):
        return self.controller.get_active_tasks()

    def load_map(self, map_path):        
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f"Map file not found")
        
        map = []
        with open(map_path, 'r') as f:
            lines = f.readlines()
        map_start = None
        for idx, line in enumerate(lines):
            if line.strip() == 'map':
                map_start = idx + 1
                break
            
        for line in lines[map_start:]:
            row = []
            for c in line.strip():
                if c in ['@', 'T']:
                    row.append(1)
                elif c in ['.', 'E', 'S']:
                    row.append(0)
                else:
                    raise ValueError(f"Invalid character in map file")
            map.append(row)
            
        return np.array(map)
    
    
    def load_agents(self, agent_path):
        _, map_width = self.map.shape
        agv_list = {}
        with open(agent_path, 'r') as f:
            lines = f.readlines()

        lines = lines[2:]
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            idx = int(line)
            row = idx // map_width
            col = idx % map_width
            if self.map[row][col] == 1:
                raise ValueError(f"Agent position ({col}, {row}) is not on a walkable cell.")
            agv_list[i] = agv((col, row), self.color.dic[i])

        return agv_list

    def load_tasks(self, task_path):
        _, map_width = self.map.shape
        tasks = []
        with open(task_path, 'r') as f:
            lines = f.readlines()

        lines = lines[2:]
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            indices = [int(x) for x in line.replace(',', ' ').strip().split()]
            path = [(idx % map_width, idx // map_width) for idx in indices]                

            for col, row in path:
                if self.map[row][col] == 1:
                    raise ValueError(f"Task path ({col}, {row}) is not on a walkable cell.")
                
            tasks.append(path)

        return tasks

    def find_intersection_center(self):
        kernel = np.array([[1, 0, 1],
                           [0, 0, 0],
                           [1, 0, 1]])

        windows = sliding_window_view(self.map, kernel.shape)
        matches = np.all(windows == kernel, axis=(2, 3))
        centers = (np.argwhere(matches) + 1).tolist()

        return centers
        
    def ray_len(self, r, c, dr, dc):
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
        intersections = []
        for r, c in self.find_intersection_center():
            len_N = self.ray_len(r, c, -1, 0)   # North
            len_E = self.ray_len(r, c, 0, 1)    # East
            len_S = self.ray_len(r, c, 1, 0)    # South
            len_W = self.ray_len(r, c, 0, -1)   # West

            if min(len_N, len_E, len_S, len_W) > 0:
                intersections.append((c, r, len_N, len_E, len_S, len_W))

        return intersections
    
    def get_active_intersections(self):
        # Since we assume a single intersection, we just return it in a list
        # to maintain compatibility with gym_env.
        return [self.intersection]
    
    def is_intersection_empty(self):
        """교차로가 비어있는지 여부를 반환합니다."""
        return self.intersection.is_empty