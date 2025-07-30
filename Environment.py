import os
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from AGV import agv
import Funct
import Network
from Controller import controller
from Intersection import Intersection


class ENV():    
    def __init__(self, prob_path):
        # number of agvs
        self.agv_num = 0
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

    def step(self, action):
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