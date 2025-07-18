import os
import json
import numpy as np

from AGV import agv
import Funct
import Controller
import Network


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
        
        self.color = Funct.Color_dict(self.agv_num)        
        self.network = Network.network()
        
        self.init_scenario()
    
    def init_scenario(self):
        self.time = 0
        
        # Set controller
        self.controller = Controller.controller(self.agv_num, self.map)
        
        # import AGV start position
        self.load_agents(self.agent_path)

        # AGVs task finish flags
        self.task_done_flags = {num: False for num in self.agv_list}
        
        return 
    
    def reset(self):
        self.init_scenario()

    def get_state(self, num):
        pos = self.agv_list[num].pos                                        # (2,)
        goal = self.agv_list[num].goal                                      # (2,)

        planner = self.controller.planners[num]
        planner.start = pos
        planner.compute_shortest_path()
        distance = planner.g[pos]                                           # (1,)
        
        state = np.array([
            pos[0], pos[1],
            goal[0], goal[1],
            distance,
        ], dtype=np.float32)                                                # (5,)         
        
        return state
    
    def compute_reward(self, state, next_state):
        total_reward = []

        for idx, agv in enumerate(self.agv_list.values()):
            reward = 0

            # Arrive Goal
            if agv.pos == agv.goal:
                reward += 100

            # Wall Collision & Deadlock
            # if agv.mode == 1 or agv.mode == 2:
            #     reward -= 10
                
            # Action penalty
            reward -= 0.2 
            
            total_reward.append(reward)

        return total_reward

    def step(self, state, actions):
        # 1 time step (sec)  
        self.time += 1

        # <1 Step>
        # All AGVs send the sensor signal
        for num, agv in self.agv_list.items():
            # Send the signal to controller through network 
            self.controller.get_sensing(num, self.network.send(agv.sensing()))

        # <2 Step>
        # Controller sends the conntrol signal through network
        control_sig = self.controller.action_control(actions)
        for num, agv in self.agv_list.items():
            agv.get_control(self.network.send([control_sig[0][num], control_sig[1][num]]))

        # <3 Step>
        # All AGVs interacts with ENV!
        dones = []
        for num, agv in self.agv_list.items():
            # Possible Move
            if (self.interact(agv, agv.next_pos()) == 0):
                agv.move()
                agv.goal = self.controller.agv_goal[num][self.controller.agv_state[num]]
                if agv.start == agv.pos and agv.pos == agv.goal:
                    dones.append("success")
                    self.task_done_flags[num] = True
                else:
                    dones.append("alive")
            # Collision with wall or move out of line
            elif (self.interact(agv, agv.next_pos()) == 1):
                dones.append("collision")
                pass                
            # Collision with other AGVs
            elif (self.interact(agv, agv.next_pos()) == 2):
                dones.append("collision")
                pass
        
        next_state = []
        for num in self.agv_list:
            next_state.append(self.get_state(num))

        reward = self.compute_reward(state, next_state)

        return next_state, reward, dones
    
    def demo_step(self, actions):
        # 1 time step (sec)  
        self.time += 1

        if self.time == 1000:
            return False

        # <1 Step>
        # All AGVs send the sensor signal
        for num, agv in self.agv_list.items():
            # Send the signal to controller through network 
            self.controller.get_sensing(num, self.network.send(agv.sensing()))

        # <2 Step>
        # Controller sends the conntrol signal through network
        control_sig = self.controller.action_control(actions)
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
        
        # Stop with 1 hour
        if self.time == 1000:
            return False
        
        # <1 Step>
        # All AGVs send the sensor signal
        for num, agv in self.agv_list.items():
            # Send the signal to controller through network 
            self.controller.get_sensing(num, self.network.send(agv.sensing()))
        
        # <2 Step>
        # Controller sends the conntrol signal through network
        control_sig = self.controller.make_control()
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
        if self.controller.grid[next_pos[1]][next_pos[0]] == 0:
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
    
    def valid_actions(self, x, y, occupied_positions):
        valid = [0, 0, 0, 0, 1]  # [Up, Down, Right, Left, Stop]
        grid = self.controller.grid
        height, width = grid.shape

        occupied = set(occupied_positions)

        if y < height - 1 and grid[y+1][x] == 1 and (x, y+1) not in occupied:
            valid[0] = 1
        if y > 0 and grid[y-1][x] == 1 and (x, y-1) not in occupied:
            valid[1] = 1
        if x < width - 1 and grid[y][x+1] == 1 and (x+1, y) not in occupied:
            valid[2] = 1
        if x > 0 and grid[y][x-1] == 1 and (x-1, y) not in occupied:
            valid[3] = 1

        return valid
    
    
    # ======================== Use for GUI ========================
    def load_map(self, map_path):        
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f"Map file not found")
        
        grid = []
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
            grid.append(row)
        return np.array(grid)
    
    
    def load_agents(self, agent_path):
        map_height, map_width = self.map.shape
        with open(agent_path, 'r') as f:
            lines = f.readlines()

        lines = lines[2:]
        i = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            idx = int(line)
            row = idx // map_width
            col = idx % map_width
            if self.map[row][col] == 1:
                raise ValueError(f"Agent position ({col}, {row}) is not on a walkable cell.")
            self.agv_list[i] = agv((col, row), self.color.dic[i])
            i += 1

    
    def find_line(self, x, y):
        line_list = []
        distance = 0
        poss_x = x
        poss_y = y
        # up
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_x += 1
            distance += 1
            if (self.map[poss_y][poss_x] == 1):
                break
            if (self.map[poss_y][poss_x] == 6):
                line_list.append([poss_x, poss_y])
                break
            if (type(self.map[poss_y][poss_x]) == str):
                line_list.append([poss_x, poss_y])
                break
                
        distance = 0
        poss_x = x
        poss_y = y
        
        # down
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_x -= 1
            distance += 1
            if (self.map[poss_y][poss_x] == 1):
                break
            if (self.map[poss_y][poss_x] == 6):
                line_list.append([poss_x, poss_y])
                break
            if (type(self.map[poss_y][poss_x]) == str):
                line_list.append([poss_x, poss_y])
                break
        
        distance = 0
        poss_x = x
        poss_y = y
        
        # right
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_y += 1
            distance += 1
            if (self.map[poss_y][poss_x] == 6):
                line_list.append([poss_x, poss_y])
                break
            if (type(self.map[poss_y][poss_x]) == str):
                line_list.append([poss_x, poss_y])
                break
            if (self.map[poss_y][poss_x] == 1):
                break
            
        distance = 0
        poss_x = x
        poss_y = y
        
        # left
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_y -= 1
            distance += 1
            if (self.map[poss_y][poss_x] == 1):
                break
            if (self.map[poss_y][poss_x] == 6):
                line_list.append([poss_x, poss_y])
                break
            if (type(self.map[poss_y][poss_x]) == str):
                line_list.append([poss_x, poss_y])
                break
            
        return line_list
