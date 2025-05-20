import numpy as np

from AGV import agv
from map import map
import Funct
import Controller
import Network

class ENV():    
    def __init__(self):
        # number of agvs
        self.agv_num = 0
        # import map
        self.map = map
        # agv_list[alphabet] = agv object
        self.agv_list = {}
        
        # Find number of AGVs
        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                entity = self.map[x][y]
                if type(entity) == str:
                    if (entity[1] in self.agv_list):
                        pass
                    else:
                        self.agv_list[entity[1]] = True
                        self.agv_num += 1
                        
        self.color = Funct.Color_dict(self.agv_num)
        
        self.network = Network.network()
        
        self.init_scenario()
    
    def init_scenario(self):
        self.time = 0
        
        # Find number of AGVs
        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                entity = self.map[x][y]
                if type(entity) == str:
                    if (entity[1] in self.agv_list):
                        pass
                    else:
                        self.agv_list[entity[1]] = True

        # Set controller
        self.controller = Controller.controller(self.agv_num, self.map)
        
        # define AGV with start position (controller knows the start position)
        for x in range(len(self.map[0])):
            for y in range(len(self.map)):
                entity = self.map[y][x]
                if type(entity) == str:
                    if (entity[0] == '2'):
                        # Initialize AGV
                        self.agv_list[entity[1]] = agv((x, y), self.color.dic[entity[1]])
                        # set start point
                        self.controller.set_start(entity[1], (x, y))
                        self.controller.agv_pos[entity[1]] = (x, y)
        
        # controller knows the pick-up, drop, rest position
        for x in range(len(self.map[0])):
            for y in range(len(self.map)):
                entity = self.map[y][x]
                if type(entity) == str:
                    if (entity[0] == '3'):
                        # set pick point
                        self.controller.set_pick(entity[1], (x, y))
                        self.agv_list[entity[1]].goal = (x, y)
                        self.controller.set_control(entity[1])
                        
                    if (entity[0] == '4'):
                        # set drop point
                        self.controller.set_drop(entity[1], (x, y)) 
                        
                    if (entity[0] == '5'):
                        # set rest point
                        self.controller.set_rest(entity[1], (x, y)) 

        self.agv_list = dict(sorted(self.agv_list.items()))

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
                reward += 10

            # Wall Collision & Deadlock
            if agv.mode == 1 or agv.mode == 2:
                reward -= 5

            # Distance difference
            cur_dist = state[idx][-1]
            next_dist = next_state[idx][-1]
            if cur_dist > next_dist:
                reward += 0.2 
            elif cur_dist <= next_dist or agv.mode == 1 or agv.mode == 2:
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
                    dones.append(True)
                else:
                    dones.append(False)
            # Collision with wall or move out of line
            elif (self.interact(agv, agv.next_pos()) == 1):
                dones.append(True)
                pass                
            # Collision with other AGVs
            elif (self.interact(agv, agv.next_pos()) == 2):
                dones.append(True)
                pass
        
        next_state = []
        for num in self.agv_list:
            next_state.append(self.get_state(num))

        reward = self.compute_reward(state, next_state)

        return next_state, reward, dones
    
    def demo_step(self, actions):
        # 1 time step (sec)  
        self.time += 1

        if self.time == 3600:
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
        if self.time == 3600:
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

    def position_type(self, pos):
        x, y = pos
        grid = self.controller.grid
        val = self.map[y][x]
        if val == 6 or isinstance(val, str):
            return 0
        
        up = (y+1 < grid.shape[0]) and (grid[y+1][x] == 1)
        down = (y-1 >= 0) and (grid[y-1][x] == 1)
        if up or down:
            return 1

        right = (x+1 < grid.shape[1]) and (grid[y][x+1] == 1)
        left = (x-1 >= 0) and (grid[y][x-1] == 1)
        if right or left:
            return 2

        raise Exception(f"[Error] Invalid AGV position: {pos}")
    
    def current_node_edge_occupancy(self, pos):
        x, y = pos
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        status = []
        neighbors = self.controller.graph.get(pos)

        for dx, dy in directions:
            found = None
            for neighbor in neighbors:
                nx, ny = neighbor
                diff_x, diff_y = nx - x, ny - y
                if (dx, dy) == (0, 1) and diff_x == 0 and diff_y > 0:           # Up
                    found = neighbor
                    break
                elif (dx, dy) == (0, -1) and diff_x == 0 and diff_y < 0:        # Down
                    found = neighbor
                    break
                elif (dx, dy) == (1, 0) and diff_y == 0 and diff_x > 0:         # Right
                    found = neighbor
                    break
                elif (dx, dy) == (-1, 0) and diff_y == 0 and diff_x < 0:        # Left
                    found = neighbor
                    break
            
            if found is None:
                status.append(0)
            else:
                occupied = 0
                min_x, max_x = sorted([x, found[0]])
                min_y, max_y = sorted([y, found[1]])
                for agv in self.agv_list.values():
                    if min_x == max_x and max_x == agv.pos[0] and min_y < agv.pos[1] < max_y:
                        occupied = 1
                        break 
                    elif min_x < agv.pos[0] < max_x and min_y == max_y and max_y == agv.pos[1]:
                        occupied = 1
                        break
                status.append(occupied)

        return status
    
    def near_node_edge_occupancy(self, pos):
        x, y = pos
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        candidates = []
        status = []

        for dx, dy in directions:
            poss_x, poss_y = x, y
            distance = 0
            found = False
            while distance < 15 and 1 <= poss_x < 99 and 1 <= poss_y < 99:
                poss_x += dx
                poss_y += dy
                distance += 1
                val = self.map[poss_y][poss_x]
                if val == 1:
                    break
                if val == 6 or isinstance(val, str):
                    candidates.append((poss_x, poss_y))
                    found = True
                    break
            if not found:
                candidates.append((-1, -1))

        for node in candidates:
            if node == (-1, -1):
                status.extend([0, 0, 0, 0])
            else:
                node_status = self.current_node_edge_occupancy(node)
                status.extend(node_status)
        
        return status
    
    def valid_actions(self, x, y):
        valid = [0, 0, 0, 0, 1]  # [Up, Down, Right, Left, Stop]
        grid = self.controller.grid
        height, width = grid.shape

        # Up
        if y < height - 1 and grid[y+1][x] == 1:
            valid[0] = 1
        # Down
        if y > 0 and grid[y-1][x] == 1:
            valid[1] = 1
        # Right
        if x < width - 1 and grid[y][x+1] == 1:
            valid[2] = 1
        # Left
        if x > 0 and grid[y][x-1] == 1:
            valid[3] = 1

        return valid    
    
    # ======================== Use for GUI ========================
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
