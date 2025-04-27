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
        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                entity = self.map[y][x]
                if type(entity) == str:
                    if (entity[0] == '2'):
                        # Initialize AGV
                        self.agv_list[entity[1]] = agv((x, y), self.color.dic[entity[1]])
                        # set start point
                        self.controller.set_start(entity[1], (x, y))
                        self.controller.agv_pos[entity[1]] = (x, y)   
        
        # controller knows the pick-up, drop, rest position
        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                entity = self.map[y][x]
                if type(entity) == str:
                    if (entity[0] == '3'):
                        # set pick point
                        self.controller.set_pick(entity[1], (x, y)) 
                        
                    if (entity[0] == '4'):
                        # set drop point
                        self.controller.set_drop(entity[1], (x, y)) 
                        
                    if (entity[0] == '5'):
                        # set rest point
                        self.controller.set_rest(entity[1], (x, y)) 

        return
    
    def get_state(self, num, agv):
        pos = agv.pos                                               # (2,)
        pos_type = self.position_type(pos)                          # (1,)

        if pos in self.controller.graph.keys():
            cur_edge_occp = self.current_node_edge_occupancy(pos)   # (4,)
            near_edge_occp = [0, 0, 0, 0] * 4                       # (16,)
        else:
            cur_edge_occp = [0, 0, 0, 0]                            # (4,)
            near_edge_occp = self.near_node_edge_occupancy(pos)     # (16,)
        
        state = np.array([
            pos[0], pos[1],
            pos_type,
            *cur_edge_occp,
            *near_edge_occp
        ], dtype=np.float32)                                        # (24,)         
        
        return state
    
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
            print(dx, dy)
            found = None
            for neighbor in neighbors:
                nx, ny = neighbor
                diff_x, diff_y = nx - x, ny - y
                if (dx, dy) == (0, 1) and diff_x == 0 and diff_y > 0:           # Up
                    print('Up found: ', nx, ny)
                    found = neighbor
                    break
                elif (dx, dy) == (0, -1) and diff_x == 0 and diff_y < 0:        # Down
                    print('Down found: ', nx, ny)
                    found = neighbor
                    break
                elif (dx, dy) == (1, 0) and diff_y == 0 and diff_x > 0:         # Right
                    print('Right found:', nx, ny)
                    found = neighbor
                    break
                elif (dx, dy) == (-1, 0) and diff_y == 0 and diff_x < 0:        # Left
                    print('Left found: ', nx, ny)
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
    
env = ENV()
print(env.agv_list.keys())
print(env.controller.graph.get((45, 12)))

g = env.agv_list['G']
o = env.agv_list['O']
h = env.agv_list['H']
j = env.agv_list['J']

g.pos = (45, 12)
o.pos = (47, 12)
h.pos = (48, 12)
j.pos = (46, 12)

print(env.get_state('G', g))