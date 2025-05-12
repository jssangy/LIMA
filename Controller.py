import heapq
import numpy as np # use for matrix calculation

import Funct

class controller():    
    def __init__(self, agv_num, map):
        self.agv_pos = {} # save the position of agv positions
        self.agv_next_pos = {} # save the next position of agv positions
        self.agv_next_rout = {} # save the next rout position of agv
        self.control_buffer = {} # save the D* algorithm based control output of agvs
        self.action_control_buffer = {} # save the action control output of agvs
        self.agv_state = {} # 0(start - pick up) 1(pick up - drop) 2(drop - rest) 3(rest - start)
        self.agv_nums = [] # agv numbers (A, B, C, ... O)
        self.agv_mode = {} # 0 (normal) 1 (Danger)
        self.agv_goal = {} # goal position of all agvs
        self.agv_info = {} # for GUI infomation
        self.agv_rout = {} # for routing of AGV
        self.agv_pre_rout = {} # previous node
        self.planners = {} # D* class for each AGV
        self.prev_distance = {}

        self.running_opt = 0
        
        # Whole Products
        self.whole_product = 0
            
        # Initialization
        for i in range (agv_num):
            self.agv_nums.append(chr(i + 65))
            self.agv_pos[chr(i + 65)] = (0, 0)
            self.agv_next_rout[chr(i + 65)] = (0, 0)
            self.agv_state[chr(i + 65)] = 0 # Initial state is 0
            self.agv_mode[chr(i + 65)] = 0 # Initial mode is normal
            self.agv_goal[chr(i + 65)] = [(0, 0), (0, 0), (0, 0), (0, 0)]
            self.agv_info[chr(i + 65)] = [0, 0]
            self.control_buffer[chr(i + 65)] = (0, 0)
            self.action_control_buffer[chr(i + 65)] = (0, 0)
            self.agv_rout[chr(i + 65)] = []
            self.agv_pre_rout[chr(i + 65)] = (0, 0)
        
        # Map of warehouse digital twin
        self.map = map
        
        # Make graph & grid for routing
        self.graphing()
        self.make_grid()
    
    # set start position
    def set_start(self, num, pos):
        self.agv_goal[num][3] = pos
    
    # set pick-up position
    def set_pick(self, num, pos):
        self.agv_goal[num][0] = pos
        
    # set drop position
    def set_drop(self, num, pos):
        self.agv_goal[num][1] = pos
        
    # set rest position
    def set_rest(self, num, pos):
        self.agv_goal[num][2] = pos
    
    # Change the AGV's state
    def change_state(self, num, state):
        if state != 3:
            self.agv_state[num] = state + 1   
        else:
            self.agv_state[num] = 0
            self.agv_info[num][0] += 1
            self.whole_product += 1
        
        return self.agv_state[num]
    
    def set_control(self, num):
        pos = self.agv_pos[num]
        goal = self.agv_goal[num][self.agv_state[num]]
        self.planners[num] = DStar(self.grid, pos, goal)
        self.planners[num].compute_shortest_path()

        path = self.planners[num].extract_path()
        self.agv_rout[num] = path
        if len(path) >= 2:
            self.agv_next_rout[num] = path[1]
        else:
            self.agv_next_rout[num] = pos

    # Update data from sensing of agv
    def get_sensing(self, num, data):
        if data != None:
            self.agv_pos[num] = data[0]
            self.agv_mode[num] = data[1]
            self.agv_info[num][1] = data[1]
                
    def action_control(self, actions):
        self.action(actions)
        return (self.action_control_buffer, self.agv_mode)
    
    def action(self, actions):
        for idx, num in enumerate(self.agv_nums):
            control = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]

            if pos == goal:
                state = self.change_state(num, state)
                new_goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0                
                self.planners[num] = DStar(self.grid, pos, new_goal)
                self.planners[num].compute_shortest_path()
                path = self.planners[num].extract_path()
                self.prev_distance[num] = self.planners[num].g.get(pos)
            else:
                self.planners[num].start = pos
                self.planners[num].compute_shortest_path()
                path = self.planners[num].extract_path()
                self.prev_distance[num] = self.planners[num].g.get(pos)

            if len(path) >= 2:
                next_pos = path[1]
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]
                self.control_buffer[num] = (dx, dy)
                self.action_control_buffer[num] = control[actions[idx]]
                self.agv_next_pos[num] = pos + self.action_control_buffer[num]

        # Collision prevention => Dead Lock
        for num1 in self.agv_nums:
            num1_pos = self.agv_pos[num1]
            num1_next_pos = self.agv_next_pos[num1]

            self.agv_mode[num1] = 0
            for num2 in self.agv_nums:
                if num1 != num2:
                    num2_pos = self.agv_pos[num2]
                    num2_next_pos = self.agv_next_pos[num2]
                    if (num1_next_pos == num2_next_pos):
                        self.agv_mode[num1] = 1
                    elif (num1_next_pos == num2_pos):
                        self.agv_mode[num1] = 1
          
            if self.map[num1_pos[1]][num1_pos[0]] == 1:
                self.agv_mode[num1] = 2
                self.control_buffer[num1] = (0, 0) 
    
    def make_control(self):
        if self.running_opt == 0:
            self.dstar_rout()
        elif self.running_opt == 1:
            self.dynamic_obstacle_dstar_rout()
        return (self.control_buffer, self.agv_mode)
    
    def dstar_rout(self):
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]

            if pos == goal:
                state = self.change_state(num, state)
                goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0                
                self.planners[num] = DStar(self.grid, pos, goal)
                self.planners[num].compute_shortest_path()
                path = self.planners[num].extract_path()
                self.prev_distance[num] = self.planners[num].g.get(pos)
            else:
                self.planners[num].start = pos
                self.planners[num].compute_shortest_path()
                path = self.planners[num].extract_path()
                self.prev_distance[num] = self.planners[num].g.get(pos)

            if len(path) >= 2:
                next_pos = path[1]
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]
                self.control_buffer[num] = (dx, dy)
                self.agv_next_pos[num] = next_pos
            else:
                self.control_buffer[num] = (0, 0)
                self.agv_next_pos[num] = pos

        # Collision prevention => Dead Lock
        for num1 in self.agv_nums:
            num1_pos = self.agv_pos[num1]
            num1_next_pos = self.agv_next_pos[num1]

            self.agv_mode[num1] = 0
            for num2 in self.agv_nums:
                if num1 != num2:
                    num2_pos = self.agv_pos[num2]
                    num2_next_pos = self.agv_next_pos[num2]
                    if (num1_next_pos == num2_next_pos):
                        self.agv_mode[num1] = 1
                    elif (num1_next_pos == num2_pos):
                        self.agv_mode[num1] = 1
          
            if self.map[num1_pos[1]][num1_pos[0]] == 1:
                self.agv_mode[num1] = 2
                self.control_buffer[num1] = (0, 0) 
    
    def dynamic_obstacle_dstar_rout(self):
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]

            dynamic_grid = self.grid.copy()
            for other in self.agv_nums:
                if other != num:
                    ox, oy = self.agv_pos[other]
                    if 0 <= ox < dynamic_grid.shape[1] and 0 <= oy < dynamic_grid.shape[0]:
                        if not isinstance(self.map[oy][ox], str):
                            dynamic_grid[oy][ox] = 0

            if pos == goal:
                state = self.change_state(num, state)
                goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0

                
            self.planners[num] = DStar(dynamic_grid, pos, goal)
            self.planners[num].compute_shortest_path()
            path = self.planners[num].extract_path()
            self.agv_rout[num] = path

            if len(path) >= 2:
                next_pos = path[1]
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]
                self.control_buffer[num] = (dx, dy)
                self.agv_next_pos[num] = next_pos
            else:
                self.control_buffer[num] = (0, 0)
                self.agv_next_pos[num] = pos

        # Collision prevention => Dead Lock
        for num1 in self.agv_nums:
            num1_pos = self.agv_pos[num1]
            num1_next_pos = self.agv_next_pos[num1]

            self.agv_mode[num1] = 0
            for num2 in self.agv_nums:
                if num1 != num2:
                    num2_pos = self.agv_pos[num2]
                    num2_next_pos = self.agv_next_pos[num2]
                    if (num1_next_pos == num2_next_pos):
                        self.agv_mode[num1] = 1
                    elif (num1_next_pos == num2_pos):
                        self.agv_mode[num1] = 1
          
            if self.map[num1_pos[1]][num1_pos[0]] == 1:
                self.agv_mode[num1] = 2
                self.control_buffer[num1] = (0, 0)   
        
    # ======================== Routing Functions ============================================
    def graphing(self):
        self.graph = {}
        for x in range (len(self.map[0])): # 35
            for y in range(len(self.map)): # 18
                # normal node 
                if self.map[y][x] == 6:
                    neighbors = self.find_neighbors(x,y)
                    nodes = {}
                    for neighbor in neighbors:
                        nodes[neighbor] = Funct.get_distance((x,y), neighbor)
                    self.graph[(x,y)] = nodes
                    
                if type(self.map[y][x]) == str:
                    neighbors = self.find_neighbors(x,y, False)
                    nodes = {}
                    for neighbor in neighbors:
                        nodes[neighbor] = Funct.get_distance((x,y), neighbor)
                    self.graph[(x,y)] = nodes  
    
    def find_neighbors(self, x, y, rout=True):
        line_list = []
        for direction in ['up', 'down', 'left', 'right']:
            dx, dy = {'up': (1, 0), 'down': (-1, 0), 'right': (0, 1), 'left': (0, -1)}[direction]
            distance, poss_x, poss_y = 0, x, y
            while distance < 15 and 1 <= poss_x < len(self.map[0])-1 and 1 <= poss_y < len(self.map)-1:
                poss_x += dx
                poss_y += dy
                distance += 1
                val = self.map[poss_y][poss_x]
                if val == 1:
                    break
                if val == 6 or (isinstance(val, str) and rout):
                    line_list.append((poss_x, poss_y))
                    break
        return line_list
    
    def make_grid(self):
        height, width = len(self.map), len(self.map[0])
        self.grid = np.zeros((height, width), dtype=np.uint8)
        white_cells = set(self.graph.keys())

        for start, neighbors in self.graph.items():
            for end in neighbors:
                x0, y0 = start
                x1, y1 = end
                steps = max(abs(x1 - x0), abs(y1 - y0))
                for i in range(steps + 1):
                    xi = x0 + round((x1 - x0) * i / steps)
                    yi = y0 + round((y1 - y0) * i / steps)
                    white_cells.add((xi, yi))

        for (x, y) in white_cells:
            if 0 <= x < width and 0 <= y < height:
                self.grid[y][x] = 1
        
        return self.grid
    
# D* Lite Algorithm
class DStar:
    def __init__(self, grid_map, start, goal):
        self.map = grid_map
        self.start = start
        self.goal = goal

        self.g = {}          # Actual cost
        self.rhs = {}        # Estimated cost
        self.queue = []      # Priority queue

        h, w = grid_map.shape
        for y in range(h):
            for x in range(w):
                if grid_map[y][x] == 1:
                    self.g[(x, y)] = float('inf')
                    self.rhs[(x, y)] = float('inf')

        self.rhs[self.goal] = 0
        self.insert(self.goal)

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map.shape[1] and 0 <= ny < self.map.shape[0]:
                if self.map[ny][nx] == 1:
                    neighbors.append((nx, ny))
        return neighbors

    def calculate_key(self, node):
        g_rhs = min(self.g[node], self.rhs[node])
        return (g_rhs + self.manhattan(self.start, node), g_rhs)

    def insert(self, node):
        heapq.heappush(self.queue, (self.calculate_key(node), node))

    def update_vertex(self, node):
        if node != self.goal:
            self.rhs[node] = min(
                self.g.get(n, float('inf')) + 1
                for n in self.get_neighbors(node)
            )
        self.queue = [(k, n) for (k, n) in self.queue if n != node]
        heapq.heapify(self.queue)
        if self.g[node] != self.rhs[node]:
            self.insert(node)

    def compute_shortest_path(self):
        while self.queue and (
            self.queue[0][0] < self.calculate_key(self.start) or
            self.rhs[self.start] != self.g[self.start]
        ):
            _, u = heapq.heappop(self.queue)
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
            for s in self.get_neighbors(u):
                self.update_vertex(s)

    def extract_path(self):
        if self.start not in self.g:
            return []

        path = [self.start]
        current = self.start
        while current != self.goal:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                return []
            current = min(
                neighbors,
                key=lambda n: self.g.get(n, float('inf'))
            )
            if self.g.get(current, float('inf')) == float('inf'):
                return []
            path.append(current)
        return path
