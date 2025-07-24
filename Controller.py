import heapq
import numpy as np # use for matrix calculation

import Funct
from global_planning import DStar

class controller():    
    def __init__(self, agv_num, map, tasks):
        self.agv_pos = {} # save the position of agv positions
        self.agv_next_pos = {} # save the next position of agv positions
        self.agv_next_rout = {} # save the next rout position of agv
        self.control_buffer = {} # save the D* algorithm based control output of agvs
        self.action_control_buffer = {} # save the action control output of agvs
        self.agv_state = {} # 0(start - pick up) 1(pick up - drop) 2(drop - rest) 3(rest - start)
        self.agv_nums = [] # agv numbers (0, 1, 2, ...)
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
            self.agv_nums.append(i)
            self.agv_pos[i] = (0, 0)
            self.agv_next_rout[i] = (0, 0)
            self.agv_state[i] = 0 # Initial state is 0
            self.agv_mode[i] = 0 # Initial mode is normal
            self.agv_goal[i] = [(0, 0), (0, 0), (0, 0), (0, 0)]
            self.agv_info[i] = [0, 0]
            self.control_buffer[i] = (0, 0)
            self.action_control_buffer[i] = (0, 0)
            self.agv_rout[i] = []
            self.agv_pre_rout[i] = (0, 0)
        
        # Map of warehouse digital twin
        self.map = map

        # AGV tasks
        self.tasks = tasks
        self.task_count = 0

        # Set pick-up & drop tasks
        for num, task in zip(range(agv_num), self.tasks):
            self.set_pick(num, task[0])
            self.set_drop(num, task[1])
            self.task_count += 1
    
    # set start position - not used
    def set_start(self, num, pos):
        self.agv_goal[num][3] = pos
    
    # set pick-up position
    def set_pick(self, num, pos):
        self.agv_goal[num][0] = pos
        
    # set drop position
    def set_drop(self, num, pos):
        self.agv_goal[num][1] = pos
        
    # set rest position - not used
    def set_rest(self, num, pos):
        self.agv_goal[num][2] = pos
    
    # Change the AGV's state
    def change_state(self, num, state):
        if state != 1:
            self.agv_state[num] = state + 1   
        else:
            if self.task_count <= len(self.tasks):
                self.agv_state[num] = 0
                self.agv_info[num][0] += 1
                self.whole_product += 1

                self.set_pick(num, self.tasks[self.task_count][0])
                self.set_drop(num, self.tasks[self.task_count][1])
                self.task_count += 1

            else:
                pass
            
        
        return self.agv_state[num]
    
    def set_control(self, num):
        pos = self.agv_pos[num]
        goal = self.agv_goal[num][self.agv_state[num]]
        self.planners[num] = DStar(self.map, pos, goal)
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
                self.planners[num] = DStar(self.map, pos, new_goal)
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
                if actions[idx] == "D*":
                    self.action_control_buffer[num] = (dx, dy)
                else:
                    self.action_control_buffer[num] = control[actions[idx]]
                self.agv_next_pos[num] = pos + self.action_control_buffer[num]
    
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
                self.planners[num] = DStar(self.map, pos, goal)
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
    
    def dynamic_obstacle_dstar_rout(self):
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]

            dynamic_grid = self.map.copy()
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
        
    # ======================== Routing Functions ============================================
    def get_active_tasks(self):
        active_tasks = {}
        for num in self.agv_nums:
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]
            active_tasks[num] = goal

        return active_tasks
    
# D* Lite Algorithm
class DStar:
    def __init__(self, map, start, goal):
        self.map = map
        self.start = start
        self.goal = goal

        self.g = {}          # Actual cost
        self.rhs = {}        # Estimated cost
        self.queue = []      # Priority queue

        h, w = map.shape
        for y in range(h):
            for x in range(w):
                if map[y][x] == 1:
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
                if self.map[ny][nx] == 0:
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
