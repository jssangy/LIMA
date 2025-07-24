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
            if self.task_count < len(self.tasks):
                self.agv_state[num] = 0
                self.agv_info[num][0] += 1
                self.whole_product += 1

                self.set_pick(num, self.tasks[self.task_count][0])
                self.set_drop(num, self.tasks[self.task_count][1])
                self.task_count += 1

            else:
                pass            
        
        return self.agv_state[num]

    # Update data from sensing of agv
    def get_sensing(self, num, data):
        if data != None:
            self.agv_pos[num] = data[0]
            self.agv_mode[num] = data[1]
            self.agv_info[num][1] = data[1]
    
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
            self.planners[num] = DStar(self.map, pos, goal)
            self.planners[num].compute_shortest_path()
            path = self.planners[num].extract_path()
            self.agv_rout[num] = path
            if len(path) >= 2:
                self.agv_next_rout[num] = path[1]
            else:
                self.agv_next_rout[num] = pos

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