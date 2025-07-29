import numpy as np

from global_planning import DStar, PIBT
from Intersection import Intersection

class controller():    
    def __init__(self, agv_num, map, agv_list, tasks, intersections):
        self.agv_pos = {}               # save the position of agv positions
        self.control_buffer = {}        # save the D* algorithm based control output of agvs
        self.agv_state = {}             # 0(start - pick up) 1(pick up - drop) 2(drop - rest) 3(rest - start)
        self.agv_nums = []              # agv numbers (0, 1, 2, ...)
        self.agv_mode = {}              # 0 (normal) 1 (Danger)
        self.agv_goal = {}              # goal position of all agvs
        self.agv_info = {}              # for GUI information
        self.planners = {}              # D* class for each AGV
        self.agv_path = {}              # save the path of each AGV

        self.running_opt = 0
        self.use_rl = False # Use RL agent for intersection control
        
        # Whole Products
        self.whole_product = 0
            
        # Initialization
        for i in range (agv_num):
            self.agv_nums.append(i)
            self.agv_pos[i] = (0, 0)
            self.agv_state[i] = 0 # Initial state is 0
            self.agv_mode[i] = 0 # Initial mode is normal
            self.agv_goal[i] = [(0, 0), (0, 0), (0, 0), (0, 0)]
            self.agv_info[i] = [0, 0]
            self.control_buffer[i] = (0, 0)
            self.planners[i] = DStar(map, agv_list[i].start, tasks[i][0])

        # Map of the environment
        self.map = map

        # AGV tasks
        self.tasks = tasks
        self.task_count = 0

        # Intersections of map
        self.intersections = [Intersection(data, self) for data in intersections]

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
                return      
        
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
            self.pibt_rout()
        return (self.control_buffer, self.agv_mode)
    
    def dstar_rout(self):
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]
            planner = self.planners[num]

            if pos == goal:
                state = self.change_state(num, state)
                goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0                
                planner.update_goal(goal)

            planner.start = pos
            planner.compute_shortest_path()
            path = planner.extract_path()
            self.agv_path[num] = path

            # If RL agent is not used
            if self.use_rl == False: 
                next_pos = path[1]
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]
                self.control_buffer[num] = (dx, dy)
            # If RL agent is used
            else:                   
                

    def pibt_rout(self):
        agv_nums = self.agv_nums
        starts = [self.agv_pos[num] for num in agv_nums]
        goals = [self.agv_goal[num][self.agv_state[num]] for num in agv_nums]
        pibt_planner = PIBT(self.map, starts, goals)
        priorities = [pibt_planner.dist_tables[i].get(starts[i]) for i in range(len(starts))]
        next_positions = pibt_planner.step(starts, priorities)
        for idx, num in enumerate(agv_nums):
            cur_pos = self.agv_pos[num]
            next_pos = next_positions[idx]
            dx = next_pos[0] - cur_pos[0]
            dy = next_pos[1] - cur_pos[1]
            self.control_buffer[num] = (dx, dy)
            self.agv_next_pos[num] = next_pos
        
    # ======================== Routing Functions ============================================
    # Get active tasks for AGVs
    def get_active_tasks(self):
        active_tasks = {}
        for num in self.agv_nums:
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]
            active_tasks[num] = goal

        return active_tasks