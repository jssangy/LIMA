import numpy as np

from global_planning import DStar, PIBT
from Intersection import Intersection

class controller():    
    def __init__(self, agv_num, map, tasks, intersections):
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
        self.use_rl = False # Use RL agent for intersection control
        
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
            self.agv_rout[i] = []
            self.agv_pre_rout[i] = (0, 0)
        
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

            # If RL agent is not used
            if self.use_rl == False: 
                next_pos = path[1]
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]
                self.control_buffer[num] = (dx, dy)
                self.agv_next_pos[num] = next_pos
            # If RL agent is used
            else:                   
                self.control_buffer[num] = (0, 0)
                self.agv_next_pos[num] = pos

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
    
    # Get the intersection state
    def get_intersection_state(self, intersection):
        """
        교차로 상태를 24차원 벡터로 반환.
        intersection: (x, y, len_N, len_E, len_S, len_W)
        """
        x, y = intersection[0], intersection[1]
        len_N, len_E, len_S, len_W = intersection[2], intersection[3], intersection[4], intersection[5]
        
        directions = ['N', 'E', 'S', 'W']
        dir_lens = {'N': len_N, 'E': len_E, 'S': len_S, 'W': len_W}
        state_vector = []

        for dir_name in directions:
            # 1. 해당 방향의 레인에서 가장 가까운 AGV 찾기
            min_dist = float('inf')
            closest_agv_num = None
            lane_len = dir_lens[dir_name]

            for num, agv_pos in self.agv_pos.items():
                ax, ay = agv_pos
                is_in_lane = False
                
                # AGV가 해당 방향의 레인에 있는지 확인
                if dir_name == 'N' and ax == x and (y - lane_len <= ay < y):
                    is_in_lane = True
                elif dir_name == 'E' and ay == y and (x < ax <= x + lane_len):
                    is_in_lane = True
                elif dir_name == 'S' and ax == x and (y < ay <= y + lane_len):
                    is_in_lane = True
                elif dir_name == 'W' and ay == y and (x - lane_len <= ax < x):
                    is_in_lane = True

                if is_in_lane:
                    dist = abs(ax - x) + abs(ay - y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_agv_num = num

            # 2. 목표 방향 원-핫 벡터
            goal_onehot = [0, 0, 0, 0]
            if closest_agv_num is not None:
                goal = self.agv_goal[closest_agv_num][self.agv_state[closest_agv_num]]
                goal_dx = goal[0] - x
                goal_dy = goal[1] - y
                if goal_dy < 0:   # 북
                    goal_onehot[0] = 1
                elif goal_dx > 0: # 동
                    goal_onehot[1] = 1
                elif goal_dy > 0: # 남
                    goal_onehot[2] = 1
                elif goal_dx < 0: # 서
                    goal_onehot[3] = 1

            # 3. 데드락 발생 여부 (예시: mode==2)
            deadlock = 0
            if closest_agv_num is not None and self.agv_mode[closest_agv_num] == 2:
                deadlock = 1

            # 4. 거리
            distance = min_dist if closest_agv_num is not None else -1

            # 5. 방향별 상태 벡터 합치기
            state_vector.extend(goal_onehot + [deadlock, distance])

        return np.array(state_vector, dtype=np.float32)