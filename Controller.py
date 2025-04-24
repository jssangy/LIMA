import heapq
import numpy as np # use for matrix calculation
from sklearn import neighbors  # use for dijkstra

import Funct

class controller():
    
    def __init__(self, agv_num, map):
        self.agv_pos = {} # save the position of agv positions
        self.agv_next_pos = {} # save the next position of agv positions
        self.agv_next_rout = {} # save the next rout position of agv
        self.control_buffer = {} # save the control output of agvs
        self.agv_state = {} # 0(start - pick up) 1(pick up - drop) 2(drop - rest) 3(rest - start)
        self.agv_nums = [] # agv numbers (A, B, C, ... O)
        self.agv_mode = {} # 0 (normal) 1 (Danger)
        self.agv_goal = {} # goal position of all agvs
        self.agv_info = {} # for GUI infomation
        self.agv_rout = {} # for routing of AGV
        self.agv_pre_rout = {} # previous node
        
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
            self.agv_rout[chr(i + 65)] = []
            self.agv_pre_rout[chr(i + 65)] = (0, 0)
        
        # Map of warehouse digital twin
        self.map = map
        
        # Make graph for routing
        self.graphing()
                
        # Time
        self.time = 0
    
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
        
    # Update data from sensing of agv
    def get_sensing(self, num, data):
        if data != None:
            self.agv_pos[num] = data[0]
            self.agv_mode[num] = data[1]
            self.agv_info[num][1] = data[1]
            
        if self.time == 0:
            self.agv_rout[num] = self.dijkstra_shortest(self.graph, self.agv_pos[num], self.agv_goal[num][self.agv_state[num]])
            self.agv_next_rout[num] = self.agv_rout[num][0]

    def update_control(self, actions):
        self.time += 1
        self.dijkstra_rout(actions)
        return (self.control_buffer, self.agv_mode)
    
    def dijkstra_rout(self, actions):
        # Get the Dijkstra rout of AGVs
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]

            # Change the state of AGVs
            if (pos == goal):
                state = self.change_state(num, state)
                goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0
                self.agv_rout[num] = self.dijkstra_shortest(self.graph, pos, goal)
            
            # If AGV need next rout! (rout node)
            if (((self.map[pos[1]][pos[0]] == 6) or (pos in self.agv_goal[num])) and (self.agv_mode[num] == 0)):
                next_rout = self.agv_rout[num].pop(0)
                
                # Save next rout position
                self.agv_next_rout[num] = next_rout  
    
    def make_control(self):
        self.time += 1
        self.dijkstra_rout()
        return (self.control_buffer, self.agv_mode)
    
    def dijkstra_rout(self):
        # Get the Dijkstra rout of AGVs
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]

            # Change the state of AGVs
            if (pos == goal):
                state = self.change_state(num, state)
                goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0
                self.agv_rout[num] = self.dijkstra_shortest(self.graph, pos, goal)
            
            # If AGV need next rout! (rout node)
            if (((self.map[pos[1]][pos[0]] == 6) or (pos in self.agv_goal[num])) and (self.agv_mode[num] == 0)):
                next_rout = self.agv_rout[num].pop(0)
                
                # Save next rout position
                self.agv_next_rout[num] = next_rout
                
                # Determine new control signal
                if next_rout[0] > pos[0]:
                    self.control_buffer[num] = (1, 0)
                elif next_rout[0] < pos[0]:
                    self.control_buffer[num] = (-1, 0)
                elif next_rout[1] > pos[1]:
                    self.control_buffer[num] = (0, 1)
                elif next_rout[1] < pos[1]:
                    self.control_buffer[num] = (0, -1)
                else:
                    self.control_buffer[num] = (0, 0)
                self.agv_next_pos[num] = (pos[0] + self.control_buffer[num][0], pos[1] + self.control_buffer[num][1]) 
        
            # Just keep going!
            else:
                self.agv_next_pos[num] = (pos[0] + self.control_buffer[num][0], pos[1] + self.control_buffer[num][1]) 
                
        # Collision prevention => Dead Lock
        for num1 in self.agv_nums:
            num1_pos = self.agv_pos[num1]
            num1_next_pos = self.agv_next_pos[num1]

            # Deadlock 상태 초기화
            self.agv_mode[num1] = 0

            for num2 in self.agv_nums:
                if num1 != num2:
                    num2_pos = self.agv_pos[num2]
                    num2_next_pos = self.agv_next_pos[num2]
                    if (num1_next_pos == num2_next_pos):
                        self.agv_mode[num1] = 1
                    elif (num1_next_pos == num2_pos and num2_next_pos == num1_pos):
                        self.agv_mode[num1] = 1
          
            if self.map[num1_pos[1]][num1_pos[0]] == 1:
                self.agv_mode[num1] = 2
                self.control_buffer[num1] = (0, 0)    
        
    # ======================== Routing Functions ============================================
    def graphing(self):
        self.graph = {}
        for x in range (100):
            for y in range(100):
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
    
    def find_neighbors(self, x, y, rout = True):
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
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
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
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
        
        distance = 0
        poss_x = x
        poss_y = y
        
        # right
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_y += 1
            distance += 1
            if (self.map[poss_y][poss_x] == 6):
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
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
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
            
        return line_list

    def dijkstra_shortest(self, graph, start, end):
        distances = {node: float('inf') for node in graph}  # start로 부터의 거리 값을 저장하기 위함
        distances[start] = 0  # 시작 값은 0이어야 함
        queue = []
        heapq.heappush(queue, [distances[start], start])  # 시작 노드부터 탐색 시작 하기 위함.
        
        parents = {start: None}
        distance = {start: 0}

        while queue:  # queue에 남아 있는 노드가 없으면 끝
            current_distance, current_destination = heapq.heappop(queue)  # 탐색 할 노드, 거리를 가져옴.

            if current_destination == end:
                return self.traceback_path(end, parents)

            if distances[current_destination] < current_distance:  # 기존에 있는 거리보다 길다면, 볼 필요도 없음
                continue
            
            for new_destination, new_distance in graph[current_destination].items():
                distance = current_distance + new_distance  # 해당 노드를 거쳐 갈 때 거리
                if distance < distances[new_destination]:  # 알고 있는 거리 보다 작으면 갱신
                    distances[new_destination] = distance
                    heapq.heappush(queue, [distance, new_destination])  # 다음 인접 거리를 계산 하기 위해 큐에 삽입
                    parents[new_destination] = current_destination
            
        return -1
    
    def traceback_path(self, target, parents):
        path = []
        while target:
            path.append(target)
            target = parents[target]
        return list(reversed(path))[1:]
