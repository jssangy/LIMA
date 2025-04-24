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
    
    def step2(self, actions, reward, events):
        self.time += 1
        total_reward = 0
        next_events = {}

        # Controller에게 Action 미리 전달
        self.controller.set_actions(actions)

        # <1단계> 모든 AGV sensing 처리
        for num, agv in self.agv_list.items():
            sensing = self.network.send(agv.sensing())
            self.controller.get_sensing(num, sensing)

        # <2단계> 한 번만 control 신호 생성 및 전송
        for num, agv in self.agv_list.items():
            if any(events[num]):
                control_sig = self.controller.update_control(num, events[num])
            else:
                control_sig = self.controller.get_control(num)
            agv.get_control(self.network.send([control_sig[0], control_sig[1]]))

        # <3단계> 모든 AGV의 이동 처리 및 보상 계산
        for num, agv in self.agv_list.items():
            if (self.interact(agv.next_pos()) == 0):
                agv.move()

            # Collision with wall
            if (self.interact(agv.next_pos()) == 1):
                pass
                
            # Collision with other AGVs
            if (self.interact(agv.next_pos()) == 2):
                pass

            pos = self.controller.agv_pos[num]
            node_arrive = int(self.map[pos[1]][pos[0]] == 6 or pos in self.controller.agv_goal[num])
            deadlock = int(self.controller.agv_mode[num] == 1)
            wait = int(self.controller.control_buffer[num] == (0, 0))

            next_events[num] = [node_arrive, deadlock, wait]

            # Goal 도착 시 보상 및 Delay Penalty 처리
            reward = 0
            if pos in self.controller.agv_goal[num]:
                reward += reward['goal']
                actual_time = self.controller.actual_time[num]
                optimal_time = self.controller.optimal_time[num]
                delay_time = actual_time - optimal_time
                if delay_time > 0:
                    reward += delay_time * reward['delayed']

            # Deadlock Penalty 처리
            if deadlock:
                reward += reward['deadlock']

            total_reward += reward

        next_states = self.get_state()
        return next_states, total_reward, next_events
    
    def step(self, actions):
        self.time += 1

        # <1 Step>
        # All AGVs send the sensor signal
        for num, agv in self.agv_list.items():
            # Send the signal to controller through network 
            self.controller.get_sensing(num, self.network.send(agv.sensing()))
        
        # <2 Step>
        # Controller sends the conntrol signal through network
        control_sig = self.controller.update_control(actions)
        for num, agv in self.agv_list.items():
            agv.get_control(self.network.send([control_sig[0][num], control_sig[1][num]]))
   

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
            if(self.interact(agv.next_pos()) == 0):
                agv.move()
                
            # Collision with wall
            if(self.interact(agv.next_pos()) == 1):
                pass
                
            # Collision with other AGVs
            if(self.interact(agv.next_pos()) == 2):
                pass

        return self.make_info()
    
    def interact(self, pos):
        if self.map[pos[1]][pos[0]] == 1:
            return 1

        for agv in self.agv_list.values():
            if (pos == agv.pos):
                self.controller.agv_mode[agv] = 1
                return 2
        
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
    
    # Get AGV state for Reinforcement Learning
    def get_state(self):
        return self.controller.get_state()
    
    
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