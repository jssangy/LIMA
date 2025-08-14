import numpy as np

from utils.global_planning import DStar

class controller():    
    def __init__(self, map_data):
        # 동적 AGV 관리를 위해 초기화 시 비워둠
        self.agv_pos = {}
        self.control_buffer = {}
        self.agv_nums = []
        self.agv_goal = {}
        self.planners = {}
        self.agv_path = {}

        # Map of the environment
        self.map = map_data

    def reset(self):
        self.agv_pos.clear()
        self.control_buffer.clear()
        self.agv_nums.clear()
        self.agv_goal.clear()
        self.planners.clear()
        self.agv_path.clear()
    
    def add_agv(self, agv_num, start_pos, goal_pos):
        """AGV를 컨트롤러에 동적으로 추가"""
        self.agv_nums.append(agv_num)
        self.agv_pos[agv_num] = start_pos
        self.agv_goal[agv_num] = goal_pos
        self.control_buffer[agv_num] = (0, 0)
        self.planners[agv_num] = DStar(self.map, start_pos, goal_pos)
        self.agv_path[agv_num] = []

    def remove_agv(self, agv_num):
        """완료된 AGV를 컨트롤러에서 제거"""
        if agv_num in self.agv_nums:
            self.agv_nums.remove(agv_num)
            self.agv_pos.pop(agv_num, None)
            self.agv_goal.pop(agv_num, None)
            self.control_buffer.pop(agv_num, None)
            self.planners.pop(agv_num, None)
            self.agv_path.pop(agv_num, None)

    def get_sensing(self, agv_num, pos):
        """GymEnv로부터 AGV의 현재 위치를 업데이트"""
        if agv_num in self.agv_nums:
            self.agv_pos[agv_num] = pos

    def make_control(self):
        """모든 활성 AGV에 대한 제어 신호 생성"""
        self.dstar_rout()
    
    def dstar_rout(self):
        """D* 알고리즘을 사용하여 각 AGV의 경로를 계산하고 제어 신호 생성"""
        for num in self.agv_nums:
            pos = self.agv_pos.get(num)
            goal = self.agv_goal.get(num)
            planner = self.planners.get(num)

            if not all([pos, goal, planner]):
                continue

            # 목표에 도달하면 이동 멈춤 (제거는 gym_env에서 처리)
            if pos == goal:
                self.control_buffer[num] = (0, 0)
                continue

            # D* 플래너의 시작점을 현재 위치로 업데이트하고 경로 재계산
            planner.start = pos
            planner.compute_shortest_path()
            path = planner.extract_path()
            self.agv_path[num] = path

            # 경로에 따라 다음 이동 방향 결정
            next_pos = path[1]
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            self.control_buffer[num] = (dx, dy)