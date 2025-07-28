import numpy as np

class Intersection:
    def __init__(self, intersection_data, controller_ref):
        """
        교차로 객체를 초기화합니다.
        - intersection_data: (x, y, len_N, len_E, len_S, len_W)
        - controller_ref: AGV 데이터에 접근하기 위한 컨트롤러 참조
        """
        self.x, self.y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
        self.controller = controller_ref
        
        # 각 방향별 레인 좌표를 미리 계산하여 Set으로 저장 (효율적인 조회를 위해)
        self.lane_coords = {
            'N': {(self.x, self.y - i) for i in range(1, self.len_N + 1)},
            'E': {(self.x + i, self.y) for i in range(1, self.len_E + 1)},
            'S': {(self.x, self.y + i) for i in range(1, self.len_S + 1)},
            'W': {(self.x - i, self.y) for i in range(1, self.len_W + 1)}
        }
        # self.rl_agent = ... # 여기에 각 교차로의 RL 에이전트를 초기화할 수 있습니다.

    def get_state(self):
        """
        이 교차로의 상태 벡터를 계산하여 반환합니다.
        """
        directions = ['N', 'E', 'S', 'W']
        state_vector = []

        for dir_name in directions:
            # 1. 해당 방향의 레인에 있는 AGV들을 찾음
            agvs_in_lane = {
                num: pos for num, pos in self.controller.agv_pos.items() 
                if pos in self.lane_coords[dir_name]
            }

            closest_agv_num = None
            if agvs_in_lane:
                # 2. 방향별 특성을 이용해 가장 가까운 AGV를 찾음
                if dir_name == 'N': # y가 가장 큰 AGV
                    closest_agv_num = max(agvs_in_lane, key=lambda num: agvs_in_lane[num][1])
                elif dir_name == 'E': # x가 가장 작은 AGV
                    closest_agv_num = min(agvs_in_lane, key=lambda num: agvs_in_lane[num][0])
                elif dir_name == 'S': # y가 가장 작은 AGV
                    closest_agv_num = min(agvs_in_lane, key=lambda num: agvs_in_lane[num][1])
                elif dir_name == 'W': # x가 가장 큰 AGV
                    closest_agv_num = max(agvs_in_lane, key=lambda num: agvs_in_lane[num][0])

            goal_onehot = [0, 0, 0, 0]
            deadlock = 0
            min_dist = -1

            if closest_agv_num is not None:
                pos = self.controller.agv_pos[closest_agv_num]
                min_dist = abs(pos[0] - self.x) + abs(pos[1] - self.y)

                goal = self.controller.agv_goal[closest_agv_num][self.controller.agv_state[closest_agv_num]]
                goal_dx, goal_dy = goal[0] - self.x, goal[1] - self.y
                if goal_dy < 0: goal_onehot[0] = 1
                elif goal_dx > 0: goal_onehot[1] = 1
                elif goal_dy > 0: goal_onehot[2] = 1
                elif goal_dx < 0: goal_onehot[3] = 1
            
            state_vector.extend(goal_onehot + [deadlock, min_dist])

        return np.array(state_vector, dtype=np.float32)

    def get_action(self, state):
        # action = self.rl_agent.get_action(state)
        # return action
        pass