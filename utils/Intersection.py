import numpy as np

class Intersection:
    def __init__(self, intersection_data, controller_ref):
        self.id = intersection_data
        self.center_x, self.center_y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
        self.controller = controller_ref

        self.lane_coords = {
            'N': {(self.center_x, self.center_y - i) for i in range(1, self.len_N + 1)},
            'E': {(self.center_x + i, self.center_y) for i in range(1, self.len_E + 1)},
            'S': {(self.center_x, self.center_y + i) for i in range(1, self.len_S + 1)},
            'W': {(self.center_x - i, self.center_y) for i in range(1, self.len_W + 1)}
        }

        self.all_lane_coords = self.lane_coords['N'].union(
            self.lane_coords['E'], 
            self.lane_coords['S'], 
            self.lane_coords['W'],
            {(self.center_x, self.center_y)}
        )

        # 이벤트 기반 AGV 추적
        self.agvs_in_intersection = {}  # 교차로 내 AGV만 추적
        self.agvs_in_lanes = {'N': {}, 'E': {}, 'S': {}, 'W': {}}
        self.center_agv = None

    def get_state(self):        
        state_vector = []
        center = (self.center_x, self.center_y)

        closest_agv_config = {
            'N': {'func': max, 'key_idx': 1}, # y좌표가 가장 큰 (가장 아래쪽)
            'E': {'func': min, 'key_idx': 0}, # x좌표가 가장 작은 (가장 왼쪽)
            'S': {'func': min, 'key_idx': 1}, # y좌표가 가장 작은 (가장 위쪽)
            'W': {'func': max, 'key_idx': 0}, # x좌표가 가장 큰 (가장 오른쪽)
        }

        for dir_name in ['N', 'E', 'S', 'W']:
            # 이미 분류된 방향별 AGV 사용
            agvs_in_direction = self.agvs_in_lanes[dir_name]
            
            closest_agv_num = None
            if agvs_in_direction:
                config = closest_agv_config[dir_name]
                key_func = lambda num: agvs_in_direction[num][config['key_idx']]
                closest_agv_num = config['func'](agvs_in_direction, key=key_func)

            # 상태 벡터 구성
            goal_onehot = [0, 0, 0, 0]
            distance = 0
            
            if closest_agv_num is not None:
                path = self.controller.agv_path[closest_agv_num]
                exit_dir = self._get_exit_direction(path)
                dir_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
                if exit_dir in dir_map:
                    goal_onehot[dir_map[exit_dir]] = 1

                pos = agvs_in_direction[closest_agv_num]
                distance = abs(pos[0] - center[0]) + abs(pos[1] - center[1])

            state_vector.extend(goal_onehot)
            state_vector.append(distance)

        center_goal_onehot = [0, 0, 0, 0]
        if self.center_agv is not None:
            path = self.controller.agv_path[self.center_agv]
            exit_dir = self._get_exit_direction(path)
            dir_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
            if exit_dir in dir_map:
                center_goal_onehot[dir_map[exit_dir]] = 1

        state_vector.extend(center_goal_onehot)

        return np.array(state_vector, dtype=np.float32)

    def action_control(self, actions):
        repulsive_magnitudes = actions[:4]
        center_direction_action = int(actions[4])
        center = (self.center_x, self.center_y)

        # 차선 위 AGV 제어
        action_map = {'N': repulsive_magnitudes[0], 'E': repulsive_magnitudes[1], 
                      'S': repulsive_magnitudes[2], 'W': repulsive_magnitudes[3]}

        for dir_name, agvs in self.agvs_in_lanes.items():
            for agv_num, agv_pos in agvs.items():
                path = self.controller.agv_path[agv_num]

                is_entering = center in path
                if is_entering:
                    target_node = center
                else:
                    target_node = path[0]
                
                dx = target_node[0] - agv_pos[0]
                dy = target_node[1] - agv_pos[1]
                F_attr_direction = (np.sign(dx), np.sign(dy))

                repulsive_magnitude = action_map[dir_name]
                
                if is_entering:
                    F_rep_scalar = repulsive_magnitude 
                else:
                    F_rep_scalar = -repulsive_magnitude

                F_total = -1 + F_rep_scalar

                move = (0, 0)
                if F_total < 0:
                    move = F_attr_direction
                elif F_total == 0:
                    move = (0, 0)
                else:
                    move = (-F_attr_direction[0], -F_attr_direction[1])
                    
                self.controller.control_buffer[agv_num] = move

        # 중앙 AGV 제어
        if self.center_agv is not None:
            move_map = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
            move = move_map[center_direction_action]
                
            self.controller.control_buffer[self.center_agv] = move

    def _get_exit_direction(self, path):
        center_node = (self.center_x, self.center_y)
        if center_node in path:
            center_index = path.index(center_node)
            exit_node = path[center_index + 1]
            return self._coords_to_direction(exit_node)
        else:
            return self._coords_to_direction(path[0])
        
    def _coords_to_direction(self, coords):
        for direction, lane_coords in self.lane_coords.items():
            if coords in lane_coords:
                return direction

    def reset(self):
        self.agvs_in_intersection.clear()
        self.agvs_in_lanes = {'N': {}, 'E': {}, 'S': {}, 'W': {}}
        self.center_agv = None

    def add_agv(self, agv_num, pos):
        self.agvs_in_intersection[agv_num] = pos

        if pos == (self.center_x, self.center_y):
            self.center_agv = agv_num
        
        for direction, coords in self.lane_coords.items():
            if pos in coords:
                self.agvs_in_lanes[direction][agv_num] = pos
                break