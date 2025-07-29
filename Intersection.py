import numpy as np

class Intersection:
    def __init__(self, intersection_data, controller_ref):
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

    def get_state(self):
        directions = ['N', 'E', 'S', 'W']
        state_vector = []
        center = (self.center_x, self.center_y)

        agvs_on_lanes = {
            num: pos for num, pos in self.controller.agv_pos.items()
            if pos in self.all_lane_coords
        }

        for dir_name in directions:
            agvs_in_dir = {
                num: pos for num, pos in agvs_on_lanes.items() 
                if pos in self.lane_coords[dir_name]
            }

            closest_agv_num = None
            if agvs_in_dir:
                if dir_name == 'N':
                    closest_agv_num = max(agvs_in_dir, key=lambda num: agvs_in_dir[num][1])
                elif dir_name == 'E':
                    closest_agv_num = min(agvs_in_dir, key=lambda num: agvs_in_dir[num][0])
                elif dir_name == 'S':
                    closest_agv_num = min(agvs_in_dir, key=lambda num: agvs_in_dir[num][1])
                elif dir_name == 'W':
                    closest_agv_num = max(agvs_in_dir, key=lambda num: agvs_in_dir[num][0])

            exit_onehot = [0, 0, 0, 0]
            deadlock = 0
            min_dist = -1
            path = self.controller.agv_path.get(closest_agv_num, [])

            if closest_agv_num is not None:
                pos = self.controller.agv_pos[closest_agv_num]
                min_dist = abs(pos[0] - center[0]) + abs(pos[1] - center[1])

                if center in path:
                    center_index = path.index(center)
                    exit_node = path[center_index + 1]
                    dx, dy = exit_node[0] - center[0], exit_node[1] - center[1]
                    if dx == 0 and dy > 0:
                        exit_onehot[0] = 1
                    elif dx > 0 and dy == 0:
                        exit_onehot[1] = 1
                    elif dx == 0 and dy < 0:
                        exit_onehot[2] = 1
                    elif dx < 0 and dy == 0:
                        exit_onehot[3] = 1
                    else:
                        raise ValueError("Invalid exit direction from center to next node.")
                    
                else:
                    dx, dy = path[0][0] - center[0], path[0][1] - center[1]
                    if dx == 0 and dy > 0:
                        exit_onehot[0] = 1
                    elif dx > 0 and dy == 0:
                        exit_onehot[1] = 1
                    elif dx == 0 and dy < 0:
                        exit_onehot[2] = 1
                    elif dx < 0 and dy == 0:
                        exit_onehot[3] = 1
                    else:
                        raise ValueError("Invalid exit direction from first path node to center.")

            state_vector.extend(exit_onehot + [min_dist, deadlock])

        center_occupied = 0
        center_agv_direction = [0, 0, 0, 0]

        for num, pos in agvs_on_lanes.items():
            if pos == center:
                center_occupied = 1
                path = self.controller.agv_path.get(num, [])
                dx, dy = path[1][0] - center[0], path[1][1] - center[1]
                if dx == 0 and dy > 0:
                    center_agv_direction[0] = 1
                elif dx > 0 and dy == 0:
                    center_agv_direction[1] = 1
                elif dx == 0 and dy < 0:        
                    center_agv_direction[2] = 1
                elif dx < 0 and dy == 0:
                    center_agv_direction[3] = 1

        state_vector.append(center_occupied)
        state_vector.extend(center_agv_direction)

        return np.array(state_vector, dtype=np.float32)
    
    def step(self):
        pass