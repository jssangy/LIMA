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

    def get_state(self):
        directions = ['N', 'E', 'S', 'W']
        state_vector = []
        center = (self.center_x, self.center_y)

        self.agv_on_lanes = {
            num: pos for num, pos in self.controller.agv_pos.items()
            if pos in self.all_lane_coords
        }

        self.agv_in_dir = {
            'N': {}, 'E': {}, 'S': {}, 'W': {}
        }

        for dir_name in directions:
            self.agv_in_dir[dir_name] = {
                num: pos for num, pos in self.agv_on_lanes.items()
                if pos in self.lane_coords[dir_name]
            }

            closest_agv_num = None
            if self.agv_in_dir[dir_name]:
                if dir_name == 'N':
                    closest_agv_num = max(self.agv_in_dir[dir_name], key=lambda num: self.agv_in_dir[dir_name][num][1])
                elif dir_name == 'E':
                    closest_agv_num = min(self.agv_in_dir[dir_name], key=lambda num: self.agv_in_dir[dir_name][num][0])
                elif dir_name == 'S':
                    closest_agv_num = min(self.agv_in_dir[dir_name], key=lambda num: self.agv_in_dir[dir_name][num][1])
                elif dir_name == 'W':
                    closest_agv_num = max(self.agv_in_dir[dir_name], key=lambda num: self.agv_in_dir[dir_name][num][0])

            exit_onehot = [0, 0, 0, 0]
            deadlock = 0
            min_dist = 0
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

        center_agv_direction = [0, 0, 0, 0]
        self.center_agv = None

        for num, pos in self.agv_on_lanes.items():
            if pos == center:
                self.center_agv = num
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
                
                break

        state_vector.extend(center_agv_direction)

        return np.array(state_vector, dtype=np.float32)
    
    def action_control(self, actions):
        repulsive_forces = actions[:4]
        center_direction_action = actions[4]

        center = (self.center_x, self.center_y)
        attractive_force = -1
        force_map = {'N': repulsive_forces[0], 'E': repulsive_forces[1], 
                     'S': repulsive_forces[2], 'W': repulsive_forces[3]}
        
        agv_on_lanes = {
            num: pos for num, pos in self.controller.agv_pos.items()
            if pos in self.all_lane_coords
        }

        agv_in_dir = {'N': {}, 'E': {}, 'S': {}, 'W': {}}
        for dir_name in ['N', 'E', 'S', 'W']:
            agv_in_dir[dir_name] = {
                num: pos for num, pos in agv_on_lanes.items()
                if pos in self.lane_coords[dir_name]
            }

        for dir_name, agvs in agv_in_dir.items():
            repulsive_force = force_map[dir_name]
            total_force = repulsive_force + attractive_force

            for agv_num, agv_pos in agvs.items():
                if total_force == 0: 
                    self.controller.control_buffer[agv_num] = (0, 0)
                elif total_force > 0: 
                    dx = agv_pos[0] - center[0]
                    dy = agv_pos[1] - center[1]
                    move_dx = 1 if dx > 0 else -1 if dx < 0 else 0
                    move_dy = 1 if dy > 0 else -1 if dy < 0 else 0
                    self.controller.control_buffer[agv_num] = (move_dx, move_dy)

        if self.center_agv is not None:
            agv_num = self.center_agv

            if center_direction_action == 0: # North
                move = (0, 1)
            elif center_direction_action == 1: # East
                move = (1, 0)
            elif center_direction_action == 2: # South
                move = (0, -1)
            elif center_direction_action == 3: # West
                move = (-1, 0)
            
            self.controller.control_buffer[agv_num] = move