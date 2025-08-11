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

        self.agv_on_lanes = {}
        self.agv_in_dir = {'N': {}, 'E': {}, 'S': {}, 'W': {}}
        self.center_agv = None
        self.agvs_inside_last_step = {}
        self.exit_events = []

    @property
    def is_empty(self):
        """교차로 내(all_lane_coords)에 AGV가 하나도 없으면 True를 반환합니다."""
        # _update_internal_state가 매 스텝 호출되므로 agv_on_lanes는 항상 최신 상태임
        return not self.agv_on_lanes

    def get_state(self):
        self._update_internal_state()

        directions = ['N', 'E', 'S', 'W']
        state_vector = []
        center = (self.center_x, self.center_y)

        for dir_name in directions:
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
                pos = self.agv_in_dir[dir_name][closest_agv_num]
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
        """
        [재설계] F_attr = -1 규칙과 F_rep의 부호 변화 규칙을 적용합니다.
        - 진입 시: F_rep은 F_attr과 반대 방향 (양수 값)
        - 탈출 시: F_rep은 F_attr과 같은 방향 (음수 값)
        """
        self._update_internal_state()
        
        print(actions)

        repulsive_magnitudes = actions[:4]
        center_direction_action = int(actions[4])
        center = (self.center_x, self.center_y)

        # --- 1. 차선 위 AGV 제어 (진입/통과) ---
        action_map = {'N': repulsive_magnitudes[0], 'E': repulsive_magnitudes[1], 
                      'S': repulsive_magnitudes[2], 'W': repulsive_magnitudes[3]}

        for dir_name, agvs in self.agv_in_dir.items():
            for agv_num, agv_pos in agvs.items():
                path = self.controller.agv_path.get(agv_num, [])
                if not path:
                    raise ValueError(f"No path found for AGV {agv_num}.")

                # 1. AGV 상태 결정 (진입 중 or 탈출 중)
                is_entering = center in path

                # 2. 인력장(F_attr)의 방향 벡터 계산
                if is_entering:
                    target_node = center
                else:
                    target_node = path[1]
                
                dx = target_node[0] - agv_pos[0]
                dy = target_node[1] - agv_pos[1]
                F_attr_direction = (np.sign(dx), np.sign(dy))

                # 3. 척력장(F_rep)의 크기 및 부호 결정
                repulsive_magnitude = action_map[dir_name]
                
                # 진입 중이면 F_rep은 양수, 탈출 중이면 F_rep은 음수
                if is_entering:
                    F_rep_scalar = repulsive_magnitude 
                else:
                    F_rep_scalar = -repulsive_magnitude

                # 4. 최종 힘(F_total) 계산
                # F_total = F_attr + F_rep
                # F_attr은 크기가 -1이므로, F_total = -1 + F_rep_scalar
                F_total = -1 + F_rep_scalar

                # 5. 최종 움직임 결정
                move = (0, 0)
                # F_total이 음수일 때만 D* 경로(F_attr_direction) 방향으로 움직임
                if F_total < 0:
                    continue
                elif F_total == 0:
                    move = (0, 0)
                else:
                    move = (-F_attr_direction[0], -F_attr_direction[1])

                self.controller.control_buffer[agv_num] = move

        # --- 2. 중앙 AGV 제어 ---
        if self.center_agv is not None:
            move_map = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)} # N, E, S, W
            move = move_map[center_direction_action]
            self.controller.control_buffer[self.center_agv] = move


    def _update_internal_state(self):
        """
        [수정] 교차로 전체를 기준으로 AGV 탈출 이벤트를 감지하고, 탈출 방향의 정합성을 평가합니다.
        """
        # 0. 이전 스텝의 탈출 이벤트 정보 초기화
        self.exit_events.clear()

        # 1. 현재 교차로 내(all_lane_coords)에 있는 AGV 목록을 dict로 가져옴
        current_agvs_inside = {
            num: pos for num, pos in self.controller.agv_pos.items()
            if pos in self.all_lane_coords
        }

        self.agv_on_lanes = current_agvs_inside

        # 2. 이전 스텝과 비교하여 '방금 탈출한' AGV를 식별
        exited_agv_nums = set(self.agvs_inside_last_step.keys()) - set(current_agvs_inside.keys())

        for num in exited_agv_nums:
            # 3. 탈출 방향 평가
            last_pos = self.agvs_inside_last_step[num]
            current_pos = self.controller.agv_pos[num]
            path = self.controller.agv_path.get(num, [])

            # 실제 이동 방향 벡터
            actual_move = (np.sign(current_pos[0] - last_pos[0]), np.sign(current_pos[1] - last_pos[1]))

            # D* 경로가 의도한 이동 방향 벡터
            intended_move = (0, 0)
            if path:
                # 경로상에서 마지막 위치(last_pos)의 다음 노드를 찾아 목표 방향 설정
                try:
                    last_pos_index = path.index(last_pos)
                    if last_pos_index + 1 < len(path):
                        target_node = path[last_pos_index + 1]
                        intended_move = (np.sign(target_node[0] - last_pos[0]), np.sign(target_node[1] - last_pos[1]))
                except ValueError:
                    # 경로에 last_pos가 없는 예외적인 경우, 경로의 마지막 노드를 목표로 설정
                    target_node = path[-1]
                    intended_move = (np.sign(target_node[0] - last_pos[0]), np.sign(target_node[1] - last_pos[1]))

            # 4. 평가 결과를 이벤트 리스트에 저장
            is_correct_exit = (actual_move == intended_move and intended_move != (0,0))
            self.exit_events.append({"agv_num": num, "correct": is_correct_exit})

        # 5. 다음 스텝을 위해 현재 상태를 '이전 상태'로 저장
        self.agvs_inside_last_step = current_agvs_inside

        # 6. 방향별 AGV 및 중앙 AGV 상태 업데이트 (기존 로직)
        self.agv_in_dir = {
            dir_name: {
                num: pos for num, pos in current_agvs_inside.items()
                if pos in self.lane_coords[dir_name]
            } for dir_name in ['N', 'E', 'S', 'W']
        }
        
        self.center_agv = None
        center_pos = (self.center_x, self.center_y)
        for num, pos in current_agvs_inside.items():
            if pos == center_pos:
                self.center_agv = num
                break