import numpy as np

class Intersection:
    def __init__(self, intersection_data, controller_ref):
        self.id = intersection_data
        self.center_x, self.center_y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
        self.controller = controller_ref
        self.map = self.controller.map

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

        # 이벤트 기반 AGV object 추적
        self.agvs_in_intersection = set()  # 교차로 내 AGV만 추적
        self.agvs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_agv = None

    def add_agv(self, agv_obj):
        self.agvs_in_intersection.add(agv_obj)

        if agv_obj.pos == (self.center_x, self.center_y):
            self.center_agv = agv_obj
        
        for direction, coords in self.lane_coords.items():
            if agv_obj.pos in coords:
                self.agvs_in_lanes[direction].append(agv_obj)

    def get_state(self):
        state_vector = []
        center = (self.center_x, self.center_y)

        closest_cfg = {
            'N': ('y',  max),  # y가 가장 큰(아래쪽)
            'E': ('x',  min),  # x가 가장 작은(왼쪽)
            'S': ('y',  min),  # y가 가장 작은(위쪽)
            'W': ('x',  max),  # x가 가장 큰(오른쪽)
        }

        for d in ['N', 'E', 'S', 'W']:
            agvs = self.agvs_in_lanes[d]  # [AGV,...]
            goal_onehot = [0, 0, 0, 0]
            distance = 0

            if agvs:
                axis, sel = closest_cfg[d]
                key_fn = (lambda a: a.pos[1]) if axis == 'y' else (lambda a: a.pos[0])
                closest_agv = sel(agvs, key=key_fn)

                agv_id = closest_agv.id
                if agv_id is not None and agv_id in self.controller.agv_path:
                    path = self.controller.agv_path[agv_id]
                    exit_dir = self._get_exit_direction(path)
                    idx = {'N':0,'E':1,'S':2,'W':3}.get(exit_dir, None)
                    if idx is not None: goal_onehot[idx] = 1

                distance = abs(closest_agv.pos[0] - center[0]) + abs(closest_agv.pos[1] - center[1])

            state_vector.extend(goal_onehot)
            state_vector.append(distance)

        center_goal_onehot = [0, 0, 0, 0]
        if self.center_agv is not None:
            agv_id = self.center_agv.id
            if agv_id is not None and agv_id in self.controller.agv_path:
                path = self.controller.agv_path[agv_id]
                exit_dir = self._get_exit_direction(path)
                idx = {'N':0,'E':1,'S':2,'W':3}.get(exit_dir, None)
                if idx is not None: center_goal_onehot[idx] = 1

        state_vector.extend(center_goal_onehot)
        return np.array(state_vector, dtype=np.float32)

    def action_control(self, actions, is_push_out=False):
        """actions: 0=N,1=E,2=S,3=W"""
        if self.center_agv is None:
            return
        
        move_map = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
        self.controller.control_buffer[self.center_agv.id] = move_map[int(actions)]
        print("Action:", actions)

        if is_push_out:
            print("Push-out activated")
            self.push_out(self.center_agv.pos, move_map[int(actions)])

    def push_out(self, pos, direction):
        # direction: (dx, dy)
        dx, dy = direction
        next_pos = (pos[0] + dx, pos[1] + dy)

        H, W = self.map.shape[0], self.map.shape[1]
        def in_bounds(p): return 0 <= p[0] < W and 0 <= p[1] < H

        # 1) 다음 칸이 맵 밖이면 실패(더 밀 곳 없음)
        if not in_bounds(next_pos):
            print(f"push_out: next {next_pos} OOB -> stop")
            return False

        # 2) 다음 칸에 AGV가 없으면(빈칸) 더 밀 필요 없음(베이스 케이스)
        target_id = None
        for agv_id, agv_p in self.controller.agv_pos.items():
            if agv_p == next_pos:
                target_id = agv_id
                break
        if target_id is None:
            # nothing ahead — success
            return True

        # 3) target이 이동해야 할 칸(그 다음 칸)
        beyond = (next_pos[0] + dx, next_pos[1] + dy)

        # 3-1) 벽/경계 체크
        if not in_bounds(beyond) or self.map[beyond[1]][beyond[0]] == 1:
            print(f"push_out: {next_pos} -> {beyond} blocked (wall/OOB)")
            return False

        # 3-2) 그 다음 칸이 다른 AGV로 점유되어 있으면, 먼저 재귀로 비워라(테일-퍼스트)
        occupied = False
        for oid, opos in self.controller.agv_pos.items():
            if opos == beyond:
                occupied = True
                break
        if occupied:
            ok = self.push_out(next_pos, direction)
            if not ok:
                # 꼬리를 못 밀면 현재도 못 민다
                return False

        # 4) 이제 비었거나 비워질 예정이므로, 현재 타깃을 한 칸 밀도록 기록
        self.controller.control_buffer[target_id] = direction
        print(f"push_out: move {target_id} {next_pos} -> {beyond}")
        return True       

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
        self.agvs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_agv = None

    def _back_action_index_from_prev(self):
        if self.center_agv is None:
            return None
        cur = (self.center_x, self.center_y)
        prev = self.center_agv.prev_pos

        # prev→cur로 들어왔으니, 그 반대가 '뒤로가기'
        vx, vy = cur[0]-prev[0], cur[1]-prev[1]
        back_vec = (-vx, -vy)
        vec2idx = {(0,-1):0,(1,0):1,(0,1):2,(-1,0):3}
        return vec2idx.get(back_vec)

    def calculate_action_mask(self):
        """
        교차로 중앙에 있는 AGV에 대한 유효 행동 마스크를 계산합니다.
        반환값: np.array([N, E, S, W]), 유효하면 True, 아니면 False
        """
        # 1. 제어 대상(중앙 AGV)이 없으면 어떤 행동도 유효하지 않음
        is_push_out = False

        if self.center_agv is None:
            return np.zeros(4, dtype=np.bool_), is_push_out

        # 모든 행동이 가능하다고 가정하고 마스크 초기화
        mask = np.ones(4, dtype=np.bool_)
        
        # 3. 규칙 2: 충돌 방지 (벽 또는 다른 AGV)
        state = self.get_state()
        state_N = state[:5]
        state_E = state[5:10]
        state_S = state[10:15]
        state_W = state[15:20]

        if state_N[4] > 0 and state_N[0] != 1:
            mask[0] = False  # 북쪽으로 이동 불가
        if state_E[4] > 0 and state_E[1] != 1:
            mask[1] = False  # 동쪽으로 이동 불가
        if state_S[4] > 0 and state_S[2] != 1:
            mask[2] = False  # 남쪽으로 이동 불가
        if state_W[4] > 0 and state_W[3] != 1:
            mask[3] = False  # 서쪽으로 이동 불가
        
        back_idx = self._back_action_index_from_prev()
        if back_idx is not None:
            mask[back_idx] = False

        # --- 2. [핵심 추가] '모두 막힘' 예외 처리 ---
        # 기본 마스킹 결과, 갈 수 있는 곳이 하나도 없는지 확인
        if not mask.any():
            is_push_out = True
            mask[:] = True  # 모든 행동을 유효하게 설정
            mask[back_idx] = False  # 뒤로가기는 여전히 불가

        return mask, is_push_out