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

    def _dir_vec(self, d):  # 'N','E','S','W' -> (dx,dy)
        return {'N':(0,-1), 'E':(1,0), 'S':(0,1), 'W':(-1,0)}[d]

    def _lane_line(self, d):
        """센터 바로 앞부터 팔 끝까지 좌표 리스트 [adj1,...,end]"""
        L = {'N': self.len_N, 'E': self.len_E, 'S': self.len_S, 'W': self.len_W}[d]
        dx, dy = self._dir_vec(d)
        return [(self.center_x + dx*i, self.center_y + dy*i) for i in range(1, L+1)]

    def _pos_occ_global(self, pos):
        """맵 전역 점유 체크(컨트롤러가 들고 있는 포지션 사용)"""
        return any(p == pos for p in self.controller.agv_pos.values())

    def _push_chain_once(self, dname) -> bool:
        """
        dname 방향으로 체인 밀어내기 세팅:
        - 팔 끝칸이 차 있으면 '그 다음 칸(out_next)'이 비어야 밀어낼 수 있음
        - 끝에서부터 한 칸씩 모두 같은 (dx,dy)로 move 지시
        - 마지막으로 센터도 같은 방향으로 move
        반환: 성공적으로 세팅했으면 True, 불가하면 False
        """
        if self.center_agv is None:
            return False

        line = self._lane_line(dname)             # [adj1, ..., end]
        dx, dy = self._dir_vec(dname)
        end = line[-1]
        out_next = (end[0] + dx, end[1] + dy)      # 팔 밖 한 칸

        # 팔 끝칸에 AGV가 있고 out_next가 이미 전역 점유면 못 민다
        occ_dir = self.agvs_in_lanes[dname]        # {agv_id: pos}
        pos2id = {pos: aid for aid,pos in occ_dir.items()}

        if end in pos2id and self._pos_occ_global(out_next):
            return False

        # 체인 세팅: far -> near (끝에서부터)
        for pos in reversed(line):
            if pos in pos2id:
                aid = pos2id[pos]
                self.controller.control_buffer[aid] = (dx, dy)

        # 센터 앞칸에 최종적으로 빈자리가 생긴다고 가정하고 센터도 전진
        self.controller.control_buffer[self.center_agv] = (dx, dy)

        # (선택) 이동 순서 보장: 먼저 먼 것부터 이동시키기
        seq = [pos2id[p] for p in reversed(line) if p in pos2id] + [self.center_agv]
        setattr(self.controller, "push_sequence", seq)
        return True

    def action_control(self, actions, is_push_out=False):
        """actions: 0=N,1=E,2=S,3=W"""
        if self.center_agv is None:
            return
        idx2dir = {0:'N',1:'E',2:'S',3:'W'}

        # 밀어내기 모드가 아니라면 센터만 이동
        if not is_push_out:
            move_map = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
            self.controller.control_buffer[self.center_agv] = move_map[int(actions)]
            return

        # 밀어내기: 선택 방향 우선, 실패 시 다른 방향 순차 시도
        order = [int(actions)] + [i for i in (0,1,2,3) if i != int(actions)]
        for k in order:
            if self._push_chain_once(idx2dir[k]):
                return
        # 모두 실패하면 아무 것도 안 함(혹은 센터만 대기/임의 방향 이동 등 정책 선택)                

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

        # --- 2. [핵심 추가] '모두 막힘' 예외 처리 ---
        # 기본 마스킹 결과, 갈 수 있는 곳이 하나도 없는지 확인
        if not mask.any():
            is_push_out = True
            mask[:] = True  # 모든 행동을 유효하게 설정

        return mask, is_push_out