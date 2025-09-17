import numpy as np
from itertools import chain

DIR2IDX = {"N": 0, "E": 1, "S": 2, "W": 3}

class Intersection:
    def __init__(self, intersection_data, neighbors_map):
        self.center_x, self.center_y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
        self.id = f'x{self.center_x}y{self.center_y}'
        self.neighbors = neighbors_map

        self.lane_coords = {
            'N': [(self.center_x, self.center_y - i) for i in range(1, self.len_N + 1)],
            'E': [(self.center_x + i, self.center_y) for i in range(1, self.len_E + 1)],
            'S': [(self.center_x, self.center_y + i) for i in range(1, self.len_S + 1)],
            'W': [(self.center_x - i, self.center_y) for i in range(1, self.len_W + 1)]
        }

        self.all_lane_coords = set(chain.from_iterable(self.lane_coords.values()))
        self.all_lane_coords.add((self.center_x, self.center_y))

        # 이벤트 기반 AMR object 추적
        self.amrs_in_intersection = set()  # 교차로 내 AMR만 추적
        self.amrs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_amr = None
        self.is_deadlock = False

        self.ingoing = {"N": False, "E": False, "S": False, "W": False}
        self.outgoing = {"N": False, "E": False, "S": False, "W": False}

    def reset(self):
        self.amrs_in_intersection.clear()
        self.amrs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_amr = None
        self.ingoing = {"N": False, "E": False, "S": False, "W": False}
        self.outgoing = {"N": False, "E": False, "S": False, "W": False}
        self.is_deadlock = False
        self.swap_conflict_arms = {'N': False, 'E': False, 'S': False, 'W': False}

    def add_amr(self, amr_object):
        nxt = amr_object.next_buffer
        self.amrs_in_intersection.add(amr_object)

        if amr_object.pos == (self.center_x, self.center_y):
            self.center_amr = amr_object
        else:        
            cx, cy = self.center_x, self.center_y
            for direction, coords in self.lane_coords.items():
                if amr_object.pos in coords:
                    self.amrs_in_lanes[direction].append(amr_object)
                    curd = (np.sign(amr_object.pos[0] - cx), np.sign(amr_object.pos[1] - cy))

                    if curd == nxt:
                        self.outgoing[direction] = True
                    elif curd == (-nxt[0], -nxt[1]):
                        self.ingoing[direction] = True
                    break

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
            amrs = self.amrs_in_lanes[d]  # [AMR,...]
            goal_onehot = [0, 0, 0, 0]
            distance = 0

            if amrs:
                axis, sel = closest_cfg[d]
                key_fn = (lambda a: a.pos[1]) if axis == 'y' else (lambda a: a.pos[0])
                closest_amr = sel(amrs, key=key_fn)

                path = closest_amr.path
                if path:
                    exit_dir = self._get_exit_direction(path)
                    idx = DIR2IDX.get(exit_dir)
                    if idx is not None: goal_onehot[idx] = 1

                distance = abs(closest_amr.pos[0] - center[0]) + abs(closest_amr.pos[1] - center[1])

            ingoing = 1.0 if self.ingoing[d] else 0.0

            state_vector.extend(goal_onehot)
            state_vector.append(distance)
            state_vector.append(ingoing)

        center_goal_onehot = [0, 0, 0, 0]
        if self.center_amr:
            path = self.center_amr.path
            if path:
                exit_dir = self._get_exit_direction(path)
                idx = DIR2IDX.get(exit_dir)
                if idx is not None: center_goal_onehot[idx] = 1

        state_vector.extend(center_goal_onehot)
        return np.array(state_vector, dtype=np.float32)

    def action_control(self, action, priority):
        if self.center_amr is None:
            return
        
        base_priority = priority
        # 1순위 그룹 (밀려나는 체인): P + 0.8 ~
        chain_base_priority = base_priority + 0.8
        # 2순위 (밀려나는 중앙 AMR): P + 0.6
        center_priority = base_priority + 0.6
        
        move_map = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
        dir_map  = {0:'N', 1:'E', 2:'S', 3:'W'}
        
        target_dir = dir_map.get(action)
        chain = self._collect_chain_near_to_far(target_dir)
        move_vec = move_map[action]

        # 체인은 안쪽->바깥쪽 순서이므로, 인덱스가 클수록 바깥쪽 AMR.
        # 바깥쪽 AMR이 더 높은 우선순위를 갖도록 인덱스를 활용.
        for i, amr_obj in enumerate(chain):
            amr_obj.control_buffer = move_vec
            amr_obj.priority = max(amr_obj.priority, chain_base_priority + i * 0.001)    # 우선순위 1순위

        # 중앙 AMR도 동일한 방향으로 이동하도록 지시
        self.center_amr.control_buffer = move_vec
        self.center_amr.priority = max(self.center_amr.priority, center_priority)   # 우선순위 2순위

    def check_deadlock(self):
        amrs_in_intersection = list(self.amrs_in_intersection)
        if len(amrs_in_intersection) < 2:
            return False

        # 모든 AMR 쌍에 대해 데드락 검사 (O(N^2))
        for i in range(len(amrs_in_intersection)):
            for j in range(i + 1, len(amrs_in_intersection)):
                amr1 = amrs_in_intersection[i]
                amr2 = amrs_in_intersection[j]

                # 1. 즉각적인 스왑 충돌 검사
                is_immediate_swap = self._check_immediate_swap(amr1, amr2)
                
                # 2. 경로 기반 스왑 충돌 검사
                is_path_conflict = self._check_swapping_path(amr1, amr2) or self._check_swapping_path(amr2, amr1)

                if is_immediate_swap or is_path_conflict:
                    self.is_deadlock = True
                    
                    # 이 스왑이 어떤 팔(arm)과 관련 있는지 식별하여 기록
                    for d in ['N', 'E', 'S', 'W']:
                        if amr1.pos in self.lane_coords[d] or amr2.pos in self.lane_coords[d]:
                            self.swap_conflict_arms[d] = True
                    
                    # 데드락이 하나라도 발견되면 즉시 종료
                    return True

        # 루프를 모두 통과했다면 데드락이 없는 것
        return False

    def _check_immediate_swap(self, amr1, amr2):
        """
        [신규] A의 다음 위치가 B의 현재 위치이고, B의 다음 위치가 A의 현재 위치인지 확인합니다.
        """
        move1 = amr1.next_buffer
        move2 = amr2.next_buffer

        pos1 = amr1.pos
        pos2 = amr2.pos

        next_pos1 = (pos1[0] + move1[0], pos1[1] + move1[1])
        next_pos2 = (pos2[0] + move2[0], pos2[1] + move2[1])

        # 스와핑 조건 확인
        return next_pos1 == pos2 and next_pos2 == pos1

    def _check_swapping_path(self, amr1, amr2):
        """
        [신규] A(amr1)의 경로에 B(amr2)의 위치가 있고, 그 경로의 일부를 뒤집은 것이 B의 경로에 포함되는지 확인합니다.
        """
        path1 = amr1.path
        path2 = amr2.path
        if not path1 or not path2:
            return False

        pos2 = amr2.pos

        # A의 경로에서 B의 현재 위치 인덱스 찾기
        try:
            index2_in_1 = path1.index(pos2)
        except ValueError:
            return False

        # A의 현재 위치 인덱스
        current_index1 = amr1.path_cursor

        # A의 현재 위치가 B의 위치보다 앞에 있어야 경로 스왑 가능
        if current_index1 >= index2_in_1:
            return False

        # A의 경로에서 스왑을 검사할 구간 추출: (A의 다음 위치)부터 (B의 이전 위치)까지
        sub_path1 = path1[current_index1 + 1 : index2_in_1]
        if not sub_path1:
            return False
        
        reversed_sub_path1 = sub_path1[::-1]
        L = len(reversed_sub_path1)

        # B의 경로에서 스왑 구간이 포함되는지 확인
        if len(path2) < L:
            return False
        
        for i in range(len(path2) - L + 1):
            if path2[i:i + L] == reversed_sub_path1:
                return True
        
        return False

    
    def resolve_all_conflicts(self, priority):
        eps = 1e-3
        base_force  = priority + 0.40   # Case3: 강제 inward tier

        center = (self.center_x, self.center_y)

        def mdist(p):
            return abs(p[0] - center[0]) + abs(p[1] - center[1])

        def sort_close_first(amrs):
            return sorted(amrs, key=lambda a: (mdist(a.pos), a.id))

        for d in ['N', 'E', 'S', 'W']:
            arm_amrs = list(self.amrs_in_lanes.get(d, []))
            if len(arm_amrs) < 2:
                continue

            if self.swap_conflict_arms[d]:
                v_in = self._dir_vec(d, inward=True)
                ordered_all = sort_close_first(arm_amrs)
                n = len(ordered_all)
                for i, amr in enumerate(ordered_all):
                    amr.control_buffer = v_in
                    amr.priority = max(amr.priority, base_force + (n - 1 - i) * eps)
                continue


    def _collect_chain_near_to_far(self, d: str):
        """
        체인에 엮인 AMR 객체 목록을 반환
        """
        cells = self.lane_coords.get(d, [])  # 반드시 center→outside 순서의 '리스트'여야 함
        pos2amrs = {a.pos: a for a in self.amrs_in_lanes.get(d, [])}

        chain = []
        started = False
        for p in cells:
            if not started:
                if p in pos2amrs:
                    chain.append(pos2amrs[p])
                    started = True
            else:
                if p in pos2amrs:
                    chain.append(pos2amrs[p])
                else:
                    break
        return chain

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

    def _back_action_index_from_prev(self):
        if self.center_amr is None:
            return None
        cur = (self.center_x, self.center_y)
        prev = self.center_amr.prev_pos

        # prev→cur로 들어왔으니, 그 반대가 '뒤로가기'
        vx, vy = cur[0]-prev[0], cur[1]-prev[1]
        back_vec = (-vx, -vy)
        vec2idx = {(0,-1):0,(1,0):1,(0,1):2,(-1,0):3}
        return vec2idx.get(back_vec)

    def calculate_action_mask(self, deadlock_queue):
        # 중앙 AMR 없으면 전부 금지
        if self.center_amr is None:
            return np.zeros(4, dtype=np.bool_)

        mask = np.ones(4, dtype=np.bool_)  # N E S W

        # 1) 뒤로가기 금지 (기존 로직)
        back_idx = self._back_action_index_from_prev()
        if back_idx is not None and 0 <= back_idx < 4:
            mask[back_idx] = False

        # 2) 용량이 가득 찬 방향으로 이동 금지
        for direction, action_idx in DIR2IDX.items():
            if not mask[action_idx]: continue
            lane_capacity = len(self.lane_coords.get(direction, []))
            current_occupancy = len(self.amrs_in_lanes.get(direction, []))
            if current_occupancy >= lane_capacity:
                mask[action_idx] = False

        # 3) 우선순위 규칙에 따른 마스킹
        if not deadlock_queue:
            return mask

        try:
            current_rank = deadlock_queue.index(self.id)
        except ValueError:
            current_rank = float('inf')

        for direction, action_idx in DIR2IDX.items():
            if not mask[action_idx]: continue

            # 해당 방향으로 이동 시 진입할 이웃 교차로 ID 확인
            neighbor_id = self.neighbors.get(direction)
            if neighbor_id is None:
                continue # 이웃이 없으면 마스킹할 필요 없음

            # 이웃 교차로의 우선순위(랭크) 확인
            try:
                neighbor_rank = deadlock_queue.index(neighbor_id)
            except ValueError:
                neighbor_rank = float('inf')

            # 우선순위 규칙: 더 높은 순위(낮은 랭크)의 교차로로 진입 금지
            if neighbor_rank < current_rank:
                mask[action_idx] = False
        
        return mask

    def _dir_vec(self, d: str, inward: bool = False):
        """
        [수정] inward 파라미터를 받아 안쪽/바깥쪽 방향 벡터를 반환합니다.
        """
        # 기본값은 바깥쪽(outward)을 향하는 벡터
        vec_map = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
        vx, vy = vec_map[d]
        
        # inward가 True이면 벡터를 뒤집어 안쪽을 향하게 함
        if inward:
            return -vx, -vy
        return vx, vy