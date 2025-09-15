import numpy as np
from itertools import chain, combinations
from typing import Dict

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
        self.center_deadlock = False

        self.ingoing = {"N": False, "E": False, "S": False, "W": False}
        self.outgoing = {"N": False, "E": False, "S": False, "W": False}

    def reset(self):
        self.amrs_in_intersection.clear()
        self.amrs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_amr = None
        self.ingoing = {"N": False, "E": False, "S": False, "W": False}
        self.outgoing = {"N": False, "E": False, "S": False, "W": False}
        self.center_deadlock = False

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
        # 1순위 그룹 (밀려나는 체인): P + 0.6 ~
        chain_base_priority = base_priority + 0.6
        # 2순위 (밀려나는 중앙 AMR): P + 0.3
        center_priority = base_priority + 0.3
        
        move_map = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
        dir_map  = {0:'N', 1:'E', 2:'S', 3:'W'}
        
        target_dir = dir_map.get(action)
        chain = self._collect_chain_near_to_far(target_dir)
        move_vec = move_map[action]

        # 체인은 안쪽->바깥쪽 순서이므로, 인덱스가 클수록 바깥쪽 AMR.
        # 바깥쪽 AMR이 더 높은 우선순위를 갖도록 인덱스를 활용.
        num_in_chain = len(chain)
        for i, amr_obj in enumerate(chain):
            amr_obj.control_buffer = move_vec
            amr_obj.priority = max(amr_obj.priority, chain_base_priority + i * 0.01) # 1순위

        # 중앙 AMR도 동일한 방향으로 이동하도록 지시
        self.center_amr.control_buffer = move_vec
        self.center_amr.priority = max(self.center_amr.priority, center_priority) # 2순위

    def check_deadlock(self):
        """
        중앙 AMR와 관련된 데드락만 신속하게 판정합니다.
        - 중앙 AMR가 나가려는 방향에, 들어오려는(ingoing) AMR가 있으면 데드락으로 간주합니다.
        """
        # is_deadlock은 더 이상 사용하지 않으므로 항상 False로 설정
        self.is_deadlock = False

        if self.center_amr is None:
            self.center_deadlock = False
            return False

        # 1. 중앙 AMR의 다음 이동 방향 알아내기
        move = self.center_amr.next_buffer      
        vec_to_dir = {(0, -1): 'N', (1, 0): 'E', (0, 1): 'S', (-1, 0): 'W'}
        target_dir = vec_to_dir.get(move)

        if target_dir and self.ingoing.get(target_dir, False):
            self.center_deadlock = True
            return True
        
        self.center_deadlock = False
        return False
    
    def resolve_arm_swaps_all(self, priority):
        """
        [수정된 최종 로직]
        1. 순수 스왑 충돌을 감지합니다.
        2. 스왑이 감지되면, 중심에 가장 가까운 AMR부터 시작하여,
           스왑에 연루된 AMR 중 더 바깥쪽에 있는 AMR까지 체인을 구축합니다.
        3. 체인에 포함된 모든 AMR을 중심으로 끌어당깁니다.
        """
        swap_chain_base_priority = priority + 0.1

        for d in ['N', 'E', 'S', 'W']:
            # 충돌 가능성이 있는 팔만 검사
            if not (self.ingoing[d] and self.outgoing[d]):
                continue

            amrs_in_arm = self.amrs_in_lanes.get(d, [])
            if len(amrs_in_arm) < 2:
                continue

            # 1. 순수 스왑 충돌 감지
            for amr_a, amr_b in combinations(amrs_in_arm, 2):
                is_swap = (
                    (amr_a.pos[0] + amr_a.next_buffer[0], amr_a.pos[1] + amr_a.next_buffer[1]) == amr_b.pos and
                    (amr_b.pos[0] + amr_b.next_buffer[0], amr_b.pos[1] + amr_b.next_buffer[1]) == amr_a.pos
                )

                if is_swap:
                    # 2. 스왑이 감지되면 체인 구축 시작
                    center_pos = (self.center_x, self.center_y)
                    
                    # 스왑에 연루된 두 AMR 중 더 바깥쪽 AMR 찾기
                    dist_a = abs(amr_a.pos[0] - center_pos[0]) + abs(amr_a.pos[1] - center_pos[1])
                    dist_b = abs(amr_b.pos[0] - center_pos[0]) + abs(amr_b.pos[1] - center_pos[1])
                    outer_swap_amr = amr_a if dist_a > dist_b else amr_b

                    # 팔에 있는 모든 AMR을 중심에서 '가까운' 순서대로 정렬
                    sorted_amrs_near_to_far = sorted(
                        amrs_in_arm,
                        key=lambda a: abs(a.pos[0] - center_pos[0]) + abs(a.pos[1] - center_pos[1])
                    )
                    
                    # 3. 체인 확장 (가장 가까운 AMR ~ 바깥쪽 스왑 AMR)
                    try:
                        # 체인이 끝나는 인덱스 (바깥쪽 스왑 AMR의 위치)
                        end_index = sorted_amrs_near_to_far.index(outer_swap_amr)
                    except ValueError:
                        continue # 혹시 모를 경우 대비

                    # 가장 가까운 AMR부터 바깥쪽 스왑 AMR까지를 체인으로 정의
                    conflict_chain = sorted_amrs_near_to_far[:end_index + 1]
                    
                    # 4. 체인에 포함된 모든 AMR 제어
                    # 체인은 안쪽->바깥쪽 순서.
                    # 안쪽 AMR이 더 높은 우선순위를 갖도록 역순으로 미세 조정.
                    num_in_chain = len(conflict_chain)
                    dir_vec_inward = self._dir_vec(d, inward=True)
                    for i, amr_in_chain in enumerate(reversed(conflict_chain)):
                        fine_tuned_priority = swap_chain_base_priority + ( (num_in_chain - 1 - i) * 0.01 )
                        amr_in_chain.control_buffer = dir_vec_inward
                        amr_in_chain.priority = max(amr_in_chain.priority, fine_tuned_priority)

                    # 해당 팔에서 스왑을 하나 해결했으면, 다음 팔로 넘어감
                    break 
            else:
                continue
            break

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