import numpy as np
from itertools import chain, combinations
from typing import Dict
from collections import defaultdict

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
        num_in_chain = len(chain)
        for i, amr_obj in enumerate(chain):
            amr_obj.control_buffer = move_vec
            amr_obj.priority = max(amr_obj.priority, chain_base_priority + i * 0.001)    # 우선순위 1순위

        # 중앙 AMR도 동일한 방향으로 이동하도록 지시
        self.center_amr.control_buffer = move_vec
        self.center_amr.priority = max(self.center_amr.priority, center_priority)   # 우선순위 2순위

    def check_deadlock(self):
        """
        [수정] 데드락 상태를 판정하고 self.is_deadlock을 설정합니다.
        1. 중앙 AMR의 진출로가 막힌 경우 (기존 로직)
        2. 특정 팔(arm)에서 스왑 충돌이 발생한 경우 (신규 로직)
        """
        # 1. 중앙 AMR 데드락 검사
        if self.center_amr:
            move = self.center_amr.next_buffer      
            vec_to_dir = {(0, -1): 'N', (1, 0): 'E', (0, 1): 'S', (-1, 0): 'W'}
            target_dir = vec_to_dir.get(move)

            if target_dir and self.ingoing.get(target_dir, False):
                self.is_deadlock = True
                return True # 데드락 확정

        # 2. 스왑 충돌 데드락 검사
        center = (self.center_x, self.center_y)
        def mdist(p):
            return abs(p[0] - center[0]) + abs(p[1] - center[1])

        for d in ['N', 'E', 'S', 'W']:
            # 해당 팔에 진입/진출 차량이 모두 있어야 스왑 가능성 존재
            if not (self.ingoing.get(d, False) and self.outgoing.get(d, False)):
                continue

            arm_amrs = self.amrs_in_lanes.get(d, [])
            v_in = self._dir_vec(d, inward=True)
            v_out = (-v_in[0], -v_in[1])

            ing_list = [a for a in arm_amrs if a.next_buffer == v_in]
            out_list = [a for a in arm_amrs if a.next_buffer == v_out]

            if not ing_list or not out_list:
                continue

            # 가장 중심에 가까운 진출 차량 vs 가장 중심에서 먼 진입 차량
            closest_out_dist = min(mdist(a.pos) for a in out_list)
            farthest_in_dist = max(mdist(a.pos) for a in ing_list)

            # 진출 차량이 진입 차량보다 안쪽에 갇혀있으면 스왑 충돌
            if closest_out_dist < farthest_in_dist:
                self.is_deadlock = True
                return True # 데드락 확정

        # 위 조건에 해당하지 않으면 데드락 아님
        self.is_deadlock = False
        return False
    
    def resolve_all_conflicts(self, priority):
        """
        통합 로직:
        Case1) ingoing만 True:   가까운→먼 순으로 우선순위 부여
        Case2) outgoing만 True:  먼→가까운 순으로 우선순위 부여
        Case3) 둘 다 True:
            - if (closest_out_dist < farthest_in_dist):
                해당 팔의 모든 AMR을 중심 방향(inward)으로 몰고,
                가까울수록 높은 priority
            else:
                ingoing(가까운→먼) > outgoing(먼→가까운) tier로 부여
        """
        eps = 1e-3
        base_single = priority   # Case1/2 tier
        base_in     = priority + 0.20   # Case3: ingoing tier
        base_out    = priority + 0.10   # Case3: outgoing tier
        base_force  = priority + 0.40   # Case3: 강제 inward tier

        center = (self.center_x, self.center_y)

        def mdist(p):
            return abs(p[0] - center[0]) + abs(p[1] - center[1])

        def sort_close_first(amrs):
            return sorted(amrs, key=lambda a: (mdist(a.pos), a.id))

        def sort_far_first(amrs):
            return sorted(amrs, key=lambda a: (-mdist(a.pos), -a.id))

        for d in ['N', 'E', 'S', 'W']:
            arm_amrs = list(self.amrs_in_lanes.get(d, []))
            if len(arm_amrs) < 2:
                continue

            v_in  = self._dir_vec(d, inward=True)
            v_out = (-v_in[0], -v_in[1])

            ing_flag = bool(self.ingoing.get(d, False))
            out_flag = bool(self.outgoing.get(d, False))

            # next_buffer 기준 그룹 분리
            ing_list = [a for a in arm_amrs if a.next_buffer == v_in]
            out_list = [a for a in arm_amrs if a.next_buffer == v_out]

            # -------- Case 1: ingoing만 True --------
            if ing_flag and not out_flag:
                targets = ing_list if ing_list else arm_amrs
                ordered = sort_close_first(targets)          # 가까운→먼
                n = len(ordered)
                for i, amr in enumerate(ordered):
                    amr.priority = max(amr.priority, base_single + (n - 1 - i) * eps)
                continue

            # -------- Case 2: outgoing만 True --------
            if out_flag and not ing_flag:
                targets = out_list if out_list else arm_amrs
                ordered = sort_far_first(targets)            # 먼→가까운
                n = len(ordered)
                for i, amr in enumerate(ordered):
                    amr.priority = max(amr.priority, base_single + (n - 1 - i) * eps)
                continue

            # -------- Case 3: 둘 다 True --------
            if ing_flag and out_flag:
                # 가까운 outgoing vs 먼 ingoing 비교
                closest_out_dist = min([mdist(a.pos) for a in out_list], default=None)
                farthest_in_dist = max([mdist(a.pos) for a in ing_list], default=None)

                # 3-A) 강제 inward 조건 충족: 모든 AMR을 중심으로 몰기
                if (closest_out_dist is not None and farthest_in_dist is not None
                        and closest_out_dist < farthest_in_dist):
                    ordered_all = sort_close_first(arm_amrs)  # 가까운→먼
                    n = len(ordered_all)
                    for i, amr in enumerate(ordered_all):
                        amr.control_buffer = v_in
                        amr.priority = max(amr.priority, base_force + (n - 1 - i) * eps)
                    continue

                # 3-B) 일반 case3: ingoing > outgoing tier
                if ing_list:
                    ordered_in = sort_close_first(ing_list)   # 가까운→먼
                    n_in = len(ordered_in)
                    for i, amr in enumerate(ordered_in):
                        amr.priority = max(amr.priority, base_in + (n_in - 1 - i) * eps)

                if out_list:
                    ordered_out = sort_far_first(out_list)    # 먼→가까운
                    n_out = len(ordered_out)
                    for i, amr in enumerate(ordered_out):
                        amr.priority = max(amr.priority, base_out + (n_out - 1 - i) * eps)

            # ing_flag==False and out_flag==False 이면 아무 것도 하지 않음


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