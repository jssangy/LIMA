import numpy as np
from itertools import chain
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

        # 이벤트 기반 AGV object 추적
        self.amrs_in_intersection = set()  # 교차로 내 AGV만 추적
        self.amrs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_amr = None
        self.center_deadlock = False

        self.ingoing = {"N": False, "E": False, "S": False, "W": False}
        self.outgoing = {"N": False, "E": False, "S": False, "W": False}

    def reset(self):
        self.amrs_in_intersection.clear()
        self.amrs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_agv = None
        self.ingoing = {"N": False, "E": False, "S": False, "W": False}
        self.outgoing = {"N": False, "E": False, "S": False, "W": False}
        self.center_deadlock = False

    def add_amr(self, amr_object):
        nxt = amr_object.control_buffer
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
                    idx = {'N':0,'E':1,'S':2,'W':3}.get(exit_dir, None)
                    if idx is not None: goal_onehot[idx] = 1

                distance = abs(closest_amr.pos[0] - center[0]) + abs(closest_amr.pos[1] - center[1])

            ingoing = 1.0 if self.ingoing[d] else 0.0

            state_vector.extend(goal_onehot)
            state_vector.append(distance)
            state_vector.append(ingoing)

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

    def action_control(self, action):
        if self.center_agv is None:
            return
        move_map = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
        dir_map  = {0:'N', 1:'E', 2:'S', 3:'W'}
        
        target_dir = dir_map.get(action)

        # 센터에서 target_dir 방향으로 밀어내는 체인에 대한 이동 계획을 버퍼에 기록
        chain = self._collect_chain_near_to_far(target_dir)
        move_vec = move_map[action]

        # 체인의 모든 AGV에게 동일한 이동 방향을 지시
        for agv_id in chain:
            self.controller.control_buffer[agv_id] = move_vec
        
        # 중앙 AGV도 동일한 방향으로 이동하도록 지시
        self.controller.control_buffer[self.center_agv.id] = move_vec

    def check_deadlock(self):
        """
        [최적화된 버전]
        중앙 AGV와 관련된 데드락만 신속하게 판정합니다.
        - 중앙 AGV가 나가려는 방향에, 들어오려는(ingoing) AGV가 있으면 데드락으로 간주합니다.
        """
        # is_deadlock은 더 이상 사용하지 않으므로 항상 False로 설정
        self.is_deadlock = False

        if self.center_agv is None:
            self.center_deadlock = False
            return False

        # 1. 중앙 AGV의 다음 이동 방향 알아내기
        move = self._planned_move(self.center_agv)
        if move is None:
            self.center_deadlock = False
            return False
        
        vec_to_dir = {(0, -1): 'N', (1, 0): 'E', (0, 1): 'S', (-1, 0): 'W'}
        target_dir = vec_to_dir.get(move)

        if target_dir is None:
            self.center_deadlock = False
            return False

        # 2. 해당 방향에 들어오려는(ingoing) AGV가 있는지 확인
        if self.ingoing.get(target_dir, False):
            self.center_deadlock = True
            return True
        
        # 3. 그 외의 경우는 데드락이 아님
        self.center_deadlock = False
        return False

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
        if self.center_agv is None:
            return None
        cur = (self.center_x, self.center_y)
        prev = self.center_agv.prev_pos

        # prev→cur로 들어왔으니, 그 반대가 '뒤로가기'
        vx, vy = cur[0]-prev[0], cur[1]-prev[1]
        back_vec = (-vx, -vy)
        vec2idx = {(0,-1):0,(1,0):1,(0,1):2,(-1,0):3}
        return vec2idx.get(back_vec)

    def calculate_action_mask(self, deadlock_queue):
        # 중앙 AMR 없으면 전부 금지
        if self.center_agv is None:
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
            current_occupancy = len(self.agvs_in_lanes.get(direction, []))
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

    def _dir_vec(self, d: str):
        return {'N': (0,-1), 'E': (1,0), 'S': (0,1), 'W': (-1,0)}[d]

    def _collect_chain_near_to_far(self, d: str):
        """팔 d에서 센터에 '가장 가까운 점유칸'부터 바깥으로 연속 점유된 AGV id 목록(near→far)."""
        cells = self.lane_coords.get(d, [])  # 반드시 center→outside 순서의 '리스트'여야 함
        pos2id = {a.pos: a.id for a in self.agvs_in_lanes.get(d, [])}

        chain = []
        started = False
        for p in cells:
            if not started:
                if p in pos2id:
                    chain.append(pos2id[p])
                    started = True
                # 아직 시작 못 했으면 다음 칸으로 continue
            else:
                if p in pos2id:
                    chain.append(pos2id[p])
                else:
                    break  # 연속 끊기면 종료
        return chain

    def _detect_arm_swap_pairs(self, d: str) -> bool:
        """
        팔 d에서 인접한 두 AMR이 서로 자리로 이동하려는지(A→Bpos & B→Apos).
        감지되면 True.
        """
        cells = self.lane_coords.get(d, [])
        pos2agv = {a.pos: a for a in self.agvs_in_lanes.get(d, [])}
        for i in range(len(cells) - 1):
            p, q = cells[i], cells[i + 1]  # p가 center에 더 가까움
            a, b = pos2agv.get(p), pos2agv.get(q)
            if not a or not b:
                continue
            ma, mb = self._planned_move(a), self._planned_move(b)
            if ma is None or mb is None:
                continue
            a_next = (a.pos[0] + ma[0], a.pos[1] + ma[1])
            b_next = (b.pos[0] + mb[0], b.pos[1] + mb[1])
            if a_next == b.pos and b_next == a.pos:
                return True
        return False

    def resolve_arm_swaps_all(self):
        dirs = ['N', 'E', 'S', 'W']
        dir_rank = {'N':0, 'E':1, 'S':2, 'W':3}
        cx, cy = self.center_x, self.center_y

        hit = [d for d in dirs if self._detect_arm_swap_pairs(d)]
        if not hit:
            return

        PR_PULL = 50  # 끌어오기 우선순위(밀어내기보다 낮게)

        for d in hit:
            chain = self._collect_chain_near_to_far(d)  # [head,...,tail]
            if not chain:
                continue
            dx, dy = self._dir_vec(d)
            move_in = (-dx, -dy)  # 센터 방향

            for k, agv_id in enumerate(chain):
                pos = self.controller.agv_pos.get(agv_id)
                if pos is None:
                    continue
                dist = abs(pos[0]-cx) + abs(pos[1]-cy)
                # head -> tail 순서가 먼저 움직이도록 order_key 구성
                order_key = (dist, dir_rank[d], k)
                self._plan_add(agv_id, move_in, PR_PULL, order_key)
