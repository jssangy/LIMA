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
        self.amr_intent_map = {}            # [수정] {amr_id: {'amr_obj': amr, 'current_arm': 'N', 'exit_arm': 'S'}}
        self.is_deadlock = False
        self.swap_conflict_arms = {'N': False, 'E': False, 'S': False, 'W': False}

    def reset(self):
        self.amr_intent_map = {}
        self.is_deadlock = False
        self.swap_conflict_arms = {'N': False, 'E': False, 'S': False, 'W': False}

    def register_amr(self, amr):
        if amr.id in self.amr_intent_map: return  # 이미 등록된 AMR
        
        path = amr.path
        if not path: return

        current_arm_direction = None
        exit_arm_direction = None

        for pos in path:
            if pos == (self.center_x, self.center_y):
                next_pos_index = path.index(pos) + 1
                exit_cell = path[next_pos_index]
                break
        
        for direction, coords in self.lane_coords.items():
            if amr.pos == (self.center_x, self.center_y):
                current_arm_direction = 'C'  # 중앙에 위치한 경우
            if amr.pos in coords:
                current_arm_direction = direction
            if exit_cell in coords:
                exit_arm_direction = direction

        if current_arm_direction and exit_arm_direction:
            self.amr_intent_map[amr.id] = {
                'amr_obj': amr,
                'current_arm': current_arm_direction,
                'exit_arm': exit_arm_direction,
            }
            self._calculate_buffer_inside(amr, current_arm_direction, exit_arm_direction)


    def update_amr_state(self, amr):
        if amr.id  not in self.amr_intent_map: return
        
        if amr.pos == (self.center_x, self.center_y):
            self.center_amr = amr
            return
        
        for direction, coords in self.lane_coords.items():
            if amr.pos == (self.center_x, self.center_y):
                self.amr_intent_map[amr.id]['current_arm'] = 'C'  # 중앙에 위치한 경우
                self._calculate_buffer_inside(amr, self.amr_intent_map[amr.id]['current_arm'], self.amr_intent_map[amr.id]['exit_arm'])
                return
            if amr.pos in coords:
                self.amr_intent_map[amr.id]['current_arm'] = direction
                self._calculate_buffer_inside(amr, self.amr_intent_map[amr.id]['current_arm'], self.amr_intent_map[amr.id]['exit_arm'])
                return
            
    def unregister_amr(self, amr):
        if amr.id in self.amr_intent_map:
            del self.amr_intent_map[amr.id]

    def _calculate_buffer_inside(self, amr, current_arm, exit_arm):
        """
        [신규] 교차로 내부의 AMR을 위한 next_buffer 계산 로직.
        (기존 AMR._recover_inside_intersection 로직을 가져와 수정)
        """
        # --- 경로 이탈 시 복귀 로직 ---
        # 잘못된 팔에 있다면, 무조건 교차로 중앙으로 이동
        if current_arm != 'C' and current_arm != exit_arm:
            center_pos = (self.center_x, self.center_y)
            dx = np.sign(center_pos[0] - amr.pos[0])
            dy = np.sign(center_pos[1] - amr.pos[1])
            amr.next_buffer = (dx, dy)

    def get_state(self):
        """
        [수정] is_ingoing 로직을 '팔에 진입하려는 AMR이 하나라도 있으면 1'로 변경합니다.
        """
        state_vector = []
        center_pos = (self.center_x, self.center_y)

        amrs_by_arm = {d: [] for d in ['N', 'E', 'S', 'W', 'C']}
        for data in self.amr_intent_map.values():
            arm = data['current_arm']
            if arm: amrs_by_arm[arm].append(data)

        for d in ['N', 'E', 'S', 'W']:
            arm_data = amrs_by_arm[d]
            goal_onehot, distance, is_ingoing = [0, 0, 0, 0], 0.0, 0.0

            if arm_data:
                # 가장 가까운 AMR의 정보는 goal_onehot과 distance 계산에 사용
                closest_data = min(arm_data, key=lambda data: self._mdist(data['amr_obj'].pos, center_pos))
                exit_dir = closest_data['exit_arm']
                idx = DIR2IDX.get(exit_dir)
                if idx is not None: goal_onehot[idx] = 1
                distance = self._mdist(closest_data['amr_obj'].pos, center_pos)
                
                # [수정] 해당 팔에 진입(ingoing)하려는 AMR이 하나라도 있는지 확인
                for data in arm_data:
                    if data['exit_arm'] != d:
                        is_ingoing = 1.0
                        break # 하나라도 있으면 더 볼 필요 없음

            state_vector.extend(goal_onehot)
            state_vector.append(distance)
            state_vector.append(is_ingoing)

        center_goal_onehot = [0, 0, 0, 0]
        if amrs_by_arm['C']:
            center_data = amrs_by_arm['C'][0]
            exit_dir = center_data['exit_arm']
            idx = DIR2IDX.get(exit_dir)
            if idx is not None: center_goal_onehot[idx] = 1
        
        state_vector.extend(center_goal_onehot)
        return np.array(state_vector, dtype=np.float32)


    def action_control(self, action, priority):
        """RL 에이전트의 행동을 수행합니다."""
        center_data = next((data for data in self.amr_intent_map.values() if data['current_arm'] == 'C'), None)
        if not center_data: return

        center_amr = center_data['amr_obj']
        chain_base_priority, center_priority = priority + 0.8, priority + 0.6
        move_map = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)}
        dir_map  = {0:'N', 1:'E', 2:'S', 3:'W'}
        
        target_dir = dir_map.get(action)
        chain = self._collect_chain_near_to_far(target_dir)
        move_vec = move_map[action]

        for i, amr_obj in enumerate(chain):
            amr_obj.control_buffer = move_vec
            amr_obj.priority = max(amr_obj.priority, chain_base_priority + i * 0.001)

        center_amr.control_buffer = move_vec
        center_amr.priority = max(center_amr.priority, center_priority)

    def check_deadlock(self):
        """
        [전면 수정] 모든 종류의 데드락을 조사하여 swap_conflict_arms를 종합적으로 업데이트합니다.
        """
        # 1. 상태 초기화
        self.is_deadlock = False
        for d in ['N', 'E', 'S', 'W']: self.swap_conflict_arms[d] = False
        
        if len(self.amr_intent_map) < 2: return False

        # 2. 데이터 준비: amr_intent_map을 기반으로 그룹 분류
        amrs_by_arm = {d: {'ingoing': [], 'outgoing': []} for d in ['N', 'E', 'S', 'W']}
        center_data = None
        intent_list = list(self.amr_intent_map.values())

        for data in intent_list:
            current = data['current_arm']
            exit_arm = data['exit_arm']
            
            if current == 'C':
                center_data = data
                continue
            
            if current in amrs_by_arm:
                if current != exit_arm:
                    amrs_by_arm[current]['ingoing'].append(data)
                else:
                    amrs_by_arm[current]['outgoing'].append(data)

        # 3. 모든 데드락 유형 검사 (즉시 반환하지 않음)
        deadlock_found_this_step = False
        
        # 유형 A: 중앙 AMR의 진출로가 막힌 경우
        if center_data:
            target_exit_arm = center_data['exit_arm']
            if amrs_by_arm.get(target_exit_arm, {}).get('ingoing'):
                deadlock_found_this_step = True

        # 유형 B: 팔 내부에서 진출 차량이 진입 차량에 의해 갇힌 경우 (요청하신 로직)
        center_pos = (self.center_x, self.center_y)
        for arm, groups in amrs_by_arm.items():
            ingoing_data = groups['ingoing']
            outgoing_data = groups['outgoing']

            if not (ingoing_data and outgoing_data): continue

            closest_out_dist = min(self._mdist(d['amr_obj'].pos, center_pos) for d in outgoing_data)
            farthest_in_dist = max(self._mdist(d['amr_obj'].pos, center_pos) for d in ingoing_data)

            if closest_out_dist < farthest_in_dist:
                deadlock_found_this_step = True
                self.swap_conflict_arms[arm] = True
        
        # 유형 C: 팔과 팔 사이의 의도 충돌 (교차 또는 대립)
        for i in range(len(intent_list)):
            for j in range(i + 1, len(intent_list)):
                data1, data2 = intent_list[i], intent_list[j]
                if data1['current_arm'] == 'C' or data2['current_arm'] == 'C': continue
                
                entry1, exit1 = data1['current_arm'], data1['exit_arm']
                entry2, exit2 = data2['current_arm'], data2['exit_arm']

                if (entry1 == exit2 and entry2 == exit1) or (entry1 == entry2 and exit1 == exit2):
                    deadlock_found_this_step = True

        # 4. 최종 결과 설정
        self.is_deadlock = deadlock_found_this_step
        return self.is_deadlock

    def resolve_all_conflicts(self, priority):
        """데드락으로 판정된 팔의 AMR들을 중앙으로 당깁니다."""
        eps, base_force = 1e-3, priority + 0.40
        center_pos = (self.center_x, self.center_y)

        for d in ['N', 'E', 'S', 'W']:
            if self.swap_conflict_arms[d]:
                arm_amrs = [data['amr_obj'] for data in self.amr_intent_map.values() if data['current_arm'] == d]
                if not arm_amrs: continue

                v_in = self._dir_vec(d, inward=True)
                ordered_amrs = sorted(arm_amrs, key=lambda a: (self._mdist(a.pos, center_pos), a.id))
                n = len(ordered_amrs)
                for i, amr in enumerate(ordered_amrs):
                    amr.control_buffer = v_in
                    amr.priority = max(amr.priority, base_force + (n - 1 - i) * eps)

    def calculate_action_mask(self, deadlock_queue):
        """RL 에이전트가 취할 수 있는 행동을 마스킹합니다."""
        center_data = next((data for data in self.amr_intent_map.values() if data['current_arm'] == 'C'), None)
        if not center_data: return np.zeros(4, dtype=np.bool_)

        mask = np.ones(4, dtype=np.bool_)
        center_amr = center_data['amr_obj']
        
        # 1. 뒤로가기 금지
        back_vec = (center_amr.pos[0] - center_amr.prev_pos[0], center_amr.pos[1] - center_amr.prev_pos[1])
        vec2idx = {(0,-1):0, (1,0):1, (0,1):2, (-1,0):3}
        back_idx = vec2idx.get(back_vec)
        if back_idx is not None: mask[back_idx] = False

        # 2. 꽉 찬 팔로 이동 금지
        for direction, action_idx in DIR2IDX.items():
            if not mask[action_idx]: continue
            lane_capacity = len(self.lane_coords.get(direction, []))
            current_occupancy = sum(1 for data in self.amr_intent_map.values() if data['current_arm'] == direction)
            if current_occupancy >= lane_capacity: mask[action_idx] = False

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


    def _collect_chain_near_to_far(self, d: str):
        if d not in self.lane_coords: return []
        amrs_on_arm = [data['amr_obj'] for data in self.amr_intent_map.values() if data['current_arm'] == d]
        if not amrs_on_arm: return []
        
        pos2amr = {a.pos: a for a in amrs_on_arm}
        chain, started = [], False
        for p in self.lane_coords[d]:
            if p in pos2amr:
                if not started: started = True
                chain.append(pos2amr[p])
            elif started: break
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

    def _mdist(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

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