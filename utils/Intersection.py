import numpy as np
from itertools import chain
from typing import Dict

DIR2IDX = {"N": 0, "E": 1, "S": 2, "W": 3}
IDX2DIRC = {0: "N", 1: "E", 2: "S", 3: "W"}
IDX2DIR = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

PAIRS = [(s, d) for s in range(4) for d in range(4) if s != d]
# => [(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0),(2,1),(2,3),(3,0),(3,1),(3,2)]

IDX2PAIR = PAIRS
PAIR2IDX = {pair: i for i, pair in enumerate(PAIRS)}

def decode_action(a12: int) -> tuple[int,int]:
    # 0..11 -> (src,dst)
    return IDX2PAIR[int(a12)]

def encode_action(src: int, dst: int) -> int:
    # (src,dst) -> 0..11
    assert 0 <= src < 4 and 0 <= dst < 4 and src != dst
    return PAIR2IDX[(src, dst)]

PR_PULL = 50      # Pull to Center (플래닝/실행 우선순위 가장 낮음)
PR_PULL_CENTER = 70  # Pull to Center (센터 앞 칸에 도달한 경우)
PR_ACTION = 90    # Center Action
PR_PUSH = 100     # Center Action Push (플래닝/실행 우선순위 가장 높음)

class Intersection:
    def __init__(self, intersection_data, controller_ref, neighbors_map, present_dirs):
        self.center_x, self.center_y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
        self.id = f'x{self.center_x}y{self.center_y}'
        self.controller = controller_ref
        self.neighbors = neighbors_map
        self.map = self.controller.map

        if present_dirs is None:
            present_dirs = {d for d,L in zip("NESW",[self.len_N,self.len_E,self.len_S,self.len_W]) if L>0}
        self.present_dirs = set(present_dirs)

        self.lane_coords = {}
        if 'N' in self.present_dirs:
            self.lane_coords['N'] = [(self.center_x, self.center_y - i) for i in range(1, self.len_N + 1)]
        if 'E' in self.present_dirs:
            self.lane_coords['E'] = [(self.center_x + i, self.center_y) for i in range(1, self.len_E + 1)]
        if 'S' in self.present_dirs:
            self.lane_coords['S'] = [(self.center_x, self.center_y + i) for i in range(1, self.len_S + 1)]
        if 'W' in self.present_dirs:
            self.lane_coords['W'] = [(self.center_x - i, self.center_y) for i in range(1, self.len_W + 1)]

        self.all_lane_coords = set(chain.from_iterable(self.lane_coords.values()))
        self.all_lane_coords.add((self.center_x, self.center_y))

        # 이벤트 기반 AGV object 추적
        self.agvs_in_intersection = set()  # 교차로 내 AGV만 추적
        self.agvs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_agv = None
        self.is_deadlock = False
        self.macro = None

        self.ingoing = {"N": [], "E": [], "S": [], "W": []}
        self.outgoing = {"N": [], "E": [], "S": [], "W": []}

        self._plan_moves: Dict[int, tuple] = {}   # agv_id -> move (dx,dy)
        self._plan_prio:  Dict[int, int]   = {}   # agv_id -> priority(큰 게 먼저)
        self._plan_order: Dict[int, tuple] = {}   # agv_id -> (order tuple)

        self.macro = None  # 할당된 매크로 액션 (없으면 None)

        self.last_dst_for_reward = None

    def reset(self):
        self.agvs_in_intersection.clear()
        self.agvs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_agv = None
        self.ingoing = {"N": [], "E": [], "S": [], "W": []}
        self.outgoing = {"N": [], "E": [], "S": [], "W": []}
        self.is_deadlock = False
        self._plan_moves.clear()
        self._plan_prio.clear()
        self._plan_order.clear()

        self.macro = None
        self.last_dst_for_reward = None

    def soft_reset(self):
        self.agvs_in_intersection.clear()
        self.agvs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_agv = None
        self.ingoing = {"N": [], "E": [], "S": [], "W": []}
        self.outgoing = {"N": [], "E": [], "S": [], "W": []}
        self.is_deadlock = False
        self._plan_moves.clear()
        self._plan_prio.clear()
        self._plan_order.clear()


    def add_agv(self, agv_object):
        agv_obj = agv_object
        nxt = self.controller.next_buffer[agv_obj.id]
        self.agvs_in_intersection.add(agv_obj)

        if agv_obj.pos == (self.center_x, self.center_y):
            self.center_agv = agv_obj
        else:        
            cx, cy = self.center_x, self.center_y
            for direction, coords in self.lane_coords.items():
                if agv_obj.pos in coords:
                    self.agvs_in_lanes[direction].append(agv_obj)
                    curd = (np.sign(agv_obj.pos[0] - cx), np.sign(agv_obj.pos[1] - cy))

                    if curd == nxt:
                        self.outgoing[direction].append(agv_obj)
                    elif curd == (-nxt[0], -nxt[1]):
                        self.ingoing[direction].append(agv_obj)
                    break

    def get_state(self):
        state_vector = []

        closest_cfg = {
            'N': ('y',  max),  # y가 가장 큰(아래쪽)
            'E': ('x',  min),  # x가 가장 작은(왼쪽)
            'S': ('y',  min),  # y가 가장 작은(위쪽)
            'W': ('x',  max),  # x가 가장 큰(오른쪽)
        }
        for d in ['N', 'E', 'S', 'W']:
            if d not in self.present_dirs:
                state_vector.extend([0, 0, 0, 0, 0, 1])  # goal_onehot(4), ingoing(1), density(1)
                continue

            agvs = self.agvs_in_lanes[d]  # [AGV,...]
            goal_onehot = [0, 0, 0, 0]

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

            ingoing = 1.0 if bool(self.ingoing[d]) else 0.0

            density = len(agvs) / len(self.lane_coords[d]) if self.lane_coords[d] else 1.0

            state_vector.extend(goal_onehot)
            state_vector.append(ingoing)
            state_vector.append(density)

        return np.array(state_vector, dtype=np.float32)
    
    def action_control(self, action: int):
        """0~11 int 액션을 받아 매크로 시작 + tick 진행"""        
        if action is not None:
            src, dst = decode_action(action)  # 0~11 -> (src,dst)
            self.macro = {"src": src, "dst": dst, "phase": "pull"}     
            self.last_dst_for_reward = dst        
        self.tick_macro()

    def tick_macro(self):
        if self.macro is None:
            return
        if not self.is_deadlock:
            self.macro = None
            return

        src = self.macro['src']; dst = self.macro['dst']
        src_dir = IDX2DIR[src]; dst_dir = IDX2DIR[dst]
        src_str = IDX2DIRC[src]; dst_str = IDX2DIRC[dst]
        front_cell = (self.center_x + src_dir[0], self.center_y + src_dir[1])

        phase = self.macro['phase']

        if phase == "pull":
            # 센터가 아직 비어있으면 계속 '풀'만 시도하고 push로 넘기지 않음
            if self.center_agv is None:
                for agv in self.agvs_in_lanes.get(src_str, []):
                    if agv.pos == front_cell:
                        self._plan_add(agv.id, (-src_dir[0], -src_dir[1]), PR_PULL_CENTER, (0, 0))
                        break
                return
            else:
                # 직전 스텝에서 실제로 센터가 채워졌음을 '관측'했을 때만 push로 전환
                self.macro['phase'] = "push"

        if self.macro['phase'] == "push":
            # 센터가 비어 있으면 지난 스텝에서 push가 실제로 끝난 것 → 매크로 종료
            if self.center_agv is None:
                self.macro = None
                return
            # 센터가 있으면 체인 밀기 계속 계획
            self._plan_push_chain(dst_str)
            self._plan_add(self.center_agv.id, (dst_dir[0], dst_dir[1]), PR_ACTION, (2, 0))

    def calculate_action_mask(self):
        # macro 진행 중이면 액션 불가
        if self.macro is not None:
            return np.stack([np.zeros(4, dtype=bool),
                            np.zeros(4, dtype=bool)])

        mask = np.zeros(12, dtype=bool)
        mask_src = np.zeros(4, dtype=bool)
        mask_dst = np.zeros(4, dtype=bool)

        # macro 없음 + 데드락 발생 시, 가능한 팔 방향만 True
        for d, idx in DIR2IDX.items():
            if d not in self.present_dirs:
                continue
            cap = len(self.lane_coords.get(d, []))
            occ = len(self.agvs_in_lanes.get(d, []))
            if occ < cap:
                mask_dst[idx] = True

            ingoing_count = len(self.ingoing.get(d, []))
            if ingoing_count > 0:
                mask_src[idx] = True

        for i, (s, d) in enumerate(PAIRS):
            mask[i] = mask_src[s] and mask_dst[d]
        
        return mask

    def check_deadlock(self):
        """
        데드락 판정:
        - is_deadlock    : 교차로 내 '어떤' 쌍이라도 스와핑 데드락이면 True
        """
        agvs = list(self.agvs_in_intersection or [])
        if len(agvs) < 2:
            self.is_deadlock = False
            return False

        center_id = getattr(self.center_agv, "id", None)

        # (선택) 센터 먼저 검사되도록 앞으로 배치
        if center_id is not None:
            agvs.sort(key=lambda a: a.id != center_id)  # center가 맨 앞

        n = len(agvs)
        for i in range(n - 1):
            ai = agvs[i]
            for j in range(i + 1, n):
                aj = agvs[j]

                # [수정] 즉각적인 스와핑과 경로 기반 스와핑을 모두 검사
                is_immediate_swap = self._check_immediate_swap(ai, aj)
                is_path_conflict = self._check_swapping_path(ai, aj) or self._check_swapping_path(aj, ai)

                if is_immediate_swap or is_path_conflict:
                    self.is_deadlock = True                    
                    return True # 데드락 발견 시 즉시 종료

        # 루프를 모두 돌았는데 데드락이 없으면 상태 초기화
        self.is_deadlock = False
        return False

    def _check_swapping_path(self, agv1, agv2):
        """
        A(agv1)의 경로 상에 B(agv2)의 현재 위치가 포함되어 있고,
        [수정] A의 '다음 위치'부터 B의 '이전 위치'까지의 경로 구간 역순이 B의 경로에 서브시퀀스로 포함되면 스와핑 위험으로 판단.
        """
        path1 = self.controller.agv_path.get(agv1.id)
        path2 = self.controller.agv_path.get(agv2.id)
        if not path1 or not path2:
            return False

        pos2 = agv2.pos

        # A의 경로에서 B의 현재 위치 인덱스 찾기
        try:
            index2_in_1 = path1.index(pos2)
        except ValueError:
            return False

        # A의 다음 위치(인덱스 1)부터 B의 이전 위치(인덱스 index2_in_1 - 1)까지
        # 경로 구간이 존재하려면, 최소한 A -> A+1 -> B 순서여야 함 (index2_in_1 >= 2)
        if index2_in_1 < 2:
            return False

        # [수정] A의 경로 구간을 'A+1'부터 'B-1'까지로 변경
        sub_path1 = path1[1:index2_in_1]
        if not sub_path1:
            return False
        reversed_sub_path1 = sub_path1[::-1]

        L = len(reversed_sub_path1)
        if len(path2) < L:
            return False
            
        for i in range(len(path2) - L + 1):
            if path2[i:i + L] == reversed_sub_path1:
                return True
        return False
    
    def _check_immediate_swap(self, agv1, agv2):
        """
        [추가된 함수]
        A의 다음 위치가 B의 현재 위치이고, B의 다음 위치가 A의 현재 위치인지 확인.
        """
        # _planned_move를 사용하여 다음 이동 벡터를 가져옴
        move1 = self._planned_move(agv1)
        move2 = self._planned_move(agv2)

        if move1 is None or move2 is None:
            return False

        pos1 = agv1.pos
        pos2 = agv2.pos

        next_pos1 = (pos1[0] + move1[0], pos1[1] + move1[1])
        next_pos2 = (pos2[0] + move2[0], pos2[1] + move2[1])

        # 스와핑 조건 확인
        if next_pos1 == pos2 and next_pos2 == pos1:
            return True
        
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

    def _planned_move(self, agv):
        """제어버퍼 > 경로기반으로 다음 이동 벡터 추정"""
        mv = self.controller.next_buffer.get(agv.id)
        if mv is not None:
            return mv
        path = self.controller.agv_path.get(agv.id)
        if not path: return None
        try:
            i = path.index(agv.pos)
            if i + 1 < len(path):
                nxt = path[i + 1]
                return (nxt[0] - agv.pos[0], nxt[1] - agv.pos[1])
        except ValueError:
            pass
        return None

    def resolve_arm_swaps_all(self):
        present = getattr(self, "present_dirs", set(self.lane_coords.keys()))
        cx, cy = self.center_x, self.center_y

        for d in present:
            agvs = self.agvs_in_lanes.get(d, [])
            if not agvs:
                continue

            ingoing_agvs = self.ingoing.get(d, [])
            if not ingoing_agvs:
                continue

            # 가장 먼 ingoing 객체 기준
            farthest_ingoing = max(ingoing_agvs, key=lambda a: abs(a.pos[0]-cx)+abs(a.pos[1]-cy))
            ref_distance = abs(farthest_ingoing.pos[0]-cx) + abs(farthest_ingoing.pos[1]-cy)

            # 기준 거리 안쪽에 있는 AGV만 선택
            agvs_to_pull = [agv for agv in agvs if abs(agv.pos[0]-cx)+abs(agv.pos[1]-cy) <= ref_distance]
            # 중심 가까운 순서로 정렬
            agvs_to_pull.sort(key=lambda a: abs(a.pos[0]-cx)+abs(a.pos[1]-cy))

            # 교차로 중심 바로 앞 칸 계산
            out_dir = self._dir_vec(d)        # 팔 바깥 방향
            in_dir = (-out_dir[0], -out_dir[1])  # 팔 안쪽 방향
            center_front = (cx + out_dir[0], cy + out_dir[1])  # 교차로 중심 바로 앞 칸

            for k, agv in enumerate(agvs_to_pull):
                order_key = (k, 0)

                # AGV가 이미 중심 앞 칸이면 당기지 않음
                if (agv.pos[0], agv.pos[1]) == center_front:
                    self._plan_add(agv.id, (0,0), PR_PULL, (k, 0))
                    continue

                self._plan_add(agv.id, in_dir, PR_PULL, order_key)


    # Intersection에 유틸 3개 추가
    def begin_plan(self):
        self._plan_moves.clear()
        self._plan_prio.clear()
        self._plan_order.clear()

    def _plan_add(self, agv_id: int, move: tuple, prio: int, order_key: tuple):
        """개별 AGV 이동 계획을 추가/갱신(prio 큰 쪽 우선)"""
        prev = self._plan_prio.get(agv_id, -10**9)
        if (prio > prev) or (prio == prev and agv_id not in self._plan_moves):
            self._plan_moves[agv_id] = move
            self._plan_prio[agv_id] = prio
            self._plan_order[agv_id] = (prio, *order_key)

    def finalize_plan(self):
        """수집된 계획을 control_buffer/push_sequence로 커밋"""
        # 1) control_buffer 채우기
        for agv_id, mv in self._plan_moves.items():
            self.controller.control_buffer[agv_id] = mv
        # 2) 우선순위 정렬: prio 내림차순 → order_key 오름차순 → agv_id
        items = []
        for agv_id, (prio, *order) in self._plan_order.items():
            items.append(( -prio, tuple(order), agv_id ))  # prio 큰 게 먼저이므로 음수
        items.sort()
        seq = []
        seen = set()
        for _, _, aid in items:
            if aid not in seen:
                seen.add(aid)
                seq.append(aid)
        self.controller.push_sequence = seq

    def _plan_push_chain(self, d: str):
        """센터에서 d 방향으로 밀어내기: tail→...→head→center 순으로 한 칸"""
        chain = self._collect_chain_near_to_far(d)  # [head,...,tail]
        if self.center_agv is None:
            return
        dx, dy = self._dir_vec(d)
        move = (dx, dy)

        PR_PUSH = 100  # 끌어오기보다 높은 우선순위

        # tail -> ... -> head 순으로 order_key를 작게
        for k, agv_id in enumerate(reversed(chain)):
            # tail이 k=0이 되도록
            order_key = (0, k)  # 같은 팔 내 상대순서만 있으면 충분
            self._plan_add(agv_id, move, PR_PUSH, order_key)

        # center는 마지막에
        center_order = (1, 0)
        self._plan_add(self.center_agv.id, move, PR_PUSH, center_order)
