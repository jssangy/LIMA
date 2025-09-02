import numpy as np
from itertools import chain
from typing import Dict

DIR2IDX = {"N": 0, "E": 1, "S": 2, "W": 3}

class Intersection:
    def __init__(self, intersection_data, controller_ref):
        self.center_x, self.center_y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
        self.id = f'x{self.center_x}y{self.center_y}'
        self.controller = controller_ref
        self.map = self.controller.map

        self.lane_coords = {
            'N': [(self.center_x, self.center_y - i) for i in range(1, self.len_N + 1)],
            'E': [(self.center_x + i, self.center_y) for i in range(1, self.len_E + 1)],
            'S': [(self.center_x, self.center_y + i) for i in range(1, self.len_S + 1)],
            'W': [(self.center_x - i, self.center_y) for i in range(1, self.len_W + 1)]
        }

        self.all_lane_coords = set(chain.from_iterable(self.lane_coords.values()))
        self.all_lane_coords.add((self.center_x, self.center_y))

        # 이벤트 기반 AGV object 추적
        self.agvs_in_intersection = set()  # 교차로 내 AGV만 추적
        self.agvs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_agv = None
        self.is_deadlock = False

        self.ingoing = {"N": False, "E": False, "S": False, "W": False}
        self.outgoing = {"N": False, "E": False, "S": False, "W": False}

        self._plan_moves: Dict[int, tuple] = {}   # agv_id -> move (dx,dy)
        self._plan_prio:  Dict[int, int]   = {}   # agv_id -> priority(큰 게 먼저)
        self._plan_order: Dict[int, tuple] = {}   # agv_id -> (order tuple)

    def reset(self):
        self.agvs_in_intersection.clear()
        self.agvs_in_lanes = {'N': [], 'E': [], 'S': [], 'W': []}
        self.center_agv = None
        self.ingoing = {"N": False, "E": False, "S": False, "W": False}
        self.outgoing = {"N": False, "E": False, "S": False, "W": False}
        self.is_deadlock = False

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

    def action_control(self, actions):
        if self.center_agv is None:
            return
        move_map = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
        dir_map  = {0:'N', 1:'E', 2:'S', 3:'W'}
        a = int(actions)

        # 평소처럼 center 이동 의도만 기록(최종 커밋은 finalize_plan에서)
        self._plan_add(self.center_agv.id, move_map[a], prio=90, order_key=(2,0))  # 기본 이동(푸시 아님)

        d = dir_map[a]
        self._plan_push_chain(d)  # ★ 체인 이동 계획만 추가 (버퍼 직접 X)

    def check_deadlock(self):
        """
        '중앙 AMR(center_agv)이 관련된' 데드락만 True로 반환.
        중앙 AMR이 없거나, 교차로 내 AGV가 2대 미만이면 False.
        """
        center = self.center_agv
        if center is None:
            self.is_deadlock = False
            return False

        agvs_inside = self.agvs_in_intersection
        if len(agvs_inside) < 2:
            self.is_deadlock = False
            return False

        # 중앙 AMR과 나머지 AGV들만 검사
        for other in agvs_inside:
            if other is center:
                continue
            # 양방향(중앙→상대, 상대→중앙) 모두 확인
            if (self._check_swapping_path(center, other) or
                self._check_swapping_path(other, center)):
                self.is_deadlock = True
                return True

        self.is_deadlock = False
        return False

    def _check_swapping_path(self, agv1, agv2):
        """
        A(agv1)의 경로 상에 B(agv2)의 현재 위치가 포함되어 있고,
        A의 해당 구간 역순이 B의 경로에 서브시퀀스로 포함되면 스와핑 위험으로 판단.
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

        # A의 경로 구간을 뒤집고, B의 경로에 포함되는지 확인
        sub_path1 = path1[:index2_in_1 + 1]
        if not sub_path1:
            return False
        reversed_sub_path1 = sub_path1[::-1]

        L = len(reversed_sub_path1)
        for i in range(len(path2) - L + 1):
            if path2[i:i + L] == reversed_sub_path1:
                return True
        return False

    def push_out(self, pos, direction):
        # direction: (dx, dy)
        dx, dy = direction
        next_pos = (pos[0] + dx, pos[1] + dy)

        H, W = self.map.shape[0], self.map.shape[1]
        def in_bounds(p): return 0 <= p[0] < W and 0 <= p[1] < H

        # 1) 다음 칸이 맵 밖이면 실패(더 밀 곳 없음)
        if not in_bounds(next_pos):
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
        # 중앙 AMR 없으면 전부 금지
        if self.center_agv is None:
            return np.zeros(4, dtype=np.bool_)

        mask = np.ones(4, dtype=np.bool_)  # N E S W

        # 1) 뒤로가기 금지 (기존 로직)
        back_idx = self._back_action_index_from_prev()
        if back_idx is not None and 0 <= back_idx < 4:
            mask[back_idx] = False

        # 2) 이웃 락다운 방향 금지
        #    env.lockdown_on_deadlock == True 이고, 이웃.is_deadlock == True 면 해당 방향 False
        env = getattr(self, "env", None) or getattr(self.controller, "env", None)
        lockdown_on = True if env is None else bool(getattr(env, "lockdown_on_deadlock", True))
        if lockdown_on:
            for d, idx in DIR2IDX.items():
                if mask[idx] and self._neighbor_lockdown_active(d):
                    mask[idx] = False

        # 3) (선택) ingoing/blocked_dirs 등 다른 금지 조건도 같이 적용하고 싶으면 여기서 AND 처리
        # 예) self.ingoing 이 있으면 해당 방향 금지
        ingo = getattr(self, "ingoing", None)
        if ingo:
            if isinstance(ingo, str):
                i = DIR2IDX.get(ingo)
                if i is not None: mask[i] = False
            else:
                for dd in ingo:
                    i = DIR2IDX.get(dd)
                    if i is not None: mask[i] = False

        # 4) 안전장치: 전부 False면 back 제외 첫 방향 하나 살려둠(원치 않으면 제거)
        if not mask.any():
            for i in range(4):
                if i != back_idx:
                    mask[i] = True
                    break

        return mask
    
    def _neighbor_lockdown_active(self, d: str) -> bool:
        """
        방향 d 이웃 교차로가 '락다운(데드락 정책 활성)'이면 True.
        env.use_rl 여부를 묶고 싶지 않다면 is_deadlock만 보면 됨.
        """
        nb = self._get_neighbor_intersection(d)
        if nb is None:
            return False

        # 락다운 조건: 이웃이 데드락 상태
        # (원하면 RL 사용 중일 때만 막으려면 and getattr(self.env, "use_rl", False) 추가)
        return bool(getattr(nb, "is_deadlock", False))

    def _get_neighbor_intersection(self, d: str):
        """
        d 방향 이웃 Intersection 객체 반환.
        self.neighbors_by_dir 가 {'N': iid or Intersection, ...} 형태라고 가정.
        """
        nb_map = getattr(self, "neighbors_by_dir", None) or getattr(self, "neighbor_by_dir", None)
        if not isinstance(nb_map, dict):
            return None

        ref = nb_map.get(d)
        if ref is None:
            return None

        # 이미 Intersection 객체면 그대로
        if hasattr(ref, "center_agv") and hasattr(ref, "is_deadlock"):
            return ref

        # iid 문자열이면 env에서 찾아오기
        env = getattr(self, "env", None) or getattr(self.controller, "env", None)
        if env is not None and hasattr(env, "intersections"):
            return env.intersections.get(ref)
        return None




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
        mv = self.controller.control_buffer.get(agv.id)
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
        # print("Swapping Detected")

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


    # Intersection에 유틸 3개 추가
    def begin_plan(self):
        """이번 스텝의 이동 계획을 수집하기 전에 호출"""
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
