import numpy as np
from itertools import chain

class Intersection:
    def __init__(self, intersection_data, controller_ref):
        self.id = intersection_data
        self.center_x, self.center_y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
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
        중앙 AMR의 행동 마스크 계산:
        - 각 lane을 스캔해 '중앙으로 들어오려는(inbound)' AMR이 하나라도 있으면 그 방향 차단
        - 모두 막히면 예외: 뒤로가기를 제외한 나머지 방향 다시 허용
        """
        import numpy as np

        is_push_out = False

        # 중앙에 AMR이 없으면 아무 행동도 불가
        if self.center_agv is None:
            return np.zeros(4, dtype=np.bool_), is_push_out

        dirs = ['N', 'E', 'S', 'W']
        mask = np.ones(4, dtype=np.bool_)

        center = (self.center_x, self.center_y)
        def manhattan(p): return abs(p[0] - center[0]) + abs(p[1] - center[1])

        # lane별 inbound 여부 탐지
        inbound = {d: False for d in dirs}
        for d in dirs:
            lane_agvs = self.agvs_in_lanes.get(d, [])
            if not lane_agvs:
                continue

            for agv in lane_agvs:
                # 1) 컨트롤러에 예약된 이동 신호가 있으면 그것으로
                move = self.controller.control_buffer.get(agv.id)

                # 2) 없으면 경로 기반 다음 스텝으로 추정
                if move is None:
                    path = self.controller.agv_path.get(agv.id)
                    if path:
                        try:
                            idx = path.index(agv.pos)
                            if idx + 1 < len(path):
                                nxt = path[idx + 1]
                                move = (nxt[0] - agv.pos[0], nxt[1] - agv.pos[1])
                        except ValueError:
                            move = None

                # 3) 여전히 불명확하면 inbound로 간주하지 않음(보수적)
                if move is None:
                    continue

                curd = manhattan(agv.pos)
                nxtp = (agv.pos[0] + move[0], agv.pos[1] + move[1])
                nxtd = manhattan(nxtp)

                if nxtd < curd:  # 중앙과의 거리가 줄어드는 이동 = inbound
                    inbound[d] = True
                    break  # 이 방향은 차단 결정 완료

        # inbound 있는 방향은 차단
        for i, d in enumerate(dirs):
            if inbound[d]:
                mask[i] = False

        # 뒤로가기 금지
        back_idx = self._back_action_index_from_prev()
        if back_idx is not None:
            mask[back_idx] = False

        # 모두 막혔으면 예외 처리: 뒤로가기를 제외하고 전부 다시 허용
        if not mask.any():
            is_push_out = True
            mask[:] = True
            if back_idx is not None:
                mask[back_idx] = False

        return mask, is_push_out

    def _dir_vec(self, d: str):
        return {'N': (0,-1), 'E': (1,0), 'S': (0,1), 'W': (-1,0)}[d]

    def _collect_chain_near_to_far(self, d: str):
        """팔 d에서 센터에 가까운 칸부터 '연속 점유'된 AGV id 목록(near→far)"""
        cells = self.lane_coords.get(d, [])
        pos2id = {a.pos: a.id for a in self.agvs_in_lanes.get(d, [])}
        chain = []
        for p in cells:
            if p in pos2id: chain.append(pos2id[p])
            else: break
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
        """
        모든 팔을 스캔해 스와핑 발생 팔에 대해
        체인을 '센터 방향으로 1칸' 이동하도록 control_buffer/push_sequence 세팅.
        - 팔 내부 순서: 헤드(가까운) → ... → 테일
        - 전팔 글로벌 순서: 센터까지 맨해튼 거리 오름차순(동률은 N,E,S,W, chain idx, id)
        - 센터 점유 여부와 무관하게 계획만 주입 (실제 이동 가능성은 _is_valid_move가 필터)
        """
        dirs = ['N', 'E', 'S', 'W']
        dir_rank = {'N':0, 'E':1, 'S':2, 'W':3}
        cx, cy = self.center_x, self.center_y

        # 스와핑 감지된 팔 수집
        hit = [d for d in dirs if self._detect_arm_swap_pairs(d)]
        if not hit:
            return

        order = []  # (dist, dir_rank, k, agv_id)
        for d in hit:
            chain = self._collect_chain_near_to_far(d)  # [head,...,tail]
            if not chain:
                continue
            dx, dy = self._dir_vec(d)
            move_in = (-dx, -dy)  # 센터 방향

            # 체인 전원 센터 쪽 1칸 이동 지시
            for k, agv_id in enumerate(chain):
                self.controller.control_buffer[agv_id] = move_in
                pos = self.controller.agv_pos.get(agv_id)
                if pos is None:
                    continue
                dist = abs(pos[0] - cx) + abs(pos[1] - cy)
                order.append((dist, dir_rank[d], k, agv_id))

        # 글로벌 우선순위: 센터 가까운 순으로 이동
        order.sort()
        seq = [agv_id for (_, _, _, agv_id) in order]

        # 기존 push_sequence와 병합(우리가 앞, 중복 제거)
        prev = getattr(self.controller, "push_sequence", [])
        merged = [x for x in seq if x not in prev] + prev
        self.controller.push_sequence = merged
