import random
from itertools import chain
from copy import deepcopy

from utils.sch import schedule
from utils.env import StackRearrangementEnv


class Intersection:
    def __init__(self, intersection_data, present_dirs):
        self.center_x, self.center_y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
        self.id = f'x{self.center_x}y{self.center_y}'

        if present_dirs is None:
            present_dirs = {d for d,L in zip("NESW",[self.len_N,self.len_E,self.len_S,self.len_W]) if L>0}
        self.present_dirs = set(present_dirs)
        self.dirs = [d for d in "NESW" if d in self.present_dirs]

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
        self.amr_intent_map = {}            # {amr_id: {'amr_obj': amr, 'current_arm': 'N', 'exit_arm': 'S'}}
        self.is_deadlock = False
        self.paths = {}                 # {amr_id: [(x,y), ...]}
        self.target_exits = {}          # {amr_id: (x,y)}  # 각 AMR의 "원래" 출구 tip 좌표


    def reset(self):
        self.amr_intent_map = {}
        self.is_deadlock = False
        self.paths = {}
        self.target_exits = {}

    
    def register_amr(self, amr):
        path = amr.path
        if not path:
            raise ValueError("AMR path is empty; cannot register intent.")

        center = (self.center_x, self.center_y)
        current_arm_direction = None
        exit_arm_direction = None
        exit_cell = None

        for i in range(len(path)-1, -1, -1):
            if path[i] == center:
                next_pos_index = i + 1
                exit_cell = path[next_pos_index]
                break

        for direction, coords in self.lane_coords.items():
            if amr.pos == center:
                current_arm_direction = "C"
            if amr.pos in coords:
                current_arm_direction = direction
            if exit_cell is not None and exit_cell in coords:
                exit_arm_direction = direction

        if exit_cell is None:
            exit_arm_direction = current_arm_direction

        if current_arm_direction is None or exit_arm_direction is None:
            raise ValueError("Could not determine current or exit arm direction for AMR.")

        self.amr_intent_map[amr.id] = {
            'amr_obj': amr,
            'current_arm': current_arm_direction,
            'exit_arm': exit_arm_direction
        }

    def check_deadlock(self):
        if self.check_cycle_deadlock():
            return True
        if self.check_center_deadlock():
            return True
        return False


    def check_cycle_deadlock(self):
        dirs = self.dirs
        adj = {d: set() for d in dirs}

        pos2rec = {}
        for rec in self.amr_intent_map.values():
            a = rec.get('amr_obj')
            if a: pos2rec[a.pos] = rec

            cur = rec['current_arm']
            nxt = rec['exit_arm']
            if cur in adj and nxt in adj and cur != nxt:
                adj[cur].add(nxt)

        visited = set()
        onstack = set()

        def dfs(u: str) -> bool:
            visited.add(u)
            onstack.add(u)
            for v in adj[u]:
                if v not in visited:
                    if dfs(v):
                        return True
                elif v in onstack:
                    return True
            onstack.remove(u)
            return False

        for u in adj.keys():
            if u not in visited and dfs(u):
                self.is_deadlock = True
                return True
        
        inline_conflict = False
        for d in dirs:
            coords = self.lane_coords.get(d, [])
            if not coords:
                continue
            # 인접 쌍 스캔
            for i in range(len(coords) - 1):
                front_pos = coords[i]       # center에 더 가까운 칸 (rank = i+1)
                behind_pos = coords[i + 1]  # 그 바로 뒤칸 (rank = i+2)

                rec_front = pos2rec.get(front_pos)
                rec_behind = pos2rec.get(behind_pos)
                if not rec_front or not rec_behind:
                    continue

                # 둘 다 같은 레인 d에 있어야 함
                if rec_front.get('current_arm') != d or rec_behind.get('current_arm') != d:
                    continue

                nxt_front  = rec_front.get('exit_arm')
                nxt_behind = rec_behind.get('exit_arm')

                # 앞칸이 '바깥쪽', 뒤칸이 '안쪽(센터)' 의도면 서로 충돌
                if nxt_front == d and nxt_behind != d:
                    inline_conflict = True
                    break
            if inline_conflict:
                break

        self.is_deadlock = inline_conflict
        return self.is_deadlock

    
    def check_center_deadlock(self):
        """
        중앙에 있는 AMR 기준 데드락 탐지.

        1) current_arm == 'C' 인 AMR이 없으면 → 데드락 아님
        2) 있으면 해당 AMR의 exit_arm = exit_dir
        3) 다시 amr_intent_map을 순회하면서
        - current_arm == exit_dir 인 AMR이 존재하고
        - 그 AMR의 exit_arm != current_arm 이면
            → 중앙 AMR과 충돌 의도가 있으므로 데드락
        """
        # 1. center에 있는 AMR 찾기
        center_exit = None
        for aid, rec in self.amr_intent_map.items():
            if rec.get('current_arm') == 'C':
                center_exit = rec.get('exit_arm')
                break

        # center 없음 또는 출구 방향 이상 → 데드락 아님
        if center_exit not in self.dirs:
            self.is_deadlock = False
            return False

        # 2. 해당 출구 방향 팔 위에서, 교차로로 들어오려는 AMR이 있는지 검사
        for rec in self.amr_intent_map.values():
            cur = rec.get('current_arm')
            nxt = rec.get('exit_arm')

            # 같은 팔(cur == center_exit)에 있고,
            # 자기 출구 방향이 현재 팔과 다르면(=교차로 진입 의도)
            if cur == center_exit and nxt is not None and nxt != cur:
                self.is_deadlock = True
                return True

        self.is_deadlock = False
        return False


    def build_prestage_paths(self):
        """
        데드락 해소용 프리-스테이지 경로 생성/주입.
        각 레인의 near(front)부터 빈칸 없이 압축되고 센터가 비워진 상태를 목표로 한다.
        - 센터에 AMR이 있으면 우선 'exit_arm'의 front로 보낸다
        (여유 없으면 가장 덜 찬 팔(N→E→S→W 타이브레이크)의 front로).
        - 모든 AMR의 경로 길이를 동일하게 맞춘다(정지 AMR은 제자리 좌표를 반복).
        반환: (lanes, target_lanes, paths, max_steps)
        - lanes/target_lanes: {'N':[aid|None,...], ...} (index 0 = near/front)
        - paths: {amr_id: [(x,y), ...]}  (길이 동일)
        - max_steps: 동기화를 위한 최대 액션(틱) 수
        """
        cx, cy = self.center_x, self.center_y
        center = (cx, cy)
        dirs = self.dirs

        # 0) 점유 스냅샷 (near→far)
        lanes = {d: [None] * len(self.lane_coords[d]) for d in dirs}
        pos2aid = {}
        for aid, rec in self.amr_intent_map.items():
            a = rec.get('amr_obj')
            pos2aid[a.pos] = aid

        for d, coords in self.lane_coords.items():
            for i, p in enumerate(coords):  # i=0: near(front)
                lanes[d][i] = pos2aid.get(p, None)
        
        # --- 초기 paths: 현재 pos 1칸 입력 ---
        paths = {}
        for aid, rec in self.amr_intent_map.items():
            a = rec.get('amr_obj')
            paths[aid] = [a.pos]

        # 1) 각 팔 near→far 압축 → target_lanes
        target_lanes = {}
        for d in dirs:
            filled = [aid for aid in lanes[d] if aid is not None]  # 현 순서 유지
            cap = len(lanes[d])
            target_lanes[d] = filled + [None] * (cap - len(filled))

        # 2) 센터 AMR 배치(front 삽입)
        center_id = pos2aid.get(center, None)
        if center_id is not None:
            center_rec = self.amr_intent_map.get(center_id, {})
            exit_dir = center_rec.get('exit_arm')

            def occ(d): return sum(1 for x in target_lanes[d] if x is not None)

            host = None
            if exit_dir in dirs and occ(exit_dir) < len(target_lanes[exit_dir]):
                host = exit_dir
            else:
                counts = {d: occ(d) for d in dirs}
                min_count = min(counts.values()) if counts else 0
                for d in dirs:  # NESW 타이브레이크
                    if counts[d] == min_count and occ(d) < len(target_lanes[d]):
                        host = d
                        break
            if host is not None:
                target_lanes[host] = [center_id] + target_lanes[host][:-1]

        # -------------------------------
        # 3) lanes vs target_lanes → 동기화 경로 생성
        # -------------------------------

        # 현재/목표 인덱스 맵(near index)
        cur_loc = {}   # aid -> (arm, idx)  (센터는 ('C', None))
        for d in dirs:
            for i, aid in enumerate(lanes[d]):
                if aid is not None:
                    cur_loc[aid] = (d, i)
        if center_id is not None:
            cur_loc[center_id] = ('C', None)

        tgt_loc = {}   # aid -> (arm, idx)
        for d in dirs:
            for i, aid in enumerate(target_lanes[d]):
                if aid is not None:
                    tgt_loc[aid] = (d, i)

        # 각 AMR의 필요 스텝 수 계산
        steps = {}
        max_steps = 0
        for aid in paths.keys():
            if aid == center_id and aid in tgt_loc:
                dist = 1  # 센터 → host.front
            elif aid in cur_loc and aid in tgt_loc:
                d0, i0 = cur_loc[aid]
                d1, i1 = tgt_loc[aid]
                if d0 == d1 and d0 in dirs:
                    dist = abs(i1 - i0)  # 같은 팔 내 인덱스 차
                else:
                    # (이 케이스는 거의 없음: 센터 외 팔 변경 없음이 기본 정책)
                    # 안전하게 center 경유의 상한치를 넣고, 아래에서 실제 좌표는 그대로 보존됨.
                    dist = 1 + (i1 if d1 in dirs else 0)
            else:
                dist = 0
            steps[aid] = dist
            if dist > max_steps:
                max_steps = dist

        # per-step 좌표 생성
        for s in range(1, max_steps + 1):
            for aid in paths.keys():
                # 초깃값: 마지막 좌표(정지 패딩)
                last = paths[aid][-1]

                if aid == center_id and aid in tgt_loc:
                    # 센터 AMR: 스텝 1에 host.front로 점프, 이후 고정
                    if s == 1:
                        d1, i1 = tgt_loc[aid]
                        pos = self.lane_coords[d1][0]
                    else:
                        pos = last
                    paths[aid].append(pos)
                    continue

                # 같은 팔 내 이동(한 칸씩)
                if aid in cur_loc and aid in tgt_loc:
                    d0, i0 = cur_loc[aid]
                    d1, i1 = tgt_loc[aid]
                    if d0 == d1 and d0 in dirs:
                        m = steps[aid]
                        if m == 0:
                            paths[aid].append(last)
                            continue
                        # 방향: target 쪽으로 한 칸
                        move_k = min(s, m)
                        # 현재 인덱스 = i0 + sign(i1-i0) * move_k
                        sign = 0
                        if i1 > i0: sign = 1
                        elif i1 < i0: sign = -1
                        idx = i0 + sign * move_k
                        pos = self.lane_coords[d0][idx]
                        paths[aid].append(pos)
                        continue

                # 그 외(정지 패딩)
                paths[aid].append(last)

        self.paths = paths.copy()

        return target_lanes, paths
        
    
    def plan_action(self):
        # 1. 현재 상태 스냅샷 (ID 기반)
        current_stacks, targets = self.build_stacks_from_snapshot()
        dirs = self.dirs

        # 2. 데이터 변환 (AMR ID → Target Index)
        # 방향(N, E, S, W)을 Solver용 인덱스(0, 1, 2, 3)로 변환
        dir_to_idx = {d: i for i, d in enumerate(dirs)}

        # sch.py에 전달할 raw input (리스트의 리스트)
        solver_input_stacks = []
        for d in dirs:
            stack_content = []
            for aid in current_stacks[d]:
                target_dir = targets.get(aid)
                stack_content.append(dir_to_idx[target_dir])
            solver_input_stacks.append(stack_content)

        # 3. StackRearrangementEnv 객체 생성 및 정의
        capacity = max(self.len_N, self.len_E, self.len_S, self.len_W)
        env = StackRearrangementEnv(
            num_stacks=len(dirs),
            stack_capacity=capacity,
            stacks=solver_input_stacks
        )

        # 4. 스케줄러 호출
        actions, elapsed_time = schedule(env.stacks, mode="h2", max_iters=1_000_000)

        return actions


    def build_stacks_from_snapshot(self):
        """
        return:
            stacks  = {'N': [5, 3], 'S': [8], 'W': [2, 7]}
            targets = {5: 'E', 3: 'E', 8: 'N', 2: 'W', 7: 'S'}
        """
        dirs = self.dirs
        target_lanes, paths = self.build_prestage_paths()

        stacks = {d: [] for d in dirs}

        for d in dirs:
            near_to_far = target_lanes.get(d, [])
            ids = [aid for aid in near_to_far if aid is not None]
            for aid in reversed(ids):
                stacks[d].append(aid)
        
        targets = {}
        for aid, rec in self.amr_intent_map.items():
            tgt = rec.get('exit_arm')
            if tgt in stacks:
                targets[aid] = tgt

        return stacks, targets



    def actions_to_paths(self):
        idxs_to_dirs = {i: d for i, d in enumerate(self.dirs)}

        # 1. 이동 계획 획득
        actions = self.plan_action()

        # 2. 시뮬레이션용 초기화 (현재 상태)
        inter_sim, targets = self.build_stacks_from_snapshot()
        center_amr_id = None
        pending_dst = None

        # 3. 액션 순차 실행 및 경로 기록
        for src, dst in actions:
            # [Phase 1] 이전 액션 마무리 (Pending Push)
            if center_amr_id is not None and pending_dst is not None:
                # 1. Center -> Dst Push
                inter_sim[pending_dst].append(center_amr_id)

                # 2. Center 비우기
                center_amr_id = None
                pending_dst = None
            
            # [Phase 2] 현재 액션 수행 (Src -> Center Pull)
            # 1. Source -> Center
            mover_id = inter_sim[idxs_to_dirs[src]].pop()
            center_amr_id = mover_id
            
            # 2. 물리적 위치 기록
            self._record_snapshot(inter_sim, center_amr_id=center_amr_id)

            # 3. 다음을 위해 목적지 저장
            pending_dst = idxs_to_dirs[dst]

        # 4. 루프 종료 후 남은 Center AMR 처리
        if center_amr_id is not None and pending_dst is not None:
            inter_sim[pending_dst].append(center_amr_id)
            self._record_snapshot(inter_sim, center_amr_id=None)

        # 5. 경로 후처리
        # 중앙을 지나지 않는 로봇 삭제 & 중앙 통과 후 경로 절삭
        self._post_process_paths(targets)

        return self.paths, self.target_exits


    def _record_snapshot(self, inter_sim, center_amr_id):
        """
        현재 inter_sim, center AMR 정보를 바탕으로
        모든 AMR의 pself.paths에 현재 좌표를 추가 기록
        """
        center_coord = (self.center_x, self.center_y)

        # 1. Center에 있는 AMR 처리
        if center_amr_id is not None:
            self.paths[center_amr_id].append(center_coord)

        # 2. 각 lane에 있는 AMR 처리
        for d in self.dirs:
            stack = inter_sim[d]
            lane_coords = self.lane_coords[d]

            for i, aid in enumerate(reversed(stack)):
                coords = lane_coords[i]
                self.paths[aid].append(coords)


    def _post_process_paths(self, targets):
        """
        생성된 self.paths를 후처리:
        1. 중앙을 지나지 않는 AMR은 path에서 제거 (단순 대기)
        2. 중앙을 지나는 AMR:
           - 원래 목적지로 가는 경우(Case A):
               '중앙 -> 출구까지만 남기고 절삭
           - 다른 곳으로 대피하는 경우(Case B):
               경로 절삭 없이 끝까지 이동 (확실한 비켜주기)

        추가:
           - self.target_exits[aid] 에
             "원래 나가고 싶어하는 출구 방향 lane의 마지막 셀(tip)" 좌표를 저장
        """
        center = (self.center_x, self.center_y)

        # 딕셔너리 크기 변경 방지를 위해 키 리스트 복사
        for aid in list(self.paths.keys()):
            path = self.paths[aid]

            # 1. 중앙 미경유 AMR 제거
            if center not in path and self.amr_intent_map[aid]['current_arm'] == self.amr_intent_map[aid]['exit_arm']:
                del self.paths[aid]
                continue

            # 2. 마지막으로 중앙에 있었던 시점 찾기 (뒤에서부터 검색)
            last_center_idx = -1
            for i in range(len(path) - 1, -1, -1):
                if path[i] == center:
                    last_center_idx = i
                    break

            # --- 여기서부터 출구 tip 계산 ---

            # 이 AMR이 "원래" 나가고 싶어했던 출구 방향 (register_amr에서 정한 exit_arm)
            intended_target_dir = targets.get(aid)  # 'N', 'E', 'S', 'W' 또는 None

            tip_cell = None
            if intended_target_dir in self.lane_coords:
                coords = self.lane_coords[intended_target_dir]
                if coords:
                    # 출구 방향 lane의 마지막 셀(팔 끝 tip)
                    tip_cell = coords[-1]

            # 출구 tip 좌표 저장
            self.target_exits[aid] = tip_cell

            # --- 이하: 기존 Case A / B 절삭 로직 유지 ---

            # 3. 실제 나가는 방향(Actual Exit) 판별
            #    중앙 바로 다음 칸의 좌표를 확인
            if last_center_idx + 1 < len(path):
                exit_cell = path[last_center_idx + 1]
                actual_exit_dir = None

                for d, coords in self.lane_coords.items():
                    # coords[0]이 보통 교차로와 가장 가까운 칸(Front)
                    if exit_cell in coords:
                        actual_exit_dir = d
                        break

                # 목적지와 실제 나가는 방향 비교
                if (
                    actual_exit_dir is not None
                    and intended_target_dir is not None
                    and actual_exit_dir == intended_target_dir
                ):
                    # Case A: 정상 탈출
                    # → 센터 이후 구간 중, center로부터 가장 먼 칸까지 남기고 그 뒤는 절삭
                    max_idx = last_center_idx
                    max_dist2 = 0
                    cx, cy = center

                    for idx in range(last_center_idx + 1, len(path)):
                        x, y = path[idx]
                        dx = x - cx
                        dy = y - cy
                        dist2 = dx * dx + dy * dy
                        if dist2 > max_dist2:
                            max_dist2 = dist2
                            max_idx = idx

                    # max_idx 이후는 잘라낸다 (max_idx까지 포함)
                    cut_idx = max_idx + 1
                    if cut_idx < len(path):
                        self.paths[aid] = path[:cut_idx]
                else:
                    # Case B: 대피/회피 (Detour) -> 절삭하지 않음
                    # 스케줄러가 계산한 대로 깊숙이 들어가야 함
                    pass
            else:
                # 중앙이 경로의 마지막인 경우 (드물지만 가능) -> 그대로 둠
                pass


    def disperse_paths(self, input_dir: str, num_amrs: int):
        """
        데드락 해소용 분산:
        - 이 교차로에서 input_dir 팔에 있는 AMR들 중 최대 num_amrs개를
          나머지 팔들로 분산시킬 수 있도록 target_lanes를 만든다.
        - overflow 후보는 input_dir 팔의 센터에 가장 가까운 AMR들(near→far 순)이다.
        - 교차로 중앙(center)에 있는 AMR도 적당한 팔로 분산한다.
        - 각 팔은 near→far 압축을 수행한다 (build_prestage_paths와 동일).
        - 실제 시간 방향 경로(self.paths)는 여기서 동기화하지 않고,
          각 AMR의 현재 위치만 1-step으로 기록해 둔다.
        
        반환:
            target_lanes: {'N': [aid|None, ...], ...}
            paths:        {aid: [(x, y)]}  (현재 위치만 포함)
        """

        cx, cy = self.center_x, self.center_y
        center = (cx, cy)
        dirs = self.dirs
        active_dirs = [d for d in dirs if d != input_dir]

        # --------------------------------------------------
        # 0) 현재 스냅샷: lanes (near→far), pos2aid
        # --------------------------------------------------
        lanes = {d: [None] * len(self.lane_coords[d]) for d in dirs}
        pos2aid = {}
        for aid, rec in self.amr_intent_map.items():
            a = rec.get("amr_obj")
            pos2aid[a.pos] = aid

        for d, coords in self.lane_coords.items():
            for i, p in enumerate(coords):  # i=0: center에 가장 가까운 칸
                lanes[d][i] = pos2aid.get(p, None)

        # center AMR (있다면)
        center_id = pos2aid.get(center, None)

        # --------------------------------------------------
        # 1) 초기 paths: 각 AMR의 현재 위치 한 칸만 기록
        #    (동기화 경로는 나중에 별도 로직에서 처리)
        # --------------------------------------------------
        paths = {}
        for aid, rec in self.amr_intent_map.items():
            a = rec.get("amr_obj")
            paths[aid] = [a.pos]

        # --------------------------------------------------
        # 2) target_lanes 기본형: 각 팔을 near→far로 압축
        #    (build_prestage_paths의 1단계와 동일)
        # --------------------------------------------------
        target_lanes = {}
        for d in dirs:
            filled = [aid for aid in lanes[d] if aid is not None]  # 순서 유지
            cap_d = len(lanes[d])
            target_lanes[d] = filled + [None] * (cap_d - len(filled))

        # 현재 점유 수 계산 함수 (target_lanes 기준)
        def occ(d: str) -> int:
            return sum(1 for x in target_lanes[d] if x is not None)

        # --------------------------------------------------
        # 3) overflow 후보 선택:
        #    - input_dir 팔의 front(0번 인덱스)에서부터 num_amrs개까지
        # --------------------------------------------------
        overflow_amrs: list[int] = []
        if input_dir in dirs and num_amrs > 0:
            for aid in target_lanes[input_dir]:
                if aid is not None:
                    overflow_amrs.append(aid)
                    if len(overflow_amrs) >= num_amrs:
                        break

        # --------------------------------------------------
        # 4) input_dir 팔에서 overflow_amrs 제거 후 다시 압축
        #    (해당 팔에서 빼고, 나머지는 near→far로 유지)
        # --------------------------------------------------
        if overflow_amrs:
            remove_set = set(overflow_amrs)
            lane_in = target_lanes[input_dir]
            cap_in = len(lane_in)
            kept = [
                aid for aid in lane_in
                if (aid is not None and aid not in remove_set)
            ]
            target_lanes[input_dir] = kept + [None] * (cap_in - len(kept))

        # --------------------------------------------------
        # 5) host 선택 + front 삽입 로직을 함수로 빼기
        #    (exit_arm 우선, 아니면 가장 덜 찬 팔)
        # --------------------------------------------------
        def assign_to_host_front(aid: int):
            rec = self.amr_intent_map.get(aid, {})
            exit_dir = rec.get("exit_arm")

            host = None
            # (1) exit_dir 우선 (단, input_dir 제외 + 여유 있을 때)
            if (
                exit_dir in active_dirs
                and occ(exit_dir) < len(target_lanes[exit_dir])
            ):
                host = exit_dir
            else:
                # (2) active_dirs 중 가장 덜 찬 팔 선택 (dirs 순서 타이브레이크)
                if active_dirs:
                    counts = {d: occ(d) for d in active_dirs}
                    min_count = min(counts.values())
                    for d in dirs:  # NESW 순서 유지
                        if d not in active_dirs:
                            continue
                        if counts[d] == min_count and occ(d) < len(target_lanes[d]):
                            host = d
                            break

            if host is None:
                # 더 이상 여유가 없으면 이 AMR은 분산 불가 → 스킵
                return

            lane = target_lanes[host]
            # front 삽입: [aid] + lane[:-1]
            #  - target_lanes가 이미 near→far 압축되어 있어서,
            #    occ(host) < len(lane) 이면 lane의 마지막 원소는 None.
            target_lanes[host] = [aid] + lane[:-1]

        # --------------------------------------------------
        # 6) center에 있는 AMR 분산
        # --------------------------------------------------
        if center_id is not None:
            assign_to_host_front(center_id)

        # --------------------------------------------------
        # 7) overflow_amrs들을 active_dirs로 분산
        # --------------------------------------------------
        for aid in overflow_amrs:
            assign_to_host_front(aid)


        # --------------------------------------------------
        # 8) self.paths는 현재 위치 1-step만 가진 상태로 두고,
        #    target_lanes는 env/동기화 로직에서 활용하도록 반환
        # --------------------------------------------------
                # -------------------------------
        # 3) lanes vs target_lanes → 동기화 경로 생성
        # -------------------------------

        center = (self.center_x, self.center_y)

        # 현재/목표 인덱스 맵(near index)
        cur_loc = {}   # aid -> (arm, idx)  (센터는 ('C', None))
        for d in dirs:
            for i, aid in enumerate(lanes[d]):
                if aid is not None:
                    cur_loc[aid] = (d, i)
        if center_id is not None:
            cur_loc[center_id] = ('C', None)

        tgt_loc = {}   # aid -> (arm, idx)
        for d in dirs:
            for i, aid in enumerate(target_lanes[d]):
                if aid is not None:
                    tgt_loc[aid] = (d, i)

        # -------------------------------
        # 8-A) 같은 팔 내 이동(same-arm)과 교차 팔 이동(cross-arm) 분류
        # -------------------------------
        same_arm_amrs = set()
        cross_amrs = []

        for aid in paths.keys():
            if aid == center_id:
                continue
            if aid in cur_loc and aid in tgt_loc:
                d0, i0 = cur_loc[aid]
                d1, i1 = tgt_loc[aid]
                if d0 == d1 and d0 in dirs:
                    same_arm_amrs.add(aid)
                elif (d0 in dirs or d0 == 'C') and d1 in dirs and d0 != d1:
                    # 팔이 바뀌어야 하는 경우(또는 center->팔)
                    cross_amrs.append(aid)

        # -------------------------------
        # 8-B) Phase 1: 같은 팔 내 슬라이딩 (build_prestage_paths 기존 로직)
        # -------------------------------
        steps1 = {}
        max_steps1 = 0

        for aid in paths.keys():
            if aid in same_arm_amrs:
                d0, i0 = cur_loc[aid]
                d1, i1 = tgt_loc[aid]
                dist = abs(i1 - i0)
            elif aid == center_id and aid in tgt_loc:
                # center_id가 있으면, host.front로 한 번에 점프할 수도 있음
                dist = 1
            else:
                dist = 0

            steps1[aid] = dist
            if dist > max_steps1:
                max_steps1 = dist

        # per-step 좌표 생성 (Phase 1)
        for s in range(1, max_steps1 + 1):
            for aid in paths.keys():
                last = paths[aid][-1]

                # center AMR: 스텝 1에 host.front로 점프, 이후 고정
                if aid == center_id and aid in tgt_loc:
                    if s == 1:
                        d1, i1 = tgt_loc[aid]
                        pos = self.lane_coords[d1][0]
                    else:
                        pos = last
                    paths[aid].append(pos)
                    continue

                # 같은 팔 내 이동(한 칸씩)
                if aid in same_arm_amrs:
                    d0, i0 = cur_loc[aid]
                    d1, i1 = tgt_loc[aid]
                    m = steps1[aid]
                    if m == 0:
                        paths[aid].append(last)
                        continue
                    move_k = min(s, m)
                    sign = 0
                    if i1 > i0: sign = 1
                    elif i1 < i0: sign = -1
                    idx = i0 + sign * move_k
                    pos = self.lane_coords[d0][idx]
                    paths[aid].append(pos)
                    continue

                # 그 외(Phase1에서는 정지)
                paths[aid].append(last)

        # Phase 1이 끝난 시점의 step index
        current_T = max_steps1

        # -------------------------------
        # 8-C) Phase 2: cross-arm 이동 (팔이 바뀌는 경우)
        #   - 각 AMR마다
        #       t_center: center로 들어가는 시점
        #       t_target: target 팔의 칸으로 가는 시점
        #   - center에는 한 번에 한 AMR만 들어가도록 slot을 순차 배정
        # -------------------------------
        K = len(cross_amrs)
        slot = {aid: j for j, aid in enumerate(cross_amrs)}  # aid -> 0..K-1
        extra_steps = 2 * K  # 각 AMR당 center, target 두 스텝
        total_T = current_T + extra_steps

        for s in range(current_T + 1, total_T + 1):
            for aid in paths.keys():
                last = paths[aid][-1]

                if aid not in slot:
                    # cross 대상이 아닌 AMR은 그대로 대기
                    paths[aid].append(last)
                    continue

                j = slot[aid]
                t_center = current_T + 2 * j + 1
                t_target = current_T + 2 * j + 2

                if s == t_center:
                    pos = center
                elif s == t_target:
                    d1, i1 = tgt_loc[aid]
                    pos = self.lane_coords[d1][i1]
                else:
                    pos = last

                paths[aid].append(pos)

        self.paths = paths.copy()
        return target_lanes, paths