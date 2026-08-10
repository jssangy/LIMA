from itertools import chain

from utils.sch import schedule


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

        self.scheduling_capacity = 0                               # 스케줄링이 가능한 최대 AMR 수
        self.available_count = 0                                   # 교차로 내 AMR 여유 공간 개수
        self.neighbor_available_count = {}                         # {N: 15, E: 15, S: 15, W: 15} 인접 교차로별 여유 공간 개수
        self.stack_quota = []                                      # [15, 15, 15, 15] 방향별 스택 할당량


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

        # 정방향으로 탐색
        for i in range(amr.path_cursor, len(path) - 1):
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

        # --- 추가: target_exits를 register_amr 시점에 바로 세팅 ---
        tip_cell = None
        if exit_arm_direction in self.lane_coords:
            coords = self.lane_coords[exit_arm_direction]
            if coords:
                tip_cell = coords[-1]  # 해당 팔의 맨 끝 tip
        self.target_exits[amr.id] = tip_cell

    def check_deadlock(self):
        if self.check_cycle_deadlock():
            return True
        if self.check_center_deadlock():
            return True
        if self.check_swap_deadlock():
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
    

    def check_swap_deadlock(self):
        for rec1 in self.amr_intent_map.values():
            cur1 = rec1.get('current_arm')
            nxt1 = rec1.get('exit_arm')
            amr1_pos = rec1.get('amr_obj').pos
            amr1_next_pos = rec1.get('amr_obj').next_pos

            for rec2 in self.amr_intent_map.values():
                if rec1 == rec2:
                    continue
                cur2 = rec2.get('current_arm')
                nxt2 = rec2.get('exit_arm')
                amr2_pos = rec2.get('amr_obj').pos
                amr2_next_pos = rec2.get('amr_obj').next_pos
                rec2_tip_cell = self.target_exits.get(rec2.get('amr_obj').id)

                if (amr1_pos == rec2_tip_cell and cur1 != nxt1 and cur2 == nxt2) \
                    or (amr1_pos == amr2_next_pos and amr2_pos == amr1_next_pos):
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

        # 2) 센터 AMR 배치(front 삽입)  ✅ 수정 버전
        center_id = pos2aid.get(center, None)
        if center_id is not None:
            center_rec = self.amr_intent_map.get(center_id, {})
            exit_dir = center_rec.get('exit_arm')

            def occ(d): 
                return sum(1 for x in target_lanes[d] if x is not None)

            # ★ 빈자리 있는 팔만 후보로
            counts = {d: occ(d) for d in dirs}
            cands = [d for d in dirs if counts[d] < len(target_lanes[d])]

            host = None

            # 1) exit_dir에 빈자리 있으면 최우선
            if exit_dir in dirs and exit_dir in cands:
                host = exit_dir

            # 2) 아니면 "빈자리 있는 팔들(cands)" 중 점유 최소를 고름 (NESW 타이브레이크)
            elif cands:
                min_count = min(counts[d] for d in cands)
                for d in dirs:  # dirs 순서가 tie-break
                    if d in cands and counts[d] == min_count:
                        host = d
                        break

            # 3) host가 있으면 front로 편입(우측 쉬프트)
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
        current_stacks, targets = self.build_stacks_from_snapshot()
        dirs = list(self.dirs)
        n = len(dirs)
        dir_to_idx = {d: i for i, d in enumerate(dirs)}

        solver_input_stacks = []
        for d in dirs:
            stack_content = []
            for aid in current_stacks[d]:
                target_dir = targets.get(aid)
                if target_dir not in dir_to_idx:
                    target_dir = d
                stack_content.append(dir_to_idx[target_dir])
            solver_input_stacks.append(stack_content)

        lane_caps = [len(self.lane_coords[d]) for d in dirs]

        actions, elapsed_time, wb, hit = schedule(
            initial_stacks=solver_input_stacks,
            stack_capacities=lane_caps,
            per_stack_quota=self.stack_quota,
            order=list(range(n)),
            cache_db_path=getattr(self, "cache_db_path", None),  # 워커는 read-only
            max_iters=1_000_000,
        )

        # 메인에 넘길 writeback만 저장(Intersection는 DB 접근 안 함)
        self.cache_writeback = wb
        self.cache_hit = hit

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

        return self.paths, self.target_exits, getattr(self, 'cache_writeback', None), getattr(self, 'cache_hit', False)


    def _record_snapshot(self, inter_sim, center_amr_id):
        """
        현재 inter_sim, center AMR 정보를 바탕으로
        모든 AMR의 self.paths에 현재 좌표를 추가 기록
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