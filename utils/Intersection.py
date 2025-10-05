import numpy as np
from itertools import chain
from typing import Dict


class Intersection:
    def __init__(self, intersection_data, neighbors_map, present_dirs):
        self.center_x, self.center_y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
        self.id = f'x{self.center_x}y{self.center_y}'
        self.neighbors = neighbors_map

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
        self.amr_intent_map = {}            # {amr_id: {'amr_obj': amr, 'current_arm': 'N', 'exit_arm': 'S'}}
        self.is_deadlock = False


    def reset(self):
        self.amr_intent_map = {}
        self.is_deadlock = False


    def register_amr(self, amr):
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
                current_arm_direction = "C"
            if amr.pos in coords:
                current_arm_direction = direction
            if exit_cell in coords:
                exit_arm_direction = direction

        if current_arm_direction and exit_arm_direction:
            self.amr_intent_map[amr.id] = {
                'amr_obj': amr,
                'current_arm': current_arm_direction,
                'exit_arm': exit_arm_direction
            }


    def check_cycle_deadlock(self):
        dirs = self.present_dirs
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
        # --- 준비 ---
        vec = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
        center = (self.center_x, self.center_y)
        dirs4 = ('N', 'E', 'S', 'W')
        front_cell = {d: (self.center_x + vec[d][0], self.center_y + vec[d][1]) for d in dirs4}

        center_rec = None
        center_amr = None
        exit_dir = None

        # 수집용(단일 순회로 채움)
        exists_ingress = {d: False for d in dirs4}   # cur==d & nxt!=d 가 존재하는지
        front_occupied = {d: False for d in dirs4}   # front 셀 점유 여부
        front_ingressor = {}                         # front에서 센터로 들어오려는 amr (최대 1/방향)

        # --- 1-pass: self.amr_intent_map 단 한 번 순회 ---
        for rec in self.amr_intent_map.values():
            a = rec.get('amr_obj')
            cur = rec.get('current_arm')
            nxt = rec.get('exit_arm')

            if cur == 'C':
                center_rec = rec
                center_amr = a
                exit_dir = nxt
                continue

            if cur in dirs4 and a:
                # cur 방향에서 '센터로 들어오려는(amr)' 존재 여부
                if nxt != cur:
                    exists_ingress[cur] = True

                # front 셀 점유/진입자 기록
                if a.pos == front_cell[cur]:
                    front_occupied[cur] = True
                    if nxt != cur and cur not in front_ingressor:
                        front_ingressor[cur] = a

        # --- 데드락 예측 조건 확인 ---
        if not center_rec or exit_dir not in dirs4:
            return False
        conflict_exists = exists_ingress[exit_dir]
        if not conflict_exists:
            return False

        # 데드락 플래그
        self.is_deadlock = True

        # --- 주입 1: 모든 front 진입자(4방향) 1틱 제자리 대기 [pos,pos]+tail ---
        for d, amr in front_ingressor.items():
            pos = amr.pos
            path = amr.path
            try:
                j = path.index(pos)
                tail = path[j+1:]
            except ValueError:
                tail = path[:]
            new_path = [pos, pos] + tail
            amr.set_path(new_path)

        # --- 주입 2: 중앙 AMR 이동(출구 우선, 막히면 빈 front로 center→alt→center) ---
        #   - 출구 front가 비어 있으면 주입 없이 종료
        if not front_occupied.get(exit_dir, False):
            return True

        #   - 비어있는 다른 front로 짧게 피신
        for d in dirs4:
            if d == exit_dir:
                continue
            # 해당 방향 레인이 존재하지 않으면 skip
            if d not in self.lane_coords:
                continue
            if not front_occupied.get(d, False):
                alt_front = front_cell[d]
                path_c = center_amr.path
                if center in path_c:
                    i = path_c.index(center)
                    tail_c = path_c[i+1:]
                else:
                    tail_c = path_c[:]
                new_path_c = [center, alt_front, center] + tail_c
                center_amr.set_path(new_path_c)
                break  # 한 방향만 주입

        return True
        
    
    def plan_action(self):
        """
        Deadlock 해소용 액션 시퀀스 계산기.
        - 입력: self.amr_intent_map 스냅샷 (각 AMR의 current_arm, exit_arm, amr_obj.pos)
        - 스택: 교차로 4방향(N,E,S,W) 레인을 스택으로 보고, TOP은 센터에 가장 가까운 칸
        - 액션: (src_dir, dst_dir)  # src의 TOP을 dst의 TOP으로 옮김 (개념적 이동)
        - 용량: 각 스택 용량은 교차로 팔 길이(len_N/E/S/W)
        - 반환: 액션 리스트 [(N,E), (E,S), ...]
        실제 AMR 이동은 하지 않으며, 경로 세그먼트 주입 전 “계획”만 산출.
        """
        actions = []

        stacks, true_target = self.build_stacks_from_snapshot()
        capacity = {'N': self.len_N, 'E': self.len_E, 'S': self.len_S, 'W': self.len_W}
        dirs = list(stacks.keys())
        if not dirs:
            return []
        
        # ---------- 1) 초기화 및 가짜 라벨링 ----------
        # 1.1 버퍼 그룹 선택
        buffer_dir = sorted(((len(stacks[d]), d) for d in dirs))[0][1]
        print(f"Buffer dir: {buffer_dir}")

        # 1.2 버퍼/주요/잉여 그룹 분류
        items_per_tgt = {d: [] for d in dirs}
        for aid, tgt in true_target.items():
            if tgt in items_per_tgt:
                items_per_tgt[tgt].append(aid)
        
        # 주요 그룹: 버퍼가 아닌 각 색(=exit_arm) d에 대해 cap[d]까지 선발(우선: 이미 d 스택에 있는 아이템)
        major = set()
        for d in dirs:
            if d == buffer_dir:
                continue
            want = items_per_tgt.get(d, [])
            in_d = [aid for aid in stacks[d] if true_target.get(aid) == d]  # 현재 d에 있는 d-타깃
            rest = [aid for aid in want if aid not in in_d]                 # d가 목표지만 d 스택엔 없는 애들
            picked = in_d + rest
            for aid in picked[:capacity.get(d, 0)]:
                major.add(aid)

        buffer_group = {aid for aid, t in true_target.items() if t == buffer_dir}
        surplus = {aid for aid, t in true_target.items() if aid not in major and aid not in buffer_group}

        # 1.3 가짜 목표: 주요=진짜 목표 유지, 버퍼/잉여는 버퍼 제외 가장 적게 배정된 스택 순으로 배정(용량 초과는 피함)
        fake_target = {}
        for aid in major:
            fake_target[aid] = true_target[aid]
        
        nonbuf = [d for d in dirs if d != buffer_dir and d in stacks]
        planned_count = {d: sum(1 for aid in major if true_target.get(aid) == d) for d in nonbuf}

        for group in (buffer_group, surplus):
            for aid in group:
                cur = self.amr_intent_map[aid]['current_arm']
                if cur not in stacks:
                    continue
                if not nonbuf:
                    continue

                # 아직 '계획상' 빈자리(capacity 여유)가 있는 비-버퍼 스택 후보들
                selectable = [d for d in nonbuf if planned_count.get(d, 0) < capacity.get(d, 0)]

                if selectable:
                    # planned_count가 가장 작은 스택(동률이면 NESW 순) 선택
                    d = min(selectable, key=lambda x: (planned_count.get(x, 0), 'NESW'.index(x)))
                    fake_target[aid] = d
                    planned_count[d] = planned_count.get(d, 0) + 1
                else:
                    # 모든 비-버퍼 스택이 계획상 가득 찬 경우: 폴백
                    # - 현재 스택이 버퍼가 아니면 거기로
                    # - 버퍼면 비-버퍼 중 planned_count 가장 작은 곳으로(동률 NESW)
                    fallback = cur if cur != buffer_dir else min(nonbuf, key=lambda x: (planned_count.get(x, 0), 'NESW'.index(x)))
                    fake_target[aid] = fallback

        print(f"Fake target: {fake_target}")
        print(f"True target: {true_target}")

        # ---------- 2) 1차 정렬(분할, 정복) ----------
        def move(src, dst):
            if src not in stacks or dst not in stacks:
                return False
            if not stacks[src]:
                return False
            if len(stacks[dst]) >= capacity.get(dst, 0):
                return False

            aid = stacks[src].pop(0)        # pop top
            stacks[dst].insert(0, aid)      # push top
            actions.append((src, dst))

            # 디버그 출력: 매 액션 직후 전체 스택 스냅샷 (TOP이 0번)
            order = "NESW"
            snap = " | ".join(
                f"{d}:TOP {stacks[d]}" for d in order if d in stacks
            )
            print(f"[{len(actions):03d}] {src}->{dst}  buffer={buffer_dir}  ||  {snap}")

            return True
        
        # ---------- 2.1 Divide: (fake_target) 기준 좌/우 파티션 (버퍼 제외) ----------
        left_canon, right_canon = ('N', 'E'), ('S', 'W')

        # 좌/우에 올 수 있는 방향 집합(버퍼는 제외)
        left_dirs = tuple(d for d in left_canon if d in dirs and d != buffer_dir)
        right_dirs = tuple(d for d in right_canon if d in dirs and d != buffer_dir)

        # ---------- 2.2 Conquer-Prep: 버퍼 스택 완전 비우기 ----------
        print("2.2 Empty buffer stack")
        nesw_order = "NESW"

        # 버퍼 TOP부터 하나씩 비우기
        while stacks[buffer_dir]:
            aid = stacks[buffer_dir][0]
            goal = fake_target[aid]

            # 1) 목표 스택으로 먼저 시도 (버퍼가 목표인 경우는 의미 없으니 제외)
            if goal in dirs and goal != buffer_dir and len(stacks[goal]) < capacity.get(goal, 0):
                print("moving to goal stack")
                move(buffer_dir, goal)
                continue

            # 2) 목표가 없거나 가득 차 있으면, 가장 덜 찬 비-버퍼로 이동
            candidates = [d for d in dirs if d != buffer_dir and len(stacks[d]) < capacity.get(d, 0)]
            dst = min(candidates, key=lambda x: (len(stacks[x]), nesw_order.index(x)))
            print("moving to least-filled non-buffer stack")
            move(buffer_dir, dst)
            
        # ---------- 2.3 Conquer: 각 스택을 좌/우 두 블록으로 재배치 ----------
        print("2.3 Reorganize stacks into two blocks")
        # 처리 순서: NESW(버퍼 제외)
        order_dirs = [d for d in nesw_order if d in dirs and d != buffer_dir]
        
        for A in order_dirs:
            # A가 속한 그룹과 반대 그룹 정의
            if A in left_dirs:
                keep_group = set(left_dirs)
                opp_group = set(right_dirs)
            else:
                keep_group = set(right_dirs)
                opp_group = set(left_dirs)

            # A에 남길(keep) 아이템 수
            K = 0
            for aid in stacks[A]:
                if fake_target.get(aid) in keep_group:
                    K += 1

            if K == len(stacks[A]):
                continue  # 이미 정렬된 상태

            # 임시 저장용 스택 B: 반대쪽 중 '남은 용량'이 가장 큰 곳(동률 NESW)
            opp_candidates = [d for d in opp_group if d in stacks]
            B = max(opp_candidates, key=lambda x: (capacity.get(x, 0) - len(stacks[x]), -nesw_order.index(x)))

            # B에 K개 담을 공간 확보: B -> buffer로 비우기
            while capacity.get(B, 0) - len(stacks[B]) < K and stacks[B]:
                print("moving to buffer to make space in B")
                move(B, buffer_dir)

            # A에서 keep 아이템을 제외한 나머지(opp 그룹)를 B로 이동
            while stacks[A]:
                aid_top = stacks[A][0]
                tgt = fake_target.get(aid_top)

                if tgt in keep_group:
                    print("moving to B (keep group)")
                    move(A, B)
                else:
                    print("moving to buffer (opp group)")
                    move(A, buffer_dir)

            # A 재조립: B에 임시 보관한 keep K개를 A로 되돌려 두 블록화 (keep이 A TOP에 연속 배치됨)
            cnt = 0
            while cnt < K and stacks[B] and len(stacks[A]) < capacity.get(A, 0):
                print("reassembling A from B")
                move(B, A)
                cnt += 1

            # buffer 비우기: A/B 중 그룹에 따라 복귀, A가 꽉 차면 B로, 둘 다 꽉 차면 중단
            while stacks.get(buffer_dir, []):
                aidb = stacks[buffer_dir][0]
                tgt = fake_target.get(aidb)

                # A/B 그룹 판단: keep_group(=A쪽), opp_group(=B쪽)
                to_A = (tgt in keep_group)
                to_B = (tgt in opp_group)

                # 기본 우선순위: 자신의 그룹 스택으로 (A 우선 또는 B 우선)
                if to_A:
                    primary, secondary = A, B
                elif to_B:
                    primary, secondary = B, A

                if len(stacks[primary]) < capacity.get(primary, 0):
                    print("moving to primary group stack")
                    move(buffer_dir, primary)
                    continue
                if len(stacks[secondary]) < capacity.get(secondary, 0):
                    print("moving to secondary group stack")
                    move(buffer_dir, secondary)
                    continue

        # ---------- 2.4 Recursive: 하위 그룹(N|E), (S|W)에도 정복 반복 ----------
        print("2.4 Recursive on sub-groups")
        # 하위 그룹 후보 만들기(버퍼 제외, 존재하는 팔만)
        subproblems = []
        pair_left = tuple(d for d in left_canon if d in dirs and d != buffer_dir)
        pair_right = tuple(d for d in right_canon if d in dirs and d != buffer_dir)
        if len(pair_left)  == 2: subproblems.append(((pair_left[0],),  (pair_left[1],)))
        if len(pair_right) == 2: subproblems.append(((pair_right[0],), (pair_right[1],)))
        
        for left_sub, right_sub in subproblems:
            sub_dirs = left_sub + right_sub
            order_sub = [d for d in nesw_order if d in sub_dirs and d != buffer_dir]

            for A in order_sub:
                if A in left_sub:
                    keep_group = set(left_sub)
                    opp_group = set(right_sub)
                else:
                    keep_group = set(right_sub)
                    opp_group = set(left_sub)

                # A에 남길(keep) 개수
                K = 0
                for aid in stacks[A]:
                    if fake_target.get(aid) in keep_group:
                        K += 1

                # 이미 원하는 그룹만 포함되어 있으면 스킵
                if K == len(stacks[A]):
                    continue
                
                # 임시 저장 B(반대쪽 단일 스택)
                opp_list = [d for d in opp_group if d in stacks]
                B = opp_list[0]

                # B에 K개 담을 공간 확보: B -> buffer (필요 시 buffer를 A/B로 흘려서 한 칸 만들기)
                while capacity.get(B, 0) - len(stacks[B]) < K and stacks[B]:
                    move(B, buffer_dir)

                # A 분류: TOP에서 꺼내며 keep은 B로 임시보관, 나머지는 buffer로
                while stacks[A]:
                    aid_top = stacks[A][0]
                    tgt = fake_target.get(aid_top)

                    if tgt in keep_group:
                        move(A, B)
                    else:
                        move(A, buffer_dir)

                # A 재조립: B에 임시 보관한 keep K개를 A로 되돌려, A에 keep 블록 연속 배치
                cnt = 0
                while cnt < K and stacks[B] and len(stacks[A]) < capacity.get(A, 0):
                    move(B, A)
                    cnt += 1

                # buffer 비우기: A/B 중 그룹에 따라 복귀 (A 꽉 차면 B, 둘 다 꽉 차면 중단)
                while stacks.get(buffer_dir, []):
                    aidb = stacks[buffer_dir][0]
                    tgt = fake_target.get(aidb)

                    to_A = (tgt in keep_group)
                    to_B = (tgt in opp_group)

                    if to_A:
                        primary, secondary = A, B
                    elif to_B:
                        primary, secondary = B, A

                    if len(stacks[primary]) < capacity.get(primary, 0):
                        move(buffer_dir, primary)
                        continue
                    if len(stacks[secondary]) < capacity.get(secondary, 0):
                        move(buffer_dir, secondary)
                        continue

        # ---------- 3) True-target refinement: fake → true ----------
        # 각 방향 A를 한 번씩만 처리: A에 가짜≠진짜인 아이템이 있고 TOP이 아니면
        # 그 아이템을 TOP까지 끌어올려 true_target으로 보낸 뒤, 양보/버퍼를 A로 복구
        print("3. True-target refinement")
        for A in order_dirs:
            candB = [d for d in dirs if d not in (A, buffer_dir)]
            B = min(candB, key=lambda x: (len(stacks[x]), nesw_order.index(x)))

            idx = None
            for i, aid in enumerate(reversed(stacks[A])):
                if true_target.get(aid) != fake_target.get(aid):
                    idx = len(stacks[A]) - 1 - i
                    break
            if idx is None or idx == 0:
                continue    # A는 이미 true_target과 일치
            
            # A에 남길(keep) 개수
            K = 0
            for aid in stacks[A]:
                if true_target.get(aid) in keep_group:
                    K += 1

            # 이미 원하는 그룹만 포함되어 있으면 스킵
            if K == len(stacks[A]):
                continue
                
            # 임시 저장용 스택 B: 아이템 수가 가장 적은 곳(동률 NESW)
            candB = [d for d in dirs if d not in (A, buffer_dir)]
            B = min(candB, key=lambda x: (len(stacks[x]), nesw_order.index(x)))

            # B에 K개 담을 공간 확보: B -> buffer로 비우기
            while capacity.get(B, 0) - len(stacks[B]) < K and stacks[B]:
                move(B, buffer_dir)

            # A에서 keep 아이템을 제외한 나머지(opp 그룹)를 B로 이동
            while stacks[A]:
                aid_top = stacks[A][0]
                tgt = true_target.get(aid_top)
                if tgt == A:
                    move(A, B)
                else:
                    move(A, buffer_dir)

            # A 재조립: B에 임시 보관한 keep K개를 A로 되돌려 두 블록화 (keep이 A TOP에 연속 배치됨)
            cnt = 0
            while cnt < K and stacks[B] and len(stacks[A]) < capacity.get(A, 0):
                move(B, A)
                cnt += 1

            # buffer 비우기: A/B 중 그룹에 따라 복귀, A가 꽉 차면 B로, 둘 다 꽉 차면 중단
            while stacks.get(buffer_dir, []):
                aidb = stacks[buffer_dir][0]
                tgt = fake_target.get(aidb)

                if tgt == A:
                    primary, secondary = A, B
                else:
                    primary, secondary = B, A

                if len(stacks[primary]) < capacity.get(primary, 0):
                    move(buffer_dir, primary)
                    continue
                if len(stacks[secondary]) < capacity.get(secondary, 0): 
                    move(buffer_dir, secondary)
                    continue

        # ---------- 3.1 Final push: buffer_group TOPs → buffer_dir ----------
        # 잉여(surplus)는 건드리지 않음. true_target이 buffer_dir인 것만 최종 이동.
        print("3.1 Final push to buffer dir")
        for A in order_dirs:
            while stacks[A]:
                aid_top = stacks[A][0]
                # A의 TOP이 버퍼 그룹이 아니면 다음 A로
                if true_target.get(aid_top) != buffer_dir:
                    break
                move(A, buffer_dir)

        return actions
        
    def build_stacks_from_snapshot(self):
        """
        return:
            stacks  = {'N': [5, 3], 'S': [8], 'W': [2, 7]}  # 5가 N의 TOP
            targets = {5: 'E', 3: 'E', 8: 'N', 2: 'W', 7: 'S'}
        """
        stacks = {d: [] for d in self.present_dirs}
        targets = {}

        pos2id = {rec['amr_obj'].pos: aid for aid, rec in self.amr_intent_map.items() if rec.get('current_arm') in stacks}

        for d, coords, in self.lane_coords.items():
            if d not in stacks:
                continue
            for p in coords:
                if p in pos2id:
                    aid = pos2id[p]
                    stacks[d].append(aid)
        
        for aid, rec in self.amr_intent_map.items():
            tgt = rec['exit_arm']
            if tgt in stacks:
                targets[aid] = tgt
        
        return stacks, targets
    