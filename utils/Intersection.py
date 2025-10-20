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
        Round Robin 스택 재정렬 규칙에 따라 (src, dst) 이동 시퀀스를 생성해 반환.
        - src, dst ∈ {'N','E','S','W'}
        - source 스택의 TOP 아이템을 destination 스택의 TOP으로 '개념적으로' 이동하는 튜플을 actions에 누적
        - 현재 self.amr_intent_map의 상태로부터 스택을 구성하여 순수하게 계획만 세움(환경 변경 X)
        """
        actions = []

        stacks, targets = self.build_stacks_from_snapshot()
        capacity = {'N': self.len_N, 'E': self.len_E, 'S': self.len_S, 'W': self.len_W}
        dirs = [d for d in "NESW" if d in self.present_dirs]

        # === [000] 초기 스냅샷 출력 ===
        order = "NESW"
        snap_parts = []
        for d in order:
            if d in stacks:
                row = "[" + ", ".join(targets.get(a, "?") for a in reversed(stacks[d])) + "]"
                snap_parts.append(f"{d}:TOP {row}")
        snap = " | ".join(snap_parts)
        print(f"[000] INIT ||  {snap}")
        # ============================

        # 라운드로빈 상태
        pivot_idx = 0
        no_progress_count = 0

        while True:
            # 1) TOP 스캔 1회
            if self.top_scan(stacks, targets, capacity, dirs, actions):
                no_progress_count = 0
                continue

            # 2) 라운드 로빈: 현재 피벗 P에서 1건만 이동
            P = dirs[pivot_idx]
            progressed = False

            if stacks[P] and not all(targets.get(aid) == P for aid in stacks[P]):
                # TOP을 '가장 적게 찬' 스택으로 이동
                candidates = [d for d in dirs if d != P and len(stacks[d]) < capacity[d]]
                candidate = min(candidates, key=lambda d: len(stacks[d]), default=None)
                if candidate:
                    aid = stacks[P].pop()
                    stacks[candidate].append(aid)
                    actions.append((P, candidate))
                    progressed = True

                    print(f"Round Robin Move (pivot {P}):")
                    order = "NESW"
                    snap_parts = []
                    for d in order:
                        if d in stacks:
                            row = "[" + ", ".join(targets.get(a, "?") for a in reversed(stacks[d])) + "]"
                            snap_parts.append(f"{d}:TOP {row}")
                    snap = " | ".join(snap_parts)
                    print(f"[{len(actions):03d}] {P}->{candidate} ||  {snap}")

            # 피벗 내 아이템이 모두 피벗과 같은 목적지이거나 피벗이 비어있으면 업데이트
            if not stacks[P] or all(targets.get(aid) == P for aid in stacks[P]):
                pivot_idx = (pivot_idx + 1) % len(dirs)

            if progressed:
                no_progress_count = 0
            else:
                no_progress_count += 1
                if no_progress_count >= len(dirs):
                    break  # 더 이상 이동할 수 없으면 종료

        return actions
    

    def top_scan(self, stacks, targets, capacity, dirs, actions):
        """
        TOP 스캔 1회 수행해 즉시 정렬 1건을 만들고 True 반환.
        이동할 것이 없으면 False 반환.
        """
        for src in dirs:
            stack_src = stacks[src]
            if not stack_src:
                continue
            top_id = stack_src[-1]
            goal = targets.get(top_id)

            # 이미 목표에 도달했으면 스킵
            if goal == src:
                continue
            
            gstack = stacks[goal]
            goal_pure = (len(gstack) == 0) or all(targets.get(aid) == goal for aid in gstack)
            has_room = len(gstack) < capacity[goal]

            # 목표 스택이 비어있거나 'goal' 색으로만 이루어져 있고, 여유칸이 있으면 즉시 이동
            if goal_pure and has_room:
                stack_src.pop()
                stacks[goal].append(top_id)
                actions.append((src, goal))


                print("TOP Scan Move:")
                order = "NESW"
                snap_parts = []
                for d in order:
                    if d in stacks:
                        row = "[" + ", ".join(targets.get(a, "?") for a in reversed(stacks[d])) + "]"
                        snap_parts.append(f"{d}:TOP {row}")
                snap = " | ".join(snap_parts)
                print(f"[{len(actions):03d}] {src}->{goal} ||  {snap}")


                return True  # 한 번 이동했으면 즉시 반환
        
        return False  # 이동할 것이 없으면 False 반환
    

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
            for p in reversed(coords):
                if p in pos2id:
                    aid = pos2id[p]
                    stacks[d].append(aid)
        
        for aid, rec in self.amr_intent_map.items():
            tgt = rec['exit_arm']
            if tgt in stacks:
                targets[aid] = tgt
        
        return stacks, targets
    
    