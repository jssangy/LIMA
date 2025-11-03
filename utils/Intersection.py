import random
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

        self.paths = {}                 # {amr_id: [(x,y), ...]}


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
        trace = []
        locks = []

        stacks, targets = self.build_stacks_from_snapshot()
        capacity = {'N': self.len_N, 'E': self.len_E, 'S': self.len_S, 'W': self.len_W}
        dirs = [d for d in "NESW" if d in self.present_dirs]

        predicted_stacks = self.predicted_stacks(stacks)

        locked = {aid: False for aid in targets.keys()}
        for d in dirs:
            for aid in stacks[d]:
                if targets.get(aid) == d:
                    locked[aid] = True
                else:
                    break
        
        trace.append({d: stacks[d][:] for d in stacks})
        locks.append(locked.copy())

        # 라운드로빈 상태
        pivot_idx = 0
        no_progress_count = 0

        while True:
            # 1) TOP 스캔 1회
            if self.top_scan(stacks, targets, capacity, dirs, actions, locked):
                trace.append({d: stacks[d][:] for d in stacks})
                locks.append(dict(locked))
                no_progress_count = 0
                continue

            # 2) 라운드 로빈: 현재 피벗 P에서 1건만 이동
            P = dirs[pivot_idx]
            progressed = False

            if stacks[P] and self.to_letter(stacks[P], targets) != predicted_stacks.get(P, []):
                # TOP을 '가장 적게 찬' 스택으로 이동
                candidates = [d for d in dirs if d != P and len(stacks[d]) < capacity[d]]
                min_len = min(len(stacks[d]) for d in candidates)
                ties = [d for d in candidates if len(stacks[d]) == min_len]
                candidate = random.choice(ties)  # 동률이면 랜덤 선택
                if candidate:
                    aid = stacks[P].pop()
                    stacks[candidate].append(aid)
                    actions.append((P, candidate))
                    progressed = True

                    trace.append({d: stacks[d][:] for d in stacks})
                    locks.append(dict(locked))

            # 피벗 내 아이템이 모두 피벗과 같은 목적지이거나 피벗이 비어있으면 업데이트
            if not stacks[P] or all(targets.get(aid) == P for aid in stacks[P]):
                pivot_idx = (pivot_idx + 1) % len(dirs)

            if progressed:
                no_progress_count = 0
            else:
                no_progress_count += 1
                if no_progress_count >= len(dirs):
                    break  # 더 이상 이동할 수 없으면 종료

        return actions, trace, locks


    def top_scan(self, stacks, targets, capacity, dirs, actions, locked):
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

                locked[top_id] = True

                return True  # 한 번 이동했으면 즉시 반환
        
        return False  # 이동할 것이 없으면 False 반환
    

    def build_stacks_from_snapshot(self):
        """
        return:
            stacks  = {'N': [5, 3], 'S': [8], 'W': [2, 7]}  # 5가 N의 TOP
            targets = {5: 'E', 3: 'E', 8: 'N', 2: 'W', 7: 'S'}
        """
        ret = self.build_prestage_paths()
        target_lanes = ret[1]

        dirs = [d for d in "NESW" if d in self.present_dirs]
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
    
    
    def predicted_stacks(self, stacks):
        """
        각 스택의 '정답 타깃 배열'을 만든다. (오른쪽 끝=TOP)
        - 반환: {"N": ["N",...], "E":[...], "S":[...], "W":[...]}
        - 각 목표 g의 prefix는 min(총개수, capacity[g])개를 자기 문자로 채움
        - 초과분(overflow)은 '가장 적게 찬' 스택 TOP에 주차(동률이면 N→E→S→W)
        """
        # 사용 팔 및 용량
        dirs = [d for d in "NESW" if d in self.present_dirs]
        cap_all = {'N': self.len_N, 'E': self.len_E, 'S': self.len_S, 'W': self.len_W}
        capacity = {d: cap_all[d] for d in dirs}

        # 현재 포함된 모든 아이템 id 수집
        present_ids = []
        for d in dirs:
            present_ids.extend(stacks.get(d, []))

        # id -> goal 매핑으로 목표별 총량 집계
        goal_counts = {d: 0 for d in dirs}
        for aid in present_ids:
            rec = self.amr_intent_map.get(aid)
            if not rec:
                continue
            g = rec.get('exit_arm')
            if g in goal_counts:
                goal_counts[g] += 1

        # 기본 채움(자기 팔 prefix), 초과 계산
        fill = {g: min(goal_counts[g], capacity[g]) for g in dirs}
        overflow = {g: goal_counts[g] - fill[g] for g in dirs}

        # 예측 스택 초기화: 바닥(prefix)을 자기 문자로 채움 (오른쪽이 TOP)
        predicted = {d: [d] * fill[d] for d in dirs}

        # NESW 타이브레이크용 인덱스
        nesw_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}

        # 초과분을 호스트 팔의 TOP에 배치
        # g 순회도 NESW로(결정성)
        for g in dirs:  # NESW
            k = overflow[g]
            while k > 0:
                # 호스트 후보(슬랙 있는 팔만), 자기 팔 제외
                candidates = [h for h in dirs if h != g and len(predicted[h]) < capacity[h]]
                if not candidates:
                    raise RuntimeError(f"predicted_stacks: overflow '{g}'를 수용할 호스트가 없습니다.")
                # 길이 → NESW 순서로 타이브레이크
                host = min(candidates, key=lambda h: (len(predicted[h]), nesw_idx[h]))
                predicted[host].append(g)  # TOP에 얹음(오른쪽 append)
                k -= 1

        return predicted


    def to_letter(self, stacks, targets):
        out = []
        for a in stacks:
            out.append(targets.get(a))
        return out


    def build_prestage_paths(self):
        """
        데드락 해소용 프리-스테이지 경로 생성/주입.
        - 레인/센터에 있는 모든 AMR을 대상으로,
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
        dirs = [d for d in "NESW" if d in self.present_dirs]

        # --- 초기 paths: 현재 pos 1칸 입력 ---
        paths = {}
        for aid, rec in self.amr_intent_map.items():
            a = rec.get('amr_obj')
            if a is None or getattr(a, "pos", None) is None:
                continue
            paths[aid] = [a.pos]

        # 0) 점유 스냅샷 (near→far)
        lanes = {d: [None] * len(self.lane_coords[d]) for d in dirs}
        pos2aid = {}
        for aid, rec in self.amr_intent_map.items():
            a = rec.get('amr_obj')
            if a is not None and getattr(a, "pos", None) is not None:
                pos2aid[a.pos] = aid

        for d, coords in self.lane_coords.items():
            for i, p in enumerate(coords):  # i=0: near(front)
                lanes[d][i] = pos2aid.get(p, None)

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
            # 만약 모든 팔 만실이면 삽입 생략(정책에 따라 외부 주차 등 추가 가능)

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

        for aid, p in paths.items():
            self.paths[aid] = p[:]

        return lanes, target_lanes, paths, max_steps