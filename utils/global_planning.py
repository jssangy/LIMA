import heapq
from typing import Dict, Tuple, Iterable, Optional, Set, List, Sequence
import random
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from bisect import bisect_left

# A* Algorithm
class AStar:
    def __init__(self, map, start, goal):
        self.map = map
        self.start = start
        self.goal = goal

        self.g = {}          # Actual cost
        self.rhs = {}        # Estimated cost
        self.queue = []      # Priority queue

        self.initialize_graph()

    def initialize_graph(self):
        self.g.clear()
        self.rhs.clear()
        self.queue.clear()

        h, w = self.map.shape
        for y in range(h):
            for x in range(w):
                if self.map[y][x] == 0:
                    self.g[(x, y)] = float('inf')
                    self.rhs[(x, y)] = float('inf')

        self.rhs[self.goal] = 0
        self.insert(self.goal)

    def update_goal(self, new_goal):
        self.goal = new_goal
        self.initialize_graph()

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map.shape[1] and 0 <= ny < self.map.shape[0]:
                if self.map[ny][nx] == 0:
                    neighbors.append((nx, ny))
        return neighbors

    def calculate_key(self, node):
        g_rhs = min(self.g[node], self.rhs[node])
        return (g_rhs + self.manhattan(self.start, node), g_rhs)

    def insert(self, node):
        heapq.heappush(self.queue, (self.calculate_key(node), node))

    def update_vertex(self, node):
        if node != self.goal:
            self.rhs[node] = min(
                self.g.get(n, float('inf')) + 1
                for n in self.get_neighbors(node)
            )
        self.queue = [(k, n) for (k, n) in self.queue if n != node]
        heapq.heapify(self.queue)
        if self.g[node] != self.rhs[node]:
            self.insert(node)

    def compute_shortest_path(self):
        while self.queue and (
            self.queue[0][0] < self.calculate_key(self.start) or
            self.rhs[self.start] != self.g[self.start]
        ):
            _, u = heapq.heappop(self.queue)
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
            for s in self.get_neighbors(u):
                self.update_vertex(s)

    def extract_path(self):
        if self.start not in self.g or self.g.get(self.start) == float('inf'):
            return []

        path = [self.start]
        current = self.start
        while current != self.goal:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                return [] # 막다른 길

            # [수정 시작] 비용이 같은 최적 경로가 여러 개일 때 무작위 선택
            
            # 1. 모든 이웃의 비용(g-value)을 계산
            costs = {n: self.g.get(n, float('inf')) for n in neighbors}
            
            # 2. 최소 비용 찾기
            min_cost = min(costs.values())
            
            # 최소 비용이 무한대이면 길이 없는 것
            if min_cost == float('inf'):
                return [] 

            # 3. 최소 비용을 가진 모든 이웃 노드를 후보로 수집
            best_neighbors = [n for n, cost in costs.items() if cost == min_cost]
            
            # 4. 후보 중에서 하나를 무작위로 선택
            current = random.choice(best_neighbors)
            # [수정 끝]

            path.append(current)
            
        return path



Pos = Tuple[int, int]
Edge = Tuple[Pos, Pos]

class PIBT:
    """
    Priority Inheritance with Backtracking (PIBT) - 한 스텝 로컬 스케줄러

    사용 예)
        pibt = PIBT(map_data, seed=1234)
        next_cells = pibt.plan_one_step(
            pos=pos_dict,              # {agent: (x,y)}
            goals=goal_dict,           # {agent: (x,y)}
            dstar_hint=hint_dict,      # (선택) {agent: (x,y) or None}, D*가 제안한 다음 칸
            subset=None,               # (선택) 이 집합에 대해서만 재결정 (없으면 전체)
            fixed_vertices=None,       # (선택) 그룹 밖 에이전트가 t+1에 점유할 칸들
            fixed_edges=None           # (선택) 그룹 밖 에이전트의 (u->v) 엣지들
        )

    반환:
        {agent: next_pos}  # subset이 주어지면 subset에 대해서만 반환

    특징:
      - 우선순위 p_i = age_i + eps_i (eps_i는 동률 방지를 위한 고유 미소 난수)
      - 우선순위 상속 + 백트래킹으로 충돌을 해소하며 1스텝 목적지를 정함
      - dstar_hint가 있으면 그 칸을 후보 최우선으로 고려
      - fixed_vertices/edges로 “그룹 밖” 제안(혹은 예약)을 침범하지 않도록 보장
    """
    def __init__(self, map_data: np.ndarray, seed: int = 1234):
        """
        map_data: 2D numpy array, 0=free, 1=obstacle (4방향 격자)
        """
        if not isinstance(map_data, np.ndarray):
            raise TypeError("map_data must be a numpy array with 0(free)/1(obstacle).")
        self.map = map_data
        self.H, self.W = map_data.shape[:2]

        self._rng = random.Random(seed)
        self._eps: Dict[int, float] = {}   # agent -> tiny random
        self._age: Dict[int, int] = {}     # agent -> fairness age

    # ---- public APIs --------------------------------------------------------

    def reset(self):
        """모든 내부 상태(age/eps) 초기화"""
        self._eps.clear()
        self._age.clear()

    def reset_age(self, agent_id: int):
        """특정 에이전트 우선순위 age를 0으로 리셋 (예: 새 목표 할당 시 호출)"""
        self._age[agent_id] = 0

    def plan_one_step(
        self,
        pos: Dict[int, Pos],
        goals: Dict[int, Pos],
        dstar_hint: Optional[Dict[int, Optional[Pos]]] = None,
        subset: Optional[Iterable[int]] = None,
        fixed_vertices: Optional[Set[Pos]] = None,
        fixed_edges: Optional[Set[Edge]] = None,
    ) -> Dict[int, Pos]:
        """
        subset에 속한 에이전트만 한 스텝 재결정(없으면 모든 에이전트).
        fixed_* 는 subset 밖의 움직임을 '고정 제약'으로 취급하여 침범하지 않도록 함.

        반환: {agent: next_cell} (subset이 주어지면 해당 subset만 반환)
        """
        agents_all = list(pos.keys())
        group = set(subset) if subset is not None else set(agents_all)

        # 내부 상태 준비
        for a in group:
            self._ensure_agent(a)

        # 스냅샷
        agent_at: Dict[Pos, int] = {p: a for a, p in pos.items()}
        fixed_vertices = set(fixed_vertices or set())
        fixed_edges = set(fixed_edges or set())
        dstar_hint = dstar_hint or {}

        # 우선순위: age + eps (내림차순)
        order = sorted(
            list(group),
            key=lambda a: (self._age.get(a, 0) + self._eps.get(a, 0.0)),
            reverse=True
        )

        reserved: Set[Pos] = set(fixed_vertices)  # 그룹 내 이미 결정된 칸

        # 도우미들 -------------------------------------------------------------
        def in_map(x: int, y: int) -> bool:
            return 0 <= x < self.W and 0 <= y < self.H

        def free_cell(c: Pos) -> bool:
            x, y = c
            return in_map(x, y) and self.map[y, x] == 0

        def neighbors(c: Pos):
            x, y = c
            cand = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
            return [q for q in cand if free_cell(q)]

        def manhattan(a: Pos, b: Pos) -> int:
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        def edge_conflict_with_fixed(a: int, v: Pos) -> bool:
            u = pos[a]
            return (u, v) in fixed_edges or (v, u) in fixed_edges

        def has_edge_conflict_in_group(me: int, cand: Pos, decided: Set[int], proposals: Dict[int, Pos]) -> bool:
            # 그룹 내 이미 결정된 에이전트와 정면 스왑 방지
            for a in decided:
                if proposals.get(a) == pos[me] and pos[a] == cand:
                    return True
            return False

        def fixed_cells(decided: Set[int], proposals: Dict[int, Pos]) -> Set[Pos]:
            # 그룹 밖 고정 점유 칸 + 이미 결정된 그룹 칸
            return set(fixed_vertices) | {proposals[a] for a in decided if a in proposals}

        def candidate_cells(a: int, parent: Optional[int], decided: Set[int], proposals: Dict[int, Pos]):
            cands = []
            hint = dstar_hint.get(a)
            if hint is not None:
                cands.append(hint)

            neigh = neighbors(pos[a])
            neigh.sort(key=lambda c: manhattan(c, goals[a]))
            for c in neigh:
                if c not in cands:
                    cands.append(c)

            # ❌ 부모 위치를 우선으로 넣는건 스왑 유도 → 제거 권장
            # if parent is not None:
            #     ppos = pos[parent]
            #     if ppos in cands:
            #         cands.remove(ppos)
            #     cands.insert(0, ppos)

            # stay 마지막
            if pos[a] in cands:
                cands.remove(pos[a])
            cands.append(pos[a])

            # 고정 점유/엣지 필터
            filtered = []
            fcells = fixed_cells(decided, proposals)
            for v in cands:
                # 부모 위치 예외도 제거
                if v in fcells:
                    continue
                if edge_conflict_with_fixed(a, v):
                    continue
                filtered.append(v)
            return filtered if filtered else [pos[a]]

        # 재귀 할당(우선순위 상속 + 백트래킹) -------------------------------
        proposals: Dict[int, Pos] = {}
        decided: Set[int] = set()

        def has_edge_swap_with_any(me: int, cand: Pos, proposals: Dict[int, Pos]) -> bool:
            # 이미 제안된 어떤 b에 대해서도 (b -> pos[me]) && (me -> pos[b]) 금지
            for b, vb in proposals.items():
                if vb == pos[me] and pos[b] == cand:
                    return True
            return False

        def assign(a: int, stack: Set[int], parent: Optional[int] = None) -> bool:
            for v in candidate_cells(a, parent, decided, proposals):
                # 예약/스왑/부모스왑 금지
                if v in reserved:
                    continue
                if parent is not None and v == pos[parent]:   # 부모 위치 진입 금지
                    continue
                if has_edge_swap_with_any(a, v, proposals):
                    continue
                if has_edge_conflict_in_group(a, v, decided, proposals):
                    continue

                occ = agent_at.get(v)

                can_take = (
                    occ is None or
                    (occ in decided and proposals.get(occ) != v)
                )
                if can_take:
                    proposals[a] = v
                    reserved.add(v)           # ★ 예약 확정
                    return True

                # 그룹 내부 미결정 점유자면 우선순위 상속
                if occ in stack or occ not in group:
                    continue
                stack.add(occ)
                if assign(occ, stack, parent=a):
                    # 자식이 성공적으로 다른 칸을 잡았으니 이제 v를 차지
                    if v in reserved:
                        continue              # 혹시 중간에 다른 루트가 잡았으면 다음 후보
                    proposals[a] = v
                    reserved.add(v)           # ★ 예약 확정
                    return True

            # 모든 후보 실패 → stay 시도
            stay = pos[a]
            if (stay not in reserved
                and stay not in fixed_cells(decided, proposals)
                and not has_edge_swap_with_any(a, stay, proposals)
                and not edge_conflict_with_fixed(a, stay)):
                proposals[a] = stay
                reserved.add(stay)            # ★ 예약 확정
                return True
            return False

        # 우선순위 순으로 결정
        for a in order:
            if a in decided:
                continue
            before = set(proposals.keys())
            _ = assign(a, stack=set([a]), parent=None)
            after = set(proposals.keys())
            # 이번 루트에서 새로 확정된 모든 에이전트 보호
            decided.update(after - before)

        # 결과 만들기 + age 업데이트
        result = {a: proposals.get(a, pos[a]) for a in group}
        for a in group:
            self._age[a] = self._age.get(a, 0) + 1
        return result

    # ---- internals ----------------------------------------------------------

    def _ensure_agent(self, a: int):
        if a not in self._eps:
            self._eps[a] = 1e-3 * self._rng.random()
        if a not in self._age:
            self._age[a] = 0


class BFS:
    """
    [신규 클래스]
    목표 지점(goal) 기반의 거리장(Distance Field)을 미리 계산하여 경로를 매우 빠르게 추출하는 플래너.
    - 특정 goal에 대한 거리장은 BFS를 통해 단 한 번만 계산되고 캐시됩니다.
    - 경로 계획은 캐시된 거리장을 따라 가장 가파른 경사(steepest descent)를 찾는 방식으로 즉시 수행됩니다.
    - 재계획 기능은 없으며, 초기 경로 생성에 특화되어 있습니다.
    """
    def __init__(self, map_data: np.ndarray):
        self.map = map_data
        self.H, self.W = map_data.shape
        self._distance_fields: Dict[Pos, np.ndarray] = {}  # goal -> distance_field 맵 캐시

        self.edge_heat = np.zeros((4, self.H, self.W), dtype=np.float32)
        self._dir2idx = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}  # 상, 하, 좌, 우
        self._opp = {0:1, 1:0, 2:3, 3:2}  # 상<->하, 좌<->우

    def reset_heat(self):
        self.edge_heat.fill(0.0)

    def add_heat_from_path(self, path: List[Pos], w: float = 1.0):
        """경로를 따라 edge_heat 누적"""
        if not path or len(path) < 2:
            return
        for (x, y), (nx, ny) in zip(path, path[1:]):
            idx = self._dir2idx.get((nx - x, ny - y))
            if idx is None:
                continue
            self.edge_heat[idx, y, x] += w

    def heat_plan_path(
        self,
        start: Pos,
        goal: Pos,
        alpha: float = 0.3,
        slack: int = 1,
        u_turn_penalty: float = 0.2,
        max_extra: int = 30,
    ) -> List[Pos]:
        if start == goal:
            return [start]

        if goal not in self._distance_fields:
            self._distance_fields[goal] = self._create_field_from_goal(goal)
        field = self._distance_fields[goal]

        if field[start[1], start[0]] < 0:
            print(f"Warning: Start position {start} is unreachable from goal {goal}.")
            return [start]

        path: List[Pos] = [start]
        current: Pos = start
        prev: Pos | None = None

        start_dist = int(field[start[1], start[0]])
        max_steps = start_dist + max_extra

        def dir_idx(cur: Pos, nxt: Pos) -> int | None:
            return self._dir2idx.get((nxt[0] - cur[0], nxt[1] - cur[1]))

        steps = 0
        while current != goal and steps < max_steps:
            steps += 1
            neighbors = self._get_neighbors(current)
            if not neighbors:
                return path

            dist_map = {n: field[n[1], n[0]] for n in neighbors}
            dist_map = {n: d for n, d in dist_map.items() if d >= 0}
            if not dist_map:
                return path

            min_dist = min(dist_map.values())

            # ✅ detour 허용: 최단만 고집하면 heat가 의미 없어질 때가 많음
            cands = [n for n, d in dist_map.items() if d <= min_dist + slack]
            if not cands:
                cands = [n for n, d in dist_map.items() if d == min_dist]

            cx, cy = current

            def opposite_edge_heat(cur: Pos, nxt: Pos) -> float:
                di = dir_idx(cur, nxt)
                if di is None:
                    return 0.0
                nx, ny = nxt
                # ✅ 반대방향(대면) 흐름만 패널티
                return float(self.edge_heat[self._opp[di], ny, nx])

            def score(n: Pos) -> float:
                s = float(dist_map[n])
                s += alpha * opposite_edge_heat(current, n)
                if prev is not None and n == prev:
                    s += u_turn_penalty
                return s

            best = min(score(n) for n in cands)
            best_neighbors = [n for n in cands if score(n) == best]
            next_node = random.choice(best_neighbors)

            prev = current
            current = next_node
            path.append(current)

        return path


    def plan_path(self, start: Pos, goal: Pos) -> List[Pos]:
        """
        주어진 시작점과 목표점에 대한 경로를 추출합니다.
        필요 시 목표점에 대한 거리장을 생성하고 캐시합니다.
        """
        if start == goal:
            return [start]

        # 1. 목표점에 대한 거리장을 얻거나 생성합니다.
        if goal not in self._distance_fields:
            self._distance_fields[goal] = self._create_field_from_goal(goal)
        
        distance_field = self._distance_fields[goal]

        # 2. 생성된 거리장을 따라 경로를 추출합니다.
        path = [start]
        current = start
        
        # 시작점에서 도달 불가능한 경우 체크
        if distance_field[current[1], current[0]] < 0:
            print(f"Warning: Start position {start} is unreachable from goal {goal}.")
            return [start] # 도달 불가능 시 제자리 경로 반환

        while current != goal:
            neighbors = self._get_neighbors(current)
            if not neighbors:
                return path # 막다른 길

            # [수정 시작] 비용이 같은 최적 경로가 여러 개일 때 무작위 선택
            
            # 1. 모든 이웃의 거리장 값을 계산
            distances = {n: distance_field[n[1], n[0]] for n in neighbors}
            
            # 2. 최소 거리 값 찾기
            min_dist = min(distances.values())

            # 도달 불가능한 곳(-1)만 남은 경우
            if min_dist < 0:
                print(f"Warning: Path extraction stuck at {current} (surrounded by unreachable cells).")
                return path

            # 3. 최소 거리를 가진 모든 이웃 노드를 후보로 수집
            best_neighbors = [n for n, dist in distances.items() if dist == min_dist]
            
            # 4. 후보 중에서 하나를 무작위로 선택
            next_node = random.choice(best_neighbors)
            # [수정 끝]
            
            # 더 이상 진행할 수 없는 경우 (주변이 모두 현재보다 멀어지는 경우)
            if distance_field[next_node[1], next_node[0]] >= distance_field[current[1], current[0]]:
                 print(f"Warning: Path extraction stuck at {current} for goal {goal}.")
                 return path

            current = next_node
            path.append(current)
            
        return path

    def _create_field_from_goal(self, goal: Pos) -> np.ndarray:
        """
        목표 지점에서부터 역방향 BFS를 실행하여 거리장을 생성합니다.
        장애물이나 도달 불가능한 지역은 -1로 표시됩니다.
        """
        field = np.full((self.H, self.W), -1, dtype=int)
        gx, gy = goal
        
        if not (0 <= gx < self.W and 0 <= gy < self.H) or self.map[gy, gx] == 1:
            return field # 목표가 맵 밖이거나 벽인 경우

        q = deque([goal])
        field[gy, gx] = 0
        
        while q:
            x, y = q.popleft()
            current_dist = field[y, x]
            
            for nx, ny in self._get_neighbors((x, y)):
                if field[ny, nx] == -1: # 아직 방문하지 않은 곳
                    field[ny, nx] = current_dist + 1
                    q.append((nx, ny))
        return field

    def _get_neighbors(self, pos: Pos) -> List[Pos]:
        x, y = pos
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny < self.H and self.map[ny, nx] == 0:
                neighbors.append((nx, ny))
        return neighbors  

    def plot_edge_heat(self, save_prefix: str | None = None):
        """
        N/S/W/E를 각각 '선분'으로 그립니다.
        - N,S: 세로선만
        - E,W: 가로선만
        """
        for tag in ["N", "S", "W", "E"]:
            save_path = f"{save_prefix}_{tag}.png" if save_prefix else None
            self._plot_one_dir_lines(tag, save_path=save_path)


    def _is_free(self, p: Pos) -> bool:
        x, y = p
        return 0 <= x < self.W and 0 <= y < self.H and self.map[y, x] == 0

    def _random_center_in_range(self, sorted_vals: Sequence[int], a: int, b: int) -> Optional[int]:
        lo, hi = (a, b) if a <= b else (b, a)
        l = bisect_left(sorted_vals, lo)
        r = bisect_left(sorted_vals, hi + 1)
        if l >= r:
            return None
        return random.choice(sorted_vals[l:r])

    def _nearest_center_in_range(self, sorted_vals: Sequence[int], a: int, b: int, target: int) -> Optional[int]:
        lo, hi = (a, b) if a <= b else (b, a)
        l = bisect_left(sorted_vals, lo)
        r = bisect_left(sorted_vals, hi + 1)

        # 범위 내 후보가 없으면 전체에서라도 가장 가까운 값
        vals = sorted_vals[l:r] if l < r else sorted_vals
        if not vals:
            return None

        i = bisect_left(vals, target)
        cands = []
        if i < len(vals): cands.append(vals[i])
        if i > 0: cands.append(vals[i - 1])
        return min(cands, key=lambda v: abs(v - target))

    def _try_straight(self, a: Pos, b: Pos) -> Optional[list[Pos]]:
        ax, ay = a
        bx, by = b
        if ax != bx and ay != by:
            return None

        path = [a]
        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by + step, step):
                if self.map[y, ax] == 1:
                    return None
                path.append((ax, y))
        else:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx + step, step):
                if self.map[ay, x] == 1:
                    return None
                path.append((x, ay))
        return path

    def _plan_segment(self, start: Pos, goal: Pos) -> Optional[list[Pos]]:
        # 직선 가능하면 직선, 아니면 기존 BFS(거리장) 사용
        if not self._is_free(goal):
            return None
        seg = self._try_straight(start, goal)
        if seg is not None:
            return seg
        seg = self.plan_path(start, goal)
        if not seg or seg[-1] != goal:
            return None
        return seg

    def _plan_via(self, start: Pos, waypoints: list[Pos]) -> Optional[list[Pos]]:
        cur = start
        full = [cur]
        for wp in waypoints:
            if wp == cur:
                continue
            seg = self._plan_segment(cur, wp)
            if seg is None:
                return None
            full.extend(seg[1:])
            cur = wp
        return full

    def plan_path_highway(self, start: Pos, goal: Pos, center_xs: list[int], center_ys: list[int], tries: int = 8) -> list[Pos]:
        """
        벽(goal)이 N/S/E/W 끝에 있을 때:
        - 내부 교차로 중심 라인(row/col)을 이용해 '차선변경-고속주행-주차진입' 형태로 경로 생성.
        실패하면 기본 plan_path로 폴백.
        """
        if start == goal:
            return [start]

        sx, sy = start
        gx, gy = goal

        # goal이 벽인지 판정
        on_top = (gy == 0)
        on_bottom = (gy == self.H - 1)
        on_left = (gx == 0)
        on_right = (gx == self.W - 1)

        # 코너면 방향 하나 랜덤 선택
        if (on_top or on_bottom) and (on_left or on_right):
            if random.random() < 0.5:
                on_left = on_right = False
            else:
                on_top = on_bottom = False

        # CASE A: 위/아래 벽 -> y_rand 랜덤(센터y), x_align은 가까운 센터x
        if on_top or on_bottom:
            x_align = self._nearest_center_in_range(center_xs, sx, gx, sx)
            if x_align is None:
                return self.plan_path(start, goal)

            for _ in range(tries):
                y_rand = self._random_center_in_range(center_ys, sy, gy)
                if y_rand is None:
                    break

                # 경유점 구성 (형태 유지용)
                waypoints = [(x_align, sy), (x_align, y_rand), (gx, y_rand), goal]

                path = self._plan_via(start, waypoints)
                if path is not None:
                    return path

            return self.plan_path(start, goal)

        # CASE B: 좌/우 벽 -> x_rand 랜덤(센터x), y_align은 가까운 센터y
        if on_left or on_right:
            y_align = self._nearest_center_in_range(center_ys, sy, gy, sy)
            if y_align is None:
                return self.plan_path(start, goal)

            for _ in range(tries):
                x_rand = self._random_center_in_range(center_xs, sx, gx)
                if x_rand is None:
                    break

                waypoints = [(sx, y_align), (x_rand, y_align), (x_rand, gy), goal]

                path = self._plan_via(start, waypoints)
                if path is not None:
                    return path

            return self.plan_path(start, goal)

        # goal이 벽이 아니면 그냥 기본
        return self.plan_path(start, goal)


    

class AStar_for_CBS:
    """ Low-level: Space-Time A* for CBS """
    def __init__(self, map_data, start, goal, solution_for_cat):
        self.map = map_data
        self.start = start
        self.goal = goal
        self.height, self.width = map_data.shape
        self.conflict_avoidance_table = self._build_cat(solution_for_cat)  # CAT: Conflict Avoidance Table

    def _build_cat(self, solution):
        """Tie-Breaking을 위한 충돌 회피 테이블(CAT)을 생성"""
        cat = defaultdict(int)  # Key: (position, time) / Value: count of agents at that position and time
        if not solution:
            return cat

        max_len = max(len(path) for path in solution.values()) if solution else 0
        for t in range(max_len):
            for path in solution.values():
                pos = path[t] if t < len(path) else path[-1]
                cat[(pos, t)] += 1
        return cat

    def _get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        # [수정] 5-way movement: wait (0,0) and 4 directions
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.map[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def find_path(self, constraints: Set):
        # (f-value, tie-breaker, g-value, position, time, path)
        open_list = [(self._manhattan_distance(self.start, self.goal), 0, 0, self.start, 0, [self.start])]
        visited = set() # (position, time)

        while open_list:
            # [수정] f-value는 g(time) + h(heuristic) 입니다.
            f, tie_breaker, g, current_pos, time, path = heapq.heappop(open_list)

            if (current_pos, time) in visited: # Duplicate Detection
                continue
            visited.add((current_pos, time))

            # [수정] A*는 GOAL을 pop했을 때가 최단 경로임을 보장합니다.
            # "wait at goal" 제약 조건을 처리하기 위한 로직을 추가합니다.
            if current_pos == self.goal:
                max_time = time
                for c_item, c_time in constraints:
                    # c_item이 (x,y) 형태의 vertex이고, goal과 일치하는지 확인
                    if c_item == self.goal:
                        max_time = max(max_time, c_time)
                # 제약시간까지 목표 지점에서 대기하는 경로를 만듭니다.
                if time < max_time:
                    path.extend([self.goal] * (max_time - time))
                # 경로를 찾았으므로 반환합니다.
                return path

            for neighbor_pos in self._get_neighbors(current_pos):
                next_time = time + 1

                # [수정] 제약조건을 명시적으로 확인합니다.
                # 1. Vertex constraint
                vertex_constraint = (neighbor_pos, next_time)
                if vertex_constraint in constraints:
                    continue
                
                # 2. Edge constraint (swap)
                edge_constraint = ((current_pos, neighbor_pos), time)
                if edge_constraint in constraints:
                    continue
                
                # [수정] A* f-value 계산
                if (neighbor_pos, next_time) not in visited:
                    new_path = path + [neighbor_pos]
                    new_g = g + 1 # g-value는 시간(time)과 동일
                    h = self._manhattan_distance(neighbor_pos, self.goal)
                    new_f = new_g + h

                    # Tie-Breaking Policy
                    new_tie_breaker = self.conflict_avoidance_table.get((neighbor_pos, next_time), 0)
                    heapq.heappush(open_list, (new_f, new_tie_breaker, new_g, neighbor_pos, next_time, new_path))
        
        # 경로를 찾지 못함
        return None


@dataclass(order=True)
class CTNode:
    """충돌 트리(CT)의 노드. 부모를 따라 제약조건을 수집하는 방식."""
    cost: int
    # Tie-Breaking을 위해 충돌 수를 두 번째 정렬 기준으로 추가
    num_conflicts: int = field(compare=True)

    node_id: int = field(compare=True)

    solution: Dict[int, List[Tuple[int, int]]] = field(compare=False)
    constraint: Optional[Tuple[int, Tuple, int]] = field(compare=False, default=None)
    parent: Optional['CTNode'] = field(compare=False, default=None)

class CBS:
    """Conflict-Based Search (CBS) 알고리즘 구현 (메모리 효율적 방식)"""
    def __init__(self, map_data: np.ndarray, agents: Dict[int, Dict[str, Tuple[int, int]]]):
        self.map = map_data
        self.agents = agents
        self.agent_ids = list(agents.keys())

    def _get_constraints_for_agent(self, node: CTNode, agent_id: int) -> Set:
        """
        [수정] CTNode의 부모를 재귀적으로 탐색하며 'loc'와 'time'만 추출합니다.
        """
        constraints = set()
        curr = node
        while curr is not None:
            if curr.constraint and curr.constraint[0] == agent_id:
                # constraint = (agent_id, loc, time)
                # constraints set에는 (loc, time)을 추가
                constraints.add((curr.constraint[1], curr.constraint[2]))
            curr = curr.parent
        return constraints

    def solve(self, time_limit: float = 10.0) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        start_time = time.time()
        open_list = []
        initial_solution = {}

        node_counter = 0

        for agent_id in self.agent_ids:
            start, goal = self.agents[agent_id]['start'], self.agents[agent_id]['goal']
            planner = AStar_for_CBS(self.map, start, goal, {})
            path = planner.find_path(set())
            if path is None: 
                print(f"Agent {agent_id} cannot find initial path.")
                return None
            initial_solution[agent_id] = path
        
        root = CTNode(
            cost=self.calculate_sic(initial_solution),
            num_conflicts=self.find_all_conflicts(initial_solution),
            solution=initial_solution,
            node_id=node_counter
        )
        node_counter += 1

        heapq.heappush(open_list, root)
        best_node_so_far = root

        while open_list:
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit:
                print(f"\n!!! CBS Timeout after {elapsed_time:.2f} seconds. !!!")
                # 제한 시간을 초과하면, 현재까지 찾은 가장 좋은 해를 반환
                print(f"    > Returning best found solution with {best_node_so_far.num_conflicts} conflicts.\n")
                return self.pad_paths(best_node_so_far.solution)

            P = heapq.heappop(open_list)

            # [디버깅]
            # print(f"Popped node {P.node_id}: cost={P.cost}, conflicts={P.num_conflicts}")

            if P.num_conflicts < best_node_so_far.num_conflicts:
                best_node_so_far = P

            conflict = self.find_first_conflict(P.solution)

            if conflict is None:
                print(f"\n[CBS Solve] Optimal solution found in {time.time() - start_time:.2f} seconds.")
                return self.pad_paths(P.solution)

            agent1, agent2, loc, conflict_time = conflict
            # [디버깅]
            # print(f"  > Found conflict: ({agent1}, {agent2}) at {loc}, t={conflict_time}")

            for agent_to_constrain in [agent1, agent2]:
                
                # [수정] 제약조건 생성 로직을 명확히 함
                # loc는 (x,y) (vertex) 또는 ((x1,y1), (x2,y2)) (edge)일 수 있습니다.
                # Edge conflict인 경우, 해당 agent의 이동 (pos1_t -> pos1_t_plus_1)을 loc로 받습니다.
                # 이 로직은 find_first_conflict의 반환 값에 의존합니다.
                
                # agent1, agent2, loc, conflict_time = conflict
                # find_first_conflict가 edge conflict시 (a1, a2, (a1_from, a1_to), t)를 반환한다고 가정
                
                new_constraint_loc = loc
                
                # [수정] Edge conflict의 경우, 각 agent에 맞는 edge constraint를 만들어야 합니다.
                if isinstance(loc, tuple) and len(loc) == 2 and isinstance(loc[0], tuple):
                    # loc = (pos1_t, pos1_t_plus_1)
                    # A*는 (current_pos, neighbor_pos)를 확인합니다.
                    # conflict는 agent1과 agent2의 스왑입니다.
                    # agent1의 이동: loc[0] -> loc[1]
                    # agent2의 이동: loc[1] -> loc[0]
                    if agent_to_constrain == agent1:
                        new_constraint_loc = (loc[0], loc[1])
                    else: # agent_to_constrain == agent2
                        new_constraint_loc = (loc[1], loc[0])

                new_constraint = (agent_to_constrain, new_constraint_loc, conflict_time)
                
                # 이 에이전트에 대한 모든 제약조건을 부모로부터 수집
                agent_constraints = self._get_constraints_for_agent(P, agent_to_constrain)
                
                # 새 제약조건 추가
                agent_constraints.add((new_constraint[1], new_constraint[2]))

                start, goal = self.agents[agent_to_constrain]['start'], self.agents[agent_to_constrain]['goal']

                # Tie-Breaking을 위해 다른 에이전트들의 경로를 전달
                other_agents_solution = {aid: p for aid, p in P.solution.items() if aid != agent_to_constrain}
                planner = AStar_for_CBS(self.map, start, goal, other_agents_solution)
                
                new_path = planner.find_path(agent_constraints)

                if new_path is not None:
                    new_solution = P.solution.copy()
                    new_solution[agent_to_constrain] = new_path
                    
                    new_cost = self.calculate_sic(new_solution)
                    new_num_conflicts = self.find_all_conflicts(new_solution)

                    child_node = CTNode(
                        cost=new_cost,
                        num_conflicts=new_num_conflicts,
                        solution=new_solution,
                        constraint=new_constraint,
                        parent=P,
                        node_id=node_counter
                    )








                    # [디버깅]
                    # print(f"    > Creating child {node_counter}: constraint=({new_constraint[0]}, {new_constraint[1]}, {new_constraint[2]}), cost={new_cost}, conflicts={new_num_conflicts}")

                    node_counter += 1
                    heapq.heappush(open_list, child_node)
                # else:
                    # [디버깅]
                    # print(f"    > Child for agent {agent_to_constrain} found NO PATH. Pruning branch.")

        print(f"\n[CBS Solve] No solution found after {time.time() - start_time:.2f} seconds.")
        return None

    def find_first_conflict(self, solution: Dict[int, List[Tuple[int, int]]]):
        max_len = max(len(p) for p in solution.values()) if solution else 0
        
        # [수정] 경로는 0-indexed이므로 max_len까지 순회 (padding 감안)
        for t in range(max_len):
            positions_at_t = defaultdict(list)
            for agent_id, path in solution.items():
                pos = path[t] if t < len(path) else path[-1]
                positions_at_t[pos].append(agent_id)

            for pos, agents in positions_at_t.items():  # vertex conflict
                if len(agents) > 1:
                    return (agents[0], agents[1], pos, t) # (A1, A2, (x,y), t)

            # Edge conflict (swaps)
            # [수정] Edge conflict는 t -> t+1 이동에서 발생합니다.
            # t+1이 max_len을 넘지 않도록 range(max_len - 1)까지 확인
            if t < max_len - 1:
                for agent1 in self.agent_ids:
                    for agent2 in self.agent_ids:
                        if agent1 >= agent2: continue
                        
                        path1, path2 = solution[agent1], solution[agent2]
                        
                        pos1_t = path1[t] if t < len(path1) else path1[-1]
                        pos1_t_plus_1 = path1[t+1] if t + 1 < len(path1) else path1[-1]
                        
                        pos2_t = path2[t] if t < len(path2) else path2[-1]
                        pos2_t_plus_1 = path2[t+1] if t + 1 < len(path2) else path2[-1]

                        # Swap
                        if pos1_t == pos2_t_plus_1 and pos2_t == pos1_t_plus_1:
                            # (A1, A2, (A1's move), time)
                            return (agent1, agent2, (pos1_t, pos1_t_plus_1), t)
        return None

    def find_all_conflicts(self, solution: Dict[int, List[Tuple[int, int]]]) -> int:
        NumOfConflicts = 0
        max_len = max(len(p) for p in solution.values()) if solution else 0
        
        for t in range(max_len):
            positions_at_t = defaultdict(list)
            for agent_id, path in solution.items():
                pos = path[t] if t < len(path) else path[-1]
                positions_at_t[pos].append(agent_id)
            
            for pos, agents in positions_at_t.items():
                if len(agents) > 1:
                    from itertools import combinations
                    for a1, a2 in combinations(agents, 2):
                        NumOfConflicts += 1
            
            if t < max_len - 1:
                for agent1 in self.agent_ids:
                    for agent2 in self.agent_ids:
                        if agent1 >= agent2: continue
                        
                        path1, path2 = solution[agent1], solution[agent2]
                        
                        pos1_t = path1[t] if t < len(path1) else path1[-1]
                        pos1_t_plus_1 = path1[t+1] if t + 1 < len(path1) else path1[-1]
                        
                        pos2_t = path2[t] if t < len(path2) else path2[-1]
                        pos2_t_plus_1 = path2[t+1] if t + 1 < len(path2) else path2[-1]
                        
                        if pos1_t == pos2_t_plus_1 and pos2_t == pos1_t_plus_1:
                            NumOfConflicts += 1

        return NumOfConflicts

    def calculate_sic(self, solution: Dict[int, List[Tuple[int, int]]]) -> int:
        """
        [수정] Sum-of-Costs는 목표에 도달한 '시간'의 합입니다.
        경로 길이가 L이면, t=0, 1, ..., L-1 이므로 비용은 L-1 입니다.
        하지만, A*가 목표 지점에서 제약 때문에 더 기다리도록 경로를 반환할 수 있습니다.
        (예: 10초짜리 경로지만, 제약 때문에 t=12까지 기다리면, 경로는 13개가 됨)
        정확한 비용은 "목표에 도달한 시간"입니다.
        
        A*가 반환하는 경로는 (start, ... , goal, [goal, ...]) 형태입니다.
        비용은 (경로의 길이 - 1)이 맞습니다.
        """
        cost = 0
        for path in solution.values():
            if not path: continue
            # 경로의 마지막이 목표지점이라고 가정
            goal = path[-1]
            last_goal_time = 0
            for t, pos in enumerate(path):
                if pos == goal:
                    last_goal_time = t
            
            # 만약 경로가 (start) -> (goal) [t=1] -> (goal) [t=2] 라면 len=3, cost=2.
            # 하지만 (start) -> (other) [t=1] -> (goal) [t=2] 라면 len=3, cost=2.
            # (start) -> (goal) [t=1] -> (other) [t=2] -> (goal) [t=3] 라면 len=4, cost=3.
            
            # [cite_start]논문에 따르면[cite: 118], 비용은 "목표에 마지막으로 도달한 시간"입니다.
            # A*가 반환한 경로의 마지막은 항상 목표이므로, len(path) - 1이 그 시간입니다.
            cost += len(path) - 1
        return cost


    def pad_paths(self, solution: Dict[int, List[Tuple[int, int]]]):
        max_len = max(len(path) for path in solution.values()) if solution else 0
        padded_solution = {}
        for agent_id, path in solution.items():
            if not path: # 경로가 없는 비상상황
                start_pos = self.agents[agent_id]['start']
                padded_solution[agent_id] = [start_pos] * max_len
                continue
                
            last_pos = path[-1]
            padded_path = path + [last_pos] * (max_len - len(path))
            padded_solution[agent_id] = padded_path
        return padded_solution


# ... (Commented out old implementations) ...