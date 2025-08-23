import heapq
from typing import Dict, Tuple, Iterable, Optional, Set
import random
import numpy as np

# D* Lite Algorithm
class DStar:
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
        if self.start not in self.g:
            return []

        path = [self.start]
        current = self.start
        while current != self.goal:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                return []
            current = min(
                neighbors,
                key=lambda n: self.g.get(n, float('inf'))
            )
            if self.g.get(current, float('inf')) == float('inf'):
                return []
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
            # 1) 힌트 우선
            cands = []
            hint = dstar_hint.get(a)
            if hint is not None:
                cands.append(hint)
            # 2) 인접칸 (목표까지 가까운 순)
            neigh = neighbors(pos[a])
            neigh.sort(key=lambda c: manhattan(c, goals[a]))
            for c in neigh:
                if c not in cands:
                    cands.append(c)
            # 3) 부모 위치(우선순위 상속으로 swap 허용)
            if parent is not None:
                ppos = pos[parent]
                if ppos in cands:
                    cands.remove(ppos)
                cands.insert(0, ppos)
            # 4) stay 마지막
            if pos[a] in cands:
                cands.remove(pos[a])
            cands.append(pos[a])

            # 고정 점유/엣지와 충돌 후보 제거
            filtered = []
            fcells = fixed_cells(decided, proposals)
            for v in cands:
                if v in fcells and not (parent is not None and v == pos[parent]):
                    continue
                if edge_conflict_with_fixed(a, v):
                    continue
                filtered.append(v)
            return filtered if filtered else [pos[a]]

        # 재귀 할당(우선순위 상속 + 백트래킹) -------------------------------
        proposals: Dict[int, Pos] = {}
        decided: Set[int] = set()

        def assign(a: int, stack: Set[int], parent: Optional[int] = None) -> bool:
            for v in candidate_cells(a, parent, decided, proposals):
                # 그룹 내 이미 결정된 자가 확보한 칸은 금지(단, 부모와의 스왑은 허용)
                if v in fixed_cells(decided, proposals) and not (parent is not None and v == pos[parent]):
                    continue
                if has_edge_conflict_in_group(a, v, decided, proposals):
                    continue

                occ = agent_at.get(v)
                # 점유자가 없거나, 이미 결정되어 그 칸을 비울 예정이면 통과
                can_take = (occ is None) or (occ in decided and proposals.get(occ) != v) or (parent is not None and occ == parent)
                if can_take:
                    proposals[a] = v
                    return True

                # 그룹 내부 미결정 점유자면 우선순위 상속
                if occ in stack or occ not in group:
                    continue
                stack.add(occ)
                if assign(occ, stack, parent=a):
                    proposals[a] = v
                    return True

            # 모든 후보 실패 → 제자리 시도
            stay = pos[a]
            if (stay not in fixed_cells(decided, proposals)
                and not has_edge_conflict_in_group(a, stay, decided, proposals)
                and not edge_conflict_with_fixed(a, stay)):
                proposals[a] = stay
                return True
            return False

        # 우선순위 순으로 결정
        for a in order:
            if a in decided:
                continue
            _ = assign(a, stack=set([a]), parent=None)
            decided.add(a)

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