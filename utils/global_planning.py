import heapq
from typing import Dict, Tuple, Iterable, Optional, Set, List
import random
import numpy as np
import time  

import heapq, random
from collections import defaultdict, deque

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


# =================================================================
# --- [추가] Conflict-Based Search (CBS) 구현 시작 ---
# =================================================================

from dataclasses import dataclass, field

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
        
        return None

@dataclass(order=True)
class CTNode:
    """충돌 트리(CT)의 노드. 부모를 따라 제약조건을 수집하는 방식."""
    cost: int
    # Tie-Breaking을 위해 충돌 수를 두 번째 정렬 기준으로 추가
    num_conflicts: int = field(compare=True)
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
        constraints = set()
        curr = node
        while curr is not None:
            if curr.constraint and curr.constraint[0] == agent_id:
                constraints.add((curr.constraint[1], curr.constraint[2]))
            curr = curr.parent
        return constraints

    def solve(self, time_limit: float = 10.0) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        start_time = time.time()

        open_list = []
        initial_solution = {}

        for agent_id in self.agent_ids:
            start, goal = self.agents[agent_id]['start'], self.agents[agent_id]['goal']
            planner = AStar_for_CBS(self.map, start, goal, {})
            path = planner.find_path(set())
            if path is None: return None
            initial_solution[agent_id] = path

        root = CTNode(
            cost=self.calculate_sic(initial_solution),
            num_conflicts=self.find_all_conflicts(initial_solution),
            solution=initial_solution
        )
        heapq.heappush(open_list, root)

        best_node_so_far = root

        while open_list:
            # [추가] 루프 시작 부분에서 시간제한 체크
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit:
                print(f"\n!!! CBS Timeout after {elapsed_time:.2f} seconds. !!!")
                # 제한 시간을 초과하면, 현재까지 찾은 가장 좋은 해를 반환
                print(f"    > Returning best found solution with {best_node_so_far.num_conflicts} conflicts.")
                return self.pad_paths(best_node_so_far.solution)

            P = heapq.heappop(open_list)

            if P.num_conflicts < best_node_so_far.num_conflicts:
                best_node_so_far = P

            conflict = self.find_first_conflict(P.solution)

            if conflict is None:
                return self.pad_paths(P.solution)

            agent1, agent2, loc, conflict_time = conflict


            for agent in [agent1, agent2]:

                # [수정] Edge conflict의 경우, 각 agent에 맞는 edge constraint를 만들어야 합니다.
                new_constraint_loc = loc

                if isinstance(loc, tuple) and len(loc) == 2 and isinstance(loc[0], tuple):
                    # loc = (pos1_t, pos1_t_plus_1)
                    # A*는 (current_pos, neighbor_pos)를 확인합니다.
                    # conflict는 agent1과 agent2의 스왑입니다.
                    # agent1의 이동: loc[0] -> loc[1]
                    # agent2의 이동: loc[1] -> loc[0]
                    if agent == agent1:
                        new_constraint_loc = (loc[0], loc[1])
                    else: # agent_to_constrain == agent2
                        new_constraint_loc = (loc[1], loc[0])

                new_constraint = (agent, new_constraint_loc, conflict_time)
                
                
                # new_constraint = None
                # if isinstance(loc, tuple) and len(loc) == 2 and isinstance(loc[0], tuple):
                #     new_constraint = (agent, (loc[0], loc[1]), conflict_time)  # edge conflict
                # else:
                #     new_constraint = (agent, loc, conflict_time)  # vertex conflict



                agent_constraints = self._get_constraints_for_agent(P, agent)  # Get constraints from parents
                agent_constraints.add((new_constraint[1], new_constraint[2]))

                start, goal = self.agents[agent]['start'], self.agents[agent]['goal']

                # Tie-Breaking을 위해 다른 에이전트들의 경로를 전달
                other_agents_solution = {aid: p for aid, p in P.solution.items() if aid != agent}
                planner = AStar_for_CBS(self.map, start, goal, other_agents_solution)
                new_path = planner.find_path(agent_constraints)

                if new_path is not None:
                    new_solution = P.solution.copy()
                    new_solution[agent] = new_path

                    child_node = CTNode(
                        cost=self.calculate_sic(new_solution),
                        num_conflicts=self.find_all_conflicts(new_solution),
                        solution=new_solution,
                        constraint=new_constraint,
                        parent=P
                    )
                    heapq.heappush(open_list, child_node)
        return None

    def find_first_conflict(self, solution: Dict[int, List[Tuple[int, int]]]):
        max_len = max(len(p) for p in solution.values()) if solution else 0
        for t in range(max_len):
            positions_at_t = defaultdict(list)
            for agent_id, path in solution.items():
                pos = path[t] if t < len(path) else path[-1]
                positions_at_t[pos].append(agent_id)

            for pos, agents in positions_at_t.items():  # vertex conflict (including the case of k > 2)
                if len(agents) > 1:
                    # k > 2 충돌의 경우, 처음 두 에이전트만 선택
                    return (agents[0], agents[1], pos, t)

            for agent1 in self.agent_ids:  # edge conflict
                for agent2 in self.agent_ids:
                    if agent1 >= agent2: continue
                    path1, path2 = solution[agent1], solution[agent2]
                    pos1_t = path1[t] if t < len(path1) else path1[-1]
                    pos1_t_plus_1 = path1[t+1] if t + 1 < len(path1) else path1[-1]
                    pos2_t = path2[t] if t < len(path2) else path2[-1]
                    pos2_t_plus_1 = path2[t+1] if t + 1 < len(path2) else path2[-1]
                    if pos1_t == pos2_t_plus_1 and pos2_t == pos1_t_plus_1:
                        return (agent1, agent2, (pos1_t, pos1_t_plus_1), t)
        return None

    def find_all_conflicts(self, solution: Dict[int, List[Tuple[int, int]]]) -> int:
        """High-level Tie-Breaking을 위해 모든 충돌을 찾기."""
        NumOfConflicts = 0
        # conflicts = []
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
                        # conflicts.append((a1, a2, pos, t))
                        NumOfConflicts += 1
            
            for agent1 in self.agent_ids:  # edge conflict
                for agent2 in self.agent_ids:
                    if agent1 >= agent2: continue
                    path1, path2 = solution[agent1], solution[agent2]
                    pos1_t = path1[t] if t < len(path1) else path1[-1]
                    pos1_t_plus_1 = path1[t+1] if t + 1 < len(path1) else path1[-1]
                    pos2_t = path2[t] if t < len(path2) else path2[-1]
                    pos2_t_plus_1 = path2[t+1] if t + 1 < len(path2) else path2[-1]
                    if pos1_t == pos2_t_plus_1 and pos2_t == pos1_t_plus_1:
                        # conflicts.append((agent1, agent2, (pos1_t, pos1_t_plus_1), t))
                        NumOfConflicts += 1

        return NumOfConflicts

    def calculate_sic(self, solution: Dict[int, List[Tuple[int, int]]]) -> int:
        return sum(len(path) - 1 for path in solution.values())

    def pad_paths(self, solution: Dict[int, List[Tuple[int, int]]]):
        max_len = max(len(path) for path in solution.values()) if solution else 0
        padded_solution = {}
        for agent_id, path in solution.items():
            last_pos = path[-1]
            padded_path = path + [last_pos] * (max_len - len(path))
            padded_solution[agent_id] = padded_path
        return padded_solution

class BFS:
    """
    - goal 기반 distance field를 캐시하여 빠르게 경로를 추출
    - plan_all_paths(...)로 모든 AGV 경로를 한 번에 반환
    - plan_path(...)는 단일 경로 호환
    """
    def __init__(self, map_data: np.ndarray, seed: Optional[int] = None):
        assert map_data.ndim == 2, "map_data must be a 2D grid (H x W)"
        self.map = map_data
        self.H, self.W = map_data.shape
        self._distance_fields: Dict[Pos, np.ndarray] = {}  # goal -> field
        self._rng = random.Random(seed)

    # ---------- Public API ----------
    def plan_all_paths(
        self,
        positions: Dict[int, Pos],   # {agv_id: (x,y)}
        goals: Dict[int, Pos],       # {agv_id: (x,y)}
    ) -> Dict[int, List[Pos]]:
        """
        현재 모든 AGV의 start->goal 전체 경로를 반환.
        - 도달 불가면 [start] 반환
        - start==goal이면 [start]
        """
        paths: Dict[int, List[Pos]] = {}

        # 1) 유니크 goal에 대해 distance field 준비
        unique_goals = {g for g in goals.values() if g is not None}
        fields: Dict[Pos, Optional[np.ndarray]] = {}
        for g in unique_goals:
            fields[g] = self._get_or_build_field(g)  # None이면 goal이 벽/유효하지 않음

        # 2) 각 AGV 경로 추출
        for aid, start in positions.items():
            goal = goals.get(aid)
            if goal is None or not self._in_bounds(start):
                paths[aid] = [start]
                continue

            if start == goal:
                paths[aid] = [start]
                continue

            field = fields.get(goal)
            if field is None or field[start[1], start[0]] < 0:
                # goal이 벽이거나 start에서 goal 불가
                paths[aid] = [start]
                continue

            paths[aid] = self._extract_path(start, goal, field)

        return paths

    def plan_path(self, start: Pos, goal: Pos) -> List[Pos]:
        """
        단일 AGV 경로 반환(호환용).
        """
        if start == goal:
            return [start]
        field = self._get_or_build_field(goal)
        if field is None or not self._in_bounds(start) or field[start[1], start[0]] < 0:
            return [start]
        return self._extract_path(start, goal, field)

    def clear_cache(self):
        """모든 distance field 캐시 초기화(맵이 바뀌었을 때 사용)."""
        self._distance_fields.clear()

    # ---------- Internals ----------
    def _get_or_build_field(self, goal: Pos) -> Optional[np.ndarray]:
        if not self._in_bounds(goal) or self.map[goal[1], goal[0]] == 1:
            return None
        field = self._distance_fields.get(goal)
        if field is None:
            field = self._create_field_from_goal(goal)
            self._distance_fields[goal] = field
        return field

    def _create_field_from_goal(self, goal: Pos) -> np.ndarray:
        field = np.full((self.H, self.W), -1, dtype=int)
        gx, gy = goal
        field[gy, gx] = 0
        q = deque([goal])

        while q:
            x, y = q.popleft()
            curd = field[y, x]
            for nx, ny in self._get_neighbors((x, y)):
                if field[ny, nx] == -1:
                    field[ny, nx] = curd + 1
                    q.append((nx, ny))
        return field

    def _extract_path(self, start: Pos, goal: Pos, field: np.ndarray) -> List[Pos]:
        """
        distance가 '엄격히 감소'하는 이웃만 선택(plateau 방지).
        동순위 후보가 여러 개면 PRNG로 타이브레이크.
        """
        path = [start]
        cur = start
        while cur != goal:
            curd = field[cur[1], cur[0]]
            best: List[Pos] = []
            best_d = curd

            for nx, ny in self._get_neighbors(cur):
                d = field[ny, nx]
                if d >= 0 and d < best_d:
                    best = [(nx, ny)]
                    best_d = d
                elif d >= 0 and d == best_d:
                    best.append((nx, ny))

            if not best:
                # 이론상 발생 X(거리장은 단조 감소해야 함). 안전하게 중단.
                break

            cur = self._rng.choice(best)
            path.append(cur)

        return path

    def _get_neighbors(self, pos: Pos) -> List[Pos]:
        x, y = pos
        neigh = []
        for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)):
            nx, ny = x+dx, y+dy
            if self._in_bounds((nx,ny)) and self.map[ny, nx] == 0:
                neigh.append((nx, ny))
        return neigh

    def _in_bounds(self, p: Pos) -> bool:
        x, y = p
        return 0 <= x < self.W and 0 <= y < self.H