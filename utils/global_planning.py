import heapq
from typing import Dict, Tuple, Optional, Set, List, Sequence
import random
import time
import signal
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from bisect import bisect_left


Pos = Tuple[int, int]

class BFS:
    """
    [신규 클래스]
    목표 지점(goal) 기반의 거리장(Distance Field)을 미리 계산하여 경로를 매우 빠르게 추출하는 플래너.
    - 특정 goal에 대한 거리장은 BFS를 통해 단 한 번만 계산되고 캐시됩니다.
    - 경로 계획은 캐시된 거리장을 따라 가장 가파른 경사(steepest descent)를 찾는 방식으로 즉시 수행됩니다.
    - 재계획 기능은 없으며, 초기 경로 생성에 특화되어 있습니다.
    """
    def __init__(self, map_data: np.ndarray, rng=None):
        self.map = map_data
        self.H, self.W = map_data.shape
        self._distance_fields: Dict[Pos, np.ndarray] = {}  # goal -> distance_field 맵 캐시

        if rng is not None:
            self.rng = rng
        else:
            self.rng = random.Random()

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
            next_node = self.rng.choice(best_neighbors)
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

    def _is_free(self, p: Pos) -> bool:
        x, y = p
        return 0 <= x < self.W and 0 <= y < self.H and self.map[y, x] == 0

    def _random_center_in_range(self, sorted_vals: Sequence[int], a: int, b: int) -> Optional[int]:
        lo, hi = (a, b) if a <= b else (b, a)
        l = bisect_left(sorted_vals, lo)
        r = bisect_left(sorted_vals, hi + 1)
        if l >= r:
            return None
        return self.rng.choice(sorted_vals[l:r])

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
            if self.rng.random() < 0.5:
                on_left = on_right = False
            else:
                on_top = on_bottom = False

        # 어느 벽에도 속하지 않으면 중앙 기준으로 좌/우 강제 분류
        if not (on_top or on_bottom or on_left or on_right):
            mid_x = self.W // 2          # W=10이면 mid_x=5 (0~4 left, 5~9 right)
            if gx < mid_x:
                on_left = True
            else:
                on_right = True

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

    def find_path(self, constraints: Set, deadline: float | None = None):
        def is_edge(loc):
            return isinstance(loc, tuple) and len(loc) == 2 and isinstance(loc[0], tuple)

        vertex_constraints = {(loc, t) for (loc, t) in constraints if not is_edge(loc)}
        edge_constraints   = {(loc, t) for (loc, t) in constraints if is_edge(loc)}

        # start 제약 체크 (t=0)
        if (self.start, 0) in vertex_constraints:
            return None

        # goal 관련 제약이 있으면, 그 시간 "이후"에 goal에 있어야 함
        forbidden_goal_times = [t for (loc, t) in vertex_constraints if loc == self.goal]
        earliest_goal_time = (max(forbidden_goal_times) + 1) if forbidden_goal_times else 0

        # (f, tie, g, pos, time, path)
        open_list = [(self._manhattan_distance(self.start, self.goal), 0, 0, self.start, 0, [self.start])]
        visited = set()

        while open_list:
            if deadline is not None and time.perf_counter() > deadline:
                return None

            f, tie, g, current_pos, time, path = heapq.heappop(open_list)

            if (current_pos, time) in visited:
                continue
            visited.add((current_pos, time))

            # ✅ 현재 상태 자체가 vertex 제약이면 폐기
            if (current_pos, time) in vertex_constraints:
                continue

            # ✅ goal 도착 조건: 제약이 허용되는 시각 이후에만 성공 처리
            if current_pos == self.goal and time >= earliest_goal_time:
                return path

            for neighbor_pos in self._get_neighbors(current_pos):
                next_time = time + 1

                # vertex constraint
                if (neighbor_pos, next_time) in vertex_constraints:
                    continue

                # edge constraint (t -> t+1)
                if ((current_pos, neighbor_pos), time) in edge_constraints:
                    continue

                if (neighbor_pos, next_time) in visited:
                    continue

                new_g = g + 1
                h = self._manhattan_distance(neighbor_pos, self.goal)
                new_f = new_g + h
                new_tie = self.conflict_avoidance_table.get((neighbor_pos, next_time), 0)

                heapq.heappush(open_list, (new_f, new_tie, new_g, neighbor_pos, next_time, path + [neighbor_pos]))

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

    def solve(self, time_limit: float = 10.0):
        start_perf = time.perf_counter()

        # SIGALRM 핸들러
        def _alarm_handler(signum, frame):
            raise _CBSHardTimeout()

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, time_limit)  # ✅ 여기서부터 정확히 time_limit초 뒤 강제 인터럽트

        best_node_so_far = None

        try:
            # ---------------------------
            # 여기부터는 네 기존 solve() 내용 그대로 두면 됨
            # (단, 아래에서 best_node_so_far를 갱신하고 있었으니 그 라인만 유지)
            # ---------------------------
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
            best_node_so_far = root  # ✅ timeout 시 반환할 best

            while open_list:
                P = heapq.heappop(open_list)

                if P.num_conflicts < best_node_so_far.num_conflicts:
                    best_node_so_far = P

                conflict = self.find_first_conflict(P.solution)
                if conflict is None:
                    print(f"\n[CBS Solve] Optimal solution found in {time.time() - start_time:.2f} seconds.")
                    return self.pad_paths(P.solution)

                agent1, agent2, loc, conflict_time = conflict

                for agent_to_constrain in [agent1, agent2]:
                    new_constraint_loc = loc
                    if isinstance(loc, tuple) and len(loc) == 2 and isinstance(loc[0], tuple):
                        if agent_to_constrain == agent1:
                            new_constraint_loc = (loc[0], loc[1])
                        else:
                            new_constraint_loc = (loc[1], loc[0])

                    new_constraint = (agent_to_constrain, new_constraint_loc, conflict_time)

                    agent_constraints = self._get_constraints_for_agent(P, agent_to_constrain)
                    agent_constraints.add((new_constraint[1], new_constraint[2]))

                    start, goal = self.agents[agent_to_constrain]['start'], self.agents[agent_to_constrain]['goal']
                    other_agents_solution = {aid: p for aid, p in P.solution.items() if aid != agent_to_constrain}
                    planner = AStar_for_CBS(self.map, start, goal, other_agents_solution)

                    new_path = planner.find_path(agent_constraints)
                    if new_path is None:
                        continue

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
                    node_counter += 1
                    heapq.heappush(open_list, child_node)

            print(f"\n[CBS Solve] No solution found after {time.time() - start_time:.2f} seconds.")
            return None

        except _CBSHardTimeout:
            elapsed = time.perf_counter() - start_perf
            print(f"\n!!! CBS Timeout after {elapsed:.2f} seconds. !!!")
            if best_node_so_far is None:
                print("    > No partial solution.\n")
                return None
            print(f"    > Returning best found solution with {best_node_so_far.num_conflicts} conflicts.\n")
            # ✅ 여기서 pad_paths까지 하면 추가 시간이 걸릴 수 있음.
            #    “60초 딱”을 원하면 pad 없이 반환 추천:
            # return best_node_so_far.solution
            return self.pad_paths(best_node_so_far.solution)

        finally:
            # 타이머/핸들러 원복 (매우 중요)
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)


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


class _CBSHardTimeout(Exception):
    pass