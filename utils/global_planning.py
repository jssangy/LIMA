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
    [New Class]
    A planner that extracts paths very quickly by pre-calculating a goal-based Distance Field.
    - The distance field for a specific goal is calculated only once via BFS and cached.
    - Path planning is performed immediately by finding the steepest descent along the cached distance field.
    - There is no replanning function, and it is specialized for initial path generation.
    """
    def __init__(self, map_data: np.ndarray, rng=None):
        self.map = map_data
        self.H, self.W = map_data.shape
        self._distance_fields: Dict[Pos, np.ndarray] = {}  # goal -> distance_field map cache

        if rng is not None:
            self.rng = rng
        else:
            self.rng = random.Random()

    def plan_path(self, start: Pos, goal: Pos) -> List[Pos]:
        """
        Extracts a path for the given start and goal points.
        Generates and caches a distance field for the goal point if necessary.
        """
        if start == goal:
            return [start]

        # 1. Get or generate a distance field for the goal point.
        if goal not in self._distance_fields:
            self._distance_fields[goal] = self._create_field_from_goal(goal)
        
        distance_field = self._distance_fields[goal]

        # 2. Extract a path along the generated distance field.
        path = [start]
        current = start
        
        # Check if the start point is unreachable
        if distance_field[current[1], current[0]] < 0:
            print(f"Warning: Start position {start} is unreachable from goal {goal}.")
            return [start] # Return stay-in-place path if unreachable

        while current != goal:
            neighbors = self._get_neighbors(current)
            if not neighbors:
                return path # Dead end

            # [Modified Start] Randomly select when there are multiple optimal paths with the same cost
            
            # 1. Calculate distance field values for all neighbors
            distances = {n: distance_field[n[1], n[0]] for n in neighbors}
            
            # 2. Find the minimum distance value
            min_dist = min(distances.values())

            # If only unreachable places (-1) remain
            if min_dist < 0:
                print(f"Warning: Path extraction stuck at {current} (surrounded by unreachable cells).")
                return path

            # 3. Collect all neighbor nodes with the minimum distance as candidates
            best_neighbors = [n for n, dist in distances.items() if dist == min_dist]
            
            # 4. Randomly select one from the candidates
            next_node = self.rng.choice(best_neighbors)
            # [Modified End]
            
            # If no further progress can be made (all neighbors are further than current)
            if distance_field[next_node[1], next_node[0]] >= distance_field[current[1], current[0]]:
                 print(f"Warning: Path extraction stuck at {current} for goal {goal}.")
                 return path

            current = next_node
            path.append(current)
            
        return path

    def _create_field_from_goal(self, goal: Pos) -> np.ndarray:
        """
        Generates a distance field by executing reverse BFS from the goal point.
        Obstacles or unreachable areas are marked as -1.
        """
        field = np.full((self.H, self.W), -1, dtype=int)
        gx, gy = goal
        
        if not (0 <= gx < self.W and 0 <= gy < self.H) or self.map[gy, gx] == 1:
            return field # If the goal is outside the map or a wall

        q = deque([goal])
        field[gy, gx] = 0
        
        while q:
            x, y = q.popleft()
            current_dist = field[y, x]
            
            for nx, ny in self._get_neighbors((x, y)):
                if field[ny, nx] == -1: # Not visited yet
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

        # If no candidate in range, closest value from the whole set
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
        # Use straight line if possible, otherwise use existing BFS (distance field)
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

    def plan_path_highway(self, start: Pos, goal: Pos, center_xs: list[int], center_ys: list[int],tries: int=8):
        if start == goal:
            return [start]

        sx, sy = start
        gx, gy = goal

        # Determine if goal is a wall
        on_top = (gy == 0)
        on_bottom = (gy == self.H - 1)
        on_left = (gx == 0)
        on_right = (gx == self.W - 1)

        # If it's a corner, randomly select one direction
        if (on_top or on_bottom) and (on_left or on_right):
            if self.rng.random() < 0.5:
                on_left = on_right = False
            else:
                on_top = on_bottom = False

        # If it doesn't belong to any wall, forcibly classify as left/right based on the center
        if not (on_top or on_bottom or on_left or on_right):
            mid_x = self.W // 2          # If W=10, mid_x=5 (0~4 left, 5~9 right)
            if gx < mid_x:
                on_left = True
            else:
                on_right = True

        # CASE A: Top/bottom wall -> y_rand random (center y), x_align is the nearest center x
        if on_top or on_bottom:
            x_align = self._nearest_center_in_range(center_xs, sx, gx, sx)
            if x_align is None:
                return self.plan_path(start, goal)

            for _ in range(tries):
                y_rand = self._random_center_in_range(center_ys, sy, gy)
                if y_rand is None:
                    break

                # Waypoint configuration (for maintaining shape)
                waypoints = [(x_align, sy), (x_align, y_rand), (gx, y_rand), goal]

                path = self._plan_via(start, waypoints)
                if path is not None:
                    return path

            return self.plan_path(start, goal)

        # CASE B: Left/right wall -> x_rand random (center x), y_align is the nearest center y
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

        # If goal is not a wall, just use basic
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
        """Create a Conflict Avoidance Table (CAT) for Tie-Breaking"""
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
        # [Modified] 5-way movement: wait (0,0) and 4 directions
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

        # Check start constraint (t=0)
        if (self.start, 0) in vertex_constraints:
            return None

        # If there are goal-related constraints, must be at goal "after" that time
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

            # ✅ Discard if the current state itself is a vertex constraint
            if (current_pos, time) in vertex_constraints:
                continue

            # ✅ Goal arrival condition: process as success only after the time allowed by constraints
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
    """Node of the Conflict Tree (CT). A method of collecting constraints along the parent."""
    cost: int
    # Add conflict count as the second sorting criterion for Tie-Breaking
    num_conflicts: int = field(compare=True)

    node_id: int = field(compare=True)

    solution: Dict[int, List[Tuple[int, int]]] = field(compare=False)
    constraint: Optional[Tuple[int, Tuple, int]] = field(compare=False, default=None)
    parent: Optional['CTNode'] = field(compare=False, default=None)

class CBS:
    """Conflict-Based Search (CBS) algorithm implementation (memory-efficient method)"""
    def __init__(self, map_data: np.ndarray, agents: Dict[int, Dict[str, Tuple[int, int]]]):
        self.map = map_data
        self.agents = agents
        self.agent_ids = list(agents.keys())

    def _get_constraints_for_agent(self, node: CTNode, agent_id: int) -> Set:
        """
        [Modified] Recursively search the parent of CTNode and extract only 'loc' and 'time'.
        """
        constraints = set()
        curr = node
        while curr is not None:
            if curr.constraint and curr.constraint[0] == agent_id:
                # constraint = (agent_id, loc, time)
                # Add (loc, time) to the constraints set
                constraints.add((curr.constraint[1], curr.constraint[2]))
            curr = curr.parent
        return constraints

    def solve(self, time_limit: float = 10.0):
        start_perf = time.perf_counter()

        # SIGALRM handler
        def _alarm_handler(signum, frame):
            raise _CBSHardTimeout()

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, time_limit)  # ✅ Force interrupt exactly time_limit seconds from here

        best_node_so_far = None

        try:
            # ---------------------------
            # From here, you can leave your existing solve() content as is
            # (However, since best_node_so_far was being updated below, keep only that line)
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
            best_node_so_far = root  # ✅ best to return on timeout

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
            # ✅ Doing pad_paths here may take additional time.
            #    If you want "exactly 60 seconds", recommended to return without padding:
            # return best_node_so_far.solution
            return self.pad_paths(best_node_so_far.solution)

        finally:
            # Restore timer/handler (very important)
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)


    def find_first_conflict(self, solution: Dict[int, List[Tuple[int, int]]]):
        max_len = max(len(p) for p in solution.values()) if solution else 0
        
        # [Modified] Since path is 0-indexed, iterate up to max_len (considering padding)
        for t in range(max_len):
            positions_at_t = defaultdict(list)
            for agent_id, path in solution.items():
                pos = path[t] if t < len(path) else path[-1]
                positions_at_t[pos].append(agent_id)

            for pos, agents in positions_at_t.items():  # vertex conflict
                if len(agents) > 1:
                    return (agents[0], agents[1], pos, t) # (A1, A2, (x,y), t)

            # Edge conflict (swaps)
            # [Modified] Edge conflict occurs in t -> t+1 movement.
            # Check up to range(max_len - 1) so that t+1 does not exceed max_len
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
        [Modified] Sum-of-Costs is the sum of 'times' to reach the goal.
        If path length is L, then t=0, 1, ..., L-1, so cost is L-1.
        However, A* may return a path that waits longer at the goal point due to constraints.
        (e.g., a 10-second path, but if it waits until t=12 due to constraints, there will be 13 path points)
        The exact cost is the "time to reach the goal".
        
        The path returned by A* is in the form (start, ... , goal, [goal, ...]).
        The cost is correctly (path length - 1).
        """
        cost = 0
        for path in solution.values():
            if not path: continue
            # Assume the end of the path is the goal point
            goal = path[-1]
            last_goal_time = 0
            for t, pos in enumerate(path):
                if pos == goal:
                    last_goal_time = t
            
            # If path is (start) -> (goal) [t=1] -> (goal) [t=2], then len=3, cost=2.
            # But if (start) -> (other) [t=1] -> (goal) [t=2], then len=3, cost=2.
            # (start) -> (goal) [t=1] -> (other) [t=2] -> (goal) [t=3], then len=4, cost=3.
            
            # According to the paper, the cost is the "time of last arrival at the goal".
            # Since the end of the path returned by A* is always the goal, len(path) - 1 is that time.
            cost += len(path) - 1
        return cost


    def pad_paths(self, solution: Dict[int, List[Tuple[int, int]]]):
        max_len = max(len(path) for path in solution.values()) if solution else 0
        padded_solution = {}
        for agent_id, path in solution.items():
            if not path: # Emergency situation where there is no path
                start_pos = self.agents[agent_id]['start']
                padded_solution[agent_id] = [start_pos] * max_len
                continue
                
            last_pos = path[-1]
            padded_path = path + [last_pos] * (max_len - len(path))
            padded_solution[agent_id] = padded_path
        return padded_solution


class _CBSHardTimeout(Exception):
    pass