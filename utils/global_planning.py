import heapq
import numpy as np

from pibt.dist_table import DistTable
from pibt.mapf_utils import get_neighbors

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


# PIBT Algorithm
class PIBT:
    def __init__(self, map, starts, goals, seed=0):
        """
        map: numpy.ndarray, 0=free, 1=obstacle
        starts: list of (x, y)
        goals: list of (x, y)
        """
        self.map = map
        self.starts = starts
        self.goals = goals
        self.N = len(starts)
        # PIBT expects grid[y, x] == True for free, False for obstacle
        self.grid = (map == 0)
        self.dist_tables = [DistTable(self.grid, goal) for goal in goals]
        self.rng = np.random.default_rng(seed)
        self.NIL = self.N
        self.NIL_COORD = (-1, -1)

    def step(self, Q_from, priorities):
        N = len(Q_from)
        Q_to = [self.NIL_COORD for _ in range(N)]
        occupied_now = {pos: i for i, pos in enumerate(Q_from)}
        occupied_nxt = {}

        # 우선순위 높은 AGV부터 이동 결정
        order = sorted(range(N), key=lambda i: priorities[i], reverse=True)
        for i in order:
            if Q_to[i] != self.NIL_COORD:
                continue
            # 후보 위치: 현 위치 + 인접 위치
            candidates = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
            self.rng.shuffle(candidates)
            candidates = sorted(candidates, key=lambda u: self.dist_tables[i].get(u))
            for v in candidates:
                # vertex collision
                if v in occupied_nxt:
                    continue
                j = occupied_now.get(v, self.NIL)
                # edge collision
                if j != self.NIL and Q_to[j] == Q_from[i]:
                    continue
                # 예약
                Q_to[i] = v
                occupied_nxt[v] = i
                # priority inheritance
                if (
                    j != self.NIL
                    and Q_to[j] == self.NIL_COORD
                ):
                    # 재귀적으로 j의 이동 결정
                    cand_j = [Q_from[j]] + get_neighbors(self.grid, Q_from[j])
                    self.rng.shuffle(cand_j)
                    cand_j = sorted(cand_j, key=lambda u: self.dist_tables[j].get(u))
                    for vj in cand_j:
                        if vj in occupied_nxt:
                            continue
                        jj = occupied_now.get(vj, self.NIL)
                        if jj != self.NIL and Q_to[jj] == Q_from[j]:
                            continue
                        Q_to[j] = vj
                        occupied_nxt[vj] = j
                        break
                break
            else:
                # 이동 실패 시 제자리
                Q_to[i] = Q_from[i]
                occupied_nxt[Q_from[i]] = i
        return Q_to

    def plan(self, max_timestep=1000):
        priorities = [self.dist_tables[i].get(self.starts[i]) / self.grid.size for i in range(self.N)]
        configs = [self.starts[:]]
        while len(configs) <= max_timestep:
            Q = self.step(configs[-1], priorities)
            configs.append(Q)
            # update priorities & goal check
            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break
        return configs