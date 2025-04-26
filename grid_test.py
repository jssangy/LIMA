import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

import Funct
from map import map


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(pos, grid_map):
    x, y = pos
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_map.shape[1] and 0 <= ny < grid_map.shape[0]:
            if grid_map[ny][nx] in (1, 2):  # 이동 가능 or 목표
                neighbors.append((nx, ny))
    return neighbors


class DStarLiteOnGrid:
    def __init__(self, grid_map, start, goal):
        self.map = grid_map
        self.start = start
        self.goal = goal
        self.rhs = {}
        self.g = {}
        self.queue = []

        h, w = grid_map.shape
        for y in range(h):
            for x in range(w):
                if grid_map[y][x] in (1, 2):  # 이동 가능한 셀
                    self.rhs[(x, y)] = float('inf')
                    self.g[(x, y)] = float('inf')

        self.rhs[goal] = 0
        self.insert(goal)

    def insert(self, node):
        heapq.heappush(self.queue, (self.calculate_key(node), node))

    def calculate_key(self, node):
        g_rhs = min(self.g[node], self.rhs[node])
        return (g_rhs + manhattan(self.start, node), g_rhs)

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(
                self.g.get(v, float('inf')) + 1 for v in get_neighbors(u, self.map)
            )
        self.queue = [(k, n) for (k, n) in self.queue if n != u]
        heapq.heapify(self.queue)
        if self.g[u] != self.rhs[u]:
            self.insert(u)

    def compute_shortest_path(self):
        while self.queue and (self.queue[0][0] < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]):
            _, u = heapq.heappop(self.queue)
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
            for s in get_neighbors(u, self.map):
                self.update_vertex(s)

    def extract_path(self):
        path = [self.start]
        current = self.start
        while current != self.goal:
            neighbors = get_neighbors(current, self.map)
            if not neighbors:
                return []
            current = min(neighbors, key=lambda n: self.g.get(n, float('inf')))
            if self.g.get(current, float('inf')) == float('inf'):
                return []
            path.append(current)
        return path


class Controller:
    def __init__(self):
        self.map = map

    def graphing(self):
        self.graph = {}
        for x in range(100):
            for y in range(100):
                if self.map[y][x] == 6:
                    neighbors = self.find_neighbors(x, y)
                    self.graph[(x, y)] = {n: Funct.get_distance((x, y), n) for n in neighbors}
                if isinstance(self.map[y][x], str):
                    neighbors = self.find_neighbors(x, y, False)
                    self.graph[(x, y)] = {n: Funct.get_distance((x, y), n) for n in neighbors}

    def find_neighbors(self, x, y, rout=True):
        line_list = []
        for direction in ['up', 'down', 'left', 'right']:
            dx, dy = {'up': (1, 0), 'down': (-1, 0), 'right': (0, 1), 'left': (0, -1)}[direction]
            distance, poss_x, poss_y = 0, x, y
            while distance < 15 and 1 <= poss_x < 99 and 1 <= poss_y < 99:
                poss_x += dx
                poss_y += dy
                distance += 1
                val = self.map[poss_y][poss_x]
                if val == 1:
                    break
                if val == 6 or (isinstance(val, str) and rout):
                    line_list.append((poss_x, poss_y))
                    break
        return line_list

    def generate_grid_from_graph_with_goal(self):
        height, width = len(self.map), len(self.map[0])
        grid = np.zeros((height, width), dtype=np.uint8)
        white_cells = set(self.graph.keys())

        for start, neighbors in self.graph.items():
            for end in neighbors:
                x0, y0 = start
                x1, y1 = end
                steps = max(abs(x1 - x0), abs(y1 - y0))
                for i in range(steps + 1):
                    xi = x0 + round((x1 - x0) * i / steps)
                    yi = y0 + round((y1 - y0) * i / steps)
                    white_cells.add((xi, yi))

        for (x, y) in white_cells:
            if 0 <= x < width and 0 <= y < height:
                if isinstance(self.map[y][x], str):
                    grid[y][x] = 2
                else:
                    grid[y][x] = 1
        plt.imshow(grid, cmap=ListedColormap(['black', 'white', 'red']))
        return grid

    def visualize_path(self, grid, path, agv_pos=None, goal_pos=None):
        color_grid = np.copy(grid)
        for (x, y) in path:
            color_grid[y][x] = 4
        if agv_pos:
            color_grid[agv_pos[1]][agv_pos[0]] = 3
        if goal_pos:
            color_grid[goal_pos[1]][goal_pos[0]] = 2

        cmap = ListedColormap(['black', 'white', 'red', 'blue', 'green'])
        plt.figure(figsize=(10, 10))
        plt.imshow(color_grid, cmap=cmap)
        plt.grid(True, color='gray', linewidth=0.5)
        plt.xticks(np.arange(grid.shape[1]))
        plt.yticks(np.arange(grid.shape[0]))
        plt.title("D* Lite Path on Grid Map")
        plt.show()

    def animate_path_on_grid(self, grid_map, path, goal_pos):
        cmap = ListedColormap(['black', 'white', 'red', 'blue', 'green'])
        fig, ax = plt.subplots(figsize=(8, 8))
        ims = []

        for (x, y) in path:
            frame = np.copy(grid_map)
            frame[y][x] = 4
            frame[goal_pos[1]][goal_pos[0]] = 2

            im = ax.imshow(frame, cmap=cmap, animated=True)

            dist = planner.g.get((x, y), float('inf'))

            # 하단 텍스트를 axes 좌표계 기준으로 추가
            dist_text = ax.text(
                0.01, -0.07, f"Distance to goal (g): {dist:.1f}",
                transform=ax.transAxes, fontsize=12, color='white',
                verticalalignment='top', animated=True
            )

            ims.append([im, dist_text])

        ani = animation.ArtistAnimation(fig, ims, interval=0, blit=True, repeat=False)
        plt.title("D* Lite Path Traversal with Real-Time Distance")
        plt.grid(True, color='gray', linewidth=0.5)
        plt.show()


ctrl = Controller()
ctrl.graphing()
grid = ctrl.generate_grid_from_graph_with_goal()

start = (66, 96)
goal = next(((x, y) for y in range(100) for x in range(100) if grid[y][x] == 2), None)

planner = DStarLiteOnGrid(grid, start, goal)
planner.compute_shortest_path()
path = planner.extract_path()

ctrl.visualize_path(grid, path, agv_pos=start, goal_pos=goal)
ctrl.animate_path_on_grid(grid, path, goal_pos=goal)
