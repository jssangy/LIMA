#!/usr/bin/env python3
"""Generate graph-valid reference routes for LIMA Route Planner experiments.

The output format is the one consumed by ``lima --routes``: one line per
agent, containing whitespace-separated ``x y`` coordinate pairs.  The tool is
deliberately self-contained so every candidate is evaluated against the same
MovingAI parser, validation, fallback, and measurement code.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import tempfile
import time
from array import array
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


MASK64 = (1 << 64) - 1
DIRECTIONS = ((0, -1), (-1, 0), (1, 0), (0, 1))
PLANNERS = (
    "direct_bfs",
    "direct_astar",
    "jps",
    "randomized_shortest",
    "yen_k",
    "xy_dor",
    "yx_dor",
    "o1turn",
    "romm",
    "valiant",
    "swr",
    "static_highway",
    "static_guidance",
    "sui",
    "tfo_gp",
)
BOUNDED_STRETCH_PLANNERS = {
    "yen_k",
    "valiant",
    "swr",
    "static_highway",
    "static_guidance",
    "tfo_gp",
}

Cell = int
Route = list[Cell]


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def mixed_seed(seed: int, *values: int) -> int:
    state = seed & MASK64
    for value in values:
        state = splitmix64(state ^ (value & MASK64))
    return state


@dataclass(frozen=True)
class Task:
    start: Cell
    goal: Cell


@dataclass
class Grid:
    width: int
    height: int
    rows: list[str]
    free: list[bool]
    free_cells: list[Cell]
    neighbors: list[tuple[Cell, ...]]

    def cell(self, x: int, y: int) -> Cell:
        return y * self.width + x

    def coord(self, cell: Cell) -> tuple[int, int]:
        return cell % self.width, cell // self.width

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def traversable_xy(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.free[self.cell(x, y)]

    def traversable(self, cell: Cell) -> bool:
        return 0 <= cell < len(self.free) and self.free[cell]


def load_map(path: Path) -> Grid:
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline().split()
        height_line = stream.readline().split()
        width_line = stream.readline().split()
        marker = stream.readline().strip()
        if len(first) != 2 or first[0] != "type":
            raise ValueError("invalid MovingAI map type header")
        if len(height_line) != 2 or height_line[0] != "height":
            raise ValueError("invalid MovingAI map height header")
        if len(width_line) != 2 or width_line[0] != "width":
            raise ValueError("invalid MovingAI map width header")
        if marker != "map":
            raise ValueError("missing MovingAI map body marker")
        height = int(height_line[1])
        width = int(width_line[1])
        rows = [stream.readline().rstrip("\r\n") for _ in range(height)]
    if height <= 0 or width <= 0 or any(len(row) != width for row in rows):
        raise ValueError("invalid MovingAI map dimensions or row length")

    allowed_free = {".", "S", "E", "G"}
    allowed_blocked = {"@", "T"}
    free = [False] * (width * height)
    free_cells: list[Cell] = []
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if value not in allowed_free and value not in allowed_blocked:
                raise ValueError(f"unsupported map cell {value!r} at ({x},{y})")
            if value in allowed_free:
                cell = y * width + x
                free[cell] = True
                free_cells.append(cell)

    neighbors: list[tuple[Cell, ...]] = [()] * (width * height)
    for cell in free_cells:
        x, y = cell % width, cell // width
        adjacent: list[Cell] = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and free[ny * width + nx]:
                adjacent.append(ny * width + nx)
        neighbors[cell] = tuple(adjacent)
    return Grid(width, height, rows, free, free_cells, neighbors)


def load_scenario(path: Path, grid: Grid, agents: int) -> list[Task]:
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as stream:
        lines = iter(stream)
        first = next(lines, "")
        if not first.lower().startswith("version"):
            lines = iter((first, *lines))
        for line_number, line in enumerate(lines, start=2):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 8:
                raise ValueError(f"invalid scenario row {line_number}")
            width, height = int(fields[2]), int(fields[3])
            sx, sy, gx, gy = map(int, fields[4:8])
            if width != grid.width or height != grid.height:
                raise ValueError(f"scenario row {line_number} map dimensions do not match")
            if not grid.traversable_xy(sx, sy) or not grid.traversable_xy(gx, gy):
                raise ValueError(f"scenario row {line_number} has a blocked endpoint")
            tasks.append(Task(grid.cell(sx, sy), grid.cell(gx, gy)))
            if len(tasks) == agents:
                break
    if len(tasks) != agents:
        raise ValueError(f"scenario has {len(tasks)} tasks, requested {agents}")
    return tasks


def reconstruct(parent: dict[Cell, Cell], start: Cell, goal: Cell) -> Route | None:
    if start == goal:
        return [start]
    if goal not in parent:
        return None
    route = [goal]
    while route[-1] != start:
        route.append(parent[route[-1]])
    route.reverse()
    return route


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


class RouteGenerator:
    def __init__(self, grid: Grid, args: argparse.Namespace) -> None:
        self.grid = grid
        self.args = args
        self.distance_fields: OrderedDict[Cell, array] = OrderedDict()
        self.vertex_load: Counter[Cell] = Counter()
        self.edge_load: Counter[tuple[Cell, Cell]] = Counter()
        self.fallback_count = 0
        self.fallback_reasons: Counter[str] = Counter()
        self.center_xs, self.center_ys = self._intersection_axes()

    def _rng(self, agent: int, salt: int = 0) -> random.Random:
        return random.Random(mixed_seed(self.args.seed, agent, salt))

    def distance_field(self, goal: Cell) -> array:
        cached = self.distance_fields.get(goal)
        if cached is not None:
            self.distance_fields.move_to_end(goal)
            return cached
        distance = array("i", [-1]) * len(self.grid.free)
        distance[goal] = 0
        frontier: deque[Cell] = deque([goal])
        while frontier:
            current = frontier.popleft()
            candidate = distance[current] + 1
            for neighbor in self.grid.neighbors[current]:
                if distance[neighbor] >= 0:
                    continue
                distance[neighbor] = candidate
                frontier.append(neighbor)
        self.distance_fields[goal] = distance
        if len(self.distance_fields) > self.args.distance_cache:
            self.distance_fields.popitem(last=False)
        return distance

    def direct_bfs(self, start: Cell, goal: Cell) -> Route | None:
        distance = self.distance_field(goal)
        if distance[start] < 0:
            return None
        route = [start]
        current = start
        while current != goal:
            target_distance = distance[current] - 1
            following = next(
                (cell for cell in self.grid.neighbors[current] if distance[cell] == target_distance),
                None,
            )
            if following is None:
                return None
            route.append(following)
            current = following
        return route

    def randomized_shortest(
        self, start: Cell, goal: Cell, rng: random.Random
    ) -> Route | None:
        distance = self.distance_field(goal)
        if distance[start] < 0:
            return None
        route = [start]
        current = start
        while current != goal:
            target_distance = distance[current] - 1
            choices = [
                cell for cell in self.grid.neighbors[current] if distance[cell] == target_distance
            ]
            if not choices:
                return None
            current = choices[rng.randrange(len(choices))]
            route.append(current)
        return route

    def astar(
        self,
        start: Cell,
        goal: Cell,
        edge_cost: Callable[[Cell, Cell], float] | None = None,
        banned_nodes: set[Cell] | None = None,
        banned_edges: set[tuple[Cell, Cell]] | None = None,
    ) -> Route | None:
        if start == goal:
            return [start]
        banned_nodes = banned_nodes or set()
        banned_edges = banned_edges or set()
        if start in banned_nodes or goal in banned_nodes:
            return None
        gx, gy = self.grid.coord(goal)

        def heuristic(cell: Cell) -> int:
            x, y = self.grid.coord(cell)
            return abs(x - gx) + abs(y - gy)

        costs: dict[Cell, float] = {start: 0.0}
        parent: dict[Cell, Cell] = {}
        queue: list[tuple[float, float, Cell]] = [(float(heuristic(start)), -0.0, start)]
        while queue:
            _, negative_cost, current = heapq.heappop(queue)
            current_cost = -negative_cost
            if current_cost > costs.get(current, math.inf) + 1e-12:
                continue
            if current == goal:
                return reconstruct(parent, start, goal)
            for neighbor in self.grid.neighbors[current]:
                if neighbor in banned_nodes or (current, neighbor) in banned_edges:
                    continue
                step_cost = 1.0 if edge_cost is None else edge_cost(current, neighbor)
                candidate = current_cost + step_cost
                if candidate + 1e-12 >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = candidate
                parent[neighbor] = current
                heapq.heappush(
                    queue, (candidate + heuristic(neighbor), -candidate, neighbor)
                )
        return None

    def _forced_directions(self, cell: Cell, dx: int, dy: int) -> list[tuple[int, int]]:
        x, y = self.grid.coord(cell)
        forced: list[tuple[int, int]] = []
        if dx:
            if self.grid.traversable_xy(x, y - 1) and not self.grid.traversable_xy(x - dx, y - 1):
                forced.append((0, -1))
            if self.grid.traversable_xy(x, y + 1) and not self.grid.traversable_xy(x - dx, y + 1):
                forced.append((0, 1))
        else:
            if self.grid.traversable_xy(x - 1, y) and not self.grid.traversable_xy(x - 1, y - dy):
                forced.append((-1, 0))
            if self.grid.traversable_xy(x + 1, y) and not self.grid.traversable_xy(x + 1, y - dy):
                forced.append((1, 0))
        return forced

    def _jump(self, cell: Cell, dx: int, dy: int, goal: Cell) -> Cell | None:
        x, y = self.grid.coord(cell)
        gx, gy = self.grid.coord(goal)
        while True:
            x += dx
            y += dy
            if not self.grid.traversable_xy(x, y):
                return None
            current = self.grid.cell(x, y)
            if current == goal:
                return current
            if self._forced_directions(current, dx, dy):
                return current
            # In a 4-connected grid, goal-axis projections are the symmetry
            # breaking turn points needed in obstacle-free rectangles.
            if (dx and x == gx) or (dy and y == gy):
                return current

    def jps(self, start: Cell, goal: Cell) -> Route | None:
        if start == goal:
            return [start]
        gx, gy = self.grid.coord(goal)
        score: dict[Cell, int] = {start: 0}
        parent: dict[Cell, Cell] = {}
        incoming: dict[Cell, tuple[int, int]] = {}

        def heuristic(cell: Cell) -> int:
            x, y = self.grid.coord(cell)
            return abs(x - gx) + abs(y - gy)

        queue: list[tuple[int, int, Cell]] = [(heuristic(start), 0, start)]
        while queue:
            _, negative_g, current = heapq.heappop(queue)
            current_g = -negative_g
            if current_g != score.get(current):
                continue
            if current == goal:
                jump_points = [goal]
                while jump_points[-1] != start:
                    jump_points.append(parent[jump_points[-1]])
                jump_points.reverse()
                expanded = [start]
                for destination in jump_points[1:]:
                    x, y = self.grid.coord(expanded[-1])
                    tx, ty = self.grid.coord(destination)
                    dx = 0 if tx == x else (1 if tx > x else -1)
                    dy = 0 if ty == y else (1 if ty > y else -1)
                    if dx and dy:
                        return None
                    while (x, y) != (tx, ty):
                        x += dx
                        y += dy
                        expanded.append(self.grid.cell(x, y))
                return expanded

            x, y = self.grid.coord(current)
            if current == start:
                directions = list(DIRECTIONS)
            else:
                dx, dy = incoming[current]
                directions = [(dx, dy), *self._forced_directions(current, dx, dy)]
                if x == gx and y != gy:
                    directions.append((0, 1 if gy > y else -1))
                if y == gy and x != gx:
                    directions.append((1 if gx > x else -1, 0))
            seen: set[tuple[int, int]] = set()
            for dx, dy in directions:
                if (dx, dy) in seen:
                    continue
                seen.add((dx, dy))
                jumped = self._jump(current, dx, dy, goal)
                if jumped is None:
                    continue
                jx, jy = self.grid.coord(jumped)
                distance = abs(jx - x) + abs(jy - y)
                candidate = current_g + distance
                if candidate >= score.get(jumped, 1 << 60):
                    continue
                score[jumped] = candidate
                parent[jumped] = current
                incoming[jumped] = (dx, dy)
                heapq.heappush(queue, (candidate + heuristic(jumped), -candidate, jumped))
        return None

    def yen(self, start: Cell, goal: Cell, agent: int) -> Route | None:
        first = self.astar(start, goal)
        if first is None:
            return None
        shortest = len(first) - 1
        max_edges = max(shortest, math.floor(shortest * self.args.max_stretch + 1e-9))
        accepted: list[Route] = [first]
        accepted_keys = {tuple(first)}
        candidates: list[tuple[int, tuple[Cell, ...]]] = []
        candidate_keys: set[tuple[Cell, ...]] = set()
        for _ in range(1, self.args.yen_k):
            previous = accepted[-1]
            for spur_index in range(len(previous) - 1):
                root = previous[: spur_index + 1]
                banned_edges: set[tuple[Cell, Cell]] = set()
                for route in accepted:
                    if len(route) > spur_index and route[: spur_index + 1] == root:
                        banned_edges.add((route[spur_index], route[spur_index + 1]))
                spur = self.astar(
                    root[-1], goal, banned_nodes=set(root[:-1]), banned_edges=banned_edges
                )
                if spur is None:
                    continue
                candidate = tuple(root[:-1] + spur)
                if len(candidate) - 1 > max_edges:
                    continue
                if candidate in accepted_keys or candidate in candidate_keys:
                    continue
                heapq.heappush(candidates, (len(candidate), candidate))
                candidate_keys.add(candidate)
            if not candidates:
                break
            _, selected = heapq.heappop(candidates)
            candidate_keys.remove(selected)
            accepted.append(list(selected))
            accepted_keys.add(selected)
        rng = self._rng(agent, 0x59454E)
        return accepted[rng.randrange(len(accepted))]

    def axis_order(self, start: Cell, goal: Cell, x_first: bool) -> Route | None:
        sx, sy = self.grid.coord(start)
        gx, gy = self.grid.coord(goal)
        route = [start]
        x, y = sx, sy
        axes = (("x", gx), ("y", gy)) if x_first else (("y", gy), ("x", gx))
        for axis, target in axes:
            if axis == "x":
                step = 0 if target == x else (1 if target > x else -1)
                while x != target:
                    x += step
                    if not self.grid.traversable_xy(x, y):
                        return None
                    route.append(self.grid.cell(x, y))
            else:
                step = 0 if target == y else (1 if target > y else -1)
                while y != target:
                    y += step
                    if not self.grid.traversable_xy(x, y):
                        return None
                    route.append(self.grid.cell(x, y))
        return route

    @staticmethod
    def concatenate(first: Route | None, second: Route | None) -> Route | None:
        if first is None or second is None or not first or not second or first[-1] != second[0]:
            return None
        return first + second[1:]

    def o1turn(self, start: Cell, goal: Cell, agent: int) -> Route | None:
        rng = self._rng(agent, 0x4F3154)
        orders = [False, True] if rng.randrange(2) else [True, False]
        for x_first in orders:
            route = self.axis_order(start, goal, x_first)
            if route is not None:
                return route
        return None

    def romm(self, start: Cell, goal: Cell, agent: int) -> Route | None:
        sx, sy = self.grid.coord(start)
        gx, gy = self.grid.coord(goal)
        rng = self._rng(agent, 0x524F4D4D)
        for _ in range(self.args.waypoint_attempts):
            x = rng.randint(min(sx, gx), max(sx, gx))
            y = rng.randint(min(sy, gy), max(sy, gy))
            if not self.grid.traversable_xy(x, y):
                continue
            waypoint = self.grid.cell(x, y)
            first = self.axis_order(start, waypoint, bool(rng.randrange(2)))
            second = self.axis_order(waypoint, goal, bool(rng.randrange(2)))
            route = self.concatenate(first, second)
            if route is not None:
                return route
        return None

    def valiant(self, start: Cell, goal: Cell, agent: int) -> Route | None:
        shortest = self.distance_field(goal)[start]
        if shortest < 0:
            return None
        maximum = max(shortest, math.floor(shortest * self.args.max_stretch + 1e-9))
        rng = self._rng(agent, 0x56414C)
        for _ in range(self.args.waypoint_attempts):
            waypoint = self.grid.free_cells[rng.randrange(len(self.grid.free_cells))]
            first = self.direct_bfs(start, waypoint)
            second = self.direct_bfs(waypoint, goal)
            route = self.concatenate(first, second)
            if route is not None and len(route) - 1 <= maximum:
                return route
        return None

    def _intersection_axes(self) -> tuple[list[int], list[int]]:
        centers: list[tuple[int, int]] = []
        goals = {cell for cell in self.grid.free_cells if self.grid.rows[cell // self.grid.width][cell % self.grid.width] in {"S", "G"}}
        for cell in self.grid.free_cells:
            x, y = self.grid.coord(cell)
            open_neighbors = sum(
                self.grid.traversable_xy(x + dx, y + dy) for dx, dy in DIRECTIONS
            )
            if open_neighbors < 3:
                continue
            if any(
                self.grid.traversable_xy(x + dx, y + dy)
                for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ):
                continue
            arms = 0
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                cx, cy = x + dx, y + dy
                length = 0
                while self.grid.traversable_xy(cx, cy):
                    current = self.grid.cell(cx, cy)
                    if current in goals:
                        break
                    corridor = (
                        not self.grid.traversable_xy(cx - 1, cy)
                        and not self.grid.traversable_xy(cx + 1, cy)
                        if dy
                        else not self.grid.traversable_xy(cx, cy - 1)
                        and not self.grid.traversable_xy(cx, cy + 1)
                    )
                    if not corridor:
                        break
                    length += 1
                    cx += dx
                    cy += dy
                arms += int(length > 0)
            if arms >= 3:
                centers.append((x, y))
        return sorted({x for x, _ in centers}), sorted({y for _, y in centers})

    @staticmethod
    def _in_range(value: int, first: int, second: int) -> bool:
        return min(first, second) <= value <= max(first, second)

    def _nearest_axis(self, values: Sequence[int], first: int, second: int, target: int) -> int | None:
        in_range = [value for value in values if self._in_range(value, first, second)]
        choices = in_range or list(values)
        return min(choices, key=lambda value: (abs(value - target), value)) if choices else None

    def swr(self, start: Cell, goal: Cell, agent: int) -> Route | None:
        sx, sy = self.grid.coord(start)
        gx, gy = self.grid.coord(goal)
        rng = self._rng(agent, 0x535752)

        def via(waypoints: Iterable[tuple[int, int]]) -> Route | None:
            current = start
            route = [start]
            for x, y in waypoints:
                if not self.grid.traversable_xy(x, y):
                    return None
                target = self.grid.cell(x, y)
                if target == current:
                    continue
                segment = self.randomized_shortest(current, target, rng)
                if segment is None:
                    return None
                route.extend(segment[1:])
                current = target
            return route if current == goal else None

        vertical_goal = gy == 0 or gy + 1 == self.grid.height
        horizontal_goal = gx == 0 or gx + 1 == self.grid.width
        if not vertical_goal and not horizontal_goal:
            horizontal_goal = True
        if vertical_goal:
            aligned = self._nearest_axis(self.center_xs, sx, gx, sx)
            choices = [y for y in self.center_ys if self._in_range(y, sy, gy)]
            for _ in range(self.args.waypoint_attempts):
                if aligned is None or not choices:
                    break
                y = choices[rng.randrange(len(choices))]
                route = via(((aligned, sy), (aligned, y), (gx, y), (gx, gy)))
                if route is not None:
                    return route
        elif horizontal_goal:
            aligned = self._nearest_axis(self.center_ys, sy, gy, sy)
            choices = [x for x in self.center_xs if self._in_range(x, sx, gx)]
            for _ in range(self.args.waypoint_attempts):
                if aligned is None or not choices:
                    break
                x = choices[rng.randrange(len(choices))]
                route = via(((sx, aligned), (x, aligned), (x, gy), (gx, gy)))
                if route is not None:
                    return route
        return None

    def static_highway(self, start: Cell, goal: Cell) -> Route | None:
        def cost(source: Cell, destination: Cell) -> float:
            sx, sy = self.grid.coord(source)
            dx, dy = self.grid.coord(destination)
            if sx != dx:
                preferred = 1 if sy % 2 == 0 else -1
                against = (dx - sx) != preferred
            else:
                preferred = 1 if sx % 2 == 0 else -1
                against = (dy - sy) != preferred
            return 1.0 + (self.args.highway_penalty if against else 0.0)

        return self.astar(start, goal, edge_cost=cost)

    def static_guidance(self, start: Cell, goal: Cell) -> Route | None:
        def cost(source: Cell, destination: Cell) -> float:
            low, high = sorted((source, destination))
            preferred_low_to_high = mixed_seed(self.args.seed, low, high, 0x47554944) & 1
            low_to_high = source == low
            against = bool(preferred_low_to_high) != low_to_high
            return 1.0 + (self.args.guidance_weight if against else 0.0)

        return self.astar(start, goal, edge_cost=cost)

    def sui(self, start: Cell, goal: Cell, agent: int) -> Route | None:
        distance = self.distance_field(goal)
        if distance[start] < 0:
            return None
        rng = self._rng(agent, 0x535549)
        current = start
        route = [start]
        while current != goal:
            target_distance = distance[current] - 1
            choices = [
                cell for cell in self.grid.neighbors[current] if distance[cell] == target_distance
            ]
            if not choices:
                return None
            rng.shuffle(choices)
            current = min(
                choices,
                key=lambda cell: (
                    self.args.sui_vertex_weight * self.vertex_load[cell]
                    + self.args.sui_edge_weight * self.edge_load[(route[-1], cell)]
                ),
            )
            route.append(current)
        return route

    def tfo_gp(self, start: Cell, goal: Cell) -> Route | None:
        distance = self.distance_field(goal)
        shortest = distance[start]
        if shortest < 0:
            return None
        if shortest == 0:
            return [start]
        maximum = max(shortest, math.floor(shortest * self.args.max_stretch + 1e-9))

        def edge_cost(source: Cell, destination: Cell) -> float:
            return (
                1.0
                + self.args.tfo_vertex_weight * self.vertex_load[destination]
                + self.args.tfo_edge_weight * self.edge_load[(source, destination)]
                + self.args.tfo_contraflow_weight * self.edge_load[(destination, source)]
            )

        start_state = (start, 0)
        labels: dict[Cell, list[tuple[int, float]]] = {start: [(0, 0.0)]}
        parent: dict[tuple[Cell, int], tuple[Cell, int]] = {}
        queue: list[tuple[float, float, int, Cell]] = [(float(shortest), 0.0, 0, start)]
        goal_state: tuple[Cell, int] | None = None
        while queue:
            _, cost, steps, current = heapq.heappop(queue)
            if not any(s == steps and abs(c - cost) <= 1e-12 for s, c in labels.get(current, [])):
                continue
            if current == goal:
                goal_state = (current, steps)
                break
            for neighbor in self.grid.neighbors[current]:
                next_steps = steps + 1
                remaining = distance[neighbor]
                if remaining < 0 or next_steps + remaining > maximum:
                    continue
                candidate = cost + edge_cost(current, neighbor)
                existing = labels.get(neighbor, [])
                if any(s <= next_steps and c <= candidate + 1e-12 for s, c in existing):
                    continue
                labels[neighbor] = [
                    (s, c)
                    for s, c in existing
                    if not (next_steps <= s and candidate <= c + 1e-12)
                ]
                labels[neighbor].append((next_steps, candidate))
                state = (neighbor, next_steps)
                parent[state] = (current, steps)
                heapq.heappush(
                    queue,
                    (candidate + remaining, candidate, next_steps, neighbor),
                )
        if goal_state is None:
            return None
        route = [goal]
        state = goal_state
        while state != start_state:
            state = parent[state]
            route.append(state[0])
        route.reverse()
        return route

    def validate(self, route: Route | None, task: Task) -> str | None:
        if not route:
            return "empty"
        if route[0] != task.start:
            return "wrong_start"
        if route[-1] != task.goal:
            return "wrong_goal"
        for cell in route:
            if not self.grid.traversable(cell):
                return "blocked_cell"
        for source, destination in zip(route, route[1:]):
            if destination not in self.grid.neighbors[source]:
                return "not_4_connected"
        return None

    def plan(self, task: Task, agent: int) -> Route:
        planner = self.args.planner
        rng = self._rng(agent, 0x504C414E)
        if planner == "direct_bfs":
            route = self.direct_bfs(task.start, task.goal)
        elif planner == "direct_astar":
            route = self.astar(task.start, task.goal)
        elif planner == "jps":
            route = self.jps(task.start, task.goal)
        elif planner == "randomized_shortest":
            route = self.randomized_shortest(task.start, task.goal, rng)
        elif planner == "yen_k":
            route = self.yen(task.start, task.goal, agent)
        elif planner == "xy_dor":
            route = self.axis_order(task.start, task.goal, True)
        elif planner == "yx_dor":
            route = self.axis_order(task.start, task.goal, False)
        elif planner == "o1turn":
            route = self.o1turn(task.start, task.goal, agent)
        elif planner == "romm":
            route = self.romm(task.start, task.goal, agent)
        elif planner == "valiant":
            route = self.valiant(task.start, task.goal, agent)
        elif planner == "swr":
            route = self.swr(task.start, task.goal, agent)
        elif planner == "static_highway":
            route = self.static_highway(task.start, task.goal)
        elif planner == "static_guidance":
            route = self.static_guidance(task.start, task.goal)
        elif planner == "sui":
            route = self.sui(task.start, task.goal, agent)
        elif planner == "tfo_gp":
            route = self.tfo_gp(task.start, task.goal)
        else:  # argparse choices make this unreachable.
            raise ValueError(f"unknown planner {planner}")

        problem = self.validate(route, task)
        if problem is None and planner in BOUNDED_STRETCH_PLANNERS:
            shortest = self.distance_field(task.goal)[task.start]
            maximum = max(
                shortest, math.floor(shortest * self.args.max_stretch + 1e-9)
            )
            if route is not None and len(route) - 1 > maximum:
                problem = "stretch_bound"
        if problem is not None:
            self.fallback_count += 1
            self.fallback_reasons[problem] += 1
            route = self.direct_bfs(task.start, task.goal)
            fallback_problem = self.validate(route, task)
            if fallback_problem is not None:
                raise RuntimeError(
                    f"agent {agent}: deterministic BFS fallback failed ({fallback_problem})"
                )
        assert route is not None
        return route

    def note_route(self, route: Route) -> None:
        self.vertex_load.update(route)
        self.edge_load.update(zip(route, route[1:]))


def route_statistics(
    grid: Grid,
    generator: RouteGenerator,
    tasks: Sequence[Task],
    routes: Sequence[Route],
    wall_seconds: float,
) -> dict[str, object]:
    lengths = [len(route) - 1 for route in routes]
    shortest = [generator.distance_field(task.goal)[task.start] for task in tasks]
    stretches = [
        1.0 if base == 0 else length / base for length, base in zip(lengths, shortest)
    ]
    vertex_load: Counter[Cell] = Counter()
    directed_load: Counter[tuple[Cell, Cell]] = Counter()
    for route in routes:
        vertex_load.update(route)
        directed_load.update(zip(route, route[1:]))
    contraflow = 0
    contraflow_pairs = 0
    examined: set[tuple[Cell, Cell]] = set()
    for source, destination in directed_load:
        edge = (min(source, destination), max(source, destination))
        if edge in examined:
            continue
        examined.add(edge)
        opposed = min(
            directed_load[(edge[0], edge[1])], directed_load[(edge[1], edge[0])]
        )
        contraflow += opposed
        contraflow_pairs += int(opposed > 0)
    route_length_sum = sum(lengths)
    return {
        "planner": generator.args.planner,
        "agents": len(routes),
        "seed": generator.args.seed,
        "planning_wall_seconds": wall_seconds,
        "route_length_sum": route_length_sum,
        "mean_route_length": route_length_sum / len(routes),
        "mean_stretch": sum(stretches) / len(stretches),
        "max_stretch": max(stretches),
        "max_vertex_load": max(vertex_load.values(), default=0),
        "max_directed_edge_load": max(directed_load.values(), default=0),
        "contraflow": contraflow,
        "contraflow_edge_pairs": contraflow_pairs,
        "contraflow_ratio": 0.0 if route_length_sum == 0 else contraflow / route_length_sum,
        "fallback_count": generator.fallback_count,
        "fallback_reasons": dict(sorted(generator.fallback_reasons.items())),
        "validated_routes": len(routes),
        "map_width": grid.width,
        "map_height": grid.height,
        "parameters": {
            "yen_k": generator.args.yen_k,
            "max_stretch_bound": generator.args.max_stretch,
            "waypoint_attempts": generator.args.waypoint_attempts,
            "highway_penalty": generator.args.highway_penalty,
            "guidance_weight": generator.args.guidance_weight,
            "sui_vertex_weight": generator.args.sui_vertex_weight,
            "sui_edge_weight": generator.args.sui_edge_weight,
            "tfo_vertex_weight": generator.args.tfo_vertex_weight,
            "tfo_edge_weight": generator.args.tfo_edge_weight,
            "tfo_contraflow_weight": generator.args.tfo_contraflow_weight,
        },
    }


def analyze_route_file(
    map_path: Path,
    scenario_path: Path,
    agents: int,
    route_path: Path,
    planner: str,
    seed: int,
) -> dict[str, object]:
    """Validate and measure an externally produced route set (for CBS)."""
    grid = load_map(map_path)
    tasks = load_scenario(scenario_path, grid, agents)
    lines = route_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != agents:
        raise ValueError(f"route file has {len(lines)} lines, expected {agents}")

    routes: list[Route] = []
    for agent, (line, task) in enumerate(zip(lines, tasks)):
        fields = line.split()
        if len(fields) < 2 or len(fields) % 2:
            raise ValueError(f"agent {agent}: malformed route line")
        values = [int(value) for value in fields]
        route: Route = []
        for index in range(0, len(values), 2):
            x, y = values[index], values[index + 1]
            if not grid.traversable_xy(x, y):
                raise ValueError(f"agent {agent}: route leaves traversable space")
            route.append(grid.cell(x, y))
        routes.append(route)

    analysis_args = argparse.Namespace(
        planner=planner,
        seed=seed,
        distance_cache=128,
        yen_k=4,
        max_stretch=1.5,
        waypoint_attempts=32,
        highway_penalty=0.25,
        guidance_weight=0.20,
        sui_vertex_weight=1.0,
        sui_edge_weight=0.25,
        tfo_vertex_weight=0.25,
        tfo_edge_weight=0.50,
        tfo_contraflow_weight=2.0,
    )
    generator = RouteGenerator(grid, analysis_args)
    for agent, (task, route) in enumerate(zip(tasks, routes)):
        problem = generator.validate(route, task)
        if problem is not None:
            raise ValueError(f"agent {agent}: invalid route ({problem})")
    statistics = route_statistics(grid, generator, tasks, routes, 0.0)
    statistics["planning_wall_seconds"] = None
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--planner", choices=PLANNERS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--yen-k", type=int, default=4)
    parser.add_argument("--max-stretch", type=float, default=1.5)
    parser.add_argument("--waypoint-attempts", type=int, default=32)
    parser.add_argument("--distance-cache", type=int, default=128)
    parser.add_argument("--highway-penalty", type=float, default=0.25)
    parser.add_argument("--guidance-weight", type=float, default=0.20)
    parser.add_argument("--sui-vertex-weight", type=float, default=1.0)
    parser.add_argument("--sui-edge-weight", type=float, default=0.25)
    parser.add_argument("--tfo-vertex-weight", type=float, default=0.25)
    parser.add_argument("--tfo-edge-weight", type=float, default=0.50)
    parser.add_argument("--tfo-contraflow-weight", type=float, default=2.0)
    args = parser.parse_args()
    if args.agents <= 0:
        parser.error("--agents must be positive")
    if args.yen_k <= 0:
        parser.error("--yen-k must be positive")
    if args.max_stretch < 1.0:
        parser.error("--max-stretch must be at least 1.0")
    if args.waypoint_attempts <= 0 or args.distance_cache <= 0:
        parser.error("attempt and cache counts must be positive")
    for name in (
        "highway_penalty",
        "guidance_weight",
        "sui_vertex_weight",
        "sui_edge_weight",
        "tfo_vertex_weight",
        "tfo_edge_weight",
        "tfo_contraflow_weight",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    grid = load_map(args.map)
    tasks = load_scenario(args.scenario, grid, args.agents)
    generator = RouteGenerator(grid, args)
    started = time.perf_counter()
    routes: list[Route] = []
    for agent, task in enumerate(tasks):
        route = generator.plan(task, agent)
        generator.note_route(route)
        routes.append(route)
    wall_seconds = time.perf_counter() - started

    # A final independent validation pass guards both route algorithms and the
    # output serializer against malformed reference routes.
    for agent, (task, route) in enumerate(zip(tasks, routes)):
        problem = generator.validate(route, task)
        if problem is not None:
            raise RuntimeError(f"agent {agent}: final route validation failed ({problem})")

    route_lines = []
    for route in routes:
        fields: list[str] = []
        for cell in route:
            x, y = grid.coord(cell)
            fields.extend((str(x), str(y)))
        route_lines.append(" ".join(fields))
    atomic_write(args.output, "\n".join(route_lines) + "\n")

    statistics = route_statistics(grid, generator, tasks, routes, wall_seconds)
    statistics.update(
        {
            "map": str(args.map),
            "scenario": str(args.scenario),
            "output": str(args.output),
        }
    )
    encoded = json.dumps(statistics, sort_keys=True, separators=(",", ":"))
    if args.stats_output is not None:
        atomic_write(args.stats_output, json.dumps(statistics, indent=2, sort_keys=True) + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
