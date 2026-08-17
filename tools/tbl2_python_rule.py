#!/usr/bin/env python3
"""Reproduce Table 2 using the ORIGINAL Python simulator's own definitions.

Recovered from Environment.py (deleted in commit 1bf5a35, read from history):

  _load_map          '@','T' -> wall ; '.','E','S' -> road ; every 'S' is a goal
  _get_goal_bbox     x range spanned by the goals, y is the full height
  _count_walkable_in_bbox
                     road cells with x_min < x < x_max, minus goals inside that
                     range  ("LaCAM style", boundaries excluded)
  _find_intersection_center
                     3x3 kernel match against a 4-way plus and the four T shapes
  _ray_len           arm walk that only continues while the corridor is single
                     lane (walls on both flanks) and stops before a goal cell
  intersection       a matched center with at least three non-empty arms

Usage: tbl2_python_rule.py <map> [<map> ...]
"""
import sys
from pathlib import Path

WALL, ROAD = 1, 0

PLUS4 = ((1, 0, 1), (0, 0, 0), (1, 0, 1))
T_NO_N = ((1, 1, 1), (0, 0, 0), (1, 0, 1))
T_NO_E = ((1, 0, 1), (0, 0, 1), (1, 0, 1))
T_NO_S = ((1, 0, 1), (0, 0, 0), (1, 1, 1))
T_NO_W = ((1, 0, 1), (1, 0, 0), (1, 0, 1))
KERNELS = (PLUS4, T_NO_N, T_NO_E, T_NO_S, T_NO_W)


def load_map(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "map":
            start = i + 1
            break
    if start is None:
        raise ValueError(f"no map section in {path}")
    grid, goals = [], set()
    for line in lines[start:]:
        row = []
        for ch in line.strip():
            if ch in "@T":
                row.append(WALL)
            elif ch in ".ES":
                row.append(ROAD)
                if ch == "S":
                    goals.add((len(row) - 1, len(grid)))
            else:
                raise ValueError(f"invalid map char {ch!r}")
        if row:
            grid.append(row)
    return grid, goals


def count_walkable_in_bbox(grid, goals):
    h, w = len(grid), len(grid[0])
    if not goals:
        walkable = sum(1 for row in grid for v in row if v == ROAD)
        return walkable - sum(1 for (x, y) in goals if grid[y][x] == ROAD)
    xs = [x for x, _ in goals]
    x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
    left, right = x_min + 1, x_max          # slice end excluded -> x < x_max
    if left >= right:
        return 0
    walkable = sum(1 for row in grid for v in row[left:right] if v == ROAD)
    inside = sum(1 for (x, y) in goals if left <= x < right and grid[y][x] == ROAD)
    return walkable - inside


def ray_len(grid, goals, r, c, dr, dc):
    h, w = len(grid), len(grid[0])
    length = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == ROAD:
        if (cc, rr) in goals:
            break
        if dr != 0:                                  # vertical arm, check flanks
            left_wall = cc - 1 < 0 or grid[rr][cc - 1] == WALL
            right_wall = cc + 1 >= w or grid[rr][cc + 1] == WALL
            if not (left_wall and right_wall):
                break
        else:                                        # horizontal arm
            up_wall = rr - 1 < 0 or grid[rr - 1][cc] == WALL
            down_wall = rr + 1 >= h or grid[rr + 1][cc] == WALL
            if not (up_wall and down_wall):
                break
        length += 1
        rr += dr
        cc += dc
    return length


def analyse(path):
    grid, goals = load_map(path)
    h, w = len(grid), len(grid[0])
    tiles = count_walkable_in_bbox(grid, goals)

    centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            window = tuple(tuple(grid[r + dr][c + dc] for dc in (-1, 0, 1))
                           for dr in (-1, 0, 1))
            if any(window == k for k in KERNELS):
                centers.append((r, c))

    inter = {}
    for r, c in centers:
        lens = {
            "N": ray_len(grid, goals, r, c, -1, 0),
            "E": ray_len(grid, goals, r, c, 0, 1),
            "S": ray_len(grid, goals, r, c, 1, 0),
            "W": ray_len(grid, goals, r, c, 0, -1),
        }
        present = {d for d, L in lens.items() if L > 0}
        if len(present) >= 3:
            inter[(c, r)] = lens

    arms = sum(len({d for d, L in lens.items() if L > 0}) for lens in inter.values())
    links = 0
    for (c, r), lens in inter.items():
        for d, (dx, dy) in (("N", (0, -1)), ("E", (1, 0)), ("S", (0, 1)), ("W", (-1, 0))):
            L = lens[d]
            if L <= 0:
                continue
            t = (c + dx * (L + 1), r + dy * (L + 1))
            if t in inter:
                links += 1
    paired = links // 2
    dangling = arms - links

    print(f"{Path(path).name:32} {w}x{h:<5} tiles={tiles:6}  V={len(inter):5} "
          f"arms={arms:5} paired={paired:5} dangling={dangling:4} "
          f"paired+dangling={paired + dangling:5} goals={len(goals):4}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyse(p)
