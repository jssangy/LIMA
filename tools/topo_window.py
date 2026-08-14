#!/usr/bin/env python3
"""Offline replica of IntersectionTopology::build for one map window.

Prints, for a rectangular window, which cells are intersection centers, arm
cells (with owner center), goal (S) cells, and plain corridor cells.  Used to
diagnose PIBT corridor jams without touching the simulator.
"""
import sys

path = sys.argv[1]
x0, x1, y0, y1 = (int(v) for v in sys.argv[2:6])

lines = open(path).read().splitlines()
h = [i for i, l in enumerate(lines) if l.strip() == "map"][0]
grid = lines[h + 1:]
H, W = len(grid), len(grid[0])

def trav(x, y):
    return 0 <= x < W and 0 <= y < H and grid[y][x] in ".SG"

def is_goal(x, y):
    return 0 <= x < W and 0 <= y < H and grid[y][x] == "S"

DELTA = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W

def detect_center(x, y):
    if not trav(x, y):
        return False
    open_n = sum(trav(x + dx, y + dy) for dx, dy in DELTA)
    if open_n < 3:
        return False
    return all(not trav(x + dx, y + dy) for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)))

def trace_arm(cx, cy, d):
    dx, dy = DELTA[d]
    cells = []
    x, y = cx + dx, cy + dy
    while trav(x, y):
        if is_goal(x, y):
            break
        if dy != 0:
            corridor = not trav(x - 1, y) and not trav(x + 1, y)
        else:
            corridor = not trav(x, y - 1) and not trav(x, y + 1)
        if not corridor:
            break
        cells.append((x, y))
        x, y = x + dx, y + dy
    return cells

member = {}
centers = []
for y in range(H):
    for x in range(W):
        if not detect_center(x, y):
            continue
        arms = [trace_arm(x, y, d) for d in range(4)]
        if sum(1 for a in arms if a) < 3:
            continue
        cid = len(centers)
        centers.append((x, y))
        member.setdefault((x, y), []).append(("C", cid))
        for d, arm in enumerate(arms):
            for i, c in enumerate(arm):
                tip = "T" if i == len(arm) - 1 else "a"
                member.setdefault(c, []).append((tip, cid))

print(f"total centers: {len(centers)}")
for y in range(y0, y1 + 1):
    row = []
    for x in range(x0, x1 + 1):
        if not trav(x, y):
            row.append("##")
        elif (x, y) in member:
            kind = member[(x, y)][0][0]
            row.append(kind + ("*" if len(member[(x, y)]) > 1 else " "))
        elif is_goal(x, y):
            row.append("S ")
        else:
            row.append(". ")
    print(f"{y:3d} " + "".join(row))
print("legend: C=center a=arm T=arm tip S=goal-cell .=free corridor ##=wall  *=multi-membership")
print("x axis:", " ".join(f"{x%100:02d}" for x in range(x0, x1 + 1)))
for (x, y) in [(135, 34), (136, 34), (136, 35), (136, 33), (137, 34), (137, 35), (134, 34), (136, 30), (136, 31), (136, 32), (136, 36)]:
    print((x, y), grid[y][x], member.get((x, y), "outside"))
