#!/usr/bin/env python3
"""Offline invariant checker for --trace-jsonl step traces.

Replays a trace and verifies, step by step, the invariants the paper claims:

  P1  moves are stay-or-4-neighbor onto traversable cells
  P2  no vertex conflict among active agents
  P3  no edge swap between adjacent agents
  P4  an agent deactivates only at its own goal, and never reactivates
  P5  topological route preservation: the original route cells that lie
      outside intersection zones are visited in order (temporary in-zone
      reshuffling and discharge loops are allowed), and completed agents
      finish exactly at their goal

Usage: tools/verify_trace.py TRACE.jsonl [--map-root .] [--max-violations 20]
Exit code 0 = all invariants hold, 1 = violations found.
"""
import argparse
import json
import sys
from pathlib import Path

DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def load_map(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid = lines[4:4 + height]
    if len(grid) < height or any(len(row) < width for row in grid):
        raise OSError(f"map body shorter than declared {width}x{height}")
    trav = set()
    goals = set()
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch in ".SEG":
                trav.add((x, y))
            if ch in "SG":
                goals.add((x, y))
    return width, height, trav, goals


def zone_cells(width, height, trav, goals):
    """Re-derive intersection zones (center + arms) as in topology.cpp."""
    def wall(c):
        return c not in trav

    def is_center(c):
        x, y = c
        open_arms = sum(1 for dx, dy in DIRS if (x + dx, y + dy) in trav)
        if open_arms < 3:
            return False
        return all(wall((x + dx, y + dy)) for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)))

    def trace_arm(center, d):
        dx, dy = DIRS[d]
        cells = []
        c = (center[0] + dx, center[1] + dy)
        while c in trav:
            if c in goals:
                break
            if dy != 0:
                corridor = wall((c[0] - 1, c[1])) and wall((c[0] + 1, c[1]))
            else:
                corridor = wall((c[0], c[1] - 1)) and wall((c[0], c[1] + 1))
            if not corridor:
                break
            cells.append(c)
            c = (c[0] + dx, c[1] + dy)
        return cells

    zones = set()
    for c in trav:
        if not is_center(c):
            continue
        arms = [trace_arm(c, d) for d in range(4)]
        if sum(1 for a in arms if a) < 3:
            continue
        zones.add(c)
        for arm in arms:
            zones.update(arm)
    return zones


def is_subsequence(needle, haystack):
    it = iter(haystack)
    return all(any(x == y for y in it) for x in needle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--map-root", default=".")
    parser.add_argument("--max-violations", type=int, default=20)
    args = parser.parse_args()

    header = None
    steps = []
    with open(args.trace, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["type"] == "header":
                header = record
            else:
                steps.append(record)
    if header is None:
        print("trace has no header record", file=sys.stderr)
        return 1

    width = header["width"]
    map_path = Path(args.map_root) / header["map"]
    try:
        map_width, map_height, trav, goal_cells = load_map(map_path)
    except OSError as error:
        print(f"cannot read map {map_path} (try --map-root): {error}", file=sys.stderr)
        return 1
    if map_width != width or map_height != header["height"]:
        print(f"map size mismatch: trace {width}x{header['height']} vs file {map_width}x{map_height}",
              file=sys.stderr)
        return 1
    zones_xy = zone_cells(width, header["height"], trav, goal_cells)
    zones = {y * width + x for (x, y) in zones_xy}
    trav_ids = {y * width + x for (x, y) in trav}

    n = header["agents"]
    goals = header["goals"]
    routes = header["routes"]
    violations = []

    def violate(t, message):
        violations.append(f"t={t}: {message}")

    prev_pos = list(header["starts"])
    prev_active = [1] * n
    visited = [[p] for p in prev_pos]

    for record in steps:
        t = record["t"]
        pos = record["pos"]
        active = record["active"]
        def check_move(i, a, b):
            ax, ay = a % width, a // width
            bx, by = b % width, b // width
            if abs(ax - bx) + abs(ay - by) != 1:
                violate(t, f"P1 agent {i} teleported {a}->{b}")
            if b not in trav_ids:
                violate(t, f"P1 agent {i} entered blocked cell {b}")

        for i in range(n):
            if prev_active[i] and active[i]:
                if prev_pos[i] != pos[i]:
                    check_move(i, prev_pos[i], pos[i])
                    visited[i].append(pos[i])
            elif not prev_active[i] and active[i]:
                violate(t, f"P4 agent {i} reactivated")
            elif prev_active[i] and not active[i]:
                if pos[i] != goals[i]:
                    violate(t, f"P4 agent {i} deactivated away from its goal")
                if pos[i] != prev_pos[i]:
                    check_move(i, prev_pos[i], pos[i])
                    visited[i].append(pos[i])
        # P2/P3 are keyed on activity at the START of the step so agents that
        # complete during the step still participate in conflict checking.
        occupied = {}
        for i in range(n):
            if not prev_active[i]:
                continue
            if pos[i] in occupied:
                violate(t, f"P2 vertex conflict: agents {occupied[pos[i]]} and {i} at {pos[i]}")
            occupied[pos[i]] = i
        for i in range(n):
            if not prev_active[i] or pos[i] == prev_pos[i]:
                continue
            j = occupied.get(prev_pos[i])
            if j is not None and j != i and prev_active[j] and prev_pos[j] == pos[i]:
                if i < j:
                    violate(t, f"P3 edge swap between agents {i} and {j}")
        prev_pos = pos
        prev_active = active
        if len(violations) >= args.max_violations:
            break

    completed = [i for i in range(n) if not prev_active[i]]
    for i in range(n):
        if len(violations) >= args.max_violations:
            break
        milestones = [c for c in routes[i] if c not in zones]
        seen = [c for c in visited[i] if c not in zones]
        if i in completed:
            if visited[i][-1] != goals[i]:
                violate("end", f"P5 agent {i} completed away from goal")
            if not is_subsequence(milestones, seen):
                violate("end", f"P5 agent {i} skipped assigned-route cells outside intersections")
        # Incomplete agents admit any milestone prefix, so P5 only binds on
        # completed agents; P1-P4 already covered their physical moves.

    print(json.dumps({
        "trace": args.trace,
        "steps": len(steps),
        "agents": n,
        "completed": len(completed),
        "zone_cells": len(zones),
        "ok": not violations,
        "violations": violations[:args.max_violations],
    }, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
