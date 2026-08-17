#!/usr/bin/env python3
"""tbl2_topo.py - faithful Python replication of lima's IntersectionTopology::build.

Mirrors src/intersection/topology.cpp + src/core/grid_map.cpp exactly so that
candidate "Tiles" quantities can be computed off-line and validated against the
simulator's own `intersections=` output.
"""
from __future__ import annotations

import sys
from collections import deque

FREE_CHARS = set(".SEG")     # grid_map.cpp:29
GOAL_CHARS = set("SG")       # grid_map.cpp:36
SINK_CHARS = set("S")        # grid_map.cpp:37
BLOCK_CHARS = set("@T")      # grid_map.cpp:30

# topology.cpp:9  kDelta = N, E, S, W
KDELTA = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIAG = ((-1, -1), (1, -1), (1, 1), (-1, 1))


class Grid:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            toks = f.read().split()
        assert toks[0] == "type"
        i = 2
        assert toks[i] == "height"
        self.h = int(toks[i + 1])
        i += 2
        assert toks[i] == "width"
        self.w = int(toks[i + 1])
        i += 2
        assert toks[i] == "map"
        i += 1
        rows = toks[i:i + self.h]
        assert len(rows) == self.h, f"rows {len(rows)} != {self.h}"
        for r in rows:
            assert len(r) == self.w, f"row width {len(r)} != {self.w}"
        self.rows = rows
        self.free = bytearray(self.w * self.h)
        self.goals = set()
        self.sinks = set()
        self.free_cells = []
        self.charcount = {}
        for y in range(self.h):
            row = rows[y]
            for x in range(self.w):
                v = row[x]
                self.charcount[v] = self.charcount.get(v, 0) + 1
                if v not in FREE_CHARS and v not in BLOCK_CHARS:
                    raise SystemExit(f"unsupported map cell char {v!r}")
                cid = y * self.w + x
                if v in FREE_CHARS:
                    self.free[cid] = 1
                    self.free_cells.append(cid)
                if v in GOAL_CHARS:
                    self.goals.add(cid)
                if v in SINK_CHARS:
                    self.sinks.add(cid)

    def in_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def trav(self, x, y):
        return self.in_bounds(x, y) and self.free[y * self.w + x] == 1

    def cell(self, x, y):
        return y * self.w + x

    def coord(self, cid):
        return cid % self.w, cid // self.w


def wall_or_outside(g, x, y):
    return not g.trav(x, y)


def detect_center(g, x, y):
    """topology.cpp:13-22"""
    if not g.trav(x, y):
        return False
    open_n = sum(1 for dx, dy in KDELTA if g.trav(x + dx, y + dy))
    if open_n < 3:
        return False
    return all(wall_or_outside(g, x + dx, y + dy) for dx, dy in DIAG)


def trace_arm(g, cx, cy, d):
    """topology.cpp:24-39"""
    dx, dy = KDELTA[d]
    cells = []
    x, y = cx + dx, cy + dy
    while g.trav(x, y):
        cid = g.cell(x, y)
        if cid in g.goals:
            break
        if dy != 0:
            corridor = wall_or_outside(g, x - 1, y) and wall_or_outside(g, x + 1, y)
        else:
            corridor = wall_or_outside(g, x, y - 1) and wall_or_outside(g, x, y + 1)
        if not corridor:
            break
        cells.append(cid)
        x, y = x + dx, y + dy
    return cells


class Topology:
    def __init__(self, g):
        self.g = g
        self.inter = []       # list of dict(center=cid, arms=[4 lists])
        by_center = {}
        for cid in g.free_cells:
            x, y = g.coord(cid)
            if not detect_center(g, x, y):
                continue
            arms = [trace_arm(g, x, y, d) for d in range(4)]
            n = sum(1 for a in arms if a)
            if n < 3:
                continue
            by_center[cid] = len(self.inter)
            self.inter.append({"center": cid, "arms": arms, "neighbors": [-1] * 4})
        self.by_center = by_center
        for it in self.inter:
            for d in range(4):
                arm = it["arms"][d]
                if not arm:
                    continue
                tx, ty = g.coord(arm[-1])
                dx, dy = KDELTA[d]
                nx, ny = tx + dx, ty + dy
                exp = g.cell(nx, ny) if g.in_bounds(nx, ny) else -1
                if exp in by_center:
                    it["neighbors"][d] = by_center[exp]

    # ---------- derived quantities ----------
    @property
    def V(self):
        return len(self.inter)

    def centers(self):
        return {it["center"] for it in self.inter}

    def arm_cells_multiset(self):
        return sum(len(a) for it in self.inter for a in it["arms"])

    def arm_cells_union(self):
        s = set()
        for it in self.inter:
            for a in it["arms"]:
                s.update(a)
        return s

    def controlled_union(self):
        return self.arm_cells_union() | self.centers()

    def arm_count(self):
        return sum(1 for it in self.inter for a in it["arms"] if a)

    def edges_neighbor_pairs(self):
        """undirected pairs of intersections joined by an arm whose tip touches
        another center (topology.cpp:99-101)"""
        pairs = set()
        for i, it in enumerate(self.inter):
            for d in range(4):
                j = it["neighbors"][d]
                if j >= 0:
                    pairs.add((min(i, j), max(i, j)))
        return pairs

    def directed_neighbor_links(self):
        return sum(1 for it in self.inter for d in range(4) if it["neighbors"][d] >= 0)

    def dangling_arms(self):
        """arms that exist but do not land on another center"""
        n = 0
        for it in self.inter:
            for d in range(4):
                if it["arms"][d] and it["neighbors"][d] < 0:
                    n += 1
        return n


# ---------- structural (non-topology) candidates ----------

def free_neighbor_stats(g):
    """classify free cells by corridor-ness"""
    one_wide = 0        # both perpendicular sides walled in at least one axis
    strict_corridor = 0  # exactly a 1-wide passage: (L&R walls) or (U&D walls)
    open_cells = 0
    for cid in g.free_cells:
        x, y = g.coord(cid)
        lr = (not g.trav(x - 1, y)) and (not g.trav(x + 1, y))
        ud = (not g.trav(x, y - 1)) and (not g.trav(x, y + 1))
        if lr or ud:
            strict_corridor += 1
        deg = sum(1 for dx, dy in KDELTA if g.trav(x + dx, y + dy))
        if deg >= 3 and not (lr or ud):
            open_cells += 1
    one_wide = strict_corridor
    return {"strict_corridor": strict_corridor, "open_ge3_noncorridor": open_cells}


def component_free(g):
    """largest 4-connected free component size"""
    seen = set()
    best = 0
    for start in g.free_cells:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        n = 0
        while q:
            c = q.popleft()
            n += 1
            x, y = g.coord(c)
            for dx, dy in KDELTA:
                nx, ny = x + dx, y + dy
                if g.trav(nx, ny):
                    nc = g.cell(nx, ny)
                    if nc not in seen:
                        seen.add(nc)
                        q.append(nc)
        best = max(best, n)
    return best


def analyse(path, label, expect_V=None, expect_E=None, expect_tiles=None):
    g = Grid(path)
    t = Topology(g)
    free_total = len(g.free_cells)
    dots = g.charcount.get(".", 0)
    S = g.charcount.get("S", 0)
    ctr = t.centers()
    arms_u = t.arm_cells_union()
    ctrl = t.controlled_union()
    fs = free_neighbor_stats(g)
    edges = t.edges_neighbor_pairs()

    out = {
        "label": label,
        "path": path,
        "size": f"{g.width if hasattr(g,'width') else g.w}x{g.h}",
        "w": g.w, "h": g.h,
        "chars": dict(sorted(g.charcount.items())),
        "free_total(.SEG)": free_total,
        "dots_only": dots,
        "S": S,
        "V_detected": t.V,
        "V_expected": expect_V,
        "E_neighbor_pairs": len(edges),
        "E_directed_links": t.directed_neighbor_links(),
        "E_arms_existing": t.arm_count(),
        "E_dangling_arms": t.dangling_arms(),
        "E_expected": expect_E,
        "tiles_expected": expect_tiles,
        # --- Tiles candidates ---
        "(a) controlled_union(centers+arms)": len(ctrl),
        "(b) arm_cells_union(excl centers)": len(arms_u),
        "(c) arm_cells_multiset(sum arm len)": t.arm_cells_multiset(),
        "(c2) arm_multiset+centers": t.arm_cells_multiset() + t.V,
        "(e) strict_corridor_free_cells": fs["strict_corridor"],
        "(e2) strict_corridor+centers": fs["strict_corridor"] + t.V,
        "(d) free - open(deg>=3,noncorr)": free_total - fs["open_ge3_noncorridor"],
        "(f) free/2": free_total / 2,
        "(f2) dots/2": dots / 2,
        "(g) free - S": free_total - S,
        "(h) largest_free_component": component_free(g),
    }
    return g, t, out


def main():
    import json
    base = sys.argv[1] if len(sys.argv) > 1 else "results/table2_forensics/maps"
    jobs = [
        (f"{base}/cur_warehouse_10_20.map", "Standard 1 (warehouse_10_20)", 189, 390, 2649),
        (f"{base}/cur_warehouse_20_40.map", "Standard 2 (warehouse_20_40)", 779, 1580, 10499),
        (f"{base}/cross_3030_OLD_1bf5a35.map", "Square 1 (OLD cross_3030 187x187)", 900, 1860, 10200),
        (f"{base}/cur_cross_3030.map", "current cross_3030 237x177 (contrast)", None, None, None),
    ]
    results = []
    for p, lbl, v, e, tl in jobs:
        try:
            _, _, o = analyse(p, lbl, v, e, tl)
        except Exception as ex:  # noqa: BLE001
            o = {"label": lbl, "path": p, "error": repr(ex)}
        results.append(o)
        print(json.dumps(o, indent=2, ensure_ascii=False))
        print("-" * 70)
    with open("results/table2_forensics/topo_candidates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
