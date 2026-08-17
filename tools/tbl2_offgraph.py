#!/usr/bin/env python3
"""tbl2_offgraph.py - how much of the sampling space lies outside the aisle graph.

Aisle graph = the cell set LIMA's IntersectionTopology actually controls
(intersection centers + every cell of every traced arm).  This is the quantity
that reproduces Table 2's "Tiles" column exactly.

Pools compared against it:
  * gen_scen.py eligible starts  = interior '.' cells (tools/gen_scen.py)
  * one-shot goal pool           = every 'S' cell
  * lifelong goal pool           = interior traversable non-sink cells
                                   (src/simulation/goal_allocator.cpp)
  * a concrete .scen file's start/goal entries
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tbl2_topo import Grid, Topology  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAPDIR = ROOT / "results" / "table2_forensics" / "maps"


def pools(map_path, scen_path=None, scen_limit=1600):
    g = Grid(map_path)
    t = Topology(g)
    graph = t.controlled_union()
    centers = t.centers()

    # gen_scen.py:  grid[y][x] == '.' for y in 1..h-2, x in 1..w-2
    starts = [g.cell(x, y)
              for y in range(1, g.h - 1)
              for x in range(1, g.w - 1)
              if g.rows[y][x] == "."]
    # goal_allocator.cpp make_goal_candidates: interior traversable non-sink
    lifelong = [cid for cid in g.free_cells
                if not (lambda c: c[0] == 0 or c[1] == 0 or c[0] + 1 == g.w or c[1] + 1 == g.h)(g.coord(cid))
                and cid not in g.sinks]
    oneshot = sorted(g.sinks)

    def frac(pool):
        off = [c for c in pool if c not in graph]
        n = len(pool)
        return {"pool": n, "on_graph": n - len(off), "off_graph": len(off),
                "off_frac": round(len(off) / n, 4) if n else None}

    out = {
        "map": str(map_path),
        "size": f"{g.w}x{g.h}",
        "free_total": len(g.free_cells),
        "S_cells": len(g.sinks),
        "V": t.V,
        "aisle_graph_cells(Tiles)": len(graph),
        "aisle_graph/free": round(len(graph) / len(g.free_cells), 4),
        "centers": len(centers),
        "(a) gen_scen_eligible_starts": frac(starts),
        "(b1) oneshot_goal_pool_S": frac(oneshot),
        "(b2) lifelong_goal_pool_interior_nonsink": frac(lifelong),
    }

    if scen_path and Path(scen_path).exists():
        lines = Path(scen_path).read_text(encoding="utf-8").splitlines()[1:]
        lines = lines[:scen_limit]
        s_off = go_off = 0
        both_off = 0
        n = 0
        for ln in lines:
            f = ln.split("\t")
            if len(f) < 8:
                continue
            sx, sy, gx, gy = int(f[4]), int(f[5]), int(f[6]), int(f[7])
            sc, gc = g.cell(sx, sy), g.cell(gx, gy)
            so = sc not in graph
            go = gc not in graph
            s_off += so
            go_off += go
            both_off += so and go
            n += 1
        out["(c) scenario"] = {
            "file": str(scen_path), "entries_checked": n,
            "starts_off_graph": s_off, "starts_off_frac": round(s_off / n, 4) if n else None,
            "goals_off_graph": go_off, "goals_off_frac": round(go_off / n, 4) if n else None,
            "both_off_graph": both_off, "both_off_frac": round(both_off / n, 4) if n else None,
        }
    return out


def ladder(tiles, free):
    pcts = [1, 5, 10, 20, 30, 40, 50, 60]
    return {
        "pct": pcts,
        "by_tiles": [int(p / 100 * tiles) for p in pcts],
        "by_free": [int(p / 100 * free) for p in pcts],
    }


def main():
    jobs = [
        (MAPDIR / "cur_warehouse_10_20.map",
         ROOT / "data/scenarios/warehouse-10-20/warehouse-10-20_s0.scen", "Standard 1", 2649),
        (MAPDIR / "cur_warehouse_20_40.map",
         ROOT / "data/scenarios/warehouse-20-40/warehouse-20-40_s0.scen", "Standard 2", 10499),
        (MAPDIR / "cross_3030_OLD_1bf5a35.map", None, "Square 1 (old cross 187x187)", 10200),
        (MAPDIR / "cur_cross_3030.map",
         ROOT / "data/scenarios/cross-30-30/cross-30-30_s0.scen", "current cross (contrast)", None),
    ]
    res = []
    for mp, sp, label, tiles in jobs:
        o = pools(str(mp), str(sp) if sp else None)
        o["label"] = label
        o["paper_tiles"] = tiles
        o["ladder"] = ladder(o["aisle_graph_cells(Tiles)"], o["free_total"])
        res.append(o)
        print(json.dumps(o, indent=2, ensure_ascii=False))
        print("-" * 70)
    with open(ROOT / "results/table2_forensics/offgraph.json", "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
