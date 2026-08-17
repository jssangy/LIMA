#!/usr/bin/env python3
"""tbl2_offgraph_shape.py - what the off-aisle-graph free cells actually are."""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tbl2_topo import Grid, Topology, KDELTA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAPDIR = ROOT / "results" / "table2_forensics" / "maps"


def shape(map_path, label):
    g = Grid(map_path)
    t = Topology(g)
    graph = t.controlled_union()
    off = {c for c in g.free_cells if c not in graph}

    # connected components of the off-graph set
    seen, comps = set(), []
    for s in off:
        if s in seen:
            continue
        q, cells = deque([s]), []
        seen.add(s)
        while q:
            c = q.popleft()
            cells.append(c)
            x, y = g.coord(c)
            for dx, dy in KDELTA:
                nx, ny = x + dx, y + dy
                if g.trav(nx, ny):
                    nc = g.cell(nx, ny)
                    if nc in off and nc not in seen:
                        seen.add(nc)
                        q.append(nc)
        xs = [g.coord(c)[0] for c in cells]
        ys = [g.coord(c)[1] for c in cells]
        comps.append({"n": len(cells), "bbox": [min(xs), min(ys), max(xs), max(ys)]})
    comps.sort(key=lambda d: -d["n"])

    # BFS distance from the aisle graph, over free cells
    dist = {c: 0 for c in graph}
    q = deque(graph)
    while q:
        c = q.popleft()
        x, y = g.coord(c)
        for dx, dy in KDELTA:
            nx, ny = x + dx, y + dy
            if g.trav(nx, ny):
                nc = g.cell(nx, ny)
                if nc not in dist:
                    dist[nc] = dist[c] + 1
                    q.append(nc)
    offd = [dist[c] for c in off if c in dist]
    hist = {}
    for d in offd:
        hist[d] = hist.get(d, 0) + 1
    unreachable = len(off) - len(offd)

    # how "wide" are off-graph cells (free 4-degree)
    degh = {}
    for c in off:
        x, y = g.coord(c)
        d = sum(1 for dx, dy in KDELTA if g.trav(x + dx, y + dy))
        degh[d] = degh.get(d, 0) + 1

    return {
        "label": label, "size": f"{g.w}x{g.h}",
        "free": len(g.free_cells), "graph": len(graph), "off": len(off),
        "off_components": len(comps),
        "off_top_components": comps[:6],
        "off_dist_from_graph_hist": dict(sorted(hist.items())),
        "off_unreachable_from_graph": unreachable,
        "off_free_degree_hist": dict(sorted(degh.items())),
    }


def main():
    jobs = [
        (MAPDIR / "cur_warehouse_10_20.map", "Standard 1 warehouse_10_20"),
        (MAPDIR / "cur_warehouse_20_40.map", "Standard 2 warehouse_20_40"),
        (MAPDIR / "cross_3030_OLD_1bf5a35.map", "Square 1 old cross 187x187"),
        (MAPDIR / "cur_cross_3030.map", "current cross 237x177"),
    ]
    res = [shape(str(p), l) for p, l in jobs]
    for o in res:
        print(json.dumps(o, indent=2, ensure_ascii=False))
        print("-" * 70)
    with open(ROOT / "results/table2_forensics/offgraph_shape.json", "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
