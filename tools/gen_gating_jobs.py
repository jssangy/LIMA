#!/usr/bin/env python3
"""Emit the Gate C/D tournament job list (cheap cells first for fast signal)."""
import sys

SCEN = {
    "warehouse_10_20": "warehouse-10-20",
    "warehouse_20_40": "warehouse-20-40",
    "cross_3030": "cross-30-30",
}

# (map, agents, seed, approximate baseline wall seconds)
STRAGGLERS = [
    ("warehouse_10_20", 279, 0, 0.2), ("warehouse_10_20", 279, 1, 0.2),
    ("warehouse_20_40", 1118, 0, 5.4), ("warehouse_20_40", 1118, 1, 5.5),
    ("warehouse_10_20", 1394, 0, 732), ("warehouse_20_40", 3354, 0, 1024),
    ("warehouse_20_40", 3354, 1, 1357), ("warehouse_10_20", 1952, 0, 2216),
]
CONTROLS = [
    ("warehouse_10_20", 56, 0, 0.03), ("warehouse_10_20", 56, 1, 0.03),
    ("cross_3030", 178, 0, 0.36), ("cross_3030", 178, 1, 0.28),
    ("warehouse_20_40", 224, 0, 0.38), ("warehouse_20_40", 224, 1, 0.33),
    ("warehouse_10_20", 558, 0, 5.8), ("warehouse_10_20", 558, 1, 2.0),
    ("cross_3030", 892, 0, 5.8), ("cross_3030", 892, 1, 5.5),
    ("warehouse_10_20", 837, 0, 37), ("warehouse_10_20", 837, 1, 90),
    ("warehouse_20_40", 2236, 0, 231), ("warehouse_20_40", 2236, 1, 204),
    ("warehouse_10_20", 1115, 0, 236), ("warehouse_10_20", 1115, 1, 437),
    ("cross_3030", 1783, 1, 20), ("warehouse_10_20", 1394, 1, 780),
    ("warehouse_10_20", 1952, 1, 2373),
]

VARIANTS_ALL = ["base", "retreat", "replan8", "retreat_replan8"]
VARIANTS_CHEAP_EXTRA = ["agerate"]
CHEAP = 300.0  # seconds; the extra variants only run on cells this fast


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cells = []
    if which in ("all", "straggler"):
        cells += [(c, "S") for c in STRAGGLERS]
    if which in ("all", "control"):
        cells += [(c, "C") for c in CONTROLS]
    rows = []
    for (mp, agents, seed, cost), _kind in cells:
        variants = list(VARIANTS_ALL)
        if cost <= CHEAP:
            variants += VARIANTS_CHEAP_EXTRA
        for v in variants:
            rows.append((cost, f"{mp}|{SCEN[mp]}|{agents}|{seed}|{v}"))
    rows.sort(key=lambda r: r[0])  # cheapest first: early signal, late tail
    for _cost, line in rows:
        print(line)


if __name__ == "__main__":
    main()
