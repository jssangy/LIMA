#!/usr/bin/env python3
"""Emit the two-arm paper-ladder job list (ascending agent count first)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "paperladder" / "jobs2.txt"
ORIG = "results/paperladder/orig"

TILES = {"warehouse_10_20": 2649, "warehouse_20_40": 10499, "cross_3030": 10200}
PCTS = (1, 5, 10, 20, 30, 40, 50, 60)
SCENS = (0, 1)

ARMS = {
    # arm: map -> (map file, scenario file template)
    "orig": {
        "warehouse_10_20": (f"{ORIG}/maps/warehouse_10_20_paper.map",
                            f"{ORIG}/scen/warehouse-10-20_s{{s}}.scen"),
        "warehouse_20_40": (f"{ORIG}/maps/warehouse_20_40_paper.map",
                            f"{ORIG}/scen/warehouse-20-40_s{{s}}.scen"),
        "cross_3030": (f"{ORIG}/maps/cross_3030_paper.map",
                       f"{ORIG}/scen/cross-30-30_s{{s}}.scen"),
    },
    "ours": {
        "warehouse_10_20": ("data/maps/warehouse_10_20.map",
                            "data/scenarios/warehouse-10-20/warehouse-10-20_s{s}.scen"),
        "warehouse_20_40": ("data/maps/warehouse_20_40.map",
                            "data/scenarios/warehouse-20-40/warehouse-20-40_s{s}.scen"),
        # our cross runs on the different upstream geometry by construction
        "cross_3030": ("data/maps/cross_3030.map",
                       "data/scenarios/cross-30-30/cross-30-30_s{s}.scen"),
    },
}

rows = []
for pct in PCTS:
    for map_name, tiles in TILES.items():
        agents = int(pct * tiles / 100)          # floor, as Table 2 does
        for arm, spec in ARMS.items():
            mapfile, scen_tpl = spec[map_name]
            for s in SCENS:
                rows.append((agents, f"{arm}|{map_name}|{mapfile}|{scen_tpl.format(s=s)}|{agents}|{s}"))

rows.sort(key=lambda r: r[0])
OUT.write_text("\n".join(r[1] for r in rows) + "\n", encoding="utf-8")
print(f"{len(rows)} jobs -> {OUT}")
