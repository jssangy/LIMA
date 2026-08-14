#!/usr/bin/env python3
"""Check whether the submitted-era scenario files place agents on the aisle graph.

The original assets/*/scen/*.scen files were deleted in commit 1bf5a35; they are
recovered from git into a scratch directory and their start/goal cells compared
against the topology replication in tools/tbl2_topo.py (validated against the
manuscript's own #V, #E and Tiles).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import tbl2_topo  # noqa: E402

REV = "1bf5a35^"
SCRATCH = ROOT / "results" / "table2_forensics" / "orig_assets"
CASES = [
    ("Standard 1 / warehouse-10-20", "assets/warehouse-10-20/warehouse-10-20.map",
     "assets/warehouse-10-20/scen/warehouse-10-20_s0.scen"),
    ("Standard 2 / warehouse-20-40", "assets/warehouse-20-40/warehouse-20-40.map",
     "assets/warehouse-20-40/scen/warehouse-20-40_s0.scen"),
    ("Square 1 / cross-30-30", "assets/cross-30-30/cross-30-30.map",
     "assets/cross-30-30/scen/cross-30-30_s0.scen"),
]


def recover(path):
    dest = SCRATCH / Path(path).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["git", "show", f"{REV}:{path}"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"cannot read {path}: {out.stderr.strip()[:200]}")
    dest.write_text(out.stdout, encoding="utf-8", newline="\n")
    return dest


def main():
    for label, map_path, scen_path in CASES:
        map_file = recover(map_path)
        scen_file = recover(scen_path)
        grid = tbl2_topo.Grid(str(map_file))
        topo = tbl2_topo.Topology(grid)
        controlled = topo.controlled_union()

        lines = [ln for ln in scen_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        header, body = lines[0], lines[1:]
        starts, goals, bad = [], [], 0
        for line in body:
            f = line.split("\t") if "\t" in line else line.split()
            try:
                sx, sy, gx, gy = int(f[4]), int(f[5]), int(f[6]), int(f[7])
            except (IndexError, ValueError):
                bad += 1
                continue
            starts.append(grid.cell(sx, sy))
            goals.append(grid.cell(gx, gy))

        s_on = sum(1 for c in starts if c in controlled)
        g_on = sum(1 for c in goals if c in controlled)
        g_sink = sum(1 for c in goals if c in grid.sinks)
        print(f"=== {label} ===")
        print(f"  header             : {header.strip()[:60]}")
        print(f"  entries            : {len(body)} (unparsed {bad})")
        print(f"  Tiles (graph cells): {len(controlled)}   #V={topo.V}")
        print(f"  starts on graph    : {s_on}/{len(starts)} = {100.0 * s_on / max(1, len(starts)):.1f}%")
        print(f"  goals  on graph    : {g_on}/{len(goals)} = {100.0 * g_on / max(1, len(goals)):.1f}%")
        print(f"  goals at sink 'S'  : {g_sink}/{len(goals)} = {100.0 * g_sink / max(1, len(goals)):.1f}%")
        print(f"  distinct starts    : {len(set(starts))}")
        print(f"  distinct goals     : {len(set(goals))}")
        print(f"  starts == graph set: {set(starts) == controlled}")
        print()


if __name__ == "__main__":
    main()
