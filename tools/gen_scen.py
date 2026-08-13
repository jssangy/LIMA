#!/usr/bin/env python3
"""Deterministic scenario generator for the Phase 0.2 canonical instance set.

Generation rule (paper rule, re-frozen 2026-08-13)
--------------------------------------------------
For a map with grid cells '.' (traversable), 'S' (workstation sink) and 'T'
(wall):

* eligible start cells = every traversable non-S cell that is NOT on the
  outermost row or column of the grid (interior '.'), enumerated in row-major
  order (y ascending, then x ascending);
* sink cells = every 'S' cell, enumerated in row-major order;
* task count per scenario = min(10000, number of eligible start cells);
* starts are a uniform random sample WITHOUT replacement from the eligible
  start cells (all starts unique within one scenario);
* goals are drawn WITH replacement uniformly from the sink cells (one draw
  per task, after all starts have been drawn).

Determinism: scenario index k uses random.Random(k) (Python stdlib Mersenne
Twister; numpy-free).  With a fixed CPython the same map file always yields
byte-identical scenario files.  Draw order is fixed: rng.sample(starts, n)
first, then n successive rng.choice(sinks).

Output: MovingAI scen format, one file per scenario index:
    version 1
    0<TAB>map_filename<TAB>width<TAB>height<TAB>sx<TAB>sy<TAB>gx<TAB>gy<TAB>0
written to data/scenarios/<dashed-map-name>/<dashed-map-name>_s<k>.scen.

Default invocation regenerates s0..s19 for warehouse_10_20 and
warehouse_20_40 (cross maps are parked and not touched).
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAPS = ["warehouse_10_20", "warehouse_20_40"]
MAX_TASKS = 10000


def parse_map(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines[0].startswith("type") or lines[3] != "map":
        raise ValueError(f"{path}: not a MovingAI map")
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid = lines[4:4 + height]
    if len(grid) != height or any(len(row) != width for row in grid):
        raise ValueError(f"{path}: grid does not match declared {width}x{height}")
    return width, height, grid


def generate(map_name: str, rollouts: int, out_root: Path) -> None:
    map_path = ROOT / "data" / "maps" / f"{map_name}.map"
    width, height, grid = parse_map(map_path)
    starts = [(x, y)
              for y in range(1, height - 1)
              for x in range(1, width - 1)
              if grid[y][x] == "."]
    sinks = [(x, y)
             for y in range(height)
             for x in range(width)
             if grid[y][x] == "S"]
    if not sinks:
        raise ValueError(f"{map_name}: no S cells; sink semantics requires them")
    tasks = min(MAX_TASKS, len(starts))

    dashed = map_name.replace("_", "-")
    out_dir = out_root / dashed
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(rollouts):
        rng = random.Random(k)
        chosen_starts = rng.sample(starts, tasks)
        chosen_goals = [rng.choice(sinks) for _ in range(tasks)]
        out_path = out_dir / f"{dashed}_s{k}.scen"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("version 1\n")
            for (sx, sy), (gx, gy) in zip(chosen_starts, chosen_goals):
                f.write(f"0\t{map_name}.map\t{width}\t{height}\t"
                        f"{sx}\t{sy}\t{gx}\t{gy}\t0\n")
    print(f"{map_name}: eligible_starts={len(starts)} sinks={len(sinks)} "
          f"tasks={tasks} rollouts=s0..s{rollouts - 1} -> {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", nargs="+", default=DEFAULT_MAPS)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "data" / "scenarios"))
    args = parser.parse_args()
    for map_name in args.maps:
        generate(map_name, args.rollouts, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
