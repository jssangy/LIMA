# LIMA C++

LIMA is being migrated to a native C++20 implementation. The current executable contains the map/scenario loaders, BFS and A* global planners, intersection topology and deadlock detection, and the native IDA* stack scheduler. Simulation follows the Python implementation's execution semantics: scheduled paths are inserted into each AMR's single route, normal AMRs move first in ID order, and blocked scheduled AMRs wait as one intersection group.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

Random tasks use `S`/`G` cells in the map as goals:

```bash
./build/lima --map maps/cross_1.map --agents 12 --planner bfs --max-steps 1000
```

MovingAI scenarios can be loaded directly:

```bash
./build/lima \
  --map assets/warehouse-10-20/warehouse-10-20.map \
  --scenario assets/warehouse-10-20/scen/warehouse-10-20_s0.scen \
  --agents 50 --planner astar --max-steps 5000
```

Add `--gui` to open the SDL2 viewer:

```bash
./build/lima \
  --map assets/warehouse-10-20/warehouse-10-20.map \
  --scenario assets/warehouse-10-20/scen/warehouse-10-20_s0.scen \
  --agents 50 --planner bfs --max-steps 5000 --gui --fps 30
```

Viewer controls:

- `Space`: pause/resume
- `Right Arrow`: advance one timestep while paused
- `Up` / `Down`: change simulation speed
- `Mouse wheel` or `+` / `-`: zoom at the mouse position
- `Left drag` or `Middle drag`: pan the map
- `F`: fit the whole map to the current window
- `Esc`: close

The viewer uses the same colors as the original Python GUI: a dark-gray road background, gray obstacles, and a golden-ratio HSV color per AMR. Each active goal is drawn with its AMR color. A scheduled AMR has a white center dot.

## Joint movement rule

Each timestep is processed as one transaction:

1. Snapshot current occupancy and generate one intent per active AMR.
2. Reserve active intersection schedule regions and arbitrate same-target intents.
3. Resolve dependencies. A move into an occupied cell succeeds only when its occupant also has an approved move in the same timestep.
4. Reject edge swaps, unresolved cycles, duplicate final cells, and partially blocked schedule groups.
5. Commit every approved position together.

This permits a safety-spaced convoy to advance together while preserving vertex and edge conflict invariants. Normal routes and intersection schedule frames use the same resolver and commit path.

The Python implementation remains beside the C++ tree only while migration is in progress. It is not required to configure or build the C++ executable.

## Execution modes

Realtime mode computes and renders each timestep. Supplying `--output` records the same run at the same time:

```bash
./build/lima --mode realtime --map maps/cross_1.map --agents 15 \
  --planner bfs --gui --fps 30 --output results/cross_1_15.txt
```

Solve mode runs without rendering and writes a LaCAM-compatible solution trace. If `--output` is omitted, it writes `build/result.txt`:

```bash
./build/lima --mode solve --map maps/cross_1.map --agents 15 \
  --planner bfs --max-steps 5000 --output results/cross_1_15.txt
```

Replay mode loads that trace without running the planner or scheduler:

```bash
./build/lima --mode replay --map maps/cross_1.map \
  --replay results/cross_1_15.txt --fps 30
```

Replay controls add `Left Arrow` for one timestep backward and `Home` / `End` for the first / last timestep. The existing pause, forward, speed, zoom, fit, and pan controls remain available.
Automatic replay interpolates positions between adjacent timestep configurations. Pausing or using the left/right controls snaps exactly to the selected timestep.

Set `--fps 0` in realtime or replay mode to remove the step-rate delay. Unlimited mode still renders every timestep, so headless solve mode remains the fastest option when visualization is unnecessary.
