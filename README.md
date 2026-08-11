# LIMA C++

LIMA is a native C++20 implementation containing map/scenario loaders, BFS and A* global planners, intersection topology and deadlock detection, and the native IDA* stack scheduler. Scheduled paths are inserted into each AMR's single route, normal AMRs move first in ID order, and blocked scheduled AMRs wait as one intersection group.

## Project layout

```text
LIMA/
├── app/                 Executable entry point
├── include/lima/        Public C++ headers
├── src/                 C++ implementation
├── data/maps/           Grid maps
├── data/scenarios/      MovingAI scenario sets
├── build/               Generated build files (ignored)
└── results/             Traces and archived experiments (ignored)
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Native CPU instructions can be tested with the optional build flag below. It
is disabled by default because `-march=native` is not portable and is not
faster on every workload:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIMA_NATIVE_OPTIMIZATION=ON
cmake --build build -j
```

## Run

Random tasks use `S`/`G` cells in the map as goals:

```bash
./build/lima --map data/maps/cross_1.map --agents 12 --planner bfs --max-steps 1000
```

MovingAI scenarios can be loaded directly:

```bash
./build/lima \
  --map data/maps/warehouse_10_20.map \
  --scenario data/scenarios/warehouse-10-20/warehouse-10-20_s0.scen \
  --agents 50 --planner astar --max-steps 5000
```

Add `--gui` to open the SDL2 viewer:

```bash
./build/lima \
  --map data/maps/warehouse_10_20.map \
  --scenario data/scenarios/warehouse-10-20/warehouse-10-20_s0.scen \
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

The viewer uses the LIMA palette: a dark-gray road background, gray obstacles, and a golden-ratio HSV color per AMR. Each active goal is drawn with its AMR color. A scheduled AMR has a white center dot.

## Simulation order

Each timestep follows a fixed deterministic order:

1. Rebuild intersection membership and detect scheduling candidates.
2. Apply new intersection schedules in intersection-ID order.
3. Move unscheduled AMRs in agent-ID order when their next cell is available.
4. Hold an entire scheduled intersection group if any scheduled move is blocked; otherwise advance the group.
5. Remove AMRs that reached their goals.

## Execution modes

Realtime mode computes and renders each timestep. Supplying `--output` records the same run at the same time:

```bash
./build/lima --mode realtime --map data/maps/cross_1.map --agents 15 \
  --planner bfs --gui --fps 30 --output results/cross_1_15.txt
```

Solve mode runs without rendering and writes a LaCAM-compatible solution trace. If `--output` is omitted, it writes `build/result.txt`:

```bash
./build/lima --mode solve --map data/maps/cross_1.map --agents 15 \
  --planner bfs --max-steps 5000 --output results/cross_1_15.txt
```

Conflict validation is disabled by default so normal recording does not pay its
per-frame cost. Enable it explicitly when checking a run:

```bash
./build/lima --mode solve --map data/maps/cross_1.map --agents 15 \
  --planner bfs --output results/cross_1_15.txt --validate-conflicts
```

Replay mode loads that trace without running the planner or scheduler:

```bash
./build/lima --mode replay --map data/maps/cross_1.map \
  --replay results/cross_1_15.txt --fps 30
```

Replay controls add `Left Arrow` for one timestep backward and `Home` / `End` for the first / last timestep. The existing pause, forward, speed, zoom, fit, and pan controls remain available.
Automatic replay interpolates positions between adjacent timestep configurations. Pausing or using the left/right controls snaps exactly to the selected timestep.

Set `--fps 0` in realtime or replay mode to remove the step-rate delay. Unlimited mode still renders every timestep, so headless solve mode remains the fastest option when visualization is unnecessary.
