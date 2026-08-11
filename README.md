# LIMA

LIMA simulates multiple AMRs on grid maps and schedules their movements through intersections.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The GUI requires the SDL2 development package. Without SDL2, only headless execution is available.

## Data directories

```text
data/maps/                  Map files
data/scenarios/             MovingAI scenario files
results/                    Generated solution traces
```

## Realtime mode

Use realtime mode to compute and display the simulation as it runs. Realtime mode always opens the GUI.

```bash
./build/lima \
  --mode realtime \
  --map data/maps/cross_3030.map \
  --agents 1000 \
  --planner bfs \
  --fps 30
```

Set `--fps 0` to run without a timestep rate limit.

```bash
./build/lima \
  --mode realtime \
  --map data/maps/cross_3030.map \
  --agents 1000 \
  --planner bfs \
  --fps 0
```

Specify a seed to reproduce the same randomly generated tasks.

```bash
./build/lima \
  --mode realtime \
  --map data/maps/cross_3030.map \
  --agents 1000 \
  --planner bfs \
  --seed 4043428185036662755 \
  --fps 30
```

When `--seed` is omitted, LIMA generates a new seed for each run.

## Scenario mode

Use `--scenario` to load tasks from a MovingAI scenario instead of generating random tasks. `--agents` controls how many tasks are read from the beginning of the scenario.

```bash
./build/lima \
  --mode realtime \
  --map data/maps/warehouse_10_20.map \
  --scenario data/scenarios/warehouse-10-20/warehouse-10-20_s0.scen \
  --agents 1000 \
  --planner bfs \
  --seed 4043428185036662755 \
  --fps 0
```

## Solve mode

Use solve mode to compute the complete solution without opening the GUI.

```bash
./build/lima \
  --mode solve \
  --map data/maps/cross_3030.map \
  --agents 1000 \
  --planner bfs \
  --seed 4043428185036662755 \
  --output results/cross_3030_1000.txt
```

If `--output` is omitted, the solution is written to `build/result.txt`. The final summary includes the timestep count, completed AMRs, movement and wait counts, and elapsed time.

Conflict validation is disabled by default. Enable it when vertex and edge conflict checks are required.

```bash
./build/lima \
  --mode solve \
  --map data/maps/cross_3030.map \
  --agents 1000 \
  --planner bfs \
  --output results/cross_3030_1000.txt \
  --validate-conflicts
```

## Replay mode

Replay a saved solution without running the planner or scheduler again.

```bash
./build/lima \
  --mode replay \
  --map data/maps/cross_3030.map \
  --replay results/cross_3030_1000.txt \
  --fps 30
```

Replay mode also supports `--fps 0`.

## GUI controls

- `Space`: pause or resume
- `Right Arrow`: advance one timestep while paused
- `Left Arrow`: move back one timestep in replay mode
- `Home` / `End`: jump to the first or last replay frame
- `Up` / `Down`: change the playback speed
- Mouse wheel or `+` / `-`: zoom in or out
- Left or middle mouse drag: pan the view
- `F`: fit the complete map to the window
- `G`: toggle lines between AMRs and their goals
- `Esc`: close the viewer

## Command-line options

| Option | Description |
|---|---|
| `--mode realtime\|solve\|replay` | Select the execution mode |
| `--map FILE` | Select the map file |
| `--scenario FILE` | Load a MovingAI scenario |
| `--agents N` | Set the number of AMRs or scenario tasks |
| `--planner bfs\|astar` | Select the global path planner |
| `--seed N` | Set the random seed for reproducible runs |
| `--max-steps N` | Set the maximum number of timesteps |
| `--fps N` | Set the timestep rate; `0` removes the limit |
| `--output FILE` | Set the solution output path |
| `--replay FILE` | Select a solution trace to replay |
| `--validate-conflicts` | Enable vertex and edge conflict validation |

Run the following command to display the complete option list:

```bash
./build/lima --help
```
