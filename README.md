# LIMA

LIMA simulates multiple AMRs on grid maps and schedules their movements through intersections.

Global routes are generated with BFS or A*. Outside managed intersection regions, AMRs use PIBT
(priority inheritance with backtracking) to select collision-free moves for the next timestep.
Inside managed intersections, the intersection scheduler remains authoritative and PIBT is not used.
Intersection stack rearrangement uses bounded Beam Search, keeping scheduling latency predictable as
the number of AMRs in an intersection grows.
Temporary PIBT deviations preserve the global route and rejoin it at a future route cell.
An unscheduled AMR leaving a managed intersection participates only as a high-priority boundary-exit
root, allowing an outside blocker to inherit its priority and move out of the way.

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

### Map endpoints and completion

Map endpoint markers determine how random goals and completed AMRs are handled:

- `S` is a shared sink. If a map contains any `S` cells, random goals are selected only from those
  cells. Multiple AMRs may target the same sink, and an AMR is removed from the simulation as soon as
  it enters its sink.
- `G` is a persistent goal when the map has no `S` cells. Completed AMRs remain on these cells and
  continue to occupy them.
- If a map has neither `S` nor `G`, random goals are sampled from traversable cells outside managed
  intersection regions. These goals are also persistent.

Random starts are unique and never placed on an `S` or `G` cell. Sink removal is recorded in solution
traces, so removed AMRs also disappear when the result is replayed.

For example, run the single-intersection warehouse with 12 AMRs as follows:

```bash
./build/lima \
  --mode realtime \
  --map data/maps/warehouse_1.map \
  --agents 12 \
  --planner bfs \
  --seed 7 \
  --max-steps 1000 \
  --fps 30
```

## Lifelong mode

One-shot workloads are the default. Use `--workload lifelong` to assign a new random goal whenever
an AMR completes its current task. The AMR population remains fixed and the run continues until
`--max-steps` is reached.

Lifelong workloads are intended for maps without `S` sinks. On an `S` map, reaching a sink always
removes the AMR instead of assigning another task.

```bash
./build/lima \
  --mode realtime \
  --workload lifelong \
  --map data/maps/warehouse_10_20.map \
  --agents 500 \
  --planner bfs \
  --seed 7 \
  --max-steps 10000 \
  --fps 30
```

New goals are unique among active tasks and are assigned only after the current timestep has been
committed. The goal generator uses a random stream separate from intersection scheduling, so the
same seed produces the same task stream when scheduler internals are unchanged. Random lifelong
workloads require at least one more valid goal candidate than the number of AMRs.

The final summary reports `tasks_completed`, throughput in tasks per timestep, and average task
latency. Lifelong runs that reach `--max-steps` finish with `status=horizon_reached`.

## Scenario mode

Use `--scenario` to load tasks from a MovingAI scenario instead of generating random tasks. `--agents`
controls how many tasks are read from the beginning of the scenario. On a map containing `S` cells,
every scenario goal must be an `S` cell, but goals may be shared because AMRs disappear at the sink.
On maps without `S` cells, selected tasks must have unique persistent goals.

The following reproducible commands exercise Beam Search on the small single-intersection maps:

```bash
./build/lima --mode solve --map data/maps/cross_1.map \
  --agents 15 --seed 7 --validate-conflicts

./build/lima --mode solve --map data/maps/warehouse_1.map \
  --agents 12 --seed 7 --validate-conflicts
```

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

## Debug mode

Debug mode runs headlessly and records a complete, searchable snapshot of the simulation at every
timestep. Conflict validation is enabled automatically.

```bash
./build/lima \
  --mode debug \
  --map data/maps/cross_3030.map \
  --agents 2000 \
  --planner bfs \
  --seed 4043428185036662755 \
  --max-steps 1000 \
  --debug-dir results/debug/cross_3030_seed4043428185036662755
```

If `--debug-dir` is omitted, logs are written under
`results/debug/<map>_<agents>_seed<seed>/`. The directory contains:

| File | Contents |
|---|---|
| `metadata.json` | Run options, map information, and static intersection topology |
| `steps.jsonl` | Per-timestep summary, deadlock queue, candidates, and new schedules |
| `agents.jsonl` | Every AMR's before/after movement state, route cursor, scheduling state, PIBT decision, arm, and target exit |
| `intersections.jsonl` | Every intersection's active/waiting/blocked state, members, intents, capacity, and rescue state |
| `schedules.jsonl` | Complete paths returned by each newly activated intersection schedule |
| `routes.jsonl` | Complete route whenever an AMR route is inserted or changed |
| `events.jsonl` | Schedule activation/release, completion, group changes, and long-wait transitions |
| `anomalies.jsonl` | Vertex conflicts, edge conflicts, invalid moves, and schedule ownership violations |
| `summary.json` | Final status and aggregate counters |
| `solution.txt` | Replayable solution trace with conflict validation results |

Each `.jsonl` file contains one JSON object per line. Examples:

```bash
# Everything involving timestep 500
rg '"timestep":500[,}]' results/debug/cross_3030_seed4043428185036662755

# State of intersection 289 at timestep 500
jq 'select(.timestep == 500 and .id == 289)' \
  results/debug/cross_3030_seed4043428185036662755/intersections.jsonl

# Full history of AMR 534
jq 'select(.id == 534)' \
  results/debug/cross_3030_seed4043428185036662755/agents.jsonl
```

The logger flushes each timestep, so completed frames remain available even if a run is interrupted.
Full debug logs can be large for dense maps; use debug mode only when diagnosing a reproducible run.

## Replay mode

Replay a saved solution without running the planner or scheduler again. Solution files store whether
each AMR is active at every timestep, so AMRs removed at an `S` sink remain hidden during replay.
Lifelong solution files also store the assigned goal of every AMR at every timestep, allowing goal
markers and goal lines to follow task changes.

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
| `--mode realtime\|solve\|replay\|debug` | Select the execution mode |
| `--map FILE` | Select the map file |
| `--scenario FILE` | Load a MovingAI scenario |
| `--agents N` | Set the number of AMRs or scenario tasks |
| `--planner bfs\|astar` | Select the global path planner |
| `--workload oneshot\|lifelong` | Use a finite one-shot workload or continuously assign new goals |
| `--seed N` | Set the random seed for reproducible runs |
| `--max-steps N` | Set the maximum number of timesteps |
| `--fps N` | Set the timestep rate; `0` removes the limit |
| `--output FILE` | Set the solution output path |
| `--replay FILE` | Select a solution trace to replay |
| `--debug-dir DIR` | Select the debug trace directory in debug mode |
| `--validate-conflicts` | Enable vertex and edge conflict validation |

Run the following command to display the complete option list:

```bash
./build/lima --help
```
