# run_our_instances.py — PRIMAL2 inference on our MovingAI-style MAPF instances

Inference harness that runs the pretrained one-shot PRIMAL2 policy
(`model_primal2_oneshot/model-97500.cptk`) on our warehouse / cross instances
(`~/lima-dev/data/maps/*.map` + `~/lima-dev/data/scenarios/<dir>/<dir>_sK.scen`)
with **disappear-at-target** semantics, greedy decoding, CPU-only TF1, fixed seeds.

## Usage

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate primal2
cd ~/mapf-baselines/PRIMAL2

python run_our_instances.py --map warehouse_10_20 --scen warehouse-10-20_s0 -n 26
python run_our_instances.py --map warehouse_10_20 --scen warehouse-10-20_s0 -n 132
python run_our_instances.py --map cross_3030     --scen cross-30-30_s0     -n 102
```

Bare `--map` / `--scen` names resolve under `~/lima-dev/data/maps` and
`~/lima-dev/data/scenarios/<dir>/`; full paths also work. Output ends with one line:

```
SUMMARY map=<m> scen=<s> N=<n> solved_fraction=<f> completed=<c>/<n> steps=<t> wall_s=<w>
```

`steps` = number of synchronous env steps executed; `wall_s` = wall-clock seconds of
the stepping loop (setup/restore times are printed separately on `[setup]` lines).

Options: `--max-steps` (default **4 × map perimeter** = `4 * 2*(H+W)` of the unpadded
map), `--seed` (default 1234), `--model`, `--obs-size` (11), `--future-steps` (3),
`--no-fast-astar`, `--progress-every K`.

Dependency note: `gym==0.17.3` was installed into the `primal2` conda env (it is
pinned in the repo's own `requirements.txt` but was missing from the env).

## How instances are mapped onto PRIMAL2's world

- Map chars `.` `S` `E` `G` → free (0); `@` `T` → obstacle (−1), per our conventions.
- Scen rows are MovingAI `(start_x=col, start_y=row, goal_x=col, goal_y=row)`; the
  first N tasks become agents 1..N (agent i = task i, IDs 1-based like PRIMAL2).
- Agents are placed into the state map (`state[start] = agentID`), goals are given
  per-agent (see deviation 3), then `Primal2Env`/`Primal2Observer` (obs 11×11,
  8+3 channels) and `listValidActions` corridor conventions are used unchanged.
- Policy stepping follows `Worker.run_episode_multithreaded`: one ACNet forward per
  agent per step with a per-agent LSTM state, PRIMAL2 valid-action masking, except
  the action is chosen **greedily** (argmax over the valid actions' probabilities)
  instead of sampled.

## Semantic deviations / implementation notes (exact list)

1. **1-cell obstacle padding.** Every map is wrapped in a ring of obstacles (all
   coordinates shift by +1 internally). PRIMAL2's generator always walls the border
   and its code crashes on traversable border cells, which `cross_3030` has 120 of
   (`corridor_map` dict KeyError in `listValidActions`, and an off-by-one bounds
   check `newPos[0] > shape[0]` at `Env_Builder.py:697`). Padding is representation
   only; the reachable topology is identical.

2. **Disappear-at-target.** PRIMAL2's own one-shot branch
   (`MAPFEnv.step_all`, `Env_Builder.py:871-874`) already removes an agent from the
   state map and goals map the moment it reaches its goal, and skips it afterwards —
   this matches our paper's semantics and is used as-is. Two leaks remained and are
   fixed in subclasses (no upstream file touched):
   - `World.CheckCollideStatus` still iterated over done agents, so a vanished agent
     kept occupying its goal cell inside the cell-wise collision resolution (it
     blocked any higher-ID agent from entering that cell forever). Overridden in
     `OneShotWorld.CheckCollideStatus` (verbatim copy restricted to live agents).
   - `Primal2Observer.get_astar_map` still projected done agents' future A* paths
     into neighbors' observations (toward a phantom "next goal"), and crashes with
     an IndexError for an agent sitting on its goal whose `next_goal == goal`.
     Overridden in `OneShotObserver.get_astar_map` (verbatim copy that skips done
     agents, leaving their channels all-zero).
   No teleport hack was needed: removal happens exactly at the arrival step, and a
   removed agent is invisible to positions, goals, observations and collisions.

3. **Duplicate goal cells.** Our scens share goal cells across tasks (warehouse s0
   first 132 tasks have only 38 unique goal cells; cross s0 first 102 have 69).
   PRIMAL2's `goals_map` is a single-channel array (one agent ID per cell) and its
   `put_goals` cannot even construct such instances. The authoritative goal is
   per-agent (`agent.goal_pos`, `agent.distanceMap`, arrival check in
   `CheckCollideStatus`), which supports duplicates natively; `goals_map` is kept
   only as a vestigial last-writer-wins representation. The one place stock code
   reads it — the *own-goal* observation channel in `Primal2Observer._get` — is
   fixed in `OneShotObserver._get` by temporarily stamping the querying agent's ID
   on its goal cell while its observation is built (the *others'-goals* channel
   already uses per-agent `getGoal` and needs no fix).

4. **No random `next_goal`.** Stock initialization samples a random future goal per
   agent (a training-time artifact of the continuous setting; it leaks into the
   observation's 3-step projected-path channels when an agent is within 3 steps of
   its goal). Under one-shot disappear semantics there is no journey after the
   goal, so `next_goal := goal` and `next_distanceMap := distanceMap`: the
   projected path shows the agent staying at its goal until it vanishes. This also
   removes the only RNG consumer at init.

5. **Bug fix: corridor endpoint sorting.** Stock `World.get_corridors`
   (`Env_Builder.py:343-348, 361-366`) crashes (`ValueError: (4,138) is not in
   list`) on warehouse geometry: a 2-cell rack-gap corridor whose two exit
   junctions are adjacent to each other makes the endpoint-neighbor lookup match
   the *other endpoint* (corridor type 2) instead of an interior corridor cell.
   `OneShotWorld.get_corridors` is a verbatim copy that additionally requires
   `corridor_map[position][1] == 1` in those two lookups — behavior-identical
   wherever the stock code worked.

6. **Fast distance maps (perf only, output-identical).**
   `Env_Builder.getAstarDistanceMap` runs a pure-Python A* with an O(open-set)
   linear min-scan, to exhaustion, from the goal, on a unit-cost 4-connected grid —
   its g-scores are exactly BFS distances. It is monkeypatched with a BFS that
   reproduces the output exactly (unreached cells keep their input map value, as in
   stock code). Disable with `--no-fast-astar` (slow: 2 full maps per agent at init).

7. **Headless rendering stub.** `Env_Builder.py:12` imports gym's pyglet/OpenGL
   rendering at module import; an empty stub module is registered in `sys.modules`
   first. Rendering is never used.

8. **Determinism.** `numpy`/`random`/TF seeds fixed (default 1234); the harness
   itself consumes no randomness (greedy argmax, deterministic init), so repeated
   runs produce identical trajectories; CPU-only (`CUDA_VISIBLE_DEVICES=-1`).

## Acceptance results (seed 1234, default step caps)

| instance | N | solved_fraction | completed | steps | wall_s |
| --- | --- | --- | --- | --- | --- |
| warehouse_10_20 s0 | 26 | 1.0000 | 26/26 | 149 | 11.8 |
| warehouse_10_20 s0 | 132 | 1.0000 | 132/132 | 354 | 135.7 |
| cross_3030 s0 | 102 | 1.0000 | 102/102 | 348 | 112.0 |

(steps cap: warehouse_10_20 `4*2*(63+161) = 1792`; cross_3030 `4*2*(187+187) = 2992`.)
