# Canonical Instance Set v2 (Phase 0.2 re-freeze, 2026-08-13)

Re-frozen for the LIMA revision. Map geometry for `warehouse_10_20` and
`warehouse_20_40` is adopted from the coworker's open (wall-removed) layout
on `github/main`; `S` sink cells are re-stamped at the original workstation
coordinates taken from our previous frozen maps (every coordinate verified
to be traversable `.` in the new geometry before stamping; zero mismatches).
`cross_3030` is parked and unchanged. These numbers feed the revised
Table 2.

## Maps

| map | dimensions (W x H) | traversable (`.`+`S`) | S (sinks) | tiles (= component - S) |
|---|---|---|---|---|
| warehouse_10_20 | 161 x 63 | 5619 | 42 | 5577 |
| warehouse_20_40 | 321 x 123 | 22440 | 82 | 22358 |
| cross_3030 | 237 x 177 | 18650 | 816 | 17834 |

cross_3030 (incorporated 2026-08-13, unparked): geometry from github/main
(obstacles use `@` and `T`; both blocked by the loader). It has no border
exits on the outermost ring (ring 0 fully walled), so S was stamped with
the one-shot boundary-exit rule on the outermost ring containing
traversable cells: every traversable cell on ring 1 (816 cells).

### Connectivity check (4-connected)

All maps pass: exactly one connected component over traversable cells, and
it contains all S cells.

| map | components | component size | S in component | tiles = component - S |
|---|---|---|---|---|
| warehouse_10_20 | 1 | 5619 | 42 | 5577 |
| warehouse_20_40 | 1 | 22440 | 82 | 22358 |
| cross_3030 | 1 | 18650 | 816 | 17834 |

## Scenarios

Generator: `tools/gen_scen.py` (stdlib `random.Random`, numpy-free,
deterministic). Rule (paper rule):

- eligible starts = unique interior traversable non-S cells (outermost
  row/col excluded), enumerated row-major; sampled WITHOUT replacement
  (`rng.sample`);
- goals = S cells drawn WITH replacement (`rng.choice`), row-major sink
  enumeration, drawn after all starts;
- tasks per scenario = min(10000, eligible starts);
- scenario index k in s0..s19 uses seed k (`random.Random(k)`);
- output: MovingAI scen v1, `0 <map>.map <W> <H> <sx> <sy> <gx> <gy> 0`,
  files `data/scenarios/<dashed>/<dashed>_s<k>.scen`.

| map | eligible starts | tasks per scenario | rollouts |
|---|---|---|---|
| warehouse_10_20 | 5577 | 5577 | s0..s19 (seeds 0..19) |
| warehouse_20_40 | 22358 | 10000 (capped) | s0..s19 (seeds 0..19) |
| cross_3030 | 17834 | 10000 (capped) | s0..s19 (seeds 0..19) |

cross_3030 scenario files live in data/scenarios/cross-30-30/ (historical
directory name, kept; gen_scen.py maps it via DASHED_OVERRIDES).

## Density -> agent-count ladder

N = round(density x tiles), round half up.

| density | warehouse_10_20 (tiles 5577) | warehouse_20_40 (tiles 22358) | cross_3030 (tiles 17834) |
|---|---|---|---|
| 1% | 56 | 224 | 178 |
| 5% | 279 | 1118 | 892 |
| 10% | 558 | 2236 | 1783 |
| 15% | 837 | 3354 | 2675 |
| 20% | 1115 | 4472 | 3567 |
| 25% | 1394 | 5590 | 4459 |
| 30% | 1673 | 6707 | 5350 |
| 35% | 1952 | 7825 | 6242 |
| 40% | 2231 | 8943 | 7134 |
| 45% | 2510 | 10061 | 8025 |
| 50% | 2789 | 11179 | 8917 |
| 55% | 3067 | 12297 | 9809 |
| 60% | 3346 | 13415 | 10700 |

Caveat: warehouse_20_40 and cross_3030 scenarios hold 10000 tasks (the
cap), so ladder cells above 10000 agents (20_40 at 45% and up: 10061,
11179, 12297, 13415; cross at 60%: 10700) exceed the per-scenario task
count and cannot be served by a single s-k file as generated. Resolve
before running those densities (raise the cap or draw from multiple
rollouts); do not silently truncate.

## Termination findings (open geometry, disappear-at-target, default flags)

Smoke and golden baselining on the re-frozen maps expose a systematic
livelock on the adopted open geometry. Findings, not fixes:

- Low densities end early via the all-stalled break (every active agent
  waiting >= stall_threshold 10), leaving 17-45% of tasks unserved,
  deterministic, conflict-free (binary exits 2, status=step_limit):
  10_20 s0 N=56: 50/56 at step 166; s1 N=279: 184/279 at 334;
  s0 N=279: 201/279 at 325; s0 N=558: 307/558 at 521;
  20_40 s0 N=224: 186/224 at 370.
- cross_3030 shows the same pattern: s0 N=178 (1%) all-stalls at step 403
  with 107/178 served (deterministic, validation ok).
- Higher densities neither complete nor trip the all-stalled break; they
  grind without terminating: 10_20 N=1673 and N=3346 killed at 1800 s
  (smoke, retried once); 10_20 N=1115 s1, N=1673 s0 and 20_40 N=1118 s1,
  N=2236 s0, N=4472 s1, N=6707 s0 killed at 3600 s (golden baseline);
  cross N=892 s1, N=1783 s0, N=3567 s1, N=5350 s0 killed at 3600 s
  (golden baseline). 20_40 smoke N=1118/6707/13415 killed at 550 s.
- Consequence: e0_quick.golden holds the 5 cells that terminate
  deterministically (10_20 a56/a279/a558, 20_40 a224, cross a178); the 10
  mandated higher-density cells are excluded until the livelock is
  addressed.
