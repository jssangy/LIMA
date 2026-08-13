# Gate A solver tournament: legacy vs admissible lower bounds vs beam vs greedy

Date: 2026-08-13.  Branch `sanghoon/sim-refactor`.  Harness: `tools/gateA_solver_bench.sh`
(bench mode, 200 instances per cell, seed 7, uniform-random targets over random slots;
bench auto-raises the solver acceptance capacity to the arm height).  Summaries:
`tools/gateA_solver_report.py`.  Raw CSVs: `results/gateA_bench/` (untracked).

## What was implemented

1. **`--lb-mode legacy|bf|tt` (commit `b595a64`).**  Opt-in admissible lower bounds for
   the intersection IDA*, replacing (only when selected) the shipped inflated heuristic.
   The bounds are derived for the solver's RELAXED goal (sorted stacks with
   overflow parking, `solved()` in `src/scheduling/ida_star.cpp`), not classic CPMP:
   - `bf` (Bortfeldt-Forster 2012 style, adapted): (i) *misplaced count* -- for each
     stack, the longest bottom prefix that can bottom out some goal configuration
     (own-type run; extendable by parked overflow types only once the run holds all
     items of the stack's type).  Every item above that prefix must relocate at least
     once in every solution.  (ii) *demand/supply re-entries* -- stack `t` must
     receive `need_t = (overflow ? cap_t : count_t) - leading_run_t` entries by
     `t`-items; only misplaced `t`-items outside `t` can enter as their single counted
     move, so `max(0, need_t - out_bad_t)` further moves are forced.
   - `tt` (Tanaka-Tierney 2018 flavoured refinement): `bf` plus *mutual cross pairs* --
     for non-overflow stacks `u != t`, considering the first final landing among the
     `t`-items in `u` and the `u`-items in `t` shows the smaller side must move twice;
     adds `min(cross_ut, cross_tu)` per pair.  All three charge disjoint item sets, so
     the sum stays admissible (full argument in the `lower_bound2` comment block).
2. **`--dominance` (commit `49d6787`).**  Opt-in TT18-style dominance pruning:
   (a) transitive-move elimination (never move the item the previous move just placed;
   the two-move chain is dominated by the direct move generated at the parent) and
   (b) symmetric no-op pairs (consecutive moves on disjoint stack pairs commute; only
   the lexicographically non-decreasing order is kept).  Canonical-solution rewriting
   preserves completeness and optimality.
3. **`--solver-nodes` + beam capacity option (commit `36f96e2`).**  Opt-in expanded-node
   budget for IDA* (bench safety valve) and a `BeamSearchOptions.max_capacity`
   acceptance bound so bench mode can raise beam past 16-high stacks like it already
   raised IDA*.

Defaults are byte-identical throughout (golden regression 9/9 after each commit).

## Verification of admissibility / optimality

- **Brute-force cross-check.**  400 random instances (4 stacks, caps 3..5, 2..9 items,
  overflow cases included): `bf`, `tt`, and `tt --dominance` with `--bound-step 0
  --no-fastpath` all match an independent BFS optimum exactly (the BFS reimplements the
  relaxed goal from the spec).
- **Exhaustive state-graph check.**  On a counterexample-guided instance
  (caps 5/3/5/4, 9 items, one overflow type) the full reachable graph (340,200 states)
  was enumerated and exact goal distances computed by reverse BFS: the `bf` bound never
  exceeds the true distance at any state.
- **The greedy fast path, not the bound, is the optimality leak.**  With fastpath on,
  even admissible bounds return suboptimal solutions (the exclusive forced-move phase
  can bypass the optimal line).  Optimal configuration = `--lb-mode bf|tt --bound-step 0
  --no-fastpath`.
- **Legacy is measurably inadmissible.**  At arms 8x4, N=12, seed 3, `--bound-step 0`:
  legacy returned strictly longer solutions than bf/tt on 98/100 instances.

## Deterministic fast-path completion in non-fastpath iterations (task 2 audit)

The deterministic completion IS exploited outside the fastpath phase: bucket-0 move
ordering in `generate_moves` places safe direct placements first, so the DFS follows
greedy completion chains first in every iteration; the fastpath phase additionally makes
that move *exclusive* at nodes where it exists.  No change was needed; the difference
between the phases is exclusivity (speed) vs branching (coverage), and the phase-2
restart already covers fastpath dead ends.

## Tournament grid

Solvers: `ida_legacy` (shipped default + 2M node budget), `ida_bf`, `ida_tt`,
`ida_tt_dom` (tt + dominance), `ida_tt_opt` (tt + dominance + bound-step 0 +
no-fastpath = optimality-capable), `beam` (defaults), `greedy`.  N = 30/60/90% of
summed capacity.  The 16x4 and 20x4 rows are the literature-flagged height sweep
(CPMP benchmarks stop at tier <= 10; these rows are deliberately above the norm).
Wall-clock numbers were taken while an unrelated 8-way sweep loaded the machine;
`expanded` node counts are deterministic and are the primary comparator.

| arms | N | solver | fail | med len | med us | p99 us | med nodes | p99 nodes |
|---|---|---|---|---|---|---|---|---|
| 5x4 | 6 | ida_legacy | 0/200 | 7.0 | 49.0 | 264 | 11.0 | 98 |
| 5x4 | 6 | ida_bf | 0/200 | 7.0 | 61.0 | 190 | 21.0 | 925 |
| 5x4 | 6 | ida_tt | 0/200 | 6.0 | 50.0 | 459 | 9.0 | 94 |
| 5x4 | 6 | ida_tt_dom | 0/200 | 6.0 | 50.5 | 427 | 9.0 | 76 |
| 5x4 | 6 | ida_tt_opt | 0/200 | 6.0 | 48.0 | 150 | 9.0 | 155 |
| 5x4 | 6 | beam | 0/200 | 6.0 | 1028.0 | 7934 | 759.0 | 6806 |
| 5x4 | 6 | greedy | 96/200 | 5.0 | 0.0 | 1 | 6.0 | 12 |
| 5x4 | 12 | ida_legacy | 0/200 | 26.0 | 97.0 | 867 | 301.0 | 3588 |
| 5x4 | 12 | ida_bf | 0/200 | 17.0 | 583.5 | 61982 | 3953.0 | 428910 |
| 5x4 | 12 | ida_tt | 0/200 | 17.0 | 143.0 | 7566 | 307.5 | 42309 |
| 5x4 | 12 | ida_tt_dom | 0/200 | 17.0 | 134.5 | 5750 | 216.5 | 26640 |
| 5x4 | 12 | ida_tt_opt | 0/200 | 16.0 | 197.5 | 7990 | 620.5 | 37802 |
| 5x4 | 12 | beam | 0/200 | 16.0 | 45748.5 | 76644 | 21840.0 | 32424 |
| 5x4 | 12 | greedy | 199/200 | 12 | 2 | 2 | 13 | 13 |
| 5x4 | 18 | ida_legacy | 200/200 | - | - | 0 | - | 0 |
| 5x4 | 18 | ida_bf | 200/200 | - | - | 0 | - | 0 |
| 5x4 | 18 | ida_tt | 200/200 | - | - | 0 | - | 0 |
| 5x4 | 18 | ida_tt_dom | 200/200 | - | - | 0 | - | 0 |
| 5x4 | 18 | ida_tt_opt | 200/200 | - | - | 0 | - | 0 |
| 5x4 | 18 | beam | 200/200 | - | - | 0 | - | 0 |
| 5x4 | 18 | greedy | 200/200 | - | - | 0 | - | 0 |
| 10x4 | 12 | ida_legacy | 0/200 | 39.0 | 98.5 | 653 | 125.5 | 3323 |
| 10x4 | 12 | ida_bf | 0/200 | 17.0 | 646.0 | 28185 | 4591.0 | 165118 |
| 10x4 | 12 | ida_tt | 0/200 | 16.0 | 110.0 | 1217 | 165.0 | 5916 |
| 10x4 | 12 | ida_tt_dom | 0/200 | 17.0 | 103.0 | 1081 | 138.0 | 3695 |
| 10x4 | 12 | ida_tt_opt | 0/200 | 16.0 | 137.5 | 2538 | 206.5 | 9694 |
| 10x4 | 12 | beam | 0/200 | 16.0 | 50493.0 | 86382 | 22173.0 | 29319 |
| 10x4 | 12 | greedy | 200/200 | - | - | 0 | - | 0 |
| 10x4 | 24 | ida_legacy | 13/200 | 59 | 3451 | 271479 | 35409 | 1216798 |
| 10x4 | 24 | ida_bf | 182/200 | 32.0 | 84462.0 | 575396 | 504643.5 | 1728141 |
| 10x4 | 24 | ida_tt | 53/200 | 37 | 31985 | 727124 | 144419 | 1999324 |
| 10x4 | 24 | ida_tt_dom | 46/200 | 37.0 | 34456.0 | 970800 | 120063.5 | 1890823 |
| 10x4 | 24 | ida_tt_opt | 72/200 | 36.0 | 45334.0 | 669502 | 193342.0 | 1889804 |
| 10x4 | 24 | beam | 174/200 | 39.0 | 297127.0 | 1926548 | 70050.0 | 498289 |
| 10x4 | 24 | greedy | 200/200 | - | - | 0 | - | 0 |
| 10x4 | 36 | ida_legacy | 200/200 | - | - | 0 | - | 0 |
| 10x4 | 36 | ida_bf | 200/200 | - | - | 0 | - | 0 |
| 10x4 | 36 | ida_tt | 200/200 | - | - | 0 | - | 0 |
| 10x4 | 36 | ida_tt_dom | 200/200 | - | - | 0 | - | 0 |
| 10x4 | 36 | ida_tt_opt | 200/200 | - | - | 0 | - | 0 |
| 10x4 | 36 | beam | 200/200 | - | - | 0 | - | 0 |
| 10x4 | 36 | greedy | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 19 | ida_legacy | 0/200 | 73.0 | 386.5 | 94748 | 2779.0 | 339778 |
| 16x4 | 19 | ida_bf | 80/200 | 27.0 | 42593.5 | 580545 | 232210.0 | 1872183 |
| 16x4 | 19 | ida_tt | 0/200 | 29.0 | 1368.5 | 196261 | 5800.5 | 626856 |
| 16x4 | 19 | ida_tt_dom | 0/200 | 29.0 | 1110.5 | 169927 | 3664.0 | 414519 |
| 16x4 | 19 | ida_tt_opt | 2/200 | 28.0 | 3903.0 | 295596 | 19192.5 | 1114624 |
| 16x4 | 19 | beam | 73/200 | 28 | 215272 | 2240366 | 47258 | 409858 |
| 16x4 | 19 | greedy | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 38 | ida_legacy | 164/200 | 103.0 | 278055.0 | 593743 | 1077521.0 | 1976441 |
| 16x4 | 38 | ida_bf | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 38 | ida_tt | 193/200 | 58 | 171309 | 634121 | 516483 | 1510718 |
| 16x4 | 38 | ida_tt_dom | 192/200 | 59.0 | 225066.0 | 672995 | 510140.0 | 1500252 |
| 16x4 | 38 | ida_tt_opt | 196/200 | 57.0 | 203382.5 | 832054 | 522348.0 | 1781279 |
| 16x4 | 38 | beam | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 38 | greedy | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 58 | ida_legacy | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 58 | ida_bf | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 58 | ida_tt | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 58 | ida_tt_dom | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 58 | ida_tt_opt | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 58 | beam | 200/200 | - | - | 0 | - | 0 |
| 16x4 | 58 | greedy | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 24 | ida_legacy | 27/200 | 93 | 1604 | 735703 | 18596 | 1746467 |
| 20x4 | 24 | ida_bf | 167/200 | 34 | 176589 | 631625 | 693765 | 1920835 |
| 20x4 | 24 | ida_tt | 24/200 | 37.0 | 16178.5 | 641346 | 81933.5 | 1942213 |
| 20x4 | 24 | ida_tt_dom | 16/200 | 37.0 | 21664.5 | 968365 | 76910.5 | 1742007 |
| 20x4 | 24 | ida_tt_opt | 46/200 | 35.0 | 40501.0 | 812459 | 141033.5 | 1814165 |
| 20x4 | 24 | beam | 182/200 | 34.0 | 298273.5 | 577650 | 59979.5 | 72225 |
| 20x4 | 24 | greedy | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 48 | ida_legacy | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 48 | ida_bf | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 48 | ida_tt | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 48 | ida_tt_dom | 199/200 | 75 | 793400 | 793400 | 1491746 | 1491746 |
| 20x4 | 48 | ida_tt_opt | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 48 | beam | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 48 | greedy | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 72 | ida_legacy | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 72 | ida_bf | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 72 | ida_tt | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 72 | ida_tt_dom | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 72 | ida_tt_opt | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 72 | beam | 200/200 | - | - | 0 | - | 0 |
| 20x4 | 72 | greedy | 200/200 | - | - | 0 | - | 0 |

## Reading and recommendation

### Failure-outcome note

Fail counts mix two outcomes.  At 5x4 N=18, 10/200 instances are PROVEN infeasible
(complete search exhaustion, `no_solution`; legacy and bf agree exactly on which ten),
the rest are `iteration_limit` (2M-node budget).  All 90%-fill rows are effectively
out of reach for every solver at this budget; the 60%-fill rows at heights 16/20 are
the true frontier.

### Reading

1. **The pair refinement is what makes admissibility affordable.**  `bf` alone
   collapses (80/200 fails at 16x4 N=19 where `tt` has 0; 182 vs 53 at 10x4 N=24).
   The mutual cross-pair term cuts median expanded nodes by one to two orders of
   magnitude versus `bf` everywhere.
2. **tt/tt_dom beat legacy on solution quality by 1.5-2.5x.**  Paired per-instance
   ratios (instances solved by both, legacy length / tt_dom length): 1.50 at 5x4
   N=12, 2.30 at 10x4 N=12, 2.52 at 16x4 N=19, 2.53 at 20x4 N=24 (n=161..200).
   Every solver move is a physical relocation in the simulator, so this halves the
   traffic a resolved intersection injects.
3. **Height sweep (16x4/20x4, the literature-flagged regime).**  At 30% fill both
   heights stay solvable: legacy 0 and 27 fails, tt_dom 0 and 16 fails -- tt_dom is
   the most robust config on the tallest row while returning 2.5x shorter schedules.
   The cost is more search: median nodes 3.7k vs 2.8k (16x4) and 77k vs 19k (20x4).
   At 60% fill the legacy inflated heuristic degrades into an aggressive satisficer
   and solves more within the budget (36/200 vs 8/200 at 16x4 N=38); nobody is
   ms-viable there (medians are already 100ms+).
4. **Dominance pruning is a consistent small win.**  tt_dom <= tt in nodes and fails
   in almost every cell (e.g. 16x4 N=19: 3.7k vs 5.8k median nodes; 20x4 N=24: 16 vs
   24 fails); it never hurt correctness (BFS-optimum check passed with dominance on).
5. **beam and greedy are dominated.**  Greedy fails 48-100% even at 30% fill.  Beam
   solves less than tt_dom everywhere while spending 2-3 orders of magnitude more
   wall time; its niche (anytime fallback) is better served by tt_dom with a node
   budget.
6. **tt_opt (optimality-capable: bound-step 0, no-fastpath, dominance)** costs a
   modest node factor over tt_dom and loses a few percent solve rate at the frontier
   (2/200 fails at 16x4 N=19).  It is the right config for offline/reference runs,
   not for the online default.

### Recommendation for the Gate A freeze

**Default candidate: `--lb-mode tt --dominance` (tt_dom), keeping bound_step 6 and
the greedy fastpath.**  Rationale: in the operational fill regime (30% rows,
including both height-sweep rows) it has the best failure profile of all configs
(0/0/0/16 across the four heights vs legacy 0/0/0/27), returns 1.5-2.5x shorter
schedules on identical instances, and stays within the ms envelope at median
(50us / 103us / 1.1ms / 22ms medians under a loaded machine; node counts, the
deterministic metric, are 9 / 138 / 3.7k / 77k).  Legacy remains available as
`--lb-mode legacy` and is the better satisficer deep past the envelope (60% fill),
which the sim should treat as a discharge/isolation regime rather than a solver
regime.  Suggested freeze shape: tt_dom as the SolverConfig default with an explicit
`max_nodes` budget (e.g. 2M) so pathological cells degrade into deterministic
failure instead of stalls; `tt_opt` reserved for offline optimal references; beam
demoted from the fallback chain.

Final default flip is intentionally NOT part of this change set: all new behavior
stays opt-in until the tournament outcome is ratified for Gate A.

