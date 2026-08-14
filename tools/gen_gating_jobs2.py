#!/usr/bin/env python3
"""Second-round Gate C/D jobs: last-resort arm retreat over the full cell set.

The eager arm-retreat variant was retired after it broke control cell
cross_3030 a1783 s1 (1748/1783), so the expensive queue runs the last-resort
form instead.  Cheapest cells first for early signal.
"""
import sys
from gen_gating_jobs import SCEN, STRAGGLERS, CONTROLS

# Variants still missing per cell after round 1.
ROUND2 = ["retreatlast", "retreatlast_replan8"]
HEAVY_EXTRA = ["replan8"]  # round 1 never reached the expensive cells
HEAVY = 300.0


def main() -> None:
    rows = []
    for mp, agents, seed, cost in STRAGGLERS + CONTROLS:
        variants = list(ROUND2)
        if cost > HEAVY:
            variants += HEAVY_EXTRA
        for v in variants:
            rows.append((cost, f"{mp}|{SCEN[mp]}|{agents}|{seed}|{v}"))
    rows.sort(key=lambda r: r[0])
    for _cost, line in rows:
        print(line)


if __name__ == "__main__":
    sys.exit(main())
