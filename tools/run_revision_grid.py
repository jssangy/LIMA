#!/usr/bin/env python3
"""Run resumable LIMA grids on the submitted-paper instance definition.

The canonical revision instance uses the submitted warehouse geometries and
the submitted 187x187 Square-1 geometry, paired with their submitted scenarios.
This matters even where topology counts match: later warehouse maps opened a
cell beside each sink and can change completion. Agent counts follow Table 2
exactly: floor(density * Tiles).

Each cell is an atomic JSON record. Existing records are skipped, so an
interrupted grid can be resumed with the same command. Broad grids default to
``--no-trace``; use ``--record-trace`` only for selected validation cells.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DENSITIES = (1, 5, 10, 20, 30, 40, 50, 60)


@dataclass(frozen=True)
class Instance:
    map_file: str
    scenario_template: str
    tiles: int


INSTANCES = {
    "warehouse_10_20": Instance(
        "data/maps/warehouse_10_20_paper.map",
        "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s{s}.scen",
        2649,
    ),
    "warehouse_20_40": Instance(
        "data/maps/warehouse_20_40_paper.map",
        "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s{s}.scen",
        10499,
    ),
    "cross_3030": Instance(
        "data/maps/cross_3030_paper.map",
        "data/scenarios/cross-30-30-paper/cross-30-30-paper_s{s}.scen",
        10200,
    ),
}


GATE_A_SOLVER = (
    "--solver", "beam", "--beam-width", "2048", "--beam-score", "tt",
    "--solver-iterations", "2000000",
)

GATE_A_COMPLETE_SOLVER = (
    "--solver", "beam-complete", "--beam-width", "2048", "--beam-score", "tt",
    "--solver-iterations", "2000000",
)

GATE_C_AIMD25 = GATE_A_SOLVER + (
    "--gate-policy", "aimd", "--gate-param", "0.25", "--gate-param2", "0.25",
)

PHASE2_FROZEN = ("--profile", "lima-default")


VARIANTS = {
    "base": (),
    "tt": ("--lb-mode", "tt"),
    "ttdom": ("--lb-mode", "tt", "--dominance"),
    "greedy": ("--solver", "greedy"),
    "beam": ("--solver", "beam"),
    "beam_paper": ("--solver", "beam", "--capacity-formula", "paper"),
    "beam_cap4": ("--solver", "beam", "--isolation-cap", "4"),
    "beam_cap8": ("--solver", "beam", "--isolation-cap", "8"),
    "beam_cap12": ("--solver", "beam", "--isolation-cap", "12"),
    "beam_margin1": ("--solver", "beam", "--isolation-margin", "1"),
    "beam_margin2": ("--solver", "beam", "--isolation-margin", "2"),
    "beam_hyst1": ("--solver", "beam", "--admit-hysteresis", "1"),
    "beam_hyst2": ("--solver", "beam", "--admit-hysteresis", "2"),
    "beam_subset": ("--solver", "beam", "--subset-scheduling"),
    "beam_noresync": ("--solver", "beam", "--no-gate-resync"),
    "beam_nodisc": ("--solver", "beam", "--no-discharge"),
    "beam_drandom": ("--solver", "beam", "--discharge-random"),
    "beam_dunweighted": ("--solver", "beam", "--discharge-unweighted"),
    "beam_dallarms": ("--solver", "beam", "--discharge-all-arms"),
    "beam_dstalled": ("--solver", "beam", "--discharge-stalled-neighbor"),
    "beam_dpartial50": ("--solver", "beam", "--discharge-partial", "0.50"),
    "beam_dpartial75": ("--solver", "beam", "--discharge-partial", "0.75"),
    "beam_dallarms_stalled": (
        "--solver", "beam", "--discharge-all-arms", "--discharge-stalled-neighbor"
    ),
    "beam_dpartial75_stalled": (
        "--solver", "beam", "--discharge-partial", "0.75", "--discharge-stalled-neighbor"
    ),
    # Gate C factorial screen.  Every row pins the frozen Gate A solver
    # explicitly and changes only admission/isolation behavior; discharge and
    # PIBT remain at their common pre-Gate-C defaults.
    "gatec_base": GATE_A_SOLVER,
    "gatec_paper": GATE_A_SOLVER + ("--capacity-formula", "paper"),
    "gatec_cap4": GATE_A_SOLVER + ("--isolation-cap", "4"),
    "gatec_cap8": GATE_A_SOLVER + ("--isolation-cap", "8"),
    "gatec_cap12": GATE_A_SOLVER + ("--isolation-cap", "12"),
    "gatec_margin1": GATE_A_SOLVER + ("--isolation-margin", "1"),
    "gatec_margin2": GATE_A_SOLVER + ("--isolation-margin", "2"),
    "gatec_hyst1": GATE_A_SOLVER + ("--admit-hysteresis", "1"),
    "gatec_hyst2": GATE_A_SOLVER + ("--admit-hysteresis", "2"),
    "gatec_subset": GATE_A_SOLVER + ("--subset-scheduling",),
    "gatec_noresync": GATE_A_SOLVER + ("--no-gate-resync",),
    "gatec_paper_margin1": GATE_A_SOLVER + (
        "--capacity-formula", "paper", "--isolation-margin", "1"
    ),
    "gatec_paper_hyst1": GATE_A_SOLVER + (
        "--capacity-formula", "paper", "--admit-hysteresis", "1"
    ),
    "gatec_paper_subset": GATE_A_SOLVER + (
        "--capacity-formula", "paper", "--subset-scheduling"
    ),
    "gatec_cap8_subset": GATE_A_SOLVER + (
        "--isolation-cap", "8", "--subset-scheduling"
    ),
    "gatec_hyst1_subset": GATE_A_SOLVER + (
        "--admit-hysteresis", "1", "--subset-scheduling"
    ),
    # Broad Gate C algorithm screen.  These policies all consume only the
    # intersection's own occupancy/request/wait state and, for neighbor
    # pressure, one-cycle-stale state from direct neighbors.
    "gatec_frac10": GATE_A_SOLVER + ("--gate-policy", "fraction", "--gate-param", "0.10"),
    "gatec_frac20": GATE_A_SOLVER + ("--gate-policy", "fraction", "--gate-param", "0.20"),
    "gatec_frac33": GATE_A_SOLVER + ("--gate-policy", "fraction", "--gate-param", "0.33"),
    "gatec_frac50": GATE_A_SOLVER + ("--gate-policy", "fraction", "--gate-param", "0.50"),
    "gatec_req05": GATE_A_SOLVER + ("--gate-policy", "request", "--gate-param", "0.50"),
    "gatec_req10": GATE_A_SOLVER + ("--gate-policy", "request", "--gate-param", "1.0"),
    "gatec_req20": GATE_A_SOLVER + ("--gate-policy", "request", "--gate-param", "2.0"),
    "gatec_req40": GATE_A_SOLVER + ("--gate-policy", "request", "--gate-param", "4.0"),
    "gatec_bp05": GATE_A_SOLVER + ("--gate-policy", "backpressure", "--gate-param", "0.50"),
    "gatec_bp10": GATE_A_SOLVER + ("--gate-policy", "backpressure", "--gate-param", "1.0"),
    "gatec_bp20": GATE_A_SOLVER + ("--gate-policy", "backpressure", "--gate-param", "2.0"),
    "gatec_bp40": GATE_A_SOLVER + ("--gate-policy", "backpressure", "--gate-param", "4.0"),
    "gatec_nbp10": GATE_A_SOLVER + ("--gate-policy", "neighbor-pressure", "--gate-param", "1.0"),
    "gatec_nbp20": GATE_A_SOLVER + ("--gate-policy", "neighbor-pressure", "--gate-param", "2.0"),
    "gatec_nbp40": GATE_A_SOLVER + ("--gate-policy", "neighbor-pressure", "--gate-param", "4.0"),
    "gatec_aimd25": GATE_A_SOLVER + ("--gate-policy", "aimd", "--gate-param", "0.25", "--gate-param2", "0.25"),
    "gatec_aimd50": GATE_A_SOLVER + ("--gate-policy", "aimd", "--gate-param", "0.50", "--gate-param2", "0.25"),
    "gatec_aimd75": GATE_A_SOLVER + ("--gate-policy", "aimd", "--gate-param", "0.75", "--gate-param2", "0.25"),
    "gatec_aimd50fast": GATE_A_SOLVER + ("--gate-policy", "aimd", "--gate-param", "0.50", "--gate-param2", "0.50"),
    "gatec_aimd75fast": GATE_A_SOLVER + ("--gate-policy", "aimd", "--gate-param", "0.75", "--gate-param2", "0.50"),
    "gatec_red40": GATE_A_SOLVER + ("--gate-policy", "red", "--gate-param", "0.40", "--gate-param2", "0.75", "--gate-param3", "0.25"),
    "gatec_red50": GATE_A_SOLVER + ("--gate-policy", "red", "--gate-param", "0.50", "--gate-param2", "0.90", "--gate-param3", "0.50"),
    "gatec_red65": GATE_A_SOLVER + ("--gate-policy", "red", "--gate-param", "0.65", "--gate-param2", "0.95", "--gate-param3", "0.50"),
    "gatec_red_aggressive": GATE_A_SOLVER + ("--gate-policy", "red", "--gate-param", "0.40", "--gate-param2", "0.80", "--gate-param3", "1.0"),
    "gatec_codel1": GATE_A_SOLVER + ("--gate-policy", "codel", "--gate-param", "1", "--gate-param2", "3"),
    "gatec_codel3": GATE_A_SOLVER + ("--gate-policy", "codel", "--gate-param", "3", "--gate-param2", "5"),
    "gatec_codel5": GATE_A_SOLVER + ("--gate-policy", "codel", "--gate-param", "5", "--gate-param2", "10"),
    "gatec_pi60": GATE_A_SOLVER + ("--gate-policy", "pi", "--gate-param", "0.60", "--gate-param2", "0.50", "--gate-param3", "0.05"),
    "gatec_pi75": GATE_A_SOLVER + ("--gate-policy", "pi", "--gate-param", "0.75", "--gate-param2", "0.50", "--gate-param3", "0.05"),
    "gatec_pi85": GATE_A_SOLVER + ("--gate-policy", "pi", "--gate-param", "0.85", "--gate-param2", "0.50", "--gate-param3", "0.05"),
    "gatec_pi_fast": GATE_A_SOLVER + ("--gate-policy", "pi", "--gate-param", "0.75", "--gate-param2", "1.0", "--gate-param3", "0.10"),
    "gatec_token025": GATE_A_SOLVER + ("--gate-policy", "token", "--gate-param", "0.25"),
    "gatec_token05": GATE_A_SOLVER + ("--gate-policy", "token", "--gate-param", "0.50"),
    "gatec_token10": GATE_A_SOLVER + ("--gate-policy", "token", "--gate-param", "1.0"),
    "gatec_token20": GATE_A_SOLVER + ("--gate-policy", "token", "--gate-param", "2.0"),
    "gatec_lqf": GATE_A_SOLVER + ("--gate-policy", "lqf"),
    "gatec_oldest": GATE_A_SOLVER + ("--gate-policy", "oldest"),
    "gatec_round_robin": GATE_A_SOLVER + ("--gate-policy", "round-robin"),
    "gatec_drr05": GATE_A_SOLVER + ("--gate-policy", "drr", "--gate-param", "0.50"),
    "gatec_drr10": GATE_A_SOLVER + ("--gate-policy", "drr", "--gate-param", "1.0"),
    "gatec_drr20": GATE_A_SOLVER + ("--gate-policy", "drr", "--gate-param", "2.0"),
    # Additional literature-derived local controllers.  BLUE/REM/AVQ/PIE
    # operate on per-intersection congestion state; SOTL and queue-CSMA
    # arbitrate only the four local arms; CHOKe/SFB use an arm as the local
    # flow class; FQ-CoDel combines per-arm DRR with local delay control.
    "gatec_blue_slow": GATE_A_SOLVER + ("--gate-policy", "blue", "--gate-param", "0.005", "--gate-param2", "0.001", "--gate-param3", "0.50"),
    "gatec_blue": GATE_A_SOLVER + ("--gate-policy", "blue", "--gate-param", "0.02", "--gate-param2", "0.002", "--gate-param3", "0.95"),
    "gatec_blue_fast": GATE_A_SOLVER + ("--gate-policy", "blue", "--gate-param", "0.05", "--gate-param2", "0.005", "--gate-param3", "0.95"),
    "gatec_rem60": GATE_A_SOLVER + ("--gate-policy", "rem", "--gate-param", "0.60", "--gate-param2", "0.02", "--gate-param3", "1.0"),
    "gatec_rem75": GATE_A_SOLVER + ("--gate-policy", "rem", "--gate-param", "0.75", "--gate-param2", "0.02", "--gate-param3", "1.0"),
    "gatec_rem85": GATE_A_SOLVER + ("--gate-policy", "rem", "--gate-param", "0.85", "--gate-param2", "0.02", "--gate-param3", "1.0"),
    "gatec_rem_fast": GATE_A_SOLVER + ("--gate-policy", "rem", "--gate-param", "0.75", "--gate-param2", "0.10", "--gate-param3", "1.0"),
    "gatec_avq70": GATE_A_SOLVER + ("--gate-policy", "avq", "--gate-param", "0.70", "--gate-param2", "0.05"),
    "gatec_avq85": GATE_A_SOLVER + ("--gate-policy", "avq", "--gate-param", "0.85", "--gate-param2", "0.05"),
    "gatec_avq95": GATE_A_SOLVER + ("--gate-policy", "avq", "--gate-param", "0.95", "--gate-param2", "0.05"),
    "gatec_avq_fast": GATE_A_SOLVER + ("--gate-policy", "avq", "--gate-param", "0.85", "--gate-param2", "0.20"),
    "gatec_pie3": GATE_A_SOLVER + ("--gate-policy", "pie", "--gate-param", "3", "--gate-param2", "0.02", "--gate-param3", "0.002"),
    "gatec_pie5": GATE_A_SOLVER + ("--gate-policy", "pie", "--gate-param", "5", "--gate-param2", "0.02", "--gate-param3", "0.002"),
    "gatec_pie10": GATE_A_SOLVER + ("--gate-policy", "pie", "--gate-param", "10", "--gate-param2", "0.02", "--gate-param3", "0.002"),
    "gatec_pie_fast": GATE_A_SOLVER + ("--gate-policy", "pie", "--gate-param", "5", "--gate-param2", "0.05", "--gate-param3", "0.01"),
    "gatec_sotl4": GATE_A_SOLVER + ("--gate-policy", "sotl", "--gate-param", "4", "--gate-param2", "2"),
    "gatec_sotl8": GATE_A_SOLVER + ("--gate-policy", "sotl", "--gate-param", "8", "--gate-param2", "2"),
    "gatec_sotl16": GATE_A_SOLVER + ("--gate-policy", "sotl", "--gate-param", "16", "--gate-param2", "2"),
    "gatec_sotl32": GATE_A_SOLVER + ("--gate-policy", "sotl", "--gate-param", "32", "--gate-param2", "2"),
    "gatec_sotl_platoon": GATE_A_SOLVER + ("--gate-policy", "sotl", "--gate-param", "8", "--gate-param2", "5"),
    "gatec_choke35": GATE_A_SOLVER + ("--gate-policy", "choke", "--gate-param", "0.35", "--gate-param2", "1.0", "--gate-param3", "0.95"),
    "gatec_choke50": GATE_A_SOLVER + ("--gate-policy", "choke", "--gate-param", "0.50", "--gate-param2", "1.0", "--gate-param3", "0.95"),
    "gatec_choke65": GATE_A_SOLVER + ("--gate-policy", "choke", "--gate-param", "0.65", "--gate-param2", "1.0", "--gate-param3", "0.95"),
    "gatec_choke_aggressive": GATE_A_SOLVER + ("--gate-policy", "choke", "--gate-param", "0.50", "--gate-param2", "2.0", "--gate-param3", "1.0"),
    "gatec_qcsma05": GATE_A_SOLVER + ("--gate-policy", "queue-csma", "--gate-param", "0.50"),
    "gatec_qcsma10": GATE_A_SOLVER + ("--gate-policy", "queue-csma", "--gate-param", "1.0"),
    "gatec_qcsma20": GATE_A_SOLVER + ("--gate-policy", "queue-csma", "--gate-param", "2.0"),
    "gatec_qcsma40": GATE_A_SOLVER + ("--gate-policy", "queue-csma", "--gate-param", "4.0"),
    "gatec_sfb1": GATE_A_SOLVER + ("--gate-policy", "sfb", "--gate-param", "0.01", "--gate-param2", "0.001", "--gate-param3", "1"),
    "gatec_sfb3": GATE_A_SOLVER + ("--gate-policy", "sfb", "--gate-param", "0.02", "--gate-param2", "0.002", "--gate-param3", "3"),
    "gatec_sfb5": GATE_A_SOLVER + ("--gate-policy", "sfb", "--gate-param", "0.01", "--gate-param2", "0.001", "--gate-param3", "5"),
    "gatec_fqcodel1": GATE_A_SOLVER + ("--gate-policy", "fq-codel", "--gate-param", "1", "--gate-param2", "3", "--gate-param3", "1"),
    "gatec_fqcodel3": GATE_A_SOLVER + ("--gate-policy", "fq-codel", "--gate-param", "3", "--gate-param2", "5", "--gate-param3", "1"),
    "gatec_fqcodel5": GATE_A_SOLVER + ("--gate-policy", "fq-codel", "--gate-param", "5", "--gate-param2", "10", "--gate-param3", "1"),
    "gatec_fqcodel_q2": GATE_A_SOLVER + ("--gate-policy", "fq-codel", "--gate-param", "3", "--gate-param2", "5", "--gate-param3", "2"),
    # Gate D family screen.  Gate A and the Gate C finalist are pinned so each
    # row changes only local recirculation/discharge behavior.
    "gated_base": GATE_C_AIMD25 + ("--discharge-policy", "composite"),
    "gated_nodisc": GATE_C_AIMD25 + ("--no-discharge",),
    "gated_random": GATE_C_AIMD25 + ("--discharge-policy", "random"),
    "gated_least_load": GATE_C_AIMD25 + ("--discharge-policy", "least-load"),
    "gated_max_slack": GATE_C_AIMD25 + ("--discharge-policy", "max-slack"),
    "gated_rotor": GATE_C_AIMD25 + ("--discharge-policy", "rotor"),
    "gated_shortest": GATE_C_AIMD25 + ("--discharge-policy", "shortest"),
    "gated_power_two": GATE_C_AIMD25 + ("--discharge-policy", "power-two"),
    "gated_backpressure": GATE_C_AIMD25 + ("--discharge-policy", "backpressure"),
    "gated_balanced025": GATE_C_AIMD25 + (
        "--discharge-policy", "balanced", "--discharge-weight", "0.25",
    ),
    "gated_balanced050": GATE_C_AIMD25 + (
        "--discharge-policy", "balanced", "--discharge-weight", "0.50",
    ),
    "gated_balanced100": GATE_C_AIMD25 + (
        "--discharge-policy", "balanced", "--discharge-weight", "1.0",
    ),
    "gated_balanced200": GATE_C_AIMD25 + (
        "--discharge-policy", "balanced", "--discharge-weight", "2.0",
    ),
    "gated_demand": GATE_C_AIMD25 + ("--discharge-policy", "demand"),
    "gated_allarms": GATE_C_AIMD25 + (
        "--discharge-policy", "composite", "--discharge-all-arms",
    ),
    "gated_stalled": GATE_C_AIMD25 + (
        "--discharge-policy", "composite", "--discharge-stalled-neighbor",
    ),
    "gated_partial50": GATE_C_AIMD25 + (
        "--discharge-policy", "composite", "--discharge-partial", "0.50",
    ),
    "gated_partial75": GATE_C_AIMD25 + (
        "--discharge-policy", "composite", "--discharge-partial", "0.75",
    ),
    "gated_partial75_stalled": GATE_C_AIMD25 + (
        "--discharge-policy", "composite", "--discharge-partial", "0.75",
        "--discharge-stalled-neighbor",
    ),
    # Phase 2 frozen composition.  The complete solver preserves the frozen
    # beam result on success and invokes exact IDA* only if beam fails.
    "phase2_frozen": PHASE2_FROZEN,
    "phase2_shuffle1": PHASE2_FROZEN + ("--shuffle-order", "1"),
    "phase2_shuffle2": PHASE2_FROZEN + ("--shuffle-order", "2"),
    "phase2_shuffle3": PHASE2_FROZEN + ("--shuffle-order", "3"),
    "phase2_shuffle4": PHASE2_FROZEN + ("--shuffle-order", "4"),
    "phase2_shuffle5": PHASE2_FROZEN + ("--shuffle-order", "5"),
    "beam_replan4": ("--solver", "beam", "--pibt-replan", "4"),
    "beam_replan8": ("--solver", "beam", "--pibt-replan", "8"),
    "beam_sink_yield": ("--solver", "beam", "--pibt-sink-yield"),
    "beam_retreat": ("--solver", "beam", "--pibt-arm-retreat"),
    "beam_retreatlast": ("--solver", "beam", "--pibt-arm-retreat-last"),
    "beam_agerate": ("--solver", "beam", "--pibt-age-rate"),
    "beam_retreat_replan8": (
        "--solver", "beam", "--pibt-arm-retreat", "--pibt-replan", "8"
    ),
    "beam_retreatlast_replan8": (
        "--solver", "beam", "--pibt-arm-retreat-last", "--pibt-replan", "8"
    ),
    "beam_yield_replan8": (
        "--solver", "beam", "--pibt-sink-yield", "--pibt-replan", "8"
    ),
    "beam_straggler_all": (
        "--solver", "beam", "--pibt-sink-yield", "--pibt-arm-retreat-last",
        "--pibt-age-rate", "--pibt-replan", "8"
    ),
    "beam_shuffle1": ("--solver", "beam", "--shuffle-order", "1"),
    "beam_shuffle2": ("--solver", "beam", "--shuffle-order", "2"),
    "beam_shuffle3": ("--solver", "beam", "--shuffle-order", "3"),
    "beam_shuffle4": ("--solver", "beam", "--shuffle-order", "4"),
    "beam_shuffle5": ("--solver", "beam", "--shuffle-order", "5"),
    "hybrid100": ("--solver", "hybrid", "--solver-nodes", "100"),
    "hybrid100_nodisc": ("--solver", "hybrid", "--solver-nodes", "100", "--no-discharge"),
    "hybrid100_drandom": ("--solver", "hybrid", "--solver-nodes", "100", "--discharge-random"),
    "hybrid100_shuffle1": ("--solver", "hybrid", "--solver-nodes", "100", "--shuffle-order", "1"),
    "hybrid100_shuffle2": ("--solver", "hybrid", "--solver-nodes", "100", "--shuffle-order", "2"),
    "hybrid100_shuffle3": ("--solver", "hybrid", "--solver-nodes", "100", "--shuffle-order", "3"),
    "hybrid100_shuffle4": ("--solver", "hybrid", "--solver-nodes", "100", "--shuffle-order", "4"),
    "hybrid100_shuffle5": ("--solver", "hybrid", "--solver-nodes", "100", "--shuffle-order", "5"),
    "hybrid1k": ("--solver", "hybrid", "--solver-nodes", "1000"),
    "hybrid10k": ("--solver", "hybrid", "--solver-nodes", "10000"),
    "hybrid100k": ("--solver", "hybrid", "--solver-nodes", "100000"),
    "hybrid500k": ("--solver", "hybrid", "--solver-nodes", "500000"),
    "nodes100k": ("--solver-nodes", "100000"),
    "nodes500k": ("--solver-nodes", "500000"),
    "nodes1m": ("--solver-nodes", "1000000"),
    "nodes2m": ("--solver-nodes", "2000000"),
    "tt_nodes500k": ("--lb-mode", "tt", "--solver-nodes", "500000"),
    "tt_nodes1m": ("--lb-mode", "tt", "--solver-nodes", "1000000"),
    "tt_nodes2m": ("--lb-mode", "tt", "--solver-nodes", "2000000"),
    "replan8": ("--pibt-replan", "8"),
    "replan8_tt_nodes2m": (
        "--pibt-replan", "8", "--lb-mode", "tt", "--solver-nodes", "2000000"
    ),
    "replan8_nodisc": ("--pibt-replan", "8", "--no-discharge"),
    "replan8_drandom": ("--pibt-replan", "8", "--discharge-random"),
    "lifelong": ("--goal-behavior", "lifelong"),
    "lifelong_nopibt": ("--goal-behavior", "lifelong", "--no-pibt-corridor"),
    "beam_failure01": ("--solver", "beam", "--failure-prob", "0.01"),
    "beam_failure05": ("--solver", "beam", "--failure-prob", "0.05"),
    "beam_failure10": ("--solver", "beam", "--failure-prob", "0.10"),
    "beam_failure20": ("--solver", "beam", "--failure-prob", "0.20"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(text: str, allowed: set[int] | None = None) -> list[int]:
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            values.update(range(lo, hi + 1))
        else:
            values.add(int(part))
    if allowed is not None and not values.issubset(allowed):
        raise argparse.ArgumentTypeError(f"values must be a subset of {sorted(allowed)}")
    return sorted(values)


def git_text(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def parse_summary(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1]))


def binary_version(path: Path) -> dict[str, str]:
    completed = subprocess.run(
        [str(path), "--version"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip() or f"exit {completed.returncode}"}
    return parse_summary(completed.stdout)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build_gating2/lima")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="base")
    parser.add_argument("--maps", default=",".join(INSTANCES))
    parser.add_argument("--densities", default=",".join(str(v) for v in DENSITIES))
    parser.add_argument("--scenarios", default="0-1")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="allow exploratory runs from tracked source changes or a mismatched binary",
    )
    args = parser.parse_args()

    maps = [name.strip() for name in args.maps.split(",") if name.strip()]
    unknown = sorted(set(maps) - set(INSTANCES))
    if unknown:
        parser.error(f"unknown maps: {', '.join(unknown)}")
    densities = parse_int_list(args.densities, set(DENSITIES))
    scenarios = parse_int_list(args.scenarios, set(range(10)))
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.record_trace and args.timeout < 1:
        parser.error("--timeout must be positive")

    binary = (ROOT / args.binary).resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    git_head = git_text("rev-parse", "HEAD")
    git_short = git_text("rev-parse", "--short", "HEAD")
    git_status_tracked = git_text("status", "--short", "--untracked-files=no")
    version = binary_version(binary)
    expected_binary_commit = f"{git_short}-dirty" if git_status_tracked else git_short
    if not args.allow_dirty and git_status_tracked:
        parser.error(
            "tracked working tree is dirty; commit the frozen source or use --allow-dirty "
            "for an explicitly exploratory run"
        )
    if not args.allow_dirty and version.get("commit") != expected_binary_commit:
        parser.error(
            f"binary/source mismatch: binary commit={version.get('commit')} "
            f"expected={expected_binary_commit}; rebuild the frozen binary"
        )
    output = ROOT / (args.output_dir or f"results/revision_grid/{args.variant}")
    runner_lock = output / ".RUNNING"
    if runner_lock.exists() and os.environ.get("LIMA_GATE_RUNNER_OWNS_LOCK") != "1":
        print(f"active runner owns variant directory; skipped: {output}")
        return 0
    records = output / "records"
    metrics_root = output / "metrics"
    traces_root = output / "traces"
    records.mkdir(parents=True, exist_ok=True)

    instance_files: dict[str, dict[str, object]] = {}
    input_files: dict[str, dict[str, object]] = {}
    jobs = []
    for map_name in maps:
        instance = INSTANCES[map_name]
        map_path = ROOT / instance.map_file
        if not map_path.is_file():
            parser.error(f"missing map: {map_path}")
        instance_files[map_name] = {
            "map": instance.map_file,
            "map_sha256": sha256(map_path),
            "scenarios": {},
        }
        input_files[instance.map_file] = {
            "sha256": sha256(map_path), "size_bytes": map_path.stat().st_size
        }
        for density in densities:
            agents = density * instance.tiles // 100
            for scenario in scenarios:
                scenario_file = instance.scenario_template.format(s=scenario)
                scenario_path = ROOT / scenario_file
                if not scenario_path.is_file():
                    parser.error(f"missing scenario: {scenario_path}")
                scenario_description = {
                    "sha256": sha256(scenario_path),
                    "size_bytes": scenario_path.stat().st_size,
                }
                input_files[scenario_file] = scenario_description
                instance_files[map_name]["scenarios"][str(scenario)] = {
                    "path": scenario_file, **scenario_description
                }
                tag = f"{map_name}_d{density:02d}_a{agents}_s{scenario}"
                jobs.append((map_name, density, agents, scenario, instance.map_file, scenario_file, tag))

    source_files = {
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "reference_config": {
            "path": "config/reference_instantiation_v2.json",
            "sha256": sha256(ROOT / "config/reference_instantiation_v2.json"),
        },
    }
    fingerprint_payload = {
        "schema_version": 1,
        "binary_sha256": sha256(binary),
        "profile": "lima-default" if args.variant.startswith("phase2_") else None,
        "profile_version": 1 if args.variant.startswith("phase2_") else None,
        "variant": args.variant,
        "variant_flags": list(VARIANTS[args.variant]),
        "maps": maps,
        "densities": densities,
        "scenarios": scenarios,
        "timeout_seconds": args.timeout,
        "max_steps": args.max_steps,
        "metrics": args.metrics,
        "record_trace": args.record_trace,
        "source_sha256": {key: value["sha256"] for key, value in source_files.items()},
        "input_sha256": {key: value["sha256"] for key, value in input_files.items()},
    }
    experiment_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "experiment_fingerprint": experiment_fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "binary": str(binary.relative_to(ROOT)),
        "binary_sha256": sha256(binary),
        "binary_version": version,
        "git_head": git_head,
        "git_status_tracked": git_status_tracked,
        "allow_dirty": args.allow_dirty,
        "variant": args.variant,
        "variant_flags": list(VARIANTS[args.variant]),
        "maps": maps,
        "densities": densities,
        "scenarios": scenarios,
        "timeout_seconds": args.timeout,
        "max_steps": args.max_steps,
        "metrics": args.metrics,
        "record_trace": args.record_trace,
        "instances": instance_files,
        "inputs": input_files,
        "sources": source_files,
        "job_count": len(jobs),
    }
    manifest_path = output / "MANIFEST.json"
    if manifest_path.is_file() and any(records.glob("*.json")):
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parser.error(f"existing manifest is invalid: {manifest_path}")
        if existing_manifest.get("experiment_fingerprint") != experiment_fingerprint:
            parser.error(
                "output directory already contains records from different inputs, source, "
                "binary, or budgets; choose a new --output-dir"
            )
    write_json_atomic(manifest_path, manifest)

    def run(job) -> tuple[str, str]:
        map_name, density, agents, scenario, map_file, scenario_file, tag = job
        record = records / f"{tag}.json"
        if record.exists() and not args.rerun:
            return tag, "skipped"
        command = [
            str(binary), "--mode", "solve", "--map", map_file,
            "--scenario", scenario_file, "--agents", str(agents),
            "--planner", "bfs", "--seed", str(scenario),
            "--max-steps", str(args.max_steps), *VARIANTS[args.variant],
        ]
        if args.record_trace:
            traces_root.mkdir(parents=True, exist_ok=True)
            command += ["--output", str(traces_root / f"{tag}.txt"), "--validate-conflicts"]
        else:
            command.append("--no-trace")
        if args.metrics:
            command += ["--metrics", str(metrics_root / tag)]

        started = time.time()
        timed_out = False
        try:
            proc = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout
            )
            returncode = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = 124
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        payload = {
            "tag": tag,
            "map": map_name,
            "density_percent": density,
            "agents": agents,
            "scenario": scenario,
            "map_file": map_file,
            "scenario_file": scenario_file,
            "variant": args.variant,
            "command": command,
            "returncode": returncode,
            "timed_out": timed_out,
            "runner_wall_seconds": time.time() - started,
            "summary": parse_summary(stdout),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
        write_json_atomic(record, payload)
        status = "timeout" if timed_out else payload["summary"].get("status", f"rc{returncode}")
        return tag, status

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run, job): job[-1] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            tag, status = future.result()
            completed += 1
            print(f"[{completed:3d}/{len(jobs):3d}] {status:10s} {tag}", flush=True)

    print(f"records: {records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
