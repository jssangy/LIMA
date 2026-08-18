#!/usr/bin/env python3
"""Prepare a certified PIBT instance and execute native one-step stochastic PIBT."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from stochastic_replan_adapter import (
    ensure_pibt_map,
    parse_scenario,
    write_pibt_instance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--scen", required=True, type=Path)
    parser.add_argument("--agents", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--delay-prob", required=True, type=float)
    parser.add_argument("--delay-seed", required=True, type=int)
    parser.add_argument("--pibt-repo", required=True, type=Path)
    parser.add_argument("--native-binary", required=True, type=Path)
    parser.add_argument("--exclusive-boundary-goals", action="store_true")
    args = parser.parse_args()
    if args.agents < 1 or args.max_steps < 1 or not 0.0 <= args.delay_prob <= 1.0:
        parser.error("invalid agents, max-steps, or delay probability")

    map_path = args.map.resolve()
    scenario_path = args.scen.resolve()
    repo = args.pibt_repo.resolve()
    binary = args.native_binary.resolve()
    if not binary.is_file():
        parser.error(f"missing native PIBT binary: {binary}")
    starts, goals = parse_scenario(scenario_path, args.agents)
    map_name = ensure_pibt_map(map_path, repo)

    with tempfile.TemporaryDirectory(prefix="lima_pibt_native_") as directory:
        instance = Path(directory) / "instance.txt"
        write_pibt_instance(
            instance, map_name, starts, goals, args.seed, args.max_steps
        )
        command = [
            str(binary),
            "--instance", str(instance),
            "--max-steps", str(args.max_steps),
            "--delay-prob", str(args.delay_prob),
            "--delay-seed", str(args.delay_seed),
        ]
        if args.exclusive_boundary_goals:
            command.append("--exclusive-boundary-goals")
        completed = subprocess.run(
            command, cwd=repo, text=True, capture_output=True, check=False
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
