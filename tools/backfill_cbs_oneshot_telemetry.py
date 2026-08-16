#!/usr/bin/env python3
"""Backfill compact timed-path telemetry for solved CBS one-shot records."""

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
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from backfill_oneshot_baseline_telemetry import (
    ROOT, atomic_json, distribution, parse_map, parse_scenario, sha256,
    strip_time_wrapper,
)


def parse_stdout(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1])) if lines else {}


def shortest_distance(
    start: tuple[int, int], goal: tuple[int, int], traversable: set[tuple[int, int]]
) -> int:
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        cell, distance = queue.popleft()
        if cell == goal:
            return distance
        x, y = cell
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in traversable and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    raise ValueError(f"unreachable task: {start}->{goal}")


def parse_timed_paths(
    path: Path, starts: list[tuple[int, int]], goals: list[tuple[int, int]],
    traversable: set[tuple[int, int]],
) -> dict:
    paths: list[list[tuple[int, int]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [int(value) for value in line.split()]
        if len(values) < 2 or len(values) % 2:
            raise ValueError("malformed CBS timed path")
        paths.append(list(zip(values[0::2], values[1::2])))
    if len(paths) != len(starts):
        raise ValueError(f"CBS paths {len(paths)} != agents {len(starts)}")

    invalid_moves = 0
    moves: list[int] = []
    waits: list[int] = []
    completion: list[int] = []
    for index, agent_path in enumerate(paths):
        if agent_path[0] != starts[index] or agent_path[-1] != goals[index]:
            raise ValueError(f"CBS endpoint mismatch for agent {index}")
        move_count = wait_count = 0
        for source, target in zip(agent_path, agent_path[1:]):
            if target not in traversable:
                invalid_moves += 1
            distance = abs(source[0] - target[0]) + abs(source[1] - target[1])
            if distance > 1:
                invalid_moves += 1
            elif distance == 0:
                wait_count += 1
            else:
                move_count += 1
        moves.append(move_count)
        waits.append(wait_count)
        completion.append(len(agent_path) - 1)

    vertex_conflicts = edge_conflicts = 0
    makespan = max(completion, default=0)
    for timestep in range(makespan + 1):
        occupant: dict[tuple[int, int], int] = {}
        edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for index, agent_path in enumerate(paths):
            if timestep >= len(agent_path):
                continue
            cell = agent_path[timestep]
            if cell in occupant:
                vertex_conflicts += 1
            else:
                occupant[cell] = index
            if timestep > 0:
                source, target = agent_path[timestep - 1], cell
                if source != target and (target, source) in edges:
                    edge_conflicts += 1
                if source != target:
                    edges.add((source, target))

    shortest = [
        shortest_distance(start, goal, traversable)
        for start, goal in zip(starts, goals)
    ]
    shortest_sum = sum(shortest)
    moves_sum = sum(moves)
    return {
        "trajectory_observed": True,
        "timesteps_observed": makespan,
        "completed_agents": len(paths),
        "residual_agents": 0,
        "agent_completion_ratio": 1.0,
        "completion_steps": distribution(completion),
        "moves_all_agents": distribution(moves),
        "waits_all_agents": distribution(waits),
        "moves_completed_agents": distribution(moves),
        "waits_completed_agents": distribution(waits),
        "moves_total": moves_sum,
        "waits_total": sum(waits),
        "observed_soc": sum(completion),
        "active_agents_per_step": distribution([]),
        "active_agent_steps": sum(completion),
        "path_conformity": {
            "malformed_frames": 0,
            "invalid_moves": invalid_moves,
            "vertex_conflicts": vertex_conflicts,
            "edge_conflicts": edge_conflicts,
            "completed_goal_mismatches": 0,
            "online_validation_ok": (
                invalid_moves == 0 and vertex_conflicts == 0 and edge_conflicts == 0
            ),
        },
        "detour": {
            "shortest_path_moves_sum": shortest_sum,
            "actual_moves_sum": moves_sum,
            "extra_moves_sum": moves_sum - shortest_sum,
            "movement_stretch_aggregate": moves_sum / shortest_sum if shortest_sum else None,
            "waits_sum": sum(waits),
            "reference_route_metrics": None,
            "reference_route_reason": "monolithic baseline has no external reference route",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    campaign = (ROOT / args.campaign).resolve()
    binary = (ROOT / args.binary).resolve()
    output = (ROOT / args.output_dir).resolve()
    records_out, logs_out = output / "records", output / "logs"
    records_out.mkdir(parents=True, exist_ok=True)
    logs_out.mkdir(parents=True, exist_ok=True)
    source_manifest = campaign / "MANIFEST.json"
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((campaign / "records").glob("*.json"))
    ]
    runner = Path(__file__).resolve()
    fingerprint_payload = {
        "schema_version": 1,
        "algorithm": "cbs",
        "source_campaign": str(campaign.relative_to(ROOT)),
        "source_manifest_sha256": sha256(source_manifest),
        "source_record_count": len(records),
        "instrumented_binary": str(binary),
        "instrumented_binary_sha256": sha256(binary),
        "runner": str(runner.relative_to(ROOT)),
        "runner_sha256": sha256(runner),
        "instrumentation": "CBS_TIMED_DUMP=1; search semantics unchanged",
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    atomic_json(output / "MANIFEST.json", {
        **fingerprint_payload,
        "experiment_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": args.jobs,
    })
    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock exists: {lock}")
    lock.write_text(f"pid={os.getpid()}\n", encoding="utf-8")

    def run(record: dict) -> tuple[str, str]:
        tag = record["tag"]
        destination = records_out / f"{tag}_cbs_telemetry.json"
        if destination.is_file():
            return tag, "skipped"
        agents = int(record["agents"])
        command_available = record.get("command") is not None
        result = record.get("result") or {}
        trajectory = None
        equivalence = {"checks": {}, "all_match": True}
        returncode = record.get("returncode")
        started = time.time()
        if record.get("solved") and command_available:
            command = strip_time_wrapper(record["command"])
            command[0] = str(binary)
            with tempfile.TemporaryDirectory(prefix="cbs-telemetry-") as directory:
                dump = Path(directory) / "timed_paths.txt"
                environment = os.environ.copy()
                environment["CBS_DUMP"] = str(dump)
                environment["CBS_TIMED_DUMP"] = "1"
                proc = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True,
                    check=False, env=environment,
                )
                header = parse_stdout(proc.stdout)
                if not dump.is_file():
                    raise RuntimeError(f"CBS replay produced no timed dump: {tag}")
                starts, goals = parse_scenario(ROOT / record["scenario_file"], agents)
                _, _, traversable = parse_map(ROOT / record["map_file"])
                trajectory = parse_timed_paths(dump, starts, goals, traversable)
                checks = {}
                for key in ("solved", "makespan", "soc", "expansions"):
                    old, new = result.get(key), header.get(key)
                    checks[key] = {"original": old, "replay": new, "match": old == new}
                checks["observed_soc"] = {
                    "header": int(header["soc"]),
                    "trajectory": trajectory["observed_soc"],
                    "match": int(header["soc"]) == trajectory["observed_soc"],
                }
                equivalence = {
                    "checks": checks,
                    "all_match": all(check["match"] for check in checks.values()),
                }
                returncode = proc.returncode
                (logs_out / f"{tag}_cbs.log").write_text(
                    proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
                    encoding="utf-8",
                )

        executed = command_available
        solved = bool(record.get("solved"))
        communication = (
            {
                "implementation": "centralized evaluated binary",
                "mode": "global-batch-plan",
                "scope": "global",
                "grid_distance": None,
                "grid_distance_reason": "global solver has no assumed on-map coordinate",
                "joint_participants": distribution([agents]),
                "event_count_by_type": {
                    "agent_agent_direct": 0,
                    "agent_global_solver_task_upload": agents,
                    "agent_global_solver_state_upload": 0,
                    "global_solver_agent_route_delivery": agents if solved else 0,
                    "global_solver_agent_action_delivery": 0,
                },
                "event_count": agents + (agents if solved else 0),
                "payload_by_type": {
                    "task_records": agents,
                    "state_records": 0,
                    "route_waypoints": trajectory["observed_soc"] if trajectory else 0,
                    "actions": 0,
                },
                "payload_units": agents + (trajectory["observed_soc"] if trajectory else 0),
            }
            if executed else {
                "observed": False,
                "reason": "logical record was early-stopped and not executed",
                "event_count": None, "payload_units": None, "scope": None,
            }
        )
        payload = {
            "tag": tag, "algorithm": "cbs", "map": record["map"],
            "target": record["target"], "scenario": record["scenario"],
            "agents": agents,
            "source_record_sha256": sha256(
                campaign / "records" / f"{tag}_cbs.json"
            ),
            "returncode": returncode,
            "replay_wall_seconds": time.time() - started,
            "scalar_equivalence": equivalence,
            "trajectory": trajectory or {
                "trajectory_observed": False,
                "completed_agents": None,
                "agent_completion_ratio": None,
                "path_conformity": None,
            },
            "detour": trajectory["detour"] if trajectory else {},
            "communication": communication,
            "experiment_fingerprint": fingerprint,
        }
        atomic_json(destination, payload)
        if not equivalence["all_match"]:
            return tag, "scalar_mismatch"
        if trajectory and not trajectory["path_conformity"]["online_validation_ok"]:
            return tag, "path_invalid"
        return tag, "backfilled" if trajectory else "not_applicable"

    failures = 0
    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run, record) for record in records]
            for future in concurrent.futures.as_completed(futures):
                tag, status = future.result()
                completed += 1
                failures += status in ("scalar_mismatch", "path_invalid")
                print(f"[{completed:3d}/{len(records):3d}] {status:16s} {tag}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
