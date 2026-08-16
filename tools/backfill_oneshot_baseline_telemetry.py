#!/usr/bin/env python3
"""Backfill compact trajectory and communication telemetry for one-shot baselines.

The canonical experiments used the upstream short-log option, which retained
only scalar solution costs.  This tool replays an already-recorded LaCAM or
PIBT command with the *same* binary, input, seed, and deterministic budget,
temporarily requests the full path, validates it while streaming, and stores
only compact telemetry in a separate provenance-linked campaign.  Canonical
records and solutions are never modified.

Communication is counted at logical actor boundaries.  Both evaluated binaries
construct the complete deterministic trajectory before deployment: every agent
uploads one task record and, on success, receives one route whose payload is
counted in waypoint records.  Executing an already-installed route locally is
not communication.  Direct robot--robot messages are zero for these centralized
binaries; server-internal memory reads are not double-counted as network
messages.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COORD_RE = re.compile(r"\((-?\d+),(-?\d+)\)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def number(value, cast=float):
    try:
        result = cast(value)
    except (TypeError, ValueError):
        return None
    if isinstance(result, float) and not math.isfinite(result):
        return None
    return result


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def distribution(values) -> dict:
    present = [float(value) for value in values]
    if not present:
        return {
            "count": 0, "min": None, "mean": None, "max": None,
            "variance": None, "p50": None, "p90": None, "p99": None,
        }
    return {
        "count": len(present),
        "min": min(present),
        "mean": statistics.fmean(present),
        "max": max(present),
        "variance": statistics.pvariance(present),
        "p50": percentile(present, 0.50),
        "p90": percentile(present, 0.90),
        "p99": percentile(present, 0.99),
    }


def parse_scenario(path: Path, agents: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rows = [
        line.split() for line in path.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    if len(rows) != agents:
        raise ValueError(f"scenario size {len(rows)} != agents {agents}: {path}")
    starts = [(int(row[4]), int(row[5])) for row in rows]
    goals = [(int(row[6]), int(row[7])) for row in rows]
    return starts, goals


def parse_map(path: Path) -> tuple[int, int, set[tuple[int, int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])
        offset = lines.index("map") + 1
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid MovingAI map: {path}") from error
    rows = lines[offset:offset + height]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError(f"map dimensions do not match header: {path}")
    traversable = {
        (x, y) for y, row in enumerate(rows) for x, value in enumerate(row)
        if value not in "@OTW"
    }
    return width, height, traversable


def scalar_header(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if line == "solution=":
                break
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def validate_and_summarize_path(
    path: Path,
    starts: list[tuple[int, int]],
    goals: list[tuple[int, int]],
    traversable: set[tuple[int, int]],
) -> dict:
    agents = len(starts)
    completion: list[int | None] = [None] * agents
    moves = [0] * agents
    waits = [0] * agents
    active = [True] * agents
    active_counts: list[int] = []
    previous: list[tuple[int, int]] | None = None
    starts_in_log = goals_in_log = None
    in_solution = False
    last_timestep = -1
    invalid_moves = vertex_conflicts = edge_conflicts = 0
    malformed_frames = 0

    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.rstrip("\n")
            if not in_solution:
                if line.startswith("starts="):
                    starts_in_log = [
                        (int(x), int(y)) for x, y in COORD_RE.findall(line)
                    ]
                elif line.startswith("goals="):
                    goals_in_log = [
                        (int(x), int(y)) for x, y in COORD_RE.findall(line)
                    ]
                elif line == "solution=":
                    in_solution = True
                continue

            if not line:
                continue
            timestep_text, separator, payload = line.partition(":")
            if not separator:
                malformed_frames += 1
                continue
            timestep = int(timestep_text)
            positions = [(int(x), int(y)) for x, y in COORD_RE.findall(payload)]
            if len(positions) != agents or timestep != last_timestep + 1:
                malformed_frames += 1
                last_timestep = timestep
                previous = positions if len(positions) == agents else previous
                continue
            if any(position not in traversable for position in positions):
                invalid_moves += sum(position not in traversable for position in positions)

            if timestep == 0:
                if positions != starts:
                    malformed_frames += 1
                previous = positions
                last_timestep = timestep
                continue

            assert previous is not None
            active_before = [index for index, value in enumerate(active) if value]
            active_counts.append(len(active_before))
            destinations: dict[tuple[int, int], int] = {}
            directed_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for index in active_before:
                source, destination = previous[index], positions[index]
                distance = abs(source[0] - destination[0]) + abs(source[1] - destination[1])
                if distance > 1:
                    invalid_moves += 1
                if distance == 0:
                    waits[index] += 1
                else:
                    moves[index] += 1
                if destination in destinations:
                    vertex_conflicts += 1
                else:
                    destinations[destination] = index
                if source != destination:
                    if (destination, source) in directed_edges:
                        edge_conflicts += 1
                    directed_edges.add((source, destination))

            for index in active_before:
                if positions[index] == goals[index]:
                    completion[index] = timestep
                    active[index] = False

            previous = positions
            last_timestep = timestep

    if starts_in_log is not None and starts_in_log != starts:
        malformed_frames += 1
    if goals_in_log is not None and goals_in_log != goals:
        malformed_frames += 1
    completed = [index for index, value in enumerate(completion) if value is not None]
    completion_values = [completion[index] for index in completed]
    completed_moves = [moves[index] for index in completed]
    completed_waits = [waits[index] for index in completed]
    return {
        "trajectory_observed": in_solution and last_timestep >= 0,
        "timesteps_observed": max(0, last_timestep),
        "completed_agents": len(completed),
        "residual_agents": agents - len(completed),
        "agent_completion_ratio": len(completed) / agents if agents else None,
        "completion_steps": distribution(completion_values),
        "moves_all_agents": distribution(moves),
        "waits_all_agents": distribution(waits),
        "moves_completed_agents": distribution(completed_moves),
        "waits_completed_agents": distribution(completed_waits),
        "moves_total": sum(moves),
        "waits_total": sum(waits),
        "observed_soc": sum(completion_values),
        "active_agents_per_step": distribution(active_counts),
        "active_agent_steps": sum(active_counts),
        "path_conformity": {
            "malformed_frames": malformed_frames,
            "invalid_moves": invalid_moves,
            "vertex_conflicts": vertex_conflicts,
            "edge_conflicts": edge_conflicts,
            "completed_goal_mismatches": 0,
            "online_validation_ok": (
                malformed_frames == 0 and invalid_moves == 0
                and vertex_conflicts == 0 and edge_conflicts == 0
            ),
        },
    }


def communication(algorithm: str, agents: int, solved: bool, trajectory: dict, header: dict) -> dict:
    task_uploads = agents
    route_deliveries = agents if solved else 0
    route_waypoints = (
        int(trajectory["observed_soc"]) if solved else 0
    )
    counts = {
        "agent_agent_direct": 0,
        "agent_global_solver_task_upload": task_uploads,
        "agent_global_solver_state_upload": 0,
        "global_solver_agent_route_delivery": route_deliveries,
        "global_solver_agent_action_delivery": 0,
    }
    payload = {
        "task_records": task_uploads,
        "state_records": 0,
        "route_waypoints": route_waypoints,
        "actions": 0,
    }
    mode = "global-batch-plan"
    participants = distribution([agents])
    return {
        "implementation": "centralized evaluated binary",
        "mode": mode,
        "scope": "global",
        "grid_distance": None,
        "grid_distance_reason": "global solver has no assumed on-map coordinate",
        "joint_participants": participants,
        "event_count_by_type": counts,
        "event_count": sum(counts.values()),
        "payload_by_type": payload,
        "payload_units": sum(payload.values()),
        "server_internal_memory_events_counted": False,
    }


def strip_time_wrapper(command: list[str]) -> list[str]:
    if not command or command[0] != "/usr/bin/time":
        return list(command)
    try:
        output_index = command.index("-o")
    except ValueError as error:
        raise ValueError("unrecognized /usr/bin/time wrapper") from error
    return list(command[output_index + 2:])


def replay_command(record: dict, solution_path: Path) -> list[str]:
    command = strip_time_wrapper(record.get("command") or [])
    if not command:
        return []
    command = [item for item in command if item not in ("--log-short", "--log_short")]
    try:
        output_index = command.index("--output")
    except ValueError as error:
        raise ValueError(f"record has no --output: {record['tag']}") from error
    command[output_index + 1] = str(solution_path)
    return command


def scalar_equivalence(record: dict, header: dict, trajectory: dict) -> dict:
    original = record.get("result") or {}
    checks: dict[str, dict] = {}
    for key in ("solved", "makespan", "soc"):
        if key not in original or key not in header:
            continue
        old = number(original[key], int)
        new = number(header[key], int)
        checks[key] = {"original": old, "replay": new, "match": old == new}
    if header.get("solved") == "1" and trajectory["trajectory_observed"]:
        checks["observed_soc"] = {
            "header": number(header.get("soc"), int),
            "trajectory": trajectory["observed_soc"],
            "match": number(header.get("soc"), int) == trajectory["observed_soc"],
        }
        checks["completed_agents"] = {
            "expected": int(record["agents"]),
            "trajectory": trajectory["completed_agents"],
            "match": trajectory["completed_agents"] == int(record["agents"]),
        }
    return {
        "checks": checks,
        "all_match": all(check["match"] for check in checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=("lacam", "pibt"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--maps")
    parser.add_argument("--targets")
    parser.add_argument("--scenarios")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    campaign = (ROOT / args.campaign).resolve()
    output = (ROOT / args.output_dir).resolve()
    records_out = output / "records"
    logs_out = output / "logs"
    records_out.mkdir(parents=True, exist_ok=True)
    logs_out.mkdir(parents=True, exist_ok=True)
    source_manifest = campaign / "MANIFEST.json"
    if not source_manifest.is_file():
        parser.error(f"missing source manifest: {source_manifest}")
    source_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((campaign / "records").glob("*.json"))
    ]
    map_filter = set(args.maps.split(",")) if args.maps else None
    target_filter = set(args.targets.split(",")) if args.targets else None
    scenario_filter = set(map(int, args.scenarios.split(","))) if args.scenarios else None
    source_records = [
        record for record in source_records
        if record.get("algorithm") == args.algorithm
        and (map_filter is None or record["map"] in map_filter)
        and (target_filter is None or str(record["target"]) in target_filter)
        and (scenario_filter is None or int(record["scenario"]) in scenario_filter)
    ]
    if not source_records:
        parser.error("filters selected no records")

    runner = Path(__file__).resolve()
    fingerprint_payload = {
        "schema_version": 1,
        "algorithm": args.algorithm,
        "source_campaign": str(campaign.relative_to(ROOT)),
        "source_manifest_sha256": sha256(source_manifest),
        "source_record_count": len(source_records),
        "runner": str(runner.relative_to(ROOT)),
        "runner_sha256": sha256(runner),
        "communication_accounting": {
            "robot_peer_and_robot_server_are_communication": True,
            "server_internal_memory_is_not_network_communication": True,
            "lacam": "N task uploads plus N route deliveries on success; route payload in waypoints",
            "pibt": "N task uploads plus N route deliveries on success; cached-route execution is communication-free",
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = output / "MANIFEST.json"
    if manifest_path.is_file() and any(records_out.glob("*.json")):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_fingerprint") != fingerprint:
            parser.error("output directory contains a different telemetry fingerprint")
    atomic_json(manifest_path, {
        **fingerprint_payload,
        "experiment_fingerprint": fingerprint,
        "jobs": args.jobs,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })

    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock exists: {lock}")
    lock.write_text(f"pid={os.getpid()}\n", encoding="utf-8")

    def backfill(record: dict) -> tuple[str, str]:
        tag = record["tag"]
        destination = records_out / f"{tag}_{args.algorithm}_telemetry.json"
        if destination.is_file() and not args.rerun:
            return tag, "skipped"
        agents = int(record["agents"])
        scenario_path = ROOT / record["scenario_file"]
        map_path = ROOT / record["map_file"]
        starts, goals = parse_scenario(scenario_path, agents)
        _, _, traversable = parse_map(map_path)
        command_available = bool(record.get("command"))
        # LaCAM exposes no executable partial trajectory when joint search fails.
        needs_replay = command_available and (args.algorithm == "pibt" or bool(record.get("solved")))
        started = time.time()
        if needs_replay:
            with tempfile.TemporaryDirectory(prefix=f"{args.algorithm}-telemetry-") as directory:
                solution_path = Path(directory) / "solution.txt"
                command = replay_command(record, solution_path)
                proc = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True, check=False
                )
                if not solution_path.is_file():
                    raise RuntimeError(f"replay produced no solution log: {tag}")
                header = scalar_header(solution_path)
                trajectory = validate_and_summarize_path(
                    solution_path, starts, goals, traversable
                )
                log_path = logs_out / f"{tag}_{args.algorithm}.log"
                log_path.write_text(
                    proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
                    encoding="utf-8",
                )
                returncode = proc.returncode
                command_record = command
        else:
            header = {key: str(value) for key, value in (record.get("result") or {}).items()}
            trajectory = {
                "trajectory_observed": False,
                "timesteps_observed": 0,
                "completed_agents": None,
                "residual_agents": None,
                "agent_completion_ratio": None,
                "completion_steps": distribution([]),
                "moves_all_agents": distribution([]),
                "waits_all_agents": distribution([]),
                "moves_completed_agents": distribution([]),
                "waits_completed_agents": distribution([]),
                "moves_total": None,
                "waits_total": None,
                "observed_soc": None,
                "active_agents_per_step": distribution([]),
                "active_agent_steps": 0,
                "path_conformity": None,
            }
            returncode = record.get("returncode")
            command_record = None

        replay_solved = header.get("solved") == "1"
        equivalence = scalar_equivalence(record, header, trajectory)
        shortest_sum = number(header.get("soc_lb") or header.get("lb_soc"), int)
        moves_total = trajectory.get("moves_total")
        detour = {
            "shortest_path_moves_sum": shortest_sum,
            "actual_moves_sum": moves_total,
            "extra_moves_sum": (
                moves_total - shortest_sum
                if moves_total is not None and shortest_sum is not None else None
            ),
            "movement_stretch_aggregate": (
                moves_total / shortest_sum
                if moves_total is not None and shortest_sum else None
            ),
            "waits_sum": trajectory.get("waits_total"),
            "reference_route_metrics": None,
            "reference_route_reason": "monolithic baseline has no external reference route",
        }
        payload = {
            "tag": tag,
            "algorithm": args.algorithm,
            "map": record["map"],
            "target": record["target"],
            "scenario": record["scenario"],
            "agents": agents,
            "source_record": str(
                (campaign / "records" / f"{tag}_{args.algorithm}.json").relative_to(ROOT)
            ),
            "source_record_sha256": sha256(
                campaign / "records" / f"{tag}_{args.algorithm}.json"
            ),
            "source_experiment_fingerprint": record.get("experiment_fingerprint"),
            "returncode": returncode,
            "replay_wall_seconds": time.time() - started,
            "replay_command": command_record,
            "scalar_header": header,
            "scalar_equivalence": equivalence,
            "trajectory": trajectory,
            "detour": detour,
            "communication": (
                communication(args.algorithm, agents, replay_solved, trajectory, header)
                if command_available else {
                    "observed": False,
                    "reason": "logical record was early-stopped and not executed",
                    "event_count": None,
                    "payload_units": None,
                    "scope": None,
                }
            ),
            "experiment_fingerprint": fingerprint,
        }
        atomic_json(destination, payload)
        if not equivalence["all_match"]:
            return tag, "scalar_mismatch"
        if trajectory.get("path_conformity") and not trajectory["path_conformity"]["online_validation_ok"]:
            return tag, "path_invalid"
        return tag, "backfilled" if needs_replay else "not_applicable"

    completed = 0
    failures = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(backfill, record) for record in source_records]
            for future in concurrent.futures.as_completed(futures):
                tag, status = future.result()
                completed += 1
                if status in ("scalar_mismatch", "path_invalid"):
                    failures += 1
                print(f"[{completed:3d}/{len(source_records):3d}] {status:16s} {tag}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    if failures:
        raise SystemExit(f"telemetry validation failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
