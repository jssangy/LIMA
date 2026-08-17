#!/usr/bin/env python3
"""Run the official PIBT transition rule on fixed lifelong goal sequences."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COMM_KEYS = (
    "pibt_comm_active_agent_steps",
    "pibt_comm_root_invocations",
    "pibt_comm_inheritance_requests",
    "pibt_comm_backtracking_responses",
    "pibt_comm_backtracking_valid",
    "pibt_comm_backtracking_invalid",
    "pibt_comm_max_propagation_depth",
    "pibt_comm_state_priority_announcements",
    "pibt_comm_decision_announcements",
    "pibt_comm_propagation_events",
    "pibt_comm_distributed_logical_events",
)


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


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_pibt_map(source: Path, destination: Path) -> None:
    """Canonicalize line framing while preserving every map-cell symbol.

    The upstream Grid reader consumes rows until EOF rather than stopping at
    the declared height. Some source maps contain a harmless trailing blank
    line, which it misinterprets as an extra grid row.
    """
    lines = source.read_text(encoding="utf-8").splitlines()
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid = lines[4:4 + height]
    if len(grid) != height or any(len(row) != width for row in grid):
        raise ValueError(f"invalid MovingAI map: {source}")
    atomic_text(destination, "\n".join(lines[:4] + grid) + "\n")


def parse_fields(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1])) if lines else {}


def parse_resource(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(re.findall(
        r"^(\w+)=([^\n]+)$",
        path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))


def read_scenario(path: Path, agents: int) -> list[tuple[int, int, int, int]]:
    rows: list[tuple[int, int, int, int]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].lower() == "version":
                continue
            if len(fields) < 8:
                raise ValueError(f"malformed scenario row in {path}")
            rows.append(tuple(map(int, fields[4:8])))
            if len(rows) == agents:
                break
    if len(rows) != agents:
        raise ValueError(f"scenario has {len(rows)} rows, expected {agents}")
    return rows


def validate_sequences(path: Path, scenario: list[tuple[int, int, int, int]]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(scenario):
        raise ValueError("goal-sequence line count differs from agent count")
    for index, (line, (_, _, goal_x, goal_y)) in enumerate(zip(lines, scenario)):
        coordinates = list(map(int, line.split()))
        if len(coordinates) < 4 or len(coordinates) % 2:
            raise ValueError(f"invalid sequence for agent {index}")
        goals = list(zip(coordinates[::2], coordinates[1::2]))
        if goals[0] != (goal_x, goal_y):
            raise ValueError(f"first sequence goal mismatch for agent {index}")
        if any(goals[position] == goals[(position + 1) % len(goals)]
               for position in range(len(goals))):
            raise ValueError(f"cyclic consecutive duplicate for agent {index}")


def instance_text(map_reference: str, scenario: list[tuple[int, int, int, int]], seed: int,
                  horizon: int) -> str:
    lines = [
        f"map_file={map_reference}", f"agents={len(scenario)}", f"seed={seed}",
        "random_problem=0", f"max_timestep={horizon}",
        "max_comp_time=2147483647",
    ]
    lines.extend(f"{sx},{sy},{gx},{gy}" for sx, sy, gx, gy in scenario)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        default="results/revision_final/lifelong_inputs_v2/MANIFEST.json")
    parser.add_argument(
        "--binary", default=str(Path.home() / "mapf-baselines/pibt2/build_bonus/lifelong_fixed"))
    parser.add_argument(
        "--adapter-source", default=str(Path.home() / "mapf-baselines/pibt2/lifelong_fixed.cpp"))
    parser.add_argument("--maps", help="optional comma-separated map filter")
    parser.add_argument("--densities", help="optional comma-separated density filter")
    parser.add_argument("--scenarios", help="optional comma-separated scenario filter")
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--output-dir", default="results/revision_final/bonus_lifelong_pibt_v1")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.horizon <= args.warmup or args.warmup < 0:
        parser.error("invalid jobs, horizon, or warmup")
    binary = Path(args.binary).resolve()
    source = Path(args.adapter_source).resolve()
    if not binary.is_file() or not source.is_file():
        parser.error("missing PIBT lifelong adapter binary or source")
    input_manifest_path = (ROOT / args.input_manifest).resolve()
    inputs = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if inputs.get("semantics", {}).get("sequence") != "fixed cyclic order; no consecutive duplicate":
        parser.error("input manifest is not the fixed cyclic lifelong dataset")
    maps = set(args.maps.split(",")) if args.maps else None
    densities = set(map(int, args.densities.split(","))) if args.densities else None
    scenarios = set(map(int, args.scenarios.split(","))) if args.scenarios else None

    cells: list[dict] = []
    for map_name, map_entry in inputs["maps"].items():
        if maps is not None and map_name not in maps:
            continue
        map_path = ROOT / map_entry["map_file"]
        if sha256(map_path) != map_entry["map_sha256"]:
            parser.error(f"map hash mismatch: {map_name}")
        for density_text, density_entry in map_entry["densities"].items():
            density = int(density_text)
            if densities is not None and density not in densities:
                continue
            agents = int(density_entry["agents"])
            for scenario_text, entry in density_entry["scenarios"].items():
                scenario = int(scenario_text)
                if scenarios is not None and scenario not in scenarios:
                    continue
                for key in ("scenario", "sequence", "certificate"):
                    path = ROOT / entry[f"{key}_file"]
                    if sha256(path) != entry[f"{key}_sha256"]:
                        parser.error(f"input hash mismatch: {path}")
                cells.append({
                    "map": map_name, "density": density, "agents": agents,
                    "scenario": scenario, "map_file": map_entry["map_file"],
                    "scenario_file": entry["scenario_file"],
                    "sequence_file": entry["sequence_file"],
                    "certificate_file": entry["certificate_file"],
                    "tag": f"{map_name}_d{density:02d}_a{agents}_s{scenario}_pibt",
                })
    if not cells:
        parser.error("filters selected no cells")

    output = (ROOT / args.output_dir).resolve()
    records = output / "records"
    resources = output / "resources"
    logs = output / "logs"
    metrics = output / "metrics"
    instances = output / "instances"
    adapted_maps = output / "adapted_maps"
    for directory in (records, resources, logs, metrics, instances, adapted_maps):
        directory.mkdir(parents=True, exist_ok=True)
    solver_maps: dict[str, Path] = {}
    for map_name in {cell["map"] for cell in cells}:
        source_map = ROOT / inputs["maps"][map_name]["map_file"]
        solver_map = adapted_maps / f"{map_name}.map"
        write_pibt_map(source_map, solver_map)
        solver_maps[map_name] = solver_map
    upstream = binary.parents[1]
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    runner = Path(__file__).resolve()
    fingerprint_payload = {
        "schema_version": 1, "algorithm": "pibt",
        "semantic_scope": "lifelong; fixed cyclic per-agent goal sequences",
        "upstream": "Kei18/pibt2", "upstream_commit": upstream_commit,
        "adapter_source": str(source), "adapter_source_sha256": sha256(source),
        "binary": str(binary), "binary_sha256": sha256(binary),
        "runner_sha256": sha256(runner),
        "input_manifest_sha256": sha256(input_manifest_path),
        "cells": [cell["tag"] for cell in cells],
        "horizon_steps": args.horizon, "warmup_steps": args.warmup,
        "termination_policy": "fixed discrete horizon; no wall-clock cutoff",
        "map_adapter": "preserve grid symbols; emit exactly the declared map rows",
        "adapted_map_sha256": {
            name: sha256(path) for name, path in sorted(solver_maps.items())
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "MANIFEST.json"
    if manifest_path.is_file() and any(records.glob("*.json")):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_fingerprint") != fingerprint:
            parser.error("output directory contains a different experiment fingerprint")
    atomic_json(manifest_path, {
        **fingerprint_payload, "experiment_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest_path.relative_to(ROOT)),
        "job_count": len(cells), "jobs_concurrency": args.jobs,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
    })
    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock already exists: {lock}")
    atomic_json(lock, {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()})

    def run(cell: dict) -> tuple[str, str]:
        tag = cell["tag"]
        record_path = records / f"{tag}.json"
        if record_path.is_file() and not args.rerun:
            return tag, "skipped"
        scenario_path = ROOT / cell["scenario_file"]
        sequence_path = ROOT / cell["sequence_file"]
        map_path = solver_maps[cell["map"]]
        scenario_rows = read_scenario(scenario_path, cell["agents"])
        validate_sequences(sequence_path, scenario_rows)
        instance_path = instances / f"{tag}.txt"
        # Upstream pibt2 compiles a fixed map-directory prefix into lib-mapf.
        # The adapter changes only file framing, preserving every grid symbol.
        map_reference = os.path.relpath(map_path, upstream / "map")
        atomic_text(instance_path, instance_text(
            map_reference, scenario_rows, cell["scenario"], args.horizon))
        metric_dir = metrics / tag
        metric_dir.mkdir(parents=True, exist_ok=True)
        events_path = metric_dir / "task_completions.csv"
        resource_path = resources / f"{tag}.txt"
        log_path = logs / f"{tag}.log"
        command = [
            "/usr/bin/time", "-f",
            "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e",
            "-o", str(resource_path), str(binary), "--instance", str(instance_path),
            "--sequences", str(sequence_path), "--horizon", str(args.horizon),
            "--events", str(events_path),
        ]
        started = time.time()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        fields = parse_fields(process.stdout)
        communication_counts = {
            key.removeprefix("pibt_comm_"): int(match.group(1))
            for key in COMM_KEYS
            if (match := re.search(
                rf"(?:^|\s){re.escape(key)}=(-?\d+)(?:\s|$)",
                process.stdout, re.MULTILINE))
        }
        if len(communication_counts) != len(COMM_KEYS):
            missing = sorted(
                key.removeprefix("pibt_comm_")
                for key in COMM_KEYS
                if key.removeprefix("pibt_comm_") not in communication_counts)
            raise RuntimeError(f"missing PIBT communication counters: {missing}")
        normal = (
            process.returncode == 0 and fields.get("status") == "step_limit"
            and fields.get("steps") == str(args.horizon)
            and fields.get("vertex_conflicts") == "0" and fields.get("edge_conflicts") == "0"
        )
        event_rows = []
        if events_path.is_file():
            with events_path.open(newline="", encoding="utf-8") as stream:
                event_rows = list(csv.DictReader(stream))
        log_path.write_text(
            process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else ""),
            encoding="utf-8")
        atomic_json(record_path, {
            **cell, "algorithm": "pibt", "variant": "native-lifelong",
            "returncode": process.returncode, "timed_out": False,
            "horizon_completed": normal, "normal_termination": normal,
            "runner_wall_seconds": time.time() - started, "result": fields,
            "resource": parse_resource(resource_path),
            "telemetry": {"path_conformity": {
                "online_validation_ok": normal, "vertex_conflicts": 0 if normal else None,
                "edge_conflicts": 0 if normal else None,
            }},
            "communication": {
                "model": "PIBT native decentralized logical broadcast-event lower bound",
                "scope": "robot-to-robot; direct communication within two graph hops",
                "counts": communication_counts,
                "maximum_direct_radius_hops": 2,
                "recipient_weighted_transmissions": None,
                "recipient_weighted_reason": (
                    "PIBT bounds direct communication by two-hop proximity but does not "
                    "specify a MAC or broadcast-recipient accounting model"
                ),
            },
            "task_event_rows": len(event_rows),
            "metrics": str(metric_dir.relative_to(ROOT)),
            "instance": str(instance_path.relative_to(ROOT)),
            "solver_map_file": str(map_path.relative_to(ROOT)),
            "solver_map_sha256": sha256(map_path),
            "command": command, "log": str(log_path.relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        })
        return tag, "horizon" if normal else "stopped"

    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run, cell) for cell in cells]
            for future in concurrent.futures.as_completed(futures):
                tag, status = future.result()
                completed += 1
                print(f"[{completed:3d}/{len(cells):3d}] {status:8s} {tag}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
