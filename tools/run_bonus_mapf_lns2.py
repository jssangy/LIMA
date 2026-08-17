#!/usr/bin/env python3
"""Run official MAPF-LNS2 on the certified disappear-at-terminal instances."""

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


def read_map(path: Path) -> tuple[int, int, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid = lines[4:4 + height]
    if len(grid) != height or any(len(row) != width for row in grid):
        raise ValueError("invalid MovingAI map")
    return width, height, grid


def write_lns_map(source: Path, destination: Path) -> None:
    """Write the same topology in the strict MovingAI alphabet used by LNS2.

    The certified LIMA maps annotate traversable cells with E/G/T. The
    upstream MAPF-LNS2 loader treats every character other than '.' as an
    obstacle, so those semantic labels must be normalized without changing
    the obstacle set.
    """
    lines = source.read_text(encoding="utf-8").splitlines()
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid = lines[4:4 + height]
    if len(grid) != height or any(len(row) != width for row in grid):
        raise ValueError(f"invalid MovingAI map: {source}")
    normalized = lines[:4] + [
        "".join("@" if cell == "@" else "." for cell in row)
        for row in grid
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=destination.parent, delete=False, encoding="utf-8"
    ) as stream:
        stream.write("\n".join(normalized) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, destination)


def read_paths(path: Path, agents: int) -> list[list[tuple[int, int]]]:
    result: list[list[tuple[int, int]]] = [[] for _ in range(agents)]
    pattern = re.compile(r"^Agent\s+(\d+):(.*)$")
    coordinate = re.compile(r"\((\d+),(\d+)\)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        agent = int(match.group(1))
        if not 0 <= agent < agents:
            raise ValueError("path output contains invalid agent id")
        # MAPF-LNS2 prints (row, column); normalize to (x, y).
        result[agent] = [(int(column), int(row))
                         for row, column in coordinate.findall(match.group(2))]
    if any(not path_row for path_row in result):
        raise ValueError("path output is missing one or more agents")
    return result


def validate_paths(paths: list[list[tuple[int, int]]],
                   scenario: list[tuple[int, int, int, int]],
                   map_path: Path, max_steps: int) -> dict:
    width, height, grid = read_map(map_path)
    for agent, (path, endpoints) in enumerate(zip(paths, scenario)):
        sx, sy, gx, gy = endpoints
        if path[0] != (sx, sy) or path[-1] != (gx, gy):
            raise ValueError(f"endpoint mismatch for agent {agent}")
        for x, y in path:
            if not (0 <= x < width and 0 <= y < height) or grid[y][x] == "@":
                raise ValueError(f"blocked or out-of-map path cell for agent {agent}")
        for previous, current in zip(path, path[1:]):
            if abs(previous[0] - current[0]) + abs(previous[1] - current[1]) > 1:
                raise ValueError(f"non-adjacent movement for agent {agent}")
    makespan = max(len(path) - 1 for path in paths)
    if makespan > max_steps:
        return {"execution_limit_ok": False, "makespan": makespan,
                "soc": sum(len(path) - 1 for path in paths)}
    for timestep in range(makespan + 1):
        occupied: dict[tuple[int, int], int] = {}
        for agent, path in enumerate(paths):
            if timestep >= len(path):
                continue  # disappear immediately after first arrival
            position = path[timestep]
            if position in occupied:
                raise ValueError(f"vertex conflict at t={timestep}")
            occupied[position] = agent
        if timestep == 0:
            continue
        edges: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for agent, path in enumerate(paths):
            if timestep >= len(path):
                continue
            edge = (path[timestep - 1], path[timestep])
            if edge[0] != edge[1] and (edge[1], edge[0]) in edges:
                raise ValueError(f"edge conflict at t={timestep}")
            edges[edge] = agent
    return {"execution_limit_ok": True, "makespan": makespan,
            "soc": sum(len(path) - 1 for path in paths)}


def read_result(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        default="results/revision_final/certified_inputs_v3/MANIFEST.json")
    parser.add_argument(
        "--binary", default=str(Path.home() / "mapf-baselines/MAPF-LNS2/build_bonus/lns"))
    parser.add_argument(
        "--source-repo", default=str(Path.home() / "mapf-baselines/MAPF-LNS2"))
    parser.add_argument("--maps", help="optional comma-separated map filter")
    parser.add_argument("--targets", help="optional comma-separated target filter")
    parser.add_argument("--scenarios", help="optional comma-separated scenario filter")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--output-dir", default="results/revision_final/bonus_mapf_lns2_v1")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.max_steps < 1 or args.max_iterations < 1:
        parser.error("invalid jobs or step/search limits")
    binary = Path(args.binary).resolve()
    source_repo = Path(args.source_repo).resolve()
    if not binary.is_file() or not (source_repo / ".git").is_dir():
        parser.error("missing MAPF-LNS2 binary or source repository")
    input_manifest_path = (ROOT / args.input_manifest).resolve()
    inputs = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    map_filter = set(args.maps.split(",")) if args.maps else None
    target_filter = set(args.targets.split(",")) if args.targets else None
    scenario_filter = set(map(int, args.scenarios.split(","))) if args.scenarios else None

    batches: list[tuple[str, str, float, list[dict]]] = []
    all_cells: list[dict] = []
    for map_name, map_entry in inputs["maps"].items():
        if map_filter is not None and map_name not in map_filter:
            continue
        map_path = ROOT / map_entry["map_file"]
        if sha256(map_path) != map_entry["map_sha256"]:
            parser.error(f"map hash mismatch: {map_name}")
        targets = sorted(
            map_entry["targets"].items(), key=lambda item: item[1]["tile_density_percent"])
        for target, target_entry in targets:
            if target_filter is not None and target not in target_filter:
                continue
            cells = []
            for scenario_text, entry in target_entry["scenarios"].items():
                scenario = int(scenario_text)
                if scenario_filter is not None and scenario not in scenario_filter:
                    continue
                for key in ("scenario", "certificate"):
                    path = ROOT / entry[f"{key}_file"]
                    if sha256(path) != entry[f"{key}_sha256"]:
                        parser.error(f"input hash mismatch: {path}")
                cell = {
                    "map": map_name, "target": target,
                    "tile_density_percent": target_entry["tile_density_percent"],
                    "capacity_load_percent": target_entry["capacity_load_percent"],
                    "agents": int(target_entry["agents"]), "scenario": scenario,
                    "map_file": map_entry["map_file"],
                    "scenario_file": entry["scenario_file"],
                    "certificate_file": entry["certificate_file"],
                    "tag": f"{map_name}_{target}_a{target_entry['agents']}_s{scenario}_lns2",
                }
                cells.append(cell)
                all_cells.append(cell)
            if cells:
                batches.append((map_name, target, target_entry["tile_density_percent"], cells))
    if not all_cells:
        parser.error("filters selected no cells")

    output = (ROOT / args.output_dir).resolve()
    records, resources, logs, paths_dir, raw, adapted_maps = (
        output / "records", output / "resources", output / "logs",
        output / "paths", output / "raw", output / "adapted_maps")
    for directory in (records, resources, logs, paths_dir, raw, adapted_maps):
        directory.mkdir(parents=True, exist_ok=True)
    solver_maps: dict[str, Path] = {}
    for map_name in {cell["map"] for cell in all_cells}:
        source_map = ROOT / inputs["maps"][map_name]["map_file"]
        solver_map = adapted_maps / f"{map_name}.map"
        write_lns_map(source_map, solver_map)
        solver_maps[map_name] = solver_map
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip()
    source_diff = subprocess.check_output(
        ["git", "-C", str(source_repo), "diff", "--binary"])
    source_diff_sha256 = hashlib.sha256(source_diff).hexdigest()
    runner = Path(__file__).resolve()
    fingerprint_payload = {
        "schema_version": 1, "algorithm": "mapf-lns2",
        "upstream": "Jiaoyang-Li/MAPF-LNS2", "upstream_commit": upstream_commit,
        "adapter_diff_sha256": source_diff_sha256,
        "binary": str(binary), "binary_sha256": sha256(binary),
        "runner_sha256": sha256(runner),
        "input_manifest_sha256": sha256(input_manifest_path),
        "cells": [cell["tag"] for cell in all_cells],
        "max_execution_steps": args.max_steps,
        "max_high_level_iterations": args.max_iterations,
        "termination_policy": "fixed iteration/execution limits; no wall-clock cutoff",
        "goal_behavior": "unique non-transit terminal; disappear validation",
        "map_adapter": "preserve @ obstacles; normalize traversable ./E/G/T cells to .",
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
        "job_count": len(all_cells), "jobs_concurrency": args.jobs,
        "early_stop_after_zero_success": not args.no_early_stop,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
    })
    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock already exists: {lock}")
    atomic_json(lock, {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()})

    def run(cell: dict) -> tuple[str, bool]:
        tag = cell["tag"]
        record_path = records / f"{tag}.json"
        if record_path.is_file() and not args.rerun:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            return tag, bool(existing.get("solved"))
        map_path = ROOT / cell["map_file"]
        solver_map_path = solver_maps[cell["map"]]
        scenario_path = ROOT / cell["scenario_file"]
        scenario = read_scenario(scenario_path, cell["agents"])
        output_base = raw / tag
        path_file = paths_dir / f"{tag}.txt"
        result_file = Path(str(output_base) + "-LNS.csv")
        resource_path = resources / f"{tag}.txt"
        log_path = logs / f"{tag}.log"
        path_file.unlink(missing_ok=True)
        command = [
            "/usr/bin/time", "-f",
            "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e",
            "-o", str(resource_path), str(binary), "--map", str(solver_map_path),
            "--agents", str(scenario_path), "--agentNum", str(cell["agents"]),
            "--output", str(output_base), "--outputPaths", str(path_file),
            "--cutoffTime", "2147483647", "--maxIterations", "0",
            "--maxInitIterations", str(args.max_iterations),
            "--solver", "LNS", "--initLNS", "1", "--initAlgo", "PP",
            "--replanAlgo", "PP", "--seed", str(cell["scenario"]), "--screen", "0",
        ]
        started = time.time()
        process = subprocess.run(command, cwd=source_repo, capture_output=True, text=True, check=False)
        raw_result = read_result(result_file)
        validation = None
        validation_error = None
        if process.returncode == 0 and path_file.is_file():
            try:
                validation = validate_paths(
                    read_paths(path_file, cell["agents"]), scenario, map_path, args.max_steps)
            except ValueError as error:
                validation_error = str(error)
        is_solved = bool(validation and validation["execution_limit_ok"] and not validation_error)
        result = {
            "solved": "1" if is_solved else "0",
            "status": "completed" if is_solved else "search_limit",
            "makespan": str(validation["makespan"]) if validation else "",
            "soc": str(validation["soc"]) if validation else "",
            "iterations": raw_result.get("iterations", ""),
            "ll_expanded": raw_result.get("LL expanded nodes", ""),
            "ll_generated": raw_result.get("LL generated", ""),
            "upstream_runtime": raw_result.get("runtime", ""),
        }
        log_path.write_text(
            process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else ""),
            encoding="utf-8")
        atomic_json(record_path, {
            **cell, "algorithm": "mapf-lns2", "returncode": process.returncode,
            "timed_out": False, "solved": is_solved, "result": result,
            "validation": validation, "validation_error": validation_error,
            "resource": parse_resource(resource_path),
            "runner_wall_seconds": time.time() - started,
            "path_file": str(path_file.relative_to(ROOT)) if path_file.is_file() else None,
            "raw_result_file": str(result_file) if result_file.is_file() else None,
            "solver_map_file": str(solver_map_path.relative_to(ROOT)),
            "solver_map_sha256": sha256(solver_map_path),
            "command": command, "log": str(log_path.relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        })
        return tag, is_solved

    def early_stop(cell: dict, source_target: str) -> None:
        record_path = records / f"{cell['tag']}.json"
        if record_path.exists() and not args.rerun:
            return
        atomic_json(record_path, {
            **cell, "algorithm": "mapf-lns2", "returncode": None,
            "timed_out": False, "solved": False, "result": {},
            "status": "early_stopped_after_zero_success",
            "early_stop_source_target": source_target,
            "resource": {}, "runner_wall_seconds": 0.0,
            "command": None, "log": None, "experiment_fingerprint": fingerprint,
        })

    stopped_maps: dict[str, str] = {}
    completed = 0
    try:
        for map_name, target, _, cells in batches:
            if map_name in stopped_maps:
                for cell in cells:
                    early_stop(cell, stopped_maps[map_name])
                    completed += 1
                    print(f"[{completed:3d}/{len(all_cells):3d}] early    {cell['tag']}", flush=True)
                continue
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                outcomes = list(pool.map(run, cells))
            successes = sum(is_solved for _, is_solved in outcomes)
            for tag, is_solved in outcomes:
                completed += 1
                print(f"[{completed:3d}/{len(all_cells):3d}] "
                      f"{'solved' if is_solved else 'failed':8s} {tag}", flush=True)
            if not args.no_early_stop and successes == 0 and len(cells) == 10:
                stopped_maps[map_name] = target
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
