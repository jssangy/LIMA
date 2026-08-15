#!/usr/bin/env python3
"""Run the staged Route Planner selection tournament for LIMA.

The default matrix is deliberately small and information-dense: each planner
is evaluated once on each of the three submitted-paper maps at the highest
Phase-2 boundary density completed by the frozen local configuration.  This
gives three screening cells per planner.  Every generated reference-route set
is then executed with the same ``lima-default`` profile and step horizon.

Each job is resumable.  A JSON record is installed atomically only after route
generation and (when possible) LIMA execution have finished.  Planner and
simulation resource measurements are kept separate so parallel screening can
be ranked by step-based outcomes without treating contended wall time as a
scientific result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from generate_route_plans import analyze_route_file


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Instance:
    map_file: str
    scenario_template: str
    tiles: int
    boundary_density: int


INSTANCES = {
    "warehouse_10_20": Instance(
        "data/maps/warehouse_10_20_paper.map",
        "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s{s}.scen",
        2649,
        50,
    ),
    "warehouse_20_40": Instance(
        "data/maps/warehouse_20_40_paper.map",
        "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s{s}.scen",
        10499,
        30,
    ),
    "cross_3030": Instance(
        "data/maps/cross_3030_paper.map",
        "data/scenarios/cross-30-30-paper/cross-30-30-paper_s{s}.scen",
        10200,
        50,
    ),
}


GENERATOR_CANDIDATES = (
    "direct_bfs",
    "direct_astar",
    "jps",
    "randomized_shortest",
    "yen_k",
    "xy_dor",
    "yx_dor",
    "o1turn",
    "romm",
    "valiant",
    "swr",
    "static_highway",
    "static_guidance",
    "sui",
    "tfo_gp",
)
CBS_CANDIDATE = "cbs_incumbent"
CANDIDATES = GENERATOR_CANDIDATES + (CBS_CANDIDATE,)

COMMUNICATION_CLASS = {
    "direct_bfs": "A",
    "direct_astar": "A",
    "jps": "A",
    "randomized_shortest": "A",
    "yen_k": "A",
    "xy_dor": "A",
    "yx_dor": "A",
    "o1turn": "A",
    "romm": "A",
    "valiant": "A",
    "swr": "A",
    "static_highway": "B",
    "static_guidance": "B",
    "sui": "C",
    "tfo_gp": "C",
    "cbs_incumbent": "D",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_description(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def git_text(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else f"ERROR: {proc.stderr.strip()}"


def binary_version(path: Path) -> dict[str, str]:
    completed = subprocess.run(
        [str(path), "--version"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip() or f"exit {completed.returncode}"}
    return parse_fields(completed.stdout)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline=""
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def parse_csv_list(text: str, allowed: Iterable[str], label: str) -> list[str]:
    allowed_set = set(allowed)
    values = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(values) - allowed_set)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {label}: {', '.join(unknown)}")
    if not values:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    return list(dict.fromkeys(values))


def parse_int_list(text: str, allowed: set[int], label: str) -> list[int]:
    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = (int(value) for value in token.split("-", 1))
            values.update(range(lo, hi + 1))
        else:
            values.add(int(token))
    if not values or not values.issubset(allowed):
        raise argparse.ArgumentTypeError(
            f"{label} must be a non-empty subset of {sorted(allowed)}"
        )
    return sorted(values)


def parse_fields(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", lines[-1]))


def read_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return parse_fields(text)


def read_resource(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    fields = dict(
        re.findall(
            r"^([A-Za-z_][A-Za-z0-9_]*)=([^\n]+)$",
            path.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
    )
    parsed: dict[str, Any] = {}
    for key, value in fields.items():
        try:
            parsed[key] = float(value) if "." in value else int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def int_field(row: dict[str, str], key: str) -> int:
    try:
        return int(row.get(key, "0"))
    except (TypeError, ValueError):
        return 0


def derive_metrics(directory: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "completion_count": 0,
        "completion_p50": None,
        "completion_p90": None,
        "completion_p99": None,
        "solver_invocations": 0,
        "solver_expanded": 0,
        "solver_wall_us": 0,
        "solver_fallbacks": 0,
        "discharge_events": 0,
        "discharged_agents": 0,
    }
    agents_file = directory / "agents.csv"
    if agents_file.is_file():
        with agents_file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        completions = [
            float(int_field(row, "completion_step"))
            for row in rows
            if int_field(row, "completed") == 1
        ]
        result["completion_count"] = len(completions)
        result["completion_p50"] = percentile(completions, 0.50)
        result["completion_p90"] = percentile(completions, 0.90)
        result["completion_p99"] = percentile(completions, 0.99)

    solver_file = directory / "solver_invocations.csv"
    if solver_file.is_file():
        with solver_file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        result["solver_invocations"] = len(rows)
        result["solver_expanded"] = sum(int_field(row, "expanded") for row in rows)
        result["solver_wall_us"] = sum(int_field(row, "wall_us") for row in rows)
        result["solver_fallbacks"] = sum(int_field(row, "fallback") for row in rows)

    discharge_file = directory / "discharge_events.csv"
    if discharge_file.is_file():
        with discharge_file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        result["discharge_events"] = len(rows)
        result["discharged_agents"] = sum(int_field(row, "rerouted") for row in rows)
    return result


def validate_route_file(path: Path, agents: int) -> tuple[bool, str, dict[str, Any]]:
    if not path.is_file():
        return False, "route file was not created", {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) != agents:
        return False, f"expected {agents} route lines, found {len(lines)}", {
            "route_lines": len(lines)
        }
    waypoints: list[int] = []
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) < 2 or len(fields) % 2:
            return False, f"route line {index} is empty or has an odd coordinate count", {
                "route_lines": len(lines)
            }
        try:
            [int(value) for value in fields]
        except ValueError:
            return False, f"route line {index} contains a non-integer coordinate", {
                "route_lines": len(lines)
            }
        waypoints.append(len(fields) // 2)
    return True, "ok", {
        "route_lines": len(lines),
        "route_waypoints_total": sum(waypoints),
        "route_waypoints_mean": statistics.fmean(waypoints) if waypoints else 0.0,
        "route_waypoints_max": max(waypoints, default=0),
        "route_file_bytes": path.stat().st_size,
        "route_sha256": sha256(path),
    }


@dataclass(frozen=True)
class ProcessResult:
    command: list[str]
    returncode: int
    timed_out: bool
    wall_seconds: float
    stdout: str
    stderr: str
    resource: dict[str, Any]


def run_process(
    command: list[str],
    timeout: float,
    resource_file: Path,
    env: dict[str, str] | None = None,
) -> ProcessResult:
    resource_file.parent.mkdir(parents=True, exist_ok=True)
    resource_file.unlink(missing_ok=True)
    timed_command = [
        "/usr/bin/time",
        "-f",
        "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S",
        "-o",
        str(resource_file),
        *command,
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        timed_command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        returncode = 124
    return ProcessResult(
        command=command,
        returncode=returncode,
        timed_out=timed_out,
        wall_seconds=time.monotonic() - started,
        stdout=stdout,
        stderr=stderr,
        resource=read_resource(resource_file),
    )


def process_payload(result: ProcessResult) -> dict[str, Any]:
    return {
        "command": result.command,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "runner_wall_seconds": result.wall_seconds,
        "resource": result.resource,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


@dataclass(frozen=True)
class Job:
    candidate: str
    map_name: str
    density: int
    agents: int
    scenario: int
    map_path: Path
    scenario_path: Path
    tag: str


def build_jobs(
    candidates: list[str],
    maps: list[str],
    scenarios: list[int],
    smoke: bool,
    include_low_density: bool,
) -> list[Job]:
    cells: list[tuple[str, int, int, int, Path, Path, str]] = []
    selected_maps = maps[:1] if smoke else maps
    selected_scenarios = scenarios[:1] if smoke else scenarios
    for map_name in selected_maps:
        spec = INSTANCES[map_name]
        densities = [10] if smoke else (
            [10, spec.boundary_density] if include_low_density else [spec.boundary_density]
        )
        for density in densities:
            agents = density * spec.tiles // 100
            for scenario in selected_scenarios:
                map_path = (ROOT / spec.map_file).resolve()
                scenario_path = (ROOT / spec.scenario_template.format(s=scenario)).resolve()
                tag = f"{map_name}_d{density:02d}_a{agents}_s{scenario}"
                cells.append((map_name, density, agents, scenario, map_path, scenario_path, tag))
    # Interleave candidates within each cell.  This avoids running an entire
    # candidate under a systematically different machine-load interval.
    return [
        Job(candidate, map_name, density, agents, scenario, map_path, scenario_path, tag)
        for map_name, density, agents, scenario, map_path, scenario_path, tag in cells
        for candidate in candidates
    ]


def generator_command(generator: Path, job: Job, route: Path, stats: Path) -> list[str]:
    return [
        sys.executable,
        str(generator),
        "--planner",
        job.candidate,
        "--map",
        str(job.map_path),
        "--scenario",
        str(job.scenario_path),
        "--agents",
        str(job.agents),
        "--seed",
        str(job.scenario),
        "--output",
        str(route),
        "--stats-output",
        str(stats),
    ]


def cbs_command(cbs: Path, job: Job, budget: float) -> list[str]:
    return [
        str(cbs),
        "--map",
        str(job.map_path),
        "--scenario",
        str(job.scenario_path),
        "--agents",
        str(job.agents),
        "--time-limit",
        str(budget),
    ]


def lima_command(
    binary: Path,
    job: Job,
    route: Path,
    metrics: Path,
    max_steps: int,
    smoke: bool,
    trace: Path,
) -> list[str]:
    command = [
        str(binary),
        "--mode",
        "solve",
        "--profile",
        "lima-default",
        "--map",
        str(job.map_path),
        "--scenario",
        str(job.scenario_path),
        "--agents",
        str(job.agents),
        "--seed",
        str(job.scenario),
        "--max-steps",
        str(max_steps),
        "--routes",
        str(route),
        "--metrics",
        str(metrics),
    ]
    if smoke:
        command += ["--output", str(trace), "--validate-conflicts"]
    else:
        command.append("--no-trace")
    return command


def record_status(record: dict[str, Any]) -> str:
    if record.get("planner", {}).get("timed_out"):
        return "planner_timeout"
    if not record.get("route_validation", {}).get("valid"):
        return "planner_failed"
    simulation = record.get("simulation")
    if not simulation:
        return "not_run"
    if simulation.get("timed_out"):
        return "watchdog"
    return simulation.get("summary", {}).get("status", f"rc{simulation.get('returncode')}")


def summarize_records(records_dir: Path, candidates: list[str], output: Path) -> None:
    records: list[dict[str, Any]] = []
    for path in sorted(records_dir.rglob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    columns = [
        "candidate",
        "communication_class",
        "runs",
        "route_valid",
        "planner_failures",
        "planner_timeouts",
        "simulation_watchdogs",
        "completed_runs",
        "completed_agents",
        "total_agents",
        "agent_completion_fraction",
        "median_completed_steps",
        "worst_run_p99_completion_step",
        "median_planner_user_seconds",
        "max_planner_rss_kb",
        "median_sim_user_seconds",
        "max_sim_rss_kb",
        "solver_invocations",
        "discharge_events",
        "median_mean_stretch",
        "max_stretch",
        "max_vertex_load",
        "max_directed_edge_load",
        "median_contraflow_ratio",
        "fallback_fraction",
    ]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        selected = [row for row in records if row.get("candidate") == candidate]
        route_valid = [row for row in selected if row.get("route_validation", {}).get("valid")]
        planner_timeouts = sum(bool(row.get("planner", {}).get("timed_out")) for row in selected)
        simulations = [row for row in route_valid if row.get("simulation")]
        watchdogs = sum(bool(row["simulation"].get("timed_out")) for row in simulations)
        completed = [
            row
            for row in simulations
            if row["simulation"].get("summary", {}).get("status") == "completed"
        ]
        completed_steps = [
            int(row["simulation"]["summary"]["steps"])
            for row in completed
            if str(row["simulation"].get("summary", {}).get("steps", "")).isdigit()
        ]
        completion_values = [
            float(row.get("metrics", {}).get("completion_p99"))
            for row in simulations
            if row.get("metrics", {}).get("completion_p99") is not None
        ]
        completed_agents = 0
        total_agents = 0
        for row in simulations:
            completed_text = row["simulation"].get("summary", {}).get("completed", "0/0")
            try:
                done, total = completed_text.split("/", 1)
                completed_agents += int(done)
                total_agents += int(total)
            except (AttributeError, ValueError):
                pass
        planner_user = [
            float(row["planner"].get("resource", {}).get("user_seconds"))
            for row in selected
            if row.get("planner", {}).get("resource", {}).get("user_seconds") is not None
        ]
        planner_rss = [
            int(row["planner"].get("resource", {}).get("max_rss_kb"))
            for row in selected
            if row.get("planner", {}).get("resource", {}).get("max_rss_kb") is not None
        ]
        sim_user = [
            float(row["simulation"].get("resource", {}).get("user_seconds"))
            for row in simulations
            if row["simulation"].get("resource", {}).get("user_seconds") is not None
        ]
        sim_rss = [
            int(row["simulation"].get("resource", {}).get("max_rss_kb"))
            for row in simulations
            if row["simulation"].get("resource", {}).get("max_rss_kb") is not None
        ]
        planner_stats = [
            row.get("planner", {}).get("stats", {})
            for row in route_valid
        ]
        mean_stretches = [
            float(stats["mean_stretch"])
            for stats in planner_stats
            if stats.get("mean_stretch") is not None
        ]
        max_stretches = [
            float(stats["max_stretch"])
            for stats in planner_stats
            if stats.get("max_stretch") is not None
        ]
        max_vertex_loads = [
            int(stats["max_vertex_load"])
            for stats in planner_stats
            if stats.get("max_vertex_load") is not None
        ]
        max_edge_loads = [
            int(stats["max_directed_edge_load"])
            for stats in planner_stats
            if stats.get("max_directed_edge_load") is not None
        ]
        contraflow_ratios = [
            float(stats["contraflow_ratio"])
            for stats in planner_stats
            if stats.get("contraflow_ratio") is not None
        ]
        fallback_count = sum(int(stats.get("fallback_count", 0)) for stats in planner_stats)
        planned_agents = sum(int(stats.get("agents", 0)) for stats in planner_stats)
        rows.append({
            "candidate": candidate,
            "communication_class": COMMUNICATION_CLASS[candidate],
            "runs": len(selected),
            "route_valid": len(route_valid),
            "planner_failures": len(selected) - len(route_valid),
            "planner_timeouts": planner_timeouts,
            "simulation_watchdogs": watchdogs,
            "completed_runs": len(completed),
            "completed_agents": completed_agents,
            "total_agents": total_agents,
            "agent_completion_fraction": completed_agents / total_agents if total_agents else 0.0,
            "median_completed_steps": statistics.median(completed_steps) if completed_steps else "",
            "worst_run_p99_completion_step": max(completion_values) if completion_values else "",
            "median_planner_user_seconds": statistics.median(planner_user) if planner_user else "",
            "max_planner_rss_kb": max(planner_rss, default=""),
            "median_sim_user_seconds": statistics.median(sim_user) if sim_user else "",
            "max_sim_rss_kb": max(sim_rss, default=""),
            "solver_invocations": sum(int(row.get("metrics", {}).get("solver_invocations", 0)) for row in simulations),
            "discharge_events": sum(int(row.get("metrics", {}).get("discharge_events", 0)) for row in simulations),
            "median_mean_stretch": statistics.median(mean_stretches) if mean_stretches else "",
            "max_stretch": max(max_stretches, default=""),
            "max_vertex_load": max(max_vertex_loads, default=""),
            "max_directed_edge_load": max(max_edge_loads, default=""),
            "median_contraflow_ratio": statistics.median(contraflow_ratios) if contraflow_ratios else "",
            "fallback_fraction": fallback_count / planned_agents if planned_agents else 0.0,
        })

    csv_path = output / "summary.csv"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output, delete=False, newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    os.replace(temporary, csv_path)

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in rows
    ]
    atomic_text(
        output / "SUMMARY.md",
        "# Route Planner tournament summary\n\n"
        + f"Records found: {len(records)}. Ranking must use completion first; "
          "runner wall time is operational metadata only.\n\n"
        + "\n".join([header, separator, *body])
        + "\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build_phase2/lima")
    parser.add_argument("--generator", default="tools/generate_route_plans.py")
    parser.add_argument("--cbs", default="build_phase2/cbs_baseline")
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--maps", default=",".join(INSTANCES))
    parser.add_argument("--scenarios", default="0")
    parser.add_argument(
        "--include-low-density",
        action="store_true",
        help="also run a 10%% cell per selected map (off for the three-cell screen)",
    )
    parser.add_argument("--max-steps", type=int, default=7000)
    parser.add_argument("--planner-timeout", type=float, default=300.0)
    parser.add_argument("--simulation-timeout", type=float, default=2400.0)
    parser.add_argument("--cbs-time-budget", type=float, default=300.0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--cbs-jobs", type=int, default=2)
    parser.add_argument("--output-dir", default="results/phase3_route_planner_screen")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="allow exploratory runs from tracked source changes or a mismatched binary",
    )
    args = parser.parse_args()

    try:
        candidates = parse_csv_list(args.candidates, CANDIDATES, "candidate")
        maps = parse_csv_list(args.maps, INSTANCES, "map")
        scenarios = parse_int_list(args.scenarios, set(range(10)), "scenarios")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.max_steps < 1 or args.jobs < 1 or args.cbs_jobs < 1:
        parser.error("max-steps, jobs, and cbs-jobs must be positive")
    if min(args.planner_timeout, args.simulation_timeout, args.cbs_time_budget) <= 0:
        parser.error("timeouts and CBS time budget must be positive")

    binary = (ROOT / args.binary).resolve()
    generator = (ROOT / args.generator).resolve()
    cbs = (ROOT / args.cbs).resolve()
    if not binary.is_file():
        parser.error(f"missing LIMA binary: {binary}")
    if (
        not args.dry_run
        and any(candidate in GENERATOR_CANDIDATES for candidate in candidates)
        and not generator.is_file()
    ):
        parser.error(f"missing route generator: {generator}")
    if CBS_CANDIDATE in candidates and not cbs.is_file():
        parser.error(f"missing CBS binary: {cbs}")

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

    output = (ROOT / args.output_dir).resolve()
    records_dir = output / "records"
    routes_dir = output / "routes"
    planner_stats_dir = output / "planner_stats"
    metrics_root = output / "metrics"
    traces_dir = output / "traces"
    resources_dir = output / "resources"
    logs_dir = output / "logs"
    for directory in (records_dir, routes_dir, planner_stats_dir, resources_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(candidates, maps, scenarios, args.smoke, args.include_low_density)
    effective_max_steps = min(args.max_steps, 1000) if args.smoke else args.max_steps
    source_files = {
        "runner": file_description(Path(__file__).resolve()),
        "generator": file_description(generator),
        "lima_main_source": file_description(ROOT / "app/main.cpp"),
        "cbs_source": file_description(ROOT / "app/cbs_baseline.cpp"),
        "reference_config": file_description(ROOT / "config/reference_instantiation_v1.json"),
    }
    input_files: dict[str, dict[str, Any]] = {}
    for job in jobs:
        for path in (job.map_path, job.scenario_path):
            input_files[str(path.relative_to(ROOT))] = file_description(path)

    planned_commands: list[dict[str, Any]] = []
    for job in jobs:
        route = routes_dir / job.candidate / f"{job.tag}.txt"
        stats = planner_stats_dir / job.candidate / f"{job.tag}.json"
        metrics = metrics_root / job.candidate / job.tag
        trace = traces_dir / job.candidate / f"{job.tag}.txt"
        planner_cmd = (
            cbs_command(cbs, job, args.cbs_time_budget)
            if job.candidate == CBS_CANDIDATE
            else generator_command(generator, job, route, stats)
        )
        planned_commands.append({
            "candidate": job.candidate,
            "tag": job.tag,
            "planner": planner_cmd,
            "planner_environment": {"CBS_DUMP": str(route)} if job.candidate == CBS_CANDIDATE else {},
            "lima": lima_command(binary, job, route, metrics, effective_max_steps, args.smoke, trace),
        })

    fingerprint_payload = {
        "schema_version": 1,
        "ranking_budget": {"simulation_steps": effective_max_steps},
        "operational_guards": {
            "planner_timeout_seconds": args.planner_timeout,
            "simulation_timeout_seconds": args.simulation_timeout,
            "cbs_time_budget_seconds": args.cbs_time_budget,
        },
        "profile": "lima-default",
        "profile_version": 1,
        "candidates": candidates,
        "maps": maps,
        "scenarios": scenarios[:1] if args.smoke else scenarios,
        "smoke": args.smoke,
        "include_low_density": args.include_low_density,
        "binary_sha256": file_description(binary)["sha256"],
        "cbs_binary_sha256": file_description(cbs)["sha256"],
        "source_sha256": {
            key: value["sha256"] for key, value in source_files.items()
        },
        "input_sha256": {
            key: value["sha256"] for key, value in input_files.items()
        },
    }
    experiment_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "experiment_fingerprint": experiment_fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_scope": "one-shot, disappear-at-goal, fixed submitted-paper scenarios",
        "ranking_budget": {"simulation_steps": effective_max_steps},
        "operational_guards": {
            "planner_timeout_seconds": args.planner_timeout,
            "simulation_timeout_seconds": args.simulation_timeout,
            "cbs_time_budget_seconds": args.cbs_time_budget,
        },
        "profile": "lima-default",
        "profile_version": 1,
        "candidates": candidates,
        "communication_class": {candidate: COMMUNICATION_CLASS[candidate] for candidate in candidates},
        "maps": maps,
        "scenarios": scenarios[:1] if args.smoke else scenarios,
        "smoke": args.smoke,
        "include_low_density": args.include_low_density,
        "jobs": args.jobs,
        "cbs_jobs": args.cbs_jobs,
        "job_count": len(jobs),
        "git_head": git_head,
        "git_status_tracked": git_status_tracked,
        "binary_version": version,
        "allow_dirty": args.allow_dirty,
        "binary": file_description(binary),
        "cbs_binary": file_description(cbs),
        "sources": source_files,
        "inputs": input_files,
        "commands": planned_commands,
    }
    manifest_path = output / "MANIFEST.json"
    if manifest_path.is_file() and any(records_dir.rglob("*.json")):
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parser.error(f"existing manifest is invalid: {manifest_path}")
        if existing_manifest.get("experiment_fingerprint") != experiment_fingerprint:
            parser.error(
                "output directory already contains records from different inputs or budgets; "
                "choose a new --output-dir"
            )
    atomic_json(manifest_path, manifest)

    if args.dry_run:
        for command in planned_commands:
            print(f"[{command['candidate']}] {command['tag']}")
            print("  planner:", " ".join(command["planner"]))
            print("  lima:   ", " ".join(command["lima"]))
        print(f"dry-run jobs={len(jobs)} manifest={output / 'MANIFEST.json'}")
        return 0

    cbs_semaphore = threading.Semaphore(args.cbs_jobs)

    def run_job(job: Job) -> tuple[str, str, str]:
        record_path = records_dir / job.candidate / f"{job.tag}.json"
        if record_path.is_file() and not args.rerun:
            try:
                existing = json.loads(record_path.read_text(encoding="utf-8"))
                return job.candidate, job.tag, f"skipped:{record_status(existing)}"
            except json.JSONDecodeError:
                pass

        route_path = routes_dir / job.candidate / f"{job.tag}.txt"
        stats_path = planner_stats_dir / job.candidate / f"{job.tag}.json"
        metrics_dir = metrics_root / job.candidate / job.tag
        trace_path = traces_dir / job.candidate / f"{job.tag}.txt"
        planner_resource = resources_dir / job.candidate / f"{job.tag}_planner.txt"
        sim_resource = resources_dir / job.candidate / f"{job.tag}_lima.txt"
        log_path = logs_dir / job.candidate / f"{job.tag}.log"
        for path in (route_path, stats_path, planner_resource, sim_resource, log_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=routes_dir, prefix=f".{job.candidate}_{job.tag}_") as tmp:
            temporary_route = Path(tmp) / "routes.txt"
            temporary_stats = Path(tmp) / "stats.json"
            if job.candidate == CBS_CANDIDATE:
                command = cbs_command(cbs, job, args.cbs_time_budget)
                environment = dict(os.environ)
                environment["CBS_DUMP"] = str(temporary_route)
                with cbs_semaphore:
                    planner_result = run_process(
                        command,
                        max(args.planner_timeout, args.cbs_time_budget + 20.0),
                        planner_resource,
                        environment,
                    )
                cbs_stats = parse_fields(planner_result.stdout)
                planner_stats = {"cbs": cbs_stats}
            else:
                command = generator_command(generator, job, temporary_route, temporary_stats)
                planner_result = run_process(command, args.planner_timeout, planner_resource)
                planner_stats = read_stats(temporary_stats)

            valid, reason, route_stats = validate_route_file(temporary_route, job.agents)
            if planner_result.timed_out:
                valid, reason = False, "planner process exceeded its operational timeout"
            elif planner_result.returncode != 0:
                valid, reason = False, f"planner process returned {planner_result.returncode}"
            if valid:
                if job.candidate == CBS_CANDIDATE:
                    measured = analyze_route_file(
                        job.map_path,
                        job.scenario_path,
                        job.agents,
                        temporary_route,
                        job.candidate,
                        job.scenario,
                    )
                    measured["cbs"] = cbs_stats
                    planner_stats = measured
                route_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_route, route_path)
                if temporary_stats.is_file():
                    os.replace(temporary_stats, stats_path)

        record: dict[str, Any] = {
            "candidate": job.candidate,
            "communication_class": COMMUNICATION_CLASS[job.candidate],
            "tag": job.tag,
            "map": job.map_name,
            "density_percent": job.density,
            "agents": job.agents,
            "scenario": job.scenario,
            "max_steps": effective_max_steps,
            "planner": {**process_payload(planner_result), "stats": planner_stats},
            "route_validation": {"valid": valid, "reason": reason, **route_stats},
            "simulation": None,
            "metrics": {},
        }

        log_text = "[planner stdout]\n" + planner_result.stdout
        if planner_result.stderr:
            log_text += "\n[planner stderr]\n" + planner_result.stderr

        if valid:
            sim_command = lima_command(
                binary,
                job,
                route_path,
                metrics_dir,
                effective_max_steps,
                args.smoke,
                trace_path,
            )
            simulation = run_process(sim_command, args.simulation_timeout, sim_resource)
            record["simulation"] = {
                **process_payload(simulation),
                "summary": parse_fields(simulation.stdout),
            }
            record["metrics"] = derive_metrics(metrics_dir)
            log_text += "\n[lima stdout]\n" + simulation.stdout
            if simulation.stderr:
                log_text += "\n[lima stderr]\n" + simulation.stderr

        atomic_text(log_path, log_text)
        record["log"] = str(log_path.relative_to(ROOT))
        atomic_json(record_path, record)
        return job.candidate, job.tag, record_status(record)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            completed += 1
            try:
                candidate, tag, status = future.result()
            except Exception as error:  # keep other resumable jobs alive
                candidate, tag, status = job.candidate, job.tag, f"runner_error:{error}"
            print(
                f"[{completed:3d}/{len(jobs):3d}] {candidate:20s} {status:22s} {tag}",
                flush=True,
            )

    summarize_records(records_dir, candidates, output)
    print(f"records: {records_dir}")
    print(f"summary: {output / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
