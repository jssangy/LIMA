#!/usr/bin/env python3
"""Build paper-facing one-shot tables without changing frozen experiments.

Success is always reported on the complete certified matrix.  Makespan and
completion-cost comparisons are reported only on explicitly paired successful
instances, preventing high-density failures from silently changing the sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAMPAIGNS = {
    "lima": ROOT / "results/revision_final/oneshot_lima_certified_step_v4_optimized",
    "cbs": ROOT / "results/revision_final/oneshot_cbs_certified_step_v3",
    "lacam": ROOT / "results/revision_final/oneshot_lacam_certified_step_v3",
    "pibt": ROOT / "results/revision_final/oneshot_pibt_certified_step_v3",
    "primal2": ROOT / "results/revision_final/oneshot_primal2_certified_step_v6_common5000_stall256",
}
DEFAULT_TELEMETRY = {
    "cbs": ROOT / "results/revision_final/oneshot_cbs_telemetry_backfill_v1",
    "lacam": ROOT / "results/revision_final/oneshot_lacam_telemetry_backfill_v2",
    "pibt": ROOT / "results/revision_final/oneshot_pibt_telemetry_backfill_v2",
}


def numeric(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def nested(payload: dict, *keys: str):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def cell_key(record: dict) -> tuple[str, str, int]:
    return record["map"], str(record["target"]), int(record["scenario"])


def solved(record: dict) -> bool:
    return bool(record.get("solved")) and not bool(record.get("timed_out"))


def parse_completed(value) -> tuple[int, int] | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    left, right = value.split("/", 1)
    try:
        completed, total = int(left), int(right)
    except ValueError:
        return None
    if total <= 0 or completed < 0 or completed > total:
        return None
    return completed, total


def lima_metrics_dir(campaign: Path, record: dict) -> Path:
    """Resolve metrics even when an optimized campaign reuses source artifacts."""
    local_dir = campaign / "metrics" / record["tag"]
    if (local_dir / "agents.csv").is_file():
        return local_dir

    command = record.get("command") or []
    try:
        metrics_arg = Path(command[command.index("--metrics") + 1])
    except (ValueError, IndexError, TypeError):
        return local_dir
    return metrics_arg if metrics_arg.is_absolute() else ROOT / metrics_arg


def overlay_record(root: Path | None, record: dict, algorithm: str) -> dict:
    if root is None:
        return {}
    path = root / "records" / f"{record['tag']}_{algorithm}_telemetry.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def record_metrics(
    algorithm: str, campaign: Path, record: dict, primary_horizon: int,
    overlay: dict | None = None,
) -> dict:
    overlay = overlay or {}
    result = record.get("result") or record.get("summary") or {}
    extended_solved = solved(record)
    span = numeric(result.get("makespan"))
    if span is None and extended_solved:
        span = numeric(result.get("steps"))
    is_solved = bool(
        extended_solved and span is not None and span <= primary_horizon
    )
    agents = int(record.get("agents", 0))
    soc = numeric(result.get("soc")) if is_solved else None
    completion_mean = soc / agents if soc is not None and agents else None
    completion_p50 = completion_p90 = completion_p99 = completion_max = None
    censored_completion_mean = None
    completed_agents = None
    parsed_completed = parse_completed(result.get("completed"))
    record_horizon = numeric(nested(record, "execution_horizon", "max_steps"))
    if (
        parsed_completed
        and parsed_completed[1] == agents
        and (
            is_solved
            or (
                algorithm == "primal2"
                and record_horizon is not None
                and record_horizon <= primary_horizon
            )
        )
    ):
        completed_agents = parsed_completed[0]
    elif is_solved:
        completed_agents = agents

    if algorithm == "lima":
        agent_rows = read_csv(lima_metrics_dir(campaign, record) / "agents.csv")
        all_completion_steps = [
            numeric(row.get("completion_step")) or 0.0 for row in agent_rows
        ]
        completion_steps = [
            value if 0 < value <= primary_horizon else 0.0
            for value in all_completion_steps
        ]
        completed_steps = [value for value in completion_steps if value > 0]
        if all_completion_steps:
            completed_agents = len(completed_steps)
        if completed_steps and len(completed_steps) == agents:
            soc = sum(completed_steps)
            completion_mean = statistics.fmean(completed_steps)
            completion_p50 = percentile(completed_steps, 0.50)
            completion_p90 = percentile(completed_steps, 0.90)
            completion_p99 = percentile(completed_steps, 0.99)
            completion_max = max(completed_steps)
        if all_completion_steps:
            censored = [
                value if 0 < value <= primary_horizon else float(primary_horizon)
                for value in all_completion_steps
            ]
            censored_completion_mean = statistics.fmean(censored)

    overlay_trajectory = overlay.get("trajectory") or {}
    if overlay_trajectory.get("trajectory_observed"):
        parsed_overlay_completed = numeric(overlay_trajectory.get("completed_agents"))
        completed_agents = (
            int(parsed_overlay_completed) if parsed_overlay_completed is not None else None
        )
        completion_summary = overlay_trajectory.get("completion_steps") or {}
        if completed_agents:
            soc = numeric(overlay_trajectory.get("observed_soc"))
            completion_mean = numeric(completion_summary.get("mean"))
            completion_p50 = numeric(completion_summary.get("p50"))
            completion_p90 = numeric(completion_summary.get("p90"))
            completion_p99 = numeric(completion_summary.get("p99"))
            completion_max = numeric(completion_summary.get("max"))

    resource = record.get("resource") or {}
    telemetry = record.get("telemetry") or {}
    steps = numeric(result.get("steps")) or span
    event_count = numeric(nested(telemetry, "communication", "event_count"))
    common_comm = overlay.get("communication") or {}
    common_comm_counts = common_comm.get("event_count_by_type") or {}
    common_comm_payload = common_comm.get("payload_by_type") or {}
    local_comm_events = event_count if algorithm == "lima" else 0.0
    if algorithm == "lima":
        agent_rows = read_csv(lima_metrics_dir(campaign, record) / "agents.csv")
        route_waypoints = sum(
            max(0, int(float(row.get("initial_route_len", 0))) - 1)
            for row in agent_rows
        )
        common_comm_counts = {
            "agent_agent_direct": 0,
            "agent_global_solver_task_upload": agents,
            "agent_global_solver_state_upload": 0,
            "global_solver_agent_route_delivery": agents,
            "global_solver_agent_action_delivery": 0,
            "agent_intersection_acquisition": int(nested(
                telemetry, "communication", "event_count_by_type", "acquisition") or 0),
            "intersection_agent_broadcast": int(nested(
                telemetry, "communication", "event_count_by_type", "broadcast") or 0),
            "intersection_intersection_gate_signal": int(nested(
                telemetry, "communication", "event_count_by_type", "gate_signal") or 0),
        }
        common_comm_payload = {
            "task_records": agents,
            "state_records": common_comm_counts["agent_intersection_acquisition"],
            "route_waypoints": route_waypoints,
            "actions_or_local_decisions": common_comm_counts["intersection_agent_broadcast"],
            "gate_records": common_comm_counts["intersection_intersection_gate_signal"],
        }
        common_comm = {
            "scope": "local+global-initialization",
            "event_count": sum(common_comm_counts.values()),
            "payload_units": sum(common_comm_payload.values()),
            "joint_participants": {"mean": agents, "max": agents},
        }
    elif not common_comm and record.get("command") is not None:
        # Exact scalar-only communication accounting.  Path backfill later adds
        # completion tails, moves/waits, and per-step PIBT participant tails.
        if algorithm in ("cbs", "lacam"):
            deliveries = agents if is_solved else 0
            route_waypoints = int(soc) if soc is not None else 0
            common_comm_counts = {
                "agent_agent_direct": 0,
                "agent_global_solver_task_upload": agents,
                "agent_global_solver_state_upload": 0,
                "global_solver_agent_route_delivery": deliveries,
                "global_solver_agent_action_delivery": 0,
            }
            common_comm_payload = {
                "task_records": agents, "state_records": 0,
                "route_waypoints": route_waypoints, "actions": 0,
            }
            common_comm = {
                "scope": "global", "event_count": sum(common_comm_counts.values()),
                "payload_units": sum(common_comm_payload.values()),
                "joint_participants": {"mean": agents, "max": agents},
            }
        elif algorithm == "pibt" and soc is not None:
            common_comm_counts = {
                "agent_agent_direct": 0,
                "agent_global_solver_task_upload": agents,
                "agent_global_solver_state_upload": 0,
                "global_solver_agent_route_delivery": agents,
                "global_solver_agent_action_delivery": 0,
            }
            common_comm_payload = {
                "task_records": agents, "state_records": 0,
                "route_waypoints": int(soc), "actions": 0,
            }
            common_comm = {
                "scope": "global", "event_count": sum(common_comm_counts.values()),
                "payload_units": sum(common_comm_payload.values()),
                "joint_participants": {"mean": None, "max": agents},
            }
        elif algorithm == "primal2":
            # The current scalar-only adapter proves the message count for a
            # batch trajectory install, but does not retain SOC/waypoint tails.
            deliveries = agents if is_solved else 0
            common_comm_counts = {
                "agent_agent_direct": 0,
                "agent_global_solver_task_upload": agents,
                "agent_global_solver_state_upload": 0,
                "global_solver_agent_route_delivery": deliveries,
                "global_solver_agent_action_delivery": 0,
            }
            common_comm = {
                "scope": "global",
                "event_count": sum(common_comm_counts.values()),
                "payload_units": None,
                "joint_participants": {"mean": agents, "max": agents},
                "payload_gap": "scalar adapter did not retain route waypoint count",
            }
    common_event_count = numeric(common_comm.get("event_count"))
    common_payload_units = numeric(common_comm.get("payload_units"))
    overlay_detour = overlay.get("detour") or {}
    overlay_path = overlay_trajectory.get("path_conformity") or {}
    return {
        "algorithm": algorithm,
        "map": record["map"], "target": record["target"],
        "density_percent": record.get("tile_density_percent"),
        "capacity_load_percent": record.get("capacity_load_percent"),
        "scenario": int(record["scenario"]), "agents": agents,
        "status": (
            "primary_step_limit_extended_success"
            if extended_solved and not is_solved
            else record.get(
                "status", result.get("status", "completed" if is_solved else "failed")
            )
        ),
        "solved": int(is_solved), "early_stopped": int(record.get("status") == "early_stopped_after_zero_success"),
        "extended_solved": int(extended_solved),
        "primary_execution_horizon": primary_horizon,
        "makespan": span, "soc": soc, "completion_mean": completion_mean,
        "completion_p50": completion_p50, "completion_p90": completion_p90,
        "completion_p99": completion_p99, "completion_max": completion_max,
        "censored_completion_mean": censored_completion_mean,
        "completed_agents": completed_agents,
        "residual_agents": agents - completed_agents if completed_agents is not None else None,
        "agent_completion_ratio": (
            completed_agents / agents if completed_agents is not None and agents else None),
        "cpu_seconds": (numeric(resource.get("user_seconds")) or 0.0) + (numeric(resource.get("system_seconds")) or 0.0),
        "max_rss_kb": numeric(resource.get("max_rss_kb")),
        "comm_events": common_event_count,
        "comm_local_events": local_comm_events,
        "comm_global_or_server_events": (
            common_event_count - local_comm_events
            if common_event_count is not None and local_comm_events is not None else None),
        "comm_payload_units": common_payload_units,
        "comm_scope": common_comm.get("scope"),
        "comm_joint_participants_mean": numeric(nested(common_comm, "joint_participants", "mean")),
        "comm_joint_participants_max": numeric(nested(common_comm, "joint_participants", "max")),
        "comm_agent_agent_direct": numeric(common_comm_counts.get("agent_agent_direct")),
        "comm_agent_solver_task_upload": numeric(common_comm_counts.get("agent_global_solver_task_upload")),
        "comm_agent_solver_state_upload": numeric(common_comm_counts.get("agent_global_solver_state_upload")),
        "comm_solver_agent_route_delivery": numeric(common_comm_counts.get("global_solver_agent_route_delivery")),
        "comm_solver_agent_action_delivery": numeric(common_comm_counts.get("global_solver_agent_action_delivery")),
        "comm_events_per_agent_step": (
            common_event_count / (agents * steps)
            if common_event_count is not None and agents and steps else None),
        "comm_distance_min": numeric(nested(telemetry, "communication", "distance_cells", "min")),
        "comm_distance_mean": numeric(nested(telemetry, "communication", "distance_cells", "mean")),
        "comm_distance_max": numeric(nested(telemetry, "communication", "distance_cells", "max")),
        "comm_distance_variance": numeric(nested(telemetry, "communication", "distance_cells", "variance")),
        "comm_distance_p99": numeric(nested(telemetry, "communication", "distance_cells", "p99")),
        "comm_hops_min": numeric(nested(telemetry, "communication", "intersection_hops", "min")),
        "comm_hops_mean": numeric(nested(telemetry, "communication", "intersection_hops", "mean")),
        "comm_hops_max": numeric(nested(telemetry, "communication", "intersection_hops", "max")),
        "comm_hops_variance": numeric(nested(telemetry, "communication", "intersection_hops", "variance")),
        "comm_hops_p99": numeric(nested(telemetry, "communication", "intersection_hops", "p99")),
        "execution_over_reference_mean": numeric(nested(telemetry, "detour", "all_agents", "execution_over_reference", "mean")),
        "execution_over_reference_p90": numeric(nested(telemetry, "detour", "all_agents", "execution_over_reference", "p90")),
        "extra_moves_mean": numeric(nested(telemetry, "detour", "all_agents", "extra_moves", "mean")),
        "common_moves_total": numeric(overlay_trajectory.get("moves_total")),
        "common_waits_total": numeric(overlay_trajectory.get("waits_total")),
        "shortest_path_moves_sum": numeric(overlay_detour.get("shortest_path_moves_sum")),
        "shortest_path_extra_moves_sum": numeric(overlay_detour.get("extra_moves_sum")),
        "shortest_path_movement_stretch": numeric(overlay_detour.get("movement_stretch_aggregate")),
        "recirculation_events": numeric(nested(telemetry, "detour", "recirculation_loop_cells", "count")),
        "vertex_conflicts": numeric(
            overlay_path.get("vertex_conflicts")
            if overlay_path else nested(telemetry, "path_conformity", "vertex_conflicts")),
        "edge_conflicts": numeric(
            overlay_path.get("edge_conflicts")
            if overlay_path else nested(telemetry, "path_conformity", "edge_conflicts")),
        "invalid_moves": numeric(
            overlay_path.get("invalid_moves")
            if overlay_path else nested(telemetry, "path_conformity", "invalid_moves")),
        "rejoin_failures": numeric(nested(telemetry, "path_conformity", "rejoin_failures")),
        "goal_preservation_failures": numeric(nested(telemetry, "path_conformity", "goal_preservation_failures")),
        "online_validation_ok": (
            overlay_path.get("online_validation_ok")
            if overlay_path else nested(telemetry, "path_conformity", "online_validation_ok")),
        "trajectory_backfill": bool(overlay_trajectory.get("trajectory_observed")),
        "trajectory_scalar_equivalence": nested(overlay, "scalar_equivalence", "all_match"),
    }


def median(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign", action="append", default=[], metavar="NAME=PATH",
        help="repeatable campaign override; defaults to the frozen revision campaigns")
    parser.add_argument(
        "--output-dir", default="results/revision_final/paper_metrics_v1")
    parser.add_argument(
        "--telemetry", action="append", default=[], metavar="NAME=PATH",
        help="repeatable compact telemetry overlay (for example lacam=PATH)")
    parser.add_argument(
        "--primary-execution-horizon", type=int, default=5000,
        help="common synchronous-step horizon for direct comparisons")
    args = parser.parse_args()
    if args.primary_execution_horizon < 1:
        parser.error("--primary-execution-horizon must be positive")
    campaigns = dict(DEFAULT_CAMPAIGNS)
    for value in args.campaign:
        if "=" not in value:
            parser.error("--campaign must be NAME=PATH")
        name, path = value.split("=", 1)
        campaigns[name] = Path(path).resolve()
    telemetry_roots = dict(DEFAULT_TELEMETRY)
    for value in args.telemetry:
        if "=" not in value:
            parser.error("--telemetry must be NAME=PATH")
        name, path = value.split("=", 1)
        telemetry_roots[name] = Path(path).resolve()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    record_counts: dict[str, int] = {}
    for algorithm, campaign in campaigns.items():
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((campaign / "records").glob("*.json"))
        ]
        record_counts[algorithm] = len(records)
        rows.extend(
            record_metrics(
                algorithm, campaign, record, args.primary_execution_horizon,
                overlay_record(telemetry_roots.get(algorithm), record, algorithm),
            )
            for record in records
        )
    write_csv(output / "oneshot_cells.csv", rows)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["map"], row["target"],
                 row["density_percent"], row["capacity_load_percent"], row["agents"])].append(row)
    success_rows: list[dict] = []
    for (algorithm, map_name, target, density, capacity, agents), items in sorted(grouped.items()):
        solved_items = [row for row in items if row["solved"]]
        measured_completion = [
            row["agent_completion_ratio"] for row in items
            if row["agent_completion_ratio"] is not None
        ]
        success_rows.append({
            "algorithm": algorithm, "map": map_name, "target": target,
            "density_percent": density, "capacity_load_percent": capacity,
            "agents": agents, "records": len(items),
            "successes": len(solved_items), "success_rate": len(solved_items) / len(items),
            "early_stopped": sum(row["early_stopped"] for row in items),
            "makespan_median_solved": median(row["makespan"] for row in solved_items),
            "completion_mean_median_solved": median(row["completion_mean"] for row in solved_items),
            "agent_completion_ratio_records": len(measured_completion),
            "agent_completion_ratio_mean": (
                statistics.fmean(measured_completion) if measured_completion else None),
            "agent_completion_ratio_std": (
                statistics.stdev(measured_completion) if len(measured_completion) > 1 else 0.0
                if measured_completion else None),
            "residual_agents_median": median(row["residual_agents"] for row in items),
            "cpu_seconds_median": median(row["cpu_seconds"] for row in items),
            "max_rss_kb_median": median(row["max_rss_kb"] for row in items),
        })
    write_csv(output / "success_and_scale.csv", success_rows)

    lima_extended_rows: list[dict] = []
    for (algorithm, map_name, target, density, capacity, agents), items in sorted(
        grouped.items()
    ):
        if algorithm != "lima":
            continue
        extended_items = [row for row in items if row["extended_solved"]]
        primary_items = [row for row in items if row["solved"]]
        lima_extended_rows.append({
            "map": map_name,
            "target": target,
            "density_percent": density,
            "capacity_load_percent": capacity,
            "agents": agents,
            "records": len(items),
            "primary_horizon": args.primary_execution_horizon,
            "primary_successes": len(primary_items),
            "extended_horizon": 100000,
            "extended_successes": len(extended_items),
            "extended_success_rate": len(extended_items) / len(items),
            "extended_makespan_median": median(
                row["makespan"] for row in extended_items
            ),
        })
    write_csv(output / "lima_extended_scalability.csv", lima_extended_rows)

    by_method_cell = {
        (row["algorithm"], row["map"], row["target"], row["scenario"]): row
        for row in rows
    }
    pair_rows: list[dict] = []
    for baseline in sorted(set(campaigns) - {"lima"}):
        pair_groups: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
        for key, lima_row in by_method_cell.items():
            algorithm, map_name, target, scenario = key
            if algorithm != "lima":
                continue
            other = by_method_cell.get((baseline, map_name, target, scenario))
            if other and lima_row["solved"] and other["solved"]:
                pair_groups[(map_name, target)].append((lima_row, other))
        for (map_name, target), pairs in sorted(pair_groups.items()):
            pair_rows.append({
                "comparison": f"lima_vs_{baseline}", "map": map_name, "target": target,
                "paired_successes": len(pairs),
                "lima_makespan_median": median(left["makespan"] for left, _ in pairs),
                "baseline_makespan_median": median(right["makespan"] for _, right in pairs),
                "lima_completion_mean_median": median(left["completion_mean"] for left, _ in pairs),
                "baseline_completion_mean_median": median(right["completion_mean"] for _, right in pairs),
                "lima_completion_p90_median": median(left["completion_p90"] for left, _ in pairs),
                "lima_completion_p99_median": median(left["completion_p99"] for left, _ in pairs),
            })
    write_csv(output / "paired_common_success.csv", pair_rows, [
        "comparison", "map", "target", "paired_successes",
        "lima_makespan_median", "baseline_makespan_median",
        "lima_completion_mean_median", "baseline_completion_mean_median",
        "lima_completion_p90_median", "lima_completion_p99_median",
    ])

    lima_rows = [row for row in rows if row["algorithm"] == "lima"]
    architecture_fields = [
        "map", "target", "scenario", "agents", "solved", "extended_solved",
        "primary_execution_horizon",
        "completed_agents", "residual_agents", "agent_completion_ratio",
        "comm_events", "comm_events_per_agent_step",
        "comm_distance_min", "comm_distance_mean", "comm_distance_max",
        "comm_distance_variance", "comm_distance_p99", "comm_hops_min",
        "comm_hops_mean", "comm_hops_max", "comm_hops_variance", "comm_hops_p99",
        "execution_over_reference_mean", "execution_over_reference_p90",
        "extra_moves_mean", "recirculation_events", "vertex_conflicts",
        "edge_conflicts", "invalid_moves", "rejoin_failures",
        "goal_preservation_failures", "online_validation_ok", "cpu_seconds", "max_rss_kb",
    ]
    write_csv(output / "lima_architecture_metrics.csv", [
        {key: row.get(key) for key in architecture_fields} for row in lima_rows
    ], architecture_fields)

    common_fields = [
        "algorithm", "map", "target", "scenario", "agents", "solved",
        "completed_agents", "agent_completion_ratio", "makespan", "soc",
        "completion_mean", "completion_p50", "completion_p90", "completion_p99",
        "completion_max", "common_moves_total", "common_waits_total",
        "shortest_path_moves_sum", "shortest_path_extra_moves_sum",
        "shortest_path_movement_stretch", "comm_events", "comm_local_events",
        "comm_global_or_server_events", "comm_payload_units", "comm_scope",
        "comm_joint_participants_mean", "comm_joint_participants_max",
        "comm_agent_agent_direct", "comm_agent_solver_task_upload",
        "comm_agent_solver_state_upload", "comm_solver_agent_route_delivery",
        "comm_solver_agent_action_delivery", "comm_events_per_agent_step",
        "comm_distance_min", "comm_distance_mean", "comm_distance_max",
        "comm_distance_variance", "comm_distance_p99", "comm_hops_min",
        "comm_hops_mean", "comm_hops_max", "comm_hops_variance", "comm_hops_p99",
        "vertex_conflicts", "edge_conflicts", "invalid_moves", "online_validation_ok",
        "cpu_seconds", "max_rss_kb", "trajectory_backfill",
        "trajectory_scalar_equivalence",
    ]
    write_csv(output / "all_methods_common_metrics.csv", [
        {key: row.get(key) for key in common_fields} for row in rows
    ], common_fields)

    all_algorithms = sorted(campaigns)
    all_common = []
    lima_cells = [key[1:] for key in by_method_cell if key[0] == "lima"]
    for map_name, target, scenario in lima_cells:
        selected = [by_method_cell.get((name, map_name, target, scenario)) for name in all_algorithms]
        if all(item and item["solved"] for item in selected):
            all_common.append((map_name, target, scenario))

    lines = [
        "# Paper metric audit", "",
        "## Coverage", "",
        *[f"- {name}: {record_counts[name]} records" for name in sorted(record_counts)],
        "",
        "## Reporting rules", "",
        f"- Direct-comparison success uses a common {args.primary_execution_horizon}-step synchronous execution horizon and every certified logical record, including explicit early-stop records.",
        "- LIMA's 100,000-step extended scalability results are exported separately and never enter direct success, paired-makespan, or statistical-comparison tables.",
        "- Makespan and completion cost use only paired instances solved by both methods; the paired sample count is always printed.",
        "- Agent completion ratio is completed/total at the execution budget and is reported only when the executable method records that count. Early-stopped cells and conflict-bearing CBS/LaCAM search nodes are not imputed as zero.",
        "- An all-method common-success table is meaningful only where every method solved the same instance; "
        f"the current partial data contain {len(all_common)} such instances.",
        "- Completion tails, moves/waits, shortest-path movement stretch, and path validation are loaded from provenance-linked trajectory backfills when present; scalar-only records are never imputed.",
        "- Communication includes both robot-peer traffic and data exchanged with a server/solver/controller. Message events and payload units are separate; an off-map global server is reported as global scope rather than assigned a fabricated grid distance.",
        "- Common communication, detour, path conformance, CPU, and RSS are exported for every method where observed. LIMA-only reference-route mutation, rejoin, recirculation, and local-module telemetry remain architecture-specific.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
