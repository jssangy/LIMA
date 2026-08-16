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


def record_metrics(
    algorithm: str, campaign: Path, record: dict, primary_horizon: int
) -> dict:
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

    resource = record.get("resource") or {}
    telemetry = record.get("telemetry") or {}
    steps = numeric(result.get("steps")) or span
    event_count = numeric(nested(telemetry, "communication", "event_count"))
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
        "comm_events": event_count,
        "comm_events_per_agent_step": (
            event_count / (agents * steps) if event_count is not None and agents and steps else None),
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
        "recirculation_events": numeric(nested(telemetry, "detour", "recirculation_loop_cells", "count")),
        "vertex_conflicts": numeric(nested(telemetry, "path_conformity", "vertex_conflicts")),
        "edge_conflicts": numeric(nested(telemetry, "path_conformity", "edge_conflicts")),
        "invalid_moves": numeric(nested(telemetry, "path_conformity", "invalid_moves")),
        "rejoin_failures": numeric(nested(telemetry, "path_conformity", "rejoin_failures")),
        "goal_preservation_failures": numeric(nested(telemetry, "path_conformity", "goal_preservation_failures")),
        "online_validation_ok": nested(telemetry, "path_conformity", "online_validation_ok"),
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
                algorithm, campaign, record, args.primary_execution_horizon
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
        "- LIMA completion p50/p90/p99 comes from per-agent completion traces. CBS, LaCAM, and PIBT expose SOC, so their mean completion time is available; their current scalar outputs do not expose completion tails. PRIMAL2 currently exposes makespan only.",
        "- Communication locality, count, detour, recirculation, path conformance, CPU, and RSS are exported separately and are not mixed into the primary success claim.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
