#!/usr/bin/env python3
"""Summarize fixed-horizon lifelong task-stream experiments.

The measurement window starts after the configured warm-up.  Throughput is
reported together with service-time tails and per-agent fairness so a high
aggregate completion count cannot hide starvation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def jain_fairness(values: list[float]) -> float | None:
    if not values:
        return None
    total = sum(values)
    squared = sum(value * value for value in values)
    return total * total / (len(values) * squared) if squared else 0.0


def telemetry_value(record: dict, *keys: str) -> float | None:
    value = record.get("telemetry", {})
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return number(value) if value is not None else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    horizon = int(manifest.get("horizon_steps", manifest.get("max_steps", 0)))
    warmup = int(manifest.get("warmup_steps", 0))
    if horizon <= warmup:
        parser.error("manifest does not define a valid measurement horizon")
    measurement_steps = horizon - warmup
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "records").glob("*.json"))
    ]
    if not records:
        parser.error("no records found")

    cells: list[dict] = []
    for record in records:
        fields = record.get("result") or record.get("summary") or {}
        tag = record["tag"]
        metrics_value = record.get("metrics")
        metrics_dir = (
            Path(metrics_value) if metrics_value and Path(metrics_value).is_absolute()
            else root.parents[2] / metrics_value if metrics_value
            else root / "metrics" / tag
        )
        if not metrics_dir.is_dir():
            metrics_dir = root / "metrics" / tag

        completion_rows = read_csv(metrics_dir / "task_completions.csv")
        measured = [
            row for row in completion_rows
            if warmup < int(number(row.get("t"))) <= horizon
        ]
        service = [number(row.get("service_steps")) for row in measured]
        per_agent = Counter(int(number(row.get("agent"))) for row in measured)
        agents = int(record.get("agents", 0))
        agent_counts = [float(per_agent.get(index, 0)) for index in range(agents)]

        # Communication count is filtered to the same post-warm-up window.
        comm_rows = read_csv(metrics_dir / "comm_steps.csv")
        comm_events = sum(
            int(number(row.get(key)))
            for row in comm_rows
            if warmup < int(number(row.get("t"))) <= horizon
            for key in ("acquisitions", "broadcasts", "gate_signals")
        )
        solver_rows = [
            row for row in read_csv(metrics_dir / "solver_invocations.csv")
            if warmup < int(number(row.get("t"))) <= horizon
        ]
        solver_us = [number(row.get("wall_us")) for row in solver_rows]
        fallbacks = sum(int(number(row.get("fallback"))) for row in solver_rows)
        resource = record.get("resource", {})
        tasks = len(measured)
        completed_total_text = str(fields.get("completed", "0/0"))
        completed_total = int(number(completed_total_text.split("/", 1)[0]))
        variant = record.get("variant", record.get("algorithm", "unknown"))
        hop_p99 = telemetry_value(
            record, "communication", "intersection_hops", "p99")
        hop_max = telemetry_value(
            record, "communication", "intersection_hops", "max")
        distance_p99 = telemetry_value(
            record, "communication", "distance_cells", "p99")
        cells.append({
            "tag": tag,
            "algorithm": record.get("algorithm", "lima"),
            "variant": variant,
            "map": record["map"],
            "density_percent": record.get("density", record.get("density_percent")),
            "agents": agents,
            "scenario": record["scenario"],
            "horizon_completed": int(bool(record.get("horizon_completed", True))),
            "warmup_steps": warmup,
            "measurement_steps": measurement_steps,
            "tasks_total": completed_total,
            "tasks_measured": tasks,
            "tasks_per_step": tasks / measurement_steps,
            "tasks_per_agent_step": tasks / (agents * measurement_steps) if agents else 0.0,
            "service_mean": statistics.fmean(service) if service else None,
            "service_p50": percentile(service, 0.50),
            "service_p90": percentile(service, 0.90),
            "service_p99": percentile(service, 0.99),
            "service_max": max(service) if service else None,
            "zero_completion_agent_fraction": (
                sum(value == 0 for value in agent_counts) / agents if agents else None),
            "jain_task_fairness": jain_fairness(agent_counts),
            "comm_events": comm_events,
            "comm_events_per_agent_step": (
                comm_events / (agents * measurement_steps) if agents else 0.0),
            "comm_events_per_task": comm_events / tasks if tasks else None,
            "comm_hops_p99": hop_p99,
            "comm_hops_max": hop_max,
            "comm_distance_cells_p99": distance_p99,
            "solver_invocations": len(solver_rows),
            "solver_wall_us_p99": percentile(solver_us, 0.99),
            "solver_fallback_rate": fallbacks / len(solver_rows) if solver_rows else None,
            "cpu_seconds": number(resource.get("user_seconds")) + number(resource.get("system_seconds")),
            "max_rss_kb": number(resource.get("max_rss_kb")),
        })
    write_csv(root / "summary_lifelong_cells.csv", cells)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in cells:
        groups[(row["algorithm"], row["variant"], row["map"],
                row["density_percent"], row["agents"])].append(row)
    summary_rows: list[dict] = []
    metric_names = (
        "tasks_per_step", "tasks_per_agent_step", "service_p50", "service_p90",
        "service_p99", "service_max", "zero_completion_agent_fraction",
        "jain_task_fairness", "comm_events_per_agent_step", "comm_events_per_task",
        "comm_hops_p99", "comm_hops_max", "comm_distance_cells_p99",
        "solver_wall_us_p99", "solver_fallback_rate", "cpu_seconds", "max_rss_kb",
    )
    for (algorithm, variant, map_name, density, agents), rows in sorted(groups.items()):
        summary = {
            "algorithm": algorithm, "variant": variant, "map": map_name,
            "density_percent": density, "agents": agents, "runs": len(rows),
            "valid_horizons": sum(row["horizon_completed"] for row in rows),
            "tasks_measured_median": statistics.median(row["tasks_measured"] for row in rows),
        }
        for name in metric_names:
            values = [float(row[name]) for row in rows if row[name] is not None]
            summary[f"{name}_median"] = median_or_none(values)
        summary_rows.append(summary)
    write_csv(root / "summary_lifelong_groups.csv", summary_rows)

    lines = [
        "# Lifelong fixed-horizon report", "",
        f"- Horizon: {horizon} steps; warm-up: {warmup}; measurement: {measurement_steps} steps.",
        f"- Records: {len(cells)}/{manifest.get('job_count', len(cells))}.",
        "- Every robot always has one active task, so queue backlog is constant by definition; service-time tails and per-agent Jain fairness expose starvation instead.",
        "- Communication counts use only post-warm-up steps. Distance and hop summaries use the run-level telemetry distribution.", "",
        "| method | map | density | agents | runs | tasks/step | service p90 | service p99 | zero-task agents | Jain fairness | comm/(agent step) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def shown(key: str, digits: int = 3) -> str:
            value = row.get(key)
            return "n/a" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {row['algorithm']}:{row['variant']} | {row['map']} | "
            f"{row['density_percent']}% | {row['agents']} | {row['runs']} | "
            f"{shown('tasks_per_step_median')} | {shown('service_p90_median', 1)} | "
            f"{shown('service_p99_median', 1)} | "
            f"{shown('zero_completion_agent_fraction_median')} | "
            f"{shown('jain_task_fairness_median')} | "
            f"{shown('comm_events_per_agent_step_median')} |"
        )
    lines.append("")
    (root / "REPORT_LIFELONG.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "REPORT_LIFELONG.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
