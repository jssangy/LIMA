#!/usr/bin/env python3
"""Aggregate ``run_revision_grid.py`` records and optional metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def number(value, cast=float, default=0):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def collect_metrics(directory: Path) -> dict:
    result: dict[str, float | int | str] = {}
    agents = read_csv(directory / "agents.csv")
    if agents:
        completed = [row for row in agents if number(row.get("completed"), int) == 1]
        completion = [number(row.get("completion_step"), int) for row in completed]
        extra = [number(row.get("extra_moves"), int) for row in agents]
        discharges = [number(row.get("discharges"), int) for row in agents]
        result.update(
            mean_completion_step=statistics.fmean(completion) if completion else math.nan,
            completion_p50=percentile(completion, 0.50),
            completion_p90=percentile(completion, 0.90),
            completion_p99=percentile(completion, 0.99),
            extra_moves_total=sum(extra),
            extra_moves_per_agent=statistics.fmean(extra) if extra else math.nan,
            discharges_total=sum(discharges),
        )

    comm = read_csv(directory / "comm_steps.csv")
    if comm:
        acquisitions = sum(number(row.get("acquisitions"), int) for row in comm)
        broadcasts = sum(number(row.get("broadcasts"), int) for row in comm)
        gate_signals = sum(number(row.get("gate_signals"), int) for row in comm)
        result.update(
            metric_steps=len(comm),
            acquisitions=acquisitions,
            broadcasts=broadcasts,
            gate_signals=gate_signals,
            comm_events_total=acquisitions + broadcasts + gate_signals,
        )

    solver = read_csv(directory / "solver_invocations.csv")
    if solver:
        wall_us = [number(row.get("wall_us"), int) for row in solver]
        expanded = [number(row.get("expanded"), int) for row in solver]
        outcomes = Counter(row.get("outcome", "") for row in solver)
        fallbacks = sum(number(row.get("fallback"), int) for row in solver)
        result.update(
            solver_calls=len(solver),
            solver_accepted=sum(number(row.get("accepted"), int) for row in solver),
            solver_failures=sum(count for name, count in outcomes.items() if name != "solved"),
            solver_fallbacks=fallbacks,
            solver_fallback_rate=fallbacks / len(solver),
            solver_wall_us_total=sum(wall_us),
            solver_wall_us_p50=percentile(wall_us, 0.50),
            solver_wall_us_p90=percentile(wall_us, 0.90),
            solver_wall_us_p99=percentile(wall_us, 0.99),
            solver_wall_us_max=max(wall_us),
            solver_expanded_p99=percentile(expanded, 0.99),
            solver_expanded_max=max(expanded),
        )
    return result


def fmt(value, digits=1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    records = root / "records"
    if not records.is_dir():
        parser.error(f"records directory does not exist: {records}")

    cells = []
    for path in sorted(records.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        summary = record.get("summary", {})
        completed_text = summary.get("completed", "0/0")
        try:
            done, total = (int(value) for value in completed_text.split("/", 1))
        except ValueError:
            done, total = 0, record.get("agents", 0)
        row = {
            "tag": record["tag"],
            "map": record["map"],
            "density_percent": record["density_percent"],
            "agents": record["agents"],
            "scenario": record["scenario"],
            "variant": record["variant"],
            "status": "wall_timeout" if record.get("timed_out") else summary.get("status", "error"),
            "completed": done,
            "total": total,
            "success": int(summary.get("status") == "completed" and done == total),
            "steps": number(summary.get("steps"), int),
            "moves": number(summary.get("moves"), int),
            "waits": number(summary.get("waits"), int),
            "deadlocks": number(summary.get("deadlocks"), int),
            "elapsed_seconds": number(summary.get("elapsed_seconds"), float, math.nan),
            "runner_wall_seconds": record.get("runner_wall_seconds", math.nan),
            "validation": summary.get("validation", "not_recorded"),
        }
        row.update(collect_metrics(root / "metrics" / record["tag"]))
        if row["agents"]:
            row["comm_events_per_agent"] = row.get("comm_events_total", math.nan) / row["agents"]
        cells.append(row)

    preferred = [
        "tag", "map", "density_percent", "agents", "scenario", "variant", "status",
        "completed", "total", "success", "steps", "moves", "waits", "deadlocks",
        "elapsed_seconds", "runner_wall_seconds", "validation", "mean_completion_step",
        "completion_p50", "completion_p90", "completion_p99", "extra_moves_total",
        "extra_moves_per_agent", "discharges_total", "acquisitions", "broadcasts",
        "gate_signals", "comm_events_total", "comm_events_per_agent", "solver_calls",
        "solver_accepted", "solver_failures", "solver_wall_us_total", "solver_wall_us_p50",
        "solver_fallbacks", "solver_fallback_rate",
        "solver_wall_us_p90", "solver_wall_us_p99", "solver_wall_us_max",
        "solver_expanded_p99", "solver_expanded_max",
    ]
    fields = [name for name in preferred if any(name in row for row in cells)]
    with (root / "summary_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)

    groups = defaultdict(list)
    for row in cells:
        groups[(row["map"], row["density_percent"], row["agents"], row["variant"])].append(row)
    group_rows = []
    for (map_name, density, agents, variant), rows in sorted(groups.items()):
        successful = [row for row in rows if row["success"]]
        def vals(name):
            return [row[name] for row in successful if name in row and not math.isnan(row[name])]
        group_rows.append({
            "map": map_name,
            "density_percent": density,
            "agents": agents,
            "variant": variant,
            "runs": len(rows),
            "successes": len(successful),
            "success_rate": len(successful) / len(rows),
            "makespan_median": percentile(vals("steps"), 0.50),
            "makespan_p90": percentile(vals("steps"), 0.90),
            "elapsed_median": percentile(vals("elapsed_seconds"), 0.50),
            "mean_completion_median": percentile(vals("mean_completion_step"), 0.50),
            "extra_moves_per_agent_median": percentile(vals("extra_moves_per_agent"), 0.50),
            "comm_events_per_agent_median": percentile(vals("comm_events_per_agent"), 0.50),
            "solver_wall_us_p99_median": percentile(vals("solver_wall_us_p99"), 0.50),
            "solver_fallback_rate_median": percentile(vals("solver_fallback_rate"), 0.50),
        })
    group_fields = list(group_rows[0]) if group_rows else []
    with (root / "summary_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(group_rows)

    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    lines = [
        f"# Revision grid report - {manifest['variant']}", "",
        f"- Git HEAD: `{manifest['git_head']}`",
        f"- Binary SHA-256: `{manifest['binary_sha256']}`",
        f"- Cells present: {len(cells)} / {manifest['job_count']}",
        f"- Metrics: {'yes' if manifest['metrics'] else 'no'}; traces: {'yes' if manifest['record_trace'] else 'no'}",
        "", "| map | density | agents | success | makespan median / p90 | runtime median | mean completion | extra moves / agent | comm events / agent | solver p99 us | fallback rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['map']} | {row['density_percent']}% | {row['agents']} | "
            f"{row['successes']}/{row['runs']} | {fmt(row['makespan_median'], 0)} / {fmt(row['makespan_p90'], 0)} | "
            f"{fmt(row['elapsed_median'], 2)} s | {fmt(row['mean_completion_median'])} | "
            f"{fmt(row['extra_moves_per_agent_median'], 2)} | {fmt(row['comm_events_per_agent_median'], 2)} | "
            f"{fmt(row['solver_wall_us_p99_median'], 0)} | {fmt(row['solver_fallback_rate_median'], 4)} |"
        )
    lines += ["", "`summary_cells.csv` contains the auditable per-cell data.", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
