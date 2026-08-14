#!/usr/bin/env python3
"""Summarize fixed-horizon lifelong task-stream experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((root / "records").glob("*.json"))]
    if not records:
        parser.error("no records found")

    cells = []
    for record in records:
        summary = record.get("summary", {})
        tasks = int(summary.get("completed", "0/0").split("/", 1)[0])
        steps = int(summary.get("steps", 0))
        comm = read_csv(root / "metrics" / record["tag"] / "comm_steps.csv")
        comm_events = sum(sum(int(row[name]) for name in
                              ("acquisitions", "broadcasts", "gate_signals")) for row in comm)
        cells.append({
            "tag": record["tag"], "map": record["map"],
            "density_percent": record["density_percent"], "agents": record["agents"],
            "scenario": record["scenario"], "steps": steps, "tasks": tasks,
            "tasks_per_step": tasks / steps if steps else 0.0,
            "tasks_per_agent": tasks / record["agents"] if record["agents"] else 0.0,
            "comm_events_per_task": comm_events / tasks if tasks else 0.0,
            "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
        })
    with (root / "summary_lifelong_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    groups = defaultdict(list)
    for row in cells:
        groups[(row["map"], row["density_percent"], row["agents"])].append(row)
    summary_rows = []
    for (map_name, density, agents), rows in sorted(groups.items()):
        summary_rows.append({
            "map": map_name, "density_percent": density, "agents": agents,
            "runs": len(rows),
            "tasks_median": statistics.median(row["tasks"] for row in rows),
            "tasks_per_step_median": statistics.median(row["tasks_per_step"] for row in rows),
            "tasks_per_agent_median": statistics.median(row["tasks_per_agent"] for row in rows),
            "comm_events_per_task_median": statistics.median(row["comm_events_per_task"] for row in rows),
            "elapsed_seconds_median": statistics.median(row["elapsed_seconds"] for row in rows),
        })
    with (root / "summary_lifelong_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    lines = [
        "# Lifelong fixed-horizon report", "",
        f"- Horizon: {manifest['max_steps']} steps; cells: {len(cells)}/{manifest['job_count']}.",
        "- `completed` is interpreted as tasks served, not agents removed.", "",
        "| map | density | agents | runs | median tasks | tasks/step | tasks/agent | comm events/task | runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['map']} | {row['density_percent']}% | {row['agents']} | {row['runs']} | "
            f"{row['tasks_median']:.0f} | {row['tasks_per_step_median']:.3f} | "
            f"{row['tasks_per_agent_median']:.2f} | {row['comm_events_per_task_median']:.2f} | "
            f"{row['elapsed_seconds_median']:.2f} |"
        )
    lines.append("")
    (root / "REPORT_LIFELONG.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "REPORT_LIFELONG.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
