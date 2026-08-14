#!/usr/bin/env python3
"""Summarize CBS/PRIMAL2 on the repeated-sink disappear task."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def number(value, cast=float, default=math.nan):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    roots = [Path(value).resolve() for value in args.directories]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
                 for root in roots]
    records = []
    for root in roots:
        records.extend(json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted((root / "records").glob("*.json")))

    cells = []
    groups = defaultdict(list)
    for record in records:
        result = record.get("result", {})
        elapsed = result.get("elapsed_s", result.get("comp_time", math.nan))
        row = {
            "tag": record["tag"], "map": record["map"],
            "density_percent": record["density_percent"], "agents": record["agents"],
            "scenario": record["scenario"], "algorithm": record["algorithm"],
            "solved": int(result.get("solved") == "1"),
            "completed": result.get("completed", ""),
            "makespan": number(result.get("makespan"), int),
            "elapsed_seconds": number(elapsed),
            "runner_wall_seconds": record["runner_wall_seconds"],
            "timed_out": int(record.get("timed_out", False)),
        }
        cells.append(row)
        groups[(row["map"], row["density_percent"], row["agents"], row["algorithm"])].append(row)

    if not cells:
        parser.error("no records found")
    with (output / "summary_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(sorted(cells, key=lambda row: (
            row["map"], row["density_percent"], row["algorithm"], row["scenario"])))

    summary = []
    for (map_name, density, agents, algorithm), rows in sorted(groups.items()):
        solved = [row for row in rows if row["solved"]]
        finite_times = [row["elapsed_seconds"] for row in solved
                        if not math.isnan(row["elapsed_seconds"])]
        summary.append({
            "map": map_name, "density_percent": density, "agents": agents,
            "algorithm": algorithm, "solved": len(solved), "runs": len(rows),
            "success_rate": len(solved) / len(rows),
            "makespan_median": statistics.median(row["makespan"] for row in solved)
            if solved else math.nan,
            "elapsed_seconds_median": statistics.median(finite_times) if finite_times else math.nan,
            "timeouts": sum(row["timed_out"] for row in rows),
        })
    with (output / "summary_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    algorithms = sorted({row["algorithm"] for row in summary})
    lines = [
        "# Repeated-sink disappear baseline report", "",
        "- Scope: exact submitted maps/scenarios; repeated sink goals; agents disappear at target.",
        f"- Records: {len(records)}; source directories: {', '.join(str(root) for root in roots)}.",
        f"- Algorithms: {', '.join(algorithms)}.", "",
        "| map | density | agents | algorithm | solved | median makespan | median solved time (s) | timeouts |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        makespan = "-" if math.isnan(row["makespan_median"]) else f"{row['makespan_median']:.0f}"
        elapsed = "-" if math.isnan(row["elapsed_seconds_median"]) else f"{row['elapsed_seconds_median']:.2f}"
        lines.append(
            f"| {row['map']} | {row['density_percent']}% | {row['agents']} | {row['algorithm']} | "
            f"{row['solved']}/{row['runs']} | {makespan} | {elapsed} | {row['timeouts']} |"
        )
    lines.append("")
    for algorithm in algorithms:
        solved = sum(row["solved"] for row in summary if row["algorithm"] == algorithm)
        runs = sum(row["runs"] for row in summary if row["algorithm"] == algorithm)
        lines.append(f"- Overall {algorithm}: {solved}/{runs}.")
    lines += ["", "## Run configuration", "", "```json",
              json.dumps(manifests, indent=2, sort_keys=True), "```", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
