#!/usr/bin/env python3
"""Summarize the fair classic-MAPF LaCAM/PIBT diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def num(value, cast=float, default=math.nan):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((root / "records").glob("*.json"))]
    groups = defaultdict(list)
    cells = []
    for record in records:
        result = record.get("result", {})
        row = {
            "tag": record["tag"], "map": record["map"],
            "density_percent": record["density_percent"], "agents": record["agents"],
            "scenario": record["scenario"], "algorithm": record["algorithm"],
            "solved": int(result.get("solved") == "1"),
            "makespan": num(result.get("makespan"), int),
            "soc": num(result.get("soc"), int),
            "comp_time_ms": num(result.get("comp_time"), int),
            "runner_wall_seconds": record["runner_wall_seconds"],
            "timed_out": int(record.get("timed_out", False)),
        }
        cells.append(row)
        groups[(row["map"], row["density_percent"], row["agents"], row["algorithm"])].append(row)
    with (root / "summary_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader(); writer.writerows(cells)

    summary = []
    for (map_name, density, agents, algorithm), rows in sorted(groups.items()):
        solved = [row for row in rows if row["solved"]]
        summary.append({
            "map": map_name, "density_percent": density, "agents": agents,
            "algorithm": algorithm, "solved": len(solved), "runs": len(rows),
            "success_rate": len(solved) / len(rows),
            "makespan_median": statistics.median(row["makespan"] for row in solved) if solved else math.nan,
            "comp_time_ms_median": statistics.median(row["comp_time_ms"] for row in solved) if solved else math.nan,
        })
    with (root / "summary_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)

    lines = [
        "# Classic MAPF fairness diagnostic", "",
        f"- Scope: {manifest['semantic_scope']}",
        "- This is not the submitted LIMA task: repeated workstation goals and disappear-at-target are intentionally removed.",
        f"- Equal time limit: {manifest['time_limit_seconds']} s; cells: {len(records)}/{manifest['job_count']}",
        "", "| map | density | agents | LaCAM solved | PIBT solved | LaCAM makespan | PIBT makespan |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    indexed = {(row["map"], row["density_percent"], row["algorithm"]): row for row in summary}
    keys = sorted({(row["map"], row["density_percent"], row["agents"]) for row in summary})
    for map_name, density, agents in keys:
        la = indexed[(map_name, density, "lacam")]
        pi = indexed[(map_name, density, "pibt")]
        lams = "-" if math.isnan(la["makespan_median"]) else f"{la['makespan_median']:.0f}"
        pims = "-" if math.isnan(pi["makespan_median"]) else f"{pi['makespan_median']:.0f}"
        lines.append(
            f"| {map_name} | {density}% | {agents} | {la['solved']}/{la['runs']} | "
            f"{pi['solved']}/{pi['runs']} | {lams} | {pims} |"
        )
    la_total = sum(row["solved"] for row in summary if row["algorithm"] == "lacam")
    pi_total = sum(row["solved"] for row in summary if row["algorithm"] == "pibt")
    per_algo = len(records) // 2
    lines += ["", f"Overall: LaCAM {la_total}/{per_algo}; PIBT {pi_total}/{per_algo}.", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
