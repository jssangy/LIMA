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
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    roots = [Path(value).resolve() for value in args.directories]
    root = Path(args.output_dir).resolve() if args.output_dir else roots[0]
    root.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
                 for source in roots]
    budgets = {manifest["time_limit_seconds"] for manifest in manifests}
    if len(budgets) != 1:
        parser.error(f"time limits are not equal: {sorted(budgets)}")
    records = []
    for source in roots:
        records.extend(json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted((source / "records").glob("*.json")))
    groups = defaultdict(list)
    cells = []
    for record in records:
        result = record.get("result", {})
        resource = record.get("resource", {})
        row = {
            "tag": record["tag"], "map": record["map"],
            "density_percent": record["density_percent"], "agents": record["agents"],
            "scenario": record["scenario"], "algorithm": record["algorithm"],
            "solved": int(result.get("solved") == "1"),
            "makespan": num(result.get("makespan"), int),
            "soc": num(result.get("soc"), int),
            "comp_time_ms": num(result.get("comp_time"), int),
            "runner_wall_seconds": record["runner_wall_seconds"],
            "max_rss_mb": num(resource.get("max_rss_kb"), float) / 1024.0,
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
            "max_rss_mb_median": statistics.median(row["max_rss_mb"] for row in rows),
            "max_rss_mb_max": max(row["max_rss_mb"] for row in rows),
        })
    with (root / "summary_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)

    lines = [
        "# Classic MAPF fairness diagnostic", "",
        f"- Scope: {manifests[0]['semantic_scope']}",
        "- This is not the submitted LIMA task: repeated workstation goals and disappear-at-target are intentionally removed.",
        f"- Equal time limit: {manifests[0]['time_limit_seconds']} s; cells: "
        f"{len(records)}/{sum(manifest['job_count'] for manifest in manifests)}",
        f"- Source directories: {', '.join(str(source) for source in roots)}",
        "", "| map | density | agents | LaCAM solved | PIBT solved | LaCAM makespan | PIBT makespan | LaCAM peak MiB | PIBT peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{pi['solved']}/{pi['runs']} | {lams} | {pims} | "
            f"{la['max_rss_mb_median']:.0f} | {pi['max_rss_mb_median']:.0f} |"
        )
    la_total = sum(row["solved"] for row in summary if row["algorithm"] == "lacam")
    pi_total = sum(row["solved"] for row in summary if row["algorithm"] == "pibt")
    la_runs = sum(row["runs"] for row in summary if row["algorithm"] == "lacam")
    pi_runs = sum(row["runs"] for row in summary if row["algorithm"] == "pibt")
    lines += ["", f"Overall: LaCAM {la_total}/{la_runs}; PIBT {pi_total}/{pi_runs}.", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
