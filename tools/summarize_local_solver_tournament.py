#!/usr/bin/env python3
"""Aggregate Gate A records into density-frontier and resource tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_POPULATIONS = {"warehouse4": 14, "warehouse3": 12, "cross4": 15}
ALIASES = {"beam": "beam2048"}


def percentile(values: list[float], p: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    return finite[min(len(finite) - 1, int((len(finite) - 1) * p))]


def fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def frontier(cells: dict[int, dict], threshold: float) -> int:
    result = 0
    for population in range(1, max(cells, default=0) + 1):
        cell = cells.get(population)
        if cell is None or cell["solve_rate"] < threshold:
            break
        result = population
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+")
    parser.add_argument("--output-dir", default="results/phase2_local_solver_summary")
    args = parser.parse_args()

    records: dict[tuple[str, str, int], dict] = {}
    manifests = []
    for raw_directory in args.input_dirs:
        directory = (ROOT / raw_directory).resolve()
        manifest = directory / "MANIFEST.json"
        if manifest.is_file():
            manifests.append(json.loads(manifest.read_text(encoding="utf-8")))
        for path in sorted((directory / "records").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            solver = ALIASES.get(payload["solver"], payload["solver"])
            summary = payload.get("summary", {})
            resource = payload.get("resource", {})
            rows = int(summary.get("rows", 0))
            try:
                user_seconds = float(resource.get("user_seconds", "nan"))
            except (TypeError, ValueError):
                user_seconds = math.nan
            try:
                max_rss = int(resource.get("max_rss_kb", 0))
            except (TypeError, ValueError):
                max_rss = 0
            records[(solver, payload["shape"], int(payload["population"]))] = {
                "solver": solver,
                "shape": payload["shape"],
                "population": int(payload["population"]),
                "bound": int(payload["bound"]),
                "rows": rows,
                "solved": int(summary.get("solved", 0)),
                "solve_rate": float(summary.get("solve_rate", 0.0)),
                "expanded_median": summary.get("expanded_median"),
                "expanded_p90": summary.get("expanded_p90"),
                "solution_len_median": summary.get("solution_len_median"),
                "user_cpu_us_per_instance": user_seconds * 1e6 / rows if rows else math.nan,
                "max_rss_kb": max_rss,
                "timed_out": bool(payload.get("timed_out", False)),
                "returncode": int(payload.get("returncode", 0)),
            }

    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cell_fields = [
        "solver", "shape", "population", "bound", "rows", "solved", "solve_rate",
        "expanded_median", "expanded_p90", "solution_len_median",
        "user_cpu_us_per_instance", "max_rss_kb", "timed_out", "returncode",
    ]
    with (output / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cell_fields)
        writer.writeheader()
        writer.writerows(sorted(records.values(), key=lambda row: (
            row["solver"], row["shape"], row["population"])))

    grouped: dict[str, list[dict]] = defaultdict(list)
    shaped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for row in records.values():
        grouped[row["solver"]].append(row)
        shaped[(row["solver"], row["shape"])][row["population"]] = row

    summaries = []
    expected_cells = sum(EXPECTED_POPULATIONS.values())
    for solver, rows in grouped.items():
        requested = sum(row["rows"] for row in rows)
        solved = sum(row["solved"] for row in rows)
        cpu = [row["user_cpu_us_per_instance"] for row in rows]
        nodes = [float(row["expanded_median"]) for row in rows
                 if row["expanded_median"] is not None]
        item = {
            "solver": solver,
            "cells": len(rows),
            "expected_cells": expected_cells,
            "instances": requested,
            "solve_rate": solved / requested if requested else 0.0,
            "frontier100_warehouse4": frontier(shaped[(solver, "warehouse4")], 1.0),
            "frontier100_warehouse3": frontier(shaped[(solver, "warehouse3")], 1.0),
            "frontier100_cross4": frontier(shaped[(solver, "cross4")], 1.0),
            "frontier99_warehouse4": frontier(shaped[(solver, "warehouse4")], 0.99),
            "frontier99_warehouse3": frontier(shaped[(solver, "warehouse3")], 0.99),
            "frontier99_cross4": frontier(shaped[(solver, "cross4")], 0.99),
            "cpu_us_cell_median": percentile(cpu, 0.50),
            "cpu_us_cell_p90": percentile(cpu, 0.90),
            "expanded_cell_median": percentile(nodes, 0.50),
            "expanded_cell_p90": percentile(nodes, 0.90),
            "max_rss_kb": max((row["max_rss_kb"] for row in rows), default=0),
            "watchdogs": sum(row["timed_out"] for row in rows),
            "bad_returns": sum(row["returncode"] != 0 for row in rows),
        }
        item["frontier100_sum"] = sum(item[key] for key in (
            "frontier100_warehouse4", "frontier100_warehouse3", "frontier100_cross4"))
        summaries.append(item)
    summaries.sort(key=lambda row: (
        -row["frontier100_sum"], -row["solve_rate"], row["cpu_us_cell_median"] or math.inf,
        row["solver"]))

    summary_fields = list(summaries[0]) if summaries else ["solver"]
    with (output / "solvers.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    complete = bool(summaries) and all(row["cells"] == expected_cells for row in summaries)
    lines = [
        "# Gate A single-intersection solver tournament",
        "",
        f"Status: **{'complete' if complete else 'provisional / partial'}**. "
        f"Loaded {len(records)} cells from {len(args.input_dirs)} result directories. "
        "Frontier = largest contiguous N from 1 with the stated solve-rate threshold.",
        "",
        "| solver | cells | solve % | 100% frontier W4/W3/C4 | 99% frontier W4/W3/C4 | CPU med/p90 us | nodes med/p90 | max RSS MiB | watchdog |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['solver']} | {row['cells']}/{expected_cells} | {row['solve_rate'] * 100:.2f} | "
            f"{row['frontier100_warehouse4']}/{row['frontier100_warehouse3']}/{row['frontier100_cross4']} | "
            f"{row['frontier99_warehouse4']}/{row['frontier99_warehouse3']}/{row['frontier99_cross4']} | "
            f"{fmt(row['cpu_us_cell_median'])}/{fmt(row['cpu_us_cell_p90'])} | "
            f"{fmt(row['expanded_cell_median'], 0)}/{fmt(row['expanded_cell_p90'], 0)} | "
            f"{row['max_rss_kb'] / 1024:.1f} | {row['watchdogs']} |"
        )
    lines.extend([
        "",
        "Notes:",
        "- W4 = 2/10/2/10 (B=14), W3 = 10/2/10 (B=12), C4 = 5/5/5/5 (B=15).",
        "- CPU is aggregate `/usr/bin/time` user CPU divided by rows; wall time is not a score.",
        "- Best-first variants cap retained nodes as well as expansions to bound memory.",
        "- A partial report must not be used to freeze Gate A.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "SOURCE_MANIFESTS.json").write_text(
        json.dumps(manifests, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
