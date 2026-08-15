#!/usr/bin/env python3
"""Aggregate a multi-variant Phase 2 Gate C/D step-budget tournament."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def number(value, cast=int, default=0):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def fmt(value: float | int, digits: int = 2) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def compare(lhs: dict, rhs: dict) -> int:
    """Return +1 when lhs wins, -1 when rhs wins, 0 for an exact rank tie."""
    lhs_key = (
        lhs["success"],
        lhs["completion_fraction"],
        -lhs["steps"] if lhs["success"] else 0,
        -lhs["moves"],
        -lhs["waits"],
    )
    rhs_key = (
        rhs["success"],
        rhs["completion_fraction"],
        -rhs["steps"] if rhs["success"] else 0,
        -rhs["moves"],
        -rhs["waits"],
    )
    return (lhs_key > rhs_key) - (lhs_key < rhs_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--base", default="gatec_base")
    parser.add_argument(
        "--variants",
        default="",
        help="optional comma-separated variant allow-list",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    selected_variants = {
        value.strip() for value in args.variants.split(",") if value.strip()
    }

    roots = [Path(value).resolve() for value in args.roots]
    if len(roots) > 1 and not args.output_dir:
        parser.error("--output-dir is required when combining multiple roots")
    output = Path(args.output_dir).resolve() if args.output_dir else roots[0] / "summary"
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    manifests: dict[str, dict] = {}
    expected_cells: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    for root in roots:
        for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            manifest_path = variant_dir / "MANIFEST.json"
            records_dir = variant_dir / "records"
            if not manifest_path.is_file() or not records_dir.is_dir():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            variant = manifest["variant"]
            if selected_variants and variant not in selected_variants:
                continue
            if variant in manifests:
                prior = manifests[variant]
                comparable = ("binary_sha256", "max_steps", "variant_flags")
                if any(prior.get(key) != manifest.get(key) for key in comparable):
                    parser.error(f"inconsistent executable/config manifests for {variant}")
            else:
                manifests[variant] = manifest
            expected_cells[variant] += int(manifest["job_count"])
            horizon = int(manifest["max_steps"])
            for record_path in sorted(records_dir.glob("*.json")):
                record = json.loads(record_path.read_text(encoding="utf-8"))
                key = (variant, record["tag"])
                if key in seen:
                    parser.error(f"duplicate variant/cell record: {variant} {record['tag']}")
                seen.add(key)
                summary = record.get("summary", {})
                completed_text = summary.get("completed", f"0/{record['agents']}")
                try:
                    completed, total = (int(value) for value in completed_text.split("/", 1))
                except ValueError:
                    completed, total = 0, int(record["agents"])
                raw_status = summary.get("status", "error")
                steps = number(summary.get("steps"))
                if record.get("timed_out"):
                    status = "watchdog"
                elif raw_status == "step_limit" and steps < horizon:
                    status = "global_stall"
                elif raw_status == "step_limit":
                    status = "horizon"
                else:
                    status = raw_status
                success = int(status == "completed" and completed == total)
                rows.append({
                    "cell": record["tag"],
                    "map": record["map"],
                    "density_percent": int(record["density_percent"]),
                    "agents": int(record["agents"]),
                    "scenario": int(record["scenario"]),
                    "variant": variant,
                    "status": status,
                    "raw_status": raw_status,
                    "success": success,
                    "completed": completed,
                    "total": total,
                    "residual": total - completed,
                    "completion_fraction": completed / total if total else 0.0,
                    "steps": steps,
                    "horizon": horizon,
                    "moves": number(summary.get("moves")),
                    "waits": number(summary.get("waits")),
                    "deadlocks": number(summary.get("deadlocks")),
                    "runner_wall_seconds": float(record.get("runner_wall_seconds", math.nan)),
                })

    if not rows:
        parser.error(f"no tournament records under {root}")

    cell_fields = list(rows[0])
    with (output / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cell_fields)
        writer.writeheader()
        writer.writerows(rows)

    by_variant: dict[str, list[dict]] = defaultdict(list)
    by_variant_cell: dict[tuple[str, str], dict] = {}
    for row in rows:
        by_variant[row["variant"]].append(row)
        by_variant_cell[(row["variant"], row["cell"])] = row

    base_cells = {row["cell"]: row for row in by_variant.get(args.base, [])}
    variants: list[dict] = []
    for variant, variant_rows in by_variant.items():
        successful = [row for row in variant_rows if row["success"]]
        wins = losses = ties = 0
        for row in variant_rows:
            baseline = base_cells.get(row["cell"])
            if baseline is None or variant == args.base:
                continue
            result = compare(row, baseline)
            wins += result > 0
            losses += result < 0
            ties += result == 0
        manifest = manifests[variant]
        variants.append({
            "variant": variant,
            "cells_present": len(variant_rows),
            "cells_expected": expected_cells[variant],
            "watchdogs": sum(row["status"] == "watchdog" for row in variant_rows),
            "horizon_cells": sum(row["status"] == "horizon" for row in variant_rows),
            "global_stall_cells": sum(row["status"] == "global_stall" for row in variant_rows),
            "completed_cells": len(successful),
            "completed_agents": sum(row["completed"] for row in variant_rows),
            "total_agents": sum(row["total"] for row in variant_rows),
            "residual_agents": sum(row["residual"] for row in variant_rows),
            "agent_completion_fraction": (
                sum(row["completed"] for row in variant_rows)
                / sum(row["total"] for row in variant_rows)
            ),
            "makespan_median": median([row["steps"] for row in successful]),
            "moves_median_completed": median([row["moves"] for row in successful]),
            "waits_median_completed": median([row["waits"] for row in successful]),
            "deadlocks_total": sum(row["deadlocks"] for row in variant_rows),
            "paired_wins_vs_base": wins,
            "paired_losses_vs_base": losses,
            "paired_ties_vs_base": ties,
        })

    variants.sort(key=lambda row: (
        row["watchdogs"],
        -row["completed_cells"],
        row["residual_agents"],
        row["makespan_median"],
        row["moves_median_completed"],
    ))
    variant_fields = list(variants[0])
    with (output / "variants.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=variant_fields)
        writer.writeheader()
        writer.writerows(variants)

    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["map"], row["density_percent"])].append(row)
    group_rows: list[dict] = []
    for (variant, map_name, density), group in sorted(groups.items()):
        group_rows.append({
            "variant": variant,
            "map": map_name,
            "density_percent": density,
            "runs": len(group),
            "completed_cells": sum(row["success"] for row in group),
            "horizon_cells": sum(row["status"] == "horizon" for row in group),
            "global_stall_cells": sum(row["status"] == "global_stall" for row in group),
            "completed_agents": sum(row["completed"] for row in group),
            "total_agents": sum(row["total"] for row in group),
            "residual_agents": sum(row["residual"] for row in group),
            "agent_completion_fraction": (
                sum(row["completed"] for row in group) / sum(row["total"] for row in group)
            ),
            "makespan_median": median([row["steps"] for row in group if row["success"]]),
        })
    group_fields = list(group_rows[0])
    with (output / "map_density.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(group_rows)

    complete = all(row["cells_present"] == row["cells_expected"] for row in variants)
    lines = [
        "# Phase 2 gate tournament",
        "",
        "- Roots: " + ", ".join(f"`{root}`" for root in roots),
        f"- Variants present: {len(variants)}",
        f"- Records: {len(rows)} / {sum(row['cells_expected'] for row in variants)}",
        f"- Complete: {'yes' if complete else 'no'}",
        "- Ranking is step-based. Runner wall time is retained only as operational metadata.",
        "",
        "| rank | variant | cells | watchdog | completed | horizon | global stall | residual agents | agent completion | makespan median | moves median | W/L/T vs base |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(variants, 1):
        lines.append(
            f"| {rank} | {row['variant']} | {row['cells_present']}/{row['cells_expected']} | "
            f"{row['watchdogs']} | {row['completed_cells']} | {row['horizon_cells']} | "
            f"{row['global_stall_cells']} | {row['residual_agents']} | "
            f"{row['agent_completion_fraction'] * 100:.2f}% | "
            f"{fmt(row['makespan_median'], 0)} | {fmt(row['moves_median_completed'], 0)} | "
            f"{row['paired_wins_vs_base']}/{row['paired_losses_vs_base']}/"
            f"{row['paired_ties_vs_base']} |"
        )
    lines += [
        "",
        "`cells.csv`, `variants.csv`, and `map_density.csv` contain the auditable data.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
