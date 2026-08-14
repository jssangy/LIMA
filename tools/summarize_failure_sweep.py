#!/usr/bin/env python3
"""Summarize E12 actuator-command failure sweeps from revision-grid records."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def probability(manifest: dict) -> float:
    flags = manifest.get("variant_flags", [])
    if "--failure-prob" not in flags:
        return 0.0
    return float(flags[flags.index("--failure-prob") + 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, int, int, float], list[dict]] = defaultdict(list)
    horizons: set[int] = set()
    for name in args.directories:
        directory = Path(name)
        manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
        p = probability(manifest)
        horizons.add(int(manifest["max_steps"]))
        for record_file in sorted((directory / "records").glob("*.json")):
            record = json.loads(record_file.read_text(encoding="utf-8"))
            grouped[(record["map"], record["density_percent"], record["agents"], p)].append(record)

    rows = []
    for (map_name, density, agents, p), records in sorted(grouped.items()):
        completed = [r for r in records if r.get("summary", {}).get("status") == "completed"]
        ratios = []
        failures_per_k_agent_steps = []
        for record in records:
            summary = record.get("summary", {})
            served, total = (int(v) for v in summary.get("completed", f"0/{agents}").split("/"))
            ratios.append(served / total if total else 0.0)
            steps = int(summary.get("steps", 0))
            failures = int(summary.get("failures", 0))
            denominator = agents * steps
            failures_per_k_agent_steps.append(1000.0 * failures / denominator if denominator else 0.0)
        rows.append({
            "map": map_name,
            "density_percent": density,
            "agents": agents,
            "failure_probability": p,
            "runs": len(records),
            "successes": len(completed),
            "completion_ratio_median": statistics.median(ratios),
            "makespan_median": statistics.median(
                int(r["summary"]["steps"]) for r in completed
            ) if completed else None,
            "runtime_seconds_median": statistics.median(
                float(r["summary"].get("elapsed_seconds", r["runner_wall_seconds"])) for r in records
            ),
            "failures_per_k_agent_steps_median": statistics.median(failures_per_k_agent_steps),
        })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "summary_failure_groups.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# E12 probabilistic command-failure report",
        "",
        f"- Fixed execution horizon: {', '.join(str(v) for v in sorted(horizons))} steps.",
        "- A sampled command loss becomes a wait; coupled/dependent moves stop conservatively to preserve safety.",
        "- Failure intensity is reported per 1,000 agent-steps; it is not a collision rate.",
        "",
        "| map | density | agents | p | success | median completed | makespan | runtime | failures / 1k agent-steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        makespan = "-" if row["makespan_median"] is None else f"{row['makespan_median']:.0f}"
        lines.append(
            f"| {row['map']} | {row['density_percent']}% | {row['agents']} | {row['failure_probability']:.2f} "
            f"| {row['successes']}/{row['runs']} | {100 * row['completion_ratio_median']:.1f}% "
            f"| {makespan} | {row['runtime_seconds_median']:.2f}s "
            f"| {row['failures_per_k_agent_steps_median']:.3f} |"
        )
    lines += ["", "`summary_failure_groups.csv` contains the auditable grouped data.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
