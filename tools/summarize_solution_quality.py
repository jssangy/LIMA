#!/usr/bin/env python3
"""Compare Gate A solution lengths on identical instances with exact references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALIASES = {"beam": "beam2048"}
EXACT_REFERENCES = {"astar_bf", "astar_tt", "ida_tt_opt", "ida_tt_optdom"}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def output_path(payload: dict) -> Path:
    command = payload["command"]
    index = command.index("--output")
    return Path(command[index + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cells: dict[tuple[str, str, int], tuple[dict, Path]] = {}
    for raw_directory in args.input_dirs:
        directory = (ROOT / raw_directory).resolve()
        for record_path in sorted((directory / "records").glob("*.json")):
            payload = json.loads(record_path.read_text())
            solver = ALIASES.get(payload["solver"], payload["solver"])
            key = (solver, payload["shape"], int(payload["population"]))
            cells[key] = (payload, output_path(payload))

    lengths: dict[tuple[str, str, int, int], int] = {}
    bounds: dict[tuple[str, int], int] = {}
    for (solver, shape, population), (payload, raw_path) in cells.items():
        bounds[(shape, population)] = int(payload["bound"])
        with raw_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["outcome"] == "solved":
                    lengths[(solver, shape, population, int(row["instance"]))] = int(
                        row["solution_len"]
                    )

    references: dict[tuple[str, int, int], int] = {}
    reference_values: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for (solver, shape, population, instance), length in lengths.items():
        if solver in EXACT_REFERENCES:
            reference_values[(shape, population, instance)].add(length)
    for key, values in reference_values.items():
        optimum = min(values)
        if optimum > 0:
            references[key] = optimum

    disagreement = sum(len(values) > 1 for values in reference_values.values())
    solvers = sorted({key[0] for key in cells})
    rows: list[dict] = []
    for scope in ("all", "density>=0.8"):
        selected_refs = {
            key: optimal
            for key, optimal in references.items()
            if scope == "all" or key[1] / bounds[(key[0], key[1])] >= 0.8
        }
        for solver in solvers:
            ratios: list[float] = []
            exact_matches = 0
            for (shape, population, instance), optimal in selected_refs.items():
                length = lengths.get((solver, shape, population, instance))
                if length is None:
                    continue
                ratios.append(length / optimal)
                exact_matches += length == optimal
            rows.append(
                {
                    "scope": scope,
                    "solver": solver,
                    "reference_instances": len(selected_refs),
                    "common_solved": len(ratios),
                    "coverage": len(ratios) / len(selected_refs),
                    "exact_match": exact_matches / len(ratios) if ratios else float("nan"),
                    "ratio_mean": sum(ratios) / len(ratios) if ratios else float("nan"),
                    "ratio_median": percentile(ratios, 0.5),
                    "ratio_p90": percentile(ratios, 0.9),
                    "ratio_p99": percentile(ratios, 0.99),
                    "ratio_max": max(ratios, default=float("nan")),
                }
            )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "solution_quality.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    high_rows = [row for row in rows if row["scope"] == "density>=0.8"]
    high_rows.sort(key=lambda row: (row["ratio_p90"], -row["coverage"]))
    lines = [
        "# Gate A solution-quality comparison",
        "",
        f"Exact-reference instances: {len(references)}; disagreements: {disagreement}.",
        "High density means N/B >= 0.8. Ratio = solver length / exact-reference length.",
        "",
        "| solver | coverage | exact match | ratio mean/median/p90/p99/max |",
        "|---|---:|---:|---:|",
    ]
    for row in high_rows:
        lines.append(
            f"| {row['solver']} | {row['coverage'] * 100:.2f}% "
            f"({row['common_solved']}/{row['reference_instances']}) | "
            f"{row['exact_match'] * 100:.2f}% | {row['ratio_mean']:.3f}/"
            f"{row['ratio_median']:.3f}/{row['ratio_p90']:.3f}/"
            f"{row['ratio_p99']:.3f}/{row['ratio_max']:.3f} |"
        )
    (output_dir / "SOLUTION_QUALITY.md").write_text("\n".join(lines) + "\n")
    print(output_dir / "SOLUTION_QUALITY.md")


if __name__ == "__main__":
    main()
