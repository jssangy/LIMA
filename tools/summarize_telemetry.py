#!/usr/bin/env python3
"""Summarize passive LIMA telemetry without rerunning an experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0, "min": None, "mean": None, "max": None,
            "variance": None, "p50": None, "p90": None, "p99": None,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)

    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "variance": statistics.pvariance(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
    }


def summarize_metrics(directory: Path) -> dict:
    communication = rows(directory / "communication_events.csv")
    communication_steps = rows(directory / "comm_steps.csv")
    communication_by_type = {}
    for event_type in sorted({row["type"] for row in communication}):
        selected = [row for row in communication if row["type"] == event_type]
        communication_by_type[event_type] = {
            "distance_cells": distribution([float(row["distance_cells"]) for row in selected]),
            "intersection_hops": distribution(
                [float(row["intersection_hops"]) for row in selected]),
        }
    agents = rows(directory / "agents.csv")
    reference_moves = [max(0.0, float(row["initial_route_len"]) - 1.0) for row in agents]
    actual_moves = [float(row["moves"]) for row in agents]
    detour_moves = [float(row["extra_moves"]) for row in agents]
    execution_over_reference = [
        actual / reference
        for actual, reference in zip(actual_moves, reference_moves)
        if reference > 0
    ]
    completed_agents = [row for row in agents if row["completed"] == "1"]
    completed_reference_moves = [
        max(0.0, float(row["initial_route_len"]) - 1.0) for row in completed_agents]
    completed_actual_moves = [float(row["moves"]) for row in completed_agents]
    completed_detour_moves = [float(row["extra_moves"]) for row in completed_agents]
    completed_execution_over_reference = [
        actual / reference
        for actual, reference in zip(completed_actual_moves, completed_reference_moves)
        if reference > 0
    ]
    recirculation = rows(directory / "recirculation_segments.csv")
    mutations = rows(directory / "route_mutations.csv")
    validation_rows = rows(directory / "path_validation.csv")
    validation = validation_rows[0] if validation_rows else {}
    acquisitions_per_step = [
        float(row["acquisitions"]) for row in communication_steps]
    broadcasts_per_step = [float(row["broadcasts"]) for row in communication_steps]
    gate_signals_per_step = [
        float(row["gate_signals"]) for row in communication_steps]
    total_per_step = [
        acquisition + broadcast + gate_signal
        for acquisition, broadcast, gate_signal in zip(
            acquisitions_per_step, broadcasts_per_step, gate_signals_per_step)
    ]
    return {
        "communication": {
            "event_count": len(communication),
            "event_count_by_type": {
                event_type: sum(row["type"] == event_type for row in communication)
                for event_type in sorted({row["type"] for row in communication})
            },
            "events_per_step": {
                "total": distribution(total_per_step),
                "acquisitions": distribution(acquisitions_per_step),
                "broadcasts": distribution(broadcasts_per_step),
                "gate_signals": distribution(gate_signals_per_step),
            },
            "distance_cells": distribution(
                [float(row["distance_cells"]) for row in communication]),
            "intersection_hops": distribution(
                [float(row["intersection_hops"]) for row in communication]),
            "by_type": communication_by_type,
        },
        "detour": {
            "all_agents": {
                "reference_moves": distribution(reference_moves),
                "actual_moves": distribution(actual_moves),
                "extra_moves": distribution(detour_moves),
                "execution_over_reference": distribution(execution_over_reference),
            },
            "completed_agents": {
                "count": len(completed_agents),
                "reference_moves": distribution(completed_reference_moves),
                "actual_moves": distribution(completed_actual_moves),
                "extra_moves": distribution(completed_detour_moves),
                "execution_over_reference": distribution(
                    completed_execution_over_reference),
            },
            "recirculation_loop_cells": distribution(
                [float(row["loop_cells"]) for row in recirculation]),
            "route_mutation_inserted_cells": distribution(
                [float(row["inserted_cells"]) for row in mutations]),
        },
        "path_conformity": {
            "steps_observed": int(validation.get("steps_observed", 0)),
            "invalid_moves": int(validation.get("invalid_moves", 0)),
            "vertex_conflicts": int(validation.get("vertex_conflicts", 0)),
            "edge_conflicts": int(validation.get("edge_conflicts", 0)),
            "completed_goal_mismatches": int(
                validation.get("completed_goal_mismatches", 0)),
            "online_validation_ok": validation.get("ok") == "1",
            "route_mutations": len(mutations),
            "rejoin_failures": sum(row["rejoin_ok"] != "1" for row in mutations),
            "goal_preservation_failures": sum(
                row["goal_preserved"] != "1" for row in mutations),
            "nonclosed_recirculations": sum(
                row["closed"] != "1" for row in recirculation),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = summarize_metrics(args.metrics_dir.resolve())
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
