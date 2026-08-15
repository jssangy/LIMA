#!/usr/bin/env python3
"""Expand certified boundary placements into a nested density ladder.

Every generated start set is a prefix of a capacity-certified maximum
placement. Since local capacity constraints are nonnegative upper bounds,
every prefix remains certified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from generate_capacity_certified_scenarios import (
    ROOT,
    SPECS,
    Grid,
    atomic_json,
    atomic_text,
    build_topology,
    component_labels,
    make_constraints,
    parse_int_list,
    sha256,
)


DENSITY_LADDERS = {
    "warehouse_10_20": (1, 5, 10, 20, 30, 40, 50, 60),
    "warehouse_20_40": (1, 5, 10, 20, 30, 40, 50),
    "cross_3030": (1, 5, 10, 20, 30, 40, 50, 60, 65, 70),
}


def parse_scenario(path: Path) -> tuple[str, list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "version 1":
        raise ValueError(f"not a MovingAI scenario: {path}")
    rows = [line.split() for line in lines[1:] if line.strip()]
    if any(len(row) < 9 for row in rows):
        raise ValueError(f"malformed MovingAI scenario row: {path}")
    return lines[0], rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=",".join(SPECS))
    parser.add_argument("--scenarios", default="0-9")
    parser.add_argument(
        "--source-manifest",
        default="results/revision_final/certified_inputs_v1/MANIFEST.json",
    )
    parser.add_argument(
        "--output-root",
        default="results/revision_final/certified_inputs_v2",
    )
    args = parser.parse_args()

    maps = [item.strip() for item in args.maps.split(",") if item.strip()]
    if not maps or not set(maps).issubset(SPECS):
        parser.error("unknown or empty map selection")
    scenarios = parse_int_list(args.scenarios, set(range(10)))
    source_manifest_path = (ROOT / args.source_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("capacity_formula") != "sum(arm capacities) - longest arm":
        parser.error("source manifest does not use the operational capacity")
    output = (ROOT / args.output_root).resolve()
    script = Path(__file__).resolve()
    manifest: dict = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(script.relative_to(ROOT)),
        "generator_sha256": sha256(script),
        "source_manifest": str(source_manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": sha256(source_manifest_path),
        "capacity_formula": "sum(arm capacities) - longest arm",
        "start_domain": "managed intersection centers and arms only",
        "construction": "nested prefixes of a certified maximum placement",
        "maps": {},
    }

    generated = 0
    for map_name in maps:
        spec = SPECS[map_name]
        source_map = source_manifest["maps"][map_name]
        boundary = source_map["targets"]["boundary"]
        map_path = ROOT / source_map["map_file"]
        if sha256(map_path) != source_map["map_sha256"]:
            raise ValueError(f"map hash mismatch: {map_name}")
        grid = Grid(map_path)
        topology = build_topology(grid)
        components = component_labels(grid)
        managed_cells = sorted({
            cell
            for intersection in topology
            for cell in [
                intersection["center"],
                *(cell for arm in intersection["arms"] for cell in arm),
            ]
        })
        managed_index = {cell: column for column, cell in enumerate(managed_cells)}
        matrix, capacities = make_constraints(topology, managed_cells)
        maximum = int(source_map["maximum_agents"])
        labels = [
            (f"d{density:02d}", density * spec.tiles // 100)
            for density in DENSITY_LADDERS[map_name]
        ]
        labels.append(("boundary", maximum))
        if any(agents > maximum for _, agents in labels):
            raise AssertionError(f"{map_name}: ladder exceeds certified maximum")

        map_entry = {
            "map_file": source_map["map_file"],
            "map_sha256": source_map["map_sha256"],
            "tiles": spec.tiles,
            "managed_cells": len(managed_cells),
            "intersections": len(topology),
            "maximum_agents": maximum,
            "boundary_percent": 100.0 * maximum / spec.tiles,
            "targets": {},
        }
        manifest["maps"][map_name] = map_entry

        for label, agents in labels:
            target_entry = {
                "agents": agents,
                "tile_density_percent": 100.0 * agents / spec.tiles,
                "capacity_load_percent": 100.0 * agents / maximum,
                "scenarios": {},
            }
            map_entry["targets"][label] = target_entry
            for scenario in scenarios:
                source_entry = boundary["scenarios"][str(scenario)]
                source_scenario = ROOT / source_entry["scenario_file"]
                source_certificate = ROOT / source_entry["certificate_file"]
                if sha256(source_scenario) != source_entry["scenario_sha256"]:
                    raise ValueError(f"source scenario hash mismatch: {source_scenario}")
                if sha256(source_certificate) != source_entry["certificate_sha256"]:
                    raise ValueError(f"source certificate hash mismatch: {source_certificate}")
                source_cert = json.loads(source_certificate.read_text(encoding="utf-8"))
                if source_cert["validation"]["capacity_violations"] != 0:
                    raise ValueError(f"invalid boundary certificate: {source_certificate}")
                header, rows = parse_scenario(source_scenario)
                if len(rows) != maximum or agents > len(rows):
                    raise ValueError(f"unexpected boundary size: {source_scenario}")
                selected_rows = rows[:agents]
                starts: list[int] = []
                for row in selected_rows:
                    sx, sy, gx, gy = map(int, row[4:8])
                    start = grid.cell(sx, sy)
                    goal = grid.cell(gx, gy)
                    if (
                        start not in managed_index
                        or goal not in grid.traversable
                        or start == goal
                        or components[start] != components[goal]
                    ):
                        raise AssertionError(f"invalid start-goal pair in {source_scenario}")
                    starts.append(start)
                if len(set(starts)) != agents:
                    raise AssertionError(f"duplicate starts in {source_scenario}")
                selected_columns = np.array(
                    [managed_index[cell] for cell in starts], dtype=int
                )
                occupancies = np.asarray(
                    matrix[:, selected_columns].sum(axis=1)
                ).ravel().astype(int)
                violations = int(np.sum(occupancies > capacities))
                if violations:
                    raise AssertionError(
                        f"{map_name} {label} s{scenario}: {violations} capacity violations"
                    )

                scenario_path = (
                    output / "scenarios" / map_name /
                    f"{map_name}_cert_{label}_s{scenario}.scen"
                )
                text = header + "\n" + "\n".join(
                    "\t".join(row) for row in selected_rows
                ) + "\n"
                atomic_text(scenario_path, text)
                certificate_path = (
                    output / "certificates" / map_name / f"{label}_s{scenario}.json"
                )
                minimum_slack = int(np.min(capacities.astype(int) - occupancies))
                certificate = {
                    "map": map_name,
                    "map_file": source_map["map_file"],
                    "map_sha256": source_map["map_sha256"],
                    "target": label,
                    "agents": agents,
                    "tile_density_percent": 100.0 * agents / spec.tiles,
                    "capacity_load_percent": 100.0 * agents / maximum,
                    "scenario": scenario,
                    "capacity_formula": "sum-minus-max",
                    "maximum_agents": maximum,
                    "construction": "prefix of certified boundary placement",
                    "source_boundary_scenario": str(source_scenario.relative_to(ROOT)),
                    "source_boundary_scenario_sha256": sha256(source_scenario),
                    "source_boundary_certificate": str(source_certificate.relative_to(ROOT)),
                    "source_boundary_certificate_sha256": sha256(source_certificate),
                    "scenario_file": str(scenario_path.relative_to(ROOT)),
                    "scenario_sha256": sha256(scenario_path),
                    "validation": {
                        "unique_starts": len(set(starts)) == agents,
                        "managed_starts": all(cell in managed_index for cell in starts),
                        "reachable_pairs": agents,
                        "capacity_violations": violations,
                        "maximum_occupancy": int(occupancies.max(initial=0)),
                        "minimum_slack": minimum_slack,
                    },
                    "intersection_constraints": [
                        {
                            "intersection": index,
                            "occupancy": int(occupancy),
                            "capacity": int(capacity),
                            "slack": int(capacity - occupancy),
                        }
                        for index, (occupancy, capacity) in enumerate(
                            zip(occupancies, capacities)
                        )
                    ],
                }
                atomic_json(certificate_path, certificate)
                target_entry["scenarios"][str(scenario)] = {
                    "scenario_file": str(scenario_path.relative_to(ROOT)),
                    "scenario_sha256": sha256(scenario_path),
                    "certificate_file": str(certificate_path.relative_to(ROOT)),
                    "certificate_sha256": sha256(certificate_path),
                }
                generated += 1
                print(
                    f"[{generated:3d}] {map_name} {label} s{scenario}: "
                    f"N={agents}/{maximum} min_slack={minimum_slack}",
                    flush=True,
                )

    atomic_json(output / "MANIFEST.json", manifest)
    print(f"generated {generated} certified scenarios under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
