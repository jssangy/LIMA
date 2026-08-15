#!/usr/bin/env python3
"""Generate paired, capacity-certified one-shot MAPF inputs.

Starts are prefixes of a certified maximum placement.  Every agent keeps its
original physical exit but receives a unique cell on a private terminal lane
outside the managed warehouse core.  Thus starts and goals are unique,
reachable, fixed-point-free, and capacity-safe at every density prefix.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def occupancy(
    matrix: np.ndarray, managed_index: dict[int, int], cells: list[int]
) -> np.ndarray:
    columns = np.array(
        [managed_index[cell] for cell in cells if cell in managed_index], dtype=int
    )
    if columns.size == 0:
        return np.zeros(matrix.shape[0], dtype=int)
    return np.asarray(matrix[:, columns].sum(axis=1)).ravel().astype(int)


def outward_direction(grid: Grid, sink: int) -> tuple[int, int]:
    x, y = grid.coord(sink)
    if x == 0:
        return -1, 0
    if x == grid.width - 1:
        return 1, 0
    if y == 0:
        return 0, -1
    if y == grid.height - 1:
        return 0, 1
    return (-1, 0) if x < grid.width // 2 else (1, 0)


def build_terminal_map(
    source_path: Path,
    boundary_rows: dict[int, list[list[str]]],
    output_path: Path,
) -> tuple[Grid, Grid, int, int, dict[int, list[int]], dict[int, int]]:
    """Append private outward terminal lanes to every physical exit."""
    source = Grid(source_path)
    capacities = {sink: 0 for sink in source.goals}
    for rows in boundary_rows.values():
        counts = Counter(
            source.cell(*map(int, row[6:8])) for row in rows
        )
        if not set(counts).issubset(source.goals):
            raise AssertionError("source task goal is not a physical exit")
        for sink, count in counts.items():
            capacities[sink] = max(capacities[sink], count)
    if not capacities or min(capacities.values()) < 1:
        raise AssertionError("every physical exit needs at least one terminal slot")
    source_components = component_labels(source)
    start_components = {
        source_components[source.cell(*map(int, row[4:6]))]
        for rows in boundary_rows.values()
        for row in rows
    }
    if len(start_components) != 1:
        raise AssertionError("certified starts must occupy one warehouse-core component")
    core_component = next(iter(start_components))

    extents = [(0, 0), (source.width - 1, source.height - 1)]
    directions = {sink: outward_direction(source, sink) for sink in capacities}
    for sink, capacity in capacities.items():
        x, y = source.coord(sink)
        dx, dy = directions[sink]
        extents.append((x + dx * capacity, y + dy * capacity))
    min_x = min(x for x, _ in extents)
    min_y = min(y for _, y in extents)
    max_x = max(x for x, _ in extents)
    max_y = max(y for _, y in extents)
    offset_x, offset_y = -min_x, -min_y
    width, height = max_x - min_x + 1, max_y - min_y + 1
    body = [["@"] * width for _ in range(height)]
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    for y, row in enumerate(source.rows):
        for x, value in enumerate(row):
            cell = source.cell(x, y)
            if cell in source.traversable and source_components[cell] != core_component:
                value = "@"
            body[y + offset_y][x + offset_x] = "G" if value == "S" else value

    slot_coords: dict[int, list[tuple[int, int]]] = {}
    occupied_tail_cells: set[tuple[int, int]] = set()
    for sink, capacity in capacities.items():
        x, y = source.coord(sink)
        dx, dy = directions[sink]
        slots = []
        for step in range(1, capacity + 1):
            tx, ty = x + dx * step + offset_x, y + dy * step + offset_y
            if (tx, ty) in occupied_tail_cells:
                raise AssertionError("terminal lanes overlap")
            existing = body[ty][tx]
            if existing == "G":
                raise AssertionError(
                    f"terminal lane from {source.coord(sink)} crosses another exit "
                    f"at source-relative {(tx - offset_x, ty - offset_y)}"
                )
            body[ty][tx] = "E"
            occupied_tail_cells.add((tx, ty))
            slots.append((tx, ty))
        slot_coords[sink] = slots

    text = "\n".join([
        source_lines[0], f"height {height}", f"width {width}", "map",
        *("".join(row) for row in body),
    ]) + "\n"
    atomic_text(output_path, text)
    expanded = Grid(output_path)
    terminal_slots = {
        sink: [expanded.cell(x, y) for x, y in slots]
        for sink, slots in slot_coords.items()
    }
    return expanded, source, offset_x, offset_y, terminal_slots, capacities


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
        default="results/revision_final/certified_inputs_v3",
    )
    args = parser.parse_args()

    maps = [item.strip() for item in args.maps.split(",") if item.strip()]
    if not maps or not set(maps).issubset(SPECS):
        parser.error("unknown or empty map selection")
    scenarios = parse_int_list(args.scenarios, set(range(10)))
    if len(scenarios) < 2:
        parser.error("paired construction requires at least two source scenarios")
    source_manifest_path = (ROOT / args.source_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("capacity_formula") != "sum(arm capacities) - longest arm":
        parser.error("source manifest does not use the operational capacity")
    output = (ROOT / args.output_root).resolve()
    script = Path(__file__).resolve()
    manifest: dict = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(script.relative_to(ROOT)),
        "generator_sha256": sha256(script),
        "source_manifest": str(source_manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": sha256(source_manifest_path),
        "capacity_formula": "sum(arm capacities) - longest arm",
        "start_domain": "managed intersection centers and arms only",
        "goal_domain": (
            "unique off-managed terminal lanes appended outward from each physical exit"
        ),
        "task_semantics": "globally unique starts and goals; no start equals its paired goal",
        "map_annotation_semantics": (
            "the warehouse core graph and intersection topology are unchanged; "
            "one private outward terminal lane is appended to each legacy S exit"
        ),
        "construction": (
            "starts are nested prefixes of a certified maximum placement; each agent "
            "keeps its original physical exit and receives the next unique cell on that "
            "exit's off-managed terminal lane"
        ),
        "maps": {},
    }

    generated = 0
    for map_name in maps:
        spec = SPECS[map_name]
        source_map = source_manifest["maps"][map_name]
        boundary = source_map["targets"]["boundary"]
        maximum = int(source_map["maximum_agents"])
        source_map_path = ROOT / source_map["map_file"]
        if sha256(source_map_path) != source_map["map_sha256"]:
            raise ValueError(f"map hash mismatch: {map_name}")
        boundary_rows: dict[int, list[list[str]]] = {}
        for scenario in scenarios:
            entry = boundary["scenarios"][str(scenario)]
            scenario_path = ROOT / entry["scenario_file"]
            if sha256(scenario_path) != entry["scenario_sha256"]:
                raise ValueError(f"source scenario hash mismatch: {scenario_path}")
            _, rows = parse_scenario(scenario_path)
            if len(rows) != maximum:
                raise ValueError(f"unexpected boundary size: {scenario_path}")
            boundary_rows[scenario] = rows
        map_path = output / "maps" / f"{map_name}_unique_goals.map"
        (
            grid, source_grid, offset_x, offset_y, terminal_slots, terminal_capacities,
        ) = build_terminal_map(
            source_map_path, boundary_rows, map_path,
        )
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
        labels = [
            (f"d{density:02d}", density * spec.tiles // 100)
            for density in DENSITY_LADDERS[map_name]
        ]
        labels.append(("boundary", maximum))
        if any(agents > maximum for _, agents in labels):
            raise AssertionError(f"{map_name}: ladder exceeds certified maximum")

        map_entry = {
            "map_file": str(map_path.relative_to(ROOT)),
            "map_sha256": sha256(map_path),
            "source_map_file": source_map["map_file"],
            "source_map_sha256": source_map["map_sha256"],
            "map_annotation_transform": (
                "S to G plus private outward E terminal lanes; warehouse core unchanged"
            ),
            "source_offset": {"x": offset_x, "y": offset_y},
            "terminal_slots": sum(len(slots) for slots in terminal_slots.values()),
            "physical_exits": len(terminal_slots),
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
                start_entry = boundary["scenarios"][str(scenario)]
                start_source = ROOT / start_entry["scenario_file"]
                start_certificate = ROOT / start_entry["certificate_file"]
                if sha256(start_source) != start_entry["scenario_sha256"]:
                    raise ValueError(f"source scenario hash mismatch: {start_source}")
                if sha256(start_certificate) != start_entry["certificate_sha256"]:
                    raise ValueError(f"source certificate hash mismatch: {start_certificate}")
                source_cert = json.loads(start_certificate.read_text(encoding="utf-8"))
                if source_cert["validation"]["capacity_violations"] != 0:
                    raise ValueError(f"invalid boundary certificate: {start_certificate}")
                header, start_rows = parse_scenario(start_source)
                if len(start_rows) != maximum or agents > len(start_rows):
                    raise ValueError(f"unexpected boundary size: {start_source}")
                full_starts = [
                    grid.cell(
                        int(row[4]) + offset_x,
                        int(row[5]) + offset_y,
                    )
                    for row in start_rows[:maximum]
                ]
                starts = full_starts[:agents]
                exit_cursors: dict[int, int] = defaultdict(int)
                full_goals: list[int] = []
                original_exits: list[int] = []
                for row in start_rows[:maximum]:
                    exit_cell = source_grid.cell(int(row[6]), int(row[7]))
                    slot = exit_cursors[exit_cell]
                    if exit_cell not in terminal_slots or slot >= len(terminal_slots[exit_cell]):
                        raise AssertionError("physical exit terminal capacity exhausted")
                    original_exits.append(exit_cell)
                    full_goals.append(terminal_slots[exit_cell][slot])
                    exit_cursors[exit_cell] += 1
                goals = full_goals[:agents]
                if len(set(starts)) != agents or len(set(goals)) != agents:
                    raise AssertionError("source placement contains duplicate cells")
                if not all(cell in managed_index for cell in starts):
                    raise AssertionError("start placement leaves the managed domain")
                if not all(cell in grid.traversable for cell in goals):
                    raise AssertionError("goal placement leaves the traversable graph")
                if {
                    components[cell] for cell in starts
                } != {components[cell] for cell in goals}:
                    raise AssertionError("start and goal placements occupy different components")
                if any(components[start] != components[goal] for start, goal in zip(starts, goals)):
                    raise AssertionError("paired start and goal are disconnected")

                start_occupancies = occupancy(matrix, managed_index, starts)
                goal_occupancies = occupancy(matrix, managed_index, goals)
                start_violations = int(np.sum(start_occupancies > capacities))
                goal_violations = int(np.sum(goal_occupancies > capacities))
                if start_violations or goal_violations:
                    raise AssertionError(
                        f"{map_name} {label} s{scenario}: start={start_violations}, "
                        f"goal={goal_violations} capacity violations"
                    )

                scenario_path = (
                    output / "scenarios" / map_name /
                    f"{map_name}_paired_{label}_s{scenario}.scen"
                )
                map_filename = map_path.name
                scenario_rows = []
                for start, goal in zip(starts, goals):
                    sx, sy = grid.coord(start)
                    gx, gy = grid.coord(goal)
                    scenario_rows.append([
                        "0", map_filename, str(grid.width), str(grid.height),
                        str(sx), str(sy), str(gx), str(gy), "0",
                    ])
                text = header + "\n" + "\n".join(
                    "\t".join(row) for row in scenario_rows
                ) + "\n"
                atomic_text(scenario_path, text)
                certificate_path = (
                    output / "certificates" / map_name / f"{label}_s{scenario}.json"
                )
                start_minimum_slack = int(np.min(capacities.astype(int) - start_occupancies))
                goal_minimum_slack = int(np.min(capacities.astype(int) - goal_occupancies))
                certificate = {
                    "map": map_name,
                    "map_file": str(map_path.relative_to(ROOT)),
                    "map_sha256": sha256(map_path),
                    "source_map_file": source_map["map_file"],
                    "source_map_sha256": source_map["map_sha256"],
                    "map_annotation_transform": (
                        "S to G plus private outward E terminal lanes; "
                        "warehouse core unchanged"
                    ),
                    "target": label,
                    "agents": agents,
                    "tile_density_percent": 100.0 * agents / spec.tiles,
                    "capacity_load_percent": 100.0 * agents / maximum,
                    "scenario": scenario,
                    "capacity_formula": "sum-minus-max",
                    "maximum_agents": maximum,
                    "construction": (
                        "certified managed start prefix; each agent keeps its original "
                        "physical exit and receives the next cell on its private "
                        "off-managed terminal lane"
                    ),
                    "goal_assignment": "next unique terminal-lane slot at original exit",
                    "source_start_boundary_scenario": str(start_source.relative_to(ROOT)),
                    "source_start_boundary_scenario_sha256": sha256(start_source),
                    "source_start_boundary_certificate": str(start_certificate.relative_to(ROOT)),
                    "source_start_boundary_certificate_sha256": sha256(start_certificate),
                    "scenario_file": str(scenario_path.relative_to(ROOT)),
                    "scenario_sha256": sha256(scenario_path),
                    "validation": {
                        "unique_starts": len(set(starts)) == agents,
                        "unique_goals": len(set(goals)) == agents,
                        "managed_starts": all(cell in managed_index for cell in starts),
                        "traversable_goals": all(cell in grid.traversable for cell in goals),
                        "goal_off_managed_count": sum(
                            cell not in managed_index for cell in goals
                        ),
                        "goal_managed_count": sum(cell in managed_index for cell in goals),
                        "goal_terminal_count": agents,
                        "goal_pool_off_managed_total": sum(
                            len(slots) for slots in terminal_slots.values()
                        ),
                        "physical_exits_used": len(set(original_exits[:agents])),
                        "same_agent_start_goal": sum(
                            start == goal for start, goal in zip(starts, goals)
                        ),
                        "reachable_pairs": agents,
                        "capacity_violations": start_violations + goal_violations,
                        "start_capacity_violations": start_violations,
                        "goal_capacity_violations": goal_violations,
                        "start_maximum_occupancy": int(start_occupancies.max(initial=0)),
                        "goal_maximum_occupancy": int(goal_occupancies.max(initial=0)),
                        "start_minimum_slack": start_minimum_slack,
                        "goal_minimum_slack": goal_minimum_slack,
                        "start_goal_set_overlap": len(set(starts) & set(goals)),
                    },
                    "intersection_constraints": [
                        {
                            "intersection": index,
                            "start_occupancy": int(start_occupancy),
                            "goal_occupancy": int(goal_occupancy),
                            "capacity": int(capacity),
                            "start_slack": int(capacity - start_occupancy),
                            "goal_slack": int(capacity - goal_occupancy),
                        }
                        for index, (start_occupancy, goal_occupancy, capacity) in enumerate(
                            zip(start_occupancies, goal_occupancies, capacities)
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
                    f"N={agents}/{maximum} start_slack={start_minimum_slack} "
                    f"goal_slack={goal_minimum_slack}",
                    flush=True,
                )

    atomic_json(output / "MANIFEST.json", manifest)
    print(f"generated {generated} certified scenarios under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
