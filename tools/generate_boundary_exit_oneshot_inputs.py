#!/usr/bin/env python3
"""Project certified one-shot tasks onto the physical warehouse exit domain.

The v3 paper inputs encode every task with a unique cell on an outward terminal
lane.  Those cells are useful for persistent-goal MAPF, but they also expose an
off-mission buffer.  This derivation preserves each task's certified start and
physical exit while replacing its terminal-lane goal by the corresponding G
workstation. Every off-managed traversable cell is blocked. A G workstation is
an agent-specific completion event: an active agent may enter only its assigned
G, and it disappears immediately after that arrival step. Repeated assignments
to the same physical G are therefore well-defined without persistent occupancy.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from generate_capacity_certified_scenarios import (
    ROOT,
    Grid,
    atomic_json,
    atomic_text,
    build_topology,
    component_labels,
    sha256,
)


DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def terminal_to_boundary(grid: Grid) -> dict[tuple[int, int], tuple[int, int]]:
    terminals = {
        (x, y)
        for y, row in enumerate(grid.rows)
        for x, value in enumerate(row)
        if value == "E"
    }
    boundaries = {
        (x, y)
        for y, row in enumerate(grid.rows)
        for x, value in enumerate(row)
        if value == "G"
    }
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    unseen = set(terminals)
    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        component = {seed}
        roots: set[tuple[int, int]] = set()
        while queue:
            x, y = queue.popleft()
            for dx, dy in DELTAS:
                neighbor = x + dx, y + dy
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
                elif neighbor in boundaries:
                    roots.add(neighbor)
        if len(roots) != 1:
            raise ValueError(
                f"terminal component has {len(roots)} physical boundaries: {seed}"
            )
        root = next(iter(roots))
        mapping.update((cell, root) for cell in component)
    return mapping


def write_restricted_map(source: Path, destination: Path) -> tuple[Grid, dict, dict]:
    expanded = Grid(source)
    topology = build_topology(expanded)
    managed = {
        cell
        for intersection in topology
        for cell in [
            intersection["center"],
            *(cell for arm in intersection["arms"] for cell in arm),
        ]
    }
    boundary = {
        expanded.cell(x, y)
        for y, row in enumerate(expanded.rows)
        for x, value in enumerate(row)
        if value == "G"
    }
    allowed = managed | boundary
    removed = 0
    body: list[str] = []
    for y, row in enumerate(expanded.rows):
        output_row: list[str] = []
        for x, value in enumerate(row):
            if value in ".SEG" and expanded.cell(x, y) not in allowed:
                output_row.append("@")
                removed += 1
            else:
                output_row.append(value)
        body.append("".join(output_row))
    header = source.read_text(encoding="utf-8").splitlines()[:4]
    atomic_text(destination, "\n".join(header + body) + "\n")
    restricted = Grid(destination)
    stats = {
        "managed_cells": len(managed),
        "boundary_goal_cells": len(boundary),
        "allowed_cells": len(allowed),
        "removed_free_space_cells": removed,
    }
    return restricted, stats, terminal_to_boundary(expanded)


def parse_scenario(path: Path) -> tuple[str, list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "version 1":
        raise ValueError(f"not a MovingAI scenario: {path}")
    rows = [line.split() for line in lines[1:] if line.strip()]
    if any(len(row) < 9 for row in rows):
        raise ValueError(f"malformed MovingAI scenario: {path}")
    return lines[0], rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="results/revision_final/certified_inputs_v3/MANIFEST.json",
    )
    parser.add_argument(
        "--output-root",
        default="results/revision_final/certified_inputs_boundary_exit_v2",
    )
    args = parser.parse_args()
    source_manifest_path = (ROOT / args.source_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output = (ROOT / args.output_root).resolve()
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capacity_formula": source_manifest["capacity_formula"],
        "source_manifest": str(source_manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": sha256(source_manifest_path),
        "movement_domain": (
            "active positions are in the IntersectionTopology managed-cell union; "
            "the only allowed exit transition is arrival at the active agent's assigned G"
        ),
        "goal_semantics": {
            "type": "physical boundary service event",
            "completion": "agent disappears on first arrival at its assigned G workstation",
            "repeated_physical_goals": (
                "allowed across agents; they represent time-separated use of the same exit, "
                "not persistent co-occupancy"
            ),
            "boundary_entry_policy": (
                "an active agent may enter G iff that G is its assigned current goal"
            ),
            "collision_scope": (
                "vertex and edge conflicts are checked through the arrival step; "
                "the completed agent is absent from the next occupancy state"
            ),
        },
        "maps": {},
    }
    for map_name, source_map_entry in source_manifest["maps"].items():
        expanded_path = (ROOT / source_map_entry["map_file"]).resolve()
        if sha256(expanded_path) != source_map_entry["map_sha256"]:
            raise ValueError(f"expanded map hash mismatch: {map_name}")
        map_path = output / "maps" / f"{map_name}_managed_boundary.map"
        restricted, domain_stats, terminal_mapping = write_restricted_map(
            expanded_path, map_path
        )
        components = component_labels(restricted)
        map_entry = copy.deepcopy(source_map_entry)
        map_entry.update({
            "map_file": str(map_path.relative_to(ROOT)),
            "map_sha256": sha256(map_path),
            "expanded_source_map_file": source_map_entry["map_file"],
            "expanded_source_map_sha256": source_map_entry["map_sha256"],
            "map_annotation_transform": (
                "retain managed-cell union and G workstations; block terminal E lanes "
                "and every other off-managed traversable cell"
            ),
            "movement_domain_stats": domain_stats,
            "targets": {},
        })
        manifest["maps"][map_name] = map_entry
        for target, source_target in source_map_entry["targets"].items():
            target_entry = {
                key: copy.deepcopy(value)
                for key, value in source_target.items()
                if key != "scenarios"
            }
            target_entry["scenarios"] = {}
            map_entry["targets"][target] = target_entry
            for scenario_text, source_scenario_entry in source_target["scenarios"].items():
                source_scenario = (ROOT / source_scenario_entry["scenario_file"]).resolve()
                source_certificate = (ROOT / source_scenario_entry["certificate_file"]).resolve()
                if sha256(source_scenario) != source_scenario_entry["scenario_sha256"]:
                    raise ValueError(f"scenario hash mismatch: {source_scenario}")
                if sha256(source_certificate) != source_scenario_entry["certificate_sha256"]:
                    raise ValueError(f"certificate hash mismatch: {source_certificate}")
                header, rows = parse_scenario(source_scenario)
                starts: list[tuple[int, int]] = []
                goals: list[tuple[int, int]] = []
                converted: list[list[str]] = []
                for row in rows:
                    start = int(row[4]), int(row[5])
                    terminal = int(row[6]), int(row[7])
                    if terminal not in terminal_mapping:
                        raise ValueError(f"goal is not on a terminal lane: {terminal}")
                    goal = terminal_mapping[terminal]
                    start_id = restricted.cell(*start)
                    goal_id = restricted.cell(*goal)
                    if start_id not in restricted.traversable or goal_id not in restricted.goals:
                        raise ValueError(f"task leaves managed-boundary domain: {start}->{goal}")
                    if components[start_id] != components[goal_id]:
                        raise ValueError(f"unreachable managed-boundary task: {start}->{goal}")
                    starts.append(start)
                    goals.append(goal)
                    output_row = list(row)
                    output_row[1] = map_path.name
                    output_row[2] = str(restricted.width)
                    output_row[3] = str(restricted.height)
                    output_row[6] = str(goal[0])
                    output_row[7] = str(goal[1])
                    converted.append(output_row)
                scenario_path = (
                    output / "scenarios" / map_name /
                    f"{map_name}_boundary_exit_{target}_s{scenario_text}.scen"
                )
                atomic_text(
                    scenario_path,
                    header + "\n" + "\n".join("\t".join(row) for row in converted) + "\n",
                )
                source_cert = json.loads(source_certificate.read_text(encoding="utf-8"))
                validation = copy.deepcopy(source_cert["validation"])
                validation.update({
                    "unique_starts": len(set(starts)) == len(starts),
                    "unique_goals": len(set(goals)) == len(goals),
                    "repeated_goal_assignments": len(goals) - len(set(goals)),
                    "physical_boundary_goals": all(
                        restricted.cell(*goal) in restricted.goals for goal in goals
                    ),
                    "traversable_goals": True,
                    "same_agent_start_goal": sum(
                        start == goal for start, goal in zip(starts, goals)
                    ),
                    "reachable_pairs": len(starts),
                    "disappear_at_goal": True,
                    "exclusive_assigned_boundary_entry": True,
                    "persistent_goal_occupancy_required": False,
                    "goal_off_managed_count": len(goals),
                    "goal_terminal_count": 0,
                    "goal_pool_off_managed_total": 0,
                    "physical_exits_used": len(set(goals)),
                    "start_goal_set_overlap": len(set(starts) & set(goals)),
                })
                certificate_path = (
                    output / "certificates" / map_name /
                    f"{target}_s{scenario_text}.json"
                )
                certificate = {
                    **source_cert,
                    "map_file": str(map_path.relative_to(ROOT)),
                    "map_sha256": sha256(map_path),
                    "expanded_source_map_file": source_map_entry["map_file"],
                    "expanded_source_map_sha256": source_map_entry["map_sha256"],
                    "scenario_file": str(scenario_path.relative_to(ROOT)),
                    "scenario_sha256": sha256(scenario_path),
                    "expanded_source_scenario_file": source_scenario_entry["scenario_file"],
                    "expanded_source_scenario_sha256": source_scenario_entry["scenario_sha256"],
                    "construction": (
                        "certified managed start prefix; original terminal-lane goal projected "
                        "to its physical G workstation; only the assigned G may be entered; "
                        "agent disappears on arrival"
                    ),
                    "goal_assignment": "original physical G exit of each certified task",
                    "map_annotation_transform": map_entry["map_annotation_transform"],
                    "validation": validation,
                }
                atomic_json(certificate_path, certificate)
                target_entry["scenarios"][scenario_text] = {
                    "scenario_file": str(scenario_path.relative_to(ROOT)),
                    "scenario_sha256": sha256(scenario_path),
                    "certificate_file": str(certificate_path.relative_to(ROOT)),
                    "certificate_sha256": sha256(certificate_path),
                }
    manifest_path = output / "MANIFEST.json"
    atomic_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
